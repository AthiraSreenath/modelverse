# Contributing to ModelVerse

Thanks for your interest in contributing. This guide covers everything you need to run ModelVerse locally, understand how it works, and make changes.

---

## Table of Contents

1. [Architecture overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Local setup](#3-local-setup)
4. [Project structure](#4-project-structure)
5. [Environment variables](#5-environment-variables)
6. [How to add a pre-baked model](#6-how-to-add-a-pre-baked-model)
7. [How to add a new model family parser](#7-how-to-add-a-new-model-family-parser)
8. [How to add a file format parser](#8-how-to-add-a-file-format-parser)
9. [Running the backend tests](#9-running-the-backend-tests)
10. [Deployment](#10-deployment)
11. [Pull request guide](#11-pull-request-guide)

---

## 1. Architecture overview

Every input (model name, uploaded file, natural language) resolves into a single **Architecture IR** - a JSON schema that represents the model's structure. Every output (the graph, compute stats, LLM context, edit diffs) is generated from that IR.

```
User input
    │
    ├─ HF model name ──→ hf_hub_download(config.json) ──→ arch_parser.py ──→ IR
    ├─ File upload   ──→ file_parser/ (header only)    ──→ arch_parser.py ──→ IR
    └─ Any NL query  ──→ LLM Brain (tool calls)        ──→ web search     ──→ IR
                                                                │
                                          ┌─────────────────────┼─────────────────┐
                                          ▼                     ▼                 ▼
                                     React Flow graph     Compute bar        Chat panel
```

The backend is a **FastAPI** Python app. The frontend is **Next.js 15**. They communicate via REST (`/resolve`, `/compute`, `/edit`) and Server-Sent Events (`/chat` for streaming).

For the full design rationale, IR schema, and API contracts: [docs/design_doc.md](docs/design_doc.md).

---

## 2. Prerequisites

- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** - Python package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **[Node.js 20+](https://nodejs.org)** - for the Next.js frontend
- **[git](https://git-scm.com)**
- An **Anthropic API key** (or OpenAI) for the LLM chat feature

---

## 3. Local setup

```bash
git clone https://github.com/athira/modelverse
cd modelverse
```

### Backend

```bash
cd backend
uv sync                     # creates .venv and installs all dependencies
cp .env.example .env        # fill in your API keys (see Section 5)
uv run fastapi dev app/main.py   # → http://localhost:8000
```

The API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs) once the server is running.

### Frontend

Open a new terminal:

```bash
cd frontend
npm install
cp .env.example .env.local      # set NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
npm run dev                     # → http://localhost:3000
```

Open [http://localhost:3000](http://localhost:3000). Type `bert-base-uncased` to verify the end-to-end flow works.

---

## 4. Project structure

```
modelverse/
├── frontend/    # Next.js 15 app → deploys to Vercel
├── backend/     # FastAPI Python app → deploys to Railway
├── data/
│   └── prebaked/  # Pre-baked Architecture IRs as static JSON
└── docs/
    └── design_doc.md  # Full design, IR schema, API contracts
```

Key files to know:

- `backend/app/models/ir.py` - the canonical Architecture IR schema (Pydantic). Everything flows through this.
- `backend/app/resolvers/arch_parser.py` - HF config → IR. Add new model families here.
- `backend/app/llm/brain.py` - LLM streaming chat + tool loop.
- `frontend/src/lib/ir.ts` - TypeScript mirror of the IR schema.
- `frontend/src/lib/store.ts` - Zustand store (IR, chat history, undo stack).
- `frontend/src/components/graph/` - React Flow graph, node types, layout.

**Logo (keep in sync):** The app header loads `frontend/public/modelverse-mark.svg`. The favicon is `frontend/src/app/icon.svg`. The GitHub README image is `assets/modelverse-logo.svg`. All three should be identical SVG (graph-style MV on the indigo gradient). Edit one, then copy to the other two paths.

---

## 5. Environment variables

### Backend (`backend/.env`)


| Variable             | Required | Description                                                           |
| -------------------- | -------- | --------------------------------------------------------------------- |
| `ANTHROPIC_API_KEY`  | Yes*     | Powers the LLM chat (Claude Sonnet)                                   |
| `OPENAI_API_KEY`     | Yes*     | Alternative LLM (GPT-4o)                                              |
| `HF_TOKEN`           | No       | HuggingFace token - raises API rate limits from ~100/day to ~1000/day |
| `EXTRA_CORS_ORIGINS` | No       | Comma-separated list of additional allowed CORS origins               |


 At least one LLM key is required for the chat feature. The backend works without it for `/resolve`, `/compute`, and `/edit`.

### Frontend (`frontend/.env.local`)


| Variable                  | Required | Description                                                      |
| ------------------------- | -------- | ---------------------------------------------------------------- |
| `NEXT_PUBLIC_BACKEND_URL` | Yes      | URL of the running FastAPI backend, e.g. `http://localhost:8000` |


---

## 6. How to add a pre-baked model

Pre-baked IRs live in `data/prebaked/` and load in <100ms with no API call. Good candidates: popular models you expect users to type frequently.

**Step 1.** Create `data/prebaked/<model-id>.json`.

Follow the Architecture IR schema in [docs/design_doc.md § 4](docs/design_doc.md#4-architecture-ir). Use `bert-base-uncased.json` as a reference. Key fields:

```json
{
  "schema_version": "1.0",
  "name": "owner/model-name",
  "display_name": "Human Readable Name",
  "family": "bert",
  "task": "fill-mask",
  "source": "prebaked",
  "source_confidence": "exact",
  "blocks": [ ... ],
  "compute": { ... }
}
```

**Step 2.** Register it in `backend/app/resolvers/prebaked.py`:

```python
_REGISTRY: dict[str, str] = {
    # existing entries ...
    "owner/model-name": "owner--model-name.json",
    "model-name": "owner--model-name.json",  # add short alias too
}
```

**Step 3.** Verify:

```bash
cd backend
uv run python -c "
from app.resolvers.prebaked import get_prebaked
ir = get_prebaked('owner/model-name')
print(ir.compute.params_total)
"
```

---

## 7. How to add a new model family parser

If HuggingFace returns a `model_type` that isn't currently handled (check `_FAMILY_PARSERS` in `backend/app/resolvers/arch_parser.py`), the system falls back to a generic parser that extracts `hidden_size`, `num_layers`, and `num_heads` but misses family-specific details.

**Step 1.** Write a parser function in `arch_parser.py`:

```python
def _parse_myfamily(config: dict, model_id: str) -> ArchitectureIR:
    h = config.get("hidden_size", 768)
    num_layers = config.get("num_hidden_layers", 12)
    # ... build blocks, call estimate_compute(ir)
    return ir
```

The function receives the raw `config.json` dict and the model ID string. It returns a fully populated `ArchitectureIR` with `compute` set. Look at `_parse_bert_family` or `_parse_llama_family` for reference.

**Step 2.** Register it in `_FAMILY_PARSERS`:

```python
_FAMILY_PARSERS = {
    # existing entries ...
    "myfamily": _parse_myfamily,
    "myfamily-variant": _parse_myfamily,
}
```

**Step 3.** Test it with a real model config:

```bash
cd backend
uv run python -c "
from app.resolvers.arch_parser import parse_hf_config
import json, pathlib
# use a real config.json you've downloaded, or construct a minimal one
config = {'model_type': 'myfamily', 'hidden_size': 768, ...}
ir = parse_hf_config(config, 'owner/my-model')
print([b.label for b in ir.blocks])
print(f'params: {ir.compute.params_total:,}')
"
```

---

## 8. How to add a file format parser

File parsers live in `backend/app/resolvers/file_parser/`. Each format is a class implementing the `ModelParser` interface (Phase 2 feature).

```python
# backend/app/resolvers/file_parser/myformat.py

class MyFormatParser:
    @staticmethod
    def match(data: bytes) -> bool:
        """Return True if the first bytes indicate this format."""
        return data[:4] == b'\x89PNG'  # example

    @staticmethod
    async def open(stream) -> ArchitectureIR:
        """Parse the stream and return an Architecture IR.
        Read header/metadata only - never load weights."""
        ...
```

Register it in `backend/app/resolvers/file_parser/base.py`:

```python
PARSERS = [
    SafetensorsParser,
    OnnxParser,
    GGUFParser,
    PyTorchParser,
    MyFormatParser,   # add here
]
```

---

## 9. Running the backend tests

```bash
cd backend
uv run pytest
```

Tests live alongside the code in `tests/`. Key test files:

```
tests/
├── test_prebaked.py       # all pre-baked IRs load and validate
├── test_arch_parser.py    # each family parser produces correct IR
├── test_estimator.py      # compute formulas match known values
└── test_edit_engine.py    # edit ops apply correctly + delta is accurate
```

---

## 10. Deployment

### Frontend → Vercel

```bash
cd frontend
npx vercel --prod
```

Set `NEXT_PUBLIC_BACKEND_URL` to your Railway backend URL in the Vercel dashboard environment variables.

### Backend → Railway

1. Connect your GitHub repo at [railway.app](https://railway.app)
2. Set **Root Directory** to `/backend`
3. Railway auto-detects Python via Nixpacks - no Dockerfile needed
4. Add environment variables: `ANTHROPIC_API_KEY/`, `HF_TOKEN`
5. Deploy triggers automatically on push to `main`

The start command Railway uses: `fastapi run app/main.py --host 0.0.0.0 --port $PORT`

---

## 11. Pull request guide

- **Open an issue first** for anything non-trivial so we can discuss direction before you write code
- Keep PRs focused - one feature or fix per PR
- All pre-baked IRs must have accurate `compute` stats (verify with `test_prebaked.py`)
- Family parsers must pass `test_arch_parser.py` before merging
- Follow the existing code style - the backend uses standard Python type hints throughout

**Good first contributions:**


| Type              | Example                                                                              |
| ----------------- | ------------------------------------------------------------------------------------ |
| Pre-baked IR      | Add `google/gemma-2-2b`, `microsoft/phi-4`                                           |
| Family parser     | Add `bart`, `bloom`, `opt`, `xlnet` parsers                                          |
| Layer explanation | Add pseudocode for a layer type in `backend/app/llm/tools.py` → `tool_explain_layer` |
| File format       | Add a GGUF or Keras file parser (Phase 2)                                            |
| Frontend          | Improve graph node styling, add keyboard shortcuts                                   |


---

Questions? Open an issue or start a discussion.