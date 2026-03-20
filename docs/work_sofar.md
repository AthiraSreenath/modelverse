# ModelVerse — Work So Far

> High-level summary of everything built. No details — just what exists and what it does.

---

## Tool Stack

### Frontend
- **Next.js 16 / React 19** — app framework
- **TypeScript 5** — language
- **Tailwind CSS 4** — styling
- **Zustand 5** — global state (current IR, chat history, upload state)
- **@xyflow/react (React Flow) 12** — interactive graph canvas
- **Vercel AI SDK (`ai`, `@ai-sdk/anthropic`)** — streaming chat UI
- **Radix UI** — accessible tooltip, scroll-area, separator primitives
- **Lucide React** — icon set
- **clsx / tailwind-merge / class-variance-authority** — class utilities

### Backend
- **FastAPI** — HTTP API + SSE streaming
- **Pydantic v2** — data models / IR schema validation
- **httpx** — async HTTP for HuggingFace Hub calls
- **huggingface-hub** — official HF library for config fetching
- **Anthropic SDK** — Claude (primary LLM)
- **OpenAI SDK** — GPT-4o (fallback LLM)
- **onnx** — ONNX protobuf parsing
- **python-multipart** — file upload handling
- **uv** — Python package manager / virtual environment

### Infrastructure
- **GitHub** — version control
- **.env** — local secrets (ANTHROPIC_API_KEY, OPENAI_API_KEY, HF_TOKEN)
- **Disk cache** (`~/.cache/modelverse/`) — persists LLM parse results

---

## Features

### Model Input
- **HuggingFace Hub lookup** — type any `org/model-name`, fetches `config.json` live
- **File upload** — drag or click to upload model files; streamed to temp file, never fully loaded into RAM
- **Supported formats** — `.safetensors`, `.gguf`, `.bin` / `.pt` / `.pth`, `.json` (HF config), `.onnx`
- **Pre-baked library** — instant resolution for a curated set of well-known models

### Architecture IR (the core data model)
- Unified `ArchitectureIR` schema shared across every input path and every output view
- Every model resolves to the same IR regardless of how it was loaded
- Blocks, children, repeat counts, param counts, notes — all in one Pydantic model
- Source tracking (`hf_config`, `file_header`, `prebaked`, `llm_knowledge`)
- Confidence levels (`EXACT`, `HIGH`, `MEDIUM`, `LOW`) per parsed result

### Three-Tier Architecture Parser
- **Tier 1 — Explicit family parsers** — 37 registered model types with exact formulas
  - BERT / RoBERTa / DistilBERT / ALBERT / DeBERTa family
  - GPT-2 / GPT-Neo / GPT-NeoX / GPT-J family
  - LLaMA / Mistral / Mixtral / Gemma / Phi / Qwen2 / Falcon / Cohere / OLMo / InternLM2
  - T5 / Flan-T5 / mT5 / LongT5
  - Mamba / Mamba2 / Falcon-Mamba
  - DeepSeek-V2 / DeepSeek-V3 (MoE + MLA attention)
  - DeepSeek-VL / DeepSeek-VL2 (vision encoder + projector + LM)
- **Tier 2 — Smart generic fallback** — auto-detects any standard HF config:
  - Decoder / encoder / encoder-decoder class detection
  - GQA, MoE, RMSNorm, RoPE, activation type all inferred from field names
  - Returns `HIGH` confidence when all key fields present, `MEDIUM`/`LOW` otherwise
- **Tier 3 — LLM-assisted enrichment** — fires only on `MEDIUM`/`LOW` confidence:
  - Single non-streaming call to Claude Haiku or GPT-4o-mini
  - Normalises non-standard field names into canonical HF schema
  - Result re-parsed by Tier 2 for a proper IR
  - Cached to disk — same config never calls the LLM twice

### Compute Estimator
- Parameter count: recursive over all blocks × repeat
- Memory: fp32 / fp16 / bf16 / int8 / int4
- FLOPs per token: Q/K/V projections + attention scores + value aggregation + O projection + FFN
- KV cache size: at configurable reference sequence length, GQA-aware, handles encoder-decoder cross-attention
- Task-switching: updates head params and memory without re-running full estimation

### Math correctness fixes
- T5 gated FFN: 3 matrices for gated activations (SwiGLU/GeGLU), 2 for non-gated
- T5 layer norms rendered as `RMSNorm` (weight-only, no bias)
- Attention FLOPs: added missing `softmax(QK^T)@V` term
- `switchTask` in store: preserves backbone `flops_per_token` correctly

### File Parser (header-only, no weights loaded)
- `.safetensors` — reads JSON header (8-byte length prefix + JSON)
- `.gguf` — pure Python binary reader for GGUF key-value metadata
- `.bin` / `.pt` / `.pth` — attempts `torch.load(map_location="meta")`
- `.onnx` — saves to temp file, calls `onnx.load(path)` (avoids truncation); falls back to hand-rolled protobuf scanner
- Shape-based architecture inference from tensor naming conventions

### Graph Visualisation
- Interactive node-edge canvas (React Flow)
- Transformer stacks rendered as compound nodes with repeat badge
- Residual/skip edges with custom routing
- RepeatLabel nodes for layer count display
- Pan, zoom, fit-to-view
- Click any node → detail panel

### Detail Panel
- Per-layer explanation: what the layer does, mathematical formula, PyTorch pseudocode
- Parameter breakdown per block
- Notes / design rationale surfaced inline

### Compute Bar (top strip)
- Live parameter count, memory (fp16 + int4), FLOPs, latency estimate
- Hardware selector (A100 40GB / 80GB, H100, etc.)
- Task switcher (updates head + recomputes)

### Chat Panel (LLM Brain)
- Streaming SSE chat, provider auto-detected (Anthropic first, OpenAI fallback)
- Full Architecture IR injected into system prompt
- Tool calling loop (up to 5 rounds):
  - `search_huggingface` — fetch any model for comparison
  - `search_web` — ArXiv / web lookup for architecture details
  - `estimate_compute` — runs compute estimator on current IR
  - `apply_edit` — modifies the live IR (set_repeat, set_param, add_block, remove_block, replace_block)
  - `explain_layer` — returns formula + pseudocode for any layer type
- Edit results reflected immediately in the graph

### Search Bar UI
- Single joined input bar: `HF Hub` label | search input | `Upload` button
- Upload button shows filename + spinner during parse
- Supported extensions listed dynamically from `/formats` endpoint
- Error banner below bar on parse failure

### API Endpoints
- `GET  /resolve?q=...` — main model lookup (all three tiers)
- `POST /upload` — file upload → IR (streaming to temp, format-dispatched)
- `GET  /formats` — list supported file extensions
- `POST /chat` — SSE streaming chat with tool calls
- `POST /edit` — apply architectural edit, return diff + compute delta

### Logo & Branding
- SVG mark: bold white "MV" on rounded indigo (#4f46e5) tile, 32×32 viewBox
- Used as favicon, app icon, and README logo

---

## Architecture Families with Correct Math

| Family | Notes |
|---|---|
| BERT / RoBERTa | Post-LN, absolute pos, type embeddings |
| GPT-2 | Pre-LN, absolute pos, weight-tied LM head |
| LLaMA / Mistral | Pre-LN, RoPE, RMSNorm, SwiGLU, GQA |
| Mixtral | Same as Mistral + MoE (all layers) |
| T5 / Flan-T5 | Enc-dec, relative pos, RMSNorm, gated/non-gated FFN |
| Mamba | SSM blocks, no attention, no KV cache |
| DeepSeek-V2/V3 | MLA attention, mixed dense+MoE layers |
| DeepSeek-VL2 | CLIP-L/14 + SAM-ViT-B + projector + DeepSeekV2 LM |

---

## Repo Structure (key paths)

```
backend/
  app/
    main.py                  # FastAPI app, endpoints
    models/ir.py             # ArchitectureIR Pydantic schema
    resolvers/
      router.py              # 3-tier resolve pipeline
      arch_parser.py         # Tier 1 + Tier 2 parsers
      llm_parser.py          # Tier 3 LLM enrichment + disk cache
      hf_fetcher.py          # HuggingFace Hub async fetch
      prebaked.py            # Curated model library
      file_parser/           # Format-specific header readers
    compute/estimator.py     # Param count + FLOPs + memory + KV cache
    llm/
      brain.py               # Streaming chat, tool dispatch
      tools.py               # Tool implementations
    edit/engine.py           # Architecture edit operations

frontend/src/
  app/page.tsx               # Root page
  components/
    SearchBar.tsx            # HF Hub input + file upload
    TaskSwitcher.tsx         # Task head selector
    graph/
      ArchGraph.tsx          # React Flow canvas
      ArchGraphClient.tsx    # Client wrapper
      GraphLayout.ts         # Dagre layout logic
      nodes/BlockNode.tsx    # Layer node renderer
      nodes/RepeatLabelNode.tsx
      edges/ResidualEdge.tsx
    panels/
      ComputeBar.tsx         # Stats strip
      DetailPanel.tsx        # Layer inspector
      ChatPanel.tsx          # LLM chat UI
    layout/ResizablePanel.tsx
  lib/
    store.ts                 # Zustand global state
    api.ts                   # fetch wrappers for all endpoints
```
