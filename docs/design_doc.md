# ModelVerse — Design Document

> **"Figma for neural network architectures."**
> Inspect, understand, edit, and reason about any ML model — by name, by file, or by description.

---

## Table of Contents

1. [Product Vision](#1-product-vision)
2. [Prior Art — What Exists and What We Take](#2-prior-art)
3. [The Core Mental Model](#3-the-core-mental-model)
4. [Architecture IR — The Universal Schema](#4-architecture-ir)
5. [Input Resolution Pipeline](#5-input-resolution-pipeline)
6. [System Components](#6-system-components)
7. [UI Layout](#7-ui-layout)
8. [Tech Stack](#8-tech-stack)
9. [Project Structure](#9-project-structure)
10. [API Contracts](#10-api-contracts)
11. [LLM Brain — Tool Definitions](#11-llm-brain)
12. [Compute Estimator — Formulas](#12-compute-estimator)
13. [Phased Roadmap](#13-phased-roadmap)
14. [Key Design Decisions](#14-key-design-decisions)
15. [Infrastructure & Hosting](#15-infrastructure--hosting)

---

## 1. Product Vision

ModelVerse is an open-source **Architecture Reasoning Engine** for machine learning models.

A user types a model name, uploads a file, or describes a model in plain language. ModelVerse shows them the full architecture as an interactive graph, lets them click any layer to understand it, chat with an LLM about it, edit it and see the parameter and compute delta instantly, and compare it against another model — all without writing a single line of code.

### Who this is for

- **ML engineers** designing or evaluating model architectures
- **Researchers** reading a new paper and wanting to explore its architecture
- **Students** learning how transformers, RNNs, or classical ML models work
- **Anyone** who wants to answer "what is this model, and what happens if I change it?"

### The daily-use scenarios

| Scenario | What they do |
|---|---|
| Reading a new paper | Paste `mistralai/Mixtral-8x7B-v0.1` → explore MoE routing |
| Designing an experiment | "What if I reduce hidden size to 512?" → see param/memory delta |
| Estimating compute budget | Load LLaMA-3-70B → see FLOPs, memory at fp16 |
| Teaching a concept | "Show me how multi-head attention works" → click the node |
| Evaluating a fine-tuned model | Upload `model.safetensors` → inspect every layer |

---

## 2. Prior Art

We studied the source code of the two best existing tools before designing anything.

### Netron — [github.com/lutzroeder/netron](https://github.com/lutzroeder/netron) (32k stars)

The gold standard for model file visualization. Studied: `view.js` (6,100 lines), `grapher.js` (1,100 lines), `safetensors.js`, `pytorch.js`, `onnx.js`.

**What makes it great — and what we copy directly:**

- **`ModelFactory` pattern**: every format has `match()` (sniffs the file) + `open()` (parses it). Clean, extensible to any format. We implement this exact pattern in Python on the backend.
- **Safetensors header-only parsing**: reads exactly 8 bytes (a little-endian size prefix) then N bytes of JSON to get every tensor name, shape, and dtype. Never touches the weights. We copy this.
- **SVG + Dagre graph rendering**: `grapher.js` builds a compound SVG graph using Dagre for rank-based layout. Compound nodes (clusters) group transformer blocks.
- **Web Worker for layout**: large graph Dagre layout runs in a Worker to avoid blocking the UI thread. We copy this.
- **Sidebar event system**: graph nodes emit `focus / blur / select / activate` events; the sidebar panel reacts. Fully decoupled from the graph renderer.
- **Drill-down property chain**: Node Properties → Connection Properties → Tensor Properties → Documentation. Each pushes onto a navigation stack.

**What it's missing:**
- No model name lookup — you must have the file
- No LLM, no natural language Q&A
- No architecture editing
- No compute estimation
- No parameter formula explanations
- No comparison view
- No pre-baked knowledge of architectures

### Model Anatomy — [modelanalysis.ai](https://model-anatomy-ram-komarraju.netlify.app/)

A pre-baked interactive LLM architecture explorer (not open source). Built in one evening using a coding agent.

**What makes it great — and what we copy:**

- **Parameter count with formula hover**: hover any number → see `3 × 768 × 768 = 1,769,472` (Q+K+V projections). Every single number is explainable. This is the killer UX detail and we make it a core principle.
- **PyTorch pseudocode panel**: click any module → see its implementation. We include this.
- **Side-by-side model comparison**: two models in parallel columns, differences visually apparent.
- **Pre-baked architecture library**: architectures are hard-coded for popular models — loads in milliseconds. We ship a JSON library of ~50 models for the same reason.
- **Family filter**: group models by Llama, Gemma, Mistral etc.

**What it's missing:**
- Fixed set of ~20 models — can't load any HF model or your own file
- No LLM chat
- No architecture editing or compute delta
- No file upload
- Not open source

### The Gap ModelVerse Fills

No existing tool combines:
- File upload (Netron's strength)
- Name-based HF lookup with zero weight download
- Agentic web search for any publicly known model
- LLM-powered natural language chat
- Live architecture editing with instant compute delta
- Formula-level parameter explanations

That combination is ModelVerse.

---

## 3. The Core Mental Model

**ModelVerse is not a visualization tool. Visualization is just the UI.**

The real product is an **Architecture Reasoning Engine** that answers three questions:

1. *What is this model?*
2. *What happens if I change it?*
3. *How expensive will it be?*

Everything — the graph, the chat panel, the compute bar, the diff view — exists to answer those three questions.

---

## 4. Architecture IR

The **Architecture IR (Intermediate Representation)** is the single JSON schema at the center of the entire system. Every input path (HF model name, uploaded file, web-searched model, NL description) resolves into this IR. Every output (visualization, compute estimate, diff, exported config, LLM context) is generated from it.

This means the visualization engine, compute estimator, edit engine, and LLM Brain are all completely decoupled from the input source. Parse once, use everywhere.

### Schema

```json
{
  "schema_version": "1.0",
  "name": "dslim/bert-base-NER",
  "display_name": "BERT Base NER",
  "family": "bert",
  "task": "token-classification",
  "source": "hf_config",
  "source_confidence": "exact",
  "source_refs": [],

  "blocks": [
    {
      "id": "embeddings",
      "label": "Embeddings",
      "type": "embedding",
      "params": {
        "vocab_size": 28996,
        "hidden_size": 768,
        "max_position_embeddings": 512,
        "type_vocab_size": 2
      }
    },
    {
      "id": "encoder",
      "label": "Transformer Encoder",
      "type": "transformer_stack",
      "repeat": 12,
      "children": [
        {
          "id": "self_attn",
          "label": "Self-Attention",
          "type": "multi_head_attention",
          "params": {
            "num_heads": 12,
            "hidden_size": 768,
            "head_dim": 64,
            "attention_type": "bidirectional"
          }
        },
        {
          "id": "attn_norm",
          "label": "LayerNorm",
          "type": "layer_norm",
          "params": { "normalized_shape": 768 }
        },
        {
          "id": "ffn",
          "label": "Feed-Forward",
          "type": "feed_forward",
          "params": {
            "hidden_size": 768,
            "intermediate_size": 3072,
            "activation": "gelu"
          }
        },
        {
          "id": "ffn_norm",
          "label": "LayerNorm",
          "type": "layer_norm",
          "params": { "normalized_shape": 768 }
        }
      ]
    },
    {
      "id": "pooler",
      "label": "Pooler",
      "type": "linear",
      "params": { "in_features": 768, "out_features": 768 }
    },
    {
      "id": "classifier",
      "label": "Token Classifier",
      "type": "linear",
      "params": { "in_features": 768, "out_features": 9 }
    }
  ],

  "total_params": 108890369,
  "compute": {
    "params_embedding": 22282752,
    "params_encoder": 85054464,
    "params_head": 6922,
    "params_total": 108890369,
    "flops_per_token": 22500000000,
    "memory_fp32_gb": 0.416,
    "memory_fp16_gb": 0.208,
    "memory_int8_gb": 0.104
  }
}
```

### `source` field values

| Value | Meaning |
|---|---|
| `hf_config` | Parsed directly from HuggingFace `config.json` — exact |
| `file_header` | Parsed from uploaded file header (safetensors/GGUF/ONNX) — exact |
| `paper` | Extracted from ArXiv or published paper — high confidence |
| `web_synthesis` | Synthesized from multiple web sources — medium confidence |
| `llm_knowledge` | From LLM training data alone — lower confidence, may be stale |

Unknown fields are represented as `null` and rendered as greyed-out nodes in the UI.

### Block types supported

| Type | Covers |
|---|---|
| `embedding` | Token + position + type embeddings |
| `transformer_stack` | Repeated transformer blocks (with `repeat` field) |
| `multi_head_attention` | MHA, GQA (grouped-query), MQA (multi-query) |
| `feed_forward` | Standard FFN, SwiGLU, GeGLU |
| `moe_feed_forward` | Mixture-of-Experts FFN with router |
| `layer_norm` | LayerNorm, RMSNorm |
| `linear` | Dense projection, classifier head |
| `conv1d` | 1D convolution (Mamba SSM blocks) |
| `ssm` | State Space Model layer (Mamba) |
| `rnn` | LSTM / GRU cell |
| `pooling` | Global average/max pool, CLS token extraction |
| `dropout` | Dropout layer |
| `activation` | Standalone activation (ReLU, GELU, SiLU…) |

---

## 5. Input Resolution Pipeline

All four phases resolve to the same Architecture IR.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User Input                                  │
└────────┬─────────────────┬───────────────────────┬──────────────────┘
         │ Phase 1         │ Phase 2               │ Phase 3
         ▼                 ▼                       ▼
   HF model ID        File upload           Any model name
   (e.g. bert-base)   (.safetensors etc)    (e.g. "GPT-3")
         │                 │                       │
         ▼                 ▼                       ▼
   Pre-baked         ModelFactory            LLM Brain
   library?          match() + open()        (Agentic)
    yes → instant    header-only parse         │
    no → HF fetch                          search_huggingface
         │                 │               → found? → HF path
         ▼                 │               → not found?
   hf_hub_download         │               search_web (ArXiv,
   config.json only        │               HF blog, GitHub)
         │                 │               synthesize + cite
         ▼                 ▼               + tag confidence
   Architecture      Architecture               │
   Parser            Parser                     │
         │                 │                    │
         └─────────────────┴────────────────────┘
                           │
                           ▼
                   Architecture IR (JSON)
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    Visualization    Compute Bar      LLM Brain
      (React Flow)   (estimator.py)   (chat panel)
```

**Phase 1 and 2 are deterministic.** Config.json or file header gives exact architecture data.

**Phase 3 is agentic.** The LLM Brain fires parallel tool calls (ArXiv, HF blog, GitHub, official docs), synthesizes from evidence, cites sources, and marks uncertain fields. Well-known models (GPT-3, PaLM, original Transformer) can be resolved from LLM training data alone; web search is a fallback for accuracy and recency.

---

## 6. System Components

### 6.1 Input Resolver

Detects input type and routes:
- Pattern `owner/model-name` or `model-name` → HF Fetcher
- File extension `.safetensors`, `.onnx`, `.pt`, `.bin`, `.gguf`, `.keras`, `.h5` → File Parser
- Everything else → LLM Brain (NL query)

### 6.2 HF Fetcher

```python
from huggingface_hub import hf_hub_download

config_path = hf_hub_download(repo_id=model_id, filename="config.json")
```

Fetches only `config.json` — never model weights. Also fetches the model card (README.md) for task and description context.

Extracts from config: `model_type`, `num_hidden_layers`, `num_attention_heads`, `hidden_size`, `intermediate_size`, `vocab_size`, `max_position_embeddings`, `architectures`, activation function, normalization type, position embedding type (absolute, RoPE, ALiBi), attention variant (MHA/GQA/MQA), and task-specific head config.

### 6.3 Pre-baked Architecture Library

A static JSON file at `data/prebaked/` with ~50 well-known model IRs. These load in <100ms with zero API calls. Checked first on every HF model request.

Initial library:
- **BERT family**: `bert-base-uncased`, `bert-large-uncased`, `distilbert-base-uncased`, `roberta-base`, `roberta-large`, `dslim/bert-base-NER`
- **GPT family**: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- **LLaMA family**: `meta-llama/Llama-3.1-8B`, `meta-llama/Llama-3.1-70B`
- **Mistral family**: `mistralai/Mistral-7B-v0.1`, `mistralai/Mixtral-8x7B-v0.1`
- **T5 family**: `t5-small`, `t5-base`, `t5-large`, `google/flan-t5-xl`
- **Other**: `google/gemma-2-9b`, `microsoft/phi-3-mini-4k-instruct`, `tiiuae/falcon-7b`, `state-spaces/mamba-1.4b-hf`

### 6.4 File Parser

Modeled on Netron's `ModelFactory` pattern. Each format is its own class:

```python
class SafetensorsParser:
    @staticmethod
    def match(data: bytes) -> bool:
        # Check 8-byte header prefix
        size = struct.unpack_from("<Q", data[:8])[0]
        return size < len(data) and data[8] == ord('{')

    @staticmethod
    def open(stream) -> ArchitectureIR:
        size = struct.unpack_from("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
        # header contains: {tensor_name: {dtype, shape, data_offsets}}
        # Never read data_offsets — weights not needed
        return parse_state_dict_keys(header)
```

**Per-format strategies:**

| Format | What we read | How |
|---|---|---|
| `.safetensors` | 8-byte prefix + JSON header | Layer names, shapes, dtypes — zero weight load |
| `.gguf` | Metadata block in header | `architecture`, `n_layer`, `n_head`, `n_embd` etc. |
| `.onnx` | `graph.node` list | `op_type`, `input`, `output`, `attribute` for each node |
| `.pt` / `.bin` | `state_dict` keys only | Infer architecture from key naming patterns |
| `.keras` / `.h5` | `model.get_config()` | Full layer tree as JSON |

**PyTorch key pattern inference** (example):

```
encoder.layer.0.attention.self.query.weight  → BERT, encoder block 0, MHA, Q proj
model.layers.0.self_attn.q_proj.weight       → LLaMA, decoder block 0, MHA, Q proj
transformer.h.0.attn.c_attn.weight           → GPT-2, transformer block 0, combined QKV
```

### 6.5 Architecture Parser

Converts raw HF config dict or inferred state_dict keys into the Architecture IR. Family-aware:

- **BERT/RoBERTa**: `BertConfig` → bidirectional encoder + token/sequence classifier head
- **GPT-2**: `GPT2Config` → causal decoder + LM head
- **LLaMA/Mistral**: `LlamaConfig` / `MistralConfig` → causal decoder with RoPE + GQA + RMSNorm + SwiGLU
- **T5**: `T5Config` → encoder-decoder with relative position bias
- **Mamba**: `MambaConfig` → SSM blocks with selective scan
- **Mixtral**: `MixtralConfig` → LLaMA-style with MoE FFN

### 6.6 Compute Estimator

Pure arithmetic — no model loading. Runs in <1ms.

See [Section 12](#12-compute-estimator) for full formulas.

### 6.7 Architecture Edit Engine

Accepts an `EditSpec` and mutates the IR. Returns the modified IR + a diff + compute delta.

```python
def apply_edit(ir: ArchitectureIR, spec: EditSpec) -> EditResult:
    new_ir = deep_copy(ir)
    match spec.op:
        case "set_repeat":     # change num transformer blocks
        case "set_param":      # change a specific parameter
        case "add_block":      # insert a new block
        case "remove_block":   # remove a block
        case "replace_block":  # swap one block type for another
    delta = compute_delta(ir, new_ir)
    diff = generate_diff(ir, new_ir)
    return EditResult(new_ir=new_ir, diff=diff, delta=delta)
```

### 6.8 LLM Brain

The reasoning engine. Receives the current Architecture IR as structured JSON in its system prompt — never raw weights. Uses tool calling for any action requiring data outside its context.

See [Section 11](#11-llm-brain) for tool definitions.

### 6.9 Visualization Engine (Frontend)

Converts Architecture IR → interactive React Flow graph.

**Graph renderer:**
- React Flow (SVG-based, Dagre layout, compound/nested nodes)
- Layout runs in a Web Worker (copied from Netron) — prevents UI freeze for 80-layer models
- Compound nodes: a `transformer_stack` renders as a single block with a `×12` badge. Click → expands to show all children (Self-Attention, FFN, LayerNorm)
- Edges carry tensor shapes: `[batch, seq_len, 768]` shown as edge labels, hover for full shape

**Detail panel (middle):**
- Click any node → parameter table with formula hover (Model Anatomy style)
- `768 × 768 × 3` hover shows `Q projection + K projection + V projection`
- PyTorch pseudocode for the selected layer type
- "Ask LLM" button → fires a chat message with the node's IR as context
- Input/output tensor shapes for the selected node

**Diff view:**
- Edited nodes: highlighted border + `Δ+2.1M params` badge
- Before/after toggle at the top

**Compare view (Phase 4):**
- Two graphs side-by-side
- Matching layer types aligned vertically
- Summary bar: params / memory / FLOPs for each model

---

## 7. UI Layout

Three-panel layout, inspired by Netron's structure and Model Anatomy's content:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ModelVerse    [  dslim/bert-base-NER  ▸ ] [Upload]   [Compare] [⚙]   │
├──────────────────────────────┬──────────────────────┬───────────────────┤
│                              │                      │                   │
│   Interactive Graph          │   Detail Panel       │   Chat Panel      │
│   (React Flow)               │                      │   (LLM Brain)     │
│                              │   Self-Attention     │                   │
│   ┌──────────────────┐       │   ──────────────     │  you: why 12      │
│   │   Embeddings     │       │   Params: 2.36M      │  attention heads? │
│   └────────┬─────────┘       │                      │                   │
│            │[768]            │   Formula:           │  llm: Multi-head  │
│   ┌────────▼─────────┐       │   Q: 768×768 = 589K  │  attention splits │
│   │ Transformer ×12  │──────▶│   K: 768×768 = 589K  │  the hidden rep   │
│   │  ├ Self-Attn     │       │   V: 768×768 = 589K  │  into 12 subspaces│
│   │  ├ FFN           │       │   O: 768×768 = 589K  │  allowing richer  │
│   │  ├ LayerNorm     │       │   Total: 2.36M        │  attention...     │
│   │  └ LayerNorm     │       │                      │                   │
│   └────────┬─────────┘       │   Pseudocode ▼       │  you: what if I   │
│            │                 │                      │  halve hidden size?│
│   ┌────────▼─────────┐       │   Shapes:            │                   │
│   │   Classifier     │       │   in:  [B, T, 768]   │  llm: Parameters  │
│   └──────────────────┘       │   out: [B, T, 768]   │  drop from 108M   │
│                              │                      │  to 29M (-73%)... │
│                              │   [Ask LLM ↗]        │                   │
├──────────────────────────────┴──────────────────────┴───────────────────┤
│   Params: 108M    Memory fp16: 208MB    FLOPs/tok: 22.5B   [Edit ✏]    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Left panel** — Interactive graph. Collapsible compound nodes, tensor shapes on edges, drag/zoom/pan.

**Middle panel** — Detail panel. Activated by clicking a node. Shows: parameter table with formula hover, PyTorch pseudocode, input/output shapes, "Ask LLM" button. Inspired by Model Anatomy.

**Right panel** — Chat panel. Streaming LLM responses. Tool call results render inline (e.g. a compute delta table appears in the chat stream when `estimate_compute` fires). The LLM has the full Architecture IR in its system context at all times.

**Bottom bar** — Always visible. Total parameters, memory at fp32/fp16/int8, FLOPs per token. "Edit" button opens a text input for NL architecture edits.

---

## 8. Tech Stack

### Frontend

| Technology | Role | Why |
|---|---|---|
| **Next.js 15 (App Router)** | Web framework | Canonical React framework, Vercel native, API routes if needed |
| **React Flow** | Interactive graph | SVG + Dagre + compound nodes within React. Equivalent to Netron's `grapher.js` but React-native |
| **Tailwind CSS** | Styling | Utility-first, consistent with shadcn/ui |
| **shadcn/ui** | UI components | High quality, accessible, unstyled by default — easy to theme dark |
| **Zustand** | Client state | Current IR, chat history, edit history, undo/redo stack |
| **Vercel AI SDK** | LLM streaming | Tool call rendering, streaming text, works out of the box with Vercel |
| **Web Worker (native)** | Graph layout | Dagre layout for large models runs off the main thread |

### Backend

| Technology | Role | Why |
|---|---|---|
| **Python 3.12+** | Language | Best ML ecosystem, huggingface_hub, safetensors all Python-native |
| **FastAPI** | API framework | Async, fast, great OpenAPI docs, streaming support |
| **uv** | Package management | Replaces pip + venv. `uv sync` = one command to reproduce environment |
| **`huggingface_hub`** | HF API | `hf_hub_download` fetches config.json without weights |
| **`transformers`** | Config introspection | `AutoConfig.from_pretrained()` for deeper config parsing |
| **`safetensors`** | File parsing | Header-only tensor metadata, zero weight load |
| **`onnx`** | File parsing | Graph node/edge inspection |
| **`torch`** | File parsing | state_dict key loading (lazy import, not always needed) |
| **Anthropic / OpenAI SDK** | LLM Brain | Tool calling + streaming |
| **`httpx`** | Web search (Phase 3) | Async HTTP for ArXiv API, web scraping |

### LLM

- **Claude Sonnet** (primary) or **GPT-4o** (fallback) — both have strong tool calling and code reasoning
- The LLM receives the Architecture IR as structured JSON in its system prompt — never raw weights
- Context window usage: IR is ~2–5KB for most models; even a 200K context window leaves room for extensive chat history

---

## 9. Project Structure

```
modelverse/
│
├── frontend/                         # Next.js app → Vercel
│   ├── package.json
│   ├── next.config.ts
│   ├── tailwind.config.ts
│   ├── .env.local                    # NEXT_PUBLIC_BACKEND_URL
│   ├── .env.example
│   └── src/
│       ├── app/
│       │   ├── layout.tsx
│       │   ├── page.tsx              # Main app page
│       │   └── api/
│       │       └── chat/
│       │           └── route.ts      # Vercel AI SDK streaming endpoint
│       ├── components/
│       │   ├── graph/
│       │   │   ├── ModelGraph.tsx    # Root React Flow component
│       │   │   ├── nodes/
│       │   │   │   ├── BlockNode.tsx         # Collapsed compound node
│       │   │   │   ├── ExpandedStackNode.tsx # Expanded transformer stack
│       │   │   │   └── LeafNode.tsx          # Leaf layer node (attn, ffn…)
│       │   │   ├── edges/
│       │   │   │   └── TensorEdge.tsx        # Edge with shape label
│       │   │   └── GraphToolbar.tsx
│       │   ├── panels/
│       │   │   ├── DetailPanel.tsx   # Middle panel: params, formula, pseudocode
│       │   │   ├── ChatPanel.tsx     # Right panel: LLM chat
│       │   │   └── ComputeBar.tsx    # Bottom bar: params/memory/FLOPs
│       │   └── ui/                   # shadcn/ui components
│       ├── lib/
│       │   ├── ir.ts                 # Architecture IR TypeScript types
│       │   ├── store.ts              # Zustand store
│       │   ├── api.ts                # Backend API client
│       │   └── formulas.ts           # Parameter formula strings per layer type
│       └── workers/
│           └── layout.worker.ts      # Dagre layout in Web Worker
│
├── backend/                          # FastAPI app → Railway
│   ├── pyproject.toml                # uv-managed dependencies + metadata
│   ├── uv.lock
│   ├── .env                          # ANTHROPIC_API_KEY, HF_TOKEN
│   ├── .env.example
│   └── app/
│       ├── main.py                   # FastAPI app, CORS, route registration
│       ├── models/
│       │   ├── ir.py                 # Pydantic Architecture IR models
│       │   └── api.py                # Request/response schemas
│       ├── resolvers/
│       │   ├── router.py             # Input type detection + routing
│       │   ├── hf_fetcher.py         # HF config.json + model card fetch
│       │   ├── arch_parser.py        # config dict → Architecture IR
│       │   └── file_parser/
│       │       ├── base.py           # ModelFactory base class
│       │       ├── safetensors.py    # Header-only safetensors parser
│       │       ├── gguf.py           # GGUF metadata parser
│       │       ├── onnx.py           # ONNX graph parser
│       │       └── pytorch.py        # PyTorch state_dict key parser
│       ├── compute/
│       │   └── estimator.py          # Formula-based param/FLOPs/memory
│       ├── edit/
│       │   └── engine.py             # EditSpec application + diff generation
│       └── llm/
│           ├── brain.py              # LLM client + tool calling loop
│           └── tools.py              # Tool implementations
│
├── data/
│   └── prebaked/                     # Pre-baked Architecture IRs as JSON
│       ├── bert-base-uncased.json
│       ├── gpt2.json
│       ├── llama-3-8b.json
│       ├── mistral-7b.json
│       └── ...
│
├── docs/
│   └── design_doc.md                 # This document
│
└── README.md
```

---

## 10. API Contracts

All requests/responses use JSON. The backend is stateless — no session storage.

### `POST /resolve`

Resolve a model name or detect input type. The primary endpoint for Phase 1.

**Request:**
```json
{
  "input": "dslim/bert-base-NER"
}
```

**Response:**
```json
{
  "ir": { /* Architecture IR */ },
  "source": "hf_config",
  "cached": true
}
```

---

### `POST /parse-file`

Parse an uploaded model file. Multipart form upload.

**Request:** `multipart/form-data` with field `file`

**Response:**
```json
{
  "ir": { /* Architecture IR */ },
  "format_detected": "safetensors",
  "parse_time_ms": 43
}
```

---

### `POST /compute`

Estimate compute statistics for a given IR.

**Request:**
```json
{
  "ir": { /* Architecture IR */ }
}
```

**Response:**
```json
{
  "params_total": 108890369,
  "params_breakdown": {
    "embedding": 22282752,
    "encoder": 85054464,
    "head": 6922
  },
  "flops_per_token": 22500000000,
  "memory_fp32_gb": 0.416,
  "memory_fp16_gb": 0.208,
  "memory_int8_gb": 0.104,
  "memory_int4_gb": 0.052
}
```

---

### `POST /edit`

Apply an edit spec to an IR and return the result.

**Request:**
```json
{
  "ir": { /* Architecture IR */ },
  "edit_spec": {
    "op": "set_repeat",
    "target": "encoder",
    "value": 14
  }
}
```

**Response:**
```json
{
  "new_ir": { /* modified Architecture IR */ },
  "diff": [
    { "path": "blocks[1].repeat", "old": 12, "new": 14 }
  ],
  "compute_delta": {
    "params_delta": 14175744,
    "params_delta_pct": 13.0,
    "memory_fp16_delta_gb": 0.027
  }
}
```

---

### `POST /chat` (streaming)

Stream a response from the LLM Brain. Server-Sent Events.

**Request:**
```json
{
  "messages": [
    { "role": "user", "content": "Why does this model have 12 attention heads?" }
  ],
  "ir": { /* current Architecture IR */ }
}
```

**Response:** SSE stream (Vercel AI SDK format) with text deltas and tool call events inline.

---

## 11. LLM Brain

The LLM receives the Architecture IR in its system prompt and has access to five tools.

### System prompt structure

```
You are ModelVerse, an expert ML architecture assistant.

The user is currently viewing this model architecture:
<architecture_ir>
{ /* compact IR JSON */ }
</architecture_ir>

You can answer questions about this architecture, suggest edits, explain
concepts, and search for additional information. When estimating the
impact of changes, use the estimate_compute tool rather than guessing.
When asked to edit the architecture, use apply_edit. Always cite your
sources when using information from search results.
```

### Tool definitions

**`search_huggingface`**
```python
def search_huggingface(model_id: str) -> dict:
    """
    Fetch the config.json and model card README for a HuggingFace model.
    Returns the raw config dict and model card text.
    Use this to resolve a model name to its architecture.
    """
```

**`search_web`**
```python
def search_web(query: str) -> list[dict]:
    """
    Search the web for architecture information.
    Returns a list of {title, snippet, url} results.
    Use for: ArXiv papers, HF blog posts, GitHub READMEs, official docs.
    Prefer specific queries: "GPT-3 architecture layers hidden size Brown 2020".
    """
```

**`estimate_compute`**
```python
def estimate_compute(ir: dict) -> dict:
    """
    Compute parameter count, FLOPs per token, and memory estimates
    for the given Architecture IR. Returns exact formula-based results.
    Always call this when the user asks about compute, memory, or
    the impact of an architectural change.
    """
```

**`apply_edit`**
```python
def apply_edit(ir: dict, edit_spec: dict) -> dict:
    """
    Apply an architectural edit to the IR and return the modified IR,
    a diff, and compute delta. Always call this when the user asks to
    change the architecture. Edit spec ops:
      set_repeat:    change num repeated blocks (e.g. transformer layers)
      set_param:     change a specific parameter (hidden_size, num_heads…)
      add_block:     insert a new block
      remove_block:  remove a block
      replace_block: swap one block type for another (e.g. FFN → MoE FFN)
    """
```

**`explain_layer`**
```python
def explain_layer(layer_type: str, params: dict) -> dict:
    """
    Return a structured explanation for a layer type including:
    - description: what it does and why it exists
    - formula: the mathematical operation
    - pseudocode: PyTorch-style implementation
    - why_it_exists: architectural motivation
    Falls back to LLM generation for unknown layer types.
    """
```

---

## 12. Compute Estimator

All estimates are formula-based — no model loading, runs in <1ms.

### Parameter counting (transformer)

```
# Attention block (Multi-Head Attention)
params_attn = 4 * hidden_size^2                     # Q, K, V, O projections
# With GQA: params_attn = hidden_size^2 + 2 * hidden_size * (hidden_size / num_kv_groups)

# Feed-Forward block (standard)
params_ffn = 2 * hidden_size * intermediate_size    # up + down projection
# SwiGLU/GeGLU (used by LLaMA):
params_ffn = 3 * hidden_size * intermediate_size    # gate + up + down

# LayerNorm
params_norm = 2 * hidden_size                       # weight + bias (or just weight for RMSNorm)

# Per transformer block
params_block = params_attn + params_ffn + 2 * params_norm

# Full encoder/decoder stack
params_stack = params_block * num_layers

# Embedding table
params_embedding = vocab_size * hidden_size

# Positional embedding (if learned absolute)
params_pos_embedding = max_position_embeddings * hidden_size

# Total
params_total = params_embedding + params_pos_embedding + params_stack + params_head
```

### FLOPs per forward token (transformer)

Based on Kaplan et al. and Chinchilla scaling laws:

```
# Attention FLOPs per token (sequence length T, hidden size H, num heads A)
flops_attn = 4 * T * H^2 + 2 * T^2 * H    # QKV proj + attention scores + O proj

# FFN FLOPs per token
flops_ffn = 2 * H * intermediate_size       # 2x for multiply + add

# Per block
flops_block = flops_attn + flops_ffn

# Full model (approximate)
flops_forward ≈ 2 * params_non_embedding    # rule of thumb: 2 FLOPs per parameter per token
flops_training ≈ 6 * params_non_embedding   # forward + backward ≈ 3x forward
```

### Memory (weights only)

```
memory_bytes = params_total * bytes_per_dtype

bytes_per_dtype:
  fp32  = 4
  fp16  = 2
  bf16  = 2
  int8  = 1
  int4  = 0.5

# Training memory (weights + gradients + optimizer states with Adam)
memory_training_fp32 = params_total * 4 * (1 + 1 + 2)  # weights + grads + Adam m/v
```

---

## 13. Phased Roadmap

### Phase 1 — HF Model Visualizer

Any model on HuggingFace Hub, by name. No file needed.

**Input modes:**
- HF model ID: `dslim/bert-base-NER`, `mistralai/Mistral-7B-v0.1`, `google/flan-t5-xl`
- Pre-baked library for ~50 popular models (instant, zero API call)

**Features:**
- Interactive React Flow graph from Architecture IR
- Collapsible transformer blocks (click to expand)
- Tensor shapes on edges
- Detail panel: parameter formula hover + PyTorch pseudocode
- LLM chat panel: ask questions about the architecture
- Bottom compute bar: params / memory / FLOPs
- Architecture editing: NL → LLM → edit_spec → diff → re-render
- Compute delta shown on edit
- Edit history (undo/redo)

**Validation models (must work correctly):**
`distilbert-base-uncased`, `bert-base-uncased`, `gpt2`, `meta-llama/Llama-3.1-8B`, `mistralai/Mistral-7B-v0.1`, `t5-base`, `state-spaces/mamba-1.4b-hf`, `mistralai/Mixtral-8x7B-v0.1`

**Build order (block by block):**

1. **Scaffold**: Next.js + FastAPI skeleton, IR types defined (TS + Pydantic)
2. **IR + Pre-baked library**: 5 hand-written IRs, `/resolve` returns them
3. **HF Fetcher + Parser**: `config.json` → IR for any HF model, BERT/GPT/LLaMA families
4. **Basic graph**: React Flow renders IR, collapsible transformer blocks
5. **Detail panel**: click node → params + formula + pseudocode
6. **Compute bar**: formula-based estimator, always-visible bottom bar
7. **LLM chat**: streaming chat, IR in system prompt, Q&A only
8. **Architecture editing**: NL → edit_spec → diff → re-render + compute delta

---

### Phase 2 — File Upload

User uploads their own model file. Parsed server-side, no cloud storage.

- Supported: `.safetensors`, `.onnx`, `.pt` / `.bin`, `.gguf`, `.keras`, `.h5`
- `ModelFactory` pattern — one parser class per format
- All parsers use header/metadata only — no weight loading
- After parse → same visualization + chat + edit as Phase 1

---

### Phase 3 — Any Model (Agentic Web Discovery)

User types any model name regardless of whether it's on HF.

**What this covers:**
- Non-HF models: GPT-3, GPT-4, PaLM, Gemini, Claude, spaCy
- Research models: "the model from the Flash Attention 3 paper"
- Historical models: "original 2017 Transformer", AlexNet, LeNet
- Classical ML: polynomial regression, logistic regression, random forest

**Agentic loop:**
1. Check pre-baked library
2. Check HuggingFace Hub
3. If not found: fire parallel `search_web` calls (ArXiv, HF blog, GitHub, docs)
4. LLM synthesizes IR from evidence
5. IR tagged with `source` + confidence; uncertain fields marked `null` + greyed in UI
6. Sources cited inline in chat panel

---

### Phase 4 — Comparison + Playground + Export

- **Side-by-side comparison**: any two models in parallel columns, differences highlighted, summary bar
- **Architecture Playground**: build from scratch visually, no code
- **Export**: modified IR → HuggingFace `config.json`, PyTorch skeleton, Markdown summary
- **Architecture generation**: "Design a 1B NER model" → LLM generates IR with rationale
- **Model evolution history**: "how did BERT evolve into RoBERTa" → architecture diff
- **Educational mode**: classical ML models as computation graphs

---

## 14. Key Design Decisions

**The Architecture IR is the contract.**
The frontend never calls HF directly. The LLM never sees raw weights. Every component speaks IR. This means the visualization, editing, compute estimation, and LLM context are all identical regardless of whether the input was a name, a file, or a web search.

**config.json-first, never weights.**
`hf_hub_download(filename="config.json")` is all Phase 1 needs. The weights file for BERT is 440MB; `config.json` is 500 bytes. We never download the weights.

**Safetensors header-only.**
The first 8 bytes of a safetensors file are a little-endian uint64 giving the header size. The next N bytes are a JSON object with every tensor's name, shape, and dtype. We never read past the header. (Learned from Netron's `safetensors.Reader`.)

**Pre-baked library for famous models.**
BERT, GPT-2, LLaMA, Mistral — these should load instantly. Shipping 50 pre-baked IRs as JSON means the most common use cases never make a network call.

**React Flow over Netron's vanilla JS grapher.**
Netron's `grapher.js` is 1,100 lines of hand-rolled SVG + Dagre genius. For a pure file viewer with no React, it's optimal. For ModelVerse — where the chat panel, LLM streaming, and Zustand state all live in React — React Flow (which provides the same SVG + Dagre + compound nodes inside React) is the right call.

**Web Worker for graph layout.**
Copied from Netron. Dagre layout for a 80-layer model (LLaMA-70B) is CPU-intensive enough to freeze the UI thread. Layout runs in a Worker; the graph renders when it's ready.

**Tool calling, not multi-agent.**
Five tools cover all Phase 1-3 capabilities. Multi-agent orchestration adds latency, failure modes, and debugging complexity for no benefit at this scale.

**Confidence and source transparency.**
Every IR field knows where it came from. Users can see whether they're looking at exact config data, paper-derived data, or web-synthesized estimates. Partial architectures (some fields unknown) are better than refusing to render.

---

## 15. Infrastructure & Hosting

### Package management

- **Backend**: `uv` — `uv init --app`, `uv add`, `uv sync`, `uv run fastapi dev`. Replaces pip + venv entirely. `uv.lock` committed for reproducible installs.
- **Frontend**: `npm` — standard for Next.js.

### Hosting

| Service | What | Cost | Why |
|---|---|---|---|
| **Vercel** | Next.js frontend | Free tier | Zero-config for Next.js, automatic PR previews, native Vercel AI SDK streaming support |
| **Railway** | FastAPI backend | ~$5–6/month | GitHub → auto-deploy, no cold starts (unlike Render free tier's 30–60s spin-up), usage-based pricing |

### Deployment flow

```
git push to main
  ├── Vercel detects /frontend change → builds and deploys (< 1 min)
  └── Railway detects /backend change → builds with Nixpacks → deploys (< 2 min)

git push to feature branch
  └── Vercel creates a preview URL for the frontend automatically
```

### Environment variables

| Variable | Where | Required | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | backend `.env` | Yes (or OpenAI) | LLM Brain |
| `OPENAI_API_KEY` | backend `.env` | Yes (or Anthropic) | LLM Brain fallback |
| `HF_TOKEN` | backend `.env` | No | Increases HF API rate limits |
| `NEXT_PUBLIC_BACKEND_URL` | frontend `.env.local` | Yes | Points frontend to FastAPI |

### Local development

```bash
# Prerequisites: uv, Node 20+, git

git clone https://github.com/you/modelverse
cd modelverse

# Backend
cd backend
uv sync
cp .env.example .env        # fill in your API keys
uv run fastapi dev           # → http://localhost:8000

# Frontend (new terminal)
cd frontend
npm install
cp .env.example .env.local  # set NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
npm run dev                  # → http://localhost:3000
```
