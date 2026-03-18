<div align="center">

# ModelVerse

**Visualize, explore, and edit any ML model — just type its name.**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-black?logo=next.js)](https://nextjs.org)
[![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9)](https://docs.astral.sh/uv)

</div>

---

ModelVerse turns any HuggingFace model into an interactive architecture diagram. Type a name, see every layer, click to explore, ask questions, and edit the architecture — all in one place, without writing a single line of code.

---

## Features

| | |
|---|---|
| 🗺️ **Interactive graph** | Every layer as a clickable node. Transformer stacks expand to show attention, FFN, and norms. |
| 📐 **Compute stats** | Parameters, memory (fp16/int4), and FLOPs per token — always visible, always up to date. |
| 🔍 **Layer inspector** | Click any node to see what it does, how its parameters are calculated, and its exact config. |
| 💬 **LLM chat** | Ask anything about the architecture in plain English. Powered by Claude or GPT-4o. |
| ✏️ **Architecture editing** | Tell the chat to change the model. See which nodes changed and the exact compute delta. |
| ↩️ **Undo history** | Every edit is reversible. Step back through the full edit history. |

---

## How it works

```
You type:  bert-base-uncased
                │
                ▼
   ┌─────────────────────────────────────────────────┐
   │  Embeddings              23M params              │
   │       │                                          │
   │  Transformer Encoder ×12    ← click to expand    │
   │  │  Self-Attention   2M  ←  hover: see formula   │
   │  │  LayerNorm        2K                          │
   │  │  Feed-Forward     5M                          │  ←  graph panel
   │  └  LayerNorm        2K                          │
   │       │                                          │
   │  Pooler               590K                       │
   └─────────────────────────────────────────────────┘
   [ 108M params · 209MB fp16 · 22.5B FLOPs/token ]  ←  compute bar

   ┌─────────────────────┐   ┌────────────────────────┐
   │  Self-Attention      │   │  you: what if I cut     │
   │  ──────────────      │   │       layers to 6?      │
   │  Q  768×768  2.36M  │   │                         │
   │  K  768×768  2.36M  │   │  ai:  −38% params       │
   │  V  768×768  2.36M  │   │  209MB → 126MB fp16     │
   │  O  768×768  2.36M  │   │  [graph updates live]   │
   └─────────────────────┘   └────────────────────────┘
         detail panel                 chat panel
```

---

## Supported models

**Loads instantly** (pre-baked, no network call):

`bert-base-uncased` · `distilbert-base-uncased` · `gpt2` · `t5-base` · `meta-llama/Llama-3.1-8B` · `mistralai/Mistral-7B-v0.1`

**Any other HuggingFace model:**

Type the model ID exactly as it appears on HuggingFace (`owner/model-name`). ModelVerse fetches only the `config.json` — never the weights — and builds the full architecture from it.

> Works with BERT, GPT-2, LLaMA, Mistral, Mixtral, T5, Mamba, DeBERTa, ELECTRA, Falcon, OLMo, Cohere, and more.

---

## Run locally

**Requirements:** [uv](https://docs.astral.sh/uv/getting-started/installation/) · [Node.js 20+](https://nodejs.org)

```bash
git clone https://github.com/AthiraSreenath/modelverse
cd modelverse
```

**Terminal 1 — backend**
```bash
cd backend
uv sync
cp .env.example .env
# → open .env, paste your ANTHROPIC_API_KEY or OPENAI_API_KEY
uv run fastapi dev app/main.py
```

**Terminal 2 — frontend**
```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Open **[http://localhost:3000](http://localhost:3000)** → type `bert-base-uncased`.

> **No API key?** The graph, compute stats, and layer inspector all work without one. An Anthropic or OpenAI key is only needed for the chat panel. Get one free at [console.anthropic.com](https://console.anthropic.com) or [platform.openai.com](https://platform.openai.com/api-keys).

---

## Roadmap

```
Phase 1  ████████████████  ✅ Now       HuggingFace models by name
Phase 2  ░░░░░░░░░░░░░░░░  🔜 Next      File upload (.safetensors, .onnx, .gguf, .pt)
Phase 3  ░░░░░░░░░░░░░░░░  🔜 Later     Any model — agentic web + ArXiv search
Phase 4  ░░░░░░░░░░░░░░░░  🔜 Future    Side-by-side comparison + playground + export
```

**Phase 2 — File Upload**
Upload a local model file. ModelVerse reads only the header (tensor names and shapes) — never the weights — and renders the full architecture. Supports `.safetensors`, `.onnx`, `.gguf`, `.pt`.

**Phase 3 — Any Model (Agentic Discovery)**
Type any model name — `GPT-3`, `"the original 2017 Transformer"`, `AlexNet`. The LLM brain searches HuggingFace, ArXiv, and the web, synthesizes the architecture from whatever evidence it finds, and shows confidence levels for each part.

**Phase 4 — Comparison + Playground + Export**
Load two models side by side. See exactly what changed — layer by layer, parameter by parameter. Export architectures as JSON, SVG, or directly to PyTorch code.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — architecture overview, how to add a model family parser, how to add a pre-baked model, and the PR process.

## License

[Apache 2.0](LICENSE)
