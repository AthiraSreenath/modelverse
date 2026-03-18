<div align="center">

# ModelVerse

### Visualize, understand, and edit any ML model — just type its name.

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/UI-Next.js%2015-black?logo=next.js&logoColor=white)
![React Flow](https://img.shields.io/badge/graph-React%20Flow-6366f1)
![uv](https://img.shields.io/badge/deps-uv-DE5FE9)
![License](https://img.shields.io/badge/license-Apache%202.0-D22128)

</div>

---

Type a HuggingFace model name. ModelVerse fetches the architecture — without downloading weights — and renders it as an interactive graph you can explore layer by layer.

Click any block to expand it. See the exact parameter formula behind every number. Ask the built-in LLM chat what a layer does, why it's shaped that way, or what would happen if you changed it. Make that change and watch the parameter count and memory footprint update live.

Everything runs locally. No data leaves your machine except the optional LLM API call.

---

## Features


|                             |                                                                                               |
| --------------------------- | --------------------------------------------------------------------------------------------- |
| 🗺️ **Interactive graph**   | Every layer as a clickable node. Transformer stacks expand to show attention, FFN, and norms. |
| 📐 **Compute stats**        | Parameters, memory (fp16/int4), and FLOPs per token — always visible, always up to date.      |
| 🔍 **Layer inspector**      | Click any node to see what it does, how its parameters are calculated, and its exact config.  |
| 💬 **LLM chat**             | Ask anything about the architecture in plain English. Powered by Claude or GPT-4o.            |
| ✏️ **Architecture editing** | Tell the chat to change the model. See which nodes changed and the exact compute delta.       |
| ↩️ **Undo history**         | Every edit is reversible. Step back through the full edit history.                            |


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

**Step 1 — Create your env files and add your keys**

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
```

Open `backend/.env` and paste in at least one LLM key:

```
ANTHROPIC_API_KEY=sk-ant-...    # → console.anthropic.com
# or
OPENAI_API_KEY=sk-...           # → platform.openai.com/api-keys
```

Anthropic is used if both are set; OpenAI is the fallback. The graph, compute stats, and layer inspector work without any key — you only need one for the chat panel.

Optionally add a HuggingFace token (free, lifts API rate limits):

```
HF_TOKEN=hf_...    # → huggingface.co/settings/tokens
```

`frontend/.env.local` needs no changes for local development.

---

**Step 2 — Start the backend** *(Terminal 1)*

```bash
cd backend
uv sync
uv run fastapi dev app/main.py
# → http://localhost:8000
```

**Step 3 — Start the frontend** *(Terminal 2)*

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

Open **[http://localhost:3000](http://localhost:3000)** and type `bert-base-uncased` to verify everything works.

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

**LLM provider roadmap**

| Provider | Status |
|---|---|
| Anthropic Claude Sonnet | ✅ Supported |
| OpenAI GPT-4o | ✅ Supported |
| Ollama (local LLMs) | 🔜 Planned — run the chat entirely on your own machine, no API key needed |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — architecture overview, how to add a model family parser, how to add a pre-baked model, and the PR process.

## License

[Apache 2.0](LICENSE)