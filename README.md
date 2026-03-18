<div align="center">

# ModelVerse

**Understand any ML model — just type its name.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://modelverse.vercel.app)

[**→ Try it now at modelverse.vercel.app**](https://modelverse.vercel.app)

</div>

---

ModelVerse turns any machine learning model into an interactive diagram you can explore, question, and edit — no code, no setup, no downloads.

Type a model name. See its full architecture. Click any layer to understand it. Ask questions. Edit it and watch the parameter count change in real time.

---

## What you can do

**Load any model**

Type a model name from HuggingFace — `bert-base-uncased`, `mistralai/Mistral-7B-v0.1`, `google/flan-t5-xl` — and see its architecture appear as an interactive diagram in seconds.

**Explore the architecture**

- Click any block to expand it and see what's inside
- Hover over any parameter count to see the exact formula behind it
- See the tensor shapes flowing between every layer

**Ask questions in plain language**

- *"Why does this model have 12 attention heads?"*
- *"What does the feed-forward layer actually do?"*
- *"Which layer is responsible for token classification?"*

**Edit and see the impact instantly**

- *"What if I reduce the number of layers to 6?"*
- *"What happens if I double the hidden size?"*
- See parameters, memory, and FLOPs update the moment you ask

---

## Example session

```
→  bert-base-uncased

   [Embeddings]  30,522 vocab × 768 dims
        ↓
   [Transformer Encoder ×12]   ← click to expand
     ├─ Self-Attention          params: 2.36M  ← hover to see formula
     ├─ Feed-Forward            params: 4.72M
     └─ LayerNorm ×2            params: 3K
        ↓
   [Pooler]

   108M params · 209MB at fp16 · 22.5B FLOPs/token

you:  what if I cut the number of layers to 6?

ai:   Parameters would drop from 108M to about 66M (−38%).
      Memory at fp16: 209MB → 126MB.
      The model would be faster but less capable — each layer
      contributes to the encoder's ability to build contextual
      representations. Here's the updated architecture: [diff shown]
```

---

## What's supported

| Input | Example | Status |
|---|---|---|
| Any HuggingFace model by name | `dslim/bert-base-NER`, `google/flan-t5-xl` | Available now |
| Upload a model file | `.safetensors`, `.onnx`, `.gguf`, `.pt` | Phase 2 — coming soon |
| Any publicly known model | `"GPT-3"`, `"the original 2017 Transformer"` | Phase 3 — coming soon |

---

## Run locally

You'll need [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python) and [Node.js 20+](https://nodejs.org).

```bash
git clone https://github.com/athira/modelverse
cd modelverse
```

**Backend** (terminal 1):
```bash
cd backend
uv sync
cp .env.example .env
# open .env and add your ANTHROPIC_API_KEY
uv run fastapi dev app/main.py
# → running at http://localhost:8000
```

**Frontend** (terminal 2):
```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
# → open http://localhost:3000
```

Get an API key from [Anthropic](https://console.anthropic.com) or [OpenAI](https://platform.openai.com/api-keys) — either works. The visualizer works without one; you only need it for the chat feature.

---

## Contributing

Want to add a model family, a file format parser, or contribute code? See [CONTRIBUTING.md](CONTRIBUTING.md) for architecture internals, how-to guides, and the PR process.

---

## License

[Apache 2.0](LICENSE)
