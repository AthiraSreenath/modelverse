"""
Tool implementations for the LLM Brain.
Each function is called when the LLM issues a tool call.
"""

from __future__ import annotations
import json
import logging
import os

import httpx

from ..models.ir import ArchitectureIR, EditSpec
from ..resolvers.hf_fetcher import fetch_hf_config, fetch_model_card, HFConfigNotFoundError
from ..resolvers.arch_parser import parse_hf_config
from ..compute.estimator import estimate_compute
from ..edit.engine import apply_edit as _apply_edit

logger = logging.getLogger(__name__)


async def tool_search_huggingface(model_id: str) -> dict:
    """
    Fetch config.json + model card for a HuggingFace model.
    Returns raw config dict and model card excerpt.
    """
    hf_token = os.getenv("HF_TOKEN")
    try:
        config = await fetch_hf_config(model_id, hf_token=hf_token)
        card = await fetch_model_card(model_id, hf_token=hf_token)
        ir = parse_hf_config(config, model_id=model_id)
        return {
            "found": True,
            "model_id": model_id,
            "config": config,
            "model_card_excerpt": card,
            "ir": ir.model_dump(),
        }
    except HFConfigNotFoundError:
        return {"found": False, "model_id": model_id, "error": f"'{model_id}' not found on HuggingFace Hub"}
    except Exception as e:
        return {"found": False, "model_id": model_id, "error": str(e)}


async def tool_search_web(query: str) -> dict:
    """
    Search the web for architecture information via DuckDuckGo instant answer API.
    For Phase 3 - finding architectures of models not on HF.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_redirect": 1, "no_html": 1},
            )
            data = resp.json()
            results = []
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading", query),
                    "snippet": data["AbstractText"][:500],
                    "url": data.get("AbstractURL", ""),
                })
            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("Text", "")[:80],
                        "snippet": topic.get("Text", "")[:300],
                        "url": topic.get("FirstURL", ""),
                    })
            return {"query": query, "results": results}
    except Exception as e:
        logger.warning("Web search failed for '%s': %s", query, e)
        return {"query": query, "results": [], "error": str(e)}


def tool_estimate_compute(ir: dict) -> dict:
    """Estimate compute stats for a given Architecture IR dict."""
    try:
        ir_obj = ArchitectureIR.model_validate(ir)
        stats = estimate_compute(ir_obj)
        return stats.model_dump(mode="json")
    except Exception as e:
        return {"error": str(e)}


def _collect_block_ids(blocks: list) -> list[str]:
    ids = []
    for b in blocks:
        ids.append(b.id)
        ids.extend(_collect_block_ids(b.children))
    return ids


def tool_apply_edit(ir: dict, edit_spec: dict) -> dict:
    """Apply an edit spec to an Architecture IR and return result + diff + compute delta."""
    try:
        ir_obj = ArchitectureIR.model_validate(ir)
        spec = EditSpec.model_validate(edit_spec)
        result = _apply_edit(ir_obj, spec)
        return result.model_dump(mode="json")
    except ValueError as e:
        # Help the LLM correct itself by listing available block IDs
        try:
            ir_obj = ArchitectureIR.model_validate(ir)
            available = _collect_block_ids(ir_obj.blocks)
            return {
                "error": str(e),
                "available_block_ids": available,
                "hint": f"Use one of the available block IDs: {available}",
            }
        except Exception:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


def tool_explain_layer(layer_type: str, params: dict) -> dict:
    """
    Return a structured explanation for a layer type.
    Pre-baked for common types; LLM generates for unknown types.
    """
    explanations = {
        "multi_head_attention": {
            "description": "Multi-head attention projects the input into multiple query, key, and value subspaces, computes scaled dot-product attention in each, then concatenates and projects back.",
            "formula": "Attention(Q,K,V) = softmax(QK^T / √d_k) V",
            "pseudocode": """class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask=None):
        Q = self.q_proj(x)  # [B, T, H]
        K = self.k_proj(x)
        V = self.v_proj(x)
        # split into heads, attend, merge
        scores = Q @ K.transpose(-2, -1) / math.sqrt(head_dim)
        if mask: scores = scores.masked_fill(mask, -inf)
        out = softmax(scores) @ V
        return self.o_proj(out)""",
            "why_it_exists": "Splitting into multiple heads lets the model attend to different aspects of the input simultaneously - syntax in one head, semantics in another. 12 heads at d=768 gives each head a 64-dim subspace.",
        },
        "feed_forward": {
            "description": "A two-layer MLP applied independently to each token position. Expands then contracts the representation, acting as a 'memory' or 'lookup table' for factual knowledge.",
            "formula": "FFN(x) = W₂ · GELU(W₁x + b₁) + b₂",
            "pseudocode": """class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))""",
            "why_it_exists": "Attention captures relationships between tokens; FFN stores and retrieves knowledge per token. Research shows FFN layers store factual associations (e.g. Paris → France).",
        },
        "embedding": {
            "description": "Looks up a learned dense vector for each token ID, position, and (in BERT) segment type. These are summed and fed into the transformer.",
            "formula": "E(token) = W_vocab[token_id] + W_pos[position] + W_seg[segment]",
            "pseudocode": """class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_pos):
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_pos, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        positions = torch.arange(input_ids.size(1))
        x = self.word_embeddings(input_ids) + self.position_embeddings(positions)
        return self.dropout(self.LayerNorm(x))""",
            "why_it_exists": "Models need a continuous vector space to operate in. The embedding table is the bridge between discrete token IDs and continuous representations.",
        },
        "layer_norm": {
            "description": "Normalizes activations across the hidden dimension to zero mean and unit variance, then scales and shifts with learned parameters.",
            "formula": "LN(x) = (x - μ) / (σ + ε) · γ + β",
            "pseudocode": """class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.weight * (x - mean) / (std + 1e-5) + self.bias""",
            "why_it_exists": "Stabilises training by preventing activations from growing or vanishing. Critical for deep networks - without it, gradients become unstable after ~4 layers.",
        },
        "moe_feed_forward": {
            "description": "A Mixture-of-Experts FFN routes each token to the top-K of N expert networks. Only K experts activate per token, keeping compute constant while multiplying model capacity.",
            "formula": "MoE(x) = Σ_{i∈TopK(router(x))} gate_i · Expert_i(x)",
            "pseudocode": """class MoELayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts, top_k):
        self.router = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([FFN(hidden_size, intermediate_size)
                                      for _ in range(num_experts)])

    def forward(self, x):
        scores = self.router(x)          # [B, T, num_experts]
        weights, indices = scores.topk(self.top_k, dim=-1)
        weights = softmax(weights)
        out = sum(weights[..., i:i+1] * self.experts[idx](x)
                  for i, idx in enumerate(indices.unbind(-1)))
        return out""",
            "why_it_exists": "Scales model capacity (total params) without scaling compute per token. Mixtral-8x7B has 8 experts but only activates 2 per token, giving ~13B active params from a 47B parameter model.",
        },
        "ssm": {
            "description": "A Selective State Space Model layer. Unlike attention, it processes sequences recurrently with state-selective dynamics - O(n) in sequence length vs O(n²) for attention.",
            "formula": "h_t = A·h_{t-1} + B·x_t,  y_t = C·h_t + D·x_t  (A, B, C input-dependent)",
            "pseudocode": """class MambaSSM(nn.Module):
    def forward(self, x):
        # Input-dependent state matrices (the 'selective' part)
        B = self.B_proj(x)    # [B, T, d_state]
        C = self.C_proj(x)
        dt = softplus(self.dt_proj(x))   # time step
        # Discretize A with ZOH
        A_bar = torch.exp(-torch.exp(self.log_A) * dt)
        # Parallel scan over sequence
        y = selective_scan(x, A_bar, B, C, self.D)
        return y""",
            "why_it_exists": "Attention is O(n²) in sequence length - expensive for long contexts. SSMs are O(n) and can model long-range dependencies through their state, though they lack random access like attention.",
        },
    }

    explanation = explanations.get(layer_type)
    if explanation:
        return explanation

    # Generic fallback for unknown types
    return {
        "description": f"A {layer_type} layer. Parameters: {params}",
        "formula": "See model documentation",
        "pseudocode": f"# {layer_type} - implementation varies by architecture",
        "why_it_exists": "Contributes to the model's representation learning capacity.",
    }
