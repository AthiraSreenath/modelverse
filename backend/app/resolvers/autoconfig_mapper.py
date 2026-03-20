"""
AutoConfig-based Architecture Mapper
=====================================
Converts any HuggingFace config.json into ArchitectureIR using the
transformers.AutoConfig object as the ground truth for field normalization.

This replaces all per-family Tier-1 parsers for HF transformer models.
It covers all 475+ model types in transformers.CONFIG_MAPPING and falls back
gracefully to raw-dict parsing for unregistered types (trust_remote_code
models, custom configs, etc).

No family-specific code. One mapper, every architecture.
"""

from __future__ import annotations

import logging
from typing import Any

from ..models.ir import (
    ArchBlock, ArchitectureIR, BlockType,
    SourceType, SourceConfidence,
)
from ..compute.estimator import estimate_compute

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw-dict adapter — attribute access for non-registry configs
# ---------------------------------------------------------------------------

class _Adapter:
    """Wraps a raw config dict so getattr() works the same as on AutoConfig."""
    def __init__(self, d: dict) -> None:
        object.__setattr__(self, "_d", d)

    def __getattr__(self, name: str) -> Any:
        return object.__getattribute__(self, "_d").get(name)


# ---------------------------------------------------------------------------
# Field extractors with multi-name fallback
# ---------------------------------------------------------------------------

def _get(cfg: Any, *attrs: str, default: Any = None) -> Any:
    for a in attrs:
        v = getattr(cfg, a, None)
        if v is not None:
            return v
    return default


def _hidden_size(cfg: Any) -> int:
    return int(_get(cfg, "hidden_size", "d_model", "n_embd", "d_embed", default=768))


def _num_layers(cfg: Any) -> int:
    return int(_get(cfg, "num_hidden_layers", "num_layers", "n_layer", default=12))


def _num_heads(cfg: Any) -> int:
    return int(_get(cfg, "num_attention_heads", "n_head", "num_heads", default=12))


def _num_kv_heads(cfg: Any, num_heads: int) -> int:
    v = _get(cfg, "num_key_value_heads", "num_kv_heads")
    return int(v) if v else num_heads


def _intermediate_size(cfg: Any, h: int) -> int:
    v = _get(cfg, "intermediate_size", "d_ff", "n_inner", "ffn_dim")
    return int(v) if v else h * 4


def _vocab_size(cfg: Any) -> int:
    return int(_get(cfg, "vocab_size", default=32000))


def _activation(cfg: Any) -> str:
    a = _get(cfg, "hidden_act", "activation_function", "act_fn", default="gelu")
    return (a or "gelu").lower()


def _norm_type(cfg: Any) -> str:
    if _get(cfg, "rms_norm_eps") is not None:
        return "rms_norm"
    nt = _get(cfg, "norm_type", "normalization_type", "layer_norm_type")
    if nt and "rms" in str(nt).lower():
        return "rms_norm"
    return "layer_norm"


def _head_dim(cfg: Any, h: int, num_heads: int) -> int:
    hd = _get(cfg, "head_dim")
    return int(hd) if hd else (h // num_heads if num_heads else 64)


def _is_gated(act: str) -> bool:
    return act in ("silu", "swiglu", "geglu", "gelu_pytorch_tanh", "gelu_fast")


def _is_moe(cfg: Any) -> bool:
    n = _get(cfg, "num_local_experts", "num_experts", "n_routed_experts", default=0)
    return int(n or 0) > 1


def _n_experts(cfg: Any) -> int:
    return int(_get(cfg, "num_local_experts", "num_experts", "n_routed_experts", default=1))


def _n_experts_per_tok(cfg: Any) -> int:
    return int(_get(cfg, "num_experts_per_tok", "top_k", "n_experts_per_tok", default=2))


def _n_shared_experts(cfg: Any) -> int:
    return int(_get(cfg, "n_shared_experts", "num_shared_experts", default=0) or 0)


def _moe_intermediate(cfg: Any, inter: int) -> int:
    v = _get(cfg, "moe_intermediate_size")
    return int(v) if v else inter


def _n_dense_layers(cfg: Any) -> int:
    v = _get(cfg, "first_k_dense_replace", "n_dense_layers", "num_dense_layers", default=0)
    return int(v or 0)


def _is_ssm_model(cfg: Any) -> bool:
    """Mamba / SSM: has state_size but no attention heads."""
    return (
        _get(cfg, "state_size") is not None
        and _get(cfg, "num_attention_heads") is None
        and _get(cfg, "n_head") is None
    )


def _is_enc_dec(cfg: Any) -> bool:
    v = _get(cfg, "is_encoder_decoder")
    return bool(v) if v is not None else False


def _tie_embeddings(cfg: Any) -> bool:
    v = _get(cfg, "tie_word_embeddings")
    # Default: True for enc-dec (T5), False for decoder-only (LLaMA)
    return bool(v) if v is not None else True


# ---------------------------------------------------------------------------
# Parameter arithmetic helpers
# ---------------------------------------------------------------------------

def _attn_p(h: int, num_heads: int, num_kv_heads: int, head_dim: int,
            bias: bool = False) -> int:
    q = h * (num_heads * head_dim)
    k = h * (num_kv_heads * head_dim)
    v = h * (num_kv_heads * head_dim)
    o = (num_heads * head_dim) * h
    b = (num_heads * head_dim + num_kv_heads * head_dim * 2 + h) if bias else 0
    return q + k + v + o + b


def _ffn_p(h: int, inter: int, act: str, bias: bool = False) -> int:
    if _is_gated(act):
        # gate_proj + up_proj + down_proj
        return 3 * h * inter + (3 * inter if bias else 0)
    return 2 * h * inter + (h + inter if bias else 0)


def _norm_p(h: int, norm_t: str) -> int:
    # RMSNorm: weight only. LayerNorm: weight + bias.
    return h if norm_t == "rms_norm" else 2 * h


def _per_layer_p(h: int, num_heads: int, num_kv_heads: int, head_dim: int,
                 inter: int, act: str, norm_t: str,
                 cross_attn: bool = False) -> int:
    n_norms = 3 if cross_attn else 2
    return (
        _attn_p(h, num_heads, num_kv_heads, head_dim)
        + (_attn_p(h, num_heads, num_kv_heads, head_dim) if cross_attn else 0)
        + _ffn_p(h, inter, act)
        + _norm_p(h, norm_t) * n_norms
    )


def _moe_layer_p(h: int, num_heads: int, num_kv_heads: int, head_dim: int,
                 n_experts: int, n_shared: int, moe_inter: int,
                 act: str, norm_t: str) -> int:
    router = h * n_experts
    routed = n_experts * _ffn_p(h, moe_inter, act)
    shared = n_shared * _ffn_p(h, moe_inter, act)
    return _attn_p(h, num_heads, num_kv_heads, head_dim) + router + routed + shared + _norm_p(h, norm_t) * 2


def _ssm_layer_p(h: int, state: int, d_conv: int, expand: int) -> int:
    d_inner = h * expand
    dt_rank = max(1, d_inner // 16)
    return (
        h * 2 * d_inner          # in_proj
        + d_inner * d_conv        # depthwise conv
        + d_inner * (dt_rank + 2 * state)  # x_proj
        + dt_rank * d_inner       # dt_proj
        + d_inner * h             # out_proj
    )


# ---------------------------------------------------------------------------
# Architecture type inference
# ---------------------------------------------------------------------------

_KNOWN_DECODERS = {
    "gpt2", "gpt_neo", "gpt_neox", "gptj", "gpt_j",
    "llama", "mistral", "mixtral", "falcon", "phi", "phi3",
    "qwen2", "qwen2_moe", "starcoder2", "codegen", "bloom", "opt",
    "gemma", "gemma2", "mamba", "rwkv", "deepseek_v2", "deepseek_v3",
    "olmo", "olmo2", "command_r", "dbrx", "stablelm", "internlm2",
    "solar", "xverse", "yuan", "persimmon", "open_llama",
}

_KNOWN_ENCODERS = {
    "bert", "roberta", "albert", "electra", "deberta", "deberta_v2",
    "distilbert", "xlm_roberta", "camembert", "flaubert",
    "ernie", "nezha", "roformer",
}


def _infer_arch_type(cfg: Any, architectures: list[str]) -> str:
    if _is_enc_dec(cfg):
        return "encoder_decoder"
    if _is_ssm_model(cfg):
        return "decoder"
    mt = (getattr(cfg, "model_type", "") or "").lower()
    if mt in _KNOWN_DECODERS:
        return "decoder"
    if mt in _KNOWN_ENCODERS:
        return "encoder"
    # Check class names
    for a in architectures:
        if "CausalLM" in a or "causal" in a.lower():
            return "decoder"
        if "MaskedLM" in a:
            return "encoder"
    # RoPE / ALiBi → likely causal decoder
    if _get(cfg, "rope_theta") or _get(cfg, "rope_scaling") or _get(cfg, "use_alibi"):
        return "decoder"
    return "decoder"  # safer default: most modern models are decoders


def _infer_task(architectures: list[str], arch_type: str) -> str:
    s = " ".join(architectures).lower()
    if "causallm" in s:
        return "text-generation"
    if "seq2seq" in s or "conditional" in s:
        return "text2text-generation"
    if "maskedlm" in s:
        return "fill-mask"
    if "sequenceclassification" in s:
        return "text-classification"
    if "tokenclassification" in s:
        return "token-classification"
    if "questionanswering" in s:
        return "question-answering"
    if arch_type == "encoder_decoder":
        return "text2text-generation"
    if arch_type == "decoder":
        return "text-generation"
    return "text-classification"


# ---------------------------------------------------------------------------
# Children block builders (per-layer sub-blocks for expand/collapse)
# ---------------------------------------------------------------------------

def _norm_block(idx: int, h: int, norm_t: str) -> ArchBlock:
    return ArchBlock(
        id=f"norm{idx}",
        label="RMSNorm" if norm_t == "rms_norm" else "LayerNorm",
        type=BlockType.LAYER_NORM,
        params={"normalized_shape": h, "norm_type": norm_t},
        param_count=_norm_p(h, norm_t),
    )


def _add_block(idx: int) -> ArchBlock:
    return ArchBlock(id=f"add{idx}", label="Add (residual)", type=BlockType.ADD,
                     params={}, param_count=0)


def _attn_block(h: int, num_heads: int, num_kv_heads: int, head_dim: int,
                is_causal: bool, cross: bool = False) -> ArchBlock:
    return ArchBlock(
        id="cross_attn" if cross else "self_attn",
        label="Cross-Attention" if cross else ("Causal Self-Attention" if is_causal else "Self-Attention"),
        type=BlockType.MULTI_HEAD_ATTENTION,
        params={"hidden_size": h, "num_heads": num_heads,
                "num_kv_heads": num_kv_heads, "head_dim": head_dim,
                "is_causal": is_causal},
        param_count=_attn_p(h, num_heads, num_kv_heads, head_dim),
    )


def _ffn_block(h: int, inter: int, act: str) -> ArchBlock:
    label = f"SwiGLU FFN" if _is_gated(act) else f"FFN ({act})"
    return ArchBlock(
        id="ffn",
        label=label,
        type=BlockType.FEED_FORWARD,
        params={"hidden_size": h, "intermediate_size": inter, "activation": act},
        param_count=_ffn_p(h, inter, act),
    )


def _make_layer_children(
    h: int, num_heads: int, num_kv_heads: int, head_dim: int,
    inter: int, act: str, norm_t: str,
    is_causal: bool, cross_attn: bool = False,
    residual_layout: str = "pre_ln",
) -> list[ArchBlock]:
    """Build one transformer layer's sub-blocks (pre-LN or post-LN)."""
    attn = _attn_block(h, num_heads, num_kv_heads, head_dim, is_causal)
    ffn = _ffn_block(h, inter, act)

    if residual_layout == "post_ln":
        children: list[ArchBlock] = [
            attn, _add_block(1), _norm_block(1, h, norm_t),
            ffn, _add_block(2), _norm_block(2, h, norm_t),
        ]
    else:  # pre_ln (most modern models)
        children = [
            _norm_block(1, h, norm_t), attn, _add_block(1),
            _norm_block(2, h, norm_t), ffn, _add_block(2),
        ]

    if cross_attn:
        children.insert(3, _attn_block(h, num_heads, num_kv_heads, head_dim, False, cross=True))
        children.insert(4, _add_block(3))
        children.insert(4, _norm_block(3, h, norm_t))

    return children


# ---------------------------------------------------------------------------
# Main mapper
# ---------------------------------------------------------------------------

def map_autoconfig_to_ir(raw_config: dict, model_id: str) -> ArchitectureIR:
    """
    Convert a raw HuggingFace config.json dict into an ArchitectureIR.

    Uses transformers.AutoConfig.for_model() for the ~475 registered model
    types to normalise field names and apply model-specific defaults.
    Falls back to a raw-dict adapter for custom / trust_remote_code models.

    Confidence:
      HIGH   — model_type is in CONFIG_MAPPING and AutoConfig loaded cleanly
      MEDIUM — model_type unknown or AutoConfig failed; raw dict used instead
    """
    try:
        from transformers import AutoConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        _transformers_available = True
    except ImportError:
        _transformers_available = False
        CONFIG_MAPPING = {}

    model_type = (raw_config.get("model_type") or "").lower()
    architectures: list[str] = raw_config.get("architectures") or []
    in_registry = _transformers_available and model_type in CONFIG_MAPPING

    cfg: Any
    if in_registry:
        try:
            # Pass all primitive fields; skip nested dicts that aren't standard
            safe = {k: v for k, v in raw_config.items()
                    if k != "model_type" and not isinstance(v, (dict,)) or k in ("rope_scaling",)}
            cfg = AutoConfig.for_model(model_type, **safe)
            confidence = SourceConfidence.HIGH
            logger.debug("AutoConfig loaded for model_type=%s", model_type)
        except Exception as e:
            logger.debug("AutoConfig.for_model failed for %s: %s — using raw dict", model_type, e)
            cfg = _Adapter(raw_config)
            confidence = SourceConfidence.MEDIUM
    else:
        cfg = _Adapter(raw_config)
        confidence = SourceConfidence.MEDIUM if model_type else SourceConfidence.LOW

    # ── Core dimensions ──────────────────────────────────────────────────────
    h = _hidden_size(cfg)
    num_heads = _num_heads(cfg)
    num_kv_heads = _num_kv_heads(cfg, num_heads)
    head_dim = _head_dim(cfg, h, num_heads)
    inter = _intermediate_size(cfg, h)
    vocab = _vocab_size(cfg)
    act = _activation(cfg)
    norm_t = _norm_type(cfg)
    arch_type = _infer_arch_type(cfg, architectures)
    is_ssm = _is_ssm_model(cfg)
    moe = _is_moe(cfg)
    tied = _tie_embeddings(cfg)

    num_layers = _num_layers(cfg)
    num_enc_layers = num_layers
    num_dec_layers = num_layers
    if arch_type == "encoder_decoder":
        el = _get(cfg, "num_encoder_layers", "num_layers")
        dl = _get(cfg, "num_decoder_layers")
        if el:
            num_enc_layers = int(el)
        if dl:
            num_dec_layers = int(dl)
        else:
            num_dec_layers = num_enc_layers

    # Residual layout
    res_layout = "pre_ln" if (norm_t == "rms_norm" or arch_type == "decoder") else "post_ln"

    blocks: list[ArchBlock] = []

    # ── 1. Token embeddings ──────────────────────────────────────────────────
    emb_p = vocab * h
    max_pos = int(_get(cfg, "max_position_embeddings", default=0) or 0)
    type_vocab = int(_get(cfg, "type_vocab_size", default=0) or 0)
    has_rope = bool(_get(cfg, "rope_theta") or _get(cfg, "rope_scaling"))
    pos_type = "none" if has_rope else ("absolute" if max_pos else "none")

    if pos_type == "absolute" and not has_rope:
        emb_p += max_pos * h
    if type_vocab:
        emb_p += type_vocab * h

    blocks.append(ArchBlock(
        id="token_embeddings",
        label="Token Embeddings",
        type=BlockType.EMBEDDING,
        params={"vocab_size": vocab, "hidden_size": h,
                "max_position_embeddings": max_pos,
                "position_embedding_type": pos_type,
                "type_vocab_size": type_vocab},
        param_count=emb_p,
        notes=(
            f"vocab_size={vocab:,}, hidden_size={h}"
            + (f"\nRoPE applied at attention time — no position embedding table." if has_rope else
               f"\nAbsolute position embeddings: {max_pos} positions × {h}." if max_pos else "")
            + (f"\nSegment embeddings: {type_vocab} types × {h}." if type_vocab else "")
        ),
    ))

    # ── 2. Transformer / SSM stack ───────────────────────────────────────────
    if is_ssm:
        state = int(_get(cfg, "state_size", default=16))
        d_conv = int(_get(cfg, "d_conv", default=4))
        expand = int(_get(cfg, "expand", default=2))
        per_layer = _ssm_layer_p(h, state, d_conv, expand)
        blocks.append(ArchBlock(
            id="ssm_layers",
            label=f"SSM Layers (×{num_layers})",
            type=BlockType.SSM,
            params={"hidden_size": h, "state_size": state, "conv_kernel": d_conv,
                    "expand": expand, "num_hidden_layers": num_layers},
            repeat=num_layers,
            children=[ArchBlock(
                id="ssm_op", label="SSM Block",
                type=BlockType.SSM,
                params={"hidden_size": h, "state_size": state,
                        "conv_kernel": d_conv, "expand": expand},
                param_count=per_layer,
            )],
            param_count=per_layer * num_layers,
            notes=f"{num_layers} SSM layers. Each: in_proj + depthwise conv + SSM scan + out_proj.\nhidden={h}, state={state}, expand={expand}.",
        ))

    elif arch_type == "encoder_decoder":
        # Encoder
        enc_per = _per_layer_p(h, num_heads, num_kv_heads, head_dim, inter, act, norm_t)
        enc_children = _make_layer_children(
            h, num_heads, num_kv_heads, head_dim, inter, act, norm_t,
            is_causal=False, residual_layout=res_layout)
        blocks.append(ArchBlock(
            id="encoder",
            label=f"Encoder ({num_enc_layers} layers)",
            type=BlockType.TRANSFORMER_STACK,
            params={"num_hidden_layers": num_enc_layers, "hidden_size": h,
                    "num_attention_heads": num_heads, "intermediate_size": inter,
                    "is_causal": False, "residual_layout": res_layout},
            repeat=num_enc_layers,
            children=enc_children,
            param_count=enc_per * num_enc_layers,
            notes=f"Bidirectional encoder — {num_enc_layers} layers, hidden={h}, heads={num_heads}, inter={inter}.",
        ))
        # Decoder
        dec_per = _per_layer_p(h, num_heads, num_kv_heads, head_dim, inter, act, norm_t, cross_attn=True)
        dec_children = _make_layer_children(
            h, num_heads, num_kv_heads, head_dim, inter, act, norm_t,
            is_causal=True, cross_attn=True, residual_layout="pre_ln")
        blocks.append(ArchBlock(
            id="decoder",
            label=f"Decoder ({num_dec_layers} layers)",
            type=BlockType.TRANSFORMER_STACK,
            params={"num_hidden_layers": num_dec_layers, "hidden_size": h,
                    "num_attention_heads": num_heads, "intermediate_size": inter,
                    "is_causal": True, "residual_layout": "pre_ln"},
            repeat=num_dec_layers,
            children=dec_children,
            param_count=dec_per * num_dec_layers,
            notes=f"Causal decoder — {num_dec_layers} layers × (self-attn + cross-attn + FFN).",
        ))

    elif moe:
        n_exp = _n_experts(cfg)
        n_shared = _n_shared_experts(cfg)
        n_per_tok = _n_experts_per_tok(cfg)
        moe_inter = _moe_intermediate(cfg, inter)
        n_dense = _n_dense_layers(cfg)
        n_moe = num_layers - n_dense

        dense_per = _per_layer_p(h, num_heads, num_kv_heads, head_dim, inter, act, norm_t)
        moe_per = _moe_layer_p(h, num_heads, num_kv_heads, head_dim,
                                n_exp, n_shared, moe_inter, act, norm_t)
        total_stored = n_dense * dense_per + n_moe * moe_per

        # Active params: attn + top-k routed FFN + shared FFN + router + norms
        router_p = h * n_exp
        active_ffn = (n_per_tok + n_shared) * _ffn_p(h, moe_inter, act)
        active_per = _attn_p(h, num_heads, num_kv_heads, head_dim) + router_p + active_ffn + _norm_p(h, norm_t) * 2
        active_total = n_dense * dense_per + n_moe * active_per

        exp_label = (f"{n_exp} routed" + (f" + {n_shared} shared" if n_shared else "")
                     + f", top-{n_per_tok} active/token")
        label = f"LM Decoder — MoE ({exp_label} · stored params)"

        blocks.append(ArchBlock(
            id="transformer_layers",
            label=label,
            type=BlockType.TRANSFORMER_STACK,
            params={"num_hidden_layers": num_layers, "hidden_size": h,
                    "num_attention_heads": num_heads, "num_key_value_heads": num_kv_heads,
                    "num_local_experts": n_exp, "num_experts_per_tok": n_per_tok,
                    "is_causal": True},
            repeat=1,
            children=[],   # no children: param_count is total; estimator uses leaf path
            param_count=total_stored,
            notes=(
                f"Params shown = stored (all experts saved to disk).\n"
                f"  • Stored:  {total_stored/1e9:.3f}B  "
                f"({n_dense} dense layer{'s' if n_dense != 1 else ''} + {n_moe} MoE layer{'s' if n_moe != 1 else ''})\n"
                f"  • Active per token: {active_total/1e9:.3f}B  "
                f"(top-{n_per_tok} of {n_exp} routed fire; {n_exp - n_per_tok} idle per step"
                + (f"; {n_shared} shared always-on" if n_shared else "") + ")\n\n"
                f"Attention: {num_heads} Q heads / {num_kv_heads} KV heads, head_dim={head_dim}.\n"
                f"MoE FFN: {n_exp} experts × {moe_inter} intermediate, act={act}."
            ),
        ))

    else:
        # Standard encoder or decoder
        is_dec = arch_type == "decoder"
        per_layer = _per_layer_p(h, num_heads, num_kv_heads, head_dim, inter, act, norm_t)
        children = _make_layer_children(
            h, num_heads, num_kv_heads, head_dim, inter, act, norm_t,
            is_causal=is_dec, residual_layout=res_layout)

        gqa_note = f" — GQA {num_heads}Q/{num_kv_heads}KV" if num_kv_heads < num_heads else ""
        label = ("LM Decoder" if is_dec else "Encoder") + gqa_note

        blocks.append(ArchBlock(
            id="transformer_layers",
            label=label,
            type=BlockType.TRANSFORMER_STACK,
            params={"num_hidden_layers": num_layers, "hidden_size": h,
                    "num_attention_heads": num_heads, "num_key_value_heads": num_kv_heads,
                    "intermediate_size": inter, "activation": act,
                    "is_causal": is_dec, "residual_layout": res_layout},
            repeat=num_layers,
            children=children,
            param_count=per_layer * num_layers,
            notes=(
                f"{num_layers} {'causal decoder' if is_dec else 'encoder'} layer{'s' if num_layers != 1 else ''}.\n"
                f"hidden={h}, {num_heads} Q heads, {num_kv_heads} KV heads, head_dim={head_dim}\n"
                f"intermediate={inter}, act={act}, norm={norm_t}"
                + (f"\nRoPE (theta={_get(cfg, 'rope_theta', default='N/A')})" if has_rope else "")
            ),
        ))

    # ── 3. Final layer norm ──────────────────────────────────────────────────
    blocks.append(ArchBlock(
        id="final_norm",
        label="RMSNorm" if norm_t == "rms_norm" else "LayerNorm",
        type=BlockType.LAYER_NORM,
        params={"normalized_shape": h, "norm_type": norm_t},
        param_count=_norm_p(h, norm_t),
    ))

    # ── 4. LM head ───────────────────────────────────────────────────────────
    lm_head_p = 0 if tied else vocab * h
    # Encoder-only models without explicit LM head architecture skip the head
    has_lm_head = arch_type != "encoder" or not tied
    if has_lm_head:
        blocks.append(ArchBlock(
            id="lm_head",
            label="LM Head",
            type=BlockType.LINEAR,
            params={"in_features": h, "out_features": vocab, "bias": False},
            param_count=lm_head_p,
            notes=(
                "Weight-tied to token embeddings — no additional parameters stored."
                if tied else
                f"Separate projection h={h} → vocab={vocab:,} ({lm_head_p/1e6:.1f}M params)."
            ),
        ))

    # ── Assemble IR ──────────────────────────────────────────────────────────
    slug = (raw_config.get("_name_or_path") or "").split("/")[-1] or model_id.split("/")[-1]

    ir = ArchitectureIR(
        name=model_id,
        display_name=slug,
        family=model_type or "unknown",
        task=_infer_task(architectures, arch_type),
        architectures=architectures,
        source=SourceType.HF_CONFIG,
        source_confidence=confidence,
        blocks=blocks,
    )
    ir.compute = estimate_compute(ir)
    return ir
