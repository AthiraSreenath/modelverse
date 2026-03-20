"""
Architecture Parser - converts a raw HuggingFace config dict into an Architecture IR.

Supports: BERT, DistilBERT, RoBERTa, GPT-2, LLaMA, Mistral, Mixtral, T5, Mamba, Falcon,
          DeepSeekV2/V3, DeepSeek-VL2.
"""

from __future__ import annotations
import logging
from typing import Any

from ..models.ir import (
    ArchBlock,
    ArchitectureIR,
    AttentionType,
    ActivationType,
    BlockType,
    ComputeStats,
    SourceType,
    SourceConfidence,
)
from ..compute.estimator import estimate_compute

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Activation mapping from HF config strings
# ---------------------------------------------------------------------------
_ACT_MAP: dict[str, ActivationType] = {
    "relu": ActivationType.RELU,
    "gelu": ActivationType.GELU,
    "gelu_new": ActivationType.GELU,
    "gelu_pytorch_tanh": ActivationType.GELU,
    "silu": ActivationType.SILU,
    "swish": ActivationType.SILU,
    "swiglu": ActivationType.SWIGLU,
    "geglu": ActivationType.GEGLU,
}


def _act(config: dict, key: str = "hidden_act") -> ActivationType:
    return _ACT_MAP.get(config.get(key, "gelu"), ActivationType.GELU)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stack_param_count(children: list[ArchBlock], repeat: int) -> int:
    """Total parameters for a transformer stack = one layer's params × repeat."""
    per_layer = sum(c.param_count for c in children if c.param_count is not None)
    return per_layer * repeat


# ---------------------------------------------------------------------------
# Family-specific parsers
# ---------------------------------------------------------------------------


def _parse_bert_family(config: dict, model_id: str) -> ArchitectureIR:
    """Handles BERT, RoBERTa, DistilBERT, ALBERT variants."""
    model_type = config.get("model_type", "bert")
    is_distilbert = model_type == "distilbert"

    h = config.get("hidden_size", 768)
    num_layers = config.get("num_hidden_layers", 6 if is_distilbert else 12)
    num_heads = config.get("num_attention_heads", 12)
    intermediate = config.get("intermediate_size", 3072)
    vocab_size = config.get("vocab_size", 30522)
    max_pos = config.get("max_position_embeddings", 512)
    act = _act(config)

    # Architectures list tells us the task head
    archs = config.get("architectures", [])
    task = _infer_task(archs, config)

    # Embedding block
    emb_params = {
        "vocab_size": vocab_size,
        "hidden_size": h,
        "max_position_embeddings": max_pos,
        "position_embedding_type": "absolute",
    }
    if not is_distilbert:
        emb_params["type_vocab_size"] = config.get("type_vocab_size", 2)

    emb_block = ArchBlock(
        id="embeddings",
        label="Embeddings",
        type=BlockType.EMBEDDING,
        params=emb_params,
        param_count=_embedding_params(vocab_size, h, max_pos, emb_params.get("type_vocab_size")),
    )

    # Transformer stack
    children = _build_encoder_children(h, num_heads, intermediate, act, is_causal=False)
    encoder_block = ArchBlock(
        id="transformer" if is_distilbert else "encoder",
        label="Transformer Encoder",
        type=BlockType.TRANSFORMER_STACK,
        params={
            "hidden_size": h,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
            "intermediate_size": intermediate,
            "attention_type": "multi_head",
            "is_causal": False,
            # Graph: Post-LN (sublayer -> Add -> LN). Second residual taps LN output.
            "residual_layout": "post_ln",
        },
        repeat=num_layers,
        children=children,
        param_count=_stack_param_count(children, num_layers),
    )

    blocks: list[ArchBlock] = [emb_block, encoder_block]

    # Task head
    head_block = _build_task_head(archs, config, h)
    if head_block:
        blocks.append(head_block)

    ir = ArchitectureIR(
        name=model_id,
        display_name=config.get("name", model_id.split("/")[-1]),
        family=model_type,
        task=task,
        architectures=archs,
        source=SourceType.HF_CONFIG,
        source_confidence=SourceConfidence.EXACT,
        blocks=blocks,
    )
    ir.compute = estimate_compute(ir)
    return ir


def _parse_gpt2_family(config: dict, model_id: str) -> ArchitectureIR:
    h = config.get("n_embd", 768)
    num_layers = config.get("n_layer", 12)
    num_heads = config.get("n_head", 12)
    intermediate = config.get("n_inner") or h * 4
    vocab_size = config.get("vocab_size", 50257)
    max_pos = config.get("n_positions", 1024)
    act = _act(config, "activation_function")

    emb_block = ArchBlock(
        id="wte",
        label="Token Embeddings",
        type=BlockType.EMBEDDING,
        params={
            "vocab_size": vocab_size,
            "hidden_size": h,
            "max_position_embeddings": max_pos,
            "position_embedding_type": "absolute",
        },
        param_count=_embedding_params(vocab_size, h, max_pos),
    )

    children = _build_encoder_children(h, num_heads, intermediate, act, is_causal=True, pre_norm=True)
    decoder_block = ArchBlock(
        id="transformer",
        label="Transformer Decoder",
        type=BlockType.TRANSFORMER_STACK,
        params={
            "hidden_size": h,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
            "intermediate_size": intermediate,
            "attention_type": "multi_head",
            "is_causal": True,
            # Graph: Pre-LN (LN -> sublayer -> Add). Second residual taps first Add output.
            "residual_layout": "pre_ln",
        },
        repeat=num_layers,
        children=children,
        param_count=_stack_param_count(children, num_layers),
    )

    ln_f = ArchBlock(
        id="ln_f",
        label="Final LayerNorm",
        type=BlockType.LAYER_NORM,
        params={"normalized_shape": h, "norm_type": "layer_norm"},
        param_count=h * 2,
    )

    ir = ArchitectureIR(
        name=model_id,
        display_name=f"GPT-2 ({_param_label(config)})",
        family="gpt2",
        task="text-generation",
        architectures=config.get("architectures", []),
        source=SourceType.HF_CONFIG,
        source_confidence=SourceConfidence.EXACT,
        blocks=[emb_block, decoder_block, ln_f],
    )
    ir.compute = estimate_compute(ir)
    return ir


def _parse_t5_family(config: dict, model_id: str) -> ArchitectureIR:
    """Handles T5, Flan-T5, mT5 (encoder-decoder)."""
    model_type = config.get("model_type", "t5")
    d_model = config.get("d_model", 512)
    d_ff = config.get("d_ff", 2048)
    d_kv = config.get("d_kv", 64)
    num_heads = config.get("num_heads", 8)
    num_encoder_layers = config.get("num_layers", 6)
    num_decoder_layers = config.get("num_decoder_layers", num_encoder_layers)
    vocab_size = config.get("vocab_size", 32128)
    act = _act(config, "dense_act_fn")

    shared_emb = ArchBlock(
        id="shared",
        label="Shared Embeddings",
        type=BlockType.EMBEDDING,
        params={"vocab_size": vocab_size, "hidden_size": d_model,
                "position_embedding_type": "relative"},
        param_count=vocab_size * d_model,
    )

    # T5LayerNorm is weight-only (no bias) — equivalent to RMSNorm
    # Gated activations (gelu_new, geglu, silu) use 3 matrices; relu uses 2
    _is_gated = config.get("dense_act_fn", "relu") in ("gelu_new", "gelu_pytorch_tanh", "geglu", "silu", "swiglu")
    _ffn_matrices = 3 if _is_gated else 2
    enc_children = [
        ArchBlock(id="self_attn", label="Self-Attention (relative pos)", type=BlockType.MULTI_HEAD_ATTENTION,
                  params={"hidden_size": d_model, "num_heads": num_heads, "head_dim": d_kv,
                          "attention_type": "multi_head", "is_causal": False},
                  param_count=4 * d_model * (num_heads * d_kv)),
        _add_block("add_attn", "h = x + Attention(LN(x))  - residual 1 (T5 Pre-LN style)."),
        ArchBlock(id="layer_norm", label="RMSNorm", type=BlockType.LAYER_NORM,
                  params={"normalized_shape": d_model, "norm_type": "rms_norm"}, param_count=d_model),
        ArchBlock(id="ffn", label="FFN" if not _is_gated else "Gated FFN", type=BlockType.FEED_FORWARD,
                  params={"hidden_size": d_model, "intermediate_size": d_ff, "activation": act.value},
                  param_count=_ffn_matrices * d_model * d_ff),
        _add_block("add_ffn", "h = h + FFN(LN(h))  - residual 2 (T5 Pre-LN style)."),
        ArchBlock(id="ffn_norm", label="RMSNorm", type=BlockType.LAYER_NORM,
                  params={"normalized_shape": d_model, "norm_type": "rms_norm"}, param_count=d_model),
    ]
    encoder = ArchBlock(id="encoder", label="T5 Encoder", type=BlockType.TRANSFORMER_STACK,
                        params={"hidden_size": d_model, "num_hidden_layers": num_encoder_layers,
                                "num_attention_heads": num_heads, "is_causal": False,
                                "residual_layout": "post_ln"},
                        repeat=num_encoder_layers, children=enc_children,
                        param_count=_stack_param_count(enc_children, num_encoder_layers))

    dec_children = [
        ArchBlock(id="self_attn", label="Causal Self-Attention", type=BlockType.MULTI_HEAD_ATTENTION,
                  params={"hidden_size": d_model, "num_heads": num_heads, "head_dim": d_kv,
                          "attention_type": "multi_head", "is_causal": True},
                  param_count=4 * d_model * (num_heads * d_kv)),
        _add_block("add_self_attn", "h = x + SelfAttention(LN(x))  - residual 1 (T5 decoder)."),
        ArchBlock(id="cross_attn", label="Cross-Attention", type=BlockType.MULTI_HEAD_ATTENTION,
                  params={"hidden_size": d_model, "num_heads": num_heads, "head_dim": d_kv,
                          "attention_type": "multi_head", "is_causal": False},
                  param_count=4 * d_model * (num_heads * d_kv)),
        _add_block("add_cross_attn", "h = h + CrossAttention(LN(h), encoder_output)  - residual 2 (T5 decoder)."),
        ArchBlock(id="layer_norm", label="RMSNorm", type=BlockType.LAYER_NORM,
                  params={"normalized_shape": d_model, "norm_type": "rms_norm"}, param_count=d_model),
        ArchBlock(id="ffn", label="FFN" if not _is_gated else "Gated FFN", type=BlockType.FEED_FORWARD,
                  params={"hidden_size": d_model, "intermediate_size": d_ff, "activation": act.value},
                  param_count=_ffn_matrices * d_model * d_ff),
        _add_block("add_ffn", "h = h + FFN(LN(h))  - residual 3 (T5 decoder)."),
        ArchBlock(id="ffn_norm", label="RMSNorm", type=BlockType.LAYER_NORM,
                  params={"normalized_shape": d_model, "norm_type": "rms_norm"}, param_count=d_model),
    ]
    decoder = ArchBlock(id="decoder", label="T5 Decoder", type=BlockType.TRANSFORMER_STACK,
                        params={"hidden_size": d_model, "num_hidden_layers": num_decoder_layers,
                                "num_attention_heads": num_heads, "is_causal": True,
                                # Three adds per layer; FFN residual must tap cross-attn Add, not pre-FFN LN.
                                "residual_layout": "t5_decoder"},
                        repeat=num_decoder_layers, children=dec_children,
                        param_count=_stack_param_count(dec_children, num_decoder_layers))

    lm_head = ArchBlock(id="lm_head", label="LM Head", type=BlockType.LINEAR,
                        params={"in_features": d_model, "out_features": vocab_size, "bias": False},
                        param_count=d_model * vocab_size)

    size_variants = {512: "Small", 768: "Base", 1024: "Large", 2048: "XL", 4096: "XXL"}
    size_label = size_variants.get(d_model, f"d{d_model}")
    flan = "Flan-" if "flan" in model_id.lower() else ""

    ir = ArchitectureIR(
        name=model_id,
        display_name=f"{flan}T5 {size_label}",
        family=model_type,
        task="text2text-generation",
        architectures=config.get("architectures", []),
        source=SourceType.HF_CONFIG,
        source_confidence=SourceConfidence.EXACT,
        blocks=[shared_emb, encoder, decoder, lm_head],
    )
    ir.compute = estimate_compute(ir)
    return ir


def _parse_mamba_family(config: dict, model_id: str) -> ArchitectureIR:
    """Handles Mamba (state space model - no attention)."""
    d_model = config.get("d_model", 2560)
    n_layer = config.get("n_layer", 64)
    vocab_size = config.get("vocab_size", 50280)
    d_state = config.get("d_state", 16)
    d_conv = config.get("d_conv", 4)
    expand = config.get("expand", 2)
    d_inner = int(expand * d_model)

    emb = ArchBlock(
        id="embeddings", label="Token Embeddings", type=BlockType.EMBEDDING,
        params={"vocab_size": vocab_size, "hidden_size": d_model,
                "max_position_embeddings": None, "position_embedding_type": "none"},
        param_count=vocab_size * d_model,
    )

    ssm_children = [
        ArchBlock(id="norm", label="RMSNorm", type=BlockType.LAYER_NORM,
                  params={"normalized_shape": d_model, "norm_type": "rms_norm"}, param_count=d_model),
        ArchBlock(id="in_proj", label="Input Projection", type=BlockType.LINEAR,
                  params={"in_features": d_model, "out_features": d_inner * 2, "bias": False},
                  param_count=d_model * d_inner * 2),
        ArchBlock(id="conv1d", label="Conv1D (local context)", type=BlockType.CONV1D,
                  params={"in_channels": d_inner, "kernel_size": d_conv},
                  param_count=d_inner * d_conv),
        ArchBlock(id="ssm", label="Selective SSM", type=BlockType.SSM,
                  params={"hidden_size": d_model, "state_size": d_state, "conv_kernel": d_conv, "expand": expand},
                  param_count=d_inner * d_state * 2),
        ArchBlock(id="out_proj", label="Output Projection", type=BlockType.LINEAR,
                  params={"in_features": d_inner, "out_features": d_model, "bias": False},
                  param_count=d_inner * d_model),
    ]

    mamba_stack = ArchBlock(
        id="layers", label="Mamba Blocks", type=BlockType.TRANSFORMER_STACK,
        params={"hidden_size": d_model, "num_hidden_layers": n_layer,
                "d_state": d_state, "d_conv": d_conv, "expand": expand},
        repeat=n_layer, children=ssm_children,
        param_count=_stack_param_count(ssm_children, n_layer),
    )

    norm_f = ArchBlock(id="norm_f", label="Final RMSNorm", type=BlockType.LAYER_NORM,
                       params={"normalized_shape": d_model, "norm_type": "rms_norm"}, param_count=d_model)
    lm_head = ArchBlock(id="lm_head", label="LM Head", type=BlockType.LINEAR,
                        params={"in_features": d_model, "out_features": vocab_size, "bias": False},
                        param_count=d_model * vocab_size)

    ir = ArchitectureIR(
        name=model_id,
        display_name=f"Mamba ({_billion_label(config)})" if d_model > 1000 else f"Mamba (d={d_model})",
        family="mamba",
        task="text-generation",
        architectures=config.get("architectures", []),
        source=SourceType.HF_CONFIG,
        source_confidence=SourceConfidence.EXACT,
        blocks=[emb, mamba_stack, norm_f, lm_head],
    )
    ir.compute = estimate_compute(ir)
    return ir


def _parse_llama_family(config: dict, model_id: str) -> ArchitectureIR:
    """Handles LLaMA 2/3, Mistral, Mixtral."""
    model_type = config.get("model_type", "llama")
    h = config.get("hidden_size", 4096)
    num_layers = config.get("num_hidden_layers", 32)
    num_heads = config.get("num_attention_heads", 32)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    intermediate = config.get("intermediate_size", 14336)
    vocab_size = config.get("vocab_size", 32000)
    max_pos = config.get("max_position_embeddings", 4096)
    act = _act(config)
    is_moe = model_type == "mixtral"

    attn_type = AttentionType.GQA if num_kv_heads != num_heads else AttentionType.MHA

    emb_block = ArchBlock(
        id="embed_tokens",
        label="Token Embeddings",
        type=BlockType.EMBEDDING,
        params={
            "vocab_size": vocab_size,
            "hidden_size": h,
            "max_position_embeddings": max_pos,
            "position_embedding_type": "rope",
        },
        param_count=vocab_size * h,
    )

    # Build children
    input_norm = ArchBlock(
        id="input_layernorm",
        label="RMSNorm",
        type=BlockType.LAYER_NORM,
        params={"normalized_shape": h, "norm_type": "rms_norm"},
        param_count=h,
    )
    attn_params: dict[str, Any] = {
        "hidden_size": h,
        "num_heads": num_heads,
        "head_dim": h // num_heads,
        "attention_type": attn_type.value,
        "is_causal": True,
    }
    if num_kv_heads != num_heads:
        attn_params["num_kv_heads"] = num_kv_heads
    if config.get("sliding_window"):
        attn_params["sliding_window"] = config["sliding_window"]

    head_dim = h // num_heads
    q_params = h * num_heads * head_dim
    k_params = h * num_kv_heads * head_dim
    v_params = h * num_kv_heads * head_dim
    o_params = num_heads * head_dim * h
    attn_total = q_params + k_params + v_params + o_params

    if attn_type == AttentionType.GQA:
        attn_notes = (
            f"Grouped-Query Attention (GQA): {num_heads} query heads, {num_kv_heads} KV heads.\n"
            f"Q projection: {h:,} -> {num_heads} x {head_dim} = {q_params / 1e6:.2f}M  (all {num_heads} query heads)\n"
            f"K projection: {h:,} -> {num_kv_heads} x {head_dim} = {k_params / 1e6:.2f}M  ({num_kv_heads} KV heads only)\n"
            f"V projection: {h:,} -> {num_kv_heads} x {head_dim} = {v_params / 1e6:.2f}M  ({num_kv_heads} KV heads only)\n"
            f"O projection: {num_heads} x {head_dim} -> {h:,} = {o_params / 1e6:.2f}M  (all {num_heads} query heads)\n"
            f"Total: {attn_total / 1e6:.2f}M  "
            f"({num_heads - num_kv_heads} fewer KV head pairs saves "
            f"{(num_heads - num_kv_heads) * 2 * h * head_dim / 1e6:.2f}M vs full MHA)"
        )
    else:
        attn_notes = (
            f"Multi-Head Attention: {num_heads} heads, head_dim={head_dim}.\n"
            f"Q+K+V projections: 3 x {h:,} x {num_heads * head_dim:,} = {(q_params + k_params + v_params) / 1e6:.2f}M\n"
            f"O projection: {num_heads * head_dim:,} x {h:,} = {o_params / 1e6:.2f}M\n"
            f"Total: {attn_total / 1e6:.2f}M"
        )

    attn_block = ArchBlock(
        id="self_attn",
        label=f"{'Sliding Window ' if config.get('sliding_window') else ''}{'GQA' if attn_type == AttentionType.GQA else 'MHA'}",
        type=BlockType.MULTI_HEAD_ATTENTION,
        params=attn_params,
        param_count=_gqa_params(h, num_heads, num_kv_heads),
        notes=attn_notes,
    )
    post_norm = ArchBlock(
        id="post_attention_layernorm",
        label="RMSNorm",
        type=BlockType.LAYER_NORM,
        params={"normalized_shape": h, "norm_type": "rms_norm"},
        param_count=h,
    )

    if is_moe:
        num_experts = config.get("num_local_experts", 8)
        num_experts_per_tok = config.get("num_experts_per_tok", 2)
        ffn_block = ArchBlock(
            id="mlp",
            label=f"MoE FFN ({num_experts} experts, top-{num_experts_per_tok})",
            type=BlockType.MOE_FEED_FORWARD,
            params={
                "hidden_size": h,
                "intermediate_size": intermediate,
                "num_experts": num_experts,
                "num_experts_per_tok": num_experts_per_tok,
                "activation": act.value,
            },
            param_count=num_experts * 3 * h * intermediate,
        )
    else:
        ffn_block = ArchBlock(
            id="mlp",
            label="SwiGLU FFN" if act in (ActivationType.SWIGLU, ActivationType.SILU) else "FFN",
            type=BlockType.FEED_FORWARD,
            params={
                "hidden_size": h,
                "intermediate_size": intermediate,
                "activation": act.value,
            },
            param_count=3 * h * intermediate,
        )

    children = [
        input_norm,
        attn_block,
        _add_block("add_attn", "h = x + Attention(RMSNorm(x))  - residual 1 (Pre-LN: Add after attention, before next norm)."),
        post_norm,
        ffn_block,
        _add_block("add_ffn", "h = h + FFN(RMSNorm(h))  - residual 2 (Pre-LN: Add after FFN)."),
    ]

    decoder_block = ArchBlock(
        id="layers",
        label="Transformer Decoder",
        type=BlockType.TRANSFORMER_STACK,
        params={
            "hidden_size": h,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "intermediate_size": intermediate,
            "attention_type": attn_type.value,
            "is_causal": True,
            # Graph: Pre-LN (RMSNorm -> sublayer -> Add). Second residual taps first Add output.
            "residual_layout": "pre_ln",
        },
        repeat=num_layers,
        children=children,
        param_count=_stack_param_count(children, num_layers),
    )

    norm = ArchBlock(
        id="norm",
        label="Final RMSNorm",
        type=BlockType.LAYER_NORM,
        params={"normalized_shape": h, "norm_type": "rms_norm"},
        param_count=h,
    )
    lm_head = ArchBlock(
        id="lm_head",
        label="LM Head",
        type=BlockType.LINEAR,
        params={"in_features": h, "out_features": vocab_size, "bias": False},
        param_count=h * vocab_size,
    )

    family_map = {"mistral": "mistral", "mixtral": "mixtral"}
    display_name_map = {
        "mistral": "Mistral",
        "mixtral": "Mixtral",
        "llama": "LLaMA",
    }

    ir = ArchitectureIR(
        name=model_id,
        display_name=f"{display_name_map.get(model_type, 'LLaMA')} ({_billion_label(config)})",
        family=family_map.get(model_type, "llama"),
        task="text-generation",
        architectures=config.get("architectures", []),
        source=SourceType.HF_CONFIG,
        source_confidence=SourceConfidence.EXACT,
        blocks=[emb_block, decoder_block, norm, lm_head],
    )
    ir.compute = estimate_compute(ir)
    return ir


# ---------------------------------------------------------------------------
# DeepSeek V2 / V3 / VL-V2
# ---------------------------------------------------------------------------


def _deepseek_lm_param_count(lm_cfg: dict) -> tuple[int, int, int, int, int]:
    """
    Return (total_layer_params, final_norm_params, emb_params, lm_head_params, h)
    for a DeepSeekV2/V3-style config (with optional MoE and MLA).
    """
    h = lm_cfg.get("hidden_size", 2048)
    num_layers = lm_cfg.get("num_hidden_layers", 28)
    num_heads = lm_cfg.get("num_attention_heads", 16)
    num_kv_heads = lm_cfg.get("num_key_value_heads", num_heads)
    vocab_size = lm_cfg.get("vocab_size", 102400)
    intermediate = lm_cfg.get("intermediate_size", h * 4)
    moe_intermediate = lm_cfg.get("moe_intermediate_size", 0)
    n_routed = lm_cfg.get("n_routed_experts", 0)
    n_shared = lm_cfg.get("n_shared_experts", 0)
    first_k_dense = lm_cfg.get("first_k_dense_replace", 0)
    use_mla = lm_cfg.get("use_mla", False)
    kv_lora_rank = lm_cfg.get("kv_lora_rank") or 0
    q_lora_rank = lm_cfg.get("q_lora_rank") or 0
    qk_nope_head_dim = lm_cfg.get("qk_nope_head_dim") or (h // num_heads)
    qk_rope_head_dim = lm_cfg.get("qk_rope_head_dim") or 0
    v_head_dim = lm_cfg.get("v_head_dim") or (h // num_heads)
    head_dim = h // num_heads

    # Attention params per layer
    if use_mla and kv_lora_rank > 0:
        # Multi-head Latent Attention: compressed KV representation
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        if q_lora_rank:
            q_params = h * q_lora_rank + q_lora_rank * num_heads * qk_head_dim
        else:
            q_params = h * num_heads * qk_head_dim
        kv_params = h * kv_lora_rank + kv_lora_rank * num_kv_heads * (qk_nope_head_dim + v_head_dim)
        o_params = num_heads * v_head_dim * h
        attn_per_layer = q_params + kv_params + o_params
    else:
        attn_per_layer = _gqa_params(h, num_heads, num_kv_heads)

    # FFN: dense (SwiGLU = 3 matrices) vs MoE
    n_dense = min(first_k_dense, num_layers)
    n_moe = num_layers - n_dense
    dense_ffn_per_layer = 3 * h * intermediate
    if n_moe > 0 and n_routed > 0 and moe_intermediate > 0:
        router = h * n_routed
        routed = n_routed * 3 * h * moe_intermediate
        shared = n_shared * 3 * h * moe_intermediate
        moe_ffn_per_layer = router + routed + shared
    else:
        moe_ffn_per_layer = dense_ffn_per_layer  # fallback: treat as dense

    # 2× RMSNorm per layer (pre-attn + pre-ffn), each with `h` weights
    norm_per_layer = 2 * h

    total_layer = (
        num_layers * (attn_per_layer + norm_per_layer)
        + n_dense * dense_ffn_per_layer
        + n_moe * moe_ffn_per_layer
    )
    final_norm = h
    emb = vocab_size * h
    lm_head = h * vocab_size

    return total_layer, final_norm, emb, lm_head, h


def _parse_deepseek_v2(config: dict, model_id: str) -> ArchitectureIR:
    """Handles standalone deepseek_v2 / deepseek_v3 language models."""
    model_type = config.get("model_type", "deepseek_v2")
    h = config.get("hidden_size", 2048)
    num_layers = config.get("num_hidden_layers", 28)
    num_heads = config.get("num_attention_heads", 16)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    vocab_size = config.get("vocab_size", 102400)
    max_pos = config.get("max_position_embeddings", 4096)
    n_routed = config.get("n_routed_experts", 0)
    n_shared = config.get("n_shared_experts", 0)
    moe_intermediate = config.get("moe_intermediate_size", 0)
    n_experts_per_tok = config.get("num_experts_per_tok", 0)
    first_k_dense = config.get("first_k_dense_replace", 0)
    use_mla = config.get("use_mla", False)

    total_layer, final_norm_p, emb_p, lm_head_p, _ = _deepseek_lm_param_count(config)

    is_moe = n_routed > 0 and moe_intermediate > 0
    n_dense = min(first_k_dense, num_layers)
    n_moe = num_layers - n_dense

    emb_block = ArchBlock(
        id="embed_tokens", label="Token Embeddings", type=BlockType.EMBEDDING,
        params={"vocab_size": vocab_size, "hidden_size": h, "max_position_embeddings": max_pos,
                "position_embedding_type": "rope"},
        param_count=emb_p,
    )

    stack_notes = None
    if is_moe:
        stack_notes = (
            f"First {n_dense} layer(s) use dense SwiGLU FFN (intermediate_size={config.get('intermediate_size')}).\n"
            f"Remaining {n_moe} layers use MoE: {n_routed} routed experts + {n_shared} shared experts "
            f"(top-{n_experts_per_tok} active per token, moe_intermediate_size={moe_intermediate}).\n"
            f"MLA: {'enabled' if use_mla else 'disabled'}."
        )
    stack_label = f"MoE Transformer Decoder ({n_routed} experts)" if is_moe else "Transformer Decoder"

    decoder_block = ArchBlock(
        id="layers", label=stack_label, type=BlockType.TRANSFORMER_STACK,
        params={
            "hidden_size": h, "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads, "num_key_value_heads": num_kv_heads,
            "use_mla": use_mla, "is_causal": True,
            **({"n_routed_experts": n_routed, "n_shared_experts": n_shared,
                "moe_intermediate_size": moe_intermediate,
                "num_experts_per_tok": n_experts_per_tok,
                "first_k_dense_replace": first_k_dense} if is_moe else {}),
        },
        # repeat=1 here: param_count is the total for all layers (not per-layer).
        # num_hidden_layers in params drives KV-cache calculation.
        repeat=1, children=[],
        param_count=total_layer + final_norm_p,
        notes=stack_notes,
    )

    lm_head_block = ArchBlock(
        id="lm_head", label="LM Head", type=BlockType.LINEAR,
        params={"in_features": h, "out_features": vocab_size, "bias": False},
        param_count=lm_head_p,
    )

    version_label = "V3" if "v3" in model_type else "V2"
    ir = ArchitectureIR(
        name=model_id,
        display_name=f"DeepSeek-{version_label} ({_approx_billion(emb_p + total_layer + final_norm_p + lm_head_p)})",
        family=model_type,
        task="text-generation",
        architectures=config.get("architectures", []),
        source=SourceType.HF_CONFIG, source_confidence=SourceConfidence.EXACT,
        blocks=[emb_block, decoder_block, lm_head_block],
    )
    ir.compute = estimate_compute(ir)
    return ir


def _vit_params(layers: int, width: int, mlp_dim: int) -> int:
    """Approximate ViT transformer block parameter count (weights only, no biases)."""
    attn = 4 * width * width          # Q, K, V, O projections
    ffn = 2 * width * mlp_dim         # FC1 + FC2
    norms = 4 * width                  # 2 × (weight + bias) per LayerNorm
    return layers * (attn + ffn + norms)


def _moe_active_params(lm_cfg: dict) -> int:
    """
    Active parameter count per token for a DeepSeekV2-style MoE LM.
    Only the top-k routed experts + all shared experts fire per token;
    the remaining (n_routed - k) experts are stored but never compute.
    """
    h = lm_cfg.get("hidden_size", 1280)
    num_layers = lm_cfg.get("num_hidden_layers", 12)
    num_heads = lm_cfg.get("num_attention_heads", 10)
    num_kv_heads = lm_cfg.get("num_key_value_heads", num_heads)
    intermediate = lm_cfg.get("intermediate_size", h * 4)
    moe_intermediate = lm_cfg.get("moe_intermediate_size", 0)
    n_routed = lm_cfg.get("n_routed_experts", 0)
    n_shared = lm_cfg.get("n_shared_experts", 0)
    n_active = lm_cfg.get("num_experts_per_tok", 0)
    first_k_dense = lm_cfg.get("first_k_dense_replace", 0)
    vocab_size = lm_cfg.get("vocab_size", 32000)

    n_dense = min(first_k_dense, num_layers)
    n_moe = num_layers - n_dense

    attn_per_layer = _gqa_params(h, num_heads, num_kv_heads)
    norm_per_layer = 2 * h
    dense_ffn = 3 * h * intermediate

    if n_routed and moe_intermediate and n_active:
        # Only n_active routed experts run per token; shared always run
        active_moe = (n_active * 3 * h * moe_intermediate    # top-k routed
                      + n_shared * 3 * h * moe_intermediate  # shared (always on)
                      + h * n_routed)                         # router
    else:
        active_moe = dense_ffn

    active_layers = (num_layers * (attn_per_layer + norm_per_layer)
                     + n_dense * dense_ffn
                     + n_moe * active_moe)
    return active_layers + h + vocab_size * h + h * vocab_size  # +norms+emb+lmhead


def _parse_deepseek_vl_v2(config: dict, model_id: str) -> ArchitectureIR:
    """
    Handles deepseek_vl_v2: two PARALLEL vision encoders → fusion → projector → LM.

    Vision path (parallel, not sequential):
        CLIP ViT-L/14  ──┐
                         ├─ concat (1024+1024=2048) ─→ projector ─→ LM
        SAM  ViT-B     ──┘
    """
    lm_cfg = config.get("language_config", config)
    vis_cfg = config.get("vision_config", {})
    proj_cfg = config.get("projector_config", {})
    vis_width_cfg = vis_cfg.get("width", {})
    mlp_ratio: float = vis_cfg.get("mlp_ratio", 4.0)

    h = lm_cfg.get("hidden_size", 1280)
    num_layers = lm_cfg.get("num_hidden_layers", 12)
    num_heads = lm_cfg.get("num_attention_heads", 10)
    num_kv_heads = lm_cfg.get("num_key_value_heads", num_heads)
    vocab_size = lm_cfg.get("vocab_size", 129280)
    max_pos = lm_cfg.get("max_position_embeddings", 8192)
    n_routed = lm_cfg.get("n_routed_experts", 0)
    n_shared = lm_cfg.get("n_shared_experts", 0)
    moe_intermediate = lm_cfg.get("moe_intermediate_size", 0)
    n_experts_per_tok = lm_cfg.get("num_experts_per_tok", 0)
    first_k_dense = lm_cfg.get("first_k_dense_replace", 0)
    use_mla = lm_cfg.get("use_mla", False)

    total_layer, final_norm_p, emb_p, lm_head_p, _ = _deepseek_lm_param_count(lm_cfg)

    # ── Vision encoders (PARALLEL paths — both process the image independently) ──
    blocks: list[ArchBlock] = []

    clip_cfg = vis_width_cfg.get("clip-l-14-224", {})
    clip_p = 0
    clip_out_dim = 0
    if clip_cfg:
        cw = clip_cfg.get("width", 1024)
        cl = clip_cfg.get("layers", 24)
        cp = clip_cfg.get("patch_size", 14)
        ci = clip_cfg.get("image_size", 224)
        clip_mlp = int(cw * mlp_ratio)
        patch_embed = 3 * cp * cp * cw
        pos_embed = ((ci // cp) ** 2 + 1) * cw
        transformer = _vit_params(cl, cw, clip_mlp)
        final_ln = 2 * cw
        clip_p = patch_embed + pos_embed + transformer + final_ln
        clip_out_dim = cw
        blocks.append(ArchBlock(
            id="clip_l_vision",
            label="CLIP ViT-L/14",
            type=BlockType.TRANSFORMER_STACK,
            params={"layers": cl, "width": cw, "heads": clip_cfg.get("heads", 16),
                    "patch_size": cp, "image_size": ci, "mlp_dim": clip_mlp,
                    "output_dim": cw},
            param_count=clip_p,
            notes=(
                f"Vision encoder — runs in parallel with SAM ViT-B.\n"
                f"Processes the full global-view image ({ci}×{ci}) through {cl} ViT-L transformer layers.\n"
                f"Output: {(ci//cp)**2} patch tokens × {cw} dims.\n\n"
                f"patch_size={cp}, width={cw}, heads={clip_cfg.get('heads',16)}, "
                f"mlp_ratio={mlp_ratio:.4f}\n"
                f"Patch embedding: 3×{cp}×{cp}×{cw} = {patch_embed:,} params\n"
                f"Positional embedding: {(ci//cp)**2+1} pos × {cw} = {pos_embed:,} params\n"
                f"Transformer: {cl} layers × (4×{cw}² attn + 2×{cw}×{clip_mlp} FFN) = {transformer:,} params"
            ),
        ))

    sam_cfg = vis_width_cfg.get("sam_vit_b", {})
    sam_p = 0
    sam_out_dim = 0
    if sam_cfg:
        sw = sam_cfg.get("width", 768)
        sl = sam_cfg.get("layers", 12)
        sam_mlp = sw * 4
        patch_embed = 3 * 16 * 16 * sw
        transformer = _vit_params(sl, sw, sam_mlp)
        downsample = sam_cfg.get("downsample_channels", [])
        neck_p = 0
        prev = sw
        for dc in downsample:
            neck_p += prev * dc + dc
            prev = dc
        sam_p = patch_embed + transformer + neck_p
        sam_out_dim = downsample[-1] if downsample else sw
        blocks.append(ArchBlock(
            id="sam_vit_b",
            label="SAM ViT-B",
            type=BlockType.TRANSFORMER_STACK,
            params={"layers": sl, "width": sw, "heads": sam_cfg.get("heads", 12),
                    "downsample_channels": downsample, "output_dim": sam_out_dim},
            param_count=sam_p,
            notes=(
                f"Vision encoder — runs in parallel with CLIP ViT-L/14.\n"
                f"Processes tiled image regions at higher resolution through {sl} ViT-B layers.\n"
                f"Neck reduces channel dim: {sw} → {downsample} → {sam_out_dim}-d output.\n\n"
                f"Patch embedding (16×16): 3×16×16×{sw} = {patch_embed:,} params\n"
                f"Transformer: {sl} layers × (4×{sw}² attn + 2×{sw}×{sam_mlp} FFN) = {transformer:,} params\n"
                f"Neck (channel reduction conv layers): {neck_p:,} params"
            ),
        ))

    # ── Fusion: merge both vision streams ─────────────────────────────────────
    proj_in = proj_cfg.get("input_dim", 2048)
    if clip_p and sam_p:
        blocks.append(ArchBlock(
            id="vision_fusion",
            label=f"Fused Vision Representation ({proj_in}-d)",
            type=BlockType.ADD,
            params={"output_dim": proj_in,
                    "clip_branch_width": clip_out_dim,
                    "sam_branch_width": sam_out_dim},
            param_count=0,
            notes=(
                f"The two parallel vision branches are merged into a {proj_in}-d fused representation.\n\n"
                f"The config specifies projector input_dim={proj_in} and the two vision branches have "
                f"widths {clip_out_dim} (CLIP) and {sam_out_dim} (SAM after neck). "
                f"The exact fusion mechanism (concat, add, or learned) is not fully specified in the "
                f"config — the {proj_in}-d output is what the config guarantees.\n\n"
                "No learned parameters in this fusion step itself."
            ),
        ))

    # ── Projector ──────────────────────────────────────────────────────────────
    proj_out = proj_cfg.get("n_embed", h)
    proj_type = proj_cfg.get("projector_type", "linear")
    proj_p = proj_in * proj_out + proj_out
    blocks.append(ArchBlock(
        id="projector",
        label=f"Vision Projector ({proj_type}: {proj_in} → {proj_out})",
        type=BlockType.LINEAR,
        params={"in_features": proj_in, "out_features": proj_out, "bias": True},
        param_count=proj_p,
        notes=(
            f"Maps fused vision features ({proj_in} dims) into the LM token space ({proj_out} dims).\n"
            f"Type: {proj_type}  |  {proj_in:,} × {proj_out:,} + {proj_out:,} bias = {proj_p:,} params."
        ),
    ))

    # ── LM blocks ──────────────────────────────────────────────────────────────
    n_dense = min(first_k_dense, num_layers)
    n_moe = num_layers - n_dense
    is_moe = n_routed > 0 and moe_intermediate > 0

    # Active params per token (only top-k experts fire per token in MoE)
    active_lm = _moe_active_params(lm_cfg) if is_moe else (total_layer + final_norm_p + emb_p + lm_head_p)
    total_stored_lm = total_layer + final_norm_p + emb_p + lm_head_p

    blocks.append(ArchBlock(
        id="embed_tokens", label="Token Embeddings", type=BlockType.EMBEDDING,
        params={"vocab_size": vocab_size, "hidden_size": h,
                "max_position_embeddings": max_pos, "position_embedding_type": "rope"},
        param_count=emb_p,
    ))
    blocks.append(ArchBlock(
        id="layers",
        label=(f"LM Decoder — MoE ({n_routed} routed + {n_shared} shared, top-{n_experts_per_tok} active/token · stored params)"
               if is_moe else "LM Decoder"),
        type=BlockType.TRANSFORMER_STACK,
        params={
            "hidden_size": h, "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads, "num_key_value_heads": num_kv_heads,
            "use_mla": use_mla, "is_causal": True,
            **({"n_routed_experts": n_routed, "n_shared_experts": n_shared,
                "moe_intermediate_size": moe_intermediate,
                "num_experts_per_tok": n_experts_per_tok,
                "first_k_dense_replace": first_k_dense} if is_moe else {}),
        },
        repeat=1, children=[],
        param_count=total_layer + final_norm_p,
        notes=(
            f"Params shown = stored (all experts saved to disk, not active compute).\n"
            f"  • Stored: {(total_stored_lm)/1e9:.2f}B — {n_routed} routed + {n_shared} shared experts all on disk\n"
            f"  • Active per token: {active_lm/1e9:.2f}B — top-{n_experts_per_tok} of {n_routed} routed "
            f"fire; remaining {n_routed - n_experts_per_tok} are idle each step\n\n"
            f"Layer breakdown ({num_layers} layers):\n"
            f"  • Layer{'s' if n_dense > 1 else ''} 0–{max(n_dense-1,0)}: "
            f"dense SwiGLU FFN (intermediate_size={lm_cfg.get('intermediate_size')})\n"
            f"  • Layers {n_dense}–{num_layers-1}: MoE FFN — "
            f"{n_routed} routed experts, {n_shared} shared (always-on), "
            f"moe_intermediate_size={moe_intermediate}\n\n"
            f"Attention: {'MLA (Multi-head Latent Attention)' if use_mla else 'standard MHA'}, "
            f"{num_heads} heads, hidden={h}, head_dim={h // num_heads}."
        ) if is_moe else None,
    ))
    _lm_head_tied = not lm_cfg.get("lm_head", True)  # lm_head:true → separate weights
    blocks.append(ArchBlock(
        id="lm_head", label="LM Head", type=BlockType.LINEAR,
        params={"in_features": h, "out_features": vocab_size, "bias": False},
        param_count=0 if _lm_head_tied else lm_head_p,
        notes=(
            f"Separate (not weight-tied) linear projection: h={h} → vocab={vocab_size:,}.\n"
            f"config lm_head=true means this head has its own {lm_head_p/1e6:.1f}M parameters, "
            f"distinct from the token embedding table.\n"
            f"Both are counted separately in the total above."
        ) if not _lm_head_tied else (
            "Weight-tied to the token embedding table — no additional parameters stored."
        ),
    ))

    # ── Display name: prefer _name_or_path model slug over generic family ──────
    _slug = (config.get("_name_or_path") or "").split("/")[-1] or "DeepSeek-VL2"
    total_all = clip_p + sam_p + proj_p + emb_p + total_layer + final_norm_p + lm_head_p
    ir = ArchitectureIR(
        name=model_id,
        display_name=f"{_slug} ({_approx_billion(total_all)} stored · DeepSeek-VL2 backbone)",
        family="deepseek_vl_v2",
        task="image-text-to-text",
        architectures=config.get("architectures", []),
        source=SourceType.HF_CONFIG, source_confidence=SourceConfidence.EXACT,
        blocks=blocks,
    )
    ir.compute = estimate_compute(ir)
    return ir


def _approx_billion(n: int) -> str:
    if n >= 1e9:
        return f"~{n / 1e9:.1f}B"
    return f"~{n / 1e6:.0f}M"


# ---------------------------------------------------------------------------
# spaCy NLP pipeline parser
# ---------------------------------------------------------------------------

def _spacy_tok2vec_params(width: int, depth: int, embed_size: int,
                           window: int, maxout: int, subword: bool) -> int:
    """Estimate HashEmbedCNN parameter count."""
    # Hash embedding tables: 4 tables for token features + 5 if subword
    n_tables = 9 if subword else 4
    embed_p = n_tables * embed_size * width
    # CNN layers: each layer is (2*window+1) conv, width in/out, maxout pieces
    cnn_p = depth * (2 * window + 1) * width * width * maxout
    return embed_p + cnn_p


_SPACY_COMPONENT_META: dict[str, tuple[str, BlockType, str]] = {
    # name → (label, block_type, short description)
    "tok2vec":        ("Tok2Vec (HashEmbedCNN)",  BlockType.CONV1D,        "Convolutional contextual encoder. Produces shared token features consumed by all neural heads."),
    "tagger":         ("POS Tagger",              BlockType.LINEAR,        "Linear classifier predicting Penn Treebank part-of-speech tags per token. Reads shared Tok2Vec features."),
    "morphologizer":  ("Morphologizer",           BlockType.LINEAR,        "Predicts morphological features (gender, tense, case, …) per token. Reads shared Tok2Vec features."),
    "parser":         ("Dependency Parser",       BlockType.UNKNOWN,       "Arc-eager transition-based dependency parser. Reads shared Tok2Vec features."),
    "senter":         ("Sentence Segmenter",      BlockType.LINEAR,        "Binary classifier (I/S) for sentence boundary detection. Reads shared Tok2Vec features."),
    "sentencizer":    ("Sentence Segmenter",      BlockType.RULE_BASED,    "Rule-based sentence boundary detection (punctuation patterns). No learned parameters."),
    "ner":            ("Named Entity Recognizer", BlockType.UNKNOWN,       "Transition-based NER using BILUO tagging scheme. Reads shared Tok2Vec features — does NOT depend on tagger or parser output."),
    "entity_linker":  ("Entity Linker",           BlockType.UNKNOWN,       "Links named entities to knowledge base entries."),
    "entity_ruler":   ("Entity Ruler",            BlockType.RULE_BASED,    "Rule-based named entity recognizer using patterns/regex. No learned parameters."),
    "textcat":        ("Text Classifier",         BlockType.LINEAR,        "Document-level text categorization head."),
    "textcat_multilabel": ("Multi-label Text Classifier", BlockType.LINEAR, "Multi-label document classification head."),
    "attribute_ruler": ("Attribute Ruler",        BlockType.RULE_BASED,    "Rule-based token attribute assignment via pattern matching. No learned parameters."),
    "lemmatizer":     ("Lemmatizer",              BlockType.RULE_BASED,    "Lookup/rule-based lemmatizer. Uses language-specific tables, not learned weights."),
    "span_ruler":     ("Span Ruler",              BlockType.RULE_BASED,    "Rule-based span/entity detection. No learned parameters."),
    "spancat":        ("Span Categorizer",        BlockType.UNKNOWN,       "Span-level categorization using a suggester + classifier."),
    "coref":          ("Coreference Resolver",    BlockType.UNKNOWN,       "Neural coreference resolution."),
}

# Components that produce predictions from Tok2Vec features (run in parallel conceptually)
_NEURAL_HEADS = {"tagger", "morphologizer", "parser", "senter", "ner", "textcat",
                  "textcat_multilabel", "spancat", "coref", "entity_linker"}

# Components that are rule-based (zero learned params, run after neural heads)
_RULE_BASED = {"attribute_ruler", "lemmatizer", "sentencizer", "entity_ruler", "span_ruler"}


def _spacy_component_block(
    name: str,
    n_labels: int,
    tok2vec_width: int,
    tok2vec_depth: int,
    tok2vec_embed_size: int,
    tok2vec_window: int,
    tok2vec_maxout: int,
    tok2vec_subword: bool,
    label_list: list[str],
    perf: dict,
) -> ArchBlock:
    label, btype, desc = _SPACY_COMPONENT_META.get(
        name, (name.replace("_", " ").title(), BlockType.UNKNOWN, "")
    )

    param_count: int | None = None
    notes_parts = [desc]

    if name == "tok2vec":
        param_count = _spacy_tok2vec_params(
            tok2vec_width, tok2vec_depth, tok2vec_embed_size,
            tok2vec_window, tok2vec_maxout, tok2vec_subword,
        )
        notes_parts.append(
            f"Architecture: HashEmbedCNN — width={tok2vec_width}, depth={tok2vec_depth}, "
            f"embed_size={tok2vec_embed_size}, window={tok2vec_window}, "
            f"maxout_pieces={tok2vec_maxout}, subword={'yes' if tok2vec_subword else 'no'}."
        )

    elif name == "tagger" and n_labels:
        param_count = tok2vec_width * n_labels
        acc = perf.get("tag_acc")
        notes_parts.append(f"{n_labels} POS tags: {', '.join(label_list[:8])}{'…' if n_labels > 8 else '.'}")
        if acc:
            notes_parts.append(f"Accuracy: {acc:.1%}")

    elif name == "morphologizer" and n_labels:
        param_count = tok2vec_width * n_labels
        notes_parts.append(f"{n_labels} morphological features.")

    elif name == "senter" and n_labels:
        param_count = tok2vec_width * 2
        f1 = perf.get("sents_f")
        notes_parts.append("Outputs I (inside sentence) or S (sentence start) per token.")
        if f1:
            notes_parts.append(f"F1: {f1:.1%}")

    elif name == "textcat" and n_labels:
        param_count = tok2vec_width * n_labels
        notes_parts.append(f"{n_labels} categories: {', '.join(label_list[:8])}.")

    elif name == "parser":
        # Transition-based: rough estimate (tok2vec output → upper MLP → actions)
        n_moves = n_labels * 4 + 3  # SHIFT, REDUCE, LEFT-ARC×n, RIGHT-ARC×n, BREAK
        param_count = tok2vec_width * 64 + 64 * n_moves  # upper MLP approximation
        uas = perf.get("dep_uas"); las = perf.get("dep_las")
        notes_parts.append(
            f"{n_labels} dependency relation labels. "
            f"Sample: {', '.join(label_list[:6])}{'…' if n_labels > 6 else '.'}"
        )
        if uas:
            notes_parts.append(f"UAS: {uas:.1%}  LAS: {las:.1%}" if las else f"UAS: {uas:.1%}")

    elif name == "ner":
        # BILUO: each entity type → 4 tags (B/I/L/U) + O = 4n+1 outputs
        n_out = 4 * n_labels + 1 if n_labels else 0
        param_count = tok2vec_width * 64 + 64 * n_out if n_out else None
        ep = perf.get("ents_p"); er = perf.get("ents_r"); ef = perf.get("ents_f")
        notes_parts.append(
            f"{n_labels} entity types: {', '.join(label_list[:8])}{'…' if n_labels > 8 else '.'}"
        )
        if ef:
            notes_parts.append(f"Precision: {ep:.1%}  Recall: {er:.1%}  F1: {ef:.1%}" if ep and er else f"F1: {ef:.1%}")

    elif name in _RULE_BASED:
        param_count = 0

    params: dict[str, Any] = {"component": name}
    if n_labels:
        params["num_labels"] = n_labels
    if name == "tok2vec":
        params.update({"width": tok2vec_width, "depth": tok2vec_depth,
                       "embed_size": tok2vec_embed_size})
    if btype == BlockType.RULE_BASED:
        params["rule_based"] = True

    return ArchBlock(
        id=name,
        label=label,
        type=btype,
        params=params,
        param_count=param_count,
        notes="\n".join(p for p in notes_parts if p),
    )


def _parse_spacy(config: dict, model_id: str) -> ArchitectureIR:
    """Parse a spaCy NLP pipeline from meta.json (+ optional config.cfg data)."""
    name = config.get("name", model_id.split("/")[-1])
    lang = config.get("lang", "en")
    version = config.get("version", "")
    pipeline: list[str] = config.get("pipeline", [])
    labels: dict = config.get("labels", {})
    vectors_info: dict = config.get("vectors", {})
    performance: dict = config.get("performance", {})
    cfg: dict = config.get("_cfg", {})
    disabled: list[str] = config.get("disabled", [])

    # Architecture dimensions — from config.cfg or well-known defaults
    tok2vec_width    = cfg.get("tok2vec_width", 96)
    tok2vec_depth    = cfg.get("tok2vec_depth", 4)
    tok2vec_embed    = cfg.get("tok2vec_embed_size", 2000)
    tok2vec_window   = cfg.get("tok2vec_window", 1)
    tok2vec_maxout   = cfg.get("tok2vec_maxout", 3)
    tok2vec_subword  = cfg.get("tok2vec_subword", True)

    blocks: list[ArchBlock] = []

    # ── Tokenizer (always present, rule-based) ────────────────────────────────
    blocks.append(ArchBlock(
        id="tokenizer",
        label="Tokenizer",
        type=BlockType.RULE_BASED,
        params={"component": "tokenizer", "rule_based": True, "lang": lang},
        param_count=0,
        notes=(
            f"Language-specific rule-based tokenizer ({lang}).\n"
            "Segments raw text into tokens using whitespace + exception rules.\n"
            "No learned parameters — writes tokens to the shared Doc object."
        ),
    ))

    # ── Word vectors table (static, not trained) ──────────────────────────────
    vec_count = vectors_info.get("vectors", 0)
    vec_width = vectors_info.get("width", 0)
    if vec_count and vec_width:
        blocks.append(ArchBlock(
            id="vectors",
            label=f"Word Vectors ({vectors_info.get('name', 'static')})",
            type=BlockType.EMBEDDING,
            params={"vocab_size": vec_count, "hidden_size": vec_width,
                    "position_embedding_type": "none"},
            param_count=vec_count * vec_width,
            notes=(
                f"Static pre-trained word vectors: {vec_count:,} entries × {vec_width} dims "
                f"= {vec_count * vec_width / 1e6:.1f}M values.\n"
                "Loaded from a lookup table at runtime — not updated during training.\n"
                "Available to all components via Doc.vector but not the primary Tok2Vec input."
            ),
        ))

    # ── Separate pipeline components into three buckets ───────────────────────
    # 1. tok2vec  2. neural heads  3. rule-based post-processors
    neural_head_comps: list[str] = []
    rule_based_comps:  list[str] = []
    has_tok2vec = "tok2vec" in pipeline

    for comp in pipeline:
        if comp == "tok2vec":
            continue
        elif comp in _NEURAL_HEADS:
            neural_head_comps.append(comp)
        elif comp in _RULE_BASED:
            rule_based_comps.append(comp)
        else:
            # Unknown — treat as neural head so it appears in the group
            neural_head_comps.append(comp)

    def _make_block(comp: str) -> ArchBlock:
        comp_labels = labels.get(comp, [])
        n_labels = len(comp_labels) if isinstance(comp_labels, list) else 0
        blk = _spacy_component_block(
            comp, n_labels,
            tok2vec_width, tok2vec_depth, tok2vec_embed,
            tok2vec_window, tok2vec_maxout, tok2vec_subword,
            comp_labels if isinstance(comp_labels, list) else [],
            performance,
        )
        if comp in disabled:
            blk.notes = (blk.notes or "") + "\n[disabled by default]"
        return blk

    # ── Tok2Vec encoder ───────────────────────────────────────────────────────
    if has_tok2vec:
        blocks.append(_make_block("tok2vec"))

    # ── Neural heads — grouped to show shared-feature fan-out ────────────────
    if neural_head_comps:
        head_blocks = [_make_block(c) for c in neural_head_comps]
        total_head_params = sum(b.param_count or 0 for b in head_blocks)

        if len(head_blocks) == 1:
            # Single head — no need for a group
            blocks.append(head_blocks[0])
        else:
            group_notes = (
                "All heads below read from the SAME shared Tok2Vec feature vectors "
                "written to the Doc — they do NOT pass tensors to each other.\n\n"
                "Each head is an independent classifier/transition-system on top of "
                "the shared contextual token representations.\n\n"
                "Expand to inspect individual heads."
            )
            head_labels = ", ".join(
                _SPACY_COMPONENT_META.get(c, (c,))[0] for c in neural_head_comps
            )
            blocks.append(ArchBlock(
                id="neural_heads",
                label=f"Neural Heads ({len(head_blocks)}×)",
                type=BlockType.NLP_HEAD_GROUP,
                params={"num_heads": len(head_blocks), "shared_encoder": "tok2vec",
                        "heads": head_labels},
                param_count=total_head_params,
                children=head_blocks,
                notes=group_notes,
            ))

    # ── Rule-based post-processors ────────────────────────────────────────────
    for comp in rule_based_comps:
        blocks.append(_make_block(comp))

    # Task inference
    task = "token-classification"
    if "textcat" in pipeline or "textcat_multilabel" in pipeline:
        task = "text-classification"
    elif pipeline == ["ner"] or pipeline == ["entity_ruler"]:
        task = "token-classification"

    # Summary performance note on first block
    perf_parts = []
    if performance.get("tag_acc"):
        perf_parts.append(f"POS acc {performance['tag_acc']:.1%}")
    if performance.get("dep_las"):
        perf_parts.append(f"dep LAS {performance['dep_las']:.1%}")
    if performance.get("ents_f"):
        perf_parts.append(f"NER F1 {performance['ents_f']:.1%}")
    if performance.get("sents_f"):
        perf_parts.append(f"senter F1 {performance['sents_f']:.1%}")
    speed = performance.get("speed", 0)
    if speed:
        perf_parts.append(f"{int(speed):,} words/sec")

    display = f"spaCy {lang}_{name}"
    if version:
        display += f" v{version}"

    ir = ArchitectureIR(
        name=model_id,
        display_name=display,
        family="spacy",
        task=task,
        architectures=[f"spacy.{lang}_{name}"],
        source=SourceType.HF_CONFIG,
        source_confidence=SourceConfidence.EXACT,
        blocks=blocks,
    )
    ir.compute = estimate_compute(ir)

    # Attach pipeline-level performance summary to the IR display_name notes
    # (surfaced in the detail panel)
    if perf_parts and ir.blocks:
        ir.blocks[0].notes = (ir.blocks[0].notes or "") + (
            f"\n\nPipeline performance: {' · '.join(perf_parts)}."
        )

    return ir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_block(block_id: str, notes: str) -> ArchBlock:
    """Create a zero-param residual addition node."""
    return ArchBlock(
        id=block_id,
        label="Add (Residual)",
        type=BlockType.ADD,
        params={},
        param_count=0,
        notes=notes,
    )


def _build_encoder_children(
    h: int,
    num_heads: int,
    intermediate: int,
    act: ActivationType,
    is_causal: bool,
    pre_norm: bool = False,
) -> list[ArchBlock]:
    """Build the standard BERT/GPT child block list."""
    if pre_norm:
        # GPT-2 / Pre-LN style: LN → sublayer → Add (residual)
        return [
            ArchBlock(id="ln_1", label="LayerNorm", type=BlockType.LAYER_NORM,
                      params={"normalized_shape": h, "norm_type": "layer_norm"}, param_count=h * 2),
            ArchBlock(id="attn", label="Causal Self-Attention" if is_causal else "Self-Attention",
                      type=BlockType.MULTI_HEAD_ATTENTION,
                      params={"hidden_size": h, "num_heads": num_heads, "head_dim": h // num_heads,
                              "attention_type": "multi_head", "is_causal": is_causal},
                      param_count=4 * h * h),
            _add_block("add_attn", "h = x + Attention(LN(x))  - residual reconnects the pre-LN input after attention."),
            ArchBlock(id="ln_2", label="LayerNorm", type=BlockType.LAYER_NORM,
                      params={"normalized_shape": h, "norm_type": "layer_norm"}, param_count=h * 2),
            ArchBlock(id="mlp", label="MLP", type=BlockType.FEED_FORWARD,
                      params={"hidden_size": h, "intermediate_size": intermediate, "activation": act.value},
                      param_count=2 * h * intermediate + h + intermediate),
            _add_block("add_ffn", "h = h + FFN(LN(h))  - second residual reconnects input after the feed-forward block."),
        ]
    else:
        # BERT / Post-LN style: sublayer → Add (residual) → LN
        return [
            ArchBlock(id="self_attn", label="Self-Attention", type=BlockType.MULTI_HEAD_ATTENTION,
                      params={"hidden_size": h, "num_heads": num_heads, "head_dim": h // num_heads,
                              "attention_type": "multi_head", "is_causal": is_causal},
                      param_count=4 * h * h),
            _add_block("add_attn", "h = x + Attention(x)  - residual 1 (Post-LN: Add before LayerNorm)."),
            ArchBlock(id="attn_layernorm", label="LayerNorm", type=BlockType.LAYER_NORM,
                      params={"normalized_shape": h, "norm_type": "layer_norm"}, param_count=h * 2),
            ArchBlock(id="ffn", label="Feed-Forward", type=BlockType.FEED_FORWARD,
                      params={"hidden_size": h, "intermediate_size": intermediate, "activation": act.value},
                      param_count=2 * h * intermediate + h + intermediate),
            _add_block("add_ffn", "h = h + FFN(h)  - residual 2 (Post-LN: Add before LayerNorm)."),
            ArchBlock(id="ffn_layernorm", label="LayerNorm", type=BlockType.LAYER_NORM,
                      params={"normalized_shape": h, "norm_type": "layer_norm"}, param_count=h * 2),
        ]


def _build_task_head(archs: list[str], config: dict, h: int) -> ArchBlock | None:
    arch_str = " ".join(archs).lower()
    if "tokenclassification" in arch_str:
        num_labels = config.get("num_labels", 2)
        return ArchBlock(id="classifier", label="Token Classifier", type=BlockType.LINEAR,
                         params={"in_features": h, "out_features": num_labels, "bias": True},
                         param_count=h * num_labels + num_labels)
    if "sequenceclassification" in arch_str:
        num_labels = config.get("num_labels", 2)
        return ArchBlock(id="classifier", label="Sequence Classifier", type=BlockType.LINEAR,
                         params={"in_features": h, "out_features": num_labels, "bias": True},
                         param_count=h * num_labels + num_labels)
    if "questionanswering" in arch_str:
        return ArchBlock(id="qa_outputs", label="QA Span Head", type=BlockType.LINEAR,
                         params={"in_features": h, "out_features": 2, "bias": True},
                         param_count=h * 2 + 2)
    if "maskedlm" in arch_str or "formaskedlm" in arch_str:
        vocab_size = config.get("vocab_size", 30522)
        return ArchBlock(
            id="lm_head",
            label="MLM Head (weight-tied)",
            type=BlockType.LINEAR,
            params={"in_features": h, "out_features": vocab_size, "bias": True},
            # Weight-tied to the input embedding matrix - same {vocab_size}x{h} tensor is reused.
            # No extra parameters are stored; counting them would double-count the embedding.
            param_count=0,
            notes=(
                f"Linear({h} -> {vocab_size:,}) applied to every token position to predict "
                f"masked tokens. The weight matrix ({h:,} x {vocab_size:,} = "
                f"{h * vocab_size / 1e6:.1f}M values) is TIED to the input embedding "
                f"table - the same tensor is shared, so no additional parameters are stored. "
                f"Only the output bias ({vocab_size:,} params) is unique to this layer."
            ),
        )
    return None


def _infer_task(archs: list[str], config: dict) -> str:
    arch_str = " ".join(archs).lower()
    if "tokenclassification" in arch_str:
        return "token-classification"
    if "sequenceclassification" in arch_str:
        return "text-classification"
    if "questionanswering" in arch_str:
        return "question-answering"
    if "causallm" in arch_str or "decoder" in arch_str.lower():
        return "text-generation"
    if "maskedlm" in arch_str or "formaskedlm" in arch_str:
        return "fill-mask"
    if "seq2seq" in arch_str or "conditional" in arch_str:
        return "text2text-generation"
    return "unknown"


def _embedding_params(vocab_size: int, h: int, max_pos: int, type_vocab: int | None = None) -> int:
    p = vocab_size * h + max_pos * h
    if type_vocab:
        p += type_vocab * h
    p += h * 2  # embedding LayerNorm
    return p


def _gqa_params(h: int, num_heads: int, num_kv_heads: int) -> int:
    head_dim = h // num_heads
    q = h * num_heads * head_dim
    k = h * num_kv_heads * head_dim
    v = h * num_kv_heads * head_dim
    o = num_heads * head_dim * h
    return q + k + v + o


def _param_label(config: dict) -> str:
    h = config.get("n_embd", config.get("hidden_size", 768))
    layers = config.get("n_layer", config.get("num_hidden_layers", 12))
    total = 12 * layers * h * h
    if total > 1e9:
        return f"{total / 1e9:.1f}B"
    return f"{total / 1e6:.0f}M"


def _billion_label(config: dict) -> str:
    h = config.get("hidden_size", 4096)
    layers = config.get("num_hidden_layers", 32)
    intermediate = config.get("intermediate_size", h * 4)
    approx = layers * (4 * h * h + 3 * h * intermediate)
    if approx > 1e9:
        return f"~{approx / 1e9:.0f}B"
    return f"~{approx / 1e6:.0f}M"


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

_FAMILY_PARSERS = {
    # BERT-style bidirectional encoders
    "bert": _parse_bert_family,
    "roberta": _parse_bert_family,
    "distilbert": _parse_bert_family,
    "albert": _parse_bert_family,
    "electra": _parse_bert_family,
    "deberta": _parse_bert_family,
    "deberta-v2": _parse_bert_family,
    "camembert": _parse_bert_family,
    "xlm-roberta": _parse_bert_family,
    # GPT-2 style causal decoders (old key names)
    "gpt2": _parse_gpt2_family,
    "gpt_neo": _parse_gpt2_family,
    "gpt_neox": _parse_gpt2_family,
    "gptj": _parse_gpt2_family,
    # LLaMA-style modern decoders (RoPE + RMSNorm + SwiGLU/GQA)
    "llama": _parse_llama_family,
    "mistral": _parse_llama_family,
    "mixtral": _parse_llama_family,
    "falcon": _parse_llama_family,
    "gemma": _parse_llama_family,
    "gemma2": _parse_llama_family,
    "phi": _parse_llama_family,
    "phi3": _parse_llama_family,
    "qwen2": _parse_llama_family,
    "cohere": _parse_llama_family,
    "olmo": _parse_llama_family,
    "olmo2": _parse_llama_family,
    "internlm2": _parse_llama_family,
    # T5-style encoder-decoders
    "t5": _parse_t5_family,
    "mt5": _parse_t5_family,
    "umt5": _parse_t5_family,
    "longt5": _parse_t5_family,
    # Mamba SSM
    "mamba": _parse_mamba_family,
    "mamba2": _parse_mamba_family,
    "falcon_mamba": _parse_mamba_family,
    # DeepSeek V2 / V3 language models
    "deepseek_v2": _parse_deepseek_v2,
    "deepseek_v3": _parse_deepseek_v2,
    # DeepSeek Vision-Language models
    "deepseek_vl_v2": _parse_deepseek_vl_v2,
    "deepseek_vl": _parse_deepseek_vl_v2,
    # spaCy NLP pipelines (parsed from meta.json)
    "spacy": _parse_spacy,
}


def parse_hf_config(config: dict, model_id: str) -> ArchitectureIR:
    """
    Convert a raw HuggingFace config dict into an Architecture IR.
    Dispatches to the appropriate family parser.
    """
    model_type = config.get("model_type", "").lower()
    parser_fn = _FAMILY_PARSERS.get(model_type)

    if parser_fn is None:
        logger.debug("Unknown model_type '%s' for '%s' - using smart generic fallback", model_type, model_id)
        return _smart_generic_fallback(config, model_id)

    return parser_fn(config, model_id)


def _smart_generic_fallback(config: dict, model_id: str) -> ArchitectureIR:
    """
    Tier-2 auto-detecting parser for any unregistered model type.

    Reads standard HuggingFace config fields and assembles an IR using the
    same formulas as the explicit family parsers.  Covers ~90% of modern LLMs
    without any hard-coded family knowledge.

    Confidence levels:
      HIGH   — all three key fields (hidden_size, num_hidden_layers,
                num_attention_heads) are present and non-zero.
      MEDIUM — one or two key fields are missing (values inferred from
                common defaults).
      LOW    — most fields are absent; result is a rough approximation.
    """
    model_type = config.get("model_type", "unknown")
    archs = config.get("architectures", [])
    arch_str = " ".join(archs).lower()

    # ── Core dimensions ────────────────────────────────────────────────────────
    h = (
        config.get("hidden_size")
        or config.get("d_model")
        or config.get("n_embd")
        or config.get("model_dim")
        or 0
    )
    num_layers = (
        config.get("num_hidden_layers")
        or config.get("n_layer")
        or config.get("num_layers")
        or config.get("n_layers")
        or 0
    )
    num_heads = (
        config.get("num_attention_heads")
        or config.get("n_head")
        or config.get("num_heads")
        or 0
    )

    # Confidence: count how many of the three key fields are present
    _key_fields_present = sum(1 for v in [h, num_layers, num_heads] if v)
    if _key_fields_present == 3:
        confidence = SourceConfidence.HIGH
    elif _key_fields_present >= 1:
        confidence = SourceConfidence.MEDIUM
    else:
        confidence = SourceConfidence.LOW

    # Apply safe defaults for inference
    h = h or 768
    num_layers = num_layers or 12
    num_heads = num_heads or (h // 64)

    num_kv_heads = (
        config.get("num_key_value_heads")
        or config.get("num_kv_heads")
        or num_heads
    )
    vocab_size = config.get("vocab_size") or 32000
    max_pos = (
        config.get("max_position_embeddings")
        or config.get("n_positions")
        or config.get("max_sequence_length")
        or 2048
    )
    intermediate = (
        config.get("intermediate_size")
        or config.get("ffn_dim")
        or config.get("d_ff")
        or h * 4
    )

    # ── Architecture class detection ───────────────────────────────────────────
    is_enc_dec = config.get("is_encoder_decoder", False)
    _dec_signals = ("causallm", "decoderlm", "gpt", "llm", "decoder", "generative")
    _enc_signals = ("maskedlm", "encoder", "bert", "roberta", "electra", "deberta")
    _seq2seq_signals = ("seq2seq", "conditional", "t5", "bart", "pegasus")

    if is_enc_dec or any(s in arch_str for s in _seq2seq_signals):
        arch_class = "encoder_decoder"
    elif any(s in arch_str for s in _enc_signals):
        arch_class = "encoder"
    elif any(s in arch_str for s in _dec_signals):
        arch_class = "decoder"
    else:
        # Fallback heuristic: modern models with vocab > 50k tend to be decoders
        arch_class = "decoder" if vocab_size > 40000 else "encoder"

    # ── Activation & norm ──────────────────────────────────────────────────────
    act_str = (
        config.get("hidden_act")
        or config.get("activation_function")
        or config.get("activation")
        or ("silu" if arch_class == "decoder" else "gelu")
    )
    act = _ACT_MAP.get(act_str, ActivationType.GELU)
    is_gated = act in (ActivationType.SILU, ActivationType.SWIGLU, ActivationType.GEGLU)
    ffn_matrices = 3 if is_gated else 2

    # RMSNorm is the standard for modern decoders
    has_rms_norm = (
        "rms_norm_eps" in config
        or config.get("norm_type") == "rms_norm"
        or arch_class == "decoder"
    )
    norm_type = "rms_norm" if has_rms_norm else "layer_norm"
    norm_params = h if has_rms_norm else h * 2

    # ── MoE detection ─────────────────────────────────────────────────────────
    n_experts = (
        config.get("num_local_experts")
        or config.get("n_routed_experts")
        or config.get("num_experts")
        or 0
    )
    n_experts_per_tok = (
        config.get("num_experts_per_tok")
        or config.get("top_k")
        or (2 if n_experts else 0)
    )
    moe_intermediate = config.get("moe_intermediate_size") or intermediate
    is_moe = n_experts > 0

    # ── Position encoding ─────────────────────────────────────────────────────
    pos_type = config.get("position_embedding_type", "")
    if not pos_type:
        if "rope_theta" in config or "rope_scaling" in config:
            pos_type = "rope"
        elif arch_class == "decoder":
            pos_type = "rope"
        else:
            pos_type = "absolute"

    # ── Param helpers ─────────────────────────────────────────────────────────
    head_dim = h // num_heads
    attn_p = _gqa_params(h, num_heads, num_kv_heads)
    ffn_p = (
        (n_experts * ffn_matrices * h * moe_intermediate + h * n_experts)
        if is_moe
        else ffn_matrices * h * intermediate
    )
    per_layer = attn_p + ffn_p + 2 * norm_params
    emb_p = vocab_size * h
    final_norm_p = norm_params
    lm_head_p = h * vocab_size

    # ── Block construction ────────────────────────────────────────────────────
    def _norm_block(bid: str, label: str) -> ArchBlock:
        return ArchBlock(
            id=bid, label=label, type=BlockType.LAYER_NORM,
            params={"normalized_shape": h, "norm_type": norm_type},
            param_count=norm_params,
        )

    def _ffn_block(bid: str) -> ArchBlock:
        if is_moe:
            return ArchBlock(
                id=bid, label=f"MoE FFN ({n_experts} experts, top-{n_experts_per_tok})",
                type=BlockType.MOE_FEED_FORWARD,
                params={"hidden_size": h, "intermediate_size": moe_intermediate,
                        "num_experts": n_experts, "num_experts_per_tok": n_experts_per_tok,
                        "activation": act.value},
                param_count=ffn_p,
            )
        return ArchBlock(
            id=bid, label=f"{'Gated ' if is_gated else ''}FFN",
            type=BlockType.FEED_FORWARD,
            params={"hidden_size": h, "intermediate_size": intermediate,
                    "activation": act.value},
            param_count=ffn_p,
        )

    def _attn_block(bid: str, is_causal: bool) -> ArchBlock:
        attn_type = AttentionType.GQA if num_kv_heads != num_heads else AttentionType.MHA
        params: dict[str, Any] = {
            "hidden_size": h, "num_heads": num_heads, "head_dim": head_dim,
            "attention_type": attn_type.value, "is_causal": is_causal,
        }
        if num_kv_heads != num_heads:
            params["num_kv_heads"] = num_kv_heads
        return ArchBlock(
            id=bid, label=f"{'Causal ' if is_causal else ''}{'GQA' if num_kv_heads != num_heads else 'MHA'}",
            type=BlockType.MULTI_HEAD_ATTENTION,
            params=params, param_count=attn_p,
        )

    def _decoder_layer_children(is_causal: bool = True) -> list[ArchBlock]:
        return [
            _norm_block("input_norm", "RMSNorm" if has_rms_norm else "LayerNorm"),
            _attn_block("self_attn", is_causal=is_causal),
            _add_block("add_attn", "Residual 1"),
            _norm_block("post_attn_norm", "RMSNorm" if has_rms_norm else "LayerNorm"),
            _ffn_block("mlp"),
            _add_block("add_ffn", "Residual 2"),
        ]

    blocks: list[ArchBlock] = []

    emb_block = ArchBlock(
        id="embed_tokens", label="Embeddings", type=BlockType.EMBEDDING,
        params={"vocab_size": vocab_size, "hidden_size": h,
                "max_position_embeddings": max_pos,
                "position_embedding_type": pos_type},
        param_count=emb_p,
    )

    if arch_class == "encoder":
        # BERT-style encoder-only
        children = _build_encoder_children(h, num_heads, intermediate, act,
                                           is_causal=False, pre_norm=False)
        blocks = [
            emb_block,
            ArchBlock(
                id="encoder", label=f"Transformer Encoder ({model_type})",
                type=BlockType.TRANSFORMER_STACK,
                params={"hidden_size": h, "num_hidden_layers": num_layers,
                        "num_attention_heads": num_heads,
                        "intermediate_size": intermediate, "is_causal": False},
                repeat=num_layers, children=children,
                param_count=_stack_param_count(children, num_layers),
            ),
        ]

    elif arch_class == "encoder_decoder":
        # T5-style encoder-decoder
        enc_children = _decoder_layer_children(is_causal=False)
        dec_children = _decoder_layer_children(is_causal=True)
        blocks = [
            emb_block,
            ArchBlock(
                id="encoder", label="Encoder", type=BlockType.TRANSFORMER_STACK,
                params={"hidden_size": h, "num_hidden_layers": num_layers,
                        "num_attention_heads": num_heads, "is_causal": False},
                repeat=num_layers, children=enc_children,
                param_count=_stack_param_count(enc_children, num_layers),
            ),
            ArchBlock(
                id="decoder", label="Decoder", type=BlockType.TRANSFORMER_STACK,
                params={"hidden_size": h, "num_hidden_layers": num_layers,
                        "num_attention_heads": num_heads, "is_causal": True},
                repeat=num_layers, children=dec_children,
                param_count=_stack_param_count(dec_children, num_layers),
            ),
            ArchBlock(id="lm_head", label="LM Head", type=BlockType.LINEAR,
                      params={"in_features": h, "out_features": vocab_size, "bias": False},
                      param_count=lm_head_p),
        ]

    else:
        # Decoder-only (the vast majority of modern LLMs)
        children = _decoder_layer_children(is_causal=True)
        blocks = [
            emb_block,
            ArchBlock(
                id="layers",
                label=f"Transformer Decoder ({model_type})",
                type=BlockType.TRANSFORMER_STACK,
                params={
                    "hidden_size": h, "num_hidden_layers": num_layers,
                    "num_attention_heads": num_heads, "num_key_value_heads": num_kv_heads,
                    "intermediate_size": intermediate, "is_causal": True,
                },
                repeat=num_layers, children=children,
                param_count=_stack_param_count(children, num_layers),
            ),
            _norm_block("norm", "Final RMSNorm" if has_rms_norm else "Final LayerNorm"),
            ArchBlock(id="lm_head", label="LM Head", type=BlockType.LINEAR,
                      params={"in_features": h, "out_features": vocab_size, "bias": False},
                      param_count=lm_head_p),
        ]

    display = model_id.split("/")[-1]
    ir = ArchitectureIR(
        name=model_id,
        display_name=display,
        family=model_type,
        task=_infer_task(archs, config),
        architectures=archs,
        source=SourceType.HF_CONFIG,
        source_confidence=confidence,
        blocks=blocks,
    )
    ir.compute = estimate_compute(ir)
    return ir
