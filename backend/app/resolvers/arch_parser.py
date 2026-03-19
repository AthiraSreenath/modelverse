"""
Architecture Parser - converts a raw HuggingFace config dict into an Architecture IR.

Supports: BERT, DistilBERT, RoBERTa, GPT-2, LLaMA, Mistral, Mixtral, T5, Mamba, Falcon.
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
}


def parse_hf_config(config: dict, model_id: str) -> ArchitectureIR:
    """
    Convert a raw HuggingFace config dict into an Architecture IR.
    Dispatches to the appropriate family parser.
    """
    model_type = config.get("model_type", "").lower()
    parser_fn = _FAMILY_PARSERS.get(model_type)

    if parser_fn is None:
        logger.warning("Unknown model_type '%s' for '%s' - using generic fallback", model_type, model_id)
        return _generic_fallback(config, model_id)

    return parser_fn(config, model_id)


def _generic_fallback(config: dict, model_id: str) -> ArchitectureIR:
    """Best-effort parsing for unknown model types."""
    h = config.get("hidden_size", config.get("d_model", 768))
    num_layers = config.get("num_hidden_layers", config.get("num_layers", 12))
    num_heads = config.get("num_attention_heads", config.get("num_heads", 12))
    vocab_size = config.get("vocab_size", 30522)
    max_pos = config.get("max_position_embeddings", 512)
    model_type = config.get("model_type", "unknown")

    emb_block = ArchBlock(
        id="embeddings",
        label="Embeddings",
        type=BlockType.EMBEDDING,
        params={"vocab_size": vocab_size, "hidden_size": h, "max_position_embeddings": max_pos},
        param_count=vocab_size * h,
    )
    encoder_block = ArchBlock(
        id="encoder",
        label=f"Transformer Stack ({model_type})",
        type=BlockType.TRANSFORMER_STACK,
        params={"hidden_size": h, "num_hidden_layers": num_layers, "num_attention_heads": num_heads},
        repeat=num_layers,
        children=[],
        unknown_fields=["intermediate_size", "activation", "norm_type"],
    )

    ir = ArchitectureIR(
        name=model_id,
        display_name=model_id.split("/")[-1],
        family=model_type,
        task=_infer_task(config.get("architectures", []), config),
        architectures=config.get("architectures", []),
        source=SourceType.HF_CONFIG,
        source_confidence=SourceConfidence.HIGH,
        blocks=[emb_block, encoder_block],
    )
    ir.compute = estimate_compute(ir)
    return ir
