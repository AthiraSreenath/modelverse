"""
Formula-based compute estimator.
Runs in <1ms — pure arithmetic, no model loading.

Formulas based on:
- Kaplan et al. (2020) — scaling laws
- Chinchilla (2022) — optimal compute
- Transformer FLOPs derivation (https://arxiv.org/pdf/2205.05198)
"""

from __future__ import annotations
import math

from ..models.ir import ArchBlock, ArchitectureIR, BlockType, ComputeStats


BYTES_PER_DTYPE = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,
}


def _count_block_params(block: ArchBlock) -> int:
    """Recursively count parameters in a block and its children."""
    # If there are children (e.g. transformer stack), always recompute
    # from children × repeat so that edits to repeat are reflected.
    if block.children:
        child_sum = sum(_count_block_params(c) for c in block.children)
        return child_sum * block.repeat

    if block.param_count is not None:
        # Leaf node with a pre-computed value — use it, scaled by repeat
        return block.param_count * block.repeat

    p = block.params
    btype = block.type

    own = 0
    if btype == BlockType.EMBEDDING:
        vocab = p.get("vocab_size", 0)
        h = p.get("hidden_size", 0)
        max_pos = p.get("max_position_embeddings", 0)
        type_vocab = p.get("type_vocab_size", 0)
        pos_type = p.get("position_embedding_type", "absolute")
        own = vocab * h
        if pos_type == "absolute":
            own += max_pos * h
        if type_vocab:
            own += type_vocab * h
        own += h * 2  # embedding LayerNorm weight + bias

    elif btype == BlockType.MULTI_HEAD_ATTENTION:
        h = p.get("hidden_size", 768)
        num_heads = p.get("num_heads", 12)
        num_kv = p.get("num_kv_heads", num_heads)
        head_dim = p.get("head_dim", h // max(num_heads, 1))
        q = h * num_heads * head_dim
        k = h * num_kv * head_dim
        v = h * num_kv * head_dim
        o = num_heads * head_dim * h
        own = q + k + v + o

    elif btype == BlockType.FEED_FORWARD:
        h = p.get("hidden_size", 768)
        inter = p.get("intermediate_size", h * 4)
        act = p.get("activation", "gelu")
        # SwiGLU / GeGLU have 3 projections (gate, up, down)
        if act in ("swiglu", "geglu", "silu"):
            own = 3 * h * inter
        else:
            own = 2 * h * inter + h + inter  # up proj + down proj + biases

    elif btype == BlockType.MOE_FEED_FORWARD:
        h = p.get("hidden_size", 4096)
        inter = p.get("intermediate_size", 14336)
        num_experts = p.get("num_experts", 8)
        own = num_experts * 3 * h * inter  # SwiGLU per expert
        own += h * num_experts  # router linear

    elif btype == BlockType.LAYER_NORM:
        shape = p.get("normalized_shape", 0)
        norm_type = p.get("norm_type", "layer_norm")
        own = shape * 2 if norm_type == "layer_norm" else shape  # RMSNorm has no bias

    elif btype == BlockType.LINEAR:
        own = p.get("in_features", 0) * p.get("out_features", 0)
        if p.get("bias", True):
            own += p.get("out_features", 0)

    return own * block.repeat


def _count_attn_flops(block: ArchBlock, seq_len: int = 512) -> int:
    """FLOPs for a single attention block (forward pass, single token at inference)."""
    p = block.params
    h = p.get("hidden_size", 768)
    num_heads = p.get("num_heads", 12)
    num_kv = p.get("num_kv_heads", num_heads)
    head_dim = p.get("head_dim", h // max(num_heads, 1))

    # Q, K, V projections: each is 2 * h * (num_heads * head_dim)
    q_flops = 2 * h * num_heads * head_dim
    k_flops = 2 * h * num_kv * head_dim
    v_flops = 2 * h * num_kv * head_dim
    # Attention scores: 2 * seq_len * num_heads * head_dim (per query position)
    score_flops = 2 * seq_len * num_heads * head_dim
    # Output projection
    o_flops = 2 * num_heads * head_dim * h
    return q_flops + k_flops + v_flops + score_flops + o_flops


def _count_ffn_flops(block: ArchBlock) -> int:
    p = block.params
    h = p.get("hidden_size", 768)
    inter = p.get("intermediate_size", h * 4)
    act = p.get("activation", "gelu")
    if act in ("swiglu", "geglu", "silu"):
        return 3 * 2 * h * inter  # 3 projections
    return 2 * 2 * h * inter  # 2 projections


def estimate_compute(ir: ArchitectureIR) -> ComputeStats:
    """
    Compute parameter counts, FLOPs per token, and memory for an Architecture IR.
    """
    total_params = 0
    params_embedding = 0
    params_encoder = 0
    params_head = 0
    total_flops = 0

    for i, block in enumerate(ir.blocks):
        p = _count_block_params(block)

        if block.type == BlockType.EMBEDDING:
            params_embedding += p
        elif block.type == BlockType.TRANSFORMER_STACK:
            params_encoder += p
            # FLOPs: sum per-child FLOPs × repeat
            for child in block.children:
                if child.type == BlockType.MULTI_HEAD_ATTENTION:
                    total_flops += _count_attn_flops(child) * block.repeat
                elif child.type in (BlockType.FEED_FORWARD, BlockType.MOE_FEED_FORWARD):
                    total_flops += _count_ffn_flops(child) * block.repeat
        elif i > 0:
            params_head += p

        total_params += p

    def mem(params: int, dtype_bytes: float) -> float:
        return round(params * dtype_bytes / (1024 ** 3), 3)

    return ComputeStats(
        params_total=total_params,
        params_embedding=params_embedding,
        params_encoder=params_encoder,
        params_head=params_head,
        flops_per_token=total_flops if total_flops > 0 else None,
        memory_fp32_gb=mem(total_params, 4),
        memory_fp16_gb=mem(total_params, 2),
        memory_bf16_gb=mem(total_params, 2),
        memory_int8_gb=mem(total_params, 1),
        memory_int4_gb=mem(total_params, 0.5),
    )
