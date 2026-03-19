"""
Architecture IR - Pydantic models for the unified Architecture Intermediate Representation.

Every input (HF model name, uploaded file, web search) resolves to this schema.
Every output (graph, compute stats, LLM context, diffs) is generated from it.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SourceType(str, Enum):
    HF_CONFIG = "hf_config"          # Parsed directly from HuggingFace config.json - exact
    FILE_HEADER = "file_header"       # Parsed from uploaded file header - exact
    PREBAKED = "prebaked"             # Shipped pre-baked IR - exact
    PAPER = "paper"                   # Extracted from ArXiv / published paper
    WEB_SYNTHESIS = "web_synthesis"   # Synthesized from multiple web sources
    LLM_KNOWLEDGE = "llm_knowledge"   # From LLM training data alone


class SourceConfidence(str, Enum):
    EXACT = "exact"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class BlockType(str, Enum):
    EMBEDDING = "embedding"
    TRANSFORMER_STACK = "transformer_stack"
    MULTI_HEAD_ATTENTION = "multi_head_attention"
    FEED_FORWARD = "feed_forward"
    MOE_FEED_FORWARD = "moe_feed_forward"
    LAYER_NORM = "layer_norm"
    LINEAR = "linear"
    CONV1D = "conv1d"
    SSM = "ssm"
    RNN = "rnn"
    POOLING = "pooling"
    DROPOUT = "dropout"
    ACTIVATION = "activation"
    ADD = "add"            # Residual addition: output = x + sublayer(x)
    UNKNOWN = "unknown"


class AttentionType(str, Enum):
    MHA = "multi_head"       # Standard multi-head attention
    GQA = "grouped_query"    # Grouped-query attention (LLaMA-3, Mistral)
    MQA = "multi_query"      # Multi-query attention


class ActivationType(str, Enum):
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SWIGLU = "swiglu"
    GEGLU = "geglu"
    TANH = "tanh"


# ---------------------------------------------------------------------------
# Block parameter models - typed params per block type
# ---------------------------------------------------------------------------


class EmbeddingParams(BaseModel):
    vocab_size: int
    hidden_size: int
    max_position_embeddings: int | None = None
    type_vocab_size: int | None = None  # BERT segment embeddings
    position_embedding_type: str | None = "absolute"  # absolute | rope | alibi | none


class MultiHeadAttentionParams(BaseModel):
    hidden_size: int
    num_heads: int
    head_dim: int | None = None
    num_kv_heads: int | None = None   # GQA: fewer KV heads than Q heads
    attention_type: AttentionType = AttentionType.MHA
    is_causal: bool = False           # decoder (causal) vs encoder (bidirectional)


class FeedForwardParams(BaseModel):
    hidden_size: int
    intermediate_size: int
    activation: ActivationType = ActivationType.GELU


class MoEFeedForwardParams(BaseModel):
    hidden_size: int
    intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    activation: ActivationType = ActivationType.SILU


class LayerNormParams(BaseModel):
    normalized_shape: int
    norm_type: str = "layer_norm"  # layer_norm | rms_norm


class LinearParams(BaseModel):
    in_features: int
    out_features: int
    bias: bool = True


class SSMParams(BaseModel):
    hidden_size: int
    state_size: int
    conv_kernel: int = 4
    expand: int = 2


# ---------------------------------------------------------------------------
# Core block / IR models
# ---------------------------------------------------------------------------


class ArchBlock(BaseModel):
    """A single layer or group of layers in the architecture."""

    id: str
    label: str
    type: BlockType
    params: dict[str, Any] = Field(default_factory=dict)
    repeat: int = 1                       # for transformer_stack: num repeated layers
    children: list[ArchBlock] = Field(default_factory=list)
    param_count: int | None = None        # pre-computed for this block (excl. children)
    notes: str | None = None              # optional human note
    unknown_fields: list[str] = Field(default_factory=list)  # fields we couldn't determine


class ComputeStats(BaseModel):
    """Formula-based compute statistics for the full model."""

    params_total: int
    params_embedding: int = 0
    params_encoder: int = 0       # or decoder
    params_head: int = 0
    flops_per_token: int | None = None
    memory_fp32_gb: float | None = None
    memory_fp16_gb: float | None = None
    memory_bf16_gb: float | None = None
    memory_int8_gb: float | None = None
    memory_int4_gb: float | None = None


class SourceRef(BaseModel):
    """Citation for web-derived architectures."""

    title: str
    url: str
    excerpt: str | None = None


class ArchitectureIR(BaseModel):
    """
    The unified Architecture Intermediate Representation.
    All inputs resolve to this. All outputs are generated from this.
    """

    schema_version: str = "1.0"
    name: str                          # e.g. "dslim/bert-base-NER" or "GPT-3"
    display_name: str | None = None    # Human-readable name
    family: str | None = None          # "bert" | "gpt2" | "llama" | "mistral" | "t5" …
    task: str | None = None            # "token-classification" | "text-generation" …
    architectures: list[str] = Field(default_factory=list)  # raw HF architectures list e.g. ["BertForMaskedLM"]
    source: SourceType
    source_confidence: SourceConfidence = SourceConfidence.EXACT
    source_refs: list[SourceRef] = Field(default_factory=list)

    blocks: list[ArchBlock]
    compute: ComputeStats | None = None

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# Edit engine models
# ---------------------------------------------------------------------------


class EditOp(str, Enum):
    SET_REPEAT = "set_repeat"        # change number of repeated blocks
    SET_PARAM = "set_param"          # change a specific parameter value
    ADD_BLOCK = "add_block"          # insert a new block
    REMOVE_BLOCK = "remove_block"    # remove a block
    REPLACE_BLOCK = "replace_block"  # swap one block type for another


class EditSpec(BaseModel):
    op: EditOp
    target: str                       # block id or dotted path e.g. "encoder.self_attn"
    value: Any | None = None          # new value for set_repeat / set_param
    key: str | None = None            # param key for set_param
    block: dict | None = None         # new block definition for add_block / replace_block
    after: str | None = None          # insert after this block id


class DiffEntry(BaseModel):
    path: str
    old: Any
    new: Any


class ComputeDelta(BaseModel):
    params_delta: int
    params_delta_pct: float
    memory_fp16_delta_gb: float | None = None
    flops_delta: int | None = None


class EditResult(BaseModel):
    new_ir: ArchitectureIR
    diff: list[DiffEntry]
    compute_delta: ComputeDelta
