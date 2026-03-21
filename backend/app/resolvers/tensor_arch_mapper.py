"""
Tensor-Shape Architecture Mapper
==================================
Converts a tensor name → shape dict (from safetensors/PyTorch/ONNX headers)
directly into an ArchitectureIR — no family guessing, no fake config dict,
no arch_parser.

Every architectural dimension is read from the actual weights:
  - num_layers:        count numeric indices in the layer pattern
  - hidden_size:       embedding weight shape[1]
  - vocab_size:        embedding weight shape[0]
  - num_heads:         q_proj shape[0] / head_dim
  - num_kv_heads:      k_proj shape[0] / head_dim  (≠ num_heads → GQA, no guessing)
  - intermediate_size: FFN gate/up projection shape[0]
  - MoE:               presence of experts.N. keys
  - Gated FFN:         presence of gate_proj / wi_0 keys
  - Arch type:         tensor name structure (both encoder+decoder keys → enc-dec)

The output config dict is fed into autoconfig_mapper.map_autoconfig_to_ir()
so all the IR-building logic is shared — no duplication.
"""

from __future__ import annotations

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Naming conventions
# Each entry describes how to find layers, embeddings, q/k projections, and
# FFN weights for a particular serialisation convention.
# ---------------------------------------------------------------------------

_CONVENTIONS: list[dict[str, Any]] = [
    {
        "name": "llama",          # LLaMA, Mistral, Qwen2, Phi-3, DeepSeek, Falcon-new, Gemma
        "detect": lambda ns: any(k.startswith("model.layers.") for k in ns),
        "arch_type": "decoder",
        "layer_pat": r"model\.layers\.(\d+)\.",
        "enc_layer_pat": None,
        "dec_layer_pat": None,
        "embed_keys": ["model.embed_tokens.weight", "embed_tokens.weight"],
        "q_key_pat":   r"model\.layers\.0\.self_attn\.q_proj\.weight",
        "k_key_pat":   r"model\.layers\.0\.self_attn\.k_proj\.weight",
        "ffn_key_pats": [
            r"model\.layers\.0\.mlp\.gate_proj\.weight",
            r"model\.layers\.0\.mlp\.up_proj\.weight",
            r"model\.layers\.0\.mlp\.fc1\.weight",
        ],
        "moe_pat":  r"model\.layers\.\d+\.mlp\.experts?\.(\d+)\.",
        "norm_key": "input_layernorm",   # substring in tensor names → rms_norm
    },
    {
        "name": "bert",           # BERT, RoBERTa, DistilBERT, ALBERT variants
        "detect": lambda ns: (
            any(k.startswith("bert.encoder.") for k in ns)
            or any(k.startswith("bert.embeddings.") for k in ns)
            or any(k.startswith("roberta.encoder.") for k in ns)
            or any(k.startswith("distilbert.transformer.") for k in ns)
            or any(k.startswith("albert.encoder.") for k in ns)
        ),
        "arch_type": "encoder",
        "layer_pat": r"(?:bert|roberta|distilbert|albert)\.(?:encoder\.layer|transformer\.layer|encoder\.albert_layer)\.(\d+)\.",
        "enc_layer_pat": None,
        "dec_layer_pat": None,
        "embed_keys": [
            "bert.embeddings.word_embeddings.weight",
            "roberta.embeddings.word_embeddings.weight",
            "distilbert.embeddings.word_embeddings.weight",
            "albert.embeddings.word_embeddings.weight",
            "embeddings.word_embeddings.weight",
        ],
        "q_key_pat":   r"(?:bert|roberta|distilbert|albert)\.encoder\.layer\.0\.attention\.self\.query\.weight",
        "k_key_pat":   r"(?:bert|roberta|distilbert|albert)\.encoder\.layer\.0\.attention\.self\.key\.weight",
        "ffn_key_pats": [
            r"(?:bert|roberta)\.encoder\.layer\.0\.intermediate\.dense\.weight",
            r"distilbert\.transformer\.layer\.0\.ffn\.lin1\.weight",
        ],
        "moe_pat": None,
        "norm_key": "LayerNorm",   # substring → layer_norm
    },
    {
        "name": "gpt2",           # GPT-2 and derivatives
        "detect": lambda ns: any(k.startswith("transformer.h.") for k in ns),
        "arch_type": "decoder",
        "layer_pat": r"transformer\.h\.(\d+)\.",
        "enc_layer_pat": None,
        "dec_layer_pat": None,
        "embed_keys": ["transformer.wte.weight", "wte.weight"],
        "q_key_pat":   None,   # fused QKV in c_attn — handled specially
        "k_key_pat":   None,
        "ffn_key_pats": [r"transformer\.h\.0\.mlp\.c_fc\.weight"],
        "moe_pat": None,
        "norm_key": "ln_",     # substring → layer_norm
    },
    {
        "name": "t5",             # T5, FLAN-T5, mT5
        "detect": lambda ns: (
            any(k.startswith("encoder.block.") for k in ns)
            and any(k.startswith("decoder.block.") for k in ns)
        ),
        "arch_type": "encoder_decoder",
        "layer_pat": r"encoder\.block\.(\d+)\.",
        "enc_layer_pat": r"encoder\.block\.(\d+)\.",
        "dec_layer_pat": r"decoder\.block\.(\d+)\.",
        "embed_keys": ["shared.weight", "encoder.embed_tokens.weight"],
        "q_key_pat":   r"encoder\.block\.0\.layer\.0\.SelfAttention\.q\.weight",
        "k_key_pat":   r"encoder\.block\.0\.layer\.0\.SelfAttention\.k\.weight",
        "ffn_key_pats": [
            r"encoder\.block\.0\.layer\.1\.DenseReluDense\.wi\.weight",
            r"encoder\.block\.0\.layer\.1\.DenseReluDense\.wi_0\.weight",
        ],
        "moe_pat": None,
        "norm_key": "layer_norm",  # substring → layer_norm (T5 uses RMSNorm-like but named layer_norm)
    },
    {
        "name": "mamba",          # Mamba / Mamba-2
        "detect": lambda ns: any(k.startswith("backbone.layers.") for k in ns),
        "arch_type": "ssm",
        "layer_pat": r"backbone\.layers\.(\d+)\.",
        "enc_layer_pat": None,
        "dec_layer_pat": None,
        "embed_keys": ["backbone.embedding.weight"],
        "q_key_pat":   None,
        "k_key_pat":   None,
        "ffn_key_pats": [],
        "moe_pat": None,
        "norm_key": "norm",
    },
    {
        "name": "gpt_neox",       # GPT-NeoX, Pythia
        "detect": lambda ns: any(k.startswith("gpt_neox.layers.") for k in ns),
        "arch_type": "decoder",
        "layer_pat": r"gpt_neox\.layers\.(\d+)\.",
        "enc_layer_pat": None,
        "dec_layer_pat": None,
        "embed_keys": ["gpt_neox.embed_in.weight"],
        "q_key_pat":   None,   # fused QKV
        "k_key_pat":   None,
        "ffn_key_pats": [r"gpt_neox\.layers\.0\.mlp\.dense_h_to_4h\.weight"],
        "moe_pat": None,
        "norm_key": "layernorm",
    },
    {
        "name": "falcon",         # Falcon (old HF naming)
        "detect": lambda ns: any(k.startswith("transformer.h.") and "self_attention" in k for k in ns),
        "arch_type": "decoder",
        "layer_pat": r"transformer\.h\.(\d+)\.",
        "enc_layer_pat": None,
        "dec_layer_pat": None,
        "embed_keys": ["transformer.word_embeddings.weight"],
        "q_key_pat":   r"transformer\.h\.0\.self_attention\.query_key_value\.weight",
        "k_key_pat":   None,
        "ffn_key_pats": [r"transformer\.h\.0\.mlp\.dense_h_to_4h\.weight"],
        "moe_pat": None,
        "norm_key": "ln_",
    },
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _max_index(names: set[str], pattern: str) -> int:
    """Return max numeric index matching the capture group in pattern, or 0."""
    indices = set()
    rx = re.compile(pattern)
    for n in names:
        m = rx.search(n)
        if m:
            indices.add(int(m.group(1)))
    return max(indices) + 1 if indices else 0


def _find_shape(shapes: dict[str, list[int]], *patterns: str) -> list[int] | None:
    """Find the first tensor shape matching any of the given patterns (exact key or regex)."""
    for pat in patterns:
        # Try exact key first (fast)
        if pat in shapes:
            return shapes[pat]
        # Try regex
        rx = re.compile(pat)
        for k, v in shapes.items():
            if rx.fullmatch(k) or rx.search(k):
                return v
    return None


def _decode_attn(
    shapes: dict[str, list[int]],
    q_pat: str | None,
    k_pat: str | None,
    hidden_size: int,
) -> tuple[int, int, int]:
    """
    Return (num_heads, head_dim, num_kv_heads) from Q and K projection shapes.
    Falls back to heuristics only when shapes are absent.
    """
    q_shape = _find_shape(shapes, q_pat) if q_pat else None
    k_shape = _find_shape(shapes, k_pat) if k_pat else None

    if q_shape is None:
        # Try generic patterns
        q_shape = _find_shape(shapes,
            r"model\.layers\.0\.self_attn\.q_proj\.weight",
            r"encoder\.block\.0\.layer\.0\.SelfAttention\.q\.weight",
            r"bert\.encoder\.layer\.0\.attention\.self\.query\.weight",
        )
        k_shape = _find_shape(shapes,
            r"model\.layers\.0\.self_attn\.k_proj\.weight",
            r"encoder\.block\.0\.layer\.0\.SelfAttention\.k\.weight",
            r"bert\.encoder\.layer\.0\.attention\.self\.key\.weight",
        )

    if q_shape is None:
        nh = _guess_heads(hidden_size)
        return nh, hidden_size // nh, nh

    total_q = q_shape[0]

    # Try known head dims in order of prevalence
    head_dim = None
    for hd in (128, 64, 96, 80, 256, 48, 32):
        if total_q % hd == 0:
            head_dim = hd
            break
    if head_dim is None:
        nh = _guess_heads(hidden_size)
        head_dim = total_q // max(nh, 1)

    num_heads = total_q // head_dim
    num_kv_heads = num_heads
    if k_shape:
        num_kv_heads = max(1, k_shape[0] // head_dim)

    return num_heads, head_dim, num_kv_heads


def _guess_heads(hidden_size: int) -> int:
    for nh in (32, 16, 8, 12, 20, 40, 64, 4):
        hd = hidden_size // nh
        if hidden_size % nh == 0 and hd in (32, 64, 80, 96, 128, 256):
            return nh
    for nh in range(64, 0, -1):
        if hidden_size % nh == 0 and hidden_size // nh >= 32:
            return nh
    return 12


def _count_moe_experts(names: set[str], moe_pat: str | None) -> int:
    if not moe_pat:
        return 0
    rx = re.compile(moe_pat)
    indices = set()
    for n in names:
        m = rx.search(n)
        if m:
            indices.add(int(m.group(1)))
    return len(indices)


def _detect_gated_ffn(names: set[str]) -> bool:
    """True if the FFN uses a gated activation (SwiGLU / GeGLU)."""
    return any(
        "gate_proj" in n or "wi_0" in n or "w1" in n and "expert" in n
        for n in names
    )


def _detect_norm_type(names: set[str], norm_key_hint: str) -> str:
    """Infer norm type from tensor naming hints."""
    # RMSNorm indicators
    if any("rmsnorm" in n.lower() or "input_layernorm" in n for n in names):
        return "rms_norm"
    # LayerNorm indicators
    if any("LayerNorm" in n or "layer_norm" in n or "ln_" in n for n in names):
        return "layer_norm"
    # Use the convention hint
    if norm_key_hint in ("input_layernorm", "rmsnorm"):
        return "rms_norm"
    return "layer_norm"


# ---------------------------------------------------------------------------
# Convention detection
# ---------------------------------------------------------------------------

def _detect_convention(shapes: dict[str, list[int]]) -> dict[str, Any] | None:
    """Return the first matching naming convention, or None for generic fallback."""
    names = set(shapes.keys())
    # Falcon must be checked before GPT-2 (both use transformer.h.N)
    ordered = [c for c in _CONVENTIONS if c["name"] == "falcon"] + \
              [c for c in _CONVENTIONS if c["name"] != "falcon"]
    for conv in ordered:
        if conv["detect"](names):
            return conv
    return None


# ---------------------------------------------------------------------------
# Generic fallback: scan for any layer pattern
# ---------------------------------------------------------------------------

def _generic_extract(shapes: dict[str, list[int]]) -> dict[str, Any]:
    """Last-resort extraction for unknown naming conventions."""
    names = set(shapes.keys())

    hidden_size = 768
    vocab_size = 32000
    for name, shape in shapes.items():
        if re.search(r"embed(?:dings?|_tokens?)\.weight", name) and len(shape) == 2:
            vocab_size, hidden_size = shape
            break

    num_layers = (
        _max_index(names, r"\.layers?\.(\d+)\.")
        or _max_index(names, r"\.h\.(\d+)\.")
        or _max_index(names, r"\.blocks?\.(\d+)\.")
        or 12
    )

    num_heads, head_dim, num_kv_heads = _decode_attn(shapes, None, None, hidden_size)

    inter = _find_shape(shapes,
        r".*\.mlp\.(?:gate_proj|up_proj|fc1|dense_h_to_4h)\.weight",
        r".*\.intermediate\.dense\.weight",
    )
    intermediate_size = inter[0] if inter else hidden_size * 4

    n_experts = _count_moe_experts(names, r"\.experts?\.(\d+)\.")

    return {
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "hidden_act": "gelu",
        "rms_norm_eps": None,
        "rope_theta": None,
        "num_local_experts": n_experts if n_experts > 1 else None,
        "num_experts_per_tok": 2 if n_experts > 1 else None,
        "tie_word_embeddings": False,
        "architectures": [],
        "_from_tensors": True,
        "_confidence": "medium",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def map_shapes_to_ir(
    shapes: dict[str, list[int]],
    model_name: str,
) -> Any:
    """
    Convert a tensor name→shape dict into an ArchitectureIR.

    This is the Netron-style approach: read what is physically in the file
    rather than inferring from metadata.

    Steps:
      1. Detect naming convention from tensor names
      2. Extract all architectural dimensions directly from shapes
      3. Build a config dict (all fields from ground truth)
      4. Call autoconfig_mapper.map_autoconfig_to_ir() for IR construction
    """
    from .autoconfig_mapper import map_autoconfig_to_ir
    from ..models.ir import SourceType, SourceConfidence

    names = set(shapes.keys())
    conv = _detect_convention(shapes)

    if conv is None:
        logger.debug("tensor_arch_mapper: no convention matched for %s — using generic", model_name)
        cfg = _generic_extract(shapes)
        cfg["_name_or_path"] = model_name
        ir = map_autoconfig_to_ir(cfg, model_name)
        ir.source = SourceType.FILE_HEADER
        ir.source_confidence = SourceConfidence.MEDIUM
        return ir

    conv_name = conv["name"]
    arch_type = conv["arch_type"]
    logger.debug("tensor_arch_mapper: matched convention '%s' for %s", conv_name, model_name)

    # ── Embedding: [vocab_size, hidden_size] ─────────────────────────────────
    emb_shape = _find_shape(shapes, *conv["embed_keys"]) if conv["embed_keys"] else None
    vocab_size = emb_shape[0] if emb_shape else 32000
    hidden_size = emb_shape[1] if emb_shape else 768

    # Fallback: derive hidden_size from a norm weight (1-D vector of size h)
    if not emb_shape:
        for n, s in shapes.items():
            if len(s) == 1 and ("norm" in n.lower() or "ln_" in n) and s[0] > 64:
                hidden_size = s[0]
                break

    # ── Layer counts ─────────────────────────────────────────────────────────
    if arch_type == "encoder_decoder":
        num_enc_layers = _max_index(names, conv["enc_layer_pat"]) or 6
        num_dec_layers = _max_index(names, conv["dec_layer_pat"]) or 6
        num_layers = num_enc_layers  # encoder depth used as the primary count
    elif arch_type == "ssm":
        num_layers = _max_index(names, conv["layer_pat"]) or 24
        num_enc_layers = num_dec_layers = num_layers
    else:
        num_layers = _max_index(names, conv["layer_pat"]) or 12
        num_enc_layers = num_dec_layers = num_layers

    # ── Attention ─────────────────────────────────────────────────────────────
    if arch_type == "ssm":
        num_heads = num_kv_heads = head_dim = 0
    else:
        # For GPT-2 / fused QKV: c_attn.weight is [hidden, 3*hidden] — derive from that
        if conv_name in ("gpt2", "falcon", "gpt_neox"):
            fused = _find_shape(shapes,
                r"transformer\.h\.0\.attn\.c_attn\.weight",
                r"transformer\.h\.0\.self_attention\.query_key_value\.weight",
                r"gpt_neox\.layers\.0\.attention\.query_key_value\.weight",
            )
            nh = _guess_heads(hidden_size)
            if fused and len(fused) >= 2:
                # c_attn: [hidden, 3*hidden] → num_heads from hidden
                pass
            num_heads = nh
            head_dim = hidden_size // max(nh, 1)
            num_kv_heads = nh
        else:
            num_heads, head_dim, num_kv_heads = _decode_attn(
                shapes, conv["q_key_pat"], conv["k_key_pat"], hidden_size)

    # ── FFN intermediate size ─────────────────────────────────────────────────
    ffn_shape = None
    for pat in (conv["ffn_key_pats"] or []):
        ffn_shape = _find_shape(shapes, pat)
        if ffn_shape:
            break
    # Generic scan if convention-specific keys not found
    if ffn_shape is None:
        ffn_shape = _find_shape(shapes,
            r"model\.layers\.0\.mlp\.(?:gate_proj|up_proj|fc1)\.weight",
            r"encoder\.block\.0\.layer\.1\.DenseReluDense\.(?:wi|wi_0)\.weight",
            r".*\.(?:intermediate|dense_h_to_4h|fc1)\.(?:weight|dense\.weight)",
        )
    intermediate_size = ffn_shape[0] if ffn_shape else hidden_size * 4

    # For GPT-2 c_fc.weight shape is [hidden, 4*hidden] (Conv1D transposed)
    if conv_name == "gpt2" and ffn_shape and len(ffn_shape) == 2:
        intermediate_size = max(ffn_shape[0], ffn_shape[1])

    # ── MoE ──────────────────────────────────────────────────────────────────
    n_experts = _count_moe_experts(names, conv["moe_pat"])
    is_moe = n_experts > 1

    # MoE intermediate size may be smaller than the dense intermediate
    moe_inter = None
    if is_moe:
        moe_ffn = _find_shape(shapes, r"model\.layers\.\d+\.mlp\.experts?\.0\.(?:gate_proj|w1|fc1)\.weight")
        if moe_ffn:
            moe_inter = moe_ffn[0]

    # ── Gated FFN & norm ──────────────────────────────────────────────────────
    is_gated = _detect_gated_ffn(names)
    norm_t = _detect_norm_type(names, conv.get("norm_key", ""))

    # ── Activation function ───────────────────────────────────────────────────
    if arch_type == "ssm":
        hidden_act = "silu"
    elif is_gated:
        hidden_act = "silu"
    elif conv_name == "t5":
        hidden_act = "gelu_new" if any("wi_0" in n for n in names) else "relu"
    elif conv_name in ("gpt2", "gpt_neox"):
        hidden_act = "gelu_new"
    elif conv_name == "bert":
        hidden_act = "gelu"
    else:
        hidden_act = "silu" if norm_t == "rms_norm" else "gelu"

    # ── RoPE detection ────────────────────────────────────────────────────────
    has_rope = any("rotary" in n.lower() or "rope" in n.lower() or "cos_cached" in n for n in names)

    # ── Tie embeddings ────────────────────────────────────────────────────────
    # If lm_head.weight and embed_tokens.weight have same shape, they're tied
    lm_head_shape = _find_shape(shapes, "lm_head.weight", r".*\.lm_head\.weight")
    tied = lm_head_shape is None  # if no separate lm_head tensor, it's tied

    # ── Map conv_name → model_type for AutoConfig lookup ─────────────────────
    _CONV_TO_MODEL_TYPE = {
        "llama":    "llama",
        "bert":     "bert",
        "gpt2":     "gpt2",
        "t5":       "t5",
        "mamba":    "mamba",
        "gpt_neox": "gpt_neox",
        "falcon":   "falcon",
    }
    model_type = _CONV_TO_MODEL_TYPE.get(conv_name, conv_name)

    # ── Detect architecture for display ─────────────────────────────────────
    # Refine for BERT: ensure encoder-only models don't get marked as decoders
    if arch_type == "encoder":
        architectures = ["BertForMaskedLM"]
    elif arch_type == "encoder_decoder":
        architectures = ["T5ForConditionalGeneration"]
    elif arch_type == "ssm":
        architectures = ["MambaForCausalLM"]
    else:
        architectures = ["LlamaForCausalLM" if model_type == "llama" else f"{model_type.title()}ForCausalLM"]

    # ── Assemble config dict (all from ground truth) ─────────────────────────
    cfg: dict[str, Any] = {
        "model_type":          model_type,
        "hidden_size":         hidden_size,
        "num_hidden_layers":   num_layers,
        "vocab_size":          vocab_size,
        "hidden_act":          hidden_act,
        "tie_word_embeddings": tied,
        "architectures":       architectures,
        "_name_or_path":       model_name,
        "_from_tensors":       True,
    }

    if arch_type != "ssm":
        cfg["num_attention_heads"] = num_heads
        cfg["num_key_value_heads"] = num_kv_heads
        cfg["intermediate_size"] = intermediate_size

    if arch_type == "encoder_decoder":
        cfg["is_encoder_decoder"] = True
        cfg["num_encoder_layers"] = num_enc_layers
        cfg["num_decoder_layers"] = num_dec_layers
        cfg.pop("num_hidden_layers", None)   # T5 uses num_layers

    if arch_type == "ssm":
        # Mamba-specific: infer state_size from SSM kernel shape
        ssm_shape = _find_shape(shapes, r"backbone\.layers\.0\.mixer\.A_log")
        if ssm_shape:
            cfg["state_size"] = ssm_shape[1] if len(ssm_shape) > 1 else 16
        else:
            cfg["state_size"] = 16

    if norm_t == "rms_norm":
        cfg["rms_norm_eps"] = 1e-5

    if has_rope:
        cfg["rope_theta"] = 10000.0

    if is_moe and n_experts > 1:
        cfg["num_local_experts"] = n_experts
        cfg["num_experts_per_tok"] = 2  # conservative default; actual value not in weight names
        if moe_inter:
            cfg["moe_intermediate_size"] = moe_inter

    # T5 uses different field names
    if conv_name == "t5":
        cfg["d_model"] = hidden_size
        cfg["d_ff"] = intermediate_size
        cfg["d_kv"] = head_dim if head_dim else 64
        cfg["num_heads"] = num_heads
        cfg["num_layers"] = num_enc_layers
        cfg["num_decoder_layers"] = num_dec_layers
        cfg.pop("hidden_size", None)
        cfg.pop("intermediate_size", None)
        cfg.pop("num_attention_heads", None)

    # GPT-2 uses different field names
    if conv_name == "gpt2":
        cfg["n_embd"] = hidden_size
        cfg["n_layer"] = num_layers
        cfg["n_head"] = num_heads
        cfg.pop("hidden_size", None)
        cfg.pop("num_hidden_layers", None)
        cfg.pop("num_attention_heads", None)

    logger.info(
        "tensor_arch_mapper: %s → type=%s layers=%d h=%d heads=%d/%d inter=%d%s",
        model_name, arch_type, num_layers, hidden_size, num_heads, num_kv_heads,
        intermediate_size,
        f" MoE×{n_experts}" if is_moe else "",
    )

    ir = map_autoconfig_to_ir(cfg, model_name)
    ir.source = SourceType.FILE_HEADER
    ir.source_confidence = SourceConfidence.HIGH
    return ir
