"""
Phase 2 file parser — infer an ArchitectureIR from an uploaded model file.

Supported formats (zero extra dependencies):
  .safetensors   SafeTensors header-only parse (reads <1KB, never loads weights)
  .gguf          GGUF metadata-header parse (llama.cpp format, skips tensor data)
  .json          HuggingFace config.json passthrough
  .bin/.pt/.pth  PyTorch state-dict shape inference via torch (if installed)
  .onnx          ONNX protobuf graph parse — extracts initializer names/shapes
                 using the `onnx` package if available, else a lightweight
                 protobuf field scanner (no deps)

Architecture inference priority:
  1. GGUF  → metadata keys contain the full config directly
  2. SafeTensors / PyTorch / ONNX → tensor name + shape → reconstruct pseudo config.json
  3. config.json → pass straight through to arch_parser
"""

from __future__ import annotations

import json
import re
import struct
import tempfile
import os
from pathlib import Path
from typing import Any

from ..arch_parser import parse_hf_config
from ...models.ir import SourceType, SourceConfidence

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = frozenset({
    ".safetensors", ".gguf", ".json", ".bin", ".pt", ".pth", ".onnx"
})


def parse_uploaded_file(path: Path, filename: str) -> dict:
    """
    Parse a model file saved at *path* and return a serialised ArchitectureIR
    dict.  Raises ValueError with a human-readable message on unsupported or
    malformed input.

    The file is read lazily — only the metadata / shape header is loaded,
    never the full tensor data.
    """
    ext = Path(filename).suffix.lower()
    stem = Path(filename).stem

    if ext == ".safetensors":
        data = _read_header_bytes(path, 32 * 1024 * 1024)
        config = _parse_safetensors(data, stem)
    elif ext == ".gguf":
        data = _read_header_bytes(path, 64 * 1024 * 1024)
        config = _parse_gguf(data, stem)
    elif ext == ".json":
        try:
            config = json.loads(path.read_bytes())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e
        if "model_type" not in config:
            raise ValueError(
                "Uploaded JSON does not look like a HuggingFace config.json "
                "(missing required 'model_type' field). "
                "Tip: download config.json from the model repo on huggingface.co."
            )
    elif ext in (".bin", ".pt", ".pth"):
        data = _read_header_bytes(path, 32 * 1024 * 1024)
        config = _parse_pytorch(data, stem)
    elif ext == ".onnx":
        config = _parse_onnx_path(path, stem)
    else:
        exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported format '{ext}'. Supported: {exts}")

    ir = parse_hf_config(config, stem)
    ir.source = SourceType.FILE_HEADER
    ir.source_confidence = SourceConfidence.HIGH
    return ir.model_dump()


def _read_header_bytes(path: Path, max_bytes: int) -> bytes:
    """Read up to max_bytes from the start of a file."""
    with open(path, "rb") as f:
        return f.read(max_bytes)


# ---------------------------------------------------------------------------
# SafeTensors parser  (https://github.com/huggingface/safetensors)
# Header format:
#   8 bytes  uint64-LE  header_length
#   N bytes  UTF-8 JSON { tensor_name: {dtype, shape, data_offsets}, __metadata__: {...} }
# ---------------------------------------------------------------------------

def _parse_safetensors(data: bytes, stem: str) -> dict:
    if len(data) < 8:
        raise ValueError("File too small to be a valid safetensors file.")
    header_len = struct.unpack_from("<Q", data, 0)[0]
    if header_len > len(data) - 8:
        raise ValueError(
            f"safetensors header claims {header_len} bytes but file is only "
            f"{len(data)} bytes. The file may be truncated."
        )
    try:
        header = json.loads(data[8: 8 + header_len])
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse safetensors header JSON: {e}") from e

    shapes: dict[str, list[int]] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if isinstance(info, dict) and "shape" in info:
            shapes[name] = info["shape"]

    if not shapes:
        raise ValueError("safetensors file contains no tensors.")

    return _infer_config_from_shapes(shapes, stem)


# ---------------------------------------------------------------------------
# GGUF parser  (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
# ---------------------------------------------------------------------------

_GGUF_MAGIC = b"GGUF"

# GGUF value type → (struct format, byte size)  for fixed-width types
_GGUF_FMT: dict[int, tuple[str, int]] = {
    0:  ("<B", 1),   # UINT8
    1:  ("<b", 1),   # INT8
    2:  ("<H", 2),   # UINT16
    3:  ("<h", 2),   # INT16
    4:  ("<I", 4),   # UINT32
    5:  ("<i", 4),   # INT32
    6:  ("<f", 4),   # FLOAT32
    7:  ("<B", 1),   # BOOL  (stored as 1 byte)
    10: ("<Q", 8),   # UINT64
    11: ("<q", 8),   # INT64
    12: ("<d", 8),   # FLOAT64
}

_GGUF_ARCH_TO_HF: dict[str, str] = {
    "llama":       "llama",
    "mistral":     "mistral",
    "mixtral":     "mixtral",
    "falcon":      "falcon",
    "gpt2":        "gpt2",
    "gptneox":     "gpt_neox",
    "phi2":        "phi",
    "phi3":        "phi3",
    "gemma":       "gemma",
    "gemma2":      "gemma2",
    "qwen2":       "qwen2",
    "mamba":       "mamba",
    "t5":          "t5",
    "bert":        "bert",
    "roberta":     "roberta",
    "nomic-bert":  "bert",
    "jina-bert-v2":"bert",
    "bloom":       "bloom",
    "starcoder2":  "starcoder2",
    "command-r":   "cohere",
    "internlm2":   "internlm2",
    "olmo":        "olmo",
    "olmo2":       "olmo2",
    "deepseek2":   "llama",
    "granite":     "llama",
}


class _BytesReader:
    """Minimal stateful reader over a bytes object."""
    __slots__ = ("data", "pos")

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def read(self, n: int) -> bytes:
        chunk = self.data[self.pos: self.pos + n]
        if len(chunk) < n:
            raise ValueError(
                f"Unexpected end of data at offset {self.pos} "
                f"(wanted {n}, got {len(chunk)})"
            )
        self.pos += n
        return chunk

    def unpack(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        return struct.unpack_from(fmt, self.read(size))[0]


def _gguf_read_string(r: _BytesReader) -> str:
    length = r.unpack("<Q")
    return r.read(length).decode("utf-8", errors="replace")


def _gguf_skip_value(r: _BytesReader, vtype: int) -> None:
    """Skip a GGUF value without returning it — handles all nested types."""
    if vtype in _GGUF_FMT:
        r.read(_GGUF_FMT[vtype][1])
    elif vtype == 8:  # STRING
        length = r.unpack("<Q")
        r.read(length)
    elif vtype == 9:  # ARRAY
        elem_type = r.unpack("<I")
        count = r.unpack("<Q")
        if elem_type in _GGUF_FMT:
            # Fixed-width elements: skip the whole block at once
            r.read(count * _GGUF_FMT[elem_type][1])
        else:
            for _ in range(count):
                _gguf_skip_value(r, elem_type)
    else:
        raise ValueError(f"Unknown GGUF value type {vtype}")


def _gguf_read_value(r: _BytesReader, vtype: int) -> Any:
    if vtype in _GGUF_FMT:
        fmt, size = _GGUF_FMT[vtype]
        raw = r.read(size)
        val = struct.unpack_from(fmt, raw)[0]
        return bool(val) if vtype == 7 else val
    if vtype == 8:  # STRING
        return _gguf_read_string(r)
    if vtype == 9:  # ARRAY
        elem_type = r.unpack("<I")
        count = r.unpack("<Q")
        # For large string arrays (e.g. tokenizer vocab), read count only and skip content
        if elem_type == 8 and count > 64:
            for _ in range(count):
                _gguf_skip_value(r, 8)
            return count  # return count as proxy (used for vocab_size inference)
        if elem_type in _GGUF_FMT:
            fmt, size = _GGUF_FMT[elem_type]
            block = r.read(count * size)
            return [struct.unpack_from(fmt, block, i * size)[0] for i in range(count)]
        return [_gguf_read_value(r, elem_type) for _ in range(count)]
    raise ValueError(f"Unknown GGUF value type {vtype}")


def _parse_gguf(data: bytes, stem: str) -> dict:
    if len(data) < 24:
        raise ValueError("File too small to be a valid GGUF file.")
    if data[:4] != _GGUF_MAGIC:
        raise ValueError(
            f"Not a valid GGUF file (expected magic 'GGUF', got {data[:4]!r})."
        )

    r = _BytesReader(data)
    r.read(4)  # magic
    version = r.unpack("<I")
    if version not in (1, 2, 3):
        raise ValueError(f"Unsupported GGUF version {version} (expected 1, 2, or 3).")

    n_tensors = r.unpack("<Q")
    n_kv = r.unpack("<Q")

    kv: dict[str, Any] = {}
    for _ in range(n_kv):
        try:
            key = _gguf_read_string(r)
            vtype = r.unpack("<I")
            kv[key] = _gguf_read_value(r, vtype)
        except ValueError as e:
            # Stop parsing if we hit an error mid-metadata; work with what we have
            break

    return _gguf_kv_to_hf_config(kv, stem)


def _gguf_kv_to_hf_config(kv: dict[str, Any], stem: str) -> dict:
    """Convert GGUF metadata key-value pairs to a HuggingFace-compatible config dict."""
    arch_gguf: str = str(kv.get("general.architecture", "")).lower()
    model_type = _GGUF_ARCH_TO_HF.get(arch_gguf, arch_gguf or _guess_family(stem))

    prefix = arch_gguf + "." if arch_gguf else ""

    def _g(key: str, default: Any = None) -> Any:
        return kv.get(prefix + key, kv.get(key, default))

    hidden_size    = int(_g("embedding_length", 4096))
    num_layers     = int(_g("block_count", 32))
    num_heads      = int(_g("attention.head_count", 32))
    num_kv_heads   = int(_g("attention.head_count_kv", num_heads))
    intermediate   = int(_g("feed_forward_length", hidden_size * 4))
    context_len    = int(_g("context_length", 4096))
    head_dim       = int(_g("attention.key_length", hidden_size // max(num_heads, 1)))
    rms_eps        = float(_g("attention.layer_norm_rms_epsilon", 1e-5))

    # vocab_size: direct field (newer GGUF) or count from tokenizer tokens array
    vocab_raw = _g("vocab_size") or _g("tokenizer.ggml.tokens")
    if isinstance(vocab_raw, int):
        vocab_size = vocab_raw
    else:
        vocab_size = 32000  # safe default

    name = kv.get("general.name", stem)

    # --- Base config (LLaMA-style, most common) ---
    config: dict[str, Any] = {
        "model_type":              model_type,
        "hidden_size":             hidden_size,
        "num_hidden_layers":       num_layers,
        "num_attention_heads":     num_heads,
        "num_key_value_heads":     num_kv_heads,
        "intermediate_size":       intermediate,
        "vocab_size":              vocab_size,
        "max_position_embeddings": context_len,
        "rms_norm_eps":            rms_eps,
        "hidden_act":              "silu",
        "name":                    name,
    }

    # --- T5 key remapping ---
    if model_type == "t5":
        config = {
            "model_type":   "t5",
            "d_model":      hidden_size,
            "d_ff":         intermediate,
            "d_kv":         head_dim,
            "num_heads":    num_heads,
            "num_layers":   num_layers,
            "vocab_size":   vocab_size,
            "dense_act_fn": "relu",
            "name":         name,
        }

    # --- GPT-2 key remapping ---
    elif model_type == "gpt2":
        config = {
            "model_type":          "gpt2",
            "n_embd":              hidden_size,
            "n_layer":             num_layers,
            "n_head":              num_heads,
            "n_positions":         context_len,
            "vocab_size":          vocab_size,
            "activation_function": "gelu_new",
            "name":                name,
        }

    # --- BERT key remapping ---
    elif model_type in ("bert", "roberta", "distilbert"):
        config["intermediate_size"] = intermediate
        config["num_hidden_layers"] = num_layers
        config["num_attention_heads"] = num_heads

    # --- Mixtral MoE extras ---
    if model_type == "mixtral":
        config["num_local_experts"]   = int(_g("expert_count", 8))
        config["num_experts_per_tok"] = int(_g("expert_used_count", 2))

    return config


# ---------------------------------------------------------------------------
# PyTorch checkpoint parser
# ---------------------------------------------------------------------------

def _parse_pytorch(data: bytes, stem: str) -> dict:
    """
    Try to infer config from a PyTorch checkpoint.

    Strategy 1: use torch.load with map_location='meta' (shapes only, no data).
    Strategy 2: minimal pickle header scan to extract tensor shapes.
    """
    # --- Strategy 1: torch meta device ---
    try:
        import torch
        import io
        buf = io.BytesIO(data)
        obj = torch.load(buf, map_location="meta", weights_only=True)
        # Unwrap nested checkpoint dicts
        if isinstance(obj, dict):
            for key in ("model", "module", "state_dict", "model_state_dict"):
                if key in obj and isinstance(obj[key], dict):
                    obj = obj[key]
                    break
        shapes = {
            k: list(v.shape)
            for k, v in obj.items()
            if hasattr(v, "shape")
        }
        if shapes:
            return _infer_config_from_shapes(shapes, stem)
    except Exception:
        pass

    # --- Strategy 2: safetensors package for .safetensors mistakenly named .pt ---
    try:
        from safetensors import safe_open
        import io
        shapes = {}
        with safe_open(io.BytesIO(data), framework="pt") as f:
            for k in f.keys():
                shapes[k] = list(f.get_tensor(k).shape)
        if shapes:
            return _infer_config_from_shapes(shapes, stem)
    except Exception:
        pass

    raise ValueError(
        "Could not parse PyTorch checkpoint. "
        "Tip: convert to .safetensors using the HuggingFace safetensors library "
        "(`safetensors.torch.save_file(state_dict, 'model.safetensors')`), "
        "or upload the model's config.json directly."
    )


# ---------------------------------------------------------------------------
# ONNX parser
# ---------------------------------------------------------------------------

# ONNX protobuf field numbers we care about (ModelProto / GraphProto / TensorProto)
# ModelProto:  field 7 = graph (GraphProto)
# GraphProto:  field 1 = node (NodeProto), field 4 = initializer (TensorProto)
# TensorProto: field 1 = dims (int64[]), field 3 = name (string)
# NodeProto:   field 4 = op_type (string), field 6 = attribute (AttributeProto)

def _proto_read_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Decode a protobuf varint starting at pos. Returns (value, new_pos)."""
    result, shift = 0, 0
    while pos < len(data):
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
    raise ValueError("Truncated varint in protobuf")


def _proto_skip_field(data: bytes, pos: int, wire: int) -> int:
    """Skip a single protobuf field value starting at pos."""
    if wire == 0:    # varint
        _, pos = _proto_read_varint(data, pos)
    elif wire == 1:  # 64-bit
        pos += 8
    elif wire == 2:  # length-delimited
        length, pos = _proto_read_varint(data, pos)
        pos += length
    elif wire == 5:  # 32-bit
        pos += 4
    else:
        raise ValueError(f"Unknown protobuf wire type {wire}")
    return pos


def _proto_iter_fields(data: bytes, start: int, end: int):
    """Yield (field_number, wire_type, payload_start, payload_end) for each field."""
    pos = start
    while pos < end:
        tag, pos = _proto_read_varint(data, pos)
        field_num = tag >> 3
        wire      = tag & 0x07
        if wire == 2:  # length-delimited — most useful fields
            length, pos = _proto_read_varint(data, pos)
            yield field_num, wire, pos, pos + length
            pos += length
        else:
            payload_start = pos
            pos = _proto_skip_field(data, pos, wire)
            yield field_num, wire, payload_start, pos


def _onnx_parse_tensor_proto(data: bytes, start: int, end: int) -> tuple[str, list[int]]:
    """Extract (name, dims) from a TensorProto message slice."""
    name = ""
    dims: list[int] = []
    for fnum, wire, ps, pe in _proto_iter_fields(data, start, end):
        if fnum == 1 and wire == 0:   # dims (repeated int64, but stored as varints)
            val, _ = _proto_read_varint(data, ps)
            dims.append(val)
        elif fnum == 1 and wire == 2:  # packed repeated dims
            p = ps
            while p < pe:
                v, p = _proto_read_varint(data, p)
                dims.append(v)
        elif fnum == 3 and wire == 2:  # name
            name = data[ps:pe].decode("utf-8", errors="replace")
    return name, dims


def _parse_onnx_path(path: Path, stem: str) -> dict:
    """
    Parse an ONNX model from a file path.

    Strategy:
      1. `onnx.load(path)` — reliable, memory-maps the file, handles any size.
      2. Fallback lightweight protobuf scanner (no extra deps).
    """
    # --- Strategy 1: onnx package (preferred) ---
    try:
        import onnx  # type: ignore
        model = onnx.load(str(path))
        shapes: dict[str, list[int]] = {
            init.name: list(init.dims)
            for init in model.graph.initializer
            if init.dims  # skip scalars
        }
        # Also collect value_info shapes (graph inputs/outputs with type info)
        for vi in list(model.graph.input) + list(model.graph.value_info):
            try:
                t = vi.type.tensor_type
                dims = [d.dim_value for d in t.shape.dim]
                if vi.name and dims:
                    shapes.setdefault(vi.name, dims)
            except Exception:
                continue
        if not shapes:
            raise ValueError(
                "ONNX model has no weight initializers. "
                "This may be an opset-only or skeleton model."
            )
        return _infer_config_from_shapes(shapes, stem)
    except ImportError:
        pass  # fall through to protobuf scanner
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not parse ONNX model: {e}") from e

    # --- Strategy 2: lightweight protobuf scanner (no deps) ---
    data = _read_header_bytes(path, 128 * 1024 * 1024)
    if len(data) < 4:
        raise ValueError("File too small to be a valid ONNX model.")
    try:
        shapes = _onnx_scan_initializers(data)
    except Exception as e:
        raise ValueError(f"Could not parse ONNX protobuf: {e}") from e
    if not shapes:
        raise ValueError("No weight initializers found in ONNX file.")
    return _infer_config_from_shapes(shapes, stem)


def _onnx_scan_initializers(data: bytes) -> dict[str, list[int]]:
    """
    Lightweight ONNX initializer scanner without the onnx package.

    ModelProto structure:
      field 7  (wire 2) = graph (GraphProto)
        field 4 (wire 2) = initializer (TensorProto)
          field 1 (wire 0 or 2) = dims
          field 3 (wire 2)      = name

    Truncation-safe: stops cleanly if we hit the end of a partial read.
    """
    shapes: dict[str, list[int]] = {}

    # Find the graph field (field 7) in the top-level ModelProto.
    # Iterate defensively — a truncated file will raise ValueError which we catch.
    graph_start = graph_end = -1
    try:
        for fnum, wire, ps, pe in _proto_iter_fields(data, 0, len(data)):
            if fnum == 7 and wire == 2:
                graph_start, graph_end = ps, pe
                break
    except (ValueError, struct.error):
        pass

    if graph_start < 0:
        graph_start, graph_end = 0, len(data)

    # Scan initializers (field 4) inside the graph — stop gracefully on truncation
    try:
        for fnum, wire, ps, pe in _proto_iter_fields(data, graph_start, graph_end):
            if fnum == 4 and wire == 2:
                try:
                    name, dims = _onnx_parse_tensor_proto(data, ps, pe)
                    if name and dims:
                        shapes[name] = dims
                except (ValueError, struct.error):
                    continue  # skip this initializer, keep going
    except (ValueError, struct.error):
        pass  # truncation — use whatever we parsed so far

    return shapes


# ---------------------------------------------------------------------------
# Shape-based architecture inference
# ---------------------------------------------------------------------------

def _infer_config_from_shapes(shapes: dict[str, list[int]], stem: str) -> dict:
    """Reconstruct a HuggingFace config dict from tensor name → shape mapping."""
    names = set(shapes)

    # --- Detect family from tensor name conventions ---
    if any(k.startswith("model.layers.") for k in names):
        return _infer_llama_config(shapes, stem)
    if any(k.startswith("bert.encoder.") or k.startswith("bert.embeddings.") for k in names):
        return _infer_bert_config(shapes, "bert", stem)
    if any(k.startswith("roberta.encoder.") for k in names):
        return _infer_bert_config(shapes, "roberta", stem)
    if any(k.startswith("distilbert.transformer.") for k in names):
        return _infer_bert_config(shapes, "distilbert", stem)
    if any(k.startswith("transformer.h.") for k in names):
        return _infer_gpt2_config(shapes, stem)
    if any(k.startswith("gpt_neox.layers.") for k in names):
        return _infer_gptneox_config(shapes, stem)
    if any(k.startswith("encoder.block.") or k.startswith("decoder.block.") for k in names):
        return _infer_t5_config(shapes, stem)
    if any(k.startswith("backbone.layers.") or "conv1d.weight" in k for k in names):
        return _infer_mamba_config(shapes, stem)
    # Mixtral / Mistral also use model.layers — handled above
    # Qwen: transformer.h → handled above; qwen2 uses model.layers → above

    # Generic fallback
    return _infer_generic_config(shapes, stem)


def _max_layer_index(shapes: dict[str, list[int]], pattern: str) -> int:
    """Return max numeric layer index matching a regex pattern, or 0."""
    indices = set()
    for name in shapes:
        m = re.search(pattern, name)
        if m:
            indices.add(int(m.group(1)))
    return max(indices) + 1 if indices else 0


def _first_shape(shapes: dict[str, list[int]], *keys: str) -> list[int] | None:
    for k in keys:
        if k in shapes:
            return shapes[k]
    return None


def _infer_llama_config(shapes: dict[str, list[int]], stem: str) -> dict:
    num_layers = _max_layer_index(shapes, r"model\.layers\.(\d+)\.")

    emb = _first_shape(shapes, "model.embed_tokens.weight", "embed_tokens.weight")
    vocab_size = emb[0] if emb else 32000
    hidden_size = emb[1] if emb else 4096

    # If emb not found, infer hidden_size from RMSNorm
    if not emb:
        for name, shape in shapes.items():
            if "input_layernorm.weight" in name and len(shape) == 1:
                hidden_size = shape[0]
                break

    # Q and K projections → num_heads, num_kv_heads, head_dim
    q_shape = _first_shape(shapes, "model.layers.0.self_attn.q_proj.weight")
    k_shape = _first_shape(shapes, "model.layers.0.self_attn.k_proj.weight")

    num_heads, head_dim, num_kv_heads = _decode_qk(q_shape, k_shape, hidden_size)

    # FFN intermediate
    ffn = _first_shape(shapes,
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.fc1.weight",
    )
    intermediate_size = ffn[0] if ffn else hidden_size * 4

    model_type = _guess_family(stem, default="llama")

    return {
        "model_type":              model_type,
        "hidden_size":             hidden_size,
        "num_hidden_layers":       num_layers or 32,
        "num_attention_heads":     num_heads,
        "num_key_value_heads":     num_kv_heads,
        "intermediate_size":       intermediate_size,
        "vocab_size":              vocab_size,
        "max_position_embeddings": 4096,
        "hidden_act":              "silu",
    }


def _infer_bert_config(shapes: dict[str, list[int]], model_type: str, stem: str) -> dict:
    prefix = model_type  # "bert" / "roberta" / "distilbert"
    num_layers = _max_layer_index(shapes, rf"{prefix}\.(?:encoder\.layer|transformer\.layer)\.(\d+)\.")

    emb = _first_shape(shapes,
        f"{prefix}.embeddings.word_embeddings.weight",
        "embeddings.word_embeddings.weight",
    )
    vocab_size  = emb[0] if emb else 30522
    hidden_size = emb[1] if emb else 768

    q = _first_shape(shapes,
        f"{prefix}.encoder.layer.0.attention.self.query.weight",
        f"{prefix}.transformer.layer.0.attention.q_lin.weight",
        "encoder.layer.0.attention.self.query.weight",
    )
    num_heads = _guess_num_heads(hidden_size)
    if q and len(q) == 2:
        # weight is [hidden_size, hidden_size] — heads inferred from config
        pass

    ffn = _first_shape(shapes,
        f"{prefix}.encoder.layer.0.intermediate.dense.weight",
        f"{prefix}.transformer.layer.0.ffn.lin1.weight",
        "encoder.layer.0.intermediate.dense.weight",
    )
    intermediate_size = ffn[0] if ffn else hidden_size * 4

    return {
        "model_type":           model_type,
        "hidden_size":          hidden_size,
        "num_hidden_layers":    num_layers or 12,
        "num_attention_heads":  num_heads,
        "intermediate_size":    intermediate_size,
        "vocab_size":           vocab_size,
        "max_position_embeddings": 512,
        "hidden_act":           "gelu",
    }


def _infer_gpt2_config(shapes: dict[str, list[int]], stem: str) -> dict:
    num_layers = _max_layer_index(shapes, r"transformer\.h\.(\d+)\.")

    emb = _first_shape(shapes, "wte.weight", "transformer.wte.weight")
    vocab_size  = emb[0] if emb else 50257
    hidden_size = emb[1] if emb else 768

    ffn = _first_shape(shapes, "transformer.h.0.mlp.c_fc.weight")
    # c_fc.weight shape is [hidden, 4*hidden] in Conv1D form
    intermediate_size = (ffn[1] if ffn and len(ffn) == 2 else hidden_size * 4)

    num_heads = _guess_num_heads(hidden_size)

    return {
        "model_type":          "gpt2",
        "n_embd":              hidden_size,
        "n_layer":             num_layers or 12,
        "n_head":              num_heads,
        "n_positions":         1024,
        "vocab_size":          vocab_size,
        "activation_function": "gelu_new",
    }


def _infer_gptneox_config(shapes: dict[str, list[int]], stem: str) -> dict:
    num_layers = _max_layer_index(shapes, r"gpt_neox\.layers\.(\d+)\.")

    emb = _first_shape(shapes, "gpt_neox.embed_in.weight")
    vocab_size  = emb[0] if emb else 50432
    hidden_size = emb[1] if emb else 768

    q_shape = _first_shape(shapes, "gpt_neox.layers.0.attention.query_key_value.weight")
    num_heads = _guess_num_heads(hidden_size)

    return {
        "model_type":      "gpt_neox",
        "hidden_size":     hidden_size,
        "num_hidden_layers": num_layers or 6,
        "num_attention_heads": num_heads,
        "intermediate_size": hidden_size * 4,
        "vocab_size":      vocab_size,
        "max_position_embeddings": 2048,
        "hidden_act":      "gelu",
    }


def _infer_t5_config(shapes: dict[str, list[int]], stem: str) -> dict:
    num_layers = _max_layer_index(shapes, r"encoder\.block\.(\d+)\.")
    num_dec_layers = _max_layer_index(shapes, r"decoder\.block\.(\d+)\.")

    emb = _first_shape(shapes, "shared.weight", "encoder.embed_tokens.weight")
    vocab_size = emb[0] if emb else 32128
    d_model    = emb[1] if emb else 512

    q = _first_shape(shapes, "encoder.block.0.layer.0.SelfAttention.q.weight")
    d_kv = (q[0] // _guess_num_heads(d_model)) if q else 64

    ffn = _first_shape(shapes, "encoder.block.0.layer.1.DenseReluDense.wi.weight",
                       "encoder.block.0.layer.1.DenseReluDense.wi_0.weight")
    d_ff = ffn[0] if ffn else d_model * 4

    is_gated = any(
        "wi_0" in k or "wi_1" in k for k in shapes
    )

    return {
        "model_type":      "t5",
        "d_model":         d_model,
        "d_ff":            d_ff,
        "d_kv":            d_kv,
        "num_heads":       _guess_num_heads(d_model),
        "num_layers":      num_layers or 6,
        "num_decoder_layers": num_dec_layers or num_layers or 6,
        "vocab_size":      vocab_size,
        "dense_act_fn":    "gelu_new" if is_gated else "relu",
    }


def _infer_mamba_config(shapes: dict[str, list[int]], stem: str) -> dict:
    num_layers = _max_layer_index(shapes, r"backbone\.layers\.(\d+)\.")

    emb = _first_shape(shapes, "backbone.embedding.weight")
    vocab_size = emb[0] if emb else 50280
    d_model    = emb[1] if emb else 2560

    return {
        "model_type": "mamba",
        "d_model":    d_model,
        "n_layer":    num_layers or 64,
        "vocab_size": vocab_size,
    }


def _infer_generic_config(shapes: dict[str, list[int]], stem: str) -> dict:
    """Last-resort: extract whatever we can."""
    # Try common embedding key patterns
    hidden_size = 768
    vocab_size  = 30522
    num_layers  = 0

    for name, shape in shapes.items():
        if re.search(r"embed(?:dings?|_tokens?)\.weight", name) and len(shape) == 2:
            vocab_size, hidden_size = shape[0], shape[1]
            break

    num_layers = (
        _max_layer_index(shapes, r"\.layers?\.(\d+)\.")
        or _max_layer_index(shapes, r"\.h\.(\d+)\.")
        or _max_layer_index(shapes, r"\.blocks?\.(\d+)\.")
        or 12
    )

    model_type = _guess_family(stem, default="bert")

    return {
        "model_type":          model_type,
        "hidden_size":         hidden_size,
        "num_hidden_layers":   num_layers,
        "num_attention_heads": _guess_num_heads(hidden_size),
        "intermediate_size":   hidden_size * 4,
        "vocab_size":          vocab_size,
        "max_position_embeddings": 512,
        "hidden_act":          "gelu",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_qk(
    q_shape: list[int] | None,
    k_shape: list[int] | None,
    hidden_size: int,
) -> tuple[int, int, int]:
    """Decode num_heads, head_dim, num_kv_heads from Q and K projection shapes."""
    # q_proj.weight: [num_heads * head_dim, hidden_size]
    # k_proj.weight: [num_kv_heads * head_dim, hidden_size]
    if q_shape is None:
        num_heads = _guess_num_heads(hidden_size)
        head_dim  = hidden_size // num_heads
        return num_heads, head_dim, num_heads

    total_q = q_shape[0]
    # Try common head_dims: 128 (LLaMA), 64 (BERT), 96 (GPT-3-like), 80, 256
    for hd in (128, 64, 96, 80, 256, 32):
        if total_q % hd == 0:
            num_heads = total_q // hd
            head_dim  = hd
            break
    else:
        # Fallback: pick plausible num_heads first
        num_heads = _guess_num_heads(hidden_size)
        head_dim  = total_q // max(num_heads, 1)

    num_kv_heads = num_heads
    if k_shape:
        num_kv_heads = k_shape[0] // max(head_dim, 1)

    return num_heads, head_dim, num_kv_heads


def _guess_num_heads(hidden_size: int) -> int:
    """Pick a plausible num_heads for a given hidden_size."""
    for nh in (32, 16, 8, 12, 20, 40, 64, 4):
        if hidden_size % nh == 0 and (hidden_size // nh) in (32, 64, 80, 96, 128, 256):
            return nh
    # Fallback: largest divisor that gives head_dim >= 32
    for nh in range(64, 0, -1):
        if hidden_size % nh == 0 and hidden_size // nh >= 32:
            return nh
    return 12


_FAMILY_HINTS: list[tuple[str, str]] = [
    ("llama",    "llama"),  ("mistral",   "mistral"), ("mixtral",  "mixtral"),
    ("qwen",     "qwen2"),  ("gemma",     "gemma"),   ("falcon",   "falcon"),
    ("phi",      "phi3"),   ("gpt2",      "gpt2"),    ("gpt-2",    "gpt2"),
    ("neox",     "gpt_neox"),("bert",     "bert"),    ("roberta",  "roberta"),
    ("t5",       "t5"),     ("flan",      "t5"),      ("mamba",    "mamba"),
    ("internlm", "internlm2"),("olmo",    "olmo"),    ("deepseek", "llama"),
    ("vicuna",   "llama"),  ("hermes",    "llama"),   ("solar",    "llama"),
    ("openchat", "llama"),  ("zephyr",    "mistral"), ("tinyllama","llama"),
]


def _guess_family(stem: str, default: str = "llama") -> str:
    s = stem.lower()
    for hint, family in _FAMILY_HINTS:
        if hint in s:
            return family
    return default
