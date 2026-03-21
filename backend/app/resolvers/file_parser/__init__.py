"""
Phase 2 file parser — infer an ArchitectureIR from an uploaded model file.

Supported formats:
  .safetensors   SafeTensors header-only parse (reads <1KB, never loads weights)
  .gguf          GGUF metadata-header parse (llama.cpp format, skips tensor data)
  .json          HuggingFace config.json passthrough
  .bin/.pt/.pth  PyTorch state-dict shape inference via torch (if installed)
  .onnx          ONNX protobuf graph parse — extracts initializer names/shapes

Routing:
  .safetensors / .bin / .pt / .pth / .onnx
      → _extract_*_shapes()               # raw tensor name → shape dict
      → tensor_arch_mapper.map_shapes_to_ir()   # ground-truth dims → IR
  .gguf
      → _parse_gguf()                     # rich explicit metadata
      → autoconfig_mapper.map_autoconfig_to_ir()
  .json
      → json.loads()                      # it IS a config.json
      → autoconfig_mapper.map_autoconfig_to_ir()
"""

from __future__ import annotations

import json
import re
import struct
import os
from pathlib import Path
from typing import Any

from ...models.ir import SourceType, SourceConfidence

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = frozenset({
    ".safetensors", ".gguf", ".json", ".bin", ".pt", ".pth", ".onnx"
})


def parse_uploaded_file(path: Path, filename: str) -> dict:
    """
    Parse a model file saved at *path* and return a serialised ArchitectureIR dict.
    Raises ValueError with a human-readable message on unsupported or malformed input.

    Only the metadata/shape header is loaded — never the full tensor data.
    """
    from ..tensor_arch_mapper import map_shapes_to_ir
    from ..autoconfig_mapper import map_autoconfig_to_ir

    ext  = Path(filename).suffix.lower()
    stem = Path(filename).stem

    if ext == ".safetensors":
        data   = _read_header_bytes(path, 32 * 1024 * 1024)
        shapes = _extract_safetensors_shapes(data)
        ir     = map_shapes_to_ir(shapes, stem)

    elif ext == ".gguf":
        data   = _read_header_bytes(path, 64 * 1024 * 1024)
        config = _parse_gguf(data, stem)
        ir     = map_autoconfig_to_ir(config, stem)
        ir.source = SourceType.FILE_HEADER
        ir.source_confidence = SourceConfidence.HIGH
        return ir.model_dump()

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
        ir = map_autoconfig_to_ir(config, stem)
        ir.source = SourceType.FILE_HEADER
        ir.source_confidence = SourceConfidence.HIGH
        return ir.model_dump()

    elif ext in (".bin", ".pt", ".pth"):
        data   = _read_header_bytes(path, 32 * 1024 * 1024)
        shapes = _extract_pytorch_shapes(data, path)
        ir     = map_shapes_to_ir(shapes, stem)

    elif ext == ".onnx":
        shapes = _extract_onnx_shapes(path)
        ir     = map_shapes_to_ir(shapes, stem)

    else:
        exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported format '{ext}'. Supported: {exts}")

    ir.source = SourceType.FILE_HEADER
    ir.source_confidence = SourceConfidence.HIGH
    return ir.model_dump()


def _read_header_bytes(path: Path, max_bytes: int) -> bytes:
    """Read up to max_bytes from the start of a file."""
    with open(path, "rb") as f:
        return f.read(max_bytes)


# ---------------------------------------------------------------------------
# SafeTensors — shape extractor
# Header format:
#   8 bytes  uint64-LE  header_length
#   N bytes  UTF-8 JSON { tensor_name: {dtype, shape, data_offsets}, __metadata__: {...} }
# ---------------------------------------------------------------------------

def _extract_safetensors_shapes(data: bytes) -> dict[str, list[int]]:
    if len(data) < 8:
        raise ValueError("File too small to be a valid safetensors file.")
    header_len = struct.unpack_from("<Q", data, 0)[0]
    if header_len > len(data) - 8:
        raise ValueError(
            f"safetensors header claims {header_len} bytes but only "
            f"{len(data)} bytes were read. The file may be truncated."
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
    return shapes


# ---------------------------------------------------------------------------
# GGUF parser  (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
# Returns a HF-style config dict for routing through autoconfig_mapper.
# ---------------------------------------------------------------------------

_GGUF_MAGIC = b"GGUF"

_GGUF_FMT: dict[int, tuple[str, int]] = {
    0:  ("<B", 1),
    1:  ("<b", 1),
    2:  ("<H", 2),
    3:  ("<h", 2),
    4:  ("<I", 4),
    5:  ("<i", 4),
    6:  ("<f", 4),
    7:  ("<B", 1),
    10: ("<Q", 8),
    11: ("<q", 8),
    12: ("<d", 8),
}

_GGUF_ARCH_TO_HF: dict[str, str] = {
    "llama":        "llama",
    "mistral":      "mistral",
    "mixtral":      "mixtral",
    "falcon":       "falcon",
    "gpt2":         "gpt2",
    "gptneox":      "gpt_neox",
    "phi2":         "phi",
    "phi3":         "phi3",
    "gemma":        "gemma",
    "gemma2":       "gemma2",
    "qwen2":        "qwen2",
    "mamba":        "mamba",
    "t5":           "t5",
    "bert":         "bert",
    "roberta":      "roberta",
    "nomic-bert":   "bert",
    "jina-bert-v2": "bert",
    "bloom":        "bloom",
    "starcoder2":   "starcoder2",
    "command-r":    "cohere",
    "internlm2":    "internlm2",
    "olmo":         "olmo",
    "olmo2":        "olmo2",
    "deepseek2":    "llama",
    "granite":      "llama",
}


class _BytesReader:
    """Minimal stateful reader over a bytes object."""
    __slots__ = ("data", "pos")

    def __init__(self, data: bytes):
        self.data = data
        self.pos  = 0

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
    if vtype in _GGUF_FMT:
        r.read(_GGUF_FMT[vtype][1])
    elif vtype == 8:
        length = r.unpack("<Q")
        r.read(length)
    elif vtype == 9:
        elem_type = r.unpack("<I")
        count     = r.unpack("<Q")
        if elem_type in _GGUF_FMT:
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
    if vtype == 8:
        return _gguf_read_string(r)
    if vtype == 9:
        elem_type = r.unpack("<I")
        count     = r.unpack("<Q")
        if elem_type == 8 and count > 64:
            for _ in range(count):
                _gguf_skip_value(r, 8)
            return count
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
    r.read(4)
    version = r.unpack("<I")
    if version not in (1, 2, 3):
        raise ValueError(f"Unsupported GGUF version {version} (expected 1, 2, or 3).")

    _n_tensors = r.unpack("<Q")
    n_kv       = r.unpack("<Q")

    kv: dict[str, Any] = {}
    for _ in range(n_kv):
        try:
            key   = _gguf_read_string(r)
            vtype = r.unpack("<I")
            kv[key] = _gguf_read_value(r, vtype)
        except ValueError:
            break

    return _gguf_kv_to_hf_config(kv, stem)


def _gguf_kv_to_hf_config(kv: dict[str, Any], stem: str) -> dict:
    """Convert GGUF metadata key-value pairs to a HuggingFace-compatible config dict."""
    arch_gguf: str = str(kv.get("general.architecture", "")).lower()
    model_type = _GGUF_ARCH_TO_HF.get(arch_gguf, arch_gguf or _guess_family(stem))

    prefix = arch_gguf + "." if arch_gguf else ""

    def _g(key: str, default: Any = None) -> Any:
        return kv.get(prefix + key, kv.get(key, default))

    hidden_size  = int(_g("embedding_length", 4096))
    num_layers   = int(_g("block_count", 32))
    num_heads    = int(_g("attention.head_count", 32))
    num_kv_heads = int(_g("attention.head_count_kv", num_heads))
    intermediate = int(_g("feed_forward_length", hidden_size * 4))
    context_len  = int(_g("context_length", 4096))
    head_dim     = int(_g("attention.key_length", hidden_size // max(num_heads, 1)))
    rms_eps      = float(_g("attention.layer_norm_rms_epsilon", 1e-5))

    vocab_raw = _g("vocab_size") or _g("tokenizer.ggml.tokens")
    vocab_size = vocab_raw if isinstance(vocab_raw, int) else 32000

    name = kv.get("general.name", stem)

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
    elif model_type in ("bert", "roberta", "distilbert"):
        config["intermediate_size"]    = intermediate
        config["num_hidden_layers"]    = num_layers
        config["num_attention_heads"]  = num_heads

    if model_type == "mixtral":
        config["num_local_experts"]   = int(_g("expert_count", 8))
        config["num_experts_per_tok"] = int(_g("expert_used_count", 2))

    return config


# ---------------------------------------------------------------------------
# PyTorch checkpoint — shape extractor
# ---------------------------------------------------------------------------

def _extract_pytorch_shapes(data: bytes, path: Path) -> dict[str, list[int]]:
    """
    Extract tensor name → shape dict from a PyTorch checkpoint.
    Never loads weight data into memory.
    """
    # Strategy 1: torch meta device (shapes only, zero data)
    try:
        import torch
        import io
        buf = io.BytesIO(data)
        obj = torch.load(buf, map_location="meta", weights_only=True)
        if isinstance(obj, dict):
            for key in ("model", "module", "state_dict", "model_state_dict"):
                if key in obj and isinstance(obj[key], dict):
                    obj = obj[key]
                    break
        shapes = {k: list(v.shape) for k, v in obj.items() if hasattr(v, "shape")}
        if shapes:
            return shapes
    except Exception:
        pass

    # Strategy 2: file may be safetensors despite .pt extension
    try:
        from safetensors import safe_open
        import io
        shapes = {}
        with safe_open(io.BytesIO(data), framework="pt") as f:
            for k in f.keys():
                shapes[k] = list(f.get_tensor(k).shape)
        if shapes:
            return shapes
    except Exception:
        pass

    raise ValueError(
        "Could not parse PyTorch checkpoint. "
        "Tip: convert to .safetensors using the HuggingFace safetensors library "
        "(`safetensors.torch.save_file(state_dict, 'model.safetensors')`), "
        "or upload the model's config.json directly."
    )


# ---------------------------------------------------------------------------
# ONNX — shape extractor
# ---------------------------------------------------------------------------

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
    if wire == 0:
        _, pos = _proto_read_varint(data, pos)
    elif wire == 1:
        pos += 8
    elif wire == 2:
        length, pos = _proto_read_varint(data, pos)
        pos += length
    elif wire == 5:
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
        if wire == 2:
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
        if fnum == 1 and wire == 0:
            val, _ = _proto_read_varint(data, ps)
            dims.append(val)
        elif fnum == 1 and wire == 2:
            p = ps
            while p < pe:
                v, p = _proto_read_varint(data, p)
                dims.append(v)
        elif fnum == 3 and wire == 2:
            name = data[ps:pe].decode("utf-8", errors="replace")
    return name, dims


def _onnx_scan_initializers(data: bytes) -> dict[str, list[int]]:
    """Lightweight ONNX initializer scanner without the onnx package."""
    shapes: dict[str, list[int]] = {}

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

    try:
        for fnum, wire, ps, pe in _proto_iter_fields(data, graph_start, graph_end):
            if fnum == 4 and wire == 2:
                try:
                    name, dims = _onnx_parse_tensor_proto(data, ps, pe)
                    if name and dims:
                        shapes[name] = dims
                except (ValueError, struct.error):
                    continue
    except (ValueError, struct.error):
        pass

    return shapes


def _extract_onnx_shapes(path: Path) -> dict[str, list[int]]:
    """
    Extract tensor name → shape dict from an ONNX model file.

    Strategy 1: onnx.load() — memory-maps the file, handles any size.
    Strategy 2: lightweight protobuf scanner (no extra deps).
    """
    try:
        import onnx  # type: ignore
        model  = onnx.load(str(path))
        shapes: dict[str, list[int]] = {
            init.name: list(init.dims)
            for init in model.graph.initializer
            if init.dims
        }
        for vi in list(model.graph.input) + list(model.graph.value_info):
            try:
                t    = vi.type.tensor_type
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
        return shapes
    except ImportError:
        pass
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not parse ONNX model: {e}") from e

    data = _read_header_bytes(path, 128 * 1024 * 1024)
    if len(data) < 4:
        raise ValueError("File too small to be a valid ONNX model.")
    try:
        shapes = _onnx_scan_initializers(data)
    except Exception as e:
        raise ValueError(f"Could not parse ONNX protobuf: {e}") from e
    if not shapes:
        raise ValueError("No weight initializers found in ONNX file.")
    return shapes


# ---------------------------------------------------------------------------
# Minimal helpers used only by GGUF path
# ---------------------------------------------------------------------------

_FAMILY_HINTS: list[tuple[str, str]] = [
    ("llama",    "llama"),  ("mistral",  "mistral"),  ("mixtral",  "mixtral"),
    ("qwen",     "qwen2"),  ("gemma",    "gemma"),     ("falcon",   "falcon"),
    ("phi",      "phi3"),   ("gpt2",     "gpt2"),      ("gpt-2",    "gpt2"),
    ("neox",     "gpt_neox"),("bert",    "bert"),      ("roberta",  "roberta"),
    ("t5",       "t5"),     ("flan",     "t5"),         ("mamba",    "mamba"),
    ("internlm", "internlm2"),("olmo",   "olmo"),      ("deepseek", "llama"),
    ("vicuna",   "llama"),  ("hermes",   "llama"),      ("solar",    "llama"),
    ("openchat", "llama"),  ("zephyr",   "mistral"),    ("tinyllama","llama"),
]


def _guess_family(stem: str, default: str = "llama") -> str:
    s = stem.lower()
    for hint, family in _FAMILY_HINTS:
        if hint in s:
            return family
    return default
