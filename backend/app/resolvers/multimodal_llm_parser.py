"""
Multimodal Architecture Parser (LLM-based)
==========================================
For any HuggingFace model whose config.json contains nested sub-configs
(vision_config, connector_config, image_config, etc.), autoconfig_mapper
only sees the top-level language backbone and misses the rest.

This module:
1. Detects multimodal configs by scanning for known sub-config keys.
2. Sends the full config + a structured schema prompt to the LLM.
3. Parses the LLM JSON response into an ArchitectureIR.
4. Caches the result (SHA-256 of config) so each model is only LLM-called once.

This is NOT model-specific. The LLM can reason about CogViT, CLIP, SAM,
CogVLM, InternViT, SigLIP, etc. without any hardcoded parser per model.
The only thing that matters is whether the config has nested sub-configs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from ..models.ir import (
    ArchBlock, ArchitectureIR, BlockType, SourceType, SourceConfidence,
    ComputeStats,
)

logger = logging.getLogger(__name__)

_CACHE_DIR = (
    Path(os.getenv("MODELVERSE_CACHE_DIR", Path.home() / ".cache" / "modelverse"))
    / "multimodal_llm"
)
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Config keys that indicate a nested sub-model component
_MULTIMODAL_KEYS: set[str] = {
    "vision_config",
    "visual_config",
    "image_config",
    "encoder_config",
    "decoder_config",
    "connector_config",
    "projector_config",
    "vit_config",
    "cogvit_config",
    "sam_config",
    "clip_config",
    "audio_config",
    "speech_config",
    # used by LLaVA, InternVL, GLM-VL, etc.
    "mm_vision_tower",
    "vision_tower",
}


def is_multimodal(config: dict) -> bool:
    """Return True if the config has any nested sub-model component."""
    return any(k in config for k in _MULTIMODAL_KEYS)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _load_cache(key: str) -> list[dict] | None:
    p = _CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return None


def _save_cache(key: str, blocks: list[dict]) -> None:
    p = _CACHE_DIR / f"{key}.json"
    try:
        p.write_text(json.dumps(blocks))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert ML engineer who understands multimodal vision-language model
architectures. You will be given a HuggingFace model config.json and must
describe the full forward-pass architecture as a structured JSON block list.
Be accurate and concise. Use your knowledge of the model family to fill in
details that are implicit in the config."""

_USER_TMPL = """\
Model ID: {model_id}

Config JSON (may be truncated):
```json
{config_json}
```

Produce the complete architecture as a JSON array of blocks representing the
forward pass in logical order (inputs first, outputs last).

Each block MUST have exactly these fields:
{{
  "id": "snake_case_unique_id",
  "label": "Human-Readable Name",
  "type": one of ["embedding","transformer_stack","linear","add","layer_norm","unknown"],
  "param_count": integer (best estimate; 0 for param-free ops; null if truly unknown),
  "merge_from": null OR list of source block IDs (use [] for branch starts with no input,
                use ["id1","id2"] for blocks that merge two streams),
  "notes": "one sentence describing this component"
}}

Rules:
- List blocks in forward-pass order.
- For the vision encoder: use merge_from=[] to start its branch (no upstream block).
- For text token embeddings: use merge_from=[] (independent input).
  Also add "layout_align_y_with": "<id of connector/projector>" so it renders
  alongside the vision output at the right level.
- For the multimodal merge/concat step: use merge_from=["vision_block_id","embed_tokens"].
- param_count should be your best estimate in raw integer form.
- If a component has sub-components (e.g. the LM decoder), represent it as one block
  with the total parameter count.
- Return ONLY the JSON array, no prose, no markdown fences.
"""


def _build_prompt(config: dict, model_id: str) -> str:
    # Trim the config to a reasonable size
    config_str = json.dumps(config, indent=2)
    if len(config_str) > 4000:
        config_str = config_str[:4000] + "\n  ... (truncated)"
    return _USER_TMPL.format(model_id=model_id, config_json=config_str)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def _call_anthropic(system: str, user: str) -> str:
    import anthropic
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.AsyncAnthropic(api_key=key)
    msg = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


async def _call_openai(system: str, user: str) -> str:
    from openai import AsyncOpenAI
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = AsyncOpenAI(api_key=key)
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=2048,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


async def _call_llm(system: str, user: str) -> str:
    if os.getenv("ANTHROPIC_API_KEY"):
        return await _call_anthropic(system, user)
    if os.getenv("OPENAI_API_KEY"):
        return await _call_openai(system, user)
    raise RuntimeError("No LLM API key set (ANTHROPIC_API_KEY or OPENAI_API_KEY)")


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

_BLOCK_TYPE_MAP: dict[str, BlockType] = {
    "embedding":        BlockType.EMBEDDING,
    "transformer_stack": BlockType.TRANSFORMER_STACK,
    "linear":           BlockType.LINEAR,
    "add":              BlockType.ADD,
    "layer_norm":       BlockType.LAYER_NORM,
    "unknown":          BlockType.UNKNOWN,
}


def _extract_json_array(text: str) -> list[dict]:
    """Pull the first JSON array out of the LLM response."""
    text = re.sub(r"```[a-z]*\n?", "", text).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        result = json.loads(text[start : end + 1])
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def _blocks_from_llm(raw_blocks: list[dict]) -> list[ArchBlock]:
    blocks: list[ArchBlock] = []
    for b in raw_blocks:
        bid = str(b.get("id", f"block_{len(blocks)}"))
        label = str(b.get("label", bid))
        type_str = str(b.get("type", "unknown")).lower()
        btype = _BLOCK_TYPE_MAP.get(type_str, BlockType.UNKNOWN)
        param_count = b.get("param_count")
        if param_count is not None:
            try:
                param_count = int(param_count)
            except (TypeError, ValueError):
                param_count = None

        merge_from: list[str] | None = b.get("merge_from")
        if merge_from is not None and not isinstance(merge_from, list):
            merge_from = None

        layout_align_y_with: str | None = b.get("layout_align_y_with")

        blocks.append(ArchBlock(
            id=bid,
            label=label,
            type=btype,
            params={},
            repeat=1,
            children=[],
            param_count=param_count,
            notes=b.get("notes"),
            merge_from=merge_from,
            layout_align_y_with=layout_align_y_with if isinstance(layout_align_y_with, str) else None,
        ))
    return blocks


def _compute_stats(blocks: list[ArchBlock]) -> ComputeStats:
    total = sum(b.param_count or 0 for b in blocks)
    emb = next((b.param_count or 0 for b in blocks if b.type == BlockType.EMBEDDING), 0)
    enc = total - emb
    return ComputeStats(
        params_total=total,
        params_embedding=emb,
        params_encoder=enc,
        params_decoder=0,
        flops_per_token=None,
        memory_bytes_fp16=total * 2 if total else None,
        kv_cache_bytes_per_token=None,
        context_window=None,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def parse_multimodal_with_llm(
    config: dict,
    model_id: str,
) -> ArchitectureIR:
    """
    Generate a full multimodal ArchitectureIR using the LLM.

    Always cached: the same config SHA-256 will not trigger a second LLM call.
    """
    key = _cache_key(config)
    cached = _load_cache(key)

    if cached is not None:
        logger.debug("Multimodal LLM parser: cache hit for %s", model_id)
        raw_blocks = cached
    else:
        prompt = _build_prompt(config, model_id)
        try:
            response = await _call_llm(_SYSTEM, prompt)
            raw_blocks = _extract_json_array(response)
            if not raw_blocks:
                raise ValueError("LLM returned no blocks")
            _save_cache(key, raw_blocks)
            logger.info("Multimodal LLM parser: generated %d blocks for %s",
                        len(raw_blocks), model_id)
        except Exception as exc:
            logger.warning("Multimodal LLM parser failed for %s: %s — falling back", model_id, exc)
            # Graceful fallback: return a minimal IR with an explicit warning
            fallback_block = ArchBlock(
                id="unknown_multimodal",
                label="Multimodal Architecture (parsing failed)",
                type=BlockType.UNKNOWN,
                params={},
                repeat=1,
                children=[],
                param_count=None,
                notes=(
                    f"Could not generate architecture via LLM ({exc}). "
                    "Check that ANTHROPIC_API_KEY or OPENAI_API_KEY is set."
                ),
            )
            return ArchitectureIR(
                name=model_id,
                display_name=model_id.split("/")[-1],
                family="multimodal",
                task="unknown",
                blocks=[fallback_block],
                compute=ComputeStats(params_total=0, params_embedding=0,
                                     params_encoder=0, params_decoder=0),
                source=SourceType.LLM_KNOWLEDGE,
                source_confidence=SourceConfidence.LOW,
            )

    blocks = _blocks_from_llm(raw_blocks)
    compute = _compute_stats(blocks)

    name = config.get("_name_or_path") or model_id.split("/")[-1]
    model_type = config.get("model_type", "multimodal")

    return ArchitectureIR(
        name=model_id,
        display_name=name,
        family=model_type,
        task="image-to-text",
        blocks=blocks,
        compute=compute,
        source=SourceType.LLM_KNOWLEDGE,
        source_confidence=SourceConfidence.MEDIUM,
    )
