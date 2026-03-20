"""
LLM-assisted architecture parser — Tier 3 fallback.

Called only when the smart generic parser (Tier 2) returns MEDIUM or LOW
confidence, meaning the config uses non-standard field names or is missing
key dimensions.

The LLM is asked to extract a normalized set of architecture fields from the
raw config.json.  That normalized dict is then fed back through the smart
generic parser, producing a proper ArchitectureIR.

Response is cached in memory (and optionally on disk) so the same config
never pays the LLM cost twice.

Provider priority:  ANTHROPIC_API_KEY → Claude  |  OPENAI_API_KEY → GPT-4o
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Disk cache (survives restarts)
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "modelverse" / "llm_parse"
_MEM_CACHE: dict[str, dict] = {}


def _cache_key(config: dict) -> str:
    raw = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_cache(key: str) -> dict | None:
    if key in _MEM_CACHE:
        return _MEM_CACHE[key]
    path = _CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            _MEM_CACHE[key] = data
            return data
        except Exception:
            pass
    return None


def _save_cache(key: str, data: dict) -> None:
    _MEM_CACHE[key] = data
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (_CACHE_DIR / f"{key}.json").write_text(json.dumps(data))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are an expert in deep learning architectures and the HuggingFace "
    "transformers library. You extract structured architecture information "
    "from config.json files with perfect accuracy."
)

_USER_TEMPLATE = """\
Given the following HuggingFace config.json, extract the canonical \
architecture parameters and return ONLY a JSON object with exactly these fields \
(use null for unknown or not-applicable fields):

{{
  "model_type": "<string>",
  "architecture_class": "<decoder|encoder|encoder_decoder>",
  "hidden_size": <int>,
  "num_hidden_layers": <int>,
  "num_attention_heads": <int>,
  "num_key_value_heads": <int>,
  "intermediate_size": <int>,
  "vocab_size": <int>,
  "max_position_embeddings": <int>,
  "position_embedding_type": "<rope|absolute|relative|alibi|none>",
  "hidden_act": "<silu|gelu|relu|swiglu|geglu|gelu_new>",
  "norm_type": "<rms_norm|layer_norm>",
  "is_moe": <bool>,
  "num_local_experts": <int|null>,
  "num_experts_per_tok": <int|null>,
  "moe_intermediate_size": <int|null>,
  "n_shared_experts": <int|null>,
  "first_k_dense_replace": <int|null>,
  "tie_word_embeddings": <bool>,
  "is_encoder_decoder": <bool>
}}

Rules:
- Return ONLY valid JSON, no explanation, no markdown fences.
- For architecture_class: use "decoder" for GPT/LLaMA/Mistral-style causal LMs, \
"encoder" for BERT/RoBERTa-style, "encoder_decoder" for T5/BART-style.
- If a field uses a non-standard name in the config, map it to the standard name above.
- For nested configs (VL models, multi-modal), use the language model sub-config \
for the main dimensions.

Config:
{config_json}"""


def _make_prompt(config: dict) -> str:
    config_json = json.dumps(config, indent=2)
    # Truncate huge configs (e.g. very long vocab mappings)
    if len(config_json) > 12_000:
        config_json = config_json[:12_000] + "\n... (truncated)"
    return _USER_TEMPLATE.format(config_json=config_json)


# ---------------------------------------------------------------------------
# LLM call (single non-streaming turn)
# ---------------------------------------------------------------------------

async def _call_anthropic(prompt: str) -> str | None:
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = await client.messages.create(
            model="claude-haiku-4-5",   # fast + cheap for structured extraction
            max_tokens=512,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else None
    except Exception as e:
        logger.debug("Anthropic LLM parse failed: %s", e)
        return None


async def _call_openai(prompt: str) -> str | None:
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.chat.completions.create(
            model="gpt-4o-mini",        # fast + cheap for structured extraction
            max_tokens=512,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        msg = response.choices[0].message.content
        return msg if msg else None
    except Exception as e:
        logger.debug("OpenAI LLM parse failed: %s", e)
        return None


def _parse_llm_response(text: str) -> dict | None:
    """Extract JSON from the LLM response, tolerating minor formatting issues."""
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            l for l in lines if not l.startswith("```")
        ).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first { ... } block
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    logger.debug("Could not parse LLM response as JSON: %.200s", text)
    return None


async def llm_enrich_config(config: dict, model_id: str) -> dict | None:
    """
    Ask the LLM to normalise a config.json into canonical architecture fields.

    Returns a dict that can be passed straight into `_smart_generic_fallback`
    (it looks like a standard HF config), or None if no LLM key is available
    or the call fails.

    The result is cached on disk keyed by the SHA-256 of the config, so
    repeated lookups for the same config are instant and free.
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not anthropic_key and not openai_key:
        return None

    key = _cache_key(config)
    cached = _load_cache(key)
    if cached is not None:
        logger.debug("LLM parse cache hit for '%s' (%s)", model_id, key)
        return cached

    logger.info("Tier-3 LLM parse triggered for unknown model type '%s'",
                config.get("model_type", "?"))

    prompt = _make_prompt(config)
    raw: str | None = None
    if anthropic_key:
        raw = await _call_anthropic(prompt)
    if raw is None and openai_key:
        raw = await _call_openai(prompt)
    if raw is None:
        return None

    normalized = _parse_llm_response(raw)
    if normalized is None:
        return None

    # Merge original config so any fields the LLM missed still resolve
    merged = {**config, **{k: v for k, v in normalized.items() if v is not None}}
    _save_cache(key, merged)
    logger.info("LLM parse succeeded for '%s' (model_type=%s)",
                model_id, normalized.get("model_type", "?"))
    return merged
