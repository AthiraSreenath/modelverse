"""
LLM Architecture Verifier
==========================
Takes a raw config.json + the ArchitectureIR we produced and asks an LLM
to spot discrepancies. Applies any corrections by re-running the mapper
with overrides.

This is fundamentally different from the old llm_parser (which asked the LLM
to generate an IR from scratch). Checking a concrete IR against a concrete
config is a much easier, more reliable task for the LLM.

Trigger conditions (in router.py):
  - source_confidence == MEDIUM or LOW  (model_type not in CONFIG_MAPPING)
  - First time we see this model_type

Cache: SHA-256 of the raw config JSON → skip repeat calls forever.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(os.getenv("MODELVERSE_CACHE_DIR", Path.home() / ".cache" / "modelverse")) / "llm_verify"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(raw_config: dict) -> str:
    blob = json.dumps(raw_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _load_cache(key: str) -> list[dict] | None:
    p = _CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return None


def _save_cache(key: str, corrections: list[dict]) -> None:
    p = _CACHE_DIR / f"{key}.json"
    try:
        p.write_text(json.dumps(corrections))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# IR summary builder (compact, fits in prompt)
# ---------------------------------------------------------------------------

def _ir_summary(ir_dict: dict) -> str:
    lines = [f"display_name: {ir_dict.get('display_name', '')}",
             f"family: {ir_dict.get('family', '')}",
             f"task: {ir_dict.get('task', '')}",
             "blocks:"]
    for b in ir_dict.get("blocks", []):
        p = b.get("param_count")
        p_str = f"{p/1e9:.3f}B" if p and p > 1e9 else (f"{p/1e6:.1f}M" if p and p > 1e4 else str(p))
        key_params = {k: v for k, v in (b.get("params") or {}).items()
                      if k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                               "num_key_value_heads", "intermediate_size", "vocab_size",
                               "num_local_experts", "num_experts_per_tok", "state_size",
                               "activation")}
        lines.append(f"  - {b['id']} [{b['type']}] {b['label']}  params={p_str}  {key_params}")
    if ir_dict.get("compute"):
        total = ir_dict["compute"].get("params_total", 0)
        lines.append(f"total_params: {total/1e9:.3f}B")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are an expert ML engineer specializing in transformer architectures. "
    "Your task is to verify that an architecture summary correctly reflects a "
    "HuggingFace config.json. Be precise and conservative — only flag genuine "
    "discrepancies, not stylistic differences."
)

_USER_TMPL = """\
Here is a HuggingFace model config.json (may be truncated for length):
```json
{config_json}
```

Here is the architecture IR we auto-generated from it:
```
{ir_summary}
```

List any genuine discrepancies between the config and the IR as a JSON array.
Each entry must be:
  {{"field": "<config field name>", "block_id": "<block id or 'global'>",
    "our_value": <what we have>, "correct_value": <what it should be>,
    "reason": "<one sentence>"}}

Rules:
- Only flag things that are WRONG or MISSING, not things that are just absent from the IR.
- Parameter count discrepancies > 5% are worth flagging.
- If the IR looks correct, return exactly: []
- Return ONLY the JSON array, no prose.
"""


def _make_prompt(raw_config: dict, ir_dict: dict) -> str:
    # Compact config: keep only architecture-relevant fields
    keep = {k: v for k, v in raw_config.items()
            if k not in ("tokenizer_class", "auto_map", "transformers_version",
                         "_commit_hash", "torch_dtype")}
    config_str = json.dumps(keep, indent=2)
    if len(config_str) > 3000:  # truncate for token budget
        config_str = config_str[:3000] + "\n  ... (truncated)"
    return _USER_TMPL.format(config_json=config_str, ir_summary=_ir_summary(ir_dict))


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> list[dict]:
    """Pull the first JSON array out of the LLM response."""
    # Strip markdown fences
    text = re.sub(r"```[a-z]*\n?", "", text).strip()
    # Find outermost [...] 
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        result = json.loads(text[start:end + 1])
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


async def _call_anthropic(prompt: str) -> str:
    import anthropic
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.AsyncAnthropic(api_key=key)
    msg = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


async def _call_openai(prompt: str) -> str:
    from openai import AsyncOpenAI
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = AsyncOpenAI(api_key=key)
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""


async def _call_llm(prompt: str) -> str:
    """Try Anthropic first, fall back to OpenAI."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return await _call_anthropic(prompt)
    if os.getenv("OPENAI_API_KEY"):
        return await _call_openai(prompt)
    raise RuntimeError("No LLM API key available (set ANTHROPIC_API_KEY or OPENAI_API_KEY)")


# ---------------------------------------------------------------------------
# Correction applicator
# ---------------------------------------------------------------------------

def _apply_corrections(raw_config: dict, corrections: list[dict]) -> dict | None:
    """
    Given LLM corrections, return an updated raw_config dict.
    Only applies corrections where the field is a top-level config key
    and the correct_value looks sane (non-null).
    """
    if not corrections:
        return None

    updated = dict(raw_config)
    applied = 0
    for c in corrections:
        field = c.get("field", "")
        correct = c.get("correct_value")
        if not field or correct is None:
            continue
        # Only update known architectural fields — don't let LLM inject garbage
        _SAFE_FIELDS = {
            "num_hidden_layers", "hidden_size", "num_attention_heads",
            "num_key_value_heads", "intermediate_size", "vocab_size",
            "num_local_experts", "num_experts_per_tok", "n_routed_experts",
            "n_shared_experts", "moe_intermediate_size", "first_k_dense_replace",
            "hidden_act", "state_size", "d_conv", "expand",
            "num_encoder_layers", "num_decoder_layers", "is_encoder_decoder",
            "tie_word_embeddings",
        }
        if field in _SAFE_FIELDS:
            updated[field] = correct
            applied += 1
            logger.info("LLM correction applied: %s = %s (was %s) — %s",
                        field, correct, raw_config.get(field), c.get("reason", ""))

    return updated if applied else None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def llm_verify_and_correct(
    raw_config: dict,
    ir_dict: dict,
    model_id: str,
) -> dict | None:
    """
    Verify the IR against the config using an LLM.

    Returns an updated raw_config dict if corrections were applied, or None
    if the IR was correct (or if verification failed/was skipped).
    The caller should re-run the mapper with the returned config if not None.
    """
    key = _cache_key(raw_config)
    cached = _load_cache(key)
    if cached is not None:
        logger.debug("LLM verifier: cache hit for %s (%d corrections)", model_id, len(cached))
        corrections = cached
    else:
        try:
            prompt = _make_prompt(raw_config, ir_dict)
            response = await _call_llm(prompt)
            corrections = _extract_json(response)
            _save_cache(key, corrections)
            logger.info("LLM verifier: %d correction(s) for %s", len(corrections), model_id)
        except Exception as e:
            logger.warning("LLM verifier failed for %s: %s", model_id, e)
            return None

    return _apply_corrections(raw_config, corrections)
