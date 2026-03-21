"""
Input resolver — detects input type and routes to the right parser.

Pipeline:
  Step 1 — Pre-baked library            (instant, exact)
  Step 2 — spaCy / non-standard NLP     (meta.json-based, no transformers)
  Step 3 — AutoConfig mapper            (transformers CONFIG_MAPPING, ~475 types)
              confidence HIGH  → done
              confidence MEDIUM/LOW → Step 4
  Step 4 — LLM verifier                 (checks IR against config, applies corrections,
                                         re-runs mapper; result cached by config SHA-256)
"""

from __future__ import annotations
import logging
import re
import time

from ..models.ir import ArchitectureIR, SourceType, SourceConfidence
from ..models.api import ResolveResponse
from .prebaked import get_prebaked
from .hf_fetcher import fetch_hf_config, HFConfigNotFoundError
from .autoconfig_mapper import map_autoconfig_to_ir

logger = logging.getLogger(__name__)

_HF_ID_RE = re.compile(r"^[a-zA-Z0-9_\-\.]+(/[a-zA-Z0-9_\-\.]+)?$")

# Model types routed through arch_parser instead of autoconfig_mapper.
# Standard transformers (bert, llama, t5, etc.) go through autoconfig_mapper.
# VLM and custom-MoE types stay here because their configs have nested
# sub-configs (language_config, vision_config, sam_config, ...) that
# autoconfig_mapper cannot read from the top level.
_LEGACY_TYPES = {
    "spacy",
    "deepseek_vl_v2",  # CLIP + SAM + projector + DeepSeek-V2 LM (nested sub-configs)
    "deepseek_vl",     # same family
    "deepseek_v2",     # custom MoE config structure
    "deepseek_v3",     # same
}


def _looks_like_hf_id(text: str) -> bool:
    return bool(_HF_ID_RE.match(text.strip())) and len(text) < 200


def _needs_llm_verification(ir: ArchitectureIR) -> bool:
    return ir.source_confidence in (SourceConfidence.MEDIUM, SourceConfidence.LOW)


async def resolve(
    input_text: str,
    hf_token: str | None = None,
) -> ResolveResponse:
    start = time.monotonic()
    text = input_text.strip()

    # ── Step 1: pre-baked library ────────────────────────────────────────────
    prebaked_ir = get_prebaked(text)
    if prebaked_ir:
        elapsed = (time.monotonic() - start) * 1000
        logger.info("Resolved '%s' from pre-baked library in %.1fms", text, elapsed)
        return ResolveResponse(ir=prebaked_ir, source=SourceType.PREBAKED,
                               cached=True, resolve_time_ms=elapsed)

    if not _looks_like_hf_id(text):
        raise ValueError(
            f"Could not resolve '{text}'. "
            "Try a HuggingFace model ID (e.g. 'bert-base-uncased')."
        )

    try:
        config = await fetch_hf_config(text, hf_token=hf_token)
    except HFConfigNotFoundError as e:
        logger.info("HF lookup failed for '%s': %s", text, e)
        raise ValueError(
            f"Could not resolve '{text}'. "
            "Try a HuggingFace model ID (e.g. 'bert-base-uncased') or use the chat to search."
        )

    model_type = (config.get("model_type") or "").lower()

    # ── Step 2: non-standard types (spaCy, etc.) — use legacy arch_parser ───
    if model_type in _LEGACY_TYPES:
        from .arch_parser import parse_hf_config as _legacy_parse
        ir = _legacy_parse(config, model_id=text)
        elapsed = (time.monotonic() - start) * 1000
        logger.info("Resolved '%s' via legacy parser (%s) in %.1fms", text, model_type, elapsed)
        return ResolveResponse(ir=ir, source=ir.source,
                               cached=False, resolve_time_ms=elapsed)

    # ── Step 3: AutoConfig mapper ────────────────────────────────────────────
    ir = map_autoconfig_to_ir(config, model_id=text)
    logger.debug("AutoConfig mapper: model_type=%s confidence=%s",
                 model_type, ir.source_confidence)

    # ── Step 4: LLM verifier (only when confidence < HIGH) ───────────────────
    if _needs_llm_verification(ir):
        try:
            from .llm_verifier import llm_verify_and_correct
            corrected_config = await llm_verify_and_correct(
                raw_config=config,
                ir_dict=ir.model_dump(),
                model_id=text,
            )
            if corrected_config:
                ir = map_autoconfig_to_ir(corrected_config, model_id=text)
                ir.source_confidence = SourceConfidence.HIGH
                logger.info("LLM verifier applied corrections for '%s'", text)
        except Exception as exc:
            logger.warning("LLM verifier failed for '%s': %s — keeping original IR", text, exc)

    elapsed = (time.monotonic() - start) * 1000
    tier_label = "AutoConfig/HIGH" if ir.source_confidence == SourceConfidence.HIGH else "AutoConfig+LLM"
    logger.info("Resolved '%s' via %s in %.1fms (confidence=%s)",
                text, tier_label, elapsed, ir.source_confidence)
    return ResolveResponse(ir=ir, source=ir.source,
                           cached=False, resolve_time_ms=elapsed)
