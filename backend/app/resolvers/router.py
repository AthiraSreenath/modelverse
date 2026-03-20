"""
Input resolver - detects input type and routes to the right fetcher.

Three-tier parsing pipeline:
  Tier 1 — Explicit family parsers (37 registered types, sub-ms, exact)
  Tier 2 — Smart generic fallback (any standard HF config, sub-ms, HIGH/MEDIUM confidence)
  Tier 3 — LLM-assisted enrichment (exotic / non-standard configs, ~1-3s, cached)
"""

from __future__ import annotations
import logging
import re
import time

from ..models.ir import ArchitectureIR, SourceType, SourceConfidence
from ..models.api import ResolveResponse
from .prebaked import get_prebaked
from .hf_fetcher import fetch_hf_config, HFConfigNotFoundError
from .arch_parser import parse_hf_config, _FAMILY_PARSERS

logger = logging.getLogger(__name__)

# Matches "owner/model-name" or "model-name" (no spaces, HF-style IDs)
_HF_ID_RE = re.compile(r"^[a-zA-Z0-9_\-\.]+(/[a-zA-Z0-9_\-\.]+)?$")


def _looks_like_hf_id(text: str) -> bool:
    text = text.strip()
    return bool(_HF_ID_RE.match(text)) and len(text) < 200


def _needs_llm_enrichment(config: dict, ir: ArchitectureIR) -> bool:
    """
    Return True if Tier-3 LLM enrichment is worth attempting.

    We only pay the LLM cost when:
      - The model type was not in the explicit registry (Tier 1 didn't handle it), AND
      - The smart generic parser is uncertain (MEDIUM or LOW confidence).

    When Tier 2 returns HIGH confidence the generic parser already has all the
    key fields and the result is accurate enough without an LLM call.
    """
    model_type = config.get("model_type", "").lower()
    if model_type in _FAMILY_PARSERS:
        return False  # Tier 1 handled it — don't waste an LLM call
    return ir.source_confidence in (SourceConfidence.MEDIUM, SourceConfidence.LOW)


async def resolve(
    input_text: str,
    hf_token: str | None = None,
) -> ResolveResponse:
    """
    Main resolver entry point.

    1. Pre-baked library   (instant, exact)
    2. HuggingFace Hub     (Tier 1 → Tier 2 parser)
    3. LLM enrichment      (Tier 3, only when Tier 2 is uncertain)
    """
    start = time.monotonic()
    text = input_text.strip()

    # --- Step 1: pre-baked library (instant) ---
    prebaked_ir = get_prebaked(text)
    if prebaked_ir:
        elapsed = (time.monotonic() - start) * 1000
        logger.info("Resolved '%s' from pre-baked library in %.1fms", text, elapsed)
        return ResolveResponse(
            ir=prebaked_ir,
            source=SourceType.PREBAKED,
            cached=True,
            resolve_time_ms=elapsed,
        )

    # --- Step 2: HuggingFace Hub → Tier 1 / Tier 2 parser ---
    if _looks_like_hf_id(text):
        try:
            config = await fetch_hf_config(text, hf_token=hf_token)
            ir = parse_hf_config(config, model_id=text)

            # --- Step 3: Tier-3 LLM enrichment (only when needed) ---
            if _needs_llm_enrichment(config, ir):
                try:
                    from .llm_parser import llm_enrich_config
                    enriched_config = await llm_enrich_config(config, text)
                    if enriched_config:
                        enriched_ir = parse_hf_config(enriched_config, model_id=text)
                        # Accept the enriched result only if it improved confidence
                        if (
                            enriched_ir.source_confidence != SourceConfidence.LOW
                            or ir.source_confidence == SourceConfidence.LOW
                        ):
                            enriched_ir.source = SourceType.HF_CONFIG
                            enriched_ir.source_confidence = SourceConfidence.HIGH
                            ir = enriched_ir
                            logger.info(
                                "Tier-3 LLM enrichment improved '%s' to %s",
                                text, enriched_ir.source_confidence,
                            )
                except Exception:
                    logger.debug("Tier-3 LLM enrichment failed for '%s', using Tier-2 result", text)

            elapsed = (time.monotonic() - start) * 1000
            tier = (
                1 if config.get("model_type", "").lower() in _FAMILY_PARSERS
                else (3 if ir.source == SourceType.LLM_KNOWLEDGE else 2)
            )
            logger.info("Resolved '%s' via Tier-%d in %.1fms (confidence=%s)",
                        text, tier, elapsed, ir.source_confidence)
            return ResolveResponse(
                ir=ir,
                source=ir.source,
                cached=False,
                resolve_time_ms=elapsed,
            )
        except HFConfigNotFoundError as e:
            logger.info("HF lookup failed for '%s': %s", text, e)
            # Fall through to not-found error

    # --- Not found ---
    raise ValueError(
        f"Could not resolve '{text}'. "
        "Try a HuggingFace model ID (e.g. 'bert-base-uncased') or use the chat to search."
    )
