"""
Input resolver - detects input type and routes to the right fetcher.
"""

from __future__ import annotations
import logging
import re
import time

from ..models.ir import ArchitectureIR, SourceType
from ..models.api import ResolveResponse
from .prebaked import get_prebaked
from .hf_fetcher import fetch_hf_config, HFConfigNotFoundError
from .arch_parser import parse_hf_config

logger = logging.getLogger(__name__)

# Matches "owner/model-name" or "model-name" (no spaces, HF-style IDs)
_HF_ID_RE = re.compile(r"^[a-zA-Z0-9_\-\.]+(/[a-zA-Z0-9_\-\.]+)?$")


def _looks_like_hf_id(text: str) -> bool:
    text = text.strip()
    return bool(_HF_ID_RE.match(text)) and len(text) < 200


async def resolve(
    input_text: str,
    hf_token: str | None = None,
) -> ResolveResponse:
    """
    Main resolver entry point.
    1. Check pre-baked library
    2. Try HuggingFace Hub (if input looks like an HF model ID)
    3. Return error details if not found (LLM fallback handled separately in /chat)
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

    # --- Step 2: HuggingFace Hub ---
    if _looks_like_hf_id(text):
        try:
            config = await fetch_hf_config(text, hf_token=hf_token)
            ir = parse_hf_config(config, model_id=text)
            elapsed = (time.monotonic() - start) * 1000
            logger.info("Resolved '%s' from HuggingFace in %.1fms", text, elapsed)
            return ResolveResponse(
                ir=ir,
                source=SourceType.HF_CONFIG,
                cached=False,
                resolve_time_ms=elapsed,
            )
        except HFConfigNotFoundError as e:
            logger.info("HF lookup failed for '%s': %s", text, e)
            # Fall through - caller should try LLM / web search for Phase 3

    # --- Not found ---
    raise ValueError(
        f"Could not resolve '{text}'. "
        "Try a HuggingFace model ID (e.g. 'bert-base-uncased') or use the chat to search."
    )
