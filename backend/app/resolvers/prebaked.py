"""
Pre-baked Architecture IR library.
These load instantly with zero network calls.
"""

import json
from pathlib import Path

from ..models.ir import ArchitectureIR

_PREBAKED_DIR = Path(__file__).parent.parent.parent.parent / "data" / "prebaked"

# Canonical model ID → filename mapping
_REGISTRY: dict[str, str] = {
    "bert-base-uncased": "bert-base-uncased.json",
    "google-bert/bert-base-uncased": "bert-base-uncased.json",
    "distilbert-base-uncased": "distilbert-base-uncased.json",
    "distilbert/distilbert-base-uncased": "distilbert-base-uncased.json",
    "gpt2": "gpt2.json",
    "openai-community/gpt2": "gpt2.json",
    "meta-llama/llama-3.1-8b": "meta-llama--Llama-3.1-8B.json",
    "meta-llama/meta-llama-3.1-8b": "meta-llama--Llama-3.1-8B.json",
    "mistralai/mistral-7b-v0.1": "mistralai--Mistral-7B-v0.1.json",
    "t5-base": "t5-base.json",
    "google-t5/t5-base": "t5-base.json",
}


def _normalize(model_id: str) -> str:
    return model_id.strip().lower()


def get_prebaked(model_id: str) -> ArchitectureIR | None:
    """Return a pre-baked IR if we have one, else None."""
    key = _normalize(model_id)
    filename = _REGISTRY.get(key)
    if not filename:
        return None
    path = _PREBAKED_DIR / filename
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return ArchitectureIR.model_validate(data)


def list_prebaked() -> list[str]:
    """Return all available pre-baked model IDs."""
    return list(_REGISTRY.keys())
