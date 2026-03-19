"""
HuggingFace config.json fetcher.
Downloads ONLY config.json - never model weights.
"""

import json
import logging
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError

logger = logging.getLogger(__name__)


class HFConfigNotFoundError(Exception):
    pass


async def fetch_hf_config(model_id: str, hf_token: str | None = None) -> dict:
    """
    Download and parse config.json from a HuggingFace model repo.
    Never downloads weights. Raises HFConfigNotFoundError if not found.
    """
    try:
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            token=hf_token,
        )
        return json.loads(Path(config_path).read_text())
    except RepositoryNotFoundError:
        raise HFConfigNotFoundError(f"Model '{model_id}' not found on HuggingFace Hub")
    except EntryNotFoundError:
        raise HFConfigNotFoundError(f"No config.json found for '{model_id}'")
    except Exception as e:
        raise HFConfigNotFoundError(f"Failed to fetch config for '{model_id}': {e}") from e


async def fetch_model_card(model_id: str, hf_token: str | None = None) -> str | None:
    """
    Fetch the model card README for context. Returns None if unavailable.
    """
    try:
        readme_path = hf_hub_download(
            repo_id=model_id,
            filename="README.md",
            token=hf_token,
        )
        content = Path(readme_path).read_text(encoding="utf-8", errors="ignore")
        # Trim to first 4000 chars to keep LLM context manageable
        return content[:4000]
    except Exception:
        return None


def model_exists_on_hf(model_id: str, hf_token: str | None = None) -> bool:
    """Quick check - does this repo exist on HF?"""
    try:
        api = HfApi(token=hf_token)
        api.repo_info(model_id)
        return True
    except Exception:
        return False
