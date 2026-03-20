"""
HuggingFace config.json fetcher.
Downloads ONLY config.json - never model weights.

Fallback chain when config.json is absent:
  meta.json   → spaCy pipelines and other non-transformer models
"""

import json
import logging
import re
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError

logger = logging.getLogger(__name__)


class HFConfigNotFoundError(Exception):
    pass


def _parse_spacy_cfg(text: str) -> dict:
    """
    Extract key architecture numbers from a spaCy config.cfg file.
    Returns a flat dict of the values we care about.
    """
    out: dict = {}
    section = ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("["):
            section = line.strip("[]").lower()
            continue
        if "=" not in line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")

        # tok2vec model section
        if "tok2vec.model" in section and "listener" not in section:
            if key == "width":
                try: out["tok2vec_width"] = int(val)
                except ValueError: pass
            elif key == "depth":
                try: out["tok2vec_depth"] = int(val)
                except ValueError: pass
            elif key == "embed_size":
                try: out["tok2vec_embed_size"] = int(val)
                except ValueError: pass
            elif key == "window_size":
                try: out["tok2vec_window"] = int(val)
                except ValueError: pass
            elif key == "maxout_pieces":
                try: out["tok2vec_maxout"] = int(val)
                except ValueError: pass
            elif key == "subword_features":
                out["tok2vec_subword"] = val.lower() == "true"

        # parser / ner upper model hidden size
        if ("parser.model" in section or "ner.model" in section) and "upper" in section:
            if key == "nO":
                try: out["upper_nO"] = int(val)
                except ValueError: pass
            elif key == "nI":
                try: out["upper_nI"] = int(val)
                except ValueError: pass
    return out


async def fetch_hf_config(model_id: str, hf_token: str | None = None) -> dict:
    """
    Download and parse config.json from a HuggingFace model repo.
    Falls back to meta.json (spaCy / non-transformer models) when config.json
    is absent. Never downloads weights.
    Raises HFConfigNotFoundError if nothing usable is found.
    """
    # ── Primary: standard HuggingFace transformer config ──────────────────────
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
        pass  # no config.json — try alternative formats below
    except Exception as e:
        raise HFConfigNotFoundError(f"Failed to fetch config for '{model_id}': {e}") from e

    # ── Fallback: spaCy meta.json ──────────────────────────────────────────────
    try:
        meta_path = hf_hub_download(
            repo_id=model_id,
            filename="meta.json",
            token=hf_token,
        )
        meta = json.loads(Path(meta_path).read_text())

        # Enrich with architecture details from config.cfg where available
        cfg_data: dict = {}
        try:
            cfg_path = hf_hub_download(
                repo_id=model_id,
                filename="config.cfg",
                token=hf_token,
            )
            cfg_data = _parse_spacy_cfg(Path(cfg_path).read_text())
            logger.debug("Parsed spaCy config.cfg for '%s': %s", model_id, cfg_data)
        except Exception:
            pass

        meta["model_type"] = "spacy"
        meta["_cfg"] = cfg_data
        logger.info("Resolved '%s' via meta.json (spaCy pipeline)", model_id)
        return meta

    except EntryNotFoundError:
        raise HFConfigNotFoundError(
            f"No config.json or meta.json found for '{model_id}'. "
            "This repo may not be a supported model format."
        )
    except Exception as e:
        raise HFConfigNotFoundError(
            f"Failed to fetch config for '{model_id}': {e}"
        ) from e


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
