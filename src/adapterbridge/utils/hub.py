"""Hugging Face Hub resolution utilities with disk caching and offline fallbacks."""

import json
import os
from typing import Any, Dict, Optional
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError

CACHE_DIR = os.path.expanduser("~/.cache/adapterbridge/hub")


def fetch_remote_model_config(
    model_id: str, filename: str = "config.json", local_files_only: bool = False
) -> Optional[Dict[str, Any]]:
    """Download and parse a JSON config file from HF Hub or local disk cache.
    
    Returns None if offline, unauthorized, or file does not exist.
    """
    if not model_id:
        return None

    # Check local cache first
    safe_model_id = model_id.replace("/", "--")
    cached_file = os.path.join(CACHE_DIR, safe_model_id, filename)
    if os.path.exists(cached_file):
        try:
            with open(cached_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    try:
        downloaded_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_files_only=local_files_only,
        )
        with open(downloaded_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Cache locally
        os.makedirs(os.path.dirname(cached_file), exist_ok=True)
        with open(cached_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return data
    except (HfHubHTTPError, LocalEntryNotFoundError, Exception):
        return None


def resolve_base_model_lineage(base_model_id: str) -> Dict[str, Any]:
    """Fetch base model lineage configs (config.json, generation_config.json, tokenizer_config.json)."""
    return {
        "config": fetch_remote_model_config(base_model_id, "config.json"),
        "generation_config": fetch_remote_model_config(base_model_id, "generation_config.json"),
        "tokenizer_config": fetch_remote_model_config(base_model_id, "tokenizer_config.json"),
    }
