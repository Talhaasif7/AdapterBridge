"""Dynamic target engine ruleset specification, schema sync, and caching manager."""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

# Built-in fallback rule profiles when remote endpoint is unreachable or offline
DEFAULT_RULESETS: Dict[str, Dict] = {
    "vllm": {
        "engine": "vllm",
        "version": "0.6.4",
        "unsupported_target_modules": ["embed_tokens", "lm_head"],
        "required_files": ["config.json", "adapter_config.json"],
        "moe_requires_3d_stacked": True,
        "key_prefix_mappings": {
            "base_model.model.model.": "model.",
            "base_model.model.": "model.",
        },
        "chat_template_required": True,
    },
    "sglang": {
        "engine": "sglang",
        "version": "0.2.0",
        "unsupported_target_modules": ["embed_tokens"],
        "required_files": ["config.json", "adapter_config.json"],
        "moe_requires_3d_stacked": True,
        "key_prefix_mappings": {
            "base_model.model.model.": "model.",
            "base_model.model.": "model.",
        },
        "chat_template_required": True,
    },
    "ollama": {
        "engine": "ollama",
        "version": "0.3.0",
        "unsupported_target_modules": [],
        "required_files": ["adapter_config.json"],
        "moe_requires_3d_stacked": False,
        "key_prefix_mappings": {},
        "chat_template_required": False,
    },
    "tensorrt": {
        "engine": "tensorrt",
        "version": "0.10.0",
        "unsupported_target_modules": [],
        "required_files": ["config.json", "adapter_config.json"],
        "moe_requires_3d_stacked": True,
        "key_prefix_mappings": {
            "base_model.model.": "",
        },
        "chat_template_required": False,
    },
}


class EngineRuleSet(BaseModel):
    """Configuration rules for a specific serving engine version."""

    engine: str
    version: str = "latest"
    unsupported_target_modules: List[str] = Field(default_factory=list)
    required_files: List[str] = Field(default_factory=list)
    moe_requires_3d_stacked: bool = False
    key_prefix_mappings: Dict[str, str] = Field(default_factory=dict)
    chat_template_required: bool = True


class SchemaSyncManager:
    """Manages downloading, caching, and loading dynamic engine compatibility schemas."""

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "adapterbridge" / "schemas"
        os.makedirs(self.cache_dir, exist_ok=True)

    def parse_target_spec(self, target_str: str) -> tuple[str, str]:
        """Parse engine and version string (e.g. 'vllm@0.6.4' -> ('vllm', '0.6.4'))."""
        if "@" in target_str:
            parts = target_str.split("@", 1)
            return parts[0].lower().strip(), parts[1].strip()
        return target_str.lower().strip(), "latest"

    def get_ruleset(self, target_str: str) -> EngineRuleSet:
        """Fetch ruleset for given target engine string (e.g. 'vllm@0.6.4')."""
        engine, version = self.parse_target_spec(target_str)
        cache_file = self.cache_dir / f"{engine}_{version}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return EngineRuleSet(**data)
            except Exception:
                pass

        # Fallback to built-in rule set
        base_data = DEFAULT_RULESETS.get(engine, DEFAULT_RULESETS.get("vllm")).copy()
        if version != "latest":
            base_data["version"] = version

        ruleset = EngineRuleSet(**base_data)

        # Save to local cache
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(ruleset.model_dump(), f, indent=2)
        except Exception:
            pass

        return ruleset

    def sync_schemas(self, force: bool = False) -> Dict[str, str]:
        """Sync remote rulesets to local cache (or write default rulesets)."""
        synced: Dict[str, str] = {}
        for engine, rules in DEFAULT_RULESETS.items():
            version = rules.get("version", "latest")
            cache_file = self.cache_dir / f"{engine}_{version}.json"
            if force or not cache_file.exists():
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(rules, f, indent=2)
                synced[engine] = version
            else:
                synced[engine] = f"{version} (cached)"
        return synced
