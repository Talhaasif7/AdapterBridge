"""Checkpoint inspection scanner and metadata lineage resolution."""

import json
import os
from typing import Dict
from adapterbridge.models.manifest import CheckpointManifest, TensorMetadata
from adapterbridge.utils.safetensors_io import extract_tensor_headers


def scan_checkpoint(checkpoint_path: str) -> CheckpointManifest:
    """Inspect a local checkpoint directory and construct an immutable CheckpointManifest."""
    if not os.path.exists(checkpoint_path):
        return CheckpointManifest(
            checkpoint_path=checkpoint_path,
            is_adapter=False,
            has_config=False,
            has_tokenizer=False,
            has_chat_template=False,
            tensor_manifest={},
            validation_errors=[f"Checkpoint path '{checkpoint_path}' does not exist."],
        )

    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    config_path = os.path.join(checkpoint_path, "config.json")
    tok_config_path = os.path.join(checkpoint_path, "tokenizer_config.json")
    tok_json_path = os.path.join(checkpoint_path, "tokenizer.json")

    is_adapter = os.path.exists(adapter_config_path)
    has_config = os.path.exists(config_path)
    has_tokenizer = os.path.exists(tok_config_path) or os.path.exists(tok_json_path)

    base_model_id = None
    adapter_type = "lora"
    lora_r = None
    lora_alpha = None
    target_modules = []

    if is_adapter:
        try:
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                ac_data = json.load(f)
                base_model_id = ac_data.get("base_model_name_or_path")
                adapter_type = ac_data.get("peft_type", "lora").lower()
                lora_r = ac_data.get("r")
                lora_alpha = ac_data.get("lora_alpha")
                t_mods = ac_data.get("target_modules", [])
                if isinstance(t_mods, list):
                    target_modules = t_mods
                elif isinstance(t_mods, str):
                    target_modules = [t_mods]
        except Exception:
            pass

    has_chat_template = False
    if os.path.exists(tok_config_path):
        try:
            with open(tok_config_path, "r", encoding="utf-8") as f:
                tc_data = json.load(f)
                if "chat_template" in tc_data and tc_data["chat_template"]:
                    has_chat_template = True
        except Exception:
            pass

    tensor_manifest: Dict[str, TensorMetadata] = {}
    for root, _, files in os.walk(checkpoint_path):
        for file in files:
            if file.endswith(".safetensors"):
                full_path = os.path.join(root, file)
                headers = extract_tensor_headers(full_path)
                tensor_manifest.update(headers)

    return CheckpointManifest(
        checkpoint_path=checkpoint_path,
        is_adapter=is_adapter,
        base_model_id=base_model_id,
        adapter_type=adapter_type,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        has_config=has_config,
        has_tokenizer=has_tokenizer,
        has_chat_template=has_chat_template,
        tensor_manifest=tensor_manifest,
        validation_errors=[],
    )
