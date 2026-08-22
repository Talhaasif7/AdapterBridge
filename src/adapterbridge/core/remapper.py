"""Tensor state-dict key normalization and remediation plan execution engine."""

import json
import os
import shutil
import uuid
from typing import Dict
from adapterbridge.models.manifest import CheckpointManifest
from adapterbridge.models.report import RemediationAction, RemediationPlan
from adapterbridge.utils.safetensors_io import remap_safetensors_file


KNOWN_PREFIX_REMAPS = [
    ("base_model.model.model.layers.", "model.layers."),
    ("base_model.model.", "model."),
]


def normalize_tensor_key(key: str) -> str:
    """Normalize a tensor key using standard framework prefix stripping."""
    for old_prefix, new_prefix in KNOWN_PREFIX_REMAPS:
        if key.startswith(old_prefix):
            return key.replace(old_prefix, new_prefix, 1)
    return key


def execute_remediation_plan(manifest: CheckpointManifest, plan: RemediationPlan) -> str:
    """Execute a remediation plan using atomic staging directories.
    
    Returns the final destination directory path.
    """
    dst_path = os.path.abspath(plan.output_path)
    staging_dir = os.path.join(
        os.path.dirname(dst_path),
        f".adapterbridge_staging_{uuid.uuid4().hex[:8]}"
    )
    os.makedirs(staging_dir, exist_ok=True)

    try:
        # Copy source files to staging
        if os.path.exists(manifest.checkpoint_path):
            for item in os.listdir(manifest.checkpoint_path):
                s_item = os.path.join(manifest.checkpoint_path, item)
                d_item = os.path.join(staging_dir, item)
                if os.path.isfile(s_item):
                    shutil.copy2(s_item, d_item)
                elif os.path.isdir(s_item):
                    shutil.copytree(s_item, d_item)

        # Execute operations
        for op in plan.operations:
            if op.action == RemediationAction.SYNTHESIZE_FILE:
                target_rel = os.path.relpath(op.target_path, plan.output_path)
                staged_file = os.path.join(staging_dir, target_rel)
                os.makedirs(os.path.dirname(staged_file), exist_ok=True)

                if staged_file.endswith(".json"):
                    with open(staged_file, "w", encoding="utf-8") as f:
                        json.dump(op.details, f, indent=2)
                elif staged_file.endswith("Modelfile"):
                    with open(staged_file, "w", encoding="utf-8") as f:
                        f.write(f"FROM {op.details.get('base_model', 'llama3.1')}\n")
                        f.write(f"ADAPTER {op.details.get('adapter_path', './adapter_model.safetensors')}\n")

            elif op.action == RemediationAction.INJECT_CHAT_TEMPLATE:
                tok_config_path = os.path.join(staging_dir, "tokenizer_config.json")
                tc_data: Dict[str, str] = {}
                if os.path.exists(tok_config_path):
                    try:
                        with open(tok_config_path, "r", encoding="utf-8") as f:
                            tc_data = json.load(f)
                    except Exception:
                        pass
                tc_data["chat_template"] = op.details.get("chat_template", "")
                with open(tok_config_path, "w", encoding="utf-8") as f:
                    json.dump(tc_data, f, indent=2)

            elif op.action == RemediationAction.REMAP_TENSOR_KEY:
                key_map = op.details.get("key_map", {})
                for root, _, files in os.walk(staging_dir):
                    for file in files:
                        if file.endswith(".safetensors"):
                            staged_st = os.path.join(root, file)
                            temp_st = staged_st + ".tmp"
                            remap_safetensors_file(staged_st, temp_st, key_map)
                            os.replace(temp_st, staged_st)

        # Atomic rename into final destination
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        shutil.move(staging_dir, dst_path)
        return dst_path

    except Exception as e:
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise RuntimeError(f"Remediation execution failed: {str(e)}") from e
