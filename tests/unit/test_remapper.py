"""Unit tests for tensor key normalization, zero-copy remapping, and atomic remediation plan execution."""

import os
import tempfile
import pytest
from adapterbridge.core.inspector import AdapterInspector
from adapterbridge.core.remapper import normalize_tensor_key
from adapterbridge.utils.safetensors_io import extract_tensor_headers, remap_safetensors_headers_zero_copy
from tests.fixtures.synthetic_checkpoint import create_synthetic_checkpoint


def test_normalize_tensor_key():
    drifted = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    normalized = normalize_tensor_key(drifted)
    assert normalized == "model.layers.0.self_attn.q_proj.lora_A.weight"


def test_zero_copy_safetensors_remapping():
    with tempfile.TemporaryDirectory() as tmpdir:
        src_ckpt = create_synthetic_checkpoint(os.path.join(tmpdir, "src"), drifted_keys=True)
        src_st = os.path.join(src_ckpt, "adapter_model.safetensors")
        dst_st = os.path.join(tmpdir, "dst", "adapter_model.safetensors")
        os.makedirs(os.path.dirname(dst_st), exist_ok=True)

        key_map = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": "model.layers.0.self_attn.q_proj.lora_A.weight"
        }
        remap_safetensors_headers_zero_copy(src_st, dst_st, key_map)

        headers = extract_tensor_headers(dst_st)
        assert "model.layers.0.self_attn.q_proj.lora_A.weight" in headers


def test_auto_repair_drifted_checkpoint():
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
        dst_path = os.path.join(dst_dir, "repaired")
        create_synthetic_checkpoint(src_dir, include_config=False, drifted_keys=True)

        inspector = AdapterInspector(checkpoint_path=src_dir, target_engine="vllm")
        initial_report = inspector.run_diagnostics()
        assert initial_report.is_compatible is False

        remediation = inspector.auto_repair(destination_path=dst_path)
        assert len(remediation.operations) >= 1

        # Check repaired checkpoint compatibility
        repaired_inspector = AdapterInspector(checkpoint_path=dst_path, target_engine="vllm")
        repaired_report = repaired_inspector.run_diagnostics()
        assert repaired_report.is_compatible is True
