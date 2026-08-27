"""Unit tests for Logit Canary & Zero-GPU activation probing."""

import pytest
from adapterbridge.core.canary import execute_canary_probe
from adapterbridge.models.manifest import CheckpointManifest


def test_canary_probe_detects_silent_dropping(tmp_path):
    manifest = CheckpointManifest(
        checkpoint_path=str(tmp_path),
        is_adapter=True,
        lora_r=16,
        lora_alpha=32.0,
        target_modules=["q_proj", "v_proj", "embed_tokens"],
    )
    res = execute_canary_probe(manifest, unsupported_modules=["embed_tokens", "lm_head"])
    assert res.passed is False
    assert "embed_tokens" in res.silent_dropped_modules
    assert "Silent drop vulnerability detected" in res.message


def test_canary_probe_passes_valid_modules(tmp_path):
    manifest = CheckpointManifest(
        checkpoint_path=str(tmp_path),
        is_adapter=True,
        lora_r=8,
        lora_alpha=16.0,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    res = execute_canary_probe(manifest, unsupported_modules=["embed_tokens", "lm_head"])
    assert res.passed is True
    assert len(res.silent_dropped_modules) == 0
