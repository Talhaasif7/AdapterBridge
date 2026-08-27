"""Unit tests for ecosystem integration hooks and HF bot."""

import pytest
from adapterbridge.integrations.unsloth import verify_unsloth_export
from adapterbridge.integrations.axolotl import verify_axolotl_checkpoint
from adapterbridge.integrations.hf_bot import generate_hf_badge_markdown, generate_hf_pr_description
from adapterbridge.models.report import ValidationReport


def test_hf_badge_markdown_generation():
    badge = generate_hf_badge_markdown("vllm", is_compatible=True)
    assert "[![AdapterBridge Certified]" in badge
    assert "VLLM%20Compatible" in badge

    badge_incomp = generate_hf_badge_markdown("sglang", is_compatible=False)
    assert "SGLANG%20Incompatible" in badge_incomp


def test_unsloth_export_hook(tmp_path):
    res = verify_unsloth_export(str(tmp_path), target_engine="vllm", raise_on_error=False)
    assert "is_compatible" in res
    assert res["target_engine"] == "vllm"


def test_axolotl_checkpoint_hook(tmp_path):
    res = verify_axolotl_checkpoint(str(tmp_path), target_engine="vllm", auto_fix=False)
    assert "is_compatible" in res
    assert res["target_engine"] == "vllm"
