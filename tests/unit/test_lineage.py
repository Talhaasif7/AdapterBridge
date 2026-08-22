"""Unit tests for metadata scanner and lineage resolution."""

import tempfile
import pytest
from adapterbridge.core.lineage import scan_checkpoint
from tests.fixtures.synthetic_checkpoint import create_synthetic_checkpoint


def test_scan_valid_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_synthetic_checkpoint(tmpdir, include_config=True, include_adapter_config=True)
        manifest = scan_checkpoint(tmpdir)

        assert manifest.is_adapter is True
        assert manifest.has_config is True
        assert manifest.base_model_id == "meta-llama/Llama-3.1-8B-Instruct"
        assert manifest.lora_r == 16
        assert manifest.lora_alpha == 32
        assert len(manifest.tensor_manifest) == 4


def test_scan_missing_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_synthetic_checkpoint(tmpdir, include_config=False)
        manifest = scan_checkpoint(tmpdir)

        assert manifest.has_config is False
        assert manifest.is_adapter is True
