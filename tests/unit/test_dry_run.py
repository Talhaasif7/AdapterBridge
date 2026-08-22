"""Unit tests for zero-GPU mock serving dry run verification."""

import tempfile
from adapterbridge.core.inspector import AdapterInspector
from tests.fixtures.synthetic_checkpoint import create_synthetic_checkpoint


def test_dry_run_verification_pass():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_synthetic_checkpoint(tmpdir, r=16, alpha=32)
        inspector = AdapterInspector(checkpoint_path=tmpdir, target_engine="vllm")
        res = inspector.verify_dry_run(tensor_parallel_size=2)
        
        assert res.success is True
        assert res.tensor_parallel_size == 2
        assert res.memory_estimate_mb > 0
