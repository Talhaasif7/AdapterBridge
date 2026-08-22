"""Unit tests for SARIF v2.1.0 and GitHub PR comment exporters."""

import tempfile
from adapterbridge.core.inspector import AdapterInspector
from tests.fixtures.synthetic_checkpoint import create_synthetic_checkpoint


def test_sarif_exporter():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_synthetic_checkpoint(tmpdir, include_config=False, drifted_keys=True)
        inspector = AdapterInspector(checkpoint_path=tmpdir, target_engine="vllm")
        report = inspector.run_diagnostics()

        sarif_dict = report.to_sarif()
        assert sarif_dict["version"] == "2.1.0"
        assert len(sarif_dict["runs"]) == 1
        assert "driver" in sarif_dict["runs"][0]["tool"]
        assert len(sarif_dict["runs"][0]["results"]) >= 1


def test_pr_comment_exporter():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_synthetic_checkpoint(tmpdir, include_config=True)
        inspector = AdapterInspector(checkpoint_path=tmpdir, target_engine="vllm")
        report = inspector.run_diagnostics()

        pr_comment = report.to_pr_comment()
        assert "AdapterBridge Pre-Flight Check" in pr_comment
        assert "PASSED" in pr_comment
