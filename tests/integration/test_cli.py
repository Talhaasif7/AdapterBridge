"""Integration tests for Typer CLI commands."""

import os
import tempfile
from typer.testing import CliRunner
from adapterbridge.cli import app
from tests.fixtures.synthetic_checkpoint import create_synthetic_checkpoint

runner = CliRunner()


def test_cli_check_command():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_synthetic_checkpoint(tmpdir, include_config=True)
        result = runner.invoke(app, ["check", "--path", tmpdir, "--target", "vllm"])
        assert result.exit_code == 0
        assert "AdapterBridge Diagnostic" in result.stdout


def test_cli_check_json_format():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_synthetic_checkpoint(tmpdir, include_config=True)
        report_file = os.path.join(tmpdir, "report.json")
        result = runner.invoke(app, ["check", "--path", tmpdir, "--target", "vllm", "--format", "json", "--output", report_file])
        assert result.exit_code == 0
        assert os.path.exists(report_file)


def test_cli_fix_command():
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
        dst_path = os.path.join(dst_dir, "repaired")
        create_synthetic_checkpoint(src_dir, include_config=False)
        result = runner.invoke(app, ["fix", "--src", src_dir, "--dst", dst_path, "--target", "vllm"])
        assert result.exit_code == 0
        assert os.path.exists(dst_path)


def test_cli_verify_command():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_synthetic_checkpoint(tmpdir)
        result = runner.invoke(app, ["verify", "--path", tmpdir, "--target", "vllm", "--tensor-parallel-size", "2"])
        assert result.exit_code == 0
        assert "SUCCESS" in result.stdout
