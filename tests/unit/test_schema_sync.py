"""Unit tests for schema sync manager and rule profiles."""

import pytest
from adapterbridge.core.schema_sync import SchemaSyncManager, EngineRuleSet


def test_schema_sync_parse_target_spec():
    mgr = SchemaSyncManager()
    engine, ver = mgr.parse_target_spec("vllm@0.6.4")
    assert engine == "vllm"
    assert ver == "0.6.4"

    engine2, ver2 = mgr.parse_target_spec("sglang")
    assert engine2 == "sglang"
    assert ver2 == "latest"


def test_schema_sync_get_ruleset(tmp_path):
    mgr = SchemaSyncManager(cache_dir=str(tmp_path))
    rules = mgr.get_ruleset("vllm@0.6.4")
    assert rules.engine == "vllm"
    assert rules.version == "0.6.4"
    assert "embed_tokens" in rules.unsupported_target_modules
    assert rules.moe_requires_3d_stacked is True


def test_schema_sync_all_default_rulesets(tmp_path):
    mgr = SchemaSyncManager(cache_dir=str(tmp_path))
    synced = mgr.sync_schemas(force=True)
    assert "vllm" in synced
    assert "sglang" in synced
    assert "ollama" in synced
    assert "tensorrt" in synced
