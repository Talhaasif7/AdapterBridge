"""Unit tests for multi-architecture canonical chat template library."""

import pytest
from adapterbridge.core.template_library import (
    CANONICAL_TEMPLATES,
    get_canonical_template_for_architecture,
)
from adapterbridge.core.template_linter import lint_chat_template


@pytest.mark.parametrize("model_type,expected_family", [
    ("llama", "llama"),
    ("LlamaForCausalLM", "llama"),
    ("qwen2", "qwen2"),
    ("Qwen2ForCausalLM", "qwen2"),
    ("mistral", "mistral"),
    ("deepseek", "deepseek"),
    ("unknown_arch", "llama"),
])
def test_get_canonical_template_for_architecture(model_type, expected_family):
    template = get_canonical_template_for_architecture(model_type)
    assert template == CANONICAL_TEMPLATES[expected_family]


def test_canonical_templates_lint_pass():
    for family, template in CANONICAL_TEMPLATES.items():
        is_valid, errors = lint_chat_template(template)
        assert is_valid is True, f"Template for {family} failed linting: {errors}"
