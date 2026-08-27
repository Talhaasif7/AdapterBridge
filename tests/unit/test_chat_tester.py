"""Unit tests for chat-completion round-trip simulation engine."""

import pytest
from adapterbridge.core.chat_tester import execute_chat_roundtrip_test


def test_chat_tester_detects_double_bos():
    tpl = "<s><s>{% for m in messages %}{{ m.content }}{% endfor %}"
    res = execute_chat_roundtrip_test(tpl)
    assert res.success is False
    assert res.has_double_bos is True


def test_chat_tester_detects_system_truncation():
    tpl = "{% for m in messages %}{% if m.role == 'user' %}{{ m.content }}{% endif %}{% endfor %}"
    res = execute_chat_roundtrip_test(tpl)
    assert res.success is False
    assert res.has_system_truncation is True


def test_chat_tester_valid_template():
    tpl = "<s>[INST] {% for m in messages %}{% if m.role == 'system' %}{{ m.content }}\n{% endif %}{% if m.role == 'user' %}{{ m.content }}{% endif %}{% endfor %} [/INST]"
    res = execute_chat_roundtrip_test(tpl)
    assert res.success is True
    assert "You are a helpful assistant." in res.rendered_prompt
    assert "Hello, world!" in res.rendered_prompt
