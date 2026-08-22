"""Unit tests for Jinja2 template linting and sandbox isolation."""

from adapterbridge.core.template_linter import lint_chat_template
from adapterbridge.utils.jinja_sandbox import validate_template_syntax, render_template_sandboxed


def test_valid_template():
    template = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
    valid, errs = lint_chat_template(template)
    assert valid is True
    assert len(errs) == 0


def test_invalid_syntax_template():
    invalid_template = "{% for m in messages %}{{ m.role }"
    valid, err = validate_template_syntax(invalid_template)
    assert valid is False
    assert "Syntax error" in err


def test_sandboxed_render_timeout():
    # Infinite loop template attempt
    loop_template = "{% for i in range(100000000) %}{{ i }}{% endfor %}"
    messages = [{"role": "user", "content": "hi"}]
    success, res = render_template_sandboxed(loop_template, messages, timeout_seconds=0.5)
    assert success is False
    assert "timed out" in res or "execution error" in res.lower()
