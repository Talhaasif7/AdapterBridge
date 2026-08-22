"""Chat template linter and evaluation test vectors."""

from typing import List, Tuple
from adapterbridge.utils.jinja_sandbox import render_template_sandboxed, validate_template_syntax


TEST_CONVERSATION_VECTORS = [
    # 1. Simple user prompt
    [{"role": "user", "content": "Hello, world!"}],
    # 2. Multi-turn system + user + assistant
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "Tell me more."},
    ],
]


def lint_chat_template(template_str: str) -> Tuple[bool, List[str]]:
    """Validate a Jinja2 chat template against standard conversation test vectors.
    
    Returns (is_valid, list_of_errors).
    """
    valid_syntax, err = validate_template_syntax(template_str)
    if not valid_syntax:
        return False, [err]

    errors: List[str] = []
    for idx, vector in enumerate(TEST_CONVERSATION_VECTORS, start=1):
        success, res = render_template_sandboxed(
            template_str=template_str,
            messages=vector,
            add_generation_prompt=True,
            timeout_seconds=2.0,
        )
        if not success:
            errors.append(f"Vector #{idx} render error: {res}")
            break
        elif not res or len(res.strip()) == 0:
            errors.append(f"Vector #{idx} produced empty output.")
            break

    return len(errors) == 0, errors
