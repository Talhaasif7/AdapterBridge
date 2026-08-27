"""Direct chat-completion round-trip simulation engine and prompt formatting verifier."""

import re
from typing import Any, Dict, List, Optional
from jinja2 import Environment, TemplateSyntaxError
from pydantic import BaseModel, Field


class ChatRoundTripResult(BaseModel):
    """Diagnostic output of chat template rendering and tokenization verification."""

    success: bool
    rendered_prompt: str = ""
    issues: List[str] = Field(default_factory=list)
    has_delimiter_leakage: bool = False
    has_double_bos: bool = False
    has_system_truncation: bool = False


DEFAULT_TEST_MESSAGES: List[Dict[str, str]] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, world!"},
]


def execute_chat_roundtrip_test(
    template_str: str,
    messages: Optional[List[Dict[str, str]]] = None,
    bos_token: str = "<s>",
    eos_token: str = "</s>",
) -> ChatRoundTripResult:
    """Render chat template with test messages payload and check for subtle tokenization artifacts."""
    test_msgs = messages or DEFAULT_TEST_MESSAGES
    issues: List[str] = []
    has_delimiter_leakage = False
    has_double_bos = False
    has_system_truncation = False
    rendered = ""

    if not template_str or not template_str.strip():
        return ChatRoundTripResult(
            success=False,
            rendered_prompt="",
            issues=["Chat template string is empty or missing."],
        )

    try:
        env = Environment(autoescape=False)
        tpl = env.from_string(template_str)
        rendered = tpl.render(
            messages=test_msgs,
            add_generation_prompt=True,
            bos_token=bos_token,
            eos_token=eos_token,
            raise_exception=lambda msg: (_ for _ in ()).throw(RuntimeError(msg)),
        )
    except Exception as e:
        return ChatRoundTripResult(
            success=False,
            rendered_prompt="",
            issues=[f"Jinja2 template rendering execution failed: {str(e)}"],
        )

    # 1. Double-BOS check
    if rendered.startswith("<s><s>") or rendered.startswith("<|begin_of_text|><|begin_of_text|>"):
        has_double_bos = True
        issues.append("Double-BOS token injection detected at start of prompt (e.g., '<s><s>' or '<|begin_of_text|><|begin_of_text|>').")

    # 2. System prompt truncation check
    sys_content = next((m["content"] for m in test_msgs if m["role"] == "system"), None)
    if sys_content and sys_content not in rendered:
        has_system_truncation = True
        issues.append(f"System instruction content ('{sys_content}') was truncated or missing in rendered output.")

    # 3. Delimiter leakage check
    raw_user_content = next((m["content"] for m in test_msgs if m["role"] == "user"), "")
    # Detect unescaped or dangling delimiters
    dangling_delimiters = [
        delim for delim in ["<|im_end|>", "<|im_start|>", "<|eot_id|>", "[INST]", "[/INST]"]
        if delim in raw_user_content  # Should not leak raw delimiters inside payload
    ]
    if dangling_delimiters:
        has_delimiter_leakage = True
        issues.append(f"Raw delimiter leakage detected in user message content: {dangling_delimiters}.")

    # 4. Generation prompt check
    if not (rendered.rstrip().endswith("assistant\n") or rendered.rstrip().endswith("[/INST]") or rendered.rstrip().endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")):
        if "add_generation_prompt" in template_str and not rendered.endswith("\n"):
            issues.append("Rendered prompt lacks trailing newline or generation marker for completion suffix.")

    success = len(issues) == 0

    return ChatRoundTripResult(
        success=success,
        rendered_prompt=rendered,
        issues=issues,
        has_delimiter_leakage=has_delimiter_leakage,
        has_double_bos=has_double_bos,
        has_system_truncation=has_system_truncation,
    )
