"""Sandboxed Jinja2 chat template validation and execution environment."""

import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple
from jinja2.exceptions import TemplateSyntaxError
from jinja2.sandbox import SandboxedEnvironment


def _render_in_sandbox(template_str: str, render_kwargs: Dict[str, Any]) -> str:
    """Internal helper to render template in a SandboxedEnvironment."""
    env = SandboxedEnvironment(autoescape=False)
    # Common helper functions used in Hugging Face / vLLM chat templates
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    
    template = env.from_string(template_str)
    return template.render(**render_kwargs)


def validate_template_syntax(template_str: str) -> Tuple[bool, str]:
    """Check if template_str is valid Jinja2 syntax without executing it."""
    try:
        env = SandboxedEnvironment(autoescape=False)
        env.parse(template_str)
        return True, ""
    except TemplateSyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.message}"
    except Exception as e:
        return False, f"Template parse error: {str(e)}"


def render_template_sandboxed(
    template_str: str,
    messages: List[Dict[str, Any]],
    add_generation_prompt: bool = True,
    timeout_seconds: float = 2.0,
    extra_context: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Render a Jinja2 chat template with multi-turn messages inside a sandboxed thread/process.
    
    Returns (success, result_or_error_message).
    """
    valid_syntax, err = validate_template_syntax(template_str)
    if not valid_syntax:
        return False, err

    render_kwargs = {
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
        "unk_token": "<|unk|>",
    }
    if extra_context:
        render_kwargs.update(extra_context)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_render_in_sandbox, template_str, render_kwargs)
        try:
            rendered = future.result(timeout=timeout_seconds)
            return True, rendered
        except concurrent.futures.TimeoutError:
            return False, f"Template rendering timed out after {timeout_seconds}s"
        except Exception as e:
            return False, f"Template execution error: {str(e)}"
