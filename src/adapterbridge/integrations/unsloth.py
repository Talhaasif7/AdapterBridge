"""Unsloth integration export hook for AdapterBridge compatibility verification."""

from typing import Any, Dict, Optional
from adapterbridge.core.inspector import AdapterInspector


def verify_unsloth_export(
    checkpoint_dir: str,
    target_engine: str = "vllm",
    raise_on_error: bool = False,
) -> Dict[str, Any]:
    """Verify saved Unsloth adapter checkpoint against target inference runtime (e.g. vLLM or SGLang).

    Usage in Unsloth export workflow:
        from adapterbridge.integrations.unsloth import verify_unsloth_export
        verify_unsloth_export("outputs/lora_model", target_engine="vllm")
    """
    inspector = AdapterInspector(checkpoint_path=checkpoint_dir, target_engine=target_engine)
    report = inspector.run_diagnostics()

    result = {
        "is_compatible": report.is_compatible,
        "target_engine": report.target_engine,
        "checkpoint_path": checkpoint_dir,
        "error_count": len(report.errors),
        "warning_count": len(report.warnings),
        "summary": report.summary,
    }

    if not report.is_compatible and raise_on_error:
        raise ValueError(
            f"Unsloth export verification failed for target engine '{target_engine}': {report.summary}"
        )

    return result
