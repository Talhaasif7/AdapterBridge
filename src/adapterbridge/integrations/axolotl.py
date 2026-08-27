"""Axolotl training pipeline integration hook for AdapterBridge."""

from typing import Any, Dict
from adapterbridge.core.inspector import AdapterInspector


def verify_axolotl_checkpoint(
    checkpoint_dir: str,
    target_engine: str = "vllm",
    auto_fix: bool = False,
) -> Dict[str, Any]:
    """Pre-flight check or auto-repair hook for Axolotl fine-tuning outputs.

    Usage in Axolotl callback:
        from adapterbridge.integrations.axolotl import verify_axolotl_checkpoint
        verify_axolotl_checkpoint("./qwen-lora", target_engine="vllm", auto_fix=True)
    """
    inspector = AdapterInspector(checkpoint_path=checkpoint_dir, target_engine=target_engine)
    report = inspector.run_diagnostics()

    repaired_path = None
    if not report.is_compatible and auto_fix:
        dst = f"{checkpoint_dir.rstrip('/\\\\')}-vllm"
        plan = inspector.auto_repair(destination_path=dst)
        repaired_path = plan.output_path

    return {
        "is_compatible": report.is_compatible,
        "checkpoint_path": checkpoint_dir,
        "target_engine": target_engine,
        "repaired_path": repaired_path,
        "summary": report.summary,
    }
