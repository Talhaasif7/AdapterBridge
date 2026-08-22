"""Example 01: Quickstart Python SDK usage for AdapterBridge."""

import json
import os
import sys
import tempfile
import numpy as np
from safetensors.numpy import save_file as numpy_save_file

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapterbridge import AdapterInspector, TargetEngine


def create_sample_checkpoint(dir_path: str) -> str:
    """Create a sample checkpoint missing base config and containing drifted tensor keys."""
    os.makedirs(dir_path, exist_ok=True)
    
    # Save adapter_config.json
    adapter_config = {
        "base_model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
    }
    with open(os.path.join(dir_path, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(adapter_config, f)

    # Save safetensors with drifted key names
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": np.zeros((16, 4096), dtype=np.float32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": np.zeros((4096, 16), dtype=np.float32),
    }
    numpy_save_file(tensors, os.path.join(dir_path, "adapter_model.safetensors"))
    return dir_path


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_sample_checkpoint(tmpdir)
        print(f"Created sample raw checkpoint in: {tmpdir}")

        # 1. Initialize Inspector against vLLM target engine
        inspector = AdapterInspector(checkpoint_path=tmpdir, target_engine=TargetEngine.VLLM)

        # 2. Run Diagnostics
        report = inspector.run_diagnostics()
        print("\n--- Diagnostic Report ---")
        print(f"Is compatible? {report.is_compatible}")
        print(f"Errors count: {len(report.errors)}")
        for issue in report.issues:
            print(f"  - [{issue.severity.value.upper()}] [{issue.code}]: {issue.message}")

        # 3. Apply Auto-Repair
        repaired_dir = tmpdir + "_repaired"
        plan = inspector.auto_repair(
            destination_path=repaired_dir,
            fallback_base_model="meta-llama/Llama-3.1-8B-Instruct"
        )
        print("\n--- Auto-Repair Executed ---")
        print(f"Repaired files saved to: {plan.output_path}")

        # 4. Verify Repaired Checkpoint
        repaired_inspector = AdapterInspector(checkpoint_path=repaired_dir, target_engine=TargetEngine.VLLM)
        repaired_report = repaired_inspector.run_diagnostics()
        print(f"Repaired checkpoint compatible? {repaired_report.is_compatible}")

        # 5. Run Zero-GPU Dry-Run Simulation
        dry_run = repaired_inspector.verify_dry_run(tensor_parallel_size=2)
        print(f"Dry-run success? {dry_run.success} | RAM footprint: {dry_run.memory_estimate_mb} MB")


if __name__ == "__main__":
    main()
