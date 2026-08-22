"""Example 03: Programmatic SARIF v2.1.0 and GitHub PR comment report generation."""

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
    """Create a sample checkpoint with adapter config and dummy safetensors."""
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_type": "llama", "hidden_size": 4096}, f)
    with open(os.path.join(dir_path, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump({"r": 16, "lora_alpha": 32}, f)
    numpy_save_file({"model.layers.0.lora_A": np.zeros((16, 4096), dtype=np.float32)}, os.path.join(dir_path, "adapter_model.safetensors"))
    return dir_path


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_sample_checkpoint(tmpdir)
        inspector = AdapterInspector(checkpoint_path=tmpdir, target_engine=TargetEngine.VLLM)
        report = inspector.run_diagnostics()

        # 1. Export SARIF v2.1.0 JSON
        sarif_dict = report.to_sarif()
        print("--- SARIF v2.1.0 Schema Export ---")
        print(json.dumps(sarif_dict, indent=2)[:300] + "\n...")

        # 2. Export GitHub PR Comment Markdown
        pr_comment = report.to_pr_comment()
        print("\n--- GitHub PR Comment Export ---")
        print(pr_comment)


if __name__ == "__main__":
    main()
