"""Synthetic test fixture generator for creating temporary malformed and valid LoRA checkpoints."""

import json
import os
import tempfile
import numpy as np
from safetensors.numpy import save_file as numpy_save_file


def create_synthetic_checkpoint(
    dir_path: str,
    include_config: bool = True,
    include_adapter_config: bool = True,
    include_tokenizer: bool = True,
    include_chat_template: bool = True,
    drifted_keys: bool = False,
    r: int = 16,
    alpha: int = 32,
) -> str:
    """Create a synthetic LoRA adapter directory structure for testing."""
    os.makedirs(dir_path, exist_ok=True)

    if include_config:
        config_data = {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "model_type": "llama",
            "num_attention_heads": 32,
        }
        with open(os.path.join(dir_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_data, f)

    if include_adapter_config:
        adapter_config = {
            "base_model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
            "peft_type": "LORA",
            "r": r,
            "lora_alpha": alpha,
            "target_modules": ["q_proj", "v_proj"],
        }
        with open(os.path.join(dir_path, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump(adapter_config, f)

    if include_tokenizer:
        tok_data = {
            "model_max_length": 4096,
        }
        if include_chat_template:
            tok_data["chat_template"] = (
                "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\n' }}{% endfor %}"
            )
        with open(os.path.join(dir_path, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(tok_data, f)

    # Create dummy safetensors tensors
    prefix = "base_model.model.model.layers.0.self_attn." if drifted_keys else "model.layers.0.self_attn."
    tensors = {
        f"{prefix}q_proj.lora_A.weight": np.zeros((r, 4096), dtype=np.float32),
        f"{prefix}q_proj.lora_B.weight": np.zeros((4096, r), dtype=np.float32),
        f"{prefix}v_proj.lora_A.weight": np.zeros((r, 4096), dtype=np.float32),
        f"{prefix}v_proj.lora_B.weight": np.zeros((4096, r), dtype=np.float32),
    }
    numpy_save_file(tensors, os.path.join(dir_path, "adapter_model.safetensors"))

    return dir_path
