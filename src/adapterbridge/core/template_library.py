"""Canonical Jinja2 chat template library for supported model architectures."""

from typing import Dict, Optional


CANONICAL_TEMPLATES: Dict[str, str] = {
    # Llama 3 / 3.1 / 3.3
    "llama": (
        "{% for message in messages %}"
        "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{% endif %}"
    ),
    # Qwen 2 / 2.5 / ChatML
    "qwen2": (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    ),
    # Mistral / Mixtral
    "mistral": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
        "{% else %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    # DeepSeek R1 / V3
    "deepseek": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<｜User｜>' + message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<｜Assistant｜>' + message['content'] }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<｜Assistant｜>' }}"
        "{% endif %}"
    ),
}

# Aliases for architectural model_type strings
ARCHITECTURE_ALIASES: Dict[str, str] = {
    "llama": "llama",
    "llamaforcausalm": "llama",
    "qwen2": "qwen2",
    "qwen2forcausallm": "qwen2",
    "mistral": "mistral",
    "mistralforcausallm": "mistral",
    "mixtral": "mistral",
    "deepseek": "deepseek",
    "deepseekv2": "deepseek",
    "deepseekv3": "deepseek",
}


def get_canonical_template_for_architecture(architecture_or_model_type: Optional[str]) -> str:
    """Return canonical Jinja2 chat template for given architecture or model_type.
    
    Defaults to Llama-3 format if model type is missing or unknown.
    """
    if not architecture_or_model_type:
        return CANONICAL_TEMPLATES["llama"]

    normalized = str(architecture_or_model_type).lower().replace("_", "").replace("-", "")
    family = ARCHITECTURE_ALIASES.get(normalized, "llama")
    return CANONICAL_TEMPLATES.get(family, CANONICAL_TEMPLATES["llama"])
