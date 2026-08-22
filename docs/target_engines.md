# Target Engine Specifications

AdapterBridge validates fine-tuned LoRA checkpoints against target engine specifications.

---

## Specification Matrix

| Feature / File | vLLM | SGLang | Ollama | TensorRT-LLM |
|---|---|---|---|---|
| `config.json` | Required (`architectures`, `hidden_size`, `model_type`) | Required | Converted to Modelfile | Required |
| `adapter_config.json` | Standard PEFT schema | Standard PEFT schema | Converted to GGUF adapter | TRT LoRA map |
| Tensor Key Naming | `model.layers.0...` | `model.layers.0...` | GGUF key paths | TRT layer names |
| Chat Template | Jinja2 (`tokenizer_config.json`) | Jinja2 (Radix) | Ollama `TEMPLATE` | Prompt template |
| Supported Engine Versions | `>=0.6.0,<0.8.0` | `>=0.3.0,<0.5.0` | `>=0.3.0` | `>=0.9.0` |

---

## Target-Specific Behaviors

### 1. vLLM Target Engine (`vllm`)
- Validates presence of `config.json` and PEFT `adapter_config.json`.
- Flags key prefix drift (`base_model.model.model.` -> `model.`).
- Injects canonical Jinja2 template (`<|start_header_id|>`) if missing.

### 2. SGLang Target Engine (`sglang`)
- Validates base architectural parameters and Radix-compatible template rendering.

### 3. Ollama Target Engine (`ollama`)
- Generates `Modelfile` with `FROM <base_model>` and `ADAPTER <path>`.

### 4. TensorRT-LLM Target Engine (`tensorrt`)
- Asserts layer dimension compatibility for TRT engine compilation.
