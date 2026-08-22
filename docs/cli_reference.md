# AdapterBridge CLI Manual & Command Reference

The `adapterbridge` Command Line Interface provides four primary verbs for inspecting, repairing, simulating, and exporting fine-tuned LoRA checkpoints.

---

## 1. `adapterbridge check`

Static linter and compatibility validator. Inspects metadata, safetensors headers, and Jinja2 chat templates without GPU or weight materialization.

```bash
adapterbridge check --path <CHECKPOINT_PATH> --target <TARGET_ENGINE> [OPTIONS]
```

### Options

| Flag | Short | Description | Default |
|---|---|---|---|
| `--path` | `-p` | Path to checkpoint directory (Required) | - |
| `--target` | `-t` | Serving engine (`vllm`, `sglang`, `ollama`, `tensorrt`) | - |
| `--format` | `-f` | Output format (`text`, `json`, `sarif`, `markdown`, `pr-comment`) | `text` |
| `--output` | `-o` | Output file path destination | stdout |

### Examples

```bash
# Terminal table output
adapterbridge check --path ./checkpoint-500 --target vllm

# Output SARIF for GitHub Code Scanning
adapterbridge check --path ./checkpoint-500 --target vllm --format sarif --output report.sarif

# Output GitHub PR Markdown comment
adapterbridge check --path ./checkpoint-500 --target vllm --format pr-comment
```

---

## 2. `adapterbridge fix`

Automated normalizer & metadata synthesizer. Remaps state-dict key drift using zero-copy binary streaming and injects missing base model configurations.

```bash
adapterbridge fix --src <SRC_PATH> --dst <DST_PATH> --target <TARGET_ENGINE> [OPTIONS]
```

### Options

| Flag | Short | Description | Default |
|---|---|---|---|
| `--src` | `-s` | Source checkpoint path (Required) | - |
| `--dst` | `-d` | Destination path for repaired checkpoint (Required) | - |
| `--target` | `-t` | Target engine (`vllm`, `sglang`, `ollama`, `tensorrt`) | - |
| `--base-model` | `-b` | Fallback base model ID (e.g. `meta-llama/Llama-3.1-8B-Instruct`) | Auto-resolved |

### Example

```bash
adapterbridge fix \
  --src ./raw-checkpoint \
  --dst ./production-checkpoint \
  --target vllm \
  --base-model meta-llama/Llama-3.1-8B-Instruct
```

---

## 3. `adapterbridge verify`

Zero-GPU mock dry-run engine. Simulates rank math, layer shapes, memory footprint, and Tensor Parallelism (TP) sharding division.

```bash
adapterbridge verify --path <CHECKPOINT_PATH> --target <TARGET_ENGINE> [OPTIONS]
```

### Options

| Flag | Short | Description | Default |
|---|---|---|---|
| `--path` | `-p` | Path to checkpoint directory (Required) | - |
| `--target` | `-t` | Target engine | - |
| `--tensor-parallel-size` | `-tp` | Tensor Parallelism degree (e.g. `2`, `4`, `8`) | `1` |

### Example

```bash
adapterbridge verify --path ./production-checkpoint --target vllm --tensor-parallel-size 4
```

---

## 4. `adapterbridge export`

Bundles a verified checkpoint for target runtime deployment.

```bash
adapterbridge export --path ./checkpoint --target ollama --output ./ollama-package
```
