# AdapterBridge

**LoRA Checkpoint & Config Compatibility Engine for Enterprise Inference Runtimes**

[![CI Workflow](https://github.com/adapterbridge/adapterbridge/actions/workflows/ci.yml/badge.svg)](https://github.com/adapterbridge/adapterbridge/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/adapterbridge.svg)](https://pypi.org/project/adapterbridge/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Docker Image](https://img.shields.io/badge/docker-ghcr.io%2Fadapterbridge-blue)](https://ghcr.io/adapterbridge/adapterbridge)

AdapterBridge is an open-source pre-flight compatibility, validation, and remediation engine that eliminates runtime deployment failures when moving fine-tuned Low-Rank Adaptation (LoRA/QLoRA) checkpoints into production inference runtimes (**vLLM**, **SGLang**, **Ollama**, **TensorRT-LLM**).

---

## Executive Summary & Case Study

### 1. Problem Statement

Post-training frameworks (Unsloth, Axolotl, LLaMA-Factory, Hugging Face PEFT) make fine-tuning open-weight foundation models (Llama 3.x, Qwen 2.5, Mistral, DeepSeek) accessible. However, exported checkpoint directories are frequently incompatible with production serving runtimes without manual intervention.

Incompatibilities are typically discovered only **after** a deployment has already consumed compute resources:

| Failure Mode | Root Cause | Downstream Impact |
|---|---|---|
| **Missing Config Files** | Training exports include `adapter_model.safetensors` + `adapter_config.json` only, omitting `config.json` or tokenizer bundles. | Serving engine process crashes at startup before accepting requests. |
| **State-Dict Key Drift** | Inconsistent framework prefixes (`base_model.model.model.layers.0...` vs. `model.layers.0...`). | Weight loading fails or silently drops layers. |
| **Malformed Chat Templates** | Custom Jinja2 chat templates with syntax errors or unhandled message roles. | OpenAI-compatible `/v1/chat/completions` endpoint fails during request processing. |
| **Dimension & Rank Mismatch** | `lora_alpha`/`r` or `target_modules` do not align with tensor parallelism sharding expectations. | Matrix shape errors at load time across multi-GPU clusters. |

### 2. Operational Cost & Pre-Flight Mitigation

In distributed inference environments (Kubernetes, Ray, Slurm), catching configuration mismatches at load time wastes compute budget and delays deployment pipelines.

```
Traditional Workflow (Late Failure):
Training Export -> CI/CD Deploy -> GPU Pod Scheduled -> Weights Load -> Load Error (High GPU Cost)

AdapterBridge Workflow (Pre-Flight Gateway):
Training Export -> AdapterBridge Check (CPU, <2s) -> Validation & Remediation -> Clean Deployment (Zero GPU Cost)
```

AdapterBridge operates as a pre-flight validation gateway: static schema inspection, automated metadata synthesis, zero-copy binary streaming tensor key normalization, and CPU-only dry-run simulation before weights reach host VRAM or GPU pods.

---

## System Architecture

```
+-------------------------------------------------------------+
|                Presentation & Ingestion Layer               |
|   CLI (Typer + Rich)   |  Python SDK  |  CI/CD Integrations |
+-------------------------------+-----------------------------+
                                |
                                v
+-------------------------------------------------------------+
|                     AdapterBridge Core Engine               |
|  +---------------+ +---------------+ +-------------------+  |
|  | Metadata      | | Tensor &      | | Chat Template     |  |
|  | Scanner /     | | Weight        | | Linter & Sandbox  |  |
|  | Lineage Engine| | Remapper      | | (Sandboxed Jinja2)|  |
|  +---------------+ +---------------+ +-------------------+  |
|  +-------------------------------------------------------+  |
|  |     Mock Serving Dry-Run Harness (PyTorch Meta/NumPy) |  |
|  +-------------------------------------------------------+  |
+-------------------------------+-----------------------------+
                                |
                                v
+-------------------------------------------------------------+
|                  Target Engine Profiles                     |
|   vLLM  |  SGLang  |  Ollama  |  TensorRT-LLM  | (Pluggable)|
+-------------------------------------------------------------+
```

### Core Design Guarantees

1. **Zero-GPU Pre-Flight Verification**: Every command (`check`, `fix`, `verify`) executes on standard CPUs, local workstations, and CI runners.
2. **Lazy Memory Inspection**: Uses memory-mapped header parsing (`safetensors.safe_open`) to inspect tensor headers, shapes, and dtypes without loading weight buffers into RAM.
3. **Zero-Copy Binary Remapping**: Re-serializes header offsets and streams raw byte buffers directly on disk without converting float arrays into Python memory.
4. **Atomic Staging Writes**: File modifications execute in temporary staging directories (`.adapterbridge_staging_<uuid>`) and atomically rename into place only after validation succeeds.
5. **Sandboxed Template Evaluation**: Jinja2 chat templates evaluate inside `jinja2.sandbox.SandboxedEnvironment` with subprocess execution timeouts.

---

## Core Capabilities (The 4 Verbs)

```
                  +------------------+
                  | Input Checkpoint |
                  +--------+---------+
                           |
     +---------------------+---------------------+---------------------+
     |                     |                     |                     |
     v                     v                     v                     v
+---------+           +---------+           +----------+          +----------+
|  CHECK  |           |   FIX   |           |  VERIFY  |          |  EXPORT  |
+---------+           +---------+           +----------+          +----------+
Static Linter       Auto-Remapper         Zero-GPU Dry-Run      Target Bundler
& Validator         & Synthesizer         Memory Simulator      & Exporter
```

### 1. `adapterbridge check`
Static schema inspection and linting against target serving engine rules (vLLM, SGLang, Ollama, TensorRT-LLM). Supports text table, JSON, SARIF, Markdown, and PR comment formats.

### 2. `adapterbridge fix`
Automated metadata synthesis and zero-copy key normalization. Resolves base model lineage from Hugging Face Hub with local disk caching (`~/.cache/adapterbridge/hub/`) and injects canonical, architecture-matched chat templates.

### 3. `adapterbridge verify`
Zero-GPU mock dry-run engine simulating layer allocation, matrix scaling (`scale = alpha / r`), and Tensor Parallelism (TP) sharding division using PyTorch `meta` device tensors (`torch.empty(..., device="meta")`) or NumPy shape math.

### 4. `adapterbridge export`
Bundles a verified checkpoint into target runtime directory layouts.

---

## Installation

### Standard CPU Installation

```bash
pip install adapterbridge
```

### Installation with PyTorch Meta-Tensor Support

```bash
pip install adapterbridge[verify]
```

### Development Installation

```bash
git clone https://github.com/adapterbridge/adapterbridge.git
cd AdapterBridge
pip install -e .[dev,verify]
```

---

## Usage Scenarios

### Scenario 1: CLI Inspection and Auto-Repair

#### 1. Static Linter Check
```bash
adapterbridge check --path ./checkpoint-500 --target vllm
```

#### 2. Auto-Repair Checkpoint
```bash
adapterbridge fix \
  --src ./checkpoint-500 \
  --dst ./checkpoint-500-repaired \
  --target vllm \
  --base-model meta-llama/Llama-3.1-8B-Instruct
```

#### 3. Verify Repaired Checkpoint
```bash
adapterbridge check --path ./checkpoint-500-repaired --target vllm
```

#### 4. Dry-Run Verification for Tensor Parallelism
```bash
adapterbridge verify --path ./checkpoint-500-repaired --target vllm --tensor-parallel-size 4
```

---

### Scenario 2: GitHub Actions CI/CD Integration

Add `.github/workflows/adapterbridge.yml` to your repository:

```yaml
name: Adapter Pre-Flight Check

on:
  pull_request:
    paths:
      - 'checkpoints/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run AdapterBridge Check
        uses: adapterbridge/adapterbridge/.github/actions/adapterbridge-check@main
        with:
          path: './checkpoints/adapter-latest'
          target: 'vllm'
          format: 'sarif'
          output: 'adapterbridge.sarif'

      - name: Upload Security & Compliance SARIF Report
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'adapterbridge.sarif'
```

---

### Scenario 3: Local Pre-Commit Hook Integration

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/adapterbridge/adapterbridge
    rev: v0.1.1
    hooks:
      - id: adapterbridge-check
        args: ["--path", "./checkpoints/my-adapter", "--target", "vllm"]
```

---

### Scenario 4: Containerized Execution (Docker)

Run inside Kubernetes admission gates or MLOps pipelines:

```bash
docker run --rm -v $(pwd)/checkpoint:/workspace/checkpoint \
  ghcr.io/adapterbridge/adapterbridge:latest check --path /workspace/checkpoint --target vllm
```

---

## Python SDK Reference

```python
from adapterbridge import AdapterInspector, TargetEngine

# Initialize inspector
inspector = AdapterInspector(
    checkpoint_path="./checkpoints/my-adapter",
    target_engine=TargetEngine.VLLM,
)

# Inspect Checkpoint Manifest
manifest = inspector.manifest
print(f"Base model ID: {manifest.base_model_id}")
print(f"Rank r: {manifest.lora_r}, Alpha: {manifest.lora_alpha}")

# Run Diagnostics
report = inspector.run_diagnostics()
if not report.is_compatible:
    print(f"Validation failed with {len(report.errors)} error(s). Executing auto-repair...")
    
    # Auto-repair checkpoint
    plan = inspector.auto_repair(
        destination_path="./checkpoints/my-adapter-production",
        fallback_base_model="meta-llama/Llama-3.1-8B-Instruct",
    )
    print(f"Repaired output path: {plan.output_path}")

# Run Dry-Run Verification
verification = inspector.verify_dry_run(tensor_parallel_size=2)
print(f"Dry-run status: {verification.success}")
print(f"Estimated RAM footprint: {verification.memory_estimate_mb} MB")
```

---

## Target Engine Specification Matrix

| Feature / Requirement | vLLM (`vllm`) | SGLang (`sglang`) | Ollama (`ollama`) | TensorRT-LLM (`tensorrt`) |
|---|---|---|---|---|
| `config.json` | Required (`model_type`, `architectures`, `hidden_size`) | Required | Converted into Modelfile | Required |
| `adapter_config.json` | Standard PEFT schema (`r`, `lora_alpha`, `target_modules`) | Standard PEFT schema | Converted to GGUF adapter | Mapped to TRT LoRA layers |
| Tensor Key Format | `model.layers.0...` | `model.layers.0...` | GGUF key paths | TRT layer names |
| Chat Template | Jinja2 (`tokenizer_config.json`) | Jinja2 (Radix format) | Ollama `TEMPLATE` string | Formatted prompt template |
| Supported Engine Versions | `>=0.6.0,<0.8.0` | `>=0.3.0,<0.5.0` | `>=0.3.0` | `>=0.9.0` |

---

## Documentation

Explore topic-specific guides in the `docs/` directory:

- [Core Principles & Overview](docs/index.md)
- [CLI Command Manual & Reference](docs/cli_reference.md)
- [Python SDK Reference & Custom Plugin Guide](docs/sdk_reference.md)
- [Target Engine Specifications](docs/target_engines.md)
- [CI/CD & MLOps Integration Guide](docs/cicd_integration.md)

---

## License

AdapterBridge is released under the [Apache 2.0 License](LICENSE).
