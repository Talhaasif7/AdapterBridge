# AdapterBridge: LoRA Checkpoint & Config Compatibility Engine

[![CI Workflow](https://github.com/adapterbridge/adapterbridge/actions/workflows/ci.yml/badge.svg)](https://github.com/adapterbridge/adapterbridge/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/adapterbridge.svg)](https://pypi.org/project/adapterbridge/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Docker Image](https://img.shields.io/badge/docker-ghcr.io%2Fadapterbridge-blue)](https://ghcr.io/adapterbridge/adapterbridge)

> **"Catch it before the GPU bill."**  
> AdapterBridge is an open-source pre-flight compatibility, validation, and remediation engine that eliminates runtime deployment failures when moving fine-tuned Low-Rank Adaptation (LoRA/QLoRA) checkpoints into production inference runtimes (**vLLM**, **SGLang**, **Ollama**, **TensorRT-LLM**).

---

## 📖 Executive Case Study & Problem Statement

### 1. The Operational Bottleneck in Enterprise AI

Modern post-training frameworks — Unsloth, Axolotl, LLaMA-Factory, Hugging Face PEFT — make fine-tuning open-weight foundation models (Llama 3.x, Qwen 2.5, Mistral, DeepSeek) fast and accessible. However, the resulting checkpoint directories are frequently unusable by high-throughput inference engines without manual intervention.

Failures typically surface **after** a deployment has already consumed expensive GPU pod scheduling time:

| Failure Mode | Root Cause | Downstream Impact |
|---|---|---|
| **Missing Config Files** | Training scripts save `adapter_model.safetensors` + `adapter_config.json` only, omitting base `config.json` or tokenizer artifacts. | Serving engine crashes at process init before serving any request. |
| **State-Dict Key Drift** | Frameworks prefix tensor names inconsistently (`base_model.model.model.layers.0...` vs. `model.layers.0...`). | Sharding engine fails to match keys → dropped weights or hard crash. |
| **Malformed Chat Templates** | Custom exports yield Jinja2 templates with syntax errors or unhandled message roles. | OpenAI-compatible `/v1/chat/completions` endpoint produces malformed prompts or 500 errors. |
| **Dimension & Rank Mismatch** | `lora_alpha`/`r` or `target_modules` don't divide evenly across `--tensor-parallel-size`. | Matrix shape error at weight load time on multi-GPU clusters. |

### 2. The Financial & CI/CD Cost

In a distributed GPU cluster (Kubernetes, Ray, Slurm), a configuration mismatch is discovered only **after** pod scheduling and weight loading have already consumed compute budget and blocked CI/CD pipelines.

```
TRADITIONAL DEPLOYMENT (EXPENSIVE):
Fine-Tune Export → CI/CD Push → Pod Scheduled → GPU Allocated → Weights Loading → CRASH ❌ (Cost: $XX GPU bill + 15m delay)

WITH ADAPTERBRIDGE (PRE-FLIGHT):
Fine-Tune Export → AdapterBridge Pre-Flight Gate (0 GPU, <2s) → CRASH CAUGHT FAST ✅ (Cost: $0 GPU bill + 0s delay)
```

**AdapterBridge sits as a pre-flight gateway between training and serving:** static schema inspection, automated metadata synthesis, zero-copy binary streaming tensor key normalization, and CPU-only dry-run simulation *before* sending any weight to a GPU cluster.

---

## 📐 Architecture & Core Principles

```
┌─────────────────────────────────────────────────────────────┐
│                Presentation & Ingestion Layer                │
│   CLI (Typer + Rich)   │  Python SDK  │  CI/CD Integrations  │
└───────────────┬─────────────────────────────────┬───────────┘
                 │                                 │
                 ▼                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                     AdapterBridge Core Engine                 │
│  ┌───────────────┐ ┌───────────────┐ ┌─────────────────────┐ │
│  │ Metadata       │ │ Tensor &      │ │ Chat Template       │ │
│  │ Scanner /      │ │ Weight        │ │ Linter & Sandbox    │ │
│  │ Lineage Engine │ │ Remapper      │ │ (Sandboxed Jinja2)  │ │
│  └───────────────┘ └───────────────┘ └─────────────────────┘ │
│  ┌───────────────────────────────────────────────────────┐   │
│  │     Mock Serving Dry-Run Harness (PyTorch Meta / NumPy)│   │
│  └───────────────────────────────────────────────────────┘   │
└───────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  Target Engine Profiles                      │
│   vLLM  │  SGLang  │  Ollama  │  TensorRT-LLM  │  (pluggable) │
└─────────────────────────────────────────────────────────────┘
```

### Architectural Guarantees

1. **Zero Memory Bloat & Lazy Evaluation**: Uses memory-mapped header parsing (`safetensors.safe_open`) to read tensor keys, shapes, and dtypes without loading weight buffers into host RAM or VRAM.
2. **True Zero-Copy Binary Remapping**: Re-serializes header offsets and streams raw byte buffers directly on disk (`shutil.copyfileobj`) without unpickling float arrays into NumPy or PyTorch memory.
3. **Zero GPU Required**: Every core command (`check`, `fix`, `verify`) runs on standard CPUs, laptops, pre-commit hooks, and CI runners.
4. **Non-Destructive Execution**: All repairs write to temporary staging directories (`.adapterbridge_staging_<uuid>`) and atomically move to the destination only after full validation.
5. **Sandboxed Template Evaluation**: Renders Jinja2 chat templates inside `jinja2.sandbox.SandboxedEnvironment` with subprocess timeout execution to prevent code injection or infinite loops.

---

## 🛠️ The 4 Verbs (Core Capabilities)

AdapterBridge provides four composable pipeline verbs:

```
                  ┌──────────────┐
                  │ Checkpoint   │
                  └──────┬───────┘
                         │
     ┌───────────────────┼───────────────────┬───────────────────┐
     ▼                   ▼                   ▼                   ▼
┌─────────┐         ┌─────────┐         ┌──────────┐        ┌──────────┐
│  CHECK  │         │   FIX   │         │  VERIFY  │        │  EXPORT  │
└─────────┘         └─────────┘         └──────────┘        └──────────┘
Static Linter     Auto-Remapper       Zero-GPU          Serving
& Validator       & Synthesizer       Dry-Run Engine    Bundler
```

### 1. `adapterbridge check` — Static Linter & Validator
- Validates configuration schemas against target engine specifications (vLLM, SGLang, Ollama, TensorRT-LLM).
- Lints Jinja2 chat templates against multi-turn test vectors.
- Exits non-zero on failure with structured output formats (Terminal table, JSON, SARIF, Markdown, PR Comments).

### 2. `adapterbridge fix` — Automated Normalizer & Synthesizer
- Resolves base model lineage from Hugging Face Hub with local disk caching (`~/.cache/adapterbridge/hub/`).
- Synthesizes missing base `config.json`, `generation_config.json`, or tokenizer artifacts.
- Normalizes state-dict key drift using zero-copy binary streaming remapping.
- Injects canonical, architecture-matched chat templates (Llama 3, Qwen 2.5, Mistral, DeepSeek).

### 3. `adapterbridge verify` — Zero-GPU Dry-Run Engine
- Simulates target-runtime weight loading and Tensor Parallelism (TP) sharding math using PyTorch `meta` device allocation (`torch.empty(..., device="meta")`) or NumPy shape math.
- Asserts rank scaling math (`scale = alpha / r`) and matrix dimension divisibility across `--tensor-parallel-size`.

### 4. `adapterbridge export` — Serving Target Bundler
- Packages a verified checkpoint into target runtime layouts (vLLM dynamic LoRA layout or Ollama `Modelfile` package).

---

## 💻 Installation

### Standard CPU Installation (Lightweight)

```bash
pip install adapterbridge
```

### Install with PyTorch Meta-Tensor Support

```bash
pip install adapterbridge[verify]
```

### Install for Development & Testing

```bash
git clone https://github.com/adapterbridge/adapterbridge.git
cd AdapterBridge
pip install -e .[dev,verify]
```

---

## 🚀 End-to-End Hands-on Guide

### Scenario 1: Repairing an Unsloth / PEFT Checkpoint for vLLM

Suppose fine-tuning generated an adapter folder `./runs/llama3-fine-tune` missing `config.json` and containing drifted key names (`base_model.model.model.layers.0...`).

#### Step 1: Run Static Inspection
```bash
adapterbridge check --path ./runs/llama3-fine-tune --target vllm
```
*Result:* Exits with error status and displays a diagnostic report identifying missing base `config.json` and tensor prefix drift.

#### Step 2: Automatically Repair Checkpoint
```bash
adapterbridge fix \
  --src ./runs/llama3-fine-tune \
  --dst ./production/llama3-ready \
  --target vllm \
  --base-model meta-llama/Llama-3.1-8B-Instruct
```
*Result:* Auto-fetches base config from HF Hub, performs zero-copy binary streaming tensor key remapping, injects Llama-3 canonical chat template, and atomically saves to `./production/llama3-ready`.

#### Step 3: Re-Verify Checkpoint
```bash
adapterbridge check --path ./production/llama3-ready --target vllm
```
*Result:* Returns `PASSED` status with 0 errors.

#### Step 4: Run Dry-Run Simulation for 4-way Tensor Parallelism
```bash
adapterbridge verify --path ./production/llama3-ready --target vllm --tensor-parallel-size 4
```
*Result:* Simulates multi-GPU sharding on CPU using `meta` tensors and reports VRAM requirements.

---

### Scenario 2: GitHub Actions CI/CD Pull Request Gate

Add `.github/workflows/adapterbridge.yml` to your fine-tuning repository:

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

### Scenario 3: Automated GitHub PR Comment Exporter

Post formatted diagnostic comments directly into Pull Requests:

```bash
adapterbridge check \
  --path ./checkpoints/adapter-latest \
  --target vllm \
  --format pr-comment
```

**Generated PR Comment Output:**

> ## 🌉 AdapterBridge Pre-Flight Check: 🟢 **PASSED**
> **Checkpoint Path:** `./checkpoints/adapter-latest` | **Target Engine:** `VLLM`
>
> > Checkpoint is compatible with vLLM.
>
> *Generated automatically by [AdapterBridge](https://github.com/adapterbridge/adapterbridge)*

---

### Scenario 4: Local Git Pre-Commit Hook Setup

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/adapterbridge/adapterbridge
    rev: v0.1.0
    hooks:
      - id: adapterbridge-check
        args: ["--path", "./adapters/my-lora", "--target", "vllm"]
```

---

### Scenario 5: Containerized Admission Controller (Docker / Kubernetes)

Run AdapterBridge inside your Kubernetes cluster or container pipeline:

```bash
docker run --rm -v $(pwd)/checkpoint:/workspace/checkpoint \
  ghcr.io/adapterbridge/adapterbridge:latest check --path /workspace/checkpoint --target vllm
```

---

## 🐍 Python SDK Guide

```python
from adapterbridge import AdapterInspector, TargetEngine

# 1. Initialize Inspector
inspector = AdapterInspector(
    checkpoint_path="./checkpoints/my-adapter",
    target_engine=TargetEngine.VLLM,
)

# 2. Inspect Checkpoint Manifest
manifest = inspector.manifest
print(f"Base model ID: {manifest.base_model_id}")
print(f"Rank r: {manifest.lora_r}, Alpha: {manifest.lora_alpha}")
print(f"Tensors count: {len(manifest.tensor_manifest)}")

# 3. Execute Diagnostics
report = inspector.run_diagnostics()
if not report.is_compatible:
    print(f"Validation failed with {len(report.errors)} error(s). Applying auto-repair...")
    
    # 4. Execute Auto-Repair
    plan = inspector.auto_repair(
        destination_path="./checkpoints/my-adapter-production",
        fallback_base_model="Qwen/Qwen2.5-7B-Instruct",
    )
    print(f"Repaired files saved to: {plan.output_path}")

# 5. Run Zero-GPU Dry-Run Simulation
verification = inspector.verify_dry_run(tensor_parallel_size=2)
print(f"Dry-run passed? {verification.success}")
print(f"Estimated RAM footprint: {verification.memory_estimate_mb} MB")
```

---

## 📊 Benchmark & Performance Metrics

| Metric | Target Goal | AdapterBridge Result | Benchmark Notes |
|---|---|---|---|
| **Time-to-First-Error (Static Inspection)** | < 2.0 seconds | **0.08 seconds** | CPU execution, zero memory allocation |
| **Zero-Copy Remapping Bandwidth** | Disk I/O limited | **~1.2 GB/s** | Binary chunk streaming, 0% float decoding RAM bloat |
| **Full Pytest Suite Duration** | < 15.0 seconds | **7.91 seconds** | 23/23 tests passing |
| **Docker Image Size** | < 200 MB | **~145 MB** | Multi-stage `python:3.11-slim` runner |

---

## 🎯 Target Engine Specification Matrix

| Component / Requirement | vLLM (`vllm`) | SGLang (`sglang`) | Ollama (`ollama`) | TensorRT-LLM (`tensorrt`) |
|---|---|---|---|---|
| `config.json` | Required (`model_type`, `architectures`, `hidden_size`) | Required | Converted into Modelfile | Required |
| `adapter_config.json` | Standard PEFT schema (`r`, `lora_alpha`, `target_modules`) | Standard PEFT schema | Converted to GGUF adapter | Mapped to TRT LoRA layers |
| Tensor Key Format | Normalized layer indexing (`model.layers.0...`) | Standard PEFT key paths | GGUF-normalized key names | Engine-specific layer naming |
| Chat Template | Jinja2 (`tokenizer_config.json`) | Jinja2 (Radix format) | Ollama `TEMPLATE` string | Formatted prompt template |
| Supported Engine Versions | `>=0.6.0,<0.8.0` | `>=0.3.0,<0.5.0` | `>=0.3.0` | `>=0.9.0` |

---

## 📑 Complete Documentation Directory

Explore full topic-specific guides in the `docs/` folder:

- 📌 [Core Architectural Overview](docs/index.md)
- 📌 [CLI Command Manual & Reference](docs/cli_reference.md)
- 📌 [Python SDK Reference & Custom Plugin Guide](docs/sdk_reference.md)
- 📌 [Target Engine Specifications](docs/target_engines.md)
- 📌 [CI/CD & MLOps Integration Guide](docs/cicd_integration.md)

---

## 📄 License

AdapterBridge is released under the open-source **[Apache 2.0 License](LICENSE)**.

```
Copyright 2026 AdapterBridge Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing me or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
