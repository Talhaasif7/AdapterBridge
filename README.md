# AdapterBridge

**LoRA Checkpoint & Config Compatibility Engine for Enterprise Inference Runtimes**

[![CI Workflow](https://img.shields.io/badge/CI%20Workflow-passing-brightgreen.svg)](https://github.com/adapterbridge/adapterbridge/actions)
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
| **"Silent Dropping" Trap** | Runtime kernels discard `embed_tokens` and `lm_head` adapters while applying projection deltas. | Model loads without errors but fine-tuned behavior is silently stripped. |
| **Upstream Engine Churn** | Hardcoded layer mapping rules break when vLLM or SGLang releases minor updates. | Validation rules lag upstream engine releases. |
| **MoE 2D vs. 3D Stacked Keys** | Conflicting 2D per-expert weight matrices vs 3D fused expert tensors (`is_3d_lora_weight: true`). | Tensor dimension mismatches during expert routing. |
| **State-Dict Key Drift** | Inconsistent framework prefixes (`base_model.model.model.layers.0...` vs. `model.layers.0...`). | Weight loading fails or silently drops layers. |
| **Malformed Chat Templates** | Custom Jinja2 chat templates with syntax errors, double-BOS tokens (`<s><s>`), or delimiter leakage (`<|im_end|>`). | OpenAI-compatible `/v1/chat/completions` endpoint fails during request processing. |

---

## High-Performance Architectural Enhancements

```
┌────────────────────────────────────────────────────────────────────────┐
│             ADVANCED ADAPTERBRIDGE RE-ENGINEERED PIPELINE              │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
      ┌────────────────────────────┼────────────────────────────┐
      ▼                            ▼                            ▼
┌──────────────┐             ┌──────────────┐             ┌──────────────┐
│  Lightweight │             │ Dynamic Spec │             │ Logit Canary │
│ Engine Core  │             │ Cloud Sync   │             │ Verification │
│ (Zero Torch, │             │ (Auto-fetch  │             │ (Micro CPU   │
│  Pure Rust/  │             │  latest vLLM │             │  activation  │
│  Safetensors)│             │  rule-sets)  │             │  assertion)  │
└──────────────┘             └──────────────┘             └──────────────┘
```

1. **Zero-Torch Architecture (Ultra-Lightweight Core):**
   - Core CLI, linting, header parsing, auto-fixing, and chat-template validation execute in pure Python using memory-mapped `safetensors` and `jinja2` (< 15MB base installation).
   - Eliminates PyTorch/CUDA installation friction in CI/CD pipelines.

2. **Dynamic Target Profiles via Remote Schema Sync:**
   - Decouples target engine rules from Python package releases.
   - Run `adapterbridge check --target vllm@0.6.4` to fetch or cache dynamic rule specifications.
   - Use `adapterbridge sync` to refresh local rule specifications.

3. **Logit Canary & Activation Probing (Zero-GPU Activation Check):**
   - CPU micro-activation pass computing $\Delta \mathbf{h} = (\alpha / r) \mathbf{B}\mathbf{A}\mathbf{x}$ on memory-mapped safetensors.
   - Asserts $\lVert\Delta \mathbf{h}\rVert_2 > \epsilon$ and flags silent module dropping (`embed_tokens`, `lm_head`).

4. **MoE 2D vs. 3D Stacked Weight Inspector:**
   - Analyzes Mixture-of-Experts (MoE) tensor layouts (DeepSeek-V3, Mixtral, Qwen-2.5-MoE).
   - Validates 2D per-expert weight matrices (`experts.0.gate_proj`) against 3D fused expert tensors (`experts.gate_up_proj.weight`).

5. **Direct Chat-Completion Round-Trip Tester:**
   - Renders OpenAI-compatible payload messages (`{"messages": [...]}`) through Jinja templates.
   - Asserts against delimiter leakage (`<|im_end|>`, `[INST]`), double-BOS token injection (`<s><s>`), and system instruction truncation.

---

## System Architecture

```
+-------------------------------------------------------------+
|                Presentation & Ingestion Layer               |
|   CLI (Typer + Rich)   |  Python SDK  |  CI/CD Gatekeepers  |
+-------------------------------+-----------------------------+
                                |
                                v
+-------------------------------------------------------------+
|                     AdapterBridge Core Engine               |
|  +---------------+ +---------------+ +-------------------+  |
|  | Dynamic Schema| | Zero-GPU      | | Logit Canary      |  |
|  | Sync Manager  | | Safetensors   | | Activation      |  |
|  | (vllm@0.6.4)  | | Header Engine | | Probe (||Δh||)  |  |
|  +---------------+ +---------------+ +-------------------+  |
|  +---------------+ +---------------+ +-------------------+  |
|  | MoE 2D/3D     | | Chat Template | | Live Cluster      |  |
|  | Stacked       | | Round-Trip    | | Endpoint        |  |
|  | Inspector     | | Verifier      | | Doctor          |  |
|  +---------------+ +---------------+ +-------------------+  |
+-------------------------------+-----------------------------+
                                |
                                v
+-------------------------------------------------------------+
|                  Target Engine Specifications               |
|   vLLM  |  SGLang  |  Ollama  |  TensorRT-LLM  | (Pluggable)|
+-------------------------------------------------------------+
```

---

## Core Commands & Capabilities

```
                  +------------------+
                  | Input Checkpoint |
                  +--------+---------+
                           |
     +---------------------+---------------------+---------------------+
     |                     |                     |                     |
     v                     v                     v                     v
+---------+           +---------+           +----------+          +----------+
|  CHECK  |           |   FIX   |           |  DOCTOR  |          |   SYNC   |
+---------+           +---------+           +----------+          +----------+
Static Linter       One-Command           Live Endpoint         Dynamic Remote
& Canary Probe      Auto-Remapper         Health Doctor         Schema Refresh
```

### 1. `adapterbridge check`
Static schema inspection, versioned target rules (`vllm@0.6.4`), zero-GPU canary activation probing (`--canary`), and chat template round-trip testing (`--chat-test`).

### 2. `adapterbridge fix`
One-command automated normalizer and metadata synthesizer. Resolves base model lineage from Hugging Face Hub, fixes unsupported target modules, normalizes state-dict prefixes, and injects architecture-matched chat templates.

### 3. `adapterbridge doctor`
Operational health checker querying live, active inference endpoints (`/v1/models`, `/v1/chat/completions`) to verify that the cluster is actively serving adapter weights rather than silently falling back to the base model.

### 4. `adapterbridge sync`
Synchronizes remote target engine rule specifications into local schema cache.

---

## Installation

### Standard Ultra-Lightweight Core Installation (< 15MB)

```bash
pip install adapterbridge
```

### Installation with PyTorch Meta-Tensor Dry-Run Support

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

## Usage Examples

### Scenario 1: CLI Inspection and One-Command Fix

```bash
# 1. Run compatibility check with canary probing and chat testing
adapterbridge check --path ./my-lora-adapter --target vllm@0.6.4 --canary --chat-test

# 2. Execute automated fix
adapterbridge fix --src ./my-lora-adapter --dst ./my-lora-vllm --target vllm

# 3. Synchronize dynamic remote schemas
adapterbridge sync

# 4. Probe live running inference cluster endpoint
adapterbridge doctor --endpoint http://vllm-service:8000/v1 --adapter-id sql-lora-v1
```

---

### Scenario 2: GitHub Actions CI/CD Gatekeeper

Add `.github/workflows/adapterbridge-validate.yml`:

```yaml
name: Validate Adapter Deployment

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  adapterbridge-preflight:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Validate Adapter Checkpoints
        uses: adapterbridge/action-validate@v1
        with:
          checkpoint-path: ./checkpoints/latest
          target-engine: vllm@0.6.4
          fail-on-warning: false
```

---

### Scenario 3: Kubernetes Pre-Flight Init-Container

Run as an init-container in Kubernetes pods to validate or auto-repair mounted adapter volumes before serving servers launch:

```yaml
initContainers:
  - name: adapterbridge-preflight
    image: ghcr.io/adapterbridge/adapterbridge-k8s:latest
    env:
      - name: ADAPTER_PATH
        value: "/mnt/adapters/latest"
      - name: TARGET_ENGINE
        value: "vllm@0.6.4"
      - name: AUTO_FIX
        value: "true"
    volumeMounts:
      - name: adapter-volume
        mountPath: /mnt/adapters
```

---

### Scenario 4: Training Framework Integration (Unsloth / Axolotl)

```python
from adapterbridge.integrations.unsloth import verify_unsloth_export

# Post-export hook in Unsloth fine-tuning pipeline
verify_unsloth_export("outputs/lora_model", target_engine="vllm@0.6.4", raise_on_error=True)
```

---

## Python SDK Reference

```python
from adapterbridge import AdapterInspector, TargetEngine

# Initialize inspector with versioned target engine
inspector = AdapterInspector(
    checkpoint_path="./checkpoints/my-adapter",
    target_engine="vllm@0.6.4",
)

# Run Diagnostics
report = inspector.run_diagnostics()

# Run Canary Activation Probe
canary_res = inspector.run_canary_probe()
print(f"Canary probe passed: {canary_res.passed}, delta norm: {canary_res.activation_delta_norm}")

# Run Chat Template Round-Trip Test
chat_res = inspector.run_chat_test()
print(f"Chat roundtrip status: {chat_res.success}")

# Auto-repair checkpoint
if not report.is_compatible:
    plan = inspector.auto_repair(
        destination_path="./checkpoints/my-adapter-repaired",
        fallback_base_model="meta-llama/Llama-3.1-8B-Instruct",
    )
```

---

## Target Engine Specification Matrix

| Feature / Requirement | vLLM (`vllm`) | SGLang (`sglang`) | Ollama (`ollama`) | TensorRT-LLM (`tensorrt`) |
|---|---|---|---|---|
| `config.json` | Required (`model_type`, `hidden_size`) | Required | Converted into Modelfile | Required |
| `adapter_config.json` | Standard PEFT schema | Standard PEFT schema | Converted to GGUF adapter | Mapped to TRT LoRA layers |
| Unsupported Target Modules | `embed_tokens`, `lm_head` | `embed_tokens` | None | None |
| MoE Weight Format | Fused 3D stacked weights (`is_3d_lora_weight: true`) | Fused 3D stacked weights | 2D per-expert matrices | Fused 3D stacked weights |
| Chat Template | Jinja2 (`tokenizer_config.json`) | Jinja2 (Radix format) | Ollama `TEMPLATE` string | Formatted prompt template |
| Dynamic Schema Sync | Enabled (`vllm@0.6.4`) | Enabled (`sglang@0.2.0`) | Enabled | Enabled |

---

## License

AdapterBridge is released under the [Apache 2.0 License](LICENSE).
