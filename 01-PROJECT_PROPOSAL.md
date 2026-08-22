# Project Proposal: AdapterBridge

**LoRA Checkpoint & Config Compatibility Engine for Enterprise Inference**

> Status: Draft v1.1 (reviewed & restructured for implementation)
> Companion document: `02-ARCHITECTURE.md`

---

## 1. Executive Summary & Vision

AdapterBridge is an open-source compatibility, validation, and conversion engine that eliminates runtime deployment failures when moving fine-tuned Low-Rank Adaptation (LoRA/QLoRA) checkpoints into production inference runtimes.

Post-training frameworks — Unsloth, Axolotl, LLaMA-Factory, Hugging Face PEFT — make fine-tuning accessible, but the resulting checkpoints are frequently unusable by high-throughput inference engines such as vLLM, SGLang, TensorRT-LLM, and Ollama without manual intervention. Failures typically show up **after** a deployment has already consumed GPU time: missing configuration metadata, drifted state-dict key names, malformed Jinja2 chat templates, or tensor shapes that don't match the target engine's sharding expectations.

AdapterBridge sits between the training side and the serving side as a **pre-flight compatibility layer**: static schema inspection, automated metadata synthesis, zero-copy tensor key normalization, and a CPU-only dry-run engine that proves a checkpoint will load correctly *before* it is ever sent to a GPU cluster.

**Engineering note:** the single idea that makes this product defensible is *"catch it before the GPU bill."* Every feature should be evaluated against that bar — if a check can only run after weights are loaded onto a GPU, it belongs in the inference engine's own validation, not in AdapterBridge.

---

## 2. Problem Statement & Root Cause Analysis

### 2.1 The Operational Bottleneck

When fine-tuning open-weight foundation models (Llama 3.x, Qwen 2.x, Mistral, DeepSeek variants), training exports routinely produce non-standard, incomplete directory structures:

| Failure Mode | Cause | Downstream Effect |
|---|---|---|
| **Missing / stripped config files** | Training scripts save `adapter_model.safetensors` + `adapter_config.json` only, omitting `config.json`, `generation_config.json`, or a complete tokenizer bundle | Serving engine fails at *process init*, before any request is served |
| **State-dict key drift** | Frameworks prefix tensor names inconsistently (`base_model.model.model.layers.0...` vs. `model.layers.0...`) | Sharding/loading engine can't resolve keys → silently dropped weights or hard crash |
| **Malformed chat templates** | Custom tokenizer exports produce Jinja2 templates with syntax errors or missing role delimiters | OpenAI-compatible `/v1/chat/completions` endpoint fails or produces malformed prompts |
| **Rank/dimension mismatch** | `lora_alpha`/`r` or `target_modules` don't match the frozen base layer's `in_features`/`out_features` | Matrix-multiply shape error at load time or, worse, silent incorrect inference |

### 2.2 Why This Is Expensive

In a distributed GPU cluster, a configuration mismatch is typically discovered only **after** pod scheduling and weight loading have already consumed compute budget and blocked a CI/CD pipeline stage. The cost of a failure scales with how late it's caught — AdapterBridge's entire value proposition is moving that discovery point as far left as possible (ideally: a laptop, pre-commit hook, or CI runner with no GPU attached).

---

## 3. Target Audience & User Personas

| Persona | Primary Environment | Core Pain Point | Value Proposition |
|---|---|---|---|
| **MLOps / Platform Engineers** | Kubernetes, Ray, Slurm, vLLM, Triton | Incompatible checkpoint formats break automated CI/CD deployment pipelines | Automated pre-flight validation gates and Kubernetes admission checks |
| **Applied AI / Fine-Tuning Engineers** | Unsloth, Axolotl, Modal, RunPod | Manual scripting and hand-editing configs to get checkpoints loading in vLLM/Ollama | Single-command CLI for automated schema repair and format conversion |
| **Model Registry Maintainers** | Hugging Face Hub, internal S3/GCS registries | Community or internal models fail to run due to incomplete metadata | Automated repository scanning, validation reports, non-destructive PR generation |

---

## 4. Product Scope & Core Capabilities

AdapterBridge ships as a unified **CLI** and a **Python SDK** built around four verbs. Each verb is intentionally a separate, composable pipeline stage rather than one monolithic command — this matters for CI/CD, where `check` and `verify` need to be usable as independent gates.

### `adapterbridge check` — Static Linter & Validator
- Inspects local directories or remote Hugging Face Hub repositories (read-only, no writes).
- Validates configuration schemas against a selected target engine specification (vLLM, SGLang, Ollama, TensorRT-LLM).
- Parses and validates Jinja2 chat templates against a fixed suite of multi-turn conversation test vectors.
- Exits non-zero on failure with a structured report — this is the command CI pipelines call.

### `adapterbridge fix` — Automated Normalizer & Synthesizer
- Resolves base-model lineage from `adapter_config.json` (`base_model_name_or_path`) and synthesizes any missing base configuration files.
- Normalizes safetensors state-dict keys by stripping/remapping inconsistent framework prefixes.
- Validates and injects a standard chat template compatible with the OpenAI chat-completions schema.
- **Never mutates the source directory** — always writes to a destination path (see §8 of the architecture doc).

### `adapterbridge verify` — In-Process Mock Serving & Dry-Run Engine
- Simulates target-runtime memory allocation, layer projection, and weight sharding using **meta tensors / shape-only computation** — no GPU allocation, no full weight materialization.
- Asserts structural compatibility with the target tensor-parallelism configuration (i.e., do hidden dims divide evenly across `--tensor-parallel-size`).

### `adapterbridge export` — Serving-Targeted Bundler
- Packages a verified checkpoint for a specific runtime target: merged safetensors, a vLLM dynamic-LoRA folder layout, or an Ollama `Modelfile` + GGUF adapter package.

---

## 5. Competitive Landscape & Positioning

| Dimension | Training Tools (Unsloth, Axolotl) | Framework Abstractions (HF PEFT) | Inference Runtimes (vLLM, SGLang) | **AdapterBridge** |
|---|---|---|---|---|
| Primary focus | Training speed, VRAM reduction | PyTorch adapter implementation | Runtime throughput, paged attention | Serving compatibility & validation |
| Target-engine awareness | Low — produces raw output artifacts | Low — agnostic to runtime requirements | High, but only as strict input expectations | High — multi-engine target validation |
| Pre-flight verification | None | None | None (fails at load/runtime) | Zero-GPU dry-run simulation harness |
| Metadata remediation | Manual, user's responsibility | Manual, user's responsibility | None | Automated synthesis and key remapping |

AdapterBridge is not competing with any of these tools — it's the missing seam between them. This should be the framing in all messaging and docs: *"we don't train, and we don't serve; we make sure what comes out of the left hand fits into the right hand."*

---

## 6. Release Milestones & Delivery Roadmap

| Phase | Version | Timeline | Scope |
|---|---|---|---|
| **1 — Foundation & Core Linter** | v0.1.0 | Month 1 | Core CLI (Typer + Rich); static validation of `adapter_config.json`, `tokenizer.json`, `tokenizer_config.json`; support for Llama 3/3.1 and Qwen 2/2.5 fine-tuned via Unsloth/PEFT; target profiles for vLLM dynamic-LoRA and Ollama |
| **2 — Automated Synthesis & Weight Remapping** | v0.2.0 | Month 2 | Base-model lineage resolution via Hugging Face Hub; zero-copy safetensors header/key remapping engine; Jinja2 chat-template linting, repair, and fallback injection |
| **3 — Ephemeral Dry-Run Engine** | v0.3.0 | Month 3 | Mock serving harness for vLLM/SGLang load simulation on CPU using meta tensors; diagnostic exports in JSON, Markdown, and SARIF |
| **4 — CI/CD Integrations & Enterprise Features** | v0.4.0+ | Months 4–6 | GitHub Action + GitLab CI runner plugins; pre-commit hooks; Docker base image for registry admission gates |

**Revision note:** the original roadmap listed Click as the CLI framework in the architecture doc but Rich alone in the proposal. Standardizing on **Typer** (built on Click, adds type-hint-driven commands and much less boilerplate) + **Rich** for output formatting — see architecture doc §7 for rationale.

---

## 7. Go-to-Market & Commercialization Strategy

### 7.1 Developer Acquisition
- **Issue remediation-as-marketing:** open PRs that fix broken/incomplete configs on popular public fine-tune repos, with AdapterBridge's diagnostic report attached — the report *is* the pitch.
- **Upstream integration:** contribute `--verify-with-adapterbridge` export flags to fine-tuning tools (Unsloth, Axolotl, LLaMA-Factory).
- **Automated compatibility bot:** a scheduled bot that scans public Hugging Face Hub fine-tunes, flags broken metadata, and opens PRs with fixes generated by `adapterbridge fix`.

  ⚠️ **Scope/legal caution:** automatically opening PRs against third-party repositories at scale needs clear rate limits, an opt-out mechanism for maintainers, and Hugging Face Hub's bot/automation policies respected. Treat this as a v0.4+ initiative gated on legal/community review, not a Phase-1 deliverable.

### 7.2 Commercialization Model (Open-Core)

**Open-Source Core (Apache 2.0):** CLI, Python SDK, all format converters, standard model target profiles, and the local dry-run engine.

**Enterprise Platform ("AdapterBridge Cloud"):**
- **Private Registry Gatekeeper** — continuous background scanning of private S3/GCS buckets and internal model registries.
- **Cluster Pre-Flight Verifier** — a Kubernetes admission controller that validates checkpoints before GPU pod scheduling.
- **Compliance & Lineage Auditing** — cryptographic verification of base weights, adapter tensors, and (where declared) alignment dataset provenance, for regulated environments.

---

## 8. Success Metrics (added)

To ground the roadmap in something measurable, Phase 1–2 should target:

- **Time-to-first-error:** median time from `adapterbridge check` invocation to a complete diagnostic report, on a checkpoint with no network calls, should be **< 2 seconds** (this is a static-inspection tool; it should feel instantaneous).
- **False-negative rate on load failures:** of checkpoints that `adapterbridge check` marks compatible, the percentage that still fail to load in the real target engine. This is the core trust metric and should be tracked via an integration-test corpus (see architecture doc §9) that runs against real vLLM/Ollama installs in CI.
- **Repair success rate:** percentage of `check`-failing checkpoints that `adapterbridge fix` converts into `check`-passing checkpoints without human intervention.

---

## 9. Key Risks (added)

| Risk | Mitigation |
|---|---|
| Target inference engines change internal config expectations across versions, silently invalidating target profiles | Pin and test against specific engine versions per target profile; version the profiles independently of the AdapterBridge core release cadence |
| "Dry-run passes but real load fails" erodes trust faster than any other bug class | Maintain the integration-test corpus (real, pinned installs of vLLM/Ollama/SGLang) as a release gate, not just unit tests against mocks |
| Executing untrusted Jinja2 templates from arbitrary downloaded checkpoints is a code-execution risk | Sandbox strictly (see architecture doc §8); never execute a template without sandboxing, even in `check` mode |
| Scope creep into full training or full serving (competing with the ecosystem instead of connecting it) | Keep the "we don't train, we don't serve" positioning as a literal PR/issue-triage rule |
