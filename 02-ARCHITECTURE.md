# Architecture Specification: AdapterBridge

**Technical Blueprint, Subsystems, and Engine Specifications**

> Status: Draft v1.1 (reviewed for buildability)
> Companion document: `01-PROJECT_PROPOSAL.md`

---

## 1. System Architectural Principles & Design Goals

1. **Zero memory bloat, lazy evaluation.** Never load full model weights into host RAM or GPU memory during inspection. Use memory-mapped I/O (`safetensors.safe_open`) to read headers, tensor keys, shapes, and dtypes without materializing tensor byte buffers.
2. **Stateless and deterministic execution.** Identical inputs + identical target profile version → identical diagnostic output. No hidden global state, no reliance on wall-clock time in report content (timestamps go in report *metadata* only).
3. **Target-engine decoupling.** Inspection/remediation logic never imports or depends on a specific serving engine's runtime code. All engine-specific knowledge lives behind the `BaseTargetProfile` interface (§8.2).
4. **Non-destructive by default.** All repair operations write to a new destination directory. In-place modification of the source checkpoint requires an explicit `--in-place` flag and is discouraged in documentation.
5. **No GPU required, ever, for `check`/`fix`/`verify`.** This is the product's core promise; any dependency that pulls in CUDA-only code paths for these three commands is a bug.

---

## 2. System Architecture & Component Hierarchy

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
│  │ Metadata       │ │ Tensor &      │ │ Chat Template        │ │
│  │ Scanner /      │ │ Weight        │ │ Linter & Sandbox      │ │
│  │ Lineage Engine │ │ Remapper      │ │                       │ │
│  └───────────────┘ └───────────────┘ └─────────────────────┘ │
│  ┌───────────────────────────────────────────────────────┐   │
│  │            Mock Serving Dry-Run Harness                │   │
│  └───────────────────────────────────────────────────────┘   │
└───────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  Target Specification Layer                  │
│   vLLM  │  SGLang  │  Ollama  │  TensorRT-LLM  │  (pluggable) │
└─────────────────────────────────────────────────────────────┘
```

- **Presentation & Ingestion Layer** — CLI (Typer for the command surface, Rich for tables/progress), a programmatic Python SDK, and CI/CD integrations (GitHub Action, GitLab CI, pre-commit).
- **Core Engine** — the four subsystems described in §4, orchestrated by `inspector.py`.
- **Target Specification Layer** — one `BaseTargetProfile` implementation per serving engine, loaded via a plugin registry (§8.2) so new engines can be added without touching core code.

---

## 3. Directory Layout & Module Structure

```
adapterbridge/
├── pyproject.toml
├── src/
│   └── adapterbridge/
│       ├── __init__.py
│       ├── cli.py                       # Typer entry points
│       ├── core/
│       │   ├── inspector.py             # Top-level orchestration
│       │   ├── lineage.py               # Base model config resolution
│       │   ├── remapper.py              # Tensor key normalization
│       │   ├── template_linter.py       # Jinja2 validation sandbox
│       │   └── dry_run.py               # Mock serving simulation
│       ├── models/
│       │   ├── manifest.py              # Pydantic: CheckpointManifest
│       │   ├── target_spec.py           # Serving engine target definitions
│       │   └── report.py                # Diagnostic / remediation report schemas
│       ├── targets/                     # Target profiles (plugin registry)
│       │   ├── registry.py              # Discovers profiles via entry_points
│       │   ├── vllm.py
│       │   ├── sglang.py
│       │   ├── ollama.py
│       │   └── tensorrt.py
│       └── utils/
│           ├── safetensors_io.py        # Memory-mapped tensor inspection
│           ├── jinja_sandbox.py         # Sandboxed template rendering
│           └── hub.py                   # Hugging Face Hub resolution (cached, offline-friendly)
└── tests/
    ├── fixtures/                        # Small synthetic checkpoints (see §9)
    ├── test_inspector.py
    ├── test_remapper.py
    ├── test_template_linter.py
    ├── test_dry_run.py
    └── integration/                     # Real-engine smoke tests, run in a separate CI job
        ├── test_vllm_load.py
        └── test_ollama_load.py
```

---

## 4. Core Subsystems & Technical Implementation

### 4.1 Checkpoint Inspection & Metadata Scanner (`lineage.py`, `inspector.py`)

Parses a target directory or a remote Hugging Face Hub repo into an in-memory `CheckpointManifest`. This is the single canonical data structure every other subsystem reads from and writes diagnostics against.

```python
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class TensorMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    shape: List[int]
    dtype: str
    n_elements: int


class CheckpointManifest(BaseModel):
    checkpoint_path: str
    is_adapter: bool
    base_model_id: Optional[str] = None
    adapter_type: Optional[str] = "lora"
    lora_r: Optional[int] = None
    lora_alpha: Optional[float] = None
    target_modules: List[str] = Field(default_factory=list)
    has_config: bool
    has_tokenizer: bool
    has_chat_template: bool
    tensor_manifest: Dict[str, TensorMetadata]
    validation_errors: List[str] = Field(default_factory=list)
```

**Change from original draft:** `CheckpointManifest` and `TensorMetadata` are frozen/immutable where practical. The manifest is produced once by the scanner and then only *read* by downstream subsystems (remapper, dry-run, target profiles) — mutation-in-place across subsystems is a common source of "it worked when I ran `check` alone but not through `fix`" bugs. Any subsystem that needs a modified manifest constructs a new one (`manifest.model_copy(update={...})`).

### 4.2 Tensor & Weight Remapping Engine (`remapper.py`, `safetensors_io.py`)

Inspects and standardizes tensor keys using memory-mapped I/O, without loading weight buffers into memory or requiring PyTorch/CUDA.

```python
import math
from typing import Dict
from safetensors import safe_open
from adapterbridge.models.manifest import TensorMetadata


def extract_tensor_headers(file_path: str) -> Dict[str, TensorMetadata]:
    """Read tensor headers only (shape/dtype) via memory-mapped I/O.

    Deliberately avoids importing torch: n_elements is computed with
    math.prod so `check`/`fix` never require a CUDA-capable torch build
    to run on a laptop or a CPU-only CI runner.
    """
    manifest: Dict[str, TensorMetadata] = {}
    with safe_open(file_path, framework="numpy") as f:
        for key in f.keys():
            slice_obj = f.get_slice(key)
            shape = list(slice_obj.get_shape())
            manifest[key] = TensorMetadata(
                name=key,
                shape=shape,
                dtype=str(slice_obj.get_dtype()),
                n_elements=math.prod(shape) if shape else 1,
            )
    return manifest
```

**Change from original draft:** the original implementation used `framework="pt"` and imported `torch` purely to multiply a shape tuple together. That makes `torch` (and transitively CUDA wheels on many install setups) a hard dependency of a command that is supposed to run without a GPU. Using `framework="numpy"` and `math.prod` removes that dependency entirely; `torch` becomes an *optional* extra needed only for the dry-run harness's shape-inference helpers (§4.4), and even there it should default to CPU/meta device.

Key operations:
- **Prefix normalization** — strip redundant wrapper prefixes (e.g. `base_model.model.model.layers...` → `model.layers...`), driven by a small, testable table of known framework prefixes rather than string-matching heuristics scattered through the code.
- **Zero-copy re-serialization** — copy unmodified byte slices from the source `.safetensors` file directly into the output file with only the header remapped; never decode-then-re-encode tensor bytes.

### 4.3 Chat Template Linter & Sandbox (`template_linter.py`, `jinja_sandbox.py`)

Validates that a checkpoint's Jinja2 chat template renders multi-turn dialogs without syntax errors or unhandled exceptions — **before** it ever reaches a serving engine's `/v1/chat/completions` route.

- **Sandboxed execution:** render inside `jinja2.sandbox.SandboxedEnvironment`, with autoescape disabled (chat templates are not HTML) but attribute/global access restricted to the sandbox's default safe subset. Additionally, run the render call inside a **subprocess with a hard wall-clock timeout and no filesystem/network access**, since a downloaded checkpoint is untrusted input and `SandboxedEnvironment` alone protects against attribute-access tricks, not against a template designed to spin forever or exhaust memory.
- **Evaluation vectors:** a fixed, version-controlled suite of multi-turn user/assistant conversations, optional system prompts, tool-invocation schemas, and multi-modal message shapes — the same suite runs against every checkpoint so results are comparable across `check` runs.
- **Automated repair:** on failure, replace the broken template with a canonical template matching the resolved base-model architecture family (Llama-3-style, Qwen-2-style, etc.), pulled from a small bundled template library — never invented at runtime.

### 4.4 Mock Serving Dry-Run Harness (`dry_run.py`)

Simulates weight loading and tensor-sharding math without GPU hardware and, ideally, without materializing tensors at all.

- **Rank/scaling verification:**

  ```
  scale = lora_alpha / lora_r
  ```

  Confirms adapter matrix dimensions align with the frozen base layer:

  ```
  lora_A.shape[1] == base_layer.in_features
  lora_B.shape[0] == base_layer.out_features
  ```

- **Sharding validation:** verify that each weight matrix's relevant dimension divides evenly across the configured `--tensor-parallel-size`.
- **Implementation note:** where PyTorch shape-inference is genuinely useful (e.g. reproducing an engine's exact splitting logic), construct tensors on the **`meta` device** (`torch.empty(shape, device="meta")`), which allocates no real memory and works identically on a CPU-only machine. This preserves the "zero GPU, effectively zero memory" principle while still reusing the target engine's own sharding code where it's open source and importable.

---

## 5. End-to-End Execution Flow

```
1. Invocation
   CLI / SDK entry (e.g. `adapterbridge check --path ./checkpoint --target vllm`)
        │
        ▼
2. Metadata Ingestion
   Metadata Scanner parses adapter_config.json, resolves base model
   lineage, reads safetensors headers via zero-copy inspection
        │
        ▼
3. Specification Evaluation
   CheckpointManifest validated against the selected TargetProfile
        │
        ▼
4. (fix only) Automated Remediation
   - synthesize missing config.json / generation_config.json
   - remap tensor keys to target naming convention
   - inject a validated Jinja2 chat template
        │
        ▼
5. (verify only) Dry-Run Simulation
   Mock harness checks parameter alignment, rank math, TP sharding
        │
        ▼
6. Report Generation
   Rich table to stdout + structured JSON / SARIF for CI
```

Each stage consumes and produces `CheckpointManifest` / `ValidationReport` objects only — no stage reaches into another stage's internals. This is what makes `check`, `fix`, and `verify` independently callable rather than one hidden pipeline.

---

## 6. Target Specification Matrix

| Component / Requirement | vLLM | SGLang | Ollama | TensorRT-LLM |
|---|---|---|---|---|
| `config.json` | Required (`model_type`, `architectures`, `hidden_size`) | Required (base architecture mappings) | Converted into Modelfile parameters | Required (TRT conversion parameters) |
| `adapter_config.json` | Standard PEFT schema (`r`, `lora_alpha`, `target_modules`) | Standard PEFT schema | Converted to GGUF adapter format | Mapped to TRT LoRA layer definitions |
| Tensor key format | Direct layer indexing or standard prefix | Standardized PEFT key paths | GGUF-normalized key names | Engine-specific layer naming |
| Chat template | Jinja2, validated for OpenAI API routes | Jinja2, compliant with Radix format | Ollama `TEMPLATE` string format | Formatted prompt template string |
| Tokenizer artifacts | `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` | Full Hugging Face tokenizer bundle | Embedded GGUF tokenizer vocabulary | Hugging Face tokenizer bundle |

**Versioning note:** treat each row of this matrix as owned by its corresponding `targets/*.py` profile and version-pinned to a specific release of the upstream engine (see proposal doc §9, risk row 1). A CI job should periodically pull the latest tagged release of each engine and re-run the integration test corpus (§9) to catch upstream drift before users do.

---

## 7. Programmatic API and CLI Specification

### CLI usage

```bash
# 1. Validate a local adapter against vLLM requirements
adapterbridge check --path ./runs/checkpoint-500 --target vllm

# 2. Automatically repair missing metadata and re-map tensor keys
adapterbridge fix \
  --src ./runs/checkpoint-500 \
  --dst ./runs/checkpoint-500-ready \
  --target vllm \
  --base-model meta-llama/Llama-3.1-8B-Instruct

# 3. Execute a zero-GPU mock dry-run
adapterbridge verify --path ./runs/checkpoint-500-ready --target vllm --tensor-parallel-size 2

# 4. Export a diagnostic report in JSON for a CI/CD gate
adapterbridge check --path ./runs/checkpoint-500 --target sglang --format json --output report.json
```

**Framework choice:** the original draft mixed "Click and Rich" (proposal roadmap) with unstated CLI internals. Standardizing on **Typer** (type-hint-driven, built on Click, far less boilerplate for a 4-verb CLI like this) for command definitions, with **Rich** used purely for report rendering (tables, progress spinners during Hub resolution). This keeps the CLI layer thin — all real logic lives in `core/` and is reachable from the SDK without going through Typer at all.

### Python SDK

```python
from adapterbridge import AdapterInspector, TargetEngine

inspector = AdapterInspector(
    checkpoint_path="./runs/checkpoint-500",
    target_engine=TargetEngine.VLLM,
)

report = inspector.run_diagnostics()

if not report.is_compatible:
    print(f"Validation failed with {len(report.errors)} errors. Applying fixes...")
    remediation = inspector.auto_repair(
        destination_path="./runs/checkpoint-500-production",
        fallback_base_model="Qwen/Qwen2.5-7B-Instruct",
    )
    print(f"Repaired files saved to: {remediation.output_path}")

verification = inspector.verify_dry_run(tensor_parallel_size=1)
assert verification.success, f"Dry-run simulation failed: {verification.reason}"
```

---

## 8. Safety Guardrails & Target Profile Interface

### 8.1 Safety & Error Handling

- **Atomic staging:** file operations write to temporary staging folders (`.adapterbridge_staging_<uuid>`) and are atomically renamed into place only after every write succeeds; a failed `fix` run leaves no partial output at the destination path.
- **Sandbox isolation:** chat templates render inside `jinja2.sandbox.SandboxedEnvironment` **and** a timeout-bounded subprocess with no filesystem/network access — see §4.3. Never execute a template from an untrusted checkpoint directly in the main process.
- **Type validation:** `CheckpointManifest`, `TargetProfile` inputs, and all report schemas are Pydantic v2 models with `strict=True` validation on ingestion boundaries (CLI args, Hub API responses, on-disk JSON).
- **Network failure handling:** Hub lookups (`utils/hub.py`) must degrade gracefully — if `base_model_id` can't be resolved remotely (offline environment, private repo without auth), `check` reports a specific, actionable error rather than a stack trace, and `fix` accepts an explicit `--base-model` override (as shown in §7) to bypass resolution entirely.

### 8.2 Target Profile Interface

```python
from abc import ABC, abstractmethod
from adapterbridge.models.manifest import CheckpointManifest
from adapterbridge.models.report import ValidationReport, RemediationPlan


class BaseTargetProfile(ABC):
    """One implementation per serving engine. Registered via the
    `adapterbridge.targets` entry-point group so third parties can ship
    their own target profiles as separate installable packages without
    forking AdapterBridge core.
    """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Name of the target inference engine."""

    @property
    @abstractmethod
    def supported_engine_versions(self) -> str:
        """PEP 440 version specifier this profile is validated against,
        e.g. '>=0.6,<0.8'. Surfaced in diagnostic reports so users know
        when a profile is stale relative to their installed engine."""

    @abstractmethod
    def validate_manifest(self, manifest: CheckpointManifest) -> ValidationReport:
        """Inspect checkpoint manifest and return compatibility status."""

    @abstractmethod
    def generate_remediation_plan(self, manifest: CheckpointManifest) -> RemediationPlan:
        """Generate file synthesis and tensor remapping operations."""
```

**Changes from original draft:**
- `generate_remediation_plan` returns a typed `RemediationPlan` (Pydantic model) rather than a bare `dict` — a loosely-typed remediation payload is exactly the kind of thing that causes silent `fix` failures when a target profile's dict shape drifts from what `core/inspector.py` expects.
- Added `supported_engine_versions` so profile staleness is a first-class, reportable fact rather than something discovered only when a user's `fix` output fails to load.
- Profile discovery via Python entry-points (`targets/registry.py`) instead of a hardcoded import list, so Phase 4's "new target engine" work is additive rather than requiring core-package releases.

---

## 9. Testing Strategy (added)

The original spec had a `tests/` directory but no strategy — worth being explicit given the "false-negative erodes trust" risk called out in the proposal doc:

| Layer | What it covers | Runs where |
|---|---|---|
| **Unit tests** | Prefix-remapping table, Jinja2 sandbox timeout/isolation, Pydantic schema validation, dry-run shape math | Every PR, no external dependencies |
| **Fixture-based integration tests** | Small synthetic checkpoints (tiny random-weight LoRA adapters, a few KB each) covering each target profile's required/optional fields | Every PR, no GPU, no network |
| **Real-engine smoke tests** (`tests/integration/`) | Actually installing a pinned vLLM/Ollama version and loading an AdapterBridge-`fix`-ed checkpoint end-to-end | Nightly / pre-release CI job, separate from the fast PR loop, tracks the "false-negative rate" success metric directly |
| **Hub-resolution tests** | Lineage resolution against a small set of known public repos, with a cached/offline fallback path exercised explicitly | Every PR, with Hub calls recorded/replayed (e.g. via VCR-style cassettes) so tests don't flake on network issues |

---

## 10. Technology Stack Summary (added)

| Concern | Choice | Why |
|---|---|---|
| CLI | Typer + Rich | Type-hint-driven commands, minimal boilerplate for 4 verbs; Rich for tables/progress |
| Data validation | Pydantic v2 | Strict schemas at every I/O boundary; frozen models for the manifest |
| Tensor header inspection | `safetensors` (numpy backend) | No torch/CUDA dependency for `check`/`fix` |
| Optional shape inference | `torch` (meta device), install-time extra | Only pulled in for `verify` when deep sharding math is needed |
| Templating | Jinja2 `SandboxedEnvironment` + subprocess isolation | Defense in depth against untrusted checkpoint templates |
| Hub access | `huggingface_hub`, with local caching | Lineage resolution, graceful offline degradation |
| Plugin discovery | Python entry-points | Third-party target profiles without forking core |
| Report formats | JSON, Markdown, SARIF | SARIF specifically for GitHub code-scanning integration in Phase 4 |
