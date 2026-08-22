# AdapterBridge Documentation

**LoRA Checkpoint & Config Compatibility Engine for Enterprise Inference Runtimes**

AdapterBridge is an open-source compatibility, validation, and remediation engine that eliminates runtime deployment failures when moving fine-tuned Low-Rank Adaptation (LoRA/QLoRA) checkpoints into production inference runtimes (vLLM, SGLang, Ollama, TensorRT-LLM).

---

## Key Principles

1. **Zero memory bloat, lazy evaluation**: Never load full model weights into host RAM or GPU memory during inspection. Memory-mapped I/O reads headers and tensor keys without materializing weight buffers.
2. **Stateless and deterministic execution**: Identical inputs + identical target profile version → identical diagnostic output.
3. **Target-engine decoupling**: Inspection logic lives behind pluggable `BaseTargetProfile` interfaces.
4. **Non-destructive by default**: All repair operations write to a new destination directory using atomic staging.
5. **Zero GPU required, ever, for check/fix/verify**: Pre-flight verification runs cleanly on laptops, pre-commit hooks, and CPU-only CI runners.

---

## System Architecture

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
│  │ Lineage Engine │ │ Remapper      │ │                     │ │
│  └───────────────┘ └───────────────┘ └─────────────────────┘ │
│  ┌───────────────────────────────────────────────────────┐   │
│  │            Mock Serving Dry-Run Harness               │   │
│  └───────────────────────────────────────────────────────┘   │
└───────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  Target Specification Layer                  │
│   vLLM  │  SGLang  │  Ollama  │  TensorRT-LLM  │  (pluggable) │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Navigation

- [CLI Reference](cli_reference.md) — Command line manual for `check`, `fix`, `verify`, `export`.
- [SDK Reference](sdk_reference.md) — Python SDK & custom target profile authoring.
- [Target Engines](target_engines.md) — Specifications for vLLM, SGLang, Ollama, and TensorRT-LLM.
- [CI/CD & MLOps Integration](cicd_integration.md) — GitHub Actions, pre-commit hooks, and Docker admission gates.
