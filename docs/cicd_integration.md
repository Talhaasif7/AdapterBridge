# CI/CD & MLOps Integration Guide

Integrate **AdapterBridge** into CI/CD pipelines, pre-commit hooks, and Kubernetes model registries.

---

## 1. GitHub Actions Integration

Use our composite GitHub Action in `.github/workflows/adapterbridge.yml`:

```yaml
name: Checkpoint Pre-Flight Check

on:
  pull_request:
    paths:
      - 'adapters/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run AdapterBridge Check
        uses: ./.github/actions/adapterbridge-check
        with:
          path: './adapters/my-lora'
          target: 'vllm'
          format: 'sarif'
          output: 'adapterbridge-results.sarif'

      - name: Upload SARIF Report
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'adapterbridge-results.sarif'
```

---

## 2. Git Pre-Commit Hook Integration

Add to your repository's `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/adapterbridge/adapterbridge
    rev: v0.1.0
    hooks:
      - id: adapterbridge-check
        args: ["--path", "./adapters/my-lora", "--target", "vllm"]
```

---

## 3. Kubernetes / Containerized Admission Gate

Run AdapterBridge inside your Kubernetes cluster or MLOps pipeline using our lightweight Docker container:

```bash
docker run --rm -v $(pwd)/checkpoint:/workspace/checkpoint \
  adapterbridge:latest check --path /workspace/checkpoint --target vllm
```
