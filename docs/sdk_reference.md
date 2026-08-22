# AdapterBridge Python SDK Reference

The `adapterbridge` Python SDK provides programmatic access to inspection, auto-repair, and dry-run verification engines.

---

## SDK Usage Example

```python
from adapterbridge import AdapterInspector, TargetEngine

# 1. Initialize Inspector
inspector = AdapterInspector(
    checkpoint_path="./runs/checkpoint-500",
    target_engine=TargetEngine.VLLM,
)

# 2. Run Diagnostics
report = inspector.run_diagnostics()

if not report.is_compatible:
    print(f"Validation failed with {len(report.errors)} errors.")
    for err in report.errors:
        print(f"[{err.code}] {err.message}")

    # 3. Apply Auto-Repair
    plan = inspector.auto_repair(
        destination_path="./runs/checkpoint-500-ready",
        fallback_base_model="meta-llama/Llama-3.1-8B-Instruct",
    )
    print(f"Repaired files written to: {plan.output_path}")

# 4. Run Dry-Run Verification
result = inspector.verify_dry_run(tensor_parallel_size=2)
print(f"Dry-run status: {result.success} | Estimated RAM: {result.memory_estimate_mb} MB")
```

---

## Custom Target Profile Authoring

You can author third-party target profiles without forking core AdapterBridge by implementing `BaseTargetProfile` and registering via Python entry-points.

```python
from adapterbridge.models.manifest import CheckpointManifest
from adapterbridge.models.report import ValidationReport, RemediationPlan
from adapterbridge.models.target_spec import BaseTargetProfile


class CustomEngineProfile(BaseTargetProfile):
    @property
    def engine_name(self) -> str:
        return "custom_engine"

    @property
    def supported_engine_versions(self) -> str:
        return ">=1.0.0"

    def validate_manifest(self, manifest: CheckpointManifest) -> ValidationReport:
        # Custom validation logic
        ...

    def generate_remediation_plan(
        self, manifest: CheckpointManifest, destination_path: str
    ) -> RemediationPlan:
        # Custom remediation logic
        ...
```

### Entry Point Registration in `pyproject.toml`:

```toml
[project.entry-points."adapterbridge.targets"]
custom_engine = "my_package.targets:CustomEngineProfile"
```
