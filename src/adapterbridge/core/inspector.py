"""Top-level SDK orchestration for checkpoint inspection, repair, and dry-run verification."""

from typing import Optional, Union
from adapterbridge.core.dry_run import run_dry_run_verification
from adapterbridge.core.lineage import scan_checkpoint
from adapterbridge.core.remapper import execute_remediation_plan
from adapterbridge.models.manifest import CheckpointManifest
from adapterbridge.models.report import RemediationPlan, ValidationReport, VerificationResult
from adapterbridge.models.target_spec import TargetEngine
from adapterbridge.targets.registry import get_target_profile


class AdapterInspector:
    """Primary SDK entry point for evaluating and remediating LoRA checkpoints."""

    def __init__(self, checkpoint_path: str, target_engine: Union[TargetEngine, str]):
        self.checkpoint_path = checkpoint_path
        self.target_engine_str = target_engine.value if isinstance(target_engine, TargetEngine) else str(target_engine)
        self.target_profile = get_target_profile(self.target_engine_str)
        self._manifest: Optional[CheckpointManifest] = None

    @property
    def manifest(self) -> CheckpointManifest:
        """Lazy-loaded CheckpointManifest."""
        if self._manifest is None:
            self._manifest = scan_checkpoint(self.checkpoint_path)
        return self._manifest

    def run_diagnostics(self) -> ValidationReport:
        """Perform static schema inspection and validation against the target profile."""
        return self.target_profile.validate_manifest(self.manifest)

    def auto_repair(
        self, destination_path: str, fallback_base_model: Optional[str] = None
    ) -> RemediationPlan:
        """Generate and execute a non-destructive remediation plan."""
        current_manifest = self.manifest
        if fallback_base_model and not current_manifest.base_model_id:
            current_manifest = current_manifest.model_copy(
                update={"base_model_id": fallback_base_model}
            )

        plan = self.target_profile.generate_remediation_plan(
            manifest=current_manifest, destination_path=destination_path
        )
        if plan.is_executable and plan.operations:
            execute_remediation_plan(current_manifest, plan)

        return plan

    def verify_dry_run(self, tensor_parallel_size: int = 1) -> VerificationResult:
        """Perform zero-GPU mock dry-run simulation."""
        return run_dry_run_verification(self.manifest, tensor_parallel_size=tensor_parallel_size)
