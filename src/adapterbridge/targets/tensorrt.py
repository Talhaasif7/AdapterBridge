"""Target profile for TensorRT-LLM runtime engine."""

import os
from typing import List
from adapterbridge.models.manifest import CheckpointManifest
from adapterbridge.models.report import (
    IssueSeverity,
    RemediationAction,
    RemediationOperation,
    RemediationPlan,
    ValidationIssue,
    ValidationReport,
)
from adapterbridge.models.target_spec import BaseTargetProfile, TargetEngine


class TensorRTTargetProfile(BaseTargetProfile):
    """Target profile for TensorRT-LLM."""

    @property
    def engine_name(self) -> str:
        return TargetEngine.TENSORRT.value

    @property
    def supported_engine_versions(self) -> str:
        return ">=0.9.0"

    def validate_manifest(self, manifest: CheckpointManifest) -> ValidationReport:
        issues: List[ValidationIssue] = []

        if not manifest.has_config:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    code="TRT_ERR_MISSING_CONFIG",
                    message="TensorRT-LLM conversion requires config.json with full layer parameters.",
                )
            )

        is_compatible = not any(i.severity == IssueSeverity.ERROR for i in issues)
        return ValidationReport(
            checkpoint_path=manifest.checkpoint_path,
            target_engine=self.engine_name,
            is_compatible=is_compatible,
            issues=issues,
            summary="Validated for TensorRT-LLM conversion.",
        )

    def generate_remediation_plan(
        self, manifest: CheckpointManifest, destination_path: str
    ) -> RemediationPlan:
        ops: List[RemediationOperation] = []

        if not manifest.has_config:
            ops.append(
                RemediationOperation(
                    action=RemediationAction.SYNTHESIZE_FILE,
                    target_path=os.path.join(destination_path, "config.json"),
                    details={"base_model_id": manifest.base_model_id or "meta-llama/Llama-3.1-8B-Instruct"},
                )
            )

        return RemediationPlan(
            output_path=destination_path,
            operations=ops,
            is_executable=True,
            summary="Generated TensorRT-LLM remediation plan.",
        )
