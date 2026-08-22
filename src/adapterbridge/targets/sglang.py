"""Target profile for SGLang inference engine."""

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


class SGLangTargetProfile(BaseTargetProfile):
    """Target engine specification for SGLang."""

    @property
    def engine_name(self) -> str:
        return TargetEngine.SGLANG.value

    @property
    def supported_engine_versions(self) -> str:
        return ">=0.3.0,<0.5.0"

    def validate_manifest(self, manifest: CheckpointManifest) -> ValidationReport:
        issues: List[ValidationIssue] = []

        if not manifest.has_config:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    code="SGLANG_ERR_MISSING_CONFIG",
                    message="Missing base 'config.json'. SGLang requires architectural metadata.",
                    path="config.json",
                )
            )

        bad_keys = [
            k for k in manifest.tensor_manifest.keys()
            if k.startswith("base_model.model.model.")
        ]
        if bad_keys:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    code="SGLANG_ERR_KEY_PREFIX",
                    message=f"Found {len(bad_keys)} tensor keys with unsupported prefix.",
                )
            )

        is_compatible = not any(i.severity == IssueSeverity.ERROR for i in issues)
        summary = (
            "Checkpoint is compatible with SGLang."
            if is_compatible
            else f"Validation failed with {len([i for i in issues if i.severity == IssueSeverity.ERROR])} error(s)."
        )

        return ValidationReport(
            checkpoint_path=manifest.checkpoint_path,
            target_engine=self.engine_name,
            is_compatible=is_compatible,
            issues=issues,
            summary=summary,
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
            summary=f"Remediation plan generated with {len(ops)} operation(s).",
        )
