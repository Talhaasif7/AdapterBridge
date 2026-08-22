"""Target profile for Ollama runtime engine."""

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


class OllamaTargetProfile(BaseTargetProfile):
    """Target profile for Ollama."""

    @property
    def engine_name(self) -> str:
        return TargetEngine.OLLAMA.value

    @property
    def supported_engine_versions(self) -> str:
        return ">=0.3.0"

    def validate_manifest(self, manifest: CheckpointManifest) -> ValidationReport:
        issues: List[ValidationIssue] = []

        if not manifest.has_config:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    code="OLLAMA_WARN_MISSING_CONFIG",
                    message="Missing base config.json; required to generate Modelfile parameters.",
                )
            )

        is_compatible = True
        return ValidationReport(
            checkpoint_path=manifest.checkpoint_path,
            target_engine=self.engine_name,
            is_compatible=is_compatible,
            issues=issues,
            summary="Checkpoint validated for Ollama export.",
        )

    def generate_remediation_plan(
        self, manifest: CheckpointManifest, destination_path: str
    ) -> RemediationPlan:
        ops = [
            RemediationOperation(
                action=RemediationAction.SYNTHESIZE_FILE,
                target_path=os.path.join(destination_path, "Modelfile"),
                details={
                    "base_model": manifest.base_model_id or "llama3.1",
                    "adapter_path": "./adapter_model.safetensors",
                },
            )
        ]
        return RemediationPlan(
            output_path=destination_path,
            operations=ops,
            is_executable=True,
            summary="Generated Ollama Modelfile synthesis plan.",
        )
