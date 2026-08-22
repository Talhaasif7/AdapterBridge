"""Example 02: Creating and authoring a custom target profile plugin."""

import os
import sys
from typing import List

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapterbridge import (
    BaseTargetProfile,
    CheckpointManifest,
    IssueSeverity,
    RemediationPlan,
    ValidationIssue,
    ValidationReport,
)


class MyCustomEngineProfile(BaseTargetProfile):
    """Custom target profile implementation for proprietary internal runtimes."""

    @property
    def engine_name(self) -> str:
        return "my_custom_engine"

    @property
    def supported_engine_versions(self) -> str:
        return ">=1.0.0"

    def validate_manifest(self, manifest: CheckpointManifest) -> ValidationReport:
        issues: List[ValidationIssue] = []
        if not manifest.has_config:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    code="CUSTOM_ERR_MISSING_CONFIG",
                    message="Custom engine requires config.json.",
                )
            )
        is_compatible = len([i for i in issues if i.severity == IssueSeverity.ERROR]) == 0
        return ValidationReport(
            checkpoint_path=manifest.checkpoint_path,
            target_engine=self.engine_name,
            is_compatible=is_compatible,
            issues=issues,
            summary="Custom engine validation complete.",
        )

    def generate_remediation_plan(
        self, manifest: CheckpointManifest, destination_path: str
    ) -> RemediationPlan:
        return RemediationPlan(output_path=destination_path, operations=[])


def main():
    profile = MyCustomEngineProfile()
    print(f"Authoring custom target profile: '{profile.engine_name}' (Supported versions: {profile.supported_engine_versions})")


if __name__ == "__main__":
    main()
