"""Data models for diagnostic reports, remediation plans, and verification results."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationIssue(BaseModel):
    """Represents a single diagnostic rule check failure or warning."""
    model_config = ConfigDict(frozen=True)

    severity: IssueSeverity
    code: str
    message: str
    path: Optional[str] = None
    field: Optional[str] = None
    quick_fix: Optional[str] = None


class ValidationReport(BaseModel):
    """Structured report returned by static validation ('adapterbridge check')."""
    model_config = ConfigDict(frozen=True)

    checkpoint_path: str
    target_engine: str
    is_compatible: bool
    issues: List[ValidationIssue] = Field(default_factory=list)
    summary: str = ""
    canary_tested: bool = False
    canary_passed: Optional[bool] = None
    canary_delta: Optional[float] = None

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]

    def to_sarif(self) -> Dict[str, Any]:
        """Export report in full SARIF v2.1.0 format for GitHub Code Scanning integration."""
        rules_map: Dict[str, Dict[str, Any]] = {}
        sarif_results = []

        for issue in self.issues:
            if issue.code not in rules_map:
                rules_map[issue.code] = {
                    "id": issue.code,
                    "name": issue.code.replace("_", " ").title(),
                    "shortDescription": {"text": issue.message},
                    "fullDescription": {"text": issue.message},
                    "help": {"text": f"Remediate this issue using 'adapterbridge fix --src {self.checkpoint_path} --target {self.target_engine}'"},
                    "defaultConfiguration": {
                        "level": "error" if issue.severity == IssueSeverity.ERROR else "warning"
                    },
                }

            result_entry: Dict[str, Any] = {
                "ruleId": issue.code,
                "level": "error" if issue.severity == IssueSeverity.ERROR else "warning",
                "message": {"text": issue.message},
            }

            if issue.path:
                result_entry["locations"] = [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": issue.path}
                        }
                    }
                ]
            sarif_results.append(result_entry)

        return {
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "AdapterBridge",
                            "version": "0.2.0",

                            "informationUri": "https://github.com/adapterbridge/adapterbridge",
                            "rules": list(rules_map.values()),
                        }
                    },
                    "results": sarif_results,
                }
            ],
        }

    def to_pr_comment(self) -> str:
        """Format report into a styled GitHub/GitLab Pull Request comment."""
        status_badge = "**[PASSED]**" if self.is_compatible else "**[FAILED]**"
        
        lines = [
            f"## AdapterBridge Pre-Flight Check: {status_badge}",
            f"**Checkpoint Path:** `{self.checkpoint_path}` | **Target Engine:** `{self.target_engine.upper()}`",
            "",
            f"> {self.summary}",
            "",
        ]

        if self.issues:
            lines.extend([
                "### Diagnostic Results",
                "| Severity | Code | Message | Path |",
                "|---|---|---|---|",
            ])
            for issue in self.issues:
                icon = "[ERROR]" if issue.severity == IssueSeverity.ERROR else "[WARN]"
                path_str = f"`{issue.path}`" if issue.path else "-"
                lines.append(f"| {icon} **{issue.severity.value.upper()}** | `{issue.code}` | {issue.message} | {path_str} |")
            lines.append("")

        if not self.is_compatible:
            lines.extend([
                "<details>",
                "<summary>Suggested Automated Remediation Command</summary>",
                "",
                "```bash",
                f"adapterbridge fix --src {self.checkpoint_path} --dst {self.checkpoint_path}-repaired --target {self.target_engine}",
                "```",
                "</details>",
                "",
            ])

        lines.append("*Generated automatically by [AdapterBridge](https://github.com/adapterbridge/adapterbridge)*")
        return "\n".join(lines)



class RemediationAction(str, Enum):
    SYNTHESIZE_FILE = "synthesize_file"
    REMAP_TENSOR_KEY = "remap_tensor_key"
    INJECT_CHAT_TEMPLATE = "inject_chat_template"


class RemediationOperation(BaseModel):
    """Single file synthesis or key remapping operation in a remediation plan."""
    model_config = ConfigDict(frozen=True)

    action: RemediationAction
    target_path: str
    details: Dict[str, Any] = Field(default_factory=dict)


class RemediationPlan(BaseModel):
    """Executable plan produced by a target profile for 'adapterbridge fix'."""
    model_config = ConfigDict(frozen=True)

    output_path: str
    operations: List[RemediationOperation] = Field(default_factory=list)
    is_executable: bool = True
    summary: str = ""


class VerificationResult(BaseModel):
    """Result returned by zero-GPU mock serving dry-run engine ('adapterbridge verify')."""
    model_config = ConfigDict(frozen=True)

    success: bool
    reason: str
    tensor_parallel_size: int = 1
    memory_estimate_mb: float = 0.0
    sharding_issues: List[str] = Field(default_factory=list)
