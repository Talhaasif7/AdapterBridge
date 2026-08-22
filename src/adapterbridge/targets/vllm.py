"""Target profile for vLLM inference engine."""

import os
from typing import List
from adapterbridge.core.template_library import get_canonical_template_for_architecture
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
from adapterbridge.utils.hub import resolve_base_model_lineage


class VLLMTargetProfile(BaseTargetProfile):
    """Target engine specification and rules for vLLM."""

    @property
    def engine_name(self) -> str:
        return TargetEngine.VLLM.value

    @property
    def supported_engine_versions(self) -> str:
        return ">=0.6.0,<0.8.0"

    def validate_manifest(self, manifest: CheckpointManifest) -> ValidationReport:
        issues: List[ValidationIssue] = []

        # 1. Base config check
        if not manifest.has_config:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    code="VLLM_ERR_MISSING_CONFIG",
                    message="Missing base 'config.json'. vLLM requires config.json with model_type and hidden_size.",
                    path="config.json",
                )
            )

        # 2. Key prefix drift check
        bad_keys = [
            k for k in manifest.tensor_manifest.keys()
            if k.startswith("base_model.model.model.") or k.startswith("base_model.model.")
        ]
        if bad_keys:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    code="VLLM_ERR_KEY_PREFIX_DRIFT",
                    message=f"Found {len(bad_keys)} tensor keys with redundant prefix 'base_model.model.'. vLLM expects 'model.layers...'",
                    field="tensor_manifest",
                )
            )

        # 3. Chat template check
        if not manifest.has_chat_template:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    code="VLLM_WARN_MISSING_CHAT_TEMPLATE",
                    message="Missing Jinja2 chat template in tokenizer_config.json. OpenAI /v1/chat/completions route will fail.",
                    path="tokenizer_config.json",
                )
            )

        # 4. Tokenizer check
        if not manifest.has_tokenizer:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    code="VLLM_WARN_MISSING_TOKENIZER",
                    message="Missing tokenizer artifacts (tokenizer.json or tokenizer_config.json).",
                )
            )

        is_compatible = not any(i.severity == IssueSeverity.ERROR for i in issues)
        summary = (
            "Checkpoint is compatible with vLLM."
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

        base_model_id = manifest.base_model_id or "meta-llama/Llama-3.1-8B-Instruct"
        remote_lineage = resolve_base_model_lineage(base_model_id) if base_model_id else {}
        base_config_data = remote_lineage.get("config") or {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "model_type": "llama",
        }

        model_type = base_config_data.get("model_type", "llama")

        # File synthesis for missing config.json
        if not manifest.has_config:
            ops.append(
                RemediationOperation(
                    action=RemediationAction.SYNTHESIZE_FILE,
                    target_path=os.path.join(destination_path, "config.json"),
                    details=base_config_data,
                )
            )

        # Tensor key remapping operations
        key_map = {}
        for k in manifest.tensor_manifest.keys():
            if k.startswith("base_model.model.model."):
                key_map[k] = k.replace("base_model.model.model.", "model.", 1)
            elif k.startswith("base_model.model."):
                key_map[k] = k.replace("base_model.model.", "model.", 1)

        if key_map:
            ops.append(
                RemediationOperation(
                    action=RemediationAction.REMAP_TENSOR_KEY,
                    target_path=destination_path,
                    details={"key_map": key_map},
                )
            )

        # Chat template injection using architecture-appropriate canonical template
        if not manifest.has_chat_template:
            canonical_template = get_canonical_template_for_architecture(model_type)
            ops.append(
                RemediationOperation(
                    action=RemediationAction.INJECT_CHAT_TEMPLATE,
                    target_path=os.path.join(destination_path, "tokenizer_config.json"),
                    details={"chat_template": canonical_template},
                )
            )

        return RemediationPlan(
            output_path=destination_path,
            operations=ops,
            is_executable=True,
            summary=f"Remediation plan contains {len(ops)} operation(s).",
        )
