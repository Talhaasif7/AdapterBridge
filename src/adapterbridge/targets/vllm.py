"""Target profile for vLLM inference engine."""

import os
from typing import List, Optional
from adapterbridge.core.canary import execute_canary_probe
from adapterbridge.core.chat_tester import execute_chat_roundtrip_test
from adapterbridge.core.moe_inspector import inspect_moe_architecture
from adapterbridge.core.schema_sync import SchemaSyncManager
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

    def __init__(self, target_spec_str: str = "vllm"):
        self.target_spec_str = target_spec_str
        self.sync_mgr = SchemaSyncManager()

    @property
    def engine_name(self) -> str:
        return TargetEngine.VLLM.value

    @property
    def supported_engine_versions(self) -> str:
        return ">=0.6.0,<0.8.0"

    def validate_manifest(self, manifest: CheckpointManifest) -> ValidationReport:
        issues: List[ValidationIssue] = []
        ruleset = self.sync_mgr.get_ruleset(self.target_spec_str)

        # 1. Base config check
        if not manifest.has_config and "config.json" in ruleset.required_files:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    code="VLLM_ERR_MISSING_CONFIG",
                    message=f"Missing base 'config.json'. vLLM ({ruleset.version}) requires config.json with model_type and hidden_size.",
                    path="config.json",
                    quick_fix=f"adapterbridge fix --src {manifest.checkpoint_path} --dst {manifest.checkpoint_path}-fixed --target {self.target_spec_str}",
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
                    quick_fix=f"adapterbridge fix --src {manifest.checkpoint_path} --dst {manifest.checkpoint_path}-fixed --target {self.target_spec_str}",
                )
            )

        # 3. Silent Dropping Failure Mode Check
        unsupported = ruleset.unsupported_target_modules
        dropped_modules = [m for m in manifest.target_modules if any(u in m for u in unsupported)]
        if dropped_modules:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    code="VLLM_ERR_SILENT_DROPPING",
                    message=f"Target modules {dropped_modules} are silently ignored by vLLM engine kernels.",
                    field="target_modules",
                    quick_fix=f"Remove or re-train adapter without {dropped_modules} or use base model merging.",
                )
            )

        # 4. MoE 2D vs 3D Stacked Weight Check
        moe_res = inspect_moe_architecture(manifest, requires_3d_stacked=ruleset.moe_requires_3d_stacked)
        if moe_res.incompatibilities:
            for inc in moe_res.incompatibilities:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        code="MOE_ERR_2D_3D_MISMATCH",
                        message=inc,
                        field="tensor_manifest",
                    )
                )

        # 5. Chat Template & Round-Trip Tokenization Check
        if not manifest.has_chat_template:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    code="VLLM_WARN_MISSING_CHAT_TEMPLATE",
                    message="Missing Jinja2 chat template in tokenizer_config.json. OpenAI /v1/chat/completions route will fail.",
                    path="tokenizer_config.json",
                    quick_fix=f"adapterbridge fix --src {manifest.checkpoint_path} --dst {manifest.checkpoint_path}-fixed --target {self.target_spec_str}",
                )
            )
        elif manifest.chat_template_str:
            chat_res = execute_chat_roundtrip_test(manifest.chat_template_str)
            if not chat_res.success:
                for c_issue in chat_res.issues:
                    issues.append(
                        ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            code="CHAT_WARN_TEMPLATE_DRIFT",
                            message=f"Chat template round-trip test warning: {c_issue}",
                            path="tokenizer_config.json",
                        )
                    )

        # 6. Tokenizer check
        if not manifest.has_tokenizer:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    code="VLLM_WARN_MISSING_TOKENIZER",
                    message="Missing tokenizer artifacts (tokenizer.json or tokenizer_config.json).",
                )
            )

        # Canary testing status
        canary_res = execute_canary_probe(manifest, unsupported_modules=unsupported)

        is_compatible = not any(i.severity == IssueSeverity.ERROR for i in issues)
        summary = (
            f"Checkpoint is compatible with vLLM ({ruleset.version})."
            if is_compatible
            else f"Validation failed with {len([i for i in issues if i.severity == IssueSeverity.ERROR])} error(s)."
        )

        return ValidationReport(
            checkpoint_path=manifest.checkpoint_path,
            target_engine=self.target_spec_str,
            is_compatible=is_compatible,
            issues=issues,
            summary=summary,
            canary_tested=True,
            canary_passed=canary_res.passed,
            canary_delta=canary_res.activation_delta_norm,
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
