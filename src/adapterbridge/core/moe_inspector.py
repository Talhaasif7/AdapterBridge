"""Mixture-of-Experts (MoE) 2D per-expert vs 3D stacked expert tensor inspector."""

import re
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from adapterbridge.models.manifest import CheckpointManifest, TensorMetadata


class MoEInspectionResult(BaseModel):
    """Diagnostic details of MoE expert weight layout analysis."""

    is_moe: bool = False
    has_2d_per_expert: bool = False
    has_3d_stacked: bool = False
    expert_count: int = 0
    expert_tensor_keys: List[str] = Field(default_factory=list)
    incompatibilities: List[str] = Field(default_factory=list)


def inspect_moe_architecture(
    manifest: CheckpointManifest, requires_3d_stacked: bool = True
) -> MoEInspectionResult:
    """Analyze tensor manifest for MoE expert weight layout and verify 2D/3D compatibility."""
    expert_keys: List[str] = []
    expert_indices = set()
    has_2d = False
    has_3d = False

    per_expert_pattern = re.compile(r"experts\.(\d+)\.(gate|up|down|w1|w2|w3|proj)")
    stacked_3d_pattern = re.compile(r"experts\.(gate_up|w13|w2|down_up)_proj|\.experts_weight|\.is_3d")

    for key, meta in manifest.tensor_manifest.items():
        if "expert" in key.lower():
            expert_keys.append(key)
            m_2d = per_expert_pattern.search(key)
            if m_2d:
                has_2d = True
                expert_indices.add(int(m_2d.group(1)))

            if stacked_3d_pattern.search(key) or len(meta.shape) >= 3:
                has_3d = True

    is_moe = len(expert_keys) > 0 or has_2d or has_3d
    expert_count = len(expert_indices)

    incompatibilities: List[str] = []
    if is_moe:
        if requires_3d_stacked and has_2d and not has_3d:
            incompatibilities.append(
                f"MoE checkpoint contains 2D per-expert weights ({expert_count} experts detected) "
                f"but target engine requires fused 3D stacked expert tensors (is_3d_lora_weight: true)."
            )
        elif not requires_3d_stacked and has_3d and not has_2d:
            incompatibilities.append(
                "MoE checkpoint contains 3D stacked expert tensors but target engine expects 2D per-expert matrices."
            )

    return MoEInspectionResult(
        is_moe=is_moe,
        has_2d_per_expert=has_2d,
        has_3d_stacked=has_3d,
        expert_count=expert_count,
        expert_tensor_keys=expert_keys,
        incompatibilities=incompatibilities,
    )
