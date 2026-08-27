"""Unit tests for MoE 2D vs 3D stacked expert layout inspector."""

import pytest
from adapterbridge.core.moe_inspector import inspect_moe_architecture
from adapterbridge.models.manifest import CheckpointManifest, TensorMetadata


def test_moe_2d_per_expert_incompatibility(tmp_path):
    t_manifest = {
        "model.layers.0.block_sparse_moe.experts.0.gate_proj.lora_A.weight": TensorMetadata(
            name="model.layers.0.block_sparse_moe.experts.0.gate_proj.lora_A.weight",
            shape=[8, 4096],
            dtype="float16",
            n_elements=32768,
        ),
        "model.layers.0.block_sparse_moe.experts.1.gate_proj.lora_A.weight": TensorMetadata(
            name="model.layers.0.block_sparse_moe.experts.1.gate_proj.lora_A.weight",
            shape=[8, 4096],
            dtype="float16",
            n_elements=32768,
        ),
    }

    manifest = CheckpointManifest(
        checkpoint_path=str(tmp_path),
        tensor_manifest=t_manifest,
    )

    res = inspect_moe_architecture(manifest, requires_3d_stacked=True)
    assert res.is_moe is True
    assert res.has_2d_per_expert is True
    assert res.expert_count == 2
    assert len(res.incompatibilities) == 1
    assert "2D per-expert weights" in res.incompatibilities[0]


def test_moe_3d_stacked_compatibility(tmp_path):
    t_manifest = {
        "model.layers.0.block_sparse_moe.experts.gate_up_proj.lora_A.weight": TensorMetadata(
            name="model.layers.0.block_sparse_moe.experts.gate_up_proj.lora_A.weight",
            shape=[8, 16, 4096],
            dtype="float16",
            n_elements=524288,
        )
    }

    manifest = CheckpointManifest(
        checkpoint_path=str(tmp_path),
        tensor_manifest=t_manifest,
    )

    res = inspect_moe_architecture(manifest, requires_3d_stacked=True)
    assert res.is_moe is True
    assert res.has_3d_stacked is True
    assert len(res.incompatibilities) == 0
