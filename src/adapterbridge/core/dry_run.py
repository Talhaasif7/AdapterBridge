"""Zero-GPU mock serving dry-run engine supporting PyTorch meta-tensors and NumPy shape math."""

from typing import Dict, List, Optional
from adapterbridge.models.manifest import CheckpointManifest
from adapterbridge.models.report import VerificationResult

# Optional PyTorch import check
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


DTYPE_BYTES: Dict[str, float] = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,
}


def run_dry_run_verification(
    manifest: CheckpointManifest, tensor_parallel_size: int = 1
) -> VerificationResult:
    """Simulate target-runtime memory allocation, rank scaling, and tensor-parallel sharding math."""
    sharding_issues: List[str] = []
    total_bytes = 0
    used_meta_tensors = False

    if HAS_TORCH:
        try:
            used_meta_tensors = True
            for t_name, t_meta in manifest.tensor_manifest.items():
                # Materialize meta tensor without allocating GPU or host RAM
                meta_t = torch.empty(t_meta.shape, device="meta")
                element_bytes = meta_t.element_size() if meta_t.element_size() > 0 else 2
                total_bytes += int(t_meta.n_elements * element_bytes)

                # Validate Tensor Parallelism sharding division
                if tensor_parallel_size > 1 and meta_t.dim() >= 2:
                    major_dim = max(meta_t.shape)
                    if major_dim % tensor_parallel_size != 0:
                        sharding_issues.append(
                            f"Tensor '{t_name}' dimension {major_dim} cannot be split evenly across tensor-parallel-size={tensor_parallel_size}."
                        )
        except Exception:
            used_meta_tensors = False

    if not used_meta_tensors:
        # Fallback NumPy shape math
        for t_name, t_meta in manifest.tensor_manifest.items():
            element_bytes = DTYPE_BYTES.get(t_meta.dtype.lower(), 2)
            total_bytes += int(t_meta.n_elements * element_bytes)

            if tensor_parallel_size > 1 and len(t_meta.shape) >= 2:
                major_dim = max(t_meta.shape)
                if major_dim % tensor_parallel_size != 0:
                    sharding_issues.append(
                        f"Tensor '{t_name}' dimension {major_dim} cannot be split evenly across tensor-parallel-size={tensor_parallel_size}."
                    )

    # Validate LoRA rank dimension matching
    for t_name, t_meta in manifest.tensor_manifest.items():
        if ("lora_A" in t_name or "lora_B" in t_name) and manifest.lora_r:
            if manifest.lora_r not in t_meta.shape:
                sharding_issues.append(
                    f"Tensor '{t_name}' shape {t_meta.shape} does not contain configured rank r={manifest.lora_r}."
                )

    memory_estimate_mb = round(total_bytes / (1024 * 1024), 3)

    scaling_info = ""
    if manifest.lora_r and manifest.lora_alpha:
        scale = manifest.lora_alpha / manifest.lora_r
        scaling_info = f" Rank r={manifest.lora_r}, alpha={manifest.lora_alpha}, scale={scale:.2f}."

    mode_info = " [PyTorch meta-tensors]" if used_meta_tensors else " [NumPy shape engine]"
    success = len(sharding_issues) == 0
    reason = (
        f"Dry-run simulation passed successfully.{scaling_info}{mode_info}"
        if success
        else f"Dry-run failed with {len(sharding_issues)} sharding/shape issue(s).{mode_info}"
    )

    return VerificationResult(
        success=success,
        reason=reason,
        tensor_parallel_size=tensor_parallel_size,
        memory_estimate_mb=memory_estimate_mb,
        sharding_issues=sharding_issues,
    )
