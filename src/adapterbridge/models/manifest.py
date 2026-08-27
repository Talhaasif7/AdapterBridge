"""Data models for checkpoint manifests and tensor metadata."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class TensorMetadata(BaseModel):
    """Metadata describing a single tensor without materializing its byte array."""
    model_config = ConfigDict(frozen=True)

    name: str
    shape: List[int]
    dtype: str
    n_elements: int


class CheckpointManifest(BaseModel):
    """Canonical, read-only representation of a inspected checkpoint directory or repo."""
    model_config = ConfigDict(frozen=True)

    checkpoint_path: str
    is_adapter: bool = True
    base_model_id: Optional[str] = None
    adapter_type: Optional[str] = "lora"
    lora_r: Optional[int] = None
    lora_alpha: Optional[float] = None
    target_modules: List[str] = Field(default_factory=list)
    has_config: bool = False
    has_tokenizer: bool = False
    has_chat_template: bool = False
    chat_template_str: Optional[str] = None
    is_moe: bool = False
    expert_count: Optional[int] = None
    is_3d_stacked: bool = False
    canary_passed: Optional[bool] = None
    activation_delta: Optional[float] = None
    tensor_manifest: Dict[str, TensorMetadata] = Field(default_factory=dict)
    validation_errors: List[str] = Field(default_factory=list)
