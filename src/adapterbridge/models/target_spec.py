"""Target engine specifications and abstract target profile interface."""

from abc import ABC, abstractmethod
from enum import Enum
from adapterbridge.models.manifest import CheckpointManifest
from adapterbridge.models.report import ValidationReport, RemediationPlan


class TargetEngine(str, Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    OLLAMA = "ollama"
    TENSORRT = "tensorrt"


class BaseTargetProfile(ABC):
    """Abstract base class for inference serving engine profiles.
    
    Registered via entry-points so third parties can implement custom target engines.
    """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Name of the target inference engine (e.g. 'vllm')."""

    @property
    @abstractmethod
    def supported_engine_versions(self) -> str:
        """PEP 440 version specifier this profile is validated against (e.g. '>=0.6,<0.8')."""

    @abstractmethod
    def validate_manifest(self, manifest: CheckpointManifest) -> ValidationReport:
        """Inspect checkpoint manifest and return compatibility status."""

    @abstractmethod
    def generate_remediation_plan(
        self, manifest: CheckpointManifest, destination_path: str
    ) -> RemediationPlan:
        """Generate file synthesis and tensor remapping operations."""
