"""Target profile registry for resolving engine specifications."""

import sys
from typing import Dict, Type
from adapterbridge.models.target_spec import BaseTargetProfile, TargetEngine
from adapterbridge.targets.vllm import VLLMTargetProfile
from adapterbridge.targets.sglang import SGLangTargetProfile
from adapterbridge.targets.ollama import OllamaTargetProfile
from adapterbridge.targets.tensorrt import TensorRTTargetProfile


BUILTIN_PROFILES: Dict[str, Type[BaseTargetProfile]] = {
    TargetEngine.VLLM.value: VLLMTargetProfile,
    TargetEngine.SGLANG.value: SGLangTargetProfile,
    TargetEngine.OLLAMA.value: OllamaTargetProfile,
    TargetEngine.TENSORRT.value: TensorRTTargetProfile,
}


def get_target_profile(target: str) -> BaseTargetProfile:
    """Resolve a target profile instance by engine name (e.g. 'vllm' or 'vllm@0.6.4').
    
    Raises ValueError if engine is unknown.
    """
    raw_str = str(target).strip()
    engine_name = raw_str.split("@", 1)[0].lower() if "@" in raw_str else raw_str.lower()
    
    if engine_name in BUILTIN_PROFILES:
        profile = BUILTIN_PROFILES[engine_name]()
        if hasattr(profile, "target_spec_str"):
            setattr(profile, "target_spec_str", raw_str)
        return profile

    # Entry point discovery fallback for 3rd party target plugins
    if sys.version_info >= (3, 10):
        from importlib.metadata import entry_points
        eps = entry_points(group="adapterbridge.targets")
    else:
        from importlib_metadata import entry_points
        eps = entry_points().get("adapterbridge.targets", [])

    for ep in eps:
        if ep.name == engine_name:
            profile_cls = ep.load()
            profile = profile_cls()
            if hasattr(profile, "target_spec_str"):
                setattr(profile, "target_spec_str", raw_str)
            return profile

    supported = list(BUILTIN_PROFILES.keys())
    raise ValueError(f"Unknown target engine '{target}'. Supported built-in engines: {supported}")
