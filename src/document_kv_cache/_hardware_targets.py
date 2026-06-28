"""Shared V1 hardware target definitions."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "HardwareTargetProfile",
    "V1_HARDWARE_TARGET_PROFILE",
    "V1_AWS_G6_L4_HARDWARE_TARGET_PROFILE",
    "V1_AWS_G5_A10G_HARDWARE_TARGET_PROFILE",
    "V1_HARDWARE_TARGET_PROFILES",
    "SUPPORTED_V1_HARDWARE_TARGETS",
    "DEFAULT_HARDWARE_TARGET",
    "DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE",
    "SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES",
    "HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES",
    "hardware_target_profile",
    "default_databricks_node_type_for_hardware_target",
    "databricks_node_type_for_hardware_target",
    "validate_v1_hardware_target",
    "validate_aws_single_node_gpu_type",
    "validate_aws_single_node_gpu_type_for_hardware_target",
    "validate_v1_vllm_kv_cache_dtype_for_hardware_target",
]


@dataclass(frozen=True, slots=True)
class HardwareTargetProfile:
    """Benchmark target identity plus its managed Databricks node policy."""

    hardware_target: str
    display_name: str
    default_databricks_node_type_id: str
    databricks_node_type_prefixes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.hardware_target:
            raise ValueError("hardware_target must be non-empty")
        if not self.display_name:
            raise ValueError("display_name must be non-empty")
        if not self.default_databricks_node_type_id:
            raise ValueError("default_databricks_node_type_id must be non-empty")
        if not self.databricks_node_type_prefixes:
            raise ValueError("databricks_node_type_prefixes must be non-empty")
        if any(not prefix for prefix in self.databricks_node_type_prefixes):
            raise ValueError("databricks_node_type_prefixes entries must be non-empty")


V1_AWS_G6_L4_HARDWARE_TARGET_PROFILE = HardwareTargetProfile(
    hardware_target="aws-g6-l4",
    display_name="AWS g6/L4",
    default_databricks_node_type_id="g6.8xlarge",
    databricks_node_type_prefixes=("g6.",),
)
V1_AWS_G5_A10G_HARDWARE_TARGET_PROFILE = HardwareTargetProfile(
    hardware_target="aws-g5-a10g",
    display_name="AWS g5/A10G",
    default_databricks_node_type_id="g5.8xlarge",
    databricks_node_type_prefixes=("g5.",),
)
V1_HARDWARE_TARGET_PROFILE = V1_AWS_G6_L4_HARDWARE_TARGET_PROFILE
V1_HARDWARE_TARGET_PROFILES = (
    V1_AWS_G6_L4_HARDWARE_TARGET_PROFILE,
    V1_AWS_G5_A10G_HARDWARE_TARGET_PROFILE,
)
SUPPORTED_V1_HARDWARE_TARGETS = tuple(
    profile.hardware_target for profile in V1_HARDWARE_TARGET_PROFILES
)
DEFAULT_HARDWARE_TARGET = V1_HARDWARE_TARGET_PROFILE.hardware_target
DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE = V1_HARDWARE_TARGET_PROFILE.default_databricks_node_type_id
SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES = tuple(
    prefix
    for profile in V1_HARDWARE_TARGET_PROFILES
    for prefix in profile.databricks_node_type_prefixes
)
HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES = {
    profile.hardware_target: profile.databricks_node_type_prefixes
    for profile in V1_HARDWARE_TARGET_PROFILES
}
_AWS_G5_A10G_UNSUPPORTED_VLLM_FP8_KV_DTYPES = frozenset({"fp8", "fp8_e4m3"})
_HARDWARE_TARGET_PROFILE_BY_ID = {
    profile.hardware_target: profile
    for profile in V1_HARDWARE_TARGET_PROFILES
}


def validate_v1_hardware_target(hardware_target: str) -> None:
    if hardware_target not in SUPPORTED_V1_HARDWARE_TARGETS:
        raise ValueError(
            f"Unsupported V1 hardware target {hardware_target!r}; expected one of {SUPPORTED_V1_HARDWARE_TARGETS}"
        )


def validate_v1_vllm_kv_cache_dtype_for_hardware_target(
    *,
    hardware_target: str,
    kv_cache_dtype: str | None,
) -> None:
    validate_v1_hardware_target(hardware_target)
    if kv_cache_dtype is None:
        return
    normalized = kv_cache_dtype.strip().lower()
    if (
        hardware_target == V1_AWS_G5_A10G_HARDWARE_TARGET_PROFILE.hardware_target
        and normalized in _AWS_G5_A10G_UNSUPPORTED_VLLM_FP8_KV_DTYPES
    ):
        raise ValueError(
            "kv_cache_dtype='fp8'/'fp8_e4m3' maps to an FP8 E4M3 format that "
            "does not start on the AWS g5/A10G vLLM path; use 'fp8_e5m2' for "
            "Q8 document KV on this hardware target"
        )


def hardware_target_profile(hardware_target: str) -> HardwareTargetProfile:
    validate_v1_hardware_target(hardware_target)
    return _HARDWARE_TARGET_PROFILE_BY_ID[hardware_target]


def default_databricks_node_type_for_hardware_target(hardware_target: str) -> str:
    return hardware_target_profile(hardware_target).default_databricks_node_type_id


def databricks_node_type_for_hardware_target(
    hardware_target: str | None = None,
    node_type_id: str | None = None,
) -> str:
    if node_type_id is None:
        return default_databricks_node_type_for_hardware_target(hardware_target or DEFAULT_HARDWARE_TARGET)
    if hardware_target is None:
        validate_aws_single_node_gpu_type(node_type_id)
        return node_type_id
    validate_aws_single_node_gpu_type_for_hardware_target(node_type_id, hardware_target)
    return node_type_id


def validate_aws_single_node_gpu_type(node_type_id: str) -> None:
    if not node_type_id:
        raise ValueError("node_type_id must be non-empty")
    if not node_type_id.lower().startswith(SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES):
        supported = ", ".join(profile.display_name for profile in V1_HARDWARE_TARGET_PROFILES)
        raise ValueError(
            f"node_type_id must be a supported V1 Databricks node type ({supported}), "
            f"got {node_type_id!r}"
        )


def validate_aws_single_node_gpu_type_for_hardware_target(node_type_id: str, hardware_target: str) -> None:
    if not node_type_id:
        raise ValueError("node_type_id must be non-empty")
    profile = hardware_target_profile(hardware_target)
    if not node_type_id.lower().startswith(profile.databricks_node_type_prefixes):
        prefixes = ", ".join(profile.databricks_node_type_prefixes)
        raise ValueError(
            f"node_type_id must match V1 hardware target {hardware_target!r} "
            f"({profile.display_name}; Databricks prefixes: {prefixes}), got {node_type_id!r}"
        )
