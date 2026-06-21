"""Shared V1 hardware target definitions."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "HardwareTargetProfile",
    "V1_HARDWARE_TARGET_PROFILE",
    "SUPPORTED_V1_HARDWARE_TARGETS",
    "DEFAULT_HARDWARE_TARGET",
    "DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE",
    "SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES",
    "HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES",
    "validate_v1_hardware_target",
    "validate_aws_single_node_gpu_type",
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


V1_HARDWARE_TARGET_PROFILE = HardwareTargetProfile(
    hardware_target="aws-g6-l4",
    display_name="AWS g6/L4",
    default_databricks_node_type_id="g6.8xlarge",
    databricks_node_type_prefixes=("g6.",),
)
SUPPORTED_V1_HARDWARE_TARGETS = (V1_HARDWARE_TARGET_PROFILE.hardware_target,)
DEFAULT_HARDWARE_TARGET = V1_HARDWARE_TARGET_PROFILE.hardware_target
DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE = V1_HARDWARE_TARGET_PROFILE.default_databricks_node_type_id
SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES = V1_HARDWARE_TARGET_PROFILE.databricks_node_type_prefixes
HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES = {
    V1_HARDWARE_TARGET_PROFILE.hardware_target: V1_HARDWARE_TARGET_PROFILE.databricks_node_type_prefixes,
}


def validate_v1_hardware_target(hardware_target: str) -> None:
    if hardware_target not in SUPPORTED_V1_HARDWARE_TARGETS:
        raise ValueError(
            f"Unsupported V1 hardware target {hardware_target!r}; expected one of {SUPPORTED_V1_HARDWARE_TARGETS}"
        )


def validate_aws_single_node_gpu_type(node_type_id: str) -> None:
    if not node_type_id:
        raise ValueError("node_type_id must be non-empty")
    if not node_type_id.lower().startswith(SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES):
        raise ValueError(
            f"node_type_id must be an {V1_HARDWARE_TARGET_PROFILE.display_name} Databricks node type, "
            f"got {node_type_id!r}"
        )
