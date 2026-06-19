"""Compatibility wrapper for :mod:`document_kv_cache.databricks_job`."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from types import FunctionType
from typing import Any

from document_kv_cache._reexport import reexport_public

import document_kv_cache.databricks_job as _document_module

__all__ = reexport_public(
    "document_kv_cache.databricks_job",
    (
        "DEFAULT_AWS_G5_NODE_TYPE",
        "DEFAULT_DATABRICKS_SPARK_VERSION",
        "DEFAULT_DATABRICKS_RUN_NAME",
        "DEFAULT_DATABRICKS_TASK_KEY",
        "DEFAULT_DATABRICKS_DATA_SECURITY_MODE",
        "DEDICATED_DATABRICKS_DATA_SECURITY_MODE",
        "SINGLE_USER_DATABRICKS_DATA_SECURITY_MODES",
    ),
    globals(),
)

__all__ += [
    "RESERVED_SINGLE_NODE_G5_TAG_KEYS",
    "RUNNER_SCRIPT",
    "DatabricksSingleNodeG5ClusterConfig",
    "DatabricksBenchmarkJobConfig",
    "validate_aws_g5_node_type",
    "build_single_node_g5_cluster",
    "build_databricks_run_submit_payload",
    "write_databricks_run_submit_json",
    "write_databricks_runner_script",
    "main",
    "argparse",
    "json",
    "Mapping",
    "Sequence",
    "dataclass",
    "field",
    "Path",
    "Any",
]

RESERVED_SINGLE_NODE_G5_TAG_KEYS = _document_module.RESERVED_SINGLE_NODE_G5_TAG_KEYS
RUNNER_SCRIPT = _document_module.RUNNER_SCRIPT


class DatabricksSingleNodeG5ClusterConfig(_document_module.DatabricksSingleNodeG5ClusterConfig):
    __slots__ = ()

    def __post_init__(self) -> None:
        if not self.purpose:
            raise ValueError("purpose must be non-empty")
        validate_aws_g5_node_type(self.node_type_id)
        if not self.spark_version:
            raise ValueError("spark_version must be non-empty")
        if not self.data_security_mode:
            raise ValueError("data_security_mode must be non-empty")
        if self.single_user_name is not None and not self.single_user_name:
            raise ValueError("single_user_name must be non-empty when provided")
        if _is_single_user_mode(self.data_security_mode) and self.single_user_name is None:
            raise ValueError("single_user_name is required when data_security_mode is SINGLE_USER")
        if not self.availability:
            raise ValueError("availability must be non-empty")
        if not self.zone_id:
            raise ValueError("zone_id must be non-empty")
        reserved_tags = RESERVED_SINGLE_NODE_G5_TAG_KEYS.intersection(self.custom_tags)
        if reserved_tags:
            raise ValueError(f"custom_tags cannot override reserved tags: {sorted(reserved_tags)!r}")


class DatabricksBenchmarkJobConfig(_document_module.DatabricksBenchmarkJobConfig):
    __slots__ = ()

    def __post_init__(self) -> None:
        if not self.plan_json_uri:
            raise ValueError("plan_json_uri must be non-empty")
        if not self.runner_python_file:
            raise ValueError("runner_python_file must be non-empty")
        if not self.run_name:
            raise ValueError("run_name must be non-empty")
        if not self.task_key:
            raise ValueError("task_key must be non-empty")
        _cluster_config_from_benchmark_job(self)
        if self.wheel_uri is not None and not self.wheel_uri:
            raise ValueError("wheel_uri must be non-empty when provided")
        if self.execution_result_json_uri is not None and not self.execution_result_json_uri:
            raise ValueError("execution_result_json_uri must be non-empty when provided")


def validate_aws_g5_node_type(node_type_id: str) -> None:
    return _call_document_function("validate_aws_g5_node_type", node_type_id)


def build_databricks_run_submit_payload(config: DatabricksBenchmarkJobConfig) -> dict[str, Any]:
    return _call_document_function("build_databricks_run_submit_payload", config)


def write_databricks_run_submit_json(config: DatabricksBenchmarkJobConfig, path: str | Path) -> None:
    return _call_document_function("write_databricks_run_submit_json", config, path)


def write_databricks_runner_script(path: str | Path) -> None:
    return _call_document_function("write_databricks_runner_script", path)


def build_single_node_g5_cluster(config: DatabricksSingleNodeG5ClusterConfig) -> dict[str, Any]:
    return _call_document_function("build_single_node_g5_cluster", config)


def _single_node_g5_cluster(config: DatabricksBenchmarkJobConfig) -> dict[str, Any]:
    return _call_document_function("_single_node_g5_cluster", config)


def _cluster_config_from_benchmark_job(config: DatabricksBenchmarkJobConfig) -> DatabricksSingleNodeG5ClusterConfig:
    return _call_document_function("_cluster_config_from_benchmark_job", config)


def _is_single_user_mode(data_security_mode: str) -> bool:
    return _call_document_function("_is_single_user_mode", data_security_mode)


def _runner_parameters(config: DatabricksBenchmarkJobConfig) -> list[str]:
    return _call_document_function("_runner_parameters", config)


def main(argv: Sequence[str] | None = None) -> int:
    return _call_document_function("main", argv)


_DEFAULT_COMPAT_FUNCTIONS = {
    "validate_aws_g5_node_type": validate_aws_g5_node_type,
    "build_databricks_run_submit_payload": build_databricks_run_submit_payload,
    "write_databricks_run_submit_json": write_databricks_run_submit_json,
    "write_databricks_runner_script": write_databricks_runner_script,
    "build_single_node_g5_cluster": build_single_node_g5_cluster,
    "_single_node_g5_cluster": _single_node_g5_cluster,
    "_cluster_config_from_benchmark_job": _cluster_config_from_benchmark_job,
    "_is_single_user_mode": _is_single_user_mode,
    "_runner_parameters": _runner_parameters,
    "main": main,
}
_DOCUMENT_DEFAULTS = {
    name: value
    for name, value in vars(_document_module).items()
    if not name.startswith("__")
}
_PATCH_LOCK = RLock()
_LEGACY_PATCH_NAMES = tuple(name for name in _DOCUMENT_DEFAULTS if name in globals())


def _call_document_function(name: str, *args, **kwargs):
    with _PATCH_LOCK:
        return _isolated_document_namespace()[name](*args, **kwargs)


def _document_global_for_legacy(name: str):
    if name not in globals():
        return _DOCUMENT_DEFAULTS[name]
    current = globals()[name]
    if _DEFAULT_COMPAT_FUNCTIONS.get(name) is current:
        return _DOCUMENT_DEFAULTS[name]
    return current


def _isolated_document_namespace() -> dict[str, Any]:
    namespace = dict(_DOCUMENT_DEFAULTS)
    for name in _LEGACY_PATCH_NAMES:
        if name in namespace:
            namespace[name] = _document_global_for_legacy(name)
    for name, value in tuple(namespace.items()):
        if _is_document_function(value):
            namespace[name] = _clone_document_function(value, namespace)
    return namespace


def _is_document_function(value: Any) -> bool:
    return isinstance(value, FunctionType) and value.__globals__ is vars(_document_module)


def _clone_document_function(function: FunctionType, namespace: dict[str, Any]) -> FunctionType:
    clone = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = dict(function.__annotations__)
    clone.__doc__ = function.__doc__
    clone.__module__ = function.__module__
    return clone


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


del reexport_public
