"""Compatibility wrapper for :mod:`document_kv_cache.databricks_vllm_smoke_job`."""

from __future__ import annotations

import argparse
import importlib.util as _importlib_util
import json
import sys as _sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from types import FunctionType
from typing import Any

from document_kv_cache._reexport import reexport_public
from document_kv_cache.databricks_job import (
    DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE,
    DEFAULT_AWS_G5_NODE_TYPE,
    DEFAULT_DATABRICKS_DATA_SECURITY_MODE,
    DEFAULT_DATABRICKS_SPARK_VERSION,
    DatabricksSingleNodeGPUClusterConfig,
    DatabricksSingleNodeG5ClusterConfig,
    build_single_node_gpu_cluster,
    build_single_node_g5_cluster,
)
from document_kv_cache.vllm_smoke import (
    DEFAULT_LOCAL_ROOT,
    SERVER_HOST,
    SERVER_PORT,
)

import document_kv_cache.databricks_vllm_smoke_job as _document_module

__all__ = reexport_public(
    "document_kv_cache.databricks_vllm_smoke_job",
    (
        "DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME",
        "DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY",
        "DEFAULT_DATABRICKS_VLLM_SMOKE_PURPOSE",
        "VLLM_SMOKE_RUNNER_SCRIPT",
        "DatabricksVLLMSmokeJobConfig",
    ),
    globals(),
)

__all__ += [
    "build_databricks_vllm_smoke_run_submit_payload",
    "write_databricks_vllm_smoke_run_submit_json",
    "write_databricks_vllm_smoke_runner_script",
    "main",
    "argparse",
    "json",
    "Mapping",
    "Sequence",
    "dataclass",
    "field",
    "Path",
    "Any",
    "DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE",
    "DEFAULT_AWS_G5_NODE_TYPE",
    "DEFAULT_DATABRICKS_DATA_SECURITY_MODE",
    "DEFAULT_DATABRICKS_SPARK_VERSION",
    "DatabricksSingleNodeGPUClusterConfig",
    "DatabricksSingleNodeG5ClusterConfig",
    "build_single_node_gpu_cluster",
    "build_single_node_g5_cluster",
    "DEFAULT_LOCAL_ROOT",
    "SERVER_HOST",
    "SERVER_PORT",
]


def _load_document_defaults_module():
    module_path = Path(_document_module.__file__)
    module_name = "_restaurant_kv_serving_databricks_vllm_smoke_job_document_defaults"
    spec = _importlib_util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load document databricks_vllm_smoke_job defaults from {module_path}")
    module = _importlib_util.module_from_spec(spec)
    _sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_document_defaults_module = _load_document_defaults_module()
_DOCUMENT_DEFAULTS = {
    name: value
    for name, value in vars(_document_defaults_module).items()
    if not name.startswith("__")
}
_PUBLIC_CONSTANT_NAMES = frozenset(
    {
        "DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME",
        "DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY",
        "DEFAULT_DATABRICKS_VLLM_SMOKE_PURPOSE",
        "VLLM_SMOKE_RUNNER_SCRIPT",
    }
)
_PUBLIC_CONFIG_CLASS_DEPENDENCY_NAMES = frozenset(
    {
        "_DEFAULT_CLUSTER_CONFIG_FROM_VLLM_SMOKE_JOB",
    }
)


def _is_pristine_public_class(name: str) -> bool:
    live_value = getattr(_document_module, name)
    default_value = _DOCUMENT_DEFAULTS[name]
    return (
        isinstance(live_value, type)
        and live_value.__module__ == _document_module.__name__
        and live_value.__qualname__ == default_value.__qualname__
        and _class_fingerprint(live_value) == _class_fingerprint(default_value)
    )


def _public_config_class_is_reusable() -> bool:
    return _is_pristine_public_class("DatabricksVLLMSmokeJobConfig") and all(
        _global_fingerprint(getattr(_document_module, name)) == _global_fingerprint(_DOCUMENT_DEFAULTS[name])
        for name in _PUBLIC_CONFIG_CLASS_DEPENDENCY_NAMES
    )


def _class_fingerprint(value: type) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (name, _class_attribute_fingerprint(name, attribute))
        for name, attribute in sorted(vars(value).items())
        if name not in {"__doc__", "__module__"}
    )


def _class_attribute_fingerprint(name: str, value: Any) -> Any:
    if name == "__dataclass_fields__":
        return _dataclass_field_fingerprint(value)
    if name == "__dataclass_params__":
        return repr(value)
    if hasattr(value, "__objclass__") and hasattr(value, "__name__"):
        return ("descriptor", type(value).__qualname__, value.__name__)
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    function_fingerprint = _function_fingerprint(value)
    if function_fingerprint is not None:
        return ("function", function_fingerprint)
    return value


def _global_fingerprint(value: Any) -> Any:
    function_fingerprint = _function_fingerprint(value)
    if function_fingerprint is not None:
        return ("function", function_fingerprint)
    return value


def _function_fingerprint(value: Any) -> tuple[Any, ...] | None:
    code = getattr(value, "__code__", None)
    if code is None:
        return None
    return (
        code.co_argcount,
        code.co_kwonlyargcount,
        code.co_posonlyargcount,
        code.co_names,
        code.co_varnames,
        code.co_consts,
        code.co_code,
        getattr(value, "__defaults__", None),
        getattr(value, "__kwdefaults__", None),
    )


def _dataclass_field_fingerprint(value: Mapping[str, Any]) -> tuple[tuple[str, Any, Any], ...]:
    return tuple(
        (name, field.default, field.default_factory)
        for name, field in value.items()
    )


def _public_export_default(name: str) -> Any:
    live_value = getattr(_document_module, name)
    default_value = _DOCUMENT_DEFAULTS[name]
    return live_value if live_value == default_value else default_value


for _name in _PUBLIC_CONSTANT_NAMES:
    globals()[_name] = _public_export_default(_name)


class DatabricksVLLMSmokeJobConfig(_DOCUMENT_DEFAULTS["DatabricksVLLMSmokeJobConfig"]):
    __slots__ = ()

    def __post_init__(self) -> None:
        if not self.benchmark_id:
            raise ValueError("benchmark_id must be non-empty")
        if not self.output_dir:
            raise ValueError("output_dir must be non-empty")
        if not self.runner_python_file:
            raise ValueError("runner_python_file must be non-empty")
        if not self.run_name:
            raise ValueError("run_name must be non-empty")
        if not self.task_key:
            raise ValueError("task_key must be non-empty")
        if self.wheel_uri is not None and not self.wheel_uri:
            raise ValueError("wheel_uri must be non-empty when provided")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.import_probe_timeout_seconds <= 0:
            raise ValueError("import_probe_timeout_seconds must be positive")
        if self.server_start_timeout_seconds <= 0:
            raise ValueError("server_start_timeout_seconds must be positive")
        if not self.local_root:
            raise ValueError("local_root must be non-empty")
        if not self.server_host:
            raise ValueError("server_host must be non-empty")
        if not 0 < self.server_port < 65536:
            raise ValueError("server_port must be between 1 and 65535")
        if not self.client_host:
            raise ValueError("client_host must be non-empty")
        _cluster_config_from_vllm_smoke_job(self)


if _public_config_class_is_reusable():
    DatabricksVLLMSmokeJobConfig = _document_module.DatabricksVLLMSmokeJobConfig


def build_databricks_vllm_smoke_run_submit_payload(config: DatabricksVLLMSmokeJobConfig) -> dict[str, Any]:
    return _call_document_function("build_databricks_vllm_smoke_run_submit_payload", config)


def write_databricks_vllm_smoke_run_submit_json(
    config: DatabricksVLLMSmokeJobConfig,
    path: str | Path,
) -> None:
    return _call_document_function("write_databricks_vllm_smoke_run_submit_json", config, path)


def write_databricks_vllm_smoke_runner_script(path: str | Path) -> None:
    return _call_document_function("write_databricks_vllm_smoke_runner_script", path)


def _cluster_config_from_vllm_smoke_job(config: DatabricksVLLMSmokeJobConfig) -> DatabricksSingleNodeG5ClusterConfig:
    return _call_document_function("_cluster_config_from_vllm_smoke_job", config)


def main(argv: Sequence[str] | None = None) -> int:
    return _call_document_function("main", argv)


_DEFAULT_COMPAT_FUNCTIONS = {
    "build_databricks_vllm_smoke_run_submit_payload": build_databricks_vllm_smoke_run_submit_payload,
    "write_databricks_vllm_smoke_run_submit_json": write_databricks_vllm_smoke_run_submit_json,
    "write_databricks_vllm_smoke_runner_script": write_databricks_vllm_smoke_runner_script,
    "_cluster_config_from_vllm_smoke_job": _cluster_config_from_vllm_smoke_job,
    "main": main,
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
    return isinstance(value, FunctionType) and value.__globals__ is vars(_document_defaults_module)


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
