"""Compatibility wrapper for :mod:`document_kv_cache.databricks_job`."""

from __future__ import annotations

import argparse
import importlib.util as _importlib_util
import json
import sys as _sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from types import CodeType, FunctionType
from typing import Any

from document_kv_cache._reexport import reexport_public

import document_kv_cache.databricks_job as _document_module

__all__ = reexport_public(
    "document_kv_cache.databricks_job",
    (
        "DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE",
        "DEFAULT_AWS_G5_NODE_TYPE",
        "DEFAULT_DATABRICKS_SPARK_VERSION",
        "DEFAULT_DATABRICKS_RUN_NAME",
        "DEFAULT_DATABRICKS_TASK_KEY",
        "DEFAULT_DATABRICKS_PURPOSE",
        "DEFAULT_DATABRICKS_DATA_SECURITY_MODE",
        "DEDICATED_DATABRICKS_DATA_SECURITY_MODE",
        "SINGLE_USER_DATABRICKS_DATA_SECURITY_MODES",
    ),
    globals(),
)

__all__ += [
    "RESERVED_SINGLE_NODE_G5_TAG_KEYS",
    "RESERVED_SINGLE_NODE_GPU_TAG_KEYS",
    "RUNNER_SCRIPT",
    "DatabricksSingleNodeG5ClusterConfig",
    "DatabricksSingleNodeGPUClusterConfig",
    "DatabricksBenchmarkJobConfig",
    "validate_aws_g5_node_type",
    "validate_aws_single_node_gpu_type",
    "build_single_node_g5_cluster",
    "build_single_node_gpu_cluster",
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

def _load_document_defaults_module():
    module_path = Path(_document_module.__file__)
    module_name = "_restaurant_kv_serving_databricks_job_document_defaults"
    spec = _importlib_util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load document databricks_job defaults from {module_path}")
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
        "DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE",
        "DEFAULT_AWS_G5_NODE_TYPE",
        "DEFAULT_DATABRICKS_SPARK_VERSION",
        "DEFAULT_DATABRICKS_RUN_NAME",
        "DEFAULT_DATABRICKS_TASK_KEY",
        "DEFAULT_DATABRICKS_PURPOSE",
        "DEFAULT_DATABRICKS_DATA_SECURITY_MODE",
        "DEDICATED_DATABRICKS_DATA_SECURITY_MODE",
        "SINGLE_USER_DATABRICKS_DATA_SECURITY_MODES",
        "RESERVED_SINGLE_NODE_G5_TAG_KEYS",
        "RESERVED_SINGLE_NODE_GPU_TAG_KEYS",
        "RUNNER_SCRIPT",
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


def _class_fingerprint(value: type) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (name, _class_attribute_fingerprint(name, attribute, owner=value))
        for name, attribute in sorted(vars(value).items())
        if name not in {"__doc__", "__module__", "_abc_impl"}
    )


def _class_attribute_fingerprint(name: str, value: Any, *, owner: type) -> Any:
    if name == "__dataclass_fields__":
        return _dataclass_field_fingerprint(value) if isinstance(value, Mapping) else _object_fingerprint(value)
    if name == "__dataclass_params__":
        return _object_fingerprint(value)
    if isinstance(value, property):
        return (
            "property",
            _function_fingerprint(value.fget, owner=owner),
            _function_fingerprint(value.fset, owner=owner),
            _function_fingerprint(value.fdel, owner=owner),
            _object_fingerprint(value.__doc__),
        )
    objclass = _safe_getattr(value, "__objclass__", None)
    descriptor_name = _safe_getattr(value, "__name__", None)
    if isinstance(objclass, type) and descriptor_name is not None:
        return (
            "descriptor",
            type(value).__qualname__,
            _object_fingerprint(descriptor_name),
            _descriptor_owner_fingerprint(objclass, owner),
        )
    if isinstance(value, dict):
        return _mapping_fingerprint(value)
    function_fingerprint = _function_fingerprint(value, owner=owner)
    if function_fingerprint is not None:
        return ("function", function_fingerprint)
    return _object_fingerprint(value)


def _descriptor_owner_fingerprint(objclass: type, owner: type) -> tuple[str, str] | tuple[str, str, str]:
    if objclass is owner:
        return ("self", owner.__qualname__)
    return ("foreign", objclass.__module__, objclass.__qualname__)


def _object_fingerprint(value: Any) -> tuple[str, str, str]:
    value_type = type(value)
    try:
        value_repr = repr(value)
    except Exception as exc:  # pragma: no cover - defensive for mutated public classes
        value_repr = f"<unrepresentable: {type(exc).__module__}.{type(exc).__qualname__}>"
    return (value_type.__module__, value_type.__qualname__, value_repr)


def _mapping_fingerprint(value: Mapping[Any, Any]) -> tuple[Any, ...]:
    try:
        return tuple(
            sorted(
                (_object_fingerprint(key), _object_fingerprint(item_value))
                for key, item_value in value.items()
            )
        )
    except Exception:
        return ("mapping", _object_fingerprint(value))


def _safe_getattr(value: Any, name: str, default: Any) -> Any:
    try:
        return getattr(value, name)
    except Exception:
        return default


def _function_fingerprint(
    value: Any,
    *,
    owner: type | None,
    include_closure: bool = True,
) -> tuple[Any, ...] | None:
    code = _safe_getattr(value, "__code__", None)
    if not isinstance(code, CodeType):
        return None
    kwdefaults = _safe_getattr(value, "__kwdefaults__", None)
    return (
        code.co_argcount,
        code.co_kwonlyargcount,
        code.co_posonlyargcount,
        code.co_names,
        code.co_varnames,
        _object_fingerprint(code.co_consts),
        code.co_code,
        _object_fingerprint(_safe_getattr(value, "__defaults__", None)),
        _mapping_fingerprint(kwdefaults) if isinstance(kwdefaults, Mapping) else _object_fingerprint(kwdefaults),
        _closure_fingerprint(value, code, owner=owner) if include_closure else (),
    )


def _closure_fingerprint(value: Any, code: CodeType, *, owner: type | None) -> tuple[Any, ...]:
    closure = _safe_getattr(value, "__closure__", None)
    if closure is None:
        return ()
    if not isinstance(closure, tuple):
        return ("closure", _object_fingerprint(closure))
    return tuple(
        (
            code.co_freevars[index] if index < len(code.co_freevars) else str(index),
            _cell_fingerprint(cell, owner=owner),
        )
        for index, cell in enumerate(closure)
    )


def _cell_fingerprint(cell: Any, *, owner: type | None) -> Any:
    try:
        value = cell.cell_contents
    except Exception as exc:
        return ("cell", "unreadable", f"{type(exc).__module__}.{type(exc).__qualname__}")
    if owner is not None and (
        value is owner
        or (
            isinstance(value, type)
            and value.__module__ == owner.__module__
            and value.__qualname__ == owner.__qualname__
        )
    ):
        return ("self_class", owner.__qualname__)
    function_fingerprint = _function_fingerprint(value, owner=owner, include_closure=False)
    if function_fingerprint is not None:
        return ("function", function_fingerprint)
    return _object_fingerprint(value)


def _dataclass_field_fingerprint(value: Mapping[str, Any]) -> tuple[Any, ...]:
    try:
        return tuple(
            (
                _object_fingerprint(name),
                _object_fingerprint(_safe_getattr(field, "default", None)),
                _object_fingerprint(_safe_getattr(field, "default_factory", None)),
            )
            for name, field in value.items()
        )
    except Exception:
        return ("dataclass_fields", _object_fingerprint(value))


def _public_class_base(name: str):
    if _is_pristine_public_class(name):
        return getattr(_document_module, name)
    return _DOCUMENT_DEFAULTS[name]


def _public_constant_default(name: str) -> Any:
    live_value = getattr(_document_module, name)
    default_value = _DOCUMENT_DEFAULTS[name]
    return live_value if live_value == default_value else default_value


for _name in _PUBLIC_CONSTANT_NAMES:
    globals()[_name] = _public_constant_default(_name)


class DatabricksSingleNodeG5ClusterConfig(_public_class_base("DatabricksSingleNodeG5ClusterConfig")):
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


DatabricksSingleNodeGPUClusterConfig = DatabricksSingleNodeG5ClusterConfig


class DatabricksBenchmarkJobConfig(_public_class_base("DatabricksBenchmarkJobConfig")):
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
        if self.vllm_native_probe_delegate_factory is not None and not self.vllm_native_probe_delegate_factory:
            raise ValueError("vllm_native_probe_delegate_factory must be non-empty when provided")
        if self.sglang_native_probe_delegate_factory is not None and not self.sglang_native_probe_delegate_factory:
            raise ValueError("sglang_native_probe_delegate_factory must be non-empty when provided")


def validate_aws_g5_node_type(node_type_id: str) -> None:
    return _call_document_function("validate_aws_g5_node_type", node_type_id)


validate_aws_single_node_gpu_type = validate_aws_g5_node_type


def build_databricks_run_submit_payload(config: DatabricksBenchmarkJobConfig) -> dict[str, Any]:
    return _call_document_function("build_databricks_run_submit_payload", config)


def write_databricks_run_submit_json(config: DatabricksBenchmarkJobConfig, path: str | Path) -> None:
    return _call_document_function("write_databricks_run_submit_json", config, path)


def write_databricks_runner_script(path: str | Path) -> None:
    return _call_document_function("write_databricks_runner_script", path)


def build_single_node_g5_cluster(config: DatabricksSingleNodeG5ClusterConfig) -> dict[str, Any]:
    return _call_document_function("build_single_node_g5_cluster", config)


build_single_node_gpu_cluster = build_single_node_g5_cluster


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
    "validate_aws_single_node_gpu_type": validate_aws_single_node_gpu_type,
    "build_databricks_run_submit_payload": build_databricks_run_submit_payload,
    "write_databricks_run_submit_json": write_databricks_run_submit_json,
    "write_databricks_runner_script": write_databricks_runner_script,
    "build_single_node_g5_cluster": build_single_node_g5_cluster,
    "build_single_node_gpu_cluster": build_single_node_gpu_cluster,
    "_single_node_g5_cluster": _single_node_g5_cluster,
    "_cluster_config_from_benchmark_job": _cluster_config_from_benchmark_job,
    "_is_single_user_mode": _is_single_user_mode,
    "_runner_parameters": _runner_parameters,
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
