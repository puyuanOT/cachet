"""Compatibility wrapper for :mod:`document_kv_cache.databricks_engine_probe_job`."""

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
    DEFAULT_AWS_G5_NODE_TYPE,
    DEFAULT_DATABRICKS_DATA_SECURITY_MODE,
    DEFAULT_DATABRICKS_SPARK_VERSION,
    DatabricksSingleNodeG5ClusterConfig,
    build_single_node_g5_cluster,
)
from document_kv_cache.engine_adapters import ServingBackend
from document_kv_cache.release_evidence import REQUIRED_ENGINE_PROBE_BACKENDS

import document_kv_cache.databricks_engine_probe_job as _document_module

__all__ = reexport_public(
    "document_kv_cache.databricks_engine_probe_job",
    (
        "DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME",
        "DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY",
        "DEFAULT_DATABRICKS_ENGINE_PROBE_PURPOSE",
        "DEFAULT_DATABRICKS_ENGINE_PROBE_BACKEND_CONFIG_KEY",
        "ENGINE_PROBE_RUNNER_SCRIPT",
    ),
    globals(),
)

__all__ += [
    "ENGINE_PROBE_TARGETS_RECORD_TYPE",
    "ENGINE_PROBE_TARGETS_SCHEMA_VERSION",
    "DatabricksEngineProbeJobConfig",
    "DatabricksEngineProbeMatrixJobConfig",
    "DatabricksEngineProbeTargetConfig",
    "DatabricksEngineProbeTargetsFile",
    "build_databricks_engine_probe_run_submit_payload",
    "build_databricks_engine_probe_matrix_run_submit_payload",
    "read_databricks_engine_probe_targets_json",
    "read_databricks_engine_probe_targets_file_json",
    "write_databricks_engine_probe_run_submit_json",
    "write_databricks_engine_probe_matrix_run_submit_json",
    "write_databricks_engine_probe_runner_script",
    "main",
    "argparse",
    "json",
    "Mapping",
    "Sequence",
    "dataclass",
    "field",
    "Path",
    "Any",
    "DEFAULT_AWS_G5_NODE_TYPE",
    "DEFAULT_DATABRICKS_DATA_SECURITY_MODE",
    "DEFAULT_DATABRICKS_SPARK_VERSION",
    "DatabricksSingleNodeG5ClusterConfig",
    "build_single_node_g5_cluster",
    "ServingBackend",
    "REQUIRED_ENGINE_PROBE_BACKENDS",
]


def _load_document_defaults_module():
    module_path = Path(_document_module.__file__)
    module_name = "_restaurant_kv_serving_databricks_engine_probe_job_document_defaults"
    spec = _importlib_util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load document databricks_engine_probe_job defaults from {module_path}")
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
        "DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME",
        "DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY",
        "DEFAULT_DATABRICKS_ENGINE_PROBE_PURPOSE",
        "DEFAULT_DATABRICKS_ENGINE_PROBE_BACKEND_CONFIG_KEY",
        "ENGINE_PROBE_TARGETS_RECORD_TYPE",
        "ENGINE_PROBE_TARGETS_SCHEMA_VERSION",
        "ENGINE_PROBE_RUNNER_SCRIPT",
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


def _public_class_base(name: str) -> type:
    if _is_pristine_public_class(name):
        return getattr(_document_module, name)
    return _DOCUMENT_DEFAULTS[name]


def _public_export_default(name: str) -> Any:
    live_value = getattr(_document_module, name)
    default_value = _DOCUMENT_DEFAULTS[name]
    return live_value if live_value == default_value else default_value


for _name in _PUBLIC_CONSTANT_NAMES:
    globals()[_name] = _public_export_default(_name)


class DatabricksEngineProbeTargetsFile(_public_class_base("DatabricksEngineProbeTargetsFile")):
    __slots__ = ()

    def __post_init__(self) -> None:
        if type(self.release_safe) is not bool:
            raise ValueError("release_safe must be a boolean")
        if not self.probe_targets:
            raise ValueError("probe_targets must be non-empty")
        object.__setattr__(self, "probe_targets", tuple(self.probe_targets))


class DatabricksEngineProbeTargetConfig(_public_class_base("DatabricksEngineProbeTargetConfig")):
    __slots__ = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected_backend", _serving_backend(self.expected_backend))
        if not self.handoff_json:
            raise ValueError("handoff_json must be non-empty")
        if not self.probe_factory:
            raise ValueError("probe_factory must be non-empty")
        if not self.output_json:
            raise ValueError("output_json must be non-empty")
        if self.payload_uri is not None and not self.payload_uri:
            raise ValueError("payload_uri must be non-empty when provided")
        if self.task_key is not None and not self.task_key:
            raise ValueError("task_key must be non-empty when provided")
        if self.engine_version is not None and not self.engine_version:
            raise ValueError("engine_version must be non-empty when provided")
        if self.actions_output_json is not None and not self.actions_output_json:
            raise ValueError("actions_output_json must be non-empty when provided")
        if type(self.allow_non_native_probe) is not bool:
            raise ValueError("allow_non_native_probe must be a boolean")
        _validate_metadata_items(self.metadata)
        object.__setattr__(self, "metadata", tuple(self.metadata))


_PROBE_TARGET_BASES = tuple(dict.fromkeys(DatabricksEngineProbeTargetConfig.__mro__[:2]))


class DatabricksEngineProbeMatrixJobConfig(_public_class_base("DatabricksEngineProbeMatrixJobConfig")):
    __slots__ = ()

    def __post_init__(self) -> None:
        if not self.probe_targets:
            raise ValueError("probe_targets must be non-empty")
        if not self.runner_python_file:
            raise ValueError("runner_python_file must be non-empty")
        if not self.run_name:
            raise ValueError("run_name must be non-empty")
        if self.wheel_uri is not None and not self.wheel_uri:
            raise ValueError("wheel_uri must be non-empty when provided")
        if type(self.release_safe) is not bool:
            raise ValueError("release_safe must be a boolean")
        targets = tuple(_coerce_probe_target(target) for target in self.probe_targets)
        _validate_probe_target_backends(targets, release_safe=self.release_safe)
        _validate_probe_target_task_keys(targets)
        _validate_release_safe_probe_targets(targets, release_safe=self.release_safe)
        object.__setattr__(self, "probe_targets", targets)
        _cluster_config_from_engine_probe_matrix_job(self)


class DatabricksEngineProbeJobConfig(_public_class_base("DatabricksEngineProbeJobConfig")):
    __slots__ = ()

    def __post_init__(self) -> None:
        if not self.handoff_json:
            raise ValueError("handoff_json must be non-empty")
        if not self.probe_factory:
            raise ValueError("probe_factory must be non-empty")
        if not self.output_json:
            raise ValueError("output_json must be non-empty")
        if not self.runner_python_file:
            raise ValueError("runner_python_file must be non-empty")
        if self.payload_uri is not None and not self.payload_uri:
            raise ValueError("payload_uri must be non-empty when provided")
        if not self.run_name:
            raise ValueError("run_name must be non-empty")
        if not self.task_key:
            raise ValueError("task_key must be non-empty")
        if self.wheel_uri is not None and not self.wheel_uri:
            raise ValueError("wheel_uri must be non-empty when provided")
        if self.engine_version is not None and not self.engine_version:
            raise ValueError("engine_version must be non-empty when provided")
        if self.actions_output_json is not None and not self.actions_output_json:
            raise ValueError("actions_output_json must be non-empty when provided")
        _validate_metadata_items(self.metadata)
        object.__setattr__(self, "metadata", tuple(self.metadata))
        _validate_release_safe_probe_job(self)
        object.__setattr__(self, "expected_backend", _serving_backend(self.expected_backend))
        _cluster_config_from_engine_probe_job(self)


def build_databricks_engine_probe_run_submit_payload(config: DatabricksEngineProbeJobConfig) -> dict[str, Any]:
    return _call_document_function("build_databricks_engine_probe_run_submit_payload", config)


def build_databricks_engine_probe_matrix_run_submit_payload(
    config: DatabricksEngineProbeMatrixJobConfig,
) -> dict[str, Any]:
    return _call_document_function("build_databricks_engine_probe_matrix_run_submit_payload", config)


def write_databricks_engine_probe_run_submit_json(
    config: DatabricksEngineProbeJobConfig,
    path: str | Path,
) -> None:
    return _call_document_function("write_databricks_engine_probe_run_submit_json", config, path)


def write_databricks_engine_probe_matrix_run_submit_json(
    config: DatabricksEngineProbeMatrixJobConfig,
    path: str | Path,
) -> None:
    return _call_document_function("write_databricks_engine_probe_matrix_run_submit_json", config, path)


def write_databricks_engine_probe_runner_script(path: str | Path) -> None:
    return _call_document_function("write_databricks_engine_probe_runner_script", path)


def read_databricks_engine_probe_targets_json(path: str | Path) -> tuple[DatabricksEngineProbeTargetConfig, ...]:
    return _call_document_function("read_databricks_engine_probe_targets_json", path)


def read_databricks_engine_probe_targets_file_json(path: str | Path) -> DatabricksEngineProbeTargetsFile:
    return _call_document_function("read_databricks_engine_probe_targets_file_json", path)


def _cluster_config_from_engine_probe_job(
    config: DatabricksEngineProbeJobConfig,
) -> DatabricksSingleNodeG5ClusterConfig:
    return DatabricksSingleNodeG5ClusterConfig(
        purpose=DEFAULT_DATABRICKS_ENGINE_PROBE_PURPOSE,
        node_type_id=config.node_type_id,
        spark_version=config.spark_version,
        data_security_mode=config.data_security_mode,
        single_user_name=config.single_user_name,
        availability=config.availability,
        zone_id=config.zone_id,
        custom_tags=config.custom_tags,
    )


def _cluster_config_from_engine_probe_matrix_job(
    config: DatabricksEngineProbeMatrixJobConfig,
) -> DatabricksSingleNodeG5ClusterConfig:
    return DatabricksSingleNodeG5ClusterConfig(
        purpose=DEFAULT_DATABRICKS_ENGINE_PROBE_PURPOSE,
        node_type_id=config.node_type_id,
        spark_version=config.spark_version,
        data_security_mode=config.data_security_mode,
        single_user_name=config.single_user_name,
        availability=config.availability,
        zone_id=config.zone_id,
        custom_tags=config.custom_tags,
    )


def _coerce_probe_target(target: DatabricksEngineProbeTargetConfig) -> DatabricksEngineProbeTargetConfig:
    if isinstance(target, _PROBE_TARGET_BASES):
        return target
    raise TypeError("probe_targets entries must be DatabricksEngineProbeTargetConfig")


def _validate_probe_target_backends(
    targets: Sequence[DatabricksEngineProbeTargetConfig],
    *,
    release_safe: bool,
) -> None:
    return _call_document_function("_validate_probe_target_backends", targets, release_safe=release_safe)


def _validate_probe_target_task_keys(targets: Sequence[DatabricksEngineProbeTargetConfig]) -> None:
    return _call_document_function("_validate_probe_target_task_keys", targets)


def _validate_release_safe_probe_targets(
    targets: Sequence[DatabricksEngineProbeTargetConfig],
    *,
    release_safe: bool,
) -> None:
    return _call_document_function("_validate_release_safe_probe_targets", targets, release_safe=release_safe)


def _validate_release_safe_probe_job(config: DatabricksEngineProbeJobConfig) -> None:
    return _call_document_function("_validate_release_safe_probe_job", config)


def _default_task_key_for_backend(backend: ServingBackend) -> str:
    return f"{DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY}_{backend.value}"


def _validate_metadata_items(items: Sequence[str]) -> None:
    return _call_document_function("_validate_metadata_items", items)


def _serving_backend(value: ServingBackend | str) -> ServingBackend:
    return _call_document_function("_serving_backend", value)


def main(argv: Sequence[str] | None = None) -> int:
    return _call_document_function("main", argv)


_DEFAULT_COMPAT_FUNCTIONS = {
    "build_databricks_engine_probe_run_submit_payload": build_databricks_engine_probe_run_submit_payload,
    "build_databricks_engine_probe_matrix_run_submit_payload": build_databricks_engine_probe_matrix_run_submit_payload,
    "write_databricks_engine_probe_run_submit_json": write_databricks_engine_probe_run_submit_json,
    "write_databricks_engine_probe_matrix_run_submit_json": write_databricks_engine_probe_matrix_run_submit_json,
    "write_databricks_engine_probe_runner_script": write_databricks_engine_probe_runner_script,
    "read_databricks_engine_probe_targets_json": read_databricks_engine_probe_targets_json,
    "read_databricks_engine_probe_targets_file_json": read_databricks_engine_probe_targets_file_json,
    "_validate_probe_target_backends": _validate_probe_target_backends,
    "_validate_probe_target_task_keys": _validate_probe_target_task_keys,
    "_validate_release_safe_probe_targets": _validate_release_safe_probe_targets,
    "_validate_release_safe_probe_job": _validate_release_safe_probe_job,
    "_validate_metadata_items": _validate_metadata_items,
    "_serving_backend": _serving_backend,
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
