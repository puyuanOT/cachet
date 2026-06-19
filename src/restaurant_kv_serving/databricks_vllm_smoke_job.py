"""Compatibility wrapper for :mod:`document_kv_cache.databricks_vllm_smoke_job`."""

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
from document_kv_cache.databricks_job import (
    DEFAULT_AWS_G5_NODE_TYPE,
    DEFAULT_DATABRICKS_DATA_SECURITY_MODE,
    DEFAULT_DATABRICKS_SPARK_VERSION,
    DatabricksSingleNodeG5ClusterConfig,
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
    "DEFAULT_AWS_G5_NODE_TYPE",
    "DEFAULT_DATABRICKS_DATA_SECURITY_MODE",
    "DEFAULT_DATABRICKS_SPARK_VERSION",
    "DatabricksSingleNodeG5ClusterConfig",
    "build_single_node_g5_cluster",
    "DEFAULT_LOCAL_ROOT",
    "SERVER_HOST",
    "SERVER_PORT",
]


def build_databricks_vllm_smoke_run_submit_payload(config: DatabricksVLLMSmokeJobConfig) -> dict[str, Any]:
    return _call_document_function("build_databricks_vllm_smoke_run_submit_payload", config)


def write_databricks_vllm_smoke_run_submit_json(
    config: DatabricksVLLMSmokeJobConfig,
    path: str | Path,
) -> None:
    return _call_document_function("write_databricks_vllm_smoke_run_submit_json", config, path)


def write_databricks_vllm_smoke_runner_script(path: str | Path) -> None:
    return _call_document_function("write_databricks_vllm_smoke_runner_script", path)


def main(argv: Sequence[str] | None = None) -> int:
    return _call_document_function("main", argv)


_DEFAULT_COMPAT_FUNCTIONS = {
    "build_databricks_vllm_smoke_run_submit_payload": build_databricks_vllm_smoke_run_submit_payload,
    "write_databricks_vllm_smoke_run_submit_json": write_databricks_vllm_smoke_run_submit_json,
    "write_databricks_vllm_smoke_runner_script": write_databricks_vllm_smoke_runner_script,
    "main": main,
}
_DOCUMENT_DEFAULTS = {
    name: value
    for name, value in vars(_document_module).items()
    if not name.startswith("__")
}
_PATCH_LOCK = RLock()


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
    for name in __all__:
        if name in namespace:
            namespace[name] = _document_global_for_legacy(name)
    for name, value in tuple(namespace.items()):
        if _is_document_function(value):
            namespace[name] = _clone_document_function(value, namespace)
    if namespace.get("DatabricksVLLMSmokeJobConfig") is _DOCUMENT_DEFAULTS["DatabricksVLLMSmokeJobConfig"]:
        namespace["DatabricksVLLMSmokeJobConfig"] = _isolated_config_class(namespace)
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


def _isolated_config_class(namespace: dict[str, Any]) -> type:
    base = _DOCUMENT_DEFAULTS["DatabricksVLLMSmokeJobConfig"]
    return type(
        base.__name__,
        (base,),
        {
            "__doc__": base.__doc__,
            "__module__": base.__module__,
            "__post_init__": _clone_document_function(base.__post_init__, namespace),
        },
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


del reexport_public
