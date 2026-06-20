"""Compatibility facade for :mod:`document_kv_cache.storage_benchmark`."""

from __future__ import annotations

import argparse
import importlib.util as _importlib_util
import json
import shutil
import sys as _sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import RLock as _RLock
from types import FunctionType as _FunctionType
from typing import Any

import document_kv_cache.storage_benchmark as _document_module
from document_kv_cache.storage_benchmark import (
    RELEASE_STORAGE_BENCHMARK_READERS,
    STORAGE_BENCHMARK_RECORD_TYPE,
    SUPPORTED_STORAGE_BENCHMARK_READERS,
)
from restaurant_kv_serving.kvpack import PackChunk, write_kvpack
from restaurant_kv_serving.models import ChunkRef, DocumentChunkType, KVCacheKey
from restaurant_kv_serving.storage import (
    DiskRangeReader,
    MemoryRangeReader,
    UnityCatalogVolumeRangeReader,
    is_real_uc_volume_root,
    local_path,
)


def _load_document_defaults_module():
    module_path = Path(_document_module.__file__)
    module_name = "_restaurant_kv_serving_storage_benchmark_document_defaults"
    spec = _importlib_util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load document storage_benchmark defaults from {module_path}")
    module = _importlib_util.module_from_spec(spec)
    _sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_document_defaults_module = _load_document_defaults_module()
_DOCUMENT_DEFAULTS = {
    name: value
    for name, value in vars(_document_defaults_module).items()
    if name != "__all__" and not name.startswith("__")
}
_PUBLIC_CLASS_NAMES = frozenset(
    {
        "StorageBenchmarkConfig",
        "StorageBenchmarkEvidence",
        "StorageBenchmarkResult",
        "StorageReaderBenchmarkResult",
    }
)
_PUBLIC_SURFACE_NAMES = _PUBLIC_CLASS_NAMES | frozenset(
    {
        "STORAGE_BENCHMARK_RECORD_TYPE",
        "SUPPORTED_STORAGE_BENCHMARK_READERS",
        "RELEASE_STORAGE_BENCHMARK_READERS",
    }
)


for _name, _value in vars(_document_defaults_module).items():
    if _name != "__all__" and not _name.startswith("__"):
        globals()[_name] = _value


def _is_pristine_public_value(name: str) -> bool:
    live_value = getattr(_document_module, name)
    default_value = _DOCUMENT_DEFAULTS[name]
    if name in _PUBLIC_CLASS_NAMES:
        return (
            isinstance(live_value, type)
            and live_value.__module__ == _document_module.__name__
            and live_value.__qualname__ == default_value.__qualname__
        )
    return live_value == default_value


def _public_surface_default(name: str) -> Any:
    if _is_pristine_public_value(name):
        return getattr(_document_module, name)
    return _DOCUMENT_DEFAULTS[name]


for _name in _PUBLIC_SURFACE_NAMES:
    globals()[_name] = _public_surface_default(_name)

_INITIAL_PUBLIC_SURFACE = {name: globals()[name] for name in _PUBLIC_SURFACE_NAMES}


def run_storage_benchmark(config: StorageBenchmarkConfig) -> StorageBenchmarkResult:
    return _call_document_function("run_storage_benchmark", config)


def storage_benchmark_result_to_record(result: StorageBenchmarkResult) -> dict[str, Any]:
    return _call_document_function("storage_benchmark_result_to_record", result)


def evaluate_storage_benchmark_evidence(
    result: StorageBenchmarkResult,
    *,
    required_readers: Sequence[str] | None = None,
    require_real_uc_volume: bool = False,
) -> StorageBenchmarkEvidence:
    return _call_document_function(
        "evaluate_storage_benchmark_evidence",
        result,
        required_readers=required_readers,
        require_real_uc_volume=require_real_uc_volume,
    )


def evaluate_release_storage_benchmark_evidence(result: StorageBenchmarkResult) -> StorageBenchmarkEvidence:
    return _call_document_function("evaluate_release_storage_benchmark_evidence", result)


def storage_benchmark_evidence_to_record(evidence: StorageBenchmarkEvidence) -> dict[str, Any]:
    return _call_document_function("storage_benchmark_evidence_to_record", evidence)


def write_storage_benchmark_result_json(result: StorageBenchmarkResult, path: str | Path) -> None:
    return _call_document_function("write_storage_benchmark_result_json", result, path)


def main(argv: Sequence[str] | None = None) -> int:
    return _call_document_function("main", argv)


_DEFAULT_COMPAT_FUNCTIONS = {
    "run_storage_benchmark": run_storage_benchmark,
    "storage_benchmark_result_to_record": storage_benchmark_result_to_record,
    "evaluate_storage_benchmark_evidence": evaluate_storage_benchmark_evidence,
    "evaluate_release_storage_benchmark_evidence": evaluate_release_storage_benchmark_evidence,
    "storage_benchmark_evidence_to_record": storage_benchmark_evidence_to_record,
    "write_storage_benchmark_result_json": write_storage_benchmark_result_json,
    "main": main,
}
_PATCH_LOCK = _RLock()
_LEGACY_PATCH_NAMES = tuple(name for name in _DOCUMENT_DEFAULTS if name in globals())


def _call_document_function(name: str, *args, **kwargs):
    excluded_names = {name}
    with _PATCH_LOCK:
        return _isolated_document_namespace(excluded_names=excluded_names)[name](*args, **kwargs)


def _document_global_for_legacy(name: str):
    if name not in globals():
        return _DOCUMENT_DEFAULTS[name]
    current = globals()[name]
    if name in _PUBLIC_SURFACE_NAMES and _INITIAL_PUBLIC_SURFACE.get(name) is current:
        return current
    if _DEFAULT_COMPAT_FUNCTIONS.get(name) is current:
        return _DOCUMENT_DEFAULTS[name]
    return current


def _isolated_document_namespace(*, excluded_names: set[str]) -> dict[str, Any]:
    namespace = dict(_DOCUMENT_DEFAULTS)
    has_legacy_overrides = False
    for name in _LEGACY_PATCH_NAMES:
        if name in namespace and name not in excluded_names:
            value = _document_global_for_legacy(name)
            namespace[name] = value
            has_legacy_overrides = has_legacy_overrides or value is not _DOCUMENT_DEFAULTS[name]
    for name, value in tuple(namespace.items()):
        if _is_document_function(value):
            namespace[name] = _clone_document_function(value, namespace)
    if not has_legacy_overrides:
        return namespace
    for name, value in tuple(namespace.items()):
        if _is_document_class(value):
            namespace[name] = _clone_document_class(value, namespace)
    return namespace


def _is_document_function(value: Any) -> bool:
    return isinstance(value, _FunctionType) and value.__globals__ is vars(_document_defaults_module)


def _is_document_class(value: Any) -> bool:
    return isinstance(value, type) and value.__module__ == _document_defaults_module.__name__


def _clone_document_function(function: _FunctionType, namespace: dict[str, Any]) -> _FunctionType:
    clone = _FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = dict(function.__annotations__)
    clone.__doc__ = function.__doc__
    clone.__module__ = function.__module__
    return clone


def _clone_document_class(cls: type, namespace: dict[str, Any]) -> type:
    attrs: dict[str, Any] = {
        "__module__": cls.__module__,
        "__doc__": cls.__doc__,
        "__slots__": (),
    }
    for name, value in vars(cls).items():
        if _is_document_function(value):
            attrs[name] = _clone_document_function(value, namespace)
        elif isinstance(value, property):
            attrs[name] = _clone_document_property(value, namespace)
    return type(cls.__name__, (cls,), attrs)


def _clone_document_property(prop: property, namespace: dict[str, Any]) -> property:
    fget = _clone_document_function(prop.fget, namespace) if _is_document_function(prop.fget) else prop.fget
    fset = _clone_document_function(prop.fset, namespace) if _is_document_function(prop.fset) else prop.fset
    fdel = _clone_document_function(prop.fdel, namespace) if _is_document_function(prop.fdel) else prop.fdel
    return property(fget, fset, fdel, prop.__doc__)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
