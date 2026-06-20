"""Compatibility facade for :mod:`document_kv_cache.benchmark_plan`."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock as _RLock
from types import FunctionType as _FunctionType
from typing import Any

import document_kv_cache.benchmark_plan as _document_module


for _name, _value in vars(_document_module).items():
    if _name != "__all__" and not _name.startswith("__"):
        globals()[_name] = _value


def build_v1_benchmark_plan(config: BenchmarkPlanConfig) -> BenchmarkJobPlan:
    return _call_document_function("build_v1_benchmark_plan", config)


def benchmark_job_plan_to_record(plan: BenchmarkJobPlan) -> dict[str, Any]:
    return _call_document_function("benchmark_job_plan_to_record", plan)


def engine_probe_targets_to_record(
    engine_probes: Sequence[EngineProbePlanConfig],
    *,
    release_safe: bool = False,
) -> dict[str, Any]:
    return _call_document_function("engine_probe_targets_to_record", engine_probes, release_safe=release_safe)


def write_engine_probe_targets_json(
    plan: BenchmarkJobPlan,
    path: str | Path,
    *,
    release_safe: bool = False,
) -> None:
    return _call_document_function("write_engine_probe_targets_json", plan, path, release_safe=release_safe)


def write_benchmark_job_plan_json(plan: BenchmarkJobPlan, path: str | Path) -> None:
    return _call_document_function("write_benchmark_job_plan_json", plan, path)


def write_benchmark_job_plan_shell(plan: BenchmarkJobPlan, path: str | Path) -> None:
    return _call_document_function("write_benchmark_job_plan_shell", plan, path)


def main(argv: Sequence[str] | None = None) -> int:
    return _call_document_function("main", argv)


_DEFAULT_COMPAT_FUNCTIONS = {
    "build_v1_benchmark_plan": build_v1_benchmark_plan,
    "benchmark_job_plan_to_record": benchmark_job_plan_to_record,
    "engine_probe_targets_to_record": engine_probe_targets_to_record,
    "write_engine_probe_targets_json": write_engine_probe_targets_json,
    "write_benchmark_job_plan_json": write_benchmark_job_plan_json,
    "write_benchmark_job_plan_shell": write_benchmark_job_plan_shell,
    "main": main,
}
_DOCUMENT_DEFAULTS = {
    name: value
    for name, value in vars(_document_module).items()
    if name != "__all__" and not name.startswith("__")
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
    return isinstance(value, _FunctionType) and value.__globals__ is vars(_document_module)


def _is_document_class(value: Any) -> bool:
    return isinstance(value, type) and value.__module__ == _document_module.__name__


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
