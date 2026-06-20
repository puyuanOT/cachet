"""Compatibility facade for :mod:`document_kv_cache.benchmark_plan_executor`."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from threading import RLock as _RLock
from types import FunctionType as _FunctionType
from typing import Any

import document_kv_cache.benchmark_plan_executor as _document_module


def _load_document_defaults_module():
    module_path = Path(_document_module.__file__)
    module_name = "_restaurant_kv_serving_benchmark_plan_executor_document_defaults"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load document benchmark_plan_executor defaults from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_document_defaults_module = _load_document_defaults_module()
_DOCUMENT_DEFAULTS = {
    name: value
    for name, value in vars(_document_defaults_module).items()
    if name != "__all__" and not name.startswith("__")
}


for _name, _value in _DOCUMENT_DEFAULTS.items():
    globals()[_name] = _value


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
        return _dataclass_field_fingerprint(value)
    if name == "__dataclass_params__":
        return repr(value)
    if isinstance(value, property):
        return (
            "property",
            _function_fingerprint(value.fget),
            _function_fingerprint(value.fset),
            _function_fingerprint(value.fdel),
            value.__doc__,
        )
    if hasattr(value, "__objclass__") and hasattr(value, "__name__"):
        return (
            "descriptor",
            type(value).__qualname__,
            value.__name__,
            _descriptor_owner_fingerprint(value.__objclass__, owner),
        )
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    function_fingerprint = _function_fingerprint(value)
    if function_fingerprint is not None:
        return ("function", function_fingerprint)
    return value


def _descriptor_owner_fingerprint(objclass: type, owner: type) -> tuple[str, str] | tuple[str, str, str]:
    if objclass is owner:
        return ("self", owner.__qualname__)
    return ("foreign", objclass.__module__, objclass.__qualname__)


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
    return tuple((name, field.default, field.default_factory) for name, field in value.items())


def _public_class_default(name: str):
    if _is_pristine_public_class(name):
        return getattr(_document_module, name)
    return _DOCUMENT_DEFAULTS[name]


BenchmarkCommandResult = _public_class_default("BenchmarkCommandResult")


def execute_benchmark_job_plan(
    plan: Mapping[str, Any],
    *,
    dry_run: bool = False,
    cwd: str | Path | None = None,
) -> tuple[BenchmarkCommandResult, ...]:
    return _call_document_function("execute_benchmark_job_plan", plan, dry_run=dry_run, cwd=cwd)


def execute_benchmark_job_plan_json(
    path: str | Path,
    *,
    dry_run: bool = False,
    cwd: str | Path | None = None,
) -> tuple[BenchmarkCommandResult, ...]:
    return _call_document_function("execute_benchmark_job_plan_json", path, dry_run=dry_run, cwd=cwd)


def benchmark_command_results_to_record(
    results: Sequence[BenchmarkCommandResult],
    *,
    plan_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _call_document_function("benchmark_command_results_to_record", results, plan_source=plan_source)


def benchmark_plan_source_to_record(path: str | Path) -> dict[str, Any]:
    return _call_document_function("benchmark_plan_source_to_record", path)


def benchmark_plan_source_payload_to_record(path: str, driver_path: str | Path, payload: bytes) -> dict[str, Any]:
    return _call_document_function("benchmark_plan_source_payload_to_record", path, driver_path, payload)


def write_benchmark_command_results_json(
    results: Sequence[BenchmarkCommandResult],
    path: str | Path,
    *,
    plan_source: Mapping[str, Any] | None = None,
) -> None:
    return _call_document_function("write_benchmark_command_results_json", results, path, plan_source=plan_source)


def main(argv: Sequence[str] | None = None) -> int:
    return _call_document_function("main", argv)


_DEFAULT_COMPAT_FUNCTIONS = {
    "execute_benchmark_job_plan": execute_benchmark_job_plan,
    "execute_benchmark_job_plan_json": execute_benchmark_job_plan_json,
    "benchmark_command_results_to_record": benchmark_command_results_to_record,
    "benchmark_plan_source_to_record": benchmark_plan_source_to_record,
    "benchmark_plan_source_payload_to_record": benchmark_plan_source_payload_to_record,
    "write_benchmark_command_results_json": write_benchmark_command_results_json,
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
    if name == "BenchmarkCommandResult":
        return _public_class_default(name)
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
