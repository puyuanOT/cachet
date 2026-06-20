"""Compatibility wrapper for :mod:`document_kv_cache.engine_probe`."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import sys
from threading import RLock
from types import FunctionType, MappingProxyType
from typing import Any

from document_kv_cache._reexport import reexport_public
from restaurant_kv_serving.engine_adapters import (
    EngineKVBlockManagerProbe,
    EngineKVConnectorProbeResult,
    EngineKVInjectionPlan,
    ServingBackend,
    build_engine_kv_connector_actions,
    build_engine_kv_injection_plan,
    engine_kv_connector_probe_result_to_record,
    probe_engine_kv_connector_actions,
    read_engine_adapter_request_json,
    validate_engine_kv_connector_probe_record,
    view_engine_adapter_payload,
)
from restaurant_kv_serving.serving_env import serving_environment_profile
from restaurant_kv_serving.storage import local_path

import document_kv_cache.engine_probe as _document_module


def _load_document_defaults_module():
    module_path = Path(_document_module.__file__)
    module_name = "_restaurant_kv_serving_engine_probe_document_defaults"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load document engine_probe defaults from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_document_defaults_module = _load_document_defaults_module()
_DOCUMENT_DEFAULTS = {
    name: value
    for name, value in vars(_document_defaults_module).items()
    if not name.startswith("__")
}


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


def _public_class_base(name: str) -> type:
    if _is_pristine_public_class(name):
        return getattr(_document_module, name)
    return _DOCUMENT_DEFAULTS[name]


__all__ = reexport_public(
    "document_kv_cache.engine_probe",
    (
        "ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND",
        "ENGINE_KV_PROBE_METADATA_HANDOFF_JSON",
        "ENGINE_KV_PROBE_METADATA_PAYLOAD_URI",
        "ENGINE_KV_PROBE_METADATA_PROBE_FACTORY",
        "ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE",
        "ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION",
    ),
    globals(),
)

ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND = _document_defaults_module.ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND
ENGINE_KV_PROBE_METADATA_HANDOFF_JSON = _document_defaults_module.ENGINE_KV_PROBE_METADATA_HANDOFF_JSON
ENGINE_KV_PROBE_METADATA_PAYLOAD_URI = _document_defaults_module.ENGINE_KV_PROBE_METADATA_PAYLOAD_URI
ENGINE_KV_PROBE_METADATA_PROBE_FACTORY = _document_defaults_module.ENGINE_KV_PROBE_METADATA_PROBE_FACTORY
ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE = (
    _document_defaults_module.ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE
)
ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION = (
    _document_defaults_module.ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION
)

__all__ += [
    "EngineKVProbeFactoryContext",
    "EngineKVProbeFactoryResult",
    "EngineKVProbeFactory",
    "EngineKVProbeConfig",
    "run_engine_kv_connector_probe",
    "read_engine_adapter_payload",
    "write_engine_kv_connector_actions_record_json",
    "write_engine_kv_connector_probe_result_json",
    "load_engine_kv_probe_factory",
    "parse_args",
    "main",
    "argparse",
    "importlib",
    "json",
    "Callable",
    "Mapping",
    "Sequence",
    "dataclass",
    "field",
    "Path",
    "MappingProxyType",
    "Any",
    "EngineKVBlockManagerProbe",
    "EngineKVConnectorProbeResult",
    "EngineKVInjectionPlan",
    "ServingBackend",
    "build_engine_kv_connector_actions",
    "build_engine_kv_injection_plan",
    "engine_kv_connector_probe_result_to_record",
    "probe_engine_kv_connector_actions",
    "read_engine_adapter_request_json",
    "validate_engine_kv_connector_probe_record",
    "view_engine_adapter_payload",
    "serving_environment_profile",
    "local_path",
]

EngineKVProbeFactory = _document_module.EngineKVProbeFactory


class EngineKVProbeFactoryContext(_public_class_base("EngineKVProbeFactoryContext")):
    __slots__ = ()

    def __post_init__(self) -> None:
        _validate_metadata_strings(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class EngineKVProbeFactoryResult(_public_class_base("EngineKVProbeFactoryResult")):
    __slots__ = ()

    def __post_init__(self) -> None:
        if not self.engine_version:
            raise ValueError("engine_version must be non-empty")
        _validate_metadata_strings(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class EngineKVProbeConfig(_public_class_base("EngineKVProbeConfig")):
    __slots__ = ()

    def __post_init__(self) -> None:
        if not self.probe_factory:
            raise ValueError("probe_factory must be non-empty")
        _validate_metadata_strings(self.metadata)
        object.__setattr__(self, "handoff_json", Path(self.handoff_json))
        if self.output_json is not None:
            object.__setattr__(self, "output_json", Path(self.output_json))
        if self.actions_output_json is not None:
            object.__setattr__(self, "actions_output_json", Path(self.actions_output_json))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def run_engine_kv_connector_probe(config: EngineKVProbeConfig) -> EngineKVConnectorProbeResult:
    return _call_document_function("run_engine_kv_connector_probe", config)


def read_engine_adapter_payload(payload_uri: str, *, expected_bytes: int | None = None) -> bytes:
    return _call_document_function("read_engine_adapter_payload", payload_uri, expected_bytes=expected_bytes)


def write_engine_kv_connector_probe_result_json(
    result: EngineKVConnectorProbeResult,
    path: str | Path,
) -> None:
    return _call_document_function("write_engine_kv_connector_probe_result_json", result, path)


def write_engine_kv_connector_actions_record_json(
    actions,
    path: str | Path,
) -> None:
    return _call_document_function("write_engine_kv_connector_actions_record_json", actions, path)


def load_engine_kv_probe_factory(factory_path: str) -> EngineKVProbeFactory:
    return _call_document_function("load_engine_kv_probe_factory", factory_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _call_document_function("parse_args", argv)


def main(argv: Sequence[str] | None = None) -> int:
    return _call_document_function("main", argv)


def _split_factory_path(factory_path: str) -> tuple[str, str]:
    return _call_document_function("_split_factory_path", factory_path)


def _validate_local_payload_uri(payload_uri: str) -> None:
    return _call_document_function("_validate_local_payload_uri", payload_uri)


def _parse_metadata_items(items: Sequence[str]) -> dict[str, str]:
    return _call_document_function("_parse_metadata_items", items)


def _probe_trace_metadata(
    config: EngineKVProbeConfig,
    *,
    payload_uri: str,
    backend: ServingBackend,
) -> dict[str, str]:
    return _call_document_function("_probe_trace_metadata", config, payload_uri=payload_uri, backend=backend)


def _validate_metadata_strings(metadata: Mapping[str, str]) -> None:
    return _call_document_function("_validate_metadata_strings", metadata)


_DEFAULT_COMPAT_FUNCTIONS = {
    "run_engine_kv_connector_probe": run_engine_kv_connector_probe,
    "read_engine_adapter_payload": read_engine_adapter_payload,
    "write_engine_kv_connector_actions_record_json": write_engine_kv_connector_actions_record_json,
    "write_engine_kv_connector_probe_result_json": write_engine_kv_connector_probe_result_json,
    "load_engine_kv_probe_factory": load_engine_kv_probe_factory,
    "parse_args": parse_args,
    "main": main,
    "_split_factory_path": _split_factory_path,
    "_validate_local_payload_uri": _validate_local_payload_uri,
    "_parse_metadata_items": _parse_metadata_items,
    "_probe_trace_metadata": _probe_trace_metadata,
    "_validate_metadata_strings": _validate_metadata_strings,
}
_PATCH_LOCK = RLock()
_LEGACY_PATCH_NAMES = tuple(name for name in _DOCUMENT_DEFAULTS if name in globals())


def _call_document_function(name: str, *args, **kwargs):
    with _PATCH_LOCK:
        return _isolated_document_namespace()[name](*args, **kwargs)


def _document_global_for_legacy(name: str):
    if name not in globals():
        return _DOCUMENT_DEFAULTS[name]
    if name == "EngineKVProbeFactoryResult":
        return _legacy_class_base_for_document_namespace(name)
    current = globals()[name]
    if _DEFAULT_COMPAT_FUNCTIONS.get(name) is current:
        return _DOCUMENT_DEFAULTS[name]
    return current


def _legacy_class_base_for_document_namespace(name: str) -> type:
    current = globals().get(name)
    if isinstance(current, type):
        for base in current.__mro__[1:]:
            if base is not object:
                return base
        return current
    return _public_class_base(name)


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
