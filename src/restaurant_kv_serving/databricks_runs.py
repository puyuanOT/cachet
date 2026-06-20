"""Compatibility wrapper for :mod:`document_kv_cache.databricks_runs`."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util as _importlib_util
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys as _sys
from threading import RLock
from types import FunctionType
from typing import Any, Protocol
import urllib.error
import urllib.parse
import urllib.request

from document_kv_cache._reexport import reexport_public

import document_kv_cache.databricks_runs as _document_module


def _load_document_defaults_module():
    module_path = Path(_document_module.__file__)
    module_name = "_restaurant_kv_serving_databricks_runs_document_defaults"
    spec = _importlib_util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load document databricks_runs defaults from {module_path}")
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

__all__ = reexport_public(
    "document_kv_cache.databricks_runs",
    (
        "DEFAULT_DATABRICKS_HOST_ENV",
        "DEFAULT_DATABRICKS_TOKEN_ENV",
        "DEFAULT_DATABRICKS_TIMEOUT_SECONDS",
        "DATABRICKS_RUN_STATUS_RECORD_TYPE",
        "DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE",
    ),
    globals(),
)

__all__ += [
    "DATABRICKS_TERMINAL_LIFE_CYCLE_STATES",
    "DatabricksHTTPResponse",
    "DatabricksURLOpener",
    "DatabricksWorkspaceConfig",
    "databricks_workspace_config_from_env",
    "submit_databricks_run",
    "get_databricks_run",
    "summarize_databricks_run",
    "summarize_databricks_run_submit_payload",
    "databricks_run_status_record",
    "databricks_run_status_sidecar_issues",
    "validate_databricks_run_status_sidecar",
    "write_databricks_run_response_json",
    "read_databricks_run_submit_payload",
    "main",
    "argparse",
    "hashlib",
    "Mapping",
    "Sequence",
    "dataclass",
    "field",
    "json",
    "os",
    "Path",
    "Any",
    "Protocol",
    "urllib",
]

_PUBLIC_EXPORT_NAMES = frozenset(
    {
        "DEFAULT_DATABRICKS_HOST_ENV",
        "DEFAULT_DATABRICKS_TOKEN_ENV",
        "DEFAULT_DATABRICKS_TIMEOUT_SECONDS",
        "DATABRICKS_RUN_STATUS_RECORD_TYPE",
        "DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE",
        "DATABRICKS_TERMINAL_LIFE_CYCLE_STATES",
        "DatabricksHTTPResponse",
        "DatabricksURLOpener",
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
        if name not in {"__doc__", "__module__", "_abc_impl"}
    )


def _class_attribute_fingerprint(name: str, value: Any) -> Any:
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


def _public_class_base(name: str):
    if _is_pristine_public_class(name):
        return getattr(_document_module, name)
    return _DOCUMENT_DEFAULTS[name]


def _public_export_default(name: str) -> Any:
    live_value = getattr(_document_module, name)
    default_value = _DOCUMENT_DEFAULTS[name]
    if isinstance(live_value, type) and _is_pristine_public_class(name):
        return live_value
    return live_value if live_value == default_value else default_value


for _name in _PUBLIC_EXPORT_NAMES:
    globals()[_name] = _public_export_default(_name)


class DatabricksWorkspaceConfig(_public_class_base("DatabricksWorkspaceConfig")):
    __slots__ = ()

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("host must be non-empty")
        if not self.token:
            raise ValueError("token must be non-empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


def databricks_workspace_config_from_env(
    *,
    host_env: str = DEFAULT_DATABRICKS_HOST_ENV,
    token_env: str = DEFAULT_DATABRICKS_TOKEN_ENV,
    timeout_seconds: float = DEFAULT_DATABRICKS_TIMEOUT_SECONDS,
    environ: dict[str, str] | None = None,
) -> DatabricksWorkspaceConfig:
    return _call_document_function(
        "databricks_workspace_config_from_env",
        host_env=host_env,
        token_env=token_env,
        timeout_seconds=timeout_seconds,
        environ=environ,
    )


def submit_databricks_run(
    config: DatabricksWorkspaceConfig,
    payload: dict[str, Any],
    *,
    opener: DatabricksURLOpener = urllib.request.urlopen,
) -> dict[str, Any]:
    return _call_document_function("submit_databricks_run", config, payload, opener=opener)


def get_databricks_run(
    config: DatabricksWorkspaceConfig,
    run_id: int | str,
    *,
    opener: DatabricksURLOpener = urllib.request.urlopen,
) -> dict[str, Any]:
    return _call_document_function("get_databricks_run", config, run_id, opener=opener)


def write_databricks_run_response_json(response: dict[str, Any], path: str | Path) -> None:
    return _call_document_function("write_databricks_run_response_json", response, path)


def read_databricks_run_submit_payload(path: str | Path) -> dict[str, Any]:
    return _call_document_function("read_databricks_run_submit_payload", path)


def summarize_databricks_run(
    run: dict[str, Any],
    *,
    submit_payload: Mapping[str, Any] | None = None,
    submit_payload_path: str | None = None,
) -> dict[str, Any]:
    return _call_document_function(
        "summarize_databricks_run",
        run,
        submit_payload=submit_payload,
        submit_payload_path=submit_payload_path,
    )


def summarize_databricks_run_submit_payload(
    payload: Mapping[str, Any],
    *,
    source_path: str | None = None,
) -> dict[str, Any]:
    return _call_document_function(
        "summarize_databricks_run_submit_payload",
        payload,
        source_path=source_path,
    )


def databricks_run_status_record(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    return _call_document_function("databricks_run_status_record", record)


def databricks_run_status_sidecar_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    return _call_document_function("databricks_run_status_sidecar_issues", record)


def validate_databricks_run_status_sidecar(record: Mapping[str, Any]) -> None:
    return _call_document_function("validate_databricks_run_status_sidecar", record)


def _databricks_api_json(
    config: DatabricksWorkspaceConfig,
    method: str,
    path_and_query: str,
    *,
    opener: DatabricksURLOpener,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _call_document_function(
        "_databricks_api_json",
        config,
        method,
        path_and_query,
        opener=opener,
        payload=payload,
    )


def _task_summary(task: Mapping[str, Any]) -> dict[str, Any]:
    return _call_document_function("_task_summary", task)


def _submit_payload_task_summary(task: Mapping[str, Any]) -> dict[str, Any]:
    return _call_document_function("_submit_payload_task_summary", task)


def _active_task_key(tasks: Sequence[Mapping[str, Any]]) -> str | None:
    return _call_document_function("_active_task_key", tasks)


def _cluster_id(record: Mapping[str, Any]) -> str | None:
    return _call_document_function("_cluster_id", record)


def _mapping(value: Any) -> Mapping[str, Any]:
    return _call_document_function("_mapping", value)


def _sequence_of_mappings(value: Any) -> tuple[Mapping[str, Any], ...]:
    return _call_document_function("_sequence_of_mappings", value)


def _optional_str(value: Any) -> str | None:
    return _call_document_function("_optional_str", value)


def _sorted_unique_texts(values: Sequence[Any]) -> list[str]:
    return _call_document_function("_sorted_unique_texts", values)


def _is_aws_g5_node_type(value: Any) -> bool:
    return _call_document_function("_is_aws_g5_node_type", value)


def _sha256_hex(payload: bytes) -> str:
    return _call_document_function("_sha256_hex", payload)


def _databricks_request(
    config: DatabricksWorkspaceConfig,
    method: str,
    path_and_query: str,
    *,
    payload: dict[str, Any] | None,
) -> urllib.request.Request:
    return _call_document_function("_databricks_request", config, method, path_and_query, payload=payload)


def _format_databricks_http_error(status_code: int, body: str, *, token: str | None = None) -> str:
    return _call_document_function("_format_databricks_http_error", status_code, body, token=token)


def _redact_databricks_secret_text(text: str, *, token: str | None = None) -> str:
    return _call_document_function("_redact_databricks_secret_text", text, token=token)


def _success_record(action: str, response: dict[str, Any] | None = None) -> dict[str, Any]:
    return _call_document_function("_success_record", action, response)


def _write_error_record_or_stdout(result: dict[str, Any], output_json: str | None) -> None:
    return _call_document_function("_write_error_record_or_stdout", result, output_json)


def main(argv: Sequence[str] | None = None) -> int:
    return _call_document_function("main", argv)


_DEFAULT_COMPAT_FUNCTIONS = {
    "databricks_workspace_config_from_env": databricks_workspace_config_from_env,
    "submit_databricks_run": submit_databricks_run,
    "get_databricks_run": get_databricks_run,
    "write_databricks_run_response_json": write_databricks_run_response_json,
    "read_databricks_run_submit_payload": read_databricks_run_submit_payload,
    "summarize_databricks_run": summarize_databricks_run,
    "summarize_databricks_run_submit_payload": summarize_databricks_run_submit_payload,
    "databricks_run_status_record": databricks_run_status_record,
    "databricks_run_status_sidecar_issues": databricks_run_status_sidecar_issues,
    "validate_databricks_run_status_sidecar": validate_databricks_run_status_sidecar,
    "_databricks_api_json": _databricks_api_json,
    "_task_summary": _task_summary,
    "_submit_payload_task_summary": _submit_payload_task_summary,
    "_active_task_key": _active_task_key,
    "_cluster_id": _cluster_id,
    "_mapping": _mapping,
    "_sequence_of_mappings": _sequence_of_mappings,
    "_optional_str": _optional_str,
    "_sorted_unique_texts": _sorted_unique_texts,
    "_is_aws_g5_node_type": _is_aws_g5_node_type,
    "_sha256_hex": _sha256_hex,
    "_databricks_request": _databricks_request,
    "_format_databricks_http_error": _format_databricks_http_error,
    "_redact_databricks_secret_text": _redact_databricks_secret_text,
    "_success_record": _success_record,
    "_write_error_record_or_stdout": _write_error_record_or_stdout,
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
