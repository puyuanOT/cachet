"""Strict runtime preflight diagnostics for Cachet's SGLang native path."""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from document_kv_cache.engine_adapters import (
    ENGINE_ADAPTER_HANDOFF_RECORD_TYPE,
    ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION,
    sglang_adapter_spec,
)
from document_kv_cache.engine_launch_config import engine_launch_config_record_issues
from sglang_kv_injection.sglang_dynamic_backend import (
    DOCUMENT_KV_HICACHE_BACKEND_CLASS,
    DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH,
    DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY,
    DOCUMENT_KV_HICACHE_RUNTIME_METHODS,
    NoOpDocumentKVHiCacheProvider,
    load_document_kv_hicache_provider_factory,
)
from sglang_kv_injection.sglang_hicache_config import (
    DOCUMENT_KV_HICACHE_CONFIG_RECORD_TYPE,
    DOCUMENT_KV_HICACHE_CONFIG_SCHEMA_VERSION,
    sglang_hicache_launch_config,
)
from sglang_kv_injection.sglang_request_metadata_bridge import (
    DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_RECORD_TYPE,
    DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SCHEMA_VERSION,
    DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SOURCE,
    sglang_request_metadata_bridge_status_to_record,
)

DOCUMENT_KV_SGLANG_INSTALLED_HICACHE_CONTRACT_RECORD_TYPE = (
    "sglang_kv_injection.installed_hicache_contract.v1"
)
DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_RECORD_TYPE = "sglang_kv_injection.runtime_preflight.v1"
DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_SCHEMA_VERSION = 4
SGLANG_HICACHE_DYNAMIC_RUNTIME = "sglang-hicache-dynamic-storage"
SGLANG_HICACHE_DYNAMIC_BACKEND = "dynamic"
SGLANG_DOCUMENT_KV_HICACHE_BACKEND_NAME = "document_kv"
SGLANG_HICACHE_EXTRA_CONFIG_REQUIRED_FIELDS = (
    "backend_name",
    "module_path",
    "class_name",
)
SGLANG_HICACHE_REQUIRED_SERVER_ARG_FIELDS = (
    "enable_hierarchical_cache",
    "hicache_io_backend",
    "hicache_mem_layout",
    "hicache_storage_backend",
    "hicache_storage_backend_extra_config",
    "hicache_storage_prefetch_policy",
    "hicache_write_policy",
)
SGLANG_HICACHE_REQUIRED_CLI_OPTIONS = (
    "--enable-hierarchical-cache",
    "--hicache-io-backend",
    "--hicache-mem-layout",
    "--hicache-storage-backend",
    "--hicache-storage-backend-extra-config",
    "--hicache-storage-prefetch-policy",
    "--hicache-write-policy",
)
SGLANG_HICACHE_REQUIRED_STORAGE_BACKEND_FACTORY_METHODS = (
    "_create_dynamic_backend",
    "_load_backend_class",
    "create_backend",
)
SGLANG_HICACHE_REQUIRED_BACKEND_METHODS = DOCUMENT_KV_HICACHE_RUNTIME_METHODS
SGLANG_HICACHE_PROVIDER_REQUIRED_METHODS = ("get", "set", "exists")
SGLANG_DYNAMIC_BACKEND_FACTORY_RECORD_TYPE = "sglang_kv_injection.dynamic_backend_factory_preflight.v1"
SGLANG_HICACHE_REQUEST_METADATA_SOURCE_MODULES = (
    "sglang.srt.managers.cache_controller",
    "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller",
    "sglang.srt.mem_cache.hiradix_cache",
    "sglang.srt.mem_cache.hi_mamba_radix_cache",
)
SGLANG_REQUEST_CUSTOM_PARAMS_SOURCE_MODULES = (
    "sglang.srt.entrypoints.openai.protocol",
    "sglang.srt.entrypoints.openai.serving_chat",
    "sglang.srt.entrypoints.openai.serving_completions",
    "sglang.srt.managers.schedule_batch",
    "sglang.srt.sampling.sampling_params",
)

_INSTALLED_CONTRACT_KEYS = frozenset(
    {
        "record_type",
        "schema_version",
        "runtime",
        "package_version",
        "importable",
        "server_args_importable",
        "storage_backend_factory_importable",
        "hicache_storage_base_importable",
        "server_arg_fields",
        "cli_options",
        "hicache_storage_backend_choices",
        "hicache_storage_extra_info_fields",
        "storage_backend_factory_methods",
        "document_kv_backend_importable",
        "document_kv_backend_subclasses_hicache_storage",
        "document_kv_backend_methods",
        "request_custom_params_available",
        "request_metadata_extra_info_bridge",
        "request_metadata_bridge_sources",
        "live_request_metadata_bridge_ok",
        "error",
        "ok",
    }
)
_LAUNCH_CONFIG_RECORD_KEYS = frozenset(
    {
        "record_type",
        "schema_version",
        "enable_hierarchical_cache",
        "hicache_storage_backend",
        "backend_name",
        "module_path",
        "class_name",
        "provider_factory",
        "document_kv_record_type",
        "document_kv_schema_version",
        "document_kv_backend",
        "document_kv_connector_package",
        "document_kv_kv_injection_method",
        "document_kv_engine_handoff_record_type",
        "document_kv_engine_handoff_schema_version",
        "document_kv_requires_native_runtime",
        "extra_config_keys",
        "ok",
    }
)
_PROVIDER_FACTORY_RECORD_KEYS = frozenset(
    {
        "record_type",
        "schema_version",
        "path",
        "syntax_ok",
        "importable",
        "callable",
        "known_noop",
        "provider_constructed",
        "provider_class",
        "provider_methods",
        "provider_method_issues",
        "returns_known_noop",
        "error",
        "ok",
    }
)
_DYNAMIC_BACKEND_FACTORY_RECORD_KEYS = frozenset(
    {
        "record_type",
        "schema_version",
        "storage_backend",
        "backend_name",
        "module_path",
        "class_name",
        "provider_factory",
        "factory_importable",
        "create_backend_callable",
        "backend_constructed",
        "backend_class",
        "backend_methods",
        "backend_provider_class",
        "backend_provider_known_noop",
        "request_metadata_bridge",
        "error",
        "ok",
    }
)
_REQUEST_METADATA_BRIDGE_RECORD_KEYS = frozenset(
    {
        "record_type",
        "schema_version",
        "source",
        "installed",
        "scheduler_prefetch_patched",
        "controller_prefetch_patched",
        "controller_hash_tracking_patched",
        "prefetch_operation_patched",
        "hicache_storage_extra_info_factory_patched",
        "storage_hit_query_patched",
        "page_transfer_patched",
        "patched_modules",
        "error",
        "ok",
    }
)
_RUNTIME_PREFLIGHT_KEYS = frozenset(
    {
        "record_type",
        "schema_version",
        "runtime",
        "installed_contract",
        "launch_config",
        "provider_factory",
        "dynamic_backend_factory",
        "request_metadata_bridge",
        "live_request_metadata_bridge_ok",
        "ok",
    }
)

__all__ = [
    "DOCUMENT_KV_SGLANG_INSTALLED_HICACHE_CONTRACT_RECORD_TYPE",
    "DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_RECORD_TYPE",
    "DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SCHEMA_VERSION",
    "DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SOURCE",
    "DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_RECORD_TYPE",
    "DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_SCHEMA_VERSION",
    "SGLANG_HICACHE_DYNAMIC_RUNTIME",
    "SGLANG_HICACHE_DYNAMIC_BACKEND",
    "SGLANG_HICACHE_EXTRA_CONFIG_REQUIRED_FIELDS",
    "SGLANG_HICACHE_PROVIDER_REQUIRED_METHODS",
    "SGLANG_HICACHE_REQUIRED_BACKEND_METHODS",
    "SGLANG_HICACHE_REQUIRED_CLI_OPTIONS",
    "SGLANG_HICACHE_REQUIRED_SERVER_ARG_FIELDS",
    "SGLANG_HICACHE_REQUIRED_STORAGE_BACKEND_FACTORY_METHODS",
    "SGLANG_HICACHE_REQUEST_METADATA_SOURCE_MODULES",
    "SGLANG_REQUEST_CUSTOM_PARAMS_SOURCE_MODULES",
    "SGLANG_DOCUMENT_KV_HICACHE_BACKEND_NAME",
    "SGLANG_DYNAMIC_BACKEND_FACTORY_RECORD_TYPE",
    "SGLangInstalledHiCacheContract",
    "document_kv_sglang_runtime_preflight_record_issues",
    "document_kv_sglang_runtime_preflight_to_record",
    "installed_sglang_hicache_contract_record_issues",
    "installed_sglang_hicache_contract_to_record",
    "validate_document_kv_sglang_runtime_preflight_record",
    "validate_installed_sglang_hicache_contract_record",
    "write_document_kv_sglang_runtime_preflight_json",
    "main",
]


@dataclass(frozen=True, slots=True)
class SGLangInstalledHiCacheContract:
    """Installed SGLang HiCache dynamic-storage surface observed by preflight."""

    package_version: str | None
    importable: bool
    server_args_importable: bool
    storage_backend_factory_importable: bool
    hicache_storage_base_importable: bool
    server_arg_fields: tuple[str, ...] = ()
    cli_options: tuple[str, ...] = ()
    hicache_storage_backend_choices: tuple[str, ...] = ()
    hicache_storage_extra_info_fields: tuple[str, ...] = ()
    storage_backend_factory_methods: tuple[str, ...] = ()
    document_kv_backend_importable: bool = False
    document_kv_backend_subclasses_hicache_storage: bool = False
    document_kv_backend_methods: tuple[str, ...] = ()
    request_custom_params_available: bool = False
    request_metadata_extra_info_bridge: bool = False
    request_metadata_bridge_sources: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _SGLangPreflightStorageConfig:
    extra_config: Mapping[str, Any]


def installed_sglang_hicache_contract_to_record(
    contract: SGLangInstalledHiCacheContract | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize the installed SGLang HiCache dynamic-storage contract."""

    if contract is None:
        contract = _inspect_installed_sglang_hicache_contract()
    if isinstance(contract, Mapping):
        return _json_safe_mapping(contract, field_name="installed_contract")
    record = {
        "record_type": DOCUMENT_KV_SGLANG_INSTALLED_HICACHE_CONTRACT_RECORD_TYPE,
        "schema_version": DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
        "runtime": SGLANG_HICACHE_DYNAMIC_RUNTIME,
        "package_version": contract.package_version,
        "importable": contract.importable,
        "server_args_importable": contract.server_args_importable,
        "storage_backend_factory_importable": contract.storage_backend_factory_importable,
        "hicache_storage_base_importable": contract.hicache_storage_base_importable,
        "server_arg_fields": sorted(set(contract.server_arg_fields)),
        "cli_options": sorted(set(contract.cli_options)),
        "hicache_storage_backend_choices": sorted(set(contract.hicache_storage_backend_choices)),
        "hicache_storage_extra_info_fields": sorted(set(contract.hicache_storage_extra_info_fields)),
        "storage_backend_factory_methods": sorted(set(contract.storage_backend_factory_methods)),
        "document_kv_backend_importable": contract.document_kv_backend_importable,
        "document_kv_backend_subclasses_hicache_storage": contract.document_kv_backend_subclasses_hicache_storage,
        "document_kv_backend_methods": sorted(set(contract.document_kv_backend_methods)),
        "request_custom_params_available": contract.request_custom_params_available,
        "request_metadata_extra_info_bridge": contract.request_metadata_extra_info_bridge,
        "request_metadata_bridge_sources": sorted(set(contract.request_metadata_bridge_sources)),
    }
    record["live_request_metadata_bridge_ok"] = _live_request_metadata_bridge_ok(record)
    if contract.error:
        record["error"] = contract.error
    record["ok"] = _installed_sglang_hicache_contract_ok(record)
    return record


def validate_installed_sglang_hicache_contract_record(record: Mapping[str, Any]) -> None:
    """Raise when an installed SGLang HiCache contract record is unsafe."""

    issues = installed_sglang_hicache_contract_record_issues(record)
    if issues:
        raise ValueError("; ".join(issues))


def installed_sglang_hicache_contract_record_issues(record: object) -> tuple[str, ...]:
    """Return structural and safety issues for an installed SGLang contract."""

    if not isinstance(record, Mapping):
        return ("installed SGLang HiCache contract must be an object",)
    issues: list[str] = []
    unexpected = sorted(str(key) for key in record if key not in _INSTALLED_CONTRACT_KEYS)
    if unexpected:
        issues.append(f"installed SGLang HiCache contract has unsupported keys: {unexpected}")
    if record.get("record_type") != DOCUMENT_KV_SGLANG_INSTALLED_HICACHE_CONTRACT_RECORD_TYPE:
        issues.append(
            "installed SGLang HiCache contract record_type must be "
            f"{DOCUMENT_KV_SGLANG_INSTALLED_HICACHE_CONTRACT_RECORD_TYPE!r}"
        )
    if record.get("schema_version") != DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_SCHEMA_VERSION:
        issues.append(
            "installed SGLang HiCache contract schema_version must be "
            f"{DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_SCHEMA_VERSION}"
        )
    if record.get("runtime") != SGLANG_HICACHE_DYNAMIC_RUNTIME:
        issues.append(f"installed SGLang HiCache contract runtime must be {SGLANG_HICACHE_DYNAMIC_RUNTIME!r}")
    if not _non_empty_string(record.get("package_version")):
        issues.append("installed SGLang HiCache contract package_version must be a non-empty string")
    for field_name in (
        "importable",
        "server_args_importable",
        "storage_backend_factory_importable",
        "hicache_storage_base_importable",
        "document_kv_backend_importable",
        "document_kv_backend_subclasses_hicache_storage",
    ):
        if record.get(field_name) is not True:
            issues.append(f"installed SGLang HiCache contract {field_name} must be true")
    issues.extend(
        _required_string_items_issues(
            record.get("server_arg_fields"),
            required=SGLANG_HICACHE_REQUIRED_SERVER_ARG_FIELDS,
            field_name="installed SGLang HiCache contract server_arg_fields",
        )
    )
    issues.extend(
        _required_string_items_issues(
            record.get("cli_options"),
            required=SGLANG_HICACHE_REQUIRED_CLI_OPTIONS,
            field_name="installed SGLang HiCache contract cli_options",
        )
    )
    issues.extend(
        _required_string_items_issues(
            record.get("hicache_storage_backend_choices"),
            required=(SGLANG_HICACHE_DYNAMIC_BACKEND,),
            field_name="installed SGLang HiCache contract hicache_storage_backend_choices",
        )
    )
    issues.extend(
        _required_string_items_issues(
            record.get("hicache_storage_extra_info_fields"),
            required=("extra_info", "prefix_keys"),
            field_name="installed SGLang HiCache contract hicache_storage_extra_info_fields",
        )
    )
    issues.extend(
        _required_string_items_issues(
            record.get("storage_backend_factory_methods"),
            required=SGLANG_HICACHE_REQUIRED_STORAGE_BACKEND_FACTORY_METHODS,
            field_name="installed SGLang HiCache contract storage_backend_factory_methods",
        )
    )
    issues.extend(
        _required_string_items_issues(
            record.get("document_kv_backend_methods"),
            required=SGLANG_HICACHE_REQUIRED_BACKEND_METHODS,
            field_name="installed SGLang HiCache contract document_kv_backend_methods",
        )
    )
    for field_name in (
        "request_custom_params_available",
        "request_metadata_extra_info_bridge",
        "live_request_metadata_bridge_ok",
    ):
        if type(record.get(field_name)) is not bool:
            issues.append(f"installed SGLang HiCache contract {field_name} must be boolean")
    if record.get("request_custom_params_available") is not True:
        issues.append("installed SGLang HiCache contract request_custom_params_available must be true")
    bridge_sources = _string_tuple(record.get("request_metadata_bridge_sources"))
    if bridge_sources is None:
        issues.append("installed SGLang HiCache contract request_metadata_bridge_sources must be a string array")
        bridge_sources = ()
    if record.get("request_metadata_extra_info_bridge") is True and not bridge_sources:
        issues.append(
            "installed SGLang HiCache contract request_metadata_bridge_sources must identify the bridge source"
        )
    expected_bridge_ok = _live_request_metadata_bridge_ok(record)
    if type(record.get("live_request_metadata_bridge_ok")) is bool and (
        record.get("live_request_metadata_bridge_ok") != expected_bridge_ok
    ):
        issues.append(
            "installed SGLang HiCache contract live_request_metadata_bridge_ok must match "
            "extra_info field, custom_params, and request-metadata bridge detection"
        )
    if "error" in record and not isinstance(record.get("error"), str):
        issues.append("installed SGLang HiCache contract error must be a string when present")
    ok = record.get("ok")
    if type(ok) is not bool:
        issues.append("installed SGLang HiCache contract ok must be boolean")
    elif ok != _installed_sglang_hicache_contract_ok(record):
        issues.append("installed SGLang HiCache contract ok must match the inspected runtime surface")
    elif ok is False:
        issues.append("installed SGLang HiCache contract ok must be true for a safe runtime preflight")
    return tuple(issues)


def document_kv_sglang_runtime_preflight_to_record(
    launch_config: Mapping[str, Any] | None = None,
    *,
    installed_contract: SGLangInstalledHiCacheContract | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize the target-runtime gates required before a native SGLang probe."""

    if launch_config is None:
        launch_config = sglang_hicache_launch_config()
    launch_config = _json_safe_mapping(launch_config, field_name="launch_config")
    extra_config = _decode_hicache_extra_config(launch_config.get("hicache_storage_backend_extra_config")) or {}
    installed_contract_record = installed_sglang_hicache_contract_to_record(installed_contract)
    launch_config_record = _sglang_launch_config_to_record(launch_config)
    provider_factory_record = _provider_factory_to_record(
        launch_config_record.get("provider_factory"),
        extra_config=extra_config,
    )
    dynamic_backend_factory_record = _dynamic_backend_factory_to_record(
        launch_config_record,
        extra_config=extra_config,
    )
    request_metadata_bridge_record = _request_metadata_bridge_to_record(dynamic_backend_factory_record)
    live_request_metadata_bridge_ok = _runtime_live_request_metadata_bridge_ok(
        installed_contract_record,
        request_metadata_bridge_record,
    )
    ok = (
        installed_contract_record.get("ok") is True
        and launch_config_record.get("ok") is True
        and provider_factory_record.get("ok") is True
        and dynamic_backend_factory_record.get("ok") is True
        and live_request_metadata_bridge_ok
        and not installed_sglang_hicache_contract_record_issues(installed_contract_record)
        and not _sglang_launch_config_record_issues(launch_config_record)
        and not _provider_factory_record_issues(provider_factory_record)
        and not _dynamic_backend_factory_record_issues(dynamic_backend_factory_record)
        and not _request_metadata_bridge_record_issues(request_metadata_bridge_record, require_ok=False)
    )
    return {
        "record_type": DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_RECORD_TYPE,
        "schema_version": DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
        "runtime": SGLANG_HICACHE_DYNAMIC_RUNTIME,
        "installed_contract": installed_contract_record,
        "launch_config": launch_config_record,
        "provider_factory": provider_factory_record,
        "dynamic_backend_factory": dynamic_backend_factory_record,
        "request_metadata_bridge": request_metadata_bridge_record,
        "live_request_metadata_bridge_ok": live_request_metadata_bridge_ok,
        "ok": ok,
    }


def validate_document_kv_sglang_runtime_preflight_record(record: Mapping[str, Any]) -> None:
    """Raise when a serialized SGLang native-runtime preflight is unsafe."""

    issues = document_kv_sglang_runtime_preflight_record_issues(record)
    if issues:
        raise ValueError("; ".join(issues))


def document_kv_sglang_runtime_preflight_record_issues(record: object) -> tuple[str, ...]:
    """Return structural and safety issues for a SGLang native-runtime preflight."""

    if not isinstance(record, Mapping):
        return ("SGLang runtime preflight record must be an object",)
    issues: list[str] = []
    unexpected = sorted(str(key) for key in record if key not in _RUNTIME_PREFLIGHT_KEYS)
    if unexpected:
        issues.append(f"SGLang runtime preflight record has unsupported keys: {unexpected}")
    if record.get("record_type") != DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_RECORD_TYPE:
        issues.append(f"record_type must be {DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_RECORD_TYPE!r}")
    if record.get("schema_version") != DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_SCHEMA_VERSION:
        issues.append(f"schema_version must be {DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_SCHEMA_VERSION}")
    if record.get("runtime") != SGLANG_HICACHE_DYNAMIC_RUNTIME:
        issues.append(f"runtime must be {SGLANG_HICACHE_DYNAMIC_RUNTIME!r}")

    installed_contract = record.get("installed_contract")
    installed_safe = False
    if not isinstance(installed_contract, Mapping):
        issues.append("installed_contract must be an object")
    else:
        installed_issues = installed_sglang_hicache_contract_record_issues(installed_contract)
        issues.extend(f"installed_contract.{issue}" for issue in installed_issues)
        installed_safe = not installed_issues and installed_contract.get("ok") is True

    launch_config = record.get("launch_config")
    launch_safe = False
    if not isinstance(launch_config, Mapping):
        issues.append("launch_config must be an object")
    else:
        launch_issues = _sglang_launch_config_record_issues(launch_config)
        issues.extend(f"launch_config.{issue}" for issue in launch_issues)
        launch_safe = not launch_issues and launch_config.get("ok") is True

    provider_factory = record.get("provider_factory")
    provider_safe = False
    if not isinstance(provider_factory, Mapping):
        issues.append("provider_factory must be an object")
    else:
        provider_issues = _provider_factory_record_issues(provider_factory)
        issues.extend(f"provider_factory.{issue}" for issue in provider_issues)
        provider_safe = not provider_issues and provider_factory.get("ok") is True

    dynamic_backend_factory = record.get("dynamic_backend_factory")
    dynamic_backend_safe = False
    if not isinstance(dynamic_backend_factory, Mapping):
        issues.append("dynamic_backend_factory must be an object")
    else:
        dynamic_backend_issues = _dynamic_backend_factory_record_issues(dynamic_backend_factory)
        issues.extend(f"dynamic_backend_factory.{issue}" for issue in dynamic_backend_issues)
        dynamic_backend_safe = not dynamic_backend_issues and dynamic_backend_factory.get("ok") is True

    request_metadata_bridge = record.get("request_metadata_bridge")
    request_metadata_bridge_safe = False
    if not isinstance(request_metadata_bridge, Mapping):
        issues.append("request_metadata_bridge must be an object")
    else:
        request_metadata_bridge_issues = _request_metadata_bridge_record_issues(
            request_metadata_bridge,
            require_ok=False,
        )
        issues.extend(f"request_metadata_bridge.{issue}" for issue in request_metadata_bridge_issues)
        request_metadata_bridge_safe = not request_metadata_bridge_issues

    provider_matches_launch = False
    if isinstance(launch_config, Mapping) and isinstance(provider_factory, Mapping):
        provider_matches_launch = launch_config.get("provider_factory") == provider_factory.get("path")
        if not provider_matches_launch:
            issues.append("provider_factory.path must match launch_config.provider_factory")

    dynamic_backend_matches_launch = False
    if isinstance(launch_config, Mapping) and isinstance(dynamic_backend_factory, Mapping):
        dynamic_backend_matches_launch = (
            launch_config.get("provider_factory") == dynamic_backend_factory.get("provider_factory")
        )
        if not dynamic_backend_matches_launch:
            issues.append("dynamic_backend_factory.provider_factory must match launch_config.provider_factory")

    live_request_metadata_bridge_ok = record.get("live_request_metadata_bridge_ok")
    if type(live_request_metadata_bridge_ok) is not bool:
        issues.append("live_request_metadata_bridge_ok must be boolean")
    elif isinstance(installed_contract, Mapping) and isinstance(request_metadata_bridge, Mapping) and (
        live_request_metadata_bridge_ok
        != _runtime_live_request_metadata_bridge_ok(installed_contract, request_metadata_bridge)
    ):
        issues.append(
            "live_request_metadata_bridge_ok must match installed_contract and request_metadata_bridge readiness"
        )

    ok = record.get("ok")
    if type(ok) is not bool:
        issues.append("ok must be boolean")
    else:
        expected_ok = (
            installed_safe
            and launch_safe
            and provider_safe
            and dynamic_backend_safe
            and request_metadata_bridge_safe
            and provider_matches_launch
            and dynamic_backend_matches_launch
            and live_request_metadata_bridge_ok is True
        )
        if ok != expected_ok:
            issues.append(
                "ok must match installed contract, launch config, provider factory, "
                "dynamic backend factory safety, and live metadata bridge readiness"
            )
        if ok is False:
            issues.append("ok must be true for a safe SGLang runtime preflight")
    return tuple(issues)


def write_document_kv_sglang_runtime_preflight_json(
    path: str | Path,
    launch_config: Mapping[str, Any] | None = None,
    *,
    installed_contract: SGLangInstalledHiCacheContract | Mapping[str, Any] | None = None,
) -> None:
    """Write a strict SGLang native-runtime preflight record as JSON."""

    record = document_kv_sglang_runtime_preflight_to_record(
        launch_config,
        installed_contract=installed_contract,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _inspect_installed_sglang_hicache_contract() -> SGLangInstalledHiCacheContract:
    package_version: str | None = None
    try:
        package_version = importlib_metadata.version("sglang")
    except importlib_metadata.PackageNotFoundError:
        pass
    except Exception as exc:  # pragma: no cover - defensive for broken metadata.
        return _failed_installed_contract(package_version=package_version, error=f"{type(exc).__name__}: {exc}")

    try:
        importlib.import_module("sglang")
        server_args_module = importlib.import_module("sglang.srt.server_args")
        storage_backend_factory_module = importlib.import_module(
            "sglang.srt.mem_cache.storage.backend_factory"
        )
        hicache_storage_module = importlib.import_module("sglang.srt.mem_cache.hicache_storage")
        server_args_cls = getattr(server_args_module, "ServerArgs")
        storage_backend_factory_cls = getattr(storage_backend_factory_module, "StorageBackendFactory")
        hicache_storage_cls = getattr(hicache_storage_module, "HiCacheStorage")
    except Exception as exc:
        return _failed_installed_contract(package_version=package_version, error=f"{type(exc).__name__}: {exc}")

    document_kv_backend_importable = False
    document_kv_backend_subclasses_hicache_storage = False
    document_kv_backend_methods: tuple[str, ...] = ()
    document_kv_backend_error: str | None = None
    try:
        backend_module = importlib.import_module(DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH)
        backend_cls = getattr(backend_module, DOCUMENT_KV_HICACHE_BACKEND_CLASS)
        document_kv_backend_importable = True
        document_kv_backend_subclasses_hicache_storage = (
            isinstance(backend_cls, type)
            and isinstance(hicache_storage_cls, type)
            and issubclass(backend_cls, hicache_storage_cls)
        )
        document_kv_backend_methods = tuple(
            method_name
            for method_name in SGLANG_HICACHE_REQUIRED_BACKEND_METHODS
            if callable(getattr(backend_cls, method_name, None))
        )
    except Exception as exc:
        document_kv_backend_error = f"{type(exc).__name__}: {exc}"

    cli_options, backend_choices = _server_args_cli_surface(server_args_cls)
    request_metadata_bridge_sources = _request_metadata_bridge_sources()
    return SGLangInstalledHiCacheContract(
        package_version=package_version,
        importable=True,
        server_args_importable=True,
        storage_backend_factory_importable=True,
        hicache_storage_base_importable=True,
        server_arg_fields=tuple(
            field_name
            for field_name in SGLANG_HICACHE_REQUIRED_SERVER_ARG_FIELDS
            if hasattr(server_args_cls, field_name)
        ),
        cli_options=tuple(cli_options),
        hicache_storage_backend_choices=tuple(backend_choices),
        hicache_storage_extra_info_fields=_hicache_storage_extra_info_fields(hicache_storage_module),
        storage_backend_factory_methods=tuple(
            method_name
            for method_name in SGLANG_HICACHE_REQUIRED_STORAGE_BACKEND_FACTORY_METHODS
            if callable(getattr(storage_backend_factory_cls, method_name, None))
        ),
        document_kv_backend_importable=document_kv_backend_importable,
        document_kv_backend_subclasses_hicache_storage=document_kv_backend_subclasses_hicache_storage,
        document_kv_backend_methods=document_kv_backend_methods,
        request_custom_params_available=_request_custom_params_available(),
        request_metadata_extra_info_bridge=bool(request_metadata_bridge_sources),
        request_metadata_bridge_sources=request_metadata_bridge_sources,
        error=document_kv_backend_error,
    )


def _failed_installed_contract(*, package_version: str | None, error: str) -> SGLangInstalledHiCacheContract:
    return SGLangInstalledHiCacheContract(
        package_version=package_version,
        importable=False,
        server_args_importable=False,
        storage_backend_factory_importable=False,
        hicache_storage_base_importable=False,
        hicache_storage_extra_info_fields=(),
        document_kv_backend_importable=False,
        document_kv_backend_subclasses_hicache_storage=False,
        document_kv_backend_methods=(),
        request_custom_params_available=False,
        request_metadata_extra_info_bridge=False,
        request_metadata_bridge_sources=(),
        error=error,
    )


def _server_args_cli_surface(server_args_cls: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    parser = argparse.ArgumentParser(add_help=False, conflict_handler="resolve")
    add_cli_args = getattr(server_args_cls, "add_cli_args", None)
    if callable(add_cli_args):
        add_cli_args(parser)
    options: list[str] = []
    backend_choices: list[str] = []
    for action in parser._actions:
        options.extend(action.option_strings)
        if action.dest == "hicache_storage_backend" and action.choices is not None:
            backend_choices.extend(str(choice) for choice in action.choices)
    return tuple(options), tuple(backend_choices)


def _sglang_launch_config_to_record(launch_config: Mapping[str, Any]) -> dict[str, Any]:
    launch_config = _json_safe_mapping(launch_config, field_name="launch_config")
    extra_config = _decode_hicache_extra_config(launch_config.get("hicache_storage_backend_extra_config"))
    record: dict[str, Any] = {
        "record_type": "sglang_kv_injection.hicache_launch_config_preflight.v1",
        "schema_version": DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
        "enable_hierarchical_cache": launch_config.get("enable_hierarchical_cache"),
        "hicache_storage_backend": launch_config.get("hicache_storage_backend"),
        "extra_config_keys": sorted(str(key) for key in extra_config) if extra_config is not None else [],
    }
    if extra_config is not None:
        record.update(
            {
                "backend_name": extra_config.get("backend_name"),
                "module_path": extra_config.get("module_path"),
                "class_name": extra_config.get("class_name"),
                "provider_factory": extra_config.get(DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY),
                "document_kv_record_type": extra_config.get("document_kv.record_type"),
                "document_kv_schema_version": extra_config.get("document_kv.schema_version"),
                "document_kv_backend": extra_config.get("document_kv.backend"),
                "document_kv_connector_package": extra_config.get("document_kv.connector_package"),
                "document_kv_kv_injection_method": extra_config.get("document_kv.kv_injection_method"),
                "document_kv_engine_handoff_record_type": extra_config.get("document_kv.engine_handoff_record_type"),
                "document_kv_engine_handoff_schema_version": extra_config.get(
                    "document_kv.engine_handoff_schema_version"
                ),
                "document_kv_requires_native_runtime": extra_config.get("document_kv.requires_native_runtime"),
            }
        )
    record["ok"] = True
    record["ok"] = not _sglang_launch_config_record_issues(record) and not engine_launch_config_record_issues(
        launch_config,
        expected_backend="sglang",
    )
    return record


def _sglang_launch_config_record_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    unexpected = sorted(str(key) for key in record if key not in _LAUNCH_CONFIG_RECORD_KEYS)
    if unexpected:
        issues.append(f"SGLang launch config preflight has unsupported keys: {unexpected}")
    if record.get("record_type") != "sglang_kv_injection.hicache_launch_config_preflight.v1":
        issues.append("SGLang launch config preflight record_type is invalid")
    if record.get("schema_version") != DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_SCHEMA_VERSION:
        issues.append(
            "SGLang launch config preflight schema_version must be "
            f"{DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_SCHEMA_VERSION}"
        )
    if record.get("enable_hierarchical_cache") is not True:
        issues.append("SGLang launch config enable_hierarchical_cache must be true")
    if record.get("hicache_storage_backend") != SGLANG_HICACHE_DYNAMIC_BACKEND:
        issues.append(f"SGLang launch config hicache_storage_backend must be {SGLANG_HICACHE_DYNAMIC_BACKEND!r}")
    missing_fields = [
        field_name
        for field_name in SGLANG_HICACHE_EXTRA_CONFIG_REQUIRED_FIELDS
        if field_name not in record
    ]
    if missing_fields:
        issues.append("SGLang launch config is missing dynamic backend fields: " + ", ".join(missing_fields))
    if record.get("backend_name") != SGLANG_DOCUMENT_KV_HICACHE_BACKEND_NAME:
        issues.append(
            "SGLang launch config backend_name must be "
            f"{SGLANG_DOCUMENT_KV_HICACHE_BACKEND_NAME!r}"
        )
    if record.get("module_path") != DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH:
        issues.append(
            "SGLang launch config module_path must be "
            f"{DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH!r}"
        )
    if record.get("class_name") != DOCUMENT_KV_HICACHE_BACKEND_CLASS:
        issues.append(
            "SGLang launch config class_name must be "
            f"{DOCUMENT_KV_HICACHE_BACKEND_CLASS!r}"
        )
    if record.get("document_kv_record_type") != DOCUMENT_KV_HICACHE_CONFIG_RECORD_TYPE:
        issues.append(
            "SGLang launch config document_kv.record_type must be "
            f"{DOCUMENT_KV_HICACHE_CONFIG_RECORD_TYPE!r}"
        )
    if record.get("document_kv_schema_version") != DOCUMENT_KV_HICACHE_CONFIG_SCHEMA_VERSION:
        issues.append(
            "SGLang launch config document_kv.schema_version must be "
            f"{DOCUMENT_KV_HICACHE_CONFIG_SCHEMA_VERSION}"
        )
    spec = sglang_adapter_spec()
    if record.get("document_kv_backend") != "sglang":
        issues.append("SGLang launch config document_kv.backend must be 'sglang'")
    if record.get("document_kv_connector_package") != spec.connector_package:
        issues.append(
            "SGLang launch config document_kv.connector_package must be "
            f"{spec.connector_package!r}"
        )
    if record.get("document_kv_kv_injection_method") != spec.kv_injection_method:
        issues.append(
            "SGLang launch config document_kv.kv_injection_method must be "
            f"{spec.kv_injection_method!r}"
        )
    if record.get("document_kv_engine_handoff_record_type") != ENGINE_ADAPTER_HANDOFF_RECORD_TYPE:
        issues.append(
            "SGLang launch config document_kv.engine_handoff_record_type must be "
            f"{ENGINE_ADAPTER_HANDOFF_RECORD_TYPE!r}"
        )
    if record.get("document_kv_engine_handoff_schema_version") != ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION:
        issues.append(
            "SGLang launch config document_kv.engine_handoff_schema_version must be "
            f"{ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION}"
        )
    if record.get("document_kv_requires_native_runtime") is not True:
        issues.append("SGLang launch config document_kv.requires_native_runtime must be true")
    provider_factory = record.get("provider_factory")
    if not _non_empty_string(provider_factory):
        issues.append(
            "SGLang runtime preflight requires "
            f"{DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY} in hicache_storage_backend_extra_config"
        )
    elif not _module_attribute_path_ok(str(provider_factory)):
        issues.append(
            f"SGLang runtime preflight {DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY} "
            "must use module:attribute syntax without whitespace"
        )
    extra_config_keys = record.get("extra_config_keys")
    if _string_tuple(extra_config_keys) is None:
        issues.append("SGLang launch config extra_config_keys must be a string array")
    ok = record.get("ok")
    if type(ok) is not bool:
        issues.append("SGLang launch config preflight ok must be boolean")
    elif ok is False:
        issues.append("SGLang launch config preflight ok must be true")
    return tuple(issues)


def _provider_factory_to_record(
    factory_path: object,
    *,
    extra_config: Mapping[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "record_type": "sglang_kv_injection.provider_factory_preflight.v1",
        "schema_version": DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
        "path": factory_path,
        "syntax_ok": _module_attribute_path_ok(factory_path),
        "importable": False,
        "callable": False,
        "known_noop": False,
        "provider_constructed": False,
        "provider_class": None,
        "provider_methods": [],
        "provider_method_issues": [],
        "returns_known_noop": False,
    }
    if record["syntax_ok"]:
        try:
            factory = load_document_kv_hicache_provider_factory(str(factory_path))
            record["importable"] = True
            record["callable"] = callable(factory)
            record["known_noop"] = factory is NoOpDocumentKVHiCacheProvider
            if record["callable"] and not record["known_noop"]:
                provider = factory(extra_config=extra_config)
                record["provider_constructed"] = True
                record["provider_class"] = provider.__class__.__name__
                record["returns_known_noop"] = isinstance(provider, NoOpDocumentKVHiCacheProvider)
                record["provider_methods"] = list(_provider_method_names(provider))
                record["provider_method_issues"] = list(_provider_method_issues(provider))
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
    record["ok"] = (
        record["syntax_ok"]
        and record["importable"]
        and record["callable"]
        and not record["known_noop"]
        and record["provider_constructed"]
        and not record["returns_known_noop"]
        and not record["provider_method_issues"]
        and "error" not in record
    )
    return record


def _provider_factory_record_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    unexpected = sorted(str(key) for key in record if key not in _PROVIDER_FACTORY_RECORD_KEYS)
    if unexpected:
        issues.append(f"SGLang provider factory preflight has unsupported keys: {unexpected}")
    if record.get("record_type") != "sglang_kv_injection.provider_factory_preflight.v1":
        issues.append("SGLang provider factory preflight record_type is invalid")
    if record.get("schema_version") != DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_SCHEMA_VERSION:
        issues.append(
            "SGLang provider factory preflight schema_version must be "
            f"{DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_SCHEMA_VERSION}"
        )
    if not _module_attribute_path_ok(record.get("path")):
        issues.append("SGLang provider factory path must use module:attribute syntax without whitespace")
    for field_name in ("syntax_ok", "importable", "callable", "known_noop"):
        if type(record.get(field_name)) is not bool:
            issues.append(f"SGLang provider factory {field_name} must be boolean")
    if type(record.get("provider_constructed")) is not bool:
        issues.append("SGLang provider factory provider_constructed must be boolean")
    if "provider_class" not in record:
        issues.append("SGLang provider factory provider_class must be present")
    elif record.get("provider_class") is not None and not _non_empty_string(record.get("provider_class")):
        issues.append("SGLang provider factory provider_class must be null or a non-empty string")
    if _string_tuple(record.get("provider_methods")) is None:
        issues.append("SGLang provider factory provider_methods must be a string array")
    method_issues = _string_tuple(record.get("provider_method_issues"))
    if method_issues is None:
        issues.append("SGLang provider factory provider_method_issues must be a string array")
        method_issues = ()
    elif method_issues:
        issues.extend(f"SGLang provider factory provider_method_issues: {issue}" for issue in method_issues)
    if type(record.get("returns_known_noop")) is not bool:
        issues.append("SGLang provider factory returns_known_noop must be boolean")
    if record.get("syntax_ok") is not True:
        issues.append("SGLang provider factory syntax_ok must be true")
    if record.get("importable") is not True:
        issues.append("SGLang provider factory importable must be true")
    if record.get("callable") is not True:
        issues.append("SGLang provider factory callable must be true")
    if record.get("known_noop") is not False:
        issues.append("SGLang provider factory cannot be NoOpDocumentKVHiCacheProvider")
    if record.get("provider_constructed") is not True:
        issues.append("SGLang provider factory must construct a provider with extra_config")
    if record.get("returns_known_noop") is not False:
        issues.append("SGLang provider factory cannot return NoOpDocumentKVHiCacheProvider")
    if "error" in record and not isinstance(record.get("error"), str):
        issues.append("SGLang provider factory error must be a string when present")
    ok = record.get("ok")
    if type(ok) is not bool:
        issues.append("SGLang provider factory ok must be boolean")
    else:
        expected_ok = (
            record.get("syntax_ok") is True
            and record.get("importable") is True
            and record.get("callable") is True
            and record.get("known_noop") is False
            and record.get("provider_constructed") is True
            and record.get("returns_known_noop") is False
            and not method_issues
            and "error" not in record
        )
        if ok != expected_ok:
            issues.append("SGLang provider factory ok must match import, callable, construction, and no-op checks")
        if ok is False:
            issues.append("SGLang provider factory ok must be true for a safe runtime preflight")
    return tuple(issues)


def _dynamic_backend_factory_to_record(
    launch_config_record: Mapping[str, Any],
    *,
    extra_config: Mapping[str, Any],
) -> dict[str, Any]:
    provider_factory = extra_config.get(DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY)
    record: dict[str, Any] = {
        "record_type": SGLANG_DYNAMIC_BACKEND_FACTORY_RECORD_TYPE,
        "schema_version": DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
        "storage_backend": launch_config_record.get("hicache_storage_backend"),
        "backend_name": extra_config.get("backend_name"),
        "module_path": extra_config.get("module_path"),
        "class_name": extra_config.get("class_name"),
        "provider_factory": provider_factory,
        "factory_importable": False,
        "create_backend_callable": False,
        "backend_constructed": False,
        "backend_class": None,
        "backend_methods": [],
        "backend_provider_class": None,
        "backend_provider_known_noop": False,
    }
    try:
        backend_factory_module = importlib.import_module("sglang.srt.mem_cache.storage.backend_factory")
        backend_factory_cls = getattr(backend_factory_module, "StorageBackendFactory")
        record["factory_importable"] = True
        create_backend = getattr(backend_factory_cls, "create_backend", None)
        record["create_backend_callable"] = callable(create_backend)
        if callable(create_backend):
            storage_config = _SGLangPreflightStorageConfig(extra_config=dict(extra_config))
            backend = create_backend(launch_config_record.get("hicache_storage_backend"), storage_config, object())
            record["backend_constructed"] = True
            record["backend_class"] = backend.__class__.__name__
            record["backend_methods"] = list(_backend_method_names(backend))
            provider = getattr(backend, "provider", None)
            if provider is not None:
                record["backend_provider_class"] = provider.__class__.__name__
                record["backend_provider_known_noop"] = isinstance(provider, NoOpDocumentKVHiCacheProvider)
            record["request_metadata_bridge"] = sglang_request_metadata_bridge_status_to_record(
                getattr(backend, "request_metadata_bridge_status", None)
            )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    if "request_metadata_bridge" not in record:
        record["request_metadata_bridge"] = sglang_request_metadata_bridge_status_to_record()
    record["ok"] = (
        record["storage_backend"] == SGLANG_HICACHE_DYNAMIC_BACKEND
        and record["backend_name"] == SGLANG_DOCUMENT_KV_HICACHE_BACKEND_NAME
        and record["module_path"] == DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH
        and record["class_name"] == DOCUMENT_KV_HICACHE_BACKEND_CLASS
        and _module_attribute_path_ok(record["provider_factory"])
        and record["factory_importable"]
        and record["create_backend_callable"]
        and record["backend_constructed"]
        and record["backend_class"] == DOCUMENT_KV_HICACHE_BACKEND_CLASS
        and _contains_required_strings(record["backend_methods"], SGLANG_HICACHE_REQUIRED_BACKEND_METHODS)
        and _non_empty_string(record["backend_provider_class"])
        and not record["backend_provider_known_noop"]
        and "error" not in record
    )
    return record


def _dynamic_backend_factory_record_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    unexpected = sorted(str(key) for key in record if key not in _DYNAMIC_BACKEND_FACTORY_RECORD_KEYS)
    if unexpected:
        issues.append(f"SGLang dynamic backend factory preflight has unsupported keys: {unexpected}")
    if record.get("record_type") != SGLANG_DYNAMIC_BACKEND_FACTORY_RECORD_TYPE:
        issues.append("SGLang dynamic backend factory preflight record_type is invalid")
    if record.get("schema_version") != DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_SCHEMA_VERSION:
        issues.append(
            "SGLang dynamic backend factory preflight schema_version must be "
            f"{DOCUMENT_KV_SGLANG_RUNTIME_PREFLIGHT_SCHEMA_VERSION}"
        )
    if record.get("storage_backend") != SGLANG_HICACHE_DYNAMIC_BACKEND:
        issues.append(f"SGLang dynamic backend factory storage_backend must be {SGLANG_HICACHE_DYNAMIC_BACKEND!r}")
    if record.get("backend_name") != SGLANG_DOCUMENT_KV_HICACHE_BACKEND_NAME:
        issues.append(
            "SGLang dynamic backend factory backend_name must be "
            f"{SGLANG_DOCUMENT_KV_HICACHE_BACKEND_NAME!r}"
        )
    if record.get("module_path") != DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH:
        issues.append(
            "SGLang dynamic backend factory module_path must be "
            f"{DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH!r}"
        )
    if record.get("class_name") != DOCUMENT_KV_HICACHE_BACKEND_CLASS:
        issues.append(
            "SGLang dynamic backend factory class_name must be "
            f"{DOCUMENT_KV_HICACHE_BACKEND_CLASS!r}"
        )
    if not _module_attribute_path_ok(record.get("provider_factory")):
        issues.append("SGLang dynamic backend factory provider_factory must use module:attribute syntax")
    for field_name in (
        "factory_importable",
        "create_backend_callable",
        "backend_constructed",
        "backend_provider_known_noop",
    ):
        if type(record.get(field_name)) is not bool:
            issues.append(f"SGLang dynamic backend factory {field_name} must be boolean")
    if record.get("factory_importable") is not True:
        issues.append("SGLang dynamic backend factory StorageBackendFactory importable must be true")
    if record.get("create_backend_callable") is not True:
        issues.append("SGLang dynamic backend factory create_backend must be callable")
    if record.get("backend_constructed") is not True:
        issues.append("SGLang dynamic backend factory must construct the backend through create_backend")
    if record.get("backend_class") != DOCUMENT_KV_HICACHE_BACKEND_CLASS:
        issues.append(
            "SGLang dynamic backend factory backend_class must be "
            f"{DOCUMENT_KV_HICACHE_BACKEND_CLASS!r}"
        )
    issues.extend(
        _required_string_items_issues(
            record.get("backend_methods"),
            required=SGLANG_HICACHE_REQUIRED_BACKEND_METHODS,
            field_name="SGLang dynamic backend factory backend_methods",
        )
    )
    if not _non_empty_string(record.get("backend_provider_class")):
        issues.append("SGLang dynamic backend factory backend_provider_class must be a non-empty string")
    if record.get("backend_provider_known_noop") is not False:
        issues.append("SGLang dynamic backend factory backend provider cannot be NoOpDocumentKVHiCacheProvider")
    request_metadata_bridge = record.get("request_metadata_bridge")
    if not isinstance(request_metadata_bridge, Mapping):
        issues.append("SGLang dynamic backend factory request_metadata_bridge must be an object")
    else:
        issues.extend(
            "SGLang dynamic backend factory "
            f"request_metadata_bridge.{issue}"
            for issue in _request_metadata_bridge_record_issues(request_metadata_bridge, require_ok=False)
        )
    if "error" in record and not isinstance(record.get("error"), str):
        issues.append("SGLang dynamic backend factory error must be a string when present")
    ok = record.get("ok")
    if type(ok) is not bool:
        issues.append("SGLang dynamic backend factory ok must be boolean")
    else:
        expected_ok = (
            record.get("storage_backend") == SGLANG_HICACHE_DYNAMIC_BACKEND
            and record.get("backend_name") == SGLANG_DOCUMENT_KV_HICACHE_BACKEND_NAME
            and record.get("module_path") == DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH
            and record.get("class_name") == DOCUMENT_KV_HICACHE_BACKEND_CLASS
            and _module_attribute_path_ok(record.get("provider_factory"))
            and record.get("factory_importable") is True
            and record.get("create_backend_callable") is True
            and record.get("backend_constructed") is True
            and record.get("backend_class") == DOCUMENT_KV_HICACHE_BACKEND_CLASS
            and _contains_required_strings(record.get("backend_methods"), SGLANG_HICACHE_REQUIRED_BACKEND_METHODS)
            and _non_empty_string(record.get("backend_provider_class"))
            and record.get("backend_provider_known_noop") is False
            and "error" not in record
        )
        if ok != expected_ok:
            issues.append("SGLang dynamic backend factory ok must match construction and backend safety")
        if ok is False:
            issues.append("SGLang dynamic backend factory ok must be true for a safe runtime preflight")
    return tuple(issues)


def _request_metadata_bridge_to_record(
    dynamic_backend_factory_record: Mapping[str, Any],
) -> dict[str, Any]:
    return sglang_request_metadata_bridge_status_to_record(
        dynamic_backend_factory_record.get("request_metadata_bridge")
    )


def _request_metadata_bridge_record_issues(
    record: object,
    *,
    require_ok: bool,
) -> tuple[str, ...]:
    if not isinstance(record, Mapping):
        return ("SGLang request metadata bridge must be an object",)
    issues: list[str] = []
    unexpected = sorted(str(key) for key in record if key not in _REQUEST_METADATA_BRIDGE_RECORD_KEYS)
    if unexpected:
        issues.append(f"SGLang request metadata bridge has unsupported keys: {unexpected}")
    if record.get("record_type") != DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_RECORD_TYPE:
        issues.append("SGLang request metadata bridge record_type is invalid")
    if record.get("schema_version") != DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SCHEMA_VERSION:
        issues.append(
            "SGLang request metadata bridge schema_version must be "
            f"{DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SCHEMA_VERSION}"
        )
    if record.get("source") != DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SOURCE:
        issues.append(
            "SGLang request metadata bridge source must be "
            f"{DOCUMENT_KV_SGLANG_REQUEST_METADATA_BRIDGE_SOURCE!r}"
        )
    for field_name in (
        "installed",
        "scheduler_prefetch_patched",
        "controller_prefetch_patched",
        "controller_hash_tracking_patched",
        "prefetch_operation_patched",
        "hicache_storage_extra_info_factory_patched",
        "storage_hit_query_patched",
        "page_transfer_patched",
    ):
        if type(record.get(field_name)) is not bool:
            issues.append(f"SGLang request metadata bridge {field_name} must be boolean")
    patched_modules = _string_tuple(record.get("patched_modules"))
    if patched_modules is None:
        issues.append("SGLang request metadata bridge patched_modules must be a string array")
        patched_modules = ()
    if record.get("installed") is True and not patched_modules:
        issues.append("SGLang request metadata bridge patched_modules must identify installed patch modules")
    if "error" in record and not isinstance(record.get("error"), str):
        issues.append("SGLang request metadata bridge error must be a string when present")
    ok = record.get("ok")
    if type(ok) is not bool:
        issues.append("SGLang request metadata bridge ok must be boolean")
    else:
        expected_ok = _request_metadata_bridge_ok(record) and "error" not in record
        if ok != expected_ok:
            issues.append("SGLang request metadata bridge ok must match installed patch points")
        if require_ok and ok is False:
            issues.append("SGLang request metadata bridge ok must be true for live handoff metadata")
    return tuple(issues)


def _request_metadata_bridge_ok(record: object) -> bool:
    if not isinstance(record, Mapping):
        return False
    return (
        record.get("installed") is True
        and record.get("scheduler_prefetch_patched") is True
        and record.get("controller_prefetch_patched") is True
        and record.get("controller_hash_tracking_patched") is True
        and record.get("prefetch_operation_patched") is True
        and record.get("hicache_storage_extra_info_factory_patched") is True
        and record.get("storage_hit_query_patched") is True
        and record.get("page_transfer_patched") is True
    )


def _installed_sglang_hicache_contract_ok(record: Mapping[str, Any]) -> bool:
    return (
        _non_empty_string(record.get("package_version"))
        and record.get("importable") is True
        and record.get("server_args_importable") is True
        and record.get("storage_backend_factory_importable") is True
        and record.get("hicache_storage_base_importable") is True
        and _contains_required_strings(record.get("server_arg_fields"), SGLANG_HICACHE_REQUIRED_SERVER_ARG_FIELDS)
        and _contains_required_strings(record.get("cli_options"), SGLANG_HICACHE_REQUIRED_CLI_OPTIONS)
        and _contains_required_strings(record.get("hicache_storage_backend_choices"), (SGLANG_HICACHE_DYNAMIC_BACKEND,))
        and _contains_required_strings(record.get("hicache_storage_extra_info_fields"), ("extra_info", "prefix_keys"))
        and _contains_required_strings(
            record.get("storage_backend_factory_methods"),
            SGLANG_HICACHE_REQUIRED_STORAGE_BACKEND_FACTORY_METHODS,
        )
        and record.get("document_kv_backend_importable") is True
        and record.get("document_kv_backend_subclasses_hicache_storage") is True
        and _contains_required_strings(
            record.get("document_kv_backend_methods"),
            SGLANG_HICACHE_REQUIRED_BACKEND_METHODS,
        )
        and record.get("request_custom_params_available") is True
    )


def _live_request_metadata_bridge_ok(record: Mapping[str, Any]) -> bool:
    return (
        _contains_required_strings(record.get("hicache_storage_extra_info_fields"), ("extra_info",))
        and record.get("request_custom_params_available") is True
        and record.get("request_metadata_extra_info_bridge") is True
    )


def _runtime_live_request_metadata_bridge_ok(
    installed_contract_record: Mapping[str, Any],
    request_metadata_bridge_record: Mapping[str, Any],
) -> bool:
    return (
        _contains_required_strings(
            installed_contract_record.get("hicache_storage_extra_info_fields"),
            ("extra_info",),
        )
        and installed_contract_record.get("request_custom_params_available") is True
        and (
            installed_contract_record.get("request_metadata_extra_info_bridge") is True
            or _request_metadata_bridge_ok(request_metadata_bridge_record)
        )
    )


def _hicache_storage_extra_info_fields(hicache_storage_module: object) -> tuple[str, ...]:
    extra_info_cls = getattr(hicache_storage_module, "HiCacheStorageExtraInfo", None)
    dataclass_fields = getattr(extra_info_cls, "__dataclass_fields__", None)
    if isinstance(dataclass_fields, Mapping):
        return tuple(str(field_name) for field_name in dataclass_fields)
    annotations = getattr(extra_info_cls, "__annotations__", None)
    if isinstance(annotations, Mapping):
        return tuple(str(field_name) for field_name in annotations)
    return ()


def _request_custom_params_available() -> bool:
    sources = _module_sources(SGLANG_REQUEST_CUSTOM_PARAMS_SOURCE_MODULES)
    has_openai_custom_params = any("custom_params" in source for source in sources.values())
    has_request_custom_params = any("sampling_params.custom_params" in source for source in sources.values())
    return has_openai_custom_params and has_request_custom_params


def _request_metadata_bridge_sources() -> tuple[str, ...]:
    return tuple(
        module_name
        for module_name, source in _module_sources(SGLANG_HICACHE_REQUEST_METADATA_SOURCE_MODULES).items()
        if _source_contains_request_metadata_bridge(source)
    )


def _source_contains_request_metadata_bridge(source: str) -> bool:
    if "HiCacheStorageExtraInfo" not in source or "extra_info" not in source:
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        _hicache_extra_info_call_has_request_metadata(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    )


def _hicache_extra_info_call_has_request_metadata(node: ast.Call) -> bool:
    if _ast_call_name(node.func) != "HiCacheStorageExtraInfo":
        return False
    for keyword in node.keywords:
        if keyword.arg == "extra_info" and _ast_expression_mentions_request_metadata(keyword.value):
            return True
    return False


def _ast_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _ast_expression_mentions_request_metadata(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in {
            "custom_params",
            "kv_transfer_params",
            "request_metadata",
            "document_kv_request_metadata",
            "document_kv_extra_info",
        }:
            return True
        if isinstance(child, ast.Attribute) and _ast_attribute_is_request_metadata(child):
            return True
        if isinstance(child, ast.Constant) and isinstance(child.value, str) and _string_is_request_metadata_key(
            child.value
        ):
            return True
    return False


def _ast_attribute_is_request_metadata(node: ast.Attribute) -> bool:
    if node.attr in {"custom_params", "kv_transfer_params", "request_metadata"}:
        return True
    if node.attr != "extra_info":
        return False
    return _ast_root_name(node.value) in {"operation", "request", "req"}


def _ast_root_name(node: ast.AST) -> str | None:
    current = node
    while isinstance(current, ast.Attribute):
        current = current.value
    if isinstance(current, ast.Name):
        return current.id
    return None


def _string_is_request_metadata_key(value: str) -> bool:
    return value in {
        "custom_params",
        "kv_transfer_params",
        "request_metadata",
        "document_kv",
        "document_kv.request_id",
        "document_kv.handoff_json",
        "document_kv.handoff_record",
        "document_kv.payload_uri",
        "document_kv.prompt_text_mode",
    }


def _module_sources(module_names: Sequence[str]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for module_name in module_names:
        source = _module_source(module_name)
        if source is not None:
            sources[module_name] = source
    return sources


def _module_source(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        module = None
    if module is not None:
        try:
            return inspect.getsource(module)
        except (OSError, TypeError):
            pass
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError):
        return None
    origin = getattr(spec, "origin", None)
    if not origin or origin in {"built-in", "frozen"}:
        return None
    try:
        path = Path(origin)
    except TypeError:
        return None
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _provider_method_names(provider: object) -> tuple[str, ...]:
    method_names = [
        method_name
        for method_name in ("get", "set", "exists", "exist")
        if callable(getattr(provider, method_name, None))
    ]
    return tuple(method_names)


def _backend_method_names(backend: object) -> tuple[str, ...]:
    return tuple(
        method_name
        for method_name in SGLANG_HICACHE_REQUIRED_BACKEND_METHODS
        if callable(getattr(backend, method_name, None))
    )


def _provider_method_issues(provider: object) -> tuple[str, ...]:
    missing = [
        method_name
        for method_name in ("get", "set")
        if not callable(getattr(provider, method_name, None))
    ]
    if not (
        callable(getattr(provider, "exists", None))
        or callable(getattr(provider, "exist", None))
    ):
        missing.append("exists")
    if not missing:
        return ()
    return (
        "document KV HiCache provider must provide callable methods: "
        + ", ".join(missing)
        + "; exist is accepted as an alias for exists",
    )


def _decode_hicache_extra_config(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, Mapping):
        return None
    return decoded


def _read_launch_config_json(value: str | Path) -> dict[str, Any]:
    payload = _json_argument_or_file(value)
    if not isinstance(payload, Mapping):
        raise ValueError("SGLang launch config JSON must be an object")
    return _json_safe_mapping(payload, field_name="launch_config")


def _json_argument_or_file(value: str | Path) -> object:
    raw_value = str(value)
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return json.loads(Path(raw_value).read_text(encoding="utf-8"))


def _json_safe_mapping(value: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized = json.loads(json.dumps(value))
    if not isinstance(normalized, dict):
        raise TypeError(f"{field_name} must serialize to a JSON object")
    return normalized


def _required_string_items_issues(value: object, *, required: Sequence[str], field_name: str) -> tuple[str, ...]:
    items = _string_tuple(value)
    if items is None:
        return (f"{field_name} must be a string array",)
    missing = [item for item in required if item not in items]
    if missing:
        return (f"{field_name} is missing required entries: {', '.join(missing)}",)
    return ()


def _contains_required_strings(value: object, required: Sequence[str]) -> bool:
    items = _string_tuple(value)
    return items is not None and all(item in items for item in required)


def _string_tuple(value: object) -> tuple[str, ...] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = tuple(value)
        if all(isinstance(item, str) and item for item in items):
            return items
    return None


def _module_attribute_path_ok(value: object) -> bool:
    if not _non_empty_string(value):
        return False
    value = str(value)
    module_name, separator, attribute_name = value.partition(":")
    return bool(separator and module_name and attribute_name and not any(character.isspace() for character in value))


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write and validate the strict Cachet SGLang native-runtime preflight "
            "record required before provider-backed native probes."
        )
    )
    parser.add_argument(
        "--launch-config-json",
        help="JSON object or path to a SGLang HiCache launch-config sidecar.",
    )
    parser.add_argument(
        "--provider-factory",
        help=(
            "Cachet SGLang HiCache provider factory in module:attribute form. "
            "Used only when --launch-config-json is not supplied."
        ),
    )
    parser.add_argument("--output-json", help="Write the preflight record to this JSON file.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    launch_config = (
        _read_launch_config_json(args.launch_config_json)
        if args.launch_config_json
        else (
            sglang_hicache_launch_config(provider_factory=args.provider_factory)
            if args.provider_factory is not None
            else sglang_hicache_launch_config()
        )
    )
    record = document_kv_sglang_runtime_preflight_to_record(launch_config)
    output = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0 if not document_kv_sglang_runtime_preflight_record_issues(record) else 2


if __name__ == "__main__":
    raise SystemExit(main())
