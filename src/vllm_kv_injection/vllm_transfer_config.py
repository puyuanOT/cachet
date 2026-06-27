"""Helpers for launching vLLM with the document KV transfer contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from document_kv_cache.engine_adapters import (
    ENGINE_ADAPTER_HANDOFF_RECORD_TYPE,
    ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION,
    vllm_adapter_spec,
)
from vllm_kv_injection.vllm_dynamic_connector import (
    DOCUMENT_KV_CONNECTOR_CLASS,
    DOCUMENT_KV_CONNECTOR_MODULE_PATH,
    DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY,
)
from vllm_kv_injection.vllm_native_provider_constants import (
    DOCUMENT_KV_HANDOFF_SOURCE_FACTORY_CONFIG_KEY,
    DOCUMENT_KV_NATIVE_PROVIDER_FACTORY,
    DOCUMENT_KV_PAYLOAD_CACHE_MAX_BYTES_CONFIG_KEY,
    DOCUMENT_KV_TELEMETRY_JSONL_CONFIG_KEY,
)

DOCUMENT_KV_TRANSFER_CONFIG_RECORD_TYPE = "vllm_kv_injection.document_kv_transfer_config.v1"
DOCUMENT_KV_TRANSFER_CONFIG_SCHEMA_VERSION = 1
DOCUMENT_KV_TRANSFER_CONFIG_PREFIX = "document_kv."
DOCUMENT_KV_DEFAULT_ROLE = "kv_both"

__all__ = [
    "DOCUMENT_KV_DEFAULT_ROLE",
    "DOCUMENT_KV_CONNECTOR_CLASS",
    "DOCUMENT_KV_CONNECTOR_MODULE_PATH",
    "DOCUMENT_KV_TRANSFER_CONFIG_PREFIX",
    "DOCUMENT_KV_TRANSFER_CONFIG_RECORD_TYPE",
    "DOCUMENT_KV_TRANSFER_CONFIG_SCHEMA_VERSION",
    "DOCUMENT_KV_NATIVE_PROVIDER_FACTORY",
    "DOCUMENT_KV_PAYLOAD_CACHE_MAX_BYTES_CONFIG_KEY",
    "DOCUMENT_KV_TELEMETRY_JSONL_CONFIG_KEY",
    "document_kv_transfer_config",
    "document_kv_transfer_config_json",
    "main",
]


def document_kv_transfer_config(
    *,
    kv_connector: str = DOCUMENT_KV_CONNECTOR_CLASS,
    kv_connector_module_path: str = DOCUMENT_KV_CONNECTOR_MODULE_PATH,
    kv_role: str = DOCUMENT_KV_DEFAULT_ROLE,
    extra_config: Mapping[str, Any] | None = None,
    provider_factory: str | None = DOCUMENT_KV_NATIVE_PROVIDER_FACTORY,
    handoff_source_factory: str | None = None,
    payload_cache_max_bytes: int | None = None,
    telemetry_jsonl: str | None = None,
) -> dict[str, Any]:
    """Return a vLLM ``KVTransferConfig``-shaped dictionary.

    The default connector is the package's dynamic vLLM V1 bridge, wired to
    the built-in runtime provider that reads Cachet handoffs from vLLM request
    ``kv_transfer_params``. Deployments can override ``provider_factory`` when
    they need a stronger engine-specific provider.
    """

    _validate_non_empty_string(kv_connector, field_name="kv_connector")
    _validate_non_empty_string(kv_connector_module_path, field_name="kv_connector_module_path")
    _validate_non_empty_string(kv_role, field_name="kv_role")
    config = {
        "kv_connector": kv_connector,
        "kv_connector_module_path": kv_connector_module_path,
        "kv_role": kv_role,
        "kv_connector_extra_config": _document_kv_extra_config(
            extra_config,
            provider_factory=provider_factory,
            handoff_source_factory=handoff_source_factory,
            payload_cache_max_bytes=payload_cache_max_bytes,
            telemetry_jsonl=telemetry_jsonl,
        ),
    }
    _validate_json_serializable(config, field_name="vLLM KV transfer config")
    return config


def document_kv_transfer_config_json(
    *,
    kv_connector: str = DOCUMENT_KV_CONNECTOR_CLASS,
    kv_connector_module_path: str = DOCUMENT_KV_CONNECTOR_MODULE_PATH,
    kv_role: str = DOCUMENT_KV_DEFAULT_ROLE,
    extra_config: Mapping[str, Any] | None = None,
    provider_factory: str | None = DOCUMENT_KV_NATIVE_PROVIDER_FACTORY,
    handoff_source_factory: str | None = None,
    payload_cache_max_bytes: int | None = None,
    telemetry_jsonl: str | None = None,
) -> str:
    """Return compact JSON for passing to vLLM CLI launch paths."""

    return json.dumps(
        document_kv_transfer_config(
            kv_connector=kv_connector,
            kv_connector_module_path=kv_connector_module_path,
            kv_role=kv_role,
            extra_config=extra_config,
            provider_factory=provider_factory,
            handoff_source_factory=handoff_source_factory,
            payload_cache_max_bytes=payload_cache_max_bytes,
            telemetry_jsonl=telemetry_jsonl,
        ),
        separators=(",", ":"),
        sort_keys=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a document KV vLLM KVTransferConfig sidecar.")
    parser.add_argument(
        "--kv-connector",
        default=DOCUMENT_KV_CONNECTOR_CLASS,
        help="vLLM connector class name. Defaults to the Document KV connector contract name.",
    )
    parser.add_argument(
        "--kv-connector-module-path",
        default=DOCUMENT_KV_CONNECTOR_MODULE_PATH,
        help="Import path for the vLLM connector module.",
    )
    parser.add_argument("--kv-role", default=DOCUMENT_KV_DEFAULT_ROLE)
    parser.add_argument(
        "--provider-factory",
        default=DOCUMENT_KV_NATIVE_PROVIDER_FACTORY,
        help="Optional module:attribute factory for the provider that loads materialized document KV.",
    )
    parser.add_argument(
        "--handoff-source-factory",
        help="Optional module:attribute factory for a custom document KV handoff source.",
    )
    parser.add_argument(
        "--payload-cache-max-bytes",
        type=int,
        help=(
            "Optional positive byte budget for the built-in vLLM provider's in-process "
            "payload URI cache. Omit or pass 0 to disable."
        ),
    )
    parser.add_argument(
        "--telemetry-jsonl",
        help="Optional local JSONL path where the built-in vLLM provider writes per-load telemetry.",
    )
    parser.add_argument(
        "--extra-config",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="Additional kv_connector_extra_config entry. Repeat as needed; values must be JSON.",
    )
    parser.add_argument("--output-json", help="Write the config to this path instead of stdout.")
    try:
        args = parser.parse_args(argv)
        config = document_kv_transfer_config(
            kv_connector=args.kv_connector,
            kv_connector_module_path=args.kv_connector_module_path,
            kv_role=args.kv_role,
            extra_config=_extra_config_from_cli(args.extra_config),
            provider_factory=args.provider_factory,
            handoff_source_factory=args.handoff_source_factory,
            payload_cache_max_bytes=args.payload_cache_max_bytes,
            telemetry_jsonl=args.telemetry_jsonl,
        )
        payload = json.dumps(config, indent=2, sort_keys=True) + "\n"
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, sort_keys=True))
        return 1
    return 0


def _document_kv_extra_config(
    extra_config: Mapping[str, Any] | None,
    *,
    provider_factory: str | None,
    handoff_source_factory: str | None,
    payload_cache_max_bytes: int | None,
    telemetry_jsonl: str | None,
) -> dict[str, Any]:
    spec = vllm_adapter_spec()
    merged: dict[str, Any] = {}
    if extra_config is not None:
        for key, value in extra_config.items():
            _validate_extra_config_key(key)
            merged[key] = value
    if provider_factory is not None:
        _validate_module_attribute(provider_factory, field_name="provider_factory")
        merged[DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY] = provider_factory
    if handoff_source_factory is not None:
        _validate_module_attribute(handoff_source_factory, field_name="handoff_source_factory")
        merged[DOCUMENT_KV_HANDOFF_SOURCE_FACTORY_CONFIG_KEY] = handoff_source_factory
    if payload_cache_max_bytes is not None:
        merged[DOCUMENT_KV_PAYLOAD_CACHE_MAX_BYTES_CONFIG_KEY] = _non_negative_int(
            payload_cache_max_bytes,
            field_name="payload_cache_max_bytes",
        )
    if telemetry_jsonl is not None:
        _validate_non_empty_string(telemetry_jsonl, field_name="telemetry_jsonl")
        merged[DOCUMENT_KV_TELEMETRY_JSONL_CONFIG_KEY] = telemetry_jsonl
    merged.update(
        {
            "document_kv.record_type": DOCUMENT_KV_TRANSFER_CONFIG_RECORD_TYPE,
            "document_kv.schema_version": DOCUMENT_KV_TRANSFER_CONFIG_SCHEMA_VERSION,
            "document_kv.backend": spec.backend.value,
            "document_kv.connector_package": spec.connector_package,
            "document_kv.kv_injection_method": spec.kv_injection_method,
            "document_kv.engine_handoff_record_type": ENGINE_ADAPTER_HANDOFF_RECORD_TYPE,
            "document_kv.engine_handoff_schema_version": ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION,
            "document_kv.requires_native_runtime": True,
        }
    )
    _validate_json_serializable(merged, field_name="kv_connector_extra_config")
    return merged


def _validate_extra_config_key(key: str) -> None:
    _validate_non_empty_string(key, field_name="extra_config key")
    if key.startswith(DOCUMENT_KV_TRANSFER_CONFIG_PREFIX):
        raise ValueError(f"extra_config keys starting with {DOCUMENT_KV_TRANSFER_CONFIG_PREFIX!r} are reserved")


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _extra_config_from_cli(values: list[str]) -> dict[str, Any]:
    extra_config: dict[str, Any] = {}
    for value in values:
        key, separator, raw_json = value.partition("=")
        _validate_non_empty_string(key, field_name="extra_config key")
        if not separator:
            raise ValueError("extra_config entries must use KEY=JSON")
        try:
            extra_config[key] = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"extra_config value for {key!r} must be valid JSON") from exc
    return extra_config


def _validate_non_empty_string(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_module_attribute(value: str, *, field_name: str) -> None:
    _validate_non_empty_string(value, field_name=field_name)
    module_name, separator, attribute_name = value.partition(":")
    if not separator or not module_name.strip() or not attribute_name.strip():
        raise ValueError(f"{field_name} must use module:attribute syntax")


def _validate_json_serializable(value: Any, *, field_name: str) -> None:
    try:
        json.dumps(value, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be JSON-serializable") from exc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
