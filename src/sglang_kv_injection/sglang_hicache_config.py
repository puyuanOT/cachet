"""Helpers for launching SGLang HiCache with a document KV backend contract."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from document_kv_cache.engine_adapters import (
    ENGINE_ADAPTER_HANDOFF_RECORD_TYPE,
    ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION,
    sglang_adapter_spec,
)
from sglang_kv_injection.sglang_dynamic_backend import (
    DOCUMENT_KV_HICACHE_BACKEND_CLASS,
    DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH,
    DOCUMENT_KV_HICACHE_PROVIDER_FACTORY,
    DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY,
)

DOCUMENT_KV_HICACHE_CONFIG_RECORD_TYPE = "sglang_kv_injection.document_kv_hicache_config.v1"
DOCUMENT_KV_HICACHE_CONFIG_SCHEMA_VERSION = 1
DOCUMENT_KV_HICACHE_CONFIG_PREFIX = "document_kv."
SGLANG_HICACHE_DYNAMIC_BACKEND = "dynamic"
_SGLANG_DYNAMIC_BACKEND_IDENTITY_KEYS = frozenset({"backend_name", "module_path", "class_name"})

__all__ = [
    "DOCUMENT_KV_HICACHE_CONFIG_PREFIX",
    "DOCUMENT_KV_HICACHE_CONFIG_RECORD_TYPE",
    "DOCUMENT_KV_HICACHE_CONFIG_SCHEMA_VERSION",
    "DOCUMENT_KV_HICACHE_PROVIDER_FACTORY",
    "DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY",
    "SGLANG_HICACHE_DYNAMIC_BACKEND",
    "main",
    "sglang_hicache_cli_args",
    "sglang_hicache_launch_config",
]


def sglang_hicache_launch_config(
    *,
    backend_name: str = "document_kv",
    module_path: str = DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH,
    class_name: str = DOCUMENT_KV_HICACHE_BACKEND_CLASS,
    provider_factory: str | None = DOCUMENT_KV_HICACHE_PROVIDER_FACTORY,
    extra_config: Mapping[str, Any] | None = None,
    page_size: int | None = None,
    hicache_ratio: float | None = None,
    hicache_size_gb: int | None = None,
    hicache_io_backend: str | None = None,
    hicache_mem_layout: str | None = None,
    hicache_storage_prefetch_policy: str | None = None,
    hicache_write_policy: str | None = None,
) -> dict[str, Any]:
    """Return SGLang launch-server arguments for a dynamic HiCache backend.

    SGLang owns RadixAttention, HiCache tiering, eviction, prefetch, and cache
    storage execution. This helper only builds a launch-time configuration that
    points a patched runtime at a document-KV-aware dynamic storage backend.
    """

    config: dict[str, Any] = {
        "enable_hierarchical_cache": True,
        "hicache_storage_backend": SGLANG_HICACHE_DYNAMIC_BACKEND,
        "hicache_storage_backend_extra_config": json.dumps(
            _document_kv_hicache_extra_config(
                backend_name=backend_name,
                module_path=module_path,
                class_name=class_name,
                provider_factory=provider_factory,
                extra_config=extra_config,
            ),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    }
    _add_positive_int(config, "page_size", page_size)
    _add_positive_number(config, "hicache_ratio", hicache_ratio)
    _add_non_negative_int(config, "hicache_size", hicache_size_gb)
    _add_optional_string(config, "hicache_io_backend", hicache_io_backend)
    _add_optional_string(config, "hicache_mem_layout", hicache_mem_layout)
    _add_optional_string(config, "hicache_storage_prefetch_policy", hicache_storage_prefetch_policy)
    _add_optional_string(config, "hicache_write_policy", hicache_write_policy)
    _validate_json_serializable(config, field_name="SGLang HiCache launch config")
    return config


def sglang_hicache_cli_args(
    *,
    backend_name: str = "document_kv",
    module_path: str = DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH,
    class_name: str = DOCUMENT_KV_HICACHE_BACKEND_CLASS,
    provider_factory: str | None = DOCUMENT_KV_HICACHE_PROVIDER_FACTORY,
    extra_config: Mapping[str, Any] | None = None,
    page_size: int | None = None,
    hicache_ratio: float | None = None,
    hicache_size_gb: int | None = None,
    hicache_io_backend: str | None = None,
    hicache_mem_layout: str | None = None,
    hicache_storage_prefetch_policy: str | None = None,
    hicache_write_policy: str | None = None,
) -> tuple[str, ...]:
    """Return CLI arguments suitable for ``python -m sglang.launch_server``."""

    config = sglang_hicache_launch_config(
        backend_name=backend_name,
        module_path=module_path,
        class_name=class_name,
        provider_factory=provider_factory,
        extra_config=extra_config,
        page_size=page_size,
        hicache_ratio=hicache_ratio,
        hicache_size_gb=hicache_size_gb,
        hicache_io_backend=hicache_io_backend,
        hicache_mem_layout=hicache_mem_layout,
        hicache_storage_prefetch_policy=hicache_storage_prefetch_policy,
        hicache_write_policy=hicache_write_policy,
    )
    args: list[str] = ["--enable-hierarchical-cache"]
    for key, value in config.items():
        if key == "enable_hierarchical_cache":
            continue
        args.extend((f"--{key.replace('_', '-')}", str(value)))
    return tuple(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a document KV SGLang HiCache launch-config sidecar.")
    parser.add_argument(
        "--backend-name",
        default="document_kv",
        help="SGLang dynamic HiCache backend name. Defaults to the Document KV backend contract name.",
    )
    parser.add_argument(
        "--module-path",
        default=DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH,
        help="Import path for the SGLang dynamic backend module.",
    )
    parser.add_argument(
        "--class-name",
        default=DOCUMENT_KV_HICACHE_BACKEND_CLASS,
        help="SGLang dynamic backend class name.",
    )
    parser.add_argument(
        "--provider-factory",
        default=DOCUMENT_KV_HICACHE_PROVIDER_FACTORY,
        help="Optional Cachet provider factory in module:attribute form.",
    )
    parser.add_argument(
        "--extra-config",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="Additional hicache_storage_backend_extra_config entry. Repeat as needed; values must be JSON.",
    )
    parser.add_argument("--page-size", type=int)
    parser.add_argument("--hicache-ratio", type=float)
    parser.add_argument("--hicache-size-gb", type=int)
    parser.add_argument("--hicache-io-backend")
    parser.add_argument("--hicache-mem-layout")
    parser.add_argument("--hicache-storage-prefetch-policy")
    parser.add_argument("--hicache-write-policy")
    parser.add_argument("--output-json", help="Write the config to this path instead of stdout.")
    try:
        args = parser.parse_args(argv)
        config = sglang_hicache_launch_config(
            backend_name=args.backend_name,
            module_path=args.module_path,
            class_name=args.class_name,
            provider_factory=args.provider_factory,
            extra_config=_extra_config_from_cli(args.extra_config),
            page_size=args.page_size,
            hicache_ratio=args.hicache_ratio,
            hicache_size_gb=args.hicache_size_gb,
            hicache_io_backend=args.hicache_io_backend,
            hicache_mem_layout=args.hicache_mem_layout,
            hicache_storage_prefetch_policy=args.hicache_storage_prefetch_policy,
            hicache_write_policy=args.hicache_write_policy,
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


def _document_kv_hicache_extra_config(
    *,
    backend_name: str,
    module_path: str,
    class_name: str,
    provider_factory: str | None,
    extra_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    spec = sglang_adapter_spec()
    _validate_non_empty_string(backend_name, field_name="backend_name")
    _validate_non_empty_string(module_path, field_name="module_path")
    _validate_non_empty_string(class_name, field_name="class_name")
    merged: dict[str, Any] = {
        "backend_name": backend_name,
        "module_path": module_path,
        "class_name": class_name,
    }
    if extra_config is not None:
        for key, value in extra_config.items():
            _validate_extra_config_key(key)
            merged[key] = value
    if provider_factory is not None:
        _validate_provider_factory(provider_factory)
        merged[DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY] = provider_factory
    merged.update(
        {
            "document_kv.record_type": DOCUMENT_KV_HICACHE_CONFIG_RECORD_TYPE,
            "document_kv.schema_version": DOCUMENT_KV_HICACHE_CONFIG_SCHEMA_VERSION,
            "document_kv.backend": spec.backend.value,
            "document_kv.connector_package": spec.connector_package,
            "document_kv.kv_injection_method": spec.kv_injection_method,
            "document_kv.engine_handoff_record_type": ENGINE_ADAPTER_HANDOFF_RECORD_TYPE,
            "document_kv.engine_handoff_schema_version": ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION,
            "document_kv.requires_native_runtime": True,
        }
    )
    _validate_json_serializable(merged, field_name="hicache_storage_backend_extra_config")
    return merged


def _add_positive_int(config: dict[str, Any], key: str, value: int | None) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    config[key] = value


def _add_non_negative_int(config: dict[str, Any], key: str, value: int | None) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    config[key] = value


def _add_positive_number(config: dict[str, Any], key: str, value: float | None) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0 or not math.isfinite(value):
        raise ValueError(f"{key} must be a positive number")
    config[key] = value


def _add_optional_string(config: dict[str, Any], key: str, value: str | None) -> None:
    if value is None:
        return
    _validate_non_empty_string(value, field_name=key)
    config[key] = value


def _validate_extra_config_key(key: str) -> None:
    _validate_non_empty_string(key, field_name="extra_config key")
    if key in _SGLANG_DYNAMIC_BACKEND_IDENTITY_KEYS:
        raise ValueError(f"extra_config cannot override SGLang dynamic backend identity key {key!r}")
    if key.startswith(DOCUMENT_KV_HICACHE_CONFIG_PREFIX):
        raise ValueError(f"extra_config keys starting with {DOCUMENT_KV_HICACHE_CONFIG_PREFIX!r} are reserved")


def _validate_provider_factory(factory_path: str) -> None:
    _validate_non_empty_string(factory_path, field_name="provider_factory")
    module_name, separator, attribute_name = factory_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("provider_factory must use module:attribute syntax")


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


def _validate_json_serializable(value: Any, *, field_name: str) -> None:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be JSON-serializable") from exc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
