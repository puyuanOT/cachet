"""Compatibility wrapper for :mod:`document_kv_cache.native_probe_factories`."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata, util
from types import MappingProxyType
from typing import Any

from document_kv_cache.native_probe_factories import (
    SGLANG_NATIVE_PROBE_FACTORY,
    VLLM_NATIVE_PROBE_FACTORY,
    NativeProbeFactoryInspection,
    NativeProbeFactoryUnavailable,
    builtin_native_probe_factories_to_record,
    builtin_native_probe_factory_path,
    inspect_builtin_native_probe_factories,
    inspect_builtin_native_probe_factory,
    main,
    native_probe_factory_inspection_to_record,
    sglang_native_probe_factory,
    vllm_native_probe_factory,
    write_builtin_native_probe_factories_record_json,
)
from restaurant_kv_serving.engine_adapters import ServingBackend
from restaurant_kv_serving.engine_probe import EngineKVProbeFactoryContext

__all__ = [
    "NativeProbeFactoryInspection",
    "NativeProbeFactoryUnavailable",
    "SGLANG_NATIVE_PROBE_FACTORY",
    "VLLM_NATIVE_PROBE_FACTORY",
    "builtin_native_probe_factories_to_record",
    "builtin_native_probe_factory_path",
    "inspect_builtin_native_probe_factories",
    "inspect_builtin_native_probe_factory",
    "main",
    "native_probe_factory_inspection_to_record",
    "sglang_native_probe_factory",
    "vllm_native_probe_factory",
    "write_builtin_native_probe_factories_record_json",
]
