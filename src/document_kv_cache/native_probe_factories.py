"""Built-in native probe factory entry points for backend integrations."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata, util
from pathlib import Path
from types import MappingProxyType
from typing import Any

from document_kv_cache.engine_adapters import (
    ENGINE_ADAPTER_HANDOFF_RECORD_TYPE,
    ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION,
    ENGINE_KV_CONNECTOR_ACTIONS_RECORD_TYPE,
    ENGINE_KV_CONNECTOR_ACTIONS_SCHEMA_VERSION,
    ENGINE_KV_CONNECTOR_PROBE_RECORD_TYPE,
    ENGINE_KV_CONNECTOR_PROBE_SCHEMA_VERSION,
    PayloadMode,
    ServingBackend,
)
from document_kv_cache.engine_probe import EngineKVProbeFactoryContext
from document_kv_cache.engine_probe import load_engine_kv_probe_factory
from document_kv_cache.serving_env import (
    serving_environment_profile,
    serving_environment_profile_to_record,
)

VLLM_NATIVE_PROBE_FACTORY = "document_kv_cache.native_probe_factories:vllm_native_probe_factory"
SGLANG_NATIVE_PROBE_FACTORY = "document_kv_cache.native_probe_factories:sglang_native_probe_factory"
VLLM_NATIVE_PROBE_DELEGATE_ENV = "DOCUMENT_KV_VLLM_NATIVE_PROBE_FACTORY"
SGLANG_NATIVE_PROBE_DELEGATE_ENV = "DOCUMENT_KV_SGLANG_NATIVE_PROBE_FACTORY"
NATIVE_PROBE_FACTORIES_RECORD_TYPE = "document_kv.native_probe_factories.v1"
_REQUIRED_NATIVE_PROBE_FACTORY_BACKENDS = ("vllm", "sglang")
_NATIVE_PROBE_FACTORIES_KEYS = frozenset({"record_type", "factories"})
_NATIVE_PROBE_FACTORY_KEYS = frozenset(
    {
        "adapter_contract",
        "backend",
        "delegate_factory_path",
        "factory_path",
        "package_name",
        "package_importable",
        "package_version",
        "serving_environment_profile",
        "supported",
        "reason",
    }
)
NATIVE_PROBE_ADAPTER_CONTRACT: Mapping[str, str | int | bool] = MappingProxyType(
    {
        "handoff_record_type": ENGINE_ADAPTER_HANDOFF_RECORD_TYPE,
        "handoff_schema_version": ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION,
        "probe_record_type": ENGINE_KV_CONNECTOR_PROBE_RECORD_TYPE,
        "probe_schema_version": ENGINE_KV_CONNECTOR_PROBE_SCHEMA_VERSION,
        "actions_record_type": ENGINE_KV_CONNECTOR_ACTIONS_RECORD_TYPE,
        "actions_schema_version": ENGINE_KV_CONNECTOR_ACTIONS_SCHEMA_VERSION,
        "layout_version": "qwen3-v1",
        "payload_mode": PayloadMode.MERGED.value,
        "requires_native_probe": True,
    }
)
_NATIVE_PROBE_ADAPTER_CONTRACT_KEYS = frozenset(NATIVE_PROBE_ADAPTER_CONTRACT)
_BUILTIN_NATIVE_PROBE_FACTORY_PATHS = {
    "vllm": VLLM_NATIVE_PROBE_FACTORY,
    "sglang": SGLANG_NATIVE_PROBE_FACTORY,
}
_BUILTIN_NATIVE_PROBE_FACTORY_TARGETS = frozenset(
    {
        ("cachet", "sglang_native_probe_factory"),
        ("cachet", "vllm_native_probe_factory"),
        ("document_kv_cache", "sglang_native_probe_factory"),
        ("document_kv_cache", "vllm_native_probe_factory"),
        ("document_kv_cache.native_probe_factories", "vllm_native_probe_factory"),
        ("document_kv_cache.native_probe_factories", "sglang_native_probe_factory"),
        ("restaurant_kv_serving", "sglang_native_probe_factory"),
        ("restaurant_kv_serving", "vllm_native_probe_factory"),
        ("restaurant_kv_serving.native_probe_factories", "sglang_native_probe_factory"),
        ("restaurant_kv_serving.native_probe_factories", "vllm_native_probe_factory"),
    }
)


class NativeProbeFactoryUnavailable(RuntimeError):
    """Raised when a built-in backend factory cannot construct a native probe."""


@dataclass(frozen=True, slots=True)
class NativeProbeFactoryInspection:
    """Import and support status for a packaged native probe factory."""

    backend: ServingBackend
    factory_path: str
    package_name: str
    package_importable: bool
    package_version: str | None
    supported: bool
    reason: str
    delegate_factory_path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _serving_backend(self.backend))
        if not self.factory_path:
            raise ValueError("factory_path must be non-empty")
        if not self.package_name:
            raise ValueError("package_name must be non-empty")
        if type(self.package_importable) is not bool:
            raise ValueError("package_importable must be a boolean")
        if self.package_version is not None and not self.package_version:
            raise ValueError("package_version must be non-empty when provided")
        if self.delegate_factory_path is not None and not self.delegate_factory_path:
            raise ValueError("delegate_factory_path must be non-empty when provided")
        if type(self.supported) is not bool:
            raise ValueError("supported must be a boolean")
        if not self.reason:
            raise ValueError("reason must be non-empty")


def builtin_native_probe_factory_path(backend: ServingBackend | str) -> str:
    """Return the public factory path reserved for a supported serving backend."""

    backend = _serving_backend(backend)
    if backend == ServingBackend.VLLM:
        return VLLM_NATIVE_PROBE_FACTORY
    if backend == ServingBackend.SGLANG:
        return SGLANG_NATIVE_PROBE_FACTORY
    raise ValueError(f"Unsupported serving backend {backend!r}")


def inspect_builtin_native_probe_factory(backend: ServingBackend | str) -> NativeProbeFactoryInspection:
    """Inspect the local environment for a built-in native probe target.

    The package owns the evidence and handoff contract. The actual vLLM/SGLang
    block-manager imports stay in a backend-specific adapter configured by the
    matching delegate environment variable.
    """

    backend = _serving_backend(backend)
    package_name = _backend_package_name(backend)
    version = _package_version(package_name)
    package_importable = util.find_spec(package_name) is not None
    delegate_factory_path = _delegate_factory_path_from_env(backend)
    if version is None and not package_importable:
        reason = f"{package_name!r} is not installed in this environment"
        supported = False
    elif not package_importable:
        reason = f"{package_name} {version} package metadata is available but the package is not importable"
        supported = False
    elif version is None:
        reason = f"{package_name!r} is importable but package metadata is unavailable"
        supported = False
    elif delegate_factory_path is None:
        reason = (
            f"{package_name} {version} is installed; set "
            f"{_delegate_factory_env_name(backend)} to a backend-native "
            "Document KV probe factory"
        )
        supported = False
    elif _is_builtin_native_probe_factory_path(delegate_factory_path):
        reason = (
            f"delegate native probe factory {delegate_factory_path!r} points at a built-in "
            "Document KV factory; set the delegate to a backend-native block-manager adapter"
        )
        supported = False
    else:
        try:
            delegate_factory = load_engine_kv_probe_factory(delegate_factory_path)
        except Exception as exc:  # pragma: no cover - exact loader failures vary by adapter
            reason = f"delegate native probe factory {delegate_factory_path!r} is unavailable: {exc}"
            supported = False
        else:
            if _is_builtin_native_probe_factory(delegate_factory):
                reason = (
                    f"delegate native probe factory {delegate_factory_path!r} resolves to a built-in "
                    "Document KV factory; set the delegate to a backend-native block-manager adapter"
                )
                supported = False
            else:
                reason = f"{package_name} {version} is installed; delegate native probe factory is loadable"
                supported = True
    return NativeProbeFactoryInspection(
        backend=backend,
        factory_path=builtin_native_probe_factory_path(backend),
        package_name=package_name,
        package_importable=package_importable,
        package_version=version,
        delegate_factory_path=delegate_factory_path,
        supported=supported,
        reason=reason,
    )


def native_probe_factory_inspection_to_record(
    inspection: NativeProbeFactoryInspection,
) -> dict[str, Any]:
    """Serialize an inspection result for diagnostics and release planning."""

    serving_profile = serving_environment_profile(inspection.backend)
    return {
        "backend": inspection.backend.value,
        "factory_path": inspection.factory_path,
        "adapter_contract": native_probe_adapter_contract_to_record(),
        "package_name": inspection.package_name,
        "package_importable": inspection.package_importable,
        "package_version": inspection.package_version,
        "delegate_factory_path": inspection.delegate_factory_path,
        "serving_environment_profile": serving_environment_profile_to_record(serving_profile),
        "supported": inspection.supported,
        "reason": inspection.reason,
    }


def native_probe_adapter_contract_to_record() -> dict[str, str | int | bool]:
    """Return the engine handoff contract required by built-in native probe factories."""

    return dict(NATIVE_PROBE_ADAPTER_CONTRACT)


def inspect_builtin_native_probe_factories() -> tuple[NativeProbeFactoryInspection, ...]:
    """Inspect all built-in native probe factory entry points."""

    return tuple(
        inspect_builtin_native_probe_factory(backend)
        for backend in (ServingBackend.VLLM, ServingBackend.SGLANG)
    )


def builtin_native_probe_factories_to_record() -> dict[str, Any]:
    """Serialize all built-in factory inspections as a stable diagnostics record."""

    return {
        "record_type": NATIVE_PROBE_FACTORIES_RECORD_TYPE,
        "factories": [
            native_probe_factory_inspection_to_record(inspection)
            for inspection in inspect_builtin_native_probe_factories()
        ],
    }


def native_probe_factories_record_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return schema and compatibility issues for native factory diagnostics."""

    issues: list[str] = []
    issues.extend(_unexpected_keys(record, _NATIVE_PROBE_FACTORIES_KEYS, "native probe factories sidecar"))
    if record.get("record_type") != NATIVE_PROBE_FACTORIES_RECORD_TYPE:
        issues.append(
            f"native probe factories sidecar record_type must be {NATIVE_PROBE_FACTORIES_RECORD_TYPE!r}"
        )
    factories = record.get("factories")
    if not isinstance(factories, Sequence) or isinstance(factories, (str, bytes, bytearray)) or not factories:
        issues.append("native probe factories sidecar factories must be a non-empty array")
        return _dedupe_strings(issues)

    backends: list[str] = []
    for index, factory in enumerate(factories):
        if not isinstance(factory, Mapping):
            issues.append(f"native probe factories sidecar factories[{index}] must be an object")
            continue
        issues.extend(_native_probe_factory_issues(factory, index=index))
        backend = factory.get("backend")
        if isinstance(backend, str):
            backends.append(backend)
    if set(backends) != set(_REQUIRED_NATIVE_PROBE_FACTORY_BACKENDS) or len(backends) != len(set(backends)):
        issues.append("native probe factories sidecar backends must match required backends")
    return _dedupe_strings(issues)


def validate_native_probe_factories_record(record: Mapping[str, Any]) -> None:
    """Validate a ``document_kv.native_probe_factories.v1`` diagnostics record."""

    issues = native_probe_factories_record_issues(record)
    if issues:
        raise ValueError("; ".join(issues))


def write_builtin_native_probe_factories_record_json(path: str | Path) -> None:
    """Write the built-in native factory diagnostics record to a JSON file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(builtin_native_probe_factories_to_record(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def vllm_native_probe_factory(context: EngineKVProbeFactoryContext) -> Any:
    """Delegate to the configured vLLM-native block-manager probe factory."""

    _validate_context_backend(context, ServingBackend.VLLM)
    return _load_delegate_factory(ServingBackend.VLLM)(context)


def sglang_native_probe_factory(context: EngineKVProbeFactoryContext) -> Any:
    """Delegate to the configured SGLang-native block-manager probe factory."""

    _validate_context_backend(context, ServingBackend.SGLANG)
    return _load_delegate_factory(ServingBackend.SGLANG)(context)


def _validate_context_backend(context: EngineKVProbeFactoryContext, expected: ServingBackend) -> None:
    observed = _serving_backend(context.backend)
    if observed != expected:
        raise ValueError(
            f"{expected.value} native probe factory received {observed.value!r} handoff context"
        )


def _backend_package_name(backend: ServingBackend) -> str:
    if backend == ServingBackend.VLLM:
        return "vllm"
    if backend == ServingBackend.SGLANG:
        return "sglang"
    raise ValueError(f"Unsupported serving backend {backend!r}")


def _delegate_factory_env_name(backend: ServingBackend) -> str:
    if backend == ServingBackend.VLLM:
        return VLLM_NATIVE_PROBE_DELEGATE_ENV
    if backend == ServingBackend.SGLANG:
        return SGLANG_NATIVE_PROBE_DELEGATE_ENV
    raise ValueError(f"Unsupported serving backend {backend!r}")


def _delegate_factory_path_from_env(backend: ServingBackend, environ: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    value = env.get(_delegate_factory_env_name(backend))
    return value or None


def _load_delegate_factory(backend: ServingBackend):
    inspection = inspect_builtin_native_probe_factory(backend)
    if not inspection.supported or inspection.delegate_factory_path is None:
        raise NativeProbeFactoryUnavailable(inspection.reason)
    return load_engine_kv_probe_factory(inspection.delegate_factory_path)


def _is_builtin_native_probe_factory_path(factory_path: str) -> bool:
    return _factory_path_target(factory_path) in _BUILTIN_NATIVE_PROBE_FACTORY_TARGETS


def _is_builtin_native_probe_factory(factory: Any) -> bool:
    return factory is vllm_native_probe_factory or factory is sglang_native_probe_factory


def _factory_path_target(factory_path: str) -> tuple[str, str] | None:
    if ":" in factory_path:
        module_name, attribute_name = factory_path.split(":", maxsplit=1)
    else:
        module_name, _, attribute_name = factory_path.rpartition(".")
    if not module_name or not attribute_name:
        return None
    return module_name, attribute_name


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _serving_backend(value: ServingBackend | str) -> ServingBackend:
    try:
        return value if isinstance(value, ServingBackend) else ServingBackend(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported serving backend {value!r}") from exc


def _native_probe_factory_issues(factory: Mapping[str, Any], *, index: int) -> tuple[str, ...]:
    label = f"native probe factories sidecar factories[{index}]"
    issues: list[str] = []
    issues.extend(_unexpected_keys(factory, _NATIVE_PROBE_FACTORY_KEYS, label))
    backend = factory.get("backend")
    if not isinstance(backend, str) or backend not in _REQUIRED_NATIVE_PROBE_FACTORY_BACKENDS:
        issues.append(f"{label}.backend must be one of {list(_REQUIRED_NATIVE_PROBE_FACTORY_BACKENDS)!r}")
    else:
        expected_factory_path = _BUILTIN_NATIVE_PROBE_FACTORY_PATHS.get(backend)
        if factory.get("factory_path") != expected_factory_path:
            issues.append(f"{label}.factory_path must match the built-in {backend} factory path")
    for field_name in ("backend", "factory_path", "package_name", "reason"):
        issues.extend(_required_str_field(factory, field_name, label))
    issues.extend(_optional_str_field(factory, "package_version", label))
    issues.extend(_optional_str_field(factory, "delegate_factory_path", label))
    for field_name in ("package_importable", "supported"):
        issues.extend(_bool_field(factory, field_name, label))
    adapter_contract = factory.get("adapter_contract")
    if not isinstance(adapter_contract, Mapping):
        issues.append(f"{label}.adapter_contract must be an object")
    else:
        issues.extend(_native_probe_adapter_contract_issues(adapter_contract, label=f"{label}.adapter_contract"))
    if factory.get("supported") is True:
        if factory.get("package_importable") is not True:
            issues.append(f"{label}.package_importable must be true when supported is true")
        if not isinstance(factory.get("package_version"), str) or not factory["package_version"]:
            issues.append(f"{label}.package_version must be non-empty when supported is true")
        if not isinstance(factory.get("delegate_factory_path"), str) or not factory["delegate_factory_path"]:
            issues.append(f"{label}.delegate_factory_path must be non-empty when supported is true")
        elif _is_builtin_native_probe_factory_path(factory["delegate_factory_path"]):
            issues.append(
                f"{label}.delegate_factory_path must not point at a built-in native probe factory "
                "when supported is true"
            )
    serving_profile = factory.get("serving_environment_profile")
    if not isinstance(serving_profile, Mapping):
        issues.append(f"{label}.serving_environment_profile must be an object")
    elif isinstance(backend, str) and backend in _REQUIRED_NATIVE_PROBE_FACTORY_BACKENDS:
        expected_profile = serving_environment_profile_to_record(serving_environment_profile(backend))
        if dict(serving_profile) != expected_profile:
            issues.append(f"{label}.serving_environment_profile must match the built-in {backend} profile")
    return tuple(issues)


def _native_probe_adapter_contract_issues(contract: Mapping[str, Any], *, label: str) -> tuple[str, ...]:
    issues: list[str] = []
    issues.extend(_unexpected_keys(contract, _NATIVE_PROBE_ADAPTER_CONTRACT_KEYS, label))
    expected = native_probe_adapter_contract_to_record()
    for field_name, expected_value in expected.items():
        value = contract.get(field_name)
        field_label = f"{label}.{field_name}"
        if isinstance(expected_value, bool):
            if type(value) is not bool:
                issues.append(f"{field_label} must be boolean")
                continue
        elif isinstance(expected_value, int):
            if type(value) is not int:
                issues.append(f"{field_label} must be an integer")
                continue
        elif not isinstance(value, str):
            issues.append(f"{field_label} must be a string")
            continue
        if value != expected_value:
            issues.append(f"{field_label} must match the built-in native probe adapter contract")
    return tuple(issues)


def _unexpected_keys(record: Mapping[str, Any], allowed_keys: frozenset[str], label: str) -> tuple[str, ...]:
    unexpected = sorted(str(key) for key in record if key not in allowed_keys)
    if not unexpected:
        return ()
    return (f"{label} has unsupported keys: {unexpected}",)


def _required_str_field(record: Mapping[str, Any], field_name: str, label: str) -> tuple[str, ...]:
    value = record.get(field_name)
    if isinstance(value, str) and value:
        return ()
    return (f"{label}.{field_name} must be a non-empty string",)


def _optional_str_field(record: Mapping[str, Any], field_name: str, label: str) -> tuple[str, ...]:
    value = record.get(field_name)
    if value is None or isinstance(value, str):
        return ()
    return (f"{label}.{field_name} must be a string or null",)


def _bool_field(record: Mapping[str, Any], field_name: str, label: str) -> tuple[str, ...]:
    if type(record.get(field_name)) is bool:
        return ()
    return (f"{label}.{field_name} must be boolean",)


def _dedupe_strings(values: Sequence[str]) -> tuple[str, ...]:
    deduped = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)


def main(argv: Sequence[str] | None = None) -> int:
    """Emit built-in native probe factory diagnostics for release planning."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect built-in vLLM/SGLang native probe factory entry points "
            "and their pinned isolated serving-environment profiles."
        )
    )
    parser.add_argument(
        "--output-json",
        help="Optional file path for the diagnostics JSON. Defaults to stdout.",
    )
    args = parser.parse_args(argv)

    if args.output_json:
        write_builtin_native_probe_factories_record_json(args.output_json)
    else:
        record = builtin_native_probe_factories_to_record()
        print(json.dumps(record, indent=2, sort_keys=True))
    return 0


__all__ = [
    "NATIVE_PROBE_ADAPTER_CONTRACT",
    "NATIVE_PROBE_FACTORIES_RECORD_TYPE",
    "NativeProbeFactoryInspection",
    "NativeProbeFactoryUnavailable",
    "SGLANG_NATIVE_PROBE_DELEGATE_ENV",
    "SGLANG_NATIVE_PROBE_FACTORY",
    "VLLM_NATIVE_PROBE_DELEGATE_ENV",
    "VLLM_NATIVE_PROBE_FACTORY",
    "builtin_native_probe_factories_to_record",
    "builtin_native_probe_factory_path",
    "inspect_builtin_native_probe_factories",
    "inspect_builtin_native_probe_factory",
    "main",
    "native_probe_adapter_contract_to_record",
    "native_probe_factories_record_issues",
    "native_probe_factory_inspection_to_record",
    "sglang_native_probe_factory",
    "validate_native_probe_factories_record",
    "vllm_native_probe_factory",
    "write_builtin_native_probe_factories_record_json",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
