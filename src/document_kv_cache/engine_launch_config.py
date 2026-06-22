"""Build and validate document KV serving-engine launch configs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_kv_cache.engine_adapters import (
    ENGINE_ADAPTER_HANDOFF_RECORD_TYPE,
    ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION,
    ServingBackend,
)
from document_kv_cache.storage import local_path
from vllm_kv_injection.vllm_dynamic_connector import DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY
from vllm_kv_injection.vllm_native_provider import DOCUMENT_KV_NATIVE_PROVIDER_FACTORY

ENGINE_LAUNCH_CONFIG_EVIDENCE_RECORD_TYPE = "document_kv.engine_launch_config_evidence.v1"
ENGINE_LAUNCH_CONFIG_EVIDENCE_SCHEMA_VERSION = 1
REQUIRED_ENGINE_LAUNCH_CONFIG_BACKENDS = tuple(backend.value for backend in ServingBackend)

_VLLM_RECORD_TYPE_PREFIX = "vllm_kv_injection."
_SGLANG_RECORD_TYPE_PREFIX = "sglang_kv_injection."
_VLLM_LAUNCH_CONFIG_KEYS = frozenset(
    {
        "kv_connector",
        "kv_connector_module_path",
        "kv_role",
        "kv_connector_extra_config",
    }
)
_SGLANG_LAUNCH_CONFIG_KEYS = frozenset(
    {
        "enable_hierarchical_cache",
        "hicache_storage_backend",
        "hicache_storage_backend_extra_config",
    }
)
_SGLANG_DYNAMIC_HICACHE_BACKEND = "dynamic"
_VLLM_DOCUMENT_KV_CONNECTOR = "DocumentKVConnector"
_VLLM_ALLOWED_KV_ROLES = frozenset({"kv_both", "kv_producer", "kv_consumer"})
_SGLANG_DOCUMENT_KV_BACKEND_NAME = "document_kv"
_SGLANG_DOCUMENT_KV_CLASS_NAME = "DocumentKVHiCacheBackend"
_VLLM_DOCUMENT_KV_MODULE_LEAVES = (
    "vllm_dynamic_connector",
    "document_kv_connector",
)
_SGLANG_DOCUMENT_KV_MODULE_LEAVES = (
    "sglang_dynamic_backend",
    "document_kv_backend",
)
DEFAULT_VLLM_DOCUMENT_KV_MODULE_PATH = "vllm_kv_injection.vllm_dynamic_connector"
DEFAULT_SGLANG_DOCUMENT_KV_MODULE_PATH = "sglang_kv_injection.sglang_dynamic_backend"
DEFAULT_ENGINE_LAUNCH_CONFIG_SCHEMA_VERSION = 1
DEFAULT_ENGINE_LAUNCH_CONFIG_KV_INJECTION_METHOD = "native-kv-import"
DEFAULT_VLLM_ENGINE_LAUNCH_CONFIG_RECORD_TYPE = f"{_VLLM_RECORD_TYPE_PREFIX}launch_config.v1"
DEFAULT_SGLANG_ENGINE_LAUNCH_CONFIG_RECORD_TYPE = f"{_SGLANG_RECORD_TYPE_PREFIX}launch_config.v1"
DEFAULT_VLLM_DOCUMENT_KV_PROVIDER_FACTORY = DOCUMENT_KV_NATIVE_PROVIDER_FACTORY

__all__ = [
    "DEFAULT_ENGINE_LAUNCH_CONFIG_KV_INJECTION_METHOD",
    "DEFAULT_ENGINE_LAUNCH_CONFIG_SCHEMA_VERSION",
    "DEFAULT_SGLANG_DOCUMENT_KV_MODULE_PATH",
    "DEFAULT_SGLANG_ENGINE_LAUNCH_CONFIG_RECORD_TYPE",
    "DEFAULT_VLLM_DOCUMENT_KV_PROVIDER_FACTORY",
    "DEFAULT_VLLM_DOCUMENT_KV_MODULE_PATH",
    "DEFAULT_VLLM_ENGINE_LAUNCH_CONFIG_RECORD_TYPE",
    "ENGINE_LAUNCH_CONFIG_EVIDENCE_RECORD_TYPE",
    "ENGINE_LAUNCH_CONFIG_EVIDENCE_SCHEMA_VERSION",
    "REQUIRED_ENGINE_LAUNCH_CONFIG_BACKENDS",
    "EngineLaunchConfigEvidence",
    "build_sglang_launch_config",
    "build_vllm_launch_config",
    "engine_launch_config_evidence_to_record",
    "engine_launch_config_record_issues",
    "evaluate_engine_launch_config_evidence",
    "main",
    "read_engine_launch_config_json",
    "validate_engine_launch_config_record",
    "write_engine_launch_config_json",
    "write_engine_launch_config_evidence_json",
]


@dataclass(frozen=True, slots=True)
class EngineLaunchConfigEvidence:
    """Summary for vLLM/SGLang launch-config sidecar validation."""

    backends: tuple[str, ...]
    missing_backends: tuple[str, ...]
    invalid_records: tuple[str, ...]
    duplicate_backends: tuple[str, ...] = ()
    required_backends: tuple[str, ...] = REQUIRED_ENGINE_LAUNCH_CONFIG_BACKENDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "backends", _validated_backend_tuple(self.backends, "backends"))
        object.__setattr__(
            self,
            "missing_backends",
            _validated_backend_tuple(self.missing_backends, "missing_backends"),
        )
        object.__setattr__(
            self,
            "duplicate_backends",
            _validated_backend_tuple(self.duplicate_backends, "duplicate_backends"),
        )
        object.__setattr__(self, "invalid_records", _validated_string_tuple(self.invalid_records, "invalid_records"))
        object.__setattr__(self, "required_backends", _validated_required_backends(self.required_backends))

    @property
    def ok(self) -> bool:
        return not self.missing_backends and not self.invalid_records and not self.duplicate_backends

    @property
    def issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.missing_backends:
            issues.append(f"missing engine launch config backends: {', '.join(self.missing_backends)}")
        if self.duplicate_backends:
            issues.append(f"duplicate engine launch config backends: {', '.join(self.duplicate_backends)}")
        issues.extend(self.invalid_records)
        return tuple(issues)


def build_vllm_launch_config(
    *,
    module_path: str = DEFAULT_VLLM_DOCUMENT_KV_MODULE_PATH,
    kv_role: str = "kv_both",
    record_type: str = DEFAULT_VLLM_ENGINE_LAUNCH_CONFIG_RECORD_TYPE,
    schema_version: int = DEFAULT_ENGINE_LAUNCH_CONFIG_SCHEMA_VERSION,
    kv_injection_method: str = DEFAULT_ENGINE_LAUNCH_CONFIG_KV_INJECTION_METHOD,
    extra_config: Mapping[str, Any] | None = None,
    provider_factory: str | None = DEFAULT_VLLM_DOCUMENT_KV_PROVIDER_FACTORY,
) -> dict[str, Any]:
    """Build a validated vLLM transfer config for the document KV connector."""

    kv_connector_extra_config = _build_document_kv_extra_config(
        ServingBackend.VLLM,
        record_type=record_type,
        schema_version=schema_version,
        kv_injection_method=kv_injection_method,
        extra_config=extra_config,
        provider_factory=provider_factory,
    )
    record = {
        "kv_connector": _VLLM_DOCUMENT_KV_CONNECTOR,
        "kv_connector_module_path": module_path,
        "kv_role": kv_role,
        "kv_connector_extra_config": kv_connector_extra_config,
    }
    validate_engine_launch_config_record(record, expected_backend=ServingBackend.VLLM)
    return record


def build_sglang_launch_config(
    *,
    module_path: str = DEFAULT_SGLANG_DOCUMENT_KV_MODULE_PATH,
    record_type: str = DEFAULT_SGLANG_ENGINE_LAUNCH_CONFIG_RECORD_TYPE,
    schema_version: int = DEFAULT_ENGINE_LAUNCH_CONFIG_SCHEMA_VERSION,
    kv_injection_method: str = DEFAULT_ENGINE_LAUNCH_CONFIG_KV_INJECTION_METHOD,
    extra_config: Mapping[str, Any] | None = None,
    encode_extra_config_as_json: bool = True,
) -> dict[str, Any]:
    """Build a validated SGLang HiCache config for the document KV backend."""

    hicache_extra_config = {
        "backend_name": _SGLANG_DOCUMENT_KV_BACKEND_NAME,
        "module_path": module_path,
        "class_name": _SGLANG_DOCUMENT_KV_CLASS_NAME,
        **_build_document_kv_extra_config(
            ServingBackend.SGLANG,
            record_type=record_type,
            schema_version=schema_version,
            kv_injection_method=kv_injection_method,
            extra_config=extra_config,
            reserved_keys={"backend_name", "module_path", "class_name"},
        ),
    }
    record = {
        "enable_hierarchical_cache": True,
        "hicache_storage_backend": _SGLANG_DYNAMIC_HICACHE_BACKEND,
        "hicache_storage_backend_extra_config": (
            json.dumps(hicache_extra_config, sort_keys=True)
            if encode_extra_config_as_json
            else hicache_extra_config
        ),
    }
    validate_engine_launch_config_record(record, expected_backend=ServingBackend.SGLANG)
    return record


def validate_engine_launch_config_record(
    record: Mapping[str, Any],
    *,
    expected_backend: str | ServingBackend | None = None,
) -> None:
    """Validate one adapter launch config record.

    The core package does not import either adapter package. Instead, it checks
    the stable JSON shape emitted by the vLLM transfer-config and SGLang
    HiCache-config helpers plus the shared ``document_kv.*`` handoff markers.
    """

    issues = engine_launch_config_record_issues(record, expected_backend=expected_backend)
    if issues:
        raise ValueError("; ".join(issues))


def engine_launch_config_record_issues(
    record: Mapping[str, Any],
    *,
    expected_backend: str | ServingBackend | None = None,
) -> tuple[str, ...]:
    """Return validation issues for one adapter launch config record."""

    issues: list[str] = []
    if not isinstance(record, Mapping):
        return ("engine launch config record must be a mapping",)
    expected = _backend_from_optional_value(expected_backend, field_name="expected_backend")
    backend, inferred_issues = _infer_launch_config_backend(record)
    issues.extend(inferred_issues)
    if backend is not None and expected is not None and backend != expected:
        issues.append(f"engine launch config backend {backend.value!r} does not match expected_backend")
    return tuple(issues)


def evaluate_engine_launch_config_evidence(
    records: Sequence[Mapping[str, Any]],
    *,
    required_backends: Sequence[str | ServingBackend] = REQUIRED_ENGINE_LAUNCH_CONFIG_BACKENDS,
) -> EngineLaunchConfigEvidence:
    """Evaluate a collection of launch-config sidecars for release readiness."""

    required = _validated_required_backends(required_backends)
    present: list[str] = []
    invalid_records: list[str] = []
    for index, record in enumerate(records):
        issues = engine_launch_config_record_issues(record)
        if issues:
            invalid_records.append(f"record[{index}]: {'; '.join(issues)}")
            continue
        backend, _ = _infer_launch_config_backend(record)
        if backend is None:  # pragma: no cover - guarded by validation above.
            invalid_records.append(f"record[{index}]: could not infer backend")
            continue
        present.append(backend.value)

    present_set = set(present)
    duplicate_backends = tuple(backend for backend in required if present.count(backend) > 1)
    present_backends = tuple(backend for backend in required if backend in present_set)
    missing_backends = tuple(backend for backend in required if backend not in present_set)
    return EngineLaunchConfigEvidence(
        backends=present_backends,
        missing_backends=missing_backends,
        duplicate_backends=duplicate_backends,
        invalid_records=tuple(invalid_records),
        required_backends=required,
    )


def engine_launch_config_evidence_to_record(evidence: EngineLaunchConfigEvidence) -> dict[str, Any]:
    return {
        "record_type": ENGINE_LAUNCH_CONFIG_EVIDENCE_RECORD_TYPE,
        "schema_version": ENGINE_LAUNCH_CONFIG_EVIDENCE_SCHEMA_VERSION,
        "ok": evidence.ok,
        "issues": list(evidence.issues),
        "backends": list(evidence.backends),
        "missing_backends": list(evidence.missing_backends),
        "duplicate_backends": list(evidence.duplicate_backends),
        "invalid_records": list(evidence.invalid_records),
        "required_backends": list(evidence.required_backends),
    }


def read_engine_launch_config_json(
    path: str | Path,
    *,
    expected_backend: str | ServingBackend | None = None,
) -> dict[str, Any]:
    record = json.loads(local_path(str(path)).read_text(encoding="utf-8"))
    validate_engine_launch_config_record(record, expected_backend=expected_backend)
    return record


def write_engine_launch_config_json(
    record: Mapping[str, Any],
    path: str | Path,
    *,
    expected_backend: str | ServingBackend | None = None,
) -> Path:
    """Validate and write one engine launch-config sidecar to JSON."""

    validate_engine_launch_config_record(record, expected_backend=expected_backend)
    target_path = local_path(str(path))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target_path


def write_engine_launch_config_evidence_json(
    evidence: EngineLaunchConfigEvidence,
    path: str | Path,
) -> Path:
    target_path = local_path(str(path))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(engine_launch_config_evidence_to_record(evidence), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target_path


def main(argv: Sequence[str] | None = None) -> int:
    """Build a validated vLLM or SGLang document KV launch-config sidecar."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        extra_config = _extra_config_from_items(args.extra_config)
        if args.command == "build-vllm":
            record = build_vllm_launch_config(
                module_path=args.module_path,
                kv_role=args.kv_role,
                record_type=args.record_type,
                schema_version=args.schema_version,
                kv_injection_method=args.kv_injection_method,
                extra_config=extra_config,
                provider_factory=args.provider_factory,
            )
            expected_backend = ServingBackend.VLLM
        elif args.command == "build-sglang":
            record = build_sglang_launch_config(
                module_path=args.module_path,
                record_type=args.record_type,
                schema_version=args.schema_version,
                kv_injection_method=args.kv_injection_method,
                extra_config=extra_config,
            )
            expected_backend = ServingBackend.SGLANG
        else:  # pragma: no cover - argparse restricts this.
            raise ValueError(f"unsupported command {args.command!r}")

        if args.output_json:
            write_engine_launch_config_json(record, args.output_json, expected_backend=expected_backend)
        else:
            print(json.dumps(record, indent=2, sort_keys=True))
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


def _infer_launch_config_backend(record: Mapping[str, Any]) -> tuple[ServingBackend | None, tuple[str, ...]]:
    if _VLLM_LAUNCH_CONFIG_KEYS.issubset(record):
        return ServingBackend.VLLM, _validate_vllm_launch_config(record)
    if _SGLANG_LAUNCH_CONFIG_KEYS.issubset(record):
        return ServingBackend.SGLANG, _validate_sglang_launch_config(record)
    return None, (
        "engine launch config must match either the vLLM transfer config shape "
        "or the SGLang HiCache config shape",
    )


def _validate_vllm_launch_config(record: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    if record.get("kv_connector") != _VLLM_DOCUMENT_KV_CONNECTOR:
        issues.append(f"vLLM launch config kv_connector must be {_VLLM_DOCUMENT_KV_CONNECTOR!r}")
    issues.extend(
        _module_path_issues(
            record.get("kv_connector_module_path"),
            field_name="vLLM launch config kv_connector_module_path",
            expected_leaves=_VLLM_DOCUMENT_KV_MODULE_LEAVES,
        )
    )
    if record.get("kv_role") not in _VLLM_ALLOWED_KV_ROLES:
        issues.append(
            "vLLM launch config kv_role must be one of "
            f"{sorted(_VLLM_ALLOWED_KV_ROLES)!r}"
        )
    extra_config = record.get("kv_connector_extra_config")
    if not isinstance(extra_config, Mapping):
        issues.append("vLLM launch config kv_connector_extra_config must be a mapping")
        return tuple(issues)
    issues.extend(
        _validate_document_kv_extra_config(
            extra_config,
            expected_backend=ServingBackend.VLLM,
            expected_record_type_prefix=_VLLM_RECORD_TYPE_PREFIX,
        )
    )
    return tuple(issues)


def _validate_sglang_launch_config(record: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    if record.get("enable_hierarchical_cache") is not True:
        issues.append("SGLang launch config enable_hierarchical_cache must be true")
    if record.get("hicache_storage_backend") != _SGLANG_DYNAMIC_HICACHE_BACKEND:
        issues.append("SGLang launch config hicache_storage_backend must be 'dynamic'")
    extra_config, extra_issues = _sglang_extra_config(record.get("hicache_storage_backend_extra_config"))
    issues.extend(extra_issues)
    if extra_config is None:
        return tuple(issues)
    if extra_config.get("backend_name") != _SGLANG_DOCUMENT_KV_BACKEND_NAME:
        issues.append(f"SGLang HiCache extra config backend_name must be {_SGLANG_DOCUMENT_KV_BACKEND_NAME!r}")
    issues.extend(
        _module_path_issues(
            extra_config.get("module_path"),
            field_name="SGLang HiCache extra config module_path",
            expected_leaves=_SGLANG_DOCUMENT_KV_MODULE_LEAVES,
        )
    )
    if extra_config.get("class_name") != _SGLANG_DOCUMENT_KV_CLASS_NAME:
        issues.append(f"SGLang HiCache extra config class_name must be {_SGLANG_DOCUMENT_KV_CLASS_NAME!r}")
    issues.extend(
        _validate_document_kv_extra_config(
            extra_config,
            expected_backend=ServingBackend.SGLANG,
            expected_record_type_prefix=_SGLANG_RECORD_TYPE_PREFIX,
        )
    )
    return tuple(issues)


def _sglang_extra_config(value: Any) -> tuple[Mapping[str, Any] | None, tuple[str, ...]]:
    if isinstance(value, Mapping):
        return value, ()
    if not isinstance(value, str) or not value.strip():
        return None, ("SGLang launch config hicache_storage_backend_extra_config must be a JSON object string",)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        return None, (f"SGLang launch config hicache_storage_backend_extra_config is not valid JSON: {exc}",)
    if not isinstance(parsed, Mapping):
        return None, ("SGLang launch config hicache_storage_backend_extra_config must decode to a JSON object",)
    return parsed, ()


def _validate_document_kv_extra_config(
    extra_config: Mapping[str, Any],
    *,
    expected_backend: ServingBackend,
    expected_record_type_prefix: str,
) -> tuple[str, ...]:
    issues: list[str] = []
    record_type = extra_config.get("document_kv.record_type")
    if not _is_non_empty_string(record_type):
        issues.append("document_kv.record_type must be a non-empty string")
    elif not str(record_type).startswith(expected_record_type_prefix):
        issues.append(
            f"document_kv.record_type must start with {expected_record_type_prefix!r} "
            f"for {expected_backend.value}"
        )
    schema_version = extra_config.get("document_kv.schema_version")
    if not _is_positive_int(schema_version):
        issues.append("document_kv.schema_version must be a positive integer")
    if extra_config.get("document_kv.backend") != expected_backend.value:
        issues.append(f"document_kv.backend must be {expected_backend.value!r}")
    if extra_config.get("document_kv.connector_package") != expected_backend.value:
        issues.append(f"document_kv.connector_package must be {expected_backend.value!r}")
    if not _is_non_empty_string(extra_config.get("document_kv.kv_injection_method")):
        issues.append("document_kv.kv_injection_method must be a non-empty string")
    if extra_config.get("document_kv.engine_handoff_record_type") != ENGINE_ADAPTER_HANDOFF_RECORD_TYPE:
        issues.append(
            "document_kv.engine_handoff_record_type must match "
            f"{ENGINE_ADAPTER_HANDOFF_RECORD_TYPE!r}"
        )
    if extra_config.get("document_kv.engine_handoff_schema_version") != ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION:
        issues.append(
            "document_kv.engine_handoff_schema_version must match "
            f"{ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION}"
        )
    if extra_config.get("document_kv.requires_native_runtime") is not True:
        issues.append("document_kv.requires_native_runtime must be true")
    if expected_backend == ServingBackend.VLLM:
        provider_factory_issues = _module_attribute_issues(
            extra_config.get(DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY),
            field_name=DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY,
        )
        issues.extend(provider_factory_issues)
    return tuple(issues)


def _build_document_kv_extra_config(
    backend: ServingBackend,
    *,
    record_type: str,
    schema_version: int,
    kv_injection_method: str,
    extra_config: Mapping[str, Any] | None,
    provider_factory: str | None = None,
    reserved_keys: set[str] | None = None,
) -> dict[str, Any]:
    record = {
        "document_kv.record_type": record_type,
        "document_kv.schema_version": schema_version,
        "document_kv.backend": backend.value,
        "document_kv.connector_package": backend.value,
        "document_kv.kv_injection_method": kv_injection_method,
        "document_kv.engine_handoff_record_type": ENGINE_ADAPTER_HANDOFF_RECORD_TYPE,
        "document_kv.engine_handoff_schema_version": ENGINE_ADAPTER_HANDOFF_SCHEMA_VERSION,
        "document_kv.requires_native_runtime": True,
    }
    record.update(_validated_extra_config(extra_config, reserved_keys=reserved_keys))
    if provider_factory is not None:
        _validate_module_attribute(provider_factory, field_name=DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY)
        record[DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY] = provider_factory
    record_type_prefix = (
        _VLLM_RECORD_TYPE_PREFIX if backend == ServingBackend.VLLM else _SGLANG_RECORD_TYPE_PREFIX
    )
    issues = _validate_document_kv_extra_config(
        record,
        expected_backend=backend,
        expected_record_type_prefix=record_type_prefix,
    )
    if issues:
        raise ValueError("; ".join(issues))
    return record


def _validated_extra_config(
    extra_config: Mapping[str, Any] | None,
    *,
    reserved_keys: set[str] | None = None,
) -> dict[str, Any]:
    if extra_config is None:
        return {}
    if not isinstance(extra_config, Mapping):
        raise TypeError("extra_config must be a mapping")
    reserved = reserved_keys or set()
    validated: dict[str, Any] = {}
    for key, value in extra_config.items():
        if not _is_non_empty_string(key):
            raise ValueError("extra_config keys must be non-empty strings")
        if key.startswith("document_kv."):
            raise ValueError("extra_config must not override reserved document_kv.* keys")
        if key in reserved:
            raise ValueError(f"extra_config must not override reserved key {key!r}")
        validated[str(key)] = value
    return validated


def _backend_from_optional_value(
    value: str | ServingBackend | None,
    *,
    field_name: str,
) -> ServingBackend | None:
    if value is None:
        return None
    return _backend_from_value(value, field_name=field_name)


def _backend_from_value(value: str | ServingBackend, *, field_name: str) -> ServingBackend:
    if isinstance(value, str):
        value = value.strip().lower()
    try:
        return value if isinstance(value, ServingBackend) else ServingBackend(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be one of {[backend.value for backend in ServingBackend]}") from exc


def _validated_backend_tuple(values: Sequence[str | ServingBackend], field_name: str) -> tuple[str, ...]:
    backends = tuple(_backend_from_value(value, field_name=field_name).value for value in values)
    if len(set(backends)) != len(backends):
        raise ValueError(f"{field_name} must not contain duplicates")
    return backends


def _validated_required_backends(values: Sequence[str | ServingBackend]) -> tuple[str, ...]:
    backends = _validated_backend_tuple(values, "required_backends")
    if not backends:
        raise ValueError("required_backends must be non-empty")
    return backends


def _validated_string_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    tuple_values = tuple(values)
    if any(not isinstance(value, str) or not value for value in tuple_values):
        raise ValueError(f"{field_name} entries must be non-empty strings")
    return tuple_values


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_module_attribute(value: Any, *, field_name: str) -> None:
    issues = _module_attribute_issues(value, field_name=field_name)
    if issues:
        raise ValueError("; ".join(issues))


def _module_attribute_issues(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not _is_non_empty_string(value):
        return (f"{field_name} must be a non-empty module:attribute string",)
    module_name, separator, attribute_name = str(value).strip().partition(":")
    if not separator or not module_name or not attribute_name:
        return (f"{field_name} must use module:attribute syntax",)
    module_segments = module_name.split(".")
    if any(not segment.isidentifier() for segment in module_segments):
        return (f"{field_name} module path must be a dotted Python module path",)
    if not attribute_name.isidentifier():
        return (f"{field_name} attribute must be a Python identifier",)
    return ()


def _module_path_issues(value: Any, *, field_name: str, expected_leaves: Sequence[str]) -> tuple[str, ...]:
    if not _is_non_empty_string(value):
        return (f"{field_name} must be a non-empty string",)
    module_path = str(value).strip()
    segments = module_path.split(".")
    if any(not segment.isidentifier() for segment in segments):
        return (f"{field_name} must be a dotted Python module path",)
    if segments[-1] not in expected_leaves:
        expected = ", ".join(repr(leaf) for leaf in expected_leaves)
        return (f"{field_name} must end with one of {expected}",)
    return ()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build validated document KV engine launch configs.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_vllm_parser(subparsers)
    _add_sglang_parser(subparsers)
    return parser


def _add_vllm_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("build-vllm", help="Build a vLLM transfer config sidecar.")
    parser.add_argument("--module-path", default=DEFAULT_VLLM_DOCUMENT_KV_MODULE_PATH)
    parser.add_argument("--kv-role", default="kv_both", choices=sorted(_VLLM_ALLOWED_KV_ROLES))
    parser.add_argument(
        "--provider-factory",
        default=DEFAULT_VLLM_DOCUMENT_KV_PROVIDER_FACTORY,
        help="vLLM document KV provider factory module:attribute path.",
    )
    _add_common_build_args(parser, default_record_type=DEFAULT_VLLM_ENGINE_LAUNCH_CONFIG_RECORD_TYPE)


def _add_sglang_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("build-sglang", help="Build an SGLang HiCache config sidecar.")
    parser.add_argument("--module-path", default=DEFAULT_SGLANG_DOCUMENT_KV_MODULE_PATH)
    _add_common_build_args(parser, default_record_type=DEFAULT_SGLANG_ENGINE_LAUNCH_CONFIG_RECORD_TYPE)


def _add_common_build_args(parser: argparse.ArgumentParser, *, default_record_type: str) -> None:
    parser.add_argument("--output-json", help="Optional output JSON path. Defaults to stdout.")
    parser.add_argument("--record-type", default=default_record_type)
    parser.add_argument("--schema-version", type=int, default=DEFAULT_ENGINE_LAUNCH_CONFIG_SCHEMA_VERSION)
    parser.add_argument("--kv-injection-method", default=DEFAULT_ENGINE_LAUNCH_CONFIG_KV_INJECTION_METHOD)
    parser.add_argument(
        "--extra-config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Optional adapter extra config entry. Values are parsed as JSON when possible.",
    )


def _extra_config_from_items(items: Sequence[str]) -> dict[str, Any]:
    extra_config: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("extra-config entries must use KEY=VALUE syntax")
        key, raw_value = item.split("=", 1)
        if not key:
            raise ValueError("extra-config keys must be non-empty")
        try:
            value: Any = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        extra_config[key] = value
    return extra_config


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
