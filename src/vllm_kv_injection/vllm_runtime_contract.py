"""vLLM V1 KV connector lifecycle contract diagnostics."""

from __future__ import annotations

import importlib
import importlib.metadata as package_metadata
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from document_kv_cache.vllm_runtime_contract_data import (
    VLLM_KV_CONNECTOR_V1_BASE_SOURCE_SHA256,
    VLLM_KV_CONNECTOR_V1_CONTRACT as VLLM_KV_CONNECTOR_V1_CONTRACT,
    VLLM_KV_CONNECTOR_V1_CONTRACT_RECORD_TYPE,
    VLLM_KV_CONNECTOR_V1_CONTRACT_SCHEMA_VERSION,
    VLLM_KV_CONNECTOR_V1_DOC_URL,
    VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS,
    VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS,
    VLLM_KV_CONNECTOR_V1_RUNTIME,
    VLLM_KV_CONNECTOR_V1_TARGET_PACKAGE_NAME,
    VLLM_KV_CONNECTOR_V1_TARGET_PACKAGE_VERSION,
    vllm_kv_connector_v1_contract_to_record as vllm_kv_connector_v1_contract_to_record,
)
VLLM_KV_CONNECTOR_V1_INSTALLED_CONTRACT_RECORD_TYPE = (
    "vllm_kv_injection.installed_kv_connector_v1_contract.v3"
)
VLLM_KV_CONNECTOR_V1_INSTALLED_CONTRACT_SCHEMA_VERSION = 3
VLLM_KV_CONNECTOR_V1_BASE_MODULE = "vllm.distributed.kv_transfer.kv_connector.v1.base"
_VLLM_KV_CONNECTOR_V1_ALLOWED_PROPERTIES = (
    "prefer_cross_layer_blocks",
    "requires_kv_delivery",
    "role",
)
_VLLM_KV_CONNECTOR_V1_CONTRACT_KEYS = frozenset(
    {
        "record_type",
        "schema_version",
        "runtime",
        "target_package_name",
        "target_package_version",
        "base_source_sha256",
        "doc_url",
        "required_methods",
        "optional_methods",
        "handoff_contract",
    }
)
_VLLM_KV_CONNECTOR_V1_INSTALLED_CONTRACT_KEYS = frozenset(
    {
        "record_type",
        "schema_version",
        "runtime",
        "base_module",
        "base_source_sha256",
        "expected_base_source_sha256",
        "base_source_matches",
        "package_name",
        "package_version",
        "expected_package_version",
        "version_matches",
        "importable",
        "import_error_type",
        "import_error",
        "required_methods",
        "optional_methods",
        "allowed_properties",
        "installed_methods",
        "installed_properties",
        "missing_required_methods",
        "missing_optional_methods",
        "extra_installed_methods",
        "extra_installed_properties",
        "ok",
    }
)


@dataclass(frozen=True, slots=True)
class VLLMInstalledKVConnectorContract:
    """Comparison between Cachet's vLLM V1 contract and an installed runtime."""

    package_version: str | None
    importable: bool
    installed_methods: tuple[str, ...]
    installed_properties: tuple[str, ...]
    base_source_sha256: str | None = None
    import_error_type: str | None = None
    import_error: str | None = None

    @property
    def missing_required_methods(self) -> tuple[str, ...]:
        return _missing(VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS, self.installed_methods)

    @property
    def missing_optional_methods(self) -> tuple[str, ...]:
        return _missing(VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS, self.installed_methods)

    @property
    def extra_installed_methods(self) -> tuple[str, ...]:
        allowed = (*VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS, *VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS)
        return _extra(self.installed_methods, allowed)

    @property
    def extra_installed_properties(self) -> tuple[str, ...]:
        return _extra(self.installed_properties, _VLLM_KV_CONNECTOR_V1_ALLOWED_PROPERTIES)

    @property
    def version_matches(self) -> bool:
        return self.package_version == VLLM_KV_CONNECTOR_V1_TARGET_PACKAGE_VERSION

    @property
    def base_source_matches(self) -> bool:
        return self.base_source_sha256 == VLLM_KV_CONNECTOR_V1_BASE_SOURCE_SHA256

    @property
    def ok(self) -> bool:
        return (
            self.importable
            and self.version_matches
            and self.base_source_matches
            and not self.missing_required_methods
            and not self.extra_installed_methods
            and not self.extra_installed_properties
        )


def vllm_kv_connector_v1_method_issues(connector: object) -> tuple[str, ...]:
    """Return missing-callable issues for a candidate vLLM V1 KV connector."""

    missing = [
        method_name
        for method_name in VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS
        if not callable(getattr(connector, method_name, None))
    ]
    if not missing:
        return ()
    return ("vLLM V1 KV connector must provide callable methods: " + ", ".join(missing),)


def validate_vllm_kv_connector_v1_methods(connector: object) -> None:
    """Raise when a candidate connector does not expose required vLLM V1 hooks."""

    issues = vllm_kv_connector_v1_method_issues(connector)
    if issues:
        raise TypeError("; ".join(issues))


def validate_vllm_kv_connector_v1_contract_record(record: Mapping[str, Any]) -> None:
    """Validate a serialized vLLM V1 runtime-contract diagnostic record."""

    issues = vllm_kv_connector_v1_contract_record_issues(record)
    if issues:
        raise ValueError("; ".join(issues))


def inspect_installed_vllm_kv_connector_v1_contract() -> VLLMInstalledKVConnectorContract:
    """Inspect the installed vLLM runtime's V1 KV connector public API."""

    package_version = _package_version(VLLM_KV_CONNECTOR_V1_TARGET_PACKAGE_NAME)
    try:
        base_module = importlib.import_module(VLLM_KV_CONNECTOR_V1_BASE_MODULE)
        base_class = getattr(base_module, "KVConnectorBase_V1")
        hma_class = getattr(base_module, "SupportsHMA")
    except Exception as exc:  # pragma: no cover - exact dependency failures vary by runtime.
        return VLLMInstalledKVConnectorContract(
            package_version=package_version,
            importable=False,
            installed_methods=(),
            installed_properties=(),
            import_error_type=type(exc).__name__,
            import_error=str(exc),
        )
    methods, properties = _vllm_kv_connector_public_surface(base_class, hma_class)
    return VLLMInstalledKVConnectorContract(
        package_version=package_version,
        importable=True,
        installed_methods=methods,
        installed_properties=properties,
        base_source_sha256=_module_source_sha256(base_module),
    )


def installed_vllm_kv_connector_v1_contract_to_record(
    inspection: VLLMInstalledKVConnectorContract | None = None,
) -> dict[str, Any]:
    """Serialize installed vLLM KV connector contract drift diagnostics."""

    observed = inspection or inspect_installed_vllm_kv_connector_v1_contract()
    return {
        "record_type": VLLM_KV_CONNECTOR_V1_INSTALLED_CONTRACT_RECORD_TYPE,
        "schema_version": VLLM_KV_CONNECTOR_V1_INSTALLED_CONTRACT_SCHEMA_VERSION,
        "runtime": VLLM_KV_CONNECTOR_V1_RUNTIME,
        "base_module": VLLM_KV_CONNECTOR_V1_BASE_MODULE,
        "base_source_sha256": observed.base_source_sha256,
        "expected_base_source_sha256": VLLM_KV_CONNECTOR_V1_BASE_SOURCE_SHA256,
        "base_source_matches": observed.base_source_matches,
        "package_name": VLLM_KV_CONNECTOR_V1_TARGET_PACKAGE_NAME,
        "package_version": observed.package_version,
        "expected_package_version": VLLM_KV_CONNECTOR_V1_TARGET_PACKAGE_VERSION,
        "version_matches": observed.version_matches,
        "importable": observed.importable,
        "import_error_type": observed.import_error_type,
        "import_error": observed.import_error,
        "required_methods": list(VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS),
        "optional_methods": list(VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS),
        "allowed_properties": list(_VLLM_KV_CONNECTOR_V1_ALLOWED_PROPERTIES),
        "installed_methods": list(observed.installed_methods),
        "installed_properties": list(observed.installed_properties),
        "missing_required_methods": list(observed.missing_required_methods),
        "missing_optional_methods": list(observed.missing_optional_methods),
        "extra_installed_methods": list(observed.extra_installed_methods),
        "extra_installed_properties": list(observed.extra_installed_properties),
        "ok": observed.ok,
    }


def validate_installed_vllm_kv_connector_v1_contract_record(record: Mapping[str, Any]) -> None:
    """Validate an installed vLLM KV connector contract diagnostic record."""

    issues = installed_vllm_kv_connector_v1_contract_record_issues(record)
    if issues:
        raise ValueError("; ".join(issues))


def installed_vllm_kv_connector_v1_contract_record_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return structural issues for an installed-runtime diagnostic record."""

    issues: list[str] = []
    unexpected = sorted(str(key) for key in record if key not in _VLLM_KV_CONNECTOR_V1_INSTALLED_CONTRACT_KEYS)
    if unexpected:
        issues.append(f"installed vLLM KV connector contract has unsupported keys: {unexpected}")
    if record.get("record_type") != VLLM_KV_CONNECTOR_V1_INSTALLED_CONTRACT_RECORD_TYPE:
        issues.append(
            "installed vLLM KV connector contract record_type must be "
            f"{VLLM_KV_CONNECTOR_V1_INSTALLED_CONTRACT_RECORD_TYPE!r}"
        )
    if record.get("schema_version") != VLLM_KV_CONNECTOR_V1_INSTALLED_CONTRACT_SCHEMA_VERSION:
        issues.append(
            "installed vLLM KV connector contract schema_version must be "
            f"{VLLM_KV_CONNECTOR_V1_INSTALLED_CONTRACT_SCHEMA_VERSION}"
        )
    if record.get("runtime") != VLLM_KV_CONNECTOR_V1_RUNTIME:
        issues.append(f"installed vLLM KV connector contract runtime must be {VLLM_KV_CONNECTOR_V1_RUNTIME!r}")
    if record.get("base_module") != VLLM_KV_CONNECTOR_V1_BASE_MODULE:
        issues.append("installed vLLM KV connector contract base_module must point at KVConnectorBase_V1")
    base_source_sha256 = record.get("base_source_sha256")
    if base_source_sha256 is not None and (
        not isinstance(base_source_sha256, str)
        or len(base_source_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in base_source_sha256
        )
    ):
        issues.append(
            "installed vLLM KV connector contract base_source_sha256 must be a lowercase SHA-256 or null"
        )
    if (
        record.get("expected_base_source_sha256")
        != VLLM_KV_CONNECTOR_V1_BASE_SOURCE_SHA256
    ):
        issues.append(
            "installed vLLM KV connector contract expected_base_source_sha256 must match the package contract"
        )
    base_source_matches = record.get("base_source_matches")
    if type(base_source_matches) is not bool:
        issues.append(
            "installed vLLM KV connector contract base_source_matches must be boolean"
        )
    elif base_source_matches != (
        base_source_sha256 == VLLM_KV_CONNECTOR_V1_BASE_SOURCE_SHA256
    ):
        issues.append(
            "installed vLLM KV connector contract base_source_matches must match base_source_sha256"
        )
    if record.get("package_name") != VLLM_KV_CONNECTOR_V1_TARGET_PACKAGE_NAME:
        issues.append(
            "installed vLLM KV connector contract package_name must be "
            f"{VLLM_KV_CONNECTOR_V1_TARGET_PACKAGE_NAME!r}"
        )
    package_version = record.get("package_version")
    if package_version is not None and (not isinstance(package_version, str) or not package_version):
        issues.append("installed vLLM KV connector contract package_version must be a non-empty string or null")
    if record.get("expected_package_version") != VLLM_KV_CONNECTOR_V1_TARGET_PACKAGE_VERSION:
        issues.append(
            "installed vLLM KV connector contract expected_package_version must be "
            f"{VLLM_KV_CONNECTOR_V1_TARGET_PACKAGE_VERSION!r}"
        )
    version_matches = record.get("version_matches")
    if type(version_matches) is not bool:
        issues.append("installed vLLM KV connector contract version_matches must be boolean")
    elif version_matches != (
        package_version == VLLM_KV_CONNECTOR_V1_TARGET_PACKAGE_VERSION
    ):
        issues.append(
            "installed vLLM KV connector contract version_matches must match package_version"
        )
    importable = record.get("importable")
    if type(importable) is not bool:
        issues.append("installed vLLM KV connector contract importable must be boolean")
    ok = record.get("ok")
    if type(ok) is not bool:
        issues.append("installed vLLM KV connector contract ok must be boolean")
    if record.get("import_error_type") is not None and not isinstance(record.get("import_error_type"), str):
        issues.append("installed vLLM KV connector contract import_error_type must be string or null")
    if record.get("import_error") is not None and not isinstance(record.get("import_error"), str):
        issues.append("installed vLLM KV connector contract import_error must be string or null")
    string_lists: dict[str, list[str] | None] = {}
    for field_name in (
        "required_methods",
        "optional_methods",
        "allowed_properties",
        "installed_methods",
        "installed_properties",
        "missing_required_methods",
        "missing_optional_methods",
        "extra_installed_methods",
        "extra_installed_properties",
    ):
        string_lists[field_name] = _string_list(record.get(field_name))
        if string_lists[field_name] is None:
            issues.append(f"installed vLLM KV connector contract {field_name} must be a string array")
    if string_lists["required_methods"] != list(VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS):
        issues.append("installed vLLM KV connector contract required_methods must match the package contract")
    if string_lists["optional_methods"] != list(VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS):
        issues.append("installed vLLM KV connector contract optional_methods must match the package contract")
    if string_lists["allowed_properties"] != list(_VLLM_KV_CONNECTOR_V1_ALLOWED_PROPERTIES):
        issues.append("installed vLLM KV connector contract allowed_properties must match the package contract")
    installed_methods = string_lists["installed_methods"]
    installed_properties = string_lists["installed_properties"]
    missing_required_methods = string_lists["missing_required_methods"]
    missing_optional_methods = string_lists["missing_optional_methods"]
    extra_installed_methods = string_lists["extra_installed_methods"]
    extra_installed_properties = string_lists["extra_installed_properties"]
    if installed_methods is not None:
        expected_missing_required = list(_missing(VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS, installed_methods))
        expected_missing_optional = list(_missing(VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS, installed_methods))
        expected_method_names = (
            *VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS,
            *VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS,
        )
        expected_extra_methods = list(
            _extra(installed_methods, expected_method_names)
        )
        if missing_required_methods is not None and missing_required_methods != expected_missing_required:
            issues.append(
                "installed vLLM KV connector contract missing_required_methods must match installed_methods"
            )
        if missing_optional_methods is not None and missing_optional_methods != expected_missing_optional:
            issues.append(
                "installed vLLM KV connector contract missing_optional_methods must match installed_methods"
            )
        if extra_installed_methods is not None and extra_installed_methods != expected_extra_methods:
            issues.append("installed vLLM KV connector contract extra_installed_methods must match installed_methods")
    else:
        expected_missing_required = None
        expected_extra_methods = None
    if installed_properties is not None:
        expected_extra_properties = list(
            _extra(installed_properties, _VLLM_KV_CONNECTOR_V1_ALLOWED_PROPERTIES)
        )
        if extra_installed_properties is not None and extra_installed_properties != expected_extra_properties:
            issues.append(
                "installed vLLM KV connector contract extra_installed_properties must match installed_properties"
            )
    else:
        expected_extra_properties = None
    if (
        type(importable) is bool
        and type(ok) is bool
        and expected_missing_required is not None
        and expected_extra_methods is not None
        and expected_extra_properties is not None
    ):
        expected_ok = (
            importable
            and version_matches is True
            and base_source_matches is True
            and not expected_missing_required
            and not expected_extra_methods
            and not expected_extra_properties
        )
        if ok != expected_ok:
            issues.append("installed vLLM KV connector contract ok must match importable and detected drift")
    return tuple(issues)


def vllm_kv_connector_v1_contract_record_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return structural issues for a vLLM V1 runtime-contract record."""

    issues: list[str] = []
    unexpected = sorted(str(key) for key in record if key not in _VLLM_KV_CONNECTOR_V1_CONTRACT_KEYS)
    if unexpected:
        issues.append(f"vLLM V1 KV connector contract has unsupported keys: {unexpected}")
    if record.get("record_type") != VLLM_KV_CONNECTOR_V1_CONTRACT_RECORD_TYPE:
        issues.append(
            f"vLLM V1 KV connector contract record_type must be {VLLM_KV_CONNECTOR_V1_CONTRACT_RECORD_TYPE!r}"
        )
    if record.get("schema_version") != VLLM_KV_CONNECTOR_V1_CONTRACT_SCHEMA_VERSION:
        issues.append(
            f"vLLM V1 KV connector contract schema_version must be {VLLM_KV_CONNECTOR_V1_CONTRACT_SCHEMA_VERSION}"
        )
    if record.get("runtime") != VLLM_KV_CONNECTOR_V1_RUNTIME:
        issues.append(f"vLLM V1 KV connector contract runtime must be {VLLM_KV_CONNECTOR_V1_RUNTIME!r}")
    if record.get("target_package_name") != VLLM_KV_CONNECTOR_V1_TARGET_PACKAGE_NAME:
        issues.append(
            "vLLM V1 KV connector contract target_package_name must be "
            f"{VLLM_KV_CONNECTOR_V1_TARGET_PACKAGE_NAME!r}"
        )
    if record.get("target_package_version") != VLLM_KV_CONNECTOR_V1_TARGET_PACKAGE_VERSION:
        issues.append(
            "vLLM V1 KV connector contract target_package_version must be "
            f"{VLLM_KV_CONNECTOR_V1_TARGET_PACKAGE_VERSION!r}"
        )
    if record.get("base_source_sha256") != VLLM_KV_CONNECTOR_V1_BASE_SOURCE_SHA256:
        issues.append(
            "vLLM V1 KV connector contract base_source_sha256 must match the approved 0.27.1 source"
        )
    if record.get("doc_url") != VLLM_KV_CONNECTOR_V1_DOC_URL:
        issues.append("vLLM V1 KV connector contract doc_url must point at the vLLM V1 KV connector docs")
    if _string_list(record.get("required_methods")) != list(VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS):
        issues.append("vLLM V1 KV connector contract required_methods must match the package contract")
    if _string_list(record.get("optional_methods")) != list(VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS):
        issues.append("vLLM V1 KV connector contract optional_methods must match the package contract")
    handoff_contract = record.get("handoff_contract")
    if handoff_contract is not None and not isinstance(handoff_contract, Mapping):
        issues.append("vLLM V1 KV connector contract handoff_contract must be an object when present")
    return tuple(issues)


def _string_list(value: Any) -> list[str] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        if all(isinstance(item, str) and item for item in items):
            return items
    return None


def _vllm_kv_connector_public_surface(base_class: type, hma_class: type) -> tuple[tuple[str, ...], tuple[str, ...]]:
    methods: set[str] = set()
    properties: set[str] = set()
    for inspected_class in (base_class, hma_class):
        class_methods, class_properties = _class_public_surface(inspected_class)
        methods.update(class_methods)
        properties.update(class_properties)
    return tuple(sorted(methods)), tuple(sorted(properties))


def _class_public_surface(value: type) -> tuple[tuple[str, ...], tuple[str, ...]]:
    methods: list[str] = []
    properties: list[str] = []
    for name, attribute in vars(value).items():
        if name.startswith("_"):
            continue
        if isinstance(attribute, property):
            properties.append(name)
        elif isinstance(attribute, (classmethod, staticmethod)) or callable(attribute):
            methods.append(name)
    return tuple(sorted(methods)), tuple(sorted(properties))


def _module_source_sha256(module: object) -> str | None:
    source_path = getattr(module, "__file__", None)
    if not isinstance(source_path, str) or not source_path:
        return None
    path = Path(source_path)
    if path.suffix in {".pyc", ".pyo"}:
        source_candidate = path.with_suffix(".py")
        if source_candidate.is_file():
            path = source_candidate
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _missing(expected: Sequence[str], observed: Sequence[str]) -> tuple[str, ...]:
    observed_set = set(observed)
    return tuple(item for item in expected if item not in observed_set)


def _extra(observed: Sequence[str], expected: Sequence[str]) -> tuple[str, ...]:
    expected_set = set(expected)
    return tuple(item for item in observed if item not in expected_set)


def _package_version(package_name: str) -> str | None:
    try:
        return package_metadata.version(package_name)
    except package_metadata.PackageNotFoundError:
        return None
