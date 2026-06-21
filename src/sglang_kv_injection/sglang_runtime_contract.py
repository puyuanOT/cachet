"""SGLang runtime-cache lifecycle contract diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from document_kv_cache.native_probe_factories import native_probe_adapter_contract_to_record

SGLANG_RUNTIME_CACHE_CONTRACT_RECORD_TYPE = "sglang_kv_injection.runtime_cache_contract.v1"
SGLANG_RUNTIME_CACHE_CONTRACT_SCHEMA_VERSION = 1
SGLANG_RUNTIME_CACHE_RUNTIME = "sglang-runtime-cache"
SGLANG_RUNTIME_CACHE_DOC_URL = "https://docs.sglang.io/docs/advanced_features/hicache_design"
SGLANG_RUNTIME_CACHE_REQUIRED_METHODS = (
    "stage",
    "attach",
    "release",
)
SGLANG_RUNTIME_CACHE_OPTIONAL_METHODS: tuple[str, ...] = ()
_SGLANG_RUNTIME_CACHE_CONTRACT_KEYS = frozenset(
    {
        "record_type",
        "schema_version",
        "runtime",
        "doc_url",
        "required_methods",
        "optional_methods",
        "handoff_contract",
    }
)


def sglang_runtime_cache_contract_to_record(
    *,
    handoff_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the SGLang runtime-cache lifecycle this adapter validates."""

    record: dict[str, Any] = {
        "record_type": SGLANG_RUNTIME_CACHE_CONTRACT_RECORD_TYPE,
        "schema_version": SGLANG_RUNTIME_CACHE_CONTRACT_SCHEMA_VERSION,
        "runtime": SGLANG_RUNTIME_CACHE_RUNTIME,
        "doc_url": SGLANG_RUNTIME_CACHE_DOC_URL,
        "required_methods": list(SGLANG_RUNTIME_CACHE_REQUIRED_METHODS),
        "optional_methods": list(SGLANG_RUNTIME_CACHE_OPTIONAL_METHODS),
    }
    if handoff_contract is not None:
        record["handoff_contract"] = dict(handoff_contract)
    return record


def sglang_runtime_cache_method_issues(connector: object) -> tuple[str, ...]:
    """Return missing-callable issues for a candidate SGLang runtime-cache bridge."""

    missing = [
        method_name
        for method_name in SGLANG_RUNTIME_CACHE_REQUIRED_METHODS
        if not callable(getattr(connector, method_name, None))
    ]
    if not missing:
        return ()
    return ("SGLang runtime-cache connector must provide callable methods: " + ", ".join(missing),)


def validate_sglang_runtime_cache_methods(connector: object) -> None:
    """Raise when a candidate connector does not expose required runtime hooks."""

    issues = sglang_runtime_cache_method_issues(connector)
    if issues:
        raise TypeError("; ".join(issues))


def validate_sglang_runtime_cache_contract_record(record: Mapping[str, Any]) -> None:
    """Validate a serialized SGLang runtime-cache contract diagnostic record."""

    issues = sglang_runtime_cache_contract_record_issues(record)
    if issues:
        raise ValueError("; ".join(issues))


def sglang_runtime_cache_contract_record_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return structural issues for an SGLang runtime-cache contract record."""

    issues: list[str] = []
    unexpected = sorted(str(key) for key in record if key not in _SGLANG_RUNTIME_CACHE_CONTRACT_KEYS)
    if unexpected:
        issues.append(f"SGLang runtime-cache contract has unsupported keys: {unexpected}")
    if record.get("record_type") != SGLANG_RUNTIME_CACHE_CONTRACT_RECORD_TYPE:
        issues.append(
            f"SGLang runtime-cache contract record_type must be {SGLANG_RUNTIME_CACHE_CONTRACT_RECORD_TYPE!r}"
        )
    if record.get("schema_version") != SGLANG_RUNTIME_CACHE_CONTRACT_SCHEMA_VERSION:
        issues.append(
            f"SGLang runtime-cache contract schema_version must be {SGLANG_RUNTIME_CACHE_CONTRACT_SCHEMA_VERSION}"
        )
    if record.get("runtime") != SGLANG_RUNTIME_CACHE_RUNTIME:
        issues.append(f"SGLang runtime-cache contract runtime must be {SGLANG_RUNTIME_CACHE_RUNTIME!r}")
    if record.get("doc_url") != SGLANG_RUNTIME_CACHE_DOC_URL:
        issues.append("SGLang runtime-cache contract doc_url must point at the SGLang HiCache docs")
    if _string_list(record.get("required_methods")) != list(SGLANG_RUNTIME_CACHE_REQUIRED_METHODS):
        issues.append("SGLang runtime-cache contract required_methods must match the package contract")
    if _string_list(record.get("optional_methods")) != list(SGLANG_RUNTIME_CACHE_OPTIONAL_METHODS):
        issues.append("SGLang runtime-cache contract optional_methods must match the package contract")
    handoff_contract = record.get("handoff_contract")
    if handoff_contract is None:
        return tuple(issues)
    if not isinstance(handoff_contract, Mapping):
        issues.append("SGLang runtime-cache contract handoff_contract must be an object when present")
    elif dict(handoff_contract) != native_probe_adapter_contract_to_record():
        issues.append("SGLang runtime-cache contract handoff_contract must match the Document KV native-probe contract")
    return tuple(issues)


def _string_list(value: Any) -> list[str] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        if all(isinstance(item, str) and item for item in items):
            return items
    return None


SGLANG_RUNTIME_CACHE_CONTRACT: Mapping[str, Any] = MappingProxyType(
    {
        "record_type": SGLANG_RUNTIME_CACHE_CONTRACT_RECORD_TYPE,
        "schema_version": SGLANG_RUNTIME_CACHE_CONTRACT_SCHEMA_VERSION,
        "runtime": SGLANG_RUNTIME_CACHE_RUNTIME,
        "doc_url": SGLANG_RUNTIME_CACHE_DOC_URL,
        "required_methods": SGLANG_RUNTIME_CACHE_REQUIRED_METHODS,
        "optional_methods": SGLANG_RUNTIME_CACHE_OPTIONAL_METHODS,
    }
)
