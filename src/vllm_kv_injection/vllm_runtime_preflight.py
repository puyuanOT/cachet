"""Strict runtime preflight diagnostics for Cachet's vLLM native path."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vllm_kv_injection.vllm_native_provider import (
    DOCUMENT_KV_NATIVE_PROVIDER_FACTORY,
    DOCUMENT_KV_VLLM_LAYER_MAPPING_RECORD_TYPE,
    DocumentKVVLLMLayerMappingInspection,
    document_kv_vllm_layer_mapping_record_issues,
    document_kv_vllm_layer_mapping_to_record,
)
from vllm_kv_injection.vllm_runtime_contract import (
    VLLM_KV_CONNECTOR_V1_RUNTIME,
    installed_vllm_kv_connector_v1_contract_record_issues,
    installed_vllm_kv_connector_v1_contract_to_record,
)

DOCUMENT_KV_VLLM_RUNTIME_PREFLIGHT_RECORD_TYPE = "vllm_kv_injection.runtime_preflight.v1"
DOCUMENT_KV_VLLM_RUNTIME_PREFLIGHT_SCHEMA_VERSION = 1
_DOCUMENT_KV_VLLM_RUNTIME_PREFLIGHT_KEYS = frozenset(
    {
        "record_type",
        "schema_version",
        "runtime",
        "provider_factory",
        "installed_contract",
        "layer_mapping",
        "ok",
    }
)

__all__ = [
    "DOCUMENT_KV_VLLM_RUNTIME_PREFLIGHT_RECORD_TYPE",
    "DOCUMENT_KV_VLLM_RUNTIME_PREFLIGHT_SCHEMA_VERSION",
    "document_kv_vllm_runtime_preflight_record_issues",
    "document_kv_vllm_runtime_preflight_to_record",
    "validate_document_kv_vllm_runtime_preflight_record",
    "write_document_kv_vllm_runtime_preflight_json",
    "main",
]


def document_kv_vllm_runtime_preflight_to_record(
    layer_mapping: (
        DocumentKVVLLMLayerMappingInspection
        | Mapping[str, object]
        | Sequence[str]
        | None
    ) = None,
    *,
    installed_contract: Mapping[str, Any] | None = None,
    provider_factory: str = DOCUMENT_KV_NATIVE_PROVIDER_FACTORY,
) -> dict[str, Any]:
    """Serialize the target-runtime gates required before a native vLLM probe."""

    contract_record = _json_safe_mapping(
        installed_contract
        if installed_contract is not None
        else installed_vllm_kv_connector_v1_contract_to_record(),
        field_name="installed_contract",
    )
    layer_mapping_record = _layer_mapping_record(layer_mapping)
    ok = (
        provider_factory == DOCUMENT_KV_NATIVE_PROVIDER_FACTORY
        and contract_record.get("ok") is True
        and layer_mapping_record.get("ok") is True
        and not installed_vllm_kv_connector_v1_contract_record_issues(contract_record)
        and not document_kv_vllm_layer_mapping_record_issues(layer_mapping_record)
    )
    return {
        "record_type": DOCUMENT_KV_VLLM_RUNTIME_PREFLIGHT_RECORD_TYPE,
        "schema_version": DOCUMENT_KV_VLLM_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
        "runtime": VLLM_KV_CONNECTOR_V1_RUNTIME,
        "provider_factory": provider_factory,
        "installed_contract": contract_record,
        "layer_mapping": layer_mapping_record,
        "ok": ok,
    }


def validate_document_kv_vllm_runtime_preflight_record(record: Mapping[str, Any]) -> None:
    """Raise when a serialized vLLM native-runtime preflight is unsafe."""

    issues = document_kv_vllm_runtime_preflight_record_issues(record)
    if issues:
        raise ValueError("; ".join(issues))


def document_kv_vllm_runtime_preflight_record_issues(record: object) -> tuple[str, ...]:
    """Return structural and safety issues for a vLLM native-runtime preflight."""

    if not isinstance(record, Mapping):
        return ("vLLM runtime preflight record must be an object",)

    issues: list[str] = []
    unexpected = sorted(str(key) for key in record if key not in _DOCUMENT_KV_VLLM_RUNTIME_PREFLIGHT_KEYS)
    if unexpected:
        issues.append(f"vLLM runtime preflight record has unsupported keys: {unexpected}")
    if record.get("record_type") != DOCUMENT_KV_VLLM_RUNTIME_PREFLIGHT_RECORD_TYPE:
        issues.append(f"record_type must be {DOCUMENT_KV_VLLM_RUNTIME_PREFLIGHT_RECORD_TYPE!r}")
    if record.get("schema_version") != DOCUMENT_KV_VLLM_RUNTIME_PREFLIGHT_SCHEMA_VERSION:
        issues.append(f"schema_version must be {DOCUMENT_KV_VLLM_RUNTIME_PREFLIGHT_SCHEMA_VERSION}")
    if record.get("runtime") != VLLM_KV_CONNECTOR_V1_RUNTIME:
        issues.append(f"runtime must be {VLLM_KV_CONNECTOR_V1_RUNTIME!r}")
    if record.get("provider_factory") != DOCUMENT_KV_NATIVE_PROVIDER_FACTORY:
        issues.append(f"provider_factory must be {DOCUMENT_KV_NATIVE_PROVIDER_FACTORY!r}")

    contract_record = record.get("installed_contract")
    contract_safe = False
    if not isinstance(contract_record, Mapping):
        issues.append("installed_contract must be an object")
    else:
        contract_issues = installed_vllm_kv_connector_v1_contract_record_issues(contract_record)
        issues.extend(f"installed_contract.{issue}" for issue in contract_issues)
        if not contract_issues and contract_record.get("ok") is not True:
            issues.append("installed_contract.ok must be true for a safe vLLM runtime preflight")
        contract_safe = not contract_issues and contract_record.get("ok") is True

    layer_mapping_record = record.get("layer_mapping")
    layer_mapping_safe = False
    if not isinstance(layer_mapping_record, Mapping):
        issues.append("layer_mapping must be an object")
    else:
        layer_mapping_issues = document_kv_vllm_layer_mapping_record_issues(layer_mapping_record)
        issues.extend(f"layer_mapping.{issue}" for issue in layer_mapping_issues)
        layer_mapping_safe = not layer_mapping_issues and layer_mapping_record.get("ok") is True

    ok = record.get("ok")
    if type(ok) is not bool:
        issues.append("ok must be boolean")
    else:
        expected_ok = (
            record.get("provider_factory") == DOCUMENT_KV_NATIVE_PROVIDER_FACTORY
            and contract_safe
            and layer_mapping_safe
        )
        if ok != expected_ok:
            issues.append("ok must match provider factory, installed contract, and layer mapping safety")
        if ok is False:
            issues.append("ok must be true for a safe vLLM runtime preflight")
    return tuple(issues)


def write_document_kv_vllm_runtime_preflight_json(
    path: str | Path,
    layer_mapping: (
        DocumentKVVLLMLayerMappingInspection
        | Mapping[str, object]
        | Sequence[str]
        | None
    ) = None,
    *,
    installed_contract: Mapping[str, Any] | None = None,
) -> None:
    """Write a strict vLLM native-runtime preflight record as JSON."""

    record = document_kv_vllm_runtime_preflight_to_record(
        layer_mapping,
        installed_contract=installed_contract,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _layer_mapping_record(
    layer_mapping: (
        DocumentKVVLLMLayerMappingInspection
        | Mapping[str, object]
        | Sequence[str]
        | None
    ),
) -> dict[str, Any]:
    if layer_mapping is None:
        layer_mapping = ()
    if (
        isinstance(layer_mapping, Mapping)
        and layer_mapping.get("record_type") == DOCUMENT_KV_VLLM_LAYER_MAPPING_RECORD_TYPE
    ):
        return _json_safe_mapping(layer_mapping, field_name="layer_mapping")
    return document_kv_vllm_layer_mapping_to_record(layer_mapping)


def _json_safe_mapping(value: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized = json.loads(json.dumps(value))
    if not isinstance(normalized, dict):
        raise TypeError(f"{field_name} must serialize to a JSON object")
    return normalized


def _read_layer_names_json(path: str | Path) -> tuple[str, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("layer_names")
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise ValueError("layer names JSON must be a string array or an object with layer_names")
    layer_names = tuple(payload)
    if not all(isinstance(layer_name, str) and layer_name for layer_name in layer_names):
        raise ValueError("layer names JSON must contain non-empty strings")
    return layer_names


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write and validate the strict Cachet vLLM native-runtime preflight "
            "record required before provider-backed native probes."
        )
    )
    parser.add_argument(
        "--layer-name",
        action="append",
        default=None,
        help="Registered vLLM KV cache layer name. Repeat for every layer.",
    )
    parser.add_argument(
        "--layer-names-json",
        help="JSON string array, or object with layer_names, containing registered vLLM KV cache layer names.",
    )
    parser.add_argument("--output-json", help="Write the preflight record to this JSON file.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    layer_names = tuple(args.layer_name or ())
    if args.layer_names_json:
        layer_names = (*layer_names, *_read_layer_names_json(args.layer_names_json))
    record = document_kv_vllm_runtime_preflight_to_record(layer_names)
    output = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0 if not document_kv_vllm_runtime_preflight_record_issues(record) else 2


if __name__ == "__main__":
    raise SystemExit(main())
