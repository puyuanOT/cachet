"""vLLM V1 KV connector lifecycle contract diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

VLLM_KV_CONNECTOR_V1_CONTRACT_RECORD_TYPE = "vllm_kv_injection.kv_connector_v1_contract.v1"
VLLM_KV_CONNECTOR_V1_CONTRACT_SCHEMA_VERSION = 1
VLLM_KV_CONNECTOR_V1_RUNTIME = "vllm-kv-connector-v1"
VLLM_KV_CONNECTOR_V1_DOC_URL = (
    "https://docs.vllm.ai/en/stable/api/vllm/distributed/kv_transfer/kv_connector/v1/"
)
VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS = (
    "get_num_new_matched_tokens",
    "update_state_after_alloc",
    "build_connector_meta",
    "register_kv_caches",
    "start_load_kv",
    "wait_for_layer_load",
    "save_kv_layer",
    "wait_for_save",
    "request_finished",
    "request_finished_all_groups",
)
VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS = (
    "bind_connector_metadata",
    "bind_gpu_block_pool",
    "build_connector_worker_meta",
    "build_kv_connector_stats",
    "build_prom_metrics",
    "clear_connector_metadata",
    "get_block_ids_with_load_errors",
    "get_finished",
    "get_finished_count",
    "get_handshake_metadata",
    "get_kv_connector_kv_cache_events",
    "get_kv_connector_stats",
    "get_required_kvcache_layout",
    "handle_preemptions",
    "has_connector_metadata",
    "on_new_request",
    "register_cross_layers_kv_cache",
    "requires_piecewise_for_cudagraph",
    "reset_cache",
    "set_host_xfer_buffer_ops",
    "set_xfer_handshake_metadata",
    "set_xfer_handshake_metadata_pp_aware",
    "shutdown",
    "take_events",
    "update_connector_output",
)
_VLLM_KV_CONNECTOR_V1_CONTRACT_KEYS = frozenset(
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


def vllm_kv_connector_v1_contract_to_record(
    *,
    handoff_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the vLLM V1 lifecycle that a native adapter must cover."""

    record: dict[str, Any] = {
        "record_type": VLLM_KV_CONNECTOR_V1_CONTRACT_RECORD_TYPE,
        "schema_version": VLLM_KV_CONNECTOR_V1_CONTRACT_SCHEMA_VERSION,
        "runtime": VLLM_KV_CONNECTOR_V1_RUNTIME,
        "doc_url": VLLM_KV_CONNECTOR_V1_DOC_URL,
        "required_methods": list(VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS),
        "optional_methods": list(VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS),
    }
    if handoff_contract is not None:
        record["handoff_contract"] = dict(handoff_contract)
    return record


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


VLLM_KV_CONNECTOR_V1_CONTRACT: Mapping[str, Any] = MappingProxyType(
    vllm_kv_connector_v1_contract_to_record()
)
