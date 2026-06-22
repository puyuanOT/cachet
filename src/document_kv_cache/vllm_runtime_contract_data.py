"""Shared vLLM V1 KV connector lifecycle contract data."""

from __future__ import annotations

from collections.abc import Mapping
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
    "has_pending_push_work",
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

__all__ = [
    "VLLM_KV_CONNECTOR_V1_CONTRACT",
    "VLLM_KV_CONNECTOR_V1_CONTRACT_RECORD_TYPE",
    "VLLM_KV_CONNECTOR_V1_CONTRACT_SCHEMA_VERSION",
    "VLLM_KV_CONNECTOR_V1_DOC_URL",
    "VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS",
    "VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS",
    "VLLM_KV_CONNECTOR_V1_RUNTIME",
    "vllm_kv_connector_v1_contract_to_record",
]


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


VLLM_KV_CONNECTOR_V1_CONTRACT: Mapping[str, Any] = MappingProxyType(
    vllm_kv_connector_v1_contract_to_record()
)
