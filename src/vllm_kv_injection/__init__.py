"""Thin KV-injection primitives for a vLLM fork/extension."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_kv_injection.block_mapping import (
        BlockSpan,
        SegmentKey,
        map_segments_to_blocks,
        map_segments_to_reserved_blocks,
        plan_token_blocks,
    )
    from vllm_kv_injection.connector import InMemoryKVConnector, KVConnector, KVPayload
    from vllm_kv_injection.paged_kv_copy import (
        PagedKVLayout,
        inject_kv_cache_layer,
        slot_mapping_from_blocks,
    )
    from vllm_kv_injection.probe import (
        NativeVLLMConnectorFactory,
        NativeVLLMConnectorFactoryResult,
        VLLM_DOCUMENT_KV_NATIVE_PROBE_CONNECTOR_FACTORY,
        VLLM_NATIVE_PROBE_CONTRACT,
        VLLM_PROBE_METADATA_CONNECTOR_CLASS,
        VLLM_PROBE_METADATA_CONNECTOR_FACTORY,
        VLLM_PROBE_METADATA_NATIVE_RUNTIME,
        VLLM_PROBE_METADATA_PROBE,
        VLLM_PROBE_METADATA_PROBE_KIND,
        VLLM_PROBE_METADATA_PROVIDER_FACTORY,
        VLLM_PROBE_METADATA_REQUEST_ID,
        VLLM_PROBE_METADATA_RUNTIME_CONTRACT,
        VLLMConnectorProbe,
        build_document_kv_native_probe_connector,
        build_in_memory_debug_probe,
        build_native_connector_probe,
        load_native_connector_factory,
    )
    from vllm_kv_injection.protocol import KVCacheHandle, KVLayout, KVSegment
    from vllm_kv_injection.vllm_adapter import VLLMDocumentKVInjector, VLLMInjectedRequest
    from vllm_kv_injection.vllm_dynamic_connector import (
        DOCUMENT_KV_CONNECTOR_CLASS,
        DOCUMENT_KV_CONNECTOR_MODULE_PATH,
        DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY,
        DocumentKVConnector,
        DocumentKVProvider,
        NoOpDocumentKVProvider,
        VLLMSupportsHMA,
        load_document_kv_provider_factory,
        vllm_runtime_import_error,
    )
    from vllm_kv_injection.vllm_layer_mapping import (
        DOCUMENT_KV_VLLM_LAYER_MAPPING_RECORD_TYPE,
        DOCUMENT_KV_VLLM_LAYER_MAPPING_SCHEMA_VERSION,
        DocumentKVVLLMLayerMappingInspection,
        document_kv_vllm_layer_index_from_name,
        document_kv_vllm_layer_mapping_record_issues,
        document_kv_vllm_layer_mapping_to_record,
        document_kv_vllm_probe_layer_names,
        inspect_document_kv_vllm_layer_mapping,
        validate_document_kv_vllm_layer_mapping_record,
    )
    from vllm_kv_injection.vllm_native_provider import (
        DocumentKVConnectorMetadata,
        DocumentKVHandoffLoad,
        DocumentKVHandoffSource,
        DocumentKVLoadRequest,
        DocumentKVNativeProvider,
        DocumentKVNativeProbeConnector,
        KVTransferParamsDocumentKVSource,
        build_document_kv_provider,
    )
    from vllm_kv_injection.vllm_native_provider_constants import (
        DOCUMENT_KV_HANDOFF_JSON_PARAM,
        DOCUMENT_KV_HANDOFF_RECORD_PARAM,
        DOCUMENT_KV_HANDOFF_SOURCE_FACTORY_CONFIG_KEY,
        DOCUMENT_KV_NATIVE_PROVIDER_FACTORY,
        DOCUMENT_KV_PAYLOAD_CACHE_MAX_BYTES_CONFIG_KEY,
        DOCUMENT_KV_PAYLOAD_URI_PARAM,
        DOCUMENT_KV_TELEMETRY_JSONL_CONFIG_KEY,
    )
    from vllm_kv_injection.vllm_runtime_contract import (
        VLLMInstalledKVConnectorContract,
        VLLM_KV_CONNECTOR_V1_BASE_MODULE,
        VLLM_KV_CONNECTOR_V1_CONTRACT,
        VLLM_KV_CONNECTOR_V1_CONTRACT_RECORD_TYPE,
        VLLM_KV_CONNECTOR_V1_CONTRACT_SCHEMA_VERSION,
        VLLM_KV_CONNECTOR_V1_DOC_URL,
        VLLM_KV_CONNECTOR_V1_INSTALLED_CONTRACT_RECORD_TYPE,
        VLLM_KV_CONNECTOR_V1_INSTALLED_CONTRACT_SCHEMA_VERSION,
        VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS,
        VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS,
        VLLM_KV_CONNECTOR_V1_RUNTIME,
        inspect_installed_vllm_kv_connector_v1_contract,
        installed_vllm_kv_connector_v1_contract_record_issues,
        installed_vllm_kv_connector_v1_contract_to_record,
        validate_installed_vllm_kv_connector_v1_contract_record,
        validate_vllm_kv_connector_v1_contract_record,
        validate_vllm_kv_connector_v1_methods,
        vllm_kv_connector_v1_contract_record_issues,
        vllm_kv_connector_v1_contract_to_record,
        vllm_kv_connector_v1_method_issues,
    )
    from vllm_kv_injection.vllm_runtime_preflight import (
        DOCUMENT_KV_VLLM_RUNTIME_PREFLIGHT_RECORD_TYPE,
        DOCUMENT_KV_VLLM_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
        document_kv_vllm_runtime_preflight_record_issues,
        document_kv_vllm_runtime_preflight_to_record,
        validate_document_kv_vllm_runtime_preflight_record,
        write_document_kv_vllm_runtime_preflight_json,
    )
    from vllm_kv_injection.vllm_transfer_config import (
        DOCUMENT_KV_DEFAULT_ROLE,
        DOCUMENT_KV_TRANSFER_CONFIG_PREFIX,
        DOCUMENT_KV_TRANSFER_CONFIG_RECORD_TYPE,
        DOCUMENT_KV_TRANSFER_CONFIG_SCHEMA_VERSION,
        document_kv_transfer_config,
        document_kv_transfer_config_json,
    )

_EXPORT_MODULES = {
    "BlockSpan": "vllm_kv_injection.block_mapping",
    "SegmentKey": "vllm_kv_injection.block_mapping",
    "map_segments_to_blocks": "vllm_kv_injection.block_mapping",
    "map_segments_to_reserved_blocks": "vllm_kv_injection.block_mapping",
    "plan_token_blocks": "vllm_kv_injection.block_mapping",
    "InMemoryKVConnector": "vllm_kv_injection.connector",
    "KVConnector": "vllm_kv_injection.connector",
    "KVPayload": "vllm_kv_injection.connector",
    "PagedKVLayout": "vllm_kv_injection.paged_kv_copy",
    "inject_kv_cache_layer": "vllm_kv_injection.paged_kv_copy",
    "slot_mapping_from_blocks": "vllm_kv_injection.paged_kv_copy",
    "NativeVLLMConnectorFactory": "vllm_kv_injection.probe",
    "NativeVLLMConnectorFactoryResult": "vllm_kv_injection.probe",
    "VLLM_DOCUMENT_KV_NATIVE_PROBE_CONNECTOR_FACTORY": "vllm_kv_injection.probe",
    "VLLM_NATIVE_PROBE_CONTRACT": "vllm_kv_injection.probe",
    "VLLM_PROBE_METADATA_CONNECTOR_CLASS": "vllm_kv_injection.probe",
    "VLLM_PROBE_METADATA_CONNECTOR_FACTORY": "vllm_kv_injection.probe",
    "VLLM_PROBE_METADATA_NATIVE_RUNTIME": "vllm_kv_injection.probe",
    "VLLM_PROBE_METADATA_PROBE": "vllm_kv_injection.probe",
    "VLLM_PROBE_METADATA_PROBE_KIND": "vllm_kv_injection.probe",
    "VLLM_PROBE_METADATA_PROVIDER_FACTORY": "vllm_kv_injection.probe",
    "VLLM_PROBE_METADATA_REQUEST_ID": "vllm_kv_injection.probe",
    "VLLM_PROBE_METADATA_RUNTIME_CONTRACT": "vllm_kv_injection.probe",
    "VLLMConnectorProbe": "vllm_kv_injection.probe",
    "build_document_kv_native_probe_connector": "vllm_kv_injection.probe",
    "build_in_memory_debug_probe": "vllm_kv_injection.probe",
    "build_native_connector_probe": "vllm_kv_injection.probe",
    "load_native_connector_factory": "vllm_kv_injection.probe",
    "KVCacheHandle": "vllm_kv_injection.protocol",
    "KVLayout": "vllm_kv_injection.protocol",
    "KVSegment": "vllm_kv_injection.protocol",
    "VLLMDocumentKVInjector": "vllm_kv_injection.vllm_adapter",
    "VLLMInjectedRequest": "vllm_kv_injection.vllm_adapter",
    "DOCUMENT_KV_CONNECTOR_CLASS": "vllm_kv_injection.vllm_dynamic_connector",
    "DOCUMENT_KV_CONNECTOR_MODULE_PATH": "vllm_kv_injection.vllm_dynamic_connector",
    "DOCUMENT_KV_PROVIDER_FACTORY_CONFIG_KEY": "vllm_kv_injection.vllm_dynamic_connector",
    "DocumentKVConnector": "vllm_kv_injection.vllm_dynamic_connector",
    "DocumentKVProvider": "vllm_kv_injection.vllm_dynamic_connector",
    "NoOpDocumentKVProvider": "vllm_kv_injection.vllm_dynamic_connector",
    "VLLMSupportsHMA": "vllm_kv_injection.vllm_dynamic_connector",
    "load_document_kv_provider_factory": "vllm_kv_injection.vllm_dynamic_connector",
    "vllm_runtime_import_error": "vllm_kv_injection.vllm_dynamic_connector",
    "DOCUMENT_KV_HANDOFF_JSON_PARAM": "vllm_kv_injection.vllm_native_provider_constants",
    "DOCUMENT_KV_HANDOFF_RECORD_PARAM": "vllm_kv_injection.vllm_native_provider_constants",
    "DOCUMENT_KV_HANDOFF_SOURCE_FACTORY_CONFIG_KEY": "vllm_kv_injection.vllm_native_provider_constants",
    "DOCUMENT_KV_NATIVE_PROVIDER_FACTORY": "vllm_kv_injection.vllm_native_provider_constants",
    "DOCUMENT_KV_PAYLOAD_CACHE_MAX_BYTES_CONFIG_KEY": "vllm_kv_injection.vllm_transfer_config",
    "DOCUMENT_KV_PAYLOAD_URI_PARAM": "vllm_kv_injection.vllm_native_provider_constants",
    "DOCUMENT_KV_TELEMETRY_JSONL_CONFIG_KEY": "vllm_kv_injection.vllm_transfer_config",
    "DOCUMENT_KV_VLLM_LAYER_MAPPING_RECORD_TYPE": "vllm_kv_injection.vllm_layer_mapping",
    "DOCUMENT_KV_VLLM_LAYER_MAPPING_SCHEMA_VERSION": "vllm_kv_injection.vllm_layer_mapping",
    "DocumentKVConnectorMetadata": "vllm_kv_injection.vllm_native_provider",
    "DocumentKVHandoffLoad": "vllm_kv_injection.vllm_native_provider",
    "DocumentKVHandoffSource": "vllm_kv_injection.vllm_native_provider",
    "DocumentKVLoadRequest": "vllm_kv_injection.vllm_native_provider",
    "DocumentKVNativeProvider": "vllm_kv_injection.vllm_native_provider",
    "DocumentKVNativeProbeConnector": "vllm_kv_injection.vllm_native_provider",
    "DocumentKVVLLMLayerMappingInspection": "vllm_kv_injection.vllm_layer_mapping",
    "KVTransferParamsDocumentKVSource": "vllm_kv_injection.vllm_native_provider",
    "build_document_kv_provider": "vllm_kv_injection.vllm_native_provider",
    "document_kv_vllm_layer_index_from_name": "vllm_kv_injection.vllm_layer_mapping",
    "document_kv_vllm_layer_mapping_record_issues": "vllm_kv_injection.vllm_layer_mapping",
    "document_kv_vllm_layer_mapping_to_record": "vllm_kv_injection.vllm_layer_mapping",
    "document_kv_vllm_probe_layer_names": "vllm_kv_injection.vllm_layer_mapping",
    "inspect_document_kv_vllm_layer_mapping": "vllm_kv_injection.vllm_layer_mapping",
    "validate_document_kv_vllm_layer_mapping_record": "vllm_kv_injection.vllm_layer_mapping",
    "VLLMInstalledKVConnectorContract": "vllm_kv_injection.vllm_runtime_contract",
    "VLLM_KV_CONNECTOR_V1_BASE_MODULE": "vllm_kv_injection.vllm_runtime_contract",
    "VLLM_KV_CONNECTOR_V1_CONTRACT": "vllm_kv_injection.vllm_runtime_contract",
    "VLLM_KV_CONNECTOR_V1_CONTRACT_RECORD_TYPE": "vllm_kv_injection.vllm_runtime_contract",
    "VLLM_KV_CONNECTOR_V1_CONTRACT_SCHEMA_VERSION": "vllm_kv_injection.vllm_runtime_contract",
    "VLLM_KV_CONNECTOR_V1_DOC_URL": "vllm_kv_injection.vllm_runtime_contract",
    "VLLM_KV_CONNECTOR_V1_INSTALLED_CONTRACT_RECORD_TYPE": "vllm_kv_injection.vllm_runtime_contract",
    "VLLM_KV_CONNECTOR_V1_INSTALLED_CONTRACT_SCHEMA_VERSION": "vllm_kv_injection.vllm_runtime_contract",
    "VLLM_KV_CONNECTOR_V1_OPTIONAL_METHODS": "vllm_kv_injection.vllm_runtime_contract",
    "VLLM_KV_CONNECTOR_V1_REQUIRED_METHODS": "vllm_kv_injection.vllm_runtime_contract",
    "VLLM_KV_CONNECTOR_V1_RUNTIME": "vllm_kv_injection.vllm_runtime_contract",
    "inspect_installed_vllm_kv_connector_v1_contract": "vllm_kv_injection.vllm_runtime_contract",
    "installed_vllm_kv_connector_v1_contract_record_issues": "vllm_kv_injection.vllm_runtime_contract",
    "installed_vllm_kv_connector_v1_contract_to_record": "vllm_kv_injection.vllm_runtime_contract",
    "validate_installed_vllm_kv_connector_v1_contract_record": "vllm_kv_injection.vllm_runtime_contract",
    "validate_vllm_kv_connector_v1_contract_record": "vllm_kv_injection.vllm_runtime_contract",
    "validate_vllm_kv_connector_v1_methods": "vllm_kv_injection.vllm_runtime_contract",
    "vllm_kv_connector_v1_contract_record_issues": "vllm_kv_injection.vllm_runtime_contract",
    "vllm_kv_connector_v1_contract_to_record": "vllm_kv_injection.vllm_runtime_contract",
    "vllm_kv_connector_v1_method_issues": "vllm_kv_injection.vllm_runtime_contract",
    "DOCUMENT_KV_VLLM_RUNTIME_PREFLIGHT_RECORD_TYPE": "vllm_kv_injection.vllm_runtime_preflight",
    "DOCUMENT_KV_VLLM_RUNTIME_PREFLIGHT_SCHEMA_VERSION": "vllm_kv_injection.vllm_runtime_preflight",
    "document_kv_vllm_runtime_preflight_record_issues": "vllm_kv_injection.vllm_runtime_preflight",
    "document_kv_vllm_runtime_preflight_to_record": "vllm_kv_injection.vllm_runtime_preflight",
    "validate_document_kv_vllm_runtime_preflight_record": "vllm_kv_injection.vllm_runtime_preflight",
    "write_document_kv_vllm_runtime_preflight_json": "vllm_kv_injection.vllm_runtime_preflight",
    "DOCUMENT_KV_DEFAULT_ROLE": "vllm_kv_injection.vllm_transfer_config",
    "DOCUMENT_KV_TRANSFER_CONFIG_PREFIX": "vllm_kv_injection.vllm_transfer_config",
    "DOCUMENT_KV_TRANSFER_CONFIG_RECORD_TYPE": "vllm_kv_injection.vllm_transfer_config",
    "DOCUMENT_KV_TRANSFER_CONFIG_SCHEMA_VERSION": "vllm_kv_injection.vllm_transfer_config",
    "document_kv_transfer_config": "vllm_kv_injection.vllm_transfer_config",
    "document_kv_transfer_config_json": "vllm_kv_injection.vllm_transfer_config",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
