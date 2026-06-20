"""Public package facade for Document KV Cache.

New document-owned modules are landing here incrementally while existing
Databricks jobs migrate from :mod:`restaurant_kv_serving`. This facade gives new
users the document-generic import path and console script targets, while legacy
restaurant-specific names remain available as compatibility aliases.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LEGACY_PACKAGE = "restaurant_kv_serving"
_LEGACY_ROOT_EXPORTS = frozenset(
    {
        "ChunkType",
        "RestaurantKVRequest",
        "RestaurantKVService",
    }
)
_PUBLIC_SUBMODULES = frozenset(
    {
        "admission",
        "benchmark_plan",
        "benchmark_plan_executor",
        "benchmark_runner",
        "benchmarks",
        "cache",
        "databricks_engine_probe_job",
        "databricks_job",
        "databricks_runs",
        "databricks_storage_benchmark_job",
        "dataset_prep",
        "engine",
        "engine_adapters",
        "engine_probe",
        "engine_protocol",
        "kvpack",
        "live_server",
        "manifest",
        "materializer",
        "model_profiles",
        "models",
        "native_probe_factories",
        "openai_compatible",
        "planner",
        "pr_evidence",
        "release_bundle",
        "release_evidence",
        "service",
        "serving_env",
        "storage",
        "storage_benchmark",
        "template_resources",
        "vllm_smoke",
        "databricks_vllm_smoke_job",
        "workflow",
    }
)
_DOCUMENT_ROOT_EXPORTS = {
    "DEFAULT_AWS_G5_NODE_TYPE": ("document_kv_cache.databricks_job", "DEFAULT_AWS_G5_NODE_TYPE"),
    "DEFAULT_DATABRICKS_SPARK_VERSION": (
        "document_kv_cache.databricks_job",
        "DEFAULT_DATABRICKS_SPARK_VERSION",
    ),
    "DEFAULT_DATABRICKS_RUN_NAME": ("document_kv_cache.databricks_job", "DEFAULT_DATABRICKS_RUN_NAME"),
    "DEFAULT_DATABRICKS_TASK_KEY": ("document_kv_cache.databricks_job", "DEFAULT_DATABRICKS_TASK_KEY"),
    "DEFAULT_DATABRICKS_DATA_SECURITY_MODE": (
        "document_kv_cache.databricks_job",
        "DEFAULT_DATABRICKS_DATA_SECURITY_MODE",
    ),
    "DEDICATED_DATABRICKS_DATA_SECURITY_MODE": (
        "document_kv_cache.databricks_job",
        "DEDICATED_DATABRICKS_DATA_SECURITY_MODE",
    ),
    "SINGLE_USER_DATABRICKS_DATA_SECURITY_MODES": (
        "document_kv_cache.databricks_job",
        "SINGLE_USER_DATABRICKS_DATA_SECURITY_MODES",
    ),
    "DatabricksSingleNodeG5ClusterConfig": (
        "document_kv_cache.databricks_job",
        "DatabricksSingleNodeG5ClusterConfig",
    ),
    "DatabricksBenchmarkJobConfig": ("document_kv_cache.databricks_job", "DatabricksBenchmarkJobConfig"),
    "validate_aws_g5_node_type": ("document_kv_cache.databricks_job", "validate_aws_g5_node_type"),
    "build_single_node_g5_cluster": ("document_kv_cache.databricks_job", "build_single_node_g5_cluster"),
    "build_databricks_run_submit_payload": (
        "document_kv_cache.databricks_job",
        "build_databricks_run_submit_payload",
    ),
    "write_databricks_run_submit_json": (
        "document_kv_cache.databricks_job",
        "write_databricks_run_submit_json",
    ),
    "write_databricks_runner_script": ("document_kv_cache.databricks_job", "write_databricks_runner_script"),
    "DEFAULT_DATABRICKS_HOST_ENV": ("document_kv_cache.databricks_runs", "DEFAULT_DATABRICKS_HOST_ENV"),
    "DEFAULT_DATABRICKS_TOKEN_ENV": ("document_kv_cache.databricks_runs", "DEFAULT_DATABRICKS_TOKEN_ENV"),
    "DEFAULT_DATABRICKS_TIMEOUT_SECONDS": (
        "document_kv_cache.databricks_runs",
        "DEFAULT_DATABRICKS_TIMEOUT_SECONDS",
    ),
    "DATABRICKS_RUN_STATUS_RECORD_TYPE": (
        "document_kv_cache.databricks_runs",
        "DATABRICKS_RUN_STATUS_RECORD_TYPE",
    ),
    "DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE": (
        "document_kv_cache.databricks_runs",
        "DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE",
    ),
    "DatabricksWorkspaceConfig": ("document_kv_cache.databricks_runs", "DatabricksWorkspaceConfig"),
    "databricks_workspace_config_from_env": (
        "document_kv_cache.databricks_runs",
        "databricks_workspace_config_from_env",
    ),
    "submit_databricks_run": ("document_kv_cache.databricks_runs", "submit_databricks_run"),
    "get_databricks_run": ("document_kv_cache.databricks_runs", "get_databricks_run"),
    "summarize_databricks_run": ("document_kv_cache.databricks_runs", "summarize_databricks_run"),
    "summarize_databricks_run_submit_payload": (
        "document_kv_cache.databricks_runs",
        "summarize_databricks_run_submit_payload",
    ),
    "write_databricks_run_response_json": (
        "document_kv_cache.databricks_runs",
        "write_databricks_run_response_json",
    ),
    "read_databricks_run_submit_payload": (
        "document_kv_cache.databricks_runs",
        "read_databricks_run_submit_payload",
    ),
    "ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND": (
        "document_kv_cache.engine_probe",
        "ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND",
    ),
    "ENGINE_KV_PROBE_METADATA_HANDOFF_JSON": (
        "document_kv_cache.engine_probe",
        "ENGINE_KV_PROBE_METADATA_HANDOFF_JSON",
    ),
    "ENGINE_KV_PROBE_METADATA_PAYLOAD_URI": (
        "document_kv_cache.engine_probe",
        "ENGINE_KV_PROBE_METADATA_PAYLOAD_URI",
    ),
    "ENGINE_KV_PROBE_METADATA_PROBE_FACTORY": (
        "document_kv_cache.engine_probe",
        "ENGINE_KV_PROBE_METADATA_PROBE_FACTORY",
    ),
    "ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE": (
        "document_kv_cache.engine_probe",
        "ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE",
    ),
    "ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION": (
        "document_kv_cache.engine_probe",
        "ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION",
    ),
    "EngineKVProbeConfig": ("document_kv_cache.engine_probe", "EngineKVProbeConfig"),
    "EngineKVProbeFactory": ("document_kv_cache.engine_probe", "EngineKVProbeFactory"),
    "EngineKVProbeFactoryContext": ("document_kv_cache.engine_probe", "EngineKVProbeFactoryContext"),
    "EngineKVProbeFactoryResult": ("document_kv_cache.engine_probe", "EngineKVProbeFactoryResult"),
    "run_engine_kv_connector_probe": (
        "document_kv_cache.engine_probe",
        "run_engine_kv_connector_probe",
    ),
    "read_engine_adapter_payload": ("document_kv_cache.engine_probe", "read_engine_adapter_payload"),
    "write_engine_kv_connector_probe_result_json": (
        "document_kv_cache.engine_probe",
        "write_engine_kv_connector_probe_result_json",
    ),
    "load_engine_kv_probe_factory": ("document_kv_cache.engine_probe", "load_engine_kv_probe_factory"),
    "CacheTier": ("document_kv_cache.cache", "CacheTier"),
    "ChunkCacheResult": ("document_kv_cache.cache", "ChunkCacheResult"),
    "ChunkCacheStats": ("document_kv_cache.cache", "ChunkCacheStats"),
    "ByteLRU": ("document_kv_cache.cache", "ByteLRU"),
    "ChunkCache": ("document_kv_cache.cache", "ChunkCache"),
    "DocumentChunkType": ("document_kv_cache.models", "DocumentChunkType"),
    "DocumentChunkRole": ("document_kv_cache.models", "DocumentChunkRole"),
    "CacheGenerationMethod": ("document_kv_cache.models", "CacheGenerationMethod"),
    "DocumentChunkMap": ("document_kv_cache.models", "DocumentChunkMap"),
    "CacheChunkType": ("document_kv_cache.models", "CacheChunkType"),
    "CacheChunkTypeSet": ("document_kv_cache.models", "CacheChunkTypeSet"),
    "DOCUMENT_CHUNK_TYPES": ("document_kv_cache.models", "DOCUMENT_CHUNK_TYPES"),
    "LEGACY_RESTAURANT_CHUNK_TYPES": (
        "document_kv_cache.models",
        "LEGACY_RESTAURANT_CHUNK_TYPES",
    ),
    "KVCacheKey": ("document_kv_cache.models", "KVCacheKey"),
    "ChunkRef": ("document_kv_cache.models", "ChunkRef"),
    "DocumentKVRequest": ("document_kv_cache.models", "DocumentKVRequest"),
    "PlanSegment": ("document_kv_cache.models", "PlanSegment"),
    "MaterializationPlan": ("document_kv_cache.models", "MaterializationPlan"),
    "chunk_type_role": ("document_kv_cache.models", "chunk_type_role"),
    "chunk_type_sort_order": ("document_kv_cache.models", "chunk_type_sort_order"),
    "chunk_types_for_request": ("document_kv_cache.models", "chunk_types_for_request"),
    "ManifestStore": ("document_kv_cache.manifest", "ManifestStore"),
    "InMemoryManifestStore": ("document_kv_cache.manifest", "InMemoryManifestStore"),
    "CacheRequest": ("document_kv_cache.planner", "CacheRequest"),
    "CachePlanner": ("document_kv_cache.planner", "CachePlanner"),
    "MaterializedKV": ("document_kv_cache.materializer", "MaterializedKV"),
    "SegmentedMaterializedKV": ("document_kv_cache.materializer", "SegmentedMaterializedKV"),
    "KVMaterializer": ("document_kv_cache.materializer", "KVMaterializer"),
    "RangeReader": ("document_kv_cache.storage", "RangeReader"),
    "MemoryRangeReader": ("document_kv_cache.storage", "MemoryRangeReader"),
    "DiskRangeReader": ("document_kv_cache.storage", "DiskRangeReader"),
    "UnityCatalogVolumeRangeReader": (
        "document_kv_cache.storage",
        "UnityCatalogVolumeRangeReader",
    ),
    "RoutedRangeReader": ("document_kv_cache.storage", "RoutedRangeReader"),
    "local_path": ("document_kv_cache.storage", "local_path"),
    "unity_catalog_volume_path": ("document_kv_cache.storage", "unity_catalog_volume_path"),
    "is_real_uc_volume_root": ("document_kv_cache.storage", "is_real_uc_volume_root"),
}

_legacy_package = import_module(_LEGACY_PACKAGE)
__all__ = [
    name
    for name in getattr(_legacy_package, "__all__", ())
    if name not in _LEGACY_ROOT_EXPORTS
]
__all__.extend(name for name in _DOCUMENT_ROOT_EXPORTS if name not in __all__)


def __getattr__(name: str) -> Any:
    if name in _PUBLIC_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _DOCUMENT_ROOT_EXPORTS:
        module_name, symbol_name = _DOCUMENT_ROOT_EXPORTS[name]
        value = getattr(import_module(module_name), symbol_name)
        globals()[name] = value
        return value
    return getattr(_legacy_package, name)


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__, *_PUBLIC_SUBMODULES})
