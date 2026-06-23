"""Public package facade for Document KV Cache.

New document-owned modules are landing here incrementally while existing
Databricks jobs migrate from the legacy package. This facade gives new users the
document-generic import path and console script targets, while legacy
restaurant-specific names remain available as source-checkout compatibility
aliases when the legacy package is present.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LEGACY_PACKAGE = "restaurant" "_kv_serving"
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
        "adapter_scaffold",
        "benchmark_plan",
        "benchmark_plan_executor",
        "benchmark_handoff_bundles",
        "benchmark_handoffs",
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
        "engine_launch_config",
        "engine_probe",
        "engine_protocol",
        "github_governance",
        "kvpack",
        "legacy_compatibility",
        "live_server",
        "manifest",
        "materializer",
        "model_profiles",
        "models",
        "native_probe_factories",
        "openai_compatible",
        "planner",
        "probe_fixtures",
        "pr_evidence",
        "release_bundle",
        "release_evidence",
        "repository_hygiene",
        "service",
        "serving_env",
        "sglang_runtime_preflight",
        "storage",
        "storage_benchmark",
        "template_resources",
        "transformers_generator",
        "vllm_runtime_contract_data",
        "vllm_runtime_preflight",
        "vllm_smoke",
        "databricks_vllm_smoke_job",
        "workflow",
    }
)
_DOCUMENT_ROOT_EXPORTS = {
    "PLAN_VERSION": ("document_kv_cache.benchmark_plan", "PLAN_VERSION"),
    "ENGINE_PROBE_TARGETS_RECORD_TYPE": (
        "document_kv_cache.benchmark_plan",
        "ENGINE_PROBE_TARGETS_RECORD_TYPE",
    ),
    "ENGINE_PROBE_TARGETS_SCHEMA_VERSION": (
        "document_kv_cache.benchmark_plan",
        "ENGINE_PROBE_TARGETS_SCHEMA_VERSION",
    ),
    "BenchmarkDatasetPath": ("document_kv_cache.benchmark_plan", "BenchmarkDatasetPath"),
    "BenchmarkCommand": ("document_kv_cache.benchmark_plan", "BenchmarkCommand"),
    "StorageBenchmarkPlanConfig": (
        "document_kv_cache.benchmark_plan",
        "StorageBenchmarkPlanConfig",
    ),
    "EngineProbePlanConfig": ("document_kv_cache.benchmark_plan", "EngineProbePlanConfig"),
    "ReleaseEvidencePlanConfig": (
        "document_kv_cache.benchmark_plan",
        "ReleaseEvidencePlanConfig",
    ),
    "ReleaseBundlePlanConfig": ("document_kv_cache.benchmark_plan", "ReleaseBundlePlanConfig"),
    "BenchmarkPlanConfig": ("document_kv_cache.benchmark_plan", "BenchmarkPlanConfig"),
    "BenchmarkJobPlan": ("document_kv_cache.benchmark_plan", "BenchmarkJobPlan"),
    "build_v1_benchmark_plan": ("document_kv_cache.benchmark_plan", "build_v1_benchmark_plan"),
    "benchmark_job_plan_to_record": (
        "document_kv_cache.benchmark_plan",
        "benchmark_job_plan_to_record",
    ),
    "engine_probe_targets_to_record": (
        "document_kv_cache.benchmark_plan",
        "engine_probe_targets_to_record",
    ),
    "write_benchmark_job_plan_json": (
        "document_kv_cache.benchmark_plan",
        "write_benchmark_job_plan_json",
    ),
    "write_benchmark_job_plan_shell": (
        "document_kv_cache.benchmark_plan",
        "write_benchmark_job_plan_shell",
    ),
    "write_engine_probe_targets_json": (
        "document_kv_cache.benchmark_plan",
        "write_engine_probe_targets_json",
    ),
    "DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE": (
        "document_kv_cache.databricks_job",
        "DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE",
    ),
    "DEFAULT_AWS_G5_NODE_TYPE": ("document_kv_cache.databricks_job", "DEFAULT_AWS_G5_NODE_TYPE"),
    "DEFAULT_DATABRICKS_SPARK_VERSION": (
        "document_kv_cache.databricks_job",
        "DEFAULT_DATABRICKS_SPARK_VERSION",
    ),
    "DEFAULT_DATABRICKS_RUN_NAME": ("document_kv_cache.databricks_job", "DEFAULT_DATABRICKS_RUN_NAME"),
    "DEFAULT_DATABRICKS_TASK_KEY": ("document_kv_cache.databricks_job", "DEFAULT_DATABRICKS_TASK_KEY"),
    "DEFAULT_DATABRICKS_PURPOSE": ("document_kv_cache.databricks_job", "DEFAULT_DATABRICKS_PURPOSE"),
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
    "RESERVED_SINGLE_NODE_GPU_TAG_KEYS": (
        "document_kv_cache.databricks_job",
        "RESERVED_SINGLE_NODE_GPU_TAG_KEYS",
    ),
    "DatabricksSingleNodeG5ClusterConfig": (
        "document_kv_cache.databricks_job",
        "DatabricksSingleNodeG5ClusterConfig",
    ),
    "DatabricksSingleNodeGPUClusterConfig": (
        "document_kv_cache.databricks_job",
        "DatabricksSingleNodeGPUClusterConfig",
    ),
    "DatabricksBenchmarkJobConfig": ("document_kv_cache.databricks_job", "DatabricksBenchmarkJobConfig"),
    "validate_aws_g5_node_type": ("document_kv_cache.databricks_job", "validate_aws_g5_node_type"),
    "validate_aws_single_node_gpu_type": (
        "document_kv_cache.databricks_job",
        "validate_aws_single_node_gpu_type",
    ),
    "build_single_node_g5_cluster": ("document_kv_cache.databricks_job", "build_single_node_g5_cluster"),
    "build_single_node_gpu_cluster": ("document_kv_cache.databricks_job", "build_single_node_gpu_cluster"),
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
    "DEFAULT_DATABRICKS_CONFIG_FILE": ("document_kv_cache.databricks_runs", "DEFAULT_DATABRICKS_CONFIG_FILE"),
    "DEFAULT_DATABRICKS_TOKEN_ENV": ("document_kv_cache.databricks_runs", "DEFAULT_DATABRICKS_TOKEN_ENV"),
    "DEFAULT_DATABRICKS_TIMEOUT_SECONDS": (
        "document_kv_cache.databricks_runs",
        "DEFAULT_DATABRICKS_TIMEOUT_SECONDS",
    ),
    "DATABRICKS_PROFILE_AUTH_MODES": (
        "document_kv_cache.databricks_runs",
        "DATABRICKS_PROFILE_AUTH_MODES",
    ),
    "DATABRICKS_DBFS_PUT_MAX_CONTENT_BYTES": (
        "document_kv_cache.databricks_runs",
        "DATABRICKS_DBFS_PUT_MAX_CONTENT_BYTES",
    ),
    "DATABRICKS_AUTH_CHECK_RECORD_TYPE": (
        "document_kv_cache.databricks_runs",
        "DATABRICKS_AUTH_CHECK_RECORD_TYPE",
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
    "databricks_workspace_config_from_profile": (
        "document_kv_cache.databricks_runs",
        "databricks_workspace_config_from_profile",
    ),
    "databricks_workspace_config_from_sdk_profile": (
        "document_kv_cache.databricks_runs",
        "databricks_workspace_config_from_sdk_profile",
    ),
    "check_databricks_auth": ("document_kv_cache.databricks_runs", "check_databricks_auth"),
    "submit_databricks_run": ("document_kv_cache.databricks_runs", "submit_databricks_run"),
    "get_databricks_run": ("document_kv_cache.databricks_runs", "get_databricks_run"),
    "put_databricks_dbfs_file": ("document_kv_cache.databricks_runs", "put_databricks_dbfs_file"),
    "plan_databricks_stage_and_submit": (
        "document_kv_cache.databricks_runs",
        "plan_databricks_stage_and_submit",
    ),
    "stage_and_submit_databricks_run": (
        "document_kv_cache.databricks_runs",
        "stage_and_submit_databricks_run",
    ),
    "summarize_databricks_run": ("document_kv_cache.databricks_runs", "summarize_databricks_run"),
    "summarize_databricks_run_submit_payload": (
        "document_kv_cache.databricks_runs",
        "summarize_databricks_run_submit_payload",
    ),
    "databricks_run_status_record": (
        "document_kv_cache.databricks_runs",
        "databricks_run_status_record",
    ),
    "databricks_run_status_sidecar_issues": (
        "document_kv_cache.databricks_runs",
        "databricks_run_status_sidecar_issues",
    ),
    "validate_databricks_run_status_sidecar": (
        "document_kv_cache.databricks_runs",
        "validate_databricks_run_status_sidecar",
    ),
    "write_databricks_run_response_json": (
        "document_kv_cache.databricks_runs",
        "write_databricks_run_response_json",
    ),
    "read_databricks_run_submit_payload": (
        "document_kv_cache.databricks_runs",
        "read_databricks_run_submit_payload",
    ),
    "EngineAdapterRequest": ("document_kv_cache.engine_adapters", "EngineAdapterRequest"),
    "EngineAdapterSpec": ("document_kv_cache.engine_adapters", "EngineAdapterSpec"),
    "ENGINE_KV_CONNECTOR_ACTIONS_RECORD_TYPE": (
        "document_kv_cache.engine_adapters",
        "ENGINE_KV_CONNECTOR_ACTIONS_RECORD_TYPE",
    ),
    "ENGINE_KV_CONNECTOR_ACTIONS_SCHEMA_VERSION": (
        "document_kv_cache.engine_adapters",
        "ENGINE_KV_CONNECTOR_ACTIONS_SCHEMA_VERSION",
    ),
    "ENGINE_KV_CONNECTOR_PROBE_RECORD_TYPE": (
        "document_kv_cache.engine_adapters",
        "ENGINE_KV_CONNECTOR_PROBE_RECORD_TYPE",
    ),
    "ENGINE_KV_CONNECTOR_PROBE_SCHEMA_VERSION": (
        "document_kv_cache.engine_adapters",
        "ENGINE_KV_CONNECTOR_PROBE_SCHEMA_VERSION",
    ),
    "EngineKVBlockManagerProbe": (
        "document_kv_cache.engine_adapters",
        "EngineKVBlockManagerProbe",
    ),
    "EngineKVBindAction": ("document_kv_cache.engine_adapters", "EngineKVBindAction"),
    "EngineKVConnectorActions": (
        "document_kv_cache.engine_adapters",
        "EngineKVConnectorActions",
    ),
    "EngineKVConnectorProbeResult": (
        "document_kv_cache.engine_adapters",
        "EngineKVConnectorProbeResult",
    ),
    "EngineKVInjectionPlan": ("document_kv_cache.engine_adapters", "EngineKVInjectionPlan"),
    "EngineKVReleaseAction": ("document_kv_cache.engine_adapters", "EngineKVReleaseAction"),
    "EngineKVReservationAction": (
        "document_kv_cache.engine_adapters",
        "EngineKVReservationAction",
    ),
    "EngineKVSegmentCopyAction": (
        "document_kv_cache.engine_adapters",
        "EngineKVSegmentCopyAction",
    ),
    "EngineKVSegmentBinding": (
        "document_kv_cache.engine_adapters",
        "EngineKVSegmentBinding",
    ),
    "PayloadMode": ("document_kv_cache.engine_adapters", "PayloadMode"),
    "ServingBackend": ("document_kv_cache.engine_adapters", "ServingBackend"),
    "build_engine_adapter_request": (
        "document_kv_cache.engine_adapters",
        "build_engine_adapter_request",
    ),
    "build_engine_kv_connector_actions": (
        "document_kv_cache.engine_adapters",
        "build_engine_kv_connector_actions",
    ),
    "build_engine_kv_injection_plan": (
        "document_kv_cache.engine_adapters",
        "build_engine_kv_injection_plan",
    ),
    "engine_kv_connector_actions_from_record": (
        "document_kv_cache.engine_adapters",
        "engine_kv_connector_actions_from_record",
    ),
    "engine_kv_connector_actions_to_record": (
        "document_kv_cache.engine_adapters",
        "engine_kv_connector_actions_to_record",
    ),
    "engine_kv_connector_probe_result_to_record": (
        "document_kv_cache.engine_adapters",
        "engine_kv_connector_probe_result_to_record",
    ),
    "engine_adapter_request_to_record": (
        "document_kv_cache.engine_adapters",
        "engine_adapter_request_to_record",
    ),
    "payload_mode_for": ("document_kv_cache.engine_adapters", "payload_mode_for"),
    "probe_engine_kv_connector_actions": (
        "document_kv_cache.engine_adapters",
        "probe_engine_kv_connector_actions",
    ),
    "read_engine_adapter_request_json": (
        "document_kv_cache.engine_adapters",
        "read_engine_adapter_request_json",
    ),
    "sglang_adapter_spec": ("document_kv_cache.engine_adapters", "sglang_adapter_spec"),
    "split_engine_adapter_payload": (
        "document_kv_cache.engine_adapters",
        "split_engine_adapter_payload",
    ),
    "validate_engine_adapter_request_record": (
        "document_kv_cache.engine_adapters",
        "validate_engine_adapter_request_record",
    ),
    "validate_engine_kv_connector_actions_record": (
        "document_kv_cache.engine_adapters",
        "validate_engine_kv_connector_actions_record",
    ),
    "validate_engine_kv_connector_probe_record": (
        "document_kv_cache.engine_adapters",
        "validate_engine_kv_connector_probe_record",
    ),
    "validate_engine_kv_connector_actions": (
        "document_kv_cache.engine_adapters",
        "validate_engine_kv_connector_actions",
    ),
    "view_engine_adapter_payload": (
        "document_kv_cache.engine_adapters",
        "view_engine_adapter_payload",
    ),
    "vllm_adapter_spec": ("document_kv_cache.engine_adapters", "vllm_adapter_spec"),
    "write_engine_adapter_request_json": (
        "document_kv_cache.engine_adapters",
        "write_engine_adapter_request_json",
    ),
    "DEFAULT_ENGINE_LAUNCH_CONFIG_KV_INJECTION_METHOD": (
        "document_kv_cache.engine_launch_config",
        "DEFAULT_ENGINE_LAUNCH_CONFIG_KV_INJECTION_METHOD",
    ),
    "DEFAULT_ENGINE_LAUNCH_CONFIG_SCHEMA_VERSION": (
        "document_kv_cache.engine_launch_config",
        "DEFAULT_ENGINE_LAUNCH_CONFIG_SCHEMA_VERSION",
    ),
    "DEFAULT_SGLANG_DOCUMENT_KV_MODULE_PATH": (
        "document_kv_cache.engine_launch_config",
        "DEFAULT_SGLANG_DOCUMENT_KV_MODULE_PATH",
    ),
    "DEFAULT_SGLANG_DOCUMENT_KV_PROVIDER_FACTORY": (
        "document_kv_cache.engine_launch_config",
        "DEFAULT_SGLANG_DOCUMENT_KV_PROVIDER_FACTORY",
    ),
    "DEFAULT_SGLANG_ENGINE_LAUNCH_CONFIG_RECORD_TYPE": (
        "document_kv_cache.engine_launch_config",
        "DEFAULT_SGLANG_ENGINE_LAUNCH_CONFIG_RECORD_TYPE",
    ),
    "DEFAULT_VLLM_DOCUMENT_KV_MODULE_PATH": (
        "document_kv_cache.engine_launch_config",
        "DEFAULT_VLLM_DOCUMENT_KV_MODULE_PATH",
    ),
    "DEFAULT_VLLM_ENGINE_LAUNCH_CONFIG_RECORD_TYPE": (
        "document_kv_cache.engine_launch_config",
        "DEFAULT_VLLM_ENGINE_LAUNCH_CONFIG_RECORD_TYPE",
    ),
    "ENGINE_LAUNCH_CONFIG_EVIDENCE_RECORD_TYPE": (
        "document_kv_cache.engine_launch_config",
        "ENGINE_LAUNCH_CONFIG_EVIDENCE_RECORD_TYPE",
    ),
    "ENGINE_LAUNCH_CONFIG_EVIDENCE_SCHEMA_VERSION": (
        "document_kv_cache.engine_launch_config",
        "ENGINE_LAUNCH_CONFIG_EVIDENCE_SCHEMA_VERSION",
    ),
    "REQUIRED_ENGINE_LAUNCH_CONFIG_BACKENDS": (
        "document_kv_cache.engine_launch_config",
        "REQUIRED_ENGINE_LAUNCH_CONFIG_BACKENDS",
    ),
    "EngineLaunchConfigEvidence": (
        "document_kv_cache.engine_launch_config",
        "EngineLaunchConfigEvidence",
    ),
    "build_sglang_launch_config": (
        "document_kv_cache.engine_launch_config",
        "build_sglang_launch_config",
    ),
    "build_vllm_launch_config": (
        "document_kv_cache.engine_launch_config",
        "build_vllm_launch_config",
    ),
    "engine_launch_config_evidence_to_record": (
        "document_kv_cache.engine_launch_config",
        "engine_launch_config_evidence_to_record",
    ),
    "engine_launch_config_record_issues": (
        "document_kv_cache.engine_launch_config",
        "engine_launch_config_record_issues",
    ),
    "evaluate_engine_launch_config_evidence": (
        "document_kv_cache.engine_launch_config",
        "evaluate_engine_launch_config_evidence",
    ),
    "read_engine_launch_config_json": (
        "document_kv_cache.engine_launch_config",
        "read_engine_launch_config_json",
    ),
    "validate_engine_launch_config_record": (
        "document_kv_cache.engine_launch_config",
        "validate_engine_launch_config_record",
    ),
    "write_engine_launch_config_json": (
        "document_kv_cache.engine_launch_config",
        "write_engine_launch_config_json",
    ),
    "write_engine_launch_config_evidence_json": (
        "document_kv_cache.engine_launch_config",
        "write_engine_launch_config_evidence_json",
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
    "write_engine_adapter_handoff_bundle": (
        "document_kv_cache.engine_probe",
        "write_engine_adapter_handoff_bundle",
    ),
    "write_engine_adapter_payload": ("document_kv_cache.engine_probe", "write_engine_adapter_payload"),
    "write_engine_kv_connector_actions_record_json": (
        "document_kv_cache.engine_probe",
        "write_engine_kv_connector_actions_record_json",
    ),
    "write_engine_kv_connector_probe_result_json": (
        "document_kv_cache.engine_probe",
        "write_engine_kv_connector_probe_result_json",
    ),
    "load_engine_kv_probe_factory": ("document_kv_cache.engine_probe", "load_engine_kv_probe_factory"),
    "AdmissionQueue": ("document_kv_cache.admission", "AdmissionQueue"),
    "PreparedRequest": ("document_kv_cache.admission", "PreparedRequest"),
    "DTYPE_BYTE_WIDTHS": ("document_kv_cache.engine_protocol", "DTYPE_BYTE_WIDTHS"),
    "AttentionMechanism": ("document_kv_cache.engine_protocol", "AttentionMechanism"),
    "KVStorageLayout": ("document_kv_cache.engine_protocol", "KVStorageLayout"),
    "dtype_byte_width": ("document_kv_cache.engine_protocol", "dtype_byte_width"),
    "kv_storage_layout_from_value": (
        "document_kv_cache.engine_protocol",
        "kv_storage_layout_from_value",
    ),
    "KVLayout": ("document_kv_cache.engine_protocol", "KVLayout"),
    "KVSegment": ("document_kv_cache.engine_protocol", "KVSegment"),
    "KVCacheHandle": ("document_kv_cache.engine_protocol", "KVCacheHandle"),
    "MODEL_PROFILE_RECORD_TYPE": ("document_kv_cache.model_profiles", "MODEL_PROFILE_RECORD_TYPE"),
    "KVModelProfile": ("document_kv_cache.model_profiles", "KVModelProfile"),
    "ModelProfileDefinition": ("document_kv_cache.model_profiles", "ModelProfileDefinition"),
    "ModelProfileRegistry": ("document_kv_cache.model_profiles", "ModelProfileRegistry"),
    "QWEN3_4B_BASE_HF_MODEL_ID": ("document_kv_cache.model_profiles", "QWEN3_4B_BASE_HF_MODEL_ID"),
    "QWEN3_4B_INSTRUCT_HF_MODEL_ID": ("document_kv_cache.model_profiles", "QWEN3_4B_INSTRUCT_HF_MODEL_ID"),
    "QWEN3_4B_INSTRUCT_PROFILE": ("document_kv_cache.model_profiles", "QWEN3_4B_INSTRUCT_PROFILE"),
    "builtin_model_profiles": ("document_kv_cache.model_profiles", "builtin_model_profiles"),
    "default_model_profile_registry": (
        "document_kv_cache.model_profiles",
        "default_model_profile_registry",
    ),
    "get_model_profile": ("document_kv_cache.model_profiles", "get_model_profile"),
    "layout_for_model": ("document_kv_cache.model_profiles", "layout_for_model"),
    "model_profile_definition_from_record": (
        "document_kv_cache.model_profiles",
        "model_profile_definition_from_record",
    ),
    "model_profile_definition_to_record": (
        "document_kv_cache.model_profiles",
        "model_profile_definition_to_record",
    ),
    "read_model_profile_definition_json": (
        "document_kv_cache.model_profiles",
        "read_model_profile_definition_json",
    ),
    "write_model_profile_definition_json": (
        "document_kv_cache.model_profiles",
        "write_model_profile_definition_json",
    ),
    "FASTAPI_CONSTRAINT": ("document_kv_cache.serving_env", "FASTAPI_CONSTRAINT"),
    "HUGGINGFACE_HUB_CONSTRAINT": (
        "document_kv_cache.serving_env",
        "HUGGINGFACE_HUB_CONSTRAINT",
    ),
    "NUMPY_CONSTRAINT": ("document_kv_cache.serving_env", "NUMPY_CONSTRAINT"),
    "PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT": (
        "document_kv_cache.serving_env",
        "PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT",
    ),
    "SERVING_ENVIRONMENT_PROFILES_RECORD_TYPE": (
        "document_kv_cache.serving_env",
        "SERVING_ENVIRONMENT_PROFILES_RECORD_TYPE",
    ),
    "SGLANG_DEPENDENCY_CONSTRAINTS": (
        "document_kv_cache.serving_env",
        "SGLANG_DEPENDENCY_CONSTRAINTS",
    ),
    "SGLANG_SERVING_ENVIRONMENT_PROFILE": (
        "document_kv_cache.serving_env",
        "SGLANG_SERVING_ENVIRONMENT_PROFILE",
    ),
    "SGLANG_VERSION": ("document_kv_cache.serving_env", "SGLANG_VERSION"),
    "ServingEnvironmentProfile": ("document_kv_cache.serving_env", "ServingEnvironmentProfile"),
    "TOKENIZERS_CONSTRAINT": ("document_kv_cache.serving_env", "TOKENIZERS_CONSTRAINT"),
    "TRANSFORMERS_CONSTRAINT": (
        "document_kv_cache.serving_env",
        "TRANSFORMERS_CONSTRAINT",
    ),
    "VLLM_DEPENDENCY_CONSTRAINTS": (
        "document_kv_cache.serving_env",
        "VLLM_DEPENDENCY_CONSTRAINTS",
    ),
    "VLLM_SERVING_ENVIRONMENT_PROFILE": (
        "document_kv_cache.serving_env",
        "VLLM_SERVING_ENVIRONMENT_PROFILE",
    ),
    "VLLM_VERSION": ("document_kv_cache.serving_env", "VLLM_VERSION"),
    "serving_environment_profile": (
        "document_kv_cache.serving_env",
        "serving_environment_profile",
    ),
    "serving_environment_profile_to_record": (
        "document_kv_cache.serving_env",
        "serving_environment_profile_to_record",
    ),
    "serving_environment_profiles": (
        "document_kv_cache.serving_env",
        "serving_environment_profiles",
    ),
    "serving_environment_profiles_to_record": (
        "document_kv_cache.serving_env",
        "serving_environment_profiles_to_record",
    ),
    "NativeProbeFactoryInspection": (
        "document_kv_cache.native_probe_factories",
        "NativeProbeFactoryInspection",
    ),
    "NativeProbeFactoryUnavailable": (
        "document_kv_cache.native_probe_factories",
        "NativeProbeFactoryUnavailable",
    ),
    "NATIVE_PROBE_ADAPTER_CONTRACT": (
        "document_kv_cache.native_probe_factories",
        "NATIVE_PROBE_ADAPTER_CONTRACT",
    ),
    "NATIVE_PROBE_FACTORIES_RECORD_TYPE": (
        "document_kv_cache.native_probe_factories",
        "NATIVE_PROBE_FACTORIES_RECORD_TYPE",
    ),
    "SGLANG_NATIVE_PROBE_FACTORY": (
        "document_kv_cache.native_probe_factories",
        "SGLANG_NATIVE_PROBE_FACTORY",
    ),
    "SGLANG_NATIVE_PROBE_DELEGATE_ENV": (
        "document_kv_cache.native_probe_factories",
        "SGLANG_NATIVE_PROBE_DELEGATE_ENV",
    ),
    "NATIVE_PROBE_DELEGATE_CONTRACT_ATTR": (
        "document_kv_cache.native_probe_factories",
        "NATIVE_PROBE_DELEGATE_CONTRACT_ATTR",
    ),
    "NATIVE_PROBE_DELEGATE_CONTRACT_MODULE_ATTR": (
        "document_kv_cache.native_probe_factories",
        "NATIVE_PROBE_DELEGATE_CONTRACT_MODULE_ATTR",
    ),
    "NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_ATTR": (
        "document_kv_cache.native_probe_factories",
        "NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_ATTR",
    ),
    "NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_MODULE_ATTR": (
        "document_kv_cache.native_probe_factories",
        "NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_MODULE_ATTR",
    ),
    "VLLM_NATIVE_PROBE_FACTORY": (
        "document_kv_cache.native_probe_factories",
        "VLLM_NATIVE_PROBE_FACTORY",
    ),
    "VLLM_NATIVE_PROBE_DELEGATE_ENV": (
        "document_kv_cache.native_probe_factories",
        "VLLM_NATIVE_PROBE_DELEGATE_ENV",
    ),
    "builtin_native_probe_factories_to_record": (
        "document_kv_cache.native_probe_factories",
        "builtin_native_probe_factories_to_record",
    ),
    "builtin_native_probe_factory_path": (
        "document_kv_cache.native_probe_factories",
        "builtin_native_probe_factory_path",
    ),
    "inspect_builtin_native_probe_factories": (
        "document_kv_cache.native_probe_factories",
        "inspect_builtin_native_probe_factories",
    ),
    "inspect_builtin_native_probe_factory": (
        "document_kv_cache.native_probe_factories",
        "inspect_builtin_native_probe_factory",
    ),
    "native_probe_adapter_contract_to_record": (
        "document_kv_cache.native_probe_factories",
        "native_probe_adapter_contract_to_record",
    ),
    "native_probe_runtime_contract_to_record": (
        "document_kv_cache.native_probe_factories",
        "native_probe_runtime_contract_to_record",
    ),
    "native_probe_factory_inspection_to_record": (
        "document_kv_cache.native_probe_factories",
        "native_probe_factory_inspection_to_record",
    ),
    "native_probe_factories_record_issues": (
        "document_kv_cache.native_probe_factories",
        "native_probe_factories_record_issues",
    ),
    "sglang_native_probe_factory": (
        "document_kv_cache.native_probe_factories",
        "sglang_native_probe_factory",
    ),
    "validate_native_probe_factories_record": (
        "document_kv_cache.native_probe_factories",
        "validate_native_probe_factories_record",
    ),
    "vllm_native_probe_factory": (
        "document_kv_cache.native_probe_factories",
        "vllm_native_probe_factory",
    ),
    "write_builtin_native_probe_factories_record_json": (
        "document_kv_cache.native_probe_factories",
        "write_builtin_native_probe_factories_record_json",
    ),
    "EngineReadyRequest": ("document_kv_cache.engine", "EngineReadyRequest"),
    "ServingEngineConnector": ("document_kv_cache.engine", "ServingEngineConnector"),
    "build_engine_ready_request": ("document_kv_cache.engine", "build_engine_ready_request"),
    "build_handle_from_materialized": ("document_kv_cache.engine", "build_handle_from_materialized"),
    "CacheTier": ("document_kv_cache.cache", "CacheTier"),
    "ChunkCacheResult": ("document_kv_cache.cache", "ChunkCacheResult"),
    "ChunkCacheStats": ("document_kv_cache.cache", "ChunkCacheStats"),
    "ByteLRU": ("document_kv_cache.cache", "ByteLRU"),
    "ChunkCache": ("document_kv_cache.cache", "ChunkCache"),
    "DocumentChunkType": ("document_kv_cache.models", "DocumentChunkType"),
    "DocumentChunkRole": ("document_kv_cache.models", "DocumentChunkRole"),
    "CacheGenerationMethod": ("document_kv_cache.models", "CacheGenerationMethod"),
    "DocumentChunkMap": ("document_kv_cache.models", "DocumentChunkMap"),
    "FrozenDocumentChunkMap": ("document_kv_cache.models", "FrozenDocumentChunkMap"),
    "DEFAULT_STATIC_CHUNK_ID": ("document_kv_cache.models", "DEFAULT_STATIC_CHUNK_ID"),
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
    "DocumentKVService": ("document_kv_cache.service", "DocumentKVService"),
    "RangeReader": ("document_kv_cache.storage", "RangeReader"),
    "RangeBatchReader": ("document_kv_cache.storage", "RangeBatchReader"),
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
    "SourceChunk": ("document_kv_cache.workflow", "SourceChunk"),
    "SourceDocument": ("document_kv_cache.workflow", "SourceDocument"),
    "CacheBuildConfig": ("document_kv_cache.workflow", "CacheBuildConfig"),
    "CacheAdapterArtifact": ("document_kv_cache.workflow", "CacheAdapterArtifact"),
    "TrainingArtifacts": ("document_kv_cache.workflow", "TrainingArtifacts"),
    "TrainingAdapter": ("document_kv_cache.workflow", "TrainingAdapter"),
    "KVChunkGenerator": ("document_kv_cache.workflow", "KVChunkGenerator"),
    "CacheGenerationResult": ("document_kv_cache.workflow", "CacheGenerationResult"),
    "DocumentKVWorkflow": ("document_kv_cache.workflow", "DocumentKVWorkflow"),
}

try:
    _legacy_package = import_module(_LEGACY_PACKAGE)
except ModuleNotFoundError as exc:
    if exc.name != _LEGACY_PACKAGE:
        raise
    _legacy_package = None

__all__ = (
    [
        name
        for name in getattr(_legacy_package, "__all__", ())
        if name not in _LEGACY_ROOT_EXPORTS
    ]
    if _legacy_package is not None
    else []
)
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
    if _legacy_package is not None:
        return getattr(_legacy_package, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__, *_PUBLIC_SUBMODULES})
