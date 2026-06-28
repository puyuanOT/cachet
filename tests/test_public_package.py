import ast
import importlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import cachet
import document_kv_cache


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DOCUMENT_ROOT_EXPORTS = (
    'AdmissionQueue',
    'PreparedRequest',
    'BenchmarkCommand',
    'BenchmarkDatasetPath',
    'BenchmarkJobPlan',
    'BenchmarkPlanConfig',
    'ENGINE_PROBE_TARGETS_RECORD_TYPE',
    'ENGINE_PROBE_TARGETS_SCHEMA_VERSION',
    'EngineProbePlanConfig',
    'PLAN_VERSION',
    'ReleaseBundlePlanConfig',
    'ReleaseEvidencePlanConfig',
    'StorageBenchmarkPlanConfig',
    'benchmark_job_plan_to_record',
    'build_v1_benchmark_plan',
    'engine_probe_targets_to_record',
    'write_benchmark_job_plan_json',
    'write_benchmark_job_plan_shell',
    'write_engine_probe_targets_json',
    'BenchmarkCommandResult',
    'benchmark_command_results_to_record',
    'execute_benchmark_job_plan',
    'execute_benchmark_job_plan_json',
    'write_benchmark_command_results_json',
    'BENCHMARK_RUN_RECORD_TYPE',
    'BenchmarkEngine',
    'BenchmarkEngineRequest',
    'BenchmarkGeneration',
    'BenchmarkRunResult',
    'OpenAICompatibleBenchmarkConfig',
    'benchmark_run_result_to_record',
    'default_benchmark_arms',
    'load_benchmark_jsonl',
    'load_v1_jsonl_suite',
    'run_benchmark_suite',
    'run_openai_compatible_v1_benchmark',
    'write_benchmark_run_result_json',
    'BASELINE_PREFILL_ARM',
    'BenchmarkArm',
    'BenchmarkComparison',
    'BenchmarkDatasetSpec',
    'BenchmarkExample',
    'BenchmarkPromptParts',
    'BenchmarkReportRow',
    'BenchmarkSuite',
    'CACHE_REUSE_ARM',
    'DEFAULT_HARDWARE_TARGET',
    'DEFAULT_V1_MODEL_ID',
    'InferenceMeasurement',
    'LatencySummary',
    'SUPPORTED_V1_DATASETS',
    'SUPPORTED_V1_HARDWARE_TARGETS',
    'V1BenchmarkEvidence',
    'answer_found',
    'baseline_prefill_arm',
    'build_cache_prefix_text',
    'build_cache_suffix_text',
    'build_prefill_prompt',
    'build_prompt_parts',
    'compare_to_baseline',
    'dataset_spec',
    'document_kv_cache_arm',
    'evaluate_v1_benchmark_evidence',
    'exact_match',
    'format_document_context',
    'normalize_answer',
    'summarize_measurements',
    'validate_v1_hardware_target',
    'v1_dataset_specs',
    'validate_v1_dataset',
    'ByteLRU',
    'CacheTier',
    'ChunkCache',
    'ChunkCacheResult',
    'ChunkCacheStats',
    'DEFAULT_DATABRICKS_ENGINE_PROBE_BACKEND_CONFIG_KEY',
    'DEFAULT_DATABRICKS_ENGINE_PROBE_PURPOSE',
    'DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME',
    'DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY',
    'DatabricksEngineProbeJobConfig',
    'DatabricksEngineProbeMatrixJobConfig',
    'DatabricksEngineProbeTargetConfig',
    'DatabricksEngineProbeTargetsFile',
    'build_databricks_engine_probe_matrix_run_submit_payload',
    'build_databricks_engine_probe_run_submit_payload',
    'read_databricks_engine_probe_targets_file_json',
    'read_databricks_engine_probe_targets_json',
    'run_engine_probe_task',
    'write_databricks_engine_probe_matrix_run_submit_json',
    'write_databricks_engine_probe_run_submit_json',
    'write_databricks_engine_probe_runner_script',
    'DEDICATED_DATABRICKS_DATA_SECURITY_MODE',
    'DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE',
    'DEFAULT_AWS_G5_NODE_TYPE',
    'DEFAULT_DATABRICKS_DATA_SECURITY_MODE',
    'DEFAULT_DATABRICKS_PURPOSE',
    'DEFAULT_DATABRICKS_RUN_NAME',
    'DEFAULT_DATABRICKS_SPARK_VERSION',
    'DEFAULT_DATABRICKS_TASK_KEY',
    'DatabricksSingleNodeGPUClusterConfig',
    'DatabricksBenchmarkJobConfig',
    'DatabricksSingleNodeG5ClusterConfig',
    'RESERVED_SINGLE_NODE_GPU_TAG_KEYS',
    'SINGLE_USER_DATABRICKS_DATA_SECURITY_MODES',
    'build_databricks_run_submit_payload',
    'build_single_node_gpu_cluster',
    'build_single_node_g5_cluster',
    'validate_aws_single_node_gpu_type',
    'validate_aws_g5_node_type',
    'write_databricks_run_submit_json',
    'write_databricks_runner_script',
    'DATABRICKS_AUTH_CHECK_RECORD_TYPE',
    'DATABRICKS_DBFS_PUT_MAX_CONTENT_BYTES',
    'DATABRICKS_PROFILE_AUTH_MODES',
    'DATABRICKS_RUN_STATUS_RECORD_TYPE',
    'DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE',
    'DEFAULT_DATABRICKS_CONFIG_FILE',
    'DEFAULT_DATABRICKS_HOST_ENV',
    'DEFAULT_DATABRICKS_TIMEOUT_SECONDS',
    'DEFAULT_DATABRICKS_TOKEN_ENV',
    'DatabricksWorkspaceConfig',
    'databricks_run_status_record',
    'databricks_run_status_sidecar_issues',
    'check_databricks_auth',
    'databricks_workspace_config_from_env',
    'databricks_workspace_config_from_profile',
    'databricks_workspace_config_from_sdk_profile',
    'get_databricks_run',
    'plan_databricks_stage_and_submit',
    'put_databricks_dbfs_file',
    'read_databricks_run_submit_payload',
    'stage_and_submit_databricks_run',
    'submit_databricks_run',
    'summarize_databricks_run',
    'summarize_databricks_run_submit_payload',
    'validate_databricks_run_status_sidecar',
    'write_databricks_run_response_json',
    'DEFAULT_DATABRICKS_STORAGE_BENCHMARK_PURPOSE',
    'DEFAULT_DATABRICKS_STORAGE_BENCHMARK_RUN_NAME',
    'DEFAULT_DATABRICKS_STORAGE_BENCHMARK_TASK_KEY',
    'DatabricksStorageBenchmarkJobConfig',
    'build_databricks_storage_benchmark_run_submit_payload',
    'write_databricks_storage_benchmark_run_submit_json',
    'write_databricks_storage_benchmark_runner_script',
    'DEFAULT_DATABRICKS_SGLANG_SMOKE_PURPOSE',
    'DEFAULT_DATABRICKS_SGLANG_SMOKE_RUN_NAME',
    'DEFAULT_DATABRICKS_SGLANG_SMOKE_TASK_KEY',
    'DatabricksSGLangSmokeJobConfig',
    'build_databricks_sglang_smoke_run_submit_payload',
    'write_databricks_sglang_smoke_run_submit_json',
    'write_databricks_sglang_smoke_runner_script',
    'DEFAULT_DATABRICKS_VLLM_SMOKE_PURPOSE',
    'DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME',
    'DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY',
    'DatabricksVLLMSmokeJobConfig',
    'build_databricks_vllm_smoke_run_submit_payload',
    'write_databricks_vllm_smoke_run_submit_json',
    'write_databricks_vllm_smoke_runner_script',
    'DEPENDENCY_FRESHNESS_RECORD_TYPE',
    'DependencyFreshnessEvidence',
    'DirectDependencyPin',
    'RuntimeDependencyPin',
    'TransitiveDependencyDrift',
    'dependency_freshness_record_issues',
    'dependency_freshness_to_record',
    'evaluate_dependency_freshness',
    'pyproject_direct_dependency_pins',
    'serving_profile_runtime_pins',
    'write_dependency_freshness_json',
    'DEFAULT_NIAH_QUERY',
    'build_niah_record',
    'convert_v1_jsonl',
    'normalize_v1_record',
    'write_v1_jsonl',
    'EngineReadyRequest',
    'ServingEngineConnector',
    'build_engine_ready_request',
    'build_handle_from_materialized',
    'ENGINE_KV_CONNECTOR_ACTIONS_RECORD_TYPE',
    'ENGINE_KV_CONNECTOR_ACTIONS_SCHEMA_VERSION',
    'ENGINE_KV_CONNECTOR_PROBE_RECORD_TYPE',
    'ENGINE_KV_CONNECTOR_PROBE_SCHEMA_VERSION',
    'EngineAdapterRequest',
    'EngineAdapterSpec',
    'EngineKVBindAction',
    'EngineKVBlockManagerProbe',
    'EngineKVConnectorActions',
    'EngineKVConnectorProbeResult',
    'EngineKVInjectionPlan',
    'EngineKVReleaseAction',
    'EngineKVReservationAction',
    'EngineKVSegmentBinding',
    'EngineKVSegmentCopyAction',
    'PayloadMode',
    'ServingBackend',
    'build_engine_adapter_request',
    'build_engine_kv_connector_actions',
    'build_engine_kv_injection_plan',
    'engine_adapter_request_to_record',
    'engine_kv_connector_actions_from_record',
    'engine_kv_connector_actions_to_record',
    'engine_kv_connector_probe_result_to_record',
    'payload_mode_for',
    'probe_engine_kv_connector_actions',
    'read_engine_adapter_request_json',
    'sglang_adapter_spec',
    'split_engine_adapter_payload',
    'validate_engine_adapter_request_record',
    'validate_engine_kv_connector_actions',
    'validate_engine_kv_connector_actions_record',
    'validate_engine_kv_connector_probe_record',
    'view_engine_adapter_payload',
    'vllm_adapter_spec',
    'write_engine_adapter_request_json',
    'DEFAULT_ENGINE_LAUNCH_CONFIG_KV_INJECTION_METHOD',
    'DEFAULT_ENGINE_LAUNCH_CONFIG_SCHEMA_VERSION',
    'DEFAULT_SGLANG_DOCUMENT_KV_MODULE_PATH',
    'DEFAULT_SGLANG_DOCUMENT_KV_PROVIDER_FACTORY',
    'DEFAULT_SGLANG_ENGINE_LAUNCH_CONFIG_RECORD_TYPE',
    'DEFAULT_VLLM_DOCUMENT_KV_MODULE_PATH',
    'DEFAULT_VLLM_ENGINE_LAUNCH_CONFIG_RECORD_TYPE',
    'ENGINE_LAUNCH_CONFIG_EVIDENCE_RECORD_TYPE',
    'ENGINE_LAUNCH_CONFIG_EVIDENCE_SCHEMA_VERSION',
    'REQUIRED_ENGINE_LAUNCH_CONFIG_BACKENDS',
    'EngineLaunchConfigEvidence',
    'build_sglang_launch_config',
    'build_vllm_launch_config',
    'engine_launch_config_evidence_to_record',
    'engine_launch_config_record_issues',
    'evaluate_engine_launch_config_evidence',
    'read_engine_launch_config_json',
    'validate_engine_launch_config_record',
    'write_engine_launch_config_json',
    'write_engine_launch_config_evidence_json',
    'ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND',
    'ENGINE_KV_PROBE_METADATA_HANDOFF_JSON',
    'ENGINE_KV_PROBE_METADATA_PAYLOAD_URI',
    'ENGINE_KV_PROBE_METADATA_PROBE_FACTORY',
    'ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE',
    'ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION',
    'EngineKVProbeConfig',
    'EngineKVProbeFactory',
    'EngineKVProbeFactoryContext',
    'EngineKVProbeFactoryResult',
    'load_engine_kv_probe_factory',
    'read_engine_adapter_payload',
    'run_engine_kv_connector_probe',
    'write_engine_adapter_handoff_bundle',
    'write_engine_adapter_payload',
    'write_engine_kv_connector_actions_record_json',
    'write_engine_kv_connector_probe_result_json',
    'AttentionMechanism',
    'DTYPE_BYTE_WIDTHS',
    'KVCacheHandle',
    'KVLayout',
    'KVSegment',
    'KVStorageLayout',
    'dtype_byte_width',
    'kv_storage_layout_from_value',
    'LIVE_CHECK_SUITE_ID',
    'DEFAULT_LIVE_CHECK_PROMPT_FORMAT',
    'LiveCheckPromptFormat',
    'LiveServerCheckConfig',
    'LiveServerCheckResult',
    'build_live_server_check_request',
    'live_check_kv_transfer_params',
    'run_openai_compatible_live_check',
    'DEFAULT_SGLANG_LIVE_CHECK_EXTRA_BODY',
    'DEFAULT_SGLANG_LIVE_CHECK_PROMPT_FORMAT',
    'DEFAULT_SGLANG_LIVE_CHECK_REQUEST_MODE',
    'DEFAULT_SGLANG_LIVE_CHECK_TEMPERATURE',
    'DEFAULT_SGLANG_FLUSH_CACHE_BEFORE_CACHE_ARM',
    'DEFAULT_SGLANG_FLUSH_CACHE_BEFORE_CANARY',
    'DEFAULT_SGLANG_FLUSH_CACHE_TIMEOUT_SECONDS',
    'DEFAULT_SGLANG_LIVE_BENCHMARK_REPEATS',
    'SGLANG_LIVE_BENCHMARK_RECORD_TYPE',
    'SGLANG_LIVE_BENCHMARK_SUITE_ID',
    'SGLangSmokeBenchmarkConfig',
    'build_sglang_hicache_provider_probe_record',
    'build_sglang_server_args',
    'flush_sglang_cache',
    'run_sglang_live_benchmark',
    'run_sglang_live_smoke',
    'sglang_hicache_config_for_smoke',
    'sglang_live_kv_transfer_params',
    'InMemoryManifestStore',
    'ManifestStore',
    'KVMaterializer',
    'MaterializedKV',
    'SegmentedMaterializedKV',
    'KVModelProfile',
    'MODEL_PROFILE_RECORD_TYPE',
    'ModelProfileDefinition',
    'ModelProfileRegistry',
    'QWEN3_4B_BASE_HF_MODEL_ID',
    'QWEN3_4B_INSTRUCT_HF_MODEL_ID',
    'QWEN3_4B_INSTRUCT_PROFILE',
    'builtin_model_profiles',
    'default_model_profile_registry',
    'get_model_profile',
    'layout_for_model',
    'model_profile_definition_from_record',
    'model_profile_definition_to_record',
    'read_model_profile_definition_json',
    'write_model_profile_definition_json',
    'CacheChunkType',
    'CacheChunkTypeSet',
    'CacheGenerationMethod',
    'ChunkRef',
    'DEFAULT_STATIC_CHUNK_ID',
    'DOCUMENT_CHUNK_TYPES',
    'DocumentChunkMap',
    'DocumentChunkRole',
    'DocumentChunkType',
    'DocumentKVRequest',
    'FrozenDocumentChunkMap',
    'KVCacheKey',
    'MaterializationPlan',
    'PlanSegment',
    'chunk_type_role',
    'chunk_type_sort_order',
    'chunk_types_for_request',
    'NATIVE_PROBE_ADAPTER_CONTRACT',
    'NATIVE_PROBE_DELEGATE_CONTRACT_ATTR',
    'NATIVE_PROBE_DELEGATE_CONTRACT_MODULE_ATTR',
    'NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_ATTR',
    'NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_MODULE_ATTR',
    'NATIVE_PROBE_FACTORIES_RECORD_TYPE',
    'NativeProbeFactoryInspection',
    'NativeProbeFactoryUnavailable',
    'SGLANG_NATIVE_PROBE_FACTORY',
    'SGLANG_NATIVE_PROBE_DELEGATE_ENV',
    'VLLM_NATIVE_PROBE_FACTORY',
    'VLLM_NATIVE_PROBE_DELEGATE_ENV',
    'builtin_native_probe_factories_to_record',
    'builtin_native_probe_factory_path',
    'inspect_builtin_native_probe_factories',
    'inspect_builtin_native_probe_factory',
    'native_probe_adapter_contract_to_record',
    'native_probe_factories_record_issues',
    'native_probe_factory_inspection_to_record',
    'native_probe_runtime_contract_to_record',
    'sglang_native_probe_factory',
    'validate_native_probe_factories_record',
    'vllm_native_probe_factory',
    'write_builtin_native_probe_factories_record_json',
    'OpenAICompatibleCompletionEngine',
    'OpenAICompatibleEngineConfig',
    'KVTransferParamsTransport',
    'OpenAICompatibleRequestMode',
    'PromptTextMode',
    'PromptTokenAccounting',
    'TokenCounter',
    'WhitespaceTokenCounter',
    'CachePlanner',
    'CacheRequest',
    'GPT55_REVIEW_OUTCOMES',
    'PR_EVIDENCE_RECORD_TYPE',
    'PR_EVIDENCE_VALIDATION_RECORD_TYPE',
    'PullRequestEvidence',
    'evaluate_pr_evidence',
    'evaluate_pr_evidence_directory',
    'evaluate_pr_evidence_file',
    'evaluate_pr_evidence_record',
    'pr_evidence_to_record',
    'pr_evidence_validation_to_record',
    'write_pr_evidence_json',
    'RELEASE_BUNDLE_ARTIFACT_ROLES',
    'RELEASE_BUNDLE_MANIFEST_FILENAME',
    'RELEASE_BUNDLE_RECORD_TYPE',
    'ReleaseBundle',
    'ReleaseBundleArtifact',
    'build_release_bundle',
    'release_bundle_to_record',
    'write_release_bundle_manifest_json',
    'RELEASE_EVIDENCE_ARTIFACT_ROLES',
    'RELEASE_EVIDENCE_INPUT_STATUS_RECORD_TYPE',
    'RELEASE_EVIDENCE_RECORD_TYPE',
    'REQUIRED_ENGINE_PROBE_BACKENDS',
    'ReleaseEvidence',
    'ReleaseEvidenceArtifactSource',
    'ReleaseEvidenceInputFileStatus',
    'ReleaseEvidenceInputStatus',
    'evaluate_release_evidence',
    'evaluate_release_evidence_files',
    'inspect_release_evidence_input_files',
    'release_evidence_input_status_to_record',
    'release_evidence_to_record',
    'write_release_evidence_input_status_json',
    'write_release_evidence_json',
    'DocumentKVService',
    'ACCELERATE_CONSTRAINT',
    'BITSANDBYTES_CONSTRAINT',
    'FASTAPI_CONSTRAINT',
    'HUGGINGFACE_HUB_CONSTRAINT',
    'NUMPY_CONSTRAINT',
    'PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT',
    'SERVING_ENVIRONMENT_PROFILES_RECORD_TYPE',
    'SGLANG_DEPENDENCY_CONSTRAINTS',
    'SGLANG_SERVING_ENVIRONMENT_PROFILE',
    'SGLANG_VERSION',
    'ServingEnvironmentProfile',
    'TOKENIZERS_CONSTRAINT',
    'TRANSFORMERS_CONSTRAINT',
    'VLLM_DEPENDENCY_CONSTRAINTS',
    'VLLM_SERVING_ENVIRONMENT_PROFILE',
    'VLLM_VERSION',
    'serving_environment_profile',
    'serving_environment_profile_to_record',
    'serving_environment_profiles',
    'serving_environment_profiles_to_record',
    'DiskRangeReader',
    'MemoryRangeReader',
    'RangeBatchReader',
    'RangeReader',
    'RoutedRangeReader',
    'UnityCatalogVolumeRangeReader',
    'is_real_uc_volume_root',
    'local_path',
    'unity_catalog_volume_path',
    'RELEASE_STORAGE_BENCHMARK_READERS',
    'STORAGE_BENCHMARK_RECORD_TYPE',
    'SUPPORTED_STORAGE_BENCHMARK_READERS',
    'StorageBenchmarkConfig',
    'StorageBenchmarkEvidence',
    'StorageBenchmarkResult',
    'StorageReaderBenchmarkResult',
    'evaluate_release_storage_benchmark_evidence',
    'evaluate_storage_benchmark_evidence',
    'run_storage_benchmark',
    'storage_benchmark_evidence_to_record',
    'storage_benchmark_result_to_record',
    'write_storage_benchmark_result_json',
    'CacheAdapterArtifact',
    'CacheBuildConfig',
    'CacheGenerationResult',
    'DocumentKVWorkflow',
    'KVChunkGenerator',
    'SourceChunk',
    'SourceDocument',
    'TrainingAdapter',
    'TrainingArtifacts',
)
EXPECTED_PUBLIC_SUBMODULES = frozenset(
    (
    'adapter_scaffold',
    'admission',
    'benchmark_handoff_bundles',
    'benchmark_handoffs',
    'benchmark_dataset_sources',
    'benchmark_plan',
    'benchmark_plan_executor',
    'benchmark_runner',
    'benchmarks',
    'cache',
    'databricks_engine_probe_job',
    'databricks_job',
    'databricks_runs',
    'databricks_sglang_smoke_job',
    'databricks_storage_benchmark_job',
    'databricks_vllm_smoke_job',
    'dependency_freshness',
    'dataset_prep',
    'engine',
    'engine_adapters',
    'engine_launch_config',
    'engine_probe',
    'engine_protocol',
    'github_governance',
    'kvpack',
    'legacy_compatibility',
    'live_server',
    'manifest',
    'materializer',
    'model_profiles',
    'models',
    'native_probe_factories',
    'openai_compatible',
    'planner',
    'pr_evidence',
    'probe_fixtures',
    'release_bundle',
    'release_evidence',
    'repository_hygiene',
    'runtime_telemetry',
    'service',
    'serving_env',
    'sglang_runtime_preflight',
    'sglang_smoke',
    'storage',
    'storage_benchmark',
    'template_resources',
    'transformers_generator',
    'vllm_runtime_contract_data',
    'vllm_runtime_preflight',
    'vllm_smoke',
    'workflow',
    )
)
EXPECTED_CONSOLE_SCRIPTS = {
    'cachet-benchmark-handoff-bundles': 'cachet.benchmark_handoffs:bundle_main',
    'cachet-benchmark-handoff-manifest': 'cachet.benchmark_handoffs:manifest_main',
    'cachet-benchmark-handoffs': 'cachet.benchmark_handoffs:main',
    'cachet-benchmark-plan': 'cachet.benchmark_plan:main',
    'cachet-databricks-job': 'cachet.databricks_job:main',
    'cachet-databricks-runs': 'cachet.databricks_runs:main',
    'cachet-dependency-freshness': 'cachet.dependency_freshness:main',
    'cachet-engine-launch-config': 'cachet.engine_launch_config:main',
    'cachet-engine-probe': 'cachet.engine_probe:main',
    'cachet-engine-probe-databricks-job': 'cachet.databricks_engine_probe_job:main',
    'cachet-engine-probe-fixture': 'cachet.probe_fixtures:main',
    'cachet-github-governance': 'cachet.github_governance:main',
    'cachet-native-probe-factories': 'cachet.native_probe_factories:main',
    'cachet-native-probe-scaffold': 'cachet.adapter_scaffold:main',
    'cachet-pr-evidence': 'cachet.pr_evidence:main',
    'cachet-release-bundle': 'cachet.release_bundle:main',
    'cachet-release-evidence': 'cachet.release_evidence:main',
    'cachet-repository-hygiene': 'cachet.repository_hygiene:main',
    'cachet-run-benchmark-plan': 'cachet.benchmark_plan_executor:main',
    'cachet-serving-env': 'cachet.serving_env:main',
    'cachet-sglang-runtime-preflight': 'cachet.sglang_runtime_preflight:main',
    'cachet-sglang-smoke': 'cachet.sglang_smoke:main',
    'cachet-sglang-smoke-databricks-job': 'cachet.databricks_sglang_smoke_job:main',
    'cachet-storage-benchmark': 'cachet.storage_benchmark:main',
    'cachet-storage-benchmark-databricks-job': 'cachet.databricks_storage_benchmark_job:main',
    'cachet-templates': 'cachet.template_resources:main',
    'cachet-vllm-runtime-preflight': 'cachet.vllm_runtime_preflight:main',
    'cachet-vllm-smoke': 'cachet.vllm_smoke:main',
    'cachet-vllm-smoke-databricks-job': 'cachet.databricks_vllm_smoke_job:main',
    'document-kv-benchmark-handoff-bundles': 'document_kv_cache.benchmark_handoffs:bundle_main',
    'document-kv-benchmark-handoff-manifest': 'document_kv_cache.benchmark_handoffs:manifest_main',
    'document-kv-benchmark-handoffs': 'document_kv_cache.benchmark_handoffs:main',
    'document-kv-benchmark-plan': 'document_kv_cache.benchmark_plan:main',
    'document-kv-databricks-job': 'document_kv_cache.databricks_job:main',
    'document-kv-databricks-runs': 'document_kv_cache.databricks_runs:main',
    'document-kv-dependency-freshness': 'document_kv_cache.dependency_freshness:main',
    'document-kv-engine-launch-config': 'document_kv_cache.engine_launch_config:main',
    'document-kv-engine-probe': 'document_kv_cache.engine_probe:main',
    'document-kv-engine-probe-databricks-job': 'document_kv_cache.databricks_engine_probe_job:main',
    'document-kv-engine-probe-fixture': 'document_kv_cache.probe_fixtures:main',
    'document-kv-github-governance': 'document_kv_cache.github_governance:main',
    'document-kv-native-probe-factories': 'document_kv_cache.native_probe_factories:main',
    'document-kv-native-probe-scaffold': 'document_kv_cache.adapter_scaffold:main',
    'document-kv-pr-evidence': 'document_kv_cache.pr_evidence:main',
    'document-kv-release-bundle': 'document_kv_cache.release_bundle:main',
    'document-kv-release-evidence': 'document_kv_cache.release_evidence:main',
    'document-kv-repository-hygiene': 'document_kv_cache.repository_hygiene:main',
    'document-kv-run-benchmark-plan': 'document_kv_cache.benchmark_plan_executor:main',
    'document-kv-serving-env': 'document_kv_cache.serving_env:main',
    'document-kv-sglang-runtime-preflight': 'document_kv_cache.sglang_runtime_preflight:main',
    'document-kv-sglang-smoke': 'document_kv_cache.sglang_smoke:main',
    'document-kv-sglang-smoke-databricks-job': 'document_kv_cache.databricks_sglang_smoke_job:main',
    'document-kv-storage-benchmark': 'document_kv_cache.storage_benchmark:main',
    'document-kv-storage-benchmark-databricks-job': 'document_kv_cache.databricks_storage_benchmark_job:main',
    'document-kv-templates': 'document_kv_cache.template_resources:main',
    'document-kv-vllm-runtime-preflight': 'document_kv_cache.vllm_runtime_preflight:main',
    'document-kv-vllm-smoke': 'document_kv_cache.vllm_smoke:main',
    'document-kv-vllm-smoke-databricks-job': 'document_kv_cache.databricks_vllm_smoke_job:main',
}


def _cachet_stub_document_exports() -> tuple[set[str], list[str], list[str]]:
    stub_path = REPO_ROOT / "src" / "cachet" / "__init__.pyi"
    stub_tree = ast.parse(stub_path.read_text(encoding="utf-8"))

    exports: set[str] = set()
    non_reexport_aliases: list[str] = []
    missing_source_symbols: list[str] = []
    for node in stub_tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module is None or not node.module.startswith("document_kv_cache"):
            continue
        source_module = importlib.import_module(node.module)
        for alias in node.names:
            if alias.asname != alias.name:
                non_reexport_aliases.append(f"{node.module}.{alias.name} as {alias.asname}")
                continue
            if not hasattr(source_module, alias.name):
                missing_source_symbols.append(f"{node.module}.{alias.name}")
                continue
            exports.add(alias.asname)
    return exports, non_reexport_aliases, missing_source_symbols


def test_public_document_package_reexports_core_api():
    missing_exports = [
        name for name in EXPECTED_DOCUMENT_ROOT_EXPORTS if not hasattr(document_kv_cache, name)
    ]

    assert tuple(document_kv_cache.__all__) == EXPECTED_DOCUMENT_ROOT_EXPORTS
    assert tuple(cachet.__all__) == EXPECTED_DOCUMENT_ROOT_EXPORTS
    assert len(set(EXPECTED_DOCUMENT_ROOT_EXPORTS)) == len(EXPECTED_DOCUMENT_ROOT_EXPORTS)
    assert missing_exports == []
    assert document_kv_cache.DocumentKVRequest.__module__ == "document_kv_cache.models"
    assert document_kv_cache.DocumentKVService.__module__ == "document_kv_cache.service"
    assert document_kv_cache.DocumentKVWorkflow.__module__ == "document_kv_cache.workflow"
    assert document_kv_cache.CacheTier.__module__ == "document_kv_cache.cache"
    assert document_kv_cache.DiskRangeReader.__module__ == "document_kv_cache.storage"
    assert document_kv_cache.build_vllm_launch_config.__module__ == "document_kv_cache.engine_launch_config"
    assert document_kv_cache.vllm_native_probe_factory.__module__ == (
        "document_kv_cache.native_probe_factories"
    )
    assert "DocumentKVRequest" in dir(document_kv_cache)
    assert not hasattr(document_kv_cache, "RestaurantKVRequest")


def test_public_document_package_star_exports_are_document_first():
    assert "AdmissionQueue" in document_kv_cache.__all__
    assert "DocumentKVRequest" in document_kv_cache.__all__
    assert "DocumentKVService" in document_kv_cache.__all__
    assert "DocumentKVWorkflow" in document_kv_cache.__all__
    assert "build_v1_benchmark_plan" in document_kv_cache.__all__
    assert "build_vllm_launch_config" in document_kv_cache.__all__
    assert "vllm_native_probe_factory" in document_kv_cache.__all__
    assert "RestaurantKVRequest" not in document_kv_cache.__all__
    assert "RestaurantKVService" not in document_kv_cache.__all__
    assert "ChunkType" not in document_kv_cache.__all__

    star_namespace: dict[str, object] = {}
    exec("from document_kv_cache import *", star_namespace)

    assert star_namespace["AdmissionQueue"] is document_kv_cache.AdmissionQueue
    assert star_namespace["DocumentKVRequest"] is document_kv_cache.DocumentKVRequest
    assert star_namespace["DocumentKVService"] is document_kv_cache.DocumentKVService
    assert star_namespace["DocumentKVWorkflow"] is document_kv_cache.DocumentKVWorkflow
    assert star_namespace["CacheTier"] is document_kv_cache.CacheTier
    assert "RestaurantKVRequest" not in star_namespace



def test_public_roots_ignore_stale_external_legacy_package(tmp_path):
    legacy_package = tmp_path / ("restaurant" "_kv_serving")
    legacy_package.mkdir()
    (legacy_package / "__init__.py").write_text(
        "\n".join(
            [
                "__all__ = ['LegacyOnly', 'RestaurantKVRequest']",
                "LegacyOnly = object()",
                "RestaurantKVRequest = object()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import json",
                    "import cachet, document_kv_cache",
                    "print(json.dumps({",
                    "    'document_has_legacy_only': hasattr(document_kv_cache, 'LegacyOnly'),",
                    "    'document_has_restaurant_request': hasattr(document_kv_cache, 'RestaurantKVRequest'),",
                    "    'document_legacy_only_in_all': 'LegacyOnly' in document_kv_cache.__all__,",
                    "    'cachet_has_legacy_only': hasattr(cachet, 'LegacyOnly'),",
                    "    'cachet_has_restaurant_request': hasattr(cachet, 'RestaurantKVRequest'),",
                    "    'cachet_legacy_only_in_all': 'LegacyOnly' in cachet.__all__,",
                    "}, sort_keys=True))",
                ]
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(tmp_path), str(REPO_ROOT / "src"))),
        },
    )

    assert json.loads(result.stdout) == {
        "cachet_has_legacy_only": False,
        "cachet_has_restaurant_request": False,
        "cachet_legacy_only_in_all": False,
        "document_has_legacy_only": False,
        "document_has_restaurant_request": False,
        "document_legacy_only_in_all": False,
    }


def test_cachet_brand_facade_delegates_to_document_package():
    from cachet import (
        CacheTier,
        DocumentKVRequest,
        DocumentKVService,
        DocumentKVWorkflow,
        NATIVE_PROBE_ADAPTER_CONTRACT,
        NATIVE_PROBE_FACTORIES_RECORD_TYPE,
        NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_ATTR,
        NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_MODULE_ATTR,
        SGLANG_NATIVE_PROBE_DELEGATE_ENV,
        VLLM_NATIVE_PROBE_DELEGATE_ENV,
        native_probe_adapter_contract_to_record,
        native_probe_runtime_contract_to_record,
        write_builtin_native_probe_factories_record_json,
        write_engine_adapter_handoff_bundle,
        write_engine_adapter_payload,
    )

    stub_text = (REPO_ROOT / "src" / "cachet" / "__init__.pyi").read_text(encoding="utf-8")

    assert cachet.__all__ == document_kv_cache.__all__
    assert CacheTier is document_kv_cache.CacheTier
    assert DocumentKVRequest is document_kv_cache.DocumentKVRequest
    assert DocumentKVService is document_kv_cache.DocumentKVService
    assert DocumentKVWorkflow is document_kv_cache.DocumentKVWorkflow
    assert NATIVE_PROBE_ADAPTER_CONTRACT is document_kv_cache.NATIVE_PROBE_ADAPTER_CONTRACT
    assert NATIVE_PROBE_FACTORIES_RECORD_TYPE is document_kv_cache.NATIVE_PROBE_FACTORIES_RECORD_TYPE
    assert (
        NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_ATTR
        == document_kv_cache.NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_ATTR
    )
    assert (
        NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_MODULE_ATTR
        == document_kv_cache.NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_MODULE_ATTR
    )
    assert SGLANG_NATIVE_PROBE_DELEGATE_ENV == document_kv_cache.SGLANG_NATIVE_PROBE_DELEGATE_ENV
    assert VLLM_NATIVE_PROBE_DELEGATE_ENV == document_kv_cache.VLLM_NATIVE_PROBE_DELEGATE_ENV
    assert native_probe_adapter_contract_to_record is document_kv_cache.native_probe_adapter_contract_to_record
    assert native_probe_runtime_contract_to_record is document_kv_cache.native_probe_runtime_contract_to_record
    assert write_engine_adapter_handoff_bundle is document_kv_cache.write_engine_adapter_handoff_bundle
    assert write_engine_adapter_payload is document_kv_cache.write_engine_adapter_payload
    assert (
        write_builtin_native_probe_factories_record_json
        is document_kv_cache.write_builtin_native_probe_factories_record_json
    )
    assert cachet.storage is document_kv_cache.storage
    assert "storage" in dir(cachet)

    star_namespace: dict[str, object] = {}
    exec("from cachet import *", star_namespace)

    assert star_namespace["DocumentKVService"] is document_kv_cache.DocumentKVService
    assert star_namespace["CacheTier"] is document_kv_cache.CacheTier
    assert "RestaurantKVRequest" not in star_namespace
    assert not hasattr(cachet, "RestaurantKVRequest")
    assert "from document_kv_cache.cache import" in stub_text
    assert "import document_kv_cache.storage as storage" not in stub_text
    assert "CacheTier as CacheTier" in stub_text
    assert "DATABRICKS_AUTH_CHECK_RECORD_TYPE as DATABRICKS_AUTH_CHECK_RECORD_TYPE" in stub_text
    assert "DATABRICKS_PROFILE_AUTH_MODES as DATABRICKS_PROFILE_AUTH_MODES" in stub_text
    assert "check_databricks_auth as check_databricks_auth" in stub_text
    assert "DEFAULT_DATABRICKS_CONFIG_FILE as DEFAULT_DATABRICKS_CONFIG_FILE" in stub_text
    assert "databricks_workspace_config_from_profile as databricks_workspace_config_from_profile" in stub_text
    assert (
        "databricks_workspace_config_from_sdk_profile as databricks_workspace_config_from_sdk_profile"
        in stub_text
    )
    assert "NATIVE_PROBE_ADAPTER_CONTRACT as NATIVE_PROBE_ADAPTER_CONTRACT" in stub_text
    assert "NATIVE_PROBE_FACTORIES_RECORD_TYPE as NATIVE_PROBE_FACTORIES_RECORD_TYPE" in stub_text
    assert (
        "NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_ATTR as NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_ATTR"
        in stub_text
    )
    assert (
        "NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_MODULE_ATTR as "
        "NATIVE_PROBE_DELEGATE_RUNTIME_CONTRACT_MODULE_ATTR"
        in stub_text
    )
    assert "SGLANG_NATIVE_PROBE_DELEGATE_ENV as SGLANG_NATIVE_PROBE_DELEGATE_ENV" in stub_text
    assert "VLLM_NATIVE_PROBE_DELEGATE_ENV as VLLM_NATIVE_PROBE_DELEGATE_ENV" in stub_text
    assert "native_probe_adapter_contract_to_record as native_probe_adapter_contract_to_record" in stub_text
    assert "native_probe_runtime_contract_to_record as native_probe_runtime_contract_to_record" in stub_text
    assert (
        "write_builtin_native_probe_factories_record_json as write_builtin_native_probe_factories_record_json"
        in stub_text
    )
    assert "write_engine_adapter_handoff_bundle as write_engine_adapter_handoff_bundle" in stub_text
    assert "write_engine_adapter_payload as write_engine_adapter_payload" in stub_text
    assert "from document_kv_cache.models import" in stub_text
    assert "DocumentKVRequest as DocumentKVRequest" in stub_text
    assert "RestaurantKVRequest" not in stub_text


def test_cachet_typing_stub_tracks_document_public_api():
    document_exports = set(EXPECTED_DOCUMENT_ROOT_EXPORTS)
    stub_exports, non_reexport_aliases, missing_source_symbols = _cachet_stub_document_exports()

    assert set(document_kv_cache.__all__) == document_exports
    assert non_reexport_aliases == []
    assert missing_source_symbols == []
    assert stub_exports == document_exports
    assert "RestaurantKVRequest" not in stub_exports
    assert "RestaurantKVService" not in stub_exports


def test_public_cli_submodules_are_importable_under_document_namespace():
    public_submodules = tuple(sorted(EXPECTED_PUBLIC_SUBMODULES))

    assert document_kv_cache._PUBLIC_SUBMODULES == EXPECTED_PUBLIC_SUBMODULES
    assert cachet._PUBLIC_SUBMODULES == EXPECTED_PUBLIC_SUBMODULES
    modules = {
        name: importlib.import_module(f"document_kv_cache.{name}")
        for name in public_submodules
    }
    cachet_modules = {
        name: importlib.import_module(f"cachet.{name}")
        for name in public_submodules
    }

    assert {name: module.__name__ for name, module in modules.items()} == {
        name: f"document_kv_cache.{name}"
        for name in public_submodules
    }
    assert cachet_modules == modules
    assert {name: getattr(cachet, name) for name in public_submodules} == modules
    assert modules["benchmark_plan"].BenchmarkPlanConfig.__module__ == (
        "document_kv_cache.benchmark_plan"
    )
    assert modules["benchmark_plan_executor"].BenchmarkCommandResult.__module__ == (
        "document_kv_cache.benchmark_plan_executor"
    )
    assert modules["benchmarks"].BenchmarkExample.__module__ == "document_kv_cache.benchmarks"
    assert modules["benchmark_runner"].BenchmarkGeneration.__module__ == (
        "document_kv_cache.benchmark_runner"
    )
    assert modules["dataset_prep"].convert_v1_jsonl.__module__ == "document_kv_cache.dataset_prep"
    assert modules["storage"].DiskRangeReader.__module__ == "document_kv_cache.storage"
    assert modules["release_bundle"].ReleaseBundle.__module__ == "document_kv_cache.release_bundle"
    assert modules["native_probe_factories"].vllm_native_probe_factory.__module__ == (
        "document_kv_cache.native_probe_factories"
    )
    assert modules["databricks_job"].DatabricksBenchmarkJobConfig.__module__ == (
        "document_kv_cache.databricks_job"
    )
    assert "scheduler" not in document_kv_cache._PUBLIC_SUBMODULES



def test_cachet_cli_module_facades_execute_with_python_m():
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cachet.benchmark_plan",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "usage:" in result.stdout
    assert "benchmark_plan" in result.stdout


def test_cachet_adapter_facades_delegate_to_vendored_compatibility_packages():
    cachet_vllm = importlib.import_module("cachet.adapters.vllm")
    cachet_sglang = importlib.import_module("cachet.adapters.sglang")
    vllm_adapter = importlib.import_module("vllm_kv_injection")
    sglang_adapter = importlib.import_module("sglang_kv_injection")

    assert cachet_vllm is vllm_adapter
    assert cachet_sglang is sglang_adapter
    assert cachet_vllm.DocumentKVNativeProvider is vllm_adapter.DocumentKVNativeProvider
    assert cachet_sglang.DocumentKVHiCacheBackend is sglang_adapter.DocumentKVHiCacheBackend


def test_cachet_adapter_submodules_delegate_to_vendored_compatibility_submodules():
    for cachet_module_name, adapter_module_name in (
        ("cachet.adapters.vllm.probe", "vllm_kv_injection.probe"),
        ("cachet.adapters.vllm.protocol", "vllm_kv_injection.protocol"),
        (
            "cachet.adapters.vllm.vllm_dynamic_connector",
            "vllm_kv_injection.vllm_dynamic_connector",
        ),
        (
            "cachet.adapters.vllm.vllm_native_provider",
            "vllm_kv_injection.vllm_native_provider",
        ),
        ("cachet.adapters.sglang.probe", "sglang_kv_injection.probe"),
        ("cachet.adapters.sglang.protocol", "sglang_kv_injection.protocol"),
        (
            "cachet.adapters.sglang.sglang_dynamic_backend",
            "sglang_kv_injection.sglang_dynamic_backend",
        ),
        (
            "cachet.adapters.sglang.sglang_request_metadata_bridge",
            "sglang_kv_injection.sglang_request_metadata_bridge",
        ),
    ):
        cachet_module = importlib.import_module(cachet_module_name)
        adapter_module = importlib.import_module(adapter_module_name)

        assert cachet_module is adapter_module


def test_cachet_adapter_facades_expose_native_runtime_symbols():
    from cachet.adapters.sglang.sglang_dynamic_backend import (
        DocumentKVHiCacheBackend as CachetSGLangBackend,
    )
    from cachet.adapters.vllm.vllm_dynamic_connector import (
        DocumentKVConnector as CachetVLLMConnector,
    )
    from cachet.adapters.vllm.vllm_native_provider import (
        DocumentKVNativeProvider as CachetVLLMProvider,
    )
    from sglang_kv_injection.sglang_dynamic_backend import (
        DocumentKVHiCacheBackend,
    )
    from vllm_kv_injection.vllm_dynamic_connector import DocumentKVConnector
    from vllm_kv_injection.vllm_native_provider import DocumentKVNativeProvider

    assert CachetVLLMConnector is DocumentKVConnector
    assert CachetVLLMProvider is DocumentKVNativeProvider
    assert CachetSGLangBackend is DocumentKVHiCacheBackend


def test_cachet_adapter_submodules_reuse_preloaded_vendored_submodules():
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, json; "
                "vllm_target = importlib.import_module('vllm_kv_injection.probe'); "
                "sglang_target = importlib.import_module('sglang_kv_injection.probe'); "
                "vllm_alias = importlib.import_module('cachet.adapters.vllm.probe'); "
                "sglang_alias = importlib.import_module('cachet.adapters.sglang.probe'); "
                "vllm_provider_target = importlib.import_module('vllm_kv_injection.vllm_native_provider'); "
                "sglang_backend_target = importlib.import_module('sglang_kv_injection.sglang_dynamic_backend'); "
                "vllm_provider_alias = importlib.import_module('cachet.adapters.vllm.vllm_native_provider'); "
                "sglang_backend_alias = importlib.import_module('cachet.adapters.sglang.sglang_dynamic_backend'); "
                "print(json.dumps({"
                "'vllm': vllm_alias is vllm_target, "
                "'sglang': sglang_alias is sglang_target, "
                "'vllm_provider': vllm_provider_alias is vllm_provider_target, "
                "'sglang_backend': sglang_backend_alias is sglang_backend_target"
                "}, sort_keys=True))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert json.loads(result.stdout) == {
        "sglang": True,
        "sglang_backend": True,
        "vllm": True,
        "vllm_provider": True,
    }


def test_public_document_submodules_have_curated_star_import_surfaces():
    assert document_kv_cache._PUBLIC_SUBMODULES == EXPECTED_PUBLIC_SUBMODULES
    for module_name in sorted(EXPECTED_PUBLIC_SUBMODULES):
        module = importlib.import_module(f"document_kv_cache.{module_name}")
        exports = getattr(module, "__all__", None)

        if exports is None:
            assert module_name == "template_resources"
            continue
        assert isinstance(exports, (tuple, list)), module_name
        assert exports, module_name
        assert all(isinstance(name, str) and name for name in exports), module_name
        assert all(not name.startswith("_") for name in exports), module_name
        assert all("Restaurant" not in name for name in exports), module_name

    cache_star_namespace: dict[str, object] = {}
    storage_star_namespace: dict[str, object] = {}
    workflow_star_namespace: dict[str, object] = {}
    exec("from document_kv_cache.cache import *", cache_star_namespace)
    exec("from document_kv_cache.storage import *", storage_star_namespace)
    exec("from document_kv_cache.workflow import *", workflow_star_namespace)

    assert cache_star_namespace["CacheTier"] is document_kv_cache.CacheTier
    assert storage_star_namespace["DiskRangeReader"] is document_kv_cache.DiskRangeReader
    assert workflow_star_namespace["DocumentKVWorkflow"] is document_kv_cache.DocumentKVWorkflow
    assert "RestaurantKVRequest" not in workflow_star_namespace



def test_scheduler_compatibility_module_stays_out_of_root_facades():
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import json",
                    "import cachet, document_kv_cache",
                    "summary = {}",
                    "for root_module in (document_kv_cache, cachet):",
                    "    root_star_namespace = {}",
                    "    exec(f'from {root_module.__name__} import *', root_star_namespace)",
                    "    summary[root_module.__name__] = {",
                    "        'in_all': 'scheduler' in root_module.__all__,",
                    "        'in_dir': 'scheduler' in dir(root_module),",
                    "        'hasattr': hasattr(root_module, 'scheduler'),",
                    "        'in_star': 'scheduler' in root_star_namespace,",
                    "    }",
                    "print(json.dumps(summary, sort_keys=True))",
                ]
            ),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    fresh_root_surfaces = json.loads(probe.stdout)

    assert fresh_root_surfaces == {
        "cachet": {"hasattr": False, "in_all": False, "in_dir": False, "in_star": False},
        "document_kv_cache": {"hasattr": False, "in_all": False, "in_dir": False, "in_star": False},
    }

    admission = importlib.import_module("document_kv_cache.admission")
    document_scheduler = importlib.import_module("document_kv_cache.scheduler")
    document_star_namespace_after_import: dict[str, object] = {}
    cachet_star_namespace_after_import: dict[str, object] = {}
    exec("from document_kv_cache import *", document_star_namespace_after_import)
    exec("from cachet import *", cachet_star_namespace_after_import)

    assert document_scheduler.AdmissionQueue is admission.AdmissionQueue
    assert document_scheduler.PreparedRequest is admission.PreparedRequest
    assert "scheduler" not in document_kv_cache._PUBLIC_SUBMODULES
    assert "scheduler" not in document_kv_cache.__all__
    assert "scheduler" not in cachet.__all__
    assert "scheduler" not in document_star_namespace_after_import
    assert "scheduler" not in cachet_star_namespace_after_import


def test_package_level_submodule_imports_use_document_namespace_after_symbol_lookup():
    from document_kv_cache import CacheTier  # noqa: F401
    from document_kv_cache import cache

    assert cache.__name__ == "document_kv_cache.cache"
    assert CacheTier is cache.CacheTier
    assert cache.CacheTier.__module__ == "document_kv_cache.cache"
    assert cache.ChunkCacheResult.__module__ == "document_kv_cache.cache"


def test_poetry_metadata_excludes_removed_package_and_scripts():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    poetry = pyproject["tool"]["poetry"]
    scripts = pyproject["project"]["scripts"]
    included_packages = {package["include"] for package in poetry["packages"]}
    included_files = {include["path"] for include in poetry["include"]}
    old_import = "restaurant" "_kv_serving"
    old_script_prefix = "restaurant" "-kv-"

    assert included_packages == {
        "cachet",
        "document_kv_cache",
        "vllm_kv_injection",
        "sglang_kv_injection",
    }
    assert scripts == EXPECTED_CONSOLE_SCRIPTS
    assert old_import not in included_packages
    assert f"src/{old_import}/py.typed" not in included_files
    assert not any(name.startswith(old_script_prefix) for name in scripts)
    assert not any(old_import in target for target in scripts.values())
    assert scripts["cachet-benchmark-plan"] == "cachet.benchmark_plan:main"
    assert scripts["document-kv-benchmark-plan"] == "document_kv_cache.benchmark_plan:main"


def test_cachet_console_script_targets_resolve_through_cachet_facades():
    scripts = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "scripts"
    ]
    cachet_scripts = {
        script_name: target
        for script_name, target in scripts.items()
        if script_name.startswith("cachet-")
    }

    assert scripts == EXPECTED_CONSOLE_SCRIPTS
    assert cachet_scripts
    for script_name, target in sorted(cachet_scripts.items()):
        module_name, separator, attribute_name = target.partition(":")
        assert separator == ":", script_name
        assert module_name.startswith("cachet."), script_name
        module = importlib.import_module(module_name)
        assert hasattr(module, attribute_name), script_name
