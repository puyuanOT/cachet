import importlib
import tomllib
from pathlib import Path

import document_kv_cache
import restaurant_kv_serving


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_public_document_package_reexports_core_api():
    from document_kv_cache import (
        AdmissionQueue,
        BENCHMARK_RUN_RECORD_TYPE,
        ByteLRU,
        CacheGenerationMethod,
        CacheTier,
        ChunkCache,
        ChunkCacheResult,
        ChunkCacheStats,
        DATABRICKS_RUN_STATUS_RECORD_TYPE,
        DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE,
        DEDICATED_DATABRICKS_DATA_SECURITY_MODE,
        DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME,
        DEFAULT_DATABRICKS_HOST_ENV,
        DEFAULT_DATABRICKS_TOKEN_ENV,
        DEFAULT_DATABRICKS_STORAGE_BENCHMARK_RUN_NAME,
        DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME,
        ENGINE_PROBE_TARGETS_RECORD_TYPE,
        ENGINE_PROBE_TARGETS_SCHEMA_VERSION,
        ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND,
        ENGINE_KV_PROBE_METADATA_HANDOFF_JSON,
        ENGINE_KV_PROBE_METADATA_PAYLOAD_URI,
        ENGINE_KV_PROBE_METADATA_PROBE_FACTORY,
        ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE,
        ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION,
        RELEASE_BUNDLE_ARTIFACT_ROLES,
        RELEASE_BUNDLE_MANIFEST_FILENAME,
        RELEASE_BUNDLE_RECORD_TYPE,
        RELEASE_EVIDENCE_ARTIFACT_ROLES,
        SERVING_ENVIRONMENT_PROFILES_RECORD_TYPE,
        DatabricksSingleNodeG5ClusterConfig,
        DatabricksVLLMSmokeJobConfig,
        DatabricksWorkspaceConfig,
        DocumentChunkRole,
        DocumentChunkType,
        DocumentKVRequest,
        DocumentKVService,
        DocumentKVWorkflow,
        DiskRangeReader,
        EngineReadyRequest,
        EngineProbePlanConfig,
        EngineKVProbeConfig,
        EngineKVProbeFactory,
        InMemoryManifestStore,
        CachePlanner,
        CacheRequest,
        KVCacheKey,
        KVMaterializer,
        KVModelProfile,
        KVStorageLayout,
        ManifestStore,
        MODEL_PROFILE_RECORD_TYPE,
        MaterializedKV,
        ModelProfileDefinition,
        ModelProfileRegistry,
        QWEN3_4B_INSTRUCT_PROFILE,
        GPT55_REVIEW_OUTCOMES,
        PR_EVIDENCE_RECORD_TYPE,
        PR_EVIDENCE_VALIDATION_RECORD_TYPE,
        RELEASE_EVIDENCE_INPUT_STATUS_RECORD_TYPE,
        RELEASE_EVIDENCE_RECORD_TYPE,
        RELEASE_STORAGE_BENCHMARK_READERS,
        REQUIRED_ENGINE_PROBE_BACKENDS,
        STORAGE_BENCHMARK_RECORD_TYPE,
        DatabricksBenchmarkJobConfig,
        DatabricksEngineProbeJobConfig,
        DatabricksEngineProbeMatrixJobConfig,
        DatabricksEngineProbeTargetConfig,
        DatabricksEngineProbeTargetsFile,
        DatabricksStorageBenchmarkJobConfig,
        PreparedRequest,
        PullRequestEvidence,
        ReleaseEvidence,
        ReleaseBundlePlanConfig,
        ReleaseEvidenceInputFileStatus,
        ReleaseEvidenceInputStatus,
        ReleaseEvidencePlanConfig,
        ReleaseEvidenceArtifactSource,
        ReleaseBundle,
        ReleaseBundleArtifact,
        SegmentedMaterializedKV,
        ServingEngineConnector,
        ServingEnvironmentProfile,
        NativeProbeFactoryInspection,
        NativeProbeFactoryUnavailable,
        CacheAdapterArtifact,
        CacheBuildConfig,
        CacheGenerationResult,
        KVChunkGenerator,
        SourceChunk,
        SourceDocument,
        TrainingAdapter,
        TrainingArtifacts,
        DOCUMENT_CHUNK_TYPES,
        LEGACY_RESTAURANT_CHUNK_TYPES,
        StorageBenchmarkConfig,
        StorageBenchmarkEvidence,
        StorageBenchmarkPlanConfig,
        SINGLE_USER_DATABRICKS_DATA_SECURITY_MODES,
        SUPPORTED_STORAGE_BENCHMARK_READERS,
        build_databricks_engine_probe_run_submit_payload,
        build_databricks_engine_probe_matrix_run_submit_payload,
        build_databricks_run_submit_payload,
        build_databricks_storage_benchmark_run_submit_payload,
        build_databricks_vllm_smoke_run_submit_payload,
        build_engine_ready_request,
        build_handle_from_materialized,
        build_single_node_g5_cluster,
        build_release_bundle,
        chunk_type_role,
        chunk_type_sort_order,
        databricks_workspace_config_from_env,
        evaluate_release_storage_benchmark_evidence,
        evaluate_storage_benchmark_evidence,
        engine_probe_targets_to_record,
        default_model_profile_registry,
        evaluate_pr_evidence,
        evaluate_pr_evidence_directory,
        evaluate_pr_evidence_file,
        evaluate_release_evidence,
        inspect_release_evidence_input_files,
        dtype_byte_width,
        kv_storage_layout_from_value,
        builtin_native_probe_factories_to_record,
        builtin_native_probe_factory_path,
        inspect_builtin_native_probe_factories,
        inspect_builtin_native_probe_factory,
        model_profile_definition_from_record,
        model_profile_definition_to_record,
        DTYPE_BYTE_WIDTHS,
        AttentionMechanism,
        KVCacheHandle,
        KVLayout,
        KVSegment,
        release_bundle_to_record,
        pr_evidence_validation_to_record,
        pr_evidence_to_record,
        read_model_profile_definition_json,
        read_databricks_engine_probe_targets_json,
        read_databricks_engine_probe_targets_file_json,
        release_evidence_input_status_to_record,
        run_engine_kv_connector_probe,
        native_probe_factory_inspection_to_record,
        serving_environment_profile,
        serving_environment_profile_to_record,
        serving_environment_profiles,
        serving_environment_profiles_to_record,
        sglang_native_probe_factory,
        vllm_native_probe_factory,
        summarize_databricks_run,
        summarize_databricks_run_submit_payload,
        validate_aws_g5_node_type,
        write_model_profile_definition_json,
        write_engine_probe_targets_json,
    )
    from document_kv_cache import (
        admission,
        cache,
        databricks_job,
        databricks_runs,
        engine,
        engine_protocol,
        engine_probe,
        materializer,
        model_profiles,
        native_probe_factories,
        service,
        serving_env,
        workflow,
    )

    assert AdmissionQueue is admission.AdmissionQueue
    assert AdmissionQueue is restaurant_kv_serving.AdmissionQueue
    assert AdmissionQueue.__module__ == "document_kv_cache.admission"
    assert BENCHMARK_RUN_RECORD_TYPE is restaurant_kv_serving.BENCHMARK_RUN_RECORD_TYPE
    assert ByteLRU is cache.ByteLRU
    assert CacheGenerationMethod is restaurant_kv_serving.CacheGenerationMethod
    assert CacheGenerationMethod.KV_PACKET.value == "kv_packet"
    assert CacheTier is cache.CacheTier
    assert CacheTier is restaurant_kv_serving.CacheTier
    assert ChunkCache is cache.ChunkCache
    assert ChunkCacheResult is cache.ChunkCacheResult
    assert ChunkCacheResult is restaurant_kv_serving.ChunkCacheResult
    assert ChunkCacheStats is cache.ChunkCacheStats
    assert ChunkCacheStats is restaurant_kv_serving.ChunkCacheStats
    assert DATABRICKS_RUN_STATUS_RECORD_TYPE is restaurant_kv_serving.DATABRICKS_RUN_STATUS_RECORD_TYPE
    assert DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE is restaurant_kv_serving.DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE
    assert DEDICATED_DATABRICKS_DATA_SECURITY_MODE is restaurant_kv_serving.DEDICATED_DATABRICKS_DATA_SECURITY_MODE
    assert DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME is restaurant_kv_serving.DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME
    assert DEFAULT_DATABRICKS_HOST_ENV is restaurant_kv_serving.DEFAULT_DATABRICKS_HOST_ENV
    assert DEFAULT_DATABRICKS_TOKEN_ENV is restaurant_kv_serving.DEFAULT_DATABRICKS_TOKEN_ENV
    assert (
        DEFAULT_DATABRICKS_STORAGE_BENCHMARK_RUN_NAME
        is restaurant_kv_serving.DEFAULT_DATABRICKS_STORAGE_BENCHMARK_RUN_NAME
    )
    assert DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME is restaurant_kv_serving.DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME
    assert ENGINE_PROBE_TARGETS_RECORD_TYPE is restaurant_kv_serving.ENGINE_PROBE_TARGETS_RECORD_TYPE
    assert ENGINE_PROBE_TARGETS_SCHEMA_VERSION is restaurant_kv_serving.ENGINE_PROBE_TARGETS_SCHEMA_VERSION
    assert ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND == restaurant_kv_serving.ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND
    assert ENGINE_KV_PROBE_METADATA_HANDOFF_JSON == restaurant_kv_serving.ENGINE_KV_PROBE_METADATA_HANDOFF_JSON
    assert ENGINE_KV_PROBE_METADATA_PAYLOAD_URI == restaurant_kv_serving.ENGINE_KV_PROBE_METADATA_PAYLOAD_URI
    assert ENGINE_KV_PROBE_METADATA_PROBE_FACTORY == restaurant_kv_serving.ENGINE_KV_PROBE_METADATA_PROBE_FACTORY
    assert (
        ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE
        == restaurant_kv_serving.ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE
    )
    assert (
        ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION
        == restaurant_kv_serving.ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION
    )
    assert RELEASE_BUNDLE_ARTIFACT_ROLES is restaurant_kv_serving.RELEASE_BUNDLE_ARTIFACT_ROLES
    assert RELEASE_BUNDLE_MANIFEST_FILENAME is restaurant_kv_serving.RELEASE_BUNDLE_MANIFEST_FILENAME
    assert RELEASE_BUNDLE_RECORD_TYPE is restaurant_kv_serving.RELEASE_BUNDLE_RECORD_TYPE
    assert RELEASE_EVIDENCE_ARTIFACT_ROLES is restaurant_kv_serving.RELEASE_EVIDENCE_ARTIFACT_ROLES
    assert (
        SERVING_ENVIRONMENT_PROFILES_RECORD_TYPE
        is restaurant_kv_serving.SERVING_ENVIRONMENT_PROFILES_RECORD_TYPE
    )
    assert DatabricksBenchmarkJobConfig is databricks_job.DatabricksBenchmarkJobConfig
    assert DatabricksSingleNodeG5ClusterConfig is databricks_job.DatabricksSingleNodeG5ClusterConfig
    assert build_databricks_run_submit_payload is databricks_job.build_databricks_run_submit_payload
    assert build_single_node_g5_cluster is databricks_job.build_single_node_g5_cluster
    assert validate_aws_g5_node_type is databricks_job.validate_aws_g5_node_type
    assert DatabricksVLLMSmokeJobConfig is restaurant_kv_serving.DatabricksVLLMSmokeJobConfig
    assert DatabricksWorkspaceConfig is databricks_runs.DatabricksWorkspaceConfig
    assert issubclass(restaurant_kv_serving.DatabricksWorkspaceConfig, databricks_runs.DatabricksWorkspaceConfig)
    assert DocumentChunkRole is restaurant_kv_serving.DocumentChunkRole
    assert DocumentChunkType is restaurant_kv_serving.DocumentChunkType
    assert DocumentKVRequest is restaurant_kv_serving.DocumentKVRequest
    assert DocumentKVService is service.DocumentKVService
    assert DocumentKVService is restaurant_kv_serving.DocumentKVService
    assert DocumentKVService.__module__ == "document_kv_cache.service"
    assert DocumentKVWorkflow is workflow.DocumentKVWorkflow
    assert DocumentKVWorkflow is restaurant_kv_serving.DocumentKVWorkflow
    assert DocumentKVWorkflow.__module__ == "document_kv_cache.workflow"
    assert CacheAdapterArtifact is workflow.CacheAdapterArtifact
    assert CacheAdapterArtifact is restaurant_kv_serving.CacheAdapterArtifact
    assert CacheAdapterArtifact.__module__ == "document_kv_cache.workflow"
    assert CacheBuildConfig is workflow.CacheBuildConfig
    assert CacheBuildConfig is restaurant_kv_serving.CacheBuildConfig
    assert CacheBuildConfig.__module__ == "document_kv_cache.workflow"
    assert CacheGenerationResult is workflow.CacheGenerationResult
    assert CacheGenerationResult is restaurant_kv_serving.CacheGenerationResult
    assert CacheGenerationResult.__module__ == "document_kv_cache.workflow"
    assert KVChunkGenerator is workflow.KVChunkGenerator
    assert KVChunkGenerator is restaurant_kv_serving.KVChunkGenerator
    assert SourceChunk is workflow.SourceChunk
    assert SourceChunk is restaurant_kv_serving.SourceChunk
    assert SourceChunk.__module__ == "document_kv_cache.workflow"
    assert SourceDocument is workflow.SourceDocument
    assert SourceDocument is restaurant_kv_serving.SourceDocument
    assert SourceDocument.__module__ == "document_kv_cache.workflow"
    assert TrainingAdapter is workflow.TrainingAdapter
    assert TrainingAdapter is restaurant_kv_serving.TrainingAdapter
    assert TrainingArtifacts is workflow.TrainingArtifacts
    assert TrainingArtifacts is restaurant_kv_serving.TrainingArtifacts
    assert TrainingArtifacts.__module__ == "document_kv_cache.workflow"
    assert EngineReadyRequest is engine.EngineReadyRequest
    assert EngineReadyRequest is restaurant_kv_serving.EngineReadyRequest
    assert EngineReadyRequest.__module__ == "document_kv_cache.engine"
    assert DTYPE_BYTE_WIDTHS is engine_protocol.DTYPE_BYTE_WIDTHS
    assert DTYPE_BYTE_WIDTHS is restaurant_kv_serving.DTYPE_BYTE_WIDTHS
    assert AttentionMechanism is engine_protocol.AttentionMechanism
    assert AttentionMechanism is restaurant_kv_serving.AttentionMechanism
    assert AttentionMechanism.__module__ == "document_kv_cache.engine_protocol"
    assert EngineProbePlanConfig is restaurant_kv_serving.EngineProbePlanConfig
    assert EngineKVProbeConfig is engine_probe.EngineKVProbeConfig
    assert issubclass(restaurant_kv_serving.EngineKVProbeConfig, engine_probe.EngineKVProbeConfig)
    assert EngineKVProbeFactory is engine_probe.EngineKVProbeFactory
    assert InMemoryManifestStore is restaurant_kv_serving.InMemoryManifestStore
    assert CachePlanner is restaurant_kv_serving.CachePlanner
    assert KVCacheKey is restaurant_kv_serving.KVCacheKey
    assert KVMaterializer is materializer.KVMaterializer
    assert KVMaterializer is restaurant_kv_serving.KVMaterializer
    assert KVCacheHandle is engine_protocol.KVCacheHandle
    assert KVCacheHandle is restaurant_kv_serving.KVCacheHandle
    assert KVCacheHandle.__module__ == "document_kv_cache.engine_protocol"
    assert KVLayout is engine_protocol.KVLayout
    assert KVLayout is restaurant_kv_serving.KVLayout
    assert KVLayout.__module__ == "document_kv_cache.engine_protocol"
    assert KVSegment is engine_protocol.KVSegment
    assert KVSegment is restaurant_kv_serving.KVSegment
    assert KVSegment.__module__ == "document_kv_cache.engine_protocol"
    assert KVStorageLayout is engine_protocol.KVStorageLayout
    assert KVStorageLayout is restaurant_kv_serving.KVStorageLayout
    assert KVStorageLayout.__module__ == "document_kv_cache.engine_protocol"
    assert dtype_byte_width is engine_protocol.dtype_byte_width
    assert dtype_byte_width is restaurant_kv_serving.dtype_byte_width
    assert dtype_byte_width.__module__ == "document_kv_cache.engine_protocol"
    assert kv_storage_layout_from_value is engine_protocol.kv_storage_layout_from_value
    assert kv_storage_layout_from_value is restaurant_kv_serving.kv_storage_layout_from_value
    assert kv_storage_layout_from_value.__module__ == "document_kv_cache.engine_protocol"
    assert ManifestStore is restaurant_kv_serving.ManifestStore
    assert MaterializedKV is materializer.MaterializedKV
    assert MaterializedKV is restaurant_kv_serving.MaterializedKV
    assert MODEL_PROFILE_RECORD_TYPE is model_profiles.MODEL_PROFILE_RECORD_TYPE
    assert MODEL_PROFILE_RECORD_TYPE is restaurant_kv_serving.MODEL_PROFILE_RECORD_TYPE
    assert KVModelProfile is model_profiles.KVModelProfile
    assert KVModelProfile is restaurant_kv_serving.KVModelProfile
    assert KVModelProfile.__module__ == "document_kv_cache.model_profiles"
    assert ModelProfileDefinition is model_profiles.ModelProfileDefinition
    assert ModelProfileDefinition is restaurant_kv_serving.ModelProfileDefinition
    assert ModelProfileDefinition.__module__ == "document_kv_cache.model_profiles"
    assert ModelProfileRegistry is model_profiles.ModelProfileRegistry
    assert ModelProfileRegistry is restaurant_kv_serving.ModelProfileRegistry
    assert ModelProfileRegistry.__module__ == "document_kv_cache.model_profiles"
    assert QWEN3_4B_INSTRUCT_PROFILE is model_profiles.QWEN3_4B_INSTRUCT_PROFILE
    assert QWEN3_4B_INSTRUCT_PROFILE is restaurant_kv_serving.QWEN3_4B_INSTRUCT_PROFILE
    assert GPT55_REVIEW_OUTCOMES is restaurant_kv_serving.GPT55_REVIEW_OUTCOMES
    assert PR_EVIDENCE_RECORD_TYPE is restaurant_kv_serving.PR_EVIDENCE_RECORD_TYPE
    assert PR_EVIDENCE_VALIDATION_RECORD_TYPE is restaurant_kv_serving.PR_EVIDENCE_VALIDATION_RECORD_TYPE
    assert RELEASE_EVIDENCE_INPUT_STATUS_RECORD_TYPE is restaurant_kv_serving.RELEASE_EVIDENCE_INPUT_STATUS_RECORD_TYPE
    assert RELEASE_EVIDENCE_RECORD_TYPE is restaurant_kv_serving.RELEASE_EVIDENCE_RECORD_TYPE
    assert RELEASE_STORAGE_BENCHMARK_READERS is restaurant_kv_serving.RELEASE_STORAGE_BENCHMARK_READERS
    assert REQUIRED_ENGINE_PROBE_BACKENDS is restaurant_kv_serving.REQUIRED_ENGINE_PROBE_BACKENDS
    assert STORAGE_BENCHMARK_RECORD_TYPE is restaurant_kv_serving.STORAGE_BENCHMARK_RECORD_TYPE
    assert DatabricksEngineProbeJobConfig is restaurant_kv_serving.DatabricksEngineProbeJobConfig
    assert DatabricksEngineProbeMatrixJobConfig is restaurant_kv_serving.DatabricksEngineProbeMatrixJobConfig
    assert DatabricksEngineProbeTargetConfig is restaurant_kv_serving.DatabricksEngineProbeTargetConfig
    assert DatabricksEngineProbeTargetsFile is restaurant_kv_serving.DatabricksEngineProbeTargetsFile
    assert DatabricksStorageBenchmarkJobConfig is restaurant_kv_serving.DatabricksStorageBenchmarkJobConfig
    assert PreparedRequest is admission.PreparedRequest
    assert PreparedRequest is restaurant_kv_serving.PreparedRequest
    assert PreparedRequest.__module__ == "document_kv_cache.admission"
    assert PullRequestEvidence is restaurant_kv_serving.PullRequestEvidence
    assert PullRequestEvidence.__module__ == "document_kv_cache.pr_evidence"
    assert ReleaseEvidence is restaurant_kv_serving.ReleaseEvidence
    assert ReleaseBundlePlanConfig is restaurant_kv_serving.ReleaseBundlePlanConfig
    assert ReleaseEvidenceInputFileStatus is restaurant_kv_serving.ReleaseEvidenceInputFileStatus
    assert ReleaseEvidenceInputStatus is restaurant_kv_serving.ReleaseEvidenceInputStatus
    assert ReleaseEvidencePlanConfig is restaurant_kv_serving.ReleaseEvidencePlanConfig
    assert ReleaseEvidenceArtifactSource is restaurant_kv_serving.ReleaseEvidenceArtifactSource
    assert ReleaseBundle is restaurant_kv_serving.ReleaseBundle
    assert ReleaseBundleArtifact is restaurant_kv_serving.ReleaseBundleArtifact
    assert SegmentedMaterializedKV is materializer.SegmentedMaterializedKV
    assert SegmentedMaterializedKV is restaurant_kv_serving.SegmentedMaterializedKV
    assert ServingEngineConnector is engine.ServingEngineConnector
    assert ServingEngineConnector is restaurant_kv_serving.ServingEngineConnector
    assert ServingEnvironmentProfile is serving_env.ServingEnvironmentProfile
    assert ServingEnvironmentProfile is restaurant_kv_serving.ServingEnvironmentProfile
    assert ServingEnvironmentProfile.__module__ == "document_kv_cache.serving_env"
    assert NativeProbeFactoryInspection is native_probe_factories.NativeProbeFactoryInspection
    assert NativeProbeFactoryInspection is restaurant_kv_serving.NativeProbeFactoryInspection
    assert NativeProbeFactoryInspection.__module__ == "document_kv_cache.native_probe_factories"
    assert NativeProbeFactoryUnavailable is native_probe_factories.NativeProbeFactoryUnavailable
    assert NativeProbeFactoryUnavailable is restaurant_kv_serving.NativeProbeFactoryUnavailable
    assert NativeProbeFactoryUnavailable.__module__ == "document_kv_cache.native_probe_factories"
    assert DOCUMENT_CHUNK_TYPES is restaurant_kv_serving.DOCUMENT_CHUNK_TYPES
    assert LEGACY_RESTAURANT_CHUNK_TYPES is restaurant_kv_serving.LEGACY_RESTAURANT_CHUNK_TYPES
    assert SUPPORTED_STORAGE_BENCHMARK_READERS is restaurant_kv_serving.SUPPORTED_STORAGE_BENCHMARK_READERS
    assert StorageBenchmarkConfig is restaurant_kv_serving.StorageBenchmarkConfig
    assert StorageBenchmarkEvidence is restaurant_kv_serving.StorageBenchmarkEvidence
    assert StorageBenchmarkPlanConfig is restaurant_kv_serving.StorageBenchmarkPlanConfig
    assert SINGLE_USER_DATABRICKS_DATA_SECURITY_MODES is restaurant_kv_serving.SINGLE_USER_DATABRICKS_DATA_SECURITY_MODES
    assert (
        build_databricks_engine_probe_run_submit_payload
        is restaurant_kv_serving.build_databricks_engine_probe_run_submit_payload
    )
    assert (
        build_databricks_engine_probe_matrix_run_submit_payload
        is restaurant_kv_serving.build_databricks_engine_probe_matrix_run_submit_payload
    )
    assert (
        build_databricks_storage_benchmark_run_submit_payload
        is restaurant_kv_serving.build_databricks_storage_benchmark_run_submit_payload
    )
    assert (
        build_databricks_vllm_smoke_run_submit_payload
        is restaurant_kv_serving.build_databricks_vllm_smoke_run_submit_payload
    )
    assert build_engine_ready_request is engine.build_engine_ready_request
    assert build_engine_ready_request is restaurant_kv_serving.build_engine_ready_request
    assert build_engine_ready_request.__module__ == "document_kv_cache.engine"
    assert build_handle_from_materialized is engine.build_handle_from_materialized
    assert build_handle_from_materialized is restaurant_kv_serving.build_handle_from_materialized
    assert build_handle_from_materialized.__module__ == "document_kv_cache.engine"
    assert databricks_workspace_config_from_env is databricks_runs.databricks_workspace_config_from_env
    assert build_release_bundle is restaurant_kv_serving.build_release_bundle
    assert chunk_type_role is restaurant_kv_serving.chunk_type_role
    assert chunk_type_sort_order is restaurant_kv_serving.chunk_type_sort_order
    assert release_bundle_to_record is restaurant_kv_serving.release_bundle_to_record
    assert evaluate_release_storage_benchmark_evidence is restaurant_kv_serving.evaluate_release_storage_benchmark_evidence
    assert evaluate_release_evidence is restaurant_kv_serving.evaluate_release_evidence
    assert inspect_release_evidence_input_files is restaurant_kv_serving.inspect_release_evidence_input_files
    assert release_evidence_input_status_to_record is restaurant_kv_serving.release_evidence_input_status_to_record
    assert evaluate_storage_benchmark_evidence is restaurant_kv_serving.evaluate_storage_benchmark_evidence
    assert engine_probe_targets_to_record is restaurant_kv_serving.engine_probe_targets_to_record
    assert builtin_native_probe_factories_to_record is native_probe_factories.builtin_native_probe_factories_to_record
    assert builtin_native_probe_factories_to_record is restaurant_kv_serving.builtin_native_probe_factories_to_record
    assert builtin_native_probe_factory_path is native_probe_factories.builtin_native_probe_factory_path
    assert builtin_native_probe_factory_path is restaurant_kv_serving.builtin_native_probe_factory_path
    assert default_model_profile_registry is model_profiles.default_model_profile_registry
    assert default_model_profile_registry is restaurant_kv_serving.default_model_profile_registry
    assert evaluate_pr_evidence is restaurant_kv_serving.evaluate_pr_evidence
    assert evaluate_pr_evidence.__module__ == "document_kv_cache.pr_evidence"
    assert evaluate_pr_evidence_directory is restaurant_kv_serving.evaluate_pr_evidence_directory
    assert evaluate_pr_evidence_directory.__module__ == "document_kv_cache.pr_evidence"
    assert evaluate_pr_evidence_file is restaurant_kv_serving.evaluate_pr_evidence_file
    assert evaluate_pr_evidence_file.__module__ == "document_kv_cache.pr_evidence"
    assert model_profile_definition_from_record is model_profiles.model_profile_definition_from_record
    assert model_profile_definition_from_record is restaurant_kv_serving.model_profile_definition_from_record
    assert model_profile_definition_from_record.__module__ == "document_kv_cache.model_profiles"
    assert model_profile_definition_to_record is model_profiles.model_profile_definition_to_record
    assert model_profile_definition_to_record is restaurant_kv_serving.model_profile_definition_to_record
    assert model_profile_definition_to_record.__module__ == "document_kv_cache.model_profiles"
    assert read_model_profile_definition_json is model_profiles.read_model_profile_definition_json
    assert read_model_profile_definition_json is restaurant_kv_serving.read_model_profile_definition_json
    assert read_databricks_engine_probe_targets_json is restaurant_kv_serving.read_databricks_engine_probe_targets_json
    assert (
        read_databricks_engine_probe_targets_file_json
        is restaurant_kv_serving.read_databricks_engine_probe_targets_file_json
    )
    assert write_model_profile_definition_json is model_profiles.write_model_profile_definition_json
    assert write_model_profile_definition_json is restaurant_kv_serving.write_model_profile_definition_json
    assert write_model_profile_definition_json.__module__ == "document_kv_cache.model_profiles"
    assert write_engine_probe_targets_json is restaurant_kv_serving.write_engine_probe_targets_json
    assert inspect_builtin_native_probe_factories is native_probe_factories.inspect_builtin_native_probe_factories
    assert inspect_builtin_native_probe_factories is restaurant_kv_serving.inspect_builtin_native_probe_factories
    assert inspect_builtin_native_probe_factory is native_probe_factories.inspect_builtin_native_probe_factory
    assert inspect_builtin_native_probe_factory is restaurant_kv_serving.inspect_builtin_native_probe_factory
    assert native_probe_factory_inspection_to_record is native_probe_factories.native_probe_factory_inspection_to_record
    assert native_probe_factory_inspection_to_record is restaurant_kv_serving.native_probe_factory_inspection_to_record
    assert run_engine_kv_connector_probe is engine_probe.run_engine_kv_connector_probe
    assert sglang_native_probe_factory is native_probe_factories.sglang_native_probe_factory
    assert sglang_native_probe_factory is restaurant_kv_serving.sglang_native_probe_factory
    assert vllm_native_probe_factory is native_probe_factories.vllm_native_probe_factory
    assert vllm_native_probe_factory is restaurant_kv_serving.vllm_native_probe_factory
    assert serving_environment_profile is restaurant_kv_serving.serving_environment_profile
    assert serving_environment_profile is serving_env.serving_environment_profile
    assert serving_environment_profile.__module__ == "document_kv_cache.serving_env"
    assert serving_environment_profile_to_record is restaurant_kv_serving.serving_environment_profile_to_record
    assert serving_environment_profile_to_record is serving_env.serving_environment_profile_to_record
    assert serving_environment_profile_to_record.__module__ == "document_kv_cache.serving_env"
    assert serving_environment_profiles is restaurant_kv_serving.serving_environment_profiles
    assert serving_environment_profiles is serving_env.serving_environment_profiles
    assert serving_environment_profiles.__module__ == "document_kv_cache.serving_env"
    assert serving_environment_profiles_to_record is restaurant_kv_serving.serving_environment_profiles_to_record
    assert serving_environment_profiles_to_record is serving_env.serving_environment_profiles_to_record
    assert serving_environment_profiles_to_record.__module__ == "document_kv_cache.serving_env"
    assert summarize_databricks_run is databricks_runs.summarize_databricks_run
    assert summarize_databricks_run_submit_payload is databricks_runs.summarize_databricks_run_submit_payload
    assert pr_evidence_validation_to_record is restaurant_kv_serving.pr_evidence_validation_to_record
    assert pr_evidence_validation_to_record.__module__ == "document_kv_cache.pr_evidence"
    assert pr_evidence_to_record is restaurant_kv_serving.pr_evidence_to_record
    assert pr_evidence_to_record.__module__ == "document_kv_cache.pr_evidence"
    assert "DocumentKVRequest" in dir(document_kv_cache)


def test_public_document_package_star_exports_are_document_first_with_legacy_getattr_aliases():
    assert "AdmissionQueue" in document_kv_cache.__all__
    assert "BENCHMARK_RUN_RECORD_TYPE" in document_kv_cache.__all__
    assert "PreparedRequest" in document_kv_cache.__all__
    assert "RELEASE_EVIDENCE_RECORD_TYPE" in document_kv_cache.__all__
    assert "RELEASE_EVIDENCE_INPUT_STATUS_RECORD_TYPE" in document_kv_cache.__all__
    assert "RELEASE_EVIDENCE_ARTIFACT_ROLES" in document_kv_cache.__all__
    assert "SERVING_ENVIRONMENT_PROFILES_RECORD_TYPE" in document_kv_cache.__all__
    assert "PR_EVIDENCE_RECORD_TYPE" in document_kv_cache.__all__
    assert "PR_EVIDENCE_VALIDATION_RECORD_TYPE" in document_kv_cache.__all__
    assert "MODEL_PROFILE_RECORD_TYPE" in document_kv_cache.__all__
    assert "RELEASE_BUNDLE_RECORD_TYPE" in document_kv_cache.__all__
    assert "RELEASE_BUNDLE_MANIFEST_FILENAME" in document_kv_cache.__all__
    assert "RELEASE_BUNDLE_ARTIFACT_ROLES" in document_kv_cache.__all__
    assert "RELEASE_STORAGE_BENCHMARK_READERS" in document_kv_cache.__all__
    assert "REQUIRED_ENGINE_PROBE_BACKENDS" in document_kv_cache.__all__
    assert "DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME" in document_kv_cache.__all__
    assert "DEFAULT_DATABRICKS_STORAGE_BENCHMARK_RUN_NAME" in document_kv_cache.__all__
    assert "DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME" in document_kv_cache.__all__
    assert "ENGINE_PROBE_TARGETS_RECORD_TYPE" in document_kv_cache.__all__
    assert "ENGINE_PROBE_TARGETS_SCHEMA_VERSION" in document_kv_cache.__all__
    assert "STORAGE_BENCHMARK_RECORD_TYPE" in document_kv_cache.__all__
    assert "DatabricksEngineProbeJobConfig" in document_kv_cache.__all__
    assert "DatabricksEngineProbeTargetsFile" in document_kv_cache.__all__
    assert "DatabricksStorageBenchmarkJobConfig" in document_kv_cache.__all__
    assert "DatabricksVLLMSmokeJobConfig" in document_kv_cache.__all__
    assert "ReleaseEvidence" in document_kv_cache.__all__
    assert "ReleaseBundlePlanConfig" in document_kv_cache.__all__
    assert "ReleaseEvidenceInputFileStatus" in document_kv_cache.__all__
    assert "ReleaseEvidenceInputStatus" in document_kv_cache.__all__
    assert "ReleaseEvidencePlanConfig" in document_kv_cache.__all__
    assert "ReleaseEvidenceArtifactSource" in document_kv_cache.__all__
    assert "PullRequestEvidence" in document_kv_cache.__all__
    assert "ModelProfileDefinition" in document_kv_cache.__all__
    assert "MaterializedKV" in document_kv_cache.__all__
    assert "SegmentedMaterializedKV" in document_kv_cache.__all__
    assert "KVMaterializer" in document_kv_cache.__all__
    assert "ReleaseBundle" in document_kv_cache.__all__
    assert "ReleaseBundleArtifact" in document_kv_cache.__all__
    assert "ServingEnvironmentProfile" in document_kv_cache.__all__
    assert "StorageBenchmarkConfig" in document_kv_cache.__all__
    assert "StorageBenchmarkEvidence" in document_kv_cache.__all__
    assert "StorageBenchmarkPlanConfig" in document_kv_cache.__all__
    assert "ByteLRU" in document_kv_cache.__all__
    assert "ChunkCache" in document_kv_cache.__all__
    assert "ChunkCacheResult" in document_kv_cache.__all__
    assert "SUPPORTED_STORAGE_BENCHMARK_READERS" in document_kv_cache.__all__
    assert "DEDICATED_DATABRICKS_DATA_SECURITY_MODE" in document_kv_cache.__all__
    assert "ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND" in document_kv_cache.__all__
    assert "ENGINE_KV_PROBE_METADATA_HANDOFF_JSON" in document_kv_cache.__all__
    assert "ENGINE_KV_PROBE_METADATA_PAYLOAD_URI" in document_kv_cache.__all__
    assert "ENGINE_KV_PROBE_METADATA_PROBE_FACTORY" in document_kv_cache.__all__
    assert "ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE" in document_kv_cache.__all__
    assert "ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION" in document_kv_cache.__all__
    assert "GPT55_REVIEW_OUTCOMES" in document_kv_cache.__all__
    assert "evaluate_pr_evidence_directory" in document_kv_cache.__all__
    assert "evaluate_pr_evidence_file" in document_kv_cache.__all__
    assert "pr_evidence_validation_to_record" in document_kv_cache.__all__
    assert "SINGLE_USER_DATABRICKS_DATA_SECURITY_MODES" in document_kv_cache.__all__
    assert "DATABRICKS_RUN_STATUS_RECORD_TYPE" in document_kv_cache.__all__
    assert "DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE" in document_kv_cache.__all__
    assert "summarize_databricks_run" in document_kv_cache.__all__
    assert "engine_probe_targets_to_record" in document_kv_cache.__all__
    assert "write_engine_probe_targets_json" in document_kv_cache.__all__
    assert "read_databricks_engine_probe_targets_file_json" in document_kv_cache.__all__
    assert "summarize_databricks_run_submit_payload" in document_kv_cache.__all__
    assert "DocumentKVRequest" in document_kv_cache.__all__
    assert "ManifestStore" in document_kv_cache.__all__
    assert "InMemoryManifestStore" in document_kv_cache.__all__
    assert "CacheRequest" in document_kv_cache.__all__
    assert "CachePlanner" in document_kv_cache.__all__
    assert "DocumentKVService" in document_kv_cache.__all__
    assert "EngineProbePlanConfig" in document_kv_cache.__all__
    assert "DocumentChunkType" in document_kv_cache.__all__
    assert "DocumentChunkRole" in document_kv_cache.__all__
    assert "DOCUMENT_CHUNK_TYPES" in document_kv_cache.__all__
    assert "LEGACY_RESTAURANT_CHUNK_TYPES" in document_kv_cache.__all__
    assert "chunk_type_role" in document_kv_cache.__all__
    assert "chunk_type_sort_order" in document_kv_cache.__all__
    assert "chunk_types_for_request" in document_kv_cache.__all__
    assert "EngineKVProbeFactory" in document_kv_cache.__all__
    assert "serving_environment_profile" in document_kv_cache.__all__
    assert "serving_environment_profiles_to_record" in document_kv_cache.__all__
    assert "KVStorageLayout" in document_kv_cache.__all__
    assert "kv_storage_layout_from_value" in document_kv_cache.__all__
    assert "RestaurantKVRequest" not in document_kv_cache.__all__
    assert "RestaurantKVService" not in document_kv_cache.__all__
    assert "ChunkType" not in document_kv_cache.__all__

    assert document_kv_cache.RestaurantKVRequest is restaurant_kv_serving.RestaurantKVRequest
    assert document_kv_cache.RestaurantKVService is restaurant_kv_serving.RestaurantKVService
    assert document_kv_cache.ChunkType is restaurant_kv_serving.ChunkType
    star_namespace: dict[str, object] = {}
    exec("from document_kv_cache import *", star_namespace)
    assert star_namespace["AdmissionQueue"] is restaurant_kv_serving.AdmissionQueue
    assert star_namespace["AdmissionQueue"].__module__ == "document_kv_cache.admission"
    assert star_namespace["PreparedRequest"] is restaurant_kv_serving.PreparedRequest
    assert star_namespace["PreparedRequest"].__module__ == "document_kv_cache.admission"
    assert star_namespace["StorageBenchmarkConfig"] is restaurant_kv_serving.StorageBenchmarkConfig
    assert star_namespace["StorageBenchmarkEvidence"] is restaurant_kv_serving.StorageBenchmarkEvidence
    assert star_namespace["StorageBenchmarkPlanConfig"] is restaurant_kv_serving.StorageBenchmarkPlanConfig
    assert (
        star_namespace["SINGLE_USER_DATABRICKS_DATA_SECURITY_MODES"]
        is restaurant_kv_serving.SINGLE_USER_DATABRICKS_DATA_SECURITY_MODES
    )


def test_public_cli_submodules_are_importable_under_document_namespace():
    public_submodules = (
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
        "databricks_vllm_smoke_job",
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
        "workflow",
    )

    modules = {
        name: importlib.import_module(f"document_kv_cache.{name}")
        for name in public_submodules
    }

    assert {name: module.__name__ for name, module in modules.items()} == {
        name: f"document_kv_cache.{name}"
        for name in public_submodules
    }
    benchmark_plan = modules["benchmark_plan"]
    benchmark_runner = modules["benchmark_runner"]
    benchmarks = modules["benchmarks"]
    admission = modules["admission"]
    dataset_prep = modules["dataset_prep"]
    engine_adapters = modules["engine_adapters"]
    engine_probe = modules["engine_probe"]
    model_profiles = modules["model_profiles"]
    native_probe_factories = modules["native_probe_factories"]
    storage = modules["storage"]
    storage_benchmark = modules["storage_benchmark"]
    template_resources = modules["template_resources"]
    vllm_smoke = modules["vllm_smoke"]
    databricks_runs = modules["databricks_runs"]
    databricks_job = modules["databricks_job"]
    databricks_engine_probe_job = modules["databricks_engine_probe_job"]
    databricks_storage_benchmark_job = modules["databricks_storage_benchmark_job"]
    databricks_vllm_smoke_job = modules["databricks_vllm_smoke_job"]
    live_server = modules["live_server"]
    release_bundle = modules["release_bundle"]
    release_evidence = modules["release_evidence"]
    pr_evidence = modules["pr_evidence"]
    serving_env = modules["serving_env"]
    workflow = modules["workflow"]
    legacy_benchmarks = importlib.import_module("restaurant_kv_serving.benchmarks")
    legacy_benchmark_runner = importlib.import_module("restaurant_kv_serving.benchmark_runner")
    legacy_dataset_prep = importlib.import_module("restaurant_kv_serving.dataset_prep")
    legacy_live_server = importlib.import_module("restaurant_kv_serving.live_server")
    legacy_storage_benchmark = importlib.import_module("restaurant_kv_serving.storage_benchmark")
    legacy_vllm_smoke = importlib.import_module("restaurant_kv_serving.vllm_smoke")

    assert admission.AdmissionQueue is restaurant_kv_serving.AdmissionQueue
    assert admission.AdmissionQueue.__module__ == "document_kv_cache.admission"
    assert benchmark_plan.BenchmarkPlanConfig is restaurant_kv_serving.BenchmarkPlanConfig
    assert benchmark_plan.ENGINE_PROBE_TARGETS_RECORD_TYPE is restaurant_kv_serving.ENGINE_PROBE_TARGETS_RECORD_TYPE
    assert benchmark_plan.EngineProbePlanConfig is restaurant_kv_serving.EngineProbePlanConfig
    assert benchmark_plan.engine_probe_targets_to_record is restaurant_kv_serving.engine_probe_targets_to_record
    assert benchmark_plan.ReleaseBundlePlanConfig is restaurant_kv_serving.ReleaseBundlePlanConfig
    assert benchmark_plan.ReleaseEvidencePlanConfig is restaurant_kv_serving.ReleaseEvidencePlanConfig
    assert benchmark_plan.StorageBenchmarkPlanConfig is restaurant_kv_serving.StorageBenchmarkPlanConfig
    assert benchmarks.BenchmarkExample.__module__ == "document_kv_cache.benchmarks"
    assert set(benchmarks.__all__) < set(legacy_benchmarks.__all__)
    assert set(legacy_benchmarks.__all__) - set(benchmarks.__all__) == {"SourceDocument"}
    assert "SourceDocument" not in benchmarks.__all__
    assert "_format_document" not in benchmarks.__all__
    assert legacy_benchmarks.BenchmarkExample is benchmarks.BenchmarkExample
    assert legacy_benchmarks.SourceDocument is document_kv_cache.SourceDocument
    assert restaurant_kv_serving.BenchmarkExample is benchmarks.BenchmarkExample
    assert benchmark_runner.BenchmarkGeneration.__module__ == "document_kv_cache.benchmark_runner"
    assert set(benchmark_runner.__all__) < set(legacy_benchmark_runner.__all__)
    assert set(legacy_benchmark_runner.__all__) - set(benchmark_runner.__all__) == {
        "Any",
        "BASELINE_PREFILL_ARM",
        "BenchmarkArm",
        "BenchmarkComparison",
        "BenchmarkExample",
        "BenchmarkPromptParts",
        "BenchmarkReportRow",
        "BenchmarkSuite",
        "CACHE_REUSE_ARM",
        "Callable",
        "DEFAULT_HARDWARE_TARGET",
        "DEFAULT_V1_MODEL_ID",
        "DocumentChunkType",
        "InferenceMeasurement",
        "Iterable",
        "Literal",
        "Mapping",
        "Path",
        "Protocol",
        "Sequence",
        "SourceChunk",
        "SourceDocument",
        "argparse",
        "baseline_prefill_arm",
        "build_prompt_parts",
        "compare_to_baseline",
        "dataclass",
        "document_kv_cache_arm",
        "evaluate_v1_benchmark_evidence",
        "field",
        "json",
        "local_path",
        "random",
        "summarize_measurements",
        "validate_v1_dataset",
    }
    assert "local_path" not in benchmark_runner.__all__
    assert legacy_benchmark_runner.BenchmarkGeneration is benchmark_runner.BenchmarkGeneration
    assert legacy_benchmark_runner._openai_compatible_engine is benchmark_runner._openai_compatible_engine
    assert legacy_benchmark_runner.SourceDocument is document_kv_cache.SourceDocument
    assert benchmark_runner.BENCHMARK_RUN_RECORD_TYPE is restaurant_kv_serving.BENCHMARK_RUN_RECORD_TYPE
    assert dataset_prep.convert_v1_jsonl.__module__ == "document_kv_cache.dataset_prep"
    assert set(dataset_prep.__all__) < set(legacy_dataset_prep.__all__)
    assert set(legacy_dataset_prep.__all__) - set(dataset_prep.__all__) == {
        "Any",
        "Iterable",
        "Mapping",
        "Path",
        "Sequence",
        "argparse",
        "json",
        "local_path",
        "validate_v1_dataset",
    }
    assert "local_path" not in dataset_prep.__all__
    assert legacy_dataset_prep.convert_v1_jsonl is dataset_prep.convert_v1_jsonl
    assert legacy_dataset_prep.local_path is dataset_prep.local_path
    assert legacy_dataset_prep.validate_v1_dataset is dataset_prep.validate_v1_dataset
    assert dataset_prep.convert_v1_jsonl is restaurant_kv_serving.convert_v1_jsonl
    assert live_server.LiveServerCheckConfig.__module__ == "document_kv_cache.live_server"
    assert set(live_server.__all__) < set(legacy_live_server.__all__)
    assert legacy_live_server.LiveServerCheckConfig is live_server.LiveServerCheckConfig
    assert legacy_live_server.run_openai_compatible_live_check is live_server.run_openai_compatible_live_check
    assert "argparse" not in live_server.__all__
    assert engine_adapters.vllm_adapter_spec is restaurant_kv_serving.vllm_adapter_spec
    assert engine_probe.ENGINE_KV_PROBE_METADATA_HANDOFF_JSON == restaurant_kv_serving.ENGINE_KV_PROBE_METADATA_HANDOFF_JSON
    assert issubclass(restaurant_kv_serving.EngineKVProbeConfig, engine_probe.EngineKVProbeConfig)
    assert engine_probe.EngineKVProbeConfig.__module__ == "document_kv_cache.engine_probe"
    assert restaurant_kv_serving.EngineKVProbeConfig.__module__ == "restaurant_kv_serving.engine_probe"
    assert model_profiles.ModelProfileRegistry is restaurant_kv_serving.ModelProfileRegistry
    assert model_profiles.ModelProfileRegistry.__module__ == "document_kv_cache.model_profiles"
    assert model_profiles.ModelProfileDefinition is restaurant_kv_serving.ModelProfileDefinition
    assert model_profiles.ModelProfileDefinition.__module__ == "document_kv_cache.model_profiles"
    assert model_profiles.KVModelProfile is restaurant_kv_serving.KVModelProfile
    assert model_profiles.KVModelProfile.__module__ == "document_kv_cache.model_profiles"
    assert model_profiles.MODEL_PROFILE_RECORD_TYPE is restaurant_kv_serving.MODEL_PROFILE_RECORD_TYPE
    assert model_profiles.layout_for_model is restaurant_kv_serving.layout_for_model
    assert model_profiles.layout_for_model.__module__ == "document_kv_cache.model_profiles"
    assert (
        model_profiles.model_profile_definition_to_record
        is restaurant_kv_serving.model_profile_definition_to_record
    )
    assert document_kv_cache.DiskRangeReader is storage.DiskRangeReader
    assert storage.DiskRangeReader.__module__ == "document_kv_cache.storage"
    assert restaurant_kv_serving.DiskRangeReader.__module__ == "restaurant_kv_serving.storage"
    assert issubclass(restaurant_kv_serving.DiskRangeReader, storage.DiskRangeReader)
    assert release_bundle.ReleaseBundle is restaurant_kv_serving.ReleaseBundle
    assert release_bundle.ReleaseBundle.__module__ == "document_kv_cache.release_bundle"
    assert release_bundle.build_release_bundle is restaurant_kv_serving.build_release_bundle
    assert release_bundle.build_release_bundle.__module__ == "document_kv_cache.release_bundle"
    assert release_evidence.RELEASE_EVIDENCE_ARTIFACT_ROLES is restaurant_kv_serving.RELEASE_EVIDENCE_ARTIFACT_ROLES
    assert release_evidence.ReleaseEvidenceArtifactSource is restaurant_kv_serving.ReleaseEvidenceArtifactSource
    assert release_evidence.ReleaseEvidenceArtifactSource.__module__ == "document_kv_cache.release_evidence"
    assert release_evidence.evaluate_release_evidence is restaurant_kv_serving.evaluate_release_evidence
    assert release_evidence.evaluate_release_evidence.__module__ == "document_kv_cache.release_evidence"
    assert release_evidence.inspect_release_evidence_input_files is restaurant_kv_serving.inspect_release_evidence_input_files
    assert (
        release_evidence.inspect_release_evidence_input_files.__module__
        == "document_kv_cache.release_evidence"
    )
    assert pr_evidence.PR_EVIDENCE_RECORD_TYPE is restaurant_kv_serving.PR_EVIDENCE_RECORD_TYPE
    assert pr_evidence.PR_EVIDENCE_VALIDATION_RECORD_TYPE is restaurant_kv_serving.PR_EVIDENCE_VALIDATION_RECORD_TYPE
    assert pr_evidence.PullRequestEvidence is restaurant_kv_serving.PullRequestEvidence
    assert pr_evidence.PullRequestEvidence.__module__ == "document_kv_cache.pr_evidence"
    assert pr_evidence.evaluate_pr_evidence is restaurant_kv_serving.evaluate_pr_evidence
    assert pr_evidence.evaluate_pr_evidence.__module__ == "document_kv_cache.pr_evidence"
    assert pr_evidence.evaluate_pr_evidence_directory is restaurant_kv_serving.evaluate_pr_evidence_directory
    assert pr_evidence.evaluate_pr_evidence_directory.__module__ == "document_kv_cache.pr_evidence"
    assert pr_evidence.evaluate_pr_evidence_file is restaurant_kv_serving.evaluate_pr_evidence_file
    assert pr_evidence.evaluate_pr_evidence_file.__module__ == "document_kv_cache.pr_evidence"
    assert pr_evidence.pr_evidence_validation_to_record is restaurant_kv_serving.pr_evidence_validation_to_record
    assert pr_evidence.pr_evidence_validation_to_record.__module__ == "document_kv_cache.pr_evidence"
    assert native_probe_factories.NativeProbeFactoryInspection is restaurant_kv_serving.NativeProbeFactoryInspection
    assert (
        native_probe_factories.NativeProbeFactoryInspection.__module__
        == "document_kv_cache.native_probe_factories"
    )
    assert native_probe_factories.builtin_native_probe_factory_path is restaurant_kv_serving.builtin_native_probe_factory_path
    assert native_probe_factories.vllm_native_probe_factory is restaurant_kv_serving.vllm_native_probe_factory
    assert serving_env.ServingEnvironmentProfile is restaurant_kv_serving.ServingEnvironmentProfile
    assert serving_env.ServingEnvironmentProfile.__module__ == "document_kv_cache.serving_env"
    assert serving_env.SERVING_ENVIRONMENT_PROFILES_RECORD_TYPE == "document_kv.serving_environment_profiles.v1"
    assert serving_env.serving_environment_profiles_to_record is restaurant_kv_serving.serving_environment_profiles_to_record
    assert serving_env.serving_environment_profiles_to_record.__module__ == "document_kv_cache.serving_env"
    assert workflow.SourceDocument is restaurant_kv_serving.SourceDocument
    assert workflow.SourceDocument.__module__ == "document_kv_cache.workflow"
    assert workflow.DocumentKVWorkflow is restaurant_kv_serving.DocumentKVWorkflow
    assert workflow.DocumentKVWorkflow.__module__ == "document_kv_cache.workflow"
    assert storage_benchmark.StorageBenchmarkConfig is restaurant_kv_serving.StorageBenchmarkConfig
    assert storage_benchmark.StorageBenchmarkConfig.__module__ == "document_kv_cache.storage_benchmark"
    assert storage_benchmark.run_storage_benchmark.__module__ == "document_kv_cache.storage_benchmark"
    assert restaurant_kv_serving.run_storage_benchmark is legacy_storage_benchmark.run_storage_benchmark
    assert restaurant_kv_serving.run_storage_benchmark.__module__ == "restaurant_kv_serving.storage_benchmark"
    assert storage_benchmark.evaluate_storage_benchmark_evidence.__module__ == "document_kv_cache.storage_benchmark"
    assert (
        restaurant_kv_serving.evaluate_storage_benchmark_evidence
        is legacy_storage_benchmark.evaluate_storage_benchmark_evidence
    )
    assert restaurant_kv_serving.evaluate_storage_benchmark_evidence.__module__ == "restaurant_kv_serving.storage_benchmark"
    assert template_resources.PACKAGED_TEMPLATE_PACKAGE == "document_kv_cache.templates"
    assert vllm_smoke.SERVED_MODEL_NAME == "qwen3:4b-instruct"
    assert vllm_smoke.VLLMSmokeBenchmarkConfig.__module__ == "document_kv_cache.vllm_smoke"
    assert set(vllm_smoke.__all__) < set(legacy_vllm_smoke.__all__)
    assert legacy_vllm_smoke.VLLMSmokeBenchmarkConfig is vllm_smoke.VLLMSmokeBenchmarkConfig
    assert legacy_vllm_smoke.run_vllm_smoke_benchmark is not vllm_smoke.run_vllm_smoke_benchmark
    assert "run" not in vllm_smoke.__all__
    assert issubclass(restaurant_kv_serving.DatabricksWorkspaceConfig, databricks_runs.DatabricksWorkspaceConfig)
    assert databricks_runs.DatabricksWorkspaceConfig.__module__ == "document_kv_cache.databricks_runs"
    assert restaurant_kv_serving.DatabricksWorkspaceConfig.__module__ == "restaurant_kv_serving.databricks_runs"
    assert issubclass(restaurant_kv_serving.DatabricksBenchmarkJobConfig, databricks_job.DatabricksBenchmarkJobConfig)
    assert issubclass(
        restaurant_kv_serving.DatabricksSingleNodeG5ClusterConfig,
        databricks_job.DatabricksSingleNodeG5ClusterConfig,
    )
    assert databricks_job.DatabricksBenchmarkJobConfig.__module__ == "document_kv_cache.databricks_job"
    assert restaurant_kv_serving.DatabricksBenchmarkJobConfig.__module__ == "restaurant_kv_serving.databricks_job"
    assert (
        issubclass(
            restaurant_kv_serving.DatabricksEngineProbeJobConfig,
            databricks_engine_probe_job.DatabricksEngineProbeJobConfig,
        )
    )
    assert (
        issubclass(
            restaurant_kv_serving.DatabricksEngineProbeMatrixJobConfig,
            databricks_engine_probe_job.DatabricksEngineProbeMatrixJobConfig,
        )
    )
    assert (
        databricks_engine_probe_job.DatabricksEngineProbeJobConfig.__module__
        == "document_kv_cache.databricks_engine_probe_job"
    )
    assert (
        restaurant_kv_serving.DatabricksEngineProbeJobConfig.__module__
        == "restaurant_kv_serving.databricks_engine_probe_job"
    )
    assert (
        issubclass(
            restaurant_kv_serving.DatabricksStorageBenchmarkJobConfig,
            databricks_storage_benchmark_job.DatabricksStorageBenchmarkJobConfig,
        )
    )
    assert (
        databricks_storage_benchmark_job.DatabricksStorageBenchmarkJobConfig.__module__
        == "document_kv_cache.databricks_storage_benchmark_job"
    )
    assert (
        restaurant_kv_serving.DatabricksStorageBenchmarkJobConfig.__module__
        == "restaurant_kv_serving.databricks_storage_benchmark_job"
    )
    assert (
        databricks_vllm_smoke_job.DatabricksVLLMSmokeJobConfig
        is restaurant_kv_serving.DatabricksVLLMSmokeJobConfig
    )
    assert (
        databricks_vllm_smoke_job.DatabricksVLLMSmokeJobConfig.__module__
        == "document_kv_cache.databricks_vllm_smoke_job"
    )
    assert storage_benchmark.evaluate_release_storage_benchmark_evidence.__module__ == "document_kv_cache.storage_benchmark"
    assert (
        restaurant_kv_serving.evaluate_release_storage_benchmark_evidence
        is legacy_storage_benchmark.evaluate_release_storage_benchmark_evidence
    )
    assert (
        restaurant_kv_serving.evaluate_release_storage_benchmark_evidence.__module__
        == "restaurant_kv_serving.storage_benchmark"
    )
    assert storage_benchmark.RELEASE_STORAGE_BENCHMARK_READERS is restaurant_kv_serving.RELEASE_STORAGE_BENCHMARK_READERS
    assert storage.is_real_uc_volume_root("/Volumes/catalog/schema/volume") is True
    assert "scheduler" not in document_kv_cache._PUBLIC_SUBMODULES


def test_public_document_submodules_have_curated_star_import_surfaces():
    admission = importlib.import_module("document_kv_cache.admission")
    cache = importlib.import_module("document_kv_cache.cache")
    engine = importlib.import_module("document_kv_cache.engine")
    engine_protocol = importlib.import_module("document_kv_cache.engine_protocol")
    engine_adapters = importlib.import_module("document_kv_cache.engine_adapters")
    kvpack = importlib.import_module("document_kv_cache.kvpack")
    manifest = importlib.import_module("document_kv_cache.manifest")
    models = importlib.import_module("document_kv_cache.models")
    planner = importlib.import_module("document_kv_cache.planner")
    openai_compatible = importlib.import_module("document_kv_cache.openai_compatible")
    pr_evidence = importlib.import_module("document_kv_cache.pr_evidence")
    release_bundle = importlib.import_module("document_kv_cache.release_bundle")
    release_evidence = importlib.import_module("document_kv_cache.release_evidence")
    serving_env = importlib.import_module("document_kv_cache.serving_env")
    storage = importlib.import_module("document_kv_cache.storage")
    materializer = importlib.import_module("document_kv_cache.materializer")
    model_profiles = importlib.import_module("document_kv_cache.model_profiles")
    native_probe_factories = importlib.import_module("document_kv_cache.native_probe_factories")
    service = importlib.import_module("document_kv_cache.service")
    storage_benchmark = importlib.import_module("document_kv_cache.storage_benchmark")
    workflow = importlib.import_module("document_kv_cache.workflow")

    assert admission.__all__ == ["PreparedRequest", "AdmissionQueue"]
    assert admission.AdmissionQueue.__module__ == "document_kv_cache.admission"
    assert admission.PreparedRequest.__module__ == "document_kv_cache.admission"
    assert restaurant_kv_serving.AdmissionQueue is admission.AdmissionQueue
    assert restaurant_kv_serving.PreparedRequest is admission.PreparedRequest
    assert engine_protocol.__all__ == [
        "DTYPE_BYTE_WIDTHS",
        "AttentionMechanism",
        "KVStorageLayout",
        "dtype_byte_width",
        "kv_storage_layout_from_value",
        "KVLayout",
        "KVSegment",
        "KVCacheHandle",
    ]
    legacy_engine_protocol = importlib.import_module("restaurant_kv_serving.engine_protocol")
    assert engine_protocol.AttentionMechanism.__module__ == "document_kv_cache.engine_protocol"
    assert engine_protocol.KVStorageLayout.__module__ == "document_kv_cache.engine_protocol"
    assert engine_protocol.KVLayout.__module__ == "document_kv_cache.engine_protocol"
    assert engine_protocol.KVSegment.__module__ == "document_kv_cache.engine_protocol"
    assert engine_protocol.KVCacheHandle.__module__ == "document_kv_cache.engine_protocol"
    assert legacy_engine_protocol.KVLayout is engine_protocol.KVLayout
    assert legacy_engine_protocol.KVStorageLayout is engine_protocol.KVStorageLayout
    assert legacy_engine_protocol.kv_storage_layout_from_value is engine_protocol.kv_storage_layout_from_value
    assert engine.__all__ == [
        "EngineReadyRequest",
        "ServingEngineConnector",
        "build_handle_from_materialized",
        "build_engine_ready_request",
    ]
    legacy_engine = importlib.import_module("restaurant_kv_serving.engine")
    assert legacy_engine.EngineReadyRequest is engine.EngineReadyRequest
    assert legacy_engine.build_engine_ready_request is engine.build_engine_ready_request
    assert legacy_engine.build_handle_from_materialized is engine.build_handle_from_materialized
    assert engine_adapters.__all__ == [
        "EngineAdapterRequest",
        "EngineAdapterSpec",
        "ENGINE_KV_CONNECTOR_PROBE_RECORD_TYPE",
        "ENGINE_KV_CONNECTOR_PROBE_SCHEMA_VERSION",
        "EngineKVBlockManagerProbe",
        "EngineKVBindAction",
        "EngineKVConnectorActions",
        "EngineKVConnectorProbeResult",
        "EngineKVInjectionPlan",
        "EngineKVReleaseAction",
        "EngineKVReservationAction",
        "EngineKVSegmentCopyAction",
        "EngineKVSegmentBinding",
        "PayloadMode",
        "ServingBackend",
        "build_engine_adapter_request",
        "build_engine_kv_connector_actions",
        "build_engine_kv_injection_plan",
        "engine_kv_connector_probe_result_to_record",
        "engine_adapter_request_to_record",
        "payload_mode_for",
        "probe_engine_kv_connector_actions",
        "read_engine_adapter_request_json",
        "sglang_adapter_spec",
        "split_engine_adapter_payload",
        "validate_engine_adapter_request_record",
        "validate_engine_kv_connector_probe_record",
        "validate_engine_kv_connector_actions",
        "view_engine_adapter_payload",
        "vllm_adapter_spec",
        "write_engine_adapter_request_json",
    ]
    engine_probe = importlib.import_module("document_kv_cache.engine_probe")
    assert engine_probe.__all__ == [
        "EngineKVProbeConfig",
        "ENGINE_KV_PROBE_METADATA_EXPECTED_BACKEND",
        "ENGINE_KV_PROBE_METADATA_HANDOFF_JSON",
        "ENGINE_KV_PROBE_METADATA_PAYLOAD_URI",
        "ENGINE_KV_PROBE_METADATA_PROBE_FACTORY",
        "ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_PACKAGE",
        "ENGINE_KV_PROBE_METADATA_SERVING_ENGINE_VERSION",
        "EngineKVProbeFactory",
        "EngineKVProbeFactoryContext",
        "EngineKVProbeFactoryResult",
        "run_engine_kv_connector_probe",
        "read_engine_adapter_payload",
        "write_engine_kv_connector_probe_result_json",
        "load_engine_kv_probe_factory",
        "parse_args",
        "main",
    ]
    assert kvpack.__all__ == ["PackChunk", "LocalRangeReader", "write_kvpack"]
    legacy_kvpack = importlib.import_module("restaurant_kv_serving.kvpack")
    assert kvpack.PackChunk.__module__ == "document_kv_cache.kvpack"
    assert kvpack.write_kvpack.__module__ == "document_kv_cache.kvpack"
    assert legacy_kvpack.PackChunk is kvpack.PackChunk
    assert legacy_kvpack.LocalRangeReader is legacy_kvpack.DiskRangeReader
    assert legacy_kvpack.write_kvpack.__module__ == "restaurant_kv_serving.kvpack"
    assert storage_benchmark.__all__ == [
        "STORAGE_BENCHMARK_RECORD_TYPE",
        "SUPPORTED_STORAGE_BENCHMARK_READERS",
        "RELEASE_STORAGE_BENCHMARK_READERS",
        "StorageBenchmarkConfig",
        "StorageBenchmarkEvidence",
        "StorageBenchmarkResult",
        "StorageReaderBenchmarkResult",
        "evaluate_storage_benchmark_evidence",
        "evaluate_release_storage_benchmark_evidence",
        "run_storage_benchmark",
        "storage_benchmark_evidence_to_record",
        "storage_benchmark_result_to_record",
        "write_storage_benchmark_result_json",
        "main",
    ]
    legacy_storage_benchmark = importlib.import_module("restaurant_kv_serving.storage_benchmark")
    assert not hasattr(legacy_storage_benchmark, "__all__")
    assert storage_benchmark.run_storage_benchmark.__module__ == "document_kv_cache.storage_benchmark"
    assert legacy_storage_benchmark.run_storage_benchmark.__module__ == "restaurant_kv_serving.storage_benchmark"
    assert release_bundle.__all__ == [
        "RELEASE_BUNDLE_RECORD_TYPE",
        "RELEASE_BUNDLE_MANIFEST_FILENAME",
        "RELEASE_BUNDLE_ARTIFACT_ROLES",
        "ReleaseBundleArtifact",
        "ReleaseBundle",
        "build_release_bundle",
        "release_bundle_to_record",
        "write_release_bundle_manifest_json",
        "main",
    ]
    legacy_release_bundle = importlib.import_module("restaurant_kv_serving.release_bundle")
    release_bundle_legacy_star_namespace: dict[str, object] = {}
    exec("from restaurant_kv_serving.release_bundle import *", release_bundle_legacy_star_namespace)
    assert not hasattr(legacy_release_bundle, "__all__")
    assert "RELEASE_BUNDLE_PACKAGE_NAME" not in release_bundle.__all__
    assert release_bundle_legacy_star_namespace["RELEASE_BUNDLE_PACKAGE_NAME"] == "document-kv-cache"
    assert release_bundle.ReleaseBundle.__module__ == "document_kv_cache.release_bundle"
    assert legacy_release_bundle.ReleaseBundle is release_bundle.ReleaseBundle
    assert release_bundle.main.__module__ == "document_kv_cache.release_bundle"
    assert legacy_release_bundle.main.__module__ == "restaurant_kv_serving.release_bundle"
    assert release_evidence.__all__ == [
        "RELEASE_EVIDENCE_RECORD_TYPE",
        "RELEASE_EVIDENCE_INPUT_STATUS_RECORD_TYPE",
        "RELEASE_EVIDENCE_ARTIFACT_ROLES",
        "REQUIRED_ENGINE_PROBE_BACKENDS",
        "ReleaseEvidenceArtifactSource",
        "ReleaseEvidence",
        "ReleaseEvidenceInputFileStatus",
        "ReleaseEvidenceInputStatus",
        "evaluate_release_evidence",
        "evaluate_release_evidence_files",
        "inspect_release_evidence_input_files",
        "release_evidence_input_status_to_record",
        "release_evidence_to_record",
        "write_release_evidence_input_status_json",
        "write_release_evidence_json",
        "main",
    ]
    legacy_release_evidence = importlib.import_module("restaurant_kv_serving.release_evidence")
    assert not hasattr(legacy_release_evidence, "__all__")
    assert release_evidence.ReleaseEvidence.__module__ == "document_kv_cache.release_evidence"
    assert legacy_release_evidence.ReleaseEvidence is release_evidence.ReleaseEvidence
    assert release_evidence.main.__module__ == "document_kv_cache.release_evidence"
    assert legacy_release_evidence.main.__module__ == "restaurant_kv_serving.release_evidence"
    assert cache.__all__ == ["CacheTier", "ChunkCacheResult", "ChunkCacheStats", "ByteLRU", "ChunkCache"]
    assert manifest.__all__ == ["ManifestStore", "InMemoryManifestStore"]
    assert manifest.ManifestStore.__module__ == "document_kv_cache.manifest"
    assert manifest.InMemoryManifestStore.__module__ == "document_kv_cache.manifest"
    assert restaurant_kv_serving.ManifestStore is manifest.ManifestStore
    assert restaurant_kv_serving.InMemoryManifestStore is manifest.InMemoryManifestStore
    assert planner.__all__ == ["CacheRequest", "CachePlanner"]
    assert planner.CachePlanner.__module__ == "document_kv_cache.planner"
    assert restaurant_kv_serving.CachePlanner is planner.CachePlanner
    assert materializer.__all__ == ["MaterializedKV", "SegmentedMaterializedKV", "KVMaterializer"]
    assert storage.__all__ == [
        "RangeReader",
        "MemoryRangeReader",
        "DiskRangeReader",
        "UnityCatalogVolumeRangeReader",
        "RoutedRangeReader",
        "local_path",
        "unity_catalog_volume_path",
        "is_real_uc_volume_root",
    ]
    assert models.__all__ == [
        "DocumentChunkType",
        "DocumentChunkRole",
        "CacheGenerationMethod",
        "DocumentChunkMap",
        "CacheChunkType",
        "CacheChunkTypeSet",
        "DOCUMENT_CHUNK_TYPES",
        "LEGACY_RESTAURANT_CHUNK_TYPES",
        "KVCacheKey",
        "ChunkRef",
        "DocumentKVRequest",
        "PlanSegment",
        "MaterializationPlan",
        "chunk_type_role",
        "chunk_type_sort_order",
        "chunk_types_for_request",
    ]
    assert models.DocumentChunkType.__module__ == "document_kv_cache.models"
    assert models.DocumentChunkRole.__module__ == "document_kv_cache.models"
    assert models.KVCacheKey.__module__ == "document_kv_cache.models"
    assert models.ChunkRef.__module__ == "document_kv_cache.models"
    assert models.DocumentKVRequest.__module__ == "document_kv_cache.models"
    assert models.MaterializationPlan.__module__ == "document_kv_cache.models"
    assert models.DocumentChunkRole is restaurant_kv_serving.DocumentChunkRole
    assert models.chunk_type_role is restaurant_kv_serving.chunk_type_role
    assert service.__all__ == ["CacheRequest", "DocumentKVService"]
    assert service.DocumentKVService.__module__ == "document_kv_cache.service"
    assert restaurant_kv_serving.DocumentKVService is service.DocumentKVService
    assert "DocumentKVService" in service.__all__
    assert "RestaurantKVRequest" not in models.__all__
    assert "RestaurantKVService" not in service.__all__
    assert "ChunkType" not in models.__all__
    assert models.RestaurantKVRequest is restaurant_kv_serving.RestaurantKVRequest
    assert models.ChunkType is restaurant_kv_serving.ChunkType
    assert service.RestaurantKVService is restaurant_kv_serving.RestaurantKVService
    assert service.RestaurantKVService is service.DocumentKVService
    star_namespace: dict[str, object] = {}
    exec(
        "from document_kv_cache.models import *\n"
        "from document_kv_cache.service import *",
        star_namespace,
    )
    assert "DocumentKVRequest" in star_namespace
    assert "DocumentKVService" in star_namespace
    assert "RestaurantKVRequest" not in star_namespace
    assert "RestaurantKVService" not in star_namespace
    assert "ChunkType" not in star_namespace
    root_star_namespace: dict[str, object] = {}
    exec("from document_kv_cache import *", root_star_namespace)
    assert root_star_namespace["AdmissionQueue"] is admission.AdmissionQueue
    assert root_star_namespace["PreparedRequest"] is admission.PreparedRequest
    assert root_star_namespace["AttentionMechanism"] is engine_protocol.AttentionMechanism
    assert root_star_namespace["KVStorageLayout"] is engine_protocol.KVStorageLayout
    assert root_star_namespace["KVLayout"] is engine_protocol.KVLayout
    assert root_star_namespace["KVSegment"] is engine_protocol.KVSegment
    assert root_star_namespace["KVCacheHandle"] is engine_protocol.KVCacheHandle
    assert root_star_namespace["DTYPE_BYTE_WIDTHS"] is engine_protocol.DTYPE_BYTE_WIDTHS
    assert root_star_namespace["dtype_byte_width"] is engine_protocol.dtype_byte_width
    assert root_star_namespace["kv_storage_layout_from_value"] is engine_protocol.kv_storage_layout_from_value
    assert root_star_namespace["DiskRangeReader"] is storage.DiskRangeReader
    assert root_star_namespace["MemoryRangeReader"] is storage.MemoryRangeReader
    assert root_star_namespace["CacheTier"] is cache.CacheTier
    assert root_star_namespace["ByteLRU"] is cache.ByteLRU
    assert root_star_namespace["ChunkCache"] is cache.ChunkCache
    assert root_star_namespace["DocumentChunkType"] is models.DocumentChunkType
    assert root_star_namespace["DocumentChunkRole"] is models.DocumentChunkRole
    assert root_star_namespace["KVCacheKey"] is models.KVCacheKey
    assert root_star_namespace["ChunkRef"] is models.ChunkRef
    assert root_star_namespace["DocumentKVRequest"] is models.DocumentKVRequest
    assert root_star_namespace["MaterializationPlan"] is models.MaterializationPlan
    assert root_star_namespace["ManifestStore"] is manifest.ManifestStore
    assert root_star_namespace["InMemoryManifestStore"] is manifest.InMemoryManifestStore
    assert root_star_namespace["CacheRequest"] is planner.CacheRequest
    assert root_star_namespace["CachePlanner"] is planner.CachePlanner
    assert root_star_namespace["MaterializedKV"] is materializer.MaterializedKV
    assert root_star_namespace["SegmentedMaterializedKV"] is materializer.SegmentedMaterializedKV
    assert root_star_namespace["KVMaterializer"] is materializer.KVMaterializer
    assert root_star_namespace["EngineReadyRequest"] is engine.EngineReadyRequest
    assert root_star_namespace["ServingEngineConnector"] is engine.ServingEngineConnector
    assert root_star_namespace["build_engine_ready_request"] is engine.build_engine_ready_request
    assert root_star_namespace["build_handle_from_materialized"] is engine.build_handle_from_materialized
    assert root_star_namespace["DocumentKVService"] is service.DocumentKVService
    assert root_star_namespace["SourceChunk"] is workflow.SourceChunk
    assert root_star_namespace["SourceDocument"] is workflow.SourceDocument
    assert root_star_namespace["CacheBuildConfig"] is workflow.CacheBuildConfig
    assert root_star_namespace["CacheAdapterArtifact"] is workflow.CacheAdapterArtifact
    assert root_star_namespace["TrainingArtifacts"] is workflow.TrainingArtifacts
    assert root_star_namespace["TrainingAdapter"] is workflow.TrainingAdapter
    assert root_star_namespace["KVChunkGenerator"] is workflow.KVChunkGenerator
    assert root_star_namespace["CacheGenerationResult"] is workflow.CacheGenerationResult
    assert root_star_namespace["DocumentKVWorkflow"] is workflow.DocumentKVWorkflow
    assert root_star_namespace["local_path"] is storage.local_path
    assert root_star_namespace["unity_catalog_volume_path"] is storage.unity_catalog_volume_path
    assert root_star_namespace["is_real_uc_volume_root"] is storage.is_real_uc_volume_root
    assert openai_compatible.__all__ == [
        "TokenCounter",
        "PromptTextMode",
        "PromptTokenAccounting",
        "WhitespaceTokenCounter",
        "OpenAICompatibleEngineConfig",
        "OpenAICompatibleCompletionEngine",
    ]
    legacy_openai_compatible = importlib.import_module("restaurant_kv_serving.openai_compatible")
    assert openai_compatible.OpenAICompatibleCompletionEngine.__module__ == "document_kv_cache.openai_compatible"
    assert legacy_openai_compatible.OpenAICompatibleCompletionEngine is openai_compatible.OpenAICompatibleCompletionEngine
    assert set(openai_compatible.__all__) < set(legacy_openai_compatible.__all__)
    assert "urlopen" not in dir(openai_compatible)
    assert pr_evidence.__all__ == [
        "PR_EVIDENCE_RECORD_TYPE",
        "PR_EVIDENCE_VALIDATION_RECORD_TYPE",
        "GPT55_REVIEW_OUTCOMES",
        "PullRequestEvidence",
        "evaluate_pr_evidence",
        "evaluate_pr_evidence_directory",
        "evaluate_pr_evidence_file",
        "evaluate_pr_evidence_record",
        "pr_evidence_validation_to_record",
        "pr_evidence_to_record",
        "write_pr_evidence_json",
        "main",
    ]
    legacy_pr_evidence = importlib.import_module("restaurant_kv_serving.pr_evidence")
    pr_evidence_legacy_star_namespace = {}
    exec("from restaurant_kv_serving.pr_evidence import *", pr_evidence_legacy_star_namespace)
    assert not hasattr(legacy_pr_evidence, "__all__")
    assert sorted(k for k in pr_evidence_legacy_star_namespace if not k.startswith("__")) == [
        "Any",
        "GPT55_REVIEW_OUTCOMES",
        "Mapping",
        "PR_EVIDENCE_RECORD_TYPE",
        "PR_EVIDENCE_VALIDATION_RECORD_TYPE",
        "Path",
        "PullRequestEvidence",
        "Sequence",
        "annotations",
        "argparse",
        "dataclass",
        "evaluate_pr_evidence",
        "evaluate_pr_evidence_directory",
        "evaluate_pr_evidence_file",
        "evaluate_pr_evidence_record",
        "field",
        "json",
        "local_path",
        "main",
        "pr_evidence_to_record",
        "pr_evidence_validation_to_record",
        "write_pr_evidence_json",
    ]
    assert legacy_pr_evidence.PullRequestEvidence is pr_evidence.PullRequestEvidence
    assert legacy_pr_evidence.evaluate_pr_evidence is pr_evidence.evaluate_pr_evidence
    assert legacy_pr_evidence._semantic_issues is pr_evidence._semantic_issues
    assert legacy_pr_evidence.local_path is storage.local_path
    assert serving_env.__all__ == [
        "FASTAPI_CONSTRAINT",
        "HUGGINGFACE_HUB_CONSTRAINT",
        "NUMPY_CONSTRAINT",
        "PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT",
        "SERVING_ENVIRONMENT_PROFILES_RECORD_TYPE",
        "SGLANG_DEPENDENCY_CONSTRAINTS",
        "SGLANG_SERVING_ENVIRONMENT_PROFILE",
        "SGLANG_VERSION",
        "ServingEnvironmentProfile",
        "TOKENIZERS_CONSTRAINT",
        "TRANSFORMERS_CONSTRAINT",
        "VLLM_DEPENDENCY_CONSTRAINTS",
        "VLLM_SERVING_ENVIRONMENT_PROFILE",
        "VLLM_VERSION",
        "serving_environment_profile",
        "serving_environment_profile_to_record",
        "serving_environment_profiles",
        "serving_environment_profiles_to_record",
    ]
    legacy_serving_env = importlib.import_module("restaurant_kv_serving.serving_env")
    assert legacy_serving_env.__all__ == serving_env.__all__
    serving_env_legacy_star_namespace = {}
    exec("from restaurant_kv_serving.serving_env import *", serving_env_legacy_star_namespace)
    assert sorted(k for k in serving_env_legacy_star_namespace if not k.startswith("__")) == serving_env.__all__
    assert legacy_serving_env.ServingBackend is engine_adapters.ServingBackend
    assert legacy_serving_env.dataclass is not None
    assert "ServingBackend" not in legacy_serving_env.__all__
    assert native_probe_factories.__all__ == [
        "NativeProbeFactoryInspection",
        "NativeProbeFactoryUnavailable",
        "SGLANG_NATIVE_PROBE_FACTORY",
        "VLLM_NATIVE_PROBE_FACTORY",
        "builtin_native_probe_factories_to_record",
        "builtin_native_probe_factory_path",
        "inspect_builtin_native_probe_factories",
        "inspect_builtin_native_probe_factory",
        "native_probe_factory_inspection_to_record",
        "sglang_native_probe_factory",
        "vllm_native_probe_factory",
    ]
    legacy_native_probe_factories = importlib.import_module("restaurant_kv_serving.native_probe_factories")
    assert legacy_native_probe_factories.__all__ == native_probe_factories.__all__
    native_probe_legacy_star_namespace = {}
    exec("from restaurant_kv_serving.native_probe_factories import *", native_probe_legacy_star_namespace)
    assert sorted(k for k in native_probe_legacy_star_namespace if not k.startswith("__")) == native_probe_factories.__all__
    assert legacy_native_probe_factories.ServingBackend is engine_adapters.ServingBackend
    assert legacy_native_probe_factories.EngineKVProbeFactoryContext is restaurant_kv_serving.EngineKVProbeFactoryContext
    assert legacy_native_probe_factories.dataclass is not None
    assert "ServingBackend" not in legacy_native_probe_factories.__all__
    assert model_profiles.__all__ == [
        "DTYPE_BYTE_WIDTHS",
        "AttentionMechanism",
        "KVStorageLayout",
        "KVLayout",
        "dtype_byte_width",
        "kv_storage_layout_from_value",
        "KVModelProfile",
        "MODEL_PROFILE_RECORD_TYPE",
        "ModelProfileDefinition",
        "ModelProfileRegistry",
        "QWEN3_4B_INSTRUCT_PROFILE",
        "builtin_model_profiles",
        "default_model_profile_registry",
        "get_model_profile",
        "layout_for_model",
        "model_profile_definition_from_record",
        "model_profile_definition_to_record",
        "read_model_profile_definition_json",
        "write_model_profile_definition_json",
    ]
    assert workflow.__all__ == [
        "SourceChunk",
        "SourceDocument",
        "CacheBuildConfig",
        "CacheAdapterArtifact",
        "TrainingArtifacts",
        "TrainingAdapter",
        "KVChunkGenerator",
        "CacheGenerationResult",
        "DocumentKVWorkflow",
    ]
    assert workflow.SourceDocument.__module__ == "document_kv_cache.workflow"
    assert workflow.DocumentKVWorkflow.__module__ == "document_kv_cache.workflow"
    legacy_workflow = importlib.import_module("restaurant_kv_serving.workflow")
    workflow_legacy_star_namespace = {}
    exec("from restaurant_kv_serving.workflow import *", workflow_legacy_star_namespace)
    assert sorted(k for k in workflow_legacy_star_namespace if not k.startswith("__")) == [
        "CacheAdapterArtifact",
        "CacheBuildConfig",
        "CacheGenerationMethod",
        "CacheGenerationResult",
        "CachePlanner",
        "ChunkRef",
        "DocumentChunkType",
        "DocumentKVRequest",
        "DocumentKVService",
        "DocumentKVWorkflow",
        "EngineReadyRequest",
        "Iterable",
        "KVChunkGenerator",
        "KVLayout",
        "KVMaterializer",
        "KVStorageLayout",
        "ManifestStore",
        "Mapping",
        "MaterializedKV",
        "PackChunk",
        "Path",
        "Protocol",
        "SegmentedMaterializedKV",
        "Sequence",
        "SourceChunk",
        "SourceDocument",
        "TrainingAdapter",
        "TrainingArtifacts",
        "annotations",
        "build_engine_ready_request",
        "dataclass",
        "field",
        "kv_storage_layout_from_value",
        "write_kvpack",
    ]
    assert legacy_workflow.SourceDocument is workflow.SourceDocument
    assert legacy_workflow.DocumentKVWorkflow is workflow.DocumentKVWorkflow
    assert legacy_workflow._effective_cache_method is workflow._effective_cache_method
    assert legacy_workflow.dataclass is not None
    assert not hasattr(legacy_workflow, "__all__")

    for module in (kvpack, materializer, openai_compatible, workflow):
        assert "reexport_public" not in dir(module)
        assert "Path" not in module.__all__
        assert "dataclass" not in module.__all__


def test_package_level_submodule_imports_use_document_namespace_after_symbol_lookup():
    from document_kv_cache import CacheTier  # noqa: F401
    from document_kv_cache import cache

    assert cache.__name__ == "document_kv_cache.cache"
    assert CacheTier is cache.CacheTier
    assert cache.CacheTier.__module__ == "document_kv_cache.cache"
    assert cache.CacheTier is restaurant_kv_serving.CacheTier
    assert cache.ChunkCacheResult is restaurant_kv_serving.ChunkCacheResult


def test_poetry_metadata_uses_public_package_name_and_legacy_script_aliases():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    poetry = pyproject["tool"]["poetry"]
    scripts = project["scripts"]
    artifact_includes = poetry["include"]
    expected_scripts = {
        "document-kv-benchmark-plan": "document_kv_cache.benchmark_plan:main",
        "document-kv-run-benchmark-plan": "document_kv_cache.benchmark_plan_executor:main",
        "document-kv-databricks-job": "document_kv_cache.databricks_job:main",
        "document-kv-databricks-runs": "document_kv_cache.databricks_runs:main",
        "document-kv-storage-benchmark": "document_kv_cache.storage_benchmark:main",
        "document-kv-storage-benchmark-databricks-job": "document_kv_cache.databricks_storage_benchmark_job:main",
        "document-kv-templates": "document_kv_cache.template_resources:main",
        "document-kv-release-evidence": "document_kv_cache.release_evidence:main",
        "document-kv-release-bundle": "document_kv_cache.release_bundle:main",
        "document-kv-pr-evidence": "document_kv_cache.pr_evidence:main",
        "document-kv-engine-probe": "document_kv_cache.engine_probe:main",
        "document-kv-engine-probe-databricks-job": "document_kv_cache.databricks_engine_probe_job:main",
        "document-kv-vllm-smoke": "document_kv_cache.vllm_smoke:main",
        "document-kv-vllm-smoke-databricks-job": "document_kv_cache.databricks_vllm_smoke_job:main",
        "restaurant-kv-benchmark-plan": "restaurant_kv_serving.benchmark_plan:main",
        "restaurant-kv-run-benchmark-plan": "restaurant_kv_serving.benchmark_plan_executor:main",
        "restaurant-kv-databricks-job": "restaurant_kv_serving.databricks_job:main",
        "restaurant-kv-databricks-runs": "restaurant_kv_serving.databricks_runs:main",
        "restaurant-kv-storage-benchmark": "restaurant_kv_serving.storage_benchmark:main",
        "restaurant-kv-storage-benchmark-databricks-job": "restaurant_kv_serving.databricks_storage_benchmark_job:main",
        "restaurant-kv-release-evidence": "restaurant_kv_serving.release_evidence:main",
        "restaurant-kv-release-bundle": "restaurant_kv_serving.release_bundle:main",
        "restaurant-kv-pr-evidence": "restaurant_kv_serving.pr_evidence:main",
        "restaurant-kv-engine-probe": "restaurant_kv_serving.engine_probe:main",
        "restaurant-kv-engine-probe-databricks-job": "restaurant_kv_serving.databricks_engine_probe_job:main",
        "restaurant-kv-vllm-smoke": "restaurant_kv_serving.vllm_smoke:main",
        "restaurant-kv-vllm-smoke-databricks-job": "restaurant_kv_serving.databricks_vllm_smoke_job:main",
    }
    expected_includes = [
        {
            "path": "src/document_kv_cache/py.typed",
            "format": ["sdist", "wheel"],
        },
        {
            "path": "src/restaurant_kv_serving/py.typed",
            "format": ["sdist", "wheel"],
        },
        {
            "path": "src/document_kv_cache/templates/**/*.yml",
            "format": ["sdist", "wheel"],
        },
        {
            "path": "src/document_kv_cache/templates/**/*.md",
            "format": ["sdist", "wheel"],
        }
    ]

    assert project["name"] == "document-kv-cache"
    assert project["description"] == "Document KV-cache orchestration and materialization for long-context LLM serving."
    assert project["requires-python"] == ">=3.11,<4.0"
    assert project["authors"] == [{"name": "OpenTable Data Science"}]
    assert set(project["keywords"]) == {
        "databricks",
        "kv-cache",
        "llm-serving",
        "long-context",
        "sglang",
        "vllm",
    }
    assert {
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    }.issubset(set(project["classifiers"]))
    assert {package["include"] for package in poetry["packages"]} == {
        "document_kv_cache",
        "restaurant_kv_serving",
    }
    assert artifact_includes == expected_includes
    assert scripts == expected_scripts
