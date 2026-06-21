"""Compatibility namespace for document KV engine launch-config validation."""

from __future__ import annotations

from document_kv_cache._reexport import reexport_public

__all__ = reexport_public(
    "document_kv_cache.engine_launch_config",
    (
        "DEFAULT_ENGINE_LAUNCH_CONFIG_KV_INJECTION_METHOD",
        "DEFAULT_ENGINE_LAUNCH_CONFIG_SCHEMA_VERSION",
        "DEFAULT_SGLANG_DOCUMENT_KV_MODULE_PATH",
        "DEFAULT_SGLANG_ENGINE_LAUNCH_CONFIG_RECORD_TYPE",
        "DEFAULT_VLLM_DOCUMENT_KV_MODULE_PATH",
        "DEFAULT_VLLM_ENGINE_LAUNCH_CONFIG_RECORD_TYPE",
        "ENGINE_LAUNCH_CONFIG_EVIDENCE_RECORD_TYPE",
        "ENGINE_LAUNCH_CONFIG_EVIDENCE_SCHEMA_VERSION",
        "REQUIRED_ENGINE_LAUNCH_CONFIG_BACKENDS",
        "EngineLaunchConfigEvidence",
        "build_sglang_launch_config",
        "build_vllm_launch_config",
        "engine_launch_config_evidence_to_record",
        "engine_launch_config_record_issues",
        "evaluate_engine_launch_config_evidence",
        "main",
        "read_engine_launch_config_json",
        "validate_engine_launch_config_record",
        "write_engine_launch_config_json",
        "write_engine_launch_config_evidence_json",
    ),
    globals(),
)

del reexport_public
