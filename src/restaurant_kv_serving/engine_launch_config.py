"""Compatibility namespace for document KV engine launch-config validation."""

from __future__ import annotations

from document_kv_cache._reexport import reexport_public

__all__ = reexport_public(
    "document_kv_cache.engine_launch_config",
    (
        "ENGINE_LAUNCH_CONFIG_EVIDENCE_RECORD_TYPE",
        "ENGINE_LAUNCH_CONFIG_EVIDENCE_SCHEMA_VERSION",
        "REQUIRED_ENGINE_LAUNCH_CONFIG_BACKENDS",
        "EngineLaunchConfigEvidence",
        "engine_launch_config_evidence_to_record",
        "engine_launch_config_record_issues",
        "evaluate_engine_launch_config_evidence",
        "read_engine_launch_config_json",
        "validate_engine_launch_config_record",
        "write_engine_launch_config_evidence_json",
    ),
    globals(),
)

del reexport_public
