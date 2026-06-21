"""Compatibility wrapper for :mod:`document_kv_cache.probe_fixtures`."""

from __future__ import annotations

from document_kv_cache._reexport import reexport_public

__all__ = reexport_public(
    "document_kv_cache.probe_fixtures",
    (
        "ENGINE_PROBE_FIXTURE_RECORD_TYPE",
        "ENGINE_PROBE_FIXTURE_SCHEMA_VERSION",
        "DEFAULT_ENGINE_PROBE_FIXTURE_REQUEST_ID",
        "EngineProbeFixtureConfig",
        "EngineProbeFixtureResult",
        "engine_probe_fixture_result_to_record",
        "write_qwen3_v1_engine_probe_fixture",
        "parse_args",
        "main",
    ),
    globals(),
)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


del reexport_public
