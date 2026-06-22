"""Compatibility wrapper for :mod:`document_kv_cache.benchmark_handoffs`."""

from __future__ import annotations

from document_kv_cache._reexport import reexport_public

__all__ = reexport_public(
    "document_kv_cache.benchmark_handoffs",
    (
        "BENCHMARK_HANDOFF_MANIFEST_RECORD_TYPE",
        "BENCHMARK_HANDOFF_MANIFEST_SCHEMA_VERSION",
        "BenchmarkHandoffBundleResult",
        "BenchmarkHandoffEntry",
        "BenchmarkHandoffManifest",
        "build_benchmark_handoff_manifest_from_jsonl",
        "benchmark_handoff_manifest_from_record",
        "benchmark_handoff_manifest_to_record",
        "generate_benchmark_handoff_bundles",
        "enrich_benchmark_jsonl_with_handoffs",
        "enrich_benchmark_records_with_handoffs",
        "load_benchmark_kv_chunk_generator",
        "read_benchmark_handoff_manifest_json",
        "write_benchmark_handoff_manifest_json",
        "bundle_main",
        "manifest_main",
        "main",
    ),
    globals(),
)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

del reexport_public
