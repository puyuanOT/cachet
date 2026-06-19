"""Public document namespace for storage reader benchmarks."""

from __future__ import annotations

from collections.abc import Sequence

from document_kv_cache._reexport import LegacyMainBridge, reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.storage_benchmark",
    (
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
    ),
    globals(),
)

_main_bridge = LegacyMainBridge(
    public_namespace=globals(),
    legacy_module_name="restaurant_kv_serving.storage_benchmark",
    hook_names=("run_storage_benchmark", "write_storage_benchmark_result_json"),
)


def main(argv: Sequence[str] | None = None) -> int:
    return _main_bridge(argv)


__all__.append("main")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


del LegacyMainBridge, reexport_public
