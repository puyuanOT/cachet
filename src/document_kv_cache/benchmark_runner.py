"""Public wrapper for :mod:`restaurant_kv_serving.benchmark_runner`."""

from __future__ import annotations

from collections.abc import Sequence

from document_kv_cache._reexport import LegacyMainBridge, reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.benchmark_runner",
    (
        "BENCHMARK_RUN_RECORD_TYPE",
        "DEFAULT_OPENAI_COMPLETIONS_ENDPOINT",
        "BenchmarkGeneration",
        "BenchmarkEngineRequest",
        "BenchmarkEngine",
        "BenchmarkRunResult",
        "OpenAICompatibleBenchmarkConfig",
        "OpenAICompatibleEngineFactory",
        "default_benchmark_arms",
        "run_benchmark_suite",
        "load_v1_jsonl_suite",
        "load_benchmark_jsonl",
        "benchmark_run_result_to_record",
        "write_benchmark_run_result_json",
        "run_openai_compatible_v1_benchmark",
        "main",
    ),
    globals(),
)


_main_bridge = LegacyMainBridge(
    legacy_module_name="restaurant_kv_serving.benchmark_runner",
    public_namespace=globals(),
    hook_names=("run_openai_compatible_v1_benchmark",),
)


def main(argv: Sequence[str] | None = None) -> int:
    return _main_bridge(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

del LegacyMainBridge
del reexport_public
