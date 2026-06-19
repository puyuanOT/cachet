"""Compatibility wrapper for :mod:`document_kv_cache.benchmark_runner`."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module

from document_kv_cache._reexport import LegacyMainBridge, reexport_public

_document_module = import_module("document_kv_cache.benchmark_runner")

__all__ = reexport_public(
    "document_kv_cache.benchmark_runner",
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
        "argparse",
        "json",
        "random",
        "Iterable",
        "Mapping",
        "Sequence",
        "dataclass",
        "field",
        "Path",
        "Any",
        "Callable",
        "Literal",
        "Protocol",
        "BASELINE_PREFILL_ARM",
        "CACHE_REUSE_ARM",
        "DEFAULT_HARDWARE_TARGET",
        "DEFAULT_V1_MODEL_ID",
        "BenchmarkArm",
        "BenchmarkComparison",
        "BenchmarkExample",
        "BenchmarkPromptParts",
        "BenchmarkReportRow",
        "BenchmarkSuite",
        "InferenceMeasurement",
        "baseline_prefill_arm",
        "build_prompt_parts",
        "compare_to_baseline",
        "document_kv_cache_arm",
        "evaluate_v1_benchmark_evidence",
        "summarize_measurements",
        "validate_v1_dataset",
        "DocumentChunkType",
        "local_path",
        "SourceChunk",
        "SourceDocument",
    ),
    globals(),
)

_openai_compatible_engine = _document_module._openai_compatible_engine

_main_bridge = LegacyMainBridge(
    legacy_module_name="document_kv_cache.benchmark_runner",
    public_namespace=globals(),
    hook_names=("run_openai_compatible_v1_benchmark",),
)


def main(argv: Sequence[str] | None = None) -> int:
    return _main_bridge(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

del LegacyMainBridge
del reexport_public
