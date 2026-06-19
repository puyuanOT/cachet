"""Public document namespace for benchmark contracts."""

from __future__ import annotations

from document_kv_cache._reexport import reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.benchmarks",
    (
        "SUPPORTED_V1_DATASETS",
        "DEFAULT_V1_MODEL_ID",
        "DEFAULT_HARDWARE_TARGET",
        "BASELINE_PREFILL_ARM",
        "CACHE_REUSE_ARM",
        "BenchmarkDatasetSpec",
        "BenchmarkPromptParts",
        "BenchmarkExample",
        "BenchmarkSuite",
        "BenchmarkArm",
        "InferenceMeasurement",
        "LatencySummary",
        "BenchmarkReportRow",
        "BenchmarkComparison",
        "V1BenchmarkEvidence",
        "baseline_prefill_arm",
        "document_kv_cache_arm",
        "v1_dataset_specs",
        "dataset_spec",
        "build_prompt_parts",
        "build_prefill_prompt",
        "build_cache_prefix_text",
        "build_cache_suffix_text",
        "format_document_context",
        "summarize_measurements",
        "compare_to_baseline",
        "evaluate_v1_benchmark_evidence",
        "normalize_answer",
        "exact_match",
        "answer_found",
        "validate_v1_dataset",
    ),
    globals(),
)

del reexport_public
