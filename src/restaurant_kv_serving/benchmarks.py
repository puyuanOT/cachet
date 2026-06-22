"""Compatibility wrapper for :mod:`document_kv_cache.benchmarks`."""

from __future__ import annotations

from document_kv_cache._reexport import reexport_public

__all__ = reexport_public(
    "document_kv_cache.benchmarks",
    (
        "SUPPORTED_V1_DATASETS",
        "SUPPORTED_V1_HARDWARE_TARGETS",
        "DEFAULT_V1_MODEL_ID",
        "DEFAULT_HARDWARE_TARGET",
        "BASELINE_PREFILL_ARM",
        "CACHE_REUSE_ARM",
        "DOCUMENT_KV_REQUEST_ID_PARAM",
        "DOCUMENT_KV_HANDOFF_JSON_PARAM",
        "DOCUMENT_KV_HANDOFF_RECORD_PARAM",
        "DOCUMENT_KV_PAYLOAD_URI_PARAM",
        "FINAL_ANSWER_CUE",
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
        "SourceDocument",
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
        "validate_v1_hardware_target",
        "validate_v1_dataset",
    ),
    globals(),
)

del reexport_public
