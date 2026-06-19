"""Public wrapper for :mod:`restaurant_kv_serving.benchmark_plan`."""

from __future__ import annotations

from document_kv_cache._reexport import reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.benchmark_plan",
    (
        "PLAN_VERSION",
        "ENGINE_PROBE_TARGETS_RECORD_TYPE",
        "ENGINE_PROBE_TARGETS_SCHEMA_VERSION",
        "BenchmarkDatasetPath",
        "BenchmarkCommand",
        "StorageBenchmarkPlanConfig",
        "EngineProbePlanConfig",
        "ReleaseEvidencePlanConfig",
        "ReleaseBundlePlanConfig",
        "BenchmarkPlanConfig",
        "BenchmarkJobPlan",
        "build_v1_benchmark_plan",
        "benchmark_job_plan_to_record",
        "engine_probe_targets_to_record",
        "write_benchmark_job_plan_json",
        "write_benchmark_job_plan_shell",
        "write_engine_probe_targets_json",
        "main",
    ),
    globals(),
)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

del reexport_public
