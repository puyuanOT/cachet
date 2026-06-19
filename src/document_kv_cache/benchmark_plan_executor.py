"""Public wrapper for :mod:`restaurant_kv_serving.benchmark_plan_executor`."""

from __future__ import annotations

from collections.abc import Sequence

from document_kv_cache._reexport import LegacyMainBridge, reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.benchmark_plan_executor",
    (
        "BENCHMARK_PLAN_EXECUTION_RECORD_TYPE",
        "BENCHMARK_PLAN_SOURCE_RECORD_TYPE",
        "BenchmarkCommandResult",
        "execute_benchmark_job_plan",
        "execute_benchmark_job_plan_json",
        "benchmark_command_results_to_record",
        "benchmark_plan_source_to_record",
        "benchmark_plan_source_payload_to_record",
        "write_benchmark_command_results_json",
        "main",
    ),
    globals(),
)


_main_bridge = LegacyMainBridge(
    legacy_module_name="restaurant_kv_serving.benchmark_plan_executor",
    public_namespace=globals(),
    hook_names=(
        "execute_benchmark_job_plan",
        "execute_benchmark_job_plan_json",
        "benchmark_command_results_to_record",
        "benchmark_plan_source_to_record",
        "benchmark_plan_source_payload_to_record",
        "write_benchmark_command_results_json",
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    return _main_bridge(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

del LegacyMainBridge
del reexport_public
