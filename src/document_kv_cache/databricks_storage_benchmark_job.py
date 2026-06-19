"""Public wrapper for :mod:`restaurant_kv_serving.databricks_storage_benchmark_job`."""

from __future__ import annotations

from collections.abc import Sequence

from document_kv_cache._reexport import LegacyMainBridge, reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.databricks_storage_benchmark_job",
    (
        "DEFAULT_DATABRICKS_STORAGE_BENCHMARK_RUN_NAME",
        "DEFAULT_DATABRICKS_STORAGE_BENCHMARK_TASK_KEY",
        "DEFAULT_DATABRICKS_STORAGE_BENCHMARK_PURPOSE",
        "STORAGE_BENCHMARK_RUNNER_SCRIPT",
        "DatabricksStorageBenchmarkJobConfig",
        "build_databricks_storage_benchmark_run_submit_payload",
        "write_databricks_storage_benchmark_run_submit_json",
        "write_databricks_storage_benchmark_runner_script",
        "main",
    ),
    globals(),
)


_main_bridge = LegacyMainBridge(
    legacy_module_name="restaurant_kv_serving.databricks_storage_benchmark_job",
    public_namespace=globals(),
    hook_names=(
        "DatabricksStorageBenchmarkJobConfig",
        "build_databricks_storage_benchmark_run_submit_payload",
        "write_databricks_storage_benchmark_runner_script",
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    return _main_bridge(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

del LegacyMainBridge
del reexport_public
