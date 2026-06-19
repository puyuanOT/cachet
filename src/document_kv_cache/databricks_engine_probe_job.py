"""Public wrapper for :mod:`restaurant_kv_serving.databricks_engine_probe_job`."""

from __future__ import annotations

from collections.abc import Sequence

from document_kv_cache._reexport import LegacyMainBridge, reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.databricks_engine_probe_job",
    (
        "DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME",
        "DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY",
        "DEFAULT_DATABRICKS_ENGINE_PROBE_PURPOSE",
        "DEFAULT_DATABRICKS_ENGINE_PROBE_BACKEND_CONFIG_KEY",
        "ENGINE_PROBE_RUNNER_SCRIPT",
        "DatabricksEngineProbeJobConfig",
        "DatabricksEngineProbeMatrixJobConfig",
        "DatabricksEngineProbeTargetConfig",
        "DatabricksEngineProbeTargetsFile",
        "build_databricks_engine_probe_run_submit_payload",
        "build_databricks_engine_probe_matrix_run_submit_payload",
        "read_databricks_engine_probe_targets_json",
        "read_databricks_engine_probe_targets_file_json",
        "write_databricks_engine_probe_run_submit_json",
        "write_databricks_engine_probe_matrix_run_submit_json",
        "write_databricks_engine_probe_runner_script",
        "main",
    ),
    globals(),
)


_main_bridge = LegacyMainBridge(
    legacy_module_name="restaurant_kv_serving.databricks_engine_probe_job",
    public_namespace=globals(),
    hook_names=(
        "DatabricksEngineProbeJobConfig",
        "DatabricksEngineProbeMatrixJobConfig",
        "DatabricksEngineProbeTargetsFile",
        "build_databricks_engine_probe_run_submit_payload",
        "build_databricks_engine_probe_matrix_run_submit_payload",
        "write_databricks_engine_probe_runner_script",
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    return _main_bridge(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

del LegacyMainBridge
del reexport_public
