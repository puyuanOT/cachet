"""Public wrapper for :mod:`restaurant_kv_serving.databricks_job`."""

from __future__ import annotations

from collections.abc import Sequence

from document_kv_cache._reexport import LegacyMainBridge, reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.databricks_job",
    (
        "DEFAULT_AWS_G5_NODE_TYPE",
        "DEFAULT_DATABRICKS_SPARK_VERSION",
        "DEFAULT_DATABRICKS_RUN_NAME",
        "DEFAULT_DATABRICKS_TASK_KEY",
        "DEFAULT_DATABRICKS_DATA_SECURITY_MODE",
        "DEDICATED_DATABRICKS_DATA_SECURITY_MODE",
        "SINGLE_USER_DATABRICKS_DATA_SECURITY_MODES",
        "DatabricksSingleNodeG5ClusterConfig",
        "DatabricksBenchmarkJobConfig",
        "validate_aws_g5_node_type",
        "build_single_node_g5_cluster",
        "build_databricks_run_submit_payload",
        "write_databricks_run_submit_json",
        "write_databricks_runner_script",
        "main",
    ),
    globals(),
)


_main_bridge = LegacyMainBridge(
    legacy_module_name="restaurant_kv_serving.databricks_job",
    public_namespace=globals(),
    hook_names=(
        "DatabricksBenchmarkJobConfig",
        "write_databricks_runner_script",
        "build_databricks_run_submit_payload",
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    return _main_bridge(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

del LegacyMainBridge
del reexport_public
