"""Public wrapper for :mod:`restaurant_kv_serving.databricks_runs`."""

from __future__ import annotations

from collections.abc import Sequence

from document_kv_cache._reexport import LegacyMainBridge, reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.databricks_runs",
    (
        "DEFAULT_DATABRICKS_HOST_ENV",
        "DEFAULT_DATABRICKS_TOKEN_ENV",
        "DEFAULT_DATABRICKS_TIMEOUT_SECONDS",
        "DATABRICKS_RUN_STATUS_RECORD_TYPE",
        "DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE",
        "DatabricksWorkspaceConfig",
        "databricks_workspace_config_from_env",
        "submit_databricks_run",
        "get_databricks_run",
        "summarize_databricks_run",
        "summarize_databricks_run_submit_payload",
        "write_databricks_run_response_json",
        "read_databricks_run_submit_payload",
        "main",
    ),
    globals(),
)


_main_bridge = LegacyMainBridge(
    legacy_module_name="restaurant_kv_serving.databricks_runs",
    public_namespace=globals(),
    hook_names=(
        "databricks_workspace_config_from_env",
        "submit_databricks_run",
        "get_databricks_run",
        "summarize_databricks_run",
        "summarize_databricks_run_submit_payload",
        "write_databricks_run_response_json",
        "read_databricks_run_submit_payload",
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    return _main_bridge(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

del LegacyMainBridge
del reexport_public
