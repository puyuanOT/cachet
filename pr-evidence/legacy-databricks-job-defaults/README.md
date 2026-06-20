# Legacy Databricks Job Defaults Evidence

This PR makes `restaurant_kv_serving.databricks_job` load clean
`document_kv_cache.databricks_job` defaults from source before exposing the
legacy facade surface.

The facade keeps honest legacy config classes and legacy namespace monkeypatch
hooks, while public module monkeypatches applied before legacy import no longer
become legacy defaults.

Verification:

- `python -m py_compile src/restaurant_kv_serving/databricks_job.py tests/test_databricks_job.py`
- `poetry run pytest tests/test_databricks_job.py::test_legacy_databricks_job_import_order_does_not_capture_public_monkeypatch tests/test_databricks_job.py::test_legacy_databricks_job_reexports_document_owned_types tests/test_databricks_job.py::test_legacy_databricks_job_config_pickle_uses_honest_legacy_module tests/test_databricks_job.py::test_legacy_databricks_job_main_respects_legacy_namespace_monkeypatch tests/test_databricks_job.py::test_legacy_databricks_job_config_respects_legacy_node_type_validator_monkeypatch -q`
- `poetry run pytest tests/test_databricks_job.py -q`
- `poetry run pytest tests/test_public_package.py tests/test_project_governance.py -q`
- `git diff --check`
- `poetry run pytest`
- `poetry check`
- `poetry build`

Review:

- GPT-5.5 review approved with no merge-blocking findings.
