# Legacy Databricks Job Class Fingerprints Evidence

This evidence covers the compatibility hardening that isolates
`restaurant_kv_serving.databricks_job` class bases from pre-import in-place
mutations to public `document_kv_cache.databricks_job` dataclass classes.

The legacy wrapper still subclasses the public Databricks job config classes on
clean imports, but now falls back to source-loaded document defaults when public
class bodies, dataclass metadata, or dataclass-generated function closures have
been mutated before legacy import. The fingerprint path avoids raw user object
ordering, equality, attribute access, and recursive closure traversal during
import.

Verification:

- `poetry run pytest tests/test_databricks_job.py -k 'source_benchmark_config_base or source_cluster_config_base or unorderable_keys or bad_equality or field_default_mutation or bad_attribute_access or fields_metadata_replacement or init_closure_mutation or recursive_function_closure'`
- `poetry run pytest tests/test_databricks_job.py::test_legacy_databricks_job_reexports_document_owned_types tests/test_public_package.py::test_public_cli_submodules_are_importable_under_document_namespace -q`
- `poetry run pytest tests/test_databricks_job.py tests/test_public_package.py tests/test_project_governance.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check`
- `poetry build`
- GPT-5.5 review with findings resolved and final approval
