# Document Storage Ownership

This PR-evidence sidecar covers the refactor slice that moves memory, disk, and
Unity Catalog KV range readers into `document_kv_cache.storage`.

The slice makes the document package the implementation owner for shard URI
resolution, UC Volume path confinement, checksum-validated range reads, and
routed memory/disk/UC reader dispatch. The legacy `restaurant_kv_serving.storage`
module remains as a compatibility wrapper.

## Review

GPT-5.5 reviewed the storage ownership inversion and found two compatibility
gaps:

- legacy reader classes inherited public document class methods, so public class
  monkeypatches could still change legacy construction/mutation behavior;
- root-level storage helpers were lazily routed from `document_kv_cache` but
  were absent from root star-imports.

The legacy wrapper now overrides constructor and mutation methods that should
remain legacy-owned, and the document package root adds explicit document-owned
storage exports to `__all__`. Regression tests cover both cases before merge.

## Verification

- `poetry run pytest tests/test_storage.py tests/test_planner_materializer.py tests/test_public_package.py tests/test_project_governance.py -q`
- `python -m py_compile src/document_kv_cache/storage.py src/restaurant_kv_serving/storage.py src/document_kv_cache/__init__.py tests/test_storage.py tests/test_public_package.py tests/test_project_governance.py`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry run pytest -q`
- `poetry check`
- `git diff --check`
- `poetry build`
- `PYTHONPATH=src poetry run python -m document_kv_cache.pr_evidence --validate-directory pr-evidence`
- repository secret scan over README, package metadata, source, tests,
  Databricks helpers, PR evidence, and GitHub metadata
- GPT-5.5 review, fixes for requested changes, and re-review before merge
