# Document Dataset Prep Ownership

This PR-evidence sidecar covers the refactor slice that moves the V1 dataset
preparation implementation from the legacy restaurant package into
`document_kv_cache.dataset_prep`.

The slice makes `document_kv_cache.dataset_prep` the implementation owner for
Biography, HotpotQA, MusiQue, and NIAH normalization into canonical benchmark
JSONL. `restaurant_kv_serving.dataset_prep` remains as a compatibility wrapper
for existing jobs and tests.

## Review

GPT-5.5 review found no issues. The review specifically checked import cycles,
CLI behavior, public API and star-import compatibility, and legacy caller
regressions. It also manually probed both module CLIs with `PYTHONPATH=src`.

The document module keeps a curated `__all__` for the dataset-prep API. The
legacy wrapper intentionally preserves the broader historical module-level
import surface, including `validate_v1_dataset` and `local_path`.

## Verification

- `poetry run pytest tests/test_dataset_prep.py tests/test_public_package.py tests/test_project_governance.py::test_legacy_restaurant_imports_in_tests_are_explicitly_scoped -q`
- `python -m py_compile src/document_kv_cache/dataset_prep.py src/restaurant_kv_serving/dataset_prep.py tests/test_dataset_prep.py tests/test_public_package.py tests/test_project_governance.py`
- `poetry run pytest tests/test_dataset_prep.py tests/test_public_package.py tests/test_benchmark_plan.py tests/test_benchmark_plan_executor.py tests/test_benchmark_runner.py tests/test_release_evidence.py tests/test_project_governance.py::test_legacy_restaurant_imports_in_tests_are_explicitly_scoped -q`
- `poetry run pytest -q`
- `poetry check`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry build`
- `PYTHONPATH=src python -m document_kv_cache.pr_evidence --validate-directory pr-evidence --output-json /tmp/pr-evidence-document-dataset-prep-ownership-validation.json`
