# Legacy Storage Benchmark Facade Evidence

This PR replaces the duplicated `restaurant_kv_serving.storage_benchmark`
implementation with a compatibility facade over
`document_kv_cache.storage_benchmark`.

The facade preserves legacy namespace monkeypatch hooks for CLI and benchmark
tests, but loads a clean copy of document defaults so public module monkeypatches
applied before importing the legacy module are not captured as legacy defaults.

Verification:

- `poetry run pytest tests/test_storage_benchmark.py -q`
- `poetry run pytest tests/test_public_package.py tests/test_project_governance.py tests/test_storage_benchmark.py -q`
- `python -m py_compile src/restaurant_kv_serving/storage_benchmark.py tests/test_storage_benchmark.py`
- `git diff --check`
- `poetry run pytest`

Review:

- GPT-5.5 review initially found public-module monkeypatch capture for
  constants/classes.
- The blocker was fixed and the same reviewer approved with no remaining merge
  blockers.
