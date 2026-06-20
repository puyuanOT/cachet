# Legacy Benchmark Plan Facade Defaults Evidence

This PR makes `restaurant_kv_serving.benchmark_plan` load clean
`document_kv_cache.benchmark_plan` defaults from source before exposing the
legacy facade surface.

The facade still preserves legacy namespace monkeypatch hooks, but public module
monkeypatches applied before importing the legacy module no longer become legacy
defaults.

Verification:

- `poetry run pytest tests/test_benchmark_plan.py::test_legacy_benchmark_plan_import_order_does_not_capture_public_monkeypatch tests/test_benchmark_plan.py::test_legacy_benchmark_plan_main_isolates_dataclass_method_globals tests/test_benchmark_plan.py::test_legacy_benchmark_plan_without_patches_returns_public_dataclass_instances tests/test_benchmark_plan.py::test_legacy_benchmark_plan_main_respects_legacy_namespace_monkeypatch tests/test_benchmark_plan.py::test_legacy_benchmark_plan_saved_original_wrapper_does_not_recurse -q`
- `poetry run pytest tests/test_benchmark_plan.py -q`
- `poetry run pytest tests/test_public_package.py tests/test_project_governance.py -q`
- `python -m py_compile src/restaurant_kv_serving/benchmark_plan.py tests/test_benchmark_plan.py`
- `git diff --check`
- `poetry run pytest`
- `poetry check`
- `poetry build`

Review:

- GPT-5.5 review approved with no merge-blocking findings.
