# Strict V1 Custom Suite Match

Strict V1 release bundle tests now cover the positive non-default suite-id path:
when the V1 benchmark artifact and benchmark plan execution sidecar use the same
custom `suite_id`, the strict bundle succeeds and preserves that suite id in the
bundled artifacts.

Verification:

- `poetry run pytest tests/test_release_bundle.py::test_build_release_bundle_strict_v1_accepts_matching_non_default_plan_source_suite_id tests/test_release_bundle.py::test_build_release_bundle_strict_v1_requires_matching_plan_source_suite_id -q`
- `poetry run pytest tests/test_release_bundle.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check`
- `poetry run python -m compileall -q src tests`
