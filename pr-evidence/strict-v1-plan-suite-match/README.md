# Strict V1 Plan Suite Match

Strict V1 release bundles now cross-check benchmark plan execution evidence
against the bundled V1 benchmark artifact. A plan execution sidecar must carry
the same `plan_source.suite_id` as the V1 benchmark `suite.suite_id`; matching
model and hardware metadata alone is not enough.

Verification:

- `poetry run pytest tests/test_release_bundle.py::test_build_release_bundle_strict_v1_accepts_complete_release_artifact_set tests/test_release_bundle.py::test_build_release_bundle_strict_v1_requires_matching_plan_source_suite_id -q`
- `poetry run pytest tests/test_release_bundle.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check`
- `poetry run python -m compileall -q src tests`
