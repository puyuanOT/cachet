# Strict V1 Plan Source Target

Strict V1 release bundles now require benchmark plan execution sidecars to
prove they executed the V1 benchmark plan: `plan_version: v1`, model
`qwen3:4b-instruct`, hardware target `aws-g5`, a non-empty suite id, and a
positive command count.

Verification:

- `poetry run pytest tests/test_release_bundle.py::test_build_release_bundle_strict_v1_accepts_complete_release_artifact_set tests/test_release_bundle.py::test_build_release_bundle_strict_v1_rejects_wrong_plan_source_target -q`
- `poetry run pytest tests/test_release_bundle.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check`
- `poetry run python -m compileall -q src tests`
