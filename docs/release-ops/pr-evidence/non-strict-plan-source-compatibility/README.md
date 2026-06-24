# Non-Strict Plan Source Compatibility

Non-strict release bundles remain compatible with legacy benchmark plan
execution sidecars that lack V1 target metadata. The strict V1 gate still
requires the target metadata; this regression protects diagnostic bundle
workflows from accidentally inheriting strict release-only requirements.

Verification:

- `poetry run pytest tests/test_release_bundle.py::test_build_release_bundle_non_strict_allows_legacy_plan_source_target tests/test_release_bundle.py::test_build_release_bundle_strict_v1_rejects_wrong_plan_source_target -q`
- `poetry run pytest tests/test_release_bundle.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check`
- `poetry run python -m compileall -q src tests`
