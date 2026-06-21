# Validate Release Suite Metadata

## Summary

- Added strict V1 release evidence validation for benchmark suite metadata: `suite_id`, release dataset list, and positive example count.
- Updated release evidence and bundle test fixtures to include the suite fields emitted by the benchmark runner.
- Added a regression test that rejects malformed suite metadata while the rest of the V1 artifact remains otherwise release-ready.

## Refactor Scope

The release evidence validator already routes V1 benchmark checks through small helper functions. This change extracts suite metadata checks into `_validate_v1_suite_metadata` and keeps the public record format and release evidence output semantics unchanged except for newly rejected malformed V1 suite records.

## Verification

- `poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_malformed_v1_suite_metadata tests/test_release_evidence.py::test_evaluate_release_evidence_accepts_complete_v1_storage_and_engine_probe_records -q`
- `poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py -q`
- `poetry run pytest -q`
- `poetry check`
- `poetry run python -m compileall -q src tests`
- `git diff --check`
