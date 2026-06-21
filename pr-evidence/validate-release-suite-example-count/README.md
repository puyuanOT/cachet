# Validate Release Suite Example Count

## Summary

- Added strict V1 release evidence validation that compares `suite.examples` against the unique `(dataset, example_id)` identities present in benchmark measurements.
- Kept repeated raw measurements valid when they reuse the same benchmark example identity.
- Added regression coverage for suite example-count drift.

## Why

The benchmark runner emits `suite.examples` as the number of benchmark examples in the suite. Strict release evidence already validates the presence of that field; this PR prevents release bundles from claiming a different benchmark size than the measurement rows prove.

## Refactor Scope

The change stays inside the existing release evidence validator. It adds one helper beside the suite metadata validator and preserves the existing row, measurement, and comparison validation helpers.

## Verification

- `poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_v1_suite_example_count_mismatch tests/test_release_evidence.py::test_evaluate_release_evidence_allows_repeated_raw_measurements -q`
- `poetry run pytest tests/test_release_evidence.py -q`
- `poetry run pytest -q`
- `poetry check`
- `poetry run python -m compileall -q src tests`
- `git diff --check`
