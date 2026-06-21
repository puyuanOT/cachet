# Validate Storage UC Root Consistency

## Scope

- Require strict release storage benchmark records to report
  `uc_volume_is_real=true` at the top level.
- Reject storage records where `release_storage_evidence.uc_volume_root` does
  not match the top-level `uc_volume_root`.
- Preserve the existing release-reader, latency, throughput, and real UC
  Volume path checks.

## Verification

- `poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_inconsistent_storage_uc_volume_metadata tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_minimal_storage_rows_and_missing_uc_root tests/test_release_evidence.py::test_evaluate_release_evidence_accepts_complete_v1_storage_and_engine_probe_records -q`
- `git diff --check`
- `poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py tests/test_storage_benchmark.py -q`
- `poetry run pytest -q`
- `poetry check`
- `poetry run python -m compileall -q src tests`
