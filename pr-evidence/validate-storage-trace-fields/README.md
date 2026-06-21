# Validate Storage Trace Fields

## Scope

- Require strict release storage benchmark records to include non-empty
  `benchmark_id`, `workspace_dir`, and `shard_uri`.
- Require emitted storage benchmark dimensions to be positive integers:
  `chunk_count`, `chunk_bytes`, `repeats`, `parallelism`, and `align_bytes`.
- Update release-ready test fixtures to match the trace fields emitted by
  `storage_benchmark_result_to_record`.

## Verification

- `poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_missing_storage_trace_fields tests/test_release_evidence.py::test_evaluate_release_evidence_accepts_complete_v1_storage_and_engine_probe_records tests/test_release_bundle.py::test_build_release_bundle_rejects_inputs_that_fail_release_evidence_validation -q`
- `git diff --check`
- `poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py tests/test_storage_benchmark.py -q`
- `poetry run pytest -q`
- `poetry check`
- `poetry run python -m compileall -q src tests`
