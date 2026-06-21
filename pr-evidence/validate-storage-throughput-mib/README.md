# Validate Storage Throughput MiB

## Scope

- Require strict release storage benchmark rows to include positive
  `throughput_mib_per_second`.
- Cross-check `throughput_mib_per_second` against
  `throughput_bytes_per_second / 1024 / 1024`.
- Update release-ready fixtures so bundle validation exercises the complete
  storage throughput contract.

## Verification

- `poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_impossible_storage_latency_rows tests/test_release_evidence.py::test_evaluate_release_evidence_accepts_complete_v1_storage_and_engine_probe_records tests/test_storage_benchmark.py::test_run_storage_benchmark_reports_memory_disk_and_uc_readers -q`
- `git diff --check`
- `poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py tests/test_storage_benchmark.py -q`
- `poetry run pytest -q`
- `poetry check`
- `poetry run python -m compileall -q src tests`
