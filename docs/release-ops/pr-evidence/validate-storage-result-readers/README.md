# Validate Storage Result Readers

This PR hardens strict release evidence validation for storage benchmarks. The
release validator now rejects duplicate, empty, and unsupported `reader_id`
values in raw storage benchmark result rows instead of relying only on the
top-level `readers` and embedded `release_storage_evidence` summaries.

The change keeps the storage benchmark output format unchanged; it only makes
ambiguous or noisy release evidence fail before a strict V1 bundle can be
published.

Verification:

- `poetry run pytest tests/test_release_evidence.py::test_evaluate_release_evidence_accepts_storage_readers_in_any_order tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_duplicate_or_missing_storage_readers tests/test_release_evidence.py::test_evaluate_release_evidence_rejects_invalid_storage_result_readers -q`
- `poetry run pytest tests/test_release_evidence.py -q`
- `poetry run pytest tests/test_storage_benchmark.py -q`
