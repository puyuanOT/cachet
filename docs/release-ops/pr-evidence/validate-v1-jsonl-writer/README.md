# Validate V1 JSONL Writer

This PR hardens `write_v1_jsonl(...)` so it validates canonical V1 benchmark
records before creating the output file. Invalid rows now fail early with a
line-numbered error instead of producing JSONL that the benchmark runner rejects
later.

The writer also normalizes document, chunk, and metadata fields through the
same dataset-preparation helpers used by the raw dataset converters, so rows it
writes remain loadable by `load_benchmark_jsonl(...)`. A final validation pass
uses the benchmark runner's record loader contract before the output path is
opened.

Verification:

- `poetry run pytest tests/test_dataset_prep.py tests/test_benchmark_runner.py::test_load_benchmark_jsonl_accepts_canonical_schema tests/test_benchmark_runner.py::test_load_benchmark_jsonl_validates_records tests/test_benchmark_runner.py::test_load_v1_jsonl_suite_combines_dataset_files tests/test_benchmark_runner.py::test_load_v1_jsonl_suite_rejects_dataset_mismatch`
- `git diff --check && poetry run pytest -q && poetry check && poetry run python -m compileall -q src tests`
