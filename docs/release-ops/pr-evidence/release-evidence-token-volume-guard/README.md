# Release Evidence Token Volume Guard

This PR tightens V1 release evidence so benchmark artifacts must prove real
tokenized work. Successful measurements now require positive prompt and
completion token counts, and report rows require at least one successful request
plus positive prompt-token, completion-token, and output-throughput summaries.

The guard prevents zero-token or summary-only benchmark stubs from passing the
release gate for the AWS g5/Qwen3 V1 benchmark.

Verification:

- `poetry run pytest tests/test_release_evidence.py -q`
- `poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py tests/test_benchmark_runner.py tests/test_benchmarks.py -q`
- `poetry run pytest tests/test_release_evidence.py tests/test_release_bundle.py tests/test_project_governance.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check`
- `poetry build`

GPT-5.5 review approved the diff with no findings.
