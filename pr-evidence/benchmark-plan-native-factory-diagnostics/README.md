# Benchmark Plan Native Factory Diagnostics Evidence

This directory records PR evidence for generating native-probe factory
diagnostics from the benchmark plan and bundling that generated sidecar into
release handoff artifacts.

The GPT-5.5 review requested canonical-path deduplication when the generated
diagnostics sidecar and an explicit release-bundle sidecar refer to the same
file through different path spellings. The implementation now dedupes with the
same canonical path helper used by generated artifact collision checks.

Verification:

- `git diff --check`
- `poetry run python -m compileall -q src tests`
- `poetry run pytest -q`
- `poetry run pytest tests/test_benchmark_plan.py tests/test_project_governance.py -q`
