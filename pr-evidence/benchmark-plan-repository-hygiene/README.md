# Benchmark Plan Repository Hygiene Sidecar

This PR-evidence sidecar covers the command-plan slice that lets
`document_kv_cache.benchmark_plan` generate repository hygiene evidence during
the benchmark workflow.

The benchmark plan can now append `document_kv_cache.repository_hygiene` before
release evidence validation, write a `document_kv.repository_hygiene.v1`
sidecar, and automatically pass that generated sidecar into release bundle
assembly. If callers also provide an explicit release-bundle hygiene sidecar,
the plan accepts equivalent paths and rejects conflicting paths.

## Review

GPT-5.5/high reviewed the branch and found one positional-compatibility issue:
the new `BenchmarkPlanConfig.repository_hygiene_output_json` field originally
preceded `native_probe_factories_output_json`, which could silently rebind
positional callers. The field now follows the existing native-probe field, and a
regression test pins that ordering. GPT-5.5 re-reviewed and approved the fix.

## Verification

- `poetry run pytest -q tests/test_benchmark_plan.py`
- `poetry run pytest -q`
- `git diff --check`
- `poetry run python -m compileall -q src tests`
- `poetry run python -m document_kv_cache.benchmark_plan --help`
- GPT-5.5/high review and re-review after resolving the positional-compatibility finding
