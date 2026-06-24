# Benchmark Plan GitHub Governance Sidecar

This PR-evidence sidecar covers the command-plan slice that lets
`document_kv_cache.benchmark_plan` generate GitHub governance evidence during
the benchmark workflow.

The benchmark plan can now append `document_kv_cache.github_governance` before
release evidence validation, write a
`document_kv.github_repository_governance.v1` sidecar, and automatically pass
that generated sidecar into release bundle assembly. If callers also provide an
explicit release-bundle GitHub-governance sidecar, the plan accepts equivalent
paths and rejects conflicting paths.

## Refactor

Generated single-sidecar path validation is now shared by GitHub governance and
repository hygiene so future release-readiness sidecars can reuse the same
canonical-path conflict check.

## Review

GPT-5.5/high reviewed the branch before merge.

## Verification

- `poetry run pytest -q tests/test_benchmark_plan.py`
- `poetry run pytest -q tests/test_benchmark_plan.py tests/test_project_governance.py tests/test_public_package.py`
- `poetry run pytest -q`
- `git diff --check`
- `poetry run python -m compileall -q src tests`
- `poetry run python -m document_kv_cache.benchmark_plan --help`
