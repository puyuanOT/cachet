# V1 Evidence Unexpected Arms

This PR makes V1 benchmark evidence fail closed when report rows or comparisons
contain unsupported arms beyond the expected baseline/cache pair.

## Verification

- `poetry run pytest tests/test_benchmarks.py tests/test_benchmark_runner.py tests/test_release_evidence.py tests/test_release_bundle.py -q`
- `git diff --check`
- `poetry check`
- `poetry run pytest -q`
- `poetry install --dry-run`
- `poetry build`

## Review

GPT-5.5 requested one compatibility fix: append `unexpected_arms` after the
existing `unexpected_datasets` dataclass field so positional constructor
semantics stay stable. The fix was applied, verification was rerun, and the
reviewer approved the branch.
