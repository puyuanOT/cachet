# Require Admin Branch Protection

## What Changed

- GitHub governance readiness now requires branch protection to apply to administrators.
- Release-bundle validation now independently rejects GitHub governance sidecars where `branch_protection.enforce_admins` is false or missing.
- Added focused regressions for both the governance collector and release-bundle sidecar validation.

## Why

The project prohibits direct pushes to `main`. If administrators can bypass branch protection, the GitHub governance sidecar should not be considered release-ready, and strict release bundles should reject the sidecar even if it incorrectly claims `ok: true`.

## Scope

- `src/document_kv_cache/github_governance.py`
- `src/document_kv_cache/release_bundle.py`
- `tests/test_github_governance.py`
- `tests/test_release_bundle.py`
- `pr-evidence/require-admin-branch-protection/README.md`

## Verification

- `python -m pytest tests/test_github_governance.py -q` -> 13 passed
- `python -m pytest tests/test_github_governance.py tests/test_release_bundle.py -q` -> 52 passed
- `python -m pytest -q` -> 871 passed
- `poetry check` -> All set
- `poetry install --dry-run` -> succeeded
- `poetry build` -> succeeded
- `git diff --check` -> clean
- GPT-5.5 review -> initial P1 finding fixed, then APPROVED with no findings
