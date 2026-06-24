# Audit Allowed Open PRs

## What Changed

- Added `allowed_count` and sanitized `allowed` pull-request summaries to GitHub governance open-PR sidecars.
- Tightened release-bundle validation so allowed PR summaries must match `allowed_numbers`, `allowed_count`, and `total_count`.
- Documented that allowed current PRs are recorded with sanitized summaries.

## Why

Release governance can allow the current PR to remain open while evidence is generated, but the release bundle needs enough metadata to audit which PR was allowed instead of only carrying a number.

## Scope

- `src/document_kv_cache/github_governance.py`
- `src/document_kv_cache/release_bundle.py`
- `tests/test_github_governance.py`
- `tests/test_release_bundle.py`
- `README.md`
- `pr-evidence/audit-allowed-open-prs/README.md`

## Verification

- `poetry run pytest tests/test_github_governance.py tests/test_release_bundle.py -q` -> 57 passed
- `poetry run pytest -q` -> 876 passed
- `poetry check` -> All set
- `poetry install --dry-run` -> succeeded
- `poetry build` -> succeeded
- `git diff --check` -> clean
- GPT-5.5 review -> APPROVED after fixing two release-bundle validation findings
