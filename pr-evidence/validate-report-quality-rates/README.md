# Validate Report Quality Rates

## What Changed

- Added strict validation for optional V1 report-row `answer_found_rate` and `exact_match_rate` values.
- Present quality rates must be finite numeric values between 0 and 1, and booleans are rejected.
- Added regression coverage for malformed string, out-of-range, and boolean quality-rate summaries.

## Why

Release evidence compares cache reuse against no-cache prefill. Malformed aggregated quality rates can make a V1 benchmark look valid even when the reported quality signal is not machine-usable.

## Scope

- `src/document_kv_cache/release_evidence.py`
- `tests/test_release_evidence.py`
- `pr-evidence/validate-report-quality-rates/README.md`

## Verification

- `python -m pytest tests/test_release_evidence.py -q` -> 50 passed
- `python -m pytest -q` -> 869 passed
- `poetry check` -> All set
- `poetry install --dry-run` -> succeeded
- `poetry build` -> succeeded
- `git diff --check` -> clean
- GPT-5.5 review -> APPROVED, no findings after row-specific test assertion fix
