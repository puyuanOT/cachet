# Validate Comparison Quality Deltas

## What Changed

- Added strict validation for V1 benchmark comparison `exact_match_delta` and `answer_found_delta` fields.
- Quality deltas must remain finite numeric values and must now be bounded to `[-1, 1]`.
- Added regression coverage for impossible positive and negative quality deltas.

## Why

V1 release evidence compares cache reuse against no-cache prefill. Quality deltas are differences of rates, so values outside `[-1, 1]` are invalid and should not be accepted into release artifacts.

## Scope

- `src/document_kv_cache/release_evidence.py`
- `tests/test_release_evidence.py`
- `pr-evidence/validate-comparison-quality-deltas/README.md`

## Verification

- `python -m pytest tests/test_release_evidence.py -q` -> 51 passed
- `python -m pytest -q` -> 870 passed
- `poetry check` -> All set
- `poetry install --dry-run` -> succeeded
- `poetry build` -> succeeded
- `git diff --check` -> clean
- GPT-5.5 review -> APPROVED, no findings
