# Recursive PR Evidence Validation

This PR-evidence sidecar covers the governance slice that makes
`document-kv-pr-evidence --validate-directory` validate PR evidence sidecars in
nested slice folders.

The validator now recursively scans matching JSON files by default and skips
only valid `document_kv.pr_evidence_validation.v1` summary records. Invalid
validation summaries, malformed JSON, and invalid PR-evidence sidecars still
surface as failures.

## Review

GPT-5.5 found two validation-summary edge cases:

- validation summaries were skipped by `record_type` alone;
- an `ok: true` validation summary with an empty `files` mapping could be
  skipped because `all([])` is true.

Both were fixed. The skip now requires `ok is True`, a non-empty `files`
mapping, string file keys, mapping values, and nested PR evidence records that
evaluate cleanly. The final GPT-5.5 pass found no remaining issues.

## Verification

- `poetry run pytest tests/test_pr_evidence.py -q`
- `PYTHONPATH=src python -m document_kv_cache.pr_evidence --validate-directory pr-evidence --output-json /tmp/pr-evidence-recursive-validation.json`
- `poetry run pytest -q`
- `poetry check`
- `find src tests -name '*.py' -print0 | xargs -0 python -m py_compile`
- `poetry build`
