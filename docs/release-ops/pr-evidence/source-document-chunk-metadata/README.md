# Source Document Chunk Metadata

This PR adds optional metadata support to `SourceDocument.from_texts(...)` for
both static context and reusable content chunks.

## Verification

- `poetry run pytest tests/test_workflow.py tests/test_project_governance.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check`
- `poetry install --dry-run`
- `poetry build`

## Review

GPT-5.5 initially found that the branch was uncommitted and that surplus
metadata could be silently ignored. The implementation now rejects unknown
`chunk_metadata` IDs and `static_chunk_metadata` without `static_text`; GPT-5.5
approved the committed branch with no findings.
