# Document Request For Text

This PR adds `DocumentKVRequest.for_text_document(...)`, a convenience
constructor that pairs with `SourceDocument.from_text(...)` for the common
single-document, single-chunk serving path.

## Verification

- `poetry run pytest tests/test_planner_materializer.py tests/test_workflow.py tests/test_project_governance.py -q`
- `git diff --check`
- `poetry run pytest -q`
- `poetry check`
- `poetry install --dry-run`
- `poetry build`

## Review

GPT-5.5 approved the branch with no blocking findings. The reviewer confirmed
the helper preserves request validation and matches planner behavior for the
default `document` chunk.
