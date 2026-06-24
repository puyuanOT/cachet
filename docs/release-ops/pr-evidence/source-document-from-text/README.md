# Source Document From Text

This PR adds `SourceDocument.from_text(...)`, a convenience constructor for the
common single-document, single-chunk workflow path.

## Verification

- `poetry run pytest tests/test_workflow.py tests/test_project_governance.py -q`
- `git diff --check`
- `poetry run pytest -q`
- `poetry check`
- `poetry install --dry-run`
- `poetry build`

## Review

GPT-5.5 approved the branch with no blocking findings. The reviewer confirmed
that the default chunk id and README `DocumentKVRequest` example match planner
behavior.
