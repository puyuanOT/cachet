# Document Selection Helper

This slice adds `DocumentKVRequest.for_document_selection(...)` for requests
that select chunks from multiple documents. The single-document
`for_document_chunks(...)` helper now delegates through it, so all public helpers
share the same constructor validation and immutable chunk-map normalization.

Verification:

- `poetry run pytest tests/test_planner_materializer.py tests/test_workflow.py tests/test_project_governance.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check`
- `poetry install --dry-run`
- `poetry build`

GPT-5.5 review approved the change as narrow and contract-preserving.
