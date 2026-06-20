# Document Request For Chunks

This slice adds `DocumentKVRequest.for_document_chunks(...)` so callers can build
single-document, multi-chunk requests without constructing the raw chunk map by
hand. `for_text_document(...)` now delegates to the same helper while preserving
its existing `include_static=False` behavior.

Verification:

- `poetry run pytest tests/test_planner_materializer.py tests/test_workflow.py tests/test_project_governance.py -q`
- `poetry run pytest -q`
- `git diff --check`
- `poetry check`
- `poetry install --dry-run`
- `poetry build`

GPT-5.5 review approved the change as narrow and contract-preserving.
