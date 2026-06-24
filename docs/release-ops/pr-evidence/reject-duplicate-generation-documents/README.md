# Reject Duplicate Generation Documents

This PR hardens `DocumentKVWorkflow.generate_cache(...)` so a cache-generation
batch cannot contain multiple `SourceDocument` entries with the same
`document_id`. The validation runs before optional training, KV generation,
shard writes, and manifest insertion.

The workflow now also reports non-`SourceDocument` entries at the generation
boundary, giving callers a direct error before plugin hooks run.

Verification:

- `poetry run pytest tests/test_workflow.py::test_workflow_generates_and_prepares_multi_document_selection tests/test_workflow.py::test_workflow_rejects_duplicate_generation_document_ids_before_training tests/test_workflow.py::test_workflow_rejects_non_source_document_generation_entries tests/test_workflow.py::test_workflow_invokes_optional_training_adapter -q`
- `poetry run pytest tests/test_workflow.py tests/test_planner_materializer.py -q`
