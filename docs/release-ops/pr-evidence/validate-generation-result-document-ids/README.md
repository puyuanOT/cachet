# Validate Generation Result Document IDs

This PR hardens `CacheGenerationResult` so its `document_ids` field must match
the unique `ChunkRef.key.document_id` values carried by its refs, in first-seen
order. The workflow already emits that relationship; this change makes direct
public construction follow the same traceability contract.

The empty-result path remains valid when both refs and document ids are empty.

Verification:

- `poetry run pytest tests/test_workflow.py::test_cache_generation_result_normalizes_public_fields tests/test_workflow.py::test_cache_generation_result_derives_document_id_order_from_refs tests/test_workflow.py::test_cache_generation_result_rejects_invalid_public_fields tests/test_workflow.py::test_cache_generation_result_allows_empty_generation_result -q`
- `poetry run pytest tests/test_workflow.py tests/test_planner_materializer.py -q`
