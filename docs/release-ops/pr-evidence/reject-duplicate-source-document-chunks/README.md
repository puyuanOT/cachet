# Reject Duplicate Source Document Chunks

This PR hardens `SourceDocument` validation so a directly constructed document
cannot contain duplicate source chunk identities. Chunk identity is the pair of
`chunk_type` and `chunk_id`, matching the fields that later become cache-key
parts during generation.

The validation keeps same-id chunks valid when their chunk types differ, so a
document may still carry separate static and content chunks with the same human
label.

Verification:

- `poetry run pytest tests/test_workflow.py::test_source_document_validates_and_normalizes_public_inputs tests/test_workflow.py::test_source_document_rejects_duplicate_chunk_identities tests/test_workflow.py::test_source_document_allows_same_chunk_id_for_different_chunk_types tests/test_workflow.py::test_source_document_from_texts_validates_helper_inputs -q`
- `poetry run pytest tests/test_workflow.py tests/test_planner_materializer.py -q`
