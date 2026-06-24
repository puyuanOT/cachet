# Reject Duplicate Request Chunk IDs

This PR hardens `DocumentKVRequest`, `FrozenDocumentChunkMap`, and the legacy
`RestaurantKVRequest` compatibility path so a single document or restaurant
selection cannot request the same generated chunk id twice.

Chunk ids are compared after the same string coercion used by the planner when
building `KVCacheKey.chunk_id`, so values like `2` and `"2"` are treated as the
same cache key within one document selection.

Verification:

- `poetry run pytest tests/test_planner_materializer.py::test_document_kv_request_validates_metadata_and_chunk_map tests/test_planner_materializer.py::test_document_kv_request_for_document_chunks_reuses_request_validation tests/test_planner_materializer.py::test_frozen_document_chunk_map_normalizes_direct_construction tests/test_planner_materializer.py::test_restaurant_kv_request_validates_legacy_review_map -q`
- `poetry run pytest tests/test_planner_materializer.py tests/test_workflow.py -q`
