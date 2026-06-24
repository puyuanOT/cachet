# Reject Duplicate Manifest Batch Keys

This PR hardens `InMemoryManifestStore.put_many(...)` so one incoming batch of
manifest refs cannot silently contain duplicate cache keys. The store still
allows an explicit key replacement in a later `put_many(...)` call, preserving
the existing update path while making ambiguous generated batches fail before
mutating the manifest.

Verification:

- `poetry run pytest tests/test_planner_materializer.py::test_manifest_validates_refs_before_mutating_store tests/test_planner_materializer.py::test_manifest_rejects_duplicate_keys_in_single_batch_before_mutating_store tests/test_planner_materializer.py::test_manifest_allows_explicit_key_replacement_across_batches tests/test_planner_materializer.py::test_manifest_validates_and_normalizes_document_filters -q`
- `poetry run pytest tests/test_planner_materializer.py tests/test_workflow.py -q`
