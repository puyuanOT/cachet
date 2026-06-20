# Document Release Evidence Ownership PR Evidence

This folder records PR evidence for the slice that moved release evidence
validation into the public `document_kv_cache.release_evidence` namespace while
keeping `restaurant_kv_serving.release_evidence` as a legacy compatibility
facade for downstream imports and CLI monkeypatch hooks.
