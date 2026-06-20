# Document Release Bundle Ownership PR Evidence

This folder records PR evidence for the slice that moved release bundle
packaging into the public `document_kv_cache.release_bundle` namespace while
keeping `restaurant_kv_serving.release_bundle` as a legacy compatibility facade
for downstream imports, star imports, and CLI monkeypatch hooks.
