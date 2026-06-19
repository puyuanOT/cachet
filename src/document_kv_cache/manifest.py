"""Public document namespace for manifest lookup interfaces."""

from __future__ import annotations

from document_kv_cache._reexport import reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.manifest",
    (
        "ManifestStore",
        "InMemoryManifestStore",
    ),
    globals(),
)

del reexport_public
