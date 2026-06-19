"""Public document namespace for engine protocol data structures."""

from __future__ import annotations

from document_kv_cache._reexport import reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.engine_protocol",
    (
        "DTYPE_BYTE_WIDTHS",
        "AttentionMechanism",
        "KVStorageLayout",
        "dtype_byte_width",
        "kv_storage_layout_from_value",
        "KVLayout",
        "KVSegment",
        "KVCacheHandle",
    ),
    globals(),
)

del reexport_public
