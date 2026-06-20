"""Compatibility namespace for old admission helper imports."""

from __future__ import annotations

from document_kv_cache._reexport import reexport_public

__all__ = reexport_public(
    "document_kv_cache.admission",
    (
        "PreparedRequest",
        "AdmissionQueue",
    ),
    globals(),
)

del reexport_public
