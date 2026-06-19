"""Public document namespace for service orchestration."""

from __future__ import annotations

from importlib import import_module

from document_kv_cache._reexport import reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.service",
    (
        "CacheRequest",
        "DocumentKVService",
    ),
    globals(),
)

_legacy_service = import_module("restaurant_kv_serving.service")
RestaurantKVService = _legacy_service.RestaurantKVService

del _legacy_service
del import_module
del reexport_public
