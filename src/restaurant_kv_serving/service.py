"""Compatibility wrapper for :mod:`document_kv_cache.service`."""

from __future__ import annotations

from collections.abc import Mapping

from document_kv_cache.admission import AdmissionQueue, PreparedRequest
from document_kv_cache.engine import EngineReadyRequest, build_engine_ready_request
from document_kv_cache.engine_protocol import KVLayout
from document_kv_cache.materializer import KVMaterializer
from document_kv_cache.models import CacheGenerationMethod, DocumentKVRequest, RestaurantKVRequest
from document_kv_cache.planner import CachePlanner, CacheRequest
from document_kv_cache.service import DocumentKVService


RestaurantKVService = DocumentKVService
