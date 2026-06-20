"""Compatibility wrapper for :mod:`document_kv_cache.workflow`."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from document_kv_cache.engine import EngineReadyRequest, KVLayout, build_engine_ready_request
from document_kv_cache.engine_protocol import KVStorageLayout, kv_storage_layout_from_value
from document_kv_cache.kvpack import PackChunk, write_kvpack
from document_kv_cache.manifest import ManifestStore
from document_kv_cache.materializer import KVMaterializer, MaterializedKV, SegmentedMaterializedKV
from document_kv_cache.models import CacheGenerationMethod, ChunkRef, DocumentChunkType, DocumentKVRequest
from document_kv_cache.planner import CachePlanner
from document_kv_cache.service import DocumentKVService
from document_kv_cache.workflow import (
    CacheAdapterArtifact,
    CacheBuildConfig,
    CacheGenerationResult,
    DocumentKVWorkflow,
    KVChunkGenerator,
    SourceChunk,
    SourceDocument,
    TrainingAdapter,
    TrainingArtifacts,
    _cache_method_value,
    _effective_cache_method,
    _engine_adapter_ids,
    _non_empty_string,
    _non_empty_string_tuple,
)
