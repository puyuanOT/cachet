"""Runtime-facing provider for loading Cachet KV payloads into vLLM V1."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import mmap
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Protocol

from document_kv_cache.artifact_identity import (
    RuntimeCompatibilityHandshake,
    RuntimeIdentity,
)
from document_kv_cache.cache import ByteLRU
from document_kv_cache.engine_protocol import (
    KVKeyPositionEncoding,
)
from document_kv_cache.storage import local_path
from document_kv_cache.engine import EngineReadyRequest
from document_kv_cache.methods import MethodRegistry, default_method_registry
from document_kv_cache.engine_adapters import (
    EngineAdapterSpec,
    EngineKVBindAction,
    EngineKVConnectorActions,
    EngineKVInjectionPlan,
    EngineKVReleaseAction,
    EngineKVReservationAction,
    EngineKVSegmentCopyAction,
    PayloadMode,
    ServingBackend,
    build_engine_kv_injection_plan,
    engine_kv_connector_actions_from_record,
    engine_kv_connector_actions_to_record,
    read_engine_adapter_request_json,
    validate_engine_kv_connector_actions,
    vllm_adapter_spec,
)
from document_kv_cache.reuse_contract import (
    RuntimeOperationHandlerRegistry,
    apply_runtime_operation_handlers,
)
from document_kv_cache.engine_probe import read_engine_adapter_payload
from vllm_kv_injection.block_mapping import BlockSpan, plan_token_blocks
from vllm_kv_injection.paged_kv_copy import inject_kv_cache_layer, slot_mapping_from_blocks
from vllm_kv_injection.vllm_native_provider_constants import (
    DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM,
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_HANDOFF_RECORD_PARAM,
    DOCUMENT_KV_HANDOFF_SOURCE_FACTORY_CONFIG_KEY,
    DOCUMENT_KV_NATIVE_PROVIDER_FACTORY,
    DOCUMENT_KV_PAYLOAD_CACHE_MAX_BYTES_CONFIG_KEY,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM,
    DOCUMENT_KV_REQUIRE_RUNTIME_HANDSHAKE_CONFIG_KEY,
    DOCUMENT_KV_REQUEST_ID_PARAM,
    DOCUMENT_KV_RUNTIME_IDENTITY_CONFIG_KEY,
    DOCUMENT_KV_SEGMENTED_LOAD_STRATEGY_ENV,
    DOCUMENT_KV_TELEMETRY_JSONL_CONFIG_KEY,
)
from vllm_kv_injection.vllm_layer_mapping import (
    DOCUMENT_KV_VLLM_LAYER_MAPPING_RECORD_TYPE,
    DOCUMENT_KV_VLLM_LAYER_MAPPING_SCHEMA_VERSION,
    DocumentKVVLLMLayerMappingInspection,
    document_kv_vllm_layer_index_from_name,
    document_kv_vllm_layer_mapping_record_issues,
    document_kv_vllm_layer_mapping_to_record,
    document_kv_vllm_probe_layer_names,
    inspect_document_kv_vllm_layer_mapping,
    validate_document_kv_vllm_layer_mapping_record,
)
from vllm_kv_injection.vllm_dynamic_connector import DocumentKVConnectorStats, VLLMSupportsHMA

__all__ = [
    "DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM",
    "DOCUMENT_KV_HANDOFF_JSON_PARAM",
    "DOCUMENT_KV_HANDOFF_RECORD_PARAM",
    "DOCUMENT_KV_HANDOFF_SOURCE_FACTORY_CONFIG_KEY",
    "DOCUMENT_KV_NATIVE_PROVIDER_FACTORY",
    "DOCUMENT_KV_PAYLOAD_CACHE_MAX_BYTES_CONFIG_KEY",
    "DOCUMENT_KV_PAYLOAD_URI_PARAM",
    "DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM",
    "DOCUMENT_KV_REQUEST_ID_PARAM",
    "DOCUMENT_KV_REQUIRE_RUNTIME_HANDSHAKE_CONFIG_KEY",
    "DOCUMENT_KV_RUNTIME_IDENTITY_CONFIG_KEY",
    "DOCUMENT_KV_SEGMENTED_LOAD_STRATEGY_ENV",
    "DOCUMENT_KV_TELEMETRY_JSONL_CONFIG_KEY",
    "DOCUMENT_KV_VLLM_LAYER_MAPPING_RECORD_TYPE",
    "DOCUMENT_KV_VLLM_LAYER_MAPPING_SCHEMA_VERSION",
    "DocumentKVHandoffLoad",
    "DocumentKVHandoffSource",
    "DocumentKVLoadRequest",
    "DocumentKVConnectorMetadata",
    "DocumentKVVLLMLayerMappingInspection",
    "DocumentKVNativeProvider",
    "DocumentKVNativeProbeConnector",
    "KVTransferParamsDocumentKVSource",
    "build_document_kv_provider",
    "document_kv_vllm_probe_layer_names",
    "document_kv_vllm_layer_index_from_name",
    "document_kv_vllm_layer_mapping_record_issues",
    "document_kv_vllm_layer_mapping_to_record",
    "inspect_document_kv_vllm_layer_mapping",
    "validate_document_kv_vllm_layer_mapping_record",
]

try:  # pragma: no cover - exercised only with live vLLM installed.
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # type: ignore[import-not-found]
        KVConnectorMetadata as _KVConnectorMetadata,
    )
except Exception:  # pragma: no cover - lightweight local test path.

    class _KVConnectorMetadata:  # type: ignore[no-redef]
        pass


class DocumentKVHandoffSource(Protocol):
    """Lookup boundary for Cachet handoffs attached to vLLM requests."""

    def get_load(self, request: object) -> "DocumentKVHandoffLoad | None": ...


@dataclass(frozen=True, slots=True)
class DocumentKVHandoffLoad:
    """Validated Cachet connector actions plus either payload bytes or a payload URI."""

    actions: EngineKVConnectorActions
    payload: bytes | tuple[bytes, ...] | None = None
    payload_uri: str | None = None
    method_registry: MethodRegistry | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        registry = _method_registry(self.method_registry)
        validate_engine_kv_connector_actions(
            self.actions,
            method_registry=registry,
        )
        object.__setattr__(self, "method_registry", registry)
        if self.actions.reservation.backend != ServingBackend.VLLM:
            raise ValueError("Document KV vLLM loads require vllm connector actions")
        _validate_payload_reference(self.actions, payload=self.payload, payload_uri=self.payload_uri)

    @property
    def request_id(self) -> str:
        return self.actions.reservation.request_id

    @property
    def total_tokens(self) -> int:
        return self.actions.reservation.total_tokens


@dataclass(frozen=True, slots=True)
class DocumentKVLoadRequest:
    """Worker metadata for one request whose external KV should be loaded."""

    request_id: str
    actions_record: Mapping[str, Any]
    payload: bytes | tuple[bytes, ...] | None
    blocks: tuple[BlockSpan, ...]
    source_token_start: int
    token_count: int
    payload_uri: str | None = None
    _validated_actions: EngineKVConnectorActions | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        actions_record = _normalized_actions_record(self.actions_record)
        _required_string(self.request_id, field_name="request_id")
        if self.source_token_start < 0:
            raise ValueError("source_token_start must be non-negative")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        object.__setattr__(self, "actions_record", actions_record)

    def validate_method_contract(
        self,
        method_registry: MethodRegistry | None = None,
    ) -> EngineKVConnectorActions:
        actions = engine_kv_connector_actions_from_record(
            self.actions_record,
            expected_backend=ServingBackend.VLLM,
            method_registry=method_registry,
        )
        if self.request_id != actions.reservation.request_id:
            raise ValueError("load request_id must match connector actions")
        if self.source_token_start + self.token_count > actions.reservation.total_tokens:
            raise ValueError("load token span exceeds connector actions")
        _validate_payload_reference(actions, payload=self.payload, payload_uri=self.payload_uri)
        object.__setattr__(self, "_validated_actions", actions)
        return actions

    @property
    def actions(self) -> EngineKVConnectorActions:
        if self._validated_actions is not None:
            return self._validated_actions
        return self.validate_method_contract()

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Exclude the process-local validated action cache from worker metadata."""

        return (
            type(self),
            (
                self.request_id,
                self.actions_record,
                self.payload,
                self.blocks,
                self.source_token_start,
                self.token_count,
                self.payload_uri,
            ),
        )


@dataclass(frozen=True, slots=True)
class _ScheduledRequestBlocks:
    """vLLM scheduler block ids for one request in the current step."""

    block_ids: object
    relative_to_new_tokens: bool = False


@dataclass(frozen=True, slots=True)
class _PayloadTensorView:
    """Token-major CPU view over one materialized Cachet payload."""

    token_major: object
    scalars_per_layer: int
    buffer: bytes | bytearray | memoryview


@dataclass(frozen=True, slots=True)
class _MaterializedPayload:
    """Payload buffer plus the host-side resolution strategy used for it.

    A segmented Cachet handoff still retains its copy descriptors in ``actions``.
    For canonical disk bundles, those descriptors prove that the flat payload file
    is already in global token order. One process-owned global snapshot can then be
    consumed without first recreating segment buffers and merging them back
    together. The owned snapshot also binds checksum validation to the exact bytes
    later copied to the device even if the backing file is concurrently replaced or
    modified.
    """

    payload: bytes | bytearray | memoryview | tuple[bytes, ...]
    configured_segmented_strategy: str
    selected_strategy: str
    payload_mode: PayloadMode
    canonical_segmented_global_view: bool
    legacy_fallback_reason: str | None = None
    checksum_validation_count: int = 0
    snapshot_copy_bytes: int = 0
    reassembly_copy_bytes: int = 0

    def telemetry(self, *, copy_count: int) -> dict[str, object]:
        return {
            "configured_segmented_strategy": self.configured_segmented_strategy,
            "selected_strategy": self.selected_strategy,
            "payload_mode": self.payload_mode.value,
            "canonical_segmented_global_view": self.canonical_segmented_global_view,
            "legacy_fallback_reason": self.legacy_fallback_reason,
            "checksum_validation_count": self.checksum_validation_count,
            "snapshot_copy_bytes": self.snapshot_copy_bytes,
            "reassembly_copy_bytes": self.reassembly_copy_bytes,
            "copy_metadata_retained": True,
            "copy_count": copy_count,
        }


_LoadIdentity = tuple[str, int, int, tuple[tuple[int, int, int, int], ...]]


@dataclass(frozen=True, slots=True)
class _PayloadCacheRead:
    payload: bytes
    hit: bool


class _PayloadReader(Protocol):
    def __call__(
        self,
        payload_uri: str,
        *,
        expected_bytes: int,
        actions: EngineKVConnectorActions,
    ) -> bytes | memoryview: ...


class _PayloadCache:
    def __init__(self, *, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._lru = ByteLRU(max_bytes)

    def read(
        self,
        payload_uri: str,
        *,
        expected_bytes: int,
        cache_identity: str,
    ) -> _PayloadCacheRead:
        key = self._cache_key(
            payload_uri,
            expected_bytes=expected_bytes,
            cache_identity=cache_identity,
        )
        cached = self._lru.get(key)
        if cached is not None:
            return _PayloadCacheRead(payload=cached, hit=True)
        payload = read_engine_adapter_payload(payload_uri, expected_bytes=expected_bytes)
        self._lru.put(key, payload)
        return _PayloadCacheRead(payload=payload, hit=False)

    def reset(self) -> None:
        self._lru = ByteLRU(self._max_bytes)

    @staticmethod
    def _cache_key(payload_uri: str, *, expected_bytes: int, cache_identity: str) -> str:
        return f"{payload_uri}\n{expected_bytes}\n{cache_identity}"


@dataclass(frozen=True, slots=True)
class DocumentKVConnectorMetadata(_KVConnectorMetadata):
    """Scheduler-to-worker metadata consumed by :class:`DocumentKVNativeProvider`."""

    loads: tuple[DocumentKVLoadRequest, ...] = ()


class KVTransferParamsDocumentKVSource:
    """Load Cachet handoff records referenced by vLLM ``kv_transfer_params``.

    Supported request parameters:
    - ``document_kv.benchmark_request_id``: exact benchmark correlation identity.
    - ``document_kv.handoff_json``: path to a Cachet engine adapter handoff JSON.
    - ``document_kv.handoff_record``: already-decoded handoff record mapping.
    - ``document_kv.payload_uri``: optional payload URI override.
    """

    def __init__(
        self,
        *,
        adapter_spec: EngineAdapterSpec | None = None,
        operation_handlers: RuntimeOperationHandlerRegistry | None = None,
        method_registry: MethodRegistry | None = None,
    ) -> None:
        self.adapter_spec = adapter_spec or vllm_adapter_spec()
        if not isinstance(self.adapter_spec, EngineAdapterSpec):
            raise TypeError("adapter_spec must be an EngineAdapterSpec")
        if self.adapter_spec.backend != ServingBackend.VLLM:
            raise ValueError("vLLM handoff sources require a vllm adapter spec")
        self.operation_handlers = (
            RuntimeOperationHandlerRegistry()
            if operation_handlers is None
            else operation_handlers
        )
        if not isinstance(
            self.operation_handlers,
            RuntimeOperationHandlerRegistry,
        ):
            raise TypeError(
                "operation_handlers must be a RuntimeOperationHandlerRegistry"
            )
        self.method_registry = _method_registry(method_registry)

    def get_load(self, request: object) -> DocumentKVHandoffLoad | None:
        params = getattr(request, "kv_transfer_params", None)
        if not isinstance(params, Mapping):
            return None

        payload_uri_override = _optional_string(params.get(DOCUMENT_KV_PAYLOAD_URI_PARAM))
        handoff_record = params.get(DOCUMENT_KV_HANDOFF_RECORD_PARAM)
        handoff_json = params.get(DOCUMENT_KV_HANDOFF_JSON_PARAM)
        if handoff_record is None and handoff_json is None:
            return None
        if handoff_record is not None and handoff_json is not None:
            raise ValueError(
                f"Use only one of {DOCUMENT_KV_HANDOFF_RECORD_PARAM} or {DOCUMENT_KV_HANDOFF_JSON_PARAM}"
            )
        if handoff_record is not None:
            if not isinstance(handoff_record, Mapping):
                raise TypeError(f"{DOCUMENT_KV_HANDOFF_RECORD_PARAM} must be a mapping")
            record = handoff_record
        else:
            handoff_path = _required_string(handoff_json, field_name=DOCUMENT_KV_HANDOFF_JSON_PARAM)
            record = read_engine_adapter_request_json(
                handoff_path,
                expected_backend=ServingBackend.VLLM,
                require_external_payload_uri=payload_uri_override is None,
                adapter_spec=self.adapter_spec,
                operation_handlers=self.operation_handlers,
                method_registry=self.method_registry,
            )

        handoff_request_id = _handoff_request_id(params, record)
        runtime_request_id = getattr(request, "request_id", None)
        if (
            handoff_request_id is None
            and isinstance(runtime_request_id, str)
            and record.get("request_id") != runtime_request_id
        ):
            raise ValueError("document KV handoff request_id does not match vLLM request_id")

        plan = build_engine_kv_injection_plan(
            record,
            expected_backend=ServingBackend.VLLM,
            require_external_payload_uri=payload_uri_override is None,
            adapter_spec=self.adapter_spec,
            operation_handlers=self.operation_handlers,
            method_registry=self.method_registry,
        )
        payload_uri = payload_uri_override or plan.payload_source_uri
        if payload_uri is None:
            raise ValueError("document KV handoff requires an external payload URI")
        actions = _connector_actions_from_plan(
            plan,
            method_registry=self.method_registry,
        )
        actions = _actions_with_benchmark_request_id(
            actions,
            _benchmark_request_id(params),
            method_registry=self.method_registry,
        )
        return DocumentKVHandoffLoad(
            actions=actions,
            payload_uri=payload_uri,
            method_registry=self.method_registry,
        )


class DocumentKVNativeProvider:
    """Synchronous vLLM V1 provider that imports Cachet payloads into paged KV.

    This provider uses vLLM's native connector lifecycle: the scheduler claims
    external matched tokens, records the allocated physical blocks in connector
    metadata, then workers copy materialized Cachet payload bytes into their
    registered paged KV cache tensors before attention executes.
    """

    document_kv_native_provider = True

    def __init__(
        self,
        *,
        source: DocumentKVHandoffSource | None = None,
        provider_factory: str = DOCUMENT_KV_NATIVE_PROVIDER_FACTORY,
        payload_cache_max_bytes: int = 0,
        telemetry_jsonl: str | None = None,
        runtime_identity: RuntimeIdentity | None = None,
        require_runtime_handshake: bool = False,
        adapter_spec: EngineAdapterSpec | None = None,
        operation_handlers: RuntimeOperationHandlerRegistry | None = None,
        method_registry: MethodRegistry | None = None,
    ) -> None:
        payload_cache_max_bytes = _non_negative_int(
            payload_cache_max_bytes,
            field_name="payload_cache_max_bytes",
        )
        telemetry_jsonl = _optional_config_path(
            telemetry_jsonl,
            field_name="telemetry_jsonl",
        )
        if runtime_identity is not None and not isinstance(runtime_identity, RuntimeIdentity):
            raise TypeError("runtime_identity must be a RuntimeIdentity or None")
        if type(require_runtime_handshake) is not bool:
            raise TypeError("require_runtime_handshake must be a boolean")
        self.adapter_spec = adapter_spec or vllm_adapter_spec()
        if not isinstance(self.adapter_spec, EngineAdapterSpec):
            raise TypeError("adapter_spec must be an EngineAdapterSpec")
        if self.adapter_spec.backend != ServingBackend.VLLM:
            raise ValueError("DocumentKVNativeProvider requires a vllm adapter spec")
        self.operation_handlers = (
            RuntimeOperationHandlerRegistry()
            if operation_handlers is None
            else operation_handlers
        )
        if not isinstance(
            self.operation_handlers,
            RuntimeOperationHandlerRegistry,
        ):
            raise TypeError(
                "operation_handlers must be a RuntimeOperationHandlerRegistry"
            )
        self.method_registry = _method_registry(method_registry)
        self.source = source or KVTransferParamsDocumentKVSource(
            adapter_spec=self.adapter_spec,
            operation_handlers=self.operation_handlers,
            method_registry=self.method_registry,
        )
        self.provider_factory = _provider_factory_path(provider_factory)
        self.telemetry_jsonl = telemetry_jsonl
        self.runtime_identity = runtime_identity
        self.require_runtime_handshake = require_runtime_handshake
        # When enabled (DOCUMENT_KV_PROFILE_STAGES=1), the per-load layer loop times
        # the host->device copy and the on-GPU scatter separately, inserting CUDA
        # synchronizations so the wall time is attributed to the correct stage. This
        # adds synchronization overhead, so it is off by default and only used for
        # dedicated profiling runs.
        self._profile_stages = _env_truthy("DOCUMENT_KV_PROFILE_STAGES")
        # Canonical Cachet segmented bundles are serialized as one flat file in
        # global token order. ``auto`` (the default) takes one owned global
        # snapshot and falls back to the historical segment/remerge path for
        # third-party action layouts. ``direct`` is the fail-closed experiment
        # mode: it rejects any action layout that cannot prove the global-order
        # invariant. ``legacy`` always exercises the old reconstruction path for
        # apples-to-apples profiling.
        self._segmented_load_strategy = _segmented_load_strategy_from_env()
        # Cold-read enforcement (opt-in): drop the payload file from the OS page cache
        # (posix_fadvise POSIX_FADV_DONTNEED) immediately before memory-mapping it, so
        # the owned-snapshot read (or a retained merged mmap's device copy) faults
        # pages from NVMe instead of RAM.
        # Handoff generation writes the payload files on the same box right before
        # serving, which leaves them warm in the page cache; without eviction the
        # "cold_disk_to_gpu_hydrate" protocol actually measures warm page-cache reads.
        # Enable with DOCUMENT_KV_EVICT_PAGE_CACHE=1 to measure honest cold-disk hydrate.
        self._evict_page_cache = _env_truthy("DOCUMENT_KV_EVICT_PAGE_CACHE")
        # posix_fadvise(DONTNEED) can only drop *clean* pages, and freshly generated
        # payloads still have dirty (not-yet-written-back) pages, so plain eviction
        # leaves the read partially warm. A one-time global flush before the first
        # eviction writes those pages back to disk so every subsequent read is fully
        # cold. The flush cost lands only on the first load.
        self._page_cache_synced = False
        # Concurrent prefetch (opt-in): the vLLM scheduler calls ``start_load_kv``
        # once and the connector hydrates each pending load serially, so at cold-disk
        # the per-load NVMe read (~0.66 s each) fully serializes (max load
        # concurrency = 1) and dominates TTFT. A small background thread pool reads
        # the payload files into the OS page cache concurrently (I/O releases the
        # GIL, so reads overlap up to the ~3 GB/s device aggregate) as soon as the
        # step's loads are known, so the on-critical-path host->device copy streams
        # from warm cache instead of blocking on disk. Purely a cache-warming hint:
        # correctness is unaffected if a prefetch has not finished (the copy simply
        # faults the remaining pages). Set DOCUMENT_KV_PREFETCH_WORKERS=N (N>0).
        self._prefetch_workers = _env_int("DOCUMENT_KV_PREFETCH_WORKERS", 0)
        # Cap on *concurrent* NVMe reads, independent of how many prefetches are
        # queued. Profiling showed the connector reads at ~1.86 GB/s single-stream
        # and the device peaks at ~3.06 GB/s with 4 O_DIRECT streams but *drops* to
        # ~2.41 GB/s at 8 (over-subscription). vLLM admits requests in waves, so a
        # whole wave's payloads get submitted at once; without a cap all 8 hammer the
        # disk together (~0.39 GB/s each, multi-second stalls -> fat P95 tail). Bound
        # concurrent reads to the sweet spot and let the rest queue (deeper pipeline,
        # bounded contention). Buffered reads (not O_DIRECT) preserve the OS page
        # cache so repeated/hot documents still hit RAM.
        self._prefetch_max_inflight = max(1, _env_int("DOCUMENT_KV_PREFETCH_MAX_INFLIGHT", 4))
        self._prefetch_pool: object | None = None
        self._prefetch_futures: dict[str, object] = {}
        self._stats_prefetch_submitted = 0
        self._payload_cache = (
            None if payload_cache_max_bytes == 0 else _PayloadCache(max_bytes=payload_cache_max_bytes)
        )
        self._loads: dict[str, DocumentKVHandoffLoad] = {}
        self._allocated: dict[str, DocumentKVLoadRequest] = {}
        self._active_external_request_ids: set[str] = set()
        self._loaded_load_identities: dict[str, set[_LoadIdentity]] = {}
        self._metadata = DocumentKVConnectorMetadata()
        self._kv_caches: dict[str, object] = {}
        self._layer_indices: dict[str, int] = {}
        self._layer_mapping_inspection = DocumentKVVLLMLayerMappingInspection((), {})
        self._load_errors: set[int] = set()
        self._events: list[dict[str, object]] = []
        self._stats_loads_started = 0
        self._stats_layers_loaded = 0
        self._stats_load_error_blocks = 0
        self._stats_payload_materialize_ns = 0
        self._stats_payload_merge_ns = 0
        self._stats_payload_view_ns = 0
        self._stats_layer_load_ns = 0
        self._stats_payload_cache_hits = 0
        self._stats_payload_cache_misses = 0
        self._cache_state_observations: dict[str, dict[str, object]] = {}

    def get_num_new_matched_tokens(
        self,
        request: object,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if num_computed_tokens < 0:
            raise ValueError("num_computed_tokens must be non-negative")
        request_id = _request_id(request)
        if request_id in self._allocated or request_id in self._active_external_request_ids:
            return 0, False
        load = self._load_for_request(request)
        if load is None:
            self._loads.pop(request_id, None)
            return 0, False
        block_size = load.actions.reservation.layout.block_size
        available_tokens = _matchable_prefix_tokens(load, request)
        if num_computed_tokens % block_size != 0:
            return 0, False
        if available_tokens <= num_computed_tokens:
            self._loads.pop(request_id, None)
            return 0, False

        return available_tokens - num_computed_tokens, False

    def update_state_after_alloc(self, request: object, blocks: object, num_external_tokens: int) -> None:
        request_id = _request_id(request)
        if num_external_tokens <= 0:
            self._allocated.pop(request_id, None)
            return
        load = self._load_for_request(request)
        if load is None:
            raise ValueError("document KV allocation received without a matched load")
        block_size = load.actions.reservation.layout.block_size
        available_tokens = _matchable_prefix_tokens(load, request)
        if num_external_tokens > available_tokens:
            raise ValueError("num_external_tokens exceeds the available document KV token count")
        source_token_start = available_tokens - num_external_tokens
        if num_external_tokens % block_size != 0:
            raise ValueError(
                "num_external_tokens must be block-aligned for document "
                "KV loads"
            )
        if source_token_start % block_size != 0:
            raise ValueError(
                "document KV load source_token_start must be block-aligned"
            )

        block_spans = _block_spans_for_token_range(
            blocks,
            block_size=block_size,
            source_token_start=source_token_start,
            token_count=num_external_tokens,
        )
        runtime_actions = _connector_actions_for_runtime_request(
            load.actions,
            request_id,
            method_registry=self.method_registry,
        )
        allocated = DocumentKVLoadRequest(
            request_id=request_id,
            actions_record=engine_kv_connector_actions_to_record(
                runtime_actions,
                method_registry=self.method_registry,
            ),
            payload=load.payload,
            blocks=block_spans,
            source_token_start=source_token_start,
            token_count=num_external_tokens,
            payload_uri=load.payload_uri,
        )
        allocated.validate_method_contract(self.method_registry)
        self._allocated[request_id] = allocated

    def build_connector_meta(self, scheduler_output: object) -> DocumentKVConnectorMetadata:
        loads: list[DocumentKVLoadRequest] = []
        scheduled_block_ids = _scheduled_request_block_ids(scheduler_output)
        missing_request_ids: list[str] = []
        for request_id, allocated in self._allocated.items():
            scheduled_blocks = scheduled_block_ids.get(request_id)
            if scheduled_blocks is None:
                missing_request_ids.append(request_id)
                continue
            source_token_start = 0 if scheduled_blocks.relative_to_new_tokens else allocated.source_token_start
            blocks = _block_spans_for_token_range(
                scheduled_blocks.block_ids,
                block_size=allocated.actions.reservation.layout.block_size,
                source_token_start=source_token_start,
                token_count=allocated.token_count,
            )
            scheduled_load = DocumentKVLoadRequest(
                request_id=allocated.request_id,
                actions_record=allocated.actions_record,
                payload=allocated.payload,
                blocks=blocks,
                source_token_start=allocated.source_token_start,
                token_count=allocated.token_count,
                payload_uri=allocated.payload_uri,
            )
            scheduled_load.validate_method_contract(self.method_registry)
            loads.append(scheduled_load)
        if missing_request_ids:
            raise ValueError(
                "Document KV allocation is missing scheduled vLLM block ids for request(s): "
                + ", ".join(sorted(missing_request_ids))
            )
        for load in loads:
            self._allocated.pop(load.request_id, None)
            self._loads.pop(load.request_id, None)
            self._active_external_request_ids.add(load.request_id)
        return DocumentKVConnectorMetadata(loads=tuple(loads))

    def bind_connector_metadata(self, connector_metadata: object) -> None:
        if not isinstance(connector_metadata, DocumentKVConnectorMetadata):
            raise TypeError("DocumentKVNativeProvider requires DocumentKVConnectorMetadata")
        for load in connector_metadata.loads:
            load.validate_method_contract(self.method_registry)
        self._metadata = connector_metadata
        self._submit_prefetch(connector_metadata.loads)

    def clear_connector_metadata(self) -> None:
        self._metadata = DocumentKVConnectorMetadata()

    def register_kv_caches(self, kv_caches: Mapping[str, object]) -> None:
        inspection = inspect_document_kv_vllm_layer_mapping(kv_caches)
        layer_indices = _vllm_layer_indices_from_inspection(inspection)
        self._kv_caches = dict(kv_caches)
        self._layer_indices = layer_indices
        self._layer_mapping_inspection = inspection

    def start_load_kv(self, forward_context: object, **kwargs: object) -> None:
        del forward_context, kwargs
        loads = self._metadata.loads
        if not loads:
            return
        if not self._kv_caches:
            raise ValueError("document KV provider has no registered vLLM KV caches")
        for index, load in enumerate(loads):
            load_identity = _load_identity(load)
            loaded_identities = self._loaded_load_identities.setdefault(load.request_id, set())
            if load_identity in loaded_identities:
                self._metadata = DocumentKVConnectorMetadata(loads=loads[index + 1 :])
                continue
            self._load_request(load)
            loaded_identities.add(load_identity)
            self._stats_loads_started += 1
            self._events.append({"event": "document_kv_loaded", "request_id": load.request_id})
            self._metadata = DocumentKVConnectorMetadata(loads=loads[index + 1 :])

    def wait_for_layer_load(self, layer_name: str) -> None:
        if layer_name not in self._kv_caches:
            raise ValueError(f"Unknown vLLM KV cache layer {layer_name!r}")

    def save_kv_layer(self, layer_name: str, kv_layer: object, attn_metadata: object, **kwargs: object) -> None:
        del layer_name, kv_layer, attn_metadata, kwargs
        return None

    def wait_for_save(self) -> None:
        return None

    def request_finished(self, request: object, block_ids: list[int]) -> tuple[bool, Mapping[str, Any] | None]:
        del block_ids
        self._release_request(_request_id(request))
        return False, None

    def request_finished_all_groups(
        self,
        request: object,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Mapping[str, Any] | None]:
        del block_ids
        self._release_request(_request_id(request))
        return False, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set(self._load_errors)

    def get_kv_connector_stats(self) -> DocumentKVConnectorStats | None:
        if (
            self._stats_loads_started == 0
            and self._stats_layers_loaded == 0
            and self._stats_load_error_blocks == 0
            and self._stats_payload_materialize_ns == 0
            and self._stats_payload_merge_ns == 0
            and self._stats_payload_view_ns == 0
            and self._stats_layer_load_ns == 0
            and self._stats_payload_cache_hits == 0
            and self._stats_payload_cache_misses == 0
        ):
            return None
        stats = DocumentKVConnectorStats.from_mapping(
            {
                "document_kv_loads_started": self._stats_loads_started,
                "document_kv_layers_loaded": self._stats_layers_loaded,
                "document_kv_load_error_blocks": self._stats_load_error_blocks,
                "document_kv_payload_materialize_ns": self._stats_payload_materialize_ns,
                "document_kv_payload_merge_ns": self._stats_payload_merge_ns,
                "document_kv_payload_view_ns": self._stats_payload_view_ns,
                "document_kv_layer_load_ns": self._stats_layer_load_ns,
                "document_kv_payload_cache_hits": self._stats_payload_cache_hits,
                "document_kv_payload_cache_misses": self._stats_payload_cache_misses,
            }
        )
        self._stats_loads_started = 0
        self._stats_layers_loaded = 0
        self._stats_load_error_blocks = 0
        self._stats_payload_materialize_ns = 0
        self._stats_payload_merge_ns = 0
        self._stats_payload_view_ns = 0
        self._stats_layer_load_ns = 0
        self._stats_payload_cache_hits = 0
        self._stats_payload_cache_misses = 0
        return stats

    def vllm_layer_mapping_record(self) -> dict[str, Any]:
        """Return the last vLLM layer-name mapping accepted by the provider."""

        return document_kv_vllm_layer_mapping_to_record(self._layer_mapping_inspection)

    def set_document_kv_provider_factory(self, provider_factory: str) -> None:
        self.provider_factory = _provider_factory_path(provider_factory)

    def get_handshake_metadata(self) -> Mapping[str, Any]:
        """Expose the strict runtime preflight record via vLLM handshake hooks."""

        from vllm_kv_injection.vllm_runtime_preflight import (
            document_kv_vllm_runtime_preflight_to_record,
        )

        return document_kv_vllm_runtime_preflight_to_record(
            self._layer_mapping_inspection,
            provider_factory=self.provider_factory,
        )

    def take_events(self) -> list[Mapping[str, object]]:
        events = list(self._events)
        self._events.clear()
        return events

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        for request_id in finished_req_ids:
            self._release_request(request_id)
        return None, None

    def _load_request(self, load: DocumentKVLoadRequest) -> None:
        load_started_ns = time.perf_counter_ns()
        wall_start_s = time.time()
        payload_materialize_ns = 0
        payload_merge_ns = 0
        payload_view_ns = 0
        layer_load_ns = 0
        h2d_ns = 0
        scatter_ns = 0
        layers_loaded = 0
        decoded_runtime_bytes = 0
        profile_stages = self._profile_stages
        cache_hits_before = self._stats_payload_cache_hits
        cache_misses_before = self._stats_payload_cache_misses
        payload_loading: Mapping[str, object] = {
            "configured_segmented_strategy": self._segmented_load_strategy,
            "selected_strategy": "unresolved",
            "payload_mode": "unknown",
            "canonical_segmented_global_view": False,
            "legacy_fallback_reason": None,
            "checksum_validation_count": 0,
            "snapshot_copy_bytes": 0,
            "reassembly_copy_bytes": 0,
            "copy_metadata_retained": True,
            "copy_count": 0,
        }
        error_type: str | None = None
        error_message: str | None = None
        try:
            layout = load.actions.reservation.layout
            block_size = layout.block_size
            started_ns = time.perf_counter_ns()
            try:
                payload = _materialized_payload(
                    load,
                    payload_reader=self._read_payload,
                    segmented_load_strategy=self._segmented_load_strategy,
                )
            finally:
                payload_materialize_ns = time.perf_counter_ns() - started_ns
                self._stats_payload_materialize_ns += payload_materialize_ns

            started_ns = time.perf_counter_ns()
            try:
                resolved_payload = _merged_payload(load.actions, payload)
                payload_loading = resolved_payload.telemetry(
                    copy_count=len(load.actions.copies),
                )
                merged_payload = resolved_payload.payload
                if isinstance(merged_payload, tuple):
                    raise TypeError("resolved document KV payload must be one flat buffer")
                assert load.actions.reuse_plan is not None
                reuse_plan = load.actions.reuse_plan
                if reuse_plan.runtime_operations:
                    # Method-owned transforms operate on immutable bytes. Only
                    # materialize here when a declared decoder/selector/recomputer
                    # actually needs them; raw KV preserves the resolved global
                    # buffer so there is no additional method-transform copy.
                    operation_result = apply_runtime_operation_handlers(
                        reuse_plan,
                        bytes(merged_payload),
                        layout=layout,
                        total_tokens=load.actions.reservation.total_tokens,
                        handler_registry=self.operation_handlers,
                        metadata=load.actions.bind.metadata,
                        runtime_context=load,
                    )
                    assert operation_result.payload is not None
                    merged_payload = operation_result.payload
                else:
                    reuse_plan.validate_runtime_layout(layout)
                decoded_runtime_bytes = len(merged_payload)
            finally:
                payload_merge_ns = time.perf_counter_ns() - started_ns
                self._stats_payload_merge_ns += payload_merge_ns

            started_ns = time.perf_counter_ns()
            try:
                payload_view = _payload_tensor_view(merged_payload, load)
            finally:
                payload_view_ns = time.perf_counter_ns() - started_ns
                self._stats_payload_view_ns += payload_view_ns

            scalars_per_layer = payload_view.scalars_per_layer
            # Slice the loaded token range once (a contiguous CPU view over the
            # materialized payload); the per-layer reshape happens on the device so
            # the transfer is a single contiguous host->device copy.
            cpu_token_slice = payload_view.token_major[
                load.source_token_start : load.source_token_start + load.token_count
            ]
            h2d_source = cpu_token_slice
            # Position-independent payloads use absolute positions. cos/sin are
            # shared across layers on each device.
            key_position_encoding = getattr(
                layout,
                "key_position_encoding",
                (
                    KVKeyPositionEncoding.PRE_ROPE
                    if bool(getattr(layout, "pre_rope", False))
                    else KVKeyPositionEncoding.STORED_POST_ROPE
                ),
            )
            key_position_encoding = KVKeyPositionEncoding(
                getattr(key_position_encoding, "value", key_position_encoding)
            )
            reposition_keys = key_position_encoding == KVKeyPositionEncoding.PRE_ROPE
            rope_theta = getattr(layout, "rope_theta", None)
            rope_rotary_dim = getattr(layout, "rope_rotary_dim", None)
            rope_cos_sin_by_device: dict[object | None, tuple[object, object]] = {}
            slot_mappings: dict[object | None, object] = {}
            device_token_slices: dict[object | None, object] = {}
            for layer_name, dst_layer in self._kv_caches.items():
                layer_index = self._layer_indices[layer_name]
                if layer_index >= layout.num_layers:
                    continue
                started_ns = time.perf_counter_ns()
                try:
                    device = getattr(dst_layer, "device", None)
                    if device not in device_token_slices:
                        if profile_stages:
                            h2d_started_ns = time.perf_counter_ns()
                            device_token_slices[device] = _to_device_contiguous(h2d_source, device)
                            slot_mappings[device] = slot_mapping_from_blocks(
                                load.blocks,
                                block_size=block_size,
                                device=device,
                            )
                            _maybe_cuda_sync(device)
                            h2d_ns += time.perf_counter_ns() - h2d_started_ns
                        else:
                            device_token_slices[device] = _to_device_contiguous(h2d_source, device)
                            slot_mappings[device] = slot_mapping_from_blocks(
                                load.blocks,
                                block_size=block_size,
                                device=device,
                            )
                    if reposition_keys and device not in rope_cos_sin_by_device:
                        rope_cos_sin_by_device[device] = _rope_cos_sin_for_load(
                            load,
                            dst_layer,
                            rope_theta=rope_theta,
                            rotary_dim=rope_rotary_dim,
                            key_position_encoding=key_position_encoding,
                        )
                    src_layer = _layer_values_from_token_slice(
                        device_token_slices[device],
                        scalars_per_layer,
                        layer_index=layer_index,
                        dst_kv_cache_layer=dst_layer,
                        layout=layout,
                    )
                    if reposition_keys:
                        cos, sin = rope_cos_sin_by_device[device]
                        src_layer = _rerope_src_layer_keys(
                            src_layer,
                            cos=cos,
                            sin=sin,
                            rope_theta=rope_theta,
                            rotary_dim=rope_rotary_dim,
                            payload_dtype=layout.dtype,
                        )
                    if profile_stages:
                        scatter_started_ns = time.perf_counter_ns()
                        inject_kv_cache_layer(
                            dst_layer,
                            src_layer,
                            slot_mappings[device],
                            block_size=block_size,
                        )
                        _maybe_cuda_sync(device)
                        scatter_ns += time.perf_counter_ns() - scatter_started_ns
                    else:
                        inject_kv_cache_layer(
                            dst_layer,
                            src_layer,
                            slot_mappings[device],
                            block_size=block_size,
                        )
                finally:
                    elapsed_ns = time.perf_counter_ns() - started_ns
                    layer_load_ns += elapsed_ns
                    self._stats_layer_load_ns += elapsed_ns
                self._stats_layers_loaded += 1
                layers_loaded += 1
        except Exception as exc:
            error_type = type(exc).__name__
            error_message = _truncated_error_message(exc)
            error_block_ids = {block.block_id for block in load.blocks}
            self._load_errors.update(error_block_ids)
            self._stats_load_error_blocks += len(error_block_ids)
            raise
        finally:
            self._write_load_telemetry(
                load,
                total_ns=time.perf_counter_ns() - load_started_ns,
                payload_materialize_ns=payload_materialize_ns,
                payload_merge_ns=payload_merge_ns,
                payload_view_ns=payload_view_ns,
                layer_load_ns=layer_load_ns,
                h2d_ns=h2d_ns if profile_stages else None,
                scatter_ns=scatter_ns if profile_stages else None,
                wall_start_s=wall_start_s,
                wall_end_s=time.time(),
                layers_loaded=layers_loaded,
                decoded_runtime_bytes=decoded_runtime_bytes,
                payload_cache_hits=self._stats_payload_cache_hits - cache_hits_before,
                payload_cache_misses=self._stats_payload_cache_misses - cache_misses_before,
                payload_loading=payload_loading,
                error_type=error_type,
                error_message=error_message,
            )

    def _read_payload(
        self,
        payload_uri: str,
        *,
        expected_bytes: int,
        actions: EngineKVConnectorActions,
    ) -> bytes | memoryview:
        if self._payload_cache is None:
            # No in-process payload cache (the cold-hydrate measurement path):
            # memory-map the payload first. Merged loads may retain the lazy view;
            # canonical segmented loads immediately take one owned global snapshot
            # so their checksum binds the bytes later copied to the device. A
            # concurrent prefetch (``_prefetch_payload_uri``) may already have
            # re-warmed the pages. When concurrent prefetch is active the
            # on-critical-path map must not re-evict those warm pages; otherwise
            # apply per-read eviction here for the honest cold-hydrate measurement.
            evict_here = self._evict_page_cache and self._prefetch_workers <= 0
            if evict_here and not self._page_cache_synced:
                # Flush dirty (freshly written) payload pages to disk once so the
                # per-file DONTNEED eviction below can drop every page and reads are
                # measured fully cold instead of partially warm.
                os.sync()
                self._page_cache_synced = True
            self._reap_prefetch(payload_uri)
            eviction_succeeded = False

            def mark_evicted() -> None:
                nonlocal eviction_succeeded
                eviction_succeeded = True

            payload = _mmap_payload_view(
                payload_uri,
                expected_bytes=expected_bytes,
                evict_page_cache=evict_here,
                on_page_cache_evicted=mark_evicted,
            )
            self._cache_state_observations[actions.reservation.request_id] = {
                "source": _payload_source_name(payload_uri),
                "bytes_read": expected_bytes,
                "payload_cache_hit": False,
                "eviction_requested": self._evict_page_cache,
                "eviction_succeeded": eviction_succeeded,
                "direct_io": False,
            }
            return payload
        result = self._payload_cache.read(
            payload_uri,
            expected_bytes=expected_bytes,
            cache_identity=_payload_cache_identity(actions),
        )
        if result.hit:
            self._stats_payload_cache_hits += 1
        else:
            self._stats_payload_cache_misses += 1
        self._cache_state_observations[actions.reservation.request_id] = {
            "source": _payload_source_name(payload_uri),
            "bytes_read": 0 if result.hit else expected_bytes,
            "payload_cache_hit": result.hit,
            "eviction_requested": False,
            "eviction_succeeded": False,
            "direct_io": False,
        }
        return result.payload

    def _ensure_prefetch_pool(self) -> object:
        pool = self._prefetch_pool
        if pool is None:
            # Concurrent reads are bounded to the disk sweet spot; a whole wave of
            # prefetches can be submitted at once and the surplus queues FIFO on the
            # executor (deep pipeline, bounded contention).
            concurrency = min(max(1, self._prefetch_workers), self._prefetch_max_inflight)
            pool = ThreadPoolExecutor(
                max_workers=concurrency,
                thread_name_prefix="dockv-prefetch",
            )
            self._prefetch_pool = pool
        return pool

    def _submit_prefetch(self, loads: "tuple[DocumentKVLoadRequest, ...]") -> None:
        # Only the mmap cold-hydrate path (no in-process payload cache) reads from
        # disk on the critical path, so prefetch is pointless when a payload cache
        # serves hydrates from memory.
        if self._prefetch_workers <= 0 or self._payload_cache is not None or not loads:
            return
        pool = self._ensure_prefetch_pool()
        for load in loads:
            payload_uri = load.payload_uri
            if payload_uri is None or payload_uri in self._prefetch_futures:
                continue
            self._prefetch_futures[payload_uri] = pool.submit(
                self._prefetch_payload_uri, payload_uri, load
            )
            self._stats_prefetch_submitted += 1

    def _reap_prefetch(self, payload_uri: str) -> None:
        future = self._prefetch_futures.pop(payload_uri, None)
        if future is None:
            return
        # Block until the background (cold) read completes, then let the mmap copy
        # stream from warm cache. Under concurrency this read already finished during
        # the request's queue wait, so the wait is ~free and the copy is warm. A
        # non-blocking variant was measured *worse*: skipping lets the background
        # read contend with the copy's own page faults for disk bandwidth. The
        # timeout is only a safety net so a wedged prefetch can never stall the
        # engine's load path indefinitely.
        try:
            future.result(timeout=_PREFETCH_WAIT_TIMEOUT_S)
        except Exception:
            # Prefetch is a best-effort cache-warming hint; on failure/timeout the
            # mmap host->device copy below still faults the pages correctly.
            pass

    def _prefetch_payload_uri(
        self, payload_uri: str, load: DocumentKVLoadRequest | None = None
    ) -> None:
        del load
        try:
            path = local_path(payload_uri)
        except Exception:
            return
        wall_start_s = time.time()
        started_ns = time.perf_counter_ns()
        evict_ns = 0
        read_ns = 0
        read_bytes = 0
        try:
            if self._evict_page_cache:
                if not self._page_cache_synced:
                    try:
                        os.sync()
                    except OSError:
                        pass
                    self._page_cache_synced = True
                evict_started_ns = time.perf_counter_ns()
                fd = os.open(path, os.O_RDONLY)
                try:
                    _evict_file_from_page_cache(fd)
                finally:
                    os.close(fd)
                evict_ns = time.perf_counter_ns() - evict_started_ns
            buffer = bytearray(_PREFETCH_CHUNK_BYTES)
            view = memoryview(buffer)
            read_started_ns = time.perf_counter_ns()
            with open(path, "rb", buffering=0) as handle:
                while True:
                    chunk = handle.readinto(view)
                    if not chunk:
                        break
                    read_bytes += chunk
            read_ns = time.perf_counter_ns() - read_started_ns
        except OSError:
            pass
        finally:
            self._write_prefetch_telemetry(
                payload_uri,
                read_bytes=read_bytes,
                read_ns=read_ns,
                evict_ns=evict_ns,
                total_ns=time.perf_counter_ns() - started_ns,
                wall_start_s=wall_start_s,
                wall_end_s=time.time(),
            )

    def _write_prefetch_telemetry(
        self,
        payload_uri: str,
        *,
        read_bytes: int,
        read_ns: int,
        evict_ns: int,
        total_ns: int,
        wall_start_s: float,
        wall_end_s: float,
    ) -> None:
        if self.telemetry_jsonl is None:
            return
        try:
            gbps = (read_bytes / 1e9) / (read_ns / 1e9) if read_ns > 0 else 0.0
            _append_jsonl(
                self.telemetry_jsonl,
                {
                    "event": "prefetch_read",
                    "record_type": "document_kv_prefetch",
                    "uri_sha256": hashlib.sha256(payload_uri.encode("utf-8")).hexdigest(),
                    "read_bytes": read_bytes,
                    "timings_ns": {"read": read_ns, "evict": evict_ns, "total": total_ns},
                    "read_gbps": gbps,
                    "prefetch_workers": self._prefetch_workers,
                    "wall_clock": {"start_s": wall_start_s, "end_s": wall_end_s},
                },
            )
        except Exception:
            pass

    def _write_load_telemetry(
        self,
        load: DocumentKVLoadRequest,
        *,
        total_ns: int,
        payload_materialize_ns: int,
        payload_merge_ns: int,
        payload_view_ns: int,
        layer_load_ns: int,
        layers_loaded: int,
        decoded_runtime_bytes: int,
        payload_cache_hits: int,
        payload_cache_misses: int,
        payload_loading: Mapping[str, object],
        error_type: str | None,
        error_message: str | None,
        h2d_ns: int | None = None,
        scatter_ns: int | None = None,
        wall_start_s: float | None = None,
        wall_end_s: float | None = None,
    ) -> None:
        cache_state_observation = self._cache_state_observations.pop(
            load.request_id,
            None,
        )
        if self.telemetry_jsonl is None:
            return
        try:
            record = _load_telemetry_record(
                load,
                provider_factory=self.provider_factory,
                payload_cache_enabled=self._payload_cache is not None,
                page_cache_evicted=self._evict_page_cache,
                cache_state_observation=cache_state_observation,
                total_ns=total_ns,
                payload_materialize_ns=payload_materialize_ns,
                payload_merge_ns=payload_merge_ns,
                payload_view_ns=payload_view_ns,
                layer_load_ns=layer_load_ns,
                h2d_ns=h2d_ns,
                scatter_ns=scatter_ns,
                wall_start_s=wall_start_s,
                wall_end_s=wall_end_s,
                layers_loaded=layers_loaded,
                decoded_runtime_bytes=decoded_runtime_bytes,
                payload_cache_hits=payload_cache_hits,
                payload_cache_misses=payload_cache_misses,
                payload_loading=payload_loading,
                error_type=error_type,
                error_message=error_message,
            )
            _append_jsonl(self.telemetry_jsonl, record)
        except Exception as exc:  # pragma: no cover - defensive runtime diagnostics path.
            warnings.warn(f"Could not write document KV telemetry: {exc}", RuntimeWarning, stacklevel=2)

    def reset_cache(self) -> bool | None:
        if self._payload_cache is None:
            return None
        self._payload_cache.reset()
        return True

    def _release_request(self, request_id: str) -> None:
        self._loads.pop(request_id, None)
        self._allocated.pop(request_id, None)
        self._active_external_request_ids.discard(request_id)
        self._loaded_load_identities.pop(request_id, None)

    def _load_for_request(self, request: object) -> DocumentKVHandoffLoad | None:
        request_id = _request_id(request)
        cached = self._loads.get(request_id)
        if cached is not None:
            return cached
        load = self.source.get_load(request)
        if load is None:
            return None
        validate_engine_kv_connector_actions(
            load.actions,
            method_registry=self.method_registry,
        )
        self._verify_runtime_compatibility(load)
        _verify_request_token_contracts(load.actions, request)
        self._loads[request_id] = load
        return load

    def _verify_runtime_compatibility(self, load: DocumentKVHandoffLoad) -> None:
        assert load.actions.reuse_plan is not None
        self.adapter_spec.validate_reuse_plan(
            load.actions.reuse_plan,
            layout=load.actions.reservation.layout,
            artifact_identity=load.actions.reservation.artifact_identity,
            operation_handlers=self.operation_handlers,
            method_registry=self.method_registry,
        )
        artifact_identity = load.actions.reservation.artifact_identity
        if artifact_identity is None:
            return
        if self.runtime_identity is None:
            if self.require_runtime_handshake:
                raise ValueError(
                    "document KV artifact requires a configured runtime identity handshake"
                )
            return
        RuntimeCompatibilityHandshake.compare(
            artifact_identity,
            self.runtime_identity,
        ).require_compatible()


class _MutableHandoffSource:
    def __init__(self) -> None:
        self._loads: dict[str, DocumentKVHandoffLoad] = {}

    def set_load(self, load: DocumentKVHandoffLoad) -> None:
        self._loads[load.request_id] = load

    def get_load(self, request: object) -> DocumentKVHandoffLoad | None:
        return self._loads.get(_request_id(request))

    def release(self, request_id: str) -> None:
        self._loads.pop(request_id, None)


class DocumentKVNativeProbeConnector(VLLMSupportsHMA):
    """Probe connector backed by the runtime ``DocumentKVNativeProvider`` path."""

    document_kv_native_probe_connector = True

    def __init__(self) -> None:
        self._probe_source = _MutableHandoffSource()
        self.provider = DocumentKVNativeProvider(source=self._probe_source)
        self._reservations: dict[str, tuple[BlockSpan, ...]] = {}
        self._handles: dict[str, object] = {}
        self._next_block_id = 0

    def get_num_new_matched_tokens(
        self,
        request: object,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return self.provider.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: object, blocks: object, num_external_tokens: int) -> None:
        self.provider.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: object) -> object:
        return self.provider.build_connector_meta(scheduler_output)

    def register_kv_caches(self, kv_caches: Mapping[str, object]) -> None:
        self.provider.register_kv_caches(kv_caches)

    def bind_connector_metadata(self, connector_metadata: object) -> None:
        self.provider.bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        self.provider.clear_connector_metadata()

    def start_load_kv(self, forward_context: object, **kwargs: object) -> None:
        self.provider.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        self.provider.wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name: str, kv_layer: object, attn_metadata: object, **kwargs: object) -> None:
        self.provider.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        self.provider.wait_for_save()

    def request_finished(self, request: object, block_ids: list[int]) -> tuple[bool, Mapping[str, Any] | None]:
        return self.provider.request_finished(request, block_ids)

    def request_finished_all_groups(
        self,
        request: object,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Mapping[str, Any] | None]:
        return self.provider.request_finished_all_groups(request, block_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        return self.provider.get_block_ids_with_load_errors()

    def get_kv_connector_stats(self) -> DocumentKVConnectorStats | None:
        return self.provider.get_kv_connector_stats()

    def get_handshake_metadata(self) -> Mapping[str, Any]:
        return self.provider.get_handshake_metadata()

    def take_events(self) -> list[Mapping[str, object]]:
        return self.provider.take_events()

    def reserve(self, handle: object) -> tuple[BlockSpan, ...]:
        validator = getattr(handle, "validate", None)
        if callable(validator):
            validator()
        layout = getattr(handle, "layout", None)
        request_id = _required_string(getattr(handle, "request_id", None), field_name="request_id")
        total_tokens = _positive_int(getattr(handle, "total_tokens", None), field_name="total_tokens")
        block_size = _positive_int(getattr(layout, "block_size", None), field_name="layout.block_size")
        blocks = plan_token_blocks(
            total_tokens=total_tokens,
            block_size=block_size,
            starting_block_id=self._next_block_id,
        )
        self._next_block_id += len(blocks)
        self._reservations[request_id] = blocks
        self._handles[request_id] = handle
        return blocks

    def inject(
        self,
        handle: object,
        blocks: tuple[BlockSpan, ...],
        *,
        payload: bytes | memoryview | tuple[bytes | memoryview, ...] | None = None,
    ) -> None:
        request_id = _required_string(getattr(handle, "request_id", None), field_name="request_id")
        expected_blocks = self._reservations.get(request_id)
        if expected_blocks != tuple(blocks):
            raise ValueError(f"Blocks for {request_id} were not reserved by this connector")
        if payload is None:
            raise ValueError("document KV native probe requires copied payload bytes")
        payload_bytes = _payload_bytes(payload)
        ready_request = EngineReadyRequest(
            handle=handle,
            payload=payload_bytes,
            estimated_gpu_bytes=_nonnegative_int(getattr(handle, "total_bytes", None), field_name="total_bytes"),
        )
        ready_request.validate()
        actions = _probe_actions_from_handle(handle, payload_bytes)
        load = DocumentKVHandoffLoad(actions=actions, payload=payload_bytes)
        self._probe_source.set_load(load)

        request = SimpleNamespace(request_id=request_id, num_tokens=ready_request.handle.total_tokens + 1)
        external_tokens, _ = self.get_num_new_matched_tokens(request, 0)
        if external_tokens <= 0:
            raise ValueError("document KV native probe requires at least one block-aligned prefix token")
        block_ids = [block.block_id for block in expected_blocks]
        self.update_state_after_alloc(request, block_ids, external_tokens)
        metadata = self.build_connector_meta(_probe_scheduler_output(request_id, block_ids))
        self.register_kv_caches(_probe_kv_caches(actions.reservation.layout, block_count=max(block_ids) + 1))
        self.bind_connector_metadata(metadata)
        try:
            self.start_load_kv(SimpleNamespace())
        finally:
            self.clear_connector_metadata()

    def release(self, request_id: str) -> None:
        self._reservations.pop(request_id, None)
        self._handles.pop(request_id, None)
        self._probe_source.release(request_id)


def build_document_kv_provider(
    *,
    vllm_config: object | None,
    extra_config: Mapping[str, Any],
    adapter_spec: EngineAdapterSpec | None = None,
    operation_handlers: RuntimeOperationHandlerRegistry | None = None,
    method_registry: MethodRegistry | None = None,
) -> DocumentKVNativeProvider:
    """Provider factory consumed by ``document_kv.provider_factory``."""

    del vllm_config
    source_factory = extra_config.get(DOCUMENT_KV_HANDOFF_SOURCE_FACTORY_CONFIG_KEY)
    payload_cache_max_bytes = _payload_cache_max_bytes_from_config(extra_config)
    telemetry_jsonl = _telemetry_jsonl_from_config(extra_config)
    runtime_identity = _runtime_identity_from_config(extra_config)
    require_runtime_handshake = _require_runtime_handshake_from_config(extra_config)
    if source_factory is None:
        return DocumentKVNativeProvider(
            payload_cache_max_bytes=payload_cache_max_bytes,
            telemetry_jsonl=telemetry_jsonl,
            runtime_identity=runtime_identity,
            require_runtime_handshake=require_runtime_handshake,
            adapter_spec=adapter_spec,
            operation_handlers=operation_handlers,
            method_registry=method_registry,
        )
    if not isinstance(source_factory, str) or not source_factory.strip():
        raise ValueError(f"{DOCUMENT_KV_HANDOFF_SOURCE_FACTORY_CONFIG_KEY} must be a non-empty module:attribute string")
    source = _load_source_factory(source_factory)()
    _validate_source(source)
    return DocumentKVNativeProvider(
        source=source,
        payload_cache_max_bytes=payload_cache_max_bytes,
        telemetry_jsonl=telemetry_jsonl,
        runtime_identity=runtime_identity,
        require_runtime_handshake=require_runtime_handshake,
        adapter_spec=adapter_spec,
        operation_handlers=operation_handlers,
        method_registry=method_registry,
    )


def _payload_cache_max_bytes_from_config(extra_config: Mapping[str, Any]) -> int:
    value = extra_config.get(DOCUMENT_KV_PAYLOAD_CACHE_MAX_BYTES_CONFIG_KEY, 0)
    return _non_negative_int(value, field_name=DOCUMENT_KV_PAYLOAD_CACHE_MAX_BYTES_CONFIG_KEY)


def _telemetry_jsonl_from_config(extra_config: Mapping[str, Any]) -> str | None:
    value = extra_config.get(DOCUMENT_KV_TELEMETRY_JSONL_CONFIG_KEY)
    return _optional_config_path(value, field_name=DOCUMENT_KV_TELEMETRY_JSONL_CONFIG_KEY)


def _runtime_identity_from_config(
    extra_config: Mapping[str, Any],
) -> RuntimeIdentity | None:
    value = extra_config.get(DOCUMENT_KV_RUNTIME_IDENTITY_CONFIG_KEY)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{DOCUMENT_KV_RUNTIME_IDENTITY_CONFIG_KEY} must be a runtime identity mapping"
        )
    return RuntimeIdentity.from_record(value)


def _require_runtime_handshake_from_config(
    extra_config: Mapping[str, Any],
) -> bool:
    value = extra_config.get(DOCUMENT_KV_REQUIRE_RUNTIME_HANDSHAKE_CONFIG_KEY, False)
    if type(value) is not bool:
        raise TypeError(
            f"{DOCUMENT_KV_REQUIRE_RUNTIME_HANDSHAKE_CONFIG_KEY} must be a boolean"
        )
    return value


def _optional_config_path(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string when provided")
    return value


def _method_registry(registry: MethodRegistry | None) -> MethodRegistry:
    if registry is None:
        return default_method_registry()
    if not isinstance(registry, MethodRegistry):
        raise TypeError("method_registry must be a MethodRegistry or None")
    return registry


def _load_source_factory(factory_path: str) -> object:
    module_name, separator, attribute_name = factory_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("document KV handoff source factory must use module:attribute syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)
    if not callable(factory):
        raise TypeError(f"document KV handoff source factory {factory_path!r} is not callable")
    return factory


def _provider_factory_path(factory_path: str) -> str:
    value = _required_string(factory_path, field_name="provider_factory")
    module_name, separator, attribute_name = value.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("provider_factory must use module:attribute syntax")
    return value


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _validate_source(source: object) -> None:
    if not callable(getattr(source, "get_load", None)):
        raise TypeError("document KV handoff source must provide callable get_load")


def _payload_bytes(payload_or_segments: bytes | memoryview | tuple[bytes | memoryview, ...]) -> bytes | tuple[bytes, ...]:
    if isinstance(payload_or_segments, bytes):
        return payload_or_segments
    if isinstance(payload_or_segments, memoryview):
        return payload_or_segments.tobytes()
    return tuple(bytes(segment) for segment in payload_or_segments)


def _validate_payload_reference(
    actions: EngineKVConnectorActions,
    *,
    payload: bytes | tuple[bytes, ...] | None,
    payload_uri: str | None,
) -> None:
    if payload is None and payload_uri is None:
        raise ValueError("document KV load requires payload bytes or payload_uri")
    if payload is not None and payload_uri is not None:
        raise ValueError("document KV load must use only one of payload bytes or payload_uri")
    if payload is not None:
        _validate_payload_matches_actions(actions, payload)
        return
    _required_string(payload_uri, field_name="payload_uri")


def _validate_payload_matches_actions(
    actions: EngineKVConnectorActions,
    payload: bytes | bytearray | memoryview | tuple[bytes, ...],
) -> None:
    expected_mode = _payload_mode(actions)
    if isinstance(payload, tuple):
        if expected_mode != PayloadMode.SEGMENTED:
            raise ValueError("segmented payload requires segmented connector actions")
        expected_count = max(copy.payload_index or 0 for copy in actions.copies) + 1
        if len(payload) != expected_count:
            raise ValueError("segmented payload count does not match connector actions")
        for copy in actions.copies:
            assert copy.payload_index is not None
            if copy.source_byte_end > len(payload[copy.payload_index]):
                raise ValueError("segmented payload is shorter than connector copy source range")
        _verify_payload_checksum(actions, payload)
        return
    if expected_mode != PayloadMode.MERGED:
        raise ValueError("merged payload requires merged connector actions")
    expected_bytes = _expected_payload_bytes(actions)
    if len(payload) != expected_bytes:
        raise ValueError(f"payload length {len(payload)} != expected {expected_bytes}")
    _verify_payload_checksum(actions, payload)


def _verify_payload_checksum(
    actions: EngineKVConnectorActions,
    payload: bytes | bytearray | memoryview | tuple[bytes | memoryview, ...],
) -> None:
    expected = actions.reservation.payload_checksum
    if not expected:
        return
    digest = hashlib.sha256()
    if isinstance(payload, (bytes, bytearray, memoryview)):
        digest.update(payload)
    else:
        for segment in payload:
            digest.update(segment)
    if digest.hexdigest() != expected:
        raise ValueError("document KV payload checksum does not match handoff")


def _payload_checksum_scan_count(actions: EngineKVConnectorActions) -> int:
    """Return one only when checksum validation actually scans the payload."""

    return 1 if actions.reservation.payload_checksum else 0


def _payload_mode(actions: EngineKVConnectorActions) -> PayloadMode:
    if all(copy.payload_index is None for copy in actions.copies):
        return PayloadMode.MERGED
    if any(copy.payload_index is None for copy in actions.copies):
        raise ValueError("connector actions cannot mix merged and segmented payload copies")
    return PayloadMode.SEGMENTED


def _normalized_actions_record(actions_record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(actions_record, Mapping):
        raise TypeError("actions_record must be a mapping")
    # Normalize away MappingProxyType and other immutable mapping wrappers so
    # vLLM can pickle scheduler-to-worker connector metadata.
    normalized = json.loads(json.dumps(actions_record))
    return normalized


def _load_identity(load: DocumentKVLoadRequest) -> _LoadIdentity:
    return (
        load.request_id,
        load.source_token_start,
        load.token_count,
        tuple(
            (block.block_id, block.token_start, block.token_count, block.block_offset)
            for block in load.blocks
        ),
    )


def _handoff_request_id(params: Mapping[str, Any], record: Mapping[str, Any]) -> str | None:
    value = params.get(DOCUMENT_KV_REQUEST_ID_PARAM)
    if value is None:
        return None
    expected_request_id = _required_string(value, field_name=DOCUMENT_KV_REQUEST_ID_PARAM)
    record_request_id = _required_string(record.get("request_id"), field_name="handoff_record.request_id")
    if expected_request_id != record_request_id:
        raise ValueError(f"{DOCUMENT_KV_REQUEST_ID_PARAM} must match handoff request_id")
    return expected_request_id


def _benchmark_request_id(params: Mapping[str, Any]) -> str | None:
    value = params.get(DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM)
    if value is None:
        return None
    return _required_string(
        value,
        field_name=DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM,
    )


def _actions_with_benchmark_request_id(
    actions: EngineKVConnectorActions,
    benchmark_request_id: str | None,
    *,
    method_registry: MethodRegistry | None = None,
) -> EngineKVConnectorActions:
    metadata = dict(actions.bind.metadata)
    existing = metadata.get(DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM)
    if benchmark_request_id is None:
        if existing is not None:
            raise ValueError(
                "reserved connector action metadata benchmark request id "
                "requires explicit kv_transfer_params binding"
            )
        return actions
    if existing is not None and existing != benchmark_request_id:
        raise ValueError(
            "connector action metadata benchmark request id conflicts with "
            "kv_transfer_params"
        )
    metadata[DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM] = benchmark_request_id
    bound = replace(
        actions,
        bind=replace(actions.bind, metadata=metadata),
    )
    validate_engine_kv_connector_actions(
        bound,
        method_registry=method_registry,
    )
    return bound


def _connector_actions_for_runtime_request(
    actions: EngineKVConnectorActions,
    runtime_request_id: str,
    *,
    method_registry: MethodRegistry | None = None,
) -> EngineKVConnectorActions:
    if actions.reservation.request_id == runtime_request_id:
        validate_engine_kv_connector_actions(
            actions,
            method_registry=method_registry,
        )
        return actions
    rebound = EngineKVConnectorActions(
        reservation=replace(actions.reservation, request_id=runtime_request_id),
        copies=tuple(replace(copy, request_id=runtime_request_id) for copy in actions.copies),
        bind=replace(actions.bind, request_id=runtime_request_id),
        release=replace(actions.release, request_id=runtime_request_id),
        reuse_plan=actions.reuse_plan,
    )
    validate_engine_kv_connector_actions(
        rebound,
        method_registry=method_registry,
    )
    return rebound


def _connector_actions_from_plan(
    plan: EngineKVInjectionPlan,
    *,
    method_registry: MethodRegistry | None = None,
) -> EngineKVConnectorActions:
    payload_mode = plan.payload_mode
    actions = EngineKVConnectorActions(
        reservation=EngineKVReservationAction(
            backend=plan.backend,
            request_id=plan.request_id,
            total_blocks=plan.total_blocks,
            total_tokens=plan.total_tokens,
            estimated_gpu_bytes=plan.estimated_gpu_bytes,
            layout=plan.layout,
            adapter_ids=plan.adapter_ids,
            artifact_identity=plan.artifact_identity,
            payload_checksum=plan.payload_checksum,
        ),
        copies=tuple(
            EngineKVSegmentCopyAction(
                request_id=plan.request_id,
                document_id=segment.document_id,
                chunk_type=segment.chunk_type,
                chunk_id=segment.chunk_id,
                payload_index=index if payload_mode == PayloadMode.SEGMENTED else None,
                source_byte_start=0 if payload_mode == PayloadMode.SEGMENTED else segment.byte_start,
                source_byte_length=segment.byte_length,
                global_byte_start=segment.byte_start,
                global_byte_end=segment.byte_end,
                token_start=segment.token_start,
                token_count=segment.token_count,
                token_end=segment.token_end,
                first_block_index=segment.first_block_index,
                last_block_index_exclusive=segment.last_block_index_exclusive,
                content_hash=segment.content_hash,
                cache_tier=segment.cache_tier,
                token_contract=segment.token_contract,
            )
            for index, segment in enumerate(plan.segments)
        ),
        bind=EngineKVBindAction(
            request_id=plan.request_id,
            handle_uri=plan.handle_uri,
            cache_method=plan.cache_method,
            adapter_ids=plan.adapter_ids,
            metadata=plan.metadata,
        ),
        release=EngineKVReleaseAction(request_id=plan.request_id),
        reuse_plan=plan.reuse_plan,
    )
    validate_engine_kv_connector_actions(
        actions,
        method_registry=method_registry,
    )
    return actions


def _materialized_payload(
    load: DocumentKVLoadRequest,
    *,
    payload_reader: _PayloadReader,
    segmented_load_strategy: str,
) -> _MaterializedPayload:
    payload_mode = _payload_mode(load.actions)
    if load.payload is not None:
        if isinstance(load.payload, tuple) and segmented_load_strategy == "direct":
            raise ValueError(
                "direct segmented payload loading requires one flat external "
                "payload buffer, not an inline segmented tuple"
            )
        selected_strategy = (
            "inline_segment_merge"
            if isinstance(load.payload, tuple)
            else "inline_merged_payload"
        )
        return _MaterializedPayload(
            payload=load.payload,
            configured_segmented_strategy=segmented_load_strategy,
            selected_strategy=selected_strategy,
            payload_mode=payload_mode,
            canonical_segmented_global_view=False,
        )
    canonical_issue = (
        _canonical_segmented_global_view_issue(load.actions)
        if payload_mode == PayloadMode.SEGMENTED
        else None
    )
    canonical = payload_mode == PayloadMode.SEGMENTED and canonical_issue is None
    if segmented_load_strategy == "direct" and payload_mode == PayloadMode.SEGMENTED and not canonical:
        raise ValueError(
            "direct segmented payload loading requires canonical Cachet copy "
            f"metadata: {canonical_issue}"
        )
    payload_uri = _required_string(load.payload_uri, field_name="payload_uri")
    expected_bytes = _expected_payload_bytes(load.actions)
    payload = payload_reader(
        payload_uri,
        expected_bytes=expected_bytes,
        actions=load.actions,
    )
    if len(payload) != expected_bytes:
        raise ValueError(
            f"payload length {len(payload)} != expected {expected_bytes}"
        )
    direct_segmented_snapshot = (
        payload_mode == PayloadMode.SEGMENTED
        and segmented_load_strategy != "legacy"
        and canonical
    )
    snapshot_copy_bytes = 0
    if direct_segmented_snapshot and not isinstance(payload, bytes):
        # A read-only mmap is still a live view of a mutable backing inode. Keep a
        # single process-owned snapshot so checksum validation binds the exact
        # bytes later copied to the device. Payload-cache hits already return an
        # owned immutable ``bytes`` value and therefore need no additional copy.
        payload = bytes(payload)
        snapshot_copy_bytes = expected_bytes
    _verify_payload_checksum(load.actions, payload)
    if payload_mode == PayloadMode.MERGED:
        return _MaterializedPayload(
            payload=payload,
            configured_segmented_strategy=segmented_load_strategy,
            selected_strategy="merged_global_view",
            payload_mode=payload_mode,
            canonical_segmented_global_view=False,
            checksum_validation_count=_payload_checksum_scan_count(load.actions),
        )

    if direct_segmented_snapshot:
        return _MaterializedPayload(
            payload=payload,
            configured_segmented_strategy=segmented_load_strategy,
            selected_strategy="direct_global_snapshot",
            payload_mode=payload_mode,
            canonical_segmented_global_view=True,
            checksum_validation_count=_payload_checksum_scan_count(load.actions),
            snapshot_copy_bytes=snapshot_copy_bytes,
        )

    segmented = _segmented_payload_from_materialized_payload(load.actions, payload)
    return _MaterializedPayload(
        payload=segmented,
        configured_segmented_strategy=segmented_load_strategy,
        selected_strategy="legacy_segment_remerge",
        payload_mode=payload_mode,
        canonical_segmented_global_view=canonical,
        legacy_fallback_reason=(
            "configured_legacy" if segmented_load_strategy == "legacy" else canonical_issue
        ),
        checksum_validation_count=_payload_checksum_scan_count(load.actions),
        reassembly_copy_bytes=_expected_payload_bytes(load.actions),
    )


def _payload_cache_identity(actions: EngineKVConnectorActions) -> str:
    layout = actions.reservation.layout
    copies = []
    for copy in actions.copies:
        if not copy.content_hash:
            raise ValueError("document KV payload cache requires non-empty content_hash on every copy action")
        copies.append(
            {
                "document_id": copy.document_id,
                "chunk_type": copy.chunk_type,
                "chunk_id": copy.chunk_id,
                "payload_index": copy.payload_index,
                "source_byte_start": copy.source_byte_start,
                "source_byte_length": copy.source_byte_length,
                "source_byte_end": copy.source_byte_end,
                "global_byte_start": copy.global_byte_start,
                "global_byte_end": copy.global_byte_end,
                "token_start": copy.token_start,
                "token_count": copy.token_count,
                "token_end": copy.token_end,
                "first_block_index": copy.first_block_index,
                "last_block_index_exclusive": copy.last_block_index_exclusive,
                "content_hash": copy.content_hash,
                "cache_tier": str(copy.cache_tier),
            }
        )
    record = {
        "schema": "vllm_kv_injection.payload_cache_identity.v1",
        "backend": actions.reservation.backend.value,
        "payload_mode": _payload_mode(actions).value,
        "reuse_capability_id": (
            "" if actions.reuse_plan is None else actions.reuse_plan.capability_id
        ),
        "total_blocks": actions.reservation.total_blocks,
        "total_tokens": actions.reservation.total_tokens,
        "layout": {
            "model_id": layout.model_id,
            "lora_id": layout.lora_id,
            "layout_version": layout.layout_version,
            "dtype": layout.dtype,
            "num_layers": layout.num_layers,
            "block_size": layout.block_size,
            "bytes_per_token": layout.bytes_per_token,
            "num_query_heads": layout.num_query_heads,
            "num_kv_heads": layout.num_kv_heads,
            "head_size": layout.head_size,
            "kv_stride_bytes": layout.kv_stride_bytes,
            "shares_kv_storage": layout.shares_kv_storage,
            "storage_layout": str(layout.storage_layout),
        },
        "adapter_ids": list(actions.reservation.adapter_ids),
        "copies": copies,
    }
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_telemetry_record(
    load: DocumentKVLoadRequest,
    *,
    provider_factory: str,
    payload_cache_enabled: bool,
    page_cache_evicted: bool = False,
    cache_state_observation: Mapping[str, object] | None = None,
    total_ns: int,
    payload_materialize_ns: int,
    payload_merge_ns: int,
    payload_view_ns: int,
    layer_load_ns: int,
    layers_loaded: int,
    payload_cache_hits: int,
    payload_cache_misses: int,
    payload_loading: Mapping[str, object] | None = None,
    error_type: str | None,
    error_message: str | None,
    decoded_runtime_bytes: int | None = None,
    h2d_ns: int | None = None,
    scatter_ns: int | None = None,
    wall_start_s: float | None = None,
    wall_end_s: float | None = None,
) -> dict[str, Any]:
    actions = load.actions
    layout = actions.reservation.layout
    block_ids = [block.block_id for block in load.blocks]
    expected_runtime_payload_bytes = (
        actions.reservation.total_tokens * layout.bytes_per_token
    )
    if decoded_runtime_bytes is None:
        decoded_runtime_bytes = (
            expected_runtime_payload_bytes if error_type is None else 0
        )
    if type(decoded_runtime_bytes) is not int or decoded_runtime_bytes < 0:
        raise ValueError("decoded_runtime_bytes must be a non-negative integer")
    if (
        error_type is None
        and decoded_runtime_bytes != expected_runtime_payload_bytes
    ):
        raise ValueError(
            "successful load decoded_runtime_bytes must match runtime KV layout"
        )
    timings_ns: dict[str, Any] = {
        "total": total_ns,
        "payload_materialize": payload_materialize_ns,
        "payload_merge": payload_merge_ns,
        "payload_view": payload_view_ns,
        "layer_load": layer_load_ns,
    }
    if h2d_ns is not None:
        timings_ns["h2d"] = h2d_ns
    if scatter_ns is not None:
        timings_ns["scatter"] = scatter_ns
    record: dict[str, Any] = {
        "record_type": "document_kv.vllm_native_provider_load.v1",
        "schema_version": 1,
        "event": "load_request",
        "success": error_type is None,
        "request_id": load.request_id,
        "provider_factory": provider_factory,
        "timings_ns": timings_ns,
        "wall_clock": {
            "start_s": wall_start_s,
            "end_s": wall_end_s,
        },
        "counts": {
            "source_token_start": load.source_token_start,
            "source_token_end": load.source_token_start + load.token_count,
            "token_count": load.token_count,
            "handoff_total_tokens": actions.reservation.total_tokens,
            "block_count": len(load.blocks),
            "first_block_id": block_ids[0] if block_ids else None,
            "last_block_id": block_ids[-1] if block_ids else None,
            "min_block_id": min(block_ids) if block_ids else None,
            "max_block_id": max(block_ids) if block_ids else None,
            "copy_count": len(actions.copies),
            "layers_loaded": layers_loaded,
            "expected_payload_bytes": _expected_payload_bytes(actions),
            "expected_stored_payload_bytes": _expected_payload_bytes(actions),
            "expected_runtime_payload_bytes": (
                expected_runtime_payload_bytes
            ),
            "decoded_runtime_payload_bytes": decoded_runtime_bytes,
            "payload_cache_hits": payload_cache_hits,
            "payload_cache_misses": payload_cache_misses,
        },
        "layout": {
            "model_id": layout.model_id,
            "lora_id": layout.lora_id,
            "layout_version": layout.layout_version,
            "dtype": layout.dtype,
            "num_layers": layout.num_layers,
            "block_size": layout.block_size,
            "bytes_per_token": layout.bytes_per_token,
            "num_query_heads": layout.num_query_heads,
            "num_kv_heads": layout.num_kv_heads,
            "head_size": layout.head_size,
            "kv_stride_bytes": layout.kv_stride_bytes,
            "shares_kv_storage": layout.shares_kv_storage,
            "storage_layout": str(layout.storage_layout),
        },
        "payload": _payload_telemetry(
            load,
            payload_cache_enabled=payload_cache_enabled,
            page_cache_evicted=page_cache_evicted,
        ),
        "payload_loading": dict(payload_loading or {}),
        "cache_state_attestation": _cache_state_attestation_record(
            load,
            cache_state_observation=cache_state_observation,
            successful=error_type is None,
            decoded_runtime_bytes=decoded_runtime_bytes,
        ),
    }
    benchmark_request_id = actions.bind.metadata.get(
        DOCUMENT_KV_BENCHMARK_REQUEST_ID_PARAM
    )
    if benchmark_request_id is not None:
        record["benchmark_request_id"] = benchmark_request_id
    if error_type is not None:
        record["error"] = {"type": error_type, "message": error_message or error_type}
    return record


def _cache_state_attestation_record(
    load: DocumentKVLoadRequest,
    *,
    cache_state_observation: Mapping[str, object] | None,
    successful: bool,
    decoded_runtime_bytes: int,
) -> dict[str, object]:
    actions = load.actions
    identity = actions.reservation.artifact_identity
    observation = dict(cache_state_observation or {})
    source = observation.get("source", "inline" if load.payload is not None else "unknown")
    bytes_read = observation.get("bytes_read", 0)
    payload_cache_hit = observation.get("payload_cache_hit", False)
    eviction_requested = observation.get("eviction_requested", False)
    eviction_succeeded = observation.get("eviction_succeeded", False)
    direct_io = observation.get("direct_io", False)
    if not isinstance(source, str):
        source = "unknown"
    if type(bytes_read) is not int or bytes_read < 0:
        bytes_read = 0
    payload_cache_hit = payload_cache_hit is True
    eviction_requested = eviction_requested is True
    eviction_succeeded = eviction_succeeded is True
    direct_io = direct_io is True
    expected_stored_bytes = _expected_payload_bytes(actions)
    expected_runtime_bytes = (
        actions.reservation.total_tokens * actions.reservation.layout.bytes_per_token
    )
    expected_tokens = load.token_count
    loaded_tokens = load.token_count if successful else 0
    cold_read_attested = (
        source in {"disk", "file", "local_path", "uri"}
        and bytes_read > 0
        and not payload_cache_hit
        and (direct_io or (eviction_requested and eviction_succeeded))
        and successful
        and bytes_read == expected_stored_bytes
        and decoded_runtime_bytes == expected_runtime_bytes
        and loaded_tokens == expected_tokens
    )
    return {
        "cache_method": actions.bind.cache_method,
        "artifact_id": "" if identity is None else identity.artifact_id,
        "source": source,
        "bytes_read": bytes_read,
        "payload_cache_hit": payload_cache_hit,
        "eviction_requested": eviction_requested,
        "eviction_succeeded": eviction_succeeded,
        "direct_io": direct_io,
        # ``expected_bytes`` remains the v1 compatibility alias for physical
        # bytes expected from storage. Encoded artifacts may expand to a larger
        # runtime KV payload after the provider decoder runs.
        "expected_bytes": expected_stored_bytes,
        "expected_stored_bytes": expected_stored_bytes,
        "expected_runtime_bytes": expected_runtime_bytes,
        "decoded_runtime_bytes": decoded_runtime_bytes,
        "expected_tokens": expected_tokens,
        "loaded_tokens": loaded_tokens,
        "successful_loads": 1 if successful else 0,
        "cold_read_attested": cold_read_attested,
    }


def _payload_telemetry(
    load: DocumentKVLoadRequest,
    *,
    payload_cache_enabled: bool,
    page_cache_evicted: bool = False,
) -> dict[str, Any]:
    payload_uri = load.payload_uri
    if payload_uri is None:
        inline_payload = load.payload
        inline_bytes = (
            sum(len(segment) for segment in inline_payload)
            if isinstance(inline_payload, tuple)
            else len(inline_payload or b"")
        )
        return {
            "source": "inline",
            "inline_segment_count": len(inline_payload) if isinstance(inline_payload, tuple) else 1,
            "inline_bytes": inline_bytes,
            "payload_cache_enabled": payload_cache_enabled,
            "page_cache_evicted": page_cache_evicted,
        }
    scheme, separator, remainder = payload_uri.partition(":")
    if not separator:
        scheme = "local_path"
        remainder = payload_uri
    return {
        "source": "uri",
        "uri_scheme": scheme or "unknown",
        "uri_sha256": hashlib.sha256(payload_uri.encode("utf-8")).hexdigest(),
        "uri_tail": _safe_uri_tail(remainder),
        "payload_cache_enabled": payload_cache_enabled,
        "page_cache_evicted": page_cache_evicted,
    }


def _payload_source_name(payload_uri: str) -> str:
    scheme, separator, _ = payload_uri.partition(":")
    if not separator:
        return "local_path"
    return scheme.lower() or "uri"


def _safe_uri_tail(value: str) -> str | None:
    if not value:
        return None
    tail = value.rstrip("/").rsplit("/", 1)[-1]
    if not tail:
        return None
    return tail[:128]


def _append_jsonl(path: str, record: Mapping[str, Any]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    line = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def _truncated_error_message(exc: Exception, *, max_chars: int = 500) -> str:
    message = str(exc) or type(exc).__name__
    if len(message) > max_chars:
        return message[: max_chars - 3] + "..."
    return message


def _expected_payload_bytes(actions: EngineKVConnectorActions) -> int:
    return max(copy.global_byte_end for copy in actions.copies)


def _evict_file_from_page_cache(fileno: int) -> None:
    """Best-effort drop of a file's pages from the OS page cache.

    Uses ``posix_fadvise(POSIX_FADV_DONTNEED)`` so a subsequent mmap+copy faults
    the pages back in from disk. Advisory and best-effort: it is a no-op on
    platforms without ``posix_fadvise`` and cannot evict still-dirty pages, so it
    is only relied on for cold-hydrate benchmarking where the payload was written
    (and flushed) earlier.
    """

    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadvise is None or dontneed is None:
        return
    try:
        fadvise(fileno, 0, 0, dontneed)
    except OSError:
        pass


def _read_payload_view(
    payload_uri: str,
    *,
    expected_bytes: int | None = None,
    evict_page_cache: bool = False,
) -> memoryview:
    """Read a merged payload fully into host RAM with a plain buffered read.

    The connector previously memory-mapped the payload and faulted pages on the
    host->device copy, which reads at only ~0.5 GB/s and does not scale with
    concurrency. A sequential ``readinto`` in 8 MiB chunks streams at the
    buffered-read rate (~2.2 GB/s single stream) and returns a ``memoryview`` over a
    resident buffer.

    When ``evict_page_cache`` is set the file's pages are dropped from the OS page
    cache before reading.
    """

    path = local_path(payload_uri)
    with open(path, "rb", buffering=0) as handle:
        if evict_page_cache:
            _evict_file_from_page_cache(handle.fileno())
        size = os.fstat(handle.fileno()).st_size
        if expected_bytes is not None and size != expected_bytes:
            raise ValueError(
                f"Engine adapter payload length {size} != expected {expected_bytes}"
            )
        buffer = bytearray(size)
        view = memoryview(buffer)
        offset = 0
        while offset < size:
            read = handle.readinto(view[offset : offset + _PREFETCH_CHUNK_BYTES])
            if not read:
                break
            offset += read
    if offset != size:
        raise ValueError(
            f"Engine adapter payload truncated: read {offset} of {size} expected bytes"
        )
    return view


def _mmap_payload_view(
    payload_uri: str,
    *,
    expected_bytes: int | None = None,
    evict_page_cache: bool = False,
    on_page_cache_evicted: Callable[[], None] | None = None,
) -> memoryview:
    """Memory-map a merged payload so the device copy faults pages on demand.

    The previous loader read the entire payload into a Python ``bytes`` object
    before any copy, which on the benchmark workers ran at only ~1.5-1.8 GB/s and
    accounted for ~70-78% of the per-request hydrate time. Memory-mapping is
    effectively free and lets the host->device transfer pull pages straight from
    the OS page cache (warm) or NVMe (cold). Falls back to a plain read when the
    backing filesystem cannot mmap the file (e.g. some FUSE mounts).

    When ``evict_page_cache`` is set the file's pages are dropped from the OS page
    cache before mapping, forcing the copy to read cold from disk (honest
    cold-hydrate measurement).
    """

    path = local_path(payload_uri)
    with open(path, "rb") as handle:
        if evict_page_cache:
            _evict_file_from_page_cache(handle.fileno())
            if on_page_cache_evicted is not None:
                on_page_cache_evicted()
        try:
            mapped = mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ)
        except (ValueError, OSError):
            data = handle.read()
            if expected_bytes is not None and len(data) != expected_bytes:
                raise ValueError(
                    f"Engine adapter payload length {len(data)} != expected {expected_bytes}"
                ) from None
            return memoryview(data)
    if _MADVISE_READAHEAD_ENABLED:
        _advise_sequential_readahead(mapped)
    if expected_bytes is not None and len(mapped) != expected_bytes:
        mapped.close()
        raise ValueError(f"Engine adapter payload length {len(mapped)} != expected {expected_bytes}")
    return memoryview(mapped)


def _advise_sequential_readahead(mapped: "mmap.mmap") -> None:
    """Ask the kernel to read the whole mapping ahead sequentially.

    NOTE: disabled by default (see ``_MADVISE_READAHEAD_ENABLED``). It was intended
    to widen the read-ahead window so the host->device copy pulls large sequential
    I/Os, but on the benchmark NVMe the MADV_WILLNEED bulk prefetch competed with
    the copy's own on-demand faulting and *doubled* the cold read (measured
    layer_load 656ms -> 1246ms). Retained behind an env flag only. Best-effort: a
    no-op where madvise is unavailable.
    """

    madvise = getattr(mapped, "madvise", None)
    if madvise is None:
        return
    for option_name in ("MADV_SEQUENTIAL", "MADV_WILLNEED"):
        option = getattr(mmap, option_name, None)
        if option is None:
            continue
        try:
            madvise(option)
        except (OSError, ValueError):
            pass


def _segmented_payload_from_materialized_payload(
    actions: EngineKVConnectorActions,
    payload: bytes | memoryview,
) -> tuple[bytes, ...]:
    max_payload_index = max(copy.payload_index or 0 for copy in actions.copies)
    segments = [bytearray() for _ in range(max_payload_index + 1)]
    for copy in actions.copies:
        assert copy.payload_index is not None
        source = payload[copy.global_byte_start : copy.global_byte_end]
        if len(source) != copy.source_byte_length:
            raise ValueError(f"Copy action {copy.chunk_id!r} source range exceeds materialized payload")
        segment = segments[copy.payload_index]
        if len(segment) < copy.source_byte_end:
            segment.extend(b"\x00" * (copy.source_byte_end - len(segment)))
        segment[copy.source_byte_start : copy.source_byte_end] = source
    return tuple(bytes(segment) for segment in segments)


def _canonical_segmented_global_view_issue(
    actions: EngineKVConnectorActions,
) -> str | None:
    """Return why segmented copy metadata cannot prove flat-file global order.

    Cachet's own adapter builder emits exactly one payload entry per copy, in copy
    order, and every entry starts at byte zero. Since connector validation already
    proves that global byte spans are contiguous, those invariants prove that
    concatenating the entries produces the global payload byte-for-byte. More
    general third-party layouts keep using the strict legacy reconstruction path.
    """

    if _payload_mode(actions) != PayloadMode.SEGMENTED:
        return "payload_mode_is_not_segmented"
    global_byte_cursor = 0
    for copy_index, copy in enumerate(actions.copies):
        if copy.payload_index != copy_index:
            return "payload_index_not_in_copy_order"
        if copy.source_byte_start != 0:
            return "source_byte_start_is_not_zero"
        if copy.source_byte_end != copy.source_byte_length:
            return "source_range_is_not_whole_segment"
        if copy.global_byte_start != global_byte_cursor:
            return "global_byte_start_is_not_cumulative"
        expected_global_byte_end = global_byte_cursor + copy.source_byte_length
        if copy.global_byte_end != expected_global_byte_end:
            return "global_byte_end_is_not_cumulative"
        global_byte_cursor = copy.global_byte_end
    if global_byte_cursor != _expected_payload_bytes(actions):
        return "global_byte_coverage_does_not_match_payload"
    return None


def _merged_payload(
    actions: EngineKVConnectorActions,
    materialized: _MaterializedPayload,
) -> _MaterializedPayload:
    payload = materialized.payload
    if materialized.selected_strategy in {
        "direct_global_snapshot",
        "merged_global_view",
    }:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("global payload view strategy requires one flat payload buffer")
        # The URI read checked the exact expected length and materialization hashed
        # the flat buffer once. Canonical segmented metadata proves that this flat
        # order is also the connector's logical merged order, so validating here
        # would only rescan the owned snapshot and defeat the single-hash
        # optimization.
        return materialized

    _validate_payload_matches_actions(actions, payload)
    checksum_validation_count = (
        materialized.checksum_validation_count
        + _payload_checksum_scan_count(actions)
    )
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return replace(
            materialized,
            checksum_validation_count=checksum_validation_count,
        )
    buffer = bytearray(_expected_payload_bytes(actions))
    for copy in actions.copies:
        assert copy.payload_index is not None
        source = payload[copy.payload_index]
        buffer[copy.global_byte_start : copy.global_byte_end] = source[
            copy.source_byte_start : copy.source_byte_end
        ]
    return replace(
        materialized,
        payload=buffer,
        checksum_validation_count=checksum_validation_count,
        reassembly_copy_bytes=(
            materialized.reassembly_copy_bytes + _expected_payload_bytes(actions)
        ),
    )


def _block_spans_for_token_range(
    blocks: object,
    *,
    block_size: int,
    source_token_start: int,
    token_count: int,
) -> tuple[BlockSpan, ...]:
    block_ids = _first_group_block_ids(blocks)
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    cursor = 0
    absolute_token = source_token_start
    spans: list[BlockSpan] = []
    while cursor < token_count:
        block_index = absolute_token // block_size
        if block_index >= len(block_ids):
            raise ValueError("allocated vLLM block ids do not cover the external KV token range")
        block_offset = absolute_token % block_size
        span_tokens = min(token_count - cursor, block_size - block_offset)
        spans.append(
            BlockSpan(
                block_id=block_ids[block_index],
                token_start=cursor,
                token_count=span_tokens,
                block_offset=block_offset,
            )
        )
        cursor += span_tokens
        absolute_token += span_tokens
    return tuple(spans)


def _first_group_block_ids(blocks: object) -> tuple[int, ...]:
    getter = getattr(blocks, "get_block_ids", None)
    if callable(getter):
        blocks = getter()
    if isinstance(blocks, tuple):
        if not blocks:
            raise ValueError("allocated vLLM block ids are empty")
        groups = blocks
        if len(groups) != 1:
            raise ValueError("DocumentKVNativeProvider currently supports a single vLLM KV cache group")
        blocks = groups[0]
    if isinstance(blocks, list) and all(isinstance(block_id, int) for block_id in blocks):
        return tuple(blocks)
    raise TypeError("allocated vLLM blocks must be KVCacheBlocks, tuple[list[int]], or list[int]")


def _scheduled_request_block_ids(scheduler_output: object) -> dict[str, _ScheduledRequestBlocks]:
    scheduled: dict[str, _ScheduledRequestBlocks] = {}
    for new_req in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
        req_id = getattr(new_req, "req_id", None)
        block_ids = getattr(new_req, "block_ids", None)
        if isinstance(req_id, str) and block_ids is not None:
            _add_scheduled_request_blocks(
                scheduled,
                req_id,
                _ScheduledRequestBlocks(block_ids=block_ids),
            )
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if cached is not None:
        req_ids = getattr(cached, "req_ids", ()) or ()
        new_block_ids = getattr(cached, "new_block_ids", ()) or ()
        resumed_req_ids = set(getattr(cached, "resumed_req_ids", ()) or ())
        for req_id, block_ids in zip(req_ids, new_block_ids, strict=False):
            if isinstance(req_id, str) and block_ids is not None:
                # cached_reqs.new_block_ids contains only the blocks allocated
                # in this scheduler step, except when a preempted request
                # resumes and vLLM sends the full block list again.
                _add_scheduled_request_blocks(
                    scheduled,
                    req_id,
                    _ScheduledRequestBlocks(
                        block_ids=block_ids,
                        relative_to_new_tokens=req_id not in resumed_req_ids,
                    ),
                )
    return scheduled


def _add_scheduled_request_blocks(
    scheduled: dict[str, _ScheduledRequestBlocks],
    request_id: str,
    blocks: _ScheduledRequestBlocks,
) -> None:
    if request_id in scheduled:
        raise ValueError(f"duplicate scheduled vLLM block ids for request {request_id!r}")
    scheduled[request_id] = blocks


def _vllm_layer_indices_from_inspection(
    inspection: DocumentKVVLLMLayerMappingInspection,
) -> dict[str, int]:
    if inspection.unresolved_layer_names:
        raise ValueError(
            "Cannot determine vLLM layer index for registered KV cache layer(s): "
            + ", ".join(sorted(inspection.unresolved_layer_names))
        )
    if inspection.duplicate_layer_indices:
        details = "; ".join(
            f"{layer_index}: {', '.join(sorted(names))}"
            for layer_index, names in sorted(inspection.duplicate_layer_indices.items())
        )
        raise ValueError("Duplicate vLLM layer index in registered KV cache layers: " + details)
    return dict(inspection.layer_indices)


def _matchable_prefix_tokens(load: DocumentKVHandoffLoad, request: object) -> int:
    block_size = load.actions.reservation.layout.block_size
    prompt_text_mode = _document_kv_prompt_text_mode(request)
    request_tokens = getattr(request, "num_tokens", None)
    # vLLM's V1 scheduler asserts ``num_computed_tokens <= request.num_tokens``: the
    # externally loaded document KV must be a strict prefix of the tokens the request
    # actually carries. We therefore never report more matched tokens than the visible
    # request length in any prompt mode. The previous "runtime" branch returned the full
    # cached prefix while ignoring ``request.num_tokens``, so a suffix-only request could
    # claim more computed tokens than it held and kill EngineCore with an AssertionError.
    if isinstance(request_tokens, int):
        if prompt_text_mode != "runtime" and request_tokens <= load.total_tokens:
            raise ValueError(
                "Document KV vLLM loads require the full logical prompt; "
                "the visible vLLM request must be longer than the cached prefix"
            )
        candidate_tokens = min(load.total_tokens, max(request_tokens - 1, 0))
    else:
        candidate_tokens = max(load.total_tokens - 1, 0)
    return (candidate_tokens // block_size) * block_size


def _verify_request_token_contracts(
    actions: EngineKVConnectorActions,
    request: object,
) -> None:
    contracts = tuple(copy.token_contract for copy in actions.copies)
    if not any(contract is not None for contract in contracts):
        return
    if any(contract is None for contract in contracts):
        raise ValueError("document KV handoff must provide token contracts for every segment")
    token_ids = _request_token_ids(request)
    if token_ids is None:
        raise ValueError(
            "vLLM request does not expose token ids required by the document KV token contract"
        )
    total_tokens = actions.reservation.total_tokens
    if len(token_ids) < total_tokens:
        raise ValueError(
            f"vLLM request exposes {len(token_ids)} token ids, fewer than "
            f"the cached prefix length {total_tokens}"
        )
    for copy, contract in zip(actions.copies, contracts, strict=True):
        assert contract is not None
        contract.require_match(
            token_ids[copy.token_start : copy.token_end],
            label=f"runtime tokens for {copy.document_id}:{copy.chunk_id}",
        )


def _request_token_ids(request: object) -> tuple[int, ...] | None:
    for attribute in ("prompt_token_ids", "all_token_ids", "token_ids"):
        value = getattr(request, attribute, None)
        if callable(value):
            value = value()
        if value is None:
            continue
        converter = getattr(value, "tolist", None)
        if callable(converter):
            value = converter()
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray, memoryview),
        ):
            values = tuple(value)
            if len(values) == 1 and isinstance(values[0], Sequence) and not isinstance(
                values[0],
                (str, bytes, bytearray, memoryview),
            ):
                values = tuple(values[0])
            if all(type(token_id) is int and token_id >= 0 for token_id in values):
                return values
    return None


def _document_kv_prompt_text_mode(request: object) -> str | None:
    params = getattr(request, "kv_transfer_params", None)
    if not isinstance(params, Mapping):
        return None
    value = params.get(DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM)
    if value is None:
        return None
    if value not in {"logical", "runtime"}:
        raise ValueError(f"{DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM} must be 'logical' or 'runtime'")
    return value


def _payload_tensor_view(
    payload: bytes | bytearray | memoryview,
    load: DocumentKVLoadRequest,
) -> _PayloadTensorView:
    torch = _torch()
    layout = load.actions.reservation.layout
    # This read path reshapes the payload as token-major
    # ([token, layer, K/V, kv_head, head_dim]). A layer-major payload is not a
    # permuted view of that layout, so reading one here would silently corrupt
    # GPU KV instead of failing. Reject any non-token-major layout loudly until a
    # layer-major streaming read path exists.
    axis_order = getattr(layout, "payload_axis_order", None)
    axis_value = getattr(axis_order, "value", axis_order)
    if axis_value not in (None, "token_major"):
        raise ValueError(
            "vLLM native provider can only read token-major KV payloads; "
            f"payload_axis_order={axis_value!r} would be misread as token-major "
            "and corrupt GPU KV (layer-major streaming is not implemented in this "
            "provider read path)."
        )
    dtype = _torch_dtype(layout.dtype)
    dtype_width = _dtype_width(layout.dtype)
    total_scalars = len(payload) // dtype_width
    if len(payload) % dtype_width != 0:
        raise ValueError("payload length is not aligned to the layout dtype")
    if len(payload) != layout.bytes_per_token * load.actions.reservation.total_tokens:
        raise ValueError("payload length does not match connector action layout")
    scalars_per_token = layout.bytes_per_token // dtype_width
    if scalars_per_token % layout.num_layers != 0:
        raise ValueError("layout bytes_per_token is not divisible by num_layers")
    scalars_per_layer = scalars_per_token // layout.num_layers
    tensor = _torch_from_payload_buffer(torch, payload, dtype=dtype, count=total_scalars)
    token_major = tensor.reshape(load.actions.reservation.total_tokens, scalars_per_token)
    return _PayloadTensorView(
        token_major=token_major,
        scalars_per_layer=scalars_per_layer,
        buffer=payload,
    )


def _torch_from_payload_buffer(
    torch: object,
    payload: bytes | bytearray | memoryview,
    *,
    dtype: object,
    count: int,
) -> object:
    """Create a read-only source tensor; injection only copies from it.

    Read-only buffers (immutable ``bytes`` and ``PROT_READ`` memory maps) make
    ``torch.frombuffer`` emit a "buffer is not writable" warning even though we
    never mutate the tensor, so suppress that specific warning.
    """

    if isinstance(payload, (bytes, memoryview)):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable.*",
                category=UserWarning,
            )
            return torch.frombuffer(payload, dtype=dtype, count=count)
    return torch.frombuffer(payload, dtype=dtype, count=count)


def _to_device_contiguous(token_slice: object, device: object) -> object:
    """Stage the contiguous token slice onto ``device`` with one host->device copy."""

    contiguous = token_slice if token_slice.is_contiguous() else token_slice.contiguous()
    if device is None:
        return contiguous
    if getattr(contiguous, "device", None) == device:
        return contiguous
    return contiguous.to(device=device)


def _rope_cos_sin_for_load(
    load: object,
    dst_kv_cache_layer: object,
    *,
    rope_theta: object,
    rotary_dim: object,
    key_position_encoding: KVKeyPositionEncoding = (
        KVKeyPositionEncoding.PRE_ROPE
    ),
) -> tuple[object, object]:
    """Precompute RoPE factors for absolute positions."""

    torch = _torch()
    from document_kv_cache.rope import rope_cos_sin

    head_dim = int(dst_kv_cache_layer.shape[-1])
    rot = int(rotary_dim) if rotary_dim else head_dim
    device = getattr(dst_kv_cache_layer, "device", None)
    if key_position_encoding != KVKeyPositionEncoding.PRE_ROPE:
        raise ValueError(
            "RoPE factors requested for non-repositionable KV keys"
        )
    positions = torch.arange(
        load.source_token_start,
        load.source_token_start + load.token_count,
        device=device,
    )
    return rope_cos_sin(positions, head_dim=head_dim, rope_theta=rope_theta, rotary_dim=rot)


def _fp8_view_dtype(payload_dtype: object) -> object | None:
    """Return the torch fp8 dtype for an fp8 payload dtype string, else ``None``.

    fp8 KV is stored as raw uint8 bytes, so re-roping must bitcast to the real fp8
    dtype (not integer-convert the bytes) before decoding to float.
    """
    torch = _torch()
    if not isinstance(payload_dtype, str):
        return None
    normalized = payload_dtype.lower()
    if normalized in ("fp8", "fp8_e5m2", "float8_e5m2"):
        return getattr(torch, "float8_e5m2", None)
    if normalized in ("fp8_e4m3", "fp8_e4m3fn", "float8_e4m3"):
        return getattr(torch, "float8_e4m3fn", None)
    return None


def _rerope_src_layer_keys(
    src_layer: object,
    *,
    cos: object,
    sin: object,
    rope_theta: object,
    rotary_dim: object,
    payload_dtype: object = None,
) -> object:
    """Return ``src_layer`` with RoPE applied to its keys; values untouched.

    ``src_layer`` is the standard ``[T, 2, kv_heads, head_dim]`` layer tensor.
    This handles pre-RoPE keys at absolute positions.
    fp8 keys arrive as raw uint8 bytes; they are bitcast to the real fp8 dtype so the
    rotation runs on decoded values, then the roped fp8 is bitcast back to uint8 so the
    injector writes the bytes through unchanged (exactly as the no-rope path does).
    """
    torch = _torch()
    from document_kv_cache.rope import apply_rope_to_keys

    if getattr(src_layer, "dim", None) is None or src_layer.dim() != 4 or src_layer.shape[1] != 2:
        raise ValueError(
            "RoPE repositioning requires a [T, 2, kv_heads, head_dim] layer tensor; "
            f"got shape {tuple(getattr(src_layer, 'shape', ()))}"
        )
    keys = src_layer[:, 0]
    values = src_layer[:, 1]
    rot = int(rotary_dim) if rotary_dim else int(keys.shape[-1])
    fp8_dtype = _fp8_view_dtype(payload_dtype)
    if fp8_dtype is not None and keys.dtype == torch.uint8:
        roped = apply_rope_to_keys(keys.view(fp8_dtype), rope_theta=rope_theta, rotary_dim=rot, cos=cos, sin=sin)
        roped = roped.view(torch.uint8)
    else:
        roped = apply_rope_to_keys(keys, rope_theta=rope_theta, rotary_dim=rot, cos=cos, sin=sin)
    return torch.stack((roped, values), dim=1)


def _layer_values_from_token_slice(
    device_token_slice: object,
    scalars_per_layer: int,
    *,
    layer_index: int,
    dst_kv_cache_layer: object,
    layout: object,
) -> object:
    torch = _torch()
    if not torch.is_tensor(dst_kv_cache_layer):
        raise TypeError("registered vLLM KV cache layer must be a torch.Tensor")
    start = layer_index * scalars_per_layer
    end = start + scalars_per_layer
    layer_values = device_token_slice[:, start:end]
    return _reshape_layer_values(layer_values, dst_kv_cache_layer, layout)


def _reshape_layer_values(layer_values: object, dst_kv_cache_layer: object, layout: object) -> object:
    token_count = int(layer_values.shape[0])
    if dst_kv_cache_layer.ndim >= 4 and dst_kv_cache_layer.shape[1] == 2:
        expected_shape = (token_count, 2, *tuple(dst_kv_cache_layer.shape[3:]))
        expected_scalars = math.prod(expected_shape[1:])
        if layer_values.shape[1] == expected_scalars:
            return layer_values.reshape(expected_shape)
        trimmed = _trim_standard_layer_values(layer_values, expected_scalars, layout, dst_kv_cache_layer)
        return trimmed.reshape(expected_shape).to(device=dst_kv_cache_layer.device, dtype=dst_kv_cache_layer.dtype)
    if dst_kv_cache_layer.ndim >= 3:
        expected_shape = (token_count, *tuple(dst_kv_cache_layer.shape[2:]))
        expected_scalars = math.prod(expected_shape[1:])
        if layer_values.shape[1] < expected_scalars:
            raise ValueError("payload layer is smaller than the vLLM flat KV cache layer shape")
        return layer_values[:, :expected_scalars].reshape(expected_shape).to(
            device=dst_kv_cache_layer.device,
            dtype=dst_kv_cache_layer.dtype,
        )
    raise ValueError("registered vLLM KV cache layer has unsupported rank")


def _trim_standard_layer_values(
    layer_values: object,
    expected_scalars: int,
    layout: object,
    dst_kv_cache_layer: object,
) -> object:
    num_kv_heads = getattr(layout, "num_kv_heads", None)
    kv_stride_bytes = getattr(layout, "kv_stride_bytes", None)
    if num_kv_heads is None or kv_stride_bytes is None:
        raise ValueError("padded standard KV payloads require num_kv_heads and kv_stride_bytes")
    dtype_width = _dtype_width(getattr(layout, "dtype"))
    stride_scalars = kv_stride_bytes // dtype_width
    token_count = int(layer_values.shape[0])
    if layer_values.shape[1] != 2 * num_kv_heads * stride_scalars:
        raise ValueError("payload layer shape does not match vLLM standard KV layout geometry")
    if dst_kv_cache_layer.ndim != 5 or dst_kv_cache_layer.shape[3] != num_kv_heads:
        raise ValueError("cannot trim padded payload for this vLLM standard KV cache shape")
    head_scalars = dst_kv_cache_layer.shape[4]
    trimmed = layer_values.reshape(token_count, 2, num_kv_heads, stride_scalars)[:, :, :, :head_scalars]
    if math.prod(trimmed.shape[1:]) != expected_scalars:
        raise ValueError("trimmed payload layer does not match the vLLM KV cache shape")
    return trimmed


def _torch_dtype(dtype: str) -> object:
    torch = _torch()
    normalized = dtype.lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp8": torch.uint8,
        "fp8_e4m3": torch.uint8,
        "fp8_e5m2": torch.uint8,
        "float8": torch.uint8,
        "int8": torch.int8,
        "uint8": torch.uint8,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported document KV payload dtype {dtype!r}") from exc


def _dtype_width(dtype: str) -> int:
    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16", "fp16", "float16"}:
        return 2
    if normalized in {"fp32", "float32"}:
        return 4
    if normalized in {"int8", "uint8", "fp8", "fp8_e4m3", "fp8_e5m2", "float8"}:
        return 1
    raise ValueError(f"Unsupported document KV payload dtype {dtype!r}")


def _request_id(request: object) -> str:
    request_id = getattr(request, "request_id", None)
    return _required_string(request_id, field_name="request_id")


def _probe_actions_from_handle(
    handle: object,
    payload: bytes | tuple[bytes, ...],
) -> EngineKVConnectorActions:
    layout = getattr(handle, "layout", None)
    request_id = _required_string(getattr(handle, "request_id", None), field_name="request_id")
    total_tokens = _positive_int(getattr(handle, "total_tokens", None), field_name="total_tokens")
    total_bytes = _nonnegative_int(getattr(handle, "total_bytes", None), field_name="total_bytes")
    block_size = _positive_int(getattr(layout, "block_size", None), field_name="layout.block_size")
    payload_mode = PayloadMode.SEGMENTED if isinstance(payload, tuple) else PayloadMode.MERGED
    segments = tuple(getattr(handle, "segments", ()))
    actions = EngineKVConnectorActions(
        reservation=EngineKVReservationAction(
            backend=ServingBackend.VLLM,
            request_id=request_id,
            total_blocks=(total_tokens + block_size - 1) // block_size,
            total_tokens=total_tokens,
            estimated_gpu_bytes=total_bytes,
            layout=layout,
            adapter_ids=tuple(getattr(handle, "adapter_ids", ())),
            artifact_identity=getattr(handle, "artifact_identity", None),
            payload_checksum=getattr(handle, "payload_checksum", ""),
        ),
        copies=tuple(
            EngineKVSegmentCopyAction(
                request_id=request_id,
                document_id=segment.document_id,
                chunk_type=segment.chunk_type,
                chunk_id=segment.chunk_id,
                payload_index=index if payload_mode == PayloadMode.SEGMENTED else None,
                source_byte_start=0 if payload_mode == PayloadMode.SEGMENTED else segment.byte_start,
                source_byte_length=segment.byte_length,
                global_byte_start=segment.byte_start,
                global_byte_end=segment.byte_end,
                token_start=segment.token_start,
                token_count=segment.token_count,
                token_end=segment.token_end,
                first_block_index=segment.token_start // block_size,
                last_block_index_exclusive=(segment.token_end + block_size - 1) // block_size,
                content_hash=segment.content_hash,
                token_contract=getattr(segment, "token_contract", None),
            )
            for index, segment in enumerate(segments)
        ),
        bind=EngineKVBindAction(
            request_id=request_id,
            handle_uri=_required_string(getattr(handle, "handle_uri", None), field_name="handle_uri"),
            cache_method=_required_string(getattr(handle, "cache_method", None), field_name="cache_method"),
            adapter_ids=tuple(getattr(handle, "adapter_ids", ())),
            metadata=dict(getattr(handle, "metadata", {})),
        ),
        release=EngineKVReleaseAction(request_id=request_id),
    )
    validate_engine_kv_connector_actions(actions)
    return actions


def _probe_scheduler_output(request_id: str, block_ids: list[int]) -> object:
    return SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id=request_id, block_ids=(block_ids,))],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
    )


def _probe_kv_caches(layout: object, *, block_count: int) -> dict[str, object]:
    torch = _torch()
    dtype = _torch_dtype(getattr(layout, "dtype"))
    shape = _probe_kv_cache_shape(layout, block_count=block_count)
    return {
        layer_name: torch.zeros(shape, dtype=dtype)
        for layer_name in document_kv_vllm_probe_layer_names(layout)
    }


def _probe_kv_cache_shape(layout: object, *, block_count: int) -> tuple[int, ...]:
    dtype_width = _dtype_width(getattr(layout, "dtype"))
    bytes_per_token = _positive_int(getattr(layout, "bytes_per_token", None), field_name="bytes_per_token")
    num_layers = _positive_int(getattr(layout, "num_layers", None), field_name="num_layers")
    block_size = _positive_int(getattr(layout, "block_size", None), field_name="block_size")
    scalars_per_token = bytes_per_token // dtype_width
    if bytes_per_token % dtype_width != 0:
        raise ValueError("layout bytes_per_token is not aligned to dtype")
    if scalars_per_token % num_layers != 0:
        raise ValueError("layout bytes_per_token is not divisible by num_layers")
    scalars_per_layer = scalars_per_token // num_layers
    if getattr(layout, "shares_kv_storage", False):
        return (block_count, block_size, scalars_per_layer)
    num_kv_heads = getattr(layout, "num_kv_heads", None) or 1
    if scalars_per_layer % (2 * num_kv_heads) != 0:
        raise ValueError("layout layer bytes cannot be represented as a standard K/V probe tensor")
    return (block_count, 2, block_size, num_kv_heads, scalars_per_layer // (2 * num_kv_heads))


def _required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _positive_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return _required_string(value, field_name=DOCUMENT_KV_PAYLOAD_URI_PARAM)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError("DocumentKVNativeProvider requires torch at runtime") from exc
    return torch


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _segmented_load_strategy_from_env() -> str:
    value = os.environ.get(DOCUMENT_KV_SEGMENTED_LOAD_STRATEGY_ENV, "auto")
    strategy = value.strip().lower()
    if strategy not in {"auto", "direct", "legacy"}:
        raise ValueError(
            f"{DOCUMENT_KV_SEGMENTED_LOAD_STRATEGY_ENV} must be one of "
            "'auto', 'direct', or 'legacy'"
        )
    return strategy


# Concurrent-prefetch tuning. The chunk size balances syscall overhead against
# transient host memory; the wait timeout is only a safety net so a wedged
# prefetch thread can never stall the engine's load path indefinitely.
_PREFETCH_CHUNK_BYTES = 8 << 20
_PREFETCH_WAIT_TIMEOUT_S = 120.0

# MADV_SEQUENTIAL|WILLNEED read-ahead on the payload mmap was measured to *hurt*
# the cold host->device copy on this NVMe (layer_load 656ms -> 1246ms at par=8,
# TTFT 7.79s -> 12.42s): WILLNEED kicks a bulk read-ahead that competes with the
# copy's own on-demand faulting. Left in as an explicit, default-off escape hatch.
_MADVISE_READAHEAD_ENABLED = _env_truthy("DOCUMENT_KV_MADVISE_READAHEAD")


def _maybe_cuda_sync(device: object) -> None:
    """Synchronize the CUDA device so the surrounding ``perf_counter_ns`` window
    captures the actual GPU execution time rather than just kernel-launch time.

    No-op for CPU tensors / when CUDA is unavailable so the profiling path stays
    safe on hosts without a GPU.
    """

    if device is None:
        return
    device_type = getattr(device, "type", None)
    if device_type != "cuda":
        return
    try:
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except Exception:  # pragma: no cover - profiling must never break a load.
        return
