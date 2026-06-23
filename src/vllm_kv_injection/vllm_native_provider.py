"""Runtime-facing provider for loading Cachet KV payloads into vLLM V1."""

from __future__ import annotations

import importlib
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Protocol

from document_kv_cache.engine import EngineReadyRequest
from document_kv_cache.engine_adapters import (
    EngineKVBindAction,
    EngineKVConnectorActions,
    EngineKVReleaseAction,
    EngineKVReservationAction,
    EngineKVSegmentCopyAction,
    PayloadMode,
    ServingBackend,
    build_engine_kv_connector_actions,
    build_engine_kv_injection_plan,
    engine_kv_connector_actions_from_record,
    engine_kv_connector_actions_to_record,
    read_engine_adapter_request_json,
    validate_engine_kv_connector_actions,
    view_engine_adapter_payload,
)
from document_kv_cache.engine_probe import read_engine_adapter_payload
from vllm_kv_injection.block_mapping import BlockSpan, plan_token_blocks
from vllm_kv_injection.paged_kv_copy import inject_kv_cache_layer, slot_mapping_from_blocks
from vllm_kv_injection.vllm_native_provider_constants import (
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_HANDOFF_RECORD_PARAM,
    DOCUMENT_KV_HANDOFF_SOURCE_FACTORY_CONFIG_KEY,
    DOCUMENT_KV_NATIVE_PROVIDER_FACTORY,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
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
    "DOCUMENT_KV_HANDOFF_JSON_PARAM",
    "DOCUMENT_KV_HANDOFF_RECORD_PARAM",
    "DOCUMENT_KV_HANDOFF_SOURCE_FACTORY_CONFIG_KEY",
    "DOCUMENT_KV_NATIVE_PROVIDER_FACTORY",
    "DOCUMENT_KV_PAYLOAD_URI_PARAM",
    "DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM",
    "DOCUMENT_KV_REQUEST_ID_PARAM",
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
    """Validated Cachet connector actions plus materialized payload bytes."""

    actions: EngineKVConnectorActions
    payload: bytes | tuple[bytes, ...]

    def __post_init__(self) -> None:
        validate_engine_kv_connector_actions(self.actions)
        if self.actions.reservation.backend != ServingBackend.VLLM:
            raise ValueError("Document KV vLLM loads require vllm connector actions")
        _validate_payload_matches_actions(self.actions, self.payload)

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
    payload: bytes | tuple[bytes, ...]
    blocks: tuple[BlockSpan, ...]
    source_token_start: int
    token_count: int

    def __post_init__(self) -> None:
        actions_record = _actions_record(self.actions_record)
        actions = engine_kv_connector_actions_from_record(actions_record, expected_backend=ServingBackend.VLLM)
        if self.request_id != actions.reservation.request_id:
            raise ValueError("load request_id must match connector actions")
        if self.source_token_start < 0:
            raise ValueError("source_token_start must be non-negative")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if self.source_token_start + self.token_count > actions.reservation.total_tokens:
            raise ValueError("load token span exceeds connector actions")
        _validate_payload_matches_actions(actions, self.payload)
        object.__setattr__(self, "actions_record", actions_record)

    @property
    def actions(self) -> EngineKVConnectorActions:
        return engine_kv_connector_actions_from_record(self.actions_record, expected_backend=ServingBackend.VLLM)


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
    buffer: bytes | bytearray


@dataclass(frozen=True, slots=True)
class DocumentKVConnectorMetadata(_KVConnectorMetadata):
    """Scheduler-to-worker metadata consumed by :class:`DocumentKVNativeProvider`."""

    loads: tuple[DocumentKVLoadRequest, ...] = ()


class KVTransferParamsDocumentKVSource:
    """Load Cachet handoff records referenced by vLLM ``kv_transfer_params``.

    Supported request parameters:
    - ``document_kv.handoff_json``: path to a Cachet engine adapter handoff JSON.
    - ``document_kv.handoff_record``: already-decoded handoff record mapping.
    - ``document_kv.payload_uri``: optional payload URI override.
    """

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
        )
        payload_uri = payload_uri_override or plan.payload_source_uri
        if payload_uri is None:
            raise ValueError("document KV handoff requires an external payload URI")
        payload = read_engine_adapter_payload(payload_uri, expected_bytes=plan.total_bytes)
        payload_or_segments = view_engine_adapter_payload(record, payload)
        actions = build_engine_kv_connector_actions(plan, payload_or_segments)
        return DocumentKVHandoffLoad(actions=actions, payload=_payload_bytes(payload_or_segments))


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
    ) -> None:
        self.source = source or KVTransferParamsDocumentKVSource()
        self.provider_factory = _provider_factory_path(provider_factory)
        self._loads: dict[str, DocumentKVHandoffLoad] = {}
        self._allocated: dict[str, DocumentKVLoadRequest] = {}
        self._metadata = DocumentKVConnectorMetadata()
        self._kv_caches: dict[str, object] = {}
        self._layer_indices: dict[str, int] = {}
        self._layer_mapping_inspection = DocumentKVVLLMLayerMappingInspection((), {})
        self._load_errors: set[int] = set()
        self._events: list[dict[str, object]] = []
        self._loads_started = 0
        self._layers_loaded = 0

    def get_num_new_matched_tokens(
        self,
        request: object,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if num_computed_tokens < 0:
            raise ValueError("num_computed_tokens must be non-negative")
        request_id = _request_id(request)
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
        if num_external_tokens % block_size != 0:
            raise ValueError("num_external_tokens must be block-aligned for document KV loads")
        source_token_start = available_tokens - num_external_tokens
        if source_token_start % block_size != 0:
            raise ValueError("document KV load source_token_start must be block-aligned")

        block_spans = _block_spans_for_token_range(
            blocks,
            block_size=block_size,
            source_token_start=source_token_start,
            token_count=num_external_tokens,
        )
        runtime_actions = _connector_actions_for_runtime_request(load.actions, request_id)
        self._allocated[request_id] = DocumentKVLoadRequest(
            request_id=request_id,
            actions_record=engine_kv_connector_actions_to_record(runtime_actions),
            payload=load.payload,
            blocks=block_spans,
            source_token_start=source_token_start,
            token_count=num_external_tokens,
        )

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
            loads.append(
                DocumentKVLoadRequest(
                    request_id=allocated.request_id,
                    actions_record=allocated.actions_record,
                    payload=allocated.payload,
                    blocks=blocks,
                    source_token_start=allocated.source_token_start,
                    token_count=allocated.token_count,
                )
            )
        if missing_request_ids:
            raise ValueError(
                "Document KV allocation is missing scheduled vLLM block ids for request(s): "
                + ", ".join(sorted(missing_request_ids))
            )
        for load in loads:
            self._allocated.pop(load.request_id, None)
            self._loads.pop(load.request_id, None)
        return DocumentKVConnectorMetadata(loads=tuple(loads))

    def bind_connector_metadata(self, connector_metadata: object) -> None:
        if not isinstance(connector_metadata, DocumentKVConnectorMetadata):
            raise TypeError("DocumentKVNativeProvider requires DocumentKVConnectorMetadata")
        self._metadata = connector_metadata

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
            self._load_request(load)
            self._loads_started += 1
            self._events.append({"event": "document_kv_loaded", "request_id": load.request_id})
            self._metadata = DocumentKVConnectorMetadata(loads=loads[index + 1 :])

    def wait_for_layer_load(self, layer_name: str) -> None:
        if layer_name in self._kv_caches:
            return
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

    def get_kv_connector_stats(self) -> DocumentKVConnectorStats:
        return DocumentKVConnectorStats.from_mapping(
            {
                "document_kv_loads_started": self._loads_started,
                "document_kv_layers_loaded": self._layers_loaded,
                "document_kv_load_error_blocks": len(self._load_errors),
            }
        )

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

    def _load_request(self, load: DocumentKVLoadRequest) -> None:
        try:
            merged_payload = _merged_payload(load.actions, load.payload)
            payload_view = _payload_tensor_view(merged_payload, load)
            block_size = load.actions.reservation.layout.block_size
            slot_mappings: dict[object | None, object] = {}
            for layer_name, dst_layer in self._kv_caches.items():
                layer_index = self._layer_indices[layer_name]
                if layer_index >= load.actions.reservation.layout.num_layers:
                    continue
                src_layer = _payload_layer_tensor(
                    payload_view,
                    load,
                    layer_index=layer_index,
                    dst_kv_cache_layer=dst_layer,
                )
                device = getattr(dst_layer, "device", None)
                if device not in slot_mappings:
                    slot_mappings[device] = slot_mapping_from_blocks(
                        load.blocks,
                        block_size=block_size,
                        device=device,
                    )
                inject_kv_cache_layer(
                    dst_layer,
                    src_layer,
                    slot_mappings[device],
                    block_size=block_size,
                )
                self._layers_loaded += 1
        except Exception:
            self._load_errors.update(block.block_id for block in load.blocks)
            raise

    def _release_request(self, request_id: str) -> None:
        self._loads.pop(request_id, None)
        self._allocated.pop(request_id, None)

    def _load_for_request(self, request: object) -> DocumentKVHandoffLoad | None:
        request_id = _request_id(request)
        cached = self._loads.get(request_id)
        if cached is not None:
            return cached
        load = self.source.get_load(request)
        if load is None:
            return None
        self._loads[request_id] = load
        return load


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

    def get_kv_connector_stats(self) -> DocumentKVConnectorStats:
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


def build_document_kv_provider(*, vllm_config: object | None, extra_config: Mapping[str, Any]) -> DocumentKVNativeProvider:
    """Provider factory consumed by ``document_kv.provider_factory``."""

    del vllm_config
    source_factory = extra_config.get(DOCUMENT_KV_HANDOFF_SOURCE_FACTORY_CONFIG_KEY)
    if source_factory is None:
        return DocumentKVNativeProvider()
    if not isinstance(source_factory, str) or not source_factory.strip():
        raise ValueError(f"{DOCUMENT_KV_HANDOFF_SOURCE_FACTORY_CONFIG_KEY} must be a non-empty module:attribute string")
    source = _load_source_factory(source_factory)()
    _validate_source(source)
    return DocumentKVNativeProvider(source=source)


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


def _validate_source(source: object) -> None:
    if not callable(getattr(source, "get_load", None)):
        raise TypeError("document KV handoff source must provide callable get_load")


def _payload_bytes(payload_or_segments: bytes | memoryview | tuple[bytes | memoryview, ...]) -> bytes | tuple[bytes, ...]:
    if isinstance(payload_or_segments, bytes):
        return payload_or_segments
    if isinstance(payload_or_segments, memoryview):
        return payload_or_segments.tobytes()
    return tuple(bytes(segment) for segment in payload_or_segments)


def _validate_payload_matches_actions(actions: EngineKVConnectorActions, payload: bytes | tuple[bytes, ...]) -> None:
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
        return
    if expected_mode != PayloadMode.MERGED:
        raise ValueError("merged payload requires merged connector actions")
    expected_bytes = actions.reservation.total_tokens * actions.reservation.layout.bytes_per_token
    if len(payload) != expected_bytes:
        raise ValueError(f"payload length {len(payload)} != expected {expected_bytes}")


def _payload_mode(actions: EngineKVConnectorActions) -> PayloadMode:
    if all(copy.payload_index is None for copy in actions.copies):
        return PayloadMode.MERGED
    if any(copy.payload_index is None for copy in actions.copies):
        raise ValueError("connector actions cannot mix merged and segmented payload copies")
    return PayloadMode.SEGMENTED


def _actions_record(actions_record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(actions_record, Mapping):
        raise TypeError("actions_record must be a mapping")
    # Normalize away MappingProxyType and other immutable mapping wrappers so
    # vLLM can pickle scheduler-to-worker connector metadata.
    normalized = json.loads(json.dumps(actions_record))
    engine_kv_connector_actions_from_record(normalized, expected_backend=ServingBackend.VLLM)
    return normalized


def _handoff_request_id(params: Mapping[str, Any], record: Mapping[str, Any]) -> str | None:
    value = params.get(DOCUMENT_KV_REQUEST_ID_PARAM)
    if value is None:
        return None
    expected_request_id = _required_string(value, field_name=DOCUMENT_KV_REQUEST_ID_PARAM)
    record_request_id = _required_string(record.get("request_id"), field_name="handoff_record.request_id")
    if expected_request_id != record_request_id:
        raise ValueError(f"{DOCUMENT_KV_REQUEST_ID_PARAM} must match handoff request_id")
    return expected_request_id


def _connector_actions_for_runtime_request(
    actions: EngineKVConnectorActions,
    runtime_request_id: str,
) -> EngineKVConnectorActions:
    if actions.reservation.request_id == runtime_request_id:
        return actions
    rebound = EngineKVConnectorActions(
        reservation=replace(actions.reservation, request_id=runtime_request_id),
        copies=tuple(replace(copy, request_id=runtime_request_id) for copy in actions.copies),
        bind=replace(actions.bind, request_id=runtime_request_id),
        release=replace(actions.release, request_id=runtime_request_id),
    )
    validate_engine_kv_connector_actions(rebound)
    return rebound


def _merged_payload(actions: EngineKVConnectorActions, payload: bytes | tuple[bytes, ...]) -> bytes | bytearray:
    _validate_payload_matches_actions(actions, payload)
    if isinstance(payload, bytes):
        return payload
    buffer = bytearray(actions.reservation.total_tokens * actions.reservation.layout.bytes_per_token)
    for copy in actions.copies:
        assert copy.payload_index is not None
        source = payload[copy.payload_index]
        buffer[copy.global_byte_start : copy.global_byte_end] = source[
            copy.source_byte_start : copy.source_byte_end
        ]
    return buffer


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
    if prompt_text_mode == "runtime":
        raise ValueError(
            "Document KV vLLM loads require the full logical prompt; "
            f"{DOCUMENT_KV_PROMPT_TEXT_MODE_PARAM}='runtime' cannot be used"
        )
    request_tokens = getattr(request, "num_tokens", None)
    if isinstance(request_tokens, int):
        if request_tokens <= load.total_tokens:
            raise ValueError(
                "Document KV vLLM loads require the full logical prompt; "
                "the visible vLLM request must be longer than the cached prefix"
            )
        candidate_tokens = min(load.total_tokens, max(request_tokens - 1, 0))
    else:
        candidate_tokens = max(load.total_tokens - 1, 0)
    return (candidate_tokens // block_size) * block_size


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
    payload: bytes | bytearray,
    load: DocumentKVLoadRequest,
) -> _PayloadTensorView:
    torch = _torch()
    layout = load.actions.reservation.layout
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


def _torch_from_payload_buffer(torch: object, payload: bytes | bytearray, *, dtype: object, count: int) -> object:
    """Create a read-only source tensor; injection only copies from it."""

    if isinstance(payload, bytes):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable.*",
                category=UserWarning,
            )
            return torch.frombuffer(payload, dtype=dtype, count=count)
    return torch.frombuffer(payload, dtype=dtype, count=count)


def _payload_layer_tensor(
    payload_view: _PayloadTensorView,
    load: DocumentKVLoadRequest,
    *,
    layer_index: int,
    dst_kv_cache_layer: object,
) -> object:
    torch = _torch()
    if not torch.is_tensor(dst_kv_cache_layer):
        raise TypeError("registered vLLM KV cache layer must be a torch.Tensor")
    layout = load.actions.reservation.layout
    token_slice = payload_view.token_major[
        load.source_token_start : load.source_token_start + load.token_count
    ]
    start = layer_index * payload_view.scalars_per_layer
    end = start + payload_view.scalars_per_layer
    layer_values = token_slice[:, start:end]
    return _reshape_layer_values(layer_values, dst_kv_cache_layer, layout)


def _reshape_layer_values(layer_values: object, dst_kv_cache_layer: object, layout: object) -> object:
    torch = _torch()
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
    if normalized in {"int8", "uint8", "fp8", "float8"}:
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
