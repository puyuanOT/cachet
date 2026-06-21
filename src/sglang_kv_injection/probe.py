"""Debug probe adapters for the document-kv engine-probe runner."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from document_kv_cache.engine_adapters import (
    EngineKVBindAction,
    EngineKVReleaseAction,
    EngineKVReservationAction,
    EngineKVSegmentCopyAction,
)
from document_kv_cache.engine_probe import EngineKVProbeFactoryContext, EngineKVProbeFactoryResult
from document_kv_cache.native_probe_factories import native_probe_adapter_contract_to_record
from sglang_kv_injection.connector import InMemorySGLangKVConnector, KVPayload, SGLangKVConnector
from sglang_kv_injection.protocol import KVCacheHandle, KVSegment
from sglang_kv_injection.record import SGLangCacheRecord
from sglang_kv_injection.sglang_runtime_contract import (
    SGLANG_RUNTIME_CACHE_RUNTIME,
    sglang_runtime_cache_contract_to_record,
    validate_sglang_runtime_cache_methods,
)

SGLANG_NATIVE_PROBE_CONTRACT = native_probe_adapter_contract_to_record()
SGLANG_PROBE_METADATA_CONNECTOR_CLASS = "sglang_kv_injection.connector_class"
SGLANG_PROBE_METADATA_NATIVE_RUNTIME = "sglang_kv_injection.native_runtime"
SGLANG_PROBE_METADATA_PROBE = "sglang_kv_injection.probe"
SGLANG_PROBE_METADATA_PROBE_KIND = "sglang_kv_injection.probe_kind"
SGLANG_PROBE_METADATA_REQUEST_ID = "sglang_kv_injection.request_id"
SGLANG_PROBE_METADATA_CONNECTOR_FACTORY = "sglang_kv_injection.connector_factory"
SGLANG_PROBE_METADATA_RUNTIME_CONTRACT = "sglang_kv_injection.runtime_contract"
_SGLANG_NATIVE_CONNECTOR_METHODS = ("stage", "attach", "release")
_SGLANG_WRAPPER_METADATA_KEYS = frozenset(
    {
        SGLANG_PROBE_METADATA_CONNECTOR_CLASS,
        SGLANG_PROBE_METADATA_CONNECTOR_FACTORY,
        SGLANG_PROBE_METADATA_NATIVE_RUNTIME,
        SGLANG_PROBE_METADATA_PROBE,
        SGLANG_PROBE_METADATA_PROBE_KIND,
        SGLANG_PROBE_METADATA_REQUEST_ID,
        SGLANG_PROBE_METADATA_RUNTIME_CONTRACT,
    }
)


@dataclass(frozen=True, slots=True)
class NativeSGLangConnectorFactoryResult:
    """Native SGLang connector returned by a patched-runtime factory."""

    connector: SGLangKVConnector
    engine_version: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.engine_version, str) or not self.engine_version:
            raise ValueError("engine_version must be a non-empty string")
        _validate_metadata_strings(self.metadata)
        _reject_wrapper_owned_metadata(self.metadata)
        _validate_native_connector_contract(self.connector)


NativeSGLangConnectorFactory = Callable[[EngineKVProbeFactoryContext], NativeSGLangConnectorFactoryResult]


@dataclass(slots=True)
class _ProbeReservation:
    action: EngineKVReservationAction
    copies: list[EngineKVSegmentCopyAction] = field(default_factory=list)
    payloads: list[bytes] = field(default_factory=list)


@dataclass(slots=True)
class SGLangConnectorProbe:
    """Adapt engine-probe reserve/copy/bind calls to an SGLang KV connector.

    This adapter validates connector semantics for local tests. It is not a
    native SGLang runtime-cache probe unless the supplied connector is backed by
    patched SGLang runtime internals.
    """

    connector: SGLangKVConnector

    def reserve_kv_blocks(self, action: EngineKVReservationAction) -> _ProbeReservation:
        return _ProbeReservation(action=action)

    def import_kv_segment(
        self,
        reservation: _ProbeReservation,
        action: EngineKVSegmentCopyAction,
        payload: memoryview,
    ) -> None:
        if action.request_id != reservation.action.request_id:
            raise ValueError("copy action request_id does not match reservation")
        reservation.copies.append(action)
        reservation.payloads.append(payload.tobytes())

    def bind_kv_handle(self, reservation: _ProbeReservation, action: EngineKVBindAction) -> None:
        handle = _handle_from_probe_actions(reservation, action)
        record = SGLangCacheRecord.from_handle(handle)
        self.connector.stage(record, payload=_payload_from_probe_actions(reservation))
        self.connector.attach(request_id=action.request_id, record=record)

    def release_kv_blocks(self, reservation: _ProbeReservation, action: EngineKVReleaseAction) -> None:
        self.connector.release(action.request_id)


def build_in_memory_debug_probe(context: EngineKVProbeFactoryContext) -> EngineKVProbeFactoryResult:
    """Return a non-native in-memory probe factory for local contract tests."""

    connector = InMemorySGLangKVConnector()
    return EngineKVProbeFactoryResult(
        probe=SGLangConnectorProbe(connector),
        engine_version="sglang-in-memory-debug",
        native_probe=False,
        metadata={
            SGLANG_PROBE_METADATA_CONNECTOR_CLASS: connector.__class__.__name__,
            SGLANG_PROBE_METADATA_NATIVE_RUNTIME: "false",
            SGLANG_PROBE_METADATA_PROBE: "in_memory_debug",
            SGLANG_PROBE_METADATA_PROBE_KIND: "debug_in_memory",
            SGLANG_PROBE_METADATA_REQUEST_ID: context.plan.request_id,
            SGLANG_PROBE_METADATA_RUNTIME_CONTRACT: "document-kv-debug-connector",
        },
    )


def build_native_connector_probe(context: EngineKVProbeFactoryContext) -> EngineKVProbeFactoryResult:
    """Build a caller-attested native SGLang connector probe wrapper."""

    factory_path = context.metadata.get(SGLANG_PROBE_METADATA_CONNECTOR_FACTORY)
    if not factory_path:
        raise ValueError(
            f"Native SGLang probe requires {SGLANG_PROBE_METADATA_CONNECTOR_FACTORY}=module:factory metadata"
        )
    _reject_wrapper_owned_metadata(
        context.metadata,
        allowed_keys=frozenset({SGLANG_PROBE_METADATA_CONNECTOR_FACTORY}),
    )
    factory_result = load_native_connector_factory(factory_path)(context)
    if not isinstance(factory_result, NativeSGLangConnectorFactoryResult):
        raise TypeError("Native SGLang connector factory must return NativeSGLangConnectorFactoryResult")
    connector = factory_result.connector
    if isinstance(connector, InMemorySGLangKVConnector):
        raise ValueError("InMemorySGLangKVConnector cannot produce native SGLang probe evidence")
    return EngineKVProbeFactoryResult(
        probe=SGLangConnectorProbe(connector),
        engine_version=factory_result.engine_version,
        native_probe=True,
        metadata={
            **factory_result.metadata,
            SGLANG_PROBE_METADATA_CONNECTOR_CLASS: connector.__class__.__name__,
            SGLANG_PROBE_METADATA_CONNECTOR_FACTORY: factory_path,
            SGLANG_PROBE_METADATA_NATIVE_RUNTIME: "true",
            SGLANG_PROBE_METADATA_PROBE: "native_connector",
            SGLANG_PROBE_METADATA_PROBE_KIND: "native_runtime",
            SGLANG_PROBE_METADATA_REQUEST_ID: context.plan.request_id,
            SGLANG_PROBE_METADATA_RUNTIME_CONTRACT: SGLANG_RUNTIME_CACHE_RUNTIME,
        },
    )


def load_native_connector_factory(factory_path: str) -> NativeSGLangConnectorFactory:
    module_name, separator, attribute_name = factory_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("Native SGLang connector factory must use module:attribute syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)
    if not callable(factory):
        raise TypeError(f"Native SGLang connector factory {factory_path!r} is not callable")
    return factory


def _handle_from_probe_actions(
    reservation: _ProbeReservation,
    bind: EngineKVBindAction,
) -> KVCacheHandle:
    copies = tuple(reservation.copies)
    segments = tuple(
        KVSegment(
            document_id=copy.document_id,
            chunk_type=copy.chunk_type,
            chunk_id=copy.chunk_id,
            token_start=copy.token_start,
            token_count=copy.token_count,
            byte_start=copy.global_byte_start,
            byte_length=copy.source_byte_length,
            content_hash=copy.content_hash,
        )
        for copy in copies
    )
    total_bytes = copies[-1].global_byte_end if copies else 0
    return KVCacheHandle(
        request_id=reservation.action.request_id,
        handle_uri=bind.handle_uri,
        layout=reservation.action.layout,
        segments=segments,
        total_tokens=reservation.action.total_tokens,
        total_bytes=total_bytes,
        metadata=dict(bind.metadata),
        cache_method=bind.cache_method,
        adapter_ids=bind.adapter_ids,
    )


def _payload_from_probe_actions(reservation: _ProbeReservation) -> KVPayload:
    copies = reservation.copies
    if all(copy.payload_index is None for copy in copies):
        return b"".join(payload for _, payload in sorted(zip(copies, reservation.payloads), key=_global_byte_start))
    if any(copy.payload_index is None for copy in copies):
        raise ValueError("Cannot mix merged and segmented probe copy actions")
    ordered = sorted(zip(copies, reservation.payloads), key=_payload_index)
    return tuple(payload for _, payload in ordered)


def _global_byte_start(item: tuple[EngineKVSegmentCopyAction, bytes]) -> int:
    return item[0].global_byte_start


def _payload_index(item: tuple[EngineKVSegmentCopyAction, bytes]) -> int:
    payload_index = item[0].payload_index
    if payload_index is None:
        raise ValueError("Segmented payload copy action is missing payload_index")
    return payload_index


def _validate_metadata_strings(metadata: Mapping[str, str]) -> None:
    invalid_entries = [
        key
        for key, value in metadata.items()
        if not isinstance(key, str) or not isinstance(value, str)
    ]
    if invalid_entries:
        raise TypeError("Native SGLang connector metadata keys and values must be strings")


def _reject_wrapper_owned_metadata(
    metadata: Mapping[str, str],
    *,
    allowed_keys: frozenset[str] = frozenset(),
) -> None:
    collisions = sorted(key for key in metadata if key in _SGLANG_WRAPPER_METADATA_KEYS - allowed_keys)
    if collisions:
        raise ValueError(
            "Native SGLang connector metadata cannot set wrapper-owned keys: "
            + ", ".join(collisions)
        )


def _validate_native_connector_contract(connector: object) -> None:
    if isinstance(connector, InMemorySGLangKVConnector):
        raise ValueError("InMemorySGLangKVConnector cannot produce native SGLang probe evidence")
    missing_methods = [
        method_name
        for method_name in _SGLANG_NATIVE_CONNECTOR_METHODS
        if not callable(getattr(connector, method_name, None))
    ]
    if missing_methods:
        raise TypeError(
            "Native SGLang connector must provide callable methods: "
            + ", ".join(missing_methods)
        )
    validate_sglang_runtime_cache_methods(connector)


build_native_connector_probe.document_kv_native_probe_contract = SGLANG_NATIVE_PROBE_CONTRACT
build_native_connector_probe.document_kv_native_probe_runtime_contract = sglang_runtime_cache_contract_to_record(
    handoff_contract=SGLANG_NATIVE_PROBE_CONTRACT,
)
