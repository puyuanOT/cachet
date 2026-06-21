"""Debug probe adapters for the document-kv engine-probe runner."""

from __future__ import annotations

import importlib
import importlib.metadata as package_metadata
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
from vllm_kv_injection.connector import InMemoryKVConnector, KVConnector, KVPayload
from vllm_kv_injection.protocol import KVCacheHandle, KVSegment
from vllm_kv_injection.vllm_dynamic_connector import VLLMSupportsHMA, vllm_runtime_import_error
from vllm_kv_injection.vllm_native_provider import (
    DOCUMENT_KV_NATIVE_PROVIDER_FACTORY,
    DocumentKVNativeProbeConnector,
)
from vllm_kv_injection.vllm_runtime_contract import (
    VLLM_KV_CONNECTOR_V1_RUNTIME,
    validate_vllm_kv_connector_v1_methods,
    vllm_kv_connector_v1_contract_to_record,
)

VLLM_NATIVE_PROBE_CONTRACT = native_probe_adapter_contract_to_record()
VLLM_PROBE_METADATA_CONNECTOR_CLASS = "vllm_kv_injection.connector_class"
VLLM_PROBE_METADATA_NATIVE_RUNTIME = "vllm_kv_injection.native_runtime"
VLLM_PROBE_METADATA_PROBE = "vllm_kv_injection.probe"
VLLM_PROBE_METADATA_PROBE_KIND = "vllm_kv_injection.probe_kind"
VLLM_PROBE_METADATA_REQUEST_ID = "vllm_kv_injection.request_id"
VLLM_PROBE_METADATA_CONNECTOR_FACTORY = "vllm_kv_injection.connector_factory"
VLLM_PROBE_METADATA_PROVIDER_FACTORY = "vllm_kv_injection.provider_factory"
VLLM_PROBE_METADATA_RUNTIME_CONTRACT = "vllm_kv_injection.runtime_contract"
VLLM_DOCUMENT_KV_NATIVE_PROBE_CONNECTOR_FACTORY = (
    "vllm_kv_injection.probe:build_document_kv_native_probe_connector"
)
_VLLM_NATIVE_CONNECTOR_METHODS = ("reserve", "inject", "release")
_VLLM_WRAPPER_METADATA_KEYS = frozenset(
    {
        VLLM_PROBE_METADATA_CONNECTOR_CLASS,
        VLLM_PROBE_METADATA_CONNECTOR_FACTORY,
        VLLM_PROBE_METADATA_NATIVE_RUNTIME,
        VLLM_PROBE_METADATA_PROBE,
        VLLM_PROBE_METADATA_PROBE_KIND,
        VLLM_PROBE_METADATA_REQUEST_ID,
        VLLM_PROBE_METADATA_RUNTIME_CONTRACT,
    }
)


@dataclass(frozen=True, slots=True)
class NativeVLLMConnectorFactoryResult:
    """Native vLLM connector returned by a patched-runtime factory."""

    connector: KVConnector
    engine_version: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.engine_version, str) or not self.engine_version:
            raise ValueError("engine_version must be a non-empty string")
        _validate_metadata_strings(self.metadata)
        _reject_wrapper_owned_metadata(self.metadata)
        _validate_native_connector_contract(self.connector)


NativeVLLMConnectorFactory = Callable[[EngineKVProbeFactoryContext], NativeVLLMConnectorFactoryResult]


@dataclass(slots=True)
class _ProbeReservation:
    action: EngineKVReservationAction
    copies: list[EngineKVSegmentCopyAction] = field(default_factory=list)
    payloads: list[bytes] = field(default_factory=list)


@dataclass(slots=True)
class VLLMConnectorProbe:
    """Adapt engine-probe reserve/copy/bind calls to a vLLM KVConnector.

    This adapter is useful for compatibility tests around connector semantics.
    It is not a native vLLM block-manager probe unless the supplied connector
    itself is backed by patched vLLM internals.
    """

    connector: KVConnector

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
        blocks = self.connector.reserve(handle)
        self.connector.inject(handle, blocks, payload=_payload_from_probe_actions(reservation))

    def release_kv_blocks(self, reservation: _ProbeReservation, action: EngineKVReleaseAction) -> None:
        self.connector.release(action.request_id)


def build_in_memory_debug_probe(context: EngineKVProbeFactoryContext) -> EngineKVProbeFactoryResult:
    """Return a non-native in-memory probe factory for local contract tests."""

    connector = InMemoryKVConnector()
    return EngineKVProbeFactoryResult(
        probe=VLLMConnectorProbe(connector),
        engine_version="vllm-in-memory-debug",
        native_probe=False,
        metadata={
            VLLM_PROBE_METADATA_CONNECTOR_CLASS: connector.__class__.__name__,
            VLLM_PROBE_METADATA_NATIVE_RUNTIME: "false",
            VLLM_PROBE_METADATA_PROBE: "in_memory_debug",
            VLLM_PROBE_METADATA_PROBE_KIND: "debug_in_memory",
            VLLM_PROBE_METADATA_REQUEST_ID: context.plan.request_id,
            VLLM_PROBE_METADATA_RUNTIME_CONTRACT: "document-kv-debug-connector",
        },
    )


def build_native_connector_probe(context: EngineKVProbeFactoryContext) -> EngineKVProbeFactoryResult:
    """Build a caller-attested native vLLM connector probe wrapper."""

    factory_path = context.metadata.get(VLLM_PROBE_METADATA_CONNECTOR_FACTORY)
    if not factory_path:
        raise ValueError(
            f"Native vLLM probe requires {VLLM_PROBE_METADATA_CONNECTOR_FACTORY}=module:factory metadata"
        )
    _reject_wrapper_owned_metadata(
        context.metadata,
        allowed_keys=frozenset({VLLM_PROBE_METADATA_CONNECTOR_FACTORY}),
    )
    factory_result = load_native_connector_factory(factory_path)(context)
    if not isinstance(factory_result, NativeVLLMConnectorFactoryResult):
        raise TypeError("Native vLLM connector factory must return NativeVLLMConnectorFactoryResult")
    connector = factory_result.connector
    if isinstance(connector, InMemoryKVConnector):
        raise ValueError("InMemoryKVConnector cannot produce native vLLM probe evidence")
    return EngineKVProbeFactoryResult(
        probe=VLLMConnectorProbe(connector),
        engine_version=factory_result.engine_version,
        native_probe=True,
        metadata={
            **factory_result.metadata,
            VLLM_PROBE_METADATA_CONNECTOR_CLASS: connector.__class__.__name__,
            VLLM_PROBE_METADATA_CONNECTOR_FACTORY: factory_path,
            VLLM_PROBE_METADATA_NATIVE_RUNTIME: "true",
            VLLM_PROBE_METADATA_PROBE: "native_connector",
            VLLM_PROBE_METADATA_PROBE_KIND: "native_runtime",
            VLLM_PROBE_METADATA_REQUEST_ID: context.plan.request_id,
            VLLM_PROBE_METADATA_RUNTIME_CONTRACT: VLLM_KV_CONNECTOR_V1_RUNTIME,
        },
    )


def build_document_kv_native_probe_connector(context: EngineKVProbeFactoryContext) -> NativeVLLMConnectorFactoryResult:
    """Return the built-in native probe connector backed by DocumentKVNativeProvider."""

    return NativeVLLMConnectorFactoryResult(
        connector=DocumentKVNativeProbeConnector(),
        engine_version=_vllm_engine_version(),
        metadata={
            "runtime.owner": context.backend.value,
            VLLM_PROBE_METADATA_PROVIDER_FACTORY: DOCUMENT_KV_NATIVE_PROVIDER_FACTORY,
        },
    )


def load_native_connector_factory(factory_path: str) -> NativeVLLMConnectorFactory:
    module_name, separator, attribute_name = factory_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("Native vLLM connector factory must use module:attribute syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)
    if not callable(factory):
        raise TypeError(f"Native vLLM connector factory {factory_path!r} is not callable")
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
        raise TypeError("Native vLLM connector metadata keys and values must be strings")


def _reject_wrapper_owned_metadata(
    metadata: Mapping[str, str],
    *,
    allowed_keys: frozenset[str] = frozenset(),
) -> None:
    collisions = sorted(key for key in metadata if key in _VLLM_WRAPPER_METADATA_KEYS - allowed_keys)
    if collisions:
        raise ValueError(
            "Native vLLM connector metadata cannot set wrapper-owned keys: "
            + ", ".join(collisions)
        )


def _validate_native_connector_contract(connector: object) -> None:
    if isinstance(connector, InMemoryKVConnector):
        raise ValueError("InMemoryKVConnector cannot produce native vLLM probe evidence")
    missing_methods = [
        method_name
        for method_name in _VLLM_NATIVE_CONNECTOR_METHODS
        if not callable(getattr(connector, method_name, None))
    ]
    if missing_methods:
        raise TypeError(
            "Native vLLM connector must provide callable methods: "
            + ", ".join(missing_methods)
        )
    validate_vllm_kv_connector_v1_methods(connector)
    runtime_error = vllm_runtime_import_error()
    if runtime_error is not None and not _is_missing_vllm_package(runtime_error):
        raise RuntimeError(
            "Native vLLM connector probe cannot validate SupportsHMA because "
            f"the vLLM runtime import failed: {runtime_error}"
        ) from runtime_error
    if not isinstance(connector, VLLMSupportsHMA):
        raise TypeError("Native vLLM connector must implement vLLM SupportsHMA")
    _validate_document_kv_native_provider_wiring(connector)


def _is_missing_vllm_package(error: Exception) -> bool:
    return isinstance(error, ModuleNotFoundError) and getattr(error, "name", None) == "vllm"


def _validate_document_kv_native_provider_wiring(connector: object) -> None:
    provider = getattr(connector, "provider", None)
    if getattr(connector, "document_kv_native_probe_connector", False):
        return
    if getattr(provider, "document_kv_native_provider", False):
        return
    if getattr(connector, "document_kv_native_provider", False):
        return
    raise TypeError("Native vLLM connector must be backed by DocumentKVNativeProvider wiring")


def _vllm_engine_version() -> str:
    try:
        return f"vllm-{package_metadata.version('vllm')}"
    except package_metadata.PackageNotFoundError:
        return "vllm-document-kv-native-provider"


build_native_connector_probe.document_kv_native_probe_contract = VLLM_NATIVE_PROBE_CONTRACT
build_native_connector_probe.document_kv_native_probe_runtime_contract = vllm_kv_connector_v1_contract_to_record(
    handoff_contract=VLLM_NATIVE_PROBE_CONTRACT,
)
