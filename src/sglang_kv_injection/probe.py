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
from document_kv_cache.serving_env import SGLANG_VERSION
from sglang_kv_injection.connector import InMemorySGLangKVConnector, KVPayload, SGLangKVConnector
from sglang_kv_injection.protocol import KVCacheHandle, KVSegment
from sglang_kv_injection.record import SGLangCacheRecord
from sglang_kv_injection.sglang_dynamic_backend import (
    DOCUMENT_KV_HICACHE_PAGE_STORE_URI_CONFIG_KEY,
    DOCUMENT_KV_HICACHE_PROVIDER_FACTORY,
    NoOpDocumentKVHiCacheProvider,
    build_document_kv_hicache_provider,
)
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
SGLANG_PROBE_METADATA_PROVIDER_FACTORY = "sglang_kv_injection.provider_factory"
SGLANG_PROBE_METADATA_RUNTIME_CONTRACT = "sglang_kv_injection.runtime_contract"
SGLANG_DOCUMENT_KV_HICACHE_PROBE_CONNECTOR_FACTORY = (
    "sglang_kv_injection.probe:build_document_kv_hicache_probe_connector"
)
SGLANG_CONNECTOR_FACTORY_METADATA_EXAMPLE = (
    f"{SGLANG_PROBE_METADATA_CONNECTOR_FACTORY}={SGLANG_DOCUMENT_KV_HICACHE_PROBE_CONNECTOR_FACTORY}"
)
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


class DocumentKVHiCacheProbeConnector:
    """Probe connector backed by Cachet's runtime-facing SGLang HiCache provider."""

    document_kv_hicache_probe_connector = True

    def __init__(
        self,
        *,
        provider: object | None = None,
        extra_config: Mapping[str, object] | None = None,
    ) -> None:
        self.provider = provider or build_document_kv_hicache_provider(extra_config=extra_config or {})
        if isinstance(self.provider, NoOpDocumentKVHiCacheProvider):
            raise ValueError("DocumentKVHiCacheProbeConnector cannot use NoOpDocumentKVHiCacheProvider")
        _validate_hicache_provider(self.provider)
        self._staged: dict[str, tuple[SGLangCacheRecord, tuple[tuple[object, bytes], ...]]] = {}
        self._staged_by_request_id: dict[str, str] = {}
        self._attached: dict[str, str] = {}

    def stage(self, record: SGLangCacheRecord, *, payload: KVPayload | None = None) -> None:
        if payload is None:
            raise ValueError("document KV SGLang HiCache probe requires copied payload bytes")
        payload_parts = _payload_parts(payload)
        payload_bytes = sum(len(part) for part in payload_parts)
        if payload_bytes != record.total_bytes:
            raise ValueError(f"staged payload bytes {payload_bytes} != record total_bytes {record.total_bytes}")
        pages = tuple((_hicache_page_key(record, index), part) for index, part in enumerate(payload_parts))
        written_keys: list[object] = []
        try:
            for key, part in pages:
                if not bool(self.provider.set(key, part)):  # type: ignore[attr-defined]
                    raise ValueError("document KV SGLang HiCache provider rejected a staged page")
                written_keys.append(key)
        except Exception:
            _delete_provider_pages(self.provider, written_keys)
            raise
        self._staged[record.handle_uri] = (record, pages)
        self._staged_by_request_id[record.request_id] = record.handle_uri

    def attach(self, *, request_id: str, record: SGLangCacheRecord) -> None:
        staged = self._staged.get(record.handle_uri)
        if staged is None or staged[0] != record:
            raise ValueError(f"Record {record.handle_uri} was not staged by this connector")
        for key, expected_page in staged[1]:
            if not bool(self.provider.exists(key)):  # type: ignore[attr-defined]
                raise ValueError("document KV SGLang HiCache provider is missing a staged page")
            observed_page = _provider_page_bytes(self.provider.get(key))  # type: ignore[attr-defined]
            if observed_page != expected_page:
                raise ValueError("document KV SGLang HiCache provider returned a mutated staged page")
        self._attached[request_id] = record.handle_uri

    def release(self, request_id: str) -> None:
        handle_uri = self._attached.pop(request_id, None)
        staged_handle_uri = self._staged_by_request_id.pop(request_id, None)
        if handle_uri is None:
            handle_uri = staged_handle_uri
        if handle_uri is None:
            return
        _record, pages = self._staged.pop(handle_uri, (None, ()))
        _delete_provider_pages(self.provider, [key for key, _page in pages])


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
            f"Native SGLang probe requires metadata entry {SGLANG_CONNECTOR_FACTORY_METADATA_EXAMPLE}"
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


def build_document_kv_hicache_probe_connector(context: EngineKVProbeFactoryContext) -> NativeSGLangConnectorFactoryResult:
    """Return the built-in native probe connector backed by Cachet's HiCache provider."""

    extra_config: dict[str, object] = {}
    page_store_uri = context.metadata.get(DOCUMENT_KV_HICACHE_PAGE_STORE_URI_CONFIG_KEY)
    if page_store_uri is not None:
        extra_config[DOCUMENT_KV_HICACHE_PAGE_STORE_URI_CONFIG_KEY] = page_store_uri
    return NativeSGLangConnectorFactoryResult(
        connector=DocumentKVHiCacheProbeConnector(extra_config=extra_config),
        engine_version=_sglang_engine_version(),
        metadata={
            "runtime.owner": context.backend.value,
            SGLANG_PROBE_METADATA_PROVIDER_FACTORY: DOCUMENT_KV_HICACHE_PROVIDER_FACTORY,
        },
    )


def load_native_connector_factory(factory_path: str) -> NativeSGLangConnectorFactory:
    module_name, separator, attribute_name = factory_path.partition(":")
    if (
        not separator
        or not module_name
        or not attribute_name
        or any(character.isspace() for character in factory_path)
    ):
        raise ValueError("Native SGLang connector factory must use module:attribute syntax without whitespace")
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
    _validate_document_kv_hicache_provider_wiring(connector)


def _validate_document_kv_hicache_provider_wiring(connector: object) -> None:
    if isinstance(connector, DocumentKVHiCacheProbeConnector):
        provider = connector.provider
        if getattr(provider, "document_kv_hicache_provider", False) is not True:
            raise TypeError("DocumentKVHiCacheProbeConnector must be backed by Cachet HiCache provider wiring")
        if isinstance(provider, NoOpDocumentKVHiCacheProvider):
            raise ValueError("Native SGLang connector cannot use NoOpDocumentKVHiCacheProvider")
        _validate_hicache_provider(provider)
        return
    provider = getattr(connector, "provider", None)
    if isinstance(provider, NoOpDocumentKVHiCacheProvider):
        raise ValueError("Native SGLang connector cannot use NoOpDocumentKVHiCacheProvider")
    raise TypeError("Native SGLang connector must be Cachet's provider-backed HiCache probe connector")


def _validate_hicache_provider(provider: object) -> None:
    missing_methods = [
        method_name
        for method_name in ("get", "set", "exists")
        if not callable(getattr(provider, method_name, None))
    ]
    if missing_methods:
        raise TypeError(
            "Document KV SGLang HiCache provider must provide callable methods: "
            + ", ".join(missing_methods)
        )


def _payload_parts(payload: KVPayload) -> tuple[bytes, ...]:
    if isinstance(payload, bytes):
        return (payload,)
    parts: list[bytes] = []
    for index, part in enumerate(payload):
        if isinstance(part, bytes):
            parts.append(part)
            continue
        if isinstance(part, memoryview):
            parts.append(part.tobytes())
            continue
        raise TypeError(f"SGLang HiCache probe payload part {index} must be bytes-like")
    return tuple(parts)


def _hicache_page_key(record: SGLangCacheRecord, index: int) -> tuple[str, ...]:
    return record.prefix_key + ("payload", str(index))


def _provider_page_bytes(value: object | None) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    raise TypeError("SGLang HiCache provider returned a non-bytes page")


def _delete_provider_pages(provider: object, keys: list[object]) -> None:
    delete = getattr(provider, "delete", None)
    if not callable(delete):
        return
    for key in reversed(keys):
        delete(key)


def _sglang_engine_version() -> str:
    try:
        return package_metadata.version("sglang")
    except package_metadata.PackageNotFoundError:
        return SGLANG_VERSION


build_native_connector_probe.document_kv_native_probe_contract = SGLANG_NATIVE_PROBE_CONTRACT
build_native_connector_probe.document_kv_native_probe_runtime_contract = sglang_runtime_cache_contract_to_record(
    handoff_contract=SGLANG_NATIVE_PROBE_CONTRACT,
)
