from __future__ import annotations

from dataclasses import dataclass

from document_kv_cache.engine import EngineReadyRequest
from sglang_kv_injection.connector import KVPayload, SGLangKVConnector
from sglang_kv_injection.protocol import KVCacheHandle
from sglang_kv_injection.record import SGLangCacheRecord


class SGLangIntegrationUnavailable(RuntimeError):
    pass


def import_sglang() -> object:
    try:
        import sglang
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path.
        if exc.name != "sglang":
            raise
        raise SGLangIntegrationUnavailable("SGLang is not installed in this environment") from exc
    return sglang


@dataclass(slots=True)
class SGLangDocumentKVInjector:
    """Narrow adapter surface for a patched SGLang runtime cache."""

    engine: object | None = None
    connector: SGLangKVConnector | None = None

    def stage_ready_request(self, request: EngineReadyRequest) -> SGLangCacheRecord:
        connector = self._require_connector()
        request.validate()
        handle = request.handle
        _validate_payload_matches_handle(request.payload, handle)
        record = SGLangCacheRecord.from_handle(handle)
        try:
            connector.stage(record, payload=request.payload)
            connector.attach(request_id=request.request_id, record=record)
        except Exception:
            connector.release(request.request_id)
            raise
        return record

    def stage_handle(self, handle: KVCacheHandle) -> SGLangCacheRecord:
        connector = self._require_connector()
        record = SGLangCacheRecord.from_handle(handle)
        try:
            connector.stage(record)
        except Exception:
            connector.release(handle.request_id)
            raise
        return record

    def release(self, request_id: str) -> None:
        self._require_connector().release(request_id)

    def _require_connector(self) -> SGLangKVConnector:
        if self.connector is not None:
            return self.connector
        raise NotImplementedError(
            "Wire this to the patched SGLang runtime cache: stage the materialized cache payload, "
            "register the prefix key, then attach it to the request before decode."
        )


def _validate_payload_matches_handle(payload: KVPayload, handle: KVCacheHandle) -> None:
    handle.validate()
    if isinstance(payload, bytes):
        if len(payload) != handle.total_bytes:
            raise ValueError(f"Merged payload bytes {len(payload)} != handle total_bytes {handle.total_bytes}")
        return
    if len(payload) != len(handle.segments):
        raise ValueError(f"Segmented payload count {len(payload)} != handle segment count {len(handle.segments)}")
    for index, (part, segment) in enumerate(zip(payload, handle.segments, strict=True)):
        if len(part) != segment.byte_length:
            raise ValueError(
                f"Segmented payload part {index} bytes {len(part)} != segment {segment.chunk_id} "
                f"byte_length {segment.byte_length}"
            )
