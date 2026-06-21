from __future__ import annotations

from typing import Protocol

from sglang_kv_injection.record import SGLangCacheRecord

KVPayload = bytes | tuple[bytes, ...]


class SGLangKVConnector(Protocol):
    def stage(self, record: SGLangCacheRecord, *, payload: KVPayload | None = None) -> None: ...

    def attach(self, *, request_id: str, record: SGLangCacheRecord) -> None: ...

    def release(self, request_id: str) -> None: ...


class InMemorySGLangKVConnector:
    """Test double for the SGLang-side connector contract."""

    def __init__(self) -> None:
        self._staged: dict[str, SGLangCacheRecord] = {}
        self._attached: dict[str, SGLangCacheRecord] = {}
        self._staged_by_request_id: dict[str, SGLangCacheRecord] = {}
        self._payloads: dict[str, KVPayload] = {}

    def stage(self, record: SGLangCacheRecord, *, payload: KVPayload | None = None) -> None:
        self._staged[record.handle_uri] = record
        self._staged_by_request_id[record.request_id] = record
        if payload is not None:
            self._payloads[record.handle_uri] = payload

    def attach(self, *, request_id: str, record: SGLangCacheRecord) -> None:
        staged = self._staged.get(record.handle_uri)
        if staged != record:
            raise ValueError(f"Record {record.handle_uri} was not staged by this connector")
        self._attached[request_id] = record

    def release(self, request_id: str) -> None:
        record = self._attached.pop(request_id, None)
        if record is None:
            record = self._staged_by_request_id.pop(request_id, None)
        else:
            self._staged_by_request_id.pop(request_id, None)
        if record is not None:
            self._staged.pop(record.handle_uri, None)
            self._payloads.pop(record.handle_uri, None)

    def is_attached(self, request_id: str) -> bool:
        return request_id in self._attached

    def is_staged(self, request_id: str) -> bool:
        return request_id in self._staged_by_request_id

    def payload_for(self, handle_uri: str) -> KVPayload | None:
        return self._payloads.get(handle_uri)
