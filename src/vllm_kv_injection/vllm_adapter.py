from __future__ import annotations

from dataclasses import dataclass

from document_kv_cache.engine import EngineReadyRequest
from vllm_kv_injection.block_mapping import BlockSpan, SegmentBlockMapping, map_segments_to_reserved_blocks
from vllm_kv_injection.connector import KVConnector
from vllm_kv_injection.protocol import KVCacheHandle


class VLLMIntegrationUnavailable(RuntimeError):
    pass


def import_vllm() -> object:
    try:
        import vllm
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path.
        if exc.name != "vllm":
            raise
        raise VLLMIntegrationUnavailable("vLLM is not installed in this environment") from exc
    return vllm


@dataclass(slots=True)
class VLLMInjectedRequest:
    handle: KVCacheHandle
    payload: bytes | tuple[bytes, ...]
    blocks: tuple[BlockSpan, ...]
    segment_blocks: SegmentBlockMapping
    estimated_gpu_bytes: int


@dataclass(slots=True)
class VLLMDocumentKVInjector:
    """Narrow adapter surface for a patched vLLM KV/block manager.

    The document package owns retrieval, storage, and CPU materialization. This
    adapter consumes an EngineReadyRequest, reserves vLLM-style blocks through a
    connector, and hands the payload to that connector for engine-native copy or
    mapping.
    """

    engine: object | None = None
    connector: KVConnector | None = None

    def inject_ready_request(self, request: EngineReadyRequest) -> VLLMInjectedRequest:
        connector = self._require_connector()
        request.validate()
        handle = request.handle
        blocks = connector.reserve(handle)
        try:
            segment_blocks = map_segments_to_reserved_blocks(handle.segments, blocks)
            connector.inject(handle, blocks, payload=request.payload)
        except Exception:
            connector.release(handle.request_id)
            raise
        return VLLMInjectedRequest(
            handle=handle,
            payload=request.payload,
            blocks=blocks,
            segment_blocks=segment_blocks,
            estimated_gpu_bytes=request.estimated_gpu_bytes,
        )

    def inject_handle(self, handle: KVCacheHandle) -> tuple[BlockSpan, ...]:
        connector = self._require_connector()
        handle.validate()
        blocks = connector.reserve(handle)
        try:
            connector.inject(handle, blocks)
        except Exception:
            connector.release(handle.request_id)
            raise
        return blocks

    def release(self, request_id: str) -> None:
        self._require_connector().release(request_id)

    def _require_connector(self) -> KVConnector:
        if self.connector is not None:
            return self.connector
        raise NotImplementedError(
            "Wire this to the patched vLLM KV/block manager: reserve blocks, copy/map the "
            "materialized cache payload, then attach the block table to the request."
        )
