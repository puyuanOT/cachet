from __future__ import annotations

from typing import Protocol

from vllm_kv_injection.block_mapping import BlockSpan, plan_token_blocks
from vllm_kv_injection.protocol import KVCacheHandle

KVPayload = bytes | tuple[bytes, ...]


class KVConnector(Protocol):
    def reserve(self, handle: KVCacheHandle) -> tuple[BlockSpan, ...]: ...

    def inject(self, handle: KVCacheHandle, blocks: tuple[BlockSpan, ...], *, payload: KVPayload | None = None) -> None:
        ...

    def release(self, request_id: str) -> None: ...


class InMemoryKVConnector:
    """Test double for the vLLM-side connector contract."""

    def __init__(self) -> None:
        self._reservations: dict[str, tuple[BlockSpan, ...]] = {}
        self._injected: set[str] = set()
        self._payloads: dict[str, KVPayload] = {}
        self._next_block_id = 0

    def reserve(self, handle: KVCacheHandle) -> tuple[BlockSpan, ...]:
        handle.validate()
        blocks = plan_token_blocks(
            total_tokens=handle.total_tokens,
            block_size=handle.layout.block_size,
            starting_block_id=self._next_block_id,
        )
        self._next_block_id += len(blocks)
        self._reservations[handle.request_id] = blocks
        return blocks

    def inject(self, handle: KVCacheHandle, blocks: tuple[BlockSpan, ...], *, payload: KVPayload | None = None) -> None:
        expected = self._reservations.get(handle.request_id)
        if expected != blocks:
            raise ValueError(f"Blocks for {handle.request_id} were not reserved by this connector")
        if payload is not None:
            self._payloads[handle.request_id] = payload
        self._injected.add(handle.request_id)

    def release(self, request_id: str) -> None:
        self._reservations.pop(request_id, None)
        self._injected.discard(request_id)
        self._payloads.pop(request_id, None)

    def is_injected(self, request_id: str) -> bool:
        return request_id in self._injected

    def is_reserved(self, request_id: str) -> bool:
        return request_id in self._reservations

    def payload_for(self, request_id: str) -> KVPayload | None:
        return self._payloads.get(request_id)
