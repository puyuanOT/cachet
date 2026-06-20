"""Request admission helpers for pending KV-cache handoffs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from document_kv_cache.materializer import MaterializedKV

__all__ = ["PreparedRequest", "AdmissionQueue"]


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    request_id: str
    kv: MaterializedKV
    estimated_gpu_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be non-empty")
        if type(self.estimated_gpu_bytes) is not int:
            raise ValueError("estimated_gpu_bytes must be an integer")
        if self.estimated_gpu_bytes < 0:
            raise ValueError("estimated_gpu_bytes must be non-negative")


class AdmissionQueue:
    """Small pending-memory gate before handing requests to a serving engine."""

    def __init__(self, *, max_pending_gpu_bytes: int) -> None:
        if type(max_pending_gpu_bytes) is not int:
            raise ValueError("max_pending_gpu_bytes must be an integer")
        if max_pending_gpu_bytes < 0:
            raise ValueError("max_pending_gpu_bytes must be non-negative")
        self.max_pending_gpu_bytes = max_pending_gpu_bytes
        self._queue: deque[PreparedRequest] = deque()
        self._pending_gpu_bytes = 0

    @property
    def pending_gpu_bytes(self) -> int:
        return self._pending_gpu_bytes

    def try_enqueue(self, request: PreparedRequest) -> bool:
        if self._pending_gpu_bytes + request.estimated_gpu_bytes > self.max_pending_gpu_bytes:
            return False
        self._queue.append(request)
        self._pending_gpu_bytes += request.estimated_gpu_bytes
        return True

    def pop_ready(self) -> PreparedRequest | None:
        if not self._queue:
            return None
        item = self._queue.popleft()
        self._pending_gpu_bytes -= item.estimated_gpu_bytes
        return item

    def __len__(self) -> int:
        return len(self._queue)
