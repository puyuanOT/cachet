from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from restaurant_kv_serving.materializer import MaterializedKV


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    request_id: str
    kv: MaterializedKV
    estimated_gpu_bytes: int


class AdmissionQueue:
    def __init__(self, *, max_pending_gpu_bytes: int) -> None:
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

