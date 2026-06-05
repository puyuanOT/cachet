from __future__ import annotations

from restaurant_kv_serving.materializer import KVMaterializer
from restaurant_kv_serving.models import RestaurantKVRequest
from restaurant_kv_serving.planner import CachePlanner
from restaurant_kv_serving.scheduler import AdmissionQueue, PreparedRequest


class RestaurantKVService:
    def __init__(
        self,
        *,
        planner: CachePlanner,
        materializer: KVMaterializer,
        admission_queue: AdmissionQueue,
        kv_gpu_bytes_per_payload_byte: float = 1.0,
    ) -> None:
        self.planner = planner
        self.materializer = materializer
        self.admission_queue = admission_queue
        self.kv_gpu_bytes_per_payload_byte = kv_gpu_bytes_per_payload_byte

    def prepare_and_enqueue(self, request: RestaurantKVRequest) -> bool:
        plan = self.planner.build_plan(request)
        materialized = self.materializer.materialize(plan)
        estimated_gpu_bytes = int(len(materialized.payload) * self.kv_gpu_bytes_per_payload_byte)
        return self.admission_queue.try_enqueue(
            PreparedRequest(
                request_id=request.request_id,
                kv=materialized,
                estimated_gpu_bytes=estimated_gpu_bytes,
            )
        )

