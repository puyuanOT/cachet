from __future__ import annotations

from collections.abc import Mapping

from restaurant_kv_serving.admission import AdmissionQueue, PreparedRequest
from restaurant_kv_serving.engine import EngineReadyRequest, KVLayout, build_engine_ready_request
from restaurant_kv_serving.materializer import KVMaterializer
from restaurant_kv_serving.models import CacheGenerationMethod, DocumentKVRequest, RestaurantKVRequest
from restaurant_kv_serving.planner import CachePlanner


CacheRequest = DocumentKVRequest | RestaurantKVRequest


class DocumentKVService:
    def __init__(
        self,
        *,
        planner: CachePlanner,
        materializer: KVMaterializer,
        admission_queue: AdmissionQueue,
        kv_gpu_bytes_per_payload_byte: float = 1.0,
    ) -> None:
        if kv_gpu_bytes_per_payload_byte < 0:
            raise ValueError("kv_gpu_bytes_per_payload_byte must be non-negative")
        self.planner = planner
        self.materializer = materializer
        self.admission_queue = admission_queue
        self.kv_gpu_bytes_per_payload_byte = kv_gpu_bytes_per_payload_byte

    def prepare_and_enqueue(self, request: CacheRequest) -> bool:
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

    def prepare_for_engine(
        self,
        request: CacheRequest,
        *,
        layout: KVLayout,
        handle_uri: str | None = None,
        metadata: Mapping[str, str] | None = None,
        cache_method: CacheGenerationMethod | str = CacheGenerationMethod.VANILLA_PREFILL,
        adapter_ids: tuple[str, ...] = (),
        segmented: bool = False,
    ) -> EngineReadyRequest:
        plan = self.planner.build_plan(request)
        materialized = self.materializer.materialize_segmented(plan) if segmented else self.materializer.materialize(plan)
        return build_engine_ready_request(
            materialized,
            layout=layout,
            handle_uri=handle_uri,
            metadata=metadata,
            cache_method=cache_method,
            adapter_ids=adapter_ids,
            kv_gpu_bytes_per_payload_byte=self.kv_gpu_bytes_per_payload_byte,
        )


RestaurantKVService = DocumentKVService
