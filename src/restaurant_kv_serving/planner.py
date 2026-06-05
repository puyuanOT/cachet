from __future__ import annotations

from restaurant_kv_serving.manifest import ManifestStore
from restaurant_kv_serving.models import ChunkType, KVCacheKey, MaterializationPlan, PlanSegment, RestaurantKVRequest


class CachePlanner:
    def __init__(self, manifest: ManifestStore) -> None:
        self._manifest = manifest

    def build_plan(self, request: RestaurantKVRequest) -> MaterializationPlan:
        segments: list[PlanSegment] = []
        token_cursor = 0
        byte_cursor = 0

        def add(key: KVCacheKey) -> None:
            nonlocal token_cursor, byte_cursor
            ref = self._manifest.get(key)
            segments.append(PlanSegment(ref=ref, output_token_start=token_cursor, output_byte_start=byte_cursor))
            token_cursor += ref.token_count
            byte_cursor += ref.byte_length

        if request.task_prefix_id:
            add(
                KVCacheKey(
                    model_id=request.model_id,
                    lora_id=request.lora_id,
                    prompt_template_version=request.prompt_template_version,
                    restaurant_id="_task",
                    chunk_type=ChunkType.TASK_PREFIX,
                    chunk_id=request.task_prefix_id,
                )
            )

        for restaurant_id, review_ids in request.restaurant_reviews.items():
            if request.include_static:
                add(
                    KVCacheKey(
                        model_id=request.model_id,
                        lora_id=request.lora_id,
                        prompt_template_version=request.prompt_template_version,
                        restaurant_id=restaurant_id,
                        chunk_type=ChunkType.RESTAURANT_STATIC,
                        chunk_id="static",
                    )
                )
            for review_id in review_ids:
                add(
                    KVCacheKey(
                        model_id=request.model_id,
                        lora_id=request.lora_id,
                        prompt_template_version=request.prompt_template_version,
                        restaurant_id=restaurant_id,
                        chunk_type=ChunkType.REVIEW,
                        chunk_id=str(review_id),
                    )
                )

        return MaterializationPlan(
            request=request,
            segments=tuple(segments),
            total_tokens=token_cursor,
            total_bytes=byte_cursor,
            selected_restaurants=tuple(request.restaurant_reviews.keys()),
        )

