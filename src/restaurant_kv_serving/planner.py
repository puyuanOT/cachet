from __future__ import annotations

from restaurant_kv_serving.manifest import ManifestStore
from restaurant_kv_serving.models import (
    DocumentKVRequest,
    KVCacheKey,
    MaterializationPlan,
    PlanSegment,
    RestaurantKVRequest,
    chunk_types_for_request,
)


CacheRequest = DocumentKVRequest | RestaurantKVRequest


class CachePlanner:
    def __init__(self, manifest: ManifestStore) -> None:
        self._manifest = manifest

    def build_plan(self, request: CacheRequest) -> MaterializationPlan:
        segments: list[PlanSegment] = []
        token_cursor = 0
        byte_cursor = 0

        def add(key: KVCacheKey) -> None:
            nonlocal token_cursor, byte_cursor
            ref = self._manifest.get(key)
            segments.append(PlanSegment(ref=ref, output_token_start=token_cursor, output_byte_start=byte_cursor))
            token_cursor += ref.token_count
            byte_cursor += ref.byte_length

        chunk_types = chunk_types_for_request(request)

        if request.task_prefix_id:
            add(
                KVCacheKey.for_document(
                    model_id=request.model_id,
                    lora_id=request.lora_id,
                    prompt_template_version=request.prompt_template_version,
                    document_id="_task",
                    chunk_type=chunk_types.task_prefix,
                    chunk_id=request.task_prefix_id,
                )
            )

        for document_id, chunk_ids in request.document_chunks.items():
            if request.include_static:
                add(
                    KVCacheKey.for_document(
                        model_id=request.model_id,
                        lora_id=request.lora_id,
                        prompt_template_version=request.prompt_template_version,
                        document_id=document_id,
                        chunk_type=chunk_types.static,
                        chunk_id="static",
                    )
                )
            for chunk_id in chunk_ids:
                add(
                    KVCacheKey.for_document(
                        model_id=request.model_id,
                        lora_id=request.lora_id,
                        prompt_template_version=request.prompt_template_version,
                        document_id=document_id,
                        chunk_type=chunk_types.content,
                        chunk_id=str(chunk_id),
                    )
                )

        return MaterializationPlan(
            request=request,
            segments=tuple(segments),
            total_tokens=token_cursor,
            total_bytes=byte_cursor,
            selected_document_ids=request.selected_document_ids,
        )
