import hashlib

import pytest

from document_kv_cache.admission import AdmissionQueue, PreparedRequest
from document_kv_cache.materializer import MaterializedKV
from document_kv_cache.models import ChunkRef, DocumentChunkType, DocumentKVRequest, KVCacheKey, MaterializationPlan, PlanSegment


def materialized_kv(payload: bytes = b"kv") -> MaterializedKV:
    key = KVCacheKey.for_document(
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        document_id="doc-1",
        chunk_type=DocumentChunkType.DOCUMENT_CHUNK,
        chunk_id="p1",
    )
    ref = ChunkRef(
        key=key,
        shard_uri="/tmp/cache.kvpack",
        byte_offset=0,
        byte_length=len(payload),
        token_count=1,
        dtype="int8",
        layout_version="toy-v1",
        checksum=hashlib.sha256(payload).hexdigest(),
    )
    request = DocumentKVRequest(
        request_id="req-1",
        task_id="qa",
        model_id="qwen3:4b-instruct",
        lora_id="base",
        prompt_template_version="v1",
        document_chunks={"doc-1": ["p1"]},
    )
    plan = MaterializationPlan(
        request=request,
        segments=(PlanSegment(ref=ref, output_token_start=0, output_byte_start=0),),
        total_tokens=1,
        total_bytes=len(payload),
        selected_document_ids=("doc-1",),
    )
    return MaterializedKV(
        plan=plan,
        payload=payload,
        segment_byte_offsets=(0,),
        materialization_seconds=0.0,
    )


def test_prepared_request_validates_serving_handoff_inputs():
    kv = materialized_kv()
    request = PreparedRequest(request_id="req-1", kv=kv, estimated_gpu_bytes=2)

    assert request.request_id == "req-1"
    assert request.kv is kv
    assert request.estimated_gpu_bytes == 2

    with pytest.raises(ValueError, match="request_id must be non-empty"):
        PreparedRequest(request_id="", kv=kv, estimated_gpu_bytes=2)
    with pytest.raises(TypeError, match="kv must be a MaterializedKV"):
        PreparedRequest(request_id="req-1", kv=object(), estimated_gpu_bytes=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="estimated_gpu_bytes must be an integer"):
        PreparedRequest(request_id="req-1", kv=kv, estimated_gpu_bytes=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="estimated_gpu_bytes must be non-negative"):
        PreparedRequest(request_id="req-1", kv=kv, estimated_gpu_bytes=-1)


def test_admission_queue_validates_entries_without_mutating_state():
    queue = AdmissionQueue(max_pending_gpu_bytes=10)

    with pytest.raises(TypeError, match="request must be a PreparedRequest"):
        queue.try_enqueue(object())  # type: ignore[arg-type]

    assert len(queue) == 0
    assert queue.pending_gpu_bytes == 0


def test_admission_queue_enforces_pending_gpu_budget_and_pop_order():
    queue = AdmissionQueue(max_pending_gpu_bytes=4)
    first = PreparedRequest(request_id="req-1", kv=materialized_kv(b"a"), estimated_gpu_bytes=2)
    second = PreparedRequest(request_id="req-2", kv=materialized_kv(b"b"), estimated_gpu_bytes=2)
    too_large = PreparedRequest(request_id="req-3", kv=materialized_kv(b"c"), estimated_gpu_bytes=1)

    assert queue.try_enqueue(first)
    assert queue.try_enqueue(second)
    assert not queue.try_enqueue(too_large)
    assert queue.pending_gpu_bytes == 4
    assert len(queue) == 2

    assert queue.pop_ready() is first
    assert queue.pending_gpu_bytes == 2
    assert queue.pop_ready() is second
    assert queue.pending_gpu_bytes == 0
    assert queue.pop_ready() is None
