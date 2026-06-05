from restaurant_kv_serving.materializer import MaterializedKV
from restaurant_kv_serving.models import MaterializationPlan, RestaurantKVRequest
from restaurant_kv_serving.scheduler import AdmissionQueue, PreparedRequest


def empty_kv(request_id: str) -> MaterializedKV:
    request = RestaurantKVRequest(
        request_id=request_id,
        task_id="selection",
        model_id="model",
        lora_id="lora",
        prompt_template_version="v1",
        restaurant_reviews={},
    )
    return MaterializedKV(
        plan=MaterializationPlan(request=request, segments=(), total_tokens=0, total_bytes=0),
        payload=b"",
        segment_byte_offsets=(),
        materialization_seconds=0.0,
    )


def test_admission_queue_limits_pending_gpu_bytes():
    queue = AdmissionQueue(max_pending_gpu_bytes=10)

    assert queue.try_enqueue(PreparedRequest("a", empty_kv("a"), estimated_gpu_bytes=6))
    assert not queue.try_enqueue(PreparedRequest("b", empty_kv("b"), estimated_gpu_bytes=5))
    assert queue.pending_gpu_bytes == 6
    assert queue.pop_ready().request_id == "a"
    assert queue.try_enqueue(PreparedRequest("b", empty_kv("b"), estimated_gpu_bytes=5))

