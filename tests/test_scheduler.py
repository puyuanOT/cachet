import pytest

from document_kv_cache.materializer import MaterializedKV
from document_kv_cache.models import DocumentKVRequest, MaterializationPlan
from document_kv_cache.admission import AdmissionQueue, PreparedRequest


def empty_kv(request_id: str) -> MaterializedKV:
    request = DocumentKVRequest(
        request_id=request_id,
        task_id="qa",
        model_id="model",
        lora_id="lora",
        prompt_template_version="v1",
        document_chunks={},
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


def test_admission_queue_accepts_zero_byte_requests_without_corrupting_accounting():
    queue = AdmissionQueue(max_pending_gpu_bytes=0)

    assert queue.try_enqueue(PreparedRequest("zero", empty_kv("zero"), estimated_gpu_bytes=0))
    assert queue.pending_gpu_bytes == 0
    assert queue.pop_ready().request_id == "zero"
    assert queue.pending_gpu_bytes == 0


@pytest.mark.parametrize("value", (-1, 1.5, True))
def test_admission_queue_rejects_invalid_gpu_byte_budgets(value):
    with pytest.raises(ValueError, match="max_pending_gpu_bytes"):
        AdmissionQueue(max_pending_gpu_bytes=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (-1, 1.5, True))
def test_prepared_request_rejects_invalid_gpu_byte_estimates(value):
    with pytest.raises(ValueError, match="estimated_gpu_bytes"):
        PreparedRequest("bad", empty_kv("bad"), estimated_gpu_bytes=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("request_id", ("", 123, True))
def test_prepared_request_requires_string_request_id(request_id):
    with pytest.raises(ValueError, match="request_id"):
        PreparedRequest(request_id, empty_kv(str(request_id)), estimated_gpu_bytes=0)  # type: ignore[arg-type]


def test_scheduler_namespace_remains_compatible():
    from document_kv_cache.scheduler import AdmissionQueue as LegacyAdmissionQueue
    from document_kv_cache.scheduler import PreparedRequest as LegacyPreparedRequest
    from restaurant_kv_serving.scheduler import AdmissionQueue as RestaurantLegacyAdmissionQueue
    from restaurant_kv_serving.scheduler import PreparedRequest as RestaurantLegacyPreparedRequest

    assert LegacyAdmissionQueue is AdmissionQueue
    assert LegacyPreparedRequest is PreparedRequest
    assert RestaurantLegacyAdmissionQueue is AdmissionQueue
    assert RestaurantLegacyPreparedRequest is PreparedRequest
