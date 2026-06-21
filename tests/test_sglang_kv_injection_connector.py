import builtins

import pytest

from document_kv_cache.engine import EngineReadyRequest
from sglang_kv_injection.connector import InMemorySGLangKVConnector
from sglang_kv_injection.protocol import KVCacheHandle, KVLayout, KVSegment
from sglang_kv_injection.record import SGLangCacheRecord
from sglang_kv_injection.sglang_adapter import (
    SGLangDocumentKVInjector,
    SGLangIntegrationUnavailable,
    import_sglang,
)


def record() -> SGLangCacheRecord:
    return SGLangCacheRecord.from_handle(handle())


def handle() -> KVCacheHandle:
    return KVCacheHandle(
        request_id="req-1",
        handle_uri="document-kv://req-1",
        layout=KVLayout(
            model_id="qwen3-4b-instruct",
            lora_id="base",
            layout_version="qwen3-sglang-int8-v1",
            dtype="int8",
            num_layers=32,
            block_size=16,
            bytes_per_token=2,
        ),
        segments=(KVSegment("doc-a", "document_static", "static", 0, 2, 0, 4),),
        total_tokens=2,
        total_bytes=4,
    )


class FailingAttachConnector(InMemorySGLangKVConnector):
    def attach(self, *, request_id: str, record: SGLangCacheRecord) -> None:
        super().attach(request_id=request_id, record=record)
        raise RuntimeError("attach failed")


class FailingStageConnector(InMemorySGLangKVConnector):
    def stage(self, *args, **kwargs) -> None:
        super().stage(*args, **kwargs)
        raise RuntimeError("stage failed")


def test_in_memory_connector_stage_attach_release():
    connector = InMemorySGLangKVConnector()
    cache_record = record()
    payload = b"payload"

    connector.stage(cache_record, payload=payload)
    connector.attach(request_id="req-1", record=cache_record)

    assert connector.is_attached("req-1")
    assert connector.is_staged("req-1")
    assert connector.payload_for("document-kv://req-1") == payload
    connector.release("req-1")
    assert not connector.is_attached("req-1")
    assert not connector.is_staged("req-1")
    assert connector.payload_for("document-kv://req-1") is None


def test_in_memory_connector_rejects_unstaged_record():
    with pytest.raises(ValueError, match="was not staged"):
        InMemorySGLangKVConnector().attach(request_id="req-1", record=record())


def test_sglang_document_injector_consumes_engine_ready_request():
    connector = InMemorySGLangKVConnector()
    ready = EngineReadyRequest(handle=handle(), payload=(b"data",), estimated_gpu_bytes=16)
    injector = SGLangDocumentKVInjector(engine=object(), connector=connector)

    cache_record = injector.stage_ready_request(ready)

    assert cache_record.request_id == "req-1"
    assert connector.is_attached("req-1")
    assert connector.payload_for("document-kv://req-1") == (b"data",)

    injector.release("req-1")

    assert not connector.is_attached("req-1")
    assert connector.payload_for("document-kv://req-1") is None


def test_sglang_document_injector_handle_path_uses_connector():
    connector = InMemorySGLangKVConnector()
    injector = SGLangDocumentKVInjector(connector=connector)

    cache_record = injector.stage_handle(handle())

    assert cache_record.handle_uri == "document-kv://req-1"
    assert connector.is_staged("req-1")
    assert connector.payload_for("document-kv://req-1") is None
    connector.release("req-1")
    assert not connector.is_staged("req-1")
    with pytest.raises(ValueError, match="was not staged"):
        connector.attach(request_id="req-1", record=cache_record)


def test_sglang_document_injector_rejects_payload_size_mismatch_before_staging():
    connector = InMemorySGLangKVConnector()
    injector = SGLangDocumentKVInjector(connector=connector)

    with pytest.raises(ValueError, match="total_bytes"):
        injector.stage_ready_request(EngineReadyRequest(handle=handle(), payload=b"too-long", estimated_gpu_bytes=16))

    assert not connector.is_attached("req-1")
    assert not connector.is_staged("req-1")
    assert connector.payload_for("document-kv://req-1") is None


def test_sglang_document_injector_rejects_invalid_engine_ready_request_before_staging():
    connector = InMemorySGLangKVConnector()
    injector = SGLangDocumentKVInjector(connector=connector)

    with pytest.raises(ValueError, match="estimated_gpu_bytes"):
        injector.stage_ready_request(EngineReadyRequest(handle=handle(), payload=(b"data",), estimated_gpu_bytes=-1))

    assert not connector.is_attached("req-1")
    assert not connector.is_staged("req-1")
    assert connector.payload_for("document-kv://req-1") is None


def test_sglang_document_injector_rejects_non_bytes_segmented_payload_before_staging():
    connector = InMemorySGLangKVConnector()
    injector = SGLangDocumentKVInjector(connector=connector)

    with pytest.raises(TypeError, match="Segmented payload 0 must be bytes"):
        injector.stage_ready_request(
            EngineReadyRequest(handle=handle(), payload=("data",), estimated_gpu_bytes=16)  # type: ignore[arg-type]
        )

    assert not connector.is_attached("req-1")
    assert not connector.is_staged("req-1")
    assert connector.payload_for("document-kv://req-1") is None


def test_sglang_document_injector_rejects_segmented_payload_shape_mismatch():
    connector = InMemorySGLangKVConnector()
    injector = SGLangDocumentKVInjector(connector=connector)

    with pytest.raises(ValueError, match="Segmented payload 0 byte length"):
        injector.stage_ready_request(EngineReadyRequest(handle=handle(), payload=(b"x",), estimated_gpu_bytes=16))

    assert not connector.is_attached("req-1")
    assert connector.payload_for("document-kv://req-1") is None


def test_sglang_document_injector_releases_staged_payload_on_attach_failure():
    connector = FailingAttachConnector()
    injector = SGLangDocumentKVInjector(connector=connector)

    with pytest.raises(RuntimeError, match="attach failed"):
        injector.stage_ready_request(EngineReadyRequest(handle=handle(), payload=b"data", estimated_gpu_bytes=16))

    assert not connector.is_attached("req-1")
    assert not connector.is_staged("req-1")
    assert connector.payload_for("document-kv://req-1") is None


def test_sglang_document_injector_releases_staged_payload_on_stage_failure():
    connector = FailingStageConnector()
    injector = SGLangDocumentKVInjector(connector=connector)

    with pytest.raises(RuntimeError, match="stage failed"):
        injector.stage_ready_request(EngineReadyRequest(handle=handle(), payload=b"data", estimated_gpu_bytes=16))

    assert not connector.is_attached("req-1")
    assert not connector.is_staged("req-1")
    assert connector.payload_for("document-kv://req-1") is None


def test_sglang_document_injector_handle_path_releases_staged_record_on_stage_failure():
    connector = FailingStageConnector()
    injector = SGLangDocumentKVInjector(connector=connector)

    with pytest.raises(RuntimeError, match="stage failed"):
        injector.stage_handle(handle())

    assert not connector.is_attached("req-1")
    assert not connector.is_staged("req-1")
    assert connector.payload_for("document-kv://req-1") is None


def test_sglang_document_injector_requires_connector_for_runtime_operations():
    with pytest.raises(NotImplementedError, match="patched SGLang"):
        SGLangDocumentKVInjector().stage_handle(handle())


def test_import_sglang_reports_missing_optional_dependency(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sglang":
            raise ModuleNotFoundError("No module named 'sglang'", name="sglang")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SGLangIntegrationUnavailable, match="SGLang is not installed"):
        import_sglang()


def test_import_sglang_preserves_transitive_import_failures(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sglang":
            raise ModuleNotFoundError("No module named 'cuda_runtime'", name="cuda_runtime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError, match="cuda_runtime"):
        import_sglang()


def test_import_sglang_preserves_non_missing_import_errors(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sglang":
            raise RuntimeError("bad runtime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="bad runtime"):
        import_sglang()
