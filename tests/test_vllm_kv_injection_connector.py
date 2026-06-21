import builtins
from dataclasses import replace

import pytest

from document_kv_cache.engine import EngineReadyRequest
from document_kv_cache.engine_protocol import KVSegment as DocumentKVSegment
from vllm_kv_injection.connector import InMemoryKVConnector
from vllm_kv_injection.protocol import KVCacheHandle, KVLayout, KVSegment
from vllm_kv_injection.vllm_adapter import (
    VLLMDocumentKVInjector,
    VLLMInjectedRequest,
    VLLMIntegrationUnavailable,
    import_vllm,
)


def segment(
    document_id: str,
    chunk_type: str,
    chunk_id: str,
    token_start: int,
    token_count: int,
    byte_start: int,
    byte_length: int,
) -> KVSegment:
    return KVSegment(
        document_id=document_id,
        chunk_type=chunk_type,
        chunk_id=chunk_id,
        token_start=token_start,
        token_count=token_count,
        byte_start=byte_start,
        byte_length=byte_length,
    )


def test_legacy_segment_constructor_returns_core_segment():
    legacy = KVSegment("static", 0, 4, 0, 40)
    legacy_keywords = KVSegment(chunk_id="static", token_start=0, token_count=4, byte_start=0, byte_length=40)
    document_positional = KVSegment("doc-a", "document_static", "static", 0, 4, 0, 40)

    assert isinstance(legacy, KVSegment)
    assert isinstance(legacy, DocumentKVSegment)
    assert legacy.document_id == "__legacy__"
    assert legacy.chunk_type == "legacy_chunk"
    assert legacy.chunk_id == "static"
    assert legacy_keywords == legacy
    assert document_positional.document_id == "doc-a"
    assert document_positional.chunk_type == "document_static"
    assert document_positional.content_hash == ""


def handle() -> KVCacheHandle:
    return KVCacheHandle(
        request_id="req-1",
        handle_uri="document-kv://req-1",
        layout=KVLayout(
            model_id="qwen3-4b-instruct",
            lora_id="base",
            layout_version="v1",
            dtype="fp8",
            num_layers=32,
            block_size=4,
            bytes_per_token=1024,
        ),
        segments=(
            segment("doc-a", "document_static", "static", 0, 4, 0, 40),
            segment("doc-a", "document_chunk", "doc-chunk-a", 4, 3, 40, 30),
        ),
        total_tokens=7,
        total_bytes=70,
    )


def segmented_payload() -> tuple[bytes, bytes]:
    return (b"s" * 40, b"c" * 30)


class FailingInjectConnector(InMemoryKVConnector):
    def inject(self, *args, **kwargs) -> None:
        super().inject(*args, **kwargs)
        raise RuntimeError("copy failed")


class IncompleteReservationConnector(InMemoryKVConnector):
    def reserve(self, kv_handle: KVCacheHandle):
        blocks = super().reserve(kv_handle)
        return blocks[:1]


def test_in_memory_connector_reserve_inject_release():
    connector = InMemoryKVConnector()
    kv_handle = handle()
    payload = b"payload"

    blocks = connector.reserve(kv_handle)
    connector.inject(kv_handle, blocks, payload=payload)

    assert len(blocks) == 2
    assert connector.is_injected("req-1")
    assert connector.is_reserved("req-1")
    assert connector.payload_for("req-1") == payload
    connector.release("req-1")
    assert not connector.is_injected("req-1")
    assert not connector.is_reserved("req-1")
    assert connector.payload_for("req-1") is None


def test_vllm_document_injector_consumes_engine_ready_request():
    connector = InMemoryKVConnector()
    ready = EngineReadyRequest(handle=handle(), payload=segmented_payload(), estimated_gpu_bytes=140)
    injector = VLLMDocumentKVInjector(engine=object(), connector=connector)

    injected = injector.inject_ready_request(ready)

    assert isinstance(injected, VLLMInjectedRequest)
    assert injected.handle.request_id == "req-1"
    assert injected.payload == segmented_payload()
    assert injected.estimated_gpu_bytes == 140
    assert len(injected.blocks) == 2
    assert connector.is_injected("req-1")
    assert connector.payload_for("req-1") == segmented_payload()
    assert injected.segment_blocks[(0, "doc-a", "document_static", "static", "")][0].block_id == 0

    injector.release("req-1")

    assert not connector.is_injected("req-1")
    assert not connector.is_reserved("req-1")
    assert connector.payload_for("req-1") is None


def test_vllm_document_injector_segment_blocks_follow_reserved_block_ids():
    connector = InMemoryKVConnector()
    injector = VLLMDocumentKVInjector(connector=connector)
    first_handle = handle()
    second_handle = replace(first_handle, request_id="req-2", handle_uri="document-kv://req-2")

    injector.inject_ready_request(EngineReadyRequest(handle=first_handle, payload=b"x" * 70, estimated_gpu_bytes=70))
    injected = injector.inject_ready_request(EngineReadyRequest(handle=second_handle, payload=b"y" * 70, estimated_gpu_bytes=70))

    assert injected.blocks[0].block_id == 2
    assert injected.segment_blocks[(0, "doc-a", "document_static", "static", "")][0].block_id == 2


def test_vllm_document_injector_rejects_payload_size_mismatch_before_reserving():
    connector = InMemoryKVConnector()
    injector = VLLMDocumentKVInjector(connector=connector)

    with pytest.raises(ValueError, match="total_bytes"):
        injector.inject_ready_request(EngineReadyRequest(handle=handle(), payload=b"too-short", estimated_gpu_bytes=70))

    assert not connector.is_injected("req-1")


def test_vllm_document_injector_validates_ready_request_before_reserving():
    connector = InMemoryKVConnector()
    injector = VLLMDocumentKVInjector(connector=connector)

    with pytest.raises(ValueError, match="estimated_gpu_bytes"):
        injector.inject_ready_request(EngineReadyRequest(handle=handle(), payload=b"x" * 70, estimated_gpu_bytes=-1))

    assert not connector.is_reserved("req-1")


def test_vllm_document_injector_rejects_segmented_payload_shape_mismatch():
    connector = InMemoryKVConnector()
    injector = VLLMDocumentKVInjector(connector=connector)

    with pytest.raises(ValueError, match="Segmented payload 0"):
        injector.inject_ready_request(
            EngineReadyRequest(handle=handle(), payload=(b"short", b"c" * 30), estimated_gpu_bytes=70)
        )

    assert not connector.is_injected("req-1")


def test_vllm_document_injector_releases_reserved_blocks_on_inject_failure():
    connector = FailingInjectConnector()
    injector = VLLMDocumentKVInjector(connector=connector)

    with pytest.raises(RuntimeError, match="copy failed"):
        injector.inject_ready_request(EngineReadyRequest(handle=handle(), payload=b"x" * 70, estimated_gpu_bytes=70))

    assert not connector.is_injected("req-1")
    assert not connector.is_reserved("req-1")
    assert connector.payload_for("req-1") is None


def test_vllm_document_injector_handle_path_releases_reserved_blocks_on_inject_failure():
    connector = FailingInjectConnector()
    injector = VLLMDocumentKVInjector(connector=connector)

    with pytest.raises(RuntimeError, match="copy failed"):
        injector.inject_handle(handle())

    assert not connector.is_injected("req-1")
    assert not connector.is_reserved("req-1")
    assert connector.payload_for("req-1") is None


def test_vllm_document_injector_releases_reserved_blocks_on_mapping_failure():
    connector = IncompleteReservationConnector()
    injector = VLLMDocumentKVInjector(connector=connector)

    with pytest.raises(ValueError, match="No reserved block covers token"):
        injector.inject_ready_request(EngineReadyRequest(handle=handle(), payload=b"x" * 70, estimated_gpu_bytes=70))

    assert not connector.is_injected("req-1")
    assert connector.payload_for("req-1") is None


def test_vllm_document_injector_handle_path_uses_connector():
    connector = InMemoryKVConnector()
    injector = VLLMDocumentKVInjector(connector=connector)

    blocks = injector.inject_handle(handle())

    assert len(blocks) == 2
    assert connector.is_injected("req-1")


def test_vllm_document_injector_requires_connector_for_runtime_operations():
    with pytest.raises(NotImplementedError, match="patched vLLM"):
        VLLMDocumentKVInjector().inject_handle(handle())


def test_handle_validation_rejects_non_contiguous_segments():
    kv_handle = KVCacheHandle(
        request_id="req-1",
        handle_uri="document-kv://req-1",
        layout=handle().layout,
        segments=(segment("doc-a", "document_chunk", "bad", 1, 2, 0, 20),),
        total_tokens=2,
        total_bytes=20,
    )

    with pytest.raises(ValueError, match="Non-contiguous token"):
        kv_handle.validate()


def test_layout_validation_rejects_invalid_gqa_shape():
    layout = KVLayout(
        model_id="qwen3",
        lora_id="base",
        layout_version="v1",
        dtype="fp8",
        num_layers=32,
        block_size=4,
        bytes_per_token=1024,
        num_query_heads=8,
        num_kv_heads=16,
        head_size=128,
        kv_stride_bytes=1024,
    )
    kv_handle = KVCacheHandle(
        request_id="req-1",
        handle_uri="document-kv://req-1",
        layout=layout,
        segments=(),
        total_tokens=0,
        total_bytes=0,
    )

    with pytest.raises(ValueError, match="num_kv_heads"):
        kv_handle.validate()


def test_layout_validation_rejects_non_divisible_gqa_shape():
    kv_handle = KVCacheHandle(
        request_id="req-1",
        handle_uri="document-kv://req-1",
        layout=KVLayout(
            model_id="qwen3",
            lora_id="base",
            layout_version="v1",
            dtype="fp8",
            num_layers=32,
            block_size=4,
            bytes_per_token=1024,
            num_query_heads=30,
            num_kv_heads=8,
            head_size=128,
            kv_stride_bytes=1024,
        ),
        segments=(),
        total_tokens=0,
        total_bytes=0,
    )

    with pytest.raises(ValueError, match="divisible"):
        kv_handle.validate()


def test_layout_validation_rejects_incomplete_shared_kv_shape():
    kv_handle = KVCacheHandle(
        request_id="req-1",
        handle_uri="document-kv://req-1",
        layout=KVLayout(
            model_id="qwen3",
            lora_id="base",
            layout_version="v1",
            dtype="fp8",
            num_layers=32,
            block_size=4,
            bytes_per_token=1024,
            num_query_heads=32,
            shares_kv_storage=True,
        ),
        segments=(),
        total_tokens=0,
        total_bytes=0,
    )

    with pytest.raises(ValueError, match="required together"):
        kv_handle.validate()


def test_import_vllm_reports_missing_optional_dependency(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "vllm":
            raise ModuleNotFoundError("No module named 'vllm'", name="vllm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(VLLMIntegrationUnavailable, match="vLLM is not installed"):
        import_vllm()


def test_import_vllm_preserves_transitive_import_failures(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "vllm":
            raise ModuleNotFoundError("No module named 'cuda_runtime'", name="cuda_runtime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError, match="cuda_runtime"):
        import_vllm()
