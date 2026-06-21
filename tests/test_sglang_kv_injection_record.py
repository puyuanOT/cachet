from sglang_kv_injection.protocol import KVCacheHandle, KVLayout, KVSegment
from sglang_kv_injection.record import SGLangCacheRecord, prefix_key_for_handle
from document_kv_cache.engine_protocol import KVCacheHandle as CoreHandle


def layout() -> KVLayout:
    return KVLayout(
        model_id="qwen3-4b-instruct",
        lora_id="base",
        layout_version="qwen3-sglang-int8-v1",
        dtype="int8",
        num_layers=32,
        block_size=16,
        bytes_per_token=2,
    )


def handle() -> KVCacheHandle:
    return KVCacheHandle(
        request_id="req-1",
        handle_uri="document-kv://req-1",
        layout=layout(),
        segments=(
            KVSegment("doc-a", "document_static", "static", 0, 2, 0, 4),
            KVSegment("doc-a", "document_chunk", "section-1", 2, 3, 4, 6),
        ),
        total_tokens=5,
        total_bytes=10,
        metadata={"source": "unit"},
        cache_method="vanilla",
        adapter_ids=("qa-lora",),
    )


def test_protocol_reexports_core_handle_types():
    assert KVCacheHandle is CoreHandle


def test_cache_record_from_handle_preserves_runtime_metadata():
    record = SGLangCacheRecord.from_handle(handle())

    assert record.request_id == "req-1"
    assert record.handle_uri == "document-kv://req-1"
    assert record.total_tokens == 5
    assert record.total_bytes == 10
    assert record.metadata == {"source": "unit"}
    assert record.prefix_key[:7] == (
        "document-kv",
        "qwen3-4b-instruct",
        "base",
        "qwen3-sglang-int8-v1",
        "int8",
        "vanilla",
        '["qa-lora"]',
    )


def test_prefix_key_distinguishes_repeated_chunk_occurrences():
    repeated = KVCacheHandle(
        request_id="req-1",
        handle_uri="document-kv://req-1",
        layout=layout(),
        segments=(
            KVSegment("doc-a", "document_chunk", "section-1", 0, 2, 0, 4),
            KVSegment("doc-a", "document_chunk", "section-1", 2, 2, 4, 4),
        ),
        total_tokens=4,
        total_bytes=8,
    )

    key = prefix_key_for_handle(repeated)

    assert key[-2] != key[-1]
    assert key[-2] == '[0,"doc-a","document_chunk","section-1","",0,2]'
    assert key[-1] == '[1,"doc-a","document_chunk","section-1","",2,2]'


def test_prefix_key_includes_content_hash_and_escapes_delimiters():
    changed = KVCacheHandle(
        request_id="req-1",
        handle_uri="document-kv://req-1",
        layout=layout(),
        segments=(KVSegment("doc|a", "document_chunk", "section|1", 0, 2, 0, 4, content_hash="hash|v2"),),
        total_tokens=2,
        total_bytes=4,
    )

    assert prefix_key_for_handle(changed)[-1] == '[0,"doc|a","document_chunk","section|1","hash|v2",0,2]'


def test_prefix_key_distinguishes_adapter_id_boundaries_and_empty_adapter_stack():
    first = handle()
    second = KVCacheHandle(
        request_id="req-2",
        handle_uri="document-kv://req-2",
        layout=layout(),
        segments=first.segments,
        total_tokens=first.total_tokens,
        total_bytes=first.total_bytes,
        adapter_ids=("qa", "lora"),
    )
    no_adapter = KVCacheHandle(
        request_id="req-3",
        handle_uri="document-kv://req-3",
        layout=layout(),
        segments=first.segments,
        total_tokens=first.total_tokens,
        total_bytes=first.total_bytes,
    )
    sentinel_adapter = KVCacheHandle(
        request_id="req-4",
        handle_uri="document-kv://req-4",
        layout=layout(),
        segments=first.segments,
        total_tokens=first.total_tokens,
        total_bytes=first.total_bytes,
        adapter_ids=("-",),
    )

    assert prefix_key_for_handle(first)[6] == '["qa-lora"]'
    assert prefix_key_for_handle(second)[6] == '["qa","lora"]'
    assert prefix_key_for_handle(no_adapter)[6] == "[]"
    assert prefix_key_for_handle(sentinel_adapter)[6] == '["-"]'
