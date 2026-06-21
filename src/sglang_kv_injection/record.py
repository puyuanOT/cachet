from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

from sglang_kv_injection.protocol import KVCacheHandle


SGLangPrefixKey = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SGLangCacheRecord:
    request_id: str
    handle_uri: str
    prefix_key: SGLangPrefixKey
    total_tokens: int
    total_bytes: int
    metadata: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_handle(cls, handle: KVCacheHandle) -> "SGLangCacheRecord":
        handle.validate()
        return cls(
            request_id=handle.request_id,
            handle_uri=handle.handle_uri,
            prefix_key=prefix_key_for_handle(handle),
            total_tokens=handle.total_tokens,
            total_bytes=handle.total_bytes,
            metadata=dict(handle.metadata),
        )


def prefix_key_for_handle(handle: KVCacheHandle) -> SGLangPrefixKey:
    handle.validate()
    adapter_part = json.dumps(list(handle.adapter_ids), ensure_ascii=False, separators=(",", ":"))
    header = (
        "document-kv",
        handle.layout.model_id,
        handle.layout.lora_id,
        handle.layout.layout_version,
        handle.layout.dtype,
        handle.cache_method,
        adapter_part,
    )
    return header + tuple(_segment_identity(index, handle) for index in range(len(handle.segments)))


def _segment_identity(index: int, handle: KVCacheHandle) -> str:
    segment = handle.segments[index]
    return json.dumps(
        [
            index,
            segment.document_id,
            segment.chunk_type,
            segment.chunk_id,
            segment.content_hash,
            segment.token_start,
            segment.token_count,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
