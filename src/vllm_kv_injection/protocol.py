from __future__ import annotations

from document_kv_cache.engine_protocol import KVCacheHandle, KVLayout, KVSegment as DocumentKVSegment


class KVSegment(DocumentKVSegment):
    """Compatibility constructor for the shared document-aware segment type."""

    __slots__ = ()

    def __init__(self, *args, **kwargs) -> None:
        fields = (
            "document_id",
            "chunk_type",
            "chunk_id",
            "token_start",
            "token_count",
            "byte_start",
            "byte_length",
            "content_hash",
        )
        legacy_fields = (
            "chunk_id",
            "token_start",
            "token_count",
            "byte_start",
            "byte_length",
        )
        if args and len(args) == 5:
            chunk_id, token_start, token_count, byte_start, byte_length = args
            positional_values = {
                "chunk_id": chunk_id,
                "token_start": token_start,
                "token_count": token_count,
                "byte_start": byte_start,
                "byte_length": byte_length,
            }
        elif args and len(args) in {7, 8}:
            positional_values = dict(zip(fields, args))
        elif args:
            raise TypeError("KVSegment accepts either 5 legacy or 7/8 document-aware positional arguments")
        else:
            positional_values = {}
        duplicate_fields = sorted(set(positional_values).intersection(kwargs))
        if duplicate_fields:
            raise TypeError(f"Duplicate KVSegment fields: {', '.join(duplicate_fields)}")
        kwargs = {**positional_values, **kwargs}
        if "document_id" not in kwargs and "chunk_type" not in kwargs:
            if all(field in kwargs for field in legacy_fields):
                kwargs["document_id"] = "__legacy__"
                kwargs["chunk_type"] = "legacy_chunk"
        kwargs.setdefault("content_hash", "")
        missing = [field for field in fields if field not in kwargs]
        if missing:
            raise TypeError(f"Missing KVSegment fields: {', '.join(missing)}")
        values = {field: kwargs.pop(field) for field in fields}
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected KVSegment fields: {unexpected}")
        for field, value in values.items():
            object.__setattr__(self, field, value)

__all__ = [
    "KVCacheHandle",
    "KVLayout",
    "KVSegment",
]
