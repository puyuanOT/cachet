"""Cache keys, chunk references, requests, and materialization plan models."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 compatibility path.
    from enum import Enum

    class StrEnum(str, Enum):
        pass

from typing import TypeAlias

from document_kv_cache.engine_protocol import (
    KVStorageLayout,
    kv_storage_layout_from_value,
)


__all__ = [
    "DocumentChunkType",
    "DocumentChunkRole",
    "CacheGenerationMethod",
    "DocumentChunkMap",
    "FrozenDocumentChunkMap",
    "CacheChunkType",
    "CacheChunkTypeSet",
    "DOCUMENT_CHUNK_TYPES",
    "LEGACY_RESTAURANT_CHUNK_TYPES",
    "KVCacheKey",
    "ChunkRef",
    "DocumentKVRequest",
    "PlanSegment",
    "MaterializationPlan",
    "chunk_type_role",
    "chunk_type_sort_order",
    "chunk_types_for_request",
]

SHA256_HEX_LENGTH = 64


class ChunkType(StrEnum):
    TASK_PREFIX = "task_prefix"
    RESTAURANT_STATIC = "restaurant_static"
    REVIEW = "review"


class DocumentChunkType(StrEnum):
    TASK_PREFIX = "task_prefix"
    DOCUMENT_STATIC = "document_static"
    DOCUMENT_CHUNK = "document_chunk"


class DocumentChunkRole(StrEnum):
    TASK_PREFIX = "task_prefix"
    STATIC = "static"
    CONTENT = "content"
    OTHER = "other"


class CacheGenerationMethod(StrEnum):
    VANILLA_PREFILL = "vanilla_prefill"
    ADAPTER_TRAINED = "adapter_trained"
    KV_PACKET = "kv_packet"
    CUSTOM = "custom"


ChunkId: TypeAlias = str | int
DocumentChunkMap: TypeAlias = Mapping[str, Sequence[ChunkId]]
NormalizedDocumentChunkMap: TypeAlias = Mapping[str, tuple[ChunkId, ...]]
CacheChunkType: TypeAlias = ChunkType | DocumentChunkType


class FrozenDocumentChunkMap(dict[str, tuple[ChunkId, ...]]):
    """Read-only dict used by request objects after chunk-map normalization."""

    __slots__ = ()

    def __init__(
        self,
        items: Mapping[str, Sequence[ChunkId]] | Sequence[tuple[str, Sequence[ChunkId]]] = (),
        *,
        field_name: str = "FrozenDocumentChunkMap",
    ) -> None:
        item_mapping = items if isinstance(items, Mapping) else dict(items)
        dict.__init__(self, _normalize_chunk_map_items(field_name, item_mapping.items()))

    def __setitem__(self, key: str, value: tuple[ChunkId, ...]) -> None:
        raise TypeError("FrozenDocumentChunkMap is immutable")

    def __delitem__(self, key: str) -> None:
        raise TypeError("FrozenDocumentChunkMap is immutable")

    def clear(self) -> None:
        raise TypeError("FrozenDocumentChunkMap is immutable")

    def pop(self, *args: object) -> object:
        raise TypeError("FrozenDocumentChunkMap is immutable")

    def popitem(self) -> tuple[str, tuple[ChunkId, ...]]:
        raise TypeError("FrozenDocumentChunkMap is immutable")

    def setdefault(self, *args: object, **kwargs: object) -> object:
        raise TypeError("FrozenDocumentChunkMap is immutable")

    def update(self, *args: object, **kwargs: object) -> None:
        raise TypeError("FrozenDocumentChunkMap is immutable")

    def __ior__(self, other: object) -> "FrozenDocumentChunkMap":
        raise TypeError("FrozenDocumentChunkMap is immutable")

    def copy(self) -> "FrozenDocumentChunkMap":
        return self

    def __copy__(self) -> "FrozenDocumentChunkMap":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "FrozenDocumentChunkMap":
        return FrozenDocumentChunkMap(
            (
                copy.deepcopy(document_id, memo),
                copy.deepcopy(chunk_ids, memo),
            )
            for document_id, chunk_ids in self.items()
        )

    def __reduce__(
        self,
    ) -> tuple[type["FrozenDocumentChunkMap"], tuple[tuple[tuple[str, tuple[ChunkId, ...]], ...]]]:
        return (type(self), (tuple(self.items()),))


@dataclass(frozen=True, slots=True)
class CacheChunkTypeSet:
    task_prefix: CacheChunkType
    static: CacheChunkType
    content: CacheChunkType


@dataclass(frozen=True, slots=True, init=False)
class KVCacheKey:
    model_id: str
    lora_id: str
    prompt_template_version: str
    document_id: str
    chunk_type: CacheChunkType
    chunk_id: str
    content_hash: str = ""

    def __init__(
        self,
        model_id: str,
        lora_id: str,
        prompt_template_version: str,
        document_id: str | None = None,
        chunk_type: CacheChunkType | None = None,
        chunk_id: str | None = None,
        content_hash: str = "",
        *,
        restaurant_id: str | None = None,
    ) -> None:
        resolved_document_id = _resolve_document_id(document_id=document_id, restaurant_id=restaurant_id)
        if chunk_type is None:
            raise TypeError("chunk_type is required")
        if chunk_id is None:
            raise TypeError("chunk_id is required")
        if not isinstance(chunk_type, (ChunkType, DocumentChunkType)):
            raise TypeError("chunk_type must be a CacheChunkType")
        for name, value in (
            ("model_id", model_id),
            ("lora_id", lora_id),
            ("prompt_template_version", prompt_template_version),
            ("document_id", resolved_document_id),
            ("chunk_id", chunk_id),
        ):
            _validate_storage_key_part(name, value)
        _validate_optional_storage_key_part("content_hash", content_hash)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "lora_id", lora_id)
        object.__setattr__(self, "prompt_template_version", prompt_template_version)
        object.__setattr__(self, "document_id", resolved_document_id)
        object.__setattr__(self, "chunk_type", chunk_type)
        object.__setattr__(self, "chunk_id", chunk_id)
        object.__setattr__(self, "content_hash", content_hash)

    def storage_key(self) -> str:
        parts = [
            self.model_id,
            self.lora_id,
            self.prompt_template_version,
            self.document_id,
            self.chunk_type.value,
            self.chunk_id,
            self.content_hash,
        ]
        return "|".join(parts)

    @classmethod
    def for_document(
        cls,
        *,
        model_id: str,
        lora_id: str,
        prompt_template_version: str,
        document_id: str,
        chunk_type: CacheChunkType,
        chunk_id: str,
        content_hash: str = "",
    ) -> "KVCacheKey":
        return cls(
            model_id=model_id,
            lora_id=lora_id,
            prompt_template_version=prompt_template_version,
            document_id=document_id,
            chunk_type=chunk_type,
            chunk_id=chunk_id,
            content_hash=content_hash,
        )

    @property
    def restaurant_id(self) -> str:
        return self.document_id


@dataclass(frozen=True, slots=True)
class ChunkRef:
    key: KVCacheKey
    shard_uri: str
    byte_offset: int
    byte_length: int
    token_count: int
    dtype: str
    layout_version: str
    checksum: str
    storage_layout: KVStorageLayout | str = KVStorageLayout.SEPARATE_KEY_VALUE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "storage_layout",
            kv_storage_layout_from_value(self.storage_layout, field_name="storage_layout"),
        )
        if not _is_non_empty_string(self.shard_uri):
            raise ValueError("shard_uri must be non-empty")
        if type(self.byte_offset) is not int:
            raise ValueError("byte_offset must be an integer")
        if self.byte_offset < 0:
            raise ValueError("byte_offset must be non-negative")
        if type(self.byte_length) is not int:
            raise ValueError("byte_length must be an integer")
        if self.byte_length <= 0:
            raise ValueError("byte_length must be positive")
        if type(self.token_count) is not int:
            raise ValueError("token_count must be an integer")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if not _is_non_empty_string(self.dtype):
            raise ValueError("dtype must be non-empty")
        if not _is_non_empty_string(self.layout_version):
            raise ValueError("layout_version must be non-empty")
        if not _is_sha256_hex(self.checksum):
            raise ValueError("checksum must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class DocumentKVRequest:
    request_id: str
    task_id: str
    model_id: str
    lora_id: str
    prompt_template_version: str
    document_chunks: DocumentChunkMap
    include_static: bool = True
    task_prefix_id: str | None = None

    def __post_init__(self) -> None:
        _validate_request_metadata(
            request_id=self.request_id,
            task_id=self.task_id,
            model_id=self.model_id,
            lora_id=self.lora_id,
            prompt_template_version=self.prompt_template_version,
            include_static=self.include_static,
            task_prefix_id=self.task_prefix_id,
        )
        object.__setattr__(
            self,
            "document_chunks",
            _normalize_chunk_map("document_chunks", self.document_chunks),
        )

    @classmethod
    def for_text_document(
        cls,
        *,
        request_id: str,
        task_id: str,
        model_id: str,
        lora_id: str,
        prompt_template_version: str,
        document_id: str,
        chunk_id: ChunkId = "document",
        task_prefix_id: str | None = None,
    ) -> "DocumentKVRequest":
        return cls.for_document_chunks(
            request_id=request_id,
            task_id=task_id,
            model_id=model_id,
            lora_id=lora_id,
            prompt_template_version=prompt_template_version,
            document_id=document_id,
            chunk_ids=(chunk_id,),
            include_static=False,
            task_prefix_id=task_prefix_id,
        )

    @classmethod
    def for_document_chunks(
        cls,
        *,
        request_id: str,
        task_id: str,
        model_id: str,
        lora_id: str,
        prompt_template_version: str,
        document_id: str,
        chunk_ids: Sequence[ChunkId],
        include_static: bool = True,
        task_prefix_id: str | None = None,
    ) -> "DocumentKVRequest":
        return cls(
            request_id=request_id,
            task_id=task_id,
            model_id=model_id,
            lora_id=lora_id,
            prompt_template_version=prompt_template_version,
            document_chunks={document_id: chunk_ids},
            include_static=include_static,
            task_prefix_id=task_prefix_id,
        )

    @property
    def selected_document_ids(self) -> tuple[str, ...]:
        return tuple(self.document_chunks.keys())

    @property
    def selected_documents(self) -> tuple[str, ...]:
        return self.selected_document_ids


@dataclass(frozen=True, slots=True)
class RestaurantKVRequest:
    request_id: str
    task_id: str
    model_id: str
    lora_id: str
    prompt_template_version: str
    restaurant_reviews: DocumentChunkMap
    include_static: bool = True
    task_prefix_id: str | None = None

    def __post_init__(self) -> None:
        _validate_request_metadata(
            request_id=self.request_id,
            task_id=self.task_id,
            model_id=self.model_id,
            lora_id=self.lora_id,
            prompt_template_version=self.prompt_template_version,
            include_static=self.include_static,
            task_prefix_id=self.task_prefix_id,
        )
        object.__setattr__(
            self,
            "restaurant_reviews",
            _normalize_chunk_map("restaurant_reviews", self.restaurant_reviews),
        )

    @property
    def document_chunks(self) -> DocumentChunkMap:
        return self.restaurant_reviews

    @property
    def selected_document_ids(self) -> tuple[str, ...]:
        return tuple(self.restaurant_reviews.keys())

    @property
    def selected_documents(self) -> tuple[str, ...]:
        return self.selected_document_ids


@dataclass(frozen=True, slots=True)
class PlanSegment:
    ref: ChunkRef
    output_token_start: int
    output_byte_start: int

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ChunkRef):
            raise TypeError("ref must be a ChunkRef")
        _validate_non_negative_integer("output_token_start", self.output_token_start)
        _validate_non_negative_integer("output_byte_start", self.output_byte_start)


@dataclass(frozen=True, slots=True, init=False)
class MaterializationPlan:
    request: DocumentKVRequest | RestaurantKVRequest
    segments: tuple[PlanSegment, ...]
    total_tokens: int
    total_bytes: int
    selected_document_ids: tuple[str, ...] = field(default_factory=tuple)

    def __init__(
        self,
        request: DocumentKVRequest | RestaurantKVRequest,
        segments: tuple[PlanSegment, ...],
        total_tokens: int,
        total_bytes: int,
        selected_document_ids: Sequence[str] = (),
        *,
        selected_restaurants: Sequence[str] | None = None,
    ) -> None:
        if selected_restaurants is not None:
            selected_document_ids = selected_restaurants
        if not isinstance(segments, tuple):
            raise TypeError("segments must be a tuple of PlanSegment")
        _validate_non_negative_integer("total_tokens", total_tokens)
        _validate_non_negative_integer("total_bytes", total_bytes)
        _validate_plan_segment_cursors(segments, total_tokens=total_tokens, total_bytes=total_bytes)
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "total_tokens", total_tokens)
        object.__setattr__(self, "total_bytes", total_bytes)
        object.__setattr__(self, "selected_document_ids", tuple(selected_document_ids))

    @property
    def chunk_count(self) -> int:
        return len(self.segments)

    @property
    def selected_documents(self) -> tuple[str, ...]:
        return self.selected_document_ids

    @property
    def selected_restaurants(self) -> tuple[str, ...]:
        return self.selected_document_ids


def _resolve_document_id(*, document_id: str | None, restaurant_id: str | None) -> str:
    if restaurant_id is not None:
        return restaurant_id
    if document_id is not None:
        return document_id
    raise TypeError("document_id is required")


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _validate_storage_key_part(name: str, value: object) -> None:
    if not _is_non_empty_string(value):
        raise ValueError(f"{name} must be non-empty")
    if "|" in value:
        raise ValueError(f"{name} must not contain '|'")


def _validate_optional_storage_key_part(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if "|" in value:
        raise ValueError(f"{name} must not contain '|'")


def _validate_non_negative_integer(name: str, value: int) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_request_metadata(
    *,
    request_id: str,
    task_id: str,
    model_id: str,
    lora_id: str,
    prompt_template_version: str,
    include_static: bool,
    task_prefix_id: str | None,
) -> None:
    for name, value in (
        ("request_id", request_id),
        ("task_id", task_id),
        ("model_id", model_id),
        ("lora_id", lora_id),
        ("prompt_template_version", prompt_template_version),
    ):
        if not _is_non_empty_string(value):
            raise ValueError(f"{name} must be non-empty")
    if type(include_static) is not bool:
        raise ValueError("include_static must be a boolean")
    if task_prefix_id is not None and not _is_non_empty_string(task_prefix_id):
        raise ValueError("task_prefix_id must be non-empty when provided")


def _normalize_chunk_map(name: str, value: DocumentChunkMap) -> NormalizedDocumentChunkMap:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return FrozenDocumentChunkMap(value, field_name=name)


def _normalize_chunk_map_items(
    name: str,
    items: Iterable[tuple[str, Sequence[ChunkId]]],
) -> dict[str, tuple[ChunkId, ...]]:
    normalized: dict[str, tuple[ChunkId, ...]] = {}
    for document_id, chunk_ids in items:
        if not _is_non_empty_string(document_id):
            raise ValueError(f"{name} keys must be non-empty strings")
        if isinstance(chunk_ids, (str, bytes, bytearray, memoryview)) or not isinstance(chunk_ids, Sequence):
            raise TypeError(f"{name} values must be sequences of chunk ids")
        for chunk_id in chunk_ids:
            _validate_chunk_id(name, chunk_id)
        normalized[document_id] = tuple(chunk_ids)
    return normalized


def _validate_chunk_id(name: str, value: object) -> None:
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{name} chunk ids must be non-empty")
        return
    if type(value) is int:
        return
    if value is None:
        raise ValueError(f"{name} chunk ids must not be None")
    raise TypeError(f"{name} chunk ids must be strings or integers")


def _validate_plan_segment_cursors(
    segments: tuple[PlanSegment, ...],
    *,
    total_tokens: int,
    total_bytes: int,
) -> None:
    token_cursor = 0
    byte_cursor = 0
    for segment in segments:
        if not isinstance(segment, PlanSegment):
            raise TypeError("segments entries must be PlanSegment")
        chunk_id = segment.ref.key.chunk_id
        if segment.output_token_start != token_cursor:
            raise ValueError(
                f"Plan segment {chunk_id} output_token_start {segment.output_token_start} "
                f"!= expected token cursor {token_cursor}"
            )
        if segment.output_byte_start != byte_cursor:
            raise ValueError(
                f"Plan segment {chunk_id} output_byte_start {segment.output_byte_start} "
                f"!= expected byte cursor {byte_cursor}"
            )
        token_cursor += segment.ref.token_count
        byte_cursor += segment.ref.byte_length
    if token_cursor != total_tokens:
        raise ValueError(f"Plan segment token counts {token_cursor} != total_tokens {total_tokens}")
    if byte_cursor != total_bytes:
        raise ValueError(f"Plan segment byte lengths {byte_cursor} != total_bytes {total_bytes}")


def _is_sha256_hex(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


DOCUMENT_CHUNK_TYPES = CacheChunkTypeSet(
    task_prefix=DocumentChunkType.TASK_PREFIX,
    static=DocumentChunkType.DOCUMENT_STATIC,
    content=DocumentChunkType.DOCUMENT_CHUNK,
)
LEGACY_RESTAURANT_CHUNK_TYPES = CacheChunkTypeSet(
    task_prefix=ChunkType.TASK_PREFIX,
    static=ChunkType.RESTAURANT_STATIC,
    content=ChunkType.REVIEW,
)

_CHUNK_TYPE_ROLES = {
    ChunkType.TASK_PREFIX.value: DocumentChunkRole.TASK_PREFIX,
    DocumentChunkType.TASK_PREFIX.value: DocumentChunkRole.TASK_PREFIX,
    ChunkType.RESTAURANT_STATIC.value: DocumentChunkRole.STATIC,
    DocumentChunkType.DOCUMENT_STATIC.value: DocumentChunkRole.STATIC,
    ChunkType.REVIEW.value: DocumentChunkRole.CONTENT,
    DocumentChunkType.DOCUMENT_CHUNK.value: DocumentChunkRole.CONTENT,
}
_CHUNK_ROLE_SORT_ORDER = {
    DocumentChunkRole.TASK_PREFIX: 0,
    DocumentChunkRole.STATIC: 1,
    DocumentChunkRole.CONTENT: 2,
    DocumentChunkRole.OTHER: 99,
}


def chunk_type_role(chunk_type: CacheChunkType | str) -> DocumentChunkRole:
    return _CHUNK_TYPE_ROLES.get(_chunk_type_value(chunk_type), DocumentChunkRole.OTHER)


def chunk_type_sort_order(chunk_type: CacheChunkType | str) -> int:
    return _CHUNK_ROLE_SORT_ORDER[chunk_type_role(chunk_type)]


def chunk_types_for_request(request: DocumentKVRequest | RestaurantKVRequest) -> CacheChunkTypeSet:
    return LEGACY_RESTAURANT_CHUNK_TYPES if isinstance(request, RestaurantKVRequest) else DOCUMENT_CHUNK_TYPES


def _chunk_type_value(chunk_type: CacheChunkType | str) -> str:
    if isinstance(chunk_type, (ChunkType, DocumentChunkType)):
        return chunk_type.value
    if isinstance(chunk_type, str):
        return chunk_type
    raise TypeError("chunk_type must be a CacheChunkType or string")
