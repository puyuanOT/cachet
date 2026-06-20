"""Engine protocol data structures for KV-cache serving handoffs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set as AbstractSet
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

__all__ = [
    "DTYPE_BYTE_WIDTHS",
    "AttentionMechanism",
    "KVStorageLayout",
    "dtype_byte_width",
    "kv_storage_layout_from_value",
    "KVLayout",
    "KVSegment",
    "KVCacheHandle",
]


DTYPE_BYTE_WIDTHS: Mapping[str, int] = MappingProxyType(
    {
        "bf16": 2,
        "bfloat16": 2,
        "fp16": 2,
        "float16": 2,
        "fp32": 4,
        "float32": 4,
        "fp8": 1,
        "float8": 1,
        "int8": 1,
        "uint8": 1,
    }
)


class AttentionMechanism(StrEnum):
    MULTI_HEAD = "mha"
    GROUPED_QUERY = "gqa"
    MULTI_QUERY = "mqa"


class KVStorageLayout(StrEnum):
    SEPARATE_KEY_VALUE = "separate_key_value"
    INTERLEAVED_KEY_VALUE = "interleaved_key_value"
    SHARED_KEY_VALUE = "shared_key_value"


def dtype_byte_width(dtype: str) -> int:
    if not isinstance(dtype, str):
        raise ValueError("dtype must be a string")
    try:
        return DTYPE_BYTE_WIDTHS[dtype.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(DTYPE_BYTE_WIDTHS))
        raise ValueError(f"Unsupported KV dtype {dtype!r}; supported dtypes: {supported}") from exc


@dataclass(frozen=True, slots=True)
class KVLayout:
    """Model-specific KV tensor layout needed by serving engine adapters."""

    model_id: str
    lora_id: str
    layout_version: str
    dtype: str
    num_layers: int
    block_size: int
    bytes_per_token: int
    num_query_heads: int | None = None
    num_kv_heads: int | None = None
    head_size: int | None = None
    kv_stride_bytes: int | None = None
    shares_kv_storage: bool = False
    storage_layout: KVStorageLayout | str | None = None

    def __post_init__(self) -> None:
        if self.storage_layout is None:
            storage_layout = (
                KVStorageLayout.SHARED_KEY_VALUE if self.shares_kv_storage else KVStorageLayout.SEPARATE_KEY_VALUE
            )
        else:
            storage_layout = self.storage_layout
        object.__setattr__(
            self,
            "storage_layout",
            kv_storage_layout_from_value(storage_layout, field_name="storage_layout"),
        )

    @property
    def attention_mechanism(self) -> AttentionMechanism | None:
        if self.num_query_heads is None or self.num_kv_heads is None:
            return None
        if self.num_kv_heads == self.num_query_heads:
            return AttentionMechanism.MULTI_HEAD
        if self.num_kv_heads == 1:
            return AttentionMechanism.MULTI_QUERY
        return AttentionMechanism.GROUPED_QUERY

    @property
    def query_heads_per_kv_head(self) -> int | None:
        if self.num_query_heads is None or self.num_kv_heads is None:
            return None
        return self.num_query_heads // self.num_kv_heads

    @property
    def expected_bytes_per_token(self) -> int | None:
        attention_fields = (
            self.num_query_heads,
            self.num_kv_heads,
            self.head_size,
            self.kv_stride_bytes,
        )
        if any(value is None for value in attention_fields):
            return None
        return self.num_layers * self.num_kv_heads * self.head_size * 2 * dtype_byte_width(self.dtype)

    def validate(self) -> None:
        _validate_nonempty_string("model_id", self.model_id)
        _validate_nonempty_string("lora_id", self.lora_id)
        _validate_nonempty_string("layout_version", self.layout_version)
        _validate_nonempty_string("dtype", self.dtype)
        dtype_byte_width(self.dtype)
        _validate_positive_integer("num_layers", self.num_layers)
        _validate_positive_integer("block_size", self.block_size)
        _validate_positive_integer("bytes_per_token", self.bytes_per_token)
        _validate_optional_positive_integer("num_query_heads", self.num_query_heads)
        _validate_optional_positive_integer("num_kv_heads", self.num_kv_heads)
        _validate_optional_positive_integer("head_size", self.head_size)
        _validate_optional_positive_integer("kv_stride_bytes", self.kv_stride_bytes)
        if type(self.shares_kv_storage) is not bool:
            raise ValueError("shares_kv_storage must be a boolean")
        attention_fields = (
            self.num_query_heads,
            self.num_kv_heads,
            self.head_size,
            self.kv_stride_bytes,
        )
        if self.shares_kv_storage or any(value is not None for value in attention_fields):
            if any(value is None for value in attention_fields):
                raise ValueError(
                    "num_query_heads, num_kv_heads, head_size, and kv_stride_bytes are required together"
                )
        if self.shares_kv_storage and self.storage_layout != KVStorageLayout.SHARED_KEY_VALUE:
            raise ValueError("shares_kv_storage requires storage_layout='shared_key_value'")
        if self.storage_layout == KVStorageLayout.SHARED_KEY_VALUE and not self.shares_kv_storage:
            raise ValueError("storage_layout='shared_key_value' requires shares_kv_storage=True")
        if self.model_id == "qwen3:4b-instruct" and self.layout_version == "qwen3-v1":
            if self.shares_kv_storage is not True or self.storage_layout != KVStorageLayout.SHARED_KEY_VALUE:
                raise ValueError("qwen3-v1 layout requires shared K/V storage")
        if self.num_query_heads is not None and self.num_kv_heads is not None:
            if self.num_kv_heads > self.num_query_heads:
                raise ValueError("num_kv_heads cannot exceed num_query_heads")
            if self.num_query_heads % self.num_kv_heads != 0:
                raise ValueError("num_query_heads must be divisible by num_kv_heads")
        if self.head_size is not None and self.kv_stride_bytes is not None:
            expected_stride_bytes = self.head_size * dtype_byte_width(self.dtype)
            if self.kv_stride_bytes != expected_stride_bytes:
                raise ValueError(
                    f"kv_stride_bytes {self.kv_stride_bytes} does not match "
                    f"head_size * dtype width {expected_stride_bytes}"
                )
        expected_bytes_per_token = self.expected_bytes_per_token
        if expected_bytes_per_token is not None and self.bytes_per_token != expected_bytes_per_token:
            raise ValueError(
                f"bytes_per_token {self.bytes_per_token} does not match layout geometry "
                f"{expected_bytes_per_token}"
            )


def kv_storage_layout_from_value(
    value: KVStorageLayout | str,
    *,
    field_name: str = "kv_storage_layout",
) -> KVStorageLayout:
    if isinstance(value, str):
        value = value.strip().lower()
    try:
        return value if isinstance(value, KVStorageLayout) else KVStorageLayout(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported {field_name} {value!r}") from exc


@dataclass(frozen=True, slots=True)
class KVSegment:
    document_id: str
    chunk_type: str
    chunk_id: str
    token_start: int
    token_count: int
    byte_start: int
    byte_length: int
    content_hash: str = ""

    @property
    def token_end(self) -> int:
        return self.token_start + self.token_count

    @property
    def byte_end(self) -> int:
        return self.byte_start + self.byte_length

    def validate(self) -> None:
        _validate_nonempty_string("segment.document_id", self.document_id)
        _validate_nonempty_string("segment.chunk_type", self.chunk_type)
        _validate_nonempty_string("segment.chunk_id", self.chunk_id)
        _validate_nonnegative_integer("segment.token_start", self.token_start)
        _validate_nonnegative_integer("segment.token_count", self.token_count)
        _validate_nonnegative_integer("segment.byte_start", self.byte_start)
        _validate_nonnegative_integer("segment.byte_length", self.byte_length)
        if not isinstance(self.content_hash, str):
            raise ValueError("segment.content_hash must be a string")


@dataclass(frozen=True, slots=True)
class KVCacheHandle:
    request_id: str
    handle_uri: str
    layout: KVLayout
    segments: tuple[KVSegment, ...]
    total_tokens: int
    total_bytes: int
    metadata: Mapping[str, str] = field(default_factory=dict)
    cache_method: str = "vanilla_prefill"
    adapter_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(_validated_string_mapping("metadata", self.metadata)))
        object.__setattr__(self, "adapter_ids", _normalized_adapter_ids(self.adapter_ids))

    def validate(self) -> None:
        if not isinstance(self.layout, KVLayout):
            raise TypeError("layout must be a KVLayout")
        self.layout.validate()
        _validate_nonempty_string("request_id", self.request_id)
        _validate_nonempty_string("handle_uri", self.handle_uri)
        if not isinstance(self.segments, tuple):
            raise TypeError("segments must be a tuple of KVSegment")
        _validate_nonnegative_integer("total_tokens", self.total_tokens)
        _validate_nonnegative_integer("total_bytes", self.total_bytes)
        _validate_nonempty_string("cache_method", self.cache_method)
        token_cursor = 0
        byte_cursor = 0
        for segment in self.segments:
            if not isinstance(segment, KVSegment):
                raise TypeError("segments entries must be KVSegment")
            segment.validate()
            if segment.token_start != token_cursor:
                raise ValueError(f"Non-contiguous token segment {segment.chunk_id}")
            if segment.byte_start != byte_cursor:
                raise ValueError(f"Non-contiguous byte segment {segment.chunk_id}")
            token_cursor = segment.token_end
            byte_cursor = segment.byte_end
        if token_cursor != self.total_tokens:
            raise ValueError(f"Segment tokens {token_cursor} != total_tokens {self.total_tokens}")
        if byte_cursor != self.total_bytes:
            raise ValueError(f"Segment bytes {byte_cursor} != total_bytes {self.total_bytes}")


def _validate_nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty")


def _validate_nonnegative_integer(name: str, value: object) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_positive_integer(name: str, value: object) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_optional_positive_integer(name: str, value: object) -> None:
    if value is None:
        return
    _validate_positive_integer(name, value)


def _validated_string_mapping(name: str, value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    invalid = [
        key
        for key, item in value.items()
        if not isinstance(key, str) or not isinstance(item, str)
    ]
    if invalid:
        raise TypeError(f"{name} keys and values must be strings")
    return dict(value)


def _normalized_adapter_ids(adapter_ids: Iterable[str]) -> tuple[str, ...]:
    if (
        isinstance(adapter_ids, (str, bytes, bytearray, memoryview, Mapping, AbstractSet))
        or not isinstance(adapter_ids, Iterable)
    ):
        raise TypeError("adapter_ids must be an ordered iterable of strings")
    normalized: list[str] = []
    seen: set[str] = set()
    for adapter_id in adapter_ids:
        if not isinstance(adapter_id, str) or not adapter_id:
            raise ValueError("adapter_ids entries must be non-empty strings")
        if adapter_id in seen:
            raise ValueError("adapter_ids entries must be unique")
        seen.add(adapter_id)
        normalized.append(adapter_id)
    return tuple(normalized)
