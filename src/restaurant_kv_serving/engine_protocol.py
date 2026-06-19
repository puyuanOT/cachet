from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType


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
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not self.lora_id:
            raise ValueError("lora_id must be non-empty")
        if not self.layout_version:
            raise ValueError("layout_version must be non-empty")
        if not self.dtype:
            raise ValueError("dtype must be non-empty")
        dtype_byte_width(self.dtype)
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.bytes_per_token <= 0:
            raise ValueError("bytes_per_token must be positive")
        if self.num_query_heads is not None and self.num_query_heads <= 0:
            raise ValueError("num_query_heads must be positive")
        if self.num_kv_heads is not None and self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if self.head_size is not None and self.head_size <= 0:
            raise ValueError("head_size must be positive")
        if self.kv_stride_bytes is not None and self.kv_stride_bytes <= 0:
            raise ValueError("kv_stride_bytes must be positive")
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

    def validate(self) -> None:
        self.layout.validate()
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.handle_uri:
            raise ValueError("handle_uri must be non-empty")
        if self.total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        if self.total_bytes < 0:
            raise ValueError("total_bytes must be non-negative")
        if not self.cache_method:
            raise ValueError("cache_method must be non-empty")
        token_cursor = 0
        byte_cursor = 0
        for segment in self.segments:
            if segment.token_start != token_cursor:
                raise ValueError(f"Non-contiguous token segment {segment.chunk_id}")
            if segment.byte_start != byte_cursor:
                raise ValueError(f"Non-contiguous byte segment {segment.chunk_id}")
            if segment.token_count < 0:
                raise ValueError(f"Segment {segment.chunk_id} token_count must be non-negative")
            if segment.byte_length < 0:
                raise ValueError(f"Segment {segment.chunk_id} byte_length must be non-negative")
            token_cursor = segment.token_end
            byte_cursor = segment.byte_end
        if token_cursor != self.total_tokens:
            raise ValueError(f"Segment tokens {token_cursor} != total_tokens {self.total_tokens}")
        if byte_cursor != self.total_bytes:
            raise ValueError(f"Segment bytes {byte_cursor} != total_bytes {self.total_bytes}")
