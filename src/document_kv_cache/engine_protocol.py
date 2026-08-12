"""Engine protocol data structures for KV-cache serving handoffs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set as AbstractSet
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from document_kv_cache.artifact_identity import (
    ArtifactIdentity,
    RuntimeCompatibilityHandshake,
    RuntimeIdentity,
    TokenContract,
)

__all__ = [
    "DTYPE_BYTE_WIDTHS",
    "AttentionMechanism",
    "KVStorageLayout",
    "KVPayloadAxisOrder",
    "KVKeyPositionEncoding",
    "dtype_byte_width",
    "kv_storage_layout_from_value",
    "kv_payload_axis_order_from_value",
    "kv_key_position_encoding_from_value",
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
        "fp8_e4m3": 1,
        "fp8_e5m2": 1,
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


class KVPayloadAxisOrder(StrEnum):
    """Ordering of the outer axes in the serialized KV payload blob.

    ``token_major`` lays the payload out as ``[token, layer, K/V, kv_head,
    head_dim]`` so a token span is one contiguous read but a single layer is
    strided. ``layer_major`` lays it out as ``[layer, token, K/V, kv_head,
    head_dim]`` so a single layer's token span is one contiguous read, which is
    what enables LMCache-style per-layer streaming (overlap a layer's disk read
    and host->device copy with the previous layer's attention compute).
    """

    TOKEN_MAJOR = "token_major"
    LAYER_MAJOR = "layer_major"


class KVKeyPositionEncoding(StrEnum):
    """How keys in an artifact encode rotary positions.

    ``stored_post_rope`` is an ordinary prefix cache and cannot be moved.
    ``pre_rope`` stores keys before RoPE and applies their absolute runtime
    positions during injection.
    """

    STORED_POST_ROPE = "stored_post_rope"
    PRE_ROPE = "pre_rope"


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
    payload_axis_order: KVPayloadAxisOrder | str = KVPayloadAxisOrder.TOKEN_MAJOR
    # Pre-RoPE keys: when True the stored K is captured before rotary position
    # embedding (post QK-norm for Qwen3), so a cached chunk is position-independent
    # and must be re-roped at its true absolute offset during injection. rope_theta
    # (and optionally rope_rotary_dim, defaulting to head_size) make the payload
    # self-describing so the injector needs no external model handle. Default False
    # preserves the legacy post-RoPE behavior.
    pre_rope: bool = False
    rope_theta: float | None = None
    rope_rotary_dim: int | None = None
    key_position_encoding: KVKeyPositionEncoding | str | None = None

    def __post_init__(self) -> None:
        storage_layout: KVStorageLayout | str
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
        object.__setattr__(
            self,
            "payload_axis_order",
            kv_payload_axis_order_from_value(self.payload_axis_order, field_name="payload_axis_order"),
        )
        if self.key_position_encoding is None:
            key_position_encoding = (
                KVKeyPositionEncoding.PRE_ROPE
                if self.pre_rope
                else KVKeyPositionEncoding.STORED_POST_ROPE
            )
        else:
            key_position_encoding = kv_key_position_encoding_from_value(
                self.key_position_encoding,
                field_name="key_position_encoding",
            )
        if self.pre_rope and key_position_encoding != KVKeyPositionEncoding.PRE_ROPE:
            raise ValueError(
                "pre_rope=True requires key_position_encoding='pre_rope'"
            )
        if key_position_encoding == KVKeyPositionEncoding.PRE_ROPE:
            object.__setattr__(self, "pre_rope", True)
        object.__setattr__(self, "key_position_encoding", key_position_encoding)

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
        assert self.num_kv_heads is not None
        assert self.kv_stride_bytes is not None
        return self.num_layers * self.num_kv_heads * self.kv_stride_bytes * 2

    @property
    def requires_rope_repositioning(self) -> bool:
        return self.key_position_encoding == KVKeyPositionEncoding.PRE_ROPE

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
        if type(self.pre_rope) is not bool:
            raise ValueError("pre_rope must be a boolean")
        if not isinstance(self.key_position_encoding, KVKeyPositionEncoding):
            raise TypeError("key_position_encoding must be a KVKeyPositionEncoding")
        if self.pre_rope != (
            self.key_position_encoding == KVKeyPositionEncoding.PRE_ROPE
        ):
            raise ValueError(
                "pre_rope must agree with key_position_encoding='pre_rope'"
            )
        if self.requires_rope_repositioning and not (
            isinstance(self.rope_theta, (int, float))
            and not isinstance(self.rope_theta, bool)
            and self.rope_theta > 0
        ):
            raise ValueError(
                "repositionable RoPE keys require a positive rope_theta"
            )
        _validate_optional_positive_integer("rope_rotary_dim", self.rope_rotary_dim)
        if self.rope_rotary_dim is not None and self.rope_rotary_dim % 2:
            raise ValueError("rope_rotary_dim must be even")
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
            dtype_width = dtype_byte_width(self.dtype)
            minimum_stride_bytes = self.head_size * dtype_width
            if self.kv_stride_bytes < minimum_stride_bytes:
                raise ValueError(
                    f"kv_stride_bytes {self.kv_stride_bytes} is smaller than "
                    f"head_size * dtype width {minimum_stride_bytes}"
                )
            if self.kv_stride_bytes % dtype_width != 0:
                raise ValueError(
                    f"kv_stride_bytes {self.kv_stride_bytes} must be a multiple of dtype width {dtype_width}"
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


def kv_payload_axis_order_from_value(
    value: KVPayloadAxisOrder | str,
    *,
    field_name: str = "payload_axis_order",
) -> KVPayloadAxisOrder:
    if isinstance(value, str):
        value = value.strip().lower()
    try:
        return value if isinstance(value, KVPayloadAxisOrder) else KVPayloadAxisOrder(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported {field_name} {value!r}") from exc


def kv_key_position_encoding_from_value(
    value: KVKeyPositionEncoding | str,
    *,
    field_name: str = "key_position_encoding",
) -> KVKeyPositionEncoding:
    if isinstance(value, str):
        value = value.strip().lower()
    try:
        return (
            value
            if isinstance(value, KVKeyPositionEncoding)
            else KVKeyPositionEncoding(value)
        )
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
    token_contract: TokenContract | None = None

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
        if self.token_contract is not None:
            if not isinstance(self.token_contract, TokenContract):
                raise TypeError("segment.token_contract must be a TokenContract or None")
            if self.token_contract.token_count != self.token_count:
                raise ValueError("segment.token_contract token_count must match segment.token_count")


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
    artifact_identity: ArtifactIdentity | None = None
    payload_checksum: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(_validated_string_mapping("metadata", self.metadata)))
        object.__setattr__(self, "adapter_ids", _normalized_adapter_ids(self.adapter_ids))
        if self.artifact_identity is not None and not isinstance(self.artifact_identity, ArtifactIdentity):
            raise TypeError("artifact_identity must be an ArtifactIdentity or None")

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
        if self.payload_checksum:
            _validate_sha256("payload_checksum", self.payload_checksum)
        if self.artifact_identity is not None:
            identity = self.artifact_identity
            if self.cache_method != identity.method_id:
                raise ValueError("cache_method must match artifact_identity.method_id")
            expected = {
                "model_id": self.layout.model_id,
                "lora_id": self.layout.lora_id,
                "layout_version": self.layout.layout_version,
                "runtime_kv_dtype": self.layout.dtype,
                "block_size": self.layout.block_size,
                "payload_axis_order": kv_payload_axis_order_from_value(
                    self.layout.payload_axis_order
                ).value,
                "key_position_encoding": (
                    self.layout.key_position_encoding.value
                ),
                "rope_theta": self.layout.rope_theta,
                "rope_rotary_dim": self.layout.rope_rotary_dim,
            }
            mismatches = [
                name
                for name, value in expected.items()
                if getattr(identity, name) != value
            ]
            if mismatches:
                raise ValueError(
                    "artifact_identity does not match KV layout: " + ", ".join(mismatches)
                )
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
            if segment.token_contract is not None and self.artifact_identity is not None:
                if segment.token_contract.tokenizer_id != self.artifact_identity.tokenizer_id:
                    raise ValueError("segment token_contract tokenizer_id does not match artifact_identity")
                if segment.token_contract.tokenizer_revision != self.artifact_identity.tokenizer_revision:
                    raise ValueError(
                        "segment token_contract tokenizer_revision does not match artifact_identity"
                    )
            token_cursor = segment.token_end
            byte_cursor = segment.byte_end
        if token_cursor != self.total_tokens:
            raise ValueError(f"Segment tokens {token_cursor} != total_tokens {self.total_tokens}")
        if byte_cursor != self.total_bytes:
            raise ValueError(f"Segment bytes {byte_cursor} != total_bytes {self.total_bytes}")

    def runtime_handshake(
        self,
        runtime: RuntimeIdentity,
        *,
        reject_unresolved: bool = True,
    ) -> RuntimeCompatibilityHandshake:
        if self.artifact_identity is None:
            raise ValueError("KV handle does not include an artifact_identity")
        return RuntimeCompatibilityHandshake.compare(
            self.artifact_identity,
            runtime,
            reject_unresolved=reject_unresolved,
        )


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


def _validate_sha256(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


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
