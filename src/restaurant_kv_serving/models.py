from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping, Sequence


class ChunkType(StrEnum):
    TASK_PREFIX = "task_prefix"
    RESTAURANT_STATIC = "restaurant_static"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class KVCacheKey:
    model_id: str
    lora_id: str
    prompt_template_version: str
    restaurant_id: str
    chunk_type: ChunkType
    chunk_id: str
    content_hash: str = ""

    def storage_key(self) -> str:
        parts = [
            self.model_id,
            self.lora_id,
            self.prompt_template_version,
            self.restaurant_id,
            self.chunk_type.value,
            self.chunk_id,
            self.content_hash,
        ]
        return "|".join(parts)


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


@dataclass(frozen=True, slots=True)
class RestaurantKVRequest:
    request_id: str
    task_id: str
    model_id: str
    lora_id: str
    prompt_template_version: str
    restaurant_reviews: Mapping[str, Sequence[str]]
    include_static: bool = True
    task_prefix_id: str | None = None


@dataclass(frozen=True, slots=True)
class PlanSegment:
    ref: ChunkRef
    output_token_start: int
    output_byte_start: int


@dataclass(frozen=True, slots=True)
class MaterializationPlan:
    request: RestaurantKVRequest
    segments: tuple[PlanSegment, ...]
    total_tokens: int
    total_bytes: int
    selected_restaurants: tuple[str, ...] = field(default_factory=tuple)

    @property
    def chunk_count(self) -> int:
        return len(self.segments)

