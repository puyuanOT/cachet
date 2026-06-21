from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

from vllm_kv_injection.protocol import KVSegment


SegmentKey = tuple[int, str, str, str, str]


@dataclass(frozen=True, slots=True)
class BlockSpan:
    block_id: int
    token_start: int
    token_count: int
    block_offset: int


def plan_token_blocks(*, total_tokens: int, block_size: int, starting_block_id: int = 0) -> tuple[BlockSpan, ...]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")

    spans: list[BlockSpan] = []
    cursor = 0
    block_id = starting_block_id
    while cursor < total_tokens:
        token_count = min(block_size, total_tokens - cursor)
        spans.append(BlockSpan(block_id=block_id, token_start=cursor, token_count=token_count, block_offset=0))
        cursor += token_count
        block_id += 1
    return tuple(spans)


def map_segments_to_blocks(
    segments: tuple[KVSegment, ...],
    *,
    block_size: int,
    starting_block_id: int = 0,
) -> "SegmentBlockMapping":
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    mapping: dict[SegmentKey, list[BlockSpan]] = {}
    chunk_id_aliases: dict[str, SegmentKey] = {}
    ambiguous_aliases: set[str] = set()
    for segment_index, segment in enumerate(segments):
        segment_key = (
            segment_index,
            segment.document_id,
            segment.chunk_type,
            segment.chunk_id,
            segment.content_hash,
        )
        if segment.chunk_id in chunk_id_aliases:
            ambiguous_aliases.add(segment.chunk_id)
        else:
            chunk_id_aliases[segment.chunk_id] = segment_key
        cursor = segment.token_start
        end = segment.token_end
        while cursor < end:
            block_index = cursor // block_size
            block_start = block_index * block_size
            block_offset = cursor - block_start
            token_count = min(end - cursor, block_size - block_offset)
            mapping.setdefault(segment_key, []).append(
                BlockSpan(
                    block_id=starting_block_id + block_index,
                    token_start=cursor,
                    token_count=token_count,
                    block_offset=block_offset,
                )
            )
            cursor += token_count
    return SegmentBlockMapping(
        {key: tuple(value) for key, value in mapping.items()},
        chunk_id_aliases={key: value for key, value in chunk_id_aliases.items() if key not in ambiguous_aliases},
    )


def map_segments_to_reserved_blocks(
    segments: tuple[KVSegment, ...],
    blocks: tuple[BlockSpan, ...],
) -> "SegmentBlockMapping":
    _validate_reserved_blocks(blocks)
    mapping: dict[SegmentKey, list[BlockSpan]] = {}
    chunk_id_aliases: dict[str, SegmentKey] = {}
    ambiguous_aliases: set[str] = set()
    for segment_index, segment in enumerate(segments):
        segment_key = (
            segment_index,
            segment.document_id,
            segment.chunk_type,
            segment.chunk_id,
            segment.content_hash,
        )
        if segment.chunk_id in chunk_id_aliases:
            ambiguous_aliases.add(segment.chunk_id)
        else:
            chunk_id_aliases[segment.chunk_id] = segment_key
        cursor = segment.token_start
        while cursor < segment.token_end:
            block = _block_for_token(blocks, cursor)
            block_end = block.token_start + block.token_count
            token_count = min(segment.token_end - cursor, block_end - cursor)
            mapping.setdefault(segment_key, []).append(
                BlockSpan(
                    block_id=block.block_id,
                    token_start=cursor,
                    token_count=token_count,
                    block_offset=block.block_offset + cursor - block.token_start,
                )
            )
            cursor += token_count
    return SegmentBlockMapping(
        {key: tuple(value) for key, value in mapping.items()},
        chunk_id_aliases={key: value for key, value in chunk_id_aliases.items() if key not in ambiguous_aliases},
    )


class SegmentBlockMapping(dict[SegmentKey | str, tuple[BlockSpan, ...]]):
    def __init__(
        self,
        mapping: dict[SegmentKey, tuple[BlockSpan, ...]],
        *,
        chunk_id_aliases: dict[str, SegmentKey],
    ) -> None:
        super().__init__(mapping)
        self._segment_keys = tuple(mapping)
        self._chunk_id_aliases = dict(chunk_id_aliases)

    def __getitem__(self, key: SegmentKey | str) -> tuple[BlockSpan, ...]:
        if isinstance(key, str):
            key = self._chunk_id_aliases[key]
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            return key in self._chunk_id_aliases
        return super().__contains__(key)

    def get(
        self,
        key: SegmentKey | str,
        default: tuple[BlockSpan, ...] | None = None,
    ) -> tuple[BlockSpan, ...] | None:
        try:
            return self[key]
        except KeyError:
            return default

    def segment_keys(self) -> tuple[SegmentKey, ...]:
        return self._segment_keys

    def unique_chunk_ids(self) -> Iterator[str]:
        yield from self._chunk_id_aliases


def _validate_reserved_blocks(blocks: tuple[BlockSpan, ...]) -> None:
    cursor = 0
    for block in blocks:
        if block.token_start != cursor:
            raise ValueError("Reserved blocks must cover a contiguous logical token range")
        if block.token_count < 0:
            raise ValueError("Reserved block token_count must be non-negative")
        if block.block_offset < 0:
            raise ValueError("Reserved block block_offset must be non-negative")
        cursor += block.token_count


def _block_for_token(blocks: tuple[BlockSpan, ...], token: int) -> BlockSpan:
    for block in blocks:
        if block.token_start <= token < block.token_start + block.token_count:
            return block
    raise ValueError(f"No reserved block covers token {token}")
