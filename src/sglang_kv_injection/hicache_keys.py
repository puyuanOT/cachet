"""SGLang HiCache page-key helpers for Cachet handoff metadata."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from hashlib import sha256
from typing import TypeAlias

SGLangTokenId: TypeAlias = int | Sequence[int]
SGLANG_HICACHE_HASH_HEX_LENGTH = 64
_UINT32_LIMIT = 2**32

__all__ = [
    "SGLangTokenId",
    "SGLANG_HICACHE_HASH_HEX_LENGTH",
    "sglang_hicache_page_hash",
    "sglang_hicache_page_keys",
]


def sglang_hicache_page_hash(
    token_ids: Iterable[SGLangTokenId],
    *,
    prior_hash: str | None = None,
) -> str:
    """Return SGLang's chained SHA-256 hash for one HiCache page."""

    hasher = sha256()
    if prior_hash is not None:
        _validate_hash_hex(prior_hash, field_name="prior_hash")
        hasher.update(bytes.fromhex(prior_hash))
    for index, token_id in enumerate(token_ids):
        _update_token_hash(hasher, token_id, field_name=f"token_ids[{index}]")
    return hasher.hexdigest()


def sglang_hicache_page_keys(
    token_ids: Sequence[SGLangTokenId],
    *,
    page_size: int,
    prior_hash: str | None = None,
) -> tuple[str, ...]:
    """Return the ordered SGLang HiCache page keys for token ids."""

    if isinstance(token_ids, (str, bytes, bytearray)):
        raise TypeError("token_ids must be a sequence of token ids")
    if type(page_size) is not int or page_size <= 0:
        raise ValueError("page_size must be a positive integer")
    keys: list[str] = []
    last_hash = prior_hash
    for start in range(0, len(token_ids), page_size):
        last_hash = sglang_hicache_page_hash(
            token_ids[start : start + page_size],
            prior_hash=last_hash,
        )
        keys.append(last_hash)
    return tuple(keys)


def _update_token_hash(hasher: object, token_id: SGLangTokenId, *, field_name: str) -> None:
    if isinstance(token_id, Sequence) and not isinstance(token_id, (str, bytes, bytearray)):
        for index, item in enumerate(token_id):
            _update_scalar_token_hash(hasher, item, field_name=f"{field_name}[{index}]")
        return
    _update_scalar_token_hash(hasher, token_id, field_name=field_name)


def _update_scalar_token_hash(hasher: object, token_id: object, *, field_name: str) -> None:
    if type(token_id) is not int or token_id < 0 or token_id >= _UINT32_LIMIT:
        raise ValueError(f"{field_name} must be an unsigned 32-bit integer")
    update = getattr(hasher, "update")
    update(token_id.to_bytes(4, byteorder="little", signed=False))


def _validate_hash_hex(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != SGLANG_HICACHE_HASH_HEX_LENGTH:
        raise ValueError(f"{field_name} must be a 64-character hex string")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a 64-character hex string") from exc
