from hashlib import sha256

import pytest

from sglang_kv_injection.hicache_keys import (
    SGLANG_HICACHE_HASH_HEX_LENGTH,
    sglang_hicache_page_hash,
    sglang_hicache_page_keys,
)


def test_sglang_hicache_page_hash_matches_upstream_little_endian_tokens():
    expected = sha256((1).to_bytes(4, "little") + (257).to_bytes(4, "little")).hexdigest()

    assert sglang_hicache_page_hash([1, 257]) == expected
    assert len(expected) == SGLANG_HICACHE_HASH_HEX_LENGTH


def test_sglang_hicache_page_keys_chain_prior_page_hashes():
    first = sha256((1).to_bytes(4, "little") + (2).to_bytes(4, "little")).hexdigest()
    second = sha256(bytes.fromhex(first) + (3).to_bytes(4, "little")).hexdigest()

    assert sglang_hicache_page_keys([1, 2, 3], page_size=2) == (first, second)


def test_sglang_hicache_page_hash_supports_bigram_token_ids():
    expected = sha256((1).to_bytes(4, "little") + (2).to_bytes(4, "little")).hexdigest()

    assert sglang_hicache_page_hash([(1, 2)]) == expected


def test_sglang_hicache_page_keys_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="page_size must be a positive integer"):
        sglang_hicache_page_keys([1], page_size=0)

    with pytest.raises(ValueError, match="token_ids\\[0\\] must be an unsigned 32-bit integer"):
        sglang_hicache_page_hash([-1])

    with pytest.raises(ValueError, match="prior_hash must be a 64-character hex string"):
        sglang_hicache_page_hash([1], prior_hash="not-hex")
