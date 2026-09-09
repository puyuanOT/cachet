"""Cachet facade for capability-gated BF16 latency handoff generation."""

from __future__ import annotations

from cachet._module_alias import install as _install
from document_kv_cache.publication_bf16_handoff_generation import *  # noqa: F401,F403
from document_kv_cache.publication_bf16_handoff_generation import __all__ as __all__

_install(__name__, "document_kv_cache.publication_bf16_handoff_generation")
