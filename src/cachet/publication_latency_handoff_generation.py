"""Cachet facade for latency handoff generation and serving reuse."""

from __future__ import annotations

from cachet._module_alias import install as _install
from document_kv_cache.publication_latency_handoff_generation import *  # noqa: F401,F403
from document_kv_cache.publication_latency_handoff_generation import __all__ as __all__

_install(__name__, "document_kv_cache.publication_latency_handoff_generation")
