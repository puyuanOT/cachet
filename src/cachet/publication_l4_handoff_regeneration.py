"""Cachet facade for single-GPU L4 publication handoff regeneration."""

from __future__ import annotations

from cachet._module_alias import install as _install
from document_kv_cache.publication_l4_handoff_regeneration import *  # noqa: F401,F403
from document_kv_cache.publication_l4_handoff_regeneration import __all__ as __all__

_install(__name__, "document_kv_cache.publication_l4_handoff_regeneration")
