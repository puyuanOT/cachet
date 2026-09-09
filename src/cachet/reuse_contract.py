"""Cachet facade for :mod:`document_kv_cache.reuse_contract`."""

from __future__ import annotations

from cachet._module_alias import install as _install
from document_kv_cache.reuse_contract import *  # noqa: F401,F403
from document_kv_cache.reuse_contract import __all__ as __all__

_install(__name__, "document_kv_cache.reuse_contract")
