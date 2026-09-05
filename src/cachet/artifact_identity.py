"""Cachet facade for :mod:`document_kv_cache.artifact_identity`."""

from __future__ import annotations

from cachet._module_alias import install as _install
from document_kv_cache.artifact_identity import *  # noqa: F401,F403
from document_kv_cache.artifact_identity import __all__ as __all__

_install(__name__, "document_kv_cache.artifact_identity")
