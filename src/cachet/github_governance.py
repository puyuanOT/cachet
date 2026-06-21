"""Cachet facade for :mod:`document_kv_cache.github_governance`."""

from __future__ import annotations

from cachet._module_alias import install as _install
from document_kv_cache.github_governance import *  # noqa: F401,F403
from document_kv_cache.github_governance import __all__ as __all__

if __name__ == "__main__":
    from cachet._module_alias import run_main as _run_main

    raise SystemExit(_run_main("document_kv_cache.github_governance"))

_install(__name__, "document_kv_cache.github_governance")
