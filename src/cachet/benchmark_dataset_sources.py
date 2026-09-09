"""Cachet facade for :mod:`document_kv_cache.benchmark_dataset_sources`."""

from __future__ import annotations

from cachet._module_alias import install as _install
from document_kv_cache.benchmark_dataset_sources import *  # noqa: F401,F403
from document_kv_cache.benchmark_dataset_sources import __all__ as __all__

if __name__ == "__main__":  # pragma: no cover
    from cachet._module_alias import run_main as _run_main

    raise SystemExit(_run_main("document_kv_cache.benchmark_dataset_sources"))

_install(__name__, "document_kv_cache.benchmark_dataset_sources")
