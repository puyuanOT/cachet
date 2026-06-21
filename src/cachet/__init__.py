"""Cachet-branded facade for :mod:`document_kv_cache`.

The canonical implementation and module identities remain under
``document_kv_cache``. This package gives users a premium brand root import and
``cachet.<module>`` aliases while preserving one set of public objects,
protocols, and implementations.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import document_kv_cache as _document_package
from document_kv_cache import *  # noqa: F403

__all__ = list(_document_package.__all__)
_PUBLIC_SUBMODULES = frozenset(getattr(_document_package, "_PUBLIC_SUBMODULES", ()))


def _document_submodule(name: str) -> Any:
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __getattr__(name: str) -> Any:
    if name in _PUBLIC_SUBMODULES:
        return _document_submodule(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__, *_PUBLIC_SUBMODULES})
