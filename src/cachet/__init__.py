"""Cachet-branded facade for :mod:`document_kv_cache`.

The canonical implementation and module identities remain under
``document_kv_cache``. This package gives users a premium brand root import
while preserving one set of public objects, protocols, and implementations.
"""

from __future__ import annotations

from typing import Any

import document_kv_cache as _document_package
from document_kv_cache import *  # noqa: F403

__all__ = list(_document_package.__all__)


def __getattr__(name: str) -> Any:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
