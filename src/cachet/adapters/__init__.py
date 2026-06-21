"""Cachet-branded engine adapter aliases."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["vllm", "sglang"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
