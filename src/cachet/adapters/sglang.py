"""Cachet facade for :mod:`sglang_kv_injection`."""

from __future__ import annotations

from cachet._module_alias import install as _install
from sglang_kv_injection import *  # noqa: F401,F403
from sglang_kv_injection import __all__ as __all__

_install(__name__, "sglang_kv_injection")
