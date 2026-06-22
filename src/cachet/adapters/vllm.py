"""Cachet facade for :mod:`vllm_kv_injection`."""

from __future__ import annotations

from cachet._module_alias import install as _install
from vllm_kv_injection import __all__ as __all__

_install(__name__, "vllm_kv_injection")
