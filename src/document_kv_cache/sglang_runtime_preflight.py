"""Document-owned CLI wrapper for Cachet's SGLang native-runtime preflight."""

from __future__ import annotations

from sglang_kv_injection.sglang_runtime_preflight import *  # noqa: F401,F403
from sglang_kv_injection.sglang_runtime_preflight import __all__ as __all__


if __name__ == "__main__":
    from sglang_kv_injection.sglang_runtime_preflight import main

    raise SystemExit(main())
