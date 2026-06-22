"""Document-owned CLI wrapper for Cachet's vLLM native-runtime preflight."""

from __future__ import annotations

from vllm_kv_injection.vllm_runtime_preflight import *  # noqa: F401,F403
from vllm_kv_injection.vllm_runtime_preflight import __all__ as __all__


if __name__ == "__main__":
    from vllm_kv_injection.vllm_runtime_preflight import main

    raise SystemExit(main())
