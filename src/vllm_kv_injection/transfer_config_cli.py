"""CLI entrypoint for writing document KV vLLM transfer configs."""

from __future__ import annotations

from vllm_kv_injection.vllm_transfer_config import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
