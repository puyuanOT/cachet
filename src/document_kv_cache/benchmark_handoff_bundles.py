"""Module entry point for benchmark handoff bundle generation."""

from __future__ import annotations

from document_kv_cache.benchmark_handoffs import bundle_main

__all__ = ["main"]


def main() -> int:
    return bundle_main()


if __name__ == "__main__":
    raise SystemExit(main())
