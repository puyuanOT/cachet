"""Public wrapper for :mod:`restaurant_kv_serving.dataset_prep`."""

from __future__ import annotations

from document_kv_cache._reexport import reexport_public

__all__ = reexport_public(
    "restaurant_kv_serving.dataset_prep",
    (
        "DEFAULT_NIAH_QUERY",
        "normalize_v1_record",
        "convert_v1_jsonl",
        "write_v1_jsonl",
        "build_niah_record",
        "main",
    ),
    globals(),
)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

del reexport_public
