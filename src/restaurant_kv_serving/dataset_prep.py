"""Compatibility wrapper for :mod:`document_kv_cache.dataset_prep`."""

from __future__ import annotations

from document_kv_cache._reexport import reexport_public

__all__ = reexport_public(
    "document_kv_cache.dataset_prep",
    (
        "DEFAULT_NIAH_QUERY",
        "normalize_v1_record",
        "convert_v1_jsonl",
        "write_v1_jsonl",
        "build_niah_record",
        "main",
        "argparse",
        "json",
        "Iterable",
        "Mapping",
        "Sequence",
        "Path",
        "Any",
        "validate_v1_dataset",
        "local_path",
    ),
    globals(),
)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

del reexport_public
