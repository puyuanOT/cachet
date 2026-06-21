from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from document_kv_cache.benchmarks import validate_v1_dataset
from document_kv_cache.storage import local_path


DEFAULT_NIAH_QUERY = "What is the hidden answer in the document?"

__all__ = [
    "DEFAULT_NIAH_QUERY",
    "normalize_v1_record",
    "convert_v1_jsonl",
    "write_v1_jsonl",
    "build_niah_record",
    "main",
]


def normalize_v1_record(
    record: Mapping[str, Any],
    dataset: str | None = None,
    *,
    line_number: int = 1,
) -> dict[str, Any]:
    """Convert a raw V1 dataset row into benchmark-runner JSONL schema."""

    if not isinstance(record, Mapping):
        raise ValueError("record must be an object")
    dataset_value = _dataset_for_record(record, dataset=dataset, line_number=line_number)
    if dataset_value == "biography":
        return _normalize_biography(record, line_number=line_number)
    if dataset_value == "hotpotqa":
        return _normalize_hotpotqa(record, line_number=line_number)
    if dataset_value == "musique":
        return _normalize_musique(record, line_number=line_number)
    if dataset_value == "niah":
        return _normalize_niah(record, line_number=line_number)
    raise ValueError(f"Unsupported V1 dataset: {dataset_value}")


def convert_v1_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    dataset: str,
    *,
    limit: int | None = None,
) -> int:
    """Convert raw dataset JSONL into canonical V1 benchmark JSONL."""

    validate_v1_dataset(dataset)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    records: list[dict[str, Any]] = []
    for line_number, record in enumerate(_iter_jsonl(input_path), start=1):
        if limit is not None and len(records) >= limit:
            break
        records.append(normalize_v1_record(record, dataset=dataset, line_number=line_number))
    if not records:
        raise ValueError(f"Input {input_path!s} did not contain any records to convert")
    write_v1_jsonl(records, output_path)
    return len(records)


def write_v1_jsonl(records: Iterable[Mapping[str, Any]], output_path: str | Path) -> int:
    """Write canonical V1 benchmark records as UTF-8 JSONL."""

    normalized_records = tuple(
        _canonical_record_for_write(record, line_number=line_number)
        for line_number, record in enumerate(records, start=1)
    )
    if not normalized_records:
        raise ValueError("records must contain at least one item")

    path = local_path(str(output_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in normalized_records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    return len(normalized_records)


def build_niah_record(
    *,
    example_id: str,
    haystack_text: str,
    needle_answer: str,
    needle_text: str | None = None,
    query: str = DEFAULT_NIAH_QUERY,
    document_id: str = "haystack",
    metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build one synthetic Needle-in-a-Haystack benchmark row."""

    if not example_id:
        raise ValueError("example_id must be non-empty")
    haystack = _require_text(haystack_text, field_name="haystack_text")
    answer = _require_text(needle_answer, field_name="needle_answer")
    needle = _require_text(needle_text or f"The hidden answer is {answer}.", field_name="needle_text")
    prompt = _require_text(query, field_name="query")
    text = haystack if needle in haystack else _insert_needle(haystack, needle)
    return _canonical_record(
        dataset="niah",
        example_id=example_id,
        query=prompt,
        expected_answer=answer,
        documents=(
            {
                "document_id": _require_text(document_id, field_name="document_id"),
                "title": "Needle in a Haystack",
                "text": text,
            },
        ),
        metadata={**_string_mapping(metadata or {}, field_name="metadata"), "needle_text": needle},
    )


def _normalize_biography(record: Mapping[str, Any], *, line_number: int) -> dict[str, Any]:
    subject = _optional_text(record, "name", fallback_fields=("subject", "title", "person"))
    example_id = _example_id(record, dataset="biography", line_number=line_number, fallback=subject)
    query = _optional_text(record, "query", fallback_fields=("question",))
    expected_answer = _optional_text(record, "expected_answer", fallback_fields=("answer", "target", "name", "subject"))
    if query is None:
        query = "Which person is described in the biography?"
    documents = _documents_from_record(
        record,
        default_document_id=_slug_or_default(subject or example_id, default=f"biography-{line_number}"),
        default_title=subject,
        text_fields=("biography", "text", "body", "context", "article", "profile"),
    )
    return _canonical_record(
        dataset="biography",
        example_id=example_id,
        query=query,
        expected_answer=expected_answer,
        documents=documents,
        metadata=_metadata_from_record(record),
    )


def _normalize_hotpotqa(record: Mapping[str, Any], *, line_number: int) -> dict[str, Any]:
    example_id = _example_id(record, dataset="hotpotqa", line_number=line_number)
    documents = _documents_from_record(
        record,
        default_document_id=f"hotpotqa-{line_number}",
        text_fields=("text", "body"),
        preferred_fields=("documents", "contexts", "context"),
    )
    return _canonical_record(
        dataset="hotpotqa",
        example_id=example_id,
        query=_required_text(record, "query", fallback_fields=("question",)),
        expected_answer=_optional_text(record, "expected_answer", fallback_fields=("answer", "target")),
        documents=documents,
        metadata=_metadata_from_record(record),
    )


def _normalize_musique(record: Mapping[str, Any], *, line_number: int) -> dict[str, Any]:
    example_id = _example_id(record, dataset="musique", line_number=line_number)
    documents = _documents_from_record(
        record,
        default_document_id=f"musique-{line_number}",
        text_fields=("text", "body"),
        preferred_fields=("documents", "contexts", "paragraphs", "context"),
    )
    return _canonical_record(
        dataset="musique",
        example_id=example_id,
        query=_required_text(record, "query", fallback_fields=("question",)),
        expected_answer=_optional_text(record, "expected_answer", fallback_fields=("answer", "target")),
        documents=documents,
        metadata=_metadata_from_record(record),
    )


def _normalize_niah(record: Mapping[str, Any], *, line_number: int) -> dict[str, Any]:
    example_id = _example_id(record, dataset="niah", line_number=line_number)
    haystack = _optional_text(record, "haystack_text", fallback_fields=("haystack",))
    answer = _optional_text(record, "needle_answer", fallback_fields=("expected_answer", "answer", "target"))
    needle = _optional_text(record, "needle_text", fallback_fields=("needle",))
    query = _optional_text(record, "query", fallback_fields=("question",)) or DEFAULT_NIAH_QUERY
    if haystack is not None and answer is not None:
        return build_niah_record(
            example_id=example_id,
            haystack_text=haystack,
            needle_answer=answer,
            needle_text=needle,
            query=query,
            metadata=_metadata_from_record(record),
        )
    documents = _documents_from_record(
        record,
        default_document_id="haystack",
        default_title="Needle in a Haystack",
        text_fields=("context", "text", "body"),
    )
    return _canonical_record(
        dataset="niah",
        example_id=example_id,
        query=query,
        expected_answer=answer,
        documents=documents,
        metadata=_metadata_from_record(record),
    )


def _canonical_record(
    *,
    dataset: str,
    example_id: str,
    query: str,
    expected_answer: str | None,
    documents: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, str],
) -> dict[str, Any]:
    validate_v1_dataset(dataset)
    if not documents:
        raise ValueError("documents must contain at least one item")
    record: dict[str, Any] = {
        "dataset": dataset,
        "example_id": _require_text(example_id, field_name="example_id"),
        "query": _require_text(query, field_name="query"),
        "documents": [dict(document) for document in documents],
    }
    if expected_answer is not None:
        record["expected_answer"] = _require_text(expected_answer, field_name="expected_answer")
    if metadata:
        record["metadata"] = dict(metadata)
    return record


def _canonical_record_for_write(record: Mapping[str, Any], *, line_number: int) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"Record line {line_number} must be an object")
    try:
        dataset = _dataset_for_record(record, dataset=None, line_number=line_number)
        normalized = _canonical_record(
            dataset=dataset,
            example_id=_required_text(record, "example_id"),
            query=_required_text(record, "query"),
            expected_answer=_optional_text(record, "expected_answer"),
            documents=_documents_from_record(
                record,
                default_document_id=f"{dataset}-{line_number}",
                text_fields=(),
                preferred_fields=("documents",),
            ),
            metadata=_metadata_from_record(record),
        )
        _validate_written_record_for_runner(normalized, dataset=dataset, line_number=line_number)
        return normalized
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Record line {line_number}: {exc}") from exc


def _validate_written_record_for_runner(record: Mapping[str, Any], *, dataset: str, line_number: int) -> None:
    from document_kv_cache.benchmark_runner import _validate_benchmark_jsonl_record

    _validate_benchmark_jsonl_record(
        record,
        dataset=dataset,
        record_index=line_number,
        require_dataset=True,
    )


def _documents_from_record(
    record: Mapping[str, Any],
    *,
    default_document_id: str,
    text_fields: Sequence[str],
    default_title: str | None = None,
    preferred_fields: Sequence[str] = ("documents", "contexts", "paragraphs", "context"),
) -> tuple[Mapping[str, Any], ...]:
    raw_documents = _first_present(record, preferred_fields)
    if raw_documents is None:
        text = _first_text(record, text_fields)
        if text is None:
            expected = ", ".join((*preferred_fields, *text_fields))
            raise ValueError(f"Record must include one of: {expected}")
        raw_documents = (
            {
                "document_id": default_document_id,
                "title": default_title,
                "text": text,
            },
        )
    normalized = tuple(_normalize_raw_documents(raw_documents))
    documents = tuple(_document_from_value(value, index=index) for index, value in enumerate(normalized))
    if not documents:
        raise ValueError("documents must contain at least one item")
    return documents


def _normalize_raw_documents(raw_documents: Any) -> Iterable[Any]:
    if isinstance(raw_documents, str):
        return (raw_documents,)
    if isinstance(raw_documents, Mapping):
        return tuple({"document_id": key, "text": value} for key, value in raw_documents.items())
    if not isinstance(raw_documents, Sequence) or isinstance(raw_documents, bytes):
        raise ValueError("documents must be a sequence, mapping, or string")
    if raw_documents and _looks_like_hotpot_context_pair(raw_documents[0]):
        return tuple(_hotpot_pair_to_document(value, index=index) for index, value in enumerate(raw_documents))
    return raw_documents


def _document_from_value(value: Any, *, index: int) -> Mapping[str, Any]:
    if isinstance(value, str):
        return {"document_id": f"doc-{index}", "text": value}
    if not isinstance(value, Mapping):
        raise ValueError("document entries must be objects or strings")
    document_id = _optional_text(value, "document_id", fallback_fields=("id", "title", "idx"))
    title = _optional_text(value, "title", fallback_fields=("name",))
    document: dict[str, Any] = {"document_id": document_id or f"doc-{index}"}
    if title is not None:
        document["title"] = title
    metadata = _string_mapping(value.get("metadata", {}), field_name="document metadata")
    if metadata:
        document["metadata"] = metadata
    static_text = _optional_text(value, "static_text", fallback_fields=("summary",))
    if static_text is not None:
        document["static_text"] = static_text
    raw_chunks = _first_present(value, ("chunks", "sentences"))
    if raw_chunks is not None:
        document["chunks"] = _chunks_from_value(raw_chunks)
    else:
        text = _optional_text(value, "text", fallback_fields=("body", "context", "paragraph_text"))
        if text is not None:
            document["text"] = text
    if not any(field in document for field in ("static_text", "chunks", "text")):
        raise ValueError("document record must include static_text, chunks, sentences, text, or paragraph_text")
    return document


def _chunks_from_value(raw_chunks: Any) -> list[Any]:
    if isinstance(raw_chunks, Mapping):
        return [
            {
                "chunk_id": str(chunk_id),
                "text": _coerce_text(text, field_name=f"chunk {chunk_id}"),
            }
            for chunk_id, text in raw_chunks.items()
        ]
    if isinstance(raw_chunks, Sequence) and not isinstance(raw_chunks, (str, bytes)):
        return [_chunk_from_value(chunk, index=index) for index, chunk in enumerate(raw_chunks)]
    raise ValueError("chunks must be a mapping or sequence")


def _chunk_from_value(value: Any, *, index: int) -> Any:
    if isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("chunk entries must be objects or strings")
    chunk: dict[str, Any] = {
        "chunk_id": _optional_text(value, "chunk_id", fallback_fields=("id", "idx")) or f"chunk-{index}",
        "text": _required_text(value, "text", fallback_fields=("body", "context", "paragraph_text")),
    }
    chunk_type = _optional_text(value, "chunk_type")
    if chunk_type is not None:
        chunk["chunk_type"] = chunk_type
    metadata = _string_mapping(value.get("metadata", {}), field_name="chunk metadata")
    if metadata:
        chunk["metadata"] = metadata
    return chunk


def _looks_like_hotpot_context_pair(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and isinstance(value[1], Sequence)
        and not isinstance(value[1], (str, bytes))
    )


def _hotpot_pair_to_document(value: Any, *, index: int) -> Mapping[str, Any]:
    if not _looks_like_hotpot_context_pair(value):
        raise ValueError(f"HotpotQA context entry {index} must be [title, sentences]")
    title = _coerce_text(value[0], field_name=f"context {index} title")
    return {
        "document_id": title or f"doc-{index}",
        "title": title,
        "chunks": [_coerce_text(sentence, field_name=f"context {index} sentence") for sentence in value[1]],
    }


def _dataset_for_record(record: Mapping[str, Any], *, dataset: str | None, line_number: int) -> str:
    raw_dataset = _optional_text(record, "dataset")
    if dataset is None and raw_dataset is None:
        raise ValueError(f"Record line {line_number} must include dataset or a default dataset must be provided")
    dataset_value = raw_dataset or dataset
    assert dataset_value is not None
    validate_v1_dataset(dataset_value)
    if dataset is not None and raw_dataset is not None and raw_dataset != dataset:
        raise ValueError(f"Record line {line_number} dataset {raw_dataset!r} does not match expected {dataset!r}")
    return dataset_value


def _example_id(
    record: Mapping[str, Any],
    *,
    dataset: str,
    line_number: int,
    fallback: str | None = None,
) -> str:
    return _optional_text(record, "example_id", fallback_fields=("id", "_id", "qid")) or _slug_or_default(
        fallback,
        default=f"{dataset}-{line_number}",
    )


def _metadata_from_record(record: Mapping[str, Any]) -> Mapping[str, str]:
    metadata = _string_mapping(record.get("metadata", {}), field_name="metadata")
    split = _optional_text(record, "split")
    if split is not None and "split" not in metadata:
        return {**metadata, "split": split}
    return metadata


def _iter_jsonl(path: str | Path) -> Iterable[Mapping[str, Any]]:
    with local_path(str(path)).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError(f"JSONL line {line_number} must be an object")
            yield record


def _first_present(record: Mapping[str, Any], field_names: Sequence[str]) -> Any | None:
    for field_name in field_names:
        if field_name in record and record[field_name] is not None:
            return record[field_name]
    return None


def _first_text(record: Mapping[str, Any], field_names: Sequence[str]) -> str | None:
    for field_name in field_names:
        value = _optional_text(record, field_name)
        if value is not None:
            return value
    return None


def _required_text(
    record: Mapping[str, Any],
    field_name: str,
    *,
    fallback_fields: Sequence[str] = (),
) -> str:
    value = _optional_text(record, field_name, fallback_fields=fallback_fields)
    if value is None:
        expected = ", ".join((field_name, *fallback_fields))
        raise ValueError(f"Missing required field: {expected}")
    return value


def _optional_text(
    record: Mapping[str, Any],
    field_name: str,
    *,
    fallback_fields: Sequence[str] = (),
) -> str | None:
    for candidate in (field_name, *fallback_fields):
        if candidate in record and record[candidate] is not None:
            return _coerce_text(record[candidate], field_name=candidate)
    return None


def _require_text(value: str, *, field_name: str) -> str:
    text = _coerce_text(value, field_name=field_name).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _coerce_text(value: Any, *, field_name: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    raise ValueError(f"{field_name} must be string-like")


def _string_mapping(value: Any, *, field_name: str) -> Mapping[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return {str(key): _coerce_text(item, field_name=f"{field_name}.{key}") for key, item in value.items()}


def _slug_or_default(value: str | None, *, default: str) -> str:
    if not value:
        return default
    slug = "".join(character.lower() if character.isalnum() else "-" for character in value)
    slug = "-".join(part for part in slug.split("-") if part)
    return slug or default


def _insert_needle(haystack: str, needle: str) -> str:
    middle = len(haystack) // 2
    boundary = haystack.find("\n", middle)
    if boundary == -1:
        boundary = haystack.find(" ", middle)
    if boundary == -1:
        boundary = middle
    prefix = haystack[:boundary].rstrip()
    suffix = haystack[boundary:].lstrip()
    return f"{prefix}\n\n{needle}\n\n{suffix}".strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare canonical V1 benchmark JSONL files.")
    parser.add_argument("--dataset", required=True, choices=("biography", "hotpotqa", "musique", "niah"))
    parser.add_argument("--input-jsonl", help="Raw dataset JSONL path. Omit only for synthetic NIAH generation.")
    parser.add_argument("--output-jsonl", required=True, help="Canonical benchmark JSONL output path.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--haystack-text", help="Synthetic NIAH haystack text.")
    parser.add_argument("--haystack-file", help="Synthetic NIAH haystack text file.")
    parser.add_argument("--needle-answer", help="Synthetic NIAH answer.")
    parser.add_argument("--needle-text", help="Synthetic NIAH sentence. Defaults to one containing --needle-answer.")
    parser.add_argument("--query", default=DEFAULT_NIAH_QUERY, help="Synthetic NIAH query.")
    parser.add_argument("--count", type=int, default=1, help="Synthetic NIAH record count.")
    parser.add_argument("--example-id-prefix", default="niah-synthetic")
    args = parser.parse_args(argv)

    try:
        if args.input_jsonl:
            count = convert_v1_jsonl(
                args.input_jsonl,
                args.output_jsonl,
                args.dataset,
                limit=args.limit,
            )
        else:
            count = _write_synthetic_niah(args)
        print(json.dumps({"ok": True, "records": count, "output_jsonl": args.output_jsonl}, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, sort_keys=True))
        return 1
    return 0


def _write_synthetic_niah(args: argparse.Namespace) -> int:
    if args.dataset != "niah":
        raise ValueError("--input-jsonl is required for biography, hotpotqa, and musique")
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if not args.needle_answer:
        raise ValueError("--needle-answer is required for synthetic NIAH generation")
    haystack = _synthetic_haystack(args)
    records = [
        build_niah_record(
            example_id=f"{args.example_id_prefix}-{index + 1}",
            haystack_text=haystack,
            needle_answer=args.needle_answer,
            needle_text=args.needle_text,
            query=args.query,
            metadata={"source": "synthetic"},
        )
        for index in range(args.count)
    ]
    return write_v1_jsonl(records, args.output_jsonl)


def _synthetic_haystack(args: argparse.Namespace) -> str:
    if args.haystack_text and args.haystack_file:
        raise ValueError("Use only one of --haystack-text or --haystack-file")
    if args.haystack_file:
        return local_path(args.haystack_file).read_text(encoding="utf-8")
    if args.haystack_text:
        return args.haystack_text
    raise ValueError("Synthetic NIAH generation requires --haystack-text or --haystack-file")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
