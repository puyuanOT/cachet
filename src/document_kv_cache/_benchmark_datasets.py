from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import Any

from document_kv_cache._benchmark_models import _json_object_mapping
from document_kv_cache.benchmarks import (
    BenchmarkExample,
    BenchmarkSuite,
    validate_v1_dataset,
    validate_v1_hardware_target,
)
from document_kv_cache.models import DocumentChunkType
from document_kv_cache.storage import local_path
from document_kv_cache.workflow import SourceChunk, SourceDocument


def load_v1_jsonl_suite(
    *,
    suite_id: str,
    paths: Mapping[str, str | Path],
    model_id: str | None = None,
    hardware_target: str | None = None,
    limit_per_dataset: int | None = None,
) -> BenchmarkSuite:
    for dataset in paths:
        validate_v1_dataset(dataset)
    if hardware_target is not None:
        validate_v1_hardware_target(hardware_target)
    return load_jsonl_suite(
        suite_id=suite_id,
        paths=paths,
        model_id=model_id,
        hardware_target=hardware_target,
        limit_per_dataset=limit_per_dataset,
    )


def load_jsonl_suite(
    *,
    suite_id: str,
    paths: Mapping[str, str | Path],
    model_id: str | None = None,
    hardware_target: str | None = None,
    limit_per_dataset: int | None = None,
) -> BenchmarkSuite:
    """Load an extensible JSONL suite; scorer/prompt support is checked at run time."""

    examples: list[BenchmarkExample] = []
    for dataset, path in paths.items():
        dataset_examples = load_benchmark_jsonl(
            path,
            dataset=dataset,
            limit=limit_per_dataset,
            require_dataset=True,
        )
        if not dataset_examples:
            raise ValueError(f"Dataset {dataset!r} must include at least one example")
        examples.extend(dataset_examples)
    kwargs: dict[str, Any] = {
        "suite_id": suite_id,
        "examples": tuple(examples),
        "datasets": tuple(paths),
    }
    if model_id is not None:
        kwargs["model_id"] = model_id
    if hardware_target is not None:
        kwargs["hardware_target"] = hardware_target
    suite = BenchmarkSuite(**kwargs)
    if not suite.examples:
        raise ValueError("Benchmark suite must include at least one example")
    return suite


def load_benchmark_jsonl(
    path: str | Path,
    *,
    dataset: str | None = None,
    limit: int | None = None,
    require_dataset: bool = False,
) -> tuple[BenchmarkExample, ...]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    examples: list[BenchmarkExample] = []
    records = _iter_jsonl(path)
    if limit is not None:
        records = islice(records, limit)
    for record_index, (line_number, record) in enumerate(records, start=1):
        try:
            example = _example_from_record(
                record,
                default_dataset=dataset,
                record_index=record_index,
                require_dataset=require_dataset,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Benchmark JSONL line {line_number}: {exc}") from exc
        examples.append(example)
    return tuple(examples)


def _iter_jsonl(path: str | Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    with local_path(str(path)).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Benchmark JSONL line {line_number} is not valid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"Benchmark JSONL line {line_number} must be an object"
                )
            yield line_number, record


def _example_from_record(
    record: Mapping[str, Any],
    *,
    default_dataset: str | None,
    record_index: int,
    require_dataset: bool,
) -> BenchmarkExample:
    dataset = _string_field(record, "dataset", default=default_dataset)
    if require_dataset and default_dataset is not None and dataset != default_dataset:
        raise ValueError(
            f"dataset {dataset!r} does not match expected dataset {default_dataset!r}"
        )
    return BenchmarkExample(
        example_id=_string_field(
            record,
            "example_id",
            fallback_fields=("id",),
            default=f"{dataset}-{record_index}",
        ),
        dataset=dataset,
        documents=tuple(_documents_from_record(record)),
        query=_string_field(record, "query", fallback_fields=("question",)),
        expected_answer=_optional_string_field(
            record,
            "expected_answer",
            fallback_fields=("answer", "target"),
        ),
        references=_references_from_record(record),
        metadata=_string_mapping(record.get("metadata", {}), field_name="metadata"),
        kv_transfer_params=_json_object_mapping(
            record.get("kv_transfer_params", {}),
            "kv_transfer_params",
        ),
        arm_kv_transfer_params=_arm_kv_transfer_params_from_record(
            record.get("arm_kv_transfer_params", {})
        ),
    )


def _references_from_record(record: Mapping[str, Any]) -> tuple[str, ...]:
    raw = record.get("references", record.get("answers", ()))
    if raw in (None, ()):
        return ()
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise ValueError("references must be an array of non-empty strings")
    references = tuple(raw)
    if any(not isinstance(reference, str) or not reference for reference in references):
        raise ValueError("references must contain non-empty strings")
    return references


def _arm_kv_transfer_params_from_record(
    value: Any,
) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("arm_kv_transfer_params must be an object")
    normalized: dict[str, Mapping[str, Any]] = {}
    for arm_id, params in value.items():
        if not isinstance(arm_id, str) or not arm_id:
            raise ValueError("arm_kv_transfer_params keys must be non-empty strings")
        normalized[arm_id] = _json_object_mapping(
            params,
            f"arm_kv_transfer_params.{arm_id}",
        )
    return normalized


def _validate_benchmark_jsonl_record(
    record: Mapping[str, Any],
    *,
    dataset: str | None = None,
    record_index: int = 1,
    require_dataset: bool = False,
) -> None:
    _example_from_record(
        record,
        default_dataset=dataset,
        record_index=record_index,
        require_dataset=require_dataset,
    )


def _documents_from_record(record: Mapping[str, Any]) -> tuple[SourceDocument, ...]:
    raw_documents = record.get("documents")
    if raw_documents is None:
        raw_documents = record.get("contexts")
    if raw_documents is None:
        raw_documents = record.get("paragraphs")
    if raw_documents is None:
        raw_documents = record.get("context")
    if raw_documents is None:
        raise ValueError(
            "Benchmark JSONL record must include documents, contexts, paragraphs, or context"
        )
    normalized_documents = _normalize_raw_documents(raw_documents)
    documents = tuple(
        _document_from_record(document, index=index)
        for index, document in enumerate(normalized_documents)
    )
    if not documents:
        raise ValueError("documents must contain at least one document")
    return documents


def _normalize_raw_documents(raw_documents: Any) -> tuple[Any, ...]:
    if isinstance(raw_documents, Mapping):
        return tuple(
            {"document_id": key, "text": value} for key, value in raw_documents.items()
        )
    if isinstance(raw_documents, str):
        return (raw_documents,)
    if not isinstance(raw_documents, Sequence) or isinstance(raw_documents, bytes):
        raise ValueError("documents must be a sequence of document objects")
    if raw_documents and _looks_like_hotpot_context_pair(raw_documents[0]):
        return tuple(
            _hotpot_pair_to_document(item, index=index)
            for index, item in enumerate(raw_documents)
        )
    return tuple(raw_documents)


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
    title = _coerce_string(value[0], field_name=f"context {index} title")
    return {
        "document_id": title or f"doc-{index}",
        "title": title,
        "sentences": value[1],
    }


def _document_from_record(record: Any, *, index: int) -> SourceDocument:
    if isinstance(record, str):
        return SourceDocument.from_texts(
            document_id=f"doc-{index}",
            chunks={"text": record},
        )
    if not isinstance(record, Mapping):
        raise ValueError("documents entries must be objects or strings")
    document_id = _string_field(
        record,
        "document_id",
        fallback_fields=("id", "title", "idx"),
        default=f"doc-{index}",
    )
    metadata = _string_mapping(
        record.get("metadata", {}),
        field_name="document metadata",
    )
    title = _optional_string_field(record, "title", fallback_fields=("name",))
    if title is not None and "title" not in metadata:
        metadata = {**metadata, "title": title}
    return SourceDocument(
        document_id=document_id,
        chunks=_chunks_from_record(record),
        metadata=metadata,
    )


def _chunks_from_record(record: Mapping[str, Any]) -> tuple[SourceChunk, ...]:
    chunks: list[SourceChunk] = []
    static_text = _optional_string_field(
        record,
        "static_text",
        fallback_fields=("summary",),
    )
    if static_text is not None:
        chunks.append(
            SourceChunk(
                chunk_id="static",
                text=static_text,
                chunk_type=DocumentChunkType.DOCUMENT_STATIC,
            )
        )
    raw_chunks = record.get("chunks")
    if raw_chunks is None:
        raw_chunks = record.get("sentences")
    if raw_chunks is None:
        text = _optional_string_field(
            record,
            "text",
            fallback_fields=("body", "context", "paragraph_text"),
        )
        if text is not None:
            raw_chunks = {"text": text}
    if raw_chunks is not None:
        chunks.extend(_iter_chunks(raw_chunks))
    if not chunks:
        raise ValueError("document record must include static_text, chunks, or text")
    return tuple(chunks)


def _iter_chunks(raw_chunks: Any) -> Iterable[SourceChunk]:
    if isinstance(raw_chunks, Mapping):
        for chunk_id, text in raw_chunks.items():
            yield SourceChunk(
                chunk_id=str(chunk_id),
                text=_coerce_string(text, field_name=f"chunk {chunk_id}"),
            )
        return
    if isinstance(raw_chunks, Sequence) and not isinstance(raw_chunks, (str, bytes)):
        for index, chunk in enumerate(raw_chunks):
            yield _chunk_from_record(chunk, index=index)
        return
    raise ValueError("chunks must be a mapping or sequence")


def _chunk_from_record(record: Any, *, index: int) -> SourceChunk:
    if isinstance(record, str):
        return SourceChunk(chunk_id=f"chunk-{index}", text=record)
    if not isinstance(record, Mapping):
        raise ValueError("chunk entries must be objects or strings")
    chunk_type_value = _string_field(
        record,
        "chunk_type",
        default=DocumentChunkType.DOCUMENT_CHUNK.value,
    )
    try:
        chunk_type = DocumentChunkType(chunk_type_value)
    except ValueError as exc:
        raise ValueError(f"Unsupported chunk_type {chunk_type_value!r}") from exc
    return SourceChunk(
        chunk_id=_string_field(
            record,
            "chunk_id",
            fallback_fields=("id", "idx"),
            default=f"chunk-{index}",
        ),
        text=_string_field(
            record,
            "text",
            fallback_fields=("body", "context", "paragraph_text"),
        ),
        chunk_type=chunk_type,
        metadata=_string_mapping(
            record.get("metadata", {}),
            field_name="chunk metadata",
        ),
    )


def _string_field(
    record: Mapping[str, Any],
    field_name: str,
    *,
    fallback_fields: Sequence[str] = (),
    default: str | None = None,
) -> str:
    value = _field_value(
        record,
        field_name,
        fallback_fields=fallback_fields,
        default=default,
    )
    if value is None:
        expected = ", ".join((field_name, *fallback_fields))
        raise ValueError(f"Missing required field: {expected}")
    return _coerce_string(value, field_name=field_name)


def _optional_string_field(
    record: Mapping[str, Any],
    field_name: str,
    *,
    fallback_fields: Sequence[str] = (),
) -> str | None:
    value = _field_value(record, field_name, fallback_fields=fallback_fields)
    if value is None:
        return None
    return _coerce_string(value, field_name=field_name)


def _field_value(
    record: Mapping[str, Any],
    field_name: str,
    *,
    fallback_fields: Sequence[str],
    default: str | None = None,
) -> Any:
    for candidate in (field_name, *fallback_fields):
        if candidate in record and record[candidate] is not None:
            return record[candidate]
    return default


def _coerce_string(value: Any, *, field_name: str) -> str:
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
    return {
        str(key): _coerce_string(item, field_name=f"{field_name}.{key}")
        for key, item in value.items()
    }


for _public_function in (
    load_v1_jsonl_suite,
    load_jsonl_suite,
    load_benchmark_jsonl,
    _validate_benchmark_jsonl_record,
):
    _public_function.__module__ = "document_kv_cache.benchmark_runner"
