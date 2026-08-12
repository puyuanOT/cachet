"""Deterministic, tokenizer-only preparation for representative HotpotQA canaries."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, cast

from document_kv_cache._benchmark_datasets import (
    _example_from_record,
    load_benchmark_jsonl,
)
from document_kv_cache.benchmarks import (
    DEFAULT_SYSTEM_PROMPT_POSITION,
    DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
    BenchmarkExample,
    build_prompt_parts,
    resolve_system_prompt_position,
)
from document_kv_cache.storage import local_path
from document_kv_cache.workflow import SourceChunk, SourceDocument


REPRESENTATIVE_HOTPOTQA_RECORD_TYPE = "cachet.representative_hotpotqa_input"
REPRESENTATIVE_HOTPOTQA_SCHEMA_VERSION = 1
REPRESENTATIVE_HOTPOTQA_TOKENIZER_ID = "Qwen/Qwen3-4B-Instruct-2507"
REPRESENTATIVE_HOTPOTQA_TOKENIZER_REVISION = (
    "cdbee75f17c01a7cc42f958dc650907174af0554"
)
REPRESENTATIVE_HOTPOTQA_TOKEN_TARGETS = (8192, 16384)
REPRESENTATIVE_HOTPOTQA_ADD_SPECIAL_TOKENS = False

_PADDING_DOCUMENT_ID = "cachet-representative-length-padding"
_PADDING_DOCUMENT_TITLE = "Deterministic irrelevant length padding"
_PADDING_CHUNK_ID = "length-padding"
_PADDING_UNITS = (" padding", " x", "x", "0", ".", "\n")
_CANONICAL_SOURCE_FIELDS = frozenset(
    {
        "dataset",
        "example_id",
        "query",
        "documents",
        "expected_answer",
        "references",
        "metadata",
    }
)


class RepresentativeTokenizer(Protocol):
    """Small tokenizer surface needed by the preparation algorithm."""

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> Sequence[int]: ...


@dataclass(frozen=True, slots=True)
class PreparedRepresentativeHotpotQA:
    """Paths and public digests for one exact-length prepared dataset."""

    jsonl_path: Path
    provenance_json_path: Path
    jsonl_sha256: str
    example_count: int
    input_tokens_target: int


@dataclass(frozen=True, slots=True)
class _SourceRow:
    record: Mapping[str, Any]
    example: BenchmarkExample
    record_sha256: str


@dataclass(frozen=True, slots=True)
class _PreparedRow:
    record: Mapping[str, Any]
    example: BenchmarkExample
    source_record_sha256: str
    unpadded_prompt_sha256: str
    unpadded_token_count: int
    padding_unit_sha256: str | None
    padding_unit_repetitions: int


@dataclass(frozen=True, slots=True)
class _PaddingResult:
    filler: str
    unit: str
    repetitions: int
    example: BenchmarkExample


def prepare_representative_hotpotqa_jsonl(
    source_path: str | Path,
    output_jsonl: str | Path,
    *,
    input_tokens_target: int,
    example_count: int = 2,
    provenance_json: str | Path | None = None,
    tokenizer: RepresentativeTokenizer | None = None,
) -> PreparedRepresentativeHotpotQA:
    """Write deterministic HotpotQA rows whose logical prefill is exactly sized.

    ``source_path`` accepts either the official HotpotQA dev-distractor JSON
    array or Cachet's canonical HotpotQA JSONL. The production path always
    loads the exact pinned tokenizer; ``tokenizer`` is dependency injection for
    deterministic unit tests and offline contract checks only.
    """

    target = _validated_target(input_tokens_target)
    if type(example_count) is not int or example_count < 2:
        raise ValueError("example_count must be an integer of at least 2")
    if resolve_system_prompt_position() != DEFAULT_SYSTEM_PROMPT_POSITION:
        raise ValueError(
            "representative HotpotQA preparation requires the default start "
            "system-prompt position"
        )
    source = local_path(str(source_path))
    destination = local_path(str(output_jsonl))
    if source.resolve() == destination.resolve():
        raise ValueError("source_path and output_jsonl must be different files")
    raw_source = source.read_bytes()
    raw_records, source_format = _read_source_records(raw_source)
    source_rows = _normalize_source_rows(raw_records)
    if len(source_rows) < example_count:
        raise ValueError(
            f"HotpotQA source must contain at least {example_count} distinct examples"
        )
    resolved_tokenizer = tokenizer or load_representative_hotpotqa_tokenizer()
    prepared_rows = _select_and_prepare_rows(
        source_rows,
        tokenizer=resolved_tokenizer,
        target=target,
        example_count=example_count,
    )
    output_bytes = _jsonl_bytes(tuple(row.record for row in prepared_rows))
    _write_and_verify_jsonl(
        destination,
        output_bytes,
        tokenizer=resolved_tokenizer,
        target=target,
        example_count=example_count,
    )
    output_digest = sha256(output_bytes).hexdigest()
    provenance_path = (
        local_path(str(provenance_json))
        if provenance_json is not None
        else destination.with_suffix(".provenance.json")
    )
    if provenance_path.resolve() in {source.resolve(), destination.resolve()}:
        raise ValueError("provenance_json must be separate from source and output JSONL")
    provenance_record = _provenance_record(
        prepared_rows,
        source_sha256=sha256(raw_source).hexdigest(),
        source_format=source_format,
        source_record_count=len(source_rows),
        output_sha256=output_digest,
        output_byte_count=len(output_bytes),
        target=target,
        requested_example_count=example_count,
    )
    _atomic_write_bytes(
        provenance_path,
        _canonical_json_bytes(provenance_record, pretty=True),
    )
    return PreparedRepresentativeHotpotQA(
        jsonl_path=destination,
        provenance_json_path=provenance_path,
        jsonl_sha256=output_digest,
        example_count=example_count,
        input_tokens_target=target,
    )


def load_representative_hotpotqa_tokenizer() -> RepresentativeTokenizer:
    """Load only the exact pinned Qwen tokenizer (never a causal model)."""

    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:  # pragma: no cover - depends on caller environment.
        raise RuntimeError(
            "representative HotpotQA preparation requires Transformers for "
            "AutoTokenizer"
        ) from exc
    auto_tokenizer = getattr(transformers, "AutoTokenizer", None)
    if auto_tokenizer is None:
        raise RuntimeError("Transformers does not expose AutoTokenizer")
    loaded = auto_tokenizer.from_pretrained(
        REPRESENTATIVE_HOTPOTQA_TOKENIZER_ID,
        revision=REPRESENTATIVE_HOTPOTQA_TOKENIZER_REVISION,
        trust_remote_code=False,
        use_fast=True,
    )
    if not callable(getattr(loaded, "encode", None)):
        raise TypeError("AutoTokenizer must return an object with encode()")
    return cast(RepresentativeTokenizer, loaded)


def representative_hotpotqa_main(argv: Sequence[str] | None = None) -> int:
    """CLI for deterministic exact-length representative HotpotQA preparation."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare exact-token HotpotQA JSONL with the pinned Qwen tokenizer; "
            "no causal model is loaded."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Official HotpotQA JSON array or canonical Cachet HotpotQA JSONL.",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--provenance-json")
    parser.add_argument(
        "--input-tokens-target",
        required=True,
        type=int,
        choices=REPRESENTATIVE_HOTPOTQA_TOKEN_TARGETS,
    )
    parser.add_argument("--example-count", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        result = prepare_representative_hotpotqa_jsonl(
            args.source,
            args.output_jsonl,
            input_tokens_target=args.input_tokens_target,
            example_count=args.example_count,
            provenance_json=args.provenance_json,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "ok": False,
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "example_count": result.example_count,
                "input_tokens_target": result.input_tokens_target,
                "jsonl_path": str(result.jsonl_path),
                "jsonl_sha256": result.jsonl_sha256,
                "ok": True,
                "provenance_json_path": str(result.provenance_json_path),
                "tokenizer_id": REPRESENTATIVE_HOTPOTQA_TOKENIZER_ID,
                "tokenizer_revision": REPRESENTATIVE_HOTPOTQA_TOKENIZER_REVISION,
            },
            sort_keys=True,
        )
    )
    return 0


def _validated_target(value: int) -> int:
    if type(value) is not int or value not in REPRESENTATIVE_HOTPOTQA_TOKEN_TARGETS:
        raise ValueError(
            "input_tokens_target must be exactly one of "
            f"{REPRESENTATIVE_HOTPOTQA_TOKEN_TARGETS}"
        )
    return value


def _read_source_records(
    raw_source: bytes,
) -> tuple[tuple[Mapping[str, Any], ...], str]:
    try:
        text = raw_source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("HotpotQA source must be UTF-8") from exc
    if not text.strip():
        raise ValueError("HotpotQA source must not be empty")
    if text.lstrip().startswith("["):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"HotpotQA JSON array is invalid: {exc.msg}") from exc
        if not isinstance(value, list):
            raise ValueError("HotpotQA JSON source must contain an array")
        records = tuple(
            _required_json_object(item, f"HotpotQA JSON record {index}")
            for index, item in enumerate(value, start=1)
        )
        source_format = "hotpotqa.dev_distractor_json_array"
    else:
        rows: list[Mapping[str, Any]] = []
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"HotpotQA JSONL line {line_number} is invalid: {exc.msg}"
                ) from exc
            rows.append(
                _required_json_object(value, f"HotpotQA JSONL line {line_number}")
            )
        records = tuple(rows)
        source_format = "cachet.canonical_hotpotqa_jsonl"
    if not records:
        raise ValueError("HotpotQA source must contain at least one record")
    return records, source_format


def _required_json_object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(
        Mapping[str, Any],
        json.loads(json.dumps(dict(value), ensure_ascii=False)),
    )


def _normalize_source_rows(
    raw_records: Sequence[Mapping[str, Any]],
) -> tuple[_SourceRow, ...]:
    rows: list[_SourceRow] = []
    seen_keys: set[tuple[str, str]] = set()
    for index, raw_record in enumerate(raw_records, start=1):
        record = _normalize_source_record(raw_record, index=index)
        try:
            example = _example_from_record(
                record,
                default_dataset="hotpotqa",
                record_index=index,
                require_dataset=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"HotpotQA source record {index}: {exc}") from exc
        if len(example.documents) < 2:
            raise ValueError(
                f"HotpotQA source record {index} must contain at least two documents"
            )
        if not example.references:
            raise ValueError(
                f"HotpotQA source record {index} must contain an answer or references"
            )
        key = (example.dataset, example.example_id)
        if key in seen_keys:
            raise ValueError("HotpotQA source contains duplicate example identities")
        seen_keys.add(key)
        rows.append(
            _SourceRow(
                record=record,
                example=example,
                record_sha256=_canonical_sha256(record),
            )
        )
    return tuple(rows)


def _normalize_source_record(
    raw_record: Mapping[str, Any],
    *,
    index: int,
) -> Mapping[str, Any]:
    if raw_record.get("dataset") == "hotpotqa" and "documents" in raw_record:
        unknown = set(raw_record).difference(_CANONICAL_SOURCE_FIELDS)
        if unknown:
            raise ValueError(
                f"canonical HotpotQA source record {index} has unknown fields: "
                f"{sorted(unknown)}"
            )
        for field_name in ("example_id", "query", "documents"):
            if field_name not in raw_record:
                raise ValueError(
                    f"canonical HotpotQA source record {index} requires {field_name}"
                )
        return cast(
            Mapping[str, Any],
            json.loads(json.dumps(dict(raw_record), ensure_ascii=False)),
        )
    required_fields = ("_id", "question", "answer", "context")
    missing = [field_name for field_name in required_fields if field_name not in raw_record]
    if missing:
        raise ValueError(
            f"official HotpotQA source record {index} is missing fields: {missing}"
        )
    example_id = _required_string(raw_record["_id"], f"record {index}._id")
    query = _required_string(raw_record["question"], f"record {index}.question")
    answer = _required_string(raw_record["answer"], f"record {index}.answer")
    documents = _official_hotpotqa_documents(raw_record["context"], index=index)
    record: dict[str, Any] = {
        "dataset": "hotpotqa",
        "documents": documents,
        "example_id": example_id,
        "expected_answer": answer,
        "metadata": {
            "level": str(raw_record.get("level", "")),
            "source": "hotpotqa/hotpot_qa",
            "split": "validation",
            "type": str(raw_record.get("type", "")),
        },
        "query": query,
    }
    raw_references = raw_record.get("references", raw_record.get("answers"))
    if raw_references is not None:
        record["references"] = _references(raw_references, index=index)
    return record


def _official_hotpotqa_documents(value: Any, *, index: int) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"official HotpotQA record {index}.context must be an array")
    documents: list[dict[str, Any]] = []
    for document_index, item in enumerate(value):
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes, bytearray))
            or len(item) != 2
        ):
            raise ValueError(
                f"official HotpotQA record {index} context entry {document_index} "
                "must be [title, sentences]"
            )
        title = _required_string(
            item[0],
            f"record {index}.context[{document_index}].title",
        )
        sentences = item[1]
        if (
            not isinstance(sentences, Sequence)
            or isinstance(sentences, (str, bytes, bytearray))
            or not sentences
        ):
            raise ValueError(
                f"official HotpotQA record {index} context entry {document_index} "
                "sentences must be a non-empty array"
            )
        documents.append(
            {
                "chunks": [
                    _required_string(
                        sentence,
                        (
                            f"record {index}.context[{document_index}]"
                            f".sentences[{sentence_index}]"
                        ),
                    )
                    for sentence_index, sentence in enumerate(sentences)
                ],
                "document_id": title,
                "title": title,
            }
        )
    return documents


def _references(value: Any, *, index: int) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"official HotpotQA record {index} references must be an array")
    references = [
        _required_string(item, f"record {index}.references[{reference_index}]")
        for reference_index, item in enumerate(value)
    ]
    if not references:
        raise ValueError(f"official HotpotQA record {index} references must not be empty")
    return references


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _select_and_prepare_rows(
    source_rows: Sequence[_SourceRow],
    *,
    tokenizer: RepresentativeTokenizer,
    target: int,
    example_count: int,
) -> tuple[_PreparedRow, ...]:
    selected: list[_PreparedRow] = []
    ordered = sorted(
        source_rows,
        key=lambda row: (row.example.example_id, row.record_sha256),
    )
    for row in ordered:
        prepared = _prepare_source_row(row, tokenizer=tokenizer, target=target)
        if prepared is not None:
            selected.append(prepared)
        if len(selected) == example_count:
            return tuple(selected)
    raise ValueError(
        f"HotpotQA source cannot produce {example_count} exact {target}-token examples"
    )


def _prepare_source_row(
    row: _SourceRow,
    *,
    tokenizer: RepresentativeTokenizer,
    target: int,
) -> _PreparedRow | None:
    unpadded_prompt = build_prompt_parts(row.example).prefill_prompt
    unpadded_count = _token_count(tokenizer, unpadded_prompt)
    if unpadded_count > target:
        return None
    if unpadded_count == target:
        return _PreparedRow(
            record=row.record,
            example=row.example,
            source_record_sha256=row.record_sha256,
            unpadded_prompt_sha256=sha256(unpadded_prompt.encode("utf-8")).hexdigest(),
            unpadded_token_count=unpadded_count,
            padding_unit_sha256=None,
            padding_unit_repetitions=0,
        )
    padding = _exact_padding(row.example, tokenizer=tokenizer, target=target)
    if padding is None:
        return None
    record = json.loads(json.dumps(dict(row.record), ensure_ascii=False))
    raw_documents = record.get("documents")
    if not isinstance(raw_documents, list):
        raise TypeError("normalized HotpotQA documents must be an array")
    raw_documents.append(_padding_document_record(padding.filler))
    prompt = build_prompt_parts(padding.example).prefill_prompt
    if _token_count(tokenizer, prompt) != target:
        raise ValueError("exact padding verification drifted from the token target")
    return _PreparedRow(
        record=record,
        example=padding.example,
        source_record_sha256=row.record_sha256,
        unpadded_prompt_sha256=sha256(unpadded_prompt.encode("utf-8")).hexdigest(),
        unpadded_token_count=unpadded_count,
        padding_unit_sha256=sha256(padding.unit.encode("utf-8")).hexdigest(),
        padding_unit_repetitions=padding.repetitions,
    )


def _exact_padding(
    example: BenchmarkExample,
    *,
    tokenizer: RepresentativeTokenizer,
    target: int,
) -> _PaddingResult | None:
    for unit in _PADDING_UNITS:
        result = _exact_repeated_unit(
            example,
            tokenizer=tokenizer,
            target=target,
            unit=unit,
        )
        if result is not None:
            return result
    return None


def _exact_repeated_unit(
    example: BenchmarkExample,
    *,
    tokenizer: RepresentativeTokenizer,
    target: int,
    unit: str,
) -> _PaddingResult | None:
    counts: dict[int, int] = {}

    def count(repetitions: int) -> int:
        if repetitions not in counts:
            candidate = _with_padding_document(example, unit * repetitions)
            counts[repetitions] = _token_count(
                tokenizer,
                build_prompt_parts(candidate).prefill_prompt,
            )
        return counts[repetitions]

    maximum = target * 32
    low = 0
    high = 1
    while high <= maximum and count(high) < target:
        low = high
        high *= 2
    high = min(high, maximum)
    high_count = count(high)
    if high_count < target:
        return None
    while low + 1 < high:
        middle = (low + high) // 2
        middle_count = count(middle)
        if middle_count < target:
            low = middle
        else:
            high = middle
    candidates = sorted(
        {
            repetition
            for repetition in range(max(1, low - 16), min(maximum, high + 16) + 1)
        }
    )
    for repetitions in candidates:
        if count(repetitions) == target:
            filler = unit * repetitions
            return _PaddingResult(
                filler=filler,
                unit=unit,
                repetitions=repetitions,
                example=_with_padding_document(example, filler),
            )
    return None


def _with_padding_document(example: BenchmarkExample, filler: str) -> BenchmarkExample:
    padding_document = SourceDocument(
        document_id=_PADDING_DOCUMENT_ID,
        chunks=(SourceChunk(chunk_id=_PADDING_CHUNK_ID, text=filler),),
        metadata={"title": _PADDING_DOCUMENT_TITLE},
    )
    return replace(example, documents=(*example.documents, padding_document))


def _padding_document_record(filler: str) -> dict[str, Any]:
    return {
        "chunks": [{"chunk_id": _PADDING_CHUNK_ID, "text": filler}],
        "document_id": _PADDING_DOCUMENT_ID,
        "title": _PADDING_DOCUMENT_TITLE,
    }


def _token_count(tokenizer: RepresentativeTokenizer, prompt: str) -> int:
    token_ids = tokenizer.encode(
        prompt,
        add_special_tokens=REPRESENTATIVE_HOTPOTQA_ADD_SPECIAL_TOKENS,
    )
    if isinstance(token_ids, (str, bytes, bytearray)) or not isinstance(
        token_ids, Sequence
    ):
        raise TypeError("tokenizer.encode() must return a sequence of token ids")
    if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
        raise ValueError("tokenizer.encode() returned an invalid token id")
    return len(token_ids)


def _write_and_verify_jsonl(
    path: Path,
    content: bytes,
    *,
    tokenizer: RepresentativeTokenizer,
    target: int,
    example_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        examples = load_benchmark_jsonl(
            temporary_path,
            dataset="hotpotqa",
            require_dataset=True,
        )
        if len(examples) != example_count:
            raise ValueError("prepared HotpotQA output example count changed on reload")
        if len({example.example_id for example in examples}) != example_count:
            raise ValueError("prepared HotpotQA output identities are not distinct")
        for example in examples:
            if len(example.documents) < 2:
                raise ValueError("prepared HotpotQA output requires at least two documents")
            prompt = build_prompt_parts(example).prefill_prompt
            count = _token_count(tokenizer, prompt)
            if count != target:
                raise ValueError(
                    f"prepared HotpotQA output has {count} tokens; expected exactly {target}"
                )
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _provenance_record(
    prepared_rows: Sequence[_PreparedRow],
    *,
    source_sha256: str,
    source_format: str,
    source_record_count: int,
    output_sha256: str,
    output_byte_count: int,
    target: int,
    requested_example_count: int,
) -> Mapping[str, Any]:
    example_rows: list[dict[str, Any]] = []
    for row in prepared_rows:
        prepared_prompt = build_prompt_parts(row.example).prefill_prompt
        example_rows.append(
            {
                "document_count": len(row.example.documents),
                "example_key_sha256": _canonical_sha256(
                    {
                        "dataset": row.example.dataset,
                        "example_id": row.example.example_id,
                    }
                ),
                "padding": {
                    "appended_document_count": int(row.padding_unit_repetitions > 0),
                    "unit_repetitions": row.padding_unit_repetitions,
                    "unit_sha256": row.padding_unit_sha256,
                },
                "prepared_prompt_sha256": sha256(
                    prepared_prompt.encode("utf-8")
                ).hexdigest(),
                "prepared_record_sha256": _canonical_sha256(row.record),
                "prepared_token_count": target,
                "source_record_sha256": row.source_record_sha256,
                "unpadded_prompt_sha256": row.unpadded_prompt_sha256,
                "unpadded_token_count": row.unpadded_token_count,
            }
        )
    return {
        "dataset": "hotpotqa",
        "examples": example_rows,
        "examples_sha256": _canonical_sha256(example_rows),
        "input_tokens_target": target,
        "output": {
            "byte_count": output_byte_count,
            "example_count": len(prepared_rows),
            "jsonl_sha256": output_sha256,
        },
        "prompt_contract": {
            "prompt_template_version": DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
            "system_prompt_position": DEFAULT_SYSTEM_PROMPT_POSITION,
        },
        "record_type": REPRESENTATIVE_HOTPOTQA_RECORD_TYPE,
        "schema_version": REPRESENTATIVE_HOTPOTQA_SCHEMA_VERSION,
        "selection": {
            "ordering": "example_id_then_canonical_source_record_sha256",
            "requested_example_count": requested_example_count,
        },
        "source": {
            "format": source_format,
            "json_sha256": source_sha256,
            "record_count": source_record_count,
        },
        "tokenizer": {
            "add_special_tokens": REPRESENTATIVE_HOTPOTQA_ADD_SPECIAL_TOKENS,
            "tokenizer_id": REPRESENTATIVE_HOTPOTQA_TOKENIZER_ID,
            "tokenizer_revision": REPRESENTATIVE_HOTPOTQA_TOKENIZER_REVISION,
        },
    }


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        _canonical_json_bytes(record, pretty=False) + b"\n" for record in records
    )


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    kwargs: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    return (json.dumps(value, **kwargs) + ("\n" if pretty else "")).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return sha256(_canonical_json_bytes(value, pretty=False)).hexdigest()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


__all__ = [
    "REPRESENTATIVE_HOTPOTQA_RECORD_TYPE",
    "REPRESENTATIVE_HOTPOTQA_SCHEMA_VERSION",
    "REPRESENTATIVE_HOTPOTQA_TOKENIZER_ID",
    "REPRESENTATIVE_HOTPOTQA_TOKENIZER_REVISION",
    "REPRESENTATIVE_HOTPOTQA_TOKEN_TARGETS",
    "REPRESENTATIVE_HOTPOTQA_ADD_SPECIAL_TOKENS",
    "RepresentativeTokenizer",
    "PreparedRepresentativeHotpotQA",
    "prepare_representative_hotpotqa_jsonl",
    "load_representative_hotpotqa_tokenizer",
    "representative_hotpotqa_main",
]
