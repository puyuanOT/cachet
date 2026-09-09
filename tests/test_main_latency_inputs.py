import json
from hashlib import sha256
from pathlib import Path

import pytest

from document_kv_cache._benchmark_datasets import _example_from_record
from document_kv_cache.benchmarks import (
    SUPPORTED_V1_DATASETS,
    benchmark_cache_prefix_segments,
    build_prompt_parts,
)
from document_kv_cache.main_latency_inputs import (
    MAIN_LATENCY_ADD_SPECIAL_TOKENS,
    MAIN_LATENCY_EXAMPLES_PER_DATASET,
    MAIN_LATENCY_INPUT_RECORD_TYPE,
    MAIN_LATENCY_INPUT_SCHEMA_VERSION,
    MAIN_LATENCY_PROVENANCE_FILENAME,
    MAIN_LATENCY_TARGET_SEGMENT_COUNTS,
    MAIN_LATENCY_TOKENIZER_ID,
    MAIN_LATENCY_TOKENIZER_REVISION,
    load_main_latency_tokenizer,
    main_latency_inputs_main,
    prepare_main_latency_inputs,
    verify_main_latency_inputs,
)


TEST_EXAMPLES_PER_DATASET = 2


class _CharacterTokenizer:
    def __init__(self):
        self.add_special_tokens_values = []

    def encode(self, text, *, add_special_tokens):
        self.add_special_tokens_values.append(add_special_tokens)
        return [ord(character) for character in text]


class _BoundaryMergingTokenizer:
    _merged = "\n\n[document"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        token_ids = []
        cursor = 0
        while cursor < len(text):
            if text.startswith(self._merged, cursor):
                token_ids.append(1_000_000)
                cursor += len(self._merged)
            else:
                token_ids.append(ord(text[cursor]))
                cursor += 1
        return token_ids


def _source_record(dataset, example_id):
    answer = f"Secret answer for {dataset} {example_id}"
    return {
        "dataset": dataset,
        "documents": [
            {
                "chunks": [
                    {
                        "chunk_id": "fact-a",
                        "text": f"First source fact for {dataset} {example_id}.",
                    },
                    {
                        "chunk_id": "fact-b",
                        "text": f"Second source fact for {dataset} {example_id}.",
                    },
                ],
                "document_id": f"{dataset}-{example_id}-document-a",
                "title": f"{dataset} title A",
            },
            {
                "document_id": f"{dataset}-{example_id}-document-b",
                "text": f"Third source fact for {dataset} {example_id}.",
                "title": f"{dataset} title B",
            },
        ],
        "example_id": f"{dataset}-{example_id}",
        "expected_answer": answer,
        "metadata": {
            "source": f"fixture-{dataset}",
            "split": "validation",
        },
        "query": f"Secret question for {dataset} {example_id}?",
        "references": [answer, f"Alternate {dataset} {example_id}"],
    }


def _write_sources(root: Path):
    paths = {}
    records = {}
    for dataset in SUPPORTED_V1_DATASETS:
        path = root / "sources" / f"{dataset}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            _source_record(dataset, f"{index:02d}")
            for index in reversed(range(TEST_EXAMPLES_PER_DATASET + 2))
        ]
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        paths[dataset] = path
        records[dataset] = tuple(
            sorted(rows, key=_selection_priority_sha256)
        )[:TEST_EXAMPLES_PER_DATASET]
    return paths, records


def _selection_priority_sha256(record):
    def canonical(value):
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    identity_sha256 = sha256(
        canonical({"dataset": record["dataset"], "example_id": record["example_id"]})
    ).hexdigest()
    source_record_sha256 = sha256(canonical(record)).hexdigest()
    return sha256(
        canonical(
            {
                "dataset": record["dataset"],
                "domain": "cachet.main_latency.content_hash_selection.v1",
                "example_identity_sha256": identity_sha256,
                "source_record_sha256": source_record_sha256,
            }
        )
    ).hexdigest()


def _read_rows(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _relative_bytes(root: Path):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_prepare_main_latency_inputs_is_exact_lossless_and_deterministic(tmp_path):
    assert MAIN_LATENCY_EXAMPLES_PER_DATASET == 32
    sources, selected_records = _write_sources(tmp_path)
    first_tokenizer = _CharacterTokenizer()
    second_tokenizer = _CharacterTokenizer()

    first = prepare_main_latency_inputs(
        sources,
        tmp_path / "first",
        tokenizer=first_tokenizer,
        examples_per_dataset=TEST_EXAMPLES_PER_DATASET,
    )
    second = prepare_main_latency_inputs(
        sources,
        tmp_path / "second",
        tokenizer=second_tokenizer,
        examples_per_dataset=TEST_EXAMPLES_PER_DATASET,
    )

    assert _relative_bytes(first.output_dir) == _relative_bytes(second.output_dir)
    assert first.bundle_sha256 == second.bundle_sha256
    assert first.provenance_sha256 == second.provenance_sha256
    assert len(first.files) == 12
    assert first_tokenizer.add_special_tokens_values
    assert set(first_tokenizer.add_special_tokens_values) == {
        MAIN_LATENCY_ADD_SPECIAL_TOKENS
    }

    for artifact in first.files:
        records = _read_rows(artifact.jsonl_path)
        assert len(records) == TEST_EXAMPLES_PER_DATASET
        for record_index, (record, source_record) in enumerate(
            zip(records, selected_records[artifact.dataset], strict=True),
            start=1,
        ):
            assert record["dataset"] == source_record["dataset"]
            assert record["example_id"] == source_record["example_id"]
            assert record["query"] == source_record["query"]
            assert record["expected_answer"] == source_record["expected_answer"]
            assert record["references"] == source_record["references"]
            assert record["metadata"] == source_record["metadata"]
            assert len(record["documents"]) == artifact.segment_count
            assert len({item["document_id"] for item in record["documents"]}) == (
                artifact.segment_count
            )

            source_example = _example_from_record(
                source_record,
                default_dataset=artifact.dataset,
                record_index=record_index,
                require_dataset=True,
            )
            tiled_source_context = "".join(
                item["chunks"][0]["text"] for item in record["documents"]
            )
            assert (
                tiled_source_context == build_prompt_parts(source_example).document_context
            )

            prepared_example = _example_from_record(
                record,
                default_dataset=artifact.dataset,
                record_index=record_index,
                require_dataset=True,
            )
            parts = build_prompt_parts(prepared_example)
            segments = benchmark_cache_prefix_segments(prepared_example)
            segment_ids = [
                first_tokenizer.encode(text, add_special_tokens=False)
                for _chunk_id, text in segments
            ]
            prefix_ids = first_tokenizer.encode(
                parts.cache_prefix_text,
                add_special_tokens=False,
            )
            prompt_ids = first_tokenizer.encode(
                parts.prefill_prompt,
                add_special_tokens=False,
            )
            assert len(prompt_ids) == artifact.input_tokens_target
            assert [token_id for ids in segment_ids for token_id in ids] == prefix_ids
            assert prompt_ids[: len(prefix_ids)] == prefix_ids
            assert len(segments) == artifact.segment_count
            assert max(map(len, segment_ids)) - min(map(len, segment_ids)) <= 1

    provenance_text = first.provenance_json_path.read_text(encoding="utf-8")
    provenance = json.loads(provenance_text)
    assert provenance["record_type"] == MAIN_LATENCY_INPUT_RECORD_TYPE
    assert provenance["schema_version"] == MAIN_LATENCY_INPUT_SCHEMA_VERSION
    assert provenance["bundle_sha256"] == first.bundle_sha256
    assert len(provenance["outputs"]) == 12
    assert all(
        output["record_count"] == TEST_EXAMPLES_PER_DATASET
        and len(output["records"]) == TEST_EXAMPLES_PER_DATASET
        for output in provenance["outputs"]
    )
    assert provenance["protocol"]["selection"][
        "selected_examples_per_dataset"
    ] == TEST_EXAMPLES_PER_DATASET
    assert [row["dataset"] for row in provenance["sources"]] == list(
        SUPPORTED_V1_DATASETS
    )
    assert all(
        len(source["selected_records"]) == TEST_EXAMPLES_PER_DATASET
        for source in provenance["sources"]
    )
    for secret in (
        "Secret answer",
        "Secret question",
        "First source fact",
        "biography-00",
        str(tmp_path),
    ):
        assert secret not in provenance_text


def test_prepare_accepts_identical_recheck_and_verify_can_omit_sources(tmp_path):
    sources, _records = _write_sources(tmp_path)
    tokenizer = _CharacterTokenizer()
    first = prepare_main_latency_inputs(
        sources,
        tmp_path / "prepared",
        tokenizer=tokenizer,
        examples_per_dataset=TEST_EXAMPLES_PER_DATASET,
    )

    repeated = prepare_main_latency_inputs(
        sources,
        tmp_path / "prepared",
        tokenizer=_CharacterTokenizer(),
        examples_per_dataset=TEST_EXAMPLES_PER_DATASET,
    )
    verified = verify_main_latency_inputs(
        tmp_path / "prepared",
        tokenizer=_CharacterTokenizer(),
        examples_per_dataset=TEST_EXAMPLES_PER_DATASET,
    )

    assert repeated.bundle_sha256 == first.bundle_sha256
    assert verified.bundle_sha256 == first.bundle_sha256
    assert verified.provenance_sha256 == first.provenance_sha256


def test_verify_rejects_output_source_and_provenance_tampering(tmp_path):
    sources, _records = _write_sources(tmp_path)
    result = prepare_main_latency_inputs(
        sources,
        tmp_path / "prepared",
        tokenizer=_CharacterTokenizer(),
        examples_per_dataset=TEST_EXAMPLES_PER_DATASET,
    )
    output_path = result.files[0].jsonl_path
    original_output = output_path.read_bytes()
    output_path.write_bytes(original_output.replace(b"Source", b"Tamper", 1))
    with pytest.raises(ValueError, match="output digest mismatch"):
        verify_main_latency_inputs(
            result.output_dir,
            tokenizer=_CharacterTokenizer(),
            examples_per_dataset=TEST_EXAMPLES_PER_DATASET,
        )
    output_path.write_bytes(original_output)

    source_path = sources["biography"]
    source_path.write_bytes(source_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source JSONL digest mismatch"):
        verify_main_latency_inputs(
            result.output_dir,
            source_paths=sources,
            tokenizer=_CharacterTokenizer(),
            examples_per_dataset=TEST_EXAMPLES_PER_DATASET,
        )

    provenance_path = result.provenance_json_path
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["protocol"]["selection"]["selected_examples_per_dataset"] = 1
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="closed_record_sha256"):
        verify_main_latency_inputs(
            result.output_dir,
            tokenizer=_CharacterTokenizer(),
            examples_per_dataset=TEST_EXAMPLES_PER_DATASET,
        )


def test_prepare_supports_explicit_single_example_compatibility_mode(tmp_path):
    sources, selected_records = _write_sources(tmp_path)

    result = prepare_main_latency_inputs(
        sources,
        tmp_path / "single",
        tokenizer=_CharacterTokenizer(),
        examples_per_dataset=1,
    )

    assert len(result.files) == 12
    for artifact in result.files:
        records = _read_rows(artifact.jsonl_path)
        assert [record["example_id"] for record in records] == [
            selected_records[artifact.dataset][0]["example_id"]
        ]
    provenance = json.loads(result.provenance_json_path.read_text(encoding="utf-8"))
    assert provenance["protocol"]["selection"][
        "selected_examples_per_dataset"
    ] == 1
    assert verify_main_latency_inputs(
        result.output_dir,
        source_paths=sources,
        tokenizer=_CharacterTokenizer(),
        examples_per_dataset=1,
    ).bundle_sha256 == result.bundle_sha256
    with pytest.raises(ValueError, match="protocol pins"):
        verify_main_latency_inputs(
            result.output_dir,
            tokenizer=_CharacterTokenizer(),
        )


@pytest.mark.parametrize("examples_per_dataset", (0, -1, True, 1.5, "2"))
def test_prepare_rejects_invalid_examples_per_dataset(
    tmp_path,
    examples_per_dataset,
):
    sources, _records = _write_sources(tmp_path)

    with pytest.raises(ValueError, match="must be a positive integer"):
        prepare_main_latency_inputs(
            sources,
            tmp_path / "invalid-count",
            tokenizer=_CharacterTokenizer(),
            examples_per_dataset=examples_per_dataset,
        )


def test_prepare_fails_closed_on_noncompositional_token_boundaries(tmp_path):
    sources, _records = _write_sources(tmp_path)
    destination = tmp_path / "prepared"

    with pytest.raises(ValueError, match="do not compose"):
        prepare_main_latency_inputs(
            sources,
            destination,
            tokenizer=_BoundaryMergingTokenizer(),
            examples_per_dataset=TEST_EXAMPLES_PER_DATASET,
        )

    assert not destination.exists()


def test_prepare_rejects_noncanonical_or_incomplete_source_sets(tmp_path):
    sources, _records = _write_sources(tmp_path)
    with pytest.raises(ValueError, match="source_paths must contain exactly"):
        prepare_main_latency_inputs(
            {"hotpotqa": sources["hotpotqa"]},
            tmp_path / "incomplete",
            tokenizer=_CharacterTokenizer(),
        )

    biography_path = sources["biography"]
    biography_path.write_text(
        json.dumps(_source_record("biography", "only")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match=f"only 1 of {TEST_EXAMPLES_PER_DATASET} required examples",
    ):
        prepare_main_latency_inputs(
            sources,
            tmp_path / "insufficient",
            tokenizer=_CharacterTokenizer(),
            examples_per_dataset=TEST_EXAMPLES_PER_DATASET,
        )

    record = _source_record("biography", "a")
    record["unknown"] = "not canonical"
    biography_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="noncanonical fields"):
        prepare_main_latency_inputs(
            sources,
            tmp_path / "noncanonical",
            tokenizer=_CharacterTokenizer(),
        )


@pytest.mark.parametrize(
    "field_name",
    ("kv_transfer_params", "arm_kv_transfer_params"),
)
def test_prepare_rejects_transfer_metadata_even_when_empty(tmp_path, field_name):
    sources, _records = _write_sources(tmp_path)
    biography_path = sources["biography"]
    record = _source_record("biography", "a")
    record[field_name] = {}
    biography_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains KV transfer metadata"):
        prepare_main_latency_inputs(
            sources,
            tmp_path / "transfer-metadata",
            tokenizer=_CharacterTokenizer(),
        )


def test_load_main_latency_tokenizer_uses_exact_pin(monkeypatch):
    sentinel = _CharacterTokenizer()
    calls = []

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append((model_id, kwargs))
            return sentinel

    class _Transformers:
        AutoTokenizer = _AutoTokenizer

    monkeypatch.setattr(
        "document_kv_cache.main_latency_inputs.importlib.import_module",
        lambda name: _Transformers if name == "transformers" else None,
    )

    assert load_main_latency_tokenizer() is sentinel
    assert calls == [
        (
            MAIN_LATENCY_TOKENIZER_ID,
            {
                "revision": MAIN_LATENCY_TOKENIZER_REVISION,
                "trust_remote_code": False,
                "use_fast": True,
            },
        )
    ]


def test_main_latency_inputs_cli_prepares_and_verifies(tmp_path, monkeypatch, capsys):
    sources, _records = _write_sources(tmp_path)
    monkeypatch.setattr(
        "document_kv_cache.main_latency_inputs.load_main_latency_tokenizer",
        _CharacterTokenizer,
    )
    source_args = [
        argument
        for dataset in SUPPORTED_V1_DATASETS
        for argument in ("--source", f"{dataset}={sources[dataset]}")
    ]
    output_dir = tmp_path / "prepared"

    exit_code = main_latency_inputs_main(
        [
            "prepare",
            *source_args,
            "--output-dir",
            str(output_dir),
            "--examples-per-dataset",
            str(TEST_EXAMPLES_PER_DATASET),
        ]
    )
    prepared_output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert prepared_output["ok"] is True
    assert prepared_output["examples_per_dataset"] == (
        TEST_EXAMPLES_PER_DATASET
    )
    assert len(prepared_output["files"]) == 12
    assert prepared_output["tokenizer_id"] == MAIN_LATENCY_TOKENIZER_ID
    assert prepared_output["provenance_json_path"] == str(
        output_dir / MAIN_LATENCY_PROVENANCE_FILENAME
    )
    assert all(
        len(_read_rows(Path(row["jsonl_path"])))
        == TEST_EXAMPLES_PER_DATASET
        for row in prepared_output["files"]
    )

    exit_code = main_latency_inputs_main(
        [
            "verify",
            "--output-dir",
            str(output_dir),
            *source_args,
            "--examples-per-dataset",
            str(TEST_EXAMPLES_PER_DATASET),
        ]
    )
    verified_output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert verified_output["ok"] is True
    assert verified_output["examples_per_dataset"] == (
        TEST_EXAMPLES_PER_DATASET
    )
    assert verified_output["bundle_sha256"] == prepared_output["bundle_sha256"]
    assert {row["input_tokens_target"] for row in verified_output["files"]} == set(
        MAIN_LATENCY_TARGET_SEGMENT_COUNTS
    )
