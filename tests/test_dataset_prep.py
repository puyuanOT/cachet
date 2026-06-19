import json

import pytest

from document_kv_cache.benchmark_runner import load_benchmark_jsonl
from document_kv_cache.dataset_prep import (
    build_niah_record,
    convert_v1_jsonl,
    main,
    normalize_v1_record,
    write_v1_jsonl,
)


def test_normalize_biography_builds_query_answer_and_document_from_subject_fields():
    record = normalize_v1_record(
        {
            "name": "Ada Lovelace",
            "biography": "Ada Lovelace wrote notes on the Analytical Engine.",
            "split": "dev",
        },
        dataset="biography",
    )

    assert record == {
        "dataset": "biography",
        "example_id": "ada-lovelace",
        "query": "Which person is described in the biography?",
        "expected_answer": "Ada Lovelace",
        "documents": [
            {
                "document_id": "ada-lovelace",
                "title": "Ada Lovelace",
                "text": "Ada Lovelace wrote notes on the Analytical Engine.",
            }
        ],
        "metadata": {"split": "dev"},
    }


def test_normalize_hotpotqa_converts_context_pairs_to_canonical_documents():
    record = normalize_v1_record(
        {
            "id": "hp-1",
            "question": "Who wrote notes?",
            "answer": "Ada Lovelace",
            "context": [["Ada", ["Ada was a writer.", "She wrote notes."]]],
        },
        dataset="hotpotqa",
    )

    assert record["dataset"] == "hotpotqa"
    assert record["example_id"] == "hp-1"
    assert record["query"] == "Who wrote notes?"
    assert record["expected_answer"] == "Ada Lovelace"
    assert record["documents"] == [
        {
            "document_id": "Ada",
            "title": "Ada",
            "chunks": ["Ada was a writer.", "She wrote notes."],
        }
    ]


def test_normalize_musique_converts_paragraphs_to_documents():
    record = normalize_v1_record(
        {
            "id": "mq-1",
            "question": "Where is Paris?",
            "answer": "France",
            "paragraphs": [
                {"idx": 0, "title": "France", "paragraph_text": "Paris is in France."},
                {"id": "p2", "paragraph_text": "Berlin is in Germany."},
            ],
        },
        dataset="musique",
    )

    assert record["documents"] == [
        {"document_id": "France", "title": "France", "text": "Paris is in France."},
        {"document_id": "p2", "text": "Berlin is in Germany."},
    ]


def test_build_niah_record_inserts_needle_when_missing():
    record = build_niah_record(
        example_id="needle-1",
        haystack_text="A long passage with many irrelevant facts.",
        needle_answer="blue lantern",
        needle_text="The secret code is blue lantern.",
    )

    assert record["dataset"] == "niah"
    assert record["expected_answer"] == "blue lantern"
    assert "The secret code is blue lantern." in record["documents"][0]["text"]
    assert record["metadata"]["needle_text"] == "The secret code is blue lantern."


def test_convert_v1_jsonl_writes_loadable_canonical_rows(tmp_path):
    input_path = tmp_path / "raw_hotpot.jsonl"
    output_path = tmp_path / "hotpot.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "question": "Who wrote notes?",
                "answer": "Ada Lovelace",
                "context": [["Ada", ["Ada was a writer.", "She wrote notes."]]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    count = convert_v1_jsonl(input_path, output_path, "hotpotqa")
    loaded = load_benchmark_jsonl(output_path, dataset="hotpotqa", require_dataset=True)

    assert count == 1
    assert loaded[0].dataset == "hotpotqa"
    assert loaded[0].documents[0].document_id == "Ada"
    assert loaded[0].documents[0].chunks[1].text == "She wrote notes."


def test_convert_v1_jsonl_accepts_file_uri_paths(tmp_path):
    input_path = tmp_path / "raw_bio.jsonl"
    output_path = tmp_path / "prepared" / "bio.jsonl"
    input_path.write_text(
        json.dumps({"name": "Katherine Johnson", "text": "Katherine Johnson computed flight paths."}) + "\n",
        encoding="utf-8",
    )

    count = convert_v1_jsonl(f"file:{input_path}", f"file:{output_path}", "biography")
    loaded = load_benchmark_jsonl(f"file:{output_path}", dataset="biography", require_dataset=True)

    assert count == 1
    assert loaded[0].expected_answer == "Katherine Johnson"


def test_write_v1_jsonl_rejects_empty_iterable(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        write_v1_jsonl([], tmp_path / "empty.jsonl")


def test_normalize_v1_record_rejects_dataset_mismatch():
    with pytest.raises(ValueError, match="does not match expected"):
        normalize_v1_record({"dataset": "biography", "query": "Q", "documents": ["D"]}, dataset="hotpotqa")


def test_main_converts_source_jsonl(tmp_path):
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "bio.jsonl"
    input_path.write_text(
        json.dumps({"name": "Grace Hopper", "text": "Grace Hopper worked on compilers."}) + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--dataset",
            "biography",
            "--input-jsonl",
            str(input_path),
            "--output-jsonl",
            str(output_path),
        ]
    )

    assert exit_code == 0
    loaded = load_benchmark_jsonl(output_path, dataset="biography", require_dataset=True)
    assert loaded[0].expected_answer == "Grace Hopper"


def test_main_generates_synthetic_niah(tmp_path):
    output_path = tmp_path / "niah.jsonl"

    exit_code = main(
        [
            "--dataset",
            "niah",
            "--output-jsonl",
            str(output_path),
            "--haystack-text",
            "irrelevant facts",
            "--needle-answer",
            "blue lantern",
            "--count",
            "2",
        ]
    )

    assert exit_code == 0
    loaded = load_benchmark_jsonl(output_path, dataset="niah", require_dataset=True)
    assert len(loaded) == 2
    assert all(example.expected_answer == "blue lantern" for example in loaded)
