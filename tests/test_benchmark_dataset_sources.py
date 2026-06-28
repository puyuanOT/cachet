import json
import zipfile

import pytest

from document_kv_cache.benchmark_dataset_sources import (
    DatasetSourceStagingConfig,
    stage_full_benchmark_datasets,
    stage_hotpotqa_parquet,
    stage_musique_jsonl,
    stage_niah_jsonl,
    stage_wikibio_zip,
)
from document_kv_cache.benchmark_runner import load_benchmark_jsonl


def test_stage_hotpotqa_parquet_writes_canonical_records(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    input_path = tmp_path / "hotpot.parquet"
    output_path = tmp_path / "hotpotqa.jsonl"
    table = pa.Table.from_pylist(
        [
            {
                "id": "hp-1",
                "question": "Who wrote notes?",
                "answer": "Ada Lovelace",
                "type": "bridge",
                "level": "easy",
                "context": {
                    "title": ["Ada"],
                    "sentences": [["Ada wrote notes.", "Ada was a mathematician."]],
                },
            }
        ]
    )
    pq.write_table(table, input_path)

    record = stage_hotpotqa_parquet(input_path, output_path)
    loaded = load_benchmark_jsonl(output_path, dataset="hotpotqa", require_dataset=True)

    assert record["records"] == 1
    assert loaded[0].example_id == "hp-1"
    assert loaded[0].expected_answer == "Ada Lovelace"
    assert loaded[0].documents[0].document_id == "Ada"
    assert loaded[0].documents[0].chunks[0].text == "Ada wrote notes."


def test_stage_musique_jsonl_filters_unanswerable_records(tmp_path):
    input_path = tmp_path / "musique.jsonl"
    output_path = tmp_path / "musique-prepared.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "mq-1",
                        "question": "Where is Paris?",
                        "answer": "France",
                        "answerable": True,
                        "paragraphs": [
                            {
                                "idx": 0,
                                "title": "France",
                                "paragraph_text": "Paris is in France.",
                                "is_supporting": True,
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "id": "mq-2",
                        "question": "Impossible?",
                        "answer": "N/A",
                        "answerable": False,
                        "paragraphs": [{"idx": 0, "paragraph_text": "No answer."}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    record = stage_musique_jsonl(input_path, output_path)
    loaded = load_benchmark_jsonl(output_path, dataset="musique", require_dataset=True)

    assert record["records"] == 1
    assert loaded[0].example_id == "mq-1"
    assert loaded[0].expected_answer == "France"
    assert loaded[0].documents[0].metadata["is_supporting"] == "true"


def test_stage_wikibio_zip_uses_title_as_expected_answer(tmp_path):
    input_path = tmp_path / "wikibio.zip"
    output_path = tmp_path / "biography.jsonl"
    with zipfile.ZipFile(input_path, "w") as archive:
        archive.writestr("wikipedia-biography-dataset/valid/valid.id", "bio-1\n")
        archive.writestr("wikipedia-biography-dataset/valid/valid.nb", "2\n")
        archive.writestr("wikipedia-biography-dataset/valid/valid.title", "Ada_Lovelace\n")
        archive.writestr("wikipedia-biography-dataset/valid/valid.sent", "Ada wrote notes.\nShe studied engines.\n")
        archive.writestr("wikipedia-biography-dataset/valid/valid.box", "name_1:Ada\n")

    record = stage_wikibio_zip(input_path, output_path)
    loaded = load_benchmark_jsonl(output_path, dataset="biography", require_dataset=True)

    assert record["records"] == 1
    assert loaded[0].example_id == "bio-1"
    assert loaded[0].expected_answer == "Ada Lovelace"
    assert "Ada wrote notes. She studied engines." == loaded[0].documents[0].chunks[0].text


def test_stage_niah_jsonl_records_synthetic_grid_metadata(tmp_path):
    output_path = tmp_path / "niah.jsonl"

    record = stage_niah_jsonl(
        output_path,
        sample_count=3,
        context_token_targets=(1024, 2048),
        needle_positions=(0.25, 0.75),
        seed=7,
    )
    loaded = load_benchmark_jsonl(output_path, dataset="niah", require_dataset=True)

    assert record["records"] == 3
    assert record["context_token_targets"] == [1024, 2048]
    assert loaded[0].expected_answer == "cachet-needle-7-00000"
    assert loaded[0].metadata["context_token_target"] == "1024"
    assert loaded[2].metadata["needle_position"] == "0.750"


def test_stage_full_benchmark_datasets_writes_metadata_with_counts(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    hotpot_path = tmp_path / "hotpot.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "hp-1",
                    "question": "Who?",
                    "answer": "Ada",
                    "context": {"title": ["Ada"], "sentences": [["Ada appears."]]},
                }
            ]
        ),
        hotpot_path,
    )
    musique_path = tmp_path / "musique.jsonl"
    musique_path.write_text(
        json.dumps(
            {
                "id": "mq-1",
                "question": "Where?",
                "answer": "France",
                "answerable": True,
                "paragraphs": [{"idx": 0, "paragraph_text": "Paris is in France."}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    wikibio_path = tmp_path / "wikibio.zip"
    with zipfile.ZipFile(wikibio_path, "w") as archive:
        archive.writestr("wikipedia-biography-dataset/valid/valid.id", "bio-1\n")
        archive.writestr("wikipedia-biography-dataset/valid/valid.nb", "1\n")
        archive.writestr("wikipedia-biography-dataset/valid/valid.title", "Ada_Lovelace\n")
        archive.writestr("wikipedia-biography-dataset/valid/valid.sent", "Ada wrote notes.\n")
        archive.writestr("wikipedia-biography-dataset/valid/valid.box", "name_1:Ada\n")

    record = stage_full_benchmark_datasets(
        DatasetSourceStagingConfig(
            output_dir=tmp_path / "prepared",
            hotpotqa_parquet=hotpot_path,
            musique_jsonl=musique_path,
            wikibio_zip=wikibio_path,
            niah_sample_count=5,
            limit_per_dataset=1,
        )
    )
    metadata = json.loads((tmp_path / "prepared" / "dataset-source-metadata.json").read_text(encoding="utf-8"))

    assert record == metadata
    assert record["datasets"]["biography"]["records"] == 1
    assert record["datasets"]["hotpotqa"]["records"] == 1
    assert record["datasets"]["musique"]["records"] == 1
    assert record["datasets"]["niah"]["records"] == 1
    assert record["dataset_specs"] == [
        f"biography={tmp_path / 'prepared' / 'biography.jsonl'}",
        f"hotpotqa={tmp_path / 'prepared' / 'hotpotqa.jsonl'}",
        f"musique={tmp_path / 'prepared' / 'musique.jsonl'}",
        f"niah={tmp_path / 'prepared' / 'niah.jsonl'}",
    ]
