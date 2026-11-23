import json
import random
import zipfile

import pytest

from document_kv_cache import dataset_prep
from document_kv_cache.benchmark_dataset_sources import (
    DatasetSourceStagingConfig,
    _niah_haystack_paragraphs,
    stage_full_benchmark_datasets,
    stage_hotpotqa_parquet,
    stage_musique_jsonl,
    stage_niah_jsonl,
    stage_wikibio_zip,
)
from document_kv_cache.benchmark_plan import (
    BenchmarkDatasetPath,
    BenchmarkPlanConfig,
    build_v1_benchmark_plan,
)
from document_kv_cache.benchmark_runner import (
    BenchmarkGeneration,
    load_benchmark_jsonl,
    load_v1_jsonl_suite,
    run_benchmark_suite,
)
from document_kv_cache.benchmarks import (
    BASELINE_PREFILL_ARM,
    CACHE_REUSE_ARM,
    format_document_context,
)
from document_kv_cache.dataset_prep import opaque_biography_example_id
from document_kv_cache.quality_v2 import (
    QUALITY_V2_NIAH_SEED,
    build_quality_v2_niah_records,
    quality_v2_scorer_registry,
)


def _paragraphs(count=1200):
    rng = random.Random(7)
    words = [f"w{index}" for index in range(500)]
    return [
        f"P{index} " + " ".join(rng.choice(words) for _ in range(rng.randrange(40, 160)))
        for index in range(count)
    ]


def _word_count(text):
    return len(text.split())


def _haystack_paragraphs(record):
    needle = record["metadata"]["needle_text"]
    return [
        paragraph
        for paragraph in record["documents"][0]["text"].split("\n\n")
        if paragraph != needle
    ]


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
                        "answer_aliases": ["French Republic"],
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
    assert loaded[0].references == ("France", "French Republic")
    assert loaded[0].documents[0].metadata["is_supporting"] == "true"


def test_stage_wikibio_zip_uses_title_as_expected_answer(tmp_path):
    input_path = tmp_path / "wikibio.zip"
    output_path = tmp_path / "biography.jsonl"
    with zipfile.ZipFile(input_path, "w") as archive:
        archive.writestr("wikipedia-biography-dataset/valid/valid.id", "bio-1\n")
        archive.writestr("wikipedia-biography-dataset/valid/valid.nb", "2\n")
        archive.writestr("wikipedia-biography-dataset/valid/valid.title", "Ada_Lovelace\n")
        archive.writestr(
            "wikipedia-biography-dataset/valid/valid.sent",
            "Ada Lovelace wrote notes.\nShe studied engines.\n",
        )
        archive.writestr("wikipedia-biography-dataset/valid/valid.box", "name_1:Ada\n")

    record = stage_wikibio_zip(input_path, output_path)
    loaded = load_benchmark_jsonl(output_path, dataset="biography", require_dataset=True)

    assert record["records"] == 1
    opaque_id = opaque_biography_example_id("bio-1", split="valid")
    assert loaded[0].example_id == opaque_id
    assert loaded[0].expected_answer == "Ada Lovelace"
    assert loaded[0].documents[0].document_id == f"{opaque_id}-doc-0"
    assert "title" not in loaded[0].documents[0].metadata
    assert "article_title" not in loaded[0].metadata
    assert loaded[0].metadata["answer_bearing_document_metadata"] == "removed"
    assert loaded[0].metadata["answer_surface_in_document_context"] == "redacted"
    assert "the subject wrote notes. She studied engines." == loaded[0].documents[0].chunks[0].text
    assert "Ada Lovelace" not in loaded[0].documents[0].chunks[0].text
    rendered = format_document_context(loaded[0].documents)
    rendered_header = next(
        line for line in rendered.splitlines() if line.startswith("[document ")
    )
    assert "Ada Lovelace" not in rendered_header
    assert "Ada_Lovelace" not in rendered_header
    assert f'id="{opaque_id}-doc-0"' in rendered_header
    assert f'title="{opaque_id}-doc-0"' in rendered_header


def test_stage_niah_jsonl_writes_the_quality_v2_grid(tmp_path):
    output_path = tmp_path / "niah.jsonl"
    paragraphs = _paragraphs()

    record = stage_niah_jsonl(
        output_path,
        paragraphs=iter(paragraphs),
        count_tokens=_word_count,
        sample_count=3,
    )
    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    loaded = load_benchmark_jsonl(output_path, dataset="niah", require_dataset=True)

    assert record["records"] == 3
    assert record["kind"] == "quality_v2_needle_grid"
    assert record["seed"] == QUALITY_V2_NIAH_SEED
    assert record["context_token_targets"] == [8192, 16384, 32768]
    assert written == build_quality_v2_niah_records(paragraphs, count_tokens=_word_count, item_count=3)
    assert [example.metadata["niah_cell_id"] for example in loaded] == [
        "niah-8k-depth-10",
        "niah-16k-depth-10",
        "niah-32k-depth-10",
    ]
    assert all(example.expected_answer.isdigit() for example in loaded)


def test_niah_haystack_paragraphs_are_every_hotpotqa_then_musique_paragraph(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    hotpot_path = tmp_path / "hotpot.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "hp-1",
                    "context": {
                        "title": ["Ada", "Paris"],
                        "sentences": [["Ada wrote notes.", " She studied engines."], ["Paris is in France."]],
                    },
                },
                {"id": "hp-2", "context": {"title": ["Rome"], "sentences": [["Rome is in Italy."]]}},
            ]
        ),
        hotpot_path,
    )
    musique_path = tmp_path / "musique.jsonl"
    musique_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"id": "mq-1", "answerable": True, "paragraphs": [{"paragraph_text": "Oslo is in Norway."}]},
                {"id": "mq-2", "answerable": False, "paragraphs": [{"paragraph_text": "No answer."}]},
                {"id": "mq-3", "paragraphs": [{"paragraph_text": "Lima is in Peru."}]},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert list(_niah_haystack_paragraphs(hotpot_path, musique_path)) == [
        "Ada wrote notes.  She studied engines.",
        "Paris is in France.",
        "Rome is in Italy.",
        "Oslo is in Norway.",
        "Lima is in Peru.",
    ]


def test_planner_prepares_staged_quality_v2_niah_records_for_scoring(tmp_path):
    staged = tmp_path / "raw" / "niah.jsonl"
    prepared = tmp_path / "prepared" / "niah.jsonl"
    stage_niah_jsonl(staged, paragraphs=_paragraphs(), count_tokens=_word_count, sample_count=9)
    plan = build_v1_benchmark_plan(
        BenchmarkPlanConfig(
            suite_id="niah-v2",
            dataset_paths=tuple(
                BenchmarkDatasetPath(
                    dataset=dataset,
                    raw_jsonl=str(tmp_path / "raw" / f"{dataset}.jsonl"),
                    prepared_jsonl=str(tmp_path / "prepared" / f"{dataset}.jsonl"),
                )
                for dataset in ("biography", "hotpotqa", "musique", "niah")
            ),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
        )
    )
    (prepare_niah,) = (command for command in plan.preparation_commands if command.name == "prepare-niah")
    assert prepare_niah.argv[1:3] == ("-m", "document_kv_cache.dataset_prep")

    assert dataset_prep.main(list(prepare_niah.argv[3:])) == 0
    suite = load_v1_jsonl_suite(suite_id="niah-v2", paths={"niah": prepared})

    class AnsweringEngine:
        def generate(self, request):
            return BenchmarkGeneration(
                output_text=f"<final_answer>{request.example.expected_answer}</final_answer>",
                prompt_tokens=_word_count(request.logical_prompt_text),
                completion_tokens=4,
                ttft_seconds=0.1,
                time_to_completion_seconds=0.2,
            )

    result = run_benchmark_suite(
        suite,
        {BASELINE_PREFILL_ARM: AnsweringEngine(), CACHE_REUSE_ARM: AnsweringEngine()},
    )
    staged_records = {
        record["example_id"]: record
        for record in map(json.loads, staged.read_text(encoding="utf-8").splitlines())
    }
    scorer = quality_v2_scorer_registry().get("niah")

    assert len(result.measurements) == 18
    for measurement in result.measurements:
        staged_record = staged_records[measurement.example_id]
        assert (measurement.scorer_id, measurement.scorer_version) == (scorer.scorer_id, scorer.version)
        assert measurement.quality_scores == {"accuracy": 1.0}
        assert measurement.metadata["niah_cell_id"] == staged_record["metadata"]["niah_cell_id"]
        assert measurement.prompt_tokens == int(staged_record["metadata"]["prompt_tokens"])


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
    source_paragraphs = _paragraphs()
    musique_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"mq-{row}",
                    "question": "Where?",
                    "answer": "France",
                    "answerable": True,
                    "paragraphs": [
                        {"idx": index, "paragraph_text": paragraph}
                        for index, paragraph in enumerate(source_paragraphs[row * 20 : (row + 1) * 20])
                    ],
                }
            )
            for row in range(len(source_paragraphs) // 20)
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
        ),
        count_tokens=_word_count,
    )
    metadata = json.loads((tmp_path / "prepared" / "dataset-source-metadata.json").read_text(encoding="utf-8"))
    (niah_record,) = map(
        json.loads,
        (tmp_path / "prepared" / "niah.jsonl").read_text(encoding="utf-8").splitlines(),
    )

    assert record == metadata
    assert record["record_type"] == "document_kv.full_benchmark_dataset_sources.v3"
    assert record["source_refs"]["niah"]["builder"] == (
        "document_kv_cache.quality_v2:build_quality_v2_niah_records"
    )
    assert record["source_refs"]["niah"]["token_counter"] == "caller-supplied"
    assert record["source_refs"]["niah"]["seed"] == QUALITY_V2_NIAH_SEED
    assert niah_record["metadata"]["niah_cell_id"] == "niah-8k-depth-10"
    assert set(_haystack_paragraphs(niah_record)) <= {"Ada appears.", *source_paragraphs}
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
