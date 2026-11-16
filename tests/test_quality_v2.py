from __future__ import annotations

import hashlib
import json
import random

import pytest

from document_kv_cache._benchmark_datasets import _example_from_record
from document_kv_cache.benchmarks import (
    NIAH_CELL_IDS,
    DatasetScoreContext,
    extract_single_final_answer,
)
import document_kv_cache.quality_v2 as quality_v2_module
from document_kv_cache.full_score_execution import FULL_SCORE_RUNNER_SCRIPT
from document_kv_cache.quality_v2 import (
    QUALITY_V2_MAX_REGRESSION,
    QUALITY_V2_PARSER_DIGEST,
    QUALITY_V2_PROMPT_TEMPLATE_VERSION,
    QUALITY_V2_README_MARKERS,
    QUALITY_V2_REPORT_RECORD_TYPE,
    biography_v3_entity_scores,
    build_quality_v2_niah_records,
    extract_first_final_answer,
    niah_v2_exact_value_scores,
    quality_v2_primary_metric,
    quality_v2_prompt_parts,
    quality_v2_protocol_record,
    quality_v2_report_record,
    quality_v2_scorer_registry,
    render_quality_v2_readme_region,
    strip_biography_disambiguator,
    summarize_quality_v2_pairs,
    validate_quality_v2_readme_region,
    validate_quality_v2_report_record,
)
from document_kv_cache.quality_v2_job import (
    QUALITY_V2_MAX_MODEL_LEN,
    QUALITY_V2_RUNNER_SCRIPT,
    QUALITY_V2_RUNNER_SHA256,
    quality_v2_submit_payload,
    quality_v2_vllm_server_command,
)


def _example(dataset: str = "hotpotqa"):
    return _example_from_record(
        {
            "dataset": dataset,
            "example_id": "e-1",
            "query": "Who wrote the novel?",
            "documents": [
                {"document_id": "d1", "text": "Alpha wrote the novel."},
                {"document_id": "d2", "text": "Beta reviewed it."},
            ],
            "expected_answer": "Alpha",
            "references": ["Alpha"],
        },
        default_dataset=dataset,
        record_index=1,
        require_dataset=True,
    )


class TestFirstFinalAnswerParser:
    def test_skips_echoed_placeholder_and_ignores_surrounding_text(self) -> None:
        output = (
            "Required response form: <final_answer>answer</final_answer>\n"
            "Thinking...\n<final_answer> Alpha </final_answer> trailing"
        )
        extraction = extract_first_final_answer(output)
        assert (extraction.status, extraction.extracted_answer, extraction.valid) == ("ok", "Alpha", True)
        assert extract_single_final_answer(output).valid is False

    def test_statuses(self) -> None:
        assert extract_first_final_answer("Alpha").status == "missing_block"
        assert extract_first_final_answer("<final_answer>Alpha").status == (
            "multiple_or_malformed_blocks"
        )
        for placeholder in ("", " ... ", "Answer", "your answer"):
            extraction = extract_first_final_answer(f"<final_answer>{placeholder}</final_answer>")
            assert (extraction.status, extraction.valid) == ("empty_answer", False)

    def test_nested_open_tags_resolve_to_innermost_block(self) -> None:
        extraction = extract_first_final_answer(
            "<final_answer>x <final_answer>Beta</final_answer></final_answer>"
        )
        assert extraction.extracted_answer == "Beta"

    def test_first_real_block_wins(self) -> None:
        extraction = extract_first_final_answer(
            "<final_answer>Alpha</final_answer><final_answer>Beta</final_answer>"
        )
        assert extraction.extracted_answer == "Alpha"

    def test_identity_matches_every_scorer_contract(self) -> None:
        registry = quality_v2_scorer_registry()
        for dataset, scorer in registry.entries:
            extraction = scorer.parse_answer("<final_answer>Alpha</final_answer>")
            assert extraction is not None and extraction.extracted_answer == "Alpha", dataset
            assert scorer.answer_parser_digest == QUALITY_V2_PARSER_DIGEST

    def test_rejects_non_string(self) -> None:
        with pytest.raises(TypeError):
            extract_first_final_answer(None)  # type: ignore[arg-type]


class TestPromptParts:
    def test_single_chat_user_turn_around_unchanged_documents(self) -> None:
        example = _example()
        parts = quality_v2_prompt_parts(example)
        prompt = parts.prefill_prompt
        assert prompt.startswith("<|im_start|>user\nBenchmark: ")
        assert prompt.endswith(
            "Required response form: <final_answer>answer</final_answer>"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        assert prompt.count("<|im_start|>") == 2 and prompt.count("<|im_end|>") == 1
        assert parts.system_prompt_position == "start"
        assert "<|im_" not in parts.document_context
        for text in ("Alpha wrote the novel.", "Beta reviewed it."):
            assert text in parts.document_context

    def test_registry_uses_v2_prompt_for_every_dataset(self) -> None:
        registry = quality_v2_scorer_registry()
        example = _example()
        for dataset, scorer in registry.entries:
            assert scorer.prompt_template_version == QUALITY_V2_PROMPT_TEMPLATE_VERSION
            assert scorer.prompt_parts(example) == quality_v2_prompt_parts(example), dataset


class TestScorers:
    def test_strip_biography_disambiguator(self) -> None:
        assert strip_biography_disambiguator("john smith -lrb- footballer , born 1990 -rrb-") == (
            "john smith"
        )
        assert strip_biography_disambiguator("John Smith (footballer)") == "John Smith"
        assert strip_biography_disambiguator("Plain Name") == "Plain Name"

    def test_biography_matches_without_disambiguator(self) -> None:
        context = DatasetScoreContext(
            dataset="biography",
            example_id="b-1",
            output_text="John Smith",
            references=("john smith -lrb- footballer , born 1990 -rrb-",),
        )
        assert biography_v3_entity_scores(context) == {"exact_match": 1.0}
        miss = DatasetScoreContext(
            dataset="biography", example_id="b-2", output_text="Jane Smith", references=("john smith",)
        )
        assert biography_v3_entity_scores(miss) == {"exact_match": 0.0}

    def test_niah_exact_value(self) -> None:
        def score(text: str) -> float:
            context = DatasetScoreContext(
                dataset="niah",
                example_id="n-1",
                output_text=text,
                references=("1234567",),
                metadata={"niah_cell_id": NIAH_CELL_IDS[0]},
            )
            return niah_v2_exact_value_scores(context)["accuracy"]

        assert score(" 1234567. ") == 1.0
        assert score("The number is 1234567") == 0.0
        assert score("12345678") == 0.0

    def test_registry_thresholds_and_versions(self) -> None:
        registry = quality_v2_scorer_registry()
        suffix = f"first_final_answer_v2@{QUALITY_V2_PARSER_DIGEST[:12]}"
        for dataset, scorer in registry.entries:
            assert scorer.version.endswith(suffix), dataset
            assert "final_answer_v1@" not in scorer.version
            expected = 1.0 if dataset == "biography" else QUALITY_V2_MAX_REGRESSION
            assert {spec.max_regression for spec in scorer.metric_specs} == {expected}
            assert quality_v2_primary_metric(dataset) in scorer.metric_names

    def test_protocol_record_is_json_and_stable(self) -> None:
        record = quality_v2_protocol_record()
        assert json.loads(json.dumps(record)) == record
        assert record["headline_datasets"] == ["hotpotqa", "musique", "niah"]
        assert record["diagnostic_datasets"] == ["biography"]
        assert record == quality_v2_protocol_record()


def _paragraphs(count: int = 1200) -> list[str]:
    rng = random.Random(7)
    words = [f"w{index}" for index in range(500)]
    return [
        f"P{index} " + " ".join(rng.choice(words) for _ in range(rng.randrange(40, 160)))
        for index in range(count)
    ]


def _word_count(text: str) -> int:
    return len(text.split())


class TestNiahBuilder:
    def test_cells_token_window_and_single_needle(self) -> None:
        records = build_quality_v2_niah_records(_paragraphs(), count_tokens=_word_count, item_count=9)
        assert len({record["metadata"]["niah_cell_id"] for record in records}) == 9
        for record in records:
            target = int(record["metadata"]["context_token_target"])
            tokens = int(record["metadata"]["prompt_tokens"])
            assert target - 384 <= tokens <= target
            text = record["documents"][0]["text"]
            value = record["expected_answer"]
            assert len(value) == 7 and value.isdigit()
            assert text.count(value) == 1
            assert text.count("special magic number") == 1
            paragraphs = text.split("\n\n")
            index = int(record["metadata"]["needle_paragraph_index"])
            assert paragraphs[index] == record["metadata"]["needle_text"]
            prompt = quality_v2_prompt_parts(
                _example_from_record(record, default_dataset="niah", record_index=1, require_dataset=True)
            ).prefill_prompt
            assert _word_count(prompt) == tokens

    def test_deterministic_and_varied(self) -> None:
        paragraphs = _paragraphs()
        first = build_quality_v2_niah_records(paragraphs, count_tokens=_word_count, item_count=18)
        assert first == build_quality_v2_niah_records(paragraphs, count_tokens=_word_count, item_count=18)
        same_cell = [first[0], first[9]]
        assert same_cell[0]["metadata"]["niah_cell_id"] == same_cell[1]["metadata"]["niah_cell_id"]
        assert same_cell[0]["documents"][0]["text"] != same_cell[1]["documents"][0]["text"]

    def test_rejects_small_pool(self) -> None:
        with pytest.raises(ValueError, match="1000 distinct"):
            build_quality_v2_niah_records(_paragraphs(999), count_tokens=_word_count, item_count=1)


def _row(example_id: str, output: str, score: float, *, tokens: int = 10, error: str | None = None):
    return {
        "dataset": "hotpotqa",
        "example_id": example_id,
        "output_text": output,
        "completion_tokens": tokens,
        "quality_scores": {"exact_match": score, "f1": score},
        "error": error,
    }


class TestSummary:
    def test_paired_difference_rates_and_interval(self) -> None:
        baseline = [_row(f"e{i}", "<final_answer>a</final_answer>", 1.0) for i in range(20)]
        cached = [
            _row(f"e{i}", "x <final_answer>a</final_answer>", 1.0 if i % 4 else 0.0, tokens=256)
            for i in range(20)
        ]
        summary = summarize_quality_v2_pairs(baseline, cached, bootstrap_draws=2000)["hotpotqa"]
        metric = summary.metrics["exact_match"]
        assert (metric["baseline"], metric["cached"], metric["difference"]) == (1.0, 0.75, -0.25)
        assert metric["difference_ci95_low"] <= -0.25 <= metric["difference_ci95_high"] <= 0.0
        assert (metric["cached_better"], metric["cached_worse"]) == (0.0, 5.0)
        assert summary.strict_format_rate == {"baseline": 1.0, "cached": 0.0}
        assert summary.tolerant_parse_rate == {"baseline": 1.0, "cached": 1.0}
        assert summary.token_cap_rate == {"baseline": 0.0, "cached": 1.0}
        assert summary.pairs == 20
        again = summarize_quality_v2_pairs(baseline, cached, bootstrap_draws=2000)["hotpotqa"]
        assert again.to_record() == summary.to_record()

    def test_errors_score_zero(self) -> None:
        baseline = [_row("e0", "", 1.0)]
        cached = [_row("e0", "", 1.0, error="timeout")]
        summary = summarize_quality_v2_pairs(baseline, cached, bootstrap_draws=10)["hotpotqa"]
        assert summary.metrics["exact_match"]["cached"] == 0.0
        assert summary.errors == {"baseline": 0, "cached": 1}

    def test_rejects_unpaired_and_duplicate_rows(self) -> None:
        with pytest.raises(ValueError, match="not paired"):
            summarize_quality_v2_pairs([_row("e0", "", 1.0)], [_row("e1", "", 1.0)])
        with pytest.raises(ValueError, match="duplicate"):
            summarize_quality_v2_pairs([_row("e0", "", 1.0)] * 2, [_row("e0", "", 1.0)])


def _report_rows() -> tuple[list[dict], list[dict], dict]:
    metrics = {
        "biography": ("exact_match",),
        "hotpotqa": ("exact_match", "f1"),
        "musique": ("answer_em", "answer_f1"),
        "niah": ("accuracy",),
    }
    baseline, cached = [], []
    for dataset, names in metrics.items():
        count = 36 if dataset == "niah" else 40
        for index in range(count):
            metadata = {"niah_cell_id": NIAH_CELL_IDS[index % 9]} if dataset == "niah" else {}
            references = ["john smith -lrb- actor -rrb-"] if index < 4 else ["jane doe"]
            base_score = 1.0 if index % 2 else 0.0
            cached_score = 0.0 if dataset == "hotpotqa" and index % 4 == 1 else base_score
            for rows, score, text in (
                (baseline, base_score, "<final_answer>a</final_answer>"),
                (cached, cached_score, "[document id=1] <final_answer>a</final_answer>"),
            ):
                rows.append(
                    {
                        "completion_tokens": 12,
                        "dataset": dataset,
                        "error": None,
                        "example_id": f"{dataset}-{index}",
                        "metadata": metadata,
                        "output_text": text,
                        "prompt_tokens": 900,
                        "quality_scores": {name: score for name in names},
                        "references": references,
                    }
                )
    execution = {
        "chunks": [
            {"chunk_id": f"{dataset}-00", "dataset": dataset, "examples": 36 if dataset == "niah" else 40}
            for dataset in metrics
        ],
        "package_wheel_sha256": "a" * 64,
        "plan_sha256": "b" * 64,
        "vanilla_arm_id": "vanilla_prefill",
    }
    return baseline, cached, execution


class TestReport:
    def test_report_seals_verdicts_and_renders_from_committed_json(self) -> None:
        baseline, cached, execution = _report_rows()
        record = quality_v2_report_record(baseline, cached, execution=execution, bootstrap_draws=400)

        assert record["record_type"] == QUALITY_V2_REPORT_RECORD_TYPE
        assert record["verdicts"]["hotpotqa"]["meets_non_inferiority"] is False
        assert record["verdicts"]["hotpotqa"]["difference"] == pytest.approx(-0.25)
        assert record["verdicts"]["musique"]["meets_non_inferiority"] is True
        assert record["verdicts"]["niah"]["meets_non_inferiority"] is True
        assert set(record["verdicts"]) == {"hotpotqa", "musique", "niah"}
        assert record["diagnostics"]["biography"]["reference_split"]["disambiguated_reference"]["pairs"] == 4
        assert record["diagnostics"]["hotpotqa"]["cached"]["document_markup_in_output_rate"] == 1.0
        assert sum(cell["pairs"] for cell in record["diagnostics"]["niah"]["cells"].values()) == 36

        committed = json.loads(json.dumps(record, indent=2, sort_keys=True))
        validate_quality_v2_report_record(committed)
        region = render_quality_v2_readme_region(committed)
        assert region == render_quality_v2_readme_region(record)
        assert "| HotpotQA | Headline, primary | Exact match | 40 | 0.500 | 0.250 | -0.250 |" in region
        assert "| Does not meet |" in region
        assert "| Biography | Diagnostic | Exact match |" in region
        assert region.index("8k, needle at 10%") < region.index("32k, needle at 90%")
        begin, end = QUALITY_V2_README_MARKERS
        readme = f"# Benchmarks\n\n{begin}{region}{end}\n"
        validate_quality_v2_readme_region(readme, committed)
        with pytest.raises(ValueError, match="differs from the sealed report"):
            validate_quality_v2_readme_region(readme.replace("0.250", "0.260"), committed)

    @pytest.mark.parametrize(
        ("tamper", "match"),
        (
            ("closure", "closure is invalid"),
            ("verdict", "closure is invalid|verdicts do not follow"),
            ("protocol", "protocol differs"),
            ("chunks", "chunk examples do not match"),
        ),
    )
    def test_validator_rejects_tampered_reports(self, tamper: str, match: str) -> None:
        baseline, cached, execution = _report_rows()
        record = quality_v2_report_record(baseline, cached, execution=execution, bootstrap_draws=50)
        if tamper == "closure":
            record["summaries"]["niah"]["pairs"] += 1
        elif tamper == "verdict":
            record["verdicts"]["hotpotqa"]["meets_non_inferiority"] = True
            record["closed_record_sha256"] = quality_v2_module._closed_sha256(record)
        elif tamper == "protocol":
            record["protocol"]["max_tokens"] = 64
            record["closed_record_sha256"] = quality_v2_module._closed_sha256(record)
        else:
            record["execution"]["chunks"][0]["examples"] += 1
            record["closed_record_sha256"] = quality_v2_module._closed_sha256(record)
        with pytest.raises(ValueError, match=match):
            validate_quality_v2_report_record(record)

    def test_report_requires_every_protocol_dataset(self) -> None:
        baseline, cached, execution = _report_rows()
        baseline = [row for row in baseline if row["dataset"] != "musique"]
        cached = [row for row in cached if row["dataset"] != "musique"]
        execution["chunks"] = [c for c in execution["chunks"] if c["dataset"] != "musique"]
        with pytest.raises(ValueError, match="every protocol dataset"):
            quality_v2_report_record(baseline, cached, execution=execution, bootstrap_draws=10)


class TestJob:
    def test_runner_is_full_score_runner_with_entry_renamed(self) -> None:
        assert hashlib.sha256(QUALITY_V2_RUNNER_SCRIPT.encode()).hexdigest() == QUALITY_V2_RUNNER_SHA256
        assert "document_kv_cache.full_score_execution" not in QUALITY_V2_RUNNER_SCRIPT
        assert QUALITY_V2_RUNNER_SCRIPT.count("document_kv_cache.quality_v2_job") == 2
        restored = QUALITY_V2_RUNNER_SCRIPT.replace(
            "document_kv_cache.quality_v2_job", "document_kv_cache.full_score_execution"
        ).replace("cachet.quality_v2.locked_runtime.v1", "cachet.full_score.locked_runtime.v2")
        assert restored == FULL_SCORE_RUNNER_SCRIPT

    def test_submit_payload(self) -> None:
        runner = f"dbfs:/Volumes/a/b/c/runner/{QUALITY_V2_RUNNER_SHA256}/quality-v2-runner.py"
        payload = quality_v2_submit_payload(
            run_name="r",
            task_key="t",
            runner_python_file=runner,
            runner_arguments=["--x", "1"],
            single_user_name="user@example.com",
            purpose="p",
            timeout_seconds=600,
        )
        (task,) = payload["tasks"]
        assert payload["timeout_seconds"] == task["timeout_seconds"] == 600
        assert task["max_retries"] == 0
        assert task["new_cluster"]["num_workers"] == 0
        assert task["new_cluster"]["aws_attributes"]["availability"] == "ON_DEMAND"
        assert task["spark_python_task"]["parameters"] == [
            "--runner-sha256",
            QUALITY_V2_RUNNER_SHA256,
            "--x",
            "1",
        ]
        with pytest.raises(ValueError, match="SHA-256"):
            quality_v2_submit_payload(
                run_name="r",
                task_key="t",
                runner_python_file="dbfs:/Volumes/a/b/c/runner/x/quality-v2-runner.py",
                runner_arguments=[],
                single_user_name="u",
                purpose="p",
            )

    def test_server_command_is_a_consumer_with_v2_identity(self) -> None:
        command = quality_v2_vllm_server_command("/py", telemetry_jsonl="/tmp/t.jsonl")
        assert command[:4] == ["/py", "-u", "-m", "vllm.entrypoints.openai.api_server"]
        assert command[command.index("--max-model-len") + 1] == str(QUALITY_V2_MAX_MODEL_LEN)
        transfer = json.loads(command[command.index("--kv-transfer-config") + 1])
        assert transfer["kv_role"] == "kv_consumer"
        assert QUALITY_V2_PROMPT_TEMPLATE_VERSION in json.dumps(transfer)
