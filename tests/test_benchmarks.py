import pytest

from document_kv_cache.benchmarks import (
    BASELINE_PREFILL_ARM,
    CACHE_REUSE_ARM,
    DEFAULT_HARDWARE_TARGET,
    DEFAULT_V1_MODEL_ID,
    SUPPORTED_V1_HARDWARE_TARGETS,
    SUPPORTED_V1_DATASETS,
    BenchmarkComparison,
    BenchmarkExample,
    BenchmarkPromptParts,
    BenchmarkSuite,
    InferenceMeasurement,
    answer_found,
    baseline_prefill_arm,
    build_cache_prefix_text,
    build_cache_suffix_text,
    build_prefill_prompt,
    build_prompt_parts,
    compare_to_baseline,
    dataset_spec,
    document_kv_cache_arm,
    evaluate_v1_benchmark_evidence,
    exact_match,
    format_document_context,
    normalize_answer,
    summarize_measurements,
    validate_v1_hardware_target,
    v1_dataset_specs,
)
from document_kv_cache.workflow import SourceDocument


def document() -> SourceDocument:
    return SourceDocument.from_texts(
        document_id="doc-1",
        static_text="Ada Lovelace biography",
        chunks={"p1": "Lovelace wrote notes on the Analytical Engine."},
    )


def measurement(
    *,
    arm_id: str,
    dataset: str = "biography",
    ttft: float,
    ttc: float,
    output_text: str = "Ada Lovelace",
    error: str | None = None,
) -> InferenceMeasurement:
    return InferenceMeasurement(
        example_id="example-1",
        dataset=dataset,
        arm_id=arm_id,
        prompt_tokens=100,
        completion_tokens=20,
        ttft_seconds=ttft,
        time_to_completion_seconds=ttc,
        output_text=output_text,
        expected_answer="Ada Lovelace",
        error=error,
    )


def qualityless_measurement(*, arm_id: str, dataset: str = "biography") -> InferenceMeasurement:
    return InferenceMeasurement(
        example_id=f"{dataset}-1",
        dataset=dataset,
        arm_id=arm_id,
        prompt_tokens=100,
        completion_tokens=20,
        ttft_seconds=1.0,
        time_to_completion_seconds=2.0,
        output_text="answer without expected answer",
    )


def test_benchmark_suite_defaults_to_v1_contract():
    example = BenchmarkExample(
        example_id="bio-1",
        dataset="biography",
        documents=(document(),),
        query="Who wrote notes on the Analytical Engine?",
        expected_answer="Ada Lovelace",
    )
    suite = BenchmarkSuite(suite_id="v1", examples=(example,))

    assert suite.model_id == DEFAULT_V1_MODEL_ID
    assert suite.hardware_target == DEFAULT_HARDWARE_TARGET
    assert DEFAULT_HARDWARE_TARGET == "aws-g6-l4"
    assert SUPPORTED_V1_HARDWARE_TARGETS == ("aws-g6-l4", "aws-g5")
    validate_v1_hardware_target("aws-g6-l4")
    validate_v1_hardware_target("aws-g5")
    with pytest.raises(ValueError, match="Unsupported V1 hardware target"):
        validate_v1_hardware_target("aws-g6e")
    assert suite.datasets == SUPPORTED_V1_DATASETS
    assert baseline_prefill_arm().uses_cache is False
    assert document_kv_cache_arm().uses_cache is True
    assert document_kv_cache_arm().arm_id == CACHE_REUSE_ARM


def test_benchmark_example_validates_identity_documents_query_and_metadata():
    source_document = document()
    example = BenchmarkExample(
        example_id="bio-1",
        dataset="biography",
        documents=[source_document],
        query="Who wrote notes on the Analytical Engine?",
        expected_answer="Ada Lovelace",
        metadata={"split": "dev"},
    )

    assert example.documents == (source_document,)
    assert example.metadata == {"split": "dev"}

    with pytest.raises(ValueError, match="example_id must be non-empty"):
        BenchmarkExample(
            example_id="",
            dataset="biography",
            documents=(source_document,),
            query="Question?",
        )
    with pytest.raises(ValueError, match="query must be non-empty"):
        BenchmarkExample(
            example_id="bio-1",
            dataset="biography",
            documents=(source_document,),
            query="",
        )
    with pytest.raises(ValueError, match="expected_answer must be non-empty"):
        BenchmarkExample(
            example_id="bio-1",
            dataset="biography",
            documents=(source_document,),
            query="Question?",
            expected_answer="",
        )
    with pytest.raises(ValueError, match="documents must include"):
        BenchmarkExample(example_id="bio-1", dataset="biography", documents=(), query="Question?")
    with pytest.raises(TypeError, match=r"documents\[0\]"):
        BenchmarkExample(
            example_id="bio-1",
            dataset="biography",
            documents=("not-a-document",),
            query="Question?",
        )
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        BenchmarkExample(
            example_id="bio-1",
            dataset="biography",
            documents=(source_document,),
            query="Question?",
            metadata=(),
        )
    with pytest.raises(ValueError, match="metadata keys"):
        BenchmarkExample(
            example_id="bio-1",
            dataset="biography",
            documents=(source_document,),
            query="Question?",
            metadata={"": "dev"},
        )
    with pytest.raises(ValueError, match="metadata.split"):
        BenchmarkExample(
            example_id="bio-1",
            dataset="biography",
            documents=(source_document,),
            query="Question?",
            metadata={"split": 1},
        )


def test_benchmark_suite_validates_identity_examples_and_datasets():
    example = BenchmarkExample(
        example_id="bio-1",
        dataset="biography",
        documents=(document(),),
        query="Who wrote notes on the Analytical Engine?",
    )
    suite = BenchmarkSuite(suite_id="v1", examples=[example], datasets=["biography"])
    hotpotqa_example = BenchmarkExample(
        example_id="bio-1",
        dataset="hotpotqa",
        documents=(document(),),
        query="Which dataset reuses this local id?",
    )
    cross_dataset_suite = BenchmarkSuite(
        suite_id="v1-cross-dataset-ids",
        examples=(example, hotpotqa_example),
        datasets=("biography", "hotpotqa"),
    )

    assert suite.examples == (example,)
    assert suite.datasets == ("biography",)
    assert cross_dataset_suite.examples == (example, hotpotqa_example)

    with pytest.raises(ValueError, match="suite_id must be non-empty"):
        BenchmarkSuite(suite_id="", examples=(example,))
    with pytest.raises(ValueError, match="model_id must be non-empty"):
        BenchmarkSuite(suite_id="v1", examples=(example,), model_id="")
    with pytest.raises(ValueError, match="hardware_target must be non-empty"):
        BenchmarkSuite(suite_id="v1", examples=(example,), hardware_target="")
    with pytest.raises(ValueError, match="Unsupported V1 hardware target"):
        BenchmarkSuite(suite_id="v1", examples=(example,), hardware_target="aws-g6e")
    with pytest.raises(ValueError, match="examples must include"):
        BenchmarkSuite(suite_id="v1", examples=())
    with pytest.raises(TypeError, match=r"examples\[0\]"):
        BenchmarkSuite(suite_id="v1", examples=("not-an-example",))
    duplicate_example = BenchmarkExample(
        example_id="bio-1",
        dataset="biography",
        documents=(document(),),
        query="Duplicate?",
    )
    with pytest.raises(ValueError, match="duplicate dataset/example ids: biography:bio-1"):
        BenchmarkSuite(
            suite_id="v1",
            examples=(example, duplicate_example),
            datasets=("biography",),
        )
    with pytest.raises(ValueError, match="datasets must include"):
        BenchmarkSuite(suite_id="v1", examples=(example,), datasets=())
    with pytest.raises(ValueError, match="duplicate V1 dataset ids: biography"):
        BenchmarkSuite(suite_id="v1", examples=(example,), datasets=("biography", "biography"))


def test_inference_measurement_validates_traceability_output_and_metadata():
    measurement_record = InferenceMeasurement(
        example_id="bio-1",
        dataset="biography",
        arm_id="baseline_prefill",
        prompt_tokens=100,
        completion_tokens=20,
        ttft_seconds=1.0,
        time_to_completion_seconds=2.0,
        output_text="Ada Lovelace",
        expected_answer="Ada Lovelace",
        metadata={"prompt_token_source": "usage"},
    )

    assert measurement_record.metadata == {"prompt_token_source": "usage"}

    base_kwargs = {
        "example_id": "bio-1",
        "dataset": "biography",
        "arm_id": "baseline_prefill",
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "ttft_seconds": 1.0,
        "time_to_completion_seconds": 2.0,
        "output_text": "Ada Lovelace",
    }
    with pytest.raises(ValueError, match="example_id must be non-empty"):
        InferenceMeasurement(**{**base_kwargs, "example_id": ""})
    with pytest.raises(ValueError, match="arm_id must be non-empty"):
        InferenceMeasurement(**{**base_kwargs, "arm_id": ""})
    with pytest.raises(ValueError, match="output_text must be a string"):
        InferenceMeasurement(**{**base_kwargs, "output_text": object()})
    with pytest.raises(ValueError, match="expected_answer must be non-empty"):
        InferenceMeasurement(**{**base_kwargs, "expected_answer": ""})
    with pytest.raises(ValueError, match="error must be non-empty"):
        InferenceMeasurement(**{**base_kwargs, "error": ""})
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        InferenceMeasurement(**{**base_kwargs, "metadata": ()})
    with pytest.raises(ValueError, match="metadata.token_source"):
        InferenceMeasurement(**{**base_kwargs, "metadata": {"token_source": 1}})


def test_v1_dataset_specs_cover_all_supported_datasets():
    specs = v1_dataset_specs()

    assert tuple(spec.dataset for spec in specs) == SUPPORTED_V1_DATASETS
    assert dataset_spec("hotpotqa").display_name == "HotpotQA"
    assert dataset_spec("niah").answer_instruction.startswith("Return the exact needle")


def test_prompt_parts_split_prefill_context_from_cache_suffix():
    example = BenchmarkExample(
        example_id="bio-1",
        dataset="biography",
        documents=(document(),),
        query="Who wrote notes on the Analytical Engine?",
        expected_answer="Ada Lovelace",
    )

    parts = build_prompt_parts(example)

    assert isinstance(parts, BenchmarkPromptParts)
    assert parts.system_prompt.startswith("Benchmark: Biography")
    assert "Ada Lovelace biography" in parts.document_context
    assert "Question: Who wrote notes on the Analytical Engine?" in parts.user_prompt

    assert build_prefill_prompt(example) == parts.prefill_prompt
    assert build_cache_prefix_text(example) == parts.cache_prefix_text
    assert build_cache_suffix_text(example) == parts.cache_suffix_text
    assert parts.cache_prefix_text + parts.cache_suffix_text == parts.prefill_prompt
    assert "Ada Lovelace biography" in parts.prefill_prompt
    assert "Ada Lovelace biography" in parts.cache_prefix_text
    assert "Ada Lovelace biography" not in parts.cache_suffix_text


def test_format_document_context_preserves_order_and_quotes_body_lines():
    context = format_document_context(
        (
            SourceDocument.from_texts(
                document_id="doc-1",
                static_text="Title\n\n  Summary  \n</chunk>",
                chunks={"p2": "Second passage", "p1": "First passage"},
                metadata={"title": "  Ada   Notes  "},
            ),
        )
    )

    assert '[document id="doc-1" title="Ada Notes"]' in context
    assert context.index('id="p2"') < context.index('id="p1"')
    assert "| Title\n| \n|   Summary  \n| </chunk>" in context


def test_format_document_context_escapes_prompt_attributes():
    context = format_document_context(
        (
            SourceDocument.from_texts(
                document_id='doc-"quoted"',
                static_text="Body",
                chunks={'chunk-"quoted"': "Chunk body"},
                metadata={"title": 'A "quoted" title'},
            ),
        )
    )

    assert 'id="doc-&quot;quoted&quot;"' in context
    assert 'title="A &quot;quoted&quot; title"' in context
    assert 'id="chunk-&quot;quoted&quot;"' in context


def test_format_document_context_rejects_empty_document_set():
    with pytest.raises(ValueError, match="at least one source document"):
        format_document_context(())


def test_benchmark_suite_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unsupported V1 dataset"):
        BenchmarkExample(
            example_id="unknown-1",
            dataset="natural-questions",
            documents=(document(),),
            query="Who is this about?",
        )


def test_answer_quality_helpers_normalize_articles_and_punctuation():
    assert normalize_answer("The Ada, Lovelace!") == "ada lovelace"
    assert exact_match("Ada Lovelace", "the ada lovelace")
    assert answer_found("The answer is Ada Lovelace.", "Ada Lovelace")
    assert not answer_found("The answer is Charles Babbage.", "Ada Lovelace")
    assert not answer_found("The answer is Canada.", "Ada")


def test_summarize_measurements_computes_latency_quality_and_errors():
    rows = summarize_measurements(
        [
            measurement(arm_id="baseline_prefill", ttft=10.0, ttc=30.0),
            measurement(arm_id="baseline_prefill", ttft=20.0, ttc=40.0, output_text="Charles Babbage"),
            measurement(arm_id="baseline_prefill", ttft=0.0, ttc=0.0, error="timeout"),
            measurement(arm_id="document_kv_cache", ttft=2.0, ttc=8.0),
            measurement(arm_id="document_kv_cache", ttft=4.0, ttc=12.0),
        ]
    )
    by_arm = {row.arm_id: row for row in rows}

    baseline = by_arm["baseline_prefill"]
    cache = by_arm["document_kv_cache"]

    assert baseline.requests == 3
    assert baseline.errors == 1
    assert baseline.ttft.count == 2
    assert baseline.ttft.p50 == pytest.approx(15.0)
    assert baseline.ttft.p95 == pytest.approx(19.5)
    assert baseline.exact_match_rate == pytest.approx(0.5)
    assert baseline.answer_found_rate == pytest.approx(0.5)
    assert baseline.output_tokens_per_second == pytest.approx(40 / 70)

    assert cache.errors == 0
    assert cache.prompt_tokens_mean == pytest.approx(100.0)
    assert cache.completion_tokens_mean == pytest.approx(20.0)
    assert cache.time_to_completion.p50 == pytest.approx(10.0)
    assert cache.answer_found_rate == pytest.approx(1.0)


def test_compare_to_baseline_reports_speedups_and_quality_deltas():
    rows = summarize_measurements(
        [
            measurement(arm_id="baseline_prefill", ttft=10.0, ttc=30.0, output_text="Charles Babbage"),
            measurement(arm_id="document_kv_cache", ttft=2.0, ttc=10.0),
        ]
    )

    comparison = compare_to_baseline(rows)[0]

    assert comparison.dataset == "biography"
    assert comparison.ttft_speedup == pytest.approx(5.0)
    assert comparison.time_to_completion_speedup == pytest.approx(3.0)
    assert comparison.exact_match_delta == pytest.approx(1.0)
    assert comparison.answer_found_delta == pytest.approx(1.0)


def test_all_error_groups_are_not_latency_comparable():
    rows = summarize_measurements(
        [
            measurement(arm_id="baseline_prefill", ttft=1.0, ttc=1.0, error="timeout"),
            measurement(arm_id="document_kv_cache", ttft=2.0, ttc=2.0, error="timeout"),
        ]
    )
    baseline = next(row for row in rows if row.arm_id == "baseline_prefill")
    comparison = compare_to_baseline(rows)[0]

    assert baseline.requests == 1
    assert baseline.errors == 1
    assert baseline.prompt_tokens_mean is None
    assert baseline.ttft.count == 0
    assert baseline.ttft.p50 is None
    assert baseline.output_tokens_per_second is None
    assert comparison.ttft_speedup is None
    assert comparison.time_to_completion_speedup is None


def test_evaluate_v1_benchmark_evidence_accepts_complete_v1_result():
    measurements = []
    for dataset in SUPPORTED_V1_DATASETS:
        measurements.extend(
            [
                measurement(arm_id="baseline_prefill", dataset=dataset, ttft=10.0, ttc=20.0),
                measurement(arm_id="document_kv_cache", dataset=dataset, ttft=2.0, ttc=8.0),
            ]
        )
    rows = summarize_measurements(measurements)
    comparisons = compare_to_baseline(rows)

    evidence = evaluate_v1_benchmark_evidence(rows, comparisons)

    assert evidence.ok
    assert evidence.required_datasets == SUPPORTED_V1_DATASETS
    assert evidence.missing_report_rows == ()
    assert evidence.missing_comparisons == ()
    assert evidence.comparisons_without_metrics == ()
    assert evidence.rows_without_latency == ()
    assert evidence.rows_without_quality == ()
    assert evidence.unexpected_arms == ()
    assert evidence.issues == ()


def test_evaluate_v1_benchmark_evidence_reports_partial_or_qualityless_runs():
    rows = summarize_measurements(
        [
            qualityless_measurement(arm_id="baseline_prefill"),
            measurement(arm_id="document_kv_cache", ttft=2.0, ttc=8.0),
        ]
    )
    comparisons = compare_to_baseline(rows)

    evidence = evaluate_v1_benchmark_evidence(rows, comparisons)

    assert not evidence.ok
    assert "biography:baseline_prefill" in evidence.rows_without_quality
    assert "hotpotqa:baseline_prefill" in evidence.missing_report_rows
    assert "hotpotqa:document_kv_cache" in evidence.missing_report_rows
    assert "hotpotqa" in evidence.missing_comparisons
    assert any(issue.startswith("missing report rows:") for issue in evidence.issues)
    assert any(issue.startswith("rows without quality evidence:") for issue in evidence.issues)


def test_evaluate_v1_benchmark_evidence_rejects_comparisons_without_metrics():
    measurements = []
    for dataset in SUPPORTED_V1_DATASETS:
        measurements.extend(
            [
                measurement(arm_id=BASELINE_PREFILL_ARM, dataset=dataset, ttft=10.0, ttc=20.0),
                measurement(arm_id=CACHE_REUSE_ARM, dataset=dataset, ttft=2.0, ttc=8.0),
            ]
        )
    rows = summarize_measurements(measurements)
    comparisons = tuple(
        BenchmarkComparison(
            dataset=dataset,
            baseline_arm_id=BASELINE_PREFILL_ARM,
            cache_arm_id=CACHE_REUSE_ARM,
            ttft_speedup=None,
            time_to_completion_speedup=None,
            exact_match_delta=None,
            answer_found_delta=None,
        )
        for dataset in SUPPORTED_V1_DATASETS
    )

    evidence = evaluate_v1_benchmark_evidence(rows, comparisons)

    assert not evidence.ok
    assert evidence.missing_report_rows == ()
    assert evidence.missing_comparisons == ()
    assert evidence.comparisons_without_metrics == SUPPORTED_V1_DATASETS
    assert any(issue.startswith("comparisons without speedup") for issue in evidence.issues)


def test_evaluate_v1_benchmark_evidence_reports_duplicate_identities():
    rows = summarize_measurements(
        [
            measurement(arm_id=BASELINE_PREFILL_ARM, dataset="biography", ttft=10.0, ttc=20.0),
            measurement(arm_id=CACHE_REUSE_ARM, dataset="biography", ttft=2.0, ttc=8.0),
        ]
    )
    duplicate_baseline = rows[0]
    comparisons = compare_to_baseline(rows)
    duplicate_comparison = comparisons[0]

    evidence = evaluate_v1_benchmark_evidence(
        (*rows, duplicate_baseline),
        (*comparisons, duplicate_comparison),
        required_datasets=("biography", "biography"),
    )

    assert not evidence.ok
    assert evidence.duplicate_required_datasets == ("biography",)
    assert evidence.duplicate_report_rows == ("biography:baseline_prefill",)
    assert evidence.duplicate_comparisons == ("biography:baseline_prefill->document_kv_cache",)
    assert "duplicate required datasets: biography" in evidence.issues
    assert "duplicate report rows: biography:baseline_prefill" in evidence.issues
    assert "duplicate comparisons: biography:baseline_prefill->document_kv_cache" in evidence.issues


def test_evaluate_v1_benchmark_evidence_reports_unexpected_arms():
    measurements = []
    for dataset in SUPPORTED_V1_DATASETS:
        measurements.extend(
            [
                measurement(arm_id=BASELINE_PREFILL_ARM, dataset=dataset, ttft=10.0, ttc=20.0),
                measurement(arm_id=CACHE_REUSE_ARM, dataset=dataset, ttft=2.0, ttc=8.0),
            ]
        )
    measurements.append(measurement(arm_id="experimental_cache", dataset="biography", ttft=1.0, ttc=4.0))
    rows = summarize_measurements(measurements)
    comparisons = (
        *compare_to_baseline(rows),
        BenchmarkComparison(
            dataset="biography",
            baseline_arm_id=BASELINE_PREFILL_ARM,
            cache_arm_id="experimental_cache",
            ttft_speedup=10.0,
            time_to_completion_speedup=5.0,
            exact_match_delta=0.0,
            answer_found_delta=0.0,
        ),
    )

    evidence = evaluate_v1_benchmark_evidence(rows, comparisons)

    assert not evidence.ok
    assert evidence.unexpected_arms == ("experimental_cache",)
    assert "unexpected arms: experimental_cache" in evidence.issues


def test_measurements_validate_latency_values():
    with pytest.raises(ValueError, match="ttft_seconds"):
        measurement(arm_id="baseline_prefill", ttft=-1.0, ttc=1.0)
    with pytest.raises(ValueError, match="time_to_completion_seconds"):
        measurement(arm_id="baseline_prefill", ttft=2.0, ttc=1.0)


@pytest.mark.parametrize("bad_latency", (float("nan"), float("inf"), True))
def test_measurements_reject_non_finite_and_boolean_latency_values(bad_latency):
    with pytest.raises(ValueError, match="ttft_seconds must be a non-negative finite number"):
        measurement(arm_id="baseline_prefill", ttft=bad_latency, ttc=2.0)

    with pytest.raises(ValueError, match="time_to_completion_seconds must be a non-negative finite number"):
        measurement(arm_id="baseline_prefill", ttft=1.0, ttc=bad_latency)


@pytest.mark.parametrize("bad_tokens", (-1, 1.5, True))
def test_measurements_reject_non_integer_token_counts(bad_tokens):
    with pytest.raises(ValueError, match="prompt_tokens must be a non-negative integer"):
        InferenceMeasurement(
            example_id="example-1",
            dataset="biography",
            arm_id=BASELINE_PREFILL_ARM,
            prompt_tokens=bad_tokens,
            completion_tokens=1,
            ttft_seconds=0.0,
            time_to_completion_seconds=0.0,
            output_text="",
        )

    with pytest.raises(ValueError, match="completion_tokens must be a non-negative integer"):
        InferenceMeasurement(
            example_id="example-1",
            dataset="biography",
            arm_id=BASELINE_PREFILL_ARM,
            prompt_tokens=1,
            completion_tokens=bad_tokens,
            ttft_seconds=0.0,
            time_to_completion_seconds=0.0,
            output_text="",
        )
