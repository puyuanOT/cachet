from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest

from document_kv_cache.artifact_identity import ArtifactIdentity
from document_kv_cache.benchmark_gates import (
    BenchmarkPublicationGateConfig,
    CacheStateAttestation,
    benchmark_publication_gate_to_record,
    cache_state_attestation_from_vllm_telemetry,
    evaluate_benchmark_publication_gate,
    evaluate_benchmark_evidence_gate,
)
from document_kv_cache.benchmark_runner import (
    BenchmarkGeneration,
    BenchmarkManifestContext,
    BenchmarkRunResult,
    benchmark_run_result_to_evidence_record,
    run_benchmark_suite,
)
from document_kv_cache.benchmark_statistics import (
    paired_benchmark_statistics,
    paired_benchmark_statistics_to_record,
)
from document_kv_cache.benchmarks import (
    DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
    FINAL_ANSWER_EXTRACTED_METADATA_KEY,
    FINAL_ANSWER_PARSER_DIGEST_METADATA_KEY,
    FINAL_ANSWER_PARSER_VERSION_METADATA_KEY,
    NIAH_CELL_IDS,
    DOCUMENT_KV_ARTIFACT_ID_PARAM,
    DOCUMENT_KV_CACHE_METHOD_PARAM,
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
    BenchmarkArm,
    BenchmarkComparison,
    BenchmarkExample,
    BenchmarkReportRow,
    BenchmarkSuite,
    DatasetScorer,
    DatasetScorerRegistry,
    DatasetMetricSpec,
    InferenceMeasurement,
    LatencySummary,
    baseline_prefill_arm,
    method_benchmark_arm,
    default_dataset_scorer_registry,
    diagnostic_answer_scores,
)
from document_kv_cache.workflow import SourceDocument


def test_publication_gate_rejects_one_example_even_when_other_evidence_is_valid() -> None:
    identity = artifact_identity()
    result = benchmark_result(identity)
    gate = evaluate_benchmark_publication_gate(
        result,
        cache_state_attestations=(cold_attestation(identity),),
        artifact_identities={identity.artifact_id: identity},
    )

    assert not gate.ok
    assert any("experiment manifest" in issue for issue in gate.issues)
    assert any("distinct examples" in issue for issue in gate.issues)
    assert gate.checked_cache_requests == 1
    assert gate.cold_attested_requests == 1
    assert benchmark_publication_gate_to_record(gate)["ok"] is False


def test_smoke_gate_accepts_execution_without_quality_or_manifest() -> None:
    identity = artifact_identity()
    original = benchmark_result(identity)
    measurements = tuple(
        replace(measurement, expected_answer=None, output_text="")
        for measurement in original.measurements
    )
    rows = tuple(
        replace(row, exact_match_rate=None, answer_found_rate=None)
        for row in original.report_rows
    )

    gate = evaluate_benchmark_evidence_gate(
        replace(original, measurements=measurements, report_rows=rows),
        policy="smoke",
    )

    assert gate.ok
    assert gate.measurement_scopes == ()


def test_declared_input_target_is_bound_to_every_arbitrary_arm_measurement() -> None:
    class TokenEngine:
        def __init__(self, logical_tokens: int) -> None:
            self.logical_tokens = logical_tokens

        def generate(self, request) -> BenchmarkGeneration:
            return BenchmarkGeneration(
                output_text="Paris",
                prompt_tokens=self.logical_tokens,
                completion_tokens=4,
                ttft_seconds=0.1,
                time_to_completion_seconds=0.2,
                metadata={"logical_prompt_tokens": str(self.logical_tokens)},
            )

    reference = BenchmarkArm(
        arm_id="reference-x",
        uses_cache=False,
        description="reference",
    )
    candidates = tuple(
        BenchmarkArm(
            arm_id=f"candidate-{index}",
            uses_cache=True,
            description=f"candidate {index}",
            cache_method=f"upstream-{index}",
            variant_id="default",
            implementation_kind="upstream",
            method_version="1",
            method_config_digest=str(index) * 64,
            physical_transform_id=f"upstream.transform.{index}",
            source_revision=f"source-{index}",
            checkpoint_identity=f"checkpoint-{index}",
            requires_cachet_handoff=False,
        )
        for index in (1, 2)
    )
    arms = (reference, *candidates)
    result = run_benchmark_suite(
        BenchmarkSuite(
            suite_id="token-target-arbitrary-arms",
            examples=(
                BenchmarkExample(
                    example_id="target-1",
                    dataset="hotpotqa",
                    documents=(
                        SourceDocument.from_text(
                            document_id="target-document",
                            text="Paris is the answer.",
                        ),
                    ),
                    query="What is the answer?",
                    expected_answer="Paris",
                ),
            ),
            datasets=("hotpotqa",),
        ),
        {
            reference.arm_id: TokenEngine(8),
            candidates[0].arm_id: TokenEngine(8),
            candidates[1].arm_id: TokenEngine(7),
        },
        arms=arms,
        manifest_context=BenchmarkManifestContext(
            input_tokens_target=8,
            max_output_tokens=4,
            reference_arm_id=reference.arm_id,
        ),
        evidence_policy="smoke",
    )

    gate = evaluate_benchmark_evidence_gate(result, policy="smoke")

    assert not gate.ok
    assert any(
        "candidate-2" in issue and "input_tokens_target=8" in issue
        for issue in gate.issues
    )


def test_forced_output_target_requires_exact_completion_only_with_ignore_eos() -> None:
    class ShortEngine:
        def generate(self, request) -> BenchmarkGeneration:
            return BenchmarkGeneration(
                output_text="Paris",
                prompt_tokens=8,
                completion_tokens=3,
                ttft_seconds=0.1,
                time_to_completion_seconds=0.2,
            )

    suite = BenchmarkSuite(
        suite_id="forced-output",
        examples=(
            BenchmarkExample(
                example_id="forced-output-1",
                dataset="hotpotqa",
                documents=(
                    SourceDocument.from_text(
                        document_id="forced-output-document",
                        text="Paris is the answer.",
                    ),
                ),
                query="What is the answer?",
                expected_answer="Paris",
            ),
        ),
        datasets=("hotpotqa",),
    )
    baseline = baseline_prefill_arm()
    forced = run_benchmark_suite(
        suite,
        {baseline.arm_id: ShortEngine()},
        arms=(baseline,),
        manifest_context=BenchmarkManifestContext(
            max_output_tokens=4,
            decode_settings={"ignore_eos": True},
        ),
        evidence_policy="smoke",
    )
    ordinary = run_benchmark_suite(
        suite,
        {baseline.arm_id: ShortEngine()},
        arms=(baseline,),
        manifest_context=BenchmarkManifestContext(max_output_tokens=4),
        evidence_policy="smoke",
    )

    forced_gate = evaluate_benchmark_evidence_gate(forced, policy="smoke")
    ordinary_gate = evaluate_benchmark_evidence_gate(ordinary, policy="smoke")

    assert any("forced output_tokens_target=4" in issue for issue in forced_gate.issues)
    assert not any(
        "forced output_tokens_target" in issue for issue in ordinary_gate.issues
    )


def test_latency_only_publication_does_not_require_quality_or_cold_attestation() -> None:
    identity = artifact_identity()
    result = scoped_benchmark_result(identity, scopes=("latency",), with_answers=False)

    gate = evaluate_benchmark_evidence_gate(
        result,
        policy="publication",
        artifact_identities={identity.artifact_id: identity},
    )

    assert gate.ok, gate.issues
    assert gate.measurement_scopes == ("latency",)


def test_publication_gate_validates_cached_reference_arm_artifacts() -> None:
    identity = artifact_identity()
    reference_arm = replace(
        method_benchmark_arm(
            identity.method_id,
            arm_id="g5",
            variant_id="g5",
            method_config_digest=identity.method_config_digest,
            setting_overrides={"hardware_target": "g5.8xlarge"},
        ),
        runtime_environment_overrides={"hardware_target": "g5.8xlarge"},
    )
    candidate_arm = replace(
        method_benchmark_arm(
            identity.method_id,
            arm_id="g6",
            variant_id="g6",
            method_config_digest=identity.method_config_digest,
            setting_overrides={"hardware_target": "g6.8xlarge"},
        ),
        runtime_environment_overrides={"hardware_target": "g6.8xlarge"},
    )
    examples = tuple(
        BenchmarkExample(
            example_id=f"cached-reference-{index}",
            dataset="hotpotqa",
            documents=(
                SourceDocument.from_text(
                    document_id=f"cached-reference-document-{index}",
                    text="The answer is Paris.",
                ),
            ),
            query="What is the answer?",
            arm_kv_transfer_params={
                arm.arm_id: {
                    DOCUMENT_KV_REQUEST_ID_PARAM: f"{arm.arm_id}-{index}",
                    DOCUMENT_KV_HANDOFF_JSON_PARAM: (
                        f"/tmp/{arm.arm_id}-{index}.json"
                    ),
                    DOCUMENT_KV_CACHE_METHOD_PARAM: identity.method_id,
                    DOCUMENT_KV_ARTIFACT_ID_PARAM: identity.artifact_id,
                }
                for arm in (reference_arm, candidate_arm)
            },
        )
        for index in range(4)
    )
    context = BenchmarkManifestContext(
        model_revision=identity.model_revision,
        canonical_model_id=identity.model_id,
        tokenizer_id=identity.tokenizer_id,
        tokenizer_revision=identity.tokenizer_revision,
        lora_id=identity.lora_id,
        engine_id="test-engine",
        engine_version="1",
        serving_platform="test-serving-platform",
        model_dtype="float16",
        runtime_kv_dtype=identity.runtime_kv_dtype,
        layout_version=identity.layout_version,
        payload_axis_order=identity.payload_axis_order,
        block_size=identity.block_size,
        key_position_encoding=identity.key_position_encoding,
        rope_theta=identity.rope_theta,
        rope_rotary_dim=identity.rope_rotary_dim,
        tensor_parallel_size=identity.tensor_parallel_size,
        pipeline_parallel_size=identity.pipeline_parallel_size,
        hardware_fingerprint="test-hardware",
        runtime_id="test-runtime",
        runtime_version="1",
        storage_identity="test-storage",
        cache_state="warm",
        max_output_tokens=16,
        temperature=0.0,
        stream=True,
        complete_dataset_split=True,
        measurement_scopes=("latency",),
        comparison_mode="single_method_setting_variation",
        varied_setting="hardware_target",
        reference_arm_id=reference_arm.arm_id,
    )
    result = run_benchmark_suite(
        BenchmarkSuite(
            suite_id="cached-reference-publication",
            examples=examples,
            hardware_target="g5.8xlarge",
            datasets=("hotpotqa",),
        ),
        {
            reference_arm.arm_id: _ScopedEngine(),
            candidate_arm.arm_id: _ScopedEngine(),
        },
        arms=(reference_arm, candidate_arm),
        manifest_context=context,
        evidence_policy="publication",
        reference_arm_id=reference_arm.arm_id,
    )
    assert result.experiment_manifest is not None
    result = replace(
        result,
        experiment_manifest=replace(
            result.experiment_manifest,
            execution_isolation_mode="separate_process_or_job",
            order_mode="physically_isolated_jobs",
            source_execution_ids=(("g5", "a" * 64), ("g6", "b" * 64)),
        ),
        execution_isolation_mode="separate_process_or_job",
    )

    gate = evaluate_benchmark_evidence_gate(
        result,
        policy="publication",
        artifact_identities={identity.artifact_id: identity},
    )

    assert gate.ok, gate.issues
    assert gate.checked_cache_requests == 8


def test_quality_only_publication_uses_approved_scorer_and_complete_split() -> None:
    identity = artifact_identity()
    result = scoped_benchmark_result(identity, scopes=("quality",), with_answers=True)

    gate = evaluate_benchmark_evidence_gate(
        result,
        policy="publication",
        artifact_identities={identity.artifact_id: identity},
    )

    assert gate.ok, gate.issues


def test_complete_niah_publication_requires_all_nine_grid_cells() -> None:
    baseline_arm = baseline_prefill_arm()
    examples = tuple(
        BenchmarkExample(
            example_id=f"niah-{index}",
            dataset="niah",
            documents=(
                SourceDocument.from_text(
                    document_id=f"niah-document-{index}",
                    text="The requested exact value is Paris.",
                ),
            ),
            query="What is the requested exact value?",
            expected_answer="Paris",
            metadata={"niah_cell_id": cell_id},
        )
        for index, cell_id in enumerate(NIAH_CELL_IDS[:-1])
    )
    result = run_benchmark_suite(
        BenchmarkSuite(
            suite_id="incomplete-niah-grid",
            examples=examples,
            datasets=("niah",),
        ),
        {baseline_arm.arm_id: _ScopedEngine()},
        arms=(baseline_arm,),
        scorer_registry=default_dataset_scorer_registry(),
        manifest_context=BenchmarkManifestContext(
            model_revision="model-revision",
            canonical_model_id="test-model",
            tokenizer_id="test-tokenizer",
            tokenizer_revision="tokenizer-revision",
            lora_id="base",
            engine_id="test-engine",
            engine_version="1",
            serving_platform="test-serving-platform",
            model_dtype="float16",
            model_quantization="none",
            runtime_kv_dtype="bfloat16",
            layout_version="test-layout",
            payload_axis_order="token_major",
            block_size=16,
            key_position_encoding="stored_post_rope",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            hardware_fingerprint="test-hardware",
            runtime_id="test-runtime",
            runtime_version="1",
            storage_identity="test-storage",
            cache_state="cold",
            max_output_tokens=64,
            temperature=0.0,
            stream=True,
            complete_dataset_split=True,
            measurement_scopes=("quality",),
        ),
        evidence_policy="publication",
    )

    gate = evaluate_benchmark_evidence_gate(result, policy="publication")

    assert not gate.ok
    assert any(
        "complete NIAH score arm is missing required cells" in issue
        and NIAH_CELL_IDS[-1] in issue
        for issue in gate.issues
    )


@pytest.mark.parametrize(
    ("metadata_key", "replacement", "issue_fragment"),
    [
        (
            FINAL_ANSWER_PARSER_DIGEST_METADATA_KEY,
            None,
            "missing or mixed final-answer parser identity",
        ),
        (
            FINAL_ANSWER_PARSER_VERSION_METADATA_KEY,
            "different-version",
            "missing or mixed final-answer parser identity",
        ),
        (
            FINAL_ANSWER_EXTRACTED_METADATA_KEY,
            "tampered-answer",
            "extraction cannot be reproduced",
        ),
    ],
)
def test_quality_publication_rejects_missing_mixed_or_tampered_answer_parser_metadata(
    metadata_key,
    replacement,
    issue_fragment,
) -> None:
    identity = artifact_identity()
    original = scoped_benchmark_result(
        identity,
        scopes=("quality",),
        with_answers=True,
    )
    first = original.measurements[0]
    metadata = dict(first.metadata)
    if replacement is None:
        metadata.pop(metadata_key)
    else:
        metadata[metadata_key] = replacement
    result = replace(
        original,
        measurements=(replace(first, metadata=metadata), *original.measurements[1:]),
    )

    gate = evaluate_benchmark_evidence_gate(
        result,
        policy="publication",
        artifact_identities={identity.artifact_id: identity},
    )

    assert not gate.ok
    assert any(issue_fragment in issue for issue in gate.issues)
    assert gate.measurement_scopes == ("quality",)


@pytest.mark.parametrize(
    "method_id",
    ("kv_packet", "cacheblend", "infoflow_kv", "unknown_method"),
)
def test_publication_gate_rejects_planned_and_unknown_cachet_methods(
    method_id: str,
) -> None:
    identity = artifact_identity()
    result = scoped_benchmark_result(identity, scopes=("latency",), with_answers=False)
    assert result.experiment_manifest is not None
    planned_identity = replace(identity, method_id=method_id)
    cache_arm_ids = set(result.cache_arm_ids)
    relabeled = replace(
        result,
        arms=tuple(
            replace(arm, cache_method=method_id)
            if arm.arm_id in cache_arm_ids
            else arm
            for arm in result.arms
        ),
        measurements=tuple(
            replace(
                measurement,
                cache_method=method_id,
                artifact_id=planned_identity.artifact_id,
            )
            if measurement.arm_id in cache_arm_ids
            else measurement
            for measurement in result.measurements
        ),
        experiment_manifest=replace(
            result.experiment_manifest,
            arms=tuple(
                replace(
                    arm,
                    method_id=method_id,
                    artifact_ids=(planned_identity.artifact_id,),
                )
                if arm.arm_id in cache_arm_ids
                else arm
                for arm in result.experiment_manifest.arms
            ),
        ),
    )

    gate = evaluate_benchmark_evidence_gate(
        relabeled,
        policy="publication",
        artifact_identities={planned_identity.artifact_id: planned_identity},
    )

    assert not gate.ok
    assert any("unknown or planned method" in issue for issue in gate.issues)


def test_publication_gate_rejects_measurement_method_or_variant_identity_forgery() -> None:
    identity = artifact_identity()
    result = scoped_benchmark_result(identity, scopes=("latency",), with_answers=False)
    cache_arm_id = result.cache_arm_ids[0]
    forged = replace(
        result,
        measurements=tuple(
            replace(
                measurement,
                cache_method="kv_packet",
                variant_id="forged-variant",
            )
            if measurement.arm_id == cache_arm_id
            else measurement
            for measurement in result.measurements
        ),
    )

    gate = evaluate_benchmark_evidence_gate(
        forged,
        policy="publication",
        artifact_identities={identity.artifact_id: identity},
    )

    assert not gate.ok
    assert any(
        "cache_method 'kv_packet' does not match manifest method" in issue
        for issue in gate.issues
    )
    assert any("variant_id 'forged-variant'" in issue for issue in gate.issues)


@pytest.mark.parametrize(
    ("changes", "expected_field"),
    (
            ({"method_version": "1"}, "method_version"),
        ({"method_config_digest": "2" * 64}, "method_config_digest"),
        ({"model_id": "other/model"}, "model_id"),
        ({"model_revision": "other-revision"}, "model_revision"),
        ({"tokenizer_id": "other/tokenizer"}, "tokenizer_id"),
        ({"tokenizer_revision": "other-tokenizer-revision"}, "tokenizer_revision"),
        ({"lora_id": "other-lora"}, "lora_id"),
        ({"prompt_template_version": "other-template"}, "prompt_template_version"),
        ({"layout_version": "other-layout"}, "layout_version"),
        ({"runtime_kv_dtype": "bfloat16"}, "runtime_kv_dtype"),
        ({"block_size": 32}, "block_size"),
        ({"payload_axis_order": "layer_major"}, "payload_axis_order"),
        ({"rope_theta": 500_000.0}, "rope_theta"),
            ({"rope_rotary_dim": 64}, "rope_rotary_dim"),
            (
                {
                    "key_position_encoding": "stored_post_rope",
                    "rope_theta": None,
                    "rope_rotary_dim": None,
                },
            "key_position_encoding",
        ),
        ({"tensor_parallel_size": 2}, "tensor_parallel_size"),
        ({"pipeline_parallel_size": 2}, "pipeline_parallel_size"),
        ({"artifact_format_id": "other_format"}, "artifact_format_id"),
        ({"artifact_format_version": "2"}, "artifact_format_version"),
    ),
)
def test_publication_gate_binds_artifact_identity_to_arm_runtime_contract(
    changes: dict[str, object],
    expected_field: str,
) -> None:
    identity = artifact_identity()
    result = scoped_benchmark_result(identity, scopes=("latency",), with_answers=False)
    assert result.experiment_manifest is not None
    mutated = replace(identity, **changes)
    cache_arm_ids = set(result.cache_arm_ids)
    relabeled = replace(
        result,
        measurements=tuple(
            replace(measurement, artifact_id=mutated.artifact_id)
            if measurement.arm_id in cache_arm_ids
            else measurement
            for measurement in result.measurements
        ),
        experiment_manifest=replace(
            result.experiment_manifest,
            arms=tuple(
                replace(arm, artifact_ids=(mutated.artifact_id,))
                if arm.arm_id in cache_arm_ids
                else arm
                for arm in result.experiment_manifest.arms
            ),
        ),
    )

    gate = evaluate_benchmark_evidence_gate(
        relabeled,
        policy="publication",
        artifact_identities={mutated.artifact_id: mutated},
    )

    assert not gate.ok
    assert any(expected_field in issue for issue in gate.issues)


def test_publication_gate_rejects_miskeyed_artifact_identity_mapping() -> None:
    identity = artifact_identity()
    result = scoped_benchmark_result(identity, scopes=("latency",), with_answers=False)

    with pytest.raises(ValueError, match="key does not match"):
        evaluate_benchmark_evidence_gate(
            result,
            policy="publication",
            artifact_identities={"0" * 64: identity},
        )


def test_publication_gates_official_metrics_and_collapses_quality_repeats() -> None:
    identity = artifact_identity()
    baseline_arm = baseline_prefill_arm()
    cache_arm = method_benchmark_arm(
        identity.method_id,
        method_config_digest="1" * 64,
    )
    examples = tuple(
        BenchmarkExample(
            example_id=f"official-{index}",
            dataset="hotpotqa",
            documents=(
                SourceDocument.from_text(
                    document_id=f"official-document-{index}",
                    text="Reference-free judging context.",
                ),
            ),
            query="Judge this response.",
            kv_transfer_params={
                DOCUMENT_KV_REQUEST_ID_PARAM: f"official-cache-{index}",
                DOCUMENT_KV_HANDOFF_JSON_PARAM: f"/tmp/official-{index}.json",
                DOCUMENT_KV_CACHE_METHOD_PARAM: identity.method_id,
                DOCUMENT_KV_ARTIFACT_ID_PARAM: identity.artifact_id,
            },
        )
        for index in range(4)
    )
    suite = BenchmarkSuite(
        suite_id="official-quality",
        examples=examples,
        datasets=("hotpotqa",),
    )

    class _OfficialEngine:
        def generate(self, request):
            output = "candidate" if request.arm.uses_cache else "baseline"
            return BenchmarkGeneration(
                output_text=f"<final_answer>{output}</final_answer>",
                prompt_tokens=32,
                completion_tokens=1,
                ttft_seconds=0.5 if request.arm.uses_cache else 1.0,
                time_to_completion_seconds=0.7 if request.arm.uses_cache else 1.2,
            )

    def official_score(context):
        return (
            {"official_f1": 0.94, "error_rate": 0.06}
            if context.output_text == "candidate"
            else {"official_f1": 0.95, "error_rate": 0.05}
        )

    scorer = DatasetScorer(
        scorer_id="tests.official",
        version="1",
        metric_names=("official_f1", "error_rate"),
        metric_specs=(
            DatasetMetricSpec("official_f1", max_regression=0.02),
            DatasetMetricSpec(
                "error_rate",
                direction="lower_is_better",
                max_regression=0.02,
            ),
        ),
        score_function=official_score,
        publication_approved=True,
        plugin_path="tests.test_benchmark_gates:official_score",
    )
    result = run_benchmark_suite(
        suite,
        {
            baseline_arm.arm_id: _OfficialEngine(),
            cache_arm.arm_id: _OfficialEngine(),
        },
        arms=(baseline_arm, cache_arm),
        repeats=3,
        scorer_registry=DatasetScorerRegistry().register("hotpotqa", scorer),
            manifest_context=BenchmarkManifestContext(
                model_revision="model-revision",
                canonical_model_id=identity.model_id,
                tokenizer_id=suite.model_id,
                tokenizer_revision="tokenizer-revision",
                lora_id=identity.lora_id,
                engine_id="test-engine",
                engine_version="1",
                serving_platform="test-serving-platform",
                model_dtype="float16",
                runtime_kv_dtype=identity.runtime_kv_dtype,
                layout_version=identity.layout_version,
                payload_axis_order=identity.payload_axis_order,
                block_size=identity.block_size,
                key_position_encoding=identity.key_position_encoding,
                rope_theta=identity.rope_theta,
                rope_rotary_dim=identity.rope_rotary_dim,
                tensor_parallel_size=identity.tensor_parallel_size,
                pipeline_parallel_size=identity.pipeline_parallel_size,
            hardware_fingerprint="test-hardware",
            runtime_id="test-runtime",
            runtime_version="1",
            storage_identity="test-storage",
            cache_state="warm",
            max_output_tokens=16,
            temperature=0.0,
            stream=True,
            complete_dataset_split=True,
            measurement_scopes=("quality",),
        ),
        evidence_policy="publication",
    )
    assert result.experiment_manifest is not None
    result = replace(
        result,
        experiment_manifest=replace(
            result.experiment_manifest,
            execution_isolation_mode="separate_process_or_job",
            order_mode="physically_isolated_jobs",
            source_execution_ids=(
                (baseline_arm.arm_id, "a" * 64),
                (cache_arm.arm_id, "b" * 64),
            ),
        ),
        execution_isolation_mode="separate_process_or_job",
    )

    gate = evaluate_benchmark_evidence_gate(
        result,
        policy="publication",
        artifact_identities={identity.artifact_id: identity},
    )
    paired = paired_benchmark_statistics(result, bootstrap_samples=100)[0]

    assert gate.ok, gate.issues
    assert set(paired.quality_score_deltas) == {"official_f1", "error_rate"}
    assert paired.quality_score_deltas["official_f1"].independent_examples == 4
    assert paired.quality_score_deltas["official_f1"].paired_samples == 4


def test_multi_document_vanilla_quality_regression_is_reported_not_gated() -> None:
    identity = artifact_identity()
    baseline_arm = baseline_prefill_arm()
    cache_arm = method_benchmark_arm(
        identity.method_id,
        method_config_digest=identity.method_config_digest,
    )
    examples = tuple(
        BenchmarkExample(
            example_id=f"multi-document-{index}",
            dataset="hotpotqa",
            documents=(
                SourceDocument.from_text(
                    document_id=f"bridge-{index}-a",
                    text="The bridge fact points to Paris.",
                ),
                SourceDocument.from_text(
                    document_id=f"bridge-{index}-b",
                    text="Paris is the final answer.",
                ),
            ),
            query="What is the final answer?",
            expected_answer="Paris",
            kv_transfer_params={
                DOCUMENT_KV_REQUEST_ID_PARAM: f"quality-cache-{index}",
                DOCUMENT_KV_HANDOFF_JSON_PARAM: f"/tmp/quality-{index}.json",
                DOCUMENT_KV_CACHE_METHOD_PARAM: identity.method_id,
                DOCUMENT_KV_ARTIFACT_ID_PARAM: identity.artifact_id,
            },
        )
        for index in range(4)
    )

    class _QualityEngine:
        def generate(self, request):
            return BenchmarkGeneration(
                output_text=(
                    "<final_answer>Lyon</final_answer>"
                    if request.arm.uses_cache
                    else "<final_answer>Paris</final_answer>"
                ),
                prompt_tokens=64,
                completion_tokens=1,
                ttft_seconds=0.5 if request.arm.uses_cache else 1.0,
                time_to_completion_seconds=0.7 if request.arm.uses_cache else 1.2,
            )

    context = BenchmarkManifestContext(
        model_revision=identity.model_revision,
        canonical_model_id=identity.model_id,
        tokenizer_id=identity.tokenizer_id,
        tokenizer_revision=identity.tokenizer_revision,
        lora_id=identity.lora_id,
        engine_id="test-engine",
        engine_version="1",
        serving_platform="test-serving-platform",
        model_dtype="float16",
        runtime_kv_dtype=identity.runtime_kv_dtype,
        layout_version=identity.layout_version,
        payload_axis_order=identity.payload_axis_order,
        block_size=identity.block_size,
        key_position_encoding=identity.key_position_encoding,
        rope_theta=identity.rope_theta,
        rope_rotary_dim=identity.rope_rotary_dim,
        tensor_parallel_size=identity.tensor_parallel_size,
        pipeline_parallel_size=identity.pipeline_parallel_size,
        hardware_fingerprint="test-hardware",
        runtime_id="test-runtime",
        runtime_version="1",
        storage_identity="test-storage",
        cache_state="warm",
        max_output_tokens=16,
        temperature=0.0,
        stream=True,
        complete_dataset_split=True,
        measurement_scopes=("quality",),
    )
    result = run_benchmark_suite(
        BenchmarkSuite(
            suite_id="multi-document-quality-regression",
            examples=examples,
            datasets=("hotpotqa",),
        ),
        {
            baseline_arm.arm_id: _QualityEngine(),
            cache_arm.arm_id: _QualityEngine(),
        },
        arms=(baseline_arm, cache_arm),
        scorer_registry=default_dataset_scorer_registry(),
        manifest_context=context,
        evidence_policy="publication",
    )
    assert result.experiment_manifest is not None
    result = replace(
        result,
        experiment_manifest=replace(
            result.experiment_manifest,
            execution_isolation_mode="separate_process_or_job",
            order_mode="physically_isolated_jobs",
            source_execution_ids=(
                (baseline_arm.arm_id, "a" * 64),
                (cache_arm.arm_id, "b" * 64),
            ),
        ),
        execution_isolation_mode="separate_process_or_job",
    )

    gate = evaluate_benchmark_evidence_gate(
        result,
        policy="publication",
        artifact_identities={identity.artifact_id: identity},
    )

    assert gate.ok, gate.issues
    comparison = result.comparisons[0]
    assert comparison.quality_score_deltas["exact_match"] == -1.0

    strict = evaluate_benchmark_publication_gate(
        replace(
            result,
            experiment_manifest=replace(
                result.experiment_manifest,
                complete_dataset_split=False,
            ),
        ),
        config=BenchmarkPublicationGateConfig(
            min_successful_requests_per_row=1,
            min_distinct_examples_per_row=1,
            min_quality_examples_per_row=5,
            allow_complete_split_below_quality_min=False,
            min_paired_samples=1,
            min_paired_examples=1,
        ),
        artifact_identities={identity.artifact_id: identity},
    )
    assert any("has 4 unique examples" in issue for issue in strict.issues)


def test_canary_gate_checks_multiple_unique_cache_salts_without_cross_product() -> None:
    identity = artifact_identity()
    original = scoped_benchmark_result(
        identity,
        scopes=("latency",),
        with_answers=False,
    )
    assert original.experiment_manifest is not None
    measurements = tuple(
        replace(
            measurement,
            metadata={
                **measurement.metadata,
                **(
                    {"prefix_cache_salt": measurement.request_id}
                    if measurement.arm_id in original.cache_arm_ids
                    else {}
                ),
            },
        )
        for measurement in original.measurements
    )
    result = replace(
        original,
        measurements=measurements,
        experiment_manifest=replace(original.experiment_manifest, cache_state="cold"),
        prefix_cache_salt_mode="per_request",
        evidence_policy="canary",
    )

    gate = evaluate_benchmark_evidence_gate(result, policy="canary")

    assert gate.ok, gate.issues
    assert gate.checked_cache_requests == 4


def test_publication_gate_rejects_quality_drop_and_unattested_cache_state() -> None:
    identity = artifact_identity()
    original = benchmark_result(identity)
    degraded_comparison = replace(
        original.comparisons[0],
        exact_match_delta=-0.25,
        answer_found_delta=-0.5,
    )
    result = replace(original, comparisons=(degraded_comparison,))

    gate = evaluate_benchmark_publication_gate(
        result,
        config=BenchmarkPublicationGateConfig(require_resolved_artifact_identity=False),
    )

    assert not gate.ok
    assert any("exact-match drop" in issue for issue in gate.issues)
    assert any("answer-found drop" in issue for issue in gate.issues)
    assert any("has no cache-state attestation" in issue for issue in gate.issues)


def test_vllm_telemetry_parser_requires_explicit_successful_eviction() -> None:
    identity = artifact_identity()
    telemetry = {
        "record_type": "document_kv.vllm_native_provider_load.v1",
        "request_id": "cache-request-1",
        "cache_state_attestation": {
            "cache_method": identity.method_id,
            "artifact_id": identity.artifact_id,
            "source": "local_path",
            "bytes_read": 4096,
            "payload_cache_hit": False,
            "eviction_requested": True,
            "eviction_succeeded": True,
            "direct_io": False,
            "expected_bytes": 4096,
            "expected_tokens": 128,
            "loaded_tokens": 128,
            "successful_loads": 1,
            "cold_read_attested": True,
        },
    }

    attestation = cache_state_attestation_from_vllm_telemetry(telemetry)

    assert attestation.cold_read_attested
    warm = replace(attestation, payload_cache_hit=True)
    assert not warm.cold_read_attested


def test_vllm_telemetry_parser_joins_explicit_benchmark_request_id_without_runtime_id_heuristics() -> None:
    identity = artifact_identity()
    telemetry = {
        "record_type": "document_kv.vllm_native_provider_load.v1",
        "request_id": "cmpl-cache-request-1-0-deadbeef",
        "benchmark_request_id": "cache-request-1",
        "cache_state_attestation": {
            "cache_method": identity.method_id,
            "artifact_id": identity.artifact_id,
            "source": "local_path",
            "bytes_read": 4096,
            "payload_cache_hit": False,
            "eviction_requested": True,
            "eviction_succeeded": True,
            "direct_io": False,
            "expected_bytes": 4096,
            "expected_tokens": 1,
            "loaded_tokens": 1,
            "successful_loads": 1,
        },
    }

    attestation = cache_state_attestation_from_vllm_telemetry(telemetry)
    gate = evaluate_benchmark_publication_gate(
        benchmark_result(identity),
        cache_state_attestations=(attestation,),
        artifact_identities={identity.artifact_id: identity},
    )

    assert attestation.request_id == "cache-request-1"
    assert gate.cold_attested_requests == 1
    assert not any("has no cache-state attestation" in issue for issue in gate.issues)

    legacy = dict(telemetry)
    legacy.pop("benchmark_request_id")
    legacy_attestation = cache_state_attestation_from_vllm_telemetry(legacy)
    assert legacy_attestation.request_id == "cmpl-cache-request-1-0-deadbeef"


def test_vllm_telemetry_correlation_survives_sanitized_publication_serialization() -> None:
    identity = artifact_identity()
    telemetry = {
        "record_type": "document_kv.vllm_native_provider_load.v1",
        "request_id": "cmpl-cache-request-1-0-deadbeef",
        "benchmark_request_id": "cache-request-1",
        "cache_state_attestation": {
            "cache_method": identity.method_id,
            "artifact_id": identity.artifact_id,
            "source": "local_path",
            "bytes_read": 4096,
            "payload_cache_hit": False,
            "eviction_requested": True,
            "eviction_succeeded": True,
            "direct_io": False,
            "expected_bytes": 4096,
            "expected_tokens": 1,
            "loaded_tokens": 1,
            "successful_loads": 1,
        },
    }
    attestation = cache_state_attestation_from_vllm_telemetry(telemetry)
    result = replace(benchmark_result(identity), evidence_policy="publication")

    record = benchmark_run_result_to_evidence_record(
        result,
        cache_state_attestations=(attestation,),
        artifact_identities={identity.artifact_id: identity},
    )

    expected_request_id = sha256(b"cache-request-1").hexdigest()
    cache_measurement = next(
        measurement
        for measurement in record["measurements"]
        if measurement["arm_id"] != result.baseline_arm_id
    )
    gate_attestation = record["gate_inputs"]["cache_state_attestations"][0]
    assert cache_measurement["request_id"] == expected_request_id
    assert gate_attestation["request_id"] == expected_request_id
    assert record["evidence_gate"]["cold_attested_requests"] == 1
    assert not any(
        "has no cache-state attestation" in issue
        for issue in record["evidence_gate"]["issues"]
    )


def test_paired_statistics_are_deterministic_and_method_aware() -> None:
    identity = artifact_identity()
    original = benchmark_result(identity)
    baseline, cache = original.measurements
    result = replace(
        original,
        measurements=(
            baseline,
            cache,
            replace(
                baseline,
                request_id="baseline-request-2",
                repeat_index=2,
                ttft_seconds=2.0,
                time_to_completion_seconds=2.4,
            ),
            replace(
                cache,
                request_id="cache-request-2",
                repeat_index=2,
                ttft_seconds=1.0,
                time_to_completion_seconds=1.4,
            ),
        ),
    )

    first = paired_benchmark_statistics(result, bootstrap_samples=100, seed=17)
    second = paired_benchmark_statistics(result, bootstrap_samples=100, seed=17)

    assert first == second
    assert len(first) == 1
    row = first[0]
    assert row.complete
    assert row.paired_samples == 2
    assert row.paired_examples == 1
    assert row.cache_method == identity.method_id
    assert row.artifact_id == identity.artifact_id
    assert row.ttft_speedup is not None
    assert row.ttft_speedup.estimate == 2.0
    serialized = paired_benchmark_statistics_to_record(first)["rows"][0]
    assert serialized["complete"] is True
    assert serialized["paired_median_ttft_speedup"]["bootstrap_unit"] == "example"
    assert row.exact_match_delta is not None
    assert row.exact_match_delta.paired_samples == 1


def test_paired_speedup_is_median_of_request_ratios_not_ratio_of_medians() -> None:
    original = benchmark_result(artifact_identity())
    baseline, cache = original.measurements
    baseline_values = (1.0, 10.0, 100.0)
    cache_values = (1.0, 1.0, 50.0)
    measurements = tuple(
        measurement
        for repeat_index, (baseline_value, cache_value) in enumerate(
            zip(baseline_values, cache_values, strict=True),
            start=1,
        )
        for measurement in (
            replace(
                baseline,
                repeat_index=repeat_index,
                request_id=f"baseline-{repeat_index}",
                ttft_seconds=baseline_value,
                time_to_completion_seconds=baseline_value,
            ),
            replace(
                cache,
                repeat_index=repeat_index,
                request_id=f"cache-{repeat_index}",
                ttft_seconds=cache_value,
                time_to_completion_seconds=cache_value,
            ),
        )
    )

    row = paired_benchmark_statistics(
        replace(original, measurements=measurements),
        bootstrap_samples=100,
    )[0]

    assert row.ttft_speedup is not None
    assert row.ttft_speedup.estimate == 2.0
    assert row.ttft_speedup.estimate != 10.0


def test_paired_statistics_accepts_python_module_result_identity() -> None:
    result = benchmark_result(artifact_identity())
    module_result = SimpleNamespace(
        measurements=result.measurements,
        baseline_arm_id=result.baseline_arm_id,
        cache_arm_ids=result.cache_arm_ids,
    )

    assert paired_benchmark_statistics(
        module_result,
        bootstrap_samples=100,
    ) == paired_benchmark_statistics(
        result,
        bootstrap_samples=100,
    )


class _ScopedEngine:
    def generate(self, request) -> BenchmarkGeneration:
        output = (
            "<final_answer>Paris</final_answer>"
            if request.example.expected_answer is not None
            else ""
        )
        return BenchmarkGeneration(
            output_text=output,
            prompt_tokens=32,
            completion_tokens=1,
            ttft_seconds=0.5 if request.arm.uses_cache else 1.0,
            time_to_completion_seconds=0.7 if request.arm.uses_cache else 1.2,
        )


def scoped_benchmark_result(
    identity: ArtifactIdentity,
    *,
    scopes: tuple[str, ...],
    with_answers: bool,
) -> BenchmarkRunResult:
    baseline_arm = baseline_prefill_arm()
    cache_arm = method_benchmark_arm(
        identity.method_id,
        method_config_digest="1" * 64,
    )
    examples = tuple(
        BenchmarkExample(
            example_id=f"example-{index}",
            dataset="hotpotqa",
            documents=(
                SourceDocument.from_text(
                    document_id=f"document-{index}",
                    text="The answer is Paris.",
                ),
            ),
            query="What is the answer?",
            expected_answer="Paris" if with_answers else None,
            kv_transfer_params={
                DOCUMENT_KV_REQUEST_ID_PARAM: f"cache-request-{index}",
                DOCUMENT_KV_HANDOFF_JSON_PARAM: f"/tmp/cache-request-{index}.json",
                DOCUMENT_KV_CACHE_METHOD_PARAM: identity.method_id,
                DOCUMENT_KV_ARTIFACT_ID_PARAM: identity.artifact_id,
            },
        )
        for index in range(4)
    )
    suite = BenchmarkSuite(
        suite_id="scoped-publication",
        examples=examples,
        datasets=("hotpotqa",),
    )
    scorer = DatasetScorer(
        scorer_id="test.hotpotqa",
        version="1",
        metric_names=("exact_match", "answer_found"),
        score_function=diagnostic_answer_scores,
        publication_approved=True,
        plugin_path="tests.test_benchmark_gates:diagnostic_answer_scores",
    )
    scorers = DatasetScorerRegistry().register("hotpotqa", scorer)
    context = BenchmarkManifestContext(
        model_revision="model-revision",
        canonical_model_id=identity.model_id,
        tokenizer_id=suite.model_id,
        tokenizer_revision="tokenizer-revision",
        lora_id=identity.lora_id,
        engine_id="test-engine",
        engine_version="1",
        serving_platform="test-serving-platform",
        model_dtype="float16",
        runtime_kv_dtype=identity.runtime_kv_dtype,
        layout_version=identity.layout_version,
        payload_axis_order=identity.payload_axis_order,
        block_size=identity.block_size,
        key_position_encoding=identity.key_position_encoding,
        rope_theta=identity.rope_theta,
        rope_rotary_dim=identity.rope_rotary_dim,
        tensor_parallel_size=identity.tensor_parallel_size,
        pipeline_parallel_size=identity.pipeline_parallel_size,
        hardware_fingerprint="test-hardware",
        runtime_id="test-runtime",
        runtime_version="1",
        storage_identity="test-storage",
        cache_state="warm",
        max_output_tokens=16,
        temperature=0.0,
        stream=True,
        complete_dataset_split=True,
        measurement_scopes=scopes,
    )
    result = run_benchmark_suite(
        suite,
        {
            baseline_arm.arm_id: _ScopedEngine(),
            cache_arm.arm_id: _ScopedEngine(),
        },
        arms=(baseline_arm, cache_arm),
        scorer_registry=scorers,
        manifest_context=context,
        evidence_policy="publication",
    )
    assert result.experiment_manifest is not None
    return replace(
        result,
        experiment_manifest=replace(
            result.experiment_manifest,
            execution_isolation_mode="separate_process_or_job",
            order_mode="physically_isolated_jobs",
            source_execution_ids=(
                (baseline_arm.arm_id, "a" * 64),
                (cache_arm.arm_id, "b" * 64),
            ),
        ),
        execution_isolation_mode="separate_process_or_job",
    )


def artifact_identity() -> ArtifactIdentity:
    return ArtifactIdentity(
        method_id="vanilla_prefill",
        method_version="2",
        method_config_digest="1" * 64,
        model_id="qwen3:4b-instruct",
        model_revision="model-revision",
        tokenizer_id="qwen3:4b-instruct",
        tokenizer_revision="tokenizer-revision",
        lora_id="none",
        prompt_template_version=DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
        layout_version="v1",
        kv_dtype="float16",
        block_size=16,
        payload_axis_order="layer,kv,token,head,dim",
        key_position_encoding="pre_rope",
        rope_theta=5_000_000.0,
        rope_rotary_dim=128,
        generator_family="transformers",
        generator_version="4.53.0",
    )


def benchmark_result(identity: ArtifactIdentity) -> BenchmarkRunResult:
    baseline_arm = baseline_prefill_arm()
    cache_arm = method_benchmark_arm(
        identity.method_id,
        method_config_digest=identity.method_config_digest,
    )
    suite = BenchmarkSuite(
        suite_id="gate-test",
        examples=(
            BenchmarkExample(
                example_id="example-1",
                dataset="hotpotqa",
                documents=(
                    SourceDocument.from_text(
                        document_id="document-1",
                        text="The answer is Paris.",
                    ),
                ),
                query="What is the answer?",
                expected_answer="Paris",
            ),
        ),
    )
    measurements = (
        InferenceMeasurement(
            example_id="example-1",
            dataset="hotpotqa",
            arm_id=baseline_arm.arm_id,
            prompt_tokens=32,
            completion_tokens=1,
            ttft_seconds=1.0,
            time_to_completion_seconds=1.2,
            output_text="Paris",
            expected_answer="Paris",
            request_id="baseline-request-1",
        ),
        InferenceMeasurement(
            example_id="example-1",
            dataset="hotpotqa",
            arm_id=cache_arm.arm_id,
            prompt_tokens=32,
            completion_tokens=1,
            ttft_seconds=0.5,
            time_to_completion_seconds=0.7,
            output_text="Paris",
            expected_answer="Paris",
            cache_method=identity.method_id,
            artifact_id=identity.artifact_id,
            variant_id="default",
            metadata={
                "prefix_cache_salt_attached": "true",
                "prefix_cache_salt": "cache-request-1",
            },
            request_id="cache-request-1",
        ),
    )
    latency = LatencySummary(count=1, mean=0.5, p50=0.5, p95=0.5)
    rows = (
        BenchmarkReportRow(
            dataset="hotpotqa",
            arm_id=baseline_arm.arm_id,
            requests=1,
            errors=0,
            prompt_tokens_mean=32.0,
            completion_tokens_mean=1.0,
            ttft=LatencySummary(count=1, mean=1.0, p50=1.0, p95=1.0),
            time_to_completion=LatencySummary(count=1, mean=1.2, p50=1.2, p95=1.2),
            exact_match_rate=1.0,
            answer_found_rate=1.0,
            output_tokens_per_second=5.0,
        ),
        BenchmarkReportRow(
            dataset="hotpotqa",
            arm_id=cache_arm.arm_id,
            requests=1,
            errors=0,
            prompt_tokens_mean=32.0,
            completion_tokens_mean=1.0,
            ttft=latency,
            time_to_completion=LatencySummary(count=1, mean=0.7, p50=0.7, p95=0.7),
            exact_match_rate=1.0,
            answer_found_rate=1.0,
            output_tokens_per_second=5.0,
            cache_method=identity.method_id,
            artifact_id=identity.artifact_id,
            variant_id="default",
        ),
    )
    comparisons = (
        BenchmarkComparison(
            dataset="hotpotqa",
            baseline_arm_id=baseline_arm.arm_id,
            cache_arm_id=cache_arm.arm_id,
            ttft_speedup=2.0,
            time_to_completion_speedup=1.7,
            exact_match_delta=0.0,
            answer_found_delta=0.0,
            cache_method=identity.method_id,
            artifact_id=identity.artifact_id,
            variant_id="default",
        ),
    )
    return BenchmarkRunResult(
        suite=suite,
        measurements=measurements,
        report_rows=rows,
        comparisons=comparisons,
        baseline_arm_id=baseline_arm.arm_id,
        cache_arm_id=cache_arm.arm_id,
        arms=(baseline_arm, cache_arm),
        prefix_cache_salt_mode="per_request",
    )


def cold_attestation(identity: ArtifactIdentity) -> CacheStateAttestation:
    return CacheStateAttestation(
        request_id="cache-request-1",
        cache_method=identity.method_id,
        artifact_id=identity.artifact_id,
        source="local_path",
        bytes_read=4096,
        payload_cache_hit=False,
        eviction_requested=True,
        eviction_succeeded=True,
        expected_bytes=4096,
        expected_tokens=1,
        loaded_tokens=1,
        successful_loads=1,
    )
