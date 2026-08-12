from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from document_kv_cache.artifact_identity import ArtifactIdentity
from document_kv_cache.benchmark_gates import (
    BenchmarkPublicationGateConfig,
    CacheStateAttestation,
    benchmark_publication_gate_to_record,
    cache_state_attestation_from_vllm_telemetry,
    evaluate_benchmark_publication_gate,
)
from document_kv_cache.benchmark_runner import BenchmarkRunResult
from document_kv_cache.benchmark_statistics import (
    paired_benchmark_statistics,
    paired_benchmark_statistics_to_record,
)
from document_kv_cache.benchmarks import (
    BenchmarkComparison,
    BenchmarkExample,
    BenchmarkReportRow,
    BenchmarkSuite,
    InferenceMeasurement,
    LatencySummary,
    baseline_prefill_arm,
    method_benchmark_arm,
)
from document_kv_cache.workflow import SourceDocument


def test_publication_gate_accepts_identified_quality_preserving_cold_run() -> None:
    identity = artifact_identity()
    result = benchmark_result(identity)
    gate = evaluate_benchmark_publication_gate(
        result,
        cache_state_attestations=(cold_attestation(identity),),
        artifact_identities={identity.artifact_id: identity},
    )

    assert gate.ok
    assert gate.issues == ()
    assert gate.checked_cache_requests == 1
    assert gate.cold_attested_requests == 1
    assert benchmark_publication_gate_to_record(gate)["ok"] is True


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
    assert row.cache_method == identity.method_id
    assert row.artifact_id == identity.artifact_id
    assert row.ttft_speedup is not None
    assert row.ttft_speedup.estimate == 2.0
    assert paired_benchmark_statistics_to_record(first)["rows"][0]["complete"] is True


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


def artifact_identity() -> ArtifactIdentity:
    return ArtifactIdentity(
        method_id="vanilla_prefill",
        method_version="1",
        method_config_digest="0" * 64,
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_revision="revision-1",
        tokenizer_id="meta-llama/Llama-3.1-8B-Instruct",
        tokenizer_revision="revision-1",
        lora_id="none",
        prompt_template_version="v1",
        layout_version="v1",
        kv_dtype="float16",
        block_size=16,
        payload_axis_order="layer,kv,token,head,dim",
        generator_family="transformers",
        generator_version="4.53.0",
    )


def benchmark_result(identity: ArtifactIdentity) -> BenchmarkRunResult:
    baseline_arm = baseline_prefill_arm()
    cache_arm = method_benchmark_arm(identity.method_id)
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
