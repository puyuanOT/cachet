from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

import pytest

from document_kv_cache.benchmark_gates import evaluate_benchmark_evidence_gate
from document_kv_cache.benchmark_runner import (
    BenchmarkGeneration,
    BenchmarkManifestContext,
    benchmark_record_payload_digest,
    benchmark_resource_evidence_to_record,
    benchmark_record_aggregate_issues,
    benchmark_run_result_from_record,
    benchmark_run_result_to_evidence_record,
    benchmark_run_result_to_record,
    merge_isolated_benchmark_run_records,
    run_benchmark_suite,
)
from document_kv_cache.benchmarks import (
    BASELINE_PREFILL_ARM,
    BenchmarkArm,
    BenchmarkExample,
    BenchmarkSuite,
)
from document_kv_cache.runtime_telemetry import (
    RUNTIME_TELEMETRY_RECORD_TYPE,
    attach_runtime_resource_evidence,
    bind_runtime_resource_evidence_record_file,
)
from document_kv_cache.workflow import SourceDocument


class _Engine:
    def generate(self, _request):
        return BenchmarkGeneration(
            output_text="<final_answer>Paris</final_answer>",
            prompt_tokens=8,
            completion_tokens=1,
            ttft_seconds=0.1,
            time_to_completion_seconds=0.2,
        )


def _bare_result():
    arms = _arms()
    return run_benchmark_suite(
        _suite(),
        {arm.arm_id: _Engine() for arm in arms},
        arms=arms,
        evidence_policy="canary",
        manifest_context=BenchmarkManifestContext(
            runtime_id="physical-execution-1",
            measurement_scopes=("resource",),
            package_revisions=_resource_package_revisions(),
        ),
    )


def _result():
    result = _bare_result()
    for index, window in enumerate(result.execution_windows):
        result = attach_runtime_resource_evidence(
            result,
            arm_id=window.arm_id,
            telemetry=_telemetry_for_window(window, gpu_bytes=1000 + index),
            source_revision="revision-1",
            source_tree_sha256="a" * 64,
            wheel_sha256="b" * 64,
            runner_sha256="c" * 64,
        )
    return result


def _arms():
    return (
        BenchmarkArm(BASELINE_PREFILL_ARM, False, "baseline"),
        BenchmarkArm(
            "baseline_alt",
            True,
            "second physical arm",
            cache_method="external-cache",
            connector_mode="external",
            variant_id="external-cache",
            implementation_kind="external",
            method_version="1",
            method_config_digest="d" * 64,
            source_revision="external-revision",
            checkpoint_identity="external-checkpoint",
            requires_cachet_handoff=False,
        ),
    )


def _suite():
    return BenchmarkSuite(
        suite_id="resource-evidence",
        datasets=("biography",),
        examples=tuple(
                BenchmarkExample(
                    example_id=f"resource-{index}",
                    dataset="biography",
                    documents=(
                        SourceDocument.from_text(
                            document_id=f"resource-document-{index}",
                            text="Paris is the answer.",
                        ),
                    ),
                    query="What is the answer?",
                    expected_answer="Paris",
                )
                for index in range(2)
        ),
    )


def _resource_package_revisions():
    return (
        ("cachet-source", "git:revision-1"),
        ("cachet-source-tree", f"sha256:{'a' * 64}"),
        ("cachet-kv", f"wheel-sha256:{'b' * 64}"),
        ("cachet-runner", f"sha256:{'c' * 64}"),
    )


def _telemetry_for_window(window, *, gpu_bytes: int, complete: bool = True):
    assert window.started_at_seconds is not None
    assert window.ended_at_seconds is not None
    timestamp = (window.started_at_seconds + window.ended_at_seconds) / 2
    return {
        "record_type": RUNTIME_TELEMETRY_RECORD_TYPE,
        "interval_seconds": 1.0,
        "samples": [
            {
                "timestamp_seconds": timestamp,
                "process_tree": {
                    "ok": True,
                    "rss_bytes": 2000,
                },
                "host_memory": {
                    "ok": True,
                    "used_bytes": 3000,
                },
                "gpu": {
                    "ok": True,
                    "devices": [
                        {
                            "ok": True,
                            "utilization_percent": 75.0,
                        }
                    ],
                    "processes": {
                        "ok": complete,
                        "process_tree_used_bytes": gpu_bytes,
                    },
                },
            }
        ],
        "errors": [],
    }


def test_resource_evidence_round_trip_and_baseline_gate():
    result = _result()

    gate = evaluate_benchmark_evidence_gate(result, policy="canary")
    record = benchmark_run_result_to_record(result)
    reconstructed = benchmark_run_result_from_record(record)

    assert gate.ok, gate.issues
    assert {item.arm_id for item in result.resource_evidence} == {
        BASELINE_PREFILL_ARM,
        "baseline_alt",
    }
    assert reconstructed.resource_evidence == result.resource_evidence
    assert benchmark_record_aggregate_issues(record) == ()


def test_finalized_telemetry_is_bound_before_the_benchmark_file_is_published(
    tmp_path,
):
    bare_result = _bare_result()
    initial_record = benchmark_run_result_to_record(bare_result)
    benchmark_path = tmp_path / "v1-benchmark.json"
    telemetry_path = tmp_path / "runtime-telemetry.json"
    telemetry = {
        "record_type": RUNTIME_TELEMETRY_RECORD_TYPE,
        "interval_seconds": 1.0,
        "samples": [
            sample
            for index, window in enumerate(bare_result.execution_windows)
            for sample in _telemetry_for_window(
                window,
                gpu_bytes=1000 + index,
            )["samples"]
        ],
        "errors": [],
    }
    benchmark_path.write_text(
        json.dumps(initial_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    telemetry_path.write_text(
        json.dumps(telemetry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    telemetry_digest = sha256(telemetry_path.read_bytes()).hexdigest()

    bound = bind_runtime_resource_evidence_record_file(
        benchmark_path,
        telemetry_path,
    )
    persisted = json.loads(benchmark_path.read_text(encoding="utf-8"))

    assert initial_record["evidence_gate"]["ok"] is False
    assert bound == persisted
    assert persisted["evidence_gate"]["ok"] is True
    assert persisted["evidence_gate"]["benchmark_payload_digest"] == (
        benchmark_record_payload_digest(persisted)
    )
    assert {
        item["telemetry_sha256"]
        for item in persisted["resource_evidence"]
    } == {telemetry_digest}
    assert benchmark_record_aggregate_issues(persisted) == ()


def test_resource_evidence_tamper_is_rejected():
    record = benchmark_run_result_to_record(_result())
    record["resource_evidence"][0]["metrics"][
        "peak_gpu_process_memory_bytes"
    ] += 1

    issues = benchmark_record_aggregate_issues(record)

    assert any("record_sha256" in issue for issue in issues)


def test_resource_scope_fails_closed_when_baseline_evidence_is_missing():
    result = _result()
    baseline_missing = tuple(
        item
        for item in result.resource_evidence
        if item.arm_id != BASELINE_PREFILL_ARM
    )
    assert result.experiment_manifest is not None
    manifest = replace(
        result.experiment_manifest,
        resource_evidence_ids=tuple(
            item
            for item in result.experiment_manifest.resource_evidence_ids
            if item[0] != BASELINE_PREFILL_ARM
        ),
    )

    gate = evaluate_benchmark_evidence_gate(
        replace(
            result,
            experiment_manifest=manifest,
            resource_evidence=baseline_missing,
        ),
        policy="canary",
    )

    assert not gate.ok
    assert any(
        "baseline_prefill" in issue and "no governed" in issue
        for issue in gate.issues
    )


def test_resource_scope_rejects_incomplete_telemetry():
    result = _result()
    baseline_window = next(
        window
        for window in result.execution_windows
        if window.arm_id == BASELINE_PREFILL_ARM
    )
    incomplete = attach_runtime_resource_evidence(
        result,
        arm_id=BASELINE_PREFILL_ARM,
        telemetry=_telemetry_for_window(
            baseline_window,
            gpu_bytes=1000,
            complete=False,
        ),
        source_revision="revision-1",
        source_tree_sha256="a" * 64,
        wheel_sha256="b" * 64,
        runner_sha256="c" * 64,
    )

    gate = evaluate_benchmark_evidence_gate(incomplete, policy="canary")

    assert not gate.ok
    assert any("telemetry coverage is incomplete" in issue for issue in gate.issues)


def test_matching_forged_arm_closures_do_not_bypass_manifest_binding():
    result = _result()
    forged_evidence = tuple(
        replace(item, source_revision="forged-revision")
        for item in result.resource_evidence
    )
    assert result.experiment_manifest is not None
    manifest = replace(
        result.experiment_manifest,
        resource_evidence_ids=tuple(
            (
                item.arm_id,
                str(
                    benchmark_resource_evidence_to_record(item)["record_sha256"]
                ),
            )
            for item in forged_evidence
        ),
    )

    gate = evaluate_benchmark_evidence_gate(
        replace(
            result,
            experiment_manifest=manifest,
            resource_evidence=forged_evidence,
        ),
        policy="canary",
    )

    assert not gate.ok
    assert all(item.source_revision == "forged-revision" for item in forged_evidence)
    assert any("does not match the manifest" in issue for issue in gate.issues)


def test_manifest_software_closure_tampering_invalidates_runtime_binding():
    result = _result()
    assert result.experiment_manifest is not None
    tampered_revisions = tuple(
        (name, "git:forged-revision" if name == "cachet-source" else value)
        for name, value in result.experiment_manifest.package_revisions
    )

    with pytest.raises(ValueError, match="runtime identity"):
        evaluate_benchmark_evidence_gate(
            replace(
                result,
                experiment_manifest=replace(
                    result.experiment_manifest,
                    package_revisions=tampered_revisions,
                ),
            ),
            policy="canary",
        )


def test_historical_untimestamped_window_cannot_be_post_hoc_bound():
    result = _result()
    window = result.execution_windows[0]
    historical = replace(
        result,
        execution_windows=(
            replace(window, started_at_seconds=None, ended_at_seconds=None),
            *result.execution_windows[1:],
        ),
        resource_evidence=(),
        experiment_manifest=replace(
            result.experiment_manifest,
            resource_evidence_ids=(),
        ),
    )

    try:
        attach_runtime_resource_evidence(
            historical,
            arm_id=window.arm_id,
            telemetry=_telemetry_for_window(window, gpu_bytes=1000),
            source_revision="revision-1",
            source_tree_sha256="a" * 64,
            wheel_sha256="b" * 64,
            runner_sha256="c" * 64,
        )
    except ValueError as exc:
        assert "historical run" in str(exc)
    else:  # pragma: no cover - fail-closed assertion.
        raise AssertionError("historical resource evidence was accepted")


def test_isolated_merge_preserves_per_arm_resource_evidence():
    records = []
    for index, arm in enumerate(_arms()):
        result = run_benchmark_suite(
            _suite(),
            {arm.arm_id: _Engine()},
            arms=(arm,),
            evidence_policy="canary",
            manifest_context=BenchmarkManifestContext(
                runtime_id=f"isolated-execution-{index}",
                measurement_scopes=("resource",),
                package_revisions=_resource_package_revisions(),
            ),
        )
        result = attach_runtime_resource_evidence(
            result,
            arm_id=arm.arm_id,
            telemetry=_telemetry_for_window(
                result.execution_windows[0],
                gpu_bytes=1000 + index,
            ),
            source_revision="revision-1",
            source_tree_sha256="a" * 64,
            wheel_sha256="b" * 64,
            runner_sha256="c" * 64,
        )
        records.append(benchmark_run_result_to_evidence_record(result))

    merged = merge_isolated_benchmark_run_records(
        records,
        reference_arm_id=BASELINE_PREFILL_ARM,
        policy="canary",
    )

    assert merged["evidence_gate"]["ok"] is True
    assert {
        item["arm_id"] for item in merged["resource_evidence"]
    } == {BASELINE_PREFILL_ARM, "baseline_alt"}
    assert benchmark_record_aggregate_issues(merged) == ()
