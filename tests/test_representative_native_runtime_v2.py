"""Native-v2 transport preserves the representative experiment and ledger contract."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from document_kv_cache.canary_orchestration import (
    BASELINE_PREFILL_ARM,
    FULL_PREFIX_CANARY_ARM,
    REPRESENTATIVE_CANARY_MODEL_ID,
    REPRESENTATIVE_CANARY_MODEL_REVISION,
    REPRESENTATIVE_POST_ROPE_HANDOFF_GENERATOR_FACTORY,
    REPRESENTATIVE_PRE_ROPE_HANDOFF_GENERATOR_FACTORY,
    VANILLA_CANARY_ARM,
    representative_canary_matrix,
    representative_canary_workload_manifest,
    validate_representative_canary_reservation,
    validate_representative_canary_workload_payload,
)
from document_kv_cache.databricks_resource_ledger import (
    DatabricksRunAttemptReservationRequest,
    create_databricks_cluster_hour_ledger_json,
    databricks_ledger_prefix,
    read_databricks_cluster_hour_ledger_json,
    reserve_databricks_run_attempt_batch_authorized_json,
)
from document_kv_cache.databricks_runs import (
    bind_databricks_run_idempotency_token,
    require_databricks_run_idempotency_token,
)
from document_kv_cache.databricks_vllm_smoke_job import (
    DatabricksVLLMSmokeJobConfig,
    build_databricks_vllm_smoke_run_submit_payload,
    main,
)
from document_kv_cache.flashinfer_wheel_repack import FLASHINFER_PATCHED_WHEEL_SHA256
from document_kv_cache.runtime_artifact_closure import (
    RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
    VLLM_PATCHED_WHEEL_SHA256,
    VLLM_RUNTIME_BASE_LOCK_SHA256,
)
from document_kv_cache.vllm_smoke import (
    DOCUMENT_KV_PACKAGE_WHEEL_SHA256_ENV,
    VLLMNativeRuntimeBundleV2,
    parse_args,
    vllm_representative_workload_profile,
)


def runtime_bundle():
    records = {}
    for name, digest, filename in (
        ("package_wheel", "f" * 64, "cachet_kv-0.2.0-py3-none-any.whl"),
        ("runtime_lock", VLLM_RUNTIME_BASE_LOCK_SHA256, "runtime.lock"),
        ("patched_vllm_wheel", VLLM_PATCHED_WHEEL_SHA256, "vllm.whl"),
        ("patched_flashinfer_wheel", FLASHINFER_PATCHED_WHEEL_SHA256, "flashinfer.whl"),
        ("runtime_closure_manifest", RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256, "closure.json"),
    ):
        records[name + "_uri"] = f"/Volumes/catalog/schema/volume/{digest}/{filename}"
        records[name + "_sha256"] = digest
    return VLLMNativeRuntimeBundleV2.from_record(records)


def job_config(index=0, *, native=True):
    workload = representative_canary_workload_manifest().workloads[index]
    profile = vllm_representative_workload_profile(workload.profile_id)
    bundle = runtime_bundle()
    handoff = {}
    if workload.arm_id != BASELINE_PREFILL_ARM:
        handoff = {
            "benchmark_handoff_generator_factory": (
                REPRESENTATIVE_POST_ROPE_HANDOFF_GENERATOR_FACTORY
                if workload.arm_id == FULL_PREFIX_CANARY_ARM
                else REPRESENTATIVE_PRE_ROPE_HANDOFF_GENERATOR_FACTORY
            ),
            "benchmark_handoff_output_dir": f"/local_disk0/{workload.workload_id}/handoffs",
            "benchmark_handoff_cache_method": (
                "full_prefix_prefill" if workload.arm_id == FULL_PREFIX_CANARY_ARM
                else "vanilla_prefill"
            ),
            "benchmark_handoff_segment_per_document": workload.arm_id == VANILLA_CANARY_ARM,
        }
    return DatabricksVLLMSmokeJobConfig(
        benchmark_id=workload.workload_id,
        output_dir=f"/Volumes/catalog/schema/volume/block-0/{workload.workload_id}",
        runner_python_file=f"/Volumes/catalog/schema/volume/{workload.runner_sha256}/{workload.runner_basename}",
        hardware_target=workload.hardware_target,
        node_type_id=workload.node_type_id,
        single_user_name="user@example.com",
        wheel_uri=bundle.package_wheel_uri,
        wheel_sha256=bundle.package_wheel_sha256,
        native_runtime_v2=bundle if native else None,
        submission_attempt_id=f"block-0-{workload.workload_id}" if native else None,
        model_id=REPRESENTATIVE_CANARY_MODEL_ID,
        model_revision=REPRESENTATIVE_CANARY_MODEL_REVISION,
        tokenizer_revision=REPRESENTATIVE_CANARY_MODEL_REVISION,
        kv_cache_dtype="bfloat16",
        max_tokens=profile.max_output_tokens,
        max_model_len=profile.max_model_len,
        max_num_seqs=profile.max_num_seqs,
        gpu_memory_utilization=profile.gpu_memory_utilization,
        benchmark_repeats=profile.benchmark_repeats,
        request_parallelism=profile.request_parallelism,
        benchmark_arm_specs=(representative_canary_matrix().run_for_arm(workload.arm_id).arm_spec,),
        benchmark_evidence_policy="canary",
        representative_canary=True,
        representative_workload_profile=profile.profile_id,
        benchmark_manifest_provenance={"input_tokens_target": profile.input_tokens_target},
        benchmark_force_max_tokens=True,
        dataset_specs=("hotpotqa=/Volumes/catalog/schema/volume/hotpotqa.jsonl",),
        allow_dataset_subset=True,
        **handoff,
    )


def parameters(payload):
    return payload["tasks"][0]["spark_python_task"]["parameters"]


@pytest.mark.parametrize("index", (0, 1, 2, 3, 4, 5, 7, 8, 9))
def test_native_representative_round_trip_preserves_every_legacy_setting(index, monkeypatch):
    config = job_config(index)
    payload = build_databricks_vllm_smoke_run_submit_payload(config)
    legacy = build_databricks_vllm_smoke_run_submit_payload(job_config(index, native=False))
    transport_free = deepcopy(payload)
    transport_free.pop("idempotency_token")
    args = parameters(transport_free)
    offset = args.index("--native-runtime-v2-json")
    assert json.loads(args[offset + 1]) == config.native_runtime_v2.to_record()
    del args[offset:offset + 2]
    assert transport_free == legacy
    workload = representative_canary_workload_manifest().workloads[index]
    reservation = validate_representative_canary_workload_payload(
        workload, payload, attempt_id=config.submission_attempt_id
    )
    assert reservation.reserved_cluster_hours == 4.0
    validate_representative_canary_reservation(reservation, payload)

    args = parameters(deepcopy(payload))
    for flag in ("--package-wheel-uri", "--package-wheel-sha256"):
        offset = args.index(flag)
        del args[offset:offset + 2]
    monkeypatch.setenv(DOCUMENT_KV_PACKAGE_WHEEL_SHA256_ENV, config.wheel_sha256)
    parsed = parse_args(args)
    assert parsed.native_runtime_v2 == config.native_runtime_v2
    assert parsed.request_parallelism == config.request_parallelism == 1


def test_native_representative_cli_emits_same_bundle_and_attempt_token(tmp_path):
    config = job_config(1)
    expected = build_databricks_vllm_smoke_run_submit_payload(config)
    args = list(parameters(expected))
    for before, after in (("--package-wheel-uri", "--wheel-uri"),
                          ("--package-wheel-sha256", "--wheel-sha256")):
        args[args.index(before)] = after
    output = tmp_path / "payload.json"
    args.extend([
        "--runner-python-file", config.runner_python_file,
        "--single-user-name", config.single_user_name,
        "--submission-attempt-id", config.submission_attempt_id,
        "--output-json", str(output),
    ])
    assert main(args) == 0
    assert json.loads(output.read_text()) == expected


@pytest.mark.parametrize("change", (
    {"submission_attempt_id": None}, {"native_runtime_v2": None},
    {"native_runtime_v2": {}}, {"wheel_sha256": "a" * 64},
    {"wheel_uri": "/Volumes/catalog/schema/volume/other.whl"},
))
def test_native_representative_config_rejects_incomplete_or_conflicting_binding(change):
    with pytest.raises((TypeError, ValueError), match="native_runtime_v2|native-v2"):
        replace(job_config(), **change)


def test_native_representative_token_requires_the_actual_attempt_and_exact_bytes():
    config = job_config()
    payload = build_databricks_vllm_smoke_run_submit_payload(config)
    workload = representative_canary_workload_manifest().workloads[0]
    with pytest.raises(ValueError, match="requires attempt_id"):
        validate_representative_canary_workload_payload(workload, payload)
    for actual in ("other-attempt", config.submission_attempt_id):
        changed = deepcopy(payload)
        if actual == config.submission_attempt_id:
            changed["run_name"] += "-changed"
        with pytest.raises(ValueError, match="idempotency token"):
            validate_representative_canary_workload_payload(workload, changed, attempt_id=actual)
    payload.pop("idempotency_token")
    with pytest.raises(ValueError, match="missing idempotency_token"):
        validate_representative_canary_workload_payload(workload, payload, attempt_id=config.submission_attempt_id)


@pytest.mark.parametrize("field,value", (
    ("patched_vllm_wheel_sha256", "0" * 64),
    ("package_wheel_sha256", "0" * 64),
    ("package_wheel_uri", "/Volumes/catalog/schema/volume/other.whl"),
    ("runtime_lock_uri", "/local_disk0/runtime.lock"),
    ("runtime_lock_uri", "/Volumes/catalog/schema/volume/runtime.lock"),
    ("unexpected", "value"),
))
def test_native_representative_closed_bundle_rejects_drift_even_with_a_fresh_token(field, value):
    config = job_config()
    payload = build_databricks_vllm_smoke_run_submit_payload(config)
    args = parameters(payload)
    offset = args.index("--native-runtime-v2-json") + 1
    record = json.loads(args[offset])
    record[field] = value
    args[offset] = json.dumps(record)
    payload.pop("idempotency_token")
    payload = bind_databricks_run_idempotency_token(payload, attempt_id=config.submission_attempt_id)
    with pytest.raises(ValueError):
        validate_representative_canary_workload_payload(
            representative_canary_workload_manifest().workloads[0], payload,
            attempt_id=config.submission_attempt_id,
        )


@pytest.mark.parametrize("change", ("duplicate_bundle", "unknown_flag", "concurrency", "extra_key"))
def test_native_successor_keeps_the_closed_parameter_and_scientific_contract(change):
    config = job_config()
    payload = build_databricks_vllm_smoke_run_submit_payload(config)
    args = parameters(payload)
    if change == "duplicate_bundle":
        offset = args.index("--native-runtime-v2-json")
        args.extend(args[offset:offset + 2])
    elif change == "unknown_flag":
        args.extend(["--invented-runtime-option", "true"])
    elif change == "concurrency":
        args[args.index("--request-parallelism") + 1] = "2"
    else:
        payload["unexpected"] = "value"
    payload.pop("idempotency_token")
    payload = bind_databricks_run_idempotency_token(payload, attempt_id=config.submission_attempt_id)
    with pytest.raises(ValueError):
        validate_representative_canary_workload_payload(
            representative_canary_workload_manifest().workloads[0], payload,
            attempt_id=config.submission_attempt_id,
        )


def test_native_representative_three_arm_group_uses_existing_atomic_12_hour_ledger_api(tmp_path):
    path = tmp_path / "ledger.json"
    opening = create_databricks_cluster_hour_ledger_json(path, ledger_id="supplement")
    requests = []
    for index in range(3):
        config = job_config(index)
        payload = build_databricks_vllm_smoke_run_submit_payload(config)
        require_databricks_run_idempotency_token(payload, attempt_id=config.submission_attempt_id)
        requests.append(DatabricksRunAttemptReservationRequest(
            attempt_id=config.submission_attempt_id,
            workload_id=config.benchmark_id,
            submit_payload=payload,
        ))

    def validate_group(_ledger, reservations, snapshots):
        for reservation, snapshot in zip(reservations, snapshots, strict=True):
            validate_representative_canary_reservation(reservation, snapshot)

    updated, authority = reserve_databricks_run_attempt_batch_authorized_json(
        path, requests, expected_predecessor_prefix=databricks_ledger_prefix(opening),
        batch_validator=validate_group,
    )
    assert updated.active_reserved_cluster_hours == 12
    assert updated.active_reserved_task_count == 3
    assert authority.attempt_ids == tuple(request.attempt_id for request in requests)
    assert read_databricks_cluster_hour_ledger_json(path) == updated
