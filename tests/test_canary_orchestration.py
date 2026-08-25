import hashlib
import json
import os
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest

from document_kv_cache.artifact_identity import ArtifactIdentity, method_config_digest
from document_kv_cache.benchmark_handoffs import (
    BenchmarkHandoffEntry,
    BenchmarkHandoffManifest,
    write_benchmark_handoff_manifest_json,
)
from document_kv_cache.benchmark_runner import (
    BenchmarkGeneration,
    BenchmarkManifestContext,
    benchmark_record_aggregate_issues,
    benchmark_run_result_to_record,
    run_benchmark_suite,
)
from document_kv_cache.benchmarks import (
    BASELINE_PREFILL_ARM,
    DOCUMENT_KV_ARTIFACT_ID_PARAM,
    DOCUMENT_KV_CACHE_METHOD_PARAM,
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_REQUEST_ID_PARAM,
    BenchmarkArm,
    BenchmarkExample,
    BenchmarkSuite,
)
from document_kv_cache.canary_orchestration import (
    FULL_PREFIX_CANARY_ARM,
    HANDOFF_TOPOLOGY_ATTESTATION_RECORD_TYPE,
    ISOLATED_CANARY_AGGREGATE_RECORD_TYPE,
    REPRESENTATIVE_CANARY_ARM_IDS,
    REPRESENTATIVE_CANARY_FIRST_WAVE_CLUSTER_HOURS,
    REPRESENTATIVE_HANDOFF_GENERATOR_FACTORY,
    REPRESENTATIVE_POST_ROPE_HANDOFF_GENERATOR_FACTORY,
    REPRESENTATIVE_PRE_ROPE_HANDOFF_GENERATOR_FACTORY,
    REPRESENTATIVE_CANARY_MODEL_ID,
    REPRESENTATIVE_CANARY_MODEL_REVISION,
    REPRESENTATIVE_CANARY_WORKLOAD_MANIFEST_RECORD_TYPE,
    REPRESENTATIVE_TASK_RUNTIME_ID_REFERENCE,
    REPRESENTATIVE_SGLANG_PACKAGE_PINS,
    REPRESENTATIVE_VLLM_PACKAGE_PINS,
    SGLANG_PAIRED_SMOKE_ARM,
    VANILLA_CANARY_ARM,
    RepresentativeCanaryWorkloadManifest,
    aggregate_isolated_canary_results,
    benchmark_manifest_provenance_runner_args,
    build_handoff_topology_attestation_record,
    create_representative_canary_cluster_hour_ledger,
    prepare_representative_canary_inputs,
    representative_canary_matrix,
    representative_canary_workload_manifest,
    representative_vllm_comparison_suite_id,
    representative_vllm_environment_provenance,
    reserve_and_submit_representative_canary_workload,
    require_pinned_revision,
    validated_benchmark_arm_specs,
    validated_benchmark_manifest_provenance,
    validate_handoff_topology_attestation,
    validate_representative_canary_reservation,
    validate_representative_canary_workload_payloads,
)
from document_kv_cache.databricks_resource_ledger import (
    databricks_submit_payload_reservation,
    read_databricks_cluster_hour_ledger_json,
)
from document_kv_cache.databricks_runs import DatabricksWorkspaceConfig
from document_kv_cache.engine import EngineReadyRequest
from document_kv_cache.engine_adapters import (
    build_engine_adapter_request,
    engine_adapter_request_to_record,
    vllm_adapter_spec,
)
from document_kv_cache.engine_protocol import KVCacheHandle, KVSegment
from document_kv_cache.methods import method_spec
from document_kv_cache.model_profiles import layout_for_model
from document_kv_cache.models import CacheGenerationMethod
from document_kv_cache.databricks_sglang_smoke_job import (
    SGLANG_SMOKE_RUNNER_SCRIPT,
    DatabricksSGLangSmokeJobConfig,
    build_databricks_sglang_smoke_run_submit_payload,
)
from document_kv_cache.databricks_vllm_smoke_job import (
    VLLM_SMOKE_RUNNER_SCRIPT,
    DatabricksVLLMSmokeJobConfig,
    build_databricks_vllm_smoke_run_submit_payload,
)
from document_kv_cache.vllm_smoke import (
    DOCUMENT_KV_PACKAGE_WHEEL_SHA256_ENV,
    build_benchmark_runner_args,
    parse_args as parse_vllm_smoke_args,
    parse_dataset_specs,
    vllm_representative_workload_profile,
)
from document_kv_cache.workflow import SourceDocument

REPRESENTATIVE_WHEEL_SHA256 = "f" * 64
REPRESENTATIVE_WHEEL_URI = (
    "dbfs:/cachet/wheels/"
    f"{REPRESENTATIVE_WHEEL_SHA256}/cachet_kv-0.2.0-py3-none-any.whl"
)


def _representative_submit_payloads():
    matrix = representative_canary_matrix()
    payloads = []
    for workload in representative_canary_workload_manifest().workloads:
        if workload.serving_platform == "vllm":
            profile = vllm_representative_workload_profile(workload.profile_id)
            config = DatabricksVLLMSmokeJobConfig(
                benchmark_id=workload.workload_id,
                output_dir=f"/Volumes/catalog/schema/volume/{workload.workload_id}",
                runner_python_file=(
                    f"dbfs:/cachet/runners/{workload.runner_sha256}/"
                    f"{workload.runner_basename}"
                ),
                hardware_target=workload.hardware_target,
                node_type_id=workload.node_type_id,
                single_user_name="user@example.com",
                wheel_uri=REPRESENTATIVE_WHEEL_URI,
                wheel_sha256=REPRESENTATIVE_WHEEL_SHA256,
                model_id=REPRESENTATIVE_CANARY_MODEL_ID,
                model_revision=REPRESENTATIVE_CANARY_MODEL_REVISION,
                tokenizer_revision=REPRESENTATIVE_CANARY_MODEL_REVISION,
                model_dtype="bfloat16",
                kv_cache_dtype="bfloat16",
                max_tokens=profile.max_output_tokens,
                max_model_len=profile.max_model_len,
                max_num_seqs=profile.max_num_seqs,
                gpu_memory_utilization=profile.gpu_memory_utilization,
                benchmark_repeats=profile.benchmark_repeats,
                request_parallelism=profile.request_parallelism,
                benchmark_arm_specs=(matrix.run_for_arm(workload.arm_id).arm_spec,),
                benchmark_evidence_policy="canary",
                representative_canary=True,
                representative_workload_profile=profile.profile_id,
                benchmark_manifest_provenance={
                    "input_tokens_target": profile.input_tokens_target,
                },
                benchmark_force_max_tokens=True,
                dataset_specs=(
                    "hotpotqa=/Volumes/catalog/schema/volume/hotpotqa.jsonl",
                ),
                allow_dataset_subset=True,
                **(
                    {}
                    if workload.arm_id == BASELINE_PREFILL_ARM
                    else {
                        "benchmark_handoff_generator_factory": (
                            REPRESENTATIVE_POST_ROPE_HANDOFF_GENERATOR_FACTORY
                            if workload.arm_id == FULL_PREFIX_CANARY_ARM
                            else REPRESENTATIVE_PRE_ROPE_HANDOFF_GENERATOR_FACTORY
                        ),
                        "benchmark_handoff_output_dir": (
                            f"/local_disk0/{workload.workload_id}/handoffs"
                        ),
                        "benchmark_handoff_cache_method": (
                            "full_prefix_prefill"
                            if workload.arm_id == FULL_PREFIX_CANARY_ARM
                            else "vanilla_prefill"
                        ),
                        "benchmark_handoff_segment_per_document": (
                            workload.arm_id == VANILLA_CANARY_ARM
                        ),
                    }
                ),
            )
            payload = build_databricks_vllm_smoke_run_submit_payload(config)
        else:
            config = DatabricksSGLangSmokeJobConfig(
                benchmark_id=workload.workload_id,
                output_dir=f"/Volumes/catalog/schema/volume/{workload.workload_id}",
                runner_python_file=(
                    f"dbfs:/cachet/runners/{workload.runner_sha256}/"
                    f"{workload.runner_basename}"
                ),
                hardware_target=workload.hardware_target,
                node_type_id=workload.node_type_id,
                single_user_name="user@example.com",
                wheel_uri=REPRESENTATIVE_WHEEL_URI,
                wheel_sha256=REPRESENTATIVE_WHEEL_SHA256,
                model_revision=REPRESENTATIVE_CANARY_MODEL_REVISION,
                tokenizer_revision=REPRESENTATIVE_CANARY_MODEL_REVISION,
                representative_canary=True,
                representative_workload_profile=workload.profile_id,
                context_length=4096,
                max_tokens=32,
                live_benchmark_repeats=2,
                sglang_attention_backend="triton",
                sglang_sampling_backend="pytorch",
                sglang_enable_deterministic_inference=True,
                generate_live_handoff=True,
                live_handoff_output_dir=(
                    f"/local_disk0/{workload.workload_id}/handoffs"
                ),
            )
            payload = build_databricks_sglang_smoke_run_submit_payload(config)
        payloads.append((workload.workload_id, payload))
    return tuple(payloads)


class _FakeDatabricksResponse:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class _FakeDatabricksOpener:
    def __init__(self):
        self.requests = []

    def __call__(self, request, *, timeout):
        self.requests.append((request, timeout))
        return _FakeDatabricksResponse({"run_id": 123})


def test_representative_workload_manifest_is_exact_ordered_and_versioned():
    manifest = representative_canary_workload_manifest()
    record = manifest.to_record()

    assert record["record_type"] == REPRESENTATIVE_CANARY_WORKLOAD_MANIFEST_RECORD_TYPE
    assert record["schema_version"] == 1
    assert record["job_count"] == 10
    assert record["first_wave_worst_case_cluster_hours"] == 40.0
    assert [workload.order for workload in manifest.workloads] == list(range(1, 11))
    assert [workload.requirement for workload in manifest.workloads] == [
        *("required",) * 7,
        *("best_effort",) * 3,
    ]
    assert [workload.workload_id for workload in manifest.workloads] == [
        "g6-vllm-8k-64-baseline",
        "g6-vllm-8k-64-full-prefix",
        "g6-vllm-8k-64-vanilla",
        "g6-vllm-16k-256-baseline",
        "g6-vllm-16k-256-full-prefix",
        "g6-vllm-16k-256-vanilla",
        "g6-sglang-4k-32-paired-smoke",
        "g5-vllm-8k-64-baseline",
        "g5-vllm-8k-64-full-prefix",
        "g5-vllm-8k-64-vanilla",
    ]
    assert manifest.workloads[6].arm_id == SGLANG_PAIRED_SMOKE_ARM
    assert manifest.workloads[0].package_pins == REPRESENTATIVE_VLLM_PACKAGE_PINS
    assert manifest.workloads[6].package_pins == REPRESENTATIVE_SGLANG_PACKAGE_PINS
    assert {row["model_id"] for row in record["workloads"]} == {
        REPRESENTATIVE_CANARY_MODEL_ID
    }
    assert {row["model_revision"] for row in record["workloads"]} == {
        REPRESENTATIVE_CANARY_MODEL_REVISION
    }
    assert [row["comparison_suite_id"] for row in record["workloads"][:3]] == [
        "g6-vllm-8k-64",
        "g6-vllm-8k-64",
        "g6-vllm-8k-64",
    ]
    assert [row["comparison_suite_id"] for row in record["workloads"][3:6]] == [
        "g6-vllm-16k-256",
        "g6-vllm-16k-256",
        "g6-vllm-16k-256",
    ]
    assert [row["comparison_suite_id"] for row in record["workloads"][7:]] == [
        "g5-vllm-8k-64",
        "g5-vllm-8k-64",
        "g5-vllm-8k-64",
    ]

    with pytest.raises(ValueError, match="exact ordered ten-job sequence"):
        RepresentativeCanaryWorkloadManifest(workloads=manifest.workloads[:-1])
    with pytest.raises(ValueError, match="exact ordered ten-job sequence"):
        RepresentativeCanaryWorkloadManifest(
            workloads=tuple(reversed(manifest.workloads))
        )


def test_representative_vllm_environment_provenance_binds_exact_node_geometry():
    assert dict(representative_vllm_environment_provenance("aws-g6-l4")) == {
        "hardware_fingerprint": (
            "aws:g6.8xlarge:gpu=l4x1:cpu=32:ram_mib=131072:"
            "local_disks=2x450gb"
        ),
        "runtime_version": "15.4.x-gpu-ml-scala2.12",
        "storage_identity": "local_nvme:/local_disk0:2x450gb",
        "cache_state": "cold",
    }
    assert dict(representative_vllm_environment_provenance("aws-g5-a10g")) == {
        "hardware_fingerprint": (
            "aws:g5.8xlarge:gpu=a10gx1:cpu=32:ram_mib=131072:"
            "local_disks=1x900gb"
        ),
        "runtime_version": "15.4.x-gpu-ml-scala2.12",
        "storage_identity": "local_nvme:/local_disk0:1x900gb",
        "cache_state": "cold",
    }


def test_real_representative_job_payloads_match_manifest_and_reserve_40_hours():
    payloads = _representative_submit_payloads()

    reservations = validate_representative_canary_workload_payloads(payloads)

    assert len(reservations) == 10
    assert sum(item.reserved_cluster_hours for item in reservations) == (
        REPRESENTATIVE_CANARY_FIRST_WAVE_CLUSTER_HOURS
    )
    assert {item.task_timeout_seconds for item in reservations} == {(14_400,)}
    assert [item.workload_id for item in reservations] == [
        workload.workload_id
        for workload in representative_canary_workload_manifest().workloads
    ]


def test_representative_vllm_payload_binds_provenance_to_wheel_digest():
    payloads = _representative_submit_payloads()
    workload_id, payload = payloads[0]
    package_revisions = {
        package: version
        for package, version in (
            pin.split("==", 1) for pin in REPRESENTATIVE_VLLM_PACKAGE_PINS
        )
    }
    package_revisions["cachet-kv"] = f"wheel-sha256:{'e' * 64}"
    _replace_provenance_value(payload, "package_revisions", package_revisions)

    with pytest.raises(ValueError, match="package_revisions"):
        validate_representative_canary_workload_payloads(
            [(workload_id, payload), *payloads[1:]]
        )


def _synthetic_records_from_first_representative_trio(
    *,
    runtime_ids: tuple[str, str, str],
    suite_ids: tuple[str, str, str] | None = None,
):
    payloads = [
        deepcopy(payload)
        for _workload_id, payload in _representative_submit_payloads()[:3]
    ]
    records = {}
    for index, (payload, runtime_id) in enumerate(
        zip(payloads, runtime_ids, strict=True)
    ):
        _replace_parameter_value(payload, "--runtime-id", runtime_id)
        if suite_ids is not None:
            _replace_parameter_value(
                payload,
                "--benchmark-suite-id",
                suite_ids[index],
            )
        parameters = payload["tasks"][0]["spark_python_task"]["parameters"]
        with patch.dict(
            os.environ,
            {
                DOCUMENT_KV_PACKAGE_WHEEL_SHA256_ENV: REPRESENTATIVE_WHEEL_SHA256,
            },
        ):
            smoke_config = parse_vllm_smoke_args(parameters[:-4])
        runner_args = build_benchmark_runner_args(
            smoke_config,
            parse_dataset_specs(
                smoke_config.dataset_specs,
                allow_subset=smoke_config.allow_dataset_subset,
            ),
        )
        suite_id = runner_args[runner_args.index("--suite-id") + 1]
        resolved_runtime_id = runner_args[runner_args.index("--runtime-id") + 1]
        package_revisions = tuple(
            (
                runner_args[argument_index + 1].partition("=")[0],
                runner_args[argument_index + 1].partition("=")[2],
            )
            for argument_index, argument in enumerate(runner_args)
            if argument == "--package-revision"
        )
        arm_id = str(smoke_config.benchmark_arm_specs[0]["arm_id"])
        records[arm_id] = json.loads(
            json.dumps(
                _result_record(
                    arm_id,
                    suite_id=suite_id,
                    runtime_id=resolved_runtime_id,
                    package_revisions=package_revisions,
                )
            )
        )
    return records


def test_representative_payload_identity_round_trips_into_isolated_aggregate():
    payloads = _representative_submit_payloads()[:3]
    expected_suite_id = representative_vllm_comparison_suite_id(
        hardware_target="aws-g6-l4",
        profile_id="vllm-8k-64-v1",
    )
    for workload_id, payload in payloads:
        parameters = payload["tasks"][0]["spark_python_task"]["parameters"]
        assert _single_parameter(parameters, "--benchmark-id") == workload_id
        assert (
            _single_parameter(parameters, "--benchmark-suite-id")
            == expected_suite_id
        )
        assert (
            _single_parameter(parameters, "--runtime-id")
            == REPRESENTATIVE_TASK_RUNTIME_ID_REFERENCE
        )

    records = _synthetic_records_from_first_representative_trio(
        runtime_ids=("task-run-101", "task-run-102", "task-run-103"),
    )
    source_runtime_by_arm = {
        arm_id: record["experiment_manifest"]["model_runtime"]
        for arm_id, record in records.items()
    }
    assert source_runtime_by_arm[BASELINE_PREFILL_ARM][
        "key_position_encoding"
    ] == "stored_post_rope"
    assert source_runtime_by_arm[FULL_PREFIX_CANARY_ARM][
        "key_position_encoding"
    ] == "stored_post_rope"
    assert source_runtime_by_arm[VANILLA_CANARY_ARM][
        "key_position_encoding"
    ] == "pre_rope"
    assert source_runtime_by_arm[VANILLA_CANARY_ARM]["rope_theta"] == 5_000_000.0
    assert source_runtime_by_arm[VANILLA_CANARY_ARM]["rope_rotary_dim"] == 128
    aggregate = aggregate_isolated_canary_results(records)

    assert aggregate["suite"]["suite_id"] == expected_suite_id
    assert aggregate["experiment_manifest"]["experiment_id"] == expected_suite_id
    assert aggregate["experiment_manifest"]["environment"][
        "runtime_id"
    ].startswith("separate_jobs:")
    assert aggregate["evidence_sanitized"] is True
    assert aggregate["experiment_manifest"]["model_runtime"]["package_revisions"][
        "cachet-kv"
    ] == f"wheel-sha256:{REPRESENTATIVE_WHEEL_SHA256}"
    aggregate_runtime_by_arm = {
        arm["arm_id"]: arm["runtime_environment"]
        for arm in aggregate["experiment_manifest"]["arms"]
    }
    assert aggregate_runtime_by_arm[BASELINE_PREFILL_ARM][
        "key_position_encoding"
    ] == "stored_post_rope"
    assert aggregate_runtime_by_arm[FULL_PREFIX_CANARY_ARM][
        "key_position_encoding"
    ] == "stored_post_rope"
    assert aggregate_runtime_by_arm[VANILLA_CANARY_ARM][
        "key_position_encoding"
    ] == "pre_rope"
    assert aggregate_runtime_by_arm[VANILLA_CANARY_ARM]["rope_theta"] == 5_000_000.0
    assert aggregate_runtime_by_arm[VANILLA_CANARY_ARM]["rope_rotary_dim"] == 128
    assert (
        aggregate["experiment_manifest"]["model_runtime"]["key_position_encoding"]
        == "varies_by_arm"
    )


def test_representative_payload_identity_rejects_duplicate_runtime_ids():
    records = _synthetic_records_from_first_representative_trio(
        runtime_ids=("task-run-101", "task-run-101", "task-run-103"),
    )

    with pytest.raises(ValueError, match="distinct execution-instance runtime_id"):
        aggregate_isolated_canary_results(records)


def test_representative_payload_identity_rejects_mismatched_suites():
    with pytest.raises(ValueError, match="comparison group"):
        _synthetic_records_from_first_representative_trio(
            runtime_ids=("task-run-101", "task-run-102", "task-run-103"),
            suite_ids=(
                "g6-vllm-8k-64",
                "forged-suite",
                "g6-vllm-8k-64",
            ),
        )


def test_representative_payload_batch_rejects_missing_or_reordered_workloads():
    payloads = _representative_submit_payloads()

    with pytest.raises(ValueError, match="exact ordered ten-workload manifest"):
        validate_representative_canary_workload_payloads(payloads[:-1])
    with pytest.raises(ValueError, match="exact ordered ten-workload manifest"):
        validate_representative_canary_workload_payloads(tuple(reversed(payloads)))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["tasks"][0]["new_cluster"].__setitem__(
                "node_type_id", "g6.12xlarge"
            ),
            "node_type_id",
        ),
        (
            lambda payload: payload["tasks"][0].__setitem__(
                "timeout_seconds", 3600
            ),
            "14400-second task",
        ),
        (
            lambda payload: _replace_parameter_value(
                payload, "--model-revision", "b" * 40
            ),
            "approved revision",
        ),
        (
            lambda payload: _replace_parameter_value(
                payload, "--benchmark-suite-id", "forged-suite"
            ),
            "benchmark-suite-id",
        ),
        (
            lambda payload: _replace_parameter_value(
                payload, "--runtime-id", "static-reused-runtime"
            ),
            "retry-unique Databricks task run reference",
        ),
        (
            lambda payload: _replace_parameter_value(
                payload, "--max-model-len", "16384"
            ),
            "max-model-len",
        ),
        (
            lambda payload: _replace_parameter_value(
                payload, "--model-id", "Other/Model"
            ),
            "model-id",
        ),
        (
            lambda payload: _replace_parameter_value(
                payload, "--model-dtype", "float16"
            ),
            "model-dtype",
        ),
        (
            lambda payload: _replace_parameter_value(
                payload, "--kv-cache-dtype", "fp8"
            ),
            "kv-cache-dtype",
        ),
        (
            lambda payload: _replace_parameter_value(
                payload, "--benchmark-prefix-cache-salt-mode", "static"
            ),
            "prefix-cache-salt-mode",
        ),
        (
            lambda payload: _replace_provenance_value(
                payload, "engine_version", "999.0"
            ),
            "engine_version",
        ),
        (
            lambda payload: _replace_provenance_value(
                payload,
                "package_revisions",
                {"vllm": "999.0"},
            ),
            "package_revisions",
        ),
        (
            lambda payload: _replace_provenance_value(
                payload,
                "hardware_fingerprint",
                "forged-hardware",
            ),
            "hardware_fingerprint",
        ),
        (
            lambda payload: _replace_provenance_value(
                payload,
                "runtime_version",
                "forged-runtime",
            ),
            "runtime_version",
        ),
        (
            lambda payload: _replace_provenance_value(
                payload,
                "storage_identity",
                "forged-storage",
            ),
            "storage_identity",
        ),
        (
            lambda payload: _replace_provenance_value(
                payload,
                "cache_state",
                "warm",
            ),
            "cache_state",
        ),
        (
            lambda payload: _replace_parameter_value(
                payload,
                "--dataset",
                "musique=/Volumes/catalog/schema/volume/musique.jsonl",
            ),
            "exactly one prepared HotpotQA",
        ),
    ],
)
def test_representative_payload_validator_rejects_fixed_contract_drift(
    mutation,
    message,
):
    workload_id, original = _representative_submit_payloads()[0]
    payload = deepcopy(original)
    mutation(payload)
    reservation = databricks_submit_payload_reservation(
        payload,
        attempt_id="drifted-attempt",
        workload_id=workload_id,
    )

    with pytest.raises(ValueError, match=message):
        validate_representative_canary_reservation(reservation, payload)


@pytest.mark.parametrize(
    ("payload_index", "mutation", "message"),
    [
        (
            0,
            lambda payload: payload["tasks"][0]["spark_python_task"].__setitem__(
                "python_file", "dbfs:/evil/untrusted.py"
            ),
            "runner python_file final component",
        ),
        (
            6,
            lambda payload: payload["tasks"][0]["spark_python_task"].__setitem__(
                "python_file", "/tmp/run_sglang_smoke.py"
            ),
            "persistent",
        ),
        (
            0,
            lambda payload: payload["tasks"][0]["spark_python_task"].__setitem__(
                "python_file", "dbfs:/unexpected/run_vllm_smoke.py"
            ),
            "approved SHA-256",
        ),
        (
            6,
            lambda payload: payload["tasks"][0]["spark_python_task"].__setitem__(
                "python_file", "dbfs:/unexpected/run_sglang_smoke.py"
            ),
            "approved SHA-256",
        ),
        (
            0,
            lambda payload: payload["tasks"][0]["new_cluster"].__setitem__(
                "spark_version", "99.0.x-unknown"
            ),
            "Spark runtime",
        ),
        (
            6,
            lambda payload: payload["tasks"][0]["new_cluster"].__setitem__(
                "data_security_mode", "NONE"
            ),
            "SINGLE_USER",
        ),
        (
            0,
            lambda payload: payload["tasks"][0]["new_cluster"].__setitem__(
                "autoscale", {"min_workers": 0, "max_workers": 1}
            ),
            "unsupported autoscale",
        ),
        (
            6,
            lambda payload: payload["tasks"][0]["new_cluster"].__setitem__(
                "aws_attributes",
                {
                    "availability": "ON_DEMAND",
                    "zone_id": "auto",
                    "ebs_volume_count": 1,
                },
            ),
            "fixed on-demand AWS attributes",
        ),
        (
            0,
            lambda payload: payload["tasks"][0]["new_cluster"][
                "spark_env_vars"
            ].__setitem__("CACHET_TRANSFORMERS_DEVICE", "cpu"),
            "spark_env_vars",
        ),
        (
            0,
            lambda payload: _replace_parameter_value(
                payload,
                "--output-dir",
                "/local_disk0/g6-vllm-8k-64-baseline",
            ),
            "persistent",
        ),
        (
            6,
            lambda payload: _replace_parameter_value(
                payload,
                "--output-dir",
                "/Volumes/catalog/schema/volume/reused-output",
            ),
            "final component",
        ),
        (
            0,
            lambda payload: _replace_parameter_value(
                payload,
                "--package-wheel-uri",
                "dbfs:/cachet/wheels/not-the-digest/cachet_kv.whl",
            ),
            "contain wheel_sha256",
        ),
        (
            6,
            lambda payload: _replace_parameter_value(
                payload,
                "--package-wheel-sha256",
                "e" * 64,
            ),
            "contain wheel_sha256",
        ),
    ],
)
def test_representative_payload_rejects_runtime_storage_and_wheel_drift(
    payload_index,
    mutation,
    message,
):
    workload_id, original = _representative_submit_payloads()[payload_index]
    payload = deepcopy(original)
    mutation(payload)
    reservation = databricks_submit_payload_reservation(
        payload,
        attempt_id=f"runtime-drift-{payload_index}",
        workload_id=workload_id,
    )

    with pytest.raises(ValueError, match=message):
        validate_representative_canary_reservation(reservation, payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.__setitem__(
                "git_source", {"git_url": "https://example.invalid/repo.git"}
            ),
            "representative submit payload.*unsupported git_source",
        ),
        (
            lambda payload: payload["tasks"][0].__setitem__(
                "libraries", [{"pypi": {"package": "unexpected-package==1.0"}}]
            ),
            "representative task.*unsupported libraries",
        ),
        (
            lambda payload: payload["tasks"][0]["spark_python_task"].__setitem__(
                "source", "GIT"
            ),
            "spark_python_task.*unsupported source",
        ),
        (
            lambda payload: payload["tasks"][0]["new_cluster"].__setitem__(
                "init_scripts",
                [{"dbfs": {"destination": "dbfs:/unexpected/init.sh"}}],
            ),
            "new_cluster.*unsupported init_scripts",
        ),
        (
            lambda payload: payload["tasks"][0]["new_cluster"].__setitem__(
                "docker_image", {"url": "example.invalid/alternate:latest"}
            ),
            "new_cluster.*unsupported docker_image",
        ),
    ],
)
def test_representative_payload_rejects_unknown_databricks_object_keys(
    mutation,
    message,
):
    workload_id, original = _representative_submit_payloads()[0]
    payload = deepcopy(original)
    mutation(payload)
    reservation = databricks_submit_payload_reservation(
        payload,
        attempt_id="outer-object-drift",
        workload_id=workload_id,
    )

    with pytest.raises(ValueError, match=message):
        validate_representative_canary_reservation(reservation, payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: _remove_parameter(
                payload, "--allow-dataset-subset", has_value=False
            ),
            "allow-dataset-subset",
        ),
        (
            lambda payload: _replace_parameter_value(
                payload,
                "--dataset",
                "hotpotqa=/tmp/hotpotqa.jsonl",
            ),
            "persistent",
        ),
        (
            lambda payload: _replace_parameter_value(
                payload,
                "--dataset",
                "hotpotqa=/Volumes/catalog/schema/volume/other.jsonl",
            ),
            "final component",
        ),
    ],
)
def test_representative_vllm_payload_rejects_dataset_contract_drift(
    mutation,
    message,
):
    workload_id, original = _representative_submit_payloads()[0]
    payload = deepcopy(original)
    mutation(payload)
    reservation = databricks_submit_payload_reservation(
        payload,
        attempt_id="dataset-drift",
        workload_id=workload_id,
    )

    with pytest.raises(ValueError, match=message):
        validate_representative_canary_reservation(reservation, payload)


def test_representative_sglang_payload_rejects_package_pin_drift():
    workload_id, original = _representative_submit_payloads()[6]
    payload = deepcopy(original)
    _replace_parameter_value(
        payload,
        "--representative-package-pin",
        "sglang==999.0",
    )
    reservation = databricks_submit_payload_reservation(
        payload,
        attempt_id="drifted-sglang-attempt",
        workload_id=workload_id,
    )

    with pytest.raises(ValueError, match="approved package pins"):
        validate_representative_canary_reservation(reservation, payload)


@pytest.mark.parametrize("payload_index", [0, 6])
def test_representative_payload_binds_runner_benchmark_id_to_workload(
    payload_index,
):
    workload_id, original = _representative_submit_payloads()[payload_index]
    payload = deepcopy(original)
    _replace_parameter_value(payload, "--benchmark-id", "forged-other-workload")
    reservation = databricks_submit_payload_reservation(
        payload,
        attempt_id=f"forged-benchmark-id-{payload_index}",
        workload_id=workload_id,
    )

    with pytest.raises(ValueError, match="benchmark-id must match"):
        validate_representative_canary_reservation(reservation, payload)


@pytest.mark.parametrize(
    ("payload_index", "mutation", "message"),
    [
        (
            0,
            lambda payload: payload["tasks"][0]["spark_python_task"][
                "parameters"
            ].extend(
                [
                    "--benchmark-handoff-generator-factory",
                    REPRESENTATIVE_HANDOFF_GENERATOR_FACTORY,
                ]
            ),
            "unsupported representative runner flags",
        ),
        (
            1,
            lambda payload: _remove_parameter(
                payload, "--benchmark-handoff-generator-factory"
            ),
            "requires exactly one --benchmark-handoff-generator-factory",
        ),
        (
            1,
            lambda payload: _replace_parameter_value(
                payload,
                "--benchmark-handoff-cache-method",
                "vanilla_prefill",
            ),
            "handoff method",
        ),
        (
            1,
            lambda payload: payload["tasks"][0]["spark_python_task"][
                "parameters"
            ].append("--benchmark-handoff-chunk-per-document"),
            "unsupported representative runner flags",
        ),
        (
            2,
            lambda payload: _remove_parameter(
                payload,
                "--benchmark-handoff-chunk-per-document",
                has_value=False,
            ),
            "handoff topology",
        ),
        (
            6,
            lambda payload: _remove_parameter(
                payload,
                "--generate-live-handoff",
                has_value=False,
            ),
            "requires exactly one --generate-live-handoff",
        ),
        (
            6,
            lambda payload: _remove_parameter(
                payload,
                "--live-handoff-output-dir",
            ),
            "requires exactly one --live-handoff-output-dir",
        ),
    ],
)
def test_representative_payload_rejects_arm_handoff_contract_drift(
    payload_index,
    mutation,
    message,
):
    workload_id, original = _representative_submit_payloads()[payload_index]
    payload = deepcopy(original)
    mutation(payload)
    reservation = databricks_submit_payload_reservation(
        payload,
        attempt_id=f"handoff-drift-{payload_index}",
        workload_id=workload_id,
    )

    with pytest.raises(ValueError, match=message):
        validate_representative_canary_reservation(reservation, payload)


@pytest.mark.parametrize(
    ("payload_index", "flag", "replacement"),
    [
        (1, "--benchmark-handoff-align-bytes", "8192"),
        (1, "--benchmark-handoff-generation-timeout-seconds", "900.0"),
        (6, "--live-handoff-align-bytes", "8192"),
        (6, "--live-handoff-generation-timeout-seconds", "900.0"),
        (6, "--sglang-hicache-page-size", "2"),
        (6, "--hicache-storage-prefetch-policy", "best_effort"),
        (6, "--hicache-storage-prefetch-threshold", "2"),
    ],
)
def test_representative_payload_rejects_exact_handoff_knob_drift(
    payload_index,
    flag,
    replacement,
):
    workload_id, original = _representative_submit_payloads()[payload_index]
    payload = deepcopy(original)
    _replace_parameter_value(payload, flag, replacement)
    reservation = databricks_submit_payload_reservation(
        payload,
        attempt_id=f"handoff-knob-drift-{payload_index}-{flag}",
        workload_id=workload_id,
    )

    with pytest.raises(ValueError, match=flag.removeprefix("--")):
        validate_representative_canary_reservation(reservation, payload)


@pytest.mark.parametrize(
    ("payload_index", "extra_parameters"),
    [
        (0, ("--package-install-spec", "/tmp/evil.whl")),
        (0, ("--attention-backend", "FLASH_ATTN")),
        (1, ("--benchmark-handoff-limit", "1")),
        (6, ("--live-check-extra-body-json", '{"temperature":1}')),
        (6, ("--hicache-ratio", "0.9")),
        (6, ("--no-stream",)),
    ],
)
def test_representative_payload_rejects_noncanonical_runner_flags(
    payload_index,
    extra_parameters,
):
    workload_id, original = _representative_submit_payloads()[payload_index]
    payload = deepcopy(original)
    payload["tasks"][0]["spark_python_task"]["parameters"].extend(
        extra_parameters
    )
    reservation = databricks_submit_payload_reservation(
        payload,
        attempt_id=f"extra-runner-flag-{payload_index}",
        workload_id=workload_id,
    )

    with pytest.raises(ValueError, match="unsupported representative runner flags"):
        validate_representative_canary_reservation(reservation, payload)


def test_representative_payload_rejects_duplicate_singleton_flag():
    workload_id, original = _representative_submit_payloads()[0]
    payload = deepcopy(original)
    payload["tasks"][0]["spark_python_task"]["parameters"].extend(
        ["--model-id", REPRESENTATIVE_CANARY_MODEL_ID]
    )
    reservation = databricks_submit_payload_reservation(
        payload,
        attempt_id="duplicate-model-id",
        workload_id=workload_id,
    )

    with pytest.raises(ValueError, match="flags must be unique: --model-id"):
        validate_representative_canary_reservation(reservation, payload)


@pytest.mark.parametrize(
    ("flag", "replacement"),
    [
        ("--cache-prompt-text-mode", "runtime"),
        ("--live-check-prompt-format", "plain"),
        ("--live-check-request-mode", "completion"),
        ("--live-check-temperature", "0.25"),
        ("--flush-cache-timeout-seconds", "5.0"),
    ],
)
def test_representative_sglang_payload_rejects_request_semantics_drift(
    flag,
    replacement,
):
    workload_id, original = _representative_submit_payloads()[6]
    payload = deepcopy(original)
    _replace_parameter_value(payload, flag, replacement)
    reservation = databricks_submit_payload_reservation(
        payload,
        attempt_id=f"sglang-request-drift-{flag}",
        workload_id=workload_id,
    )

    with pytest.raises(ValueError, match=flag.removeprefix("--")):
        validate_representative_canary_reservation(reservation, payload)


@pytest.mark.parametrize(
    "flag",
    ["--no-flush-cache-before-cache-arm", "--no-flush-cache-before-canary"],
)
def test_representative_sglang_payload_requires_cache_flushes(flag):
    workload_id, original = _representative_submit_payloads()[6]
    payload = deepcopy(original)
    payload["tasks"][0]["spark_python_task"]["parameters"].append(flag)
    reservation = databricks_submit_payload_reservation(
        payload,
        attempt_id=f"sglang-flush-drift-{flag}",
        workload_id=workload_id,
    )

    with pytest.raises(ValueError, match=flag.removeprefix("--")):
        validate_representative_canary_reservation(reservation, payload)


@pytest.mark.parametrize(
    "runner_script",
    [VLLM_SMOKE_RUNNER_SCRIPT, SGLANG_SMOKE_RUNNER_SCRIPT],
)
def test_databricks_bootstrap_verifies_wheel_bytes_before_install(
    runner_script,
    tmp_path,
    monkeypatch,
):
    namespace = {"__name__": "cachet_test_bootstrap"}
    exec(compile(runner_script, "<cachet-bootstrap>", "exec"), namespace)
    install_package_wheel = namespace["_install_package_wheel"]
    wheel_path = tmp_path / "cachet_kv-0.2.0-py3-none-any.whl"
    wheel_path.write_bytes(b"trusted-cachet-wheel")
    digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    install_calls = []
    monkeypatch.setitem(
        namespace["os"].environ,
        "DOCUMENT_KV_PACKAGE_INSTALL_SPEC",
        "cachet-bootstrap-test-sentinel",
    )
    monkeypatch.setitem(namespace["os"].environ, "PIP_CONFIG_FILE", "/attacker/pip.conf")
    monkeypatch.setitem(
        namespace["os"].environ,
        "PIP_INDEX_URL",
        "https://attacker.invalid/simple",
    )
    monkeypatch.setitem(namespace["os"].environ, "pip_no_index", "1")
    monkeypatch.setitem(
        namespace["os"].environ,
        "PIP_REQUIREMENT",
        "/attacker/requirements.txt",
    )
    monkeypatch.setitem(namespace["os"].environ, "VIRTUAL_ENV", "/attacker/venv")
    monkeypatch.setattr(
        namespace["subprocess"],
        "check_call",
        lambda argv, **kwargs: install_calls.append((argv, kwargs)),
    )

    remaining = install_package_wheel(
        [
            "--package-wheel-uri",
            str(wheel_path),
            "--package-wheel-sha256",
            digest,
            "--benchmark-id",
            "wheel-hash-test",
        ]
    )

    assert remaining == ["--benchmark-id", "wheel-hash-test"]
    assert len(install_calls) == 1
    _argv, kwargs = install_calls[0]
    if runner_script == VLLM_SMOKE_RUNNER_SCRIPT:
        assert _argv[-2:] == ["--no-deps", str(wheel_path)]
        environment = kwargs["env"]
        assert {
            key for key in environment if key.upper().startswith("PIP_")
        } == {
            "PIP_CONFIG_FILE",
            "PIP_DISABLE_PIP_VERSION_CHECK",
            "PIP_NO_INPUT",
        }
        assert environment["PIP_CONFIG_FILE"] == os.devnull
        assert environment["PYTHONNOUSERSITE"] == "1"
        assert all(
            variable not in environment
            for variable in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")
        )
    else:
        assert "--no-deps" not in _argv
        assert kwargs == {}
    wheel_path.write_bytes(b"tampered-cachet-wheel")
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        install_package_wheel(
            [
                "--package-wheel-uri",
                str(wheel_path),
                "--package-wheel-sha256",
                digest,
            ]
        )
    assert len(install_calls) == 1


def test_representative_submit_reserves_exact_payload_before_fake_post(tmp_path):
    workload_id, payload = _representative_submit_payloads()[0]
    ledger_path = tmp_path / "representative-ledger.json"
    create_representative_canary_cluster_hour_ledger(
        ledger_path,
        ledger_id="representative-canary-test",
    )
    opener = _FakeDatabricksOpener()
    config = DatabricksWorkspaceConfig(
        "https://dbc.example",
        "secret-token",
        timeout_seconds=9,
    )

    response = reserve_and_submit_representative_canary_workload(
        config,
        payload,
        ledger_path=ledger_path,
        attempt_id="g6-vllm-8k-64-baseline-attempt-1",
        workload_id=workload_id,
        opener=opener,
    )

    assert response == {"run_id": 123}
    assert len(opener.requests) == 1
    request, timeout = opener.requests[0]
    assert timeout == 9
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    reservation = ledger.reservations[0]
    assert reservation.workload_id == "g6-vllm-8k-64-baseline"
    assert reservation.submit_payload_sha256 == hashlib.sha256(request.data).hexdigest()
    assert json.loads(request.data.decode("utf-8")) == payload


def test_representative_submit_rejects_alias_before_reservation_or_post(tmp_path):
    _workload_id, payload = _representative_submit_payloads()[0]
    ledger_path = tmp_path / "representative-ledger.json"
    create_representative_canary_cluster_hour_ledger(
        ledger_path,
        ledger_id="representative-canary-test",
    )
    opener = _FakeDatabricksOpener()
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")

    with pytest.raises(ValueError, match="unknown representative workload_id"):
        reserve_and_submit_representative_canary_workload(
            config,
            payload,
            ledger_path=ledger_path,
            attempt_id="alias-attempt",
            workload_id="vllm-baseline",
            opener=opener,
        )

    assert opener.requests == []
    assert read_databricks_cluster_hour_ledger_json(ledger_path).reservations == ()


def _replace_parameter_value(payload, flag, replacement):
    parameters = payload["tasks"][0]["spark_python_task"]["parameters"]
    parameters[parameters.index(flag) + 1] = replacement


def _single_parameter(parameters, flag):
    assert parameters.count(flag) == 1
    return parameters[parameters.index(flag) + 1]


def _remove_parameter(payload, flag, *, has_value=True):
    parameters = payload["tasks"][0]["spark_python_task"]["parameters"]
    index = parameters.index(flag)
    del parameters[index : index + (2 if has_value else 1)]


def _replace_provenance_value(payload, field_name, replacement):
    parameters = payload["tasks"][0]["spark_python_task"]["parameters"]
    index = parameters.index("--benchmark-manifest-provenance-json") + 1
    provenance = json.loads(parameters[index])
    provenance[field_name] = replacement
    parameters[index] = json.dumps(provenance, separators=(",", ":"), sort_keys=True)


def _identity(method_id: str) -> ArtifactIdentity:
    return ArtifactIdentity(
        method_id=method_id,
        method_version="2" if method_id == "vanilla_prefill" else "1",
        method_config_digest="0" * 64,
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        model_revision="a" * 40,
        tokenizer_id="Qwen/Qwen3-4B-Instruct-2507",
        tokenizer_revision="a" * 40,
        lora_id="base",
        prompt_template_version="v1-benchmark",
        layout_version="qwen3-v1",
        kv_dtype="bfloat16",
        block_size=16,
        payload_axis_order="token_major",
        generator_version="test-generator-v1",
        key_position_encoding=(
            "pre_rope" if method_id == "vanilla_prefill" else "stored_post_rope"
        ),
        rope_theta=5_000_000.0 if method_id == "vanilla_prefill" else None,
        rope_rotary_dim=128 if method_id == "vanilla_prefill" else None,
    )


@pytest.mark.parametrize(
    ("method_id", "segment_count"),
    [
        ("full_prefix_prefill", 2),
        ("vanilla_prefill", 1),
    ],
)
def test_topology_attestation_enforces_method_owned_segment_counts(
    method_id,
    segment_count,
):
    row = {
        "example_key_sha256": "1" * 64,
        "method_id": method_id,
        "method_version": "2" if method_id == "vanilla_prefill" else "1",
        "method_config_digest": "2" * 64,
        "artifact_id": "3" * 64,
        "document_count": 2,
        "segment_count": segment_count,
        "logical_token_count": 32,
        "logical_prompt_sha256": "4" * 64,
    }

    with pytest.raises(ValueError, match="requires segment_count"):
        build_handoff_topology_attestation_record((row,))


def _handoff_manifest(tmp_path, *, method_id: str, segments: int, examples: int = 2):
    identity = _identity(method_id)
    entries = []
    for example_index in range(1, examples + 1):
        handoff_path = tmp_path / f"{method_id}.example-{example_index}.handoff.json"
        payload_path = tmp_path / f"{method_id}.example-{example_index}.kvpack"
        base_layout = layout_for_model(identity.model_id, dtype=identity.runtime_kv_dtype)
        layout = (
            replace(
                base_layout,
                pre_rope=True,
                rope_theta=5_000_000.0,
                rope_rotary_dim=128,
                key_position_encoding="pre_rope",
                shares_kv_storage=False,
                storage_layout="separate_key_value",
            )
            if method_id == CacheGenerationMethod.VANILLA_PREFILL.value
            else base_layout
        )
        segment_records = tuple(
            KVSegment(
                document_id=f"document-{example_index}-{index}",
                chunk_type="document_chunk",
                chunk_id=f"segment-{index}",
                token_start=index,
                token_count=1,
                byte_start=index * layout.bytes_per_token,
                byte_length=layout.bytes_per_token,
            )
            for index in range(segments)
        )
        payload = b"x" * (segments * layout.bytes_per_token)
        handle = KVCacheHandle(
            request_id=f"request-{method_id}-{example_index}",
            handle_uri=f"document-kv://request-{method_id}-{example_index}",
            layout=layout,
            segments=segment_records,
            total_tokens=segments,
            total_bytes=len(payload),
            metadata={
                "cachet.benchmark.dataset": "hotpotqa",
                "cachet.benchmark.example_id": f"example-{example_index}",
            },
            cache_method=method_id,
            artifact_identity=identity,
            payload_checksum=hashlib.sha256(payload).hexdigest(),
        )
        ready = EngineReadyRequest(
            handle=handle,
            payload=payload,
            estimated_gpu_bytes=len(payload),
            reuse_plan=method_spec(method_id).reuse_plan(),
        )
        handoff = engine_adapter_request_to_record(
            build_engine_adapter_request(ready, spec=vllm_adapter_spec()),
            payload_uri=str(payload_path),
        )
        handoff_path.write_text(
            json.dumps(handoff),
            encoding="utf-8",
        )
        entries.append(
            BenchmarkHandoffEntry(
                dataset="hotpotqa",
                example_id=f"example-{example_index}",
                request_id=f"request-{method_id}-{example_index}",
                handoff_json=str(handoff_path),
                payload_uri=str(payload_path),
                prompt_text_mode="runtime",
                cache_method=method_id,
                artifact_id=identity.artifact_id,
            )
        )
    manifest = BenchmarkHandoffManifest(
        entries=tuple(entries)
    )
    path = tmp_path / f"{method_id}.manifest.json"
    write_benchmark_handoff_manifest_json(manifest, path)
    return path, identity.artifact_id


def _input_jsonl(tmp_path, *, document_count: int = 2, examples: int = 2):
    path = tmp_path / "input.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "dataset": "hotpotqa",
                    "example_id": f"example-{example_index}",
                    "query": "Who wrote the notes?",
                    "expected_answer": "Ada Lovelace",
                    "documents": [
                        {
                            "document_id": f"document-{example_index}-{index}",
                            "text": f"document text {example_index} {index}",
                        }
                        for index in range(document_count)
                    ],
                }
            )
            + "\n"
            for example_index in range(1, examples + 1)
        ),
        encoding="utf-8",
    )
    return path


def test_representative_canary_matrix_is_registered_and_fixed():
    matrix = representative_canary_matrix()

    assert tuple(run.arm_id for run in matrix.runs) == REPRESENTATIVE_CANARY_ARM_IDS
    assert matrix.run_for_arm(BASELINE_PREFILL_ARM).method_id == ""
    assert matrix.run_for_arm(FULL_PREFIX_CANARY_ARM).method_id == "full_prefix_prefill"
    assert matrix.run_for_arm(FULL_PREFIX_CANARY_ARM).expected_segments == "one"
    assert matrix.run_for_arm(VANILLA_CANARY_ARM).method_id == "vanilla_prefill"
    assert matrix.run_for_arm(VANILLA_CANARY_ARM).expected_segments == "per_document"
    assert json.loads(matrix.run_for_arm(VANILLA_CANARY_ARM).runner_args()[1])[
        "cache_method"
    ] == "vanilla_prefill"


def test_arm_specs_reject_canonical_request_overrides():
    with pytest.raises(ValueError, match="reserved request fields"):
        validated_benchmark_arm_specs(
            (
                {
                    "arm_id": "unsafe",
                    "uses_cache": False,
                    "description": "unsafe arm",
                    "extra_body": {"max_tokens": 1},
                },
            )
        )

    records = validated_benchmark_arm_specs(
        (
            {
                "arm_id": "safe",
                "uses_cache": False,
                "description": "safe arm",
                "extra_body": {"cache_salt": "isolated", "ignore_eos": True},
            },
        )
    )
    assert records[0]["extra_body"] == {
        "cache_salt": "isolated",
        "ignore_eos": True,
    }
    with pytest.raises(TypeError):
        records[0]["arm_id"] = "mutated"
    with pytest.raises(TypeError):
        records[0]["extra_body"]["cache_salt"] = "mutated"


def test_manifest_provenance_has_stable_runner_arguments():
    args = benchmark_manifest_provenance_runner_args(
        {
            "model_revision": "a" * 40,
            "tokenizer_id": "Qwen/Qwen3-4B-Instruct-2507",
            "engine_id": "vllm",
            "engine_version": "0.10.2",
            "rope_theta": 1_000_000,
            "rope_rotary_dim": 128,
            "package_revisions": {"z-package": "2", "a-package": "1"},
            "input_tokens_target": 8192,
            "measurement_scopes": ["latency", "resource"],
        }
    )

    assert args[:4] == (
        "--model-revision",
        "a" * 40,
        "--tokenizer-id",
        "Qwen/Qwen3-4B-Instruct-2507",
    )
    assert args[args.index("--engine-id") + 1] == "vllm"
    assert args[args.index("--rope-theta") + 1] == "1000000.0"
    assert args[args.index("--rope-rotary-dim") + 1] == "128"
    first_package = args.index("--package-revision")
    assert args[first_package + 1] == "a-package=1"
    assert args[first_package + 3] == "z-package=2"
    assert args[-4:] == (
        "--measurement-scope",
        "latency",
        "--measurement-scope",
        "resource",
    )


def test_manifest_provenance_requires_paired_rope_geometry():
    with pytest.raises(ValueError, match="provided together"):
        validated_benchmark_manifest_provenance({"rope_theta": 1_000_000})
    with pytest.raises(ValueError, match="provided together"):
        validated_benchmark_manifest_provenance({"rope_rotary_dim": 128})


@pytest.mark.parametrize("revision", [None, "", "   ", " unresolved ", "unresolved"])
def test_representative_revision_must_be_exact_and_resolved(revision):
    with pytest.raises((TypeError, ValueError)):
        require_pinned_revision(revision, "model_revision")


def test_prepare_representative_canary_inputs_writes_combined_and_isolated_projections(tmp_path):
    input_jsonl = _input_jsonl(tmp_path)
    full_manifest, full_artifact = _handoff_manifest(
        tmp_path,
        method_id="full_prefix_prefill",
        segments=1,
    )
    vanilla_manifest, vanilla_artifact = _handoff_manifest(
        tmp_path,
        method_id="vanilla_prefill",
        segments=2,
    )

    prepared = prepare_representative_canary_inputs(
        input_jsonl,
        full_prefix_manifest_json=full_manifest,
        vanilla_manifest_json=vanilla_manifest,
        output_dir=tmp_path / "prepared",
        dataset="hotpotqa",
        input_tokens_target=8192,
        token_counter=lambda _prompt: 8192,
        tokenizer_id="Qwen/Qwen3-4B-Instruct-2507",
        tokenizer_revision="a" * 40,
        tokenizer_add_special_tokens=False,
    )

    combined_rows = [
        json.loads(line)
        for line in prepared.combined_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert len(combined_rows) == 2
    combined = combined_rows[0]
    assert set(combined["arm_kv_transfer_params"]) == {
        FULL_PREFIX_CANARY_ARM,
        VANILLA_CANARY_ARM,
    }
    assert (
        combined["arm_kv_transfer_params"][FULL_PREFIX_CANARY_ARM][
            "document_kv.artifact_id"
        ]
        == full_artifact
    )
    assert (
        combined["arm_kv_transfer_params"][VANILLA_CANARY_ARM][
            "document_kv.artifact_id"
        ]
        == vanilla_artifact
    )
    baseline = json.loads(
        prepared.arm_jsonl[BASELINE_PREFILL_ARM].read_text(encoding="utf-8").splitlines()[0]
    )
    full = json.loads(
        prepared.arm_jsonl[FULL_PREFIX_CANARY_ARM].read_text(encoding="utf-8").splitlines()[0]
    )
    vanilla = json.loads(
        prepared.arm_jsonl[VANILLA_CANARY_ARM].read_text(encoding="utf-8").splitlines()[0]
    )
    assert "arm_kv_transfer_params" not in baseline
    assert "arm_kv_transfer_params" not in full
    assert "arm_kv_transfer_params" not in vanilla
    assert full["kv_transfer_params"]["document_kv.cache_method"] == "full_prefix_prefill"
    assert vanilla["kv_transfer_params"]["document_kv.cache_method"] == "vanilla_prefill"
    assert prepared.logical_sample_digest
    preparation_record = json.loads(
        prepared.preparation_manifest_json.read_text(encoding="utf-8")
    )
    assert preparation_record["artifacts"][0]["full_prefix_segments"] == 1
    assert preparation_record["artifacts"][0]["vanilla_segments"] == 2
    assert preparation_record["input_tokens_target"] == 8192
    assert preparation_record["tokenizer"] == {
        "tokenizer_id": "Qwen/Qwen3-4B-Instruct-2507",
        "tokenizer_revision": "a" * 40,
        "add_special_tokens": False,
    }
    assert len(preparation_record["logical_token_counts"]) == 2
    assert {
        row["logical_tokens"] for row in preparation_record["logical_token_counts"]
    } == {8192}
    topology_by_arm = preparation_record["handoff_topology_attestations"]
    assert set(topology_by_arm) == {
        FULL_PREFIX_CANARY_ARM,
        VANILLA_CANARY_ARM,
    }
    full_topology = validate_handoff_topology_attestation(
        topology_by_arm[FULL_PREFIX_CANARY_ARM]
    )
    vanilla_topology = validate_handoff_topology_attestation(
        topology_by_arm[VANILLA_CANARY_ARM]
    )
    assert full_topology["record_type"] == HANDOFF_TOPOLOGY_ATTESTATION_RECORD_TYPE
    assert [row["segment_count"] for row in full_topology["examples"]] == [1, 1]
    assert [row["segment_count"] for row in vanilla_topology["examples"]] == [2, 2]
    assert {row["document_count"] for row in vanilla_topology["examples"]} == {2}
    assert {row["logical_token_count"] for row in vanilla_topology["examples"]} == {
        8192
    }
    sanitized_topology = json.dumps(topology_by_arm, sort_keys=True)
    assert "Who wrote the notes?" not in sanitized_topology
    assert "document text" not in sanitized_topology
    assert str(tmp_path) not in sanitized_topology


def test_prepare_representative_canary_inputs_rejects_stale_vanilla_v1_handoff(
    tmp_path,
):
    input_jsonl = _input_jsonl(tmp_path)
    full_manifest, _ = _handoff_manifest(
        tmp_path,
        method_id="full_prefix_prefill",
        segments=1,
    )
    vanilla_manifest, _ = _handoff_manifest(
        tmp_path,
        method_id="vanilla_prefill",
        segments=2,
    )
    manifest_record = json.loads(vanilla_manifest.read_text(encoding="utf-8"))
    for entry in manifest_record["entries"]:
        handoff_path = tmp_path / f"vanilla_prefill.{entry['example_id']}.handoff.json"
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        identity_record = handoff["handle"]["artifact_identity"]
        identity_record["method_version"] = "1"
        stale_identity = ArtifactIdentity.from_record(identity_record)
        handoff["metadata"]["document_kv.artifact_id"] = stale_identity.artifact_id
        handoff["metadata"]["document_kv.method_version"] = "1"
        handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
        entry["artifact_id"] = stale_identity.artifact_id
    vanilla_manifest.write_text(json.dumps(manifest_record), encoding="utf-8")

    with pytest.raises(ValueError, match="method_version"):
        prepare_representative_canary_inputs(
            input_jsonl,
            full_prefix_manifest_json=full_manifest,
            vanilla_manifest_json=vanilla_manifest,
            output_dir=tmp_path / "prepared",
            dataset="hotpotqa",
            input_tokens_target=8192,
            token_counter=lambda _prompt: 8192,
            tokenizer_id="Qwen/Qwen3-4B-Instruct-2507",
            tokenizer_revision="a" * 40,
            tokenizer_add_special_tokens=False,
        )

    assert not (tmp_path / "prepared").exists()


def test_prepare_representative_canary_inputs_requires_multiple_documents(tmp_path):
    input_jsonl = _input_jsonl(tmp_path, document_count=1)
    full_manifest, _ = _handoff_manifest(
        tmp_path,
        method_id="full_prefix_prefill",
        segments=1,
    )
    vanilla_manifest, _ = _handoff_manifest(
        tmp_path,
        method_id="vanilla_prefill",
        segments=1,
    )

    with pytest.raises(ValueError, match="at least two documents"):
        prepare_representative_canary_inputs(
            input_jsonl,
            full_prefix_manifest_json=full_manifest,
            vanilla_manifest_json=vanilla_manifest,
            output_dir=tmp_path / "prepared",
            dataset="hotpotqa",
            input_tokens_target=8192,
            token_counter=lambda _prompt: 8192,
            tokenizer_id="Qwen/Qwen3-4B-Instruct-2507",
            tokenizer_revision="a" * 40,
            tokenizer_add_special_tokens=False,
        )


def test_prepare_representative_canary_inputs_requires_two_distinct_examples(tmp_path):
    input_jsonl = _input_jsonl(tmp_path, examples=1)
    full_manifest, _ = _handoff_manifest(
        tmp_path,
        method_id="full_prefix_prefill",
        segments=1,
        examples=1,
    )
    vanilla_manifest, _ = _handoff_manifest(
        tmp_path,
        method_id="vanilla_prefill",
        segments=2,
        examples=1,
    )

    with pytest.raises(ValueError, match="at least two distinct examples"):
        prepare_representative_canary_inputs(
            input_jsonl,
            full_prefix_manifest_json=full_manifest,
            vanilla_manifest_json=vanilla_manifest,
            output_dir=tmp_path / "prepared",
            dataset="hotpotqa",
            input_tokens_target=8192,
            token_counter=lambda _prompt: 8192,
            tokenizer_id="Qwen/Qwen3-4B-Instruct-2507",
            tokenizer_revision="a" * 40,
            tokenizer_add_special_tokens=False,
        )


def test_prepare_representative_canary_inputs_rejects_token_count_drift(tmp_path):
    input_jsonl = _input_jsonl(tmp_path)
    full_manifest, _ = _handoff_manifest(
        tmp_path,
        method_id="full_prefix_prefill",
        segments=1,
    )
    vanilla_manifest, _ = _handoff_manifest(
        tmp_path,
        method_id="vanilla_prefill",
        segments=2,
    )

    with pytest.raises(ValueError, match="expected exactly 8192"):
        prepare_representative_canary_inputs(
            input_jsonl,
            full_prefix_manifest_json=full_manifest,
            vanilla_manifest_json=vanilla_manifest,
            output_dir=tmp_path / "prepared",
            dataset="hotpotqa",
            input_tokens_target=8192,
            token_counter=lambda _prompt: 8191,
            tokenizer_id="Qwen/Qwen3-4B-Instruct-2507",
            tokenizer_revision="a" * 40,
            tokenizer_add_special_tokens=False,
        )


class _CanaryEngine:
    def __init__(self, arm_id: str) -> None:
        self.arm_id = arm_id

    def generate(self, _request):
        ttft, completion = {
            BASELINE_PREFILL_ARM: (4.0, 6.0),
            FULL_PREFIX_CANARY_ARM: (2.0, 4.0),
            VANILLA_CANARY_ARM: (1.0, 3.0),
        }[self.arm_id]
        return BenchmarkGeneration(
            output_text="Ada Lovelace",
            prompt_tokens=8192,
            completion_tokens=64,
            ttft_seconds=ttft,
            time_to_completion_seconds=completion,
            metadata={"engine": "test"},
        )


def _result_record(
    arm_id: str,
    *,
    generation_seed: int = 11,
    runtime_id: str = "physical-canary-runtime",
    suite_id: str = "canary-8k-64",
    package_revisions: tuple[tuple[str, str], ...] = (("cachet-kv", "commit-1"),),
):
    method_id = {
        BASELINE_PREFILL_ARM: "",
        FULL_PREFIX_CANARY_ARM: "full_prefix_prefill",
        VANILLA_CANARY_ARM: "vanilla_prefill",
    }[arm_id]
    artifact_id = f"artifact-{method_id}" if method_id else ""
    arm = BenchmarkArm(
        arm_id=arm_id,
        uses_cache=bool(method_id),
        description=f"isolated {arm_id}",
        cache_method=method_id,
        connector_mode="cachet" if method_id else "",
        variant_id="default" if method_id else "",
        implementation_kind="cachet" if method_id else "baseline",
        method_version=(
            "2"
            if method_id == "vanilla_prefill"
            else "1"
            if method_id
            else ""
        ),
        method_config_digest=method_config_digest({}) if method_id else "",
        physical_transform_id=(f"cachet.{method_id}" if method_id else "identity"),
        requires_cachet_handoff=bool(method_id),
    )
    examples = []
    for index in (1, 2):
        transfer_params = (
            {
                DOCUMENT_KV_REQUEST_ID_PARAM: f"request-{method_id}-{index}",
                DOCUMENT_KV_HANDOFF_JSON_PARAM: f"/tmp/{method_id}-{index}.handoff.json",
                DOCUMENT_KV_CACHE_METHOD_PARAM: method_id,
                DOCUMENT_KV_ARTIFACT_ID_PARAM: artifact_id,
            }
            if method_id
            else {}
        )
        examples.append(
            BenchmarkExample(
                example_id=f"example-{index}",
                dataset="hotpotqa",
                documents=(
                    SourceDocument.from_text(
                        document_id=f"document-{index}-1",
                        text="Ada Lovelace wrote notes on the Analytical Engine.",
                    ),
                    SourceDocument.from_text(
                        document_id=f"document-{index}-2",
                        text="The notes described an algorithm for Bernoulli numbers.",
                    ),
                ),
                query="Who wrote the notes?",
                expected_answer="Ada Lovelace",
                kv_transfer_params=transfer_params,
            )
        )
    suite = BenchmarkSuite(
        suite_id=suite_id,
        examples=tuple(examples),
        model_id="qwen3:4b-instruct",
        hardware_target="aws-g6-l4",
        datasets=("hotpotqa",),
    )
    environment = representative_vllm_environment_provenance("aws-g6-l4")
    context = BenchmarkManifestContext(
        model_revision="a" * 40,
        tokenizer_id="Qwen/Qwen3-4B-Instruct-2507",
        tokenizer_revision="a" * 40,
        engine_id="vllm",
        engine_version="0.10.2",
        package_revisions=package_revisions,
        input_tokens_target=8192,
        max_output_tokens=64,
        temperature=0.0,
        stream=True,
        generation_seed=generation_seed,
        decode_settings={"ignore_eos": True},
        hardware_fingerprint=environment["hardware_fingerprint"],
        runtime_id=runtime_id,
        runtime_version=environment["runtime_version"],
        storage_identity=environment["storage_identity"],
        cache_state=environment["cache_state"],
        key_position_encoding=(
            "pre_rope" if method_id == "vanilla_prefill" else "stored_post_rope"
        ),
        rope_theta=5_000_000.0 if method_id == "vanilla_prefill" else None,
        rope_rotary_dim=128 if method_id == "vanilla_prefill" else None,
        comparison_mode="methods_same_setting",
    )
    result = run_benchmark_suite(
        suite,
        {arm_id: _CanaryEngine(arm_id)},
        arms=(arm,),
        repeats=1,
        request_parallelism=1,
        seed=7,
        isolate_arms=True,
        prefix_cache_salt_mode="per_request",
        manifest_context=context,
        evidence_policy="canary",
    )
    record = benchmark_run_result_to_record(result)
    assert [item["arm_id"] for item in record["suite"]["arms"]] == [arm_id]
    assert [item["arm_id"] for item in record["experiment_manifest"]["arms"]] == [
        arm_id
    ]
    return record


def test_aggregate_isolated_canary_results_validates_and_rebuilds_paired_statistics(tmp_path):
    results = {
        arm_id: _result_record(arm_id, runtime_id=f"cluster-{arm_id}")
        for arm_id in reversed(REPRESENTATIVE_CANARY_ARM_IDS)
    }
    output_path = tmp_path / "aggregate.json"
    assert {
        record["experiment_manifest"]["comparison"]["reference_arm_id"]
        for record in results.values()
    } == set(REPRESENTATIVE_CANARY_ARM_IDS)

    aggregate = aggregate_isolated_canary_results(results, output_json=output_path)

    assert aggregate["record_type"] == ISOLATED_CANARY_AGGREGATE_RECORD_TYPE
    assert [arm["arm_id"] for arm in aggregate["experiment_manifest"]["arms"]] == list(
        REPRESENTATIVE_CANARY_ARM_IDS
    )
    assert aggregate["experiment_manifest"]["comparison"]["reference_arm_id"] == (
        BASELINE_PREFILL_ARM
    )
    assert aggregate["experiment_manifest"]["environment"]["runtime_id"].startswith(
        "separate_jobs:"
    )
    rows = aggregate["paired_statistics"]["rows"]
    assert {row["cache_arm_id"] for row in rows} == {
        FULL_PREFIX_CANARY_ARM,
        VANILLA_CANARY_ARM,
    }
    assert all(row["paired_examples"] == 2 for row in rows)
    assert {row["cache_arm_id"] for row in aggregate["comparisons"]} == {
        FULL_PREFIX_CANARY_ARM,
        VANILLA_CANARY_ARM,
    }
    assert [
        window["arm_id"] for window in aggregate["execution_windows"]
    ] == list(REPRESENTATIVE_CANARY_ARM_IDS)
    baseline_window = aggregate["execution_windows"][0]
    assert baseline_window["completion_tokens"] == 128
    assert aggregate["report_rows"][0]["aggregate_output_tokens_per_second"] == (
        pytest.approx(baseline_window["aggregate_output_tokens_per_second"])
    )
    assert aggregate["evidence_gate"]["policy"] == "canary"
    assert benchmark_record_aggregate_issues(aggregate) == ()
    assert aggregate_isolated_canary_results(results) == aggregate
    assert output_path.read_text(encoding="utf-8").endswith("\n")


def test_aggregate_isolated_canary_results_rejects_decode_drift():
    results = {
        arm_id: _result_record(arm_id)
        for arm_id in REPRESENTATIVE_CANARY_ARM_IDS
    }
    results[VANILLA_CANARY_ARM] = _result_record(
        VANILLA_CANARY_ARM,
        generation_seed=12,
    )

    with pytest.raises(ValueError, match="manifest decoding differs"):
        aggregate_isolated_canary_results(results)
