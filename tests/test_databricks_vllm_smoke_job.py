from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import document_kv_cache.databricks_vllm_smoke_job as public_databricks_vllm_smoke_job
from document_kv_cache.artifact_identity import RuntimeIdentity
from document_kv_cache.canary_orchestration import (
    REPRESENTATIVE_TASK_RUNTIME_ID_REFERENCE,
    REPRESENTATIVE_VLLM_RUNNER_SHA256,
    representative_canary_matrix,
    representative_vllm_environment_provenance,
)
from document_kv_cache.databricks_vllm_smoke_job import (
    DEFAULT_DATABRICKS_VLLM_SMOKE_PURPOSE,
    DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME,
    DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY,
    VLLM_SMOKE_RUNNER_SCRIPT,
    DatabricksVLLMSmokeJobConfig,
    build_databricks_vllm_smoke_run_submit_payload,
    main,
    write_databricks_vllm_smoke_run_submit_json,
    write_databricks_vllm_smoke_runner_script,
)
from document_kv_cache.serving_env import VLLM_VERSION
from document_kv_cache.vllm_smoke import (
    DOCUMENT_KV_PACKAGE_WHEEL_SHA256_ENV,
    build_benchmark_runner_args,
    parse_args as parse_vllm_smoke_args,
    parse_dataset_specs,
    vllm_representative_workload_profile,
)


WHEEL_URI = "/Volumes/catalog/schema/volume/wheels/cachet_kv-0.2.0-py3-none-any.whl"
SINGLE_USER_NAME = "user@example.com"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_SPECS = tuple(
    f"{dataset}=/Volumes/catalog/schema/volume/v1/{dataset}.jsonl"
    for dataset in ("biography", "hotpotqa", "musique", "niah")
)
MODEL_REVISION = "a" * 40
REPRESENTATIVE_WHEEL_SHA256 = "f" * 64
REPRESENTATIVE_WHEEL_URI = (
    "dbfs:/cachet/wheels/"
    f"{REPRESENTATIVE_WHEEL_SHA256}/cachet_kv-0.2.0-py3-none-any.whl"
)


def test_payload_cache_option_is_appended_to_preserve_positional_job_config_api():
    field_names = [field.name for field in fields(DatabricksVLLMSmokeJobConfig)]

    assert field_names[-1] == "benchmark_prewarm_payload_cache"
    assert field_names.index("benchmark_cache_runtime_prompt") == (
        field_names.index("benchmark_prewarm_cache_prefix") + 1
    )


def representative_job_kwargs(
    *,
    profile_id="vllm-8k-64-v1",
    arm_index=0,
    arm_specs=None,
    provenance=None,
):
    profile = vllm_representative_workload_profile(profile_id)
    resolved_provenance = {"input_tokens_target": profile.input_tokens_target}
    if provenance is not None:
        resolved_provenance.update(provenance)
    if arm_specs is None:
        arm_specs = (representative_canary_matrix().runs[arm_index].arm_spec,)
    return {
        "wheel_uri": REPRESENTATIVE_WHEEL_URI,
        "wheel_sha256": REPRESENTATIVE_WHEEL_SHA256,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "representative_canary": True,
        "representative_workload_profile": profile.profile_id,
        "benchmark_arm_specs": arm_specs,
        "benchmark_evidence_policy": "canary",
        "benchmark_manifest_provenance": resolved_provenance,
        "max_tokens": profile.max_output_tokens,
        "max_model_len": profile.max_model_len,
        "max_num_seqs": profile.max_num_seqs,
        "gpu_memory_utilization": profile.gpu_memory_utilization,
        "benchmark_repeats": profile.benchmark_repeats,
        "request_parallelism": profile.request_parallelism,
        "benchmark_force_max_tokens": True,
        "dataset_specs": DATASET_SPECS,
    }


def test_databricks_vllm_representative_requires_content_addressed_wheel():
    kwargs = representative_job_kwargs()
    kwargs.pop("wheel_sha256")

    with pytest.raises(ValueError, match="wheel_sha256"):
        DatabricksVLLMSmokeJobConfig(
            benchmark_id="representative-vllm-missing-wheel-digest",
            output_dir=(
                "/Volumes/catalog/schema/volume/"
                "representative-vllm-missing-wheel-digest"
            ),
            runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
            node_type_id="g6.8xlarge",
            single_user_name=SINGLE_USER_NAME,
            **kwargs,
        )


def stored_post_rope_runtime_identity() -> RuntimeIdentity:
    return RuntimeIdentity(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        model_revision=MODEL_REVISION,
        tokenizer_id="Qwen/Qwen3-4B-Instruct-2507",
        tokenizer_revision=MODEL_REVISION,
        lora_id="base",
        prompt_template_version="v1",
        layout_version="qwen3-v1",
        kv_dtype="bfloat16",
        block_size=16,
        payload_axis_order="token_major",
        key_position_encoding="stored_post_rope",
    )


def test_build_databricks_vllm_smoke_payload_uses_single_node_g5_cluster():
    config = DatabricksVLLMSmokeJobConfig(
        benchmark_id="v1-vllm-smoke-001",
        output_dir="/Volumes/catalog/schema/volume/v1-vllm-smoke",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        node_type_id="g6.8xlarge",
        wheel_uri=WHEEL_URI,
        single_user_name=SINGLE_USER_NAME,
        max_tokens=48,
        timeout_seconds=300,
        import_probe_timeout_seconds=90,
        server_start_timeout_seconds=600,
        local_root="/local_disk0",
        server_host="0.0.0.0",
        server_port=8123,
        client_host="127.0.0.1",
        max_model_len=32768,
        max_num_seqs=8,
        gpu_memory_utilization=0.72,
        dataset_specs=DATASET_SPECS,
        custom_tags={"team": "document-kv"},
    )

    payload = build_databricks_vllm_smoke_run_submit_payload(config)
    task = payload["tasks"][0]
    cluster = task["new_cluster"]

    assert payload["run_name"] == DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME
    assert payload["timeout_seconds"] == 14400
    assert task["task_key"] == DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY
    assert task["timeout_seconds"] == 14400
    assert task["max_retries"] == 0
    assert "libraries" not in task
    assert cluster["node_type_id"] == "g6.8xlarge"
    assert cluster["driver_node_type_id"] == "g6.8xlarge"
    assert cluster["data_security_mode"] == "SINGLE_USER"
    assert cluster["single_user_name"] == SINGLE_USER_NAME
    assert cluster["num_workers"] == 0
    assert cluster["custom_tags"]["ResourceClass"] == "SingleNode"
    assert cluster["custom_tags"]["purpose"] == DEFAULT_DATABRICKS_VLLM_SMOKE_PURPOSE
    assert cluster["custom_tags"]["team"] == "document-kv"
    assert task["spark_python_task"] == {
        "python_file": "dbfs:/benchmarks/run_vllm_smoke.py",
        "parameters": [
            "--benchmark-id",
            "v1-vllm-smoke-001",
            "--output-dir",
            "/Volumes/catalog/schema/volume/v1-vllm-smoke",
            "--max-tokens",
            "48",
            "--timeout-seconds",
            "300",
            "--import-probe-timeout-seconds",
            "90",
            "--server-start-timeout-seconds",
            "600",
            "--local-root",
            "/local_disk0",
            "--server-host",
            "0.0.0.0",
            "--server-port",
            "8123",
            "--client-host",
            "127.0.0.1",
            "--max-model-len",
            "32768",
            "--max-num-seqs",
            "8",
            "--gpu-memory-utilization",
            "0.72",
            "--hardware-target",
            "aws-g6-l4",
            "--benchmark-repeats",
            "1",
            "--request-parallelism",
            "1",
            "--runtime-telemetry-interval-seconds",
            "1.0",
            "--dataset",
            DATASET_SPECS[0],
            "--dataset",
            DATASET_SPECS[1],
            "--dataset",
            DATASET_SPECS[2],
            "--dataset",
            DATASET_SPECS[3],
            "--package-wheel-uri",
            WHEEL_URI,
        ],
    }


def test_build_databricks_vllm_smoke_payload_includes_payload_cache_budget():
    config = DatabricksVLLMSmokeJobConfig(
        benchmark_id="v1-vllm-smoke-cache-001",
        output_dir="/Volumes/catalog/schema/volume/v1-vllm-smoke",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        node_type_id="g6.8xlarge",
        single_user_name=SINGLE_USER_NAME,
        benchmark_repeats=3,
        request_parallelism=8,
        benchmark_arms=("baseline_prefill",),
        benchmark_prewarm_cache_prefix=True,
        benchmark_cache_runtime_prompt=True,
        benchmark_force_max_tokens=True,
        benchmark_prefix_cache_salt_mode="static",
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        model_dtype="float16",
        model_quantization="bitsandbytes",
        kv_cache_dtype="fp8",
        attention_backend="TRITON_ATTN",
        payload_cache_max_bytes=4096,
        dataset_specs=DATASET_SPECS,
    )

    payload = build_databricks_vllm_smoke_run_submit_payload(config)
    parameters = payload["tasks"][0]["spark_python_task"]["parameters"]

    assert parameters[parameters.index("--benchmark-repeats") + 1] == "3"
    assert parameters[parameters.index("--request-parallelism") + 1] == "8"
    assert parameters[parameters.index("--runtime-telemetry-interval-seconds") + 1] == "1.0"
    assert parameters[parameters.index("--benchmark-arm") + 1] == "baseline_prefill"
    assert "--benchmark-prewarm-cache-prefix" in parameters
    assert "--benchmark-cache-runtime-prompt" in parameters
    assert "--benchmark-force-max-tokens" in parameters
    assert parameters[parameters.index("--benchmark-prefix-cache-salt-mode") + 1] == "static"
    assert parameters[parameters.index("--model-id") + 1] == "Qwen/Qwen3-4B-Instruct-2507"
    assert parameters[parameters.index("--model-dtype") + 1] == "float16"
    assert parameters[parameters.index("--model-quantization") + 1] == "bitsandbytes"
    assert parameters[parameters.index("--kv-cache-dtype") + 1] == "fp8"
    assert parameters[parameters.index("--attention-backend") + 1] == "TRITON_ATTN"
    assert parameters[parameters.index("--payload-cache-max-bytes") + 1] == "4096"
    assert parameters.index("--benchmark-repeats") < parameters.index("--dataset")
    assert parameters.index("--request-parallelism") < parameters.index("--dataset")
    assert parameters.index("--runtime-telemetry-interval-seconds") < parameters.index("--dataset")
    assert parameters.index("--benchmark-arm") < parameters.index("--dataset")
    assert parameters.index("--benchmark-prewarm-cache-prefix") < parameters.index("--dataset")
    assert parameters.index("--benchmark-cache-runtime-prompt") < parameters.index("--dataset")
    assert parameters.index("--benchmark-prefix-cache-salt-mode") < parameters.index("--dataset")
    assert parameters.index("--model-id") < parameters.index("--dataset")
    assert parameters.index("--model-dtype") < parameters.index("--dataset")
    assert parameters.index("--model-quantization") < parameters.index("--dataset")
    assert parameters.index("--kv-cache-dtype") < parameters.index("--dataset")
    assert parameters.index("--attention-backend") < parameters.index("--dataset")
    assert parameters.index("--payload-cache-max-bytes") < parameters.index("--dataset")


def test_build_databricks_vllm_smoke_payload_enables_fail_closed_ram_priming():
    config = DatabricksVLLMSmokeJobConfig(
        benchmark_id="v1-vllm-ram-cache-001",
        output_dir="/Volumes/catalog/schema/volume/v1-vllm-ram-cache",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        node_type_id="g6.8xlarge",
        single_user_name=SINGLE_USER_NAME,
        benchmark_prewarm_payload_cache=True,
        payload_cache_max_bytes=8 * 1024 * 1024 * 1024,
        dataset_specs=DATASET_SPECS,
    )

    parameters = build_databricks_vllm_smoke_run_submit_payload(config)["tasks"][0][
        "spark_python_task"
    ]["parameters"]

    assert "--benchmark-prewarm-payload-cache" in parameters
    assert parameters[parameters.index("--payload-cache-max-bytes") + 1] == str(
        8 * 1024 * 1024 * 1024
    )
    assert parameters.index("--benchmark-prewarm-payload-cache") < parameters.index(
        "--dataset"
    )


def test_databricks_payload_forwards_arbitrary_arms_evidence_and_provenance():
    matrix = representative_canary_matrix()
    isolated_run = matrix.runs[2]
    provenance = {
        "engine_id": "vllm",
        "engine_version": VLLM_VERSION,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": "Qwen/Qwen3-4B-Instruct-2507",
        "tokenizer_revision": MODEL_REVISION,
        "input_tokens_target": 16384,
        "hardware_fingerprint": representative_vllm_environment_provenance(
            "aws-g6-l4"
        )["hardware_fingerprint"],
        "measurement_scopes": ["latency", "resource"],
    }
    config = DatabricksVLLMSmokeJobConfig(
        benchmark_id="representative-canary-16k-256",
        output_dir="/Volumes/catalog/schema/volume/canary-16k-256",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        node_type_id="g6.8xlarge",
        single_user_name=SINGLE_USER_NAME,
        **representative_job_kwargs(
            profile_id="vllm-16k-256-v1",
            arm_index=2,
            provenance=provenance,
        ),
    )

    payload = build_databricks_vllm_smoke_run_submit_payload(config)
    task = payload["tasks"][0]
    parameters = task["spark_python_task"]["parameters"]

    arm_specs = [
        json.loads(parameters[index + 1])
        for index, value in enumerate(parameters)
        if value == "--benchmark-arm-spec-json"
    ]
    assert [arm["arm_id"] for arm in arm_specs] == [isolated_run.arm_id]
    assert parameters[parameters.index("--benchmark-evidence-policy") + 1] == "canary"
    forwarded_provenance = json.loads(
        parameters[parameters.index("--benchmark-manifest-provenance-json") + 1]
    )
    assert all(
        forwarded_provenance[key] == value for key, value in provenance.items()
    )
    assert forwarded_provenance["canonical_model_id"] == (
        "Qwen/Qwen3-4B-Instruct-2507"
    )
    assert forwarded_provenance["serving_platform"] == "vllm"
    assert forwarded_provenance["model_dtype"] == "bfloat16"
    assert forwarded_provenance["model_quantization"] == "none"
    assert forwarded_provenance["runtime_kv_dtype"] == "bfloat16"
    assert forwarded_provenance["lora_id"] == "base"
    assert "--benchmark-force-max-tokens" in parameters
    assert "--representative-canary" in parameters
    assert parameters[
        parameters.index("--representative-workload-profile") + 1
    ] == "vllm-16k-256-v1"
    assert payload["timeout_seconds"] == 14400
    assert task["timeout_seconds"] == 14400
    assert task["max_retries"] == 0
    assert task["new_cluster"]["spark_env_vars"][
        "DOCUMENT_KV_EVICT_PAGE_CACHE"
    ] == "1"


def test_databricks_representative_provenance_builds_benchmark_runner_args(monkeypatch):
    config = DatabricksVLLMSmokeJobConfig(
        benchmark_id="representative-canary-runner-args",
        output_dir="/Volumes/catalog/schema/volume/representative-canary-runner-args",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        node_type_id="g6.8xlarge",
        single_user_name=SINGLE_USER_NAME,
        **representative_job_kwargs(),
    )
    parameters = build_databricks_vllm_smoke_run_submit_payload(config)["tasks"][0][
        "spark_python_task"
    ]["parameters"]
    assert parameters[-4::2] == ["--package-wheel-uri", "--package-wheel-sha256"]

    monkeypatch.setenv(
        DOCUMENT_KV_PACKAGE_WHEEL_SHA256_ENV,
        REPRESENTATIVE_WHEEL_SHA256,
    )
    smoke_config = parse_vllm_smoke_args(parameters[:-4])
    dataset_paths = parse_dataset_specs(
        smoke_config.dataset_specs,
        allow_subset=smoke_config.allow_dataset_subset,
    )
    runner_args = build_benchmark_runner_args(smoke_config, dataset_paths)

    package_revisions = [
        runner_args[index + 1]
        for index, value in enumerate(runner_args)
        if value == "--package-revision"
    ]
    assert f"vllm={VLLM_VERSION}" in package_revisions
    assert (
        f"cachet-kv=wheel-sha256:{REPRESENTATIVE_WHEEL_SHA256}"
        in package_revisions
    )


def test_databricks_representative_provenance_binds_resolved_rope_geometry(
    monkeypatch,
):
    class PreRopeLayout:
        lora_id = "base"
        layout_version = "qwen3-prerope-v1"
        payload_axis_order = "token_major"
        block_size = 16
        key_position_encoding = "pre_rope"
        rope_theta = 5_000_000.0
        rope_rotary_dim = 128

    monkeypatch.setattr(
        public_databricks_vllm_smoke_job,
        "layout_for_model",
        lambda *_args, **_kwargs: PreRopeLayout(),
    )
    config = DatabricksVLLMSmokeJobConfig(
        benchmark_id="representative-prerope-binding",
        output_dir="/Volumes/catalog/schema/volume/representative-prerope-binding",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        node_type_id="g6.8xlarge",
        single_user_name=SINGLE_USER_NAME,
        **representative_job_kwargs(arm_index=1),
    )

    parameters = build_databricks_vllm_smoke_run_submit_payload(config)["tasks"][0][
        "spark_python_task"
    ]["parameters"]
    provenance = json.loads(
        parameters[parameters.index("--benchmark-manifest-provenance-json") + 1]
    )
    assert provenance["rope_theta"] == 5_000_000.0
    assert provenance["rope_rotary_dim"] == 128


def test_databricks_representative_rejects_unresolved_rope_pair():
    with pytest.raises(ValueError, match="rope_rotary_dim, rope_theta"):
        DatabricksVLLMSmokeJobConfig(
            benchmark_id="representative-conflicting-rope",
            output_dir="/Volumes/catalog/schema/volume/representative-conflicting-rope",
            runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
            node_type_id="g6.8xlarge",
            single_user_name=SINGLE_USER_NAME,
            **representative_job_kwargs(
                arm_index=1,
                provenance={
                    "rope_theta": 1_000_000.0,
                    "rope_rotary_dim": 128,
                },
            ),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"run_timeout_seconds": 0}, "run_timeout_seconds"),
        ({"run_timeout_seconds": 14401}, "run_timeout_seconds"),
        ({"task_max_retries": 1}, "task_max_retries"),
    ],
)
def test_databricks_vllm_submission_bounds_cluster_runtime(overrides, message):
    with pytest.raises(ValueError, match=message):
        DatabricksVLLMSmokeJobConfig(
            benchmark_id="bounded-canary",
            output_dir="/Volumes/catalog/schema/volume/bounded-canary",
            runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
            node_type_id="g6.8xlarge",
            single_user_name=SINGLE_USER_NAME,
            **overrides,
        )


def test_databricks_vllm_representative_canary_requires_exact_timeout():
    with pytest.raises(ValueError, match="exactly 14400"):
        DatabricksVLLMSmokeJobConfig(
            benchmark_id="short-representative-canary",
            output_dir="/Volumes/catalog/schema/volume/short-representative-canary",
            runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
            node_type_id="g6.8xlarge",
            single_user_name=SINGLE_USER_NAME,
            run_timeout_seconds=3600,
            **representative_job_kwargs(),
        )


def test_databricks_vllm_representative_canary_requires_pins_and_local_handoffs():
    unpinned_kwargs = representative_job_kwargs(arm_index=2)
    unpinned_kwargs.pop("model_revision")
    unpinned_kwargs.pop("tokenizer_revision")
    with pytest.raises(ValueError, match="model_revision"):
        DatabricksVLLMSmokeJobConfig(
            benchmark_id="unpinned-canary",
            output_dir="/Volumes/catalog/schema/volume/unpinned-canary",
            runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
            node_type_id="g6.8xlarge",
            single_user_name=SINGLE_USER_NAME,
            **unpinned_kwargs,
        )
    with pytest.raises(ValueError, match="under /local_disk0"):
        DatabricksVLLMSmokeJobConfig(
            benchmark_id="remote-handoff-canary",
            output_dir="/Volumes/catalog/schema/volume/remote-handoff-canary",
            runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
            node_type_id="g6.8xlarge",
            single_user_name=SINGLE_USER_NAME,
            benchmark_handoff_generator_factory="module:factory",
            benchmark_handoff_output_dir="/Volumes/catalog/schema/volume/handoffs",
            **representative_job_kwargs(arm_index=2),
        )
    with pytest.raises(ValueError, match="under /local_disk0"):
        DatabricksVLLMSmokeJobConfig(
            benchmark_id="traversal-handoff-canary",
            output_dir="/Volumes/catalog/schema/volume/traversal-handoff-canary",
            runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
            node_type_id="g6.8xlarge",
            single_user_name=SINGLE_USER_NAME,
            benchmark_handoff_generator_factory="module:factory",
            benchmark_handoff_output_dir="/local_disk0/../tmp/handoffs",
            **representative_job_kwargs(arm_index=2),
        )


@pytest.mark.parametrize(
    ("hardware_target", "node_type_id"),
    [
        ("aws-g6-l4", "g6.4xlarge"),
        ("aws-g5-a10g", "g5.12xlarge"),
    ],
)
def test_databricks_vllm_representative_canary_requires_exact_node_size(
    hardware_target,
    node_type_id,
):
    with pytest.raises(ValueError, match="exact V1 node type"):
        DatabricksVLLMSmokeJobConfig(
            benchmark_id="wrong-size-canary",
            output_dir="/Volumes/catalog/schema/volume/wrong-size-canary",
            runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
            hardware_target=hardware_target,
            node_type_id=node_type_id,
            single_user_name=SINGLE_USER_NAME,
            **representative_job_kwargs(),
        )


@pytest.mark.parametrize(
    ("hardware_target", "node_type_id"),
    [
        ("aws-g6-l4", "g6.4xlarge"),
        ("aws-g5-a10g", "g5.12xlarge"),
    ],
)
def test_databricks_vllm_debug_job_preserves_family_node_overrides(
    hardware_target,
    node_type_id,
):
    config = DatabricksVLLMSmokeJobConfig(
        benchmark_id="debug-node-override",
        output_dir="/Volumes/catalog/schema/volume/debug-node-override",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        hardware_target=hardware_target,
        node_type_id=node_type_id,
        single_user_name=SINGLE_USER_NAME,
    )

    assert config.node_type_id == node_type_id


def test_databricks_vllm_nonrepresentative_canary_evidence_preserves_node_override():
    config = DatabricksVLLMSmokeJobConfig(
        benchmark_id="nonrepresentative-canary-evidence",
        output_dir="/Volumes/catalog/schema/volume/nonrepresentative-canary-evidence",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        hardware_target="aws-g6-l4",
        node_type_id="g6.4xlarge",
        single_user_name=SINGLE_USER_NAME,
        benchmark_evidence_policy="canary",
        model_revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
    )

    assert config.node_type_id == "g6.4xlarge"
    assert config.is_representative_submission is False


@pytest.mark.parametrize("arm_order", [(0, 1, 2), (2, 0, 1)])
def test_databricks_vllm_generic_matrix_remains_an_unlabelled_experiment(arm_order):
    matrix = representative_canary_matrix()

    config = DatabricksVLLMSmokeJobConfig(
        benchmark_id="generic-three-arm-matrix",
        output_dir="/Volumes/catalog/schema/volume/generic-three-arm-matrix",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        hardware_target="aws-g6-l4",
        node_type_id="g6.4xlarge",
        single_user_name=SINGLE_USER_NAME,
        benchmark_arm_specs=tuple(matrix.runs[index].arm_spec for index in arm_order),
    )

    parameters = build_databricks_vllm_smoke_run_submit_payload(config)["tasks"][0][
        "spark_python_task"
    ]["parameters"]
    assert config.is_representative_submission is False
    assert "--representative-canary" not in parameters


@pytest.mark.parametrize("arm_order", [(0, 1, 2), (2, 0, 1)])
def test_databricks_vllm_labelled_matrix_requires_isolated_jobs(arm_order):
    matrix = representative_canary_matrix()
    arm_specs = tuple(matrix.runs[index].arm_spec for index in arm_order)

    with pytest.raises(ValueError, match="exactly one fixed matrix arm"):
        DatabricksVLLMSmokeJobConfig(
            benchmark_id="labelled-three-arm-matrix",
            output_dir="/Volumes/catalog/schema/volume/labelled-three-arm-matrix",
            runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
            node_type_id="g6.8xlarge",
            single_user_name=SINGLE_USER_NAME,
            **representative_job_kwargs(arm_specs=arm_specs),
        )


@pytest.mark.parametrize("arm_index", [0, 1, 2])
def test_databricks_vllm_representative_accepts_one_fixed_arm_per_task(arm_index):
    config = DatabricksVLLMSmokeJobConfig(
        benchmark_id=f"isolated-representative-arm-{arm_index}",
        output_dir=f"/Volumes/catalog/schema/volume/isolated-arm-{arm_index}",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        node_type_id="g6.8xlarge",
        single_user_name=SINGLE_USER_NAME,
        **representative_job_kwargs(arm_index=arm_index),
    )

    parameters = build_databricks_vllm_smoke_run_submit_payload(config)["tasks"][0][
        "spark_python_task"
    ]["parameters"]
    assert parameters.count("--benchmark-arm-spec-json") == 1
    assert parameters[parameters.index("--benchmark-suite-id") + 1] == (
        "g6-vllm-8k-64"
    )
    assert parameters[parameters.index("--runtime-id") + 1] == (
        REPRESENTATIVE_TASK_RUNTIME_ID_REFERENCE
    )
    assert "--representative-canary" in parameters
    assert "--representative-workload-profile" in parameters


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"benchmark_suite_id": "forged-suite"}, "comparison group"),
        ({"benchmark_runtime_id": "static-run"}, "retry-unique"),
    ],
)
def test_databricks_vllm_representative_rejects_mismatched_execution_identity(
    override,
    message,
):
    with pytest.raises(ValueError, match=message):
        DatabricksVLLMSmokeJobConfig(
            benchmark_id="g6-vllm-8k-64-baseline",
            output_dir=(
                "/Volumes/catalog/schema/volume/g6-vllm-8k-64-baseline"
            ),
            runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
            node_type_id="g6.8xlarge",
            single_user_name=SINGLE_USER_NAME,
            **representative_job_kwargs(),
            **override,
        )


@pytest.mark.parametrize(
    ("representative_canary", "profile_id"),
    [(True, None), (False, "vllm-8k-64-v1")],
)
def test_databricks_vllm_representative_flag_and_profile_are_atomic(
    representative_canary,
    profile_id,
):
    with pytest.raises(ValueError, match="must be provided together"):
        DatabricksVLLMSmokeJobConfig(
            benchmark_id="partial-representative-label",
            output_dir="/Volumes/catalog/schema/volume/partial-label",
            runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
            node_type_id="g6.8xlarge",
            single_user_name=SINGLE_USER_NAME,
            representative_canary=representative_canary,
            representative_workload_profile=profile_id,
        )


@pytest.mark.parametrize("profile_id", ["vllm-8k-64-v1", "vllm-16k-256-v1"])
def test_databricks_vllm_accepts_registered_representative_workloads(profile_id):
    config = DatabricksVLLMSmokeJobConfig(
        benchmark_id=profile_id,
        output_dir=f"/Volumes/catalog/schema/volume/{profile_id}",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        node_type_id="g6.8xlarge",
        single_user_name=SINGLE_USER_NAME,
        **representative_job_kwargs(profile_id=profile_id),
    )

    assert config.is_representative_submission is True
    assert config.representative_workload_profile is not None
    assert config.representative_workload_profile.profile_id == profile_id


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"max_tokens": 32}, "max_tokens"),
        ({"max_model_len": 4096}, "max_model_len"),
        ({"benchmark_repeats": 2}, "benchmark_repeats"),
        ({"request_parallelism": 2}, "request_parallelism"),
        ({"benchmark_force_max_tokens": False}, "benchmark_force_max_tokens"),
        (
            {"benchmark_prefix_cache_salt_mode": "static"},
            "benchmark_prefix_cache_salt_mode",
        ),
        ({"benchmark_cache_runtime_prompt": True}, "benchmark_cache_runtime_prompt"),
        ({"payload_cache_max_bytes": 1}, "payload_cache_max_bytes"),
        (
            {"benchmark_manifest_provenance": {"input_tokens_target": 8193}},
            "input_tokens_target",
        ),
    ],
)
def test_databricks_vllm_representative_profile_rejects_workload_drift(
    override,
    message,
):
    kwargs = representative_job_kwargs()
    kwargs.update(override)

    with pytest.raises(ValueError, match=message):
        DatabricksVLLMSmokeJobConfig(
            benchmark_id="drifted-representative-workload",
            output_dir="/Volumes/catalog/schema/volume/drifted-workload",
            runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
            node_type_id="g6.8xlarge",
            single_user_name=SINGLE_USER_NAME,
            **kwargs,
        )


def test_databricks_vllm_fixed_single_arm_without_profile_stays_generic():
    config = DatabricksVLLMSmokeJobConfig(
        benchmark_id="generic-fixed-arm",
        output_dir="/Volumes/catalog/schema/volume/generic-fixed-arm",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        hardware_target="aws-g6-l4",
        node_type_id="g6.4xlarge",
        single_user_name=SINGLE_USER_NAME,
        local_root="/tmp/generic-fixed-arm",
        benchmark_arm_specs=(representative_canary_matrix().runs[0].arm_spec,),
    )

    parameters = build_databricks_vllm_smoke_run_submit_payload(config)["tasks"][0][
        "spark_python_task"
    ]["parameters"]
    assert config.is_representative_submission is False
    assert "--representative-canary" not in parameters
    assert "--representative-workload-profile" not in parameters


def test_databricks_vllm_representative_rejects_non_matrix_arm():
    non_matrix_arm = {
        "arm_id": "baseline_prefill",
        "uses_cache": False,
        "description": "modified semantics",
        "implementation_kind": "baseline",
        "physical_transform_id": "identity",
        "physical_transform_version": "1",
    }
    with pytest.raises(ValueError, match="exactly one fixed matrix arm"):
        DatabricksVLLMSmokeJobConfig(
            benchmark_id="non-matrix-representative-arm",
            output_dir="/Volumes/catalog/schema/volume/non-matrix-arm",
            runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
            node_type_id="g6.8xlarge",
            single_user_name=SINGLE_USER_NAME,
            **representative_job_kwargs(arm_specs=(non_matrix_arm,)),
        )


def test_databricks_vllm_generic_canary_only_requires_pinned_identity():
    config = DatabricksVLLMSmokeJobConfig(
        benchmark_id="generic-canary-on-other-g6-shape",
        output_dir="/Volumes/catalog/schema/volume/generic-canary",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        node_type_id="g6.12xlarge",
        single_user_name=SINGLE_USER_NAME,
        local_root="/tmp/cachet-generic-canary",
        model_revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        benchmark_evidence_policy="canary",
    )

    payload = build_databricks_vllm_smoke_run_submit_payload(config)
    task = payload["tasks"][0]
    parameters = task["spark_python_task"]["parameters"]
    assert "--representative-canary" not in parameters
    assert "DOCUMENT_KV_EVICT_PAGE_CACHE" not in task["new_cluster"].get(
        "spark_env_vars", {}
    )


def test_databricks_vllm_smoke_config_requires_single_user_name():
    try:
        DatabricksVLLMSmokeJobConfig(
            benchmark_id="v1-vllm-smoke-001",
            output_dir="/Volumes/catalog/schema/volume/v1-vllm-smoke",
            runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        )
    except ValueError as exc:
        assert "single_user_name is required" in str(exc)
    else:
        raise AssertionError("expected SINGLE_USER validation to fail")


def test_databricks_vllm_smoke_config_validates_benchmark_sizing_and_datasets():
    invalid_cases = [
        ({"max_model_len": 0}, "max_model_len must be positive"),
        ({"model_id": ""}, "model_id must be non-empty"),
        ({"model_dtype": ""}, "model_dtype must be non-empty"),
        ({"model_quantization": ""}, "model_quantization must be non-empty"),
        ({"kv_cache_dtype": ""}, "kv_cache_dtype must be non-empty"),
        (
            {"node_type_id": "g5.8xlarge", "kv_cache_dtype": "fp8"},
            "fp8_e5m2",
        ),
        ({"attention_backend": ""}, "attention_backend must be non-empty"),
        ({"max_num_seqs": 0}, "max_num_seqs must be positive"),
        ({"gpu_memory_utilization": 0}, "gpu_memory_utilization must be in"),
        ({"gpu_memory_utilization": 1.1}, "gpu_memory_utilization must be in"),
        ({"benchmark_repeats": 0}, "benchmark_repeats must be a positive integer"),
        ({"request_parallelism": 0}, "request_parallelism must be a positive integer"),
        ({"runtime_telemetry_interval_seconds": 0}, "runtime_telemetry_interval_seconds must be positive"),
        ({"benchmark_arms": ("unknown",)}, "Unknown benchmark arms"),
        ({"allow_dataset_subset": "yes"}, "allow_dataset_subset must be a boolean"),
        (
            {"benchmark_prewarm_cache_prefix": True},
            "benchmark_prewarm_cache_prefix requires prepared dataset specs",
        ),
        (
            {"benchmark_prewarm_payload_cache": True},
            "benchmark_prewarm_payload_cache requires prepared dataset specs",
        ),
        (
            {
                "benchmark_prewarm_payload_cache": True,
                "dataset_specs": DATASET_SPECS,
            },
            "requires a positive payload_cache_max_bytes",
        ),
        (
            {
                "benchmark_prewarm_cache_prefix": True,
                "benchmark_prefix_cache_salt_mode": "per_request",
                "dataset_specs": DATASET_SPECS,
            },
            "requires benchmark_prefix_cache_salt_mode='static'",
        ),
        ({"benchmark_force_max_tokens": "yes"}, "benchmark_force_max_tokens must be a boolean"),
        ({"benchmark_prefix_cache_salt_mode": "dynamic"}, "benchmark_prefix_cache_salt_mode"),
        ({"payload_cache_max_bytes": -1}, "payload_cache_max_bytes must be a non-negative integer"),
        ({"dataset_specs": ("biography=/tmp/biography.jsonl",)}, "dataset specs missing required V1 datasets"),
        (
            {"benchmark_handoff_generator_factory": "document_kv_cache.transformers_generator:build"},
            "requires prepared dataset specs",
        ),
        (
            {"benchmark_cache_runtime_prompt": True},
            "benchmark_cache_runtime_prompt requires prepared dataset specs",
        ),
        (
            {"benchmark_handoff_output_dir": "/Volumes/catalog/schema/volume/handoffs"},
            "requires benchmark_handoff_generator_factory",
        ),
        ({"benchmark_handoff_dtype": ""}, "benchmark_handoff_dtype must be non-empty"),
        ({"benchmark_handoff_align_bytes": 0}, "benchmark_handoff_align_bytes must be a positive integer"),
        (
            {"benchmark_handoff_generation_timeout_seconds": 0},
            "benchmark_handoff_generation_timeout_seconds must be positive",
        ),
        (
            {"benchmark_handoff_limit": -1},
            "benchmark_handoff_limit must be a non-negative integer",
        ),
        (
            {"benchmark_handoff_segment_per_document": True},
            "benchmark handoff options require",
        ),
        (
            {
                "benchmark_handoff_generator_factory": "module:factory",
                "benchmark_handoff_cache_method": "vanilla_prefill",
                "dataset_specs": DATASET_SPECS,
            },
            "vanilla_prefill handoff generation requires one segment per document",
        ),
        ({"spark_env_vars": {"BAD-NAME": "value"}}, "valid environment variable name"),
        ({"spark_env_vars": {"DATABRICKS_TOKEN": "redacted"}}, "looks secret-bearing"),
    ]

    for overrides, message in invalid_cases:
        kwargs = {
            "benchmark_id": "v1-vllm-smoke-001",
            "output_dir": "/Volumes/catalog/schema/volume/v1-vllm-smoke",
            "runner_python_file": "dbfs:/benchmarks/run_vllm_smoke.py",
            "single_user_name": SINGLE_USER_NAME,
        }
        kwargs.update(overrides)
        try:
            DatabricksVLLMSmokeJobConfig(**kwargs)
        except (TypeError, ValueError) as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"expected validation to fail for {overrides!r}")

    config = DatabricksVLLMSmokeJobConfig(
        benchmark_id="v1-vllm-smoke-001",
        output_dir="/Volumes/catalog/schema/volume/v1-vllm-smoke",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        single_user_name=SINGLE_USER_NAME,
        node_type_id="g5.8xlarge",
        kv_cache_dtype="fp8_e5m2",
    )
    assert config.hardware_target == "aws-g5-a10g"
    assert config.kv_cache_dtype == "fp8_e5m2"

    subset_config = DatabricksVLLMSmokeJobConfig(
        benchmark_id="v1-vllm-score-biography",
        output_dir="/Volumes/catalog/schema/volume/v1-vllm-score-biography",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        single_user_name=SINGLE_USER_NAME,
        dataset_specs=("biography=dbfs:/benchmarks/cachet/full-score-datasets-20260628/biography.jsonl",),
        allow_dataset_subset=True,
    )
    parameters = build_databricks_vllm_smoke_run_submit_payload(subset_config)["tasks"][0]["spark_python_task"][
        "parameters"
    ]
    assert "--allow-dataset-subset" in parameters
    assert parameters[parameters.index("--dataset") + 1].startswith("biography=")


def test_databricks_vllm_smoke_payload_passes_prepared_handoff_generation_flags():
    config = DatabricksVLLMSmokeJobConfig(
        benchmark_id="v1-vllm-prepared-001",
        output_dir="/Volumes/catalog/schema/volume/v1-vllm-prepared",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        single_user_name=SINGLE_USER_NAME,
        dataset_specs=DATASET_SPECS,
        benchmark_handoff_generator_factory=(
            "document_kv_cache.transformers_generator:build_transformers_kv_chunk_generator"
        ),
        benchmark_handoff_output_dir="/Volumes/catalog/schema/volume/v1-vllm-prepared/handoffs",
        benchmark_handoff_dtype="bfloat16",
        benchmark_handoff_align_bytes=1,
        benchmark_handoff_generation_timeout_seconds=1234.0,
        benchmark_handoff_limit=2,
        benchmark_handoff_segment_per_document=True,
        benchmark_handoff_cache_method="vanilla_prefill",
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        model_revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        runtime_identity=stored_post_rope_runtime_identity(),
        spark_env_vars={
            "CACHET_TRANSFORMERS_DEVICE": "cuda",
            "CACHET_TRANSFORMERS_TORCH_DTYPE": "bfloat16",
            "CACHET_TRANSFORMERS_TRUST_REMOTE_CODE": "true",
        },
    )

    task = build_databricks_vllm_smoke_run_submit_payload(config)["tasks"][0]
    parameters = task["spark_python_task"]["parameters"]

    assert parameters[parameters.index("--benchmark-handoff-generator-factory") + 1] == (
        "document_kv_cache.transformers_generator:build_transformers_kv_chunk_generator"
    )
    assert parameters[parameters.index("--benchmark-handoff-output-dir") + 1] == (
        "/Volumes/catalog/schema/volume/v1-vllm-prepared/handoffs"
    )
    assert parameters[parameters.index("--benchmark-handoff-dtype") + 1] == "bfloat16"
    assert parameters[parameters.index("--benchmark-handoff-align-bytes") + 1] == "1"
    assert (
        parameters[parameters.index("--benchmark-handoff-generation-timeout-seconds") + 1]
        == "1234.0"
    )
    assert parameters[parameters.index("--benchmark-handoff-limit") + 1] == "2"
    assert "--benchmark-handoff-chunk-per-document" in parameters
    assert (
        parameters[parameters.index("--benchmark-handoff-cache-method") + 1]
        == "vanilla_prefill"
    )
    assert "--benchmark-handoff-allow-legacy-artifact-contract" not in parameters
    assert parameters[parameters.index("--model-revision") + 1] == MODEL_REVISION
    assert (
        parameters[parameters.index("--tokenizer-revision") + 1]
        == MODEL_REVISION
    )
    runtime_identity = json.loads(
        parameters[parameters.index("--runtime-identity-json") + 1]
    )
    assert runtime_identity == stored_post_rope_runtime_identity().to_record()
    assert task["new_cluster"]["spark_env_vars"] == {
        "CACHET_TRANSFORMERS_DEVICE": "cuda",
        "CACHET_TRANSFORMERS_TORCH_DTYPE": "bfloat16",
        "CACHET_TRANSFORMERS_TRUST_REMOTE_CODE": "true",
    }


def test_databricks_handoff_contract_defaults_strict_and_legacy_opt_out_is_explicit():
    strict = DatabricksVLLMSmokeJobConfig(
        benchmark_id="strict-handoff",
        output_dir="/Volumes/catalog/schema/volume/strict-handoff",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        single_user_name=SINGLE_USER_NAME,
        benchmark_handoff_generator_factory="module:factory",
        dataset_specs=DATASET_SPECS,
    )
    strict_parameters = build_databricks_vllm_smoke_run_submit_payload(strict)["tasks"][0][
        "spark_python_task"
    ]["parameters"]
    assert strict.benchmark_handoff_require_artifact_contract is True
    assert "--benchmark-handoff-allow-legacy-artifact-contract" not in strict_parameters

    legacy = DatabricksVLLMSmokeJobConfig(
        benchmark_id="legacy-handoff",
        output_dir="/Volumes/catalog/schema/volume/legacy-handoff",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        single_user_name=SINGLE_USER_NAME,
        benchmark_handoff_generator_factory="module:factory",
        benchmark_handoff_require_artifact_contract=False,
        dataset_specs=DATASET_SPECS,
    )
    legacy_parameters = build_databricks_vllm_smoke_run_submit_payload(legacy)["tasks"][0][
        "spark_python_task"
    ]["parameters"]
    assert "--benchmark-handoff-allow-legacy-artifact-contract" in legacy_parameters

    with pytest.raises(ValueError, match="canary and publication"):
        DatabricksVLLMSmokeJobConfig(
            benchmark_id="legacy-canary",
            output_dir="/Volumes/catalog/schema/volume/legacy-canary",
            runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
            single_user_name=SINGLE_USER_NAME,
            benchmark_handoff_generator_factory="module:factory",
            benchmark_handoff_require_artifact_contract=False,
            benchmark_evidence_policy="canary",
            dataset_specs=DATASET_SPECS,
        )


def test_write_databricks_vllm_smoke_runner_script_imports_smoke_main(tmp_path):
    path = tmp_path / "run_vllm_smoke.py"

    write_databricks_vllm_smoke_runner_script(path)

    runner_text = path.read_text(encoding="utf-8")
    assert "--package-wheel-uri" in runner_text
    assert "DOCUMENT_KV_PACKAGE_INSTALL_SPEC" in runner_text
    assert "pip\", \"install\"" in runner_text
    assert "dbfs:/" in runner_text
    assert "document_kv_cache.vllm_smoke" in runner_text
    assert "if exit_code:" in runner_text


def test_representative_vllm_runner_digest_matches_embedded_script():
    assert (
        hashlib.sha256(VLLM_SMOKE_RUNNER_SCRIPT.encode("utf-8")).hexdigest()
        == REPRESENTATIVE_VLLM_RUNNER_SHA256
    )


def test_generated_vllm_smoke_runner_installs_wheel_before_forwarding_args(tmp_path):
    runner_path = tmp_path / "run_vllm_smoke.py"
    pip_call_path = tmp_path / "pip-call.json"
    main_args_path = tmp_path / "main-args.json"
    events_path = tmp_path / "events.jsonl"
    wheel_path = tmp_path / "cachet_kv-0.2.0-py3-none-any.whl"
    wheel_path.write_bytes(b"verified Cachet wheel bytes")
    wheel_sha256 = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    package_dir = tmp_path / "document_kv_cache"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "vllm_smoke.py").write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "",
                "with open(os.environ['RUNNER_EVENTS_JSONL'], 'a', encoding='utf-8') as handle:",
                "    handle.write(json.dumps({'event': 'vllm_smoke_import'}) + '\\n')",
                "",
                "def main(argv=None):",
                "    with open(os.environ['RUNNER_EVENTS_JSONL'], 'a', encoding='utf-8') as handle:",
                "        handle.write(json.dumps({'event': 'main'}) + '\\n')",
                "    with open(os.environ['MAIN_ARGS_JSON'], 'w', encoding='utf-8') as handle:",
                "        json.dump({",
                "            'argv': argv,",
                "            'package_install_spec': os.environ.get('DOCUMENT_KV_PACKAGE_INSTALL_SPEC'),",
                "            'package_wheel_sha256': os.environ.get('DOCUMENT_KV_PACKAGE_WHEEL_SHA256'),",
                "        }, handle)",
                "    return 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "sitecustomize.py").write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "import subprocess",
                "",
                "def _capture_check_call(argv):",
                "    with open(os.environ['RUNNER_EVENTS_JSONL'], 'a', encoding='utf-8') as handle:",
                "        handle.write(json.dumps({'event': 'pip_install'}) + '\\n')",
                "    with open(os.environ['PIP_CALL_JSON'], 'w', encoding='utf-8') as handle:",
                "        json.dump(argv, handle)",
                "    return 0",
                "",
                "subprocess.check_call = _capture_check_call",
                "",
            ]
        ),
        encoding="utf-8",
    )

    write_databricks_vllm_smoke_runner_script(runner_path)
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "PIP_CALL_JSON": str(pip_call_path),
        "MAIN_ARGS_JSON": str(main_args_path),
        "RUNNER_EVENTS_JSONL": str(events_path),
        "DOCUMENT_KV_PACKAGE_WHEEL_SHA256": "a" * 64,
    }

    subprocess.run(
        [
            sys.executable,
            str(runner_path),
            "--package-wheel-uri",
            str(wheel_path),
            "--package-wheel-sha256",
            wheel_sha256,
            "--benchmark-id",
            "v1-vllm-smoke-001",
            "--output-dir",
            "/dbfs/tmp/cachet/output",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    pip_call = json.loads(pip_call_path.read_text(encoding="utf-8"))
    assert Path(pip_call[0]).resolve() == Path(sys.executable).resolve()
    assert pip_call[1:] == [
        "-m",
        "pip",
        "install",
        str(wheel_path),
    ]
    main_payload = json.loads(main_args_path.read_text(encoding="utf-8"))
    assert main_payload == {
        "argv": [
            "--benchmark-id",
            "v1-vllm-smoke-001",
            "--output-dir",
            "/dbfs/tmp/cachet/output",
        ],
        "package_install_spec": str(wheel_path),
        "package_wheel_sha256": wheel_sha256,
    }
    events = [json.loads(line)["event"] for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert events == ["pip_install", "vllm_smoke_import", "main"]

    subprocess.run(
        [
            sys.executable,
            str(runner_path),
            "--package-wheel-uri",
            str(wheel_path),
            "--benchmark-id",
            "generic-vllm-smoke",
            "--output-dir",
            "/dbfs/tmp/cachet/generic-output",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    generic_main_payload = json.loads(main_args_path.read_text(encoding="utf-8"))
    assert generic_main_payload["package_wheel_sha256"] is None


def test_generated_vllm_smoke_runner_rejects_tampered_wheel(tmp_path):
    runner_path = tmp_path / "run_vllm_smoke.py"
    wheel_path = tmp_path / "cachet_kv-0.2.0-py3-none-any.whl"
    wheel_path.write_bytes(b"tampered Cachet wheel bytes")
    write_databricks_vllm_smoke_runner_script(runner_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(runner_path),
            "--package-wheel-uri",
            str(wheel_path),
            "--package-wheel-sha256",
            "0" * 64,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "Cachet package wheel SHA-256 does not match" in completed.stderr


def test_write_databricks_vllm_smoke_run_submit_json_writes_payload(tmp_path):
    path = tmp_path / "payload.json"

    write_databricks_vllm_smoke_run_submit_json(
        DatabricksVLLMSmokeJobConfig(
            benchmark_id="v1-vllm-smoke-001",
            output_dir="/Volumes/catalog/schema/volume/v1-vllm-smoke",
            runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
            single_user_name=SINGLE_USER_NAME,
        ),
        path,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["tasks"][0]["task_key"] == DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY


def test_main_writes_vllm_smoke_payload_and_runner_script(tmp_path):
    payload_path = tmp_path / "payload.json"
    runner_path = tmp_path / "run_vllm_smoke.py"

    exit_code = main(
        [
            "--benchmark-id",
            "v1-vllm-smoke-001",
            "--output-dir",
            "/Volumes/catalog/schema/volume/v1-vllm-smoke",
            "--runner-python-file",
            "dbfs:/benchmarks/run_vllm_smoke.py",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--wheel-uri",
            WHEEL_URI,
            "--run-timeout-seconds",
            "3600",
            "--task-max-retries",
            "0",
            "--spark-env-var",
            "CACHET_TRANSFORMERS_DEVICE=cuda",
            "--output-json",
            str(payload_path),
            "--runner-script-output",
            str(runner_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    task = payload["tasks"][0]
    assert payload["timeout_seconds"] == 3600
    assert task["timeout_seconds"] == 3600
    assert task["max_retries"] == 0
    assert "libraries" not in task
    assert task["spark_python_task"]["parameters"][-2:] == ["--package-wheel-uri", WHEEL_URI]
    assert task["new_cluster"]["spark_env_vars"] == {"CACHET_TRANSFORMERS_DEVICE": "cuda"}
    assert "vllm_smoke" in runner_path.read_text(encoding="utf-8")


def test_main_forwards_registered_vllm_representative_profile(tmp_path):
    payload_path = tmp_path / "representative-payload.json"
    arm_spec = representative_canary_matrix().runs[0].arm_spec
    argv = [
        "--benchmark-id",
        "vllm-8k-64-v1",
        "--output-dir",
        "/Volumes/catalog/schema/volume/vllm-8k-64-v1",
        "--runner-python-file",
        "dbfs:/benchmarks/run_vllm_smoke.py",
        "--run-timeout-seconds",
        "14400",
        "--single-user-name",
        SINGLE_USER_NAME,
        "--node-type-id",
        "g6.8xlarge",
        "--model-revision",
        MODEL_REVISION,
        "--wheel-uri",
        REPRESENTATIVE_WHEEL_URI,
        "--wheel-sha256",
        REPRESENTATIVE_WHEEL_SHA256,
        "--tokenizer-revision",
        MODEL_REVISION,
        "--max-tokens",
        "64",
        "--max-model-len",
        "8512",
        "--benchmark-repeats",
        "3",
        "--benchmark-arm-spec-json",
        json.dumps(dict(arm_spec)),
        "--benchmark-evidence-policy",
        "canary",
        "--representative-canary",
        "--representative-workload-profile",
        "vllm-8k-64-v1",
        "--benchmark-manifest-provenance-json",
        json.dumps({"input_tokens_target": 8192}),
        "--benchmark-force-max-tokens",
    ]
    for dataset_spec in DATASET_SPECS:
        argv.extend(["--dataset", dataset_spec])
    argv.extend(["--output-json", str(payload_path)])

    assert main(argv) == 0

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    task = payload["tasks"][0]
    parameters = task["spark_python_task"]["parameters"]
    assert payload["timeout_seconds"] == 14400
    assert task["timeout_seconds"] == 14400
    assert parameters[
        parameters.index("--representative-workload-profile") + 1
    ] == "vllm-8k-64-v1"


def test_main_derives_vllm_smoke_node_type_from_g5_hardware_target(tmp_path):
    payload_path = tmp_path / "payload.json"

    exit_code = main(
        [
            "--benchmark-id",
            "v1-vllm-smoke-001",
            "--output-dir",
            "/Volumes/catalog/schema/volume/v1-vllm-smoke",
            "--runner-python-file",
            "dbfs:/benchmarks/run_vllm_smoke.py",
            "--hardware-target",
            "aws-g5-a10g",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(payload_path),
        ]
    )

    task = json.loads(payload_path.read_text(encoding="utf-8"))["tasks"][0]
    cluster = task["new_cluster"]
    parameters = task["spark_python_task"]["parameters"]
    assert exit_code == 0
    assert cluster["node_type_id"] == "g5.8xlarge"
    assert cluster["driver_node_type_id"] == "g5.8xlarge"
    assert parameters[parameters.index("--hardware-target") + 1] == "aws-g5-a10g"


def test_main_preserves_legacy_vllm_smoke_g5_node_type_without_hardware_target(tmp_path):
    payload_path = tmp_path / "payload.json"

    exit_code = main(
        [
            "--benchmark-id",
            "v1-vllm-smoke-001",
            "--output-dir",
            "/Volumes/catalog/schema/volume/v1-vllm-smoke",
            "--runner-python-file",
            "dbfs:/benchmarks/run_vllm_smoke.py",
            "--node-type-id",
            "g5.8xlarge",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(payload_path),
        ]
    )

    task = json.loads(payload_path.read_text(encoding="utf-8"))["tasks"][0]
    cluster = task["new_cluster"]
    parameters = task["spark_python_task"]["parameters"]
    assert exit_code == 0
    assert cluster["node_type_id"] == "g5.8xlarge"
    assert cluster["driver_node_type_id"] == "g5.8xlarge"
    assert parameters[parameters.index("--hardware-target") + 1] == "aws-g5-a10g"
