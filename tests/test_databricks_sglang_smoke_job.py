import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from document_kv_cache.benchmarks import SUPPORTED_V1_DATASETS
from document_kv_cache.canary_orchestration import (
    REPRESENTATIVE_SGLANG_RUNNER_SHA256,
)
from document_kv_cache.databricks_sglang_smoke_job import (
    DEFAULT_DATABRICKS_SGLANG_SMOKE_PURPOSE,
    DEFAULT_DATABRICKS_SGLANG_SMOKE_RUN_NAME,
    DEFAULT_DATABRICKS_SGLANG_SMOKE_TASK_KEY,
    SGLANG_SMOKE_RUNNER_SCRIPT,
    DatabricksSGLangSmokeJobConfig,
    build_databricks_sglang_smoke_run_submit_payload,
    main,
    write_databricks_sglang_smoke_run_submit_json,
    write_databricks_sglang_smoke_runner_script,
)
from document_kv_cache.sglang_smoke import (
    DEFAULT_SGLANG_HICACHE_PAGE_SIZE,
    DEFAULT_SGLANG_PREPARED_HICACHE_PAGE_SIZE,
    DEFAULT_SGLANG_HICACHE_STORAGE_PREFETCH_POLICY,
    DEFAULT_SGLANG_HICACHE_STORAGE_PREFETCH_THRESHOLD,
    DEFAULT_SGLANG_LIVE_HANDOFF_GENERATOR_FACTORY,
    DEFAULT_SGLANG_LIVE_CHECK_PROMPT_FORMAT,
    DEFAULT_SGLANG_LIVE_CHECK_REQUEST_MODE,
    DEFAULT_SGLANG_LIVE_CHECK_TEMPERATURE,
    DEFAULT_SGLANG_FLUSH_CACHE_BEFORE_CACHE_ARM,
    DEFAULT_SGLANG_FLUSH_CACHE_BEFORE_CANARY,
    DEFAULT_SGLANG_FLUSH_CACHE_TIMEOUT_SECONDS,
    DEFAULT_SGLANG_LIVE_BENCHMARK_REPEATS,
    SGLANG_REPRESENTATIVE_WORKLOAD_PROFILES,
    SGLANG_BASELINE_HANDOFF_FIELDS_UNSUPPORTED_MESSAGE,
    SGLANG_GENERATED_HANDOFF_EXPLICIT_FIELDS_UNSUPPORTED_MESSAGE,
    SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE,
)


WHEEL_URI = "/Volumes/catalog/schema/volume/wheels/cachet_kv-0.2.0-py3-none-any.whl"
SINGLE_USER_NAME = "user@example.com"
HANDOFF_JSON = "/Volumes/catalog/schema/volume/live/sglang-live.handoff.json"
PAGE_KEYS_JSON = '["page-a","page-b"]'
DATASET_SPECS = tuple(
    f"{dataset}=/Volumes/catalog/schema/volume/v1/{dataset}.jsonl"
    for dataset in SUPPORTED_V1_DATASETS
)
REPRESENTATIVE_WORKLOAD_PROFILE_ID = "sglang-4k-32-v1"
REPRESENTATIVE_WHEEL_SHA256 = "f" * 64
REPRESENTATIVE_WHEEL_URI = (
    "dbfs:/cachet/wheels/"
    f"{REPRESENTATIVE_WHEEL_SHA256}/cachet_kv-0.2.0-py3-none-any.whl"
)
REPRESENTATIVE_WORKLOAD_KWARGS = {
    "wheel_uri": REPRESENTATIVE_WHEEL_URI,
    "wheel_sha256": REPRESENTATIVE_WHEEL_SHA256,
    "representative_canary": True,
    "representative_workload_profile": REPRESENTATIVE_WORKLOAD_PROFILE_ID,
    "context_length": 4096,
    "max_tokens": 32,
    "live_benchmark_repeats": 2,
    "sglang_attention_backend": "triton",
    "sglang_sampling_backend": "pytorch",
    "sglang_enable_deterministic_inference": True,
}


def test_databricks_sglang_representative_requires_content_addressed_wheel():
    kwargs = dict(REPRESENTATIVE_WORKLOAD_KWARGS)
    kwargs.pop("wheel_sha256")

    with pytest.raises(ValueError, match="wheel_sha256"):
        DatabricksSGLangSmokeJobConfig(
            benchmark_id="representative-sglang-missing-wheel-digest",
            output_dir=(
                "/Volumes/catalog/schema/volume/"
                "representative-sglang-missing-wheel-digest"
            ),
            runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
            node_type_id="g6.8xlarge",
            single_user_name=SINGLE_USER_NAME,
            model_revision="a" * 40,
            tokenizer_revision="a" * 40,
            generate_live_handoff=True,
            **kwargs,
        )


def test_databricks_sglang_forwards_matching_pinned_model_and_tokenizer_revision():
    revision = "a" * 40
    config = DatabricksSGLangSmokeJobConfig(
        benchmark_id="pinned-sglang-canary",
        output_dir="/Volumes/catalog/schema/volume/pinned-sglang-canary",
        runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
        node_type_id="g6.8xlarge",
        single_user_name=SINGLE_USER_NAME,
        model_revision=revision,
        tokenizer_revision=revision,
        generate_live_handoff=True,
        **REPRESENTATIVE_WORKLOAD_KWARGS,
    )

    parameters = build_databricks_sglang_smoke_run_submit_payload(config)["tasks"][0][
        "spark_python_task"
    ]["parameters"]
    assert parameters[parameters.index("--model-revision") + 1] == revision
    assert parameters[parameters.index("--tokenizer-revision") + 1] == revision
    assert "--representative-canary" in parameters
    assert (
        parameters[parameters.index("--representative-workload-profile") + 1]
        == REPRESENTATIVE_WORKLOAD_PROFILE_ID
    )
    payload = build_databricks_sglang_smoke_run_submit_payload(config)
    assert payload["timeout_seconds"] == 14400
    assert payload["tasks"][0]["timeout_seconds"] == 14400
    assert payload["tasks"][0]["max_retries"] == 0
    assert payload["tasks"][0]["new_cluster"]["spark_env_vars"][
        "DOCUMENT_KV_EVICT_PAGE_CACHE"
    ] == "1"

    with pytest.raises(ValueError, match="revisions must match"):
        DatabricksSGLangSmokeJobConfig(
            benchmark_id="mismatched-sglang-canary",
            output_dir="/Volumes/catalog/schema/volume/mismatched-sglang-canary",
            runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
            node_type_id="g6.8xlarge",
            single_user_name=SINGLE_USER_NAME,
            baseline_only=True,
            model_revision="a" * 40,
            tokenizer_revision="b" * 40,
        )
    with pytest.raises(ValueError, match="under /local_disk0"):
        DatabricksSGLangSmokeJobConfig(
            benchmark_id="traversal-sglang-canary",
            output_dir="/Volumes/catalog/schema/volume/traversal-sglang-canary",
            runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
            node_type_id="g6.8xlarge",
            single_user_name=SINGLE_USER_NAME,
            model_revision=revision,
            tokenizer_revision=revision,
            generate_live_handoff=True,
            live_handoff_output_dir="/local_disk0/../tmp/handoff",
            **REPRESENTATIVE_WORKLOAD_KWARGS,
        )


def test_databricks_sglang_representative_profile_is_registered():
    assert len(SGLANG_REPRESENTATIVE_WORKLOAD_PROFILES) == 1
    assert (
        SGLANG_REPRESENTATIVE_WORKLOAD_PROFILES[0].profile_id
        == REPRESENTATIVE_WORKLOAD_PROFILE_ID
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"representative_canary": True},
        {"representative_workload_profile": REPRESENTATIVE_WORKLOAD_PROFILE_ID},
    ],
)
def test_databricks_sglang_representative_flag_and_profile_are_required_together(
    kwargs,
):
    with pytest.raises(ValueError, match="must be provided together"):
        DatabricksSGLangSmokeJobConfig(
            benchmark_id="unpaired-sglang-canary",
            output_dir="/Volumes/catalog/schema/volume/unpaired-sglang-canary",
            runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
            node_type_id="g6.8xlarge",
            single_user_name=SINGLE_USER_NAME,
            baseline_only=True,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("override", "field_name"),
    [
        ({"context_length": 8192}, "context_length"),
        ({"max_tokens": 64}, "max_tokens"),
        ({"live_benchmark_repeats": 1}, "live_benchmark_repeats"),
        ({"sglang_attention_backend": "fa3"}, "sglang_attention_backend"),
        ({"sglang_sampling_backend": "flashinfer"}, "sglang_sampling_backend"),
        (
            {"sglang_enable_deterministic_inference": False},
            "sglang_enable_deterministic_inference",
        ),
    ],
)
def test_databricks_sglang_representative_profile_rejects_arbitrary_values(
    override,
    field_name,
):
    revision = "a" * 40
    kwargs = {
        "benchmark_id": "mismatched-sglang-canary",
        "output_dir": "/Volumes/catalog/schema/volume/mismatched-sglang-canary",
        "runner_python_file": "dbfs:/benchmarks/run_sglang_smoke.py",
        "node_type_id": "g6.8xlarge",
        "single_user_name": SINGLE_USER_NAME,
        "model_revision": revision,
        "tokenizer_revision": revision,
        "generate_live_handoff": True,
        **REPRESENTATIVE_WORKLOAD_KWARGS,
        **override,
    }

    with pytest.raises(ValueError, match=field_name):
        DatabricksSGLangSmokeJobConfig(**kwargs)


@pytest.mark.parametrize(
    ("hardware_target", "node_type_id"),
    [
        ("aws-g6-l4", "g6.4xlarge"),
        ("aws-g5-a10g", "g5.12xlarge"),
    ],
)
def test_databricks_sglang_representative_canary_requires_exact_node_size(
    hardware_target,
    node_type_id,
):
    revision = "a" * 40
    with pytest.raises(ValueError, match="exact V1 node type"):
        DatabricksSGLangSmokeJobConfig(
            benchmark_id="wrong-size-sglang-canary",
            output_dir="/Volumes/catalog/schema/volume/wrong-size-sglang-canary",
            runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
            hardware_target=hardware_target,
            node_type_id=node_type_id,
            single_user_name=SINGLE_USER_NAME,
            generate_live_handoff=True,
            model_revision=revision,
            tokenizer_revision=revision,
            **REPRESENTATIVE_WORKLOAD_KWARGS,
        )


@pytest.mark.parametrize(
    ("hardware_target", "node_type_id"),
    [
        ("aws-g6-l4", "g6.4xlarge"),
        ("aws-g5-a10g", "g5.12xlarge"),
    ],
)
def test_databricks_sglang_debug_job_preserves_family_node_overrides(
    hardware_target,
    node_type_id,
):
    config = DatabricksSGLangSmokeJobConfig(
        benchmark_id="debug-sglang-node-override",
        output_dir="/Volumes/catalog/schema/volume/debug-sglang-node-override",
        runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
        hardware_target=hardware_target,
        node_type_id=node_type_id,
        single_user_name=SINGLE_USER_NAME,
        baseline_only=True,
    )

    assert config.node_type_id == node_type_id


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"run_timeout_seconds": 0}, "run_timeout_seconds"),
        ({"run_timeout_seconds": 14401}, "run_timeout_seconds"),
        ({"task_max_retries": 1}, "task_max_retries"),
    ],
)
def test_databricks_sglang_submission_bounds_cluster_runtime(overrides, message):
    with pytest.raises(ValueError, match=message):
        DatabricksSGLangSmokeJobConfig(
            benchmark_id="bounded-sglang-canary",
            output_dir="/Volumes/catalog/schema/volume/bounded-sglang-canary",
            runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
            node_type_id="g6.8xlarge",
            single_user_name=SINGLE_USER_NAME,
            baseline_only=True,
            **overrides,
        )


def test_databricks_sglang_representative_canary_requires_exact_timeout():
    revision = "a" * 40
    with pytest.raises(ValueError, match="exactly 14400"):
        DatabricksSGLangSmokeJobConfig(
            benchmark_id="short-representative-sglang-canary",
            output_dir=(
                "/Volumes/catalog/schema/volume/short-representative-sglang-canary"
            ),
            runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
            node_type_id="g6.8xlarge",
            single_user_name=SINGLE_USER_NAME,
            model_revision=revision,
            tokenizer_revision=revision,
            generate_live_handoff=True,
            run_timeout_seconds=3600,
            **REPRESENTATIVE_WORKLOAD_KWARGS,
        )


def test_build_databricks_sglang_smoke_payload_uses_single_node_g6_cluster():
    config = DatabricksSGLangSmokeJobConfig(
        benchmark_id="v1-sglang-smoke-001",
        output_dir="/Volumes/catalog/schema/volume/v1-sglang-smoke",
        runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
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
        context_length=8192,
        mem_fraction_static=0.72,
        stream=False,
        baseline_only=True,
        hicache_page_store_uri="/local_disk0/cachet/sglang-hicache",
        hicache_size_gb=4,
        custom_tags={"team": "document-kv"},
    )

    payload = build_databricks_sglang_smoke_run_submit_payload(config)
    task = payload["tasks"][0]
    cluster = task["new_cluster"]

    assert payload["run_name"] == DEFAULT_DATABRICKS_SGLANG_SMOKE_RUN_NAME
    assert payload["timeout_seconds"] == 14400
    assert task["task_key"] == DEFAULT_DATABRICKS_SGLANG_SMOKE_TASK_KEY
    assert task["timeout_seconds"] == 14400
    assert task["max_retries"] == 0
    assert "libraries" not in task
    assert cluster["node_type_id"] == "g6.8xlarge"
    assert cluster["driver_node_type_id"] == "g6.8xlarge"
    assert cluster["data_security_mode"] == "SINGLE_USER"
    assert cluster["single_user_name"] == SINGLE_USER_NAME
    assert cluster["num_workers"] == 0
    assert cluster["custom_tags"]["ResourceClass"] == "SingleNode"
    assert cluster["custom_tags"]["purpose"] == DEFAULT_DATABRICKS_SGLANG_SMOKE_PURPOSE
    assert cluster["custom_tags"]["team"] == "document-kv"
    assert task["spark_python_task"] == {
        "python_file": "dbfs:/benchmarks/run_sglang_smoke.py",
        "parameters": [
            "--benchmark-id",
            "v1-sglang-smoke-001",
            "--output-dir",
            "/Volumes/catalog/schema/volume/v1-sglang-smoke",
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
            "--context-length",
            "8192",
            "--mem-fraction-static",
            "0.72",
            "--hardware-target",
            "aws-g6-l4",
            "--cache-prompt-text-mode",
            "logical",
            "--live-check-prompt-format",
            DEFAULT_SGLANG_LIVE_CHECK_PROMPT_FORMAT,
            "--live-check-request-mode",
            DEFAULT_SGLANG_LIVE_CHECK_REQUEST_MODE,
            "--live-check-temperature",
            str(DEFAULT_SGLANG_LIVE_CHECK_TEMPERATURE),
            "--flush-cache-timeout-seconds",
            str(DEFAULT_SGLANG_FLUSH_CACHE_TIMEOUT_SECONDS),
            "--no-stream",
            "--baseline-only",
            "--hicache-page-store-uri",
            "/local_disk0/cachet/sglang-hicache",
            "--hicache-size-gb",
            "4",
            "--hicache-storage-prefetch-policy",
            DEFAULT_SGLANG_HICACHE_STORAGE_PREFETCH_POLICY,
            "--hicache-storage-prefetch-threshold",
            str(DEFAULT_SGLANG_HICACHE_STORAGE_PREFETCH_THRESHOLD),
            "--package-wheel-uri",
            WHEEL_URI,
        ],
    }


def test_databricks_sglang_smoke_config_requires_handoff_and_page_keys_for_cache_arm():
    try:
        DatabricksSGLangSmokeJobConfig(
            benchmark_id="v1-sglang-smoke-001",
            output_dir="/Volumes/catalog/schema/volume/v1-sglang-smoke",
            runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
            single_user_name=SINGLE_USER_NAME,
        )
    except ValueError as exc:
        assert str(exc) == SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE
    else:
        raise AssertionError("expected missing handoff validation to fail")

    try:
        DatabricksSGLangSmokeJobConfig(
            benchmark_id="v1-sglang-smoke-001",
            output_dir="/Volumes/catalog/schema/volume/v1-sglang-smoke",
            runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
            single_user_name=SINGLE_USER_NAME,
            handoff_json=HANDOFF_JSON,
        )
    except ValueError as exc:
        assert str(exc) == SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE
    else:
        raise AssertionError("expected missing page-key validation to fail")

    config = DatabricksSGLangSmokeJobConfig(
        benchmark_id="v1-sglang-smoke-001",
        output_dir="/Volumes/catalog/schema/volume/v1-sglang-smoke",
        runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
        single_user_name=SINGLE_USER_NAME,
        handoff_json=HANDOFF_JSON,
        payload_uri="/Volumes/catalog/schema/volume/live/sglang-live.kv",
        request_id="cachet-live-sglang-1",
        sglang_hicache_page_keys_json=PAGE_KEYS_JSON,
    )

    parameters = build_databricks_sglang_smoke_run_submit_payload(config)["tasks"][0][
        "spark_python_task"
    ]["parameters"]
    assert "--baseline-only" not in parameters
    assert parameters[parameters.index("--handoff-json") + 1] == HANDOFF_JSON
    assert (
        parameters[parameters.index("--sglang-hicache-page-keys-json") + 1]
        == PAGE_KEYS_JSON
    )

    config = DatabricksSGLangSmokeJobConfig(
        benchmark_id="v1-sglang-baseline-001",
        output_dir="/Volumes/catalog/schema/volume/v1-sglang-baseline",
        runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
        single_user_name=SINGLE_USER_NAME,
        baseline_only=True,
    )

    parameters = build_databricks_sglang_smoke_run_submit_payload(config)["tasks"][0][
        "spark_python_task"
    ]["parameters"]
    assert "--baseline-only" in parameters
    assert "--handoff-json" not in parameters


def test_databricks_sglang_smoke_config_supports_generated_live_handoff_cache_arm():
    default_config = DatabricksSGLangSmokeJobConfig(
        benchmark_id="v1-sglang-generated-defaults",
        output_dir="/Volumes/catalog/schema/volume/v1-sglang-generated",
        runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
        single_user_name=SINGLE_USER_NAME,
        generate_live_handoff=True,
    )
    assert default_config.cache_prompt_text_mode == "logical"
    assert (
        default_config.live_check_request_mode == DEFAULT_SGLANG_LIVE_CHECK_REQUEST_MODE
    )
    assert default_config.live_check_temperature == DEFAULT_SGLANG_LIVE_CHECK_TEMPERATURE
    assert (
        default_config.live_handoff_generator_factory
        == DEFAULT_SGLANG_LIVE_HANDOFF_GENERATOR_FACTORY
    )
    assert default_config.sglang_hicache_page_size == DEFAULT_SGLANG_HICACHE_PAGE_SIZE
    assert (
        default_config.hicache_storage_prefetch_policy
        == DEFAULT_SGLANG_HICACHE_STORAGE_PREFETCH_POLICY
    )
    assert (
        default_config.hicache_storage_prefetch_threshold
        == DEFAULT_SGLANG_HICACHE_STORAGE_PREFETCH_THRESHOLD
    )
    assert default_config.sglang_attention_backend is None
    assert default_config.sglang_sampling_backend is None
    assert default_config.sglang_enable_deterministic_inference is False
    assert (
        default_config.flush_cache_before_cache_arm
        is DEFAULT_SGLANG_FLUSH_CACHE_BEFORE_CACHE_ARM
    )
    assert (
        default_config.flush_cache_before_canary
        is DEFAULT_SGLANG_FLUSH_CACHE_BEFORE_CANARY
    )
    assert (
        default_config.flush_cache_timeout_seconds
        == DEFAULT_SGLANG_FLUSH_CACHE_TIMEOUT_SECONDS
    )
    assert (
        default_config.live_benchmark_repeats
        == DEFAULT_SGLANG_LIVE_BENCHMARK_REPEATS
    )

    config = DatabricksSGLangSmokeJobConfig(
        benchmark_id="v1-sglang-generated-001",
        output_dir="/Volumes/catalog/schema/volume/v1-sglang-generated",
        runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
        node_type_id="g6.8xlarge",
        single_user_name=SINGLE_USER_NAME,
        generate_live_handoff=True,
        live_handoff_output_dir="/Volumes/catalog/schema/volume/v1-sglang-generated/live-handoff",
        live_handoff_generator_factory="module:factory",
        live_check_temperature=0.25,
        live_check_extra_body_json='{"reasoning_effort":"none"}',
        live_handoff_dtype="float16",
        live_handoff_align_bytes=8,
        sglang_hicache_page_size=2,
        live_handoff_generation_timeout_seconds=12.5,
        sglang_attention_backend="triton",
        sglang_sampling_backend="pytorch",
        sglang_enable_deterministic_inference=True,
        flush_cache_before_cache_arm=False,
        flush_cache_before_canary=False,
        flush_cache_timeout_seconds=12.5,
        live_benchmark_repeats=3,
        spark_env_vars={"CACHET_TRANSFORMERS_DEVICE": "cuda"},
    )

    payload = build_databricks_sglang_smoke_run_submit_payload(config)
    cluster = payload["tasks"][0]["new_cluster"]
    parameters = payload["tasks"][0]["spark_python_task"]["parameters"]

    assert cluster["node_type_id"] == "g6.8xlarge"
    assert cluster["spark_env_vars"] == {"CACHET_TRANSFORMERS_DEVICE": "cuda"}
    assert "--baseline-only" not in parameters
    assert "--handoff-json" not in parameters
    assert "--sglang-hicache-page-keys-json" not in parameters
    assert "--generate-live-handoff" in parameters
    assert (
        parameters[parameters.index("--live-check-request-mode") + 1]
        == DEFAULT_SGLANG_LIVE_CHECK_REQUEST_MODE
    )
    assert parameters[parameters.index("--live-check-temperature") + 1] == "0.25"
    assert (
        parameters[parameters.index("--live-check-extra-body-json") + 1]
        == '{"reasoning_effort":"none"}'
    )
    assert parameters[parameters.index("--sglang-attention-backend") + 1] == "triton"
    assert parameters[parameters.index("--sglang-sampling-backend") + 1] == "pytorch"
    assert "--sglang-enable-deterministic-inference" in parameters
    assert "--no-flush-cache-before-cache-arm" in parameters
    assert "--no-flush-cache-before-canary" in parameters
    assert parameters[parameters.index("--flush-cache-timeout-seconds") + 1] == "12.5"
    assert parameters[parameters.index("--live-benchmark-repeats") + 1] == "3"
    assert parameters[parameters.index("--live-handoff-output-dir") + 1].endswith(
        "/live-handoff"
    )
    assert (
        parameters[parameters.index("--live-handoff-generator-factory") + 1]
        == "module:factory"
    )
    assert parameters[parameters.index("--live-handoff-dtype") + 1] == "float16"
    assert parameters[parameters.index("--live-handoff-align-bytes") + 1] == "8"
    assert parameters[parameters.index("--sglang-hicache-page-size") + 1] == "2"
    assert (
        parameters[parameters.index("--live-handoff-generation-timeout-seconds") + 1]
        == "12.5"
    )
    assert parameters[parameters.index("--hicache-storage-prefetch-policy") + 1] == (
        DEFAULT_SGLANG_HICACHE_STORAGE_PREFETCH_POLICY
    )
    assert parameters[
        parameters.index("--hicache-storage-prefetch-threshold") + 1
    ] == str(DEFAULT_SGLANG_HICACHE_STORAGE_PREFETCH_THRESHOLD)


def test_databricks_sglang_smoke_config_supports_prepared_v1_datasets():
    default_config = DatabricksSGLangSmokeJobConfig(
        benchmark_id="v1-sglang-prepared-defaults",
        output_dir="/Volumes/catalog/schema/volume/v1-sglang-prepared",
        runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
        single_user_name=SINGLE_USER_NAME,
        dataset_specs=DATASET_SPECS,
        live_benchmark_repeats=1,
    )
    default_parameters = build_databricks_sglang_smoke_run_submit_payload(
        default_config
    )["tasks"][0]["spark_python_task"]["parameters"]

    assert (
        default_config.sglang_hicache_page_size
        == DEFAULT_SGLANG_PREPARED_HICACHE_PAGE_SIZE
    )
    assert default_parameters[
        default_parameters.index("--sglang-hicache-page-size") + 1
    ] == str(DEFAULT_SGLANG_PREPARED_HICACHE_PAGE_SIZE)

    config = DatabricksSGLangSmokeJobConfig(
        benchmark_id="v1-sglang-prepared",
        output_dir="/Volumes/catalog/schema/volume/v1-sglang-prepared",
        runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
        single_user_name=SINGLE_USER_NAME,
        dataset_specs=DATASET_SPECS,
        live_benchmark_repeats=1,
        sglang_hicache_page_size=2,
    )

    parameters = build_databricks_sglang_smoke_run_submit_payload(config)["tasks"][0][
        "spark_python_task"
    ]["parameters"]

    assert "--baseline-only" not in parameters
    assert "--generate-live-handoff" not in parameters
    assert "--handoff-json" not in parameters
    assert parameters[parameters.index("--live-benchmark-repeats") + 1] == "1"
    assert parameters[parameters.index("--sglang-hicache-page-size") + 1] == "2"
    dataset_positions = [
        index
        for index, value in enumerate(parameters)
        if value == "--dataset"
    ]
    assert [parameters[index + 1] for index in dataset_positions] == list(DATASET_SPECS)

    generated_config = DatabricksSGLangSmokeJobConfig(
        benchmark_id="v1-sglang-prepared-generated",
        output_dir="/Volumes/catalog/schema/volume/v1-sglang-prepared-generated",
        runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
        single_user_name=SINGLE_USER_NAME,
        dataset_specs=DATASET_SPECS,
        live_benchmark_repeats=1,
        benchmark_handoff_generator_factory="module:factory",
        benchmark_handoff_output_dir=(
            "/Volumes/catalog/schema/volume/v1-sglang-prepared-generated/handoffs"
        ),
        benchmark_handoff_dtype="float16",
        benchmark_handoff_align_bytes=8,
        benchmark_handoff_generation_timeout_seconds=12.5,
    )

    generated_parameters = build_databricks_sglang_smoke_run_submit_payload(
        generated_config
    )["tasks"][0]["spark_python_task"]["parameters"]

    assert generated_config.sglang_hicache_page_size == (
        DEFAULT_SGLANG_PREPARED_HICACHE_PAGE_SIZE
    )
    assert generated_parameters[
        generated_parameters.index("--sglang-hicache-page-size") + 1
    ] == str(DEFAULT_SGLANG_PREPARED_HICACHE_PAGE_SIZE)
    assert (
        generated_parameters[
            generated_parameters.index("--benchmark-handoff-generator-factory") + 1
        ]
        == "module:factory"
    )
    assert generated_parameters[
        generated_parameters.index("--benchmark-handoff-output-dir") + 1
    ].endswith("/handoffs")
    assert (
        generated_parameters[generated_parameters.index("--benchmark-handoff-dtype") + 1]
        == "float16"
    )
    assert (
        generated_parameters[
            generated_parameters.index("--benchmark-handoff-align-bytes") + 1
        ]
        == "8"
    )
    assert (
        generated_parameters[
            generated_parameters.index(
                "--benchmark-handoff-generation-timeout-seconds"
            )
            + 1
        ]
        == "12.5"
    )


def test_databricks_sglang_smoke_config_validates_cluster_and_runtime_fields():
    invalid_cases = [
        (
            {"context_length": 0, "baseline_only": True},
            "context_length must be positive",
        ),
        (
            {"mem_fraction_static": 0, "baseline_only": True},
            "mem_fraction_static must be in",
        ),
        (
            {"cache_prompt_text_mode": "full", "baseline_only": True},
            "cache_prompt_text_mode",
        ),
        (
            {"live_check_request_mode": "responses", "baseline_only": True},
            "live_check_request_mode",
        ),
        (
            {"live_check_temperature": True, "baseline_only": True},
            "live_check_temperature",
        ),
        (
            {"live_check_extra_body_json": "[]", "baseline_only": True},
            "live_check_extra_body_json must decode",
        ),
        (
            {"live_check_extra_body_json": "{", "baseline_only": True},
            "live_check_extra_body_json must decode",
        ),
        (
            {"flush_cache_before_cache_arm": "yes", "baseline_only": True},
            "flush_cache_before_cache_arm",
        ),
        (
            {"flush_cache_before_canary": "yes", "baseline_only": True},
            "flush_cache_before_canary",
        ),
        (
            {"flush_cache_timeout_seconds": 0, "baseline_only": True},
            "flush_cache_timeout_seconds",
        ),
        (
            {"live_benchmark_repeats": -1, "baseline_only": True},
            "live_benchmark_repeats",
        ),
        (
            {"live_benchmark_repeats": 1, "baseline_only": True},
            "live_benchmark_repeats",
        ),
        (
            {
                "dataset_specs": DATASET_SPECS,
                "baseline_only": True,
            },
            "dataset specs require cache-arm SGLang live benchmark",
        ),
        (
            {"dataset_specs": DATASET_SPECS, "baseline_only": False},
            "dataset specs require live_benchmark_repeats",
        ),
        (
            {
                "dataset_specs": DATASET_SPECS,
                "generate_live_handoff": True,
                "live_benchmark_repeats": 1,
                "baseline_only": False,
            },
            "prepared SGLang benchmark datasets must not be combined",
        ),
        (
            {
                "benchmark_handoff_generator_factory": "module:factory",
                "baseline_only": False,
            },
            "requires prepared dataset specs",
        ),
        (
            {
                "benchmark_handoff_output_dir": "/Volumes/catalog/schema/volume/handoffs",
                "baseline_only": False,
            },
            "requires benchmark_handoff_generator_factory",
        ),
        (
            {
                "dataset_specs": DATASET_SPECS,
                "live_benchmark_repeats": 1,
                "benchmark_handoff_generator_factory": "",
                "baseline_only": False,
            },
            "benchmark_handoff_generator_factory",
        ),
        (
            {
                "dataset_specs": DATASET_SPECS,
                "live_benchmark_repeats": 1,
                "benchmark_handoff_generator_factory": "module:factory",
                "benchmark_handoff_align_bytes": 0,
                "baseline_only": False,
            },
            "benchmark_handoff_align_bytes",
        ),
        (
            {
                "dataset_specs": ("biography",),
                "live_benchmark_repeats": 1,
                "baseline_only": False,
            },
            "dataset specs must use DATASET=JSONL_PATH syntax",
        ),
        (
            {
                "dataset_specs": (DATASET_SPECS[0],),
                "live_benchmark_repeats": 1,
                "baseline_only": False,
            },
            "dataset specs missing required V1 datasets",
        ),
        (
            {
                "handoff_json": HANDOFF_JSON,
                "handoff_record_json": "{}",
                "baseline_only": True,
            },
            "only one of handoff_json",
        ),
        (
            {"handoff_record_json": "[]", "baseline_only": True},
            "handoff_record_json must decode",
        ),
        (
            {"sglang_hicache_page_keys_json": PAGE_KEYS_JSON, "baseline_only": True},
            SGLANG_BASELINE_HANDOFF_FIELDS_UNSUPPORTED_MESSAGE,
        ),
        (
            {"sglang_hicache_page_keys_json": "[]", "baseline_only": True},
            SGLANG_BASELINE_HANDOFF_FIELDS_UNSUPPORTED_MESSAGE,
        ),
        (
            {"sglang_hicache_page_keys_json": '"page-a"', "baseline_only": False},
            "sglang_hicache_page_keys_json must decode",
        ),
        (
            {"sglang_hicache_page_keys_json": PAGE_KEYS_JSON, "baseline_only": False},
            SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE,
        ),
        (
            {
                "handoff_json": HANDOFF_JSON,
                "sglang_hicache_page_keys_json": "[]",
                "baseline_only": False,
            },
            SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE,
        ),
        (
            {"handoff_json": HANDOFF_JSON, "baseline_only": True},
            SGLANG_BASELINE_HANDOFF_FIELDS_UNSUPPORTED_MESSAGE,
        ),
        (
            {"generate_live_handoff": True, "baseline_only": True},
            SGLANG_BASELINE_HANDOFF_FIELDS_UNSUPPORTED_MESSAGE,
        ),
        (
            {
                "generate_live_handoff": True,
                "handoff_json": HANDOFF_JSON,
                "baseline_only": False,
            },
            SGLANG_GENERATED_HANDOFF_EXPLICIT_FIELDS_UNSUPPORTED_MESSAGE,
        ),
        (
            {
                "generate_live_handoff": True,
                "live_handoff_generator_factory": "",
                "baseline_only": False,
            },
            "factory",
        ),
        (
            {
                "generate_live_handoff": True,
                "live_handoff_align_bytes": 0,
                "baseline_only": False,
            },
            "align",
        ),
        (
            {
                "generate_live_handoff": True,
                "sglang_hicache_page_size": 0,
                "baseline_only": False,
            },
            "page_size",
        ),
        (
            {"hicache_storage_prefetch_threshold": 0, "baseline_only": True},
            "hicache_storage_prefetch_threshold",
        ),
        (
            {"sglang_attention_backend": "flash-attention", "baseline_only": True},
            "sglang_attention_backend",
        ),
        (
            {"sglang_sampling_backend": "flash-attention", "baseline_only": True},
            "sglang_sampling_backend",
        ),
        (
            {"sglang_enable_deterministic_inference": True, "baseline_only": True},
            "sglang_attention_backend",
        ),
        (
            {
                "sglang_attention_backend": "flashinfer",
                "sglang_enable_deterministic_inference": True,
                "baseline_only": True,
            },
            "sglang_attention_backend",
        ),
        (
            {
                "sglang_attention_backend": "triton",
                "sglang_sampling_backend": "flashinfer",
                "sglang_enable_deterministic_inference": True,
                "baseline_only": True,
            },
            "sglang_sampling_backend",
        ),
        (
            {"spark_env_vars": {"DATABRICKS_TOKEN": "redacted"}, "baseline_only": True},
            "looks secret-bearing",
        ),
    ]

    for overrides, message in invalid_cases:
        kwargs = {
            "benchmark_id": "v1-sglang-smoke-001",
            "output_dir": "/Volumes/catalog/schema/volume/v1-sglang-smoke",
            "runner_python_file": "dbfs:/benchmarks/run_sglang_smoke.py",
            "single_user_name": SINGLE_USER_NAME,
            "baseline_only": True,
        }
        kwargs.update(overrides)
        try:
            DatabricksSGLangSmokeJobConfig(**kwargs)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"expected validation to fail for {overrides!r}")


def test_write_databricks_sglang_smoke_runner_script_imports_smoke_main(tmp_path):
    path = tmp_path / "run_sglang_smoke.py"

    write_databricks_sglang_smoke_runner_script(path)

    runner_text = path.read_text(encoding="utf-8")
    assert "--package-wheel-uri" in runner_text
    assert "DOCUMENT_KV_PACKAGE_INSTALL_SPEC" in runner_text
    assert 'pip", "install"' in runner_text
    assert "dbfs:/" in runner_text
    assert "document_kv_cache.sglang_smoke" in runner_text
    assert "if exit_code:" in runner_text


def test_representative_sglang_runner_digest_matches_embedded_script():
    assert (
        hashlib.sha256(SGLANG_SMOKE_RUNNER_SCRIPT.encode("utf-8")).hexdigest()
        == REPRESENTATIVE_SGLANG_RUNNER_SHA256
    )


@pytest.mark.parametrize("verify_sha256", [True, False])
def test_generated_sglang_smoke_runner_installs_wheel_before_forwarding_args(
    tmp_path,
    verify_sha256,
):
    runner_path = tmp_path / "run_sglang_smoke.py"
    pip_call_path = tmp_path / "pip-call.json"
    main_args_path = tmp_path / "main-args.json"
    events_path = tmp_path / "events.jsonl"
    wheel_path = tmp_path / "cachet_kv-0.2.0-py3-none-any.whl"
    wheel_path.write_bytes(b"verified Cachet wheel bytes")
    wheel_sha256 = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    package_dir = tmp_path / "document_kv_cache"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "sglang_smoke.py").write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "",
                "with open(os.environ['RUNNER_EVENTS_JSONL'], 'a', encoding='utf-8') as handle:",
                "    handle.write(json.dumps({'event': 'sglang_smoke_import'}) + '\\n')",
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

    write_databricks_sglang_smoke_runner_script(runner_path)
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "PIP_CALL_JSON": str(pip_call_path),
        "MAIN_ARGS_JSON": str(main_args_path),
        "RUNNER_EVENTS_JSONL": str(events_path),
        "DOCUMENT_KV_PACKAGE_WHEEL_SHA256": "a" * 64,
    }

    wheel_arguments = ["--package-wheel-uri", str(wheel_path)]
    if verify_sha256:
        wheel_arguments.extend(["--package-wheel-sha256", wheel_sha256])
    subprocess.run(
        [
            sys.executable,
            str(runner_path),
            *wheel_arguments,
            "--benchmark-id",
            "v1-sglang-smoke-001",
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
            "v1-sglang-smoke-001",
            "--output-dir",
            "/dbfs/tmp/cachet/output",
        ],
        "package_install_spec": str(wheel_path),
        "package_wheel_sha256": wheel_sha256 if verify_sha256 else None,
    }
    events = [
        json.loads(line)["event"]
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events == ["pip_install", "sglang_smoke_import", "main"]


def test_generated_sglang_smoke_runner_rejects_tampered_wheel(tmp_path):
    runner_path = tmp_path / "run_sglang_smoke.py"
    wheel_path = tmp_path / "cachet_kv-0.2.0-py3-none-any.whl"
    wheel_path.write_bytes(b"tampered Cachet wheel bytes")
    write_databricks_sglang_smoke_runner_script(runner_path)

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


def test_write_databricks_sglang_smoke_run_submit_json_writes_payload(tmp_path):
    path = tmp_path / "payload.json"

    write_databricks_sglang_smoke_run_submit_json(
        DatabricksSGLangSmokeJobConfig(
            benchmark_id="v1-sglang-smoke-001",
            output_dir="/Volumes/catalog/schema/volume/v1-sglang-smoke",
            runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
            single_user_name=SINGLE_USER_NAME,
            baseline_only=True,
        ),
        path,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["tasks"][0]["task_key"] == DEFAULT_DATABRICKS_SGLANG_SMOKE_TASK_KEY


def test_main_writes_sglang_smoke_payload_and_runner_script(tmp_path):
    payload_path = tmp_path / "payload.json"
    runner_path = tmp_path / "run_sglang_smoke.py"

    exit_code = main(
        [
            "--benchmark-id",
            "v1-sglang-smoke-001",
            "--output-dir",
            "/Volumes/catalog/schema/volume/v1-sglang-smoke",
            "--runner-python-file",
            "dbfs:/benchmarks/run_sglang_smoke.py",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--wheel-uri",
            WHEEL_URI,
            "--run-timeout-seconds",
            "3600",
            "--task-max-retries",
            "0",
            "--baseline-only",
            "--live-check-temperature",
            "0.25",
            "--sglang-attention-backend",
            "triton",
            "--sglang-sampling-backend",
            "pytorch",
            "--sglang-enable-deterministic-inference",
            "--spark-env-var",
            "CACHET_SGLANG_TRACE=1",
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
    assert task["spark_python_task"]["parameters"][-2:] == [
        "--package-wheel-uri",
        WHEEL_URI,
    ]
    parameters = task["spark_python_task"]["parameters"]
    assert parameters[parameters.index("--live-check-temperature") + 1] == "0.25"
    assert parameters[parameters.index("--flush-cache-timeout-seconds") + 1] == str(
        DEFAULT_SGLANG_FLUSH_CACHE_TIMEOUT_SECONDS
    )
    assert parameters[parameters.index("--sglang-attention-backend") + 1] == "triton"
    assert parameters[parameters.index("--sglang-sampling-backend") + 1] == "pytorch"
    assert "--sglang-enable-deterministic-inference" in parameters
    assert task["new_cluster"]["spark_env_vars"] == {"CACHET_SGLANG_TRACE": "1"}
    assert "sglang_smoke" in runner_path.read_text(encoding="utf-8")


def test_main_propagates_explicit_representative_workload_profile(tmp_path):
    revision = "a" * 40
    payload_path = tmp_path / "representative-payload.json"

    exit_code = main(
        [
            "--benchmark-id",
            "representative-sglang-canary",
            "--output-dir",
            "/Volumes/catalog/schema/volume/representative-sglang-canary",
            "--runner-python-file",
            "dbfs:/benchmarks/run_sglang_smoke.py",
            "--run-timeout-seconds",
            "14400",
            "--node-type-id",
            "g6.8xlarge",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--wheel-uri",
            REPRESENTATIVE_WHEEL_URI,
            "--wheel-sha256",
            REPRESENTATIVE_WHEEL_SHA256,
            "--model-revision",
            revision,
            "--tokenizer-revision",
            revision,
            "--representative-canary",
            "--representative-workload-profile",
            REPRESENTATIVE_WORKLOAD_PROFILE_ID,
            "--context-length",
            "4096",
            "--max-tokens",
            "32",
            "--live-benchmark-repeats",
            "2",
            "--sglang-attention-backend",
            "triton",
            "--sglang-sampling-backend",
            "pytorch",
            "--sglang-enable-deterministic-inference",
            "--generate-live-handoff",
            "--output-json",
            str(payload_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    task = payload["tasks"][0]
    parameters = task["spark_python_task"]["parameters"]
    assert payload["timeout_seconds"] == 14400
    assert task["timeout_seconds"] == 14400
    assert "--representative-canary" in parameters
    assert (
        parameters[parameters.index("--representative-workload-profile") + 1]
        == REPRESENTATIVE_WORKLOAD_PROFILE_ID
    )
    assert parameters[parameters.index("--live-benchmark-repeats") + 1] == "2"
    assert payload["tasks"][0]["new_cluster"]["spark_env_vars"][
        "DOCUMENT_KV_EVICT_PAGE_CACHE"
    ] == "1"


def test_main_writes_prepared_v1_dataset_parameters(tmp_path):
    payload_path = tmp_path / "payload.json"

    exit_code = main(
        [
            "--benchmark-id",
            "v1-sglang-prepared",
            "--output-dir",
            "/Volumes/catalog/schema/volume/v1-sglang-prepared",
            "--runner-python-file",
            "dbfs:/benchmarks/run_sglang_smoke.py",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--wheel-uri",
            WHEEL_URI,
            "--live-benchmark-repeats",
            "1",
            *[
                item
                for spec in DATASET_SPECS
                for item in ("--dataset", spec)
            ],
            "--output-json",
            str(payload_path),
        ]
    )

    assert exit_code == 0
    parameters = json.loads(payload_path.read_text(encoding="utf-8"))["tasks"][0][
        "spark_python_task"
    ]["parameters"]
    dataset_positions = [
        index
        for index, value in enumerate(parameters)
        if value == "--dataset"
    ]
    assert [parameters[index + 1] for index in dataset_positions] == list(DATASET_SPECS)
    assert parameters[parameters.index("--live-benchmark-repeats") + 1] == "1"
    assert parameters[parameters.index("--sglang-hicache-page-size") + 1] == str(
        DEFAULT_SGLANG_PREPARED_HICACHE_PAGE_SIZE
    )
    assert "--baseline-only" not in parameters
    assert "--generate-live-handoff" not in parameters


def test_main_derives_sglang_smoke_node_type_from_g5_hardware_target(tmp_path):
    payload_path = tmp_path / "payload.json"

    exit_code = main(
        [
            "--benchmark-id",
            "v1-sglang-smoke-001",
            "--output-dir",
            "/Volumes/catalog/schema/volume/v1-sglang-smoke",
            "--runner-python-file",
            "dbfs:/benchmarks/run_sglang_smoke.py",
            "--hardware-target",
            "aws-g5-a10g",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--baseline-only",
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
