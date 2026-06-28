import json
import os
from pathlib import Path
import subprocess
import sys

import document_kv_cache.databricks_vllm_smoke_job as public_vllm_smoke_job
from document_kv_cache.databricks_vllm_smoke_job import (
    DEFAULT_DATABRICKS_VLLM_SMOKE_PURPOSE,
    DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME,
    DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY,
    DatabricksVLLMSmokeJobConfig,
    build_databricks_vllm_smoke_run_submit_payload,
    main,
    write_databricks_vllm_smoke_run_submit_json,
    write_databricks_vllm_smoke_runner_script,
)


WHEEL_URI = "/Volumes/catalog/schema/volume/wheels/cachet_kv-0.2.0-py3-none-any.whl"
SINGLE_USER_NAME = "user@example.com"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_SPECS = tuple(
    f"{dataset}=/Volumes/catalog/schema/volume/v1/{dataset}.jsonl"
    for dataset in ("biography", "hotpotqa", "musique", "niah")
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
    assert task["task_key"] == DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY
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
    assert task["new_cluster"]["spark_env_vars"] == {
        "CACHET_TRANSFORMERS_DEVICE": "cuda",
        "CACHET_TRANSFORMERS_TORCH_DTYPE": "bfloat16",
        "CACHET_TRANSFORMERS_TRUST_REMOTE_CODE": "true",
    }


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


def test_generated_vllm_smoke_runner_installs_wheel_before_forwarding_args(tmp_path):
    runner_path = tmp_path / "run_vllm_smoke.py"
    pip_call_path = tmp_path / "pip-call.json"
    main_args_path = tmp_path / "main-args.json"
    events_path = tmp_path / "events.jsonl"
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
    }

    subprocess.run(
        [
            sys.executable,
            str(runner_path),
            "--package-wheel-uri",
            "dbfs:/tmp/cachet/cachet_kv-0.2.0-py3-none-any.whl",
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
        "/dbfs/tmp/cachet/cachet_kv-0.2.0-py3-none-any.whl",
    ]
    main_payload = json.loads(main_args_path.read_text(encoding="utf-8"))
    assert main_payload == {
        "argv": [
            "--benchmark-id",
            "v1-vllm-smoke-001",
            "--output-dir",
            "/dbfs/tmp/cachet/output",
        ],
        "package_install_spec": "/dbfs/tmp/cachet/cachet_kv-0.2.0-py3-none-any.whl",
    }
    events = [json.loads(line)["event"] for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert events == ["pip_install", "vllm_smoke_import", "main"]


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
            "--spark-env-var",
            "CACHET_TRANSFORMERS_DEVICE=cuda",
            "--output-json",
            str(payload_path),
            "--runner-script-output",
            str(runner_path),
        ]
    )

    assert exit_code == 0
    task = json.loads(payload_path.read_text(encoding="utf-8"))["tasks"][0]
    assert "libraries" not in task
    assert task["spark_python_task"]["parameters"][-2:] == ["--package-wheel-uri", WHEEL_URI]
    assert task["new_cluster"]["spark_env_vars"] == {"CACHET_TRANSFORMERS_DEVICE": "cuda"}
    assert "vllm_smoke" in runner_path.read_text(encoding="utf-8")


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
