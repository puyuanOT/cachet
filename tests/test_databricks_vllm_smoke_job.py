import json
import os
from pathlib import Path
import subprocess
import sys

import document_kv_cache.databricks_vllm_smoke_job as public_vllm_smoke_job
import restaurant_kv_serving.databricks_vllm_smoke_job as legacy_vllm_smoke_job
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


WHEEL_URI = "/Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl"
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
        payload_cache_max_bytes=4096,
        dataset_specs=DATASET_SPECS,
    )

    payload = build_databricks_vllm_smoke_run_submit_payload(config)
    parameters = payload["tasks"][0]["spark_python_task"]["parameters"]

    assert parameters[parameters.index("--payload-cache-max-bytes") + 1] == "4096"
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
        ({"max_num_seqs": 0}, "max_num_seqs must be positive"),
        ({"gpu_memory_utilization": 0}, "gpu_memory_utilization must be in"),
        ({"gpu_memory_utilization": 1.1}, "gpu_memory_utilization must be in"),
        ({"payload_cache_max_bytes": -1}, "payload_cache_max_bytes must be a non-negative integer"),
        ({"dataset_specs": ("biography=/tmp/biography.jsonl",)}, "dataset specs missing required V1 datasets"),
        (
            {"benchmark_handoff_generator_factory": "document_kv_cache.transformers_generator:build"},
            "requires prepared dataset specs",
        ),
        (
            {"benchmark_handoff_output_dir": "/Volumes/catalog/schema/volume/handoffs"},
            "requires benchmark_handoff_generator_factory",
        ),
        ({"benchmark_handoff_dtype": ""}, "benchmark_handoff_dtype must be non-empty"),
        ({"benchmark_handoff_align_bytes": 0}, "benchmark_handoff_align_bytes must be a positive integer"),
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
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"expected validation to fail for {overrides!r}")


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
            "dbfs:/tmp/cachet/document_kv_cache-0.2.0-py3-none-any.whl",
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
        "/dbfs/tmp/cachet/document_kv_cache-0.2.0-py3-none-any.whl",
    ]
    main_payload = json.loads(main_args_path.read_text(encoding="utf-8"))
    assert main_payload == {
        "argv": [
            "--benchmark-id",
            "v1-vllm-smoke-001",
            "--output-dir",
            "/dbfs/tmp/cachet/output",
        ],
        "package_install_spec": "/dbfs/tmp/cachet/document_kv_cache-0.2.0-py3-none-any.whl",
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

    cluster = json.loads(payload_path.read_text(encoding="utf-8"))["tasks"][0]["new_cluster"]
    assert exit_code == 0
    assert cluster["node_type_id"] == "g5.8xlarge"
    assert cluster["driver_node_type_id"] == "g5.8xlarge"


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

    cluster = json.loads(payload_path.read_text(encoding="utf-8"))["tasks"][0]["new_cluster"]
    assert exit_code == 0
    assert cluster["node_type_id"] == "g5.8xlarge"
    assert cluster["driver_node_type_id"] == "g5.8xlarge"


def test_public_vllm_smoke_job_main_respects_document_namespace_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"
    original_legacy_build = legacy_vllm_smoke_job.build_databricks_vllm_smoke_run_submit_payload

    def fake_build(config):
        assert config.benchmark_id == "v1-vllm-smoke-001"
        return {"ok": True, "source": "public-hook"}

    monkeypatch.setattr(public_vllm_smoke_job, "build_databricks_vllm_smoke_run_submit_payload", fake_build)

    exit_code = public_vllm_smoke_job.main(
        [
            "--benchmark-id",
            "v1-vllm-smoke-001",
            "--output-dir",
            "/Volumes/catalog/schema/volume/v1-vllm-smoke",
            "--runner-python-file",
            "dbfs:/benchmarks/run_vllm_smoke.py",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"ok": True, "source": "public-hook"}
    assert legacy_vllm_smoke_job.build_databricks_vllm_smoke_run_submit_payload is original_legacy_build


def test_legacy_vllm_smoke_job_main_respects_legacy_namespace_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"
    original_public_build = public_vllm_smoke_job.build_databricks_vllm_smoke_run_submit_payload

    def fake_build(config):
        assert config.benchmark_id == "v1-vllm-smoke-001"
        return {"ok": True, "source": "legacy-hook"}

    monkeypatch.setattr(legacy_vllm_smoke_job, "build_databricks_vllm_smoke_run_submit_payload", fake_build)

    exit_code = legacy_vllm_smoke_job.main(
        [
            "--benchmark-id",
            "v1-vllm-smoke-001",
            "--output-dir",
            "/Volumes/catalog/schema/volume/v1-vllm-smoke",
            "--runner-python-file",
            "dbfs:/benchmarks/run_vllm_smoke.py",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"ok": True, "source": "legacy-hook"}
    assert public_vllm_smoke_job.build_databricks_vllm_smoke_run_submit_payload is original_public_build


def test_legacy_vllm_smoke_job_ignores_document_namespace_build_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"

    def fake_public_build(config):
        return {"ok": True, "source": "unexpected-public-hook"}

    monkeypatch.setattr(public_vllm_smoke_job, "build_databricks_vllm_smoke_run_submit_payload", fake_public_build)

    exit_code = legacy_vllm_smoke_job.main(
        [
            "--benchmark-id",
            "v1-vllm-smoke-001",
            "--output-dir",
            "/Volumes/catalog/schema/volume/v1-vllm-smoke",
            "--runner-python-file",
            "dbfs:/benchmarks/run_vllm_smoke.py",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload != {"ok": True, "source": "unexpected-public-hook"}
    assert payload["tasks"][0]["task_key"] == DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY


def test_legacy_vllm_smoke_job_ignores_document_namespace_writer_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"
    runner_path = tmp_path / "run_vllm_smoke.py"

    def fake_public_runner_writer(path):
        Path(path).write_text("# unexpected public hook\n", encoding="utf-8")

    monkeypatch.setattr(public_vllm_smoke_job, "write_databricks_vllm_smoke_runner_script", fake_public_runner_writer)

    exit_code = legacy_vllm_smoke_job.main(
        [
            "--benchmark-id",
            "v1-vllm-smoke-001",
            "--output-dir",
            "/Volumes/catalog/schema/volume/v1-vllm-smoke",
            "--runner-python-file",
            "dbfs:/benchmarks/run_vllm_smoke.py",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(output_path),
            "--runner-script-output",
            str(runner_path),
        ]
    )

    assert exit_code == 0
    assert "# unexpected public hook" not in runner_path.read_text(encoding="utf-8")
    assert "document_kv_cache.vllm_smoke" in runner_path.read_text(encoding="utf-8")


def test_legacy_vllm_smoke_job_ignores_document_private_helper_monkeypatch(monkeypatch):
    config = legacy_vllm_smoke_job.DatabricksVLLMSmokeJobConfig(
        benchmark_id="v1-vllm-smoke-001",
        output_dir="/Volumes/catalog/schema/volume/v1-vllm-smoke",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        single_user_name=SINGLE_USER_NAME,
    )

    def fake_public_runner_parameters(config):
        return ["--unexpected-public-private-hook"]

    monkeypatch.setattr(public_vllm_smoke_job, "_runner_parameters", fake_public_runner_parameters)

    payload = legacy_vllm_smoke_job.build_databricks_vllm_smoke_run_submit_payload(config)

    assert payload["tasks"][0]["spark_python_task"]["parameters"] != ["--unexpected-public-private-hook"]
    assert payload["tasks"][0]["spark_python_task"]["parameters"][:2] == [
        "--benchmark-id",
        "v1-vllm-smoke-001",
    ]


def test_legacy_vllm_smoke_job_config_ignores_document_private_helper_monkeypatch(monkeypatch):
    def broken_public_cluster_config(config):
        raise RuntimeError(f"unexpected document private hook for {config.benchmark_id}")

    monkeypatch.setattr(public_vllm_smoke_job, "_cluster_config_from_vllm_smoke_job", broken_public_cluster_config)

    config = legacy_vllm_smoke_job.DatabricksVLLMSmokeJobConfig(
        benchmark_id="v1-vllm-smoke-001",
        output_dir="/Volumes/catalog/schema/volume/v1-vllm-smoke",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        single_user_name=SINGLE_USER_NAME,
    )

    assert config.benchmark_id == "v1-vllm-smoke-001"


def test_legacy_vllm_smoke_job_direct_writer_respects_legacy_build_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"

    def fake_build(config):
        assert config.benchmark_id == "v1-vllm-smoke-001"
        return {"ok": True, "source": "legacy-direct-writer-hook"}

    monkeypatch.setattr(legacy_vllm_smoke_job, "build_databricks_vllm_smoke_run_submit_payload", fake_build)

    legacy_vllm_smoke_job.write_databricks_vllm_smoke_run_submit_json(
        legacy_vllm_smoke_job.DatabricksVLLMSmokeJobConfig(
            benchmark_id="v1-vllm-smoke-001",
            output_dir="/Volumes/catalog/schema/volume/v1-vllm-smoke",
            runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
            single_user_name=SINGLE_USER_NAME,
        ),
        output_path,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "ok": True,
        "source": "legacy-direct-writer-hook",
    }


def test_legacy_vllm_smoke_job_restores_document_hooks_after_error(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"
    original_public_build = public_vllm_smoke_job.build_databricks_vllm_smoke_run_submit_payload

    def broken_build(config):
        raise RuntimeError(f"boom for {config.benchmark_id}")

    monkeypatch.setattr(legacy_vllm_smoke_job, "build_databricks_vllm_smoke_run_submit_payload", broken_build)

    exit_code = legacy_vllm_smoke_job.main(
        [
            "--benchmark-id",
            "v1-vllm-smoke-001",
            "--output-dir",
            "/Volumes/catalog/schema/volume/v1-vllm-smoke",
            "--runner-python-file",
            "dbfs:/benchmarks/run_vllm_smoke.py",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 1
    assert public_vllm_smoke_job.build_databricks_vllm_smoke_run_submit_payload is original_public_build


def test_legacy_vllm_smoke_job_module_execution_shows_help():
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }

    result = subprocess.run(
        [sys.executable, "-m", "restaurant_kv_serving.databricks_vllm_smoke_job", "--help"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "Emit a Databricks runs/submit payload for a V1 AWS single-node GPU vLLM smoke." in result.stdout


def test_legacy_vllm_smoke_job_reexports_document_owned_types():
    assert (
        legacy_vllm_smoke_job.DatabricksVLLMSmokeJobConfig
        is public_vllm_smoke_job.DatabricksVLLMSmokeJobConfig
    )
    assert (
        legacy_vllm_smoke_job.DatabricksVLLMSmokeJobConfig.__module__
        == "document_kv_cache.databricks_vllm_smoke_job"
    )
    assert (
        legacy_vllm_smoke_job.DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE
        == legacy_vllm_smoke_job.DEFAULT_AWS_G5_NODE_TYPE
    )
    assert (
        legacy_vllm_smoke_job.DatabricksSingleNodeGPUClusterConfig
        is legacy_vllm_smoke_job.DatabricksSingleNodeG5ClusterConfig
    )
    assert (
        legacy_vllm_smoke_job.build_single_node_gpu_cluster
        is legacy_vllm_smoke_job.build_single_node_g5_cluster
    )
    assert set(public_vllm_smoke_job.__all__) < set(legacy_vllm_smoke_job.__all__)


def test_legacy_vllm_smoke_job_keeps_previous_star_import_surface():
    assert set(legacy_vllm_smoke_job.__all__) == {
        "Any",
        "DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE",
        "DEFAULT_AWS_G5_NODE_TYPE",
        "DEFAULT_DATABRICKS_DATA_SECURITY_MODE",
        "DEFAULT_DATABRICKS_SPARK_VERSION",
        "DEFAULT_DATABRICKS_VLLM_SMOKE_PURPOSE",
        "DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME",
        "DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY",
        "DEFAULT_LOCAL_ROOT",
        "DatabricksSingleNodeGPUClusterConfig",
        "DatabricksSingleNodeG5ClusterConfig",
        "DatabricksVLLMSmokeJobConfig",
        "Mapping",
        "Path",
        "SERVER_HOST",
        "SERVER_PORT",
        "Sequence",
        "VLLM_SMOKE_RUNNER_SCRIPT",
        "argparse",
        "build_databricks_vllm_smoke_run_submit_payload",
        "build_single_node_gpu_cluster",
        "build_single_node_g5_cluster",
        "dataclass",
        "field",
        "json",
        "main",
        "write_databricks_vllm_smoke_run_submit_json",
        "write_databricks_vllm_smoke_runner_script",
    }


def test_legacy_vllm_smoke_job_star_import_uses_previous_surface():
    namespace: dict[str, object] = {}

    exec("from restaurant_kv_serving.databricks_vllm_smoke_job import *", namespace)

    assert {key for key in namespace if key != "__builtins__"} == set(legacy_vllm_smoke_job.__all__)
    assert namespace["DatabricksVLLMSmokeJobConfig"] is legacy_vllm_smoke_job.DatabricksVLLMSmokeJobConfig
    assert (
        namespace["DatabricksSingleNodeGPUClusterConfig"]
        is legacy_vllm_smoke_job.DatabricksSingleNodeGPUClusterConfig
    )
    assert namespace["build_single_node_gpu_cluster"] is legacy_vllm_smoke_job.build_single_node_gpu_cluster


def test_legacy_vllm_smoke_job_import_order_does_not_capture_public_monkeypatch(tmp_path):
    script = f"""
import json
import sys
from pathlib import Path

sys.path.insert(0, {str(REPO_ROOT / "src")!r})

import document_kv_cache.databricks_vllm_smoke_job as public_smoke_job


class FakeVLLMSmokeJobConfig:
    pass


def fake_runner_writer(path):
    Path(path).write_text("# unexpected public hook\\n", encoding="utf-8")


public_smoke_job.DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME = "public-patched-run"
public_smoke_job.DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY = "public_patched_task"
public_smoke_job.VLLM_SMOKE_RUNNER_SCRIPT = "# unexpected public default\\n"
public_smoke_job.DatabricksVLLMSmokeJobConfig = FakeVLLMSmokeJobConfig
public_smoke_job.write_databricks_vllm_smoke_runner_script = fake_runner_writer

import restaurant_kv_serving.databricks_vllm_smoke_job as legacy_smoke_job

assert legacy_smoke_job.DatabricksVLLMSmokeJobConfig is not FakeVLLMSmokeJobConfig
assert legacy_smoke_job.DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME == "document-kv-vllm-smoke"
assert legacy_smoke_job.DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY == "document_kv_vllm_smoke"

config = legacy_smoke_job.DatabricksVLLMSmokeJobConfig(
    benchmark_id="v1-vllm-smoke-001",
    output_dir="/Volumes/catalog/schema/volume/v1-vllm-smoke",
    runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
    single_user_name={SINGLE_USER_NAME!r},
)
payload = legacy_smoke_job.build_databricks_vllm_smoke_run_submit_payload(config)
assert payload["run_name"] == "document-kv-vllm-smoke"
assert payload["tasks"][0]["task_key"] == "document_kv_vllm_smoke"

runner_path = Path({str(tmp_path / "vllm_import_order_runner.py")!r})
legacy_smoke_job.write_databricks_vllm_smoke_runner_script(runner_path)
runner_text = runner_path.read_text(encoding="utf-8")
assert "# unexpected public hook" not in runner_text
assert "# unexpected public default" not in runner_text
assert "document_kv_cache.vllm_smoke" in runner_text

print(json.dumps({{"ok": True}}, sort_keys=True))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {"ok": True}


def test_legacy_vllm_smoke_job_import_order_ignores_in_place_public_class_mutation():
    script = f"""
import json
import pickle
import sys

sys.path.insert(0, {str(REPO_ROOT / "src")!r})

import document_kv_cache.databricks_vllm_smoke_job as public_smoke_job


def fake_public_config_init(self, *args, **kwargs):
    raise AssertionError("legacy inherited patched public config __init__")


public_smoke_job.DatabricksVLLMSmokeJobConfig.__init__ = fake_public_config_init
public_smoke_job.DatabricksVLLMSmokeJobConfig.__setattr__ = object.__setattr__

import restaurant_kv_serving.databricks_vllm_smoke_job as legacy_smoke_job

assert legacy_smoke_job.DatabricksVLLMSmokeJobConfig is not public_smoke_job.DatabricksVLLMSmokeJobConfig

config = legacy_smoke_job.DatabricksVLLMSmokeJobConfig(
    benchmark_id="v1-vllm-smoke-001",
    output_dir="/Volumes/catalog/schema/volume/v1-vllm-smoke",
    runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
    single_user_name={SINGLE_USER_NAME!r},
)
assert config.run_name == "document-kv-vllm-smoke"
assert not hasattr(config, "__dict__")
restored = pickle.loads(pickle.dumps(config))
assert type(restored) is legacy_smoke_job.DatabricksVLLMSmokeJobConfig
assert restored == config

print(json.dumps({{"ok": True}}, sort_keys=True))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {"ok": True}


def test_legacy_vllm_smoke_job_import_order_ignores_public_config_helper_monkeypatch(tmp_path):
    output_path = tmp_path / "payload.json"
    script = f"""
import json
import sys
from pathlib import Path

sys.path.insert(0, {str(REPO_ROOT / "src")!r})

import document_kv_cache.databricks_vllm_smoke_job as public_smoke_job


def broken_public_cluster_config(config):
    raise AssertionError("legacy used patched public config helper")


public_smoke_job._DEFAULT_CLUSTER_CONFIG_FROM_VLLM_SMOKE_JOB = broken_public_cluster_config

import restaurant_kv_serving.databricks_vllm_smoke_job as legacy_smoke_job

assert legacy_smoke_job.DatabricksVLLMSmokeJobConfig is not public_smoke_job.DatabricksVLLMSmokeJobConfig

exit_code = legacy_smoke_job.main(
    [
        "--benchmark-id",
        "v1-vllm-smoke-001",
        "--output-dir",
        "/Volumes/catalog/schema/volume/v1-vllm-smoke",
        "--runner-python-file",
        "dbfs:/benchmarks/run_vllm_smoke.py",
        "--single-user-name",
        {SINGLE_USER_NAME!r},
        "--output-json",
        {str(output_path)!r},
    ]
)

assert exit_code == 0
payload = json.loads(Path({str(output_path)!r}).read_text(encoding="utf-8"))
assert payload["run_name"] == "document-kv-vllm-smoke"
assert payload["tasks"][0]["task_key"] == "document_kv_vllm_smoke"

print(json.dumps({{"ok": True}}, sort_keys=True))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {"ok": True}
