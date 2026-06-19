import json
from pathlib import Path

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


def test_build_databricks_vllm_smoke_payload_uses_single_node_g5_cluster():
    config = DatabricksVLLMSmokeJobConfig(
        benchmark_id="v1-vllm-smoke-001",
        output_dir="/Volumes/catalog/schema/volume/v1-vllm-smoke",
        runner_python_file="dbfs:/benchmarks/run_vllm_smoke.py",
        node_type_id="g5.8xlarge",
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
        custom_tags={"team": "document-kv"},
    )

    payload = build_databricks_vllm_smoke_run_submit_payload(config)
    task = payload["tasks"][0]
    cluster = task["new_cluster"]

    assert payload["run_name"] == DEFAULT_DATABRICKS_VLLM_SMOKE_RUN_NAME
    assert task["task_key"] == DEFAULT_DATABRICKS_VLLM_SMOKE_TASK_KEY
    assert task["libraries"] == [{"whl": WHEEL_URI}]
    assert cluster["node_type_id"] == "g5.8xlarge"
    assert cluster["driver_node_type_id"] == "g5.8xlarge"
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
        ],
    }


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


def test_write_databricks_vllm_smoke_runner_script_imports_smoke_main(tmp_path):
    path = tmp_path / "run_vllm_smoke.py"

    write_databricks_vllm_smoke_runner_script(path)

    runner_text = path.read_text(encoding="utf-8")
    assert "document_kv_cache.vllm_smoke" in runner_text
    assert "if exit_code:" in runner_text


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
            "--output-json",
            str(payload_path),
            "--runner-script-output",
            str(runner_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(payload_path.read_text(encoding="utf-8"))["tasks"][0]["libraries"] == [{"whl": WHEEL_URI}]
    assert "vllm_smoke" in runner_path.read_text(encoding="utf-8")


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
