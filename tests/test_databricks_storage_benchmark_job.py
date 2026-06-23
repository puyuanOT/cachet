import json
import os
import pickle
from pathlib import Path
import subprocess
import sys

import document_kv_cache.databricks_storage_benchmark_job as public_storage_benchmark_job
from document_kv_cache.databricks_storage_benchmark_job import (
    DEFAULT_DATABRICKS_STORAGE_BENCHMARK_PURPOSE,
    DEFAULT_DATABRICKS_STORAGE_BENCHMARK_RUN_NAME,
    DEFAULT_DATABRICKS_STORAGE_BENCHMARK_TASK_KEY,
    DatabricksStorageBenchmarkJobConfig,
    build_databricks_storage_benchmark_run_submit_payload,
    main,
    write_databricks_storage_benchmark_run_submit_json,
    write_databricks_storage_benchmark_runner_script,
)


WHEEL_URI = "/Volumes/catalog/schema/volume/wheels/cachet_kv-0.2.0-py3-none-any.whl"
SINGLE_USER_NAME = "user@example.com"
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_databricks_storage_benchmark_payload_uses_single_node_g5_cluster():
    config = DatabricksStorageBenchmarkJobConfig(
        workspace_dir="/local_disk0/document-kv-storage-benchmark",
        output_json="/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
        runner_python_file="dbfs:/benchmarks/run_storage_benchmark.py",
        benchmark_id="storage-reader-benchmark-001",
        chunk_count=8,
        chunk_bytes=131072,
        repeats=3,
        parallelism=6,
        readers=("memory", "disk", "unity_catalog"),
        align_bytes=8192,
        uc_volume_root="/Volumes/catalog/schema/volume/storage",
        node_type_id="g6.8xlarge",
        wheel_uri=WHEEL_URI,
        single_user_name=SINGLE_USER_NAME,
        custom_tags={"team": "document-kv"},
    )

    payload = build_databricks_storage_benchmark_run_submit_payload(config)
    task = payload["tasks"][0]
    cluster = task["new_cluster"]

    assert payload["run_name"] == DEFAULT_DATABRICKS_STORAGE_BENCHMARK_RUN_NAME
    assert task["task_key"] == DEFAULT_DATABRICKS_STORAGE_BENCHMARK_TASK_KEY
    assert "libraries" not in task
    assert cluster["node_type_id"] == "g6.8xlarge"
    assert cluster["driver_node_type_id"] == "g6.8xlarge"
    assert cluster["data_security_mode"] == "SINGLE_USER"
    assert cluster["single_user_name"] == SINGLE_USER_NAME
    assert cluster["num_workers"] == 0
    assert cluster["custom_tags"]["ResourceClass"] == "SingleNode"
    assert cluster["custom_tags"]["purpose"] == DEFAULT_DATABRICKS_STORAGE_BENCHMARK_PURPOSE
    assert cluster["custom_tags"]["team"] == "document-kv"
    assert task["spark_python_task"] == {
        "python_file": "dbfs:/benchmarks/run_storage_benchmark.py",
        "parameters": [
            "--workspace-dir",
            "/local_disk0/document-kv-storage-benchmark",
            "--benchmark-id",
            "storage-reader-benchmark-001",
            "--chunk-count",
            "8",
            "--chunk-bytes",
            "131072",
            "--repeats",
            "3",
            "--parallelism",
            "6",
            "--align-bytes",
            "8192",
            "--output-json",
            "/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
            "--reader",
            "memory",
            "--reader",
            "disk",
            "--reader",
            "unity_catalog",
            "--uc-volume-root",
            "/Volumes/catalog/schema/volume/storage",
            "--package-wheel-uri",
            WHEEL_URI,
        ],
    }


def test_databricks_storage_benchmark_config_requires_single_user_name_and_validates_storage_args():
    try:
        DatabricksStorageBenchmarkJobConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json="/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
            runner_python_file="dbfs:/benchmarks/run_storage_benchmark.py",
            uc_volume_root="/Volumes/catalog/schema/volume/storage",
        )
    except ValueError as exc:
        assert "single_user_name is required" in str(exc)
    else:
        raise AssertionError("expected SINGLE_USER validation to fail")

    try:
        DatabricksStorageBenchmarkJobConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json="/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
            runner_python_file="dbfs:/benchmarks/run_storage_benchmark.py",
            uc_volume_root="/Volumes/catalog/schema/volume/storage",
            readers=("object-store",),
            single_user_name=SINGLE_USER_NAME,
        )
    except ValueError as exc:
        assert "Unsupported storage benchmark readers" in str(exc)
    else:
        raise AssertionError("expected reader validation to fail")


def test_write_databricks_storage_benchmark_runner_script_imports_storage_main(tmp_path):
    path = tmp_path / "run_storage_benchmark.py"

    write_databricks_storage_benchmark_runner_script(path)

    runner_text = path.read_text(encoding="utf-8")
    assert "--package-wheel-uri" in runner_text
    assert "pip\", \"install\"" in runner_text
    assert "dbfs:/" in runner_text
    assert "document_kv_cache.storage_benchmark" in runner_text
    assert "if exit_code:" in runner_text


def test_generated_storage_benchmark_runner_installs_wheel_before_forwarding_args(tmp_path):
    runner_path = tmp_path / "run_storage_benchmark.py"
    pip_call_path = tmp_path / "pip-call.json"
    main_args_path = tmp_path / "main-args.json"
    events_path = tmp_path / "events.jsonl"
    package_dir = tmp_path / "document_kv_cache"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "storage_benchmark.py").write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "",
                "with open(os.environ['RUNNER_EVENTS_JSONL'], 'a', encoding='utf-8') as handle:",
                "    handle.write(json.dumps({'event': 'storage_benchmark_import'}) + '\\n')",
                "",
                "def main(argv=None):",
                "    with open(os.environ['RUNNER_EVENTS_JSONL'], 'a', encoding='utf-8') as handle:",
                "        handle.write(json.dumps({'event': 'main'}) + '\\n')",
                "    with open(os.environ['MAIN_ARGS_JSON'], 'w', encoding='utf-8') as handle:",
                "        json.dump(argv, handle)",
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

    write_databricks_storage_benchmark_runner_script(runner_path)
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
            "--workspace-dir",
            "/local_disk0/document-kv-storage-benchmark",
            "--output-json",
            "/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
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
    assert json.loads(main_args_path.read_text(encoding="utf-8")) == [
        "--workspace-dir",
        "/local_disk0/document-kv-storage-benchmark",
        "--output-json",
        "/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
    ]
    events = [json.loads(line)["event"] for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert events == ["pip_install", "storage_benchmark_import", "main"]


def test_write_databricks_storage_benchmark_run_submit_json_writes_payload(tmp_path):
    path = tmp_path / "payload.json"

    write_databricks_storage_benchmark_run_submit_json(
        DatabricksStorageBenchmarkJobConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json="/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
            runner_python_file="dbfs:/benchmarks/run_storage_benchmark.py",
            uc_volume_root="/Volumes/catalog/schema/volume/storage",
            single_user_name=SINGLE_USER_NAME,
        ),
        path,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["tasks"][0]["task_key"] == DEFAULT_DATABRICKS_STORAGE_BENCHMARK_TASK_KEY


def test_main_writes_storage_benchmark_payload_and_runner_script(tmp_path):
    payload_path = tmp_path / "payload.json"
    runner_path = tmp_path / "run_storage_benchmark.py"

    exit_code = main(
        [
            "--workspace-dir",
            "/local_disk0/document-kv-storage-benchmark",
            "--benchmark-output-json",
            "/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_storage_benchmark.py",
            "--uc-volume-root",
            "/Volumes/catalog/schema/volume/storage",
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
    task = json.loads(payload_path.read_text(encoding="utf-8"))["tasks"][0]
    assert "libraries" not in task
    assert task["spark_python_task"]["parameters"][-2:] == ["--package-wheel-uri", WHEEL_URI]
    assert "storage_benchmark" in runner_path.read_text(encoding="utf-8")


def test_main_derives_storage_node_type_from_g5_hardware_target(tmp_path):
    payload_path = tmp_path / "payload.json"

    exit_code = main(
        [
            "--workspace-dir",
            "/local_disk0/document-kv-storage-benchmark",
            "--benchmark-output-json",
            "/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_storage_benchmark.py",
            "--uc-volume-root",
            "/Volumes/catalog/schema/volume/storage",
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


def test_main_preserves_legacy_storage_g5_node_type_without_hardware_target(tmp_path):
    payload_path = tmp_path / "payload.json"

    exit_code = main(
        [
            "--workspace-dir",
            "/local_disk0/document-kv-storage-benchmark",
            "--benchmark-output-json",
            "/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_storage_benchmark.py",
            "--uc-volume-root",
            "/Volumes/catalog/schema/volume/storage",
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


def test_databricks_storage_benchmark_config_rejects_non_uc_roots_before_job_submission():
    try:
        DatabricksStorageBenchmarkJobConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json="/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
            runner_python_file="dbfs:/benchmarks/run_storage_benchmark.py",
            uc_volume_root="/local_disk0/not-a-real-uc-volume",
            single_user_name=SINGLE_USER_NAME,
        )
    except ValueError as exc:
        assert "real /Volumes" in str(exc)
    else:
        raise AssertionError("expected real UC root validation to fail")


def test_storage_benchmark_bundle_template_matches_packaged_copy():
    repo_bundle = REPO_ROOT / "databricks" / "storage-benchmark"
    package_bundle = REPO_ROOT / "src" / "document_kv_cache" / "templates" / "databricks" / "storage-benchmark"

    assert (repo_bundle / "databricks.yml").read_text(encoding="utf-8") == (
        package_bundle / "databricks.yml"
    ).read_text(encoding="utf-8")
    assert (repo_bundle / "README.md").read_text(encoding="utf-8") == (package_bundle / "README.md").read_text(
        encoding="utf-8"
    )
