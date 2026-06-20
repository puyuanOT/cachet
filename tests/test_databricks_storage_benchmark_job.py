import json
import os
import pickle
from pathlib import Path
import subprocess
import sys

import document_kv_cache.databricks_storage_benchmark_job as public_storage_benchmark_job
import restaurant_kv_serving.databricks_storage_benchmark_job as legacy_storage_benchmark_job
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


WHEEL_URI = "/Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl"
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
        node_type_id="g5.8xlarge",
        wheel_uri=WHEEL_URI,
        single_user_name=SINGLE_USER_NAME,
        custom_tags={"team": "document-kv"},
    )

    payload = build_databricks_storage_benchmark_run_submit_payload(config)
    task = payload["tasks"][0]
    cluster = task["new_cluster"]

    assert payload["run_name"] == DEFAULT_DATABRICKS_STORAGE_BENCHMARK_RUN_NAME
    assert task["task_key"] == DEFAULT_DATABRICKS_STORAGE_BENCHMARK_TASK_KEY
    assert task["libraries"] == [{"whl": WHEEL_URI}]
    assert cluster["node_type_id"] == "g5.8xlarge"
    assert cluster["driver_node_type_id"] == "g5.8xlarge"
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
    assert "document_kv_cache.storage_benchmark" in runner_text
    assert "if exit_code:" in runner_text


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
    assert json.loads(payload_path.read_text(encoding="utf-8"))["tasks"][0]["libraries"] == [{"whl": WHEEL_URI}]
    assert "storage_benchmark" in runner_path.read_text(encoding="utf-8")


def test_public_storage_benchmark_job_main_respects_document_namespace_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"
    original_legacy_build = legacy_storage_benchmark_job.build_databricks_storage_benchmark_run_submit_payload

    def fake_build(config):
        assert config.benchmark_id == "storage-reader-benchmark-001"
        return {"ok": True, "source": "public-hook"}

    monkeypatch.setattr(public_storage_benchmark_job, "build_databricks_storage_benchmark_run_submit_payload", fake_build)

    exit_code = public_storage_benchmark_job.main(
        [
            "--workspace-dir",
            "/local_disk0/document-kv-storage-benchmark",
            "--benchmark-output-json",
            "/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_storage_benchmark.py",
            "--benchmark-id",
            "storage-reader-benchmark-001",
            "--uc-volume-root",
            "/Volumes/catalog/schema/volume/storage",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"ok": True, "source": "public-hook"}
    assert legacy_storage_benchmark_job.build_databricks_storage_benchmark_run_submit_payload is original_legacy_build


def test_legacy_storage_benchmark_job_main_respects_legacy_namespace_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"
    original_public_build = public_storage_benchmark_job.build_databricks_storage_benchmark_run_submit_payload

    def fake_build(config):
        assert config.benchmark_id == "storage-reader-benchmark-001"
        return {"ok": True, "source": "legacy-hook"}

    monkeypatch.setattr(
        legacy_storage_benchmark_job,
        "build_databricks_storage_benchmark_run_submit_payload",
        fake_build,
    )

    exit_code = legacy_storage_benchmark_job.main(
        [
            "--workspace-dir",
            "/local_disk0/document-kv-storage-benchmark",
            "--benchmark-output-json",
            "/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_storage_benchmark.py",
            "--benchmark-id",
            "storage-reader-benchmark-001",
            "--uc-volume-root",
            "/Volumes/catalog/schema/volume/storage",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"ok": True, "source": "legacy-hook"}
    assert public_storage_benchmark_job.build_databricks_storage_benchmark_run_submit_payload is original_public_build


def test_legacy_storage_benchmark_job_ignores_document_namespace_build_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"

    def fake_public_build(config):
        return {"ok": True, "source": "unexpected-public-hook"}

    monkeypatch.setattr(
        public_storage_benchmark_job,
        "build_databricks_storage_benchmark_run_submit_payload",
        fake_public_build,
    )

    exit_code = legacy_storage_benchmark_job.main(
        [
            "--workspace-dir",
            "/local_disk0/document-kv-storage-benchmark",
            "--benchmark-output-json",
            "/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_storage_benchmark.py",
            "--benchmark-id",
            "storage-reader-benchmark-001",
            "--uc-volume-root",
            "/Volumes/catalog/schema/volume/storage",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload != {"ok": True, "source": "unexpected-public-hook"}
    assert payload["tasks"][0]["task_key"] == DEFAULT_DATABRICKS_STORAGE_BENCHMARK_TASK_KEY


def test_legacy_storage_benchmark_job_ignores_document_namespace_writer_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"
    runner_path = tmp_path / "run_storage_benchmark.py"

    def fake_public_runner_writer(path):
        Path(path).write_text("# unexpected public hook\n", encoding="utf-8")

    monkeypatch.setattr(public_storage_benchmark_job, "write_databricks_storage_benchmark_runner_script", fake_public_runner_writer)

    exit_code = legacy_storage_benchmark_job.main(
        [
            "--workspace-dir",
            "/local_disk0/document-kv-storage-benchmark",
            "--benchmark-output-json",
            "/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_storage_benchmark.py",
            "--benchmark-id",
            "storage-reader-benchmark-001",
            "--uc-volume-root",
            "/Volumes/catalog/schema/volume/storage",
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
    assert "document_kv_cache.storage_benchmark" in runner_path.read_text(encoding="utf-8")


def test_legacy_storage_benchmark_job_ignores_document_private_helper_monkeypatch(monkeypatch):
    config = legacy_storage_benchmark_job.DatabricksStorageBenchmarkJobConfig(
        workspace_dir="/local_disk0/document-kv-storage-benchmark",
        output_json="/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
        runner_python_file="dbfs:/benchmarks/run_storage_benchmark.py",
        benchmark_id="storage-reader-benchmark-001",
        uc_volume_root="/Volumes/catalog/schema/volume/storage",
        single_user_name=SINGLE_USER_NAME,
    )

    def fake_public_runner_parameters(config):
        return ["--unexpected-public-private-hook"]

    monkeypatch.setattr(public_storage_benchmark_job, "_runner_parameters", fake_public_runner_parameters)

    payload = legacy_storage_benchmark_job.build_databricks_storage_benchmark_run_submit_payload(config)

    assert payload["tasks"][0]["spark_python_task"]["parameters"] != ["--unexpected-public-private-hook"]
    assert payload["tasks"][0]["spark_python_task"]["parameters"][:2] == [
        "--workspace-dir",
        "/local_disk0/document-kv-storage-benchmark",
    ]


def test_legacy_storage_benchmark_job_payload_respects_legacy_private_cluster_monkeypatch(monkeypatch):
    config = legacy_storage_benchmark_job.DatabricksStorageBenchmarkJobConfig(
        workspace_dir="/local_disk0/document-kv-storage-benchmark",
        output_json="/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
        runner_python_file="dbfs:/benchmarks/run_storage_benchmark.py",
        benchmark_id="storage-reader-benchmark-001",
        uc_volume_root="/Volumes/catalog/schema/volume/storage",
        single_user_name=SINGLE_USER_NAME,
    )

    def broken_legacy_cluster_config(config):
        raise RuntimeError(f"legacy cluster config hook for {config.benchmark_id}")

    monkeypatch.setattr(
        legacy_storage_benchmark_job,
        "_cluster_config_from_storage_benchmark_job",
        broken_legacy_cluster_config,
    )

    try:
        legacy_storage_benchmark_job.build_databricks_storage_benchmark_run_submit_payload(config)
    except RuntimeError as exc:
        assert "legacy cluster config hook" in str(exc)
    else:
        raise AssertionError("expected legacy private cluster monkeypatch to be observed")


def test_legacy_storage_benchmark_job_config_ignores_document_private_helper_monkeypatch(monkeypatch):
    def broken_public_uc_root_check(value):
        raise RuntimeError(f"unexpected document private hook for {value}")

    monkeypatch.setattr(public_storage_benchmark_job, "is_real_uc_volume_root", broken_public_uc_root_check)

    config = legacy_storage_benchmark_job.DatabricksStorageBenchmarkJobConfig(
        workspace_dir="/local_disk0/document-kv-storage-benchmark",
        output_json="/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
        runner_python_file="dbfs:/benchmarks/run_storage_benchmark.py",
        benchmark_id="storage-reader-benchmark-001",
        uc_volume_root="/Volumes/catalog/schema/volume/storage",
        single_user_name=SINGLE_USER_NAME,
    )

    assert config.benchmark_id == "storage-reader-benchmark-001"


def test_legacy_storage_benchmark_job_direct_writer_respects_legacy_build_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"

    def fake_build(config):
        assert config.benchmark_id == "storage-reader-benchmark-001"
        return {"ok": True, "source": "legacy-direct-writer-hook"}

    monkeypatch.setattr(
        legacy_storage_benchmark_job,
        "build_databricks_storage_benchmark_run_submit_payload",
        fake_build,
    )

    legacy_storage_benchmark_job.write_databricks_storage_benchmark_run_submit_json(
        legacy_storage_benchmark_job.DatabricksStorageBenchmarkJobConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json="/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
            runner_python_file="dbfs:/benchmarks/run_storage_benchmark.py",
            benchmark_id="storage-reader-benchmark-001",
            uc_volume_root="/Volumes/catalog/schema/volume/storage",
            single_user_name=SINGLE_USER_NAME,
        ),
        output_path,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "ok": True,
        "source": "legacy-direct-writer-hook",
    }


def test_legacy_storage_benchmark_job_restores_document_hooks_after_error(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"
    original_public_build = public_storage_benchmark_job.build_databricks_storage_benchmark_run_submit_payload

    def broken_build(config):
        raise RuntimeError(f"boom for {config.benchmark_id}")

    monkeypatch.setattr(
        legacy_storage_benchmark_job,
        "build_databricks_storage_benchmark_run_submit_payload",
        broken_build,
    )

    exit_code = legacy_storage_benchmark_job.main(
        [
            "--workspace-dir",
            "/local_disk0/document-kv-storage-benchmark",
            "--benchmark-output-json",
            "/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_storage_benchmark.py",
            "--benchmark-id",
            "storage-reader-benchmark-001",
            "--uc-volume-root",
            "/Volumes/catalog/schema/volume/storage",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 1
    assert public_storage_benchmark_job.build_databricks_storage_benchmark_run_submit_payload is original_public_build


def test_legacy_storage_benchmark_job_module_execution_shows_help():
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }

    result = subprocess.run(
        [sys.executable, "-m", "restaurant_kv_serving.databricks_storage_benchmark_job", "--help"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "Emit a Databricks runs/submit payload for an AWS g5 storage-reader benchmark." in result.stdout


def test_legacy_storage_benchmark_job_reexports_document_owned_types():
    assert issubclass(
        legacy_storage_benchmark_job.DatabricksStorageBenchmarkJobConfig,
        public_storage_benchmark_job.DatabricksStorageBenchmarkJobConfig,
    )
    assert (
        legacy_storage_benchmark_job.DatabricksStorageBenchmarkJobConfig.__module__
        == "restaurant_kv_serving.databricks_storage_benchmark_job"
    )
    assert set(public_storage_benchmark_job.__all__) < set(legacy_storage_benchmark_job.__all__)


def test_legacy_storage_benchmark_job_config_pickle_uses_honest_legacy_module():
    config = legacy_storage_benchmark_job.DatabricksStorageBenchmarkJobConfig(
        workspace_dir="/local_disk0/document-kv-storage-benchmark",
        output_json="/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
        runner_python_file="dbfs:/benchmarks/run_storage_benchmark.py",
        benchmark_id="storage-reader-benchmark-001",
        uc_volume_root="/Volumes/catalog/schema/volume/storage",
        single_user_name=SINGLE_USER_NAME,
    )

    restored = pickle.loads(pickle.dumps(config))

    assert type(restored) is legacy_storage_benchmark_job.DatabricksStorageBenchmarkJobConfig
    assert restored == config


def test_legacy_storage_benchmark_job_config_keeps_slotted_layout():
    config = legacy_storage_benchmark_job.DatabricksStorageBenchmarkJobConfig(
        workspace_dir="/local_disk0/document-kv-storage-benchmark",
        output_json="/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
        runner_python_file="dbfs:/benchmarks/run_storage_benchmark.py",
        benchmark_id="storage-reader-benchmark-001",
        uc_volume_root="/Volumes/catalog/schema/volume/storage",
        single_user_name=SINGLE_USER_NAME,
    )

    assert not hasattr(config, "__dict__")


def test_legacy_storage_benchmark_job_keeps_previous_star_import_surface():
    assert set(legacy_storage_benchmark_job.__all__) == {
        "Any",
        "DEFAULT_AWS_G5_NODE_TYPE",
        "DEFAULT_DATABRICKS_DATA_SECURITY_MODE",
        "DEFAULT_DATABRICKS_SPARK_VERSION",
        "DEFAULT_DATABRICKS_STORAGE_BENCHMARK_PURPOSE",
        "DEFAULT_DATABRICKS_STORAGE_BENCHMARK_RUN_NAME",
        "DEFAULT_DATABRICKS_STORAGE_BENCHMARK_TASK_KEY",
        "DatabricksSingleNodeG5ClusterConfig",
        "DatabricksStorageBenchmarkJobConfig",
        "Mapping",
        "Path",
        "STORAGE_BENCHMARK_RUNNER_SCRIPT",
        "SUPPORTED_STORAGE_BENCHMARK_READERS",
        "Sequence",
        "StorageBenchmarkConfig",
        "argparse",
        "build_databricks_storage_benchmark_run_submit_payload",
        "build_single_node_g5_cluster",
        "dataclass",
        "field",
        "is_real_uc_volume_root",
        "json",
        "main",
        "write_databricks_storage_benchmark_run_submit_json",
        "write_databricks_storage_benchmark_runner_script",
    }


def test_legacy_storage_benchmark_job_star_import_uses_previous_surface():
    namespace: dict[str, object] = {}

    exec("from restaurant_kv_serving.databricks_storage_benchmark_job import *", namespace)

    assert {key for key in namespace if key != "__builtins__"} == set(legacy_storage_benchmark_job.__all__)
    assert namespace["DatabricksStorageBenchmarkJobConfig"] is legacy_storage_benchmark_job.DatabricksStorageBenchmarkJobConfig


def test_legacy_storage_benchmark_job_config_respects_legacy_uc_root_monkeypatch(monkeypatch):
    monkeypatch.setattr(legacy_storage_benchmark_job, "is_real_uc_volume_root", lambda value: False)

    try:
        legacy_storage_benchmark_job.DatabricksStorageBenchmarkJobConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json="/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
            runner_python_file="dbfs:/benchmarks/run_storage_benchmark.py",
            benchmark_id="storage-reader-benchmark-001",
            uc_volume_root="/Volumes/catalog/schema/volume/storage",
            single_user_name=SINGLE_USER_NAME,
        )
    except ValueError as exc:
        assert "real /Volumes" in str(exc)
    else:
        raise AssertionError("expected legacy UC root monkeypatch to be observed")


def test_legacy_storage_benchmark_job_config_respects_legacy_storage_config_monkeypatch(monkeypatch):
    def broken_storage_config(**kwargs):
        raise RuntimeError(f"legacy storage config hook for {kwargs['benchmark_id']}")

    monkeypatch.setattr(legacy_storage_benchmark_job, "StorageBenchmarkConfig", broken_storage_config)

    try:
        legacy_storage_benchmark_job.DatabricksStorageBenchmarkJobConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json="/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
            runner_python_file="dbfs:/benchmarks/run_storage_benchmark.py",
            benchmark_id="storage-reader-benchmark-001",
            uc_volume_root="/Volumes/catalog/schema/volume/storage",
            single_user_name=SINGLE_USER_NAME,
        )
    except RuntimeError as exc:
        assert "legacy storage config hook" in str(exc)
    else:
        raise AssertionError("expected legacy StorageBenchmarkConfig monkeypatch to be observed")


def test_legacy_storage_benchmark_job_import_order_does_not_capture_public_monkeypatch(tmp_path):
    script = f"""
import json
import sys
from pathlib import Path

sys.path.insert(0, {str(REPO_ROOT / "src")!r})

import document_kv_cache.databricks_storage_benchmark_job as public_storage_job


class FakeStorageBenchmarkJobConfig:
    pass


def fake_uc_root_check(value):
    raise AssertionError("legacy imported patched public UC root check")


def fake_storage_config(**kwargs):
    raise AssertionError("legacy imported patched public storage config")


def fake_runner_writer(path):
    Path(path).write_text("# unexpected public hook\\n", encoding="utf-8")


public_storage_job.DEFAULT_DATABRICKS_STORAGE_BENCHMARK_RUN_NAME = "public-patched-run"
public_storage_job.DEFAULT_DATABRICKS_STORAGE_BENCHMARK_TASK_KEY = "public_patched_task"
public_storage_job.STORAGE_BENCHMARK_RUNNER_SCRIPT = "# unexpected public default\\n"
public_storage_job.DatabricksStorageBenchmarkJobConfig = FakeStorageBenchmarkJobConfig
public_storage_job.is_real_uc_volume_root = fake_uc_root_check
public_storage_job.StorageBenchmarkConfig = fake_storage_config
public_storage_job.write_databricks_storage_benchmark_runner_script = fake_runner_writer

import restaurant_kv_serving.databricks_storage_benchmark_job as legacy_storage_job

assert not issubclass(legacy_storage_job.DatabricksStorageBenchmarkJobConfig, FakeStorageBenchmarkJobConfig)
assert legacy_storage_job.DEFAULT_DATABRICKS_STORAGE_BENCHMARK_RUN_NAME == "document-kv-storage-benchmark"
assert legacy_storage_job.DEFAULT_DATABRICKS_STORAGE_BENCHMARK_TASK_KEY == "document_kv_storage_benchmark"

config = legacy_storage_job.DatabricksStorageBenchmarkJobConfig(
    workspace_dir="/local_disk0/document-kv-storage-benchmark",
    output_json="/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
    runner_python_file="dbfs:/benchmarks/run_storage_benchmark.py",
    uc_volume_root="/Volumes/catalog/schema/volume/storage",
    single_user_name={SINGLE_USER_NAME!r},
)
payload = legacy_storage_job.build_databricks_storage_benchmark_run_submit_payload(config)
assert payload["run_name"] == "document-kv-storage-benchmark"
assert payload["tasks"][0]["task_key"] == "document_kv_storage_benchmark"

runner_path = Path({str(tmp_path / "storage_import_order_runner.py")!r})
legacy_storage_job.write_databricks_storage_benchmark_runner_script(runner_path)
runner_text = runner_path.read_text(encoding="utf-8")
assert "# unexpected public hook" not in runner_text
assert "# unexpected public default" not in runner_text
assert "document_kv_cache.storage_benchmark" in runner_text

print(json.dumps({{"ok": True}}, sort_keys=True))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {"ok": True}


def test_legacy_storage_benchmark_job_import_order_ignores_in_place_public_class_mutation():
    script = f"""
import json
import sys

sys.path.insert(0, {str(REPO_ROOT / "src")!r})

import document_kv_cache.databricks_storage_benchmark_job as public_storage_job


def fake_public_config_init(self, *args, **kwargs):
    raise AssertionError("legacy inherited patched public config __init__")


public_storage_job.DatabricksStorageBenchmarkJobConfig.__init__ = fake_public_config_init
public_storage_job.DatabricksStorageBenchmarkJobConfig.__setattr__ = object.__setattr__

import restaurant_kv_serving.databricks_storage_benchmark_job as legacy_storage_job

assert not issubclass(
    legacy_storage_job.DatabricksStorageBenchmarkJobConfig,
    public_storage_job.DatabricksStorageBenchmarkJobConfig,
)

config = legacy_storage_job.DatabricksStorageBenchmarkJobConfig(
    workspace_dir="/local_disk0/document-kv-storage-benchmark",
    output_json="/Volumes/catalog/schema/volume/storage/storage-benchmark.json",
    runner_python_file="dbfs:/benchmarks/run_storage_benchmark.py",
    uc_volume_root="/Volumes/catalog/schema/volume/storage",
    single_user_name={SINGLE_USER_NAME!r},
)
assert config.run_name == "document-kv-storage-benchmark"

print(json.dumps({{"ok": True}}, sort_keys=True))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {"ok": True}


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
