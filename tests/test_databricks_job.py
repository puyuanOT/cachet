import json
import os
import pickle
from pathlib import Path
import subprocess
import sys

import pytest

import document_kv_cache.databricks_job as public_databricks_job
import restaurant_kv_serving.databricks_job as legacy_databricks_job
from document_kv_cache.benchmarks import DEFAULT_HARDWARE_TARGET
from document_kv_cache.databricks_job import (
    DEDICATED_DATABRICKS_DATA_SECURITY_MODE,
    DEFAULT_DATABRICKS_PURPOSE,
    RESERVED_SINGLE_NODE_G5_TAG_KEYS,
    RESERVED_SINGLE_NODE_GPU_TAG_KEYS,
    DatabricksBenchmarkJobConfig,
    DatabricksSingleNodeG5ClusterConfig,
    DatabricksSingleNodeGPUClusterConfig,
    build_databricks_run_submit_payload,
    build_single_node_g5_cluster,
    build_single_node_gpu_cluster,
    main,
    validate_aws_g5_node_type,
    validate_aws_single_node_gpu_type,
    write_databricks_runner_script,
    write_databricks_run_submit_json,
)
from document_kv_cache.databricks_engine_probe_job import (
    VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,
)
from document_kv_cache._hardware_targets import (
    SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES,
    V1_HARDWARE_TARGET_PROFILE,
)
from document_kv_cache.native_probe_factories import (
    SGLANG_NATIVE_PROBE_DELEGATE_ENV,
    VLLM_NATIVE_PROBE_DELEGATE_ENV,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WHEEL_URI = "/Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl"
SINGLE_USER_NAME = "user@example.com"
REPO_BUNDLE_TEMPLATE = REPO_ROOT / "databricks" / "databricks.yml"
PACKAGED_BUNDLE_TEMPLATE = REPO_ROOT / "src" / "document_kv_cache" / "templates" / "databricks" / "databricks.yml"
REPO_VLLM_SMOKE_BUNDLE_TEMPLATE = REPO_ROOT / "databricks" / "vllm-smoke" / "databricks.yml"
PACKAGED_VLLM_SMOKE_BUNDLE_TEMPLATE = (
    REPO_ROOT / "src" / "document_kv_cache" / "templates" / "databricks" / "vllm-smoke" / "databricks.yml"
)
REPO_ENGINE_PROBE_BUNDLE_TEMPLATE = REPO_ROOT / "databricks" / "engine-probe" / "databricks.yml"
PACKAGED_ENGINE_PROBE_BUNDLE_TEMPLATE = (
    REPO_ROOT / "src" / "document_kv_cache" / "templates" / "databricks" / "engine-probe" / "databricks.yml"
)


def test_build_databricks_run_submit_payload_uses_single_node_g5_cluster():
    config = DatabricksBenchmarkJobConfig(
        plan_json_uri="dbfs:/benchmarks/v1-plan.json",
        runner_python_file="dbfs:/benchmarks/run_plan.py",
        wheel_uri=WHEEL_URI,
        execution_result_json_uri="dbfs:/benchmarks/result.json",
        single_user_name=SINGLE_USER_NAME,
        custom_tags={"team": "document-kv"},
    )

    payload = build_databricks_run_submit_payload(config)
    task = payload["tasks"][0]
    cluster = task["new_cluster"]

    assert payload["run_name"] == "document-kv-v1-benchmark"
    assert cluster["node_type_id"] == "g6.8xlarge"
    assert cluster["driver_node_type_id"] == "g6.8xlarge"
    assert cluster["data_security_mode"] == "SINGLE_USER"
    assert cluster["single_user_name"] == SINGLE_USER_NAME
    assert cluster["num_workers"] == 0
    assert cluster["spark_conf"]["spark.databricks.cluster.profile"] == "singleNode"
    assert cluster["aws_attributes"] == {"availability": "ON_DEMAND", "zone_id": "auto"}
    assert cluster["custom_tags"]["ResourceClass"] == "SingleNode"
    assert cluster["custom_tags"]["purpose"] == DEFAULT_DATABRICKS_PURPOSE
    assert cluster["custom_tags"]["team"] == "document-kv"
    assert task["spark_python_task"] == {
        "python_file": "dbfs:/benchmarks/run_plan.py",
        "parameters": [
            "--plan-json",
            "dbfs:/benchmarks/v1-plan.json",
            "--result-json",
            "dbfs:/benchmarks/result.json",
            "--package-wheel-uri",
            WHEEL_URI,
        ],
    }
    assert "libraries" not in task


def test_build_single_node_g5_cluster_is_reusable_with_custom_purpose():
    cluster = build_single_node_g5_cluster(
        DatabricksSingleNodeG5ClusterConfig(
            purpose="document-kv-vllm-smoke",
            node_type_id="g5.8xlarge",
            single_user_name=SINGLE_USER_NAME,
            custom_tags={"team": "document-kv"},
        )
    )

    assert cluster["node_type_id"] == "g5.8xlarge"
    assert cluster["driver_node_type_id"] == "g5.8xlarge"
    assert cluster["custom_tags"] == {
        "ResourceClass": "SingleNode",
        "purpose": "document-kv-vllm-smoke",
        "team": "document-kv",
    }
    assert cluster["single_user_name"] == SINGLE_USER_NAME


def test_build_databricks_run_submit_payload_sets_native_probe_delegate_env_vars():
    config = DatabricksBenchmarkJobConfig(
        plan_json_uri="dbfs:/benchmarks/v1-plan.json",
        runner_python_file="dbfs:/benchmarks/run_plan.py",
        single_user_name=SINGLE_USER_NAME,
        vllm_native_probe_delegate_factory="document_kv_vllm_native_adapter:build_probe",
        sglang_native_probe_delegate_factory="document_kv_sglang_native_adapter:build_probe",
    )

    payload = build_databricks_run_submit_payload(config)
    task = payload["tasks"][0]

    assert task["new_cluster"]["spark_env_vars"] == {
        VLLM_NATIVE_PROBE_DELEGATE_ENV: "document_kv_vllm_native_adapter:build_probe",
        SGLANG_NATIVE_PROBE_DELEGATE_ENV: "document_kv_sglang_native_adapter:build_probe",
    }
    assert "--vllm-native-probe-delegate-factory" not in task["spark_python_task"]["parameters"]
    assert "--sglang-native-probe-delegate-factory" not in task["spark_python_task"]["parameters"]


def test_single_node_g5_cluster_rejects_reserved_custom_tags():
    with pytest.raises(ValueError, match="reserved tags"):
        DatabricksSingleNodeG5ClusterConfig(
            purpose="document-kv-vllm-smoke",
            single_user_name=SINGLE_USER_NAME,
            custom_tags={"ResourceClass": "MultiNode"},
        )

    with pytest.raises(ValueError, match="reserved tags"):
        DatabricksSingleNodeG5ClusterConfig(
            purpose="document-kv-vllm-smoke",
            single_user_name=SINGLE_USER_NAME,
            custom_tags={"purpose": "wrong-purpose"},
        )


def test_databricks_config_requires_single_user_name_for_single_user_clusters():
    with pytest.raises(ValueError, match="single_user_name is required"):
        DatabricksBenchmarkJobConfig(
            plan_json_uri="dbfs:/benchmarks/v1-plan.json",
            runner_python_file="dbfs:/benchmarks/run_plan.py",
        )


def test_databricks_config_omits_single_user_name_for_non_single_user_clusters():
    config = DatabricksBenchmarkJobConfig(
        plan_json_uri="dbfs:/benchmarks/v1-plan.json",
        runner_python_file="dbfs:/benchmarks/run_plan.py",
        data_security_mode="USER_ISOLATION",
        single_user_name=SINGLE_USER_NAME,
    )

    payload = build_databricks_run_submit_payload(config)

    assert payload["tasks"][0]["new_cluster"]["data_security_mode"] == "USER_ISOLATION"
    assert "single_user_name" not in payload["tasks"][0]["new_cluster"]


def test_databricks_config_keeps_single_user_name_for_dedicated_clusters():
    config = DatabricksBenchmarkJobConfig(
        plan_json_uri="dbfs:/benchmarks/v1-plan.json",
        runner_python_file="dbfs:/benchmarks/run_plan.py",
        data_security_mode=DEDICATED_DATABRICKS_DATA_SECURITY_MODE,
        single_user_name=SINGLE_USER_NAME,
    )

    payload = build_databricks_run_submit_payload(config)

    assert payload["tasks"][0]["new_cluster"]["data_security_mode"] == DEDICATED_DATABRICKS_DATA_SECURITY_MODE
    assert payload["tasks"][0]["new_cluster"]["single_user_name"] == SINGLE_USER_NAME


def test_validate_aws_g5_node_type_alias_accepts_v1_g5_and_g6_families():
    validate_aws_g5_node_type("g5.8xlarge")
    validate_aws_g5_node_type("g6.8xlarge")

    with pytest.raises(ValueError, match="supported V1 Databricks node type"):
        validate_aws_g5_node_type("g6e.8xlarge")


def test_databricks_defaults_share_v1_hardware_target_profile():
    assert DEFAULT_HARDWARE_TARGET == V1_HARDWARE_TARGET_PROFILE.hardware_target
    assert (
        public_databricks_job.DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE
        == V1_HARDWARE_TARGET_PROFILE.default_databricks_node_type_id
    )
    assert (
        public_databricks_job.SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES
        == SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES
    )


def test_generic_single_node_gpu_aliases_preserve_g5_compatibility_names():
    assert RESERVED_SINGLE_NODE_GPU_TAG_KEYS is RESERVED_SINGLE_NODE_G5_TAG_KEYS
    assert DatabricksSingleNodeGPUClusterConfig is DatabricksSingleNodeG5ClusterConfig
    assert validate_aws_single_node_gpu_type is validate_aws_g5_node_type
    assert build_single_node_gpu_cluster is build_single_node_g5_cluster
    assert DatabricksSingleNodeGPUClusterConfig.__name__ == "DatabricksSingleNodeGPUClusterConfig"
    assert validate_aws_single_node_gpu_type.__name__ == "validate_aws_single_node_gpu_type"
    assert validate_aws_single_node_gpu_type.__module__ == "document_kv_cache.databricks_job"
    assert validate_aws_g5_node_type.__module__ == "document_kv_cache.databricks_job"
    assert build_single_node_gpu_cluster.__name__ == "build_single_node_gpu_cluster"

    validate_aws_single_node_gpu_type("g6.8xlarge")
    validate_aws_single_node_gpu_type("g5.8xlarge")


def test_write_databricks_runner_script_imports_plan_executor(tmp_path):
    path = tmp_path / "run_plan.py"

    write_databricks_runner_script(path)

    runner_text = path.read_text(encoding="utf-8")
    assert "--package-wheel-uri" in runner_text
    assert "pip\", \"install\"" in runner_text
    assert "dbfs:/" in runner_text
    assert "document_kv_cache.benchmark_plan_executor" in runner_text
    assert "raise SystemExit(main())" not in runner_text
    assert "if exit_code:" in runner_text


def test_generated_databricks_runner_installs_wheel_before_forwarding_args(tmp_path):
    runner_path = tmp_path / "run_plan.py"
    pip_call_path = tmp_path / "pip-call.json"
    main_args_path = tmp_path / "main-args.json"
    events_path = tmp_path / "events.jsonl"
    package_dir = tmp_path / "document_kv_cache"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "benchmark_plan_executor.py").write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "",
                "with open(os.environ['RUNNER_EVENTS_JSONL'], 'a', encoding='utf-8') as handle:",
                "    handle.write(json.dumps({'event': 'benchmark_plan_executor_import'}) + '\\n')",
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

    write_databricks_runner_script(runner_path)
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
            "--plan-json",
            "dbfs:/benchmarks/v1-plan.json",
            "--result-json",
            "dbfs:/benchmarks/result.json",
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
    assert json.loads(main_args_path.read_text(encoding="utf-8")) == [
        "--plan-json",
        "dbfs:/benchmarks/v1-plan.json",
        "--result-json",
        "dbfs:/benchmarks/result.json",
    ]
    events = [json.loads(line)["event"] for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert events == ["pip_install", "benchmark_plan_executor_import", "main"]


def test_write_databricks_run_submit_json_writes_payload(tmp_path):
    path = tmp_path / "payload.json"

    write_databricks_run_submit_json(
        DatabricksBenchmarkJobConfig(
            plan_json_uri="dbfs:/benchmarks/v1-plan.json",
            runner_python_file="dbfs:/benchmarks/run_plan.py",
            single_user_name=SINGLE_USER_NAME,
        ),
        path,
    )

    assert json.loads(path.read_text(encoding="utf-8"))["tasks"][0]["task_key"] == "document_kv_v1_benchmark"


def test_main_writes_payload_and_runner_script(tmp_path):
    payload_path = tmp_path / "payload.json"
    runner_path = tmp_path / "run_plan.py"

    exit_code = main(
        [
            "--plan-json-uri",
            "dbfs:/benchmarks/v1-plan.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_plan.py",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--wheel-uri",
            WHEEL_URI,
            "--vllm-native-probe-delegate-factory",
            "document_kv_vllm_native_adapter:build_probe",
            "--sglang-native-probe-delegate-factory",
            "document_kv_sglang_native_adapter:build_probe",
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
    assert task["new_cluster"]["spark_env_vars"] == {
        VLLM_NATIVE_PROBE_DELEGATE_ENV: "document_kv_vllm_native_adapter:build_probe",
        SGLANG_NATIVE_PROBE_DELEGATE_ENV: "document_kv_sglang_native_adapter:build_probe",
    }
    assert "--vllm-native-probe-delegate-factory" not in task["spark_python_task"]["parameters"]
    assert "--sglang-native-probe-delegate-factory" not in task["spark_python_task"]["parameters"]
    assert "benchmark_plan_executor" in runner_path.read_text(encoding="utf-8")


def test_public_databricks_job_main_respects_document_namespace_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"
    original_legacy_build = legacy_databricks_job.build_databricks_run_submit_payload

    def fake_build(config):
        assert config.plan_json_uri == "dbfs:/benchmarks/v1-plan.json"
        return {"ok": True, "source": "public-hook"}

    monkeypatch.setattr(public_databricks_job, "build_databricks_run_submit_payload", fake_build)

    exit_code = public_databricks_job.main(
        [
            "--plan-json-uri",
            "dbfs:/benchmarks/v1-plan.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_plan.py",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"ok": True, "source": "public-hook"}
    assert legacy_databricks_job.build_databricks_run_submit_payload is original_legacy_build


def test_legacy_databricks_job_main_respects_legacy_namespace_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"
    original_public_build = public_databricks_job.build_databricks_run_submit_payload

    def fake_build(config):
        assert config.plan_json_uri == "dbfs:/benchmarks/v1-plan.json"
        return {"ok": True, "source": "legacy-hook"}

    monkeypatch.setattr(legacy_databricks_job, "build_databricks_run_submit_payload", fake_build)

    exit_code = legacy_databricks_job.main(
        [
            "--plan-json-uri",
            "dbfs:/benchmarks/v1-plan.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_plan.py",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"ok": True, "source": "legacy-hook"}
    assert public_databricks_job.build_databricks_run_submit_payload is original_public_build


def test_legacy_databricks_job_ignores_document_namespace_build_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"

    def fake_public_build(config):
        return {"ok": True, "source": "unexpected-public-hook"}

    monkeypatch.setattr(public_databricks_job, "build_databricks_run_submit_payload", fake_public_build)

    exit_code = legacy_databricks_job.main(
        [
            "--plan-json-uri",
            "dbfs:/benchmarks/v1-plan.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_plan.py",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload != {"ok": True, "source": "unexpected-public-hook"}
    assert payload["tasks"][0]["task_key"] == "document_kv_v1_benchmark"


def test_legacy_databricks_job_ignores_document_namespace_writer_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"
    runner_path = tmp_path / "run_plan.py"

    def fake_public_runner_writer(path):
        Path(path).write_text("# unexpected public hook\n", encoding="utf-8")

    monkeypatch.setattr(public_databricks_job, "write_databricks_runner_script", fake_public_runner_writer)

    exit_code = legacy_databricks_job.main(
        [
            "--plan-json-uri",
            "dbfs:/benchmarks/v1-plan.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_plan.py",
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
    assert "document_kv_cache.benchmark_plan_executor" in runner_path.read_text(encoding="utf-8")


def test_legacy_databricks_job_ignores_document_private_helper_monkeypatch(monkeypatch):
    config = legacy_databricks_job.DatabricksBenchmarkJobConfig(
        plan_json_uri="dbfs:/benchmarks/v1-plan.json",
        runner_python_file="dbfs:/benchmarks/run_plan.py",
        execution_result_json_uri="dbfs:/benchmarks/result.json",
        single_user_name=SINGLE_USER_NAME,
    )

    def fake_public_runner_parameters(config):
        return ["--unexpected-public-private-hook"]

    monkeypatch.setattr(public_databricks_job, "_runner_parameters", fake_public_runner_parameters)

    payload = legacy_databricks_job.build_databricks_run_submit_payload(config)

    assert payload["tasks"][0]["spark_python_task"]["parameters"] != ["--unexpected-public-private-hook"]
    assert payload["tasks"][0]["spark_python_task"]["parameters"][:2] == [
        "--plan-json",
        "dbfs:/benchmarks/v1-plan.json",
    ]


def test_legacy_databricks_job_payload_respects_legacy_private_cluster_monkeypatch(monkeypatch):
    config = legacy_databricks_job.DatabricksBenchmarkJobConfig(
        plan_json_uri="dbfs:/benchmarks/v1-plan.json",
        runner_python_file="dbfs:/benchmarks/run_plan.py",
        single_user_name=SINGLE_USER_NAME,
    )

    def broken_legacy_cluster(config):
        raise RuntimeError(f"legacy cluster hook for {config.plan_json_uri}")

    monkeypatch.setattr(legacy_databricks_job, "_single_node_g5_cluster", broken_legacy_cluster)

    try:
        legacy_databricks_job.build_databricks_run_submit_payload(config)
    except RuntimeError as exc:
        assert "legacy cluster hook" in str(exc)
    else:
        raise AssertionError("expected legacy private cluster monkeypatch to be observed")


def test_legacy_databricks_job_config_ignores_document_private_helper_monkeypatch(monkeypatch):
    def broken_public_cluster_config(config):
        raise RuntimeError(f"unexpected document private hook for {config.plan_json_uri}")

    monkeypatch.setattr(public_databricks_job, "_cluster_config_from_benchmark_job", broken_public_cluster_config)

    config = legacy_databricks_job.DatabricksBenchmarkJobConfig(
        plan_json_uri="dbfs:/benchmarks/v1-plan.json",
        runner_python_file="dbfs:/benchmarks/run_plan.py",
        single_user_name=SINGLE_USER_NAME,
    )

    assert config.plan_json_uri == "dbfs:/benchmarks/v1-plan.json"


def test_legacy_databricks_job_config_respects_legacy_private_cluster_config_monkeypatch(monkeypatch):
    def broken_legacy_cluster_config(config):
        raise RuntimeError(f"legacy config hook for {config.plan_json_uri}")

    monkeypatch.setattr(legacy_databricks_job, "_cluster_config_from_benchmark_job", broken_legacy_cluster_config)

    try:
        legacy_databricks_job.DatabricksBenchmarkJobConfig(
            plan_json_uri="dbfs:/benchmarks/v1-plan.json",
            runner_python_file="dbfs:/benchmarks/run_plan.py",
            single_user_name=SINGLE_USER_NAME,
        )
    except RuntimeError as exc:
        assert "legacy config hook" in str(exc)
    else:
        raise AssertionError("expected legacy cluster config monkeypatch to be observed")


def test_legacy_databricks_job_config_respects_legacy_node_type_validator_monkeypatch(monkeypatch):
    def broken_validator(node_type_id):
        raise RuntimeError(f"legacy validator hook for {node_type_id}")

    monkeypatch.setattr(legacy_databricks_job, "validate_aws_g5_node_type", broken_validator)

    try:
        legacy_databricks_job.DatabricksSingleNodeG5ClusterConfig(
            purpose="document-kv-v1-benchmark",
            single_user_name=SINGLE_USER_NAME,
        )
    except RuntimeError as exc:
        assert "legacy validator hook" in str(exc)
    else:
        raise AssertionError("expected legacy node-type validator monkeypatch to be observed")


def test_legacy_databricks_job_import_order_does_not_capture_public_monkeypatch():
    script = f"""
import json
from pathlib import Path
import tempfile

import document_kv_cache.databricks_job as public_databricks_job

def public_validator_should_not_run(node_type_id):
    raise AssertionError(f"legacy imported patched public validator for {{node_type_id}}")

def public_runner_writer_should_not_run(path):
    Path(path).write_text("# unexpected public hook\\n", encoding="utf-8")

class FakeDatabricksBenchmarkJobConfig:
    def __init__(self, *args, **kwargs):
        raise AssertionError("legacy imported patched public job config")

public_databricks_job.DEFAULT_AWS_G5_NODE_TYPE = "g6.8xlarge"
public_databricks_job.DatabricksBenchmarkJobConfig = FakeDatabricksBenchmarkJobConfig
public_databricks_job.validate_aws_g5_node_type = public_validator_should_not_run
public_databricks_job.write_databricks_runner_script = public_runner_writer_should_not_run

import restaurant_kv_serving.databricks_job as legacy_databricks_job

assert legacy_databricks_job.DatabricksBenchmarkJobConfig is not FakeDatabricksBenchmarkJobConfig
assert legacy_databricks_job.DEFAULT_AWS_G5_NODE_TYPE == "g6.8xlarge"
legacy_databricks_job.validate_aws_g5_node_type("g6.8xlarge")

with tempfile.TemporaryDirectory() as raw_tmp:
    tmp_path = Path(raw_tmp)
    output_json = tmp_path / "payload.json"
    runner_path = tmp_path / "run_plan.py"
    exit_code = legacy_databricks_job.main(
        [
            "--plan-json-uri",
            "dbfs:/benchmarks/v1-plan.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_plan.py",
            "--single-user-name",
            {SINGLE_USER_NAME!r},
            "--output-json",
            str(output_json),
            "--runner-script-output",
            str(runner_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["tasks"][0]["new_cluster"]["node_type_id"] == "g6.8xlarge"
    runner_text = runner_path.read_text(encoding="utf-8")
    assert "# unexpected public hook" not in runner_text
    assert "document_kv_cache.benchmark_plan_executor" in runner_text
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_legacy_databricks_job_uses_source_benchmark_config_base_when_public_class_is_mutated_before_import():
    script = f"""
import document_kv_cache.databricks_job as public_databricks_job

public_databricks_job.DatabricksBenchmarkJobConfig.plan_json_uri = property(lambda self: "")

import restaurant_kv_serving.databricks_job as legacy_databricks_job

config = legacy_databricks_job.DatabricksBenchmarkJobConfig(
    plan_json_uri="dbfs:/benchmarks/v1-plan.json",
    runner_python_file="dbfs:/benchmarks/run_plan.py",
    single_user_name={SINGLE_USER_NAME!r},
)
assert config.plan_json_uri == "dbfs:/benchmarks/v1-plan.json"
assert legacy_databricks_job.DatabricksBenchmarkJobConfig.__module__ == "restaurant_kv_serving.databricks_job"
try:
    public_databricks_job.DatabricksBenchmarkJobConfig(
        plan_json_uri="dbfs:/benchmarks/v1-plan.json",
        runner_python_file="dbfs:/benchmarks/run_plan.py",
        single_user_name={SINGLE_USER_NAME!r},
    )
except (AttributeError, ValueError):
    pass
else:
    raise AssertionError("public benchmark config mutation did not affect construction or validation")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_legacy_databricks_job_uses_source_cluster_config_base_when_public_class_is_mutated_before_import():
    script = f"""
import document_kv_cache.databricks_job as public_databricks_job

public_databricks_job.DatabricksSingleNodeG5ClusterConfig.node_type_id = property(lambda self: "g6.8xlarge")

import restaurant_kv_serving.databricks_job as legacy_databricks_job

config = legacy_databricks_job.DatabricksSingleNodeG5ClusterConfig(
    purpose="document-kv-v1-benchmark",
    single_user_name={SINGLE_USER_NAME!r},
)
assert config.node_type_id == "g6.8xlarge"
assert legacy_databricks_job.DatabricksSingleNodeG5ClusterConfig.__module__ == "restaurant_kv_serving.databricks_job"
try:
    public_databricks_job.DatabricksSingleNodeG5ClusterConfig(
        purpose="document-kv-v1-benchmark",
        single_user_name={SINGLE_USER_NAME!r},
    )
except (AttributeError, ValueError):
    pass
else:
    raise AssertionError("public cluster config mutation did not affect construction or validation")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_legacy_databricks_job_handles_public_class_dict_mutation_with_unorderable_keys():
    script = f"""
import document_kv_cache.databricks_job as public_databricks_job


class Key:
    pass


public_databricks_job.DatabricksBenchmarkJobConfig.extra = {{Key(): "a", Key(): "b"}}

import restaurant_kv_serving.databricks_job as legacy_databricks_job

config = legacy_databricks_job.DatabricksBenchmarkJobConfig(
    plan_json_uri="dbfs:/benchmarks/v1-plan.json",
    runner_python_file="dbfs:/benchmarks/run_plan.py",
    single_user_name={SINGLE_USER_NAME!r},
)
assert config.plan_json_uri == "dbfs:/benchmarks/v1-plan.json"
assert legacy_databricks_job.DatabricksBenchmarkJobConfig.__mro__[1] is not public_databricks_job.DatabricksBenchmarkJobConfig
assert not hasattr(legacy_databricks_job.DatabricksBenchmarkJobConfig.__mro__[1], "extra")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_legacy_databricks_job_handles_public_class_mutation_with_bad_equality():
    script = f"""
import document_kv_cache.databricks_job as public_databricks_job


class BadEq:
    def __eq__(self, other):
        raise RuntimeError("bad equality during fingerprint comparison")


public_databricks_job.DatabricksBenchmarkJobConfig.__match_args__ = BadEq()

import restaurant_kv_serving.databricks_job as legacy_databricks_job

config = legacy_databricks_job.DatabricksBenchmarkJobConfig(
    plan_json_uri="dbfs:/benchmarks/v1-plan.json",
    runner_python_file="dbfs:/benchmarks/run_plan.py",
    single_user_name={SINGLE_USER_NAME!r},
)
assert config.plan_json_uri == "dbfs:/benchmarks/v1-plan.json"
assert legacy_databricks_job.DatabricksBenchmarkJobConfig.__mro__[1] is not public_databricks_job.DatabricksBenchmarkJobConfig
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_legacy_databricks_job_handles_public_dataclass_field_default_mutation_with_bad_equality():
    script = f"""
import document_kv_cache.databricks_job as public_databricks_job


class BadEq:
    def __eq__(self, other):
        raise RuntimeError("bad field default equality during fingerprint comparison")


public_databricks_job.DatabricksBenchmarkJobConfig.__dataclass_fields__["run_name"].default = BadEq()

import restaurant_kv_serving.databricks_job as legacy_databricks_job

config = legacy_databricks_job.DatabricksBenchmarkJobConfig(
    plan_json_uri="dbfs:/benchmarks/v1-plan.json",
    runner_python_file="dbfs:/benchmarks/run_plan.py",
    single_user_name={SINGLE_USER_NAME!r},
)
assert config.run_name == "document-kv-v1-benchmark"
assert legacy_databricks_job.DatabricksBenchmarkJobConfig.__mro__[1] is not public_databricks_job.DatabricksBenchmarkJobConfig
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_legacy_databricks_job_handles_public_class_mutation_with_bad_attribute_access():
    script = f"""
import document_kv_cache.databricks_job as public_databricks_job


class BadAttr:
    def __getattr__(self, name):
        raise RuntimeError(f"bad attribute lookup {{name}}")


public_databricks_job.DatabricksBenchmarkJobConfig.extra = BadAttr()

import restaurant_kv_serving.databricks_job as legacy_databricks_job

config = legacy_databricks_job.DatabricksBenchmarkJobConfig(
    plan_json_uri="dbfs:/benchmarks/v1-plan.json",
    runner_python_file="dbfs:/benchmarks/run_plan.py",
    single_user_name={SINGLE_USER_NAME!r},
)
assert config.plan_json_uri == "dbfs:/benchmarks/v1-plan.json"
assert legacy_databricks_job.DatabricksBenchmarkJobConfig.__mro__[1] is not public_databricks_job.DatabricksBenchmarkJobConfig
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_legacy_databricks_job_handles_public_dataclass_fields_metadata_replacement():
    script = f"""
import document_kv_cache.databricks_job as public_databricks_job


class BadFields:
    def __getattr__(self, name):
        raise RuntimeError(f"bad fields attribute lookup {{name}}")


public_databricks_job.DatabricksBenchmarkJobConfig.__dataclass_fields__ = BadFields()

import restaurant_kv_serving.databricks_job as legacy_databricks_job

config = legacy_databricks_job.DatabricksBenchmarkJobConfig(
    plan_json_uri="dbfs:/benchmarks/v1-plan.json",
    runner_python_file="dbfs:/benchmarks/run_plan.py",
    single_user_name={SINGLE_USER_NAME!r},
)
assert config.plan_json_uri == "dbfs:/benchmarks/v1-plan.json"
assert legacy_databricks_job.DatabricksBenchmarkJobConfig.__mro__[1] is not public_databricks_job.DatabricksBenchmarkJobConfig
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_legacy_databricks_job_handles_public_dataclass_init_closure_mutation():
    script = f"""
import document_kv_cache.databricks_job as public_databricks_job

freevars = public_databricks_job.DatabricksBenchmarkJobConfig.__init__.__code__.co_freevars
factory_index = freevars.index("_dflt_custom_tags")
public_databricks_job.DatabricksBenchmarkJobConfig.__init__.__closure__[factory_index].cell_contents = (
    lambda: {{"source": "mutated-public-closure"}}
)

import restaurant_kv_serving.databricks_job as legacy_databricks_job

config = legacy_databricks_job.DatabricksBenchmarkJobConfig(
    plan_json_uri="dbfs:/benchmarks/v1-plan.json",
    runner_python_file="dbfs:/benchmarks/run_plan.py",
    single_user_name={SINGLE_USER_NAME!r},
)
assert config.custom_tags == {{}}
assert legacy_databricks_job.DatabricksBenchmarkJobConfig.__mro__[1] is not public_databricks_job.DatabricksBenchmarkJobConfig
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_legacy_databricks_job_handles_public_class_mutation_with_recursive_function_closure():
    script = f"""
import document_kv_cache.databricks_job as public_databricks_job


def make_recursive():
    def recursive():
        return recursive

    return recursive


public_databricks_job.DatabricksBenchmarkJobConfig.extra = make_recursive()

import restaurant_kv_serving.databricks_job as legacy_databricks_job

config = legacy_databricks_job.DatabricksBenchmarkJobConfig(
    plan_json_uri="dbfs:/benchmarks/v1-plan.json",
    runner_python_file="dbfs:/benchmarks/run_plan.py",
    single_user_name={SINGLE_USER_NAME!r},
)
assert config.plan_json_uri == "dbfs:/benchmarks/v1-plan.json"
assert legacy_databricks_job.DatabricksBenchmarkJobConfig.__mro__[1] is not public_databricks_job.DatabricksBenchmarkJobConfig
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_legacy_databricks_job_direct_writer_respects_legacy_build_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"

    def fake_build(config):
        assert config.plan_json_uri == "dbfs:/benchmarks/v1-plan.json"
        return {"ok": True, "source": "legacy-direct-writer-hook"}

    monkeypatch.setattr(legacy_databricks_job, "build_databricks_run_submit_payload", fake_build)

    legacy_databricks_job.write_databricks_run_submit_json(
        legacy_databricks_job.DatabricksBenchmarkJobConfig(
            plan_json_uri="dbfs:/benchmarks/v1-plan.json",
            runner_python_file="dbfs:/benchmarks/run_plan.py",
            single_user_name=SINGLE_USER_NAME,
        ),
        output_path,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "ok": True,
        "source": "legacy-direct-writer-hook",
    }


def test_legacy_databricks_job_restores_document_hooks_after_error(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"
    original_public_build = public_databricks_job.build_databricks_run_submit_payload

    def broken_build(config):
        raise RuntimeError(f"boom for {config.plan_json_uri}")

    monkeypatch.setattr(legacy_databricks_job, "build_databricks_run_submit_payload", broken_build)

    exit_code = legacy_databricks_job.main(
        [
            "--plan-json-uri",
            "dbfs:/benchmarks/v1-plan.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_plan.py",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 1
    assert public_databricks_job.build_databricks_run_submit_payload is original_public_build


def test_legacy_databricks_job_module_execution_shows_help():
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }

    result = subprocess.run(
        [sys.executable, "-m", "restaurant_kv_serving.databricks_job", "--help"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "Emit a Databricks runs/submit payload for a V1 AWS single-node GPU benchmark." in result.stdout


def test_legacy_databricks_job_reexports_document_owned_types():
    assert issubclass(
        legacy_databricks_job.DatabricksBenchmarkJobConfig,
        public_databricks_job.DatabricksBenchmarkJobConfig,
    )
    assert issubclass(
        legacy_databricks_job.DatabricksSingleNodeG5ClusterConfig,
        public_databricks_job.DatabricksSingleNodeG5ClusterConfig,
    )
    assert (
        public_databricks_job.DatabricksBenchmarkJobConfig.__module__
        == "document_kv_cache.databricks_job"
    )
    assert (
        legacy_databricks_job.DatabricksBenchmarkJobConfig.__module__
        == "restaurant_kv_serving.databricks_job"
    )
    assert set(public_databricks_job.__all__) < set(legacy_databricks_job.__all__)
    assert legacy_databricks_job.RESERVED_SINGLE_NODE_GPU_TAG_KEYS is legacy_databricks_job.RESERVED_SINGLE_NODE_G5_TAG_KEYS
    assert (
        legacy_databricks_job.DatabricksSingleNodeGPUClusterConfig
        is legacy_databricks_job.DatabricksSingleNodeG5ClusterConfig
    )
    assert legacy_databricks_job.validate_aws_single_node_gpu_type is legacy_databricks_job.validate_aws_g5_node_type
    assert legacy_databricks_job.build_single_node_gpu_cluster is legacy_databricks_job.build_single_node_g5_cluster


def test_legacy_databricks_job_config_pickle_uses_honest_legacy_module():
    config = legacy_databricks_job.DatabricksBenchmarkJobConfig(
        plan_json_uri="dbfs:/benchmarks/v1-plan.json",
        runner_python_file="dbfs:/benchmarks/run_plan.py",
        single_user_name=SINGLE_USER_NAME,
    )

    restored = pickle.loads(pickle.dumps(config))

    assert type(restored) is legacy_databricks_job.DatabricksBenchmarkJobConfig
    assert restored == config


def test_legacy_databricks_job_config_keeps_slotted_layout():
    config = legacy_databricks_job.DatabricksBenchmarkJobConfig(
        plan_json_uri="dbfs:/benchmarks/v1-plan.json",
        runner_python_file="dbfs:/benchmarks/run_plan.py",
        single_user_name=SINGLE_USER_NAME,
    )

    assert not hasattr(config, "__dict__")


def test_legacy_databricks_job_keeps_previous_star_import_surface():
    assert set(legacy_databricks_job.__all__) == {
        "Any",
        "DEDICATED_DATABRICKS_DATA_SECURITY_MODE",
        "DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE",
        "DEFAULT_AWS_G5_NODE_TYPE",
        "DEFAULT_DATABRICKS_DATA_SECURITY_MODE",
        "DEFAULT_DATABRICKS_PURPOSE",
        "DEFAULT_DATABRICKS_RUN_NAME",
        "DEFAULT_DATABRICKS_SPARK_VERSION",
        "DEFAULT_DATABRICKS_TASK_KEY",
        "DatabricksBenchmarkJobConfig",
        "DatabricksSingleNodeG5ClusterConfig",
        "DatabricksSingleNodeGPUClusterConfig",
        "Mapping",
        "Path",
        "RESERVED_SINGLE_NODE_G5_TAG_KEYS",
        "RESERVED_SINGLE_NODE_GPU_TAG_KEYS",
        "RUNNER_SCRIPT",
        "SINGLE_USER_DATABRICKS_DATA_SECURITY_MODES",
        "Sequence",
        "argparse",
        "build_databricks_run_submit_payload",
        "build_single_node_g5_cluster",
        "build_single_node_gpu_cluster",
        "dataclass",
        "field",
        "json",
        "main",
        "validate_aws_g5_node_type",
        "validate_aws_single_node_gpu_type",
        "write_databricks_run_submit_json",
        "write_databricks_runner_script",
    }


def test_legacy_databricks_job_star_import_uses_previous_surface():
    namespace: dict[str, object] = {}

    exec("from restaurant_kv_serving.databricks_job import *", namespace)

    assert {key for key in namespace if key != "__builtins__"} == set(legacy_databricks_job.__all__)
    assert namespace["DatabricksBenchmarkJobConfig"] is legacy_databricks_job.DatabricksBenchmarkJobConfig


def test_databricks_asset_bundle_template_matches_v1_g5_contract():
    bundle_text = REPO_BUNDLE_TEMPLATE.read_text(encoding="utf-8")
    packaged_bundle_text = PACKAGED_BUNDLE_TEMPLATE.read_text(encoding="utf-8")
    readme_text = (REPO_ROOT / "databricks" / "README.md").read_text(encoding="utf-8")
    root_readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert packaged_bundle_text == bundle_text

    bundle = _parse_simple_yaml(bundle_text)
    variables = bundle["variables"]
    jobs = bundle["resources"]["jobs"]
    task = jobs["document_kv_v1_benchmark"]["tasks"][0]
    cluster = task["new_cluster"]

    assert bundle["bundle"]["name"] == "document-kv-cache-v1"
    assert set(variables) == {
        "plan_json_uri",
        "runner_python_file",
        "wheel_uri",
        "execution_result_json_uri",
        "vllm_native_probe_delegate_factory",
        "sglang_native_probe_delegate_factory",
        "node_type_id",
        "spark_version",
        "single_user_name",
    }
    assert variables["node_type_id"]["default"] == "g6.8xlarge"
    assert variables["vllm_native_probe_delegate_factory"]["default"] == '""'
    assert variables["sglang_native_probe_delegate_factory"]["default"] == '""'
    assert variables["spark_version"]["default"] == "15.4.x-gpu-ml-scala2.12"
    assert "data_security_mode" not in variables
    assert variables["single_user_name"]["default"] == "${workspace.current_user.userName}"
    assert "UC Volume or workspace file path" in variables["wheel_uri"]["description"]
    assert set(jobs) == {"document_kv_v1_benchmark"}
    assert jobs["document_kv_v1_benchmark"]["name"] == "document-kv-v1-benchmark"
    assert task["task_key"] == "document_kv_v1_benchmark"
    assert cluster["spark_version"] == "${var.spark_version}"
    assert cluster["node_type_id"] == "${var.node_type_id}"
    assert cluster["driver_node_type_id"] == "${var.node_type_id}"
    assert cluster["data_security_mode"] == "SINGLE_USER"
    assert cluster["single_user_name"] == "${var.single_user_name}"
    assert cluster["num_workers"] == 0
    assert cluster["spark_conf"] == {
        "spark.master": "local[*]",
        "spark.databricks.cluster.profile": "singleNode",
    }
    assert cluster["custom_tags"]["ResourceClass"] == "SingleNode"
    assert cluster["custom_tags"]["purpose"] == DEFAULT_DATABRICKS_PURPOSE
    assert cluster["spark_env_vars"] == {
        "DOCUMENT_KV_VLLM_NATIVE_PROBE_FACTORY": "${var.vllm_native_probe_delegate_factory}",
        "DOCUMENT_KV_SGLANG_NATIVE_PROBE_FACTORY": "${var.sglang_native_probe_delegate_factory}",
    }
    assert cluster["aws_attributes"] == {"availability": "ON_DEMAND", "zone_id": "auto"}
    assert "libraries" not in task
    assert task["spark_python_task"] == {
        "python_file": "${var.runner_python_file}",
        "parameters": [
            "--plan-json",
            "${var.plan_json_uri}",
            "--result-json",
            "${var.execution_result_json_uri}",
            "--package-wheel-uri",
            "${var.wheel_uri}",
        ],
    }

    assert "Databricks Asset Bundle" in readme_text
    assert "cd databricks" in readme_text
    assert "dbfs:/benchmarks/document_kv_cache" not in readme_text
    assert "dbfs:/benchmarks/document_kv_cache" not in root_readme_text
    assert WHEEL_URI in readme_text
    assert WHEEL_URI in root_readme_text
    assert "document-kv-databricks-job" in readme_text
    assert "document-kv-vllm-smoke-databricks-job" in readme_text
    assert "--vllm-native-probe-delegate-factory" in readme_text
    assert "--sglang-native-probe-delegate-factory" in readme_text
    assert f"--engine-probe-metadata vllm={VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA}" in readme_text
    assert f"--engine-probe-metadata vllm={VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA}" in root_readme_text
    assert "--var vllm_native_probe_delegate_factory=" in readme_text
    assert "--var sglang_native_probe_delegate_factory=" in readme_text
    assert "single-node AWS `g6`/L4 GPU cluster" in readme_text
    assert "document_kv_cache/templates/databricks/" in readme_text


def test_databricks_vllm_smoke_asset_bundle_template_is_independent():
    bundle_text = REPO_VLLM_SMOKE_BUNDLE_TEMPLATE.read_text(encoding="utf-8")
    packaged_bundle_text = PACKAGED_VLLM_SMOKE_BUNDLE_TEMPLATE.read_text(encoding="utf-8")
    readme_text = (REPO_ROOT / "databricks" / "README.md").read_text(encoding="utf-8")
    smoke_readme_text = (REPO_ROOT / "databricks" / "vllm-smoke" / "README.md").read_text(encoding="utf-8")
    packaged_smoke_readme_text = (
        REPO_ROOT / "src" / "document_kv_cache" / "templates" / "databricks" / "vllm-smoke" / "README.md"
    ).read_text(encoding="utf-8")
    assert packaged_bundle_text == bundle_text

    bundle = _parse_simple_yaml(bundle_text)
    variables = bundle["variables"]
    jobs = bundle["resources"]["jobs"]
    task = jobs["document_kv_vllm_smoke"]["tasks"][0]
    cluster = task["new_cluster"]

    assert bundle["bundle"]["name"] == "document-kv-vllm-smoke"
    assert set(jobs) == {"document_kv_vllm_smoke"}
    assert set(variables) == {
        "runner_python_file",
        "benchmark_id",
        "output_dir",
        "wheel_uri",
        "node_type_id",
        "spark_version",
        "single_user_name",
    }
    assert variables["node_type_id"]["default"] == "g6.8xlarge"
    assert variables["spark_version"]["default"] == "15.4.x-gpu-ml-scala2.12"
    assert variables["single_user_name"]["default"] == "${workspace.current_user.userName}"
    assert jobs["document_kv_vllm_smoke"]["name"] == "document-kv-vllm-smoke"
    assert task["task_key"] == "document_kv_vllm_smoke"
    assert cluster["spark_version"] == "${var.spark_version}"
    assert cluster["node_type_id"] == "${var.node_type_id}"
    assert cluster["driver_node_type_id"] == "${var.node_type_id}"
    assert cluster["data_security_mode"] == "SINGLE_USER"
    assert cluster["single_user_name"] == "${var.single_user_name}"
    assert cluster["num_workers"] == 0
    assert cluster["spark_conf"] == {
        "spark.master": "local[*]",
        "spark.databricks.cluster.profile": "singleNode",
    }
    assert cluster["custom_tags"]["ResourceClass"] == "SingleNode"
    assert cluster["custom_tags"]["purpose"] == "document-kv-vllm-smoke"
    assert cluster["aws_attributes"] == {"availability": "ON_DEMAND", "zone_id": "auto"}
    assert "libraries" not in task
    assert task["spark_python_task"] == {
        "python_file": "${var.runner_python_file}",
        "parameters": [
            "--benchmark-id",
            "${var.benchmark_id}",
            "--output-dir",
            "${var.output_dir}",
            "--package-wheel-uri",
            "${var.wheel_uri}",
        ],
    }

    assert "vllm-smoke/databricks.yml" in readme_text
    assert "cd databricks/vllm-smoke" in readme_text
    assert "does not require full V1 raw datasets" in " ".join(readme_text.split())
    assert "smallest runtime check" in " ".join(smoke_readme_text.split())
    assert "target AWS g6/L4 Databricks runtime" in packaged_smoke_readme_text


def test_databricks_engine_probe_asset_bundle_template_is_independent_and_release_safe():
    bundle_text = REPO_ENGINE_PROBE_BUNDLE_TEMPLATE.read_text(encoding="utf-8")
    packaged_bundle_text = PACKAGED_ENGINE_PROBE_BUNDLE_TEMPLATE.read_text(encoding="utf-8")
    readme_text = (REPO_ROOT / "databricks" / "README.md").read_text(encoding="utf-8")
    probe_readme_text = (REPO_ROOT / "databricks" / "engine-probe" / "README.md").read_text(encoding="utf-8")
    root_readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    module_readme_text = (REPO_ROOT / "src" / "restaurant_kv_serving" / "README.md").read_text(encoding="utf-8")
    packaged_probe_readme_text = (
        REPO_ROOT / "src" / "document_kv_cache" / "templates" / "databricks" / "engine-probe" / "README.md"
    ).read_text(encoding="utf-8")
    assert packaged_bundle_text == bundle_text

    bundle = _parse_simple_yaml(bundle_text)
    variables = bundle["variables"]
    jobs = bundle["resources"]["jobs"]
    task = jobs["document_kv_engine_probe"]["tasks"][0]
    cluster = task["new_cluster"]

    assert bundle["bundle"]["name"] == "document-kv-engine-probe"
    assert set(jobs) == {"document_kv_engine_probe"}
    assert set(variables) == {
        "runner_python_file",
        "handoff_json",
        "probe_factory",
        "probe_output_json",
        "actions_output_json",
        "payload_uri",
        "expected_backend",
        "vllm_native_probe_delegate_factory",
        "sglang_native_probe_delegate_factory",
        "native_probe_metadata",
        "wheel_uri",
        "node_type_id",
        "spark_version",
        "single_user_name",
    }
    assert variables["vllm_native_probe_delegate_factory"]["default"] == '""'
    assert variables["sglang_native_probe_delegate_factory"]["default"] == '""'
    assert VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA in variables["native_probe_metadata"]["description"]
    assert variables["node_type_id"]["default"] == "g6.8xlarge"
    assert variables["spark_version"]["default"] == "15.4.x-gpu-ml-scala2.12"
    assert variables["single_user_name"]["default"] == "${workspace.current_user.userName}"
    assert jobs["document_kv_engine_probe"]["name"] == "document-kv-engine-probe"
    assert task["task_key"] == "document_kv_engine_probe"
    assert cluster["spark_version"] == "${var.spark_version}"
    assert cluster["node_type_id"] == "${var.node_type_id}"
    assert cluster["driver_node_type_id"] == "${var.node_type_id}"
    assert cluster["data_security_mode"] == "SINGLE_USER"
    assert cluster["single_user_name"] == "${var.single_user_name}"
    assert cluster["num_workers"] == 0
    assert cluster["spark_conf"] == {
        "spark.master": "local[*]",
        "spark.databricks.cluster.profile": "singleNode",
    }
    assert cluster["custom_tags"]["ResourceClass"] == "SingleNode"
    assert cluster["custom_tags"]["purpose"] == "document-kv-engine-probe"
    assert cluster["spark_env_vars"] == {
        "DOCUMENT_KV_VLLM_NATIVE_PROBE_FACTORY": "${var.vllm_native_probe_delegate_factory}",
        "DOCUMENT_KV_SGLANG_NATIVE_PROBE_FACTORY": "${var.sglang_native_probe_delegate_factory}",
    }
    assert cluster["aws_attributes"] == {"availability": "ON_DEMAND", "zone_id": "auto"}
    assert "libraries" not in task
    assert task["spark_python_task"] == {
        "python_file": "${var.runner_python_file}",
        "parameters": [
            "--handoff-json",
            "${var.handoff_json}",
            "--probe-factory",
            "${var.probe_factory}",
            "--output-json",
            "${var.probe_output_json}",
            "--actions-output-json",
            "${var.actions_output_json}",
            "--payload-uri",
            "${var.payload_uri}",
            "--expected-backend",
            "${var.expected_backend}",
            "--metadata",
            "${var.native_probe_metadata}",
            "--package-wheel-uri",
            "${var.wheel_uri}",
        ],
    }
    assert "--allow-non-native-probe" not in bundle_text
    assert "--engine-version" not in bundle_text

    assert "engine-probe/databricks.yml" in readme_text
    assert "cd databricks/engine-probe" in readme_text
    assert "--provider-backed-vllm-native-probe" in readme_text
    assert "--provider-backed-vllm-native-probe" in root_readme_text
    assert "--provider-backed-vllm-native-probe" in module_readme_text
    assert "--fixture-output-dir /Volumes/catalog/schema/volume/probes/vllm-fixture" in readme_text
    assert "--fixture-output-dir /Volumes/catalog/schema/volume/probes/vllm-fixture" in root_readme_text
    assert "--fixture-output-dir /Volumes/catalog/schema/volume/probes/vllm-fixture" in module_readme_text
    assert "--var probe_factory=document_kv_cache.native_probe_factories:vllm_native_probe_factory" in readme_text
    assert (
        "--actions-output-json /Volumes/catalog/schema/volume/probes/vllm-fixture/qwen3-v1-fixture.actions.json"
        in readme_text
    )
    assert (
        "--actions-output-json /Volumes/catalog/schema/volume/probes/vllm-fixture/qwen3-v1-fixture.actions.json"
        in root_readme_text
    )
    assert "--var payload_uri=" in readme_text
    assert "--var payload_uri=" in probe_readme_text
    assert "--var actions_output_json=" in readme_text
    assert "--var actions_output_json=" in probe_readme_text
    assert "--var native_probe_metadata=" in readme_text
    assert "--var native_probe_metadata=" in probe_readme_text
    assert "--var vllm_native_probe_delegate_factory=" in probe_readme_text
    assert VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA in probe_readme_text
    sglang_connector_example = (
        "sglang_kv_injection.connector_factory=company_sglang_patch.probe:build_connector"
    )
    assert sglang_connector_example in probe_readme_text
    assert sglang_connector_example in packaged_probe_readme_text
    assert "placeholder connector factory metadata" in probe_readme_text
    assert "placeholder connector factory metadata" in packaged_probe_readme_text
    assert "sglang_kv_injection.connector_factory=module:factory" not in probe_readme_text
    assert "sglang_kv_injection.connector_factory=module:factory" not in packaged_probe_readme_text
    assert "DOCUMENT_KV_VLLM_NATIVE_PROBE_FACTORY" in probe_readme_text
    assert "DOCUMENT_KV_SGLANG_NATIVE_PROBE_FACTORY" in probe_readme_text
    assert "native vLLM or SGLang" in probe_readme_text
    assert "target AWS g6/L4 Databricks runtime" in packaged_probe_readme_text
    assert "uploaded payload URI" in packaged_probe_readme_text
    assert "document_kv.engine_kv_connector_actions.v1" in packaged_probe_readme_text
    assert "DOCUMENT_KV_VLLM_NATIVE_PROBE_FACTORY" in packaged_probe_readme_text


def _parse_simple_yaml(text: str) -> dict:
    lines = [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    parsed, next_index = _parse_yaml_block(lines, 0, 0)
    assert next_index == len(lines)
    assert isinstance(parsed, dict)
    return parsed


def _parse_yaml_block(lines: list[str], index: int, indent: int):
    if index >= len(lines):
        return {}, index
    stripped = lines[index][indent:]
    if stripped.startswith("- "):
        return _parse_yaml_list(lines, index, indent)
    return _parse_yaml_mapping(lines, index, indent)


def _parse_yaml_mapping(lines: list[str], index: int, indent: int) -> tuple[dict, int]:
    result = {}
    while index < len(lines):
        line = lines[index]
        current_indent = _yaml_indent(line)
        if current_indent < indent:
            break
        if current_indent > indent:
            raise AssertionError(f"Unexpected YAML indentation: {line!r}")
        content = line[indent:]
        if content.startswith("- "):
            break
        key, value = _split_yaml_key_value(content)
        if key in result:
            raise AssertionError(f"Duplicate YAML key {key!r}")
        index += 1
        if value == "":
            child, index = _parse_yaml_block(lines, index, indent + 2)
            result[key] = child
        else:
            result[key] = _yaml_scalar(value)
    return result, index


def _parse_yaml_list(lines: list[str], index: int, indent: int) -> tuple[list, int]:
    result = []
    while index < len(lines):
        line = lines[index]
        current_indent = _yaml_indent(line)
        if current_indent < indent:
            break
        if current_indent != indent:
            raise AssertionError(f"Unexpected YAML list indentation: {line!r}")
        content = line[indent:]
        if not content.startswith("- "):
            break
        item_content = content[2:]
        index += 1
        if ":" in item_content:
            key, value = _split_yaml_key_value(item_content)
            item = {key: _yaml_scalar(value)} if value else {key: {}}
            if index < len(lines) and _yaml_indent(lines[index]) > indent:
                child, index = _parse_yaml_mapping(lines, index, indent + 2)
                if item[key] == {} and set(item) == {key}:
                    item[key] = child
                else:
                    duplicate_keys = set(item).intersection(child)
                    if duplicate_keys:
                        raise AssertionError(f"Duplicate YAML keys in list item: {sorted(duplicate_keys)}")
                    item.update(child)
            result.append(item)
        else:
            result.append(_yaml_scalar(item_content))
    return result, index


def _split_yaml_key_value(content: str) -> tuple[str, str]:
    key, separator, value = content.partition(":")
    if not separator:
        raise AssertionError(f"Expected YAML key/value line: {content!r}")
    return key.strip(), value.strip()


def _yaml_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _yaml_scalar(value: str):
    if value.isdecimal():
        return int(value)
    return value
