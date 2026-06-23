import json
import os
import pickle
from pathlib import Path
import subprocess
import sys

import pytest

import document_kv_cache.databricks_job as public_databricks_job
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
    SGLANG_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,
    VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,
)
from document_kv_cache._hardware_targets import (
    SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES,
    V1_HARDWARE_TARGET_PROFILE,
    databricks_node_type_for_hardware_target,
    default_databricks_node_type_for_hardware_target,
    validate_aws_single_node_gpu_type_for_hardware_target,
)
from document_kv_cache.native_probe_factories import (
    SGLANG_NATIVE_PROBE_DELEGATE_ENV,
    VLLM_NATIVE_PROBE_DELEGATE_ENV,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WHEEL_URI = "/Volumes/catalog/schema/volume/wheels/cachet_kv-0.2.0-py3-none-any.whl"
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


def test_build_databricks_run_submit_payload_sets_generator_runtime_env_vars():
    config = DatabricksBenchmarkJobConfig(
        plan_json_uri="dbfs:/benchmarks/v1-plan.json",
        runner_python_file="dbfs:/benchmarks/run_plan.py",
        single_user_name=SINGLE_USER_NAME,
        vllm_native_probe_delegate_factory="document_kv_vllm_native_adapter:build_probe",
        spark_env_vars={
            "CACHET_TRANSFORMERS_MODEL_ID": "Qwen/Qwen3-4B-Instruct-2507",
            "CACHET_TRANSFORMERS_TOKENIZER_ID": "Qwen/Qwen3-4B-Instruct-2507",
            "CACHET_TRANSFORMERS_DEVICE": "cuda",
            "CACHET_TRANSFORMERS_TORCH_DTYPE": "bfloat16",
            "CACHET_TRANSFORMERS_MODEL_KWARGS_JSON": '{"attn_implementation":"eager"}',
        },
    )

    payload = build_databricks_run_submit_payload(config)
    cluster = payload["tasks"][0]["new_cluster"]

    assert cluster["spark_env_vars"] == {
        "CACHET_TRANSFORMERS_MODEL_ID": "Qwen/Qwen3-4B-Instruct-2507",
        "CACHET_TRANSFORMERS_TOKENIZER_ID": "Qwen/Qwen3-4B-Instruct-2507",
        "CACHET_TRANSFORMERS_DEVICE": "cuda",
        "CACHET_TRANSFORMERS_TORCH_DTYPE": "bfloat16",
        "CACHET_TRANSFORMERS_MODEL_KWARGS_JSON": '{"attn_implementation":"eager"}',
        VLLM_NATIVE_PROBE_DELEGATE_ENV: "document_kv_vllm_native_adapter:build_probe",
    }


def test_databricks_run_submit_payload_rejects_secret_like_spark_env_vars():
    kwargs = {
        "plan_json_uri": "dbfs:/benchmarks/v1-plan.json",
        "runner_python_file": "dbfs:/benchmarks/run_plan.py",
        "single_user_name": SINGLE_USER_NAME,
    }
    token_like_value = "dapi" + "0" * 32

    with pytest.raises(ValueError, match="valid environment variable name"):
        DatabricksBenchmarkJobConfig(**kwargs, spark_env_vars={"BAD-NAME": "value"})
    with pytest.raises(ValueError, match="secret-bearing"):
        DatabricksBenchmarkJobConfig(**kwargs, spark_env_vars={"DATABRICKS_TOKEN": "redacted"})
    with pytest.raises(ValueError, match="Databricks token pattern"):
        DatabricksBenchmarkJobConfig(
            **kwargs,
            spark_env_vars={"CACHET_TRANSFORMERS_DEVICE": token_like_value},
        )
    with pytest.raises(ValueError, match="reserved native-probe key"):
        DatabricksBenchmarkJobConfig(
            **kwargs,
            spark_env_vars={VLLM_NATIVE_PROBE_DELEGATE_ENV: "document_kv_vllm_native_adapter:build_probe"},
        )
    with pytest.raises(ValueError, match="JSON object"):
        DatabricksBenchmarkJobConfig(
            **kwargs,
            spark_env_vars={"CACHET_TRANSFORMERS_MODEL_KWARGS_JSON": "not-json"},
        )
    with pytest.raises(ValueError, match="secret-bearing"):
        DatabricksBenchmarkJobConfig(
            **kwargs,
            spark_env_vars={"CACHET_TRANSFORMERS_MODEL_KWARGS_JSON": '{"api_key":"redacted"}'},
        )


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


def test_hardware_target_node_type_helpers_map_v1_databricks_defaults():
    assert default_databricks_node_type_for_hardware_target("aws-g6-l4") == "g6.8xlarge"
    assert default_databricks_node_type_for_hardware_target("aws-g5-a10g") == "g5.8xlarge"
    assert databricks_node_type_for_hardware_target() == "g6.8xlarge"
    assert databricks_node_type_for_hardware_target(node_type_id="g5.8xlarge") == "g5.8xlarge"
    assert databricks_node_type_for_hardware_target("aws-g5-a10g") == "g5.8xlarge"
    assert databricks_node_type_for_hardware_target("aws-g5-a10g", "g5.12xlarge") == "g5.12xlarge"

    validate_aws_single_node_gpu_type_for_hardware_target("g6.8xlarge", "aws-g6-l4")
    with pytest.raises(ValueError, match="hardware target 'aws-g5-a10g'"):
        validate_aws_single_node_gpu_type_for_hardware_target("g6.8xlarge", "aws-g5-a10g")


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
            "dbfs:/tmp/cachet/cachet_kv-0.2.0-py3-none-any.whl",
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
        "/dbfs/tmp/cachet/cachet_kv-0.2.0-py3-none-any.whl",
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
            "--spark-env-var",
            "CACHET_TRANSFORMERS_DEVICE=cuda",
            "--spark-env-var",
            "CACHET_TRANSFORMERS_TOKENIZER_ID=Qwen/Qwen3-4B-Instruct-2507",
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
        "CACHET_TRANSFORMERS_DEVICE": "cuda",
        "CACHET_TRANSFORMERS_TOKENIZER_ID": "Qwen/Qwen3-4B-Instruct-2507",
        VLLM_NATIVE_PROBE_DELEGATE_ENV: "document_kv_vllm_native_adapter:build_probe",
        SGLANG_NATIVE_PROBE_DELEGATE_ENV: "document_kv_sglang_native_adapter:build_probe",
    }
    assert "--vllm-native-probe-delegate-factory" not in task["spark_python_task"]["parameters"]
    assert "--sglang-native-probe-delegate-factory" not in task["spark_python_task"]["parameters"]
    assert "benchmark_plan_executor" in runner_path.read_text(encoding="utf-8")


def test_main_derives_node_type_from_g5_hardware_target(tmp_path):
    payload_path = tmp_path / "payload.json"

    exit_code = main(
        [
            "--plan-json-uri",
            "dbfs:/benchmarks/v1-plan.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_plan.py",
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


def test_main_preserves_legacy_g5_node_type_without_hardware_target(tmp_path):
    payload_path = tmp_path / "payload.json"

    exit_code = main(
        [
            "--plan-json-uri",
            "dbfs:/benchmarks/v1-plan.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_plan.py",
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


def test_main_rejects_node_type_that_does_not_match_hardware_target(tmp_path, capsys):
    payload_path = tmp_path / "payload.json"

    exit_code = main(
        [
            "--plan-json-uri",
            "dbfs:/benchmarks/v1-plan.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_plan.py",
            "--hardware-target",
            "aws-g5-a10g",
            "--node-type-id",
            "g6.8xlarge",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(payload_path),
        ]
    )

    assert exit_code == 1
    assert not payload_path.exists()
    assert "hardware target 'aws-g5-a10g'" in json.loads(capsys.readouterr().out)["error"]


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

    assert bundle["bundle"]["name"] == "cachet-v1"
    assert set(variables) == {
        "plan_json_uri",
        "runner_python_file",
        "wheel_uri",
        "execution_result_json_uri",
        "vllm_native_probe_delegate_factory",
        "sglang_native_probe_delegate_factory",
        "transformers_model_id",
        "transformers_tokenizer_id",
        "transformers_device",
        "transformers_torch_dtype",
        "transformers_trust_remote_code",
        "transformers_add_special_tokens",
        "transformers_cache_axis_order",
        "node_type_id",
        "spark_version",
        "single_user_name",
    }
    assert variables["node_type_id"]["default"] == "g6.8xlarge"
    assert variables["vllm_native_probe_delegate_factory"]["default"] == '""'
    assert variables["sglang_native_probe_delegate_factory"]["default"] == '""'
    assert variables["transformers_model_id"]["default"] == '""'
    assert "non-secret CACHET_TRANSFORMERS_MODEL_ID" in variables["transformers_model_id"]["description"]
    assert variables["transformers_tokenizer_id"]["default"] == '""'
    assert variables["transformers_device"]["default"] == '""'
    assert variables["transformers_torch_dtype"]["default"] == '""'
    assert variables["transformers_trust_remote_code"]["default"] == '""'
    assert variables["transformers_add_special_tokens"]["default"] == '""'
    assert variables["transformers_cache_axis_order"]["default"] == '""'
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
        "CACHET_TRANSFORMERS_MODEL_ID": "${var.transformers_model_id}",
        "CACHET_TRANSFORMERS_TOKENIZER_ID": "${var.transformers_tokenizer_id}",
        "CACHET_TRANSFORMERS_DEVICE": "${var.transformers_device}",
        "CACHET_TRANSFORMERS_TORCH_DTYPE": "${var.transformers_torch_dtype}",
        "CACHET_TRANSFORMERS_TRUST_REMOTE_CODE": "${var.transformers_trust_remote_code}",
        "CACHET_TRANSFORMERS_ADD_SPECIAL_TOKENS": "${var.transformers_add_special_tokens}",
        "CACHET_TRANSFORMERS_CACHE_AXIS_ORDER": "${var.transformers_cache_axis_order}",
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
        "hardware_target",
        "spark_version",
        "single_user_name",
    }
    assert variables["node_type_id"]["default"] == "g6.8xlarge"
    assert variables["hardware_target"]["default"] == "aws-g6-l4"
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
            "--hardware-target",
            "${var.hardware_target}",
            "--package-wheel-uri",
            "${var.wheel_uri}",
        ],
    }

    assert "vllm-smoke/databricks.yml" in readme_text
    assert "cd databricks/vllm-smoke" in readme_text
    assert "does not require full V1 raw datasets" in " ".join(readme_text.split())
    assert "smallest runtime check" in " ".join(smoke_readme_text.split())
    assert "target AWS g6/L4 Databricks runtime" in packaged_smoke_readme_text


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
