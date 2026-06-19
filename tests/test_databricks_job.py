import json
from pathlib import Path

import pytest

import document_kv_cache.databricks_job as public_databricks_job
import restaurant_kv_serving.databricks_job as legacy_databricks_job
from document_kv_cache.databricks_job import (
    DEDICATED_DATABRICKS_DATA_SECURITY_MODE,
    DatabricksBenchmarkJobConfig,
    DatabricksSingleNodeG5ClusterConfig,
    build_single_node_g5_cluster,
    build_databricks_run_submit_payload,
    main,
    validate_aws_g5_node_type,
    write_databricks_runner_script,
    write_databricks_run_submit_json,
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
    assert cluster["node_type_id"] == "g5.4xlarge"
    assert cluster["driver_node_type_id"] == "g5.4xlarge"
    assert cluster["data_security_mode"] == "SINGLE_USER"
    assert cluster["single_user_name"] == SINGLE_USER_NAME
    assert cluster["num_workers"] == 0
    assert cluster["spark_conf"]["spark.databricks.cluster.profile"] == "singleNode"
    assert cluster["aws_attributes"] == {"availability": "ON_DEMAND", "zone_id": "auto"}
    assert cluster["custom_tags"]["ResourceClass"] == "SingleNode"
    assert cluster["custom_tags"]["team"] == "document-kv"
    assert task["spark_python_task"] == {
        "python_file": "dbfs:/benchmarks/run_plan.py",
        "parameters": [
            "--plan-json",
            "dbfs:/benchmarks/v1-plan.json",
            "--result-json",
            "dbfs:/benchmarks/result.json",
        ],
    }
    assert task["libraries"] == [{"whl": WHEEL_URI}]


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


def test_validate_aws_g5_node_type_rejects_other_gpu_families():
    validate_aws_g5_node_type("g5.xlarge")

    with pytest.raises(ValueError, match="AWS g5"):
        validate_aws_g5_node_type("g6.8xlarge")

    with pytest.raises(ValueError, match="AWS g5"):
        validate_aws_g5_node_type("g6e.8xlarge")


def test_write_databricks_runner_script_imports_plan_executor(tmp_path):
    path = tmp_path / "run_plan.py"

    write_databricks_runner_script(path)

    runner_text = path.read_text(encoding="utf-8")
    assert "document_kv_cache.benchmark_plan_executor" in runner_text
    assert "raise SystemExit(main())" not in runner_text
    assert "if exit_code:" in runner_text


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
            "--output-json",
            str(payload_path),
            "--runner-script-output",
            str(runner_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(payload_path.read_text(encoding="utf-8"))["tasks"][0]["libraries"]
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
    assert variables["node_type_id"]["default"] == "g5.4xlarge"
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
    assert cluster["custom_tags"]["purpose"] == "document-kv-v1-benchmark"
    assert cluster["aws_attributes"] == {"availability": "ON_DEMAND", "zone_id": "auto"}
    assert task["libraries"] == [{"whl": "${var.wheel_uri}"}]
    assert task["spark_python_task"] == {
        "python_file": "${var.runner_python_file}",
        "parameters": [
            "--plan-json",
            "${var.plan_json_uri}",
            "--result-json",
            "${var.execution_result_json_uri}",
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
    assert "single-node AWS `g5` GPU cluster" in readme_text
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
    assert variables["node_type_id"]["default"] == "g5.4xlarge"
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
    assert task["libraries"] == [{"whl": "${var.wheel_uri}"}]
    assert task["spark_python_task"] == {
        "python_file": "${var.runner_python_file}",
        "parameters": [
            "--benchmark-id",
            "${var.benchmark_id}",
            "--output-dir",
            "${var.output_dir}",
        ],
    }

    assert "vllm-smoke/databricks.yml" in readme_text
    assert "cd databricks/vllm-smoke" in readme_text
    assert "does not require full V1 raw datasets" in " ".join(readme_text.split())
    assert "smallest runtime check" in " ".join(smoke_readme_text.split())
    assert "target AWS g5 Databricks runtime" in packaged_smoke_readme_text


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
        "payload_uri",
        "expected_backend",
        "wheel_uri",
        "node_type_id",
        "spark_version",
        "single_user_name",
    }
    assert variables["node_type_id"]["default"] == "g5.4xlarge"
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
    assert cluster["aws_attributes"] == {"availability": "ON_DEMAND", "zone_id": "auto"}
    assert task["libraries"] == [{"whl": "${var.wheel_uri}"}]
    assert task["spark_python_task"] == {
        "python_file": "${var.runner_python_file}",
        "parameters": [
            "--handoff-json",
            "${var.handoff_json}",
            "--probe-factory",
            "${var.probe_factory}",
            "--output-json",
            "${var.probe_output_json}",
            "--payload-uri",
            "${var.payload_uri}",
            "--expected-backend",
            "${var.expected_backend}",
        ],
    }
    assert "--allow-non-native-probe" not in bundle_text
    assert "--engine-version" not in bundle_text

    assert "engine-probe/databricks.yml" in readme_text
    assert "cd databricks/engine-probe" in readme_text
    assert "--payload-uri /Volumes/catalog/schema/volume/probes/vllm-payload.kv" in readme_text
    assert "--payload-uri /Volumes/catalog/schema/volume/probes/vllm-payload.kv" in root_readme_text
    assert "--payload-uri /Volumes/catalog/schema/volume/probes/vllm-payload.kv" in module_readme_text
    assert "--var payload_uri=" in readme_text
    assert "--var payload_uri=" in probe_readme_text
    assert "native vLLM or SGLang" in probe_readme_text
    assert "target AWS g5 Databricks runtime" in packaged_probe_readme_text
    assert "uploaded payload URI" in packaged_probe_readme_text


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
