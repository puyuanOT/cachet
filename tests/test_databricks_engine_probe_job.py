import json
import os
import pickle
from pathlib import Path
import subprocess
import sys

import pytest

import document_kv_cache.databricks_engine_probe_job as public_engine_probe_job
import restaurant_kv_serving.databricks_engine_probe_job as legacy_engine_probe_job
from document_kv_cache.databricks_engine_probe_job import (
    DEFAULT_DATABRICKS_ENGINE_PROBE_BACKEND_CONFIG_KEY,
    DEFAULT_DATABRICKS_ENGINE_PROBE_PURPOSE,
    DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME,
    DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY,
    DatabricksEngineProbeJobConfig,
    DatabricksEngineProbeMatrixJobConfig,
    DatabricksEngineProbeTargetConfig,
    DatabricksEngineProbeTargetsFile,
    build_databricks_engine_probe_matrix_run_submit_payload,
    build_databricks_engine_probe_run_submit_payload,
    main,
    read_databricks_engine_probe_targets_file_json,
    read_databricks_engine_probe_targets_json,
    run_engine_probe_task,
    write_databricks_engine_probe_matrix_run_submit_json,
    write_databricks_engine_probe_run_submit_json,
    write_databricks_engine_probe_runner_script,
)
from document_kv_cache.engine_adapters import ServingBackend
from document_kv_cache.native_probe_factories import (
    SGLANG_NATIVE_PROBE_DELEGATE_ENV,
    VLLM_NATIVE_PROBE_DELEGATE_ENV,
)
from document_kv_cache.probe_fixtures import DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES


WHEEL_URI = "/Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl"
VLLM_ADAPTER_WHEEL_URI = "/Volumes/catalog/schema/volume/wheels/vllm_kv_injection-0.2.0-py3-none-any.whl"
SGLANG_ADAPTER_WHEEL_URI = "/Volumes/catalog/schema/volume/wheels/sglang_kv_injection-0.2.0-py3-none-any.whl"
SINGLE_USER_NAME = "user@example.com"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _target(backend: str, **overrides):
    values = {
        "expected_backend": backend,
        "handoff_json": f"/Volumes/catalog/schema/volume/probes/{backend}-handoff.json",
        "probe_factory": f"document_kv_cache_{backend}_probe:build_probe",
        "output_json": f"/Volumes/catalog/schema/volume/probes/{backend}-probe.json",
        "payload_uri": f"/Volumes/catalog/schema/volume/probes/{backend}-payload.kv",
    }
    values.update(overrides)
    return DatabricksEngineProbeTargetConfig(**values)


def test_build_databricks_engine_probe_payload_uses_single_node_g5_cluster():
    config = DatabricksEngineProbeJobConfig(
        handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        probe_factory="document_kv_cache_vllm_probe:build_probe",
        output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        expected_backend=ServingBackend.VLLM,
        payload_uri="/Volumes/catalog/schema/volume/probes/vllm-payload.kv",
        node_type_id="g5.8xlarge",
        wheel_uri=WHEEL_URI,
        single_user_name=SINGLE_USER_NAME,
        engine_version="debug-vllm",
        allow_non_native_probe=True,
        metadata=("probe.source=single",),
        actions_output_json="/Volumes/catalog/schema/volume/probes/vllm-actions.json",
        custom_tags={"team": "document-kv"},
    )

    payload = build_databricks_engine_probe_run_submit_payload(config)
    task = payload["tasks"][0]
    cluster = task["new_cluster"]

    assert payload["run_name"] == DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME
    assert task["task_key"] == DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY
    assert "libraries" not in task
    assert cluster["node_type_id"] == "g5.8xlarge"
    assert cluster["driver_node_type_id"] == "g5.8xlarge"
    assert cluster["data_security_mode"] == "SINGLE_USER"
    assert cluster["single_user_name"] == SINGLE_USER_NAME
    assert cluster["num_workers"] == 0
    assert cluster["custom_tags"]["ResourceClass"] == "SingleNode"
    assert cluster["custom_tags"]["purpose"] == DEFAULT_DATABRICKS_ENGINE_PROBE_PURPOSE
    assert cluster["custom_tags"]["team"] == "document-kv"
    assert task["spark_python_task"] == {
        "python_file": "dbfs:/benchmarks/run_engine_probe.py",
        "parameters": [
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            "--probe-factory",
            "document_kv_cache_vllm_probe:build_probe",
            "--output-json",
            "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            "--expected-backend",
            "vllm",
            "--actions-output-json",
            "/Volumes/catalog/schema/volume/probes/vllm-actions.json",
            "--payload-uri",
            "/Volumes/catalog/schema/volume/probes/vllm-payload.kv",
            "--engine-version",
            "debug-vllm",
            "--allow-non-native-probe",
            "--metadata",
            "probe.source=single",
            "--package-wheel-uri",
            WHEEL_URI,
        ],
    }


def test_build_databricks_engine_probe_payload_accepts_g6_l4_cluster():
    config = DatabricksEngineProbeJobConfig(
        handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        probe_factory="document_kv_cache_vllm_probe:build_probe",
        output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        expected_backend=ServingBackend.VLLM,
        payload_uri="/Volumes/catalog/schema/volume/probes/vllm-payload.kv",
        node_type_id="g6.8xlarge",
        single_user_name=SINGLE_USER_NAME,
        allow_non_native_probe=True,
    )

    payload = build_databricks_engine_probe_run_submit_payload(config)
    cluster = payload["tasks"][0]["new_cluster"]

    assert cluster["node_type_id"] == "g6.8xlarge"
    assert cluster["driver_node_type_id"] == "g6.8xlarge"


def test_build_databricks_engine_probe_release_safe_payload_omits_debug_flags():
    config = DatabricksEngineProbeJobConfig(
        handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        probe_factory="document_kv_cache_vllm_probe:build_probe",
        output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        expected_backend=ServingBackend.VLLM,
        payload_uri="/Volumes/catalog/schema/volume/probes/vllm-payload.kv",
        wheel_uri=WHEEL_URI,
        single_user_name=SINGLE_USER_NAME,
        release_safe=True,
    )

    payload = build_databricks_engine_probe_run_submit_payload(config)
    parameters = payload["tasks"][0]["spark_python_task"]["parameters"]

    assert "--engine-version" not in parameters
    assert "--allow-non-native-probe" not in parameters


def test_databricks_engine_probe_job_config_preserves_existing_positional_arguments():
    config = DatabricksEngineProbeJobConfig(
        "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        "document_kv_cache.native_probe_factories:vllm_native_probe_factory",
        "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
        "dbfs:/benchmarks/run_engine_probe.py",
        "vllm",
        None,
        DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME,
        DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY,
        "g5.4xlarge",
        "15.4.x-gpu-ml-scala2.12",
        "SINGLE_USER",
        SINGLE_USER_NAME,
        WHEEL_URI,
        "debug-vllm",
        True,
        ("probe.source=positional",),
        False,
        "ON_DEMAND",
        "auto",
        {"team": "document-kv"},
        "/Volumes/catalog/schema/volume/probes/vllm-actions.json",
        "document_kv_vllm_native_adapter:build_probe",
        None,
        "segmented",
    )

    assert config.engine_version == "debug-vllm"
    assert config.allow_non_native_probe is True
    assert config.metadata == ("probe.source=positional",)
    assert config.release_safe is False
    assert config.actions_output_json == "/Volumes/catalog/schema/volume/probes/vllm-actions.json"
    assert config.native_probe_delegate_factory == "document_kv_vllm_native_adapter:build_probe"
    assert config.fixture_payload_mode.value == "segmented"
    assert config.extra_wheel_uris == ()


def test_build_databricks_engine_probe_payload_installs_extra_wheels_in_order():
    config = DatabricksEngineProbeJobConfig(
        handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        probe_factory="document_kv_cache.native_probe_factories:vllm_native_probe_factory",
        output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        expected_backend=ServingBackend.VLLM,
        wheel_uri=WHEEL_URI,
        extra_wheel_uris=(VLLM_ADAPTER_WHEEL_URI, SGLANG_ADAPTER_WHEEL_URI),
        single_user_name=SINGLE_USER_NAME,
        release_safe=True,
    )

    payload = build_databricks_engine_probe_run_submit_payload(config)
    parameters = payload["tasks"][0]["spark_python_task"]["parameters"]

    assert parameters[-6:] == [
        "--package-wheel-uri",
        WHEEL_URI,
        "--package-wheel-uri",
        VLLM_ADAPTER_WHEEL_URI,
        "--package-wheel-uri",
        SGLANG_ADAPTER_WHEEL_URI,
    ]


def test_build_databricks_engine_probe_payload_sets_native_delegate_env_var():
    config = DatabricksEngineProbeJobConfig(
        handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        probe_factory="document_kv_cache.native_probe_factories:vllm_native_probe_factory",
        output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        expected_backend=ServingBackend.VLLM,
        payload_uri="/Volumes/catalog/schema/volume/probes/vllm-payload.kv",
        wheel_uri=WHEEL_URI,
        single_user_name=SINGLE_USER_NAME,
        release_safe=True,
        native_probe_delegate_factory="document_kv_vllm_native_adapter:build_probe",
    )

    payload = build_databricks_engine_probe_run_submit_payload(config)
    task = payload["tasks"][0]

    assert task["new_cluster"]["spark_env_vars"] == {
        VLLM_NATIVE_PROBE_DELEGATE_ENV: "document_kv_vllm_native_adapter:build_probe"
    }
    assert "--native-probe-delegate-factory" not in task["spark_python_task"]["parameters"]


def test_build_databricks_engine_probe_matrix_release_safe_payload_runs_required_backends():
    config = DatabricksEngineProbeMatrixJobConfig(
        probe_targets=(
            _target(
                "vllm",
                metadata=("probe.source=matrix",),
                actions_output_json="/Volumes/catalog/schema/volume/probes/vllm-actions.json",
            ),
            _target("sglang"),
        ),
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        wheel_uri=WHEEL_URI,
        single_user_name=SINGLE_USER_NAME,
        release_safe=True,
        custom_tags={"team": "document-kv"},
    )

    payload = build_databricks_engine_probe_matrix_run_submit_payload(config)

    assert payload["run_name"] == DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME
    assert [task["task_key"] for task in payload["tasks"]] == [
        f"{DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY}_vllm",
        f"{DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY}_sglang",
    ]
    for task, backend in zip(payload["tasks"], ("vllm", "sglang"), strict=True):
        cluster = task["new_cluster"]
        parameters = task["spark_python_task"]["parameters"]
        assert "libraries" not in task
        assert cluster["node_type_id"].startswith(("g5.", "g6."))
        assert cluster["driver_node_type_id"] == cluster["node_type_id"]
        assert cluster["data_security_mode"] == "SINGLE_USER"
        assert cluster["single_user_name"] == SINGLE_USER_NAME
        assert cluster["num_workers"] == 0
        assert cluster["custom_tags"]["ResourceClass"] == "SingleNode"
        assert cluster["custom_tags"]["purpose"] == DEFAULT_DATABRICKS_ENGINE_PROBE_PURPOSE
        assert cluster["custom_tags"]["team"] == "document-kv"
        expected_parameters = [
            "--handoff-json",
            f"/Volumes/catalog/schema/volume/probes/{backend}-handoff.json",
            "--probe-factory",
            f"document_kv_cache_{backend}_probe:build_probe",
            "--output-json",
            f"/Volumes/catalog/schema/volume/probes/{backend}-probe.json",
            "--expected-backend",
            backend,
            "--payload-uri",
            f"/Volumes/catalog/schema/volume/probes/{backend}-payload.kv",
        ]
        if backend == "vllm":
            expected_parameters[8:8] = [
                "--actions-output-json",
                "/Volumes/catalog/schema/volume/probes/vllm-actions.json",
            ]
        if backend == "vllm":
            expected_parameters.extend(["--metadata", "probe.source=matrix"])
        expected_parameters.extend(["--package-wheel-uri", WHEEL_URI])
        assert parameters == expected_parameters
        assert "--engine-version" not in parameters
        assert "--allow-non-native-probe" not in parameters


def test_databricks_engine_probe_matrix_config_preserves_existing_positional_arguments():
    config = DatabricksEngineProbeMatrixJobConfig(
        (_target("vllm"), _target("sglang")),
        "dbfs:/benchmarks/run_engine_probe.py",
        DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME,
        "g5.4xlarge",
        "15.4.x-gpu-ml-scala2.12",
        "SINGLE_USER",
        SINGLE_USER_NAME,
        WHEEL_URI,
        True,
        "ON_DEMAND",
        "auto",
        {"team": "document-kv"},
    )

    assert config.release_safe is True
    assert config.availability == "ON_DEMAND"
    assert config.zone_id == "auto"
    assert config.custom_tags == {"team": "document-kv"}
    assert config.extra_wheel_uris == ()


def test_build_databricks_engine_probe_matrix_payload_installs_extra_wheels_for_each_task():
    config = DatabricksEngineProbeMatrixJobConfig(
        probe_targets=(_target("vllm"), _target("sglang")),
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        wheel_uri=WHEEL_URI,
        extra_wheel_uris=(VLLM_ADAPTER_WHEEL_URI, SGLANG_ADAPTER_WHEEL_URI),
        single_user_name=SINGLE_USER_NAME,
        release_safe=True,
    )

    payload = build_databricks_engine_probe_matrix_run_submit_payload(config)

    for task in payload["tasks"]:
        assert task["spark_python_task"]["parameters"][-6:] == [
            "--package-wheel-uri",
            WHEEL_URI,
            "--package-wheel-uri",
            VLLM_ADAPTER_WHEEL_URI,
            "--package-wheel-uri",
            SGLANG_ADAPTER_WHEEL_URI,
        ]


def test_databricks_engine_probe_matrix_payload_runs_fixture_before_probe():
    fixture_dir = "/Volumes/catalog/schema/volume/probes/vllm-fixture"
    config = DatabricksEngineProbeMatrixJobConfig(
        probe_targets=(
            _target(
                "vllm",
                handoff_json=f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['handoff']}",
                fixture_output_dir=fixture_dir,
                fixture_payload_mode="merged",
                payload_uri=None,
            ),
            _target("sglang"),
        ),
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        wheel_uri=WHEEL_URI,
        single_user_name=SINGLE_USER_NAME,
    )

    payload = build_databricks_engine_probe_matrix_run_submit_payload(config)
    parameters = payload["tasks"][0]["spark_python_task"]["parameters"]

    assert parameters[:6] == [
        "--fixture-output-dir",
        fixture_dir,
        "--fixture-backend",
        "vllm",
        "--fixture-payload-mode",
        "merged",
    ]
    assert parameters[6:14] == [
        "--handoff-json",
        f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['handoff']}",
        "--probe-factory",
        "document_kv_cache_vllm_probe:build_probe",
        "--output-json",
        "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
        "--expected-backend",
        "vllm",
    ]


def test_build_databricks_engine_probe_matrix_payload_sets_backend_delegate_env_vars():
    config = DatabricksEngineProbeMatrixJobConfig(
        probe_targets=(
            _target(
                "vllm",
                probe_factory="document_kv_cache.native_probe_factories:vllm_native_probe_factory",
                native_probe_delegate_factory="document_kv_vllm_native_adapter:build_probe",
            ),
            _target(
                "sglang",
                probe_factory="document_kv_cache.native_probe_factories:sglang_native_probe_factory",
                native_probe_delegate_factory="document_kv_sglang_native_adapter:build_probe",
            ),
        ),
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        wheel_uri=WHEEL_URI,
        single_user_name=SINGLE_USER_NAME,
        release_safe=True,
    )

    payload = build_databricks_engine_probe_matrix_run_submit_payload(config)

    cluster_by_backend = {
        task["spark_python_task"]["parameters"][
            task["spark_python_task"]["parameters"].index("--expected-backend") + 1
        ]: task["new_cluster"]
        for task in payload["tasks"]
    }
    assert cluster_by_backend["vllm"]["spark_env_vars"] == {
        VLLM_NATIVE_PROBE_DELEGATE_ENV: "document_kv_vllm_native_adapter:build_probe"
    }
    assert cluster_by_backend["sglang"]["spark_env_vars"] == {
        SGLANG_NATIVE_PROBE_DELEGATE_ENV: "document_kv_sglang_native_adapter:build_probe"
    }


def test_databricks_engine_probe_matrix_release_safe_requires_exact_backend_set():
    with pytest.raises(ValueError, match="exactly required backends"):
        DatabricksEngineProbeMatrixJobConfig(
            probe_targets=(_target("vllm"),),
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            single_user_name=SINGLE_USER_NAME,
            release_safe=True,
        )

    with pytest.raises(ValueError, match="duplicate backends"):
        DatabricksEngineProbeMatrixJobConfig(
            probe_targets=(_target("vllm"), _target("vllm", task_key="second-vllm")),
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            single_user_name=SINGLE_USER_NAME,
        )


def test_databricks_engine_probe_matrix_release_safe_rejects_debug_target_options():
    with pytest.raises(ValueError, match="engine_version"):
        DatabricksEngineProbeMatrixJobConfig(
            probe_targets=(
                _target("vllm", engine_version="debug-vllm"),
                _target("sglang"),
            ),
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            single_user_name=SINGLE_USER_NAME,
            release_safe=True,
        )

    with pytest.raises(ValueError, match="non-native"):
        DatabricksEngineProbeMatrixJobConfig(
            probe_targets=(
                _target("vllm", allow_non_native_probe=True),
                _target("sglang"),
            ),
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            single_user_name=SINGLE_USER_NAME,
            release_safe=True,
        )


def test_read_databricks_engine_probe_targets_json_accepts_object_and_aliases(tmp_path):
    path = tmp_path / "probe-targets.json"
    path.write_text(
        json.dumps(
            {
                DEFAULT_DATABRICKS_ENGINE_PROBE_BACKEND_CONFIG_KEY: [
                    {
                        "backend": "vllm",
                        "handoff_json": "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
                        "probe_factory": "document_kv_cache_vllm_probe:build_probe",
                        "probe_output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                        "connector_actions_output_json": "/Volumes/catalog/schema/volume/probes/vllm-actions.json",
                    },
                    {
                        "expected_backend": "sglang",
                        "handoff_json": "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
                        "probe_factory": "document_kv_cache_sglang_probe:build_probe",
                        "output_json": "/Volumes/catalog/schema/volume/probes/sglang-probe.json",
                        "actions_output_json": "/Volumes/catalog/schema/volume/probes/sglang-actions.json",
                        "allow_non_native_probe": False,
                        "metadata": ["probe.source=targets"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    targets = read_databricks_engine_probe_targets_json(path)
    targets_file = read_databricks_engine_probe_targets_file_json(path)

    assert isinstance(targets_file, DatabricksEngineProbeTargetsFile)
    assert targets_file.release_safe is False
    assert [target.expected_backend for target in targets] == [ServingBackend.VLLM, ServingBackend.SGLANG]
    assert targets[0].output_json == "/Volumes/catalog/schema/volume/probes/vllm-probe.json"
    assert targets[0].actions_output_json == "/Volumes/catalog/schema/volume/probes/vllm-actions.json"
    assert targets[1].output_json == "/Volumes/catalog/schema/volume/probes/sglang-probe.json"
    assert targets[1].actions_output_json == "/Volumes/catalog/schema/volume/probes/sglang-actions.json"
    assert targets[1].metadata == ("probe.source=targets",)


def test_read_databricks_engine_probe_targets_json_accepts_native_delegate_factory(tmp_path):
    path = tmp_path / "probe-targets.json"
    path.write_text(
        json.dumps(
            [
                {
                    "backend": "vllm",
                    "handoff_json": "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
                    "probe_factory": "document_kv_cache.native_probe_factories:vllm_native_probe_factory",
                    "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                    "native_probe_delegate_factory": "document_kv_vllm_native_adapter:build_probe",
                }
            ]
        ),
        encoding="utf-8",
    )

    (target,) = read_databricks_engine_probe_targets_json(path)

    assert target.native_probe_delegate_factory == "document_kv_vllm_native_adapter:build_probe"


def test_read_databricks_engine_probe_targets_json_accepts_fixture_fields(tmp_path):
    path = tmp_path / "probe-targets.json"
    fixture_dir = "/Volumes/catalog/schema/volume/probes/vllm-fixture"
    path.write_text(
        json.dumps(
            [
                {
                    "backend": "vllm",
                    "fixture_output_dir": fixture_dir,
                    "fixture_payload_mode": "merged",
                    "handoff_json": f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['handoff']}",
                    "probe_factory": "document_kv_cache_vllm_probe:build_probe",
                    "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                }
            ]
        ),
        encoding="utf-8",
    )

    (target,) = read_databricks_engine_probe_targets_json(path)

    assert target.fixture_output_dir == fixture_dir
    assert target.fixture_payload_mode == "merged"


def test_read_databricks_engine_probe_targets_json_rejects_fixture_handoff_mismatch(tmp_path):
    path = tmp_path / "probe-targets.json"
    path.write_text(
        json.dumps(
            [
                {
                    "backend": "vllm",
                    "fixture_output_dir": "/Volumes/catalog/schema/volume/probes/vllm-fixture",
                    "handoff_json": "/Volumes/catalog/schema/volume/probes/custom-handoff.json",
                    "probe_factory": "document_kv_cache_vllm_probe:build_probe",
                    "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="derived fixture handoff path"):
        read_databricks_engine_probe_targets_json(path)


def test_read_databricks_engine_probe_targets_json_rejects_absolute_alias_for_relative_fixture_dir(tmp_path):
    path = tmp_path / "probe-targets.json"
    fixture_dir = "vllm-fixture"
    path.write_text(
        json.dumps(
            [
                {
                    "backend": "vllm",
                    "fixture_output_dir": fixture_dir,
                    "handoff_json": str(
                        (tmp_path / fixture_dir / DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES["handoff"]).resolve()
                    ),
                    "probe_factory": "document_kv_cache_vllm_probe:build_probe",
                    "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="derived fixture handoff path"):
        read_databricks_engine_probe_targets_json(path)


def test_read_databricks_engine_probe_targets_json_rejects_unsupported_fixture_output_uri_scheme(tmp_path):
    path = tmp_path / "probe-targets.json"
    fixture_dir = "s3://bucket/probes/vllm-fixture"
    path.write_text(
        json.dumps(
            [
                {
                    "backend": "vllm",
                    "fixture_output_dir": fixture_dir,
                    "handoff_json": f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['handoff']}",
                    "probe_factory": "document_kv_cache_vllm_probe:build_probe",
                    "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fixture_output_dir URI scheme"):
        read_databricks_engine_probe_targets_json(path)


def test_read_databricks_engine_probe_targets_json_rejects_fixture_payload_mismatch(tmp_path):
    path = tmp_path / "probe-targets.json"
    fixture_dir = "/Volumes/catalog/schema/volume/probes/vllm-fixture"
    path.write_text(
        json.dumps(
            [
                {
                    "backend": "vllm",
                    "fixture_output_dir": fixture_dir,
                    "handoff_json": f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['handoff']}",
                    "payload_uri": "/Volumes/catalog/schema/volume/probes/custom-payload.kv",
                    "probe_factory": "document_kv_cache_vllm_probe:build_probe",
                    "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="derived fixture payload path"):
        read_databricks_engine_probe_targets_json(path)


def test_read_databricks_engine_probe_targets_json_rejects_non_boolean_debug_flag(tmp_path):
    path = tmp_path / "probe-targets.json"
    path.write_text(
        json.dumps(
            [
                {
                    "backend": "vllm",
                    "handoff_json": "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
                    "probe_factory": "document_kv_cache_vllm_probe:build_probe",
                    "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                    "allow_non_native_probe": "false",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="allow_non_native_probe"):
        read_databricks_engine_probe_targets_json(path)


def test_read_databricks_engine_probe_targets_json_rejects_unsupported_envelope_keys(tmp_path):
    path = tmp_path / "probe-targets.json"
    path.write_text(
        json.dumps(
            {
                "record_type": "document_kv.engine_probe_targets.v1",
                "schema_version": 1,
                "release_safe": False,
                "debug": {"accepted": False},
                "probes": [
                    {
                        "backend": "vllm",
                        "handoff_json": "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
                        "probe_factory": "document_kv_cache_vllm_probe:build_probe",
                        "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"engine probe targets record has unsupported keys: \['debug'\]"):
        read_databricks_engine_probe_targets_json(path)


def test_read_databricks_engine_probe_targets_json_rejects_unsupported_probe_keys(tmp_path):
    path = tmp_path / "probe-targets.json"
    path.write_text(
        json.dumps(
            [
                {
                    "backend": "vllm",
                    "handoff_json": "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
                    "probe_factory": "document_kv_cache_vllm_probe:build_probe",
                    "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                    "debug": {"accepted": False},
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"backend config probe 0 has unsupported keys: \['debug'\]"):
        read_databricks_engine_probe_targets_json(path)


def test_run_engine_probe_task_generates_fixture_then_runs_probe(monkeypatch):
    import document_kv_cache.engine_probe as engine_probe
    import document_kv_cache.probe_fixtures as probe_fixtures

    calls = []

    def fake_fixture_main(argv):
        calls.append(("fixture", tuple(argv)))
        return 0

    def fake_probe_main(argv):
        calls.append(("probe", tuple(argv)))
        return 0

    monkeypatch.setattr(probe_fixtures, "main", fake_fixture_main)
    monkeypatch.setattr(engine_probe, "main", fake_probe_main)

    exit_code = run_engine_probe_task(
        [
            "--fixture-output-dir",
            "/Volumes/catalog/schema/volume/probes/vllm-fixture",
            "--fixture-backend",
            "vllm",
            "--fixture-payload-mode",
            "merged",
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-fixture/qwen3-v1-fixture.handoff.json",
            "--probe-factory",
            "document_kv_cache_vllm_probe:build_probe",
            "--output-json",
            "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            "--expected-backend",
            "vllm",
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            "fixture",
            (
                "--output-dir",
                "/Volumes/catalog/schema/volume/probes/vllm-fixture",
                "--backend",
                "vllm",
                "--payload-mode",
                "merged",
            ),
        ),
        (
            "probe",
            (
                "--handoff-json",
                "/Volumes/catalog/schema/volume/probes/vllm-fixture/qwen3-v1-fixture.handoff.json",
                "--probe-factory",
                "document_kv_cache_vllm_probe:build_probe",
                "--output-json",
                "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                "--expected-backend",
                "vllm",
            ),
        ),
    ]


def test_run_engine_probe_task_stops_when_fixture_generation_fails(monkeypatch):
    import document_kv_cache.engine_probe as engine_probe
    import document_kv_cache.probe_fixtures as probe_fixtures

    calls = []

    def fake_fixture_main(argv):
        calls.append(("fixture", tuple(argv)))
        return 7

    def fake_probe_main(argv):
        calls.append(("probe", tuple(argv)))
        return 0

    monkeypatch.setattr(probe_fixtures, "main", fake_fixture_main)
    monkeypatch.setattr(engine_probe, "main", fake_probe_main)

    exit_code = run_engine_probe_task(
        [
            "--fixture-output-dir",
            "/Volumes/catalog/schema/volume/probes/vllm-fixture",
            "--fixture-backend",
            "vllm",
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-fixture/qwen3-v1-fixture.handoff.json",
        ]
    )

    assert exit_code == 7
    assert [name for name, _argv in calls] == ["fixture"]


def test_read_databricks_engine_probe_targets_json_honors_release_safe_envelope(tmp_path):
    path = tmp_path / "probe-targets.json"
    path.write_text(
        json.dumps(
            {
                "record_type": "document_kv.engine_probe_targets.v1",
                "schema_version": 1,
                "release_safe": True,
                "probes": [
                    {
                        "backend": "vllm",
                        "handoff_json": "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
                        "probe_factory": "document_kv_cache_vllm_probe:build_probe",
                        "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                    },
                    {
                        "backend": "sglang",
                        "handoff_json": "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
                        "probe_factory": "document_kv_cache_sglang_probe:build_probe",
                        "output_json": "/Volumes/catalog/schema/volume/probes/sglang-probe.json",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    targets_file = read_databricks_engine_probe_targets_file_json(path)

    assert targets_file.release_safe is True
    assert [target.expected_backend for target in targets_file.probe_targets] == [
        ServingBackend.VLLM,
        ServingBackend.SGLANG,
    ]


def test_main_honors_release_safe_engine_probe_targets_envelope_without_cli_flag(tmp_path):
    backend_config_path = tmp_path / "probe-targets.json"
    payload_path = tmp_path / "payload.json"
    backend_config_path.write_text(
        json.dumps(
            {
                "record_type": "document_kv.engine_probe_targets.v1",
                "schema_version": 1,
                "release_safe": True,
                "probes": [
                    {
                        "backend": "vllm",
                        "handoff_json": "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
                        "probe_factory": "document_kv_cache_vllm_probe:build_probe",
                        "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                    },
                    {
                        "backend": "sglang",
                        "handoff_json": "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
                        "probe_factory": "document_kv_cache_sglang_probe:build_probe",
                        "output_json": "/Volumes/catalog/schema/volume/probes/sglang-probe.json",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--backend-config-json",
            str(backend_config_path),
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(payload_path),
        ]
    )

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert [task["task_key"] for task in payload["tasks"]] == [
        f"{DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY}_vllm",
        f"{DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY}_sglang",
    ]


def test_main_rejects_debug_targets_when_release_safe_envelope_is_true(tmp_path, capsys):
    backend_config_path = tmp_path / "probe-targets.json"
    backend_config_path.write_text(
        json.dumps(
            {
                "record_type": "document_kv.engine_probe_targets.v1",
                "schema_version": 1,
                "release_safe": True,
                "probes": [
                    {
                        "backend": "vllm",
                        "handoff_json": "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
                        "probe_factory": "document_kv_cache_vllm_probe:build_probe",
                        "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                        "engine_version": "debug-vllm",
                    },
                    {
                        "backend": "sglang",
                        "handoff_json": "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
                        "probe_factory": "document_kv_cache_sglang_probe:build_probe",
                        "output_json": "/Volumes/catalog/schema/volume/probes/sglang-probe.json",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--backend-config-json",
            str(backend_config_path),
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--single-user-name",
            SINGLE_USER_NAME,
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert "release-safe" in output["error"]
    assert "engine_version" in output["error"]


def test_write_databricks_engine_probe_matrix_run_submit_json_writes_payload(tmp_path):
    path = tmp_path / "payload.json"

    write_databricks_engine_probe_matrix_run_submit_json(
        DatabricksEngineProbeMatrixJobConfig(
            probe_targets=(_target("vllm"), _target("sglang")),
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            single_user_name=SINGLE_USER_NAME,
            release_safe=True,
        ),
        path,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [task["task_key"] for task in payload["tasks"]] == [
        f"{DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY}_vllm",
        f"{DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY}_sglang",
    ]


def test_databricks_engine_probe_release_safe_rejects_debug_options():
    with pytest.raises(ValueError, match="release-safe.*engine_version"):
        DatabricksEngineProbeJobConfig(
            handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            probe_factory="document_kv_cache_vllm_probe:build_probe",
            output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            expected_backend=ServingBackend.VLLM,
            engine_version="debug-vllm",
            single_user_name=SINGLE_USER_NAME,
            release_safe=True,
        )

    with pytest.raises(ValueError, match="release-safe.*non-native"):
        DatabricksEngineProbeJobConfig(
            handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            probe_factory="document_kv_cache_vllm_probe:build_probe",
            output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            expected_backend=ServingBackend.VLLM,
            allow_non_native_probe=True,
            single_user_name=SINGLE_USER_NAME,
            release_safe=True,
        )


def test_databricks_engine_probe_config_normalizes_backend_and_requires_single_user_name():
    config = DatabricksEngineProbeJobConfig(
        handoff_json="/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
        probe_factory="sglang_probe:build_probe",
        output_json="/Volumes/catalog/schema/volume/probes/sglang-probe.json",
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        expected_backend="sglang",
        single_user_name=SINGLE_USER_NAME,
    )

    assert config.expected_backend == ServingBackend.SGLANG

    try:
        DatabricksEngineProbeJobConfig(
            handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            probe_factory="vllm_probe:build_probe",
            output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            expected_backend="vllm",
        )
    except ValueError as exc:
        assert "single_user_name is required" in str(exc)
    else:
        raise AssertionError("expected SINGLE_USER validation to fail")


def test_databricks_engine_probe_config_rejects_unknown_backend():
    try:
        DatabricksEngineProbeJobConfig(
            handoff_json="/Volumes/catalog/schema/volume/probes/probe-handoff.json",
            probe_factory="probe:build_probe",
            output_json="/Volumes/catalog/schema/volume/probes/probe.json",
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            expected_backend="triton",
            single_user_name=SINGLE_USER_NAME,
        )
    except ValueError as exc:
        assert "expected_backend" in str(exc)
    else:
        raise AssertionError("expected unknown backend validation to fail")


def test_write_databricks_engine_probe_runner_script_imports_task_runner(tmp_path):
    path = tmp_path / "run_engine_probe.py"

    write_databricks_engine_probe_runner_script(path)

    runner_text = path.read_text(encoding="utf-8")
    assert "--package-wheel-uri" in runner_text
    assert "pip\", \"install\"" in runner_text
    assert "dbfs:/" in runner_text
    assert "document_kv_cache.databricks_engine_probe_job" in runner_text
    assert "run_engine_probe_task" in runner_text
    assert "if exit_code:" in runner_text


def test_generated_engine_probe_runner_installs_wheel_before_forwarding_args(tmp_path):
    runner_path = tmp_path / "run_engine_probe.py"
    pip_calls_path = tmp_path / "pip-calls.jsonl"
    task_args_path = tmp_path / "task-args.json"
    events_path = tmp_path / "events.jsonl"
    package_dir = tmp_path / "document_kv_cache"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "databricks_engine_probe_job.py").write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "",
                "with open(os.environ['RUNNER_EVENTS_JSONL'], 'a', encoding='utf-8') as handle:",
                "    handle.write(json.dumps({'event': 'engine_probe_job_import'}) + '\\n')",
                "",
                "def run_engine_probe_task(argv=None):",
                "    with open(os.environ['RUNNER_EVENTS_JSONL'], 'a', encoding='utf-8') as handle:",
                "        handle.write(json.dumps({'event': 'run_engine_probe_task'}) + '\\n')",
                "    with open(os.environ['TASK_ARGS_JSON'], 'w', encoding='utf-8') as handle:",
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
                "    with open(os.environ['PIP_CALLS_JSONL'], 'a', encoding='utf-8') as handle:",
                "        handle.write(json.dumps(argv) + '\\n')",
                "    return 0",
                "",
                "subprocess.check_call = _capture_check_call",
                "",
            ]
        ),
        encoding="utf-8",
    )

    write_databricks_engine_probe_runner_script(runner_path)
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "PIP_CALLS_JSONL": str(pip_calls_path),
        "TASK_ARGS_JSON": str(task_args_path),
        "RUNNER_EVENTS_JSONL": str(events_path),
    }

    subprocess.run(
        [
            sys.executable,
            str(runner_path),
            "--package-wheel-uri",
            "dbfs:/tmp/cachet/document_kv_cache-0.2.0-py3-none-any.whl",
            "--package-wheel-uri",
            "dbfs:/tmp/cachet/vllm_kv_injection-0.2.0-py3-none-any.whl",
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            "--expected-backend",
            "vllm",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    pip_calls = [json.loads(line) for line in pip_calls_path.read_text(encoding="utf-8").splitlines()]
    assert [Path(call[0]).resolve() for call in pip_calls] == [Path(sys.executable).resolve()] * 2
    assert [call[1:] for call in pip_calls] == [
        ["-m", "pip", "install", "/dbfs/tmp/cachet/document_kv_cache-0.2.0-py3-none-any.whl"],
        ["-m", "pip", "install", "/dbfs/tmp/cachet/vllm_kv_injection-0.2.0-py3-none-any.whl"],
    ]
    assert json.loads(task_args_path.read_text(encoding="utf-8")) == [
        "--handoff-json",
        "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        "--expected-backend",
        "vllm",
    ]
    events = [json.loads(line)["event"] for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert events == ["pip_install", "pip_install", "engine_probe_job_import", "run_engine_probe_task"]


def test_write_databricks_engine_probe_run_submit_json_writes_payload(tmp_path):
    path = tmp_path / "payload.json"

    write_databricks_engine_probe_run_submit_json(
        DatabricksEngineProbeJobConfig(
            handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            probe_factory="vllm_probe:build_probe",
            output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            expected_backend="vllm",
            single_user_name=SINGLE_USER_NAME,
        ),
        path,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["tasks"][0]["task_key"] == DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY


def test_main_writes_engine_probe_payload_and_runner_script(tmp_path):
    payload_path = tmp_path / "payload.json"
    runner_path = tmp_path / "run_engine_probe.py"

    exit_code = main(
        [
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            "--probe-factory",
            "vllm_probe:build_probe",
            "--probe-output-json",
            "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            "--actions-output-json",
            "/Volumes/catalog/schema/volume/probes/vllm-actions.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--expected-backend",
            "vllm",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--wheel-uri",
            WHEEL_URI,
            "--extra-wheel-uri",
            VLLM_ADAPTER_WHEEL_URI,
            "--output-json",
            str(payload_path),
            "--runner-script-output",
            str(runner_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    task = payload["tasks"][0]
    assert "libraries" not in task
    assert "--actions-output-json" in task["spark_python_task"]["parameters"]
    assert task["spark_python_task"]["parameters"][-4:] == [
        "--package-wheel-uri",
        WHEEL_URI,
        "--package-wheel-uri",
        VLLM_ADAPTER_WHEEL_URI,
    ]
    assert "engine_probe" in runner_path.read_text(encoding="utf-8")


def test_main_derives_single_engine_probe_handoff_from_fixture_output_dir(tmp_path):
    payload_path = tmp_path / "payload.json"
    fixture_dir = "/Volumes/catalog/schema/volume/probes/vllm-fixture"

    exit_code = main(
        [
            "--fixture-output-dir",
            fixture_dir,
            "--fixture-payload-mode",
            "merged",
            "--probe-factory",
            "vllm_probe:build_probe",
            "--probe-output-json",
            "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--expected-backend",
            "vllm",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(payload_path),
        ]
    )

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    parameters = payload["tasks"][0]["spark_python_task"]["parameters"]

    assert exit_code == 0
    assert parameters[:8] == [
        "--fixture-output-dir",
        fixture_dir,
        "--fixture-backend",
        "vllm",
        "--fixture-payload-mode",
        "merged",
        "--handoff-json",
        f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['handoff']}",
    ]


def test_main_writes_single_engine_probe_payload_with_native_delegate_env(tmp_path):
    payload_path = tmp_path / "payload.json"

    exit_code = main(
        [
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
            "--probe-factory",
            "document_kv_cache.native_probe_factories:sglang_native_probe_factory",
            "--probe-output-json",
            "/Volumes/catalog/schema/volume/probes/sglang-probe.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--expected-backend",
            "sglang",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--native-probe-delegate-factory",
            "document_kv_sglang_native_adapter:build_probe",
            "--output-json",
            str(payload_path),
        ]
    )

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["tasks"][0]["new_cluster"]["spark_env_vars"] == {
        SGLANG_NATIVE_PROBE_DELEGATE_ENV: "document_kv_sglang_native_adapter:build_probe"
    }


def test_main_writes_engine_probe_matrix_payload_from_backend_config_json(tmp_path):
    backend_config_path = tmp_path / "probe-targets.json"
    payload_path = tmp_path / "payload.json"
    runner_path = tmp_path / "run_engine_probe.py"
    backend_config_path.write_text(
        json.dumps(
            {
                "probes": [
                    {
                        "backend": "vllm",
                        "handoff_json": "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
                        "probe_factory": "document_kv_cache_vllm_probe:build_probe",
                        "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                    },
                    {
                        "backend": "sglang",
                        "handoff_json": "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
                        "probe_factory": "document_kv_cache_sglang_probe:build_probe",
                        "output_json": "/Volumes/catalog/schema/volume/probes/sglang-probe.json",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--backend-config-json",
            str(backend_config_path),
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--wheel-uri",
            WHEEL_URI,
            "--release-safe",
            "--output-json",
            str(payload_path),
            "--runner-script-output",
            str(runner_path),
        ]
    )

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert [task["task_key"] for task in payload["tasks"]] == [
        f"{DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY}_vllm",
        f"{DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY}_sglang",
    ]
    assert all("libraries" not in task for task in payload["tasks"])
    assert all(
        task["spark_python_task"]["parameters"][-2:] == ["--package-wheel-uri", WHEEL_URI]
        for task in payload["tasks"]
    )
    assert "engine_probe" in runner_path.read_text(encoding="utf-8")


def test_main_rejects_single_target_debug_flags_in_matrix_mode(tmp_path, capsys):
    backend_config_path = tmp_path / "probe-targets.json"
    backend_config_path.write_text(
        json.dumps(
            {
                "probes": [
                    {
                        "backend": "vllm",
                        "handoff_json": "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
                        "probe_factory": "document_kv_cache_vllm_probe:build_probe",
                        "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                    },
                    {
                        "backend": "sglang",
                        "handoff_json": "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
                        "probe_factory": "document_kv_cache_sglang_probe:build_probe",
                        "output_json": "/Volumes/catalog/schema/volume/probes/sglang-probe.json",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--backend-config-json",
            str(backend_config_path),
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--release-safe",
            "--engine-version",
            "debug-vllm",
            "--native-probe-delegate-factory",
            "document_kv_vllm_native_adapter:build_probe",
            "--allow-non-native-probe",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["error_type"] == "ValueError"
    assert "--backend-config-json cannot be combined" in output["error"]
    assert "--engine-version" in output["error"]
    assert "--native-probe-delegate-factory" in output["error"]
    assert "--allow-non-native-probe" in output["error"]


def test_main_rejects_single_target_task_key_in_matrix_mode(tmp_path, capsys):
    backend_config_path = tmp_path / "probe-targets.json"
    backend_config_path.write_text(
        json.dumps(
            [
                {
                    "backend": "vllm",
                    "handoff_json": "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
                    "probe_factory": "document_kv_cache_vllm_probe:build_probe",
                    "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                }
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--backend-config-json",
            str(backend_config_path),
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--task-key",
            "ignored-single-task",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert "--task-key" in output["error"]


def test_main_rejects_debug_options_in_release_safe_mode(capsys):
    exit_code = main(
        [
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            "--probe-factory",
            "vllm_probe:build_probe",
            "--probe-output-json",
            "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--expected-backend",
            "vllm",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--engine-version",
            "debug-vllm",
            "--release-safe",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["error_type"] == "ValueError"
    assert "release-safe" in output["error"]


def test_public_engine_probe_job_main_respects_document_namespace_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"
    original_legacy_build = legacy_engine_probe_job.build_databricks_engine_probe_run_submit_payload

    def fake_build(config):
        assert config.expected_backend == ServingBackend.SGLANG
        return {"ok": True, "source": "public-hook"}

    monkeypatch.setattr(public_engine_probe_job, "build_databricks_engine_probe_run_submit_payload", fake_build)

    exit_code = public_engine_probe_job.main(
        [
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
            "--probe-factory",
            "sglang_probe:build_probe",
            "--probe-output-json",
            "/Volumes/catalog/schema/volume/probes/sglang-probe.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--expected-backend",
            "sglang",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"ok": True, "source": "public-hook"}
    assert legacy_engine_probe_job.build_databricks_engine_probe_run_submit_payload is original_legacy_build


def test_legacy_engine_probe_job_main_respects_legacy_namespace_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"
    original_public_build = public_engine_probe_job.build_databricks_engine_probe_run_submit_payload

    def fake_build(config):
        assert config.expected_backend == ServingBackend.SGLANG
        return {"ok": True, "source": "legacy-hook"}

    monkeypatch.setattr(legacy_engine_probe_job, "build_databricks_engine_probe_run_submit_payload", fake_build)

    exit_code = legacy_engine_probe_job.main(
        [
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
            "--probe-factory",
            "sglang_probe:build_probe",
            "--probe-output-json",
            "/Volumes/catalog/schema/volume/probes/sglang-probe.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--expected-backend",
            "sglang",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"ok": True, "source": "legacy-hook"}
    assert public_engine_probe_job.build_databricks_engine_probe_run_submit_payload is original_public_build


def test_legacy_engine_probe_job_ignores_document_namespace_build_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"

    def fake_public_build(config):
        return {"ok": True, "source": "unexpected-public-hook"}

    monkeypatch.setattr(public_engine_probe_job, "build_databricks_engine_probe_run_submit_payload", fake_public_build)

    exit_code = legacy_engine_probe_job.main(
        [
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            "--probe-factory",
            "vllm_probe:build_probe",
            "--probe-output-json",
            "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--expected-backend",
            "vllm",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload != {"ok": True, "source": "unexpected-public-hook"}
    assert payload["tasks"][0]["task_key"] == DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY


def test_legacy_engine_probe_job_ignores_document_namespace_writer_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"
    runner_path = tmp_path / "run_engine_probe.py"

    def fake_public_runner_writer(path):
        Path(path).write_text("# unexpected public hook\n", encoding="utf-8")

    monkeypatch.setattr(public_engine_probe_job, "write_databricks_engine_probe_runner_script", fake_public_runner_writer)

    exit_code = legacy_engine_probe_job.main(
        [
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            "--probe-factory",
            "vllm_probe:build_probe",
            "--probe-output-json",
            "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--expected-backend",
            "vllm",
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
    assert "run_engine_probe_task" in runner_path.read_text(encoding="utf-8")


def test_legacy_engine_probe_job_ignores_document_private_helper_monkeypatch(monkeypatch):
    config = legacy_engine_probe_job.DatabricksEngineProbeJobConfig(
        handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        probe_factory="vllm_probe:build_probe",
        output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        expected_backend="vllm",
        single_user_name=SINGLE_USER_NAME,
    )

    def fake_public_runner_parameters(config):
        return ["--unexpected-public-private-hook"]

    monkeypatch.setattr(public_engine_probe_job, "_runner_parameters", fake_public_runner_parameters)

    payload = legacy_engine_probe_job.build_databricks_engine_probe_run_submit_payload(config)

    assert payload["tasks"][0]["spark_python_task"]["parameters"] != ["--unexpected-public-private-hook"]
    assert payload["tasks"][0]["spark_python_task"]["parameters"][:2] == [
        "--handoff-json",
        "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
    ]


def test_legacy_engine_probe_job_payload_respects_legacy_private_cluster_monkeypatch(monkeypatch):
    config = legacy_engine_probe_job.DatabricksEngineProbeJobConfig(
        handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        probe_factory="vllm_probe:build_probe",
        output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        expected_backend="vllm",
        single_user_name=SINGLE_USER_NAME,
    )

    def broken_legacy_cluster_config(config):
        raise RuntimeError(f"legacy cluster config hook for {config.expected_backend.value}")

    monkeypatch.setattr(
        legacy_engine_probe_job,
        "_cluster_config_from_engine_probe_job",
        broken_legacy_cluster_config,
    )

    try:
        legacy_engine_probe_job.build_databricks_engine_probe_run_submit_payload(config)
    except RuntimeError as exc:
        assert "legacy cluster config hook" in str(exc)
    else:
        raise AssertionError("expected legacy private cluster monkeypatch to be observed")


def test_legacy_engine_probe_job_config_ignores_document_private_helper_monkeypatch(monkeypatch):
    def broken_public_cluster_config(config):
        raise RuntimeError(f"unexpected document private hook for {config.expected_backend}")

    monkeypatch.setattr(public_engine_probe_job, "_cluster_config_from_engine_probe_job", broken_public_cluster_config)

    config = legacy_engine_probe_job.DatabricksEngineProbeJobConfig(
        handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        probe_factory="vllm_probe:build_probe",
        output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        expected_backend="vllm",
        single_user_name=SINGLE_USER_NAME,
    )

    assert config.expected_backend == ServingBackend.VLLM


def test_legacy_engine_probe_job_config_respects_legacy_backend_monkeypatch(monkeypatch):
    def broken_serving_backend(value):
        raise RuntimeError(f"legacy backend hook for {value}")

    monkeypatch.setattr(legacy_engine_probe_job, "_serving_backend", broken_serving_backend)

    try:
        legacy_engine_probe_job.DatabricksEngineProbeJobConfig(
            handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            probe_factory="vllm_probe:build_probe",
            output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            expected_backend="vllm",
            single_user_name=SINGLE_USER_NAME,
        )
    except RuntimeError as exc:
        assert "legacy backend hook" in str(exc)
    else:
        raise AssertionError("expected legacy backend monkeypatch to be observed")


def test_legacy_engine_probe_job_ignores_preimport_document_backend_monkeypatch():
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    script = f"""
import json
import document_kv_cache.databricks_engine_probe_job as public_job

def broken_backend(value):
    raise RuntimeError(f"unexpected import-order backend hook for {{value}}")

public_job._serving_backend = broken_backend

import restaurant_kv_serving.databricks_engine_probe_job as legacy_job

config = legacy_job.DatabricksEngineProbeJobConfig(
    handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
    probe_factory="vllm_probe:build_probe",
    output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
    runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
    expected_backend="vllm",
    single_user_name={SINGLE_USER_NAME!r},
)
print(json.dumps({{"backend": config.expected_backend.value}}))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert json.loads(result.stdout) == {"backend": "vllm"}


def test_legacy_engine_probe_job_uses_source_config_base_when_public_class_is_replaced_before_import():
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    script = f"""
import json
import document_kv_cache.databricks_engine_probe_job as public_job

public_job.DatabricksEngineProbeJobConfig = object

import restaurant_kv_serving.databricks_engine_probe_job as legacy_job

config = legacy_job.DatabricksEngineProbeJobConfig(
    handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
    probe_factory="vllm_probe:build_probe",
    output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
    runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
    expected_backend="vllm",
    single_user_name={SINGLE_USER_NAME!r},
)
print(json.dumps({{"backend": config.expected_backend.value, "module": type(config).__module__}}))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "backend": "vllm",
        "module": "restaurant_kv_serving.databricks_engine_probe_job",
    }


def test_legacy_engine_probe_job_matrix_uses_source_target_base_when_public_class_is_replaced_before_import():
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    script = f"""
import json
import document_kv_cache.databricks_engine_probe_job as public_job

public_job.DatabricksEngineProbeTargetConfig = object

import restaurant_kv_serving.databricks_engine_probe_job as legacy_job

target = legacy_job.DatabricksEngineProbeTargetConfig(
    expected_backend="vllm",
    handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
    probe_factory="vllm_probe:build_probe",
    output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
)
config = legacy_job.DatabricksEngineProbeMatrixJobConfig(
    probe_targets=(target,),
    runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
    single_user_name={SINGLE_USER_NAME!r},
)

try:
    legacy_job.DatabricksEngineProbeMatrixJobConfig(
        probe_targets=(object(),),
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        single_user_name={SINGLE_USER_NAME!r},
    )
except TypeError as exc:
    invalid_error = str(exc)
else:
    invalid_error = "<none>"

print(json.dumps({{
    "backend": config.probe_targets[0].expected_backend.value,
    "target_module": type(config.probe_targets[0]).__module__,
    "invalid_error": invalid_error,
}}))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "backend": "vllm",
        "target_module": "restaurant_kv_serving.databricks_engine_probe_job",
        "invalid_error": "probe_targets entries must be DatabricksEngineProbeTargetConfig",
    }


def test_legacy_engine_probe_job_direct_writer_respects_legacy_build_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"

    def fake_build(config):
        assert config.expected_backend == ServingBackend.VLLM
        return {"ok": True, "source": "legacy-direct-writer-hook"}

    monkeypatch.setattr(legacy_engine_probe_job, "build_databricks_engine_probe_run_submit_payload", fake_build)

    legacy_engine_probe_job.write_databricks_engine_probe_run_submit_json(
        legacy_engine_probe_job.DatabricksEngineProbeJobConfig(
            handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            probe_factory="vllm_probe:build_probe",
            output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            expected_backend="vllm",
            single_user_name=SINGLE_USER_NAME,
        ),
        output_path,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "ok": True,
        "source": "legacy-direct-writer-hook",
    }


def test_legacy_engine_probe_job_restores_document_hooks_after_error(monkeypatch, tmp_path):
    output_path = tmp_path / "payload.json"
    original_public_build = public_engine_probe_job.build_databricks_engine_probe_run_submit_payload

    def broken_build(config):
        raise RuntimeError(f"boom for {config.expected_backend.value}")

    monkeypatch.setattr(legacy_engine_probe_job, "build_databricks_engine_probe_run_submit_payload", broken_build)

    exit_code = legacy_engine_probe_job.main(
        [
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            "--probe-factory",
            "vllm_probe:build_probe",
            "--probe-output-json",
            "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--expected-backend",
            "vllm",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 1
    assert public_engine_probe_job.build_databricks_engine_probe_run_submit_payload is original_public_build


def test_legacy_engine_probe_job_module_execution_shows_help():
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }

    result = subprocess.run(
        [sys.executable, "-m", "restaurant_kv_serving.databricks_engine_probe_job", "--help"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "Emit a Databricks runs/submit payload for an AWS g5/g6 engine probe." in result.stdout


def test_legacy_engine_probe_job_reexports_document_owned_types():
    assert issubclass(
        legacy_engine_probe_job.DatabricksEngineProbeJobConfig,
        public_engine_probe_job.DatabricksEngineProbeJobConfig,
    )
    assert issubclass(
        legacy_engine_probe_job.DatabricksEngineProbeMatrixJobConfig,
        public_engine_probe_job.DatabricksEngineProbeMatrixJobConfig,
    )
    assert issubclass(
        legacy_engine_probe_job.DatabricksEngineProbeTargetConfig,
        public_engine_probe_job.DatabricksEngineProbeTargetConfig,
    )
    assert issubclass(
        legacy_engine_probe_job.DatabricksEngineProbeTargetsFile,
        public_engine_probe_job.DatabricksEngineProbeTargetsFile,
    )
    assert (
        public_engine_probe_job.DatabricksEngineProbeJobConfig.__module__
        == "document_kv_cache.databricks_engine_probe_job"
    )
    assert (
        legacy_engine_probe_job.DatabricksEngineProbeJobConfig.__module__
        == "restaurant_kv_serving.databricks_engine_probe_job"
    )
    assert (
        legacy_engine_probe_job.DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE
        == legacy_engine_probe_job.DEFAULT_AWS_G5_NODE_TYPE
    )
    assert (
        legacy_engine_probe_job.DatabricksSingleNodeGPUClusterConfig
        is legacy_engine_probe_job.DatabricksSingleNodeG5ClusterConfig
    )
    assert (
        legacy_engine_probe_job.build_single_node_gpu_cluster
        is legacy_engine_probe_job.build_single_node_g5_cluster
    )
    assert set(public_engine_probe_job.__all__) < set(legacy_engine_probe_job.__all__)


def test_legacy_engine_probe_job_config_pickle_uses_honest_legacy_module():
    config = legacy_engine_probe_job.DatabricksEngineProbeJobConfig(
        handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        probe_factory="vllm_probe:build_probe",
        output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        expected_backend="vllm",
        single_user_name=SINGLE_USER_NAME,
    )

    restored = pickle.loads(pickle.dumps(config))

    assert type(restored) is legacy_engine_probe_job.DatabricksEngineProbeJobConfig
    assert restored == config


def test_legacy_engine_probe_job_config_keeps_slotted_layout():
    config = legacy_engine_probe_job.DatabricksEngineProbeJobConfig(
        handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        probe_factory="vllm_probe:build_probe",
        output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        expected_backend="vllm",
        single_user_name=SINGLE_USER_NAME,
    )

    assert not hasattr(config, "__dict__")


def test_legacy_engine_probe_job_keeps_previous_star_import_surface():
    assert set(legacy_engine_probe_job.__all__) == {
        "Any",
        "DEFAULT_AWS_SINGLE_NODE_GPU_NODE_TYPE",
        "DEFAULT_AWS_G5_NODE_TYPE",
        "DEFAULT_DATABRICKS_DATA_SECURITY_MODE",
        "DEFAULT_DATABRICKS_ENGINE_PROBE_BACKEND_CONFIG_KEY",
        "DEFAULT_DATABRICKS_ENGINE_PROBE_PURPOSE",
        "DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME",
        "DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY",
        "DEFAULT_DATABRICKS_SPARK_VERSION",
        "DatabricksEngineProbeJobConfig",
        "DatabricksEngineProbeMatrixJobConfig",
        "DatabricksEngineProbeTargetConfig",
        "DatabricksEngineProbeTargetsFile",
        "DatabricksSingleNodeGPUClusterConfig",
        "DatabricksSingleNodeG5ClusterConfig",
        "ENGINE_PROBE_RUNNER_SCRIPT",
        "ENGINE_PROBE_TARGETS_RECORD_TYPE",
        "ENGINE_PROBE_TARGETS_SCHEMA_VERSION",
        "Mapping",
        "PayloadMode",
        "Path",
        "REQUIRED_ENGINE_PROBE_BACKENDS",
        "Sequence",
        "ServingBackend",
        "argparse",
        "build_databricks_engine_probe_matrix_run_submit_payload",
        "build_databricks_engine_probe_run_submit_payload",
        "build_single_node_gpu_cluster",
        "build_single_node_g5_cluster",
        "dataclass",
        "field",
        "json",
        "main",
        "read_databricks_engine_probe_targets_file_json",
        "read_databricks_engine_probe_targets_json",
        "run_engine_probe_task",
        "write_databricks_engine_probe_matrix_run_submit_json",
        "write_databricks_engine_probe_run_submit_json",
        "write_databricks_engine_probe_runner_script",
    }


def test_legacy_engine_probe_job_star_import_uses_previous_surface():
    namespace: dict[str, object] = {}

    exec("from restaurant_kv_serving.databricks_engine_probe_job import *", namespace)

    assert {key for key in namespace if key != "__builtins__"} == set(legacy_engine_probe_job.__all__)
    assert namespace["DatabricksEngineProbeJobConfig"] is legacy_engine_probe_job.DatabricksEngineProbeJobConfig
    assert (
        namespace["DatabricksSingleNodeGPUClusterConfig"]
        is legacy_engine_probe_job.DatabricksSingleNodeGPUClusterConfig
    )
    assert namespace["build_single_node_gpu_cluster"] is legacy_engine_probe_job.build_single_node_gpu_cluster
