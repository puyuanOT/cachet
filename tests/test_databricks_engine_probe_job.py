import json

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
    write_databricks_engine_probe_matrix_run_submit_json,
    write_databricks_engine_probe_run_submit_json,
    write_databricks_engine_probe_runner_script,
)
from document_kv_cache.engine_adapters import ServingBackend


WHEEL_URI = "/Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl"
SINGLE_USER_NAME = "user@example.com"


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
        custom_tags={"team": "document-kv"},
    )

    payload = build_databricks_engine_probe_run_submit_payload(config)
    task = payload["tasks"][0]
    cluster = task["new_cluster"]

    assert payload["run_name"] == DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME
    assert task["task_key"] == DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY
    assert task["libraries"] == [{"whl": WHEEL_URI}]
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
            "--payload-uri",
            "/Volumes/catalog/schema/volume/probes/vllm-payload.kv",
            "--engine-version",
            "debug-vllm",
            "--allow-non-native-probe",
            "--metadata",
            "probe.source=single",
        ],
    }


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


def test_build_databricks_engine_probe_matrix_release_safe_payload_runs_required_backends():
    config = DatabricksEngineProbeMatrixJobConfig(
        probe_targets=(
            _target("vllm", metadata=("probe.source=matrix",)),
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
        assert task["libraries"] == [{"whl": WHEEL_URI}]
        assert cluster["node_type_id"].startswith("g5.")
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
            expected_parameters.extend(["--metadata", "probe.source=matrix"])
        assert parameters == expected_parameters
        assert "--engine-version" not in parameters
        assert "--allow-non-native-probe" not in parameters


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
                    },
                    {
                        "expected_backend": "sglang",
                        "handoff_json": "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
                        "probe_factory": "document_kv_cache_sglang_probe:build_probe",
                        "output_json": "/Volumes/catalog/schema/volume/probes/sglang-probe.json",
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
    assert targets[1].output_json == "/Volumes/catalog/schema/volume/probes/sglang-probe.json"
    assert targets[1].metadata == ("probe.source=targets",)


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


def test_write_databricks_engine_probe_runner_script_imports_probe_main(tmp_path):
    path = tmp_path / "run_engine_probe.py"

    write_databricks_engine_probe_runner_script(path)

    runner_text = path.read_text(encoding="utf-8")
    assert "document_kv_cache.engine_probe" in runner_text
    assert "if exit_code:" in runner_text


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
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--expected-backend",
            "vllm",
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
    assert "engine_probe" in runner_path.read_text(encoding="utf-8")


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
    assert all(task["libraries"] == [{"whl": WHEEL_URI}] for task in payload["tasks"])
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
            "--allow-non-native-probe",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["error_type"] == "ValueError"
    assert "--backend-config-json cannot be combined" in output["error"]
    assert "--engine-version" in output["error"]
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
