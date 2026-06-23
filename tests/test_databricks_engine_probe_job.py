import json
import os
import pickle
from pathlib import Path
import subprocess
import sys

import pytest

import document_kv_cache.databricks_engine_probe_job as public_engine_probe_job
import document_kv_cache._databricks_engine_probe_runner as engine_probe_runner
import restaurant_kv_serving.databricks_engine_probe_job as legacy_engine_probe_job
from document_kv_cache.databricks_engine_probe_job import (
    DEFAULT_DATABRICKS_ENGINE_PROBE_BACKEND_CONFIG_KEY,
    DEFAULT_DATABRICKS_ENGINE_PROBE_PURPOSE,
    DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME,
    DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY,
    DEFAULT_SGLANG_ENGINE_PROBE_RUNTIME_PACKAGE,
    DEFAULT_VLLM_ENGINE_PROBE_RUNTIME_PACKAGE,
    SGLANG_NATIVE_PROBE_DELEGATE_FACTORY,
    SGLANG_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,
    VLLM_NATIVE_PROBE_DELEGATE_FACTORY,
    VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,
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
    SGLANG_NATIVE_PROBE_FACTORY,
    VLLM_NATIVE_PROBE_DELEGATE_ENV,
    VLLM_NATIVE_PROBE_FACTORY,
)
from document_kv_cache.probe_fixtures import DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES
from document_kv_cache.serving_env import (
    SGLANG_DEPENDENCY_CONSTRAINTS,
    VLLM_DEPENDENCY_CONSTRAINTS,
)


WHEEL_URI = "/Volumes/catalog/schema/volume/wheels/document_kv_cache-0.2.0-py3-none-any.whl"
CUSTOM_VLLM_EXTENSION_WHEEL_URI = (
    "/Volumes/catalog/schema/volume/wheels/custom_vllm_probe_extension-0.1.0-py3-none-any.whl"
)
CUSTOM_SGLANG_EXTENSION_WHEEL_URI = (
    "/Volumes/catalog/schema/volume/wheels/custom_sglang_probe_extension-0.1.0-py3-none-any.whl"
)
VLLM_RUNTIME_PACKAGE = DEFAULT_VLLM_ENGINE_PROBE_RUNTIME_PACKAGE
SGLANG_RUNTIME_PACKAGE = DEFAULT_SGLANG_ENGINE_PROBE_RUNTIME_PACKAGE
VLLM_RUNTIME_PACKAGES = VLLM_DEPENDENCY_CONSTRAINTS
SGLANG_RUNTIME_PACKAGES = SGLANG_DEPENDENCY_CONSTRAINTS
VLLM_FIPS_OPENCV_OVERRIDE_PACKAGE = "opencv-python-headless==4.12.0.88"
VLLM_RUNTIME_PREFLIGHT_OUTPUT_JSON = "/Volumes/catalog/schema/volume/probes/vllm-runtime-preflight.json"
VLLM_RUNTIME_PREFLIGHT_LAYER_NAMES_JSON = "/Volumes/catalog/schema/volume/probes/vllm-layer-names.json"
SGLANG_RUNTIME_PREFLIGHT_OUTPUT_JSON = "/Volumes/catalog/schema/volume/probes/sglang-runtime-preflight.json"
SGLANG_RUNTIME_PREFLIGHT_LAUNCH_CONFIG_JSON = "/Volumes/catalog/schema/volume/probes/sglang-launch-config.json"
VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON = "/Volumes/catalog/schema/volume/probes/vllm-native-probe-factories.json"
SGLANG_NATIVE_PROBE_FACTORIES_OUTPUT_JSON = (
    "/Volumes/catalog/schema/volume/probes/sglang-native-probe-factories.json"
)
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
    if backend == "sglang":
        values.update(
            {
                "sglang_runtime_preflight_output_json": SGLANG_RUNTIME_PREFLIGHT_OUTPUT_JSON,
                "sglang_runtime_preflight_launch_config_json": SGLANG_RUNTIME_PREFLIGHT_LAUNCH_CONFIG_JSON,
            }
        )
    values.update(overrides)
    return DatabricksEngineProbeTargetConfig(**values)


def _release_target(backend: str, **overrides):
    values = {
        "actions_output_json": f"/Volumes/catalog/schema/volume/probes/{backend}-actions.json",
        "native_probe_factories_output_json": (
            VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON
            if backend == "vllm"
            else SGLANG_NATIVE_PROBE_FACTORIES_OUTPUT_JSON
        )
    }
    values.update(overrides)
    return _target(backend, **values)


def _parameter_values(parameters, flag):
    return [parameters[index + 1] for index, value in enumerate(parameters[:-1]) if value == flag]


def test_build_databricks_engine_probe_payload_uses_single_node_g5_cluster():
    config = DatabricksEngineProbeJobConfig(
        handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        probe_factory="document_kv_cache_vllm_probe:build_probe",
        output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        expected_backend=ServingBackend.VLLM,
        payload_uri="/Volumes/catalog/schema/volume/probes/vllm-payload.kv",
        node_type_id="g6.8xlarge",
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
    assert cluster["node_type_id"] == "g6.8xlarge"
    assert cluster["driver_node_type_id"] == "g6.8xlarge"
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
        actions_output_json="/Volumes/catalog/schema/volume/probes/vllm-actions.json",
        native_probe_factories_output_json=VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
    )

    payload = build_databricks_engine_probe_run_submit_payload(config)
    task = payload["tasks"][0]
    parameters = task["spark_python_task"]["parameters"]

    assert task["task_key"] == f"{DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY}_vllm"
    assert "--engine-version" not in parameters
    assert "--allow-non-native-probe" not in parameters


def test_build_databricks_engine_probe_release_safe_sglang_payload_uses_backend_task_key():
    config = DatabricksEngineProbeJobConfig(
        handoff_json="/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
        probe_factory="document_kv_cache_sglang_probe:build_probe",
        output_json="/Volumes/catalog/schema/volume/probes/sglang-probe.json",
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        expected_backend=ServingBackend.SGLANG,
        wheel_uri=WHEEL_URI,
        single_user_name=SINGLE_USER_NAME,
        release_safe=True,
        actions_output_json="/Volumes/catalog/schema/volume/probes/sglang-actions.json",
        sglang_runtime_preflight_output_json=SGLANG_RUNTIME_PREFLIGHT_OUTPUT_JSON,
        sglang_runtime_preflight_launch_config_json=SGLANG_RUNTIME_PREFLIGHT_LAUNCH_CONFIG_JSON,
        native_probe_factories_output_json=SGLANG_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
    )

    payload = build_databricks_engine_probe_run_submit_payload(config)

    assert payload["tasks"][0]["task_key"] == f"{DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY}_sglang"


def test_databricks_engine_probe_release_safe_requires_native_factories_output():
    with pytest.raises(ValueError, match="native_probe_factories_output_json"):
        DatabricksEngineProbeJobConfig(
            handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            probe_factory="document_kv_cache_vllm_probe:build_probe",
            output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            expected_backend=ServingBackend.VLLM,
            single_user_name=SINGLE_USER_NAME,
            release_safe=True,
            actions_output_json="/Volumes/catalog/schema/volume/probes/vllm-actions.json",
        )


def test_databricks_engine_probe_release_safe_requires_actions_output():
    with pytest.raises(ValueError, match="actions_output_json"):
        DatabricksEngineProbeJobConfig(
            handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            probe_factory="document_kv_cache_vllm_probe:build_probe",
            output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            expected_backend=ServingBackend.VLLM,
            single_user_name=SINGLE_USER_NAME,
            release_safe=True,
            native_probe_factories_output_json=VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
        )


def test_build_databricks_engine_probe_payload_forwards_native_factories_output():
    config = DatabricksEngineProbeJobConfig(
        handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        probe_factory="document_kv_cache_vllm_probe:build_probe",
        output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        expected_backend=ServingBackend.VLLM,
        single_user_name=SINGLE_USER_NAME,
        native_probe_factories_output_json=VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
    )

    payload = build_databricks_engine_probe_run_submit_payload(config)
    parameters = payload["tasks"][0]["spark_python_task"]["parameters"]

    assert parameters[:2] == [
        "--native-probe-factories-output-json",
        VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
    ]


def test_databricks_engine_probe_rejects_colliding_runner_output_paths():
    with pytest.raises(ValueError, match="output paths must be distinct"):
        DatabricksEngineProbeJobConfig(
            handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            probe_factory="document_kv_cache_vllm_probe:build_probe",
            output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            expected_backend=ServingBackend.VLLM,
            single_user_name=SINGLE_USER_NAME,
            native_probe_factories_output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
        )


@pytest.mark.parametrize(
    ("field_name", "fixture_filename_key"),
    [
        ("output_json", "handoff"),
        ("native_probe_factories_output_json", "pack"),
        ("native_probe_factories_output_json", "payload"),
        ("native_probe_factories_output_json", "actions"),
        ("native_probe_factories_output_json", "manifest"),
        ("native_probe_factories_output_json", "vllm_layer_names"),
    ],
)
def test_databricks_engine_probe_rejects_fixture_child_output_collisions(
    field_name,
    fixture_filename_key,
):
    fixture_dir = "/Volumes/catalog/schema/volume/probes/vllm-fixture"
    values = {
        "handoff_json": f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['handoff']}",
        "probe_factory": "document_kv_cache_vllm_probe:build_probe",
        "output_json": f"{fixture_dir}/vllm-probe.json",
        "runner_python_file": "dbfs:/benchmarks/run_engine_probe.py",
        "expected_backend": ServingBackend.VLLM,
        "payload_uri": None,
        "single_user_name": SINGLE_USER_NAME,
        "fixture_output_dir": fixture_dir,
        "native_probe_factories_output_json": f"{fixture_dir}/vllm-native-probe-factories.json",
    }
    values[field_name] = f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES[fixture_filename_key]}"

    with pytest.raises(ValueError, match="output paths must be distinct"):
        DatabricksEngineProbeJobConfig(**values)


def test_databricks_engine_probe_allows_derived_fixture_actions_output_once():
    fixture_dir = "/Volumes/catalog/schema/volume/probes/vllm-fixture"

    config = DatabricksEngineProbeJobConfig(
        handoff_json=f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['handoff']}",
        probe_factory="document_kv_cache_vllm_probe:build_probe",
        output_json=f"{fixture_dir}/vllm-probe.json",
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        expected_backend=ServingBackend.VLLM,
        payload_uri=None,
        single_user_name=SINGLE_USER_NAME,
        fixture_output_dir=fixture_dir,
        native_probe_factories_output_json=f"{fixture_dir}/vllm-native-probe-factories.json",
    )

    assert config.actions_output_json == f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['actions']}"


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
        "g6.4xlarge",
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
        extra_wheel_uris=(CUSTOM_VLLM_EXTENSION_WHEEL_URI, CUSTOM_SGLANG_EXTENSION_WHEEL_URI),
        single_user_name=SINGLE_USER_NAME,
        release_safe=True,
        actions_output_json="/Volumes/catalog/schema/volume/probes/vllm-actions.json",
        native_probe_factories_output_json=VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
    )

    payload = build_databricks_engine_probe_run_submit_payload(config)
    parameters = payload["tasks"][0]["spark_python_task"]["parameters"]

    assert parameters[-6:] == [
        "--package-wheel-uri",
        WHEEL_URI,
        "--package-wheel-uri",
        CUSTOM_VLLM_EXTENSION_WHEEL_URI,
        "--package-wheel-uri",
        CUSTOM_SGLANG_EXTENSION_WHEEL_URI,
    ]


def test_build_databricks_engine_probe_payload_installs_pip_packages_before_wheels():
    config = DatabricksEngineProbeJobConfig(
        handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        probe_factory="document_kv_cache.native_probe_factories:vllm_native_probe_factory",
        output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        expected_backend=ServingBackend.VLLM,
        wheel_uri=WHEEL_URI,
        extra_wheel_uris=(CUSTOM_VLLM_EXTENSION_WHEEL_URI,),
        extra_pip_packages=(VLLM_RUNTIME_PACKAGE,),
        single_user_name=SINGLE_USER_NAME,
        release_safe=True,
        actions_output_json="/Volumes/catalog/schema/volume/probes/vllm-actions.json",
        native_probe_factories_output_json=VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
    )

    payload = build_databricks_engine_probe_run_submit_payload(config)
    parameters = payload["tasks"][0]["spark_python_task"]["parameters"]

    assert parameters[-6:] == [
        "--pip-package",
        VLLM_RUNTIME_PACKAGE,
        "--package-wheel-uri",
        WHEEL_URI,
        "--package-wheel-uri",
        CUSTOM_VLLM_EXTENSION_WHEEL_URI,
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
        actions_output_json="/Volumes/catalog/schema/volume/probes/vllm-actions.json",
        native_probe_delegate_factory="document_kv_vllm_native_adapter:build_probe",
        native_probe_factories_output_json=VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
    )

    payload = build_databricks_engine_probe_run_submit_payload(config)
    task = payload["tasks"][0]

    assert task["new_cluster"]["spark_env_vars"] == {
        VLLM_NATIVE_PROBE_DELEGATE_ENV: "document_kv_vllm_native_adapter:build_probe"
    }
    assert "--native-probe-delegate-factory" not in task["spark_python_task"]["parameters"]


def test_provider_backed_vllm_probe_installs_runtime_and_cachet_wheel_only():
    config = DatabricksEngineProbeJobConfig(
        handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        probe_factory="document_kv_cache.native_probe_factories:vllm_native_probe_factory",
        output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        expected_backend=ServingBackend.VLLM,
        payload_uri="/Volumes/catalog/schema/volume/probes/vllm-payload.kv",
        wheel_uri=WHEEL_URI,
        extra_pip_packages=(VLLM_RUNTIME_PACKAGE,),
        single_user_name=SINGLE_USER_NAME,
        release_safe=True,
        actions_output_json="/Volumes/catalog/schema/volume/probes/vllm-actions.json",
        native_probe_delegate_factory=VLLM_NATIVE_PROBE_DELEGATE_FACTORY,
        metadata=(VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,),
        vllm_runtime_preflight_output_json=VLLM_RUNTIME_PREFLIGHT_OUTPUT_JSON,
        vllm_runtime_preflight_layer_names_json=VLLM_RUNTIME_PREFLIGHT_LAYER_NAMES_JSON,
        native_probe_factories_output_json=VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
    )

    payload = build_databricks_engine_probe_run_submit_payload(config)
    task = payload["tasks"][0]
    parameters = task["spark_python_task"]["parameters"]

    assert task["new_cluster"]["spark_env_vars"] == {
        VLLM_NATIVE_PROBE_DELEGATE_ENV: VLLM_NATIVE_PROBE_DELEGATE_FACTORY
    }
    assert _parameter_values(parameters, "--pip-package") == list(VLLM_RUNTIME_PACKAGES)
    assert _parameter_values(parameters, "--pip-override-package") == [VLLM_FIPS_OPENCV_OVERRIDE_PACKAGE]
    package_wheel_uris = _parameter_values(parameters, "--package-wheel-uri")
    assert package_wheel_uris == [WHEEL_URI]
    assert not any("vllm_kv_injection" in wheel_uri for wheel_uri in package_wheel_uris)
    assert parameters[:6] == [
        "--native-probe-factories-output-json",
        VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
        "--vllm-runtime-preflight-output-json",
        VLLM_RUNTIME_PREFLIGHT_OUTPUT_JSON,
        "--vllm-runtime-preflight-layer-names-json",
        VLLM_RUNTIME_PREFLIGHT_LAYER_NAMES_JSON,
    ]
    assert VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA in _parameter_values(parameters, "--metadata")


def test_vllm_runtime_preflight_layer_names_derive_from_fixture_output_dir():
    fixture_dir = "/Volumes/catalog/schema/volume/probes/vllm-fixture"
    derived_layer_names_json = f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['vllm_layer_names']}"
    config = DatabricksEngineProbeJobConfig(
        handoff_json=f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['handoff']}",
        probe_factory="document_kv_cache.native_probe_factories:vllm_native_probe_factory",
        output_json=f"{fixture_dir}/vllm-probe.json",
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        expected_backend="vllm",
        wheel_uri=WHEEL_URI,
        extra_pip_packages=(VLLM_RUNTIME_PACKAGE,),
        single_user_name=SINGLE_USER_NAME,
        fixture_output_dir=fixture_dir,
        fixture_payload_mode="merged",
        release_safe=True,
        native_probe_delegate_factory=VLLM_NATIVE_PROBE_DELEGATE_FACTORY,
        metadata=(VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,),
        vllm_runtime_preflight_output_json=f"{fixture_dir}/vllm-runtime-preflight.json",
        native_probe_factories_output_json=f"{fixture_dir}/vllm-native-probe-factories.json",
    )

    payload = build_databricks_engine_probe_run_submit_payload(config)
    parameters = payload["tasks"][0]["spark_python_task"]["parameters"]

    assert config.expected_backend == ServingBackend.VLLM
    assert config.vllm_runtime_preflight_layer_names_json == derived_layer_names_json
    assert parameters[parameters.index("--vllm-runtime-preflight-layer-names-json") + 1] == (
        derived_layer_names_json
    )


def test_vllm_runtime_preflight_rejects_conflicting_fixture_layer_names():
    fixture_dir = "/Volumes/catalog/schema/volume/probes/vllm-fixture"

    with pytest.raises(ValueError, match="fixture layer-name path"):
        DatabricksEngineProbeJobConfig(
            handoff_json=f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['handoff']}",
            probe_factory="document_kv_cache.native_probe_factories:vllm_native_probe_factory",
            output_json=f"{fixture_dir}/vllm-probe.json",
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            expected_backend=ServingBackend.VLLM,
            single_user_name=SINGLE_USER_NAME,
            fixture_output_dir=fixture_dir,
            vllm_runtime_preflight_output_json=f"{fixture_dir}/vllm-runtime-preflight.json",
            vllm_runtime_preflight_layer_names_json=VLLM_RUNTIME_PREFLIGHT_LAYER_NAMES_JSON,
        )


def test_release_safe_provider_backed_vllm_probe_requires_runtime_preflight():
    with pytest.raises(ValueError, match="release-safe provider-backed vLLM"):
        DatabricksEngineProbeJobConfig(
            handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            probe_factory="document_kv_cache.native_probe_factories:vllm_native_probe_factory",
            output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            expected_backend=ServingBackend.VLLM,
            payload_uri="/Volumes/catalog/schema/volume/probes/vllm-payload.kv",
            wheel_uri=WHEEL_URI,
            extra_pip_packages=(VLLM_RUNTIME_PACKAGE,),
            single_user_name=SINGLE_USER_NAME,
            release_safe=True,
            actions_output_json="/Volumes/catalog/schema/volume/probes/vllm-actions.json",
            native_probe_delegate_factory=VLLM_NATIVE_PROBE_DELEGATE_FACTORY,
            metadata=(VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,),
            native_probe_factories_output_json=VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
        )


def test_vllm_runtime_preflight_paths_must_be_provided_together():
    with pytest.raises(ValueError, match="requires both"):
        DatabricksEngineProbeJobConfig(
            handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            probe_factory="document_kv_cache.native_probe_factories:vllm_native_probe_factory",
            output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            expected_backend=ServingBackend.VLLM,
            single_user_name=SINGLE_USER_NAME,
            vllm_runtime_preflight_output_json=VLLM_RUNTIME_PREFLIGHT_OUTPUT_JSON,
        )


def test_sglang_runtime_preflight_paths_must_be_provided_together():
    with pytest.raises(ValueError, match="requires both"):
        DatabricksEngineProbeJobConfig(
            handoff_json="/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
            probe_factory="document_kv_cache_sglang_probe:build_probe",
            output_json="/Volumes/catalog/schema/volume/probes/sglang-probe.json",
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            expected_backend=ServingBackend.SGLANG,
            single_user_name=SINGLE_USER_NAME,
            sglang_runtime_preflight_output_json=SGLANG_RUNTIME_PREFLIGHT_OUTPUT_JSON,
        )


def test_sglang_runtime_preflight_rejects_vllm_backend():
    with pytest.raises(ValueError, match="only supported for expected_backend sglang"):
        DatabricksEngineProbeJobConfig(
            handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            probe_factory="document_kv_cache.native_probe_factories:vllm_native_probe_factory",
            output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            expected_backend=ServingBackend.VLLM,
            single_user_name=SINGLE_USER_NAME,
            sglang_runtime_preflight_output_json=SGLANG_RUNTIME_PREFLIGHT_OUTPUT_JSON,
            sglang_runtime_preflight_launch_config_json=SGLANG_RUNTIME_PREFLIGHT_LAUNCH_CONFIG_JSON,
        )


def test_build_databricks_engine_probe_matrix_release_safe_payload_runs_required_backends():
    config = DatabricksEngineProbeMatrixJobConfig(
        probe_targets=(
            _release_target(
                "vllm",
                metadata=("probe.source=matrix",),
                actions_output_json="/Volumes/catalog/schema/volume/probes/vllm-actions.json",
            ),
            _release_target("sglang"),
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
        assert cluster["node_type_id"].startswith(("g6.",))
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
        expected_parameters[8:8] = [
            "--actions-output-json",
            f"/Volumes/catalog/schema/volume/probes/{backend}-actions.json",
        ]
        if backend == "vllm":
            expected_parameters.extend(["--metadata", "probe.source=matrix"])
        if backend == "sglang":
            expected_parameters[0:0] = [
                "--sglang-runtime-preflight-output-json",
                SGLANG_RUNTIME_PREFLIGHT_OUTPUT_JSON,
                "--sglang-runtime-preflight-launch-config-json",
                SGLANG_RUNTIME_PREFLIGHT_LAUNCH_CONFIG_JSON,
            ]
        expected_parameters[0:0] = [
            "--native-probe-factories-output-json",
            (
                VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON
                if backend == "vllm"
                else SGLANG_NATIVE_PROBE_FACTORIES_OUTPUT_JSON
            ),
        ]
        expected_parameters.extend(["--package-wheel-uri", WHEEL_URI])
        assert parameters == expected_parameters
        assert "--engine-version" not in parameters
        assert "--allow-non-native-probe" not in parameters


def test_databricks_engine_probe_matrix_config_preserves_existing_positional_arguments():
    config = DatabricksEngineProbeMatrixJobConfig(
        (_release_target("vllm"), _release_target("sglang")),
        "dbfs:/benchmarks/run_engine_probe.py",
        DEFAULT_DATABRICKS_ENGINE_PROBE_RUN_NAME,
        "g6.4xlarge",
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
    assert config.serial_tasks is False


def test_build_databricks_engine_probe_matrix_payload_installs_extra_wheels_for_each_task():
    config = DatabricksEngineProbeMatrixJobConfig(
        probe_targets=(_release_target("vllm"), _release_target("sglang")),
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        wheel_uri=WHEEL_URI,
        extra_wheel_uris=(CUSTOM_VLLM_EXTENSION_WHEEL_URI, CUSTOM_SGLANG_EXTENSION_WHEEL_URI),
        single_user_name=SINGLE_USER_NAME,
        release_safe=True,
    )

    payload = build_databricks_engine_probe_matrix_run_submit_payload(config)

    for task in payload["tasks"]:
        assert task["spark_python_task"]["parameters"][-6:] == [
            "--package-wheel-uri",
            WHEEL_URI,
            "--package-wheel-uri",
            CUSTOM_VLLM_EXTENSION_WHEEL_URI,
            "--package-wheel-uri",
            CUSTOM_SGLANG_EXTENSION_WHEEL_URI,
        ]


def test_build_databricks_engine_probe_matrix_payload_installs_backend_pip_packages_and_cachet_wheel_per_task():
    config = DatabricksEngineProbeMatrixJobConfig(
        probe_targets=(
            _release_target("vllm", pip_packages=(VLLM_RUNTIME_PACKAGE,)),
            _release_target("sglang", pip_packages=(SGLANG_RUNTIME_PACKAGE,)),
        ),
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        wheel_uri=WHEEL_URI,
        single_user_name=SINGLE_USER_NAME,
        release_safe=True,
        extra_pip_packages=("typing-extensions==4.15.0",),
    )

    payload = build_databricks_engine_probe_matrix_run_submit_payload(config)

    parameters_by_backend = {
        task["spark_python_task"]["parameters"][
            task["spark_python_task"]["parameters"].index("--expected-backend") + 1
        ]: task["spark_python_task"]["parameters"]
        for task in payload["tasks"]
    }
    assert parameters_by_backend["vllm"][-6:] == [
        "--pip-package",
        "typing-extensions==4.15.0",
        "--pip-package",
        VLLM_RUNTIME_PACKAGE,
        "--package-wheel-uri",
        WHEEL_URI,
    ]
    assert parameters_by_backend["sglang"][-6:] == [
        "--pip-package",
        "typing-extensions==4.15.0",
        "--pip-package",
        SGLANG_RUNTIME_PACKAGE,
        "--package-wheel-uri",
        WHEEL_URI,
    ]
    assert _parameter_values(parameters_by_backend["vllm"], "--package-wheel-uri") == [WHEEL_URI]
    assert _parameter_values(parameters_by_backend["sglang"], "--package-wheel-uri") == [WHEEL_URI]


def test_build_databricks_engine_probe_matrix_payload_can_run_tasks_serially():
    config = DatabricksEngineProbeMatrixJobConfig(
        probe_targets=(_release_target("vllm"), _release_target("sglang")),
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        wheel_uri=WHEEL_URI,
        single_user_name=SINGLE_USER_NAME,
        release_safe=True,
        serial_tasks=True,
    )

    payload = build_databricks_engine_probe_matrix_run_submit_payload(config)

    first_task, second_task = payload["tasks"]
    assert "depends_on" not in first_task
    assert second_task["depends_on"] == [{"task_key": first_task["task_key"]}]


def test_release_safe_engine_probe_matrix_rejects_fixture_payload_mode_outside_contract():
    fixture_dir = "/Volumes/catalog/schema/volume/probes/vllm-fixture"

    with pytest.raises(ValueError, match="fixture_payload_mode.*'merged'"):
        DatabricksEngineProbeMatrixJobConfig(
            probe_targets=(
                _release_target(
                    "vllm",
                    handoff_json=f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['handoff']}",
                    payload_uri=None,
                    fixture_output_dir=fixture_dir,
                    fixture_payload_mode="segmented",
                ),
                _release_target("sglang"),
            ),
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            wheel_uri=WHEEL_URI,
            single_user_name=SINGLE_USER_NAME,
            release_safe=True,
        )


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


def test_databricks_engine_probe_matrix_payload_skips_fixture_owned_actions_output():
    fixture_dir = "/Volumes/catalog/schema/volume/probes/vllm-fixture"
    actions_json = f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['actions']}"
    config = DatabricksEngineProbeMatrixJobConfig(
        probe_targets=(
            _target(
                "vllm",
                handoff_json=f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['handoff']}",
                fixture_output_dir=fixture_dir,
                actions_output_json=actions_json,
                payload_uri=None,
            ),
        ),
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        wheel_uri=WHEEL_URI,
        single_user_name=SINGLE_USER_NAME,
    )

    payload = build_databricks_engine_probe_matrix_run_submit_payload(config)
    parameters = payload["tasks"][0]["spark_python_task"]["parameters"]

    assert "--fixture-output-dir" in parameters
    assert "--actions-output-json" not in parameters


def test_databricks_engine_probe_matrix_payload_skips_fixture_owned_actions_alias(tmp_path):
    fixture_dir = tmp_path / "vllm-fixture"
    actions_json = f"disk:{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['actions']}"
    config = DatabricksEngineProbeMatrixJobConfig(
        probe_targets=(
            _target(
                "vllm",
                handoff_json=str(fixture_dir / DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES["handoff"]),
                fixture_output_dir=str(fixture_dir),
                actions_output_json=actions_json,
                payload_uri=None,
            ),
        ),
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        wheel_uri=WHEEL_URI,
        single_user_name=SINGLE_USER_NAME,
    )

    payload = build_databricks_engine_probe_matrix_run_submit_payload(config)
    parameters = payload["tasks"][0]["spark_python_task"]["parameters"]

    assert "--fixture-output-dir" in parameters
    assert "--actions-output-json" not in parameters


def test_build_databricks_engine_probe_matrix_payload_sets_backend_delegate_env_vars():
    config = DatabricksEngineProbeMatrixJobConfig(
        probe_targets=(
            _release_target(
                "vllm",
                probe_factory="document_kv_cache.native_probe_factories:vllm_native_probe_factory",
                native_probe_delegate_factory="document_kv_vllm_native_adapter:build_probe",
            ),
            _release_target(
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


def test_build_databricks_engine_probe_matrix_payload_forwards_vllm_runtime_preflight():
    config = DatabricksEngineProbeMatrixJobConfig(
        probe_targets=(
            _release_target(
                "vllm",
                probe_factory="document_kv_cache.native_probe_factories:vllm_native_probe_factory",
                native_probe_delegate_factory=VLLM_NATIVE_PROBE_DELEGATE_FACTORY,
                metadata=(VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,),
                pip_packages=(VLLM_RUNTIME_PACKAGE,),
                vllm_runtime_preflight_output_json=VLLM_RUNTIME_PREFLIGHT_OUTPUT_JSON,
                vllm_runtime_preflight_layer_names_json=VLLM_RUNTIME_PREFLIGHT_LAYER_NAMES_JSON,
            ),
            _release_target("sglang"),
        ),
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        wheel_uri=WHEEL_URI,
        single_user_name=SINGLE_USER_NAME,
        release_safe=True,
    )

    payload = build_databricks_engine_probe_matrix_run_submit_payload(config)
    vllm_parameters = payload["tasks"][0]["spark_python_task"]["parameters"]

    assert vllm_parameters[:6] == [
        "--native-probe-factories-output-json",
        VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
        "--vllm-runtime-preflight-output-json",
        VLLM_RUNTIME_PREFLIGHT_OUTPUT_JSON,
        "--vllm-runtime-preflight-layer-names-json",
        VLLM_RUNTIME_PREFLIGHT_LAYER_NAMES_JSON,
    ]
    assert _parameter_values(vllm_parameters, "--pip-package") == list(VLLM_RUNTIME_PACKAGES)
    assert _parameter_values(vllm_parameters, "--pip-override-package") == [VLLM_FIPS_OPENCV_OVERRIDE_PACKAGE]


def test_build_databricks_engine_probe_matrix_payload_rejects_conflicting_provider_backed_profile():
    config = DatabricksEngineProbeMatrixJobConfig(
        probe_targets=(
            _release_target(
                "vllm",
                probe_factory="document_kv_cache.native_probe_factories:vllm_native_probe_factory",
                native_probe_delegate_factory=VLLM_NATIVE_PROBE_DELEGATE_FACTORY,
                metadata=(VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,),
                pip_packages=("transformers==5.11.0",),
                vllm_runtime_preflight_output_json=VLLM_RUNTIME_PREFLIGHT_OUTPUT_JSON,
                vllm_runtime_preflight_layer_names_json=VLLM_RUNTIME_PREFLIGHT_LAYER_NAMES_JSON,
            ),
            _release_target("sglang"),
        ),
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        wheel_uri=WHEEL_URI,
        single_user_name=SINGLE_USER_NAME,
        release_safe=True,
    )

    with pytest.raises(ValueError, match="transformers==5.12.1"):
        build_databricks_engine_probe_matrix_run_submit_payload(config)


def test_build_databricks_engine_probe_matrix_payload_forwards_sglang_runtime_preflight():
    config = DatabricksEngineProbeMatrixJobConfig(
        probe_targets=(_release_target("vllm"), _release_target("sglang")),
        runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
        wheel_uri=WHEEL_URI,
        single_user_name=SINGLE_USER_NAME,
        release_safe=True,
    )

    payload = build_databricks_engine_probe_matrix_run_submit_payload(config)
    sglang_parameters = payload["tasks"][1]["spark_python_task"]["parameters"]

    assert sglang_parameters[:6] == [
        "--native-probe-factories-output-json",
        SGLANG_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
        "--sglang-runtime-preflight-output-json",
        SGLANG_RUNTIME_PREFLIGHT_OUTPUT_JSON,
        "--sglang-runtime-preflight-launch-config-json",
        SGLANG_RUNTIME_PREFLIGHT_LAUNCH_CONFIG_JSON,
    ]


def test_databricks_engine_probe_target_derives_fixture_preflight_layer_names():
    fixture_dir = "/Volumes/catalog/schema/volume/probes/vllm-fixture"
    target = _target(
        "vllm",
        handoff_json=f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['handoff']}",
        payload_uri=None,
        fixture_output_dir=fixture_dir,
        vllm_runtime_preflight_output_json=f"{fixture_dir}/vllm-runtime-preflight.json",
    )

    assert target.vllm_runtime_preflight_layer_names_json == (
        f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['vllm_layer_names']}"
    )


def test_databricks_engine_probe_target_rejects_conflicting_fixture_layer_names():
    fixture_dir = "/Volumes/catalog/schema/volume/probes/vllm-fixture"

    with pytest.raises(ValueError, match="fixture layer-name path"):
        _target(
            "vllm",
            handoff_json=f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['handoff']}",
            payload_uri=None,
            fixture_output_dir=fixture_dir,
            vllm_runtime_preflight_output_json=f"{fixture_dir}/vllm-runtime-preflight.json",
            vllm_runtime_preflight_layer_names_json=VLLM_RUNTIME_PREFLIGHT_LAYER_NAMES_JSON,
        )


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


def test_release_safe_single_engine_probe_rejects_fixture_payload_mode_outside_contract():
    fixture_dir = "/Volumes/catalog/schema/volume/probes/vllm-fixture"

    with pytest.raises(ValueError, match="fixture_payload_mode.*'merged'"):
        DatabricksEngineProbeJobConfig(
            handoff_json=f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['handoff']}",
            probe_factory="document_kv_cache.native_probe_factories:vllm_native_probe_factory",
            output_json=f"{fixture_dir}/vllm-probe.json",
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            expected_backend="vllm",
            single_user_name=SINGLE_USER_NAME,
            fixture_output_dir=fixture_dir,
            fixture_payload_mode="segmented",
            release_safe=True,
            native_probe_factories_output_json=f"{fixture_dir}/vllm-native-probe-factories.json",
        )


def test_databricks_engine_probe_matrix_release_safe_requires_provider_backed_vllm_preflight():
    with pytest.raises(ValueError, match="release-safe provider-backed vLLM"):
        DatabricksEngineProbeMatrixJobConfig(
            probe_targets=(
                _release_target(
                    "vllm",
                    probe_factory="document_kv_cache.native_probe_factories:vllm_native_probe_factory",
                    native_probe_delegate_factory=VLLM_NATIVE_PROBE_DELEGATE_FACTORY,
                    metadata=(VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,),
                    pip_packages=(VLLM_RUNTIME_PACKAGE,),
                ),
                _release_target("sglang"),
            ),
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            single_user_name=SINGLE_USER_NAME,
            release_safe=True,
        )


def test_databricks_engine_probe_matrix_release_safe_requires_sglang_runtime_preflight():
    with pytest.raises(ValueError, match="release-safe SGLang.*sglang_runtime_preflight_output_json"):
        DatabricksEngineProbeMatrixJobConfig(
            probe_targets=(
                _release_target("vllm"),
                _release_target(
                    "sglang",
                    sglang_runtime_preflight_output_json=None,
                    sglang_runtime_preflight_launch_config_json=None,
                ),
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
                        "native_probe_factories_output_json": VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
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
    assert targets[0].native_probe_factories_output_json == VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON
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


def test_read_databricks_engine_probe_targets_json_accepts_pip_packages(tmp_path):
    path = tmp_path / "probe-targets.json"
    path.write_text(
        json.dumps(
            [
                {
                    "backend": "vllm",
                    "handoff_json": "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
                    "probe_factory": "document_kv_cache.native_probe_factories:vllm_native_probe_factory",
                    "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                    "pip_packages": [VLLM_RUNTIME_PACKAGE],
                }
            ]
        ),
        encoding="utf-8",
    )

    (target,) = read_databricks_engine_probe_targets_json(path)

    assert target.pip_packages == (VLLM_RUNTIME_PACKAGE,)


def test_read_databricks_engine_probe_targets_json_accepts_vllm_runtime_preflight_fields(tmp_path):
    path = tmp_path / "probe-targets.json"
    path.write_text(
        json.dumps(
            [
                {
                    "backend": "vllm",
                    "handoff_json": "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
                    "probe_factory": "document_kv_cache.native_probe_factories:vllm_native_probe_factory",
                    "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                    "vllm_runtime_preflight_output_json": VLLM_RUNTIME_PREFLIGHT_OUTPUT_JSON,
                    "vllm_runtime_preflight_layer_names_json": VLLM_RUNTIME_PREFLIGHT_LAYER_NAMES_JSON,
                }
            ]
        ),
        encoding="utf-8",
    )

    (target,) = read_databricks_engine_probe_targets_json(path)

    assert target.vllm_runtime_preflight_output_json == VLLM_RUNTIME_PREFLIGHT_OUTPUT_JSON
    assert target.vllm_runtime_preflight_layer_names_json == VLLM_RUNTIME_PREFLIGHT_LAYER_NAMES_JSON


def test_read_databricks_engine_probe_targets_json_rejects_string_pip_packages(tmp_path):
    path = tmp_path / "probe-targets.json"
    path.write_text(
        json.dumps(
            [
                {
                    "backend": "vllm",
                    "handoff_json": "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
                    "probe_factory": "document_kv_cache.native_probe_factories:vllm_native_probe_factory",
                    "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                    "pip_packages": VLLM_RUNTIME_PACKAGE,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pip_packages"):
        read_databricks_engine_probe_targets_json(path)


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


def test_databricks_engine_probe_config_rejects_pip_options():
    with pytest.raises(ValueError, match="package specs"):
        DatabricksEngineProbeJobConfig(
            handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            probe_factory="document_kv_cache.native_probe_factories:vllm_native_probe_factory",
            output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            expected_backend=ServingBackend.VLLM,
            extra_pip_packages=("--no-index",),
            single_user_name=SINGLE_USER_NAME,
        )


def test_legacy_databricks_engine_probe_config_rejects_pip_options():
    with pytest.raises(ValueError, match="package specs"):
        legacy_engine_probe_job.DatabricksEngineProbeJobConfig(
            handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            probe_factory="document_kv_cache.native_probe_factories:vllm_native_probe_factory",
            output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            expected_backend=ServingBackend.VLLM,
            extra_pip_packages=("--no-index",),
            single_user_name=SINGLE_USER_NAME,
        )


def test_legacy_databricks_engine_probe_matrix_config_rejects_pip_options():
    with pytest.raises(ValueError, match="package specs"):
        legacy_engine_probe_job.DatabricksEngineProbeMatrixJobConfig(
            probe_targets=(_target("vllm"), _target("sglang")),
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            extra_pip_packages=("--no-index",),
            single_user_name=SINGLE_USER_NAME,
        )


def test_legacy_databricks_engine_probe_target_config_rejects_pip_options():
    with pytest.raises(ValueError, match="package specs"):
        legacy_engine_probe_job.DatabricksEngineProbeTargetConfig(
            expected_backend="vllm",
            handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            probe_factory="document_kv_cache.native_probe_factories:vllm_native_probe_factory",
            output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            pip_packages=("--no-index",),
        )


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


def test_run_engine_probe_task_runs_vllm_runtime_preflight_before_probe(monkeypatch):
    import document_kv_cache.engine_probe as engine_probe
    import vllm_kv_injection.vllm_runtime_preflight as vllm_runtime_preflight

    calls = []

    def fake_preflight_main(argv):
        calls.append(("preflight", tuple(argv)))
        return 0

    def fake_probe_main(argv):
        calls.append(("probe", tuple(argv)))
        return 0

    monkeypatch.setattr(vllm_runtime_preflight, "main", fake_preflight_main)
    monkeypatch.setattr(engine_probe, "main", fake_probe_main)

    exit_code = run_engine_probe_task(
        [
            "--vllm-runtime-preflight-output-json",
            VLLM_RUNTIME_PREFLIGHT_OUTPUT_JSON,
            "--vllm-runtime-preflight-layer-names-json",
            VLLM_RUNTIME_PREFLIGHT_LAYER_NAMES_JSON,
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
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
            "preflight",
            (
                "--layer-names-json",
                VLLM_RUNTIME_PREFLIGHT_LAYER_NAMES_JSON,
                "--output-json",
                VLLM_RUNTIME_PREFLIGHT_OUTPUT_JSON,
            ),
        ),
        (
            "probe",
            (
                "--handoff-json",
                "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
                "--probe-factory",
                "document_kv_cache_vllm_probe:build_probe",
                "--output-json",
                "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                "--expected-backend",
                "vllm",
            ),
        ),
    ]


def test_run_engine_probe_task_normalizes_dbfs_vllm_runtime_preflight_paths(monkeypatch):
    import document_kv_cache.engine_probe as engine_probe
    import vllm_kv_injection.vllm_runtime_preflight as vllm_runtime_preflight

    calls = []

    def fake_preflight_main(argv):
        calls.append(("preflight", tuple(argv)))
        return 0

    def fake_probe_main(argv):
        calls.append(("probe", tuple(argv)))
        return 0

    monkeypatch.setattr(vllm_runtime_preflight, "main", fake_preflight_main)
    monkeypatch.setattr(engine_probe, "main", fake_probe_main)

    exit_code = run_engine_probe_task(
        [
            "--vllm-runtime-preflight-output-json",
            "dbfs:/benchmarks/cachet/probes/vllm-runtime-preflight.json",
            "--vllm-runtime-preflight-layer-names-json",
            "dbfs:/benchmarks/cachet/probes/vllm-layer-names.json",
            "--handoff-json",
            "dbfs:/benchmarks/cachet/probes/vllm-handoff.json",
        ]
    )

    assert exit_code == 0
    assert calls[0] == (
        "preflight",
        (
            "--layer-names-json",
            "/dbfs/benchmarks/cachet/probes/vllm-layer-names.json",
            "--output-json",
            "/dbfs/benchmarks/cachet/probes/vllm-runtime-preflight.json",
        ),
    )


def test_run_engine_probe_task_preserves_inline_vllm_layer_names_json(monkeypatch):
    import document_kv_cache.engine_probe as engine_probe
    import vllm_kv_injection.vllm_runtime_preflight as vllm_runtime_preflight

    calls = []
    inline_layer_names = '{"layer_names": ["probe.layer.0"]}'

    def fake_preflight_main(argv):
        calls.append(("preflight", tuple(argv)))
        return 0

    def fake_probe_main(argv):
        calls.append(("probe", tuple(argv)))
        return 0

    monkeypatch.setattr(vllm_runtime_preflight, "main", fake_preflight_main)
    monkeypatch.setattr(engine_probe, "main", fake_probe_main)

    exit_code = run_engine_probe_task(
        [
            "--vllm-runtime-preflight-output-json",
            "dbfs:/benchmarks/cachet/probes/vllm-runtime-preflight.json",
            "--vllm-runtime-preflight-layer-names-json",
            inline_layer_names,
            "--handoff-json",
            "dbfs:/benchmarks/cachet/probes/vllm-handoff.json",
        ]
    )

    assert exit_code == 0
    assert calls[0] == (
        "preflight",
        (
            "--layer-names-json",
            inline_layer_names,
            "--output-json",
            "/dbfs/benchmarks/cachet/probes/vllm-runtime-preflight.json",
        ),
    )


def test_run_engine_probe_task_stops_when_vllm_runtime_preflight_fails(monkeypatch):
    import document_kv_cache.engine_probe as engine_probe
    import vllm_kv_injection.vllm_runtime_preflight as vllm_runtime_preflight

    calls = []

    def fake_preflight_main(argv):
        calls.append(("preflight", tuple(argv)))
        return 2

    def fake_probe_main(argv):
        calls.append(("probe", tuple(argv)))
        return 0

    monkeypatch.setattr(vllm_runtime_preflight, "main", fake_preflight_main)
    monkeypatch.setattr(engine_probe, "main", fake_probe_main)

    exit_code = run_engine_probe_task(
        [
            "--vllm-runtime-preflight-output-json",
            VLLM_RUNTIME_PREFLIGHT_OUTPUT_JSON,
            "--vllm-runtime-preflight-layer-names-json",
            VLLM_RUNTIME_PREFLIGHT_LAYER_NAMES_JSON,
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        ]
    )

    assert exit_code == 2
    assert [name for name, _argv in calls] == ["preflight"]


def test_run_engine_probe_task_runs_sglang_runtime_preflight_before_probe(monkeypatch):
    import document_kv_cache.engine_probe as engine_probe
    import sglang_kv_injection.sglang_runtime_preflight as sglang_runtime_preflight

    calls = []

    def fake_preflight_main(argv):
        calls.append(("preflight", tuple(argv)))
        return 0

    def fake_probe_main(argv):
        calls.append(("probe", tuple(argv)))
        return 0

    monkeypatch.setattr(sglang_runtime_preflight, "main", fake_preflight_main)
    monkeypatch.setattr(engine_probe, "main", fake_probe_main)

    exit_code = run_engine_probe_task(
        [
            "--sglang-runtime-preflight-output-json",
            SGLANG_RUNTIME_PREFLIGHT_OUTPUT_JSON,
            "--sglang-runtime-preflight-launch-config-json",
            SGLANG_RUNTIME_PREFLIGHT_LAUNCH_CONFIG_JSON,
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
            "--probe-factory",
            "document_kv_cache_sglang_probe:build_probe",
            "--output-json",
            "/Volumes/catalog/schema/volume/probes/sglang-probe.json",
            "--expected-backend",
            "sglang",
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            "preflight",
            (
                "--launch-config-json",
                SGLANG_RUNTIME_PREFLIGHT_LAUNCH_CONFIG_JSON,
                "--output-json",
                SGLANG_RUNTIME_PREFLIGHT_OUTPUT_JSON,
            ),
        ),
        (
            "probe",
            (
                "--handoff-json",
                "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
                "--probe-factory",
                "document_kv_cache_sglang_probe:build_probe",
                "--output-json",
                "/Volumes/catalog/schema/volume/probes/sglang-probe.json",
                "--expected-backend",
                "sglang",
            ),
        ),
    ]


def test_run_engine_probe_task_normalizes_dbfs_sglang_runtime_preflight_paths(monkeypatch):
    import document_kv_cache.engine_probe as engine_probe
    import sglang_kv_injection.sglang_runtime_preflight as sglang_runtime_preflight

    calls = []

    def fake_preflight_main(argv):
        calls.append(("preflight", tuple(argv)))
        return 0

    def fake_probe_main(argv):
        calls.append(("probe", tuple(argv)))
        return 0

    monkeypatch.setattr(sglang_runtime_preflight, "main", fake_preflight_main)
    monkeypatch.setattr(engine_probe, "main", fake_probe_main)

    exit_code = run_engine_probe_task(
        [
            "--sglang-runtime-preflight-output-json",
            "dbfs:/benchmarks/cachet/probes/sglang-runtime-preflight.json",
            "--sglang-runtime-preflight-launch-config-json",
            "dbfs:/benchmarks/cachet/probes/sglang-launch-config.json",
            "--handoff-json",
            "dbfs:/benchmarks/cachet/probes/sglang-handoff.json",
        ]
    )

    assert exit_code == 0
    assert calls[0] == (
        "preflight",
        (
            "--launch-config-json",
            "/dbfs/benchmarks/cachet/probes/sglang-launch-config.json",
            "--output-json",
            "/dbfs/benchmarks/cachet/probes/sglang-runtime-preflight.json",
        ),
    )


def test_run_engine_probe_task_stops_when_sglang_runtime_preflight_fails(monkeypatch):
    import document_kv_cache.engine_probe as engine_probe
    import sglang_kv_injection.sglang_runtime_preflight as sglang_runtime_preflight

    calls = []

    def fake_preflight_main(argv):
        calls.append(("preflight", tuple(argv)))
        return 3

    def fake_probe_main(argv):
        calls.append(("probe", tuple(argv)))
        return 0

    monkeypatch.setattr(sglang_runtime_preflight, "main", fake_preflight_main)
    monkeypatch.setattr(engine_probe, "main", fake_probe_main)

    exit_code = run_engine_probe_task(
        [
            "--sglang-runtime-preflight-output-json",
            SGLANG_RUNTIME_PREFLIGHT_OUTPUT_JSON,
            "--sglang-runtime-preflight-launch-config-json",
            SGLANG_RUNTIME_PREFLIGHT_LAUNCH_CONFIG_JSON,
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
        ]
    )

    assert exit_code == 3
    assert [name for name, _argv in calls] == ["preflight"]


def test_run_engine_probe_task_writes_native_probe_factories_before_probe(monkeypatch):
    import document_kv_cache.engine_probe as engine_probe
    import document_kv_cache.native_probe_factories as native_probe_factories

    calls = []

    def fake_native_probe_factories_main(argv):
        calls.append(("native_probe_factories", tuple(argv)))
        return 0

    def fake_probe_main(argv):
        calls.append(("probe", tuple(argv)))
        return 0

    monkeypatch.setattr(native_probe_factories, "main", fake_native_probe_factories_main)
    monkeypatch.setattr(engine_probe, "main", fake_probe_main)

    exit_code = run_engine_probe_task(
        [
            "--native-probe-factories-output-json",
            "dbfs:/benchmarks/cachet/probes/native-probe-factories.json",
            "--handoff-json",
            "dbfs:/benchmarks/cachet/probes/vllm-handoff.json",
            "--probe-factory",
            "document_kv_cache_vllm_probe:build_probe",
            "--output-json",
            "dbfs:/benchmarks/cachet/probes/vllm-probe.json",
            "--expected-backend",
            "vllm",
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            "native_probe_factories",
            (
                "--output-json",
                "/dbfs/benchmarks/cachet/probes/native-probe-factories.json",
            ),
        ),
        (
            "probe",
            (
                "--handoff-json",
                "dbfs:/benchmarks/cachet/probes/vllm-handoff.json",
                "--probe-factory",
                "document_kv_cache_vllm_probe:build_probe",
                "--output-json",
                "dbfs:/benchmarks/cachet/probes/vllm-probe.json",
                "--expected-backend",
                "vllm",
            ),
        ),
    ]


def test_run_engine_probe_task_stops_when_native_probe_factories_fail(monkeypatch):
    import document_kv_cache.engine_probe as engine_probe
    import document_kv_cache.native_probe_factories as native_probe_factories

    calls = []

    def fake_native_probe_factories_main(argv):
        calls.append(("native_probe_factories", tuple(argv)))
        return 4

    def fake_probe_main(argv):
        calls.append(("probe", tuple(argv)))
        return 0

    monkeypatch.setattr(native_probe_factories, "main", fake_native_probe_factories_main)
    monkeypatch.setattr(engine_probe, "main", fake_probe_main)

    exit_code = run_engine_probe_task(
        [
            "--native-probe-factories-output-json",
            VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        ]
    )

    assert exit_code == 4
    assert [name for name, _argv in calls] == ["native_probe_factories"]


def test_write_databricks_engine_probe_runner_script_installs_pip_packages(tmp_path):
    path = tmp_path / "run_engine_probe.py"

    write_databricks_engine_probe_runner_script(path)

    script = path.read_text(encoding="utf-8")
    assert "--pip-package" in script
    assert "_install_runtime_packages" in script
    assert "venv\", \"--clear\"" in script
    assert "virtualenv==20.39.1" in script
    assert "--pip-override-package" in script
    assert "--skip-runtime-package-install" in script
    assert "PYTHONPATH" in script
    assert "PYTHONNOUSERSITE" in script
    assert "pip\", \"install\", *args.pip_package" in script
    assert "\"--force-reinstall\"" in script
    assert "\"--no-deps\"" in script


def test_generated_runner_installs_pip_packages_and_wheels_before_venv_reexec(tmp_path, monkeypatch):
    path = tmp_path / "run_engine_probe.py"
    write_databricks_engine_probe_runner_script(path)
    venv_dir = tmp_path / "serving-venv"
    venv_python = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
    install_calls = []
    install_envs = []
    reexec_calls = []
    reexec_envs = []
    probe_calls = []

    def fake_check_call(argv, **kwargs):
        install_calls.append(tuple(argv))
        install_envs.append(kwargs.get("env"))

    def fake_call(argv, **kwargs):
        reexec_calls.append(tuple(argv))
        reexec_envs.append(kwargs.get("env"))
        return 0

    def fake_run_engine_probe_task(argv):
        probe_calls.append(tuple(argv))
        return 0

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(subprocess, "call", fake_call)
    monkeypatch.setattr(engine_probe_runner, "run_engine_probe_task", fake_run_engine_probe_task)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--serving-venv-dir",
            str(venv_dir),
            "--pip-package",
            VLLM_RUNTIME_PACKAGE,
            "--pip-package",
            "transformers==5.12.1",
            "--pip-override-package",
            VLLM_FIPS_OPENCV_OVERRIDE_PACKAGE,
            "--package-wheel-uri",
            "dbfs:/wheels/document_kv_cache-0.2.0-py3-none-any.whl",
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        ],
    )

    exec(
        compile(path.read_text(encoding="utf-8"), str(path), "exec"),
        {"__name__": "__main__", "__file__": str(path)},
    )

    assert install_calls == [
        (sys.executable, "-m", "venv", "--clear", str(venv_dir)),
        (str(venv_python), "-m", "pip", "install", "--upgrade", "pip"),
        (str(venv_python), "-m", "pip", "install", VLLM_RUNTIME_PACKAGE, "transformers==5.12.1"),
        (
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            VLLM_FIPS_OPENCV_OVERRIDE_PACKAGE,
        ),
        (str(venv_python), "-m", "pip", "install", "/dbfs/wheels/document_kv_cache-0.2.0-py3-none-any.whl"),
    ]
    assert install_envs[0] is None
    assert all(env is not None and "PYTHONPATH" not in env for env in install_envs[1:])
    assert all(env is not None and env["PYTHONNOUSERSITE"] == "1" for env in install_envs[1:])
    assert reexec_calls == [
        (
            str(venv_python),
            str(path),
            "--skip-runtime-package-install",
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        )
    ]
    assert reexec_envs[0] is not None
    assert "PYTHONPATH" not in reexec_envs[0]
    assert reexec_envs[0]["PYTHONNOUSERSITE"] == "1"
    assert probe_calls == []


def test_generated_runner_propagates_nonzero_venv_reexec_exit(tmp_path, monkeypatch):
    path = tmp_path / "run_engine_probe.py"
    write_databricks_engine_probe_runner_script(path)
    venv_dir = tmp_path / "serving-venv"
    probe_calls = []

    monkeypatch.setattr(subprocess, "check_call", lambda argv, **kwargs: None)
    monkeypatch.setattr(subprocess, "call", lambda argv, **kwargs: 9)
    monkeypatch.setattr(
        engine_probe_runner,
        "run_engine_probe_task",
        lambda argv: probe_calls.append(tuple(argv)) or 0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--serving-venv-dir",
            str(venv_dir),
            "--package-wheel-uri",
            "dbfs:/wheels/document_kv_cache-0.2.0-py3-none-any.whl",
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        exec(
            compile(path.read_text(encoding="utf-8"), str(path), "exec"),
            {"__name__": "__main__", "__file__": str(path)},
        )

    assert exc_info.value.code == 9
    assert probe_calls == []


def test_generated_runner_falls_back_to_virtualenv_when_stdlib_venv_lacks_ensurepip(tmp_path, monkeypatch):
    path = tmp_path / "run_engine_probe.py"
    write_databricks_engine_probe_runner_script(path)
    venv_dir = tmp_path / "serving-venv"
    venv_python = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
    install_calls = []

    def fake_check_call(argv, **kwargs):
        call = tuple(argv)
        install_calls.append(call)
        if call == (sys.executable, "-m", "venv", "--clear", str(venv_dir)):
            raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(subprocess, "call", lambda argv, **kwargs: 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--serving-venv-dir",
            str(venv_dir),
            "--package-wheel-uri",
            "dbfs:/wheels/document_kv_cache-0.2.0-py3-none-any.whl",
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        ],
    )

    exec(
        compile(path.read_text(encoding="utf-8"), str(path), "exec"),
        {"__name__": "__main__", "__file__": str(path)},
    )
    assert install_calls[:5] == [
        (sys.executable, "-m", "venv", "--clear", str(venv_dir)),
        (sys.executable, "-m", "pip", "install", "virtualenv==20.39.1"),
        (sys.executable, "-m", "virtualenv", "--clear", str(venv_dir)),
        (str(venv_python), "-m", "pip", "install", "--upgrade", "pip"),
        (str(venv_python), "-m", "pip", "install", "/dbfs/wheels/document_kv_cache-0.2.0-py3-none-any.whl"),
    ]


def test_generated_runner_reexec_uses_argv0_when_databricks_exec_omits_file(tmp_path, monkeypatch):
    path = tmp_path / "run_engine_probe.py"
    write_databricks_engine_probe_runner_script(path)
    venv_dir = tmp_path / "serving-venv"
    venv_python = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
    reexec_calls = []

    monkeypatch.setattr(subprocess, "check_call", lambda argv, **kwargs: None)
    monkeypatch.setattr(subprocess, "call", lambda argv, **kwargs: reexec_calls.append(tuple(argv)) or 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--serving-venv-dir",
            str(venv_dir),
            "--package-wheel-uri",
            "dbfs:/wheels/document_kv_cache-0.2.0-py3-none-any.whl",
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        ],
    )

    exec(
        compile(path.read_text(encoding="utf-8"), str(path), "exec"),
        {"__name__": "__main__"},
    )
    assert reexec_calls == [
        (
            str(venv_python),
            str(path),
            "--skip-runtime-package-install",
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        )
    ]


def test_generated_runner_skip_runtime_package_install_forwards_args(tmp_path, monkeypatch):
    path = tmp_path / "run_engine_probe.py"
    write_databricks_engine_probe_runner_script(path)
    install_calls = []
    reexec_calls = []
    probe_calls = []

    def fake_run_engine_probe_task(argv):
        probe_calls.append(tuple(argv))
        return 0

    monkeypatch.setattr(subprocess, "check_call", lambda argv, **kwargs: install_calls.append(tuple(argv)))
    monkeypatch.setattr(subprocess, "call", lambda argv, **kwargs: reexec_calls.append(tuple(argv)) or 0)
    monkeypatch.setattr(engine_probe_runner, "run_engine_probe_task", fake_run_engine_probe_task)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--skip-runtime-package-install",
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        ],
    )

    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})

    assert install_calls == []
    assert reexec_calls == []
    assert probe_calls == [
        (
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        )
    ]


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
                        "actions_output_json": "/Volumes/catalog/schema/volume/probes/vllm-actions.json",
                        "native_probe_factories_output_json": VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
                    },
                    {
                        "backend": "sglang",
                        "handoff_json": "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
                        "probe_factory": "document_kv_cache_sglang_probe:build_probe",
                        "output_json": "/Volumes/catalog/schema/volume/probes/sglang-probe.json",
                        "actions_output_json": "/Volumes/catalog/schema/volume/probes/sglang-actions.json",
                        "sglang_runtime_preflight_output_json": SGLANG_RUNTIME_PREFLIGHT_OUTPUT_JSON,
                        "sglang_runtime_preflight_launch_config_json": SGLANG_RUNTIME_PREFLIGHT_LAUNCH_CONFIG_JSON,
                        "native_probe_factories_output_json": SGLANG_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
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


def test_read_databricks_engine_probe_targets_json_rejects_known_delegate_missing_connector_factory(tmp_path):
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
                        "probe_factory": "document_kv_cache.native_probe_factories:vllm_native_probe_factory",
                        "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                        "native_probe_delegate_factory": VLLM_NATIVE_PROBE_DELEGATE_FACTORY,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="build_document_kv_native_probe_connector"):
        read_databricks_engine_probe_targets_json(path)


def test_read_databricks_engine_probe_targets_json_accepts_known_delegate_connector_factory_metadata(tmp_path):
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
                        "probe_factory": "document_kv_cache.native_probe_factories:vllm_native_probe_factory",
                        "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                        "native_probe_delegate_factory": VLLM_NATIVE_PROBE_DELEGATE_FACTORY,
                        "metadata": [
                            VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    targets = read_databricks_engine_probe_targets_json(path)

    assert targets[0].metadata == (VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,)


def test_read_databricks_engine_probe_targets_json_accepts_sglang_provider_backed_metadata(tmp_path):
    path = tmp_path / "probe-targets.json"
    path.write_text(
        json.dumps(
            {
                "record_type": "document_kv.engine_probe_targets.v1",
                "schema_version": 1,
                "release_safe": True,
                "probes": [
                    {
                        "backend": "sglang",
                        "handoff_json": "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
                        "probe_factory": "document_kv_cache.native_probe_factories:sglang_native_probe_factory",
                        "output_json": "/Volumes/catalog/schema/volume/probes/sglang-probe.json",
                        "native_probe_delegate_factory": "sglang_kv_injection.probe:build_native_connector_probe",
                        "metadata": [SGLANG_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    targets = read_databricks_engine_probe_targets_json(path)

    assert targets[0].metadata == (SGLANG_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,)


@pytest.mark.parametrize(
    ("backend", "delegate_factory", "metadata"),
    [
        (
            "vllm",
            VLLM_NATIVE_PROBE_DELEGATE_FACTORY,
            "vllm_kv_injection.connector_factory=module:factory",
        ),
        (
            "vllm",
            VLLM_NATIVE_PROBE_DELEGATE_FACTORY,
            "vllm_kv_injection.connector_factory=module:factory ",
        ),
        (
            "vllm",
            VLLM_NATIVE_PROBE_DELEGATE_FACTORY,
            "vllm_kv_injection.connector_factory=company_vllm_patch_probe",
        ),
        (
            "sglang",
            "sglang_kv_injection.probe:build_native_connector_probe",
            "sglang_kv_injection.connector_factory=module:factory",
        ),
        (
            "sglang",
            "sglang_kv_injection.probe:build_native_connector_probe",
            "sglang_kv_injection.connector_factory=module:factory ",
        ),
        (
            "sglang",
            "sglang_kv_injection.probe:build_native_connector_probe",
            "sglang_kv_injection.connector_factory=company_sglang_patch_probe",
        ),
    ],
)
def test_read_databricks_engine_probe_targets_json_rejects_placeholder_connector_factory_metadata(
    tmp_path,
    backend,
    delegate_factory,
    metadata,
):
    path = tmp_path / "probe-targets.json"
    path.write_text(
        json.dumps(
            {
                "record_type": "document_kv.engine_probe_targets.v1",
                "schema_version": 1,
                "release_safe": True,
                "probes": [
                    {
                        "backend": backend,
                        "handoff_json": f"/Volumes/catalog/schema/volume/probes/{backend}-handoff.json",
                        "probe_factory": f"document_kv_cache.native_probe_factories:{backend}_native_probe_factory",
                        "output_json": f"/Volumes/catalog/schema/volume/probes/{backend}-probe.json",
                        "native_probe_delegate_factory": delegate_factory,
                        "metadata": [metadata],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="real module:attribute connector factory"):
        read_databricks_engine_probe_targets_json(path)


def test_databricks_engine_probe_config_rejects_known_delegate_backend_mismatch():
    with pytest.raises(ValueError, match="is for sglang, but expected_backend is vllm"):
        DatabricksEngineProbeJobConfig(
            handoff_json="/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            probe_factory="document_kv_cache.native_probe_factories:vllm_native_probe_factory",
            output_json="/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            expected_backend=ServingBackend.VLLM,
            single_user_name=SINGLE_USER_NAME,
            native_probe_delegate_factory="sglang_kv_injection.probe:build_native_connector_probe",
            metadata=("sglang_kv_injection.connector_factory=company_sglang_patch.probe:build_connector",),
        )


def test_legacy_databricks_engine_probe_targets_reject_known_delegate_missing_connector_factory(tmp_path):
    path = tmp_path / "probe-targets.json"
    path.write_text(
        json.dumps(
            {
                "record_type": "document_kv.engine_probe_targets.v1",
                "schema_version": 1,
                "release_safe": True,
                "probes": [
                    {
                        "backend": "sglang",
                        "handoff_json": "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
                        "probe_factory": "document_kv_cache.native_probe_factories:sglang_native_probe_factory",
                        "output_json": "/Volumes/catalog/schema/volume/probes/sglang-probe.json",
                        "native_probe_delegate_factory": "sglang_kv_injection.probe:build_native_connector_probe",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="build_document_kv_hicache_probe_connector"):
        legacy_engine_probe_job.read_databricks_engine_probe_targets_json(path)


def test_legacy_databricks_engine_probe_config_rejects_known_delegate_backend_mismatch():
    with pytest.raises(ValueError, match="is for vllm, but expected_backend is sglang"):
        legacy_engine_probe_job.DatabricksEngineProbeJobConfig(
            handoff_json="/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
            probe_factory="document_kv_cache.native_probe_factories:sglang_native_probe_factory",
            output_json="/Volumes/catalog/schema/volume/probes/sglang-probe.json",
            runner_python_file="dbfs:/benchmarks/run_engine_probe.py",
            expected_backend=ServingBackend.SGLANG,
            single_user_name=SINGLE_USER_NAME,
            native_probe_delegate_factory="vllm_kv_injection.probe:build_native_connector_probe",
            metadata=("vllm_kv_injection.connector_factory=company_vllm_patch.probe:build_connector",),
        )


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
                        "actions_output_json": "/Volumes/catalog/schema/volume/probes/vllm-actions.json",
                        "native_probe_factories_output_json": VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
                    },
                    {
                        "backend": "sglang",
                        "handoff_json": "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
                        "probe_factory": "document_kv_cache_sglang_probe:build_probe",
                        "output_json": "/Volumes/catalog/schema/volume/probes/sglang-probe.json",
                        "actions_output_json": "/Volumes/catalog/schema/volume/probes/sglang-actions.json",
                        "sglang_runtime_preflight_output_json": SGLANG_RUNTIME_PREFLIGHT_OUTPUT_JSON,
                        "sglang_runtime_preflight_launch_config_json": SGLANG_RUNTIME_PREFLIGHT_LAUNCH_CONFIG_JSON,
                        "native_probe_factories_output_json": SGLANG_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
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


def test_main_rejects_release_safe_targets_without_actions_output(tmp_path, capsys):
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
                        "native_probe_factories_output_json": VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
                    },
                    {
                        "backend": "sglang",
                        "handoff_json": "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
                        "probe_factory": "document_kv_cache_sglang_probe:build_probe",
                        "output_json": "/Volumes/catalog/schema/volume/probes/sglang-probe.json",
                        "sglang_runtime_preflight_output_json": SGLANG_RUNTIME_PREFLIGHT_OUTPUT_JSON,
                        "sglang_runtime_preflight_launch_config_json": SGLANG_RUNTIME_PREFLIGHT_LAUNCH_CONFIG_JSON,
                        "native_probe_factories_output_json": SGLANG_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
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

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["error_type"] == "ValueError"
    assert "actions_output_json" in output["error"]
    assert not payload_path.exists()


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
                        "sglang_runtime_preflight_output_json": SGLANG_RUNTIME_PREFLIGHT_OUTPUT_JSON,
                        "sglang_runtime_preflight_launch_config_json": SGLANG_RUNTIME_PREFLIGHT_LAUNCH_CONFIG_JSON,
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
            probe_targets=(_release_target("vllm"), _release_target("sglang")),
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
    assert "document_kv_cache._databricks_engine_probe_runner" in runner_text
    assert "document_kv_cache.databricks_engine_probe_job" not in runner_text
    assert "run_engine_probe_task" in runner_text
    assert "if exit_code:" in runner_text


def test_generated_engine_probe_runner_installs_wheel_before_forwarding_args(tmp_path):
    runner_path = tmp_path / "run_engine_probe.py"
    pip_calls_path = tmp_path / "pip-calls.jsonl"
    reexec_calls_path = tmp_path / "reexec-calls.jsonl"
    task_args_path = tmp_path / "task-args.json"
    events_path = tmp_path / "events.jsonl"
    venv_dir = tmp_path / "serving-venv"
    venv_python = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
    package_dir = tmp_path / "document_kv_cache"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "_databricks_engine_probe_runner.py").write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "",
                "with open(os.environ['RUNNER_EVENTS_JSONL'], 'a', encoding='utf-8') as handle:",
                "    handle.write(json.dumps({'event': 'engine_probe_runner_import'}) + '\\n')",
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
                "def _capture_check_call(argv, **kwargs):",
                "    with open(os.environ['RUNNER_EVENTS_JSONL'], 'a', encoding='utf-8') as handle:",
                "        handle.write(json.dumps({'event': 'subprocess_check_call'}) + '\\n')",
                "    with open(os.environ['PIP_CALLS_JSONL'], 'a', encoding='utf-8') as handle:",
                "        handle.write(json.dumps({'argv': argv, 'has_env': kwargs.get('env') is not None}) + '\\n')",
                "    return 0",
                "",
                "def _capture_call(argv, **kwargs):",
                "    with open(os.environ['RUNNER_EVENTS_JSONL'], 'a', encoding='utf-8') as handle:",
                "        handle.write(json.dumps({'event': 'subprocess_call'}) + '\\n')",
                "    with open(os.environ['REEXEC_CALLS_JSONL'], 'a', encoding='utf-8') as handle:",
                "        handle.write(json.dumps({'argv': argv, 'has_env': kwargs.get('env') is not None}) + '\\n')",
                "    return 0",
                "",
                "subprocess.check_call = _capture_check_call",
                "subprocess.call = _capture_call",
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
        "REEXEC_CALLS_JSONL": str(reexec_calls_path),
        "TASK_ARGS_JSON": str(task_args_path),
        "RUNNER_EVENTS_JSONL": str(events_path),
    }

    subprocess.run(
        [
            sys.executable,
            str(runner_path),
            "--serving-venv-dir",
            str(venv_dir),
            "--package-wheel-uri",
            "dbfs:/tmp/cachet/document_kv_cache-0.2.0-py3-none-any.whl",
            "--package-wheel-uri",
            "dbfs:/tmp/cachet/custom_vllm_probe_extension-0.1.0-py3-none-any.whl",
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

    pip_records = [json.loads(line) for line in pip_calls_path.read_text(encoding="utf-8").splitlines()]
    pip_calls = [record["argv"] for record in pip_records]
    assert Path(pip_calls[0][0]).resolve() == Path(sys.executable).resolve()
    assert pip_calls[0][1:] == ["-m", "venv", "--clear", str(venv_dir)]
    assert pip_calls[1:] == [
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        [str(venv_python), "-m", "pip", "install", "/dbfs/tmp/cachet/document_kv_cache-0.2.0-py3-none-any.whl"],
        [str(venv_python), "-m", "pip", "install", "/dbfs/tmp/cachet/custom_vllm_probe_extension-0.1.0-py3-none-any.whl"],
    ]
    assert pip_records[0]["has_env"] is False
    assert all(record["has_env"] is True for record in pip_records[1:])
    assert pip_calls[2][0] == str(venv_python)
    assert pip_calls[3][0] == str(venv_python)
    reexec_records = [json.loads(line) for line in reexec_calls_path.read_text(encoding="utf-8").splitlines()]
    reexec_calls = [record["argv"] for record in reexec_records]
    assert reexec_calls == [
        [
            str(venv_python),
            str(runner_path),
            "--skip-runtime-package-install",
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            "--expected-backend",
            "vllm",
        ]
    ]
    assert reexec_records == [{"argv": reexec_calls[0], "has_env": True}]
    assert not task_args_path.exists()

    subprocess.run(
        [
            sys.executable,
            str(runner_path),
            "--skip-runtime-package-install",
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

    assert json.loads(task_args_path.read_text(encoding="utf-8")) == [
        "--handoff-json",
        "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
        "--expected-backend",
        "vllm",
    ]
    events = [json.loads(line)["event"] for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert events == [
        "subprocess_check_call",
        "subprocess_check_call",
        "subprocess_check_call",
        "subprocess_check_call",
        "subprocess_call",
        "engine_probe_runner_import",
        "run_engine_probe_task",
    ]


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
            CUSTOM_VLLM_EXTENSION_WHEEL_URI,
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
        CUSTOM_VLLM_EXTENSION_WHEEL_URI,
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


def test_main_provider_backed_vllm_preset_writes_g6_payload(tmp_path):
    payload_path = tmp_path / "payload.json"
    fixture_dir = "/Volumes/catalog/schema/volume/probes/vllm-fixture"

    exit_code = main(
        [
            "--provider-backed-vllm-native-probe",
            "--fixture-output-dir",
            fixture_dir,
            "--probe-output-json",
            "/Volumes/catalog/schema/volume/probes/vllm-fixture/vllm-probe.json",
            "--actions-output-json",
            f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['actions']}",
            "--vllm-runtime-preflight-output-json",
            f"{fixture_dir}/vllm-runtime-preflight.json",
            "--native-probe-factories-output-json",
            f"{fixture_dir}/vllm-native-probe-factories.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--wheel-uri",
            WHEEL_URI,
            "--node-type-id",
            "g6.8xlarge",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--metadata",
            "probe.source=qa-g6",
            "--release-safe",
            "--output-json",
            str(payload_path),
        ]
    )

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    task = payload["tasks"][0]
    cluster = task["new_cluster"]
    parameters = task["spark_python_task"]["parameters"]

    assert exit_code == 0
    assert task["task_key"] == f"{DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY}_vllm"
    assert cluster["node_type_id"] == "g6.8xlarge"
    assert cluster["driver_node_type_id"] == "g6.8xlarge"
    assert cluster["spark_env_vars"] == {
        VLLM_NATIVE_PROBE_DELEGATE_ENV: VLLM_NATIVE_PROBE_DELEGATE_FACTORY
    }
    assert parameters[:12] == [
        "--native-probe-factories-output-json",
        f"{fixture_dir}/vllm-native-probe-factories.json",
        "--fixture-output-dir",
        fixture_dir,
        "--fixture-backend",
        "vllm",
        "--fixture-payload-mode",
        "merged",
        "--vllm-runtime-preflight-output-json",
        f"{fixture_dir}/vllm-runtime-preflight.json",
        "--vllm-runtime-preflight-layer-names-json",
        f"{fixture_dir}/vllm-layer-names.json",
    ]
    assert parameters[parameters.index("--probe-factory") + 1] == VLLM_NATIVE_PROBE_FACTORY
    assert parameters[parameters.index("--expected-backend") + 1] == "vllm"
    assert _parameter_values(parameters, "--metadata") == [
        "probe.source=qa-g6",
        VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,
    ]
    assert _parameter_values(parameters, "--pip-package") == list(VLLM_RUNTIME_PACKAGES)
    assert _parameter_values(parameters, "--pip-override-package") == [VLLM_FIPS_OPENCV_OVERRIDE_PACKAGE]
    assert _parameter_values(parameters, "--package-wheel-uri") == [WHEEL_URI]
    assert "--allow-non-native-probe" not in parameters
    assert "--engine-version" not in parameters


def test_main_provider_backed_sglang_preset_writes_g6_payload(tmp_path):
    payload_path = tmp_path / "payload.json"
    fixture_dir = "/Volumes/catalog/schema/volume/probes/sglang-fixture"

    exit_code = main(
        [
            "--provider-backed-sglang-native-probe",
            "--fixture-output-dir",
            fixture_dir,
            "--probe-output-json",
            "/Volumes/catalog/schema/volume/probes/sglang-fixture/sglang-probe.json",
            "--actions-output-json",
            f"{fixture_dir}/{DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES['actions']}",
            "--sglang-runtime-preflight-output-json",
            f"{fixture_dir}/sglang-runtime-preflight.json",
            "--sglang-runtime-preflight-launch-config-json",
            f"{fixture_dir}/sglang-launch-config.json",
            "--native-probe-factories-output-json",
            f"{fixture_dir}/sglang-native-probe-factories.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--wheel-uri",
            WHEEL_URI,
            "--node-type-id",
            "g6.8xlarge",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--metadata",
            "probe.source=qa-g6",
            "--release-safe",
            "--output-json",
            str(payload_path),
        ]
    )

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    task = payload["tasks"][0]
    cluster = task["new_cluster"]
    parameters = task["spark_python_task"]["parameters"]

    assert exit_code == 0
    assert task["task_key"] == f"{DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY}_sglang"
    assert cluster["node_type_id"] == "g6.8xlarge"
    assert cluster["driver_node_type_id"] == "g6.8xlarge"
    assert cluster["spark_env_vars"] == {
        SGLANG_NATIVE_PROBE_DELEGATE_ENV: SGLANG_NATIVE_PROBE_DELEGATE_FACTORY
    }
    assert parameters[:12] == [
        "--native-probe-factories-output-json",
        f"{fixture_dir}/sglang-native-probe-factories.json",
        "--fixture-output-dir",
        fixture_dir,
        "--fixture-backend",
        "sglang",
        "--fixture-payload-mode",
        "merged",
        "--sglang-runtime-preflight-output-json",
        f"{fixture_dir}/sglang-runtime-preflight.json",
        "--sglang-runtime-preflight-launch-config-json",
        f"{fixture_dir}/sglang-launch-config.json",
    ]
    assert parameters[parameters.index("--probe-factory") + 1] == SGLANG_NATIVE_PROBE_FACTORY
    assert parameters[parameters.index("--expected-backend") + 1] == "sglang"
    assert _parameter_values(parameters, "--metadata") == [
        "probe.source=qa-g6",
        SGLANG_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,
    ]
    assert _parameter_values(parameters, "--pip-package") == list(SGLANG_RUNTIME_PACKAGES)
    assert _parameter_values(parameters, "--package-wheel-uri") == [WHEEL_URI]
    assert "--actions-output-json" not in parameters
    assert "--allow-non-native-probe" not in parameters
    assert "--engine-version" not in parameters


def test_main_matrix_derives_node_type_from_g5_hardware_target(tmp_path):
    payload_path = tmp_path / "payload.json"
    backend_config_path = tmp_path / "targets.json"
    backend_config_path.write_text(
        json.dumps(
            [
                {
                    "backend": "vllm",
                    "handoff_json": "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
                    "probe_factory": "document_kv_cache_vllm_probe:build_probe",
                    "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                    "payload_uri": "/Volumes/catalog/schema/volume/probes/vllm-payload.kv",
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


def test_main_matrix_preserves_legacy_g5_node_type_without_hardware_target(tmp_path):
    payload_path = tmp_path / "payload.json"
    backend_config_path = tmp_path / "targets.json"
    backend_config_path.write_text(
        json.dumps(
            [
                {
                    "backend": "vllm",
                    "handoff_json": "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
                    "probe_factory": "document_kv_cache_vllm_probe:build_probe",
                    "output_json": "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
                    "payload_uri": "/Volumes/catalog/schema/volume/probes/vllm-payload.kv",
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


def test_main_provider_backed_vllm_preset_requires_runtime_preflight_in_release_safe_mode(
    capsys,
    tmp_path,
):
    payload_path = tmp_path / "payload.json"

    exit_code = main(
        [
            "--provider-backed-vllm-native-probe",
            "--fixture-output-dir",
            "/Volumes/catalog/schema/volume/probes/vllm-fixture",
            "--probe-output-json",
            "/Volumes/catalog/schema/volume/probes/vllm-fixture/vllm-probe.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--wheel-uri",
            WHEEL_URI,
            "--native-probe-factories-output-json",
            "/Volumes/catalog/schema/volume/probes/vllm-fixture/vllm-native-probe-factories.json",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--release-safe",
            "--output-json",
            str(payload_path),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["error_type"] == "ValueError"
    assert "release-safe provider-backed vLLM" in output["error"]
    assert not payload_path.exists()


def test_main_provider_backed_sglang_preset_requires_runtime_preflight_in_release_safe_mode(
    capsys,
    tmp_path,
):
    payload_path = tmp_path / "payload.json"

    exit_code = main(
        [
            "--provider-backed-sglang-native-probe",
            "--fixture-output-dir",
            "/Volumes/catalog/schema/volume/probes/sglang-fixture",
            "--probe-output-json",
            "/Volumes/catalog/schema/volume/probes/sglang-fixture/sglang-probe.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--wheel-uri",
            WHEEL_URI,
            "--native-probe-factories-output-json",
            "/Volumes/catalog/schema/volume/probes/sglang-fixture/sglang-native-probe-factories.json",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--release-safe",
            "--output-json",
            str(payload_path),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["error_type"] == "ValueError"
    assert "release-safe SGLang" in output["error"]
    assert not payload_path.exists()


@pytest.mark.parametrize(
    ("extra_args", "expected_error"),
    [
        (("--expected-backend", "sglang"), "--expected-backend"),
        (("--probe-factory", "custom_vllm_probe:build_probe"), "--probe-factory"),
        (("--native-probe-delegate-factory", "custom_vllm_probe:build_delegate"), "--native-probe-delegate-factory"),
        (
            ("--metadata", "vllm_kv_injection.connector_factory=custom_vllm_probe:build_connector"),
            VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,
        ),
        (("--extra-pip-package", "vllm==0.22.0"), VLLM_RUNTIME_PACKAGE),
        (("--extra-pip-package", "transformers==5.11.0"), "transformers==5.12.1"),
        (("--extra-wheel-uri", CUSTOM_VLLM_EXTENSION_WHEEL_URI), "--extra-wheel-uri"),
        (("--engine-version", "debug-vllm"), "--engine-version"),
        (("--allow-non-native-probe",), "--allow-non-native-probe"),
        (("--fixture-payload-mode", "segmented"), "--fixture-payload-mode"),
    ],
)
def test_main_provider_backed_vllm_preset_rejects_conflicting_values(
    capsys,
    tmp_path,
    extra_args,
    expected_error,
):
    payload_path = tmp_path / "payload.json"

    exit_code = main(
        [
            "--provider-backed-vllm-native-probe",
            "--fixture-output-dir",
            "/Volumes/catalog/schema/volume/probes/vllm-fixture",
            "--probe-output-json",
            "/Volumes/catalog/schema/volume/probes/vllm-fixture/vllm-probe.json",
            "--vllm-runtime-preflight-output-json",
            "/Volumes/catalog/schema/volume/probes/vllm-fixture/vllm-runtime-preflight.json",
            "--vllm-runtime-preflight-layer-names-json",
            "/Volumes/catalog/schema/volume/probes/vllm-fixture/vllm-layer-names.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(payload_path),
            *extra_args,
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["error_type"] == "ValueError"
    assert expected_error in output["error"]
    assert not payload_path.exists()


@pytest.mark.parametrize(
    ("extra_args", "expected_error"),
    [
        (("--expected-backend", "vllm"), "--expected-backend"),
        (("--probe-factory", "custom_sglang_probe:build_probe"), "--probe-factory"),
        (
            ("--native-probe-delegate-factory", "custom_sglang_probe:build_delegate"),
            "--native-probe-delegate-factory",
        ),
        (
            ("--metadata", "sglang_kv_injection.connector_factory=custom_sglang_probe:build_connector"),
            SGLANG_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA,
        ),
        (("--extra-pip-package", "sglang==0.5.9"), SGLANG_RUNTIME_PACKAGE),
        (("--extra-wheel-uri", CUSTOM_SGLANG_EXTENSION_WHEEL_URI), "--extra-wheel-uri"),
        (("--engine-version", "debug-sglang"), "--engine-version"),
        (("--allow-non-native-probe",), "--allow-non-native-probe"),
        (("--fixture-payload-mode", "segmented"), "--fixture-payload-mode"),
        (("--provider-backed-vllm-native-probe",), "mutually exclusive"),
    ],
)
def test_main_provider_backed_sglang_preset_rejects_conflicting_values(
    capsys,
    tmp_path,
    extra_args,
    expected_error,
):
    payload_path = tmp_path / "payload.json"

    exit_code = main(
        [
            "--provider-backed-sglang-native-probe",
            "--fixture-output-dir",
            "/Volumes/catalog/schema/volume/probes/sglang-fixture",
            "--probe-output-json",
            "/Volumes/catalog/schema/volume/probes/sglang-fixture/sglang-probe.json",
            "--sglang-runtime-preflight-output-json",
            "/Volumes/catalog/schema/volume/probes/sglang-fixture/sglang-runtime-preflight.json",
            "--sglang-runtime-preflight-launch-config-json",
            "/Volumes/catalog/schema/volume/probes/sglang-fixture/sglang-launch-config.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--output-json",
            str(payload_path),
            *extra_args,
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["error_type"] == "ValueError"
    assert expected_error in output["error"]
    assert not payload_path.exists()


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
                        "actions_output_json": "/Volumes/catalog/schema/volume/probes/vllm-actions.json",
                        "native_probe_factories_output_json": VLLM_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
                    },
                    {
                        "backend": "sglang",
                        "handoff_json": "/Volumes/catalog/schema/volume/probes/sglang-handoff.json",
                        "probe_factory": "document_kv_cache_sglang_probe:build_probe",
                        "output_json": "/Volumes/catalog/schema/volume/probes/sglang-probe.json",
                        "actions_output_json": "/Volumes/catalog/schema/volume/probes/sglang-actions.json",
                        "sglang_runtime_preflight_output_json": SGLANG_RUNTIME_PREFLIGHT_OUTPUT_JSON,
                        "sglang_runtime_preflight_launch_config_json": SGLANG_RUNTIME_PREFLIGHT_LAUNCH_CONFIG_JSON,
                        "native_probe_factories_output_json": SGLANG_NATIVE_PROBE_FACTORIES_OUTPUT_JSON,
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
            "--serial-tasks",
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
    assert "depends_on" not in payload["tasks"][0]
    assert payload["tasks"][1]["depends_on"] == [
        {"task_key": f"{DEFAULT_DATABRICKS_ENGINE_PROBE_TASK_KEY}_vllm"}
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


def test_main_rejects_serial_tasks_without_backend_config_json(capsys):
    exit_code = main(
        [
            "--handoff-json",
            "/Volumes/catalog/schema/volume/probes/vllm-handoff.json",
            "--probe-factory",
            "document_kv_cache_vllm_probe:build_probe",
            "--probe-output-json",
            "/Volumes/catalog/schema/volume/probes/vllm-probe.json",
            "--runner-python-file",
            "dbfs:/benchmarks/run_engine_probe.py",
            "--expected-backend",
            "vllm",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--serial-tasks",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["error_type"] == "ValueError"
    assert "--serial-tasks requires --backend-config-json" in output["error"]


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

    help_text = " ".join(result.stdout.split())
    assert "Emit a Databricks runs/submit payload for a V1 AWS single-node GPU engine probe." in help_text


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
        "DEFAULT_SGLANG_ENGINE_PROBE_RUNTIME_PACKAGE",
        "DEFAULT_VLLM_ENGINE_PROBE_RUNTIME_PACKAGE",
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
        "SGLANG_NATIVE_PROBE_DELEGATE_FACTORY",
        "SGLANG_PROVIDER_BACKED_CONNECTOR_FACTORY",
        "SGLANG_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA",
        "VLLM_NATIVE_PROBE_DELEGATE_FACTORY",
        "VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY",
        "VLLM_PROVIDER_BACKED_CONNECTOR_FACTORY_METADATA",
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
