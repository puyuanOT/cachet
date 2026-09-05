import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from typing import Any

import pytest

import document_kv_cache._gpu_qualification_sentinels_v2 as sentinels_v2
import document_kv_cache.gpu_qualification_databricks as databricks_v1
import document_kv_cache.gpu_qualification_databricks_v2 as databricks_v2
from document_kv_cache.flashinfer_wheel_repack import (
    FLASHINFER_PACKAGE_VERSION,
    FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256,
    FLASHINFER_PATCHED_MANIFEST_FILE_SHA256,
    FLASHINFER_PATCHED_WHEEL_SHA256,
    FLASHINFER_TARGET_PATCHED_SHA256,
)
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
    GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    GPU_QUALIFICATION_VLLM_VERSION,
    build_gpu_qualification_system_cuda_parent_attestation,
    build_gpu_qualification_plan,
    canonical_gpu_qualification_json,
)
from document_kv_cache.gpu_qualification_v2 import (
    GPU_QUALIFICATION_V2_ARTIFACT_KEYS,
    GPU_QUALIFICATION_V2_FLASHINFER_RETURN_ANNOTATION,
    GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX,
    GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS,
    GPUQualificationArtifactPinsV2,
    build_gpu_qualification_plan_v2,
    build_gpu_runtime_verification_v2,
)
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
    PUBLICATION_CAMPAIGN_ID,
    PUBLICATION_CAMPAIGN_LEDGER_ID,
    PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
    PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX,
    PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS,
)
from document_kv_cache.runtime_artifact_closure import (
    RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256,
    RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
    VLLM_PATCHED_MANIFEST_SHA256,
    VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT,
    VLLM_RUNTIME_BASE_LOCK_FILENAME,
    VLLM_RUNTIME_BASE_LOCK_HASH_COUNT,
    VLLM_RUNTIME_BASE_LOCK_SHA256,
)
from document_kv_cache.serving_env import (
    GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL,
    GPU_RUNTIME_PYTHONWARNINGS,
)


def test_v2_staggered_progress_rejects_batch_authority_subclasses():
    class SmuggledBatchAuthorization(
        databricks_v2.DatabricksBatchReservationAuthorization
    ):
        pass

    with pytest.raises(TypeError, match="atomic batch authority"):
        databricks_v2._staggered_batch_progress_v2(
            databricks_v2.DatabricksClusterHourLedger(ledger_id="test-ledger"),
            object.__new__(SmuggledBatchAuthorization),
        )


def _pins() -> GPUQualificationArtifactPinsV2:
    return GPUQualificationArtifactPinsV2(
        runtime_lock_sha256=VLLM_RUNTIME_BASE_LOCK_SHA256,
        patched_vllm_wheel_sha256=GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        patched_flashinfer_wheel_sha256=FLASHINFER_PATCHED_WHEEL_SHA256,
        runtime_closure_manifest_sha256=RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        package_wheel_sha256="a" * 64,
        cachet_source_tree_sha256="b" * 64,
        runner_sha256=databricks_v2.GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SHA256,
        input_bundle_sha256=GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    )


def _plan() -> dict[str, Any]:
    return build_gpu_qualification_plan_v2(
        campaign_id=PUBLICATION_CAMPAIGN_ID,
        campaign_record_sha256=PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
        campaign_ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
        campaign_ledger_path_sha256=PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
        campaign_ledger_prefix=GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX,
        campaign_opening_terminal_gpu_hours=(
            GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS
        ),
        artifact_pins=_pins(),
    )


def _artifact_uris() -> dict[str, str]:
    return {
        key: f"dbfs:/Volumes/catalog/schema/volume/{key}/artifact"
        for key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS
    }


def _attestation() -> dict[str, Any]:
    return {
        "base_lock_distribution_count": 195,
        "base_lock_hash_count": 4137,
        "base_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
        "cachet_package_version": "0.2.0",
        "flashinfer_annotation": GPU_QUALIFICATION_V2_FLASHINFER_RETURN_ANNOTATION,
        "flashinfer_direct_url": "file:///runtime/flashinfer.whl",
        "flashinfer_import_ok": True,
        "closure_bound_flashinfer_manifest_closed_record_sha256": (
            FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256
        ),
        "closure_bound_flashinfer_manifest_file_sha256": (
            FLASHINFER_PATCHED_MANIFEST_FILE_SHA256
        ),
        "closure_bound_vllm_manifest_file_sha256": VLLM_PATCHED_MANIFEST_SHA256,
        "flashinfer_member_sha256": FLASHINFER_TARGET_PATCHED_SHA256,
        "flashinfer_package_version": FLASHINFER_PACKAGE_VERSION,
        "flashinfer_wheel_sha256": FLASHINFER_PATCHED_WHEEL_SHA256,
        "installed_distribution_count": 198,
        "ok": True,
        "packaged_base_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
        "pip_check_ok": True,
        "runtime_closure_closed_record_sha256": (
            RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256
        ),
        "runtime_closure_file_sha256": RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        "system_cuda_parent_attestation": (
            build_gpu_qualification_system_cuda_parent_attestation(
                distribution_root="/databricks/python/lib/python3.11/site-packages",
                libcudart_path=(
                    "/databricks/python/lib/python3.11/site-packages/"
                    "nvidia/cuda_runtime/lib/libcudart.so.12"
                ),
            )
        ),
        "unexpected_distributions": [],
        "vllm_direct_url": "file:///runtime/vllm-0.27.1%2Bcu129.whl",
        "vllm_member_sha256": {
            "vllm/model_executor/layers/attention/attention.py": (
                "5735acfb390cf344caeec950c2f286344bcd84721ce287e0a56701f2a18bc839"
            ),
            "vllm/v1/attention/backends/triton_attn.py": (
                "4dae0ff6c4ee8f11c1f195151a11673d595d457c413032e7bae7550913f94390"
            ),
            "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py": (
                "0682ca7bc56edf7cea5419188a81c78510b54192471472b160aa447ac0ceeb08"
            ),
        },
        "vllm_package_version": GPU_QUALIFICATION_VLLM_VERSION,
        "vllm_wheel_sha256": GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        "with_flashinfer_distribution_count": 196,
        "with_vllm_distribution_count": 197,
    }


def _payloads() -> tuple[dict[str, Any], ...]:
    return databricks_v2.render_gpu_qualification_submit_payloads_v2(
        _plan(),
        single_user_name="test@example.com",
        artifact_uris=_artifact_uris(),
        output_root="dbfs:/Volumes/catalog/schema/volume/output",
    )


def test_v2_bootstrap_and_renderer_have_stable_golden_bytes() -> None:
    assert databricks_v2.GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SHA256 == (
        "322606772efcfc5209bd4c7426f5f05605f7c0fd5ff5ea541128b0a36eb93493"
    )
    assert len(databricks_v2.GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SCRIPT) == 25315
    plan = _plan()
    assert plan["closed_record_sha256"] == (
        "1aff4e6d1e475f89f58a234f26bb1592c803eeca6facaaa827c5d34edf8fc296"
    )
    assert len(canonical_gpu_qualification_json(plan).encode("utf-8")) == 17306
    payloads = _payloads()
    payload_bytes = canonical_gpu_qualification_json(
        {"payloads": list(payloads)}
    ).encode("utf-8")
    assert len(payload_bytes) == 116041
    assert hashlib.sha256(payload_bytes).hexdigest() == (
        "62fb40df6d907a236d5c42a262b83599f44f692c57b308aa2007fca8814a5fed"
    )


def test_v2_renderer_uses_plan_pins_and_eight_uris_with_safe_argument_headroom() -> (
    None
):
    payloads = _payloads()
    assert len(payloads) == 14
    sizes = []
    for payload in payloads:
        task = payload["tasks"][0]
        parameters = task["spark_python_task"]["parameters"]
        sizes.append(
            len(
                json.dumps(
                    parameters,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        )
        assert len(parameters) == 34
        assert parameters.count("--artifact-uri") == 8
        assert "--artifact-sha256" not in parameters
        assert "--runner-uri" not in parameters
        assert "--package-wheel-uri" not in parameters
        assert "--patched-vllm-wheel-uri" not in parameters
        assert (
            task["spark_python_task"]["python_file"]
            == (_artifact_uris()["runner_sha256"])
        )
        assert "spark_env_vars" not in task["new_cluster"]
    assert min(sizes) == 7204
    assert max(sizes) == 7276
    assert (
        max(sizes) < databricks_v2.GPU_QUALIFICATION_V2_DATABRICKS_PARAMETERS_MAX_BYTES
    )

    def production_hash(label: str) -> str:
        return hashlib.sha256(label.encode("ascii")).hexdigest()

    production_pins = GPUQualificationArtifactPinsV2(
        runtime_lock_sha256=VLLM_RUNTIME_BASE_LOCK_SHA256,
        patched_vllm_wheel_sha256=GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        patched_flashinfer_wheel_sha256=FLASHINFER_PATCHED_WHEEL_SHA256,
        runtime_closure_manifest_sha256=RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        package_wheel_sha256=production_hash("production-package"),
        cachet_source_tree_sha256=production_hash("production-source"),
        runner_sha256=databricks_v2.GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SHA256,
        input_bundle_sha256=GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    )
    production_plan = build_gpu_qualification_plan_v2(
        campaign_id=PUBLICATION_CAMPAIGN_ID,
        campaign_record_sha256=PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
        campaign_ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
        campaign_ledger_path_sha256=PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
        campaign_ledger_prefix=GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX,
        campaign_opening_terminal_gpu_hours=(
            GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS
        ),
        artifact_pins=production_pins,
    )
    production_base = (
        "dbfs:/Volumes/datascience_qa/kv_cache_restaurant_cls/"
        "kv_cache_storage_benchmark/vllm_0271_publication_v2"
    )
    production_uris = {
        "cachet_source_tree_sha256": (
            f"{production_base}/inputs/cachet-source-v2/"
            f"{production_pins.cachet_source_tree_sha256}/cachet-source-closure.json"
        ),
        "input_bundle_sha256": (
            f"{production_base}/inputs/main-latency/"
            f"{production_pins.input_bundle_sha256}"
        ),
        "package_wheel_sha256": (
            f"{production_base}/inputs/cachet-wheel/"
            f"{production_pins.package_wheel_sha256}/"
            "cachet_kv-0.2.0-py3-none-any.whl"
        ),
        "patched_flashinfer_wheel_sha256": (
            f"{production_base}/inputs/flashinfer-wheel/"
            f"{production_pins.patched_flashinfer_wheel_sha256}/"
            "flashinfer_python-0.6.16.post3-1cachetpy31104e032c70234e876-"
            "py3-none-any.whl"
        ),
        "patched_vllm_wheel_sha256": (
            f"{production_base}/inputs/vllm-wheel/"
            f"{production_pins.patched_vllm_wheel_sha256}/"
            "vllm-0.27.1+cu129-1cachete5m265120c48a9352b9e-cp38-abi3-"
            "manylinux_2_28_x86_64.whl"
        ),
        "runner_sha256": (
            f"{production_base}/inputs/runner/"
            f"{production_plan['closed_record_sha256']}/"
            "gpu-qualification-bootstrap-v2.py"
        ),
        "runtime_closure_manifest_sha256": (
            f"{production_base}/inputs/runtime-closure/"
            f"{production_pins.runtime_closure_manifest_sha256}/"
            "vllm-0.27.1-flashinfer-0.6.16.post3-runtime-closure.json"
        ),
        "runtime_lock_sha256": (
            f"{production_base}/inputs/runtime-lock/"
            f"{production_pins.runtime_lock_sha256}/"
            "vllm-0.27.1-cu129-py311-manylinux_2_35-flashinfer-direct.lock"
        ),
    }
    production_output_root = (
        f"{production_base}/qualification-results-v2-"
        f"{production_hash('production-commit')[:12]}-"
        f"{production_pins.cachet_source_tree_sha256[:16]}"
    )
    production_payloads = databricks_v2.render_gpu_qualification_submit_payloads_v2(
        production_plan,
        single_user_name="pliu@opentable.com",
        artifact_uris=production_uris,
        output_root=production_output_root,
    )
    production_sizes = [
        len(
            json.dumps(
                payload["tasks"][0]["spark_python_task"]["parameters"],
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for payload in production_payloads
    ]
    assert min(production_sizes) == 8765
    assert max(production_sizes) == 8837
    assert max(production_sizes) <= (
        databricks_v2.GPU_QUALIFICATION_V2_DATABRICKS_PARAMETERS_MAX_BYTES - 600
    )


def test_v2_full_local_payload_contract_rejects_tamper() -> None:
    payloads = _payloads()
    assert (
        databricks_v2.validate_gpu_qualification_submit_payloads_v2(
            payloads,
            plan_record=_plan(),
            single_user_name="test@example.com",
            artifact_uris=_artifact_uris(),
            output_root="dbfs:/Volumes/catalog/schema/volume/output",
        )
        == payloads
    )
    tampered = json.loads(json.dumps(payloads))
    tampered[0]["tasks"][0]["max_retries"] = 1
    with pytest.raises(ValueError, match="payload closure"):
        databricks_v2.validate_gpu_qualification_submit_payloads_v2(
            tampered,
            plan_record=_plan(),
            single_user_name="test@example.com",
            artifact_uris=_artifact_uris(),
            output_root="dbfs:/Volumes/catalog/schema/volume/output",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_retries", False),
        ("timeout_seconds", 14_400.0),
    ],
)
def test_v2_full_local_payload_contract_rejects_type_confusion(
    field: str,
    value: object,
) -> None:
    payloads = json.loads(json.dumps(_payloads()))
    if field == "max_retries":
        payloads[0]["tasks"][0][field] = value
    else:
        payloads[0][field] = value
    with pytest.raises(ValueError, match="payload closure"):
        databricks_v2.validate_gpu_qualification_submit_payloads_v2(
            payloads,
            plan_record=_plan(),
            single_user_name="test@example.com",
            artifact_uris=_artifact_uris(),
            output_root="dbfs:/Volumes/catalog/schema/volume/output",
        )


def test_v2_payload_contract_distinguishes_integer_from_boolean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        databricks_v2,
        "render_gpu_qualification_submit_payloads_v2",
        lambda *_args, **_kwargs: ({"reviewed": True},),
    )
    with pytest.raises(ValueError, match="payload closure"):
        databricks_v2.validate_gpu_qualification_submit_payloads_v2(
            ({"reviewed": 1},),
            plan_record={},
            single_user_name="test@example.com",
            artifact_uris={},
            output_root="dbfs:/Volumes/catalog/schema/volume/output",
        )


@pytest.mark.parametrize("mutation", ["missing", "reordered", "conflated"])
def test_v2_renderer_rejects_open_or_conflated_uri_maps(mutation: str) -> None:
    uris = _artifact_uris()
    if mutation == "missing":
        uris.pop("runtime_lock_sha256")
    elif mutation == "reordered":
        uris = dict(reversed(tuple(uris.items())))
    else:
        uris["runtime_lock_sha256"] = uris["runner_sha256"]
    with pytest.raises(ValueError):
        databricks_v2.render_gpu_qualification_submit_payloads_v2(
            _plan(),
            single_user_name="test@example.com",
            artifact_uris=uris,
            output_root="dbfs:/Volumes/catalog/schema/volume/output",
        )


def test_v2_renderer_rejects_physical_path_aliases_and_accepts_file_uri_form() -> None:
    uris = _artifact_uris()
    uris["cachet_source_tree_sha256"] = (
        "file:/Volumes/catalog/schema/volume/cachet-source/artifact"
    )
    assert (
        len(
            databricks_v2.render_gpu_qualification_submit_payloads_v2(
                _plan(),
                single_user_name="test@example.com",
                artifact_uris=uris,
                output_root="dbfs:/Volumes/catalog/schema/volume/output",
            )
        )
        == 14
    )
    uris["runtime_lock_sha256"] = "dbfs:/Volumes/catalog/schema/volume/shared/artifact"
    uris["cachet_source_tree_sha256"] = (
        "file:/Volumes/catalog/schema/volume/shared/artifact"
    )
    with pytest.raises(ValueError, match="distinct paths"):
        databricks_v2.render_gpu_qualification_submit_payloads_v2(
            _plan(),
            single_user_name="test@example.com",
            artifact_uris=uris,
            output_root="dbfs:/Volumes/catalog/schema/volume/output",
        )


def test_v1_and_v2_plan_decoders_reject_cross_version_records() -> None:
    v2_plan = _plan()
    encoded_v2 = databricks_v2._encode_plan_parameter(
        canonical_gpu_qualification_json(v2_plan)
    )
    with pytest.raises(ValueError):
        databricks_v1._decode_qualification_plan_parameter(
            encoded_v2,
            expected_plan_sha256=v2_plan["closed_record_sha256"],
        )
    v1_plan = build_gpu_qualification_plan(
        campaign_id=PUBLICATION_CAMPAIGN_ID,
        campaign_record_sha256=PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
        campaign_ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
        campaign_ledger_path_sha256=PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
        campaign_ledger_prefix=PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX,
        campaign_opening_terminal_gpu_hours=(
            PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
        ),
        artifact_pins=_pins().v1_projection(),
    )
    encoded_v1 = databricks_v2._encode_plan_parameter(
        canonical_gpu_qualification_json(v1_plan)
    )
    with pytest.raises(ValueError):
        databricks_v2._decode_plan_parameter(
            encoded_v1,
            expected_plan_sha256=v1_plan["closed_record_sha256"],
        )


def _bootstrap_case(
    tmp_path: Path,
) -> tuple[dict[str, Any], list[str], dict[str, Path]]:
    runner = tmp_path / "gpu-qualification-v2-bootstrap.py"
    assert databricks_v2.write_gpu_qualification_bootstrap_runner_v2(runner) == (
        databricks_v2.GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SHA256
    )
    namespace = {"__name__": "bootstrap_unit_test"}
    exec(compile(runner.read_bytes(), str(runner), "exec"), namespace)
    paths: dict[str, Path] = {}
    for key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS:
        if key == "runner_sha256":
            paths[key] = runner
        elif key == "input_bundle_sha256":
            paths[key] = tmp_path / key
            paths[key].mkdir()
        elif key == "package_wheel_sha256":
            paths[key] = tmp_path / "cachet_kv-0.2.0-py3-none-any.whl"
            paths[key].write_bytes(key.encode("ascii"))
        else:
            paths[key] = tmp_path / f"{key}.bin"
            paths[key].write_bytes(key.encode("ascii"))
    digest = {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in paths.items()
        if path.is_file()
    }
    pins = GPUQualificationArtifactPinsV2(
        runtime_lock_sha256=VLLM_RUNTIME_BASE_LOCK_SHA256,
        patched_vllm_wheel_sha256=GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        patched_flashinfer_wheel_sha256=FLASHINFER_PATCHED_WHEEL_SHA256,
        runtime_closure_manifest_sha256=RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        package_wheel_sha256=digest["package_wheel_sha256"],
        cachet_source_tree_sha256=digest["cachet_source_tree_sha256"],
        runner_sha256=digest["runner_sha256"],
        input_bundle_sha256=GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    )
    plan = build_gpu_qualification_plan_v2(
        campaign_id=PUBLICATION_CAMPAIGN_ID,
        campaign_record_sha256=PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
        campaign_ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
        campaign_ledger_path_sha256=PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
        campaign_ledger_prefix=GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX,
        campaign_opening_terminal_gpu_hours=(
            GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS
        ),
        artifact_pins=pins,
    )
    job_id = plan["cloud_qualification"]["jobs"][0]["job_id"]
    plan_digest = plan["closed_record_sha256"]
    argv = databricks_v2._runner_parameters_v2(
        encoded_plan=databricks_v2._encode_plan_parameter(
            canonical_gpu_qualification_json(plan)
        ),
        plan_digest=plan_digest,
        job_id=job_id,
        reservation_attempt_id=(
            databricks_v1.gpu_qualification_reservation_attempt_id(plan_digest, job_id)
        ),
        output_json=(
            "dbfs:/Volumes/catalog/schema/volume/output/"
            f"{plan_digest}/{job_id}/gpu-job-result.json"
        ),
        work_dir=(
            f"{databricks_v2.GPU_QUALIFICATION_V2_LOCAL_WORK_ROOT}/"
            f"{plan_digest}/{job_id}"
        ),
        artifact_uris={key: path.as_uri() for key, path in paths.items()},
    )
    reviewed_fixed = {
        "patched_flashinfer_wheel_sha256": FLASHINFER_PATCHED_WHEEL_SHA256,
        "patched_vllm_wheel_sha256": GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        "runtime_closure_manifest_sha256": RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        "runtime_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
    }
    key_by_path = {str(path): key for key, path in paths.items()}
    real_sha256 = namespace["_sha256"]

    def fixture_sha256(value: str) -> str:
        key = key_by_path.get(str(value))
        if key in reviewed_fixed:
            return reviewed_fixed[key]
        return real_sha256(value)

    namespace["_sha256"] = fixture_sha256
    return namespace, argv, paths


def _valid_runner_handoff(
    namespace: dict[str, Any],
    argv: list[str],
    *,
    cluster_id: str = "Cluster-ID_Exact.Case",
    sources: tuple[str, ...] = (
        "DATABRICKS_CLUSTER_ID",
        "spark.databricks.clusterUsageTags.clusterId",
    ),
) -> str:
    return namespace["_sealed_handoff"](
        argv,
        cluster_id=cluster_id,
        sources=sources,
        runner_sha256=databricks_v2.GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SHA256,
    )


def _reseal_handoff(record: dict[str, Any]) -> str:
    record["closed_record_sha256"] = ""
    record["closed_record_sha256"] = hashlib.sha256(
        databricks_v2._canonical_bootstrap_json_v2(record).encode("utf-8")
    ).hexdigest()
    return databricks_v2._canonical_bootstrap_json_v2(record)


def _set_sanitized_child_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setenv("PYTHONSAFEPATH", "1")
    monkeypatch.setenv("FLASHINFER_LOGGING_LEVEL", GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL)
    monkeypatch.setenv("PYTHONWARNINGS", GPU_RUNTIME_PYTHONWARNINGS)
    monkeypatch.setattr(
        sys,
        "flags",
        types.SimpleNamespace(safe_path=True, no_user_site=1),
    )
    monkeypatch.setattr(
        sys,
        "warnoptions",
        list(GPU_RUNTIME_PYTHONWARNINGS.split(",")),
    )


def test_v2_bootstrap_runner_and_emitted_child_stub_compile_exactly() -> None:
    namespace: dict[str, Any] = {"__name__": "bootstrap_compile_test"}
    runner_code = compile(
        databricks_v2.GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SCRIPT.encode("utf-8"),
        "gpu-qualification-v2-bootstrap.py",
        "exec",
    )
    exec(runner_code, namespace)
    child_stub = namespace["_CHILD_STUB"]
    compile(child_stub, "gpu-qualification-v2-child.py", "exec")
    first_cachet_import = child_stub.index("from document_kv_cache")
    for startup_check in (
        "_cachet_required_environment.items()",
        "_cachet_forbidden_environment",
        "sys.flags.safe_path",
        "sys.warnoptions",
        "os.environ.pop",
    ):
        assert child_stub.index(startup_check) < first_cachet_import

    exact_environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"}
    }
    exact_environment.update(
        {
            "FLASHINFER_LOGGING_LEVEL": GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL,
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONWARNINGS": GPU_RUNTIME_PYTHONWARNINGS,
        }
    )
    cases: list[tuple[list[str], dict[str, str], str]] = []
    missing_policy = dict(exact_environment)
    missing_policy.pop("PYTHONWARNINGS")
    cases.append(
        (
            [sys.executable, "-P", "-c", child_stub],
            missing_policy,
            "exact startup environment",
        )
    )
    hostile_policy = {
        **exact_environment,
        "FLASHINFER_LOGGING_LEVEL": "DEBUG",
        "PYTHONWARNINGS": "ignore",
    }
    cases.append(
        (
            [sys.executable, "-P", "-c", child_stub],
            hostile_policy,
            "exact startup environment",
        )
    )
    cases.append(
        (
            [sys.executable, "-P", "-W", "ignore", "-c", child_stub],
            exact_environment,
            "exact warning startup options",
        )
    )
    for command, environment, expected_error in cases:
        completed = subprocess.run(
            command,
            capture_output=True,
            env=environment,
            text=True,
        )
        assert completed.returncode != 0
        assert (
            f"RuntimeError: GPU qualification v2 child lacks its {expected_error}"
            in (completed.stderr)
        )
        assert "ModuleNotFoundError" not in completed.stderr

    parsed_runner = ast.parse(
        databricks_v2.GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SCRIPT,
        filename="gpu-qualification-v2-bootstrap.py",
    )
    main_guard = parsed_runner.body[-1]
    assert isinstance(main_guard, ast.If)
    main_code = compile(
        ast.Module(body=[main_guard], type_ignores=[]),
        "gpu-qualification-v2-bootstrap-main.py",
        "exec",
    )
    calls: list[tuple[list[str], dict[str, str]]] = []

    def execute_top_level(returncode: int) -> None:
        def run(argv: list[str], environment: dict[str, str]) -> int:
            calls.append((argv, environment))
            return returncode

        exec(
            main_code,
            {
                "__name__": "__main__",
                "_run": run,
                "os": types.SimpleNamespace(environ={"EXACT": "value"}),
                "sys": types.SimpleNamespace(argv=["runner.py", "sentinel"]),
            },
        )

    execute_top_level(0)
    with pytest.raises(SystemExit) as failure:
        execute_top_level(17)
    assert failure.value.code == 17
    assert calls == [
        (["sentinel"], {"EXACT": "value"}),
        (["sentinel"], {"EXACT": "value"}),
    ]


def test_v2_bootstrap_validates_transport_and_snapshots_package_before_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, argv, paths = _bootstrap_case(tmp_path)
    namespace["_spark_cluster_id"] = lambda: None
    base_env = {"DATABRICKS_CLUSTER_ID": "Cluster-ID_Exact.Case"}
    source_index = next(
        index
        for index, value in enumerate(argv)
        if value.startswith("cachet_source_tree_sha256=")
    )
    argv[source_index] = argv[source_index].replace("=file://", "=file:", 1)
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> types.SimpleNamespace:
        assert kwargs["check"] is True
        snapshot = Path(command[-1])
        assert snapshot != paths["package_wheel_sha256"]
        assert snapshot.name.endswith(".whl")
        assert snapshot.read_bytes() == paths["package_wheel_sha256"].read_bytes()
        assert Path(kwargs["cwd"]) == snapshot.parent
        assert (
            kwargs["env"]["FLASHINFER_LOGGING_LEVEL"]
            == GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL
        )
        assert kwargs["env"]["PYTHONSAFEPATH"] == "1"
        assert kwargs["env"]["PYTHONWARNINGS"] == GPU_RUNTIME_PYTHONWARNINGS
        calls.append(command)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    remaining, handoff, subprocess_env = namespace["_bootstrap"](argv, base_env)
    assert remaining == argv
    assert json.loads(handoff)["cluster_id"] == "Cluster-ID_Exact.Case"
    assert subprocess_env["DATABRICKS_CLUSTER_ID"] == "Cluster-ID_Exact.Case"
    assert len(calls) == 1
    assert calls[0][1:4] == ["-P", "-m", "pip"]
    assert "--no-deps" in calls[0]
    assert ["--only-binary", ":all:"] == calls[0][
        calls[0].index("--only-binary") : calls[0].index("--only-binary") + 2
    ]
    assert not Path(calls[0][-1]).exists()

    namespace["_spark_cluster_id"] = lambda: "Cluster-ID_Exact.Case"
    remaining, spark_only_handoff, spark_only_env = namespace["_bootstrap"](argv, {})
    assert remaining == argv
    assert json.loads(spark_only_handoff)["sources"] == [
        "spark.databricks.clusterUsageTags.clusterId"
    ]
    assert "DATABRICKS_CLUSTER_ID" not in spark_only_env
    assert len(calls) == 2

    paths["cachet_source_tree_sha256"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="cachet_source_tree_sha256 SHA-256 mismatch"):
        namespace["_bootstrap"](argv, base_env)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "unknown",
        "plan",
        "uri",
        "retry",
        "output",
        "work",
        "alias",
    ],
)
def test_v2_bootstrap_rejects_transport_tamper_before_install(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, argv, paths = _bootstrap_case(tmp_path)
    namespace["_spark_cluster_id"] = lambda: None
    tampered = list(argv)
    if mutation == "missing":
        tampered = tampered[:-1]
    elif mutation == "unknown":
        tampered[0] = "--unknown-plan"
    elif mutation == "plan":
        tampered[1] = tampered[1][:-1] + ("A" if tampered[1][-1] != "A" else "B")
    elif mutation == "uri":
        index = next(
            index
            for index, value in enumerate(tampered)
            if value.startswith("runtime_lock_sha256=")
        )
        tampered[index] = "runtime_lock_sha256=" + "d" * 64
    elif mutation == "retry":
        tampered[tampered.index("--retry-count") + 1] = "1"
    elif mutation == "output":
        tampered[tampered.index("--output-json") + 1] += ".tampered"
    elif mutation == "work":
        tampered[tampered.index("--work-dir") + 1] += ".tampered"
    else:
        runtime_lock_uri = paths["runtime_lock_sha256"].as_uri()
        source_index = next(
            index
            for index, value in enumerate(tampered)
            if value.startswith("cachet_source_tree_sha256=")
        )
        tampered[source_index] = (
            "cachet_source_tree_sha256=file:" + runtime_lock_uri.removeprefix("file://")
        )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command),
    )
    with pytest.raises(ValueError):
        namespace["_bootstrap"](
            tampered,
            {"DATABRICKS_CLUSTER_ID": "Cluster-ID_Exact.Case"},
        )
    assert calls == []


def test_v2_bootstrap_handoff_uses_one_snapshot_and_isolates_private_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, argv, _paths = _bootstrap_case(tmp_path)
    cluster_id = "Cluster-ID_Exact.Case"
    base_env = {
        "DATABRICKS_CLUSTER_ID": cluster_id,
        "DB_CLUSTER_ID": cluster_id,
        "FLASHINFER_LOGGING_LEVEL": "DEBUG",
        "KEEP_FROM_BASE": "original",
        "PIP_INDEX_URL": "https://example.invalid/simple",
        "PYTHONPATH": "/tmp/untrusted",
        "PYTHONWARNINGS": "ignore",
    }
    spark_calls: list[str] = []

    def spark_cluster_id() -> str:
        spark_calls.append("spark")
        monkeypatch.setenv("DATABRICKS_CLUSTER_ID", "mutated-after-snapshot")
        monkeypatch.setenv("KEEP_FROM_BASE", "mutated-after-snapshot")
        return cluster_id

    namespace["_spark_cluster_id"] = spark_cluster_id
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **kwargs: Any) -> types.SimpleNamespace:
        calls.append((list(command), dict(kwargs)))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)

    assert namespace["_run"](argv, base_env) == 0
    assert spark_calls == ["spark"]
    assert len(calls) == 2
    pip_command, pip_kwargs = calls[0]
    child_command, child_kwargs = calls[1]
    assert pip_command[1:4] == ["-P", "-m", "pip"]
    assert child_command[1:3] == ["-P", "-c"]
    assert child_command[3].index("os.environ.pop") < child_command[3].index(
        "from document_kv_cache"
    )
    private_name = namespace["_HANDOFF_ENV"]
    pip_env = pip_kwargs["env"]
    child_env = child_kwargs["env"]
    assert private_name not in pip_env
    raw_handoff = child_env[private_name]
    assert {key: value for key, value in child_env.items() if key != private_name} == (
        pip_env
    )
    assert pip_env["DATABRICKS_CLUSTER_ID"] == cluster_id
    assert pip_env["KEEP_FROM_BASE"] == "original"
    assert "PIP_INDEX_URL" not in pip_env
    assert "PYTHONPATH" not in pip_env
    assert pip_env["PIP_CONFIG_FILE"] == os.devnull
    assert pip_env["FLASHINFER_LOGGING_LEVEL"] == GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL
    assert pip_env["PYTHONWARNINGS"] == GPU_RUNTIME_PYTHONWARNINGS
    assert {
        key: child_env[key] for key in ("FLASHINFER_LOGGING_LEVEL", "PYTHONWARNINGS")
    } == {
        "FLASHINFER_LOGGING_LEVEL": GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL,
        "PYTHONWARNINGS": GPU_RUNTIME_PYTHONWARNINGS,
    }
    handoff = json.loads(raw_handoff)
    assert handoff["cluster_id"] == cluster_id
    assert handoff["sources"] == [
        "DATABRICKS_CLUSTER_ID",
        "DB_CLUSTER_ID",
        "spark.databricks.clusterUsageTags.clusterId",
    ]
    assert handoff["spark_checked"] is True
    assert handoff["runner_sha256"] == (
        databricks_v2.GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SHA256
    )
    assert (
        handoff["argv_sha256"]
        == hashlib.sha256(
            namespace["_canonical_json"](argv).encode("utf-8")
        ).hexdigest()
    )
    assert len(raw_handoff.encode("utf-8")) <= namespace["_HANDOFF_MAX_BYTES"]
    with pytest.raises(ValueError, match="handoff exceeds its size cap"):
        namespace["_sealed_handoff"](
            argv,
            cluster_id="x" * (namespace["_HANDOFF_MAX_BYTES"] + 1),
            sources=("DATABRICKS_CLUSTER_ID",),
            runner_sha256=(databricks_v2.GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SHA256),
        )
    assert (
        databricks_v2.GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SCRIPT.count(
            "dict(os.environ)"
        )
        == 1
    )


@pytest.mark.parametrize(
    "case",
    [
        "inherited-private-empty",
        "inherited-private-sealed",
        "missing",
        "environment-conflict",
        "spark-conflict",
        "spark-failure",
        "invalid-environment",
        "invalid-spark",
        "self-tamper",
    ],
)
def test_v2_bootstrap_cluster_identity_fails_before_subprocess(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, argv, paths = _bootstrap_case(tmp_path)
    cluster_id = "Cluster-ID_Exact.Case"
    base_env: dict[str, str] = {"DATABRICKS_CLUSTER_ID": cluster_id}
    spark_calls: list[str] = []

    def spark_value(value: object) -> object:
        spark_calls.append("spark")
        return value

    namespace["_spark_cluster_id"] = lambda: spark_value(None)
    if case == "inherited-private-empty":
        base_env[namespace["_HANDOFF_ENV"]] = ""
    elif case == "inherited-private-sealed":
        base_env[namespace["_HANDOFF_ENV"]] = _valid_runner_handoff(namespace, argv)
    elif case == "missing":
        base_env = {}
    elif case == "environment-conflict":
        base_env["DB_CLUSTER_ID"] = "different-cluster"
    elif case == "spark-conflict":
        namespace["_spark_cluster_id"] = lambda: spark_value("different-cluster")
    elif case == "spark-failure":

        def fail_spark() -> object:
            spark_calls.append("spark")
            raise RuntimeError("forced Spark failure")

        namespace["_spark_cluster_id"] = fail_spark
    elif case == "invalid-environment":
        base_env = {"DB_CLUSTER_ID": " invalid"}
    elif case == "invalid-spark":
        base_env = {}
        namespace["_spark_cluster_id"] = lambda: spark_value(True)
    elif case == "self-tamper":
        paths["runner_sha256"].write_bytes(
            paths["runner_sha256"].read_bytes() + b"\n# tampered\n"
        )
    subprocess_calls: list[list[str]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess_calls.append(list(command)),
    )

    with pytest.raises((ValueError, RuntimeError)):
        namespace["_bootstrap"](argv, base_env)
    assert subprocess_calls == []
    if case.startswith("inherited-private"):
        assert spark_calls == []


def test_v2_private_handoff_cross_codec_consumes_without_spark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, argv, _paths = _bootstrap_case(tmp_path)
    cluster_id = "Cluster-ID_Exact.Case"
    raw_handoff = _valid_runner_handoff(namespace, argv, cluster_id=cluster_id)
    private_name = databricks_v2._V2_BOOTSTRAP_HANDOFF_ENV
    for name in (*databricks_v2._V2_CLUSTER_ID_ENV_NAMES, private_name):
        monkeypatch.delenv(name, raising=False)
    _set_sanitized_child_environment(monkeypatch)
    monkeypatch.setenv("DATABRICKS_CLUSTER_ID", cluster_id)
    monkeypatch.setenv(private_name, raw_handoff)
    monkeypatch.setattr(sys, "argv", ["gpu-qualification-bootstrap-v2.py", *argv])
    observed: list[dict[str, Any]] = []

    def execute(**kwargs: Any) -> None:
        assert private_name not in os.environ
        observed.append(kwargs)

    def unexpected_spark() -> str:
        raise AssertionError("private v2 handoff must not call the v1 Spark resolver")

    monkeypatch.setattr(databricks_v2, "execute_gpu_qualification_job_v2", execute)
    monkeypatch.setattr(databricks_v1, "_cloud_cluster_id", unexpected_spark)

    with pytest.raises(SystemExit) as completed:
        exec(namespace["_CHILD_STUB"], {"__name__": "__main__"})
    assert completed.value.code == 0
    assert private_name not in os.environ
    assert len(observed) == 1
    assert observed[0]["cloud_cluster_id"] == cluster_id
    decoded_plan = databricks_v2._decode_plan_parameter(
        argv[argv.index("--plan-record-zlib-base64") + 1],
        expected_plan_sha256=argv[argv.index("--expected-plan-sha256") + 1],
    )
    assert observed[0]["artifact_sha256"] == (
        databricks_v2.pins_from_gpu_qualification_plan_v2(decoded_plan).to_record()
    )

    public_calls: list[str] = []

    def public_cluster_id() -> str:
        public_calls.append("public")
        return "public-cluster"

    monkeypatch.setattr(databricks_v1, "_cloud_cluster_id", public_cluster_id)
    assert databricks_v2.main(argv) == 0
    assert public_calls == ["public"]
    assert observed[-1]["cloud_cluster_id"] == "public-cluster"
    assert tuple(inspect.signature(databricks_v2.main).parameters) == ("argv",)
    assert "_main_from_bootstrap_handoff_v2" not in databricks_v2.__all__
    assert databricks_v1.GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256 == (
        "ca93baeda09f3df050b0dad3b8f3091c0f74235c426bd66555b67bd4b6eeafbc"
    )


@pytest.mark.parametrize("outcome", ["success", "malformed", "executor-failure"])
def test_v2_private_handoff_direct_environment_is_always_consumed(
    outcome: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, argv, _paths = _bootstrap_case(tmp_path)
    cluster_id = "Cluster-ID_Exact.Case"
    private_name = databricks_v2._V2_BOOTSTRAP_HANDOFF_ENV
    for name in (*databricks_v2._V2_CLUSTER_ID_ENV_NAMES, private_name):
        monkeypatch.delenv(name, raising=False)
    _set_sanitized_child_environment(monkeypatch)
    monkeypatch.setenv("DATABRICKS_CLUSTER_ID", cluster_id)
    monkeypatch.setenv(
        private_name,
        "{" if outcome == "malformed" else _valid_runner_handoff(namespace, argv),
    )
    execute_calls: list[dict[str, Any]] = []

    def execute(**kwargs: Any) -> None:
        assert private_name not in os.environ
        execute_calls.append(kwargs)
        if outcome == "executor-failure":
            raise RuntimeError("forced executor failure")

    monkeypatch.setattr(databricks_v2, "execute_gpu_qualification_job_v2", execute)
    monkeypatch.setattr(
        databricks_v1,
        "_cloud_cluster_id",
        lambda: (_ for _ in ()).throw(
            AssertionError("private v2 handoff must not call Spark")
        ),
    )

    if outcome == "success":
        assert databricks_v2._main_from_bootstrap_handoff_v2(argv=argv) == 0
        assert len(execute_calls) == 1
    else:
        with pytest.raises((ValueError, RuntimeError)):
            databricks_v2._main_from_bootstrap_handoff_v2(argv=argv)
        assert len(execute_calls) == (1 if outcome == "executor-failure" else 0)
    assert private_name not in os.environ


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "noncanonical",
        "duplicate-key",
        "extra-key-resealed",
        "bad-seal",
        "argv-binding-resealed",
        "runner-binding-resealed",
        "cluster-binding-resealed",
        "source-order-resealed",
        "source-duplicate-resealed",
        "source-unknown-resealed",
        "spark-type-resealed",
        "environment-added",
        "environment-conflict",
        "child-pythonpath",
        "child-safety-flag",
        "conflicting-inputs",
        "argv-missing",
        "argv-reordered-resealed",
    ],
)
def test_v2_private_handoff_rejects_strict_or_resealed_tamper(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, argv, _paths = _bootstrap_case(tmp_path)
    cluster_id = "Cluster-ID_Exact.Case"
    raw_handoff: object = _valid_runner_handoff(namespace, argv, cluster_id=cluster_id)
    record = json.loads(str(raw_handoff))
    candidate_argv = list(argv)
    private_name = databricks_v2._V2_BOOTSTRAP_HANDOFF_ENV
    for name in (*databricks_v2._V2_CLUSTER_ID_ENV_NAMES, private_name):
        monkeypatch.delenv(name, raising=False)
    _set_sanitized_child_environment(monkeypatch)
    monkeypatch.setenv("DATABRICKS_CLUSTER_ID", cluster_id)
    if mutation == "missing":
        raw_handoff = None
    elif mutation == "noncanonical":
        raw_handoff = str(raw_handoff) + "\n"
    elif mutation == "duplicate-key":
        field = f'"argv_sha256":"{record["argv_sha256"]}"'
        raw_handoff = str(raw_handoff).replace(field, f"{field},{field}", 1)
    elif mutation == "extra-key-resealed":
        record["extra"] = True
        raw_handoff = _reseal_handoff(record)
    elif mutation == "bad-seal":
        record["closed_record_sha256"] = "0" * 64
        raw_handoff = databricks_v2._canonical_bootstrap_json_v2(record)
    elif mutation == "argv-binding-resealed":
        record["argv_sha256"] = "0" * 64
        raw_handoff = _reseal_handoff(record)
    elif mutation == "runner-binding-resealed":
        record["runner_sha256"] = "0" * 64
        raw_handoff = _reseal_handoff(record)
    elif mutation == "cluster-binding-resealed":
        record["cluster_id"] = "different-cluster"
        raw_handoff = _reseal_handoff(record)
    elif mutation == "source-order-resealed":
        record["sources"] = list(reversed(record["sources"]))
        raw_handoff = _reseal_handoff(record)
    elif mutation == "source-duplicate-resealed":
        record["sources"].insert(0, "DATABRICKS_CLUSTER_ID")
        raw_handoff = _reseal_handoff(record)
    elif mutation == "source-unknown-resealed":
        record["sources"].insert(1, "UNKNOWN_CLUSTER_SOURCE")
        raw_handoff = _reseal_handoff(record)
    elif mutation == "spark-type-resealed":
        record["spark_checked"] = 1
        raw_handoff = _reseal_handoff(record)
    elif mutation == "environment-added":
        monkeypatch.setenv("DB_CLUSTER_ID", cluster_id)
    elif mutation == "environment-conflict":
        monkeypatch.setenv("DATABRICKS_CLUSTER_ID", "different-cluster")
    elif mutation == "child-pythonpath":
        monkeypatch.setenv("PYTHONPATH", "/tmp/untrusted")
    elif mutation == "child-safety-flag":
        monkeypatch.setenv("PYTHONNOUSERSITE", "0")
    elif mutation == "conflicting-inputs":
        monkeypatch.setenv(private_name, str(raw_handoff))
    elif mutation == "argv-missing":
        candidate_argv.pop()
    elif mutation == "argv-reordered-resealed":
        candidate_argv[:4] = candidate_argv[2:4] + candidate_argv[:2]
        record["argv_sha256"] = hashlib.sha256(
            databricks_v2._canonical_bootstrap_json_v2(candidate_argv).encode("utf-8")
        ).hexdigest()
        raw_handoff = _reseal_handoff(record)
    execute_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        databricks_v2,
        "execute_gpu_qualification_job_v2",
        lambda **kwargs: execute_calls.append(kwargs),
    )
    monkeypatch.setattr(
        databricks_v1,
        "_cloud_cluster_id",
        lambda: (_ for _ in ()).throw(
            AssertionError("private v2 handoff must not call Spark")
        ),
    )

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        databricks_v2._main_from_bootstrap_handoff_v2(
            raw_handoff,
            candidate_argv,
        )
    assert private_name not in os.environ
    assert execute_calls == []


def test_v2_bootstrap_writer_is_exclusive_and_transport_signature_is_narrow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "runner.py"
    previous_umask = os.umask(0o077)
    try:
        databricks_v2.write_gpu_qualification_bootstrap_runner_v2(path)
    finally:
        os.umask(previous_umask)
    assert path.read_text(encoding="utf-8") == (
        databricks_v2.GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SCRIPT
    )
    assert path.stat().st_mode & 0o777 == 0o750
    with pytest.raises(FileExistsError):
        databricks_v2.write_gpu_qualification_bootstrap_runner_v2(path)
    signature = inspect.signature(
        databricks_v2.render_gpu_qualification_submit_payloads_v2
    )
    assert "runner_uri" not in signature.parameters
    assert "package_wheel_uri" not in signature.parameters

    failed_path = tmp_path / "fchmod-failure.py"

    def fail_fchmod(_descriptor: int, _mode: int) -> None:
        raise OSError("forced fchmod failure")

    monkeypatch.setattr(databricks_v2.os, "fchmod", fail_fchmod)
    previous_umask = os.umask(0o077)
    try:
        with pytest.raises(OSError, match="forced fchmod failure"):
            databricks_v2.write_gpu_qualification_bootstrap_runner_v2(failed_path)
    finally:
        os.umask(previous_umask)
    assert not failed_path.exists()
    assert not failed_path.is_symlink()
    assert "patched_vllm_wheel_uri" not in signature.parameters


def test_v2_executor_publishes_valid_result_and_logs_rejected_measurements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = _plan()
    pins = _pins()
    job_id = "aws-g6-l4-auto-backend-diagnostic"
    attempt_id = databricks_v1.gpu_qualification_reservation_attempt_id(
        plan["closed_record_sha256"], job_id
    )
    output_uri = (
        "dbfs:/Volumes/catalog/schema/volume/output/"
        f"{plan['closed_record_sha256']}/{job_id}/gpu-job-result.json"
    )
    output_path = (
        tmp_path / plan["closed_record_sha256"] / job_id / "gpu-job-result.json"
    )
    work_dir = tmp_path / "work"
    artifact_paths = {key: tmp_path / key for key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS}

    monkeypatch.setattr(
        databricks_v2,
        "_validated_local_work_dir_v2",
        lambda *_args, **_kwargs: work_dir,
    )
    monkeypatch.setattr(
        databricks_v1,
        "_cluster_file_path",
        lambda value: (
            output_path
            if str(value) == output_uri
            else tmp_path / hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        ),
    )
    monkeypatch.setattr(
        databricks_v2,
        "_snapshot_artifacts_v2",
        lambda *_args, **_kwargs: artifact_paths,
    )
    monkeypatch.setattr(
        databricks_v1,
        "_observe_gpu_runtime",
        lambda *_args, **_kwargs: {
            "gpu": "NVIDIA L4",
            "gpu_compute_capability": "8.9",
            "nvidia_driver_version": "580.65.06",
            "torch_cuda_version": "12.9",
            "vllm_version": GPU_QUALIFICATION_VLLM_VERSION,
        },
    )
    timestamps = iter(
        [
            datetime_from_iso("2026-08-25T00:00:00Z"),
            datetime_from_iso("2026-08-25T00:00:01Z"),
        ]
    )

    diagnostic_measurements = {
        "backend_selection_mode": "auto",
        "observed_backend": "FLASHINFER",
        "publication_backend_changed": False,
        "trust_remote_code": False,
    }

    def runner(**kwargs: Any) -> dict[str, Any]:
        assert tuple(kwargs["artifact_paths"]) == GPU_QUALIFICATION_V2_ARTIFACT_KEYS
        return {
            "measurements": dict(diagnostic_measurements),
            "runtime_verification": build_gpu_runtime_verification_v2(
                plan_sha256=plan["closed_record_sha256"],
                job_id=job_id,
                artifact_sha256=pins.to_record(),
                attestation=_attestation(),
            ),
        }

    record = databricks_v2.execute_gpu_qualification_job_v2(
        plan_record=plan,
        expected_plan_sha256=plan["closed_record_sha256"],
        job_id=job_id,
        reservation_attempt_id=attempt_id,
        artifact_uris=_artifact_uris(),
        artifact_sha256=pins.to_record(),
        output_json=output_uri,
        work_dir=str(work_dir),
        cloud_run_id="123",
        cloud_cluster_id="cluster-1",
        sentinel_runner=runner,
        now=lambda: next(timestamps),
    )
    assert record["record_type"] == "cachet.vllm_0271_gpu_job_result.v2"
    assert json.loads(output_path.read_text(encoding="utf-8")) == record
    monkeypatch.setattr(
        databricks_v1,
        "download_databricks_volume_file_bytes",
        lambda *_args, **_kwargs: output_path.read_bytes(),
    )
    reread = databricks_v1._read_gpu_qualification_result(
        databricks_v1.DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
        output_uri,
        label="GPU v2 result",
        closed_record_convention="field_blank",
    )
    databricks_v2.validate_gpu_job_result_v2_record(
        reread,
        plan_record=plan,
        expected_artifact_pins=pins,
    )
    assert reread == record
    assert not work_dir.exists()

    failed_output_uri = (
        "dbfs:/Volumes/catalog/schema/volume/failed-output/"
        f"{plan['closed_record_sha256']}/{job_id}/gpu-job-result.json"
    )
    failed_output_path = (
        tmp_path
        / "failed-output"
        / plan["closed_record_sha256"]
        / job_id
        / "gpu-job-result.json"
    )
    monkeypatch.setattr(
        databricks_v1,
        "_cluster_file_path",
        lambda value: (
            failed_output_path
            if str(value) == failed_output_uri
            else tmp_path / hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        ),
    )
    failed_timestamps = iter(
        [
            datetime_from_iso("2026-08-25T00:00:02Z"),
            datetime_from_iso("2026-08-25T00:00:03Z"),
        ]
    )

    def reject_result(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("forced result validation failure")

    monkeypatch.setattr(
        databricks_v2,
        "validate_gpu_job_result_v2_record",
        reject_result,
    )
    capsys.readouterr()
    with pytest.raises(ValueError, match="forced result validation failure"):
        databricks_v2.execute_gpu_qualification_job_v2(
            plan_record=plan,
            expected_plan_sha256=plan["closed_record_sha256"],
            job_id=job_id,
            reservation_attempt_id=attempt_id,
            artifact_uris=_artifact_uris(),
            artifact_sha256=pins.to_record(),
            output_json=failed_output_uri,
            work_dir=str(work_dir),
            cloud_run_id="456",
            cloud_cluster_id="cluster-2",
            sentinel_runner=runner,
            now=lambda: next(failed_timestamps),
        )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        databricks_v2._V2_UNVALIDATED_MEASUREMENTS_LOG_PREFIX
        + canonical_gpu_qualification_json(diagnostic_measurements)
        + "\n"
    )
    assert not failed_output_path.exists()
    assert work_dir.is_dir()


def test_v2_runtime_lock_and_closure_parsers_accept_tracked_authority() -> None:
    lock_path = (
        Path(__file__).parents[1]
        / "src"
        / "document_kv_cache"
        / "runtime_locks"
        / VLLM_RUNTIME_BASE_LOCK_FILENAME
    )
    versions, hash_count = sentinels_v2._base_lock_projection(lock_path)
    assert len(versions) == VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT
    assert hash_count == VLLM_RUNTIME_BASE_LOCK_HASH_COUNT
    assert "flashinfer-python" not in versions
    closure_paths = tuple(
        (Path(__file__).parents[1] / "databricks-runs" / "_campaign-inputs").glob(
            "vllm-0.27.1-runtime-closure/sha256/*/"
            "vllm-0.27.1-flashinfer-0.6.16.post3-runtime-closure.json"
        )
    )
    assert len(closure_paths) == 1
    closure = sentinels_v2._read_exact_runtime_closure(closure_paths[0])
    assert closure["closed_record_sha256"] == (
        RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256
    )


class _DirectURLDistribution:
    def __init__(self, value: dict[str, Any]):
        self._value = value

    def read_text(self, name: str) -> str:
        assert name == "direct_url.json"
        return json.dumps(self._value)


def test_v2_direct_url_validation_rehashes_bytes_and_canonicalizes_uri(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "wheel+patched.whl"
    wheel.write_bytes(b"wheel")
    digest = hashlib.sha256(b"wheel").hexdigest()
    distribution = _DirectURLDistribution(
        {
            "archive_info": {"hashes": {"sha256": digest}},
            "url": wheel.resolve().as_uri(),
        }
    )
    assert (
        sentinels_v2._validate_direct_url(
            distribution,  # type: ignore[arg-type]
            expected_uri=wheel.resolve().as_uri(),
            expected_sha256=digest,
        )
        == wheel.resolve().as_uri()
    )
    wheel.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="source bytes"):
        sentinels_v2._validate_direct_url(
            distribution,  # type: ignore[arg-type]
            expected_uri=wheel.resolve().as_uri(),
            expected_sha256=digest,
        )


def datetime_from_iso(value: str) -> Any:
    from datetime import UTC, datetime

    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
