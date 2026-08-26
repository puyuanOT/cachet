import hashlib
import inspect
import json
from pathlib import Path
import subprocess
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
        "bb72afee845f85a9c4069931a98f1b4136be13edc88da4d44b994334612c85ea"
    )
    assert len(databricks_v2.GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SCRIPT) == 17524
    plan = _plan()
    assert plan["closed_record_sha256"] == (
        "48347f4329386d6f676450a047d41908cf7efec134af64fab964338f275d2de3"
    )
    assert len(canonical_gpu_qualification_json(plan).encode("utf-8")) == 15392
    payloads = _payloads()
    payload_bytes = canonical_gpu_qualification_json(
        {"payloads": list(payloads)}
    ).encode("utf-8")
    assert len(payload_bytes) == 121627
    assert hashlib.sha256(payload_bytes).hexdigest() == (
        "d97d2ff506c3cc185dff9402e652bc294c67e4be30375106209f52b2714be3ea"
    )


def test_v2_renderer_uses_only_eight_role_maps_with_safe_argument_headroom() -> None:
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
        assert len(parameters) == 50
        assert parameters.count("--artifact-uri") == 8
        assert parameters.count("--artifact-sha256") == 8
        assert "--runner-uri" not in parameters
        assert "--package-wheel-uri" not in parameters
        assert "--patched-vllm-wheel-uri" not in parameters
        assert (
            task["spark_python_task"]["python_file"]
            == (_artifact_uris()["runner_sha256"])
        )
    assert min(sizes) == 7603
    assert max(sizes) == 7675
    assert (
        max(sizes) < databricks_v2.GPU_QUALIFICATION_V2_DATABRICKS_PARAMETERS_MAX_BYTES
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
        artifact_pins=pins,
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


def test_v2_bootstrap_validates_transport_and_snapshots_package_before_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, argv, paths = _bootstrap_case(tmp_path)
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
        assert kwargs["env"]["PYTHONSAFEPATH"] == "1"
        calls.append(command)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    assert namespace["_bootstrap"](argv) == argv
    assert len(calls) == 1
    assert calls[0][1:4] == ["-P", "-m", "pip"]
    assert "--no-deps" in calls[0]
    assert ["--only-binary", ":all:"] == calls[0][
        calls[0].index("--only-binary") : calls[0].index("--only-binary") + 2
    ]
    assert not Path(calls[0][-1]).exists()

    paths["cachet_source_tree_sha256"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="cachet_source_tree_sha256 SHA-256 mismatch"):
        namespace["_bootstrap"](argv)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "unknown",
        "plan",
        "pin",
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
    tampered = list(argv)
    if mutation == "missing":
        tampered = tampered[:-1]
    elif mutation == "unknown":
        tampered[0] = "--unknown-plan"
    elif mutation == "plan":
        tampered[1] = tampered[1][:-1] + ("A" if tampered[1][-1] != "A" else "B")
    elif mutation == "pin":
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
        namespace["_bootstrap"](tampered)
    assert calls == []


def test_v2_bootstrap_writer_is_exclusive_and_transport_signature_is_narrow(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runner.py"
    databricks_v2.write_gpu_qualification_bootstrap_runner_v2(path)
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
    assert "patched_vllm_wheel_uri" not in signature.parameters


def test_v2_executor_requires_closed_sentinel_wrapper_and_publishes_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    def runner(**kwargs: Any) -> dict[str, Any]:
        assert tuple(kwargs["artifact_paths"]) == GPU_QUALIFICATION_V2_ARTIFACT_KEYS
        return {
            "measurements": {
                "backend_selection_mode": "auto",
                "observed_backend": "FLASHINFER",
                "publication_backend_changed": False,
                "trust_remote_code": False,
            },
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
    assert not work_dir.exists()


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
