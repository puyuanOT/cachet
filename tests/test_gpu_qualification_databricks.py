import base64
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import types
import zlib
from datetime import UTC, datetime
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import document_kv_cache.gpu_qualification_databricks as qualification_job
import document_kv_cache._gpu_qualification_sentinel_worker as sentinel_worker
import document_kv_cache.gpu_qualification_sentinels as qualification_sentinels
import document_kv_cache.databricks_resource_ledger as resource_ledger
from document_kv_cache._hardware_targets import SUPPORTED_V1_HARDWARE_TARGETS
from document_kv_cache.databricks_resource_ledger import (
    DatabricksClusterHourLedger,
    DatabricksLedgerPrefix,
    create_databricks_cluster_hour_ledger_json,
    databricks_cluster_hour_ledger_to_record,
    databricks_ledger_prefix,
    read_databricks_cluster_hour_ledger_json,
    record_databricks_run_terminal_actual_json,
    reserve_databricks_run_attempt_json,
)
from document_kv_cache.databricks_runs import DatabricksWorkspaceConfig
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_GENERATION_DATABRICKS_NODE_TYPE,
    GPU_QUALIFICATION_GENERATION_HARDWARE_ID,
    GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
    GPU_QUALIFICATION_VLLM_VERSION,
    GPUQualificationArtifactPins,
    GPUQualificationSelection,
    build_local_preflight_evidence,
    build_gpu_qualification_plan,
    canonical_gpu_qualification_json,
)
from document_kv_cache.gpu_qualification_databricks import (
    GPU_QUALIFICATION_ARTIFACT_KEYS,
    GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SCRIPT,
    GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256,
    GPU_QUALIFICATION_DATABRICKS_PARAMETERS_MAX_BYTES,
    GPU_QUALIFICATION_LOCAL_WORK_ROOT,
    GPU_QUALIFICATION_SUBMISSION_REJECTION_RECORD_TYPE,
    GPUQualificationLaunchAuthorization,
    collect_gpu_qualification_evidence,
    execute_gpu_qualification_job,
    gpu_qualification_reservation_attempt_id,
    render_gpu_qualification_submit_payloads,
    resume_gpu_qualification_job_submissions,
    require_gpu_qualification_launch_authorization,
    submit_gpu_qualification_jobs,
    validate_gpu_qualification_submission_rejection_record,
    write_gpu_qualification_bootstrap_runner,
)
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
    PUBLICATION_CAMPAIGN_ID,
    PUBLICATION_CAMPAIGN_LEDGER_ID,
    PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
    PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX,
    PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS,
)
from document_kv_cache.serving_env import (
    VLLM_PATCHED_WHEEL_SHA256_ENV,
    VLLM_PATCHED_WHEEL_URI_ENV,
    VLLM_RUNTIME_LOCK_DISTRIBUTION_COUNT,
    VLLM_RUNTIME_LOCK_SHA256,
)


CAMPAIGN_ID = PUBLICATION_CAMPAIGN_ID
CAMPAIGN_LEDGER_ID = PUBLICATION_CAMPAIGN_LEDGER_ID
CAMPAIGN_RECORD_SHA256 = PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256
SINGLE_USER_NAME = "publication@example.com"
_RETAINED_LEDGER_PATH = (
    Path(__file__).parents[1]
    / "databricks-runs"
    / "vllm-0271-publication-prep"
    / "cluster-hours.json"
)


@pytest.fixture(autouse=True)
def _stub_expensive_live_preflight_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep Databricks unit tests focused; publication_freeze tests the replay."""

    def validate(
        path: Path,
        *,
        plan: dict[str, Any],
        submit_payloads: tuple[dict[str, Any], ...],
        config: DatabricksWorkspaceConfig,
        require_fresh_workspace: bool,
    ) -> dict[str, Any]:
        del plan, submit_payloads, config, require_fresh_workspace
        return json.loads(path.read_text(encoding="utf-8"))

    monkeypatch.setattr(
        qualification_job,
        "_require_gpu_qualification_local_preflight_bundle",
        validate,
    )


class _FakeResponse:
    status = 200

    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()


class _SequentialOpener:
    def __init__(self, payloads: list[dict[str, Any]]):
        self._payloads = list(payloads)
        self.requests = []

    def __call__(self, request, *, timeout):
        self.requests.append(request)
        return _FakeResponse(self._payloads.pop(0))


def test_worker_uses_the_canonical_patched_wheel_environment_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    wheel = tmp_path / "patched-vllm.whl"
    wheel.write_bytes(b"reviewed-patched-wheel")

    class Distribution:
        @staticmethod
        def read_text(name: str) -> str | None:
            assert name == "direct_url.json"
            return json.dumps({"url": wheel.resolve().as_uri()})

    monkeypatch.setenv(VLLM_PATCHED_WHEEL_URI_ENV, str(wheel))
    monkeypatch.setenv(VLLM_PATCHED_WHEEL_SHA256_ENV, _digest(wheel.read_bytes()))
    monkeypatch.setattr(
        sentinel_worker.importlib.metadata,
        "distribution",
        lambda name: Distribution(),
    )

    assert sentinel_worker._direct_url_matches_patched_wheel() is True


def test_worker_requires_the_full_closed_runtime_lock_attestation(
    monkeypatch: pytest.MonkeyPatch,
):
    attestation = {
        "locked_distribution_count": VLLM_RUNTIME_LOCK_DISTRIBUTION_COUNT,
        "ok": True,
        "runtime_lock_sha256": VLLM_RUNTIME_LOCK_SHA256,
        "unexpected_distributions": [],
        "vllm_direct_url": "file:///local/patched-vllm.whl",
        "vllm_package_version": GPU_QUALIFICATION_VLLM_VERSION,
        "vllm_wheel_sha256": GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
    }
    monkeypatch.setenv(
        "CACHET_GPU_QUALIFICATION_RUNTIME_LOCK_ATTESTATION",
        json.dumps(attestation),
    )
    monkeypatch.setenv(
        VLLM_PATCHED_WHEEL_SHA256_ENV,
        GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
    )
    assert sentinel_worker._runtime_lock_attestation() == attestation

    partial = dict(attestation)
    partial.pop("unexpected_distributions")
    monkeypatch.setenv(
        "CACHET_GPU_QUALIFICATION_RUNTIME_LOCK_ATTESTATION",
        json.dumps(partial),
    )
    with pytest.raises(RuntimeError, match="open schema"):
        sentinel_worker._runtime_lock_attestation()


def _digest(value: str | bytes) -> str:
    payload = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _pins(*, runner_sha256: str = GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256):
    return GPUQualificationArtifactPins(
        runtime_lock_sha256=VLLM_RUNTIME_LOCK_SHA256,
        patched_vllm_wheel_sha256=GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        package_wheel_sha256=_digest("cachet-package-wheel"),
        cachet_source_tree_sha256=_digest("cachet-source-closure"),
        runner_sha256=runner_sha256,
        input_bundle_sha256=GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    )


def _plan(
    *,
    pins: GPUQualificationArtifactPins | None = None,
    campaign_ledger_prefix=None,
    campaign_ledger_path_sha256=PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
    campaign_opening_terminal_gpu_hours=(
        PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
    ),
) -> dict[str, Any]:
    return build_gpu_qualification_plan(
        campaign_id=CAMPAIGN_ID,
        campaign_record_sha256=CAMPAIGN_RECORD_SHA256,
        campaign_ledger_id=CAMPAIGN_LEDGER_ID,
        campaign_ledger_path_sha256=campaign_ledger_path_sha256,
        campaign_ledger_prefix=(
            campaign_ledger_prefix
            if campaign_ledger_prefix is not None
            else PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX
        ),
        campaign_opening_terminal_gpu_hours=campaign_opening_terminal_gpu_hours,
        artifact_pins=pins or _pins(),
    )


def _copy_retained_campaign_ledger(
    destination: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained = read_databricks_cluster_hour_ledger_json(_RETAINED_LEDGER_PATH)
    opening_prefix = PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX
    opening = replace(
        retained,
        reservations=retained.reservations[: opening_prefix.reservation_count],
        submission_receipts=retained.submission_receipts[
            : opening_prefix.submission_receipt_count
        ],
        terminal_actuals=retained.terminal_actuals[
            : opening_prefix.terminal_actual_count
        ],
    )
    destination.write_text(
        json.dumps(
            databricks_cluster_hour_ledger_to_record(opening),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        qualification_job,
        "databricks_ledger_path_sha256",
        lambda _path: PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
    )


def _artifact_uris(root: str = "dbfs:/cachet/qualification") -> dict[str, str]:
    return {
        key: f"{root}/{index}-{key}.artifact"
        for index, key in enumerate(GPU_QUALIFICATION_ARTIFACT_KEYS)
    }


def _publication_artifact_uris() -> dict[str, str]:
    pins = _pins().to_record()
    root = (
        "dbfs:/Volumes/datascience_qa/kv_cache_restaurant_cls/"
        "kv_cache_storage_benchmark/vllm_0271_publication_v1/inputs"
    )
    return {
        "cachet_source_tree_sha256": (
            f"{root}/cachet-source/{pins['cachet_source_tree_sha256']}/"
            "cachet-source-closure.json"
        ),
        "input_bundle_sha256": (
            f"{root}/main-latency/{pins['input_bundle_sha256']}"
        ),
        "package_wheel_sha256": (
            f"{root}/cachet-wheel/{pins['package_wheel_sha256']}/"
            "cachet_kv-0.2.0-py3-none-any.whl"
        ),
        "patched_vllm_wheel_sha256": (
            f"{root}/vllm-wheel/{pins['patched_vllm_wheel_sha256']}/"
            "vllm-0.27.1+cu129-1cachete5m265120c48a9352b9e-"
            "cp38-abi3-manylinux_2_28_x86_64.whl"
        ),
        "runner_sha256": (
            f"{root}/runner/{pins['runner_sha256']}/"
            "gpu-qualification-bootstrap.py"
        ),
        "runtime_lock_sha256": (
            f"{root}/runtime-lock/{pins['runtime_lock_sha256']}/"
            "vllm-0.27.1-cu129-py311-manylinux_2_35.lock"
        ),
    }


def _render(plan: dict[str, Any], uris: dict[str, str]):
    return render_gpu_qualification_submit_payloads(
        plan,
        single_user_name=SINGLE_USER_NAME,
        runner_uri=uris["runner_sha256"],
        package_wheel_uri=uris["package_wheel_sha256"],
        patched_vllm_wheel_uri=uris["patched_vllm_wheel_sha256"],
        artifact_uris=uris,
        output_root="dbfs:/cachet/qualification-results",
    )


def _submission_rejection_record(plan: dict[str, Any]) -> dict[str, Any]:
    plan_sha256 = plan["closed_record_sha256"]
    record: dict[str, Any] = {
        "attempt_ids": [
            gpu_qualification_reservation_attempt_id(plan_sha256, job["job_id"])
            for job in plan["cloud_qualification"]["jobs"]
        ],
        "batch_marker_file_sha256": _digest("batch-marker"),
        "closed_record_sha256": "",
        "failed_before_run_creation": True,
        "first_post_intent_file_sha256": _digest("first-post-intent"),
        "http_status": 400,
        "observed_parameters_json_bytes": 18_292,
        "plan_sha256": plan_sha256,
        "reconciled_actual_gpu_seconds_per_attempt": 0,
        "record_type": GPU_QUALIFICATION_SUBMISSION_REJECTION_RECORD_TYPE,
        "rejected_at_utc": "2026-08-25T03:50:53.503774Z",
        "remote_active_runs_observed": 0,
        "schema_version": 1,
        "server_parameters_json_limit_bytes": 10_000,
        "server_reason": "provided parameters exceeded limit",
        "submit_payloads_file_sha256": _digest("submit-payloads"),
    }
    unsigned = dict(record)
    unsigned.pop("closed_record_sha256")
    record["closed_record_sha256"] = _digest(
        canonical_gpu_qualification_json(unsigned)
    )
    return record


def _closed_result_bytes(**values: Any) -> bytes:
    record = {"closed_record_sha256": "", **values}
    qualification_job._seal_record(record)
    return (canonical_gpu_qualification_json(record) + "\n").encode("utf-8")


def test_controller_reads_volume_result_through_authenticated_package_transport(
    monkeypatch: pytest.MonkeyPatch,
):
    output_json = (
        "dbfs:/Volumes/catalog/schema/volume/qualification-results/plan/job/"
        "gpu-job-result.json"
    )
    observed: dict[str, Any] = {}

    def download(config, uri, *, max_bytes):
        observed.update(config=config, uri=uri, max_bytes=max_bytes)
        return _closed_result_bytes(value="remote")

    monkeypatch.setattr(
        qualification_job, "download_databricks_volume_file_bytes", download
    )
    monkeypatch.setattr(
        qualification_job,
        "_cluster_file_path",
        lambda _value: pytest.fail("dbfs:/Volumes result must not require local /dbfs"),
    )
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")

    result = qualification_job._read_gpu_qualification_result(
        config, output_json, label="GPU result"
    )

    assert result["value"] == "remote"
    assert observed == {
        "config": config,
        "uri": output_json,
        "max_bytes": qualification_job.DATABRICKS_VOLUME_FILE_MAX_DOWNLOAD_BYTES,
    }


def test_controller_result_reader_retains_safe_local_path_support(tmp_path: Path):
    result_path = tmp_path / "gpu-job-result.json"
    result_path.write_bytes(_closed_result_bytes(value="local"))

    result = qualification_job._read_gpu_qualification_result(
        DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
        str(result_path),
        label="GPU result",
    )

    assert result["value"] == "local"


@pytest.mark.parametrize("case", ("noncanonical", "tampered"))
def test_controller_rejects_noncanonical_or_tampered_remote_result(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
):
    if case == "noncanonical":
        content = b'{"closed_record_sha256":"' + b"0" * 64 + b'", "value":1}\n'
        error = "canonically encoded"
    else:
        record = json.loads(_closed_result_bytes(value="original"))
        record["value"] = "tampered"
        content = (canonical_gpu_qualification_json(record) + "\n").encode("utf-8")
        error = "closed_record_sha256 mismatch"
    monkeypatch.setattr(
        qualification_job,
        "download_databricks_volume_file_bytes",
        lambda *args, **kwargs: content,
    )

    with pytest.raises(ValueError, match=error):
        qualification_job._read_gpu_qualification_result(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            "dbfs:/Volumes/catalog/schema/volume/result.json",
            label="GPU result",
        )


def test_submission_rejection_record_uses_the_governed_closure_convention():
    plan = _plan()
    record = _submission_rejection_record(plan)
    validate_gpu_qualification_submission_rejection_record(record, plan_record=plan)

    blank_field_seal = dict(record)
    blank_field_seal["closed_record_sha256"] = ""
    blank_field_seal["closed_record_sha256"] = _digest(
        canonical_gpu_qualification_json(blank_field_seal)
    )
    with pytest.raises(ValueError, match="closed_record_sha256 mismatch"):
        validate_gpu_qualification_submission_rejection_record(
            blank_field_seal,
            plan_record=plan,
        )

    for field_name in (
        "schema_version",
        "remote_active_runs_observed",
        "reconciled_actual_gpu_seconds_per_attempt",
    ):
        mutated = dict(record)
        mutated[field_name] = False
        unsigned = dict(mutated)
        unsigned.pop("closed_record_sha256")
        mutated["closed_record_sha256"] = _digest(
            canonical_gpu_qualification_json(unsigned)
        )
        with pytest.raises(ValueError):
            validate_gpu_qualification_submission_rejection_record(
                mutated,
                plan_record=plan,
            )

    shortened = dict(record)
    shortened["attempt_ids"] = shortened["attempt_ids"][:-1]
    unsigned = dict(shortened)
    unsigned.pop("closed_record_sha256")
    shortened["closed_record_sha256"] = _digest(
        canonical_gpu_qualification_json(unsigned)
    )
    with pytest.raises(ValueError, match="attempt IDs differ"):
        validate_gpu_qualification_submission_rejection_record(
            shortened,
            plan_record=plan,
        )


def _write_local_preflight(
    plan: dict[str, Any],
    path: Path,
    *,
    completed_at_utc: str = "2026-08-23T00:00:00Z",
) -> Path:
    record = build_local_preflight_evidence(
        plan_sha256=plan["closed_record_sha256"],
        completed_at_utc=completed_at_utc,
        check_evidence_sha256={
            check_id: _digest(f"local-preflight:{check_id}")
            for check_id in plan["local_preflight"]["check_ids"]
        },
    )
    path.write_text(
        canonical_gpu_qualification_json(record) + "\n",
        encoding="utf-8",
    )
    return path


def _invalid_local_preflight_path(
    case: str,
    *,
    plan: dict[str, Any],
    path: Path,
) -> Path:
    if case == "missing":
        return path
    if case == "wrong-plan":
        wrong_plan = _plan(
            pins=replace(
                _pins(),
                package_wheel_sha256=_digest("wrong-plan-package-wheel"),
            )
        )
        assert wrong_plan["closed_record_sha256"] != plan["closed_record_sha256"]
        return _write_local_preflight(wrong_plan, path)
    if case == "future-completion":
        return _write_local_preflight(
            plan,
            path,
            completed_at_utc="2026-08-25T00:00:00Z",
        )
    if case == "tampered":
        _write_local_preflight(plan, path)
        record = json.loads(path.read_text(encoding="utf-8"))
        record["checks"][0]["evidence_sha256"] = _digest("tampered-check-output")
        path.write_text(
            canonical_gpu_qualification_json(record) + "\n",
            encoding="utf-8",
        )
        return path
    if case in {"failed-check", "wrong-check-id", "resealed-tampered"}:
        _write_local_preflight(plan, path)
        record = json.loads(path.read_text(encoding="utf-8"))
        if case == "failed-check":
            record["checks"][0]["status"] = "failed"
        elif case == "wrong-check-id":
            record["checks"][0]["check_id"] = "unfrozen-check"
        else:
            record["checks"][0]["evidence_sha256"] = _digest(
                "resealed-tampered-check-output"
            )
        record["closed_record_sha256"] = ""
        qualification_job._seal_record(record)
        path.write_text(
            canonical_gpu_qualification_json(record) + "\n",
            encoding="utf-8",
        )
        return path
    raise AssertionError(f"unhandled local preflight case: {case}")


def _option_values(parameters: list[str], option: str) -> list[str]:
    return [
        parameters[index + 1]
        for index, value in enumerate(parameters[:-1])
        if value == option
    ]


def test_renderer_emits_fourteen_unique_single_task_no_retry_payloads():
    plan = _plan()
    uris = _artifact_uris()

    payloads = _render(plan, uris)

    assert len(payloads) == 14
    assert len({payload["run_name"] for payload in payloads}) == 14
    output_paths = set()
    observed_job_ids = []
    for payload in payloads:
        assert len(payload["tasks"]) == 1
        task = payload["tasks"][0]
        assert task["max_retries"] == 0
        assert task["new_cluster"]["data_security_mode"] == "SINGLE_USER"
        assert task["new_cluster"]["single_user_name"] == SINGLE_USER_NAME
        assert task["spark_python_task"]["python_file"] == uris["runner_sha256"]
        parameters = task["spark_python_task"]["parameters"]
        assert _option_values(parameters, "--attempt-number") == ["0"]
        assert _option_values(parameters, "--retry-count") == ["0"]
        assert _option_values(parameters, "--expected-plan-sha256") == [
            plan["closed_record_sha256"]
        ]
        encoded_plans = _option_values(parameters, "--plan-record-zlib-base64")
        assert len(encoded_plans) == 1
        assert "--plan-record-json" not in parameters
        assert qualification_job._decode_qualification_plan_parameter(
            encoded_plans[0],
            expected_plan_sha256=plan["closed_record_sha256"],
        ) == plan
        assert (
            qualification_job._qualification_parameters_json_bytes(parameters)
            <= GPU_QUALIFICATION_DATABRICKS_PARAMETERS_MAX_BYTES
        )
        assert set(_option_values(parameters, "--artifact-uri")) == {
            f"{key}={value}" for key, value in uris.items()
        }
        assert set(_option_values(parameters, "--artifact-sha256")) == {
            f"{key}={value}" for key, value in _pins().to_record().items()
        }
        job_id = _option_values(parameters, "--job-id")[0]
        observed_job_ids.append(job_id)
        output = _option_values(parameters, "--output-json")[0]
        work_dir = _option_values(parameters, "--work-dir")[0]
        assert plan["closed_record_sha256"] in output
        assert job_id in output
        assert output.startswith("dbfs:/cachet/qualification-results/")
        assert work_dir == (
            f"{GPU_QUALIFICATION_LOCAL_WORK_ROOT}/"
            f"{plan['closed_record_sha256']}/{job_id}"
        )
        assert not work_dir.startswith(("dbfs:/", "file:"))
        output_paths.add(output)
    assert len(output_paths) == 14
    assert observed_job_ids == [
        job["job_id"] for job in plan["cloud_qualification"]["jobs"]
    ]


def test_renderer_repairs_the_legacy_none_shape_for_l4_a10g_and_l40s():
    plan = _plan()
    payloads = _render(plan, _artifact_uris())
    clusters_by_hardware: dict[str, dict[str, Any]] = {}
    for job, payload in zip(
        plan["cloud_qualification"]["jobs"], payloads, strict=True
    ):
        clusters_by_hardware.setdefault(
            job["hardware_id"], payload["tasks"][0]["new_cluster"]
        )

    assert {
        hardware_id: cluster["node_type_id"]
        for hardware_id, cluster in clusters_by_hardware.items()
    } == {
        "aws-g5-a10g": "g5.8xlarge",
        "aws-g6-l4": "g6.8xlarge",
        "aws-g6e-l40s": "g6e.4xlarge",
    }
    assert all(
        cluster["data_security_mode"] == "SINGLE_USER"
        and cluster["single_user_name"] == SINGLE_USER_NAME
        for cluster in clusters_by_hardware.values()
    )

    # This is the exact security fragment that produced the current
    # "No Unity token" launch failures: NONE and no principal binding.
    legacy_payloads = json.loads(json.dumps(payloads))
    for payload in legacy_payloads:
        cluster = payload["tasks"][0]["new_cluster"]
        cluster["data_security_mode"] = "NONE"
        cluster.pop("single_user_name")
    assert all(
        payload["tasks"][0]["new_cluster"]["data_security_mode"] == "NONE"
        and "single_user_name" not in payload["tasks"][0]["new_cluster"]
        for payload in legacy_payloads
    )
    with pytest.raises(ValueError, match="must use SINGLE_USER"):
        qualification_job._validated_qualification_payloads(plan, legacy_payloads)


@pytest.mark.parametrize("case", ("missing", "drift", "none"))
def test_structural_validator_rejects_missing_or_drifted_principal_and_none_mode(
    case: str,
):
    plan = _plan()
    payloads = json.loads(json.dumps(_render(plan, _artifact_uris())))
    cluster = payloads[1]["tasks"][0]["new_cluster"]
    if case == "missing":
        cluster.pop("single_user_name")
        error = "single_user_name"
    elif case == "drift":
        cluster["single_user_name"] = "other@example.com"
        error = "values drift"
    else:
        cluster["data_security_mode"] = "NONE"
        cluster.pop("single_user_name")
        error = "must use SINGLE_USER"
    with pytest.raises(ValueError, match=error):
        qualification_job._validated_qualification_payloads(plan, payloads)


def test_current_plan_production_parameters_stay_below_databricks_safety_cap():
    plan = _plan()
    uris = _publication_artifact_uris()
    payloads = render_gpu_qualification_submit_payloads(
        plan,
        single_user_name=SINGLE_USER_NAME,
        runner_uri=uris["runner_sha256"],
        package_wheel_uri=uris["package_wheel_sha256"],
        patched_vllm_wheel_uri=uris["patched_vllm_wheel_sha256"],
        artifact_uris=uris,
        output_root=(
            "dbfs:/Volumes/datascience_qa/kv_cache_restaurant_cls/"
            "kv_cache_storage_benchmark/vllm_0271_publication_v1/"
            "qualification-results"
        ),
    )

    sizes = [
        qualification_job._qualification_parameters_json_bytes(
            payload["tasks"][0]["spark_python_task"]["parameters"]
        )
        for payload in payloads
    ]
    assert len(payloads) == 14
    assert max(sizes) <= GPU_QUALIFICATION_DATABRICKS_PARAMETERS_MAX_BYTES == 9_500
    assert min(sizes) > 0


def test_parameters_cap_counts_unicode_as_databricks_json_escapes():
    plan = _plan()
    uris = _publication_artifact_uris()
    uris["cachet_source_tree_sha256"] += "/" + ("😀" * 200)

    with pytest.raises(ValueError, match="9500-byte safety cap"):
        render_gpu_qualification_submit_payloads(
            plan,
            single_user_name=SINGLE_USER_NAME,
            runner_uri=uris["runner_sha256"],
            package_wheel_uri=uris["package_wheel_sha256"],
            patched_vllm_wheel_uri=uris["patched_vllm_wheel_sha256"],
            artifact_uris=uris,
            output_root="dbfs:/cachet/qualification-results",
        )


@pytest.mark.parametrize(
    ("encoded", "error"),
    [
        ("not+strict_base64!", "strict base64url"),
        (
            base64.urlsafe_b64encode(b"not a zlib stream").decode("ascii"),
            "valid zlib stream",
        ),
        (
            base64.urlsafe_b64encode(
                zlib.compress(b'{"not": "canonical"}', level=9)
            ).decode("ascii"),
            "not canonical JSON",
        ),
        (
            base64.urlsafe_b64encode(
                zlib.compress(
                    b"x"
                    * (qualification_job._QUALIFICATION_PLAN_MAX_CANONICAL_BYTES + 1),
                    level=9,
                )
            ).decode("ascii"),
            "exceeds the canonical size cap",
        ),
        (
            base64.urlsafe_b64encode(zlib.compress(b"{}", level=9)[:-1]).decode(
                "ascii"
            ),
            "invalid zlib closure",
        ),
        (
            base64.urlsafe_b64encode(
                zlib.compress(b"{}", level=9) + b"trailing-bytes"
            ).decode("ascii"),
            "invalid zlib closure",
        ),
        (
            base64.urlsafe_b64encode(
                zlib.compress(b"{}", level=9) + zlib.compress(b"{}", level=9)
            ).decode("ascii"),
            "invalid zlib closure",
        ),
    ],
)
def test_worker_rejects_corrupt_or_oversize_encoded_plan(
    encoded: str,
    error: str,
):
    with pytest.raises(ValueError, match=error):
        qualification_job._decode_qualification_plan_parameter(
            encoded,
            expected_plan_sha256="0" * 64,
        )


def test_worker_decodes_only_the_exact_canonical_plan_and_sha():
    plan = _plan()
    canonical = canonical_gpu_qualification_json(plan)
    encoded = qualification_job._encode_qualification_plan_parameter(canonical)

    assert qualification_job._decode_qualification_plan_parameter(
        encoded,
        expected_plan_sha256=plan["closed_record_sha256"],
    ) == plan
    alternate_encoding = base64.urlsafe_b64encode(
        zlib.compress(canonical.encode("utf-8"), level=1)
    ).decode("ascii")
    assert alternate_encoding != encoded
    assert qualification_job._decode_qualification_plan_parameter(
        alternate_encoding,
        expected_plan_sha256=plan["closed_record_sha256"],
    ) == plan
    with pytest.raises(ValueError, match="SHA-256 differs"):
        qualification_job._decode_qualification_plan_parameter(
            encoded,
            expected_plan_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="transport size cap"):
        qualification_job._decode_qualification_plan_parameter(
            "A" * (GPU_QUALIFICATION_DATABRICKS_PARAMETERS_MAX_BYTES + 1),
            expected_plan_sha256=plan["closed_record_sha256"],
        )
    tampered = json.loads(canonical)
    tampered["unsupported_methods"][0]["reason"] = "tampered"
    tampered_encoded = qualification_job._encode_qualification_plan_parameter(
        canonical_gpu_qualification_json(tampered)
    )
    with pytest.raises(ValueError, match="closed_record_sha256"):
        qualification_job._decode_qualification_plan_parameter(
            tampered_encoded,
            expected_plan_sha256=plan["closed_record_sha256"],
        )


def test_worker_cli_decodes_compact_plan_before_execution(
    monkeypatch: pytest.MonkeyPatch,
):
    plan = _plan()
    parameters = _render(plan, _artifact_uris())[0]["tasks"][0][
        "spark_python_task"
    ]["parameters"]
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        qualification_job,
        "execute_gpu_qualification_job",
        lambda **kwargs: observed.update(kwargs),
    )
    monkeypatch.setattr(
        qualification_job,
        "_cloud_cluster_id",
        lambda: "cluster-compact-plan",
    )

    assert qualification_job.main(parameters) == 0
    assert observed["plan_record"] == plan
    assert observed["expected_plan_sha256"] == plan["closed_record_sha256"]

    corrupt_parameters = list(parameters)
    encoded_index = corrupt_parameters.index("--plan-record-zlib-base64") + 1
    corrupt_parameters[encoded_index] = "not+strict_base64!"
    observed.clear()
    with pytest.raises(ValueError, match="strict base64url"):
        qualification_job.main(corrupt_parameters)
    assert observed == {}


def _clear_cluster_identity_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DATABRICKS_CLUSTER_ID", "DB_CLUSTER_ID"):
        monkeypatch.delenv(name, raising=False)


def test_worker_cluster_id_accepts_exact_environment_identity(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_cluster_identity_environment(monkeypatch)
    monkeypatch.setenv("DATABRICKS_CLUSTER_ID", "Cluster-ID_Exact.Case")
    monkeypatch.setattr(qualification_job, "_spark_cloud_cluster_id", lambda: None)

    assert qualification_job._cloud_cluster_id() == "Cluster-ID_Exact.Case"


def test_worker_cluster_id_accepts_agreeing_environment_and_spark_sources(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_cluster_identity_environment(monkeypatch)
    for name in ("DATABRICKS_CLUSTER_ID", "DB_CLUSTER_ID"):
        monkeypatch.setenv(name, "0825-121314-AbCd1234")
    monkeypatch.setattr(
        qualification_job,
        "_spark_cloud_cluster_id",
        lambda: "0825-121314-AbCd1234",
    )

    assert qualification_job._cloud_cluster_id() == "0825-121314-AbCd1234"


def test_worker_cluster_id_uses_driver_spark_conf_when_env_is_absent(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_cluster_identity_environment(monkeypatch)
    calls: list[str] = []

    class SparkConf:
        def get(self, key: str, default: object) -> str:
            calls.append("SparkConf.get")
            assert key == "spark.databricks.clusterUsageTags.clusterId"
            assert default is None
            return "0825-121314-spark123"

    class DriverSparkContext:
        def getConf(self) -> SparkConf:
            calls.append("SparkContext.getConf")
            return SparkConf()

    class SparkContext:
        @classmethod
        def getOrCreate(cls) -> DriverSparkContext:
            calls.append("SparkContext.getOrCreate")
            return DriverSparkContext()

    fake_pyspark = types.ModuleType("pyspark")
    fake_pyspark.SparkContext = SparkContext  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyspark", fake_pyspark)
    monkeypatch.delitem(sys.modules, "pyspark.sql", raising=False)

    assert qualification_job._cloud_cluster_id() == "0825-121314-spark123"
    assert calls == [
        "SparkContext.getOrCreate",
        "SparkContext.getConf",
        "SparkConf.get",
    ]


def test_worker_cluster_id_accepts_environment_when_spark_key_is_cleanly_absent(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_cluster_identity_environment(monkeypatch)
    monkeypatch.setenv("DB_CLUSTER_ID", "0825-121314-env12345")

    class SparkConf:
        def get(self, key: str, default: object) -> object:
            assert key == "spark.databricks.clusterUsageTags.clusterId"
            return default

    monkeypatch.setattr(qualification_job, "_active_spark_conf", SparkConf)

    assert qualification_job._cloud_cluster_id() == "0825-121314-env12345"


@pytest.mark.parametrize("value", [True, 123, b"cluster-id"])
def test_worker_cluster_id_rejects_non_string_spark_values(
    monkeypatch: pytest.MonkeyPatch,
    value: object,
):
    _clear_cluster_identity_environment(monkeypatch)
    monkeypatch.setattr(
        qualification_job,
        "_spark_cloud_cluster_id",
        lambda: value,
    )

    with pytest.raises(ValueError, match="is not a string"):
        qualification_job._cloud_cluster_id()


@pytest.mark.parametrize(
    "value",
    ["", " cluster-id", "cluster-id ", "x" * 257],
)
def test_worker_cluster_id_rejects_noncanonical_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
):
    _clear_cluster_identity_environment(monkeypatch)
    monkeypatch.setenv("DB_CLUSTER_ID", value)
    monkeypatch.setattr(qualification_job, "_spark_cloud_cluster_id", lambda: None)

    with pytest.raises(ValueError, match="is not canonical"):
        qualification_job._cloud_cluster_id()


def test_worker_cluster_id_rejects_control_characters_from_spark(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_cluster_identity_environment(monkeypatch)
    monkeypatch.setattr(
        qualification_job,
        "_spark_cloud_cluster_id",
        lambda: "a\x00b",
    )

    with pytest.raises(ValueError, match="is not canonical"):
        qualification_job._cloud_cluster_id()


def test_worker_cluster_id_rejects_environment_ambiguity_without_precedence(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_cluster_identity_environment(monkeypatch)
    monkeypatch.setenv("DATABRICKS_CLUSTER_ID", "cluster-a")
    monkeypatch.setenv("DB_CLUSTER_ID", "cluster-b")
    monkeypatch.setattr(
        qualification_job,
        "_spark_cloud_cluster_id",
        lambda: "cluster-a",
    )

    with pytest.raises(RuntimeError, match="sources are ambiguous"):
        qualification_job._cloud_cluster_id()


def test_worker_cluster_id_rejects_spark_environment_conflict(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_cluster_identity_environment(monkeypatch)
    monkeypatch.setenv("DATABRICKS_CLUSTER_ID", "cluster-env")
    monkeypatch.setattr(
        qualification_job,
        "_spark_cloud_cluster_id",
        lambda: "cluster-spark",
    )

    with pytest.raises(RuntimeError, match="sources are ambiguous"):
        qualification_job._cloud_cluster_id()


def test_worker_cluster_id_does_not_downgrade_spark_access_failure_to_env(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_cluster_identity_environment(monkeypatch)
    monkeypatch.setenv("DATABRICKS_CLUSTER_ID", "cluster-env")

    def fail() -> None:
        raise RuntimeError("Spark access failed")

    monkeypatch.setattr(qualification_job, "_spark_cloud_cluster_id", fail)
    with pytest.raises(RuntimeError, match="Spark access failed"):
        qualification_job._cloud_cluster_id()


def test_worker_cluster_id_rejects_missing_sources(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_cluster_identity_environment(monkeypatch)
    monkeypatch.setattr(qualification_job, "_spark_cloud_cluster_id", lambda: None)

    with pytest.raises(RuntimeError, match="unavailable at runtime"):
        qualification_job._cloud_cluster_id()


def test_worker_cluster_id_rejects_spark_conf_exception(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_cluster_identity_environment(monkeypatch)

    class SparkConf:
        def get(self, _key: str, _default: object) -> None:
            raise RuntimeError("Py4J failed")

    monkeypatch.setattr(qualification_job, "_active_spark_conf", SparkConf)
    with pytest.raises(RuntimeError, match="Spark runtime lookup failed"):
        qualification_job._cloud_cluster_id()


def test_worker_cluster_id_rejects_missing_pyspark_runtime(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "pyspark", None)

    with pytest.raises(RuntimeError, match="Spark runtime is unavailable"):
        qualification_job._active_spark_conf()


def test_worker_cluster_id_remains_exactly_bound_to_control_plane_identity():
    contract = {
        "job_id": "job-a",
        "output_json": "dbfs:/Volumes/cat/schema/volume/result.json",
        "reservation_attempt_id": "attempt-a",
        "task_key": "task_a",
    }
    receipt = {"cloud_run_id": "123"}
    run_identity = {"cloud_cluster_id": "Cluster-Exact"}
    result = {
        "cloud_cluster_id": "Cluster-Exact",
        "cloud_run_id": "123",
        "job_id": "job-a",
        "output_json": contract["output_json"],
        "reservation_attempt_id": "attempt-a",
        "task_key": "task_a",
    }
    qualification_job._validate_result_submission_binding(
        result,
        contract=contract,
        submit_receipt=receipt,
        run_identity=run_identity,
    )

    result["cloud_cluster_id"] = "cluster-exact"
    with pytest.raises(ValueError, match="cloud_cluster_id differs"):
        qualification_job._validate_result_submission_binding(
            result,
            contract=contract,
            submit_receipt=receipt,
            run_identity=run_identity,
        )


def test_renderer_maps_only_qualification_generation_job_to_g6e_l40s():
    plan = _plan()
    payloads = _render(plan, _artifact_uris())

    by_job_id = {
        _option_values(
            payload["tasks"][0]["spark_python_task"]["parameters"], "--job-id"
        )[0]: payload
        for payload in payloads
    }
    l40s = by_job_id["aws-g6e-l40s-generation-throughput"]["tasks"][0]["new_cluster"]
    assert l40s["node_type_id"] == GPU_QUALIFICATION_GENERATION_DATABRICKS_NODE_TYPE
    assert l40s["driver_node_type_id"] == (
        GPU_QUALIFICATION_GENERATION_DATABRICKS_NODE_TYPE
    )
    assert GPU_QUALIFICATION_GENERATION_HARDWARE_ID not in SUPPORTED_V1_HARDWARE_TARGETS
    assert {
        payload["tasks"][0]["new_cluster"]["node_type_id"]
        for job_id, payload in by_job_id.items()
        if job_id != "aws-g6e-l40s-generation-throughput"
    } == {"g5.8xlarge", "g6.8xlarge"}


def test_renderer_rejects_conflated_source_and_package_uri_roles():
    uris = _artifact_uris()
    uris["cachet_source_tree_sha256"] = uris["package_wheel_sha256"]

    with pytest.raises(ValueError, match="distinct.*conflated"):
        _render(_plan(), uris)


def test_renderer_rejects_an_unreviewed_bootstrap_runner_pin():
    pins = _pins(runner_sha256=_digest("different-runner"))

    with pytest.raises(ValueError, match="reviewed bootstrap runner"):
        _render(_plan(pins=pins), _artifact_uris())


def test_renderer_rejects_any_nonpublication_input_bundle():
    pins = replace(_pins(), input_bundle_sha256=_digest("other-input-bundle"))

    with pytest.raises(ValueError, match="frozen 7ff6 publication input bundle"):
        _render(_plan(pins=pins), _artifact_uris())


def test_submitter_receipt_binds_the_exact_fourteen_reserved_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan = _plan()
    payloads = _render(plan, _artifact_uris())
    ledger_path = tmp_path / "cluster-hours.json"
    receipt_root = tmp_path / "submit-receipts"
    _copy_retained_campaign_ledger(ledger_path, monkeypatch)
    plan = _plan()
    payloads = _render(plan, _artifact_uris())
    opener = _SequentialOpener(
        [{"run_id": 10_000 + index} for index in range(len(payloads))]
    )
    local_preflight_path = _write_local_preflight(
        plan, tmp_path / "local-preflight.json"
    )

    receipts = submit_gpu_qualification_jobs(
        DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
        plan_record=plan,
        submit_payloads=payloads,
        ledger_path=ledger_path,
        submit_receipt_root=receipt_root,
        local_preflight_evidence_path=local_preflight_path,
        opener=opener,
        now=lambda: datetime(2026, 8, 24, 0, 0, tzinfo=UTC),
    )

    assert len(receipts) == len(payloads) == 14
    assert len(opener.requests) == 14
    assert [json.loads(request.data) for request in opener.requests] == list(payloads)
    assert {path.name for path in receipt_root.iterdir()} == {
        "batch-reserved.json",
        "phase-lease.json",
        *(f"{job['job_id']}.json" for job in plan["cloud_qualification"]["jobs"]),
    }
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    assert len(ledger.reservations) == (
        PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX.reservation_count + 14
    )
    assert len(ledger.submission_receipts) == (
        PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX.submission_receipt_count + 14
    )
    assert len(ledger.terminal_actuals) == (
        PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX.terminal_actual_count
    )
    assert all(
        receipt["authorization_scope"]
        == "submission_identity_only_requires_direct_terminal_collection"
        for receipt in receipts
    )
    assert [receipt["cloud_run_id"] for receipt in receipts] == [
        str(10_000 + index) for index in range(14)
    ]
    resumed = resume_gpu_qualification_job_submissions(
        DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
        plan_record=plan,
        submit_payloads=payloads,
        ledger_path=ledger_path,
        submit_receipt_root=receipt_root,
        local_preflight_evidence_path=local_preflight_path,
        opener=lambda *_args, **_kwargs: pytest.fail("completed phase must not POST"),
    )
    assert [item["cloud_run_id"] for item in resumed] == [
        str(10_000 + index) for index in range(14)
    ]


def test_submitter_rejects_incomplete_or_mutated_job_closure_before_post(
    tmp_path: Path,
):
    plan = _plan()
    payloads = list(_render(plan, _artifact_uris()))
    ledger_path = tmp_path / "cluster-hours.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="gpu-qualification",
    )
    opener = _SequentialOpener([{"run_id": 1}])
    local_preflight_path = _write_local_preflight(
        plan, tmp_path / "local-preflight.json"
    )

    with pytest.raises(ValueError, match="exact planned job closure"):
        submit_gpu_qualification_jobs(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            plan_record=plan,
            submit_payloads=payloads[:-1],
            ledger_path=ledger_path,
            submit_receipt_root=tmp_path / "missing-receipts",
            local_preflight_evidence_path=local_preflight_path,
            opener=opener,
        )
    mutated = json.loads(json.dumps(payloads))
    mutated[0]["tasks"][0]["max_retries"] = 1
    with pytest.raises(ValueError, match="retry/timeout identity"):
        submit_gpu_qualification_jobs(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            plan_record=plan,
            submit_payloads=mutated,
            ledger_path=ledger_path,
            submit_receipt_root=tmp_path / "mutated-receipts",
            local_preflight_evidence_path=local_preflight_path,
            opener=opener,
        )
    assert opener.requests == []
    assert read_databricks_cluster_hour_ledger_json(ledger_path).reservations == ()


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("raw-plan", "parameters differ from the renderer"),
        ("raw-plan-oversize", "9500-byte safety cap"),
        ("oversize", "9500-byte safety cap"),
        ("corrupt-encoded-plan", "parameters differ from the renderer"),
        ("alternate-valid-encoding", "parameters differ from the renderer"),
    ],
)
def test_invalid_plan_transport_rejects_before_ledger_or_post(
    case: str,
    error: str,
    tmp_path: Path,
):
    plan = _plan()
    payloads = list(_render(plan, _artifact_uris()))
    mutated = json.loads(json.dumps(payloads))
    parameters = mutated[0]["tasks"][0]["spark_python_task"]["parameters"]
    plan_option_index = parameters.index("--plan-record-zlib-base64")
    if case == "raw-plan":
        parameters[plan_option_index] = "--plan-record-json"
        parameters[plan_option_index + 1] = "{}"
    elif case == "raw-plan-oversize":
        parameters[plan_option_index] = "--plan-record-json"
        parameters[plan_option_index + 1] = canonical_gpu_qualification_json(plan)
    elif case == "oversize":
        parameters.extend(("--unexpected", "x" * 10_000))
    else:
        encoded = parameters[plan_option_index + 1]
        if case == "alternate-valid-encoding":
            parameters[plan_option_index + 1] = base64.urlsafe_b64encode(
                zlib.compress(
                    canonical_gpu_qualification_json(plan).encode("utf-8"),
                    level=1,
                )
            ).decode("ascii")
        else:
            parameters[plan_option_index + 1] = (
                ("A" if encoded[0] != "A" else "B") + encoded[1:]
            )
    if case in {
        "raw-plan",
        "corrupt-encoded-plan",
        "alternate-valid-encoding",
    }:
        mutated_payload = dict(mutated[0])
        mutated_payload.pop("idempotency_token")
        mutated[0] = qualification_job.bind_databricks_run_idempotency_token(
            mutated_payload,
            attempt_id=gpu_qualification_reservation_attempt_id(
                plan["closed_record_sha256"],
                plan["cloud_qualification"]["jobs"][0]["job_id"],
            ),
        )
    ledger_path = tmp_path / "must-not-be-created-ledger.json"
    receipt_root = tmp_path / "must-not-be-created-receipts"
    opener = _SequentialOpener([{"run_id": 1}])

    with pytest.raises(ValueError, match=error):
        submit_gpu_qualification_jobs(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            plan_record=plan,
            submit_payloads=mutated,
            ledger_path=ledger_path,
            submit_receipt_root=receipt_root,
            local_preflight_evidence_path=tmp_path / "unused-preflight.json",
            opener=opener,
        )

    assert not ledger_path.exists()
    assert not receipt_root.exists()
    assert opener.requests == []


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("missing", "must be one regular file"),
        ("tampered", "closed_record_sha256 is invalid"),
        ("wrong-plan", "plan_sha256 mismatch"),
        ("future-completion", "must complete before qualification submission"),
        ("failed-check", "did not pass in canonical order"),
        ("wrong-check-id", "did not pass in canonical order"),
    ],
)
def test_submitter_rejects_invalid_local_preflight_without_ledger_or_post_side_effects(
    case: str,
    error: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ledger_path = tmp_path / "cluster-hours.json"
    _copy_retained_campaign_ledger(ledger_path, monkeypatch)
    plan = _plan()
    payloads = _render(plan, _artifact_uris())
    local_preflight_path = _invalid_local_preflight_path(
        case,
        plan=plan,
        path=tmp_path / "local-preflight.json",
    )
    receipt_root = tmp_path / "submit-receipts"
    opener = _SequentialOpener([{"run_id": 1}])
    ledger_before = ledger_path.read_bytes()

    with pytest.raises(ValueError, match=error):
        submit_gpu_qualification_jobs(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            plan_record=plan,
            submit_payloads=payloads,
            ledger_path=ledger_path,
            submit_receipt_root=receipt_root,
            local_preflight_evidence_path=local_preflight_path,
            opener=opener,
            now=lambda: datetime(2026, 8, 24, 0, 0, tzinfo=UTC),
        )

    assert ledger_path.read_bytes() == ledger_before
    assert len(read_databricks_cluster_hour_ledger_json(ledger_path).reservations) == (
        PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX.reservation_count
    )
    assert opener.requests == []
    assert not receipt_root.exists()


def test_all_authority_boundaries_require_live_preflight_bundle_before_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ledger_path = tmp_path / "cluster-hours.json"
    _copy_retained_campaign_ledger(ledger_path, monkeypatch)
    ledger_before = ledger_path.read_bytes()
    plan = _plan()
    payloads = _render(plan, _artifact_uris())
    preflight = _write_local_preflight(plan, tmp_path / "local-preflight.json")
    opener = _SequentialOpener([{"run_id": 1}])

    def reject_live_bundle(
        _path: Path,
        *,
        plan: dict[str, Any],
        submit_payloads: tuple[dict[str, Any], ...],
        config: DatabricksWorkspaceConfig,
        require_fresh_workspace: bool,
    ) -> dict[str, Any]:
        del plan, config, require_fresh_workspace
        assert submit_payloads is not payloads
        assert tuple(submit_payloads) == payloads
        raise RuntimeError("live seven-check replay rejected")

    monkeypatch.setattr(
        qualification_job,
        "_require_gpu_qualification_local_preflight_bundle",
        reject_live_bundle,
    )
    operations = (
        lambda: submit_gpu_qualification_jobs(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            plan_record=plan,
            submit_payloads=payloads,
            ledger_path=ledger_path,
            submit_receipt_root=tmp_path / "submit-root",
            local_preflight_evidence_path=preflight,
            opener=opener,
        ),
        lambda: resume_gpu_qualification_job_submissions(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            plan_record=plan,
            submit_payloads=payloads,
            ledger_path=ledger_path,
            submit_receipt_root=tmp_path / "resume-root",
            local_preflight_evidence_path=preflight,
            opener=opener,
        ),
        lambda: collect_gpu_qualification_evidence(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            plan_record=plan,
            submit_payloads=payloads,
            ledger_path=ledger_path,
            submit_receipt_root=tmp_path / "collect-submit-root",
            local_preflight_evidence_path=preflight,
            terminal_receipt_root=tmp_path / "terminal-root",
            evidence_output_json=tmp_path / "qualification-evidence.json",
        ),
        lambda: qualification_job.replay_gpu_qualification_launch_authorization(
            config=DatabricksWorkspaceConfig(
                "https://dbc.example", "secret-token"
            ),
            plan_record=plan,
            submit_payloads=payloads,
            ledger_path=ledger_path,
            submit_receipt_root=tmp_path / "replay-submit-root",
            local_preflight_evidence_path=preflight,
            terminal_receipt_root=tmp_path / "replay-terminal-root",
            evidence_path=tmp_path / "replay-evidence.json",
            expected_campaign_id=CAMPAIGN_ID,
            expected_artifact_pins=_pins(),
        ),
    )
    for operation in operations:
        with pytest.raises(RuntimeError, match="live seven-check replay rejected"):
            operation()

    assert ledger_path.read_bytes() == ledger_before
    assert opener.requests == []
    assert not (tmp_path / "submit-root").exists()
    assert not (tmp_path / "resume-root").exists()
    assert not (tmp_path / "terminal-root").exists()
    assert not (tmp_path / "replay-terminal-root").exists()


def test_submitter_preexisting_phase_lease_leaves_zero_reservations_and_posts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan = _plan()
    payloads = _render(plan, _artifact_uris())
    ledger_path = tmp_path / "cluster-hours.json"
    _copy_retained_campaign_ledger(ledger_path, monkeypatch)
    plan = _plan()
    payloads = _render(plan, _artifact_uris())
    receipt_root = tmp_path / "submit-receipts"
    receipt_root.mkdir()
    opener = _SequentialOpener([{"run_id": 1}])
    local_preflight_path = _write_local_preflight(
        plan, tmp_path / "local-preflight.json"
    )

    with pytest.raises(FileExistsError, match="already exists"):
        submit_gpu_qualification_jobs(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            plan_record=plan,
            submit_payloads=payloads,
            ledger_path=ledger_path,
            submit_receipt_root=receipt_root,
            local_preflight_evidence_path=local_preflight_path,
            opener=opener,
        )
    assert opener.requests == []
    assert len(read_databricks_cluster_hour_ledger_json(ledger_path).reservations) == (
        PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX.reservation_count
    )


def test_resume_recovers_post_reservation_pre_marker_controller_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ledger_path = tmp_path / "cluster-hours.json"
    receipt_root = tmp_path / "submit-receipts"
    _copy_retained_campaign_ledger(ledger_path, monkeypatch)
    plan = _plan()
    payloads = _render(plan, _artifact_uris())
    local_preflight_path = _write_local_preflight(
        plan, tmp_path / "local-preflight.json"
    )
    original_writer = qualification_job._write_canonical_exclusive

    def crash_before_marker(record, path):
        if path.name == "batch-reserved.json":
            raise RuntimeError("simulated controller crash before batch marker")
        original_writer(record, path)

    monkeypatch.setattr(
        qualification_job, "_write_canonical_exclusive", crash_before_marker
    )
    with pytest.raises(RuntimeError, match="simulated controller crash"):
        submit_gpu_qualification_jobs(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            plan_record=plan,
            submit_payloads=payloads,
            ledger_path=ledger_path,
            submit_receipt_root=receipt_root,
            local_preflight_evidence_path=local_preflight_path,
            opener=lambda *_args, **_kwargs: pytest.fail("crash occurs before POST"),
        )
    crashed = read_databricks_cluster_hour_ledger_json(ledger_path)
    assert len(crashed.reservations) == (
        PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX.reservation_count + 14
    )
    assert len(crashed.submission_receipts) == (
        PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX.submission_receipt_count
    )
    assert {item.name for item in receipt_root.iterdir()} == {"phase-lease.json"}

    monkeypatch.setattr(
        qualification_job, "_write_canonical_exclusive", original_writer
    )
    resumed = resume_gpu_qualification_job_submissions(
        DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
        plan_record=plan,
        submit_payloads=payloads,
        ledger_path=ledger_path,
        submit_receipt_root=receipt_root,
        local_preflight_evidence_path=local_preflight_path,
        opener=_SequentialOpener([{"run_id": 70_000 + index} for index in range(14)]),
    )
    assert len(resumed) == 14
    assert (receipt_root / "batch-reserved.json").is_file()
    assert (
        len(read_databricks_cluster_hour_ledger_json(ledger_path).submission_receipts)
        == PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX.submission_receipt_count + 14
    )


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("missing", "must be one regular file"),
        ("tampered", "closed_record_sha256 is invalid"),
        ("wrong-plan", "plan_sha256 mismatch"),
        ("future-completion", "must complete before qualification submission"),
        ("failed-check", "did not pass in canonical order"),
        ("wrong-check-id", "did not pass in canonical order"),
        ("resealed-tampered", "phase lease differs from the frozen batch"),
    ],
)
def test_resumer_rejects_invalid_local_preflight_without_reservation_or_post_side_effects(
    case: str,
    error: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ledger_path = tmp_path / "cluster-hours.json"
    receipt_root = tmp_path / "submit-receipts"
    local_preflight_path = tmp_path / "local-preflight.json"
    _copy_retained_campaign_ledger(ledger_path, monkeypatch)
    plan = _plan()
    payloads = _render(plan, _artifact_uris())
    _write_local_preflight(plan, local_preflight_path)
    original_writer = qualification_job._write_canonical_exclusive

    def crash_before_marker(record, path):
        if path.name == "batch-reserved.json":
            raise RuntimeError("simulated controller crash before batch marker")
        original_writer(record, path)

    monkeypatch.setattr(
        qualification_job, "_write_canonical_exclusive", crash_before_marker
    )
    with pytest.raises(RuntimeError, match="simulated controller crash"):
        submit_gpu_qualification_jobs(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            plan_record=plan,
            submit_payloads=payloads,
            ledger_path=ledger_path,
            submit_receipt_root=receipt_root,
            local_preflight_evidence_path=local_preflight_path,
            opener=lambda *_args, **_kwargs: pytest.fail("crash occurs before POST"),
            now=lambda: datetime(2026, 8, 24, 0, 0, tzinfo=UTC),
        )
    monkeypatch.setattr(
        qualification_job, "_write_canonical_exclusive", original_writer
    )

    if case == "missing":
        local_preflight_path.unlink()
    else:
        _invalid_local_preflight_path(
            case,
            plan=plan,
            path=local_preflight_path,
        )
    ledger_before_resume = ledger_path.read_bytes()
    opener = _SequentialOpener([{"run_id": 1}])

    with pytest.raises(ValueError, match=error):
        resume_gpu_qualification_job_submissions(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            plan_record=plan,
            submit_payloads=payloads,
            ledger_path=ledger_path,
            submit_receipt_root=receipt_root,
            local_preflight_evidence_path=local_preflight_path,
            opener=opener,
            now=lambda: datetime(2026, 8, 24, 0, 0, tzinfo=UTC),
        )

    assert ledger_path.read_bytes() == ledger_before_resume
    assert len(read_databricks_cluster_hour_ledger_json(ledger_path).reservations) == (
        PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX.reservation_count + 14
    )
    assert opener.requests == []
    assert {path.name for path in receipt_root.iterdir()} == {"phase-lease.json"}


def test_submitter_rejects_qualification_when_global_active_tasks_cannot_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan = _plan()
    payloads = _render(plan, _artifact_uris())
    ledger_path = tmp_path / "cluster-hours.json"
    _copy_retained_campaign_ledger(ledger_path, monkeypatch)
    for index in range(3):
        reserve_databricks_run_attempt_json(
            ledger_path,
            payloads[0],
            attempt_id=f"existing/{index}",
            workload_id="other-publication-work",
        )
    plan = _plan()
    payloads = _render(plan, _artifact_uris())
    opener = _SequentialOpener([{"run_id": 1}])
    local_preflight_path = _write_local_preflight(
        plan, tmp_path / "local-preflight.json"
    )

    with pytest.raises(ValueError, match="complete current ledger|global 16-job"):
        submit_gpu_qualification_jobs(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            plan_record=plan,
            submit_payloads=payloads,
            ledger_path=ledger_path,
            submit_receipt_root=tmp_path / "submit-receipts",
            local_preflight_evidence_path=local_preflight_path,
            opener=opener,
        )
    assert opener.requests == []
    assert not (tmp_path / "submit-receipts").exists()


def _seed_terminal_cluster_hours(ledger_path: Path, hours: float) -> None:
    remaining = hours
    index = 0
    while remaining > 0:
        actual_hours = min(12.0, remaining)
        payload = {
            "run_name": f"opening-{index}",
            "tasks": [
                {
                    "max_retries": 0,
                    "new_cluster": {"node_type_id": "g6.8xlarge"},
                    "task_key": f"opening_{index}",
                    "timeout_seconds": 12 * 60 * 60,
                }
            ],
            "timeout_seconds": 12 * 60 * 60,
        }
        attempt_id = f"opening/{index}"
        reserve_databricks_run_attempt_json(
            ledger_path,
            payload,
            attempt_id=attempt_id,
            workload_id="opening-balance",
        )
        record_databricks_run_terminal_actual_json(
            ledger_path,
            attempt_id=attempt_id,
            terminal_state="succeeded",
            actual_cluster_duration_seconds=actual_hours * 3600.0,
        )
        remaining -= actual_hours
        index += 1


def test_qualification_requires_1024_cap_and_preserves_exact_124h_headroom(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")

    small_ledger = tmp_path / "small-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        small_ledger,
        ledger_id=CAMPAIGN_LEDGER_ID,
        cap_cluster_hours=120.0,
    )
    monkeypatch.setattr(
        qualification_job,
        "databricks_ledger_path_sha256",
        lambda _path: PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
    )
    plan = _plan()
    payloads = _render(plan, _artifact_uris())
    small_opener = _SequentialOpener([{"run_id": 1}])
    local_preflight_path = _write_local_preflight(
        plan, tmp_path / "local-preflight.json"
    )
    with pytest.raises(ValueError, match="1024-hour"):
        submit_gpu_qualification_jobs(
            workspace,
            plan_record=plan,
            submit_payloads=payloads,
            ledger_path=small_ledger,
            submit_receipt_root=tmp_path / "small-receipts",
            local_preflight_evidence_path=local_preflight_path,
            opener=small_opener,
        )
    assert small_opener.requests == []
    assert read_databricks_cluster_hour_ledger_json(small_ledger).reservations == ()

    with pytest.raises(ValueError, match="frozen campaign"):
        _plan(campaign_opening_terminal_gpu_hours=844.0)


def test_qualification_rejects_fresh_same_id_ledger_reset_before_batch_or_post(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    reset_ledger = tmp_path / "reset-ledger.json"
    create_databricks_cluster_hour_ledger_json(
        reset_ledger,
        ledger_id=CAMPAIGN_LEDGER_ID,
    )
    monkeypatch.setattr(
        qualification_job,
        "databricks_ledger_path_sha256",
        lambda _path: PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
    )
    plan = _plan()
    payloads = _render(plan, _artifact_uris())
    opener = _SequentialOpener([{"run_id": 1}])
    local_preflight_path = _write_local_preflight(
        plan, tmp_path / "local-preflight.json"
    )
    with pytest.raises(ValueError, match="shorter than its authorized prefix"):
        submit_gpu_qualification_jobs(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            plan_record=plan,
            submit_payloads=payloads,
            ledger_path=reset_ledger,
            submit_receipt_root=tmp_path / "reset-receipts",
            local_preflight_evidence_path=local_preflight_path,
            opener=opener,
        )
    assert opener.requests == []
    assert read_databricks_cluster_hour_ledger_json(reset_ledger).reservations == ()


def test_authorizing_collector_has_no_caller_supplied_http_transport():
    assert (
        "opener" not in inspect.signature(collect_gpu_qualification_evidence).parameters
    )


def test_control_plane_terminal_validation_binds_run_task_and_launch_cluster():
    plan = _plan()
    payloads = _render(plan, _artifact_uris())
    contract = qualification_job._validated_qualification_payloads(plan, payloads)[0]
    planned_job = plan["cloud_qualification"]["jobs"][0]
    submitted_task = payloads[0]["tasks"][0]
    run = {
        "end_time": 2_000_000,
        "run_id": 10_000,
        "run_name": payloads[0]["run_name"],
        "run_type": "SUBMIT_RUN",
        "start_time": 1_000_000,
        "state": {
            "life_cycle_state": "TERMINATED",
            "result_state": "SUCCESS",
        },
        "tasks": [
            {
                "attempt_number": 0,
                "cluster_instance": {"cluster_id": "cluster-10000"},
                "end_time": 1_900_000,
                "new_cluster": submitted_task["new_cluster"],
                "run_id": 20_000,
                "start_time": 1_100_000,
                "state": {
                    "life_cycle_state": "TERMINATED",
                    "result_state": "SUCCESS",
                },
                "task_key": submitted_task["task_key"],
            }
        ],
    }
    submit_receipt = {"cloud_run_id": "10000"}

    identity = qualification_job._validate_control_plane_run(
        run,
        planned_job=planned_job,
        contract=contract,
        submit_receipt=submit_receipt,
    )
    assert identity["succeeded"] is True
    assert identity["cloud_cluster_id"] == "cluster-10000"
    assert identity["task_run_id"] == "20000"

    retried = json.loads(json.dumps(run))
    retried["tasks"][0]["attempt_number"] = 1
    with pytest.raises(ValueError, match="retried"):
        qualification_job._validate_control_plane_run(
            retried,
            planned_job=planned_job,
            contract=contract,
            submit_receipt=submit_receipt,
        )
    substituted = json.loads(json.dumps(run))
    substituted["tasks"][0]["new_cluster"]["node_type_id"] = "g4dn.xlarge"
    with pytest.raises(ValueError, match="node_type_id differs"):
        qualification_job._validate_control_plane_run(
            substituted,
            planned_job=planned_job,
            contract=contract,
            submit_receipt=submit_receipt,
        )


def test_terminal_receipt_closure_is_published_atomically(tmp_path: Path, monkeypatch):
    root = tmp_path / "terminal-receipts"
    receipts = [
        {"job_id": "job-a", "closed_record_sha256": "a" * 64},
        {"job_id": "job-b", "closed_record_sha256": "b" * 64},
    ]
    real_write = qualification_job._write_canonical_exclusive
    calls = 0

    def fail_second(record, path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated durable write failure")
        real_write(record, path)

    monkeypatch.setattr(qualification_job, "_write_canonical_exclusive", fail_second)
    with pytest.raises(OSError, match="simulated durable write failure"):
        qualification_job._publish_terminal_receipts_atomic(root, receipts)
    assert not root.exists()
    assert not list(tmp_path.glob(".terminal-receipts.staging-*"))


def test_terminal_receipt_publish_requires_post_rename_directory_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "terminal-receipts"
    receipts = [
        {"job_id": "job-a", "closed_record_sha256": "a" * 64},
        {"job_id": "job-b", "closed_record_sha256": "b" * 64},
    ]
    real_fsync_directory = qualification_job._fsync_directory

    def fail_post_rename(path: Path) -> None:
        if path == tmp_path and root.is_dir():
            raise OSError("simulated post-rename directory fsync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(qualification_job, "_fsync_directory", fail_post_rename)

    with pytest.raises(OSError, match="post-rename directory fsync failure"):
        qualification_job._publish_terminal_receipts_atomic(root, receipts)

    assert {path.name for path in root.iterdir()} == {"job-a.json", "job-b.json"}


def test_exclusive_evidence_write_fails_closed_when_directory_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "evidence.json"
    calls = 0

    def fail_first(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(qualification_job, "_fsync_directory", fail_first)

    with pytest.raises(OSError, match="directory fsync failure"):
        qualification_job._write_canonical_exclusive(
            {"closed_record_sha256": "a" * 64}, output
        )

    assert calls == 2
    assert not output.exists()


def test_collected_identity_closure_rejects_reused_cluster_or_task_run_id():
    contracts = (
        {
            "job_id": "job-a",
            "output_json": "dbfs:/results/plan/job-a/gpu-job-result.json",
            "reservation_attempt_id": "attempt-a",
            "submit_payload_sha256": "a" * 64,
            "task_key": "task_a",
        },
        {
            "job_id": "job-b",
            "output_json": "dbfs:/results/plan/job-b/gpu-job-result.json",
            "reservation_attempt_id": "attempt-b",
            "submit_payload_sha256": "b" * 64,
            "task_key": "task_b",
        },
    )
    receipts = [
        {
            **contract,
            "cloud_cluster_id": "shared-cluster",
            "cloud_run_id": str(100 + index),
            "task_run_id": str(200 + index),
        }
        for index, contract in enumerate(contracts)
    ]

    with pytest.raises(ValueError, match="cloud_cluster_id values must be unique"):
        qualification_job._validate_collected_identity_closure(
            receipts, contracts=contracts
        )

    receipts[1]["cloud_cluster_id"] = "cluster-b"
    receipts[1]["task_run_id"] = receipts[0]["task_run_id"]
    with pytest.raises(ValueError, match="task_run_id values must be unique"):
        qualification_job._validate_collected_identity_closure(
            receipts, contracts=contracts
        )


def test_retained_uc_failure_evidence_refuses_false_zero_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = (
        Path(__file__).parents[1]
        / "databricks-runs/vllm-0271-publication-prep/"
        "gpu-qualification-plan-sha256-"
        "ebfeaf53cfa9c74400be59546b391b77ebde4e85defa1f1b11bc4b4255c80341"
    ).resolve()
    plan = json.loads((root / "gpu-qualification-plan.json").read_text())
    assert plan["runtime_contract"]["artifact_sha256"][
        "runtime_lock_sha256"
    ] == qualification_job.GPU_QUALIFICATION_LEGACY_UC_RUNTIME_LOCK_SHA256
    payloads = json.loads((root / "submit-payloads.json").read_text())
    contracts = qualification_job._validated_qualification_payloads(
        plan,
        payloads,
        require_legacy_uc_broken_security_shape=True,
    )
    binding = qualification_job._non_authorizing_local_preflight_binding(
        root / "local-preflight-valid/local-preflight-evidence.json",
        plan=plan,
    )
    batch_authorization, marker = qualification_job._replay_qualification_batch_marker(
        plan=plan,
        contracts=contracts,
        ledger_path=_RETAINED_LEDGER_PATH,
        submit_receipt_root=root / "submit-receipts",
        local_preflight_binding=binding,
    )
    ledger_path = tmp_path.resolve() / "cluster-hours.json"
    shutil.copyfile(_RETAINED_LEDGER_PATH, ledger_path)
    before = ledger_path.read_bytes()
    monkeypatch.setattr(
        qualification_job,
        "databricks_ledger_path_sha256",
        lambda _path: plan["campaign_ledger_path_sha256"],
    )
    monkeypatch.setattr(
        qualification_job,
        "_replay_qualification_batch_marker",
        lambda **_kwargs: (batch_authorization, marker),
    )
    terminal_prefix = qualification_job._require_qualification_phase_ledger_closure(
        read_databricks_cluster_hour_ledger_json(_RETAINED_LEDGER_PATH),
        batch_authorization=batch_authorization,
        contracts=contracts,
    )
    assert terminal_prefix.to_record() == {
        "cap_cluster_hours": 1024.0,
        "ledger_id": "representative-canary-823bd9d82a5c1730",
        "prefix_sha256": (
            "4bbe1144d4ce037fd8cf3376fc20c4e19ad00641f84c0a54d0cc2c17e37bf728"
        ),
        "reservation_count": 152,
        "submission_receipt_count": 14,
        "terminal_actual_count": 152,
    }

    reviewed_manifest_sha256 = (
        qualification_job.GPU_QUALIFICATION_LEGACY_UC_FAILURE_MANIFEST_CLOSED_RECORD_SHA256
    )
    monkeypatch.setattr(
        qualification_job,
        "GPU_QUALIFICATION_LEGACY_UC_FAILURE_MANIFEST_CLOSED_RECORD_SHA256",
        "0" * 64,
    )
    with pytest.raises(ValueError, match="manifest is not reviewed"):
        qualification_job.reconcile_gpu_qualification_failed_attempt_evidence(
            plan_record=plan,
            submit_payloads=payloads,
            ledger_path=ledger_path,
            submit_receipt_root=root / "submit-receipts",
            local_preflight_evidence_path=(
                root / "local-preflight-valid/local-preflight-evidence.json"
            ),
            runs_get_evidence_root=root / "failed-attempt-uc-volume-access",
        )
    monkeypatch.setattr(
        qualification_job,
        "GPU_QUALIFICATION_LEGACY_UC_FAILURE_MANIFEST_CLOSED_RECORD_SHA256",
        reviewed_manifest_sha256,
    )
    assert ledger_path.read_bytes() == before

    with pytest.raises(ValueError, match="nonzero task intervals"):
        qualification_job.reconcile_gpu_qualification_failed_attempt_evidence(
            plan_record=plan,
            submit_payloads=payloads,
            ledger_path=ledger_path,
            submit_receipt_root=root / "submit-receipts",
            local_preflight_evidence_path=(
                root / "local-preflight-valid/local-preflight-evidence.json"
            ),
            runs_get_evidence_root=root / "failed-attempt-uc-volume-access",
            require_zero_actual=True,
        )

    assert ledger_path.read_bytes() == before

    retained = read_databricks_cluster_hour_ledger_json(_RETAINED_LEDGER_PATH)
    pre_terminal = replace(
        retained,
        reservations=retained.reservations[:152],
        submission_receipts=retained.submission_receipts[:14],
        terminal_actuals=retained.terminal_actuals[:138],
    )
    ledger_path.write_text(
        json.dumps(
            databricks_cluster_hour_ledger_to_record(pre_terminal),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    reconciled = qualification_job.reconcile_gpu_qualification_failed_attempt_evidence(
        plan_record=plan,
        submit_payloads=payloads,
        ledger_path=ledger_path,
        submit_receipt_root=root / "submit-receipts",
        local_preflight_evidence_path=(
            root / "local-preflight-valid/local-preflight-evidence.json"
        ),
        runs_get_evidence_root=root / "failed-attempt-uc-volume-access",
    )
    assert databricks_ledger_prefix(reconciled) == terminal_prefix


def _failed_v2_capture_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    Path,
    Path,
    Path,
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    retained = read_databricks_cluster_hour_ledger_json(_RETAINED_LEDGER_PATH)
    opening_prefix = PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX
    opening = replace(
        retained,
        reservations=retained.reservations[: opening_prefix.reservation_count],
        submission_receipts=retained.submission_receipts[
            : opening_prefix.submission_receipt_count
        ],
        terminal_actuals=retained.terminal_actuals[
            : opening_prefix.terminal_actual_count
        ],
    )
    ledger_path = tmp_path / "cluster-hours.json"
    ledger_path.write_text(
        json.dumps(
            databricks_cluster_hour_ledger_to_record(opening),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        qualification_job,
        "databricks_ledger_path_sha256",
        lambda _path: PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
    )
    plan = _plan()
    payloads = _render(plan, _artifact_uris())
    submit_root = tmp_path / "submit-receipts"
    preflight_path = _write_local_preflight(plan, tmp_path / "local-preflight.json")
    parent_run_ids = [70_000 + index for index in range(14)]
    submit_gpu_qualification_jobs(
        DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
        plan_record=plan,
        submit_payloads=payloads,
        ledger_path=ledger_path,
        submit_receipt_root=submit_root,
        local_preflight_evidence_path=preflight_path,
        opener=_SequentialOpener(
            [{"run_id": run_id} for run_id in parent_run_ids]
        ),
        now=lambda: datetime(2026, 8, 24, 0, 0, tzinfo=UTC),
    )
    runs: dict[str, dict[str, Any]] = {}
    outputs: dict[str, dict[str, Any]] = {}
    for index, (payload, parent_run_id) in enumerate(
        zip(payloads, parent_run_ids, strict=True)
    ):
        submitted_task = payload["tasks"][0]
        child_run_id = 80_000 + index
        task_start = 1_787_533_140_000 + index * 10_000
        task_end = task_start + 300_000 + index * 1_000
        task = {
            "attempt_number": 0,
            "cluster_instance": {"cluster_id": f"cluster-{parent_run_id}"},
            "end_time": task_end,
            "new_cluster": {
                **submitted_task["new_cluster"],
                "enable_elastic_disk": False,
            },
            "run_id": child_run_id,
            "spark_python_task": submitted_task["spark_python_task"],
            "start_time": task_start,
            "state": {
                "life_cycle_state": "INTERNAL_ERROR",
                "result_state": "FAILED",
            },
            "status": {"state": "TERMINATED"},
            "task_key": submitted_task["task_key"],
        }
        run = {
            "end_time": task_end + 1_000,
            "run_id": parent_run_id,
            "run_name": payload["run_name"],
            "run_type": "SUBMIT_RUN",
            "start_time": task_start - 1_000,
            "state": {
                "life_cycle_state": "INTERNAL_ERROR",
                "result_state": "FAILED",
            },
            "status": {"state": "TERMINATED"},
            "tasks": [task],
        }
        run_output = {
            "error": "NameError: name '__file__' is not defined",
            "error_trace": (
                "Traceback (most recent call last):\n"
                "  File \"gpu-qualification-bootstrap.py\", line 1\n"
                "NameError: name '__file__' is not defined\n"
            ),
            "logs": "driver log prefix\nNameError: name '__file__' is not defined\n",
            "logs_truncated": False,
            "metadata": {
                "end_time": task_end,
                "job_run_id": parent_run_id,
                "parent_run_id": parent_run_id,
                "run_id": child_run_id,
                "start_time": task_start,
                "status": {"state": "TERMINATED"},
                "task_key": submitted_task["task_key"],
                "tasks": [
                    {
                        "run_id": child_run_id,
                        "status": {"state": "TERMINATED"},
                        "task_key": submitted_task["task_key"],
                    }
                ],
            },
        }
        runs[str(parent_run_id)] = run
        outputs[str(child_run_id)] = run_output
    return (
        plan,
        payloads,
        ledger_path,
        submit_root,
        preflight_path,
        runs,
        outputs,
    )


def _reviewed_logged_run_output() -> dict[str, Any]:
    return {
        "error": "RuntimeError: cluster identity unavailable",
        "error_trace": "RuntimeError: cluster identity unavailable\n",
        "logs": "driver output\n",
        "logs_truncated": False,
        "metadata": {},
    }


def test_failed_run_output_schema_accepts_historical_and_logged_shapes():
    logged = _reviewed_logged_run_output()
    qualification_job._validate_failed_run_output_schema(logged)
    qualification_job._validate_failed_run_output_schema(
        {key: value for key, value in logged.items() if not key.startswith("logs")}
    )


def test_failed_run_output_schema_rejects_unknown_extra_field():
    output = _reviewed_logged_run_output()
    output["unreviewed"] = "field"

    with pytest.raises(ValueError, match="open schema"):
        qualification_job._validate_failed_run_output_schema(output)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("logs", 1, "logs must be an exact string"),
        ("logs", True, "logs must be an exact string"),
        ("logs_truncated", 0, "logs_truncated must be an exact bool"),
        ("logs_truncated", "false", "logs_truncated must be an exact bool"),
    ],
)
def test_failed_run_output_schema_rejects_wrong_log_types(
    field_name: str,
    value: object,
    message: str,
):
    output = _reviewed_logged_run_output()
    output[field_name] = value

    with pytest.raises(ValueError, match=message):
        qualification_job._validate_failed_run_output_schema(output)


def test_failed_run_output_schema_rejects_oversize_and_invalid_utf8_logs():
    output = _reviewed_logged_run_output()
    output["logs"] = "x" * (
        qualification_job.GPU_QUALIFICATION_RUN_OUTPUT_LOG_MAX_UTF8_BYTES + 1
    )
    with pytest.raises(ValueError, match="UTF-8 byte cap"):
        qualification_job._validate_failed_run_output_schema(output)

    output["logs"] = "invalid-surrogate-\ud800"
    with pytest.raises(ValueError, match="valid UTF-8"):
        qualification_job._validate_failed_run_output_schema(output)


def test_logged_run_output_tamper_changes_raw_record_and_file_bindings():
    original = _reviewed_logged_run_output()
    tampered = json.loads(json.dumps(original))
    tampered["logs"] = "different driver output\n"

    assert qualification_job._canonical_json_sha256(original) != (
        qualification_job._canonical_json_sha256(tampered)
    )
    assert qualification_job._canonical_record_file_sha256(original) != (
        qualification_job._canonical_record_file_sha256(tampered)
    )


def test_v2_failed_capture_is_read_only_and_reviewed_reconciliation_is_ordered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        plan,
        payloads,
        ledger_path,
        submit_root,
        preflight_path,
        runs,
        outputs,
    ) = _failed_v2_capture_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        qualification_job,
        "get_databricks_run",
        lambda _config, run_id: runs[run_id],
    )
    monkeypatch.setattr(
        qualification_job,
        "get_databricks_run_output",
        lambda _config, run_id: outputs[run_id],
    )
    evidence_root = tmp_path / "failed-attempt-v2"
    ledger_before_capture = ledger_path.read_bytes()
    failure_reason = "bootstrap self-identity used an unavailable __file__ global"
    expected_error = "NameError: name '__file__' is not defined"

    manifest = (
        qualification_job.capture_gpu_qualification_failed_attempt_evidence_v2(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            plan_record=plan,
            submit_payloads=payloads,
            ledger_path=ledger_path,
            submit_receipt_root=submit_root,
            local_preflight_evidence_path=preflight_path,
            evidence_root=evidence_root,
            failure_reason=failure_reason,
            expected_error=expected_error,
        )
    )

    assert ledger_path.read_bytes() == ledger_before_capture
    assert len(list(evidence_root.iterdir())) == 29
    assert not list(tmp_path.glob(".failed-attempt-v2.staging-*"))
    assert [entry["job_id"] for entry in manifest["entries"]] == sorted(
        entry["job_id"] for entry in manifest["entries"]
    )
    assert all(
        entry["run_output_error"] == expected_error
        and len(entry["run_output_error_trace_sha256"]) == 64
        and len(entry["run_output_file_sha256"]) == 64
        for entry in manifest["entries"]
    )
    for entry in manifest["entries"]:
        output_path = evidence_root / (
            f"{entry['job_id']}.runs-get-output.json"
        )
        raw_output = json.loads(output_path.read_text(encoding="utf-8"))
        assert set(raw_output) == {
            "error",
            "error_trace",
            "logs",
            "logs_truncated",
            "metadata",
        }
        assert entry["run_output_file_sha256"] == qualification_job._file_sha256(
            output_path
        )
        assert entry["run_output_record_sha256"] == (
            qualification_job._canonical_json_sha256(raw_output)
        )
    terminal_prefix = manifest["ledger_lineage"]["terminal_prefix"]
    manifest_file_sha256 = qualification_job._file_sha256(
        evidence_root / "reconciliation-manifest.json"
    )
    opening_prefix = PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX
    assert terminal_prefix["reservation_count"] == (
        opening_prefix.reservation_count + 14
    )
    assert terminal_prefix["submission_receipt_count"] == (
        opening_prefix.submission_receipt_count + 14
    )
    assert terminal_prefix["terminal_actual_count"] == (
        opening_prefix.terminal_actual_count + 14
    )

    before_rejected_review = ledger_path.read_bytes()
    with pytest.raises(ValueError, match="manifest is not reviewed"):
        qualification_job._reconcile_reviewed_gpu_qualification_failed_attempt_evidence_v2(
            plan_record=plan,
            submit_payloads=payloads,
            ledger_path=ledger_path,
            submit_receipt_root=submit_root,
            local_preflight_evidence_path=preflight_path,
            runs_get_evidence_root=evidence_root,
            expected_plan_sha256=plan["closed_record_sha256"],
            expected_runner_sha256=GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256,
            expected_manifest_closed_record_sha256="0" * 64,
            expected_manifest_file_sha256=manifest_file_sha256,
            expected_terminal_prefix_sha256=terminal_prefix["prefix_sha256"],
            expected_failure_reason=failure_reason,
            expected_error=expected_error,
            expected_run_output_keys=qualification_job._FAILED_RUN_OUTPUT_LOGGED_KEYS,
        )
    assert ledger_path.read_bytes() == before_rejected_review

    tampered_output_path = evidence_root / (
        f"{manifest['entries'][0]['job_id']}.runs-get-output.json"
    )
    untampered_output_bytes = tampered_output_path.read_bytes()
    tampered_output = json.loads(untampered_output_bytes)
    tampered_output["logs"] += "tampered\n"
    tampered_output_path.write_bytes(
        qualification_job._canonical_stdlib_json_bytes(
            tampered_output,
            pretty=False,
        )
        + b"\n"
    )
    before_tampered_review = ledger_path.read_bytes()
    with pytest.raises(ValueError, match="entries differ"):
        qualification_job._reconcile_reviewed_gpu_qualification_failed_attempt_evidence_v2(
            plan_record=plan,
            submit_payloads=payloads,
            ledger_path=ledger_path,
            submit_receipt_root=submit_root,
            local_preflight_evidence_path=preflight_path,
            runs_get_evidence_root=evidence_root,
            expected_plan_sha256=plan["closed_record_sha256"],
            expected_runner_sha256=GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256,
            expected_manifest_closed_record_sha256=manifest[
                "closed_record_sha256"
            ],
            expected_manifest_file_sha256=manifest_file_sha256,
            expected_terminal_prefix_sha256=terminal_prefix["prefix_sha256"],
            expected_failure_reason=failure_reason,
            expected_error=expected_error,
            expected_run_output_keys=qualification_job._FAILED_RUN_OUTPUT_LOGGED_KEYS,
        )
    assert ledger_path.read_bytes() == before_tampered_review
    tampered_output_path.write_bytes(untampered_output_bytes)

    reconciled = qualification_job._reconcile_reviewed_gpu_qualification_failed_attempt_evidence_v2(
        plan_record=plan,
        submit_payloads=payloads,
        ledger_path=ledger_path,
        submit_receipt_root=submit_root,
        local_preflight_evidence_path=preflight_path,
        runs_get_evidence_root=evidence_root,
        expected_plan_sha256=plan["closed_record_sha256"],
        expected_runner_sha256=GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256,
        expected_manifest_closed_record_sha256=manifest["closed_record_sha256"],
        expected_manifest_file_sha256=manifest_file_sha256,
        expected_terminal_prefix_sha256=terminal_prefix["prefix_sha256"],
        expected_failure_reason=failure_reason,
        expected_error=expected_error,
        expected_run_output_keys=qualification_job._FAILED_RUN_OUTPUT_LOGGED_KEYS,
    )
    contracts = qualification_job._validated_qualification_payloads(plan, payloads)
    expected_attempt_order = [
        contract["reservation_attempt_id"]
        for contract in sorted(contracts, key=lambda item: item["job_id"])
    ]
    assert [item.attempt_id for item in reconciled.terminal_actuals[-14:]] == (
        expected_attempt_order
    )
    assert reconciled.active_reserved_cluster_hours == 0.0
    assert databricks_ledger_prefix(reconciled).to_record() == terminal_prefix

    assert (
        qualification_job._reconcile_reviewed_gpu_qualification_failed_attempt_evidence_v2(
            plan_record=plan,
            submit_payloads=payloads,
            ledger_path=ledger_path,
            submit_receipt_root=submit_root,
            local_preflight_evidence_path=preflight_path,
            runs_get_evidence_root=evidence_root,
            expected_plan_sha256=plan["closed_record_sha256"],
            expected_runner_sha256=GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256,
            expected_manifest_closed_record_sha256=manifest[
                "closed_record_sha256"
            ],
            expected_manifest_file_sha256=manifest_file_sha256,
            expected_terminal_prefix_sha256=terminal_prefix["prefix_sha256"],
            expected_failure_reason=failure_reason,
            expected_error=expected_error,
            expected_run_output_keys=qualification_job._FAILED_RUN_OUTPUT_LOGGED_KEYS,
        )
        == reconciled
    )


def test_v2_failed_capture_accepts_reversed_per_job_expected_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        plan,
        payloads,
        ledger_path,
        submit_root,
        preflight_path,
        runs,
        outputs,
    ) = _failed_v2_capture_fixture(tmp_path, monkeypatch)
    contracts = qualification_job._validated_qualification_payloads(plan, payloads)
    expected_errors: dict[str, str] = {}
    for index, contract in enumerate(contracts):
        job_id = str(contract["job_id"])
        error = f"RuntimeError: job-specific bootstrap failure {index:02d}"
        child_run_id = str(80_000 + index)
        outputs[child_run_id]["error"] = error
        outputs[child_run_id]["error_trace"] = f"Traceback:\n{error}\n"
        outputs[child_run_id]["logs"] = f"driver log prefix\n{error}\n"
        expected_errors[job_id] = error
    reversed_expected_errors = dict(reversed(tuple(expected_errors.items())))
    monkeypatch.setattr(
        qualification_job,
        "get_databricks_run",
        lambda _config, run_id: runs[run_id],
    )
    monkeypatch.setattr(
        qualification_job,
        "get_databricks_run_output",
        lambda _config, run_id: outputs[run_id],
    )
    evidence_root = tmp_path / "failed-attempt-v2-by-job"
    ledger_before_capture = ledger_path.read_bytes()

    manifest = (
        qualification_job.capture_gpu_qualification_failed_attempt_evidence_v2_by_job(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            plan_record=plan,
            submit_payloads=payloads,
            ledger_path=ledger_path,
            submit_receipt_root=submit_root,
            local_preflight_evidence_path=preflight_path,
            evidence_root=evidence_root,
            failure_reason="each job reported its own exact bootstrap failure",
            expected_errors_by_job=reversed_expected_errors,
        )
    )

    assert ledger_path.read_bytes() == ledger_before_capture
    assert len(list(evidence_root.iterdir())) == 29
    assert not list(tmp_path.glob(".failed-attempt-v2-by-job.staging-*"))
    assert tuple(reversed_expected_errors) == tuple(reversed(tuple(expected_errors)))
    assert {
        str(entry["job_id"]): str(entry["run_output_error"])
        for entry in manifest["entries"]
    } == expected_errors


def test_v2_failed_capture_rejects_invalid_per_job_errors_before_get_or_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        plan,
        payloads,
        ledger_path,
        submit_root,
        preflight_path,
        _runs,
        _outputs,
    ) = _failed_v2_capture_fixture(tmp_path, monkeypatch)
    contracts = qualification_job._validated_qualification_payloads(plan, payloads)
    job_ids = tuple(str(contract["job_id"]) for contract in contracts)
    error = "NameError: name '__file__' is not defined"
    exact: dict[str, str] = {job_id: error for job_id in job_ids}
    missing: dict[Any, Any] = dict(exact)
    missing.pop(job_ids[-1])
    extra: dict[Any, Any] = dict(exact)
    extra["unexpected-job"] = error
    bool_key: dict[Any, Any] = dict(exact)
    bool_key[True] = error
    bool_value: dict[Any, Any] = dict(exact)
    bool_value[job_ids[0]] = True
    cases: tuple[tuple[Any, type[Exception], str], ...] = (
        (missing, ValueError, "exact planned job IDs"),
        (extra, ValueError, "exact planned job IDs"),
        (bool_key, ValueError, "job ID must be a non-empty, trimmed string"),
        (bool_value, ValueError, "must be a non-empty, trimmed string"),
        ([], TypeError, "must be a mapping"),
    )
    get_calls: list[str] = []

    def unexpected_get(_config: DatabricksWorkspaceConfig, run_id: str) -> Any:
        get_calls.append(run_id)
        raise AssertionError("invalid expected errors reached a package-owned GET")

    monkeypatch.setattr(qualification_job, "get_databricks_run", unexpected_get)
    monkeypatch.setattr(
        qualification_job,
        "get_databricks_run_output",
        unexpected_get,
    )
    ledger_before_capture = ledger_path.read_bytes()

    for index, (candidate, error_type, message) in enumerate(cases):
        evidence_root = tmp_path / f"invalid-failed-capture-{index}"
        with pytest.raises(error_type, match=message):
            qualification_job.capture_gpu_qualification_failed_attempt_evidence_v2_by_job(
                DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
                plan_record=plan,
                submit_payloads=payloads,
                ledger_path=ledger_path,
                submit_receipt_root=submit_root,
                local_preflight_evidence_path=preflight_path,
                evidence_root=evidence_root,
                failure_reason="reviewed exact per-job bootstrap failures",
                expected_errors_by_job=candidate,
            )
        assert not evidence_root.exists()
        assert not list(tmp_path.glob(f".{evidence_root.name}.staging-*"))

    assert get_calls == []
    assert ledger_path.read_bytes() == ledger_before_capture


def test_v2_failure_capture_boundary_has_no_caller_supplied_transport():
    legacy_parameters = inspect.signature(
        qualification_job.capture_gpu_qualification_failed_attempt_evidence_v2
    ).parameters
    per_job_parameters = inspect.signature(
        qualification_job.capture_gpu_qualification_failed_attempt_evidence_v2_by_job
    ).parameters

    assert "opener" not in legacy_parameters
    assert "expected_error" in legacy_parameters
    assert "expected_errors_by_job" not in legacy_parameters
    assert "opener" not in per_job_parameters
    assert "expected_errors_by_job" in per_job_parameters
    assert "expected_error" not in per_job_parameters


def test_v2_legacy_capture_delegates_one_error_to_every_planned_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        plan,
        payloads,
        ledger_path,
        submit_root,
        preflight_path,
        _runs,
        _outputs,
    ) = _failed_v2_capture_fixture(tmp_path, monkeypatch)
    captured: dict[str, Any] = {}

    def capture_by_job(
        _config: DatabricksWorkspaceConfig,
        **kwargs: Any,
    ) -> dict[str, Any]:
        captured.update(kwargs)
        return {"delegated": True}

    monkeypatch.setattr(
        qualification_job,
        "capture_gpu_qualification_failed_attempt_evidence_v2_by_job",
        capture_by_job,
    )
    expected_error = "NameError: name '__file__' is not defined"

    result = qualification_job.capture_gpu_qualification_failed_attempt_evidence_v2(
        DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
        plan_record=plan,
        submit_payloads=payloads,
        ledger_path=ledger_path,
        submit_receipt_root=submit_root,
        local_preflight_evidence_path=preflight_path,
        evidence_root=tmp_path / "delegated-capture",
        failure_reason="legacy reviewed failure",
        expected_error=expected_error,
    )

    contracts = qualification_job._validated_qualification_payloads(plan, payloads)
    assert result == {"delegated": True}
    assert captured["expected_errors_by_job"] == {
        str(contract["job_id"]): expected_error for contract in contracts
    }


def test_reviewed_bootstrap_file_global_wrapper_reconciles_retained_v2_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan_root = (
        Path(__file__).parents[1]
        / "databricks-runs/vllm-0271-publication-prep/"
        "gpu-qualification-plan-sha256-"
        "2cf4ef1092a435c1e713f2a94115021ea7069ab6295d18ce5fcb5d4a479ce997"
    ).resolve()
    evidence_root = plan_root / "failed-attempt-bootstrap-file-global-v2"
    plan = json.loads((plan_root / "gpu-qualification-plan.json").read_text())
    payloads = json.loads((plan_root / "submit-payloads.json").read_text())
    manifest = json.loads(
        (evidence_root / "reconciliation-manifest.json").read_text()
    )
    retained = read_databricks_cluster_hour_ledger_json(_RETAINED_LEDGER_PATH)
    active_incident = replace(
        retained,
        reservations=retained.reservations[:166],
        submission_receipts=retained.submission_receipts[:28],
        terminal_actuals=retained.terminal_actuals[:152],
    )
    ledger_path = tmp_path / "cluster-hours.json"
    ledger_path.write_text(
        json.dumps(
            databricks_cluster_hour_ledger_to_record(active_incident),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        qualification_job,
        "databricks_ledger_path_sha256",
        lambda _path: plan["campaign_ledger_path_sha256"],
    )
    monkeypatch.setattr(
        resource_ledger,
        "databricks_ledger_path_sha256",
        lambda _path: plan["campaign_ledger_path_sha256"],
    )

    reconciled = qualification_job.reconcile_gpu_qualification_bootstrap_file_global_failure_evidence(
        plan_record=plan,
        submit_payloads=payloads,
        ledger_path=ledger_path,
        submit_receipt_root=plan_root / "submit-receipts",
        local_preflight_evidence_path=(
            plan_root / "local-preflight-valid/local-preflight-evidence.json"
        ),
        runs_get_evidence_root=evidence_root,
    )

    assert plan["closed_record_sha256"] == (
        qualification_job.GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_PLAN_SHA256
    )
    assert plan["runtime_contract"]["artifact_sha256"]["runner_sha256"] == (
        qualification_job.GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_RUNNER_SHA256
    )
    assert manifest["closed_record_sha256"] == (
        qualification_job.GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_MANIFEST_SHA256
    )
    assert qualification_job._file_sha256(
        evidence_root / "reconciliation-manifest.json"
    ) == "1d0246ece1d6f844420d22a26b729d3f0d971ca0b30c0bf1ef0b5a84dcf6f360"
    assert databricks_ledger_prefix(reconciled).prefix_sha256 == (
        qualification_job.GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_TERMINAL_PREFIX_SHA256
    )
    assert sum(
        item.actual_cluster_duration_seconds
        for item in reconciled.terminal_actuals[-14:]
    ) == pytest.approx(4585.717999999999)
    assert reconciled.active_reserved_cluster_hours == 0.0


def _cluster_identity_failure_replay_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan_root = (
        Path(__file__).parents[1]
        / "databricks-runs/vllm-0271-publication-prep/"
        "gpu-qualification-plan-sha256-"
        "d6f7619f6a70311fac571b31bedc7974e756a1679218cf63b76a7e7ceb91ebec"
    ).resolve()
    evidence_root = plan_root / "failed-attempt-cluster-identity-v2"
    plan = json.loads((plan_root / "gpu-qualification-plan.json").read_text())
    payloads = json.loads((plan_root / "submit-payloads.json").read_text())
    manifest = json.loads(
        (evidence_root / "reconciliation-manifest.json").read_text()
    )
    retained = read_databricks_cluster_hour_ledger_json(_RETAINED_LEDGER_PATH)
    active_incident = replace(
        retained,
        reservations=retained.reservations[:180],
        submission_receipts=retained.submission_receipts[:42],
        terminal_actuals=retained.terminal_actuals[:166],
    )
    ledger_path = tmp_path / "cluster-hours.json"
    ledger_path.write_text(
        json.dumps(
            databricks_cluster_hour_ledger_to_record(active_incident),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        qualification_job,
        "databricks_ledger_path_sha256",
        lambda _path: plan["campaign_ledger_path_sha256"],
    )
    monkeypatch.setattr(
        resource_ledger,
        "databricks_ledger_path_sha256",
        lambda _path: plan["campaign_ledger_path_sha256"],
    )
    return plan_root, evidence_root, plan, payloads, manifest, ledger_path


def _reconcile_cluster_identity_failure(
    *,
    plan_root: Path,
    evidence_root: Path,
    plan: dict[str, Any],
    payloads: list[dict[str, Any]],
    ledger_path: Path,
) -> DatabricksClusterHourLedger:
    return qualification_job.reconcile_gpu_qualification_bootstrap_cluster_identity_failure_evidence(
        plan_record=plan,
        submit_payloads=payloads,
        ledger_path=ledger_path,
        submit_receipt_root=plan_root / "submit-receipts",
        local_preflight_evidence_path=(
            plan_root / "local-preflight-valid/local-preflight-evidence.json"
        ),
        runs_get_evidence_root=evidence_root,
    )


def test_reviewed_cluster_identity_wrapper_reconciles_exact_historical_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        plan_root,
        evidence_root,
        plan,
        payloads,
        manifest,
        ledger_path,
    ) = _cluster_identity_failure_replay_fixture(tmp_path, monkeypatch)
    live_ledger_before = _RETAINED_LEDGER_PATH.read_bytes()
    evidence_before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in evidence_root.iterdir()
    }

    assert len(evidence_before) == 29
    assert all(
        set(json.loads(path.read_text()))
        == qualification_job._FAILED_RUN_OUTPUT_LOGGED_KEYS
        for path in evidence_root.glob("*.runs-get-output.json")
    )
    reconciled = _reconcile_cluster_identity_failure(
        plan_root=plan_root,
        evidence_root=evidence_root,
        plan=plan,
        payloads=payloads,
        ledger_path=ledger_path,
    )

    assert plan["closed_record_sha256"] == (
        qualification_job.GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_PLAN_SHA256
    )
    assert plan["runtime_contract"]["artifact_sha256"]["runner_sha256"] == (
        qualification_job.GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_RUNNER_SHA256
    )
    assert manifest["closed_record_sha256"] == (
        qualification_job.GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_MANIFEST_SHA256
    )
    assert manifest["reason"] == (
        qualification_job.GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_REASON
    )
    assert qualification_job._file_sha256(
        evidence_root / "reconciliation-manifest.json"
    ) == (
        qualification_job.GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_MANIFEST_FILE_SHA256
    )
    first_output = json.loads(
        next(evidence_root.glob("*.runs-get-output.json")).read_text()
    )
    assert first_output["error"] == (
        qualification_job.GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_ERROR
    )
    assert (
        len(reconciled.reservations),
        len(reconciled.submission_receipts),
        len(reconciled.terminal_actuals),
    ) == (180, 42, 180)
    assert reconciled.active_reserved_cluster_hours == 0.0
    assert databricks_ledger_prefix(reconciled).prefix_sha256 == (
        qualification_job.GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_TERMINAL_PREFIX_SHA256
    )
    contracts = qualification_job._validated_qualification_payloads(plan, payloads)
    expected_attempt_order = [
        contract["reservation_attempt_id"]
        for contract in sorted(contracts, key=lambda item: item["job_id"])
    ]
    assert [item.attempt_id for item in reconciled.terminal_actuals[-14:]] == (
        expected_attempt_order
    )
    assert sum(
        item.actual_cluster_duration_seconds
        for item in reconciled.terminal_actuals[-14:]
    ) == pytest.approx(4564.259)
    assert (
        "reconcile_gpu_qualification_bootstrap_cluster_identity_failure_evidence"
        in qualification_job.__all__
    )
    assert not any(
        name.startswith("expected_")
        for name in inspect.signature(
            qualification_job.reconcile_gpu_qualification_bootstrap_cluster_identity_failure_evidence
        ).parameters
    )

    closed_ledger_bytes = ledger_path.read_bytes()
    assert (
        _reconcile_cluster_identity_failure(
            plan_root=plan_root,
            evidence_root=evidence_root,
            plan=plan,
            payloads=payloads,
            ledger_path=ledger_path,
        )
        == reconciled
    )
    assert ledger_path.read_bytes() == closed_ledger_bytes
    assert _RETAINED_LEDGER_PATH.read_bytes() == live_ledger_before
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in evidence_root.iterdir()
    } == evidence_before


def test_cluster_identity_reconciliation_resumes_canonical_partial_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        plan_root,
        evidence_root,
        plan,
        payloads,
        _manifest,
        ledger_path,
    ) = _cluster_identity_failure_replay_fixture(tmp_path, monkeypatch)
    real_record = qualification_job.record_databricks_verified_run_terminal_actual_json
    completed = 0

    def interrupt_after_five(ledger_path_arg, *, attempt_id, run_record):
        nonlocal completed
        if completed == 5:
            raise RuntimeError("simulated reconciliation interruption")
        updated = real_record(
            ledger_path_arg,
            attempt_id=attempt_id,
            run_record=run_record,
        )
        completed += 1
        return updated

    monkeypatch.setattr(
        qualification_job,
        "record_databricks_verified_run_terminal_actual_json",
        interrupt_after_five,
    )
    with pytest.raises(RuntimeError, match="simulated reconciliation interruption"):
        _reconcile_cluster_identity_failure(
            plan_root=plan_root,
            evidence_root=evidence_root,
            plan=plan,
            payloads=payloads,
            ledger_path=ledger_path,
        )
    partial = read_databricks_cluster_hour_ledger_json(ledger_path)
    assert len(partial.terminal_actuals) == 171
    contracts = qualification_job._validated_qualification_payloads(plan, payloads)
    expected_attempt_order = [
        contract["reservation_attempt_id"]
        for contract in sorted(contracts, key=lambda item: item["job_id"])
    ]
    assert [item.attempt_id for item in partial.terminal_actuals[-5:]] == (
        expected_attempt_order[:5]
    )

    monkeypatch.setattr(
        qualification_job,
        "record_databricks_verified_run_terminal_actual_json",
        real_record,
    )
    reconciled = _reconcile_cluster_identity_failure(
        plan_root=plan_root,
        evidence_root=evidence_root,
        plan=plan,
        payloads=payloads,
        ledger_path=ledger_path,
    )
    assert [item.attempt_id for item in reconciled.terminal_actuals[-14:]] == (
        expected_attempt_order
    )
    assert databricks_ledger_prefix(reconciled).prefix_sha256 == (
        qualification_job.GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_TERMINAL_PREFIX_SHA256
    )
    assert reconciled.active_reserved_cluster_hours == 0.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra-file", "exact batch closure"),
        ("legacy-output-shape", "reviewed incident schema"),
    ],
)
def test_cluster_identity_reconciliation_rejects_inexact_closure_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
):
    (
        plan_root,
        source_evidence_root,
        plan,
        payloads,
        _manifest,
        ledger_path,
    ) = _cluster_identity_failure_replay_fixture(tmp_path, monkeypatch)
    evidence_root = tmp_path / "evidence-copy"
    shutil.copytree(source_evidence_root, evidence_root)
    if mutation == "extra-file":
        (evidence_root / "unreviewed.json").write_text("{}\n", encoding="utf-8")
    else:
        output_path = next(evidence_root.glob("*.runs-get-output.json"))
        output = json.loads(output_path.read_text())
        output.pop("logs")
        output.pop("logs_truncated")
        output_path.write_text(
            canonical_gpu_qualification_json(output) + "\n",
            encoding="utf-8",
        )
    ledger_before = ledger_path.read_bytes()

    with pytest.raises(ValueError, match=message):
        _reconcile_cluster_identity_failure(
            plan_root=plan_root,
            evidence_root=evidence_root,
            plan=plan,
            payloads=payloads,
            ledger_path=ledger_path,
        )
    assert ledger_path.read_bytes() == ledger_before


def _runtime_lock_index_failure_replay_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan_root = (
        Path(__file__).parents[1]
        / "databricks-runs/vllm-0271-publication-prep/"
        "gpu-qualification-plan-sha256-"
        "f991036176d59df70f0e339be4eb4a67a7c03a51536f62bf440df1ac72fd0e33"
    ).resolve()
    evidence_root = plan_root / "failed-attempt-runtime-lock-index-v2"
    plan = json.loads((plan_root / "gpu-qualification-plan.json").read_text())
    payloads = json.loads((plan_root / "submit-payloads.json").read_text())
    manifest = json.loads(
        (evidence_root / "reconciliation-manifest.json").read_text()
    )
    retained = read_databricks_cluster_hour_ledger_json(_RETAINED_LEDGER_PATH)
    active_incident = replace(
        retained,
        reservations=retained.reservations[:194],
        submission_receipts=retained.submission_receipts[:56],
        terminal_actuals=retained.terminal_actuals[:180],
    )
    ledger_path = tmp_path / "cluster-hours.json"
    ledger_path.write_text(
        json.dumps(
            databricks_cluster_hour_ledger_to_record(active_incident),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        qualification_job,
        "databricks_ledger_path_sha256",
        lambda _path: plan["campaign_ledger_path_sha256"],
    )
    monkeypatch.setattr(
        resource_ledger,
        "databricks_ledger_path_sha256",
        lambda _path: plan["campaign_ledger_path_sha256"],
    )
    return plan_root, evidence_root, plan, payloads, manifest, ledger_path


def _reconcile_runtime_lock_index_failure(
    *,
    plan_root: Path,
    evidence_root: Path,
    plan: dict[str, Any],
    payloads: list[dict[str, Any]],
    ledger_path: Path,
) -> DatabricksClusterHourLedger:
    return qualification_job.reconcile_gpu_qualification_runtime_lock_index_failure_evidence(
        plan_record=plan,
        submit_payloads=payloads,
        ledger_path=ledger_path,
        submit_receipt_root=plan_root / "submit-receipts",
        local_preflight_evidence_path=(
            plan_root / "local-preflight-valid/local-preflight-evidence.json"
        ),
        runs_get_evidence_root=evidence_root,
    )


def test_runtime_lock_index_failure_source_pins_normalize_exact_retained_outputs():
    plan_root = (
        Path(__file__).parents[1]
        / "databricks-runs/vllm-0271-publication-prep/"
        "gpu-qualification-plan-sha256-"
        "f991036176d59df70f0e339be4eb4a67a7c03a51536f62bf440df1ac72fd0e33"
    ).resolve()
    evidence_root = plan_root / "failed-attempt-runtime-lock-index-v2"
    plan = json.loads((plan_root / "gpu-qualification-plan.json").read_text())
    payloads = json.loads((plan_root / "submit-payloads.json").read_text())
    contracts = qualification_job._validated_qualification_payloads(plan, payloads)
    expected_error_sha256 = dict(
        qualification_job.GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_ERROR_SHA256_BY_JOB
    )

    assert len(expected_error_sha256) == 14
    assert set(expected_error_sha256) == {
        str(contract["job_id"]) for contract in contracts
    }
    normalized_errors = set()
    for contract in contracts:
        job_id = str(contract["job_id"])
        run_output = json.loads(
            (evidence_root / f"{job_id}.runs-get-output.json").read_text()
        )
        error = run_output["error"]
        assert hashlib.sha256(error.encode("utf-8")).hexdigest() == (
            expected_error_sha256[job_id]
        )
        assert (
            qualification_job._validated_runtime_lock_index_failure_error(
                run_output,
                plan_sha256=plan["closed_record_sha256"],
                job_id=job_id,
                expected_error_sha256=expected_error_sha256[job_id],
            )
            == error
        )
        normalized_errors.add(
            qualification_job._normalize_runtime_lock_index_failure_error(
                error,
                plan_sha256=plan["closed_record_sha256"],
                job_id=job_id,
            )
        )

    assert normalized_errors == {
        qualification_job._RUNTIME_LOCK_INDEX_FAILURE_NORMALIZED_ERROR
    }
    normalized = next(iter(normalized_errors))
    assert hashlib.sha256(normalized.encode("utf-8")).hexdigest() == (
        qualification_job.GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_NORMALIZED_ERROR_SHA256
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("raw-sha", "raw error is not reviewed"),
        ("runtime-path", "runtime Python path once"),
        ("uuid", "UUIDv4 lock path"),
        ("argv", "argv grammar differs"),
        ("marker-missing", "resolution marker once"),
        ("marker-duplicate", "resolution marker once"),
        ("truncated", "logs must be complete"),
        ("extra-schema", "open schema"),
    ],
)
def test_runtime_lock_index_failure_validator_rejects_adversarial_variants(
    mutation: str,
    message: str,
):
    plan_sha256 = (
        qualification_job.GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_PLAN_SHA256
    )
    evidence_root = (
        Path(__file__).parents[1]
        / "databricks-runs/vllm-0271-publication-prep/"
        f"gpu-qualification-plan-sha256-{plan_sha256}/"
        "failed-attempt-runtime-lock-index-v2"
    ).resolve()
    output_path = sorted(evidence_root.glob("*.runs-get-output.json"))[0]
    job_id = output_path.name.removesuffix(".runs-get-output.json")
    run_output = json.loads(output_path.read_text())
    marker = (
        qualification_job.GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_LOG_MARKER
    )
    expected_error_sha256 = hashlib.sha256(
        run_output["error"].encode("utf-8")
    ).hexdigest()
    if mutation == "raw-sha":
        expected_error_sha256 = "0" * 64
    elif mutation == "runtime-path":
        run_output["error"] = run_output["error"].replace(
            f"/{job_id}/runtime/bin/python",
            "/foreign-job/runtime/bin/python",
        )
        expected_error_sha256 = hashlib.sha256(
            run_output["error"].encode("utf-8")
        ).hexdigest()
    elif mutation == "uuid":
        run_output["error"] = run_output["error"].replace(
            "pythonEnv-359ea7ae-1c5f-473c-89df-3f693b82d1cd",
            "pythonEnv-359ea7ae-1c5f-573c-89df-3f693b82d1cd",
        )
        expected_error_sha256 = hashlib.sha256(
            run_output["error"].encode("utf-8")
        ).hexdigest()
    elif mutation == "argv":
        run_output["error"] = run_output["error"].replace(
            "'--require-hashes'",
            "'--no-deps'",
        )
        expected_error_sha256 = hashlib.sha256(
            run_output["error"].encode("utf-8")
        ).hexdigest()
    elif mutation == "marker-missing":
        run_output["logs"] = run_output["logs"].replace(marker, "unreviewed")
    elif mutation == "marker-duplicate":
        run_output["logs"] += marker
    elif mutation == "truncated":
        run_output["logs_truncated"] = True
    else:
        run_output["unreviewed"] = "field"

    with pytest.raises(ValueError, match=message):
        qualification_job._validated_runtime_lock_index_failure_error(
            run_output,
            plan_sha256=plan_sha256,
            job_id=job_id,
            expected_error_sha256=expected_error_sha256,
        )


def test_runtime_lock_index_wrapper_is_deterministic_idempotent_and_source_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        plan_root,
        evidence_root,
        plan,
        payloads,
        manifest,
        ledger_path,
    ) = _runtime_lock_index_failure_replay_fixture(tmp_path, monkeypatch)
    second_ledger_path = tmp_path / "cluster-hours-second.json"
    second_ledger_path.write_bytes(ledger_path.read_bytes())
    live_ledger_before = _RETAINED_LEDGER_PATH.read_bytes()
    evidence_before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in evidence_root.iterdir()
    }

    reconciled = _reconcile_runtime_lock_index_failure(
        plan_root=plan_root,
        evidence_root=evidence_root,
        plan=plan,
        payloads=payloads,
        ledger_path=ledger_path,
    )
    second = _reconcile_runtime_lock_index_failure(
        plan_root=plan_root,
        evidence_root=evidence_root,
        plan=plan,
        payloads=payloads,
        ledger_path=second_ledger_path,
    )

    assert reconciled == second
    assert ledger_path.read_bytes() == second_ledger_path.read_bytes()
    assert hashlib.sha256(ledger_path.read_bytes()).hexdigest() == (
        "1ac7ee076d2a5aa3b12bfd18d3cb6f8843aa9f8f7b8e07686c519869985a6916"
    )
    assert plan["closed_record_sha256"] == (
        qualification_job.GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_PLAN_SHA256
    )
    assert plan["runtime_contract"]["artifact_sha256"]["runner_sha256"] == (
        qualification_job.GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_RUNNER_SHA256
    )
    assert manifest["closed_record_sha256"] == (
        qualification_job.GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_MANIFEST_SHA256
    )
    assert manifest["reason"] == (
        qualification_job.GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_REASON
    )
    assert qualification_job._file_sha256(
        evidence_root / "reconciliation-manifest.json"
    ) == (
        qualification_job.GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_MANIFEST_FILE_SHA256
    )
    assert (
        len(reconciled.reservations),
        len(reconciled.submission_receipts),
        len(reconciled.terminal_actuals),
    ) == (194, 56, 194)
    assert reconciled.active_reserved_task_count == 0
    assert reconciled.active_reserved_cluster_hours == 0.0
    assert reconciled.terminal_actual_cluster_hours == pytest.approx(
        61.28905027777782
    )
    assert sum(
        item.actual_cluster_duration_seconds
        for item in reconciled.terminal_actuals[-14:]
    ) == pytest.approx(7754.755)
    assert databricks_ledger_prefix(reconciled).prefix_sha256 == (
        qualification_job.GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_TERMINAL_PREFIX_SHA256
    )
    contracts = qualification_job._validated_qualification_payloads(plan, payloads)
    expected_attempt_order = [
        contract["reservation_attempt_id"]
        for contract in sorted(contracts, key=lambda item: item["job_id"])
    ]
    assert [item.attempt_id for item in reconciled.terminal_actuals[-14:]] == (
        expected_attempt_order
    )
    assert (
        "reconcile_gpu_qualification_runtime_lock_index_failure_evidence"
        in qualification_job.__all__
    )
    assert not any(
        name.startswith("expected_")
        for name in inspect.signature(
            qualification_job.reconcile_gpu_qualification_runtime_lock_index_failure_evidence
        ).parameters
    )

    closed_bytes = ledger_path.read_bytes()
    assert (
        _reconcile_runtime_lock_index_failure(
            plan_root=plan_root,
            evidence_root=evidence_root,
            plan=plan,
            payloads=payloads,
            ledger_path=ledger_path,
        )
        == reconciled
    )
    assert ledger_path.read_bytes() == closed_bytes
    assert _RETAINED_LEDGER_PATH.read_bytes() == live_ledger_before
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in evidence_root.iterdir()
    } == evidence_before


def test_runtime_lock_index_reconciliation_resumes_canonical_partial_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        plan_root,
        evidence_root,
        plan,
        payloads,
        _manifest,
        ledger_path,
    ) = _runtime_lock_index_failure_replay_fixture(tmp_path, monkeypatch)
    clean_ledger_path = tmp_path / "cluster-hours-clean.json"
    clean_ledger_path.write_bytes(ledger_path.read_bytes())
    real_record = qualification_job.record_databricks_verified_run_terminal_actual_json
    completed = 0

    def interrupt_after_five(ledger_path_arg, *, attempt_id, run_record):
        nonlocal completed
        if completed == 5:
            raise RuntimeError("simulated runtime-index reconciliation interruption")
        updated = real_record(
            ledger_path_arg,
            attempt_id=attempt_id,
            run_record=run_record,
        )
        completed += 1
        return updated

    monkeypatch.setattr(
        qualification_job,
        "record_databricks_verified_run_terminal_actual_json",
        interrupt_after_five,
    )
    with pytest.raises(RuntimeError, match="simulated runtime-index"):
        _reconcile_runtime_lock_index_failure(
            plan_root=plan_root,
            evidence_root=evidence_root,
            plan=plan,
            payloads=payloads,
            ledger_path=ledger_path,
        )
    partial = read_databricks_cluster_hour_ledger_json(ledger_path)
    assert len(partial.terminal_actuals) == 185
    contracts = qualification_job._validated_qualification_payloads(plan, payloads)
    expected_attempt_order = [
        contract["reservation_attempt_id"]
        for contract in sorted(contracts, key=lambda item: item["job_id"])
    ]
    assert [item.attempt_id for item in partial.terminal_actuals[-5:]] == (
        expected_attempt_order[:5]
    )

    monkeypatch.setattr(
        qualification_job,
        "record_databricks_verified_run_terminal_actual_json",
        real_record,
    )
    resumed = _reconcile_runtime_lock_index_failure(
        plan_root=plan_root,
        evidence_root=evidence_root,
        plan=plan,
        payloads=payloads,
        ledger_path=ledger_path,
    )
    clean = _reconcile_runtime_lock_index_failure(
        plan_root=plan_root,
        evidence_root=evidence_root,
        plan=plan,
        payloads=payloads,
        ledger_path=clean_ledger_path,
    )
    assert resumed == clean
    assert ledger_path.read_bytes() == clean_ledger_path.read_bytes()
    assert databricks_ledger_prefix(resumed).prefix_sha256 == (
        qualification_job.GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_TERMINAL_PREFIX_SHA256
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra-file", "exact batch closure"),
        ("raw-error", "raw error is not reviewed"),
        ("marker", "resolution marker once"),
        ("truncated", "logs must be complete"),
        ("extra-schema", "reviewed incident schema"),
        ("manifest", "manifest file is not reviewed"),
    ],
)
def test_runtime_lock_index_reconciliation_rejects_tamper_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
):
    (
        plan_root,
        source_evidence_root,
        plan,
        payloads,
        _manifest,
        ledger_path,
    ) = _runtime_lock_index_failure_replay_fixture(tmp_path, monkeypatch)
    evidence_root = tmp_path / "runtime-index-evidence-copy"
    shutil.copytree(source_evidence_root, evidence_root)
    if mutation == "extra-file":
        (evidence_root / "unreviewed.json").write_text("{}\n", encoding="utf-8")
    elif mutation == "manifest":
        manifest_path = evidence_root / "reconciliation-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["reason"] = "unreviewed"
        manifest_path.write_text(
            canonical_gpu_qualification_json(manifest) + "\n",
            encoding="utf-8",
        )
    else:
        output_path = sorted(evidence_root.glob("*.runs-get-output.json"))[0]
        output = json.loads(output_path.read_text())
        if mutation == "raw-error":
            output["error"] += " tampered"
        elif mutation == "marker":
            output["logs"] = output["logs"].replace(
                qualification_job.GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_LOG_MARKER,
                "unreviewed",
            )
        elif mutation == "truncated":
            output["logs_truncated"] = True
        else:
            output["unreviewed"] = "field"
        output_path.write_text(
            canonical_gpu_qualification_json(output) + "\n",
            encoding="utf-8",
        )
    ledger_before = ledger_path.read_bytes()

    with pytest.raises(ValueError, match=message):
        _reconcile_runtime_lock_index_failure(
            plan_root=plan_root,
            evidence_root=evidence_root,
            plan=plan,
            payloads=payloads,
            ledger_path=ledger_path,
        )
    assert ledger_path.read_bytes() == ledger_before


def test_v2_failed_capture_publication_removes_partial_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "failed-attempt-v2"
    contracts = ({"job_id": "job-a"}, {"job_id": "job-b"})
    parent_runs = ({"run_id": 1}, {"run_id": 2})
    run_outputs = (
        {"error": "failed-a", "metadata": {}},
        {"error": "failed-b", "metadata": {}},
    )
    real_write = qualification_job._write_canonical_exclusive
    calls = 0

    def fail_second(record, path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated failed-attempt evidence write failure")
        real_write(record, path)

    monkeypatch.setattr(qualification_job, "_write_canonical_exclusive", fail_second)
    with pytest.raises(OSError, match="simulated failed-attempt"):
        qualification_job._publish_failed_attempt_evidence_atomic(
            root,
            contracts=contracts,
            parent_runs=parent_runs,
            run_outputs=run_outputs,
            manifest={"closed_record_sha256": "a" * 64},
        )
    assert not root.exists()
    assert not list(tmp_path.glob(".failed-attempt-v2.staging-*"))


def test_launch_capability_cannot_be_constructed_by_record_only_callers():
    selection = GPUQualificationSelection(
        attention_backend="TRITON_ATTN",
        gpu_memory_utilization=0.75,
        generation_hardware_id="aws-g6e-l40s",
        generation_databricks_node_type_id="g6e.4xlarge",
        generation_artifacts_sha256="a" * 64,
        generation_prefix_tokens_per_second=40.0,
        plan_sha256="b" * 64,
    )

    predecessor = databricks_ledger_prefix(
        DatabricksClusterHourLedger(ledger_id="ledger")
    )
    batch = DatabricksLedgerPrefix(
        ledger_id="ledger",
        cap_cluster_hours=1024.0,
        reservation_count=14,
        submission_receipt_count=0,
        terminal_actual_count=0,
        prefix_sha256="1" * 64,
    )
    terminal = DatabricksLedgerPrefix(
        ledger_id="ledger",
        cap_cluster_hours=1024.0,
        reservation_count=14,
        submission_receipt_count=14,
        terminal_actual_count=14,
        prefix_sha256="2" * 64,
    )
    with pytest.raises(TypeError, match="live collection"):
        GPUQualificationLaunchAuthorization(
            selection=selection,
            plan_sha256="b" * 64,
            evidence_closed_record_sha256="c" * 64,
            evidence_file_sha256="d" * 64,
            ledger_id="ledger",
            ledger_path_sha256="f" * 64,
            predecessor_prefix=predecessor,
            producer_batch_prefix=batch,
            ledger_prefix=terminal,
            causal_closure_sha256="e" * 64,
            _issuer=object(),
        )


def test_collector_closes_all_fourteen_direct_terminal_runs_and_ledger_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan = _plan()
    payloads = _render(plan, _artifact_uris())
    ledger_path = tmp_path / "cluster-hours.json"
    submit_root = tmp_path / "submit-receipts"
    terminal_root = tmp_path / "terminal-receipts"
    evidence_path = tmp_path / "qualification-evidence.json"
    _copy_retained_campaign_ledger(ledger_path, monkeypatch)
    plan = _plan()
    payloads = _render(plan, _artifact_uris())
    run_ids = [30_000 + index for index in range(14)]
    local_preflight_path = _write_local_preflight(
        plan, tmp_path / "local-preflight.json"
    )
    submit_gpu_qualification_jobs(
        DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
        plan_record=plan,
        submit_payloads=payloads,
        ledger_path=ledger_path,
        submit_receipt_root=submit_root,
        local_preflight_evidence_path=local_preflight_path,
        opener=_SequentialOpener([{"run_id": run_id} for run_id in run_ids]),
        now=lambda: datetime(2026, 8, 24, 0, 0, tzinfo=UTC),
    )

    dbfs_root = tmp_path / "dbfs"
    original_cluster_file_path = qualification_job._cluster_file_path

    def cluster_file_path(value: str | Path) -> Path:
        raw = str(value)
        if raw.startswith("dbfs:/"):
            return dbfs_root / raw.removeprefix("dbfs:/")
        return original_cluster_file_path(value)

    monkeypatch.setattr(qualification_job, "_cluster_file_path", cluster_file_path)
    contracts = qualification_job._validated_qualification_payloads(plan, payloads)
    runs: dict[str, dict[str, Any]] = {}
    for index, (planned_job, payload, contract, run_id) in enumerate(
        zip(
            plan["cloud_qualification"]["jobs"],
            payloads,
            contracts,
            run_ids,
            strict=True,
        )
    ):
        task = payload["tasks"][0]
        run_start = 1_787_533_140_000 + index * 10_000
        task_start = run_start + 1_000
        task_end = task_start + 5_000
        runs[str(run_id)] = {
            "end_time": task_end + 1_000,
            "run_id": run_id,
            "run_name": payload["run_name"],
            "run_type": "SUBMIT_RUN",
            "start_time": run_start,
            "state": {
                "life_cycle_state": "TERMINATED",
                "result_state": "SUCCESS",
            },
            "tasks": [
                {
                    "attempt_number": 0,
                    "cluster_instance": {"cluster_id": f"cluster-{run_id}"},
                    "end_time": task_end,
                    "new_cluster": task["new_cluster"],
                    "run_id": 40_000 + index,
                    "start_time": task_start,
                    "state": {
                        "life_cycle_state": "TERMINATED",
                        "result_state": "SUCCESS",
                    },
                    "task_key": task["task_key"],
                }
            ],
        }
        result = {
            "closed_record_sha256": "",
            "cloud_cluster_id": f"cluster-{run_id}",
            "cloud_run_id": str(run_id),
            "job_id": planned_job["job_id"],
            "measurements": (
                {
                    "candidate_qualified": True,
                    "gpu_memory_utilization": 0.70,
                }
                if planned_job["job_id"].startswith("aws-g6-l4-32k-c4-gmu-")
                else {}
            ),
            "output_json": contract["output_json"],
            "reservation_attempt_id": contract["reservation_attempt_id"],
            "task_key": contract["task_key"],
        }
        qualification_job._seal_record(result)
        result_path = cluster_file_path(str(contract["output_json"]))
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            canonical_gpu_qualification_json(result) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        qualification_job,
        "get_databricks_run",
        lambda config, run_id: runs[run_id],
    )
    monkeypatch.setattr(
        qualification_job,
        "validate_gpu_job_result_record",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        qualification_job,
        "_build_governed_cloud_gpu_evidence",
        lambda **kwargs: {
            "terminal_receipts": list(kwargs["terminal_receipts"]),
        },
    )

    def build_evidence(**kwargs):
        record = {"closed_record_sha256": "", **kwargs}
        qualification_job._seal_record(record)
        return record

    selection = GPUQualificationSelection(
        attention_backend="TRITON_ATTN",
        gpu_memory_utilization=0.70,
        generation_hardware_id="aws-g6e-l40s",
        generation_databricks_node_type_id="g6e.8xlarge",
        generation_artifacts_sha256="a" * 64,
        generation_prefix_tokens_per_second=40.0,
        plan_sha256=plan["closed_record_sha256"],
    )
    monkeypatch.setattr(
        qualification_job,
        "_build_governed_gpu_qualification_evidence",
        build_evidence,
    )
    monkeypatch.setattr(
        qualification_job,
        "validate_gpu_qualification_evidence_record",
        lambda *args, **kwargs: selection,
    )

    _evidence, authorization = collect_gpu_qualification_evidence(
        DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
        plan_record=plan,
        submit_payloads=payloads,
        ledger_path=ledger_path,
        submit_receipt_root=submit_root,
        local_preflight_evidence_path=local_preflight_path,
        terminal_receipt_root=terminal_root,
        evidence_output_json=evidence_path,
        now=lambda: datetime(2026, 8, 24, 2, 2, tzinfo=UTC),
    )

    assert isinstance(authorization, GPUQualificationLaunchAuthorization)
    observed_selection = require_gpu_qualification_launch_authorization(
        authorization,
        expected_plan_sha256=plan["closed_record_sha256"],
        expected_evidence_file_sha256=qualification_job._file_sha256(evidence_path),
    )
    assert observed_selection == selection
    assert evidence_path.is_file()
    assert len(list(terminal_root.glob("*.json"))) == 14
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    assert len(ledger.terminal_actuals) == (
        PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX.terminal_actual_count + 14
    )
    assert ledger.active_reserved_cluster_hours == 0.0
    qualification_attempt_ids = {
        contract["reservation_attempt_id"]
        for contract in qualification_job._validated_qualification_payloads(
            plan, payloads
        )
    }
    assert all(
        item.verification_source == "direct_databricks_runs_get"
        for item in ledger.terminal_actuals
        if item.attempt_id in qualification_attempt_ids
    )
    for receipt_path in terminal_root.glob("*.json"):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["ledger_id"] == CAMPAIGN_LEDGER_ID
        assert receipt["ledger_actual_cluster_duration_seconds"] == 5.0
        assert len(receipt["ledger_terminal_actual_sha256"]) == 64


def test_collector_reconciles_a_never_started_failure_before_rejecting_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan = _plan()
    payloads = _render(plan, _artifact_uris())
    ledger_path = tmp_path / "cluster-hours.json"
    submit_root = tmp_path / "submit-receipts"
    _copy_retained_campaign_ledger(ledger_path, monkeypatch)
    plan = _plan()
    payloads = _render(plan, _artifact_uris())
    run_ids = [50_000 + index for index in range(14)]
    local_preflight_path = _write_local_preflight(
        plan, tmp_path / "local-preflight.json"
    )
    submit_gpu_qualification_jobs(
        DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
        plan_record=plan,
        submit_payloads=payloads,
        ledger_path=ledger_path,
        submit_receipt_root=submit_root,
        local_preflight_evidence_path=local_preflight_path,
        opener=_SequentialOpener([{"run_id": run_id} for run_id in run_ids]),
    )
    first_task = payloads[0]["tasks"][0]
    failed_run = {
        "end_time": 1_000_001,
        "run_id": run_ids[0],
        "run_name": payloads[0]["run_name"],
        "run_type": "SUBMIT_RUN",
        "start_time": 1_000_000,
        "state": {
            "life_cycle_state": "TERMINATED",
            "result_state": "FAILED",
        },
        "tasks": [
            {
                "attempt_number": 0,
                "cluster_instance": {"cluster_id": "never-started-cluster"},
                "new_cluster": first_task["new_cluster"],
                "run_id": 60_000,
                "state": {
                    "life_cycle_state": "TERMINATED",
                    "result_state": "FAILED",
                },
                "task_key": first_task["task_key"],
            }
        ],
    }
    monkeypatch.setattr(
        qualification_job,
        "get_databricks_run",
        lambda config, run_id: failed_run,
    )
    terminal_root = tmp_path / "terminal-receipts"
    evidence_path = tmp_path / "qualification-evidence.json"

    with pytest.raises(ValueError, match="task.start_time"):
        collect_gpu_qualification_evidence(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            plan_record=plan,
            submit_payloads=payloads,
            ledger_path=ledger_path,
            submit_receipt_root=submit_root,
            local_preflight_evidence_path=local_preflight_path,
            terminal_receipt_root=terminal_root,
            evidence_output_json=evidence_path,
        )

    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    assert len(ledger.terminal_actuals) == (
        PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX.terminal_actual_count + 1
    )
    failed_actual = next(
        item
        for item in ledger.terminal_actuals
        if item.attempt_id
        == qualification_job._validated_qualification_payloads(plan, payloads)[0][
            "reservation_attempt_id"
        ]
    )
    assert failed_actual.terminal_state == "failed"
    assert failed_actual.actual_cluster_duration_seconds == 0.0
    assert not terminal_root.exists()
    assert not evidence_path.exists()


def test_qualifier_selects_minimum_example_id_from_each_32_row_shard(
    tmp_path: Path,
):
    path = tmp_path / "hotpotqa.jsonl"
    records = [
        {"dataset": "hotpotqa", "example_id": f"hotpotqa-{index:02d}"}
        for index in reversed(range(32))
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    selected = sentinel_worker._selected_jsonl_record(path)

    assert selected["example_id"] == "hotpotqa-00"


def test_bootstrap_writer_publishes_exact_content_addressed_stdlib_script(
    tmp_path: Path,
):
    destination = tmp_path / "gpu-qualification-bootstrap.py"

    assert GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256 == (
        "ca93baeda09f3df050b0dad3b8f3091c0f74235c426bd66555b67bd4b6eeafbc"
    )
    assert _pins().runner_sha256 == GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256
    observed = write_gpu_qualification_bootstrap_runner(destination)

    assert observed == GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256
    assert destination.read_text(encoding="utf-8") == (
        GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SCRIPT
    )
    assert _digest(destination.read_bytes()) == observed
    assert GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SCRIPT.index(
        "subprocess.run("
    ) < GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SCRIPT.index(
        "from document_kv_cache.gpu_qualification_databricks import main"
    )
    assert "transformers" not in GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SCRIPT.lower()
    with pytest.raises(FileExistsError):
        write_gpu_qualification_bootstrap_runner(destination)


def test_emitted_bootstrap_and_worker_resolve_uc_volumes_at_official_mount(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    namespace: dict[str, Any] = {"__name__": "gpuq_bootstrap_test"}
    exec(GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SCRIPT, namespace)

    assert namespace["_cluster_path"](
        "dbfs:/Volumes/catalog/schema/volume/package.whl"
    ) == "/Volumes/catalog/schema/volume/package.whl"
    assert namespace["_cluster_path"]("dbfs:/legacy/package.whl") == (
        "/dbfs/legacy/package.whl"
    )
    assert qualification_job._cluster_file_path(
        "dbfs:/Volumes/catalog/schema/volume/result.json"
    ) == Path("/Volumes/catalog/schema/volume/result.json")
    assert qualification_job._cluster_file_path("dbfs:/legacy/result.json") == Path(
        "/dbfs/legacy/result.json"
    )
    monkeypatch.setenv("PIP_CONFIG_FILE", "/attacker/pip.conf")
    monkeypatch.setenv("PIP_INDEX_URL", "https://attacker.invalid/simple")
    monkeypatch.setenv("pip_no_index", "1")
    monkeypatch.setenv("PIP_REQUIREMENT", "/attacker/requirements.txt")
    monkeypatch.setenv("PYTHONHOME", "/attacker/python-home")
    monkeypatch.setenv("PYTHONPATH", "/attacker/python-path")
    monkeypatch.setenv("VIRTUAL_ENV", "/attacker/venv")
    environments = (
        namespace["_pip_subprocess_environment"](),
        qualification_sentinels._pip_subprocess_environment(),
        sentinel_worker._pip_subprocess_environment(),
    )
    for environment in environments:
        assert {
            key for key in environment if key.upper().startswith("PIP_")
        } == {
            "PIP_CONFIG_FILE",
            "PIP_DISABLE_PIP_VERSION_CHECK",
            "PIP_NO_INPUT",
        }
        assert environment["PIP_CONFIG_FILE"] == os.devnull
        assert environment["PYTHONNOUSERSITE"] == "1"
        assert all(
            variable not in environment
            for variable in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")
        )
    site_packages = tmp_path / "runtime" / "site-packages"
    site_packages.mkdir(parents=True)
    discovery_calls: list[tuple[list[str], dict[str, Any]]] = []

    def discover_site_packages(argv, **kwargs):
        discovery_calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps([str(site_packages)]),
            stderr="",
        )

    monkeypatch.setattr(
        qualification_sentinels.subprocess,
        "run",
        discover_site_packages,
    )
    qualification_sentinels._make_site_packages_read_only(tmp_path / "python")
    assert len(discovery_calls) == 1
    assert discovery_calls[0][1]["env"] == environments[1]


def test_emitted_bootstrap_verifies_its_compiled_path_without_file_global(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner_path = tmp_path / "gpu-qualification-bootstrap.py"
    runner_path.write_text(
        GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SCRIPT,
        encoding="utf-8",
    )
    package_path = tmp_path / "cachet.whl"
    package_path.write_bytes(b"reviewed package bytes")
    namespace: dict[str, Any] = {"__name__": "gpuq_bootstrap_test"}
    exec(
        compile(
            GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SCRIPT,
            str(runner_path),
            "exec",
        ),
        namespace,
    )
    assert "__file__" not in namespace
    calls: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(
        namespace["subprocess"],
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    monkeypatch.setenv("PIP_CONFIG_FILE", "/attacker/pip.conf")
    monkeypatch.setenv("PIP_INDEX_URL", "https://attacker.invalid/simple")
    monkeypatch.setenv("pip_no_index", "1")
    monkeypatch.setenv("PIP_REQUIREMENT", "/attacker/requirements.txt")
    monkeypatch.setenv("PYTHONHOME", "/attacker/python-home")
    monkeypatch.setenv("PYTHONPATH", "/attacker/python-path")
    monkeypatch.setenv("VIRTUAL_ENV", "/attacker/venv")
    pins = {
        "cachet_source_tree_sha256": "1" * 64,
        "input_bundle_sha256": "2" * 64,
        "package_wheel_sha256": _digest(package_path.read_bytes()),
        "patched_vllm_wheel_sha256": "3" * 64,
        "runner_sha256": GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256,
        "runtime_lock_sha256": "4" * 64,
    }
    argv = ["--package-wheel-uri", package_path.as_uri()]
    for key, digest in pins.items():
        argv.extend(("--artifact-sha256", f"{key}={digest}"))

    bootstrap = namespace["_bootstrap"]
    assert bootstrap(argv) == argv
    assert len(calls) == 1
    assert calls[0][0][-1] == str(package_path)
    environment = calls[0][1]["env"]
    assert {
        key for key in environment if key.upper().startswith("PIP_")
    } == {
        "PIP_CONFIG_FILE",
        "PIP_DISABLE_PIP_VERSION_CHECK",
        "PIP_NO_INPUT",
    }
    assert environment["PIP_CONFIG_FILE"] == os.devnull
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert all(
        variable not in environment
        for variable in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")
    )

    reviewed_copy = tmp_path / "reviewed-copy.py"
    reviewed_copy.write_text(
        GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SCRIPT,
        encoding="utf-8",
    )
    replacement_namespace: dict[str, Any] = {}
    exec(
        compile("def replacement(): pass\n", str(reviewed_copy), "exec"),
        replacement_namespace,
    )
    namespace["_bootstrap"] = replacement_namespace["replacement"]
    runner_path.write_text("# tampered after compilation\n", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="GPU qualification bootstrap runner SHA-256 mismatch",
    ):
        bootstrap(argv)


def test_uc_volume_artifact_and_output_resolvers_preserve_uri_and_mount_identity():
    plan = _plan()
    job = plan["cloud_qualification"]["jobs"][0]
    root = "dbfs:/Volumes/catalog/schema/volume/results"
    output = (
        f"{root}/{plan['closed_record_sha256']}/{job['job_id']}/"
        "gpu-job-result.json"
    )

    assert qualification_job._validated_cluster_artifact_uri(
        "dbfs:/Volumes/catalog/schema/volume/package.whl", "package"
    ) == "dbfs:/Volumes/catalog/schema/volume/package.whl"
    assert qualification_job._validated_result_output_json(
        output,
        plan_digest=plan["closed_record_sha256"],
        job_id=job["job_id"],
    ) == output
    assert qualification_job._cluster_file_path(output).parts[:5] == (
        "/",
        "Volumes",
        "catalog",
        "schema",
        "volume",
    )


def test_work_dir_rejects_durable_dbfs_and_file_uris():
    plan = _plan()
    job = _auto_job(plan)

    for value in (
        "dbfs:/cachet/qualification-results/work",
        "file:///dbfs/cachet/qualification-results/work",
    ):
        with pytest.raises(ValueError, match="node-local absolute path, not a URI"):
            qualification_job._validated_local_work_dir(
                value,
                plan_digest=plan["closed_record_sha256"],
                job_id=job["job_id"],
            )


def test_fresh_local_work_dir_rejects_any_symlink_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    real = tmp_path / "real"
    real.mkdir()
    symlink = tmp_path / "redirect"
    symlink.symlink_to(real, target_is_directory=True)
    root = symlink / "qualification"
    monkeypatch.setattr(
        qualification_job, "GPU_QUALIFICATION_LOCAL_WORK_ROOT", str(root)
    )
    work = qualification_job._expected_local_work_dir("a" * 64, "job-a")

    with pytest.raises(ValueError, match="cannot traverse a symlink"):
        qualification_job._create_fresh_work_dir(work)

    assert not (real / "qualification").exists()


def test_renderer_rejects_node_local_artifact_and_output_paths(tmp_path: Path):
    uris = _artifact_uris()
    uris["runner_sha256"] = str(tmp_path / "runner.py")
    with pytest.raises(ValueError, match="DBFS or a UC Volume"):
        _render(_plan(), uris)

    with pytest.raises(ValueError, match="DBFS or a UC Volume"):
        render_gpu_qualification_submit_payloads(
            _plan(),
            single_user_name=SINGLE_USER_NAME,
            runner_uri=_artifact_uris()["runner_sha256"],
            package_wheel_uri=_artifact_uris()["package_wheel_sha256"],
            patched_vllm_wheel_uri=_artifact_uris()["patched_vllm_wheel_sha256"],
            artifact_uris=_artifact_uris(),
            output_root=str(tmp_path / "results"),
        )


def test_tokenizer_aware_bundle_verifier_runs_in_isolated_runtime(
    monkeypatch: pytest.MonkeyPatch,
):
    expected = _digest("frozen-input-bundle")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(argv: list[str], **kwargs: Any):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=expected + "\n",
            stderr="",
        )

    monkeypatch.setattr(qualification_sentinels.subprocess, "run", fake_run)

    observed = qualification_sentinels._verify_input_bundle_in_isolated_runtime(
        Path("/local_disk0/runtime/bin/python"),
        Path("/dbfs/cachet/frozen-inputs"),
        expected_sha256=expected,
        environment={"HF_HOME": "/local_disk0/hf"},
    )

    assert observed == expected
    argv, kwargs = calls[0]
    assert argv[0] == "/local_disk0/runtime/bin/python"
    assert "verify_main_latency_inputs" in argv[2]
    assert "examples_per_dataset=32" in argv[2]
    assert kwargs["check"] is True
    assert kwargs["env"] == {"HF_HOME": "/local_disk0/hf"}


def test_isolated_bundle_verifier_fails_closed_on_token_invariant_tamper(
    monkeypatch: pytest.MonkeyPatch,
):
    expected = _digest("frozen-input-bundle")

    def reject_tamper(argv: list[str], **kwargs: Any):
        raise subprocess.CalledProcessError(
            1,
            argv,
            stderr="prepared output provenance mismatch after token recomputation",
        )

    monkeypatch.setattr(qualification_sentinels.subprocess, "run", reject_tamper)

    with pytest.raises(subprocess.CalledProcessError):
        qualification_sentinels._verify_input_bundle_in_isolated_runtime(
            Path("/local_disk0/runtime/bin/python"),
            Path("/dbfs/cachet/tampered-inputs"),
            expected_sha256=expected,
            environment={},
        )


def test_artifact_verifier_keeps_source_closure_and_package_wheel_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifact_paths: dict[str, Path] = {}
    expected: dict[str, str] = {}
    for key in GPU_QUALIFICATION_ARTIFACT_KEYS:
        path = tmp_path / key
        if key == "input_bundle_sha256":
            path.mkdir()
            expected[key] = _digest("verified-input-closure")
        else:
            content = f"raw bytes for {key}".encode()
            path.write_bytes(content)
            expected[key] = _digest(content)
        artifact_paths[key] = path
    monkeypatch.setattr(
        qualification_job,
        "_verify_input_bundle_byte_closure",
        lambda path, *, expected_sha256: expected_sha256,
    )

    qualification_job._verify_artifact_files(artifact_paths, expected=expected)

    swapped = dict(artifact_paths)
    swapped["cachet_source_tree_sha256"] = artifact_paths["package_wheel_sha256"]
    swapped["package_wheel_sha256"] = artifact_paths["cachet_source_tree_sha256"]
    with pytest.raises(ValueError, match="cachet_source_tree_sha256 SHA-256 mismatch"):
        qualification_job._verify_artifact_files(swapped, expected=expected)


def _auto_job(plan: dict[str, Any]) -> dict[str, Any]:
    return next(
        job
        for job in plan["cloud_qualification"]["jobs"]
        if job["sentinel"] == "auto_backend_diagnostic"
    )


def _runtime_for(job: dict[str, Any]) -> dict[str, str]:
    return {
        "gpu": job["gpu"],
        "gpu_compute_capability": job["compute_capability"],
        "torch_cuda_version": "12.9",
        "vllm_version": GPU_QUALIFICATION_VLLM_VERSION,
        "nvidia_driver_version": "570.172.08",
    }


def _execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    measurements: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    plan = _plan()
    pins = _pins()
    job = _auto_job(plan)
    uris = {
        key: f"dbfs:/cachet/test-artifacts/{index}-{key}.artifact"
        for index, key in enumerate(GPU_QUALIFICATION_ARTIFACT_KEYS)
    }
    dbfs_root = tmp_path / "dbfs"
    for key, uri in uris.items():
        source = dbfs_root / uri.removeprefix("dbfs:/")
        if key == "input_bundle_sha256":
            source.mkdir(parents=True)
        else:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(key.encode())

    original_cluster_file_path = qualification_job._cluster_file_path

    def cluster_file_path(value: str | Path) -> Path:
        raw = str(value)
        if raw.startswith("dbfs:/"):
            return dbfs_root / raw.removeprefix("dbfs:/")
        return original_cluster_file_path(value)

    monkeypatch.setattr(qualification_job, "_cluster_file_path", cluster_file_path)
    monkeypatch.setattr(
        qualification_job, "_verify_artifact_files", lambda *a, **k: None
    )
    monkeypatch.setattr(
        qualification_job, "_observe_gpu_runtime", lambda work_dir: _runtime_for(job)
    )
    timestamps = iter(
        (
            datetime(2026, 8, 24, 1, 0, tzinfo=UTC),
            datetime(2026, 8, 24, 1, 1, tzinfo=UTC),
        )
    )
    output_uri = (
        f"dbfs:/cachet/test-results/{plan['closed_record_sha256']}/"
        f"{job['job_id']}/gpu-job-result.json"
    )
    output = cluster_file_path(output_uri)
    local_root = tmp_path / "node-local-qualification"
    monkeypatch.setattr(
        qualification_job,
        "GPU_QUALIFICATION_LOCAL_WORK_ROOT",
        str(local_root),
    )
    work_dir = local_root / plan["closed_record_sha256"] / job["job_id"]
    record = execute_gpu_qualification_job(
        plan_record=plan,
        expected_plan_sha256=plan["closed_record_sha256"],
        job_id=job["job_id"],
        reservation_attempt_id=gpu_qualification_reservation_attempt_id(
            plan["closed_record_sha256"], job["job_id"]
        ),
        runner_uri=uris["runner_sha256"],
        package_wheel_uri=uris["package_wheel_sha256"],
        patched_vllm_wheel_uri=uris["patched_vllm_wheel_sha256"],
        artifact_uris=uris,
        artifact_sha256=pins.to_record(),
        output_json=output_uri,
        work_dir=work_dir,
        cloud_run_id="123",
        cloud_cluster_id="cluster-123",
        sentinel_runner=lambda **kwargs: measurements,
        now=lambda: next(timestamps),
    )
    assert not work_dir.exists()
    return record, output


def test_executor_independently_validates_and_exclusively_seals_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    measurements = {
        "backend_selection_mode": "auto",
        "observed_backend": "FLASHINFER",
        "publication_backend_changed": False,
        "trust_remote_code": False,
    }

    record, output = _execute(tmp_path, monkeypatch, measurements=measurements)

    assert record["measurements"] == measurements
    assert record["attempt_number"] == 0
    assert record["retry_count"] == 0
    assert json.loads(output.read_text(encoding="utf-8")) == record
    assert output.read_bytes().endswith(b"\n")


def test_executor_rejects_arbitrary_extra_measurement_json_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    measurements = {
        "backend_selection_mode": "auto",
        "observed_backend": "FLASHINFER",
        "publication_backend_changed": False,
        "trust_remote_code": False,
        "caller_supplied_success": True,
    }

    with pytest.raises(ValueError, match="closed schema"):
        _execute(tmp_path, monkeypatch, measurements=measurements)
    assert not list((tmp_path / "dbfs").rglob("gpu-job-result.json"))


def test_executor_cleanup_failure_cannot_publish_a_success_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def fail_cleanup(path: Path) -> None:
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr(qualification_job, "_remove_success_work_dir", fail_cleanup)
    with pytest.raises(OSError, match="simulated cleanup failure"):
        _execute(
            tmp_path,
            monkeypatch,
            measurements={
                "backend_selection_mode": "auto",
                "observed_backend": "FLASHINFER",
                "publication_backend_changed": False,
                "trust_remote_code": False,
            },
        )
    assert not list((tmp_path / "dbfs").rglob("gpu-job-result.json"))
