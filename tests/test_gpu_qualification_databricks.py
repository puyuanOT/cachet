import base64
import hashlib
import inspect
import json
import shutil
import subprocess
import zlib
from datetime import UTC, datetime
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import document_kv_cache.gpu_qualification_databricks as qualification_job
import document_kv_cache._gpu_qualification_sentinel_worker as sentinel_worker
import document_kv_cache.gpu_qualification_sentinels as qualification_sentinels
from document_kv_cache._hardware_targets import SUPPORTED_V1_HARDWARE_TARGETS
from document_kv_cache.databricks_resource_ledger import (
    DatabricksClusterHourLedger,
    DatabricksLedgerPrefix,
    create_databricks_cluster_hour_ledger_json,
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
    shutil.copyfile(_RETAINED_LEDGER_PATH, destination)
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


def test_current_plan_production_parameters_stay_below_databricks_safety_cap():
    plan = _plan()
    uris = _publication_artifact_uris()
    payloads = render_gpu_qualification_submit_payloads(
        plan,
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
    assert len(ledger.reservations) == 152
    assert len(ledger.submission_receipts) == 14
    assert len(ledger.terminal_actuals) == 138
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
    assert len(read_databricks_cluster_hour_ledger_json(ledger_path).reservations) == 138
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
    assert (
        len(read_databricks_cluster_hour_ledger_json(ledger_path).reservations) == 138
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
    assert len(crashed.reservations) == 152
    assert len(crashed.submission_receipts) == 0
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
        == 14
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
    assert len(read_databricks_cluster_hour_ledger_json(ledger_path).reservations) == 152
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
    assert len(ledger.terminal_actuals) == 152
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
    assert len(ledger.terminal_actuals) == 139
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
        "acec0bf48ffcd67ee005e2c017b86540e3601ab3d9739f71f243069cae9007db"
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
