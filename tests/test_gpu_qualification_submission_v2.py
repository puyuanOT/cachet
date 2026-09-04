from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

import document_kv_cache.databricks_resource_ledger as ledger_api
import document_kv_cache.databricks_runs as runs_api
import document_kv_cache.gpu_qualification as qualification_v1
import document_kv_cache.gpu_qualification_databricks as databricks_v1
import document_kv_cache.gpu_qualification_databricks_v2 as databricks_v2
import document_kv_cache.gpu_qualification_v2 as qualification_v2
import document_kv_cache.publication_freeze_v2 as freeze_v2
from document_kv_cache.databricks_runs import DatabricksWorkspaceConfig
from document_kv_cache.flashinfer_wheel_repack import (
    FLASHINFER_PATCHED_WHEEL_SHA256,
)
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
    GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    canonical_gpu_qualification_json,
)
from document_kv_cache.gpu_qualification_v2 import (
    GPU_QUALIFICATION_V2_ARTIFACT_KEYS,
    GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS,
    GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX,
    GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS,
    GPUQualificationArtifactPinsV2,
    build_gpu_qualification_plan_v2,
    build_local_preflight_evidence_v2,
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
    RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
    VLLM_RUNTIME_BASE_LOCK_SHA256,
)


_RETAINED_LEDGER_PATH = (
    Path(__file__).parents[1]
    / "databricks-runs"
    / "vllm-0271-publication-prep"
    / "cluster-hours.json"
)
_SINGLE_USER_NAME = "v2-controller@example.com"
_OUTPUT_ROOT = "dbfs:/Volumes/catalog/schema/volume/gpuq-v2-results"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pins() -> GPUQualificationArtifactPinsV2:
    return GPUQualificationArtifactPinsV2(
        runtime_lock_sha256=VLLM_RUNTIME_BASE_LOCK_SHA256,
        patched_vllm_wheel_sha256=GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        patched_flashinfer_wheel_sha256=FLASHINFER_PATCHED_WHEEL_SHA256,
        runtime_closure_manifest_sha256=RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        package_wheel_sha256=_digest("v2-package-wheel"),
        cachet_source_tree_sha256=_digest("v2-source-closure"),
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
        key: f"dbfs:/Volumes/catalog/schema/volume/{index:02d}-{key}/artifact"
        for index, key in enumerate(GPU_QUALIFICATION_V2_ARTIFACT_KEYS)
    }


def _write_preflight(plan: Mapping[str, Any], path: Path) -> Path:
    record = build_local_preflight_evidence_v2(
        plan_sha256=str(plan["closed_record_sha256"]),
        completed_at_utc="2020-01-01T00:00:00Z",
        check_evidence_sha256={
            check_id: _digest(f"preflight:{check_id}")
            for check_id in GPU_QUALIFICATION_V2_LOCAL_CHECK_IDS
        },
    )
    path.write_text(
        canonical_gpu_qualification_json(record) + "\n",
        encoding="utf-8",
    )
    return path


def _write_ledger(path: Path, ledger: ledger_api.DatabricksClusterHourLedger) -> None:
    path.write_text(
        json.dumps(
            ledger_api.databricks_cluster_hour_ledger_to_record(ledger),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _bind_isolated_ledger_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def frozen_path_sha256(_path: str | Path) -> str:
        return PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256

    # These modules intentionally retain imported aliases to the path-binding
    # primitive.  Bind all package-owned call sites to the production identity
    # while exercising an isolated copy of the retained ledger.
    monkeypatch.setattr(
        databricks_v1,
        "databricks_ledger_path_sha256",
        frozen_path_sha256,
    )
    monkeypatch.setattr(
        databricks_v2,
        "databricks_ledger_path_sha256",
        frozen_path_sha256,
    )
    monkeypatch.setattr(
        ledger_api,
        "databricks_ledger_path_sha256",
        frozen_path_sha256,
    )
    monkeypatch.setattr(
        runs_api,
        "databricks_ledger_path_sha256",
        frozen_path_sha256,
    )


def _copy_opening_ledger(
    destination: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained_bytes = _RETAINED_LEDGER_PATH.read_bytes()
    retained_stat = _stable_stat(_RETAINED_LEDGER_PATH)
    retained = ledger_api.read_databricks_cluster_hour_ledger_json(
        _RETAINED_LEDGER_PATH
    )
    opening_prefix = GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX
    ledger_api.require_databricks_ledger_prefix(retained, opening_prefix)
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
    assert ledger_api.databricks_ledger_prefix(opening) == opening_prefix
    _write_ledger(destination, opening)
    assert _RETAINED_LEDGER_PATH.read_bytes() == retained_bytes
    assert _stable_stat(_RETAINED_LEDGER_PATH) == retained_stat
    _bind_isolated_ledger_path(monkeypatch)


def _copy_retained_live_ledger(
    destination: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _copy_opening_ledger(destination, monkeypatch)
    opening = GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX
    payload = databricks_v2.render_gpu_qualification_submit_payloads_v2(
        _plan(),
        single_user_name=_SINGLE_USER_NAME,
        artifact_uris=_artifact_uris(),
        output_root=_OUTPUT_ROOT,
    )[0]
    attempt_id = "v2-test-completed-live-successor"
    ledger_api.reserve_databricks_run_attempt_json(
        destination,
        payload,
        attempt_id=attempt_id,
        workload_id="v2-test-completed-live-successor",
    )
    successor = ledger_api.record_databricks_run_terminal_actual_json(
        destination,
        attempt_id=attempt_id,
        terminal_state="succeeded",
        actual_cluster_duration_seconds=0.0,
    )
    assert successor.active_reserved_task_count == 0
    assert successor.active_reserved_cluster_hours == 0.0
    assert len(successor.reservations) == opening.reservation_count + 1
    assert len(successor.submission_receipts) == opening.submission_receipt_count
    assert len(successor.terminal_actuals) == opening.terminal_actual_count + 1


def _install_preflight_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> list[bool]:
    freshness: list[bool] = []

    def validate(
        path: str | Path,
        *,
        plan_record: Mapping[str, Any],
        submit_payloads: tuple[dict[str, Any], ...],
        workspace_config: DatabricksWorkspaceConfig,
        require_fresh_workspace: bool,
    ) -> dict[str, Any]:
        assert plan_record["record_type"].endswith(".v2")
        assert len(submit_payloads) == 14
        assert isinstance(workspace_config, DatabricksWorkspaceConfig)
        freshness.append(require_fresh_workspace)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    monkeypatch.setattr(
        freeze_v2,
        "validate_gpu_qualification_local_preflight_bundle_v2",
        validate,
    )
    return freshness


class _PackageOwnedCloud:
    def __init__(
        self,
        *,
        config: DatabricksWorkspaceConfig,
        plan: Mapping[str, Any],
    ) -> None:
        self.config = config
        self.attempt_ids = tuple(
            databricks_v1.gpu_qualification_reservation_attempt_id(
                str(plan["closed_record_sha256"]),
                str(job["job_id"]),
            )
            for job in plan["cloud_qualification"]["jobs"]
        )
        self.post_attempt_ids: list[str] = []
        self.resume_attempt_ids: list[str] = []
        self.post_payloads: list[dict[str, Any]] = []
        self.get_run_ids: list[str] = []
        self.events: list[tuple[str, str]] = []
        self.runs: dict[str, dict[str, Any]] = {}

    def _run_id(self, attempt_id: str) -> int:
        return 900_000 + self.attempt_ids.index(attempt_id)

    def submit(
        self,
        config: DatabricksWorkspaceConfig,
        payload: Mapping[str, Any],
        *,
        ledger_path: str | Path,
        attempt_id: str,
        batch_authorization: Any,
    ) -> dict[str, Any]:
        assert config is self.config
        assert attempt_id in batch_authorization.attempt_ids
        assert batch_authorization.attempt_ids == self.attempt_ids
        member_index = self.attempt_ids.index(attempt_id)
        if member_index > 0:
            assert self.get_run_ids[-1] == str(
                self._run_id(self.attempt_ids[member_index - 1])
            )
        self.post_attempt_ids.append(attempt_id)
        self.post_payloads.append(deepcopy(dict(payload)))
        response = {"run_id": self._run_id(attempt_id)}
        run_id = str(response["run_id"])
        task = payload["tasks"][0]
        run_start = 1_787_533_140_000 + member_index * 10_000
        task_start = run_start + 1_000
        task_end = task_start + 5_000
        self.runs[run_id] = {
            "end_time": task_end + 1_000,
            "run_id": response["run_id"],
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
                    "run_id": 990_000 + member_index,
                    "start_time": task_start,
                    "state": {
                        "life_cycle_state": "TERMINATED",
                        "result_state": "SUCCESS",
                    },
                    "task_key": task["task_key"],
                }
            ],
        }
        self.events.append(("post", attempt_id))
        ledger_api.record_databricks_run_submission_receipt_json(
            ledger_path,
            attempt_id=attempt_id,
            submit_response=response,
        )
        return response

    def get_run(
        self,
        config: DatabricksWorkspaceConfig,
        run_id: str,
    ) -> dict[str, Any]:
        assert config is self.config
        normalized = str(run_id)
        self.get_run_ids.append(normalized)
        self.events.append(("get", normalized))
        return deepcopy(self.runs[normalized])

    def resume(
        self,
        config: DatabricksWorkspaceConfig,
        payload: Mapping[str, Any],
        *,
        ledger_path: str | Path,
        attempt_id: str,
        batch_authorization: Any,
    ) -> dict[str, Any]:
        self.resume_attempt_ids.append(attempt_id)
        ledger = ledger_api.read_databricks_cluster_hour_ledger_json(ledger_path)
        existing = next(
            (
                receipt
                for receipt in ledger.submission_receipts
                if receipt.attempt_id == attempt_id
            ),
            None,
        )
        if existing is not None:
            return {"run_id": existing.run_id}
        return self.submit(
            config,
            payload,
            ledger_path=ledger_path,
            attempt_id=attempt_id,
            batch_authorization=batch_authorization,
        )


@dataclass(slots=True)
class _Case:
    config: DatabricksWorkspaceConfig
    plan: dict[str, Any]
    artifact_uris: dict[str, str]
    ledger_path: Path
    phase_predecessor: ledger_api.DatabricksLedgerPrefix
    receipt_root: Path
    preflight_path: Path
    preflight_freshness: list[bool]
    cloud: _PackageOwnedCloud

    def call_kwargs(self) -> dict[str, Any]:
        return {
            "plan_record": self.plan,
            "single_user_name": _SINGLE_USER_NAME,
            "artifact_uris": self.artifact_uris,
            "output_root": _OUTPUT_ROOT,
            "ledger_path": self.ledger_path,
            "expected_phase_predecessor_prefix": self.phase_predecessor,
            "submit_receipt_root": self.receipt_root,
            "local_preflight_evidence_path": self.preflight_path,
        }


def _case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    retained_live_ledger: bool = False,
) -> _Case:
    ledger_path = tmp_path / "cluster-hours.json"
    if retained_live_ledger:
        _copy_retained_live_ledger(ledger_path, monkeypatch)
    else:
        _copy_opening_ledger(ledger_path, monkeypatch)
    plan = _plan()
    config = DatabricksWorkspaceConfig(
        "https://dbc.example",
        "secret-token",
    )
    cloud = _PackageOwnedCloud(config=config, plan=plan)
    monkeypatch.setattr(
        databricks_v2,
        "submit_pre_reserved_databricks_run",
        cloud.submit,
    )
    monkeypatch.setattr(
        databricks_v2,
        "resume_pre_reserved_databricks_run",
        cloud.resume,
    )
    monkeypatch.setattr(databricks_v2, "get_databricks_run", cloud.get_run)
    return _Case(
        config=config,
        plan=plan,
        artifact_uris=_artifact_uris(),
        ledger_path=ledger_path,
        phase_predecessor=ledger_api.databricks_ledger_prefix(
            ledger_api.read_databricks_cluster_hour_ledger_json(ledger_path)
        ),
        receipt_root=tmp_path / "submit-receipts-v2",
        preflight_path=_write_preflight(plan, tmp_path / "preflight-v2.json"),
        preflight_freshness=_install_preflight_stub(monkeypatch),
        cloud=cloud,
    )


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _root_bytes(root: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(root.iterdir())}


def _root_snapshot(root: Path) -> tuple[Any, ...]:
    root_stat = root.stat()
    entries: list[Any] = []
    for path in sorted(root.iterdir()):
        observed = path.lstat()
        payload: str | bytes
        if path.is_symlink():
            payload = str(path.readlink())
        else:
            payload = path.read_bytes()
        entries.append(
            (
                path.name,
                observed.st_mode,
                observed.st_size,
                observed.st_mtime_ns,
                observed.st_ctime_ns,
                payload,
            )
        )
    return (
        root_stat.st_ino,
        root_stat.st_mode,
        root_stat.st_size,
        root_stat.st_mtime_ns,
        root_stat.st_ctime_ns,
        tuple(entries),
    )


def _stable_stat(path: Path) -> tuple[int, int, int, int, int]:
    observed = path.stat()
    return (
        observed.st_ino,
        observed.st_mode,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _assert_v2_seal(record: Mapping[str, Any]) -> None:
    assert record["closed_record_sha256"] == (
        databricks_v2._controller_record_digest_v2(record)
    )
    with pytest.raises(ValueError, match="closed_record_sha256 mismatch"):
        databricks_v1._require_closed_record_digest(record, "v2 record")


def _planned_job_ids(plan: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(job["job_id"]) for job in plan["cloud_qualification"]["jobs"])


def _completed_submission_with_terminal_fixtures(
    case: _Case,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    databricks_v2.submit_gpu_qualification_jobs_v2(
        case.config,
        **case.call_kwargs(),
    )
    _validated_plan, _validated_pins, payloads, contracts = (
        databricks_v2._validated_controller_contract_v2(
            plan_record=case.plan,
            single_user_name=_SINGLE_USER_NAME,
            artifact_uris=case.artifact_uris,
            output_root=_OUTPUT_ROOT,
        )
    )
    runs = deepcopy(case.cloud.runs)
    results: dict[str, dict[str, Any]] = {}
    for planned_job, payload, contract in zip(
        case.plan["cloud_qualification"]["jobs"],
        payloads,
        contracts,
        strict=True,
    ):
        run_id = str(case.cloud._run_id(str(contract["reservation_attempt_id"])))
        cluster_id = f"cluster-{run_id}"
        assert runs[run_id]["run_name"] == payload["run_name"]
        assert runs[run_id]["tasks"][0]["cluster_instance"] == {
            "cluster_id": cluster_id
        }
        results[str(contract["output_json"])] = {
            "closed_record_sha256": _digest(f"result:{contract['job_id']}"),
            "cloud_cluster_id": cluster_id,
            "cloud_run_id": run_id,
            "job_id": contract["job_id"],
            "measurements": (
                {
                    "candidate_qualified": True,
                    "gpu_memory_utilization": 0.70,
                }
                if str(planned_job["job_id"]).startswith(
                    "aws-g6-l4-32k-c4-gmu-"
                )
                else {}
            ),
            "output_json": contract["output_json"],
            "reservation_attempt_id": contract["reservation_attempt_id"],
            "task_key": contract["task_key"],
        }
    return payloads, contracts, runs, results


def _collector_kwargs(case: _Case, evidence_root: Path) -> dict[str, Any]:
    return {
        **case.call_kwargs(),
        "evidence_root": evidence_root,
    }


def _install_minimal_governed_evidence_stubs(
    case: _Case,
    results: Mapping[str, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> qualification_v1.GPUQualificationSelection:
    def read_result(
        _config: Any,
        output_json: str,
        *,
        label: str,
        closed_record_convention: str,
    ) -> dict[str, Any]:
        assert label.startswith("GPU v2 result ")
        assert closed_record_convention == "field_blank"
        return deepcopy(results[output_json])

    monkeypatch.setattr(
        databricks_v1,
        "_read_gpu_qualification_result",
        read_result,
    )
    monkeypatch.setattr(
        databricks_v2,
        "validate_gpu_job_result_v2_record",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        databricks_v2,
        "_build_governed_cloud_gpu_evidence_v2",
        lambda **kwargs: {
            "terminal_receipts": list(kwargs["terminal_receipts"]),
        },
    )

    def build_evidence(**kwargs: Any) -> dict[str, Any]:
        record = {"closed_record_sha256": "", **kwargs}
        record["closed_record_sha256"] = qualification_v2._closed_record_sha256(
            record
        )
        return record

    monkeypatch.setattr(
        databricks_v2,
        "_build_governed_gpu_qualification_evidence_v2",
        build_evidence,
    )
    selection = qualification_v1.GPUQualificationSelection(
        attention_backend="TRITON_ATTN",
        gpu_memory_utilization=0.70,
        generation_hardware_id="aws-g6e-l40s",
        generation_databricks_node_type_id="g6e.8xlarge",
        generation_artifacts_sha256=_digest("v2-generation-artifacts"),
        generation_prefix_tokens_per_second=40.0,
        plan_sha256=str(case.plan["closed_record_sha256"]),
    )
    monkeypatch.setattr(
        databricks_v2,
        "validate_gpu_qualification_evidence_v2_record",
        lambda *_args, **_kwargs: selection,
    )
    return selection


def test_v2_submit_and_resume_signatures_own_payload_transport_and_clock() -> None:
    expected = (
        "config",
        "plan_record",
        "single_user_name",
        "artifact_uris",
        "output_root",
        "ledger_path",
        "expected_phase_predecessor_prefix",
        "submit_receipt_root",
        "local_preflight_evidence_path",
    )
    for function in (
        databricks_v2.submit_gpu_qualification_jobs_v2,
        databricks_v2.resume_gpu_qualification_job_submissions_v2,
    ):
        signature = inspect.signature(function)
        assert tuple(signature.parameters) == expected
        assert signature.parameters["config"].kind is (
            inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        assert all(
            signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            for name in expected[1:]
        )
        assert not {
            "submit_payloads",
            "opener",
            "now",
            "get_run",
            "sleep",
            "monotonic",
        }.intersection(
            signature.parameters
        )


def test_fresh_v2_submit_reserves_and_receipts_exact_fourteen_with_v2_seals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    expected_payloads = databricks_v2.render_gpu_qualification_submit_payloads_v2(
        case.plan,
        single_user_name=_SINGLE_USER_NAME,
        artifact_uris=case.artifact_uris,
        output_root=_OUTPUT_ROOT,
    )

    receipts = databricks_v2.submit_gpu_qualification_jobs_v2(
        case.config,
        **case.call_kwargs(),
    )

    job_ids = _planned_job_ids(case.plan)
    expected_names = {
        "phase-lease-v2.json",
        "batch-reserved-v2.json",
        *(f"{job_id}.json" for job_id in job_ids),
    }
    assert len(receipts) == len(expected_payloads) == 14
    assert case.preflight_freshness == [True]
    assert case.cloud.post_attempt_ids == list(case.cloud.attempt_ids)
    assert case.cloud.post_payloads == list(expected_payloads)
    assert {path.name for path in case.receipt_root.iterdir()} == expected_names

    ledger = ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
    opening = GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX
    assert len(ledger.reservations) == opening.reservation_count + 14
    assert len(ledger.submission_receipts) == opening.submission_receipt_count + 14
    assert len(ledger.terminal_actuals) == opening.terminal_actual_count + 14
    assert ledger.active_reserved_cluster_hours == 0.0
    assert tuple(
        item.attempt_id for item in ledger.reservations[opening.reservation_count :]
    ) == case.cloud.attempt_ids
    assert tuple(
        item.workload_id for item in ledger.reservations[opening.reservation_count :]
    ) == tuple(
        f"gpuq-v2/{case.plan['closed_record_sha256'][:16]}/{job_id}"
        for job_id in job_ids
    )
    assert tuple(
        item.attempt_id
        for item in ledger.submission_receipts[opening.submission_receipt_count :]
    ) == case.cloud.attempt_ids
    assert tuple(
        item.attempt_id
        for item in ledger.terminal_actuals[opening.terminal_actual_count :]
    ) == case.cloud.attempt_ids
    assert case.cloud.events == [
        event
        for attempt_id in case.cloud.attempt_ids
        for event in (
            ("post", attempt_id),
            ("get", str(case.cloud._run_id(attempt_id))),
        )
    ]

    lease = _json(case.receipt_root / "phase-lease-v2.json")
    marker = _json(case.receipt_root / "batch-reserved-v2.json")
    assert lease["record_type"] == (
        databricks_v2.GPU_QUALIFICATION_V2_PHASE_LEASE_RECORD_TYPE
    )
    assert marker["record_type"] == (
        databricks_v2.GPU_QUALIFICATION_V2_BATCH_MARKER_RECORD_TYPE
    )
    assert lease["schema_version"] == marker["schema_version"] == 2
    assert tuple(lease["artifact_sha256"]) == GPU_QUALIFICATION_V2_ARTIFACT_KEYS
    assert lease["artifact_sha256"] == _pins().to_record()
    assert lease["attempt_ids"] == list(case.cloud.attempt_ids)
    assert len(lease["submit_payload_sha256"]) == 14
    assert marker["phase_lease_record_sha256"] == lease["closed_record_sha256"]
    expected_payload_closure = hashlib.sha256(
        canonical_gpu_qualification_json(
            {"payloads": list(expected_payloads)}
        ).encode("utf-8")
    ).hexdigest()
    assert lease["local_preflight"]["submit_payloads_sha256"] == (
        expected_payload_closure
    )
    _assert_v2_seal(lease)
    _assert_v2_seal(marker)
    for receipt, job_id in zip(receipts, job_ids, strict=True):
        assert receipt == _json(case.receipt_root / f"{job_id}.json")
        assert receipt["record_type"] == (
            databricks_v2.GPU_QUALIFICATION_V2_SUBMIT_RECEIPT_RECORD_TYPE
        )
        assert receipt["schema_version"] == 2
        assert "submitted_at_utc" not in receipt
        _assert_v2_seal(receipt)


def test_terminal_barrier_timeout_stops_before_next_post_and_resume_closes_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    original_wait_seconds = databricks_v2._V2_TERMINAL_WAIT_SECONDS

    def running_get(
        config: DatabricksWorkspaceConfig,
        run_id: str,
    ) -> dict[str, Any]:
        run = case.cloud.get_run(config, run_id)
        run["state"] = {"life_cycle_state": "RUNNING", "result_state": None}
        return run

    monkeypatch.setattr(databricks_v2, "get_databricks_run", running_get)
    monkeypatch.setattr(databricks_v2, "_V2_TERMINAL_WAIT_SECONDS", 0.0)

    with pytest.raises(TimeoutError, match="timed out waiting for v2 qualification"):
        databricks_v2.submit_gpu_qualification_jobs_v2(
            case.config,
            **case.call_kwargs(),
        )

    opening = GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX
    interrupted = ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
    assert len(interrupted.submission_receipts) == opening.submission_receipt_count + 1
    assert len(interrupted.terminal_actuals) == opening.terminal_actual_count
    assert case.cloud.post_attempt_ids == [case.cloud.attempt_ids[0]]
    assert {path.name for path in case.receipt_root.iterdir()} == {
        "phase-lease-v2.json",
        "batch-reserved-v2.json",
        f"{_planned_job_ids(case.plan)[0]}.json",
    }

    monkeypatch.setattr(databricks_v2, "get_databricks_run", case.cloud.get_run)
    monkeypatch.setattr(
        databricks_v2,
        "_V2_TERMINAL_WAIT_SECONDS",
        original_wait_seconds,
    )
    receipts = databricks_v2.resume_gpu_qualification_job_submissions_v2(
        case.config,
        **case.call_kwargs(),
    )

    completed = ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
    assert len(receipts) == 14
    assert len(completed.terminal_actuals) == opening.terminal_actual_count + 14
    first_terminal_get = case.cloud.events.index(
        ("get", str(case.cloud._run_id(case.cloud.attempt_ids[0]))),
        2,
    )
    second_post = case.cloud.events.index(("post", case.cloud.attempt_ids[1]))
    assert first_terminal_get < second_post


def test_successor_fresh_submit_uses_complete_live_opening_predecessor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch, retained_live_ledger=True)
    historical = GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX
    before = ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
    live_predecessor = ledger_api.databricks_ledger_prefix(before)
    assert case.plan["campaign_ledger_prefix"] == historical.to_record()
    assert (
        historical.reservation_count,
        historical.submission_receipt_count,
        historical.terminal_actual_count,
    ) == (430, 292, 430)
    assert (
        live_predecessor.reservation_count,
        live_predecessor.submission_receipt_count,
        live_predecessor.terminal_actual_count,
    ) == (431, 292, 431)

    receipts = databricks_v2.submit_gpu_qualification_jobs_v2(
        case.config,
        **case.call_kwargs(),
    )

    after = ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
    assert len(receipts) == 14
    assert after.reservations[: live_predecessor.reservation_count] == (
        before.reservations
    )
    assert after.submission_receipts[: live_predecessor.submission_receipt_count] == (
        before.submission_receipts
    )
    assert after.terminal_actuals[: live_predecessor.terminal_actual_count] == (
        before.terminal_actuals
    )
    assert len(after.reservations) == live_predecessor.reservation_count + 14
    assert len(after.submission_receipts) == (
        live_predecessor.submission_receipt_count + 14
    )
    assert len(after.terminal_actuals) == live_predecessor.terminal_actual_count + 14
    assert after.active_reserved_cluster_hours == 0.0
    assert tuple(
        item.attempt_id
        for item in after.reservations[live_predecessor.reservation_count :]
    ) == case.cloud.attempt_ids
    assert tuple(
        item.attempt_id
        for item in after.submission_receipts[
            live_predecessor.submission_receipt_count :
        ]
    ) == case.cloud.attempt_ids

    lease = _json(case.receipt_root / "phase-lease-v2.json")
    marker = _json(case.receipt_root / "batch-reserved-v2.json")
    assert lease["predecessor_prefix"] == live_predecessor.to_record()
    assert marker["predecessor_prefix"] == live_predecessor.to_record()
    expected_batch_prefix = ledger_api.databricks_ledger_prefix_at_counts(
        after,
        reservation_count=live_predecessor.reservation_count + 14,
        submission_receipt_count=live_predecessor.submission_receipt_count,
        terminal_actual_count=live_predecessor.terminal_actual_count,
    )
    assert marker["batch_prefix"] == expected_batch_prefix.to_record()
    assert case.cloud.post_attempt_ids == list(case.cloud.attempt_ids)
    assert case.preflight_freshness == [True]


def test_successor_submit_rejects_nonquiescent_live_predecessor_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch, retained_live_ledger=True)
    live = ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
    _write_ledger(
        case.ledger_path,
        replace(live, terminal_actuals=live.terminal_actuals[:-1]),
    )
    nonquiescent = ledger_api.read_databricks_cluster_hour_ledger_json(
        case.ledger_path
    )
    assert nonquiescent.active_reserved_task_count == 1
    assert nonquiescent.active_reserved_cluster_hours > 0.0
    case.phase_predecessor = ledger_api.databricks_ledger_prefix(nonquiescent)
    ledger_before = case.ledger_path.read_bytes()
    ledger_stat = _stable_stat(case.ledger_path)

    with pytest.raises(ValueError, match="phase predecessor must be quiescent"):
        databricks_v2.submit_gpu_qualification_jobs_v2(
            case.config,
            **case.call_kwargs(),
        )

    assert case.ledger_path.read_bytes() == ledger_before
    assert _stable_stat(case.ledger_path) == ledger_stat
    assert case.cloud.post_attempt_ids == []
    assert not case.receipt_root.exists()


def test_resume_recovers_exact_lease_only_crash_and_reserves_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    ledger_before = case.ledger_path.read_bytes()
    original_writer = databricks_v1._write_canonical_exclusive

    def crash_after_lease(record: Mapping[str, Any], path: str | Path) -> None:
        original_writer(record, path)
        if Path(path).name == "phase-lease-v2.json":
            raise RuntimeError("simulated hard stop after the v2 lease")

    monkeypatch.setattr(
        databricks_v1,
        "_write_canonical_exclusive",
        crash_after_lease,
    )
    with pytest.raises(RuntimeError, match="hard stop after the v2 lease"):
        databricks_v2.submit_gpu_qualification_jobs_v2(
            case.config,
            **case.call_kwargs(),
        )
    assert case.ledger_path.read_bytes() == ledger_before
    assert {path.name for path in case.receipt_root.iterdir()} == {
        "phase-lease-v2.json"
    }
    assert case.cloud.post_attempt_ids == []

    monkeypatch.setattr(
        databricks_v1,
        "_write_canonical_exclusive",
        original_writer,
    )
    receipts = databricks_v2.resume_gpu_qualification_job_submissions_v2(
        case.config,
        **case.call_kwargs(),
    )

    ledger = ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
    opening = GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX
    assert len(receipts) == 14
    assert len(ledger.reservations) == opening.reservation_count + 14
    assert len(ledger.submission_receipts) == opening.submission_receipt_count + 14
    assert case.cloud.post_attempt_ids == list(case.cloud.attempt_ids)
    assert case.cloud.resume_attempt_ids == list(case.cloud.attempt_ids)
    assert case.preflight_freshness == [True, False]


def test_reservation_postcommit_exception_retains_lease_for_exact_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch, retained_live_ledger=True)
    predecessor = case.phase_predecessor
    original_reserver = (
        databricks_v2.reserve_databricks_run_attempt_batch_authorized_json
    )

    def reserve_then_interrupt(*args: Any, **kwargs: Any) -> Any:
        original_reserver(*args, **kwargs)
        raise RuntimeError("simulated postcommit reservation interruption")

    monkeypatch.setattr(
        databricks_v2,
        "reserve_databricks_run_attempt_batch_authorized_json",
        reserve_then_interrupt,
    )
    with pytest.raises(RuntimeError, match="postcommit reservation interruption"):
        databricks_v2.submit_gpu_qualification_jobs_v2(
            case.config,
            **case.call_kwargs(),
        )

    interrupted = ledger_api.read_databricks_cluster_hour_ledger_json(
        case.ledger_path
    )
    assert len(interrupted.reservations) == predecessor.reservation_count + 14
    assert (
        len(interrupted.submission_receipts)
        == predecessor.submission_receipt_count
    )
    assert len(interrupted.terminal_actuals) == predecessor.terminal_actual_count
    assert {path.name for path in case.receipt_root.iterdir()} == {
        "phase-lease-v2.json"
    }
    assert case.cloud.post_attempt_ids == []

    monkeypatch.setattr(
        databricks_v2,
        "reserve_databricks_run_attempt_batch_authorized_json",
        original_reserver,
    )
    receipts = databricks_v2.resume_gpu_qualification_job_submissions_v2(
        case.config,
        **case.call_kwargs(),
    )
    completed = ledger_api.read_databricks_cluster_hour_ledger_json(
        case.ledger_path
    )
    assert len(receipts) == 14
    assert len(completed.reservations) == predecessor.reservation_count + 14
    assert (
        len(completed.submission_receipts)
        == predecessor.submission_receipt_count + 14
    )
    assert tuple(
        item.attempt_id
        for item in completed.reservations[predecessor.reservation_count :]
    ) == case.cloud.attempt_ids
    assert case.cloud.post_attempt_ids == list(case.cloud.attempt_ids)
    assert case.cloud.resume_attempt_ids == list(case.cloud.attempt_ids)


def test_successor_lease_only_resume_reserves_once_after_complete_live_predecessor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch, retained_live_ledger=True)
    before = ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
    live_predecessor = ledger_api.databricks_ledger_prefix(before)
    ledger_before = case.ledger_path.read_bytes()
    original_writer = databricks_v1._write_canonical_exclusive

    def crash_after_lease(record: Mapping[str, Any], path: str | Path) -> None:
        original_writer(record, path)
        if Path(path).name == "phase-lease-v2.json":
            raise RuntimeError("simulated successor stop after the v2 lease")

    monkeypatch.setattr(
        databricks_v1,
        "_write_canonical_exclusive",
        crash_after_lease,
    )
    with pytest.raises(RuntimeError, match="successor stop after the v2 lease"):
        databricks_v2.submit_gpu_qualification_jobs_v2(
            case.config,
            **case.call_kwargs(),
        )
    assert case.ledger_path.read_bytes() == ledger_before
    lease = _json(case.receipt_root / "phase-lease-v2.json")
    assert lease["predecessor_prefix"] == live_predecessor.to_record()
    assert case.cloud.post_attempt_ids == []

    monkeypatch.setattr(
        databricks_v1,
        "_write_canonical_exclusive",
        original_writer,
    )
    receipts = databricks_v2.resume_gpu_qualification_job_submissions_v2(
        case.config,
        **case.call_kwargs(),
    )

    after = ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
    assert len(receipts) == 14
    assert after.reservations[: live_predecessor.reservation_count] == (
        before.reservations
    )
    assert after.submission_receipts[: live_predecessor.submission_receipt_count] == (
        before.submission_receipts
    )
    assert after.terminal_actuals[: live_predecessor.terminal_actual_count] == (
        before.terminal_actuals
    )
    assert len(after.reservations) == live_predecessor.reservation_count + 14
    assert len(after.submission_receipts) == (
        live_predecessor.submission_receipt_count + 14
    )
    assert len(after.terminal_actuals) == live_predecessor.terminal_actual_count + 14
    assert after.active_reserved_cluster_hours == 0.0
    marker = _json(case.receipt_root / "batch-reserved-v2.json")
    assert marker["predecessor_prefix"] == live_predecessor.to_record()
    assert case.cloud.post_attempt_ids == list(case.cloud.attempt_ids)
    assert case.cloud.resume_attempt_ids == list(case.cloud.attempt_ids)
    assert case.preflight_freshness == [True, False]


def test_resume_rejects_resealed_live_lease_predecessor_without_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    historical = GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX
    assert case.phase_predecessor == historical
    original_writer = databricks_v1._write_canonical_exclusive

    def crash_after_lease(record: Mapping[str, Any], path: str | Path) -> None:
        original_writer(record, path)
        if Path(path).name == "phase-lease-v2.json":
            raise RuntimeError("successor lease-only fixture")

    monkeypatch.setattr(
        databricks_v1,
        "_write_canonical_exclusive",
        crash_after_lease,
    )
    with pytest.raises(RuntimeError, match="successor lease-only fixture"):
        databricks_v2.submit_gpu_qualification_jobs_v2(
            case.config,
            **case.call_kwargs(),
        )
    monkeypatch.setattr(
        databricks_v1,
        "_write_canonical_exclusive",
        original_writer,
    )

    lease_path = case.receipt_root / "phase-lease-v2.json"
    lease = _json(lease_path)
    assert lease["predecessor_prefix"] == historical.to_record()
    _copy_retained_live_ledger(case.ledger_path, monkeypatch)
    live_predecessor = ledger_api.databricks_ledger_prefix(
        ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
    )
    assert live_predecessor != historical
    lease["predecessor_prefix"] = live_predecessor.to_record()
    lease["closed_record_sha256"] = ""
    databricks_v2._seal_controller_record_v2(lease)
    lease_path.write_text(
        canonical_gpu_qualification_json(lease) + "\n",
        encoding="utf-8",
    )
    ledger_before = case.ledger_path.read_bytes()
    ledger_stat = _stable_stat(case.ledger_path)
    root_before = _root_snapshot(case.receipt_root)

    def forbidden_resume(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        pytest.fail("a drifted successor predecessor reached cloud recovery")

    monkeypatch.setattr(
        databricks_v2,
        "resume_pre_reserved_databricks_run",
        forbidden_resume,
    )
    with pytest.raises(ValueError, match="phase lease differs"):
        databricks_v2.resume_gpu_qualification_job_submissions_v2(
            case.config,
            **case.call_kwargs(),
        )
    assert case.ledger_path.read_bytes() == ledger_before
    assert _stable_stat(case.ledger_path) == ledger_stat
    assert _root_snapshot(case.receipt_root) == root_before
    assert case.cloud.post_attempt_ids == []
    assert case.cloud.resume_attempt_ids == []


def _interrupt_after_sixth_ledger_receipt(
    case: _Case,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    original_recorder = databricks_v2.record_databricks_run_submission_receipt_json
    calls = 0

    def interrupt(
        ledger_path: str | Path,
        *,
        attempt_id: str,
        submit_response: Mapping[str, Any],
    ) -> ledger_api.DatabricksClusterHourLedger:
        nonlocal calls
        calls += 1
        if calls == 6:
            raise RuntimeError("simulated stop after five controller receipts")
        return original_recorder(
            ledger_path,
            attempt_id=attempt_id,
            submit_response=submit_response,
        )

    monkeypatch.setattr(
        databricks_v2,
        "record_databricks_run_submission_receipt_json",
        interrupt,
    )
    with pytest.raises(RuntimeError, match="five controller receipts"):
        databricks_v2.submit_gpu_qualification_jobs_v2(
            case.config,
            **case.call_kwargs(),
        )
    monkeypatch.setattr(
        databricks_v2,
        "record_databricks_run_submission_receipt_json",
        original_recorder,
    )
    assert calls == 6
    return original_recorder


def test_resume_after_five_posts_is_canonical_and_second_resume_is_byte_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    _interrupt_after_sixth_ledger_receipt(case, monkeypatch)
    opening = GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX
    interrupted = ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
    job_ids = _planned_job_ids(case.plan)
    assert len(interrupted.reservations) == opening.reservation_count + 14
    assert len(interrupted.submission_receipts) == opening.submission_receipt_count + 6
    assert len(interrupted.terminal_actuals) == opening.terminal_actual_count + 5
    assert case.cloud.get_run_ids == [
        str(case.cloud._run_id(attempt_id))
        for attempt_id in case.cloud.attempt_ids[:5]
    ]
    assert case.cloud.post_attempt_ids == list(case.cloud.attempt_ids[:6])
    assert {path.name for path in case.receipt_root.iterdir()} == {
        "phase-lease-v2.json",
        "batch-reserved-v2.json",
        *(f"{job_id}.json" for job_id in job_ids[:5]),
        f"{job_ids[5]}.post-intent-v2",
    }
    intent = _json(case.receipt_root / f"{job_ids[5]}.post-intent-v2")
    assert intent["record_type"] == (
        databricks_v2.GPU_QUALIFICATION_V2_POST_INTENT_RECORD_TYPE
    )
    assert intent["schema_version"] == 2
    _assert_v2_seal(intent)

    receipts = databricks_v2.resume_gpu_qualification_job_submissions_v2(
        case.config,
        **case.call_kwargs(),
    )
    assert len(receipts) == 14
    assert case.cloud.resume_attempt_ids == list(case.cloud.attempt_ids[5:])
    assert case.cloud.post_attempt_ids == list(case.cloud.attempt_ids)
    assert len(set(case.cloud.post_attempt_ids)) == 14
    final = ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
    assert len(final.submission_receipts) == opening.submission_receipt_count + 14
    assert len(final.terminal_actuals) == opening.terminal_actual_count + 14
    assert final.active_reserved_cluster_hours == 0.0
    sixth_get = case.cloud.events.index(
        ("get", str(case.cloud._run_id(case.cloud.attempt_ids[5])))
    )
    seventh_post = case.cloud.events.index(("post", case.cloud.attempt_ids[6]))
    assert sixth_get < seventh_post

    ledger_bytes = case.ledger_path.read_bytes()
    ledger_stat = _stable_stat(case.ledger_path)
    root_bytes = _root_bytes(case.receipt_root)
    root_stats = {
        path.name: _stable_stat(path) for path in case.receipt_root.iterdir()
    }
    resume_calls_before = list(case.cloud.resume_attempt_ids)
    get_calls_before = list(case.cloud.get_run_ids)

    def forbidden_resume(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        pytest.fail("a completed v2 phase must not invoke cloud submission")

    monkeypatch.setattr(
        databricks_v2,
        "resume_pre_reserved_databricks_run",
        forbidden_resume,
    )
    replayed = databricks_v2.resume_gpu_qualification_job_submissions_v2(
        case.config,
        **case.call_kwargs(),
    )
    assert replayed == receipts
    assert case.cloud.resume_attempt_ids == resume_calls_before
    assert case.cloud.get_run_ids == get_calls_before
    assert case.ledger_path.read_bytes() == ledger_bytes
    assert _stable_stat(case.ledger_path) == ledger_stat
    assert _root_bytes(case.receipt_root) == root_bytes
    assert {
        path.name: _stable_stat(path) for path in case.receipt_root.iterdir()
    } == root_stats
    assert case.preflight_freshness == [True, False, False]


@pytest.mark.parametrize(
    "tamper",
    ["path", "prefix", "plan", "preflight", "cross-v1-plan"],
)
def test_v2_authority_tamper_rejects_without_ledger_receipt_or_cloud_write(
    tamper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    plan: Mapping[str, Any] = case.plan
    if tamper == "path":
        monkeypatch.setattr(
            databricks_v2,
            "databricks_ledger_path_sha256",
            lambda _path: "f" * 64,
        )
    elif tamper == "prefix":
        ledger = ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
        _write_ledger(
            case.ledger_path,
            replace(ledger, terminal_actuals=ledger.terminal_actuals[:-1]),
        )
    elif tamper == "plan":
        changed = deepcopy(case.plan)
        changed["cloud_qualification"]["max_retries"] = 1
        changed["closed_record_sha256"] = qualification_v2._closed_record_sha256(
            changed
        )
        plan = changed
    elif tamper == "preflight":
        changed = _json(case.preflight_path)
        changed["plan_sha256"] = "f" * 64
        changed["closed_record_sha256"] = qualification_v2._closed_record_sha256(
            changed
        )
        case.preflight_path.write_text(
            canonical_gpu_qualification_json(changed) + "\n",
            encoding="utf-8",
        )
    else:
        plan = qualification_v1.build_gpu_qualification_plan(
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

    ledger_before = case.ledger_path.read_bytes()
    with pytest.raises((TypeError, ValueError)):
        databricks_v2.submit_gpu_qualification_jobs_v2(
            case.config,
            **{**case.call_kwargs(), "plan_record": plan},
        )
    assert case.ledger_path.read_bytes() == ledger_before
    assert case.cloud.post_attempt_ids == []
    assert not case.receipt_root.exists()


def test_resume_cross_v1_phase_record_rejects_without_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    original_writer = databricks_v1._write_canonical_exclusive

    def crash_after_lease(record: Mapping[str, Any], path: str | Path) -> None:
        original_writer(record, path)
        if Path(path).name == "phase-lease-v2.json":
            raise RuntimeError("lease-only fixture")

    monkeypatch.setattr(
        databricks_v1,
        "_write_canonical_exclusive",
        crash_after_lease,
    )
    with pytest.raises(RuntimeError, match="lease-only fixture"):
        databricks_v2.submit_gpu_qualification_jobs_v2(
            case.config,
            **case.call_kwargs(),
        )
    monkeypatch.setattr(
        databricks_v1,
        "_write_canonical_exclusive",
        original_writer,
    )
    lease_path = case.receipt_root / "phase-lease-v2.json"
    lease = _json(lease_path)
    lease["record_type"] = databricks_v1._QUALIFICATION_PHASE_LEASE_RECORD_TYPE
    databricks_v1._seal_record(lease)
    lease_path.write_text(
        canonical_gpu_qualification_json(lease) + "\n",
        encoding="utf-8",
    )
    ledger_before = case.ledger_path.read_bytes()
    root_before = _root_bytes(case.receipt_root)

    with pytest.raises(ValueError, match="phase lease differs"):
        databricks_v2.resume_gpu_qualification_job_submissions_v2(
            case.config,
            **case.call_kwargs(),
        )
    assert case.ledger_path.read_bytes() == ledger_before
    assert _root_bytes(case.receipt_root) == root_before
    assert case.cloud.post_attempt_ids == []
    assert case.cloud.resume_attempt_ids == []


@pytest.mark.parametrize(
    ("suffix_kind", "error_pattern"),
    (
        ("receipt", "receipt"),
        ("terminal", "terminal"),
        ("terminal-count", "more than one active cloud run"),
    ),
)
def test_resume_rejects_noncanonical_batch_progress_prefix_without_writes(
    suffix_kind: str,
    error_pattern: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resequenced post-batch slice must never be normalized by resume."""

    case = _case(tmp_path, monkeypatch)
    _interrupt_after_sixth_ledger_receipt(case, monkeypatch)
    ledger = ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
    opening = GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX
    if suffix_kind == "receipt":
        receipt_items = list(ledger.submission_receipts)
        start = opening.submission_receipt_count
        receipt_items[start], receipt_items[start + 1] = (
            receipt_items[start + 1],
            receipt_items[start],
        )
        changed = replace(ledger, submission_receipts=tuple(receipt_items))
    elif suffix_kind == "terminal":
        terminal_items = list(ledger.terminal_actuals)
        start = opening.terminal_actual_count
        terminal_items[start], terminal_items[start + 1] = (
            terminal_items[start + 1],
            terminal_items[start],
        )
        changed = replace(ledger, terminal_actuals=tuple(terminal_items))
    else:
        changed = replace(ledger, terminal_actuals=ledger.terminal_actuals[:-1])
    _write_ledger(case.ledger_path, changed)
    ledger_before = case.ledger_path.read_bytes()
    root_before = _root_bytes(case.receipt_root)

    def forbidden_resume(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        pytest.fail("noncanonical ledger receipt order reached cloud recovery")

    monkeypatch.setattr(
        databricks_v2,
        "resume_pre_reserved_databricks_run",
        forbidden_resume,
    )
    with pytest.raises(ValueError, match=error_pattern):
        databricks_v2.resume_gpu_qualification_job_submissions_v2(
            case.config,
            **case.call_kwargs(),
        )
    assert case.ledger_path.read_bytes() == ledger_before
    assert _root_bytes(case.receipt_root) == root_before


def test_resume_rejects_terminal_progress_ahead_of_controller_receipt_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    _interrupt_after_sixth_ledger_receipt(case, monkeypatch)
    sixth_attempt = case.cloud.attempt_ids[5]
    sixth_run_id = str(case.cloud._run_id(sixth_attempt))
    ledger_api.record_databricks_verified_run_terminal_actual_json(
        case.ledger_path,
        attempt_id=sixth_attempt,
        run_record=case.cloud.runs[sixth_run_id],
    )
    opening = GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX
    drifted = ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
    assert len(drifted.submission_receipts) == opening.submission_receipt_count + 6
    assert len(drifted.terminal_actuals) == opening.terminal_actual_count + 6
    assert len(list(case.receipt_root.glob("*.json"))) == 7
    ledger_before = case.ledger_path.read_bytes()
    root_before = _root_bytes(case.receipt_root)
    post_calls = list(case.cloud.post_attempt_ids)
    get_calls = list(case.cloud.get_run_ids)

    with pytest.raises(ValueError, match="controller receipt/terminal progress"):
        databricks_v2.resume_gpu_qualification_job_submissions_v2(
            case.config,
            **case.call_kwargs(),
        )

    assert case.ledger_path.read_bytes() == ledger_before
    assert _root_bytes(case.receipt_root) == root_before
    assert case.cloud.post_attempt_ids == post_calls
    assert case.cloud.get_run_ids == get_calls


@pytest.mark.parametrize(
    "tamper",
    (
        "extra",
        "future-receipt",
        "missing-prefix-receipt",
        "missing-marker",
        "multiple-intents",
        "wrong-receipt-semantics",
        "wrong-intent-index",
        "wrong-intent-semantics",
        "next-intent-before-terminal",
        "symlink",
    ),
)
def test_resume_rejects_noncanonical_partial_root_before_any_write(
    tamper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    _interrupt_after_sixth_ledger_receipt(case, monkeypatch)
    job_ids = _planned_job_ids(case.plan)
    current_intent = case.receipt_root / f"{job_ids[5]}.post-intent-v2"
    if tamper == "extra":
        (case.receipt_root / "unexpected.json").write_text(
            "{}\n", encoding="utf-8"
        )
    elif tamper == "future-receipt":
        (case.receipt_root / f"{job_ids[8]}.json").write_bytes(
            (case.receipt_root / f"{job_ids[0]}.json").read_bytes()
        )
    elif tamper == "missing-prefix-receipt":
        (case.receipt_root / f"{job_ids[2]}.json").unlink()
    elif tamper == "missing-marker":
        (case.receipt_root / "batch-reserved-v2.json").unlink()
    elif tamper == "multiple-intents":
        (case.receipt_root / f"{job_ids[6]}.post-intent-v2").write_bytes(
            current_intent.read_bytes()
        )
    elif tamper == "wrong-receipt-semantics":
        receipt_path = case.receipt_root / f"{job_ids[0]}.json"
        receipt = _json(receipt_path)
        receipt["output_json"] = "dbfs:/Volumes/forged/output.json"
        receipt["closed_record_sha256"] = ""
        databricks_v2._seal_controller_record_v2(receipt)
        receipt_path.write_text(
            canonical_gpu_qualification_json(receipt) + "\n",
            encoding="utf-8",
        )
    elif tamper == "wrong-intent-index":
        current_intent.rename(
            case.receipt_root / f"{job_ids[7]}.post-intent-v2"
        )
    elif tamper == "wrong-intent-semantics":
        intent = _json(current_intent)
        intent["state"] = "forged-but-resealed"
        intent["closed_record_sha256"] = ""
        databricks_v2._seal_controller_record_v2(intent)
        current_intent.write_text(
            canonical_gpu_qualification_json(intent) + "\n",
            encoding="utf-8",
        )
    elif tamper == "next-intent-before-terminal":
        _validated_plan, _validated_pins, _payloads, contracts = (
            databricks_v2._validated_controller_contract_v2(
                plan_record=case.plan,
                single_user_name=_SINGLE_USER_NAME,
                artifact_uris=case.artifact_uris,
                output_root=_OUTPUT_ROOT,
            )
        )
        ledger = ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
        marker = _json(case.receipt_root / "batch-reserved-v2.json")
        sixth_receipt = databricks_v2._submit_receipt_record_v2(
            plan=case.plan,
            contract=contracts[5],
            ledger=ledger,
            phase_batch_record_sha256=str(marker["closed_record_sha256"]),
        )
        (case.receipt_root / f"{job_ids[5]}.json").write_text(
            canonical_gpu_qualification_json(sixth_receipt) + "\n",
            encoding="utf-8",
        )
        current_intent.unlink()
        authorization = (
            ledger_api.replay_databricks_run_attempt_batch_authorization_json(
                case.ledger_path,
                databricks_v2._batch_requests_v2(case.plan, contracts),
                expected_predecessor_prefix=case.phase_predecessor,
            )
        )
        next_intent = databricks_v2._post_intent_record_v2(
            contract=contracts[6],
            batch_authorization=authorization,
            phase_batch_record_sha256=str(marker["closed_record_sha256"]),
        )
        (case.receipt_root / f"{job_ids[6]}.post-intent-v2").write_text(
            canonical_gpu_qualification_json(next_intent) + "\n",
            encoding="utf-8",
        )
    else:
        (case.receipt_root / "unexpected-link").symlink_to(
            "phase-lease-v2.json"
        )

    ledger_before = case.ledger_path.read_bytes()
    ledger_stat = _stable_stat(case.ledger_path)
    root_before = _root_snapshot(case.receipt_root)
    post_calls = list(case.cloud.post_attempt_ids)
    resume_calls = list(case.cloud.resume_attempt_ids)

    def forbidden_resume(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        pytest.fail("noncanonical v2 evidence reached cloud recovery")

    monkeypatch.setattr(
        databricks_v2,
        "resume_pre_reserved_databricks_run",
        forbidden_resume,
    )
    with pytest.raises(ValueError):
        databricks_v2.resume_gpu_qualification_job_submissions_v2(
            case.config,
            **case.call_kwargs(),
        )
    assert case.ledger_path.read_bytes() == ledger_before
    assert _stable_stat(case.ledger_path) == ledger_stat
    assert _root_snapshot(case.receipt_root) == root_before
    assert case.cloud.post_attempt_ids == post_calls
    assert case.cloud.resume_attempt_ids == resume_calls


def test_v2_collector_rejects_terminal_control_plane_drift_before_result_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    _payloads, contracts, runs, _results = (
        _completed_submission_with_terminal_fixtures(case)
    )
    first_run_id = str(case.cloud._run_id(case.cloud.attempt_ids[0]))
    runs[first_run_id]["run_name"] = "forged-first-run-name"
    observed_gets: list[str] = []

    def get_run(
        config: DatabricksWorkspaceConfig,
        run_id: str,
    ) -> dict[str, Any]:
        assert config is case.config
        observed_gets.append(run_id)
        return deepcopy(runs[run_id])

    monkeypatch.setattr(databricks_v2, "get_databricks_run", get_run)
    evidence_root = tmp_path / "qualification-evidence-v2"

    with pytest.raises(
        ValueError, match="terminal runs/get reconciliation differs from the ledger"
    ):
        databricks_v2.collect_gpu_qualification_evidence_v2(
            case.config,
            **_collector_kwargs(case, evidence_root),
        )

    assert observed_gets == [
        str(case.cloud._run_id(attempt_id))
        for attempt_id in case.cloud.attempt_ids
    ]
    ledger = ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
    opening = GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX
    assert len(ledger.terminal_actuals) == opening.terminal_actual_count + 14
    assert tuple(
        actual.attempt_id
        for actual in ledger.terminal_actuals[opening.terminal_actual_count :]
    ) == tuple(contract["reservation_attempt_id"] for contract in contracts)
    assert ledger.active_reserved_cluster_hours == 0.0
    assert not evidence_root.exists()
    assert not list(tmp_path.glob(".qualification-evidence-v2.staging-*"))


def test_v2_collector_resumes_canonical_partial_terminal_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    _payloads, contracts, runs, results = (
        _completed_submission_with_terminal_fixtures(case)
    )
    opening = GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX
    completed_submission = ledger_api.read_databricks_cluster_hour_ledger_json(
        case.ledger_path
    )
    _write_ledger(
        case.ledger_path,
        replace(
            completed_submission,
            terminal_actuals=completed_submission.terminal_actuals[:-1],
        ),
    )
    last_run_id = str(case.cloud._run_id(case.cloud.attempt_ids[-1]))
    terminal_last_run = deepcopy(runs[last_run_id])
    runs[last_run_id]["state"] = {
        "life_cycle_state": "RUNNING",
        "result_state": None,
    }
    monkeypatch.setattr(
        databricks_v2,
        "get_databricks_run",
        lambda config, run_id: deepcopy(runs[run_id]),
    )
    selection = _install_minimal_governed_evidence_stubs(
        case,
        results,
        monkeypatch,
    )
    evidence_root = tmp_path / "qualification-evidence-v2"

    with pytest.raises(ValueError, match="runs/get response is not terminal"):
        databricks_v2.collect_gpu_qualification_evidence_v2(
            case.config,
            **_collector_kwargs(case, evidence_root),
        )

    partial = ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
    partial_suffix = partial.terminal_actuals[opening.terminal_actual_count :]
    assert tuple(actual.attempt_id for actual in partial_suffix) == tuple(
        contract["reservation_attempt_id"] for contract in contracts[:-1]
    )
    assert len(partial_suffix) == 13
    assert partial.active_reserved_cluster_hours > 0.0
    assert not evidence_root.exists()

    runs[last_run_id] = terminal_last_run
    evidence, authorization = databricks_v2.collect_gpu_qualification_evidence_v2(
        case.config,
        **_collector_kwargs(case, evidence_root),
        now=lambda: datetime(2026, 8, 26, 12, 30, tzinfo=UTC),
    )

    completed = ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
    completed_suffix = completed.terminal_actuals[opening.terminal_actual_count :]
    assert tuple(actual.attempt_id for actual in completed_suffix) == tuple(
        contract["reservation_attempt_id"] for contract in contracts
    )
    assert len(completed_suffix) == 14
    assert completed.active_reserved_cluster_hours == 0.0
    assert authorization.selection == selection
    assert _json(
        evidence_root / databricks_v2.GPU_QUALIFICATION_V2_EVIDENCE_FILENAME
    ) == evidence
    assert len(list(evidence_root.glob("*.terminal-receipt-v2.json"))) == 14


@pytest.mark.parametrize("retained_live_ledger", (False, True))
def test_v2_collector_atomically_publishes_exact14_and_replay_is_read_only(
    retained_live_ledger: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(
        tmp_path,
        monkeypatch,
        retained_live_ledger=retained_live_ledger,
    )
    predecessor = ledger_api.databricks_ledger_prefix(
        ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
    )
    _payloads, _contracts, runs, results = (
        _completed_submission_with_terminal_fixtures(case)
    )
    monkeypatch.setattr(
        databricks_v2,
        "get_databricks_run",
        lambda config, run_id: deepcopy(runs[run_id]),
    )
    selection = _install_minimal_governed_evidence_stubs(
        case,
        results,
        monkeypatch,
    )
    evidence_root = tmp_path / "qualification-evidence-v2"

    evidence, authorization = databricks_v2.collect_gpu_qualification_evidence_v2(
        case.config,
        **_collector_kwargs(case, evidence_root),
        now=lambda: datetime(2026, 8, 26, 12, 0, tzinfo=UTC),
    )

    job_ids = _planned_job_ids(case.plan)
    assert {path.name for path in evidence_root.iterdir()} == {
        databricks_v2.GPU_QUALIFICATION_V2_EVIDENCE_FILENAME,
        *(
            f"{job_id}.terminal-receipt-v2.json"
            for job_id in job_ids
        ),
    }
    assert len(list(evidence_root.glob("*.terminal-receipt-v2.json"))) == 14
    results_by_job_id = {str(result["job_id"]): result for result in results.values()}
    for planned_job in case.plan["cloud_qualification"]["jobs"]:
        job_id = str(planned_job["job_id"])
        receipt = _json(evidence_root / f"{job_id}.terminal-receipt-v2.json")
        _assert_v2_seal(receipt)
        assert (
            qualification_v2._validate_terminal_receipt_v2_original(
                receipt,
                result=results_by_job_id[job_id],
                planned_job=planned_job,
                plan_record=case.plan,
            )
            == receipt
        )
    assert _json(
        evidence_root / databricks_v2.GPU_QUALIFICATION_V2_EVIDENCE_FILENAME
    ) == evidence
    assert isinstance(
        authorization,
        databricks_v1.GPUQualificationLaunchAuthorization,
    )
    assert authorization.selection == selection
    assert authorization.predecessor_prefix == predecessor
    ledger = ledger_api.read_databricks_cluster_hour_ledger_json(case.ledger_path)
    assert ledger.active_reserved_cluster_hours == 0.0
    assert len(ledger.terminal_actuals) == (
        predecessor.terminal_actual_count + 14
    )
    assert not list(tmp_path.glob(".qualification-evidence-v2.staging-*"))

    ledger_before = case.ledger_path.read_bytes()
    ledger_stat = _stable_stat(case.ledger_path)
    evidence_before = _root_snapshot(evidence_root)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("durable v2 replay performed a cloud or reconciliation write")

    monkeypatch.setattr(databricks_v2, "get_databricks_run", forbidden)
    monkeypatch.setattr(
        databricks_v2,
        "record_databricks_verified_run_terminal_actual_json",
        forbidden,
    )
    monkeypatch.setattr(
        databricks_v1,
        "_read_gpu_qualification_result",
        forbidden,
    )
    replayed = databricks_v2.replay_gpu_qualification_launch_authorization_v2(
        config=case.config,
        **_collector_kwargs(case, evidence_root),
        expected_campaign_id=PUBLICATION_CAMPAIGN_ID,
        expected_artifact_pins=_pins(),
    )

    assert replayed == authorization
    assert case.ledger_path.read_bytes() == ledger_before
    assert _stable_stat(case.ledger_path) == ledger_stat
    assert _root_snapshot(evidence_root) == evidence_before


def test_v2_replay_rejects_partial_terminal_closure_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    _payloads, _contracts, runs, results = (
        _completed_submission_with_terminal_fixtures(case)
    )
    monkeypatch.setattr(
        databricks_v2,
        "get_databricks_run",
        lambda config, run_id: deepcopy(runs[run_id]),
    )
    _install_minimal_governed_evidence_stubs(case, results, monkeypatch)
    evidence_root = tmp_path / "qualification-evidence-v2"
    databricks_v2.collect_gpu_qualification_evidence_v2(
        case.config,
        **_collector_kwargs(case, evidence_root),
    )
    missing = evidence_root / (
        f"{_planned_job_ids(case.plan)[-1]}.terminal-receipt-v2.json"
    )
    missing.unlink()
    ledger_before = case.ledger_path.read_bytes()
    surviving_root = _root_snapshot(evidence_root)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("partial v2 replay reached cloud or reconciliation")

    monkeypatch.setattr(databricks_v2, "get_databricks_run", forbidden)
    monkeypatch.setattr(
        databricks_v2,
        "record_databricks_verified_run_terminal_actual_json",
        forbidden,
    )
    with pytest.raises(ValueError, match="exact terminal closure"):
        databricks_v2.replay_gpu_qualification_launch_authorization_v2(
            config=case.config,
            **_collector_kwargs(case, evidence_root),
            expected_campaign_id=PUBLICATION_CAMPAIGN_ID,
            expected_artifact_pins=_pins(),
        )

    assert case.ledger_path.read_bytes() == ledger_before
    assert _root_snapshot(evidence_root) == surviving_root
