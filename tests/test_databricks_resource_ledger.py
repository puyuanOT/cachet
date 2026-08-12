import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from document_kv_cache.canary_orchestration import (
    REPRESENTATIVE_CANARY_AGGREGATE_CLUSTER_HOUR_CAP,
    REPRESENTATIVE_CANARY_JOB_COUNT,
    create_representative_canary_cluster_hour_ledger,
)
from document_kv_cache.databricks_resource_ledger import (
    DATABRICKS_CLUSTER_HOUR_LEDGER_RECORD_TYPE,
    DatabricksClusterHourLedger,
    DatabricksClusterHourReservation,
    databricks_cluster_hour_ledger_from_record,
    databricks_cluster_hour_ledger_to_record,
    databricks_submit_payload_reservation,
    main,
    read_databricks_cluster_hour_ledger_json,
    record_databricks_run_terminal_actual_json,
    reserve_databricks_run_attempt_json,
)


def _submit_payload(
    *,
    timeout_seconds=14400,
    task_count=1,
    max_retries=0,
    secret=None,
):
    tasks = []
    for index in range(task_count):
        cluster = {"num_workers": 0, "node_type_id": "g6.8xlarge"}
        if secret is not None:
            cluster["spark_env_vars"] = {"DATABRICKS_TOKEN": secret}
        tasks.append(
            {
                "task_key": f"canary-{index}",
                "timeout_seconds": timeout_seconds,
                "max_retries": max_retries,
                "new_cluster": cluster,
                "spark_python_task": {
                    "python_file": "dbfs:/cachet/run.py",
                    "parameters": ["--output-json", "/Volumes/c/s/v/result.json"],
                },
            }
        )
    return {
        "run_name": "representative-canary",
        "timeout_seconds": timeout_seconds,
        "tasks": tasks,
    }


def test_submit_payload_reservation_is_bounded_digest_only_metadata(tmp_path):
    secret = "dapi" + "0123456789abcdef0123456789abcdef"
    payload = _submit_payload(task_count=2, secret=secret)

    reservation = databricks_submit_payload_reservation(
        payload,
        attempt_id="attempt-001",
        workload_id="representative-vllm",
    )
    ledger = DatabricksClusterHourLedger(ledger_id="canary-2026").reserve(reservation)
    record_text = json.dumps(
        databricks_cluster_hour_ledger_to_record(ledger),
        sort_keys=True,
    )

    assert reservation.task_timeout_seconds == (14400, 14400)
    assert reservation.reserved_cluster_hours == 8.0
    assert len(reservation.submit_payload_sha256) == 64
    assert secret not in record_text
    assert "spark_env_vars" not in record_text
    assert "parameters" not in record_text

    ledger_path = tmp_path / "cluster-hours.json"
    create_representative_canary_cluster_hour_ledger(
        ledger_path,
        ledger_id="representative-canary-2026",
    )
    reserve_databricks_run_attempt_json(
        ledger_path,
        payload,
        attempt_id="attempt-001",
        workload_id="representative-vllm",
    )
    persisted = ledger_path.read_text(encoding="utf-8")
    assert secret not in persisted
    assert "spark_env_vars" not in persisted
    assert "parameters" not in persisted


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.pop("timeout_seconds"), "run_timeout_seconds"),
        (
            lambda payload: payload.__setitem__("timeout_seconds", 14401),
            "run_timeout_seconds",
        ),
        (
            lambda payload: payload["tasks"][0].__setitem__("timeout_seconds", 0),
            "run_timeout_seconds",
        ),
        (
            lambda payload: payload["tasks"][0].__setitem__("max_retries", 1),
            "task_max_retries",
        ),
        (
            lambda payload: payload["tasks"][0].pop("new_cluster"),
            "explicit new_cluster",
        ),
    ),
)
def test_submit_payload_reservation_rejects_unbounded_payloads(
    mutation,
    message,
):
    payload = _submit_payload()
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        databricks_submit_payload_reservation(
            payload,
            attempt_id="attempt-001",
            workload_id="representative-vllm",
        )


def test_representative_canary_ledger_reserves_ten_attempts_then_reconciles_actual(
    tmp_path,
):
    ledger_path = tmp_path / "cluster-hours.json"
    ledger = create_representative_canary_cluster_hour_ledger(
        ledger_path,
        ledger_id="representative-canary-2026",
    )

    assert REPRESENTATIVE_CANARY_JOB_COUNT == 10
    assert REPRESENTATIVE_CANARY_AGGREGATE_CLUSTER_HOUR_CAP == 120.0
    assert ledger.cap_cluster_hours == 120.0
    for index in range(REPRESENTATIVE_CANARY_JOB_COUNT):
        ledger = reserve_databricks_run_attempt_json(
            ledger_path,
            _submit_payload(),
            attempt_id=f"attempt-{index:02d}",
            workload_id=f"representative-job-{index:02d}",
        )

    assert ledger.active_reserved_cluster_hours == 40.0
    assert ledger.remaining_cluster_hours == 80.0

    ledger = record_databricks_run_terminal_actual_json(
        ledger_path,
        attempt_id="attempt-00",
        terminal_state="succeeded",
        actual_cluster_duration_seconds=3600,
    )

    assert ledger.active_reserved_cluster_hours == 36.0
    assert ledger.terminal_actual_cluster_hours == 1.0
    assert ledger.accounted_cluster_hours == 37.0
    assert read_databricks_cluster_hour_ledger_json(ledger_path) == ledger


def test_ledger_rejects_reservation_over_cap_and_duplicate_attempt(tmp_path):
    ledger_path = tmp_path / "cluster-hours.json"
    create_representative_canary_cluster_hour_ledger(
        ledger_path,
        ledger_id="representative-canary-2026",
    )
    for wave in range(3):
        for index in range(10):
            reserve_databricks_run_attempt_json(
                ledger_path,
                _submit_payload(),
                attempt_id=f"attempt-{wave}-{index:02d}",
                workload_id="representative-vllm",
            )
        assert read_databricks_cluster_hour_ledger_json(
            ledger_path
        ).accounted_cluster_hours == float((wave + 1) * 40)

    with pytest.raises(ValueError, match="would exceed.*cluster-hour cap"):
        reserve_databricks_run_attempt_json(
            ledger_path,
            _submit_payload(),
            attempt_id="attempt-over-cap",
            workload_id="representative-vllm",
        )
    with pytest.raises(ValueError, match="already reserved"):
        reserve_databricks_run_attempt_json(
            ledger_path,
            _submit_payload(),
            attempt_id="attempt-0-00",
            workload_id="representative-vllm",
        )

    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    assert ledger.accounted_cluster_hours == 120.0
    assert ledger.remaining_cluster_hours == 0.0

    ledger = record_databricks_run_terminal_actual_json(
        ledger_path,
        attempt_id="attempt-0-00",
        terminal_state="succeeded",
        actual_cluster_duration_seconds=3600,
    )
    assert ledger.accounted_cluster_hours == 117.0
    assert ledger.remaining_cluster_hours == 3.0
    with pytest.raises(ValueError, match="would exceed.*cluster-hour cap"):
        reserve_databricks_run_attempt_json(
            ledger_path,
            _submit_payload(),
            attempt_id="attempt-still-over-cap",
            workload_id="representative-vllm",
        )

    ledger = record_databricks_run_terminal_actual_json(
        ledger_path,
        attempt_id="attempt-0-01",
        terminal_state="failed",
        actual_cluster_duration_seconds=0,
    )
    assert ledger.remaining_cluster_hours == 7.0
    ledger = reserve_databricks_run_attempt_json(
        ledger_path,
        _submit_payload(),
        attempt_id="attempt-after-reconciliation",
        workload_id="representative-vllm",
    )
    assert ledger.accounted_cluster_hours == 117.0


def test_terminal_actual_is_single_closed_record_bounded_by_reservation(tmp_path):
    ledger_path = tmp_path / "cluster-hours.json"
    create_representative_canary_cluster_hour_ledger(
        ledger_path,
        ledger_id="representative-canary-2026",
    )
    reserve_databricks_run_attempt_json(
        ledger_path,
        _submit_payload(),
        attempt_id="attempt-00",
        workload_id="representative-vllm",
    )

    with pytest.raises(ValueError, match="no pre-submission reservation"):
        record_databricks_run_terminal_actual_json(
            ledger_path,
            attempt_id="unknown-attempt",
            terminal_state="failed",
            actual_cluster_duration_seconds=1,
        )
    with pytest.raises(ValueError, match="exceeds.*reservation"):
        record_databricks_run_terminal_actual_json(
            ledger_path,
            attempt_id="attempt-00",
            terminal_state="timed_out",
            actual_cluster_duration_seconds=14401,
        )

    record_databricks_run_terminal_actual_json(
        ledger_path,
        attempt_id="attempt-00",
        terminal_state="failed",
        actual_cluster_duration_seconds=120,
    )
    with pytest.raises(ValueError, match="already has a terminal actual"):
        record_databricks_run_terminal_actual_json(
            ledger_path,
            attempt_id="attempt-00",
            terminal_state="failed",
            actual_cluster_duration_seconds=120,
        )


def test_ledger_records_are_frozen_closed_and_canonically_recomputed():
    reservation = databricks_submit_payload_reservation(
        _submit_payload(),
        attempt_id="attempt-001",
        workload_id="representative-vllm",
    )
    ledger = DatabricksClusterHourLedger(ledger_id="canary-2026").reserve(reservation)
    record = databricks_cluster_hour_ledger_to_record(ledger)

    assert record["record_type"] == DATABRICKS_CLUSTER_HOUR_LEDGER_RECORD_TYPE
    assert databricks_cluster_hour_ledger_from_record(record) == ledger
    with pytest.raises(FrozenInstanceError):
        reservation.attempt_id = "changed"

    unknown = deepcopy(record)
    unknown["raw_output"] = "must never be accepted"
    with pytest.raises(ValueError, match="closed schema"):
        databricks_cluster_hour_ledger_from_record(unknown)

    forged_accounting = deepcopy(record)
    forged_accounting["accounting"]["remaining_cluster_hours"] = 120.0
    with pytest.raises(ValueError, match="does not match immutable ledger events"):
        databricks_cluster_hour_ledger_from_record(forged_accounting)

    forged_reservation = deepcopy(record)
    forged_reservation["reservations"][0]["reserved_cluster_hours"] = 0.0
    with pytest.raises(ValueError, match="not canonical"):
        databricks_cluster_hour_ledger_from_record(forged_reservation)


def test_ledger_cli_persists_reservation_and_terminal_actual(tmp_path, capsys):
    ledger_path = tmp_path / "cluster-hours.json"
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(_submit_payload()), encoding="utf-8")

    assert (
        main(
            [
                "init",
                "--ledger-json",
                str(ledger_path),
                "--ledger-id",
                "representative-canary-2026",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "reserve",
                "--ledger-json",
                str(ledger_path),
                "--submit-payload-json",
                str(payload_path),
                "--attempt-id",
                "attempt-001",
                "--workload-id",
                "representative-vllm",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "terminal",
                "--ledger-json",
                str(ledger_path),
                "--attempt-id",
                "attempt-001",
                "--terminal-state",
                "succeeded",
                "--actual-cluster-duration-seconds",
                "60",
            ]
        )
        == 0
    )

    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    assert ledger.terminal_actuals[0].actual_cluster_duration_seconds == 60.0
    assert "raw_output" not in capsys.readouterr().out


def test_concurrent_duplicate_reservation_is_serialized_by_file_lock(tmp_path):
    ledger_path = tmp_path / "cluster-hours.json"
    create_representative_canary_cluster_hour_ledger(
        ledger_path,
        ledger_id="representative-canary-2026",
    )

    def reserve_once():
        try:
            reserve_databricks_run_attempt_json(
                ledger_path,
                _submit_payload(),
                attempt_id="attempt-concurrent",
                workload_id="representative-vllm",
            )
        except ValueError as exc:
            return str(exc)
        return "reserved"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _index: reserve_once(), range(2)))

    assert sorted(outcomes) == [
        "attempt 'attempt-concurrent' is already reserved",
        "reserved",
    ]
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    assert len(ledger.reservations) == 1
    assert ledger.accounted_cluster_hours == 4.0


def test_identifier_fields_reject_databricks_credentials():
    with pytest.raises(ValueError, match="credential"):
        DatabricksClusterHourReservation(
            attempt_id="dapi" + "0123456789abcdef0123456789abcdef",
            workload_id="representative-vllm",
            submit_payload_sha256="0" * 64,
            run_timeout_seconds=14400,
            task_timeout_seconds=(14400,),
        )
