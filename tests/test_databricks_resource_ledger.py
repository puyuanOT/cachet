import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

import document_kv_cache.databricks_resource_ledger as resource_ledger
from document_kv_cache.canary_orchestration import (
    REPRESENTATIVE_CANARY_AGGREGATE_CLUSTER_HOUR_CAP,
    REPRESENTATIVE_CANARY_JOB_COUNT,
    create_representative_canary_cluster_hour_ledger,
)
from document_kv_cache.databricks_resource_ledger import (
    DATABRICKS_CLUSTER_HOUR_LEDGER_RECORD_TYPE,
    DATABRICKS_CLUSTER_HOUR_LEDGER_SCHEMA_VERSION,
    MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS,
    MAX_DATABRICKS_ACTIVE_RESERVED_TASKS,
    MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
    DatabricksClusterHourLedger,
    DatabricksClusterHourReservation,
    DatabricksClusterHourTerminalActual,
    DatabricksLedgerPrefix,
    DatabricksRunAttemptReservationRequest,
    DatabricksRunSubmissionReceipt,
    create_databricks_cluster_hour_ledger_json,
    databricks_cluster_hour_ledger_from_record,
    databricks_cluster_hour_ledger_to_record,
    databricks_ledger_prefix,
    databricks_ledger_prefix_from_record,
    databricks_submit_payload_reservation,
    main,
    raise_databricks_cluster_hour_ledger_cap_json,
    read_databricks_cluster_hour_ledger_json,
    record_databricks_run_submission_receipt_json,
    record_databricks_run_terminal_actual_json,
    record_databricks_verified_run_terminal_actual_json,
    require_databricks_ledger_prefix,
    require_databricks_batch_terminal_closure,
    reserve_databricks_run_attempt_batch_authorized_json,
    reserve_databricks_run_attempt_batch_json,
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


def _terminal_run(*, run_id: int = 101, task_count: int = 1):
    tasks = []
    for index in range(task_count):
        start_time = 1_000_000 + index * 200_000
        tasks.append(
            {
                "end_time": start_time + (index + 1) * 60_000,
                "run_id": run_id * 100 + index,
                "start_time": start_time,
                "state": {
                    "life_cycle_state": "TERMINATED",
                    "result_state": "SUCCESS",
                },
                "task_key": f"canary-{index}",
            }
        )
    return {
        "end_time": max(task["end_time"] for task in tasks) + 1_000,
        "run_id": run_id,
        "start_time": min(task["start_time"] for task in tasks) - 1_000,
        "state": {
            "life_cycle_state": "TERMINATED",
            "result_state": "SUCCESS",
        },
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
    assert ledger.active_reserved_task_count == 2
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


def test_generic_publication_ledger_allows_the_frozen_1024_hour_campaign_cap():
    ledger = DatabricksClusterHourLedger(
        ledger_id="vllm-0271-publication-campaign",
        cap_cluster_hours=1024.0,
    )

    assert MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS == 1024.0
    assert MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS == 900.0
    assert ledger.cap_cluster_hours == 1024.0
    assert ledger.remaining_cluster_hours == 1024.0

    with pytest.raises(ValueError, match="no greater than 1024.0"):
        DatabricksClusterHourLedger(
            ledger_id="over-budget-campaign",
            cap_cluster_hours=1024.000001,
        )


def test_publication_ledger_preserves_124_hours_of_unreserved_headroom():
    ledger = DatabricksClusterHourLedger(
        ledger_id="vllm-0271-publication-campaign",
        cap_cluster_hours=1024.0,
    )
    for index in range(75):
        attempt_id = f"publication-history-{index:02d}"
        ledger = ledger.reserve(
            DatabricksClusterHourReservation(
                attempt_id=attempt_id,
                workload_id="vllm-0271-publication-campaign",
                submit_payload_sha256=f"{index:064x}",
                run_timeout_seconds=43_200,
                task_timeout_seconds=(43_200,),
            )
        ).record_terminal_actual(
            DatabricksClusterHourTerminalActual(
                attempt_id=attempt_id,
                terminal_state="succeeded",
                actual_cluster_duration_seconds=43_200,
            )
        )

    assert ledger.terminal_actual_cluster_hours == 900.0
    assert ledger.active_reserved_cluster_hours == 0.0
    assert ledger.remaining_cluster_hours == 124.0

    with pytest.raises(ValueError, match="protected 124-hour"):
        ledger.reserve(
            DatabricksClusterHourReservation(
                attempt_id="publication-wave-overflow",
                workload_id="vllm-0271-publication-campaign",
                submit_payload_sha256="b" * 64,
                run_timeout_seconds=43_200,
                task_timeout_seconds=(43_200,),
            )
        )


def test_cap_raise_preserves_the_existing_append_only_gpu_hour_balance(tmp_path):
    ledger_path = tmp_path / "cluster-hours.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="vllm-publication-history",
        cap_cluster_hours=120.0,
    )
    reserve_databricks_run_attempt_json(
        ledger_path,
        _submit_payload(),
        attempt_id="historical-attempt",
        workload_id="historical-vllm",
    )
    before = record_databricks_run_terminal_actual_json(
        ledger_path,
        attempt_id="historical-attempt",
        terminal_state="succeeded",
        actual_cluster_duration_seconds=3600.0,
    )

    after = raise_databricks_cluster_hour_ledger_cap_json(
        ledger_path,
        expected_current_cap_cluster_hours=120.0,
        new_cap_cluster_hours=1024.0,
    )

    assert after.cap_cluster_hours == 1024.0
    assert after.reservations == before.reservations
    assert after.terminal_actuals == before.terminal_actuals
    assert after.accounted_cluster_hours == before.accounted_cluster_hours == 1.0
    assert after.remaining_cluster_hours == 1023.0
    assert read_databricks_cluster_hour_ledger_json(ledger_path) == after
    with pytest.raises(ValueError, match="differs from the expected migration source"):
        raise_databricks_cluster_hour_ledger_cap_json(
            ledger_path,
            expected_current_cap_cluster_hours=120.0,
            new_cap_cluster_hours=1024.0,
        )


def test_cap_raise_rejects_an_active_reservation(tmp_path):
    ledger_path = tmp_path / "cluster-hours.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="vllm-publication-history",
        cap_cluster_hours=120.0,
    )
    reserve_databricks_run_attempt_json(
        ledger_path,
        _submit_payload(),
        attempt_id="still-running",
        workload_id="historical-vllm",
    )

    with pytest.raises(ValueError, match="zero active reservations"):
        raise_databricks_cluster_hour_ledger_cap_json(
            ledger_path,
            expected_current_cap_cluster_hours=120.0,
            new_cap_cluster_hours=1024.0,
        )


def test_ledger_prefix_round_trips_and_accepts_only_ordered_extensions():
    first = databricks_submit_payload_reservation(
        _submit_payload(),
        attempt_id="prefix-attempt-001",
        workload_id="publication-wave-001",
    )
    second = databricks_submit_payload_reservation(
        _submit_payload(),
        attempt_id="prefix-attempt-002",
        workload_id="publication-wave-002",
    )
    third = databricks_submit_payload_reservation(
        _submit_payload(),
        attempt_id="prefix-attempt-003",
        workload_id="publication-wave-003",
    )
    authorized = DatabricksClusterHourLedger(
        ledger_id="publication-campaign",
        reservations=(first, second),
    )
    expected = databricks_ledger_prefix(authorized)
    extended = authorized.reserve(third)

    assert databricks_ledger_prefix_from_record(expected.to_record()) == expected
    assert require_databricks_ledger_prefix(extended, expected) == (
        databricks_ledger_prefix(extended)
    )

    reordered = DatabricksClusterHourLedger(
        ledger_id="publication-campaign",
        reservations=(second, first, third),
    )
    with pytest.raises(ValueError, match="ordered prefix"):
        require_databricks_ledger_prefix(reordered, expected)

    divergent_second = databricks_submit_payload_reservation(
        _submit_payload(),
        attempt_id="prefix-attempt-002",
        workload_id="publication-wave-divergent",
    )
    divergent_history = DatabricksClusterHourLedger(
        ledger_id="publication-campaign",
        reservations=(first, divergent_second, third),
    )
    with pytest.raises(ValueError, match="ordered prefix"):
        require_databricks_ledger_prefix(divergent_history, expected)

    unknown = expected.to_record()
    unknown["not_closed"] = True
    with pytest.raises(ValueError, match="closed schema"):
        databricks_ledger_prefix_from_record(unknown)


def test_ledger_prefix_rejects_same_id_fresh_ledger_reset():
    reservation = databricks_submit_payload_reservation(
        _submit_payload(),
        attempt_id="historical-attempt",
        workload_id="publication-history",
    )
    historical = DatabricksClusterHourLedger(
        ledger_id="publication-campaign",
        reservations=(reservation,),
    )
    expected = databricks_ledger_prefix(historical)
    same_id_fresh_reset = DatabricksClusterHourLedger(
        ledger_id="publication-campaign",
    )

    with pytest.raises(ValueError, match="shorter than its authorized prefix"):
        require_databricks_ledger_prefix(same_id_fresh_reset, expected)

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        DatabricksLedgerPrefix(
            ledger_id=expected.ledger_id,
            cap_cluster_hours=expected.cap_cluster_hours,
            reservation_count=expected.reservation_count,
            submission_receipt_count=expected.submission_receipt_count,
            terminal_actual_count=expected.terminal_actual_count,
            prefix_sha256=expected.prefix_sha256.upper(),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.pop("timeout_seconds"), "run_timeout_seconds"),
        (
            lambda payload: payload.__setitem__("timeout_seconds", 43201),
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
        for index in range(10):
            if wave == 0 and index in {0, 1}:
                continue
            record_databricks_run_terminal_actual_json(
                ledger_path,
                attempt_id=f"attempt-{wave}-{index:02d}",
                terminal_state="succeeded",
                actual_cluster_duration_seconds=14_400,
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


def test_batch_reservation_is_atomic_ordered_and_rejects_duplicate_attempts(
    tmp_path,
):
    ledger_path = tmp_path / "cluster-hours.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-campaign",
    )
    requests = tuple(
        DatabricksRunAttemptReservationRequest(
            attempt_id=f"wave-attempt-{index}",
            workload_id=f"wave-workload-{index}",
            submit_payload=_submit_payload(timeout_seconds=3600 * index),
        )
        for index in (1, 2, 3)
    )
    validator_observations = []

    def validate_batch(live_ledger, reservations, snapshots):
        validator_observations.append(
            (
                live_ledger.reservations,
                tuple(item.attempt_id for item in reservations),
                tuple(snapshot["timeout_seconds"] for snapshot in snapshots),
            )
        )
        if (
            live_ledger.active_reserved_task_count
            + sum(len(item.task_timeout_seconds) for item in reservations)
            > 3
        ):
            raise ValueError("batch exceeds the campaign task policy")

    updated = reserve_databricks_run_attempt_batch_json(
        ledger_path,
        requests,
        batch_validator=validate_batch,
    )

    assert [item.attempt_id for item in updated.reservations] == [
        "wave-attempt-1",
        "wave-attempt-2",
        "wave-attempt-3",
    ]
    assert validator_observations == [
        (
            (),
            ("wave-attempt-1", "wave-attempt-2", "wave-attempt-3"),
            (3600, 7200, 10800),
        )
    ]

    before_replay = ledger_path.read_bytes()
    with pytest.raises(ValueError, match="already reserved"):
        reserve_databricks_run_attempt_batch_json(ledger_path, requests)
    assert ledger_path.read_bytes() == before_replay

    duplicate_in_one_batch = (
        DatabricksRunAttemptReservationRequest(
            attempt_id="duplicate-attempt",
            workload_id="duplicate-workload-a",
            submit_payload=_submit_payload(timeout_seconds=3600),
        ),
        DatabricksRunAttemptReservationRequest(
            attempt_id="duplicate-attempt",
            workload_id="duplicate-workload-b",
            submit_payload=_submit_payload(timeout_seconds=7200),
        ),
    )
    with pytest.raises(ValueError, match="attempt_id values must be unique"):
        reserve_databricks_run_attempt_batch_json(
            ledger_path,
            duplicate_in_one_batch,
        )
    assert ledger_path.read_bytes() == before_replay


def test_core_ledger_concurrently_rejects_a_seventeenth_active_task(tmp_path):
    ledger_path = tmp_path / "concurrency-ledger.json"
    create_databricks_cluster_hour_ledger_json(ledger_path, ledger_id="campaign")
    payload = _submit_payload(timeout_seconds=3600)

    def reserve(index: int) -> bool:
        try:
            reserve_databricks_run_attempt_json(
                ledger_path,
                payload,
                attempt_id=f"concurrent/{index:02d}",
                workload_id="global-concurrency-guard",
            )
        except ValueError as exc:
            assert "active task concurrency guard" in str(exc)
            return False
        return True

    with ThreadPoolExecutor(max_workers=17) as pool:
        outcomes = tuple(pool.map(reserve, range(17)))

    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    assert sum(outcomes) == MAX_DATABRICKS_ACTIVE_RESERVED_TASKS
    assert ledger.active_reserved_task_count == MAX_DATABRICKS_ACTIVE_RESERVED_TASKS
    assert len(ledger.reservations) == MAX_DATABRICKS_ACTIVE_RESERVED_TASKS

    reservations = tuple(
        databricks_submit_payload_reservation(
            payload,
            attempt_id=f"tampered/{index:02d}",
            workload_id="tampered-ledger",
        )
        for index in range(17)
    )
    with pytest.raises(ValueError, match="active tasks exceed"):
        DatabricksClusterHourLedger(
            ledger_id="campaign",
            reservations=reservations,
        )


def test_batch_terminal_closure_rejects_unrelated_live_suffix(tmp_path):
    ledger_path = tmp_path / "phase-ledger.json"
    opening = create_databricks_cluster_hour_ledger_json(
        ledger_path, ledger_id="campaign"
    )
    requests = tuple(
        DatabricksRunAttemptReservationRequest(
            attempt_id=f"phase/{index}",
            workload_id="publication-phase",
            submit_payload=_submit_payload(timeout_seconds=3600),
        )
        for index in range(2)
    )
    _batch, authorization = reserve_databricks_run_attempt_batch_authorized_json(
        ledger_path,
        requests,
        expected_predecessor_prefix=databricks_ledger_prefix(opening),
    )
    for index in range(2):
        record_databricks_run_submission_receipt_json(
            ledger_path,
            attempt_id=f"phase/{index}",
            submit_response={"run_id": 101 + index},
        )
        record_databricks_verified_run_terminal_actual_json(
            ledger_path,
            attempt_id=f"phase/{index}",
            run_record=_terminal_run(run_id=101 + index),
        )
    terminal = read_databricks_cluster_hour_ledger_json(ledger_path)
    terminal_prefix = require_databricks_batch_terminal_closure(terminal, authorization)
    assert terminal_prefix == databricks_ledger_prefix(terminal)

    class SmuggledBatchAuthorization(
        resource_ledger.DatabricksBatchReservationAuthorization
    ):
        pass

    smuggled = object.__new__(SmuggledBatchAuthorization)
    with pytest.raises(TypeError, match="authorization has the wrong type"):
        require_databricks_batch_terminal_closure(terminal, smuggled)
    with pytest.raises(TypeError, match="DatabricksBatchReservationAuthorization"):
        resource_ledger.require_databricks_batch_reservation_authorization(
            smuggled,
            expected_predecessor_prefix=authorization.predecessor_prefix,
            expected_attempt_ids=authorization.attempt_ids,
            expected_submit_payload_sha256s=authorization.submit_payload_sha256s,
        )
    with pytest.raises(TypeError, match="authorization has the wrong type"):
        resource_ledger.require_databricks_publication_batch_admission(
            terminal, smuggled
        )

    reserve_databricks_run_attempt_json(
        ledger_path,
        _submit_payload(timeout_seconds=3600),
        attempt_id="unrelated/after-phase",
        workload_id="unrelated",
    )
    extended = read_databricks_cluster_hour_ledger_json(ledger_path)
    with pytest.raises(ValueError, match="not the complete live ledger"):
        require_databricks_batch_terminal_closure(extended, authorization)
    assert (
        require_databricks_batch_terminal_closure(
            extended,
            authorization,
            require_complete_current_prefix=False,
        )
        == terminal_prefix
    )


def test_batch_n_minus_one_policy_failure_persists_zero_reservations(tmp_path):
    ledger_path = tmp_path / "cluster-hours.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-campaign",
    )
    requests = tuple(
        DatabricksRunAttemptReservationRequest(
            attempt_id=f"atomic-attempt-{index}",
            workload_id=f"atomic-workload-{index}",
            submit_payload=_submit_payload(timeout_seconds=3600),
        )
        for index in range(3)
    )

    def maximum_two_active_tasks(live_ledger, reservations, snapshots):
        assert len(reservations) == len(snapshots) == 3
        projected_task_count = live_ledger.active_reserved_task_count + sum(
            len(item.task_timeout_seconds) for item in reservations
        )
        if projected_task_count > 2:
            raise ValueError("batch exceeds the campaign task policy")

    with pytest.raises(ValueError, match="campaign task policy"):
        reserve_databricks_run_attempt_batch_json(
            ledger_path,
            requests,
            batch_validator=maximum_two_active_tasks,
        )

    unchanged = read_databricks_cluster_hour_ledger_json(ledger_path)
    assert unchanged.reservations == ()
    assert unchanged.accounted_cluster_hours == 0.0


def test_batch_n_minus_one_cap_failure_persists_zero_reservations(tmp_path):
    ledger_path = tmp_path / "cluster-hours.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-campaign",
        cap_cluster_hours=2.0,
    )
    requests = tuple(
        DatabricksRunAttemptReservationRequest(
            attempt_id=f"cap-attempt-{index}",
            workload_id=f"cap-workload-{index}",
            submit_payload=_submit_payload(timeout_seconds=3600),
        )
        for index in range(3)
    )

    with pytest.raises(ValueError, match="would exceed.*cluster-hour cap"):
        reserve_databricks_run_attempt_batch_json(ledger_path, requests)

    unchanged = read_databricks_cluster_hour_ledger_json(ledger_path)
    assert unchanged.reservations == ()
    assert unchanged.accounted_cluster_hours == 0.0


def test_authorized_batch_uses_the_in_lock_frozen_payload_snapshot(tmp_path):
    ledger_path = tmp_path / "cluster-hours.json"
    opening = create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-campaign",
    )
    payload = _submit_payload(timeout_seconds=3600)
    request = DatabricksRunAttemptReservationRequest(
        attempt_id="frozen-attempt",
        workload_id="frozen-workload",
        submit_payload=payload,
    )

    def mutate_caller_owned_payload(_live_ledger, _reservations, _snapshots):
        payload["run_name"] = "mutated-after-the-transaction-snapshot"

    updated, authorization = reserve_databricks_run_attempt_batch_authorized_json(
        ledger_path,
        (request,),
        expected_predecessor_prefix=databricks_ledger_prefix(opening),
        batch_validator=mutate_caller_owned_payload,
    )

    assert authorization.attempt_ids == ("frozen-attempt",)
    assert authorization.submit_payload_sha256s == (
        updated.reservations[0].submit_payload_sha256,
    )
    assert authorization.batch_prefix == databricks_ledger_prefix(updated)
    assert (
        databricks_submit_payload_reservation(
            payload,
            attempt_id="frozen-attempt",
            workload_id="frozen-workload",
        ).submit_payload_sha256
        != authorization.submit_payload_sha256s[0]
    )


def test_batch_validator_cannot_mutate_nested_canonical_payload_snapshots(tmp_path):
    ledger_path = tmp_path / "cluster-hours.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-campaign",
    )
    request = DatabricksRunAttemptReservationRequest(
        attempt_id="mutation-attempt",
        workload_id="mutation-workload",
        submit_payload=_submit_payload(timeout_seconds=3600),
    )

    def mutate_snapshot(_live_ledger, _reservations, snapshots):
        snapshots[0]["tasks"][0]["new_cluster"]["node_type_id"] = "forged-node"

    with pytest.raises(ValueError, match="must not mutate.*index 0"):
        reserve_databricks_run_attempt_batch_json(
            ledger_path,
            (request,),
            batch_validator=mutate_snapshot,
        )

    assert read_databricks_cluster_hour_ledger_json(ledger_path).reservations == ()


def test_identifier_fields_reject_databricks_credentials():
    with pytest.raises(ValueError, match="credential"):
        DatabricksClusterHourReservation(
            attempt_id="dapi" + "0123456789abcdef0123456789abcdef",
            workload_id="representative-vllm",
            submit_payload_sha256="0" * 64,
            run_timeout_seconds=14400,
            task_timeout_seconds=(14400,),
        )


@pytest.mark.parametrize("run_id", ["0", "001", "-1", "+1", " 1", "run-1"])
def test_submission_receipt_rejects_noncanonical_databricks_run_ids(run_id):
    with pytest.raises(ValueError, match="canonical decimal"):
        DatabricksRunSubmissionReceipt(
            attempt_id="attempt-001",
            run_id=run_id,
            submit_payload_sha256="a" * 64,
            submit_response_sha256="b" * 64,
        )


def test_receipt_bound_attempt_requires_direct_terminal_reconciliation(tmp_path):
    ledger_path = tmp_path / "cluster-hours.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-qualification",
    )
    reserve_databricks_run_attempt_json(
        ledger_path,
        _submit_payload(),
        attempt_id="gpuq-attempt-001",
        workload_id="gpu-qualification",
    )

    receipt_bound = record_databricks_run_submission_receipt_json(
        ledger_path,
        attempt_id="gpuq-attempt-001",
        submit_response={"run_id": 101},
    )
    receipt = receipt_bound.submission_receipts[0]
    assert receipt.run_id == "101"
    assert receipt.submit_payload_sha256 == (
        receipt_bound.reservations[0].submit_payload_sha256
    )
    with pytest.raises(ValueError, match="require direct runs/get"):
        record_databricks_run_terminal_actual_json(
            ledger_path,
            attempt_id="gpuq-attempt-001",
            terminal_state="succeeded",
            actual_cluster_duration_seconds=1,
        )

    terminal = record_databricks_verified_run_terminal_actual_json(
        ledger_path,
        attempt_id="gpuq-attempt-001",
        run_record=_terminal_run(),
    )
    actual = terminal.terminal_actuals[0]
    assert actual.actual_cluster_duration_seconds == 60.0
    assert actual.verification_source == "direct_databricks_runs_get"
    assert actual.run_id == "101"
    assert actual.submit_payload_sha256 == receipt.submit_payload_sha256
    assert len(actual.control_plane_status_sha256 or "") == 64
    assert terminal.active_reserved_cluster_hours == 0.0
    assert terminal.active_reserved_task_count == 0
    assert (
        record_databricks_verified_run_terminal_actual_json(
            ledger_path,
            attempt_id="gpuq-attempt-001",
            run_record=_terminal_run(),
        )
        == terminal
    )
    changed_response = _terminal_run()
    changed_response["run_page_url"] = "https://dbc.example/runs/101"
    with pytest.raises(ValueError, match="reconciliation differs"):
        record_databricks_verified_run_terminal_actual_json(
            ledger_path,
            attempt_id="gpuq-attempt-001",
            run_record=changed_response,
        )


def test_verified_terminal_rejects_wrong_run_nonterminal_and_task_count(tmp_path):
    ledger_path = tmp_path / "cluster-hours.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-qualification",
    )
    reserve_databricks_run_attempt_json(
        ledger_path,
        _submit_payload(),
        attempt_id="gpuq-attempt-001",
        workload_id="gpu-qualification",
    )
    record_databricks_run_submission_receipt_json(
        ledger_path,
        attempt_id="gpuq-attempt-001",
        submit_response={"run_id": 101},
    )

    with pytest.raises(ValueError, match="run_id does not match"):
        record_databricks_verified_run_terminal_actual_json(
            ledger_path,
            attempt_id="gpuq-attempt-001",
            run_record=_terminal_run(run_id=102),
        )
    running = _terminal_run()
    running["state"] = {"life_cycle_state": "RUNNING"}
    with pytest.raises(ValueError, match="not terminal"):
        record_databricks_verified_run_terminal_actual_json(
            ledger_path,
            attempt_id="gpuq-attempt-001",
            run_record=running,
        )
    with pytest.raises(ValueError, match="task count differs"):
        record_databricks_verified_run_terminal_actual_json(
            ledger_path,
            attempt_id="gpuq-attempt-001",
            run_record=_terminal_run(task_count=2),
        )


def test_verified_multi_task_terminal_sums_task_billing_intervals(tmp_path):
    ledger_path = tmp_path / "cluster-hours.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-full-score",
    )
    reserve_databricks_run_attempt_json(
        ledger_path,
        _submit_payload(task_count=2),
        attempt_id="full-score-attempt-001",
        workload_id="full-score-phase",
    )
    record_databricks_run_submission_receipt_json(
        ledger_path,
        attempt_id="full-score-attempt-001",
        submit_response={"run_id": 101},
    )

    terminal = record_databricks_verified_run_terminal_actual_json(
        ledger_path,
        attempt_id="full-score-attempt-001",
        run_record=_terminal_run(task_count=2),
    )
    assert terminal.terminal_actuals[0].actual_cluster_duration_seconds == 180.0

    duplicate = _terminal_run(task_count=2)
    duplicate["tasks"][1]["run_id"] = duplicate["tasks"][0]["run_id"]
    with pytest.raises(ValueError, match="keys and run IDs must be unique"):
        record_databricks_verified_run_terminal_actual_json(
            ledger_path,
            attempt_id="full-score-attempt-001",
            run_record=duplicate,
        )


def test_failed_never_started_task_reconciles_to_canonical_zero(tmp_path):
    ledger_path = tmp_path / "cluster-hours.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-qualification",
    )
    reserve_databricks_run_attempt_json(
        ledger_path,
        _submit_payload(),
        attempt_id="gpuq-rejected-001",
        workload_id="gpu-qualification",
    )
    record_databricks_run_submission_receipt_json(
        ledger_path,
        attempt_id="gpuq-rejected-001",
        submit_response={"run_id": 101},
    )
    rejected = _terminal_run()
    rejected["state"]["result_state"] = "FAILED"
    rejected["tasks"][0]["state"]["result_state"] = "FAILED"
    rejected["tasks"][0].pop("start_time")
    rejected["tasks"][0].pop("end_time")

    terminal = record_databricks_verified_run_terminal_actual_json(
        ledger_path,
        attempt_id="gpuq-rejected-001",
        run_record=rejected,
    )

    assert terminal.terminal_actuals[0].terminal_state == "failed"
    assert terminal.terminal_actuals[0].actual_cluster_duration_seconds == 0.0
    assert terminal.active_reserved_cluster_hours == 0.0


@pytest.mark.parametrize(
    ("start_time", "end_time", "child_result_state"),
    [
        (None, None, "SUCCESS"),
        (False, False, "FAILED"),
        (0.0, 0.0, "FAILED"),
        (None, 0, "FAILED"),
        (0, None, "FAILED"),
    ],
)
def test_zero_duration_reconciliation_rejects_contradictory_or_noncanonical_tasks(
    tmp_path, start_time, end_time, child_result_state
):
    ledger_path = tmp_path / "cluster-hours.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-qualification",
    )
    reserve_databricks_run_attempt_json(
        ledger_path,
        _submit_payload(),
        attempt_id="gpuq-rejected-001",
        workload_id="gpu-qualification",
    )
    record_databricks_run_submission_receipt_json(
        ledger_path,
        attempt_id="gpuq-rejected-001",
        submit_response={"run_id": 101},
    )
    rejected = _terminal_run()
    rejected["state"]["result_state"] = "FAILED"
    rejected["tasks"][0]["state"]["result_state"] = child_result_state
    rejected["tasks"][0]["start_time"] = start_time
    rejected["tasks"][0]["end_time"] = end_time

    with pytest.raises(ValueError, match="increasing millisecond times"):
        record_databricks_verified_run_terminal_actual_json(
            ledger_path,
            attempt_id="gpuq-rejected-001",
            run_record=rejected,
        )

    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    assert ledger.terminal_actuals == ()
    assert ledger.active_reserved_cluster_hours == 4.0


def test_failed_never_started_task_accepts_exact_integer_zero_pair(tmp_path):
    ledger_path = tmp_path / "cluster-hours.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="publication-qualification",
    )
    reserve_databricks_run_attempt_json(
        ledger_path,
        _submit_payload(),
        attempt_id="gpuq-rejected-001",
        workload_id="gpu-qualification",
    )
    record_databricks_run_submission_receipt_json(
        ledger_path,
        attempt_id="gpuq-rejected-001",
        submit_response={"run_id": 101},
    )
    rejected = _terminal_run()
    rejected["state"]["result_state"] = "FAILED"
    rejected["tasks"][0]["state"]["result_state"] = "FAILED"
    rejected["tasks"][0]["start_time"] = 0
    rejected["tasks"][0]["end_time"] = 0

    terminal = record_databricks_verified_run_terminal_actual_json(
        ledger_path,
        attempt_id="gpuq-rejected-001",
        run_record=rejected,
    )

    assert terminal.terminal_actuals[0].actual_cluster_duration_seconds == 0.0


def test_atomic_ledger_publication_fsyncs_directory_before_and_after_replace(
    tmp_path, monkeypatch
):
    ledger_path = tmp_path / "cluster-hours.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path, ledger_id="publication-qualification"
    )
    calls = []
    real_fsync_directory = resource_ledger._fsync_directory

    def observe(path):
        calls.append(path)
        real_fsync_directory(path)

    monkeypatch.setattr(resource_ledger, "_fsync_directory", observe)
    reserve_databricks_run_attempt_json(
        ledger_path,
        _submit_payload(),
        attempt_id="gpuq-001",
        workload_id="gpu-qualification",
    )

    assert calls == [tmp_path, tmp_path]


def test_atomic_ledger_pre_replace_directory_fsync_failure_preserves_old_state(
    tmp_path, monkeypatch
):
    ledger_path = tmp_path / "cluster-hours.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path, ledger_id="publication-qualification"
    )

    def fail(path):
        raise OSError("simulated ledger directory fsync failure")

    monkeypatch.setattr(resource_ledger, "_fsync_directory", fail)
    with pytest.raises(OSError, match="directory fsync failure"):
        reserve_databricks_run_attempt_json(
            ledger_path,
            _submit_payload(),
            attempt_id="gpuq-001",
            workload_id="gpu-qualification",
        )

    assert read_databricks_cluster_hour_ledger_json(ledger_path).reservations == ()
    assert not list(tmp_path.glob(".cluster-hours.json.*.tmp"))


def test_ledger_path_rejects_a_symlink_ancestor(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    redirect = tmp_path / "redirect"
    redirect.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="cannot traverse a symlink"):
        create_databricks_cluster_hour_ledger_json(
            redirect / "cluster-hours.json",
            ledger_id="publication-qualification",
        )

    assert not (real / "cluster-hours.json").exists()


def test_schema_v1_ledgers_migrate_without_fabricating_provenance():
    reservation = databricks_submit_payload_reservation(
        _submit_payload(),
        attempt_id="historical-attempt",
        workload_id="historical-vllm",
    )
    ledger = DatabricksClusterHourLedger(ledger_id="historical-ledger").reserve(
        reservation
    )
    record = databricks_cluster_hour_ledger_to_record(ledger)
    record["schema_version"] = 1
    record.pop("submission_receipts")

    migrated = databricks_cluster_hour_ledger_from_record(record)

    assert DATABRICKS_CLUSTER_HOUR_LEDGER_SCHEMA_VERSION == 2
    assert migrated.submission_receipts == ()
    assert migrated.terminal_actuals == ()
    assert databricks_cluster_hour_ledger_to_record(migrated)["schema_version"] == 2
