"""Persistent, fail-closed cluster-hour accounting for Databricks run attempts.

The ledger stores only bounded execution metadata and a digest of each submitted
payload.  It never persists a workspace URL, credential, request body, run
output, or Databricks response.  Reservations are append-only; terminal actuals
are separate immutable records that reconcile an active worst-case reservation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Literal

from document_kv_cache.databricks_job import (
    _validated_databricks_run_timeout_seconds,
    _validated_databricks_task_max_retries,
)


DATABRICKS_CLUSTER_HOUR_LEDGER_RECORD_TYPE = (
    "document_kv.databricks_cluster_hour_ledger.v1"
)
DATABRICKS_CLUSTER_HOUR_LEDGER_SCHEMA_VERSION = 1
MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS = 120.0
DATABRICKS_LEDGER_TERMINAL_STATES = (
    "succeeded",
    "failed",
    "canceled",
    "timed_out",
    "internal_error",
    "skipped",
)

__all__ = [
    "DATABRICKS_CLUSTER_HOUR_LEDGER_RECORD_TYPE",
    "DATABRICKS_CLUSTER_HOUR_LEDGER_SCHEMA_VERSION",
    "MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS",
    "DATABRICKS_LEDGER_TERMINAL_STATES",
    "DatabricksClusterHourReservation",
    "DatabricksClusterHourTerminalActual",
    "DatabricksClusterHourLedger",
    "DatabricksReservationValidator",
    "canonical_databricks_submit_payload_snapshot",
    "databricks_submit_payload_reservation",
    "databricks_cluster_hour_ledger_to_record",
    "databricks_cluster_hour_ledger_from_record",
    "read_databricks_cluster_hour_ledger_json",
    "create_databricks_cluster_hour_ledger_json",
    "reserve_databricks_run_attempt_json",
    "record_databricks_run_terminal_actual_json",
    "main",
]

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DATABRICKS_PAT_RE = re.compile(r"dapi[0-9a-fA-F]{32}")
_LEDGER_RECORD_KEYS = frozenset(
    {
        "record_type",
        "schema_version",
        "ledger_id",
        "cap_cluster_hours",
        "reservations",
        "terminal_actuals",
        "accounting",
    }
)
_RESERVATION_RECORD_KEYS = frozenset(
    {
        "attempt_id",
        "workload_id",
        "submit_payload_sha256",
        "run_timeout_seconds",
        "task_timeout_seconds",
        "reserved_cluster_hours",
    }
)
_TERMINAL_RECORD_KEYS = frozenset(
    {
        "attempt_id",
        "terminal_state",
        "actual_cluster_duration_seconds",
        "actual_cluster_hours",
    }
)
_ACCOUNTING_RECORD_KEYS = frozenset(
    {
        "active_reserved_cluster_hours",
        "terminal_actual_cluster_hours",
        "accounted_cluster_hours",
        "remaining_cluster_hours",
    }
)

DatabricksReservationValidator = Callable[
    ["DatabricksClusterHourReservation", Mapping[str, Any]],
    None,
]


@dataclass(frozen=True, slots=True)
class DatabricksClusterHourReservation:
    """Worst-case cluster-time reservation for one runs/submit attempt."""

    attempt_id: str
    workload_id: str
    submit_payload_sha256: str
    run_timeout_seconds: int
    task_timeout_seconds: tuple[int, ...]

    def __post_init__(self) -> None:
        _validated_identifier(self.attempt_id, "attempt_id")
        _validated_identifier(self.workload_id, "workload_id")
        if not _SHA256_RE.fullmatch(self.submit_payload_sha256):
            raise ValueError("submit_payload_sha256 must be a lowercase SHA-256 digest")
        _validated_databricks_run_timeout_seconds(self.run_timeout_seconds)
        task_timeouts = tuple(self.task_timeout_seconds)
        if not task_timeouts:
            raise ValueError("task_timeout_seconds must be non-empty")
        for timeout_seconds in task_timeouts:
            _validated_databricks_run_timeout_seconds(timeout_seconds)
        object.__setattr__(self, "task_timeout_seconds", task_timeouts)

    @property
    def reserved_cluster_seconds(self) -> int:
        return sum(self.task_timeout_seconds)

    @property
    def reserved_cluster_hours(self) -> float:
        return self.reserved_cluster_seconds / 3600.0


@dataclass(frozen=True, slots=True)
class DatabricksClusterHourTerminalActual:
    """Terminal aggregate cluster duration for one previously reserved attempt."""

    attempt_id: str
    terminal_state: (
        Literal[
            "succeeded",
            "failed",
            "canceled",
            "timed_out",
            "internal_error",
            "skipped",
        ]
        | str
    )
    actual_cluster_duration_seconds: float

    def __post_init__(self) -> None:
        _validated_identifier(self.attempt_id, "attempt_id")
        if self.terminal_state not in DATABRICKS_LEDGER_TERMINAL_STATES:
            raise ValueError(
                f"terminal_state must be one of {DATABRICKS_LEDGER_TERMINAL_STATES}"
            )
        if (
            not isinstance(self.actual_cluster_duration_seconds, (int, float))
            or isinstance(self.actual_cluster_duration_seconds, bool)
            or not math.isfinite(float(self.actual_cluster_duration_seconds))
            or self.actual_cluster_duration_seconds < 0
        ):
            raise ValueError(
                "actual_cluster_duration_seconds must be a non-negative finite number"
            )
        object.__setattr__(
            self,
            "actual_cluster_duration_seconds",
            float(self.actual_cluster_duration_seconds),
        )

    @property
    def actual_cluster_hours(self) -> float:
        return self.actual_cluster_duration_seconds / 3600.0


@dataclass(frozen=True, slots=True)
class DatabricksClusterHourLedger:
    """Immutable aggregate budget state reconstructed from append-only records."""

    ledger_id: str
    cap_cluster_hours: float = MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
    reservations: tuple[DatabricksClusterHourReservation, ...] = ()
    terminal_actuals: tuple[DatabricksClusterHourTerminalActual, ...] = ()

    def __post_init__(self) -> None:
        _validated_identifier(self.ledger_id, "ledger_id")
        if (
            not isinstance(self.cap_cluster_hours, (int, float))
            or isinstance(self.cap_cluster_hours, bool)
            or not math.isfinite(float(self.cap_cluster_hours))
            or not 0
            < float(self.cap_cluster_hours)
            <= MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
        ):
            raise ValueError(
                "cap_cluster_hours must be a positive finite number no greater than "
                f"{MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS}"
            )
        reservations = tuple(self.reservations)
        terminal_actuals = tuple(self.terminal_actuals)
        if any(
            not isinstance(item, DatabricksClusterHourReservation)
            for item in reservations
        ):
            raise TypeError(
                "reservations entries must be DatabricksClusterHourReservation"
            )
        if any(
            not isinstance(item, DatabricksClusterHourTerminalActual)
            for item in terminal_actuals
        ):
            raise TypeError(
                "terminal_actuals entries must be DatabricksClusterHourTerminalActual"
            )
        reservation_ids = [item.attempt_id for item in reservations]
        terminal_ids = [item.attempt_id for item in terminal_actuals]
        if len(set(reservation_ids)) != len(reservation_ids):
            raise ValueError("reservation attempt_id values must be unique")
        if len(set(terminal_ids)) != len(terminal_ids):
            raise ValueError("terminal actual attempt_id values must be unique")
        unknown_terminal_ids = sorted(set(terminal_ids).difference(reservation_ids))
        if unknown_terminal_ids:
            raise ValueError(
                "terminal actuals reference unreserved attempts: "
                f"{unknown_terminal_ids}"
            )
        reservations_by_id = {item.attempt_id: item for item in reservations}
        for actual in terminal_actuals:
            reserved_seconds = reservations_by_id[
                actual.attempt_id
            ].reserved_cluster_seconds
            if actual.actual_cluster_duration_seconds > reserved_seconds:
                raise ValueError(
                    f"terminal actual for attempt {actual.attempt_id!r} exceeds "
                    "its worst-case timeout reservation"
                )
        object.__setattr__(self, "cap_cluster_hours", float(self.cap_cluster_hours))
        object.__setattr__(self, "reservations", reservations)
        object.__setattr__(self, "terminal_actuals", terminal_actuals)
        if self.accounted_cluster_hours > self.cap_cluster_hours:
            raise ValueError(
                "ledger accounted cluster-hours exceed cap: "
                f"{self.accounted_cluster_hours} > {self.cap_cluster_hours}"
            )

    @property
    def closed_attempt_ids(self) -> frozenset[str]:
        return frozenset(item.attempt_id for item in self.terminal_actuals)

    @property
    def active_reserved_cluster_hours(self) -> float:
        closed = self.closed_attempt_ids
        return sum(
            reservation.reserved_cluster_hours
            for reservation in self.reservations
            if reservation.attempt_id not in closed
        )

    @property
    def terminal_actual_cluster_hours(self) -> float:
        return sum(item.actual_cluster_hours for item in self.terminal_actuals)

    @property
    def accounted_cluster_hours(self) -> float:
        return self.active_reserved_cluster_hours + self.terminal_actual_cluster_hours

    @property
    def remaining_cluster_hours(self) -> float:
        return self.cap_cluster_hours - self.accounted_cluster_hours

    def reserve(
        self,
        reservation: DatabricksClusterHourReservation,
    ) -> DatabricksClusterHourLedger:
        if not isinstance(reservation, DatabricksClusterHourReservation):
            raise TypeError("reservation must be DatabricksClusterHourReservation")
        if any(item.attempt_id == reservation.attempt_id for item in self.reservations):
            raise ValueError(f"attempt {reservation.attempt_id!r} is already reserved")
        projected = self.accounted_cluster_hours + reservation.reserved_cluster_hours
        if projected > self.cap_cluster_hours:
            raise ValueError(
                f"attempt {reservation.attempt_id!r} would exceed the aggregate "
                f"cluster-hour cap: {projected} > {self.cap_cluster_hours}"
            )
        return DatabricksClusterHourLedger(
            ledger_id=self.ledger_id,
            cap_cluster_hours=self.cap_cluster_hours,
            reservations=(*self.reservations, reservation),
            terminal_actuals=self.terminal_actuals,
        )

    def record_terminal_actual(
        self,
        actual: DatabricksClusterHourTerminalActual,
    ) -> DatabricksClusterHourLedger:
        if not isinstance(actual, DatabricksClusterHourTerminalActual):
            raise TypeError("actual must be DatabricksClusterHourTerminalActual")
        if all(item.attempt_id != actual.attempt_id for item in self.reservations):
            raise ValueError(
                f"attempt {actual.attempt_id!r} has no pre-submission reservation"
            )
        if actual.attempt_id in self.closed_attempt_ids:
            raise ValueError(
                f"attempt {actual.attempt_id!r} already has a terminal actual"
            )
        return DatabricksClusterHourLedger(
            ledger_id=self.ledger_id,
            cap_cluster_hours=self.cap_cluster_hours,
            reservations=self.reservations,
            terminal_actuals=(*self.terminal_actuals, actual),
        )


def databricks_submit_payload_reservation(
    submit_payload: Mapping[str, Any],
    *,
    attempt_id: str,
    workload_id: str,
) -> DatabricksClusterHourReservation:
    """Validate a bounded runs/submit payload and derive its worst-case reservation."""

    if not isinstance(submit_payload, Mapping):
        raise TypeError("submit_payload must be a mapping")
    run_timeout_seconds = _validated_databricks_run_timeout_seconds(
        submit_payload.get("timeout_seconds")
    )
    raw_tasks = submit_payload.get("tasks")
    if isinstance(raw_tasks, (str, bytes)) or not isinstance(raw_tasks, Sequence):
        raise ValueError("submit_payload.tasks must be a non-empty sequence")
    tasks = tuple(raw_tasks)
    if not tasks:
        raise ValueError("submit_payload.tasks must be a non-empty sequence")
    task_keys: list[str] = []
    task_timeouts: list[int] = []
    for index, raw_task in enumerate(tasks):
        if not isinstance(raw_task, Mapping):
            raise ValueError(f"submit_payload.tasks[{index}] must be an object")
        task_key = _validated_identifier(
            raw_task.get("task_key"),
            f"submit_payload.tasks[{index}].task_key",
        )
        task_keys.append(task_key)
        task_timeouts.append(
            _validated_databricks_run_timeout_seconds(raw_task.get("timeout_seconds"))
        )
        _validated_databricks_task_max_retries(raw_task.get("max_retries"))
        if not isinstance(raw_task.get("new_cluster"), Mapping):
            raise ValueError(
                f"submit_payload.tasks[{index}] must use an explicit new_cluster"
            )
    if len(set(task_keys)) != len(task_keys):
        raise ValueError("submit_payload task_key values must be unique")
    _snapshot, canonical_payload = canonical_databricks_submit_payload_snapshot(
        submit_payload
    )
    payload_digest = sha256(canonical_payload).hexdigest()
    return DatabricksClusterHourReservation(
        attempt_id=attempt_id,
        workload_id=workload_id,
        submit_payload_sha256=payload_digest,
        run_timeout_seconds=run_timeout_seconds,
        task_timeout_seconds=tuple(task_timeouts),
    )


def databricks_cluster_hour_ledger_to_record(
    ledger: DatabricksClusterHourLedger,
) -> dict[str, Any]:
    if not isinstance(ledger, DatabricksClusterHourLedger):
        raise TypeError("ledger must be DatabricksClusterHourLedger")
    return {
        "record_type": DATABRICKS_CLUSTER_HOUR_LEDGER_RECORD_TYPE,
        "schema_version": DATABRICKS_CLUSTER_HOUR_LEDGER_SCHEMA_VERSION,
        "ledger_id": ledger.ledger_id,
        "cap_cluster_hours": ledger.cap_cluster_hours,
        "reservations": [
            {
                "attempt_id": item.attempt_id,
                "workload_id": item.workload_id,
                "submit_payload_sha256": item.submit_payload_sha256,
                "run_timeout_seconds": item.run_timeout_seconds,
                "task_timeout_seconds": list(item.task_timeout_seconds),
                "reserved_cluster_hours": item.reserved_cluster_hours,
            }
            for item in ledger.reservations
        ],
        "terminal_actuals": [
            {
                "attempt_id": item.attempt_id,
                "terminal_state": item.terminal_state,
                "actual_cluster_duration_seconds": (
                    item.actual_cluster_duration_seconds
                ),
                "actual_cluster_hours": item.actual_cluster_hours,
            }
            for item in ledger.terminal_actuals
        ],
        "accounting": {
            "active_reserved_cluster_hours": (ledger.active_reserved_cluster_hours),
            "terminal_actual_cluster_hours": (ledger.terminal_actual_cluster_hours),
            "accounted_cluster_hours": ledger.accounted_cluster_hours,
            "remaining_cluster_hours": ledger.remaining_cluster_hours,
        },
    }


def databricks_cluster_hour_ledger_from_record(
    record: Mapping[str, Any],
) -> DatabricksClusterHourLedger:
    if not isinstance(record, Mapping):
        raise TypeError("ledger record must be a mapping")
    _require_exact_keys(record, _LEDGER_RECORD_KEYS, "ledger")
    if record.get("record_type") != DATABRICKS_CLUSTER_HOUR_LEDGER_RECORD_TYPE:
        raise ValueError("ledger record_type is invalid")
    if record.get("schema_version") != DATABRICKS_CLUSTER_HOUR_LEDGER_SCHEMA_VERSION:
        raise ValueError("ledger schema_version is invalid")
    reservations = tuple(
        _reservation_from_record(item, index=index)
        for index, item in enumerate(_record_sequence(record, "reservations"))
    )
    terminal_actuals = tuple(
        _terminal_actual_from_record(item, index=index)
        for index, item in enumerate(_record_sequence(record, "terminal_actuals"))
    )
    ledger = DatabricksClusterHourLedger(
        ledger_id=_record_string(record, "ledger_id"),
        cap_cluster_hours=_record_number(record, "cap_cluster_hours"),
        reservations=reservations,
        terminal_actuals=terminal_actuals,
    )
    accounting = record.get("accounting")
    if not isinstance(accounting, Mapping):
        raise ValueError("ledger.accounting must be an object")
    _require_exact_keys(accounting, _ACCOUNTING_RECORD_KEYS, "ledger.accounting")
    expected_accounting = databricks_cluster_hour_ledger_to_record(ledger)["accounting"]
    if dict(accounting) != expected_accounting:
        raise ValueError("ledger.accounting does not match immutable ledger events")
    return ledger


def read_databricks_cluster_hour_ledger_json(
    path: str | Path,
) -> DatabricksClusterHourLedger:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("ledger JSON must contain an object")
    return databricks_cluster_hour_ledger_from_record(raw)


def create_databricks_cluster_hour_ledger_json(
    path: str | Path,
    *,
    ledger_id: str,
    cap_cluster_hours: float = MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
) -> DatabricksClusterHourLedger:
    ledger_path = Path(path)
    with _exclusive_ledger_lock(ledger_path):
        if ledger_path.exists():
            raise FileExistsError(f"cluster-hour ledger already exists: {ledger_path}")
        ledger = DatabricksClusterHourLedger(
            ledger_id=ledger_id,
            cap_cluster_hours=cap_cluster_hours,
        )
        _atomic_write_ledger(ledger_path, ledger)
        return ledger


def reserve_databricks_run_attempt_json(
    ledger_path: str | Path,
    submit_payload: Mapping[str, Any],
    *,
    attempt_id: str,
    workload_id: str,
    reservation_validator: DatabricksReservationValidator | None = None,
) -> DatabricksClusterHourLedger:
    """Persist the worst-case reservation before the caller submits the run."""

    path = Path(ledger_path)
    with _exclusive_ledger_lock(path):
        snapshot, _canonical_payload = canonical_databricks_submit_payload_snapshot(
            submit_payload
        )
        reservation = databricks_submit_payload_reservation(
            snapshot,
            attempt_id=attempt_id,
            workload_id=workload_id,
        )
        if reservation_validator is not None:
            if not callable(reservation_validator):
                raise TypeError("reservation_validator must be callable or None")
            reservation_validator(reservation, snapshot)
            validated_reservation = databricks_submit_payload_reservation(
                snapshot,
                attempt_id=attempt_id,
                workload_id=workload_id,
            )
            if validated_reservation != reservation:
                raise ValueError(
                    "reservation_validator must not mutate the submit payload"
                )
        ledger = read_databricks_cluster_hour_ledger_json(path)
        updated = ledger.reserve(reservation)
        _atomic_write_ledger(path, updated)
        return updated


def record_databricks_run_terminal_actual_json(
    ledger_path: str | Path,
    *,
    attempt_id: str,
    terminal_state: str,
    actual_cluster_duration_seconds: float,
) -> DatabricksClusterHourLedger:
    path = Path(ledger_path)
    actual = DatabricksClusterHourTerminalActual(
        attempt_id=attempt_id,
        terminal_state=terminal_state,
        actual_cluster_duration_seconds=actual_cluster_duration_seconds,
    )
    with _exclusive_ledger_lock(path):
        ledger = read_databricks_cluster_hour_ledger_json(path)
        updated = ledger.record_terminal_actual(actual)
        _atomic_write_ledger(path, updated)
        return updated


def _reservation_from_record(
    value: Any,
    *,
    index: int,
) -> DatabricksClusterHourReservation:
    if not isinstance(value, Mapping):
        raise ValueError(f"ledger.reservations[{index}] must be an object")
    label = f"ledger.reservations[{index}]"
    _require_exact_keys(value, _RESERVATION_RECORD_KEYS, label)
    timeouts = tuple(
        _record_integer_value(item, f"{label}.task_timeout_seconds[{item_index}]")
        for item_index, item in enumerate(
            _record_sequence(value, "task_timeout_seconds")
        )
    )
    reservation = DatabricksClusterHourReservation(
        attempt_id=_record_string(value, "attempt_id"),
        workload_id=_record_string(value, "workload_id"),
        submit_payload_sha256=_record_string(value, "submit_payload_sha256"),
        run_timeout_seconds=_record_integer(value, "run_timeout_seconds"),
        task_timeout_seconds=timeouts,
    )
    if _record_number(value, "reserved_cluster_hours") != (
        reservation.reserved_cluster_hours
    ):
        raise ValueError(f"{label}.reserved_cluster_hours is not canonical")
    return reservation


def _terminal_actual_from_record(
    value: Any,
    *,
    index: int,
) -> DatabricksClusterHourTerminalActual:
    if not isinstance(value, Mapping):
        raise ValueError(f"ledger.terminal_actuals[{index}] must be an object")
    label = f"ledger.terminal_actuals[{index}]"
    _require_exact_keys(value, _TERMINAL_RECORD_KEYS, label)
    actual = DatabricksClusterHourTerminalActual(
        attempt_id=_record_string(value, "attempt_id"),
        terminal_state=_record_string(value, "terminal_state"),
        actual_cluster_duration_seconds=_record_number(
            value,
            "actual_cluster_duration_seconds",
        ),
    )
    if _record_number(value, "actual_cluster_hours") != actual.actual_cluster_hours:
        raise ValueError(f"{label}.actual_cluster_hours is not canonical")
    return actual


def _validated_identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            f"{field_name} must be a 1..128 character non-secret identifier"
        )
    if _DATABRICKS_PAT_RE.search(value):
        raise ValueError(f"{field_name} must not contain a Databricks credential")
    return value


def canonical_databricks_submit_payload_snapshot(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    """Return an isolated JSON snapshot and the exact canonical bytes it represents."""

    if not isinstance(value, Mapping):
        raise TypeError("submit_payload must be a mapping")
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("submit_payload must contain canonical JSON values") from exc
    snapshot = json.loads(encoded)
    if not isinstance(snapshot, dict):  # pragma: no cover - Mapping encodes as object.
        raise ValueError("submit_payload must encode as a JSON object")
    return snapshot, encoded


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    field_name: str,
) -> None:
    keys = set(value)
    if keys != expected:
        missing = sorted(expected.difference(keys))
        unexpected = sorted(str(item) for item in keys.difference(expected))
        raise ValueError(
            f"{field_name} must use the closed schema; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _record_sequence(value: Mapping[str, Any], field_name: str) -> Sequence[Any]:
    result = value.get(field_name)
    if isinstance(result, (str, bytes)) or not isinstance(result, Sequence):
        raise ValueError(f"{field_name} must be a sequence")
    return result


def _record_string(value: Mapping[str, Any], field_name: str) -> str:
    result = value.get(field_name)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{field_name} must be a non-empty string")
    return result


def _record_number(value: Mapping[str, Any], field_name: str) -> float:
    result = value.get(field_name)
    if (
        not isinstance(result, (int, float))
        or isinstance(result, bool)
        or not math.isfinite(float(result))
    ):
        raise ValueError(f"{field_name} must be a finite number")
    return float(result)


def _record_integer(value: Mapping[str, Any], field_name: str) -> int:
    return _record_integer_value(value.get(field_name), field_name)


def _record_integer_value(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field_name} must be an integer")
    return value


@contextmanager
def _exclusive_ledger_lock(path: Path) -> Iterator[None]:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_write_ledger(
    path: Path,
    ledger: DatabricksClusterHourLedger,
) -> None:
    record = databricks_cluster_hour_ledger_to_record(ledger)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_submit_payload(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("submit payload JSON must contain an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Maintain a credential-free Databricks cluster-hour ledger."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--ledger-json", required=True)
    init_parser.add_argument("--ledger-id", required=True)
    init_parser.add_argument(
        "--cap-cluster-hours",
        type=float,
        default=MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
    )

    reserve_parser = subparsers.add_parser("reserve")
    reserve_parser.add_argument("--ledger-json", required=True)
    reserve_parser.add_argument("--submit-payload-json", required=True)
    reserve_parser.add_argument("--attempt-id", required=True)
    reserve_parser.add_argument("--workload-id", required=True)

    terminal_parser = subparsers.add_parser("terminal")
    terminal_parser.add_argument("--ledger-json", required=True)
    terminal_parser.add_argument("--attempt-id", required=True)
    terminal_parser.add_argument(
        "--terminal-state",
        choices=DATABRICKS_LEDGER_TERMINAL_STATES,
        required=True,
    )
    terminal_parser.add_argument(
        "--actual-cluster-duration-seconds",
        type=float,
        required=True,
    )

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("--ledger-json", required=True)
    args = parser.parse_args(argv)

    try:
        if args.action == "init":
            ledger = create_databricks_cluster_hour_ledger_json(
                args.ledger_json,
                ledger_id=args.ledger_id,
                cap_cluster_hours=args.cap_cluster_hours,
            )
        elif args.action == "reserve":
            ledger = reserve_databricks_run_attempt_json(
                args.ledger_json,
                _load_submit_payload(args.submit_payload_json),
                attempt_id=args.attempt_id,
                workload_id=args.workload_id,
            )
        elif args.action == "terminal":
            ledger = record_databricks_run_terminal_actual_json(
                args.ledger_json,
                attempt_id=args.attempt_id,
                terminal_state=args.terminal_state,
                actual_cluster_duration_seconds=(args.actual_cluster_duration_seconds),
            )
        else:
            ledger = read_databricks_cluster_hour_ledger_json(args.ledger_json)
        print(
            json.dumps(
                databricks_cluster_hour_ledger_to_record(ledger),
                indent=2,
                sort_keys=True,
            )
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
            )
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
