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
DATABRICKS_CLUSTER_HOUR_LEDGER_SCHEMA_VERSION = 2
_LEGACY_DATABRICKS_CLUSTER_HOUR_LEDGER_SCHEMA_VERSION = 1
MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS = 1024.0
MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS = 900.0
MAX_DATABRICKS_ACTIVE_RESERVED_TASKS = 16
DATABRICKS_PUBLICATION_UNRESERVED_HEADROOM_HOURS = 124.0
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
    "MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS",
    "MAX_DATABRICKS_ACTIVE_RESERVED_TASKS",
    "DATABRICKS_PUBLICATION_UNRESERVED_HEADROOM_HOURS",
    "DATABRICKS_LEDGER_TERMINAL_STATES",
    "DatabricksClusterHourReservation",
    "DatabricksRunSubmissionReceipt",
    "DatabricksClusterHourTerminalActual",
    "DatabricksClusterHourLedger",
    "DatabricksLedgerPrefix",
    "DatabricksBatchReservationAuthorization",
    "DatabricksRunAttemptReservationRequest",
    "DatabricksReservationValidator",
    "DatabricksBatchReservationValidator",
    "canonical_databricks_submit_payload_snapshot",
    "databricks_submit_payload_reservation",
    "databricks_cluster_hour_ledger_to_record",
    "databricks_cluster_hour_ledger_from_record",
    "databricks_ledger_prefix",
    "databricks_ledger_prefix_at_counts",
    "databricks_ledger_prefix_from_record",
    "databricks_ledger_path_sha256",
    "require_databricks_ledger_prefix",
    "read_databricks_cluster_hour_ledger_json",
    "create_databricks_cluster_hour_ledger_json",
    "raise_databricks_cluster_hour_ledger_cap_json",
    "replay_databricks_run_attempt_batch_authorization_json",
    "reserve_databricks_run_attempt_json",
    "reserve_databricks_run_attempt_batch_json",
    "reserve_databricks_run_attempt_batch_authorized_json",
    "require_databricks_batch_reservation_authorization",
    "require_databricks_batch_terminal_closure",
    "require_databricks_publication_batch_admission",
    "record_databricks_run_submission_receipt_json",
    "record_databricks_verified_run_terminal_actual_json",
    "record_databricks_run_terminal_actual_json",
    "main",
]

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")
_DATABRICKS_RUN_ID_RE = re.compile(r"[1-9][0-9]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_LEDGER_PATH_HASH_DOMAIN = "cachet.databricks_ledger_path.v1"
_DATABRICKS_PAT_RE = re.compile(r"dapi[0-9a-fA-F]{32}")
_LEDGER_RECORD_KEYS_V1 = frozenset(
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
_LEDGER_RECORD_KEYS = frozenset({*_LEDGER_RECORD_KEYS_V1, "submission_receipts"})
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
_SUBMISSION_RECEIPT_RECORD_KEYS = frozenset(
    {
        "attempt_id",
        "run_id",
        "submit_payload_sha256",
        "submit_response_sha256",
    }
)
_TERMINAL_RECORD_KEYS_V1 = frozenset(
    {
        "attempt_id",
        "terminal_state",
        "actual_cluster_duration_seconds",
        "actual_cluster_hours",
    }
)
_TERMINAL_RECORD_KEYS = frozenset(
    {
        *_TERMINAL_RECORD_KEYS_V1,
        "control_plane_status_sha256",
        "run_id",
        "submit_payload_sha256",
        "verification_source",
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
_LEDGER_PREFIX_RECORD_KEYS = frozenset(
    {
        "cap_cluster_hours",
        "ledger_id",
        "prefix_sha256",
        "reservation_count",
        "submission_receipt_count",
        "terminal_actual_count",
    }
)

DatabricksReservationValidator = Callable[
    ["DatabricksClusterHourReservation", Mapping[str, Any]],
    None,
]
DatabricksBatchReservationValidator = Callable[
    [
        "DatabricksClusterHourLedger",
        tuple["DatabricksClusterHourReservation", ...],
        tuple[Mapping[str, Any], ...],
    ],
    None,
]


@dataclass(frozen=True, slots=True)
class DatabricksRunAttemptReservationRequest:
    """One ordered runs/submit payload requested for atomic reservation."""

    attempt_id: str
    workload_id: str
    submit_payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _validated_identifier(self.attempt_id, "attempt_id")
        _validated_identifier(self.workload_id, "workload_id")
        if not isinstance(self.submit_payload, Mapping):
            raise TypeError("submit_payload must be a mapping")


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
class DatabricksRunSubmissionReceipt:
    """Immutable run identity returned after one reserved ``runs/submit`` POST."""

    attempt_id: str
    run_id: str
    submit_payload_sha256: str
    submit_response_sha256: str

    def __post_init__(self) -> None:
        _validated_identifier(self.attempt_id, "attempt_id")
        if _validated_databricks_run_id(self.run_id, "run_id") != self.run_id:
            raise ValueError("run_id must already be a canonical decimal string")
        for field_name in ("submit_payload_sha256", "submit_response_sha256"):
            value = getattr(self, field_name)
            if not _SHA256_RE.fullmatch(value):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


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
    verification_source: str = "legacy_manual"
    run_id: str | None = None
    submit_payload_sha256: str | None = None
    control_plane_status_sha256: str | None = None

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
        if self.verification_source not in {
            "legacy_manual",
            "direct_databricks_runs_get",
        }:
            raise ValueError("terminal verification_source is unsupported")
        provenance = (
            self.run_id,
            self.submit_payload_sha256,
            self.control_plane_status_sha256,
        )
        if self.verification_source == "legacy_manual":
            if any(value is not None for value in provenance):
                raise ValueError(
                    "legacy terminal actuals cannot claim control-plane provenance"
                )
        else:
            if self.run_id is None:
                raise ValueError("verified terminal actuals require run_id")
            if _validated_databricks_run_id(self.run_id, "run_id") != self.run_id:
                raise ValueError("run_id must already be a canonical decimal string")
            for field_name, value in (
                ("submit_payload_sha256", self.submit_payload_sha256),
                ("control_plane_status_sha256", self.control_plane_status_sha256),
            ):
                if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                    raise ValueError(
                        f"verified terminal actuals require a lowercase {field_name}"
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
    submission_receipts: tuple[DatabricksRunSubmissionReceipt, ...] = ()
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
        submission_receipts = tuple(self.submission_receipts)
        terminal_actuals = tuple(self.terminal_actuals)
        if any(
            not isinstance(item, DatabricksClusterHourReservation)
            for item in reservations
        ):
            raise TypeError(
                "reservations entries must be DatabricksClusterHourReservation"
            )
        if any(
            not isinstance(item, DatabricksRunSubmissionReceipt)
            for item in submission_receipts
        ):
            raise TypeError(
                "submission_receipts entries must be DatabricksRunSubmissionReceipt"
            )
        if any(
            not isinstance(item, DatabricksClusterHourTerminalActual)
            for item in terminal_actuals
        ):
            raise TypeError(
                "terminal_actuals entries must be DatabricksClusterHourTerminalActual"
            )
        reservation_ids = [item.attempt_id for item in reservations]
        submission_ids = [item.attempt_id for item in submission_receipts]
        terminal_ids = [item.attempt_id for item in terminal_actuals]
        if len(set(reservation_ids)) != len(reservation_ids):
            raise ValueError("reservation attempt_id values must be unique")
        if len(set(terminal_ids)) != len(terminal_ids):
            raise ValueError("terminal actual attempt_id values must be unique")
        if len(set(submission_ids)) != len(submission_ids):
            raise ValueError("submission receipt attempt_id values must be unique")
        run_ids = [item.run_id for item in submission_receipts]
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("submission receipt run_id values must be unique")
        unknown_submission_ids = sorted(set(submission_ids).difference(reservation_ids))
        if unknown_submission_ids:
            raise ValueError(
                "submission receipts reference unreserved attempts: "
                f"{unknown_submission_ids}"
            )
        unknown_terminal_ids = sorted(set(terminal_ids).difference(reservation_ids))
        if unknown_terminal_ids:
            raise ValueError(
                "terminal actuals reference unreserved attempts: "
                f"{unknown_terminal_ids}"
            )
        reservations_by_id = {item.attempt_id: item for item in reservations}
        submissions_by_id = {item.attempt_id: item for item in submission_receipts}
        for receipt in submission_receipts:
            reservation = reservations_by_id[receipt.attempt_id]
            if receipt.submit_payload_sha256 != reservation.submit_payload_sha256:
                raise ValueError(
                    f"submission receipt for attempt {receipt.attempt_id!r} does not "
                    "match its reserved submit payload"
                )
        for actual in terminal_actuals:
            reserved_seconds = reservations_by_id[
                actual.attempt_id
            ].reserved_cluster_seconds
            if actual.actual_cluster_duration_seconds > reserved_seconds:
                raise ValueError(
                    f"terminal actual for attempt {actual.attempt_id!r} exceeds "
                    "its worst-case timeout reservation"
                )
            bound_receipt = submissions_by_id.get(actual.attempt_id)
            if bound_receipt is not None:
                if actual.verification_source != "direct_databricks_runs_get":
                    raise ValueError(
                        "receipt-bound attempts require direct Databricks terminal "
                        "verification"
                    )
                if (
                    actual.run_id != bound_receipt.run_id
                    or actual.submit_payload_sha256
                    != bound_receipt.submit_payload_sha256
                ):
                    raise ValueError(
                        "verified terminal actual does not match its submission receipt"
                    )
            elif actual.verification_source != "legacy_manual":
                raise ValueError(
                    "verified terminal actuals require a recorded submission receipt"
                )
        object.__setattr__(self, "cap_cluster_hours", float(self.cap_cluster_hours))
        object.__setattr__(self, "reservations", reservations)
        object.__setattr__(self, "submission_receipts", submission_receipts)
        object.__setattr__(self, "terminal_actuals", terminal_actuals)
        if self.accounted_cluster_hours > self.cap_cluster_hours:
            raise ValueError(
                "ledger accounted cluster-hours exceed cap: "
                f"{self.accounted_cluster_hours} > {self.cap_cluster_hours}"
            )
        if (
            self.cap_cluster_hours == MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
            and self.accounted_cluster_hours
            + DATABRICKS_PUBLICATION_UNRESERVED_HEADROOM_HOURS
            > self.cap_cluster_hours
        ):
            raise ValueError(
                "publication ledger accounted hours consume the protected "
                f"{DATABRICKS_PUBLICATION_UNRESERVED_HEADROOM_HOURS:g}-hour headroom"
            )
        if (
            self.active_reserved_cluster_hours
            > MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS
        ):
            raise ValueError(
                "ledger active reservations exceed the publication campaign "
                "headroom guard: "
                f"{self.active_reserved_cluster_hours} > "
                f"{MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS}"
            )
        if self.active_reserved_task_count > MAX_DATABRICKS_ACTIVE_RESERVED_TASKS:
            raise ValueError(
                "ledger active tasks exceed the global concurrency guard: "
                f"{self.active_reserved_task_count} > "
                f"{MAX_DATABRICKS_ACTIVE_RESERVED_TASKS}"
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
    def active_reserved_task_count(self) -> int:
        """Conservatively count tasks in every nonterminal reservation."""

        closed = self.closed_attempt_ids
        return sum(
            len(reservation.task_timeout_seconds)
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
        if (
            self.cap_cluster_hours == MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
            and projected + DATABRICKS_PUBLICATION_UNRESERVED_HEADROOM_HOURS
            > self.cap_cluster_hours
        ):
            raise ValueError(
                f"attempt {reservation.attempt_id!r} would consume the protected "
                f"{DATABRICKS_PUBLICATION_UNRESERVED_HEADROOM_HOURS:g}-hour "
                "publication headroom"
            )
        projected_active = (
            self.active_reserved_cluster_hours + reservation.reserved_cluster_hours
        )
        projected_active_tasks = self.active_reserved_task_count + len(
            reservation.task_timeout_seconds
        )
        if projected_active_tasks > MAX_DATABRICKS_ACTIVE_RESERVED_TASKS:
            raise ValueError(
                f"attempt {reservation.attempt_id!r} would exceed the active "
                "task concurrency guard: "
                f"{projected_active_tasks} > {MAX_DATABRICKS_ACTIVE_RESERVED_TASKS}"
            )
        if projected_active > MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS:
            raise ValueError(
                f"attempt {reservation.attempt_id!r} would exceed the active "
                "reservation headroom guard: "
                f"{projected_active} > "
                f"{MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS}"
            )
        return DatabricksClusterHourLedger(
            ledger_id=self.ledger_id,
            cap_cluster_hours=self.cap_cluster_hours,
            reservations=(*self.reservations, reservation),
            submission_receipts=self.submission_receipts,
            terminal_actuals=self.terminal_actuals,
        )

    def record_submission_receipt(
        self,
        receipt: DatabricksRunSubmissionReceipt,
    ) -> DatabricksClusterHourLedger:
        if not isinstance(receipt, DatabricksRunSubmissionReceipt):
            raise TypeError("receipt must be DatabricksRunSubmissionReceipt")
        existing = next(
            (
                item
                for item in self.submission_receipts
                if item.attempt_id == receipt.attempt_id
            ),
            None,
        )
        if existing is not None:
            if existing == receipt:
                return self
            raise ValueError(
                f"attempt {receipt.attempt_id!r} already has a different submission receipt"
            )
        if receipt.run_id in {item.run_id for item in self.submission_receipts}:
            raise ValueError(
                f"Databricks run {receipt.run_id!r} is already receipt-bound"
            )
        return DatabricksClusterHourLedger(
            ledger_id=self.ledger_id,
            cap_cluster_hours=self.cap_cluster_hours,
            reservations=self.reservations,
            submission_receipts=(*self.submission_receipts, receipt),
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
            submission_receipts=self.submission_receipts,
            terminal_actuals=(*self.terminal_actuals, actual),
        )


@dataclass(frozen=True, slots=True)
class DatabricksLedgerPrefix:
    """Canonical authority over one append-only prefix of a campaign ledger."""

    ledger_id: str
    cap_cluster_hours: float
    reservation_count: int
    submission_receipt_count: int
    terminal_actual_count: int
    prefix_sha256: str

    def __post_init__(self) -> None:
        _validated_identifier(self.ledger_id, "ledger_id")
        cap = _validated_cap_cluster_hours(self.cap_cluster_hours, "cap_cluster_hours")
        object.__setattr__(self, "cap_cluster_hours", cap)
        for field_name in (
            "reservation_count",
            "submission_receipt_count",
            "terminal_actual_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if not _SHA256_RE.fullmatch(self.prefix_sha256):
            raise ValueError("prefix_sha256 must be a lowercase SHA-256 digest")

    def to_record(self) -> dict[str, Any]:
        return {
            "cap_cluster_hours": self.cap_cluster_hours,
            "ledger_id": self.ledger_id,
            "prefix_sha256": self.prefix_sha256,
            "reservation_count": self.reservation_count,
            "submission_receipt_count": self.submission_receipt_count,
            "terminal_actual_count": self.terminal_actual_count,
        }


_BATCH_RESERVATION_AUTHORIZATION_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class DatabricksBatchReservationAuthorization:
    """Ephemeral proof that one exact phase batch was reserved atomically.

    A ledger prefix alone proves ordered history but cannot prove that a group
    of reservations was admitted in one lock transaction.  This capability is
    therefore issued only by the atomic batch API and is required before any
    member may be POSTed by the pre-reserved submit path.
    """

    predecessor_prefix: DatabricksLedgerPrefix
    batch_prefix: DatabricksLedgerPrefix
    ledger_path_sha256: str
    attempt_ids: tuple[str, ...]
    submit_payload_sha256s: tuple[str, ...]

    def __init__(
        self,
        *,
        predecessor_prefix: DatabricksLedgerPrefix,
        batch_prefix: DatabricksLedgerPrefix,
        ledger_path_sha256: str,
        attempt_ids: Sequence[str],
        submit_payload_sha256s: Sequence[str],
        _issuer: object,
    ) -> None:
        if _issuer is not _BATCH_RESERVATION_AUTHORIZATION_ISSUER:
            raise TypeError(
                "batch reservation authority requires atomic ledger admission"
            )
        if not isinstance(predecessor_prefix, DatabricksLedgerPrefix) or not isinstance(
            batch_prefix, DatabricksLedgerPrefix
        ):
            raise TypeError("batch ledger prefixes have the wrong type")
        normalized_attempts = tuple(attempt_ids)
        normalized_digests = tuple(submit_payload_sha256s)
        if not normalized_attempts or len(normalized_attempts) != len(
            normalized_digests
        ):
            raise ValueError("batch authority requires aligned non-empty members")
        if len(set(normalized_attempts)) != len(normalized_attempts):
            raise ValueError("batch authority attempt IDs must be unique")
        for attempt_id in normalized_attempts:
            _validated_identifier(attempt_id, "attempt_id")
        if any(not _SHA256_RE.fullmatch(item) for item in normalized_digests):
            raise ValueError("batch authority payload digests must be SHA-256 values")
        if not _SHA256_RE.fullmatch(ledger_path_sha256):
            raise ValueError("batch authority ledger path digest must be SHA-256")
        if (
            predecessor_prefix.ledger_id != batch_prefix.ledger_id
            or predecessor_prefix.cap_cluster_hours != batch_prefix.cap_cluster_hours
            or batch_prefix.reservation_count
            != predecessor_prefix.reservation_count + len(normalized_attempts)
            or batch_prefix.submission_receipt_count
            != predecessor_prefix.submission_receipt_count
            or batch_prefix.terminal_actual_count
            != predecessor_prefix.terminal_actual_count
        ):
            raise ValueError("batch authority prefix transition is invalid")
        object.__setattr__(self, "predecessor_prefix", predecessor_prefix)
        object.__setattr__(self, "batch_prefix", batch_prefix)
        object.__setattr__(self, "ledger_path_sha256", ledger_path_sha256)
        object.__setattr__(self, "attempt_ids", normalized_attempts)
        object.__setattr__(self, "submit_payload_sha256s", normalized_digests)


def databricks_ledger_prefix(
    ledger: DatabricksClusterHourLedger,
) -> DatabricksLedgerPrefix:
    """Close the complete current append-only history as a reusable prefix."""

    if not isinstance(ledger, DatabricksClusterHourLedger):
        raise TypeError("ledger must be a DatabricksClusterHourLedger")
    return _databricks_ledger_prefix_at_counts(
        ledger,
        reservation_count=len(ledger.reservations),
        submission_receipt_count=len(ledger.submission_receipts),
        terminal_actual_count=len(ledger.terminal_actuals),
    )


def databricks_ledger_prefix_at_counts(
    ledger: DatabricksClusterHourLedger,
    *,
    reservation_count: int,
    submission_receipt_count: int,
    terminal_actual_count: int,
) -> DatabricksLedgerPrefix:
    """Close one exact historical position in the append-only ledger.

    Unlike a caller-side reconstructed record, this helper can only return a
    prefix that is actually present in the supplied validated ledger.  It is
    used by phase replay to bind the reservation, receipt, and terminal
    boundaries without absorbing later campaign events.
    """

    if not isinstance(ledger, DatabricksClusterHourLedger):
        raise TypeError("ledger must be a DatabricksClusterHourLedger")
    for field_name, value, maximum in (
        ("reservation_count", reservation_count, len(ledger.reservations)),
        (
            "submission_receipt_count",
            submission_receipt_count,
            len(ledger.submission_receipts),
        ),
        ("terminal_actual_count", terminal_actual_count, len(ledger.terminal_actuals)),
    ):
        if type(value) is not int or value < 0 or value > maximum:
            raise ValueError(
                f"{field_name} must be an integer in the closed ledger range"
            )
    return _databricks_ledger_prefix_at_counts(
        ledger,
        reservation_count=reservation_count,
        submission_receipt_count=submission_receipt_count,
        terminal_actual_count=terminal_actual_count,
    )


def databricks_ledger_prefix_from_record(
    record: Mapping[str, Any],
) -> DatabricksLedgerPrefix:
    """Parse the closed canonical embedded record for a ledger prefix."""

    if not isinstance(record, Mapping):
        raise TypeError("ledger prefix record must be a mapping")
    _require_exact_keys(record, _LEDGER_PREFIX_RECORD_KEYS, "ledger prefix")
    prefix = DatabricksLedgerPrefix(
        ledger_id=_record_string(record, "ledger_id"),
        cap_cluster_hours=_record_number(record, "cap_cluster_hours"),
        reservation_count=_record_integer(record, "reservation_count"),
        submission_receipt_count=_record_integer(record, "submission_receipt_count"),
        terminal_actual_count=_record_integer(record, "terminal_actual_count"),
        prefix_sha256=_record_string(record, "prefix_sha256"),
    )
    if prefix.to_record() != dict(record):
        raise ValueError("ledger prefix record is not canonical")
    return prefix


def databricks_ledger_path_sha256(path: str | Path) -> str:
    """Return a privacy-safe binding to one symlink-free ledger location."""

    ledger_path = Path(path).expanduser().absolute()
    _require_no_symlink_ancestors(
        ledger_path,
        label="ledger path",
        include_leaf=True,
    )
    if not ledger_path.is_file():
        raise ValueError("ledger path must be an existing regular file")
    resolved = ledger_path.resolve(strict=True)
    binding = json.dumps(
        {"domain": _LEDGER_PATH_HASH_DOMAIN, "resolved_absolute_path": str(resolved)},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(binding).hexdigest()


def require_databricks_ledger_prefix(
    ledger: DatabricksClusterHourLedger,
    expected: DatabricksLedgerPrefix,
) -> DatabricksLedgerPrefix:
    """Require *ledger* to extend the exact ordered append-only prefix."""

    if not isinstance(ledger, DatabricksClusterHourLedger):
        raise TypeError("ledger must be a DatabricksClusterHourLedger")
    if not isinstance(expected, DatabricksLedgerPrefix):
        raise TypeError("expected must be a DatabricksLedgerPrefix")
    if (
        ledger.ledger_id != expected.ledger_id
        or ledger.cap_cluster_hours != expected.cap_cluster_hours
    ):
        raise ValueError("live ledger identity/cap differs from its authorized prefix")
    if (
        len(ledger.reservations) < expected.reservation_count
        or len(ledger.submission_receipts) < expected.submission_receipt_count
        or len(ledger.terminal_actuals) < expected.terminal_actual_count
    ):
        raise ValueError("live ledger is shorter than its authorized prefix")
    observed = _databricks_ledger_prefix_at_counts(
        ledger,
        reservation_count=expected.reservation_count,
        submission_receipt_count=expected.submission_receipt_count,
        terminal_actual_count=expected.terminal_actual_count,
    )
    if observed != expected:
        raise ValueError("live ledger does not extend the authorized ordered prefix")
    return databricks_ledger_prefix(ledger)


def _databricks_ledger_prefix_at_counts(
    ledger: DatabricksClusterHourLedger,
    *,
    reservation_count: int,
    submission_receipt_count: int,
    terminal_actual_count: int,
) -> DatabricksLedgerPrefix:
    prefix_ledger = DatabricksClusterHourLedger(
        ledger_id=ledger.ledger_id,
        cap_cluster_hours=ledger.cap_cluster_hours,
        reservations=ledger.reservations[:reservation_count],
        submission_receipts=ledger.submission_receipts[:submission_receipt_count],
        terminal_actuals=ledger.terminal_actuals[:terminal_actual_count],
    )
    record = databricks_cluster_hour_ledger_to_record(prefix_ledger)
    prefix_sha256 = sha256(
        json.dumps(
            record,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return DatabricksLedgerPrefix(
        ledger_id=ledger.ledger_id,
        cap_cluster_hours=ledger.cap_cluster_hours,
        reservation_count=reservation_count,
        submission_receipt_count=submission_receipt_count,
        terminal_actual_count=terminal_actual_count,
        prefix_sha256=prefix_sha256,
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
        "submission_receipts": [
            {
                "attempt_id": item.attempt_id,
                "run_id": item.run_id,
                "submit_payload_sha256": item.submit_payload_sha256,
                "submit_response_sha256": item.submit_response_sha256,
            }
            for item in ledger.submission_receipts
        ],
        "terminal_actuals": [
            {
                "attempt_id": item.attempt_id,
                "terminal_state": item.terminal_state,
                "actual_cluster_duration_seconds": (
                    item.actual_cluster_duration_seconds
                ),
                "actual_cluster_hours": item.actual_cluster_hours,
                "verification_source": item.verification_source,
                "run_id": item.run_id,
                "submit_payload_sha256": item.submit_payload_sha256,
                "control_plane_status_sha256": item.control_plane_status_sha256,
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
    schema_version = record.get("schema_version")
    if schema_version == _LEGACY_DATABRICKS_CLUSTER_HOUR_LEDGER_SCHEMA_VERSION:
        _require_exact_keys(record, _LEDGER_RECORD_KEYS_V1, "ledger")
    elif schema_version == DATABRICKS_CLUSTER_HOUR_LEDGER_SCHEMA_VERSION:
        _require_exact_keys(record, _LEDGER_RECORD_KEYS, "ledger")
    else:
        raise ValueError("ledger schema_version is invalid")
    if record.get("record_type") != DATABRICKS_CLUSTER_HOUR_LEDGER_RECORD_TYPE:
        raise ValueError("ledger record_type is invalid")
    reservations = tuple(
        _reservation_from_record(item, index=index)
        for index, item in enumerate(_record_sequence(record, "reservations"))
    )
    terminal_actuals = tuple(
        _terminal_actual_from_record(
            item,
            index=index,
            legacy=schema_version
            == _LEGACY_DATABRICKS_CLUSTER_HOUR_LEDGER_SCHEMA_VERSION,
        )
        for index, item in enumerate(_record_sequence(record, "terminal_actuals"))
    )
    submission_receipts = (
        ()
        if schema_version == _LEGACY_DATABRICKS_CLUSTER_HOUR_LEDGER_SCHEMA_VERSION
        else tuple(
            _submission_receipt_from_record(item, index=index)
            for index, item in enumerate(
                _record_sequence(record, "submission_receipts")
            )
        )
    )
    ledger = DatabricksClusterHourLedger(
        ledger_id=_record_string(record, "ledger_id"),
        cap_cluster_hours=_record_number(record, "cap_cluster_hours"),
        reservations=reservations,
        submission_receipts=submission_receipts,
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
    ledger_path = Path(path)
    _require_no_symlink_ancestors(ledger_path, label="ledger path", include_leaf=True)
    raw = json.loads(ledger_path.read_text(encoding="utf-8"))
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


def raise_databricks_cluster_hour_ledger_cap_json(
    path: str | Path,
    *,
    expected_current_cap_cluster_hours: float,
    new_cap_cluster_hours: float,
) -> DatabricksClusterHourLedger:
    """Raise a quiescent ledger cap without discarding any accounting events.

    Publication budget increases must extend the existing append-only ledger;
    creating a new zero-balance ledger would silently forget prior GPU-hours.
    The explicit expected cap makes concurrent or repeated migrations fail
    closed, and active reservations must be reconciled before migration.
    """

    ledger_path = Path(path)
    with _exclusive_ledger_lock(ledger_path):
        ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
        expected_cap = _validated_cap_cluster_hours(
            expected_current_cap_cluster_hours,
            "expected_current_cap_cluster_hours",
        )
        new_cap = _validated_cap_cluster_hours(
            new_cap_cluster_hours,
            "new_cap_cluster_hours",
        )
        if ledger.cap_cluster_hours != expected_cap:
            raise ValueError(
                "cluster-hour ledger cap differs from the expected migration "
                f"source: {ledger.cap_cluster_hours} != {expected_cap}"
            )
        if ledger.active_reserved_cluster_hours != 0:
            raise ValueError(
                "cluster-hour ledger cap can be raised only with zero active "
                "reservations"
            )
        if new_cap <= ledger.cap_cluster_hours:
            raise ValueError(
                "new_cap_cluster_hours must be greater than the current cap"
            )
        updated = DatabricksClusterHourLedger(
            ledger_id=ledger.ledger_id,
            cap_cluster_hours=new_cap,
            reservations=ledger.reservations,
            submission_receipts=ledger.submission_receipts,
            terminal_actuals=ledger.terminal_actuals,
        )
        if (
            updated.reservations != ledger.reservations
            or updated.submission_receipts != ledger.submission_receipts
            or updated.terminal_actuals != ledger.terminal_actuals
            or updated.accounted_cluster_hours != ledger.accounted_cluster_hours
        ):
            raise RuntimeError("cluster-hour cap migration changed accounting events")
        _atomic_write_ledger(ledger_path, updated)
        return updated


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


def reserve_databricks_run_attempt_batch_json(
    ledger_path: str | Path,
    requests: Sequence[DatabricksRunAttemptReservationRequest],
    *,
    batch_validator: DatabricksBatchReservationValidator | None = None,
) -> DatabricksClusterHourLedger:
    """Atomically reserve an exact ordered batch before any runs/submit POST.

    Every payload is isolated as a canonical JSON snapshot while the ledger lock
    is held.  The complete batch is then projected against one live ledger state,
    an optional campaign policy validates that same state and those same
    snapshots, and exactly one durable ledger replacement publishes the batch.
    No prefix of a rejected batch is persisted.
    """

    path = Path(ledger_path)
    with _exclusive_ledger_lock(path):
        if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
            raise TypeError("requests must be a non-empty sequence")
        ordered_requests = tuple(requests)
        if not ordered_requests:
            raise ValueError("requests must be a non-empty sequence")
        if any(
            not isinstance(item, DatabricksRunAttemptReservationRequest)
            for item in ordered_requests
        ):
            raise TypeError(
                "requests entries must be DatabricksRunAttemptReservationRequest"
            )
        if batch_validator is not None and not callable(batch_validator):
            raise TypeError("batch_validator must be callable or None")

        ledger = read_databricks_cluster_hour_ledger_json(path)
        snapshots: list[dict[str, Any]] = []
        canonical_payloads: list[bytes] = []
        reservations: list[DatabricksClusterHourReservation] = []
        for request in ordered_requests:
            snapshot, canonical_payload = canonical_databricks_submit_payload_snapshot(
                request.submit_payload
            )
            snapshots.append(snapshot)
            canonical_payloads.append(canonical_payload)
            reservations.append(
                databricks_submit_payload_reservation(
                    snapshot,
                    attempt_id=request.attempt_id,
                    workload_id=request.workload_id,
                )
            )

        attempt_ids = [item.attempt_id for item in reservations]
        if len(set(attempt_ids)) != len(attempt_ids):
            raise ValueError("batch reservation attempt_id values must be unique")

        projected = ledger
        for reservation in reservations:
            projected = projected.reserve(reservation)

        frozen_reservations = tuple(reservations)
        frozen_snapshots: tuple[Mapping[str, Any], ...] = tuple(snapshots)
        if batch_validator is not None:
            batch_validator(ledger, frozen_reservations, frozen_snapshots)

        for index, (request, snapshot, canonical_payload, reservation) in enumerate(
            zip(
                ordered_requests,
                snapshots,
                canonical_payloads,
                reservations,
                strict=True,
            )
        ):
            _validated_snapshot, validated_payload = (
                canonical_databricks_submit_payload_snapshot(snapshot)
            )
            validated_reservation = databricks_submit_payload_reservation(
                snapshot,
                attempt_id=request.attempt_id,
                workload_id=request.workload_id,
            )
            if (
                validated_payload != canonical_payload
                or validated_reservation != reservation
            ):
                raise ValueError(
                    "batch_validator must not mutate submit payload snapshot "
                    f"at index {index}"
                )

        _atomic_write_ledger(path, projected)
        return projected


def reserve_databricks_run_attempt_batch_authorized_json(
    ledger_path: str | Path,
    requests: Sequence[DatabricksRunAttemptReservationRequest],
    *,
    expected_predecessor_prefix: DatabricksLedgerPrefix,
    batch_validator: DatabricksBatchReservationValidator | None = None,
) -> tuple[DatabricksClusterHourLedger, DatabricksBatchReservationAuthorization]:
    """Atomically reserve a phase batch and issue non-record POST authority.

    The live ledger must equal, rather than merely extend, the predecessor.  A
    competing event therefore cannot interleave between phase authorization
    and the all-or-none reservation write.
    """

    if not isinstance(expected_predecessor_prefix, DatabricksLedgerPrefix):
        raise TypeError("expected_predecessor_prefix has the wrong type")
    observed_predecessor: DatabricksLedgerPrefix | None = None
    frozen_attempt_ids: tuple[str, ...] | None = None
    frozen_payload_digests: tuple[str, ...] | None = None

    def validate_batch(
        live: DatabricksClusterHourLedger,
        reservations: tuple[DatabricksClusterHourReservation, ...],
        snapshots: tuple[Mapping[str, Any], ...],
    ) -> None:
        nonlocal observed_predecessor, frozen_attempt_ids, frozen_payload_digests
        current_prefix = databricks_ledger_prefix(live)
        require_databricks_ledger_prefix(live, expected_predecessor_prefix)
        if current_prefix != expected_predecessor_prefix:
            raise ValueError(
                "atomic batch predecessor is not the complete current ledger"
            )
        if batch_validator is not None:
            batch_validator(live, reservations, snapshots)
        observed_predecessor = current_prefix
        frozen_attempt_ids = tuple(item.attempt_id for item in reservations)
        frozen_payload_digests = tuple(
            sha256(
                canonical_databricks_submit_payload_snapshot(snapshot)[1]
            ).hexdigest()
            for snapshot in snapshots
        )

    updated = reserve_databricks_run_attempt_batch_json(
        ledger_path,
        requests,
        batch_validator=validate_batch,
    )
    if observed_predecessor is None:  # pragma: no cover - validator is mandatory.
        raise RuntimeError("atomic batch predecessor was not observed")
    if frozen_attempt_ids is None or frozen_payload_digests is None:  # pragma: no cover
        raise RuntimeError("atomic batch members were not observed")
    batch_prefix = databricks_ledger_prefix(updated)
    authorization = DatabricksBatchReservationAuthorization(
        predecessor_prefix=observed_predecessor,
        batch_prefix=batch_prefix,
        ledger_path_sha256=databricks_ledger_path_sha256(ledger_path),
        attempt_ids=frozen_attempt_ids,
        submit_payload_sha256s=frozen_payload_digests,
        _issuer=_BATCH_RESERVATION_AUTHORIZATION_ISSUER,
    )
    return updated, authorization


def replay_databricks_run_attempt_batch_authorization_json(
    ledger_path: str | Path,
    requests: Sequence[DatabricksRunAttemptReservationRequest],
    *,
    expected_predecessor_prefix: DatabricksLedgerPrefix,
) -> DatabricksBatchReservationAuthorization:
    """Reissue authority for one exact already-reserved append-only batch.

    This is the crash-recovery counterpart to atomic batch reservation.  It
    never mutates the ledger: under the ledger lock it re-snapshots the exact
    requested payload bytes, requires the predecessor, and proves that the next
    reservation suffix is precisely those members in order.  Later receipts or
    terminals may extend the batch prefix without weakening the proof.
    """

    if not isinstance(expected_predecessor_prefix, DatabricksLedgerPrefix):
        raise TypeError("expected_predecessor_prefix has the wrong type")
    path = Path(ledger_path)
    with _exclusive_ledger_lock(path):
        if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
            raise TypeError("requests must be a non-empty sequence")
        ordered_requests = tuple(requests)
        if not ordered_requests:
            raise ValueError("requests must be a non-empty sequence")
        if any(
            not isinstance(item, DatabricksRunAttemptReservationRequest)
            for item in ordered_requests
        ):
            raise TypeError(
                "requests entries must be DatabricksRunAttemptReservationRequest"
            )
        ledger = read_databricks_cluster_hour_ledger_json(path)
        require_databricks_ledger_prefix(ledger, expected_predecessor_prefix)
        reservations: list[DatabricksClusterHourReservation] = []
        payload_digests: list[str] = []
        for request in ordered_requests:
            snapshot, canonical = canonical_databricks_submit_payload_snapshot(
                request.submit_payload
            )
            reservations.append(
                databricks_submit_payload_reservation(
                    snapshot,
                    attempt_id=request.attempt_id,
                    workload_id=request.workload_id,
                )
            )
            payload_digests.append(sha256(canonical).hexdigest())
        attempt_ids = tuple(item.attempt_id for item in reservations)
        if len(set(attempt_ids)) != len(attempt_ids):
            raise ValueError("batch replay attempt_id values must be unique")
        start = expected_predecessor_prefix.reservation_count
        stop = start + len(reservations)
        if ledger.reservations[start:stop] != tuple(reservations):
            raise ValueError("live ledger does not contain the exact reserved batch")
        batch_prefix = _databricks_ledger_prefix_at_counts(
            ledger,
            reservation_count=stop,
            submission_receipt_count=(
                expected_predecessor_prefix.submission_receipt_count
            ),
            terminal_actual_count=expected_predecessor_prefix.terminal_actual_count,
        )
        return DatabricksBatchReservationAuthorization(
            predecessor_prefix=expected_predecessor_prefix,
            batch_prefix=batch_prefix,
            ledger_path_sha256=databricks_ledger_path_sha256(path),
            attempt_ids=attempt_ids,
            submit_payload_sha256s=tuple(payload_digests),
            _issuer=_BATCH_RESERVATION_AUTHORIZATION_ISSUER,
        )


def require_databricks_batch_reservation_authorization(
    authorization: object,
    *,
    expected_predecessor_prefix: DatabricksLedgerPrefix,
    expected_attempt_ids: Sequence[str],
    expected_submit_payload_sha256s: Sequence[str],
) -> DatabricksLedgerPrefix:
    """Validate an exact atomic batch capability and return its post-batch prefix."""

    if not isinstance(authorization, DatabricksBatchReservationAuthorization):
        raise TypeError(
            "atomic submission requires DatabricksBatchReservationAuthorization"
        )
    if (
        authorization.predecessor_prefix != expected_predecessor_prefix
        or authorization.attempt_ids != tuple(expected_attempt_ids)
        or authorization.submit_payload_sha256s
        != tuple(expected_submit_payload_sha256s)
    ):
        raise ValueError("atomic batch reservation authority binding drift")
    return authorization.batch_prefix


def require_databricks_batch_terminal_closure(
    ledger: DatabricksClusterHourLedger,
    authorization: DatabricksBatchReservationAuthorization,
    *,
    require_complete_current_prefix: bool = True,
) -> DatabricksLedgerPrefix:
    """Close the exact ordered receipt/terminal suffix for one atomic batch.

    The returned prefix stops at this phase even when a durable replay occurs
    after later campaign events.  Publication collectors set
    ``require_complete_current_prefix`` so unrelated or interleaved events
    cannot be absorbed into newly issued authority.
    """

    if not isinstance(ledger, DatabricksClusterHourLedger):
        raise TypeError("ledger must be a DatabricksClusterHourLedger")
    if not isinstance(authorization, DatabricksBatchReservationAuthorization):
        raise TypeError("authorization has the wrong type")
    require_databricks_ledger_prefix(ledger, authorization.batch_prefix)
    attempts = authorization.attempt_ids
    payload_digests = authorization.submit_payload_sha256s
    predecessor = authorization.predecessor_prefix
    receipt_start = predecessor.submission_receipt_count
    receipt_stop = receipt_start + len(attempts)
    terminal_start = predecessor.terminal_actual_count
    terminal_stop = terminal_start + len(attempts)
    receipts = ledger.submission_receipts[receipt_start:receipt_stop]
    terminals = ledger.terminal_actuals[terminal_start:terminal_stop]
    if tuple(item.attempt_id for item in receipts) != attempts:
        raise ValueError("ledger receipt suffix is not the exact atomic batch")
    if tuple(item.submit_payload_sha256 for item in receipts) != payload_digests:
        raise ValueError("ledger receipt suffix payload digest drift")
    if tuple(item.attempt_id for item in terminals) != attempts:
        raise ValueError("ledger terminal suffix is not the exact atomic batch")
    if tuple(item.submit_payload_sha256 for item in terminals) != payload_digests:
        raise ValueError("ledger terminal suffix payload digest drift")
    terminal_prefix = databricks_ledger_prefix_at_counts(
        ledger,
        reservation_count=authorization.batch_prefix.reservation_count,
        submission_receipt_count=receipt_stop,
        terminal_actual_count=terminal_stop,
    )
    if require_complete_current_prefix and databricks_ledger_prefix(ledger) != (
        terminal_prefix
    ):
        raise ValueError("atomic batch terminal prefix is not the complete live ledger")
    return terminal_prefix


def require_databricks_publication_batch_admission(
    ledger: DatabricksClusterHourLedger,
    authorization: DatabricksBatchReservationAuthorization,
) -> DatabricksLedgerPrefix:
    """Replay the global publication admission policy at a batch predecessor."""

    if not isinstance(ledger, DatabricksClusterHourLedger):
        raise TypeError("ledger must be a DatabricksClusterHourLedger")
    if not isinstance(authorization, DatabricksBatchReservationAuthorization):
        raise TypeError("authorization has the wrong type")
    require_databricks_ledger_prefix(ledger, authorization.batch_prefix)
    predecessor = authorization.predecessor_prefix
    predecessor_ledger = DatabricksClusterHourLedger(
        ledger_id=ledger.ledger_id,
        cap_cluster_hours=ledger.cap_cluster_hours,
        reservations=ledger.reservations[: predecessor.reservation_count],
        submission_receipts=ledger.submission_receipts[
            : predecessor.submission_receipt_count
        ],
        terminal_actuals=ledger.terminal_actuals[: predecessor.terminal_actual_count],
    )
    if databricks_ledger_prefix(predecessor_ledger) != predecessor:
        raise ValueError("publication batch predecessor history drift")
    reservations = ledger.reservations[
        predecessor.reservation_count : authorization.batch_prefix.reservation_count
    ]
    if tuple(item.attempt_id for item in reservations) != authorization.attempt_ids:
        raise ValueError("publication batch reservation suffix identity drift")
    if tuple(item.submit_payload_sha256 for item in reservations) != (
        authorization.submit_payload_sha256s
    ):
        raise ValueError("publication batch reservation suffix payload drift")
    proposed_tasks = sum(len(item.task_timeout_seconds) for item in reservations)
    proposed_hours = sum(item.reserved_cluster_hours for item in reservations)
    if predecessor_ledger.cap_cluster_hours != MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS:
        raise ValueError("publication batch requires the 1024-hour campaign ledger")
    if (
        predecessor_ledger.active_reserved_task_count + proposed_tasks
        > MAX_DATABRICKS_ACTIVE_RESERVED_TASKS
    ):
        raise ValueError("publication batch exceeds the global 16-task guard")
    if (
        predecessor_ledger.active_reserved_cluster_hours + proposed_hours
        > MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS
    ):
        raise ValueError("publication batch exceeds the 900-hour active guard")
    if (
        predecessor_ledger.accounted_cluster_hours
        + proposed_hours
        + DATABRICKS_PUBLICATION_UNRESERVED_HEADROOM_HOURS
        > MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
    ):
        raise ValueError("publication batch consumes the 124-hour headroom")
    return authorization.batch_prefix


def record_databricks_run_submission_receipt_json(
    ledger_path: str | Path,
    *,
    attempt_id: str,
    submit_response: Mapping[str, Any],
) -> DatabricksClusterHourLedger:
    """Bind a reserved attempt to the run ID returned by ``runs/submit``.

    This must be called on the direct response from the POST.  Only the run ID
    and a digest of the closed response are persisted; credentials and response
    bodies never enter the ledger.
    """

    if not isinstance(submit_response, Mapping):
        raise TypeError("submit_response must be a mapping")
    response_bytes = _canonical_json_bytes(submit_response, "submit_response")
    run_id = _validated_databricks_run_id(
        submit_response.get("run_id"), "submit_response.run_id"
    )
    path = Path(ledger_path)
    with _exclusive_ledger_lock(path):
        ledger = read_databricks_cluster_hour_ledger_json(path)
        reservation = next(
            (item for item in ledger.reservations if item.attempt_id == attempt_id),
            None,
        )
        if reservation is None:
            raise ValueError(
                f"attempt {attempt_id!r} has no pre-submission reservation"
            )
        if attempt_id in ledger.closed_attempt_ids:
            raise ValueError("cannot receipt-bind an already terminal attempt")
        receipt = DatabricksRunSubmissionReceipt(
            attempt_id=attempt_id,
            run_id=run_id,
            submit_payload_sha256=reservation.submit_payload_sha256,
            submit_response_sha256=sha256(response_bytes).hexdigest(),
        )
        updated = ledger.record_submission_receipt(receipt)
        if updated is not ledger:
            _atomic_write_ledger(path, updated)
        return updated


def record_databricks_verified_run_terminal_actual_json(
    ledger_path: str | Path,
    *,
    attempt_id: str,
    run_record: Mapping[str, Any],
) -> DatabricksClusterHourLedger:
    """Reconcile from a direct terminal ``jobs/runs/get`` response.

    Receipt-bound attempts deliberately cannot use the legacy scalar terminal
    API.  The run ID, submit-payload digest, and raw control-plane response digest
    are retained in the append-only terminal event.
    """

    if not isinstance(run_record, Mapping):
        raise TypeError("run_record must be a mapping")
    status_bytes = _canonical_json_bytes(run_record, "run_record")
    run_id = _validated_databricks_run_id(run_record.get("run_id"), "run_record.run_id")
    state = run_record.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("run_record.state must be an object")
    life_cycle_state = state.get("life_cycle_state")
    result_state = state.get("result_state")
    terminal_state = _ledger_terminal_state(life_cycle_state, result_state)
    tasks = run_record.get("tasks")
    if (
        isinstance(tasks, (str, bytes, bytearray))
        or not isinstance(tasks, Sequence)
        or not tasks
        or any(not isinstance(task, Mapping) for task in tasks)
    ):
        raise ValueError("verified Databricks runs must contain nonempty task objects")
    task_keys: list[str] = []
    task_run_ids: list[str] = []
    duration_seconds = 0.0
    for index, raw_task in enumerate(tasks):
        task = raw_task
        assert isinstance(task, Mapping)
        task_key = task.get("task_key")
        task_run_id = _validated_databricks_run_id(
            task.get("run_id"), f"verified Databricks task {index}.run_id"
        )
        if not isinstance(task_key, str) or not task_key:
            raise ValueError(f"verified Databricks task {index} has no task_key")
        task_keys.append(task_key)
        task_run_ids.append(task_run_id)
        task_state = task.get("state")
        if not isinstance(task_state, Mapping):
            raise ValueError("verified Databricks task state must be an object")
        task_terminal_state = _ledger_terminal_state(
            task_state.get("life_cycle_state"),
            task_state.get("result_state"),
        )
        if terminal_state == "succeeded" and task_terminal_state != "succeeded":
            raise ValueError("successful Databricks run contains a non-success task")
        start_time = task.get("start_time")
        end_time = task.get("end_time")
        never_started = (start_time is None and end_time is None) or (
            type(start_time) is int
            and start_time == 0
            and type(end_time) is int
            and end_time == 0
        )
        if (
            never_started
            and terminal_state != "succeeded"
            and task_terminal_state != "succeeded"
        ):
            task_duration_seconds = 0.0
        elif (
            type(start_time) is int
            and type(end_time) is int
            and start_time >= 0
            and end_time > start_time
        ):
            task_duration_seconds = (end_time - start_time) / 1000.0
        else:
            raise ValueError(
                "verified Databricks task requires increasing millisecond times, "
                "or canonical zero/missing times for a never-started failed task"
            )
        duration_seconds += task_duration_seconds
    if len(set(task_keys)) != len(task_keys) or len(set(task_run_ids)) != len(
        task_run_ids
    ):
        raise ValueError("verified Databricks task keys and run IDs must be unique")

    path = Path(ledger_path)
    with _exclusive_ledger_lock(path):
        ledger = read_databricks_cluster_hour_ledger_json(path)
        receipt = next(
            (
                item
                for item in ledger.submission_receipts
                if item.attempt_id == attempt_id
            ),
            None,
        )
        if receipt is None:
            raise ValueError(
                f"attempt {attempt_id!r} has no persisted runs/submit receipt"
            )
        if receipt.run_id != run_id:
            raise ValueError("runs/get run_id does not match the submission receipt")
        reservation = next(
            item for item in ledger.reservations if item.attempt_id == attempt_id
        )
        if len(reservation.task_timeout_seconds) != len(tasks):
            raise ValueError(
                "runs/get task count differs from the reserved submit payload"
            )
        actual = DatabricksClusterHourTerminalActual(
            attempt_id=attempt_id,
            terminal_state=terminal_state,
            actual_cluster_duration_seconds=duration_seconds,
            verification_source="direct_databricks_runs_get",
            run_id=run_id,
            submit_payload_sha256=receipt.submit_payload_sha256,
            control_plane_status_sha256=sha256(status_bytes).hexdigest(),
        )
        existing = next(
            (item for item in ledger.terminal_actuals if item.attempt_id == attempt_id),
            None,
        )
        if existing is not None:
            if (
                existing.attempt_id == actual.attempt_id
                and existing.terminal_state == actual.terminal_state
                and existing.actual_cluster_duration_seconds
                == actual.actual_cluster_duration_seconds
                and existing.verification_source == actual.verification_source
                and existing.run_id == actual.run_id
                and existing.submit_payload_sha256 == actual.submit_payload_sha256
                and existing.control_plane_status_sha256
                == actual.control_plane_status_sha256
            ):
                return ledger
            raise ValueError("terminal runs/get reconciliation differs from the ledger")
        updated = ledger.record_terminal_actual(actual)
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
        if any(item.attempt_id == attempt_id for item in ledger.submission_receipts):
            raise ValueError(
                "receipt-bound attempts require direct runs/get terminal reconciliation"
            )
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


def _submission_receipt_from_record(
    value: Any,
    *,
    index: int,
) -> DatabricksRunSubmissionReceipt:
    if not isinstance(value, Mapping):
        raise ValueError(f"ledger.submission_receipts[{index}] must be an object")
    label = f"ledger.submission_receipts[{index}]"
    _require_exact_keys(value, _SUBMISSION_RECEIPT_RECORD_KEYS, label)
    return DatabricksRunSubmissionReceipt(
        attempt_id=_record_string(value, "attempt_id"),
        run_id=_record_string(value, "run_id"),
        submit_payload_sha256=_record_string(value, "submit_payload_sha256"),
        submit_response_sha256=_record_string(value, "submit_response_sha256"),
    )


def _terminal_actual_from_record(
    value: Any,
    *,
    index: int,
    legacy: bool,
) -> DatabricksClusterHourTerminalActual:
    if not isinstance(value, Mapping):
        raise ValueError(f"ledger.terminal_actuals[{index}] must be an object")
    label = f"ledger.terminal_actuals[{index}]"
    _require_exact_keys(
        value,
        _TERMINAL_RECORD_KEYS_V1 if legacy else _TERMINAL_RECORD_KEYS,
        label,
    )
    actual = DatabricksClusterHourTerminalActual(
        attempt_id=_record_string(value, "attempt_id"),
        terminal_state=_record_string(value, "terminal_state"),
        actual_cluster_duration_seconds=_record_number(
            value,
            "actual_cluster_duration_seconds",
        ),
        verification_source=(
            "legacy_manual" if legacy else _record_string(value, "verification_source")
        ),
        run_id=None if legacy else _record_optional_string(value, "run_id"),
        submit_payload_sha256=(
            None if legacy else _record_optional_string(value, "submit_payload_sha256")
        ),
        control_plane_status_sha256=(
            None
            if legacy
            else _record_optional_string(value, "control_plane_status_sha256")
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


def _validated_databricks_run_id(value: Any, field_name: str) -> str:
    if type(value) is int and value > 0:
        return str(value)
    if isinstance(value, str) and _DATABRICKS_RUN_ID_RE.fullmatch(value):
        return value
    raise ValueError(
        f"{field_name} must be a strictly positive canonical decimal Databricks run ID"
    )


def _validated_cap_cluster_hours(value: Any, field_name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0 < float(value) <= MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
    ):
        raise ValueError(
            f"{field_name} must be a positive finite number no greater than "
            f"{MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS}"
        )
    return float(value)


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


def _canonical_json_bytes(value: Mapping[str, Any], field_name: str) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain canonical JSON values") from exc


def _ledger_terminal_state(life_cycle_state: Any, result_state: Any) -> str:
    if life_cycle_state not in {
        "TERMINATED",
        "SKIPPED",
        "INTERNAL_ERROR",
        "BLOCKED",
    }:
        raise ValueError("runs/get response is not terminal")
    if life_cycle_state == "SKIPPED":
        return "skipped"
    if life_cycle_state in {"INTERNAL_ERROR", "BLOCKED"}:
        return "internal_error"
    mapping = {
        "SUCCESS": "succeeded",
        "FAILED": "failed",
        "TIMEDOUT": "timed_out",
        "CANCELED": "canceled",
        "UPSTREAM_FAILED": "failed",
        "EXCLUDED": "skipped",
    }
    terminal_state = mapping.get(result_state)
    if terminal_state is None:
        raise ValueError(
            f"unsupported terminal Databricks result_state: {result_state!r}"
        )
    return terminal_state


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


def _record_optional_string(value: Mapping[str, Any], field_name: str) -> str | None:
    result = value.get(field_name)
    if result is None:
        return None
    if not isinstance(result, str) or not result:
        raise ValueError(f"{field_name} must be null or a non-empty string")
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

    _require_no_symlink_ancestors(path, label="ledger path", include_leaf=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_ancestors(path, label="ledger path", include_leaf=True)
    lock_path = path.with_name(f".{path.name}.lock")
    _require_no_symlink_ancestors(
        lock_path, label="ledger lock path", include_leaf=True
    )
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    with os.fdopen(descriptor, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_write_ledger(
    path: Path,
    ledger: DatabricksClusterHourLedger,
) -> None:
    _require_no_symlink_ancestors(path, label="ledger path", include_leaf=True)
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
        _fsync_directory(path.parent)
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"ledger directory durability target is invalid: {path}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_no_symlink_ancestors(
    path: Path, *, label: str, include_leaf: bool
) -> None:
    candidates = ((path,) if include_leaf else ()) + tuple(path.parents)
    for candidate in candidates:
        if candidate.is_symlink():
            raise ValueError(f"{label} cannot traverse a symlink: {candidate}")


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

    raise_cap_parser = subparsers.add_parser("raise-cap")
    raise_cap_parser.add_argument("--ledger-json", required=True)
    raise_cap_parser.add_argument(
        "--expected-current-cap-cluster-hours",
        type=float,
        required=True,
    )
    raise_cap_parser.add_argument(
        "--new-cap-cluster-hours",
        type=float,
        required=True,
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
        elif args.action == "raise-cap":
            ledger = raise_databricks_cluster_hour_ledger_cap_json(
                args.ledger_json,
                expected_current_cap_cluster_hours=(
                    args.expected_current_cap_cluster_hours
                ),
                new_cap_cluster_hours=args.new_cap_cluster_hours,
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
