"""Mac-safe remote coordination for publication full-score artifact trees.

The full-score ready trees can approach 500 GB and therefore never cross the
controller boundary.  A single-node CPU Databricks task mounts the governed UC
Volume, invokes the exact validators from :mod:`full_score_execution`, and
publishes only compact canonical result and attestation records.  The Mac
controller downloads those bounded records through the authenticated Files API
and corroborates ready-tree presence or deletion through paginated metadata.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import stat
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, cast

from document_kv_cache import full_score_execution as full_score
from document_kv_cache.databricks_job import (
    DEFAULT_DATABRICKS_DATA_SECURITY_MODE,
)
from document_kv_cache.databricks_runs import (
    DATABRICKS_VOLUME_FILE_MAX_DOWNLOAD_BYTES,
    DatabricksWorkspaceConfig,
    bind_databricks_run_idempotency_token,
    download_databricks_volume_file_bytes,
    get_databricks_run,
    list_databricks_volume_directory,
    require_databricks_run_idempotency_token,
    submit_databricks_run,
    upload_databricks_volume_file_bytes_exclusive,
)
from document_kv_cache.databricks_resource_ledger import (
    databricks_ledger_prefix_from_record,
)
from document_kv_cache.publication_inputs import FullScoreInventory
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
    PUBLICATION_CAMPAIGN_CPU_COORDINATOR_NODE_TYPE_ID,
    PUBLICATION_CAMPAIGN_CPU_COORDINATOR_SPARK_VERSION,
    PUBLICATION_CAMPAIGN_FULL_SCORE_REMOTE_CPU_JOBS,
    PUBLICATION_CAMPAIGN_FULL_SCORE_REMOTE_CPU_TIMEOUT_SECONDS,
)
from document_kv_cache.serving_env import VLLM_RUNTIME_LOCK_SHA256


FULL_SCORE_REMOTE_COORDINATOR_REQUEST_RECORD_TYPE: Final = (
    "cachet.full_score_remote_coordinator_request.v2"
)
FULL_SCORE_REMOTE_COORDINATOR_REQUEST_SCHEMA_VERSION: Final = 2
FULL_SCORE_REMOTE_COORDINATOR_REQUEST_AUTHORIZATION_RECORD_TYPE: Final = (
    "cachet.full_score_remote_coordinator_request_authorization.v2"
)
FULL_SCORE_REMOTE_COORDINATOR_ATTESTATION_RECORD_TYPE: Final = (
    "cachet.full_score_remote_coordinator_attestation.v1"
)
FULL_SCORE_REMOTE_COORDINATOR_ATTESTATION_SCHEMA_VERSION: Final = 1
FULL_SCORE_REMOTE_FINAL_COVERAGE_RECORD_TYPE: Final = (
    "cachet.full_score_remote_final_coverage.v1"
)
FULL_SCORE_REMOTE_FINAL_COVERAGE_SCHEMA_VERSION: Final = 1
FULL_SCORE_REMOTE_COORDINATOR_ACTIONS: Final = frozenset(
    {"producer_ready", "consumer_evidence"}
)
FULL_SCORE_REMOTE_COORDINATOR_NODE_TYPE_ID: Final = (
    PUBLICATION_CAMPAIGN_CPU_COORDINATOR_NODE_TYPE_ID
)
FULL_SCORE_REMOTE_COORDINATOR_SPARK_VERSION: Final = (
    PUBLICATION_CAMPAIGN_CPU_COORDINATOR_SPARK_VERSION
)
FULL_SCORE_REMOTE_COORDINATOR_JOB_COUNT: Final = (
    PUBLICATION_CAMPAIGN_FULL_SCORE_REMOTE_CPU_JOBS
)
FULL_SCORE_REMOTE_COORDINATOR_TIMEOUT_SECONDS: Final = (
    PUBLICATION_CAMPAIGN_FULL_SCORE_REMOTE_CPU_TIMEOUT_SECONDS
)
FULL_SCORE_REMOTE_CAMPAIGN_CLOSED_RECORD_SHA256: Final = (
    PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256
)
FULL_SCORE_REMOTE_COORDINATOR_PARAMETERS_MAX_BYTES: Final = 9_500
FULL_SCORE_REMOTE_COMPACT_RECORD_MAX_BYTES: Final = 16 * 1024 * 1024
FULL_SCORE_REMOTE_COORDINATOR_OUTPUT_MAX_BYTES: Final = 2 * 1024 * 1024
FULL_SCORE_REMOTE_EVIDENCE_DELETION_MAX_BYTES: Final = 64 * 1024
FULL_SCORE_REMOTE_EVIDENCE_PAIR_OUTPUT_MAX_BYTES: Final = (
    len(full_score.FULL_SCORE_METHODS) * full_score.FULL_SCORE_MAX_TOKENS * 8
)
FULL_SCORE_REMOTE_EVIDENCE_PAIR_METADATA_MAX_BYTES: Final = 12 * 1024
FULL_SCORE_REMOTE_EVIDENCE_PAIR_MAX_BYTES: Final = (
    FULL_SCORE_REMOTE_EVIDENCE_PAIR_OUTPUT_MAX_BYTES
    + FULL_SCORE_REMOTE_EVIDENCE_PAIR_METADATA_MAX_BYTES
)
FULL_SCORE_REMOTE_EVIDENCE_SHARD_ENVELOPE_MAX_BYTES: Final = 2 * 1024 * 1024
FULL_SCORE_REMOTE_CAS_MAX_BINDINGS: Final = 400
FULL_SCORE_REMOTE_CAS_BINDING_MAX_BYTES: Final = 4 * 1024
FULL_SCORE_REMOTE_CAS_MAX_BYTES: Final = (
    full_score.FULL_SCORE_PUBLICATION_ITEM_COUNT
    * FULL_SCORE_REMOTE_EVIDENCE_PAIR_MAX_BYTES
    + full_score.FULL_SCORE_PUBLICATION_SHARD_COUNT
    * FULL_SCORE_REMOTE_EVIDENCE_SHARD_ENVELOPE_MAX_BYTES
    + full_score.FULL_SCORE_PUBLICATION_SHARD_COUNT
    * FULL_SCORE_REMOTE_EVIDENCE_DELETION_MAX_BYTES
    + 2
    * 2
    * (
        full_score.FULL_SCORE_PUBLICATION_SHARD_COUNT
        // full_score.FULL_SCORE_DEFAULT_MAX_SHARDS_PER_WAVE
    )
    * FULL_SCORE_REMOTE_COORDINATOR_OUTPUT_MAX_BYTES
)
FULL_SCORE_REMOTE_CAS_MIRROR_MAX_BYTES: Final = (
    FULL_SCORE_REMOTE_CAS_MAX_BYTES
    + FULL_SCORE_REMOTE_CAS_MAX_BINDINGS * FULL_SCORE_REMOTE_CAS_BINDING_MAX_BYTES
)
FULL_SCORE_REMOTE_CONTROLLER_POST_INTENT_RECORD_TYPE: Final = (
    "cachet.full_score_remote_controller_post_intent.v1"
)
FULL_SCORE_REMOTE_CONTROLLER_UPLOAD_RECEIPT_RECORD_TYPE: Final = (
    "cachet.full_score_remote_controller_upload_receipt.v1"
)
FULL_SCORE_REMOTE_CONTROLLER_RUNNER_UPLOAD_RECEIPT_RECORD_TYPE: Final = (
    "cachet.full_score_remote_controller_runner_upload_receipt.v1"
)
FULL_SCORE_REMOTE_CONTROLLER_SUBMIT_RESPONSE_RECORD_TYPE: Final = (
    "cachet.full_score_remote_controller_submit_response.v1"
)
FULL_SCORE_REMOTE_CONTROLLER_RUN_RECEIPT_RECORD_TYPE: Final = (
    "cachet.full_score_remote_controller_runs_get_receipt.v1"
)
FULL_SCORE_REMOTE_CONTROLLER_AUTHORIZATION_RECORD_TYPE: Final = (
    "cachet.full_score_remote_controller_authorization.v1"
)
FULL_SCORE_REMOTE_CONTROLLER_SCHEMA_VERSION: Final = 1
FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SCRIPT = r"""from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def _mount_path(uri: str) -> str:
    prefix = "dbfs:/Volumes/"
    if not uri.startswith(prefix):
        raise ValueError("coordinator artifacts must use dbfs:/Volumes URIs")
    return "/Volumes/" + uri.removeprefix(prefix)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--package-wheel-uri", required=True)
    parser.add_argument("--package-wheel-sha256", required=True)
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--expected-request-file-sha256", required=True)
    parser.add_argument("--expected-request-record-sha256", required=True)
    args = parser.parse_args(argv)
    if _sha256(__file__) != args.runner_sha256:
        raise ValueError("full-score remote coordinator runner SHA-256 drift")
    wheel = _mount_path(args.package_wheel_uri)
    request = _mount_path(args.request_json)
    if _sha256(request) != args.expected_request_file_sha256:
        raise ValueError("full-score remote coordinator request file drift")
    with open(request, "rb") as handle:
        request_record = json.load(handle)
    package = request_record.get("package")
    if not isinstance(package, dict) or package != {
        "runner_sha256": args.runner_sha256,
        "wheel_sha256": args.package_wheel_sha256,
        "wheel_uri": args.package_wheel_uri,
    }:
        raise ValueError("full-score remote coordinator package authority drift")
    if _sha256(wheel) != args.package_wheel_sha256:
        raise ValueError("full-score remote coordinator package wheel drift")
    sys.path.insert(0, wheel)
    from document_kv_cache.full_score_remote_control import coordinator_main

    return coordinator_main(
        [
            "run-coordinator",
            "--request-json",
            request,
            "--expected-request-file-sha256",
            args.expected_request_file_sha256,
            "--expected-request-record-sha256",
            args.expected_request_record_sha256,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
"""
FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SHA256: Final = sha256(
    FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SCRIPT.encode("utf-8")
).hexdigest()

_SHA256_LENGTH = 64
_REMOTE_AUTHORIZATION_ISSUER = object()


@dataclass(frozen=True, slots=True)
class FullScoreRemoteCoordinatorJobConfig:
    """Immutable CPU coordinator topology and staged artifact identities."""

    runner_python_file: str
    package_wheel_uri: str
    package_wheel_sha256: str
    single_user_name: str
    runner_sha256: str = FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SHA256
    node_type_id: str = FULL_SCORE_REMOTE_COORDINATOR_NODE_TYPE_ID
    spark_version: str = FULL_SCORE_REMOTE_COORDINATOR_SPARK_VERSION
    data_security_mode: str = DEFAULT_DATABRICKS_DATA_SECURITY_MODE
    timeout_seconds: int = FULL_SCORE_REMOTE_COORDINATOR_TIMEOUT_SECONDS
    custom_tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _volume_file_uri(self.runner_python_file, "runner_python_file")
        _volume_file_uri(self.package_wheel_uri, "package_wheel_uri")
        _require_sha256(self.runner_sha256, "runner_sha256")
        _require_sha256(self.package_wheel_sha256, "package_wheel_sha256")
        if self.runner_sha256 != FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SHA256:
            raise ValueError("remote coordinator runner SHA-256 drift")
        if self.node_type_id != FULL_SCORE_REMOTE_COORDINATOR_NODE_TYPE_ID:
            raise ValueError("remote coordinator must use c5d.4xlarge")
        if self.spark_version != FULL_SCORE_REMOTE_COORDINATOR_SPARK_VERSION:
            raise ValueError("remote coordinator Databricks Runtime drift")
        if (
            self.data_security_mode != "SINGLE_USER"
            or not isinstance(self.single_user_name, str)
            or not self.single_user_name
            or self.single_user_name.strip() != self.single_user_name
        ):
            raise ValueError("remote coordinator requires a SINGLE_USER principal")
        if self.timeout_seconds != FULL_SCORE_REMOTE_COORDINATOR_TIMEOUT_SECONDS:
            raise ValueError("remote coordinator timeout is frozen to two hours")
        tags = dict(self.custom_tags)
        reserved_tags = {"ResourceClass", "campaign_closure", "purpose"}
        if reserved_tags.intersection(tags) or any(
            not isinstance(key, str)
            or not key
            or key.strip() != key
            or not isinstance(value, str)
            or not value
            or value.strip() != value
            for key, value in tags.items()
        ):
            raise ValueError("remote coordinator custom_tags are invalid or reserved")
        object.__setattr__(self, "custom_tags", MappingProxyType(tags))

    def to_record(self) -> dict[str, Any]:
        """Return every payload-affecting CPU job pin as canonical data."""

        return {
            "custom_tags": dict(self.custom_tags),
            "data_security_mode": self.data_security_mode,
            "node_type_id": self.node_type_id,
            "package_wheel_sha256": self.package_wheel_sha256,
            "package_wheel_uri": self.package_wheel_uri,
            "runner_python_file": self.runner_python_file,
            "runner_sha256": self.runner_sha256,
            "single_user_name": self.single_user_name,
            "spark_version": self.spark_version,
            "timeout_seconds": self.timeout_seconds,
        }


def _full_score_remote_coordinator_job_config_from_record(
    value: Mapping[str, Any],
) -> FullScoreRemoteCoordinatorJobConfig:
    record = _json_mapping(value, "remote coordinator job config")
    if set(record) != {
        "custom_tags",
        "data_security_mode",
        "node_type_id",
        "package_wheel_sha256",
        "package_wheel_uri",
        "runner_python_file",
        "runner_sha256",
        "single_user_name",
        "spark_version",
        "timeout_seconds",
    }:
        raise ValueError("remote coordinator job config schema drift")
    raw_tags = record.get("custom_tags")
    if not isinstance(raw_tags, Mapping):
        raise ValueError("remote coordinator job config custom_tags drift")
    config = FullScoreRemoteCoordinatorJobConfig(
        runner_python_file=cast(str, record.get("runner_python_file")),
        package_wheel_uri=cast(str, record.get("package_wheel_uri")),
        package_wheel_sha256=cast(str, record.get("package_wheel_sha256")),
        single_user_name=cast(str, record.get("single_user_name")),
        runner_sha256=cast(str, record.get("runner_sha256")),
        node_type_id=cast(str, record.get("node_type_id")),
        spark_version=cast(str, record.get("spark_version")),
        data_security_mode=cast(str, record.get("data_security_mode")),
        timeout_seconds=cast(int, record.get("timeout_seconds")),
        custom_tags=cast(Mapping[str, str], raw_tags),
    )
    if config.to_record() != record:
        raise ValueError("remote coordinator job config normalization drift")
    return config


class FullScoreRemoteCoordinatorRequestAuthorization(Mapping[str, Any]):
    """Issuer-only capability for one reviewed coordinator request.

    The canonical request bytes are private and every mapping access returns a
    decoded copy.  A caller can therefore inspect the request like a mapping,
    but cannot turn a resealed raw mapping into controller authority.
    """

    __slots__ = (
        "_authorization_bytes",
        "_controller_lease_root",
        "_request_bytes",
    )

    def __init__(
        self,
        *,
        request: Mapping[str, Any],
        authorization_record: Mapping[str, Any],
        controller_lease_root: str | Path,
        _issuer: object,
    ) -> None:
        if _issuer is not _REMOTE_AUTHORIZATION_ISSUER:
            raise TypeError(
                "remote coordinator request authority requires direct construction"
            )
        canonical_request = _json_mapping(request, "remote coordinator request")
        validate_full_score_remote_coordinator_request(canonical_request)
        canonical_authorization = _json_mapping(
            authorization_record,
            "remote coordinator request authorization",
        )
        _validate_remote_coordinator_request_authorization_record(
            canonical_authorization,
            request=canonical_request,
        )
        lease_root = _normalized_controller_lease_root(controller_lease_root)
        singleton = _required_mapping(canonical_request, "controller_singleton")
        if singleton.get("controller_lease_root_sha256") != (
            _controller_lease_root_sha256(lease_root)
        ):
            raise ValueError("remote coordinator controller lease root binding drift")
        self._request_bytes = _pretty_json_bytes(canonical_request)
        self._authorization_bytes = _pretty_json_bytes(canonical_authorization)
        self._controller_lease_root = lease_root

    def __getitem__(self, key: str) -> Any:
        return self.to_record()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_record())

    def __len__(self) -> int:
        return len(self.to_record())

    @property
    def authorization_record_sha256(self) -> str:
        return cast(str, self.authorization_record()["closed_record_sha256"])

    @property
    def controller_lease_root(self) -> Path:
        return self._controller_lease_root

    def authorization_record(self) -> dict[str, Any]:
        return _json_object(
            self._authorization_bytes,
            "remote coordinator request authorization",
        )

    def to_record(self) -> dict[str, Any]:
        return _json_object(self._request_bytes, "remote coordinator request")


@dataclass(frozen=True, slots=True, init=False)
class FullScoreRemoteTreeAuthorization:
    """Issuer-only controller authority over one remotely verified wave tree."""

    action: str
    execution_plan_sha256: str
    wave_index: int
    durable_output_root: str
    request_sha256: str
    result_uri: str
    result_file_sha256: str
    result_record_sha256: str
    result_record: Mapping[str, Any]
    attestation_uri: str
    attestation_file_sha256: str
    attestation_record_sha256: str
    coordinator_run_id: str
    coordinator_run_record_sha256: str
    controller_authorization_record_sha256: str
    runs_get_receipt_record_sha256: str
    phase_terminal_record_sha256: str
    evidence_bindings: tuple[Mapping[str, Any], ...]

    def __init__(
        self,
        *,
        action: str,
        execution_plan_sha256: str,
        wave_index: int,
        durable_output_root: str,
        request_sha256: str,
        result_uri: str,
        result_file_sha256: str,
        result_record_sha256: str,
        result_record: Mapping[str, Any],
        attestation_uri: str,
        attestation_file_sha256: str,
        attestation_record_sha256: str,
        coordinator_run_id: str,
        coordinator_run_record_sha256: str,
        controller_authorization_record_sha256: str,
        runs_get_receipt_record_sha256: str,
        phase_terminal_record_sha256: str,
        evidence_bindings: Sequence[Mapping[str, Any]],
        _issuer: object,
    ) -> None:
        if _issuer is not _REMOTE_AUTHORIZATION_ISSUER:
            raise TypeError(
                "remote tree authority requires direct coordinator collection"
            )
        if action not in FULL_SCORE_REMOTE_COORDINATOR_ACTIONS:
            raise ValueError("remote tree authorization action drift")
        if type(wave_index) is not int or wave_index < 0:
            raise ValueError("remote tree authorization wave_index is invalid")
        _volume_directory_uri(durable_output_root, "durable_output_root")
        _volume_file_uri(result_uri, "result_uri")
        _volume_file_uri(attestation_uri, "attestation_uri")
        for field_name, value in (
            ("execution_plan_sha256", execution_plan_sha256),
            ("request_sha256", request_sha256),
            ("result_file_sha256", result_file_sha256),
            ("result_record_sha256", result_record_sha256),
            ("attestation_file_sha256", attestation_file_sha256),
            ("attestation_record_sha256", attestation_record_sha256),
            ("coordinator_run_record_sha256", coordinator_run_record_sha256),
            (
                "controller_authorization_record_sha256",
                controller_authorization_record_sha256,
            ),
            ("runs_get_receipt_record_sha256", runs_get_receipt_record_sha256),
            ("phase_terminal_record_sha256", phase_terminal_record_sha256),
        ):
            _require_sha256(value, field_name)
        if (
            not coordinator_run_id.isascii()
            or not coordinator_run_id.isdigit()
            or coordinator_run_id.startswith("0")
        ):
            raise ValueError("coordinator_run_id must be canonical decimal digits")
        canonical_result = _json_mapping(result_record, "result_record")
        if canonical_result.get("closed_record_sha256") != result_record_sha256:
            raise ValueError("remote authorization result-record binding drift")
        canonical_evidence_bindings = _validate_authorization_evidence_bindings(
            evidence_bindings,
            action=action,
            durable_output_root=durable_output_root,
            wave_index=wave_index,
            result_record=canonical_result,
        )
        for name, attribute_value in (
            ("action", action),
            ("execution_plan_sha256", execution_plan_sha256),
            ("wave_index", wave_index),
            ("durable_output_root", durable_output_root),
            ("request_sha256", request_sha256),
            ("result_uri", result_uri),
            ("result_file_sha256", result_file_sha256),
            ("result_record_sha256", result_record_sha256),
            ("attestation_uri", attestation_uri),
            ("attestation_file_sha256", attestation_file_sha256),
            ("attestation_record_sha256", attestation_record_sha256),
            ("coordinator_run_id", coordinator_run_id),
            ("coordinator_run_record_sha256", coordinator_run_record_sha256),
            (
                "controller_authorization_record_sha256",
                controller_authorization_record_sha256,
            ),
            ("runs_get_receipt_record_sha256", runs_get_receipt_record_sha256),
            ("phase_terminal_record_sha256", phase_terminal_record_sha256),
        ):
            object.__setattr__(self, name, attribute_value)
        object.__setattr__(self, "result_record", canonical_result)
        object.__setattr__(self, "evidence_bindings", canonical_evidence_bindings)


def _require_real_directory(path: Path, label: str) -> None:
    full_score._require_no_symlink_ancestors(
        path,
        label=label,
        include_leaf=True,
    )
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing") from exc
    if not stat.S_ISDIR(mode):
        raise ValueError(f"{label} must be a real directory")


def _require_regular_entry(path: Path, label: str) -> None:
    full_score._require_no_symlink_ancestors(
        path,
        label=label,
        include_leaf=True,
    )
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular file")


def _require_regular_child_entry(path: Path, label: str) -> os.stat_result:
    """Validate one child after its complete parent chain was checked once."""

    try:
        metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return metadata


def _read_regular_file_no_follow(
    path: Path,
    label: str,
    *,
    max_bytes: int,
) -> bytes:
    _require_regular_entry(path, label)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ValueError(f"{label} must be a regular file") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= max_bytes:
            raise ValueError(f"{label} violates the compact file cap")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            content = handle.read(max_bytes + 1)
    finally:
        os.close(descriptor)
    if not content or len(content) > max_bytes:
        raise ValueError(f"{label} violates the compact file cap")
    return content


@dataclass(slots=True)
class FullScoreCompactArtifactCAS:
    """Content-addressed bounded mirror for compact controller records only."""

    root: Path
    max_file_bytes: int = FULL_SCORE_REMOTE_COMPACT_RECORD_MAX_BYTES
    max_total_bytes: int = FULL_SCORE_REMOTE_CAS_MAX_BYTES

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().absolute()
        full_score._require_no_symlink_ancestors(
            self.root,
            label="compact CAS root",
            include_leaf=True,
        )
        if (
            type(self.max_file_bytes) is not int
            or self.max_file_bytes <= 0
            or self.max_file_bytes > DATABRICKS_VOLUME_FILE_MAX_DOWNLOAD_BYTES
        ):
            raise ValueError("compact CAS file cap is invalid")
        if (
            type(self.max_total_bytes) is not int
            or self.max_total_bytes <= 0
            or self.max_total_bytes > FULL_SCORE_REMOTE_CAS_MAX_BYTES
        ):
            raise ValueError("compact CAS total cap is invalid")
        try:
            (self.root / "blobs" / "sha256").mkdir(parents=True, exist_ok=True)
            (self.root / "bindings").mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError("compact CAS root must be a real directory tree") from exc
        self._validate_layout(allow_missing_lock=True)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        full_score._require_no_symlink_ancestors(
            self.root,
            label="compact CAS root",
            include_leaf=True,
        )
        lock_path = self.root / ".cas.lock"
        full_score._require_no_symlink_ancestors(
            lock_path,
            label="compact CAS lock",
            include_leaf=True,
        )
        try:
            descriptor = os.open(
                lock_path,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except OSError as exc:
            raise ValueError("compact CAS lock must be a regular file") from exc
        locked = False
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("compact CAS lock must be a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            locked = True
            self._validate_layout()
            yield
            self._validate_layout()
        finally:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _validate_layout(self, *, allow_missing_lock: bool = False) -> int:
        full_score._require_no_symlink_ancestors(
            self.root,
            label="compact CAS root",
            include_leaf=True,
        )
        _require_real_directory(self.root, "compact CAS root")
        blobs = self.root / "blobs"
        sha_root = blobs / "sha256"
        bindings = self.root / "bindings"
        for path, label in (
            (blobs, "compact CAS blobs directory"),
            (sha_root, "compact CAS SHA-256 directory"),
            (bindings, "compact CAS bindings directory"),
        ):
            _require_real_directory(path, label)
        allowed_root = {"blobs", "bindings", ".cas.lock"}
        observed_root = {entry.name for entry in self.root.iterdir()}
        if (
            not observed_root <= allowed_root
            or not {"blobs", "bindings"} <= observed_root
        ):
            raise ValueError("compact CAS root contains unexpected entries")
        if not allow_missing_lock and ".cas.lock" not in observed_root:
            raise ValueError("compact CAS lock is missing")
        if ".cas.lock" in observed_root:
            _require_regular_entry(self.root / ".cas.lock", "compact CAS lock")
        if {entry.name for entry in blobs.iterdir()} != {"sha256"}:
            raise ValueError("compact CAS blobs directory contains unexpected entries")
        total = 0
        for entry in sha_root.iterdir():
            metadata = _require_regular_child_entry(entry, "compact CAS blob")
            _require_sha256(entry.name, "compact CAS blob name")
            size = metadata.st_size
            if size <= 0 or size > self.max_file_bytes:
                raise ValueError("compact CAS blob violates the per-file cap")
            total += size
        binding_entries = list(bindings.iterdir())
        if len(binding_entries) > FULL_SCORE_REMOTE_CAS_MAX_BINDINGS:
            raise ValueError("compact CAS binding count exceeds its publication cap")
        for entry in binding_entries:
            metadata = _require_regular_child_entry(entry, "compact CAS binding")
            if not entry.name.endswith(".json"):
                raise ValueError("compact CAS binding filename is invalid")
            _require_sha256(
                entry.name.removesuffix(".json"), "compact CAS binding name"
            )
            size = metadata.st_size
            if size <= 0 or size > FULL_SCORE_REMOTE_CAS_BINDING_MAX_BYTES:
                raise ValueError("compact CAS binding violates the per-file cap")
        if total > self.max_total_bytes:
            raise ValueError("compact artifact CAS exceeds its total byte cap")
        return total

    def bind_bytes(self, uri: str, content: bytes) -> Path:
        """Persist immutable bytes and one URI binding, or require exact replay."""

        _volume_file_uri(uri, "compact artifact URI")
        if not isinstance(content, bytes) or not content:
            raise ValueError("compact artifact content must be non-empty bytes")
        if len(content) > self.max_file_bytes:
            raise ValueError("compact artifact exceeds the per-file CAS cap")
        digest = sha256(content).hexdigest()
        blob = self.root / "blobs" / "sha256" / digest
        binding: dict[str, Any] = {
            "byte_count": len(content),
            "content_sha256": digest,
            "record_type": "cachet.full_score_compact_cas_binding.v1",
            "uri": uri,
        }
        binding_bytes = _pretty_json_bytes(binding)
        if len(binding_bytes) > FULL_SCORE_REMOTE_CAS_BINDING_MAX_BYTES:
            raise ValueError("compact CAS binding exceeds its metadata cap")
        binding_path = self.root / "bindings" / f"{_canonical_sha256(uri)}.json"
        with self._locked():
            current_total = self._validate_layout()
            if (
                not binding_path.exists()
                and len(list((self.root / "bindings").iterdir()))
                >= FULL_SCORE_REMOTE_CAS_MAX_BINDINGS
            ):
                raise ValueError(
                    "compact CAS binding count exceeds its publication cap"
                )
            if (
                not blob.exists()
                and current_total + len(content) > self.max_total_bytes
            ):
                raise ValueError("compact artifact CAS exceeds its total byte cap")
            _write_or_require_bytes(blob, content, "compact CAS blob")
            if self._validate_layout() > self.max_total_bytes:
                raise ValueError("compact artifact CAS exceeds its total byte cap")
            _write_or_require_bytes(binding_path, binding_bytes, "compact CAS binding")
            self._validate_layout()
            return blob

    def bind_record(self, uri: str, record: Mapping[str, Any]) -> Path:
        return self.bind_bytes(uri, _pretty_json_bytes(record))

    def resolve(self, uri: str) -> Path:
        """Resolve only a previously byte-bound URI to its immutable CAS blob."""

        _volume_file_uri(uri, "compact artifact URI")
        with self._locked():
            binding_path = self.root / "bindings" / f"{_canonical_sha256(uri)}.json"
            binding_content = _read_regular_file_no_follow(
                binding_path,
                "compact CAS binding",
                max_bytes=FULL_SCORE_REMOTE_CAS_BINDING_MAX_BYTES,
            )
            binding = _json_object(binding_content, "compact CAS binding")
            if binding.get("uri") != uri:
                raise ValueError("compact CAS URI binding drift")
            digest = _require_sha256(binding.get("content_sha256"), "content_sha256")
            blob = self.root / "blobs" / "sha256" / digest
            content = _read_regular_file_no_follow(
                blob,
                "compact CAS blob",
                max_bytes=self.max_file_bytes,
            )
            if sha256(content).hexdigest() != digest or len(content) != binding.get(
                "byte_count"
            ):
                raise ValueError("compact CAS blob binding drift")
            return blob


@dataclass(frozen=True, slots=True)
class FullScoreRemoteCompactArtifactIO:
    """Authenticated Files transport joined to one bounded local compact CAS."""

    workspace: DatabricksWorkspaceConfig = field(repr=False)
    cas: FullScoreCompactArtifactCAS

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, DatabricksWorkspaceConfig):
            raise TypeError("compact artifact I/O requires Databricks workspace config")
        if not isinstance(self.cas, FullScoreCompactArtifactCAS):
            raise TypeError("compact artifact I/O requires full-score CAS")

    def resolve(self, uri: str) -> Path:
        return self.cas.resolve(uri)

    def download(self, uri: str) -> Path:
        canonical_uri = _volume_file_uri(uri, "compact artifact download URI")
        content = download_databricks_volume_file_bytes(
            self.workspace,
            canonical_uri,
            max_bytes=self.cas.max_file_bytes,
        )
        return self.cas.bind_bytes(canonical_uri, content)

    def publish(self, uri: str, content: bytes) -> Path:
        canonical_uri = _volume_file_uri(uri, "compact artifact publication URI")
        if not isinstance(content, bytes) or not content:
            raise ValueError("compact artifact publication requires nonempty bytes")
        if len(content) > self.cas.max_file_bytes:
            raise ValueError("compact artifact publication exceeds its file cap")
        receipt = upload_databricks_volume_file_bytes_exclusive(
            self.workspace,
            canonical_uri,
            content,
            max_bytes=self.cas.max_file_bytes,
        )
        expected = {
            "created": receipt.get("created"),
            "dbfs_uri": canonical_uri,
            "file_sha256": sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
        if receipt != expected or type(expected["created"]) is not bool:
            raise ValueError("compact artifact Files publication receipt drift")
        blob = self.cas.bind_bytes(canonical_uri, content)
        if self.cas.resolve(canonical_uri) != blob or blob.read_bytes() != content:
            raise ValueError("compact artifact Files/CAS publication drift")
        return blob


def collect_governed_full_score_remote_phase_attempt(
    workspace: DatabricksWorkspaceConfig,
    *,
    cas: FullScoreCompactArtifactCAS,
    execution_plan: Mapping[str, Any],
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    ledger_path: str | Path,
    submission_authorization: full_score.FullScorePhaseSubmissionAuthorization,
    submit_payload_uri: str,
    control_plane_run_uri: str,
    terminal_record_uri: str,
    opener: Any | None = None,
) -> tuple[dict[str, Any], full_score.FullScorePhaseAuthorization]:
    """Collect a GPU phase on Mac through Files/CAS with a local ledger only."""

    if not isinstance(
        submission_authorization,
        full_score.FullScorePhaseSubmissionAuthorization,
    ):
        raise TypeError("remote phase collection requires submission authority")
    compact_io = FullScoreRemoteCompactArtifactIO(workspace, cas)
    submit_uri = _volume_file_uri(submit_payload_uri, "submit_payload_uri")
    run_uri = _volume_file_uri(control_plane_run_uri, "control_plane_run_uri")
    terminal_uri = _volume_file_uri(terminal_record_uri, "terminal_record_uri")
    if len({submit_uri, run_uri, terminal_uri}) != 3:
        raise ValueError("remote phase compact artifact URIs must be distinct")
    submit_file = compact_io.download(submit_uri)
    submit_record = _json_object(
        submit_file.read_bytes(),
        "remote phase submit payload",
    )
    tasks = submit_record.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("remote phase submit payload has no tasks")
    worker_payload_uris: list[str] = []
    for index, raw_task in enumerate(tasks):
        task = _json_mapping(raw_task, f"remote phase task[{index}]")
        spark_task = _required_mapping(task, "spark_python_task")
        parameters = spark_task.get("parameters")
        if not isinstance(parameters, list) or any(
            not isinstance(item, str) for item in parameters
        ):
            raise ValueError("remote phase task parameters are invalid")
        parameter_bindings = full_score._full_score_task_parameter_bindings(
            cast(list[str], parameters),
            phase=submission_authorization.phase,
        )
        worker_payload_uris.append(
            _volume_file_uri(
                parameter_bindings["worker_payload_uri"],
                f"remote phase worker payload URI[{index}]",
            )
        )
    if len(set(worker_payload_uris)) != len(worker_payload_uris):
        raise ValueError("remote phase submit payload duplicates a worker artifact")
    for worker_payload_uri in worker_payload_uris:
        compact_io.download(worker_payload_uri)
    return full_score.collect_governed_full_score_phase_attempt(
        workspace,
        execution_plan=execution_plan,
        inventory=inventory,
        shard_plan=shard_plan,
        ledger_path=ledger_path,
        submission_authorization=submission_authorization,
        submit_payload_path=submit_uri,
        control_plane_run_path=run_uri,
        terminal_record_path=terminal_uri,
        compact_artifact_resolver=compact_io.resolve,
        compact_artifact_publisher=compact_io.publish,
        opener=opener,
    )


def _download_remote_consumer_evidence_bindings(
    compact_io: FullScoreRemoteCompactArtifactIO,
    authorization: object,
    *,
    execution_plan: Mapping[str, Any],
    evidence_directory_uri: str | None = None,
) -> FullScoreRemoteTreeAuthorization:
    remote_authorization = require_full_score_remote_consumer_evidence_authorization(
        authorization,
        execution_plan=execution_plan,
    )
    matched = 0
    for binding in remote_authorization.evidence_bindings:
        evidence_uri = _volume_file_uri(
            binding.get("evidence_uri"),
            "remote evidence URI",
        )
        deletion_uri = _volume_file_uri(
            binding.get("deletion_uri"),
            "remote deletion URI",
        )
        if evidence_directory_uri is None or (
            evidence_uri.rsplit("/", 1)[0] == evidence_directory_uri.rstrip("/")
        ):
            matched += 1
        compact_io.download(evidence_uri)
        compact_io.download(deletion_uri)
    expected = (
        1
        if evidence_directory_uri is not None
        else len(remote_authorization.evidence_bindings)
    )
    if matched != expected:
        raise ValueError("remote consumer evidence download coverage drift")
    return remote_authorization


def write_governed_full_score_remote_matched_billing_block(
    workspace: DatabricksWorkspaceConfig,
    *,
    cas: FullScoreCompactArtifactCAS,
    path: str,
    execution_plan: Mapping[str, Any],
    inventory: FullScoreInventory,
    shard_plan: Mapping[str, Any],
    evidence_dir: str,
    producer_terminal_uri: str,
    consumer_terminal_uri: str,
    ledger_path: str | Path,
    remote_consumer_authorization: object,
) -> dict[str, Any]:
    """Publish one matched billing record with no mounted controller paths."""

    compact_io = FullScoreRemoteCompactArtifactIO(workspace, cas)
    output_uri = _volume_file_uri(path, "matched billing output URI")
    producer_uri = _volume_file_uri(
        producer_terminal_uri,
        "producer terminal URI",
    )
    consumer_uri = _volume_file_uri(
        consumer_terminal_uri,
        "consumer terminal URI",
    )
    evidence_directory_uri = _volume_directory_uri(
        evidence_dir,
        "evidence directory URI",
    )
    if len({output_uri, producer_uri, consumer_uri}) != 3:
        raise ValueError("matched billing compact artifact URIs must be distinct")
    compact_io.download(producer_uri)
    compact_io.download(consumer_uri)
    _download_remote_consumer_evidence_bindings(
        compact_io,
        remote_consumer_authorization,
        execution_plan=execution_plan,
        evidence_directory_uri=evidence_directory_uri,
    )
    return full_score.write_governed_full_score_matched_billing_block(
        output_uri,
        execution_plan,
        inventory=inventory,
        shard_plan=shard_plan,
        evidence_dir=evidence_directory_uri,
        producer_terminal_path=producer_uri,
        consumer_terminal_path=consumer_uri,
        ledger_path=ledger_path,
        remote_consumer_authorization=remote_consumer_authorization,
        compact_artifact_resolver=compact_io.resolve,
        compact_artifact_publisher=compact_io.publish,
    )


def write_governed_full_score_remote_live_p90_budget_admission(
    workspace: DatabricksWorkspaceConfig,
    *,
    cas: FullScoreCompactArtifactCAS,
    path: str,
    execution_plan: Mapping[str, Any],
    completed_block_paths: Sequence[str],
    remote_consumer_authorizations: Sequence[object],
    **kwargs: Any,
) -> dict[str, Any]:
    """Publish and replay the wave-boundary P90 gate through Files/CAS."""

    compact_io = FullScoreRemoteCompactArtifactIO(workspace, cas)
    output_uri = _volume_file_uri(path, "live P90 output URI")
    block_uris = [
        _volume_file_uri(item, f"completed_block_paths[{index}]")
        for index, item in enumerate(completed_block_paths)
    ]
    if not block_uris or len(set(block_uris)) != len(block_uris):
        raise ValueError("live P90 remote matched-block URI coverage drift")
    if output_uri in set(block_uris):
        raise ValueError("live P90 output collides with a matched block")
    for block_uri in block_uris:
        compact_io.download(block_uri)
    for authorization in remote_consumer_authorizations:
        _download_remote_consumer_evidence_bindings(
            compact_io,
            authorization,
            execution_plan=execution_plan,
        )
    return full_score.write_governed_full_score_live_p90_budget_admission(
        output_uri,
        execution_plan,
        completed_block_paths=block_uris,
        remote_consumer_authorizations=remote_consumer_authorizations,
        compact_artifact_resolver=compact_io.resolve,
        compact_artifact_publisher=compact_io.publish,
        **kwargs,
    )


def _require_worker_package_binding(
    worker_payloads: Sequence[Mapping[str, Any]],
    *,
    package_wheel_uri: str,
    package_wheel_sha256: str,
) -> None:
    if not worker_payloads:
        raise ValueError("remote coordinator requires worker package bindings")
    expected_uri = _volume_file_uri(package_wheel_uri, "package_wheel_uri")
    expected_sha = _require_sha256(package_wheel_sha256, "package_wheel_sha256")
    observed: set[tuple[str, str]] = set()
    for index, raw_payload in enumerate(worker_payloads):
        payload = _json_mapping(raw_payload, f"worker_payloads[{index}]")
        bootstrap = _required_mapping(payload, "bootstrap_artifacts")
        qualification = _required_mapping(payload, "gpu_qualification")
        pins = _required_mapping(qualification, "artifact_pins")
        runtime = _required_mapping(payload, "runtime")
        worker_uri = _volume_file_uri(
            bootstrap.get("package_wheel_uri"),
            f"worker_payloads[{index}].package_wheel_uri",
        )
        worker_sha = _require_sha256(
            bootstrap.get("package_wheel_sha256"),
            f"worker_payloads[{index}].package_wheel_sha256",
        )
        runner_sha = _require_sha256(
            bootstrap.get("runner_sha256"),
            f"worker_payloads[{index}].runner_sha256",
        )
        runtime_lock_sha = _require_sha256(
            bootstrap.get("runtime_lock_sha256"),
            f"worker_payloads[{index}].runtime_lock_sha256",
        )
        patched_wheel_sha = _require_sha256(
            bootstrap.get("patched_vllm_wheel_sha256"),
            f"worker_payloads[{index}].patched_vllm_wheel_sha256",
        )
        locked_runtime_sha = _require_sha256(
            bootstrap.get("locked_runtime_identity_sha256"),
            f"worker_payloads[{index}].locked_runtime_identity_sha256",
        )
        if (
            runner_sha != full_score.FULL_SCORE_RUNNER_SHA256
            or runtime_lock_sha != VLLM_RUNTIME_LOCK_SHA256
            or locked_runtime_sha
            != full_score._locked_runtime_identity_sha256(
                runner_sha256=runner_sha,
                package_wheel_sha256=worker_sha,
                runtime_lock_sha256=runtime_lock_sha,
                patched_vllm_wheel_sha256=patched_wheel_sha,
            )
            or pins.get("package_wheel_sha256") != worker_sha
            or pins.get("runner_sha256") != runner_sha
            or pins.get("runtime_lock_sha256") != runtime_lock_sha
            or pins.get("patched_vllm_wheel_sha256") != patched_wheel_sha
            or runtime.get("runtime_lock_sha256") != runtime_lock_sha
            or runtime.get("patched_vllm_wheel_sha256") != patched_wheel_sha
        ):
            raise ValueError(
                "remote coordinator worker qualification/runtime package binding drift"
            )
        observed.add((worker_uri, worker_sha))
    if observed != {(expected_uri, expected_sha)}:
        raise ValueError("remote coordinator package differs from worker payloads")


def _require_worker_request_scope_binding(
    worker_payloads: Sequence[Mapping[str, Any]],
    *,
    durable_output_root: str,
    inventory_uri: str,
    shard_plan_uri: str,
    execution_plan_uri: str,
) -> None:
    expected = {
        "durable_output_root": _volume_directory_uri(
            durable_output_root,
            "durable_output_root",
        ),
        "execution_plan_uri": _volume_file_uri(
            execution_plan_uri,
            "execution_plan_uri",
        ),
        "inventory_uri": _volume_file_uri(inventory_uri, "inventory_uri"),
        "shard_plan_uri": _volume_file_uri(shard_plan_uri, "shard_plan_uri"),
    }
    observed: set[tuple[str, str, str, str]] = set()
    for index, raw_payload in enumerate(worker_payloads):
        payload = _json_mapping(raw_payload, f"worker_payloads[{index}]")
        observed.add(
            (
                _volume_directory_uri(
                    payload.get("durable_output_root"),
                    f"worker_payloads[{index}].durable_output_root",
                ),
                _volume_file_uri(
                    _required_mapping(payload, "execution_plan").get("uri"),
                    f"worker_payloads[{index}].execution_plan.uri",
                ),
                _volume_file_uri(
                    _required_mapping(payload, "inventory").get("uri"),
                    f"worker_payloads[{index}].inventory.uri",
                ),
                _volume_file_uri(
                    _required_mapping(payload, "shard_plan").get("uri"),
                    f"worker_payloads[{index}].shard_plan.uri",
                ),
            )
        )
    if observed != {
        (
            expected["durable_output_root"],
            expected["execution_plan_uri"],
            expected["inventory_uri"],
            expected["shard_plan_uri"],
        )
    }:
        raise ValueError("remote coordinator scope differs from worker payloads")


def _full_score_remote_control_uris(
    durable_output_root: str,
    *,
    action: str,
    wave_index: int,
) -> dict[str, str]:
    root = _volume_directory_uri(durable_output_root, "durable_output_root").rstrip("/")
    _full_score_remote_coordinator_attempt_id(action, wave_index)
    phase_root = f"{root}/control/full-score-remote/{action}/wave-{wave_index:03d}"
    return {
        "attestation": f"{phase_root}/attestation.json",
        "request": f"{phase_root}/request.json",
        "result": f"{phase_root}/result.json",
        "runner": (
            f"{root}/control/full-score-remote/runtime/runner-"
            f"{FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SHA256}.py"
        ),
    }


def _require_phase_terminal_worker_payload_binding(
    phase_terminal_record: Mapping[str, Any],
    *,
    phase_authorization: full_score.FullScorePhaseAuthorization,
    execution_plan_sha256: str,
    worker_payloads: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    terminal = _json_mapping(phase_terminal_record, "phase terminal record")
    if (
        terminal.get("record_type") != full_score.FULL_SCORE_PHASE_TERMINAL_RECORD_TYPE
        or terminal.get("schema_version")
        != full_score.FULL_SCORE_PHASE_TERMINAL_SCHEMA_VERSION
        or terminal.get("authorization_scope")
        != full_score.FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
        or terminal.get("closed_record_sha256") != _closed_record_sha256(terminal)
        or terminal.get("closed_record_sha256")
        != phase_authorization.terminal_record_sha256
        or terminal.get("execution_plan_sha256") != execution_plan_sha256
        or terminal.get("wave_index") != phase_authorization.wave_index
        or terminal.get("phase") != phase_authorization.phase
    ):
        raise ValueError("remote coordinator phase-terminal authority binding drift")
    raw_billing = terminal.get("task_billing")
    if not isinstance(raw_billing, list) or not raw_billing:
        raise ValueError("remote coordinator phase terminal lacks task billing")
    terminal_bindings: list[tuple[str, str, str]] = []
    for index, raw_item in enumerate(raw_billing):
        item = _json_mapping(raw_item, f"phase_terminal.task_billing[{index}]")
        terminal_bindings.append(
            (
                _volume_file_uri(
                    item.get("worker_payload_uri"),
                    "phase terminal worker_payload_uri",
                ),
                _require_sha256(
                    item.get("worker_payload_file_sha256"),
                    "phase terminal worker_payload_file_sha256",
                ),
                _require_sha256(
                    item.get("worker_payload_record_sha256"),
                    "phase terminal worker_payload_record_sha256",
                ),
            )
        )
    supplied_bindings = [
        (
            binding["uri"],
            binding["file_sha256"],
            binding["record_sha256"],
        )
        for binding in (
            _bound_record(uri, record, f"worker_payloads[{index}]")
            for index, (uri, record) in enumerate(worker_payloads)
        )
    ]
    if sorted(terminal_bindings) != sorted(supplied_bindings) or len(
        terminal_bindings
    ) != len(supplied_bindings):
        raise ValueError(
            "remote coordinator worker payloads differ from phase terminal"
        )


def _remote_coordinator_request_authorization_record(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    canonical = _json_mapping(request, "remote coordinator request")
    validate_full_score_remote_coordinator_request(canonical)
    controller_singleton = _required_mapping(canonical, "controller_singleton")
    coordinator = _required_mapping(canonical, "coordinator")
    package = _required_mapping(canonical, "package")
    terminal = _required_mapping(canonical, "phase_terminal")
    record: dict[str, Any] = {
        "attempt_id": canonical["attempt_id"],
        "authorization_scope": "publication",
        "closed_record_sha256": "",
        "controller_lease_root_sha256": controller_singleton[
            "controller_lease_root_sha256"
        ],
        "coordinator_sha256": _canonical_sha256(coordinator),
        "execution_plan_sha256": canonical["execution_plan_sha256"],
        "package": dict(package),
        "phase_terminal_record_sha256": terminal["record_sha256"],
        "record_type": (
            FULL_SCORE_REMOTE_COORDINATOR_REQUEST_AUTHORIZATION_RECORD_TYPE
        ),
        "request_file_sha256": sha256(_pretty_json_bytes(canonical)).hexdigest(),
        "request_record_sha256": canonical["closed_record_sha256"],
        "runner": dict(_required_mapping(canonical, "runner")),
        "schema_version": FULL_SCORE_REMOTE_COORDINATOR_REQUEST_SCHEMA_VERSION,
        "source_bindings_sha256": _canonical_sha256(canonical["sources"]),
        "wave_index": canonical["wave_index"],
        "worker_payload_bindings_sha256": _canonical_sha256(
            canonical["worker_payloads"]
        ),
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def _validate_remote_coordinator_request_authorization_record(
    record: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> None:
    canonical = _json_mapping(record, "remote coordinator request authorization")
    expected = _remote_coordinator_request_authorization_record(request)
    if canonical != expected:
        raise ValueError("remote coordinator request authorization binding drift")


def _issue_full_score_remote_coordinator_request_authorization(
    request: Mapping[str, Any],
    *,
    controller_lease_root: str | Path,
) -> FullScoreRemoteCoordinatorRequestAuthorization:
    return FullScoreRemoteCoordinatorRequestAuthorization(
        request=request,
        authorization_record=_remote_coordinator_request_authorization_record(request),
        controller_lease_root=controller_lease_root,
        _issuer=_REMOTE_AUTHORIZATION_ISSUER,
    )


def _require_full_score_remote_coordinator_request_authorization(
    value: object,
) -> FullScoreRemoteCoordinatorRequestAuthorization:
    if not isinstance(value, FullScoreRemoteCoordinatorRequestAuthorization):
        raise TypeError(
            "remote coordinator requires builder-issued request authorization"
        )
    request = value.to_record()
    _validate_remote_coordinator_request_authorization_record(
        value.authorization_record(),
        request=request,
    )
    singleton = _required_mapping(request, "controller_singleton")
    if singleton.get("controller_lease_root_sha256") != (
        _controller_lease_root_sha256(value.controller_lease_root)
    ):
        raise ValueError("remote coordinator controller lease authority drift")
    return value


def _normalized_controller_lease_root(value: str | Path) -> Path:
    root = Path(value).expanduser().absolute()
    full_score._require_no_symlink_ancestors(
        root,
        label="remote coordinator controller lease root",
        include_leaf=True,
    )
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ValueError("remote coordinator controller lease root is not a directory")
    return root


def _controller_lease_root_sha256(
    value: str | Path,
    *,
    domain: str = "cachet.full_score_remote_controller_lease_root.v1",
) -> str:
    root = _normalized_controller_lease_root(value)
    if not isinstance(domain, str) or not domain:
        raise ValueError("controller lease digest domain is invalid")
    return _canonical_sha256({"domain": domain, "path": str(root)})


def _full_score_remote_controller_lease_root(
    phase_authorization: full_score.FullScorePhaseAuthorization,
    *,
    action: str,
    wave_index: int,
) -> Path:
    if not isinstance(phase_authorization, full_score.FullScorePhaseAuthorization):
        raise TypeError("remote coordinator requires full-score phase authority")
    expected_phase = "producer" if action == "producer_ready" else "consumer"
    if (
        action not in FULL_SCORE_REMOTE_COORDINATOR_ACTIONS
        or phase_authorization.phase != expected_phase
        or phase_authorization.wave_index != wave_index
    ):
        raise ValueError("remote coordinator phase lease identity drift")
    phase_root = _normalized_controller_lease_root(phase_authorization.phase_lease_root)
    return _normalized_controller_lease_root(
        phase_root / f"remote-{action}-wave-{wave_index:03d}"
    )


def _require_full_score_remote_controller_lease_root(
    authorization: FullScoreRemoteCoordinatorRequestAuthorization,
    value: str | Path,
) -> Path:
    observed = _normalized_controller_lease_root(value)
    expected = authorization.controller_lease_root
    if observed != expected or not _controller_lease_root_sha256(
        observed
    ) == _controller_lease_root_sha256(expected):
        raise ValueError(
            "remote coordinator controller_root differs from singleton authority"
        )
    return observed


def _full_score_remote_coordinator_attempt_id(
    action: str,
    wave_index: int,
) -> str:
    if action not in FULL_SCORE_REMOTE_COORDINATOR_ACTIONS:
        raise ValueError("remote coordinator action is unsupported")
    if type(wave_index) is not int or wave_index < 0:
        raise ValueError("wave_index must be a non-negative integer")
    return f"vllm-0271-publication-v1:full-score-remote:{action}:wave:{wave_index:03d}"


def build_full_score_remote_coordinator_request(
    *,
    action: str,
    wave_index: int,
    inventory_uri: str,
    inventory_record: Mapping[str, Any],
    shard_plan_uri: str,
    shard_plan: Mapping[str, Any],
    execution_plan_uri: str,
    execution_plan: Mapping[str, Any],
    worker_payloads: Sequence[tuple[str, Mapping[str, Any]]],
    durable_output_root: str,
    result_uri: str,
    attestation_uri: str,
    coordinator_config: FullScoreRemoteCoordinatorJobConfig,
    phase_authorization: full_score.FullScorePhaseAuthorization,
    phase_terminal_record: Mapping[str, Any],
) -> FullScoreRemoteCoordinatorRequestAuthorization:
    """Close one remote tree-verification request after a terminal GPU phase."""

    if action not in FULL_SCORE_REMOTE_COORDINATOR_ACTIONS:
        raise ValueError("remote coordinator action is unsupported")
    if type(wave_index) is not int or wave_index < 0:
        raise ValueError("wave_index must be a non-negative integer")
    if not isinstance(phase_authorization, full_score.FullScorePhaseAuthorization):
        raise TypeError("remote coordinator requires full-score phase authority")
    if not isinstance(coordinator_config, FullScoreRemoteCoordinatorJobConfig):
        raise TypeError(
            "coordinator_config must be FullScoreRemoteCoordinatorJobConfig"
        )
    expected_phase = "producer" if action == "producer_ready" else "consumer"
    execution_sha = _require_sha256(
        execution_plan.get("closed_record_sha256"), "execution_plan_sha256"
    )
    if (
        phase_authorization.execution_plan_sha256 != execution_sha
        or phase_authorization.wave_index != wave_index
        or phase_authorization.phase != expected_phase
    ):
        raise ValueError("remote coordinator phase authority binding drift")
    _require_phase_terminal_worker_payload_binding(
        phase_terminal_record,
        phase_authorization=phase_authorization,
        execution_plan_sha256=execution_sha,
        worker_payloads=worker_payloads,
    )
    durable_output_root = _volume_directory_uri(
        durable_output_root,
        "durable_output_root",
    )
    _require_worker_request_scope_binding(
        [record for _uri, record in worker_payloads],
        durable_output_root=durable_output_root,
        inventory_uri=inventory_uri,
        shard_plan_uri=shard_plan_uri,
        execution_plan_uri=execution_plan_uri,
    )
    control_uris = _full_score_remote_control_uris(
        durable_output_root,
        action=action,
        wave_index=wave_index,
    )
    for uri, label in (
        (result_uri, "result_uri"),
        (attestation_uri, "attestation_uri"),
        (coordinator_config.package_wheel_uri, "package_wheel_uri"),
    ):
        _volume_file_uri(uri, label)
    if result_uri != control_uris["result"]:
        raise ValueError("remote coordinator result URI is not canonical")
    if attestation_uri != control_uris["attestation"]:
        raise ValueError("remote coordinator attestation URI is not canonical")
    if coordinator_config.runner_python_file != control_uris["runner"]:
        raise ValueError("remote coordinator runner URI is not canonical")
    if result_uri == attestation_uri:
        raise ValueError("result and attestation URIs must be distinct")
    _require_control_output_uri(durable_output_root, result_uri, "result_uri")
    _require_control_output_uri(
        durable_output_root,
        attestation_uri,
        "attestation_uri",
    )
    package_uri = _volume_file_uri(
        coordinator_config.package_wheel_uri,
        "package_wheel_uri",
    )
    package_sha = _require_sha256(
        coordinator_config.package_wheel_sha256,
        "package_wheel_sha256",
    )
    runner_uri = _volume_file_uri(
        coordinator_config.runner_python_file,
        "runner_python_file",
    )
    _require_worker_package_binding(
        [record for _uri, record in worker_payloads],
        package_wheel_uri=package_uri,
        package_wheel_sha256=package_sha,
    )
    sources = {
        "execution_plan": _bound_record(
            execution_plan_uri, execution_plan, "execution_plan"
        ),
        "inventory": _bound_record(inventory_uri, inventory_record, "inventory"),
        "shard_plan": _bound_record(shard_plan_uri, shard_plan, "shard_plan"),
    }
    bound_payloads = [
        _bound_record(uri, record, f"worker_payloads[{index}]")
        for index, (uri, record) in enumerate(worker_payloads)
    ]
    if not bound_payloads:
        raise ValueError("remote coordinator requires worker payloads")
    bound_payloads.sort(key=lambda binding: cast(str, binding["uri"]))
    expected_shards = sorted(_wave_shard_ids(execution_plan, wave_index))
    controller_lease_root = _full_score_remote_controller_lease_root(
        phase_authorization,
        action=action,
        wave_index=wave_index,
    )
    record: dict[str, Any] = {
        "action": action,
        "attestation_uri": attestation_uri,
        "attempt_id": _full_score_remote_coordinator_attempt_id(
            action,
            wave_index,
        ),
        "closed_record_sha256": "",
        "controller_singleton": {
            "controller_lease_root_sha256": _controller_lease_root_sha256(
                controller_lease_root
            ),
            "phase_lease_root_sha256": _controller_lease_root_sha256(
                phase_authorization.phase_lease_root,
                domain="cachet.full_score_phase_lease_root.v1",
            ),
        },
        "coordinator": coordinator_config.to_record(),
        "durable_output_root": durable_output_root,
        "execution_plan_sha256": execution_sha,
        "package": {
            "runner_sha256": FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SHA256,
            "wheel_sha256": package_sha,
            "wheel_uri": package_uri,
        },
        "phase_terminal": {
            "causal_closure_sha256": phase_authorization.causal_closure_sha256,
            "ledger_prefix": phase_authorization.ledger_prefix.to_record(),
            "phase": phase_authorization.phase,
            "record_sha256": phase_authorization.terminal_record_sha256,
            "wave_index": phase_authorization.wave_index,
        },
        "record_type": FULL_SCORE_REMOTE_COORDINATOR_REQUEST_RECORD_TYPE,
        "result_uri": result_uri,
        "runner": {
            "file_sha256": FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SHA256,
            "uri": runner_uri,
        },
        "schema_version": FULL_SCORE_REMOTE_COORDINATOR_REQUEST_SCHEMA_VERSION,
        "shard_ids": expected_shards,
        "sources": sources,
        "wave_index": wave_index,
        "worker_payloads": bound_payloads,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    validate_full_score_remote_coordinator_request(record)
    return _issue_full_score_remote_coordinator_request_authorization(
        record,
        controller_lease_root=controller_lease_root,
    )


def validate_full_score_remote_coordinator_request(
    record: Mapping[str, Any],
) -> None:
    request = _json_mapping(record, "remote coordinator request")
    if set(request) != {
        "action",
        "attestation_uri",
        "attempt_id",
        "closed_record_sha256",
        "controller_singleton",
        "coordinator",
        "durable_output_root",
        "execution_plan_sha256",
        "package",
        "phase_terminal",
        "record_type",
        "result_uri",
        "runner",
        "schema_version",
        "shard_ids",
        "sources",
        "wave_index",
        "worker_payloads",
    }:
        raise ValueError("remote coordinator request schema drift")
    if (
        request.get("record_type") != FULL_SCORE_REMOTE_COORDINATOR_REQUEST_RECORD_TYPE
        or request.get("schema_version")
        != FULL_SCORE_REMOTE_COORDINATOR_REQUEST_SCHEMA_VERSION
        or request.get("closed_record_sha256") != _closed_record_sha256(request)
    ):
        raise ValueError("remote coordinator request identity/closure drift")
    action = request.get("action")
    if action not in FULL_SCORE_REMOTE_COORDINATOR_ACTIONS:
        raise ValueError("remote coordinator request action drift")
    wave_index = request.get("wave_index")
    if type(wave_index) is not int or wave_index < 0:
        raise ValueError("remote coordinator request wave_index drift")
    if request.get("attempt_id") != _full_score_remote_coordinator_attempt_id(
        cast(str, action),
        wave_index,
    ):
        raise ValueError("remote coordinator request attempt_id drift")
    execution_sha = _require_sha256(
        request.get("execution_plan_sha256"), "execution_plan_sha256"
    )
    durable_root = _volume_directory_uri(
        request.get("durable_output_root"), "durable_output_root"
    )
    result_uri = _volume_file_uri(request.get("result_uri"), "result_uri")
    attestation_uri = _volume_file_uri(
        request.get("attestation_uri"), "attestation_uri"
    )
    canonical_control_uris = _full_score_remote_control_uris(
        durable_root,
        action=cast(str, action),
        wave_index=wave_index,
    )
    if result_uri != canonical_control_uris["result"]:
        raise ValueError("remote coordinator result URI is not canonical")
    if attestation_uri != canonical_control_uris["attestation"]:
        raise ValueError("remote coordinator attestation URI is not canonical")
    if result_uri == attestation_uri:
        raise ValueError("remote coordinator result/attestation URI collision")
    _require_control_output_uri(durable_root, result_uri, "result_uri")
    _require_control_output_uri(
        durable_root,
        attestation_uri,
        "attestation_uri",
    )
    coordinator_record = _required_mapping(request, "coordinator")
    coordinator = _full_score_remote_coordinator_job_config_from_record(
        coordinator_record
    )
    singleton = _required_mapping(request, "controller_singleton")
    if set(singleton) != {
        "controller_lease_root_sha256",
        "phase_lease_root_sha256",
    }:
        raise ValueError("remote coordinator singleton schema drift")
    _require_sha256(
        singleton.get("controller_lease_root_sha256"),
        "controller_lease_root_sha256",
    )
    _require_sha256(
        singleton.get("phase_lease_root_sha256"),
        "phase_lease_root_sha256",
    )
    shard_ids = request.get("shard_ids")
    if (
        not isinstance(shard_ids, list)
        or not shard_ids
        or any(not isinstance(item, str) or not item for item in shard_ids)
        or shard_ids != sorted(shard_ids)
        or len(set(shard_ids)) != len(shard_ids)
        or len(shard_ids) > full_score.FULL_SCORE_DEFAULT_MAX_SHARDS_PER_WAVE
    ):
        raise ValueError("remote coordinator request shard_ids drift")
    package = _required_mapping(request, "package")
    if set(package) != {"runner_sha256", "wheel_sha256", "wheel_uri"}:
        raise ValueError("remote coordinator package binding schema drift")
    if (
        _require_sha256(package.get("runner_sha256"), "runner_sha256")
        != FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SHA256
    ):
        raise ValueError("remote coordinator runner identity drift")
    _require_sha256(package.get("wheel_sha256"), "wheel_sha256")
    _volume_file_uri(package.get("wheel_uri"), "wheel_uri")
    runner = _required_mapping(request, "runner")
    if set(runner) != {"file_sha256", "uri"}:
        raise ValueError("remote coordinator runner binding schema drift")
    runner_sha256 = _require_sha256(
        runner.get("file_sha256"),
        "runner file_sha256",
    )
    if (
        runner_sha256 != FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SHA256
        or package.get("runner_sha256") != runner_sha256
    ):
        raise ValueError("remote coordinator runner identity drift")
    _volume_file_uri(runner.get("uri"), "runner URI")
    if runner.get("uri") != canonical_control_uris["runner"]:
        raise ValueError("remote coordinator runner URI is not canonical")
    if (
        coordinator.runner_python_file != runner.get("uri")
        or coordinator.runner_sha256 != runner_sha256
        or coordinator.package_wheel_uri != package.get("wheel_uri")
        or coordinator.package_wheel_sha256 != package.get("wheel_sha256")
    ):
        raise ValueError("remote coordinator job/package/runner binding drift")
    terminal = _required_mapping(request, "phase_terminal")
    if set(terminal) != {
        "causal_closure_sha256",
        "ledger_prefix",
        "phase",
        "record_sha256",
        "wave_index",
    }:
        raise ValueError("remote coordinator phase-terminal schema drift")
    expected_phase = "producer" if action == "producer_ready" else "consumer"
    if (
        terminal.get("phase") != expected_phase
        or terminal.get("wave_index") != wave_index
    ):
        raise ValueError("remote coordinator phase-terminal ordering drift")
    _require_sha256(terminal.get("causal_closure_sha256"), "causal_closure_sha256")
    _require_sha256(terminal.get("record_sha256"), "terminal_record_sha256")
    if not isinstance(terminal.get("ledger_prefix"), Mapping):
        raise ValueError("remote coordinator ledger prefix must be an object")
    databricks_ledger_prefix_from_record(
        cast(Mapping[str, Any], terminal["ledger_prefix"])
    )
    sources = _required_mapping(request, "sources")
    if set(sources) != {"execution_plan", "inventory", "shard_plan"}:
        raise ValueError("remote coordinator source coverage drift")
    for name, binding in sources.items():
        _validate_bound_record(binding, str(name))
    if (
        _required_mapping(sources, "execution_plan").get("record_sha256")
        != execution_sha
    ):
        raise ValueError("remote coordinator execution-plan binding drift")
    payloads = request.get("worker_payloads")
    if not isinstance(payloads, list) or not payloads:
        raise ValueError("remote coordinator worker payloads must be a non-empty array")
    seen_uris: set[str] = set()
    for index, binding in enumerate(payloads):
        validated = _validate_bound_record(binding, f"worker_payloads[{index}]")
        uri = cast(str, validated["uri"])
        if uri in seen_uris:
            raise ValueError("remote coordinator duplicates a worker-payload URI")
        seen_uris.add(uri)
    if not durable_root.startswith("dbfs:/Volumes/"):
        raise AssertionError("validated durable root is not a Volume URI")


def _full_score_remote_coordinator_cluster_record(
    config: FullScoreRemoteCoordinatorJobConfig,
) -> dict[str, Any]:
    return {
        "aws_attributes": {"availability": "ON_DEMAND", "zone_id": "auto"},
        "custom_tags": {
            **dict(config.custom_tags),
            "ResourceClass": "SingleNode",
            "campaign_closure": FULL_SCORE_REMOTE_CAMPAIGN_CLOSED_RECORD_SHA256,
            "purpose": "cachet-vllm-0271-full-score-remote-verifier",
        },
        "data_security_mode": config.data_security_mode,
        "driver_node_type_id": config.node_type_id,
        "node_type_id": config.node_type_id,
        "num_workers": 0,
        "single_user_name": config.single_user_name,
        "spark_conf": {
            "spark.databricks.cluster.profile": "singleNode",
            "spark.master": "local[*]",
        },
        "spark_version": config.spark_version,
    }


def render_full_score_remote_coordinator_submit_payload(
    config: FullScoreRemoteCoordinatorJobConfig,
    request_uri: str,
    request: FullScoreRemoteCoordinatorRequestAuthorization,
) -> dict[str, Any]:
    """Render one idempotent, no-retry, single-node CPU verifier job."""

    if not isinstance(config, FullScoreRemoteCoordinatorJobConfig):
        raise TypeError("config must be FullScoreRemoteCoordinatorJobConfig")
    request_authorization = (
        _require_full_score_remote_coordinator_request_authorization(request)
    )
    canonical_request = request_authorization.to_record()
    request_path = _volume_file_uri(request_uri, "request_uri")
    canonical_control_uris = _full_score_remote_control_uris(
        cast(str, canonical_request["durable_output_root"]),
        action=cast(str, canonical_request["action"]),
        wave_index=cast(int, canonical_request["wave_index"]),
    )
    if request_path != canonical_control_uris["request"]:
        raise ValueError("remote coordinator request URI is not canonical")
    _require_control_output_uri(
        cast(str, canonical_request["durable_output_root"]),
        request_path,
        "request_uri",
    )
    request_package = _required_mapping(canonical_request, "package")
    request_runner = _required_mapping(canonical_request, "runner")
    authorized_config = _full_score_remote_coordinator_job_config_from_record(
        _required_mapping(canonical_request, "coordinator")
    )
    if config.package_wheel_uri != request_package.get(
        "wheel_uri"
    ) or config.package_wheel_sha256 != request_package.get("wheel_sha256"):
        raise ValueError("remote coordinator config/request package binding drift")
    if config.runner_python_file != request_runner.get(
        "uri"
    ) or config.runner_sha256 != request_runner.get("file_sha256"):
        raise ValueError("remote coordinator config/request runner binding drift")
    if config.to_record() != authorized_config.to_record():
        raise ValueError("remote coordinator config/request job binding drift")
    config = authorized_config
    request_bytes = _pretty_json_bytes(canonical_request)
    parameters = [
        "--runner-sha256",
        config.runner_sha256,
        "--package-wheel-uri",
        config.package_wheel_uri,
        "--package-wheel-sha256",
        config.package_wheel_sha256,
        "--request-json",
        request_path,
        "--expected-request-file-sha256",
        sha256(request_bytes).hexdigest(),
        "--expected-request-record-sha256",
        canonical_request["closed_record_sha256"],
    ]
    _require_bounded_remote_coordinator_parameters(parameters)
    task = {
        "max_retries": 0,
        "new_cluster": _full_score_remote_coordinator_cluster_record(config),
        "spark_python_task": {
            "parameters": parameters,
            "python_file": config.runner_python_file,
        },
        "task_key": (
            f"full_score_remote_{canonical_request['action']}_wave_"
            f"{cast(int, canonical_request['wave_index']):03d}"
        ),
        "timeout_seconds": config.timeout_seconds,
    }
    payload = {
        "run_name": (
            "cachet-vllm-0271-full-score-remote-"
            f"{canonical_request['action']}-wave-"
            f"{cast(int, canonical_request['wave_index']):03d}"
        ),
        "tasks": [task],
        "timeout_seconds": config.timeout_seconds,
    }
    bound_payload = bind_databricks_run_idempotency_token(
        payload,
        attempt_id=cast(str, canonical_request["attempt_id"]),
    )
    _validate_remote_coordinator_submit_payload(bound_payload)
    _validate_remote_coordinator_submit_request_binding(
        bound_payload,
        canonical_request,
    )
    return bound_payload


def submit_full_score_remote_coordinator(
    workspace: DatabricksWorkspaceConfig,
    submit_payload: Mapping[str, Any],
    *,
    request_uri: str,
    request: FullScoreRemoteCoordinatorRequestAuthorization,
    controller_root: str | Path,
) -> dict[str, Any]:
    """Stage and causally submit one CPU verifier from a durable local root.

    The request and payload snapshots, exclusive Files API upload receipt, and
    pre-POST intent are durable before the Jobs API is called.  A replay uses
    the same deterministic idempotency token, so an accepted response that was
    lost can be recovered without creating a second remote run.
    """

    request_authorization = (
        _require_full_score_remote_coordinator_request_authorization(request)
    )
    controller_lease_root = _require_full_score_remote_controller_lease_root(
        request_authorization,
        controller_root,
    )
    canonical_request = request_authorization.to_record()
    canonical_payload = _json_mapping(
        submit_payload, "remote coordinator submit payload"
    )
    validate_full_score_remote_coordinator_request(canonical_request)
    _validate_remote_coordinator_submit_payload(canonical_payload)
    _validate_remote_coordinator_submit_request_binding(
        canonical_payload, canonical_request
    )
    _volume_file_uri(request_uri, "request_uri")
    _require_control_output_uri(
        cast(str, canonical_request["durable_output_root"]),
        request_uri,
        "request_uri",
    )
    canonical_control_uris = _full_score_remote_control_uris(
        cast(str, canonical_request["durable_output_root"]),
        action=cast(str, canonical_request["action"]),
        wave_index=cast(int, canonical_request["wave_index"]),
    )
    if request_uri != canonical_control_uris["request"]:
        raise ValueError("remote coordinator request URI is not canonical")
    _require_submit_payload_request_uri(canonical_payload, request_uri)
    attempt_id = cast(str, canonical_request["attempt_id"])
    token = require_databricks_run_idempotency_token(
        canonical_payload, attempt_id=attempt_id
    )
    root = _prepare_full_score_remote_controller_root(controller_lease_root)
    with _full_score_remote_controller_lock(root):
        paths = _full_score_remote_controller_paths(root)
        request_bytes = _pretty_json_bytes(canonical_request)
        payload_bytes = _pretty_json_bytes(canonical_payload)
        _write_or_require_bytes(
            paths["request_authorization"],
            _pretty_json_bytes(request_authorization.authorization_record()),
            "remote controller request authorization",
        )
        _write_or_require_bytes(
            paths["request"], request_bytes, "remote controller request snapshot"
        )
        _write_or_require_bytes(
            paths["submit_payload"],
            payload_bytes,
            "remote controller submit-payload snapshot",
        )
        runner = _required_mapping(canonical_request, "runner")
        runner_bytes = FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SCRIPT.encode("utf-8")
        if paths["runner_upload_receipt"].exists():
            runner_upload_receipt = _read_closed_controller_record(
                paths["runner_upload_receipt"],
                "runner upload receipt",
            )
            _validate_controller_runner_upload_receipt(
                runner_upload_receipt,
                runner=runner,
            )
        else:
            runner_upload = upload_databricks_volume_file_bytes_exclusive(
                workspace,
                cast(str, runner["uri"]),
                runner_bytes,
                max_bytes=FULL_SCORE_REMOTE_COMPACT_RECORD_MAX_BYTES,
            )
            runner_upload_receipt = _controller_runner_upload_receipt(
                runner_upload,
                runner=runner,
            )
            _write_or_require_bytes(
                paths["runner_upload_receipt"],
                _pretty_json_bytes(runner_upload_receipt),
                "remote controller runner upload receipt",
            )
        if paths["upload_receipt"].exists():
            upload_receipt = _read_closed_controller_record(
                paths["upload_receipt"], "request upload receipt"
            )
            _validate_controller_upload_receipt(
                upload_receipt,
                request_uri=request_uri,
                request_bytes=request_bytes,
                request=canonical_request,
            )
        else:
            upload = upload_databricks_volume_file_bytes_exclusive(
                workspace,
                request_uri,
                request_bytes,
                max_bytes=FULL_SCORE_REMOTE_COMPACT_RECORD_MAX_BYTES,
            )
            upload_receipt = _controller_upload_receipt(
                upload,
                request_uri=request_uri,
                request_bytes=request_bytes,
                request=canonical_request,
            )
            _write_or_require_bytes(
                paths["upload_receipt"],
                _pretty_json_bytes(upload_receipt),
                "remote controller request upload receipt",
            )
        post_intent = _controller_post_intent(
            workspace,
            idempotency_token=token,
            request_uri=request_uri,
            request_bytes=request_bytes,
            request=canonical_request,
            payload_bytes=payload_bytes,
            payload=canonical_payload,
            upload_receipt=upload_receipt,
            runner_upload_receipt=runner_upload_receipt,
            request_authorization=request_authorization.authorization_record(),
        )
        _write_or_require_bytes(
            paths["post_intent"],
            _pretty_json_bytes(post_intent),
            "remote controller post intent",
        )
        return _recover_full_score_remote_coordinator_submission_locked(
            workspace,
            root=root,
            request_authorization=request_authorization,
        )


def recover_full_score_remote_coordinator_submission(
    workspace: DatabricksWorkspaceConfig,
    *,
    controller_root: str | Path,
    request_authorization: FullScoreRemoteCoordinatorRequestAuthorization,
) -> dict[str, Any]:
    """Recover the one idempotent Jobs submission from durable local state."""

    request_authorization = (
        _require_full_score_remote_coordinator_request_authorization(
            request_authorization
        )
    )
    root = _prepare_full_score_remote_controller_root(
        _require_full_score_remote_controller_lease_root(
            request_authorization,
            controller_root,
        )
    )
    with _full_score_remote_controller_lock(root):
        return _recover_full_score_remote_coordinator_submission_locked(
            workspace,
            root=root,
            request_authorization=request_authorization,
        )


def run_full_score_remote_coordinator(
    request_path: str | Path,
    *,
    expected_request_file_sha256: str,
    expected_request_record_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute one exact tree verifier against mounted ``/Volumes`` paths."""

    path = Path(request_path)
    raw_request = path.read_bytes()
    if sha256(raw_request).hexdigest() != _require_sha256(
        expected_request_file_sha256, "expected_request_file_sha256"
    ):
        raise ValueError("remote coordinator request file SHA-256 drift")
    request = _json_object(raw_request, "remote coordinator request")
    if raw_request != _pretty_json_bytes(request):
        raise ValueError("remote coordinator request file is not canonical")
    validate_full_score_remote_coordinator_request(request)
    if request.get("closed_record_sha256") != _require_sha256(
        expected_request_record_sha256,
        "expected_request_record_sha256",
    ):
        raise ValueError("remote coordinator request record SHA-256 drift")
    inventory_record = _read_bound_volume_record(
        _required_mapping(_required_mapping(request, "sources"), "inventory")
    )
    inventory = full_score.full_score_inventory_from_record(inventory_record)
    shard_plan = _read_bound_volume_record(
        _required_mapping(_required_mapping(request, "sources"), "shard_plan")
    )
    execution_plan = _read_bound_volume_record(
        _required_mapping(_required_mapping(request, "sources"), "execution_plan")
    )
    payloads = tuple(
        _read_bound_volume_record(_json_mapping(binding, "worker payload binding"))
        for binding in cast(list[Mapping[str, Any]], request["worker_payloads"])
    )
    package = _required_mapping(request, "package")
    _require_worker_package_binding(
        payloads,
        package_wheel_uri=cast(str, package["wheel_uri"]),
        package_wheel_sha256=cast(str, package["wheel_sha256"]),
    )
    action = cast(str, request["action"])
    wave_index = cast(int, request["wave_index"])
    expected_role = "producer" if action == "producer_ready" else "consumer"
    if any(
        payload.get("wave_index") != wave_index
        or payload.get("role") != expected_role
        or payload.get("durable_output_root") != request["durable_output_root"]
        for payload in payloads
    ):
        raise ValueError("remote coordinator worker scope/root binding drift")
    expected_shards = _wave_shard_ids(execution_plan, wave_index)
    payload_shards = {
        cast(str, shard["shard_id"])
        for payload in payloads
        for shard in cast(list[Mapping[str, Any]], payload.get("shards"))
    }
    if payload_shards != expected_shards:
        raise ValueError("remote coordinator worker shard coverage drift")
    root = _volume_mount_path(
        cast(str, request["durable_output_root"]), "durable_output_root"
    )
    ready_wave = root / "ready" / f"wave-{wave_index:03d}"
    original_cluster_path = full_score._cluster_path
    full_score._cluster_path = _remote_cluster_path
    try:
        if action == "producer_ready":
            tree_entries = _require_exact_child_directories(
                ready_wave,
                expected_shards,
                label="producer ready wave",
            )
            result = (
                full_score.build_governed_full_score_producer_phase_completion_record(
                    execution_plan,
                    inventory=inventory,
                    shard_plan=shard_plan,
                    wave_index=wave_index,
                    producer_payloads=payloads,
                )
            )
            tree_closures = _producer_tree_closures(ready_wave, expected_shards)
        else:
            evidence_dirs = [
                root / "evidence" / f"wave-{wave_index:03d}" / shard_id
                for shard_id in sorted(expected_shards)
            ]
            result = full_score.build_governed_full_score_wave_completion_record(
                execution_plan,
                inventory=inventory,
                shard_plan=shard_plan,
                wave_index=wave_index,
                evidence_dirs=evidence_dirs,
            )
            tree_entries = _require_exact_child_directories(
                ready_wave,
                set(),
                label="deleted ready wave",
                allow_missing=True,
            )
            tree_closures = _consumer_evidence_closures(
                root,
                wave_index=wave_index,
                shard_ids=expected_shards,
            )
    finally:
        full_score._cluster_path = original_cluster_path
    result_bytes = _pretty_json_bytes(result)
    result_path = _volume_mount_path(cast(str, request["result_uri"]), "result_uri")
    _write_or_require_bytes(result_path, result_bytes, "remote coordinator result")
    attestation: dict[str, Any] = {
        "action": action,
        "attempt_id": request["attempt_id"],
        "closed_record_sha256": "",
        "durable_output_root": request["durable_output_root"],
        "execution_plan_sha256": request["execution_plan_sha256"],
        "package": dict(_required_mapping(request, "package")),
        "phase_terminal": dict(_required_mapping(request, "phase_terminal")),
        "record_type": FULL_SCORE_REMOTE_COORDINATOR_ATTESTATION_RECORD_TYPE,
        "request_sha256": request["closed_record_sha256"],
        "result": {
            "file_sha256": sha256(result_bytes).hexdigest(),
            "record_sha256": result["closed_record_sha256"],
            "record_type": result["record_type"],
            "uri": request["result_uri"],
        },
        "runner": dict(_required_mapping(request, "runner")),
        "schema_version": FULL_SCORE_REMOTE_COORDINATOR_ATTESTATION_SCHEMA_VERSION,
        "shard_count": len(expected_shards),
        "shard_ids": sorted(expected_shards),
        "tree_closures": tree_closures,
        "tree_entries": tree_entries,
        "wave_index": wave_index,
    }
    attestation["closed_record_sha256"] = _closed_record_sha256(attestation)
    validate_full_score_remote_coordinator_attestation(
        attestation,
        request=request,
        result=result,
        result_file_sha256=sha256(result_bytes).hexdigest(),
    )
    attestation_path = _volume_mount_path(
        cast(str, request["attestation_uri"]), "attestation_uri"
    )
    _write_or_require_bytes(
        attestation_path,
        _pretty_json_bytes(attestation),
        "remote coordinator attestation",
    )
    return result, attestation


def validate_full_score_remote_coordinator_attestation(
    attestation: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    result_file_sha256: str,
) -> None:
    validate_full_score_remote_coordinator_request(request)
    record = _json_mapping(attestation, "remote coordinator attestation")
    normalized_result = _json_mapping(result, "remote coordinator result")
    if normalized_result.get("closed_record_sha256") != _closed_record_sha256(
        normalized_result
    ):
        raise ValueError("remote coordinator result closure drift")
    if set(record) != {
        "action",
        "attempt_id",
        "closed_record_sha256",
        "durable_output_root",
        "execution_plan_sha256",
        "package",
        "phase_terminal",
        "record_type",
        "request_sha256",
        "result",
        "runner",
        "schema_version",
        "shard_count",
        "shard_ids",
        "tree_closures",
        "tree_entries",
        "wave_index",
    }:
        raise ValueError("remote coordinator attestation schema drift")
    if (
        record.get("record_type")
        != FULL_SCORE_REMOTE_COORDINATOR_ATTESTATION_RECORD_TYPE
        or record.get("schema_version")
        != FULL_SCORE_REMOTE_COORDINATOR_ATTESTATION_SCHEMA_VERSION
        or record.get("closed_record_sha256") != _closed_record_sha256(record)
    ):
        raise ValueError("remote coordinator attestation identity/closure drift")
    for field_name in (
        "action",
        "attempt_id",
        "durable_output_root",
        "execution_plan_sha256",
        "package",
        "phase_terminal",
        "runner",
        "wave_index",
    ):
        if record.get(field_name) != request.get(field_name):
            raise ValueError(f"remote coordinator attestation {field_name} drift")
    if record.get("request_sha256") != request.get("closed_record_sha256"):
        raise ValueError("remote coordinator attestation request binding drift")
    expected_shards = cast(list[str], request["shard_ids"])
    if record.get("shard_ids") != expected_shards or record.get("shard_count") != len(
        expected_shards
    ):
        raise ValueError("remote coordinator attestation shard coverage drift")
    result_binding = _required_mapping(record, "result")
    if result_binding != {
        "file_sha256": _require_sha256(result_file_sha256, "result_file_sha256"),
        "record_sha256": result.get("closed_record_sha256"),
        "record_type": result.get("record_type"),
        "uri": request.get("result_uri"),
    }:
        raise ValueError("remote coordinator attestation result binding drift")
    closures = record.get("tree_closures")
    if not isinstance(closures, list):
        raise ValueError("remote coordinator tree closures must be an array")
    expected_closure_keys = (
        {"file_sha256", "files_sha256", "record_sha256", "shard_id"}
        if request.get("action") == "producer_ready"
        else {
            "deletion_file_sha256",
            "deletion_record_sha256",
            "evidence_file_sha256",
            "evidence_record_sha256",
            "shard_id",
        }
    )
    observed_closure_shards: list[str] = []
    for raw_closure in closures:
        closure = _json_mapping(raw_closure, "remote tree closure")
        if set(closure) != expected_closure_keys:
            raise ValueError("remote coordinator tree-closure schema drift")
        shard_id = closure.get("shard_id")
        if not isinstance(shard_id, str):
            raise ValueError("remote coordinator tree-closure shard ID drift")
        observed_closure_shards.append(shard_id)
        for key, value in closure.items():
            if key != "shard_id":
                _require_sha256(value, f"tree_closures.{key}")
    if sorted(observed_closure_shards) != expected_shards or len(
        set(observed_closure_shards)
    ) != len(expected_shards):
        raise ValueError("remote coordinator tree-closure coverage drift")
    entries = record.get("tree_entries")
    if not isinstance(entries, list):
        raise ValueError("remote coordinator tree entries must be an array")
    if request.get("action") == "producer_ready":
        if entries != expected_shards:
            raise ValueError("remote coordinator producer tree inventory drift")
        if (
            result.get("record_type")
            != full_score.FULL_SCORE_PRODUCER_PHASE_COMPLETION_RECORD_TYPE
            or result.get("wave_index") != request.get("wave_index")
            or result.get("execution_plan_sha256")
            != request.get("execution_plan_sha256")
            or result.get("shard_ids") != expected_shards
            or result.get("consumer_phase_authorized") is not True
        ):
            raise ValueError("remote coordinator producer result identity drift")
    else:
        if entries != []:
            raise ValueError("remote coordinator deletion inventory is not empty")
        if (
            result.get("record_type")
            != full_score.FULL_SCORE_WAVE_COMPLETION_RECORD_TYPE
            or result.get("wave_index") != request.get("wave_index")
            or result.get("execution_plan_sha256")
            != request.get("execution_plan_sha256")
            or result.get("shard_ids") != expected_shards
            or result.get("next_wave_authorized") is not True
        ):
            raise ValueError("remote coordinator consumer result identity drift")


def _consumer_evidence_binding_from_bytes(
    *,
    request: Mapping[str, Any],
    closure: Mapping[str, Any],
    evidence_uri: str,
    evidence_bytes: bytes,
    deletion_uri: str,
    deletion_bytes: bytes,
) -> dict[str, Any]:
    shard_id = closure.get("shard_id")
    if not isinstance(shard_id, str):
        raise ValueError("consumer evidence closure shard ID drift")
    wave_index = cast(int, request["wave_index"])
    evidence = _canonical_compact_record(evidence_bytes, "consumer shard evidence")
    deletion = _canonical_compact_record(
        deletion_bytes,
        "consumer deletion attestation",
    )
    paired_examples = evidence.get("paired_examples")
    if not isinstance(paired_examples, list):
        raise ValueError("consumer shard evidence paired_examples is invalid")
    evidence_admission_bytes = (
        FULL_SCORE_REMOTE_EVIDENCE_SHARD_ENVELOPE_MAX_BYTES
        + len(paired_examples) * FULL_SCORE_REMOTE_EVIDENCE_PAIR_MAX_BYTES
    )
    if len(evidence_bytes) > min(
        FULL_SCORE_REMOTE_COMPACT_RECORD_MAX_BYTES,
        evidence_admission_bytes,
    ):
        raise ValueError("consumer shard evidence exceeds its paired-example bound")
    if len(deletion_bytes) > FULL_SCORE_REMOTE_EVIDENCE_DELETION_MAX_BYTES:
        raise ValueError("consumer deletion attestation exceeds its byte bound")
    if (
        sha256(evidence_bytes).hexdigest() != closure.get("evidence_file_sha256")
        or evidence.get("closed_record_sha256") != closure.get("evidence_record_sha256")
        or sha256(deletion_bytes).hexdigest() != closure.get("deletion_file_sha256")
        or deletion.get("closed_record_sha256") != closure.get("deletion_record_sha256")
    ):
        raise ValueError("consumer evidence artifact closure drift")
    if (
        evidence.get("record_type") != full_score.FULL_SCORE_SHARD_EVIDENCE_RECORD_TYPE
        or evidence.get("schema_version")
        != full_score.FULL_SCORE_SHARD_EVIDENCE_SCHEMA_VERSION
        or evidence.get("authorization_scope")
        != full_score.FULL_SCORE_PUBLICATION_AUTHORIZATION_SCOPE
        or evidence.get("durable_evidence_committed") is not True
        or evidence.get("execution_plan_sha256") != request.get("execution_plan_sha256")
        or evidence.get("wave_index") != wave_index
        or evidence.get("shard_id") != shard_id
    ):
        raise ValueError("consumer shard evidence identity drift")
    full_score._validate_full_score_deletion_attestation(
        deletion,
        evidence_record=evidence,
        execution_plan_sha256=cast(str, request["execution_plan_sha256"]),
        shard_id=shard_id,
        wave_index=wave_index,
    )
    return {
        "deletion_file_sha256": closure["deletion_file_sha256"],
        "deletion_record_sha256": closure["deletion_record_sha256"],
        "deletion_uri": deletion_uri,
        "evidence_file_sha256": closure["evidence_file_sha256"],
        "evidence_record_sha256": closure["evidence_record_sha256"],
        "evidence_uri": evidence_uri,
        "shard_id": shard_id,
    }


def _collect_consumer_evidence_bindings(
    workspace: DatabricksWorkspaceConfig,
    *,
    request: Mapping[str, Any],
    attestation: Mapping[str, Any],
    cas: FullScoreCompactArtifactCAS,
) -> tuple[Mapping[str, Any], ...]:
    if request.get("action") == "producer_ready":
        return ()
    raw_closures = attestation.get("tree_closures")
    if not isinstance(raw_closures, list):
        raise ValueError("consumer evidence closures are missing")
    bindings: list[Mapping[str, Any]] = []
    for raw_closure in sorted(
        raw_closures,
        key=lambda item: cast(str, cast(Mapping[str, Any], item).get("shard_id")),
    ):
        closure = _json_mapping(raw_closure, "consumer evidence closure")
        shard_id = cast(str, closure["shard_id"])
        evidence_uri = _consumer_evidence_artifact_uri(
            cast(str, request["durable_output_root"]),
            wave_index=cast(int, request["wave_index"]),
            shard_id=shard_id,
            filename="evidence.json",
        )
        deletion_uri = _consumer_evidence_artifact_uri(
            cast(str, request["durable_output_root"]),
            wave_index=cast(int, request["wave_index"]),
            shard_id=shard_id,
            filename="deletion-attestation.json",
        )
        evidence_bytes = download_databricks_volume_file_bytes(
            workspace,
            evidence_uri,
            max_bytes=FULL_SCORE_REMOTE_COMPACT_RECORD_MAX_BYTES,
        )
        deletion_bytes = download_databricks_volume_file_bytes(
            workspace,
            deletion_uri,
            max_bytes=FULL_SCORE_REMOTE_EVIDENCE_DELETION_MAX_BYTES,
        )
        binding = _consumer_evidence_binding_from_bytes(
            request=request,
            closure=closure,
            evidence_uri=evidence_uri,
            evidence_bytes=evidence_bytes,
            deletion_uri=deletion_uri,
            deletion_bytes=deletion_bytes,
        )
        cas.bind_bytes(evidence_uri, evidence_bytes)
        cas.bind_bytes(deletion_uri, deletion_bytes)
        bindings.append(binding)
    return _validate_authorization_evidence_bindings(
        bindings,
        action="consumer_evidence",
        durable_output_root=cast(str, request["durable_output_root"]),
        wave_index=cast(int, request["wave_index"]),
        result_record={"shard_ids": list(attestation["shard_ids"])},
    )


def collect_full_score_remote_coordinator(
    workspace: DatabricksWorkspaceConfig,
    *,
    controller_root: str | Path,
    cas: FullScoreCompactArtifactCAS,
    request_authorization: FullScoreRemoteCoordinatorRequestAuthorization,
) -> FullScoreRemoteTreeAuthorization:
    """Collect direct runs/get and compact evidence into durable local state."""

    if not isinstance(cas, FullScoreCompactArtifactCAS):
        raise TypeError("cas must be FullScoreCompactArtifactCAS")
    request_authorization = (
        _require_full_score_remote_coordinator_request_authorization(
            request_authorization
        )
    )
    root = _prepare_full_score_remote_controller_root(
        _require_full_score_remote_controller_lease_root(
            request_authorization,
            controller_root,
        )
    )
    with _full_score_remote_controller_lock(root):
        paths = _full_score_remote_controller_paths(root)
        request_authorization, submit_payload, post_intent, submit_response = (
            _read_full_score_remote_submission_closure(
                workspace,
                root=root,
                request_authorization=request_authorization,
            )
        )
        request = request_authorization.to_record()
        run_id = cast(str, submit_response["run_id"])
        run = _json_mapping(
            get_databricks_run(workspace, run_id),
            "remote coordinator runs/get response",
        )
        canonical_run_id = _validate_successful_remote_coordinator_run(
            run,
            submit_payload=submit_payload,
        )
        if canonical_run_id != run_id:
            raise ValueError("remote coordinator runs/get response identity drift")
        run_receipt = _controller_run_receipt(
            run=run,
            run_id=run_id,
            request=request,
            submit_payload=submit_payload,
            post_intent=post_intent,
            submit_response=submit_response,
        )
        _write_or_require_bytes(
            paths["run_receipt"],
            _pretty_json_bytes(run_receipt),
            "remote controller runs/get receipt",
        )
        result_uri = cast(str, request["result_uri"])
        attestation_uri = cast(str, request["attestation_uri"])
        result_bytes = download_databricks_volume_file_bytes(
            workspace,
            result_uri,
            max_bytes=FULL_SCORE_REMOTE_COORDINATOR_OUTPUT_MAX_BYTES,
        )
        attestation_bytes = download_databricks_volume_file_bytes(
            workspace,
            attestation_uri,
            max_bytes=FULL_SCORE_REMOTE_COORDINATOR_OUTPUT_MAX_BYTES,
        )
        result = _canonical_compact_record(result_bytes, "remote coordinator result")
        attestation = _canonical_compact_record(
            attestation_bytes,
            "remote coordinator attestation",
        )
        validate_full_score_remote_coordinator_attestation(
            attestation,
            request=request,
            result=result,
            result_file_sha256=sha256(result_bytes).hexdigest(),
        )
        ready_parent_uri = (
            f"{request['durable_output_root']}/ready/"
            f"wave-{cast(int, request['wave_index']):03d}"
        )
        metadata = list_databricks_volume_directory(
            workspace,
            ready_parent_uri,
            max_entries=full_score.FULL_SCORE_DEFAULT_MAX_SHARDS_PER_WAVE,
        )
        expected_names = (
            sorted(cast(list[str], attestation["shard_ids"]))
            if request["action"] == "producer_ready"
            else []
        )
        observed_names = sorted(cast(str, entry["name"]) for entry in metadata)
        if observed_names != expected_names or any(
            entry.get("is_directory") is not True for entry in metadata
        ):
            raise ValueError("remote ready-tree metadata corroboration drift")
        evidence_bindings = _collect_consumer_evidence_bindings(
            workspace,
            request=request,
            attestation=attestation,
            cas=cas,
        )
        result_path = cas.bind_bytes(result_uri, result_bytes)
        attestation_path = cas.bind_bytes(attestation_uri, attestation_bytes)
        if (
            cas.resolve(result_uri) != result_path
            or cas.resolve(attestation_uri) != attestation_path
        ):
            raise RuntimeError("remote coordinator compact CAS reread drift")
        authorization_record = _controller_authorization_record(
            request=request,
            result=result,
            result_bytes=result_bytes,
            attestation=attestation,
            attestation_bytes=attestation_bytes,
            evidence_bindings=evidence_bindings,
            run_receipt=run_receipt,
        )
        _write_or_require_bytes(
            paths["authorization"],
            _pretty_json_bytes(authorization_record),
            "remote controller authorization record",
        )
        return _replay_full_score_remote_coordinator_authorization_locked(
            workspace,
            root=root,
            cas=cas,
            request_authorization=request_authorization,
        )


def replay_full_score_remote_coordinator_authorization(
    workspace: DatabricksWorkspaceConfig,
    *,
    controller_root: str | Path,
    cas: FullScoreCompactArtifactCAS,
    request_authorization: FullScoreRemoteCoordinatorRequestAuthorization,
) -> FullScoreRemoteTreeAuthorization:
    """Reissue authority only from the complete durable controller closure."""

    if not isinstance(cas, FullScoreCompactArtifactCAS):
        raise TypeError("cas must be FullScoreCompactArtifactCAS")
    request_authorization = (
        _require_full_score_remote_coordinator_request_authorization(
            request_authorization
        )
    )
    root = _prepare_full_score_remote_controller_root(
        _require_full_score_remote_controller_lease_root(
            request_authorization,
            controller_root,
        )
    )
    with _full_score_remote_controller_lock(root):
        return _replay_full_score_remote_coordinator_authorization_locked(
            workspace,
            root=root,
            cas=cas,
            request_authorization=request_authorization,
        )


def require_full_score_remote_ready_authorization(
    authorization: object,
    *,
    execution_plan_sha256: str,
    wave_index: int,
    durable_output_root: str,
    completion_uri: str,
    completion_record: Mapping[str, Any],
) -> FullScoreRemoteTreeAuthorization:
    """Require the exact producer-ready authority used by a consumer phase."""

    if not isinstance(authorization, FullScoreRemoteTreeAuthorization):
        raise TypeError("consumer launch requires remote ready-tree authority")
    completion_sha = _require_sha256(
        completion_record.get("closed_record_sha256"),
        "completion_record_sha256",
    )
    if (
        authorization.action != "producer_ready"
        or authorization.execution_plan_sha256 != execution_plan_sha256
        or authorization.wave_index != wave_index
        or authorization.durable_output_root != durable_output_root
        or authorization.result_uri != completion_uri
        or authorization.result_record_sha256 != completion_sha
        or dict(authorization.result_record) != dict(completion_record)
    ):
        raise ValueError("remote ready-tree authorization binding drift")
    return authorization


def require_full_score_remote_consumer_evidence_authorizations(
    authorizations: Sequence[object],
    *,
    execution_plan: Mapping[str, Any],
) -> tuple[FullScoreRemoteTreeAuthorization, ...]:
    """Require issuer-only evidence authority for all ten publication waves."""

    execution_sha = _require_sha256(
        execution_plan.get("closed_record_sha256"),
        "execution_plan_sha256",
    )
    if execution_sha != _closed_record_sha256(execution_plan):
        raise ValueError("remote consumer authority execution-plan closure drift")
    waves = execution_plan.get("waves")
    expected_wave_count = (
        full_score.FULL_SCORE_PUBLICATION_SHARD_COUNT
        // full_score.FULL_SCORE_DEFAULT_MAX_SHARDS_PER_WAVE
    )
    if not isinstance(waves, list) or len(waves) != expected_wave_count:
        raise ValueError("remote consumer authority requires exactly ten waves")
    observed: dict[int, FullScoreRemoteTreeAuthorization] = {}
    durable_roots: set[str] = set()
    all_shards: list[str] = []
    for authorization in authorizations:
        authorization = require_full_score_remote_consumer_evidence_authorization(
            authorization,
            execution_plan=execution_plan,
        )
        wave_index = authorization.wave_index
        if wave_index in observed:
            raise ValueError("remote consumer authority phase/wave binding drift")
        binding_shards = [
            cast(str, binding["shard_id"])
            for binding in authorization.evidence_bindings
        ]
        observed[wave_index] = authorization
        durable_roots.add(authorization.durable_output_root)
        all_shards.extend(binding_shards)
    if set(observed) != set(range(expected_wave_count)):
        raise ValueError("remote consumer authority omits an execution wave")
    if len(durable_roots) != 1:
        raise ValueError("remote consumer authorities bind different durable roots")
    if (
        len(all_shards) != full_score.FULL_SCORE_PUBLICATION_SHARD_COUNT
        or len(set(all_shards)) != full_score.FULL_SCORE_PUBLICATION_SHARD_COUNT
    ):
        raise ValueError("remote consumer authority is not exact 160-shard coverage")
    return tuple(observed[index] for index in sorted(observed))


def require_full_score_remote_consumer_evidence_authorization(
    authorization: object,
    *,
    execution_plan: Mapping[str, Any],
    expected_wave_index: int | None = None,
    completion_record: Mapping[str, Any] | None = None,
) -> FullScoreRemoteTreeAuthorization:
    """Require one issuer-only consumer-evidence wave capability."""

    if not isinstance(authorization, FullScoreRemoteTreeAuthorization):
        raise TypeError("publication workflow requires remote consumer authority")
    execution_sha = _require_sha256(
        execution_plan.get("closed_record_sha256"),
        "execution_plan_sha256",
    )
    if execution_sha != _closed_record_sha256(execution_plan):
        raise ValueError("remote consumer authority execution-plan closure drift")
    waves = execution_plan.get("waves")
    wave_index = authorization.wave_index
    if (
        not isinstance(waves, list)
        or not 0 <= wave_index < len(waves)
        or (expected_wave_index is not None and wave_index != expected_wave_index)
        or authorization.action != "consumer_evidence"
        or authorization.execution_plan_sha256 != execution_sha
    ):
        raise ValueError("remote consumer authority phase/wave binding drift")
    raw_wave = _json_mapping(waves[wave_index], "execution wave")
    expected_shards = raw_wave.get("shard_ids")
    if not isinstance(expected_shards, list):
        raise ValueError("execution wave shard_ids must be an array")
    result = authorization.result_record
    binding_shards = [
        cast(str, binding["shard_id"]) for binding in authorization.evidence_bindings
    ]
    if (
        result.get("record_type") != full_score.FULL_SCORE_WAVE_COMPLETION_RECORD_TYPE
        or result.get("closed_record_sha256") != authorization.result_record_sha256
        or result.get("closed_record_sha256") != _closed_record_sha256(result)
        or result.get("wave_index") != wave_index
        or result.get("execution_plan_sha256") != execution_sha
        or result.get("next_wave_authorized") is not True
        or result.get("shard_ids") != expected_shards
        or binding_shards != sorted(cast(list[str], expected_shards))
        or (completion_record is not None and dict(result) != dict(completion_record))
    ):
        raise ValueError("remote consumer authority result/evidence coverage drift")
    return authorization


def build_full_score_remote_final_coverage_record(
    execution_plan: Mapping[str, Any],
    wave_attestations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Close exact ten-wave/160-shard coverage from compact attestations."""

    execution_sha = _require_sha256(
        execution_plan.get("closed_record_sha256"), "execution_plan_sha256"
    )
    if execution_sha != _closed_record_sha256(execution_plan):
        raise ValueError("final coverage rejects execution-plan closure drift")
    waves = execution_plan.get("waves")
    expected_wave_count = (
        full_score.FULL_SCORE_PUBLICATION_SHARD_COUNT
        // full_score.FULL_SCORE_DEFAULT_MAX_SHARDS_PER_WAVE
    )
    if not isinstance(waves, list) or len(waves) != expected_wave_count:
        raise ValueError("final coverage requires exactly ten execution waves")
    expected_by_wave = {}
    for index, raw_wave in enumerate(waves):
        wave = _json_mapping(raw_wave, "execution wave")
        shard_ids = wave.get("shard_ids")
        if (
            not isinstance(shard_ids, list)
            or len(shard_ids) != full_score.FULL_SCORE_DEFAULT_MAX_SHARDS_PER_WAVE
            or any(
                not isinstance(shard_id, str) or not shard_id for shard_id in shard_ids
            )
            or len(set(shard_ids)) != len(shard_ids)
        ):
            raise ValueError(
                "final coverage requires sixteen unique shard ids per wave"
            )
        expected_by_wave[index] = set(cast(list[str], shard_ids))
    observed: dict[int, Mapping[str, Any]] = {}
    for raw in wave_attestations:
        attestation = _json_mapping(raw, "wave attestation")
        if (
            attestation.get("record_type")
            != FULL_SCORE_REMOTE_COORDINATOR_ATTESTATION_RECORD_TYPE
            or attestation.get("closed_record_sha256")
            != _closed_record_sha256(attestation)
            or attestation.get("action") != "consumer_evidence"
            or attestation.get("execution_plan_sha256") != execution_sha
        ):
            raise ValueError("final coverage rejects invalid wave attestation")
        wave_index = attestation.get("wave_index")
        if type(wave_index) is not int or wave_index in observed:
            raise ValueError("final coverage has duplicate/invalid wave index")
        shard_ids = attestation.get("shard_ids")
        if not isinstance(shard_ids, list) or set(shard_ids) != expected_by_wave.get(
            wave_index
        ):
            raise ValueError("final coverage wave shard identity drift")
        observed[wave_index] = attestation
    if set(observed) != set(expected_by_wave):
        raise ValueError("final coverage omits an execution wave")
    all_shards = [
        shard_id
        for wave_index in sorted(observed)
        for shard_id in cast(list[str], observed[wave_index]["shard_ids"])
    ]
    if (
        len(all_shards) != full_score.FULL_SCORE_PUBLICATION_SHARD_COUNT
        or len(set(all_shards)) != full_score.FULL_SCORE_PUBLICATION_SHARD_COUNT
    ):
        raise ValueError("final coverage is not exactly 160 unique shards")
    record: dict[str, Any] = {
        "attestation_sha256": [
            observed[index]["closed_record_sha256"] for index in sorted(observed)
        ],
        "closed_record_sha256": "",
        "execution_plan_sha256": execution_sha,
        "record_type": FULL_SCORE_REMOTE_FINAL_COVERAGE_RECORD_TYPE,
        "schema_version": FULL_SCORE_REMOTE_FINAL_COVERAGE_SCHEMA_VERSION,
        "shard_count": len(all_shards),
        "shard_ids_sha256": _canonical_sha256(sorted(all_shards)),
        "wave_count": len(observed),
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def write_full_score_remote_coordinator_runner(path: str | Path) -> None:
    destination = Path(path)
    _write_or_require_bytes(
        destination,
        FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SCRIPT.encode("utf-8"),
        "remote coordinator runner",
    )


def coordinator_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-coordinator")
    run.add_argument("--request-json", required=True)
    run.add_argument("--expected-request-file-sha256", required=True)
    run.add_argument("--expected-request-record-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        result, attestation = run_full_score_remote_coordinator(
            args.request_json,
            expected_request_file_sha256=args.expected_request_file_sha256,
            expected_request_record_sha256=args.expected_request_record_sha256,
        )
        print(
            json.dumps(
                {
                    "attestation_sha256": attestation["closed_record_sha256"],
                    "ok": True,
                    "result_sha256": result["closed_record_sha256"],
                },
                sort_keys=True,
            )
        )
    except Exception as exc:
        print(
            json.dumps(
                {"error": str(exc), "error_type": type(exc).__name__, "ok": False},
                sort_keys=True,
            )
        )
        return 1
    return 0


def _prepare_full_score_remote_controller_root(value: str | Path) -> Path:
    root = Path(value).expanduser().absolute()
    full_score._require_no_symlink_ancestors(
        root,
        label="remote controller root",
        include_leaf=True,
    )
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    full_score._require_no_symlink_ancestors(
        root,
        label="remote controller root",
        include_leaf=True,
    )
    if root.is_symlink() or not root.is_dir():
        raise ValueError("remote controller root must be a real local directory")
    return root


def _full_score_remote_controller_paths(root: Path) -> dict[str, Path]:
    return {
        "authorization": root / "authorization.json",
        "lock": root / ".controller.lock",
        "post_intent": root / "post-intent.json",
        "request": root / "request.json",
        "request_authorization": root / "request-authorization.json",
        "run_receipt": root / "runs-get-receipt.json",
        "runner_upload_receipt": root / "runner-upload-receipt.json",
        "submit_payload": root / "submit-payload.json",
        "submit_response": root / "submit-response.json",
        "upload_receipt": root / "request-upload-receipt.json",
    }


@contextmanager
def _full_score_remote_controller_lock(root: Path) -> Iterator[None]:
    lock_path = _full_score_remote_controller_paths(root)["lock"]
    full_score._require_no_symlink_ancestors(
        lock_path,
        label="remote controller lock",
        include_leaf=True,
    )
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _controller_upload_receipt(
    upload: Mapping[str, Any],
    *,
    request_uri: str,
    request_bytes: bytes,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_upload = _json_mapping(upload, "request upload result")
    expected_upload = {
        "created": normalized_upload.get("created"),
        "dbfs_uri": request_uri,
        "file_sha256": sha256(request_bytes).hexdigest(),
        "size_bytes": len(request_bytes),
    }
    if (
        normalized_upload != expected_upload
        or type(expected_upload["created"]) is not bool
    ):
        raise ValueError("remote controller request upload result binding drift")
    record: dict[str, Any] = {
        "closed_record_sha256": "",
        "record_type": FULL_SCORE_REMOTE_CONTROLLER_UPLOAD_RECEIPT_RECORD_TYPE,
        "request_file_sha256": expected_upload["file_sha256"],
        "request_record_sha256": request["closed_record_sha256"],
        "request_uri": request_uri,
        "schema_version": FULL_SCORE_REMOTE_CONTROLLER_SCHEMA_VERSION,
        "upload": normalized_upload,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def _controller_runner_upload_receipt(
    upload: Mapping[str, Any],
    *,
    runner: Mapping[str, Any],
) -> dict[str, Any]:
    runner_uri = _volume_file_uri(runner.get("uri"), "runner URI")
    runner_sha256 = _require_sha256(
        runner.get("file_sha256"),
        "runner file_sha256",
    )
    runner_bytes = FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SCRIPT.encode("utf-8")
    if sha256(runner_bytes).hexdigest() != runner_sha256:
        raise RuntimeError("embedded remote coordinator runner identity drift")
    normalized_upload = _json_mapping(upload, "runner upload result")
    expected_upload = {
        "created": normalized_upload.get("created"),
        "dbfs_uri": runner_uri,
        "file_sha256": runner_sha256,
        "size_bytes": len(runner_bytes),
    }
    if (
        normalized_upload != expected_upload
        or type(expected_upload["created"]) is not bool
    ):
        raise ValueError("remote controller runner upload result binding drift")
    record: dict[str, Any] = {
        "closed_record_sha256": "",
        "record_type": FULL_SCORE_REMOTE_CONTROLLER_RUNNER_UPLOAD_RECEIPT_RECORD_TYPE,
        "runner_file_sha256": runner_sha256,
        "runner_uri": runner_uri,
        "schema_version": FULL_SCORE_REMOTE_CONTROLLER_SCHEMA_VERSION,
        "upload": normalized_upload,
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def _validate_controller_runner_upload_receipt(
    record: Mapping[str, Any],
    *,
    runner: Mapping[str, Any],
) -> None:
    if set(record) != {
        "closed_record_sha256",
        "record_type",
        "runner_file_sha256",
        "runner_uri",
        "schema_version",
        "upload",
    } or (
        record.get("record_type")
        != FULL_SCORE_REMOTE_CONTROLLER_RUNNER_UPLOAD_RECEIPT_RECORD_TYPE
        or record.get("schema_version") != FULL_SCORE_REMOTE_CONTROLLER_SCHEMA_VERSION
        or record.get("closed_record_sha256") != _closed_record_sha256(record)
    ):
        raise ValueError("remote controller runner upload receipt schema drift")
    runner_uri = _volume_file_uri(runner.get("uri"), "runner URI")
    runner_sha256 = _require_sha256(
        runner.get("file_sha256"),
        "runner file_sha256",
    )
    runner_bytes = FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SCRIPT.encode("utf-8")
    upload = _required_mapping(record, "upload")
    if (
        record.get("runner_uri") != runner_uri
        or record.get("runner_file_sha256") != runner_sha256
        or set(upload) != {"created", "dbfs_uri", "file_sha256", "size_bytes"}
        or type(upload.get("created")) is not bool
        or upload.get("dbfs_uri") != runner_uri
        or upload.get("file_sha256") != runner_sha256
        or upload.get("size_bytes") != len(runner_bytes)
        or sha256(runner_bytes).hexdigest() != runner_sha256
    ):
        raise ValueError("remote controller runner upload receipt binding drift")


def _validate_controller_upload_receipt(
    record: Mapping[str, Any],
    *,
    request_uri: str,
    request_bytes: bytes,
    request: Mapping[str, Any],
) -> None:
    if set(record) != {
        "closed_record_sha256",
        "record_type",
        "request_file_sha256",
        "request_record_sha256",
        "request_uri",
        "schema_version",
        "upload",
    } or (
        record.get("record_type")
        != FULL_SCORE_REMOTE_CONTROLLER_UPLOAD_RECEIPT_RECORD_TYPE
        or record.get("schema_version") != FULL_SCORE_REMOTE_CONTROLLER_SCHEMA_VERSION
        or record.get("closed_record_sha256") != _closed_record_sha256(record)
    ):
        raise ValueError("remote controller request upload receipt schema drift")
    expected_file_sha = sha256(request_bytes).hexdigest()
    if (
        record.get("request_uri") != request_uri
        or record.get("request_file_sha256") != expected_file_sha
        or record.get("request_record_sha256") != request.get("closed_record_sha256")
    ):
        raise ValueError("remote controller request upload receipt binding drift")
    upload = _required_mapping(record, "upload")
    if set(upload) != {"created", "dbfs_uri", "file_sha256", "size_bytes"} or (
        type(upload.get("created")) is not bool
        or upload.get("dbfs_uri") != request_uri
        or upload.get("file_sha256") != expected_file_sha
        or upload.get("size_bytes") != len(request_bytes)
    ):
        raise ValueError("remote controller request upload receipt result drift")


def _controller_post_intent(
    workspace: DatabricksWorkspaceConfig,
    *,
    idempotency_token: str,
    request_uri: str,
    request_bytes: bytes,
    request: Mapping[str, Any],
    payload_bytes: bytes,
    payload: Mapping[str, Any],
    upload_receipt: Mapping[str, Any],
    runner_upload_receipt: Mapping[str, Any],
    request_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    attempt_id = cast(str, request["attempt_id"])
    record: dict[str, Any] = {
        "attempt_id": attempt_id,
        "closed_record_sha256": "",
        "idempotency_token": idempotency_token,
        "record_type": FULL_SCORE_REMOTE_CONTROLLER_POST_INTENT_RECORD_TYPE,
        "request_file_sha256": sha256(request_bytes).hexdigest(),
        "request_authorization_record_sha256": request_authorization[
            "closed_record_sha256"
        ],
        "request_record_sha256": request["closed_record_sha256"],
        "request_uri": request_uri,
        "runner_upload_receipt_record_sha256": runner_upload_receipt[
            "closed_record_sha256"
        ],
        "schema_version": FULL_SCORE_REMOTE_CONTROLLER_SCHEMA_VERSION,
        "submit_payload_file_sha256": sha256(payload_bytes).hexdigest(),
        "submit_payload_sha256": _canonical_sha256(payload),
        "upload_receipt_record_sha256": upload_receipt["closed_record_sha256"],
        "workspace_host_sha256": sha256(
            workspace.normalized_host.encode("utf-8")
        ).hexdigest(),
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def _validate_controller_post_intent(
    record: Mapping[str, Any],
    *,
    workspace: DatabricksWorkspaceConfig,
    request_uri: str,
    request_bytes: bytes,
    request: Mapping[str, Any],
    payload_bytes: bytes,
    payload: Mapping[str, Any],
    upload_receipt: Mapping[str, Any],
    runner_upload_receipt: Mapping[str, Any],
    request_authorization: Mapping[str, Any],
) -> None:
    if set(record) != {
        "attempt_id",
        "closed_record_sha256",
        "idempotency_token",
        "record_type",
        "request_authorization_record_sha256",
        "request_file_sha256",
        "request_record_sha256",
        "request_uri",
        "runner_upload_receipt_record_sha256",
        "schema_version",
        "submit_payload_file_sha256",
        "submit_payload_sha256",
        "upload_receipt_record_sha256",
        "workspace_host_sha256",
    } or (
        record.get("record_type")
        != FULL_SCORE_REMOTE_CONTROLLER_POST_INTENT_RECORD_TYPE
        or record.get("schema_version") != FULL_SCORE_REMOTE_CONTROLLER_SCHEMA_VERSION
        or record.get("closed_record_sha256") != _closed_record_sha256(record)
    ):
        raise ValueError("remote controller post-intent schema drift")
    attempt_id = record.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("remote controller post-intent attempt_id is invalid")
    if attempt_id != request.get("attempt_id"):
        raise ValueError("remote controller post-intent attempt_id drift")
    token = require_databricks_run_idempotency_token(
        payload,
        attempt_id=attempt_id,
    )
    if (
        record.get("idempotency_token") != token
        or record.get("request_authorization_record_sha256")
        != request_authorization.get("closed_record_sha256")
        or record.get("request_uri") != request_uri
        or record.get("request_file_sha256") != sha256(request_bytes).hexdigest()
        or record.get("request_record_sha256") != request.get("closed_record_sha256")
        or record.get("submit_payload_file_sha256") != sha256(payload_bytes).hexdigest()
        or record.get("submit_payload_sha256") != _canonical_sha256(payload)
        or record.get("upload_receipt_record_sha256")
        != upload_receipt.get("closed_record_sha256")
        or record.get("runner_upload_receipt_record_sha256")
        != runner_upload_receipt.get("closed_record_sha256")
        or record.get("workspace_host_sha256")
        != sha256(workspace.normalized_host.encode("utf-8")).hexdigest()
    ):
        raise ValueError("remote controller post-intent binding drift")


def _read_controller_json(path: Path, label: str) -> dict[str, Any]:
    full_score._require_no_symlink_ancestors(
        path,
        label=label,
        include_leaf=True,
    )
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or is not a regular file")
    if path.stat().st_size > FULL_SCORE_REMOTE_COMPACT_RECORD_MAX_BYTES:
        raise ValueError(f"{label} exceeds the compact controller bound")
    content = path.read_bytes()
    if len(content) > FULL_SCORE_REMOTE_COMPACT_RECORD_MAX_BYTES:
        raise ValueError(f"{label} exceeds the compact controller bound")
    record = _json_object(content, label)
    if content != _pretty_json_bytes(record):
        raise ValueError(f"{label} is not canonical pretty JSON")
    return record


def _read_closed_controller_record(path: Path, label: str) -> dict[str, Any]:
    record = _read_controller_json(path, label)
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError(f"{label} closure drift")
    return record


def _read_full_score_remote_prepost_closure(
    workspace: DatabricksWorkspaceConfig,
    *,
    root: Path,
    request_authorization: FullScoreRemoteCoordinatorRequestAuthorization,
) -> tuple[
    FullScoreRemoteCoordinatorRequestAuthorization,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    paths = _full_score_remote_controller_paths(root)
    request_authorization = (
        _require_full_score_remote_coordinator_request_authorization(
            request_authorization
        )
    )
    request = _read_closed_controller_record(paths["request"], "request snapshot")
    validate_full_score_remote_coordinator_request(request)
    request_authorization_record = _read_closed_controller_record(
        paths["request_authorization"],
        "request authorization",
    )
    if (
        request != request_authorization.to_record()
        or request_authorization_record != request_authorization.authorization_record()
    ):
        raise ValueError("durable request authorization differs from live authority")
    submit_payload = _read_controller_json(
        paths["submit_payload"], "submit-payload snapshot"
    )
    _validate_remote_coordinator_submit_payload(submit_payload)
    _validate_remote_coordinator_submit_request_binding(submit_payload, request)
    request_uri = _submit_payload_request_uri(submit_payload)
    _require_control_output_uri(
        cast(str, request["durable_output_root"]),
        request_uri,
        "request_uri",
    )
    request_bytes = _pretty_json_bytes(request)
    payload_bytes = _pretty_json_bytes(submit_payload)
    upload_receipt = _read_closed_controller_record(
        paths["upload_receipt"], "request upload receipt"
    )
    _validate_controller_upload_receipt(
        upload_receipt,
        request_uri=request_uri,
        request_bytes=request_bytes,
        request=request,
    )
    runner_upload_receipt = _read_closed_controller_record(
        paths["runner_upload_receipt"],
        "runner upload receipt",
    )
    _validate_controller_runner_upload_receipt(
        runner_upload_receipt,
        runner=_required_mapping(request, "runner"),
    )
    post_intent = _read_closed_controller_record(
        paths["post_intent"], "post-intent record"
    )
    _validate_controller_post_intent(
        post_intent,
        workspace=workspace,
        request_uri=request_uri,
        request_bytes=request_bytes,
        request=request,
        payload_bytes=payload_bytes,
        payload=submit_payload,
        upload_receipt=upload_receipt,
        runner_upload_receipt=runner_upload_receipt,
        request_authorization=request_authorization_record,
    )
    return request_authorization, submit_payload, upload_receipt, post_intent


def _controller_submit_response_record(
    response: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    submit_payload: Mapping[str, Any],
    post_intent: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_response = _json_mapping(response, "coordinator submit response")
    run_id = _canonical_databricks_run_id(
        canonical_response.get("run_id"), "coordinator submit response run_id"
    )
    record: dict[str, Any] = {
        "closed_record_sha256": "",
        "idempotency_token": submit_payload["idempotency_token"],
        "post_intent_record_sha256": post_intent["closed_record_sha256"],
        "record_type": FULL_SCORE_REMOTE_CONTROLLER_SUBMIT_RESPONSE_RECORD_TYPE,
        "request_sha256": request["closed_record_sha256"],
        "response": canonical_response,
        "run_id": run_id,
        "schema_version": FULL_SCORE_REMOTE_CONTROLLER_SCHEMA_VERSION,
        "submit_payload_sha256": _canonical_sha256(submit_payload),
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def _validate_controller_submit_response(
    record: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    submit_payload: Mapping[str, Any],
    post_intent: Mapping[str, Any],
) -> None:
    if set(record) != {
        "closed_record_sha256",
        "idempotency_token",
        "post_intent_record_sha256",
        "record_type",
        "request_sha256",
        "response",
        "run_id",
        "schema_version",
        "submit_payload_sha256",
    } or (
        record.get("record_type")
        != FULL_SCORE_REMOTE_CONTROLLER_SUBMIT_RESPONSE_RECORD_TYPE
        or record.get("schema_version") != FULL_SCORE_REMOTE_CONTROLLER_SCHEMA_VERSION
        or record.get("closed_record_sha256") != _closed_record_sha256(record)
    ):
        raise ValueError("remote controller submit-response schema drift")
    response = _required_mapping(record, "response")
    if (
        record.get("run_id")
        != _canonical_databricks_run_id(
            response.get("run_id"), "coordinator submit response run_id"
        )
        or record.get("idempotency_token") != submit_payload.get("idempotency_token")
        or record.get("post_intent_record_sha256")
        != post_intent.get("closed_record_sha256")
        or record.get("request_sha256") != request.get("closed_record_sha256")
        or record.get("submit_payload_sha256") != _canonical_sha256(submit_payload)
    ):
        raise ValueError("remote controller submit-response binding drift")


def _recover_full_score_remote_coordinator_submission_locked(
    workspace: DatabricksWorkspaceConfig,
    *,
    root: Path,
    request_authorization: FullScoreRemoteCoordinatorRequestAuthorization,
) -> dict[str, Any]:
    request_authorization, submit_payload, _upload_receipt, post_intent = (
        _read_full_score_remote_prepost_closure(
            workspace,
            root=root,
            request_authorization=request_authorization,
        )
    )
    request = request_authorization.to_record()
    response_path = _full_score_remote_controller_paths(root)["submit_response"]
    if response_path.exists():
        response_record = _read_closed_controller_record(
            response_path, "submit-response record"
        )
        _validate_controller_submit_response(
            response_record,
            request=request,
            submit_payload=submit_payload,
            post_intent=post_intent,
        )
        return dict(_required_mapping(response_record, "response"))
    response = _json_mapping(
        submit_databricks_run(workspace, dict(submit_payload)),
        "coordinator submit response",
    )
    response_record = _controller_submit_response_record(
        response,
        request=request,
        submit_payload=submit_payload,
        post_intent=post_intent,
    )
    _write_or_require_bytes(
        response_path,
        _pretty_json_bytes(response_record),
        "remote controller submit response",
    )
    persisted = _read_closed_controller_record(response_path, "submit-response record")
    _validate_controller_submit_response(
        persisted,
        request=request,
        submit_payload=submit_payload,
        post_intent=post_intent,
    )
    return dict(_required_mapping(persisted, "response"))


def _read_full_score_remote_submission_closure(
    workspace: DatabricksWorkspaceConfig,
    *,
    root: Path,
    request_authorization: FullScoreRemoteCoordinatorRequestAuthorization,
) -> tuple[
    FullScoreRemoteCoordinatorRequestAuthorization,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    request_authorization, submit_payload, _upload_receipt, post_intent = (
        _read_full_score_remote_prepost_closure(
            workspace,
            root=root,
            request_authorization=request_authorization,
        )
    )
    request = request_authorization.to_record()
    submit_response = _read_closed_controller_record(
        _full_score_remote_controller_paths(root)["submit_response"],
        "submit-response record",
    )
    _validate_controller_submit_response(
        submit_response,
        request=request,
        submit_payload=submit_payload,
        post_intent=post_intent,
    )
    return request_authorization, submit_payload, post_intent, submit_response


def _controller_run_receipt(
    *,
    run: Mapping[str, Any],
    run_id: str,
    request: Mapping[str, Any],
    submit_payload: Mapping[str, Any],
    post_intent: Mapping[str, Any],
    submit_response: Mapping[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "closed_record_sha256": "",
        "post_intent_record_sha256": post_intent["closed_record_sha256"],
        "record_type": FULL_SCORE_REMOTE_CONTROLLER_RUN_RECEIPT_RECORD_TYPE,
        "request_sha256": request["closed_record_sha256"],
        "run": dict(run),
        "run_id": run_id,
        "run_record_sha256": _canonical_sha256(run),
        "schema_version": FULL_SCORE_REMOTE_CONTROLLER_SCHEMA_VERSION,
        "submit_payload_sha256": _canonical_sha256(submit_payload),
        "submit_response_record_sha256": submit_response["closed_record_sha256"],
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def _validate_controller_run_receipt(
    record: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    submit_payload: Mapping[str, Any],
    post_intent: Mapping[str, Any],
    submit_response: Mapping[str, Any],
) -> None:
    if set(record) != {
        "closed_record_sha256",
        "post_intent_record_sha256",
        "record_type",
        "request_sha256",
        "run",
        "run_id",
        "run_record_sha256",
        "schema_version",
        "submit_payload_sha256",
        "submit_response_record_sha256",
    } or (
        record.get("record_type")
        != FULL_SCORE_REMOTE_CONTROLLER_RUN_RECEIPT_RECORD_TYPE
        or record.get("schema_version") != FULL_SCORE_REMOTE_CONTROLLER_SCHEMA_VERSION
        or record.get("closed_record_sha256") != _closed_record_sha256(record)
    ):
        raise ValueError("remote controller runs/get receipt schema drift")
    run = _required_mapping(record, "run")
    run_id = _validate_successful_remote_coordinator_run(
        run,
        submit_payload=submit_payload,
    )
    if (
        record.get("run_id") != run_id
        or run_id != submit_response.get("run_id")
        or record.get("run_record_sha256") != _canonical_sha256(run)
        or record.get("post_intent_record_sha256")
        != post_intent.get("closed_record_sha256")
        or record.get("request_sha256") != request.get("closed_record_sha256")
        or record.get("submit_payload_sha256") != _canonical_sha256(submit_payload)
        or record.get("submit_response_record_sha256")
        != submit_response.get("closed_record_sha256")
    ):
        raise ValueError("remote controller runs/get receipt binding drift")


def _controller_authorization_record(
    *,
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    result_bytes: bytes,
    attestation: Mapping[str, Any],
    attestation_bytes: bytes,
    evidence_bindings: Sequence[Mapping[str, Any]],
    run_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    terminal = _required_mapping(request, "phase_terminal")
    record: dict[str, Any] = {
        "action": request["action"],
        "attestation": {
            "file_sha256": sha256(attestation_bytes).hexdigest(),
            "record_sha256": attestation["closed_record_sha256"],
            "uri": request["attestation_uri"],
        },
        "closed_record_sha256": "",
        "coordinator_run_id": run_receipt["run_id"],
        "coordinator_run_record_sha256": run_receipt["run_record_sha256"],
        "durable_output_root": request["durable_output_root"],
        "execution_plan_sha256": request["execution_plan_sha256"],
        "evidence": [dict(binding) for binding in evidence_bindings],
        "phase_terminal_record_sha256": terminal["record_sha256"],
        "record_type": FULL_SCORE_REMOTE_CONTROLLER_AUTHORIZATION_RECORD_TYPE,
        "request_sha256": request["closed_record_sha256"],
        "result": {
            "file_sha256": sha256(result_bytes).hexdigest(),
            "record": dict(result),
            "record_sha256": result["closed_record_sha256"],
            "uri": request["result_uri"],
        },
        "runs_get_receipt_record_sha256": run_receipt["closed_record_sha256"],
        "schema_version": FULL_SCORE_REMOTE_CONTROLLER_SCHEMA_VERSION,
        "wave_index": request["wave_index"],
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def _validate_controller_authorization_record(
    record: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    run_receipt: Mapping[str, Any],
) -> None:
    if set(record) != {
        "action",
        "attestation",
        "closed_record_sha256",
        "coordinator_run_id",
        "coordinator_run_record_sha256",
        "durable_output_root",
        "execution_plan_sha256",
        "evidence",
        "phase_terminal_record_sha256",
        "record_type",
        "request_sha256",
        "result",
        "runs_get_receipt_record_sha256",
        "schema_version",
        "wave_index",
    } or (
        record.get("record_type")
        != FULL_SCORE_REMOTE_CONTROLLER_AUTHORIZATION_RECORD_TYPE
        or record.get("schema_version") != FULL_SCORE_REMOTE_CONTROLLER_SCHEMA_VERSION
        or record.get("closed_record_sha256") != _closed_record_sha256(record)
    ):
        raise ValueError("remote controller authorization-record schema drift")
    terminal = _required_mapping(request, "phase_terminal")
    for field_name in (
        "action",
        "durable_output_root",
        "execution_plan_sha256",
        "wave_index",
    ):
        if record.get(field_name) != request.get(field_name):
            raise ValueError(
                f"remote controller authorization-record {field_name} drift"
            )
    if (
        record.get("request_sha256") != request.get("closed_record_sha256")
        or record.get("phase_terminal_record_sha256") != terminal.get("record_sha256")
        or record.get("coordinator_run_id") != run_receipt.get("run_id")
        or record.get("coordinator_run_record_sha256")
        != run_receipt.get("run_record_sha256")
        or record.get("runs_get_receipt_record_sha256")
        != run_receipt.get("closed_record_sha256")
    ):
        raise ValueError("remote controller authorization-record causal drift")
    result = _required_mapping(record, "result")
    attestation = _required_mapping(record, "attestation")
    raw_evidence = record.get("evidence")
    if not isinstance(raw_evidence, list):
        raise ValueError("remote controller authorization evidence schema drift")
    if set(result) != {"file_sha256", "record", "record_sha256", "uri"} or set(
        attestation
    ) != {"file_sha256", "record_sha256", "uri"}:
        raise ValueError("remote controller authorization artifact schema drift")
    result_record = _required_mapping(result, "record")
    if (
        result.get("uri") != request.get("result_uri")
        or attestation.get("uri") != request.get("attestation_uri")
        or result.get("record_sha256") != result_record.get("closed_record_sha256")
        or result_record.get("closed_record_sha256")
        != _closed_record_sha256(result_record)
    ):
        raise ValueError("remote controller authorization artifact binding drift")
    for binding, label in (
        (result, "authorization result"),
        (attestation, "authorization attestation"),
    ):
        _require_sha256(binding.get("file_sha256"), f"{label} file_sha256")
        _require_sha256(binding.get("record_sha256"), f"{label} record_sha256")
        _volume_file_uri(binding.get("uri"), f"{label} URI")
    _validate_authorization_evidence_bindings(
        cast(list[Mapping[str, Any]], raw_evidence),
        action=cast(str, request["action"]),
        durable_output_root=cast(str, request["durable_output_root"]),
        wave_index=cast(int, request["wave_index"]),
        result_record=result_record,
    )


def _replay_full_score_remote_coordinator_authorization_locked(
    workspace: DatabricksWorkspaceConfig,
    *,
    root: Path,
    cas: FullScoreCompactArtifactCAS,
    request_authorization: FullScoreRemoteCoordinatorRequestAuthorization,
) -> FullScoreRemoteTreeAuthorization:
    request_authorization, submit_payload, post_intent, submit_response = (
        _read_full_score_remote_submission_closure(
            workspace,
            root=root,
            request_authorization=request_authorization,
        )
    )
    request = request_authorization.to_record()
    paths = _full_score_remote_controller_paths(root)
    run_receipt = _read_closed_controller_record(
        paths["run_receipt"], "runs/get receipt"
    )
    _validate_controller_run_receipt(
        run_receipt,
        request=request,
        submit_payload=submit_payload,
        post_intent=post_intent,
        submit_response=submit_response,
    )
    authorization = _read_closed_controller_record(
        paths["authorization"], "authorization record"
    )
    _validate_controller_authorization_record(
        authorization,
        request=request,
        run_receipt=run_receipt,
    )
    result_binding = _required_mapping(authorization, "result")
    attestation_binding = _required_mapping(authorization, "attestation")
    result_uri = cast(str, result_binding["uri"])
    attestation_uri = cast(str, attestation_binding["uri"])
    result_bytes = cas.resolve(result_uri).read_bytes()
    attestation_bytes = cas.resolve(attestation_uri).read_bytes()
    if sha256(result_bytes).hexdigest() != result_binding.get("file_sha256") or sha256(
        attestation_bytes
    ).hexdigest() != attestation_binding.get("file_sha256"):
        raise ValueError("remote controller authorization CAS file binding drift")
    result = _canonical_compact_record(result_bytes, "replayed coordinator result")
    attestation = _canonical_compact_record(
        attestation_bytes, "replayed coordinator attestation"
    )
    if (
        dict(result) != dict(_required_mapping(result_binding, "record"))
        or result.get("closed_record_sha256") != result_binding.get("record_sha256")
        or attestation.get("closed_record_sha256")
        != attestation_binding.get("record_sha256")
    ):
        raise ValueError("remote controller authorization CAS record binding drift")
    validate_full_score_remote_coordinator_attestation(
        attestation,
        request=request,
        result=result,
        result_file_sha256=sha256(result_bytes).hexdigest(),
    )
    evidence_bindings = _validate_authorization_evidence_bindings(
        cast(list[Mapping[str, Any]], authorization["evidence"]),
        action=cast(str, authorization["action"]),
        durable_output_root=cast(str, authorization["durable_output_root"]),
        wave_index=cast(int, authorization["wave_index"]),
        result_record=result,
    )
    closure_by_shard = {
        cast(str, closure["shard_id"]): closure
        for closure in cast(list[Mapping[str, Any]], attestation["tree_closures"])
    }
    for binding in evidence_bindings:
        evidence_uri = cast(str, binding["evidence_uri"])
        deletion_uri = cast(str, binding["deletion_uri"])
        replayed_binding = _consumer_evidence_binding_from_bytes(
            request=request,
            closure=closure_by_shard[cast(str, binding["shard_id"])],
            evidence_uri=evidence_uri,
            evidence_bytes=cas.resolve(evidence_uri).read_bytes(),
            deletion_uri=deletion_uri,
            deletion_bytes=cas.resolve(deletion_uri).read_bytes(),
        )
        if replayed_binding != binding:
            raise ValueError("remote controller authorization evidence CAS drift")
    return FullScoreRemoteTreeAuthorization(
        action=cast(str, authorization["action"]),
        execution_plan_sha256=cast(str, authorization["execution_plan_sha256"]),
        wave_index=cast(int, authorization["wave_index"]),
        durable_output_root=cast(str, authorization["durable_output_root"]),
        request_sha256=cast(str, authorization["request_sha256"]),
        result_uri=result_uri,
        result_file_sha256=cast(str, result_binding["file_sha256"]),
        result_record_sha256=cast(str, result_binding["record_sha256"]),
        result_record=result,
        attestation_uri=attestation_uri,
        attestation_file_sha256=cast(str, attestation_binding["file_sha256"]),
        attestation_record_sha256=cast(str, attestation_binding["record_sha256"]),
        coordinator_run_id=cast(str, authorization["coordinator_run_id"]),
        coordinator_run_record_sha256=cast(
            str, authorization["coordinator_run_record_sha256"]
        ),
        controller_authorization_record_sha256=cast(
            str, authorization["closed_record_sha256"]
        ),
        runs_get_receipt_record_sha256=cast(str, run_receipt["closed_record_sha256"]),
        phase_terminal_record_sha256=cast(
            str, authorization["phase_terminal_record_sha256"]
        ),
        evidence_bindings=evidence_bindings,
        _issuer=_REMOTE_AUTHORIZATION_ISSUER,
    )


def _read_request_inventory(request: Mapping[str, Any]) -> FullScoreInventory:
    return full_score.full_score_inventory_from_record(
        _read_bound_volume_record(
            _required_mapping(_required_mapping(request, "sources"), "inventory")
        )
    )


def _read_request_shard_plan(request: Mapping[str, Any]) -> dict[str, Any]:
    return _read_bound_volume_record(
        _required_mapping(_required_mapping(request, "sources"), "shard_plan")
    )


def _read_request_execution_plan(request: Mapping[str, Any]) -> dict[str, Any]:
    return _read_bound_volume_record(
        _required_mapping(_required_mapping(request, "sources"), "execution_plan")
    )


def _bound_record(uri: str, record: Mapping[str, Any], label: str) -> dict[str, Any]:
    canonical = _json_mapping(record, label)
    record_sha = _require_sha256(
        canonical.get("closed_record_sha256"), f"{label}.record_sha256"
    )
    if record_sha != _closed_record_sha256(canonical):
        raise ValueError(f"{label} closure drift")
    return {
        "file_sha256": sha256(_pretty_json_bytes(canonical)).hexdigest(),
        "record_sha256": record_sha,
        "uri": _volume_file_uri(uri, f"{label}.uri"),
    }


def _validate_bound_record(value: Any, label: str) -> dict[str, Any]:
    binding = _json_mapping(value, label)
    if set(binding) != {"file_sha256", "record_sha256", "uri"}:
        raise ValueError(f"{label} binding schema drift")
    _require_sha256(binding.get("file_sha256"), f"{label}.file_sha256")
    _require_sha256(binding.get("record_sha256"), f"{label}.record_sha256")
    _volume_file_uri(binding.get("uri"), f"{label}.uri")
    return binding


def _read_bound_volume_record(binding: Mapping[str, Any]) -> dict[str, Any]:
    validated = _validate_bound_record(binding, "bound record")
    path = _volume_mount_path(cast(str, validated["uri"]), "bound record URI")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != validated["file_sha256"]:
        raise ValueError("bound remote record file SHA-256 drift")
    record = _json_object(raw, "bound remote record")
    if raw != _pretty_json_bytes(record):
        raise ValueError("bound remote record is not canonical")
    if record.get("closed_record_sha256") != validated["record_sha256"] or record.get(
        "closed_record_sha256"
    ) != _closed_record_sha256(record):
        raise ValueError("bound remote record closure drift")
    return record


def _producer_tree_closures(
    ready_wave: Path,
    shard_ids: set[str],
) -> list[dict[str, Any]]:
    records = []
    for shard_id in sorted(shard_ids):
        record_path = ready_wave / shard_id / "ready-record.json"
        raw = record_path.read_bytes()
        record = _json_object(raw, "ready-shard record")
        records.append(
            {
                "file_sha256": sha256(raw).hexdigest(),
                "files_sha256": _require_sha256(
                    record.get("files_sha256"), "files_sha256"
                ),
                "record_sha256": _require_sha256(
                    record.get("closed_record_sha256"), "ready_record_sha256"
                ),
                "shard_id": shard_id,
            }
        )
    return records


def _consumer_evidence_closures(
    root: Path,
    *,
    wave_index: int,
    shard_ids: set[str],
) -> list[dict[str, Any]]:
    records = []
    for shard_id in sorted(shard_ids):
        directory = root / "evidence" / f"wave-{wave_index:03d}" / shard_id
        evidence_raw = (directory / "evidence.json").read_bytes()
        deletion_raw = (directory / "deletion-attestation.json").read_bytes()
        evidence = _json_object(evidence_raw, "shard evidence")
        deletion = _json_object(deletion_raw, "deletion attestation")
        records.append(
            {
                "deletion_file_sha256": sha256(deletion_raw).hexdigest(),
                "deletion_record_sha256": _require_sha256(
                    deletion.get("closed_record_sha256"),
                    "deletion_record_sha256",
                ),
                "evidence_file_sha256": sha256(evidence_raw).hexdigest(),
                "evidence_record_sha256": _require_sha256(
                    evidence.get("closed_record_sha256"),
                    "evidence_record_sha256",
                ),
                "shard_id": shard_id,
            }
        )
    return records


def _require_exact_child_directories(
    parent: Path,
    expected_names: set[str],
    *,
    label: str,
    allow_missing: bool = False,
) -> list[str]:
    if not parent.exists():
        if allow_missing and not expected_names:
            return []
        raise ValueError(f"{label} directory is missing")
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"{label} must be a real directory")
    observed: list[str] = []
    for child in parent.iterdir():
        if child.is_symlink() or not child.is_dir():
            raise ValueError(f"{label} contains a non-directory or symlink")
        observed.append(child.name)
    if set(observed) != expected_names or len(observed) != len(expected_names):
        raise ValueError(f"{label} has missing or extra shard trees")
    return sorted(observed)


def _require_bounded_remote_coordinator_parameters(parameters: Any) -> list[str]:
    if not isinstance(parameters, list) or any(
        not isinstance(item, str) for item in parameters
    ):
        raise ValueError("remote coordinator parameters must be an array of strings")
    size_bytes = len(
        json.dumps(
            parameters,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if size_bytes > FULL_SCORE_REMOTE_COORDINATOR_PARAMETERS_MAX_BYTES:
        raise ValueError(
            "remote coordinator parameters exceed the 9500-byte safety bound"
        )
    return cast(list[str], parameters)


def _submit_payload_request_uri(payload: Mapping[str, Any]) -> str:
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise ValueError("remote coordinator requires exactly one task")
    task = _json_mapping(tasks[0], "remote coordinator task")
    spark_task = _required_mapping(task, "spark_python_task")
    parameters = _require_bounded_remote_coordinator_parameters(
        spark_task.get("parameters")
    )
    if len(parameters) != 12 or parameters[6] != "--request-json":
        raise ValueError("remote coordinator request parameter schema drift")
    return _volume_file_uri(parameters[7], "coordinator request URI")


def _require_submit_payload_request_uri(
    payload: Mapping[str, Any], request_uri: str
) -> None:
    if _submit_payload_request_uri(payload) != request_uri:
        raise ValueError("remote coordinator submit request URI binding drift")


def _canonical_databricks_run_id(value: Any, field_name: str) -> str:
    if type(value) is int and value > 0:
        return str(value)
    if (
        isinstance(value, str)
        and value.isascii()
        and value.isdigit()
        and not value.startswith("0")
    ):
        return value
    raise ValueError(f"{field_name} is invalid")


def _validate_remote_coordinator_submit_payload(payload: Mapping[str, Any]) -> None:
    if set(payload) != {"idempotency_token", "run_name", "tasks", "timeout_seconds"}:
        raise ValueError("remote coordinator submit payload schema drift")
    token = payload.get("idempotency_token")
    if (
        not isinstance(token, str)
        or len(token) != 64
        or not token.startswith("cachet-")
        or any(character not in "0123456789abcdef" for character in token[7:])
        or not isinstance(payload.get("run_name"), str)
        or not payload.get("run_name")
        or payload.get("timeout_seconds")
        != FULL_SCORE_REMOTE_COORDINATOR_TIMEOUT_SECONDS
    ):
        raise ValueError("remote coordinator run identity/timeout drift")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise ValueError("remote coordinator requires exactly one task")
    task = _json_mapping(tasks[0], "remote coordinator task")
    if (
        set(task)
        != {
            "max_retries",
            "new_cluster",
            "spark_python_task",
            "task_key",
            "timeout_seconds",
        }
        or task.get("max_retries") != 0
        or task.get("timeout_seconds") != FULL_SCORE_REMOTE_COORDINATOR_TIMEOUT_SECONDS
        or not isinstance(task.get("task_key"), str)
        or not task.get("task_key")
    ):
        raise ValueError("remote coordinator task retry/timeout drift")
    cluster = _required_mapping(task, "new_cluster")
    custom_tags = cluster.get("custom_tags")
    if (
        set(cluster)
        != {
            "aws_attributes",
            "custom_tags",
            "data_security_mode",
            "driver_node_type_id",
            "node_type_id",
            "num_workers",
            "single_user_name",
            "spark_conf",
            "spark_version",
        }
        or cluster.get("aws_attributes")
        != {"availability": "ON_DEMAND", "zone_id": "auto"}
        or not isinstance(custom_tags, Mapping)
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in custom_tags.items()
        )
        or custom_tags.get("ResourceClass") != "SingleNode"
        or custom_tags.get("campaign_closure")
        != FULL_SCORE_REMOTE_CAMPAIGN_CLOSED_RECORD_SHA256
        or custom_tags.get("purpose") != "cachet-vllm-0271-full-score-remote-verifier"
        or cluster.get("node_type_id") != FULL_SCORE_REMOTE_COORDINATOR_NODE_TYPE_ID
        or cluster.get("driver_node_type_id")
        != FULL_SCORE_REMOTE_COORDINATOR_NODE_TYPE_ID
        or cluster.get("num_workers") != 0
        or cluster.get("spark_conf")
        != {
            "spark.databricks.cluster.profile": "singleNode",
            "spark.master": "local[*]",
        }
        or cluster.get("data_security_mode") != "SINGLE_USER"
        or not isinstance(cluster.get("single_user_name"), str)
        or not cluster.get("single_user_name")
        or cluster.get("spark_version") != FULL_SCORE_REMOTE_COORDINATOR_SPARK_VERSION
    ):
        raise ValueError("remote coordinator CPU topology drift")


def _validate_remote_coordinator_submit_request_binding(
    payload: Mapping[str, Any],
    request: Mapping[str, Any],
) -> None:
    task = _json_mapping(cast(list[Any], payload["tasks"])[0], "coordinator task")
    coordinator = _full_score_remote_coordinator_job_config_from_record(
        _required_mapping(request, "coordinator")
    )
    cluster = _required_mapping(task, "new_cluster")
    if dict(cluster) != _full_score_remote_coordinator_cluster_record(coordinator):
        raise ValueError("remote coordinator CPU job config binding drift")
    expected_run_name = (
        "cachet-vllm-0271-full-score-remote-"
        f"{request['action']}-wave-{cast(int, request['wave_index']):03d}"
    )
    expected_task_key = (
        f"full_score_remote_{request['action']}_wave_"
        f"{cast(int, request['wave_index']):03d}"
    )
    if (
        payload.get("run_name") != expected_run_name
        or payload.get("timeout_seconds") != coordinator.timeout_seconds
        or task.get("task_key") != expected_task_key
        or task.get("timeout_seconds") != coordinator.timeout_seconds
    ):
        raise ValueError("remote coordinator run/task job config binding drift")
    spark_task = _required_mapping(task, "spark_python_task")
    if set(spark_task) != {"parameters", "python_file"}:
        raise ValueError("remote coordinator Spark task schema drift")
    runner = _required_mapping(request, "runner")
    if _volume_file_uri(
        spark_task.get("python_file"),
        "coordinator python_file",
    ) != runner.get("uri"):
        raise ValueError("remote coordinator runner URI binding drift")
    parameters = _require_bounded_remote_coordinator_parameters(
        spark_task.get("parameters")
    )
    package = _required_mapping(request, "package")
    if len(parameters) != 12:
        raise ValueError("remote coordinator parameter schema drift")
    expected_prefix = [
        "--runner-sha256",
        package.get("runner_sha256"),
        "--package-wheel-uri",
        package.get("wheel_uri"),
        "--package-wheel-sha256",
        package.get("wheel_sha256"),
        "--request-json",
    ]
    if parameters[:7] != expected_prefix:
        raise ValueError("remote coordinator package/request parameter drift")
    if parameters[1] != runner.get("file_sha256"):
        raise ValueError("remote coordinator runner SHA-256 binding drift")
    request_uri = _volume_file_uri(parameters[7], "coordinator request URI")
    canonical_control_uris = _full_score_remote_control_uris(
        cast(str, request["durable_output_root"]),
        action=cast(str, request["action"]),
        wave_index=cast(int, request["wave_index"]),
    )
    if request_uri != canonical_control_uris["request"]:
        raise ValueError("remote coordinator request URI is not canonical")
    if parameters[8:] != [
        "--expected-request-file-sha256",
        sha256(_pretty_json_bytes(request)).hexdigest(),
        "--expected-request-record-sha256",
        request.get("closed_record_sha256"),
    ]:
        raise ValueError("remote coordinator request file binding drift")
    require_databricks_run_idempotency_token(
        payload,
        attempt_id=cast(str, request["attempt_id"]),
    )


def _validate_successful_remote_coordinator_run(
    run: Mapping[str, Any],
    *,
    submit_payload: Mapping[str, Any],
) -> str:
    state = _required_mapping(run, "state")
    if (
        state.get("life_cycle_state") != "TERMINATED"
        or state.get("result_state") != "SUCCESS"
        or run.get("run_name") != submit_payload.get("run_name")
    ):
        raise ValueError("remote coordinator run is not successful and terminal")
    canonical_run_id = _canonical_databricks_run_id(
        run.get("run_id"), "remote coordinator run_id"
    )
    original_attempt_run_id = _canonical_databricks_run_id(
        run.get("original_attempt_run_id"),
        "remote coordinator original_attempt_run_id",
    )
    if original_attempt_run_id != canonical_run_id:
        raise ValueError("remote coordinator run is not the original attempt")
    if run.get("repair_history") not in (None, []):
        raise ValueError("remote coordinator repaired runs are not admissible")
    tasks = run.get("tasks")
    expected_tasks = submit_payload.get("tasks")
    if (
        not isinstance(tasks, list)
        or not isinstance(expected_tasks, list)
        or len(tasks) != 1
        or len(expected_tasks) != 1
    ):
        raise ValueError("remote coordinator run task coverage drift")
    task = _json_mapping(tasks[0], "remote coordinator run task")
    expected = _json_mapping(expected_tasks[0], "remote coordinator submit task")
    task_state = _required_mapping(task, "state")
    cluster = _required_mapping(expected, "new_cluster")
    attempt_number = task.get("attempt_number")
    if type(attempt_number) is not int or attempt_number != 0:
        raise ValueError("remote coordinator task is not exact attempt zero")
    task_run_id = _canonical_databricks_run_id(
        task.get("run_id"), "remote coordinator task run_id"
    )
    if task_run_id == canonical_run_id:
        raise ValueError("remote coordinator task run_id must differ from its parent")
    cluster_instance = _required_mapping(task, "cluster_instance")
    cluster_id = cluster_instance.get("cluster_id")
    if (
        not isinstance(cluster_id, str)
        or not cluster_id
        or cluster_id.strip() != cluster_id
    ):
        raise ValueError("remote coordinator task cluster_id is invalid")
    direct_cluster = task.get("new_cluster")
    cluster_spec = task.get("cluster_spec")
    if isinstance(direct_cluster, Mapping):
        observed_cluster = direct_cluster
    elif isinstance(cluster_spec, Mapping):
        cluster_spec_new_cluster = cluster_spec.get("new_cluster")
        if not isinstance(cluster_spec_new_cluster, Mapping):
            raise ValueError("remote coordinator run cluster_spec drift")
        observed_cluster = cluster_spec_new_cluster
    else:
        raise ValueError("remote coordinator run cluster topology is missing")
    if (
        task.get("task_key") != expected.get("task_key")
        or task_state.get("life_cycle_state") != "TERMINATED"
        or task_state.get("result_state") != "SUCCESS"
        or observed_cluster.get("node_type_id") != cluster.get("node_type_id")
        or observed_cluster.get("driver_node_type_id")
        != cluster.get("driver_node_type_id")
        or type(observed_cluster.get("num_workers")) is not int
        or observed_cluster.get("num_workers") != cluster.get("num_workers")
        or observed_cluster.get("spark_version") != cluster.get("spark_version")
        or observed_cluster.get("data_security_mode")
        != cluster.get("data_security_mode")
    ):
        raise ValueError("remote coordinator run task status/topology drift")
    return canonical_run_id


def _canonical_compact_record(content: bytes, label: str) -> dict[str, Any]:
    if not content or len(content) > FULL_SCORE_REMOTE_COMPACT_RECORD_MAX_BYTES:
        raise ValueError(f"{label} exceeds the compact record bound")
    record = _json_object(content, label)
    if content != _pretty_json_bytes(record):
        raise ValueError(f"{label} is not canonical pretty JSON")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError(f"{label} closure drift")
    return record


def _wave_shard_ids(
    execution_plan: Mapping[str, Any] | None = None,
    wave_index: int | None = None,
    *,
    request: Mapping[str, Any] | None = None,
) -> set[str]:
    if request is not None:
        execution_plan = _read_request_execution_plan(request)
        wave_index = cast(int, request["wave_index"])
    if execution_plan is None or type(wave_index) is not int:
        raise TypeError("wave shard identity requires execution_plan and wave_index")
    waves = execution_plan.get("waves")
    if not isinstance(waves, list) or not 0 <= wave_index < len(waves):
        raise ValueError("wave index is outside the execution plan")
    wave = _json_mapping(waves[wave_index], "execution wave")
    shard_ids = wave.get("shard_ids")
    if not isinstance(shard_ids, list) or any(
        not isinstance(item, str) or not item for item in shard_ids
    ):
        raise ValueError("execution wave shard IDs are invalid")
    if len(set(shard_ids)) != len(shard_ids):
        raise ValueError("execution wave duplicates a shard ID")
    return set(shard_ids)


def _remote_cluster_path(value: str | Path) -> Path:
    raw = str(value)
    if raw.startswith("dbfs:/Volumes/"):
        return Path("/Volumes") / raw.removeprefix("dbfs:/Volumes/")
    return Path(raw)


def _volume_mount_path(value: Any, field_name: str) -> Path:
    uri = _volume_file_or_directory_uri(value, field_name)
    return Path("/Volumes") / uri.removeprefix("dbfs:/Volumes/")


def _volume_file_uri(value: Any, field_name: str) -> str:
    uri = _volume_file_or_directory_uri(value, field_name)
    path = PurePosixPath(uri.removeprefix("dbfs:"))
    if len(path.parts) < 6:
        raise ValueError(f"{field_name} must identify a file beneath a UC Volume")
    return uri


def _volume_directory_uri(value: Any, field_name: str) -> str:
    return _volume_file_or_directory_uri(value, field_name)


def _consumer_evidence_artifact_uri(
    durable_output_root: str,
    *,
    wave_index: int,
    shard_id: str,
    filename: str,
) -> str:
    root = _volume_directory_uri(durable_output_root, "durable_output_root")
    if (
        type(wave_index) is not int
        or wave_index < 0
        or not isinstance(shard_id, str)
        or not shard_id
        or PurePosixPath(shard_id).name != shard_id
        or shard_id in {".", ".."}
        or filename not in {"evidence.json", "deletion-attestation.json"}
    ):
        raise ValueError("consumer evidence artifact identity is invalid")
    return _volume_file_uri(
        f"{root.rstrip('/')}/evidence/wave-{wave_index:03d}/{shard_id}/{filename}",
        "consumer evidence artifact URI",
    )


def _validate_authorization_evidence_bindings(
    value: Sequence[Mapping[str, Any]],
    *,
    action: str,
    durable_output_root: str,
    wave_index: int,
    result_record: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    bindings = tuple(
        _json_mapping(binding, f"evidence_bindings[{index}]")
        for index, binding in enumerate(value)
    )
    if action == "producer_ready":
        if bindings:
            raise ValueError("producer-ready authority cannot bind consumer evidence")
        return ()
    shard_ids = result_record.get("shard_ids")
    if not isinstance(shard_ids, list) or any(
        not isinstance(shard_id, str) or not shard_id for shard_id in shard_ids
    ):
        raise ValueError("consumer authority result shard coverage is invalid")
    expected_keys = {
        "deletion_file_sha256",
        "deletion_record_sha256",
        "deletion_uri",
        "evidence_file_sha256",
        "evidence_record_sha256",
        "evidence_uri",
        "shard_id",
    }
    observed: set[str] = set()
    for binding in bindings:
        if set(binding) != expected_keys:
            raise ValueError("consumer evidence authorization binding schema drift")
        shard_id = binding.get("shard_id")
        if not isinstance(shard_id, str) or shard_id in observed:
            raise ValueError("consumer evidence authorization shard coverage drift")
        observed.add(shard_id)
        expected_evidence_uri = _consumer_evidence_artifact_uri(
            durable_output_root,
            wave_index=wave_index,
            shard_id=shard_id,
            filename="evidence.json",
        )
        expected_deletion_uri = _consumer_evidence_artifact_uri(
            durable_output_root,
            wave_index=wave_index,
            shard_id=shard_id,
            filename="deletion-attestation.json",
        )
        if (
            binding.get("evidence_uri") != expected_evidence_uri
            or binding.get("deletion_uri") != expected_deletion_uri
        ):
            raise ValueError("consumer evidence authorization URI drift")
        for field_name in (
            "deletion_file_sha256",
            "deletion_record_sha256",
            "evidence_file_sha256",
            "evidence_record_sha256",
        ):
            _require_sha256(binding.get(field_name), field_name)
    if observed != set(cast(list[str], shard_ids)) or len(bindings) != len(shard_ids):
        raise ValueError("consumer evidence authorization shard coverage drift")
    return tuple(sorted(bindings, key=lambda item: cast(str, item["shard_id"])))


def _volume_file_or_directory_uri(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    if not value.startswith("dbfs:/Volumes/"):
        raise ValueError(f"{field_name} must be a dbfs:/Volumes URI")
    raw = value.removeprefix("dbfs:")
    path = PurePosixPath(raw)
    if (
        path.as_posix() != raw
        or len(path.parts) < 5
        or path.parts[:2] != ("/", "Volumes")
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or any(character in value for character in ("?", "#", "%", "\\"))
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{field_name} must be a canonical confined UC Volume URI")
    return value


def _require_control_output_uri(
    durable_output_root: str,
    output_uri: str,
    field_name: str,
) -> None:
    root = _volume_directory_uri(durable_output_root, "durable_output_root")
    output = _volume_file_uri(output_uri, field_name)
    if not output.startswith(root.rstrip("/") + "/control/"):
        raise ValueError(
            f"{field_name} must be confined beneath durable_output_root/control"
        )


def _write_or_require_bytes(path: Path, content: bytes, label: str) -> None:
    full_score._require_no_symlink_ancestors(
        path,
        label=f"{label} path",
        include_leaf=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != content:
            raise ValueError(f"{label} existing bytes drift")
        return
    temporary = path.with_name(f".{path.name}.pending-{os.getpid()}-{time.time_ns()}")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if not path.is_file() or path.is_symlink() or path.read_bytes() != content:
                raise ValueError(f"{label} existing bytes drift") from None
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _require_sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _required_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    result = value.get(field_name)
    if not isinstance(result, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return result


def _json_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    normalized = json.loads(json.dumps(dict(value), allow_nan=False, sort_keys=True))
    if not isinstance(normalized, dict):
        raise AssertionError("JSON mapping normalization changed the root type")
    return normalized


def _json_object(content: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _closed_record_sha256(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("closed_record_sha256", None)
    return _canonical_sha256(payload)


def _pretty_json_bytes(record: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(record),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(coordinator_main())


__all__ = [
    "FULL_SCORE_REMOTE_CAMPAIGN_CLOSED_RECORD_SHA256",
    "FULL_SCORE_REMOTE_CAS_MAX_BINDINGS",
    "FULL_SCORE_REMOTE_CAS_MAX_BYTES",
    "FULL_SCORE_REMOTE_CAS_MIRROR_MAX_BYTES",
    "FULL_SCORE_REMOTE_COMPACT_RECORD_MAX_BYTES",
    "FULL_SCORE_REMOTE_COORDINATOR_ACTIONS",
    "FULL_SCORE_REMOTE_COORDINATOR_ATTESTATION_RECORD_TYPE",
    "FULL_SCORE_REMOTE_COORDINATOR_JOB_COUNT",
    "FULL_SCORE_REMOTE_COORDINATOR_NODE_TYPE_ID",
    "FULL_SCORE_REMOTE_COORDINATOR_OUTPUT_MAX_BYTES",
    "FULL_SCORE_REMOTE_COORDINATOR_PARAMETERS_MAX_BYTES",
    "FULL_SCORE_REMOTE_COORDINATOR_REQUEST_AUTHORIZATION_RECORD_TYPE",
    "FULL_SCORE_REMOTE_COORDINATOR_REQUEST_RECORD_TYPE",
    "FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SCRIPT",
    "FULL_SCORE_REMOTE_COORDINATOR_RUNNER_SHA256",
    "FULL_SCORE_REMOTE_COORDINATOR_SPARK_VERSION",
    "FULL_SCORE_REMOTE_COORDINATOR_TIMEOUT_SECONDS",
    "FULL_SCORE_REMOTE_EVIDENCE_DELETION_MAX_BYTES",
    "FULL_SCORE_REMOTE_EVIDENCE_PAIR_MAX_BYTES",
    "FULL_SCORE_REMOTE_FINAL_COVERAGE_RECORD_TYPE",
    "FullScoreCompactArtifactCAS",
    "FullScoreRemoteCompactArtifactIO",
    "FullScoreRemoteCoordinatorJobConfig",
    "FullScoreRemoteCoordinatorRequestAuthorization",
    "FullScoreRemoteTreeAuthorization",
    "build_full_score_remote_coordinator_request",
    "build_full_score_remote_final_coverage_record",
    "collect_governed_full_score_remote_phase_attempt",
    "collect_full_score_remote_coordinator",
    "coordinator_main",
    "render_full_score_remote_coordinator_submit_payload",
    "recover_full_score_remote_coordinator_submission",
    "replay_full_score_remote_coordinator_authorization",
    "require_full_score_remote_consumer_evidence_authorization",
    "require_full_score_remote_consumer_evidence_authorizations",
    "require_full_score_remote_ready_authorization",
    "run_full_score_remote_coordinator",
    "submit_full_score_remote_coordinator",
    "validate_full_score_remote_coordinator_attestation",
    "validate_full_score_remote_coordinator_request",
    "write_governed_full_score_remote_live_p90_budget_admission",
    "write_governed_full_score_remote_matched_billing_block",
    "write_full_score_remote_coordinator_runner",
]
