"""Submit, inspect, and summarize Databricks Jobs runs."""

from __future__ import annotations

import argparse
import base64
import hashlib
from contextlib import contextmanager
from configparser import ConfigParser
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Protocol, cast
import urllib.error
import urllib.parse
import urllib.request

from document_kv_cache._hardware_targets import (
    HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES,
    SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES,
    SUPPORTED_V1_HARDWARE_TARGETS,
    V1_HARDWARE_TARGET_PROFILES,
)
from document_kv_cache.probe_fixtures import DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES
from document_kv_cache.databricks_resource_ledger import (
    DatabricksBatchReservationAuthorization,
    DatabricksClusterHourReservation,
    DatabricksReservationValidator,
    canonical_databricks_submit_payload_snapshot,
    databricks_ledger_path_sha256,
    record_databricks_run_submission_receipt_json,
    read_databricks_cluster_hour_ledger_json,
    require_databricks_batch_reservation_authorization,
    require_databricks_ledger_prefix,
    reserve_databricks_run_attempt_json,
)


__all__ = [
    "DEFAULT_DATABRICKS_HOST_ENV",
    "DEFAULT_DATABRICKS_CONFIG_FILE",
    "DEFAULT_DATABRICKS_TOKEN_ENV",
    "DEFAULT_DATABRICKS_TIMEOUT_SECONDS",
    "DATABRICKS_PROFILE_AUTH_MODES",
    "DATABRICKS_AUTH_CHECK_RECORD_TYPE",
    "DATABRICKS_DBFS_PUT_MAX_CONTENT_BYTES",
    "DATABRICKS_ACTIVE_RUNS_MAX_ENTRIES",
    "DATABRICKS_ACTIVE_RUNS_MAX_PAGES",
    "DATABRICKS_API_PAGE_MAX_BYTES",
    "DATABRICKS_API_PAGE_TOKEN_MAX_BYTES",
    "DATABRICKS_NODE_TYPES_MAX_ENTRIES",
    "DATABRICKS_VOLUME_DIRECTORY_MAX_ENTRIES",
    "DATABRICKS_VOLUME_DIRECTORY_MAX_PAGES",
    "DATABRICKS_VOLUME_DIRECTORY_MAX_PAGE_TOKEN_BYTES",
    "DATABRICKS_VOLUME_FILE_MAX_DOWNLOAD_BYTES",
    "DATABRICKS_VOLUME_FILE_MAX_STREAM_BYTES",
    "DATABRICKS_VOLUME_FILE_MAX_UPLOAD_BYTES",
    "DATABRICKS_RUN_STATUS_RECORD_TYPE",
    "DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE",
    "DatabricksPreReservedPostClaimExistsError",
    "DatabricksWorkspaceConfig",
    "databricks_workspace_config_from_env",
    "databricks_workspace_config_from_profile",
    "databricks_workspace_config_from_sdk_profile",
    "check_databricks_auth",
    "require_databricks_current_user_name",
    "bind_databricks_run_idempotency_token",
    "require_databricks_run_idempotency_token",
    "submit_databricks_run",
    "reserve_and_submit_databricks_run",
    "submit_pre_reserved_databricks_run",
    "recover_pre_reserved_databricks_run",
    "resume_pre_reserved_databricks_run",
    "reserve_and_submit_databricks_run_json",
    "get_databricks_run",
    "get_databricks_run_output",
    "download_databricks_volume_file_bytes",
    "stream_databricks_volume_file_sha256",
    "get_databricks_volume_file_metadata",
    "list_active_databricks_runs",
    "list_databricks_node_types",
    "list_databricks_volume_directory",
    "create_databricks_volume_directory_idempotent",
    "upload_databricks_volume_file_bytes_exclusive",
    "upload_databricks_volume_file_path_exclusive",
    "put_databricks_dbfs_file",
    "plan_databricks_stage_and_submit",
    "stage_and_submit_databricks_run",
    "summarize_databricks_run",
    "summarize_databricks_run_submit_payload",
    "databricks_run_status_record",
    "databricks_run_status_sidecar_issues",
    "validate_databricks_run_status_sidecar",
    "write_databricks_run_response_json",
    "read_databricks_run_submit_payload",
    "main",
]
DEFAULT_DATABRICKS_HOST_ENV = "DATABRICKS_HOST"
DEFAULT_DATABRICKS_TOKEN_ENV = "DATABRICKS_TOKEN"
DEFAULT_DATABRICKS_CONFIG_FILE = "~/.databrickscfg"
DEFAULT_DATABRICKS_TIMEOUT_SECONDS = 60.0
DATABRICKS_PROFILE_AUTH_MODES = ("auto", "static", "sdk")
DATABRICKS_DBFS_PUT_MAX_CONTENT_BYTES = 1_000_000
DATABRICKS_ACTIVE_RUNS_MAX_ENTRIES = 256
DATABRICKS_ACTIVE_RUNS_MAX_PAGES = 64
DATABRICKS_API_PAGE_MAX_BYTES = 16 * 1024 * 1024
DATABRICKS_API_PAGE_TOKEN_MAX_BYTES = 4_096
DATABRICKS_NODE_TYPES_MAX_ENTRIES = 1_024
DATABRICKS_VOLUME_DIRECTORY_MAX_ENTRIES = 4_096
DATABRICKS_VOLUME_DIRECTORY_MAX_PAGES = 64
DATABRICKS_VOLUME_DIRECTORY_MAX_PAGE_TOKEN_BYTES = 4_096
DATABRICKS_VOLUME_FILE_MAX_DOWNLOAD_BYTES = 16 * 1024 * 1024
DATABRICKS_VOLUME_FILE_MAX_STREAM_BYTES = 1024 * 1024 * 1024
DATABRICKS_VOLUME_FILE_MAX_UPLOAD_BYTES = 16 * 1024 * 1024
_DATABRICKS_VOLUME_FILE_STREAM_CHUNK_BYTES = 1024 * 1024
_DATABRICKS_ACTIVE_RUNS_PAGE_SIZE = 20
_DATABRICKS_ERROR_BODY_MAX_BYTES = 64 * 1024
DATABRICKS_AUTH_CHECK_RECORD_TYPE = "document_kv.databricks_auth_check.v1"
DATABRICKS_RUN_STATUS_RECORD_TYPE = "document_kv.databricks_run_status.v1"
DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE = (
    "document_kv.databricks_run_submit_payload.v1"
)
_DATABRICKS_AUTH_CHECK_ENDPOINT = "/api/2.0/preview/scim/v2/Me"
_DATABRICKS_IDEMPOTENCY_TOKEN_DOMAIN = "cachet.databricks_runs.submit_idempotency.v1"
_DATABRICKS_IDEMPOTENCY_TOKEN_RE = re.compile(r"cachet-[0-9a-f]{57}\Z")
_DATABRICKS_GPU_TYPE_FIELD = "aws_single_node_gpu_type"
_LEGACY_DATABRICKS_GPU_TYPE_FIELD = "aws_g5_node_type"
_DATABRICKS_GPU_TYPE_FIELDS = (
    _DATABRICKS_GPU_TYPE_FIELD,
    _LEGACY_DATABRICKS_GPU_TYPE_FIELD,
)
_DATABRICKS_CONFIG_DEFAULT_SECTION = "__cachet_no_inherited_databricks_defaults__"
_DATABRICKS_ALWAYS_STAGED_ARTIFACT_PARAMETER_FLAGS = frozenset(
    {
        "--package-wheel-uri",
        "--plan-json",
        "--sglang-runtime-preflight-launch-config-json",
    }
)
_DATABRICKS_ENGINE_PROBE_INPUT_ARTIFACT_PARAMETER_FLAGS = frozenset(
    {
        "--handoff-json",
        "--payload-uri",
        "--vllm-runtime-preflight-layer-names-json",
    }
)
_DATABRICKS_SDK_PROFILE_ISOLATION_ATTRIBUTES = frozenset(
    {
        "account_id",
        "auth_type",
        "azure_workspace_resource_id",
        "cloud",
        "config_file",
        "discovery_url",
        "host",
        "oidc_token_env",
        "oidc_token_filepath",
        "profile",
        "token_audience",
        "workspace_id",
    }
)
DATABRICKS_TERMINAL_LIFE_CYCLE_STATES = frozenset(
    {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}
)
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_DATABRICKS_PAT_TOKEN_RE = re.compile(r"dapi[0-9a-fA-F]{32}")
_DATABRICKS_RUN_STATUS_WRAPPER_KEYS = frozenset({"ok", "action", "summary"})
_DATABRICKS_RUN_STATUS_KEYS = frozenset(
    {
        "record_type",
        "run_id",
        "run_name",
        "run_page_url",
        "life_cycle_state",
        "result_state",
        "state_message",
        "start_time",
        "end_time",
        "terminal",
        "succeeded",
        "active_task_key",
        "task_count",
        "tasks",
        "cluster_id",
        "submit_payload",
    }
)
_DATABRICKS_RUN_STATUS_TASK_KEYS = frozenset(
    {
        "task_key",
        "run_id",
        "life_cycle_state",
        "result_state",
        "state_message",
        "cluster_id",
        "node_type_id",
        "driver_node_type_id",
        "start_time",
        "end_time",
        "spark_env_keys",
    }
)
_DATABRICKS_SUBMIT_PAYLOAD_KEYS = frozenset(
    {
        "record_type",
        "source_path",
        "sha256",
        "run_name",
        "task_count",
        "task_keys",
        "tasks",
        "node_type_ids",
        "driver_node_type_ids",
        "hardware_targets",
        "spark_versions",
        "spark_env_keys",
        "data_security_modes",
        "single_node",
        *_DATABRICKS_GPU_TYPE_FIELDS,
    }
)
_DATABRICKS_SUBMIT_PAYLOAD_TASK_KEYS = frozenset(
    {
        "task_key",
        "node_type_id",
        "driver_node_type_id",
        "spark_version",
        "spark_env_keys",
        "data_security_mode",
        "num_workers",
        "single_node",
        "purpose",
    }
)
_SPARK_ENV_VAR_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SECRET_LIKE_SPARK_ENV_KEY_PARTS = frozenset(
    {
        "CREDENTIAL",
        "CREDENTIALS",
        "KEY",
        "PASS",
        "PASSWORD",
        "PAT",
        "SECRET",
        "TOKEN",
    }
)
_ENV_KEY_PART_RE = re.compile(r"[A-Za-z0-9]+")
_REDACTED_SPARK_ENV_TOKEN_KEY = "[REDACTED_DATABRICKS_TOKEN_KEY]"


class DatabricksHTTPResponse(Protocol):
    status: int

    def __enter__(self) -> "DatabricksHTTPResponse": ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool: ...

    def read(self, amt: int = -1) -> bytes: ...


class DatabricksBinaryHTTPResponse(Protocol):
    status: int
    headers: Mapping[str, str]

    def __enter__(self) -> "DatabricksBinaryHTTPResponse": ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool: ...

    def read(self, amt: int = -1) -> bytes: ...


class DatabricksBinaryURLOpener(Protocol):
    def __call__(
        self, request: urllib.request.Request, *, timeout: float
    ) -> DatabricksBinaryHTTPResponse: ...


class _DatabricksNoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject every redirect before a bearer-authenticated follow-up request."""

    def redirect_request(
        self,
        _request: urllib.request.Request,
        _file_pointer: Any,
        _status_code: int,
        _message: str,
        _headers: Any,
        _new_url: str,
    ) -> None:
        return None


def _databricks_no_redirect_urlopen(
    request: urllib.request.Request,
    *,
    timeout: float,
) -> DatabricksBinaryHTTPResponse:
    opener = urllib.request.build_opener(_DatabricksNoRedirectHandler())
    return cast(
        DatabricksBinaryHTTPResponse,
        opener.open(request, timeout=timeout),
    )


class _DatabricksLocalUploadError(RuntimeError):
    """A local source invariant failed before an upload could be trusted."""


class _DatabricksStreamingUploadBody:
    """Single-use bounded iterable over one already-verified file descriptor."""

    def __init__(
        self,
        file_descriptor: int,
        source_path: Path,
        identity: tuple[int, ...],
        size_bytes: int,
    ) -> None:
        self._file_descriptor = file_descriptor
        self._source_path = source_path
        self._identity = identity
        self._size_bytes = size_bytes
        self._started = False
        self._complete = False
        self._digest = hashlib.sha256()

    def __iter__(self) -> Iterator[bytes]:
        if self._started:
            raise _DatabricksLocalUploadError(
                "Databricks Files API local upload body is single-use"
            )
        self._started = True
        sent_bytes = 0
        while sent_bytes < self._size_bytes:
            _require_stable_databricks_local_upload_source(
                self._file_descriptor,
                self._source_path,
                self._identity,
            )
            read_size = min(
                _DATABRICKS_VOLUME_FILE_STREAM_CHUNK_BYTES,
                self._size_bytes - sent_bytes,
            )
            try:
                chunk = os.read(self._file_descriptor, read_size)
            except OSError:
                raise _DatabricksLocalUploadError(
                    "Databricks Files API local upload source read failed"
                ) from None
            if type(chunk) is not bytes:
                raise _DatabricksLocalUploadError(
                    "Databricks Files API local upload source chunk must be bytes"
                )
            if not chunk:
                raise _DatabricksLocalUploadError(
                    "Databricks Files API local upload source ended before its "
                    "verified size"
                )
            if len(chunk) > read_size:
                raise _DatabricksLocalUploadError(
                    "Databricks Files API local upload source exceeded the chunk "
                    "byte cap"
                )
            sent_bytes += len(chunk)
            self._digest.update(chunk)
            _require_stable_databricks_local_upload_source(
                self._file_descriptor,
                self._source_path,
                self._identity,
            )
            yield chunk
        try:
            trailing = os.read(self._file_descriptor, 1)
        except OSError:
            raise _DatabricksLocalUploadError(
                "Databricks Files API local upload source EOF probe failed"
            ) from None
        if type(trailing) is not bytes:
            raise _DatabricksLocalUploadError(
                "Databricks Files API local upload source EOF probe must return bytes"
            )
        if trailing:
            raise _DatabricksLocalUploadError(
                "Databricks Files API local upload source contains bytes beyond its "
                "verified size"
            )
        _require_stable_databricks_local_upload_source(
            self._file_descriptor,
            self._source_path,
            self._identity,
        )
        self._complete = True

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


class DatabricksPreReservedPostClaimExistsError(RuntimeError):
    """A pre-reserved member already owns the one durable POST claim."""


class DatabricksURLOpener(Protocol):
    def __call__(
        self, request: urllib.request.Request, *, timeout: float
    ) -> DatabricksHTTPResponse: ...


@dataclass(frozen=True, slots=True)
class DatabricksWorkspaceConfig:
    host: str
    token: str = field(repr=False)
    timeout_seconds: float = DEFAULT_DATABRICKS_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("host must be non-empty")
        if not self.token:
            raise ValueError("token must be non-empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    @property
    def normalized_host(self) -> str:
        return self.host.rstrip("/")


def databricks_workspace_config_from_env(
    *,
    host_env: str = DEFAULT_DATABRICKS_HOST_ENV,
    token_env: str = DEFAULT_DATABRICKS_TOKEN_ENV,
    timeout_seconds: float = DEFAULT_DATABRICKS_TIMEOUT_SECONDS,
    environ: dict[str, str] | None = None,
) -> DatabricksWorkspaceConfig:
    env = os.environ if environ is None else environ
    host = env.get(host_env, "")
    token = env.get(token_env, "")
    if not host:
        raise ValueError(f"{host_env} must be set")
    if not token:
        raise ValueError(f"{token_env} must be set")
    return DatabricksWorkspaceConfig(
        host=host, token=token, timeout_seconds=timeout_seconds
    )


def databricks_workspace_config_from_profile(
    profile: str,
    *,
    config_file: str | Path = DEFAULT_DATABRICKS_CONFIG_FILE,
    timeout_seconds: float = DEFAULT_DATABRICKS_TIMEOUT_SECONDS,
    profile_auth_mode: str = "auto",
) -> DatabricksWorkspaceConfig:
    auth_mode = _databricks_profile_auth_mode(profile_auth_mode)
    profile_name = _required_profile_name(profile)
    values = _databricks_profile_values(profile_name, config_file=config_file)
    host = values.get("host", "").strip()
    token = values.get("token", "").strip()
    if auth_mode == "sdk":
        return databricks_workspace_config_from_sdk_profile(
            profile_name,
            config_file=config_file,
            timeout_seconds=timeout_seconds,
        )
    if host and token:
        return DatabricksWorkspaceConfig(
            host=host, token=token, timeout_seconds=timeout_seconds
        )
    if auth_mode == "auto" and values.get("auth_type", "").strip():
        return databricks_workspace_config_from_sdk_profile(
            profile_name,
            config_file=config_file,
            timeout_seconds=timeout_seconds,
        )
    if not host:
        raise ValueError(f"Databricks profile {profile_name!r} is missing host")
    raise ValueError(f"Databricks profile {profile_name!r} is missing token")


def databricks_workspace_config_from_sdk_profile(
    profile: str,
    *,
    config_file: str | Path = DEFAULT_DATABRICKS_CONFIG_FILE,
    timeout_seconds: float = DEFAULT_DATABRICKS_TIMEOUT_SECONDS,
) -> DatabricksWorkspaceConfig:
    profile_name = _required_profile_name(profile)
    values = _databricks_profile_values(profile_name, config_file=config_file)
    host = values.get("host", "").strip()
    if not host:
        raise ValueError(f"Databricks profile {profile_name!r} is missing host")
    try:
        sdk_config = _databricks_sdk_config(
            profile_name,
            config_file=config_file,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        message = _redact_databricks_secret_text(str(exc))
        raise ValueError(
            f"Databricks SDK profile {profile_name!r} could not be loaded: {message}"
        ) from None
    resolved_host = str(getattr(sdk_config, "host", "") or host).strip()
    try:
        auth_headers = sdk_config.authenticate()
    except Exception as exc:
        message = _redact_databricks_secret_text(str(exc))
        raise ValueError(
            f"Databricks SDK profile {profile_name!r} could not authenticate: {message}"
        ) from None
    token = _databricks_bearer_token(auth_headers)
    if not token:
        raise ValueError(
            f"Databricks SDK profile {profile_name!r} did not return a Bearer Authorization header"
        )
    return DatabricksWorkspaceConfig(
        host=resolved_host,
        token=token,
        timeout_seconds=timeout_seconds,
    )


def _databricks_profile_values(
    profile_name: str,
    *,
    config_file: str | Path,
) -> Mapping[str, str]:
    path = Path(config_file).expanduser()
    parser = ConfigParser(default_section=_DATABRICKS_CONFIG_DEFAULT_SECTION)
    read_files = parser.read(path)
    if not read_files:
        raise ValueError(f"Databricks config file was not found: {path}")
    if profile_name not in parser:
        raise ValueError(f"Databricks profile {profile_name!r} was not found in {path}")
    return {key: value for key, value in parser[profile_name].items()}


def _databricks_sdk_config(
    profile_name: str,
    *,
    config_file: str | Path,
    timeout_seconds: float,
) -> Any:
    try:
        from databricks.sdk.core import Config
    except ModuleNotFoundError:
        raise ValueError(
            "Databricks SDK profile auth requires installing the databricks extra "
            "with cachet-kv[databricks]"
        ) from None
    env_snapshot = _unset_environment(_databricks_sdk_profile_env_names(Config))
    try:
        return Config(
            profile=profile_name,
            config_file=str(Path(config_file).expanduser()),
            http_timeout_seconds=timeout_seconds,
            disable_async_token_refresh=True,
        )
    except Exception as exc:
        message = _redact_databricks_secret_text(str(exc))
        raise ValueError(
            f"Databricks SDK profile {profile_name!r} could not be loaded: {message}"
        ) from None
    finally:
        _restore_environment(env_snapshot)


def _databricks_sdk_profile_env_names(config_type: Any) -> tuple[str, ...]:
    env_names: set[str] = set()
    for attribute in config_type.attributes():
        if not getattr(attribute, "auth", None) and (
            getattr(attribute, "name", None)
            not in _DATABRICKS_SDK_PROFILE_ISOLATION_ATTRIBUTES
        ):
            continue
        env_name = getattr(attribute, "env", None)
        if env_name:
            env_names.add(env_name)
        env_names.update(getattr(attribute, "env_aliases", ()) or ())
    return tuple(sorted(env_names))


def _unset_environment(env_names: Sequence[str]) -> dict[str, str]:
    snapshot = {name: os.environ[name] for name in env_names if name in os.environ}
    for name in snapshot:
        del os.environ[name]
    return snapshot


def _restore_environment(snapshot: Mapping[str, str]) -> None:
    os.environ.update(snapshot)


def _databricks_bearer_token(headers: Mapping[str, str]) -> str:
    if not isinstance(headers, Mapping):
        return ""
    authorization = headers.get("Authorization") or headers.get("authorization") or ""
    match = re.fullmatch(r"(?i)Bearer\s+(.+)", authorization.strip())
    return match.group(1).strip() if match else ""


def _required_profile_name(profile: str) -> str:
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("profile must be a non-empty string")
    return profile.strip()


def _databricks_profile_auth_mode(value: str) -> str:
    if value not in DATABRICKS_PROFILE_AUTH_MODES:
        raise ValueError(
            f"profile_auth_mode must be one of {DATABRICKS_PROFILE_AUTH_MODES!r}, got {value!r}"
        )
    return value


def bind_databricks_run_idempotency_token(
    payload: Mapping[str, Any],
    *,
    attempt_id: str,
) -> dict[str, Any]:
    """Return canonical payload bytes bound to one deterministic Jobs token.

    The token is derived from the attempt identity and the complete payload with
    the token field removed, so changing either the intended attempt or any
    submitted byte produces a different token.  Existing caller-supplied tokens
    are accepted only when they are exactly the package-derived value.
    """

    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("attempt_id must be a non-empty string")
    snapshot, _canonical = canonical_databricks_submit_payload_snapshot(payload)
    existing = snapshot.pop("idempotency_token", None)
    identity = {
        "attempt_id": attempt_id,
        "domain": _DATABRICKS_IDEMPOTENCY_TOKEN_DOMAIN,
        "submit_payload_without_idempotency_token": snapshot,
    }
    identity_bytes = json.dumps(
        identity,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    token = "cachet-" + hashlib.sha256(identity_bytes).hexdigest()[:57]
    if len(token) != 64 or _DATABRICKS_IDEMPOTENCY_TOKEN_RE.fullmatch(token) is None:
        raise RuntimeError("derived Databricks idempotency token is invalid")
    if existing is not None and existing != token:
        raise ValueError("caller-supplied Databricks idempotency token drift")
    snapshot["idempotency_token"] = token
    canonical, _canonical_bytes = canonical_databricks_submit_payload_snapshot(snapshot)
    return canonical


def require_databricks_run_idempotency_token(
    payload: Mapping[str, Any],
    *,
    attempt_id: str,
) -> str:
    """Require the exact deterministic token and return it."""

    observed = payload.get("idempotency_token")
    if not isinstance(observed, str):
        raise ValueError("Databricks runs/submit payload lacks idempotency_token")
    expected = bind_databricks_run_idempotency_token(
        payload,
        attempt_id=attempt_id,
    )["idempotency_token"]
    if observed != expected:
        raise ValueError("Databricks runs/submit idempotency_token binding drift")
    return observed


def submit_databricks_run(
    config: DatabricksWorkspaceConfig,
    payload: dict[str, Any],
    *,
    opener: DatabricksURLOpener = _databricks_no_redirect_urlopen,
) -> dict[str, Any]:
    return _databricks_api_json(
        config,
        "POST",
        "/api/2.1/jobs/runs/submit",
        payload=payload,
        opener=opener,
    )


def reserve_and_submit_databricks_run(
    config: DatabricksWorkspaceConfig,
    payload: Mapping[str, Any],
    *,
    ledger_path: str | Path,
    attempt_id: str,
    workload_id: str,
    reservation_validator: DatabricksReservationValidator | None = None,
    opener: DatabricksURLOpener | None = None,
) -> dict[str, Any]:
    """Reserve and submit one immutable payload snapshot as a single local action.

    Reservation is durably written immediately before the POST.  The wire body
    is the same canonical byte snapshot whose digest is stored in the ledger.
    Submission failures deliberately do not reconcile or remove the reservation.
    """

    resolved_opener = (
        cast(DatabricksURLOpener, _databricks_no_redirect_urlopen)
        if opener is None
        else opener
    )
    snapshot, canonical_payload = canonical_databricks_submit_payload_snapshot(payload)
    payload_sha256 = hashlib.sha256(canonical_payload).hexdigest()

    def validate_exact_reservation(
        reservation: DatabricksClusterHourReservation,
        validated_snapshot: Mapping[str, Any],
    ) -> None:
        if reservation.submit_payload_sha256 != payload_sha256:
            raise RuntimeError(
                "ledger reservation digest does not match the submit payload snapshot"
            )
        if reservation_validator is not None:
            reservation_validator(reservation, validated_snapshot)

    ledger = reserve_databricks_run_attempt_json(
        ledger_path,
        snapshot,
        attempt_id=attempt_id,
        workload_id=workload_id,
        reservation_validator=validate_exact_reservation,
    )
    persisted_reservation = next(
        item for item in ledger.reservations if item.attempt_id == attempt_id
    )
    if persisted_reservation.submit_payload_sha256 != payload_sha256:
        raise RuntimeError(
            "persisted ledger reservation digest does not match the submit payload snapshot"
        )
    response = _databricks_api_json(
        config,
        "POST",
        "/api/2.1/jobs/runs/submit",
        payload_json_bytes=canonical_payload,
        opener=resolved_opener,
    )
    # The success path is not complete until the returned cloud run identity is
    # durably joined to the pre-POST reservation.  HTTP failures remain
    # conservatively reserved with no fabricated receipt because the remote
    # outcome may be ambiguous.
    record_databricks_run_submission_receipt_json(
        ledger_path,
        attempt_id=attempt_id,
        submit_response=response,
    )
    return response


def submit_pre_reserved_databricks_run(
    config: DatabricksWorkspaceConfig,
    payload: Mapping[str, Any],
    *,
    ledger_path: str | Path,
    attempt_id: str,
    batch_authorization: DatabricksBatchReservationAuthorization,
    opener: DatabricksURLOpener | None = None,
) -> dict[str, Any]:
    """Submit one member only after its exact batch was atomically reserved."""

    resolved_opener = (
        cast(DatabricksURLOpener, _databricks_no_redirect_urlopen)
        if opener is None
        else opener
    )
    snapshot, canonical_payload = canonical_databricks_submit_payload_snapshot(payload)
    payload_sha256 = hashlib.sha256(canonical_payload).hexdigest()
    idempotency_token = require_databricks_run_idempotency_token(
        snapshot,
        attempt_id=attempt_id,
    )
    if databricks_ledger_path_sha256(ledger_path) != (
        batch_authorization.ledger_path_sha256
    ):
        raise ValueError("pre-reserved submission ledger path binding drift")
    batch_prefix = require_databricks_batch_reservation_authorization(
        batch_authorization,
        expected_predecessor_prefix=batch_authorization.predecessor_prefix,
        expected_attempt_ids=batch_authorization.attempt_ids,
        expected_submit_payload_sha256s=batch_authorization.submit_payload_sha256s,
    )
    if attempt_id not in batch_authorization.attempt_ids:
        raise ValueError("attempt is not a member of the authorized atomic batch")
    member_index = batch_authorization.attempt_ids.index(attempt_id)
    if batch_authorization.submit_payload_sha256s[member_index] != payload_sha256:
        raise ValueError("authorized batch member payload digest drift")
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    require_databricks_ledger_prefix(ledger, batch_prefix)
    reservation = next(
        (item for item in ledger.reservations if item.attempt_id == attempt_id),
        None,
    )
    if reservation is None or reservation.submit_payload_sha256 != payload_sha256:
        raise ValueError("pre-reserved attempt does not match the submit payload")
    if any(item.attempt_id == attempt_id for item in ledger.submission_receipts):
        raise ValueError("pre-reserved attempt already has a submission receipt")
    if attempt_id in ledger.closed_attempt_ids:
        raise ValueError("pre-reserved attempt is already terminal")
    claim_path = _write_pre_reserved_post_claim(
        ledger_path,
        attempt_id=attempt_id,
        batch_authorization=batch_authorization,
        submit_payload_sha256=payload_sha256,
        idempotency_token=idempotency_token,
    )
    with _exclusive_pre_reserved_recovery_lock(claim_path):
        response = _databricks_api_json(
            config,
            "POST",
            "/api/2.1/jobs/runs/submit",
            payload_json_bytes=canonical_payload,
            opener=resolved_opener,
        )
        record_databricks_run_submission_receipt_json(
            ledger_path,
            attempt_id=attempt_id,
            submit_response=response,
        )
    return response


def recover_pre_reserved_databricks_run(
    config: DatabricksWorkspaceConfig,
    payload: Mapping[str, Any],
    *,
    ledger_path: str | Path,
    attempt_id: str,
    batch_authorization: DatabricksBatchReservationAuthorization,
    opener: DatabricksURLOpener | None = None,
) -> dict[str, Any]:
    """Safely repeat an ambiguous POST using its exact idempotent wire body.

    Recovery is possible only after the original durable claim exists.  A
    per-member advisory lock serializes the original request and all recoveries;
    once any caller records the receipt, later callers return its run ID without
    issuing another POST.
    """

    resolved_opener = (
        cast(DatabricksURLOpener, _databricks_no_redirect_urlopen)
        if opener is None
        else opener
    )
    snapshot, canonical_payload = canonical_databricks_submit_payload_snapshot(payload)
    payload_sha256 = hashlib.sha256(canonical_payload).hexdigest()
    idempotency_token = require_databricks_run_idempotency_token(
        snapshot,
        attempt_id=attempt_id,
    )
    _validate_pre_reserved_batch_member(
        ledger_path,
        attempt_id=attempt_id,
        payload_sha256=payload_sha256,
        batch_authorization=batch_authorization,
        allow_existing_receipt=True,
    )
    claim_path = _pre_reserved_post_claim_path(ledger_path, attempt_id=attempt_id)
    with _exclusive_pre_reserved_recovery_lock(claim_path):
        _require_pre_reserved_post_claim(
            claim_path,
            attempt_id=attempt_id,
            batch_authorization=batch_authorization,
            submit_payload_sha256=payload_sha256,
            idempotency_token=idempotency_token,
        )
        ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
        receipt = next(
            (
                item
                for item in ledger.submission_receipts
                if item.attempt_id == attempt_id
            ),
            None,
        )
        if receipt is not None:
            return {"run_id": receipt.run_id}
        response = _databricks_api_json(
            config,
            "POST",
            "/api/2.1/jobs/runs/submit",
            payload_json_bytes=canonical_payload,
            opener=resolved_opener,
        )
        updated = record_databricks_run_submission_receipt_json(
            ledger_path,
            attempt_id=attempt_id,
            submit_response=response,
        )
        receipt = next(
            item
            for item in updated.submission_receipts
            if item.attempt_id == attempt_id
        )
        return {"run_id": receipt.run_id}


def resume_pre_reserved_databricks_run(
    config: DatabricksWorkspaceConfig,
    payload: Mapping[str, Any],
    *,
    ledger_path: str | Path,
    attempt_id: str,
    batch_authorization: DatabricksBatchReservationAuthorization,
    opener: DatabricksURLOpener | None = None,
) -> dict[str, Any]:
    """Resume one exact batch member after any controller crash point.

    A missing durable POST claim means the member was never submitted and is
    claimed/submitted now.  An existing claim is recovered with the identical
    idempotency token and wire bytes.  Races converge through O_EXCL claim
    creation plus the per-member advisory lock, so at most one new cloud run is
    created and all successful callers observe the same receipt-bound run ID.
    """

    claim_path = _pre_reserved_post_claim_path(ledger_path, attempt_id=attempt_id)
    if claim_path.exists() or claim_path.is_symlink():
        return recover_pre_reserved_databricks_run(
            config,
            payload,
            ledger_path=ledger_path,
            attempt_id=attempt_id,
            batch_authorization=batch_authorization,
            opener=opener,
        )
    try:
        return submit_pre_reserved_databricks_run(
            config,
            payload,
            ledger_path=ledger_path,
            attempt_id=attempt_id,
            batch_authorization=batch_authorization,
            opener=opener,
        )
    except DatabricksPreReservedPostClaimExistsError:
        return recover_pre_reserved_databricks_run(
            config,
            payload,
            ledger_path=ledger_path,
            attempt_id=attempt_id,
            batch_authorization=batch_authorization,
            opener=opener,
        )


def _write_pre_reserved_post_claim(
    ledger_path: str | Path,
    *,
    attempt_id: str,
    batch_authorization: DatabricksBatchReservationAuthorization,
    submit_payload_sha256: str,
    idempotency_token: str,
) -> Path:
    """Durably claim one batch member once before its potentially ambiguous POST."""

    path = Path(ledger_path).expanduser().absolute()
    claim_path = _pre_reserved_post_claim_path(path, attempt_id=attempt_id)
    claim_root = claim_path.parent
    if claim_root.is_symlink():
        raise ValueError("pre-reserved POST claim root must not be a symlink")
    try:
        claim_root.mkdir(mode=0o700)
        _fsync_local_directory(claim_root.parent)
    except FileExistsError:
        pass
    if not claim_root.is_dir() or claim_root.is_symlink():
        raise ValueError("pre-reserved POST claim root must be a real directory")
    record = {
        "attempt_id": attempt_id,
        "batch_prefix_sha256": batch_authorization.batch_prefix.prefix_sha256,
        "idempotency_token": idempotency_token,
        "ledger_path_sha256": batch_authorization.ledger_path_sha256,
        "record_type": "cachet.databricks_pre_reserved_post_claim.v1",
        "submit_payload_sha256": submit_payload_sha256,
    }
    content = (
        json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(claim_path, flags, 0o600)
    except FileExistsError as exc:
        raise DatabricksPreReservedPostClaimExistsError(
            "pre-reserved POST already has a durable claim; its outcome may be "
            "ambiguous and must be reconciled before any retry"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A created claim is deliberately retained even if its durable write is
        # interrupted: absence of a receipt is an ambiguous recovery state.
        raise
    _fsync_local_directory(claim_root)
    return claim_path


def _pre_reserved_post_claim_path(
    ledger_path: str | Path,
    *,
    attempt_id: str,
) -> Path:
    path = Path(ledger_path).expanduser().absolute()
    claim_root = path.with_name(f"{path.name}.post-claims")
    claim_name = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest() + ".json"
    return claim_root / claim_name


def _require_pre_reserved_post_claim(
    claim_path: Path,
    *,
    attempt_id: str,
    batch_authorization: DatabricksBatchReservationAuthorization,
    submit_payload_sha256: str,
    idempotency_token: str,
) -> None:
    if claim_path.is_symlink() or not claim_path.is_file():
        raise ValueError("pre-reserved recovery requires the durable POST claim")
    content = claim_path.read_bytes()
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("pre-reserved POST claim is invalid JSON") from exc
    expected = {
        "attempt_id": attempt_id,
        "batch_prefix_sha256": batch_authorization.batch_prefix.prefix_sha256,
        "idempotency_token": idempotency_token,
        "ledger_path_sha256": batch_authorization.ledger_path_sha256,
        "record_type": "cachet.databricks_pre_reserved_post_claim.v1",
        "submit_payload_sha256": submit_payload_sha256,
    }
    canonical = (
        json.dumps(
            expected,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if value != expected or content != canonical:
        raise ValueError("pre-reserved POST claim binding drift")


def _validate_pre_reserved_batch_member(
    ledger_path: str | Path,
    *,
    attempt_id: str,
    payload_sha256: str,
    batch_authorization: DatabricksBatchReservationAuthorization,
    allow_existing_receipt: bool,
) -> None:
    if databricks_ledger_path_sha256(ledger_path) != (
        batch_authorization.ledger_path_sha256
    ):
        raise ValueError("pre-reserved submission ledger path binding drift")
    batch_prefix = require_databricks_batch_reservation_authorization(
        batch_authorization,
        expected_predecessor_prefix=batch_authorization.predecessor_prefix,
        expected_attempt_ids=batch_authorization.attempt_ids,
        expected_submit_payload_sha256s=batch_authorization.submit_payload_sha256s,
    )
    if attempt_id not in batch_authorization.attempt_ids:
        raise ValueError("attempt is not a member of the authorized atomic batch")
    member_index = batch_authorization.attempt_ids.index(attempt_id)
    if batch_authorization.submit_payload_sha256s[member_index] != payload_sha256:
        raise ValueError("authorized batch member payload digest drift")
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    require_databricks_ledger_prefix(ledger, batch_prefix)
    reservation = next(
        (item for item in ledger.reservations if item.attempt_id == attempt_id),
        None,
    )
    if reservation is None or reservation.submit_payload_sha256 != payload_sha256:
        raise ValueError("pre-reserved attempt does not match the submit payload")
    has_receipt = any(
        item.attempt_id == attempt_id for item in ledger.submission_receipts
    )
    if has_receipt and not allow_existing_receipt:
        raise ValueError("pre-reserved attempt already has a submission receipt")
    if attempt_id in ledger.closed_attempt_ids:
        raise ValueError("pre-reserved attempt is already terminal")


@contextmanager
def _exclusive_pre_reserved_recovery_lock(claim_path: Path):
    lock_path = claim_path.with_suffix(".lock")
    if lock_path.is_symlink():
        raise ValueError("pre-reserved recovery lock must not be a symlink")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _fsync_local_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def reserve_and_submit_databricks_run_json(
    config: DatabricksWorkspaceConfig,
    payload_path: str | Path,
    *,
    ledger_path: str | Path,
    attempt_id: str,
    workload_id: str,
    reservation_validator: DatabricksReservationValidator | None = None,
    opener: DatabricksURLOpener | None = None,
) -> dict[str, Any]:
    """Read a payload once, then reserve and submit that isolated snapshot."""

    payload = read_databricks_run_submit_payload(payload_path)
    return reserve_and_submit_databricks_run(
        config,
        payload,
        ledger_path=ledger_path,
        attempt_id=attempt_id,
        workload_id=workload_id,
        reservation_validator=reservation_validator,
        opener=opener,
    )


def check_databricks_auth(
    config: DatabricksWorkspaceConfig,
    *,
    opener: DatabricksURLOpener = _databricks_no_redirect_urlopen,
) -> dict[str, Any]:
    response, status = _databricks_api_response_json(
        config,
        "GET",
        _DATABRICKS_AUTH_CHECK_ENDPOINT,
        opener=opener,
    )
    return {
        "record_type": DATABRICKS_AUTH_CHECK_RECORD_TYPE,
        "authenticated": True,
        "endpoint": _DATABRICKS_AUTH_CHECK_ENDPOINT,
        "http_status": status,
        "workspace_host_sha256": _sha256_hex(config.normalized_host.encode("utf-8")),
        "response_keys": sorted(str(key) for key in response),
    }


def require_databricks_current_user_name(
    config: DatabricksWorkspaceConfig,
    *,
    expected_user_name: str,
    opener: DatabricksURLOpener | None = None,
) -> dict[str, Any]:
    """Authenticate one exact Databricks ``SINGLE_USER`` principal.

    The SCIM ``Me`` response is consumed only inside this package boundary.  The
    returned attestation binds a digest of the principal so callers do not need
    to persist or log the workspace identity.
    """

    if (
        not isinstance(expected_user_name, str)
        or not expected_user_name
        or expected_user_name.strip() != expected_user_name
    ):
        raise ValueError("expected_user_name must be a normalized non-empty string")
    resolved_opener = (
        cast(DatabricksURLOpener, _databricks_no_redirect_urlopen)
        if opener is None
        else opener
    )
    response, status = _databricks_api_response_json(
        config,
        "GET",
        _DATABRICKS_AUTH_CHECK_ENDPOINT,
        opener=resolved_opener,
    )
    observed_user_name = response.get("userName")
    if (
        not isinstance(observed_user_name, str)
        or not observed_user_name
        or observed_user_name.strip() != observed_user_name
    ):
        raise ValueError("Databricks current-user response lacks a normalized userName")
    if response.get("active") is not True:
        raise ValueError("Databricks current-user identity is not explicitly active")
    if observed_user_name != expected_user_name:
        raise ValueError(
            "Databricks current-user identity differs from single_user_name"
        )
    return {
        "record_type": "document_kv.databricks_current_user_binding.v1",
        "authenticated": True,
        "endpoint": _DATABRICKS_AUTH_CHECK_ENDPOINT,
        "http_status": status,
        "user_name_sha256": _sha256_hex(observed_user_name.encode("utf-8")),
        "workspace_host_sha256": _sha256_hex(config.normalized_host.encode("utf-8")),
    }


def get_databricks_run(
    config: DatabricksWorkspaceConfig,
    run_id: int | str,
    *,
    opener: DatabricksURLOpener = _databricks_no_redirect_urlopen,
) -> dict[str, Any]:
    run_id_text = str(run_id)
    if not run_id_text:
        raise ValueError("run_id must be non-empty")
    return _databricks_api_json(
        config,
        "GET",
        f"/api/2.1/jobs/runs/get?{urllib.parse.urlencode({'run_id': run_id_text})}",
        opener=opener,
    )


def get_databricks_run_output(
    config: DatabricksWorkspaceConfig,
    run_id: int | str,
    *,
    opener: DatabricksBinaryURLOpener | None = None,
) -> dict[str, Any]:
    """Return one bounded direct Jobs ``runs/get-output`` response."""

    run_id_text = str(run_id)
    if not run_id_text:
        raise ValueError("run_id must be non-empty")
    request = _databricks_request(
        config,
        "GET",
        (
            "/api/2.1/jobs/runs/get-output?"
            f"{urllib.parse.urlencode({'run_id': run_id_text})}"
        ),
        payload=None,
    )
    resolved_opener = (
        _databricks_no_redirect_urlopen
        if opener is None
        else opener
    )
    return _bounded_databricks_json_object(
        config,
        request,
        max_bytes=DATABRICKS_API_PAGE_MAX_BYTES,
        opener=resolved_opener,
        label="Databricks run output",
    )


def list_active_databricks_runs(
    config: DatabricksWorkspaceConfig,
    *,
    max_runs: int = DATABRICKS_ACTIVE_RUNS_MAX_ENTRIES,
    opener: DatabricksBinaryURLOpener | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return a bounded, sanitized snapshot of every active Jobs run."""

    _validate_databricks_entry_cap(
        max_runs,
        upper_bound=DATABRICKS_ACTIVE_RUNS_MAX_ENTRIES,
        label="max_runs",
    )
    resolved_opener = (
        _databricks_no_redirect_urlopen
        if opener is None
        else opener
    )
    runs: dict[int, dict[str, Any]] = {}
    seen_tokens: set[str] = set()
    page_token: str | None = None
    page_count = 0
    while True:
        page_count += 1
        if page_count > DATABRICKS_ACTIVE_RUNS_MAX_PAGES:
            raise RuntimeError(
                "Databricks active-runs snapshot exceeds the controller page cap"
            )
        page_limit = min(_DATABRICKS_ACTIVE_RUNS_PAGE_SIZE, max_runs - len(runs))
        if page_limit <= 0:
            raise RuntimeError(
                "Databricks active-runs snapshot exceeds the controller entry cap"
            )
        query = {"active_only": "true", "limit": str(page_limit)}
        if page_token is not None:
            query["page_token"] = page_token
        request = urllib.request.Request(
            f"{config.normalized_host}/api/2.1/jobs/runs/list?"
            f"{urllib.parse.urlencode(query)}",
            method="GET",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {config.token}",
            },
        )
        page = _bounded_databricks_json_object(
            config,
            request,
            max_bytes=DATABRICKS_API_PAGE_MAX_BYTES,
            opener=resolved_opener,
            label="Databricks active-runs snapshot",
        )
        if set(page) - {"has_more", "next_page_token", "prev_page_token", "runs"}:
            raise RuntimeError("Databricks active-runs response schema drift")
        raw_runs = page.get("runs", [])
        if not isinstance(raw_runs, list) or len(raw_runs) > page_limit:
            raise RuntimeError("Databricks active-runs page entry cap drift")
        for raw_run in raw_runs:
            run = _validated_active_databricks_run(raw_run, token=config.token)
            run_id = cast(int, run["run_id"])
            if run_id in runs:
                raise RuntimeError("Databricks active-runs response duplicates run_id")
            runs[run_id] = run
            if len(runs) > max_runs:
                raise RuntimeError(
                    "Databricks active-runs snapshot exceeds the controller entry cap"
                )
        has_more = page.get("has_more")
        if has_more is not None and type(has_more) is not bool:
            raise RuntimeError("Databricks active-runs has_more is invalid")
        _validated_optional_databricks_page_token(
            page.get("prev_page_token"),
            label="Databricks active-runs prev_page_token",
        )
        next_page_token = _validated_optional_databricks_page_token(
            page.get("next_page_token"),
            label="Databricks active-runs next_page_token",
        )
        if next_page_token is None:
            if has_more is True:
                raise RuntimeError(
                    "Databricks active-runs response omitted its next page token"
                )
            break
        if has_more is False:
            raise RuntimeError(
                "Databricks active-runs pagination metadata is contradictory"
            )
        if next_page_token in seen_tokens:
            raise RuntimeError("Databricks active-runs page token repeated")
        if len(runs) >= max_runs:
            raise RuntimeError(
                "Databricks active-runs snapshot exceeds the controller entry cap"
            )
        seen_tokens.add(next_page_token)
        page_token = next_page_token
    return tuple(runs[run_id] for run_id in sorted(runs))


def list_databricks_node_types(
    config: DatabricksWorkspaceConfig,
    *,
    max_node_types: int = DATABRICKS_NODE_TYPES_MAX_ENTRIES,
    opener: DatabricksBinaryURLOpener | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return the bounded set of launchable node-type identifiers."""

    _validate_databricks_entry_cap(
        max_node_types,
        upper_bound=DATABRICKS_NODE_TYPES_MAX_ENTRIES,
        label="max_node_types",
    )
    request = urllib.request.Request(
        f"{config.normalized_host}/api/2.0/clusters/list-node-types",
        method="GET",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {config.token}",
        },
    )
    resolved_opener = (
        _databricks_no_redirect_urlopen
        if opener is None
        else opener
    )
    response = _bounded_databricks_json_object(
        config,
        request,
        max_bytes=DATABRICKS_API_PAGE_MAX_BYTES,
        opener=resolved_opener,
        label="Databricks node-types snapshot",
    )
    if "node_types" not in response or set(response) - {"node_types", "success"}:
        raise RuntimeError("Databricks node-types response schema drift")
    if "success" in response and (
        not isinstance(response["success"], Mapping) or response["success"]
    ):
        raise RuntimeError("Databricks node-types success marker drift")
    raw_node_types = response.get("node_types")
    if not isinstance(raw_node_types, list):
        raise RuntimeError("Databricks node-types response must contain an array")
    if len(raw_node_types) > max_node_types:
        raise RuntimeError(
            "Databricks node-types snapshot exceeds the controller entry cap"
        )
    node_types: dict[str, dict[str, Any]] = {}
    for raw_node_type in raw_node_types:
        if not isinstance(raw_node_type, Mapping):
            raise RuntimeError("Databricks node-type entry must be an object")
        node_type_id = _validated_databricks_identifier(
            raw_node_type.get("node_type_id"),
            label="Databricks node_type_id",
        )
        _require_databricks_non_secret_text(
            node_type_id,
            token=config.token,
            label="Databricks node_type_id",
        )
        if node_type_id in node_types:
            raise RuntimeError("Databricks node-types response duplicates node_type_id")
        node_types[node_type_id] = {"node_type_id": node_type_id}
    return tuple(node_types[node_type_id] for node_type_id in sorted(node_types))


def download_databricks_volume_file_bytes(
    config: DatabricksWorkspaceConfig,
    dbfs_uri: str,
    *,
    max_bytes: int = DATABRICKS_VOLUME_FILE_MAX_DOWNLOAD_BYTES,
    opener: DatabricksBinaryURLOpener | None = None,
) -> bytes:
    """Download one canonical ``dbfs:/Volumes`` file through the Files API.

    The hard response cap keeps an authenticated but unexpected remote object
    from consuming unbounded controller memory.  Qualification collection calls
    this package-owned transport without exposing its opener at the authority
    boundary; the argument exists here only for ordinary transport testing.
    """

    volume_path = _canonical_databricks_volume_file_path(dbfs_uri)
    if (
        type(max_bytes) is not int
        or max_bytes <= 0
        or max_bytes > DATABRICKS_VOLUME_FILE_MAX_DOWNLOAD_BYTES
    ):
        raise ValueError(
            "max_bytes must be a positive integer no greater than "
            f"{DATABRICKS_VOLUME_FILE_MAX_DOWNLOAD_BYTES}"
        )
    encoded_path = urllib.parse.quote(volume_path, safe="/-._~")
    request = urllib.request.Request(
        f"{config.normalized_host}/api/2.0/fs/files{encoded_path}",
        method="GET",
        headers={
            "Accept": "application/octet-stream",
            "Authorization": f"Bearer {config.token}",
        },
    )
    resolved_opener = (
        _databricks_no_redirect_urlopen
        if opener is None
        else opener
    )
    try:
        with resolved_opener(request, timeout=config.timeout_seconds) as response:
            status = getattr(response, "status", None)
            if type(status) is not int or status != 200:
                raise RuntimeError(
                    f"Databricks Files API returned unexpected HTTP status {status!r}"
                )
            content = _read_databricks_response_bytes_bounded(
                response,
                max_bytes=max_bytes,
                label="Databricks Files API response",
            )
    except urllib.error.HTTPError as exc:
        raise _databricks_binary_http_error(exc, token=config.token) from None
    except urllib.error.URLError as exc:
        reason = _redact_databricks_secret_text(str(exc.reason), token=config.token)
        raise RuntimeError(f"Databricks request failed: {reason}") from None
    except TimeoutError as exc:
        reason = _redact_databricks_secret_text(str(exc), token=config.token)
        raise TimeoutError(reason) from None
    except ConnectionError as exc:
        reason = _redact_databricks_secret_text(str(exc), token=config.token)
        raise ConnectionError(reason) from None
    except Exception as exc:
        reason = _redact_databricks_secret_text(str(exc), token=config.token)
        raise RuntimeError(
            f"Databricks Files API download failed: {reason}"
        ) from None
    if not isinstance(content, bytes):
        raise RuntimeError("Databricks Files API response body must be bytes")
    if len(content) > max_bytes:
        raise RuntimeError(
            "Databricks Files API response exceeds the controller byte cap: "
            f"more than {max_bytes} bytes"
        )
    return content


def stream_databricks_volume_file_sha256(
    config: DatabricksWorkspaceConfig,
    dbfs_uri: str,
    *,
    max_bytes: int = DATABRICKS_VOLUME_FILE_MAX_STREAM_BYTES,
    opener: DatabricksBinaryURLOpener | None = None,
) -> dict[str, Any]:
    """Stream one canonical Volume file and return its exact identity.

    The response is never accumulated in controller memory.  A mandatory
    ``Content-Length`` is admitted before the first body byte, every read is
    bounded to one MiB, and an explicit EOF probe rejects trailing bytes.
    """

    volume_path = _canonical_databricks_volume_file_path(dbfs_uri)
    _validate_databricks_volume_byte_cap(
        max_bytes,
        upper_bound=DATABRICKS_VOLUME_FILE_MAX_STREAM_BYTES,
        label="max_bytes",
    )
    encoded_path = urllib.parse.quote(volume_path, safe="/-._~")
    request = urllib.request.Request(
        f"{config.normalized_host}/api/2.0/fs/files{encoded_path}",
        method="GET",
        headers={
            "Accept": "application/octet-stream",
            "Accept-Encoding": "identity",
            "Authorization": f"Bearer {config.token}",
        },
    )
    resolved_opener = (
        _databricks_no_redirect_urlopen
        if opener is None
        else opener
    )
    try:
        with resolved_opener(request, timeout=config.timeout_seconds) as response:
            status = getattr(response, "status", None)
            if type(status) is not int or status != 200:
                raise RuntimeError(
                    "Databricks Files API stream returned unexpected HTTP "
                    f"status {status!r}"
                )
            raw_headers = getattr(response, "headers", None)
            if raw_headers is None or not hasattr(raw_headers, "get"):
                raise RuntimeError(
                    "Databricks Files API stream response headers are missing"
                )
            transfer_encoding = raw_headers.get("transfer-encoding")
            if transfer_encoding not in (None, ""):
                raise RuntimeError(
                    "Databricks Files API stream transfer-encoding is unexpected"
                )
            content_length = _required_databricks_content_length(
                raw_headers.get("content-length"),
                max_bytes=max_bytes,
                label="Databricks Files API stream",
            )
            content_encoding = raw_headers.get("content-encoding")
            if content_encoding not in (None, "", "identity"):
                raise RuntimeError(
                    "Databricks Files API stream content-encoding is not identity"
                )
            digest = hashlib.sha256()
            size_bytes = 0
            while size_bytes < content_length:
                read_size = min(
                    _DATABRICKS_VOLUME_FILE_STREAM_CHUNK_BYTES,
                    content_length - size_bytes,
                )
                chunk = response.read(read_size)
                if type(chunk) is not bytes:
                    raise RuntimeError(
                        "Databricks Files API stream chunk must be bytes"
                    )
                if not chunk:
                    raise RuntimeError(
                        "Databricks Files API stream ended before content-length"
                    )
                if len(chunk) > read_size:
                    raise RuntimeError(
                        "Databricks Files API stream exceeded the chunk byte cap"
                    )
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    raise RuntimeError(
                        "Databricks Files API stream exceeds the controller byte cap"
                    )
                digest.update(chunk)
            eof = response.read(1)
            if type(eof) is not bytes:
                raise RuntimeError("Databricks Files API EOF probe must return bytes")
            if eof:
                raise RuntimeError(
                    "Databricks Files API stream contains bytes beyond content-length"
                )
    except urllib.error.HTTPError as exc:
        raise _databricks_binary_http_error(exc, token=config.token) from None
    except urllib.error.URLError as exc:
        reason = _redact_databricks_secret_text(str(exc.reason), token=config.token)
        raise RuntimeError(f"Databricks request failed: {reason}") from None
    except Exception as exc:
        reason = _redact_databricks_secret_text(str(exc), token=config.token)
        raise RuntimeError(
            reason or "Databricks Files API stream failed"
        ) from None
    if size_bytes != content_length:
        raise AssertionError("Databricks Files API stream size accounting drift")
    return {
        "dbfs_uri": dbfs_uri,
        "file_sha256": digest.hexdigest(),
        "size_bytes": size_bytes,
    }


def get_databricks_volume_file_metadata(
    config: DatabricksWorkspaceConfig,
    dbfs_uri: str,
    *,
    max_bytes: int = DATABRICKS_VOLUME_FILE_MAX_DOWNLOAD_BYTES,
    opener: DatabricksBinaryURLOpener | None = None,
) -> dict[str, Any]:
    """Read bounded metadata for one canonical UC Volume file via ``HEAD``.

    The Files API exposes file metadata in standard response headers.  A
    content length above *max_bytes* is rejected so a caller cannot use a
    successful metadata probe to authorize an unbounded controller download.
    """

    volume_path = _canonical_databricks_volume_file_path(dbfs_uri)
    _validate_databricks_volume_byte_cap(
        max_bytes,
        upper_bound=DATABRICKS_VOLUME_FILE_MAX_DOWNLOAD_BYTES,
        label="max_bytes",
    )
    encoded_path = urllib.parse.quote(volume_path, safe="/-._~")
    request = urllib.request.Request(
        f"{config.normalized_host}/api/2.0/fs/files{encoded_path}",
        method="HEAD",
        headers={
            "Accept": "application/octet-stream",
            "Authorization": f"Bearer {config.token}",
        },
    )
    resolved_opener = (
        _databricks_no_redirect_urlopen
        if opener is None
        else opener
    )
    try:
        with resolved_opener(request, timeout=config.timeout_seconds) as response:
            status = getattr(response, "status", None)
            if type(status) is not int or status != 200:
                raise RuntimeError(
                    "Databricks Files API metadata returned unexpected HTTP "
                    f"status {status!r}"
                )
            raw_headers = getattr(response, "headers", None)
            if raw_headers is None or not hasattr(raw_headers, "get"):
                raise RuntimeError(
                    "Databricks Files API metadata response headers are missing"
                )
            content_length_raw = raw_headers.get("content-length")
            content_type_raw = raw_headers.get("content-type")
            last_modified_raw = raw_headers.get("last-modified")
            unexpected_body = response.read(1)
    except urllib.error.HTTPError as exc:
        raise _databricks_binary_http_error(exc, token=config.token) from None
    except urllib.error.URLError as exc:
        reason = _redact_databricks_secret_text(str(exc.reason), token=config.token)
        raise RuntimeError(f"Databricks request failed: {reason}") from None
    except Exception as exc:
        reason = _redact_databricks_secret_text(str(exc), token=config.token)
        raise RuntimeError(
            f"Databricks Files API metadata failed: {reason}"
        ) from None
    if unexpected_body != b"":
        raise RuntimeError("Databricks Files API metadata response body is not empty")
    try:
        content_length = int(content_length_raw)
    except (TypeError, ValueError):
        raise RuntimeError(
            "Databricks Files API metadata content-length is invalid"
        ) from None
    if content_length < 0 or content_length > max_bytes:
        raise RuntimeError(
            "Databricks Files API metadata content-length exceeds the controller "
            f"byte cap: {content_length} > {max_bytes}"
        )
    metadata: dict[str, Any] = {
        "content_length": content_length,
        "dbfs_uri": dbfs_uri,
    }
    if content_type_raw is not None:
        metadata["content_type"] = _validated_databricks_metadata_header(
            content_type_raw, "content-type"
        )
    if last_modified_raw is not None:
        metadata["last_modified"] = _validated_databricks_metadata_header(
            last_modified_raw, "last-modified"
        )
    return metadata


def upload_databricks_volume_file_bytes_exclusive(
    config: DatabricksWorkspaceConfig,
    dbfs_uri: str,
    content: bytes,
    *,
    max_bytes: int = DATABRICKS_VOLUME_FILE_MAX_UPLOAD_BYTES,
    opener: DatabricksBinaryURLOpener | None = None,
    readback_opener: DatabricksBinaryURLOpener | None = None,
) -> dict[str, Any]:
    """Exclusively publish bounded bytes to a canonical UC Volume file.

    Every PUT explicitly sends ``overwrite=false``.  If an earlier identical
    PUT won, or the original response was lost, a bounded authenticated GET
    converts that replay into success only when the existing bytes match
    exactly.  Different bytes always fail closed and are never overwritten.
    """

    volume_path = _canonical_databricks_volume_file_path(dbfs_uri)
    _validate_databricks_volume_byte_cap(
        max_bytes,
        upper_bound=DATABRICKS_VOLUME_FILE_MAX_UPLOAD_BYTES,
        label="max_bytes",
    )
    if type(content) is not bytes:
        raise TypeError("content must be bytes")
    if len(content) > max_bytes:
        raise ValueError(
            "content exceeds the Databricks Files API controller upload cap: "
            f"{len(content)} > {max_bytes}"
        )
    encoded_path = urllib.parse.quote(volume_path, safe="/-._~")
    request = urllib.request.Request(
        f"{config.normalized_host}/api/2.0/fs/files{encoded_path}?overwrite=false",
        data=content,
        method="PUT",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {config.token}",
            "Content-Type": "application/octet-stream",
        },
    )
    resolved_opener = (
        _databricks_no_redirect_urlopen
        if opener is None
        else opener
    )
    replay_reason: BaseException | None = None
    try:
        with resolved_opener(request, timeout=config.timeout_seconds) as response:
            status = getattr(response, "status", None)
            if type(status) is not int or status != 204:
                raise RuntimeError(
                    "Databricks Files API exclusive upload returned unexpected "
                    f"HTTP status {status!r}"
                )
            response_body = response.read(1)
            if response_body != b"":
                raise RuntimeError(
                    "Databricks Files API exclusive upload response body is not empty"
                )
    except urllib.error.HTTPError as exc:
        if exc.code not in {400, 409}:
            raise _databricks_binary_http_error(exc, token=config.token) from None
        replay_reason = exc
    except (urllib.error.URLError, TimeoutError) as exc:
        replay_reason = exc
    except Exception as exc:
        reason = _redact_databricks_secret_text(str(exc), token=config.token)
        raise RuntimeError(
            f"Databricks Files API exclusive upload failed: {reason}"
        ) from None
    if replay_reason is not None:
        resolved_readback_opener = readback_opener
        if resolved_readback_opener is None:
            resolved_readback_opener = opener
        try:
            existing = download_databricks_volume_file_bytes(
                config,
                dbfs_uri,
                max_bytes=max_bytes,
                opener=resolved_readback_opener,
            )
        except Exception:
            if isinstance(replay_reason, urllib.error.HTTPError):
                formatted = _databricks_binary_http_error(
                    replay_reason, token=config.token
                )
                raise RuntimeError(
                    f"{formatted}; exclusive upload readback did not prove replay"
                ) from None
            reason = _redact_databricks_secret_text(
                str(replay_reason), token=config.token
            )
            raise RuntimeError(
                "Databricks Files API exclusive upload outcome was uncertain and "
                f"readback did not prove replay: {reason}"
            ) from None
        if existing != content:
            raise RuntimeError(
                "Databricks Files API exclusive upload conflicts with different "
                "existing bytes"
            ) from None
        created = False
    else:
        created = True
    return {
        "created": created,
        "dbfs_uri": dbfs_uri,
        "file_sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def upload_databricks_volume_file_path_exclusive(
    config: DatabricksWorkspaceConfig,
    dbfs_uri: str,
    local_path: str | Path,
    *,
    expected_sha256: str,
    expected_size: int,
    max_bytes: int = DATABRICKS_VOLUME_FILE_MAX_STREAM_BYTES,
    opener: DatabricksBinaryURLOpener | None = None,
    readback_opener: DatabricksBinaryURLOpener | None = None,
) -> dict[str, Any]:
    """Exclusively stream one pinned local file to a UC Volume.

    The source is opened once with ``O_NOFOLLOW`` and must be a canonical,
    same-user, single-link regular file.  Its exact bytes are hashed before the
    PUT through that descriptor, then streamed from the rewound descriptor in
    bounded chunks with an explicit ``Content-Length``.  A streaming remote
    SHA-256/size proof is mandatory both after a new 204 response and when a
    400, 409, URL, or timeout error makes the exclusive PUT outcome uncertain.
    """

    volume_path = _canonical_databricks_volume_file_path(dbfs_uri)
    _validate_databricks_volume_byte_cap(
        max_bytes,
        upper_bound=DATABRICKS_VOLUME_FILE_MAX_STREAM_BYTES,
        label="max_bytes",
    )
    pinned_sha256 = _required_databricks_sha256(
        expected_sha256,
        label="expected_sha256",
    )
    if (
        type(expected_size) is not int
        or expected_size < 0
        or expected_size > max_bytes
    ):
        raise ValueError(
            "expected_size must be a non-negative integer no greater than max_bytes"
        )
    source_path = _canonical_databricks_local_upload_path(local_path)
    file_descriptor = _open_databricks_local_upload_source(source_path)
    try:
        initial_stat = os.fstat(file_descriptor)
        identity = _databricks_local_upload_identity(initial_stat)
        _require_stable_databricks_local_upload_source(
            file_descriptor,
            source_path,
            identity,
        )
        if initial_stat.st_size != expected_size:
            raise ValueError(
                "local upload source size does not match expected_size: "
                f"{initial_stat.st_size} != {expected_size}"
            )
        observed_sha256 = _sha256_databricks_local_upload_source(
            file_descriptor,
            source_path,
            identity,
            expected_size=expected_size,
        )
        if observed_sha256 != pinned_sha256:
            raise ValueError(
                "local upload source SHA-256 does not match expected_sha256"
            )
        try:
            offset = os.lseek(file_descriptor, 0, os.SEEK_SET)
        except OSError:
            raise _DatabricksLocalUploadError(
                "Databricks Files API local upload source rewind failed"
            ) from None
        if offset != 0:
            raise _DatabricksLocalUploadError(
                "Databricks Files API local upload source rewind drifted"
            )
        _require_stable_databricks_local_upload_source(
            file_descriptor,
            source_path,
            identity,
        )

        upload_body = _DatabricksStreamingUploadBody(
            file_descriptor,
            source_path,
            identity,
            expected_size,
        )
        encoded_path = urllib.parse.quote(volume_path, safe="/-._~")
        request = urllib.request.Request(
            f"{config.normalized_host}/api/2.0/fs/files{encoded_path}"
            "?overwrite=false",
            data=cast(Any, upload_body),
            method="PUT",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "Authorization": f"Bearer {config.token}",
                "Content-Length": str(expected_size),
                "Content-Type": "application/octet-stream",
            },
        )
        resolved_opener = (
            _databricks_no_redirect_urlopen
            if opener is None
            else opener
        )
        replay_reason: BaseException | None = None
        try:
            with resolved_opener(
                request, timeout=config.timeout_seconds
            ) as response:
                status = getattr(response, "status", None)
                if type(status) is int and status in {400, 409}:
                    replay_reason = RuntimeError(
                        "Databricks Files API exclusive path upload returned "
                        f"HTTP {status}"
                    )
                elif type(status) is not int or status != 204:
                    raise RuntimeError(
                        "Databricks Files API exclusive path upload returned "
                        f"unexpected HTTP status {status!r}"
                    )
                else:
                    _require_empty_databricks_binary_response(
                        response,
                        label="Databricks Files API exclusive path upload",
                    )
        except _DatabricksLocalUploadError:
            raise
        except urllib.error.HTTPError as exc:
            if exc.code not in {400, 409}:
                raise _databricks_binary_http_error(
                    exc, token=config.token
                ) from None
            replay_reason = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            replay_reason = exc
        except Exception as exc:
            reason = _redact_databricks_secret_text(
                str(exc),
                token=config.token,
            )
            raise RuntimeError(
                "Databricks Files API exclusive path upload failed: "
                f"{reason}"
            ) from None
        if replay_reason is None:
            if not upload_body.complete:
                raise RuntimeError(
                    "Databricks Files API exclusive path upload did not consume "
                    "the complete verified source"
                )
            if upload_body.sha256 != pinned_sha256:
                raise _DatabricksLocalUploadError(
                    "Databricks Files API local upload stream SHA-256 drifted"
                )
        _require_stable_databricks_local_upload_source(
            file_descriptor,
            source_path,
            identity,
        )

        resolved_readback_opener = readback_opener
        if resolved_readback_opener is None:
            resolved_readback_opener = opener
        try:
            readback = stream_databricks_volume_file_sha256(
                config,
                dbfs_uri,
                max_bytes=max(1, expected_size),
                opener=resolved_readback_opener,
            )
        except Exception:
            if isinstance(replay_reason, urllib.error.HTTPError):
                formatted = _databricks_binary_http_error(
                    replay_reason,
                    token=config.token,
                )
                raise RuntimeError(
                    f"{formatted}; exclusive path upload readback did not prove "
                    "replay"
                ) from None
            if replay_reason is not None:
                reason = _redact_databricks_secret_text(
                    str(replay_reason),
                    token=config.token,
                )
                raise RuntimeError(
                    "Databricks Files API exclusive path upload outcome was "
                    "uncertain and readback did not prove replay: "
                    f"{reason}"
                ) from None
            raise RuntimeError(
                "Databricks Files API exclusive path upload post-PUT readback "
                "did not prove the remote file"
            ) from None
        if readback != {
            "dbfs_uri": dbfs_uri,
            "file_sha256": pinned_sha256,
            "size_bytes": expected_size,
        }:
            if replay_reason is None:
                raise RuntimeError(
                    "Databricks Files API exclusive path upload post-PUT readback "
                    "does not match the verified local file"
                )
            raise RuntimeError(
                "Databricks Files API exclusive path upload conflicts with a "
                "different remote file"
            ) from None
        _require_stable_databricks_local_upload_source(
            file_descriptor,
            source_path,
            identity,
        )
        return {
            "created": replay_reason is None,
            "dbfs_uri": dbfs_uri,
            "file_sha256": pinned_sha256,
            "size_bytes": expected_size,
        }
    finally:
        os.close(file_descriptor)


def create_databricks_volume_directory_idempotent(
    config: DatabricksWorkspaceConfig,
    dbfs_uri: str,
    *,
    opener: DatabricksBinaryURLOpener | None = None,
    proof_opener: DatabricksBinaryURLOpener | None = None,
) -> dict[str, Any]:
    """Create a canonical UC Volume directory, proving uncertain outcomes."""

    directory_path = _canonical_databricks_volume_directory_path(dbfs_uri)
    encoded_path = urllib.parse.quote(directory_path, safe="/-._~")
    request = urllib.request.Request(
        f"{config.normalized_host}/api/2.0/fs/directories{encoded_path}",
        method="PUT",
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Authorization": f"Bearer {config.token}",
            "Content-Length": "0",
        },
    )
    resolved_opener = (
        _databricks_no_redirect_urlopen
        if opener is None
        else opener
    )
    uncertain_reason: BaseException | None = None
    try:
        with resolved_opener(request, timeout=config.timeout_seconds) as response:
            status = getattr(response, "status", None)
            if type(status) is int and status in {400, 409}:
                uncertain_reason = RuntimeError(
                    "Databricks Files API directory create returned "
                    f"HTTP {status}"
                )
            elif type(status) is not int or status != 204:
                raise RuntimeError(
                    "Databricks Files API directory create returned unexpected "
                    f"HTTP status {status!r}"
                )
            else:
                _require_empty_databricks_binary_response(
                    response,
                    label="Databricks Files API directory create",
                )
    except urllib.error.HTTPError as exc:
        if exc.code not in {400, 409}:
            raise _databricks_binary_http_error(exc, token=config.token) from None
        uncertain_reason = exc
    except (urllib.error.URLError, TimeoutError) as exc:
        uncertain_reason = exc
    except Exception as exc:
        reason = _redact_databricks_secret_text(str(exc), token=config.token)
        raise RuntimeError(
            f"Databricks Files API directory create failed: {reason}"
        ) from None
    if uncertain_reason is not None:
        resolved_proof_opener = proof_opener
        if resolved_proof_opener is None:
            resolved_proof_opener = opener
        try:
            _prove_databricks_volume_directory_exists(
                config,
                directory_path,
                opener=resolved_proof_opener,
            )
        except Exception:
            if isinstance(uncertain_reason, urllib.error.HTTPError):
                formatted = _databricks_binary_http_error(
                    uncertain_reason,
                    token=config.token,
                )
                raise RuntimeError(
                    f"{formatted}; directory existence proof failed"
                ) from None
            reason = _redact_databricks_secret_text(
                str(uncertain_reason),
                token=config.token,
            )
            raise RuntimeError(
                "Databricks Files API directory create outcome was uncertain and "
                f"existence proof failed: {reason}"
            ) from None
    return {"dbfs_uri": dbfs_uri}


def list_databricks_volume_directory(
    config: DatabricksWorkspaceConfig,
    dbfs_uri: str,
    *,
    max_entries: int = DATABRICKS_VOLUME_DIRECTORY_MAX_ENTRIES,
    opener: DatabricksBinaryURLOpener | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return one complete, bounded UC Volume directory listing.

    Pagination terminates only when ``next_page_token`` is absent.  Every
    returned entry must be one canonical direct child of *dbfs_uri*; duplicate
    paths, token cycles, malformed metadata, and listings above the controller
    cap fail closed.  This is metadata-only transport: callers must use
    :func:`download_databricks_volume_file_bytes` for explicitly bounded files.
    """

    directory_path = _canonical_databricks_volume_directory_path(dbfs_uri)
    if (
        type(max_entries) is not int
        or max_entries < 0
        or max_entries > DATABRICKS_VOLUME_DIRECTORY_MAX_ENTRIES
    ):
        raise ValueError(
            "max_entries must be a non-negative integer no greater than "
            f"{DATABRICKS_VOLUME_DIRECTORY_MAX_ENTRIES}"
        )
    resolved_opener = (
        _databricks_no_redirect_urlopen
        if opener is None
        else opener
    )
    encoded_path = urllib.parse.quote(directory_path, safe="/-._~")
    entries: dict[str, dict[str, Any]] = {}
    seen_tokens: set[str] = set()
    page_token: str | None = None
    page_count = 0
    while True:
        page_count += 1
        if page_count > DATABRICKS_VOLUME_DIRECTORY_MAX_PAGES:
            raise RuntimeError(
                "Databricks Files API directory listing exceeds the controller "
                f"page cap: more than {DATABRICKS_VOLUME_DIRECTORY_MAX_PAGES} pages"
            )
        query = {"page_size": "1000"}
        if page_token is not None:
            query["page_token"] = page_token
        request = urllib.request.Request(
            f"{config.normalized_host}/api/2.0/fs/directories{encoded_path}?"
            f"{urllib.parse.urlencode(query)}",
            method="GET",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {config.token}",
            },
        )
        raw_page = _bounded_databricks_binary_response(
            config,
            request,
            max_bytes=DATABRICKS_VOLUME_FILE_MAX_DOWNLOAD_BYTES,
            opener=resolved_opener,
            label="Databricks Files API directory listing",
        )
        try:
            page = json.loads(raw_page)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RuntimeError(
                "Databricks Files API directory listing was not valid UTF-8 JSON"
            ) from None
        if not isinstance(page, dict) or set(page) - {
            "contents",
            "next_page_token",
        }:
            raise RuntimeError("Databricks Files API directory listing schema drift")
        if page == {} and page_token is None and not entries:
            contents: object = []
        elif "contents" not in page:
            raise RuntimeError("Databricks Files API directory listing schema drift")
        else:
            contents = page["contents"]
        if not isinstance(contents, list):
            raise RuntimeError(
                "Databricks Files API directory listing contents must be an array"
            )
        for raw_entry in contents:
            entry = _validated_databricks_volume_directory_entry(
                raw_entry,
                parent_path=directory_path,
            )
            entry_path = cast(str, entry["path"])
            if entry_path in entries:
                raise RuntimeError(
                    "Databricks Files API directory listing contains duplicate paths"
                )
            entries[entry_path] = entry
            if len(entries) > max_entries:
                raise RuntimeError(
                    "Databricks Files API directory listing exceeds the controller "
                    f"entry cap: more than {max_entries} entries"
                )
        next_token = page.get("next_page_token")
        if next_token is None:
            break
        if not isinstance(next_token, str) or not next_token:
            raise RuntimeError(
                "Databricks Files API directory next_page_token is invalid"
            )
        if (
            len(next_token.encode("utf-8"))
            > DATABRICKS_VOLUME_DIRECTORY_MAX_PAGE_TOKEN_BYTES
        ):
            raise RuntimeError(
                "Databricks Files API directory next_page_token exceeds the "
                "controller byte cap"
            )
        if next_token in seen_tokens:
            raise RuntimeError(
                "Databricks Files API directory pagination token repeated"
            )
        seen_tokens.add(next_token)
        page_token = next_token
    return tuple(entries[path] for path in sorted(entries))


def put_databricks_dbfs_file(
    config: DatabricksWorkspaceConfig,
    local_path: str | Path,
    dbfs_path: str,
    *,
    overwrite: bool = False,
    opener: DatabricksURLOpener = _databricks_no_redirect_urlopen,
) -> dict[str, Any]:
    response, _metadata = _put_databricks_dbfs_file_response_and_metadata(
        config,
        local_path,
        dbfs_path,
        overwrite=overwrite,
        opener=opener,
    )
    return response


def _put_databricks_dbfs_file_record(
    config: DatabricksWorkspaceConfig,
    local_path: str | Path,
    dbfs_path: str,
    *,
    overwrite: bool = False,
    opener: DatabricksURLOpener = _databricks_no_redirect_urlopen,
) -> dict[str, Any]:
    response, metadata = _put_databricks_dbfs_file_response_and_metadata(
        config,
        local_path,
        dbfs_path,
        overwrite=overwrite,
        opener=opener,
    )
    result = _success_record("put-dbfs-file", response)
    result["artifact"] = metadata
    return result


def plan_databricks_stage_and_submit(
    payload: dict[str, Any],
    artifacts: Sequence[tuple[str | Path, str]],
    *,
    overwrite: bool = False,
    require_payload_dbfs_artifacts: bool = False,
    require_payload_staged_dbfs_artifacts: bool = False,
    submit_payload_path: str | None = None,
) -> dict[str, Any]:
    prepared_artifacts = _prepare_databricks_stage_artifacts(
        payload,
        artifacts,
        overwrite=overwrite,
        require_payload_dbfs_artifacts=require_payload_dbfs_artifacts,
        require_payload_staged_dbfs_artifacts=require_payload_staged_dbfs_artifacts,
    )
    result = _success_record("stage-and-submit-plan")
    result["artifact_uploads"] = [
        _stage_and_submit_artifact_plan_record(upload_payload, metadata)
        for upload_payload, metadata in prepared_artifacts
    ]
    result["submit_payload"] = summarize_databricks_run_submit_payload(
        payload,
        source_path=submit_payload_path,
    )
    return result


def stage_and_submit_databricks_run(
    config: DatabricksWorkspaceConfig,
    payload: dict[str, Any],
    artifacts: Sequence[tuple[str | Path, str]],
    *,
    overwrite: bool = False,
    require_payload_dbfs_artifacts: bool = False,
    require_payload_staged_dbfs_artifacts: bool = False,
    preflight_auth_check: bool = False,
    opener: DatabricksURLOpener = _databricks_no_redirect_urlopen,
) -> dict[str, Any]:
    prepared_artifacts = _prepare_databricks_stage_artifacts(
        payload,
        artifacts,
        overwrite=overwrite,
        require_payload_dbfs_artifacts=require_payload_dbfs_artifacts,
        require_payload_staged_dbfs_artifacts=require_payload_staged_dbfs_artifacts,
    )
    auth_record = (
        check_databricks_auth(config, opener=opener) if preflight_auth_check else None
    )
    artifact_uploads = [
        _put_prepared_databricks_dbfs_file_record(
            config,
            upload_payload,
            metadata,
            opener=opener,
        )
        for upload_payload, metadata in prepared_artifacts
    ]
    response = submit_databricks_run(config, payload, opener=opener)
    result = _success_record("stage-and-submit", response)
    if auth_record is not None:
        result["auth"] = auth_record
    result["artifact_uploads"] = [
        _stage_and_submit_artifact_upload_record(record) for record in artifact_uploads
    ]
    return result


def write_databricks_run_response_json(
    response: dict[str, Any], path: str | Path
) -> None:
    Path(path).write_text(
        json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_databricks_run_submit_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Databricks run-submit payload must be a JSON object")
    return payload


def summarize_databricks_run(
    run: dict[str, Any],
    *,
    submit_payload: Mapping[str, Any] | None = None,
    submit_payload_path: str | None = None,
) -> dict[str, Any]:
    state = _mapping(run.get("state"))
    life_cycle_state = _optional_str(state.get("life_cycle_state"))
    result_state = _optional_str(state.get("result_state"))
    tasks = tuple(
        _task_summary(task) for task in _sequence_of_mappings(run.get("tasks"))
    )
    summary = {
        "record_type": DATABRICKS_RUN_STATUS_RECORD_TYPE,
        "run_id": run.get("run_id"),
        "run_name": run.get("run_name"),
        "run_page_url": run.get("run_page_url"),
        "life_cycle_state": life_cycle_state,
        "result_state": result_state,
        "state_message": state.get("state_message"),
        "start_time": run.get("start_time"),
        "end_time": run.get("end_time"),
        "terminal": life_cycle_state in DATABRICKS_TERMINAL_LIFE_CYCLE_STATES,
        "succeeded": life_cycle_state == "TERMINATED" and result_state == "SUCCESS",
        "active_task_key": _active_task_key(tasks),
        "task_count": len(tasks),
        "tasks": list(tasks),
        "cluster_id": _cluster_id(run),
    }
    if submit_payload is not None:
        summary["submit_payload"] = summarize_databricks_run_submit_payload(
            submit_payload,
            source_path=submit_payload_path,
        )
    return summary


def summarize_databricks_run_submit_payload(
    payload: Mapping[str, Any],
    *,
    source_path: str | None = None,
) -> dict[str, Any]:
    tasks = tuple(_sequence_of_mappings(payload.get("tasks")))
    task_summaries = tuple(_submit_payload_task_summary(task) for task in tasks)
    node_type_ids = _sorted_unique_texts(
        summary.get("node_type_id") for summary in task_summaries
    )
    driver_node_type_ids = _sorted_unique_texts(
        summary.get("driver_node_type_id") for summary in task_summaries
    )
    hardware_targets = _hardware_targets_for_task_summaries(task_summaries)
    spark_versions = _sorted_unique_texts(
        summary.get("spark_version") for summary in task_summaries
    )
    spark_env_keys = _sorted_task_list_field_values(task_summaries, "spark_env_keys")
    data_security_modes = _sorted_unique_texts(
        summary.get("data_security_mode") for summary in task_summaries
    )
    canonical_payload = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    aws_single_node_gpu_type = (
        bool(task_summaries)
        and all(
            _is_supported_aws_single_node_gpu_type(summary.get("node_type_id"))
            for summary in task_summaries
        )
        and all(
            _is_supported_aws_single_node_gpu_type(summary.get("driver_node_type_id"))
            for summary in task_summaries
        )
    )
    return {
        "record_type": DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE,
        "source_path": source_path,
        "sha256": _sha256_hex(canonical_payload),
        "run_name": payload.get("run_name"),
        "task_count": len(task_summaries),
        "task_keys": [
            summary["task_key"]
            for summary in task_summaries
            if isinstance(summary.get("task_key"), str) and summary["task_key"]
        ],
        "tasks": list(task_summaries),
        "node_type_ids": node_type_ids,
        "driver_node_type_ids": driver_node_type_ids,
        "hardware_targets": hardware_targets,
        "spark_versions": spark_versions,
        "spark_env_keys": spark_env_keys,
        "data_security_modes": data_security_modes,
        "single_node": bool(task_summaries)
        and all(summary["single_node"] for summary in task_summaries),
        _DATABRICKS_GPU_TYPE_FIELD: aws_single_node_gpu_type,
        _LEGACY_DATABRICKS_GPU_TYPE_FIELD: aws_single_node_gpu_type,
    }


def databricks_run_status_record(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the inner Databricks run-status record from a direct or CLI wrapper sidecar."""

    if record.get("record_type") == DATABRICKS_RUN_STATUS_RECORD_TYPE:
        return record
    summary = record.get("summary")
    if (
        record.get("ok") is True
        and summary is not None
        and isinstance(summary, Mapping)
        and summary.get("record_type") == DATABRICKS_RUN_STATUS_RECORD_TYPE
    ):
        return summary
    return None


def databricks_run_status_sidecar_issues(
    record: Mapping[str, Any],
    *,
    expected_hardware_target: str | None = None,
    expected_node_type_id: str | None = None,
) -> tuple[str, ...]:
    """Return release-oriented issues for a Databricks run-status sidecar."""

    status_record = databricks_run_status_record(record)
    issues: list[str] = []
    if "response" in record:
        issues.append(
            "Databricks run status sidecar must not include the raw Jobs API response"
        )
    issues.extend(_databricks_run_status_container_key_issues(record))
    issues.extend(_databricks_run_status_wrapper_field_issues(record))
    if status_record is None:
        issues.append(
            "Databricks run status sidecar must be a status record or databricks_runs get --summary output"
        )
        return _dedupe_strings(issues)
    issues.extend(
        _unexpected_keys(
            status_record,
            _DATABRICKS_RUN_STATUS_KEYS,
            "Databricks run status sidecar summary",
        )
    )
    issues.extend(_databricks_run_status_field_issues(status_record))
    if status_record.get("record_type") != DATABRICKS_RUN_STATUS_RECORD_TYPE:
        issues.append(
            f"Databricks run status sidecar record_type must be {DATABRICKS_RUN_STATUS_RECORD_TYPE!r}"
        )
    if status_record.get("terminal") is not True:
        issues.append("Databricks run status sidecar terminal must be true")
    if status_record.get("succeeded") is not True:
        issues.append("Databricks run status sidecar succeeded must be true")
    if status_record.get("life_cycle_state") != "TERMINATED":
        issues.append(
            "Databricks run status sidecar life_cycle_state must be 'TERMINATED'"
        )
    if status_record.get("result_state") != "SUCCESS":
        issues.append("Databricks run status sidecar result_state must be 'SUCCESS'")
    if (
        status_record.get("terminal") is True
        and status_record.get("succeeded") is True
        and status_record.get("active_task_key") is not None
    ):
        issues.append(
            "Databricks run status sidecar active_task_key must be null for successful terminal runs"
        )
    run_id = status_record.get("run_id")
    if not (
        (type(run_id) is int and run_id >= 0) or (isinstance(run_id, str) and run_id)
    ):
        issues.append(
            "Databricks run status sidecar run_id must be a non-negative integer or non-empty string"
        )
    task_count = status_record.get("task_count")
    tasks = status_record.get("tasks")
    if type(task_count) is not int or task_count <= 0:
        issues.append(
            "Databricks run status sidecar task_count must be a positive integer"
        )
    if (
        not isinstance(tasks, Sequence)
        or isinstance(tasks, (str, bytes, bytearray))
        or not tasks
    ):
        issues.append("Databricks run status sidecar tasks must be a non-empty array")
    else:
        if type(task_count) is int and task_count > 0 and len(tasks) != task_count:
            issues.append(
                "Databricks run status sidecar task_count must match tasks length"
            )
        issues.extend(
            _databricks_run_status_task_issues(
                tasks,
                expected_hardware_target=expected_hardware_target,
                expected_node_type_id=expected_node_type_id,
            )
        )
    submit_payload = status_record.get("submit_payload")
    if not isinstance(submit_payload, Mapping):
        issues.append("Databricks run status sidecar submit_payload must be an object")
    else:
        issues.extend(
            _databricks_submit_payload_sidecar_issues(
                submit_payload,
                tasks=tasks,
                expected_hardware_target=expected_hardware_target,
                expected_node_type_id=expected_node_type_id,
            )
        )
        issues.extend(
            _databricks_run_submit_payload_identity_issues(
                status_record, submit_payload
            )
        )
        issues.extend(
            _databricks_run_submit_payload_spark_env_identity_issues(
                tasks, submit_payload
            )
        )
    return _dedupe_strings(issues)


def validate_databricks_run_status_sidecar(
    record: Mapping[str, Any],
    *,
    expected_hardware_target: str | None = None,
    expected_node_type_id: str | None = None,
) -> None:
    """Validate a release-oriented Databricks run-status sidecar."""

    issues = databricks_run_status_sidecar_issues(
        record,
        expected_hardware_target=expected_hardware_target,
        expected_node_type_id=expected_node_type_id,
    )
    if issues:
        raise ValueError("; ".join(issues))


def _databricks_api_json(
    config: DatabricksWorkspaceConfig,
    method: str,
    path_and_query: str,
    *,
    opener: DatabricksURLOpener,
    payload: dict[str, Any] | None = None,
    payload_json_bytes: bytes | None = None,
) -> dict[str, Any]:
    parsed, _status = _databricks_api_response_json(
        config,
        method,
        path_and_query,
        opener=opener,
        payload=payload,
        payload_json_bytes=payload_json_bytes,
    )
    return parsed


def _databricks_api_response_json(
    config: DatabricksWorkspaceConfig,
    method: str,
    path_and_query: str,
    *,
    opener: DatabricksURLOpener,
    payload: dict[str, Any] | None = None,
    payload_json_bytes: bytes | None = None,
) -> tuple[dict[str, Any], int | None]:
    request = _databricks_request(
        config,
        method,
        path_and_query,
        payload=payload,
        payload_json_bytes=payload_json_bytes,
    )
    try:
        with opener(request, timeout=config.timeout_seconds) as response:
            status = getattr(response, "status", None)
            if type(status) is not int or status != 200:
                raise RuntimeError(
                    "Databricks API response returned unexpected HTTP status "
                    f"{status!r}"
                )
            raw_body = _read_databricks_response_bytes_bounded(
                response,
                max_bytes=DATABRICKS_API_PAGE_MAX_BYTES,
                label="Databricks API response",
            )
    except urllib.error.HTTPError as exc:
        raise _databricks_binary_http_error(exc, token=config.token) from None
    except urllib.error.URLError as exc:
        reason = _redact_databricks_secret_text(str(exc.reason), token=config.token)
        raise RuntimeError(f"Databricks request failed: {reason}") from None
    except TimeoutError as exc:
        reason = _redact_databricks_secret_text(str(exc), token=config.token)
        raise TimeoutError(reason) from None
    except ConnectionError as exc:
        reason = _redact_databricks_secret_text(str(exc), token=config.token)
        raise ConnectionError(reason) from None
    except Exception as exc:
        reason = _redact_databricks_secret_text(str(exc), token=config.token)
        raise RuntimeError(f"Databricks API response failed: {reason}") from None
    try:
        body = raw_body.decode("utf-8")
    except UnicodeDecodeError:
        raise RuntimeError("Databricks response was not valid UTF-8") from None
    try:
        parsed = json.loads(body) if body else {}
    except json.JSONDecodeError:
        raise RuntimeError("Databricks response was not valid JSON") from None
    if not isinstance(parsed, dict):
        raise RuntimeError("Databricks response JSON must be an object")
    return parsed, status


def _task_summary(task: Mapping[str, Any]) -> dict[str, Any]:
    state = _mapping(task.get("state"))
    life_cycle_state = _optional_str(state.get("life_cycle_state"))
    summary = {
        "task_key": task.get("task_key"),
        "run_id": task.get("run_id"),
        "life_cycle_state": life_cycle_state,
        "result_state": state.get("result_state"),
        "state_message": state.get("state_message"),
        "cluster_id": _cluster_id(task),
        "start_time": task.get("start_time"),
        "end_time": task.get("end_time"),
        "spark_env_keys": _launch_cluster_spark_env_keys(task),
    }
    launch_cluster = _launch_cluster(task)
    for field_name in ("node_type_id", "driver_node_type_id"):
        value = _optional_str(launch_cluster.get(field_name))
        if value is not None:
            summary[field_name] = value
    return summary


def _submit_payload_task_summary(task: Mapping[str, Any]) -> dict[str, Any]:
    cluster = _mapping(task.get("new_cluster"))
    custom_tags = _mapping(cluster.get("custom_tags"))
    spark_env_vars = _mapping(cluster.get("spark_env_vars"))
    return {
        "task_key": task.get("task_key"),
        "node_type_id": _optional_str(cluster.get("node_type_id")),
        "driver_node_type_id": _optional_str(cluster.get("driver_node_type_id")),
        "spark_version": _optional_str(cluster.get("spark_version")),
        "spark_env_keys": _spark_env_key_names(spark_env_vars),
        "data_security_mode": _optional_str(cluster.get("data_security_mode")),
        "num_workers": cluster.get("num_workers"),
        "single_node": cluster.get("num_workers") == 0
        and custom_tags.get("ResourceClass") == "SingleNode",
        "purpose": _optional_str(custom_tags.get("purpose")),
    }


def _active_task_key(tasks: Sequence[Mapping[str, Any]]) -> str | None:
    for task in tasks:
        if task.get("life_cycle_state") not in DATABRICKS_TERMINAL_LIFE_CYCLE_STATES:
            task_key = task.get("task_key")
            return task_key if isinstance(task_key, str) and task_key else None
    return None


def _cluster_id(record: Mapping[str, Any]) -> str | None:
    cluster_instance = _mapping(record.get("cluster_instance"))
    cluster_id = cluster_instance.get("cluster_id")
    return cluster_id if isinstance(cluster_id, str) and cluster_id else None


def _launch_cluster_spark_env_keys(record: Mapping[str, Any]) -> list[str]:
    return _spark_env_key_names(_mapping(_launch_cluster(record).get("spark_env_vars")))


def _launch_cluster(record: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = _mapping(record.get("new_cluster"))
    if direct:
        return direct
    cluster_spec = _mapping(record.get("cluster_spec"))
    return _mapping(cluster_spec.get("new_cluster"))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence_of_mappings(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _sorted_unique_texts(values: Sequence[Any]) -> list[str]:
    return sorted({value for value in values if isinstance(value, str) and value})


_SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES = SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES
_HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES = (
    HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES
)
_V1_HARDWARE_TARGET_PREFIXES = tuple(
    (profile.hardware_target, profile.databricks_node_type_prefixes)
    for profile in V1_HARDWARE_TARGET_PROFILES
)


def _is_supported_aws_single_node_gpu_type(value: Any) -> bool:
    return isinstance(value, str) and value.lower().startswith(
        _SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES
    )


def _is_expected_aws_single_node_gpu_type(
    value: Any, expected_hardware_target: str | None
) -> bool:
    if expected_hardware_target is None:
        return _is_supported_aws_single_node_gpu_type(value)
    prefixes = _HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES.get(
        expected_hardware_target
    )
    return (
        isinstance(value, str)
        and prefixes is not None
        and value.lower().startswith(prefixes)
    )


def _hardware_target_for_node_type(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.lower()
    for hardware_target, prefixes in _V1_HARDWARE_TARGET_PREFIXES:
        if lowered.startswith(prefixes):
            return hardware_target
    return None


def _hardware_targets_for_task_summaries(tasks: Sequence[Any]) -> list[str]:
    hardware_targets = {
        hardware_target
        for task in tasks
        if isinstance(task, Mapping)
        for field_name in ("node_type_id", "driver_node_type_id")
        for hardware_target in (_hardware_target_for_node_type(task.get(field_name)),)
        if hardware_target is not None
    }
    return sorted(hardware_targets)


def _submit_payload_gpu_type_supported(record: Mapping[str, Any]) -> bool:
    return all(
        record[field_name] is True for field_name in _present_gpu_type_fields(record)
    )


def _present_gpu_type_fields(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        field_name for field_name in _DATABRICKS_GPU_TYPE_FIELDS if field_name in record
    )


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _databricks_run_status_task_issues(
    tasks: Sequence[Any],
    *,
    expected_hardware_target: str | None,
    expected_node_type_id: str | None,
) -> tuple[str, ...]:
    issues: list[str] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            issues.append(
                f"Databricks run status sidecar tasks[{index}] must be an object"
            )
            continue
        issues.extend(
            _unexpected_keys(
                task,
                _DATABRICKS_RUN_STATUS_TASK_KEYS,
                f"Databricks run status sidecar tasks[{index}]",
            )
        )
        issues.extend(_databricks_run_status_task_field_issues(task, index=index))
        issues.extend(
            _list_of_strings_field(
                task, "spark_env_keys", f"Databricks run status sidecar tasks[{index}]"
            )
        )
        spark_env_keys = _valid_string_list(task.get("spark_env_keys"))
        if spark_env_keys is not None:
            issues.extend(
                _spark_env_key_issues(
                    spark_env_keys, f"Databricks run status sidecar tasks[{index}]"
                )
            )
        if not isinstance(task.get("task_key"), str) or not task["task_key"]:
            issues.append(
                f"Databricks run status sidecar tasks[{index}].task_key must be non-empty"
            )
        issues.extend(
            _databricks_run_status_task_node_type_issues(
                task,
                index=index,
                expected_hardware_target=expected_hardware_target,
                expected_node_type_id=expected_node_type_id,
            )
        )
        if task.get("life_cycle_state") != "TERMINATED":
            issues.append(
                f"Databricks run status sidecar tasks[{index}].life_cycle_state must be 'TERMINATED'"
            )
        if task.get("result_state") != "SUCCESS":
            issues.append(
                f"Databricks run status sidecar tasks[{index}].result_state must be 'SUCCESS'"
            )
    return tuple(issues)


def _databricks_run_status_task_node_type_issues(
    task: Mapping[str, Any],
    *,
    index: int,
    expected_hardware_target: str | None,
    expected_node_type_id: str | None,
) -> tuple[str, ...]:
    issues: list[str] = []
    for field_name in ("node_type_id", "driver_node_type_id"):
        value = task.get(field_name)
        if value is None:
            if expected_node_type_id is not None:
                issues.append(
                    f"Databricks run status sidecar tasks[{index}].{field_name} must be present for "
                    f"node_type_id {expected_node_type_id!r} validation"
                )
            continue
        if not _is_supported_aws_single_node_gpu_type(value):
            issues.append(
                f"Databricks run status sidecar tasks[{index}].{field_name} "
                "must be a supported V1 AWS GPU node type"
            )
        elif not _is_expected_aws_single_node_gpu_type(value, expected_hardware_target):
            issues.append(
                f"Databricks run status sidecar tasks[{index}].{field_name} must match "
                f"hardware_target {expected_hardware_target!r}"
            )
        elif expected_node_type_id is not None and value != expected_node_type_id:
            issues.append(
                f"Databricks run status sidecar tasks[{index}].{field_name} must be "
                f"node_type_id {expected_node_type_id!r}"
            )
    return tuple(issues)


def _databricks_submit_payload_sidecar_issues(
    record: Mapping[str, Any],
    *,
    tasks: Any,
    expected_hardware_target: str | None,
    expected_node_type_id: str | None,
) -> tuple[str, ...]:
    issues: list[str] = []
    issues.extend(
        _unexpected_keys(
            record,
            _DATABRICKS_SUBMIT_PAYLOAD_KEYS,
            "Databricks run status sidecar submit_payload",
        )
    )
    issues.extend(_databricks_submit_payload_field_issues(record))
    if record.get("record_type") != DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE:
        issues.append(
            f"Databricks run status sidecar submit_payload.record_type must be {DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE!r}"
        )
    source_path = record.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        issues.append(
            "Databricks run status sidecar submit_payload.source_path must be non-empty"
        )
    if not isinstance(record.get("sha256"), str) or not _SHA256_HEX_RE.fullmatch(
        record["sha256"]
    ):
        issues.append(
            "Databricks run status sidecar submit_payload.sha256 must be a 64-character lowercase hex digest"
        )
    if record.get("single_node") is not True:
        issues.append(
            "Databricks run status sidecar submit_payload.single_node must be true"
        )
    if _submit_payload_gpu_type_supported(record) is not True:
        issues.append(
            "Databricks run status sidecar submit_payload.aws_single_node_gpu_type must be true"
        )
    task_count = record.get("task_count")
    payload_tasks = record.get("tasks")
    if type(task_count) is not int or task_count <= 0:
        issues.append(
            "Databricks run status sidecar submit_payload.task_count must be a positive integer"
        )
    if (
        not isinstance(payload_tasks, Sequence)
        or isinstance(payload_tasks, (str, bytes, bytearray))
        or not payload_tasks
    ):
        issues.append(
            "Databricks run status sidecar submit_payload.tasks must be a non-empty array"
        )
    else:
        if (
            type(task_count) is int
            and task_count > 0
            and len(payload_tasks) != task_count
        ):
            issues.append(
                "Databricks run status sidecar submit_payload.task_count must match tasks length"
            )
        issues.extend(
            _databricks_submit_payload_task_issues(
                payload_tasks,
                expected_hardware_target=expected_hardware_target,
                expected_node_type_id=expected_node_type_id,
            )
        )
        issues.extend(
            _databricks_submit_payload_summary_field_issues(record, payload_tasks)
        )
    data_security_modes = record.get("data_security_modes")
    if not isinstance(data_security_modes, Sequence) or isinstance(
        data_security_modes, (str, bytes, bytearray)
    ):
        issues.append(
            "Databricks run status sidecar submit_payload.data_security_modes must be an array"
        )
    elif "SINGLE_USER" not in data_security_modes:
        issues.append(
            "Databricks run status sidecar submit_payload.data_security_modes must include SINGLE_USER"
        )
    if isinstance(tasks, Sequence) and not isinstance(tasks, (str, bytes, bytearray)):
        status_task_keys = _task_key_list(tasks)
        payload_task_keys = _task_key_list(record.get("tasks"))
        if not status_task_keys:
            issues.append("Databricks run status sidecar tasks must include task keys")
        if not payload_task_keys:
            issues.append(
                "Databricks run status sidecar submit_payload.tasks must include task keys"
            )
        if (
            status_task_keys
            and payload_task_keys
            and status_task_keys != payload_task_keys
        ):
            issues.append(
                "Databricks run status sidecar submit_payload.task_keys must match status task keys"
            )
    if isinstance(record.get("task_keys"), Sequence) and not isinstance(
        record.get("task_keys"), (str, bytes, bytearray)
    ):
        declared_task_keys = [
            key for key in record["task_keys"] if isinstance(key, str) and key
        ]
        payload_task_keys = _task_key_list(record.get("tasks"))
        if declared_task_keys != payload_task_keys:
            issues.append(
                "Databricks run status sidecar submit_payload.task_keys must match submit_payload.tasks"
            )
    return tuple(issues)


def _databricks_run_submit_payload_identity_issues(
    status_record: Mapping[str, Any],
    submit_payload: Mapping[str, Any],
) -> tuple[str, ...]:
    run_name = status_record.get("run_name")
    submit_run_name = submit_payload.get("run_name")
    if (
        isinstance(run_name, str)
        and run_name
        and isinstance(submit_run_name, str)
        and submit_run_name
        and submit_run_name != run_name
    ):
        return (
            "Databricks run status sidecar submit_payload.run_name must match run_name",
        )
    return ()


def _databricks_run_submit_payload_spark_env_identity_issues(
    status_tasks: Any,
    submit_payload: Mapping[str, Any],
) -> tuple[str, ...]:
    payload_tasks = submit_payload.get("tasks")
    if not isinstance(status_tasks, Sequence) or isinstance(
        status_tasks, (str, bytes, bytearray)
    ):
        return ()
    if not isinstance(payload_tasks, Sequence) or isinstance(
        payload_tasks, (str, bytes, bytearray)
    ):
        return ()
    status_by_task_key = {
        task["task_key"]: task
        for task in status_tasks
        if isinstance(task, Mapping)
        and isinstance(task.get("task_key"), str)
        and task["task_key"]
    }
    issues: list[str] = []
    for payload_task in payload_tasks:
        if not isinstance(payload_task, Mapping):
            continue
        task_key = payload_task.get("task_key")
        if not isinstance(task_key, str) or not task_key:
            continue
        status_task = status_by_task_key.get(task_key)
        if status_task is None:
            continue
        payload_spark_env_keys = _valid_string_list(payload_task.get("spark_env_keys"))
        status_spark_env_keys = _valid_string_list(status_task.get("spark_env_keys"))
        if payload_spark_env_keys is None or status_spark_env_keys is None:
            continue
        if sorted(payload_spark_env_keys) != sorted(status_spark_env_keys):
            issues.append(
                "Databricks run status sidecar submit_payload.tasks "
                f"spark_env_keys must match run task {task_key!r} spark_env_keys"
            )
    return tuple(issues)


def _databricks_submit_payload_summary_field_issues(
    record: Mapping[str, Any],
    tasks: Sequence[Any],
) -> tuple[str, ...]:
    issues: list[str] = []
    for summary_field, task_field in (
        ("node_type_ids", "node_type_id"),
        ("driver_node_type_ids", "driver_node_type_id"),
        ("spark_versions", "spark_version"),
        ("data_security_modes", "data_security_mode"),
    ):
        actual_values = _valid_string_list(record.get(summary_field))
        if actual_values is None:
            continue
        expected_values = _sorted_task_field_values(tasks, task_field)
        if actual_values != expected_values:
            issues.append(
                f"Databricks run status sidecar submit_payload.{summary_field} must match submit_payload.tasks"
            )
    actual_spark_env_keys = _valid_string_list(record.get("spark_env_keys"))
    if actual_spark_env_keys is None:
        issues.append(
            "Databricks run status sidecar submit_payload.spark_env_keys must be an array of non-empty strings"
        )
    else:
        expected_spark_env_keys = _sorted_task_list_field_values(
            tasks, "spark_env_keys"
        )
        if actual_spark_env_keys != expected_spark_env_keys:
            issues.append(
                "Databricks run status sidecar submit_payload.spark_env_keys must match submit_payload.tasks"
            )
        issues.extend(
            _spark_env_key_issues(
                actual_spark_env_keys, "Databricks run status sidecar submit_payload"
            )
        )
    if "hardware_targets" in record:
        actual_hardware_targets = _valid_string_list(record.get("hardware_targets"))
    else:
        actual_hardware_targets = None
    if actual_hardware_targets is not None:
        expected_hardware_targets = _hardware_targets_for_task_summaries(tasks)
        if actual_hardware_targets != expected_hardware_targets:
            issues.append(
                "Databricks run status sidecar submit_payload.hardware_targets must match submit_payload.tasks"
            )
    return tuple(issues)


def _databricks_submit_payload_task_issues(
    tasks: Sequence[Any],
    *,
    expected_hardware_target: str | None,
    expected_node_type_id: str | None,
) -> tuple[str, ...]:
    issues: list[str] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            issues.append(
                f"Databricks run status sidecar submit_payload.tasks[{index}] must be an object"
            )
            continue
        issues.extend(
            _unexpected_keys(
                task,
                _DATABRICKS_SUBMIT_PAYLOAD_TASK_KEYS,
                f"Databricks run status sidecar submit_payload.tasks[{index}]",
            )
        )
        issues.extend(_databricks_submit_payload_task_field_issues(task, index=index))
        if not isinstance(task.get("task_key"), str) or not task["task_key"]:
            issues.append(
                f"Databricks run status sidecar submit_payload.tasks[{index}].task_key must be non-empty"
            )
        for field_name in ("node_type_id", "driver_node_type_id"):
            value = task.get(field_name)
            if not _is_supported_aws_single_node_gpu_type(value):
                issues.append(
                    f"Databricks run status sidecar submit_payload.tasks[{index}].{field_name} "
                    "must be a supported V1 AWS GPU node type"
                )
            elif not _is_expected_aws_single_node_gpu_type(
                value, expected_hardware_target
            ):
                issues.append(
                    f"Databricks run status sidecar submit_payload.tasks[{index}].{field_name} must match "
                    f"hardware_target {expected_hardware_target!r}"
                )
            elif expected_node_type_id is not None and value != expected_node_type_id:
                issues.append(
                    f"Databricks run status sidecar submit_payload.tasks[{index}].{field_name} must be "
                    f"node_type_id {expected_node_type_id!r}"
                )
        if task.get("single_node") is not True:
            issues.append(
                f"Databricks run status sidecar submit_payload.tasks[{index}].single_node must be true"
            )
        if task.get("data_security_mode") != "SINGLE_USER":
            issues.append(
                f"Databricks run status sidecar submit_payload.tasks[{index}].data_security_mode must be 'SINGLE_USER'"
            )
    return tuple(issues)


def _databricks_run_status_field_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    issues.extend(
        _required_str_field(record, "record_type", "Databricks run status sidecar")
    )
    issues.extend(
        _run_id_field_issues(record, "run_id", "Databricks run status sidecar")
    )
    for field_name in (
        "run_name",
        "run_page_url",
        "state_message",
        "active_task_key",
        "cluster_id",
    ):
        issues.extend(
            _optional_str_field(record, field_name, "Databricks run status sidecar")
        )
    for field_name in ("life_cycle_state", "result_state"):
        issues.extend(
            _required_str_field(record, field_name, "Databricks run status sidecar")
        )
    for field_name in ("start_time", "end_time"):
        issues.extend(
            _optional_int_field(record, field_name, "Databricks run status sidecar")
        )
    for field_name in ("terminal", "succeeded"):
        issues.extend(_bool_field(record, field_name, "Databricks run status sidecar"))
    return tuple(issues)


def _databricks_run_status_task_field_issues(
    task: Mapping[str, Any], *, index: int
) -> tuple[str, ...]:
    label = f"Databricks run status sidecar tasks[{index}]"
    issues: list[str] = []
    issues.extend(_required_str_field(task, "task_key", label))
    issues.extend(_run_id_field_issues(task, "run_id", label))
    for field_name in ("life_cycle_state", "result_state"):
        issues.extend(_required_str_field(task, field_name, label))
    for field_name in (
        "state_message",
        "cluster_id",
        "node_type_id",
        "driver_node_type_id",
    ):
        issues.extend(_optional_str_field(task, field_name, label))
    for field_name in ("start_time", "end_time"):
        issues.extend(_optional_int_field(task, field_name, label))
    return tuple(issues)


def _databricks_submit_payload_field_issues(
    record: Mapping[str, Any],
) -> tuple[str, ...]:
    issues: list[str] = []
    for field_name in ("record_type", "source_path"):
        issues.extend(
            _required_str_field(
                record, field_name, "Databricks run status sidecar submit_payload"
            )
        )
    issues.extend(
        _optional_str_field(
            record, "run_name", "Databricks run status sidecar submit_payload"
        )
    )
    for field_name in (
        "task_keys",
        "node_type_ids",
        "driver_node_type_ids",
        "spark_versions",
        "data_security_modes",
    ):
        issues.extend(
            _list_of_strings_field(
                record, field_name, "Databricks run status sidecar submit_payload"
            )
        )
    if "hardware_targets" in record:
        issues.extend(
            _list_of_strings_field(
                record,
                "hardware_targets",
                "Databricks run status sidecar submit_payload",
            )
        )
    issues.extend(
        _bool_field(
            record, "single_node", "Databricks run status sidecar submit_payload"
        )
    )
    if not _present_gpu_type_fields(record):
        issues.append(
            "Databricks run status sidecar submit_payload.aws_single_node_gpu_type or aws_g5_node_type must be present"
        )
    for field_name in _DATABRICKS_GPU_TYPE_FIELDS:
        if field_name in record:
            issues.extend(
                _bool_field(
                    record, field_name, "Databricks run status sidecar submit_payload"
                )
            )
    if _gpu_type_fields_contradict(record):
        issues.append(
            "Databricks run status sidecar submit_payload.aws_single_node_gpu_type and aws_g5_node_type must match"
        )
    return tuple(issues)


def _gpu_type_fields_contradict(record: Mapping[str, Any]) -> bool:
    return (
        type(record.get(_DATABRICKS_GPU_TYPE_FIELD)) is bool
        and type(record.get(_LEGACY_DATABRICKS_GPU_TYPE_FIELD)) is bool
        and record[_DATABRICKS_GPU_TYPE_FIELD]
        != record[_LEGACY_DATABRICKS_GPU_TYPE_FIELD]
    )


def _databricks_submit_payload_task_field_issues(
    task: Mapping[str, Any], *, index: int
) -> tuple[str, ...]:
    label = f"Databricks run status sidecar submit_payload.tasks[{index}]"
    issues: list[str] = []
    for field_name in (
        "task_key",
        "node_type_id",
        "driver_node_type_id",
        "spark_version",
        "data_security_mode",
        "purpose",
    ):
        issues.extend(_required_str_field(task, field_name, label))
    issues.extend(_list_of_strings_field(task, "spark_env_keys", label))
    spark_env_keys = _valid_string_list(task.get("spark_env_keys"))
    if spark_env_keys is not None:
        issues.extend(_spark_env_key_issues(spark_env_keys, label))
    if type(task.get("num_workers")) is not int:
        issues.append(f"{label}.num_workers must be an integer")
    issues.extend(_bool_field(task, "single_node", label))
    return tuple(issues)


def _databricks_run_status_container_key_issues(
    record: Mapping[str, Any],
) -> tuple[str, ...]:
    if record.get("record_type") == DATABRICKS_RUN_STATUS_RECORD_TYPE:
        return ()
    return _unexpected_keys(
        record,
        _DATABRICKS_RUN_STATUS_WRAPPER_KEYS,
        "Databricks run status sidecar wrapper",
    )


def _databricks_run_status_wrapper_field_issues(
    record: Mapping[str, Any],
) -> tuple[str, ...]:
    if record.get("record_type") == DATABRICKS_RUN_STATUS_RECORD_TYPE:
        return ()
    issues: list[str] = []
    if record.get("ok") is not True:
        issues.append("Databricks run status sidecar wrapper.ok must be true")
    if record.get("action") != "get":
        issues.append("Databricks run status sidecar wrapper.action must be 'get'")
    if not isinstance(record.get("summary"), Mapping):
        issues.append("Databricks run status sidecar wrapper.summary must be an object")
    return tuple(issues)


def _unexpected_keys(
    record: Mapping[str, Any], allowed_keys: frozenset[str], label: str
) -> tuple[str, ...]:
    unexpected = sorted(str(key) for key in record if key not in allowed_keys)
    if not unexpected:
        return ()
    return (f"{label} has unsupported keys: {unexpected}",)


def _required_str_field(
    record: Mapping[str, Any], field_name: str, label: str
) -> tuple[str, ...]:
    value = record.get(field_name)
    if isinstance(value, str) and value:
        return ()
    return (f"{label}.{field_name} must be a non-empty string",)


def _optional_str_field(
    record: Mapping[str, Any], field_name: str, label: str
) -> tuple[str, ...]:
    value = record.get(field_name)
    if value is None or isinstance(value, str):
        return ()
    return (f"{label}.{field_name} must be a string or null",)


def _optional_int_field(
    record: Mapping[str, Any], field_name: str, label: str
) -> tuple[str, ...]:
    value = record.get(field_name)
    if value is None or type(value) is int:
        return ()
    return (f"{label}.{field_name} must be an integer or null",)


def _bool_field(
    record: Mapping[str, Any], field_name: str, label: str
) -> tuple[str, ...]:
    if type(record.get(field_name)) is bool:
        return ()
    return (f"{label}.{field_name} must be boolean",)


def _run_id_field_issues(
    record: Mapping[str, Any], field_name: str, label: str
) -> tuple[str, ...]:
    value = record.get(field_name)
    if (type(value) is int and value >= 0) or (isinstance(value, str) and value):
        return ()
    return (f"{label}.{field_name} must be a non-negative integer or non-empty string",)


def _list_of_strings_field(
    record: Mapping[str, Any], field_name: str, label: str
) -> tuple[str, ...]:
    value = record.get(field_name)
    if _valid_string_list(value) is not None:
        return ()
    return (f"{label}.{field_name} must be an array of non-empty strings",)


def _valid_string_list(value: Any) -> list[str] | None:
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and all(isinstance(item, str) and item for item in value)
    ):
        return list(value)
    return None


def _sorted_task_field_values(tasks: Sequence[Any], field_name: str) -> list[str]:
    return sorted(
        {
            task[field_name]
            for task in tasks
            if isinstance(task, Mapping)
            and isinstance(task.get(field_name), str)
            and task[field_name]
        }
    )


def _sorted_task_list_field_values(tasks: Sequence[Any], field_name: str) -> list[str]:
    return sorted(
        {
            item
            for task in tasks
            if isinstance(task, Mapping)
            for values in (_valid_string_list(task.get(field_name)),)
            if values is not None
            for item in values
        }
    )


def _spark_env_key_names(spark_env_vars: Mapping[str, Any]) -> list[str]:
    return _sorted_unique_texts(
        _safe_spark_env_key_name(key) for key in spark_env_vars.keys()
    )


def _safe_spark_env_key_name(value: str) -> str:
    if isinstance(value, str) and _DATABRICKS_PAT_TOKEN_RE.search(value):
        return _REDACTED_SPARK_ENV_TOKEN_KEY
    return value


def _spark_env_key_issues(values: Sequence[str], label: str) -> tuple[str, ...]:
    issues: list[str] = []
    for value in values:
        if value == _REDACTED_SPARK_ENV_TOKEN_KEY:
            issues.append(
                f"{label}.spark_env_keys contains redacted Databricks token-pattern environment variable name"
            )
            continue
        if _DATABRICKS_PAT_TOKEN_RE.search(value):
            issues.append(
                f"{label}.spark_env_keys contains Databricks token-pattern environment variable name"
            )
            continue
        if _SPARK_ENV_VAR_KEY_RE.fullmatch(value) is None:
            issues.append(
                f"{label}.spark_env_keys contains invalid environment variable name {value!r}"
            )
        if _looks_secret_like_spark_env_key(value):
            issues.append(
                f"{label}.spark_env_keys contains secret-looking environment variable name {value!r}"
            )
    return tuple(issues)


def _looks_secret_like_spark_env_key(value: str) -> bool:
    parts = {part.upper() for part in _ENV_KEY_PART_RE.findall(value)}
    return bool(parts.intersection(_SECRET_LIKE_SPARK_ENV_KEY_PARTS))


def _task_key_list(tasks: Any) -> list[str]:
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes, bytearray)):
        return []
    return [
        task["task_key"]
        for task in tasks
        if isinstance(task, Mapping)
        and isinstance(task.get("task_key"), str)
        and task["task_key"]
    ]


def _dedupe_strings(values: Sequence[str]) -> tuple[str, ...]:
    deduped = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)


def _databricks_dbfs_put_payload(
    local_path: str | Path,
    dbfs_path: str,
    *,
    overwrite: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _databricks_dbfs_file_metadata(local_path, dbfs_path)
    raw = Path(local_path).read_bytes()
    contents = base64.b64encode(raw).decode("ascii")
    if len(contents.encode("ascii")) > DATABRICKS_DBFS_PUT_MAX_CONTENT_BYTES:
        raise ValueError(
            "Databricks DBFS put contents must be at most "
            f"{DATABRICKS_DBFS_PUT_MAX_CONTENT_BYTES} base64 bytes; "
            "stage larger files with a streaming Databricks upload mechanism."
        )
    return {
        "path": metadata["dbfs_api_path"],
        "contents": contents,
        "overwrite": bool(overwrite),
    }, metadata


def _prepare_databricks_stage_artifacts(
    payload: Mapping[str, Any],
    artifacts: Sequence[tuple[str | Path, str]],
    *,
    overwrite: bool,
    require_payload_dbfs_artifacts: bool,
    require_payload_staged_dbfs_artifacts: bool,
) -> tuple[tuple[dict[str, Any], dict[str, Any]], ...]:
    artifact_pairs = tuple(artifacts)
    if not artifact_pairs:
        raise ValueError("stage-and-submit requires at least one artifact")
    if require_payload_dbfs_artifacts:
        _validate_payload_dbfs_artifacts_are_staged(payload, artifact_pairs)
    if require_payload_staged_dbfs_artifacts:
        _validate_payload_staged_dbfs_artifacts_are_staged(payload, artifact_pairs)
    return tuple(
        _databricks_dbfs_put_payload(local_path, dbfs_path, overwrite=overwrite)
        for local_path, dbfs_path in artifact_pairs
    )


def _put_databricks_dbfs_file_response_and_metadata(
    config: DatabricksWorkspaceConfig,
    local_path: str | Path,
    dbfs_path: str,
    *,
    overwrite: bool,
    opener: DatabricksURLOpener,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, metadata = _databricks_dbfs_put_payload(
        local_path, dbfs_path, overwrite=overwrite
    )
    response = _put_prepared_databricks_dbfs_file(config, payload, opener=opener)
    return response, metadata


def _put_prepared_databricks_dbfs_file(
    config: DatabricksWorkspaceConfig,
    payload: dict[str, Any],
    *,
    opener: DatabricksURLOpener,
) -> dict[str, Any]:
    return _databricks_api_json(
        config,
        "POST",
        "/api/2.0/dbfs/put",
        payload=payload,
        opener=opener,
    )


def _put_prepared_databricks_dbfs_file_record(
    config: DatabricksWorkspaceConfig,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    *,
    opener: DatabricksURLOpener,
) -> dict[str, Any]:
    response = _put_prepared_databricks_dbfs_file(config, payload, opener=opener)
    result = _success_record("put-dbfs-file", response)
    result["artifact"] = metadata
    return result


def _databricks_dbfs_file_metadata(
    local_path: str | Path, dbfs_path: str
) -> dict[str, Any]:
    path = Path(local_path)
    if not path.is_file():
        raise ValueError(f"local_path must be an existing file: {path}")
    dbfs_api_path = _databricks_dbfs_api_path(dbfs_path)
    raw = path.read_bytes()
    return {
        "local_path": str(path),
        "dbfs_path": _canonical_dbfs_uri(dbfs_path),
        "dbfs_api_path": dbfs_api_path,
        "bytes": len(raw),
        "sha256": _sha256_hex(raw),
    }


def _databricks_dbfs_api_path(dbfs_path: str) -> str:
    api_path = dbfs_path
    if dbfs_path.startswith("dbfs:/"):
        api_path = dbfs_path[len("dbfs:") :]
    if not api_path.startswith("/") or api_path == "/" or api_path.startswith("//"):
        raise ValueError(
            "dbfs_path must be a non-empty absolute DBFS path or dbfs:/ URI"
        )
    return api_path


def _canonical_dbfs_uri(dbfs_path: str) -> str:
    return f"dbfs:{_databricks_dbfs_api_path(dbfs_path)}"


def _parse_dbfs_artifact_mapping(value: str) -> tuple[str, str]:
    local_path, separator, dbfs_path = value.partition("=")
    if not separator or not local_path or not dbfs_path:
        raise ValueError("--artifact must use LOCAL_PATH=DBFS_PATH")
    return local_path, dbfs_path


def _submit_payload_dbfs_uris(value: Any) -> tuple[str, ...]:
    found: list[str] = []
    _collect_submit_payload_dbfs_uris(value, found)
    return _dedupe_strings(found)


def _collect_submit_payload_dbfs_uris(value: Any, found: list[str]) -> None:
    if isinstance(value, str):
        if value.startswith("dbfs:/"):
            found.append(_canonical_dbfs_uri(value))
        return
    if isinstance(value, Mapping):
        for nested in value.values():
            _collect_submit_payload_dbfs_uris(nested, found)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _collect_submit_payload_dbfs_uris(nested, found)


def _validate_payload_dbfs_artifacts_are_staged(
    payload: Mapping[str, Any],
    artifacts: Sequence[tuple[str | Path, str]],
) -> None:
    staged_uris = {
        _canonical_dbfs_uri(dbfs_path) for _local_path, dbfs_path in artifacts
    }
    missing = tuple(
        uri for uri in _submit_payload_dbfs_uris(payload) if uri not in staged_uris
    )
    if missing:
        raise ValueError(
            "Databricks submit payload references DBFS URIs without staged artifacts: "
            + ", ".join(missing)
        )


def _submit_payload_staged_dbfs_uris(payload: Mapping[str, Any]) -> tuple[str, ...]:
    found: list[str] = []
    for task in _payload_task_mappings(payload):
        spark_python_task = task.get("spark_python_task")
        if isinstance(spark_python_task, Mapping):
            _collect_staged_dbfs_uri(spark_python_task.get("python_file"), found)
            _collect_staged_parameter_dbfs_uris(
                spark_python_task.get("parameters"), found
            )
        _collect_library_dbfs_uris(task.get("libraries"), found)
    return _dedupe_strings(found)


def _payload_task_mappings(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    tasks = payload.get("tasks")
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes, bytearray)):
        return ()
    return tuple(task for task in tasks if isinstance(task, Mapping))


def _collect_staged_parameter_dbfs_uris(parameters: Any, found: list[str]) -> None:
    if not isinstance(parameters, Sequence) or isinstance(
        parameters, (str, bytes, bytearray)
    ):
        return
    generated_fixture_uris = _generated_fixture_dbfs_uris(parameters)
    for index, value in enumerate(parameters[:-1]):
        next_value = parameters[index + 1]
        if value in _DATABRICKS_ALWAYS_STAGED_ARTIFACT_PARAMETER_FLAGS:
            _collect_staged_dbfs_uri(next_value, found)
        elif value in _DATABRICKS_ENGINE_PROBE_INPUT_ARTIFACT_PARAMETER_FLAGS:
            _collect_staged_dbfs_uri(next_value, found, exclude=generated_fixture_uris)


def _generated_fixture_dbfs_uris(parameters: Sequence[Any]) -> frozenset[str]:
    fixture_output_dir = _parameter_value(parameters, "--fixture-output-dir")
    if not isinstance(fixture_output_dir, str) or not fixture_output_dir.startswith(
        "dbfs:/"
    ):
        return frozenset()
    fixture_root = _canonical_dbfs_uri(fixture_output_dir).rstrip("/")
    return frozenset(
        _canonical_dbfs_uri(f"{fixture_root}/{filename}")
        for filename in DEFAULT_ENGINE_PROBE_FIXTURE_FILENAMES.values()
    )


def _parameter_value(parameters: Sequence[Any], flag: str) -> Any:
    for index, value in enumerate(parameters[:-1]):
        if value == flag:
            return parameters[index + 1]
    return None


def _collect_library_dbfs_uris(libraries: Any, found: list[str]) -> None:
    if not isinstance(libraries, Sequence) or isinstance(
        libraries, (str, bytes, bytearray)
    ):
        return
    for library in libraries:
        if not isinstance(library, Mapping):
            continue
        for key in ("whl", "jar", "egg"):
            _collect_staged_dbfs_uri(library.get(key), found)


def _collect_staged_dbfs_uri(
    value: Any, found: list[str], *, exclude: frozenset[str] = frozenset()
) -> None:
    if isinstance(value, str) and value.startswith("dbfs:/"):
        uri = _canonical_dbfs_uri(value)
        if uri not in exclude:
            found.append(uri)


def _validate_payload_staged_dbfs_artifacts_are_staged(
    payload: Mapping[str, Any],
    artifacts: Sequence[tuple[str | Path, str]],
) -> None:
    staged_uris = {
        _canonical_dbfs_uri(dbfs_path) for _local_path, dbfs_path in artifacts
    }
    missing = tuple(
        uri
        for uri in _submit_payload_staged_dbfs_uris(payload)
        if uri not in staged_uris
    )
    if missing:
        raise ValueError(
            "Databricks submit payload references staged DBFS artifacts without matching --artifact entries: "
            + ", ".join(missing)
        )


def _stage_and_submit_artifact_upload_record(
    upload_record: Mapping[str, Any],
) -> dict[str, Any]:
    record = {"artifact": upload_record.get("artifact")}
    if "response" in upload_record:
        record["response"] = upload_record["response"]
    return record


def _stage_and_submit_artifact_plan_record(
    upload_payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "artifact": dict(metadata),
        "upload_request": {
            "path": upload_payload.get("path"),
            "overwrite": upload_payload.get("overwrite"),
            "contents_base64_bytes": len(
                str(upload_payload.get("contents", "")).encode("ascii")
            ),
        },
    }


def _databricks_request(
    config: DatabricksWorkspaceConfig,
    method: str,
    path_and_query: str,
    *,
    payload: dict[str, Any] | None,
    payload_json_bytes: bytes | None = None,
) -> urllib.request.Request:
    if payload is not None and payload_json_bytes is not None:
        raise ValueError("payload and payload_json_bytes are mutually exclusive")
    data = (
        payload_json_bytes
        if payload_json_bytes is not None
        else None
        if payload is None
        else json.dumps(payload).encode("utf-8")
    )
    return urllib.request.Request(
        f"{config.normalized_host}{path_and_query}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {config.token}",
            "Content-Type": "application/json",
        },
    )


def _canonical_databricks_volume_file_path(dbfs_uri: str) -> str:
    if not isinstance(dbfs_uri, str) or not dbfs_uri:
        raise ValueError("dbfs_uri must be a non-empty string")
    prefix = "dbfs:"
    if not dbfs_uri.startswith(prefix):
        raise ValueError("Databricks Files API download requires a dbfs:/Volumes URI")
    raw_path = dbfs_uri.removeprefix(prefix)
    if any(ord(character) < 32 or ord(character) == 127 for character in raw_path):
        raise ValueError("dbfs:/Volumes URI cannot contain control characters")
    if any(character in raw_path for character in ("?", "#", "%", "\\")):
        raise ValueError(
            "dbfs:/Volumes URI must be a canonical path without URL syntax"
        )
    path = PurePosixPath(raw_path)
    if (
        not raw_path.startswith("/Volumes/")
        or path.as_posix() != raw_path
        or len(path.parts) < 6
        or path.parts[:2] != ("/", "Volumes")
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise ValueError(
            "Databricks Files API download requires one canonical "
            "dbfs:/Volumes/<catalog>/<schema>/<volume>/<file> URI"
        )
    return path.as_posix()


def _canonical_databricks_volume_directory_path(dbfs_uri: str) -> str:
    if not isinstance(dbfs_uri, str) or not dbfs_uri:
        raise ValueError("dbfs_uri must be a non-empty string")
    if not dbfs_uri.startswith("dbfs:"):
        raise ValueError("Databricks Files API listing requires a dbfs:/Volumes URI")
    raw_path = dbfs_uri.removeprefix("dbfs:")
    if any(ord(character) < 32 or ord(character) == 127 for character in raw_path):
        raise ValueError("dbfs:/Volumes URI cannot contain control characters")
    if any(character in raw_path for character in ("?", "#", "%", "\\")):
        raise ValueError(
            "dbfs:/Volumes URI must be a canonical path without URL syntax"
        )
    path = PurePosixPath(raw_path)
    if (
        not raw_path.startswith("/Volumes/")
        or path.as_posix() != raw_path
        or len(path.parts) < 5
        or path.parts[:2] != ("/", "Volumes")
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise ValueError(
            "Databricks Files API listing requires one canonical "
            "dbfs:/Volumes/<catalog>/<schema>/<volume>[/<directory>] URI"
        )
    return path.as_posix()


def _required_databricks_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_HEX_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase hexadecimal SHA-256")
    return value


def _canonical_databricks_local_upload_path(value: str | Path) -> Path:
    try:
        raw_path = os.fspath(value)
    except TypeError:
        raise TypeError("local_path must be a string or Path") from None
    if type(raw_path) is not str:
        raise TypeError("local_path must be a string or Path")
    if (
        not raw_path
        or not os.path.isabs(raw_path)
        or raw_path.startswith("//")
        or os.path.normpath(raw_path) != raw_path
        or any(ord(character) < 32 or ord(character) == 127 for character in raw_path)
    ):
        raise ValueError("local_path must be one canonical absolute path")
    try:
        resolved_path = os.path.realpath(raw_path, strict=True)
    except (OSError, ValueError):
        raise ValueError(
            "local_path must identify an existing canonical regular file"
        ) from None
    if resolved_path != raw_path:
        raise ValueError(
            "local_path must be canonical and cannot traverse symbolic links"
        )
    return Path(raw_path)


def _databricks_local_upload_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_databricks_local_upload_stat(value: os.stat_result) -> None:
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("local_path must identify a regular non-symlink file")
    if value.st_uid != os.getuid():
        raise ValueError("local_path must be owned by the current user")
    if value.st_nlink != 1:
        raise ValueError("local_path must have exactly one hard link")


def _open_databricks_local_upload_source(source_path: Path) -> int:
    try:
        path_stat = os.lstat(source_path)
    except OSError:
        raise ValueError(
            "local_path must identify an existing regular file"
        ) from None
    _require_databricks_local_upload_stat(path_stat)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if type(nofollow) is not int:
        raise RuntimeError("this platform cannot open local_path with O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NONBLOCK | nofollow
    cloexec = getattr(os, "O_CLOEXEC", 0)
    if type(cloexec) is int:
        flags |= cloexec
    try:
        file_descriptor = os.open(source_path, flags)
    except OSError:
        raise ValueError(
            "local_path could not be opened as a regular non-symlink file"
        ) from None
    try:
        descriptor_stat = os.fstat(file_descriptor)
        _require_databricks_local_upload_stat(descriptor_stat)
        if _databricks_local_upload_identity(
            descriptor_stat
        ) != _databricks_local_upload_identity(path_stat):
            raise ValueError("local_path identity changed while it was opened")
    except Exception:
        os.close(file_descriptor)
        raise
    return file_descriptor


def _require_stable_databricks_local_upload_source(
    file_descriptor: int,
    source_path: Path,
    identity: tuple[int, ...],
) -> None:
    try:
        descriptor_stat = os.fstat(file_descriptor)
        path_stat = os.lstat(source_path)
    except OSError:
        raise _DatabricksLocalUploadError(
            "Databricks Files API local upload source identity became unavailable"
        ) from None
    try:
        _require_databricks_local_upload_stat(descriptor_stat)
        _require_databricks_local_upload_stat(path_stat)
    except ValueError:
        raise _DatabricksLocalUploadError(
            "Databricks Files API local upload source invariants drifted"
        ) from None
    if (
        _databricks_local_upload_identity(descriptor_stat) != identity
        or _databricks_local_upload_identity(path_stat) != identity
    ):
        raise _DatabricksLocalUploadError(
            "Databricks Files API local upload source identity drifted"
        )


def _sha256_databricks_local_upload_source(
    file_descriptor: int,
    source_path: Path,
    identity: tuple[int, ...],
    *,
    expected_size: int,
) -> str:
    digest = hashlib.sha256()
    observed_size = 0
    while observed_size < expected_size:
        _require_stable_databricks_local_upload_source(
            file_descriptor,
            source_path,
            identity,
        )
        read_size = min(
            _DATABRICKS_VOLUME_FILE_STREAM_CHUNK_BYTES,
            expected_size - observed_size,
        )
        try:
            chunk = os.read(file_descriptor, read_size)
        except OSError:
            raise _DatabricksLocalUploadError(
                "Databricks Files API local upload source prehash read failed"
            ) from None
        if type(chunk) is not bytes:
            raise _DatabricksLocalUploadError(
                "Databricks Files API local upload source prehash chunk must be bytes"
            )
        if not chunk:
            raise _DatabricksLocalUploadError(
                "Databricks Files API local upload source ended during prehash"
            )
        if len(chunk) > read_size:
            raise _DatabricksLocalUploadError(
                "Databricks Files API local upload source prehash exceeded the "
                "chunk byte cap"
            )
        observed_size += len(chunk)
        digest.update(chunk)
    try:
        trailing = os.read(file_descriptor, 1)
    except OSError:
        raise _DatabricksLocalUploadError(
            "Databricks Files API local upload source prehash EOF probe failed"
        ) from None
    if type(trailing) is not bytes:
        raise _DatabricksLocalUploadError(
            "Databricks Files API local upload source prehash EOF probe must be bytes"
        )
    if trailing:
        raise _DatabricksLocalUploadError(
            "Databricks Files API local upload source contains bytes beyond "
            "expected_size"
        )
    _require_stable_databricks_local_upload_source(
        file_descriptor,
        source_path,
        identity,
    )
    return digest.hexdigest()


def _require_empty_databricks_binary_response(
    response: DatabricksBinaryHTTPResponse,
    *,
    label: str,
) -> None:
    raw_headers = getattr(response, "headers", None)
    if raw_headers is None or not hasattr(raw_headers, "get"):
        raise RuntimeError(f"{label} response headers are missing")
    content_length = raw_headers.get("content-length")
    if content_length not in (None, "0"):
        raise RuntimeError(f"{label} response content-length is not zero")
    content_encoding = raw_headers.get("content-encoding")
    if content_encoding not in (None, "", "identity"):
        raise RuntimeError(f"{label} response content-encoding is not identity")
    transfer_encoding = raw_headers.get("transfer-encoding")
    if transfer_encoding not in (None, ""):
        raise RuntimeError(f"{label} response transfer-encoding is unexpected")
    response_body = response.read(1)
    if type(response_body) is not bytes:
        raise RuntimeError(f"{label} response body must be bytes")
    if response_body:
        raise RuntimeError(f"{label} response body is not empty")


def _prove_databricks_volume_directory_exists(
    config: DatabricksWorkspaceConfig,
    directory_path: str,
    *,
    opener: DatabricksBinaryURLOpener | None,
) -> None:
    encoded_path = urllib.parse.quote(directory_path, safe="/-._~")
    request = urllib.request.Request(
        f"{config.normalized_host}/api/2.0/fs/directories{encoded_path}",
        method="HEAD",
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Authorization": f"Bearer {config.token}",
        },
    )
    resolved_opener = (
        _databricks_no_redirect_urlopen
        if opener is None
        else opener
    )
    try:
        with resolved_opener(request, timeout=config.timeout_seconds) as response:
            status = getattr(response, "status", None)
            if type(status) is not int or status != 200:
                raise RuntimeError(
                    "Databricks Files API directory existence proof returned "
                    f"unexpected HTTP status {status!r}"
                )
            _require_empty_databricks_binary_response(
                response,
                label="Databricks Files API directory existence proof",
            )
    except urllib.error.HTTPError as exc:
        raise _databricks_binary_http_error(exc, token=config.token) from None
    except urllib.error.URLError as exc:
        reason = _redact_databricks_secret_text(str(exc.reason), token=config.token)
        raise RuntimeError(f"Databricks request failed: {reason}") from None
    except Exception as exc:
        reason = _redact_databricks_secret_text(str(exc), token=config.token)
        raise RuntimeError(
            "Databricks Files API directory existence proof failed: "
            f"{reason}"
        ) from None


def _validated_databricks_volume_directory_entry(
    value: Any,
    *,
    parent_path: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("Databricks Files API directory entry must be an object")
    is_directory = value.get("is_directory")
    expected_keys = (
        {"is_directory", "name", "path"}
        if is_directory is True
        else {"file_size", "is_directory", "last_modified", "name", "path"}
    )
    if set(value) != expected_keys or type(is_directory) is not bool:
        raise RuntimeError("Databricks Files API directory entry schema drift")
    name = value.get("name")
    entry_path = value.get("path")
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or not isinstance(entry_path, str)
    ):
        raise RuntimeError("Databricks Files API directory entry name/path is invalid")
    canonical_entry_path = (
        entry_path.removesuffix("/") if is_directory is True else entry_path
    )
    if (is_directory is True) != entry_path.endswith("/"):
        raise RuntimeError("Databricks Files API directory entry path kind is invalid")
    parsed_path = PurePosixPath(canonical_entry_path)
    if (
        parsed_path.as_posix() != canonical_entry_path
        or parsed_path.parent.as_posix() != parent_path
        or parsed_path.name != name
    ):
        raise RuntimeError(
            "Databricks Files API directory entry is not a canonical direct child"
        )
    if is_directory is False:
        last_modified = value.get("last_modified")
        if type(last_modified) is not int or last_modified < 0:
            raise RuntimeError(
                "Databricks Files API directory entry last_modified is invalid"
            )
        file_size = value.get("file_size")
        if type(file_size) is not int or file_size < 0:
            raise RuntimeError(
                "Databricks Files API directory entry file_size is invalid"
            )
    return dict(value)


def _validate_databricks_volume_byte_cap(
    value: int,
    *,
    upper_bound: int,
    label: str,
) -> None:
    if type(value) is not int or value <= 0 or value > upper_bound:
        raise ValueError(
            f"{label} must be a positive integer no greater than {upper_bound}"
        )


def _validate_databricks_entry_cap(
    value: int,
    *,
    upper_bound: int,
    label: str,
) -> None:
    if type(value) is not int or value <= 0 or value > upper_bound:
        raise ValueError(
            f"{label} must be a positive integer no greater than {upper_bound}"
        )


def _required_databricks_content_length(
    value: Any,
    *,
    max_bytes: int,
    label: str,
) -> int:
    if (
        not isinstance(value, str)
        or len(value) > 20
        or re.fullmatch(r"[0-9]+", value) is None
    ):
        raise RuntimeError(f"{label} content-length is missing or invalid")
    content_length = int(value)
    if content_length > max_bytes:
        raise RuntimeError(
            f"{label} content-length exceeds the controller byte cap: "
            f"{content_length} > {max_bytes}"
        )
    return content_length


def _validated_optional_databricks_page_token(
    value: Any,
    *,
    label: str,
) -> str | None:
    if value in (None, ""):
        return None
    if (
        not isinstance(value, str)
        or len(value.encode("utf-8")) > DATABRICKS_API_PAGE_TOKEN_MAX_BYTES
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise RuntimeError(f"{label} is invalid or exceeds the token byte cap")
    return value


def _validated_databricks_identifier(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or len(value.encode("utf-8")) > 256
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise RuntimeError(f"{label} is invalid")
    return value


def _require_databricks_non_secret_text(
    value: str,
    *,
    token: str,
    label: str,
) -> None:
    if _redact_databricks_secret_text(value, token=token) != value:
        raise RuntimeError(f"{label} contains secret-like text")


def _validated_active_databricks_run(
    value: Any,
    *,
    token: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("Databricks active-run entry must be an object")
    run_id = value.get("run_id")
    if type(run_id) is not int or run_id <= 0:
        raise RuntimeError("Databricks active-run run_id is invalid")
    state = value.get("state")
    if not isinstance(state, Mapping):
        raise RuntimeError("Databricks active-run state is invalid")
    life_cycle_state = _validated_databricks_identifier(
        state.get("life_cycle_state"),
        label="Databricks active-run life_cycle_state",
    )
    _require_databricks_non_secret_text(
        life_cycle_state,
        token=token,
        label="Databricks active-run life_cycle_state",
    )
    return {
        "life_cycle_state": life_cycle_state,
        "run_id": run_id,
    }


def _validated_databricks_metadata_header(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1_024
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise RuntimeError(f"Databricks Files API metadata {label} is invalid")
    return value


def _databricks_binary_http_error(
    error: urllib.error.HTTPError,
    *,
    token: str,
) -> RuntimeError:
    try:
        error_content = error.read(_DATABRICKS_ERROR_BODY_MAX_BYTES + 1)
    except Exception as exc:
        reason = _redact_databricks_secret_text(str(exc), token=token)
        return RuntimeError(
            f"Databricks request failed with HTTP {error.code}; "
            f"error body read failed: {reason}"
        )
    if type(error_content) is not bytes:
        return RuntimeError(
            f"Databricks request failed with HTTP {error.code}; "
            "error body was not bytes"
        )
    if len(error_content) > _DATABRICKS_ERROR_BODY_MAX_BYTES:
        error_content = (
            error_content[:_DATABRICKS_ERROR_BODY_MAX_BYTES] + b"...[truncated]"
        )
    body = error_content.decode("utf-8", errors="replace")
    return RuntimeError(_format_databricks_http_error(error.code, body, token=token))


def _read_databricks_response_bytes_bounded(
    response: DatabricksHTTPResponse,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    raw_headers = getattr(response, "headers", None)
    if raw_headers is not None and not hasattr(raw_headers, "get"):
        raise RuntimeError(f"{label} headers are invalid")
    content_length_raw = (
        None if raw_headers is None else raw_headers.get("content-length")
    )
    transfer_encoding = (
        None if raw_headers is None else raw_headers.get("transfer-encoding")
    )
    if content_length_raw is not None:
        if transfer_encoding not in (None, ""):
            raise RuntimeError(
                f"{label} cannot combine content-length and transfer-encoding"
            )
        content_length = _required_databricks_content_length(
            content_length_raw,
            max_bytes=max_bytes,
            label=label,
        )
        exact_chunks: list[bytes] = []
        total_bytes = 0
        while total_bytes < content_length:
            read_size = min(
                _DATABRICKS_VOLUME_FILE_STREAM_CHUNK_BYTES,
                content_length - total_bytes,
            )
            chunk = response.read(read_size)
            if type(chunk) is not bytes:
                raise RuntimeError(f"{label} chunk must be bytes")
            if not chunk:
                raise RuntimeError(f"{label} ended before content-length")
            if len(chunk) > read_size:
                raise RuntimeError(f"{label} exceeded the response chunk byte cap")
            total_bytes += len(chunk)
            exact_chunks.append(chunk)
        eof = response.read(1)
        if type(eof) is not bytes:
            raise RuntimeError(f"{label} EOF probe must return bytes")
        if eof:
            raise RuntimeError(f"{label} contains bytes beyond content-length")
        return b"".join(exact_chunks)
    if transfer_encoding not in (None, "", "chunked", "identity"):
        raise RuntimeError(f"{label} transfer-encoding is unsupported")
    streamed_chunks: list[bytes] = []
    total_bytes = 0
    while True:
        read_size = min(
            _DATABRICKS_VOLUME_FILE_STREAM_CHUNK_BYTES,
            max_bytes - total_bytes + 1,
        )
        chunk = response.read(read_size)
        if type(chunk) is not bytes:
            raise RuntimeError(f"{label} chunk must be bytes")
        if len(chunk) > read_size:
            raise RuntimeError(f"{label} exceeded the response chunk byte cap")
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise RuntimeError(
                f"{label} exceeds the controller byte cap: more than "
                f"{max_bytes} bytes"
            )
        streamed_chunks.append(chunk)
    return b"".join(streamed_chunks)


def _bounded_databricks_binary_response(
    config: DatabricksWorkspaceConfig,
    request: urllib.request.Request,
    *,
    max_bytes: int,
    opener: DatabricksBinaryURLOpener,
    label: str,
) -> bytes:
    try:
        with opener(request, timeout=config.timeout_seconds) as response:
            status = getattr(response, "status", None)
            if type(status) is not int or status != 200:
                raise RuntimeError(
                    f"{label} returned unexpected HTTP status {status!r}"
                )
            content = _read_databricks_response_bytes_bounded(
                response,
                max_bytes=max_bytes,
                label=f"{label} response",
            )
    except urllib.error.HTTPError as exc:
        raise _databricks_binary_http_error(exc, token=config.token) from None
    except urllib.error.URLError as exc:
        reason = _redact_databricks_secret_text(str(exc.reason), token=config.token)
        raise RuntimeError(f"Databricks request failed: {reason}") from None
    except Exception as exc:
        reason = _redact_databricks_secret_text(str(exc), token=config.token)
        raise RuntimeError(f"{label} failed: {reason}") from None
    if not isinstance(content, bytes):
        raise RuntimeError(f"{label} response body must be bytes")
    if len(content) > max_bytes:
        raise RuntimeError(
            f"{label} response exceeds the controller byte cap: "
            f"more than {max_bytes} bytes"
        )
    return content


def _bounded_databricks_json_object(
    config: DatabricksWorkspaceConfig,
    request: urllib.request.Request,
    *,
    max_bytes: int,
    opener: DatabricksBinaryURLOpener,
    label: str,
) -> dict[str, Any]:
    raw = _bounded_databricks_binary_response(
        config,
        request,
        max_bytes=max_bytes,
        opener=opener,
        label=label,
    )
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=_unique_databricks_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError(f"{label} was not valid UTF-8 JSON") from None
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{label} JSON must be an object")
    return parsed


def _unique_databricks_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError("Databricks response JSON contains a duplicate key")
        result[key] = value
    return result


def _format_databricks_http_error(
    status_code: int, body: str, *, token: str | None = None
) -> str:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = {}
    if isinstance(parsed, dict):
        message = parsed.get("message") or parsed.get("error_code") or body
    else:
        message = body
    message = _redact_databricks_secret_text(str(message), token=token)
    return f"Databricks request failed with HTTP {status_code}: {message}"


def _redact_databricks_secret_text(text: str, *, token: str | None = None) -> str:
    redacted = text.replace(token, "[REDACTED]") if token else text
    redacted = _DATABRICKS_PAT_TOKEN_RE.sub("[REDACTED]", redacted)
    return re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/\-=]+", r"\1[REDACTED]", redacted)


def _success_record(
    action: str, response: dict[str, Any] | None = None
) -> dict[str, Any]:
    result = {
        "ok": True,
        "action": action,
    }
    if response is not None:
        result["response"] = response
    return result


def _write_error_record_or_stdout(
    result: dict[str, Any], output_json: str | None
) -> None:
    if not output_json:
        print(json.dumps(result, sort_keys=True))
        return
    try:
        write_databricks_run_response_json(result, output_json)
    except Exception as exc:
        fallback_result = dict(result)
        fallback_result["output_json_error"] = str(exc)
        fallback_result["output_json_error_type"] = type(exc).__name__
        print(json.dumps(fallback_result, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Submit, inspect, or stage Databricks artifacts using env or profile credentials."
    )
    parser.add_argument("--host-env", default=DEFAULT_DATABRICKS_HOST_ENV)
    parser.add_argument("--token-env", default=DEFAULT_DATABRICKS_TOKEN_ENV)
    parser.add_argument(
        "--profile", help="Databricks profile name from ~/.databrickscfg."
    )
    parser.add_argument(
        "--profile-auth-mode",
        choices=DATABRICKS_PROFILE_AUTH_MODES,
        default="auto",
        help=(
            "How --profile resolves credentials: auto keeps static token behavior and uses SDK "
            "only for auth_type profiles without a token; static requires a profile token; "
            "sdk forces Databricks SDK profile auth such as OAuth/CLI refresh."
        ),
    )
    parser.add_argument(
        "--config-file",
        default=DEFAULT_DATABRICKS_CONFIG_FILE,
        help="Databricks CLI config file used with --profile.",
    )
    parser.add_argument(
        "--timeout-seconds", type=float, default=DEFAULT_DATABRICKS_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--output-json",
        help="Write the command result JSON to this path instead of stdout.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser(
        "submit", help="POST a Jobs runs/submit payload JSON."
    )
    submit_parser.add_argument("--payload-json", required=True)

    reserved_submit_parser = subparsers.add_parser(
        "reserve-and-submit",
        help=(
            "Atomically reserve a bounded payload in the cluster-hour ledger, "
            "then POST that exact immutable snapshot."
        ),
    )
    reserved_submit_parser.add_argument("--payload-json", required=True)
    reserved_submit_parser.add_argument("--ledger-json", required=True)
    reserved_submit_parser.add_argument("--attempt-id", required=True)
    reserved_submit_parser.add_argument("--workload-id", required=True)
    reserved_submit_parser.add_argument(
        "--representative-canary",
        action="store_true",
        help=(
            "Require workload_id and the exact payload to match Cachet's ordered "
            "representative canary manifest."
        ),
    )

    subparsers.add_parser(
        "auth-check",
        help="GET the current Databricks workspace user endpoint to verify credentials without launching a run.",
    )

    get_parser = subparsers.add_parser("get", help="GET a Databricks run by run id.")
    get_parser.add_argument("--run-id", required=True)
    get_parser.add_argument(
        "--summary",
        action="store_true",
        help="Write only a compact run/task status summary.",
    )
    get_parser.add_argument(
        "--submit-payload-json",
        help="Attach a sanitized hash and V1 AWS single-node GPU cluster summary for the runs/submit payload that launched this run.",
    )
    get_parser.add_argument(
        "--include-response",
        action="store_true",
        help="Also include the raw Jobs API response when using --summary.",
    )
    get_parser.add_argument(
        "--expected-hardware-target",
        choices=SUPPORTED_V1_HARDWARE_TARGETS,
        help=(
            "Validate the compact summary and attached submit payload against a V1 hardware target. "
            "Requires --summary and --submit-payload-json."
        ),
    )
    get_parser.add_argument(
        "--expected-node-type-id",
        help=(
            "Validate the compact summary and attached submit payload against an exact Databricks "
            "node_type_id. Requires --summary and --submit-payload-json."
        ),
    )
    payload_summary_parser = subparsers.add_parser(
        "payload-summary",
        help="Summarize and optionally validate a Databricks runs/submit payload without credentials.",
    )
    payload_summary_parser.add_argument("--payload-json", required=True)
    payload_summary_parser.add_argument(
        "--expected-hardware-target",
        choices=SUPPORTED_V1_HARDWARE_TARGETS,
        help="Validate the payload summary against a V1 hardware target.",
    )
    payload_summary_parser.add_argument(
        "--expected-node-type-id",
        help="Validate the payload summary against an exact Databricks node_type_id.",
    )
    put_parser = subparsers.add_parser(
        "put-dbfs-file", help="Upload a small local artifact to DBFS."
    )
    put_parser.add_argument("--local-path", required=True, help="Local file to upload.")
    put_parser.add_argument(
        "--dbfs-path",
        required=True,
        help="Destination path such as dbfs:/FileStore/cachet/file.whl.",
    )
    put_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite an existing DBFS file."
    )
    stage_submit_parser = subparsers.add_parser(
        "stage-and-submit",
        help="Upload small DBFS artifacts, then submit a Databricks runs/submit payload.",
    )
    stage_submit_parser.add_argument("--payload-json", required=True)
    stage_submit_parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="LOCAL_PATH=DBFS_PATH",
        help="Artifact to stage before submit. Repeat for each runner or wheel required by the payload.",
    )
    stage_submit_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing DBFS artifacts."
    )
    stage_submit_parser.add_argument(
        "--require-payload-dbfs-artifacts",
        action="store_true",
        help="Fail before uploading unless every dbfs:/ URI in the payload has a matching --artifact destination.",
    )
    stage_submit_parser.add_argument(
        "--require-payload-staged-dbfs-artifacts",
        action="store_true",
        help=(
            "Fail before uploading unless DBFS runner, wheel, plan, and SGLang launch-config artifacts "
            "that the payload reads have matching --artifact destinations. Generated DBFS outputs are ignored."
        ),
    )
    stage_submit_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate artifacts and payload DBFS references without Databricks credentials or network requests.",
    )
    stage_submit_parser.add_argument(
        "--preflight-auth-check",
        action="store_true",
        help="Verify Databricks credentials before uploading artifacts or submitting the run.",
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "get" and args.expected_hardware_target:
            if not args.summary:
                raise ValueError("--expected-hardware-target requires --summary")
            if not args.submit_payload_json:
                raise ValueError(
                    "--expected-hardware-target requires --submit-payload-json"
                )
            if args.include_response:
                raise ValueError(
                    "--expected-hardware-target cannot be combined with --include-response"
                )
        if args.command == "get" and args.expected_node_type_id:
            if not args.summary:
                raise ValueError("--expected-node-type-id requires --summary")
            if not args.submit_payload_json:
                raise ValueError(
                    "--expected-node-type-id requires --submit-payload-json"
                )
            if args.include_response:
                raise ValueError(
                    "--expected-node-type-id cannot be combined with --include-response"
                )
        if args.command == "stage-and-submit" and args.dry_run:
            result = plan_databricks_stage_and_submit(
                read_databricks_run_submit_payload(args.payload_json),
                tuple(
                    _parse_dbfs_artifact_mapping(artifact) for artifact in args.artifact
                ),
                overwrite=args.overwrite,
                require_payload_dbfs_artifacts=args.require_payload_dbfs_artifacts,
                require_payload_staged_dbfs_artifacts=args.require_payload_staged_dbfs_artifacts,
                submit_payload_path=args.payload_json,
            )
        elif args.command == "payload-summary":
            payload = read_databricks_run_submit_payload(args.payload_json)
            _validate_databricks_run_submit_payload_tasks(payload)
            summary = summarize_databricks_run_submit_payload(
                payload,
                source_path=args.payload_json,
            )
            if args.expected_hardware_target:
                _validate_databricks_submit_payload_summary(
                    summary,
                    expected_hardware_target=args.expected_hardware_target,
                    expected_node_type_id=args.expected_node_type_id,
                )
            elif args.expected_node_type_id:
                _validate_databricks_submit_payload_summary(
                    summary,
                    expected_node_type_id=args.expected_node_type_id,
                )
            result = _success_record(args.command)
            result["summary"] = summary
        else:
            config = _databricks_workspace_config_from_args(args)
            if args.command == "submit":
                response = submit_databricks_run(
                    config, read_databricks_run_submit_payload(args.payload_json)
                )
            elif args.command == "reserve-and-submit":
                reservation_validator = _cli_reservation_validator(
                    args.workload_id,
                    representative_canary=args.representative_canary,
                )
                response = reserve_and_submit_databricks_run_json(
                    config,
                    args.payload_json,
                    ledger_path=args.ledger_json,
                    attempt_id=args.attempt_id,
                    workload_id=args.workload_id,
                    reservation_validator=reservation_validator,
                )
            elif args.command == "auth-check":
                result = _success_record(args.command)
                result["auth"] = check_databricks_auth(config)
                response = None
            elif args.command == "get":
                response = get_databricks_run(config, args.run_id)
            elif args.command == "put-dbfs-file":
                result = _put_databricks_dbfs_file_record(
                    config,
                    args.local_path,
                    args.dbfs_path,
                    overwrite=args.overwrite,
                )
                response = None
            elif args.command == "stage-and-submit":
                result = stage_and_submit_databricks_run(
                    config,
                    read_databricks_run_submit_payload(args.payload_json),
                    tuple(
                        _parse_dbfs_artifact_mapping(artifact)
                        for artifact in args.artifact
                    ),
                    overwrite=args.overwrite,
                    require_payload_dbfs_artifacts=args.require_payload_dbfs_artifacts,
                    require_payload_staged_dbfs_artifacts=args.require_payload_staged_dbfs_artifacts,
                    preflight_auth_check=args.preflight_auth_check,
                )
                response = None
            else:  # pragma: no cover - argparse enforces this.
                raise ValueError(f"unknown command {args.command!r}")
        if args.command == "get" and args.summary:
            submit_payload = (
                read_databricks_run_submit_payload(args.submit_payload_json)
                if args.submit_payload_json
                else None
            )
            result = _success_record(
                args.command, response if args.include_response else None
            )
            summary = summarize_databricks_run(
                response,
                submit_payload=submit_payload,
                submit_payload_path=args.submit_payload_json,
            )
            if args.expected_hardware_target:
                validate_databricks_run_status_sidecar(
                    summary,
                    expected_hardware_target=args.expected_hardware_target,
                    expected_node_type_id=args.expected_node_type_id,
                )
            elif args.expected_node_type_id:
                validate_databricks_run_status_sidecar(
                    summary,
                    expected_node_type_id=args.expected_node_type_id,
                )
            result["summary"] = summary
        elif args.command == "auth-check":
            pass
        elif args.command == "put-dbfs-file":
            pass
        elif args.command == "payload-summary":
            pass
        elif args.command == "stage-and-submit":
            pass
        else:
            result = _success_record(args.command, response)
        if args.output_json:
            write_databricks_run_response_json(result, args.output_json)
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        _write_error_record_or_stdout(result, args.output_json)
        return 1
    return 0


def _validate_databricks_run_submit_payload_tasks(payload: Mapping[str, Any]) -> None:
    tasks = payload.get("tasks")
    if (
        not isinstance(tasks, Sequence)
        or isinstance(tasks, (str, bytes, bytearray))
        or not tasks
    ):
        raise ValueError(
            "Databricks run-submit payload tasks must be a non-empty array"
        )
    invalid_indices = [
        str(index) for index, task in enumerate(tasks) if not isinstance(task, Mapping)
    ]
    if invalid_indices:
        raise ValueError(
            "Databricks run-submit payload tasks must contain only objects; "
            f"invalid task indices: {', '.join(invalid_indices)}"
        )


def _validate_databricks_submit_payload_summary(
    summary: Mapping[str, Any],
    *,
    expected_hardware_target: str | None = None,
    expected_node_type_id: str | None = None,
) -> None:
    issues = _databricks_submit_payload_sidecar_issues(
        summary,
        tasks=summary.get("tasks"),
        expected_hardware_target=expected_hardware_target,
        expected_node_type_id=expected_node_type_id,
    )
    if issues:
        raise ValueError("; ".join(_dedupe_strings(issues)))


def _databricks_workspace_config_from_args(
    args: argparse.Namespace,
) -> DatabricksWorkspaceConfig:
    if args.profile:
        return databricks_workspace_config_from_profile(
            args.profile,
            config_file=args.config_file,
            timeout_seconds=args.timeout_seconds,
            profile_auth_mode=args.profile_auth_mode,
        )
    if args.profile_auth_mode != "auto":
        raise ValueError("--profile-auth-mode requires --profile")
    if args.config_file != DEFAULT_DATABRICKS_CONFIG_FILE:
        raise ValueError("--config-file requires --profile")
    return databricks_workspace_config_from_env(
        host_env=args.host_env,
        token_env=args.token_env,
        timeout_seconds=args.timeout_seconds,
    )


def _cli_reservation_validator(
    workload_id: str,
    *,
    representative_canary: bool,
) -> DatabricksReservationValidator | None:
    from document_kv_cache.canary_orchestration import (
        representative_canary_workload_manifest,
        validate_representative_canary_reservation,
    )

    manifest = representative_canary_workload_manifest()
    try:
        manifest.workload_for_id(workload_id)
    except KeyError:
        is_representative_workload = False
    else:
        is_representative_workload = True
    if representative_canary and not is_representative_workload:
        raise ValueError(
            "--representative-canary requires a workload_id from the exact "
            "representative canary manifest"
        )
    if is_representative_workload and not representative_canary:
        raise ValueError(
            "representative canary workload_id requires --representative-canary"
        )
    return validate_representative_canary_reservation if representative_canary else None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
