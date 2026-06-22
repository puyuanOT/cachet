"""Submit, inspect, and summarize Databricks Jobs runs."""

from __future__ import annotations

import argparse
import base64
import hashlib
from configparser import ConfigParser
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any, Protocol
import urllib.error
import urllib.parse
import urllib.request

from document_kv_cache._hardware_targets import (
    HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES,
    SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES,
    SUPPORTED_V1_HARDWARE_TARGETS,
    V1_HARDWARE_TARGET_PROFILES,
)


__all__ = [
    "DEFAULT_DATABRICKS_HOST_ENV",
    "DEFAULT_DATABRICKS_CONFIG_FILE",
    "DEFAULT_DATABRICKS_TOKEN_ENV",
    "DEFAULT_DATABRICKS_TIMEOUT_SECONDS",
    "DATABRICKS_PROFILE_AUTH_MODES",
    "DATABRICKS_AUTH_CHECK_RECORD_TYPE",
    "DATABRICKS_DBFS_PUT_MAX_CONTENT_BYTES",
    "DATABRICKS_RUN_STATUS_RECORD_TYPE",
    "DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE",
    "DatabricksWorkspaceConfig",
    "databricks_workspace_config_from_env",
    "databricks_workspace_config_from_profile",
    "databricks_workspace_config_from_sdk_profile",
    "check_databricks_auth",
    "submit_databricks_run",
    "get_databricks_run",
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
DATABRICKS_AUTH_CHECK_RECORD_TYPE = "document_kv.databricks_auth_check.v1"
DATABRICKS_RUN_STATUS_RECORD_TYPE = "document_kv.databricks_run_status.v1"
DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE = "document_kv.databricks_run_submit_payload.v1"
_DATABRICKS_AUTH_CHECK_ENDPOINT = "/api/2.0/preview/scim/v2/Me"
_DATABRICKS_GPU_TYPE_FIELD = "aws_single_node_gpu_type"
_LEGACY_DATABRICKS_GPU_TYPE_FIELD = "aws_g5_node_type"
_DATABRICKS_GPU_TYPE_FIELDS = (
    _DATABRICKS_GPU_TYPE_FIELD,
    _LEGACY_DATABRICKS_GPU_TYPE_FIELD,
)
_DATABRICKS_CONFIG_DEFAULT_SECTION = "__cachet_no_inherited_databricks_defaults__"
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
DATABRICKS_TERMINAL_LIFE_CYCLE_STATES = frozenset({"TERMINATED", "SKIPPED", "INTERNAL_ERROR"})
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

    def read(self) -> bytes: ...


class DatabricksURLOpener(Protocol):
    def __call__(self, request: urllib.request.Request, *, timeout: float) -> DatabricksHTTPResponse: ...


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
    return DatabricksWorkspaceConfig(host=host, token=token, timeout_seconds=timeout_seconds)


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
        return DatabricksWorkspaceConfig(host=host, token=token, timeout_seconds=timeout_seconds)
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
        raise ValueError(f"Databricks SDK profile {profile_name!r} could not be loaded: {message}") from exc
    resolved_host = str(getattr(sdk_config, "host", "") or host).strip()
    try:
        auth_headers = sdk_config.authenticate()
    except Exception as exc:
        message = _redact_databricks_secret_text(str(exc))
        raise ValueError(f"Databricks SDK profile {profile_name!r} could not authenticate: {message}") from exc
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
    except ModuleNotFoundError as exc:
        raise ValueError(
            "Databricks SDK profile auth requires installing the databricks extra "
            "with document-kv-cache[databricks]"
        ) from exc
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
        raise ValueError(f"Databricks SDK profile {profile_name!r} could not be loaded: {message}") from exc
    finally:
        _restore_environment(env_snapshot)


def _databricks_sdk_profile_env_names(config_type: Any) -> tuple[str, ...]:
    env_names: set[str] = set()
    for attribute in config_type.attributes():
        if not getattr(attribute, "auth", None) and (
            getattr(attribute, "name", None) not in _DATABRICKS_SDK_PROFILE_ISOLATION_ATTRIBUTES
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


def submit_databricks_run(
    config: DatabricksWorkspaceConfig,
    payload: dict[str, Any],
    *,
    opener: DatabricksURLOpener = urllib.request.urlopen,
) -> dict[str, Any]:
    return _databricks_api_json(
        config,
        "POST",
        "/api/2.1/jobs/runs/submit",
        payload=payload,
        opener=opener,
    )


def check_databricks_auth(
    config: DatabricksWorkspaceConfig,
    *,
    opener: DatabricksURLOpener = urllib.request.urlopen,
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


def get_databricks_run(
    config: DatabricksWorkspaceConfig,
    run_id: int | str,
    *,
    opener: DatabricksURLOpener = urllib.request.urlopen,
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


def put_databricks_dbfs_file(
    config: DatabricksWorkspaceConfig,
    local_path: str | Path,
    dbfs_path: str,
    *,
    overwrite: bool = False,
    opener: DatabricksURLOpener = urllib.request.urlopen,
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
    opener: DatabricksURLOpener = urllib.request.urlopen,
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
    submit_payload_path: str | None = None,
) -> dict[str, Any]:
    prepared_artifacts = _prepare_databricks_stage_artifacts(
        payload,
        artifacts,
        overwrite=overwrite,
        require_payload_dbfs_artifacts=require_payload_dbfs_artifacts,
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
    preflight_auth_check: bool = False,
    opener: DatabricksURLOpener = urllib.request.urlopen,
) -> dict[str, Any]:
    prepared_artifacts = _prepare_databricks_stage_artifacts(
        payload,
        artifacts,
        overwrite=overwrite,
        require_payload_dbfs_artifacts=require_payload_dbfs_artifacts,
    )
    auth_record = check_databricks_auth(config, opener=opener) if preflight_auth_check else None
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
        _stage_and_submit_artifact_upload_record(record)
        for record in artifact_uploads
    ]
    return result


def write_databricks_run_response_json(response: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    tasks = tuple(_task_summary(task) for task in _sequence_of_mappings(run.get("tasks")))
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
    node_type_ids = _sorted_unique_texts(summary.get("node_type_id") for summary in task_summaries)
    driver_node_type_ids = _sorted_unique_texts(summary.get("driver_node_type_id") for summary in task_summaries)
    hardware_targets = _hardware_targets_for_task_summaries(task_summaries)
    spark_versions = _sorted_unique_texts(summary.get("spark_version") for summary in task_summaries)
    spark_env_keys = _sorted_task_list_field_values(task_summaries, "spark_env_keys")
    data_security_modes = _sorted_unique_texts(summary.get("data_security_mode") for summary in task_summaries)
    canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    aws_single_node_gpu_type = (
        bool(task_summaries)
        and all(_is_supported_aws_single_node_gpu_type(summary.get("node_type_id")) for summary in task_summaries)
        and all(_is_supported_aws_single_node_gpu_type(summary.get("driver_node_type_id")) for summary in task_summaries)
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
        "single_node": bool(task_summaries) and all(summary["single_node"] for summary in task_summaries),
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
) -> tuple[str, ...]:
    """Return release-oriented issues for a Databricks run-status sidecar."""

    status_record = databricks_run_status_record(record)
    issues: list[str] = []
    if "response" in record:
        issues.append("Databricks run status sidecar must not include the raw Jobs API response")
    issues.extend(_databricks_run_status_container_key_issues(record))
    issues.extend(_databricks_run_status_wrapper_field_issues(record))
    if status_record is None:
        issues.append("Databricks run status sidecar must be a status record or databricks_runs get --summary output")
        return _dedupe_strings(issues)
    issues.extend(_unexpected_keys(status_record, _DATABRICKS_RUN_STATUS_KEYS, "Databricks run status sidecar summary"))
    issues.extend(_databricks_run_status_field_issues(status_record))
    if status_record.get("record_type") != DATABRICKS_RUN_STATUS_RECORD_TYPE:
        issues.append(f"Databricks run status sidecar record_type must be {DATABRICKS_RUN_STATUS_RECORD_TYPE!r}")
    if status_record.get("terminal") is not True:
        issues.append("Databricks run status sidecar terminal must be true")
    if status_record.get("succeeded") is not True:
        issues.append("Databricks run status sidecar succeeded must be true")
    if status_record.get("life_cycle_state") != "TERMINATED":
        issues.append("Databricks run status sidecar life_cycle_state must be 'TERMINATED'")
    if status_record.get("result_state") != "SUCCESS":
        issues.append("Databricks run status sidecar result_state must be 'SUCCESS'")
    if (
        status_record.get("terminal") is True
        and status_record.get("succeeded") is True
        and status_record.get("active_task_key") is not None
    ):
        issues.append("Databricks run status sidecar active_task_key must be null for successful terminal runs")
    run_id = status_record.get("run_id")
    if not ((type(run_id) is int and run_id >= 0) or (isinstance(run_id, str) and run_id)):
        issues.append("Databricks run status sidecar run_id must be a non-negative integer or non-empty string")
    task_count = status_record.get("task_count")
    tasks = status_record.get("tasks")
    if type(task_count) is not int or task_count <= 0:
        issues.append("Databricks run status sidecar task_count must be a positive integer")
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes, bytearray)) or not tasks:
        issues.append("Databricks run status sidecar tasks must be a non-empty array")
    else:
        if type(task_count) is int and task_count > 0 and len(tasks) != task_count:
            issues.append("Databricks run status sidecar task_count must match tasks length")
        issues.extend(_databricks_run_status_task_issues(tasks))
    submit_payload = status_record.get("submit_payload")
    if not isinstance(submit_payload, Mapping):
        issues.append("Databricks run status sidecar submit_payload must be an object")
    else:
        issues.extend(
            _databricks_submit_payload_sidecar_issues(
                submit_payload,
                tasks=tasks,
                expected_hardware_target=expected_hardware_target,
            )
        )
        issues.extend(_databricks_run_submit_payload_identity_issues(status_record, submit_payload))
        issues.extend(_databricks_run_submit_payload_spark_env_identity_issues(tasks, submit_payload))
    return _dedupe_strings(issues)


def validate_databricks_run_status_sidecar(
    record: Mapping[str, Any],
    *,
    expected_hardware_target: str | None = None,
) -> None:
    """Validate a release-oriented Databricks run-status sidecar."""

    issues = databricks_run_status_sidecar_issues(
        record,
        expected_hardware_target=expected_hardware_target,
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
) -> dict[str, Any]:
    parsed, _status = _databricks_api_response_json(
        config,
        method,
        path_and_query,
        opener=opener,
        payload=payload,
    )
    return parsed


def _databricks_api_response_json(
    config: DatabricksWorkspaceConfig,
    method: str,
    path_and_query: str,
    *,
    opener: DatabricksURLOpener,
    payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int | None]:
    request = _databricks_request(config, method, path_and_query, payload=payload)
    try:
        with opener(request, timeout=config.timeout_seconds) as response:
            body = response.read().decode("utf-8")
            status = getattr(response, "status", None)
            parsed = json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(_format_databricks_http_error(exc.code, body, token=config.token)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Databricks request failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Databricks response was not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Databricks response JSON must be an object")
    return parsed, status


def _task_summary(task: Mapping[str, Any]) -> dict[str, Any]:
    state = _mapping(task.get("state"))
    life_cycle_state = _optional_str(state.get("life_cycle_state"))
    return {
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
        "single_node": cluster.get("num_workers") == 0 and custom_tags.get("ResourceClass") == "SingleNode",
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
_HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES = HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES
_V1_HARDWARE_TARGET_PREFIXES = tuple(
    (profile.hardware_target, profile.databricks_node_type_prefixes)
    for profile in V1_HARDWARE_TARGET_PROFILES
)


def _is_supported_aws_single_node_gpu_type(value: Any) -> bool:
    return isinstance(value, str) and value.lower().startswith(_SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES)


def _is_expected_aws_single_node_gpu_type(value: Any, expected_hardware_target: str | None) -> bool:
    if expected_hardware_target is None:
        return _is_supported_aws_single_node_gpu_type(value)
    prefixes = _HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES.get(expected_hardware_target)
    return isinstance(value, str) and prefixes is not None and value.lower().startswith(prefixes)


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
    return all(record[field_name] is True for field_name in _present_gpu_type_fields(record))


def _present_gpu_type_fields(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(field_name for field_name in _DATABRICKS_GPU_TYPE_FIELDS if field_name in record)


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _databricks_run_status_task_issues(tasks: Sequence[Any]) -> tuple[str, ...]:
    issues: list[str] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            issues.append(f"Databricks run status sidecar tasks[{index}] must be an object")
            continue
        issues.extend(
            _unexpected_keys(task, _DATABRICKS_RUN_STATUS_TASK_KEYS, f"Databricks run status sidecar tasks[{index}]")
        )
        issues.extend(_databricks_run_status_task_field_issues(task, index=index))
        issues.extend(_list_of_strings_field(task, "spark_env_keys", f"Databricks run status sidecar tasks[{index}]"))
        spark_env_keys = _valid_string_list(task.get("spark_env_keys"))
        if spark_env_keys is not None:
            issues.extend(_spark_env_key_issues(spark_env_keys, f"Databricks run status sidecar tasks[{index}]"))
        if not isinstance(task.get("task_key"), str) or not task["task_key"]:
            issues.append(f"Databricks run status sidecar tasks[{index}].task_key must be non-empty")
        if task.get("life_cycle_state") != "TERMINATED":
            issues.append(f"Databricks run status sidecar tasks[{index}].life_cycle_state must be 'TERMINATED'")
        if task.get("result_state") != "SUCCESS":
            issues.append(f"Databricks run status sidecar tasks[{index}].result_state must be 'SUCCESS'")
    return tuple(issues)


def _databricks_submit_payload_sidecar_issues(
    record: Mapping[str, Any],
    *,
    tasks: Any,
    expected_hardware_target: str | None,
) -> tuple[str, ...]:
    issues: list[str] = []
    issues.extend(_unexpected_keys(record, _DATABRICKS_SUBMIT_PAYLOAD_KEYS, "Databricks run status sidecar submit_payload"))
    issues.extend(_databricks_submit_payload_field_issues(record))
    if record.get("record_type") != DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE:
        issues.append(
            f"Databricks run status sidecar submit_payload.record_type must be {DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE!r}"
        )
    source_path = record.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        issues.append("Databricks run status sidecar submit_payload.source_path must be non-empty")
    if not isinstance(record.get("sha256"), str) or not _SHA256_HEX_RE.fullmatch(record["sha256"]):
        issues.append("Databricks run status sidecar submit_payload.sha256 must be a 64-character lowercase hex digest")
    if record.get("single_node") is not True:
        issues.append("Databricks run status sidecar submit_payload.single_node must be true")
    if _submit_payload_gpu_type_supported(record) is not True:
        issues.append("Databricks run status sidecar submit_payload.aws_single_node_gpu_type must be true")
    task_count = record.get("task_count")
    payload_tasks = record.get("tasks")
    if type(task_count) is not int or task_count <= 0:
        issues.append("Databricks run status sidecar submit_payload.task_count must be a positive integer")
    if not isinstance(payload_tasks, Sequence) or isinstance(payload_tasks, (str, bytes, bytearray)) or not payload_tasks:
        issues.append("Databricks run status sidecar submit_payload.tasks must be a non-empty array")
    else:
        if type(task_count) is int and task_count > 0 and len(payload_tasks) != task_count:
            issues.append("Databricks run status sidecar submit_payload.task_count must match tasks length")
        issues.extend(
            _databricks_submit_payload_task_issues(
                payload_tasks,
                expected_hardware_target=expected_hardware_target,
            )
        )
        issues.extend(_databricks_submit_payload_summary_field_issues(record, payload_tasks))
    data_security_modes = record.get("data_security_modes")
    if not isinstance(data_security_modes, Sequence) or isinstance(data_security_modes, (str, bytes, bytearray)):
        issues.append("Databricks run status sidecar submit_payload.data_security_modes must be an array")
    elif "SINGLE_USER" not in data_security_modes:
        issues.append("Databricks run status sidecar submit_payload.data_security_modes must include SINGLE_USER")
    if isinstance(tasks, Sequence) and not isinstance(tasks, (str, bytes, bytearray)):
        status_task_keys = _task_key_list(tasks)
        payload_task_keys = _task_key_list(record.get("tasks"))
        if not status_task_keys:
            issues.append("Databricks run status sidecar tasks must include task keys")
        if not payload_task_keys:
            issues.append("Databricks run status sidecar submit_payload.tasks must include task keys")
        if status_task_keys and payload_task_keys and status_task_keys != payload_task_keys:
            issues.append("Databricks run status sidecar submit_payload.task_keys must match status task keys")
    if isinstance(record.get("task_keys"), Sequence) and not isinstance(record.get("task_keys"), (str, bytes, bytearray)):
        declared_task_keys = [key for key in record["task_keys"] if isinstance(key, str) and key]
        payload_task_keys = _task_key_list(record.get("tasks"))
        if declared_task_keys != payload_task_keys:
            issues.append("Databricks run status sidecar submit_payload.task_keys must match submit_payload.tasks")
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
        return ("Databricks run status sidecar submit_payload.run_name must match run_name",)
    return ()


def _databricks_run_submit_payload_spark_env_identity_issues(
    status_tasks: Any,
    submit_payload: Mapping[str, Any],
) -> tuple[str, ...]:
    payload_tasks = submit_payload.get("tasks")
    if not isinstance(status_tasks, Sequence) or isinstance(status_tasks, (str, bytes, bytearray)):
        return ()
    if not isinstance(payload_tasks, Sequence) or isinstance(payload_tasks, (str, bytes, bytearray)):
        return ()
    status_by_task_key = {
        task["task_key"]: task
        for task in status_tasks
        if isinstance(task, Mapping) and isinstance(task.get("task_key"), str) and task["task_key"]
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
        expected_spark_env_keys = _sorted_task_list_field_values(tasks, "spark_env_keys")
        if actual_spark_env_keys != expected_spark_env_keys:
            issues.append("Databricks run status sidecar submit_payload.spark_env_keys must match submit_payload.tasks")
        issues.extend(_spark_env_key_issues(actual_spark_env_keys, "Databricks run status sidecar submit_payload"))
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
) -> tuple[str, ...]:
    issues: list[str] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            issues.append(f"Databricks run status sidecar submit_payload.tasks[{index}] must be an object")
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
            issues.append(f"Databricks run status sidecar submit_payload.tasks[{index}].task_key must be non-empty")
        for field_name in ("node_type_id", "driver_node_type_id"):
            value = task.get(field_name)
            if not _is_supported_aws_single_node_gpu_type(value):
                issues.append(
                    f"Databricks run status sidecar submit_payload.tasks[{index}].{field_name} "
                    "must be a supported V1 AWS GPU node type"
                )
            elif not _is_expected_aws_single_node_gpu_type(value, expected_hardware_target):
                issues.append(
                    f"Databricks run status sidecar submit_payload.tasks[{index}].{field_name} must match "
                    f"hardware_target {expected_hardware_target!r}"
                )
        if task.get("single_node") is not True:
            issues.append(f"Databricks run status sidecar submit_payload.tasks[{index}].single_node must be true")
        if task.get("data_security_mode") != "SINGLE_USER":
            issues.append(
                f"Databricks run status sidecar submit_payload.tasks[{index}].data_security_mode must be 'SINGLE_USER'"
            )
    return tuple(issues)


def _databricks_run_status_field_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    issues.extend(_required_str_field(record, "record_type", "Databricks run status sidecar"))
    issues.extend(_run_id_field_issues(record, "run_id", "Databricks run status sidecar"))
    for field_name in ("run_name", "run_page_url", "state_message", "active_task_key", "cluster_id"):
        issues.extend(_optional_str_field(record, field_name, "Databricks run status sidecar"))
    for field_name in ("life_cycle_state", "result_state"):
        issues.extend(_required_str_field(record, field_name, "Databricks run status sidecar"))
    for field_name in ("start_time", "end_time"):
        issues.extend(_optional_int_field(record, field_name, "Databricks run status sidecar"))
    for field_name in ("terminal", "succeeded"):
        issues.extend(_bool_field(record, field_name, "Databricks run status sidecar"))
    return tuple(issues)


def _databricks_run_status_task_field_issues(task: Mapping[str, Any], *, index: int) -> tuple[str, ...]:
    label = f"Databricks run status sidecar tasks[{index}]"
    issues: list[str] = []
    issues.extend(_required_str_field(task, "task_key", label))
    issues.extend(_run_id_field_issues(task, "run_id", label))
    for field_name in ("life_cycle_state", "result_state"):
        issues.extend(_required_str_field(task, field_name, label))
    for field_name in ("state_message", "cluster_id"):
        issues.extend(_optional_str_field(task, field_name, label))
    for field_name in ("start_time", "end_time"):
        issues.extend(_optional_int_field(task, field_name, label))
    return tuple(issues)


def _databricks_submit_payload_field_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    for field_name in ("record_type", "source_path"):
        issues.extend(_required_str_field(record, field_name, "Databricks run status sidecar submit_payload"))
    issues.extend(_optional_str_field(record, "run_name", "Databricks run status sidecar submit_payload"))
    for field_name in ("task_keys", "node_type_ids", "driver_node_type_ids", "spark_versions", "data_security_modes"):
        issues.extend(_list_of_strings_field(record, field_name, "Databricks run status sidecar submit_payload"))
    if "hardware_targets" in record:
        issues.extend(
            _list_of_strings_field(record, "hardware_targets", "Databricks run status sidecar submit_payload")
        )
    issues.extend(_bool_field(record, "single_node", "Databricks run status sidecar submit_payload"))
    if not _present_gpu_type_fields(record):
        issues.append(
            "Databricks run status sidecar submit_payload.aws_single_node_gpu_type or aws_g5_node_type must be present"
        )
    for field_name in _DATABRICKS_GPU_TYPE_FIELDS:
        if field_name in record:
            issues.extend(_bool_field(record, field_name, "Databricks run status sidecar submit_payload"))
    if _gpu_type_fields_contradict(record):
        issues.append(
            "Databricks run status sidecar submit_payload.aws_single_node_gpu_type and aws_g5_node_type must match"
        )
    return tuple(issues)


def _gpu_type_fields_contradict(record: Mapping[str, Any]) -> bool:
    return (
        type(record.get(_DATABRICKS_GPU_TYPE_FIELD)) is bool
        and type(record.get(_LEGACY_DATABRICKS_GPU_TYPE_FIELD)) is bool
        and record[_DATABRICKS_GPU_TYPE_FIELD] != record[_LEGACY_DATABRICKS_GPU_TYPE_FIELD]
    )


def _databricks_submit_payload_task_field_issues(task: Mapping[str, Any], *, index: int) -> tuple[str, ...]:
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


def _databricks_run_status_container_key_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    if record.get("record_type") == DATABRICKS_RUN_STATUS_RECORD_TYPE:
        return ()
    return _unexpected_keys(record, _DATABRICKS_RUN_STATUS_WRAPPER_KEYS, "Databricks run status sidecar wrapper")


def _databricks_run_status_wrapper_field_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
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


def _unexpected_keys(record: Mapping[str, Any], allowed_keys: frozenset[str], label: str) -> tuple[str, ...]:
    unexpected = sorted(str(key) for key in record if key not in allowed_keys)
    if not unexpected:
        return ()
    return (f"{label} has unsupported keys: {unexpected}",)


def _required_str_field(record: Mapping[str, Any], field_name: str, label: str) -> tuple[str, ...]:
    value = record.get(field_name)
    if isinstance(value, str) and value:
        return ()
    return (f"{label}.{field_name} must be a non-empty string",)


def _optional_str_field(record: Mapping[str, Any], field_name: str, label: str) -> tuple[str, ...]:
    value = record.get(field_name)
    if value is None or isinstance(value, str):
        return ()
    return (f"{label}.{field_name} must be a string or null",)


def _optional_int_field(record: Mapping[str, Any], field_name: str, label: str) -> tuple[str, ...]:
    value = record.get(field_name)
    if value is None or type(value) is int:
        return ()
    return (f"{label}.{field_name} must be an integer or null",)


def _bool_field(record: Mapping[str, Any], field_name: str, label: str) -> tuple[str, ...]:
    if type(record.get(field_name)) is bool:
        return ()
    return (f"{label}.{field_name} must be boolean",)


def _run_id_field_issues(record: Mapping[str, Any], field_name: str, label: str) -> tuple[str, ...]:
    value = record.get(field_name)
    if (type(value) is int and value >= 0) or (isinstance(value, str) and value):
        return ()
    return (f"{label}.{field_name} must be a non-negative integer or non-empty string",)


def _list_of_strings_field(record: Mapping[str, Any], field_name: str, label: str) -> tuple[str, ...]:
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
            if isinstance(task, Mapping) and isinstance(task.get(field_name), str) and task[field_name]
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
    return _sorted_unique_texts(_safe_spark_env_key_name(key) for key in spark_env_vars.keys())


def _safe_spark_env_key_name(value: str) -> str:
    if isinstance(value, str) and _DATABRICKS_PAT_TOKEN_RE.search(value):
        return _REDACTED_SPARK_ENV_TOKEN_KEY
    return value


def _spark_env_key_issues(values: Sequence[str], label: str) -> tuple[str, ...]:
    issues: list[str] = []
    for value in values:
        if value == _REDACTED_SPARK_ENV_TOKEN_KEY:
            issues.append(f"{label}.spark_env_keys contains redacted Databricks token-pattern environment variable name")
            continue
        if _DATABRICKS_PAT_TOKEN_RE.search(value):
            issues.append(f"{label}.spark_env_keys contains Databricks token-pattern environment variable name")
            continue
        if _SPARK_ENV_VAR_KEY_RE.fullmatch(value) is None:
            issues.append(f"{label}.spark_env_keys contains invalid environment variable name {value!r}")
        if _looks_secret_like_spark_env_key(value):
            issues.append(f"{label}.spark_env_keys contains secret-looking environment variable name {value!r}")
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
        if isinstance(task, Mapping) and isinstance(task.get("task_key"), str) and task["task_key"]
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
) -> tuple[tuple[dict[str, Any], dict[str, Any]], ...]:
    artifact_pairs = tuple(artifacts)
    if not artifact_pairs:
        raise ValueError("stage-and-submit requires at least one artifact")
    if require_payload_dbfs_artifacts:
        _validate_payload_dbfs_artifacts_are_staged(payload, artifact_pairs)
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
    payload, metadata = _databricks_dbfs_put_payload(local_path, dbfs_path, overwrite=overwrite)
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


def _databricks_dbfs_file_metadata(local_path: str | Path, dbfs_path: str) -> dict[str, Any]:
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
        raise ValueError("dbfs_path must be a non-empty absolute DBFS path or dbfs:/ URI")
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
        _canonical_dbfs_uri(dbfs_path)
        for _local_path, dbfs_path in artifacts
    }
    missing = tuple(
        uri
        for uri in _submit_payload_dbfs_uris(payload)
        if uri not in staged_uris
    )
    if missing:
        raise ValueError(
            "Databricks submit payload references DBFS URIs without staged artifacts: "
            + ", ".join(missing)
        )


def _stage_and_submit_artifact_upload_record(upload_record: Mapping[str, Any]) -> dict[str, Any]:
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
            "contents_base64_bytes": len(str(upload_payload.get("contents", "")).encode("ascii")),
        },
    }


def _databricks_request(
    config: DatabricksWorkspaceConfig,
    method: str,
    path_and_query: str,
    *,
    payload: dict[str, Any] | None,
) -> urllib.request.Request:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    return urllib.request.Request(
        f"{config.normalized_host}{path_and_query}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {config.token}",
            "Content-Type": "application/json",
        },
    )


def _format_databricks_http_error(status_code: int, body: str, *, token: str | None = None) -> str:
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


def _success_record(action: str, response: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "ok": True,
        "action": action,
    }
    if response is not None:
        result["response"] = response
    return result


def _write_error_record_or_stdout(result: dict[str, Any], output_json: str | None) -> None:
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
    parser.add_argument("--profile", help="Databricks profile name from ~/.databrickscfg.")
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
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_DATABRICKS_TIMEOUT_SECONDS)
    parser.add_argument("--output-json", help="Write the command result JSON to this path instead of stdout.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="POST a Jobs runs/submit payload JSON.")
    submit_parser.add_argument("--payload-json", required=True)

    subparsers.add_parser(
        "auth-check",
        help="GET the current Databricks workspace user endpoint to verify credentials without launching a run.",
    )

    get_parser = subparsers.add_parser("get", help="GET a Databricks run by run id.")
    get_parser.add_argument("--run-id", required=True)
    get_parser.add_argument("--summary", action="store_true", help="Write only a compact run/task status summary.")
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
    put_parser = subparsers.add_parser("put-dbfs-file", help="Upload a small local artifact to DBFS.")
    put_parser.add_argument("--local-path", required=True, help="Local file to upload.")
    put_parser.add_argument("--dbfs-path", required=True, help="Destination path such as dbfs:/FileStore/cachet/file.whl.")
    put_parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing DBFS file.")
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
    stage_submit_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing DBFS artifacts.")
    stage_submit_parser.add_argument(
        "--require-payload-dbfs-artifacts",
        action="store_true",
        help="Fail before uploading unless every dbfs:/ URI in the payload has a matching --artifact destination.",
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
                raise ValueError("--expected-hardware-target requires --submit-payload-json")
            if args.include_response:
                raise ValueError("--expected-hardware-target cannot be combined with --include-response")
        if args.command == "stage-and-submit" and args.dry_run:
            result = plan_databricks_stage_and_submit(
                read_databricks_run_submit_payload(args.payload_json),
                tuple(_parse_dbfs_artifact_mapping(artifact) for artifact in args.artifact),
                overwrite=args.overwrite,
                require_payload_dbfs_artifacts=args.require_payload_dbfs_artifacts,
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
                )
            result = _success_record(args.command)
            result["summary"] = summary
        else:
            config = _databricks_workspace_config_from_args(args)
            if args.command == "submit":
                response = submit_databricks_run(config, read_databricks_run_submit_payload(args.payload_json))
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
                    tuple(_parse_dbfs_artifact_mapping(artifact) for artifact in args.artifact),
                    overwrite=args.overwrite,
                    require_payload_dbfs_artifacts=args.require_payload_dbfs_artifacts,
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
            result = _success_record(args.command, response if args.include_response else None)
            summary = summarize_databricks_run(
                response,
                submit_payload=submit_payload,
                submit_payload_path=args.submit_payload_json,
            )
            if args.expected_hardware_target:
                validate_databricks_run_status_sidecar(
                    summary,
                    expected_hardware_target=args.expected_hardware_target,
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
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes, bytearray)) or not tasks:
        raise ValueError("Databricks run-submit payload tasks must be a non-empty array")
    invalid_indices = [
        str(index)
        for index, task in enumerate(tasks)
        if not isinstance(task, Mapping)
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
) -> None:
    issues = _databricks_submit_payload_sidecar_issues(
        summary,
        tasks=summary.get("tasks"),
        expected_hardware_target=expected_hardware_target,
    )
    if issues:
        raise ValueError("; ".join(_dedupe_strings(issues)))


def _databricks_workspace_config_from_args(args: argparse.Namespace) -> DatabricksWorkspaceConfig:
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
