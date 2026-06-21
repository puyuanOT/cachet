"""Submit, inspect, and summarize Databricks Jobs runs."""

from __future__ import annotations

import argparse
import hashlib
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


__all__ = [
    "DEFAULT_DATABRICKS_HOST_ENV",
    "DEFAULT_DATABRICKS_TOKEN_ENV",
    "DEFAULT_DATABRICKS_TIMEOUT_SECONDS",
    "DATABRICKS_RUN_STATUS_RECORD_TYPE",
    "DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE",
    "DatabricksWorkspaceConfig",
    "databricks_workspace_config_from_env",
    "submit_databricks_run",
    "get_databricks_run",
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
DEFAULT_DATABRICKS_TIMEOUT_SECONDS = 60.0
DATABRICKS_RUN_STATUS_RECORD_TYPE = "document_kv.databricks_run_status.v1"
DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE = "document_kv.databricks_run_submit_payload.v1"
DATABRICKS_TERMINAL_LIFE_CYCLE_STATES = frozenset({"TERMINATED", "SKIPPED", "INTERNAL_ERROR"})
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
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
        "spark_versions",
        "data_security_modes",
        "single_node",
        "aws_single_node_gpu_type",
        "aws_g5_node_type",
    }
)
_DATABRICKS_SUBMIT_PAYLOAD_TASK_KEYS = frozenset(
    {
        "task_key",
        "node_type_id",
        "driver_node_type_id",
        "spark_version",
        "data_security_mode",
        "num_workers",
        "single_node",
        "purpose",
    }
)


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
    spark_versions = _sorted_unique_texts(summary.get("spark_version") for summary in task_summaries)
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
        "spark_versions": spark_versions,
        "data_security_modes": data_security_modes,
        "single_node": bool(task_summaries) and all(summary["single_node"] for summary in task_summaries),
        "aws_single_node_gpu_type": aws_single_node_gpu_type,
        "aws_g5_node_type": aws_single_node_gpu_type,
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
    request = _databricks_request(config, method, path_and_query, payload=payload)
    try:
        with opener(request, timeout=config.timeout_seconds) as response:
            body = response.read().decode("utf-8")
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
    return parsed


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
    }


def _submit_payload_task_summary(task: Mapping[str, Any]) -> dict[str, Any]:
    cluster = _mapping(task.get("new_cluster"))
    custom_tags = _mapping(cluster.get("custom_tags"))
    return {
        "task_key": task.get("task_key"),
        "node_type_id": _optional_str(cluster.get("node_type_id")),
        "driver_node_type_id": _optional_str(cluster.get("driver_node_type_id")),
        "spark_version": _optional_str(cluster.get("spark_version")),
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


_SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES = ("g6.",)
_HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES = {
    "aws-g6-l4": ("g6.",),
}


def _is_supported_aws_single_node_gpu_type(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(_SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES)


def _is_expected_aws_single_node_gpu_type(value: Any, expected_hardware_target: str | None) -> bool:
    if expected_hardware_target is None:
        return _is_supported_aws_single_node_gpu_type(value)
    prefixes = _HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES.get(expected_hardware_target)
    return isinstance(value, str) and prefixes is not None and value.startswith(prefixes)


def _submit_payload_gpu_type_supported(record: Mapping[str, Any]) -> bool:
    return all(record[field_name] is True for field_name in _present_gpu_type_fields(record))


def _present_gpu_type_fields(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        field_name
        for field_name in ("aws_single_node_gpu_type", "aws_g5_node_type")
        if field_name in record
    )


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
                    f"Databricks run status sidecar submit_payload.tasks[{index}].{field_name} must be an AWS g6/L4 node type"
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
    issues.extend(_bool_field(record, "single_node", "Databricks run status sidecar submit_payload"))
    if "aws_single_node_gpu_type" not in record and "aws_g5_node_type" not in record:
        issues.append(
            "Databricks run status sidecar submit_payload.aws_single_node_gpu_type or aws_g5_node_type must be present"
        )
    for field_name in ("aws_single_node_gpu_type", "aws_g5_node_type"):
        if field_name in record:
            issues.extend(_bool_field(record, field_name, "Databricks run status sidecar submit_payload"))
    if (
        type(record.get("aws_single_node_gpu_type")) is bool
        and type(record.get("aws_g5_node_type")) is bool
        and record["aws_single_node_gpu_type"] != record["aws_g5_node_type"]
    ):
        issues.append(
            "Databricks run status sidecar submit_payload.aws_single_node_gpu_type and aws_g5_node_type must match"
        )
    return tuple(issues)


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
    parser = argparse.ArgumentParser(description="Submit or inspect Databricks runs using env-provided credentials.")
    parser.add_argument("--host-env", default=DEFAULT_DATABRICKS_HOST_ENV)
    parser.add_argument("--token-env", default=DEFAULT_DATABRICKS_TOKEN_ENV)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_DATABRICKS_TIMEOUT_SECONDS)
    parser.add_argument("--output-json", help="Write the command result JSON to this path instead of stdout.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="POST a Jobs runs/submit payload JSON.")
    submit_parser.add_argument("--payload-json", required=True)

    get_parser = subparsers.add_parser("get", help="GET a Databricks run by run id.")
    get_parser.add_argument("--run-id", required=True)
    get_parser.add_argument("--summary", action="store_true", help="Write only a compact run/task status summary.")
    get_parser.add_argument(
        "--submit-payload-json",
        help="Attach a sanitized hash and AWS g6/L4 cluster summary for the runs/submit payload that launched this run.",
    )
    get_parser.add_argument(
        "--include-response",
        action="store_true",
        help="Also include the raw Jobs API response when using --summary.",
    )

    args = parser.parse_args(argv)
    try:
        config = databricks_workspace_config_from_env(
            host_env=args.host_env,
            token_env=args.token_env,
            timeout_seconds=args.timeout_seconds,
        )
        if args.command == "submit":
            response = submit_databricks_run(config, read_databricks_run_submit_payload(args.payload_json))
        elif args.command == "get":
            response = get_databricks_run(config, args.run_id)
        else:  # pragma: no cover - argparse enforces this.
            raise ValueError(f"unknown command {args.command!r}")
        if args.command == "get" and args.summary:
            submit_payload = (
                read_databricks_run_submit_payload(args.submit_payload_json)
                if args.submit_payload_json
                else None
            )
            result = _success_record(args.command, response if args.include_response else None)
            result["summary"] = summarize_databricks_run(
                response,
                submit_payload=submit_payload,
                submit_payload_path=args.submit_payload_json,
            )
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
