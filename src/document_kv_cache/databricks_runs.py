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
        "aws_g5_node_type": bool(task_summaries)
        and all(_is_aws_g5_node_type(summary.get("node_type_id")) for summary in task_summaries)
        and all(_is_aws_g5_node_type(summary.get("driver_node_type_id")) for summary in task_summaries),
    }


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


def _is_aws_g5_node_type(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("g5.")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
        help="Attach a sanitized hash and AWS g5 cluster summary for the runs/submit payload that launched this run.",
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
