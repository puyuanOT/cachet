"""Benchmark plan execution helpers for Document KV Cache."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_kv_cache.storage import local_path


BENCHMARK_PLAN_EXECUTION_RECORD_TYPE = "document_kv.benchmark_plan_execution.v1"
BENCHMARK_PLAN_SOURCE_RECORD_TYPE = "document_kv.benchmark_plan_source.v1"
_PRELOADED_PLAN_PAYLOAD: ContextVar[tuple[str, bytes] | None] = ContextVar(
    "_PRELOADED_PLAN_PAYLOAD",
    default=None,
)

__all__ = [
    "BENCHMARK_PLAN_EXECUTION_RECORD_TYPE",
    "BENCHMARK_PLAN_SOURCE_RECORD_TYPE",
    "BenchmarkCommandResult",
    "execute_benchmark_job_plan",
    "execute_benchmark_job_plan_json",
    "benchmark_command_results_to_record",
    "benchmark_plan_source_to_record",
    "benchmark_plan_source_payload_to_record",
    "write_benchmark_command_results_json",
    "main",
]


@dataclass(frozen=True, slots=True)
class BenchmarkCommandResult:
    name: str
    argv: tuple[str, ...]
    returncode: int
    skipped: bool = False
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.argv:
            raise ValueError("argv must be non-empty")

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.error is None


def execute_benchmark_job_plan(
    plan: Mapping[str, Any],
    *,
    dry_run: bool = False,
    cwd: str | Path | None = None,
) -> tuple[BenchmarkCommandResult, ...]:
    commands = tuple(_commands_from_plan(plan))
    results: list[BenchmarkCommandResult] = []
    for name, argv in commands:
        runtime_argv = _runtime_argv(argv)
        if dry_run:
            results.append(BenchmarkCommandResult(name=name, argv=runtime_argv, returncode=0, skipped=True))
            continue
        try:
            completed = subprocess.run(runtime_argv, cwd=str(cwd) if cwd is not None else None, check=False)
        except OSError as exc:
            results.append(BenchmarkCommandResult(name=name, argv=runtime_argv, returncode=127, error=str(exc)))
            break
        results.append(BenchmarkCommandResult(name=name, argv=runtime_argv, returncode=completed.returncode))
        if completed.returncode != 0:
            break
    return tuple(results)


def execute_benchmark_job_plan_json(
    path: str | Path,
    *,
    dry_run: bool = False,
    cwd: str | Path | None = None,
) -> tuple[BenchmarkCommandResult, ...]:
    return execute_benchmark_job_plan(_plan_from_payload(_plan_payload_for(path)), dry_run=dry_run, cwd=cwd)


def benchmark_command_results_to_record(
    results: Sequence[BenchmarkCommandResult],
    *,
    plan_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "record_type": BENCHMARK_PLAN_EXECUTION_RECORD_TYPE,
        "ok": all(result.ok for result in results),
        "commands": [
            {
                "name": result.name,
                "argv": list(result.argv),
                "returncode": result.returncode,
                "skipped": result.skipped,
                "error": result.error,
            }
            for result in results
        ],
    }
    if plan_source is not None:
        record["plan_source"] = dict(plan_source)
    return record


def benchmark_plan_source_to_record(path: str | Path) -> dict[str, Any]:
    path_text = str(path)
    payload = _read_plan_payload(path)
    return benchmark_plan_source_payload_to_record(path_text, _driver_path(path), payload)


def benchmark_plan_source_payload_to_record(path: str, driver_path: str | Path, payload: bytes) -> dict[str, Any]:
    record: dict[str, Any] = {
        "record_type": BENCHMARK_PLAN_SOURCE_RECORD_TYPE,
        "path": path,
        "driver_path": str(driver_path),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return record
    if isinstance(parsed, Mapping):
        for field_name in ("plan_version", "suite_id", "model_id", "hardware_target"):
            if isinstance(parsed.get(field_name), str) and parsed[field_name]:
                record[field_name] = parsed[field_name]
        commands = parsed.get("commands")
        if isinstance(commands, Sequence) and not isinstance(commands, (str, bytes, bytearray)):
            record["command_count"] = len(commands)
    return record


def write_benchmark_command_results_json(
    results: Sequence[BenchmarkCommandResult],
    path: str | Path,
    *,
    plan_source: Mapping[str, Any] | None = None,
) -> None:
    output_path = _driver_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = _benchmark_command_results_to_record(results, plan_source=plan_source)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )


def _benchmark_command_results_to_record(
    results: Sequence[BenchmarkCommandResult],
    *,
    plan_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if _accepts_plan_source(benchmark_command_results_to_record):
        record = benchmark_command_results_to_record(results, plan_source=plan_source)
    else:
        record = benchmark_command_results_to_record(results)
    _ensure_plan_source(record, plan_source)
    return record


def _accepts_plan_source(function) -> bool:
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return True
    return "plan_source" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _ensure_plan_source(record: dict[str, Any], plan_source: Mapping[str, Any] | None) -> None:
    if plan_source is not None:
        record.setdefault("plan_source", dict(plan_source))


def _write_benchmark_command_results_json(
    results: Sequence[BenchmarkCommandResult],
    path: str | Path,
    *,
    plan_source: Mapping[str, Any] | None = None,
) -> None:
    if _accepts_plan_source(write_benchmark_command_results_json):
        write_benchmark_command_results_json(results, path, plan_source=plan_source)
        return
    write_benchmark_command_results_json(results, path)
    _patch_result_json_plan_source(path, plan_source)


def _patch_result_json_plan_source(path: str | Path, plan_source: Mapping[str, Any] | None) -> None:
    if plan_source is None:
        return
    output_path = _driver_path(path)
    record = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("command execution result JSON root must be an object")
    _ensure_plan_source(record, plan_source)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def _commands_from_plan(plan: Mapping[str, Any]) -> list[tuple[str, tuple[str, ...]]]:
    raw_commands = plan.get("commands")
    if not isinstance(raw_commands, Sequence) or isinstance(raw_commands, (str, bytes)):
        raise ValueError("Benchmark plan JSON must include a commands array")
    commands: list[tuple[str, tuple[str, ...]]] = []
    for index, raw_command in enumerate(raw_commands):
        if not isinstance(raw_command, Mapping):
            raise ValueError(f"commands[{index}] must be an object")
        name = raw_command.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"commands[{index}].name must be non-empty")
        raw_argv = raw_command.get("argv")
        if not isinstance(raw_argv, Sequence) or isinstance(raw_argv, (str, bytes)) or not raw_argv:
            raise ValueError(f"commands[{index}].argv must be a non-empty array")
        argv = tuple(_argv_item(item, command_index=index, item_index=item_index) for item_index, item in enumerate(raw_argv))
        commands.append((name, argv))
    return commands


def _runtime_argv(argv: tuple[str, ...]) -> tuple[str, ...]:
    if argv[0] in {"python", "python3"}:
        return (sys.executable, *argv[1:])
    return argv


def _argv_item(value: Any, *, command_index: int, item_index: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"commands[{command_index}].argv[{item_index}] must be a non-empty string")
    return value


def _driver_path(path: str | Path) -> Path:
    return local_path(str(path))


def _read_plan_payload(path: str | Path) -> bytes:
    return _driver_path(path).read_bytes()


def _plan_payload_for(path: str | Path) -> bytes:
    preloaded = _PRELOADED_PLAN_PAYLOAD.get()
    path_text = str(path)
    if preloaded is not None and preloaded[0] == path_text:
        return preloaded[1]
    return _read_plan_payload(path)


def _plan_from_payload(payload: bytes) -> Mapping[str, Any]:
    plan = json.loads(payload.decode("utf-8"))
    if not isinstance(plan, Mapping):
        raise ValueError("Benchmark plan JSON root must be an object")
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute a V1 benchmark command plan JSON.")
    parser.add_argument("--plan-json", required=True, help="Benchmark plan JSON produced by benchmark_plan.")
    parser.add_argument("--cwd", help="Optional working directory for all plan commands.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report commands without running them.")
    parser.add_argument("--result-json", help="Optional path for command execution result JSON.")
    args = parser.parse_args(argv)

    try:
        plan_payload = _read_plan_payload(args.plan_json)
        plan_source = benchmark_plan_source_payload_to_record(args.plan_json, _driver_path(args.plan_json), plan_payload)
        token = _PRELOADED_PLAN_PAYLOAD.set((args.plan_json, plan_payload))
        try:
            results = execute_benchmark_job_plan_json(args.plan_json, dry_run=args.dry_run, cwd=args.cwd)
        finally:
            _PRELOADED_PLAN_PAYLOAD.reset(token)
        record = _benchmark_command_results_to_record(results, plan_source=plan_source)
        if args.result_json:
            _write_benchmark_command_results_json(results, args.result_json, plan_source=plan_source)
        else:
            print(json.dumps(record, indent=2, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, sort_keys=True))
        return 1
    return 0 if record["ok"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
