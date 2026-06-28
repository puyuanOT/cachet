from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Any


RUNTIME_TELEMETRY_RECORD_TYPE = "document_kv.runtime_telemetry.v1"

__all__ = [
    "RUNTIME_TELEMETRY_RECORD_TYPE",
    "RuntimeTelemetrySampler",
    "collect_runtime_telemetry_sample",
    "runtime_telemetry_summary",
]


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass
class RuntimeTelemetrySampler:
    output_path: Path
    process_pid: int | None = None
    interval_seconds: float = 1.0
    command_runner: CommandRunner = subprocess.run
    clock: Callable[[], float] = time.time
    sleeper: Callable[[float], None] = time.sleep
    _samples: list[dict[str, Any]] = field(default_factory=list, init=False)
    _errors: list[dict[str, Any]] = field(default_factory=list, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.output_path = Path(self.output_path)

    def start(self) -> "RuntimeTelemetrySampler":
        if self._thread is not None:
            raise RuntimeError("runtime telemetry sampler already started")
        self._sample_once()
        self._thread = threading.Thread(
            target=self._run,
            name="cachet-runtime-telemetry",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_seconds * 3))
        self._sample_once()
        record = runtime_telemetry_summary(
            self._samples,
            process_pid=self.process_pid,
            interval_seconds=self.interval_seconds,
            errors=self._errors,
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return record

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample_once()

    def _sample_once(self) -> None:
        try:
            self._samples.append(
                collect_runtime_telemetry_sample(
                    process_pid=self.process_pid,
                    command_runner=self.command_runner,
                    timestamp_seconds=self.clock(),
                )
            )
        except Exception as exc:
            self._errors.append(
                {
                    "timestamp_seconds": self.clock(),
                    "error_type": type(exc).__name__,
                    "error": str(exc) or type(exc).__name__,
                }
            )


def collect_runtime_telemetry_sample(
    *,
    process_pid: int | None = None,
    command_runner: CommandRunner = subprocess.run,
    timestamp_seconds: float | None = None,
) -> dict[str, Any]:
    timestamp = time.time() if timestamp_seconds is None else timestamp_seconds
    process_tree = _process_tree_sample(process_pid, command_runner=command_runner)
    gpu = _nvidia_smi_sample(command_runner=command_runner)
    host_memory = _host_memory_sample()
    return {
        "timestamp_seconds": timestamp,
        "process_tree": process_tree,
        "gpu": gpu,
        "host_memory": host_memory,
    }


def runtime_telemetry_summary(
    samples: Sequence[Mapping[str, Any]],
    *,
    process_pid: int | None,
    interval_seconds: float,
    errors: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    sample_list = [dict(sample) for sample in samples]
    peak_process_rss_bytes = max(
        (
            int(process_tree.get("rss_bytes", 0))
            for sample in sample_list
            for process_tree in (sample.get("process_tree"),)
            if isinstance(process_tree, Mapping)
        ),
        default=None,
    )
    peak_gpu_memory_used_bytes = max(
        (
            int(gpu.get("memory_used_bytes", 0))
            for sample in sample_list
            for gpu in _gpu_rows(sample)
            if gpu.get("memory_used_bytes") is not None
        ),
        default=None,
    )
    peak_gpu_utilization_percent = max(
        (
            float(gpu.get("utilization_percent", 0.0))
            for sample in sample_list
            for gpu in _gpu_rows(sample)
            if gpu.get("utilization_percent") is not None
        ),
        default=None,
    )
    peak_host_memory_used_bytes = max(
        (
            int(host.get("used_bytes", 0))
            for sample in sample_list
            for host in (sample.get("host_memory"),)
            if isinstance(host, Mapping) and host.get("used_bytes") is not None
        ),
        default=None,
    )
    return {
        "record_type": RUNTIME_TELEMETRY_RECORD_TYPE,
        "ok": True,
        "process_pid": process_pid,
        "interval_seconds": interval_seconds,
        "samples": sample_list,
        "sample_count": len(sample_list),
        "peak_process_tree_rss_bytes": peak_process_rss_bytes,
        "peak_gpu_memory_used_bytes": peak_gpu_memory_used_bytes,
        "peak_gpu_utilization_percent": peak_gpu_utilization_percent,
        "peak_host_memory_used_bytes": peak_host_memory_used_bytes,
        "errors": [dict(error) for error in errors],
    }


def _gpu_rows(sample: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    gpu = sample.get("gpu")
    if not isinstance(gpu, Mapping):
        return []
    rows = gpu.get("devices")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _nvidia_smi_sample(*, command_runner: CommandRunner) -> dict[str, Any]:
    argv = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = _run_command(argv, command_runner=command_runner)
    if completed.returncode != 0:
        return {
            "ok": False,
            "command": argv,
            "error": _command_error(completed),
            "devices": [],
        }
    devices = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            devices.append({"ok": False, "raw": line, "error": "unexpected nvidia-smi row shape"})
            continue
        index, name, used_mib, total_mib, utilization_percent = parts
        devices.append(
            {
                "ok": True,
                "index": _int_or_none(index),
                "name": name,
                "memory_used_bytes": _mib_to_bytes(used_mib),
                "memory_total_bytes": _mib_to_bytes(total_mib),
                "utilization_percent": _float_or_none(utilization_percent),
            }
        )
    return {"ok": True, "command": argv, "devices": devices}


def _process_tree_sample(process_pid: int | None, *, command_runner: CommandRunner) -> dict[str, Any]:
    if process_pid is None:
        return {"ok": False, "error": "process_pid not provided"}
    completed = _run_command(["ps", "-eo", "pid=,ppid=,rss="], command_runner=command_runner)
    if completed.returncode != 0:
        return {"ok": False, "pid": process_pid, "error": _command_error(completed)}
    rows = _parse_ps_rows(completed.stdout)
    descendants = _process_tree_pids(process_pid, rows)
    rss_kib = sum(rows[pid]["rss_kib"] for pid in descendants if pid in rows)
    return {
        "ok": True,
        "pid": process_pid,
        "process_count": len(descendants),
        "pids": sorted(descendants),
        "rss_bytes": rss_kib * 1024,
    }


def _host_memory_sample() -> dict[str, Any]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return {"ok": False, "error": "/proc/meminfo is not available"}
    values: dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, raw_value = line.partition(":")
        if not separator:
            continue
        parts = raw_value.strip().split()
        if not parts:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        multiplier = 1024 if len(parts) > 1 and parts[1].lower() == "kb" else 1
        values[key] = value * multiplier
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    used = total - available if total is not None and available is not None else None
    return {
        "ok": total is not None and available is not None,
        "total_bytes": total,
        "available_bytes": available,
        "used_bytes": used,
    }


def _parse_ps_rows(text: str) -> dict[int, dict[str, int]]:
    rows: dict[int, dict[str, int]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        pid = _int_or_none(parts[0])
        ppid = _int_or_none(parts[1])
        rss_kib = _int_or_none(parts[2])
        if pid is None or ppid is None or rss_kib is None:
            continue
        rows[pid] = {"ppid": ppid, "rss_kib": rss_kib}
    return rows


def _process_tree_pids(root_pid: int, rows: Mapping[int, Mapping[str, int]]) -> set[int]:
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, row in rows.items():
            if pid in descendants:
                continue
            if row.get("ppid") in descendants:
                descendants.add(pid)
                changed = True
    return descendants


def _run_command(argv: Sequence[str], *, command_runner: CommandRunner) -> subprocess.CompletedProcess[str]:
    try:
        return command_runner(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(list(argv), 127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            list(argv),
            124,
            _text_or_empty(exc.stdout),
            _text_or_empty(exc.stderr) or f"timed out after {exc.timeout}s",
        )


def _command_error(completed: subprocess.CompletedProcess[str]) -> str:
    return (completed.stderr or completed.stdout or f"command exited with {completed.returncode}").strip()


def _mib_to_bytes(value: str) -> int | None:
    parsed = _float_or_none(value)
    if parsed is None:
        return None
    return int(parsed * 1024 * 1024)


def _int_or_none(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _text_or_empty(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
