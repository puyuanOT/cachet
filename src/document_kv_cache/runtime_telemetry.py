from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

from document_kv_cache._benchmark_manifest import (
    _resource_execution_id_digest,
    _resource_runtime_identity_digest,
    _resource_software_identity,
    _sha256_json,
)
from document_kv_cache._benchmark_models import (
    BenchmarkResourceEvidence,
    BenchmarkRunResult,
)
from document_kv_cache._benchmark_records import (
    benchmark_gate_inputs_from_record,
    benchmark_resource_evidence_to_record,
    benchmark_run_result_from_record,
    benchmark_run_result_to_record,
)


RUNTIME_TELEMETRY_RECORD_TYPE = "document_kv.runtime_telemetry.v1"

__all__ = [
    "RUNTIME_TELEMETRY_RECORD_TYPE",
    "RuntimeTelemetrySampler",
    "attach_runtime_resource_evidence",
    "attach_runtime_resource_evidence_file",
    "benchmark_resource_evidence_from_runtime_telemetry",
    "bind_runtime_resource_evidence_record",
    "bind_runtime_resource_evidence_record_file",
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
    raw_pids = process_tree.get("pids")
    process_pids = (
        tuple(pid for pid in raw_pids if type(pid) is int)
        if isinstance(raw_pids, Sequence)
        else ()
    )
    gpu = _nvidia_smi_sample(
        command_runner=command_runner,
        process_pids=process_pids,
    )
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
    peak_gpu_process_memory_bytes = max(
        (
            int(processes.get("process_tree_used_bytes", 0))
            for sample in sample_list
            for gpu in (sample.get("gpu"),)
            if isinstance(gpu, Mapping)
            for processes in (gpu.get("processes"),)
            if isinstance(processes, Mapping)
            and processes.get("process_tree_used_bytes") is not None
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
    gpu_utilization_values = [
        float(gpu["utilization_percent"])
        for sample in sample_list
        for gpu in _gpu_rows(sample)
        if gpu.get("utilization_percent") is not None
    ]
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
        "peak_gpu_process_memory_bytes": peak_gpu_process_memory_bytes,
        "mean_gpu_utilization_percent": (
            sum(gpu_utilization_values) / len(gpu_utilization_values)
            if gpu_utilization_values
            else None
        ),
        "peak_gpu_utilization_percent": peak_gpu_utilization_percent,
        "peak_host_memory_used_bytes": peak_host_memory_used_bytes,
        "errors": [dict(error) for error in errors],
    }


def benchmark_resource_evidence_from_runtime_telemetry(
    result: BenchmarkRunResult,
    *,
    arm_id: str,
    telemetry: Mapping[str, Any],
    source_revision: str,
    source_tree_sha256: str,
    wheel_sha256: str,
    runner_sha256: str,
    telemetry_sha256: str | None = None,
) -> BenchmarkResourceEvidence:
    """Reduce raw telemetry to one manifest-bound arm measurement record.

    Samples are selected strictly inside the arm's timestamped execution window.
    Server startup, handoff generation, and post-run teardown are therefore not
    reinterpreted as online serving resource measurements.
    """

    if not isinstance(result, BenchmarkRunResult):
        raise TypeError("result must be a BenchmarkRunResult")
    manifest = result.experiment_manifest
    if manifest is None:
        raise ValueError("resource evidence requires an experiment manifest")
    if not isinstance(telemetry, Mapping):
        raise TypeError("telemetry must be a mapping")
    if telemetry.get("record_type") != RUNTIME_TELEMETRY_RECORD_TYPE:
        raise ValueError("unsupported runtime telemetry record_type")
    interval = _positive_finite_number(
        telemetry.get("interval_seconds"),
        "telemetry.interval_seconds",
    )
    windows = [window for window in result.execution_windows if window.arm_id == arm_id]
    if len(windows) != 1:
        raise ValueError(
            f"resource evidence requires exactly one execution window for arm {arm_id!r}"
        )
    window = windows[0]
    if window.started_at_seconds is None or window.ended_at_seconds is None:
        raise ValueError(
            "resource evidence cannot be attached to an un-timestamped historical run"
        )
    started_at = window.started_at_seconds
    ended_at = window.ended_at_seconds
    raw_samples = telemetry.get("samples")
    if not isinstance(raw_samples, Sequence) or isinstance(
        raw_samples,
        (str, bytes, bytearray),
    ):
        raise ValueError("telemetry.samples must be an array")
    selected: list[Mapping[str, Any]] = []
    for index, raw_sample in enumerate(raw_samples):
        if not isinstance(raw_sample, Mapping):
            raise ValueError(f"telemetry.samples[{index}] must be an object")
        timestamp = _non_negative_finite_number(
            raw_sample.get("timestamp_seconds"),
            f"telemetry.samples[{index}].timestamp_seconds",
        )
        if started_at <= timestamp <= ended_at:
            selected.append(raw_sample)
    if not selected:
        raise ValueError(
            f"runtime telemetry has no samples inside arm {arm_id!r}'s measurement window"
        )
    selected.sort(key=lambda item: float(item["timestamp_seconds"]))
    timestamps = [float(sample["timestamp_seconds"]) for sample in selected]
    gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
    max_gap = max(gaps, default=0.0)
    expected_sample_count = max(1, math.floor((ended_at - started_at) / interval))
    errors = telemetry.get("errors", ())
    if not isinstance(errors, Sequence) or isinstance(
        errors,
        (str, bytes, bytearray),
    ):
        raise ValueError("telemetry.errors must be an array")
    error_count = 0
    for raw_error in errors:
        if not isinstance(raw_error, Mapping):
            error_count += 1
            continue
        raw_timestamp = raw_error.get("timestamp_seconds")
        if raw_timestamp is None:
            error_count += 1
            continue
        try:
            error_timestamp = float(raw_timestamp)
        except (TypeError, ValueError):
            error_count += 1
            continue
        if started_at <= error_timestamp <= ended_at:
            error_count += 1

    process_gpu_memory: list[int] = []
    gpu_utilization: list[float] = []
    process_rss: list[int] = []
    host_memory: list[int] = []
    invalid_samples = 0
    for sample in selected:
        sample_valid = True
        process_tree = sample.get("process_tree")
        if not isinstance(process_tree, Mapping) or process_tree.get("ok") is not True:
            sample_valid = False
        else:
            rss = _non_negative_int_or_none(process_tree.get("rss_bytes"))
            if rss is None:
                sample_valid = False
            else:
                process_rss.append(rss)
        host = sample.get("host_memory")
        if not isinstance(host, Mapping) or host.get("ok") is not True:
            sample_valid = False
        else:
            used = _non_negative_int_or_none(host.get("used_bytes"))
            if used is None:
                sample_valid = False
            else:
                host_memory.append(used)
        gpu = sample.get("gpu")
        if not isinstance(gpu, Mapping) or gpu.get("ok") is not True:
            sample_valid = False
        else:
            devices = gpu.get("devices")
            if not isinstance(devices, Sequence) or not devices:
                sample_valid = False
            else:
                per_sample_utilization: list[float] = []
                for device in devices:
                    if not isinstance(device, Mapping) or device.get("ok") is not True:
                        sample_valid = False
                        continue
                    utilization = _percentage_or_none(
                        device.get("utilization_percent")
                    )
                    if utilization is None:
                        sample_valid = False
                    else:
                        per_sample_utilization.append(utilization)
                gpu_utilization.extend(per_sample_utilization)
            processes = gpu.get("processes")
            if not isinstance(processes, Mapping) or processes.get("ok") is not True:
                sample_valid = False
            else:
                memory = _non_negative_int_or_none(
                    processes.get("process_tree_used_bytes")
                )
                if memory is None:
                    sample_valid = False
                else:
                    process_gpu_memory.append(memory)
        if not sample_valid:
            invalid_samples += 1
    error_count += invalid_samples
    complete = (
        len(selected) >= expected_sample_count
        and error_count == 0
        and timestamps[0] <= started_at + interval
        and timestamps[-1] >= ended_at - interval
        and max_gap <= interval * 2
        and len(process_gpu_memory) == len(selected)
        and len(process_rss) == len(selected)
        and len(host_memory) == len(selected)
        and bool(gpu_utilization)
    )
    execution_id_digest = _resource_execution_id_digest(manifest, arm_id)
    runtime_identity_sha256 = _resource_runtime_identity_digest(
        manifest,
        arm_id,
        execution_id_digest=execution_id_digest,
    )
    declared_software_identity = _resource_software_identity(manifest)
    supplied_software_identity = {
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "wheel_sha256": wheel_sha256,
        "runner_sha256": runner_sha256,
    }
    if supplied_software_identity != declared_software_identity:
        raise ValueError(
            "resource software identity does not match manifest package_revisions"
        )
    return BenchmarkResourceEvidence(
        experiment_id=manifest.experiment_id,
        arm_id=arm_id,
        execution_id_digest=execution_id_digest,
        measurement_started_at_seconds=started_at,
        measurement_ended_at_seconds=ended_at,
        sampling_interval_seconds=interval,
        first_sample_at_seconds=timestamps[0],
        last_sample_at_seconds=timestamps[-1],
        max_sample_gap_seconds=max_gap,
        expected_sample_count=expected_sample_count,
        sample_count=len(selected),
        error_count=error_count,
        complete=complete,
        telemetry_sha256=(
            _sha256_json(telemetry)
            if telemetry_sha256 is None
            else telemetry_sha256
        ),
        peak_gpu_process_memory_bytes=max(process_gpu_memory, default=0),
        mean_gpu_utilization_percent=(
            sum(gpu_utilization) / len(gpu_utilization)
            if gpu_utilization
            else 0.0
        ),
        peak_gpu_utilization_percent=max(gpu_utilization, default=0.0),
        peak_process_tree_rss_bytes=max(process_rss, default=0),
        peak_host_memory_used_bytes=max(host_memory, default=0),
        source_revision=source_revision,
        source_tree_sha256=source_tree_sha256,
        wheel_sha256=wheel_sha256,
        runner_sha256=runner_sha256,
        runtime_identity_sha256=runtime_identity_sha256,
    )


def attach_runtime_resource_evidence(
    result: BenchmarkRunResult,
    *,
    arm_id: str,
    telemetry: Mapping[str, Any],
    source_revision: str,
    source_tree_sha256: str,
    wheel_sha256: str,
    runner_sha256: str,
    telemetry_sha256: str | None = None,
) -> BenchmarkRunResult:
    """Return a result with one authenticated resource record attached."""

    evidence = benchmark_resource_evidence_from_runtime_telemetry(
        result,
        arm_id=arm_id,
        telemetry=telemetry,
        source_revision=source_revision,
        source_tree_sha256=source_tree_sha256,
        wheel_sha256=wheel_sha256,
        runner_sha256=runner_sha256,
        telemetry_sha256=telemetry_sha256,
    )
    manifest = result.experiment_manifest
    if manifest is None:  # pragma: no cover - checked by the builder above.
        raise ValueError("resource evidence requires an experiment manifest")
    evidence_by_arm = {
        item.arm_id: item
        for item in result.resource_evidence
        if item.arm_id != arm_id
    }
    evidence_by_arm[arm_id] = evidence
    ordered_evidence = tuple(
        evidence_by_arm[key] for key in sorted(evidence_by_arm)
    )
    evidence_ids = tuple(
        (
            item.arm_id,
            str(benchmark_resource_evidence_to_record(item)["record_sha256"]),
        )
        for item in ordered_evidence
    )
    return replace(
        result,
        experiment_manifest=replace(
            manifest,
            resource_evidence_ids=evidence_ids,
        ),
        resource_evidence=ordered_evidence,
    )


def attach_runtime_resource_evidence_file(
    result: BenchmarkRunResult,
    *,
    arm_id: str,
    telemetry_path: str | Path,
    source_revision: str,
    source_tree_sha256: str,
    wheel_sha256: str,
    runner_sha256: str,
) -> BenchmarkRunResult:
    """Attach telemetry while preserving the exact sidecar-file byte digest."""

    path = Path(telemetry_path)
    raw = path.read_bytes()
    try:
        telemetry = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"runtime telemetry JSON is invalid: {exc.msg}") from exc
    if not isinstance(telemetry, Mapping):
        raise ValueError("runtime telemetry JSON must contain an object")
    return attach_runtime_resource_evidence(
        result,
        arm_id=arm_id,
        telemetry=telemetry,
        source_revision=source_revision,
        source_tree_sha256=source_tree_sha256,
        wheel_sha256=wheel_sha256,
        runner_sha256=runner_sha256,
        telemetry_sha256=sha256(raw).hexdigest(),
    )


def bind_runtime_resource_evidence_record(
    record: Mapping[str, Any],
    *,
    telemetry: Mapping[str, Any],
    telemetry_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind finalized telemetry to every arm and recompute the evidence gate.

    A benchmark runner necessarily writes its record before the surrounding
    process sampler is stopped.  This function is the governed second phase:
    it reconstructs the typed result, attaches only samples inside each arm's
    timestamped execution window, and serializes a new payload and gate.  It
    never upgrades records that did not declare the resource scope.
    """

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    if not isinstance(telemetry, Mapping):
        raise TypeError("telemetry must be a mapping")
    result = benchmark_run_result_from_record(record)
    manifest = result.experiment_manifest
    if manifest is None or "resource" not in manifest.measurement_scopes:
        return dict(record)

    software_identity = _resource_software_identity(manifest)
    attached = result
    for arm in manifest.arms:
        attached = attach_runtime_resource_evidence(
            attached,
            arm_id=arm.arm_id,
            telemetry=telemetry,
            telemetry_sha256=telemetry_sha256,
            **software_identity,
        )

    artifact_identities, cache_state_attestations = (
        benchmark_gate_inputs_from_record(record)
    )
    return benchmark_run_result_to_record(
        attached,
        artifact_identities=artifact_identities,
        cache_state_attestations=cache_state_attestations,
        sanitize_evidence=record.get("evidence_sanitized") is True,
    )


def bind_runtime_resource_evidence_record_file(
    benchmark_path: str | Path,
    telemetry_path: str | Path,
) -> dict[str, Any]:
    """Atomically finalize a benchmark file after its telemetry sidecar exists."""

    benchmark_file = Path(benchmark_path)
    telemetry_file = Path(telemetry_path)
    benchmark_raw = benchmark_file.read_bytes()
    telemetry_raw = telemetry_file.read_bytes()
    try:
        record = json.loads(benchmark_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"benchmark JSON is invalid: {exc.msg}") from exc
    try:
        telemetry = json.loads(telemetry_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"runtime telemetry JSON is invalid: {exc.msg}") from exc
    if not isinstance(record, Mapping):
        raise ValueError("benchmark JSON must contain an object")
    if not isinstance(telemetry, Mapping):
        raise ValueError("runtime telemetry JSON must contain an object")

    bound = bind_runtime_resource_evidence_record(
        record,
        telemetry=telemetry,
        telemetry_sha256=sha256(telemetry_raw).hexdigest(),
    )
    temporary_path = benchmark_file.with_name(
        f".{benchmark_file.name}.resource-evidence.tmp"
    )
    temporary_path.write_text(
        json.dumps(bound, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(benchmark_file)
    return bound


def _gpu_rows(sample: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    gpu = sample.get("gpu")
    if not isinstance(gpu, Mapping):
        return []
    rows = gpu.get("devices")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _nvidia_smi_sample(
    *,
    command_runner: CommandRunner,
    process_pids: Sequence[int] = (),
) -> dict[str, Any]:
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
    process_argv = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
        "--format=csv,noheader,nounits",
    ]
    process_completed = _run_command(process_argv, command_runner=command_runner)
    process_rows: list[dict[str, Any]] = []
    process_ok = process_completed.returncode == 0
    if process_ok:
        for line in process_completed.stdout.splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                process_ok = False
                process_rows.append(
                    {"ok": False, "error": "unexpected compute-app row shape"}
                )
                continue
            pid, used_mib, gpu_uuid = parts
            process_rows.append(
                {
                    "ok": True,
                    "pid": _int_or_none(pid),
                    "memory_used_bytes": _mib_to_bytes(used_mib),
                    "gpu_uuid": gpu_uuid,
                }
            )
    process_pid_set = set(process_pids)
    process_tree_values = [
        row["memory_used_bytes"]
        for row in process_rows
        if row.get("ok") is True
        and row.get("pid") in process_pid_set
        and row.get("memory_used_bytes") is not None
    ]
    return {
        "ok": True,
        "command": argv,
        "devices": devices,
        "processes": {
            "ok": process_ok,
            "command": process_argv,
            "rows": process_rows,
            "process_tree_pids": sorted(process_pid_set),
            "process_tree_used_bytes": (
                sum(process_tree_values) if process_ok else None
            ),
            "error": (
                None if process_ok else _command_error(process_completed)
            ),
        },
    }


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


def _non_negative_int_or_none(value: object) -> int | None:
    parsed = _int_or_none(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def _percentage_or_none(value: object) -> float | None:
    parsed = _float_or_none(value)
    if parsed is None or not math.isfinite(parsed) or not 0 <= parsed <= 100:
        return None
    return parsed


def _non_negative_finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return normalized


def _positive_finite_number(value: object, field_name: str) -> float:
    normalized = _non_negative_finite_number(value, field_name)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be positive")
    return normalized


def _text_or_empty(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
