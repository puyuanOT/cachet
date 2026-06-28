import json
import subprocess

from document_kv_cache.runtime_telemetry import (
    RUNTIME_TELEMETRY_RECORD_TYPE,
    RuntimeTelemetrySampler,
    collect_runtime_telemetry_sample,
    runtime_telemetry_summary,
)


def test_collect_runtime_telemetry_sample_parses_gpu_and_process_tree():
    def command_runner(argv, **kwargs):
        if argv[0] == "nvidia-smi":
            return subprocess.CompletedProcess(
                argv,
                0,
                "0, NVIDIA A10G, 1234, 23000, 87\n",
                "",
            )
        if argv[0] == "ps":
            return subprocess.CompletedProcess(
                argv,
                0,
                "\n".join(
                    [
                        "100 1 1000",
                        "101 100 2000",
                        "102 101 3000",
                        "999 1 4000",
                    ]
                ),
                "",
            )
        raise AssertionError(f"unexpected command: {argv}")

    sample = collect_runtime_telemetry_sample(
        process_pid=100,
        command_runner=command_runner,
        timestamp_seconds=123.0,
    )

    assert sample["timestamp_seconds"] == 123.0
    assert sample["process_tree"]["ok"] is True
    assert sample["process_tree"]["pids"] == [100, 101, 102]
    assert sample["process_tree"]["rss_bytes"] == (1000 + 2000 + 3000) * 1024
    assert sample["gpu"]["ok"] is True
    assert sample["gpu"]["devices"][0]["memory_used_bytes"] == 1234 * 1024 * 1024
    assert sample["gpu"]["devices"][0]["memory_total_bytes"] == 23000 * 1024 * 1024
    assert sample["gpu"]["devices"][0]["utilization_percent"] == 87.0
    assert "ok" in sample["host_memory"]


def test_runtime_telemetry_summary_reports_peaks():
    samples = [
        {
            "process_tree": {"rss_bytes": 10},
            "gpu": {"devices": [{"memory_used_bytes": 100, "utilization_percent": 50.0}]},
            "host_memory": {"used_bytes": 1000},
        },
        {
            "process_tree": {"rss_bytes": 20},
            "gpu": {"devices": [{"memory_used_bytes": 90, "utilization_percent": 75.0}]},
            "host_memory": {"used_bytes": 900},
        },
    ]

    record = runtime_telemetry_summary(
        samples,
        process_pid=42,
        interval_seconds=2.0,
        errors=({"error": "transient"},),
    )

    assert record["record_type"] == RUNTIME_TELEMETRY_RECORD_TYPE
    assert record["sample_count"] == 2
    assert record["process_pid"] == 42
    assert record["interval_seconds"] == 2.0
    assert record["peak_process_tree_rss_bytes"] == 20
    assert record["peak_gpu_memory_used_bytes"] == 100
    assert record["peak_gpu_utilization_percent"] == 75.0
    assert record["peak_host_memory_used_bytes"] == 1000
    assert record["errors"] == [{"error": "transient"}]


def test_runtime_telemetry_sampler_writes_summary(tmp_path):
    def command_runner(argv, **kwargs):
        if argv[0] == "nvidia-smi":
            return subprocess.CompletedProcess(argv, 127, "", "nvidia-smi missing")
        if argv[0] == "ps":
            return subprocess.CompletedProcess(argv, 0, "10 1 7\n", "")
        raise AssertionError(f"unexpected command: {argv}")

    output_path = tmp_path / "runtime-telemetry.json"

    record = (
        RuntimeTelemetrySampler(
            output_path,
            process_pid=10,
            interval_seconds=60.0,
            command_runner=command_runner,
            clock=lambda: 1.0,
        )
        .start()
        .stop()
    )

    assert record["record_type"] == RUNTIME_TELEMETRY_RECORD_TYPE
    assert record["peak_process_tree_rss_bytes"] == 7 * 1024
    assert json.loads(output_path.read_text(encoding="utf-8")) == record
