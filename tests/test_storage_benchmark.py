import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import document_kv_cache.storage_benchmark as public_storage_benchmark
from document_kv_cache.storage_benchmark import (
    STORAGE_BENCHMARK_RECORD_TYPE,
    RELEASE_STORAGE_BENCHMARK_READERS,
    SUPPORTED_STORAGE_BENCHMARK_READERS,
    StorageBenchmarkConfig,
    StorageBenchmarkResult,
    StorageReaderBenchmarkResult,
    evaluate_release_storage_benchmark_evidence,
    evaluate_storage_benchmark_evidence,
    run_storage_benchmark,
    storage_benchmark_evidence_to_record,
    storage_benchmark_result_to_record,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_storage_benchmark_reports_memory_disk_and_uc_readers(tmp_path):
    result = run_storage_benchmark(
        StorageBenchmarkConfig(
            workspace_dir=tmp_path / "workspace",
            benchmark_id="storage-smoke",
            chunk_count=3,
            chunk_bytes=64,
            repeats=2,
            parallelism=2,
            align_bytes=8,
        )
    )

    record = storage_benchmark_result_to_record(result)
    by_reader = {row["reader_id"]: row for row in record["results"]}

    assert record["record_type"] == STORAGE_BENCHMARK_RECORD_TYPE
    assert record["benchmark_id"] == "storage-smoke"
    assert record["chunk_count"] == 3
    assert record["chunk_bytes"] == 64
    assert record["uc_volume_root"].endswith("/workspace/uc-volume")
    assert record["uc_volume_is_real"] is False
    assert record["storage_evidence"]["ok"] is True
    assert record["storage_evidence"]["required_readers"] == list(SUPPORTED_STORAGE_BENCHMARK_READERS)
    assert record["release_storage_evidence"]["ok"] is False
    assert record["release_storage_evidence"]["required_readers"] == list(RELEASE_STORAGE_BENCHMARK_READERS)
    assert record["release_storage_evidence"]["missing_readers"] == []
    assert "unity_catalog reader requires a real /Volumes UC Volume root" in record["release_storage_evidence"]["issues"]
    assert set(by_reader) == set(SUPPORTED_STORAGE_BENCHMARK_READERS)
    for row in by_reader.values():
        assert row["total_reads"] == 6
        assert row["total_bytes"] == 384
        assert row["parallelism"] == 2
        assert row["errors"] == 0
        assert row["latency_p50_seconds"] is not None
        assert row["latency_p95_seconds"] is not None
        assert row["throughput_bytes_per_second"] > 0
        assert row["throughput_mib_per_second"] > 0


def test_storage_benchmark_reader_selection_does_not_prepare_unselected_uc(tmp_path):
    workspace_dir = tmp_path / "workspace"

    result = run_storage_benchmark(
        StorageBenchmarkConfig(
            workspace_dir=workspace_dir,
            chunk_count=2,
            chunk_bytes=16,
            repeats=1,
            parallelism=1,
            readers=("memory",),
        )
    )

    record = storage_benchmark_result_to_record(result)

    assert [row["reader_id"] for row in record["results"]] == ["memory"]
    assert not (workspace_dir / "uc-volume").exists()


def test_storage_benchmark_disk_only_does_not_load_full_shard_into_memory(tmp_path, monkeypatch):
    original_read_bytes = Path.read_bytes
    calls: list[Path] = []

    def tracking_read_bytes(path):
        calls.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", tracking_read_bytes)

    result = run_storage_benchmark(
        StorageBenchmarkConfig(
            workspace_dir=tmp_path / "workspace",
            chunk_count=2,
            chunk_bytes=16,
            repeats=1,
            parallelism=1,
            readers=("disk",),
        )
    )

    record = storage_benchmark_result_to_record(result)

    assert [row["reader_id"] for row in record["results"]] == ["disk"]
    assert calls == []


def test_evaluate_storage_benchmark_evidence_can_require_real_uc_volume(tmp_path):
    result = StorageBenchmarkResult(
        config=StorageBenchmarkConfig(
            workspace_dir=tmp_path / "workspace",
            readers=("unity_catalog",),
            uc_volume_root="/Volumes/catalog/schema/volume/storage-benchmark",
        ),
        shard_uri="/Volumes/catalog/schema/volume/storage-benchmark/storage-benchmark.kvpack",
        uc_volume_root="/Volumes/catalog/schema/volume/storage-benchmark",
        results=(
            StorageReaderBenchmarkResult(
                reader_id="unity_catalog",
                total_reads=1,
                total_bytes=16,
                parallelism=1,
                wall_seconds=0.001,
                latency_mean_seconds=0.001,
                latency_p50_seconds=0.001,
                latency_p95_seconds=0.001,
                throughput_bytes_per_second=16_000,
                errors=0,
            ),
        ),
    )

    evidence = evaluate_storage_benchmark_evidence(
        result,
        required_readers=("unity_catalog",),
        require_real_uc_volume=True,
    )
    record = storage_benchmark_evidence_to_record(evidence)

    assert evidence.ok
    assert record["ok"] is True
    assert record["uc_volume_is_real"] is True
    assert record["issues"] == []
    assert evaluate_release_storage_benchmark_evidence(result).missing_readers == ("memory", "disk")


@pytest.mark.parametrize(
    "uc_volume_root",
    [
        "/Volumes",
        "/Volumes/catalog",
        "/Volumes/catalog/schema",
        "Volumes/catalog/schema/volume",
    ],
)
def test_evaluate_storage_benchmark_evidence_rejects_incomplete_uc_volume_paths(tmp_path, uc_volume_root):
    result = StorageBenchmarkResult(
        config=StorageBenchmarkConfig(
            workspace_dir=tmp_path / "workspace",
            readers=("unity_catalog",),
            uc_volume_root=uc_volume_root,
        ),
        shard_uri=f"{uc_volume_root}/storage-benchmark.kvpack",
        uc_volume_root=uc_volume_root,
        results=(
            StorageReaderBenchmarkResult(
                reader_id="unity_catalog",
                total_reads=1,
                total_bytes=16,
                parallelism=1,
                wall_seconds=0.001,
                latency_mean_seconds=0.001,
                latency_p50_seconds=0.001,
                latency_p95_seconds=0.001,
                throughput_bytes_per_second=16_000,
                errors=0,
            ),
        ),
    )

    evidence = evaluate_storage_benchmark_evidence(
        result,
        required_readers=("unity_catalog",),
        require_real_uc_volume=True,
    )

    assert not evidence.ok
    assert evidence.uc_volume_is_real is False
    assert "unity_catalog reader requires a real /Volumes UC Volume root" in evidence.issues


def test_evaluate_storage_benchmark_evidence_reports_missing_errors_and_weak_metrics(tmp_path):
    config = StorageBenchmarkConfig(workspace_dir=tmp_path / "workspace", readers=("memory",))
    result = StorageBenchmarkResult(
        config=config,
        shard_uri=str(tmp_path / "workspace" / "storage-benchmark.kvpack"),
        uc_volume_root=None,
        results=(
            StorageReaderBenchmarkResult(
                reader_id="memory",
                total_reads=1,
                total_bytes=0,
                parallelism=1,
                wall_seconds=0.0,
                latency_mean_seconds=None,
                latency_p50_seconds=None,
                latency_p95_seconds=None,
                throughput_bytes_per_second=None,
                errors=1,
            ),
        ),
    )

    evidence = evaluate_storage_benchmark_evidence(
        result,
        required_readers=("memory", "disk", "unity_catalog"),
        require_real_uc_volume=True,
    )

    assert not evidence.ok
    assert evidence.missing_readers == ("disk", "unity_catalog")
    assert evidence.readers_with_errors == ("memory",)
    assert evidence.readers_without_latency == ("memory",)
    assert evidence.readers_without_throughput == ("memory",)
    assert any(issue.startswith("missing storage readers") for issue in evidence.issues)
    assert "unity_catalog reader requires a real /Volumes UC Volume root" in evidence.issues


def test_evaluate_storage_benchmark_evidence_validates_required_readers(tmp_path):
    result = run_storage_benchmark(
        StorageBenchmarkConfig(
            workspace_dir=tmp_path / "workspace",
            chunk_count=1,
            chunk_bytes=16,
            repeats=1,
            parallelism=1,
            readers=("memory",),
        )
    )

    with pytest.raises(ValueError, match="required_readers"):
        evaluate_storage_benchmark_evidence(result, required_readers=())

    with pytest.raises(ValueError, match="required_readers"):
        evaluate_storage_benchmark_evidence(result, required_readers=("",))

    with pytest.raises(ValueError, match="Unsupported"):
        evaluate_storage_benchmark_evidence(result, required_readers=("object-store",))


def test_storage_benchmark_config_validates_inputs(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        StorageBenchmarkConfig(workspace_dir=tmp_path, readers=("object-store",))

    with pytest.raises(ValueError, match="chunk_count"):
        StorageBenchmarkConfig(workspace_dir=tmp_path, chunk_count=0)

    with pytest.raises(ValueError, match="parallelism"):
        StorageBenchmarkConfig(workspace_dir=tmp_path, parallelism=0)

    with pytest.raises(ValueError, match="align_bytes"):
        StorageBenchmarkConfig(workspace_dir=tmp_path, align_bytes=True)


def test_public_storage_benchmark_main_writes_json(tmp_path):
    output_json = tmp_path / "storage-result.json"
    exit_code = public_storage_benchmark.main(
        [
            "--workspace-dir",
            str(tmp_path / "workspace"),
            "--benchmark-id",
            "cli-storage-smoke",
            "--chunk-count",
            "2",
            "--chunk-bytes",
            "16",
            "--repeats",
            "1",
            "--parallelism",
            "1",
            "--reader",
            "memory",
            "--output-json",
            str(output_json),
        ]
    )

    record = json.loads(output_json.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert record["record_type"] == STORAGE_BENCHMARK_RECORD_TYPE
    assert record["benchmark_id"] == "cli-storage-smoke"
    assert [row["reader_id"] for row in record["results"]] == ["memory"]
    assert record["storage_evidence"]["ok"] is True
    assert record["release_storage_evidence"]["ok"] is False
    assert record["release_storage_evidence"]["missing_readers"] == ["disk", "unity_catalog"]


def test_public_storage_benchmark_module_executes_with_python_m(tmp_path):
    output_json = tmp_path / "storage-result.json"
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "document_kv_cache.storage_benchmark",
            "--workspace-dir",
            str(tmp_path / "workspace"),
            "--chunk-count",
            "1",
            "--chunk-bytes",
            "16",
            "--repeats",
            "1",
            "--parallelism",
            "1",
            "--reader",
            "memory",
            "--output-json",
            str(output_json),
        ],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    record = json.loads(output_json.read_text(encoding="utf-8"))

    assert completed.stdout == ""
    assert record["record_type"] == STORAGE_BENCHMARK_RECORD_TYPE
    assert [row["reader_id"] for row in record["results"]] == ["memory"]
    assert record["release_storage_evidence"]["ok"] is False
