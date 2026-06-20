import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import document_kv_cache.storage_benchmark as public_storage_benchmark
import restaurant_kv_serving.storage_benchmark as legacy_storage_benchmark
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


def test_storage_benchmark_public_module_owns_implementation_and_legacy_wraps_it():
    assert public_storage_benchmark.StorageBenchmarkConfig.__module__ == "document_kv_cache.storage_benchmark"
    assert public_storage_benchmark.run_storage_benchmark.__module__ == "document_kv_cache.storage_benchmark"
    assert public_storage_benchmark.storage_benchmark_result_to_record.__module__ == "document_kv_cache.storage_benchmark"
    assert legacy_storage_benchmark.StorageBenchmarkConfig is public_storage_benchmark.StorageBenchmarkConfig
    assert legacy_storage_benchmark.StorageBenchmarkResult is public_storage_benchmark.StorageBenchmarkResult
    assert legacy_storage_benchmark.run_storage_benchmark.__module__ == "restaurant_kv_serving.storage_benchmark"
    assert legacy_storage_benchmark.storage_benchmark_result_to_record.__module__ == "restaurant_kv_serving.storage_benchmark"
    assert not hasattr(legacy_storage_benchmark, "__all__")


def test_legacy_storage_benchmark_run_preserves_local_path_and_writer_hooks(monkeypatch, tmp_path):
    target_workspace = tmp_path / "redirected-workspace"
    calls: list[str] = []

    def fake_local_path(raw_path: str):
        calls.append(raw_path)
        return target_workspace

    monkeypatch.setattr(legacy_storage_benchmark, "local_path", fake_local_path)

    result = legacy_storage_benchmark.run_storage_benchmark(
        StorageBenchmarkConfig(
            workspace_dir="legacy-logical-workspace",
            chunk_count=1,
            chunk_bytes=16,
            repeats=1,
            parallelism=1,
            readers=("disk",),
        )
    )

    assert calls == ["legacy-logical-workspace"]
    assert result.shard_uri == (target_workspace / "storage-benchmark.kvpack").as_posix()
    assert (target_workspace / "storage-benchmark.kvpack").exists()


def test_legacy_storage_benchmark_run_preserves_private_helper_hooks(monkeypatch, tmp_path):
    payload_calls: list[tuple[int, int]] = []
    read_calls: list[ChunkRef] = []
    percentile_calls: list[tuple[tuple[float, ...], float]] = []

    def fake_payload_for(index: int, chunk_bytes: int) -> bytes:
        payload_calls.append((index, chunk_bytes))
        return bytes([65 + index]) * chunk_bytes

    def fake_read_once(reader, ref):
        read_calls.append(ref)
        return 0.123, ref.byte_length, None

    def fake_percentile(values, percentile):
        percentile_calls.append((tuple(values), percentile))
        return 0.456 if percentile == 0.50 else 0.789

    monkeypatch.setattr(legacy_storage_benchmark, "_payload_for", fake_payload_for)
    monkeypatch.setattr(legacy_storage_benchmark, "_read_once", fake_read_once)
    monkeypatch.setattr(legacy_storage_benchmark, "_percentile", fake_percentile)

    result = legacy_storage_benchmark.run_storage_benchmark(
        StorageBenchmarkConfig(
            workspace_dir=tmp_path / "workspace",
            chunk_count=2,
            chunk_bytes=16,
            repeats=2,
            parallelism=1,
            readers=("memory",),
        )
    )

    assert payload_calls == [(0, 16), (1, 16)]
    assert len(read_calls) == 4
    assert percentile_calls == [((0.123, 0.123, 0.123, 0.123), 0.50), ((0.123, 0.123, 0.123, 0.123), 0.95)]
    assert result.results[0].latency_p50_seconds == 0.456
    assert result.results[0].latency_p95_seconds == 0.789


def test_legacy_storage_benchmark_main_uses_legacy_namespace_hooks(monkeypatch, tmp_path):
    output_json = tmp_path / "storage-result.json"
    calls = {}

    result = StorageBenchmarkResult(
        config=StorageBenchmarkConfig(
            workspace_dir="legacy-workspace",
            benchmark_id="legacy-hook",
            readers=("memory",),
        ),
        shard_uri="memory:legacy",
        uc_volume_root=None,
        results=(
            StorageReaderBenchmarkResult(
                reader_id="memory",
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

    def fake_run(config):
        calls["config"] = config
        return result

    def fake_write(run_result, path):
        calls["written"] = (run_result, path)
        output_json.write_text(json.dumps({"ok": True, "source": "legacy-hook"}), encoding="utf-8")

    monkeypatch.setattr(legacy_storage_benchmark, "run_storage_benchmark", fake_run)
    monkeypatch.setattr(legacy_storage_benchmark, "write_storage_benchmark_result_json", fake_write)

    exit_code = legacy_storage_benchmark.main(
        [
            "--workspace-dir",
            "legacy-workspace",
            "--benchmark-id",
            "legacy-hook",
            "--reader",
            "memory",
            "--output-json",
            str(output_json),
        ]
    )

    assert exit_code == 0
    assert calls["config"].benchmark_id == "legacy-hook"
    assert calls["config"].readers == ("memory",)
    assert calls["written"] == (result, str(output_json))
    assert json.loads(output_json.read_text(encoding="utf-8")) == {"ok": True, "source": "legacy-hook"}


def test_legacy_storage_benchmark_main_writes_json_without_monkeypatches(tmp_path):
    output_json = tmp_path / "storage-result.json"

    exit_code = legacy_storage_benchmark.main(
        [
            "--workspace-dir",
            str(tmp_path / "workspace"),
            "--benchmark-id",
            "legacy-cli-smoke",
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
        ]
    )

    record = json.loads(output_json.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert record["record_type"] == STORAGE_BENCHMARK_RECORD_TYPE
    assert record["benchmark_id"] == "legacy-cli-smoke"
    assert [row["reader_id"] for row in record["results"]] == ["memory"]


def test_legacy_storage_benchmark_serializers_do_not_recurse(tmp_path):
    result = legacy_storage_benchmark.run_storage_benchmark(
        StorageBenchmarkConfig(
            workspace_dir=tmp_path / "workspace",
            benchmark_id="legacy-serializer-smoke",
            chunk_count=1,
            chunk_bytes=16,
            repeats=1,
            parallelism=1,
            readers=("memory",),
        )
    )

    record = legacy_storage_benchmark.storage_benchmark_result_to_record(result)
    release_evidence = legacy_storage_benchmark.evaluate_release_storage_benchmark_evidence(result)

    assert record["benchmark_id"] == "legacy-serializer-smoke"
    assert record["storage_evidence"]["ok"] is True
    assert release_evidence.missing_readers == ("disk", "unity_catalog")


def test_legacy_storage_benchmark_hooks_do_not_leak_into_document_namespace(monkeypatch, tmp_path):
    def legacy_local_path_should_not_run(raw_path: str):  # pragma: no cover - defensive assertion
        raise AssertionError(f"document namespace used legacy local_path for {raw_path}")

    monkeypatch.setattr(legacy_storage_benchmark, "local_path", legacy_local_path_should_not_run)

    result = public_storage_benchmark.run_storage_benchmark(
        StorageBenchmarkConfig(
            workspace_dir=tmp_path / "document-workspace",
            chunk_count=1,
            chunk_bytes=16,
            repeats=1,
            parallelism=1,
            readers=("memory",),
        )
    )

    record = public_storage_benchmark.storage_benchmark_result_to_record(result)
    assert record["storage_evidence"]["ok"] is True


def test_legacy_storage_benchmark_import_order_does_not_capture_public_monkeypatch():
    script = """
import json
from pathlib import Path
import sys
import tempfile

import document_kv_cache.storage_benchmark as public_storage_benchmark

def public_payload_should_not_run(*args, **kwargs):
    raise AssertionError("legacy imported a public monkeypatch as its default")

class FakeStorageBenchmarkConfig:
    def __init__(self, *args, **kwargs):
        raise AssertionError("legacy imported patched public config class")

public_storage_benchmark._payload_for = public_payload_should_not_run
public_storage_benchmark.RELEASE_STORAGE_BENCHMARK_READERS = ("memory",)
public_storage_benchmark.StorageBenchmarkConfig = FakeStorageBenchmarkConfig

import restaurant_kv_serving.storage_benchmark as legacy_storage_benchmark

assert legacy_storage_benchmark.StorageBenchmarkConfig is not FakeStorageBenchmarkConfig
assert legacy_storage_benchmark.RELEASE_STORAGE_BENCHMARK_READERS == ("memory", "disk", "unity_catalog")
payload = legacy_storage_benchmark._payload_for(0, 32)
assert payload.startswith(b"document-kv-cache:00000000:")
try:
    public_storage_benchmark._payload_for(0, 32)
except AssertionError:
    pass
else:
    raise AssertionError("public monkeypatch was not installed")

with tempfile.TemporaryDirectory() as raw_tmp:
    tmp_path = Path(raw_tmp)
    output_json = tmp_path / "storage-result.json"
    exit_code = legacy_storage_benchmark.main(
        [
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
        ]
    )
    assert exit_code == 0
    record = json.loads(output_json.read_text(encoding="utf-8"))
    assert record["release_storage_evidence"]["required_readers"] == ["memory", "disk", "unity_catalog"]
    assert record["release_storage_evidence"]["missing_readers"] == ["disk", "unity_catalog"]
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_storage_benchmark_star_import_surfaces_are_curated_for_document_and_preserved_for_legacy():
    public_namespace: dict[str, object] = {}
    legacy_namespace: dict[str, object] = {}

    exec("from document_kv_cache.storage_benchmark import *", public_namespace)
    exec("from restaurant_kv_serving.storage_benchmark import *", legacy_namespace)

    assert sorted(k for k in public_namespace if not k.startswith("__")) == sorted(public_storage_benchmark.__all__)
    assert "write_kvpack" not in public_namespace
    assert sorted(k for k in legacy_namespace if not k.startswith("__")) == [
        "Any",
        "ChunkRef",
        "DiskRangeReader",
        "DocumentChunkType",
        "KVCacheKey",
        "MemoryRangeReader",
        "PackChunk",
        "Path",
        "RELEASE_STORAGE_BENCHMARK_READERS",
        "STORAGE_BENCHMARK_RECORD_TYPE",
        "SUPPORTED_STORAGE_BENCHMARK_READERS",
        "Sequence",
        "StorageBenchmarkConfig",
        "StorageBenchmarkEvidence",
        "StorageBenchmarkResult",
        "StorageReaderBenchmarkResult",
        "ThreadPoolExecutor",
        "UnityCatalogVolumeRangeReader",
        "annotations",
        "argparse",
        "dataclass",
        "evaluate_release_storage_benchmark_evidence",
        "evaluate_storage_benchmark_evidence",
        "is_real_uc_volume_root",
        "json",
        "local_path",
        "main",
        "run_storage_benchmark",
        "shutil",
        "storage_benchmark_evidence_to_record",
        "storage_benchmark_result_to_record",
        "time",
        "write_kvpack",
        "write_storage_benchmark_result_json",
    ]
