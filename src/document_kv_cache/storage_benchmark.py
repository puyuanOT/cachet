"""Storage reader benchmarks for Document KV Cache."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_kv_cache.kvpack import PackChunk, write_kvpack
from document_kv_cache.models import ChunkRef, DocumentChunkType, KVCacheKey
from document_kv_cache.storage import (
    DiskRangeReader,
    MemoryRangeReader,
    UnityCatalogVolumeRangeReader,
    is_real_uc_volume_root,
    local_path,
)


STORAGE_BENCHMARK_RECORD_TYPE = "document_kv.storage_benchmark.v1"
SUPPORTED_STORAGE_BENCHMARK_READERS = ("memory", "disk", "unity_catalog")
RELEASE_STORAGE_BENCHMARK_READERS = SUPPORTED_STORAGE_BENCHMARK_READERS

__all__ = [
    "STORAGE_BENCHMARK_RECORD_TYPE",
    "SUPPORTED_STORAGE_BENCHMARK_READERS",
    "RELEASE_STORAGE_BENCHMARK_READERS",
    "StorageBenchmarkConfig",
    "StorageBenchmarkEvidence",
    "StorageBenchmarkResult",
    "StorageReaderBenchmarkResult",
    "evaluate_storage_benchmark_evidence",
    "evaluate_release_storage_benchmark_evidence",
    "run_storage_benchmark",
    "storage_benchmark_evidence_to_record",
    "storage_benchmark_result_to_record",
    "write_storage_benchmark_result_json",
    "main",
]


@dataclass(frozen=True, slots=True)
class StorageBenchmarkConfig:
    workspace_dir: str | Path
    benchmark_id: str = "storage-reader-benchmark"
    chunk_count: int = 64
    chunk_bytes: int = 1024 * 1024
    repeats: int = 4
    parallelism: int = 4
    readers: tuple[str, ...] = SUPPORTED_STORAGE_BENCHMARK_READERS
    align_bytes: int = 4096
    uc_volume_root: str | Path | None = None

    def __post_init__(self) -> None:
        if not self.benchmark_id:
            raise ValueError("benchmark_id must be non-empty")
        if self.chunk_count <= 0:
            raise ValueError("chunk_count must be positive")
        if self.chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        if self.repeats <= 0:
            raise ValueError("repeats must be positive")
        if self.parallelism <= 0:
            raise ValueError("parallelism must be positive")
        if type(self.align_bytes) is not int:
            raise ValueError("align_bytes must be an integer")
        if self.align_bytes <= 0:
            raise ValueError("align_bytes must be positive")
        object.__setattr__(
            self,
            "readers",
            _validate_storage_benchmark_reader_ids(self.readers, field_name="readers"),
        )


@dataclass(frozen=True, slots=True)
class StorageReaderBenchmarkResult:
    reader_id: str
    total_reads: int
    total_bytes: int
    parallelism: int
    wall_seconds: float
    latency_mean_seconds: float | None
    latency_p50_seconds: float | None
    latency_p95_seconds: float | None
    throughput_bytes_per_second: float | None
    errors: int


@dataclass(frozen=True, slots=True)
class StorageBenchmarkResult:
    config: StorageBenchmarkConfig
    shard_uri: str
    uc_volume_root: str | None
    results: tuple[StorageReaderBenchmarkResult, ...]


@dataclass(frozen=True, slots=True)
class StorageBenchmarkEvidence:
    required_readers: tuple[str, ...]
    missing_readers: tuple[str, ...]
    readers_with_errors: tuple[str, ...]
    readers_without_latency: tuple[str, ...]
    readers_without_throughput: tuple[str, ...]
    require_real_uc_volume: bool = False
    uc_volume_root: str | None = None
    uc_volume_is_real: bool | None = None

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def issues(self) -> tuple[str, ...]:
        issues = []
        if self.missing_readers:
            issues.append(f"missing storage readers: {', '.join(self.missing_readers)}")
        if self.readers_with_errors:
            issues.append(f"storage readers with errors: {', '.join(self.readers_with_errors)}")
        if self.readers_without_latency:
            issues.append(f"storage readers without latency evidence: {', '.join(self.readers_without_latency)}")
        if self.readers_without_throughput:
            issues.append(f"storage readers without throughput evidence: {', '.join(self.readers_without_throughput)}")
        if self.require_real_uc_volume and self.uc_volume_is_real is not True:
            issues.append("unity_catalog reader requires a real /Volumes UC Volume root")
        return tuple(issues)


def run_storage_benchmark(config: StorageBenchmarkConfig) -> StorageBenchmarkResult:
    workspace_dir = local_path(str(config.workspace_dir))
    workspace_dir.mkdir(parents=True, exist_ok=True)
    shard_path = workspace_dir / "storage-benchmark.kvpack"
    refs = write_kvpack(
        shard_path,
        _synthetic_chunks(config.chunk_count, config.chunk_bytes),
        align_bytes=config.align_bytes,
    )
    reader_refs = _reader_refs(config, refs, workspace_dir)
    results = tuple(
        _benchmark_reader(
            reader_id=reader_id,
            reader=reader,
            refs=refs_for_reader,
            repeats=config.repeats,
            parallelism=config.parallelism,
        )
        for reader_id, reader, refs_for_reader in reader_refs
        if reader_id in config.readers
    )
    uc_volume_root = _uc_volume_root_for_record(config, workspace_dir)
    return StorageBenchmarkResult(
        config=config,
        shard_uri=shard_path.as_posix(),
        uc_volume_root=uc_volume_root,
        results=results,
    )


def storage_benchmark_result_to_record(result: StorageBenchmarkResult) -> dict[str, Any]:
    evidence = evaluate_storage_benchmark_evidence(result)
    release_evidence = evaluate_release_storage_benchmark_evidence(result)
    return {
        "record_type": STORAGE_BENCHMARK_RECORD_TYPE,
        "benchmark_id": result.config.benchmark_id,
        "workspace_dir": str(result.config.workspace_dir),
        "shard_uri": result.shard_uri,
        "uc_volume_root": result.uc_volume_root,
        "uc_volume_is_real": _is_real_uc_volume_root(result.uc_volume_root),
        "chunk_count": result.config.chunk_count,
        "chunk_bytes": result.config.chunk_bytes,
        "repeats": result.config.repeats,
        "parallelism": result.config.parallelism,
        "align_bytes": result.config.align_bytes,
        "readers": list(result.config.readers),
        "results": [_reader_result_to_record(reader_result) for reader_result in result.results],
        "storage_evidence": storage_benchmark_evidence_to_record(evidence),
        "release_storage_evidence": storage_benchmark_evidence_to_record(release_evidence),
    }


def evaluate_storage_benchmark_evidence(
    result: StorageBenchmarkResult,
    *,
    required_readers: Sequence[str] | None = None,
    require_real_uc_volume: bool = False,
) -> StorageBenchmarkEvidence:
    required = _validate_storage_benchmark_reader_ids(
        required_readers if required_readers is not None else result.config.readers,
        field_name="required_readers",
    )
    by_reader = {reader_result.reader_id: reader_result for reader_result in result.results}
    missing_readers = tuple(reader for reader in required if reader not in by_reader)
    readers_with_errors = tuple(
        reader
        for reader in required
        if reader in by_reader and by_reader[reader].errors > 0
    )
    readers_without_latency = tuple(
        reader
        for reader in required
        if reader in by_reader
        and (by_reader[reader].latency_p50_seconds is None or by_reader[reader].latency_p95_seconds is None)
    )
    readers_without_throughput = tuple(
        reader
        for reader in required
        if reader in by_reader
        and (
            by_reader[reader].throughput_bytes_per_second is None
            or by_reader[reader].throughput_bytes_per_second <= 0
        )
    )
    return StorageBenchmarkEvidence(
        required_readers=required,
        missing_readers=missing_readers,
        readers_with_errors=readers_with_errors,
        readers_without_latency=readers_without_latency,
        readers_without_throughput=readers_without_throughput,
        require_real_uc_volume=require_real_uc_volume,
        uc_volume_root=result.uc_volume_root,
        uc_volume_is_real=_is_real_uc_volume_root(result.uc_volume_root),
    )


def evaluate_release_storage_benchmark_evidence(result: StorageBenchmarkResult) -> StorageBenchmarkEvidence:
    return evaluate_storage_benchmark_evidence(
        result,
        required_readers=RELEASE_STORAGE_BENCHMARK_READERS,
        require_real_uc_volume=True,
    )


def _validate_storage_benchmark_reader_ids(
    readers: Sequence[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(readers, (str, bytes)) or not isinstance(readers, Sequence):
        raise ValueError(f"{field_name} must be a sequence of storage reader ids")
    if not readers:
        raise ValueError(f"{field_name} must be non-empty")

    normalized = []
    seen = set()
    duplicates = []
    unsupported = []
    for reader in readers:
        if not isinstance(reader, str) or not reader:
            raise ValueError(f"{field_name} entries must be non-empty strings")
        if reader not in SUPPORTED_STORAGE_BENCHMARK_READERS:
            unsupported.append(reader)
            continue
        if reader in seen:
            if reader not in duplicates:
                duplicates.append(reader)
            continue
        seen.add(reader)
        normalized.append(reader)

    if unsupported:
        raise ValueError(f"Unsupported storage benchmark readers: {sorted(set(unsupported))}")
    if duplicates:
        raise ValueError(f"Duplicate storage benchmark readers in {field_name}: {duplicates}")
    return tuple(normalized)


def storage_benchmark_evidence_to_record(evidence: StorageBenchmarkEvidence) -> dict[str, Any]:
    return {
        "ok": evidence.ok,
        "required_readers": list(evidence.required_readers),
        "missing_readers": list(evidence.missing_readers),
        "readers_with_errors": list(evidence.readers_with_errors),
        "readers_without_latency": list(evidence.readers_without_latency),
        "readers_without_throughput": list(evidence.readers_without_throughput),
        "require_real_uc_volume": evidence.require_real_uc_volume,
        "uc_volume_root": evidence.uc_volume_root,
        "uc_volume_is_real": evidence.uc_volume_is_real,
        "issues": list(evidence.issues),
    }


def write_storage_benchmark_result_json(result: StorageBenchmarkResult, path: str | Path) -> None:
    output_path = local_path(str(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(storage_benchmark_result_to_record(result), indent=2, sort_keys=True) + "\n")


def _reader_result_to_record(result: StorageReaderBenchmarkResult) -> dict[str, Any]:
    throughput_mib = (
        result.throughput_bytes_per_second / (1024 * 1024)
        if result.throughput_bytes_per_second is not None
        else None
    )
    return {
        "reader_id": result.reader_id,
        "total_reads": result.total_reads,
        "total_bytes": result.total_bytes,
        "parallelism": result.parallelism,
        "wall_seconds": result.wall_seconds,
        "latency_mean_seconds": result.latency_mean_seconds,
        "latency_p50_seconds": result.latency_p50_seconds,
        "latency_p95_seconds": result.latency_p95_seconds,
        "throughput_bytes_per_second": result.throughput_bytes_per_second,
        "throughput_mib_per_second": throughput_mib,
        "errors": result.errors,
    }


def _synthetic_chunks(chunk_count: int, chunk_bytes: int) -> tuple[PackChunk, ...]:
    chunks: list[PackChunk] = []
    for index in range(chunk_count):
        chunks.append(
            PackChunk(
                key=KVCacheKey.for_document(
                    model_id="qwen3:4b-instruct",
                    lora_id="base",
                    prompt_template_version="storage-benchmark",
                    document_id=f"doc-{index:06d}",
                    chunk_type=DocumentChunkType.DOCUMENT_CHUNK,
                    chunk_id="chunk-0000",
                ),
                payload=_payload_for(index, chunk_bytes),
                token_count=1,
                dtype="int8",
                layout_version="storage-benchmark-v1",
            )
        )
    return tuple(chunks)


def _payload_for(index: int, chunk_bytes: int) -> bytes:
    pattern = f"cachet-kv:{index:08d}:".encode("ascii")
    repeats = (chunk_bytes // len(pattern)) + 1
    return (pattern * repeats)[:chunk_bytes]


def _reader_refs(
    config: StorageBenchmarkConfig,
    refs: Sequence[ChunkRef],
    workspace_dir: Path,
) -> tuple[tuple[str, Any, tuple[ChunkRef, ...]], ...]:
    readers = []
    if "memory" in config.readers:
        memory_uri = "memory:storage-benchmark.kvpack"
        shard_bytes = local_path(refs[0].shard_uri).read_bytes()
        memory_refs = tuple(_replace_shard_uri(ref, memory_uri) for ref in refs)
        readers.append(("memory", MemoryRangeReader({memory_uri: shard_bytes}), memory_refs))
    if "disk" in config.readers:
        readers.append(("disk", DiskRangeReader(), tuple(refs)))
    if "unity_catalog" in config.readers:
        uc_root = Path(config.uc_volume_root) if config.uc_volume_root is not None else workspace_dir / "uc-volume"
        uc_root.mkdir(parents=True, exist_ok=True)
        uc_shard = uc_root / "storage-benchmark.kvpack"
        shutil.copyfile(local_path(refs[0].shard_uri), uc_shard)
        uc_refs = tuple(_replace_shard_uri(ref, uc_shard.name) for ref in refs)
        readers.append(("unity_catalog", UnityCatalogVolumeRangeReader(volume_root=uc_root), uc_refs))
    return tuple(readers)


def _uc_volume_root_for_record(config: StorageBenchmarkConfig, workspace_dir: Path) -> str | None:
    if "unity_catalog" not in config.readers:
        return None
    root = Path(config.uc_volume_root) if config.uc_volume_root is not None else workspace_dir / "uc-volume"
    return root.as_posix()


def _is_real_uc_volume_root(uc_volume_root: str | None) -> bool | None:
    return is_real_uc_volume_root(uc_volume_root)


def _replace_shard_uri(ref: ChunkRef, shard_uri: str) -> ChunkRef:
    return ChunkRef(
        key=ref.key,
        shard_uri=shard_uri,
        byte_offset=ref.byte_offset,
        byte_length=ref.byte_length,
        token_count=ref.token_count,
        dtype=ref.dtype,
        layout_version=ref.layout_version,
        checksum=ref.checksum,
        storage_layout=ref.storage_layout,
    )


def _benchmark_reader(
    *,
    reader_id: str,
    reader: Any,
    refs: Sequence[ChunkRef],
    repeats: int,
    parallelism: int,
) -> StorageReaderBenchmarkResult:
    scheduled_refs = tuple(ref for _ in range(repeats) for ref in refs)
    started = time.perf_counter()
    if parallelism == 1:
        observations = tuple(_read_once(reader, ref) for ref in scheduled_refs)
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            observations = tuple(executor.map(lambda ref: _read_once(reader, ref), scheduled_refs))
    wall_seconds = time.perf_counter() - started
    latencies = [duration for duration, _, error in observations if error is None]
    total_bytes = sum(byte_count for _, byte_count, error in observations if error is None)
    errors = sum(1 for _, _, error in observations if error is not None)
    return StorageReaderBenchmarkResult(
        reader_id=reader_id,
        total_reads=len(scheduled_refs),
        total_bytes=total_bytes,
        parallelism=parallelism,
        wall_seconds=wall_seconds,
        latency_mean_seconds=(sum(latencies) / len(latencies)) if latencies else None,
        latency_p50_seconds=_percentile(latencies, 0.50),
        latency_p95_seconds=_percentile(latencies, 0.95),
        throughput_bytes_per_second=(total_bytes / wall_seconds) if wall_seconds > 0 else None,
        errors=errors,
    )


def _read_once(reader: Any, ref: ChunkRef) -> tuple[float, int, str | None]:
    started = time.perf_counter()
    try:
        payload = reader.read(ref)
    except Exception as exc:  # pragma: no cover - exercised by callers with custom readers.
        return time.perf_counter() - started, 0, f"{type(exc).__name__}: {exc}"
    return time.perf_counter() - started, len(payload), None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = percentile * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark document KV-cache storage readers.")
    parser.add_argument("--workspace-dir", required=True, help="Directory for synthetic shard artifacts.")
    parser.add_argument("--benchmark-id", default="storage-reader-benchmark")
    parser.add_argument("--chunk-count", type=int, default=64)
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument(
        "--reader",
        action="append",
        choices=SUPPORTED_STORAGE_BENCHMARK_READERS,
        help="Reader to benchmark. Repeat for multiple readers; defaults to all readers.",
    )
    parser.add_argument("--align-bytes", type=int, default=4096)
    parser.add_argument("--uc-volume-root", help="Optional real UC Volume root, usually /Volumes/catalog/schema/volume.")
    parser.add_argument("--output-json", help="Write the benchmark result JSON to this path instead of stdout.")
    args = parser.parse_args(argv)

    try:
        config = StorageBenchmarkConfig(
            workspace_dir=args.workspace_dir,
            benchmark_id=args.benchmark_id,
            chunk_count=args.chunk_count,
            chunk_bytes=args.chunk_bytes,
            repeats=args.repeats,
            parallelism=args.parallelism,
            readers=tuple(args.reader) if args.reader else SUPPORTED_STORAGE_BENCHMARK_READERS,
            align_bytes=args.align_bytes,
            uc_volume_root=args.uc_volume_root,
        )
        result = run_storage_benchmark(config)
        if args.output_json:
            write_storage_benchmark_result_json(result, args.output_json)
        else:
            print(json.dumps(storage_benchmark_result_to_record(result), indent=2, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
