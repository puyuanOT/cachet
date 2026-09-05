"""Empirical probe for runtime KV offload config and Cachet KV persistence tiers."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import pkgutil
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from document_kv_cache.cache import CacheTier, ChunkCache
from document_kv_cache.engine_launch_config import (
    build_sglang_hicache_server_args,
    build_vllm_kv_offloading_config,
)
from document_kv_cache.kvpack import PackChunk, write_kvpack, write_kvpack_bytes
from document_kv_cache.models import DocumentChunkType, KVCacheKey
from document_kv_cache.storage import MemoryRangeReader, RoutedRangeReader

RUNTIME_KV_OFFLOAD_PROBE_RECORD_TYPE = "document_kv.runtime_kv_offload_probe.v1"
RUNTIME_KV_OFFLOAD_PROBE_SCHEMA_VERSION = 1

__all__ = [
    "RUNTIME_KV_OFFLOAD_PROBE_RECORD_TYPE",
    "RUNTIME_KV_OFFLOAD_PROBE_SCHEMA_VERSION",
    "run_runtime_kv_offload_probe",
    "runtime_kv_offload_probe_record_issues",
    "write_runtime_kv_offload_probe_json",
    "main",
]


def run_runtime_kv_offload_probe(
    *,
    work_dir: str | Path,
    require_vllm_offloading_import: bool = False,
    require_sglang_package: bool = False,
) -> dict[str, Any]:
    """Run a local, deterministic probe and return a JSON-safe evidence record."""

    started = time.perf_counter()
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    vllm_probe = _probe_vllm_offloading_config(
        work_path / "vllm-offload",
        require_import=require_vllm_offloading_import,
    )
    sglang_probe = _probe_sglang_hicache_config(require_package=require_sglang_package)
    hierarchy_probe = _probe_hierarchical_document_kv(work_path / "hierarchical-document-kv")
    record: dict[str, Any] = {
        "record_type": RUNTIME_KV_OFFLOAD_PROBE_RECORD_TYPE,
        "schema_version": RUNTIME_KV_OFFLOAD_PROBE_SCHEMA_VERSION,
        "environment": _environment_record(),
        "vllm_runtime_kv_offload": vllm_probe,
        "sglang_hicache": sglang_probe,
        "hierarchical_document_kv": hierarchy_probe,
        "elapsed_seconds": time.perf_counter() - started,
    }
    record["ok"] = not runtime_kv_offload_probe_record_issues(record)
    return record


def write_runtime_kv_offload_probe_json(
    path: str | Path,
    *,
    work_dir: str | Path,
    require_vllm_offloading_import: bool = False,
    require_sglang_package: bool = False,
) -> dict[str, Any]:
    """Run the probe, write its evidence JSON, and return the record."""

    record = run_runtime_kv_offload_probe(
        work_dir=work_dir,
        require_vllm_offloading_import=require_vllm_offloading_import,
        require_sglang_package=require_sglang_package,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def runtime_kv_offload_probe_record_issues(record: object) -> tuple[str, ...]:
    """Return issues that make a runtime KV offload probe record unfit as evidence."""

    if not isinstance(record, Mapping):
        return ("runtime KV offload probe record must be an object",)
    issues: list[str] = []
    if record.get("record_type") != RUNTIME_KV_OFFLOAD_PROBE_RECORD_TYPE:
        issues.append(f"record_type must be {RUNTIME_KV_OFFLOAD_PROBE_RECORD_TYPE!r}")
    if record.get("schema_version") != RUNTIME_KV_OFFLOAD_PROBE_SCHEMA_VERSION:
        issues.append(f"schema_version must be {RUNTIME_KV_OFFLOAD_PROBE_SCHEMA_VERSION}")
    for section in ("vllm_runtime_kv_offload", "sglang_hicache", "hierarchical_document_kv"):
        section_record = record.get(section)
        if not isinstance(section_record, Mapping):
            issues.append(f"{section} must be an object")
        elif section_record.get("ok") is not True:
            section_issues = section_record.get("issues")
            if isinstance(section_issues, Sequence) and not isinstance(section_issues, (str, bytes, bytearray)):
                detail = "; ".join(str(issue) for issue in section_issues)
                issues.append(f"{section} failed: {detail}")
            else:
                issues.append(f"{section} failed")
    elapsed_seconds = record.get("elapsed_seconds")
    if isinstance(elapsed_seconds, bool) or not isinstance(elapsed_seconds, int | float) or elapsed_seconds < 0:
        issues.append("elapsed_seconds must be non-negative")
    ok = record.get("ok")
    if type(ok) is not bool and ok is not None:
        issues.append("ok must be boolean when present")
    return tuple(issues)


def _probe_vllm_offloading_config(work_dir: Path, *, require_import: bool) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    secondary_root = work_dir / "secondary-fs-tier"
    config = build_vllm_kv_offloading_config(
        cpu_bytes_to_use=2 * 1024**3,
        block_size=64,
        eviction_policy="lru",
        offload_prompt_only=True,
        secondary_fs_root=str(secondary_root),
        secondary_fs_read_threads=2,
        secondary_fs_write_threads=2,
    )
    import_probe = _find_vllm_offloading_connector()
    issues = _vllm_offloading_config_issues(config)
    if require_import and not import_probe["found"]:
        issues.append("vLLM OffloadingConnector import was required but not found")
    return {
        "ok": not issues,
        "issues": issues,
        "config": config,
        "required_runtime_import": require_import,
        "runtime_import": import_probe,
    }


def _probe_sglang_hicache_config(*, require_package: bool) -> dict[str, Any]:
    args = build_sglang_hicache_server_args(
        hicache_size_gb=64,
        page_size=8,
        hicache_storage_prefetch_policy="timeout",
        hicache_write_policy="write_through_selective",
        hicache_io_backend="direct",
        hicache_mem_layout="page_first",
    )
    package_probe = _package_probe("sglang")
    issues: list[str] = []
    if "--enable-hierarchical-cache" not in args:
        issues.append("SGLang HiCache args must enable hierarchical cache")
    if "--hicache-size" not in args or "64" not in args:
        issues.append("SGLang HiCache args must include the configured host-pool size")
    if require_package and not package_probe["found"]:
        issues.append("SGLang package import was required but not found")
    return {
        "ok": not issues,
        "issues": issues,
        "required_runtime_package": require_package,
        "runtime_package": package_probe,
        "server_args": list(args),
    }


def _probe_hierarchical_document_kv(work_dir: Path) -> dict[str, Any]:
    authoritative_dir = work_dir / "authoritative-shards"
    local_dir = work_dir / "local-tier"
    authoritative_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    disk_refs = write_kvpack(
        authoritative_dir / "disk.kvpack",
        (
            PackChunk(_document_key("hot-disk"), b"hot001", 6, "q8", "probe-v1"),
            PackChunk(_document_key("evict-a"), b"aaa001", 6, "q8", "probe-v1"),
            PackChunk(_document_key("evict-b"), b"bbb001", 6, "q8", "probe-v1"),
        ),
        align_bytes=1,
    )
    memory_payload, memory_refs = write_kvpack_bytes(
        "memory:cachet-runtime-kv-probe",
        (PackChunk(_document_key("hot-memory"), b"mem001", 6, "q8", "probe-v1"),),
        align_bytes=1,
    )
    reader = RoutedRangeReader(memory=MemoryRangeReader({"memory:cachet-runtime-kv-probe": memory_payload}))
    cache = ChunkCache(
        cpu_max_bytes=0,
        local_dir=local_dir,
        local_max_bytes=10,
        local_promotion_threshold=2,
    )
    issues: list[str] = []

    hot_disk_tiers = [
        cache.get_or_load_with_tier(disk_refs[0], reader.read).tier,
        cache.get_or_load_with_tier(disk_refs[0], reader.read).tier,
        cache.get_or_load_with_tier(disk_refs[0], reader.read).tier,
    ]
    if hot_disk_tiers != [CacheTier.COLD_STORAGE, CacheTier.COLD_STORAGE, CacheTier.LOCAL_DISK]:
        issues.append(f"disk hot-chunk tiers were {[tier.value for tier in hot_disk_tiers]}")

    hot_memory_tiers = [
        cache.get_or_load_with_tier(memory_refs[0], reader.read).tier,
        cache.get_or_load_with_tier(memory_refs[0], reader.read).tier,
        cache.get_or_load_with_tier(memory_refs[0], reader.read).tier,
    ]
    if hot_memory_tiers != [CacheTier.COLD_STORAGE, CacheTier.COLD_STORAGE, CacheTier.LOCAL_DISK]:
        issues.append(f"memory hot-chunk tiers were {[tier.value for tier in hot_memory_tiers]}")

    for ref in disk_refs[1:]:
        cache.get_or_load_with_tier(ref, reader.read)
        cache.get_or_load_with_tier(ref, reader.read)

    stats = cache.stats()
    if stats.local_promotions < 4:
        issues.append(f"expected at least four local promotions, observed {stats.local_promotions}")
    if stats.local_evictions < 3:
        issues.append(f"expected at least three local evictions, observed {stats.local_evictions}")
    if stats.local_items > 1 or stats.local_bytes > 10:
        issues.append(f"local tier exceeded budget: items={stats.local_items}, bytes={stats.local_bytes}")

    cpu_probe = _probe_cpu_to_local_promotion(work_dir / "cpu-to-local")
    if not cpu_probe["ok"]:
        issues.extend(f"cpu_to_local: {issue}" for issue in cpu_probe["issues"])

    return {
        "ok": not issues,
        "issues": issues,
        "disk_hot_tiers": [tier.value for tier in hot_disk_tiers],
        "memory_hot_tiers": [tier.value for tier in hot_memory_tiers],
        "stats": _cache_stats_record(stats),
        "cpu_to_local_promotion": cpu_probe,
        "local_dir": str(local_dir),
        "authoritative_dir": str(authoritative_dir),
    }


def _probe_cpu_to_local_promotion(work_dir: Path) -> dict[str, Any]:
    authoritative_dir = work_dir / "authoritative-shards"
    local_dir = work_dir / "local-tier"
    authoritative_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    refs = write_kvpack(
        authoritative_dir / "cpu.kvpack",
        (PackChunk(_document_key("cpu-hot"), b"cpu001", 6, "q8", "probe-v1"),),
        align_bytes=1,
    )
    reader = RoutedRangeReader()
    cache = ChunkCache(
        cpu_max_bytes=64,
        local_dir=local_dir,
        local_max_bytes=64,
        local_promotion_threshold=2,
    )
    tiers = [
        cache.get_or_load_with_tier(refs[0], reader.read).tier,
        cache.get_or_load_with_tier(refs[0], reader.read).tier,
    ]
    stats = cache.stats()
    issues: list[str] = []
    if tiers != [CacheTier.COLD_STORAGE, CacheTier.CPU]:
        issues.append(f"CPU promotion tiers were {[tier.value for tier in tiers]}")
    if stats.local_promotions != 1:
        issues.append(f"expected one local promotion from CPU hit, observed {stats.local_promotions}")
    return {
        "ok": not issues,
        "issues": issues,
        "tiers": [tier.value for tier in tiers],
        "stats": _cache_stats_record(stats),
    }


def _vllm_offloading_config_issues(config: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    if config.get("kv_connector") != "OffloadingConnector":
        issues.append("vLLM offload config must use OffloadingConnector")
    if config.get("kv_role") != "kv_both":
        issues.append("vLLM offload config must default to kv_both")
    extra_config = config.get("kv_connector_extra_config")
    if not isinstance(extra_config, Mapping):
        return [*issues, "vLLM offload config extra config must be an object"]
    expected = {
        "cpu_bytes_to_use": 2 * 1024**3,
        "block_size": 64,
        "eviction_policy": "lru",
        "offload_prompt_only": True,
        "spec_name": "TieringOffloadingSpec",
    }
    for key, value in expected.items():
        if extra_config.get(key) != value:
            issues.append(f"vLLM offload config {key} must be {value!r}")
    secondary_tiers = extra_config.get("secondary_tiers")
    if not isinstance(secondary_tiers, list) or not secondary_tiers:
        issues.append("vLLM offload config must include a filesystem secondary tier")
    elif not isinstance(secondary_tiers[0], Mapping) or secondary_tiers[0].get("type") != "fs":
        issues.append("vLLM offload config secondary tier must be filesystem-backed")
    return issues


def _find_vllm_offloading_connector() -> dict[str, Any]:
    package = _package_probe("vllm")
    if not package["found"]:
        return {
            "found": False,
            "package": package,
            "module": None,
            "symbol": None,
            "attempts": ["vllm package not installed"],
        }

    attempts: list[str] = []
    direct_modules = (
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector.connector",
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector.offloading_connector",
    )
    for module_name in direct_modules:
        found = _symbol_in_module(module_name, ("OffloadingConnector",))
        attempts.extend(found["attempts"])
        if found["found"]:
            return {"found": True, "package": package, **found}

    scanned = _scan_vllm_kv_connector_modules()
    attempts.extend(scanned["attempts"])
    if scanned["found"]:
        return {"found": True, "package": package, **scanned}
    return {
        "found": False,
        "package": package,
        "module": None,
        "symbol": None,
        "attempts": attempts,
    }


def _scan_vllm_kv_connector_modules() -> dict[str, Any]:
    root_module_name = "vllm.distributed.kv_transfer.kv_connector.v1"
    try:
        root_module = importlib.import_module(root_module_name)
    except Exception as exc:  # pragma: no cover - depends on optional vLLM runtime.
        return {
            "found": False,
            "module": None,
            "symbol": None,
            "attempts": [f"{root_module_name}: {type(exc).__name__}: {exc}"],
        }
    root_paths = getattr(root_module, "__path__", None)
    if root_paths is None:
        return {
            "found": False,
            "module": None,
            "symbol": None,
            "attempts": [f"{root_module_name} has no package path"],
        }
    attempts: list[str] = []
    for module_info in pkgutil.walk_packages(root_paths, root_module.__name__ + "."):
        if "offload" not in module_info.name:
            continue
        found = _symbol_in_module(module_info.name, ("OffloadingConnector",))
        attempts.extend(found["attempts"])
        if found["found"]:
            return found
    return {"found": False, "module": None, "symbol": None, "attempts": attempts}


def _symbol_in_module(module_name: str, symbols: Sequence[str]) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - depends on optional serving runtimes.
        return {
            "found": False,
            "module": module_name,
            "symbol": None,
            "attempts": [f"{module_name}: {type(exc).__name__}: {exc}"],
        }
    for symbol in symbols:
        if hasattr(module, symbol):
            return {
                "found": True,
                "module": module_name,
                "symbol": symbol,
                "attempts": [f"{module_name}: found {symbol}"],
            }
    return {
        "found": False,
        "module": module_name,
        "symbol": None,
        "attempts": [f"{module_name}: none of {list(symbols)!r} found"],
    }


def _package_probe(package_name: str) -> dict[str, Any]:
    found = importlib.util.find_spec(package_name) is not None
    version = None
    if found:
        try:
            version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            version = None
    return {"found": found, "version": version}


def _environment_record() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "nvidia_smi": _nvidia_smi_record(),
    }


def _nvidia_smi_record() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if completed.returncode != 0:
        return {"available": False, "error": completed.stderr.strip()}
    gpus = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        name, total_mib, used_mib, util_percent = parts
        gpus.append(
            {
                "name": name,
                "memory_total_mib": _int_or_none(total_mib),
                "memory_used_mib": _int_or_none(used_mib),
                "utilization_percent": _int_or_none(util_percent),
            }
        )
    return {"available": True, "gpus": gpus}


def _cache_stats_record(stats: Any) -> dict[str, Any]:
    return {
        "cpu_hits": stats.cpu_hits,
        "local_hits": stats.local_hits,
        "cold_misses": stats.cold_misses,
        "cpu_items": stats.cpu_items,
        "cpu_bytes": stats.cpu_bytes,
        "cpu_max_bytes": stats.cpu_max_bytes,
        "local_items": stats.local_items,
        "local_bytes": stats.local_bytes,
        "local_max_bytes": stats.local_max_bytes,
        "local_promotions": stats.local_promotions,
        "local_evictions": stats.local_evictions,
    }


def _document_key(chunk_id: str) -> KVCacheKey:
    return KVCacheKey.for_document(
        model_id="qwen3-4b-q4",
        lora_id="base",
        prompt_template_version="probe-v1",
        document_id="runtime-kv-offload-probe",
        chunk_type=DocumentChunkType.DOCUMENT_CHUNK,
        chunk_id=chunk_id,
        content_hash=chunk_id,
    )


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe Cachet's hierarchical document-KV persistence and validated "
            "runtime KV offload launch configs."
        )
    )
    parser.add_argument("--work-dir", required=True, help="Scratch directory for synthetic KV shards.")
    parser.add_argument("--output-json", help="Write the probe record to this JSON file.")
    parser.add_argument(
        "--require-vllm-offloading-import",
        action="store_true",
        help="Fail the probe unless the installed vLLM package exposes OffloadingConnector.",
    )
    parser.add_argument(
        "--require-sglang-package",
        action="store_true",
        help="Fail the probe unless the sglang package is installed.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    record = run_runtime_kv_offload_probe(
        work_dir=args.work_dir,
        require_vllm_offloading_import=args.require_vllm_offloading_import,
        require_sglang_package=args.require_sglang_package,
    )
    output = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0 if record["ok"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
