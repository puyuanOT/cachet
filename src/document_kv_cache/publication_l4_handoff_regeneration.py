"""Regenerate one lost publication handoff bundle on single-GPU L4 producers.

The distributed L40S producers' qualification envelope and execution records
were removed from the campaign volume together with the 32k Q8 and 16k BF16
bundle bytes.  This path reruns the same pinned pre-RoPE Transformers
generator inside the same locked runtime on L4.  Each shard reproduces the
original producers' worker directories under the original producer root, so
the enriched rows carry the same absolute handoff paths and every regenerated
file can be compared with the surviving L40S reference (a staging attestation
or the closed bundle manifest).  The closer seals the tree with the same
bundle manifest builder the distributed coordinator used; plan, shard, and
execution records live under a separate regeneration root.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import shutil
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

from document_kv_cache import publication_bf16_handoff_generation as bf16_generation
from document_kv_cache import publication_latency_handoff_generation as q8_generation
from document_kv_cache.benchmarks import SUPPORTED_V1_DATASETS
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256,
)
from document_kv_cache.main_latency_inputs import (
    MainLatencyTokenizer,
    load_main_latency_tokenizer,
)
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
)
from document_kv_cache.publication_handoff_artifacts import (
    PUBLICATION_HANDOFF_STAGING_ATTESTATION_RECORD_TYPE,
    close_publication_latency_handoff_bundle,
    read_publication_latency_handoff_bundle,
    validate_publication_latency_handoff_bundle,
    write_publication_latency_handoff_bundle,
)
from document_kv_cache.storage import local_path
from document_kv_cache.transformers_generator import (
    build_pre_rope_transformers_kv_chunk_generator,
)

L4_REGENERATION_PLAN_RECORD_TYPE: Final = (
    "cachet.publication_l4_handoff_regeneration_plan.v1"
)
L4_REGENERATION_SHARD_RESULT_RECORD_TYPE: Final = (
    "cachet.publication_l4_handoff_regeneration_shard_result.v1"
)
L4_REGENERATION_EXECUTION_RECORD_TYPE: Final = (
    "cachet.publication_l4_handoff_regeneration_execution.v1"
)
L4_REGENERATION_SCHEMA_VERSION: Final = 1
L4_REGENERATION_EXECUTION_FILENAME: Final = "l4-handoff-regeneration.execution.json"
L4_REGENERATION_PRECISIONS: Final = {
    "q8_e5m2": "fp8_e5m2",
    "bf16": "bfloat16",
}
L4_REGENERATION_HARDWARE: Final = {
    "gpu_model": "NVIDIA L4",
    "hardware_target": "aws-g6-l4",
    "node_type_id": "g6.8xlarge",
}
L4_REGENERATION_REFERENCE_KINDS: Final = {"staging_attestation", "bundle_manifest"}
_ORIGINAL_WORKER_COUNT: Final = 16
_LOCAL_WORK_ROOT = Path("/local_disk0/cachet-l4-regeneration")
_PLAN_KEYS: Final = frozenset(
    {
        "campaign_id",
        "closed_record_sha256",
        "context_tokens",
        "durable_output_root",
        "execution_contract",
        "generation_root",
        "hardware",
        "input_bundle_sha256",
        "input_files",
        "precision",
        "reason",
        "record_type",
        "reference",
        "reference_datasets",
        "reference_files",
        "schema_version",
        "shards",
        "workers",
    }
)


def _canonical_json_bytes(value: Any, *, pretty: bool) -> bytes:
    if pretty:
        return (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return sha256(_canonical_json_bytes(value, pretty=False)).hexdigest()


def _closed_record_sha256(record: Mapping[str, Any]) -> str:
    return _canonical_sha256({**record, "closed_record_sha256": ""})


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cluster_path(uri: str) -> Path:
    return Path(local_path(uri)).expanduser().absolute()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _read_closed_json(
    path: Path,
    *,
    expected_file_sha256: str | None,
    record_type: str,
    label: str,
) -> dict[str, Any]:
    content = path.read_bytes()
    if expected_file_sha256 is not None:
        _require(
            sha256(content).hexdigest() == expected_file_sha256,
            f"{label} file SHA-256 differs",
        )
    record = json.loads(content)
    _require(isinstance(record, dict), f"{label} must be a JSON object")
    _require(content == _canonical_json_bytes(record, pretty=True), f"{label} is not canonical")
    _require(record.get("record_type") == record_type, f"{label} record type differs")
    _require(
        record.get("closed_record_sha256") == _closed_record_sha256(record),
        f"{label} closure differs",
    )
    return cast(dict[str, Any], record)


def _execution_config(precision: str) -> Any:
    if precision == "q8_e5m2":
        return q8_generation.build_publication_latency_handoff_execution_config(
            vllm_bitsandbytes_loader_source_sha256=(
                GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256
            ),
        )
    if precision == "bf16":
        return bf16_generation.build_publication_bf16_handoff_execution_config(
            vllm_bitsandbytes_loader_source_sha256=(
                GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256
            ),
        )
    raise ValueError(f"unsupported regeneration precision: {precision!r}")


def _generation_module(precision: str) -> Any:
    return q8_generation if precision == "q8_e5m2" else bf16_generation


def _execution_config_record(precision: str) -> dict[str, Any]:
    module = _generation_module(precision)
    return cast(dict[str, Any], module._execution_config_record(_execution_config(precision)))


def _reference_files(reference_record: Mapping[str, Any], kind: str) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for item in cast(Sequence[Mapping[str, Any]], reference_record["files"]):
        if item.get("role") == "enriched_dataset":
            continue
        if kind == "staging_attestation":
            files.append(
                {
                    "byte_count": int(item["source_byte_count"]),
                    "relative_name": str(item["relative_name"]),
                    "role": str(item["role"]),
                    "sha256": str(item["source_sha256"]),
                }
            )
        else:
            files.append(
                {
                    "byte_count": int(item["byte_count"]),
                    "relative_name": str(item["relative_name"]),
                    "role": str(item["role"]),
                    "sha256": str(item["sha256"]),
                }
            )
    return sorted(files, key=lambda item: item["relative_name"])


def _reference_datasets(reference_record: Mapping[str, Any], kind: str) -> list[dict[str, Any]]:
    """Return each reference dataset's producer-side and path-free digests."""

    source_sha256 = {
        str(item["relative_name"]): str(
            item["source_sha256"] if kind == "staging_attestation" else item["sha256"]
        )
        for item in cast(Sequence[Mapping[str, Any]], reference_record["files"])
        if item.get("role") == "enriched_dataset"
    }
    datasets = [
        {
            "dataset": str(item["dataset"]),
            "normalized_records_sha256": str(item["normalized_records_sha256"]),
            "relative_name": str(item["relative_name"]),
            "source_sha256": source_sha256[str(item["relative_name"])],
        }
        for item in cast(Sequence[Mapping[str, Any]], reference_record["datasets"])
    ]
    _require(
        [item["dataset"] for item in datasets] == list(SUPPORTED_V1_DATASETS),
        "reference datasets differ",
    )
    return datasets


def _worker_index_from_relative_name(relative_name: str) -> int:
    head = PurePosixPath(relative_name).parts[0]
    _require(head.startswith("worker-") and len(head) == 9, "reference worker directory is invalid")
    return int(head[len("worker-"):])


def build_l4_regeneration_plan(
    *,
    precision: str,
    context_tokens: int,
    input_bundle_sha256: str,
    input_files: Sequence[Mapping[str, Any]],
    input_paths: Mapping[str, Path],
    reference: Mapping[str, Any],
    reference_record: Mapping[str, Any],
    durable_output_root: str,
    generation_root: str,
    shard_worker_indices: Sequence[Sequence[int]],
    campaign_id: str,
    reason: str,
    tokenizer: MainLatencyTokenizer | None = None,
) -> dict[str, Any]:
    """Build one closed plan that reproduces the reference worker layout."""

    _require(precision in L4_REGENERATION_PRECISIONS, "unsupported precision")
    kind = str(reference.get("kind"))
    _require(kind in L4_REGENERATION_REFERENCE_KINDS, "unsupported reference kind")
    _require(
        reference_record.get("context_tokens") == context_tokens
        and reference_record.get("input_bundle_sha256") == input_bundle_sha256,
        "reference does not describe this context and input bundle",
    )
    if kind == "staging_attestation":
        _require(
            reference_record.get("record_type")
            == PUBLICATION_HANDOFF_STAGING_ATTESTATION_RECORD_TYPE,
            "reference attestation record type differs",
        )
    expected_dtype = L4_REGENERATION_PRECISIONS[precision]
    if kind == "bundle_manifest":
        layout = cast(Mapping[str, Any], reference_record["identity"])["layout_identity"]
        _require(layout.get("dtype") == expected_dtype, "reference manifest dtype differs")
    _require(
        tuple(item.get("dataset") for item in input_files) == tuple(SUPPORTED_V1_DATASETS),
        "input files must cover the four datasets in order",
    )
    resolved_tokenizer = tokenizer or load_main_latency_tokenizer()
    reference_files = _reference_files(reference_record, kind)
    payloads = [item for item in reference_files if item["role"] == "payload"]
    _require(
        len(payloads) == len(SUPPORTED_V1_DATASETS) * PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
        "reference payload coverage is incomplete",
    )
    worker_by_request: dict[str, int] = {}
    for item in payloads:
        name = PurePosixPath(item["relative_name"])
        _require(name.suffix == ".kv", "reference payload name is invalid")
        worker_by_request[name.stem] = _worker_index_from_relative_name(item["relative_name"])

    workers: dict[int, list[dict[str, Any]]] = {}
    for spec in input_files:
        dataset = str(spec["dataset"])
        path = input_paths[dataset]
        content = path.read_bytes()
        _require(sha256(content).hexdigest() == spec["sha256"], f"input {dataset} SHA-256 differs")
        rows = q8_generation._canonical_jsonl_records(path)
        _require(
            len(rows) == PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
            f"input {dataset} must contain 32 rows",
        )
        for row_index, row in enumerate(rows):
            _require(row.get("dataset") == dataset, "input row dataset differs from its file")
            example_id = str(row["example_id"])
            prefix = f"cachet-{dataset}-{example_id}-"
            matches = [request for request in worker_by_request if request.startswith(prefix)]
            _require(len(matches) == 1, f"reference request for {dataset}/{example_id} is not unique")
            worker_index = worker_by_request.pop(matches[0])
            contracts = q8_generation._cache_prefix_segment_token_contracts(
                row,
                tokenizer=resolved_tokenizer,
            )
            task_identity = {
                "context_tokens": context_tokens,
                "dataset": dataset,
                "example_id": example_id,
            }
            workers.setdefault(worker_index, []).append(
                {
                    "cache_prefix_tokens": sum(int(item["token_count"]) for item in contracts),
                    "context_tokens": context_tokens,
                    "dataset": dataset,
                    "example_id": example_id,
                    "input_token_slots": context_tokens,
                    "prepared_record_sha256": q8_generation._canonical_sha256(row),
                    "reference_request_id": matches[0],
                    "row_index": row_index,
                    "segment_token_contracts": contracts,
                    "segment_token_contracts_sha256": q8_generation._canonical_sha256(
                        contracts
                    ),
                    "task_id": (
                        f"{context_tokens}-{dataset}-"
                        f"{q8_generation._canonical_sha256(task_identity)[:20]}"
                    ),
                }
            )
    _require(not worker_by_request, "reference contains requests absent from the inputs")
    _require(sorted(workers) == list(range(_ORIGINAL_WORKER_COUNT)), "reference worker set differs")
    shard_records = [
        {"shard_index": index, "worker_indices": sorted(int(value) for value in indices)}
        for index, indices in enumerate(shard_worker_indices)
    ]
    assigned = sorted(value for shard in shard_records for value in shard["worker_indices"])
    _require(assigned == list(range(_ORIGINAL_WORKER_COUNT)), "shards must cover workers exactly once")
    record: dict[str, Any] = {
        "campaign_id": campaign_id,
        "closed_record_sha256": "",
        "context_tokens": context_tokens,
        "durable_output_root": durable_output_root,
        "execution_contract": _execution_config_record(precision),
        "generation_root": generation_root,
        "hardware": dict(L4_REGENERATION_HARDWARE),
        "input_bundle_sha256": input_bundle_sha256,
        "input_files": [dict(item) for item in input_files],
        "precision": precision,
        "reason": reason,
        "record_type": L4_REGENERATION_PLAN_RECORD_TYPE,
        "reference": dict(reference),
        "reference_datasets": _reference_datasets(reference_record, kind),
        "reference_files": reference_files,
        "schema_version": L4_REGENERATION_SCHEMA_VERSION,
        "shards": shard_records,
        "workers": [
            {
                "items": sorted(
                    workers[index],
                    key=lambda item: (item["dataset"], item["row_index"]),
                ),
                "worker_index": index,
            }
            for index in range(_ORIGINAL_WORKER_COUNT)
        ],
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    validate_l4_regeneration_plan(record)
    return record


def validate_l4_regeneration_plan(record: Mapping[str, Any]) -> None:
    _require(set(record) == _PLAN_KEYS, "regeneration plan keys differ")
    _require(record.get("record_type") == L4_REGENERATION_PLAN_RECORD_TYPE, "plan type differs")
    _require(record.get("schema_version") == L4_REGENERATION_SCHEMA_VERSION, "plan schema differs")
    _require(
        record.get("closed_record_sha256") == _closed_record_sha256(record),
        "regeneration plan closure differs",
    )
    precision = cast(str, record.get("precision"))
    _require(precision in L4_REGENERATION_PRECISIONS, "plan precision differs")
    _require(
        record.get("execution_contract") == _execution_config_record(precision),
        "plan execution contract differs from the frozen generator contract",
    )
    _require(record.get("hardware") == L4_REGENERATION_HARDWARE, "plan hardware differs")
    durable_root = str(record.get("durable_output_root", ""))
    generation_root = str(record.get("generation_root", ""))
    for label, value in (("durable output", durable_root), ("generation", generation_root)):
        _require(
            value.startswith("dbfs:/Volumes/")
            and value == value.rstrip("/")
            and not any(part in {"", ".", ".."} for part in value.split("/")[1:]),
            f"{label} root must be a canonical UC volume URI",
        )
    _require(
        not generation_root.startswith(durable_root + "/")
        and not durable_root.startswith(generation_root + "/")
        and generation_root != durable_root,
        "generation and record roots must be disjoint",
    )
    workers = cast(Sequence[Mapping[str, Any]], record.get("workers"))
    _require(
        [item.get("worker_index") for item in workers] == list(range(_ORIGINAL_WORKER_COUNT)),
        "plan workers differ",
    )
    task_ids = [
        item["task_id"] for worker in workers for item in cast(Sequence[Mapping[str, Any]], worker["items"])
    ]
    _require(
        len(task_ids) == len(set(task_ids))
        == len(SUPPORTED_V1_DATASETS) * PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
        "plan task coverage differs",
    )
    shards = cast(Sequence[Mapping[str, Any]], record.get("shards"))
    assigned = sorted(value for shard in shards for value in shard["worker_indices"])
    _require(assigned == list(range(_ORIGINAL_WORKER_COUNT)), "plan shards differ")


def read_l4_regeneration_plan(path: Path, *, expected_file_sha256: str) -> dict[str, Any]:
    record = _read_closed_json(
        path,
        expected_file_sha256=expected_file_sha256,
        record_type=L4_REGENERATION_PLAN_RECORD_TYPE,
        label="regeneration plan",
    )
    validate_l4_regeneration_plan(record)
    return record


def _read_plan_inputs(plan: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for spec in cast(Sequence[Mapping[str, Any]], plan["input_files"]):
        path = _cluster_path(str(spec["uri"]))
        _require(_file_sha256(path) == spec["sha256"], f"input {spec['dataset']} SHA-256 differs")
        for row_index, row in enumerate(q8_generation._canonical_jsonl_records(path)):
            rows[(str(spec["dataset"]), row_index)] = row
    return rows


def _probe_single_l4_hardware() -> dict[str, Any]:
    torch = importlib.import_module("torch")
    count = int(torch.cuda.device_count())
    _require(count == 1, "L4 regeneration requires exactly one visible GPU")
    name = str(torch.cuda.get_device_name(0))
    upper = name.upper()
    _require("L4" in upper and "L40" not in upper, f"producer GPU is not an NVIDIA L4: {name}")
    properties = torch.cuda.get_device_properties(0)
    return {
        "cuda_device_count": count,
        "cuda_device_name": name,
        "cuda_device_total_memory_bytes": int(properties.total_memory),
        "cuda_major": int(properties.major),
        "cuda_minor": int(properties.minor),
        **L4_REGENERATION_HARDWARE,
    }


def _worker_files(context_root: Path, worker_index: int) -> list[dict[str, Any]]:
    worker_root = context_root / f"worker-{worker_index:02d}"
    _require(worker_root.is_dir() and not worker_root.is_symlink(), "worker output is missing")
    files: list[dict[str, Any]] = []
    for path in sorted(worker_root.rglob("*")):
        _require(not path.is_symlink(), "worker output contains a symlink")
        if path.is_file():
            files.append(
                {
                    "byte_count": path.stat().st_size,
                    "relative_name": path.relative_to(context_root).as_posix(),
                    "sha256": _file_sha256(path),
                }
            )
    _require(bool(files), "worker produced no files")
    return files


def _compare_with_reference(
    produced: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    produced_by_name = {str(item["relative_name"]): item for item in produced}
    missing = sorted(set(reference) - set(produced_by_name))
    extra = sorted(set(produced_by_name) - set(reference))
    identical = {"handoff_json": 0, "payload": 0}
    size_equal = {"handoff_json": 0, "payload": 0}
    total = {"handoff_json": 0, "payload": 0}
    for name, item in produced_by_name.items():
        expected = reference.get(name)
        if expected is None:
            continue
        role = str(expected["role"])
        total[role] += 1
        if item["byte_count"] == expected["byte_count"]:
            size_equal[role] += 1
        if item["sha256"] == expected["sha256"]:
            identical[role] += 1
    return {
        "byte_identical_files": identical,
        "compared_files": total,
        "extra_relative_names": extra,
        "missing_relative_names": missing,
        "size_equal_files": size_equal,
    }


def run_l4_regeneration_shard(
    plan_uri: str,
    *,
    expected_plan_sha256: str,
    shard_index: int,
    cloud_run_id: str,
    task_run_id: str,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Generate every planned original worker directory assigned to one shard."""

    plan_path = _cluster_path(plan_uri)
    plan = read_l4_regeneration_plan(plan_path, expected_file_sha256=expected_plan_sha256)
    shards = cast(Sequence[Mapping[str, Any]], plan["shards"])
    _require(0 <= shard_index < len(shards), "shard index is outside the plan")
    shard = shards[shard_index]
    precision = cast(str, plan["precision"])
    module = _generation_module(precision)
    context_tokens = int(plan["context_tokens"])
    root = _cluster_path(str(plan["durable_output_root"]))
    generation_root = _cluster_path(str(plan["generation_root"]))
    context_root = generation_root / "pending" / str(context_tokens)
    result_path = root / "shard-results" / f"shard-{shard_index:02d}.json"
    _require(not result_path.exists(), "shard result already exists")
    worker_indices = [int(value) for value in shard["worker_indices"]]
    for worker_index in worker_indices:
        _require(
            not (context_root / f"worker-{worker_index:02d}").exists(),
            "shard worker output already exists",
        )
    rows = _read_plan_inputs(plan)
    config = _execution_config(precision)
    _require(
        module._execution_config_record(config) == plan["execution_contract"],
        "runtime generator contract differs from the plan",
    )
    module._apply_production_generator_environment(config)
    module._verify_installed_vllm_bitsandbytes_loader_source(config)
    observed_hardware = _probe_single_l4_hardware()
    work_root = _LOCAL_WORK_ROOT / f"shard-{shard_index:02d}"
    _require(not work_root.exists(), "shard work root already exists")
    work_root.mkdir(parents=True)
    reference = {str(item["relative_name"]): item for item in plan["reference_files"]}
    shard_start = clock()
    generator = None
    try:
        generator = build_pre_rope_transformers_kv_chunk_generator()
        if precision == "q8_e5m2":
            q8_generation._validate_publication_generator(
                generator,
                config=config,
                enforce_production_identity=True,
            )
        else:
            bf16_generation._validate_generator(generator, config=config)
        initialized = clock()
        workers_out: list[dict[str, Any]] = []
        all_files: list[dict[str, Any]] = []
        for worker_index in worker_indices:
            worker = cast(Sequence[Mapping[str, Any]], plan["workers"])[worker_index]
            items = cast(Sequence[Mapping[str, Any]], worker["items"])
            worker_start = clock()
            if precision == "q8_e5m2":
                batch = q8_generation._execute_worker_context_batch(
                    worker_index=worker_index,
                    context_tokens=context_tokens,
                    items=items,
                    rows_by_task={
                        (context_tokens, dataset, row_index): row
                        for (dataset, row_index), row in rows.items()
                    },
                    work_root=work_root,
                    pending_root=generation_root / "pending",
                    generator=generator,
                    config=config,
                    clock=clock,
                )
            else:
                worker_work = work_root / f"worker-{worker_index:02d}"
                worker_work.mkdir()
                context_root.mkdir(parents=True, exist_ok=True)
                batch = bf16_generation._execute_worker_batch(
                    {"items": items, "worker_index": worker_index},
                    rows_by_task=rows,
                    work_root=worker_work,
                    bundle_root=context_root / f"worker-{worker_index:02d}",
                    generator=generator,
                    config=config,
                    clock=clock,
                )
            record_path = root / "shard-records" / str(context_tokens) / f"worker-{worker_index:02d}.jsonl"
            _write_exclusive(
                record_path,
                b"".join(_canonical_json_bytes(row, pretty=False) + b"\n" for row in batch.records),
            )
            files = _worker_files(context_root, worker_index)
            all_files.extend(files)
            workers_out.append(
                {
                    "cache_prefix_tokens": batch.cache_prefix_tokens,
                    "generation_seconds": batch.generation_seconds,
                    "partial_record_file": {
                        "byte_count": record_path.stat().st_size,
                        "record_count": len(batch.records),
                        "relative_path": record_path.relative_to(root).as_posix(),
                        "sha256": _file_sha256(record_path),
                    },
                    "task_ids": sorted(batch.task_ids),
                    "wall_seconds": clock() - worker_start,
                    "worker_index": worker_index,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "worker_done",
                        "files": len(files),
                        "seconds": round(clock() - worker_start, 1),
                        "worker_index": worker_index,
                    }
                ),
                flush=True,
            )
        shard_seconds = clock() - shard_start
        comparison = _compare_with_reference(
            all_files,
            {
                name: item
                for name, item in reference.items()
                if _worker_index_from_relative_name(name) in worker_indices
            },
        )
        result: dict[str, Any] = {
            "accounting": {
                "cache_prefix_tokens": sum(item["cache_prefix_tokens"] for item in workers_out),
                "generation_seconds": sum(item["generation_seconds"] for item in workers_out),
                "initialization_seconds": initialized - shard_start,
                "shard_wall_seconds": shard_seconds,
            },
            "bundle_files": sorted(all_files, key=lambda item: item["relative_name"]),
            "closed_record_sha256": "",
            "cloud_run_id": cloud_run_id,
            "context_tokens": context_tokens,
            "observed_hardware": observed_hardware,
            "plan_closed_record_sha256": plan["closed_record_sha256"],
            "precision": precision,
            "record_type": L4_REGENERATION_SHARD_RESULT_RECORD_TYPE,
            "reference_comparison": comparison,
            "schema_version": L4_REGENERATION_SCHEMA_VERSION,
            "shard_index": shard_index,
            "task_run_id": task_run_id,
            "workers": workers_out,
        }
        result["closed_record_sha256"] = _closed_record_sha256(result)
        _write_exclusive(result_path, _canonical_json_bytes(result, pretty=True))
        return result
    finally:
        if generator is not None:
            close = getattr(generator, "close", None)
            if callable(close):
                close()
        shutil.rmtree(work_root, ignore_errors=True)
        gc.collect()
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            pass
        else:
            torch.cuda.empty_cache()


def close_l4_regeneration(
    plan_uri: str,
    *,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    """Verify every shard, seal the bundle, and compare it with the reference."""

    plan = read_l4_regeneration_plan(
        _cluster_path(plan_uri),
        expected_file_sha256=expected_plan_sha256,
    )
    precision = cast(str, plan["precision"])
    context_tokens = int(plan["context_tokens"])
    reference = cast(Mapping[str, Any], plan["reference"])
    reference_path = _cluster_path(str(reference["uri"]))
    _require(_file_sha256(reference_path) == reference["sha256"], "reference file SHA-256 differs")
    reference_record = json.loads(reference_path.read_bytes())
    root = _cluster_path(str(plan["durable_output_root"]))
    generation_root_uri = str(plan["generation_root"])
    generation_root = _cluster_path(generation_root_uri)
    execution_path = root / L4_REGENERATION_EXECUTION_FILENAME
    _require(not execution_path.exists(), "regeneration execution already exists")
    context_root = generation_root / "pending" / str(context_tokens)
    plan_task_ids = sorted(
        item["task_id"]
        for worker in cast(Sequence[Mapping[str, Any]], plan["workers"])
        for item in cast(Sequence[Mapping[str, Any]], worker["items"])
    )
    shard_bindings: list[dict[str, Any]] = []
    produced_files: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    task_ids: list[str] = []
    observed: set[str] = set()
    for shard in cast(Sequence[Mapping[str, Any]], plan["shards"]):
        index = int(shard["shard_index"])
        path = root / "shard-results" / f"shard-{index:02d}.json"
        result = _read_closed_json(
            path,
            expected_file_sha256=None,
            record_type=L4_REGENERATION_SHARD_RESULT_RECORD_TYPE,
            label=f"shard {index} result",
        )
        _require(
            result.get("plan_closed_record_sha256") == plan["closed_record_sha256"]
            and result.get("shard_index") == index
            and [item["worker_index"] for item in result["workers"]] == list(shard["worker_indices"]),
            f"shard {index} result does not belong to this plan",
        )
        hardware = cast(Mapping[str, Any], result["observed_hardware"])
        _require(
            {key: hardware.get(key) for key in L4_REGENERATION_HARDWARE} == L4_REGENERATION_HARDWARE
            and "L4" in str(hardware.get("cuda_device_name", "")).upper(),
            f"shard {index} did not run on L4",
        )
        observed.add(_canonical_sha256(hardware))
        for worker in cast(Sequence[Mapping[str, Any]], result["workers"]):
            partial = cast(Mapping[str, Any], worker["partial_record_file"])
            partial_path = root / str(partial["relative_path"])
            _require(_file_sha256(partial_path) == partial["sha256"], "partial record SHA-256 differs")
            records.extend(q8_generation._canonical_jsonl_records(partial_path))
            task_ids.extend(cast(list[str], worker["task_ids"]))
        produced_files.extend(cast(list[dict[str, Any]], result["bundle_files"]))
        shard_bindings.append(
            {
                "closed_record_sha256": result["closed_record_sha256"],
                "cloud_run_id": result["cloud_run_id"],
                "observed_hardware": dict(hardware),
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": _file_sha256(path),
                "shard_index": index,
                "task_run_id": result["task_run_id"],
            }
        )
    _require(sorted(task_ids) == plan_task_ids, "shard task coverage differs from the plan")
    _require(
        Counter((row["dataset"], row["example_id"]) for row in records)
        == Counter(
            (item["dataset"], item["example_id"])
            for worker in cast(Sequence[Mapping[str, Any]], plan["workers"])
            for item in cast(Sequence[Mapping[str, Any]], worker["items"])
        ),
        "enriched row coverage differs from the plan",
    )
    dataset_paths: dict[str, Path] = {}
    for dataset in SUPPORTED_V1_DATASETS:
        selected = sorted(
            (row for row in records if row.get("dataset") == dataset),
            key=lambda row: str(row["example_id"]),
        )
        _require(len(selected) == PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET, f"{dataset} coverage differs")
        path = context_root / "datasets" / f"{dataset}.jsonl"
        # The bundle closer only accepts json.dumps(sort_keys=True) rows; the BF16
        # module's own writer emits compact rows it rejects. The original BF16
        # bundle's dataset digests reproduce only in the closer's format.
        q8_generation._write_handoff_dataset_jsonl_exclusive(selected, path)
        dataset_paths[dataset] = path
    q8_generation._sync_tree(context_root)
    manifest = close_publication_latency_handoff_bundle(
        context_root,
        dataset_paths,
        context_tokens=context_tokens,
        input_bundle_sha256=str(plan["input_bundle_sha256"]),
    )
    if precision == "q8_e5m2":
        q8_generation._validate_publication_manifest_contract(manifest)
    else:
        bf16_generation._validate_bf16_manifest_contract(manifest)
    manifest_files = {
        str(item["relative_name"]): item
        for item in cast(Sequence[Mapping[str, Any]], manifest["files"])
        if item.get("role") != "enriched_dataset"
    }
    _require(
        {name: (item["sha256"], item["byte_count"]) for name, item in manifest_files.items()}
        == {
            str(item["relative_name"]): (item["sha256"], item["byte_count"])
            for item in produced_files
        },
        "sealed bundle files differ from the shard results",
    )
    portable = str(manifest["portable_bundle_sha256"])
    bundle_name = f"{context_tokens}-{portable}"
    source_root = generation_root / "bundles" / bundle_name
    manifest_path = generation_root / "manifests" / f"{bundle_name}.json"
    _require(
        not source_root.exists() and not manifest_path.exists(),
        "content-addressed bundle collision",
    )
    source_root.parent.mkdir(parents=True, exist_ok=True)
    os.rename(context_root, source_root)
    validate_publication_latency_handoff_bundle(manifest, bundle_root=source_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_publication_latency_handoff_bundle(manifest, manifest_path)
    _require(
        read_publication_latency_handoff_bundle(manifest_path) == manifest,
        "written manifest does not round-trip",
    )
    pending = generation_root / "pending"
    if pending.exists() and not any(pending.iterdir()):
        pending.rmdir()

    reference_kind = str(reference["kind"])
    comparison = _compare_with_reference(
        [
            {"byte_count": item["byte_count"], "relative_name": name, "sha256": item["sha256"]}
            for name, item in manifest_files.items()
        ],
        {str(item["relative_name"]): item for item in plan["reference_files"]},
    )
    manifest_datasets = {
        str(item["dataset"]): item for item in cast(Sequence[Mapping[str, Any]], manifest["datasets"])
    }
    reference_datasets = cast(Sequence[Mapping[str, Any]], plan["reference_datasets"])
    comparison["dataset_normalized_records_equal"] = sum(
        manifest_datasets[str(item["dataset"])]["normalized_records_sha256"]
        == item["normalized_records_sha256"]
        for item in reference_datasets
    )
    comparison["dataset_source_bytes_equal"] = sum(
        manifest_datasets[str(item["dataset"])]["sha256"] == item["source_sha256"]
        for item in reference_datasets
    )
    if reference_kind == "staging_attestation":
        reference_closed = reference_record.get("bundle_closed_record_sha256")
        closure_checks: dict[str, Any] = {}
    else:
        reference_closed = reference_record.get("closed_record_sha256")
        closure_checks = {
            field_name: manifest.get(field_name) == reference_record.get(field_name)
            for field_name in (
                "identity",
                "portable_bundle_sha256",
                "request_closure_sha256",
                "token_closure_sha256",
            )
        }
    comparison["bundle_closed_record_equal_reference"] = (
        manifest["closed_record_sha256"] == reference_closed
    )
    comparison["reference_closure_checks"] = closure_checks
    execution: dict[str, Any] = {
        "bundle": {
            "closed_record_sha256": manifest["closed_record_sha256"],
            "context_tokens": context_tokens,
            "input_bundle_sha256": manifest["input_bundle_sha256"],
            "kv_dtype": L4_REGENERATION_PRECISIONS[precision],
            "manifest_sha256": _file_sha256(manifest_path),
            "manifest_uri": f"{generation_root_uri}/manifests/{bundle_name}.json",
            "portable_bundle_sha256": portable,
            "source_root_uri": f"{generation_root_uri}/bundles/{bundle_name}",
        },
        "closed_record_sha256": "",
        "observed_hardware_identity_sha256": sorted(observed),
        "plan": {
            "closed_record_sha256": plan["closed_record_sha256"],
            "sha256": expected_plan_sha256,
            "uri": plan_uri,
        },
        "precision": precision,
        "record_type": L4_REGENERATION_EXECUTION_RECORD_TYPE,
        "reference": dict(reference),
        "reference_comparison": comparison,
        "schema_version": L4_REGENERATION_SCHEMA_VERSION,
        "shards": shard_bindings,
    }
    execution["closed_record_sha256"] = _closed_record_sha256(execution)
    _write_exclusive(execution_path, _canonical_json_bytes(execution, pretty=True))
    return execution


def read_l4_regeneration_execution(path: Path, *, expected_file_sha256: str) -> dict[str, Any]:
    return _read_closed_json(
        path,
        expected_file_sha256=expected_file_sha256,
        record_type=L4_REGENERATION_EXECUTION_RECORD_TYPE,
        label="regeneration execution",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="L4 publication handoff regeneration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("run-l4-regeneration-shard")
    shard.add_argument("--plan-json", required=True)
    shard.add_argument("--expected-plan-sha256", required=True)
    shard.add_argument("--shard-index", required=True, type=int)
    shard.add_argument("--cloud-run-id", required=True)
    shard.add_argument("--task-run-id", required=True)
    close = subparsers.add_parser("close-l4-regeneration")
    close.add_argument("--plan-json", required=True)
    close.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run-l4-regeneration-shard":
            result = run_l4_regeneration_shard(
                args.plan_json,
                expected_plan_sha256=args.expected_plan_sha256,
                shard_index=args.shard_index,
                cloud_run_id=args.cloud_run_id,
                task_run_id=args.task_run_id,
            )
            summary = {
                "closed_record_sha256": result["closed_record_sha256"],
                "reference_comparison": result["reference_comparison"],
                "shard_index": result["shard_index"],
            }
        else:
            result = close_l4_regeneration(
                args.plan_json,
                expected_plan_sha256=args.expected_plan_sha256,
            )
            summary = {
                "bundle": result["bundle"],
                "closed_record_sha256": result["closed_record_sha256"],
                "reference_comparison": result["reference_comparison"],
            }
        print(json.dumps({"ok": True, **summary}, sort_keys=True), flush=True)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(
            json.dumps({"error": str(exc), "error_type": type(exc).__name__, "ok": False}),
            flush=True,
        )
        return 1
    return 0



__all__ = [
    "L4_REGENERATION_EXECUTION_FILENAME",
    "L4_REGENERATION_EXECUTION_RECORD_TYPE",
    "L4_REGENERATION_HARDWARE",
    "L4_REGENERATION_PLAN_RECORD_TYPE",
    "L4_REGENERATION_PRECISIONS",
    "L4_REGENERATION_REFERENCE_KINDS",
    "L4_REGENERATION_SCHEMA_VERSION",
    "L4_REGENERATION_SHARD_RESULT_RECORD_TYPE",
    "build_l4_regeneration_plan",
    "close_l4_regeneration",
    "main",
    "read_l4_regeneration_execution",
    "read_l4_regeneration_plan",
    "run_l4_regeneration_shard",
    "validate_l4_regeneration_plan",
]


if __name__ == "__main__":
    raise SystemExit(main())
