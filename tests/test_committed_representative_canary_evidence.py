import json
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Any, cast

import pytest

from document_kv_cache.benchmark_gates import (
    CacheStateAttestation,
    benchmark_evidence_gate_to_record,
    evaluate_benchmark_evidence_gate,
    evaluate_benchmark_publication_gate,
)
from document_kv_cache.benchmark_runner import (
    BENCHMARK_RUN_RECORD_TYPE,
    benchmark_record_aggregate_issues,
    benchmark_record_payload_digest,
    benchmark_run_result_from_record,
)
from document_kv_cache.benchmarks import BASELINE_PREFILL_ARM
from document_kv_cache.canary_orchestration import (
    FULL_PREFIX_CANARY_ARM,
    REPRESENTATIVE_CANARY_ARM_IDS,
    REPRESENTATIVE_CANARY_MODEL_ID,
    REPRESENTATIVE_CANARY_MODEL_REVISION,
    VANILLA_CANARY_ARM,
)
from document_kv_cache.release_evidence import (
    SGLANG_REPRESENTATIVE_CANARY_EVIDENCE_RECORD_TYPE,
    sglang_representative_canary_evidence_issues,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
APPENDIX_ROOT = (
    REPO_ROOT / "benchmarks" / "appendix" / "representative-bf16-qwen3-4b-canaries"
)
FINAL_SOURCE_COMMIT = "6e0f501a52c6b19f66d36e53a3fe6035b4b36ea2"
FINAL_WHEEL_SHA256 = "d820c01c5bee4d3bcb1e4338e4081c1ea9b4b59c8cb725588d7b973c07fe6f47"
EXPECTED_APPENDIX_FILES = {
    "README.md",
    "g5-vllm-8k-64-three-arm-canary.json",
    "g6-sglang-4k-32-paired-smoke-evidence.json",
    "g6-vllm-16k-256-three-arm-canary.json",
    "g6-vllm-8k-64-three-arm-canary.json",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SANITIZED_EXAMPLE_ID_RE = re.compile(r"^[0-9a-f]{24}$")
PAT_RE = re.compile(r"dapi[A-Za-z0-9_-]{16,}")
ALLOWED_STORAGE_IDENTITIES = {
    "local_nvme:/local_disk0:1x900gb",
    "local_nvme:/local_disk0:2x450gb",
}
FORBIDDEN_EVIDENCE_TEXT = (
    "/dbfs/",
    "/Users/",
    "/Volumes/",
    "/Workspace/",
    ".databricks.com",
    ".azuredatabricks.net",
    "@opentable",
    "cachet-hotpotqa-",
    "dbfs:",
    "file://",
    "http://",
    "https://",
)


@dataclass(frozen=True, slots=True)
class VLLMEvidenceCase:
    filename: str
    suite_id: str
    hardware_target: str
    hardware_fingerprint: str
    storage_identity: str
    input_tokens: int
    output_tokens: int
    loaded_tokens: int


VLLM_EVIDENCE_CASES = (
    VLLMEvidenceCase(
        filename="g6-vllm-8k-64-three-arm-canary.json",
        suite_id="g6-vllm-8k-64",
        hardware_target="aws-g6-l4",
        hardware_fingerprint=(
            "aws:g6.8xlarge:gpu=l4x1:cpu=32:ram_mib=131072:local_disks=2x450gb"
        ),
        storage_identity="local_nvme:/local_disk0:2x450gb",
        input_tokens=8_192,
        output_tokens=64,
        loaded_tokens=8_144,
    ),
    VLLMEvidenceCase(
        filename="g6-vllm-16k-256-three-arm-canary.json",
        suite_id="g6-vllm-16k-256",
        hardware_target="aws-g6-l4",
        hardware_fingerprint=(
            "aws:g6.8xlarge:gpu=l4x1:cpu=32:ram_mib=131072:local_disks=2x450gb"
        ),
        storage_identity="local_nvme:/local_disk0:2x450gb",
        input_tokens=16_384,
        output_tokens=256,
        loaded_tokens=16_336,
    ),
    VLLMEvidenceCase(
        filename="g5-vllm-8k-64-three-arm-canary.json",
        suite_id="g5-vllm-8k-64",
        hardware_target="aws-g5-a10g",
        hardware_fingerprint=(
            "aws:g5.8xlarge:gpu=a10gx1:cpu=32:ram_mib=131072:local_disks=1x900gb"
        ),
        storage_identity="local_nvme:/local_disk0:1x900gb",
        input_tokens=8_192,
        output_tokens=64,
        loaded_tokens=8_144,
    ),
)


def _load_canonical_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    decoded = json.loads(raw)
    assert isinstance(decoded, dict)
    record = cast(dict[str, Any], decoded)
    canonical = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    assert raw == canonical
    return record


def _cache_state_attestations(
    record: Mapping[str, Any],
) -> tuple[CacheStateAttestation, ...]:
    gate_inputs = record.get("gate_inputs")
    assert isinstance(gate_inputs, Mapping)
    assert gate_inputs.get("artifact_identities") == []
    rows = gate_inputs.get("cache_state_attestations")
    assert isinstance(rows, list)
    attestations = []
    for row in rows:
        assert isinstance(row, dict)
        assert set(row) == {
            "artifact_id",
            "bytes_read",
            "cache_method",
            "cold_read_attested",
            "direct_io",
            "eviction_requested",
            "eviction_succeeded",
            "expected_bytes",
            "expected_tokens",
            "loaded_tokens",
            "payload_cache_hit",
            "record_type",
            "request_id",
            "source",
            "successful_loads",
        }
        assert row["record_type"] == "document_kv.cache_state_attestation.v1"
        claimed_cold_state = row["cold_read_attested"]
        values = {
            key: value
            for key, value in row.items()
            if key not in {"cold_read_attested", "record_type"}
        }
        attestation = CacheStateAttestation(**values)
        assert attestation.cold_read_attested is claimed_cold_state
        attestations.append(attestation)
    return tuple(attestations)


def _iter_strings(value: object, path: str = "$") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _iter_strings(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_strings(child, f"{path}[{index}]")


def _assert_no_secret_path_or_raw_text_leaks(record: Mapping[str, Any]) -> None:
    serialized = json.dumps(record, sort_keys=True)
    assert PAT_RE.search(serialized) is None
    for forbidden in FORBIDDEN_EVIDENCE_TEXT:
        assert forbidden not in serialized
    for path, value in _iter_strings(record):
        if value in ALLOWED_STORAGE_IDENTITIES:
            continue
        assert not value.startswith("/"), f"absolute path leaked at {path}"


def test_representative_canary_appendix_has_only_reviewed_files_and_provenance() -> (
    None
):
    actual_files = {
        path.relative_to(APPENDIX_ROOT).as_posix()
        for path in APPENDIX_ROOT.rglob("*")
        if path.is_file()
    }
    assert actual_files == EXPECTED_APPENDIX_FILES

    readme = (APPENDIX_ROOT / "README.md").read_text(encoding="utf-8")
    for identity in (
        FINAL_SOURCE_COMMIT,
        FINAL_WHEEL_SHA256,
        REPRESENTATIVE_CANARY_MODEL_ID,
        REPRESENTATIVE_CANARY_MODEL_REVISION,
    ):
        assert identity in readme
    for filename in sorted(EXPECTED_APPENDIX_FILES - {"README.md"}):
        digest = sha256((APPENDIX_ROOT / filename).read_bytes()).hexdigest()
        assert digest in readme


@pytest.mark.parametrize("case", VLLM_EVIDENCE_CASES, ids=lambda case: case.suite_id)
def test_committed_vllm_canary_is_canonical_and_replays_cold_joins(
    case: VLLMEvidenceCase,
) -> None:
    record = _load_canonical_json(APPENDIX_ROOT / case.filename)
    assert record.get("record_type") == BENCHMARK_RUN_RECORD_TYPE
    assert record.get("evidence_sanitized") is True
    assert benchmark_record_aggregate_issues(record) == ()
    v1_evidence = record.get("v1_evidence")
    assert isinstance(v1_evidence, Mapping)
    assert v1_evidence.get("ok") is False

    result = benchmark_run_result_from_record(record)
    manifest = result.experiment_manifest
    assert manifest is not None
    assert result.suite.suite_id == case.suite_id
    assert result.suite.hardware_target == case.hardware_target
    assert result.suite.datasets == ("hotpotqa",)
    assert len(result.suite.examples) == 2
    assert result.repeats == 3
    assert result.request_parallelism == 1
    assert result.isolate_arms is True
    assert result.prefix_cache_salt_mode == "per_request"
    assert tuple(arm.arm_id for arm in result.arms) == REPRESENTATIVE_CANARY_ARM_IDS
    assert manifest.input_tokens_target == case.input_tokens
    assert manifest.output_tokens_target == case.output_tokens
    assert manifest.example_count == 2
    assert manifest.complete_dataset_split is False
    assert manifest.measurement_scopes == ("latency", "quality")
    assert manifest.hardware_target == case.hardware_target
    assert manifest.hardware_fingerprint == case.hardware_fingerprint
    assert manifest.storage_identity == case.storage_identity
    assert manifest.cache_state == "cold"
    assert manifest.execution_isolation_mode == "separate_process_or_job"

    measurement_counts = Counter(row.arm_id for row in result.measurements)
    assert measurement_counts == Counter(
        {
            BASELINE_PREFILL_ARM: 6,
            FULL_PREFIX_CANARY_ARM: 6,
            VANILLA_CANARY_ARM: 6,
        }
    )
    assert len(result.measurements) == 18
    assert all(row.prompt_tokens == case.input_tokens for row in result.measurements)
    assert all(
        row.completion_tokens == case.output_tokens for row in result.measurements
    )
    assert all(row.error is None for row in result.measurements)
    assert {row.repeat_index for row in result.measurements} == {1, 2, 3}
    assert len({row.example_id for row in result.measurements}) == 2
    assert all(
        SANITIZED_EXAMPLE_ID_RE.fullmatch(row.example_id) is not None
        for row in result.measurements
    )
    assert all(row.output_text == "" for row in result.measurements)
    assert all(row.expected_answer is None for row in result.measurements)
    assert all(row.references == () for row in result.measurements)

    model_runtime = record["experiment_manifest"]["model_runtime"]
    assert model_runtime["canonical_model_id"] == REPRESENTATIVE_CANARY_MODEL_ID
    assert model_runtime["model_revision"] == REPRESENTATIVE_CANARY_MODEL_REVISION
    assert model_runtime["tokenizer_id"] == REPRESENTATIVE_CANARY_MODEL_ID
    assert model_runtime["tokenizer_revision"] == REPRESENTATIVE_CANARY_MODEL_REVISION
    assert model_runtime["model_dtype"] == "bfloat16"
    assert model_runtime["runtime_kv_dtype"] == "bfloat16"
    assert model_runtime["model_quantization"] == "none"
    assert (
        model_runtime["package_revisions"]["cachet-kv"]
        == f"wheel-sha256:{FINAL_WHEEL_SHA256}"
    )

    attestations = _cache_state_attestations(record)
    assert len(attestations) == 12
    assert all(attestation.cold_read_attested for attestation in attestations)
    assert all(
        attestation.expected_tokens == case.loaded_tokens
        and attestation.loaded_tokens == case.loaded_tokens
        and attestation.successful_loads == 1
        for attestation in attestations
    )
    cache_request_ids = Counter(
        row.request_id
        for row in result.measurements
        if row.arm_id in {FULL_PREFIX_CANARY_ARM, VANILLA_CANARY_ARM}
    )
    attestation_ids = Counter(attestation.request_id for attestation in attestations)
    assert cache_request_ids == attestation_ids
    assert len(cache_request_ids) == 12
    assert all(count == 1 for count in cache_request_ids.values())
    assert all(
        request_id is not None and SHA256_RE.fullmatch(request_id) is not None
        for request_id in cache_request_ids
    )
    measurement_contracts = Counter(
        (row.request_id, row.cache_method, row.artifact_id)
        for row in result.measurements
        if row.arm_id in {FULL_PREFIX_CANARY_ARM, VANILLA_CANARY_ARM}
    )
    attestation_contracts = Counter(
        (row.request_id, row.cache_method, row.artifact_id) for row in attestations
    )
    assert measurement_contracts == attestation_contracts

    payload_digest = benchmark_record_payload_digest(record)
    stored_gate = record.get("evidence_gate")
    assert isinstance(stored_gate, dict)
    assert stored_gate.get("benchmark_payload_digest") == payload_digest
    canary_gate = evaluate_benchmark_evidence_gate(
        result,
        policy="canary",
        cache_state_attestations=attestations,
        benchmark_payload_digest=payload_digest,
    )
    assert benchmark_evidence_gate_to_record(canary_gate) == stored_gate

    publication_gate = evaluate_benchmark_publication_gate(
        result,
        cache_state_attestations=attestations,
        benchmark_payload_digest=payload_digest,
    )
    assert publication_gate.checked_cache_requests == 12
    assert publication_gate.cold_attested_requests == 12
    assert publication_gate.ok is False
    assert not any(
        "cache-state attestation" in issue or "attests a different" in issue
        for issue in publication_gate.issues
    )
    _assert_no_secret_path_or_raw_text_leaks(record)


def test_committed_sglang_smoke_uses_safe_nonpublication_actuals() -> None:
    record = _load_canonical_json(
        APPENDIX_ROOT / "g6-sglang-4k-32-paired-smoke-evidence.json"
    )
    assert (
        record.get("record_type") == SGLANG_REPRESENTATIVE_CANARY_EVIDENCE_RECORD_TYPE
    )
    assert (
        sglang_representative_canary_evidence_issues(
            record,
            expected_cachet_wheel_sha256=FINAL_WHEEL_SHA256,
        )
        == ()
    )
    assert record.get("evidence_sanitized") is True
    assert record.get("publication_qualified") is False
    assert record.get("engine") == "sglang"
    assert record.get("hardware_target") == "aws-g6-l4"
    assert record.get("workload_profile") == "sglang-4k-32-v1"
    assert record.get("suite") == {
        "datasets": ["niah"],
        "examples": 1,
        "release_v1_suite": False,
        "repeats": 2,
        "scope": "live_synthetic_niah",
        "suite_id": "sglang-live-synthetic-niah",
    }

    provenance = record["model_provenance"]
    assert provenance["canonical_model_id"] == REPRESENTATIVE_CANARY_MODEL_ID
    assert provenance["model_revision"] == REPRESENTATIVE_CANARY_MODEL_REVISION
    assert provenance["tokenizer_id"] == REPRESENTATIVE_CANARY_MODEL_ID
    assert provenance["tokenizer_revision"] == REPRESENTATIVE_CANARY_MODEL_REVISION
    assert provenance["model_dtype"] == "bfloat16"
    assert provenance["runtime_kv_dtype"] == "bfloat16"
    assert provenance["model_quantization"] == "none"
    assert (
        provenance["package_revisions"]["cachet-kv"]
        == f"wheel-sha256:{FINAL_WHEEL_SHA256}"
    )

    measurements = record["measurements"]
    assert len(measurements) == 4
    assert Counter(
        (row["arm_id"], row["repeat_index"]) for row in measurements
    ) == Counter(
        {
            (BASELINE_PREFILL_ARM, 1): 1,
            (BASELINE_PREFILL_ARM, 2): 1,
            ("document_kv_cache", 1): 1,
            ("document_kv_cache", 2): 1,
        }
    )
    assert all(
        row["prompt_tokens"] == 205
        and row["logical_prompt_tokens"] == 205
        and row["runtime_prompt_tokens"] == 205
        and row["completion_tokens"] == 7
        for row in measurements
    )
    assert all(
        SHA256_RE.fullmatch(row["example_identity_sha256"]) for row in measurements
    )
    cache_hits = record["cache_hit_validations"]
    assert len(cache_hits) == 2
    assert all(
        row["cache_hit"] is True
        and row["cache_request_prompt_tokens"] == 205
        and row["cache_request_cached_tokens"] == 176
        for row in cache_hits
    )
    _assert_no_secret_path_or_raw_text_leaks(record)
