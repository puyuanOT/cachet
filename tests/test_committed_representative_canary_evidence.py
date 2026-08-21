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
from document_kv_cache.methods import default_method_registry
from document_kv_cache.release_evidence import (
    SGLANG_REPRESENTATIVE_CANARY_EVIDENCE_RECORD_TYPE,
    sglang_representative_canary_evidence_issues,
)
from document_kv_cache.reuse_contract import PositionHandling


REPO_ROOT = Path(__file__).resolve().parents[1]
APPENDIX_ROOT = (
    REPO_ROOT / "benchmarks" / "appendix" / "representative-bf16-qwen3-4b-canaries"
)
FINAL_SOURCE_COMMIT = "b4b142c79443fcca62b08044d0937298eab3f71d"
FINAL_WHEEL_SHA256 = "5d91052aa5e92db64c3ba21924ae1805b7671c8c19bdc600fb477956dca78f90"
FINAL_SGLANG_HANDOFF_GENERATION_SHA256 = (
    "49cf15b2d53f55a9f48594c120dc1cafe9d905c407a51116c8b54d5606eb405a"
)
FINAL_SGLANG_RAW_BENCHMARK_SHA256 = (
    "f032a85e1ed65c4082238092491175c7d7bf38acc9d00a449e4ab3d62f15f958"
)
FINAL_SGLANG_HANDOFF_GENERATION_PROVENANCE = {
    "content_digests": [
        {
            "artifact_sha256": "4af461e0122c38ec0ca730af74ae0ce873064bd3ef33e32ac74bc8e852b5a72f",
            "logical_prompt_sha256": "2b2270897173ac96dfdbdaaadf3161edbb8966399a6958eca90578646140df5e",
            "method_config_sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        }
    ],
    "generator_factory": (
        "document_kv_cache.transformers_generator:"
        "build_pre_rope_transformers_kv_chunk_generator"
    ),
    "generator_version": "5.3.0",
    "method_id": "vanilla_prefill",
    "method_version": "2",
    "raw_sidecar_sha256": FINAL_SGLANG_HANDOFF_GENERATION_SHA256,
    "topology": {
        "attestation_sha256": "1dbb3f25d2997008d05a5023863ca1503e199b9412e1e5ffd63a03ca973d6814",
        "document_count": 1,
        "example_count": 1,
        "examples_sha256": "5fdbba056662c2dd584c22b0a37149393fd8eaba8243ca65b6585535ce4f21c5",
        "record_type": "document_kv.handoff_topology_attestation.v1",
        "schema_version": 1,
        "segment_count": 1,
        "topology_id": "per_document",
    },
}
CURRENT_METHOD_REGISTRY = default_method_registry()
EXPECTED_APPENDIX_FILES = {
    "README.md",
    "g5-vllm-8k-64-three-arm-canary-v2.json",
    "g6-sglang-4k-32-paired-smoke-evidence-v2.json",
    "g6-vllm-16k-256-three-arm-canary-v2.json",
    "g6-vllm-8k-64-three-arm-canary-v2.json",
    "vanilla-v2-cold-optimization.json",
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
    canary_gate_ok: bool


VLLM_EVIDENCE_CASES = (
    VLLMEvidenceCase(
        filename="g6-vllm-8k-64-three-arm-canary-v2.json",
        suite_id="g6-vllm-8k-64",
        hardware_target="aws-g6-l4",
        hardware_fingerprint=(
            "aws:g6.8xlarge:gpu=l4x1:cpu=32:ram_mib=131072:local_disks=2x450gb"
        ),
        storage_identity="local_nvme:/local_disk0:2x450gb",
        input_tokens=8_192,
        output_tokens=64,
        loaded_tokens=8_144,
        canary_gate_ok=False,
    ),
    VLLMEvidenceCase(
        filename="g6-vllm-16k-256-three-arm-canary-v2.json",
        suite_id="g6-vllm-16k-256",
        hardware_target="aws-g6-l4",
        hardware_fingerprint=(
            "aws:g6.8xlarge:gpu=l4x1:cpu=32:ram_mib=131072:local_disks=2x450gb"
        ),
        storage_identity="local_nvme:/local_disk0:2x450gb",
        input_tokens=16_384,
        output_tokens=256,
        loaded_tokens=16_336,
        canary_gate_ok=True,
    ),
    VLLMEvidenceCase(
        filename="g5-vllm-8k-64-three-arm-canary-v2.json",
        suite_id="g5-vllm-8k-64",
        hardware_target="aws-g5-a10g",
        hardware_fingerprint=(
            "aws:g5.8xlarge:gpu=a10gx1:cpu=32:ram_mib=131072:local_disks=1x900gb"
        ),
        storage_identity="local_nvme:/local_disk0:1x900gb",
        input_tokens=8_192,
        output_tokens=64,
        loaded_tokens=8_144,
        canary_gate_ok=False,
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
def test_vanilla_v2_vllm_canary_is_canonical_and_replays_cold_joins(
    case: VLLMEvidenceCase,
) -> None:
    record = _load_canonical_json(APPENDIX_ROOT / case.filename)
    assert record.get("record_type") == BENCHMARK_RUN_RECORD_TYPE
    assert record.get("evidence_sanitized") is True
    assert benchmark_record_aggregate_issues(record) == ()
    assert all(
        arm.get("source_revision") is None
        for arm in record["experiment_manifest"]["arms"]
    )
    v1_evidence = record.get("v1_evidence")
    assert isinstance(v1_evidence, Mapping)
    assert v1_evidence.get("ok") is False

    result = benchmark_run_result_from_record(record)
    manifest = result.experiment_manifest
    assert manifest is not None
    vanilla_arm = next(arm for arm in manifest.arms if arm.arm_id == VANILLA_CANARY_ARM)
    assert vanilla_arm.method_version == "2"
    assert vanilla_arm.runtime_environment.key_position_encoding == "pre_rope"
    assert vanilla_arm.runtime_environment.rope_theta == 5_000_000.0
    assert vanilla_arm.runtime_environment.rope_rotary_dim == 128
    vanilla_spec = CURRENT_METHOD_REGISTRY.get("vanilla_prefill")
    assert vanilla_spec.artifact_version == vanilla_arm.method_version
    assert vanilla_spec.pre_rope is True
    assert vanilla_spec.position_handling == PositionHandling.REROPE_AT_INJECTION
    assert vanilla_spec.generator_factory == (
        "document_kv_cache.transformers_generator:"
        "build_pre_rope_transformers_kv_chunk_generator"
    )
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
        method_registry=CURRENT_METHOD_REGISTRY,
        benchmark_payload_digest=payload_digest,
    )
    assert benchmark_evidence_gate_to_record(canary_gate) == stored_gate
    assert canary_gate.ok is case.canary_gate_ok
    if case.canary_gate_ok:
        assert canary_gate.issues == ()
    else:
        assert canary_gate.issues == (
            "hotpotqa:document_kv_cache:vanilla_prefill paired 'f1' lower "
            "bound -0.0625 exceeds allowed regression 0.02",
        )

    publication_gate = evaluate_benchmark_publication_gate(
        result,
        cache_state_attestations=attestations,
        method_registry=CURRENT_METHOD_REGISTRY,
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


def test_vanilla_v2_sglang_smoke_uses_safe_nonpublication_actuals() -> None:
    record = _load_canonical_json(
        APPENDIX_ROOT / "g6-sglang-4k-32-paired-smoke-evidence-v2.json"
    )
    assert (
        record.get("record_type") == SGLANG_REPRESENTATIVE_CANARY_EVIDENCE_RECORD_TYPE
    )
    assert (
        sglang_representative_canary_evidence_issues(
            record,
            expected_cachet_wheel_sha256=FINAL_WHEEL_SHA256,
            expected_handoff_generation_provenance=(
                FINAL_SGLANG_HANDOFF_GENERATION_PROVENANCE
            ),
        )
        == ()
    )
    assert record.get("evidence_sanitized") is True
    assert record.get("publication_qualified") is False
    assert record.get("engine") == "sglang"
    assert record.get("hardware_target") == "aws-g6-l4"
    assert record.get("workload_profile") == "sglang-4k-32-v1"
    assert record.get("raw_record_sha256") == FINAL_SGLANG_RAW_BENCHMARK_SHA256
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
    assert provenance["key_position_encoding"] == "pre_rope"
    assert provenance["rope_theta"] == 5_000_000.0
    assert provenance["rope_rotary_dim"] == 128
    assert (
        provenance["package_revisions"]["cachet-kv"]
        == f"wheel-sha256:{FINAL_WHEEL_SHA256}"
    )

    assert (
        record["handoff_generation_provenance"]
        == FINAL_SGLANG_HANDOFF_GENERATION_PROVENANCE
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
    comparison = record["comparisons"]
    assert len(comparison) == 1
    assert comparison[0]["ttft_speedup"] == pytest.approx(0.6370049784976665)
    assert comparison[0]["time_to_completion_speedup"] == pytest.approx(
        0.6561250595570515
    )
    assert comparison[0]["ttft_speedup"] < 1.0
    assert comparison[0]["time_to_completion_speedup"] < 1.0
    _assert_no_secret_path_or_raw_text_leaks(record)


def test_vanilla_v2_cold_optimization_is_a_six_job_matched_ablation() -> None:
    record = _load_canonical_json(APPENDIX_ROOT / "vanilla-v2-cold-optimization.json")
    assert record["record_type"] == "cachet.vanilla_v2_cold_optimization_evidence.v2"
    assert record["evidence_level"] == "canary"
    assert record["evidence_sanitized"] is True
    assert record["publication_qualified"] is False
    assert record["source_revision"] == FINAL_SOURCE_COMMIT
    assert record["wheel_sha256"] == FINAL_WHEEL_SHA256
    assert record["model_id"] == REPRESENTATIVE_CANARY_MODEL_ID
    assert record["model_revision"] == REPRESENTATIVE_CANARY_MODEL_REVISION
    assert record["position_contract"] == {
        "key_position_encoding": "pre_rope",
        "method_id": "vanilla_prefill",
        "method_version": "2",
        "position_handling": "rerope_at_injection",
        "rope_rotary_dim": 128,
        "rope_theta": 5_000_000.0,
    }
    assert record["submission_validation_flag_attestation"] == {
        "affected_direct_benchmark_ids": [
            "g6-vllm-8k-64-vanilla",
            "g6-vllm-16k-256-vanilla",
        ],
        "classification": "validation_only",
        "complete_effective_suite_and_manifest_settings_identical": True,
        "effective_benchmark_manifest_impact": "none",
        "matched_legacy_benchmark_ids": [
            "g6-vllm-8k-64-vanilla-legacy",
            "g6-vllm-16k-256-vanilla-legacy",
        ],
        "note": (
            "The representative-canary and workload-profile submission flags "
            "select validation policy only. They are absent from the matched "
            "generic legacy submissions and do not alter the emitted effective "
            "benchmark suite or manifest settings."
        ),
        "submission_flags": [
            "--representative-canary",
            "--representative-workload-profile",
        ],
    }

    expected_jobs = {
        "g6-vllm-8k-64-vanilla": (
            "auto",
            "direct_global_snapshot",
            False,
            8_192,
            64,
            2.9097911000000067,
        ),
        "g6-vllm-8k-64-vanilla-legacy": (
            "legacy",
            "legacy_segment_remerge",
            False,
            8_192,
            64,
            7.315308377999941,
        ),
        "g6-vllm-16k-256-vanilla": (
            "auto",
            "direct_global_snapshot",
            False,
            16_384,
            256,
            5.846155845999874,
        ),
        "g6-vllm-16k-256-vanilla-legacy": (
            "legacy",
            "legacy_segment_remerge",
            False,
            16_384,
            256,
            15.264030808500138,
        ),
        "g6-vllm-8k-64-vanilla-direct-profiled": (
            "direct",
            "direct_global_snapshot",
            True,
            8_192,
            64,
            2.901357098999938,
        ),
        "g6-vllm-8k-64-vanilla-legacy-profiled": (
            "legacy",
            "legacy_segment_remerge",
            True,
            8_192,
            64,
            7.581663182999932,
        ),
    }
    jobs = {job["benchmark_id"]: job for job in record["jobs"]}
    assert set(jobs) == set(expected_jobs)
    assert len(jobs) == 6

    for benchmark_id, expected in expected_jobs.items():
        configured, selected, profiled, input_tokens, output_tokens, ttft = expected
        job = jobs[benchmark_id]
        assert job["configured_strategy"] == configured
        assert job["selected_strategy"] == selected
        assert job["profiled"] is profiled
        assert job["input_tokens"] == input_tokens
        assert job["output_tokens"] == output_tokens
        assert job["metrics"]["requests"] == 6
        assert job["metrics"]["unique_examples"] == 2
        assert job["metrics"]["prompt_tokens"] == [input_tokens]
        assert job["metrics"]["completion_tokens"] == [output_tokens]
        assert job["metrics"]["ttft_seconds"]["p50"] == pytest.approx(ttft)
        assert SHA256_RE.fullmatch(job["raw_record_sha256"])
        assert SHA256_RE.fullmatch(job["raw_telemetry_sha256"])

        contract = job["load_contract"]
        assert contract["cold_load_joins"] == 6
        assert contract["successful_loads"] == 6
        assert contract["copy_byte_relation_verified_requests"] == 6
        assert contract["page_cache_eviction_succeeded_for_all"] is True
        assert contract["payload_cache_disabled_for_all"] is True
        assert contract["payload_mode"] == "segmented"
        assert contract["canonical_segmented_global_view"] is True
        assert contract["copy_metadata_retained"] is True
        assert contract["copy_count"] == 11
        assert contract["prefetch_event_count"] == 0

        load_metrics = job["load_metrics"]
        assert (
            load_metrics["checksum_validations_per_request"]
            == (contract["checksum_validations_per_request"])
        )
        payload_total = load_metrics["payload_bytes_per_request"]["total"]
        snapshot_total = load_metrics["snapshot_copy_bytes_per_request"]["total"]
        reassembly_total = load_metrics["reassembly_copy_bytes_per_request"]["total"]
        if selected == "direct_global_snapshot":
            assert contract["checksum_validations_per_request"] == 1
            assert contract["copy_byte_relation"] == (
                "snapshot_equals_payload_and_reassembly_is_zero"
            )
            assert snapshot_total == payload_total
            assert reassembly_total == 0
        else:
            assert contract["checksum_validations_per_request"] == 2
            assert contract["copy_byte_relation"] == (
                "snapshot_is_zero_and_reassembly_equals_twice_payload"
            )
            assert snapshot_total == 0
            assert reassembly_total == 2 * payload_total

    comparisons = record["comparisons"]
    assert len(comparisons) == 3
    comparison_pairs = {
        (row["direct_benchmark_id"], row["legacy_benchmark_id"]): row
        for row in comparisons
    }
    assert set(comparison_pairs) == {
        (
            "g6-vllm-8k-64-vanilla",
            "g6-vllm-8k-64-vanilla-legacy",
        ),
        (
            "g6-vllm-16k-256-vanilla",
            "g6-vllm-16k-256-vanilla-legacy",
        ),
        (
            "g6-vllm-8k-64-vanilla-direct-profiled",
            "g6-vllm-8k-64-vanilla-legacy-profiled",
        ),
    }
    assert comparison_pairs[("g6-vllm-8k-64-vanilla", "g6-vllm-8k-64-vanilla-legacy")][
        "ttft_direct_reduction_percent"
    ] == pytest.approx(60.22326128108402)
    assert comparison_pairs[
        ("g6-vllm-16k-256-vanilla", "g6-vllm-16k-256-vanilla-legacy")
    ]["ttft_direct_reduction_percent"] == pytest.approx(61.69979005319942)

    proofs = record["matched_setting_proofs"]
    assert len(proofs) == 3
    assert {
        (proof["direct_benchmark_id"], proof["legacy_benchmark_id"]) for proof in proofs
    } == set(comparison_pairs)
    expected_suite_ids = {
        "g6-vllm-8k-64-vanilla": "g6-vllm-8k-64",
        "g6-vllm-16k-256-vanilla": "g6-vllm-16k-256",
        "g6-vllm-8k-64-vanilla-direct-profiled": ("g6-vllm-8k-64-cold-load-profiled"),
    }
    expected_decoding_digests = {
        64: "f6150c4cea70a82be4ef0844e6347db74bbc9e1fd88165245e6c9d7f965b278c",
        256: "be26151b2b12d0b2521c3caaf89d957658dbda158106d1c2ac86ea917be915d9",
    }
    for proof in proofs:
        assert proof["all_other_comparison_inputs_identical"] is True
        assert proof["complete_effective_suite_settings_identical"] is True
        assert proof["complete_effective_manifest_settings_identical"] is True
        assert proof["private_quality_inputs_identical"] is True
        assert proof["topology_retained_in_matched_setting"] is True
        assert proof["artifact_identity_and_geometry_matched"] is True
        assert proof["artifact_identity_scope"] == (
            "generation_identity_and_contract_only"
        )
        assert proof["payload_content_equality_verified"] is False
        assert proof["payload_content_equality_limitation"] == (
            "The jobs separately regenerated equal-sized payloads. Their artifact "
            "IDs retain generation identity and contract, but no cross-job payload-"
            "content checksum was retained; byte-for-byte payload equality is not "
            "independently verified."
        )
        assert proof["varied_factor"] == "segmented_load_strategy"
        assert proof["ignored_derived_identity_field"] == (
            "physical_transform.config_digest"
        )
        assert SHA256_RE.fullmatch(proof["matched_setting_sha256"])
        assert all(
            SHA256_RE.fullmatch(digest)
            for digest in proof["synthesized_variant_config_digests"].values()
        )
        setting = proof["matched_setting"]
        recomputed_digest = sha256(
            json.dumps(setting, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        assert proof["matched_setting_sha256"] == recomputed_digest
        assert setting["method_id"] == "vanilla_prefill"
        assert setting["method_version"] == "2"
        assert setting["model_runtime"]["key_position_encoding"] == "pre_rope"
        assert setting["model_runtime"]["rope_theta"] == 5_000_000.0
        assert setting["model_runtime"]["rope_rotary_dim"] == 128
        assert len(setting["measurement_inputs"]) == 6
        direct_job = jobs[proof["direct_benchmark_id"]]
        output_tokens = direct_job["output_tokens"]
        assert setting["effective_suite"] == {
            "datasets": ["hotpotqa"],
            "examples": 2,
            "hardware_target": "aws-g6-l4",
            "interleave_examples": False,
            "isolate_arms": True,
            "model_id": "qwen3:4b-instruct",
            "prefix_cache_salt_mode": "per_request",
            "repeats": 3,
            "request_parallelism": 1,
            "seed": None,
            "shuffle": False,
            "suite_id": expected_suite_ids[proof["direct_benchmark_id"]],
        }
        assert setting["manifest_decoding"] == {
            "config_digest": expected_decoding_digests[output_tokens],
            "generation_seed": None,
            "max_output_tokens": output_tokens,
            "settings": {"ignore_eos": True},
            "stream": True,
            "temperature": 0.0,
        }
        assert setting["manifest_execution"] == {
            "benchmark_seed": None,
            "isolate_arms": True,
            "isolation_mode": "shared_process_sequential",
            "order_mode": "arm_isolated",
            "repeats": 3,
            "request_parallelism": 1,
            "shuffle": False,
            "source_execution_ids": [],
            "warmups": 0,
        }
        topology = setting["handoff_topology_attestation"]
        assert topology["example_count"] == 2
        assert all(
            example["document_count"] == 11 and example["segment_count"] == 11
            for example in topology["examples"]
        )

    _assert_no_secret_path_or_raw_text_leaks(record)
