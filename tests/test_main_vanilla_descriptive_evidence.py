import json
from hashlib import sha256
from pathlib import Path
import re
from typing import Any, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_README = REPO_ROOT / "benchmarks" / "README.md"
EVIDENCE_ROOT = (
    REPO_ROOT / "benchmarks" / "appendix" / "main-vanilla-descriptive-evidence"
)
EVIDENCE_PATH = EVIDENCE_ROOT / "evidence.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_EVIDENCE_TEXT = (
    "/Users/",
    "/Volumes/",
    "/Workspace/",
    "/dbfs/",
    ".databricks.com",
    ".azuredatabricks.net",
    "@opentable",
    "dbfs:",
    "file://",
    "http://",
    "https://",
)


def _load_canonical_evidence() -> dict[str, Any]:
    raw = EVIDENCE_PATH.read_bytes()
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    record = cast(dict[str, Any], json.loads(raw))
    assert raw == (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    return record


def _table_rows(readme: str, heading: str) -> list[list[str]]:
    section = readme.split(heading, maxsplit=1)[1]
    section = section.split("\n## ", maxsplit=1)[0]
    section = section.split("\n### ", maxsplit=1)[0]
    lines = [line for line in section.splitlines() if line.startswith("|")]
    assert len(lines) >= 3
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]]


def _latency_cells(row: dict[str, Any], context_tokens: int) -> list[str]:
    return [
        f"{row['ttft_seconds_p50']:.4f}",
        f"{row['ttft_seconds_p95']:.4f}",
        f"{row['ttc_seconds_p50']:.4f}",
        f"{row['ttc_seconds_p95']:.4f}",
        f"{row['request_decode_tokens_per_second_p50']:.4f}",
        f"{row['gpu_kv_capacity_tokens'] / context_tokens:.2f}x",
        f"{row['peak_gpu_memory_bytes'] / 2**30:.2f} GiB",
    ]


def _ablation_resource_cells(row: dict[str, Any], context_tokens: int) -> list[str]:
    return [
        *_latency_cells(row, context_tokens)[:5],
        f"{row['artifact_storage_bytes'] / 2**30:.2f} GiB",
        f"{row['gpu_kv_capacity_tokens'] / context_tokens:.2f}x",
        f"{row['peak_gpu_memory_bytes'] / 2**30:.2f} GiB",
        (
            f"{row['peak_process_tree_rss_bytes'] / 2**30:.2f} / "
            f"{row['peak_host_memory_used_bytes'] / 2**30:.2f} GiB"
        ),
    ]


def test_main_vanilla_evidence_is_canonical_sanitized_and_nonpublication() -> None:
    record = _load_canonical_evidence()
    serialized = json.dumps(record, sort_keys=True)

    assert record["record_type"] == "cachet.main_vanilla_descriptive_evidence.v1"
    assert record["schema_version"] == 1
    assert record["evidence_sanitized"] is True
    assert record["evidence_class"] == "descriptive_nonpublication"
    assert record["publication_claim_allowed"] is False
    assert record["qualification"] == {
        "baseline_raw_repo_gate_issues": [
            "benchmark does not contain a cache arm",
            "resource arm 'baseline_prefill' has no resource measurements",
        ],
        "baseline_raw_repo_gate_passed": False,
        "canonical_canary_gate_passed": False,
        "reason": (
            "Isolated Baseline records contain no cache arm, and hash-bound "
            "run-level resource telemetry is outside the per-arm resource schema."
        ),
    }
    assert all(forbidden not in serialized for forbidden in FORBIDDEN_EVIDENCE_TEXT)

    def check_hashes(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.endswith("_sha256"):
                    assert isinstance(child, str)
                    assert SHA256_RE.fullmatch(child)
                check_hashes(child)
        elif isinstance(value, list):
            for child in value:
                check_hashes(child)

    check_hashes(record)


def test_main_vanilla_evidence_binds_protocol_and_complete_main_pairs() -> None:
    record = _load_canonical_evidence()
    protocol = record["protocol"]

    assert protocol["source_revision"] == "38919a6b64681d647868696ccf7d6b736ec29e2b"
    assert protocol["wheel_sha256"] == (
        "74038cef655805add05688e307aa92596e2b6236d949d001e94964534af3e9af"
    )
    assert protocol["input_bundle_sha256"] == (
        "832c1e4fbb8371d93ee7eb08d3b727fffbcbbb14831e88c996a3bd9e30250896"
    )
    assert protocol["input_provenance_sha256"] == (
        "6ebf23b59ce994b69355504c71a3214863abf7c7d2bd356733364996c43f363d"
    )
    assert protocol["examples_per_dataset"] == 2
    assert protocol["example_count"] == 8
    assert protocol["repeats_per_example"] == 32
    assert protocol["measurements_per_job"] == 256
    assert protocol["request_parallelism"] == 4
    assert protocol["output_tokens"] == 256
    assert protocol["attention_backend"] == "TRITON_ATTN"
    assert protocol["example_ordering"] == "round_robin_interleaved"
    assert protocol["temperature"] == 0
    assert protocol["ignore_eos"] is True

    assert record["comparison_contract"] == {
        "ablation_common_fields": [
            "context_tokens",
            "gpu_memory_utilization",
            "method_arm",
        ],
        "common_protocol_reference": "$.protocol",
        "hardware_varied_fields": [
            "hardware_target",
            "node_type",
            "local_disk_topology",
        ],
        "latency_resource_common_fields": [
            "model_id",
            "model_revision",
            "model_quantization",
            "tokenizer_revision",
            "source_revision",
            "wheel_sha256",
            "input_bundle_sha256",
            "input_provenance_sha256",
            "engine",
            "engine_version",
            "package_versions",
            "request_parallelism",
            "repeats_per_example",
            "measurements_per_job",
            "output_tokens",
            "temperature",
            "ignore_eos",
            "attention_backend",
            "prefix_cache_salt_mode",
            "example_ordering",
        ],
        "main_latency_varied_field": "arm",
        "precision_varied_fields": [
            "document_kv_payload_dtype",
            "runtime_kv_dtype",
        ],
        "storage_varied_fields": [
            "tier",
            "cache_state",
        ],
    }

    comparisons = record["latency_method_comparisons"]
    assert [row["context_tokens"] for row in comparisons] == [8192, 16384, 32768]
    assert [row["gpu_memory_utilization"] for row in comparisons] == [0.9, 0.9, 0.7]
    for comparison in comparisons:
        assert comparison["evidence_class"] == "descriptive_nonpublication"
        assert comparison["normalized_setting_scope"] == (
            "cluster_scheduler_common_runner_protocol_and_provenance; "
            "method_identity_runtime_position_encoding_and_paths_excluded"
        )
        assert [row["arm"] for row in comparison["rows"]] == [
            "baseline_prefill",
            "document_kv_cache:vanilla_prefill",
        ]
        assert all(row["successful_requests"] == 256 for row in comparison["rows"])
        assert all(row["ttft_seconds_p50"] > 0 for row in comparison["rows"])
        assert all(row["ttc_seconds_p50"] > row["ttft_seconds_p50"] for row in comparison["rows"])
        assert all(
            row["capacity_derived_max_whole_requests"]
            == row["gpu_kv_capacity_tokens"] // comparison["context_tokens"]
            for row in comparison["rows"]
        )

    proofs = record["proofs"]
    assert proofs["core_validation_file_sha256"] == (
        "a08314e1faf9e6cda9831b8bfd59eb0f5e216b065f9fdf74d0668f163801824a"
    )
    assert proofs["core_validation_sha256"] == (
        "f7aeec3c6cf6bf39dadf0a7a1fa7366de55854601674f8d4f3184406295c676a"
    )
    assert [row["context_tokens"] for row in proofs["core_main_pair_proofs"]] == [
        8192,
        16384,
        32768,
    ]
    assert [row["pair_proof_sha256"] for row in proofs["core_main_pair_proofs"]] == [
        "644b7298a9d09feef59b86b79736b54eb1a08e96bd897e2935f66de7bda35e3f",
        "8f3a3d47a1bdc4a484018e975081aebb76522aa46137d2a6f5510d623e86ea76",
        "0cd25fd0cb009e30e061df807bf00584a99be44573a715bd645086aa368e3add",
    ]
    assert len(proofs["core_job_validation_files"]) == 7


def test_main_vanilla_evidence_records_score_and_ablation_boundaries() -> None:
    record = _load_canonical_evidence()

    score = record["score_diagnostic"]
    assert score["examples_per_dataset"] == 5
    assert score["publication_claim_allowed"] is False
    assert score["rows"] == [
        {
            "biography_answer_found": 1.0,
            "hotpotqa_f1": 0.1089749545847107,
            "method": "Baseline",
            "musique_answer_found": 0.2,
            "niah_exact_match": 0.0,
        },
        {
            "biography_answer_found": 1.0,
            "hotpotqa_f1": 0.04082687338501292,
            "method": "Vanilla KV",
            "musique_answer_found": 0.0,
            "niah_exact_match": 0.0,
        },
    ]

    assert record["precision_comparison"]["coupled_fields"] == [
        "document_kv_payload_dtype",
        "runtime_kv_dtype",
    ]
    assert record["proofs"]["precision_pair_proof_sha256"] == (
        "2c524f02418901101282ec76683190a19bba0d7dfb659adf5acd426c56e4022c"
    )
    precision_rows = record["precision_comparison"]["rows"]
    assert [row["document_kv_payload_dtype"] for row in precision_rows] == [
        "fp8_e5m2",
        "bfloat16",
    ]
    assert [row["runtime_kv_dtype"] for row in precision_rows] == [
        "fp8_e5m2",
        "bfloat16",
    ]
    assert all(row["successful_requests"] == 256 for row in precision_rows)
    for comparison_name in (
        "precision_comparison",
        "storage_comparison",
        "hardware_comparison",
    ):
        comparison = record[comparison_name]
        assert comparison["context_tokens"] == 16384
        assert comparison["method_arm"] == "document_kv_cache:vanilla_prefill"
    storage = {row["tier"]: row for row in record["storage_comparison"]["rows"]}
    assert storage["ram"]["payload_cache_hit_count"] == 256
    assert storage["ram"]["storage_bytes_read_total"] == 0
    assert storage["unity_catalog"]["strict_cold_claim_allowed"] is False
    assert record["hardware_comparison"]["gpu_only_ablation"] is False
    assert record["unsupported_or_unrun"] == {
        "cacheblend": "method is not implemented",
        "full_dataset_scores": (
            "Biography, HotpotQA, MusiQue, and NIAH full-dataset runs were not "
            "conducted"
        ),
        "hybrid_storage": "not run under the current protocol",
        "infoflow_kv": "method is not implemented",
        "kv_packet": "method is not implemented",
        "longbench_v2": "dataset runner is not implemented",
        "packed_q4_kv": "payload layout and serving support are not implemented",
        "ruler": "dataset runner is not implemented",
        "sglang_q8_pre_rope": "serving path is not implemented",
    }


def test_benchmark_tables_have_no_blank_current_vanilla_cells() -> None:
    readme = BENCHMARK_README.read_text(encoding="utf-8")

    main_rows = _table_rows(readme, "## Main Latency And Resource Table")
    current_main = [row for row in main_rows if row[0] in {"Baseline", "Vanilla&nbsp;KV"}]
    assert len(current_main) == 6
    assert all(all(cell for cell in row) for row in current_main)
    assert all("N/A" not in cell for row in current_main for cell in row[2:])
    assert all(
        cell.startswith("N/A")
        for row in main_rows
        if row[0] not in {"Baseline", "Vanilla&nbsp;KV"}
        for cell in row[2:]
    )

    score_rows = _table_rows(readme, "## Benchmark Dataset Score Table")
    assert all(all(cell for cell in row) for row in score_rows)
    assert all("N/A" in cell for row in score_rows for cell in row[1:])

    for heading in (
        "## Document KV Precision Ablation",
        "## Storage Tier Ablation",
        "## Hardware Ablation",
        "## Serving Platform Ablation",
    ):
        assert all(
            all(cell for cell in row)
            for row in _table_rows(readme, heading)
        )


def test_benchmark_table_numbers_are_derived_from_sanitized_evidence() -> None:
    record = _load_canonical_evidence()
    readme = BENCHMARK_README.read_text(encoding="utf-8")

    main_rows = {
        (row[0], row[1]): row
        for row in _table_rows(readme, "## Main Latency And Resource Table")
    }
    for comparison in record["latency_method_comparisons"]:
        context_tokens = comparison["context_tokens"]
        context_label = f"{context_tokens // 1024}k"
        for evidence_row in comparison["rows"]:
            method = (
                "Baseline"
                if evidence_row["arm"] == "baseline_prefill"
                else "Vanilla&nbsp;KV"
            )
            assert main_rows[(method, context_label)][2:] == _latency_cells(
                evidence_row,
                context_tokens,
            )

    score_rows = {
        row[0]: row
        for row in _table_rows(readme, "### Five-example score diagnostic")
    }
    for evidence_row in record["score_diagnostic"]["rows"]:
        method = (
            "Baseline"
            if evidence_row["method"] == "Baseline"
            else "Vanilla&nbsp;KV"
        )
        assert score_rows[method][1:] == [
            f"{evidence_row['biography_answer_found']:.6f}",
            f"{evidence_row['hotpotqa_f1']:.6f}",
            f"{evidence_row['musique_answer_found']:.6f}",
            f"{evidence_row['niah_exact_match']:.6f}",
        ]

    precision_rows = _table_rows(readme, "## Document KV Precision Ablation")
    precision_by_dtype = {
        row["document_kv_payload_dtype"]: row
        for row in record["precision_comparison"]["rows"]
    }
    for table_label, dtype in (("bf16", "bfloat16"), ("Q8", "fp8_e5m2")):
        table_row = next(row for row in precision_rows if row[0].startswith(table_label))
        evidence_row = precision_by_dtype[dtype]
        expected = _ablation_resource_cells(evidence_row, 16384)
        assert table_row[1:6] == expected[:5]
        assert table_row[6].startswith("N/A")
        assert table_row[7:11] == expected[5:]
    q4_row = next(row for row in precision_rows if row[0].startswith("Q4"))
    assert all(cell.startswith("N/A") for cell in q4_row[1:11])

    storage_rows = _table_rows(readme, "## Storage Tier Ablation")
    storage_by_tier = {
        row["tier"]: row for row in record["storage_comparison"]["rows"]
    }
    for table_label, tier in (
        ("RAM", "ram"),
        ("Disk", "disk"),
        ("Unity Catalog", "unity_catalog"),
    ):
        table_row = next(row for row in storage_rows if row[0] == table_label)
        assert table_row[1:10] == _ablation_resource_cells(
            storage_by_tier[tier],
            16384,
        )
    hybrid_row = next(row for row in storage_rows if row[0].startswith("Hybrid"))
    assert all(cell.startswith("N/A") for cell in hybrid_row[1:10])

    hardware_rows = _table_rows(readme, "## Hardware Ablation")
    hardware_by_target = {
        row["hardware_target"]: row
        for row in record["hardware_comparison"]["rows"]
    }
    for table_label, target in (
        ("AWS g6/L4", "aws-g6-l4"),
        ("AWS g5/A10G", "aws-g5-a10g"),
    ):
        table_row = next(row for row in hardware_rows if row[0].startswith(table_label))
        assert table_row[1:10] == _ablation_resource_cells(
            hardware_by_target[target],
            16384,
        )

    platform_rows = _table_rows(readme, "## Serving Platform Ablation")
    vllm_row = next(row for row in platform_rows if row[0] == "vLLM")
    assert vllm_row[1:10] == _ablation_resource_cells(
        hardware_by_target["aws-g6-l4"],
        16384,
    )
    sglang_row = next(row for row in platform_rows if row[0] == "SGLang")
    assert all(cell.startswith("N/A") for cell in sglang_row[1:10])


def test_main_vanilla_evidence_readme_binds_record_digest() -> None:
    readme = (EVIDENCE_ROOT / "README.md").read_text(encoding="utf-8")
    digest = sha256(EVIDENCE_PATH.read_bytes()).hexdigest()

    assert digest in readme
    assert "does not pass Cachet's canonical canary" in readme
    assert "descriptive" in readme
    assert "nonpublication" in readme
