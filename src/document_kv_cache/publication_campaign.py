"""Closed publication-campaign design for the vLLM 0.27.1 benchmark reset."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from document_kv_cache.benchmarks import SUPPORTED_V1_DATASETS
from document_kv_cache.databricks_resource_ledger import (
    DatabricksLedgerPrefix,
    MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS,
    MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
    databricks_ledger_path_sha256,
    databricks_ledger_prefix,
    databricks_ledger_prefix_from_record,
    read_databricks_cluster_hour_ledger_json,
)


PUBLICATION_CAMPAIGN_RECORD_TYPE = "cachet.vllm_0271_publication_campaign.v1"
PUBLICATION_CAMPAIGN_SCHEMA_VERSION = 1
PUBLICATION_CAMPAIGN_ID = "vllm-0271-publication-v1"
PUBLICATION_CAMPAIGN_PRE_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_CAMPAIGN_FILE_SHA256 = "c805c303a92dba3fdd0390699c757974c1f738ebc4c553bb651618cb27bf8056"
PUBLICATION_CAMPAIGN_PRE_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_CAMPAIGN_CLOSED_RECORD_SHA256 = "1f1682a99e69ad691dfab68a85cc9555eff4daea437d5095d93410af2430c490"
PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256 = (
    "5f90b531b30ac6f4b29e0151d688a005b0377b205ca39645376d7d43aef5e305"
)
PUBLICATION_CAMPAIGN_ENGINE_VERSION = "0.27.1"
PUBLICATION_CAMPAIGN_METHODS = ("baseline_prefill", "vanilla_prefill")
PUBLICATION_CAMPAIGN_CONTEXT_TOKENS = (8192, 16384, 32768)
PUBLICATION_CAMPAIGN_REQUEST_PARALLELISM = (1, 2, 4)
PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS = 5
PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET = 32
PUBLICATION_CAMPAIGN_REPEATS_PER_EXAMPLE = 2
PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL = (
    len(SUPPORTED_V1_DATASETS)
    * PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
    * PUBLICATION_CAMPAIGN_REPEATS_PER_EXAMPLE
)
PUBLICATION_CAMPAIGN_STORAGE_EXAMPLES_PER_DATASET = 2
PUBLICATION_CAMPAIGN_STORAGE_SELECTION_DOMAIN = (
    "cachet.publication.storage_subset.selection.v1"
)
PUBLICATION_CAMPAIGN_STORAGE_REPEATS_PER_EXAMPLE = 32
PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL = (
    len(SUPPORTED_V1_DATASETS)
    * PUBLICATION_CAMPAIGN_STORAGE_EXAMPLES_PER_DATASET
    * PUBLICATION_CAMPAIGN_STORAGE_REPEATS_PER_EXAMPLE
)
PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS = 16
PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS = 124.0
PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND = 35.0
PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS = 16
PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_TASK_TIMEOUT_SECONDS = 5 * 60 * 60
PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_MAX_RESERVED_GPU_HOURS = (
    PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS
    * PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_TASK_TIMEOUT_SECONDS
    / 3600.0
)
PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_INPUT_TOKEN_SLOTS = (
    len(SUPPORTED_V1_DATASETS)
    * PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
    * sum(PUBLICATION_CAMPAIGN_CONTEXT_TOKENS)
)
PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_CACHE_PREFIX_TOKENS = 7_323_967
PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_MAX_GPU_HOURS_AT_GATE = (
    PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_CACHE_PREFIX_TOKENS
    / PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
    / 3600.0
)
PUBLICATION_CAMPAIGN_BF16_HANDOFF_CACHE_PREFIX_TOKENS = 2_091_797
PUBLICATION_CAMPAIGN_BF16_HANDOFF_PRODUCER_TASKS = 16
PUBLICATION_CAMPAIGN_BF16_HANDOFF_TASK_TIMEOUT_SECONDS = 5 * 60 * 60
PUBLICATION_CAMPAIGN_BF16_HANDOFF_MAX_RESERVED_GPU_HOURS = (
    PUBLICATION_CAMPAIGN_BF16_HANDOFF_PRODUCER_TASKS
    * PUBLICATION_CAMPAIGN_BF16_HANDOFF_TASK_TIMEOUT_SECONDS
    / 3600.0
)
PUBLICATION_CAMPAIGN_BF16_HANDOFF_MAX_GPU_HOURS_AT_GATE = (
    PUBLICATION_CAMPAIGN_BF16_HANDOFF_CACHE_PREFIX_TOKENS
    / PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
    / 3600.0
)
PUBLICATION_CAMPAIGN_BF16_HANDOFF_PAYLOAD_BYTES = 308_448_018_432
PUBLICATION_CAMPAIGN_BF16_HANDOFF_PAYLOAD_GIB = (
    PUBLICATION_CAMPAIGN_BF16_HANDOFF_PAYLOAD_BYTES / 1024**3
)
PUBLICATION_CAMPAIGN_BF16_HANDOFF_SLOT_ENVELOPE_BYTES = 288 * 1024**3
PUBLICATION_CAMPAIGN_FULL_SCORE_EXAMPLES = 83_653
PUBLICATION_CAMPAIGN_FULL_SCORE_SHARDS = 160
PUBLICATION_CAMPAIGN_FULL_SCORE_WAVES = 10
PUBLICATION_CAMPAIGN_FULL_SCORE_CACHE_PREFIX_TOKENS = 63_455_746
PUBLICATION_CAMPAIGN_FULL_SCORE_NATURAL_PROMPT_TOKENS = 66_448_937
PUBLICATION_CAMPAIGN_FULL_SCORE_INVENTORY_SHA256 = (
    "e19fefa656d8975946b13bb9987f801ec486c4bfde5e9d5ed82a877e80676b11"
)
PUBLICATION_CAMPAIGN_FULL_SCORE_SHARD_PLAN_SHA256 = (
    "605c15ef5317bb0b6d6f6a4057dbacbd97ae31af94a3d497585a88c138c9ba84"
)
PUBLICATION_CAMPAIGN_FULL_SCORE_EXECUTION_PLAN_SHA256 = (
    "f4e80b89bcb5153c20e7c9275dbc9d30282514cec76bdc72279262d5fca63b60"
)
PUBLICATION_CAMPAIGN_FULL_SCORE_TASK_TIMEOUT_SECONDS = 6 * 60 * 60
PUBLICATION_CAMPAIGN_FULL_SCORE_TASKS_PER_PHASE = 16
PUBLICATION_CAMPAIGN_FULL_SCORE_PHASES = PUBLICATION_CAMPAIGN_FULL_SCORE_WAVES * 2
PUBLICATION_CAMPAIGN_FULL_SCORE_MAX_RESERVED_GPU_HOURS_PER_PHASE = (
    PUBLICATION_CAMPAIGN_FULL_SCORE_TASKS_PER_PHASE
    * PUBLICATION_CAMPAIGN_FULL_SCORE_TASK_TIMEOUT_SECONDS
    / 3600.0
)
PUBLICATION_CAMPAIGN_FULL_SCORE_GENERATION_MAX_GPU_HOURS_AT_GATE = (
    PUBLICATION_CAMPAIGN_FULL_SCORE_CACHE_PREFIX_TOKENS
    / PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
    / 3600.0
)
PUBLICATION_CAMPAIGN_TOTAL_GENERATION_CACHE_PREFIX_TOKENS = (
    PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_CACHE_PREFIX_TOKENS
    + PUBLICATION_CAMPAIGN_BF16_HANDOFF_CACHE_PREFIX_TOKENS
    + PUBLICATION_CAMPAIGN_FULL_SCORE_CACHE_PREFIX_TOKENS
)
PUBLICATION_CAMPAIGN_TOTAL_GENERATION_MAX_GPU_HOURS_AT_GATE = (
    PUBLICATION_CAMPAIGN_TOTAL_GENERATION_CACHE_PREFIX_TOKENS
    / PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
    / 3600.0
)
PUBLICATION_CAMPAIGN_GPU_QUALIFICATION_JOBS = 14
PUBLICATION_CAMPAIGN_GPU_QUALIFICATION_TASK_TIMEOUT_SECONDS = 4 * 60 * 60
PUBLICATION_CAMPAIGN_GPU_QUALIFICATION_MAX_RESERVED_GPU_HOURS = (
    PUBLICATION_CAMPAIGN_GPU_QUALIFICATION_JOBS
    * PUBLICATION_CAMPAIGN_GPU_QUALIFICATION_TASK_TIMEOUT_SECONDS
    / 3600.0
)
PUBLICATION_CAMPAIGN_CPU_COORDINATOR_NODE_TYPE_ID = "c5d.4xlarge"
PUBLICATION_CAMPAIGN_CPU_COORDINATOR_SPARK_VERSION = "15.4.x-cpu-ml-scala2.12"
PUBLICATION_CAMPAIGN_HANDOFF_CLOSURE_CPU_JOBS = 2
PUBLICATION_CAMPAIGN_HANDOFF_CLOSURE_CPU_TIMEOUT_SECONDS = 12 * 60 * 60
PUBLICATION_CAMPAIGN_LATENCY_SOURCE_CLOSURE_CPU_JOBS = 1
PUBLICATION_CAMPAIGN_LATENCY_SOURCE_CLOSURE_CPU_TIMEOUT_SECONDS = 2 * 60 * 60
PUBLICATION_CAMPAIGN_FULL_SCORE_REMOTE_CPU_JOBS = PUBLICATION_CAMPAIGN_FULL_SCORE_PHASES
PUBLICATION_CAMPAIGN_FULL_SCORE_REMOTE_CPU_TIMEOUT_SECONDS = 2 * 60 * 60
PUBLICATION_CAMPAIGN_TOTAL_CPU_COORDINATOR_JOBS = (
    PUBLICATION_CAMPAIGN_HANDOFF_CLOSURE_CPU_JOBS
    + PUBLICATION_CAMPAIGN_LATENCY_SOURCE_CLOSURE_CPU_JOBS
    + PUBLICATION_CAMPAIGN_FULL_SCORE_REMOTE_CPU_JOBS
)
PUBLICATION_CAMPAIGN_CPU_COORDINATOR_TIMEOUT_NODE_HOURS = (
    PUBLICATION_CAMPAIGN_HANDOFF_CLOSURE_CPU_JOBS
    * PUBLICATION_CAMPAIGN_HANDOFF_CLOSURE_CPU_TIMEOUT_SECONDS
    + PUBLICATION_CAMPAIGN_LATENCY_SOURCE_CLOSURE_CPU_JOBS
    * PUBLICATION_CAMPAIGN_LATENCY_SOURCE_CLOSURE_CPU_TIMEOUT_SECONDS
    + PUBLICATION_CAMPAIGN_FULL_SCORE_REMOTE_CPU_JOBS
    * PUBLICATION_CAMPAIGN_FULL_SCORE_REMOTE_CPU_TIMEOUT_SECONDS
) / 3600.0
PUBLICATION_CAMPAIGN_MAX_LATENCY_WAVE_RESERVED_GPU_HOURS = (
    PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS * 12.0
)
PUBLICATION_CAMPAIGN_LATENCY_TIMEOUT_JOB_COUNTS = (
    (4, 65),
    (6, 20),
    (8, 20),
    (12, 10),
)
PUBLICATION_CAMPAIGN_LATENCY_TIMEOUT_UPPER_BOUND_GPU_HOURS = sum(
    timeout_hours * job_count
    for timeout_hours, job_count in PUBLICATION_CAMPAIGN_LATENCY_TIMEOUT_JOB_COUNTS
)
PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS = 67.93033611111115
PUBLICATION_CAMPAIGN_LEDGER_ID = "representative-canary-823bd9d82a5c1730"
PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256 = (
    "fd00fcc39375aa8c96dabba9e3e4c576ae2674dd911324622ef99293b9cfe865"
)
PUBLICATION_CAMPAIGN_PRE_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_LEDGER_FILE_SHA256 = "fd0b6774928f77166657c8d35652e4d557f6708552d88c7c6725fc42d7723e87"
PUBLICATION_CAMPAIGN_OPENING_LEDGER_FILE_SHA256 = (
    "38677fff866e0a7268398c4b616b4be968df3a8191381db74ebd8fcb71af50ef"
)
PUBLICATION_CAMPAIGN_PRE_SITE_PACKAGES_PATH_FAILURE_LEDGER_FILE_SHA256 = (
    "1ac7ee076d2a5aa3b12bfd18d3cb6f8843aa9f8f7b8e07686c519869985a6916"
)
PUBLICATION_CAMPAIGN_PRE_RUNTIME_LOCK_INDEX_FAILURE_LEDGER_FILE_SHA256 = (
    "f76cce3b68417f8d14a5e030d9eacaef3e61d17f123a2a2b5d38be5428a89b94"
)
PUBLICATION_CAMPAIGN_PRE_REJECTED_QUALIFICATION_LEDGER_PREFIX = DatabricksLedgerPrefix(
    ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
    cap_cluster_hours=MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
    reservation_count=124,
    submission_receipt_count=0,
    terminal_actual_count=124,
    prefix_sha256=("1c4bfb602657393b3fb2a20570d8658e8b5ed4b00e9d3ec3461be83454c366ad"),
)
PUBLICATION_CAMPAIGN_PRE_FAILED_QUALIFICATION_LEDGER_PREFIX = DatabricksLedgerPrefix(
    ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
    cap_cluster_hours=MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
    reservation_count=138,
    submission_receipt_count=0,
    terminal_actual_count=138,
    prefix_sha256=("a12b5e754da84e4c7b3e0f273c14d2b79ce9cb1483b02dcc77ca522185e89dea"),
)
PUBLICATION_CAMPAIGN_PRE_BOOTSTRAP_FAILURE_LEDGER_PREFIX = DatabricksLedgerPrefix(
    ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
    cap_cluster_hours=MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
    reservation_count=152,
    submission_receipt_count=14,
    terminal_actual_count=152,
    prefix_sha256=("4bbe1144d4ce037fd8cf3376fc20c4e19ad00641f84c0a54d0cc2c17e37bf728"),
)
PUBLICATION_CAMPAIGN_PRE_CLUSTER_IDENTITY_FAILURE_LEDGER_PREFIX = (
    DatabricksLedgerPrefix(
        ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
        cap_cluster_hours=MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
        reservation_count=166,
        submission_receipt_count=28,
        terminal_actual_count=166,
        prefix_sha256=(
            "273aeb12c61060ca8d7850f5583f8912fa2a44ede44ddcba030da63926bff368"
        ),
    )
)
PUBLICATION_CAMPAIGN_PRE_RUNTIME_LOCK_INDEX_FAILURE_LEDGER_PREFIX = (
    DatabricksLedgerPrefix(
        ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
        cap_cluster_hours=MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
        reservation_count=180,
        submission_receipt_count=42,
        terminal_actual_count=180,
        prefix_sha256=(
            "376114c27f35725bab5418969d28a77d4a3600dba44d049b597512142856d86f"
        ),
    )
)
PUBLICATION_CAMPAIGN_PRE_SITE_PACKAGES_PATH_FAILURE_LEDGER_PREFIX = (
    DatabricksLedgerPrefix(
        ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
        cap_cluster_hours=MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
        reservation_count=194,
        submission_receipt_count=56,
        terminal_actual_count=194,
        prefix_sha256=(
            "381ed88dfca75a17cf11b09b7e3dedb435328e518e8f1f0f0d9591be27796f26"
        ),
    )
)
PUBLICATION_CAMPAIGN_PRE_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_LEDGER_PREFIX = DatabricksLedgerPrefix(
    ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
    cap_cluster_hours=MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
    reservation_count=208,
    submission_receipt_count=70,
    terminal_actual_count=208,
    prefix_sha256=("a71cee32c1ae056d7db7c72c70fa72bcf5622d8a3ae6d72590c4435bb9db4af9"),
)
PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX = DatabricksLedgerPrefix(
    ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
    cap_cluster_hours=MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
    reservation_count=222,
    submission_receipt_count=84,
    terminal_actual_count=222,
    prefix_sha256=("22ac65492fa0871f528552cfcae0bd6332b1429cd9fc2e92c373c5e534202d4a"),
)
PUBLICATION_CAMPAIGN_OPENING_ACTIVE_RESERVED_GPU_HOURS = 0.0
PUBLICATION_CAMPAIGN_OPENING_ACCOUNTED_GPU_HOURS = 67.93033611111115
PUBLICATION_CAMPAIGN_OPENING_REMAINING_GPU_HOURS = 956.0696638888888
PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ACTUAL_CLUSTER_DURATION_SECONDS = 12_410.279
PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ACTUAL_GPU_HOURS = 3.447299722222222
PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_PLAN_FILE_SHA256 = "fe59e32c44ab50f91bae5114a587268d44ebb9acfba74500aedb66158e2541b7"
PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_PLAN_SHA256 = (
    "c0bede45ea211798c9a5eb31010a91074ded70e370f8ea4fcbeb59b3b9f95598"
)
PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_RUNNER_SHA256 = (
    "ca93baeda09f3df050b0dad3b8f3091c0f74235c426bd66555b67bd4b6eeafbc"
)
PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_MANIFEST_FILE_SHA256 = "53fd4b076a642101790d21ebbc03b1eb7e609428c2ccd7eafb8cbad5a9a3a112"
PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_MANIFEST_SHA256 = "6c4cca0ec4fbcf4ccb434573f965eeb8022909ce5bdd6afdf31d61085807fa9b"
PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_EVIDENCE_TREE_SHA256 = "bb6636f3b9bdf5afae0b7d1beb97f5f3192017ba5b04abb651f2a389889aa57f"
PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_EVIDENCE_TREE_FILE_COUNT = 29
PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_EVIDENCE_TREE_TOTAL_BYTES = 1_828_218
PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_NORMALIZED_OBSERVER_ERROR_SHA256 = "3662915979987aef1fe4bcf9e0e62f06c67992ee73da679e44f6b6a261e634f5"
PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_NORMALIZED_WORKER_ERROR_SHA256 = "3f1ddd73298cd46347cf57b84d6cf22f7d6e98802b50ded9457d7a999563786b"
PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_REASON = (
    "all fourteen hash-locked qualification runtimes installed and verified; "
    "the two packed-page-roundtrip workers returned measurements before "
    "post-success runtime observation rejected the virtualenv-created "
    "runtime/bin/python symlink, while the other twelve sentinel-worker "
    "subprocesses exited nonzero and the reviewed launcher did not surface their "
    "captured child stdout/stderr, so their underlying worker causes remain unknown"
)
PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OBSERVER_JOB_IDS = (
    "aws-g5-a10g-packed-page-roundtrip",
    "aws-g6-l4-packed-page-roundtrip",
)
PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OPAQUE_WORKER_JOB_IDS = (
    "aws-g5-a10g-16k-c4-capacity",
    "aws-g5-a10g-auto-backend-diagnostic",
    "aws-g5-a10g-forced-triton-runtime-handoff",
    "aws-g5-a10g-matched-token-logit",
    "aws-g6-l4-32k-c4-gmu-70",
    "aws-g6-l4-32k-c4-gmu-75",
    "aws-g6-l4-32k-c4-gmu-80",
    "aws-g6-l4-auto-backend-diagnostic",
    "aws-g6-l4-forced-triton-runtime-handoff",
    "aws-g6-l4-generation-throughput",
    "aws-g6-l4-matched-token-logit",
    "aws-g6e-l40s-generation-throughput",
)
PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_SHA256_BY_JOB = (
    (
        "aws-g5-a10g-16k-c4-capacity",
        "3a4094c819734ed0dddfe7b32bd9602b801ec6e95127a9e9bac82d6947032892",
    ),
    (
        "aws-g5-a10g-auto-backend-diagnostic",
        "7a67afd26245bb8ceed7e518a72c6f4a8432bd5ac27dd2e9467a7f325db33609",
    ),
    (
        "aws-g5-a10g-forced-triton-runtime-handoff",
        "b2292081d99661861af1eade188b179e2a2e459764845f9a8227df6aea559708",
    ),
    (
        "aws-g5-a10g-matched-token-logit",
        "e6655eafb38fd5ffc514848b5d0b58a0f596c6ae774b24bd701c7ebd30e4a542",
    ),
    (
        "aws-g5-a10g-packed-page-roundtrip",
        "4f57e38b2c170b37907bcacc1d64d18af14751a05a11fe7a8ca5ca83dab84a4d",
    ),
    (
        "aws-g6-l4-32k-c4-gmu-70",
        "42d5c7f85201050c07e1ef702151b1016eccf91b6e9b00fab4c992dcd83d9e57",
    ),
    (
        "aws-g6-l4-32k-c4-gmu-75",
        "178e7fc6e44a38819f4bdb12a74019e66826244f933e1bf5261774ef6484d8ce",
    ),
    (
        "aws-g6-l4-32k-c4-gmu-80",
        "023f005c378816eba82665743ea578810c81e654f68c098eabb4c3e8a9dd1d56",
    ),
    (
        "aws-g6-l4-auto-backend-diagnostic",
        "54715e131389ec2a14aca8489ef578d1126d629ed2dd275e0b6d320c7aed2086",
    ),
    (
        "aws-g6-l4-forced-triton-runtime-handoff",
        "5ee2a73a8860e1ad1ce72a3b8dce8c6e00271a6db1aae7f485cbdf955d339be8",
    ),
    (
        "aws-g6-l4-generation-throughput",
        "16428f8492bfef5cdf5e75cb5cd97c530915815394d678a0c25ad9eee167b583",
    ),
    (
        "aws-g6-l4-matched-token-logit",
        "a0df4005431bb97344c6a6affcd237a6e0d4683a1754d4cec5b7bb6e48895b86",
    ),
    (
        "aws-g6-l4-packed-page-roundtrip",
        "6c2a32edd3da00f486c8256abe68b09b66f9d759cc0c42c21d33ead2d780c1d7",
    ),
    (
        "aws-g6e-l40s-generation-throughput",
        "be19d2e9041301609073014961eb3eb7cf162168384f5d72345a4a3787e1dd75",
    ),
)
PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_UTF8_BYTES_BY_JOB = (
    ("aws-g5-a10g-16k-c4-capacity", 1_186),
    ("aws-g5-a10g-auto-backend-diagnostic", 1_234),
    ("aws-g5-a10g-forced-triton-runtime-handoff", 1_270),
    ("aws-g5-a10g-matched-token-logit", 1_210),
    ("aws-g5-a10g-packed-page-roundtrip", 244),
    ("aws-g6-l4-32k-c4-gmu-70", 1_162),
    ("aws-g6-l4-32k-c4-gmu-75", 1_162),
    ("aws-g6-l4-32k-c4-gmu-80", 1_162),
    ("aws-g6-l4-auto-backend-diagnostic", 1_222),
    ("aws-g6-l4-forced-triton-runtime-handoff", 1_258),
    ("aws-g6-l4-generation-throughput", 1_210),
    ("aws-g6-l4-matched-token-logit", 1_198),
    ("aws-g6-l4-packed-page-roundtrip", 242),
    ("aws-g6e-l40s-generation-throughput", 1_228),
)
PUBLICATION_CAMPAIGN_NON_GENERATION_GPU_HOURS_AVAILABLE_AT_GATE = (
    MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
    - PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS
    - PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
    - PUBLICATION_CAMPAIGN_TOTAL_GENERATION_MAX_GPU_HOURS_AT_GATE
)
PUBLICATION_CAMPAIGN_BOOTSTRAP_DRAWS = 20_000
PUBLICATION_CAMPAIGN_CORE_LATENCY_TIMEOUT_HOURS = (
    (8_192, ((1, 6), (2, 4), (4, 4))),
    (16_384, ((1, 8), (2, 6), (4, 4))),
    (32_768, ((1, 12), (2, 8), (4, 4))),
)
PUBLICATION_CAMPAIGN_AUXILIARY_LATENCY_TIMEOUT_HOURS = 4
PUBLICATION_CAMPAIGN_AUXILIARY_SETTINGS = (
    (
        "precision-bf16",
        "precision",
        "bf16_payload_and_runtime_kv",
    ),
    (
        "storage-ram",
        "storage",
        "ram_hot_host_cold_gpu",
    ),
    (
        "storage-uc",
        "storage",
        "uc_mounted_path_eviction_backend_cache_unproven",
    ),
    (
        "hardware-a10g",
        "hardware",
        "a10g_local_nvme",
    ),
)
PUBLICATION_CAMPAIGN_STORAGE_CONTROL_SETTING = (
    "storage-disk",
    "storage",
    "local_nvme_strict_cold_control",
)
PUBLICATION_CAMPAIGN_AUXILIARY_JOBS = PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS * (
    len(PUBLICATION_CAMPAIGN_AUXILIARY_SETTINGS) + 1
)
PUBLICATION_CAMPAIGN_TOTAL_LATENCY_JOBS = (
    len(PUBLICATION_CAMPAIGN_METHODS)
    * len(PUBLICATION_CAMPAIGN_CONTEXT_TOKENS)
    * len(PUBLICATION_CAMPAIGN_REQUEST_PARALLELISM)
    * PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS
    + PUBLICATION_CAMPAIGN_AUXILIARY_JOBS
)

_PLAN_KEYS = frozenset(
    {
        "analysis",
        "auxiliary_latency_cells",
        "budget",
        "campaign_id",
        "campaign_ledger_id",
        "campaign_ledger_path_sha256",
        "campaign_ledger_prefix",
        "campaign_opening_terminal_gpu_hours",
        "closed_record_sha256",
        "engine_version",
        "full_score_program",
        "latency_cells",
        "latency_timeout_policy",
        "record_type",
        "request_protocol",
        "schema_version",
        "storage_request_protocol",
    }
)
_AUXILIARY_CELL_KEYS = frozenset(
    {
        "cell_id",
        "comparison_family",
        "deployment_block",
        "examples_per_dataset",
        "input_tokens",
        "method_id",
        "reference_core_cell_id",
        "repeats_per_example",
        "request_count",
        "request_parallelism",
        "setting_id",
    }
)
_CELL_KEYS = frozenset(
    {
        "cell_id",
        "deployment_block",
        "examples_per_dataset",
        "input_tokens",
        "matched_pair_id",
        "method_id",
        "repeats_per_example",
        "request_count",
        "request_parallelism",
    }
)


@dataclass(frozen=True, slots=True)
class PublicationLatencyCell:
    """One isolated serving job in the publication latency factorial."""

    cell_id: str
    matched_pair_id: str
    deployment_block: int
    method_id: str
    input_tokens: int
    request_parallelism: int
    examples_per_dataset: int = PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
    repeats_per_example: int = PUBLICATION_CAMPAIGN_REPEATS_PER_EXAMPLE
    request_count: int = PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL

    def __post_init__(self) -> None:
        if not self.cell_id or not self.matched_pair_id:
            raise ValueError("cell_id and matched_pair_id must be non-empty")
        if self.deployment_block not in range(
            1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1
        ):
            raise ValueError("deployment_block is outside the frozen campaign")
        if self.method_id not in PUBLICATION_CAMPAIGN_METHODS:
            raise ValueError("method_id is outside the frozen campaign")
        if self.input_tokens not in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS:
            raise ValueError("input_tokens is outside the frozen campaign")
        if self.request_parallelism not in PUBLICATION_CAMPAIGN_REQUEST_PARALLELISM:
            raise ValueError("request_parallelism is outside the frozen campaign")
        if self.examples_per_dataset != PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET:
            raise ValueError(
                "examples_per_dataset must equal the frozen campaign value"
            )
        if self.repeats_per_example != PUBLICATION_CAMPAIGN_REPEATS_PER_EXAMPLE:
            raise ValueError("repeats_per_example must equal the frozen campaign value")
        if self.request_count != PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL:
            raise ValueError("request_count must equal the frozen campaign value")


@dataclass(frozen=True, slots=True)
class PublicationAuxiliaryLatencyCell:
    """One incremental job in the precision/storage/hardware program."""

    cell_id: str
    reference_core_cell_id: str
    deployment_block: int
    setting_id: str
    comparison_family: str
    method_id: str = "vanilla_prefill"
    input_tokens: int = 16_384
    request_parallelism: int = 4
    examples_per_dataset: int = PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
    repeats_per_example: int = PUBLICATION_CAMPAIGN_REPEATS_PER_EXAMPLE
    request_count: int = PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL

    def __post_init__(self) -> None:
        if not self.cell_id or not self.reference_core_cell_id:
            raise ValueError("cell_id and reference_core_cell_id must be non-empty")
        if self.deployment_block not in range(
            1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1
        ):
            raise ValueError("deployment_block is outside the frozen campaign")
        expected_settings = {
            (setting_id, family)
            for setting_id, family, _description in (
                PUBLICATION_CAMPAIGN_AUXILIARY_SETTINGS
            )
        }
        expected_settings.add(PUBLICATION_CAMPAIGN_STORAGE_CONTROL_SETTING[:2])
        if (self.setting_id, self.comparison_family) not in expected_settings:
            raise ValueError("auxiliary setting is outside the frozen campaign")
        if self.method_id != "vanilla_prefill":
            raise ValueError("auxiliary latency jobs must use Vanilla")
        if self.input_tokens != 16_384 or self.request_parallelism != 4:
            raise ValueError("auxiliary latency jobs must reuse the 16k c4 anchor")
        storage_setting = self.setting_id in {
            "storage-disk",
            "storage-ram",
            "storage-uc",
        }
        expected_examples = (
            PUBLICATION_CAMPAIGN_STORAGE_EXAMPLES_PER_DATASET
            if storage_setting
            else PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
        )
        expected_repeats = (
            PUBLICATION_CAMPAIGN_STORAGE_REPEATS_PER_EXAMPLE
            if storage_setting
            else PUBLICATION_CAMPAIGN_REPEATS_PER_EXAMPLE
        )
        expected_requests = (
            PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL
            if storage_setting
            else PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL
        )
        if self.examples_per_dataset != expected_examples:
            raise ValueError("examples_per_dataset differs from the setting protocol")
        if self.repeats_per_example != expected_repeats:
            raise ValueError("repeats_per_example differs from the setting protocol")
        if self.request_count != expected_requests:
            raise ValueError("request_count differs from the setting protocol")


@dataclass(frozen=True, slots=True)
class PublicationCampaignPlan:
    """The complete 115-job latency design plus one complete score pass."""

    campaign_id: str
    campaign_ledger_id: str
    campaign_ledger_path_sha256: str
    campaign_ledger_prefix: DatabricksLedgerPrefix
    campaign_opening_terminal_gpu_hours: float
    latency_cells: tuple[PublicationLatencyCell, ...]
    auxiliary_latency_cells: tuple[PublicationAuxiliaryLatencyCell, ...]

    def __post_init__(self) -> None:
        if self.campaign_id != PUBLICATION_CAMPAIGN_ID:
            raise ValueError("campaign_id differs from the frozen publication campaign")
        if not self.campaign_ledger_id:
            raise ValueError("campaign_ledger_id must be non-empty")
        if len(self.campaign_ledger_path_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.campaign_ledger_path_sha256
        ):
            raise ValueError(
                "campaign_ledger_path_sha256 must be a lowercase SHA-256 digest"
            )
        if not isinstance(self.campaign_ledger_prefix, DatabricksLedgerPrefix):
            raise TypeError("campaign_ledger_prefix must be a DatabricksLedgerPrefix")
        if self.campaign_ledger_prefix.ledger_id != self.campaign_ledger_id:
            raise ValueError("campaign ledger ID differs from its retained prefix")
        if self.campaign_ledger_prefix.cap_cluster_hours != (
            MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
        ):
            raise ValueError("campaign ledger prefix must use the 1,024-hour cap")
        if self.campaign_ledger_id != PUBLICATION_CAMPAIGN_LEDGER_ID:
            raise ValueError("campaign ledger ID differs from the retained ledger")
        if self.campaign_ledger_path_sha256 != (
            PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256
        ):
            raise ValueError("campaign ledger path differs from the retained ledger")
        if self.campaign_ledger_prefix != PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX:
            raise ValueError("campaign ledger prefix differs from the retained opening")
        if self.campaign_opening_terminal_gpu_hours != (
            PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
        ):
            raise ValueError("campaign opening terminal GPU-hours drift")
        cells = tuple(self.latency_cells)
        expected = _publication_latency_cells()
        if cells != expected:
            raise ValueError(
                "latency_cells do not match the frozen publication factorial"
            )
        object.__setattr__(self, "latency_cells", cells)
        auxiliary_cells = tuple(self.auxiliary_latency_cells)
        expected_auxiliary = _publication_auxiliary_latency_cells()
        if auxiliary_cells != expected_auxiliary:
            raise ValueError(
                "auxiliary_latency_cells do not match the frozen publication design"
            )
        object.__setattr__(self, "auxiliary_latency_cells", auxiliary_cells)


def build_publication_campaign_plan(
    campaign_id: str,
    *,
    campaign_ledger_id: str,
    campaign_ledger_path_sha256: str,
    campaign_ledger_prefix: DatabricksLedgerPrefix,
    campaign_opening_terminal_gpu_hours: float,
) -> PublicationCampaignPlan:
    """Return the immutable vLLM 0.27.1 publication latency plan."""

    return PublicationCampaignPlan(
        campaign_id=campaign_id,
        campaign_ledger_id=campaign_ledger_id,
        campaign_ledger_path_sha256=campaign_ledger_path_sha256,
        campaign_ledger_prefix=campaign_ledger_prefix,
        campaign_opening_terminal_gpu_hours=campaign_opening_terminal_gpu_hours,
        latency_cells=_publication_latency_cells(),
        auxiliary_latency_cells=_publication_auxiliary_latency_cells(),
    )


def publication_campaign_plan_to_record(
    plan: PublicationCampaignPlan,
) -> dict[str, Any]:
    """Serialize a campaign plan with a canonical closure digest."""

    if not isinstance(plan, PublicationCampaignPlan):
        raise TypeError("plan must be a PublicationCampaignPlan")
    reviewed_error_sha256_by_job = dict(
        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_SHA256_BY_JOB
    )
    reviewed_error_utf8_bytes_by_job = dict(
        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_UTF8_BYTES_BY_JOB
    )
    reviewed_failure_job_ids = (
        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OBSERVER_JOB_IDS
        + PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OPAQUE_WORKER_JOB_IDS
    )
    if (
        len(reviewed_error_sha256_by_job) != PUBLICATION_CAMPAIGN_GPU_QUALIFICATION_JOBS
        or tuple(reviewed_error_sha256_by_job)
        != tuple(sorted(reviewed_error_sha256_by_job))
        or tuple(reviewed_error_utf8_bytes_by_job)
        != tuple(sorted(reviewed_error_utf8_bytes_by_job))
        or set(reviewed_error_sha256_by_job) != set(reviewed_failure_job_ids)
        or set(reviewed_error_utf8_bytes_by_job) != set(reviewed_failure_job_ids)
        or len(
            PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OBSERVER_JOB_IDS
        )
        != 2
        or len(
            PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OPAQUE_WORKER_JOB_IDS
        )
        != 12
        or len(set(reviewed_failure_job_ids))
        != PUBLICATION_CAMPAIGN_GPU_QUALIFICATION_JOBS
        or any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in reviewed_error_sha256_by_job.values()
        )
        or any(
            type(byte_count) is not int or byte_count <= 0
            for byte_count in reviewed_error_utf8_bytes_by_job.values()
        )
    ):
        raise RuntimeError(
            "runtime observation and worker subprocess failure source closure drift"
        )
    record: dict[str, Any] = {
        "record_type": PUBLICATION_CAMPAIGN_RECORD_TYPE,
        "schema_version": PUBLICATION_CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": plan.campaign_id,
        "campaign_ledger_id": plan.campaign_ledger_id,
        "campaign_ledger_path_sha256": plan.campaign_ledger_path_sha256,
        "campaign_ledger_prefix": plan.campaign_ledger_prefix.to_record(),
        "campaign_opening_terminal_gpu_hours": (
            plan.campaign_opening_terminal_gpu_hours
        ),
        "engine_version": PUBLICATION_CAMPAIGN_ENGINE_VERSION,
        "request_protocol": {
            "datasets": list(SUPPORTED_V1_DATASETS),
            "examples_per_dataset": PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
            "repeats_per_example": PUBLICATION_CAMPAIGN_REPEATS_PER_EXAMPLE,
            "request_count_per_cell": PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL,
            "request_parallelism": list(PUBLICATION_CAMPAIGN_REQUEST_PARALLELISM),
            "closed_loop": True,
            "think_time_seconds": 0,
            "deployment_blocks": PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS,
        },
        "storage_request_protocol": {
            "datasets": list(SUPPORTED_V1_DATASETS),
            "examples_per_dataset": (PUBLICATION_CAMPAIGN_STORAGE_EXAMPLES_PER_DATASET),
            "repeats_per_example": (PUBLICATION_CAMPAIGN_STORAGE_REPEATS_PER_EXAMPLE),
            "request_count_per_cell": (PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL),
            "request_parallelism": 4,
            "deployment_blocks": PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS,
            "matched_settings": ["storage-disk", "storage-ram", "storage-uc"],
            "payload_cache_max_bytes": 16 * 1024**3,
            "selection": {
                "caller_selectable": False,
                "domain": PUBLICATION_CAMPAIGN_STORAGE_SELECTION_DOMAIN,
                "rule": "lowest_domain_separated_sha256_per_dataset",
                "source_examples_per_dataset": (
                    PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
                ),
            },
        },
        "latency_timeout_policy": publication_campaign_latency_timeout_policy(),
        "analysis": {
            "experimental_units": {
                "core_method": "matched_fresh_cluster_baseline_vanilla_pair",
                "precision_hardware": (
                    "matched_16k_c4_core_pair_plus_bf16_and_a10g_wave"
                ),
                "storage": "matched_fresh_cluster_disk_ram_uc_trio",
            },
            "bootstrap": "paired_hierarchical_deployment_and_example",
            "bootstrap_draws": PUBLICATION_CAMPAIGN_BOOTSTRAP_DRAWS,
            "post_hoc_cell_significance_allowed": False,
            "quality_preservation_gate": False,
            "opening_ledger_provenance": {
                "prefix_before_rejected_gpu_qualification": (
                    PUBLICATION_CAMPAIGN_PRE_REJECTED_QUALIFICATION_LEDGER_PREFIX.to_record()
                ),
                "rejected_gpu_qualification": {
                    "actual_gpu_hours": 0.0,
                    "evidence_closed_record_sha256": (
                        "6102fe08f1ea3385ce862201c3b6b3351396315aba35bda56309bc240554f083"
                    ),
                    "evidence_file_sha256": (
                        "1f6a76369658f6dd39021dc72807345194cd43e2ecd746e831846391dc4e5a2b"
                    ),
                    "failed_before_run_creation": True,
                    "http_status": 400,
                    "observed_parameters_json_bytes": 18_292,
                    "plan_sha256": (
                        "b0bf7fdc182a099fae8f7d2fef1441974f69e49e840601f20397032709baf9f4"
                    ),
                    "reservation_count_delta": 14,
                    "remote_active_runs_observed": 0,
                    "server_parameters_json_limit_bytes": 10_000,
                    "submission_receipt_count_delta": 0,
                    "terminal_actual_count_delta": 14,
                    "terminal_state": "failed",
                    "verification_source": "legacy_manual",
                },
                "prefix_before_failed_live_gpu_qualification": (
                    PUBLICATION_CAMPAIGN_PRE_FAILED_QUALIFICATION_LEDGER_PREFIX.to_record()
                ),
                "failed_live_gpu_qualification": {
                    "actual_gpu_hours": 1.5991302777777774,
                    "data_security_mode": "NONE",
                    "failed_before_run_creation": False,
                    "failure_class": "unity_catalog_volume_access",
                    "failure_reason": (
                        "qualification payload used NONE access mode and could not "
                        "resolve Unity Catalog Volume bootstrap; remaining pending "
                        "jobs canceled after first failures"
                    ),
                    "plan_sha256": (
                        "ebfeaf53cfa9c74400be59546b391b77ebde4e85defa1f1b11bc4b4255c80341"
                    ),
                    "reconciliation_manifest_closed_record_sha256": (
                        "644048afcd8f478aa6ba2776be97f4e6fce4396ddf853001c3d200cfbbd259eb"
                    ),
                    "reservation_count_delta": 14,
                    "run_creation_count": 14,
                    "submission_receipt_count_delta": 14,
                    "terminal_actual_count_delta": 14,
                    "terminal_result_state_counts": {
                        "CANCELED": 7,
                        "FAILED": 7,
                    },
                    "verification_source": "direct_runs_get",
                },
                "prefix_before_bootstrap_failure_gpu_qualification": (
                    PUBLICATION_CAMPAIGN_PRE_BOOTSTRAP_FAILURE_LEDGER_PREFIX.to_record()
                ),
                "bootstrap_failure_gpu_qualification": {
                    "actual_cluster_duration_seconds": 4_585.718,
                    "actual_gpu_hours": 1.2738105555555554,
                    "data_security_mode": "SINGLE_USER",
                    "failed_before_run_creation": False,
                    "failure_class": "spark_python_task_missing_dunder_file",
                    "failure_reason": (
                        "all fourteen tasks failed before package installation "
                        "because the reviewed bootstrap referenced undefined "
                        "__file__ under Databricks spark_python_task execution"
                    ),
                    "plan_sha256": (
                        "2cf4ef1092a435c1e713f2a94115021ea7069ab6295d18ce5fcb5d4a479ce997"
                    ),
                    "reconciliation_manifest_closed_record_sha256": (
                        "8c7623aa2618066ea0ccedcba1d35a340308da04aaa040f89364bc4ea3d1b71c"
                    ),
                    "reconciliation_manifest_file_sha256": (
                        "1d0246ece1d6f844420d22a26b729d3f0d971ca0b30c0bf1ef0b5a84dcf6f360"
                    ),
                    "reservation_count_delta": 14,
                    "reviewed_runner_sha256": (
                        "f5ee833621428d630df1a59952a485d4ac55cabf987186d98a40274a2cf8a958"
                    ),
                    "run_creation_count": 14,
                    "submission_receipt_count_delta": 14,
                    "terminal_actual_count_delta": 14,
                    "terminal_life_cycle_state_counts": {"INTERNAL_ERROR": 14},
                    "terminal_result_state_counts": {"FAILED": 14},
                    "verification_source": "direct_runs_get_and_runs_get_output",
                },
                "prefix_before_cluster_identity_failure_gpu_qualification": (
                    PUBLICATION_CAMPAIGN_PRE_CLUSTER_IDENTITY_FAILURE_LEDGER_PREFIX.to_record()
                ),
                "cluster_identity_failure_gpu_qualification": {
                    "actual_cluster_duration_seconds": 4_564.259,
                    "actual_gpu_hours": 1.2678497222222225,
                    "data_security_mode": "SINGLE_USER",
                    "expected_error": (
                        "RuntimeError: Databricks cluster identity is unavailable; "
                        "expected DATABRICKS_CLUSTER_ID or DB_CLUSTER_ID"
                    ),
                    "failed_before_run_creation": False,
                    "failure_class": "databricks_cluster_identity_unavailable",
                    "failure_reason": (
                        "qualification bootstrap could not resolve Databricks "
                        "cluster identity"
                    ),
                    "plan_sha256": (
                        "d6f7619f6a70311fac571b31bedc7974e756a1679218cf63b76a7e7ceb91ebec"
                    ),
                    "reconciled_ledger_file_sha256": (
                        PUBLICATION_CAMPAIGN_PRE_RUNTIME_LOCK_INDEX_FAILURE_LEDGER_FILE_SHA256
                    ),
                    "reconciliation_manifest_closed_record_sha256": (
                        "fbb1fd4250b3fc62b58778047b12fe3775e6cffbc8641b38a00c721a9d4c768d"
                    ),
                    "reconciliation_manifest_file_sha256": (
                        "06c527102283bb379ecb26a345e76467d7e1614771d9a3c8313e9ebe6d941cf9"
                    ),
                    "reservation_count_delta": 14,
                    "reviewed_runner_sha256": (
                        "04cfe3a16200f011710317d829b7c52c0e4ca12f95fd8d277c949e7d6856d5b0"
                    ),
                    "run_creation_count": 14,
                    "runs_get_output_keys": [
                        "error",
                        "error_trace",
                        "logs",
                        "logs_truncated",
                        "metadata",
                    ],
                    "single_user_name": "pliu@opentable.com",
                    "submission_receipt_count_delta": 14,
                    "task_life_cycle_state_counts": {"TERMINATED": 14},
                    "task_result_state_counts": {"FAILED": 14},
                    "terminal_actual_count_delta": 14,
                    "terminal_life_cycle_state_counts": {"INTERNAL_ERROR": 14},
                    "terminal_prefix_sha256": (
                        "376114c27f35725bab5418969d28a77d4a3600dba44d049b597512142856d86f"
                    ),
                    "terminal_result_state_counts": {"FAILED": 14},
                    "verification_source": "direct_runs_get_and_runs_get_output",
                },
                "prefix_before_runtime_lock_index_failure_gpu_qualification": (
                    PUBLICATION_CAMPAIGN_PRE_RUNTIME_LOCK_INDEX_FAILURE_LEDGER_PREFIX.to_record()
                ),
                "runtime_lock_index_failure_gpu_qualification": {
                    "actual_cluster_duration_seconds": 7_754.755,
                    "actual_gpu_hours": 2.1540986111111113,
                    "data_security_mode": "SINGLE_USER",
                    "evidence_tree_byte_count": 1_564_133,
                    "evidence_tree_file_count": 29,
                    "evidence_tree_sha256": (
                        "5016ed50001b77b77f329e858c01b1a65c5e927f1c55eec7fbc01208d8f25886"
                    ),
                    "failed_before_run_creation": False,
                    "failure_class": "pip_requirements_file_index_precedence",
                    "failure_reason": (
                        "pip requirements-file index precedence omitted the PyTorch "
                        "CU129 index and prevented hash-locked torch resolution"
                    ),
                    "normalized_error_sha256": (
                        "7544cab6366fc1813af8d04da00a8a1f76f1098e3b06c738d8ff8ddd392ae235"
                    ),
                    "plan_sha256": (
                        "f991036176d59df70f0e339be4eb4a67a7c03a51536f62bf440df1ac72fd0e33"
                    ),
                    "predicted_terminal_prefix_sha256": (
                        "381ed88dfca75a17cf11b09b7e3dedb435328e518e8f1f0f0d9591be27796f26"
                    ),
                    "reconciled_ledger_file_sha256": (
                        PUBLICATION_CAMPAIGN_PRE_SITE_PACKAGES_PATH_FAILURE_LEDGER_FILE_SHA256
                    ),
                    "reconciliation_manifest_closed_record_sha256": (
                        "2ee650e0e05ea059bd9f552d6975149c05cbda6dc8d3a715a73594913f078b29"
                    ),
                    "reconciliation_manifest_file_sha256": (
                        "e0f56f1250c4ce213d1a8ba0384ccdad1a1b38fb964c1b6bfcf5729006150455"
                    ),
                    "reservation_count_delta": 14,
                    "reviewed_runner_sha256": (
                        "04cfe3a16200f011710317d829b7c52c0e4ca12f95fd8d277c949e7d6856d5b0"
                    ),
                    "run_creation_count": 14,
                    "runs_get_output_keys": [
                        "error",
                        "error_trace",
                        "logs",
                        "logs_truncated",
                        "metadata",
                    ],
                    "single_user_name": "pliu@opentable.com",
                    "submission_receipt_count_delta": 14,
                    "task_life_cycle_state_counts": {"TERMINATED": 14},
                    "task_result_state_counts": {"FAILED": 14},
                    "terminal_actual_count_delta": 14,
                    "terminal_life_cycle_state_counts": {"INTERNAL_ERROR": 14},
                    "terminal_prefix_sha256": (
                        "381ed88dfca75a17cf11b09b7e3dedb435328e518e8f1f0f0d9591be27796f26"
                    ),
                    "terminal_result_state_counts": {"FAILED": 14},
                    "torch_resolution_log_marker": (
                        "No matching distribution found for torch==2.13.0+cu129"
                    ),
                    "verification_source": "direct_runs_get_and_runs_get_output",
                },
                "prefix_before_site_packages_path_failure_gpu_qualification": (
                    PUBLICATION_CAMPAIGN_PRE_SITE_PACKAGES_PATH_FAILURE_LEDGER_PREFIX.to_record()
                ),
                "site_packages_path_failure_gpu_qualification": {
                    "actual_cluster_duration_seconds": 11_498.35,
                    "actual_gpu_hours": 3.193986111111111,
                    "data_security_mode": "SINGLE_USER",
                    "evidence_tree_byte_count": 1_945_499,
                    "evidence_tree_file_count": 29,
                    "evidence_tree_sha256": (
                        "2c555ea534fc3d41d3bc998fcaff8f07aedf42e1872200e39f9ed46796081607"
                    ),
                    "failed_before_run_creation": False,
                    "failed_before_sentinel_worker_launch": True,
                    "failure_class": "nonexistent_debian_site_packages_scheme_path",
                    "failure_reason": (
                        "all fourteen hash-locked qualification runtimes installed "
                        "and verified, then failed before sentinel worker launch "
                        "because the site-packages read-only freezer rejected a "
                        "nonexistent Debian local dist-packages scheme path reported "
                        "by site.getsitepackages()"
                    ),
                    "normalized_error_sha256": (
                        "8937fb907ae789c647754b2bbe9dbc4d9e167b67b8e437613260373b658c0da3"
                    ),
                    "plan_file_sha256": (
                        "c63521b29233addc1c5ab4435dfa0d639135765bce7a54298c0b0b1200741651"
                    ),
                    "plan_sha256": (
                        "be4cb0e80e17c99d9c4bd8abb89b24efb6e1202072fb734c739d322812218c9c"
                    ),
                    "predicted_terminal_prefix_sha256": (
                        "a71cee32c1ae056d7db7c72c70fa72bcf5622d8a3ae6d72590c4435bb9db4af9"
                    ),
                    "reconciled_ledger_file_sha256": (
                        PUBLICATION_CAMPAIGN_PRE_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_LEDGER_FILE_SHA256
                    ),
                    "reconciliation_manifest_closed_record_sha256": (
                        "a685849f6446063bdd5b220cd3ac5218c6e49a1e2d8487acac36316537b35eb7"
                    ),
                    "reconciliation_manifest_file_sha256": (
                        "2996e67b6c6305544c11231266500dcb9c53aa2bbc701fa6d6e626299c2ab06e"
                    ),
                    "reservation_count_delta": 14,
                    "reviewed_runner_sha256": (
                        "ca93baeda09f3df050b0dad3b8f3091c0f74235c426bd66555b67bd4b6eeafbc"
                    ),
                    "run_creation_count": 14,
                    "runs_get_output_keys": [
                        "error",
                        "error_trace",
                        "logs",
                        "logs_truncated",
                        "metadata",
                    ],
                    "single_user_name": "pliu@opentable.com",
                    "submission_receipt_count_delta": 14,
                    "task_life_cycle_state_counts": {"TERMINATED": 14},
                    "task_result_state_counts": {"FAILED": 14},
                    "terminal_actual_count_delta": 14,
                    "terminal_life_cycle_state_counts": {"INTERNAL_ERROR": 14},
                    "terminal_prefix_sha256": (
                        "a71cee32c1ae056d7db7c72c70fa72bcf5622d8a3ae6d72590c4435bb9db4af9"
                    ),
                    "terminal_result_state_counts": {"FAILED": 14},
                    "verification_source": "direct_runs_get_and_runs_get_output",
                },
                "prefix_before_runtime_observation_and_worker_subprocess_failure_gpu_qualification": (
                    PUBLICATION_CAMPAIGN_PRE_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_LEDGER_PREFIX.to_record()
                ),
                "runtime_observation_and_worker_subprocess_failure_gpu_qualification": {
                    "actual_cluster_duration_seconds": (
                        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ACTUAL_CLUSTER_DURATION_SECONDS
                    ),
                    "actual_gpu_hours": (
                        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ACTUAL_GPU_HOURS
                    ),
                    "data_security_mode": "SINGLE_USER",
                    "evidence_tree_byte_count": (
                        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_EVIDENCE_TREE_TOTAL_BYTES
                    ),
                    "evidence_tree_file_count": (
                        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_EVIDENCE_TREE_FILE_COUNT
                    ),
                    "evidence_tree_sha256": (
                        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_EVIDENCE_TREE_SHA256
                    ),
                    "failed_before_run_creation": False,
                    "failure_class": (
                        "post_success_runtime_observation_and_opaque_worker_subprocess"
                    ),
                    "failure_reason": (
                        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_REASON
                    ),
                    "normalized_observer_error_sha256": (
                        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_NORMALIZED_OBSERVER_ERROR_SHA256
                    ),
                    "normalized_worker_error_sha256": (
                        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_NORMALIZED_WORKER_ERROR_SHA256
                    ),
                    "observer_job_count": len(
                        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OBSERVER_JOB_IDS
                    ),
                    "observer_job_ids": list(
                        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OBSERVER_JOB_IDS
                    ),
                    "observer_jobs_returned_measurements": True,
                    "opaque_worker_job_count": len(
                        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OPAQUE_WORKER_JOB_IDS
                    ),
                    "opaque_worker_job_ids": list(
                        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OPAQUE_WORKER_JOB_IDS
                    ),
                    "opaque_worker_underlying_causes_known": False,
                    "plan_file_sha256": (
                        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_PLAN_FILE_SHA256
                    ),
                    "plan_sha256": (
                        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_PLAN_SHA256
                    ),
                    "predecessor_campaign_closed_record_sha256": (
                        PUBLICATION_CAMPAIGN_PRE_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_CAMPAIGN_CLOSED_RECORD_SHA256
                    ),
                    "predecessor_campaign_file_sha256": (
                        PUBLICATION_CAMPAIGN_PRE_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_CAMPAIGN_FILE_SHA256
                    ),
                    "predicted_terminal_prefix_sha256": (
                        PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX.prefix_sha256
                    ),
                    "raw_error_sha256_by_job": reviewed_error_sha256_by_job,
                    "raw_error_utf8_bytes_by_job": reviewed_error_utf8_bytes_by_job,
                    "reconciled_accounted_gpu_hours": (
                        PUBLICATION_CAMPAIGN_OPENING_ACCOUNTED_GPU_HOURS
                    ),
                    "reconciled_active_reserved_gpu_hours": (
                        PUBLICATION_CAMPAIGN_OPENING_ACTIVE_RESERVED_GPU_HOURS
                    ),
                    "reconciled_ledger_file_sha256": (
                        PUBLICATION_CAMPAIGN_OPENING_LEDGER_FILE_SHA256
                    ),
                    "reconciled_remaining_gpu_hours": (
                        PUBLICATION_CAMPAIGN_OPENING_REMAINING_GPU_HOURS
                    ),
                    "reconciled_terminal_actual_gpu_hours": (
                        PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
                    ),
                    "reconciliation_manifest_closed_record_sha256": (
                        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_MANIFEST_SHA256
                    ),
                    "reconciliation_manifest_file_sha256": (
                        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_MANIFEST_FILE_SHA256
                    ),
                    "reservation_count_delta": 14,
                    "reviewed_runner_sha256": (
                        PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_RUNNER_SHA256
                    ),
                    "run_creation_count": 14,
                    "runs_get_output_keys": [
                        "error",
                        "error_trace",
                        "logs",
                        "logs_truncated",
                        "metadata",
                    ],
                    "single_user_name": "pliu@opentable.com",
                    "submission_receipt_count_delta": 14,
                    "task_life_cycle_state_counts": {"TERMINATED": 14},
                    "task_result_state_counts": {"FAILED": 14},
                    "terminal_actual_count_delta": 14,
                    "terminal_life_cycle_state_counts": {"INTERNAL_ERROR": 14},
                    "terminal_prefix_sha256": (
                        PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX.prefix_sha256
                    ),
                    "terminal_result_state_counts": {"FAILED": 14},
                    "verification_source": "direct_runs_get_and_runs_get_output",
                },
                "retained_opening_prefix": (
                    PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX.to_record()
                ),
            },
        },
        "budget": {
            "aggregate_gpu_hour_cap": MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
            "active_reservation_hour_cap": (
                MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS
            ),
            "unreserved_headroom_hours": (
                PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS
            ),
            "opening_terminal_gpu_hours": (
                PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
            ),
            "max_parallel_jobs": PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS,
            "full_launch_min_generation_tokens_per_second": (
                PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
            ),
            "gpu_qualification": {
                "all_jobs_required": True,
                "job_count": PUBLICATION_CAMPAIGN_GPU_QUALIFICATION_JOBS,
                "max_retries": 0,
                "task_timeout_seconds": (
                    PUBLICATION_CAMPAIGN_GPU_QUALIFICATION_TASK_TIMEOUT_SECONDS
                ),
                "worst_case_reserved_gpu_hours": (
                    PUBLICATION_CAMPAIGN_GPU_QUALIFICATION_MAX_RESERVED_GPU_HOURS
                ),
            },
            "cpu_control_plane": {
                "data_security_mode": "SINGLE_USER",
                "databricks_node_type_id": (
                    PUBLICATION_CAMPAIGN_CPU_COORDINATOR_NODE_TYPE_ID
                ),
                "gpu_tasks": 0,
                "handoff_tree_closure": {
                    "job_count": PUBLICATION_CAMPAIGN_HANDOFF_CLOSURE_CPU_JOBS,
                    "stages": ["q8", "bf16"],
                    "task_timeout_seconds": (
                        PUBLICATION_CAMPAIGN_HANDOFF_CLOSURE_CPU_TIMEOUT_SECONDS
                    ),
                },
                "included_in_gpu_hour_ledger": False,
                "latency_source_closure": {
                    "job_count": PUBLICATION_CAMPAIGN_LATENCY_SOURCE_CLOSURE_CPU_JOBS,
                    "task_timeout_seconds": (
                        PUBLICATION_CAMPAIGN_LATENCY_SOURCE_CLOSURE_CPU_TIMEOUT_SECONDS
                    ),
                },
                "full_score_tree_closure": {
                    "actions_per_wave": ["producer_ready", "consumer_evidence"],
                    "job_count": PUBLICATION_CAMPAIGN_FULL_SCORE_REMOTE_CPU_JOBS,
                    "task_timeout_seconds": (
                        PUBLICATION_CAMPAIGN_FULL_SCORE_REMOTE_CPU_TIMEOUT_SECONDS
                    ),
                    "wave_count": PUBLICATION_CAMPAIGN_FULL_SCORE_WAVES,
                },
                "max_retries": 0,
                "num_workers": 0,
                "single_node": True,
                "spark_version": PUBLICATION_CAMPAIGN_CPU_COORDINATOR_SPARK_VERSION,
                "timeout_upper_bound_cpu_node_hours": (
                    PUBLICATION_CAMPAIGN_CPU_COORDINATOR_TIMEOUT_NODE_HOURS
                ),
                "total_job_count": PUBLICATION_CAMPAIGN_TOTAL_CPU_COORDINATOR_JOBS,
            },
            "latency_handoff_generation": {
                "accounting_input_token_slots": (
                    PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_INPUT_TOKEN_SLOTS
                ),
                "cache_prefix_generation_tokens": (
                    PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_CACHE_PREFIX_TOKENS
                ),
                "included_in_aggregate_gpu_hour_cap": True,
                "max_gpu_hours_at_min_throughput": (
                    PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_MAX_GPU_HOURS_AT_GATE
                ),
                "max_persistent_gpu_workers": (PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS),
                "producer_gpu_tasks": (
                    PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS
                ),
                "producer_submission_shape": (
                    "independent_single_gpu_single_task_runs"
                ),
                "producer_task_timeout_seconds": (
                    PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_TASK_TIMEOUT_SECONDS
                ),
                "reservation_reconciliation": "per_producer_attempt",
                "coordinator_gpu_hours": 0.0,
                "worst_case_reserved_gpu_hours": (
                    PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_MAX_RESERVED_GPU_HOURS
                ),
                "throughput_scope": (
                    "generation_plus_worker_result_durable_write_per_gpu_second"
                ),
            },
            "bf16_handoff_generation": {
                "cache_prefix_generation_tokens": (
                    PUBLICATION_CAMPAIGN_BF16_HANDOFF_CACHE_PREFIX_TOKENS
                ),
                "included_in_aggregate_gpu_hour_cap": True,
                "max_gpu_hours_at_min_throughput": (
                    PUBLICATION_CAMPAIGN_BF16_HANDOFF_MAX_GPU_HOURS_AT_GATE
                ),
                "producer_gpu_tasks": (
                    PUBLICATION_CAMPAIGN_BF16_HANDOFF_PRODUCER_TASKS
                ),
                "producer_task_timeout_seconds": (
                    PUBLICATION_CAMPAIGN_BF16_HANDOFF_TASK_TIMEOUT_SECONDS
                ),
                "payload_bytes": PUBLICATION_CAMPAIGN_BF16_HANDOFF_PAYLOAD_BYTES,
                "payload_gib": PUBLICATION_CAMPAIGN_BF16_HANDOFF_PAYLOAD_GIB,
                "absolute_slot_envelope_bytes": (
                    PUBLICATION_CAMPAIGN_BF16_HANDOFF_SLOT_ENVELOPE_BYTES
                ),
                "worst_case_reserved_gpu_hours": (
                    PUBLICATION_CAMPAIGN_BF16_HANDOFF_MAX_RESERVED_GPU_HOURS
                ),
            },
            "full_score_execution": {
                "cache_prefix_generation_tokens": (
                    PUBLICATION_CAMPAIGN_FULL_SCORE_CACHE_PREFIX_TOKENS
                ),
                "example_count": PUBLICATION_CAMPAIGN_FULL_SCORE_EXAMPLES,
                "generation_max_gpu_hours_at_min_throughput": (
                    PUBLICATION_CAMPAIGN_FULL_SCORE_GENERATION_MAX_GPU_HOURS_AT_GATE
                ),
                "execution_plan_sha256": (
                    PUBLICATION_CAMPAIGN_FULL_SCORE_EXECUTION_PLAN_SHA256
                ),
                "inventory_sha256": (PUBLICATION_CAMPAIGN_FULL_SCORE_INVENTORY_SHA256),
                "live_p90_admission_required_after_each_matched_wave": True,
                "natural_prompt_inference_tokens": (
                    PUBLICATION_CAMPAIGN_FULL_SCORE_NATURAL_PROMPT_TOKENS
                ),
                "phase_count": PUBLICATION_CAMPAIGN_FULL_SCORE_PHASES,
                "producer_and_consumer_phases_per_wave": 2,
                "producer_timeout_upper_bound_gpu_hours": (
                    PUBLICATION_CAMPAIGN_FULL_SCORE_WAVES
                    * PUBLICATION_CAMPAIGN_FULL_SCORE_MAX_RESERVED_GPU_HOURS_PER_PHASE
                ),
                "consumer_timeout_upper_bound_gpu_hours": (
                    PUBLICATION_CAMPAIGN_FULL_SCORE_WAVES
                    * PUBLICATION_CAMPAIGN_FULL_SCORE_MAX_RESERVED_GPU_HOURS_PER_PHASE
                ),
                "shard_count": PUBLICATION_CAMPAIGN_FULL_SCORE_SHARDS,
                "shard_plan_sha256": (
                    PUBLICATION_CAMPAIGN_FULL_SCORE_SHARD_PLAN_SHA256
                ),
                "task_timeout_seconds": (
                    PUBLICATION_CAMPAIGN_FULL_SCORE_TASK_TIMEOUT_SECONDS
                ),
                "tasks_per_phase": PUBLICATION_CAMPAIGN_FULL_SCORE_TASKS_PER_PHASE,
                "wave_count": PUBLICATION_CAMPAIGN_FULL_SCORE_WAVES,
                "worst_case_reserved_gpu_hours_per_phase": (
                    PUBLICATION_CAMPAIGN_FULL_SCORE_MAX_RESERVED_GPU_HOURS_PER_PHASE
                ),
            },
            "generation_workload_total": {
                "cache_prefix_generation_tokens": (
                    PUBLICATION_CAMPAIGN_TOTAL_GENERATION_CACHE_PREFIX_TOKENS
                ),
                "max_gpu_hours_at_min_throughput": (
                    PUBLICATION_CAMPAIGN_TOTAL_GENERATION_MAX_GPU_HOURS_AT_GATE
                ),
                "scope": [
                    "latency_q8_handoffs",
                    "latency_bf16_handoffs",
                    "full_score_q8_handoffs",
                ],
                "non_generation_gpu_hours_available_after_opening_balance_and_headroom": (
                    PUBLICATION_CAMPAIGN_NON_GENERATION_GPU_HOURS_AVAILABLE_AT_GATE
                ),
            },
            "core_latency_jobs": len(plan.latency_cells),
            "auxiliary_latency_jobs": len(plan.auxiliary_latency_cells),
            "max_latency_wave_reserved_gpu_hours": (
                PUBLICATION_CAMPAIGN_MAX_LATENCY_WAVE_RESERVED_GPU_HOURS
            ),
            "latency_timeout_upper_bound": {
                "completion_guaranteed_by_timeout_bounds": False,
                "gpu_hours": (
                    PUBLICATION_CAMPAIGN_LATENCY_TIMEOUT_UPPER_BOUND_GPU_HOURS
                ),
                "job_counts_by_timeout_hours": {
                    str(timeout_hours): job_count
                    for timeout_hours, job_count in (
                        PUBLICATION_CAMPAIGN_LATENCY_TIMEOUT_JOB_COUNTS
                    )
                },
                "launch_policy": "terminal_actual_and_hard_headroom_gated",
            },
            "total_latency_jobs": PUBLICATION_CAMPAIGN_TOTAL_LATENCY_JOBS,
        },
        "latency_cells": [_latency_cell_to_record(cell) for cell in plan.latency_cells],
        "auxiliary_latency_cells": [
            _auxiliary_latency_cell_to_record(cell)
            for cell in plan.auxiliary_latency_cells
        ],
        "full_score_program": {
            "complete_population_required": True,
            "cache_prefix_generation_tokens": (
                PUBLICATION_CAMPAIGN_FULL_SCORE_CACHE_PREFIX_TOKENS
            ),
            "datasets": list(SUPPORTED_V1_DATASETS),
            "example_count": PUBLICATION_CAMPAIGN_FULL_SCORE_EXAMPLES,
            "max_natural_prompt_tokens": 32_768,
            "max_parallel_workers": PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS,
            "methods": list(PUBLICATION_CAMPAIGN_METHODS),
            "paired_example_bootstrap_draws": PUBLICATION_CAMPAIGN_BOOTSTRAP_DRAWS,
            "passes_per_method": 1,
            "padding": False,
            "quality_preservation_gate": False,
            "natural_prompt_inference_tokens": (
                PUBLICATION_CAMPAIGN_FULL_SCORE_NATURAL_PROMPT_TOKENS
            ),
            "shard_count": PUBLICATION_CAMPAIGN_FULL_SCORE_SHARDS,
            "wave_count": PUBLICATION_CAMPAIGN_FULL_SCORE_WAVES,
            "streaming_lifecycle": [
                "generate_q8_kv",
                "baseline_inference",
                "vanilla_inference",
                "validate_paired_outputs",
                "delete_ephemeral_q8_kv",
            ],
            "tokenizer_truncation": False,
            "unsupported_datasets_remain_na": ["longbench_v2", "ruler"],
        },
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    if record["closed_record_sha256"] != PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256:
        raise RuntimeError("frozen publication campaign record closure drift")
    return record


def publication_campaign_latency_timeout_policy() -> dict[str, Any]:
    """Return the closed, condition-specific no-retry latency timeout policy."""

    return {
        "auxiliary_c4_hours": (PUBLICATION_CAMPAIGN_AUXILIARY_LATENCY_TIMEOUT_HOURS),
        "core_hours_by_context_and_concurrency": {
            f"{input_tokens // 1024}k": {
                f"c{concurrency}": timeout_hours
                for concurrency, timeout_hours in concurrency_timeouts
            }
            for input_tokens, concurrency_timeouts in (
                PUBLICATION_CAMPAIGN_CORE_LATENCY_TIMEOUT_HOURS
            )
        },
        "max_retries": 0,
    }


def validate_publication_campaign_plan_record(record: Mapping[str, Any]) -> None:
    """Fail closed unless *record* is the canonical frozen campaign plan."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    _require_exact_keys(record, _PLAN_KEYS, "publication campaign")
    digest = record.get("closed_record_sha256")
    if digest != _closed_record_sha256(record):
        raise ValueError("publication campaign closed_record_sha256 is invalid")
    campaign_id = record.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError("publication campaign campaign_id must be non-empty")
    campaign_ledger_id = record.get("campaign_ledger_id")
    if not isinstance(campaign_ledger_id, str) or not campaign_ledger_id:
        raise ValueError("publication campaign campaign_ledger_id must be non-empty")
    campaign_ledger_path_sha256 = record.get("campaign_ledger_path_sha256")
    if not isinstance(campaign_ledger_path_sha256, str):
        raise ValueError(
            "publication campaign campaign_ledger_path_sha256 must be a string"
        )
    raw_campaign_ledger_prefix = record.get("campaign_ledger_prefix")
    if not isinstance(raw_campaign_ledger_prefix, Mapping):
        raise ValueError(
            "publication campaign campaign_ledger_prefix must be an object"
        )
    campaign_ledger_prefix = databricks_ledger_prefix_from_record(
        raw_campaign_ledger_prefix
    )
    campaign_opening_terminal_gpu_hours = record.get(
        "campaign_opening_terminal_gpu_hours"
    )
    if not isinstance(campaign_opening_terminal_gpu_hours, (int, float)) or isinstance(
        campaign_opening_terminal_gpu_hours, bool
    ):
        raise ValueError(
            "publication campaign opening terminal GPU-hours must be numeric"
        )
    expected = publication_campaign_plan_to_record(
        build_publication_campaign_plan(
            campaign_id,
            campaign_ledger_id=campaign_ledger_id,
            campaign_ledger_path_sha256=campaign_ledger_path_sha256,
            campaign_ledger_prefix=campaign_ledger_prefix,
            campaign_opening_terminal_gpu_hours=float(
                campaign_opening_terminal_gpu_hours
            ),
        )
    )
    if dict(record) != expected:
        raise ValueError("publication campaign record does not match the frozen plan")


def write_publication_campaign_plan_json(
    plan: PublicationCampaignPlan,
    path: str | Path,
) -> None:
    """Write a canonical campaign record without overwriting existing evidence."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(
            f"publication campaign record already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(publication_campaign_plan_to_record(plan), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def publication_campaign_full_launch_budget_projection(
    *,
    latency_handoff_generation_tokens_per_second: float,
    latency_handoff_generation_gpu_hours: float,
    other_terminal_gpu_hours: float,
    current_active_reserved_gpu_hours: float,
    proposed_full_launch_reserved_gpu_hours: float,
) -> dict[str, Any]:
    """Fail closed unless generation and the remaining launch preserve headroom.

    ``other_terminal_gpu_hours`` excludes latency-handoff generation.  The
    generation cost is therefore charged exactly once in the projection.
    """

    values = {
        "latency_handoff_generation_tokens_per_second": (
            latency_handoff_generation_tokens_per_second
        ),
        "latency_handoff_generation_gpu_hours": (latency_handoff_generation_gpu_hours),
        "other_terminal_gpu_hours": other_terminal_gpu_hours,
        "current_active_reserved_gpu_hours": current_active_reserved_gpu_hours,
        "proposed_full_launch_reserved_gpu_hours": (
            proposed_full_launch_reserved_gpu_hours
        ),
    }
    normalized: dict[str, float] = {}
    for field_name, value in values.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"{field_name} must be non-negative and finite")
        normalized[field_name] = float(value)
    throughput = normalized["latency_handoff_generation_tokens_per_second"]
    if throughput < PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND:
        raise ValueError(
            "latency handoff end-to-end throughput is below the 35-token/s "
            "full-launch gate"
        )
    generation_hours = normalized["latency_handoff_generation_gpu_hours"]
    if generation_hours > PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_MAX_GPU_HOURS_AT_GATE:
        raise ValueError(
            "latency handoff generation GPU-hours exceed the campaign allowance"
        )
    max_hours_for_observed_throughput = (
        PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_CACHE_PREFIX_TOKENS / throughput / 3600.0
    )
    if generation_hours > max_hours_for_observed_throughput + 1e-12:
        raise ValueError(
            "latency handoff generation cost is inconsistent with observed throughput"
        )
    projected_active = (
        normalized["current_active_reserved_gpu_hours"]
        + normalized["proposed_full_launch_reserved_gpu_hours"]
    )
    if projected_active > MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS:
        raise ValueError("projected active reservations exceed the 900-hour cap")
    projected_accounted = (
        normalized["other_terminal_gpu_hours"] + generation_hours + projected_active
    )
    if projected_accounted > MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS:
        raise ValueError("projected publication GPU-hours exceed the 1024-hour cap")
    remaining = MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS - projected_accounted
    if remaining < PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS:
        raise ValueError("projected publication launch does not preserve 124 hours")
    return {
        "active_reservation_hour_cap": (MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS),
        "aggregate_gpu_hour_cap": MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
        "latency_handoff_generation_gpu_hours": generation_hours,
        "latency_handoff_generation_tokens_per_second": throughput,
        "latency_handoff_generation_token_slots": (
            PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_INPUT_TOKEN_SLOTS
        ),
        "latency_handoff_generation_cache_prefix_tokens": (
            PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_CACHE_PREFIX_TOKENS
        ),
        "minimum_generation_tokens_per_second": (
            PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
        ),
        "projected_accounted_gpu_hours": projected_accounted,
        "projected_active_reserved_gpu_hours": projected_active,
        "projected_unreserved_gpu_hours": remaining,
        "required_unreserved_headroom_hours": (
            PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write the frozen vLLM 0.27.1 publication campaign plan."
    )
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--campaign-ledger-id", required=True)
    parser.add_argument("--campaign-ledger-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    ledger = read_databricks_cluster_hour_ledger_json(args.campaign_ledger_json)
    if ledger.ledger_id != args.campaign_ledger_id:
        raise ValueError("campaign ledger JSON has the wrong ledger ID")
    if ledger.cap_cluster_hours != MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS:
        raise ValueError("campaign ledger must be migrated to the 1,024-hour cap")
    if ledger.active_reserved_cluster_hours != 0:
        raise ValueError("campaign ledger must be quiescent before plan closure")
    write_publication_campaign_plan_json(
        build_publication_campaign_plan(
            args.campaign_id,
            campaign_ledger_id=args.campaign_ledger_id,
            campaign_ledger_path_sha256=databricks_ledger_path_sha256(
                args.campaign_ledger_json
            ),
            campaign_ledger_prefix=databricks_ledger_prefix(ledger),
            campaign_opening_terminal_gpu_hours=(ledger.terminal_actual_cluster_hours),
        ),
        args.output_json,
    )
    return 0


def _publication_latency_cells() -> tuple[PublicationLatencyCell, ...]:
    cells: list[PublicationLatencyCell] = []
    for block in range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1):
        for input_tokens in PUBLICATION_CAMPAIGN_CONTEXT_TOKENS:
            context_label = f"{input_tokens // 1024}k"
            for request_parallelism in PUBLICATION_CAMPAIGN_REQUEST_PARALLELISM:
                pair_id = f"block-{block:02d}-{context_label}-c{request_parallelism}"
                for method_id in PUBLICATION_CAMPAIGN_METHODS:
                    method_label = (
                        "baseline" if method_id == "baseline_prefill" else "vanilla"
                    )
                    cells.append(
                        PublicationLatencyCell(
                            cell_id=f"{pair_id}-{method_label}",
                            matched_pair_id=pair_id,
                            deployment_block=block,
                            method_id=method_id,
                            input_tokens=input_tokens,
                            request_parallelism=request_parallelism,
                        )
                    )
    return tuple(cells)


def _publication_auxiliary_latency_cells() -> tuple[
    PublicationAuxiliaryLatencyCell, ...
]:
    cells: list[PublicationAuxiliaryLatencyCell] = []
    for block in range(1, PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS + 1):
        reference_core_cell_id = f"block-{block:02d}-16k-c4-vanilla"
        storage_control_cell_id = f"block-{block:02d}-storage-disk"
        cells.append(
            PublicationAuxiliaryLatencyCell(
                cell_id=storage_control_cell_id,
                reference_core_cell_id=reference_core_cell_id,
                deployment_block=block,
                setting_id="storage-disk",
                comparison_family="storage",
                examples_per_dataset=(
                    PUBLICATION_CAMPAIGN_STORAGE_EXAMPLES_PER_DATASET
                ),
                repeats_per_example=(PUBLICATION_CAMPAIGN_STORAGE_REPEATS_PER_EXAMPLE),
                request_count=PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL,
            )
        )
        for (
            setting_id,
            comparison_family,
            _description,
        ) in PUBLICATION_CAMPAIGN_AUXILIARY_SETTINGS:
            storage_setting = setting_id in {"storage-ram", "storage-uc"}
            cells.append(
                PublicationAuxiliaryLatencyCell(
                    cell_id=f"block-{block:02d}-{setting_id}",
                    reference_core_cell_id=(
                        storage_control_cell_id
                        if storage_setting
                        else reference_core_cell_id
                    ),
                    deployment_block=block,
                    setting_id=setting_id,
                    comparison_family=comparison_family,
                    examples_per_dataset=(
                        PUBLICATION_CAMPAIGN_STORAGE_EXAMPLES_PER_DATASET
                        if storage_setting
                        else PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
                    ),
                    repeats_per_example=(
                        PUBLICATION_CAMPAIGN_STORAGE_REPEATS_PER_EXAMPLE
                        if storage_setting
                        else PUBLICATION_CAMPAIGN_REPEATS_PER_EXAMPLE
                    ),
                    request_count=(
                        PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL
                        if storage_setting
                        else PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL
                    ),
                )
            )
    return tuple(cells)


def _latency_cell_to_record(cell: PublicationLatencyCell) -> dict[str, Any]:
    record = {
        "cell_id": cell.cell_id,
        "matched_pair_id": cell.matched_pair_id,
        "deployment_block": cell.deployment_block,
        "method_id": cell.method_id,
        "input_tokens": cell.input_tokens,
        "request_parallelism": cell.request_parallelism,
        "examples_per_dataset": cell.examples_per_dataset,
        "repeats_per_example": cell.repeats_per_example,
        "request_count": cell.request_count,
    }
    _require_exact_keys(record, _CELL_KEYS, "publication latency cell")
    return record


def _auxiliary_latency_cell_to_record(
    cell: PublicationAuxiliaryLatencyCell,
) -> dict[str, Any]:
    record = {
        "cell_id": cell.cell_id,
        "reference_core_cell_id": cell.reference_core_cell_id,
        "deployment_block": cell.deployment_block,
        "setting_id": cell.setting_id,
        "comparison_family": cell.comparison_family,
        "method_id": cell.method_id,
        "input_tokens": cell.input_tokens,
        "request_parallelism": cell.request_parallelism,
        "examples_per_dataset": cell.examples_per_dataset,
        "repeats_per_example": cell.repeats_per_example,
        "request_count": cell.request_count,
    }
    _require_exact_keys(record, _AUXILIARY_CELL_KEYS, "auxiliary latency cell")
    return record


def _closed_record_sha256(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("closed_record_sha256", None)
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{label} must use a closed schema; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PUBLICATION_CAMPAIGN_AUXILIARY_LATENCY_TIMEOUT_HOURS",
    "PUBLICATION_CAMPAIGN_AUXILIARY_JOBS",
    "PUBLICATION_CAMPAIGN_AUXILIARY_SETTINGS",
    "PUBLICATION_CAMPAIGN_BOOTSTRAP_DRAWS",
    "PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256",
    "PUBLICATION_CAMPAIGN_PRE_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_CAMPAIGN_CLOSED_RECORD_SHA256",
    "PUBLICATION_CAMPAIGN_PRE_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_CAMPAIGN_FILE_SHA256",
    "PUBLICATION_CAMPAIGN_CPU_COORDINATOR_NODE_TYPE_ID",
    "PUBLICATION_CAMPAIGN_CPU_COORDINATOR_SPARK_VERSION",
    "PUBLICATION_CAMPAIGN_CPU_COORDINATOR_TIMEOUT_NODE_HOURS",
    "PUBLICATION_CAMPAIGN_BF16_HANDOFF_CACHE_PREFIX_TOKENS",
    "PUBLICATION_CAMPAIGN_BF16_HANDOFF_MAX_GPU_HOURS_AT_GATE",
    "PUBLICATION_CAMPAIGN_BF16_HANDOFF_MAX_RESERVED_GPU_HOURS",
    "PUBLICATION_CAMPAIGN_BF16_HANDOFF_PAYLOAD_BYTES",
    "PUBLICATION_CAMPAIGN_BF16_HANDOFF_PAYLOAD_GIB",
    "PUBLICATION_CAMPAIGN_BF16_HANDOFF_PRODUCER_TASKS",
    "PUBLICATION_CAMPAIGN_BF16_HANDOFF_SLOT_ENVELOPE_BYTES",
    "PUBLICATION_CAMPAIGN_BF16_HANDOFF_TASK_TIMEOUT_SECONDS",
    "PUBLICATION_CAMPAIGN_CONTEXT_TOKENS",
    "PUBLICATION_CAMPAIGN_CORE_LATENCY_TIMEOUT_HOURS",
    "PUBLICATION_CAMPAIGN_DEPLOYMENT_BLOCKS",
    "PUBLICATION_CAMPAIGN_ENGINE_VERSION",
    "PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET",
    "PUBLICATION_CAMPAIGN_FULL_SCORE_CACHE_PREFIX_TOKENS",
    "PUBLICATION_CAMPAIGN_FULL_SCORE_EXAMPLES",
    "PUBLICATION_CAMPAIGN_FULL_SCORE_EXECUTION_PLAN_SHA256",
    "PUBLICATION_CAMPAIGN_FULL_SCORE_GENERATION_MAX_GPU_HOURS_AT_GATE",
    "PUBLICATION_CAMPAIGN_FULL_SCORE_MAX_RESERVED_GPU_HOURS_PER_PHASE",
    "PUBLICATION_CAMPAIGN_FULL_SCORE_NATURAL_PROMPT_TOKENS",
    "PUBLICATION_CAMPAIGN_FULL_SCORE_INVENTORY_SHA256",
    "PUBLICATION_CAMPAIGN_FULL_SCORE_PHASES",
    "PUBLICATION_CAMPAIGN_FULL_SCORE_SHARDS",
    "PUBLICATION_CAMPAIGN_FULL_SCORE_SHARD_PLAN_SHA256",
    "PUBLICATION_CAMPAIGN_FULL_SCORE_TASKS_PER_PHASE",
    "PUBLICATION_CAMPAIGN_FULL_SCORE_TASK_TIMEOUT_SECONDS",
    "PUBLICATION_CAMPAIGN_FULL_SCORE_WAVES",
    "PUBLICATION_CAMPAIGN_FULL_SCORE_REMOTE_CPU_JOBS",
    "PUBLICATION_CAMPAIGN_FULL_SCORE_REMOTE_CPU_TIMEOUT_SECONDS",
    "PUBLICATION_CAMPAIGN_GPU_QUALIFICATION_JOBS",
    "PUBLICATION_CAMPAIGN_GPU_QUALIFICATION_MAX_RESERVED_GPU_HOURS",
    "PUBLICATION_CAMPAIGN_GPU_QUALIFICATION_TASK_TIMEOUT_SECONDS",
    "PUBLICATION_CAMPAIGN_ID",
    "PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS",
    "PUBLICATION_CAMPAIGN_MAX_LATENCY_WAVE_RESERVED_GPU_HOURS",
    "PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_INPUT_TOKEN_SLOTS",
    "PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_CACHE_PREFIX_TOKENS",
    "PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_MAX_GPU_HOURS_AT_GATE",
    "PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_MAX_RESERVED_GPU_HOURS",
    "PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS",
    "PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_TASK_TIMEOUT_SECONDS",
    "PUBLICATION_CAMPAIGN_LATENCY_SOURCE_CLOSURE_CPU_JOBS",
    "PUBLICATION_CAMPAIGN_LATENCY_SOURCE_CLOSURE_CPU_TIMEOUT_SECONDS",
    "PUBLICATION_CAMPAIGN_LATENCY_TIMEOUT_JOB_COUNTS",
    "PUBLICATION_CAMPAIGN_LATENCY_TIMEOUT_UPPER_BOUND_GPU_HOURS",
    "PUBLICATION_CAMPAIGN_LEDGER_ID",
    "PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256",
    "PUBLICATION_CAMPAIGN_METHODS",
    "PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND",
    "PUBLICATION_CAMPAIGN_NON_GENERATION_GPU_HOURS_AVAILABLE_AT_GATE",
    "PUBLICATION_CAMPAIGN_OPENING_ACCOUNTED_GPU_HOURS",
    "PUBLICATION_CAMPAIGN_OPENING_ACTIVE_RESERVED_GPU_HOURS",
    "PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS",
    "PUBLICATION_CAMPAIGN_OPENING_LEDGER_FILE_SHA256",
    "PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX",
    "PUBLICATION_CAMPAIGN_OPENING_REMAINING_GPU_HOURS",
    "PUBLICATION_CAMPAIGN_PRE_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_LEDGER_FILE_SHA256",
    "PUBLICATION_CAMPAIGN_PRE_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_LEDGER_PREFIX",
    "PUBLICATION_CAMPAIGN_PRE_SITE_PACKAGES_PATH_FAILURE_LEDGER_FILE_SHA256",
    "PUBLICATION_CAMPAIGN_PRE_SITE_PACKAGES_PATH_FAILURE_LEDGER_PREFIX",
    "PUBLICATION_CAMPAIGN_PRE_RUNTIME_LOCK_INDEX_FAILURE_LEDGER_FILE_SHA256",
    "PUBLICATION_CAMPAIGN_PRE_RUNTIME_LOCK_INDEX_FAILURE_LEDGER_PREFIX",
    "PUBLICATION_CAMPAIGN_PRE_CLUSTER_IDENTITY_FAILURE_LEDGER_PREFIX",
    "PUBLICATION_CAMPAIGN_PRE_BOOTSTRAP_FAILURE_LEDGER_PREFIX",
    "PUBLICATION_CAMPAIGN_PRE_FAILED_QUALIFICATION_LEDGER_PREFIX",
    "PUBLICATION_CAMPAIGN_PRE_REJECTED_QUALIFICATION_LEDGER_PREFIX",
    "PUBLICATION_CAMPAIGN_RECORD_TYPE",
    "PUBLICATION_CAMPAIGN_REPEATS_PER_EXAMPLE",
    "PUBLICATION_CAMPAIGN_REQUEST_PARALLELISM",
    "PUBLICATION_CAMPAIGN_REQUESTS_PER_CELL",
    "PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ACTUAL_CLUSTER_DURATION_SECONDS",
    "PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ACTUAL_GPU_HOURS",
    "PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_SHA256_BY_JOB",
    "PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_UTF8_BYTES_BY_JOB",
    "PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_EVIDENCE_TREE_FILE_COUNT",
    "PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_EVIDENCE_TREE_SHA256",
    "PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_EVIDENCE_TREE_TOTAL_BYTES",
    "PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_MANIFEST_FILE_SHA256",
    "PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_MANIFEST_SHA256",
    "PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_NORMALIZED_OBSERVER_ERROR_SHA256",
    "PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_NORMALIZED_WORKER_ERROR_SHA256",
    "PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OBSERVER_JOB_IDS",
    "PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OPAQUE_WORKER_JOB_IDS",
    "PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_PLAN_FILE_SHA256",
    "PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_PLAN_SHA256",
    "PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_REASON",
    "PUBLICATION_CAMPAIGN_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_RUNNER_SHA256",
    "PUBLICATION_CAMPAIGN_SCHEMA_VERSION",
    "PUBLICATION_CAMPAIGN_STORAGE_CONTROL_SETTING",
    "PUBLICATION_CAMPAIGN_STORAGE_EXAMPLES_PER_DATASET",
    "PUBLICATION_CAMPAIGN_STORAGE_REPEATS_PER_EXAMPLE",
    "PUBLICATION_CAMPAIGN_STORAGE_REQUESTS_PER_CELL",
    "PUBLICATION_CAMPAIGN_STORAGE_SELECTION_DOMAIN",
    "PUBLICATION_CAMPAIGN_TOTAL_LATENCY_JOBS",
    "PUBLICATION_CAMPAIGN_TOTAL_CPU_COORDINATOR_JOBS",
    "PUBLICATION_CAMPAIGN_TOTAL_GENERATION_CACHE_PREFIX_TOKENS",
    "PUBLICATION_CAMPAIGN_TOTAL_GENERATION_MAX_GPU_HOURS_AT_GATE",
    "PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS",
    "PUBLICATION_CAMPAIGN_HANDOFF_CLOSURE_CPU_JOBS",
    "PUBLICATION_CAMPAIGN_HANDOFF_CLOSURE_CPU_TIMEOUT_SECONDS",
    "PublicationAuxiliaryLatencyCell",
    "PublicationCampaignPlan",
    "PublicationLatencyCell",
    "build_publication_campaign_plan",
    "main",
    "publication_campaign_plan_to_record",
    "publication_campaign_latency_timeout_policy",
    "publication_campaign_full_launch_budget_projection",
    "validate_publication_campaign_plan_record",
    "write_publication_campaign_plan_json",
]
