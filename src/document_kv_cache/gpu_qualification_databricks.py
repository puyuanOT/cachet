"""Databricks execution boundary for the closed vLLM GPU qualification plan.

The plan module deliberately has no cloud side effects.  This module turns one
validated plan into one independent ``runs/submit`` payload per sentinel and
provides the GPU-side result sealing boundary.  It still does not upload or
submit anything.

Every task is attempt-zero-only, binds all immutable artifacts by URI and
SHA-256, and writes to a plan/job-specific path with exclusive creation.  A
sentinel implementation returns measurements only through an in-process
callable; the executor validates the complete sentinel-specific schema before
publishing a canonical job-result record.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import shutil
import stat
import subprocess
import urllib.request
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Final, Protocol, cast
from urllib.parse import unquote, urlsplit

from document_kv_cache._hardware_targets import (
    databricks_node_type_for_hardware_target,
)
from document_kv_cache.databricks_job import (
    DEFAULT_DATABRICKS_RUN_TIMEOUT_SECONDS,
    DEFAULT_DATABRICKS_SPARK_VERSION,
    DatabricksSingleNodeGPUClusterConfig,
    build_single_node_gpu_cluster,
)
from document_kv_cache.databricks_resource_ledger import (
    MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS,
    MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
    DatabricksBatchReservationAuthorization,
    DatabricksClusterHourLedger,
    DatabricksClusterHourReservation,
    DatabricksClusterHourTerminalActual,
    DatabricksLedgerPrefix,
    DatabricksRunAttemptReservationRequest,
    canonical_databricks_submit_payload_snapshot,
    databricks_cluster_hour_ledger_to_record,
    databricks_ledger_prefix_at_counts,
    databricks_ledger_prefix_from_record,
    databricks_ledger_path_sha256,
    read_databricks_cluster_hour_ledger_json,
    replay_databricks_run_attempt_batch_authorization_json,
    record_databricks_run_submission_receipt_json,
    record_databricks_verified_run_terminal_actual_json,
    require_databricks_ledger_prefix,
    require_databricks_publication_batch_admission,
    reserve_databricks_run_attempt_batch_authorized_json,
)
from document_kv_cache.databricks_runs import (
    DATABRICKS_VOLUME_FILE_MAX_DOWNLOAD_BYTES,
    DatabricksURLOpener,
    DatabricksWorkspaceConfig,
    bind_databricks_run_idempotency_token,
    download_databricks_volume_file_bytes,
    get_databricks_run,
    get_databricks_run_output,
    require_databricks_run_idempotency_token,
    resume_pre_reserved_databricks_run,
    submit_pre_reserved_databricks_run,
)
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_SCHEMA_VERSION,
    GPU_QUALIFICATION_TERMINAL_RECEIPT_RECORD_TYPE,
    GPU_QUALIFICATION_GENERATION_DATABRICKS_NODE_TYPE,
    GPU_QUALIFICATION_GENERATION_HARDWARE_ID,
    GPU_QUALIFICATION_MAX_CLOUD_JOBS,
    GPU_QUALIFICATION_PLAN_RECORD_TYPE,
    GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
    GPUQualificationArtifactPins,
    GPUQualificationSelection,
    _build_governed_cloud_gpu_evidence,
    _build_governed_gpu_qualification_evidence,
    build_gpu_job_result,
    canonical_gpu_qualification_json,
    validate_gpu_job_result_record,
    validate_gpu_qualification_evidence_record,
    validate_gpu_qualification_plan_record,
    validate_local_preflight_evidence_record,
)
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_GPU_QUALIFICATION_JOBS,
    PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS,
    PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS,
)
from document_kv_cache.serving_env import VLLM_RUNTIME_LOCK_SHA256


GPU_QUALIFICATION_DATABRICKS_PURPOSE: Final = "cachet-vllm-0271-gpu-qualification"
GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS: Final = (
    DEFAULT_DATABRICKS_RUN_TIMEOUT_SECONDS
)
GPU_QUALIFICATION_DATABRICKS_PARAMETERS_MAX_BYTES: Final = 9_500
GPU_QUALIFICATION_DATABRICKS_DATA_SECURITY_MODE: Final = "SINGLE_USER"
GPU_QUALIFICATION_RUN_OUTPUT_LOG_MAX_UTF8_BYTES: Final = 5 * 1024 * 1024
GPU_QUALIFICATION_LEGACY_UC_FAILURE_PLAN_SHA256: Final = (
    "ebfeaf53cfa9c74400be59546b391b77ebde4e85defa1f1b11bc4b4255c80341"
)
GPU_QUALIFICATION_LEGACY_UC_BROKEN_RUNNER_SHA256: Final = (
    "acec0bf48ffcd67ee005e2c017b86540e3601ab3d9739f71f243069cae9007db"
)
GPU_QUALIFICATION_LEGACY_UC_RUNTIME_LOCK_SHA256: Final = (
    "5788ee492a9a9ff48c8e1eae68cd0576fcec625263858129cc9dd918bcb856a6"
)
GPU_QUALIFICATION_LEGACY_UC_FAILURE_MANIFEST_CLOSED_RECORD_SHA256: Final = (
    "644048afcd8f478aa6ba2776be97f4e6fce4396ddf853001c3d200cfbbd259eb"
)
GPU_QUALIFICATION_LEGACY_UC_FAILURE_TERMINAL_PREFIX_SHA256: Final = (
    "4bbe1144d4ce037fd8cf3376fc20c4e19ad00641f84c0a54d0cc2c17e37bf728"
)
GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_PLAN_SHA256: Final = (
    "2cf4ef1092a435c1e713f2a94115021ea7069ab6295d18ce5fcb5d4a479ce997"
)
GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_RUNNER_SHA256: Final = (
    "f5ee833621428d630df1a59952a485d4ac55cabf987186d98a40274a2cf8a958"
)
GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_MANIFEST_SHA256: Final = (
    "8c7623aa2618066ea0ccedcba1d35a340308da04aaa040f89364bc4ea3d1b71c"
)
GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_MANIFEST_FILE_SHA256: Final = (
    "1d0246ece1d6f844420d22a26b729d3f0d971ca0b30c0bf1ef0b5a84dcf6f360"
)
GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_TERMINAL_PREFIX_SHA256: Final = (
    "273aeb12c61060ca8d7850f5583f8912fa2a44ede44ddcba030da63926bff368"
)
GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_REASON: Final = (
    "all fourteen SINGLE_USER qualification tasks failed before package installation "
    "because reviewed bootstrap "
    "f5ee833621428d630df1a59952a485d4ac55cabf987186d98a40274a2cf8a958 "
    "referenced undefined __file__ under Databricks spark_python_task execution"
)
GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_ERROR: Final = (
    "NameError: name '__file__' is not defined"
)
GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_PLAN_SHA256: Final = (
    "d6f7619f6a70311fac571b31bedc7974e756a1679218cf63b76a7e7ceb91ebec"
)
GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_RUNNER_SHA256: Final = (
    "04cfe3a16200f011710317d829b7c52c0e4ca12f95fd8d277c949e7d6856d5b0"
)
GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_MANIFEST_SHA256: Final = (
    "fbb1fd4250b3fc62b58778047b12fe3775e6cffbc8641b38a00c721a9d4c768d"
)
GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_MANIFEST_FILE_SHA256: Final = (
    "06c527102283bb379ecb26a345e76467d7e1614771d9a3c8313e9ebe6d941cf9"
)
GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_TERMINAL_PREFIX_SHA256: Final = (
    "376114c27f35725bab5418969d28a77d4a3600dba44d049b597512142856d86f"
)
GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_REASON: Final = (
    "qualification bootstrap could not resolve Databricks cluster identity"
)
GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_ERROR: Final = (
    "RuntimeError: Databricks cluster identity is unavailable; expected "
    "DATABRICKS_CLUSTER_ID or DB_CLUSTER_ID"
)
GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_PLAN_SHA256: Final = (
    "f991036176d59df70f0e339be4eb4a67a7c03a51536f62bf440df1ac72fd0e33"
)
GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_RUNNER_SHA256: Final = (
    "04cfe3a16200f011710317d829b7c52c0e4ca12f95fd8d277c949e7d6856d5b0"
)
GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_MANIFEST_SHA256: Final = (
    "2ee650e0e05ea059bd9f552d6975149c05cbda6dc8d3a715a73594913f078b29"
)
GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_MANIFEST_FILE_SHA256: Final = (
    "e0f56f1250c4ce213d1a8ba0384ccdad1a1b38fb964c1b6bfcf5729006150455"
)
GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_TERMINAL_PREFIX_SHA256: Final = (
    "381ed88dfca75a17cf11b09b7e3dedb435328e518e8f1f0f0d9591be27796f26"
)
GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_REASON: Final = (
    "pip requirements-file index precedence omitted the PyTorch CU129 index "
    "and prevented hash-locked torch resolution"
)
GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_NORMALIZED_ERROR_SHA256: Final = (
    "7544cab6366fc1813af8d04da00a8a1f76f1098e3b06c738d8ff8ddd392ae235"
)
GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_LOG_MARKER: Final = (
    "No matching distribution found for torch==2.13.0+cu129"
)
GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_ERROR_SHA256_BY_JOB: Final = (
    (
        "aws-g5-a10g-16k-c4-capacity",
        "03c34df076b0eb16e1af6effca533ff2eb009f513a9000f45477a6a88ffd02bc",
    ),
    (
        "aws-g5-a10g-auto-backend-diagnostic",
        "0ca4734369079b0262cfb574fc087130503585381646fe015074a4226bdfb65d",
    ),
    (
        "aws-g5-a10g-forced-triton-runtime-handoff",
        "d3fa07c4e20514c6b3efd97bafde8d3db4525848244a711f10adf2e4a179efeb",
    ),
    (
        "aws-g5-a10g-matched-token-logit",
        "d56238b226901bcbebb00a9e1d6b2cba06627334ae0e10a827060d99f2ca0a5d",
    ),
    (
        "aws-g5-a10g-packed-page-roundtrip",
        "5c502d96b254c0bd229e5d81e75f9a99d9c2ca7f6fe36137ea8b971742a81a36",
    ),
    (
        "aws-g6-l4-32k-c4-gmu-70",
        "5d5b3e21a9fb8bff509dd30f65120b278f061cfef4587ea15eea1a87946644b5",
    ),
    (
        "aws-g6-l4-32k-c4-gmu-75",
        "a2eec14578235f86478cd0a3df527ff24efb8e3c5c5548882ba20b3540a9b813",
    ),
    (
        "aws-g6-l4-32k-c4-gmu-80",
        "0ef523fe6dc72f093ab1fdfd0601e090e7338a48d8cb4404a107de139ef33650",
    ),
    (
        "aws-g6-l4-auto-backend-diagnostic",
        "34346041c0f03a7fa991f2223ad2ad5656dd6cac36b9bc228453055af4a2aa3e",
    ),
    (
        "aws-g6-l4-forced-triton-runtime-handoff",
        "f280d4f9d4744311465641e2ebe08bda2fa7f0dc5f269f1869546443a542adfa",
    ),
    (
        "aws-g6-l4-generation-throughput",
        "216f5bbba90d7efcb2494128be781e2ddb6d883335c4dcd1dbc1cccc3aaf2afd",
    ),
    (
        "aws-g6-l4-matched-token-logit",
        "d0de6d6286427688266b122c4039ae1bfabf635077bfb5561e46428b881f6705",
    ),
    (
        "aws-g6-l4-packed-page-roundtrip",
        "9b1ecf2d68c83012751827ecb094f2d3bd29a386377acdfadbf263ec15cf78a5",
    ),
    (
        "aws-g6e-l40s-generation-throughput",
        "62f7d0aac6c1a2e9d4cb84ae074e815fb718ad179574f6d2a8b54ff2118d9437",
    ),
)
GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_PLAN_SHA256: Final = (
    "be4cb0e80e17c99d9c4bd8abb89b24efb6e1202072fb734c739d322812218c9c"
)
GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_RUNNER_SHA256: Final = (
    "ca93baeda09f3df050b0dad3b8f3091c0f74235c426bd66555b67bd4b6eeafbc"
)
GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_MANIFEST_SHA256: Final = (
    "a685849f6446063bdd5b220cd3ac5218c6e49a1e2d8487acac36316537b35eb7"
)
GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_MANIFEST_FILE_SHA256: Final = (
    "2996e67b6c6305544c11231266500dcb9c53aa2bbc701fa6d6e626299c2ab06e"
)
GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_EVIDENCE_TREE_SHA256: Final = (
    "2c555ea534fc3d41d3bc998fcaff8f07aedf42e1872200e39f9ed46796081607"
)
GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_EVIDENCE_TREE_FILE_COUNT: Final = 29
GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_EVIDENCE_TREE_TOTAL_BYTES: Final = (
    1_945_499
)
GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_TERMINAL_PREFIX_SHA256: Final = (
    "a71cee32c1ae056d7db7c72c70fa72bcf5622d8a3ae6d72590c4435bb9db4af9"
)
GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_REASON: Final = (
    "all fourteen hash-locked qualification runtimes installed and verified, "
    "then failed before sentinel worker launch because the site-packages read-only "
    "freezer rejected a nonexistent Debian local dist-packages scheme path reported "
    "by site.getsitepackages()"
)
GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_NORMALIZED_ERROR_SHA256: Final = (
    "8937fb907ae789c647754b2bbe9dbc4d9e167b67b8e437613260373b658c0da3"
)
GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_PIP_CHECK_LOG_MARKER: Final = (
    "No broken requirements found."
)
GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_FREEZER_TRACE_MARKER: Final = (
    "--> 115 _make_site_packages_read_only(runtime_python)"
)
GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_WORKER_MODULE_MARKER: Final = (
    "document_kv_cache._gpu_qualification_sentinel_worker"
)
GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_ERROR_SHA256_BY_JOB: Final = (
    (
        "aws-g5-a10g-16k-c4-capacity",
        "ff4a9c3082cdf1d63334f231646c0075e224baafb061df736d2a81d34d5a0682",
    ),
    (
        "aws-g5-a10g-auto-backend-diagnostic",
        "a1fb17abfb013b962bd082ef1cfe697f1607677856a16711404ebb0c39a544dd",
    ),
    (
        "aws-g5-a10g-forced-triton-runtime-handoff",
        "1544442770a99cae1b323cbf33077aa1a0f32f0b8e49d678f7bcbac8a71c33d6",
    ),
    (
        "aws-g5-a10g-matched-token-logit",
        "de8da0bb3eb3263110f86cde882d047ddf3caf7cbe37d2b82f6270131233c795",
    ),
    (
        "aws-g5-a10g-packed-page-roundtrip",
        "baee1999b43ee8e68b4dd3699eadd16407e5e03360e70de2c01c0381116be7a7",
    ),
    (
        "aws-g6-l4-32k-c4-gmu-70",
        "b6b3d69abdc302f0f2a90acc26a788ce835e3d2abf42c63a0fafb72f6da296d4",
    ),
    (
        "aws-g6-l4-32k-c4-gmu-75",
        "2670bca02ec39cc4acb997ef1550c1145ca1543eff96b02013e3e8d36407e2cc",
    ),
    (
        "aws-g6-l4-32k-c4-gmu-80",
        "c97f3c29f579c17ae8a2f8a707661098ddfd9fe632b73ef77b58f8effe1b35c8",
    ),
    (
        "aws-g6-l4-auto-backend-diagnostic",
        "b86c51071effabc7b29c023d907de8fc1337f06f3225656a91e0db86bbd3f130",
    ),
    (
        "aws-g6-l4-forced-triton-runtime-handoff",
        "9f9c87bb76014c2c9ec682039c8221e7bbd20e7682fb29a2ab0b70a3939eddcc",
    ),
    (
        "aws-g6-l4-generation-throughput",
        "b32c049c34fb10be220e2d6a6bbda1b6de0b07f89f142cbb11f3aeb45bb0cfe9",
    ),
    (
        "aws-g6-l4-matched-token-logit",
        "7b21ac504ec057eabcc4d625f6624aec7c23177727df496c6eda80c695db7e8f",
    ),
    (
        "aws-g6-l4-packed-page-roundtrip",
        "d412a5163fb9ce920a07a46cc7c04416a8b47bdfcbb4bca45bcca7de8cb97042",
    ),
    (
        "aws-g6e-l40s-generation-throughput",
        "fc37553551e0850b595d3a6be360fe41efdb58fb04252ca2f60f7a1b58567ca8",
    ),
)
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_PLAN_SHA256: Final = "c0bede45ea211798c9a5eb31010a91074ded70e370f8ea4fcbeb59b3b9f95598"
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_RUNNER_SHA256: Final = "ca93baeda09f3df050b0dad3b8f3091c0f74235c426bd66555b67bd4b6eeafbc"
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_MANIFEST_SHA256: Final = "6c4cca0ec4fbcf4ccb434573f965eeb8022909ce5bdd6afdf31d61085807fa9b"
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_MANIFEST_FILE_SHA256: Final = "53fd4b076a642101790d21ebbc03b1eb7e609428c2ccd7eafb8cbad5a9a3a112"
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_EVIDENCE_TREE_SHA256: Final = "bb6636f3b9bdf5afae0b7d1beb97f5f3192017ba5b04abb651f2a389889aa57f"
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_EVIDENCE_TREE_FILE_COUNT: Final = 29
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_EVIDENCE_TREE_TOTAL_BYTES: Final = 1_828_218
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_TERMINAL_PREFIX_SHA256: Final = "22ac65492fa0871f528552cfcae0bd6332b1429cd9fc2e92c373c5e534202d4a"
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_REASON: Final = (
    "all fourteen hash-locked qualification runtimes installed and verified; "
    "the two packed-page-roundtrip workers returned measurements before "
    "post-success runtime observation rejected the virtualenv-created "
    "runtime/bin/python symlink, while the other twelve sentinel-worker "
    "subprocesses exited nonzero and the reviewed launcher did not surface their "
    "captured child stdout/stderr, so their underlying worker causes remain unknown"
)
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_PACKED_PAGE_ROUNDTRIP_JOB_IDS: Final = (
    "aws-g5-a10g-packed-page-roundtrip",
    "aws-g6-l4-packed-page-roundtrip",
)
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_NORMALIZED_OBSERVER_ERROR_SHA256: Final = "3662915979987aef1fe4bcf9e0e62f06c67992ee73da679e44f6b6a261e634f5"
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_NORMALIZED_WORKER_ERROR_SHA256: Final = "3f1ddd73298cd46347cf57b84d6cf22f7d6e98802b50ded9457d7a999563786b"
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_PIP_CHECK_LOG_MARKER: Final = "No broken requirements found."
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_VIRTUALENV_LOG_PREFIX: Final = "created virtual environment CPython3.11.11.final.0-64-x86_64 in "
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ENSUREPIP_LOG_ARGV: Final = "'-m', 'ensurepip', '--upgrade', '--default-pip'"
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_WORKER_MODULE_MARKER: Final = "document_kv_cache._gpu_qualification_sentinel_worker"
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OBSERVER_TRACE_MARKER: Final = "_observe_gpu_runtime"
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OBSERVER_TRACE_MARKERS: Final = (
    "runtime = _observe_gpu_runtime(local_work_dir)",
    "if not runtime_python.is_file() or runtime_python.is_symlink():",
)
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_WORKER_TRACE_MARKERS: Final = (
    "measurements = sentinel_runner(",
    "_make_site_packages_read_only(runtime_python)",
    "completed = subprocess.run(",
    "capture_output=True",
    "check=True",
    "subprocess.py:571",
)
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_SHA256_BY_JOB: Final = (
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
GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_UTF8_BYTES_BY_JOB: Final = (
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
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_PLAN_SHA256: Final = (
    "694441bffc253141156f9c808666112d39bb5829d22825d1d88c93ab47a5e830"
)
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_RUNNER_SHA256: Final = (
    "ca93baeda09f3df050b0dad3b8f3091c0f74235c426bd66555b67bd4b6eeafbc"
)
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_MANIFEST_SHA256: Final = (
    "13ad4eabd10bde1b5c7e0aa7b9721dd3bd8fbe57f6c20204093749df8d84954f"
)
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_MANIFEST_FILE_SHA256: Final = (
    "a6e0c985d64b0072776dd1247094600d81b885cfe4a3fb0f6418e8b811134304"
)
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_EVIDENCE_TREE_SHA256: Final = (
    "7455fa1e30356bb79ccb75a8dbe24df32f33a365141505e0270eb13c7f39b71d"
)
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_EVIDENCE_TREE_FILE_COUNT: Final = 29
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_EVIDENCE_TREE_TOTAL_BYTES: Final = 2_094_892
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_TERMINAL_PREFIX_SHA256: Final = (
    "07b9663e42c2dd8040f689d08fabdd6d7eefaf25f8f1decedc23af683e0011c7"
)
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_REASON: Final = (
    "the fourteen terminal qualification failures comprise two post-measurement "
    "vLLM version-contract mismatches, two forced-runtime-handoff unresolved-native-"
    "object failures, eight sentinel layout-conflict failures, and two auto-backend "
    "FlashInfer array.array TypeError engine-initialization failures"
)
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_VERSION_MISMATCH_JOB_IDS: Final = (
    "aws-g5-a10g-packed-page-roundtrip",
    "aws-g6-l4-packed-page-roundtrip",
)
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_UNRESOLVED_NATIVE_JOB_IDS: Final = (
    "aws-g5-a10g-forced-triton-runtime-handoff",
    "aws-g6-l4-forced-triton-runtime-handoff",
)
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_LAYOUT_CONFLICT_JOB_IDS: Final = (
    "aws-g5-a10g-16k-c4-capacity",
    "aws-g5-a10g-matched-token-logit",
    "aws-g6-l4-32k-c4-gmu-70",
    "aws-g6-l4-32k-c4-gmu-75",
    "aws-g6-l4-32k-c4-gmu-80",
    "aws-g6-l4-generation-throughput",
    "aws-g6-l4-matched-token-logit",
    "aws-g6e-l40s-generation-throughput",
)
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FLASHINFER_JOB_IDS: Final = (
    "aws-g5-a10g-auto-backend-diagnostic",
    "aws-g6-l4-auto-backend-diagnostic",
)
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_ERROR_SHA256_BY_JOB: Final = (
    (
        "aws-g5-a10g-16k-c4-capacity",
        "59825b10900d049102b32f58e0b25bb6626f3539c6cde57b89f33e779da738b5",
    ),
    (
        "aws-g5-a10g-auto-backend-diagnostic",
        "8e08607cb3549eb9ea1cb80857044c44b6037c2bf4208127a31b9d2e312cb412",
    ),
    (
        "aws-g5-a10g-forced-triton-runtime-handoff",
        "355a98d0e8824fcd3535a9373ab3eaa754421b288ea9d4421bbeb38f159ea1cb",
    ),
    (
        "aws-g5-a10g-matched-token-logit",
        "794f86dcb4e3b89762a9912bfb2140657a91ab0df4dee95567af9c2b87f6e460",
    ),
    (
        "aws-g5-a10g-packed-page-roundtrip",
        "525c315ed7781384cef5fa876eac7aadfd19398edef393c3484fe73b87a5a869",
    ),
    (
        "aws-g6-l4-32k-c4-gmu-70",
        "443693966c8832bfc21b9318818a2db6a985d8324c6a4de45d1a8eec6ecd7f2b",
    ),
    (
        "aws-g6-l4-32k-c4-gmu-75",
        "220c4912372cfa444b14b35f15564b4fe93f5055fd416be794dba372517a0b98",
    ),
    (
        "aws-g6-l4-32k-c4-gmu-80",
        "9e1326b6abef77a80367c3e0fe0c1b1c3f45d7ff5ab0a4f8ecbaa9513c10aca8",
    ),
    (
        "aws-g6-l4-auto-backend-diagnostic",
        "ff228ca25675d3c8ee31327e46d633302023a6fd8fe8f684ae6ff205acedaceb",
    ),
    (
        "aws-g6-l4-forced-triton-runtime-handoff",
        "bd1aaa527607060c813c0fdc089ee331b7d0c95c98075d89de67c1416f613b7d",
    ),
    (
        "aws-g6-l4-generation-throughput",
        "0b5cfff5762b6e19484961c57bb782d9745a9b47ef50f8ae8a2323e49cfb26f7",
    ),
    (
        "aws-g6-l4-matched-token-logit",
        "015a8e1887c894b8056340eeef2e1241f1d47315400b50b3ceca8f9293b80b58",
    ),
    (
        "aws-g6-l4-packed-page-roundtrip",
        "1fb0a09907cee02a066ea424f16f7893f0a7e804b60a0f9a315a4c79b4b4d6f2",
    ),
    (
        "aws-g6e-l40s-generation-throughput",
        "2f408a4de16b3314b9ee261720e4c161aaa8c3a4acf37f56c926b8a16cd86315",
    ),
)
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_ERROR_UTF8_BYTES_BY_JOB: Final = (
    ("aws-g5-a10g-16k-c4-capacity", 4_385),
    ("aws-g5-a10g-auto-backend-diagnostic", 6_416),
    ("aws-g5-a10g-forced-triton-runtime-handoff", 2_009),
    ("aws-g5-a10g-matched-token-logit", 4_390),
    ("aws-g5-a10g-packed-page-roundtrip", 78),
    ("aws-g6-l4-32k-c4-gmu-70", 4_380),
    ("aws-g6-l4-32k-c4-gmu-75", 4_380),
    ("aws-g6-l4-32k-c4-gmu-80", 4_380),
    ("aws-g6-l4-auto-backend-diagnostic", 6_415),
    ("aws-g6-l4-forced-triton-runtime-handoff", 1_999),
    ("aws-g6-l4-generation-throughput", 4_390),
    ("aws-g6-l4-matched-token-logit", 4_388),
    ("aws-g6-l4-packed-page-roundtrip", 76),
    ("aws-g6e-l40s-generation-throughput", 4_393),
)
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_PIP_CHECK_LOG_MARKER: Final = "No broken requirements found."
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_VIRTUALENV_LOG_PREFIX: Final = "created virtual environment CPython3.11.11.final.0-64-x86_64 in "
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_ENSUREPIP_LOG_ARGV: Final = "'-m', 'ensurepip', '--upgrade', '--default-pip'"
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_WORKER_MODULE_MARKER: Final = "document_kv_cache._gpu_qualification_sentinel_worker"
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_WORKER_TRACE_MARKERS: Final = (
    ("execute_gpu_qualification_job", 2),
    ("_builtin_sentinel_runner", 2),
    ("run_gpu_qualification_sentinel", 2),
    ("_run_bounded_worker_process", 2),
    ("_worker_process_failure", 1),
)
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_VERSION_TRACE_MARKERS: Final = (
    ("execute_gpu_qualification_job", 2),
    ("_builtin_sentinel_runner", 1),
    ("validate_gpu_job_result_record", 2),
    ("_validate_job_result_common", 2),
)
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_UNRESOLVED_NATIVE_ERROR_MARKERS: Final = (
    '_gpu_qualification_sentinel_worker.py", line 509, in _runtime_handoff_sentinel',
    'raise RuntimeError("the isolated runtime has unresolved native objects")',
    "RuntimeError: the isolated runtime has unresolved native objects",
)
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_LAYOUT_ERROR_MARKERS: Final = (
    "bind_layout(layout)",
    'transformers_generator.py", line 263, in bind_layout',
    'raise ValueError("generator layout conflicts with the resolved handoff layout")',
    "ValueError: generator layout conflicts with the resolved handoff layout",
)
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FLASHINFER_ERROR_MARKERS: Final = (
    'flashinfer/comm/fd_exchange.py", line 55, in <module>',
    "def _fd_ancillary(fd: int) -> tuple[tuple[int, int, array.array[int]]]",
    "TypeError: type \\'array.array\\' is not subscriptable",
    "RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}",
)
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_LEDGER_SHA256: Final = (
    "784a43eafec2f6d6086b4258959b308043e183f361218463be14dea3702bd62d"
)
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_LEDGER_BYTES: Final = 220_426
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_RESERVATION_COUNT: Final = 236
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_SUBMISSION_RECEIPT_COUNT: Final = 98
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_TERMINAL_ACTUAL_COUNT: Final = 236
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_TERMINAL_HOURS: Final = 71.39012833333337
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_REMAINING_HOURS: Final = 952.6098716666667
GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_NEW_TERMINAL_SECONDS: Final = 12_455.252
_RUNTIME_LOCK_INDEX_FAILURE_NORMALIZED_ERROR: Final = (
    "CalledProcessError: Command '['{runtime_python}', '-m', 'pip', 'install', "
    "'--extra-index-url', 'https://download.pytorch.org/whl/cu129', "
    "'--extra-index-url', 'https://flashinfer.ai/whl/', '--extra-index-url', "
    "'https://flashinfer.ai/whl/cu129', '--require-hashes', '--only-binary', "
    "':all:', '--requirement', '{runtime_lock}']' returned non-zero exit status 1."
)
_RUNTIME_LOCK_INDEX_FAILURE_LOCK_PATH_RE: Final = re.compile(
    r"/local_disk0/\.ephemeral_nfs/envs/pythonEnv-"
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
    r"/lib/python3\.11/site-packages/document_kv_cache/runtime_locks/"
    r"vllm-0\.27\.1-cu129-py311-manylinux_2_35\.lock"
)
_SITE_PACKAGES_PATH_FAILURE_NORMALIZED_ERROR: Final = (
    "RuntimeError: invalid isolated site-packages path: {invalid_site_packages}"
)
_RUNTIME_OBSERVATION_FAILURE_NORMALIZED_ERROR: Final = (
    "RuntimeError: sentinel did not materialize the required isolated runtime "
    "Python at {work_root}/runtime/bin/python"
)
_WORKER_SUBPROCESS_FAILURE_NORMALIZED_ERROR: Final = (
    "CalledProcessError: Command '['{work_root}/runtime/bin/python', '-m', "
    "'document_kv_cache._gpu_qualification_sentinel_worker', '--plan-json', "
    "'{work_root}/worker/plan.json', '--job-json', "
    "'{work_root}/worker/planned-job.json', '--input-bundle', "
    "'{work_root}/artifact-snapshot/input_bundle_sha256', '--work-dir', "
    "'{work_root}/worker/runtime-work', '--output-json', "
    "'{work_root}/worker/measurements.json']' returned non-zero exit status 1."
)
GPU_QUALIFICATION_ARTIFACT_KEYS: Final = (
    "cachet_source_tree_sha256",
    "input_bundle_sha256",
    "package_wheel_sha256",
    "patched_vllm_wheel_sha256",
    "runner_sha256",
    "runtime_lock_sha256",
)
GPU_QUALIFICATION_OUTPUT_FILENAME: Final = "gpu-job-result.json"
GPU_QUALIFICATION_SUBMIT_RECEIPT_RECORD_TYPE: Final = (
    "cachet.vllm_0271_gpu_submit_receipt.v1"
)
GPU_QUALIFICATION_SUBMISSION_REJECTION_RECORD_TYPE: Final = (
    "cachet.vllm_0271_gpu_qualification_submission_rejection.v1"
)
GPU_QUALIFICATION_FAILED_ATTEMPT_RECONCILIATION_RECORD_TYPE: Final = (
    "cachet.gpu_qualification_failed_attempt_reconciliation.v1"
)
GPU_QUALIFICATION_FAILED_ATTEMPT_RECONCILIATION_V2_RECORD_TYPE: Final = (
    "cachet.gpu_qualification_failed_attempt_reconciliation.v2"
)
GPU_QUALIFICATION_FAILED_ATTEMPT_RECONCILIATION_V2_SCHEMA_VERSION: Final = 2
GPU_QUALIFICATION_FAILED_ATTEMPT_RECONCILIATION_REASON: Final = (
    "qualification payload used NONE access mode and could not resolve Unity Catalog "
    "Volume bootstrap; remaining pending jobs canceled after first failures"
)
GPU_QUALIFICATION_LOCAL_WORK_ROOT: Final = "/local_disk0/cachet-vllm-0271-qualification"
_QUALIFICATION_PHASE_LEASE_FILENAME: Final = "phase-lease.json"
_QUALIFICATION_BATCH_MARKER_FILENAME: Final = "batch-reserved.json"
_QUALIFICATION_PHASE_LEASE_RECORD_TYPE: Final = (
    "cachet.vllm_0271_gpu_qualification_phase_lease.v1"
)
_QUALIFICATION_BATCH_MARKER_RECORD_TYPE: Final = (
    "cachet.vllm_0271_gpu_qualification_batch_reserved.v1"
)
_QUALIFICATION_PREFLIGHT_PATH_DOMAIN: Final = (
    "cachet.vllm_0271_gpu_qualification_preflight_path.v1"
)
_QUALIFICATION_PREFLIGHT_BINDING_KEYS: Final = frozenset(
    {
        "completed_at_utc",
        "file_sha256",
        "path_sha256",
        "record_sha256",
    }
)
_QUALIFICATION_PLAN_PARAMETER_OPTION: Final = "--plan-record-zlib-base64"
_QUALIFICATION_PLAN_ZLIB_LEVEL: Final = 9
_QUALIFICATION_PLAN_MAX_CANONICAL_BYTES: Final = 64 * 1024
_QUALIFICATION_PLAN_MAX_ENCODED_CHARS: Final = (
    GPU_QUALIFICATION_DATABRICKS_PARAMETERS_MAX_BYTES
)
_DATABRICKS_CLUSTER_ID_MAX_UTF8_BYTES: Final = 256
_DATABRICKS_CLUSTER_ID_ENV_NAMES: Final = (
    "DATABRICKS_CLUSTER_ID",
    "DB_CLUSTER_ID",
)
_DATABRICKS_CLUSTER_ID_SPARK_CONF_KEY: Final = (
    "spark.databricks.clusterUsageTags.clusterId"
)
_FAILED_RUN_OUTPUT_LEGACY_KEYS: Final = frozenset(
    {"error", "error_trace", "metadata"}
)
_FAILED_RUN_OUTPUT_LOGGED_KEYS: Final = frozenset(
    {*_FAILED_RUN_OUTPUT_LEGACY_KEYS, "logs", "logs_truncated"}
)
_DATABRICKS_ANSI_RUNTIME_ERROR_HTML_PREFIX: Final = (
    "<span class='ansi-red-fg'>RuntimeError</span>"
)
_QUALIFICATION_SUBMISSION_REJECTION_KEYS: Final = frozenset(
    {
        "attempt_ids",
        "batch_marker_file_sha256",
        "closed_record_sha256",
        "failed_before_run_creation",
        "first_post_intent_file_sha256",
        "http_status",
        "observed_parameters_json_bytes",
        "plan_sha256",
        "reconciled_actual_gpu_seconds_per_attempt",
        "record_type",
        "rejected_at_utc",
        "remote_active_runs_observed",
        "schema_version",
        "server_parameters_json_limit_bytes",
        "server_reason",
        "submit_payloads_file_sha256",
    }
)

_SUBMIT_RECEIPT_KEYS: Final = frozenset(
    {
        "authorization_scope",
        "closed_record_sha256",
        "cloud_run_id",
        "job_id",
        "ledger_id",
        "output_json",
        "plan_sha256",
        "phase_batch_record_sha256",
        "record_type",
        "reservation_attempt_id",
        "schema_version",
        "submit_payload_sha256",
        "submit_response_sha256",
        "submitted_at_utc",
        "task_key",
    }
)

_INPUT_PROVENANCE_FILENAME: Final = "main-latency-inputs.provenance.json"
_INPUT_PROVENANCE_FIELDS: Final = frozenset(
    {
        "bundle_sha256",
        "closed_record_sha256",
        "outputs",
        "outputs_sha256",
        "protocol",
        "record_type",
        "schema_version",
        "sources",
        "sources_sha256",
    }
)
_INPUT_OUTPUT_FIELDS: Final = frozenset(
    {
        "byte_count",
        "dataset",
        "input_tokens_target",
        "jsonl_sha256",
        "record_count",
        "records",
        "records_sha256",
        "relative_path",
        "segment_count",
    }
)
_INPUT_DATASETS: Final = ("biography", "hotpotqa", "musique", "niah")
_INPUT_TARGET_SEGMENT_COUNTS: Final = ((8192, 4), (16384, 8), (32768, 16))
_INPUT_EXAMPLES_PER_DATASET: Final = 32
_INPUT_PROTOCOL: Final = {
    "datasets": list(_INPUT_DATASETS),
    "prompt_contract": {
        "prompt_template_version": "v2-final-answer",
        "system_prompt_position": "start",
    },
    "selection": {
        "domain": "cachet.main_latency.content_hash_selection.v1",
        "identity_reused_across_targets": True,
        "ordering": "sha256_domain_dataset_identity_and_source_record",
        "selected_examples_per_dataset": _INPUT_EXAMPLES_PER_DATASET,
    },
    "targets": [
        {"input_tokens_target": target, "segment_count": segment_count}
        for target, segment_count in _INPUT_TARGET_SEGMENT_COUNTS
    ],
    "tokenizer": {
        "add_special_tokens": False,
        "tokenizer_id": "Qwen/Qwen3-4B-Instruct-2507",
        "tokenizer_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
    },
    "transformation": {
        "document_context_tiling": "lossless_contiguous_unicode_codepoints",
        "id": "cachet.main_latency.lossless_context_tiling.v1",
        "padding": "balanced_exact_token_count_irrelevant_units",
        "vanilla_composition": (
            "concatenated_independent_segment_token_ids_equal_logical_cache_prefix"
        ),
    },
}

GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SCRIPT: Final = """from __future__ import annotations

import hashlib
import os
import subprocess
import sys


_KEYS = {
    "cachet_source_tree_sha256",
    "input_bundle_sha256",
    "package_wheel_sha256",
    "patched_vllm_wheel_sha256",
    "runner_sha256",
    "runtime_lock_sha256",
}


def _cluster_path(value: str) -> str:
    if value.startswith("dbfs:/Volumes/"):
        return "/" + value.removeprefix("dbfs:/").lstrip("/")
    if value.startswith("dbfs:/"):
        return "/dbfs/" + value.removeprefix("dbfs:/").lstrip("/")
    if value.startswith("file://"):
        from urllib.parse import unquote, urlsplit

        parsed = urlsplit(value)
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError("unsupported file URI authority")
        return unquote(parsed.path)
    return value


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pip_subprocess_environment() -> dict[str, str]:
    env = dict(os.environ)
    for variable_name in tuple(env):
        if variable_name.upper().startswith("PIP_"):
            env.pop(variable_name)
    for variable_name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        env.pop(variable_name, None)
    env.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return env


def _bootstrap(argv: list[str]) -> list[str]:
    package_uri = None
    pins = {}
    index = 0
    while index < len(argv):
        option = argv[index]
        if option in {"--package-wheel-uri", "--artifact-sha256"}:
            if index + 1 >= len(argv):
                raise ValueError(f"{option} requires a value")
            value = argv[index + 1]
            if option == "--package-wheel-uri":
                if package_uri is not None:
                    raise ValueError("duplicate --package-wheel-uri")
                package_uri = value
            else:
                key, separator, digest = value.partition("=")
                if not separator or key in pins:
                    raise ValueError("invalid or duplicate --artifact-sha256")
                pins[key] = digest
            index += 2
            continue
        index += 1
    if package_uri is None or set(pins) != _KEYS:
        raise ValueError("bootstrap requires the closed artifact pin set")
    for key, digest in pins.items():
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid SHA-256 for {key}")
    # Databricks executes ``spark_python_task`` files through a wrapper that
    # compiles the downloaded file without defining ``__file__``.  The code
    # object's filename is still the exact downloaded path (the wrapper opens
    # that same path before compiling it), so use it as the fail-closed
    # self-identity rather than depending on interpreter globals.
    runner_path = os.path.realpath(sys._getframe().f_code.co_filename)
    if _sha256(runner_path) != pins["runner_sha256"]:
        raise ValueError("GPU qualification bootstrap runner SHA-256 mismatch")
    package_path = _cluster_path(package_uri)
    if _sha256(package_path) != pins["package_wheel_sha256"]:
        raise ValueError("Cachet package wheel SHA-256 mismatch")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--no-deps",
            "--force-reinstall",
            package_path,
        ],
        check=True,
        env=_pip_subprocess_environment(),
    )
    return argv


if __name__ == "__main__":
    remaining = _bootstrap(sys.argv[1:])
    from document_kv_cache.gpu_qualification_databricks import main

    raise SystemExit(main(remaining))
"""
GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256: Final = sha256(
    GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SCRIPT.encode("utf-8")
).hexdigest()

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")
_DATABRICKS_RUN_ID_TEMPLATE = "{{job.run_id}}"
_LAUNCH_AUTHORIZATION_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class GPUQualificationLaunchAuthorization:
    """Non-record capability issued only by live collection or durable replay."""

    selection: GPUQualificationSelection
    plan_sha256: str
    evidence_closed_record_sha256: str
    evidence_file_sha256: str
    ledger_id: str
    ledger_path_sha256: str
    predecessor_prefix: DatabricksLedgerPrefix
    producer_batch_prefix: DatabricksLedgerPrefix
    ledger_prefix: DatabricksLedgerPrefix
    causal_closure_sha256: str

    def __init__(
        self,
        *,
        selection: GPUQualificationSelection,
        plan_sha256: str,
        evidence_closed_record_sha256: str,
        evidence_file_sha256: str,
        ledger_id: str,
        ledger_path_sha256: str,
        predecessor_prefix: DatabricksLedgerPrefix,
        producer_batch_prefix: DatabricksLedgerPrefix,
        ledger_prefix: DatabricksLedgerPrefix,
        causal_closure_sha256: str,
        _issuer: object,
    ) -> None:
        if _issuer is not _LAUNCH_AUTHORIZATION_ISSUER:
            raise TypeError(
                "GPU qualification launch authority must come from live collection "
                "or its durable replay boundary"
            )
        if not isinstance(selection, GPUQualificationSelection):
            raise TypeError("selection must be GPUQualificationSelection")
        object.__setattr__(self, "selection", selection)
        object.__setattr__(
            self, "plan_sha256", _required_sha256(plan_sha256, "plan_sha256")
        )
        object.__setattr__(
            self,
            "evidence_closed_record_sha256",
            _required_sha256(
                evidence_closed_record_sha256,
                "evidence_closed_record_sha256",
            ),
        )
        object.__setattr__(
            self,
            "evidence_file_sha256",
            _required_sha256(evidence_file_sha256, "evidence_file_sha256"),
        )
        object.__setattr__(self, "ledger_id", _non_empty_string(ledger_id, "ledger_id"))
        object.__setattr__(
            self,
            "ledger_path_sha256",
            _required_sha256(ledger_path_sha256, "ledger_path_sha256"),
        )
        if any(
            not isinstance(prefix, DatabricksLedgerPrefix)
            for prefix in (predecessor_prefix, producer_batch_prefix, ledger_prefix)
        ):
            raise TypeError("authorization ledger prefixes have the wrong type")
        if any(
            prefix.ledger_id != ledger_id
            for prefix in (predecessor_prefix, producer_batch_prefix, ledger_prefix)
        ):
            raise ValueError("authorization ledger prefix identity drift")
        object.__setattr__(self, "predecessor_prefix", predecessor_prefix)
        object.__setattr__(self, "producer_batch_prefix", producer_batch_prefix)
        object.__setattr__(self, "ledger_prefix", ledger_prefix)
        object.__setattr__(
            self,
            "causal_closure_sha256",
            _required_sha256(causal_closure_sha256, "causal_closure_sha256"),
        )


class GPUQualificationSentinelRunner(Protocol):
    """Internal callable that performs one frozen GPU sentinel."""

    def __call__(
        self,
        *,
        plan_record: Mapping[str, Any],
        planned_job: Mapping[str, Any],
        artifact_paths: Mapping[str, Path],
        work_dir: Path,
    ) -> Mapping[str, Any]: ...


def render_gpu_qualification_submit_payloads(
    plan_record: Mapping[str, Any],
    *,
    single_user_name: str,
    runner_uri: str,
    package_wheel_uri: str,
    patched_vllm_wheel_uri: str,
    artifact_uris: Mapping[str, str],
    output_root: str,
) -> tuple[dict[str, Any], ...]:
    """Render one exact, no-retry Databricks payload for every planned job.

    ``artifact_uris`` is a closed mapping keyed by the six names in
    :data:`GPU_QUALIFICATION_ARTIFACT_KEYS`.  The runner, Cachet package wheel,
    and patched vLLM wheel arguments must repeat their corresponding mapping
    values; this makes accidental URI substitution visible before submission.
    """

    plan, pins = _validated_plan_and_pins(plan_record)
    principal = _validated_single_user_name(single_user_name)
    uris = _validated_artifact_uris(
        artifact_uris,
        runner_uri=runner_uri,
        package_wheel_uri=package_wheel_uri,
        patched_vllm_wheel_uri=patched_vllm_wheel_uri,
    )
    normalized_output_root = _validated_output_root(output_root)
    plan_digest = _required_sha256(
        plan.get("closed_record_sha256"), "plan.closed_record_sha256"
    )
    jobs = _planned_jobs(plan)
    if not jobs or len(jobs) > GPU_QUALIFICATION_MAX_CLOUD_JOBS:
        raise ValueError("GPU qualification plan has an invalid cloud job count")

    encoded_plan = _encode_qualification_plan_parameter(
        canonical_gpu_qualification_json(plan)
    )
    payloads: list[dict[str, Any]] = []
    output_paths: set[str] = set()
    for planned_job in jobs:
        job_id = _safe_id(planned_job.get("job_id"), "planned job_id")
        hardware_id = _safe_id(planned_job.get("hardware_id"), "planned hardware_id")
        if (
            planned_job.get("attempt_number") != 0
            or planned_job.get("max_retries") != 0
        ):
            raise ValueError(f"planned job {job_id!r} is not attempt-zero-only")
        output_dir = _join_cluster_uri(normalized_output_root, plan_digest, job_id)
        output_json = _join_cluster_uri(output_dir, GPU_QUALIFICATION_OUTPUT_FILENAME)
        work_dir = str(_expected_local_work_dir(plan_digest, job_id))
        if output_json in output_paths:
            raise ValueError("GPU qualification output paths must be unique")
        output_paths.add(output_json)

        parameters = _runner_parameters(
            encoded_plan=encoded_plan,
            plan_digest=plan_digest,
            job_id=job_id,
            output_json=output_json,
            work_dir=work_dir,
            runner_uri=runner_uri,
            package_wheel_uri=package_wheel_uri,
            patched_vllm_wheel_uri=patched_vllm_wheel_uri,
            artifact_uris=uris,
            artifact_pins=pins,
            reservation_attempt_id=gpu_qualification_reservation_attempt_id(
                plan_digest, job_id
            ),
        )
        cluster = _qualification_cluster(
            hardware_id=hardware_id,
            single_user_name=principal,
            custom_tags={
                "campaign": _safe_tag_value(plan.get("campaign_id")),
                "job_id": job_id,
                "plan_sha256": plan_digest[:32],
            },
        )
        task = {
            "task_key": _task_key(job_id),
            "timeout_seconds": GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS,
            "max_retries": 0,
            "new_cluster": cluster,
            "spark_python_task": {
                "python_file": runner_uri,
                "parameters": parameters,
            },
        }
        attempt_id = gpu_qualification_reservation_attempt_id(plan_digest, job_id)
        payloads.append(
            bind_databricks_run_idempotency_token(
                {
                    "run_name": _run_name(plan.get("campaign_id"), job_id),
                    "timeout_seconds": GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS,
                    "tasks": [task],
                },
                attempt_id=attempt_id,
            )
        )
    return tuple(payloads)


def _qualification_batch_requests(
    plan: Mapping[str, Any],
    contracts: Sequence[Mapping[str, Any]],
) -> tuple[DatabricksRunAttemptReservationRequest, ...]:
    return tuple(
        DatabricksRunAttemptReservationRequest(
            attempt_id=str(contract["reservation_attempt_id"]),
            workload_id=(
                f"gpuq/{plan['closed_record_sha256'][:16]}/{contract['job_id']}"
            ),
            submit_payload=_required_mapping(contract.get("payload"), "payload"),
        )
        for contract in contracts
    )


def _qualification_contract_submit_payloads(
    contracts: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Return the immutable canonical snapshots later consumed by submission."""

    return tuple(
        _required_mapping(contract.get("payload"), "qualification contract payload")
        for contract in contracts
    )


def _validated_local_preflight_binding(
    path: str | Path,
    *,
    plan: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    config: DatabricksWorkspaceConfig,
    require_fresh_workspace: bool,
) -> tuple[dict[str, str], datetime, dict[str, Any]]:
    preflight_path = _validated_existing_regular_file(
        path, "local_preflight_evidence_path"
    )
    record = _read_canonical_json_object_file(
        preflight_path, "local preflight evidence"
    )
    plan_sha256 = _required_sha256(
        plan.get("closed_record_sha256"), "plan.closed_record_sha256"
    )
    completed_at = validate_local_preflight_evidence_record(
        record,
        plan_sha256=plan_sha256,
    )
    authoritative_record = _require_gpu_qualification_local_preflight_bundle(
        preflight_path,
        plan=plan,
        submit_payloads=submit_payloads,
        config=config,
        require_fresh_workspace=require_fresh_workspace,
    )
    if authoritative_record != record:
        raise ValueError("live local preflight bundle record differs")
    binding = {
        "completed_at_utc": _non_empty_string(
            record.get("completed_at_utc"), "local preflight completed_at_utc"
        ),
        "file_sha256": _file_sha256(preflight_path),
        "path_sha256": _canonical_json_sha256(
            {
                "domain": _QUALIFICATION_PREFLIGHT_PATH_DOMAIN,
                "path": str(preflight_path),
            }
        ),
        "record_sha256": _required_sha256(
            record.get("closed_record_sha256"),
            "local preflight closed_record_sha256",
        ),
    }
    return binding, completed_at, record


def _non_authorizing_local_preflight_binding(
    path: str | Path,
    *,
    plan: Mapping[str, Any],
) -> dict[str, str]:
    """Bind retained preflight bytes without replaying or granting authority."""

    preflight_path = _validated_existing_regular_file(
        path, "local_preflight_evidence_path"
    )
    record = _read_canonical_json_object_file(
        preflight_path, "local preflight evidence"
    )
    validate_local_preflight_evidence_record(
        record,
        plan_sha256=_required_sha256(
            plan.get("closed_record_sha256"), "plan.closed_record_sha256"
        ),
    )
    return {
        "completed_at_utc": _non_empty_string(
            record.get("completed_at_utc"), "local preflight completed_at_utc"
        ),
        "file_sha256": _file_sha256(preflight_path),
        "path_sha256": _canonical_json_sha256(
            {
                "domain": _QUALIFICATION_PREFLIGHT_PATH_DOMAIN,
                "path": str(preflight_path),
            }
        ),
        "record_sha256": _required_sha256(
            record.get("closed_record_sha256"),
            "local preflight closed_record_sha256",
        ),
    }


def _require_gpu_qualification_local_preflight_bundle(
    path: Path,
    *,
    plan: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    config: DatabricksWorkspaceConfig,
    require_fresh_workspace: bool,
) -> dict[str, Any]:
    # Local import avoids a module cycle: publication_freeze uses this module's
    # reviewed payload renderer while this authority boundary consumes its
    # live, non-injectable seven-check replay.
    from document_kv_cache.publication_freeze import (
        validate_gpu_qualification_local_preflight_bundle,
    )

    return validate_gpu_qualification_local_preflight_bundle(
        path,
        plan_record=plan,
        submit_payloads=submit_payloads,
        workspace_config=config,
        require_fresh_workspace=require_fresh_workspace,
    )


def _require_local_preflight_before_submission(
    completed_at: datetime,
    *,
    submission_boundary: datetime,
) -> None:
    boundary = _parse_utc_timestamp(
        _utc_timestamp(submission_boundary),
        "submission boundary",
    )
    if completed_at >= boundary:
        raise ValueError("local preflight must complete before qualification submission")


def _qualification_phase_lease_record(
    *,
    plan: Mapping[str, Any],
    ledger_path_sha256: str,
    predecessor_prefix: DatabricksLedgerPrefix,
    contracts: Sequence[Mapping[str, Any]],
    local_preflight_binding: Mapping[str, str],
) -> dict[str, Any]:
    if frozenset(local_preflight_binding) != _QUALIFICATION_PREFLIGHT_BINDING_KEYS:
        raise ValueError("local preflight binding has an open schema")
    record: dict[str, Any] = {
        "attempt_ids": [str(item["reservation_attempt_id"]) for item in contracts],
        "closed_record_sha256": "",
        "ledger_path_sha256": ledger_path_sha256,
        "local_preflight": dict(local_preflight_binding),
        "plan_sha256": plan["closed_record_sha256"],
        "predecessor_prefix": predecessor_prefix.to_record(),
        "record_type": _QUALIFICATION_PHASE_LEASE_RECORD_TYPE,
        "submit_payload_sha256": [
            str(item["submit_payload_sha256"]) for item in contracts
        ],
    }
    _seal_record(record)
    return record


def _qualification_batch_marker_record(
    *,
    lease_record: Mapping[str, Any],
    batch_authorization: DatabricksBatchReservationAuthorization,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "attempt_ids": list(batch_authorization.attempt_ids),
        "batch_prefix": batch_authorization.batch_prefix.to_record(),
        "closed_record_sha256": "",
        "ledger_path_sha256": batch_authorization.ledger_path_sha256,
        "phase_lease_record_sha256": lease_record["closed_record_sha256"],
        "predecessor_prefix": batch_authorization.predecessor_prefix.to_record(),
        "record_type": _QUALIFICATION_BATCH_MARKER_RECORD_TYPE,
        "submit_payload_sha256": list(batch_authorization.submit_payload_sha256s),
    }
    _seal_record(record)
    return record


def _replay_qualification_batch_marker(
    *,
    plan: Mapping[str, Any],
    contracts: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_binding: Mapping[str, str],
    require_existing_marker: bool = False,
) -> tuple[DatabricksBatchReservationAuthorization, dict[str, Any]]:
    if type(require_existing_marker) is not bool:
        raise TypeError("require_existing_marker must be a bool")
    root = _validated_existing_controller_evidence_root(
        submit_receipt_root, "submit_receipt_root"
    )
    lease = _read_canonical_json_object_file(
        root / _QUALIFICATION_PHASE_LEASE_FILENAME,
        "qualification phase lease",
    )
    expected_predecessor = databricks_ledger_prefix_from_record(
        _required_mapping(plan.get("campaign_ledger_prefix"), "campaign_ledger_prefix")
    )
    expected_lease = _qualification_phase_lease_record(
        plan=plan,
        ledger_path_sha256=_required_sha256(
            plan.get("campaign_ledger_path_sha256"),
            "campaign_ledger_path_sha256",
        ),
        predecessor_prefix=expected_predecessor,
        contracts=contracts,
        local_preflight_binding=local_preflight_binding,
    )
    if lease != expected_lease:
        raise ValueError("qualification phase lease differs from the frozen batch")
    live = read_databricks_cluster_hour_ledger_json(ledger_path)
    predecessor_ledger = DatabricksClusterHourLedger(
        ledger_id=live.ledger_id,
        cap_cluster_hours=live.cap_cluster_hours,
        reservations=live.reservations[: expected_predecessor.reservation_count],
        submission_receipts=live.submission_receipts[
            : expected_predecessor.submission_receipt_count
        ],
        terminal_actuals=live.terminal_actuals[
            : expected_predecessor.terminal_actual_count
        ],
    )
    if (
        databricks_ledger_prefix_at_counts(
            predecessor_ledger,
            reservation_count=len(predecessor_ledger.reservations),
            submission_receipt_count=len(predecessor_ledger.submission_receipts),
            terminal_actual_count=len(predecessor_ledger.terminal_actuals),
        )
        != expected_predecessor
    ):
        raise ValueError("qualification phase lease predecessor history drift")
    _require_qualification_ledger_admission(
        predecessor_ledger,
        proposed_task_count=len(contracts),
        proposed_reserved_cluster_hours=(
            len(contracts) * GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS / 3600.0
        ),
        label="qualification durable batch replay",
    )
    authorization = replay_databricks_run_attempt_batch_authorization_json(
        ledger_path,
        _qualification_batch_requests(plan, contracts),
        expected_predecessor_prefix=expected_predecessor,
    )
    require_databricks_publication_batch_admission(live, authorization)
    expected_marker = _qualification_batch_marker_record(
        lease_record=lease,
        batch_authorization=authorization,
    )
    marker_path = root / _QUALIFICATION_BATCH_MARKER_FILENAME
    if marker_path.exists() or marker_path.is_symlink():
        marker = _read_canonical_json_object_file(
            marker_path,
            "qualification batch marker",
        )
        if marker != expected_marker:
            raise ValueError("qualification batch marker differs from the ledger batch")
    elif require_existing_marker:
        raise ValueError("qualification batch marker must already exist")
    else:
        _write_canonical_exclusive(expected_marker, marker_path)
        marker = expected_marker
    return authorization, marker


def _require_existing_qualification_batch_marker(
    submit_receipt_root: str | Path,
) -> None:
    root = _validated_existing_controller_evidence_root(
        submit_receipt_root,
        "submit_receipt_root",
    )
    marker = root / _QUALIFICATION_BATCH_MARKER_FILENAME
    if not marker.is_file() or marker.is_symlink():
        raise ValueError("qualification batch marker must already exist")


def _require_failed_batch_is_current_ledger_suffix(
    ledger: DatabricksClusterHourLedger,
    *,
    batch_authorization: DatabricksBatchReservationAuthorization,
    contracts: Sequence[Mapping[str, Any]],
) -> None:
    """Require an exact latest batch and an ordered terminal-resume prefix."""

    require_databricks_ledger_prefix(ledger, batch_authorization.batch_prefix)
    predecessor = batch_authorization.predecessor_prefix
    receipt_stop = predecessor.submission_receipt_count + len(contracts)
    terminal_stop = predecessor.terminal_actual_count + len(contracts)
    if (
        len(ledger.reservations)
        != batch_authorization.batch_prefix.reservation_count
        or len(ledger.submission_receipts) != receipt_stop
        or len(ledger.terminal_actuals) > terminal_stop
    ):
        raise ValueError("failed qualification batch is not the current ledger suffix")
    ordered_attempts = tuple(
        str(contract["reservation_attempt_id"])
        for contract in sorted(contracts, key=lambda item: str(item["job_id"]))
    )
    observed = tuple(
        item.attempt_id
        for item in ledger.terminal_actuals[
            predecessor.terminal_actual_count : terminal_stop
        ]
    )
    if observed != ordered_attempts[: len(observed)]:
        raise ValueError("failed qualification terminal resume prefix is not canonical")


def _require_qualification_phase_ledger_closure(
    ledger: DatabricksClusterHourLedger,
    *,
    batch_authorization: DatabricksBatchReservationAuthorization,
    contracts: Sequence[Mapping[str, Any]],
) -> DatabricksLedgerPrefix:
    attempts = tuple(str(item["reservation_attempt_id"]) for item in contracts)
    digests = tuple(str(item["submit_payload_sha256"]) for item in contracts)
    if (
        batch_authorization.attempt_ids != attempts
        or batch_authorization.submit_payload_sha256s != digests
    ):
        raise ValueError("qualification batch authority member closure drift")
    predecessor = batch_authorization.predecessor_prefix
    batch_prefix = batch_authorization.batch_prefix
    require_databricks_ledger_prefix(ledger, batch_prefix)
    receipt_start = predecessor.submission_receipt_count
    receipt_stop = receipt_start + len(attempts)
    terminal_start = predecessor.terminal_actual_count
    terminal_stop = terminal_start + len(attempts)
    receipts = ledger.submission_receipts[receipt_start:receipt_stop]
    terminals = ledger.terminal_actuals[terminal_start:terminal_stop]
    if tuple(item.attempt_id for item in receipts) != attempts:
        raise ValueError("qualification ledger receipt slice is not the exact batch")
    expected_terminal_digests = dict(zip(attempts, digests, strict=True))
    observed_terminal_ids = tuple(item.attempt_id for item in terminals)
    if (
        len(terminals) != len(attempts)
        or len(set(observed_terminal_ids)) != len(attempts)
        or set(observed_terminal_ids) != set(attempts)
    ):
        raise ValueError("qualification ledger terminal slice is not the exact batch")
    if tuple(item.submit_payload_sha256 for item in receipts) != digests:
        raise ValueError("qualification ledger receipt payload slice drift")
    if any(
        item.submit_payload_sha256 != expected_terminal_digests[item.attempt_id]
        for item in terminals
    ):
        raise ValueError("qualification ledger terminal payload slice drift")
    if any(attempt_id not in ledger.closed_attempt_ids for attempt_id in attempts):
        raise ValueError("qualification ledger still has an active batch member")
    receipt_prefix = databricks_ledger_prefix_at_counts(
        ledger,
        reservation_count=batch_prefix.reservation_count,
        submission_receipt_count=receipt_stop,
        terminal_actual_count=terminal_start,
    )
    terminal_prefix = databricks_ledger_prefix_at_counts(
        ledger,
        reservation_count=batch_prefix.reservation_count,
        submission_receipt_count=receipt_stop,
        terminal_actual_count=terminal_stop,
    )
    if (
        receipt_prefix.reservation_count != batch_prefix.reservation_count
        or terminal_prefix.reservation_count != batch_prefix.reservation_count
    ):
        raise RuntimeError("qualification historical prefix reconstruction drift")
    return terminal_prefix


def _qualification_submit_receipt_record(
    *,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    ledger: DatabricksClusterHourLedger,
    phase_batch_record_sha256: str,
    submitted_at_utc: str,
) -> dict[str, Any]:
    attempt_id = str(contract["reservation_attempt_id"])
    ledger_receipt = next(
        item for item in ledger.submission_receipts if item.attempt_id == attempt_id
    )
    receipt: dict[str, Any] = {
        "authorization_scope": (
            "submission_identity_only_requires_direct_terminal_collection"
        ),
        "closed_record_sha256": "",
        "cloud_run_id": ledger_receipt.run_id,
        "job_id": contract["job_id"],
        "ledger_id": ledger.ledger_id,
        "output_json": contract["output_json"],
        "phase_batch_record_sha256": phase_batch_record_sha256,
        "plan_sha256": plan["closed_record_sha256"],
        "record_type": GPU_QUALIFICATION_SUBMIT_RECEIPT_RECORD_TYPE,
        "reservation_attempt_id": attempt_id,
        "schema_version": GPU_QUALIFICATION_SCHEMA_VERSION,
        "submit_payload_sha256": contract["submit_payload_sha256"],
        "submit_response_sha256": ledger_receipt.submit_response_sha256,
        "submitted_at_utc": submitted_at_utc,
        "task_key": contract["task_key"],
    }
    _seal_record(receipt)
    return receipt


def _qualification_post_intent_record(
    *,
    contract: Mapping[str, Any],
    batch_authorization: DatabricksBatchReservationAuthorization,
    phase_batch_record_sha256: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "attempt_id": contract["reservation_attempt_id"],
        "batch_prefix": batch_authorization.batch_prefix.to_record(),
        "closed_record_sha256": "",
        "job_id": contract["job_id"],
        "phase_batch_record_sha256": phase_batch_record_sha256,
        "state": "post_may_be_ambiguous_if_no_ledger_receipt",
        "submit_payload_sha256": contract["submit_payload_sha256"],
    }
    _seal_record(record)
    return record


def submit_gpu_qualification_jobs(
    config: DatabricksWorkspaceConfig,
    *,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    opener: DatabricksURLOpener | None = None,
    now: Callable[[], datetime] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Reserve, submit, and durably receipt-bind the exact fourteen jobs."""

    plan, _pins = _validated_plan_and_pins(plan_record)
    contracts = _validated_qualification_payloads(plan, submit_payloads)
    clock = now or _utc_now
    local_preflight_binding, preflight_completed_at, _preflight_record = (
        _validated_local_preflight_binding(
            local_preflight_evidence_path,
            plan=plan,
            submit_payloads=_qualification_contract_submit_payloads(contracts),
            config=config,
            require_fresh_workspace=True,
        )
    )
    _require_local_preflight_before_submission(
        preflight_completed_at,
        submission_boundary=clock(),
    )
    initial_ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    if databricks_ledger_path_sha256(ledger_path) != plan.get(
        "campaign_ledger_path_sha256"
    ):
        raise ValueError("qualification ledger path differs from the campaign plan")
    if initial_ledger.ledger_id != plan["campaign_ledger_id"]:
        raise ValueError("qualification ledger differs from the campaign plan")
    campaign_ledger_prefix = databricks_ledger_prefix_from_record(
        _required_mapping(plan.get("campaign_ledger_prefix"), "campaign_ledger_prefix")
    )
    _require_qualification_ledger_admission(
        initial_ledger,
        proposed_task_count=len(contracts),
        proposed_reserved_cluster_hours=(
            len(contracts) * GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS / 3600.0
        ),
        label="qualification launch",
    )
    require_databricks_ledger_prefix(initial_ledger, campaign_ledger_prefix)
    if initial_ledger.terminal_actual_cluster_hours != plan.get(
        "campaign_opening_terminal_gpu_hours"
    ):
        raise ValueError("qualification ledger opening terminal balance drift")
    requests = _qualification_batch_requests(plan, contracts)

    def validate_batch(
        live: DatabricksClusterHourLedger,
        reservations: tuple[DatabricksClusterHourReservation, ...],
        snapshots: tuple[Mapping[str, Any], ...],
    ) -> None:
        if databricks_ledger_path_sha256(ledger_path) != plan.get(
            "campaign_ledger_path_sha256"
        ):
            raise ValueError("qualification ledger path differs from the campaign plan")
        if live.ledger_id != plan["campaign_ledger_id"]:
            raise ValueError("qualification ledger differs from the campaign plan")
        if live.terminal_actual_cluster_hours != plan.get(
            "campaign_opening_terminal_gpu_hours"
        ):
            raise ValueError("qualification ledger opening terminal balance drift")
        require_databricks_ledger_prefix(live, campaign_ledger_prefix)
        if len(reservations) != len(contracts) or len(snapshots) != len(contracts):
            raise ValueError("qualification batch is not the exact fourteen jobs")
        for contract, reservation, snapshot in zip(
            contracts, reservations, snapshots, strict=True
        ):
            if (
                reservation.attempt_id != contract["reservation_attempt_id"]
                or reservation.submit_payload_sha256
                != contract["submit_payload_sha256"]
                or canonical_gpu_qualification_json(snapshot)
                != canonical_gpu_qualification_json(
                    _required_mapping(contract.get("payload"), "payload")
                )
            ):
                raise ValueError("qualification batch reservation changed a payload")
        _require_qualification_ledger_admission(
            live,
            proposed_task_count=sum(
                len(item.task_timeout_seconds) for item in reservations
            ),
            proposed_reserved_cluster_hours=sum(
                item.reserved_cluster_hours for item in reservations
            ),
            label="qualification batch reservation",
        )

    receipt_root = _create_fresh_controller_evidence_root(submit_receipt_root)
    lease_record = _qualification_phase_lease_record(
        plan=plan,
        ledger_path_sha256=_required_sha256(
            plan.get("campaign_ledger_path_sha256"),
            "campaign_ledger_path_sha256",
        ),
        predecessor_prefix=campaign_ledger_prefix,
        contracts=contracts,
        local_preflight_binding=local_preflight_binding,
    )
    lease_path = receipt_root / _QUALIFICATION_PHASE_LEASE_FILENAME
    _write_canonical_exclusive(lease_record, lease_path)
    try:
        _batch_ledger, batch_authorization = (
            reserve_databricks_run_attempt_batch_authorized_json(
                ledger_path,
                requests,
                expected_predecessor_prefix=campaign_ledger_prefix,
                batch_validator=validate_batch,
            )
        )
    except BaseException:
        if lease_path.is_file() and not lease_path.is_symlink():
            lease_path.unlink()
            _fsync_directory(receipt_root)
        if receipt_root.is_dir() and not any(receipt_root.iterdir()):
            receipt_root.rmdir()
            _fsync_directory(receipt_root.parent)
        raise
    batch_marker = _qualification_batch_marker_record(
        lease_record=lease_record,
        batch_authorization=batch_authorization,
    )
    batch_marker_path = receipt_root / _QUALIFICATION_BATCH_MARKER_FILENAME
    _write_canonical_exclusive(batch_marker, batch_marker_path)
    resolved_opener = (
        cast(DatabricksURLOpener, urllib.request.urlopen) if opener is None else opener
    )
    receipts: list[dict[str, Any]] = []
    for contract in contracts:
        attempt_id = str(contract["reservation_attempt_id"])
        payload = _required_mapping(contract.get("payload"), "payload")
        intent_path = receipt_root / f"{contract['job_id']}.post-intent"
        intent = _qualification_post_intent_record(
            contract=contract,
            batch_authorization=batch_authorization,
            phase_batch_record_sha256=str(batch_marker["closed_record_sha256"]),
        )
        _write_canonical_exclusive(intent, intent_path)
        response = submit_pre_reserved_databricks_run(
            config,
            payload,
            ledger_path=ledger_path,
            attempt_id=attempt_id,
            batch_authorization=batch_authorization,
            opener=resolved_opener,
        )
        ledger = record_databricks_run_submission_receipt_json(
            ledger_path,
            attempt_id=attempt_id,
            submit_response=response,
        )
        submitted_at = _utc_timestamp(clock())
        receipt = _qualification_submit_receipt_record(
            plan=plan,
            contract=contract,
            ledger=ledger,
            phase_batch_record_sha256=str(batch_marker["closed_record_sha256"]),
            submitted_at_utc=submitted_at,
        )
        _write_canonical_exclusive(receipt, receipt_root / f"{contract['job_id']}.json")
        intent_path.unlink()
        _fsync_directory(receipt_root)
        receipts.append(receipt)
    return tuple(receipts)


def resume_gpu_qualification_job_submissions(
    config: DatabricksWorkspaceConfig,
    *,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    opener: DatabricksURLOpener | None = None,
    now: Callable[[], datetime] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Resume the exact durable fourteen-job phase after a controller restart."""

    plan, _pins = _validated_plan_and_pins(plan_record)
    contracts = _validated_qualification_payloads(plan, submit_payloads)
    clock = now or _utc_now
    local_preflight_binding, preflight_completed_at, _preflight_record = (
        _validated_local_preflight_binding(
            local_preflight_evidence_path,
            plan=plan,
            submit_payloads=_qualification_contract_submit_payloads(contracts),
            config=config,
            require_fresh_workspace=False,
        )
    )
    _require_local_preflight_before_submission(
        preflight_completed_at,
        submission_boundary=clock(),
    )
    batch_authorization, batch_marker = _replay_qualification_batch_marker(
        plan=plan,
        contracts=contracts,
        ledger_path=ledger_path,
        submit_receipt_root=submit_receipt_root,
        local_preflight_binding=local_preflight_binding,
    )
    root = _validated_existing_controller_evidence_root(
        submit_receipt_root, "submit_receipt_root"
    )
    receipts: list[dict[str, Any]] = []
    batch_marker_sha256 = str(batch_marker["closed_record_sha256"])
    for contract in contracts:
        job_id = str(contract["job_id"])
        attempt_id = str(contract["reservation_attempt_id"])
        payload = _required_mapping(contract.get("payload"), "payload")
        receipt_path = root / f"{job_id}.json"
        intent_path = root / f"{job_id}.post-intent"
        if receipt_path.exists() or receipt_path.is_symlink():
            ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
            receipt = _read_canonical_json_object_file(
                receipt_path, f"submit receipt {job_id}"
            )
            _validate_submit_receipt(
                receipt,
                contract=contract,
                plan=plan,
                ledger=ledger,
                phase_batch_record_sha256=batch_marker_sha256,
            )
            if intent_path.is_file() and not intent_path.is_symlink():
                intent_path.unlink()
                _fsync_directory(root)
            receipts.append(receipt)
            continue
        expected_intent = _qualification_post_intent_record(
            contract=contract,
            batch_authorization=batch_authorization,
            phase_batch_record_sha256=batch_marker_sha256,
        )
        if intent_path.exists() or intent_path.is_symlink():
            observed_intent = _read_canonical_json_object_file(
                intent_path, f"post intent {job_id}"
            )
            if observed_intent != expected_intent:
                raise ValueError(f"qualification post intent {job_id!r} drift")
        else:
            _write_canonical_exclusive(expected_intent, intent_path)
        resume_pre_reserved_databricks_run(
            config,
            payload,
            ledger_path=ledger_path,
            attempt_id=attempt_id,
            batch_authorization=batch_authorization,
            opener=opener,
        )
        ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
        receipt = _qualification_submit_receipt_record(
            plan=plan,
            contract=contract,
            ledger=ledger,
            phase_batch_record_sha256=batch_marker_sha256,
            submitted_at_utc=_utc_timestamp(clock()),
        )
        try:
            _write_canonical_exclusive(receipt, receipt_path)
        except FileExistsError:
            observed_receipt = _read_canonical_json_object_file(
                receipt_path, f"submit receipt {job_id}"
            )
            _validate_submit_receipt(
                observed_receipt,
                contract=contract,
                plan=plan,
                ledger=ledger,
                phase_batch_record_sha256=batch_marker_sha256,
            )
            receipt = observed_receipt
        if intent_path.is_file() and not intent_path.is_symlink():
            intent_path.unlink()
            _fsync_directory(root)
        receipts.append(receipt)
    expected_names = {
        _QUALIFICATION_PHASE_LEASE_FILENAME,
        _QUALIFICATION_BATCH_MARKER_FILENAME,
        *(f"{item['job_id']}.json" for item in contracts),
    }
    if {item.name for item in root.iterdir()} != expected_names:
        raise ValueError("resumed qualification receipt directory is not closed")
    return tuple(receipts)


def _require_qualification_ledger_admission(
    ledger: DatabricksClusterHourLedger,
    *,
    proposed_task_count: int,
    proposed_reserved_cluster_hours: float,
    label: str,
) -> None:
    if ledger.cap_cluster_hours != MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS:
        raise ValueError(f"{label} requires the migrated 1024-hour campaign ledger")
    if (
        ledger.active_reserved_task_count + proposed_task_count
        > PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS
    ):
        raise ValueError(f"{label} exceeds the global 16-job concurrency cap")
    if (
        ledger.active_reserved_cluster_hours + proposed_reserved_cluster_hours
        > MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS
    ):
        raise ValueError(f"{label} exceeds the 900-hour active reservation cap")
    if (
        ledger.accounted_cluster_hours
        + proposed_reserved_cluster_hours
        + PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS
        > MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
    ):
        raise ValueError(f"{label} would consume the 124-hour campaign headroom")


def _validated_qualification_payloads(
    plan: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    *,
    require_legacy_uc_broken_security_shape: bool = False,
) -> tuple[dict[str, Any], ...]:
    if isinstance(submit_payloads, (str, bytes, bytearray)) or not isinstance(
        submit_payloads, Sequence
    ):
        raise TypeError("submit_payloads must be a sequence")
    jobs = _planned_jobs(plan)
    if len(submit_payloads) != len(jobs):
        raise ValueError(
            "qualification submission requires the exact planned job closure"
        )
    if type(require_legacy_uc_broken_security_shape) is not bool:
        raise TypeError("require_legacy_uc_broken_security_shape must be a bool")
    single_user_name = (
        None
        if require_legacy_uc_broken_security_shape
        else _qualification_single_user_name_from_payloads(submit_payloads)
    )
    pins = pins_from_plan_record(plan)
    plan_digest = _required_sha256(
        plan.get("closed_record_sha256"), "plan.closed_record_sha256"
    )
    encoded_plan = _encode_qualification_plan_parameter(
        canonical_gpu_qualification_json(plan)
    )
    contracts: list[dict[str, Any]] = []
    for planned_job, raw_payload in zip(jobs, submit_payloads, strict=True):
        payload = _json_object(raw_payload, "qualification submit payload")
        if set(payload) != {
            "idempotency_token",
            "run_name",
            "tasks",
            "timeout_seconds",
        }:
            raise ValueError("qualification submit payload has an open schema")
        job_id = _safe_id(planned_job.get("job_id"), "planned job_id")
        if payload.get("run_name") != _run_name(plan.get("campaign_id"), job_id):
            raise ValueError("qualification run_name does not match the plan")
        if (
            payload.get("timeout_seconds")
            != GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS
        ):
            raise ValueError("qualification run timeout differs")
        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, list) or len(raw_tasks) != 1:
            raise ValueError("qualification payload must contain exactly one task")
        task = _required_mapping(raw_tasks[0], "qualification task")
        if set(task) != {
            "max_retries",
            "new_cluster",
            "spark_python_task",
            "task_key",
            "timeout_seconds",
        }:
            raise ValueError("qualification task has an open schema")
        task_key = _task_key(job_id)
        if (
            task.get("task_key") != task_key
            or task.get("max_retries") != 0
            or task.get("timeout_seconds")
            != GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS
        ):
            raise ValueError("qualification task retry/timeout identity differs")
        hardware_id = _safe_id(planned_job.get("hardware_id"), "hardware_id")
        custom_tags = {
            "campaign": _safe_tag_value(plan.get("campaign_id")),
            "job_id": job_id,
            "plan_sha256": plan_digest[:32],
        }
        expected_cluster = (
            _legacy_uc_broken_qualification_cluster(
                hardware_id=hardware_id,
                custom_tags=custom_tags,
            )
            if require_legacy_uc_broken_security_shape
            else _qualification_cluster(
                hardware_id=hardware_id,
                single_user_name=cast(str, single_user_name),
                custom_tags=custom_tags,
            )
        )
        if task.get("new_cluster") != expected_cluster:
            raise ValueError("qualification cluster specification differs")
        python_task = _required_mapping(
            task.get("spark_python_task"), "spark_python_task"
        )
        if set(python_task) != {"parameters", "python_file"}:
            raise ValueError("qualification spark_python_task has an open schema")
        parameters = python_task.get("parameters")
        if not isinstance(parameters, list) or any(
            not isinstance(item, str) for item in parameters
        ):
            raise ValueError("qualification parameters must be strings")
        _require_qualification_parameters_size(parameters)
        runner_uri = _one_parameter(parameters, "--runner-uri")
        package_wheel_uri = _one_parameter(parameters, "--package-wheel-uri")
        patched_wheel_uri = _one_parameter(parameters, "--patched-vllm-wheel-uri")
        artifact_uris = _parse_key_value_args(
            _all_parameters(parameters, "--artifact-uri"),
            option_name="--artifact-uri",
        )
        validated_uris = _validated_artifact_uris(
            artifact_uris,
            runner_uri=runner_uri,
            package_wheel_uri=package_wheel_uri,
            patched_vllm_wheel_uri=patched_wheel_uri,
        )
        output_json = _validated_result_output_json(
            _one_parameter(parameters, "--output-json"),
            plan_digest=plan_digest,
            job_id=job_id,
        )
        work_dir = str(_expected_local_work_dir(plan_digest, job_id))
        attempt_id = gpu_qualification_reservation_attempt_id(plan_digest, job_id)
        require_databricks_run_idempotency_token(payload, attempt_id=attempt_id)
        expected_parameters = _runner_parameters(
            encoded_plan=encoded_plan,
            plan_digest=plan_digest,
            job_id=job_id,
            output_json=output_json,
            work_dir=work_dir,
            runner_uri=runner_uri,
            package_wheel_uri=package_wheel_uri,
            patched_vllm_wheel_uri=patched_wheel_uri,
            artifact_uris=validated_uris,
            artifact_pins=pins,
            reservation_attempt_id=attempt_id,
        )
        if (
            parameters != expected_parameters
            or python_task.get("python_file") != runner_uri
        ):
            raise ValueError("qualification task parameters differ from the renderer")
        snapshot, canonical_payload = canonical_databricks_submit_payload_snapshot(
            payload
        )
        contracts.append(
            {
                "job_id": job_id,
                "output_json": output_json,
                "payload": snapshot,
                "reservation_attempt_id": attempt_id,
                "submit_payload_sha256": sha256(canonical_payload).hexdigest(),
                "task_key": task_key,
            }
        )
    return tuple(contracts)


def _validated_failed_capture_expected_errors_by_job(
    expected_errors_by_job: Mapping[str, str],
    *,
    contracts: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    if not isinstance(expected_errors_by_job, Mapping):
        raise TypeError("expected_errors_by_job must be a mapping")
    observed: dict[str, str] = {}
    for raw_job_id, raw_expected_error in expected_errors_by_job.items():
        job_id = _safe_id(raw_job_id, "expected_errors_by_job job ID")
        observed[job_id] = _non_empty_string(
            raw_expected_error,
            f"expected_errors_by_job[{job_id!r}]",
        )
    planned_job_ids = tuple(str(contract["job_id"]) for contract in contracts)
    if len(observed) != len(planned_job_ids) or set(observed) != set(
        planned_job_ids
    ):
        raise ValueError(
            "expected_errors_by_job must cover the exact planned job IDs"
        )
    return {job_id: observed[job_id] for job_id in planned_job_ids}


@dataclass(frozen=True, slots=True)
class _LiteralFailedCaptureErrorExpectation:
    error: str


@dataclass(frozen=True, slots=True)
class _DigestFailedCaptureErrorExpectation:
    error_sha256: str
    error_utf8_bytes: int


_FailedCaptureErrorExpectation = (
    _LiteralFailedCaptureErrorExpectation | _DigestFailedCaptureErrorExpectation
)


def _validated_failed_capture_error_digest_expectations(
    expected_error_sha256_by_job: Mapping[str, str],
    expected_error_utf8_bytes_by_job: Mapping[str, int],
    *,
    contracts: Sequence[Mapping[str, Any]],
) -> dict[str, _DigestFailedCaptureErrorExpectation]:
    expected_sha256 = _validated_reviewed_error_sha256_by_job(
        expected_error_sha256_by_job,
        contracts=contracts,
    )
    expected_utf8_bytes = _validated_reviewed_error_utf8_bytes_by_job(
        expected_error_utf8_bytes_by_job,
        contracts=contracts,
    )
    return {
        job_id: _DigestFailedCaptureErrorExpectation(
            error_sha256=expected_sha256[job_id],
            error_utf8_bytes=expected_utf8_bytes[job_id],
        )
        for job_id in expected_sha256
    }


def _failed_capture_expected_error(
    run_output: Mapping[str, Any],
    *,
    expectation: _FailedCaptureErrorExpectation,
) -> str:
    error = _non_empty_string(run_output.get("error"), "runs/get-output error")
    if isinstance(expectation, _LiteralFailedCaptureErrorExpectation):
        if error != expectation.error:
            raise ValueError(
                "failed runs/get-output error differs from the reviewed cause"
            )
        return expectation.error
    try:
        encoded_error = error.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("runs/get-output error must be valid UTF-8") from exc
    if sha256(encoded_error).hexdigest() != expectation.error_sha256:
        raise ValueError(
            "failed runs/get-output error SHA-256 differs from the reviewed digest"
        )
    if len(encoded_error) != expectation.error_utf8_bytes:
        raise ValueError(
            "failed runs/get-output error UTF-8 byte length differs from the "
            "reviewed length"
        )
    return error


def capture_gpu_qualification_failed_attempt_evidence_v2(
    config: DatabricksWorkspaceConfig,
    *,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    evidence_root: str | Path,
    failure_reason: str,
    expected_error: str,
) -> dict[str, Any]:
    """Capture one failed qualification batch through direct read-only APIs.

    The transport is intentionally package-owned and non-injectable.  Tests may
    monkeypatch the two package-owned read functions, but production callers
    cannot supply fabricated responses.  The only local mutation is an atomic,
    write-once evidence-directory publication after the complete closure has
    validated; the campaign ledger and Databricks workspace are never mutated.
    """

    reason = _non_empty_string(failure_reason, "failure_reason")
    error = _non_empty_string(expected_error, "expected_error")
    plan, _pins = _validated_historical_qualification_plan_and_pins(plan_record)
    contracts = _validated_qualification_payloads(plan, submit_payloads)
    return capture_gpu_qualification_failed_attempt_evidence_v2_by_job(
        config,
        plan_record=plan_record,
        submit_payloads=submit_payloads,
        ledger_path=ledger_path,
        submit_receipt_root=submit_receipt_root,
        local_preflight_evidence_path=local_preflight_evidence_path,
        evidence_root=evidence_root,
        failure_reason=reason,
        expected_errors_by_job={
            str(contract["job_id"]): error for contract in contracts
        },
    )


def capture_gpu_qualification_failed_attempt_evidence_v2_by_job(
    config: DatabricksWorkspaceConfig,
    *,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    evidence_root: str | Path,
    failure_reason: str,
    expected_errors_by_job: Mapping[str, str],
) -> dict[str, Any]:
    """Capture one failed qualification batch with one exact error per job.

    The transport is intentionally package-owned and non-injectable.  Expected
    errors must cover the exact planned job IDs before any read-only API call or
    evidence-directory publication can occur.
    """

    reason = _non_empty_string(failure_reason, "failure_reason")
    plan, _pins = _validated_historical_qualification_plan_and_pins(plan_record)
    contracts = _validated_qualification_payloads(plan, submit_payloads)
    expected_errors = _validated_failed_capture_expected_errors_by_job(
        expected_errors_by_job,
        contracts=contracts,
    )
    expectations: dict[str, _FailedCaptureErrorExpectation] = {
        job_id: _LiteralFailedCaptureErrorExpectation(error=error)
        for job_id, error in expected_errors.items()
    }
    return _capture_validated_gpu_qualification_failed_attempt_evidence_v2(
        config,
        plan=plan,
        contracts=contracts,
        ledger_path=ledger_path,
        submit_receipt_root=submit_receipt_root,
        local_preflight_evidence_path=local_preflight_evidence_path,
        evidence_root=evidence_root,
        failure_reason=reason,
        error_expectations=expectations,
    )


def capture_gpu_qualification_failed_attempt_evidence_v2_by_job_digest(
    config: DatabricksWorkspaceConfig,
    *,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    evidence_root: str | Path,
    failure_reason: str,
    expected_error_sha256_by_job: Mapping[str, str],
    expected_error_utf8_bytes_by_job: Mapping[str, int],
) -> dict[str, Any]:
    """Capture one failed batch using exact reviewed error digests per job.

    Both maps must cover the exact planned job set before any package-owned GET.
    The fetched raw error remains the manifest authority only after its SHA-256
    and UTF-8 byte length both match the reviewed pins.  This compact mode is
    otherwise identical to the literal-error capture path and does not accept
    a caller-supplied transport.
    """

    reason = _non_empty_string(failure_reason, "failure_reason")
    plan, _pins = _validated_historical_qualification_plan_and_pins(plan_record)
    contracts = _validated_qualification_payloads(plan, submit_payloads)
    expectations = _validated_failed_capture_error_digest_expectations(
        expected_error_sha256_by_job,
        expected_error_utf8_bytes_by_job,
        contracts=contracts,
    )
    return _capture_validated_gpu_qualification_failed_attempt_evidence_v2(
        config,
        plan=plan,
        contracts=contracts,
        ledger_path=ledger_path,
        submit_receipt_root=submit_receipt_root,
        local_preflight_evidence_path=local_preflight_evidence_path,
        evidence_root=evidence_root,
        failure_reason=reason,
        error_expectations=expectations,
    )


def _capture_validated_gpu_qualification_failed_attempt_evidence_v2(
    config: DatabricksWorkspaceConfig,
    *,
    plan: Mapping[str, Any],
    contracts: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    evidence_root: str | Path,
    failure_reason: str,
    error_expectations: Mapping[str, _FailedCaptureErrorExpectation],
) -> dict[str, Any]:
    """Capture an already-validated failed batch without mutating its ledger."""

    planned_job_ids = tuple(str(contract["job_id"]) for contract in contracts)
    if tuple(error_expectations) != planned_job_ids:
        raise ValueError("failed capture error expectation order differs")
    output_root = _validated_fresh_controller_evidence_root(evidence_root)
    local_preflight_binding = _non_authorizing_local_preflight_binding(
        local_preflight_evidence_path,
        plan=plan,
    )
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    if databricks_ledger_path_sha256(ledger_path) != plan.get(
        "campaign_ledger_path_sha256"
    ):
        raise ValueError("capture ledger path differs from the qualification plan")
    _require_existing_qualification_batch_marker(submit_receipt_root)
    batch_authorization, batch_marker = _replay_qualification_batch_marker(
        plan=plan,
        contracts=contracts,
        ledger_path=ledger_path,
        submit_receipt_root=submit_receipt_root,
        local_preflight_binding=local_preflight_binding,
        require_existing_marker=True,
    )
    submit_receipts = _load_submit_receipts(
        submit_receipt_root,
        contracts=contracts,
        plan=plan,
        ledger=ledger,
        phase_batch_record_sha256=str(batch_marker["closed_record_sha256"]),
    )
    _require_failed_batch_is_current_ledger_suffix(
        ledger,
        batch_authorization=batch_authorization,
        contracts=contracts,
    )

    parent_runs: list[dict[str, Any]] = []
    run_outputs: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for contract, submit_receipt in zip(contracts, submit_receipts, strict=True):
        run = get_databricks_run(config, str(submit_receipt["cloud_run_id"]))
        run_file_sha256 = _canonical_record_file_sha256(run)
        base_entry = _failed_attempt_reconciliation_entry(
            run,
            contract=contract,
            submit_receipt=submit_receipt,
            evidence_file_sha256=run_file_sha256,
        )
        task_run_id = str(base_entry["task_run_id"])
        run_output = get_databricks_run_output(config, task_run_id)
        expected_error = _failed_capture_expected_error(
            run_output,
            expectation=error_expectations[str(contract["job_id"])],
        )
        entry = _failed_attempt_reconciliation_v2_entry(
            run_output,
            run=run,
            base_entry=base_entry,
            contract=contract,
            expected_error=expected_error,
            evidence_file_sha256=_canonical_record_file_sha256(run_output),
        )
        parent_runs.append(run)
        run_outputs.append(run_output)
        entries.append(entry)

    predicted, terminal_prefix = _predicted_failed_batch_terminal_ledger(
        ledger,
        batch_authorization=batch_authorization,
        contracts=contracts,
        runs=parent_runs,
        entries=entries,
    )
    del predicted
    manifest: dict[str, Any] = {
        "closed_record_sha256": "",
        "entries": sorted(entries, key=lambda item: str(item["job_id"])),
        "ledger_lineage": {
            "predecessor_prefix": batch_authorization.predecessor_prefix.to_record(),
            "producer_batch_prefix": batch_authorization.batch_prefix.to_record(),
            "terminal_prefix": terminal_prefix.to_record(),
        },
        "plan_sha256": plan["closed_record_sha256"],
        "reason": failure_reason,
        "record_type": (
            GPU_QUALIFICATION_FAILED_ATTEMPT_RECONCILIATION_V2_RECORD_TYPE
        ),
        "schema_version": (
            GPU_QUALIFICATION_FAILED_ATTEMPT_RECONCILIATION_V2_SCHEMA_VERSION
        ),
    }
    _seal_record(manifest)
    _validate_failed_attempt_reconciliation_v2_manifest(
        manifest,
        plan=plan,
        batch_authorization=batch_authorization,
        expected_entries=entries,
        expected_reason=failure_reason,
        expected_terminal_prefix=terminal_prefix,
    )
    _publish_failed_attempt_evidence_atomic(
        output_root,
        contracts=contracts,
        parent_runs=parent_runs,
        run_outputs=run_outputs,
        manifest=manifest,
    )
    return manifest


def _validated_reviewed_error_sha256_by_job(
    expected_error_sha256_by_job: Mapping[str, str],
    *,
    contracts: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    if not isinstance(expected_error_sha256_by_job, Mapping):
        raise TypeError("expected_error_sha256_by_job must be a mapping")
    observed: dict[str, str] = {}
    for raw_job_id, raw_digest in expected_error_sha256_by_job.items():
        job_id = _safe_id(raw_job_id, "expected_error_sha256_by_job job ID")
        observed[job_id] = _required_sha256(
            raw_digest,
            f"expected_error_sha256_by_job[{job_id!r}]",
        )
    planned_job_ids = tuple(str(contract["job_id"]) for contract in contracts)
    if len(observed) != len(planned_job_ids) or set(observed) != set(
        planned_job_ids
    ):
        raise ValueError(
            "expected_error_sha256_by_job must cover the exact planned job IDs"
        )
    return {job_id: observed[job_id] for job_id in planned_job_ids}


def _validated_reviewed_error_utf8_bytes_by_job(
    expected_error_utf8_bytes_by_job: Mapping[str, int],
    *,
    contracts: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    if not isinstance(expected_error_utf8_bytes_by_job, Mapping):
        raise TypeError("expected_error_utf8_bytes_by_job must be a mapping")
    observed: dict[str, int] = {}
    for raw_job_id, raw_byte_count in expected_error_utf8_bytes_by_job.items():
        job_id = _safe_id(raw_job_id, "expected_error_utf8_bytes_by_job job ID")
        observed[job_id] = _positive_int(
            raw_byte_count,
            f"expected_error_utf8_bytes_by_job[{job_id!r}]",
        )
    planned_job_ids = tuple(str(contract["job_id"]) for contract in contracts)
    if len(observed) != len(planned_job_ids) or set(observed) != set(planned_job_ids):
        raise ValueError(
            "expected_error_utf8_bytes_by_job must cover the exact planned job IDs"
        )
    return {job_id: observed[job_id] for job_id in planned_job_ids}


def _normalize_runtime_lock_index_failure_error(
    error: str,
    *,
    plan_sha256: str,
    job_id: str,
) -> str:
    reviewed_error = _non_empty_string(error, "runtime lock index failure error")
    reviewed_plan_sha256 = _required_sha256(plan_sha256, "plan_sha256")
    if (
        reviewed_plan_sha256
        != GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_PLAN_SHA256
    ):
        raise ValueError("runtime lock index failure plan is not reviewed")
    reviewed_job_id = _safe_id(job_id, "job_id")
    runtime_python = (
        f"{GPU_QUALIFICATION_LOCAL_WORK_ROOT}/{reviewed_plan_sha256}/"
        f"{reviewed_job_id}/runtime/bin/python"
    )
    if reviewed_error.count(runtime_python) != 1:
        raise ValueError(
            "runtime lock index failure error must contain the exact planned "
            "runtime Python path once"
        )
    lock_paths = tuple(
        match.group(0)
        for match in _RUNTIME_LOCK_INDEX_FAILURE_LOCK_PATH_RE.finditer(
            reviewed_error
        )
    )
    if len(lock_paths) != 1:
        raise ValueError(
            "runtime lock index failure error must contain one canonical "
            "ephemeral UUIDv4 lock path"
        )
    normalized = reviewed_error.replace(
        runtime_python,
        "{runtime_python}",
    ).replace(
        lock_paths[0],
        "{runtime_lock}",
    )
    if normalized != _RUNTIME_LOCK_INDEX_FAILURE_NORMALIZED_ERROR:
        raise ValueError("runtime lock index failure argv grammar differs")
    if (
        sha256(normalized.encode("utf-8")).hexdigest()
        != GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_NORMALIZED_ERROR_SHA256
    ):
        raise RuntimeError("runtime lock index normalized error source pin drift")
    return normalized


def _validated_runtime_lock_index_failure_error(
    run_output: Mapping[str, Any],
    *,
    plan_sha256: str,
    job_id: str,
    expected_error_sha256: str,
) -> str:
    _validate_failed_run_output_schema(run_output)
    if set(run_output) != _FAILED_RUN_OUTPUT_LOGGED_KEYS:
        raise ValueError(
            "runtime lock index failure requires the exact logged output schema"
        )
    error = _non_empty_string(
        run_output.get("error"),
        "runtime lock index failure error",
    )
    reviewed_error_sha256 = _required_sha256(
        expected_error_sha256,
        "expected_error_sha256",
    )
    if sha256(error.encode("utf-8")).hexdigest() != reviewed_error_sha256:
        raise ValueError("runtime lock index failure raw error is not reviewed")
    _normalize_runtime_lock_index_failure_error(
        error,
        plan_sha256=plan_sha256,
        job_id=job_id,
    )
    if run_output.get("logs_truncated") is not False:
        raise ValueError("runtime lock index failure logs must be complete")
    logs = run_output.get("logs")
    if (
        type(logs) is not str
        or logs.count(GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_LOG_MARKER) != 1
    ):
        raise ValueError(
            "runtime lock index failure logs must contain the exact torch "
            "resolution marker once"
        )
    return error


def _normalize_site_packages_path_failure_error(
    error: str,
    *,
    plan_sha256: str,
    job_id: str,
) -> str:
    reviewed_error = _non_empty_string(error, "site-packages path failure error")
    reviewed_plan_sha256 = _required_sha256(plan_sha256, "plan_sha256")
    if (
        reviewed_plan_sha256
        != GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_PLAN_SHA256
    ):
        raise ValueError("site-packages path failure plan is not reviewed")
    reviewed_job_id = _safe_id(job_id, "job_id")
    invalid_site_packages = (
        f"{GPU_QUALIFICATION_LOCAL_WORK_ROOT}/{reviewed_plan_sha256}/"
        f"{reviewed_job_id}/runtime/local/lib/python3.11/dist-packages"
    )
    if reviewed_error.count(invalid_site_packages) != 1:
        raise ValueError(
            "site-packages path failure error must contain the exact planned "
            "invalid dist-packages path once"
        )
    normalized = reviewed_error.replace(
        invalid_site_packages,
        "{invalid_site_packages}",
    )
    if normalized != _SITE_PACKAGES_PATH_FAILURE_NORMALIZED_ERROR:
        raise ValueError("site-packages path failure error grammar differs")
    if (
        sha256(normalized.encode("utf-8")).hexdigest()
        != GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_NORMALIZED_ERROR_SHA256
    ):
        raise RuntimeError("site-packages path normalized error source pin drift")
    return normalized


def _validated_site_packages_path_failure_error(
    run_output: Mapping[str, Any],
    *,
    plan_sha256: str,
    job_id: str,
    expected_error_sha256: str,
) -> str:
    _validate_failed_run_output_schema(run_output)
    if set(run_output) != _FAILED_RUN_OUTPUT_LOGGED_KEYS:
        raise ValueError(
            "site-packages path failure requires the exact logged output schema"
        )
    error = _non_empty_string(
        run_output.get("error"),
        "site-packages path failure error",
    )
    reviewed_error_sha256 = _required_sha256(
        expected_error_sha256,
        "expected_error_sha256",
    )
    if sha256(error.encode("utf-8")).hexdigest() != reviewed_error_sha256:
        raise ValueError("site-packages path failure raw error is not reviewed")
    _normalize_site_packages_path_failure_error(
        error,
        plan_sha256=plan_sha256,
        job_id=job_id,
    )
    if run_output.get("logs_truncated") is not False:
        raise ValueError("site-packages path failure logs must be complete")
    logs = run_output.get("logs")
    if (
        type(logs) is not str
        or logs.count(
            GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_PIP_CHECK_LOG_MARKER
        )
        != 2
    ):
        raise ValueError(
            "site-packages path failure logs must prove exactly two successful "
            "pip checks"
        )
    worker_marker = GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_WORKER_MODULE_MARKER
    if worker_marker in logs:
        raise ValueError(
            "site-packages path failure logs must precede sentinel worker launch"
        )
    error_trace = _non_empty_string(
        run_output.get("error_trace"),
        "site-packages path failure error_trace",
    )
    normalized_trace = re.sub(r"\x1b\[[0-9;]*m", "", error_trace)
    if (
        normalized_trace.count(
            GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_FREEZER_TRACE_MARKER
        )
        != 1
        or worker_marker in normalized_trace
    ):
        raise ValueError(
            "site-packages path failure trace must prove failure in the read-only "
            "freezer before sentinel worker launch"
        )
    return error


def _normalize_runtime_observation_and_worker_subprocess_failure_error(
    error: str,
    *,
    plan_sha256: str,
    job_id: str,
) -> str:
    reviewed_error = _non_empty_string(
        error,
        "runtime observation and worker subprocess failure error",
    )
    reviewed_plan_sha256 = _required_sha256(plan_sha256, "plan_sha256")
    if (
        reviewed_plan_sha256
        != GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_PLAN_SHA256
    ):
        raise ValueError(
            "runtime observation and worker subprocess failure plan is not reviewed"
        )
    reviewed_job_id = _safe_id(job_id, "job_id")
    reviewed_job_ids = {
        pinned_job_id
        for pinned_job_id, _ in (
            GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_SHA256_BY_JOB
        )
    }
    if reviewed_job_id not in reviewed_job_ids:
        raise ValueError(
            "runtime observation and worker subprocess failure job is not reviewed"
        )
    work_root = (
        f"{GPU_QUALIFICATION_LOCAL_WORK_ROOT}/{reviewed_plan_sha256}/{reviewed_job_id}"
    )
    observer_failure = reviewed_job_id in (
        GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_PACKED_PAGE_ROUNDTRIP_JOB_IDS
    )
    expected_work_root_count = 1 if observer_failure else 6
    if reviewed_error.count(work_root) != expected_work_root_count:
        raise ValueError(
            "runtime observation and worker subprocess failure error must contain "
            "the exact planned work root the reviewed number of times"
        )
    normalized = reviewed_error.replace(work_root, "{work_root}")
    expected_normalized = (
        _RUNTIME_OBSERVATION_FAILURE_NORMALIZED_ERROR
        if observer_failure
        else _WORKER_SUBPROCESS_FAILURE_NORMALIZED_ERROR
    )
    if normalized != expected_normalized:
        raise ValueError(
            "runtime observation and worker subprocess failure error grammar differs"
        )
    expected_normalized_sha256 = (
        GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_NORMALIZED_OBSERVER_ERROR_SHA256
        if observer_failure
        else GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_NORMALIZED_WORKER_ERROR_SHA256
    )
    if sha256(normalized.encode("utf-8")).hexdigest() != expected_normalized_sha256:
        raise RuntimeError(
            "runtime observation and worker subprocess normalized error source pin drift"
        )
    return normalized


def _validated_runtime_observation_and_worker_subprocess_failure_error(
    run_output: Mapping[str, Any],
    *,
    run: Mapping[str, Any],
    plan_sha256: str,
    job_id: str,
    expected_error_sha256: str,
    expected_error_utf8_bytes: int,
) -> str:
    _validate_failed_run_output_schema(run_output)
    if set(run_output) != _FAILED_RUN_OUTPUT_LOGGED_KEYS:
        raise ValueError(
            "runtime observation and worker subprocess failure requires the exact "
            "logged output schema"
        )
    reviewed_job_id = _safe_id(job_id, "job_id")
    error = _non_empty_string(
        run_output.get("error"),
        "runtime observation and worker subprocess failure error",
    )
    try:
        encoded_error = error.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "runtime observation and worker subprocess failure error must be valid UTF-8"
        ) from exc
    reviewed_error_sha256 = _required_sha256(
        expected_error_sha256,
        "expected_error_sha256",
    )
    reviewed_error_utf8_bytes = _positive_int(
        expected_error_utf8_bytes,
        "expected_error_utf8_bytes",
    )
    if sha256(encoded_error).hexdigest() != reviewed_error_sha256:
        raise ValueError(
            "runtime observation and worker subprocess failure raw error is not reviewed"
        )
    if len(encoded_error) != reviewed_error_utf8_bytes:
        raise ValueError(
            "runtime observation and worker subprocess failure raw error UTF-8 byte "
            "count is not reviewed"
        )
    _normalize_runtime_observation_and_worker_subprocess_failure_error(
        error,
        plan_sha256=plan_sha256,
        job_id=reviewed_job_id,
    )
    if run_output.get("logs_truncated") is not False:
        raise ValueError(
            "runtime observation and worker subprocess failure logs must be complete"
        )
    logs = run_output.get("logs")
    if type(logs) is not str:
        raise ValueError(
            "runtime observation and worker subprocess failure logs must be an exact string"
        )
    expected_log_markers = (
        (
            GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_PIP_CHECK_LOG_MARKER,
            2,
            "two successful pip checks",
        ),
        (
            GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_VIRTUALENV_LOG_PREFIX,
            1,
            "one reviewed virtualenv creation prefix",
        ),
        (
            GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ENSUREPIP_LOG_ARGV,
            1,
            "one reviewed ensurepip argv",
        ),
        (
            GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_WORKER_MODULE_MARKER,
            0,
            "no surfaced sentinel-worker output",
        ),
    )
    for marker, expected_count, description in expected_log_markers:
        if logs.count(marker) != expected_count:
            raise ValueError(
                "runtime observation and worker subprocess failure logs must prove "
                f"exactly {description}"
            )

    error_trace = _non_empty_string(
        run_output.get("error_trace"),
        "runtime observation and worker subprocess failure error_trace",
    )
    normalized_trace = re.sub(r"\x1b\[[0-9;]*m", "", error_trace)
    worker_marker = GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_WORKER_MODULE_MARKER
    observer_marker = GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OBSERVER_TRACE_MARKER
    observer_failure = reviewed_job_id in (
        GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_PACKED_PAGE_ROUNDTRIP_JOB_IDS
    )
    if normalized_trace.count(error) != 1:
        raise ValueError(
            "runtime observation and worker subprocess failure trace must contain "
            "the exact reviewed error once"
        )
    if observer_failure:
        if (
            normalized_trace.count(observer_marker) != 2
            or any(
                normalized_trace.count(marker) != 1
                for marker in (
                    GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OBSERVER_TRACE_MARKERS
                )
            )
            or worker_marker in normalized_trace
            or "CalledProcessError" in normalized_trace
        ):
            raise ValueError(
                "runtime observation failure trace must prove the exact post-success "
                "observer call and symlink guard without a worker subprocess failure"
            )
    elif (
        normalized_trace.count(worker_marker) != 2
        or any(
            normalized_trace.count(marker) != 1
            for marker in (
                GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_WORKER_TRACE_MARKERS
            )
        )
        or observer_marker in normalized_trace
        or normalized_trace.count("CalledProcessError") != 3
    ):
        raise ValueError(
            "worker subprocess failure trace must prove the exact reviewed sentinel, "
            "freezer, captured checked subprocess and stdlib failure frames without "
            "final runtime observation"
        )

    raw_tasks = run.get("tasks")
    metadata = _required_mapping(run_output.get("metadata"), "run-output metadata")
    metadata_tasks = metadata.get("tasks")
    if (
        not isinstance(raw_tasks, list)
        or len(raw_tasks) != 1
        or not isinstance(raw_tasks[0], Mapping)
        or not isinstance(metadata_tasks, list)
        or len(metadata_tasks) != 1
        or not isinstance(metadata_tasks[0], Mapping)
    ):
        raise ValueError(
            "runtime observation and worker subprocess failure terminal task closure "
            "differs"
        )
    observed_task = raw_tasks[0]
    metadata_task = metadata_tasks[0]
    expected_states = (
        (run, "INTERNAL_ERROR", "FAILED"),
        (observed_task, "TERMINATED", "FAILED"),
        (metadata, "TERMINATED", "FAILED"),
        (metadata_task, "TERMINATED", "FAILED"),
    )
    for (
        terminal_record,
        expected_life_cycle_state,
        expected_result_state,
    ) in expected_states:
        state = _required_mapping(
            terminal_record.get("state"),
            "runtime observation and worker subprocess failure state",
        )
        status = _required_mapping(
            terminal_record.get("status"),
            "runtime observation and worker subprocess failure status",
        )
        if (
            state.get("life_cycle_state") != expected_life_cycle_state
            or state.get("result_state") != expected_result_state
            or status.get("state") != "TERMINATED"
        ):
            raise ValueError(
                "runtime observation and worker subprocess failure terminal states "
                "are not reviewed"
            )
    return error


def _mixed_sentinel_and_result_validation_failure_categories() -> dict[str, str]:
    """Return the source-reviewed disjoint category for every 694441 job."""

    groups = (
        (
            "version_mismatch",
            GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_VERSION_MISMATCH_JOB_IDS,
            2,
        ),
        (
            "unresolved_native",
            GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_UNRESOLVED_NATIVE_JOB_IDS,
            2,
        ),
        (
            "layout_conflict",
            GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_LAYOUT_CONFLICT_JOB_IDS,
            8,
        ),
        (
            "flashinfer",
            GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FLASHINFER_JOB_IDS,
            2,
        ),
    )
    categories: dict[str, str] = {}
    for category, job_ids, expected_count in groups:
        if (
            len(job_ids) != expected_count
            or len(set(job_ids)) != expected_count
            or tuple(job_ids) != tuple(sorted(job_ids))
        ):
            raise ValueError(
                "mixed sentinel/result-validation failure category job set differs"
            )
        for job_id in job_ids:
            reviewed_job_id = _safe_id(job_id, "reviewed mixed failure job_id")
            if reviewed_job_id in categories:
                raise ValueError(
                    "mixed sentinel/result-validation failure categories overlap"
                )
            categories[reviewed_job_id] = category
    sha256_job_ids = tuple(
        job_id
        for job_id, _ in (
            GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_ERROR_SHA256_BY_JOB
        )
    )
    byte_job_ids = tuple(
        job_id
        for job_id, _ in (
            GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_ERROR_UTF8_BYTES_BY_JOB
        )
    )
    if (
        len(categories) != 14
        or sha256_job_ids != tuple(sorted(categories))
        or byte_job_ids != sha256_job_ids
    ):
        raise ValueError(
            "mixed sentinel/result-validation failure job sets are not exhaustive"
        )
    return categories


def _require_mixed_sentinel_and_result_validation_terminal_topology(
    run: Mapping[str, Any],
    run_output: Mapping[str, Any],
) -> None:
    """Require the exact parent/task/metadata attempt-zero terminal topology."""

    raw_tasks = run.get("tasks")
    metadata = _required_mapping(run_output.get("metadata"), "run-output metadata")
    metadata_tasks = metadata.get("tasks")
    if (
        not isinstance(raw_tasks, list)
        or len(raw_tasks) != 1
        or not isinstance(raw_tasks[0], Mapping)
        or not isinstance(metadata_tasks, list)
        or len(metadata_tasks) != 1
        or not isinstance(metadata_tasks[0], Mapping)
    ):
        raise ValueError(
            "mixed sentinel/result-validation terminal task topology differs"
        )
    observed_task = raw_tasks[0]
    metadata_task = metadata_tasks[0]
    task_key = _safe_id(observed_task.get("task_key"), "failed task_key")
    parent_message = (
        f"Task {task_key} failed with message: Workload failed, see run output "
        "for details."
    )
    task_message = "Workload failed, see run output for details"
    expected_records = (
        (run, "INTERNAL_ERROR", parent_message),
        (observed_task, "TERMINATED", task_message),
        (metadata, "TERMINATED", task_message),
        (metadata_task, "TERMINATED", task_message),
    )
    for terminal_record, life_cycle_state, state_message in expected_records:
        if terminal_record.get("state") != {
            "life_cycle_state": life_cycle_state,
            "result_state": "FAILED",
            "state_message": state_message,
            "user_cancelled_or_timedout": False,
        }:
            raise ValueError(
                "mixed sentinel/result-validation terminal states are not reviewed"
            )
        status = _required_mapping(
            terminal_record.get("status"),
            "mixed sentinel/result-validation status",
        )
        termination_details = _required_mapping(
            status.get("termination_details"),
            "mixed sentinel/result-validation termination details",
        )
        if status.get("state") != "TERMINATED" or termination_details != {
            "code": "RUN_EXECUTION_ERROR",
            "message": state_message,
            "type": "CLIENT_ERROR",
        }:
            raise ValueError(
                "mixed sentinel/result-validation terminal statuses are not reviewed"
            )
    for terminal_record, label in (
        (run, "parent"),
        (observed_task, "task"),
        (metadata, "metadata"),
        (metadata_task, "metadata task"),
    ):
        if (
            terminal_record.get("repair_history") not in (None, [])
            or terminal_record.get("original_attempt_run_id") is not None
        ):
            raise ValueError(
                "mixed sentinel/result-validation failure has repair history"
            )
        if (terminal_record is observed_task or terminal_record is metadata_task) and (
            type(terminal_record.get("attempt_number")) is not int
            or terminal_record.get("attempt_number") != 0
        ):
            raise ValueError(
                f"mixed sentinel/result-validation {label} is not attempt zero"
            )


def _validated_mixed_sentinel_and_result_validation_failure_error(
    run_output: Mapping[str, Any],
    *,
    run: Mapping[str, Any],
    plan_sha256: str,
    job_id: str,
    expected_error_sha256: str,
    expected_error_utf8_bytes: int,
) -> str:
    """Validate one exact 694441 failure record without caller review authority."""

    _validate_failed_run_output_schema(run_output)
    if set(run_output) != _FAILED_RUN_OUTPUT_LOGGED_KEYS:
        raise ValueError(
            "mixed sentinel/result-validation failure requires the exact logged "
            "output schema"
        )
    if (
        _required_sha256(plan_sha256, "plan_sha256")
        != GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_PLAN_SHA256
    ):
        raise ValueError("mixed sentinel/result-validation failure plan is not reviewed")
    reviewed_job_id = _safe_id(job_id, "job_id")
    category = _mixed_sentinel_and_result_validation_failure_categories().get(
        reviewed_job_id
    )
    if category is None:
        raise ValueError("mixed sentinel/result-validation failure job is not reviewed")
    error = _non_empty_string(
        run_output.get("error"),
        "mixed sentinel/result-validation failure error",
    )
    try:
        encoded_error = error.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "mixed sentinel/result-validation failure error must be valid UTF-8"
        ) from exc
    if sha256(encoded_error).hexdigest() != _required_sha256(
        expected_error_sha256,
        "expected_error_sha256",
    ):
        raise ValueError(
            "mixed sentinel/result-validation failure raw error is not reviewed"
        )
    if len(encoded_error) != _positive_int(
        expected_error_utf8_bytes,
        "expected_error_utf8_bytes",
    ):
        raise ValueError(
            "mixed sentinel/result-validation failure raw error UTF-8 byte count "
            "is not reviewed"
        )
    if run_output.get("logs_truncated") is not False:
        raise ValueError(
            "mixed sentinel/result-validation failure logs must be complete"
        )
    logs = run_output.get("logs")
    if type(logs) is not str:
        raise ValueError(
            "mixed sentinel/result-validation failure logs must be an exact string"
        )
    expected_log_markers = (
        (
            GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_PIP_CHECK_LOG_MARKER,
            2,
        ),
        (
            GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_VIRTUALENV_LOG_PREFIX,
            1,
        ),
        (
            GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_ENSUREPIP_LOG_ARGV,
            1,
        ),
        (
            GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_WORKER_MODULE_MARKER,
            0,
        ),
    )
    if any(logs.count(marker) != count for marker, count in expected_log_markers):
        raise ValueError(
            "mixed sentinel/result-validation failure logs differ from the reviewed "
            "install and worker-output grammar"
        )
    work_root_prefix = f"{GPU_QUALIFICATION_LOCAL_WORK_ROOT}/"
    work_root = f"{work_root_prefix}{plan_sha256}/{reviewed_job_id}"
    if (
        logs.count(work_root_prefix) != 12
        or logs.count(work_root) != 12
    ):
        raise ValueError(
            "mixed sentinel/result-validation logs contain an unreviewed work path"
        )

    error_trace = _non_empty_string(
        run_output.get("error_trace"),
        "mixed sentinel/result-validation failure error_trace",
    )
    normalized_error = _normalized_failed_run_exception_text(error)
    normalized_trace = _normalized_failed_run_exception_text(error_trace)
    if normalized_trace.count(normalized_error) != 1:
        raise ValueError(
            "mixed sentinel/result-validation trace must contain the normalized "
            "reviewed error exactly once"
        )
    html_prefix = _DATABRICKS_ANSI_RUNTIME_ERROR_HTML_PREFIX
    worker_module = (
        GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_WORKER_MODULE_MARKER
    )
    if category == "version_mismatch":
        if (
            error
            != f"ValueError: job result {reviewed_job_id} vLLM version mismatch"
            or html_prefix in error
            or "<span" in error
            or any(
                normalized_trace.count(marker) != count
                for marker, count in (
                    GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_VERSION_TRACE_MARKERS
                )
            )
            or any(
                marker in normalized_trace
                for marker in (
                    worker_module,
                    "run_gpu_qualification_sentinel",
                    "_run_bounded_worker_process",
                    "_worker_process_failure",
                )
            )
        ):
            raise ValueError(
                "mixed result-validation failure grammar or trace differs"
            )
    else:
        if (
            normalized_error.count(work_root_prefix) == 0
            or normalized_error.count(work_root_prefix)
            != normalized_error.count(work_root)
            or normalized_trace.count(work_root_prefix)
            != normalized_trace.count(work_root)
        ):
            raise ValueError(
                "mixed sentinel/result-validation failure contains an unreviewed "
                "work path"
            )
        if (
            not error.startswith(f"{html_prefix}: ")
            or error.count(html_prefix) != 1
            or error.count("<span") != 1
            or error.count("</span>") != 1
            or html_prefix in normalized_error
            or "<span" in normalized_trace
            or any(
                normalized_trace.count(marker) != count
                for marker, count in (
                    GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_WORKER_TRACE_MARKERS
                )
            )
            or normalized_trace.count(worker_module) != 1
        ):
            raise ValueError(
                "mixed sentinel failure HTML projection or worker trace differs"
            )
        envelope = re.fullmatch(
            r"RuntimeError: GPU sentinel '([^']+)' worker exited with status 1; "
            r"stdout\(bytes=(0|[1-9][0-9]*),sha256=([0-9a-f]{64}),"
            r"truncated=(true|false),tail=(.+?)\); stderr\(bytes=([1-9][0-9]*),"
            r"sha256=([0-9a-f]{64}),truncated=(true|false),tail=(.+)\)",
            normalized_error,
            flags=re.DOTALL,
        )
        if envelope is None or envelope.group(1) != reviewed_job_id:
            raise ValueError("mixed sentinel failure stream envelope differs")
        (
            stdout_bytes,
            stdout_sha256,
            stdout_truncated,
            stdout_tail,
            _stderr_bytes,
            _stderr_sha256,
            stderr_truncated,
            stderr_tail,
        ) = envelope.groups()[1:]
        if (
            not stdout_tail.startswith("'")
            or not stdout_tail.endswith("'")
            or not stderr_tail.startswith("'")
            or not stderr_tail.endswith("'")
            or stderr_tail == "''"
        ):
            raise ValueError("mixed sentinel failure stream tail grammar differs")
        empty_sha256 = sha256(b"").hexdigest()
        if category == "flashinfer":
            expected_stream_shape = (
                stdout_bytes != "0"
                and stdout_truncated == "true"
                and stdout_tail != "''"
                and stderr_truncated == "true"
            )
        else:
            expected_stream_shape = (
                stdout_bytes == "0"
                and stdout_sha256 == empty_sha256
                and stdout_truncated == "false"
                and stdout_tail == "''"
                and stderr_truncated
                == ("false" if category == "unresolved_native" else "true")
            )
        if not expected_stream_shape:
            raise ValueError("mixed sentinel failure stream category differs")

        category_markers = {
            "unresolved_native": (
                GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_UNRESOLVED_NATIVE_ERROR_MARKERS
            ),
            "layout_conflict": (
                GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_LAYOUT_ERROR_MARKERS
            ),
            "flashinfer": (
                GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FLASHINFER_ERROR_MARKERS
            ),
        }
        reviewed_markers = category_markers[category]
        other_markers = {
            marker
            for other_category, markers in category_markers.items()
            if other_category != category
            for marker in markers
        }
        if (
            any(normalized_error.count(marker) != 1 for marker in reviewed_markers)
            or any(normalized_trace.count(marker) != 1 for marker in reviewed_markers)
            or any(marker in normalized_error for marker in other_markers)
            or any(marker in normalized_trace for marker in other_markers)
            or any(marker in logs for marker in reviewed_markers)
        ):
            raise ValueError(
                "mixed sentinel failure category-specific grammar differs"
            )

    _require_mixed_sentinel_and_result_validation_terminal_topology(run, run_output)
    return error


def _require_mixed_sentinel_and_result_validation_predicted_ledger(
    predicted: DatabricksClusterHourLedger,
    *,
    predecessor_terminal_count: int,
    terminal_prefix: DatabricksLedgerPrefix,
) -> None:
    """Pin the complete canonical ledger projection before the first append."""

    if terminal_prefix.prefix_sha256 != (
        GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_TERMINAL_PREFIX_SHA256
    ):
        raise ValueError("mixed failure predicted terminal prefix is not reviewed")
    expected_counts = (
        GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_RESERVATION_COUNT,
        GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_SUBMISSION_RECEIPT_COUNT,
        GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_TERMINAL_ACTUAL_COUNT,
    )
    if (
        (
            len(predicted.reservations),
            len(predicted.submission_receipts),
            len(predicted.terminal_actuals),
        )
        != expected_counts
        or predicted.active_reserved_task_count != 0
        or predicted.active_reserved_cluster_hours != 0.0
        or predicted.terminal_actual_cluster_hours
        != GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_TERMINAL_HOURS
        or predicted.accounted_cluster_hours
        != GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_TERMINAL_HOURS
        or predicted.remaining_cluster_hours
        != GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_REMAINING_HOURS
        or sum(
            actual.actual_cluster_duration_seconds
            for actual in predicted.terminal_actuals[predecessor_terminal_count:]
        )
        != GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_NEW_TERMINAL_SECONDS
    ):
        raise ValueError("mixed failure predicted ledger accounting is not reviewed")
    canonical_bytes = (
        json.dumps(
            databricks_cluster_hour_ledger_to_record(predicted),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if (
        len(canonical_bytes)
        != GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_LEDGER_BYTES
        or sha256(canonical_bytes).hexdigest()
        != GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_LEDGER_SHA256
    ):
        raise ValueError("mixed failure canonical predicted ledger is not reviewed")


def _failed_attempt_evidence_tree_binding(
    evidence_root: Path,
) -> tuple[int, int, str]:
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(evidence_root.iterdir(), key=lambda item: item.name):
        validated = _validated_existing_regular_file(
            path,
            "failed-attempt evidence tree file",
        )
        byte_count = validated.stat().st_size
        total_bytes += byte_count
        rows.append(
            {
                "byte_count": byte_count,
                "relative_path": validated.name,
                "sha256": _file_sha256(validated),
            }
        )
    return len(rows), total_bytes, _canonical_json_sha256(rows)


def _reconcile_reviewed_gpu_qualification_failed_attempt_evidence_v2(
    *,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    runs_get_evidence_root: str | Path,
    expected_plan_sha256: str,
    expected_runner_sha256: str,
    expected_manifest_closed_record_sha256: str,
    expected_manifest_file_sha256: str,
    expected_terminal_prefix_sha256: str,
    expected_failure_reason: str,
    expected_error: str | None,
    expected_run_output_keys: frozenset[str],
    expected_runtime_lock_index_error_sha256_by_job: Mapping[str, str]
    | None = None,
    expected_site_packages_path_error_sha256_by_job: Mapping[str, str]
    | None = None,
    expected_runtime_observation_and_worker_subprocess_error_sha256_by_job: Mapping[
        str, str
    ]
    | None = None,
    expected_runtime_observation_and_worker_subprocess_error_utf8_bytes_by_job: Mapping[
        str, int
    ]
    | None = None,
    expected_mixed_sentinel_and_result_validation_failure: bool = False,
    expected_evidence_tree_sha256: str | None = None,
    expected_evidence_tree_file_count: int | None = None,
    expected_evidence_tree_total_bytes: int | None = None,
) -> DatabricksClusterHourLedger:
    """Reconcile one source-reviewed v2 failure closure without authority.

    This generic core is private on purpose.  A public incident-specific wrapper
    must source-pin every ``expected_*`` value in reviewed code; accepting those
    values from an operational CLI would let an unreviewed manifest approve
    itself.  Every evidence and prefix check runs before the first ledger write.
    """

    reviewed_plan_sha256 = _required_sha256(
        expected_plan_sha256, "expected_plan_sha256"
    )
    reviewed_runner_sha256 = _required_sha256(
        expected_runner_sha256, "expected_runner_sha256"
    )
    reviewed_manifest_sha256 = _required_sha256(
        expected_manifest_closed_record_sha256,
        "expected_manifest_closed_record_sha256",
    )
    reviewed_manifest_file_sha256 = _required_sha256(
        expected_manifest_file_sha256,
        "expected_manifest_file_sha256",
    )
    reviewed_terminal_prefix_sha256 = _required_sha256(
        expected_terminal_prefix_sha256,
        "expected_terminal_prefix_sha256",
    )
    reason = _non_empty_string(expected_failure_reason, "expected_failure_reason")
    runtime_observation_and_worker_subprocess_error_pins = (
        expected_runtime_observation_and_worker_subprocess_error_sha256_by_job,
        expected_runtime_observation_and_worker_subprocess_error_utf8_bytes_by_job,
    )
    if any(
        value is None for value in runtime_observation_and_worker_subprocess_error_pins
    ) and any(
        value is not None
        for value in runtime_observation_and_worker_subprocess_error_pins
    ):
        raise ValueError(
            "reviewed runtime observation and worker subprocess error pins must "
            "include both SHA-256 and UTF-8 byte maps"
        )
    if type(expected_mixed_sentinel_and_result_validation_failure) is not bool:
        raise TypeError(
            "expected_mixed_sentinel_and_result_validation_failure must be a bool"
        )
    expected_error_modes = (
        expected_error is not None,
        expected_runtime_lock_index_error_sha256_by_job is not None,
        expected_site_packages_path_error_sha256_by_job is not None,
        all(
            value is not None
            for value in runtime_observation_and_worker_subprocess_error_pins
        ),
        expected_mixed_sentinel_and_result_validation_failure,
    )
    if sum(expected_error_modes) != 1:
        raise ValueError(
            "reviewed reconciliation requires exactly one expected-error mode"
        )
    error = (
        None
        if expected_error is None
        else _non_empty_string(expected_error, "expected_error")
    )
    if expected_run_output_keys not in (
        _FAILED_RUN_OUTPUT_LEGACY_KEYS,
        _FAILED_RUN_OUTPUT_LOGGED_KEYS,
    ):
        raise ValueError("reviewed runs/get-output schema is unsupported")
    evidence_tree_pins = (
        expected_evidence_tree_sha256,
        expected_evidence_tree_file_count,
        expected_evidence_tree_total_bytes,
    )
    if any(value is None for value in evidence_tree_pins) and any(
        value is not None for value in evidence_tree_pins
    ):
        raise ValueError("reviewed evidence tree pins must be complete")
    reviewed_evidence_tree_sha256 = (
        None
        if expected_evidence_tree_sha256 is None
        else _required_sha256(
            expected_evidence_tree_sha256,
            "expected_evidence_tree_sha256",
        )
    )
    reviewed_evidence_tree_file_count = (
        None
        if expected_evidence_tree_file_count is None
        else _positive_int(
            expected_evidence_tree_file_count,
            "expected_evidence_tree_file_count",
        )
    )
    reviewed_evidence_tree_total_bytes = (
        None
        if expected_evidence_tree_total_bytes is None
        else _positive_int(
            expected_evidence_tree_total_bytes,
            "expected_evidence_tree_total_bytes",
        )
    )
    plan, pins = _validated_historical_qualification_plan_and_pins(plan_record)
    if (
        plan.get("closed_record_sha256") != reviewed_plan_sha256
        or pins.runner_sha256 != reviewed_runner_sha256
    ):
        raise ValueError("failed-attempt reconciliation plan is not reviewed")
    contracts = _validated_qualification_payloads(plan, submit_payloads)
    mixed_failure_categories = (
        _mixed_sentinel_and_result_validation_failure_categories()
        if expected_mixed_sentinel_and_result_validation_failure
        else None
    )
    contract_job_ids = {str(contract["job_id"]) for contract in contracts}
    if (
        mixed_failure_categories is not None
        and set(mixed_failure_categories) != contract_job_ids
    ):
        raise ValueError(
            "mixed sentinel/result-validation failure job sets differ from the plan"
        )
    reviewed_error_sha256_by_job = (
        None
        if expected_runtime_lock_index_error_sha256_by_job is None
        else _validated_reviewed_error_sha256_by_job(
            expected_runtime_lock_index_error_sha256_by_job,
            contracts=contracts,
        )
    )
    reviewed_site_packages_error_sha256_by_job = (
        None
        if expected_site_packages_path_error_sha256_by_job is None
        else _validated_reviewed_error_sha256_by_job(
            expected_site_packages_path_error_sha256_by_job,
            contracts=contracts,
        )
    )
    reviewed_runtime_observation_and_worker_subprocess_error_sha256_by_job = (
        None
        if expected_runtime_observation_and_worker_subprocess_error_sha256_by_job
        is None
        else _validated_reviewed_error_sha256_by_job(
            expected_runtime_observation_and_worker_subprocess_error_sha256_by_job,
            contracts=contracts,
        )
    )
    reviewed_runtime_observation_and_worker_subprocess_error_utf8_bytes_by_job = (
        None
        if expected_runtime_observation_and_worker_subprocess_error_utf8_bytes_by_job
        is None
        else _validated_reviewed_error_utf8_bytes_by_job(
            expected_runtime_observation_and_worker_subprocess_error_utf8_bytes_by_job,
            contracts=contracts,
        )
    )
    reviewed_mixed_failure_error_sha256_by_job = (
        _validated_reviewed_error_sha256_by_job(
            dict(
                GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_ERROR_SHA256_BY_JOB
            ),
            contracts=contracts,
        )
        if mixed_failure_categories is not None
        else None
    )
    reviewed_mixed_failure_error_utf8_bytes_by_job = (
        _validated_reviewed_error_utf8_bytes_by_job(
            dict(
                GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_ERROR_UTF8_BYTES_BY_JOB
            ),
            contracts=contracts,
        )
        if mixed_failure_categories is not None
        else None
    )
    local_preflight_binding = _non_authorizing_local_preflight_binding(
        local_preflight_evidence_path,
        plan=plan,
    )
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    if databricks_ledger_path_sha256(ledger_path) != plan.get(
        "campaign_ledger_path_sha256"
    ):
        raise ValueError("reconciliation ledger path differs from the reviewed plan")
    _require_existing_qualification_batch_marker(submit_receipt_root)
    batch_authorization, batch_marker = _replay_qualification_batch_marker(
        plan=plan,
        contracts=contracts,
        ledger_path=ledger_path,
        submit_receipt_root=submit_receipt_root,
        local_preflight_binding=local_preflight_binding,
        require_existing_marker=True,
    )
    submit_receipts = _load_submit_receipts(
        submit_receipt_root,
        contracts=contracts,
        plan=plan,
        ledger=ledger,
        phase_batch_record_sha256=str(batch_marker["closed_record_sha256"]),
    )
    _require_failed_batch_is_current_ledger_suffix(
        ledger,
        batch_authorization=batch_authorization,
        contracts=contracts,
    )
    evidence_root = _validated_existing_controller_evidence_root(
        runs_get_evidence_root,
        "runs_get_evidence_root",
    )
    expected_names = {
        "reconciliation-manifest.json",
        *(f"{contract['job_id']}.runs-get.json" for contract in contracts),
        *(f"{contract['job_id']}.runs-get-output.json" for contract in contracts),
    }
    if {path.name for path in evidence_root.iterdir()} != expected_names:
        raise ValueError("v2 failure evidence root is not the exact batch closure")

    runs: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for contract, receipt in zip(contracts, submit_receipts, strict=True):
        run_path = evidence_root / f"{contract['job_id']}.runs-get.json"
        run = _read_canonical_json_object_file(
            run_path,
            f"failed runs/get {contract['job_id']}",
        )
        base_entry = _failed_attempt_reconciliation_entry(
            run,
            contract=contract,
            submit_receipt=receipt,
            evidence_file_sha256=_file_sha256(run_path),
        )
        output_path = evidence_root / f"{contract['job_id']}.runs-get-output.json"
        run_output = _read_canonical_json_object_file(
            output_path,
            f"failed runs/get-output {contract['job_id']}",
        )
        if set(run_output) != expected_run_output_keys:
            raise ValueError(
                "failed runs/get-output response differs from the reviewed "
                "incident schema"
            )
        job_id = str(contract["job_id"])
        reviewed_job_error = error
        if reviewed_job_error is None:
            if reviewed_error_sha256_by_job is not None:
                reviewed_job_error = _validated_runtime_lock_index_failure_error(
                    run_output,
                    plan_sha256=reviewed_plan_sha256,
                    job_id=job_id,
                    expected_error_sha256=reviewed_error_sha256_by_job[job_id],
                )
            elif reviewed_site_packages_error_sha256_by_job is not None:
                reviewed_job_error = _validated_site_packages_path_failure_error(
                    run_output,
                    plan_sha256=reviewed_plan_sha256,
                    job_id=job_id,
                    expected_error_sha256=(
                        reviewed_site_packages_error_sha256_by_job[job_id]
                    ),
                )
            elif (
                reviewed_runtime_observation_and_worker_subprocess_error_sha256_by_job
                is not None
                and reviewed_runtime_observation_and_worker_subprocess_error_utf8_bytes_by_job
                is not None
            ):
                reviewed_job_error = _validated_runtime_observation_and_worker_subprocess_failure_error(
                    run_output,
                    run=run,
                    plan_sha256=reviewed_plan_sha256,
                    job_id=job_id,
                    expected_error_sha256=(
                        reviewed_runtime_observation_and_worker_subprocess_error_sha256_by_job[
                            job_id
                        ]
                    ),
                    expected_error_utf8_bytes=(
                        reviewed_runtime_observation_and_worker_subprocess_error_utf8_bytes_by_job[
                            job_id
                        ]
                    ),
                )
            elif (
                reviewed_mixed_failure_error_sha256_by_job is not None
                and reviewed_mixed_failure_error_utf8_bytes_by_job is not None
            ):
                reviewed_job_error = _validated_mixed_sentinel_and_result_validation_failure_error(
                    run_output,
                    run=run,
                    plan_sha256=reviewed_plan_sha256,
                    job_id=job_id,
                    expected_error_sha256=(
                        reviewed_mixed_failure_error_sha256_by_job[job_id]
                    ),
                    expected_error_utf8_bytes=(
                        reviewed_mixed_failure_error_utf8_bytes_by_job[job_id]
                    ),
                )
            else:
                raise RuntimeError("reviewed per-job error closure is unavailable")
        entry = _failed_attempt_reconciliation_v2_entry(
            run_output,
            run=run,
            base_entry=base_entry,
            contract=contract,
            expected_error=reviewed_job_error,
            evidence_file_sha256=_file_sha256(output_path),
        )
        runs.append(run)
        entries.append(entry)

    predicted, terminal_prefix = _predicted_failed_batch_terminal_ledger(
        ledger,
        batch_authorization=batch_authorization,
        contracts=contracts,
        runs=runs,
        entries=entries,
    )
    if terminal_prefix.prefix_sha256 != reviewed_terminal_prefix_sha256:
        raise ValueError("failed-attempt terminal prefix is not reviewed")
    if mixed_failure_categories is not None:
        _require_mixed_sentinel_and_result_validation_predicted_ledger(
            predicted,
            predecessor_terminal_count=(
                batch_authorization.predecessor_prefix.terminal_actual_count
            ),
            terminal_prefix=terminal_prefix,
        )
    manifest_path = _validated_existing_regular_file(
        evidence_root / "reconciliation-manifest.json",
        "failed-attempt reconciliation v2 manifest",
    )
    if _file_sha256(manifest_path) != reviewed_manifest_file_sha256:
        raise ValueError("failed-attempt reconciliation v2 manifest file is not reviewed")
    manifest = _canonical_json_object_from_record_bytes(
        manifest_path.read_bytes(),
        label="failed-attempt reconciliation v2 manifest",
    )
    _validate_failed_attempt_reconciliation_v2_manifest(
        manifest,
        plan=plan,
        batch_authorization=batch_authorization,
        expected_entries=entries,
        expected_reason=reason,
        expected_terminal_prefix=terminal_prefix,
    )
    if manifest.get("closed_record_sha256") != reviewed_manifest_sha256:
        raise ValueError("failed-attempt reconciliation v2 manifest is not reviewed")
    if reviewed_evidence_tree_sha256 is not None:
        observed_tree = _failed_attempt_evidence_tree_binding(evidence_root)
        expected_tree = (
            reviewed_evidence_tree_file_count,
            reviewed_evidence_tree_total_bytes,
            reviewed_evidence_tree_sha256,
        )
        if observed_tree != expected_tree:
            raise ValueError("failed-attempt evidence tree is not reviewed")

    ordered = sorted(
        zip(contracts, runs, entries, strict=True),
        key=lambda item: str(item[0]["job_id"]),
    )
    already_closed = len(ledger.terminal_actuals) - (
        batch_authorization.predecessor_prefix.terminal_actual_count
    )
    updated = ledger
    for index, (contract, run, entry) in enumerate(ordered):
        if index < already_closed:
            continue
        updated = record_databricks_verified_run_terminal_actual_json(
            ledger_path,
            attempt_id=str(contract["reservation_attempt_id"]),
            run_record=run,
        )
        actual = next(
            item
            for item in updated.terminal_actuals
            if item.attempt_id == contract["reservation_attempt_id"]
        )
        expected_actual = _failed_attempt_terminal_actual(
            run,
            contract=contract,
            entry=entry,
        )
        if actual != expected_actual:
            raise RuntimeError("ledger actual differs from reviewed failure evidence")
    final_prefix = _require_qualification_phase_ledger_closure(
        updated,
        batch_authorization=batch_authorization,
        contracts=contracts,
    )
    if final_prefix != terminal_prefix or updated != predicted:
        raise RuntimeError("failed-attempt reconciliation final ledger drift")
    return updated


def reconcile_gpu_qualification_bootstrap_file_global_failure_evidence(
    *,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    runs_get_evidence_root: str | Path,
) -> DatabricksClusterHourLedger:
    """Account the reviewed 2cf4 bootstrap ``__file__`` failure closure.

    This incident-specific wrapper is the only public v2 mutation boundary.  It
    pins the plan, exact retained manifest, failure cause, and resulting ledger
    prefix in source; it cannot be redirected to caller-authored review values
    and never issues qualification launch authority.
    """

    return _reconcile_reviewed_gpu_qualification_failed_attempt_evidence_v2(
        plan_record=plan_record,
        submit_payloads=submit_payloads,
        ledger_path=ledger_path,
        submit_receipt_root=submit_receipt_root,
        local_preflight_evidence_path=local_preflight_evidence_path,
        runs_get_evidence_root=runs_get_evidence_root,
        expected_plan_sha256=(
            GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_PLAN_SHA256
        ),
        expected_runner_sha256=(
            GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_RUNNER_SHA256
        ),
        expected_manifest_closed_record_sha256=(
            GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_MANIFEST_SHA256
        ),
        expected_manifest_file_sha256=(
            GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_MANIFEST_FILE_SHA256
        ),
        expected_terminal_prefix_sha256=(
            GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_TERMINAL_PREFIX_SHA256
        ),
        expected_failure_reason=(
            GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_REASON
        ),
        expected_error=GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_ERROR,
        expected_run_output_keys=_FAILED_RUN_OUTPUT_LEGACY_KEYS,
    )


def reconcile_gpu_qualification_bootstrap_cluster_identity_failure_evidence(
    *,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    runs_get_evidence_root: str | Path,
) -> DatabricksClusterHourLedger:
    """Account the reviewed d6f7 bootstrap cluster-identity failure closure.

    The immutable plan, runner, exact five-key task-output schema, captured
    manifest, incident cause, and resulting ledger prefix are source-pinned.
    Reconciliation remains offline and validates the complete 29-file closure
    before the first deterministic terminal-actual append.
    """

    return _reconcile_reviewed_gpu_qualification_failed_attempt_evidence_v2(
        plan_record=plan_record,
        submit_payloads=submit_payloads,
        ledger_path=ledger_path,
        submit_receipt_root=submit_receipt_root,
        local_preflight_evidence_path=local_preflight_evidence_path,
        runs_get_evidence_root=runs_get_evidence_root,
        expected_plan_sha256=(
            GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_PLAN_SHA256
        ),
        expected_runner_sha256=(
            GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_RUNNER_SHA256
        ),
        expected_manifest_closed_record_sha256=(
            GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_MANIFEST_SHA256
        ),
        expected_manifest_file_sha256=(
            GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_MANIFEST_FILE_SHA256
        ),
        expected_terminal_prefix_sha256=(
            GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_TERMINAL_PREFIX_SHA256
        ),
        expected_failure_reason=(
            GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_REASON
        ),
        expected_error=GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_ERROR,
        expected_run_output_keys=_FAILED_RUN_OUTPUT_LOGGED_KEYS,
    )


def reconcile_gpu_qualification_runtime_lock_index_failure_evidence(
    *,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    runs_get_evidence_root: str | Path,
) -> DatabricksClusterHourLedger:
    """Account the reviewed f991 pip runtime-lock index failure closure.

    The source-pinned boundary requires each job's exact raw error, the one
    canonical path-normalized argv grammar, complete logs with the reviewed
    torch-resolution marker, and the sealed 29-file evidence closure before
    the first deterministic terminal-actual append.
    """

    return _reconcile_reviewed_gpu_qualification_failed_attempt_evidence_v2(
        plan_record=plan_record,
        submit_payloads=submit_payloads,
        ledger_path=ledger_path,
        submit_receipt_root=submit_receipt_root,
        local_preflight_evidence_path=local_preflight_evidence_path,
        runs_get_evidence_root=runs_get_evidence_root,
        expected_plan_sha256=(
            GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_PLAN_SHA256
        ),
        expected_runner_sha256=(
            GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_RUNNER_SHA256
        ),
        expected_manifest_closed_record_sha256=(
            GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_MANIFEST_SHA256
        ),
        expected_manifest_file_sha256=(
            GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_MANIFEST_FILE_SHA256
        ),
        expected_terminal_prefix_sha256=(
            GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_TERMINAL_PREFIX_SHA256
        ),
        expected_failure_reason=(
            GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_REASON
        ),
        expected_error=None,
        expected_run_output_keys=_FAILED_RUN_OUTPUT_LOGGED_KEYS,
        expected_runtime_lock_index_error_sha256_by_job=dict(
            GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_ERROR_SHA256_BY_JOB
        ),
    )


def reconcile_gpu_qualification_site_packages_path_failure_evidence(
    *,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    runs_get_evidence_root: str | Path,
) -> DatabricksClusterHourLedger:
    """Account the reviewed be4cb site-packages freezer failure closure.

    The source-pinned boundary requires every job's exact raw error and planned
    invalid Debian ``dist-packages`` path, the normalized error grammar, two
    successful pip checks, and trace proof that failure preceded worker launch.
    The exact 29-file evidence tree is closed before the first terminal append.
    """

    return _reconcile_reviewed_gpu_qualification_failed_attempt_evidence_v2(
        plan_record=plan_record,
        submit_payloads=submit_payloads,
        ledger_path=ledger_path,
        submit_receipt_root=submit_receipt_root,
        local_preflight_evidence_path=local_preflight_evidence_path,
        runs_get_evidence_root=runs_get_evidence_root,
        expected_plan_sha256=(
            GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_PLAN_SHA256
        ),
        expected_runner_sha256=(
            GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_RUNNER_SHA256
        ),
        expected_manifest_closed_record_sha256=(
            GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_MANIFEST_SHA256
        ),
        expected_manifest_file_sha256=(
            GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_MANIFEST_FILE_SHA256
        ),
        expected_terminal_prefix_sha256=(
            GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_TERMINAL_PREFIX_SHA256
        ),
        expected_failure_reason=GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_REASON,
        expected_error=None,
        expected_run_output_keys=_FAILED_RUN_OUTPUT_LOGGED_KEYS,
        expected_site_packages_path_error_sha256_by_job=dict(
            GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_ERROR_SHA256_BY_JOB
        ),
        expected_evidence_tree_sha256=(
            GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_EVIDENCE_TREE_SHA256
        ),
        expected_evidence_tree_file_count=(
            GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_EVIDENCE_TREE_FILE_COUNT
        ),
        expected_evidence_tree_total_bytes=(
            GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_EVIDENCE_TREE_TOTAL_BYTES
        ),
    )


def reconcile_gpu_qualification_runtime_observation_and_worker_subprocess_failure_evidence(
    *,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    runs_get_evidence_root: str | Path,
) -> DatabricksClusterHourLedger:
    """Account the reviewed c0bede split runtime/worker failure closure.

    The source-pinned boundary recognizes only the two packed-page post-success
    runtime-observation errors and the other twelve exact worker-subprocess argv
    errors.  It binds every raw error by SHA-256 and UTF-8 length, validates the
    complete logs, category-specific traces and terminal states, and closes the
    exact 29-file evidence tree before the first terminal-actual append.
    """

    return _reconcile_reviewed_gpu_qualification_failed_attempt_evidence_v2(
        plan_record=plan_record,
        submit_payloads=submit_payloads,
        ledger_path=ledger_path,
        submit_receipt_root=submit_receipt_root,
        local_preflight_evidence_path=local_preflight_evidence_path,
        runs_get_evidence_root=runs_get_evidence_root,
        expected_plan_sha256=(
            GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_PLAN_SHA256
        ),
        expected_runner_sha256=(
            GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_RUNNER_SHA256
        ),
        expected_manifest_closed_record_sha256=(
            GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_MANIFEST_SHA256
        ),
        expected_manifest_file_sha256=(
            GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_MANIFEST_FILE_SHA256
        ),
        expected_terminal_prefix_sha256=(
            GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_TERMINAL_PREFIX_SHA256
        ),
        expected_failure_reason=(
            GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_REASON
        ),
        expected_error=None,
        expected_run_output_keys=_FAILED_RUN_OUTPUT_LOGGED_KEYS,
        expected_runtime_observation_and_worker_subprocess_error_sha256_by_job=dict(
            GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_SHA256_BY_JOB
        ),
        expected_runtime_observation_and_worker_subprocess_error_utf8_bytes_by_job=dict(
            GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_UTF8_BYTES_BY_JOB
        ),
        expected_evidence_tree_sha256=(
            GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_EVIDENCE_TREE_SHA256
        ),
        expected_evidence_tree_file_count=(
            GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_EVIDENCE_TREE_FILE_COUNT
        ),
        expected_evidence_tree_total_bytes=(
            GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_EVIDENCE_TREE_TOTAL_BYTES
        ),
    )


def reconcile_gpu_qualification_mixed_sentinel_and_result_validation_failure_evidence(
    *,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    evidence_root: str | Path,
) -> DatabricksClusterHourLedger:
    """Account the reviewed 694441 mixed terminal-failure closure.

    This zero-authority boundary admits only the source-pinned two version
    mismatches, two unresolved-native failures, eight layout conflicts, and two
    FlashInfer engine-initialization failures.  It validates every raw error,
    exact terminal topology, 29-file evidence tree, manifest, and final canonical
    ledger projection before the first deterministic terminal-actual append.
    """

    return _reconcile_reviewed_gpu_qualification_failed_attempt_evidence_v2(
        plan_record=plan_record,
        submit_payloads=submit_payloads,
        ledger_path=ledger_path,
        submit_receipt_root=submit_receipt_root,
        local_preflight_evidence_path=local_preflight_evidence_path,
        runs_get_evidence_root=evidence_root,
        expected_plan_sha256=(
            GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_PLAN_SHA256
        ),
        expected_runner_sha256=(
            GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_RUNNER_SHA256
        ),
        expected_manifest_closed_record_sha256=(
            GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_MANIFEST_SHA256
        ),
        expected_manifest_file_sha256=(
            GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_MANIFEST_FILE_SHA256
        ),
        expected_terminal_prefix_sha256=(
            GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_TERMINAL_PREFIX_SHA256
        ),
        expected_failure_reason=(
            GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_REASON
        ),
        expected_error=None,
        expected_run_output_keys=_FAILED_RUN_OUTPUT_LOGGED_KEYS,
        expected_mixed_sentinel_and_result_validation_failure=True,
        expected_evidence_tree_sha256=(
            GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_EVIDENCE_TREE_SHA256
        ),
        expected_evidence_tree_file_count=(
            GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_EVIDENCE_TREE_FILE_COUNT
        ),
        expected_evidence_tree_total_bytes=(
            GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_EVIDENCE_TREE_TOTAL_BYTES
        ),
    )


def reconcile_gpu_qualification_failed_attempt_evidence(
    *,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    runs_get_evidence_root: str | Path,
    require_zero_actual: bool = False,
) -> DatabricksClusterHourLedger:
    """Reconcile the retained legacy UC failures without issuing launch authority.

    This boundary is intentionally offline: it consumes the canonical direct
    ``runs/get`` files and their sealed manifest, never calls Databricks, and
    returns only the updated ledger.  ``require_zero_actual`` is a fail-before-
    write audit guard; it never rewrites a nonzero observed interval to zero.
    """

    if type(require_zero_actual) is not bool:
        raise TypeError("require_zero_actual must be a bool")
    plan, _pins = _validated_legacy_uc_failure_plan_and_pins(plan_record)
    contracts = _validated_qualification_payloads(
        plan,
        submit_payloads,
        require_legacy_uc_broken_security_shape=True,
    )
    local_preflight_binding = _non_authorizing_local_preflight_binding(
        local_preflight_evidence_path,
        plan=plan,
    )
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    if databricks_ledger_path_sha256(ledger_path) != plan.get(
        "campaign_ledger_path_sha256"
    ):
        raise ValueError("reconciliation ledger path differs from the legacy plan")
    batch_authorization, batch_marker = _replay_qualification_batch_marker(
        plan=plan,
        contracts=contracts,
        ledger_path=ledger_path,
        submit_receipt_root=submit_receipt_root,
        local_preflight_binding=local_preflight_binding,
    )
    submit_receipts = _load_submit_receipts(
        submit_receipt_root,
        contracts=contracts,
        plan=plan,
        ledger=ledger,
        phase_batch_record_sha256=str(batch_marker["closed_record_sha256"]),
    )
    evidence_root = _validated_existing_controller_evidence_root(
        runs_get_evidence_root,
        "runs_get_evidence_root",
    )
    expected_names = {
        "reconciliation-manifest.json",
        *(f"{contract['job_id']}.runs-get.json" for contract in contracts),
    }
    if {path.name for path in evidence_root.iterdir()} != expected_names:
        raise ValueError("runs/get evidence root is not the exact fourteen-job closure")
    runs: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for contract, receipt in zip(contracts, submit_receipts, strict=True):
        evidence_path = evidence_root / f"{contract['job_id']}.runs-get.json"
        run = _read_canonical_json_object_file(
            evidence_path,
            f"failed runs/get {contract['job_id']}",
        )
        entry = _failed_attempt_reconciliation_entry(
            run,
            contract=contract,
            submit_receipt=receipt,
            evidence_file_sha256=_file_sha256(evidence_path),
        )
        runs.append(run)
        entries.append(entry)
    manifest_path = _validated_existing_regular_file(
        evidence_root / "reconciliation-manifest.json",
        "failed-attempt reconciliation manifest",
    )
    manifest = _canonical_json_object_from_bytes(
        manifest_path.read_bytes(),
        pretty=True,
        label="failed-attempt reconciliation manifest",
    )
    _validate_failed_attempt_reconciliation_manifest(
        manifest,
        plan=plan,
        expected_entries=entries,
    )
    if manifest.get("closed_record_sha256") != (
        GPU_QUALIFICATION_LEGACY_UC_FAILURE_MANIFEST_CLOSED_RECORD_SHA256
    ):
        raise ValueError("failed-attempt reconciliation manifest is not reviewed")
    if require_zero_actual and any(
        entry["actual_cluster_duration_seconds"] != 0.0 for entry in entries
    ):
        raise ValueError(
            "runs/get evidence contains nonzero task intervals; zero reconciliation refused"
        )

    updated = ledger
    ordered_reconciliations = sorted(
        zip(contracts, runs, entries, strict=True),
        key=lambda item: str(item[0]["job_id"]),
    )
    for contract, run, entry in ordered_reconciliations:
        updated = record_databricks_verified_run_terminal_actual_json(
            ledger_path,
            attempt_id=str(contract["reservation_attempt_id"]),
            run_record=run,
        )
        actual = next(
            item
            for item in updated.terminal_actuals
            if item.attempt_id == contract["reservation_attempt_id"]
        )
        if (
            actual.actual_cluster_duration_seconds
            != entry["actual_cluster_duration_seconds"]
            or actual.control_plane_status_sha256
            != entry["control_plane_status_sha256"]
        ):
            raise RuntimeError("ledger actual differs from retained runs/get evidence")
    terminal_prefix = _require_qualification_phase_ledger_closure(
        updated,
        batch_authorization=batch_authorization,
        contracts=contracts,
    )
    if (
        terminal_prefix.reservation_count != 152
        or terminal_prefix.submission_receipt_count != 14
        or terminal_prefix.terminal_actual_count != 152
        or terminal_prefix.prefix_sha256
        != GPU_QUALIFICATION_LEGACY_UC_FAILURE_TERMINAL_PREFIX_SHA256
    ):
        raise RuntimeError("failed-attempt reconciliation terminal prefix drift")
    return updated


def collect_gpu_qualification_evidence(
    config: DatabricksWorkspaceConfig,
    *,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    terminal_receipt_root: str | Path,
    evidence_output_json: str | Path,
    now: Callable[[], datetime] | None = None,
) -> tuple[dict[str, Any], GPUQualificationLaunchAuthorization]:
    """Authorize qualification only through the real terminal Jobs API transport.

    Unlike submission, this authority-bearing boundary intentionally has no
    injectable opener.  Tests may monkeypatch the package-owned ``runs/get``
    function, but callers cannot pass fabricated status mappings into an
    authorizing production invocation.
    """

    plan, pins = _validated_plan_and_pins(plan_record)
    contracts = _validated_qualification_payloads(plan, submit_payloads)
    local_preflight_binding, _preflight_completed_at, local_preflight = (
        _validated_local_preflight_binding(
            local_preflight_evidence_path,
            plan=plan,
            submit_payloads=_qualification_contract_submit_payloads(contracts),
            config=config,
            require_fresh_workspace=False,
        )
    )
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    if databricks_ledger_path_sha256(ledger_path) != plan.get(
        "campaign_ledger_path_sha256"
    ):
        raise ValueError("collection ledger path differs from the campaign plan")
    batch_authorization, batch_marker = _replay_qualification_batch_marker(
        plan=plan,
        contracts=contracts,
        ledger_path=ledger_path,
        submit_receipt_root=submit_receipt_root,
        local_preflight_binding=local_preflight_binding,
    )
    submit_receipts = _load_submit_receipts(
        submit_receipt_root,
        contracts=contracts,
        plan=plan,
        ledger=ledger,
        phase_batch_record_sha256=str(batch_marker["closed_record_sha256"]),
    )
    terminal_root = _validated_fresh_controller_evidence_root(terminal_receipt_root)
    _require_fresh_output_path(Path(evidence_output_json))
    clock = now or _utc_now
    job_results: list[dict[str, Any]] = []
    terminal_receipts: list[dict[str, Any]] = []
    for planned_job, contract, submit_receipt in zip(
        _planned_jobs(plan), contracts, submit_receipts, strict=True
    ):
        run = get_databricks_run(
            config,
            str(submit_receipt["cloud_run_id"]),
        )
        # Reconcile every direct terminal outcome before applying the stricter
        # success-only qualification launch/result contract.  A rejected job
        # with no allocated task timestamps must release its reservation while
        # still failing the campaign.
        reconciled = record_databricks_verified_run_terminal_actual_json(
            ledger_path,
            attempt_id=str(contract["reservation_attempt_id"]),
            run_record=run,
        )
        actual = next(
            item
            for item in reconciled.terminal_actuals
            if item.attempt_id == contract["reservation_attempt_id"]
        )
        run_identity = _validate_control_plane_run(
            run,
            planned_job=planned_job,
            contract=contract,
            submit_receipt=submit_receipt,
        )
        control_plane_status_sha256 = _canonical_json_sha256(run)
        if (
            actual.verification_source != "direct_databricks_runs_get"
            or actual.run_id != run_identity["cloud_run_id"]
            or actual.submit_payload_sha256 != contract["submit_payload_sha256"]
            or actual.control_plane_status_sha256 != control_plane_status_sha256
        ):
            raise RuntimeError(
                "qualification ledger terminal is not bound to this control-plane response"
            )
        if (
            run_identity["succeeded"] is not True
            or actual.terminal_state != "succeeded"
        ):
            raise RuntimeError(
                f"GPU qualification job {contract['job_id']!r} did not succeed"
            )
        result = _read_gpu_qualification_result(
            config,
            str(contract["output_json"]),
            label=f"GPU result {contract['job_id']}",
        )
        validate_gpu_job_result_record(
            result,
            plan_record=plan,
            expected_artifact_pins=pins,
        )
        _validate_result_submission_binding(
            result,
            contract=contract,
            submit_receipt=submit_receipt,
            run_identity=run_identity,
        )
        terminal_receipt = _terminal_receipt_record(
            plan=plan,
            contract=contract,
            submit_receipt=submit_receipt,
            ledger_id=reconciled.ledger_id,
            ledger_terminal_actual=actual,
            run=run,
            run_identity=run_identity,
            result=result,
            collected_at_utc=_utc_timestamp(clock()),
            phase_batch_record_sha256=str(batch_marker["closed_record_sha256"]),
        )
        job_results.append(result)
        terminal_receipts.append(terminal_receipt)

    final_ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    terminal_prefix = _require_qualification_phase_ledger_closure(
        final_ledger,
        batch_authorization=batch_authorization,
        contracts=contracts,
    )
    for receipt in terminal_receipts:
        receipt["phase_terminal_prefix"] = terminal_prefix.to_record()
        receipt["closed_record_sha256"] = ""
        _seal_record(receipt)
    _validate_collected_identity_closure(terminal_receipts, contracts=contracts)
    selected_gmus = [
        float(result["measurements"]["gpu_memory_utilization"])
        for result in job_results
        if result["job_id"].startswith("aws-g6-l4-32k-c4-gmu-")
        and result["measurements"].get("candidate_qualified") is True
    ]
    if not selected_gmus:
        raise RuntimeError("no governed GMU result qualified")
    cloud = _build_governed_cloud_gpu_evidence(
        plan_sha256=str(plan["closed_record_sha256"]),
        jobs=job_results,
        terminal_receipts=terminal_receipts,
        selected_gpu_memory_utilization=max(selected_gmus),
    )
    evidence = _build_governed_gpu_qualification_evidence(
        campaign_id=str(plan["campaign_id"]),
        plan_sha256=str(plan["closed_record_sha256"]),
        local_preflight_evidence=local_preflight,
        cloud_gpu_evidence=cloud,
    )
    validate_gpu_qualification_evidence_record(
        evidence,
        plan_record=plan,
        expected_campaign_id=str(plan["campaign_id"]),
        expected_artifact_pins=pins,
    )
    _publish_terminal_receipts_atomic(terminal_root, terminal_receipts)
    evidence_path = Path(evidence_output_json)
    _write_canonical_exclusive(evidence, evidence_path)
    authorization = replay_gpu_qualification_launch_authorization(
        config=config,
        plan_record=plan,
        submit_payloads=submit_payloads,
        ledger_path=ledger_path,
        submit_receipt_root=submit_receipt_root,
        local_preflight_evidence_path=local_preflight_evidence_path,
        terminal_receipt_root=terminal_root,
        evidence_path=evidence_path,
        expected_campaign_id=str(plan["campaign_id"]),
        expected_artifact_pins=pins,
    )
    return evidence, authorization


def replay_gpu_qualification_launch_authorization(
    *,
    config: DatabricksWorkspaceConfig,
    plan_record: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    terminal_receipt_root: str | Path,
    evidence_path: str | Path,
    expected_campaign_id: str,
    expected_artifact_pins: GPUQualificationArtifactPins,
) -> GPUQualificationLaunchAuthorization:
    """Reissue launch authority from the complete durable causal closure.

    A qualification JSON record by itself is intentionally insufficient.  A
    replay must rejoin the exact rendered submit payloads, append-only ledger,
    submit receipts, terminal receipts, and canonical evidence file.
    """

    plan, pins = _validated_plan_and_pins(plan_record)
    if str(plan["campaign_id"]) != expected_campaign_id:
        raise ValueError("replay campaign_id differs from the frozen expectation")
    if pins != expected_artifact_pins:
        raise ValueError("replay artifact pins differ from the frozen expectation")
    contracts = _validated_qualification_payloads(plan, submit_payloads)
    local_preflight_binding, _preflight_completed_at, local_preflight = (
        _validated_local_preflight_binding(
            local_preflight_evidence_path,
            plan=plan,
            submit_payloads=_qualification_contract_submit_payloads(contracts),
            config=config,
            require_fresh_workspace=False,
        )
    )
    ledger_file = _validated_existing_regular_file(ledger_path, "ledger_path")
    ledger = read_databricks_cluster_hour_ledger_json(ledger_file)
    ledger_path_sha256 = databricks_ledger_path_sha256(ledger_file)
    if ledger_path_sha256 != plan.get("campaign_ledger_path_sha256"):
        raise ValueError("replay ledger path differs from the campaign plan")
    require_databricks_ledger_prefix(
        ledger,
        databricks_ledger_prefix_from_record(
            _required_mapping(
                plan.get("campaign_ledger_prefix"), "campaign_ledger_prefix"
            )
        ),
    )
    batch_authorization, batch_marker = _replay_qualification_batch_marker(
        plan=plan,
        contracts=contracts,
        ledger_path=ledger_file,
        submit_receipt_root=submit_receipt_root,
        local_preflight_binding=local_preflight_binding,
    )
    submit_receipts = _load_submit_receipts(
        submit_receipt_root,
        contracts=contracts,
        plan=plan,
        ledger=ledger,
        phase_batch_record_sha256=str(batch_marker["closed_record_sha256"]),
    )
    terminal_root = _validated_existing_controller_evidence_root(
        terminal_receipt_root, "terminal_receipt_root"
    )
    expected_names = {f"{contract['job_id']}.json" for contract in contracts}
    observed_names = {path.name for path in terminal_root.iterdir()}
    if observed_names != expected_names:
        raise ValueError("terminal receipt directory is not the exact planned closure")
    terminal_receipts = tuple(
        _read_canonical_json_object_file(
            terminal_root / f"{contract['job_id']}.json",
            f"terminal receipt {contract['job_id']}",
        )
        for contract in contracts
    )
    evidence_file = _validated_existing_regular_file(evidence_path, "evidence_path")
    evidence = _read_canonical_json_object_file(
        evidence_file, "GPU qualification evidence"
    )
    selection = validate_gpu_qualification_evidence_record(
        evidence,
        plan_record=plan,
        expected_campaign_id=expected_campaign_id,
        expected_artifact_pins=expected_artifact_pins,
    )
    if dict(
        _required_mapping(
            evidence.get("local_preflight_evidence"),
            "local_preflight_evidence",
        )
    ) != local_preflight:
        raise ValueError(
            "persisted local preflight differs from qualification evidence"
        )
    cloud = _required_mapping(evidence.get("cloud_gpu_evidence"), "cloud_gpu_evidence")
    embedded_receipts = cloud.get("terminal_receipts")
    if not isinstance(embedded_receipts, list) or embedded_receipts != list(
        terminal_receipts
    ):
        raise ValueError(
            "persisted terminal receipt closure differs from qualification evidence"
        )

    terminal_actual_hashes: list[str] = []
    terminal_prefix = _require_qualification_phase_ledger_closure(
        ledger,
        batch_authorization=batch_authorization,
        contracts=contracts,
    )
    for contract, submit_receipt, terminal_receipt in zip(
        contracts, submit_receipts, terminal_receipts, strict=True
    ):
        attempt_id = str(contract["reservation_attempt_id"])
        actual = next(
            (item for item in ledger.terminal_actuals if item.attempt_id == attempt_id),
            None,
        )
        if actual is None:
            raise ValueError(f"replay ledger has no terminal actual for {attempt_id!r}")
        actual_record = _ledger_terminal_actual_record(actual)
        actual_sha256 = _canonical_json_sha256(actual_record)
        if (
            actual.verification_source != "direct_databricks_runs_get"
            or actual.terminal_state != "succeeded"
            or actual.run_id != submit_receipt["cloud_run_id"]
            or actual.submit_payload_sha256 != contract["submit_payload_sha256"]
            or actual.control_plane_status_sha256
            != terminal_receipt["control_plane_status_sha256"]
            or actual_sha256 != terminal_receipt["ledger_terminal_actual_sha256"]
            or terminal_receipt.get("phase_batch_record_sha256")
            != batch_marker["closed_record_sha256"]
            or terminal_receipt.get("phase_terminal_prefix")
            != terminal_prefix.to_record()
        ):
            raise ValueError(
                f"replay ledger terminal does not match {contract['job_id']!r}"
            )
        terminal_actual_hashes.append(actual_sha256)

    evidence_file_sha256 = _file_sha256(evidence_file)
    ledger_prefix = terminal_prefix
    causal_closure = {
        "evidence_closed_record_sha256": evidence["closed_record_sha256"],
        "evidence_file_sha256": evidence_file_sha256,
        "ledger_id": ledger.ledger_id,
        "ledger_path_sha256": ledger_path_sha256,
        "ledger_prefix": ledger_prefix.to_record(),
        "predecessor_prefix": batch_authorization.predecessor_prefix.to_record(),
        "producer_batch_prefix": batch_authorization.batch_prefix.to_record(),
        "plan_sha256": plan["closed_record_sha256"],
        "submit_payload_sha256": [
            contract["submit_payload_sha256"] for contract in contracts
        ],
        "submit_receipt_sha256": [
            receipt["closed_record_sha256"] for receipt in submit_receipts
        ],
        "terminal_actual_sha256": terminal_actual_hashes,
        "terminal_receipt_sha256": [
            receipt["closed_record_sha256"] for receipt in terminal_receipts
        ],
    }
    return GPUQualificationLaunchAuthorization(
        selection=selection,
        plan_sha256=str(plan["closed_record_sha256"]),
        evidence_closed_record_sha256=str(evidence["closed_record_sha256"]),
        evidence_file_sha256=evidence_file_sha256,
        ledger_id=ledger.ledger_id,
        ledger_path_sha256=ledger_path_sha256,
        predecessor_prefix=batch_authorization.predecessor_prefix,
        producer_batch_prefix=batch_authorization.batch_prefix,
        ledger_prefix=ledger_prefix,
        causal_closure_sha256=_canonical_json_sha256(causal_closure),
        _issuer=_LAUNCH_AUTHORIZATION_ISSUER,
    )


def require_gpu_qualification_launch_authorization(
    value: object,
    *,
    expected_plan_sha256: str,
    expected_evidence_file_sha256: str,
) -> GPUQualificationSelection:
    """Return the selection only from a collector/replay-issued capability."""

    if not isinstance(value, GPUQualificationLaunchAuthorization):
        raise TypeError(
            "publication launch requires GPUQualificationLaunchAuthorization"
        )
    if value.plan_sha256 != _required_sha256(
        expected_plan_sha256, "expected_plan_sha256"
    ):
        raise ValueError("GPU qualification authorization plan binding differs")
    if value.evidence_file_sha256 != _required_sha256(
        expected_evidence_file_sha256, "expected_evidence_file_sha256"
    ):
        raise ValueError("GPU qualification authorization evidence binding differs")
    return value.selection


def _ledger_terminal_actual_record(
    actual: DatabricksClusterHourTerminalActual,
) -> dict[str, Any]:
    return {
        "actual_cluster_duration_seconds": actual.actual_cluster_duration_seconds,
        "actual_cluster_hours": actual.actual_cluster_hours,
        "attempt_id": actual.attempt_id,
        "control_plane_status_sha256": actual.control_plane_status_sha256,
        "run_id": actual.run_id,
        "submit_payload_sha256": actual.submit_payload_sha256,
        "terminal_state": actual.terminal_state,
        "verification_source": actual.verification_source,
    }


def _validate_collected_identity_closure(
    terminal_receipts: Sequence[Mapping[str, Any]],
    *,
    contracts: Sequence[Mapping[str, Any]],
) -> None:
    if len(terminal_receipts) != len(contracts):
        raise ValueError("collected identity closure is incomplete")
    run_ids: set[str] = set()
    task_run_ids: set[str] = set()
    cluster_ids: set[str] = set()
    attempt_ids: set[str] = set()
    task_keys: set[str] = set()
    output_paths: set[str] = set()
    for receipt, contract in zip(terminal_receipts, contracts, strict=True):
        exact = {
            "job_id": contract["job_id"],
            "output_json": contract["output_json"],
            "reservation_attempt_id": contract["reservation_attempt_id"],
            "submit_payload_sha256": contract["submit_payload_sha256"],
            "task_key": contract["task_key"],
        }
        for field_name, expected in exact.items():
            if receipt.get(field_name) != expected:
                raise ValueError(
                    f"collected terminal receipt {field_name} differs from submission"
                )
        identities = (
            (
                run_ids,
                _databricks_run_id(receipt.get("cloud_run_id"), "cloud_run_id"),
                "cloud_run_id",
            ),
            (
                task_run_ids,
                _databricks_run_id(receipt.get("task_run_id"), "task_run_id"),
                "task_run_id",
            ),
            (
                cluster_ids,
                _non_empty_string(receipt.get("cloud_cluster_id"), "cloud_cluster_id"),
                "cloud_cluster_id",
            ),
            (
                attempt_ids,
                str(receipt["reservation_attempt_id"]),
                "reservation_attempt_id",
            ),
            (task_keys, str(receipt["task_key"]), "task_key"),
            (output_paths, str(receipt["output_json"]), "output_json"),
        )
        for observed, value, field_name in identities:
            if value in observed:
                raise ValueError(f"collected {field_name} values must be unique")
            observed.add(value)


def _load_submit_receipts(
    root: str | Path,
    *,
    contracts: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    ledger: DatabricksClusterHourLedger,
    phase_batch_record_sha256: str,
) -> tuple[dict[str, Any], ...]:
    directory = _validated_existing_controller_evidence_root(
        root, "submit_receipt_root"
    )
    expected_names = {
        _QUALIFICATION_PHASE_LEASE_FILENAME,
        _QUALIFICATION_BATCH_MARKER_FILENAME,
        *(f"{contract['job_id']}.json" for contract in contracts),
    }
    observed_names = {path.name for path in directory.iterdir()}
    if observed_names != expected_names:
        raise ValueError("submit receipt directory is not the exact planned closure")
    receipts: list[dict[str, Any]] = []
    for contract in contracts:
        receipt = _read_canonical_json_object_file(
            directory / f"{contract['job_id']}.json",
            f"submit receipt {contract['job_id']}",
        )
        _validate_submit_receipt(
            receipt,
            contract=contract,
            plan=plan,
            ledger=ledger,
            phase_batch_record_sha256=phase_batch_record_sha256,
        )
        receipts.append(receipt)
    return tuple(receipts)


def _validate_submit_receipt(
    receipt: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
    ledger: DatabricksClusterHourLedger,
    phase_batch_record_sha256: str,
) -> None:
    if set(receipt) != _SUBMIT_RECEIPT_KEYS:
        raise ValueError("submit receipt has an open schema")
    _require_closed_record_digest(receipt, "submit receipt")
    expected = {
        "authorization_scope": (
            "submission_identity_only_requires_direct_terminal_collection"
        ),
        "job_id": contract["job_id"],
        "ledger_id": ledger.ledger_id,
        "output_json": contract["output_json"],
        "plan_sha256": plan["closed_record_sha256"],
        "phase_batch_record_sha256": phase_batch_record_sha256,
        "record_type": GPU_QUALIFICATION_SUBMIT_RECEIPT_RECORD_TYPE,
        "reservation_attempt_id": contract["reservation_attempt_id"],
        "schema_version": GPU_QUALIFICATION_SCHEMA_VERSION,
        "submit_payload_sha256": contract["submit_payload_sha256"],
        "task_key": contract["task_key"],
    }
    for field_name, expected_value in expected.items():
        if receipt.get(field_name) != expected_value:
            raise ValueError(f"submit receipt {field_name} differs")
    _required_sha256(receipt.get("submit_response_sha256"), "submit_response_sha256")
    _parse_utc_timestamp(receipt.get("submitted_at_utc"), "submitted_at_utc")
    cloud_run_id = _non_empty_string(receipt.get("cloud_run_id"), "cloud_run_id")
    ledger_receipt = next(
        (
            item
            for item in ledger.submission_receipts
            if item.attempt_id == contract["reservation_attempt_id"]
        ),
        None,
    )
    if ledger_receipt is None or (
        ledger_receipt.run_id != cloud_run_id
        or ledger_receipt.submit_payload_sha256 != contract["submit_payload_sha256"]
        or ledger_receipt.submit_response_sha256 != receipt["submit_response_sha256"]
    ):
        raise ValueError("submit receipt does not match the append-only ledger")


def _validate_control_plane_run(
    run: Mapping[str, Any],
    *,
    planned_job: Mapping[str, Any],
    contract: Mapping[str, Any],
    submit_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    control_plane_run_id = _databricks_run_id(run.get("run_id"), "runs/get run_id")
    if control_plane_run_id != submit_receipt["cloud_run_id"]:
        raise ValueError("runs/get run_id differs from the submit receipt")
    payload = _required_mapping(contract.get("payload"), "payload")
    if run.get("run_name") != payload.get("run_name"):
        raise ValueError("runs/get run_name differs from the immutable submit payload")
    if run.get("run_type") not in (None, "SUBMIT_RUN"):
        raise ValueError("qualification run is not a one-time submit run")
    if run.get("repair_history") not in (None, []):
        raise ValueError("qualification run has repair history")
    if run.get("original_attempt_run_id") not in (None, 0, "0"):
        raise ValueError("qualification run is not attempt zero")
    state = _required_mapping(run.get("state"), "runs/get state")
    life_cycle_state = state.get("life_cycle_state")
    result_state = state.get("result_state")
    if life_cycle_state not in {"TERMINATED", "SKIPPED", "INTERNAL_ERROR", "BLOCKED"}:
        raise ValueError("qualification runs/get response is not terminal")
    start_time = _nonnegative_int(run.get("start_time"), "run.start_time")
    end_time = _positive_int(run.get("end_time"), "run.end_time")
    if end_time <= start_time:
        raise ValueError("qualification run terminal times do not increase")
    tasks = run.get("tasks")
    if (
        not isinstance(tasks, list)
        or len(tasks) != 1
        or not isinstance(tasks[0], Mapping)
    ):
        raise ValueError("qualification runs/get must contain exactly one task")
    task = tasks[0]
    if task.get("task_key") != contract["task_key"]:
        raise ValueError("runs/get task_key differs")
    if task.get("attempt_number") not in (None, 0):
        raise ValueError("runs/get task was retried")
    task_state = _required_mapping(task.get("state"), "runs/get task state")
    task_life_cycle_state = task_state.get("life_cycle_state")
    task_result_state = task_state.get("result_state")
    if task_life_cycle_state not in {
        "TERMINATED",
        "SKIPPED",
        "INTERNAL_ERROR",
        "BLOCKED",
    }:
        raise ValueError("qualification task is not terminal")
    task_start = _nonnegative_int(task.get("start_time"), "task.start_time")
    task_end = _positive_int(task.get("end_time"), "task.end_time")
    if not start_time <= task_start < task_end <= end_time:
        raise ValueError("qualification task interval is not nested in the run")
    task_run_id_value = _databricks_run_id(
        task.get("run_id"), "qualification task run_id"
    )
    cluster_instance = _required_mapping(
        task.get("cluster_instance"), "task.cluster_instance"
    )
    cluster_id = _non_empty_string(
        cluster_instance.get("cluster_id"), "task cluster_id"
    )
    submitted_task = _required_mapping(payload["tasks"][0], "submitted task")
    submitted_cluster = _required_mapping(
        submitted_task.get("new_cluster"), "submitted cluster"
    )
    observed_cluster = _control_plane_launch_cluster(task)
    for field_name in (
        "node_type_id",
        "driver_node_type_id",
        "spark_version",
        "data_security_mode",
        "single_user_name",
        "num_workers",
    ):
        if observed_cluster.get(field_name) != submitted_cluster.get(field_name):
            raise ValueError(f"runs/get cluster {field_name} differs")
    expected_node_type = submitted_cluster["node_type_id"]
    if planned_job.get("hardware_id") == GPU_QUALIFICATION_GENERATION_HARDWARE_ID:
        if expected_node_type != GPU_QUALIFICATION_GENERATION_DATABRICKS_NODE_TYPE:
            raise ValueError("L40S control-plane node type differs")
    succeeded = (
        life_cycle_state == "TERMINATED"
        and result_state == "SUCCESS"
        and task_life_cycle_state == "TERMINATED"
        and task_result_state == "SUCCESS"
    )
    return {
        "cloud_cluster_id": cluster_id,
        "cloud_run_id": control_plane_run_id,
        "driver_node_type_id": str(observed_cluster["driver_node_type_id"]),
        "end_time_ms": end_time,
        "life_cycle_state": life_cycle_state,
        "node_type_id": str(observed_cluster["node_type_id"]),
        "result_state": result_state,
        "run_name": str(run["run_name"]),
        "start_time_ms": start_time,
        "succeeded": succeeded,
        "task_end_time_ms": task_end,
        "task_life_cycle_state": task_life_cycle_state,
        "task_result_state": task_result_state,
        "task_run_id": task_run_id_value,
        "task_start_time_ms": task_start,
    }


def _failed_attempt_reconciliation_entry(
    run: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    submit_receipt: Mapping[str, Any],
    evidence_file_sha256: str,
) -> dict[str, Any]:
    run_id = _databricks_run_id(run.get("run_id"), "failed runs/get run_id")
    if run_id != submit_receipt.get("cloud_run_id"):
        raise ValueError("failed runs/get run_id differs from the submit receipt")
    payload = _required_mapping(contract.get("payload"), "legacy submit payload")
    if run.get("run_name") != payload.get("run_name"):
        raise ValueError("failed runs/get run_name differs from the submit payload")
    if run.get("run_type") != "SUBMIT_RUN":
        raise ValueError("failed qualification run is not a one-time submit run")
    if run.get("repair_history") not in (None, []):
        raise ValueError("failed qualification run has repair history")
    original_attempt_run_id = run.get("original_attempt_run_id")
    if original_attempt_run_id is not None and not (
        (type(original_attempt_run_id) is int and original_attempt_run_id == 0)
        or original_attempt_run_id == "0"
    ):
        raise ValueError("failed qualification run is not attempt zero")
    state = _required_mapping(run.get("state"), "failed runs/get state")
    life_cycle_state = state.get("life_cycle_state")
    result_state = state.get("result_state")
    if life_cycle_state not in {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}:
        raise ValueError("failed qualification runs/get response is not terminal")
    if result_state == "SUCCESS":
        raise ValueError("successful qualification run cannot be failure evidence")
    tasks = run.get("tasks")
    if (
        not isinstance(tasks, list)
        or len(tasks) != 1
        or not isinstance(tasks[0], Mapping)
    ):
        raise ValueError("failed qualification runs/get must contain exactly one task")
    task = tasks[0]
    if task.get("task_key") != contract.get("task_key"):
        raise ValueError("failed qualification task_key differs")
    if type(task.get("attempt_number")) is not int or task.get("attempt_number") != 0:
        raise ValueError("failed qualification task was retried")
    task_state = _required_mapping(task.get("state"), "failed task state")
    if task_state.get("life_cycle_state") not in {
        "TERMINATED",
        "SKIPPED",
        "INTERNAL_ERROR",
    }:
        raise ValueError("failed qualification task is not terminal")
    if task_state.get("result_state") == "SUCCESS":
        raise ValueError("successful qualification task cannot be failure evidence")
    task_run_id = task.get("run_id")
    normalized_task_run_id = _databricks_run_id(
        task_run_id, "failed qualification task run_id"
    )
    if normalized_task_run_id == run_id:
        raise ValueError("failed qualification task run_id must differ from its parent")
    task_start = task.get("start_time")
    task_end = task.get("end_time")
    never_started = (task_start is None and task_end is None) or (
        type(task_start) is int
        and task_start == 0
        and type(task_end) is int
        and task_end == 0
    )
    if never_started:
        duration_seconds = 0.0
    elif (
        type(task_start) is int
        and type(task_end) is int
        and task_start >= 0
        and task_end > task_start
    ):
        duration_seconds = (task_end - task_start) / 1000.0
    else:
        raise ValueError("failed qualification task interval is invalid")
    return {
        "actual_cluster_duration_seconds": duration_seconds,
        "attempt_id": contract["reservation_attempt_id"],
        "control_plane_status_sha256": _canonical_json_sha256(run),
        "file_sha256": _required_sha256(
            evidence_file_sha256, "evidence_file_sha256"
        ),
        "job_id": contract["job_id"],
        "life_cycle_state": life_cycle_state,
        "result_state": result_state,
        "run_id": run_id,
        "task_end_time": task_end,
        "task_run_id": task_run_id,
        "task_start_time": task_start,
    }


def _failed_attempt_reconciliation_v2_entry(
    run_output: Mapping[str, Any],
    *,
    run: Mapping[str, Any],
    base_entry: Mapping[str, Any],
    contract: Mapping[str, Any],
    expected_error: str,
    evidence_file_sha256: str,
) -> dict[str, Any]:
    """Bind one child ``runs/get-output`` failure to its parent run entry."""

    _validate_failed_run_output_schema(run_output)
    error = _non_empty_string(run_output.get("error"), "runs/get-output error")
    if error != expected_error:
        raise ValueError("failed runs/get-output error differs from the reviewed cause")
    error_trace = run_output.get("error_trace")
    if not isinstance(error_trace, str) or not error_trace:
        raise ValueError("runs/get-output error_trace must be a non-empty string")
    normalized_error = _normalized_failed_run_exception_text(error)
    normalized_trace = _normalized_failed_run_exception_text(error_trace)
    if normalized_error not in normalized_trace:
        raise ValueError("runs/get-output trace does not contain the reviewed error")
    metadata = _required_mapping(run_output.get("metadata"), "run-output metadata")
    parent_run_id = _databricks_run_id(
        base_entry.get("run_id"), "failed parent run_id"
    )
    task_run_id = _databricks_run_id(
        base_entry.get("task_run_id"), "failed child run_id"
    )
    for field_name in ("job_run_id", "parent_run_id"):
        if _databricks_run_id(
            metadata.get(field_name), f"run-output metadata.{field_name}"
        ) != parent_run_id:
            raise ValueError(f"run-output metadata.{field_name} differs from parent")
    if _databricks_run_id(
        metadata.get("run_id"), "run-output metadata.run_id"
    ) != task_run_id:
        raise ValueError("run-output metadata.run_id differs from the child task")
    if metadata.get("task_key") != contract.get("task_key"):
        raise ValueError("run-output metadata.task_key differs")
    if (
        metadata.get("start_time") != base_entry.get("task_start_time")
        or metadata.get("end_time") != base_entry.get("task_end_time")
    ):
        raise ValueError("run-output metadata interval differs from the child task")
    metadata_tasks = metadata.get("tasks")
    if (
        not isinstance(metadata_tasks, list)
        or len(metadata_tasks) != 1
        or not isinstance(metadata_tasks[0], Mapping)
    ):
        raise ValueError("run-output metadata must contain exactly one task")
    metadata_task = metadata_tasks[0]
    if (
        _databricks_run_id(
            metadata_task.get("run_id"),
            "run-output metadata task.run_id",
        )
        != task_run_id
        or metadata_task.get("task_key") != contract.get("task_key")
    ):
        raise ValueError("run-output metadata task identity differs")

    raw_tasks = run.get("tasks")
    if (
        not isinstance(raw_tasks, list)
        or len(raw_tasks) != 1
        or not isinstance(raw_tasks[0], Mapping)
    ):
        raise ValueError("failed runs/get must contain exactly one task")
    observed_task = raw_tasks[0]
    for terminal_record, label in (
        (run, "failed parent run"),
        (observed_task, "failed parent task"),
        (metadata, "failed run-output metadata"),
        (metadata_task, "failed run-output metadata task"),
    ):
        status = _required_mapping(terminal_record.get("status"), f"{label} status")
        if status.get("state") != "TERMINATED":
            raise ValueError(f"{label} status is not terminal")
    submitted_payload = _required_mapping(contract.get("payload"), "submit payload")
    submitted_tasks = submitted_payload.get("tasks")
    if (
        not isinstance(submitted_tasks, list)
        or len(submitted_tasks) != 1
        or not isinstance(submitted_tasks[0], Mapping)
    ):
        raise ValueError("submitted qualification payload task closure differs")
    submitted_task = submitted_tasks[0]
    if observed_task.get("spark_python_task") != submitted_task.get(
        "spark_python_task"
    ):
        raise ValueError("failed runs/get spark_python_task differs from submission")
    observed_cluster = _control_plane_launch_cluster(observed_task)
    submitted_cluster = _required_mapping(
        submitted_task.get("new_cluster"), "submitted cluster"
    )
    if any(
        observed_cluster.get(field_name) != submitted_value
        for field_name, submitted_value in submitted_cluster.items()
    ):
        raise ValueError("failed runs/get launch cluster differs from submission")
    task_state = _required_mapping(observed_task.get("state"), "failed task state")
    entry = dict(base_entry)
    entry.update(
        {
            "run_output_error": error,
            "run_output_error_sha256": sha256(error.encode("utf-8")).hexdigest(),
            "run_output_error_trace_sha256": sha256(
                error_trace.encode("utf-8")
            ).hexdigest(),
            "run_output_file_sha256": _required_sha256(
                evidence_file_sha256,
                "run_output_file_sha256",
            ),
            "run_output_metadata_sha256": _canonical_json_sha256(metadata),
            "run_output_record_sha256": _canonical_json_sha256(run_output),
            "task_life_cycle_state": task_state.get("life_cycle_state"),
            "task_result_state": task_state.get("result_state"),
            "task_run_id": task_run_id,
        }
    )
    return entry


def _normalized_failed_run_exception_text(value: str) -> str:
    """Normalize only reviewed Databricks exception-color projections."""

    normalized = re.sub(r"\x1b\[[0-9;]*m", "", value)
    if normalized.startswith(_DATABRICKS_ANSI_RUNTIME_ERROR_HTML_PREFIX):
        normalized = (
            "RuntimeError"
            + normalized[len(_DATABRICKS_ANSI_RUNTIME_ERROR_HTML_PREFIX) :]
        )
    return normalized


def _validate_failed_run_output_schema(run_output: Mapping[str, Any]) -> None:
    """Validate the two reviewed Jobs ``runs/get-output`` response shapes.

    Historical retained evidence predates Databricks returning inline task logs.
    Current Spark Python task failures include ``logs`` and ``logs_truncated``.
    Both shapes remain exact: accepting arbitrary optional fields would let an
    unreviewed response escape the raw record/file hash closure.
    """

    observed_keys = set(run_output)
    if observed_keys not in (
        _FAILED_RUN_OUTPUT_LEGACY_KEYS,
        _FAILED_RUN_OUTPUT_LOGGED_KEYS,
    ):
        raise ValueError("failed runs/get-output response has an open schema")
    if observed_keys == _FAILED_RUN_OUTPUT_LEGACY_KEYS:
        return
    logs = run_output.get("logs")
    if type(logs) is not str:
        raise ValueError("runs/get-output logs must be an exact string")
    try:
        encoded_logs = logs.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("runs/get-output logs must be valid UTF-8") from exc
    if len(encoded_logs) > GPU_QUALIFICATION_RUN_OUTPUT_LOG_MAX_UTF8_BYTES:
        raise ValueError("runs/get-output logs exceed the UTF-8 byte cap")
    if type(run_output.get("logs_truncated")) is not bool:
        raise ValueError("runs/get-output logs_truncated must be an exact bool")


def _failed_attempt_terminal_actual(
    run: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> DatabricksClusterHourTerminalActual:
    state = _required_mapping(run.get("state"), "failed runs/get state")
    life_cycle_state = state.get("life_cycle_state")
    result_state = state.get("result_state")
    terminal_state: str
    if life_cycle_state == "SKIPPED":
        terminal_state = "skipped"
    elif life_cycle_state in {"INTERNAL_ERROR", "BLOCKED"}:
        terminal_state = "internal_error"
    else:
        terminal_states = {
            "CANCELED": "canceled",
            "EXCLUDED": "skipped",
            "FAILED": "failed",
            "TIMEDOUT": "timed_out",
            "UPSTREAM_FAILED": "failed",
        }
        resolved_terminal_state = (
            terminal_states.get(result_state)
            if isinstance(result_state, str)
            else None
        )
        if life_cycle_state != "TERMINATED" or resolved_terminal_state is None:
            raise ValueError("failed runs/get terminal state is unsupported")
        terminal_state = resolved_terminal_state
    return DatabricksClusterHourTerminalActual(
        attempt_id=str(contract["reservation_attempt_id"]),
        terminal_state=terminal_state,
        actual_cluster_duration_seconds=float(
            entry["actual_cluster_duration_seconds"]
        ),
        verification_source="direct_databricks_runs_get",
        run_id=_databricks_run_id(entry.get("run_id"), "failed run_id"),
        submit_payload_sha256=_required_sha256(
            contract.get("submit_payload_sha256"),
            "submit_payload_sha256",
        ),
        control_plane_status_sha256=_required_sha256(
            entry.get("control_plane_status_sha256"),
            "control_plane_status_sha256",
        ),
    )


def _predicted_failed_batch_terminal_ledger(
    ledger: DatabricksClusterHourLedger,
    *,
    batch_authorization: DatabricksBatchReservationAuthorization,
    contracts: Sequence[Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
) -> tuple[DatabricksClusterHourLedger, DatabricksLedgerPrefix]:
    if len(contracts) != len(runs) or len(contracts) != len(entries):
        raise ValueError("failed batch prediction requires complete parallel inputs")
    _require_failed_batch_is_current_ledger_suffix(
        ledger,
        batch_authorization=batch_authorization,
        contracts=contracts,
    )
    ordered = sorted(
        zip(contracts, runs, entries, strict=True),
        key=lambda item: str(item[0]["job_id"]),
    )
    predecessor_terminal_count = (
        batch_authorization.predecessor_prefix.terminal_actual_count
    )
    already_closed = len(ledger.terminal_actuals) - predecessor_terminal_count
    if already_closed < 0 or already_closed > len(ordered):
        raise ValueError("failed batch terminal count is outside its exact closure")
    predicted = ledger
    for index, (contract, run, entry) in enumerate(ordered):
        actual = _failed_attempt_terminal_actual(
            run,
            contract=contract,
            entry=entry,
        )
        if index < already_closed:
            if ledger.terminal_actuals[predecessor_terminal_count + index] != actual:
                raise ValueError("existing failed terminal prefix differs from evidence")
            continue
        predicted = predicted.record_terminal_actual(actual)
    terminal_prefix = _require_qualification_phase_ledger_closure(
        predicted,
        batch_authorization=batch_authorization,
        contracts=contracts,
    )
    return predicted, terminal_prefix


def _validate_failed_attempt_reconciliation_manifest(
    manifest: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    expected_entries: Sequence[Mapping[str, Any]],
) -> None:
    if set(manifest) != {
        "closed_record_sha256",
        "entries",
        "plan_sha256",
        "reason",
        "record_type",
        "schema_version",
    }:
        raise ValueError("failed-attempt reconciliation manifest has an open schema")
    observed_digest = _required_sha256(
        manifest.get("closed_record_sha256"),
        "failed-attempt reconciliation manifest.closed_record_sha256",
    )
    unsigned = dict(manifest)
    unsigned["closed_record_sha256"] = ""
    if _canonical_json_sha256(unsigned) != observed_digest:
        raise ValueError(
            "failed-attempt reconciliation manifest closed_record_sha256 mismatch"
        )
    if (
        manifest.get("record_type")
        != GPU_QUALIFICATION_FAILED_ATTEMPT_RECONCILIATION_RECORD_TYPE
        or manifest.get("schema_version") != GPU_QUALIFICATION_SCHEMA_VERSION
        or manifest.get("plan_sha256") != plan.get("closed_record_sha256")
        or manifest.get("reason")
        != GPU_QUALIFICATION_FAILED_ATTEMPT_RECONCILIATION_REASON
    ):
        raise ValueError("failed-attempt reconciliation manifest identity differs")
    entries = manifest.get("entries")
    ordered_expected = sorted(expected_entries, key=lambda item: str(item["job_id"]))
    if not isinstance(entries, list) or entries != ordered_expected:
        raise ValueError("failed-attempt reconciliation manifest entries differ")


def _validate_failed_attempt_reconciliation_v2_manifest(
    manifest: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    batch_authorization: DatabricksBatchReservationAuthorization,
    expected_entries: Sequence[Mapping[str, Any]],
    expected_reason: str,
    expected_terminal_prefix: DatabricksLedgerPrefix,
) -> None:
    if set(manifest) != {
        "closed_record_sha256",
        "entries",
        "ledger_lineage",
        "plan_sha256",
        "reason",
        "record_type",
        "schema_version",
    }:
        raise ValueError("failed-attempt reconciliation v2 manifest has an open schema")
    _require_closed_record_digest(manifest, "failed-attempt reconciliation v2 manifest")
    if (
        manifest.get("record_type")
        != GPU_QUALIFICATION_FAILED_ATTEMPT_RECONCILIATION_V2_RECORD_TYPE
        or manifest.get("schema_version")
        != GPU_QUALIFICATION_FAILED_ATTEMPT_RECONCILIATION_V2_SCHEMA_VERSION
        or manifest.get("plan_sha256") != plan.get("closed_record_sha256")
        or manifest.get("reason") != expected_reason
    ):
        raise ValueError("failed-attempt reconciliation v2 identity differs")
    entries = manifest.get("entries")
    ordered_expected = sorted(expected_entries, key=lambda item: str(item["job_id"]))
    if not isinstance(entries, list) or entries != ordered_expected:
        raise ValueError("failed-attempt reconciliation v2 entries differ")
    lineage = _required_mapping(manifest.get("ledger_lineage"), "ledger_lineage")
    if set(lineage) != {
        "predecessor_prefix",
        "producer_batch_prefix",
        "terminal_prefix",
    }:
        raise ValueError("failed-attempt reconciliation v2 lineage has an open schema")
    expected_lineage = {
        "predecessor_prefix": batch_authorization.predecessor_prefix.to_record(),
        "producer_batch_prefix": batch_authorization.batch_prefix.to_record(),
        "terminal_prefix": expected_terminal_prefix.to_record(),
    }
    if dict(lineage) != expected_lineage:
        raise ValueError("failed-attempt reconciliation v2 ledger lineage differs")


def _validate_result_submission_binding(
    result: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    submit_receipt: Mapping[str, Any],
    run_identity: Mapping[str, Any],
) -> None:
    expected = {
        "cloud_cluster_id": run_identity["cloud_cluster_id"],
        "cloud_run_id": submit_receipt["cloud_run_id"],
        "job_id": contract["job_id"],
        "output_json": contract["output_json"],
        "reservation_attempt_id": contract["reservation_attempt_id"],
        "task_key": contract["task_key"],
    }
    for field_name, expected_value in expected.items():
        if result.get(field_name) != expected_value:
            raise ValueError(f"GPU result {field_name} differs from submission")


def _terminal_receipt_record(
    *,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    submit_receipt: Mapping[str, Any],
    ledger_id: str,
    ledger_terminal_actual: DatabricksClusterHourTerminalActual,
    run: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    result: Mapping[str, Any],
    collected_at_utc: str,
    phase_batch_record_sha256: str,
) -> dict[str, Any]:
    result_bytes = (canonical_gpu_qualification_json(result) + "\n").encode("utf-8")
    control_plane_bytes = json.dumps(
        run,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    ledger_terminal_record = {
        "actual_cluster_duration_seconds": (
            ledger_terminal_actual.actual_cluster_duration_seconds
        ),
        "actual_cluster_hours": ledger_terminal_actual.actual_cluster_hours,
        "attempt_id": ledger_terminal_actual.attempt_id,
        "control_plane_status_sha256": (
            ledger_terminal_actual.control_plane_status_sha256
        ),
        "run_id": ledger_terminal_actual.run_id,
        "submit_payload_sha256": (ledger_terminal_actual.submit_payload_sha256),
        "terminal_state": ledger_terminal_actual.terminal_state,
        "verification_source": ledger_terminal_actual.verification_source,
    }
    receipt: dict[str, Any] = {
        "authorization_source": "direct_databricks_runs_get",
        "closed_record_sha256": "",
        "cloud_cluster_id": run_identity["cloud_cluster_id"],
        "cloud_run_id": run_identity["cloud_run_id"],
        "collected_at_utc": collected_at_utc,
        "control_plane_status_sha256": sha256(control_plane_bytes).hexdigest(),
        "driver_node_type_id": run_identity["driver_node_type_id"],
        "end_time_ms": run_identity["end_time_ms"],
        "job_id": contract["job_id"],
        "ledger_actual_cluster_duration_seconds": (
            ledger_terminal_actual.actual_cluster_duration_seconds
        ),
        "ledger_id": _non_empty_string(ledger_id, "ledger_id"),
        "ledger_terminal_actual_sha256": _canonical_json_sha256(ledger_terminal_record),
        "life_cycle_state": run_identity["life_cycle_state"],
        "node_type_id": run_identity["node_type_id"],
        "output_json": contract["output_json"],
        "phase_batch_record_sha256": _required_sha256(
            phase_batch_record_sha256, "phase_batch_record_sha256"
        ),
        "plan_sha256": plan["closed_record_sha256"],
        "record_type": GPU_QUALIFICATION_TERMINAL_RECEIPT_RECORD_TYPE,
        "reservation_attempt_id": contract["reservation_attempt_id"],
        "result_file_sha256": sha256(result_bytes).hexdigest(),
        "result_record_sha256": result["closed_record_sha256"],
        "result_state": run_identity["result_state"],
        "run_name": run_identity["run_name"],
        "schema_version": GPU_QUALIFICATION_SCHEMA_VERSION,
        "start_time_ms": run_identity["start_time_ms"],
        "submit_payload_sha256": contract["submit_payload_sha256"],
        "task_attempt_number": 0,
        "task_end_time_ms": run_identity["task_end_time_ms"],
        "task_key": contract["task_key"],
        "task_life_cycle_state": run_identity["task_life_cycle_state"],
        "task_max_retries": 0,
        "task_result_state": run_identity["task_result_state"],
        "task_run_id": run_identity["task_run_id"],
        "task_start_time_ms": run_identity["task_start_time_ms"],
    }
    _seal_record(receipt)
    return receipt


def _control_plane_launch_cluster(task: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = task.get("new_cluster")
    if isinstance(direct, Mapping):
        return direct
    cluster_spec = task.get("cluster_spec")
    if isinstance(cluster_spec, Mapping):
        nested = cluster_spec.get("new_cluster")
        if isinstance(nested, Mapping):
            return nested
    raise ValueError("runs/get task does not expose its launch cluster")


def _validated_single_user_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("single_user_name must be a normalized non-empty string")
    return value


def _qualification_single_user_name_from_payloads(
    submit_payloads: Sequence[Mapping[str, Any]],
) -> str:
    """Require one exact ``SINGLE_USER`` principal across the payload closure."""

    if isinstance(submit_payloads, (str, bytes, bytearray)) or not isinstance(
        submit_payloads, Sequence
    ):
        raise TypeError("submit_payloads must be a sequence")
    principals: list[str] = []
    for index, raw_payload in enumerate(submit_payloads):
        payload = _required_mapping(raw_payload, f"qualification payload {index}")
        tasks = payload.get("tasks")
        if (
            not isinstance(tasks, list)
            or len(tasks) != 1
            or not isinstance(tasks[0], Mapping)
        ):
            raise ValueError("qualification payload must contain exactly one task")
        cluster = _required_mapping(
            tasks[0].get("new_cluster"), f"qualification cluster {index}"
        )
        if cluster.get("data_security_mode") != (
            GPU_QUALIFICATION_DATABRICKS_DATA_SECURITY_MODE
        ):
            raise ValueError("qualification cluster must use SINGLE_USER")
        principals.append(
            _validated_single_user_name(cluster.get("single_user_name"))
        )
    if not principals:
        raise ValueError("qualification payload closure must be non-empty")
    if any(principal != principals[0] for principal in principals[1:]):
        raise ValueError("qualification single_user_name values drift")
    return principals[0]


def _qualification_cluster(
    *,
    hardware_id: str,
    single_user_name: str,
    custom_tags: Mapping[str, str],
) -> dict[str, Any]:
    """Build a closed single-node cluster without widening V1 serving targets."""

    principal = _validated_single_user_name(single_user_name)
    if hardware_id != GPU_QUALIFICATION_GENERATION_HARDWARE_ID:
        return build_single_node_gpu_cluster(
            DatabricksSingleNodeGPUClusterConfig(
                purpose=GPU_QUALIFICATION_DATABRICKS_PURPOSE,
                node_type_id=databricks_node_type_for_hardware_target(hardware_id),
                spark_version=DEFAULT_DATABRICKS_SPARK_VERSION,
                data_security_mode=GPU_QUALIFICATION_DATABRICKS_DATA_SECURITY_MODE,
                single_user_name=principal,
                custom_tags=custom_tags,
            )
        )
    # g6e/L40S is qualification/producer-only.  Construct this one reviewed
    # shape locally rather than registering it as a benchmark serving target.
    tags = {
        "ResourceClass": "SingleNode",
        "purpose": GPU_QUALIFICATION_DATABRICKS_PURPOSE,
        **dict(custom_tags),
    }
    return {
        "spark_version": DEFAULT_DATABRICKS_SPARK_VERSION,
        "node_type_id": GPU_QUALIFICATION_GENERATION_DATABRICKS_NODE_TYPE,
        "driver_node_type_id": GPU_QUALIFICATION_GENERATION_DATABRICKS_NODE_TYPE,
        "data_security_mode": GPU_QUALIFICATION_DATABRICKS_DATA_SECURITY_MODE,
        "single_user_name": principal,
        "num_workers": 0,
        "spark_conf": {
            "spark.master": "local[*]",
            "spark.databricks.cluster.profile": "singleNode",
        },
        "custom_tags": tags,
        "aws_attributes": {"availability": "ON_DEMAND", "zone_id": "auto"},
    }


def _legacy_uc_broken_qualification_cluster(
    *, hardware_id: str, custom_tags: Mapping[str, str]
) -> dict[str, Any]:
    """Reconstruct only the retained failed launch payloads for reconciliation."""

    if hardware_id != GPU_QUALIFICATION_GENERATION_HARDWARE_ID:
        return build_single_node_gpu_cluster(
            DatabricksSingleNodeGPUClusterConfig(
                purpose=GPU_QUALIFICATION_DATABRICKS_PURPOSE,
                node_type_id=databricks_node_type_for_hardware_target(hardware_id),
                spark_version=DEFAULT_DATABRICKS_SPARK_VERSION,
                data_security_mode="NONE",
                custom_tags=custom_tags,
            )
        )
    tags = {
        "ResourceClass": "SingleNode",
        "purpose": GPU_QUALIFICATION_DATABRICKS_PURPOSE,
        **dict(custom_tags),
    }
    return {
        "spark_version": DEFAULT_DATABRICKS_SPARK_VERSION,
        "node_type_id": GPU_QUALIFICATION_GENERATION_DATABRICKS_NODE_TYPE,
        "driver_node_type_id": GPU_QUALIFICATION_GENERATION_DATABRICKS_NODE_TYPE,
        "data_security_mode": "NONE",
        "num_workers": 0,
        "spark_conf": {
            "spark.master": "local[*]",
            "spark.databricks.cluster.profile": "singleNode",
        },
        "custom_tags": tags,
        "aws_attributes": {"availability": "ON_DEMAND", "zone_id": "auto"},
    }


def execute_gpu_qualification_job(
    *,
    plan_record: Mapping[str, Any],
    expected_plan_sha256: str,
    job_id: str,
    reservation_attempt_id: str,
    runner_uri: str,
    package_wheel_uri: str,
    patched_vllm_wheel_uri: str,
    artifact_uris: Mapping[str, str],
    artifact_sha256: Mapping[str, str],
    output_json: str | Path,
    work_dir: str | Path,
    cloud_run_id: str,
    cloud_cluster_id: str,
    sentinel_runner: GPUQualificationSentinelRunner,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Execute, validate, and exclusively publish one planned GPU result.

    The callable boundary is intentionally in-process and is not configurable
    through the command line.  Production runners pass Cachet's reviewed
    sentinel dispatcher; tests can pass a deterministic implementation without
    pretending that CPU execution is GPU qualification.
    """

    if not callable(sentinel_runner):
        raise TypeError("sentinel_runner must be callable")
    plan, pins = _validated_plan_and_pins(plan_record)
    plan_digest = _required_sha256(
        plan.get("closed_record_sha256"), "plan.closed_record_sha256"
    )
    if _required_sha256(expected_plan_sha256, "expected_plan_sha256") != plan_digest:
        raise ValueError("expected plan SHA-256 does not match the closed plan")
    normalized_job_id = _safe_id(job_id, "job_id")
    planned_job = _planned_job(plan, normalized_job_id)
    expected_attempt_id = gpu_qualification_reservation_attempt_id(
        plan_digest, normalized_job_id
    )
    if reservation_attempt_id != expected_attempt_id:
        raise ValueError("reservation_attempt_id does not match the frozen plan/job")
    uris = _validated_artifact_uris(
        artifact_uris,
        runner_uri=runner_uri,
        package_wheel_uri=package_wheel_uri,
        patched_vllm_wheel_uri=patched_vllm_wheel_uri,
    )
    observed_pin_mapping = _validated_artifact_sha256(artifact_sha256)
    if observed_pin_mapping != pins.to_record():
        raise ValueError("artifact SHA-256 arguments do not match the plan")

    normalized_output_json = _validated_result_output_json(
        output_json,
        plan_digest=plan_digest,
        job_id=normalized_job_id,
    )
    output_path = _cluster_file_path(normalized_output_json)
    local_work_dir = _validated_local_work_dir(
        work_dir,
        plan_digest=plan_digest,
        job_id=normalized_job_id,
    )
    _require_fresh_output_path(output_path)
    _create_fresh_work_dir(local_work_dir)
    try:
        source_artifact_paths = {
            key: _cluster_file_path(uri) for key, uri in uris.items()
        }
        artifact_paths = _snapshot_artifacts_to_local_work(
            source_artifact_paths,
            expected=pins.to_record(),
            snapshot_root=local_work_dir / "artifact-snapshot",
        )

        started_clock = now or _utc_now
        started_at = _utc_timestamp(started_clock())
        measurements = sentinel_runner(
            plan_record=plan,
            planned_job=planned_job,
            artifact_paths=artifact_paths,
            work_dir=local_work_dir,
        )
        if not isinstance(measurements, Mapping):
            raise TypeError("sentinel runner must return a measurement mapping")
        normalized_measurements = _json_object(measurements, "measurements")
        finished_at = _utc_timestamp(started_clock())
        if finished_at <= started_at:
            # A canonical job record requires a strict interval.  Wall-clock
            # resolution can be coarse on managed images, so obtain a later sample
            # instead of manufacturing a timestamp.
            finished_at = _utc_timestamp(started_clock())
        if finished_at <= started_at:
            raise RuntimeError("GPU sentinel timestamps did not advance")

        runtime = _observe_gpu_runtime(
            local_work_dir,
            expected_python_version=_plan_runtime_python_version(plan),
        )
        record = build_gpu_job_result(
            plan_record=plan,
            job_id=normalized_job_id,
            reservation_attempt_id=expected_attempt_id,
            task_key=_task_key(normalized_job_id),
            output_json=normalized_output_json,
            cloud_run_id=_non_empty_string(cloud_run_id, "cloud_run_id"),
            cloud_cluster_id=_non_empty_string(cloud_cluster_id, "cloud_cluster_id"),
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            nvidia_driver_version=runtime["nvidia_driver_version"],
            observed_gpu=runtime["gpu"],
            observed_gpu_compute_capability=runtime["gpu_compute_capability"],
            observed_vllm_version=runtime["vllm_version"],
            observed_torch_cuda_version=runtime["torch_cuda_version"],
            observed_artifact_sha256=observed_pin_mapping,
            measurements=normalized_measurements,
        )
        validate_gpu_job_result_record(
            record,
            plan_record=plan,
            expected_artifact_pins=pins,
        )
        # A terminal task must never leave a valid SUCCESS result behind when
        # cleanup itself fails.  Make read-only runtime directories removable,
        # complete cleanup, and only then seal the durable result.
        _remove_success_work_dir(local_work_dir)
        _write_canonical_exclusive(record, output_path)
        return record
    except BaseException:
        # Failed attempts retain node-local diagnostics and never publish a
        # canonical SUCCESS record.
        raise


def write_gpu_qualification_bootstrap_runner(path: str | Path) -> str:
    """Write the exact stdlib-only pre-install runner and return its digest."""

    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"bootstrap runner already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o750)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SCRIPT)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256


def pins_from_plan_record(
    plan_record: Mapping[str, Any],
) -> GPUQualificationArtifactPins:
    """Extract the exact artifact pins from a closed qualification plan."""

    runtime_contract = _required_mapping(
        plan_record.get("runtime_contract"), "plan.runtime_contract"
    )
    raw_pins = _required_mapping(
        runtime_contract.get("artifact_sha256"),
        "plan.runtime_contract.artifact_sha256",
    )
    if frozenset(raw_pins) != frozenset(GPU_QUALIFICATION_ARTIFACT_KEYS):
        raise ValueError("plan artifact_sha256 mapping must use the closed key set")
    return GPUQualificationArtifactPins(
        runtime_lock_sha256=_required_sha256(
            raw_pins.get("runtime_lock_sha256"), "runtime_lock_sha256"
        ),
        patched_vllm_wheel_sha256=_required_sha256(
            raw_pins.get("patched_vllm_wheel_sha256"),
            "patched_vllm_wheel_sha256",
        ),
        package_wheel_sha256=_required_sha256(
            raw_pins.get("package_wheel_sha256"), "package_wheel_sha256"
        ),
        cachet_source_tree_sha256=_required_sha256(
            raw_pins.get("cachet_source_tree_sha256"),
            "cachet_source_tree_sha256",
        ),
        runner_sha256=_required_sha256(raw_pins.get("runner_sha256"), "runner_sha256"),
        input_bundle_sha256=_required_sha256(
            raw_pins.get("input_bundle_sha256"), "input_bundle_sha256"
        ),
    )


def _validated_plan_and_pins(
    plan_record: Mapping[str, Any],
) -> tuple[dict[str, Any], GPUQualificationArtifactPins]:
    if not isinstance(plan_record, Mapping):
        raise TypeError("plan_record must be a mapping")
    plan = _json_object(plan_record, "plan_record")
    if plan.get("record_type") != GPU_QUALIFICATION_PLAN_RECORD_TYPE:
        raise ValueError("unexpected GPU qualification plan record_type")
    campaign_id = _non_empty_string(plan.get("campaign_id"), "campaign_id")
    pins = pins_from_plan_record(plan)
    if pins.runner_sha256 != GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256:
        raise ValueError(
            "plan runner_sha256 does not identify the reviewed bootstrap runner"
        )
    if pins.input_bundle_sha256 != (GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256):
        raise ValueError(
            "qualification jobs require the frozen 7ff6 publication input bundle"
        )
    if pins.patched_vllm_wheel_sha256 != GPU_QUALIFICATION_PATCHED_WHEEL_SHA256:
        raise ValueError(
            "qualification jobs require the reviewed 65120c48 patched vLLM wheel"
        )
    if pins.runtime_lock_sha256 != VLLM_RUNTIME_LOCK_SHA256:
        raise ValueError(
            "qualification jobs require the reviewed packaged runtime lock"
        )
    validate_gpu_qualification_plan_record(
        plan,
        expected_campaign_id=campaign_id,
        expected_artifact_pins=pins,
    )
    return plan, pins


def _validated_historical_qualification_plan_and_pins(
    plan_record: Mapping[str, Any],
) -> tuple[dict[str, Any], GPUQualificationArtifactPins]:
    """Validate a closed historical plan using the pins sealed inside it.

    Historical incident evidence remains replayable after the campaign opening
    advances.  The incident mutation boundary separately source-pins the whole
    plan digest; rebuilding through current campaign constants would reject
    that already-reviewed history.
    """

    if not isinstance(plan_record, Mapping):
        raise TypeError("plan_record must be a mapping")
    plan = _json_object(plan_record, "historical plan_record")
    if plan.get("record_type") != GPU_QUALIFICATION_PLAN_RECORD_TYPE:
        raise ValueError("unexpected historical GPU qualification record_type")
    if plan.get("schema_version") != GPU_QUALIFICATION_SCHEMA_VERSION:
        raise ValueError("unexpected historical GPU qualification schema_version")
    _non_empty_string(plan.get("campaign_id"), "campaign_id")
    _require_closed_record_digest(plan, "historical GPU qualification plan")
    pins = pins_from_plan_record(plan)
    if len(_planned_jobs(plan)) != PUBLICATION_CAMPAIGN_GPU_QUALIFICATION_JOBS:
        raise ValueError("historical qualification plan job closure differs")
    return plan, pins


def _validated_legacy_uc_failure_plan_and_pins(
    plan_record: Mapping[str, Any],
) -> tuple[dict[str, Any], GPUQualificationArtifactPins]:
    """Validate the one historical plan after campaign genesis has advanced."""

    if not isinstance(plan_record, Mapping):
        raise TypeError("plan_record must be a mapping")
    plan = _json_object(plan_record, "legacy plan_record")
    if (
        plan.get("record_type") != GPU_QUALIFICATION_PLAN_RECORD_TYPE
        or plan.get("closed_record_sha256")
        != GPU_QUALIFICATION_LEGACY_UC_FAILURE_PLAN_SHA256
    ):
        raise ValueError("failed-attempt reconciliation requires the exact legacy plan")
    _require_closed_record_digest(plan, "legacy qualification plan")
    pins = pins_from_plan_record(plan)
    if (
        pins.runner_sha256 != GPU_QUALIFICATION_LEGACY_UC_BROKEN_RUNNER_SHA256
        or pins.input_bundle_sha256
        != GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256
        or pins.patched_vllm_wheel_sha256
        != GPU_QUALIFICATION_PATCHED_WHEEL_SHA256
        or pins.runtime_lock_sha256
        != GPU_QUALIFICATION_LEGACY_UC_RUNTIME_LOCK_SHA256
        or len(_planned_jobs(plan)) != 14
    ):
        raise ValueError("legacy qualification plan immutable closure differs")
    return plan, pins


def _validated_artifact_uris(
    artifact_uris: Mapping[str, str],
    *,
    runner_uri: str,
    package_wheel_uri: str,
    patched_vllm_wheel_uri: str,
) -> dict[str, str]:
    if not isinstance(artifact_uris, Mapping):
        raise TypeError("artifact_uris must be a mapping")
    if frozenset(artifact_uris) != frozenset(GPU_QUALIFICATION_ARTIFACT_KEYS):
        raise ValueError("artifact_uris must use the closed artifact key set")
    normalized = {
        key: _validated_cluster_artifact_uri(artifact_uris[key], key)
        for key in GPU_QUALIFICATION_ARTIFACT_KEYS
    }
    repeated = {
        "runner_sha256": _validated_cluster_artifact_uri(runner_uri, "runner_uri"),
        "package_wheel_sha256": _validated_cluster_artifact_uri(
            package_wheel_uri, "package_wheel_uri"
        ),
        "patched_vllm_wheel_sha256": _validated_cluster_artifact_uri(
            patched_vllm_wheel_uri, "patched_vllm_wheel_uri"
        ),
    }
    for key, value in repeated.items():
        if normalized[key] != value:
            raise ValueError(f"{key} URI does not match its dedicated argument")
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("artifact URI roles must be distinct and cannot be conflated")
    return normalized


def _validated_artifact_sha256(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("artifact_sha256 must be a mapping")
    if frozenset(value) != frozenset(GPU_QUALIFICATION_ARTIFACT_KEYS):
        raise ValueError("artifact_sha256 must use the closed artifact key set")
    return {
        key: _required_sha256(value[key], f"artifact_sha256.{key}")
        for key in GPU_QUALIFICATION_ARTIFACT_KEYS
    }


def validate_gpu_qualification_submission_rejection_record(
    record: Mapping[str, Any],
    *,
    plan_record: Mapping[str, Any],
) -> None:
    """Validate one immutable, pre-run Databricks submission rejection record."""

    normalized = _json_object(record, "GPU qualification submission rejection")
    if frozenset(normalized) != _QUALIFICATION_SUBMISSION_REJECTION_KEYS:
        raise ValueError(
            "GPU qualification submission rejection does not use the closed schema"
        )
    if normalized["record_type"] != GPU_QUALIFICATION_SUBMISSION_REJECTION_RECORD_TYPE:
        raise ValueError("GPU qualification submission rejection identity differs")
    if type(normalized["schema_version"]) is not int or normalized["schema_version"] != 1:
        raise ValueError("GPU qualification submission rejection schema differs")
    _require_closed_record_digest(
        normalized,
        "GPU qualification submission rejection",
    )
    validated_plan = _json_object(plan_record, "historical GPU qualification plan")
    if (
        validated_plan.get("record_type") != GPU_QUALIFICATION_PLAN_RECORD_TYPE
        or type(validated_plan.get("schema_version")) is not int
        or validated_plan.get("schema_version") != GPU_QUALIFICATION_SCHEMA_VERSION
    ):
        raise ValueError("historical GPU qualification plan identity differs")
    _require_closed_record_digest(
        validated_plan,
        "historical GPU qualification plan",
    )
    plan_sha256 = _required_sha256(
        validated_plan.get("closed_record_sha256"),
        "plan_record.closed_record_sha256",
    )
    if normalized["plan_sha256"] != plan_sha256:
        raise ValueError("submission rejection plan SHA-256 differs")
    attempts = normalized["attempt_ids"]
    expected_attempts = [
        gpu_qualification_reservation_attempt_id(
            plan_sha256,
            _safe_id(job.get("job_id"), "cloud job ID"),
        )
        for job in _planned_jobs(validated_plan)
    ]
    if attempts != expected_attempts:
        raise ValueError("submission rejection attempt IDs differ from the plan")
    for field_name in (
        "batch_marker_file_sha256",
        "first_post_intent_file_sha256",
        "submit_payloads_file_sha256",
    ):
        _required_sha256(normalized[field_name], field_name)
    if normalized["failed_before_run_creation"] is not True:
        raise ValueError("submission rejection must precede run creation")
    http_status = _positive_int(normalized["http_status"], "http_status")
    if not 400 <= http_status < 500:
        raise ValueError("submission rejection must carry a client-error HTTP status")
    observed_bytes = _positive_int(
        normalized["observed_parameters_json_bytes"],
        "observed_parameters_json_bytes",
    )
    server_limit = _positive_int(
        normalized["server_parameters_json_limit_bytes"],
        "server_parameters_json_limit_bytes",
    )
    if observed_bytes <= server_limit:
        raise ValueError("submission rejection must exceed the server parameter limit")
    if (
        _nonnegative_int(
            normalized["remote_active_runs_observed"],
            "remote_active_runs_observed",
        )
        != 0
    ):
        raise ValueError("submission rejection must observe zero remote active runs")
    if (
        _nonnegative_int(
            normalized["reconciled_actual_gpu_seconds_per_attempt"],
            "reconciled_actual_gpu_seconds_per_attempt",
        )
        != 0
    ):
        raise ValueError("submission rejection must reconcile zero GPU seconds")
    _non_empty_string(normalized["server_reason"], "server_reason")
    _parse_utc_timestamp(normalized["rejected_at_utc"], "rejected_at_utc")


def _encode_qualification_plan_parameter(canonical_plan: str) -> str:
    if not isinstance(canonical_plan, str):
        raise TypeError("canonical_plan must be a string")
    canonical_bytes = canonical_plan.encode("utf-8")
    if len(canonical_bytes) > _QUALIFICATION_PLAN_MAX_CANONICAL_BYTES:
        raise ValueError("canonical qualification plan exceeds the decoded size cap")
    encoded = base64.urlsafe_b64encode(
        zlib.compress(canonical_bytes, level=_QUALIFICATION_PLAN_ZLIB_LEVEL)
    ).decode("ascii")
    if len(encoded) > _QUALIFICATION_PLAN_MAX_ENCODED_CHARS:
        raise ValueError("encoded qualification plan exceeds the transport size cap")
    return encoded


def _decode_qualification_plan_parameter(
    encoded_plan: str,
    *,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    expected_digest = _required_sha256(
        expected_plan_sha256,
        "expected_plan_sha256",
    )
    if not isinstance(encoded_plan, str) or not encoded_plan:
        raise ValueError("encoded qualification plan must be a non-empty string")
    if len(encoded_plan) > _QUALIFICATION_PLAN_MAX_ENCODED_CHARS:
        raise ValueError("encoded qualification plan exceeds the transport size cap")
    try:
        encoded_bytes = encoded_plan.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("encoded qualification plan must be ASCII") from exc
    try:
        compressed = base64.b64decode(
            encoded_bytes,
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError) as exc:
        raise ValueError("encoded qualification plan is not strict base64url") from exc
    if base64.urlsafe_b64encode(compressed) != encoded_bytes:
        raise ValueError("encoded qualification plan is not canonical base64url")
    decompressor = zlib.decompressobj()
    try:
        canonical_bytes = decompressor.decompress(
            compressed,
            _QUALIFICATION_PLAN_MAX_CANONICAL_BYTES + 1,
        )
        if (
            len(canonical_bytes) > _QUALIFICATION_PLAN_MAX_CANONICAL_BYTES
            or decompressor.unconsumed_tail
        ):
            raise ValueError(
                "decoded qualification plan exceeds the canonical size cap"
            )
        canonical_bytes += decompressor.flush()
    except zlib.error as exc:
        raise ValueError("encoded qualification plan is not a valid zlib stream") from exc
    if (
        len(canonical_bytes) > _QUALIFICATION_PLAN_MAX_CANONICAL_BYTES
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise ValueError("encoded qualification plan has an invalid zlib closure")
    try:
        canonical_plan = canonical_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("decoded qualification plan is not UTF-8") from exc
    try:
        decoded = json.loads(canonical_plan)
    except json.JSONDecodeError as exc:
        raise ValueError("decoded qualification plan is not JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("decoded qualification plan must contain an object")
    plan = dict(decoded)
    if canonical_gpu_qualification_json(plan) != canonical_plan:
        raise ValueError("decoded qualification plan is not canonical JSON")
    if plan.get("closed_record_sha256") != expected_digest:
        raise ValueError("decoded qualification plan SHA-256 differs from expectation")
    validated_plan, _pins = _validated_plan_and_pins(plan)
    return validated_plan


def _qualification_parameters_json_bytes(parameters: Sequence[str]) -> int:
    return len(
        json.dumps(
            list(parameters),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _require_qualification_parameters_size(parameters: Sequence[str]) -> None:
    observed_bytes = _qualification_parameters_json_bytes(parameters)
    if observed_bytes > GPU_QUALIFICATION_DATABRICKS_PARAMETERS_MAX_BYTES:
        raise ValueError(
            "qualification parameters JSON exceeds the 9500-byte safety cap: "
            f"{observed_bytes} bytes"
        )


def _runner_parameters(
    *,
    encoded_plan: str,
    plan_digest: str,
    job_id: str,
    output_json: str,
    work_dir: str,
    runner_uri: str,
    package_wheel_uri: str,
    patched_vllm_wheel_uri: str,
    artifact_uris: Mapping[str, str],
    artifact_pins: GPUQualificationArtifactPins,
    reservation_attempt_id: str,
) -> list[str]:
    parameters = [
        _QUALIFICATION_PLAN_PARAMETER_OPTION,
        encoded_plan,
        "--expected-plan-sha256",
        plan_digest,
        "--job-id",
        job_id,
        "--reservation-attempt-id",
        reservation_attempt_id,
        "--cloud-run-id",
        _DATABRICKS_RUN_ID_TEMPLATE,
        "--attempt-number",
        "0",
        "--retry-count",
        "0",
        "--runner-uri",
        runner_uri,
        "--package-wheel-uri",
        package_wheel_uri,
        "--patched-vllm-wheel-uri",
        patched_vllm_wheel_uri,
        "--output-json",
        output_json,
        "--work-dir",
        work_dir,
    ]
    pin_mapping = artifact_pins.to_record()
    for key in GPU_QUALIFICATION_ARTIFACT_KEYS:
        parameters.extend(("--artifact-uri", f"{key}={artifact_uris[key]}"))
        parameters.extend(("--artifact-sha256", f"{key}={pin_mapping[key]}"))
    _require_qualification_parameters_size(parameters)
    return parameters


def _planned_jobs(plan: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    cloud = _required_mapping(plan.get("cloud_qualification"), "cloud_qualification")
    jobs = cloud.get("jobs")
    if not isinstance(jobs, Sequence) or isinstance(jobs, (str, bytes, bytearray)):
        raise ValueError("cloud_qualification.jobs must be an array")
    normalized: list[Mapping[str, Any]] = []
    for index, job in enumerate(jobs):
        normalized.append(_required_mapping(job, f"cloud job {index}"))
    return tuple(normalized)


def _planned_job(plan: Mapping[str, Any], job_id: str) -> Mapping[str, Any]:
    matches = [job for job in _planned_jobs(plan) if job.get("job_id") == job_id]
    if len(matches) != 1:
        raise ValueError(f"job_id is not unique in the plan: {job_id!r}")
    return matches[0]


def _plan_runtime_python_version(plan: Mapping[str, Any]) -> str:
    runtime = _required_mapping(plan.get("runtime_contract"), "runtime_contract")
    platform = _required_mapping(runtime.get("platform"), "runtime platform")
    return _non_empty_string(
        platform.get("python_version"),
        "runtime platform Python version",
    )


def _observe_gpu_runtime(
    work_dir: Path,
    *,
    expected_python_version: str,
) -> dict[str, str]:
    """Read GPU identity without weakening the copied-runtime authority.

    The work tree remains under this launcher's exclusive ownership. Pre/post
    no-follow file bindings detect replacement or mutation during observation;
    they do not claim isolation from a hostile process sharing the same UID.
    """

    from document_kv_cache.vllm_smoke import (
        _attest_isolated_python,
        _isolated_python_environment,
    )

    runtime_root = work_dir / "runtime"
    runtime_python = work_dir / "runtime" / "bin" / "python"
    environment = _isolated_python_environment()
    before = _attest_isolated_python(
        runtime_root,
        expected_python_version=expected_python_version,
        environment=environment,
    )
    probe = (
        "import json,sys,torch,vllm; "
        "p=torch.cuda.get_device_properties(0); "
        "print(json.dumps({'gpu':p.name,'gpu_compute_capability':"
        "f'{p.major}.{p.minor}','torch_cuda_version':torch.version.cuda,"
        "'vllm_version':vllm.__version__,'python_executable':sys.executable,"
        "'python_prefix':sys.prefix,'python_base_prefix':sys.base_prefix,"
        "'python_implementation':sys.implementation.name,"
        "'python_version':'.'.join(map(str,sys.version_info[:3]))},"
        "sort_keys=True))"
    )
    completed = subprocess.run(
        [str(runtime_python), "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
        cwd=runtime_root,
    )
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
        cwd=runtime_root,
    ).stdout.splitlines()
    after = _attest_isolated_python(
        runtime_root,
        expected_python_version=expected_python_version,
        environment=environment,
        expected_file_binding=before.file_binding,
    )
    if after != before:
        raise RuntimeError("isolated runtime Python identity changed during observation")
    observed = json.loads(completed.stdout)
    if not isinstance(observed, dict) or set(observed) != {
        "gpu",
        "gpu_compute_capability",
        "python_base_prefix",
        "python_executable",
        "python_implementation",
        "python_prefix",
        "python_version",
        "torch_cuda_version",
        "vllm_version",
    }:
        raise RuntimeError("GPU runtime identity probe did not return its closed object")
    for field_name, expected_value in (
        ("python_executable", before.executable),
        ("python_prefix", before.prefix),
        ("python_base_prefix", before.base_prefix),
        ("python_implementation", before.python_implementation),
        ("python_version", before.python_version),
    ):
        if observed.get(field_name) != expected_value:
            raise RuntimeError(
                f"GPU runtime identity probe reported the wrong {field_name}"
            )
    driver_versions = {item.strip() for item in driver if item.strip()}
    if len(driver_versions) != 1:
        raise RuntimeError("GPU job must observe one NVIDIA driver version")
    result = {
        "gpu": _non_empty_string(observed.get("gpu"), "observed GPU"),
        "gpu_compute_capability": _non_empty_string(
            observed.get("gpu_compute_capability"),
            "observed GPU compute capability",
        ),
        "torch_cuda_version": _non_empty_string(
            observed.get("torch_cuda_version"), "observed torch CUDA version"
        ),
        "vllm_version": _non_empty_string(
            observed.get("vllm_version"), "observed vLLM version"
        ),
        "nvidia_driver_version": next(iter(driver_versions)),
    }
    return result


def _verify_artifact_files(
    artifact_paths: Mapping[str, Path], *, expected: Mapping[str, str]
) -> None:
    for key in GPU_QUALIFICATION_ARTIFACT_KEYS:
        path = artifact_paths[key]
        if key == "input_bundle_sha256":
            _verify_input_bundle_byte_closure(path, expected_sha256=expected[key])
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"artifact {key} is not one regular file: {path}")
        observed = _file_sha256(path)
        if observed != expected[key]:
            raise ValueError(
                f"artifact {key} SHA-256 mismatch: expected {expected[key]}, "
                f"found {observed}"
            )


def _snapshot_artifacts_to_local_work(
    source_paths: Mapping[str, Path],
    *,
    expected: Mapping[str, str],
    snapshot_root: Path,
) -> dict[str, Path]:
    """Materialize the durable closure once, then execute only from local bytes."""

    if snapshot_root.exists() or snapshot_root.is_symlink():
        raise FileExistsError(f"artifact snapshot already exists: {snapshot_root}")
    snapshot_root.mkdir(parents=False, exist_ok=False)
    snapshots: dict[str, Path] = {}
    for key in GPU_QUALIFICATION_ARTIFACT_KEYS:
        source = source_paths[key]
        _require_no_symlink_ancestors(
            source,
            label=f"artifact {key} source path",
            include_leaf=True,
        )
        role_root = snapshot_root / key
        if key == "input_bundle_sha256":
            if not source.is_dir() or source.is_symlink():
                raise ValueError("input bundle source must be one regular directory")
            shutil.copytree(source, role_root, symlinks=True)
            destination = role_root
        else:
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"artifact {key} source must be one regular file")
            if not source.name or source.name in {".", ".."}:
                raise ValueError(f"artifact {key} source filename is unsafe")
            role_root.mkdir()
            destination = role_root / source.name
            shutil.copyfile(source, destination, follow_symlinks=False)
        snapshots[key] = destination
    _verify_artifact_files(snapshots, expected=expected)
    _make_tree_read_only(snapshot_root)
    return snapshots


def _make_tree_read_only(root: Path) -> None:
    for current_root, directories, files in os.walk(root, followlinks=False):
        current = Path(current_root)
        for name in files:
            child = current / name
            if not child.is_symlink():
                child.chmod(
                    child.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                )
        for name in directories:
            child = current / name
            if not child.is_symlink():
                child.chmod(
                    child.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                )
    root.chmod(root.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _verify_input_bundle_byte_closure(
    path: Path,
    *,
    expected_sha256: str,
) -> str:
    """Verify the frozen directory/provenance/raw-byte closure without ML deps.

    The tokenizer-aware invariant check intentionally runs later inside the
    hash-locked isolated runtime.  This early check is stdlib-only so the
    Databricks bootstrap interpreter cannot silently supply ambient
    Transformers behavior.
    """

    expected_digest = _required_sha256(expected_sha256, "input bundle closure SHA-256")
    if not path.is_dir() or path.is_symlink():
        raise ValueError(
            "input_bundle_sha256 must identify a verified input-bundle directory"
        )
    provenance_path = path / _INPUT_PROVENANCE_FILENAME
    if not provenance_path.is_file() or provenance_path.is_symlink():
        raise ValueError("input bundle is missing its regular provenance file")
    provenance = _canonical_json_object_from_bytes(
        provenance_path.read_bytes(),
        pretty=True,
        label="input bundle provenance",
    )
    if frozenset(provenance) != _INPUT_PROVENANCE_FIELDS:
        raise ValueError("input bundle provenance does not use the closed schema")
    if provenance.get("record_type") != "cachet.main_latency_inputs":
        raise ValueError("input bundle provenance record_type is unsupported")
    if provenance.get("schema_version") != 3:
        raise ValueError("input bundle provenance schema_version is unsupported")
    if provenance.get("protocol") != _INPUT_PROTOCOL:
        raise ValueError("input bundle protocol pins do not match qualification")
    closed_digest = _required_sha256(
        provenance.get("closed_record_sha256"),
        "input bundle closed_record_sha256",
    )
    unsigned = dict(provenance)
    unsigned.pop("closed_record_sha256")
    if _canonical_json_sha256(unsigned) != closed_digest:
        raise ValueError("input bundle provenance closure digest mismatch")

    raw_outputs = provenance.get("outputs")
    if not isinstance(raw_outputs, list):
        raise ValueError("input bundle provenance outputs must be an array")
    expected_order = [
        (dataset, target, segment_count)
        for target, segment_count in _INPUT_TARGET_SEGMENT_COUNTS
        for dataset in _INPUT_DATASETS
    ]
    if len(raw_outputs) != len(expected_order):
        raise ValueError("input bundle must describe exactly twelve output shards")
    manifest: list[dict[str, str]] = []
    expected_files = {_INPUT_PROVENANCE_FILENAME}
    for index, (raw_output, expected_output) in enumerate(
        zip(raw_outputs, expected_order, strict=True)
    ):
        if not isinstance(raw_output, dict):
            raise ValueError(f"input bundle output {index} must be an object")
        if frozenset(raw_output) != _INPUT_OUTPUT_FIELDS:
            raise ValueError(
                f"input bundle output {index} does not use the closed schema"
            )
        dataset, target, segment_count = expected_output
        relative_path = f"{target}/{dataset}.jsonl"
        if (
            raw_output.get("dataset") != dataset
            or raw_output.get("input_tokens_target") != target
            or raw_output.get("segment_count") != segment_count
            or raw_output.get("relative_path") != relative_path
            or raw_output.get("record_count") != _INPUT_EXAMPLES_PER_DATASET
        ):
            raise ValueError(
                f"input bundle output {index} does not match the frozen shard layout"
            )
        byte_count = raw_output.get("byte_count")
        if type(byte_count) is not int or byte_count <= 0:
            raise ValueError(f"input bundle output {index} has invalid byte_count")
        jsonl_digest = _required_sha256(
            raw_output.get("jsonl_sha256"),
            f"input bundle output {index} jsonl_sha256",
        )
        raw_records = raw_output.get("records")
        if not isinstance(raw_records, list) or len(raw_records) != (
            _INPUT_EXAMPLES_PER_DATASET
        ):
            raise ValueError(
                f"input bundle output {index} must describe exactly 32 records"
            )
        records_digest = _required_sha256(
            raw_output.get("records_sha256"),
            f"input bundle output {index} records_sha256",
        )
        if _canonical_json_sha256(raw_records) != records_digest:
            raise ValueError(
                f"input bundle output {index} records_sha256 does not close records"
            )

        shard_path = path / PurePosixPath(relative_path)
        if not shard_path.is_file() or shard_path.is_symlink():
            raise ValueError(
                f"input bundle shard is not one regular file: {relative_path}"
            )
        shard_bytes = shard_path.read_bytes()
        if len(shard_bytes) != byte_count:
            raise ValueError(f"input bundle shard byte count mismatch: {relative_path}")
        if sha256(shard_bytes).hexdigest() != jsonl_digest:
            raise ValueError(f"input bundle shard SHA-256 mismatch: {relative_path}")
        _verify_canonical_input_jsonl(
            shard_bytes,
            dataset=dataset,
            relative_path=relative_path,
        )
        expected_files.add(relative_path)
        manifest.append({"jsonl_sha256": jsonl_digest, "relative_path": relative_path})

    outputs_digest = _required_sha256(
        provenance.get("outputs_sha256"), "input bundle outputs_sha256"
    )
    if _canonical_json_sha256(raw_outputs) != outputs_digest:
        raise ValueError("input bundle outputs_sha256 does not close outputs")
    observed_bundle_digest = _canonical_json_sha256(manifest)
    if (
        _required_sha256(provenance.get("bundle_sha256"), "input bundle bundle_sha256")
        != observed_bundle_digest
    ):
        raise ValueError("input bundle manifest digest does not match provenance")
    if observed_bundle_digest != expected_digest:
        raise ValueError(
            "input bundle closure digest mismatch: expected "
            f"{expected_digest}, found {observed_bundle_digest}"
        )
    _verify_closed_input_directory(path, expected_files=expected_files)
    return observed_bundle_digest


def _verify_canonical_input_jsonl(
    content: bytes,
    *,
    dataset: str,
    relative_path: str,
) -> None:
    if not content or not content.endswith(b"\n"):
        raise ValueError(
            f"input bundle shard is not newline-terminated: {relative_path}"
        )
    lines = content[:-1].split(b"\n")
    if len(lines) != _INPUT_EXAMPLES_PER_DATASET or any(not line for line in lines):
        raise ValueError(
            f"input bundle shard must contain exactly 32 rows: {relative_path}"
        )
    example_ids: set[str] = set()
    for row_index, line in enumerate(lines, start=1):
        record = _canonical_json_object_from_bytes(
            line,
            pretty=False,
            label=f"{relative_path} row {row_index}",
        )
        if record.get("dataset") != dataset:
            raise ValueError(f"input bundle row dataset mismatch: {relative_path}")
        example_id = record.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"input bundle row example_id is invalid: {relative_path}")
        if example_id in example_ids:
            raise ValueError(
                f"input bundle row example_id is duplicated: {relative_path}"
            )
        example_ids.add(example_id)


def _verify_closed_input_directory(path: Path, *, expected_files: set[str]) -> None:
    expected_directories = {str(target) for target, _ in _INPUT_TARGET_SEGMENT_COUNTS}
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for root, directory_names, file_names in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in directory_names:
            child = root_path / name
            if child.is_symlink():
                raise ValueError(f"input bundle contains a symlink directory: {child}")
            observed_directories.add(child.relative_to(path).as_posix())
        for name in file_names:
            child = root_path / name
            if not child.is_file() or child.is_symlink():
                raise ValueError(f"input bundle contains a non-regular file: {child}")
            observed_files.add(child.relative_to(path).as_posix())
    if observed_directories != expected_directories or observed_files != expected_files:
        raise ValueError("input bundle directory is not the closed twelve-shard layout")


def _canonical_json_object_from_bytes(
    content: bytes,
    *,
    pretty: bool,
    label: str,
) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate object key {key!r}")
            result[key] = value
        return result

    try:
        decoded = content.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    if content != _canonical_stdlib_json_bytes(value, pretty=pretty):
        raise ValueError(f"{label} is not canonically encoded")
    return value


def _canonical_stdlib_json_bytes(value: Any, *, pretty: bool) -> bytes:
    kwargs: dict[str, Any] = {"ensure_ascii": False, "sort_keys": True}
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    suffix = "\n" if pretty else ""
    return (json.dumps(value, allow_nan=False, **kwargs) + suffix).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return sha256(_canonical_stdlib_json_bytes(value, pretty=False)).hexdigest()


def _canonical_record_file_sha256(value: Mapping[str, Any]) -> str:
    content = (canonical_gpu_qualification_json(value) + "\n").encode("utf-8")
    return sha256(content).hexdigest()


def _seal_record(record: dict[str, Any]) -> None:
    if "closed_record_sha256" not in record:
        raise ValueError("sealed record is missing closed_record_sha256")
    unsigned = dict(record)
    unsigned.pop("closed_record_sha256")
    record["closed_record_sha256"] = _canonical_json_sha256(unsigned)


def _require_closed_record_digest(record: Mapping[str, Any], field_name: str) -> str:
    observed = _required_sha256(
        record.get("closed_record_sha256"),
        f"{field_name}.closed_record_sha256",
    )
    unsigned = dict(record)
    unsigned.pop("closed_record_sha256")
    if _canonical_json_sha256(unsigned) != observed:
        raise ValueError(f"{field_name} closed_record_sha256 mismatch")
    return observed


def _read_canonical_json_object_file(path: str | Path, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"{label} must be one regular file")
    return _canonical_json_object_from_record_bytes(candidate.read_bytes(), label=label)


def _read_gpu_qualification_result(
    config: DatabricksWorkspaceConfig,
    output_json: str,
    *,
    label: str,
) -> dict[str, Any]:
    if output_json.startswith("dbfs:/Volumes/"):
        content = download_databricks_volume_file_bytes(
            config,
            output_json,
            max_bytes=DATABRICKS_VOLUME_FILE_MAX_DOWNLOAD_BYTES,
        )
        record = _canonical_json_object_from_record_bytes(content, label=label)
    else:
        record = _read_canonical_json_object_file(
            _cluster_file_path(output_json), label
        )
    _require_closed_record_digest(record, label)
    return record


def _canonical_json_object_from_record_bytes(
    content: bytes,
    *,
    label: str,
) -> dict[str, Any]:
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise ValueError(f"{label} must contain one newline-terminated JSON object")
    record = _canonical_json_object_from_bytes(
        content[:-1],
        pretty=False,
        label=label,
    )
    expected = (canonical_gpu_qualification_json(record) + "\n").encode("utf-8")
    if content != expected:
        raise ValueError(f"{label} is not canonically encoded")
    return record


def _create_fresh_controller_evidence_root(value: str | Path) -> Path:
    directory = _validated_fresh_controller_evidence_root(value)
    directory.mkdir(parents=True, exist_ok=False)
    if directory.is_symlink():  # pragma: no cover - concurrent filesystem attack.
        raise ValueError("controller evidence root cannot be a symlink")
    _fsync_directory(directory)
    _fsync_directory(directory.parent)
    return directory


def _validated_fresh_controller_evidence_root(value: str | Path) -> Path:
    raw = str(value)
    directory = Path(raw)
    if (
        not directory.is_absolute()
        or directory != Path(os.path.normpath(raw))
        or directory == Path("/")
    ):
        raise ValueError("controller evidence root must be a normalized absolute path")
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"controller evidence root already exists: {directory}")
    _require_no_symlink_ancestors(
        directory, label="controller evidence root", include_leaf=True
    )
    return directory


def _validated_existing_controller_evidence_root(value: str | Path, label: str) -> Path:
    raw = str(value)
    directory = Path(raw)
    if (
        not directory.is_absolute()
        or directory != Path(os.path.normpath(raw))
        or directory == Path("/")
    ):
        raise ValueError(f"{label} must be a normalized absolute path")
    _require_no_symlink_ancestors(directory, label=label, include_leaf=True)
    if not directory.is_dir():
        raise ValueError(f"{label} must be one regular directory")
    return directory


def _validated_existing_regular_file(value: str | Path, label: str) -> Path:
    raw = str(value)
    path = Path(raw)
    if not path.is_absolute() or path != Path(os.path.normpath(raw)):
        raise ValueError(f"{label} must be a normalized absolute path")
    _require_no_symlink_ancestors(path, label=label, include_leaf=True)
    if not path.is_file():
        raise ValueError(f"{label} must be one regular file")
    return path


def _require_no_symlink_ancestors(
    path: Path, *, label: str, include_leaf: bool
) -> None:
    candidates = ((path,) if include_leaf else ()) + tuple(path.parents)
    for candidate in candidates:
        if candidate.is_symlink():
            raise ValueError(f"{label} cannot traverse a symlink: {candidate}")


def _publish_terminal_receipts_atomic(
    root: Path, receipts: Sequence[Mapping[str, Any]]
) -> None:
    """Publish only the complete terminal closure, never a partial job prefix."""

    _validated_fresh_controller_evidence_root(root)
    closure_digest = _canonical_json_sha256({"receipts": list(receipts)})
    staging = root.with_name(f".{root.name}.staging-{closure_digest[:16]}")
    staging_root = _create_fresh_controller_evidence_root(staging)
    try:
        for receipt in receipts:
            job_id = _safe_id(receipt.get("job_id"), "terminal receipt job_id")
            _write_canonical_exclusive(receipt, staging_root / f"{job_id}.json")
        _fsync_directory(staging_root)
        os.rename(staging_root, root)
        _fsync_directory(root.parent)
    except BaseException:
        if staging_root.exists() and not staging_root.is_symlink():
            shutil.rmtree(staging_root)
            _fsync_directory(staging_root.parent)
        raise


def _publish_failed_attempt_evidence_atomic(
    root: Path,
    *,
    contracts: Sequence[Mapping[str, Any]],
    parent_runs: Sequence[Mapping[str, Any]],
    run_outputs: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    """Publish a complete 2-file-per-job failure closure atomically."""

    if len(contracts) != len(parent_runs) or len(contracts) != len(run_outputs):
        raise ValueError("failed-attempt evidence publication is incomplete")
    _validated_fresh_controller_evidence_root(root)
    closure_digest = _required_sha256(
        manifest.get("closed_record_sha256"),
        "failed-attempt manifest closed_record_sha256",
    )
    staging = root.with_name(f".{root.name}.staging-{closure_digest[:16]}")
    staging_root = _create_fresh_controller_evidence_root(staging)
    try:
        for contract, run, run_output in zip(
            contracts,
            parent_runs,
            run_outputs,
            strict=True,
        ):
            job_id = _safe_id(contract.get("job_id"), "failed-attempt job_id")
            _write_canonical_exclusive(
                run,
                staging_root / f"{job_id}.runs-get.json",
            )
            _write_canonical_exclusive(
                run_output,
                staging_root / f"{job_id}.runs-get-output.json",
            )
        _write_canonical_exclusive(
            manifest,
            staging_root / "reconciliation-manifest.json",
        )
        _fsync_directory(staging_root)
        os.rename(staging_root, root)
        _fsync_directory(root.parent)
    except BaseException:
        if staging_root.exists() and not staging_root.is_symlink():
            shutil.rmtree(staging_root)
            _fsync_directory(staging_root.parent)
        raise


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_canonical_exclusive(record: Mapping[str, Any], path: Path) -> None:
    _require_no_symlink_ancestors(
        path, label="canonical output path", include_leaf=True
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_ancestors(
        path, label="canonical output path", include_leaf=True
    )
    content = (canonical_gpu_qualification_json(record) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        path.unlink(missing_ok=True)
        if path.parent.is_dir() and not path.parent.is_symlink():
            _fsync_directory(path.parent)
        raise


def _fsync_directory(path: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"directory durability target is invalid: {path}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_fresh_output_path(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"GPU qualification output already exists: {path}")


def _create_fresh_work_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(
            f"GPU qualification work directory already exists: {path}"
        )
    _require_no_symlink_ancestors(
        path,
        label="GPU qualification local-work path",
        include_leaf=True,
    )
    path.mkdir(parents=True, exist_ok=False)
    _require_no_symlink_ancestors(
        path,
        label="GPU qualification local-work path",
        include_leaf=True,
    )


def _remove_success_work_dir(path: Path) -> None:
    """Remove read-only success workspaces without weakening failure diagnostics."""

    _require_no_symlink_ancestors(
        path,
        label="GPU qualification local-work path",
        include_leaf=True,
    )
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError("successful qualification work directory is unavailable")
    for current_root, directories, files in os.walk(path, followlinks=False):
        current = Path(current_root)
        current.chmod(
            current.stat().st_mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
        )
        for name in directories:
            child = current / name
            if not child.is_symlink():
                child.chmod(
                    child.stat().st_mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
                )
        for name in files:
            child = current / name
            if not child.is_symlink():
                child.chmod(child.stat().st_mode | stat.S_IRUSR | stat.S_IWUSR)
    shutil.rmtree(path)
    if path.exists() or path.is_symlink():
        raise RuntimeError("successful qualification work directory was not removed")


def _expected_local_work_dir(plan_digest: str, job_id: str) -> Path:
    root = Path(GPU_QUALIFICATION_LOCAL_WORK_ROOT)
    if not root.is_absolute() or str(root) in {"/", "/local_disk0"}:
        raise ValueError("GPU qualification local-work root is unsafe")
    return (
        root / _required_sha256(plan_digest, "plan digest") / _safe_id(job_id, "job_id")
    )


def _validated_local_work_dir(
    value: str | Path,
    *,
    plan_digest: str,
    job_id: str,
) -> Path:
    raw = str(value)
    parsed = urlsplit(raw)
    if raw.startswith("dbfs:/") or parsed.scheme:
        raise ValueError(
            "GPU qualification work_dir must be a node-local absolute path, not a URI"
        )
    path = Path(raw)
    if not path.is_absolute() or path != Path(os.path.normpath(raw)):
        raise ValueError(
            "GPU qualification work_dir must be a normalized absolute path"
        )
    expected = _expected_local_work_dir(plan_digest, job_id)
    if path != expected:
        raise ValueError(
            "GPU qualification work_dir must equal the frozen node-local plan/job path"
        )
    return path


def _validated_cluster_artifact_uri(value: Any, field_name: str) -> str:
    raw = _non_empty_string(value, field_name)
    if raw.startswith("dbfs:/"):
        path = PurePosixPath("/", raw.removeprefix("dbfs:/").lstrip("/"))
        _reject_unsafe_parts(path, field_name)
        return "dbfs:/" + path.as_posix().lstrip("/")
    parsed = urlsplit(raw)
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
            raise ValueError(f"{field_name} has an unsupported file URI")
        path = PurePosixPath(unquote(parsed.path))
        _reject_unsafe_parts(path, field_name)
        if not _is_durable_cluster_path(path):
            raise ValueError(f"{field_name} file URI must use DBFS or a UC Volume")
        return raw
    if parsed.scheme:
        raise ValueError(
            f"{field_name} must be a dbfs:/ URI, file URI, or absolute path"
        )
    path = PurePosixPath(raw)
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be cluster-visible and absolute")
    _reject_unsafe_parts(path, field_name)
    if not _is_durable_cluster_path(path):
        raise ValueError(f"{field_name} must use DBFS or a UC Volume")
    return path.as_posix()


def _validated_output_root(value: Any) -> str:
    root = _validated_cluster_artifact_uri(value, "output_root").rstrip("/")
    if not root:
        raise ValueError("output_root must not be the filesystem root")
    if root in {"dbfs:", "file:", "/"}:
        raise ValueError("output_root must not be a broad filesystem root")
    return root


def _validated_result_output_json(
    value: str | Path,
    *,
    plan_digest: str,
    job_id: str,
) -> str:
    normalized = _validated_cluster_artifact_uri(value, "output_json")
    cluster_path = _cluster_file_path(normalized)
    expected_suffix = (
        _required_sha256(plan_digest, "plan_digest"),
        _safe_id(job_id, "job_id"),
        GPU_QUALIFICATION_OUTPUT_FILENAME,
    )
    if tuple(cluster_path.parts[-3:]) != expected_suffix:
        raise ValueError("output_json must use the frozen plan/job result path")
    return normalized


def _is_durable_cluster_path(path: PurePosixPath) -> bool:
    parts = path.parts
    return len(parts) >= 3 and (
        parts[:2] == ("/", "dbfs") or parts[:2] == ("/", "Volumes")
    )


def _cluster_file_path(value: str | Path) -> Path:
    raw = str(value)
    if raw.startswith("dbfs:/Volumes/"):
        return Path("/", raw.removeprefix("dbfs:/").lstrip("/"))
    if raw.startswith("dbfs:/"):
        return Path("/dbfs", raw.removeprefix("dbfs:/").lstrip("/"))
    parsed = urlsplit(raw)
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError("file URI authority must be empty or localhost")
        return Path(unquote(parsed.path))
    if parsed.scheme:
        raise ValueError(f"unsupported cluster artifact URI scheme: {parsed.scheme}")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError("cluster path must be absolute")
    return path


def _join_cluster_uri(root: str, *parts: str) -> str:
    suffix = "/".join(_safe_id(part, "output path component") for part in parts)
    return f"{root.rstrip('/')}/{suffix}"


def _reject_unsafe_parts(path: PurePosixPath, field_name: str) -> None:
    if not path.is_absolute() or any(
        part in {"", ".", ".."} for part in path.parts[1:]
    ):
        raise ValueError(f"{field_name} contains an unsafe path")


def _run_name(campaign_id: Any, job_id: str) -> str:
    campaign = _safe_id(campaign_id, "campaign_id")
    return f"cachet-gpu-qualification-{campaign}-{job_id}"[:4096]


def _task_key(job_id: str) -> str:
    value = "gpu_qualification_" + re.sub(r"[^a-zA-Z0-9_]", "_", job_id)
    if not value[0].isalpha() or len(value) > 100:
        raise ValueError(f"job_id cannot form a Databricks task key: {job_id!r}")
    return value


def gpu_qualification_reservation_attempt_id(plan_sha256: str, job_id: str) -> str:
    """Return the deterministic ledger attempt ID embedded in one submit payload."""

    digest = _required_sha256(plan_sha256, "plan_sha256")
    normalized_job_id = _safe_id(job_id, "job_id")
    return f"gpuq-{digest[:16]}-{normalized_job_id}"


def _safe_tag_value(value: Any) -> str:
    normalized = _safe_id(value, "tag value")
    if len(normalized) > 255:
        raise ValueError("Databricks custom tag value is too long")
    return normalized


def _safe_id(value: Any, field_name: str) -> str:
    normalized = _non_empty_string(value, field_name)
    if _SAFE_ID_RE.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} contains unsupported characters")
    return normalized


def _required_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty, trimmed string")
    return value


def _databricks_run_id(value: Any, field_name: str) -> str:
    if type(value) is int and value > 0:
        return str(value)
    if (
        isinstance(value, str)
        and len(value) <= 128
        and re.fullmatch(r"[1-9][0-9]*", value) is not None
    ):
        return value
    raise ValueError(
        f"{field_name} must be a strictly positive canonical decimal run ID"
    )


def _required_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be an object with string keys")
    return value


def _json_object(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    try:
        normalized = json.loads(
            json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite JSON object") from exc
    if not isinstance(normalized, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return normalized


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_timestamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("timestamp provider must return timezone-aware datetimes")
    normalized = value.astimezone(UTC)
    return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_utc_timestamp(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field_name} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a canonical UTC timestamp") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != datetime.min.replace(tzinfo=UTC).utcoffset()
    ):
        raise ValueError(f"{field_name} must be UTC")
    if _utc_timestamp(parsed) != value:
        raise ValueError(f"{field_name} is not canonically encoded")
    return parsed


def _nonnegative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _all_parameters(parameters: Sequence[str], option: str) -> list[str]:
    if len(parameters) % 2 != 0:
        raise ValueError("qualification parameters must contain option/value pairs")
    values: list[str] = []
    for index in range(0, len(parameters), 2):
        observed_option = parameters[index]
        observed_value = parameters[index + 1]
        if not observed_option.startswith("--") or not observed_value:
            raise ValueError(
                "qualification parameters contain an invalid option/value pair"
            )
        if observed_option == option:
            values.append(observed_value)
    return values


def _one_parameter(parameters: Sequence[str], option: str) -> str:
    values = _all_parameters(parameters, option)
    if len(values) != 1:
        raise ValueError(f"qualification parameters require exactly one {option}")
    return values[0]


def _parse_key_value_args(values: Sequence[str], *, option_name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        key, separator, value = raw.partition("=")
        if not separator or not key or not value:
            raise ValueError(f"{option_name} entries must use KEY=VALUE")
        if key in result:
            raise ValueError(f"{option_name} contains duplicate key {key!r}")
        result[key] = value
    return result


def _cloud_cluster_id() -> str:
    """Resolve the exact task-local Databricks cluster identity.

    Databricks does not guarantee that either documented cluster environment
    variable is exported to a Spark Python task.  The driver startup SparkConf
    is therefore also inspected.  Every source that is present must agree
    byte-for-byte; no source wins by precedence and no controller lookup is
    available from this worker boundary.
    """

    candidates: list[tuple[str, str]] = []
    for name in _DATABRICKS_CLUSTER_ID_ENV_NAMES:
        if name in os.environ:
            candidates.append(
                (name, _validated_cloud_cluster_id(os.environ[name], source=name))
            )
    spark_value = _spark_cloud_cluster_id()
    if spark_value is not None:
        candidates.append(
            (
                _DATABRICKS_CLUSTER_ID_SPARK_CONF_KEY,
                _validated_cloud_cluster_id(
                    spark_value,
                    source=_DATABRICKS_CLUSTER_ID_SPARK_CONF_KEY,
                ),
            )
        )
    if not candidates:
        raise RuntimeError("Databricks cluster identity is unavailable at runtime")
    values = {value for _source, value in candidates}
    if len(values) != 1:
        raise RuntimeError("Databricks cluster identity sources are ambiguous")
    return candidates[0][1]


def _spark_cloud_cluster_id() -> object | None:
    """Read the cluster ID from the active driver's startup SparkConf.

    ``None`` is the only cleanly absent value.  Import, context, and Py4J
    failures are suspicious inside a ``spark_python_task`` and must not be
    downgraded to an environment-only success.
    """

    spark_conf = _active_spark_conf()
    try:
        getter = getattr(spark_conf, "get")
        return cast(
            object | None,
            getter(_DATABRICKS_CLUSTER_ID_SPARK_CONF_KEY, None),
        )
    except Exception as exc:
        raise RuntimeError(
            "Databricks cluster identity Spark runtime lookup failed"
        ) from exc


def _active_spark_conf() -> object:
    try:
        from pyspark import SparkContext  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "Databricks cluster identity Spark runtime is unavailable"
        ) from exc
    try:
        return cast(object, SparkContext.getOrCreate().getConf())
    except Exception as exc:
        raise RuntimeError(
            "Databricks cluster identity Spark runtime lookup failed"
        ) from exc


def _validated_cloud_cluster_id(value: object, *, source: str) -> str:
    if type(value) is not str:
        raise ValueError(f"Databricks cluster identity from {source} is not a string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            f"Databricks cluster identity from {source} is not valid UTF-8"
        ) from exc
    if (
        not value
        or value.strip() != value
        or len(encoded) > _DATABRICKS_CLUSTER_ID_MAX_UTF8_BYTES
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(
            f"Databricks cluster identity from {source} is not canonical"
        )
    return value


def _builtin_sentinel_runner(
    *,
    plan_record: Mapping[str, Any],
    planned_job: Mapping[str, Any],
    artifact_paths: Mapping[str, Path],
    work_dir: Path,
) -> Mapping[str, Any]:
    """Run the reviewed GPU sentinel dispatcher packaged with Cachet.

    The dispatcher is imported lazily so local payload rendering does not
    import torch/vLLM.  It is a package-owned callable, never a CLI-provided
    factory or an externally supplied measurement JSON file.
    """

    try:
        from document_kv_cache.gpu_qualification_sentinels import (
            run_gpu_qualification_sentinel,
        )
    except ImportError as exc:  # pragma: no cover - packaging failure on GPU.
        raise RuntimeError(
            "the packaged GPU qualification sentinel dispatcher is unavailable"
        ) from exc
    return run_gpu_qualification_sentinel(
        plan_record=plan_record,
        planned_job=planned_job,
        artifact_paths=artifact_paths,
        work_dir=work_dir,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute one exact vLLM 0.27.1 GPU qualification sentinel and "
            "write its canonical first-attempt result."
        )
    )
    parser.add_argument(_QUALIFICATION_PLAN_PARAMETER_OPTION, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--reservation-attempt-id", required=True)
    parser.add_argument("--cloud-run-id", required=True)
    parser.add_argument("--attempt-number", type=int, required=True)
    parser.add_argument("--retry-count", type=int, required=True)
    parser.add_argument("--runner-uri", required=True)
    parser.add_argument("--package-wheel-uri", required=True)
    parser.add_argument("--patched-vllm-wheel-uri", required=True)
    parser.add_argument("--artifact-uri", action="append", default=[])
    parser.add_argument("--artifact-sha256", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--work-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.attempt_number != 0 or args.retry_count != 0:
        raise ValueError("GPU qualification jobs must execute on attempt zero")
    plan = _decode_qualification_plan_parameter(
        args.plan_record_zlib_base64,
        expected_plan_sha256=args.expected_plan_sha256,
    )
    artifact_uris = _parse_key_value_args(
        args.artifact_uri, option_name="--artifact-uri"
    )
    artifact_sha256 = _parse_key_value_args(
        args.artifact_sha256, option_name="--artifact-sha256"
    )
    execute_gpu_qualification_job(
        plan_record=plan,
        expected_plan_sha256=args.expected_plan_sha256,
        job_id=args.job_id,
        reservation_attempt_id=args.reservation_attempt_id,
        runner_uri=args.runner_uri,
        package_wheel_uri=args.package_wheel_uri,
        patched_vllm_wheel_uri=args.patched_vllm_wheel_uri,
        artifact_uris=artifact_uris,
        artifact_sha256=artifact_sha256,
        output_json=args.output_json,
        work_dir=args.work_dir,
        cloud_run_id=args.cloud_run_id,
        cloud_cluster_id=_cloud_cluster_id(),
        sentinel_runner=_builtin_sentinel_runner,
    )
    return 0


__all__ = [
    "GPU_QUALIFICATION_ARTIFACT_KEYS",
    "GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_ERROR",
    "GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_MANIFEST_FILE_SHA256",
    "GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_MANIFEST_SHA256",
    "GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_PLAN_SHA256",
    "GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_REASON",
    "GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_RUNNER_SHA256",
    "GPU_QUALIFICATION_BOOTSTRAP_CLUSTER_IDENTITY_FAILURE_TERMINAL_PREFIX_SHA256",
    "GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_ERROR",
    "GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_MANIFEST_SHA256",
    "GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_MANIFEST_FILE_SHA256",
    "GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_PLAN_SHA256",
    "GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_REASON",
    "GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_RUNNER_SHA256",
    "GPU_QUALIFICATION_BOOTSTRAP_FILE_GLOBAL_FAILURE_TERMINAL_PREFIX_SHA256",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_ENSUREPIP_LOG_ARGV",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_ERROR_SHA256_BY_JOB",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_ERROR_UTF8_BYTES_BY_JOB",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_EVIDENCE_TREE_FILE_COUNT",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_EVIDENCE_TREE_SHA256",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_EVIDENCE_TREE_TOTAL_BYTES",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_LEDGER_BYTES",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_LEDGER_SHA256",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_REMAINING_HOURS",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_RESERVATION_COUNT",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_SUBMISSION_RECEIPT_COUNT",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_TERMINAL_ACTUAL_COUNT",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FINAL_TERMINAL_HOURS",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FLASHINFER_ERROR_MARKERS",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_FLASHINFER_JOB_IDS",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_LAYOUT_CONFLICT_JOB_IDS",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_LAYOUT_ERROR_MARKERS",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_MANIFEST_FILE_SHA256",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_MANIFEST_SHA256",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_NEW_TERMINAL_SECONDS",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_PIP_CHECK_LOG_MARKER",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_PLAN_SHA256",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_REASON",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_RUNNER_SHA256",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_TERMINAL_PREFIX_SHA256",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_UNRESOLVED_NATIVE_ERROR_MARKERS",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_UNRESOLVED_NATIVE_JOB_IDS",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_VERSION_MISMATCH_JOB_IDS",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_VERSION_TRACE_MARKERS",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_VIRTUALENV_LOG_PREFIX",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_WORKER_MODULE_MARKER",
    "GPU_QUALIFICATION_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_WORKER_TRACE_MARKERS",
    "GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_ERROR_SHA256_BY_JOB",
    "GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_LOG_MARKER",
    "GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_MANIFEST_FILE_SHA256",
    "GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_MANIFEST_SHA256",
    "GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_NORMALIZED_ERROR_SHA256",
    "GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_PLAN_SHA256",
    "GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_REASON",
    "GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_RUNNER_SHA256",
    "GPU_QUALIFICATION_RUNTIME_LOCK_INDEX_FAILURE_TERMINAL_PREFIX_SHA256",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ENSUREPIP_LOG_ARGV",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_SHA256_BY_JOB",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_ERROR_UTF8_BYTES_BY_JOB",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_EVIDENCE_TREE_FILE_COUNT",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_EVIDENCE_TREE_SHA256",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_EVIDENCE_TREE_TOTAL_BYTES",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_MANIFEST_FILE_SHA256",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_MANIFEST_SHA256",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_NORMALIZED_OBSERVER_ERROR_SHA256",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_NORMALIZED_WORKER_ERROR_SHA256",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OBSERVER_TRACE_MARKER",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_OBSERVER_TRACE_MARKERS",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_PACKED_PAGE_ROUNDTRIP_JOB_IDS",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_PIP_CHECK_LOG_MARKER",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_PLAN_SHA256",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_REASON",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_RUNNER_SHA256",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_TERMINAL_PREFIX_SHA256",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_VIRTUALENV_LOG_PREFIX",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_WORKER_MODULE_MARKER",
    "GPU_QUALIFICATION_RUNTIME_OBSERVATION_AND_WORKER_SUBPROCESS_FAILURE_WORKER_TRACE_MARKERS",
    "GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_ERROR_SHA256_BY_JOB",
    "GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_EVIDENCE_TREE_FILE_COUNT",
    "GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_EVIDENCE_TREE_SHA256",
    "GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_EVIDENCE_TREE_TOTAL_BYTES",
    "GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_FREEZER_TRACE_MARKER",
    "GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_MANIFEST_FILE_SHA256",
    "GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_MANIFEST_SHA256",
    "GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_NORMALIZED_ERROR_SHA256",
    "GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_PIP_CHECK_LOG_MARKER",
    "GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_PLAN_SHA256",
    "GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_REASON",
    "GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_RUNNER_SHA256",
    "GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_TERMINAL_PREFIX_SHA256",
    "GPU_QUALIFICATION_SITE_PACKAGES_PATH_FAILURE_WORKER_MODULE_MARKER",
    "GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SCRIPT",
    "GPU_QUALIFICATION_BOOTSTRAP_RUNNER_SHA256",
    "GPU_QUALIFICATION_DATABRICKS_PURPOSE",
    "GPU_QUALIFICATION_DATABRICKS_DATA_SECURITY_MODE",
    "GPU_QUALIFICATION_DATABRICKS_PARAMETERS_MAX_BYTES",
    "GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS",
    "GPU_QUALIFICATION_FAILED_ATTEMPT_RECONCILIATION_RECORD_TYPE",
    "GPU_QUALIFICATION_FAILED_ATTEMPT_RECONCILIATION_V2_RECORD_TYPE",
    "GPU_QUALIFICATION_LOCAL_WORK_ROOT",
    "GPU_QUALIFICATION_OUTPUT_FILENAME",
    "GPU_QUALIFICATION_SUBMIT_RECEIPT_RECORD_TYPE",
    "GPU_QUALIFICATION_SUBMISSION_REJECTION_RECORD_TYPE",
    "GPUQualificationLaunchAuthorization",
    "GPUQualificationSentinelRunner",
    "capture_gpu_qualification_failed_attempt_evidence_v2",
    "capture_gpu_qualification_failed_attempt_evidence_v2_by_job",
    "capture_gpu_qualification_failed_attempt_evidence_v2_by_job_digest",
    "collect_gpu_qualification_evidence",
    "execute_gpu_qualification_job",
    "gpu_qualification_reservation_attempt_id",
    "main",
    "pins_from_plan_record",
    "reconcile_gpu_qualification_bootstrap_cluster_identity_failure_evidence",
    "reconcile_gpu_qualification_bootstrap_file_global_failure_evidence",
    "reconcile_gpu_qualification_failed_attempt_evidence",
    "reconcile_gpu_qualification_mixed_sentinel_and_result_validation_failure_evidence",
    "reconcile_gpu_qualification_runtime_lock_index_failure_evidence",
    "reconcile_gpu_qualification_runtime_observation_and_worker_subprocess_failure_evidence",
    "reconcile_gpu_qualification_site_packages_path_failure_evidence",
    "replay_gpu_qualification_launch_authorization",
    "render_gpu_qualification_submit_payloads",
    "resume_gpu_qualification_job_submissions",
    "require_gpu_qualification_launch_authorization",
    "submit_gpu_qualification_jobs",
    "validate_gpu_qualification_submission_rejection_record",
    "write_gpu_qualification_bootstrap_runner",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
