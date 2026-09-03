"""Additive Databricks execution boundary for GPU qualification v2.

The retained v1 renderer and executor remain unchanged.  This module binds the
parallel v2 plan to eight artifact roles, installs only the Cachet launcher in
the Databricks driver, and delegates the four-step isolated runtime creation to
the package-owned v2 sentinel dispatcher.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Final, Protocol, cast
import zlib

from document_kv_cache.databricks_resource_ledger import (
    DatabricksBatchReservationAuthorization,
    DatabricksClusterHourLedger,
    DatabricksLedgerPrefix,
    DatabricksRunAttemptReservationRequest,
    canonical_databricks_submit_payload_snapshot,
    databricks_ledger_prefix_at_counts,
    databricks_ledger_prefix_from_record,
    databricks_ledger_path_sha256,
    read_databricks_cluster_hour_ledger_json,
    record_databricks_verified_run_terminal_actual_json,
    replay_databricks_run_attempt_batch_authorization_json,
    require_databricks_batch_terminal_closure,
    record_databricks_run_submission_receipt_json,
    require_databricks_ledger_prefix,
    require_databricks_publication_batch_admission,
    reserve_databricks_run_attempt_batch_authorized_json,
)
from document_kv_cache.databricks_runs import (
    DatabricksWorkspaceConfig,
    bind_databricks_run_idempotency_token,
    get_databricks_run,
    resume_pre_reserved_databricks_run,
    submit_pre_reserved_databricks_run,
)
import document_kv_cache.gpu_qualification_databricks as databricks_v1
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_MAX_CLOUD_JOBS,
    canonical_gpu_qualification_json,
)
from document_kv_cache.gpu_qualification_v2 import (
    GPU_QUALIFICATION_V2_ARTIFACT_KEYS,
    GPU_QUALIFICATION_V2_PLAN_RECORD_TYPE,
    GPU_QUALIFICATION_V2_TERMINAL_RECEIPT_RECORD_TYPE,
    GPUQualificationArtifactPinsV2,
    _build_governed_cloud_gpu_evidence_v2,
    _build_governed_gpu_qualification_evidence_v2,
    build_gpu_job_result_v2,
    pins_from_gpu_qualification_plan_v2,
    validate_gpu_job_result_v2_record,
    validate_gpu_qualification_evidence_v2_record,
    validate_gpu_qualification_plan_v2_record,
    validate_local_preflight_evidence_v2_record,
)
from document_kv_cache.serving_env import (
    GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL,
    GPU_RUNTIME_PYTHONWARNINGS,
    gpu_runtime_warning_environment_overrides,
)


GPU_QUALIFICATION_V2_DATABRICKS_PARAMETERS_MAX_BYTES: Final = (
    databricks_v1.GPU_QUALIFICATION_DATABRICKS_PARAMETERS_MAX_BYTES
)
GPU_QUALIFICATION_V2_LOCAL_WORK_ROOT: Final = (
    "/local_disk0/cachet-vllm-0271-qualification-v2"
)
GPU_QUALIFICATION_V2_OUTPUT_FILENAME: Final = (
    databricks_v1.GPU_QUALIFICATION_OUTPUT_FILENAME
)
_PLAN_PARAMETER_OPTION: Final = "--plan-record-zlib-base64"
_PLAN_ZLIB_LEVEL: Final = 9
_PLAN_MAX_CANONICAL_BYTES: Final = 64 * 1024
_PLAN_MAX_ENCODED_CHARS: Final = GPU_QUALIFICATION_V2_DATABRICKS_PARAMETERS_MAX_BYTES
_DATABRICKS_RUN_ID_TEMPLATE: Final = "{{job.run_id}}"
_V2_BOOTSTRAP_HANDOFF_ENV: Final = (
    "_CACHET_GPU_QUALIFICATION_V2_BOOTSTRAP_HANDOFF"
)
_V2_BOOTSTRAP_HANDOFF_RECORD_TYPE: Final = (
    "cachet.vllm_0271_gpu_qualification_bootstrap_handoff.v2"
)
_V2_BOOTSTRAP_HANDOFF_SCHEMA_VERSION: Final = 2
_V2_BOOTSTRAP_HANDOFF_MAX_BYTES: Final = 4096
_V2_BOOTSTRAP_HANDOFF_MISSING: Final = object()
_V2_CHILD_REQUIRED_ENV: Final = {
    **gpu_runtime_warning_environment_overrides(),
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
}
_V2_CHILD_FORBIDDEN_ENV: Final = (
    "PYTHONHOME",
    "PYTHONPATH",
    "VIRTUAL_ENV",
)
_V2_CLUSTER_ID_MAX_UTF8_BYTES: Final = 256
_V2_CLUSTER_ID_ENV_NAMES: Final = (
    "DATABRICKS_CLUSTER_ID",
    "DB_CLUSTER_ID",
)
_V2_CLUSTER_ID_SPARK_CONF_KEY: Final = (
    "spark.databricks.clusterUsageTags.clusterId"
)
_V2_CLUSTER_ID_SOURCE_ORDER: Final = (
    *_V2_CLUSTER_ID_ENV_NAMES,
    _V2_CLUSTER_ID_SPARK_CONF_KEY,
)
_V2_BOOTSTRAP_SINGLETON_OPTIONS: Final = (
    "--plan-record-zlib-base64",
    "--expected-plan-sha256",
    "--job-id",
    "--reservation-attempt-id",
    "--cloud-run-id",
    "--attempt-number",
    "--retry-count",
    "--output-json",
    "--work-dir",
)
_V2_BOOTSTRAP_HANDOFF_KEYS: Final = frozenset(
    {
        "argv_sha256",
        "closed_record_sha256",
        "cluster_id",
        "record_type",
        "runner_sha256",
        "schema_version",
        "sources",
        "spark_checked",
    }
)

GPU_QUALIFICATION_V2_SUBMIT_RECEIPT_RECORD_TYPE: Final = (
    "cachet.vllm_0271_gpu_submit_receipt.v2"
)
GPU_QUALIFICATION_V2_PHASE_LEASE_RECORD_TYPE: Final = (
    "cachet.vllm_0271_gpu_qualification_phase_lease.v2"
)
GPU_QUALIFICATION_V2_BATCH_MARKER_RECORD_TYPE: Final = (
    "cachet.vllm_0271_gpu_qualification_batch_reserved.v2"
)
GPU_QUALIFICATION_V2_POST_INTENT_RECORD_TYPE: Final = (
    "cachet.vllm_0271_gpu_qualification_post_intent.v2"
)
GPU_QUALIFICATION_V2_EVIDENCE_FILENAME: Final = "qualification-evidence-v2.json"
_GPU_QUALIFICATION_V2_TERMINAL_RECEIPT_SUFFIX: Final = ".terminal-receipt-v2.json"
_V2_PHASE_LEASE_FILENAME: Final = "phase-lease-v2.json"
_V2_BATCH_MARKER_FILENAME: Final = "batch-reserved-v2.json"
_V2_PREFLIGHT_PATH_DOMAIN: Final = (
    "cachet.gpu_qualification_v2.local_preflight.absolute_path.v1"
)
_V2_PREFLIGHT_BINDING_KEYS: Final = frozenset(
    {
        "completed_at_utc",
        "file_sha256",
        "path_sha256",
        "record_sha256",
        "submit_payloads_sha256",
    }
)
_V2_TERMINAL_LIFE_CYCLE_STATES: Final = frozenset(
    {"TERMINATED", "SKIPPED", "INTERNAL_ERROR", "BLOCKED"}
)
_V2_TERMINAL_POLL_SECONDS: Final = 30.0
_V2_TERMINAL_WAIT_SECONDS: Final = (
    databricks_v1.GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS + 3600
)

GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SCRIPT: Final = """from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit
import zlib


_KEYS = (
    "cachet_source_tree_sha256",
    "input_bundle_sha256",
    "package_wheel_sha256",
    "patched_flashinfer_wheel_sha256",
    "patched_vllm_wheel_sha256",
    "runner_sha256",
    "runtime_closure_manifest_sha256",
    "runtime_lock_sha256",
)
_FILE_KEYS = tuple(key for key in _KEYS if key != "input_bundle_sha256")
_SINGLETON_OPTIONS = (
    "--plan-record-zlib-base64",
    "--expected-plan-sha256",
    "--job-id",
    "--reservation-attempt-id",
    "--cloud-run-id",
    "--attempt-number",
    "--retry-count",
    "--output-json",
    "--work-dir",
)
_PLAN_RECORD_TYPE = "cachet.vllm_0271_gpu_qualification_plan.v2"
_PLAN_SCHEMA_VERSION = 2
_PLAN_MAX_CANONICAL_BYTES = 64 * 1024
_PLAN_MAX_ENCODED_CHARS = 9500
_PLAN_JOB_COUNT = 14
_LOCAL_WORK_ROOT = "/local_disk0/cachet-vllm-0271-qualification-v2"
_OUTPUT_FILENAME = "gpu-job-result.json"
_MAX_PACKAGE_WHEEL_BYTES = 256 * 1024 * 1024
_HANDOFF_ENV = "_CACHET_GPU_QUALIFICATION_V2_BOOTSTRAP_HANDOFF"
_HANDOFF_RECORD_TYPE = (
    "cachet.vllm_0271_gpu_qualification_bootstrap_handoff.v2"
)
_HANDOFF_SCHEMA_VERSION = 2
_HANDOFF_MAX_BYTES = 4096
_CHILD_REQUIRED_ENV = {
    "FLASHINFER_LOGGING_LEVEL": "__GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL__",
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
    "PYTHONWARNINGS": "__GPU_RUNTIME_PYTHONWARNINGS__",
}
_CHILD_FORBIDDEN_ENV = (
    "PYTHONHOME",
    "PYTHONPATH",
    "VIRTUAL_ENV",
)
_CLUSTER_ID_MAX_UTF8_BYTES = 256
_CLUSTER_ID_ENV_NAMES = (
    "DATABRICKS_CLUSTER_ID",
    "DB_CLUSTER_ID",
)
_CLUSTER_ID_SPARK_CONF_KEY = "spark.databricks.clusterUsageTags.clusterId"
_CLUSTER_ID_SOURCE_ORDER = (*_CLUSTER_ID_ENV_NAMES, _CLUSTER_ID_SPARK_CONF_KEY)
_FIXED_PINS = {
    "input_bundle_sha256": "7ff6cf6a1553c0e844853d21de9780c75211f1be8304754da72e9cbebbd164ec",
    "patched_flashinfer_wheel_sha256": "04e032c70234e8769f5ab7e787231c339a5b5230fca5f5b0b80f1a2a0ccad6ec",
    "patched_vllm_wheel_sha256": "65120c48a9352b9eb65bab7a67090558d27af985ad366e469d3b87751073cff4",
    "runtime_closure_manifest_sha256": "c13c25a4e116f15db31e2efdbaebdd2d76418c5e4eb2f72fb2af3d8b8090e7df",
    "runtime_lock_sha256": "c4fc0e055f0838ff397012f52bd4c4f0d22426db8a5fc8faf01689510e258903",
}


def _required_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"invalid SHA-256 for {label}")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("decoded v2 plan contains a duplicate JSON key")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"decoded v2 plan contains invalid JSON constant {value!r}")


def _parse_transport(
    argv: list[str],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    if len(argv) != 2 * len(_SINGLETON_OPTIONS) + 4 * len(_KEYS):
        raise ValueError("bootstrap requires the exact 50-value transport closure")
    values = {}
    index = 0
    for option_name in _SINGLETON_OPTIONS:
        if argv[index] != option_name:
            raise ValueError(f"bootstrap expected transport option {option_name}")
        value = argv[index + 1]
        if not value:
            raise ValueError(f"bootstrap option {option_name} must not be empty")
        values[option_name] = value
        index += 2
    uris = {}
    pins = {}
    for expected_key in _KEYS:
        if argv[index] != "--artifact-uri":
            raise ValueError("bootstrap expected canonical --artifact-uri ordering")
        uri_key, separator, uri = argv[index + 1].partition("=")
        if not separator or uri_key != expected_key or not uri:
            raise ValueError("bootstrap artifact URI closure differs")
        if argv[index + 2] != "--artifact-sha256":
            raise ValueError(
                "bootstrap expected canonical --artifact-sha256 ordering"
            )
        pin_key, separator, pin = argv[index + 3].partition("=")
        if not separator or pin_key != expected_key:
            raise ValueError("bootstrap artifact SHA-256 closure differs")
        uris[expected_key] = uri
        pins[expected_key] = _required_sha256(pin, expected_key)
        index += 4
    if index != len(argv):
        raise ValueError("bootstrap transport has an unexpected trailing value")
    if len(set(uris.values())) != len(_KEYS):
        raise ValueError("bootstrap artifact URI roles must be distinct")
    return values, uris, pins


def _decode_plan(encoded_plan: str, expected_digest: str) -> dict[str, object]:
    digest = _required_sha256(expected_digest, "expected plan")
    if not encoded_plan or len(encoded_plan) > _PLAN_MAX_ENCODED_CHARS:
        raise ValueError("encoded v2 plan exceeds its transport closure")
    try:
        encoded_bytes = encoded_plan.encode("ascii")
        compressed = base64.b64decode(encoded_bytes, altchars=b"-_", validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise ValueError("encoded v2 plan is not strict base64url") from exc
    if base64.urlsafe_b64encode(compressed) != encoded_bytes:
        raise ValueError("encoded v2 plan is not canonical base64url")
    decompressor = zlib.decompressobj()
    try:
        canonical_bytes = decompressor.decompress(
            compressed, _PLAN_MAX_CANONICAL_BYTES + 1
        )
        if (
            len(canonical_bytes) > _PLAN_MAX_CANONICAL_BYTES
            or decompressor.unconsumed_tail
        ):
            raise ValueError("decoded v2 plan exceeds its size closure")
        canonical_bytes += decompressor.flush()
    except zlib.error as exc:
        raise ValueError("encoded v2 plan is not a valid zlib stream") from exc
    if (
        len(canonical_bytes) > _PLAN_MAX_CANONICAL_BYTES
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise ValueError("encoded v2 plan has an invalid zlib closure")
    try:
        canonical_plan = canonical_bytes.decode("utf-8")
        decoded = json.loads(
            canonical_plan,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("decoded v2 plan is not canonical UTF-8 JSON") from exc
    if not isinstance(decoded, dict) or _canonical_json(decoded) != canonical_plan:
        raise ValueError("decoded v2 plan is not one canonical JSON object")
    if decoded.get("record_type") != _PLAN_RECORD_TYPE:
        raise ValueError("decoded v2 plan has an unexpected record type")
    schema_version = decoded.get("schema_version")
    if type(schema_version) is not int or schema_version != _PLAN_SCHEMA_VERSION:
        raise ValueError("decoded v2 plan has an unexpected schema version")
    if decoded.get("closed_record_sha256") != digest:
        raise ValueError("decoded v2 plan SHA-256 differs from expectation")
    open_record = dict(decoded)
    open_record["closed_record_sha256"] = ""
    observed_digest = hashlib.sha256(
        _canonical_json(open_record).encode("utf-8")
    ).hexdigest()
    if observed_digest != digest:
        raise ValueError("decoded v2 plan seal differs")
    return decoded


def _planned_job(plan: dict[str, object], job_id: str) -> dict[str, object]:
    cloud = plan.get("cloud_qualification")
    if not isinstance(cloud, dict):
        raise ValueError("decoded v2 plan lacks cloud qualification")
    jobs = cloud.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != _PLAN_JOB_COUNT:
        raise ValueError("decoded v2 plan lacks its exact fourteen-job closure")
    matches = [
        job
        for job in jobs
        if isinstance(job, dict) and job.get("job_id") == job_id
    ]
    if len(matches) != 1:
        raise ValueError("bootstrap job ID is not unique in the decoded v2 plan")
    job = matches[0]
    if (
        type(job.get("attempt_number")) is not int
        or job.get("attempt_number") != 0
        or type(job.get("max_retries")) is not int
        or job.get("max_retries") != 0
    ):
        raise ValueError("decoded v2 planned job is not attempt-zero-only")
    return job


def _cluster_path(value: str) -> str:
    if value.startswith("dbfs:/Volumes/"):
        path = "/" + value.removeprefix("dbfs:/").lstrip("/")
    elif value.startswith("dbfs:/"):
        path = "/dbfs/" + value.removeprefix("dbfs:/").lstrip("/")
    else:
        parsed = urlsplit(value)
        if parsed.scheme == "file":
            if (
                parsed.netloc not in {"", "localhost"}
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("unsupported file URI authority")
            path = unquote(parsed.path)
        elif parsed.scheme:
            raise ValueError("unsupported artifact URI scheme")
        else:
            path = value
    if (
        not path.startswith("/")
        or path.startswith("//")
        or os.path.normpath(path) != path
        or any(
            ord(character) < 32 or ord(character) == 127 for character in path
        )
    ):
        raise ValueError(
            "artifact URI does not resolve to one canonical absolute path"
        )
    return path


def _validate_transport(
    argv: list[str],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    values, uris, pins = _parse_transport(argv)
    plan = _decode_plan(
        values["--plan-record-zlib-base64"],
        values["--expected-plan-sha256"],
    )
    runtime_contract = plan.get("runtime_contract")
    if not isinstance(runtime_contract, dict):
        raise ValueError("decoded v2 plan lacks its runtime contract")
    plan_pins = runtime_contract.get("artifact_sha256")
    if (
        not isinstance(plan_pins, dict)
        or tuple(plan_pins) != _KEYS
        or plan_pins != pins
    ):
        raise ValueError("bootstrap artifact pins differ from the decoded v2 plan")
    for key, expected_pin in _FIXED_PINS.items():
        if pins[key] != expected_pin:
            raise ValueError(f"bootstrap reviewed artifact pin differs for {key}")
    job_id = values["--job-id"]
    allowed_job_characters = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )
    if not job_id or any(
        character not in allowed_job_characters for character in job_id
    ):
        raise ValueError("bootstrap job ID is unsafe")
    _planned_job(plan, job_id)
    if values["--attempt-number"] != "0" or values["--retry-count"] != "0":
        raise ValueError("bootstrap transport is not attempt-zero-only")
    plan_digest = values["--expected-plan-sha256"]
    expected_attempt_id = f"gpuq-{plan_digest[:16]}-{job_id}"
    if values["--reservation-attempt-id"] != expected_attempt_id:
        raise ValueError("bootstrap reservation attempt ID differs")
    expected_work_dir = f"{_LOCAL_WORK_ROOT}/{plan_digest}/{job_id}"
    if values["--work-dir"] != expected_work_dir:
        raise ValueError("bootstrap work directory differs from the plan/job")
    output_path = _cluster_path(values["--output-json"])
    if (
        not (
            output_path.startswith("/dbfs/")
            or output_path.startswith("/Volumes/")
        )
        or tuple(output_path.split("/")[-3:])
        != (plan_digest, job_id, _OUTPUT_FILENAME)
    ):
        raise ValueError("bootstrap output path differs from the plan/job")
    paths = {key: _cluster_path(uris[key]) for key in _KEYS}
    if len(set(paths.values())) != len(_KEYS):
        raise ValueError("bootstrap artifact URI roles resolve to aliased paths")
    return values, paths, pins


def _local_staging_parent() -> str | None:
    candidate = "/local_disk0"
    if os.path.isdir(candidate) and not os.path.islink(candidate):
        return candidate
    return None


def _snapshot_package_wheel(
    source: str, expected_sha256: str
) -> tuple[str, str]:
    filename = os.path.basename(source)
    if (
        not filename.endswith(".whl")
        or filename in {"", ".", ".."}
        or any(character in filename for character in ("/", "\\\\", "\\x00"))
    ):
        raise ValueError("v2 package artifact must be one wheel filename")
    stage = tempfile.mkdtemp(
        prefix="cachet-gpuq-v2-bootstrap-", dir=_local_staging_parent()
    )
    os.chmod(stage, 0o700)
    destination = os.path.join(stage, filename)
    source_descriptor = -1
    destination_descriptor = -1
    try:
        source_descriptor = os.open(
            source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        before = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size > _MAX_PACKAGE_WHEEL_BYTES
        ):
            raise ValueError(
                "v2 package wheel source is not one bounded regular file"
            )
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            if copied > _MAX_PACKAGE_WHEEL_BYTES:
                raise ValueError("v2 package wheel exceeds its bootstrap size cap")
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise OSError(
                        "v2 package wheel snapshot write did not advance"
                    )
                view = view[written:]
        os.fsync(destination_descriptor)
        after = os.fstat(source_descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or copied != before.st_size
        ):
            raise ValueError("v2 package wheel changed while it was snapshotted")
        if digest.hexdigest() != expected_sha256:
            raise ValueError("v2 package wheel snapshot SHA-256 mismatch")
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        if source_descriptor >= 0:
            os.close(source_descriptor)
    if _sha256(destination) != expected_sha256:
        shutil.rmtree(stage, ignore_errors=True)
        raise ValueError("v2 package wheel stable snapshot differs")
    return stage, destination


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_cluster_id(value: object, *, source: str) -> str:
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
        or len(encoded) > _CLUSTER_ID_MAX_UTF8_BYTES
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(
            f"Databricks cluster identity from {source} is not canonical"
        )
    return value


def _spark_cluster_id() -> object | None:
    try:
        from pyspark import SparkContext
    except Exception as exc:
        raise RuntimeError(
            "Databricks cluster identity Spark runtime is unavailable"
        ) from exc
    try:
        spark_conf = SparkContext.getOrCreate().getConf()
        return spark_conf.get(_CLUSTER_ID_SPARK_CONF_KEY, None)
    except Exception as exc:
        raise RuntimeError(
            "Databricks cluster identity Spark runtime lookup failed"
        ) from exc


def _resolve_cluster_id(
    base_env: dict[str, str],
) -> tuple[str, tuple[str, ...]]:
    candidates = []
    for name in _CLUSTER_ID_ENV_NAMES:
        if name in base_env:
            candidates.append(
                (name, _validated_cluster_id(base_env[name], source=name))
            )
    spark_value = _spark_cluster_id()
    if spark_value is not None:
        candidates.append(
            (
                _CLUSTER_ID_SPARK_CONF_KEY,
                _validated_cluster_id(
                    spark_value,
                    source=_CLUSTER_ID_SPARK_CONF_KEY,
                ),
            )
        )
    if not candidates:
        raise RuntimeError("Databricks cluster identity is unavailable at runtime")
    values = {value for _source, value in candidates}
    if len(values) != 1:
        raise RuntimeError("Databricks cluster identity sources are ambiguous")
    return candidates[0][1], tuple(source for source, _value in candidates)


def _argv_sha256(argv: list[str]) -> str:
    return hashlib.sha256(_canonical_json(argv).encode("utf-8")).hexdigest()


def _sealed_handoff(
    argv: list[str],
    *,
    cluster_id: str,
    sources: tuple[str, ...],
    runner_sha256: str,
) -> str:
    if not sources or tuple(
        source for source in _CLUSTER_ID_SOURCE_ORDER if source in sources
    ) != sources:
        raise ValueError("Databricks cluster identity source ordering differs")
    record = {
        "argv_sha256": _argv_sha256(argv),
        "closed_record_sha256": "",
        "cluster_id": cluster_id,
        "record_type": _HANDOFF_RECORD_TYPE,
        "runner_sha256": _required_sha256(runner_sha256, "runner handoff"),
        "schema_version": _HANDOFF_SCHEMA_VERSION,
        "sources": list(sources),
        "spark_checked": True,
    }
    record["closed_record_sha256"] = hashlib.sha256(
        _canonical_json(record).encode("utf-8")
    ).hexdigest()
    sealed = _canonical_json(record)
    if len(sealed.encode("utf-8")) > _HANDOFF_MAX_BYTES:
        raise ValueError("GPU qualification v2 bootstrap handoff exceeds its size cap")
    return sealed


def _subprocess_environment(base_env: dict[str, str]) -> dict[str, str]:
    env = dict(base_env)
    for variable_name in tuple(env):
        if variable_name.upper().startswith("PIP_"):
            env.pop(variable_name)
    for variable_name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        env.pop(variable_name, None)
    env.update(
        {
            "FLASHINFER_LOGGING_LEVEL": "__GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL__",
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONWARNINGS": "__GPU_RUNTIME_PYTHONWARNINGS__",
        }
    )
    return env


def _bootstrap(
    argv: list[str], base_env: dict[str, str]
) -> tuple[list[str], str, dict[str, str]]:
    if _HANDOFF_ENV in base_env:
        raise ValueError("inherited GPU qualification v2 bootstrap handoff is forbidden")
    _values, paths, pins = _validate_transport(argv)
    input_bundle = paths["input_bundle_sha256"]
    if not os.path.isdir(input_bundle) or os.path.islink(input_bundle):
        raise ValueError("v2 input bundle must be one regular directory")
    for key in _FILE_KEYS:
        path = paths[key]
        if not os.path.isfile(path) or os.path.islink(path):
            raise ValueError(f"v2 artifact {key} must be one regular file")
        if _sha256(path) != pins[key]:
            raise ValueError(f"v2 artifact {key} SHA-256 mismatch")
    runner_path = os.path.realpath(sys._getframe().f_code.co_filename)
    observed_runner_sha256 = _sha256(runner_path)
    if observed_runner_sha256 != pins["runner_sha256"]:
        raise ValueError("GPU qualification v2 bootstrap runner SHA-256 mismatch")
    cluster_id, sources = _resolve_cluster_id(base_env)
    handoff = _sealed_handoff(
        argv,
        cluster_id=cluster_id,
        sources=sources,
        runner_sha256=observed_runner_sha256,
    )
    subprocess_env = _subprocess_environment(base_env)
    stage, package_snapshot = _snapshot_package_wheel(
        paths["package_wheel_sha256"], pins["package_wheel_sha256"]
    )
    try:
        subprocess.run(
            [
                sys.executable,
                "-P",
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-deps",
                "--only-binary",
                ":all:",
                "--force-reinstall",
                package_snapshot,
            ],
            check=True,
            cwd=stage,
            env=dict(subprocess_env),
        )
    finally:
        shutil.rmtree(stage)
    return argv, handoff, subprocess_env


_CHILD_STUB = (
    "import os,sys\\n"
    f"_cachet_required_environment = {_CHILD_REQUIRED_ENV!r}\\n"
    f"_cachet_forbidden_environment = {_CHILD_FORBIDDEN_ENV!r}\\n"
    "if any(os.environ.get(name) != expected for name, expected in "
    "_cachet_required_environment.items()):\\n"
    "    raise RuntimeError('GPU qualification v2 child lacks its exact startup environment')\\n"
    "if any(name in os.environ for name in _cachet_forbidden_environment):\\n"
    "    raise RuntimeError('GPU qualification v2 child inherited an unsafe Python path')\\n"
    "if not sys.flags.safe_path or not sys.flags.no_user_site:\\n"
    "    raise RuntimeError('GPU qualification v2 child lacks its exact Python startup flags')\\n"
    "if tuple(sys.warnoptions) != tuple("
    "_cachet_required_environment['PYTHONWARNINGS'].split(',')):\\n"
    "    raise RuntimeError('GPU qualification v2 child lacks its exact warning startup options')\\n"
    f"_cachet_handoff = os.environ.pop({_HANDOFF_ENV!r}, None)\\n"
    "from document_kv_cache.gpu_qualification_databricks_v2 import "
    "_main_from_bootstrap_handoff_v2\\n"
    "raise SystemExit(_main_from_bootstrap_handoff_v2(_cachet_handoff))"
)


def _run(argv: list[str], base_env: dict[str, str]) -> int:
    remaining, handoff, subprocess_env = _bootstrap(argv, base_env)
    safe_cwd = tempfile.mkdtemp(
        prefix="cachet-gpuq-v2-main-", dir=_local_staging_parent()
    )
    os.chmod(safe_cwd, 0o700)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-P",
                "-c",
                _CHILD_STUB,
                *remaining,
            ],
            cwd=safe_cwd,
            env={**subprocess_env, _HANDOFF_ENV: handoff},
        )
    finally:
        shutil.rmtree(safe_cwd)
    return completed.returncode


if __name__ == "__main__":
    exit_code = _run(sys.argv[1:], dict(os.environ))
    if exit_code != 0:
        raise SystemExit(exit_code)
""".replace(
    "__GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL__",
    GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL,
).replace("__GPU_RUNTIME_PYTHONWARNINGS__", GPU_RUNTIME_PYTHONWARNINGS)
GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SHA256: Final = sha256(
    GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SCRIPT.encode("utf-8")
).hexdigest()


class GPUQualificationSentinelRunnerV2(Protocol):
    """Package-owned callable returning measurements and runtime verification."""

    def __call__(
        self,
        *,
        plan_record: Mapping[str, Any],
        planned_job: Mapping[str, Any],
        artifact_paths: Mapping[str, Path],
        work_dir: Path,
    ) -> Mapping[str, Any]: ...


def render_gpu_qualification_submit_payloads_v2(
    plan_record: Mapping[str, Any],
    *,
    single_user_name: str,
    artifact_uris: Mapping[str, str],
    output_root: str,
) -> tuple[dict[str, Any], ...]:
    """Render one exact attempt-zero payload for every v2 planned job."""

    plan, pins = _validated_plan_and_pins_v2(plan_record)
    principal = databricks_v1._validated_single_user_name(single_user_name)
    uris = _validated_artifact_uris_v2(artifact_uris)
    normalized_output_root = databricks_v1._validated_output_root(output_root)
    plan_digest = databricks_v1._required_sha256(
        plan.get("closed_record_sha256"), "plan.closed_record_sha256"
    )
    jobs = databricks_v1._planned_jobs(plan)
    if not jobs or len(jobs) > GPU_QUALIFICATION_MAX_CLOUD_JOBS:
        raise ValueError("GPU qualification v2 plan has an invalid cloud job count")
    encoded_plan = _encode_plan_parameter(canonical_gpu_qualification_json(plan))
    payloads: list[dict[str, Any]] = []
    output_paths: set[str] = set()
    for planned_job in jobs:
        job_id = databricks_v1._safe_id(planned_job.get("job_id"), "planned job_id")
        hardware_id = databricks_v1._safe_id(
            planned_job.get("hardware_id"), "planned hardware_id"
        )
        if (
            planned_job.get("attempt_number") != 0
            or planned_job.get("max_retries") != 0
        ):
            raise ValueError(f"planned v2 job {job_id!r} is not attempt-zero-only")
        output_dir = databricks_v1._join_cluster_uri(
            normalized_output_root, plan_digest, job_id
        )
        output_json = databricks_v1._join_cluster_uri(
            output_dir, GPU_QUALIFICATION_V2_OUTPUT_FILENAME
        )
        work_dir = str(_expected_local_work_dir_v2(plan_digest, job_id))
        if output_json in output_paths:
            raise ValueError("GPU qualification v2 output paths must be unique")
        output_paths.add(output_json)
        attempt_id = databricks_v1.gpu_qualification_reservation_attempt_id(
            plan_digest, job_id
        )
        parameters = _runner_parameters_v2(
            encoded_plan=encoded_plan,
            plan_digest=plan_digest,
            job_id=job_id,
            reservation_attempt_id=attempt_id,
            output_json=output_json,
            work_dir=work_dir,
            artifact_uris=uris,
            artifact_pins=pins,
        )
        cluster = databricks_v1._qualification_cluster(
            hardware_id=hardware_id,
            single_user_name=principal,
            custom_tags={
                "campaign": databricks_v1._safe_tag_value(plan.get("campaign_id")),
                "job_id": job_id,
                "plan_sha256": plan_digest[:32],
                "protocol": "gpu-qualification-v2",
            },
        )
        task = {
            "task_key": databricks_v1._task_key(job_id),
            "timeout_seconds": (
                databricks_v1.GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS
            ),
            "max_retries": 0,
            "new_cluster": cluster,
            "spark_python_task": {
                "python_file": uris["runner_sha256"],
                "parameters": parameters,
            },
        }
        payloads.append(
            bind_databricks_run_idempotency_token(
                {
                    "run_name": databricks_v1._run_name(
                        plan.get("campaign_id"), job_id
                    ),
                    "timeout_seconds": (
                        databricks_v1.GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS
                    ),
                    "tasks": [task],
                },
                attempt_id=attempt_id,
            )
        )
    return tuple(payloads)


def validate_gpu_qualification_submit_payloads_v2(
    submit_payloads: Sequence[Mapping[str, Any]],
    *,
    plan_record: Mapping[str, Any],
    single_user_name: str,
    artifact_uris: Mapping[str, str],
    output_root: str,
) -> tuple[dict[str, Any], ...]:
    """Validate a complete local v2 payload contract with no cloud side effect."""

    if isinstance(submit_payloads, (str, bytes, bytearray)) or not isinstance(
        submit_payloads, Sequence
    ):
        raise TypeError("submit_payloads must be a sequence")
    observed = tuple(
        databricks_v1._json_object(payload, f"v2 submit payload {index}")
        for index, payload in enumerate(submit_payloads)
    )
    expected = render_gpu_qualification_submit_payloads_v2(
        plan_record,
        single_user_name=single_user_name,
        artifact_uris=artifact_uris,
        output_root=output_root,
    )
    observed_json = canonical_gpu_qualification_json({"payloads": list(observed)})
    expected_json = canonical_gpu_qualification_json({"payloads": list(expected)})
    if observed_json != expected_json:
        raise ValueError("GPU qualification v2 submit payload closure differs")
    return observed


def execute_gpu_qualification_job_v2(
    *,
    plan_record: Mapping[str, Any],
    expected_plan_sha256: str,
    job_id: str,
    reservation_attempt_id: str,
    artifact_uris: Mapping[str, str],
    artifact_sha256: Mapping[str, str],
    output_json: str | Path,
    work_dir: str | Path,
    cloud_run_id: str,
    cloud_cluster_id: str,
    sentinel_runner: GPUQualificationSentinelRunnerV2,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Execute, validate, and exclusively publish one planned v2 result."""

    if not callable(sentinel_runner):
        raise TypeError("sentinel_runner must be callable")
    plan, pins = _validated_plan_and_pins_v2(plan_record)
    plan_digest = databricks_v1._required_sha256(
        plan.get("closed_record_sha256"), "plan.closed_record_sha256"
    )
    if (
        databricks_v1._required_sha256(expected_plan_sha256, "expected_plan_sha256")
        != plan_digest
    ):
        raise ValueError("expected v2 plan SHA-256 does not match the closed plan")
    normalized_job_id = databricks_v1._safe_id(job_id, "job_id")
    planned_job = databricks_v1._planned_job(plan, normalized_job_id)
    expected_attempt_id = databricks_v1.gpu_qualification_reservation_attempt_id(
        plan_digest, normalized_job_id
    )
    if reservation_attempt_id != expected_attempt_id:
        raise ValueError("reservation_attempt_id does not match the v2 plan/job")
    uris = _validated_artifact_uris_v2(artifact_uris)
    observed_pins = _validated_artifact_sha256_v2(artifact_sha256)
    if observed_pins != pins.to_record():
        raise ValueError("v2 artifact SHA-256 arguments do not match the plan")
    normalized_output_json = databricks_v1._validated_result_output_json(
        output_json,
        plan_digest=plan_digest,
        job_id=normalized_job_id,
    )
    output_path = databricks_v1._cluster_file_path(normalized_output_json)
    local_work_dir = _validated_local_work_dir_v2(
        work_dir,
        plan_digest=plan_digest,
        job_id=normalized_job_id,
    )
    databricks_v1._require_fresh_output_path(output_path)
    databricks_v1._create_fresh_work_dir(local_work_dir)
    source_artifact_paths = {
        key: databricks_v1._cluster_file_path(uri) for key, uri in uris.items()
    }
    artifact_paths = _snapshot_artifacts_v2(
        source_artifact_paths,
        expected=observed_pins,
        snapshot_root=local_work_dir / "artifact-snapshot",
    )
    started_clock = now or databricks_v1._utc_now
    started_at = databricks_v1._utc_timestamp(started_clock())
    sentinel_output = sentinel_runner(
        plan_record=plan,
        planned_job=planned_job,
        artifact_paths=artifact_paths,
        work_dir=local_work_dir,
    )
    if not isinstance(sentinel_output, Mapping):
        raise TypeError("v2 sentinel runner must return one mapping")
    normalized_output = databricks_v1._json_object(
        sentinel_output, "v2 sentinel output"
    )
    if set(normalized_output) != {"measurements", "runtime_verification"}:
        raise ValueError("v2 sentinel output does not use its closed schema")
    measurements = databricks_v1._json_object(
        normalized_output["measurements"], "v2 measurements"
    )
    runtime_verification = databricks_v1._json_object(
        normalized_output["runtime_verification"], "v2 runtime verification"
    )
    finished_at = databricks_v1._utc_timestamp(started_clock())
    if finished_at <= started_at:
        finished_at = databricks_v1._utc_timestamp(started_clock())
    if finished_at <= started_at:
        raise RuntimeError("GPU sentinel v2 timestamps did not advance")
    runtime = databricks_v1._observe_gpu_runtime(
        local_work_dir,
        expected_python_version=databricks_v1._plan_runtime_python_version(plan),
    )
    record = build_gpu_job_result_v2(
        plan_record=plan,
        job_id=normalized_job_id,
        reservation_attempt_id=expected_attempt_id,
        task_key=databricks_v1._task_key(normalized_job_id),
        output_json=normalized_output_json,
        cloud_run_id=databricks_v1._non_empty_string(cloud_run_id, "cloud_run_id"),
        cloud_cluster_id=databricks_v1._non_empty_string(
            cloud_cluster_id, "cloud_cluster_id"
        ),
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        nvidia_driver_version=runtime["nvidia_driver_version"],
        observed_gpu=runtime["gpu"],
        observed_gpu_compute_capability=runtime["gpu_compute_capability"],
        observed_vllm_version=runtime["vllm_version"],
        observed_torch_cuda_version=runtime["torch_cuda_version"],
        observed_artifact_sha256=observed_pins,
        runtime_verification=runtime_verification,
        measurements=measurements,
    )
    validate_gpu_job_result_v2_record(
        record,
        plan_record=plan,
        expected_artifact_pins=pins,
    )
    databricks_v1._remove_success_work_dir(local_work_dir)
    databricks_v1._write_canonical_exclusive(record, output_path)
    return record


def write_gpu_qualification_bootstrap_runner_v2(path: str | Path) -> str:
    """Exclusively write the exact stdlib-only v2 bootstrap runner."""

    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"v2 bootstrap runner already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o750)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SCRIPT)
            stream.flush()
            os.fchmod(stream.fileno(), 0o750)
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SHA256


def _validated_plan_and_pins_v2(
    plan_record: Mapping[str, Any],
) -> tuple[dict[str, Any], GPUQualificationArtifactPinsV2]:
    if not isinstance(plan_record, Mapping):
        raise TypeError("plan_record must be a mapping")
    plan = databricks_v1._json_object(plan_record, "v2 plan_record")
    if plan.get("record_type") != GPU_QUALIFICATION_V2_PLAN_RECORD_TYPE:
        raise ValueError("unexpected GPU qualification v2 plan record_type")
    campaign_id = databricks_v1._non_empty_string(
        plan.get("campaign_id"), "campaign_id"
    )
    pins = pins_from_gpu_qualification_plan_v2(plan)
    if pins.runner_sha256 != GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SHA256:
        raise ValueError("v2 plan does not identify the reviewed bootstrap runner")
    validate_gpu_qualification_plan_v2_record(
        plan,
        expected_campaign_id=campaign_id,
        expected_artifact_pins=pins,
    )
    return plan, pins


def _validated_artifact_uris_v2(
    artifact_uris: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(artifact_uris, Mapping):
        raise TypeError("artifact_uris must be a mapping")
    if tuple(artifact_uris) != GPU_QUALIFICATION_V2_ARTIFACT_KEYS:
        raise ValueError("artifact_uris must use the canonical v2 artifact key order")
    normalized = {
        key: databricks_v1._validated_cluster_artifact_uri(artifact_uris[key], key)
        for key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS
    }
    cluster_paths = {
        key: databricks_v1._cluster_file_path(value)
        for key, value in normalized.items()
    }
    if len(set(cluster_paths.values())) != len(cluster_paths):
        raise ValueError("v2 artifact URI roles must resolve to distinct paths")
    return normalized


def _validated_artifact_sha256_v2(
    value: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("artifact_sha256 must be a mapping")
    if tuple(value) != GPU_QUALIFICATION_V2_ARTIFACT_KEYS:
        raise ValueError("artifact_sha256 must use canonical v2 artifact key order")
    return {
        key: databricks_v1._required_sha256(value[key], f"artifact_sha256.{key}")
        for key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS
    }


def _runner_parameters_v2(
    *,
    encoded_plan: str,
    plan_digest: str,
    job_id: str,
    reservation_attempt_id: str,
    output_json: str,
    work_dir: str,
    artifact_uris: Mapping[str, str],
    artifact_pins: GPUQualificationArtifactPinsV2,
) -> list[str]:
    parameters = [
        _PLAN_PARAMETER_OPTION,
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
        "--output-json",
        output_json,
        "--work-dir",
        work_dir,
    ]
    pins = artifact_pins.to_record()
    for key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS:
        parameters.extend(("--artifact-uri", f"{key}={artifact_uris[key]}"))
        parameters.extend(("--artifact-sha256", f"{key}={pins[key]}"))
    if len(parameters) != 50:
        raise RuntimeError("v2 runner argument closure differs")
    observed_bytes = databricks_v1._qualification_parameters_json_bytes(parameters)
    if observed_bytes > GPU_QUALIFICATION_V2_DATABRICKS_PARAMETERS_MAX_BYTES:
        raise ValueError(
            "v2 qualification parameters JSON exceeds the 9500-byte safety cap: "
            f"{observed_bytes} bytes"
        )
    return parameters


def _encode_plan_parameter(canonical_plan: str) -> str:
    if not isinstance(canonical_plan, str):
        raise TypeError("canonical_plan must be a string")
    canonical_bytes = canonical_plan.encode("utf-8")
    if len(canonical_bytes) > _PLAN_MAX_CANONICAL_BYTES:
        raise ValueError("canonical v2 plan exceeds the decoded size cap")
    encoded = base64.urlsafe_b64encode(
        zlib.compress(canonical_bytes, level=_PLAN_ZLIB_LEVEL)
    ).decode("ascii")
    if len(encoded) > _PLAN_MAX_ENCODED_CHARS:
        raise ValueError("encoded v2 plan exceeds the transport size cap")
    return encoded


def _decode_plan_parameter(
    encoded_plan: str,
    *,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    expected_digest = databricks_v1._required_sha256(
        expected_plan_sha256, "expected_plan_sha256"
    )
    if not isinstance(encoded_plan, str) or not encoded_plan:
        raise ValueError("encoded v2 plan must be a non-empty string")
    if len(encoded_plan) > _PLAN_MAX_ENCODED_CHARS:
        raise ValueError("encoded v2 plan exceeds the transport size cap")
    try:
        encoded_bytes = encoded_plan.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("encoded v2 plan must be ASCII") from exc
    try:
        compressed = base64.b64decode(
            encoded_bytes,
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError) as exc:
        raise ValueError("encoded v2 plan is not strict base64url") from exc
    if base64.urlsafe_b64encode(compressed) != encoded_bytes:
        raise ValueError("encoded v2 plan is not canonical base64url")
    decompressor = zlib.decompressobj()
    try:
        canonical_bytes = decompressor.decompress(
            compressed,
            _PLAN_MAX_CANONICAL_BYTES + 1,
        )
        if (
            len(canonical_bytes) > _PLAN_MAX_CANONICAL_BYTES
            or decompressor.unconsumed_tail
        ):
            raise ValueError("decoded v2 plan exceeds the canonical size cap")
        canonical_bytes += decompressor.flush()
    except zlib.error as exc:
        raise ValueError("encoded v2 plan is not a valid zlib stream") from exc
    if (
        len(canonical_bytes) > _PLAN_MAX_CANONICAL_BYTES
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise ValueError("encoded v2 plan has an invalid zlib closure")
    try:
        canonical_plan = canonical_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("decoded v2 plan is not UTF-8") from exc
    try:
        decoded = json.loads(canonical_plan)
    except json.JSONDecodeError as exc:
        raise ValueError("decoded v2 plan is not JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("decoded v2 plan must contain an object")
    plan = dict(decoded)
    if canonical_gpu_qualification_json(plan) != canonical_plan:
        raise ValueError("decoded v2 plan is not canonical JSON")
    if plan.get("closed_record_sha256") != expected_digest:
        raise ValueError("decoded v2 plan SHA-256 differs from expectation")
    validated, _pins = _validated_plan_and_pins_v2(plan)
    return validated


def _snapshot_artifacts_v2(
    source_paths: Mapping[str, Path],
    *,
    expected: Mapping[str, str],
    snapshot_root: Path,
) -> dict[str, Path]:
    if tuple(source_paths) != GPU_QUALIFICATION_V2_ARTIFACT_KEYS:
        raise ValueError("v2 source artifact paths lack canonical coverage")
    if tuple(expected) != GPU_QUALIFICATION_V2_ARTIFACT_KEYS:
        raise ValueError("v2 artifact pins lack canonical coverage")
    if snapshot_root.exists() or snapshot_root.is_symlink():
        raise FileExistsError(f"v2 artifact snapshot already exists: {snapshot_root}")
    snapshot_root.mkdir(parents=False, exist_ok=False)
    snapshots: dict[str, Path] = {}
    for key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS:
        source = source_paths[key]
        databricks_v1._require_no_symlink_ancestors(
            source,
            label=f"v2 artifact {key} source path",
            include_leaf=True,
        )
        role_root = snapshot_root / key
        if key == "input_bundle_sha256":
            if not source.is_dir() or source.is_symlink():
                raise ValueError("v2 input bundle source must be a regular directory")
            shutil.copytree(source, role_root, symlinks=True)
            destination = role_root
        else:
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"v2 artifact {key} source must be one regular file")
            if not source.name or source.name in {".", ".."}:
                raise ValueError(f"v2 artifact {key} source filename is unsafe")
            role_root.mkdir()
            destination = role_root / source.name
            shutil.copyfile(source, destination, follow_symlinks=False)
        snapshots[key] = destination
    _verify_artifacts_v2(snapshots, expected=expected)
    databricks_v1._make_tree_read_only(snapshot_root)
    return snapshots


def _verify_artifacts_v2(
    artifact_paths: Mapping[str, Path],
    *,
    expected: Mapping[str, str],
) -> None:
    for key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS:
        path = artifact_paths[key]
        if key == "input_bundle_sha256":
            databricks_v1._verify_input_bundle_byte_closure(
                path, expected_sha256=expected[key]
            )
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"v2 artifact {key} is not one regular file")
        if _file_sha256(path) != expected[key]:
            raise ValueError(f"v2 artifact {key} SHA-256 mismatch")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_local_work_dir_v2(plan_digest: str, job_id: str) -> Path:
    root = Path(GPU_QUALIFICATION_V2_LOCAL_WORK_ROOT)
    if not root.is_absolute() or str(root) in {"/", "/local_disk0"}:
        raise ValueError("GPU qualification v2 local-work root is unsafe")
    return (
        root
        / str(databricks_v1._required_sha256(plan_digest, "plan digest"))
        / str(databricks_v1._safe_id(job_id, "job_id"))
    )


def _validated_local_work_dir_v2(
    value: str | Path,
    *,
    plan_digest: str,
    job_id: str,
) -> Path:
    raw = str(value)
    path = Path(raw)
    if (
        not path.is_absolute()
        or path != Path(os.path.normpath(raw))
        or ":" in path.parts[0]
    ):
        raise ValueError("GPU qualification v2 work_dir must be normalized and local")
    expected = _expected_local_work_dir_v2(plan_digest, job_id)
    if path != expected:
        raise ValueError("GPU qualification v2 work_dir differs from the plan/job")
    return path


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute one exact GPU qualification v2 sentinel."
    )
    parser.add_argument(_PLAN_PARAMETER_OPTION, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--reservation-attempt-id", required=True)
    parser.add_argument("--cloud-run-id", required=True)
    parser.add_argument("--attempt-number", type=int, required=True)
    parser.add_argument("--retry-count", type=int, required=True)
    parser.add_argument("--artifact-uri", action="append", default=[])
    parser.add_argument("--artifact-sha256", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--work-dir", required=True)
    return parser.parse_args(argv)


def _builtin_sentinel_runner_v2(
    *,
    plan_record: Mapping[str, Any],
    planned_job: Mapping[str, Any],
    artifact_paths: Mapping[str, Path],
    work_dir: Path,
) -> Mapping[str, Any]:
    from document_kv_cache._gpu_qualification_sentinels_v2 import (
        run_gpu_qualification_sentinel_v2,
    )

    return run_gpu_qualification_sentinel_v2(
        plan_record=plan_record,
        planned_job=planned_job,
        artifact_paths=artifact_paths,
        work_dir=work_dir,
    )


def _staggered_batch_progress_v2(
    ledger: DatabricksClusterHourLedger,
    authorization: DatabricksBatchReservationAuthorization,
) -> tuple[int, int]:
    """Validate and return the exact receipt/terminal progress of one v2 batch."""

    if not isinstance(ledger, DatabricksClusterHourLedger):
        raise TypeError("v2 staggered progress requires a Databricks ledger")
    if not isinstance(authorization, DatabricksBatchReservationAuthorization):
        raise TypeError("v2 staggered progress requires atomic batch authority")
    require_databricks_ledger_prefix(ledger, authorization.batch_prefix)
    predecessor = authorization.predecessor_prefix
    member_count = len(authorization.attempt_ids)
    if member_count != 14:
        raise ValueError("v2 staggered progress requires the exact fourteen-job batch")
    if len(ledger.reservations) != authorization.batch_prefix.reservation_count:
        raise ValueError("v2 staggered batch has an unrelated reservation suffix")
    receipt_count = len(ledger.submission_receipts) - (
        predecessor.submission_receipt_count
    )
    terminal_count = len(ledger.terminal_actuals) - predecessor.terminal_actual_count
    if not 0 <= terminal_count <= receipt_count <= member_count:
        raise ValueError("v2 staggered receipt/terminal counts are not canonical")
    if receipt_count - terminal_count > 1:
        raise ValueError("v2 staggered batch has more than one active cloud run")
    receipts = ledger.submission_receipts[
        predecessor.submission_receipt_count :
    ]
    terminals = ledger.terminal_actuals[predecessor.terminal_actual_count :]
    if tuple(item.attempt_id for item in receipts) != (
        authorization.attempt_ids[:receipt_count]
    ):
        raise ValueError("v2 staggered receipt suffix is not the canonical job prefix")
    if tuple(item.submit_payload_sha256 for item in receipts) != (
        authorization.submit_payload_sha256s[:receipt_count]
    ):
        raise ValueError("v2 staggered receipt suffix payload digest drift")
    if tuple(item.attempt_id for item in terminals) != (
        authorization.attempt_ids[:terminal_count]
    ):
        raise ValueError("v2 staggered terminal suffix is not the canonical job prefix")
    if tuple(item.submit_payload_sha256 for item in terminals) != (
        authorization.submit_payload_sha256s[:terminal_count]
    ):
        raise ValueError("v2 staggered terminal suffix payload digest drift")
    return receipt_count, terminal_count


def _wait_and_record_staggered_terminal_v2(
    config: DatabricksWorkspaceConfig,
    *,
    ledger_path: str | Path,
    contract: Mapping[str, Any],
    batch_authorization: DatabricksBatchReservationAuthorization,
) -> DatabricksClusterHourLedger:
    """Wait for one receipt-bound terminal and durably close it before the next POST."""

    attempt_id = str(contract["reservation_attempt_id"])
    try:
        member_index = batch_authorization.attempt_ids.index(attempt_id)
    except ValueError as exc:
        raise ValueError("v2 terminal barrier attempt is outside the atomic batch") from exc
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    receipt_count, terminal_count = _staggered_batch_progress_v2(
        ledger, batch_authorization
    )
    if terminal_count > member_index:
        return ledger
    if receipt_count != member_index + 1 or terminal_count != member_index:
        raise ValueError("v2 terminal barrier is not the sole active batch member")
    receipt = ledger.submission_receipts[
        batch_authorization.predecessor_prefix.submission_receipt_count
        + member_index
    ]
    deadline = time.monotonic() + _V2_TERMINAL_WAIT_SECONDS
    while True:
        run = get_databricks_run(config, receipt.run_id)
        observed_run_id = databricks_v1._databricks_run_id(
            run.get("run_id"), "v2 staggered runs/get run_id"
        )
        if observed_run_id != receipt.run_id:
            raise ValueError("v2 staggered runs/get run_id differs from its receipt")
        state = databricks_v1._required_mapping(
            run.get("state"), "v2 staggered runs/get state"
        )
        life_cycle_state = state.get("life_cycle_state")
        if not isinstance(life_cycle_state, str) or not life_cycle_state:
            raise ValueError("v2 staggered runs/get lifecycle state is invalid")
        if life_cycle_state in _V2_TERMINAL_LIFE_CYCLE_STATES:
            updated = record_databricks_verified_run_terminal_actual_json(
                ledger_path,
                attempt_id=attempt_id,
                run_record=run,
            )
            updated_receipts, updated_terminals = _staggered_batch_progress_v2(
                updated, batch_authorization
            )
            if (
                updated_receipts != member_index + 1
                or updated_terminals != member_index + 1
            ):
                raise RuntimeError("v2 terminal barrier did not close its exact member")
            return updated
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"timed out waiting for v2 qualification job {contract['job_id']!r}"
            )
        time.sleep(min(_V2_TERMINAL_POLL_SECONDS, remaining))


def submit_gpu_qualification_jobs_v2(
    config: DatabricksWorkspaceConfig,
    *,
    plan_record: Mapping[str, Any],
    single_user_name: str,
    artifact_uris: Mapping[str, str],
    output_root: str,
    ledger_path: str | Path,
    expected_phase_predecessor_prefix: DatabricksLedgerPrefix,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Reserve and submit the package-rendered fourteen-job v2 batch."""

    plan, pins, payloads, contracts = _validated_controller_contract_v2(
        plan_record=plan_record,
        single_user_name=single_user_name,
        artifact_uris=artifact_uris,
        output_root=output_root,
    )
    preflight_binding, completed_at = _validated_local_preflight_binding_v2(
        local_preflight_evidence_path,
        plan=plan,
        submit_payloads=payloads,
        config=config,
        require_fresh_workspace=True,
    )
    boundary = _utc_now()
    databricks_v1._require_local_preflight_before_submission(
        completed_at,
        submission_boundary=boundary,
    )
    initial = read_databricks_cluster_hour_ledger_json(ledger_path)
    predecessor = _require_v2_ledger_predecessor(
        initial,
        plan=plan,
        ledger_path=ledger_path,
        expected_predecessor=expected_phase_predecessor_prefix,
        contracts=contracts,
        label="v2 qualification launch",
    )
    requests = _batch_requests_v2(plan, contracts)

    def validate_batch(
        live: DatabricksClusterHourLedger,
        reservations: tuple[Any, ...],
        snapshots: tuple[Mapping[str, Any], ...],
    ) -> None:
        observed_predecessor = _require_v2_ledger_predecessor(
            live,
            plan=plan,
            ledger_path=ledger_path,
            expected_predecessor=expected_phase_predecessor_prefix,
            contracts=contracts,
            label="v2 qualification batch reservation",
        )
        if observed_predecessor != predecessor:
            raise ValueError("v2 phase predecessor changed before reservation")
        if len(reservations) != len(contracts) or len(snapshots) != len(contracts):
            raise ValueError("v2 batch is not the exact fourteen jobs")
        for contract, reservation, snapshot in zip(
            contracts, reservations, snapshots, strict=True
        ):
            if (
                reservation.attempt_id != contract["reservation_attempt_id"]
                or reservation.submit_payload_sha256
                != contract["submit_payload_sha256"]
                or canonical_gpu_qualification_json(snapshot)
                != canonical_gpu_qualification_json(
                    databricks_v1._required_mapping(
                        contract.get("payload"), "v2 payload"
                    )
                )
            ):
                raise ValueError("v2 reservation changed a payload")

    receipt_root = databricks_v1._create_fresh_controller_evidence_root(
        submit_receipt_root
    )
    lease = _phase_lease_record_v2(
        plan=plan,
        pins=pins,
        ledger_path_sha256=_required_sha256_v2(
            plan.get("campaign_ledger_path_sha256"),
            "campaign_ledger_path_sha256",
        ),
        predecessor_prefix=predecessor,
        contracts=contracts,
        local_preflight_binding=preflight_binding,
    )
    lease_path = receipt_root / _V2_PHASE_LEASE_FILENAME
    databricks_v1._write_canonical_exclusive(lease, lease_path)
    try:
        _ledger, authorization = (
            reserve_databricks_run_attempt_batch_authorized_json(
                ledger_path,
                requests,
                expected_predecessor_prefix=predecessor,
                batch_validator=validate_batch,
            )
        )
    except BaseException:
        if _v2_batch_reservation_provably_absent(
            ledger_path,
            predecessor=predecessor,
        ):
            if lease_path.is_file() and not lease_path.is_symlink():
                lease_path.unlink()
                databricks_v1._fsync_directory(receipt_root)
            if receipt_root.is_dir() and not any(receipt_root.iterdir()):
                receipt_root.rmdir()
                databricks_v1._fsync_directory(receipt_root.parent)
        raise
    marker = _batch_marker_record_v2(
        plan=plan,
        lease_record=lease,
        batch_authorization=authorization,
    )
    marker_path = receipt_root / _V2_BATCH_MARKER_FILENAME
    databricks_v1._write_canonical_exclusive(marker, marker_path)
    if _staggered_batch_progress_v2(
        read_databricks_cluster_hour_ledger_json(ledger_path), authorization
    ) != (0, 0):
        raise RuntimeError("fresh v2 batch did not start at zero staggered progress")
    receipts: list[dict[str, Any]] = []
    for member_index, contract in enumerate(contracts):
        if _staggered_batch_progress_v2(
            read_databricks_cluster_hour_ledger_json(ledger_path), authorization
        ) != (member_index, member_index):
            raise ValueError("fresh v2 batch progress changed before its next POST")
        payload = databricks_v1._required_mapping(
            contract.get("payload"), "v2 payload"
        )
        attempt_id = str(contract["reservation_attempt_id"])
        intent_path = receipt_root / f"{contract['job_id']}.post-intent-v2"
        intent = _post_intent_record_v2(
            contract=contract,
            batch_authorization=authorization,
            phase_batch_record_sha256=str(marker["closed_record_sha256"]),
        )
        databricks_v1._write_canonical_exclusive(intent, intent_path)
        response = submit_pre_reserved_databricks_run(
            config,
            payload,
            ledger_path=ledger_path,
            attempt_id=attempt_id,
            batch_authorization=authorization,
        )
        ledger = record_databricks_run_submission_receipt_json(
            ledger_path,
            attempt_id=attempt_id,
            submit_response=response,
        )
        receipt = _submit_receipt_record_v2(
            plan=plan,
            contract=contract,
            ledger=ledger,
            phase_batch_record_sha256=str(marker["closed_record_sha256"]),
        )
        databricks_v1._write_canonical_exclusive(
            receipt, receipt_root / f"{contract['job_id']}.json"
        )
        intent_path.unlink()
        databricks_v1._fsync_directory(receipt_root)
        receipts.append(receipt)
        _wait_and_record_staggered_terminal_v2(
            config,
            ledger_path=ledger_path,
            contract=contract,
            batch_authorization=authorization,
        )
    expected_names = {
        _V2_PHASE_LEASE_FILENAME,
        _V2_BATCH_MARKER_FILENAME,
        *(f"{contract['job_id']}.json" for contract in contracts),
    }
    if {item.name for item in receipt_root.iterdir()} != expected_names:
        raise ValueError("submitted v2 receipt directory is not closed")
    final_ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    require_databricks_batch_terminal_closure(
        final_ledger,
        authorization,
        require_complete_current_prefix=True,
    )
    return tuple(receipts)


def resume_gpu_qualification_job_submissions_v2(
    config: DatabricksWorkspaceConfig,
    *,
    plan_record: Mapping[str, Any],
    single_user_name: str,
    artifact_uris: Mapping[str, str],
    output_root: str,
    ledger_path: str | Path,
    expected_phase_predecessor_prefix: DatabricksLedgerPrefix,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Resume only the durable package-rendered v2 submission batch."""

    plan, _pins, payloads, contracts = _validated_controller_contract_v2(
        plan_record=plan_record,
        single_user_name=single_user_name,
        artifact_uris=artifact_uris,
        output_root=output_root,
    )
    preflight_binding, completed_at = _validated_local_preflight_binding_v2(
        local_preflight_evidence_path,
        plan=plan,
        submit_payloads=payloads,
        config=config,
        require_fresh_workspace=False,
    )
    databricks_v1._require_local_preflight_before_submission(
        completed_at,
        submission_boundary=_utc_now(),
    )
    authorization, marker = _replay_batch_marker_v2(
        plan=plan,
        contracts=contracts,
        ledger_path=ledger_path,
        expected_phase_predecessor_prefix=expected_phase_predecessor_prefix,
        submit_receipt_root=submit_receipt_root,
        local_preflight_binding=preflight_binding,
    )
    root = databricks_v1._validated_existing_controller_evidence_root(
        submit_receipt_root,
        "v2 submit_receipt_root",
    )
    marker_sha256 = str(marker["closed_record_sha256"])
    receipts: list[dict[str, Any]] = []
    for contract in contracts:
        job_id = str(contract["job_id"])
        attempt_id = str(contract["reservation_attempt_id"])
        payload = databricks_v1._required_mapping(
            contract.get("payload"), "v2 payload"
        )
        receipt_path = root / f"{job_id}.json"
        intent_path = root / f"{job_id}.post-intent-v2"
        if receipt_path.exists() or receipt_path.is_symlink():
            ledger = read_databricks_cluster_hour_ledger_json(
                ledger_path
            )
            receipt = databricks_v1._read_canonical_json_object_file(
                receipt_path,
                f"v2 submit receipt {job_id}",
            )
            _validate_submit_receipt_v2(
                receipt,
                contract=contract,
                plan=plan,
                ledger=ledger,
                phase_batch_record_sha256=marker_sha256,
            )
            if intent_path.is_file() and not intent_path.is_symlink():
                intent_path.unlink()
                databricks_v1._fsync_directory(root)
            receipts.append(receipt)
            _wait_and_record_staggered_terminal_v2(
                config,
                ledger_path=ledger_path,
                contract=contract,
                batch_authorization=authorization,
            )
            continue
        expected_intent = _post_intent_record_v2(
            contract=contract,
            batch_authorization=authorization,
            phase_batch_record_sha256=marker_sha256,
        )
        if intent_path.exists() or intent_path.is_symlink():
            observed_intent = databricks_v1._read_canonical_json_object_file(
                intent_path,
                f"v2 post intent {job_id}",
            )
            if canonical_gpu_qualification_json(
                observed_intent
            ) != canonical_gpu_qualification_json(expected_intent):
                raise ValueError(f"v2 post intent {job_id!r} drift")
        else:
            databricks_v1._write_canonical_exclusive(
                expected_intent, intent_path
            )
        resume_pre_reserved_databricks_run(
            config,
            payload,
            ledger_path=ledger_path,
            attempt_id=attempt_id,
            batch_authorization=authorization,
        )
        ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
        receipt = _submit_receipt_record_v2(
            plan=plan,
            contract=contract,
            ledger=ledger,
            phase_batch_record_sha256=marker_sha256,
        )
        try:
            databricks_v1._write_canonical_exclusive(receipt, receipt_path)
        except FileExistsError:
            observed_receipt = databricks_v1._read_canonical_json_object_file(
                receipt_path,
                f"v2 submit receipt {job_id}",
            )
            _validate_submit_receipt_v2(
                observed_receipt,
                contract=contract,
                plan=plan,
                ledger=ledger,
                phase_batch_record_sha256=marker_sha256,
            )
            receipt = observed_receipt
        if intent_path.is_file() and not intent_path.is_symlink():
            intent_path.unlink()
            databricks_v1._fsync_directory(root)
        receipts.append(receipt)
        _wait_and_record_staggered_terminal_v2(
            config,
            ledger_path=ledger_path,
            contract=contract,
            batch_authorization=authorization,
        )
    expected_names = {
        _V2_PHASE_LEASE_FILENAME,
        _V2_BATCH_MARKER_FILENAME,
        *(f"{contract['job_id']}.json" for contract in contracts),
    }
    if {item.name for item in root.iterdir()} != expected_names:
        raise ValueError("resumed v2 receipt directory is not closed")
    final_ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    require_databricks_batch_terminal_closure(
        final_ledger,
        authorization,
        require_complete_current_prefix=True,
    )
    return tuple(receipts)


def _validated_controller_contract_v2(
    *,
    plan_record: Mapping[str, Any],
    single_user_name: str,
    artifact_uris: Mapping[str, str],
    output_root: str,
) -> tuple[
    dict[str, Any],
    GPUQualificationArtifactPinsV2,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    plan, pins = _validated_plan_and_pins_v2(plan_record)
    payloads = render_gpu_qualification_submit_payloads_v2(
        plan,
        single_user_name=single_user_name,
        artifact_uris=artifact_uris,
        output_root=output_root,
    )
    contracts: list[dict[str, Any]] = []
    jobs = databricks_v1._planned_jobs(plan)
    for job, payload in zip(jobs, payloads, strict=True):
        job_id = databricks_v1._safe_id(job.get("job_id"), "v2 job_id")
        task = databricks_v1._required_mapping(
            cast(list[Mapping[str, Any]], payload["tasks"])[0],
            "v2 task",
        )
        python_task = databricks_v1._required_mapping(
            task.get("spark_python_task"), "v2 spark_python_task"
        )
        parameters = python_task.get("parameters")
        if not isinstance(parameters, list) or any(
            not isinstance(item, str) for item in parameters
        ):
            raise ValueError("v2 rendered parameters are invalid")
        output_json = databricks_v1._one_parameter(
            parameters, "--output-json"
        )
        attempt_id = databricks_v1.gpu_qualification_reservation_attempt_id(
            str(plan["closed_record_sha256"]), job_id
        )
        snapshot, canonical_payload = (
            canonical_databricks_submit_payload_snapshot(payload)
        )
        contracts.append(
            {
                "job_id": job_id,
                "output_json": output_json,
                "payload": snapshot,
                "reservation_attempt_id": attempt_id,
                "submit_payload_sha256": sha256(canonical_payload).hexdigest(),
                "task_key": str(task["task_key"]),
            }
        )
    if len(contracts) != 14:
        raise ValueError("v2 controller contract lacks fourteen jobs")
    return plan, pins, payloads, tuple(contracts)


def _validated_local_preflight_binding_v2(
    path: str | Path,
    *,
    plan: Mapping[str, Any],
    submit_payloads: Sequence[Mapping[str, Any]],
    config: DatabricksWorkspaceConfig,
    require_fresh_workspace: bool,
) -> tuple[dict[str, str], datetime]:
    preflight_path = databricks_v1._validated_existing_regular_file(
        path, "v2 local_preflight_evidence_path"
    )
    record = databricks_v1._read_canonical_json_object_file(
        preflight_path,
        "v2 local preflight evidence",
    )
    plan_sha256 = _required_sha256_v2(
        plan.get("closed_record_sha256"), "plan.closed_record_sha256"
    )
    completed_at = validate_local_preflight_evidence_v2_record(
        record,
        plan_sha256=plan_sha256,
    )
    from document_kv_cache.publication_freeze_v2 import (
        validate_gpu_qualification_local_preflight_bundle_v2,
    )

    authoritative = validate_gpu_qualification_local_preflight_bundle_v2(
        preflight_path,
        plan_record=plan,
        submit_payloads=submit_payloads,
        workspace_config=config,
        require_fresh_workspace=require_fresh_workspace,
    )
    if authoritative != record:
        raise ValueError("live v2 preflight bundle record differs")
    payload_bytes = canonical_gpu_qualification_json(
        {"payloads": list(submit_payloads)}
    ).encode("utf-8")
    return (
        {
            "completed_at_utc": _required_string_v2(
                record.get("completed_at_utc"), "completed_at_utc"
            ),
            "file_sha256": databricks_v1._file_sha256(preflight_path),
            "path_sha256": sha256(
                canonical_gpu_qualification_json(
                    {
                        "domain": _V2_PREFLIGHT_PATH_DOMAIN,
                        "path": str(preflight_path),
                    }
                ).encode("utf-8")
            ).hexdigest(),
            "record_sha256": _required_sha256_v2(
                record.get("closed_record_sha256"),
                "v2 preflight closed_record_sha256",
            ),
            "submit_payloads_sha256": sha256(payload_bytes).hexdigest(),
        },
        completed_at,
    )


def _require_v2_ledger_predecessor(
    ledger: DatabricksClusterHourLedger,
    *,
    plan: Mapping[str, Any],
    ledger_path: str | Path,
    expected_predecessor: DatabricksLedgerPrefix,
    contracts: Sequence[Mapping[str, Any]],
    label: str,
) -> DatabricksLedgerPrefix:
    if not isinstance(expected_predecessor, DatabricksLedgerPrefix):
        raise TypeError("expected_phase_predecessor_prefix has the wrong type")
    predecessor = databricks_ledger_prefix_at_counts(
        ledger,
        reservation_count=len(ledger.reservations),
        submission_receipt_count=len(ledger.submission_receipts),
        terminal_actual_count=len(ledger.terminal_actuals),
    )
    if predecessor != expected_predecessor:
        raise ValueError("v2 live ledger differs from the authorized phase predecessor")
    _require_v2_phase_predecessor(
        ledger,
        plan=plan,
        ledger_path=ledger_path,
        predecessor=predecessor,
        contracts=contracts,
        label=label,
    )
    return predecessor


def _v2_batch_reservation_provably_absent(
    ledger_path: str | Path,
    *,
    predecessor: DatabricksLedgerPrefix,
) -> bool:
    """Return true only when a failed reservation left the ledger untouched."""

    try:
        live = read_databricks_cluster_hour_ledger_json(ledger_path)
        current = databricks_ledger_prefix_at_counts(
            live,
            reservation_count=len(live.reservations),
            submission_receipt_count=len(live.submission_receipts),
            terminal_actual_count=len(live.terminal_actuals),
        )
    except BaseException:
        return False
    return current == predecessor


def _require_v2_phase_predecessor(
    ledger: DatabricksClusterHourLedger,
    *,
    plan: Mapping[str, Any],
    ledger_path: str | Path,
    predecessor: DatabricksLedgerPrefix,
    contracts: Sequence[Mapping[str, Any]],
    label: str,
) -> DatabricksClusterHourLedger:
    """Validate one live phase predecessor against the frozen plan opening."""

    if databricks_ledger_path_sha256(ledger_path) != plan.get(
        "campaign_ledger_path_sha256"
    ):
        raise ValueError("v2 ledger path differs from the plan")
    if ledger.ledger_id != plan.get("campaign_ledger_id"):
        raise ValueError("v2 ledger ID differs from the plan")
    opening = databricks_ledger_prefix_from_record(
        databricks_v1._required_mapping(
            plan.get("campaign_ledger_prefix"), "campaign_ledger_prefix"
        )
    )
    require_databricks_ledger_prefix(ledger, opening)
    require_databricks_ledger_prefix(ledger, predecessor)
    predecessor_ledger = DatabricksClusterHourLedger(
        ledger_id=ledger.ledger_id,
        cap_cluster_hours=ledger.cap_cluster_hours,
        reservations=ledger.reservations[: predecessor.reservation_count],
        submission_receipts=ledger.submission_receipts[
            : predecessor.submission_receipt_count
        ],
        terminal_actuals=ledger.terminal_actuals[: predecessor.terminal_actual_count],
    )
    require_databricks_ledger_prefix(predecessor_ledger, opening)
    opening_ledger = DatabricksClusterHourLedger(
        ledger_id=ledger.ledger_id,
        cap_cluster_hours=ledger.cap_cluster_hours,
        reservations=ledger.reservations[: opening.reservation_count],
        submission_receipts=ledger.submission_receipts[
            : opening.submission_receipt_count
        ],
        terminal_actuals=ledger.terminal_actuals[: opening.terminal_actual_count],
    )
    if opening_ledger.terminal_actual_cluster_hours != plan.get(
        "campaign_opening_terminal_gpu_hours"
    ):
        raise ValueError("v2 ledger opening terminal balance drift")
    if (
        predecessor_ledger.active_reserved_task_count != 0
        or predecessor_ledger.active_reserved_cluster_hours != 0.0
    ):
        raise ValueError("v2 phase predecessor must be quiescent")
    databricks_v1._require_qualification_ledger_admission(
        predecessor_ledger,
        proposed_task_count=len(contracts),
        proposed_reserved_cluster_hours=(
            len(contracts)
            * databricks_v1.GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS
            / 3600.0
        ),
        label=label,
    )
    return predecessor_ledger


def _batch_requests_v2(
    plan: Mapping[str, Any],
    contracts: Sequence[Mapping[str, Any]],
) -> tuple[Any, ...]:
    return tuple(
        DatabricksRunAttemptReservationRequest(
            attempt_id=str(contract["reservation_attempt_id"]),
            workload_id=(
                f"gpuq-v2/{plan['closed_record_sha256'][:16]}/"
                f"{contract['job_id']}"
            ),
            submit_payload=databricks_v1._required_mapping(
                contract.get("payload"), "v2 payload"
            ),
        )
        for contract in contracts
    )


def _phase_lease_record_v2(
    *,
    plan: Mapping[str, Any],
    pins: GPUQualificationArtifactPinsV2,
    ledger_path_sha256: str,
    predecessor_prefix: Any,
    contracts: Sequence[Mapping[str, Any]],
    local_preflight_binding: Mapping[str, str],
) -> dict[str, Any]:
    if frozenset(local_preflight_binding) != _V2_PREFLIGHT_BINDING_KEYS:
        raise ValueError("v2 local preflight binding has an open schema")
    record: dict[str, Any] = {
        "artifact_sha256": pins.to_record(),
        "attempt_ids": [str(item["reservation_attempt_id"]) for item in contracts],
        "closed_record_sha256": "",
        "ledger_path_sha256": ledger_path_sha256,
        "local_preflight": dict(local_preflight_binding),
        "plan_sha256": plan["closed_record_sha256"],
        "predecessor_prefix": predecessor_prefix.to_record(),
        "record_type": GPU_QUALIFICATION_V2_PHASE_LEASE_RECORD_TYPE,
        "schema_version": 2,
        "submit_payload_sha256": [
            str(item["submit_payload_sha256"]) for item in contracts
        ],
    }
    _seal_controller_record_v2(record)
    return record


def _batch_marker_record_v2(
    *,
    plan: Mapping[str, Any],
    lease_record: Mapping[str, Any],
    batch_authorization: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "attempt_ids": list(batch_authorization.attempt_ids),
        "batch_prefix": batch_authorization.batch_prefix.to_record(),
        "closed_record_sha256": "",
        "ledger_path_sha256": batch_authorization.ledger_path_sha256,
        "phase_lease_record_sha256": lease_record["closed_record_sha256"],
        "plan_sha256": plan["closed_record_sha256"],
        "predecessor_prefix": batch_authorization.predecessor_prefix.to_record(),
        "record_type": GPU_QUALIFICATION_V2_BATCH_MARKER_RECORD_TYPE,
        "schema_version": 2,
        "submit_payload_sha256": list(
            batch_authorization.submit_payload_sha256s
        ),
    }
    _seal_controller_record_v2(record)
    return record


def _post_intent_record_v2(
    *,
    contract: Mapping[str, Any],
    batch_authorization: Any,
    phase_batch_record_sha256: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "attempt_id": contract["reservation_attempt_id"],
        "batch_prefix": batch_authorization.batch_prefix.to_record(),
        "closed_record_sha256": "",
        "job_id": contract["job_id"],
        "phase_batch_record_sha256": phase_batch_record_sha256,
        "record_type": GPU_QUALIFICATION_V2_POST_INTENT_RECORD_TYPE,
        "schema_version": 2,
        "state": "post_may_be_ambiguous_if_no_ledger_receipt",
        "submit_payload_sha256": contract["submit_payload_sha256"],
    }
    _seal_controller_record_v2(record)
    return record


def _submit_receipt_record_v2(
    *,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    ledger: DatabricksClusterHourLedger,
    phase_batch_record_sha256: str,
) -> dict[str, Any]:
    attempt_id = str(contract["reservation_attempt_id"])
    ledger_receipts = [
        item for item in ledger.submission_receipts if item.attempt_id == attempt_id
    ]
    if len(ledger_receipts) != 1:
        raise ValueError("v2 ledger lacks one exact submission receipt")
    ledger_receipt = ledger_receipts[0]
    record: dict[str, Any] = {
        "authorization_scope": (
            "submission_identity_only_requires_direct_terminal_collection_v2"
        ),
        "closed_record_sha256": "",
        "cloud_run_id": ledger_receipt.run_id,
        "job_id": contract["job_id"],
        "ledger_id": ledger.ledger_id,
        "output_json": contract["output_json"],
        "phase_batch_record_sha256": phase_batch_record_sha256,
        "plan_sha256": plan["closed_record_sha256"],
        "record_type": GPU_QUALIFICATION_V2_SUBMIT_RECEIPT_RECORD_TYPE,
        "reservation_attempt_id": attempt_id,
        "schema_version": 2,
        "submit_payload_sha256": contract["submit_payload_sha256"],
        "submit_response_sha256": ledger_receipt.submit_response_sha256,
        "task_key": contract["task_key"],
    }
    _seal_controller_record_v2(record)
    return record


def _validate_submit_receipt_v2(
    receipt: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
    ledger: DatabricksClusterHourLedger,
    phase_batch_record_sha256: str,
) -> None:
    if receipt.get("record_type") != GPU_QUALIFICATION_V2_SUBMIT_RECEIPT_RECORD_TYPE:
        raise ValueError("v2 submit receipt record type differs")
    if type(receipt.get("schema_version")) is not int or receipt.get(
        "schema_version"
    ) != 2:
        raise ValueError("v2 submit receipt schema version differs")
    _require_controller_record_seal_v2(receipt, "v2 submit receipt")
    expected = _submit_receipt_record_v2(
        plan=plan,
        contract=contract,
        ledger=ledger,
        phase_batch_record_sha256=phase_batch_record_sha256,
    )
    if canonical_gpu_qualification_json(receipt) != canonical_gpu_qualification_json(
        expected
    ):
        raise ValueError("v2 submit receipt differs from ledger authority")


def _replay_batch_marker_v2(
    *,
    plan: Mapping[str, Any],
    contracts: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    expected_phase_predecessor_prefix: DatabricksLedgerPrefix,
    submit_receipt_root: str | Path,
    local_preflight_binding: Mapping[str, str],
) -> tuple[Any, dict[str, Any]]:
    root = databricks_v1._validated_existing_controller_evidence_root(
        submit_receipt_root,
        "v2 submit_receipt_root",
    )
    lease = databricks_v1._read_canonical_json_object_file(
        root / _V2_PHASE_LEASE_FILENAME,
        "v2 qualification phase lease",
    )
    pins = pins_from_gpu_qualification_plan_v2(plan)
    if not isinstance(expected_phase_predecessor_prefix, DatabricksLedgerPrefix):
        raise TypeError("expected_phase_predecessor_prefix has the wrong type")
    phase_predecessor = expected_phase_predecessor_prefix
    expected_lease = _phase_lease_record_v2(
        plan=plan,
        pins=pins,
        ledger_path_sha256=_required_sha256_v2(
            plan.get("campaign_ledger_path_sha256"),
            "campaign_ledger_path_sha256",
        ),
        predecessor_prefix=phase_predecessor,
        contracts=contracts,
        local_preflight_binding=local_preflight_binding,
    )
    if canonical_gpu_qualification_json(lease) != canonical_gpu_qualification_json(
        expected_lease
    ):
        raise ValueError("v2 phase lease differs from the frozen batch")
    _require_controller_record_seal_v2(lease, "v2 phase lease")
    live = read_databricks_cluster_hour_ledger_json(ledger_path)
    _require_v2_phase_predecessor(
        live,
        plan=plan,
        ledger_path=ledger_path,
        predecessor=phase_predecessor,
        contracts=contracts,
        label="v2 durable batch replay",
    )
    requests = _batch_requests_v2(plan, contracts)
    if (
        len(live.reservations) == phase_predecessor.reservation_count
        and len(live.submission_receipts)
        == phase_predecessor.submission_receipt_count
        and len(live.terminal_actuals) == phase_predecessor.terminal_actual_count
    ):
        if {item.name for item in root.iterdir()} != {_V2_PHASE_LEASE_FILENAME}:
            raise ValueError("v2 lease-only resume root has unexpected evidence")
        _reserved_ledger, authorization = (
            reserve_databricks_run_attempt_batch_authorized_json(
                ledger_path,
                requests,
                expected_predecessor_prefix=phase_predecessor,
            )
        )
    else:
        if (
            len(live.reservations)
            != phase_predecessor.reservation_count + len(contracts)
            or not (
                phase_predecessor.submission_receipt_count
                <= len(live.submission_receipts)
                <= phase_predecessor.submission_receipt_count + len(contracts)
            )
            or not (
                phase_predecessor.terminal_actual_count
                <= len(live.terminal_actuals)
                <= phase_predecessor.terminal_actual_count + len(contracts)
            )
        ):
            raise ValueError("v2 replay ledger is not a canonical batch prefix")
        authorization = replay_databricks_run_attempt_batch_authorization_json(
            ledger_path,
            requests,
            expected_predecessor_prefix=phase_predecessor,
        )
    live = read_databricks_cluster_hour_ledger_json(ledger_path)
    receipt_count, terminal_count = _staggered_batch_progress_v2(
        live, authorization
    )
    require_databricks_publication_batch_admission(
        live,
        authorization,
    )
    expected_marker = _batch_marker_record_v2(
        plan=plan,
        lease_record=lease,
        batch_authorization=authorization,
    )
    marker = _validated_resume_evidence_prefix_v2(
        root=root,
        plan=plan,
        contracts=contracts,
        ledger=live,
        receipt_count=receipt_count,
        terminal_count=terminal_count,
        batch_authorization=authorization,
        expected_marker=expected_marker,
    )
    if marker is None:
        marker_path = root / _V2_BATCH_MARKER_FILENAME
        databricks_v1._write_canonical_exclusive(expected_marker, marker_path)
        marker = expected_marker
    return authorization, marker


def _validated_resume_evidence_prefix_v2(
    *,
    root: Path,
    plan: Mapping[str, Any],
    contracts: Sequence[Mapping[str, Any]],
    ledger: DatabricksClusterHourLedger,
    receipt_count: int,
    terminal_count: int,
    batch_authorization: DatabricksBatchReservationAuthorization,
    expected_marker: Mapping[str, Any],
) -> dict[str, Any] | None:
    entries = tuple(root.iterdir())
    if any(item.is_symlink() or not item.is_file() for item in entries):
        raise ValueError("v2 resume evidence must contain only regular files")
    names = {item.name for item in entries}
    receipt_names = tuple(f"{contract['job_id']}.json" for contract in contracts)
    intent_names = tuple(
        f"{contract['job_id']}.post-intent-v2" for contract in contracts
    )
    allowed_names = {
        _V2_PHASE_LEASE_FILENAME,
        _V2_BATCH_MARKER_FILENAME,
        *receipt_names,
        *intent_names,
    }
    if names - allowed_names:
        raise ValueError("v2 resume evidence contains an unexpected file")
    receipt_indices = tuple(
        index for index, name in enumerate(receipt_names) if name in names
    )
    controller_receipt_count = len(receipt_indices)
    if receipt_indices != tuple(range(controller_receipt_count)):
        raise ValueError("v2 resume receipts are not a canonical job prefix")
    allowed_receipt_counts = {receipt_count}
    if receipt_count > 0:
        allowed_receipt_counts.add(receipt_count - 1)
    if controller_receipt_count not in allowed_receipt_counts:
        raise ValueError("v2 controller receipts differ from the ledger prefix")
    if not (
        terminal_count
        <= controller_receipt_count
        <= receipt_count
        <= len(contracts)
    ) or controller_receipt_count - terminal_count > 1:
        raise ValueError("v2 controller receipt/terminal progress is not canonical")
    intent_indices = tuple(
        index for index, name in enumerate(intent_names) if name in names
    )
    marker_path = root / _V2_BATCH_MARKER_FILENAME
    if _V2_BATCH_MARKER_FILENAME not in names:
        if (
            names != {_V2_PHASE_LEASE_FILENAME}
            or receipt_count != 0
            or terminal_count != 0
        ):
            raise ValueError("v2 batch marker is absent outside lease-only recovery")
        return None
    marker = databricks_v1._read_canonical_json_object_file(
        marker_path,
        "v2 qualification batch marker",
    )
    if canonical_gpu_qualification_json(marker) != canonical_gpu_qualification_json(
        expected_marker
    ):
        raise ValueError("v2 batch marker differs from the ledger batch")
    _require_controller_record_seal_v2(marker, "v2 batch marker")
    allowed_intent_indices: set[tuple[int, ...]]
    if controller_receipt_count == receipt_count - 1:
        if terminal_count != controller_receipt_count:
            raise ValueError("v2 ledger receipt advanced beyond controller terminals")
        allowed_intent_indices = {(controller_receipt_count,)}
    else:
        allowed_intent_indices = {()}
        if receipt_count < len(contracts) and terminal_count == receipt_count:
            allowed_intent_indices.add((receipt_count,))
        if receipt_count > 0 and terminal_count == receipt_count - 1:
            allowed_intent_indices.add((receipt_count - 1,))
    if intent_indices not in allowed_intent_indices:
        raise ValueError("v2 resume intent is not the canonical durable crash point")
    marker_sha256 = str(marker["closed_record_sha256"])
    for index in receipt_indices:
        receipt = databricks_v1._read_canonical_json_object_file(
            root / receipt_names[index],
            f"v2 submit receipt {contracts[index]['job_id']}",
        )
        _validate_submit_receipt_v2(
            receipt,
            contract=contracts[index],
            plan=plan,
            ledger=ledger,
            phase_batch_record_sha256=marker_sha256,
        )
    for index in intent_indices:
        intent = databricks_v1._read_canonical_json_object_file(
            root / intent_names[index],
            f"v2 post intent {contracts[index]['job_id']}",
        )
        expected_intent = _post_intent_record_v2(
            contract=contracts[index],
            batch_authorization=batch_authorization,
            phase_batch_record_sha256=marker_sha256,
        )
        if canonical_gpu_qualification_json(intent) != canonical_gpu_qualification_json(
            expected_intent
        ):
            raise ValueError("v2 post intent differs from the batch authority")
    return marker


def _load_submit_receipts_v2(
    root: Path,
    *,
    contracts: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    ledger: DatabricksClusterHourLedger,
    phase_batch_record_sha256: str,
) -> tuple[dict[str, Any], ...]:
    expected_names = {
        _V2_PHASE_LEASE_FILENAME,
        _V2_BATCH_MARKER_FILENAME,
        *(f"{contract['job_id']}.json" for contract in contracts),
    }
    if {path.name for path in root.iterdir()} != expected_names:
        raise ValueError("v2 submit receipt directory is not the exact batch closure")
    receipts: list[dict[str, Any]] = []
    for contract in contracts:
        receipt = databricks_v1._read_canonical_json_object_file(
            root / f"{contract['job_id']}.json",
            f"v2 submit receipt {contract['job_id']}",
        )
        _validate_submit_receipt_v2(
            receipt,
            contract=contract,
            plan=plan,
            ledger=ledger,
            phase_batch_record_sha256=phase_batch_record_sha256,
        )
        receipts.append(receipt)
    return tuple(receipts)


def _replay_completed_batch_v2(
    *,
    plan: Mapping[str, Any],
    contracts: Sequence[Mapping[str, Any]],
    ledger_path: str | Path,
    expected_phase_predecessor_prefix: DatabricksLedgerPrefix,
    submit_receipt_root: str | Path,
    local_preflight_binding: Mapping[str, str],
) -> tuple[
    DatabricksBatchReservationAuthorization,
    dict[str, Any],
    tuple[dict[str, Any], ...],
    DatabricksClusterHourLedger,
]:
    """Replay a submitted batch without reserving or weakening terminal suffixes."""

    root = databricks_v1._validated_existing_controller_evidence_root(
        submit_receipt_root,
        "v2 submit_receipt_root",
    )
    lease = databricks_v1._read_canonical_json_object_file(
        root / _V2_PHASE_LEASE_FILENAME,
        "v2 qualification phase lease",
    )
    pins = pins_from_gpu_qualification_plan_v2(plan)
    if not isinstance(expected_phase_predecessor_prefix, DatabricksLedgerPrefix):
        raise TypeError("expected_phase_predecessor_prefix has the wrong type")
    phase_predecessor = expected_phase_predecessor_prefix
    expected_lease = _phase_lease_record_v2(
        plan=plan,
        pins=pins,
        ledger_path_sha256=_required_sha256_v2(
            plan.get("campaign_ledger_path_sha256"),
            "campaign_ledger_path_sha256",
        ),
        predecessor_prefix=phase_predecessor,
        contracts=contracts,
        local_preflight_binding=local_preflight_binding,
    )
    if canonical_gpu_qualification_json(lease) != canonical_gpu_qualification_json(
        expected_lease
    ):
        raise ValueError("v2 completed-batch phase lease differs")
    _require_controller_record_seal_v2(lease, "v2 phase lease")
    if databricks_ledger_path_sha256(ledger_path) != plan.get(
        "campaign_ledger_path_sha256"
    ):
        raise ValueError("v2 completed-batch ledger path differs from the plan")
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    if ledger.ledger_id != plan.get("campaign_ledger_id"):
        raise ValueError("v2 completed-batch ledger ID differs from the plan")
    _require_v2_phase_predecessor(
        ledger,
        plan=plan,
        ledger_path=ledger_path,
        predecessor=phase_predecessor,
        contracts=contracts,
        label="v2 completed batch replay",
    )
    authorization = replay_databricks_run_attempt_batch_authorization_json(
        ledger_path,
        _batch_requests_v2(plan, contracts),
        expected_predecessor_prefix=phase_predecessor,
    )
    require_databricks_publication_batch_admission(ledger, authorization)
    receipt_count, terminal_count = _staggered_batch_progress_v2(
        ledger, authorization
    )
    if receipt_count != len(contracts) or terminal_count not in {
        len(contracts) - 1,
        len(contracts),
    }:
        raise ValueError("v2 completed batch is not the current ledger suffix")
    marker = databricks_v1._read_canonical_json_object_file(
        root / _V2_BATCH_MARKER_FILENAME,
        "v2 qualification batch marker",
    )
    expected_marker = _batch_marker_record_v2(
        plan=plan,
        lease_record=lease,
        batch_authorization=authorization,
    )
    if canonical_gpu_qualification_json(marker) != canonical_gpu_qualification_json(
        expected_marker
    ):
        raise ValueError("v2 completed-batch marker differs")
    _require_controller_record_seal_v2(marker, "v2 batch marker")
    receipts = _load_submit_receipts_v2(
        root,
        contracts=contracts,
        plan=plan,
        ledger=ledger,
        phase_batch_record_sha256=str(marker["closed_record_sha256"]),
    )
    return authorization, marker, receipts, ledger


def _terminal_receipt_record_v2(
    *,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    submit_receipt: Mapping[str, Any],
    ledger_id: str,
    ledger_terminal_actual: Any,
    run: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    result: Mapping[str, Any],
    collected_at_utc: str,
    phase_batch_record_sha256: str,
) -> dict[str, Any]:
    receipt = databricks_v1._terminal_receipt_record(
        plan=plan,
        contract=contract,
        submit_receipt=submit_receipt,
        ledger_id=ledger_id,
        ledger_terminal_actual=ledger_terminal_actual,
        run=run,
        run_identity=run_identity,
        result=result,
        collected_at_utc=collected_at_utc,
        phase_batch_record_sha256=phase_batch_record_sha256,
    )
    receipt["record_type"] = GPU_QUALIFICATION_V2_TERMINAL_RECEIPT_RECORD_TYPE
    receipt["schema_version"] = 2
    receipt["closed_record_sha256"] = ""
    _seal_controller_record_v2(receipt)
    return receipt


def _publish_gpu_qualification_evidence_v2_atomic(
    root: Path,
    *,
    receipts: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
) -> None:
    """Publish the complete v2 terminal/evidence closure with one rename."""

    databricks_v1._validated_fresh_controller_evidence_root(root)
    closure_digest = databricks_v1._canonical_json_sha256(
        {"evidence": dict(evidence), "receipts": list(receipts)}
    )
    staging = root.with_name(f".{root.name}.staging-{closure_digest[:16]}")
    staging_root = databricks_v1._create_fresh_controller_evidence_root(staging)
    try:
        for receipt in receipts:
            job_id = databricks_v1._safe_id(
                receipt.get("job_id"), "v2 terminal receipt job_id"
            )
            databricks_v1._write_canonical_exclusive(
                receipt,
                staging_root
                / f"{job_id}{_GPU_QUALIFICATION_V2_TERMINAL_RECEIPT_SUFFIX}",
            )
        databricks_v1._write_canonical_exclusive(
            evidence,
            staging_root / GPU_QUALIFICATION_V2_EVIDENCE_FILENAME,
        )
        databricks_v1._fsync_directory(staging_root)
        os.rename(staging_root, root)
        databricks_v1._fsync_directory(root.parent)
    except BaseException:
        if staging_root.exists() and not staging_root.is_symlink():
            shutil.rmtree(staging_root)
            databricks_v1._fsync_directory(staging_root.parent)
        raise


def collect_gpu_qualification_evidence_v2(
    config: DatabricksWorkspaceConfig,
    *,
    plan_record: Mapping[str, Any],
    single_user_name: str,
    artifact_uris: Mapping[str, str],
    output_root: str,
    ledger_path: str | Path,
    expected_phase_predecessor_prefix: DatabricksLedgerPrefix,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    evidence_root: str | Path,
    now: Callable[[], datetime] | None = None,
) -> tuple[dict[str, Any], databricks_v1.GPUQualificationLaunchAuthorization]:
    """Reconcile every v2 terminal before validating and publishing success."""

    plan, pins, payloads, contracts = _validated_controller_contract_v2(
        plan_record=plan_record,
        single_user_name=single_user_name,
        artifact_uris=artifact_uris,
        output_root=output_root,
    )
    preflight_binding, _completed_at = _validated_local_preflight_binding_v2(
        local_preflight_evidence_path,
        plan=plan,
        submit_payloads=payloads,
        config=config,
        require_fresh_workspace=False,
    )
    batch_authorization, marker, submit_receipts, _opening_ledger = (
        _replay_completed_batch_v2(
            plan=plan,
            contracts=contracts,
            ledger_path=ledger_path,
            expected_phase_predecessor_prefix=(
                expected_phase_predecessor_prefix
            ),
            submit_receipt_root=submit_receipt_root,
            local_preflight_binding=preflight_binding,
        )
    )
    target_root = databricks_v1._validated_fresh_controller_evidence_root(evidence_root)
    runs = tuple(
        get_databricks_run(config, str(receipt["cloud_run_id"]))
        for receipt in submit_receipts
    )
    for contract, run in zip(contracts, runs, strict=True):
        # Record every terminal outcome, including failed jobs, in canonical
        # plan order. A malformed or nonterminal response must stop here:
        # appending later attempts around that gap would make durable ordered
        # batch replay impossible.
        record_databricks_verified_run_terminal_actual_json(
            ledger_path,
            attempt_id=str(contract["reservation_attempt_id"]),
            run_record=run,
        )
    final_ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    terminal_prefix = require_databricks_batch_terminal_closure(
        final_ledger,
        batch_authorization,
        require_complete_current_prefix=True,
    )
    clock = now or _utc_now
    job_results: list[dict[str, Any]] = []
    terminal_receipts: list[dict[str, Any]] = []
    for planned_job, contract, submit_receipt, run in zip(
        databricks_v1._planned_jobs(plan),
        contracts,
        submit_receipts,
        runs,
        strict=True,
    ):
        actual = next(
            item
            for item in final_ledger.terminal_actuals
            if item.attempt_id == contract["reservation_attempt_id"]
        )
        run_identity = databricks_v1._validate_control_plane_run(
            run,
            planned_job=planned_job,
            contract=contract,
            submit_receipt=submit_receipt,
        )
        if (
            actual.verification_source != "direct_databricks_runs_get"
            or actual.run_id != run_identity["cloud_run_id"]
            or actual.submit_payload_sha256 != contract["submit_payload_sha256"]
            or actual.control_plane_status_sha256
            != databricks_v1._canonical_json_sha256(run)
        ):
            raise RuntimeError("v2 terminal actual differs from direct runs/get")
        if (
            run_identity["succeeded"] is not True
            or actual.terminal_state != "succeeded"
        ):
            raise RuntimeError(
                f"GPU qualification v2 job {contract['job_id']!r} did not succeed"
            )
        result = databricks_v1._read_gpu_qualification_result(
            config,
            str(contract["output_json"]),
            label=f"GPU v2 result {contract['job_id']}",
            closed_record_convention="field_blank",
        )
        validate_gpu_job_result_v2_record(
            result,
            plan_record=plan,
            expected_artifact_pins=pins,
        )
        databricks_v1._validate_result_submission_binding(
            result,
            contract=contract,
            submit_receipt=submit_receipt,
            run_identity=run_identity,
        )
        receipt = _terminal_receipt_record_v2(
            plan=plan,
            contract=contract,
            submit_receipt=submit_receipt,
            ledger_id=final_ledger.ledger_id,
            ledger_terminal_actual=actual,
            run=run,
            run_identity=run_identity,
            result=result,
            collected_at_utc=databricks_v1._utc_timestamp(clock()),
            phase_batch_record_sha256=str(marker["closed_record_sha256"]),
        )
        job_results.append(result)
        terminal_receipts.append(receipt)
    for receipt in terminal_receipts:
        receipt["phase_terminal_prefix"] = terminal_prefix.to_record()
        receipt["closed_record_sha256"] = ""
        _seal_controller_record_v2(receipt)
    databricks_v1._validate_collected_identity_closure(
        terminal_receipts,
        contracts=contracts,
    )
    selected_gmus = [
        float(result["measurements"]["gpu_memory_utilization"])
        for result in job_results
        if result["job_id"].startswith("aws-g6-l4-32k-c4-gmu-")
        and result["measurements"].get("candidate_qualified") is True
    ]
    if not selected_gmus:
        raise RuntimeError("no governed v2 GMU result qualified")
    cloud = _build_governed_cloud_gpu_evidence_v2(
        plan_sha256=str(plan["closed_record_sha256"]),
        jobs=job_results,
        terminal_receipts=terminal_receipts,
        selected_gpu_memory_utilization=max(selected_gmus),
    )
    local_preflight = databricks_v1._read_canonical_json_object_file(
        local_preflight_evidence_path,
        "v2 local preflight evidence",
    )
    evidence = _build_governed_gpu_qualification_evidence_v2(
        campaign_id=str(plan["campaign_id"]),
        plan_sha256=str(plan["closed_record_sha256"]),
        local_preflight_evidence=local_preflight,
        cloud_gpu_evidence=cloud,
    )
    validate_gpu_qualification_evidence_v2_record(
        evidence,
        plan_record=plan,
        expected_campaign_id=str(plan["campaign_id"]),
        expected_artifact_pins=pins,
    )
    _publish_gpu_qualification_evidence_v2_atomic(
        target_root,
        receipts=terminal_receipts,
        evidence=evidence,
    )
    authorization = replay_gpu_qualification_launch_authorization_v2(
        config=config,
        plan_record=plan,
        single_user_name=single_user_name,
        artifact_uris=artifact_uris,
        output_root=output_root,
        ledger_path=ledger_path,
        expected_phase_predecessor_prefix=expected_phase_predecessor_prefix,
        submit_receipt_root=submit_receipt_root,
        local_preflight_evidence_path=local_preflight_evidence_path,
        evidence_root=target_root,
        expected_campaign_id=str(plan["campaign_id"]),
        expected_artifact_pins=pins,
    )
    return evidence, authorization


def replay_gpu_qualification_launch_authorization_v2(
    *,
    config: DatabricksWorkspaceConfig,
    plan_record: Mapping[str, Any],
    single_user_name: str,
    artifact_uris: Mapping[str, str],
    output_root: str,
    ledger_path: str | Path,
    expected_phase_predecessor_prefix: DatabricksLedgerPrefix,
    submit_receipt_root: str | Path,
    local_preflight_evidence_path: str | Path,
    evidence_root: str | Path,
    expected_campaign_id: str,
    expected_artifact_pins: GPUQualificationArtifactPinsV2,
) -> databricks_v1.GPUQualificationLaunchAuthorization:
    """Reissue launch authority from the complete durable v2 closure."""

    plan, pins, payloads, contracts = _validated_controller_contract_v2(
        plan_record=plan_record,
        single_user_name=single_user_name,
        artifact_uris=artifact_uris,
        output_root=output_root,
    )
    if plan["campaign_id"] != expected_campaign_id or pins != expected_artifact_pins:
        raise ValueError("v2 replay campaign or artifact authority differs")
    preflight_binding, _completed_at = _validated_local_preflight_binding_v2(
        local_preflight_evidence_path,
        plan=plan,
        submit_payloads=payloads,
        config=config,
        require_fresh_workspace=False,
    )
    batch_authorization, marker, submit_receipts, ledger = _replay_completed_batch_v2(
        plan=plan,
        contracts=contracts,
        ledger_path=ledger_path,
        expected_phase_predecessor_prefix=expected_phase_predecessor_prefix,
        submit_receipt_root=submit_receipt_root,
        local_preflight_binding=preflight_binding,
    )
    terminal_prefix = require_databricks_batch_terminal_closure(
        ledger,
        batch_authorization,
        require_complete_current_prefix=True,
    )
    root = databricks_v1._validated_existing_controller_evidence_root(
        evidence_root,
        "v2 evidence_root",
    )
    expected_names = {
        GPU_QUALIFICATION_V2_EVIDENCE_FILENAME,
        *(
            f"{contract['job_id']}{_GPU_QUALIFICATION_V2_TERMINAL_RECEIPT_SUFFIX}"
            for contract in contracts
        ),
    }
    if {path.name for path in root.iterdir()} != expected_names:
        raise ValueError("v2 evidence root is not the exact terminal closure")
    terminal_receipts = tuple(
        databricks_v1._read_canonical_json_object_file(
            root
            / (f"{contract['job_id']}{_GPU_QUALIFICATION_V2_TERMINAL_RECEIPT_SUFFIX}"),
            f"v2 terminal receipt {contract['job_id']}",
        )
        for contract in contracts
    )
    evidence_file = root / GPU_QUALIFICATION_V2_EVIDENCE_FILENAME
    evidence = databricks_v1._read_canonical_json_object_file(
        evidence_file,
        "GPU qualification v2 evidence",
    )
    selection = validate_gpu_qualification_evidence_v2_record(
        evidence,
        plan_record=plan,
        expected_campaign_id=expected_campaign_id,
        expected_artifact_pins=expected_artifact_pins,
    )
    local_preflight = databricks_v1._read_canonical_json_object_file(
        local_preflight_evidence_path,
        "v2 local preflight evidence",
    )
    if evidence.get("local_preflight_evidence") != local_preflight:
        raise ValueError("persisted v2 local preflight differs from evidence")
    cloud = databricks_v1._required_mapping(
        evidence.get("cloud_gpu_evidence"),
        "v2 cloud_gpu_evidence",
    )
    if cloud.get("terminal_receipts") != list(terminal_receipts):
        raise ValueError("persisted v2 terminal receipts differ from evidence")
    terminal_actual_hashes: list[str] = []
    for contract, submit_receipt, receipt in zip(
        contracts,
        submit_receipts,
        terminal_receipts,
        strict=True,
    ):
        actual = next(
            (
                item
                for item in ledger.terminal_actuals
                if item.attempt_id == contract["reservation_attempt_id"]
            ),
            None,
        )
        if actual is None:
            raise ValueError("v2 replay ledger lacks a terminal actual")
        actual_record = databricks_v1._ledger_terminal_actual_record(actual)
        actual_sha256 = databricks_v1._canonical_json_sha256(actual_record)
        if (
            actual.verification_source != "direct_databricks_runs_get"
            or actual.terminal_state != "succeeded"
            or actual.run_id != submit_receipt["cloud_run_id"]
            or actual.submit_payload_sha256 != contract["submit_payload_sha256"]
            or actual.control_plane_status_sha256
            != receipt["control_plane_status_sha256"]
            or actual_sha256 != receipt["ledger_terminal_actual_sha256"]
            or receipt.get("phase_batch_record_sha256")
            != marker["closed_record_sha256"]
            or receipt.get("phase_terminal_prefix") != terminal_prefix.to_record()
        ):
            raise ValueError("v2 replay terminal receipt differs from ledger authority")
        terminal_actual_hashes.append(actual_sha256)
    evidence_file_sha256 = databricks_v1._file_sha256(evidence_file)
    causal_closure = {
        "evidence_closed_record_sha256": evidence["closed_record_sha256"],
        "evidence_file_sha256": evidence_file_sha256,
        "ledger_id": ledger.ledger_id,
        "ledger_path_sha256": databricks_ledger_path_sha256(ledger_path),
        "ledger_prefix": terminal_prefix.to_record(),
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
    return databricks_v1._issue_gpu_qualification_launch_authorization(
        selection=selection,
        plan_sha256=str(plan["closed_record_sha256"]),
        evidence_closed_record_sha256=str(evidence["closed_record_sha256"]),
        evidence_file_sha256=evidence_file_sha256,
        ledger_id=ledger.ledger_id,
        ledger_path_sha256=databricks_ledger_path_sha256(ledger_path),
        predecessor_prefix=batch_authorization.predecessor_prefix,
        producer_batch_prefix=batch_authorization.batch_prefix,
        ledger_prefix=terminal_prefix,
        causal_closure_sha256=databricks_v1._canonical_json_sha256(causal_closure),
    )


def _controller_record_digest_v2(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload["closed_record_sha256"] = ""
    return sha256(canonical_gpu_qualification_json(payload).encode("utf-8")).hexdigest()


def _seal_controller_record_v2(record: dict[str, Any]) -> None:
    if record.get("closed_record_sha256") != "":
        raise ValueError("v2 controller record must begin with an empty seal")
    record["closed_record_sha256"] = _controller_record_digest_v2(record)


def _require_controller_record_seal_v2(
    record: Mapping[str, Any],
    label: str,
) -> None:
    observed = _required_sha256_v2(
        record.get("closed_record_sha256"),
        f"{label}.closed_record_sha256",
    )
    if observed != _controller_record_digest_v2(record):
        raise ValueError(f"{label} seal differs")


def _required_sha256_v2(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _required_string_v2(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a canonical nonempty string")
    return value


def _canonical_bootstrap_json_v2(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _bootstrap_object_without_duplicate_keys_v2(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("v2 bootstrap handoff contains a duplicate JSON key")
        value[key] = item
    return value


def _reject_bootstrap_json_constant_v2(value: str) -> object:
    raise ValueError(f"v2 bootstrap handoff contains invalid JSON constant {value!r}")


def _exact_bootstrap_argv_v2(
    argv: Sequence[str],
) -> tuple[tuple[str, ...], str]:
    if isinstance(argv, (str, bytes, bytearray)) or not isinstance(argv, Sequence):
        raise TypeError("v2 bootstrap argv must be a sequence")
    normalized = tuple(argv)
    if any(type(value) is not str for value in normalized):
        raise TypeError("v2 bootstrap argv values must be exact strings")
    if len(normalized) != (
        2 * len(_V2_BOOTSTRAP_SINGLETON_OPTIONS)
        + 4 * len(GPU_QUALIFICATION_V2_ARTIFACT_KEYS)
    ):
        raise ValueError("v2 bootstrap requires the exact 50-value argv closure")
    index = 0
    for option_name in _V2_BOOTSTRAP_SINGLETON_OPTIONS:
        if normalized[index] != option_name or not normalized[index + 1]:
            raise ValueError("v2 bootstrap singleton argv ordering differs")
        index += 2
    runner_pin = ""
    uris: list[str] = []
    for expected_key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS:
        if normalized[index] != "--artifact-uri":
            raise ValueError("v2 bootstrap artifact URI argv ordering differs")
        uri_key, separator, uri = normalized[index + 1].partition("=")
        if not separator or uri_key != expected_key or not uri:
            raise ValueError("v2 bootstrap artifact URI argv closure differs")
        uris.append(uri)
        if normalized[index + 2] != "--artifact-sha256":
            raise ValueError("v2 bootstrap artifact SHA-256 argv ordering differs")
        pin_key, separator, pin = normalized[index + 3].partition("=")
        if not separator or pin_key != expected_key:
            raise ValueError("v2 bootstrap artifact SHA-256 argv closure differs")
        pin = _required_sha256_v2(pin, f"artifact_sha256.{expected_key}")
        if expected_key == "runner_sha256":
            runner_pin = pin
        index += 4
    if index != len(normalized) or len(set(uris)) != len(uris) or not runner_pin:
        raise ValueError("v2 bootstrap argv closure differs")
    return normalized, runner_pin


def _bootstrap_argv_sha256_v2(argv: Sequence[str]) -> str:
    normalized, _runner_pin = _exact_bootstrap_argv_v2(argv)
    return sha256(
        _canonical_bootstrap_json_v2(list(normalized)).encode("utf-8")
    ).hexdigest()


def _require_sanitized_child_environment_v2(
    environment: Mapping[str, str],
) -> None:
    if any(name in environment for name in _V2_CHILD_FORBIDDEN_ENV):
        raise RuntimeError("v2 bootstrap child inherited an unsafe Python path")
    if any(
        environment.get(name) != expected
        for name, expected in _V2_CHILD_REQUIRED_ENV.items()
    ):
        raise RuntimeError("v2 bootstrap child lacks its exact Python safety flags")


def _decode_bootstrap_handoff_v2(
    raw_handoff: object,
    *,
    argv: Sequence[str],
    environment: Mapping[str, str],
) -> str:
    _require_sanitized_child_environment_v2(environment)
    if type(raw_handoff) is not str or not raw_handoff:
        raise ValueError("v2 bootstrap handoff must be one nonempty string")
    try:
        encoded_handoff = raw_handoff.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("v2 bootstrap handoff is not valid UTF-8") from exc
    if len(encoded_handoff) > _V2_BOOTSTRAP_HANDOFF_MAX_BYTES:
        raise ValueError("v2 bootstrap handoff exceeds its size cap")
    try:
        decoded = json.loads(
            raw_handoff,
            object_pairs_hook=_bootstrap_object_without_duplicate_keys_v2,
            parse_constant=_reject_bootstrap_json_constant_v2,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("v2 bootstrap handoff is not strict JSON") from exc
    if type(decoded) is not dict or _canonical_bootstrap_json_v2(decoded) != raw_handoff:
        raise ValueError("v2 bootstrap handoff is not one canonical JSON object")
    if set(decoded) != _V2_BOOTSTRAP_HANDOFF_KEYS:
        raise ValueError("v2 bootstrap handoff does not use its exact schema")
    if decoded.get("record_type") != _V2_BOOTSTRAP_HANDOFF_RECORD_TYPE:
        raise ValueError("v2 bootstrap handoff record type differs")
    schema_version = decoded.get("schema_version")
    if (
        type(schema_version) is not int
        or schema_version != _V2_BOOTSTRAP_HANDOFF_SCHEMA_VERSION
    ):
        raise ValueError("v2 bootstrap handoff schema version differs")
    if decoded.get("spark_checked") is not True:
        raise ValueError("v2 bootstrap handoff did not check Spark")
    cluster_id = databricks_v1._validated_cloud_cluster_id(
        decoded.get("cluster_id"),
        source="v2 bootstrap handoff",
    )
    argv_sha256 = _required_sha256_v2(
        decoded.get("argv_sha256"), "v2 bootstrap handoff argv_sha256"
    )
    runner_sha256 = _required_sha256_v2(
        decoded.get("runner_sha256"), "v2 bootstrap handoff runner_sha256"
    )
    observed_seal = _required_sha256_v2(
        decoded.get("closed_record_sha256"),
        "v2 bootstrap handoff closed_record_sha256",
    )
    open_record = dict(decoded)
    open_record["closed_record_sha256"] = ""
    expected_seal = sha256(
        _canonical_bootstrap_json_v2(open_record).encode("utf-8")
    ).hexdigest()
    if observed_seal != expected_seal:
        raise ValueError("v2 bootstrap handoff seal differs")
    normalized_argv, runner_pin = _exact_bootstrap_argv_v2(argv)
    expected_argv_sha256 = sha256(
        _canonical_bootstrap_json_v2(list(normalized_argv)).encode("utf-8")
    ).hexdigest()
    if argv_sha256 != expected_argv_sha256:
        raise ValueError("v2 bootstrap handoff argv binding differs")
    if not (
        runner_sha256
        == runner_pin
        == GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SHA256
    ):
        raise ValueError("v2 bootstrap handoff runner binding differs")
    raw_sources = decoded.get("sources")
    if type(raw_sources) is not list or any(
        type(source) is not str for source in raw_sources
    ):
        raise ValueError("v2 bootstrap handoff sources must be exact strings")
    sources = tuple(raw_sources)
    if (
        not sources
        or len(set(sources)) != len(sources)
        or tuple(
            source for source in _V2_CLUSTER_ID_SOURCE_ORDER if source in sources
        )
        != sources
    ):
        raise ValueError("v2 bootstrap handoff source ordering differs")
    observed_env_sources = tuple(
        name for name in _V2_CLUSTER_ID_ENV_NAMES if name in environment
    )
    recorded_env_sources = tuple(
        source for source in sources if source in _V2_CLUSTER_ID_ENV_NAMES
    )
    if observed_env_sources != recorded_env_sources:
        raise ValueError("v2 bootstrap handoff environment sources differ")
    for source in observed_env_sources:
        observed_cluster_id = databricks_v1._validated_cloud_cluster_id(
            environment[source], source=source
        )
        if observed_cluster_id != cluster_id:
            raise RuntimeError("Databricks cluster identity sources are ambiguous")
    return cluster_id


def _validated_main_inputs_v2(
    argv: Sequence[str] | None,
) -> tuple[argparse.Namespace, dict[str, Any], dict[str, str], dict[str, str]]:
    args = _parse_args(argv)
    if args.attempt_number != 0 or args.retry_count != 0:
        raise ValueError("GPU qualification v2 jobs must execute on attempt zero")
    plan = _decode_plan_parameter(
        args.plan_record_zlib_base64,
        expected_plan_sha256=args.expected_plan_sha256,
    )
    artifact_uris = databricks_v1._parse_key_value_args(
        args.artifact_uri, option_name="--artifact-uri"
    )
    artifact_sha256 = databricks_v1._parse_key_value_args(
        args.artifact_sha256, option_name="--artifact-sha256"
    )
    return args, plan, artifact_uris, artifact_sha256


def _execute_main_inputs_v2(
    args: argparse.Namespace,
    plan: Mapping[str, Any],
    artifact_uris: Mapping[str, str],
    artifact_sha256: Mapping[str, str],
    *,
    cloud_cluster_id: str,
) -> int:
    execute_gpu_qualification_job_v2(
        plan_record=plan,
        expected_plan_sha256=args.expected_plan_sha256,
        job_id=args.job_id,
        reservation_attempt_id=args.reservation_attempt_id,
        artifact_uris=artifact_uris,
        artifact_sha256=artifact_sha256,
        output_json=args.output_json,
        work_dir=args.work_dir,
        cloud_run_id=args.cloud_run_id,
        cloud_cluster_id=cloud_cluster_id,
        sentinel_runner=_builtin_sentinel_runner_v2,
    )
    return 0


def _main_from_bootstrap_handoff_v2(
    raw_handoff: object = _V2_BOOTSTRAP_HANDOFF_MISSING,
    argv: Sequence[str] | None = None,
) -> int:
    environment_handoff = os.environ.pop(
        _V2_BOOTSTRAP_HANDOFF_ENV,
        _V2_BOOTSTRAP_HANDOFF_MISSING,
    )
    if (
        raw_handoff is not _V2_BOOTSTRAP_HANDOFF_MISSING
        and environment_handoff is not _V2_BOOTSTRAP_HANDOFF_MISSING
    ):
        raise RuntimeError("v2 bootstrap handoff has conflicting input sources")
    resolved_handoff = (
        environment_handoff
        if raw_handoff is _V2_BOOTSTRAP_HANDOFF_MISSING
        else raw_handoff
    )
    resolved_argv: Sequence[str] = sys.argv[1:] if argv is None else argv
    cluster_id = _decode_bootstrap_handoff_v2(
        resolved_handoff,
        argv=resolved_argv,
        environment=dict(os.environ),
    )
    args, plan, artifact_uris, artifact_sha256 = _validated_main_inputs_v2(
        resolved_argv
    )
    return _execute_main_inputs_v2(
        args,
        plan,
        artifact_uris,
        artifact_sha256,
        cloud_cluster_id=cluster_id,
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def main(argv: Sequence[str] | None = None) -> int:
    args, plan, artifact_uris, artifact_sha256 = _validated_main_inputs_v2(argv)
    return _execute_main_inputs_v2(
        args,
        plan,
        artifact_uris,
        artifact_sha256,
        cloud_cluster_id=databricks_v1._cloud_cluster_id(),
    )


__all__ = [
    "GPU_QUALIFICATION_V2_ARTIFACT_KEYS",
    "GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SCRIPT",
    "GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SHA256",
    "GPU_QUALIFICATION_V2_DATABRICKS_PARAMETERS_MAX_BYTES",
    "GPU_QUALIFICATION_V2_EVIDENCE_FILENAME",
    "GPU_QUALIFICATION_V2_LOCAL_WORK_ROOT",
    "GPU_QUALIFICATION_V2_OUTPUT_FILENAME",
    "GPU_QUALIFICATION_V2_BATCH_MARKER_RECORD_TYPE",
    "GPU_QUALIFICATION_V2_PHASE_LEASE_RECORD_TYPE",
    "GPU_QUALIFICATION_V2_POST_INTENT_RECORD_TYPE",
    "GPU_QUALIFICATION_V2_SUBMIT_RECEIPT_RECORD_TYPE",
    "GPUQualificationSentinelRunnerV2",
    "collect_gpu_qualification_evidence_v2",
    "execute_gpu_qualification_job_v2",
    "main",
    "render_gpu_qualification_submit_payloads_v2",
    "replay_gpu_qualification_launch_authorization_v2",
    "resume_gpu_qualification_job_submissions_v2",
    "submit_gpu_qualification_jobs_v2",
    "validate_gpu_qualification_submit_payloads_v2",
    "write_gpu_qualification_bootstrap_runner_v2",
]
