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
from typing import Any, Final, Protocol, cast
import zlib

from document_kv_cache.databricks_resource_ledger import (
    DatabricksClusterHourLedger,
    DatabricksRunAttemptReservationRequest,
    canonical_databricks_submit_payload_snapshot,
    databricks_ledger_prefix_at_counts,
    databricks_ledger_prefix_from_record,
    databricks_ledger_path_sha256,
    read_databricks_cluster_hour_ledger_json,
    replay_databricks_run_attempt_batch_authorization_json,
    record_databricks_run_submission_receipt_json,
    require_databricks_ledger_prefix,
    require_databricks_publication_batch_admission,
    reserve_databricks_run_attempt_batch_authorized_json,
)
from document_kv_cache.databricks_runs import (
    DatabricksWorkspaceConfig,
    bind_databricks_run_idempotency_token,
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
    GPUQualificationArtifactPinsV2,
    build_gpu_job_result_v2,
    pins_from_gpu_qualification_plan_v2,
    validate_gpu_job_result_v2_record,
    validate_gpu_qualification_plan_v2_record,
    validate_local_preflight_evidence_v2_record,
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
            "PYTHONSAFEPATH": "1",
        }
    )
    return env


def _bootstrap(argv: list[str]) -> list[str]:
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
    if _sha256(runner_path) != pins["runner_sha256"]:
        raise ValueError("GPU qualification v2 bootstrap runner SHA-256 mismatch")
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
            env=_pip_subprocess_environment(),
        )
    finally:
        shutil.rmtree(stage)
    return argv


if __name__ == "__main__":
    remaining = _bootstrap(sys.argv[1:])
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
                "from document_kv_cache.gpu_qualification_databricks_v2 import main; raise SystemExit(main())",
                *remaining,
            ],
            cwd=safe_cwd,
            env=_pip_subprocess_environment(),
        )
    finally:
        shutil.rmtree(safe_cwd)
    raise SystemExit(completed.returncode)
"""
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


def submit_gpu_qualification_jobs_v2(
    config: DatabricksWorkspaceConfig,
    *,
    plan_record: Mapping[str, Any],
    single_user_name: str,
    artifact_uris: Mapping[str, str],
    output_root: str,
    ledger_path: str | Path,
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
        contracts=contracts,
        label="v2 qualification launch",
    )
    requests = _batch_requests_v2(plan, contracts)

    def validate_batch(
        live: DatabricksClusterHourLedger,
        reservations: tuple[Any, ...],
        snapshots: tuple[Mapping[str, Any], ...],
    ) -> None:
        _require_v2_ledger_predecessor(
            live,
            plan=plan,
            ledger_path=ledger_path,
            contracts=contracts,
            label="v2 qualification batch reservation",
        )
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
    receipts: list[dict[str, Any]] = []
    for contract in contracts:
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
    expected_names = {
        _V2_PHASE_LEASE_FILENAME,
        _V2_BATCH_MARKER_FILENAME,
        *(f"{contract['job_id']}.json" for contract in contracts),
    }
    if {item.name for item in receipt_root.iterdir()} != expected_names:
        raise ValueError("submitted v2 receipt directory is not closed")
    return tuple(receipts)


def resume_gpu_qualification_job_submissions_v2(
    config: DatabricksWorkspaceConfig,
    *,
    plan_record: Mapping[str, Any],
    single_user_name: str,
    artifact_uris: Mapping[str, str],
    output_root: str,
    ledger_path: str | Path,
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
    expected_names = {
        _V2_PHASE_LEASE_FILENAME,
        _V2_BATCH_MARKER_FILENAME,
        *(f"{contract['job_id']}.json" for contract in contracts),
    }
    if {item.name for item in root.iterdir()} != expected_names:
        raise ValueError("resumed v2 receipt directory is not closed")
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
    contracts: Sequence[Mapping[str, Any]],
    label: str,
) -> Any:
    if databricks_ledger_path_sha256(ledger_path) != plan.get(
        "campaign_ledger_path_sha256"
    ):
        raise ValueError("v2 ledger path differs from the plan")
    if ledger.ledger_id != plan.get("campaign_ledger_id"):
        raise ValueError("v2 ledger ID differs from the plan")
    predecessor = databricks_ledger_prefix_from_record(
        databricks_v1._required_mapping(
            plan.get("campaign_ledger_prefix"), "campaign_ledger_prefix"
        )
    )
    require_databricks_ledger_prefix(ledger, predecessor)
    if ledger.terminal_actual_cluster_hours != plan.get(
        "campaign_opening_terminal_gpu_hours"
    ):
        raise ValueError("v2 ledger opening terminal balance drift")
    databricks_v1._require_qualification_ledger_admission(
        ledger,
        proposed_task_count=len(contracts),
        proposed_reserved_cluster_hours=(
            len(contracts)
            * databricks_v1.GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS
            / 3600.0
        ),
        label=label,
    )
    return predecessor


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
    expected_predecessor = databricks_ledger_prefix_from_record(
        databricks_v1._required_mapping(
            plan.get("campaign_ledger_prefix"), "campaign_ledger_prefix"
        )
    )
    expected_lease = _phase_lease_record_v2(
        plan=plan,
        pins=pins,
        ledger_path_sha256=_required_sha256_v2(
            plan.get("campaign_ledger_path_sha256"),
            "campaign_ledger_path_sha256",
        ),
        predecessor_prefix=expected_predecessor,
        contracts=contracts,
        local_preflight_binding=local_preflight_binding,
    )
    if canonical_gpu_qualification_json(lease) != canonical_gpu_qualification_json(
        expected_lease
    ):
        raise ValueError("v2 phase lease differs from the frozen batch")
    _require_controller_record_seal_v2(lease, "v2 phase lease")
    if databricks_ledger_path_sha256(ledger_path) != plan.get(
        "campaign_ledger_path_sha256"
    ):
        raise ValueError("v2 replay ledger path differs from the plan")
    live = read_databricks_cluster_hour_ledger_json(ledger_path)
    if live.ledger_id != plan.get("campaign_ledger_id"):
        raise ValueError("v2 replay ledger ID differs from the plan")
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
    if databricks_ledger_prefix_at_counts(
        predecessor_ledger,
        reservation_count=len(predecessor_ledger.reservations),
        submission_receipt_count=len(predecessor_ledger.submission_receipts),
        terminal_actual_count=len(predecessor_ledger.terminal_actuals),
    ) != expected_predecessor:
        raise ValueError("v2 phase predecessor history drift")
    if predecessor_ledger.terminal_actual_cluster_hours != plan.get(
        "campaign_opening_terminal_gpu_hours"
    ):
        raise ValueError("v2 replay opening terminal balance drift")
    databricks_v1._require_qualification_ledger_admission(
        predecessor_ledger,
        proposed_task_count=len(contracts),
        proposed_reserved_cluster_hours=(
            len(contracts)
            * databricks_v1.GPU_QUALIFICATION_DATABRICKS_RUN_TIMEOUT_SECONDS
            / 3600.0
        ),
        label="v2 durable batch replay",
    )
    requests = _batch_requests_v2(plan, contracts)
    if (
        len(live.reservations) == expected_predecessor.reservation_count
        and len(live.submission_receipts)
        == expected_predecessor.submission_receipt_count
        and len(live.terminal_actuals) == expected_predecessor.terminal_actual_count
    ):
        if {item.name for item in root.iterdir()} != {_V2_PHASE_LEASE_FILENAME}:
            raise ValueError("v2 lease-only resume root has unexpected evidence")
        _reserved_ledger, authorization = (
            reserve_databricks_run_attempt_batch_authorized_json(
                ledger_path,
                requests,
                expected_predecessor_prefix=expected_predecessor,
            )
        )
    else:
        if (
            len(live.reservations)
            != expected_predecessor.reservation_count + len(contracts)
            or not (
                expected_predecessor.submission_receipt_count
                <= len(live.submission_receipts)
                <= expected_predecessor.submission_receipt_count + len(contracts)
            )
            or len(live.terminal_actuals)
            != expected_predecessor.terminal_actual_count
        ):
            raise ValueError("v2 replay ledger is not a canonical batch prefix")
        authorization = (
            replay_databricks_run_attempt_batch_authorization_json(
                ledger_path,
                requests,
                expected_predecessor_prefix=expected_predecessor,
            )
        )
    live = read_databricks_cluster_hour_ledger_json(ledger_path)
    receipt_start = expected_predecessor.submission_receipt_count
    receipt_suffix = live.submission_receipts[receipt_start:]
    receipt_count = len(receipt_suffix)
    if tuple(item.attempt_id for item in receipt_suffix) != (
        authorization.attempt_ids[:receipt_count]
    ):
        raise ValueError("v2 replay receipt suffix is not the canonical batch prefix")
    if tuple(item.submit_payload_sha256 for item in receipt_suffix) != (
        authorization.submit_payload_sha256s[:receipt_count]
    ):
        raise ValueError("v2 replay receipt suffix payload digest drift")
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
    batch_authorization: Any,
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
    intent_indices = tuple(
        index for index, name in enumerate(intent_names) if name in names
    )
    marker_path = root / _V2_BATCH_MARKER_FILENAME
    if _V2_BATCH_MARKER_FILENAME not in names:
        if names != {_V2_PHASE_LEASE_FILENAME} or receipt_count != 0:
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
        allowed_intent_indices = {(controller_receipt_count,)}
    else:
        allowed_intent_indices = {()}
        if receipt_count < len(contracts):
            allowed_intent_indices.add((receipt_count,))
        if receipt_count > 0:
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


def _utc_now() -> datetime:
    return datetime.now(UTC)


def main(argv: Sequence[str] | None = None) -> int:
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
        cloud_cluster_id=databricks_v1._cloud_cluster_id(),
        sentinel_runner=_builtin_sentinel_runner_v2,
    )
    return 0


__all__ = [
    "GPU_QUALIFICATION_V2_ARTIFACT_KEYS",
    "GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SCRIPT",
    "GPU_QUALIFICATION_V2_BOOTSTRAP_RUNNER_SHA256",
    "GPU_QUALIFICATION_V2_DATABRICKS_PARAMETERS_MAX_BYTES",
    "GPU_QUALIFICATION_V2_LOCAL_WORK_ROOT",
    "GPU_QUALIFICATION_V2_OUTPUT_FILENAME",
    "GPU_QUALIFICATION_V2_BATCH_MARKER_RECORD_TYPE",
    "GPU_QUALIFICATION_V2_PHASE_LEASE_RECORD_TYPE",
    "GPU_QUALIFICATION_V2_POST_INTENT_RECORD_TYPE",
    "GPU_QUALIFICATION_V2_SUBMIT_RECEIPT_RECORD_TYPE",
    "GPUQualificationSentinelRunnerV2",
    "execute_gpu_qualification_job_v2",
    "main",
    "render_gpu_qualification_submit_payloads_v2",
    "resume_gpu_qualification_job_submissions_v2",
    "submit_gpu_qualification_jobs_v2",
    "validate_gpu_qualification_submit_payloads_v2",
    "write_gpu_qualification_bootstrap_runner_v2",
]
