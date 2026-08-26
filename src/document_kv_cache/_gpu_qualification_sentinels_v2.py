"""Isolated runtime installer and dispatcher for GPU qualification v2."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.resources
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast
from urllib.parse import unquote, urlsplit

from document_kv_cache.flashinfer_wheel_repack import (
    FLASHINFER_PACKAGE_VERSION,
    FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256,
    FLASHINFER_PATCHED_MANIFEST_FILE_SHA256,
    FLASHINFER_PATCHED_WHEEL_SHA256,
    FLASHINFER_TARGET_MEMBER,
    FLASHINFER_TARGET_PATCHED_SHA256,
)
from document_kv_cache.gpu_qualification import canonical_gpu_qualification_json
import document_kv_cache.gpu_qualification as qualification_v1
from document_kv_cache.gpu_qualification_sentinels import (
    _make_site_packages_read_only,
    _run_bounded_worker_process,
    _verify_input_bundle_in_isolated_runtime,
    _worker_stream_diagnostic,
)
from document_kv_cache.gpu_qualification_v2 import (
    GPU_QUALIFICATION_V2_ARTIFACT_KEYS,
    GPU_QUALIFICATION_V2_CACHET_PACKAGE_VERSION,
    GPU_QUALIFICATION_V2_FLASHINFER_RETURN_ANNOTATION,
    GPU_QUALIFICATION_V2_INSTALLED_DISTRIBUTION_COUNT,
    GPU_QUALIFICATION_V2_WITH_FLASHINFER_DISTRIBUTION_COUNT,
    GPU_QUALIFICATION_V2_WITH_VLLM_DISTRIBUTION_COUNT,
    build_gpu_runtime_verification_v2,
    gpu_qualification_v2_runtime_closure,
    pins_from_gpu_qualification_plan_v2,
    validate_gpu_qualification_plan_v2_record,
    validate_gpu_qualification_v2_runtime_attestation,
)
from document_kv_cache.runtime_artifact_closure import (
    RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256,
    RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
    RUNTIME_ARTIFACT_CLOSURE_FILE_SIZE,
    RUNTIME_ARTIFACT_CLOSURE_RECORD_TYPE,
    RUNTIME_ARTIFACT_CLOSURE_SCHEMA_VERSION,
    VLLM_PACKAGE_VERSION,
    VLLM_PATCHED_MANIFEST_SHA256,
    VLLM_PATCHED_WHEEL_SHA256,
    VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT,
    VLLM_RUNTIME_BASE_LOCK_FILENAME,
    VLLM_RUNTIME_BASE_LOCK_HASH_COUNT,
    VLLM_RUNTIME_BASE_LOCK_SHA256,
)
from document_kv_cache.serving_env import (
    VLLM_PATCHED_WHEEL_SHA256_ENV,
    VLLM_PATCHED_WHEEL_URI_ENV,
)
from document_kv_cache.vllm_smoke import (
    _attest_isolated_python,
    _pip_subprocess_environment,
    create_venv,
)


_RUNTIME_LOCK_ATTESTATION_ENV: Final = (
    "CACHET_GPU_QUALIFICATION_RUNTIME_LOCK_ATTESTATION"
)
_BASE_LOCK_REQUIREMENT_RE: Final = re.compile(r"^([A-Za-z0-9_.-]+)==([^ ]+?)(?: \\)?$")
_BASE_LOCK_HASH_RE: Final = re.compile(r"^\s+--hash=sha256:[0-9a-f]{64}(?: \\)?$")
_CANONICAL_NAME_RE: Final = re.compile(r"[-_.]+")
_RUNTIME_CLOSURE = gpu_qualification_v2_runtime_closure()
_VLLM_MEMBER_SHA256: Final = MappingProxyType(
    dict(
        cast(
            Mapping[str, str],
            cast(Mapping[str, Any], _RUNTIME_CLOSURE["vllm"])["member_sha256"],
        )
    )
)
del _RUNTIME_CLOSURE


def run_gpu_qualification_sentinel_v2(
    *,
    plan_record: Mapping[str, Any],
    planned_job: Mapping[str, Any],
    artifact_paths: Mapping[str, Path],
    work_dir: Path,
) -> Mapping[str, Any]:
    """Install the exact four-step runtime and run the bounded GPU worker."""

    if tuple(artifact_paths) != GPU_QUALIFICATION_V2_ARTIFACT_KEYS:
        raise ValueError("v2 sentinel artifacts lack canonical eight-role coverage")
    pins = pins_from_gpu_qualification_plan_v2(plan_record)
    validate_gpu_qualification_plan_v2_record(
        plan_record,
        expected_campaign_id=_required_string(
            plan_record.get("campaign_id"), "campaign_id"
        ),
        expected_artifact_pins=pins,
    )
    job_id = _required_string(planned_job.get("job_id"), "job_id")
    expected_job = qualification_v1._plan_job(plan_record, job_id)  # noqa: SLF001
    if canonical_gpu_qualification_json(planned_job) != (
        canonical_gpu_qualification_json(expected_job)
    ):
        raise ValueError("v2 planned job differs from the sealed plan")
    pin_record = pins.to_record()
    for key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS:
        path = artifact_paths[key]
        if key == "input_bundle_sha256":
            if not path.is_dir() or path.is_symlink():
                raise ValueError("v2 input bundle snapshot is not a regular directory")
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"v2 artifact {key} is not one regular file")
        if _file_sha256(path) != pin_record[key]:
            raise ValueError(f"v2 artifact {key} differs from the plan")
    closure_path = artifact_paths["runtime_closure_manifest_sha256"]
    _read_exact_runtime_closure(closure_path)

    runtime_dir = work_dir / "runtime"
    runtime_python = runtime_dir / "bin" / "python"
    expected_python_version = _runtime_python_version(plan_record)
    create_venv(runtime_dir, copies=True)
    created_python_identity = _attest_isolated_python(
        runtime_dir,
        expected_python_version=expected_python_version,
    )
    environment = _pip_subprocess_environment()
    environment["PYTHONSAFEPATH"] = "1"
    runtime_lock = artifact_paths["runtime_lock_sha256"]
    patched_vllm = artifact_paths["patched_vllm_wheel_sha256"]
    patched_flashinfer = artifact_paths["patched_flashinfer_wheel_sha256"]
    package_wheel = artifact_paths["package_wheel_sha256"]
    vllm_uri = patched_vllm.resolve(strict=True).as_uri()
    flashinfer_uri = patched_flashinfer.resolve(strict=True).as_uri()
    package_uri = package_wheel.resolve(strict=True).as_uri()
    commands = (
        (
            "base_lock",
            (
                "--require-hashes",
                "--only-binary",
                ":all:",
                "-r",
                str(runtime_lock),
            ),
        ),
        (
            "vllm",
            (
                "--no-deps",
                f"vllm @ {vllm_uri}#sha256={pins.patched_vllm_wheel_sha256}",
            ),
        ),
        (
            "flashinfer",
            (
                "--no-deps",
                "flashinfer-python @ "
                f"{flashinfer_uri}#sha256={pins.patched_flashinfer_wheel_sha256}",
            ),
        ),
        (
            "cachet",
            (
                "--no-deps",
                f"cachet-kv @ {package_uri}#sha256={pins.package_wheel_sha256}",
            ),
        ),
    )
    if tuple(label for label, _arguments in commands) != (
        "base_lock",
        "vllm",
        "flashinfer",
        "cachet",
    ):
        raise RuntimeError("v2 runtime install order differs")
    for _label, arguments in commands:
        subprocess.run(
            [
                str(runtime_python),
                "-m",
                "pip",
                "install",
                *arguments,
            ],
            check=True,
            timeout=3_600,
            env=environment,
            cwd=runtime_dir,
        )
    subprocess.run(
        [str(runtime_python), "-m", "pip", "check"],
        check=True,
        timeout=300,
        env=environment,
        cwd=runtime_dir,
    )
    installed_python_identity = _attest_isolated_python(
        runtime_dir,
        expected_python_version=expected_python_version,
        expected_file_binding=created_python_identity.file_binding,
    )
    if installed_python_identity != created_python_identity:
        raise RuntimeError("v2 isolated Python identity changed during installation")

    attestation = _run_final_runtime_verifier(
        runtime_python,
        runtime_lock=runtime_lock,
        vllm_uri=vllm_uri,
        flashinfer_uri=flashinfer_uri,
        closure_path=closure_path,
        package_uri=package_uri,
        package_sha256=pins.package_wheel_sha256,
        environment=environment,
    )
    validate_gpu_qualification_v2_runtime_attestation(attestation)
    os.environ[VLLM_PATCHED_WHEEL_URI_ENV] = str(patched_vllm)
    os.environ[VLLM_PATCHED_WHEEL_SHA256_ENV] = pins.patched_vllm_wheel_sha256

    worker_environment = dict(environment)
    worker_environment.update(
        {
            "HF_HOME": "/local_disk0/cachet-vllm-0271-hf",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONSAFEPATH": "1",
            "TOKENIZERS_PARALLELISM": "false",
            VLLM_PATCHED_WHEEL_URI_ENV: str(patched_vllm),
            VLLM_PATCHED_WHEEL_SHA256_ENV: pins.patched_vllm_wheel_sha256,
            _RUNTIME_LOCK_ATTESTATION_ENV: json.dumps(
                attestation,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
    )
    _verify_input_bundle_in_isolated_runtime(
        runtime_python,
        artifact_paths["input_bundle_sha256"],
        expected_sha256=pins.input_bundle_sha256,
        environment=worker_environment,
    )

    worker_dir = work_dir / "worker"
    worker_dir.mkdir()
    job_path = worker_dir / "planned-job.json"
    job_path.write_text(
        canonical_gpu_qualification_json(planned_job) + "\n", encoding="utf-8"
    )
    plan_path = worker_dir / "plan.json"
    plan_path.write_text(
        canonical_gpu_qualification_json(plan_record) + "\n", encoding="utf-8"
    )
    _make_site_packages_read_only(runtime_python)
    worker_output = worker_dir / "measurements.json"
    completed = _run_bounded_worker_process(
        [
            str(runtime_python),
            "-m",
            "document_kv_cache._gpu_qualification_sentinel_worker",
            "--plan-json",
            str(plan_path),
            "--job-json",
            str(job_path),
            "--input-bundle",
            str(artifact_paths["input_bundle_sha256"]),
            "--work-dir",
            str(worker_dir / "runtime-work"),
            "--output-json",
            str(worker_output),
        ],
        job_id=job_id,
        timeout_seconds=14_000,
        environment=worker_environment,
        cwd=worker_dir,
    )
    if not worker_output.is_file() or worker_output.is_symlink():
        raise RuntimeError(
            f"GPU sentinel {job_id!r} did not write its measurement record; "
            f"{_worker_stream_diagnostic('stdout', completed.stdout)}; "
            f"{_worker_stream_diagnostic('stderr', completed.stderr)}"
        )
    measurements = json.loads(worker_output.read_text(encoding="utf-8"))
    if not isinstance(measurements, dict):
        raise RuntimeError("GPU sentinel v2 measurement output must be an object")
    runtime_verification = build_gpu_runtime_verification_v2(
        plan_sha256=_required_string(
            plan_record.get("closed_record_sha256"), "plan_sha256"
        ),
        job_id=job_id,
        artifact_sha256=pin_record,
        attestation=attestation,
    )
    return {
        "measurements": measurements,
        "runtime_verification": runtime_verification,
    }


def verify_gpu_qualification_v2_runtime_installation(
    *,
    runtime_lock: str | Path,
    vllm_uri: str,
    flashinfer_uri: str,
    runtime_closure_manifest: str | Path,
    package_uri: str,
    package_sha256: str,
) -> dict[str, Any]:
    """Run the final Linux/CPython/install/provenance/import verifier."""

    _require_runtime_platform()
    verifier_environment = _pip_subprocess_environment()
    verifier_environment["PYTHONSAFEPATH"] = "1"
    subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=True,
        timeout=300,
        env=verifier_environment,
        cwd=Path(sys.prefix),
    )
    pip_check_ok = True
    lock_path = Path(runtime_lock)
    if _file_sha256(lock_path) != VLLM_RUNTIME_BASE_LOCK_SHA256:
        raise RuntimeError("v2 installed runtime base lock SHA-256 differs")
    versions, hash_count = _base_lock_projection(lock_path)
    if (
        len(versions) != VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT
        or hash_count != VLLM_RUNTIME_BASE_LOCK_HASH_COUNT
        or "flashinfer-python" in versions
        or "vllm" in versions
        or "cachet-kv" in versions
    ):
        raise RuntimeError("v2 installed runtime base lock closure differs")
    closure = _read_exact_runtime_closure(Path(runtime_closure_manifest))

    installed: dict[str, list[str]] = {}
    distributions: dict[str, importlib.metadata.Distribution] = {}
    for distribution in importlib.metadata.distributions():
        try:
            raw_name = distribution.metadata["Name"]
        except KeyError as exc:
            raise RuntimeError("installed distribution has no package name") from exc
        if not isinstance(raw_name, str) or not raw_name:
            raise RuntimeError("installed distribution has an invalid package name")
        name = _canonical_name(raw_name)
        installed.setdefault(name, []).append(distribution.version)
        if name in distributions:
            raise RuntimeError(f"duplicate installed distribution {name!r}")
        distributions[name] = distribution
    expected_names = {
        *versions,
        "cachet-kv",
        "flashinfer-python",
        "vllm",
    }
    unexpected = sorted(set(installed) - expected_names)
    if unexpected or set(installed) != expected_names:
        raise RuntimeError("v2 installed distribution name closure differs")
    for name, expected_version in versions.items():
        if installed.get(name) != [expected_version]:
            raise RuntimeError(f"v2 installed base distribution {name!r} differs")
    if len(installed) != GPU_QUALIFICATION_V2_INSTALLED_DISTRIBUTION_COUNT:
        raise RuntimeError("v2 installed distribution count differs")
    if installed.get("vllm") != [VLLM_PACKAGE_VERSION]:
        raise RuntimeError("v2 installed vLLM version differs")
    if installed.get("flashinfer-python") != [FLASHINFER_PACKAGE_VERSION]:
        raise RuntimeError("v2 installed FlashInfer version differs")
    cachet_versions = installed.get("cachet-kv")
    if cachet_versions != [GPU_QUALIFICATION_V2_CACHET_PACKAGE_VERSION]:
        raise RuntimeError("v2 installed Cachet package identity differs")

    _require_sha256(package_sha256, "package_sha256")
    if _uri_file_sha256(package_uri) != package_sha256:
        raise RuntimeError("v2 Cachet package URI bytes differ")
    vllm_direct_url = _validate_direct_url(
        distributions["vllm"],
        expected_uri=vllm_uri,
        expected_sha256=VLLM_PATCHED_WHEEL_SHA256,
    )
    flashinfer_direct_url = _validate_direct_url(
        distributions["flashinfer-python"],
        expected_uri=flashinfer_uri,
        expected_sha256=FLASHINFER_PATCHED_WHEEL_SHA256,
    )
    _validate_direct_url(
        distributions["cachet-kv"],
        expected_uri=package_uri,
        expected_sha256=package_sha256,
    )
    vllm_members = _installed_member_hashes(distributions["vllm"], _VLLM_MEMBER_SHA256)
    flashinfer_members = _installed_member_hashes(
        distributions["flashinfer-python"],
        {FLASHINFER_TARGET_MEMBER: FLASHINFER_TARGET_PATCHED_SHA256},
    )
    if flashinfer_members[FLASHINFER_TARGET_MEMBER] != (
        FLASHINFER_TARGET_PATCHED_SHA256
    ):
        raise RuntimeError("v2 installed FlashInfer patched member differs")
    module = importlib.import_module("flashinfer.comm.fd_exchange")
    module_file = getattr(module, "__file__", None)
    expected_module_file = (
        Path(str(distributions["flashinfer-python"].locate_file("")))
        / FLASHINFER_TARGET_MEMBER
    )
    if not isinstance(module_file, str) or not module_file:
        raise RuntimeError("v2 imported FlashInfer module has no source origin")
    try:
        observed_module_file = Path(module_file).resolve(strict=True)
        reviewed_module_file = expected_module_file.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            "v2 imported FlashInfer module origin is unavailable"
        ) from exc
    if (
        observed_module_file != reviewed_module_file
        or _file_sha256(observed_module_file) != FLASHINFER_TARGET_PATCHED_SHA256
    ):
        raise RuntimeError("v2 imported FlashInfer module origin differs")
    function = getattr(module, "_fd_ancillary", None)
    annotations = getattr(function, "__annotations__", None)
    if not isinstance(annotations, dict) or annotations.get("return") != (
        GPU_QUALIFICATION_V2_FLASHINFER_RETURN_ANNOTATION
    ):
        raise RuntimeError("v2 FlashInfer postponed return annotation differs")
    packaged_lock = (
        importlib.resources.files("document_kv_cache")
        .joinpath("runtime_locks")
        .joinpath(VLLM_RUNTIME_BASE_LOCK_FILENAME)
        .read_bytes()
    )
    packaged_lock_sha256 = sha256(packaged_lock).hexdigest()
    if packaged_lock_sha256 != VLLM_RUNTIME_BASE_LOCK_SHA256:
        raise RuntimeError("v2 installed Cachet package base lock differs")
    record = {
        "base_lock_distribution_count": len(versions),
        "base_lock_hash_count": hash_count,
        "base_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
        "cachet_package_version": cachet_versions[0],
        "flashinfer_annotation": GPU_QUALIFICATION_V2_FLASHINFER_RETURN_ANNOTATION,
        "flashinfer_direct_url": flashinfer_direct_url,
        "flashinfer_import_ok": True,
        "closure_bound_flashinfer_manifest_closed_record_sha256": (
            FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256
        ),
        "closure_bound_flashinfer_manifest_file_sha256": (
            FLASHINFER_PATCHED_MANIFEST_FILE_SHA256
        ),
        "closure_bound_vllm_manifest_file_sha256": VLLM_PATCHED_MANIFEST_SHA256,
        "flashinfer_member_sha256": flashinfer_members[FLASHINFER_TARGET_MEMBER],
        "flashinfer_package_version": FLASHINFER_PACKAGE_VERSION,
        "flashinfer_wheel_sha256": FLASHINFER_PATCHED_WHEEL_SHA256,
        "installed_distribution_count": len(installed),
        "ok": True,
        "packaged_base_lock_sha256": packaged_lock_sha256,
        "pip_check_ok": pip_check_ok,
        "runtime_closure_closed_record_sha256": closure["closed_record_sha256"],
        "runtime_closure_file_sha256": RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        "unexpected_distributions": unexpected,
        "vllm_direct_url": vllm_direct_url,
        "vllm_member_sha256": vllm_members,
        "vllm_package_version": VLLM_PACKAGE_VERSION,
        "vllm_wheel_sha256": VLLM_PATCHED_WHEEL_SHA256,
        "with_flashinfer_distribution_count": (
            GPU_QUALIFICATION_V2_WITH_FLASHINFER_DISTRIBUTION_COUNT
        ),
        "with_vllm_distribution_count": (
            GPU_QUALIFICATION_V2_WITH_VLLM_DISTRIBUTION_COUNT
        ),
    }
    validate_gpu_qualification_v2_runtime_attestation(record)
    return record


def _run_final_runtime_verifier(
    runtime_python: Path,
    *,
    runtime_lock: Path,
    vllm_uri: str,
    flashinfer_uri: str,
    closure_path: Path,
    package_uri: str,
    package_sha256: str,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    code = (
        "import json,sys;"
        "from document_kv_cache._gpu_qualification_sentinels_v2 import "
        "verify_gpu_qualification_v2_runtime_installation as verify;"
        "print(json.dumps(verify(runtime_lock=sys.argv[1],vllm_uri=sys.argv[2],"
        "flashinfer_uri=sys.argv[3],runtime_closure_manifest=sys.argv[4],"
        "package_uri=sys.argv[5],package_sha256=sys.argv[6]),"
        "allow_nan=False,separators=(',',':'),sort_keys=True))"
    )
    completed = subprocess.run(
        [
            str(runtime_python),
            "-c",
            code,
            str(runtime_lock),
            vllm_uri,
            flashinfer_uri,
            str(closure_path),
            package_uri,
            package_sha256,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env=dict(environment),
        cwd=runtime_python.parent.parent,
    )
    try:
        record = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("v2 final runtime verifier returned invalid JSON") from exc
    if not isinstance(record, dict):
        raise RuntimeError("v2 final runtime verifier did not return an object")
    return record


def _read_exact_runtime_closure(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("v2 runtime closure is not one regular file")
    raw = path.read_bytes()
    if (
        len(raw) != RUNTIME_ARTIFACT_CLOSURE_FILE_SIZE
        or sha256(raw).hexdigest() != RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256
    ):
        raise ValueError("v2 runtime closure file identity differs")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("v2 runtime closure is not UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("v2 runtime closure must contain an object")
    expected_bytes = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if raw != expected_bytes:
        raise ValueError("v2 runtime closure is not canonical JSON")
    if (
        value.get("record_type") != RUNTIME_ARTIFACT_CLOSURE_RECORD_TYPE
        or value.get("schema_version") != RUNTIME_ARTIFACT_CLOSURE_SCHEMA_VERSION
        or value.get("closed_record_sha256")
        != RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256
    ):
        raise ValueError("v2 runtime closure record identity differs")
    payload = dict(value)
    payload.pop("closed_record_sha256")
    if (
        sha256(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        != RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256
    ):
        raise ValueError("v2 runtime closure seal differs")
    return value


def _base_lock_projection(path: Path) -> tuple[dict[str, str], int]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("v2 base lock is not UTF-8") from exc
    versions: dict[str, str] = {}
    hash_count = 0
    for line in text.splitlines():
        requirement = _BASE_LOCK_REQUIREMENT_RE.fullmatch(line)
        if requirement is not None:
            name = _canonical_name(requirement.group(1))
            if name in versions:
                raise ValueError("v2 base lock repeats a distribution")
            versions[name] = requirement.group(2)
        if _BASE_LOCK_HASH_RE.fullmatch(line) is not None:
            hash_count += 1
    return versions, hash_count


def _validate_direct_url(
    distribution: importlib.metadata.Distribution,
    *,
    expected_uri: str,
    expected_sha256: str,
) -> str:
    if _uri_file_sha256(expected_uri) != expected_sha256:
        raise RuntimeError("v2 direct-install source bytes differ")
    text = distribution.read_text("direct_url.json")
    if text is None:
        raise RuntimeError("v2 direct install lacks PEP 610 direct_url.json")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("v2 PEP 610 direct_url.json is invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeError("v2 PEP 610 direct_url.json must contain an object")
    observed_uri = value.get("url")
    expected_canonical_uri = _normalized_uri(expected_uri)
    if not isinstance(observed_uri, str) or _normalized_uri(observed_uri) != (
        expected_canonical_uri
    ):
        raise RuntimeError("v2 PEP 610 direct URL differs")
    archive = value.get("archive_info")
    hashes = archive.get("hashes") if isinstance(archive, dict) else None
    if not isinstance(hashes, dict) or hashes.get("sha256") != expected_sha256:
        raise RuntimeError("v2 PEP 610 archive SHA-256 differs")
    return expected_canonical_uri


def _installed_member_hashes(
    distribution: importlib.metadata.Distribution,
    expected: Mapping[str, str],
) -> dict[str, str]:
    root = Path(str(distribution.locate_file("")))
    observed: dict[str, str] = {}
    for relative_path, expected_sha256 in expected.items():
        path = root / relative_path
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"v2 installed member is missing: {relative_path}")
        digest = _file_sha256(path)
        if digest != expected_sha256:
            raise RuntimeError(f"v2 installed member differs: {relative_path}")
        observed[relative_path] = digest
    return observed


def _require_runtime_platform() -> None:
    if (
        platform.python_implementation() != "CPython"
        or sys.version_info[:2] != (3, 11)
        or platform.python_version() != "3.11.11"
        or sys.platform != "linux"
        or platform.machine().lower() not in {"x86_64", "amd64"}
        or platform.libc_ver() != ("glibc", "2.35")
    ):
        raise RuntimeError(
            "v2 final runtime verifier requires Linux CPython3.11 x86_64 glibc2.35"
        )


def _runtime_python_version(plan_record: Mapping[str, Any]) -> str:
    runtime = plan_record.get("runtime_contract")
    if not isinstance(runtime, Mapping):
        raise ValueError("v2 plan runtime_contract must be an object")
    platform_record = runtime.get("platform")
    if not isinstance(platform_record, Mapping):
        raise ValueError("v2 plan runtime platform must be an object")
    return _required_string(platform_record.get("python_version"), "python_version")


def _normalized_uri(value: str) -> str:
    parsed = urlsplit(value)
    decoded_path = unquote(parsed.path)
    if (
        parsed.scheme != "file"
        or parsed.netloc not in {"", "localhost"}
        or parsed.query
        or parsed.fragment
        or decoded_path.startswith("//")
        or any(
            ord(character) < 32 or ord(character) == 127 for character in decoded_path
        )
        or any(part in {".", ".."} for part in decoded_path.split("/"))
    ):
        raise RuntimeError("v2 direct runtime origin must be a local file URI")
    path = Path(decoded_path)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RuntimeError("v2 direct runtime origin must be one regular file")
    return path.resolve(strict=True).as_uri()


def _uri_file_sha256(value: str) -> str:
    normalized = _normalized_uri(value)
    parsed = urlsplit(normalized)
    path = Path(unquote(parsed.path))
    return _file_sha256(path)


def _canonical_name(value: str) -> str:
    return _CANONICAL_NAME_RE.sub("-", value).lower()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a non-empty canonical string")
    return value


def _require_sha256(value: Any, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be lowercase SHA-256")


__all__ = [
    "run_gpu_qualification_sentinel_v2",
    "verify_gpu_qualification_v2_runtime_installation",
]
