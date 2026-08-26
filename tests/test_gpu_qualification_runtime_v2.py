from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

import document_kv_cache._gpu_qualification_sentinel_worker as sentinel_worker
import document_kv_cache._gpu_qualification_sentinels_v2 as runtime_v2
import document_kv_cache.gpu_qualification as qualification_v1
from document_kv_cache.flashinfer_wheel_repack import (
    FLASHINFER_PACKAGE_VERSION,
    FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256,
    FLASHINFER_PATCHED_MANIFEST_FILE_SHA256,
    FLASHINFER_PATCHED_WHEEL_SHA256,
    FLASHINFER_TARGET_MEMBER,
    FLASHINFER_TARGET_PATCHED_SHA256,
)
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
    GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    GPU_QUALIFICATION_VLLM_VERSION,
    canonical_gpu_qualification_json,
)
from document_kv_cache.gpu_qualification_v2 import (
    GPU_QUALIFICATION_V2_ARTIFACT_KEYS,
    GPU_QUALIFICATION_V2_CACHET_PACKAGE_VERSION,
    GPU_QUALIFICATION_V2_FLASHINFER_RETURN_ANNOTATION,
    GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX,
    GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS,
    GPUQualificationArtifactPinsV2,
    build_gpu_qualification_plan_v2,
    gpu_qualification_v2_runtime_closure,
    validate_gpu_runtime_verification_v2_record,
)
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
    PUBLICATION_CAMPAIGN_ID,
    PUBLICATION_CAMPAIGN_LEDGER_ID,
    PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
)
from document_kv_cache.runtime_artifact_closure import (
    RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256,
    RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
    RUNTIME_ARTIFACT_CLOSURE_FILE_SIZE,
    VLLM_PATCHED_MANIFEST_SHA256,
    VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT,
    VLLM_RUNTIME_BASE_LOCK_FILENAME,
    VLLM_RUNTIME_BASE_LOCK_HASH_COUNT,
    VLLM_RUNTIME_BASE_LOCK_SHA256,
    VLLM_RUNTIME_BASE_LOCK_SIZE,
)
from document_kv_cache.serving_env import (
    VLLM_PATCHED_WHEEL_SHA256_ENV,
    VLLM_PATCHED_WHEEL_URI_ENV,
)


_ROOT = Path(__file__).resolve().parents[1]
_LOCK_PATH = (
    _ROOT
    / "src"
    / "document_kv_cache"
    / "runtime_locks"
    / VLLM_RUNTIME_BASE_LOCK_FILENAME
)
_CLOSURE_PATH = (
    _ROOT
    / "databricks-runs"
    / "_campaign-inputs"
    / "vllm-0.27.1-runtime-closure"
    / "sha256"
    / RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256
    / "vllm-0.27.1-flashinfer-0.6.16.post3-runtime-closure.json"
)
_PACKAGE_SHA256 = "a" * 64
_SOURCE_SHA256 = "b" * 64
_RUNNER_SHA256 = "c" * 64


def _pins() -> GPUQualificationArtifactPinsV2:
    return GPUQualificationArtifactPinsV2(
        runtime_lock_sha256=VLLM_RUNTIME_BASE_LOCK_SHA256,
        patched_vllm_wheel_sha256=GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        patched_flashinfer_wheel_sha256=FLASHINFER_PATCHED_WHEEL_SHA256,
        runtime_closure_manifest_sha256=RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        package_wheel_sha256=_PACKAGE_SHA256,
        cachet_source_tree_sha256=_SOURCE_SHA256,
        runner_sha256=_RUNNER_SHA256,
        input_bundle_sha256=GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
    )


def _plan() -> dict[str, Any]:
    return build_gpu_qualification_plan_v2(
        campaign_id=PUBLICATION_CAMPAIGN_ID,
        campaign_record_sha256=PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
        campaign_ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
        campaign_ledger_path_sha256=PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
        campaign_ledger_prefix=GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX,
        campaign_opening_terminal_gpu_hours=(
            GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS
        ),
        artifact_pins=_pins(),
    )


def _seal_v2(record: dict[str, Any]) -> None:
    record["closed_record_sha256"] = ""
    record["closed_record_sha256"] = sha256(
        canonical_gpu_qualification_json(record).encode("utf-8")
    ).hexdigest()


def _attestation(*, vllm_uri: str, flashinfer_uri: str) -> dict[str, Any]:
    closure = gpu_qualification_v2_runtime_closure()
    return {
        "base_lock_distribution_count": VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT,
        "base_lock_hash_count": VLLM_RUNTIME_BASE_LOCK_HASH_COUNT,
        "base_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
        "cachet_package_version": GPU_QUALIFICATION_V2_CACHET_PACKAGE_VERSION,
        "flashinfer_annotation": GPU_QUALIFICATION_V2_FLASHINFER_RETURN_ANNOTATION,
        "flashinfer_direct_url": flashinfer_uri,
        "flashinfer_import_ok": True,
        "closure_bound_flashinfer_manifest_closed_record_sha256": (
            FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256
        ),
        "closure_bound_flashinfer_manifest_file_sha256": (
            FLASHINFER_PATCHED_MANIFEST_FILE_SHA256
        ),
        "closure_bound_vllm_manifest_file_sha256": VLLM_PATCHED_MANIFEST_SHA256,
        "flashinfer_member_sha256": FLASHINFER_TARGET_PATCHED_SHA256,
        "flashinfer_package_version": FLASHINFER_PACKAGE_VERSION,
        "flashinfer_wheel_sha256": FLASHINFER_PATCHED_WHEEL_SHA256,
        "installed_distribution_count": 198,
        "ok": True,
        "packaged_base_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
        "pip_check_ok": True,
        "runtime_closure_closed_record_sha256": (
            RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256
        ),
        "runtime_closure_file_sha256": RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        "unexpected_distributions": [],
        "vllm_direct_url": vllm_uri,
        "vllm_member_sha256": closure["vllm"]["member_sha256"],
        "vllm_package_version": GPU_QUALIFICATION_VLLM_VERSION,
        "vllm_wheel_sha256": GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        "with_flashinfer_distribution_count": 196,
        "with_vllm_distribution_count": 197,
    }


def _artifact_paths(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir()
    paths: dict[str, Path] = {}
    for key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS:
        path = tmp_path / key
        if key == "input_bundle_sha256":
            path.mkdir()
        else:
            path.write_text(key, encoding="utf-8")
        paths[key] = path
    return paths


def test_worker_non_v2_attestation_dispatch_is_exact_legacy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = {"legacy": "exact"}
    calls = 0

    def legacy() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return marker

    monkeypatch.setattr(sentinel_worker, "_runtime_lock_attestation", legacy)
    assert sentinel_worker._runtime_lock_attestation_for_plan({}) is marker
    assert (
        sentinel_worker._runtime_lock_attestation_for_plan(
            {
                "record_type": "cachet.vllm_0271_gpu_qualification_plan.v1",
                "schema_version": 1,
            }
        )
        is marker
    )
    assert calls == 2


def test_runtime_installer_uses_exact_commands_environment_and_sequence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pins = _pins()
    plan = _plan()
    job_id = plan["cloud_qualification"]["jobs"][0]["job_id"]
    planned_job = deepcopy(qualification_v1._plan_job(plan, job_id))
    artifact_paths = _artifact_paths(tmp_path / "artifacts")
    work_dir = tmp_path / "work"
    runtime_dir = work_dir / "runtime"
    runtime_python = runtime_dir / "bin" / "python"
    events: list[str] = []
    subprocess_calls: list[tuple[list[str], dict[str, Any]]] = []
    identity = SimpleNamespace(file_binding="reviewed-python-binding")

    digest_by_path = {
        artifact_paths[key]: pins.to_record()[key]
        for key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS
        if key != "input_bundle_sha256"
    }
    monkeypatch.setattr(runtime_v2, "_file_sha256", lambda path: digest_by_path[path])
    monkeypatch.setattr(runtime_v2, "_read_exact_runtime_closure", lambda _path: {})

    def create_venv(path: Path, *, copies: bool) -> None:
        assert copies is True
        (path / "bin").mkdir(parents=True)
        (path / "bin" / "python").write_bytes(b"python")

    attest_calls: list[tuple[str, str | None]] = []

    def attest(
        path: Path,
        *,
        expected_python_version: str,
        expected_file_binding: str | None = None,
    ) -> SimpleNamespace:
        assert path == runtime_dir
        assert expected_python_version == "3.11.11"
        attest_calls.append((expected_python_version, expected_file_binding))
        return identity

    def run(
        arguments: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        subprocess_calls.append((list(arguments), dict(kwargs)))
        if arguments[-1] == "check":
            events.append("pip-check")
        else:
            events.append(f"install-{len(subprocess_calls)}")
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    monkeypatch.setattr(runtime_v2, "create_venv", create_venv)
    monkeypatch.setattr(runtime_v2, "_attest_isolated_python", attest)
    monkeypatch.setattr(runtime_v2, "_pip_subprocess_environment", lambda: {})
    monkeypatch.setattr(runtime_v2.subprocess, "run", run)

    vllm_uri = artifact_paths["patched_vllm_wheel_sha256"].resolve().as_uri()
    flashinfer_uri = (
        artifact_paths["patched_flashinfer_wheel_sha256"].resolve().as_uri()
    )
    package_uri = artifact_paths["package_wheel_sha256"].resolve().as_uri()

    def final_verifier(
        observed_python: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        events.append("final-verifier")
        assert observed_python == runtime_python
        assert kwargs == {
            "closure_path": artifact_paths["runtime_closure_manifest_sha256"],
            "environment": {"PYTHONSAFEPATH": "1"},
            "flashinfer_uri": flashinfer_uri,
            "package_sha256": _PACKAGE_SHA256,
            "package_uri": package_uri,
            "runtime_lock": artifact_paths["runtime_lock_sha256"],
            "vllm_uri": vllm_uri,
        }
        return _attestation(vllm_uri=vllm_uri, flashinfer_uri=flashinfer_uri)

    monkeypatch.setattr(runtime_v2, "_run_final_runtime_verifier", final_verifier)

    def verify_input(
        observed_python: Path,
        input_bundle: Path,
        *,
        expected_sha256: str,
        environment: dict[str, str],
    ) -> None:
        events.append("input-verifier")
        assert observed_python == runtime_python
        assert input_bundle == artifact_paths["input_bundle_sha256"]
        assert expected_sha256 == GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256
        assert environment["PYTHONSAFEPATH"] == "1"

    monkeypatch.setattr(
        runtime_v2, "_verify_input_bundle_in_isolated_runtime", verify_input
    )
    monkeypatch.setattr(runtime_v2, "_make_site_packages_read_only", lambda _p: None)

    def worker(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        events.append("worker")
        output_path = Path(arguments[arguments.index("--output-json") + 1])
        output_path.write_text("{}\n", encoding="utf-8")
        assert kwargs["environment"]["PYTHONSAFEPATH"] == "1"
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    monkeypatch.setattr(runtime_v2, "_run_bounded_worker_process", worker)
    monkeypatch.setenv(VLLM_PATCHED_WHEEL_URI_ENV, "prior-uri")
    monkeypatch.setenv(VLLM_PATCHED_WHEEL_SHA256_ENV, "prior-sha")

    result = runtime_v2.run_gpu_qualification_sentinel_v2(
        plan_record=plan,
        planned_job=planned_job,
        artifact_paths=artifact_paths,
        work_dir=work_dir,
    )

    expected_commands = [
        [
            str(runtime_python),
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "--only-binary",
            ":all:",
            "-r",
            str(artifact_paths["runtime_lock_sha256"]),
        ],
        [
            str(runtime_python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            f"vllm @ {vllm_uri}#sha256={GPU_QUALIFICATION_PATCHED_WHEEL_SHA256}",
        ],
        [
            str(runtime_python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "flashinfer-python @ "
            f"{flashinfer_uri}#sha256={FLASHINFER_PATCHED_WHEEL_SHA256}",
        ],
        [
            str(runtime_python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            f"cachet-kv @ {package_uri}#sha256={_PACKAGE_SHA256}",
        ],
        [str(runtime_python), "-m", "pip", "check"],
    ]
    assert [call[0] for call in subprocess_calls] == expected_commands
    assert "--no-deps" not in subprocess_calls[0][0]
    assert all("--no-deps" in call[0] for call in subprocess_calls[1:4])
    for _arguments, kwargs in subprocess_calls:
        assert kwargs["check"] is True
        assert kwargs["cwd"] == runtime_dir
        assert kwargs["env"]["PYTHONSAFEPATH"] == "1"
    assert [call[1]["timeout"] for call in subprocess_calls] == [3600] * 4 + [300]
    assert attest_calls == [("3.11.11", None), ("3.11.11", identity.file_binding)]
    assert events == [
        "install-1",
        "install-2",
        "install-3",
        "install-4",
        "pip-check",
        "final-verifier",
        "input-verifier",
        "worker",
    ]
    assert result["measurements"] == {}
    runtime_verification = result["runtime_verification"]
    assert (
        runtime_verification["closed_record_sha256"]
        == sha256(
            canonical_gpu_qualification_json(
                {**runtime_verification, "closed_record_sha256": ""}
            ).encode("utf-8")
        ).hexdigest()
    )
    validate_gpu_runtime_verification_v2_record(
        runtime_verification,
        plan_record=plan,
        expected_job_id=job_id,
        expected_artifact_pins=pins,
    )


@pytest.mark.parametrize("tamper", ["plan", "job"])
def test_runtime_installer_rejects_nonexact_sealed_plan_or_job_before_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tamper: str,
) -> None:
    plan = _plan()
    job_id = plan["cloud_qualification"]["jobs"][0]["job_id"]
    planned_job = deepcopy(qualification_v1._plan_job(plan, job_id))
    if tamper == "plan":
        plan["cloud_qualification"]["jobs"][0]["gpu"] = "NVIDIA TAMPER"
        _seal_v2(plan)
    else:
        planned_job["gpu"] = "NVIDIA TAMPER"

    def unexpected_install(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("invalid plan/job reached installation")

    monkeypatch.setattr(runtime_v2.subprocess, "run", unexpected_install)
    monkeypatch.setattr(runtime_v2, "create_venv", unexpected_install)
    with pytest.raises(ValueError):
        runtime_v2.run_gpu_qualification_sentinel_v2(
            plan_record=plan,
            planned_job=planned_job,
            artifact_paths={
                key: tmp_path / key for key in GPU_QUALIFICATION_V2_ARTIFACT_KEYS
            },
            work_dir=tmp_path / "work",
        )


class _Distribution:
    def __init__(
        self,
        name: str | None,
        version: str,
        *,
        direct_url: dict[str, Any] | None = None,
        root: Path | None = None,
        omit_name: bool = False,
    ) -> None:
        self.metadata = {} if omit_name else {"Name": name}
        self.version = version
        self._direct_url = direct_url
        self._root = root

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        if self._direct_url is None:
            return None
        return json.dumps(self._direct_url)

    def locate_file(self, name: str) -> Path:
        assert name == ""
        assert self._root is not None
        return self._root


def _patch_verifier_through_distribution_scan(
    monkeypatch: pytest.MonkeyPatch,
    distributions: list[_Distribution],
) -> None:
    monkeypatch.setattr(runtime_v2, "_require_runtime_platform", lambda: None)
    monkeypatch.setattr(runtime_v2, "_pip_subprocess_environment", lambda: {})
    monkeypatch.setattr(
        runtime_v2.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0),
    )
    monkeypatch.setattr(
        runtime_v2, "_file_sha256", lambda _path: VLLM_RUNTIME_BASE_LOCK_SHA256
    )
    monkeypatch.setattr(
        runtime_v2, "_base_lock_projection", lambda _path: ({"base-dist": "1"}, 1)
    )
    monkeypatch.setattr(runtime_v2, "VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT", 1)
    monkeypatch.setattr(runtime_v2, "VLLM_RUNTIME_BASE_LOCK_HASH_COUNT", 1)
    monkeypatch.setattr(
        runtime_v2,
        "_read_exact_runtime_closure",
        lambda _path: {
            "closed_record_sha256": RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256
        },
    )
    monkeypatch.setattr(
        runtime_v2.importlib.metadata, "distributions", lambda: distributions
    )


@pytest.mark.parametrize(
    ("distributions", "error"),
    [
        (
            [_Distribution("base-dist", "1"), _Distribution("base_dist", "1")],
            "duplicate installed distribution",
        ),
        ([_Distribution(None, "1", omit_name=True)], "no package name"),
        ([_Distribution("base-dist", "1")], "name closure differs"),
    ],
)
def test_standalone_verifier_rejects_duplicate_anonymous_or_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
    distributions: list[_Distribution],
    error: str,
) -> None:
    _patch_verifier_through_distribution_scan(monkeypatch, distributions)
    with pytest.raises(RuntimeError, match=error):
        runtime_v2.verify_gpu_qualification_v2_runtime_installation(
            runtime_lock="base.lock",
            vllm_uri="file:///vllm.whl",
            flashinfer_uri="file:///flashinfer.whl",
            runtime_closure_manifest="closure.json",
            package_uri="file:///cachet.whl",
            package_sha256=_PACKAGE_SHA256,
        )


def test_pep610_validation_checks_uri_archive_hash_and_origin_rehash(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "wheel+patched.whl"
    other = tmp_path / "other.whl"
    wheel.write_bytes(b"reviewed wheel")
    other.write_bytes(b"reviewed wheel")
    digest = sha256(b"reviewed wheel").hexdigest()
    value = {
        "archive_info": {"hashes": {"sha256": digest}},
        "url": wheel.resolve().as_uri(),
    }
    distribution = _Distribution("example", "1", direct_url=value)
    assert (
        runtime_v2._validate_direct_url(
            distribution,  # type: ignore[arg-type]
            expected_uri=wheel.resolve().as_uri(),
            expected_sha256=digest,
        )
        == wheel.resolve().as_uri()
    )

    distribution._direct_url = {**value, "url": other.resolve().as_uri()}
    with pytest.raises(RuntimeError, match="direct URL differs"):
        runtime_v2._validate_direct_url(
            distribution,  # type: ignore[arg-type]
            expected_uri=wheel.resolve().as_uri(),
            expected_sha256=digest,
        )

    distribution._direct_url = {
        **value,
        "archive_info": {"hashes": {"sha256": "0" * 64}},
    }
    with pytest.raises(RuntimeError, match="archive SHA-256 differs"):
        runtime_v2._validate_direct_url(
            distribution,  # type: ignore[arg-type]
            expected_uri=wheel.resolve().as_uri(),
            expected_sha256=digest,
        )

    distribution._direct_url = value
    wheel.write_bytes(b"tampered wheel")
    with pytest.raises(RuntimeError, match="source bytes differ"):
        runtime_v2._validate_direct_url(
            distribution,  # type: ignore[arg-type]
            expected_uri=wheel.resolve().as_uri(),
            expected_sha256=digest,
        )


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        ("member", "installed member differs"),
        ("origin", "imported FlashInfer module origin differs"),
        ("annotation", "postponed return annotation differs"),
    ],
)
def test_standalone_verifier_rejects_member_hash_or_import_annotation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
    error: str,
) -> None:
    flashinfer_root = tmp_path / "site-packages"
    flashinfer_member = flashinfer_root / FLASHINFER_TARGET_MEMBER
    flashinfer_member.parent.mkdir(parents=True)
    flashinfer_member.write_bytes(b"reviewed FlashInfer member")
    distributions = [
        _Distribution("base-dist", "1"),
        _Distribution("cachet-kv", GPU_QUALIFICATION_V2_CACHET_PACKAGE_VERSION),
        _Distribution(
            "flashinfer-python",
            FLASHINFER_PACKAGE_VERSION,
            root=flashinfer_root,
        ),
        _Distribution("vllm", GPU_QUALIFICATION_VLLM_VERSION),
    ]
    _patch_verifier_through_distribution_scan(monkeypatch, distributions)
    monkeypatch.setattr(
        runtime_v2, "GPU_QUALIFICATION_V2_INSTALLED_DISTRIBUTION_COUNT", 4
    )
    monkeypatch.setattr(runtime_v2, "_uri_file_sha256", lambda _uri: _PACKAGE_SHA256)
    monkeypatch.setattr(
        runtime_v2,
        "_validate_direct_url",
        lambda _distribution, *, expected_uri, expected_sha256: expected_uri,
    )
    if failure == "member":

        def member_failure(*_args: Any, **_kwargs: Any) -> dict[str, str]:
            raise RuntimeError("v2 installed member differs: reviewed.py")

        monkeypatch.setattr(runtime_v2, "_installed_member_hashes", member_failure)
    else:
        monkeypatch.setattr(
            runtime_v2,
            "_installed_member_hashes",
            lambda _distribution, expected: dict(expected),
        )
        monkeypatch.setattr(
            runtime_v2,
            "_file_sha256",
            lambda path: (
                FLASHINFER_TARGET_PATCHED_SHA256
                if Path(path).name == Path(FLASHINFER_TARGET_MEMBER).name
                else VLLM_RUNTIME_BASE_LOCK_SHA256
            ),
        )
        function = SimpleNamespace(
            __annotations__={
                "return": (
                    "wrong"
                    if failure == "annotation"
                    else GPU_QUALIFICATION_V2_FLASHINFER_RETURN_ANNOTATION
                )
            }
        )
        module_path = flashinfer_member
        if failure == "origin":
            module_path = tmp_path / "shadow" / "fd_exchange.py"
            module_path.parent.mkdir()
            module_path.write_bytes(b"shadow module")
        module = SimpleNamespace(_fd_ancillary=function, __file__=str(module_path))
        monkeypatch.setattr(runtime_v2.importlib, "import_module", lambda _name: module)

    with pytest.raises(RuntimeError, match=error):
        runtime_v2.verify_gpu_qualification_v2_runtime_installation(
            runtime_lock="base.lock",
            vllm_uri="file:///vllm.whl",
            flashinfer_uri="file:///flashinfer.whl",
            runtime_closure_manifest="closure.json",
            package_uri="file:///cachet.whl",
            package_sha256=_PACKAGE_SHA256,
        )


def test_runtime_platform_requires_exact_python_3_11_11(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_v2.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(runtime_v2.platform, "python_version", lambda: "3.11.11")
    monkeypatch.setattr(runtime_v2.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime_v2.platform, "libc_ver", lambda: ("glibc", "2.35"))
    monkeypatch.setattr(runtime_v2.sys, "version_info", (3, 11, 11, "final", 0))
    monkeypatch.setattr(runtime_v2.sys, "platform", "linux")

    runtime_v2._require_runtime_platform()
    monkeypatch.setattr(runtime_v2.platform, "python_version", lambda: "3.11.12")
    with pytest.raises(RuntimeError, match="requires Linux CPython3.11"):
        runtime_v2._require_runtime_platform()


def test_real_runtime_closure_has_exact_identity_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    raw = _CLOSURE_PATH.read_bytes()
    assert len(raw) == RUNTIME_ARTIFACT_CLOSURE_FILE_SIZE == 6634
    assert sha256(raw).hexdigest() == RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256
    closure = runtime_v2._read_exact_runtime_closure(_CLOSURE_PATH)
    assert closure["closed_record_sha256"] == (
        RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256
    )

    tampered = tmp_path / _CLOSURE_PATH.name
    tampered.write_bytes(raw[:-2] + b" \n")
    assert tampered.stat().st_size == 6634
    with pytest.raises(ValueError, match="file identity differs"):
        runtime_v2._read_exact_runtime_closure(tampered)


def test_real_base_lock_has_exact_projection_and_verifier_rejects_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = _LOCK_PATH.read_bytes()
    assert len(raw) == VLLM_RUNTIME_BASE_LOCK_SIZE == 376326
    assert sha256(raw).hexdigest() == VLLM_RUNTIME_BASE_LOCK_SHA256
    versions, hash_count = runtime_v2._base_lock_projection(_LOCK_PATH)
    assert len(versions) == VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT == 195
    assert hash_count == VLLM_RUNTIME_BASE_LOCK_HASH_COUNT == 4137
    assert not {"cachet-kv", "flashinfer-python", "vllm"} & versions.keys()

    tampered = tmp_path / VLLM_RUNTIME_BASE_LOCK_FILENAME
    tampered.write_bytes(raw[:-1] + (b" " if raw[-1:] != b" " else b"\n"))
    assert tampered.stat().st_size == 376326
    monkeypatch.setattr(runtime_v2, "_require_runtime_platform", lambda: None)
    monkeypatch.setattr(runtime_v2, "_pip_subprocess_environment", lambda: {})
    monkeypatch.setattr(
        runtime_v2.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0),
    )
    with pytest.raises(RuntimeError, match="base lock SHA-256 differs"):
        runtime_v2.verify_gpu_qualification_v2_runtime_installation(
            runtime_lock=tampered,
            vllm_uri="file:///vllm.whl",
            flashinfer_uri="file:///flashinfer.whl",
            runtime_closure_manifest=_CLOSURE_PATH,
            package_uri="file:///cachet.whl",
            package_sha256=_PACKAGE_SHA256,
        )
