"""Serving-engine environment profiles for isolated backend installs."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.metadata as package_metadata
import json
import os
import platform
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
import urllib.parse

from document_kv_cache.engine_adapters import ServingBackend

SERVING_ENVIRONMENT_PROFILES_RECORD_TYPE = "document_kv.serving_environment_profiles.v1"

VLLM_VERSION = "0.27.1"
# vLLM 0.27.1 defaults to CUDA 13.0.  Cachet deliberately selects the separate
# official CUDA 12.9 x86_64 release asset for the benchmark fleet.  The asset
# hash, indexes, and tagged-source digests make that binary choice auditable.
VLLM_CUDA_VERSION = "12.9"
VLLM_CUDA_VARIANT = "cu129"
VLLM_PACKAGE_VERSION = f"{VLLM_VERSION}+{VLLM_CUDA_VARIANT}"
VLLM_WHEEL_FILENAME = (
    "vllm-0.27.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
)
VLLM_WHEEL_URL = (
    "https://github.com/vllm-project/vllm/releases/download/v0.27.1/"
    "vllm-0.27.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
)
VLLM_WHEEL_SHA256 = (
    "bf0d52faa2a51e7a01c6856a7a8a2d1307fd0ff711415d34168a67ffac0fa47b"
)
VLLM_WHEEL_INSTALL_SPEC = (
    f"vllm @ {VLLM_WHEEL_URL}#sha256={VLLM_WHEEL_SHA256}"
)
VLLM_PATCHED_WHEEL_URI_ENV = "DOCUMENT_KV_VLLM_PATCHED_WHEEL_URI"
VLLM_PATCHED_WHEEL_SHA256_ENV = "DOCUMENT_KV_VLLM_PATCHED_WHEEL_SHA256"
# ``PYTHONWARNINGS`` uses commas to delimit filters, so reviewed messages that
# contain commas are intentionally matched through an invariant pre-comma
# prefix.  The pinned cuda-bindings wheel and CPython/vLLM runtime fix the
# corresponding suffixes.  The two bitsandbytes filters supply their full
# reviewed 0.49.2/PyTorch 2.13 message because it has no comma.  FlashInfer's
# reviewed multiline message cannot be represented exactly: Python strips its
# leading field whitespace and its later comma is a filter delimiter.  Its
# empty message field is therefore confined to the exact category, attributed
# module, and line of the sole expression in the pinned vLLM wheel.  The
# torch.jit allowance supplies its full pinned PyTorch 2.13 message.  Every
# other warning is promoted to an exception.
GPU_RUNTIME_PYTHONWARNINGS = ",".join(
    (
        "error",
        (
            "ignore:The cuda.cuda module is deprecated and will be removed in a "
            "future release:FutureWarning:importlib._bootstrap_external:1241"
        ),
        (
            "ignore:The cuda.cudart module is deprecated and will be removed in a "
            "future release:FutureWarning:importlib._bootstrap_external:1241"
        ),
        (
            "ignore:The cuda.nvrtc module is deprecated and will be removed in a "
            "future release:FutureWarning:importlib._bootstrap_external:1241"
        ),
        (
            "ignore:_check_is_size will be removed in a future PyTorch release "
            "along with guard_size_oblivious.     Use _check(i >= 0) instead."
            ":FutureWarning:bitsandbytes.backends.cuda.ops:213"
        ),
        (
            "ignore:_check_is_size will be removed in a future PyTorch release "
            "along with guard_size_oblivious.     Use _check(i >= 0) instead."
            ":FutureWarning:bitsandbytes.backends.cuda.ops:468"
        ),
        (
            "ignore:'vllm.model_executor.models.registry' found in sys.modules "
            "after import of package 'vllm.model_executor.models'"
            ":RuntimeWarning:runpy:128"
        ),
        (
            "ignore::DeprecationWarning:"
            "vllm.v1.attention.backends.flashinfer:1234"
        ),
        (
            "ignore:`torch.jit.script_method` is deprecated. Please switch to "
            "`torch.compile` or `torch.export`."
            ":DeprecationWarning:torch.jit._script:365"
        ),
    )
)
GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL = "ERROR"


def gpu_runtime_warning_environment_overrides() -> dict[str, str]:
    """Return the exact warning policy for verifier and worker descendants."""

    return {
        "FLASHINFER_LOGGING_LEVEL": GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL,
        "PYTHONWARNINGS": GPU_RUNTIME_PYTHONWARNINGS,
    }


VLLM_CUDA_REQUIREMENTS_SHA256 = (
    "30091f418325ea9f97bc546cb03eb1a35e1cc20b2500b522b3dabd3b1aaee241"
)
VLLM_DOCKERFILE_SHA256 = (
    "9876efaec74111cad4ce074225740fafa9461166973d1837dc0d4cd23c2f2509"
)
VLLM_PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu129"
VLLM_FLASHINFER_INDEX_URL = "https://flashinfer.ai/whl/"
VLLM_FLASHINFER_JIT_INDEX_URL = "https://flashinfer.ai/whl/cu129"
VLLM_PACKAGE_INDEX_URLS = (
    VLLM_PYTORCH_INDEX_URL,
    VLLM_FLASHINFER_INDEX_URL,
    VLLM_FLASHINFER_JIT_INDEX_URL,
)
VLLM_RUNTIME_LOCK_FILENAME = (
    "vllm-0.27.1-cu129-py311-manylinux_2_35.lock"
)
VLLM_RUNTIME_LOCK_PRE_AUGMENT_SHA256 = (
    "5788ee492a9a9ff48c8e1eae68cd0576fcec625263858129cc9dd918bcb856a6"
)
VLLM_RUNTIME_LOCK_SHA256 = (
    "71c2c3e344ebdf1d8996adf2127a519328b6bad78a4eb7134c73e2a3f6115c44"
)
_VLLM_RUNTIME_LOCK_COMPILED_INDEX_HEADER = (
    "--index-url https://pypi.org/simple\n"
    f"--extra-index-url {VLLM_FLASHINFER_JIT_INDEX_URL}\n"
    f"--extra-index-url {VLLM_FLASHINFER_INDEX_URL}\n"
)
VLLM_RUNTIME_LOCK_INDEX_HEADER = (
    "--index-url https://pypi.org/simple\n"
    f"--extra-index-url {VLLM_PYTORCH_INDEX_URL}\n"
    f"--extra-index-url {VLLM_FLASHINFER_JIT_INDEX_URL}\n"
    f"--extra-index-url {VLLM_FLASHINFER_INDEX_URL}\n"
)
# The generated lock has 196 hashed distributions. vLLM is deliberately
# excluded and installed separately from the reviewed patched wheel, yielding
# a 197-distribution runtime closure.
VLLM_RUNTIME_LOCK_DISTRIBUTION_COUNT = 196

# The isolated-runtime bootstrap is exact as well; an unbounded toolchain
# upgrade would make otherwise identical benchmark jobs resolve differently.
# This compatible Python 3.11 bootstrap is also included in the generated
# DBR 15.4 lock. Python 3.12+ is outside that platform-specific lock.
PIP_BOOTSTRAP_CONSTRAINTS = (
    "pip==26.2.1",
    "setuptools==80.9.0",
    "wheel==0.48.0",
)
VIRTUALENV_BOOTSTRAP_VERSION = "20.39.1"
VIRTUALENV_BOOTSTRAP_FILENAME = "virtualenv.pyz"
VIRTUALENV_BOOTSTRAP_URL = (
    "https://github.com/pypa/virtualenv/releases/download/20.39.1/virtualenv.pyz"
)
VIRTUALENV_BOOTSTRAP_SHA256 = (
    "8a22f3495357316e30db13f4bdee8487fe27137fa94d383ba1e6fe3e242b6165"
)

# Pinned compiled direct dependencies.  The cu12 extras mirror the
# transformation in vLLM's tagged Dockerfile when CUDA_VERSION has major
# version 12.  These direct pins are not a substitute for the benchmark
# runtime's generated, hash-locked transitive closure; ``pip freeze --all``
# is retained only as post-install evidence.
TORCH_CONSTRAINT = "torch==2.13.0+cu129"
TORCHAUDIO_CONSTRAINT = "torchaudio==2.11.0+cu129"
TORCHVISION_CONSTRAINT = "torchvision==0.28.0+cu129"
TRITON_CONSTRAINT = "triton==3.7.1"
NUMBA_CONSTRAINT = "numba==0.65.0"
TORCHCODEC_CONSTRAINT = "torchcodec==0.16.0+cu129"
PYNVVIDEOCODEC_CONSTRAINT = "PyNvVideoCodec==2.0.4"
FLASHINFER_PYTHON_CONSTRAINT = "flashinfer-python==0.6.16.post3"
FLASHINFER_CUBIN_CONSTRAINT = "flashinfer-cubin==0.6.16.post3"
FLASHINFER_JIT_CACHE_CONSTRAINT = "flashinfer-jit-cache==0.6.16.post3+cu129"
APACHE_TVM_FFI_CONSTRAINT = "apache-tvm-ffi==0.1.11"
TILELANG_CONSTRAINT = "tilelang==0.1.12"
NVIDIA_CUDNN_FRONTEND_CONSTRAINT = "nvidia-cudnn-frontend==1.27.0"
NVTX_CONSTRAINT = "nvtx==0.2.15"
FASTSAFETENSORS_CONSTRAINT = "fastsafetensors==0.3.3"
NVIDIA_CUTLASS_DSL_CONSTRAINT = "nvidia-cutlass-dsl==4.6.0"
QUACK_KERNELS_CONSTRAINT = "quack-kernels==0.6.1"
TOKENSPEED_MLA_CONSTRAINT = "tokenspeed-mla==0.1.8"
HUMMING_KERNELS_CONSTRAINT = "humming-kernels[cu12]==0.1.10"
# Backwards-compatible name for callers that only need the Python frontend.
FLASHINFER_CONSTRAINT = FLASHINFER_PYTHON_CONSTRAINT
VLLM_PINNED_CUDA_DIRECT_CONSTRAINTS = (
    TORCH_CONSTRAINT,
    TORCHAUDIO_CONSTRAINT,
    TORCHVISION_CONSTRAINT,
    TRITON_CONSTRAINT,
    NUMBA_CONSTRAINT,
    TORCHCODEC_CONSTRAINT,
    PYNVVIDEOCODEC_CONSTRAINT,
    FLASHINFER_PYTHON_CONSTRAINT,
    FLASHINFER_CUBIN_CONSTRAINT,
    FLASHINFER_JIT_CACHE_CONSTRAINT,
    APACHE_TVM_FFI_CONSTRAINT,
    TILELANG_CONSTRAINT,
    NVIDIA_CUDNN_FRONTEND_CONSTRAINT,
    NVTX_CONSTRAINT,
    FASTSAFETENSORS_CONSTRAINT,
    NVIDIA_CUTLASS_DSL_CONSTRAINT,
    QUACK_KERNELS_CONSTRAINT,
    TOKENSPEED_MLA_CONSTRAINT,
    HUMMING_KERNELS_CONSTRAINT,
)
TRANSFORMERS_CONSTRAINT = "transformers==5.12.1"
HUGGINGFACE_HUB_CONSTRAINT = "huggingface-hub==1.20.1"
TOKENIZERS_CONSTRAINT = "tokenizers==0.22.2"
NUMPY_CONSTRAINT = "numpy==2.3.5"
FASTAPI_CONSTRAINT = "fastapi[standard]==0.136.0"
PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT = "prometheus-fastapi-instrumentator==8.0.0"
BITSANDBYTES_CONSTRAINT = "bitsandbytes==0.49.2"
ACCELERATE_CONSTRAINT = "accelerate==1.14.0"
OPENCV_PYTHON_HEADLESS_CONSTRAINT = "opencv-python-headless==4.13.0.92"
VLLM_DEPENDENCY_CONSTRAINTS = (
    f"vllm=={VLLM_PACKAGE_VERSION}",
    *VLLM_PINNED_CUDA_DIRECT_CONSTRAINTS,
    TRANSFORMERS_CONSTRAINT,
    HUGGINGFACE_HUB_CONSTRAINT,
    TOKENIZERS_CONSTRAINT,
    NUMPY_CONSTRAINT,
    FASTAPI_CONSTRAINT,
    PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT,
    BITSANDBYTES_CONSTRAINT,
    ACCELERATE_CONSTRAINT,
    OPENCV_PYTHON_HEADLESS_CONSTRAINT,
)
VLLM_INSTALL_REQUIREMENTS = (
    VLLM_WHEEL_INSTALL_SPEC,
    *VLLM_DEPENDENCY_CONSTRAINTS[1:],
)


def patched_vllm_wheel_install_spec() -> str:
    """Return the reviewed content-addressed runtime wheel PEP 508 spec."""

    raw_uri = os.environ.get(VLLM_PATCHED_WHEEL_URI_ENV)
    raw_digest = os.environ.get(VLLM_PATCHED_WHEEL_SHA256_ENV)
    if raw_uri is None or not raw_uri.strip():
        raise RuntimeError(
            f"{VLLM_PATCHED_WHEEL_URI_ENV} must point at the reviewed, "
            "content-addressed E5M2-patched vLLM wheel"
        )
    if raw_digest is None:
        raise RuntimeError(
            f"{VLLM_PATCHED_WHEEL_SHA256_ENV} must pin the patched wheel SHA-256"
        )
    digest = raw_digest.strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{VLLM_PATCHED_WHEEL_SHA256_ENV} must be a lowercase SHA-256")
    if digest == VLLM_WHEEL_SHA256:
        raise ValueError("patched vLLM wheel SHA-256 must differ from the pristine cu129 asset")

    uri = raw_uri.strip()
    if uri.startswith("dbfs:/"):
        uri = Path("/dbfs", uri.removeprefix("dbfs:/").lstrip("/")).resolve().as_uri()
    elif uri.startswith("/"):
        uri = Path(uri).resolve().as_uri()
    parsed = urllib.parse.urlsplit(uri)
    if parsed.scheme not in {"file", "https"} or not parsed.path:
        raise ValueError(
            f"{VLLM_PATCHED_WHEEL_URI_ENV} must be an absolute path, dbfs:/ URI, "
            "file URI, or HTTPS URI"
        )
    if parsed.query or parsed.fragment:
        raise ValueError(f"{VLLM_PATCHED_WHEEL_URI_ENV} must not include a query or fragment")
    wheel_name = urllib.parse.unquote(PurePosixPath(parsed.path).name)
    expected_name = (
        f"vllm-{VLLM_PACKAGE_VERSION}-1cachete5m2{digest[:16]}-"
        "cp38-abi3-manylinux_2_28_x86_64.whl"
    )
    if wheel_name != expected_name:
        raise ValueError(
            "patched vLLM wheel filename must bind its SHA-256 prefix and exact "
            "0.27.1+cu129 ABI/platform tags"
        )
    return f"vllm @ {uri}#sha256={digest}"


def vllm_runtime_install_requirements() -> tuple[str, ...]:
    """Return exact runtime requirements headed by the prepatched wheel."""

    return (
        patched_vllm_wheel_install_spec(),
        *VLLM_DEPENDENCY_CONSTRAINTS[1:],
    )


def augment_vllm_runtime_lock_indexes(compiled_lock_bytes: bytes) -> bytes:
    """Add the one PyTorch runtime index omitted by uv's torch backend output."""

    if not isinstance(compiled_lock_bytes, bytes):
        raise TypeError("compiled_lock_bytes must be bytes")
    compiled_digest = sha256(compiled_lock_bytes).hexdigest()
    if compiled_digest != VLLM_RUNTIME_LOCK_PRE_AUGMENT_SHA256:
        raise RuntimeError(
            "Compiled vLLM runtime lock failed its pre-augmentation hash: "
            f"expected {VLLM_RUNTIME_LOCK_PRE_AUGMENT_SHA256}, "
            f"found {compiled_digest}"
        )
    compiled_header = _VLLM_RUNTIME_LOCK_COMPILED_INDEX_HEADER.encode("utf-8")
    if compiled_lock_bytes.count(compiled_header) != 1:
        raise RuntimeError("Compiled vLLM runtime lock index header differs")
    augmented = compiled_lock_bytes.replace(
        compiled_header,
        VLLM_RUNTIME_LOCK_INDEX_HEADER.encode("utf-8"),
        1,
    )
    augmented_digest = sha256(augmented).hexdigest()
    if augmented_digest != VLLM_RUNTIME_LOCK_SHA256:
        raise RuntimeError(
            "Augmented vLLM runtime lock failed its content hash: "
            f"expected {VLLM_RUNTIME_LOCK_SHA256}, found {augmented_digest}"
        )
    return augmented


def vllm_runtime_lock_path() -> Path:
    """Return and replay-verify the packaged Python 3.11 Linux hash lock."""

    lock_path = Path(__file__).with_name("runtime_locks") / VLLM_RUNTIME_LOCK_FILENAME
    try:
        lock_bytes = lock_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Packaged vLLM runtime lock is unavailable: {lock_path}") from exc
    observed_digest = sha256(lock_bytes).hexdigest()
    if observed_digest != VLLM_RUNTIME_LOCK_SHA256:
        raise RuntimeError(
            "Packaged vLLM runtime lock failed its content hash: "
            f"expected {VLLM_RUNTIME_LOCK_SHA256}, found {observed_digest}"
        )
    runtime_header = VLLM_RUNTIME_LOCK_INDEX_HEADER.encode("utf-8")
    if lock_bytes.count(runtime_header) != 1:
        raise RuntimeError("Packaged vLLM runtime lock index header differs")
    compiled_lock_bytes = lock_bytes.replace(
        runtime_header,
        _VLLM_RUNTIME_LOCK_COMPILED_INDEX_HEADER.encode("utf-8"),
        1,
    )
    if augment_vllm_runtime_lock_indexes(compiled_lock_bytes) != lock_bytes:
        raise RuntimeError("Packaged vLLM runtime lock augmentation is not replayable")
    return lock_path


def validate_vllm_runtime_lock_platform(
    *,
    python_version: tuple[int, int] | None = None,
    machine: str | None = None,
    libc: tuple[str, str] | None = None,
    operating_system: str | None = None,
) -> dict[str, str]:
    """Fail closed outside the DBR 15.4 lock's exact runtime platform."""

    observed_python = python_version or sys.version_info[:2]
    observed_machine = (machine or platform.machine()).lower()
    observed_libc = libc or platform.libc_ver()
    observed_os = operating_system or sys.platform
    if observed_python != (3, 11):
        raise RuntimeError("vLLM runtime lock requires CPython 3.11")
    if observed_os != "linux":
        raise RuntimeError("vLLM runtime lock requires Linux")
    if observed_machine not in {"x86_64", "amd64"}:
        raise RuntimeError("vLLM runtime lock requires x86_64")
    if observed_libc != ("glibc", "2.35"):
        raise RuntimeError("vLLM runtime lock requires glibc 2.35")
    return {
        "python": "3.11",
        "operating_system": "linux",
        "machine": "x86_64",
        "libc": "glibc-2.35",
    }


def verify_installed_vllm_runtime_lock(
    expected_wheel_install_spec: str,
) -> dict[str, Any]:
    """Verify installed versions and PEP 610 provenance against the lock."""

    expected_versions = _vllm_runtime_lock_versions(vllm_runtime_lock_path())
    installed_versions: dict[str, list[str]] = {}
    for distribution in package_metadata.distributions():
        package_name = distribution.metadata.get("Name")
        if not isinstance(package_name, str) or not package_name:
            continue
        installed_versions.setdefault(_canonical_package_name(package_name), []).append(
            distribution.version
        )
    issues: list[str] = []
    for package_name, expected_version in expected_versions.items():
        observed_versions = installed_versions.get(package_name, [])
        if observed_versions != [expected_version]:
            issues.append(
                f"{package_name} expected {expected_version}, found {observed_versions!r}"
            )
    allowed_distributions = {
        *expected_versions,
        "vllm",
        "cachet-kv",
    }
    unexpected_distributions = sorted(
        set(installed_versions) - allowed_distributions
    )
    if unexpected_distributions:
        issues.append(
            "unexpected distributions: " + ", ".join(unexpected_distributions)
        )

    vllm_distribution = package_metadata.distribution("vllm")
    if vllm_distribution.version != VLLM_PACKAGE_VERSION:
        issues.append(
            f"vllm expected {VLLM_PACKAGE_VERSION}, found {vllm_distribution.version}"
        )
    expected_uri, expected_digest = _wheel_identity_from_install_spec(
        expected_wheel_install_spec
    )
    direct_url_text = vllm_distribution.read_text("direct_url.json")
    if direct_url_text is None:
        issues.append("vllm installation is missing PEP 610 direct_url.json")
        direct_url: Mapping[str, Any] = {}
    else:
        try:
            decoded_direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError as exc:
            issues.append(f"vllm direct_url.json is invalid: {exc}")
            direct_url = {}
        else:
            direct_url = (
                decoded_direct_url
                if isinstance(decoded_direct_url, Mapping)
                else {}
            )
            if not direct_url:
                issues.append("vllm direct_url.json must contain an object")
    observed_uri = direct_url.get("url")
    if not isinstance(observed_uri, str) or (
        _normalized_artifact_uri(observed_uri)
        != _normalized_artifact_uri(expected_uri)
    ):
        issues.append(
            f"vllm direct URL expected {expected_uri!r}, found {observed_uri!r}"
        )
    observed_digest = _direct_url_sha256(direct_url)
    if observed_digest != expected_digest:
        issues.append(
            f"vllm direct URL SHA-256 expected {expected_digest}, found {observed_digest!r}"
        )
    if issues:
        raise RuntimeError("vLLM runtime lock verification failed: " + "; ".join(issues))
    return {
        "runtime_lock_sha256": VLLM_RUNTIME_LOCK_SHA256,
        "locked_distribution_count": len(expected_versions),
        "vllm_package_version": vllm_distribution.version,
        "vllm_direct_url": observed_uri,
        "vllm_wheel_sha256": observed_digest,
        "unexpected_distributions": unexpected_distributions,
        "ok": True,
    }


def _vllm_runtime_lock_versions(lock_path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        if not line or not line[0].isalnum() or "==" not in line:
            continue
        package_name, version_tail = line.split("==", maxsplit=1)
        version = version_tail.removesuffix(" \\").strip()
        canonical_name = _canonical_package_name(package_name)
        if canonical_name in versions:
            raise RuntimeError(f"vLLM runtime lock repeats {canonical_name}")
        versions[canonical_name] = version
    if len(versions) != VLLM_RUNTIME_LOCK_DISTRIBUTION_COUNT:
        raise RuntimeError(
            "vLLM runtime lock distribution count changed: "
            f"expected {VLLM_RUNTIME_LOCK_DISTRIBUTION_COUNT}, found {len(versions)}"
        )
    return versions


def _canonical_package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _wheel_identity_from_install_spec(install_spec: str) -> tuple[str, str]:
    prefix = "vllm @ "
    if not install_spec.startswith(prefix):
        raise ValueError("patched vLLM install spec must be a vllm PEP 508 direct reference")
    uri_with_fragment = install_spec.removeprefix(prefix)
    parsed = urllib.parse.urlsplit(uri_with_fragment)
    fragment = urllib.parse.parse_qs(parsed.fragment, strict_parsing=True)
    digests = fragment.get("sha256", [])
    if len(digests) != 1:
        raise ValueError("patched vLLM install spec must include one SHA-256 fragment")
    digest = digests[0].lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("patched vLLM install spec SHA-256 must be lowercase hex")
    return urllib.parse.urlunsplit(parsed._replace(fragment="")), digest


def _normalized_artifact_uri(uri: str) -> str:
    parsed = urllib.parse.urlsplit(uri)
    return urllib.parse.urlunsplit(
        parsed._replace(path=urllib.parse.unquote(parsed.path), fragment="")
    )


def _direct_url_sha256(direct_url: Mapping[str, Any]) -> str | None:
    archive_info = direct_url.get("archive_info")
    if not isinstance(archive_info, Mapping):
        return None
    hashes = archive_info.get("hashes")
    if isinstance(hashes, Mapping):
        digest = hashes.get("sha256")
        if isinstance(digest, str):
            return digest.lower()
    legacy_hash = archive_info.get("hash")
    if isinstance(legacy_hash, str) and legacy_hash.startswith("sha256="):
        return legacy_hash.removeprefix("sha256=").lower()
    return None

SGLANG_VERSION = "0.5.10.post1"
SGLANG_DEPENDENCY_CONSTRAINTS = (f"sglang=={SGLANG_VERSION}",)


@dataclass(frozen=True, slots=True)
class ServingEnvironmentProfile:
    """Pinned pip-install profile for one serving backend environment."""

    backend: ServingBackend
    engine_package: str
    engine_version: str
    dependency_constraints: tuple[str, ...]
    isolated_environment_required: bool
    notes: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _serving_backend(self.backend))
        if not self.engine_package:
            raise ValueError("engine_package must be non-empty")
        if not self.engine_version:
            raise ValueError("engine_version must be non-empty")
        constraints = tuple(self.dependency_constraints)
        if not constraints:
            raise ValueError("dependency_constraints must be non-empty")
        for constraint in constraints:
            _validate_exact_constraint(constraint)
        engine_constraint = f"{self.engine_package}=={self.engine_version}"
        if engine_constraint not in constraints:
            raise ValueError("dependency_constraints must include the exact engine package pin")
        object.__setattr__(self, "dependency_constraints", constraints)
        if type(self.isolated_environment_required) is not bool:
            raise ValueError("isolated_environment_required must be a boolean")
        if not self.notes:
            raise ValueError("notes must be non-empty")


def _serving_backend(value: ServingBackend | str) -> ServingBackend:
    try:
        return value if isinstance(value, ServingBackend) else ServingBackend(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported serving backend {value!r}") from exc


def _validate_exact_constraint(constraint: str) -> None:
    if not constraint or "==" not in constraint:
        raise ValueError("dependency_constraints must be exact package pins")
    package, version = constraint.split("==", maxsplit=1)
    if not package or not version:
        raise ValueError("dependency_constraints must be exact package pins")
    if any(marker in version for marker in ("<", ">", "~", "^", "*", ",")):
        raise ValueError("dependency_constraints must not include version ranges")


VLLM_SERVING_ENVIRONMENT_PROFILE = ServingEnvironmentProfile(
    backend=ServingBackend.VLLM,
    engine_package="vllm",
    engine_version=VLLM_PACKAGE_VERSION,
    dependency_constraints=VLLM_DEPENDENCY_CONSTRAINTS,
    isolated_environment_required=True,
    notes=(
        "Install vLLM in a dedicated serving environment because current vLLM "
        "and SGLang releases pin incompatible runtime stacks."
    ),
)

SGLANG_SERVING_ENVIRONMENT_PROFILE = ServingEnvironmentProfile(
    backend=ServingBackend.SGLANG,
    engine_package="sglang",
    engine_version=SGLANG_VERSION,
    dependency_constraints=SGLANG_DEPENDENCY_CONSTRAINTS,
    isolated_environment_required=True,
    notes=(
        "Install SGLang in a dedicated serving environment because current "
        "SGLang and vLLM releases pin incompatible runtime stacks."
    ),
)


def serving_environment_profile(backend: ServingBackend | str) -> ServingEnvironmentProfile:
    """Return the pinned isolated-environment profile for a backend."""

    backend = _serving_backend(backend)
    if backend == ServingBackend.VLLM:
        return VLLM_SERVING_ENVIRONMENT_PROFILE
    if backend == ServingBackend.SGLANG:
        return SGLANG_SERVING_ENVIRONMENT_PROFILE
    raise ValueError(f"Unsupported serving backend {backend!r}")


def serving_environment_profiles() -> tuple[ServingEnvironmentProfile, ...]:
    """Return all built-in serving environment profiles."""

    return (
        serving_environment_profile(ServingBackend.VLLM),
        serving_environment_profile(ServingBackend.SGLANG),
    )


def serving_environment_profile_to_record(profile: ServingEnvironmentProfile) -> dict[str, Any]:
    """Serialize a serving environment profile as a stable diagnostics record."""

    return {
        "backend": profile.backend.value,
        "engine_package": profile.engine_package,
        "engine_version": profile.engine_version,
        "dependency_constraints": list(profile.dependency_constraints),
        "isolated_environment_required": profile.isolated_environment_required,
        "notes": profile.notes,
    }


def serving_environment_profiles_to_record() -> dict[str, Any]:
    """Serialize all built-in serving environment profiles."""

    return {
        "record_type": SERVING_ENVIRONMENT_PROFILES_RECORD_TYPE,
        "profiles": [
            serving_environment_profile_to_record(profile)
            for profile in serving_environment_profiles()
        ],
    }


def write_serving_environment_profiles_record_json(path: str | Path) -> None:
    """Write the built-in serving environment profiles record to a JSON file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(serving_environment_profiles_to_record(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Emit pinned serving environment profiles for release diagnostics."""

    parser = argparse.ArgumentParser(
        description="Emit pinned isolated vLLM/SGLang serving environment profiles."
    )
    parser.add_argument(
        "--output-json",
        help="Optional file path for the profiles JSON. Defaults to stdout.",
    )
    args = parser.parse_args(argv)

    if args.output_json:
        write_serving_environment_profiles_record_json(args.output_json)
    else:
        print(json.dumps(serving_environment_profiles_to_record(), indent=2, sort_keys=True))
    return 0


__all__ = [
    "ACCELERATE_CONSTRAINT",
    "APACHE_TVM_FFI_CONSTRAINT",
    "BITSANDBYTES_CONSTRAINT",
    "FASTAPI_CONSTRAINT",
    "FASTSAFETENSORS_CONSTRAINT",
    "FLASHINFER_CUBIN_CONSTRAINT",
    "FLASHINFER_CONSTRAINT",
    "FLASHINFER_JIT_CACHE_CONSTRAINT",
    "FLASHINFER_PYTHON_CONSTRAINT",
    "GPU_RUNTIME_FLASHINFER_LOGGING_LEVEL",
    "GPU_RUNTIME_PYTHONWARNINGS",
    "HUGGINGFACE_HUB_CONSTRAINT",
    "HUMMING_KERNELS_CONSTRAINT",
    "NVIDIA_CUTLASS_DSL_CONSTRAINT",
    "NVIDIA_CUDNN_FRONTEND_CONSTRAINT",
    "NVTX_CONSTRAINT",
    "NUMPY_CONSTRAINT",
    "NUMBA_CONSTRAINT",
    "OPENCV_PYTHON_HEADLESS_CONSTRAINT",
    "PIP_BOOTSTRAP_CONSTRAINTS",
    "PYNVVIDEOCODEC_CONSTRAINT",
    "PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT",
    "QUACK_KERNELS_CONSTRAINT",
    "SERVING_ENVIRONMENT_PROFILES_RECORD_TYPE",
    "SGLANG_DEPENDENCY_CONSTRAINTS",
    "SGLANG_SERVING_ENVIRONMENT_PROFILE",
    "SGLANG_VERSION",
    "ServingEnvironmentProfile",
    "TOKENIZERS_CONSTRAINT",
    "TORCHCODEC_CONSTRAINT",
    "TOKENSPEED_MLA_CONSTRAINT",
    "TILELANG_CONSTRAINT",
    "TORCH_CONSTRAINT",
    "TORCHAUDIO_CONSTRAINT",
    "TORCHVISION_CONSTRAINT",
    "TRANSFORMERS_CONSTRAINT",
    "TRITON_CONSTRAINT",
    "VLLM_DEPENDENCY_CONSTRAINTS",
    "VLLM_CUDA_REQUIREMENTS_SHA256",
    "VLLM_CUDA_VARIANT",
    "VLLM_CUDA_VERSION",
    "VLLM_DOCKERFILE_SHA256",
    "VLLM_FLASHINFER_INDEX_URL",
    "VLLM_FLASHINFER_JIT_INDEX_URL",
    "VLLM_PACKAGE_VERSION",
    "VLLM_PACKAGE_INDEX_URLS",
    "VLLM_PINNED_CUDA_DIRECT_CONSTRAINTS",
    "VLLM_PATCHED_WHEEL_SHA256_ENV",
    "VLLM_PATCHED_WHEEL_URI_ENV",
    "VLLM_PYTORCH_INDEX_URL",
    "VLLM_RUNTIME_LOCK_DISTRIBUTION_COUNT",
    "VLLM_RUNTIME_LOCK_FILENAME",
    "VLLM_RUNTIME_LOCK_INDEX_HEADER",
    "VLLM_RUNTIME_LOCK_PRE_AUGMENT_SHA256",
    "VLLM_RUNTIME_LOCK_SHA256",
    "VLLM_INSTALL_REQUIREMENTS",
    "VLLM_SERVING_ENVIRONMENT_PROFILE",
    "VLLM_VERSION",
    "VLLM_WHEEL_FILENAME",
    "VLLM_WHEEL_INSTALL_SPEC",
    "VLLM_WHEEL_SHA256",
    "VLLM_WHEEL_URL",
    "VIRTUALENV_BOOTSTRAP_FILENAME",
    "VIRTUALENV_BOOTSTRAP_SHA256",
    "VIRTUALENV_BOOTSTRAP_URL",
    "VIRTUALENV_BOOTSTRAP_VERSION",
    "gpu_runtime_warning_environment_overrides",
    "augment_vllm_runtime_lock_indexes",
    "patched_vllm_wheel_install_spec",
    "vllm_runtime_install_requirements",
    "vllm_runtime_lock_path",
    "validate_vllm_runtime_lock_platform",
    "verify_installed_vllm_runtime_lock",
    "main",
    "serving_environment_profile",
    "serving_environment_profile_to_record",
    "serving_environment_profiles",
    "serving_environment_profiles_to_record",
    "write_serving_environment_profiles_record_json",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
