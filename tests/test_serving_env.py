import json
from hashlib import sha256

import pytest

import document_kv_cache.serving_env as public_serving_env
from document_kv_cache import ServingBackend
from document_kv_cache.serving_env import (
    ACCELERATE_CONSTRAINT,
    APACHE_TVM_FFI_CONSTRAINT,
    BITSANDBYTES_CONSTRAINT,
    FASTAPI_CONSTRAINT,
    FASTSAFETENSORS_CONSTRAINT,
    FLASHINFER_CUBIN_CONSTRAINT,
    FLASHINFER_CONSTRAINT,
    FLASHINFER_JIT_CACHE_CONSTRAINT,
    FLASHINFER_PYTHON_CONSTRAINT,
    HUGGINGFACE_HUB_CONSTRAINT,
    HUMMING_KERNELS_CONSTRAINT,
    NVIDIA_CUTLASS_DSL_CONSTRAINT,
    NVIDIA_CUDNN_FRONTEND_CONSTRAINT,
    NVTX_CONSTRAINT,
    NUMPY_CONSTRAINT,
    NUMBA_CONSTRAINT,
    OPENCV_PYTHON_HEADLESS_CONSTRAINT,
    PYNVVIDEOCODEC_CONSTRAINT,
    PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT,
    QUACK_KERNELS_CONSTRAINT,
    SERVING_ENVIRONMENT_PROFILES_RECORD_TYPE,
    SGLANG_DEPENDENCY_CONSTRAINTS,
    SGLANG_SERVING_ENVIRONMENT_PROFILE,
    SGLANG_VERSION,
    ServingEnvironmentProfile,
    TOKENIZERS_CONSTRAINT,
    TORCHCODEC_CONSTRAINT,
    TOKENSPEED_MLA_CONSTRAINT,
    TILELANG_CONSTRAINT,
    TORCH_CONSTRAINT,
    TORCHAUDIO_CONSTRAINT,
    TORCHVISION_CONSTRAINT,
    TRANSFORMERS_CONSTRAINT,
    TRITON_CONSTRAINT,
    VLLM_DEPENDENCY_CONSTRAINTS,
    VLLM_CUDA_REQUIREMENTS_SHA256,
    VLLM_CUDA_VARIANT,
    VLLM_CUDA_VERSION,
    VLLM_DOCKERFILE_SHA256,
    VLLM_FLASHINFER_INDEX_URL,
    VLLM_FLASHINFER_JIT_INDEX_URL,
    VLLM_INSTALL_REQUIREMENTS,
    VLLM_PACKAGE_VERSION,
    VLLM_PACKAGE_INDEX_URLS,
    VLLM_PINNED_CUDA_DIRECT_CONSTRAINTS,
    VLLM_PYTORCH_INDEX_URL,
    VLLM_RUNTIME_LOCK_DISTRIBUTION_COUNT,
    VLLM_RUNTIME_LOCK_FILENAME,
    VLLM_RUNTIME_LOCK_SHA256,
    VLLM_SERVING_ENVIRONMENT_PROFILE,
    VLLM_VERSION,
    VLLM_WHEEL_FILENAME,
    VLLM_WHEEL_INSTALL_SPEC,
    VLLM_WHEEL_SHA256,
    VLLM_WHEEL_URL,
    serving_environment_profile,
    serving_environment_profile_to_record,
    serving_environment_profiles,
    serving_environment_profiles_to_record,
    validate_vllm_runtime_lock_platform,
    vllm_runtime_lock_path,
    write_serving_environment_profiles_record_json,
)


def test_vllm_runtime_lock_is_packaged_hashed_and_excludes_vllm():
    lock_path = vllm_runtime_lock_path()
    lock_bytes = lock_path.read_bytes()
    lock_text = lock_bytes.decode("utf-8")
    requirement_lines = [
        line
        for line in lock_text.splitlines()
        if line and line[0].isalnum() and "==" in line
    ]

    assert lock_path.name == VLLM_RUNTIME_LOCK_FILENAME
    assert sha256(lock_bytes).hexdigest() == VLLM_RUNTIME_LOCK_SHA256
    assert len(requirement_lines) == VLLM_RUNTIME_LOCK_DISTRIBUTION_COUNT == 196
    assert not any(line.startswith("vllm") for line in requirement_lines)
    assert "pip==26.2.1 \\" in lock_text
    assert "setuptools==80.9.0 \\" in lock_text
    assert "wheel==0.48.0 \\" in lock_text
    assert "torchcodec==0.16.0+cu129 \\" in lock_text
    assert lock_text.count("--hash=sha256:") >= len(requirement_lines)


def test_vllm_runtime_lock_readme_separates_replay_from_regeneration():
    readme = vllm_runtime_lock_path().with_name("README.md").read_text()

    assert "Byte-for-byte artifact replay" in readme
    assert "consuming the checked-in lock unchanged" in readme
    assert VLLM_RUNTIME_LOCK_SHA256 in readme
    assert "python -m pip install uv==0.11.6" in readme
    assert "--python-version 3.11.11" in readme
    assert "--python-platform x86_64-manylinux_2_35" in readme
    assert "--index-strategy first-index" in readme


def test_vllm_runtime_lock_platform_is_exact():
    assert validate_vllm_runtime_lock_platform(
        python_version=(3, 11),
        operating_system="linux",
        machine="x86_64",
        libc=("glibc", "2.35"),
    ) == {
        "python": "3.11",
        "operating_system": "linux",
        "machine": "x86_64",
        "libc": "glibc-2.35",
    }
    with pytest.raises(RuntimeError, match="CPython 3.11"):
        validate_vllm_runtime_lock_platform(
            python_version=(3, 12),
            operating_system="linux",
            machine="x86_64",
            libc=("glibc", "2.35"),
        )
    with pytest.raises(RuntimeError, match="glibc 2.35"):
        validate_vllm_runtime_lock_platform(
            python_version=(3, 11),
            operating_system="linux",
            machine="x86_64",
            libc=("glibc", "2.36"),
        )


def test_installed_vllm_runtime_lock_verifies_versions_and_direct_url(monkeypatch):
    class FakeDistribution:
        def __init__(self, name, version, *, direct_url=None):
            self.metadata = {"Name": name}
            self.version = version
            self._direct_url = direct_url

        def read_text(self, filename):
            if filename != "direct_url.json" or self._direct_url is None:
                return None
            return json.dumps(self._direct_url)

    locked_versions = public_serving_env._vllm_runtime_lock_versions(
        vllm_runtime_lock_path()
    )
    locked_distributions = [
        FakeDistribution(name, version)
        for name, version in locked_versions.items()
    ]
    digest = "d" * 64
    wheel_uri = (
        "https://artifacts.example/vllm-0.27.1%2Bcu129-"
        "1cachete5m2dddddddddddddddd-cp38-abi3-manylinux_2_28_x86_64.whl"
    )
    vllm_distribution = FakeDistribution(
        "vllm",
        VLLM_PACKAGE_VERSION,
        direct_url={
            "url": wheel_uri,
            "archive_info": {"hashes": {"sha256": digest}},
        },
    )
    monkeypatch.setattr(
        public_serving_env.package_metadata,
        "distributions",
        lambda: [*locked_distributions, vllm_distribution],
    )
    monkeypatch.setattr(
        public_serving_env.package_metadata,
        "distribution",
        lambda name: vllm_distribution if name == "vllm" else None,
    )

    record = public_serving_env.verify_installed_vllm_runtime_lock(
        f"vllm @ {wheel_uri}#sha256={digest}"
    )

    assert record == {
        "runtime_lock_sha256": VLLM_RUNTIME_LOCK_SHA256,
        "locked_distribution_count": 196,
        "vllm_package_version": VLLM_PACKAGE_VERSION,
        "vllm_direct_url": wheel_uri,
        "vllm_wheel_sha256": digest,
        "unexpected_distributions": [],
        "ok": True,
    }

    locked_distributions[0].version = "0.0.0"
    with pytest.raises(RuntimeError, match="expected"):
        public_serving_env.verify_installed_vllm_runtime_lock(
            f"vllm @ {wheel_uri}#sha256={digest}"
        )


def test_serving_environment_profiles_are_backend_scoped_and_exactly_pinned():
    vllm_profile = serving_environment_profile("vllm")
    sglang_profile = serving_environment_profile(ServingBackend.SGLANG)

    assert vllm_profile is VLLM_SERVING_ENVIRONMENT_PROFILE
    assert sglang_profile is SGLANG_SERVING_ENVIRONMENT_PROFILE
    assert serving_environment_profiles() == (vllm_profile, sglang_profile)

    assert vllm_profile.backend == ServingBackend.VLLM
    assert vllm_profile.engine_package == "vllm"
    assert VLLM_VERSION == "0.27.1"
    assert vllm_profile.engine_version == "0.27.1+cu129"
    assert vllm_profile.dependency_constraints == VLLM_DEPENDENCY_CONSTRAINTS
    assert VLLM_DEPENDENCY_CONSTRAINTS == (
        f"vllm=={VLLM_PACKAGE_VERSION}",
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
    assert FLASHINFER_CONSTRAINT == FLASHINFER_PYTHON_CONSTRAINT
    assert VLLM_CUDA_VERSION == "12.9"
    assert VLLM_CUDA_VARIANT == "cu129"
    assert VLLM_CUDA_REQUIREMENTS_SHA256 == (
        "30091f418325ea9f97bc546cb03eb1a35e1cc20b2500b522b3dabd3b1aaee241"
    )
    assert VLLM_PACKAGE_INDEX_URLS == (
        VLLM_PYTORCH_INDEX_URL,
        VLLM_FLASHINFER_INDEX_URL,
        VLLM_FLASHINFER_JIT_INDEX_URL,
    )
    assert VLLM_PINNED_CUDA_DIRECT_CONSTRAINTS == (
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
    assert TORCH_CONSTRAINT.endswith("+cu129")
    assert TORCHAUDIO_CONSTRAINT.endswith("+cu129")
    assert TORCHVISION_CONSTRAINT.endswith("+cu129")
    assert "[cu12]" in HUMMING_KERNELS_CONSTRAINT
    assert "[cu13]" not in NVIDIA_CUTLASS_DSL_CONSTRAINT
    assert VLLM_DOCKERFILE_SHA256 == (
        "9876efaec74111cad4ce074225740fafa9461166973d1837dc0d4cd23c2f2509"
    )
    assert VLLM_WHEEL_FILENAME == (
        "vllm-0.27.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
    )
    assert VLLM_WHEEL_URL.endswith(VLLM_WHEEL_FILENAME.replace("+", "%2B"))
    assert VLLM_WHEEL_SHA256 == (
        "bf0d52faa2a51e7a01c6856a7a8a2d1307fd0ff711415d34168a67ffac0fa47b"
    )
    assert VLLM_WHEEL_INSTALL_SPEC == (
        f"vllm @ {VLLM_WHEEL_URL}#sha256={VLLM_WHEEL_SHA256}"
    )
    assert VLLM_INSTALL_REQUIREMENTS == (
        VLLM_WHEEL_INSTALL_SPEC,
        *VLLM_DEPENDENCY_CONSTRAINTS[1:],
    )

    assert sglang_profile.backend == ServingBackend.SGLANG
    assert sglang_profile.engine_package == "sglang"
    assert sglang_profile.engine_version == "0.5.10.post1"
    assert sglang_profile.dependency_constraints == SGLANG_DEPENDENCY_CONSTRAINTS
    assert SGLANG_DEPENDENCY_CONSTRAINTS == (f"sglang=={SGLANG_VERSION}",)

    for profile in serving_environment_profiles():
        assert profile.isolated_environment_required is True
        assert profile.notes
        assert all("==" in constraint for constraint in profile.dependency_constraints)


def test_serving_environment_profiles_serialize_to_stable_records():
    vllm_record = serving_environment_profile_to_record(VLLM_SERVING_ENVIRONMENT_PROFILE)

    assert vllm_record == {
        "backend": "vllm",
        "engine_package": "vllm",
        "engine_version": VLLM_PACKAGE_VERSION,
        "dependency_constraints": list(VLLM_DEPENDENCY_CONSTRAINTS),
        "isolated_environment_required": True,
        "notes": VLLM_SERVING_ENVIRONMENT_PROFILE.notes,
    }
    assert serving_environment_profiles_to_record() == {
        "record_type": SERVING_ENVIRONMENT_PROFILES_RECORD_TYPE,
        "profiles": [
            serving_environment_profile_to_record(VLLM_SERVING_ENVIRONMENT_PROFILE),
            serving_environment_profile_to_record(SGLANG_SERVING_ENVIRONMENT_PROFILE),
        ],
    }


def test_serving_environment_profiles_writer_and_cli_emit_stable_records(tmp_path, capsys):
    output_path = tmp_path / "serving-env.json"

    write_serving_environment_profiles_record_json(output_path)
    written_record = json.loads(output_path.read_text(encoding="utf-8"))

    assert written_record == serving_environment_profiles_to_record()

    assert public_serving_env.main([]) == 0
    stdout_record = json.loads(capsys.readouterr().out)

    assert stdout_record == serving_environment_profiles_to_record()


def test_serving_environment_profile_rejects_ambiguous_or_combined_runtime_pins():
    with pytest.raises(ValueError, match="Unsupported serving backend"):
        serving_environment_profile("unknown")

    with pytest.raises(ValueError, match="dependency_constraints must include"):
        ServingEnvironmentProfile(
            backend=ServingBackend.VLLM,
            engine_package="vllm",
            engine_version=VLLM_VERSION,
            dependency_constraints=(TRANSFORMERS_CONSTRAINT,),
            isolated_environment_required=True,
            notes="missing engine package pin",
        )

    with pytest.raises(ValueError, match="dependency_constraints must be exact package pins"):
        ServingEnvironmentProfile(
            backend=ServingBackend.VLLM,
            engine_package="vllm",
            engine_version=VLLM_VERSION,
            dependency_constraints=("vllm>=0.27.1",),
            isolated_environment_required=True,
            notes="range pin",
        )

    with pytest.raises(ValueError, match="isolated_environment_required must be a boolean"):
        ServingEnvironmentProfile(
            backend=ServingBackend.VLLM,
            engine_package="vllm",
            engine_version=VLLM_VERSION,
            dependency_constraints=(f"vllm=={VLLM_VERSION}",),
            isolated_environment_required=1,
            notes="bad boolean",
        )
