import json

import pytest

import document_kv_cache.serving_env as public_serving_env
from document_kv_cache import ServingBackend
from document_kv_cache.serving_env import (
    ACCELERATE_CONSTRAINT,
    BITSANDBYTES_CONSTRAINT,
    FASTAPI_CONSTRAINT,
    HUGGINGFACE_HUB_CONSTRAINT,
    NUMPY_CONSTRAINT,
    PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT,
    SERVING_ENVIRONMENT_PROFILES_RECORD_TYPE,
    SGLANG_DEPENDENCY_CONSTRAINTS,
    SGLANG_SERVING_ENVIRONMENT_PROFILE,
    SGLANG_VERSION,
    ServingEnvironmentProfile,
    TOKENIZERS_CONSTRAINT,
    TRANSFORMERS_CONSTRAINT,
    VLLM_DEPENDENCY_CONSTRAINTS,
    VLLM_SERVING_ENVIRONMENT_PROFILE,
    VLLM_VERSION,
    serving_environment_profile,
    serving_environment_profile_to_record,
    serving_environment_profiles,
    serving_environment_profiles_to_record,
    write_serving_environment_profiles_record_json,
)


def test_serving_environment_profiles_are_backend_scoped_and_exactly_pinned():
    vllm_profile = serving_environment_profile("vllm")
    sglang_profile = serving_environment_profile(ServingBackend.SGLANG)

    assert vllm_profile is VLLM_SERVING_ENVIRONMENT_PROFILE
    assert sglang_profile is SGLANG_SERVING_ENVIRONMENT_PROFILE
    assert serving_environment_profiles() == (vllm_profile, sglang_profile)

    assert vllm_profile.backend == ServingBackend.VLLM
    assert vllm_profile.engine_package == "vllm"
    assert vllm_profile.engine_version == "0.23.0"
    assert vllm_profile.dependency_constraints == VLLM_DEPENDENCY_CONSTRAINTS
    assert VLLM_DEPENDENCY_CONSTRAINTS == (
        f"vllm=={VLLM_VERSION}",
        TRANSFORMERS_CONSTRAINT,
        HUGGINGFACE_HUB_CONSTRAINT,
        TOKENIZERS_CONSTRAINT,
        NUMPY_CONSTRAINT,
        FASTAPI_CONSTRAINT,
        PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT,
        BITSANDBYTES_CONSTRAINT,
        ACCELERATE_CONSTRAINT,
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
        "engine_version": VLLM_VERSION,
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
            dependency_constraints=("vllm>=0.23",),
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
