"""Serving-engine environment profiles for isolated backend installs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from restaurant_kv_serving.engine_adapters import ServingBackend

SERVING_ENVIRONMENT_PROFILES_RECORD_TYPE = "document_kv.serving_environment_profiles.v1"

VLLM_VERSION = "0.23.0"
TRANSFORMERS_CONSTRAINT = "transformers==5.12.1"
HUGGINGFACE_HUB_CONSTRAINT = "huggingface-hub==1.20.1"
TOKENIZERS_CONSTRAINT = "tokenizers==0.22.2"
NUMPY_CONSTRAINT = "numpy==2.4.6"
FASTAPI_CONSTRAINT = "fastapi[standard]==0.137.2"
PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT = "prometheus-fastapi-instrumentator==8.0.0"
VLLM_DEPENDENCY_CONSTRAINTS = (
    f"vllm=={VLLM_VERSION}",
    TRANSFORMERS_CONSTRAINT,
    HUGGINGFACE_HUB_CONSTRAINT,
    TOKENIZERS_CONSTRAINT,
    NUMPY_CONSTRAINT,
    FASTAPI_CONSTRAINT,
    PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT,
)

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
    engine_version=VLLM_VERSION,
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


__all__ = [
    "FASTAPI_CONSTRAINT",
    "HUGGINGFACE_HUB_CONSTRAINT",
    "NUMPY_CONSTRAINT",
    "PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT",
    "SERVING_ENVIRONMENT_PROFILES_RECORD_TYPE",
    "SGLANG_DEPENDENCY_CONSTRAINTS",
    "SGLANG_SERVING_ENVIRONMENT_PROFILE",
    "SGLANG_VERSION",
    "ServingEnvironmentProfile",
    "TOKENIZERS_CONSTRAINT",
    "TRANSFORMERS_CONSTRAINT",
    "VLLM_DEPENDENCY_CONSTRAINTS",
    "VLLM_SERVING_ENVIRONMENT_PROFILE",
    "VLLM_VERSION",
    "serving_environment_profile",
    "serving_environment_profile_to_record",
    "serving_environment_profiles",
    "serving_environment_profiles_to_record",
]
