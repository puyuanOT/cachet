from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from typing import Any

import pytest

import document_kv_cache.gpu_qualification as qualification_v1
import document_kv_cache.gpu_qualification_v2 as qualification_v2
from document_kv_cache.gpu_qualification_v2 import (
    GPU_QUALIFICATION_V2_ARTIFACT_KEYS,
    GPU_QUALIFICATION_V2_INSTALLED_DISTRIBUTION_COUNT,
    GPU_QUALIFICATION_V2_JOB_RESULT_RECORD_TYPE,
    GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX,
    GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS,
    GPU_QUALIFICATION_V2_PLAN_RECORD_TYPE,
    GPU_QUALIFICATION_V2_RUNTIME_VERIFICATION_RECORD_TYPE,
    GPU_QUALIFICATION_V2_SCHEMA_VERSION,
    GPU_QUALIFICATION_V2_WITH_FLASHINFER_DISTRIBUTION_COUNT,
    GPU_QUALIFICATION_V2_WITH_VLLM_DISTRIBUTION_COUNT,
    GPUQualificationArtifactPinsV2,
    build_gpu_job_result_v2,
    build_gpu_qualification_plan_v2,
    build_gpu_runtime_verification_v2,
    gpu_qualification_v2_runtime_closure,
    pins_from_gpu_qualification_plan_v2,
    validate_gpu_qualification_plan_v2_record,
    validate_gpu_qualification_v2_runtime_attestation,
    validate_gpu_job_result_v2_record,
    validate_gpu_runtime_verification_v2_record,
)
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
    PUBLICATION_CAMPAIGN_ID,
    PUBLICATION_CAMPAIGN_LEDGER_ID,
    PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
    PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX,
    PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS,
    PUBLICATION_CAMPAIGN_PRE_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_CAMPAIGN_OPENING_LEDGER_PREFIX,
    PUBLICATION_CAMPAIGN_PRE_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS,
)


EXPECTED_ARTIFACT_SHA256 = {
    "cachet_source_tree_sha256": "a" * 64,
    "input_bundle_sha256": (
        "7ff6cf6a1553c0e844853d21de9780c75211f1be8304754da72e9cbebbd164ec"
    ),
    "package_wheel_sha256": "b" * 64,
    "patched_flashinfer_wheel_sha256": (
        "04e032c70234e8769f5ab7e787231c339a5b5230fca5f5b0b80f1a2a0ccad6ec"
    ),
    "patched_vllm_wheel_sha256": (
        "65120c48a9352b9eb65bab7a67090558d27af985ad366e469d3b87751073cff4"
    ),
    "runner_sha256": "c" * 64,
    "runtime_closure_manifest_sha256": (
        "c13c25a4e116f15db31e2efdbaebdd2d76418c5e4eb2f72fb2af3d8b8090e7df"
    ),
    "runtime_lock_sha256": (
        "c4fc0e055f0838ff397012f52bd4c4f0d22426db8a5fc8faf01689510e258903"
    ),
}
EXPECTED_JOB_IDS = (
    "aws-g6-l4-forced-triton-runtime-handoff",
    "aws-g6-l4-packed-page-roundtrip",
    "aws-g6-l4-matched-token-logit",
    "aws-g5-a10g-forced-triton-runtime-handoff",
    "aws-g5-a10g-packed-page-roundtrip",
    "aws-g5-a10g-matched-token-logit",
    "aws-g6-l4-32k-c4-gmu-70",
    "aws-g6-l4-32k-c4-gmu-75",
    "aws-g6-l4-32k-c4-gmu-80",
    "aws-g5-a10g-16k-c4-capacity",
    "aws-g6-l4-generation-throughput",
    "aws-g6e-l40s-generation-throughput",
    "aws-g6-l4-auto-backend-diagnostic",
    "aws-g5-a10g-auto-backend-diagnostic",
)
EXPECTED_PLAN_SHA256 = (
    "ee3633398b77e9181e7dfa21bdb867c64f8c82bedf2ffd4c0ce8508646b31479"
)
EXPECTED_RUNTIME_VERIFICATION_SHA256 = (
    "9cbd020655cd389d020d3b3528dbb6f3f1473cf971a93887b2b6d83e10d50493"
)
EXPECTED_VLLM_MEMBER_SHA256 = {
    "vllm/model_executor/layers/attention/attention.py": (
        "5735acfb390cf344caeec950c2f286344bcd84721ce287e0a56701f2a18bc839"
    ),
    "vllm/v1/attention/backends/triton_attn.py": (
        "4dae0ff6c4ee8f11c1f195151a11673d595d457c413032e7bae7550913f94390"
    ),
    "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py": (
        "0682ca7bc56edf7cea5419188a81c78510b54192471472b160aa447ac0ceeb08"
    ),
}
EXPECTED_RUNTIME_CLOSURE = {
    "base_lock": {
        "bytes": 376_326,
        "distribution_count": 195,
        "hash_count": 4_137,
        "sha256": EXPECTED_ARTIFACT_SHA256["runtime_lock_sha256"],
    },
    "distribution_counts": {
        "base_lock": 195,
        "separately_allowed_cachet": 1,
        "with_flashinfer": 196,
        "with_vllm": 197,
    },
    "flashinfer": {
        "manifest_closed_record_sha256": (
            "60c5a3aa75914c8fb6d790802adb2a5291a5eaffc281ee17080a3609193e229d"
        ),
        "manifest_file_bytes": 2_970,
        "manifest_file_sha256": (
            "4b5c3f726552697a2afa0c3f64655621d2201f9576de9306a986a696681d7303"
        ),
        "patched_member_sha256": (
            "05a4e1fa20c92b71de07f83695e8209c9f6d226072a6ea79a766af89c9fc3f25"
        ),
        "version": "0.6.16.post3",
        "wheel_bytes": 83_113_106,
        "wheel_sha256": EXPECTED_ARTIFACT_SHA256["patched_flashinfer_wheel_sha256"],
    },
    "install_order": [
        "runtime_lock_sha256",
        "patched_vllm_wheel_sha256",
        "patched_flashinfer_wheel_sha256",
        "package_wheel_sha256",
    ],
    "manifest": {
        "closed_record_sha256": (
            "b2cc4f90bf3e5e47ca23bc7b2117725faa9b114f0d1d803af6c89ae18ca05aaf"
        ),
        "file_bytes": 6_634,
        "file_sha256": EXPECTED_ARTIFACT_SHA256["runtime_closure_manifest_sha256"],
        "record_type": "document_kv.vllm_flashinfer_runtime_artifact_closure.v1",
        "schema_version": 1,
    },
    "pep610_direct_url_required": ["flashinfer-python", "vllm"],
    "pip_check_required": True,
    "vllm": {
        "manifest_file_bytes": 2_615,
        "manifest_file_sha256": (
            "14611e163e720f0fdeae6ef2704cecd9202eef6adc6336f892afd94a96726ef6"
        ),
        "member_sha256": EXPECTED_VLLM_MEMBER_SHA256,
        "version": "0.27.1+cu129",
        "wheel_bytes": 537_751_595,
        "wheel_sha256": EXPECTED_ARTIFACT_SHA256["patched_vllm_wheel_sha256"],
    },
}


def _pins() -> GPUQualificationArtifactPinsV2:
    return GPUQualificationArtifactPinsV2(**EXPECTED_ARTIFACT_SHA256)


def _valid_plan() -> dict[str, Any]:
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


def _valid_v1_plan() -> dict[str, Any]:
    return qualification_v1.build_gpu_qualification_plan(
        campaign_id=PUBLICATION_CAMPAIGN_ID,
        campaign_record_sha256=PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
        campaign_ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
        campaign_ledger_path_sha256=PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
        campaign_ledger_prefix=PUBLICATION_CAMPAIGN_OPENING_LEDGER_PREFIX,
        campaign_opening_terminal_gpu_hours=(
            PUBLICATION_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
        ),
        artifact_pins=_pins().v1_projection(),
    )


def _seal_v2(record: dict[str, Any]) -> None:
    record["closed_record_sha256"] = ""
    record["closed_record_sha256"] = sha256(
        qualification_v1.canonical_gpu_qualification_json(record).encode("utf-8")
    ).hexdigest()


def _valid_attestation() -> dict[str, Any]:
    return {
        "base_lock_distribution_count": 195,
        "base_lock_hash_count": 4_137,
        "base_lock_sha256": EXPECTED_ARTIFACT_SHA256["runtime_lock_sha256"],
        "cachet_package_version": "0.2.0",
        "flashinfer_annotation": "tuple[tuple[int, int, array.array[int]]]",
        "flashinfer_direct_url": (
            "file:///runtime/flashinfer_python-0.6.16.post3-py3-none-any.whl"
        ),
        "flashinfer_import_ok": True,
        "closure_bound_flashinfer_manifest_closed_record_sha256": (
            "60c5a3aa75914c8fb6d790802adb2a5291a5eaffc281ee17080a3609193e229d"
        ),
        "closure_bound_flashinfer_manifest_file_sha256": (
            "4b5c3f726552697a2afa0c3f64655621d2201f9576de9306a986a696681d7303"
        ),
        "closure_bound_vllm_manifest_file_sha256": (
            "14611e163e720f0fdeae6ef2704cecd9202eef6adc6336f892afd94a96726ef6"
        ),
        "flashinfer_member_sha256": (
            "05a4e1fa20c92b71de07f83695e8209c9f6d226072a6ea79a766af89c9fc3f25"
        ),
        "flashinfer_package_version": "0.6.16.post3",
        "flashinfer_wheel_sha256": EXPECTED_ARTIFACT_SHA256[
            "patched_flashinfer_wheel_sha256"
        ],
        "installed_distribution_count": 198,
        "ok": True,
        "packaged_base_lock_sha256": EXPECTED_ARTIFACT_SHA256["runtime_lock_sha256"],
        "pip_check_ok": True,
        "runtime_closure_closed_record_sha256": (
            "b2cc4f90bf3e5e47ca23bc7b2117725faa9b114f0d1d803af6c89ae18ca05aaf"
        ),
        "runtime_closure_file_sha256": EXPECTED_ARTIFACT_SHA256[
            "runtime_closure_manifest_sha256"
        ],
        "unexpected_distributions": [],
        "vllm_direct_url": "file:///runtime/vllm-0.27.1%2Bcu129.whl",
        "vllm_member_sha256": deepcopy(EXPECTED_VLLM_MEMBER_SHA256),
        "vllm_package_version": "0.27.1+cu129",
        "vllm_wheel_sha256": EXPECTED_ARTIFACT_SHA256["patched_vllm_wheel_sha256"],
        "with_flashinfer_distribution_count": 196,
        "with_vllm_distribution_count": 197,
    }


def _valid_runtime_verification() -> tuple[dict[str, Any], dict[str, Any], str]:
    plan = _valid_plan()
    job_id = plan["cloud_qualification"]["jobs"][0]["job_id"]
    record = build_gpu_runtime_verification_v2(
        plan_sha256=plan["closed_record_sha256"],
        job_id=job_id,
        artifact_sha256=_pins().to_record(),
        attestation=_valid_attestation(),
    )
    return plan, record, job_id


def test_v2_artifact_pins_have_exact_keys_order_and_authority_hashes() -> None:
    pins = _pins()

    assert GPU_QUALIFICATION_V2_ARTIFACT_KEYS == tuple(EXPECTED_ARTIFACT_SHA256)
    assert tuple(pins.to_record()) == GPU_QUALIFICATION_V2_ARTIFACT_KEYS
    assert pins.to_record() == EXPECTED_ARTIFACT_SHA256


@pytest.mark.parametrize(
    "field_name",
    [
        "input_bundle_sha256",
        "patched_flashinfer_wheel_sha256",
        "patched_vllm_wheel_sha256",
        "runtime_closure_manifest_sha256",
        "runtime_lock_sha256",
    ],
)
def test_v2_artifact_pins_reject_alternate_fixed_hashes(field_name: str) -> None:
    values = dict(EXPECTED_ARTIFACT_SHA256)
    values[field_name] = "0" * 64

    with pytest.raises(ValueError, match="reviewed v2 authority"):
        GPUQualificationArtifactPinsV2(**values)


def test_v2_artifact_pins_require_lowercase_sha256() -> None:
    values = dict(EXPECTED_ARTIFACT_SHA256)
    values["runner_sha256"] = "A" * 64

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        GPUQualificationArtifactPinsV2(**values)


def test_v2_plan_is_deterministic_and_closes_exactly_fourteen_jobs() -> None:
    first = _valid_plan()
    second = _valid_plan()
    jobs = first["cloud_qualification"]["jobs"]

    assert first == second
    assert first["closed_record_sha256"] == EXPECTED_PLAN_SHA256
    assert first["record_type"] == GPU_QUALIFICATION_V2_PLAN_RECORD_TYPE
    assert first["schema_version"] == GPU_QUALIFICATION_V2_SCHEMA_VERSION
    assert first["cloud_qualification"]["job_count"] == 14
    assert tuple(job["job_id"] for job in jobs) == EXPECTED_JOB_IDS
    assert all(job["attempt_number"] == 0 for job in jobs)
    assert all(job["max_retries"] == 0 for job in jobs)
    assert first["runtime_contract"]["artifact_sha256"] == (EXPECTED_ARTIFACT_SHA256)
    validate_gpu_qualification_plan_v2_record(
        first,
        expected_campaign_id=PUBLICATION_CAMPAIGN_ID,
        expected_artifact_pins=_pins(),
    )


def test_v2_plan_rejects_rebased_opening_authority() -> None:
    with pytest.raises(ValueError, match="ledger prefix differs"):
        build_gpu_qualification_plan_v2(
            campaign_id=PUBLICATION_CAMPAIGN_ID,
            campaign_record_sha256=PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
            campaign_ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
            campaign_ledger_path_sha256=PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
            campaign_ledger_prefix=(
                PUBLICATION_CAMPAIGN_PRE_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_CAMPAIGN_OPENING_LEDGER_PREFIX
            ),
            campaign_opening_terminal_gpu_hours=(
                GPU_QUALIFICATION_V2_OPENING_TERMINAL_GPU_HOURS
            ),
            artifact_pins=_pins(),
        )
    with pytest.raises(ValueError, match="opening balance differs"):
        build_gpu_qualification_plan_v2(
            campaign_id=PUBLICATION_CAMPAIGN_ID,
            campaign_record_sha256=PUBLICATION_CAMPAIGN_CLOSED_RECORD_SHA256,
            campaign_ledger_id=PUBLICATION_CAMPAIGN_LEDGER_ID,
            campaign_ledger_path_sha256=PUBLICATION_CAMPAIGN_LEDGER_PATH_SHA256,
            campaign_ledger_prefix=GPU_QUALIFICATION_V2_OPENING_LEDGER_PREFIX,
            campaign_opening_terminal_gpu_hours=(
                PUBLICATION_CAMPAIGN_PRE_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_CAMPAIGN_OPENING_TERMINAL_GPU_HOURS
            ),
            artifact_pins=_pins(),
        )

    forged = _valid_plan()
    forged["campaign_ledger_prefix"] = (
        PUBLICATION_CAMPAIGN_PRE_MIXED_SENTINEL_AND_RESULT_VALIDATION_FAILURE_CAMPAIGN_OPENING_LEDGER_PREFIX.to_record()
    )
    forged["closed_record_sha256"] = qualification_v2._closed_record_sha256(  # noqa: SLF001
        forged
    )
    with pytest.raises(ValueError, match="ledger prefix differs"):
        validate_gpu_qualification_plan_v2_record(
            forged,
            expected_campaign_id=PUBLICATION_CAMPAIGN_ID,
            expected_artifact_pins=_pins(),
        )


def test_v1_and_v2_plans_are_cross_rejected() -> None:
    v1_plan = _valid_v1_plan()
    v2_plan = _valid_plan()

    with pytest.raises(ValueError):
        validate_gpu_qualification_plan_v2_record(
            v1_plan,
            expected_campaign_id=PUBLICATION_CAMPAIGN_ID,
            expected_artifact_pins=_pins(),
        )
    with pytest.raises(ValueError):
        qualification_v1.validate_gpu_qualification_plan_record(
            v2_plan,
            expected_campaign_id=PUBLICATION_CAMPAIGN_ID,
            expected_artifact_pins=_pins().v1_projection(),
        )


@pytest.mark.parametrize("schema_version", [True, 2.0])
def test_v2_plan_rejects_non_exact_integer_schema(schema_version: Any) -> None:
    plan = _valid_plan()
    plan["schema_version"] = schema_version
    _seal_v2(plan)

    with pytest.raises(ValueError, match="schema_version"):
        validate_gpu_qualification_plan_v2_record(
            plan,
            expected_campaign_id=PUBLICATION_CAMPAIGN_ID,
            expected_artifact_pins=_pins(),
        )


@pytest.mark.parametrize("malformation", ["reordered", "missing", "extra"])
def test_v2_plan_pin_extraction_requires_exact_order_and_coverage(
    malformation: str,
) -> None:
    plan = _valid_plan()
    pins = dict(plan["runtime_contract"]["artifact_sha256"])
    if malformation == "reordered":
        pins = dict(reversed(tuple(pins.items())))
    elif malformation == "missing":
        pins.pop("runner_sha256")
    else:
        pins["extra_sha256"] = "d" * 64
    plan["runtime_contract"]["artifact_sha256"] = pins

    with pytest.raises(ValueError, match="canonical eight-key coverage"):
        pins_from_gpu_qualification_plan_v2(plan)


def test_v2_plan_validator_rejects_resealed_reordered_pin_mapping() -> None:
    plan = _valid_plan()
    pins = plan["runtime_contract"]["artifact_sha256"]
    plan["runtime_contract"]["artifact_sha256"] = dict(reversed(tuple(pins.items())))
    _seal_v2(plan)
    with pytest.raises(ValueError, match="canonical eight-key coverage"):
        validate_gpu_qualification_plan_v2_record(
            plan,
            expected_campaign_id=PUBLICATION_CAMPAIGN_ID,
            expected_artifact_pins=_pins(),
        )


def test_v2_runtime_closure_freezes_install_order_counts_and_identities() -> None:
    assert gpu_qualification_v2_runtime_closure() == EXPECTED_RUNTIME_CLOSURE
    assert GPU_QUALIFICATION_V2_WITH_FLASHINFER_DISTRIBUTION_COUNT == 196
    assert GPU_QUALIFICATION_V2_WITH_VLLM_DISTRIBUTION_COUNT == 197
    assert GPU_QUALIFICATION_V2_INSTALLED_DISTRIBUTION_COUNT == 198
    assert GPU_QUALIFICATION_V2_JOB_RESULT_RECORD_TYPE == (
        "cachet.vllm_0271_gpu_job_result.v2"
    )


def test_v2_runtime_attestation_accepts_only_the_exact_closed_contract() -> None:
    attestation = _valid_attestation()

    assert len(attestation) == 26
    validate_gpu_qualification_v2_runtime_attestation(attestation)


@pytest.mark.parametrize("malformation", ["missing", "extra"])
def test_v2_runtime_attestation_rejects_missing_or_extra_keys(
    malformation: str,
) -> None:
    attestation = _valid_attestation()
    if malformation == "missing":
        attestation.pop("flashinfer_import_ok")
    else:
        attestation["extra"] = True

    with pytest.raises(ValueError, match="closed schema"):
        validate_gpu_qualification_v2_runtime_attestation(attestation)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("ok", False),
        ("base_lock_distribution_count", 195.0),
        ("installed_distribution_count", True),
        ("flashinfer_wheel_sha256", "0" * 64),
        ("unexpected_distributions", ["foreign-package"]),
        ("flashinfer_direct_url", "https://example.invalid/flashinfer.whl"),
        ("flashinfer_direct_url", "file:///runtime/../flashinfer.whl"),
        ("flashinfer_direct_url", "file:////runtime/flashinfer.whl"),
        ("flashinfer_direct_url", "file:///runtime/%00flashinfer.whl"),
        ("cachet_package_version", ""),
        ("cachet_package_version", " 0.2.0 "),
    ],
)
def test_v2_runtime_attestation_rejects_tampered_values(
    field_name: str, replacement: Any
) -> None:
    attestation = _valid_attestation()
    attestation[field_name] = replacement

    with pytest.raises(ValueError, match=field_name):
        validate_gpu_qualification_v2_runtime_attestation(attestation)


def test_v2_runtime_attestation_rejects_tampered_member_digest() -> None:
    attestation = _valid_attestation()
    members = attestation["vllm_member_sha256"]
    assert isinstance(members, dict)
    members["vllm/model_executor/layers/attention/attention.py"] = "0" * 64

    with pytest.raises(ValueError, match="vllm_member_sha256"):
        validate_gpu_qualification_v2_runtime_attestation(attestation)


def test_v2_runtime_verification_is_sealed_to_plan_job_and_all_eight_pins() -> None:
    plan, record, job_id = _valid_runtime_verification()

    assert record["record_type"] == (
        GPU_QUALIFICATION_V2_RUNTIME_VERIFICATION_RECORD_TYPE
    )
    assert record["schema_version"] == GPU_QUALIFICATION_V2_SCHEMA_VERSION
    assert record["closed_record_sha256"] == (EXPECTED_RUNTIME_VERIFICATION_SHA256)
    assert record["plan_sha256"] == EXPECTED_PLAN_SHA256
    assert record["job_id"] == job_id
    assert tuple(record["artifact_sha256"]) == GPU_QUALIFICATION_V2_ARTIFACT_KEYS
    assert record["artifact_sha256"] == EXPECTED_ARTIFACT_SHA256
    validate_gpu_runtime_verification_v2_record(
        record,
        plan_record=plan,
        expected_job_id=job_id,
        expected_artifact_pins=_pins(),
    )


def test_v2_runtime_verification_builder_rejects_reordered_pins() -> None:
    plan = _valid_plan()
    reordered = dict(reversed(tuple(_pins().to_record().items())))

    with pytest.raises(ValueError, match="exact eight-key artifacts"):
        build_gpu_runtime_verification_v2(
            plan_sha256=plan["closed_record_sha256"],
            job_id=EXPECTED_JOB_IDS[0],
            artifact_sha256=reordered,
            attestation=_valid_attestation(),
        )


def test_v2_runtime_verification_validator_rejects_resealed_reordered_pins() -> None:
    plan, record, job_id = _valid_runtime_verification()
    record["artifact_sha256"] = dict(reversed(tuple(record["artifact_sha256"].items())))
    _seal_v2(record)
    with pytest.raises(ValueError, match="artifact pins differ"):
        validate_gpu_runtime_verification_v2_record(
            record,
            plan_record=plan,
            expected_job_id=job_id,
            expected_artifact_pins=_pins(),
        )


@pytest.mark.parametrize(
    ("binding", "message"),
    [
        ("pin", "artifact pins differ"),
        ("job", "job ID differs"),
        ("plan", "plan SHA-256 differs"),
    ],
)
def test_v2_runtime_verification_rejects_resealed_binding_tamper(
    binding: str, message: str
) -> None:
    plan, record, job_id = _valid_runtime_verification()
    if binding == "pin":
        artifact_sha256 = record["artifact_sha256"]
        assert isinstance(artifact_sha256, dict)
        artifact_sha256["runner_sha256"] = "0" * 64
    elif binding == "job":
        record["job_id"] = "aws-g5-a10g-packed-page-roundtrip"
    else:
        record["plan_sha256"] = "0" * 64
    _seal_v2(record)

    with pytest.raises(ValueError, match=message):
        validate_gpu_runtime_verification_v2_record(
            record,
            plan_record=plan,
            expected_job_id=job_id,
            expected_artifact_pins=_pins(),
        )


def test_v2_runtime_verification_rejects_closed_digest_tamper() -> None:
    plan, record, job_id = _valid_runtime_verification()
    record["closed_record_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="closed_record_sha256 differs"):
        validate_gpu_runtime_verification_v2_record(
            record,
            plan_record=plan,
            expected_job_id=job_id,
            expected_artifact_pins=_pins(),
        )


@pytest.mark.parametrize("schema_version", [True, 2.0])
def test_v2_runtime_verification_rejects_non_exact_integer_schema(
    schema_version: Any,
) -> None:
    plan, record, job_id = _valid_runtime_verification()
    record["schema_version"] = schema_version
    _seal_v2(record)

    with pytest.raises(ValueError, match="schema_version"):
        validate_gpu_runtime_verification_v2_record(
            record,
            plan_record=plan,
            expected_job_id=job_id,
            expected_artifact_pins=_pins(),
        )


def test_v2_forced_handoff_result_requires_one_identical_runtime_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _valid_plan()
    pins = _pins()
    job_id = "aws-g6-l4-forced-triton-runtime-handoff"
    plan_sha256 = plan["closed_record_sha256"]
    attestation = _valid_attestation()
    verification = build_gpu_runtime_verification_v2(
        plan_sha256=plan_sha256,
        job_id=job_id,
        artifact_sha256=pins.to_record(),
        attestation=attestation,
    )

    def validate_measurements(
        value: dict[str, Any],
        *,
        hardware_id: str,
        attestation_validator: Any,
    ) -> None:
        assert hardware_id == "aws-g6-l4"
        attestation_validator(value["runtime_lock_attestation"])

    monkeypatch.setattr(
        qualification_v1,
        "_validate_runtime_handoff_measurements_with_attestation",
        validate_measurements,
    )

    def result(measured_attestation: dict[str, Any]) -> dict[str, Any]:
        return build_gpu_job_result_v2(
            plan_record=plan,
            job_id=job_id,
            reservation_attempt_id=qualification_v1._expected_reservation_attempt_id(
                plan_sha256, job_id
            ),
            task_key=qualification_v1._expected_task_key(job_id),
            output_json=(
                "dbfs:/Volumes/catalog/schema/volume/output/"
                f"{plan_sha256}/{job_id}/gpu-job-result.json"
            ),
            cloud_run_id="123",
            cloud_cluster_id="cluster-1",
            started_at_utc="2026-08-25T00:00:00Z",
            finished_at_utc="2026-08-25T00:00:01Z",
            nvidia_driver_version="580.65.06",
            observed_gpu="NVIDIA L4",
            observed_gpu_compute_capability="8.9",
            observed_vllm_version="0.27.1+cu129",
            observed_torch_cuda_version="12.9",
            observed_artifact_sha256=pins.to_record(),
            runtime_verification=verification,
            measurements={"runtime_lock_attestation": measured_attestation},
        )

    matching = result(attestation)
    validate_gpu_job_result_v2_record(
        matching,
        plan_record=plan,
        expected_artifact_pins=pins,
    )
    reordered = deepcopy(matching)
    reordered["artifact_sha256"] = dict(
        reversed(tuple(reordered["artifact_sha256"].items()))
    )
    _seal_v2(reordered)
    with pytest.raises(ValueError, match="artifact hashes differ"):
        validate_gpu_job_result_v2_record(
            reordered,
            plan_record=plan,
            expected_artifact_pins=pins,
        )
    alternate = deepcopy(attestation)
    alternate["vllm_direct_url"] = "file:///runtime/alternate-vllm.whl"
    with pytest.raises(ValueError, match="attestations differ"):
        validate_gpu_job_result_v2_record(
            result(alternate),
            plan_record=plan,
            expected_artifact_pins=pins,
        )
