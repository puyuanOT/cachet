"""Run publication handoff tree closure on a mounted Databricks CPU node.

The Q8 and BF16 handoff trees are hundreds of gigabytes.  A controller must
not mirror them merely to execute the existing publication closure checks.
This module keeps those checks unchanged at their natural trust boundary: a
single-node CPU coordinator mounts the Unity Catalog volume, invokes the
existing close functions, and emits one compact, hash-bound result.  The Mac
controller issues an in-memory capability only after a direct ``runs/get``
join and an authenticated Files API download of that result.
"""

from __future__ import annotations

import argparse
import fcntl
import hmac
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, cast

import document_kv_cache.publication_bf16_handoff_generation as _bf16
import document_kv_cache.publication_handoff_artifacts as _handoff_artifacts
import document_kv_cache.publication_latency_handoff_generation as _q8
from document_kv_cache.databricks_resource_ledger import (
    DatabricksLedgerPrefix,
    canonical_databricks_submit_payload_snapshot,
    databricks_cluster_hour_ledger_from_record,
    databricks_cluster_hour_ledger_to_record,
    databricks_ledger_path_sha256,
    databricks_ledger_prefix,
    databricks_ledger_prefix_from_record,
    read_databricks_cluster_hour_ledger_json,
    require_databricks_batch_terminal_closure,
    require_databricks_ledger_prefix,
)
from document_kv_cache.databricks_runs import (
    DatabricksWorkspaceConfig,
    bind_databricks_run_idempotency_token,
    databricks_run_status_record,
    download_databricks_volume_file_bytes,
    get_databricks_run,
    submit_databricks_run,
    summarize_databricks_run,
    upload_databricks_volume_file_bytes_exclusive,
)
from document_kv_cache.main_latency_inputs import load_main_latency_tokenizer
from document_kv_cache.flashinfer_wheel_repack import (
    FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256,
    FLASHINFER_PATCHED_MANIFEST_FILE_SHA256,
    FLASHINFER_PATCHED_MANIFEST_SIZE,
    FLASHINFER_PATCHED_WHEEL_SHA256,
    FLASHINFER_PATCHED_WHEEL_SIZE,
    FLASHINFER_SOURCE_WHEEL_SHA256,
    FLASHINFER_SOURCE_WHEEL_SIZE,
)
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
    GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
)
from document_kv_cache.gpu_qualification_v2 import GPUQualificationArtifactPinsV2
from document_kv_cache.gpu_qualification_databricks import (
    GPUQualificationLaunchAuthorization,
    require_gpu_qualification_launch_authorization,
)
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_CPU_COORDINATOR_NODE_TYPE_ID,
    PUBLICATION_CAMPAIGN_CPU_COORDINATOR_SPARK_VERSION,
    PUBLICATION_CAMPAIGN_HANDOFF_CLOSURE_CPU_JOBS,
    PUBLICATION_CAMPAIGN_HANDOFF_CLOSURE_CPU_TIMEOUT_SECONDS,
    PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS,
)
from document_kv_cache.runtime_artifact_closure import (
    RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256,
    RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
    RUNTIME_ARTIFACT_CLOSURE_FILE_SIZE,
    VLLM_PATCHED_MANIFEST_SHA256,
    VLLM_PATCHED_MANIFEST_SIZE,
    VLLM_PATCHED_WHEEL_SIZE,
    VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT,
    VLLM_RUNTIME_BASE_LOCK_HASH_COUNT,
    VLLM_RUNTIME_BASE_LOCK_SHA256,
    VLLM_RUNTIME_BASE_LOCK_SIZE,
    VLLM_RUNTIME_SOURCE_LOCK_DISTRIBUTION_COUNT,
    VLLM_RUNTIME_SOURCE_LOCK_HASH_COUNT,
    VLLM_RUNTIME_SOURCE_LOCK_SIZE,
)
from document_kv_cache.serving_env import VLLM_RUNTIME_LOCK_SHA256
from document_kv_cache.storage import local_path


PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE: Final = "q8"
PUBLICATION_HANDOFF_CLOSURE_BF16_STAGE: Final = "bf16"
PUBLICATION_HANDOFF_CLOSURE_STAGES: Final = (
    PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE,
    PUBLICATION_HANDOFF_CLOSURE_BF16_STAGE,
)
if len(PUBLICATION_HANDOFF_CLOSURE_STAGES) != (
    PUBLICATION_CAMPAIGN_HANDOFF_CLOSURE_CPU_JOBS
):
    raise RuntimeError("campaign handoff coordinator job count drift")
PUBLICATION_HANDOFF_CLOSURE_NODE_TYPE_ID: Final = (
    PUBLICATION_CAMPAIGN_CPU_COORDINATOR_NODE_TYPE_ID
)
PUBLICATION_HANDOFF_CLOSURE_SPARK_VERSION: Final = (
    PUBLICATION_CAMPAIGN_CPU_COORDINATOR_SPARK_VERSION
)
PUBLICATION_HANDOFF_CLOSURE_TIMEOUT_SECONDS: Final = (
    PUBLICATION_CAMPAIGN_HANDOFF_CLOSURE_CPU_TIMEOUT_SECONDS
)
PUBLICATION_HANDOFF_CLOSURE_PARAMETER_BYTES_MAX: Final = 9_500
PUBLICATION_HANDOFF_CLOSURE_REQUEST_BYTES_MAX: Final = 16 * 1024 * 1024
PUBLICATION_HANDOFF_CLOSURE_RESULT_BYTES_MAX: Final = 16 * 1024 * 1024
_PUBLICATION_SOURCE_CLOSURE_V2_RECORD_TYPE: Final = (
    "cachet.publication_source_closure.v2"
)
_PUBLICATION_SOURCE_CLOSURE_V2_SCHEMA_VERSION: Final = 2
PUBLICATION_HANDOFF_CLOSURE_CONFIG_RECORD_TYPE: Final = (
    "cachet.publication_handoff_closure_coordinator_config.v2"
)
PUBLICATION_HANDOFF_CLOSURE_REQUEST_RECORD_TYPE: Final = (
    "cachet.publication_handoff_closure_request.v2"
)
PUBLICATION_HANDOFF_CLOSURE_RESULT_RECORD_TYPE: Final = (
    "cachet.publication_handoff_closure_result.v2"
)
PUBLICATION_HANDOFF_CLOSURE_RESERVATION_RECORD_TYPE: Final = (
    "cachet.publication_handoff_closure_reservation.v2"
)
PUBLICATION_HANDOFF_CLOSURE_SCHEMA_VERSION: Final = 2
PUBLICATION_HANDOFF_CLOSURE_RUNNER_FILENAME: Final = (
    "publication_handoff_closure_coordinator_runner.py"
)
PUBLICATION_HANDOFF_CLOSURE_RUNNER_SCRIPT: Final = (
    _q8.PUBLICATION_LATENCY_HANDOFF_RUNNER_SCRIPT.replace(
        "publication_latency_handoff_generation",
        "publication_handoff_closure_coordinator",
    )
    .replace("latency handoff", "handoff closure coordinator")
    .replace(
        "CACHET_LATENCY_HANDOFF_LOCKED_RUNTIME",
        "CACHET_HANDOFF_CLOSURE_LOCKED_RUNTIME",
    )
)
PUBLICATION_HANDOFF_CLOSURE_RUNNER_SHA256: Final = sha256(
    PUBLICATION_HANDOFF_CLOSURE_RUNNER_SCRIPT.encode("utf-8")
).hexdigest()

_ATTEMPT_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_SOURCE_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_REQUEST_AUTHORIZATION_ISSUER = object()
_AUTHORIZATION_ISSUER = object()


@dataclass(frozen=True, slots=True)
class PublicationHandoffClosureCoordinatorConfig:
    """Exact bootstrap and CPU cluster pins for one closure coordinator."""

    runner_python_file: str
    package_wheel_uri: str
    package_wheel_sha256: str
    runtime_lock_uri: str
    runtime_lock_sha256: str
    patched_vllm_wheel_uri: str
    patched_vllm_wheel_sha256: str
    patched_flashinfer_wheel_uri: str
    patched_flashinfer_wheel_sha256: str
    runtime_closure_manifest_uri: str
    runtime_closure_manifest_sha256: str
    source_closure_uri: str
    cachet_source_tree_sha256: str
    request_root_uri: str
    source_revision: str
    single_user_name: str
    runtime_venv_dir: str = "/local_disk0/cachet-handoff-closure-runtime"
    runner_sha256: str = PUBLICATION_HANDOFF_CLOSURE_RUNNER_SHA256
    node_type_id: str = PUBLICATION_HANDOFF_CLOSURE_NODE_TYPE_ID
    spark_version: str = PUBLICATION_HANDOFF_CLOSURE_SPARK_VERSION
    data_security_mode: str = "SINGLE_USER"
    timeout_seconds: int = PUBLICATION_HANDOFF_CLOSURE_TIMEOUT_SECONDS
    custom_tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "runner_python_file",
            "package_wheel_uri",
            "runtime_lock_uri",
            "patched_vllm_wheel_uri",
            "patched_flashinfer_wheel_uri",
            "runtime_closure_manifest_uri",
            "source_closure_uri",
        ):
            _canonical_volume_file_uri(getattr(self, field_name), field_name)
        _canonical_volume_directory_uri(self.request_root_uri, "request_root_uri")
        for field_name in (
            "package_wheel_sha256",
            "runtime_lock_sha256",
            "patched_vllm_wheel_sha256",
            "patched_flashinfer_wheel_sha256",
            "runtime_closure_manifest_sha256",
            "cachet_source_tree_sha256",
            "runner_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if self.runner_sha256 != PUBLICATION_HANDOFF_CLOSURE_RUNNER_SHA256:
            raise ValueError("coordinator runner SHA-256 differs from package source")
        expected_runtime_artifacts = {
            "runtime_lock_sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
            "patched_vllm_wheel_sha256": GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
            "patched_flashinfer_wheel_sha256": FLASHINFER_PATCHED_WHEEL_SHA256,
            "runtime_closure_manifest_sha256": (
                RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256
            ),
        }
        for field_name, expected in expected_runtime_artifacts.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"handoff coordinator {field_name} differs")
        if _SOURCE_REVISION_RE.fullmatch(self.source_revision) is None:
            raise ValueError("source_revision must be one full lowercase Git SHA")
        if self.node_type_id != PUBLICATION_HANDOFF_CLOSURE_NODE_TYPE_ID:
            raise ValueError("handoff closure coordinator must use c5d.4xlarge")
        if self.spark_version != PUBLICATION_HANDOFF_CLOSURE_SPARK_VERSION:
            raise ValueError(
                "handoff closure coordinator must use the frozen CPU ML DBR"
            )
        if self.data_security_mode != "SINGLE_USER" or not self.single_user_name:
            raise ValueError("coordinator requires one SINGLE_USER principal")
        runtime_root = Path(self.runtime_venv_dir)
        if (
            not runtime_root.is_absolute()
            or runtime_root.parts[:2] != ("/", "local_disk0")
            or ".." in runtime_root.parts
        ):
            raise ValueError("runtime_venv_dir must be confined beneath /local_disk0")
        if self.timeout_seconds != PUBLICATION_HANDOFF_CLOSURE_TIMEOUT_SECONDS:
            raise ValueError(
                "coordinator timeout must equal the frozen twelve-hour bound"
            )
        tags = dict(self.custom_tags)
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in tags.items()
        ):
            raise TypeError("custom_tags must contain only strings")
        object.__setattr__(self, "custom_tags", MappingProxyType(tags))

    def to_record(self) -> dict[str, Any]:
        return {
            "cachet_source_tree_sha256": self.cachet_source_tree_sha256,
            "custom_tags": dict(self.custom_tags),
            "data_security_mode": self.data_security_mode,
            "node_type_id": self.node_type_id,
            "package_wheel_sha256": self.package_wheel_sha256,
            "package_wheel_uri": self.package_wheel_uri,
            "patched_vllm_wheel_sha256": self.patched_vllm_wheel_sha256,
            "patched_vllm_wheel_uri": self.patched_vllm_wheel_uri,
            "patched_flashinfer_wheel_sha256": (
                self.patched_flashinfer_wheel_sha256
            ),
            "patched_flashinfer_wheel_uri": self.patched_flashinfer_wheel_uri,
            "record_type": PUBLICATION_HANDOFF_CLOSURE_CONFIG_RECORD_TYPE,
            "request_root_uri": self.request_root_uri,
            "runner_python_file": self.runner_python_file,
            "runner_sha256": self.runner_sha256,
            "runtime_lock_sha256": self.runtime_lock_sha256,
            "runtime_lock_uri": self.runtime_lock_uri,
            "runtime_closure_manifest_sha256": (
                self.runtime_closure_manifest_sha256
            ),
            "runtime_closure_manifest_uri": self.runtime_closure_manifest_uri,
            "runtime_venv_dir": self.runtime_venv_dir,
            "schema_version": PUBLICATION_HANDOFF_CLOSURE_SCHEMA_VERSION,
            "single_user_name": self.single_user_name,
            "source_closure_uri": self.source_closure_uri,
            "source_revision": self.source_revision,
            "spark_version": self.spark_version,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True, slots=True, init=False)
class PublicationHandoffClosureRequestAuthorization:
    """Issuer-only authority for one exact typed handoff closure batch.

    The canonical request remains serializable for the mounted CPU worker, but
    it is deliberately not itself launch authority.  Only the Q8/BF16 builders
    can join it to the typed atomic batch and worker evidence that produced it.
    """

    stage: str
    attempt_id: str
    request_closed_record_sha256: str
    request_file_sha256: str
    batch_evidence_sha256: str
    qualified_artifact_pins_sha256: str
    qualification_authorization_binding_sha256: str
    controller_lease_root: Path
    controller_lease_root_sha256: str
    authorization_sha256: str
    ledger_id: str
    ledger_path_sha256: str
    predecessor_prefix: DatabricksLedgerPrefix
    producer_batch_prefix: DatabricksLedgerPrefix
    ledger_prefix: DatabricksLedgerPrefix
    input_bundle_sha256: str
    qualification_closed_record_sha256: str
    _request_canonical_bytes: bytes

    def __init__(
        self,
        *,
        request: Mapping[str, Any],
        batch_evidence: Mapping[str, Any],
        qualified_artifact_pins: GPUQualificationArtifactPinsV2,
        qualification_authorization_binding: Mapping[str, Any],
        controller_lease_root: str | Path,
        _issuer: object,
    ) -> None:
        if _issuer is not _REQUEST_AUTHORIZATION_ISSUER:
            raise TypeError(
                "handoff closure request authority requires the typed batch issuer"
            )
        _validate_closure_request(request)
        if not isinstance(qualified_artifact_pins, GPUQualificationArtifactPinsV2):
            raise TypeError("qualified_artifact_pins must be native v2")
        pins = qualified_artifact_pins.to_record()
        request_pins = _q8.gpu_qualification_artifact_pins_v2_from_record(
            _required_mapping(request, "qualified_artifact_pins")
        ).to_record()
        if pins != request_pins:
            raise ValueError("closure request qualification artifact pins drift")
        coordinator = _required_mapping(request, "coordinator")
        expected_pins = {
            "cachet_source_tree_sha256": coordinator.get(
                "cachet_source_tree_sha256"
            ),
            "input_bundle_sha256": request.get("input_bundle_sha256"),
            "package_wheel_sha256": coordinator.get("package_wheel_sha256"),
            "patched_vllm_wheel_sha256": coordinator.get(
                "patched_vllm_wheel_sha256"
            ),
            "patched_flashinfer_wheel_sha256": coordinator.get(
                "patched_flashinfer_wheel_sha256"
            ),
            "runtime_closure_manifest_sha256": coordinator.get(
                "runtime_closure_manifest_sha256"
            ),
            "runtime_lock_sha256": coordinator.get("runtime_lock_sha256"),
        }
        if any(pins.get(name) != value for name, value in expected_pins.items()):
            raise ValueError(
                "handoff coordinator package/source pins differ from qualified "
                "producer artifacts"
            )
        lease_root = Path(controller_lease_root).expanduser().absolute()
        _require_no_symlink_ancestors(lease_root, include_leaf=True)
        lease_root_sha256 = _controller_path_sha256(
            lease_root,
            domain="cachet.publication.handoff_closure.controller_lease.v2",
        )
        singleton = _required_mapping(request, "controller_singleton")
        if singleton.get("controller_lease_root_sha256") != lease_root_sha256:
            raise ValueError("handoff closure controller lease root binding drift")
        qualification_binding = _canonical_json_object_from_bytes(
            _canonical_json_bytes(
                qualification_authorization_binding,
                pretty=True,
            ),
            "qualification launch authorization binding",
        )
        _validate_qualification_authorization_binding(
            qualification_binding,
            expected_artifact_pins_sha256=_canonical_sha256(pins),
            expected_qualification_closed_record_sha256=_required_string(
                request, "expected_qualification_closed_record_sha256"
            ),
        )
        evidence = _canonical_json_object_from_bytes(
            _canonical_json_bytes(batch_evidence, pretty=True),
            "handoff closure typed batch evidence",
        )
        lineage = _required_mapping(request, "ledger_lineage")
        batch = _required_mapping(evidence, "batch_authorization")
        worker_evidence = [
            dict(_mapping(item, "typed worker evidence"))
            for item in _required_sequence(evidence, "worker_evidence")
        ]
        request_worker_evidence = [
            dict(_mapping(item, "request worker evidence"))
            for item in _required_sequence(request, "worker_evidence")
        ]
        if (
            worker_evidence != request_worker_evidence
            or _required_mapping(evidence, "phase_evidence")
            != _required_mapping(singleton, "phase_evidence")
            or singleton.get("batch_identity_sha256")
            != _closure_batch_identity_sha256(batch)
            or batch.get("ledger_path_sha256")
            != lineage.get("ledger_path_sha256")
            or batch.get("predecessor_prefix")
            != lineage.get("predecessor_prefix")
            or batch.get("batch_prefix")
            != lineage.get("producer_batch_prefix")
            or batch.get("attempt_ids")
            != [item["attempt_id"] for item in request_worker_evidence]
        ):
            raise ValueError("typed batch evidence differs from closure request")
        request_bytes = _canonical_json_bytes(request, pretty=True)
        request_file_sha256 = sha256(request_bytes).hexdigest()
        batch_evidence_sha256 = _canonical_sha256(evidence)
        qualified_artifact_pins_sha256 = _canonical_sha256(pins)
        qualification_authorization_binding_sha256 = _canonical_sha256(
            qualification_binding
        )
        stage = _required_stage(request.get("stage"))
        request_closed_record_sha256 = _required_string(
            request, "closed_record_sha256"
        )
        authorization_sha256 = _canonical_sha256(
            {
                "batch_evidence_sha256": batch_evidence_sha256,
                "domain": "cachet.publication.handoff_closure.request_authority.v2",
                "qualified_artifact_pins_sha256": (
                    qualified_artifact_pins_sha256
                ),
                "qualification_authorization_binding_sha256": (
                    qualification_authorization_binding_sha256
                ),
                "controller_lease_root_sha256": lease_root_sha256,
                "request_closed_record_sha256": request_closed_record_sha256,
                "request_file_sha256": request_file_sha256,
                "stage": stage,
            }
        )
        object.__setattr__(self, "stage", stage)
        object.__setattr__(
            self, "attempt_id", _required_string(request, "attempt_id")
        )
        object.__setattr__(
            self, "request_closed_record_sha256", request_closed_record_sha256
        )
        object.__setattr__(self, "request_file_sha256", request_file_sha256)
        object.__setattr__(
            self, "batch_evidence_sha256", batch_evidence_sha256
        )
        object.__setattr__(
            self,
            "qualified_artifact_pins_sha256",
            qualified_artifact_pins_sha256,
        )
        object.__setattr__(
            self,
            "qualification_authorization_binding_sha256",
            qualification_authorization_binding_sha256,
        )
        object.__setattr__(self, "controller_lease_root", lease_root)
        object.__setattr__(
            self, "controller_lease_root_sha256", lease_root_sha256
        )
        object.__setattr__(self, "authorization_sha256", authorization_sha256)
        object.__setattr__(self, "ledger_id", _required_string(lineage, "ledger_id"))
        object.__setattr__(
            self,
            "ledger_path_sha256",
            _required_string(lineage, "ledger_path_sha256"),
        )
        object.__setattr__(
            self,
            "predecessor_prefix",
            databricks_ledger_prefix_from_record(
                _required_mapping(lineage, "predecessor_prefix")
            ),
        )
        object.__setattr__(
            self,
            "producer_batch_prefix",
            databricks_ledger_prefix_from_record(
                _required_mapping(lineage, "producer_batch_prefix")
            ),
        )
        object.__setattr__(
            self,
            "ledger_prefix",
            databricks_ledger_prefix_from_record(
                _required_mapping(lineage, "terminal_prefix")
            ),
        )
        object.__setattr__(
            self,
            "input_bundle_sha256",
            _required_string(request, "input_bundle_sha256"),
        )
        object.__setattr__(
            self,
            "qualification_closed_record_sha256",
            _required_string(
                request, "expected_qualification_closed_record_sha256"
            ),
        )
        object.__setattr__(self, "_request_canonical_bytes", request_bytes)

    @property
    def request_record(self) -> Mapping[str, Any]:
        """Return a fresh immutable top-level view of the authorized request."""

        return MappingProxyType(
            _canonical_json_object_from_bytes(
                self._request_canonical_bytes,
                "authorized handoff closure request",
            )
        )


@dataclass(frozen=True, slots=True, init=False)
class PublicationHandoffRemoteClosureAuthorization:
    """Issuer-only capability for compact, remotely validated handoff trees."""

    stage: str
    request_closed_record_sha256: str
    result_uri: str
    result_file_sha256: str
    result_closed_record_sha256: str
    output_root_uri: str
    execution_uri: str
    execution_file_sha256: str
    execution_closed_record_sha256: str
    coordinator_run_id: str
    control_plane_status_sha256: str
    ledger_id: str
    ledger_path_sha256: str
    predecessor_prefix: DatabricksLedgerPrefix
    producer_batch_prefix: DatabricksLedgerPrefix
    ledger_prefix: DatabricksLedgerPrefix
    causal_closure_sha256: str
    request_record: Mapping[str, Any]
    result_record: Mapping[str, Any]
    execution_record: Mapping[str, Any]
    manifest_records: tuple[Mapping[str, Any], ...]

    def __init__(
        self,
        *,
        request: Mapping[str, Any],
        result: Mapping[str, Any],
        result_file_sha256: str,
        coordinator_run_id: str,
        control_plane_status_sha256: str,
        _issuer: object,
    ) -> None:
        if _issuer is not _AUTHORIZATION_ISSUER:
            raise TypeError("remote closure authority requires the collector issuer")
        _validate_closure_request(request)
        _validate_closure_result(result, request=request)
        normalized_run_id = _required_run_id(coordinator_run_id, "coordinator_run_id")
        normalized_result_file_sha256 = _require_sha256(
            result_file_sha256, "result_file_sha256"
        )
        normalized_control_sha256 = _require_sha256(
            control_plane_status_sha256, "control_plane_status_sha256"
        )
        if (
            _required_string(_required_mapping(result, "coordinator"), "run_id")
            != normalized_run_id
            or normalized_result_file_sha256
            != sha256(_canonical_json_bytes(result, pretty=True)).hexdigest()
        ):
            raise ValueError("remote closure issuer inputs drift from result")
        stage = _required_stage(request.get("stage"))
        lineage = _required_mapping(request, "ledger_lineage")
        predecessor = databricks_ledger_prefix_from_record(
            _required_mapping(lineage, "predecessor_prefix")
        )
        producer = databricks_ledger_prefix_from_record(
            _required_mapping(lineage, "producer_batch_prefix")
        )
        terminal = databricks_ledger_prefix_from_record(
            _required_mapping(lineage, "terminal_prefix")
        )
        execution = _required_mapping(result, "execution")
        manifests = tuple(
            MappingProxyType(dict(_mapping(item, "manifest binding")["record"]))
            for item in _required_sequence(result, "manifests")
        )
        causal = _canonical_sha256(
            {
                "control_plane_status_sha256": normalized_control_sha256,
                "coordinator_run_id": normalized_run_id,
                "ledger_lineage": dict(lineage),
                "request_closed_record_sha256": request["closed_record_sha256"],
                "result_closed_record_sha256": result["closed_record_sha256"],
                "result_file_sha256": normalized_result_file_sha256,
            }
        )
        object.__setattr__(self, "stage", stage)
        object.__setattr__(
            self, "request_closed_record_sha256", request["closed_record_sha256"]
        )
        object.__setattr__(self, "result_uri", _required_string(request, "result_uri"))
        object.__setattr__(
            self,
            "result_file_sha256",
            normalized_result_file_sha256,
        )
        object.__setattr__(
            self, "result_closed_record_sha256", result["closed_record_sha256"]
        )
        object.__setattr__(
            self, "output_root_uri", _required_string(request, "output_root_uri")
        )
        object.__setattr__(self, "execution_uri", _required_string(execution, "uri"))
        object.__setattr__(
            self, "execution_file_sha256", _required_string(execution, "file_sha256")
        )
        object.__setattr__(
            self,
            "execution_closed_record_sha256",
            _required_string(execution, "closed_record_sha256"),
        )
        object.__setattr__(self, "coordinator_run_id", normalized_run_id)
        object.__setattr__(
            self,
            "control_plane_status_sha256",
            normalized_control_sha256,
        )
        object.__setattr__(self, "ledger_id", _required_string(lineage, "ledger_id"))
        object.__setattr__(
            self,
            "ledger_path_sha256",
            _required_string(lineage, "ledger_path_sha256"),
        )
        object.__setattr__(self, "predecessor_prefix", predecessor)
        object.__setattr__(self, "producer_batch_prefix", producer)
        object.__setattr__(self, "ledger_prefix", terminal)
        object.__setattr__(self, "causal_closure_sha256", causal)
        object.__setattr__(self, "request_record", MappingProxyType(dict(request)))
        object.__setattr__(self, "result_record", MappingProxyType(dict(result)))
        object.__setattr__(
            self,
            "execution_record",
            MappingProxyType(dict(_required_mapping(execution, "record"))),
        )
        object.__setattr__(self, "manifest_records", manifests)


def build_q8_handoff_closure_request(
    *,
    attempt_id: str | None = None,
    coordinator_config: PublicationHandoffClosureCoordinatorConfig,
    plan_uri: str,
    plan_file_sha256: str,
    plan_record: Mapping[str, Any],
    prepared_input_root_uri: str,
    durable_output_root_uri: str,
    execution_contract: Mapping[str, Any],
    ledger_path: str | Path,
    attempt_ids_by_worker: Mapping[int, str],
    attestations_by_worker: Mapping[
        int, _q8.PublicationLatencyHandoffDatabricksAttestationBinding
    ],
    submission_authorization: _q8.PublicationLatencyHandoffSubmissionAuthorization,
    hardware_qualification: _q8.PublicationLatencyGeneratorHardwareQualificationV2,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    expected_qualification_closed_record_sha256: str,
) -> PublicationHandoffClosureRequestAuthorization:
    """Build a CPU closure request from the exact terminal Q8 worker batch."""

    _q8._validate_closed_plan_envelope(plan_record)
    qualification_binding = _require_matching_qualified_producer(
        coordinator_config,
        hardware_qualification,
        qualification_launch_authorization,
        expected_input_bundle_sha256=_required_string(
            plan_record, "input_bundle_sha256"
        ),
        expected_qualification_closed_record_sha256=(
            expected_qualification_closed_record_sha256
        ),
    )
    batch = _q8.require_publication_latency_handoff_submission_authorization(
        submission_authorization
    )
    output_root_uri = _require_submission_output_root(
        submission_authorization.durable_output_root,
        durable_output_root_uri,
        stage=PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE,
    )
    controller_lease_root = _handoff_closure_controller_lease_root(
        submission_authorization.phase_lease_root,
        stage=PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE,
    )
    singleton = _handoff_closure_singleton(
        stage=PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE,
        submission_authorization=submission_authorization,
        batch_authorization=batch,
        output_root_uri=output_root_uri,
        controller_lease_root=controller_lease_root,
    )
    evidence: list[dict[str, Any]] = []
    expected_workers = set(range(PUBLICATION_CAMPAIGN_LATENCY_HANDOFF_PRODUCER_TASKS))
    if (
        set(attempt_ids_by_worker) != expected_workers
        or set(attestations_by_worker) != expected_workers
    ):
        raise ValueError("Q8 closure request requires workers 0..15 exactly")
    for worker_index in sorted(expected_workers):
        binding = attestations_by_worker[worker_index]
        if not isinstance(
            binding, _q8.PublicationLatencyHandoffDatabricksAttestationBinding
        ):
            raise TypeError("Q8 closure requires attestation bindings")
        _require_handoff_attestation_path(
            binding.path,
            output_root_uri,
            directory=_q8.PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_DIRECTORY,
            worker_index=worker_index,
            stage=PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE,
        )
        evidence.append(
            {
                "attempt_id": _nonempty(
                    attempt_ids_by_worker[worker_index], "attempt_id"
                ),
                "attestation_closed_record_sha256": binding.closed_record_sha256,
                "attestation_file_sha256": binding.file_sha256,
                "worker_index": worker_index,
            }
        )
    request = _build_closure_request(
        stage=PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE,
        attempt_id=attempt_id,
        coordinator_config=coordinator_config,
        plan_uri=plan_uri,
        plan_file_sha256=plan_file_sha256,
        plan_closed_record_sha256=_required_string(plan_record, "closed_record_sha256"),
        input_bundle_sha256=_required_string(plan_record, "input_bundle_sha256"),
        qualified_artifact_pins=hardware_qualification.expected_artifact_pins,
        prepared_input_root_uri=prepared_input_root_uri,
        durable_output_root_uri=durable_output_root_uri,
        controller_singleton=singleton,
        execution_contract=execution_contract,
        ledger_path=ledger_path,
        batch_authorization=batch,
        worker_evidence=evidence,
        expected_qualification_closed_record_sha256=(
            expected_qualification_closed_record_sha256
        ),
    )
    return _issue_closure_request_authorization(
        request,
        submission_authorization=submission_authorization,
        batch_authorization=batch,
        hardware_qualification=hardware_qualification,
        qualification_authorization_binding=qualification_binding,
        controller_lease_root=controller_lease_root,
        worker_evidence=evidence,
    )


def build_bf16_handoff_closure_request(
    *,
    attempt_id: str | None = None,
    coordinator_config: PublicationHandoffClosureCoordinatorConfig,
    plan_uri: str,
    plan_file_sha256: str,
    plan_record: Mapping[str, Any],
    prepared_input_root_uri: str,
    durable_output_root_uri: str,
    execution_contract: Mapping[str, Any],
    ledger_path: str | Path,
    attempt_ids_by_worker: Mapping[int, str],
    worker_authorizations: Mapping[
        int, _bf16.PublicationBF16HandoffWorkerAuthorization
    ],
    submission_authorization: _bf16.PublicationBF16HandoffSubmissionAuthorization,
    hardware_qualification: _q8.PublicationLatencyGeneratorHardwareQualificationV2,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    expected_qualification_closed_record_sha256: str,
) -> PublicationHandoffClosureRequestAuthorization:
    """Build a CPU closure request from issuer-bound BF16 worker evidence."""

    _bf16._validate_plan_envelope(plan_record)
    qualification_binding = _require_matching_qualified_producer(
        coordinator_config,
        hardware_qualification,
        qualification_launch_authorization,
        expected_input_bundle_sha256=_required_string(
            plan_record, "input_bundle_sha256"
        ),
        expected_qualification_closed_record_sha256=(
            expected_qualification_closed_record_sha256
        ),
    )
    batch = _bf16.require_publication_bf16_handoff_submission_authorization(
        submission_authorization
    )
    output_root_uri = _require_submission_output_root(
        submission_authorization.durable_output_root,
        durable_output_root_uri,
        stage=PUBLICATION_HANDOFF_CLOSURE_BF16_STAGE,
    )
    controller_lease_root = _handoff_closure_controller_lease_root(
        submission_authorization.phase_lease_root,
        stage=PUBLICATION_HANDOFF_CLOSURE_BF16_STAGE,
    )
    singleton = _handoff_closure_singleton(
        stage=PUBLICATION_HANDOFF_CLOSURE_BF16_STAGE,
        submission_authorization=submission_authorization,
        batch_authorization=batch,
        output_root_uri=output_root_uri,
        controller_lease_root=controller_lease_root,
    )
    expected_workers = set(range(_bf16.PUBLICATION_BF16_HANDOFF_WORKER_COUNT))
    if (
        set(attempt_ids_by_worker) != expected_workers
        or set(worker_authorizations) != expected_workers
    ):
        raise ValueError("BF16 closure request requires workers 0..15 exactly")
    evidence: list[dict[str, Any]] = []
    for worker_index in sorted(expected_workers):
        authority = worker_authorizations[worker_index]
        if not isinstance(authority, _bf16.PublicationBF16HandoffWorkerAuthorization):
            raise TypeError("BF16 closure requires live worker authorizations")
        _require_handoff_attestation_path(
            authority.binding.path,
            output_root_uri,
            directory=_bf16.PUBLICATION_BF16_HANDOFF_ATTESTATION_DIRECTORY,
            worker_index=worker_index,
            stage=PUBLICATION_HANDOFF_CLOSURE_BF16_STAGE,
        )
        if authority.attempt_id != attempt_ids_by_worker[worker_index]:
            raise ValueError("BF16 worker authorization attempt mapping drift")
        if authority.producer_batch_prefix != batch.batch_prefix:
            raise ValueError("BF16 workers bind a different producer batch")
        evidence.append(
            {
                "attempt_id": authority.attempt_id,
                "attestation_closed_record_sha256": authority.binding.closed_record_sha256,
                "attestation_file_sha256": authority.binding.file_sha256,
                "control_plane_status_sha256": authority.control_plane_status_sha256,
                "worker_index": worker_index,
            }
        )
    request = _build_closure_request(
        stage=PUBLICATION_HANDOFF_CLOSURE_BF16_STAGE,
        attempt_id=attempt_id,
        coordinator_config=coordinator_config,
        plan_uri=plan_uri,
        plan_file_sha256=plan_file_sha256,
        plan_closed_record_sha256=_required_string(plan_record, "closed_record_sha256"),
        input_bundle_sha256=_required_string(plan_record, "input_bundle_sha256"),
        qualified_artifact_pins=hardware_qualification.expected_artifact_pins,
        prepared_input_root_uri=prepared_input_root_uri,
        durable_output_root_uri=durable_output_root_uri,
        controller_singleton=singleton,
        execution_contract=execution_contract,
        ledger_path=ledger_path,
        batch_authorization=batch,
        worker_evidence=evidence,
        expected_qualification_closed_record_sha256=(
            expected_qualification_closed_record_sha256
        ),
    )
    return _issue_closure_request_authorization(
        request,
        submission_authorization=submission_authorization,
        batch_authorization=batch,
        hardware_qualification=hardware_qualification,
        qualification_authorization_binding=qualification_binding,
        controller_lease_root=controller_lease_root,
        worker_evidence=evidence,
    )


def _issue_closure_request_authorization(
    request: Mapping[str, Any],
    *,
    submission_authorization: object,
    batch_authorization: Any,
    hardware_qualification: _q8.PublicationLatencyGeneratorHardwareQualificationV2,
    qualification_authorization_binding: Mapping[str, Any],
    controller_lease_root: Path,
    worker_evidence: Sequence[Mapping[str, Any]],
) -> PublicationHandoffClosureRequestAuthorization:
    phase_evidence = _submission_phase_evidence(submission_authorization)
    batch_evidence = {
        "batch_authorization": {
            "attempt_ids": list(batch_authorization.attempt_ids),
            "batch_prefix": batch_authorization.batch_prefix.to_record(),
            "ledger_path_sha256": batch_authorization.ledger_path_sha256,
            "predecessor_prefix": (
                batch_authorization.predecessor_prefix.to_record()
            ),
            "submit_payload_sha256s": list(
                batch_authorization.submit_payload_sha256s
            ),
        },
        "phase_evidence": phase_evidence,
        "worker_evidence": [dict(item) for item in worker_evidence],
    }
    return PublicationHandoffClosureRequestAuthorization(
        request=request,
        batch_evidence=batch_evidence,
        qualified_artifact_pins=hardware_qualification.expected_artifact_pins,
        qualification_authorization_binding=qualification_authorization_binding,
        controller_lease_root=controller_lease_root,
        _issuer=_REQUEST_AUTHORIZATION_ISSUER,
    )


def _submission_phase_evidence(value: object) -> dict[str, str]:
    evidence = {
        name: getattr(value, name)
        for name in (
            "batch_marker_closed_record_sha256",
            "batch_marker_file_sha256",
            "phase_lease_closed_record_sha256",
            "phase_lease_file_sha256",
            "phase_lease_root_sha256",
        )
    }
    for name, digest in evidence.items():
        _require_sha256(digest, name)
    return evidence


def _handoff_closure_singleton(
    *,
    stage: str,
    submission_authorization: object,
    batch_authorization: Any,
    output_root_uri: str,
    controller_lease_root: Path,
) -> dict[str, Any]:
    normalized_stage = _required_stage(stage)
    normalized_output = _canonical_volume_directory_uri(
        output_root_uri, "durable_output_root_uri"
    )
    phase_evidence = _submission_phase_evidence(submission_authorization)
    batch_identity_sha256 = _closure_batch_identity_sha256(
        {
            "attempt_ids": list(batch_authorization.attempt_ids),
            "batch_prefix": batch_authorization.batch_prefix.to_record(),
            "ledger_path_sha256": batch_authorization.ledger_path_sha256,
            "predecessor_prefix": batch_authorization.predecessor_prefix.to_record(),
            "submit_payload_sha256s": list(
                batch_authorization.submit_payload_sha256s
            ),
        }
    )
    lease_sha256 = _controller_path_sha256(
        controller_lease_root,
        domain="cachet.publication.handoff_closure.controller_lease.v2",
    )
    identity = {
        "batch_identity_sha256": batch_identity_sha256,
        "controller_lease_root_sha256": lease_sha256,
        "durable_output_root_uri": normalized_output,
        "phase_evidence": phase_evidence,
        "stage": normalized_stage,
    }
    singleton = {
        **identity,
        "identity_sha256": _canonical_sha256(
            {
                "domain": "cachet.publication.handoff_closure.singleton.v2",
                **identity,
            }
        ),
    }
    _validate_handoff_closure_singleton(
        singleton,
        expected_stage=normalized_stage,
        expected_output_root_uri=normalized_output,
    )
    return singleton


def _closure_batch_identity_sha256(batch: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {
            "attempt_ids": list(_required_sequence(batch, "attempt_ids")),
            "batch_prefix": dict(_required_mapping(batch, "batch_prefix")),
            "ledger_path_sha256": _require_sha256(
                _required_string(batch, "ledger_path_sha256"),
                "batch ledger_path_sha256",
            ),
            "predecessor_prefix": dict(
                _required_mapping(batch, "predecessor_prefix")
            ),
            "submit_payload_sha256s": list(
                _required_sequence(batch, "submit_payload_sha256s")
            ),
        }
    )


def _validate_handoff_closure_singleton(
    singleton: Mapping[str, Any],
    *,
    expected_stage: str,
    expected_output_root_uri: str,
) -> None:
    if set(singleton) != {
        "batch_identity_sha256",
        "controller_lease_root_sha256",
        "durable_output_root_uri",
        "identity_sha256",
        "phase_evidence",
        "stage",
    }:
        raise ValueError("handoff closure singleton keys drift")
    stage = _required_stage(singleton.get("stage"))
    output_root = _canonical_volume_directory_uri(
        _required_string(singleton, "durable_output_root_uri"),
        "singleton durable_output_root_uri",
    )
    if stage != _required_stage(expected_stage) or output_root != (
        _canonical_volume_directory_uri(
            expected_output_root_uri, "expected durable_output_root_uri"
        )
    ):
        raise ValueError("handoff closure singleton stage/output binding drift")
    phase_evidence = _required_mapping(singleton, "phase_evidence")
    if set(phase_evidence) != {
        "batch_marker_closed_record_sha256",
        "batch_marker_file_sha256",
        "phase_lease_closed_record_sha256",
        "phase_lease_file_sha256",
        "phase_lease_root_sha256",
    }:
        raise ValueError("handoff closure singleton phase evidence keys drift")
    for name in (
        "batch_identity_sha256",
        "controller_lease_root_sha256",
        "identity_sha256",
    ):
        _require_sha256(_required_string(singleton, name), name)
    for name, digest in phase_evidence.items():
        _require_sha256(_nonempty(digest, name), name)
    identity = {
        "batch_identity_sha256": singleton["batch_identity_sha256"],
        "controller_lease_root_sha256": singleton[
            "controller_lease_root_sha256"
        ],
        "durable_output_root_uri": output_root,
        "phase_evidence": dict(phase_evidence),
        "stage": stage,
    }
    expected_identity = _canonical_sha256(
        {
            "domain": "cachet.publication.handoff_closure.singleton.v2",
            **identity,
        }
    )
    if singleton.get("identity_sha256") != expected_identity:
        raise ValueError("handoff closure singleton identity digest drift")


def _handoff_closure_attempt_id(singleton: Mapping[str, Any]) -> str:
    stage = _required_stage(singleton.get("stage"))
    identity = _require_sha256(
        _required_string(singleton, "identity_sha256"), "identity_sha256"
    )
    return f"{stage}-handoff-closure-{identity[:24]}"


def _require_submission_output_root(
    authorized_output_root: str,
    requested_output_root: str,
    *,
    stage: str,
) -> str:
    authorized = _canonical_volume_directory_uri(
        authorized_output_root, f"{stage} authorized durable_output_root"
    )
    requested = _canonical_volume_directory_uri(
        requested_output_root, f"{stage} durable_output_root_uri"
    )
    if requested != authorized:
        raise ValueError(
            f"{stage} closure output root differs from the producer phase authority"
        )
    return authorized


def _require_handoff_attestation_path(
    path: str | Path,
    output_root_uri: str,
    *,
    directory: str,
    worker_index: int,
    stage: str,
) -> None:
    output_root = Path(
        local_path(
            _canonical_volume_directory_uri(
                output_root_uri, f"{stage} durable_output_root_uri"
            )
        )
    ).expanduser().absolute()
    expected = output_root / directory / f"worker-{worker_index:02d}.json"
    observed = Path(path).expanduser().absolute()
    if observed != expected:
        raise ValueError(
            f"{stage} attestation path differs from the producer phase output root"
        )


def _handoff_closure_request_root_uri(output_root_uri: str, *, stage: str) -> str:
    output = _canonical_volume_directory_uri(output_root_uri, "output_root_uri")
    normalized_stage = _required_stage(stage)
    return _canonical_volume_directory_uri(
        f"{output}-{normalized_stage}-cpu-closure-control",
        "handoff closure request_root_uri",
    )


def publication_handoff_closure_request_root_uri(
    output_root_uri: str, *, stage: str
) -> str:
    """Return the sole durable request root for one bound handoff output."""

    return _handoff_closure_request_root_uri(output_root_uri, stage=stage)


def _handoff_closure_result_uri(output_root_uri: str, *, stage: str) -> str:
    output = _canonical_volume_directory_uri(output_root_uri, "output_root_uri")
    return _join_volume_uri(
        output,
        f"coordinator-results/{_required_stage(stage)}/result.json",
    )


def _handoff_closure_controller_lease_root(
    phase_lease_root: str | Path,
    *,
    stage: str,
) -> Path:
    phase_root = Path(phase_lease_root).expanduser().absolute()
    _require_no_symlink_ancestors(phase_root, include_leaf=True)
    root = phase_root.parent / f"{phase_root.name}-{_required_stage(stage)}-closure"
    _require_no_symlink_ancestors(root, include_leaf=True)
    return root


def _controller_path_sha256(path: str | Path, *, domain: str) -> str:
    root = Path(path).expanduser().absolute()
    _require_no_symlink_ancestors(root, include_leaf=True)
    return _canonical_sha256(
        {"domain": _nonempty(domain, "controller path domain"), "path": str(root)}
    )


def _require_matching_qualified_producer(
    coordinator_config: PublicationHandoffClosureCoordinatorConfig,
    hardware_qualification: _q8.PublicationLatencyGeneratorHardwareQualificationV2,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    *,
    expected_input_bundle_sha256: str,
    expected_qualification_closed_record_sha256: str,
) -> dict[str, Any]:
    if not isinstance(
        hardware_qualification,
        _q8.PublicationLatencyGeneratorHardwareQualificationV2,
    ):
        raise TypeError(
            "handoff closure requires "
            "PublicationLatencyGeneratorHardwareQualificationV2"
        )
    qualification_record = (
        _q8.publication_latency_generator_hardware_qualification_v2_record(
            hardware_qualification
        )
    )
    _q8.validate_publication_latency_generator_hardware_qualification_v2_record(
        qualification_record
    )
    expected_qualification = _require_sha256(
        expected_qualification_closed_record_sha256,
        "expected_qualification_closed_record_sha256",
    )
    if hardware_qualification.evidence_record.get(
        "closed_record_sha256"
    ) != expected_qualification:
        raise ValueError("handoff closure qualification evidence binding drift")
    selection = require_gpu_qualification_launch_authorization(
        qualification_launch_authorization,
        expected_plan_sha256=hardware_qualification.selection.plan_sha256,
        expected_evidence_file_sha256=hardware_qualification.evidence_file_sha256,
    )
    if (
        selection != hardware_qualification.selection
        or qualification_launch_authorization.evidence_closed_record_sha256
        != expected_qualification
    ):
        raise ValueError(
            "handoff closure qualification launch authority selection/evidence drift"
        )
    pins = hardware_qualification.expected_artifact_pins
    expected = {
        "cachet_source_tree_sha256": coordinator_config.cachet_source_tree_sha256,
        "input_bundle_sha256": _require_sha256(
            expected_input_bundle_sha256, "expected_input_bundle_sha256"
        ),
        "package_wheel_sha256": coordinator_config.package_wheel_sha256,
        "patched_vllm_wheel_sha256": (
            coordinator_config.patched_vllm_wheel_sha256
        ),
        "patched_flashinfer_wheel_sha256": (
            coordinator_config.patched_flashinfer_wheel_sha256
        ),
        "runtime_closure_manifest_sha256": (
            coordinator_config.runtime_closure_manifest_sha256
        ),
        "runtime_lock_sha256": coordinator_config.runtime_lock_sha256,
    }
    if any(pins.to_record().get(name) != value for name, value in expected.items()):
        raise ValueError(
            "handoff coordinator package/source pins differ from qualified "
            "producer artifacts"
        )
    if pins.runner_sha256 == coordinator_config.runner_sha256:
        raise ValueError(
            "qualification runner must remain distinct from the closure runner"
        )
    binding = {
        "artifact_pins_sha256": _canonical_sha256(pins.to_record()),
        "authorization_causal_closure_sha256": (
            qualification_launch_authorization.causal_closure_sha256
        ),
        "authorization_ledger_id": qualification_launch_authorization.ledger_id,
        "authorization_ledger_path_sha256": (
            qualification_launch_authorization.ledger_path_sha256
        ),
        "authorization_ledger_prefix": (
            qualification_launch_authorization.ledger_prefix.to_record()
        ),
        "evidence_closed_record_sha256": expected_qualification,
        "evidence_file_sha256": hardware_qualification.evidence_file_sha256,
        "evidence_uri": hardware_qualification.evidence_uri,
        "plan_closed_record_sha256": hardware_qualification.selection.plan_sha256,
        "plan_file_sha256": hardware_qualification.plan_file_sha256,
        "plan_uri": hardware_qualification.plan_uri,
        "selection": {
            "attention_backend": selection.attention_backend,
            "generation_artifacts_sha256": selection.generation_artifacts_sha256,
            "generation_databricks_node_type_id": (
                selection.generation_databricks_node_type_id
            ),
            "generation_hardware_id": selection.generation_hardware_id,
            "generation_prefix_tokens_per_second": (
                selection.generation_prefix_tokens_per_second
            ),
            "gpu_memory_utilization": selection.gpu_memory_utilization,
            "plan_sha256": selection.plan_sha256,
        },
    }
    _validate_qualification_authorization_binding(
        binding,
        expected_artifact_pins_sha256=_canonical_sha256(pins.to_record()),
        expected_qualification_closed_record_sha256=expected_qualification,
    )
    return binding


def _validate_qualification_authorization_binding(
    binding: Mapping[str, Any],
    *,
    expected_artifact_pins_sha256: str,
    expected_qualification_closed_record_sha256: str,
) -> None:
    expected_keys = {
        "artifact_pins_sha256",
        "authorization_causal_closure_sha256",
        "authorization_ledger_id",
        "authorization_ledger_path_sha256",
        "authorization_ledger_prefix",
        "evidence_closed_record_sha256",
        "evidence_file_sha256",
        "evidence_uri",
        "plan_closed_record_sha256",
        "plan_file_sha256",
        "plan_uri",
        "selection",
    }
    if set(binding) != expected_keys:
        raise ValueError("qualification launch authorization binding keys drift")
    for name in (
        "artifact_pins_sha256",
        "authorization_causal_closure_sha256",
        "authorization_ledger_path_sha256",
        "evidence_closed_record_sha256",
        "evidence_file_sha256",
        "plan_closed_record_sha256",
        "plan_file_sha256",
    ):
        _require_sha256(_required_string(binding, name), name)
    if (
        binding.get("artifact_pins_sha256")
        != _require_sha256(
            expected_artifact_pins_sha256,
            "expected_artifact_pins_sha256",
        )
        or binding.get("evidence_closed_record_sha256")
        != _require_sha256(
            expected_qualification_closed_record_sha256,
            "expected_qualification_closed_record_sha256",
        )
    ):
        raise ValueError("qualification launch authorization identity drift")
    ledger_id = _nonempty(
        _required_string(binding, "authorization_ledger_id"), "ledger_id"
    )
    _nonempty(_required_string(binding, "evidence_uri"), "evidence_uri")
    _nonempty(_required_string(binding, "plan_uri"), "plan_uri")
    ledger_prefix = databricks_ledger_prefix_from_record(
        _required_mapping(binding, "authorization_ledger_prefix")
    )
    if ledger_prefix.ledger_id != ledger_id:
        raise ValueError("qualification authorization ledger prefix identity drift")
    selection = _required_mapping(binding, "selection")
    if set(selection) != {
        "attention_backend",
        "generation_artifacts_sha256",
        "generation_databricks_node_type_id",
        "generation_hardware_id",
        "generation_prefix_tokens_per_second",
        "gpu_memory_utilization",
        "plan_sha256",
    }:
        raise ValueError("qualification launch selection binding keys drift")
    for name in ("generation_artifacts_sha256", "plan_sha256"):
        _require_sha256(_required_string(selection, name), name)
    if selection.get("plan_sha256") != binding.get("plan_closed_record_sha256"):
        raise ValueError("qualification authorization selection plan identity drift")


def require_publication_handoff_closure_request_authorization(
    authorization: object,
) -> PublicationHandoffClosureRequestAuthorization:
    """Replay the exact canonical request behind issuer-only batch authority."""

    if not isinstance(
        authorization, PublicationHandoffClosureRequestAuthorization
    ):
        raise TypeError(
            "handoff closure launch requires "
            "PublicationHandoffClosureRequestAuthorization"
        )
    request = dict(authorization.request_record)
    _validate_closure_request(request)
    request_bytes = _canonical_json_bytes(request, pretty=True)
    expected_authorization_sha256 = _canonical_sha256(
        {
            "batch_evidence_sha256": authorization.batch_evidence_sha256,
            "domain": "cachet.publication.handoff_closure.request_authority.v2",
            "qualified_artifact_pins_sha256": (
                authorization.qualified_artifact_pins_sha256
            ),
            "qualification_authorization_binding_sha256": (
                authorization.qualification_authorization_binding_sha256
            ),
            "controller_lease_root_sha256": (
                authorization.controller_lease_root_sha256
            ),
            "request_closed_record_sha256": request["closed_record_sha256"],
            "request_file_sha256": sha256(request_bytes).hexdigest(),
            "stage": request["stage"],
        }
    )
    lineage = _required_mapping(request, "ledger_lineage")
    if (
        authorization.stage != request.get("stage")
        or authorization.attempt_id != request.get("attempt_id")
        or authorization.request_closed_record_sha256
        != request.get("closed_record_sha256")
        or authorization.request_file_sha256 != sha256(request_bytes).hexdigest()
        or authorization.authorization_sha256 != expected_authorization_sha256
        or authorization.controller_lease_root_sha256
        != _controller_path_sha256(
            authorization.controller_lease_root,
            domain="cachet.publication.handoff_closure.controller_lease.v2",
        )
        or authorization.ledger_id != lineage.get("ledger_id")
        or authorization.ledger_path_sha256 != lineage.get("ledger_path_sha256")
        or authorization.predecessor_prefix.to_record()
        != lineage.get("predecessor_prefix")
        or authorization.producer_batch_prefix.to_record()
        != lineage.get("producer_batch_prefix")
        or authorization.ledger_prefix.to_record() != lineage.get("terminal_prefix")
        or authorization.input_bundle_sha256
        != request.get("input_bundle_sha256")
        or authorization.qualification_closed_record_sha256
        != request.get("expected_qualification_closed_record_sha256")
    ):
        raise ValueError("handoff closure request authorization binding drift")
    return authorization


def render_publication_handoff_closure_submit_payload(
    request_authorization: PublicationHandoffClosureRequestAuthorization,
) -> dict[str, Any]:
    """Render the one-task, no-GPU Databricks coordinator submission."""

    authorization = require_publication_handoff_closure_request_authorization(
        request_authorization
    )
    return _render_publication_handoff_closure_submit_payload(
        dict(authorization.request_record)
    )


def _render_publication_handoff_closure_submit_payload(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_closure_request(request)
    coordinator = _required_mapping(request, "coordinator")
    stage = _required_stage(request.get("stage"))
    request_bytes = _canonical_json_bytes(request, pretty=True)
    task_key = f"handoff_closure_{stage}"
    tags = {
        **cast(dict[str, str], coordinator.get("custom_tags", {})),
        "ResourceClass": "SingleNode",
        "campaign": "vllm-0271-publication-v2",
        "closure_stage": stage,
        "purpose": "cachet-vllm-0271-handoff-closure",
        "request_sha256": _required_string(request, "closed_record_sha256")[:32],
    }
    parameters = [
        "--runner-sha256",
        _required_string(coordinator, "runner_sha256"),
        "--package-wheel-uri",
        _required_string(coordinator, "package_wheel_uri"),
        "--package-wheel-sha256",
        _required_string(coordinator, "package_wheel_sha256"),
        "--runtime-lock-uri",
        _required_string(coordinator, "runtime_lock_uri"),
        "--runtime-lock-sha256",
        _required_string(coordinator, "runtime_lock_sha256"),
        "--patched-vllm-wheel-uri",
        _required_string(coordinator, "patched_vllm_wheel_uri"),
        "--patched-vllm-wheel-sha256",
        _required_string(coordinator, "patched_vllm_wheel_sha256"),
        "--patched-flashinfer-wheel-uri",
        _required_string(coordinator, "patched_flashinfer_wheel_uri"),
        "--patched-flashinfer-wheel-sha256",
        _required_string(coordinator, "patched_flashinfer_wheel_sha256"),
        "--runtime-closure-manifest-uri",
        _required_string(coordinator, "runtime_closure_manifest_uri"),
        "--runtime-closure-manifest-sha256",
        _required_string(coordinator, "runtime_closure_manifest_sha256"),
        "--runtime-venv-dir",
        _required_string(coordinator, "runtime_venv_dir"),
        "run-coordinator",
        "--request-uri",
        _required_string(request, "request_uri"),
        "--request-file-sha256",
        sha256(request_bytes).hexdigest(),
        "--expected-request-closed-record-sha256",
        _required_string(request, "closed_record_sha256"),
        "--coordinator-run-id",
        "{{job.run_id}}",
    ]
    _validate_parameter_bytes(parameters)
    payload: dict[str, Any] = {
        "run_name": f"cachet-vllm-0271-{stage}-handoff-closure",
        "tasks": [
            {
                "max_retries": 0,
                "new_cluster": {
                    "aws_attributes": {"availability": "ON_DEMAND", "zone_id": "auto"},
                    "custom_tags": tags,
                    "data_security_mode": "SINGLE_USER",
                    "driver_node_type_id": PUBLICATION_HANDOFF_CLOSURE_NODE_TYPE_ID,
                    "node_type_id": PUBLICATION_HANDOFF_CLOSURE_NODE_TYPE_ID,
                    "num_workers": 0,
                    "single_user_name": _required_string(
                        coordinator, "single_user_name"
                    ),
                    "spark_conf": {
                        "spark.databricks.cluster.profile": "singleNode",
                        "spark.master": "local[*]",
                    },
                    "spark_version": PUBLICATION_HANDOFF_CLOSURE_SPARK_VERSION,
                },
                "spark_python_task": {
                    "parameters": parameters,
                    "python_file": _required_string(coordinator, "runner_python_file"),
                },
                "task_key": task_key,
                "timeout_seconds": PUBLICATION_HANDOFF_CLOSURE_TIMEOUT_SECONDS,
            }
        ],
        "timeout_seconds": PUBLICATION_HANDOFF_CLOSURE_TIMEOUT_SECONDS,
    }
    return bind_databricks_run_idempotency_token(
        payload,
        attempt_id=_required_string(request, "attempt_id"),
    )


def reserve_and_submit_publication_handoff_closure(
    workspace: DatabricksWorkspaceConfig,
    request_authorization: PublicationHandoffClosureRequestAuthorization,
    *,
    reservation_root: str | Path,
) -> dict[str, Any]:
    """Durably reserve one CPU POST, then submit its exact idempotent payload."""

    if not isinstance(workspace, DatabricksWorkspaceConfig):
        raise TypeError("workspace has the wrong type")
    authorization = require_publication_handoff_closure_request_authorization(
        request_authorization
    )
    _require_handoff_controller_lease_root(authorization, reservation_root)
    request = dict(authorization.request_record)
    payload = _render_publication_handoff_closure_submit_payload(request)
    root = _reserve_closure_attempt(
        authorization,
        payload,
        reservation_root=reservation_root,
    )
    lock_path = root / ".submit.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        request_bytes = _canonical_json_bytes(request, pretty=True)
        upload_receipt = upload_databricks_volume_file_bytes_exclusive(
            workspace,
            _required_string(request, "request_uri"),
            request_bytes,
            max_bytes=PUBLICATION_HANDOFF_CLOSURE_REQUEST_BYTES_MAX,
        )
        expected_upload_receipt = {
            "dbfs_uri": request["request_uri"],
            "file_sha256": sha256(request_bytes).hexdigest(),
            "size_bytes": len(request_bytes),
        }
        if any(
            upload_receipt.get(key) != value
            for key, value in expected_upload_receipt.items()
        ) or upload_receipt.get("created") not in {True, False}:
            raise ValueError("coordinator request upload receipt binding drift")
        _write_or_require_exact(
            root / "request-upload.json",
            _canonical_json_bytes(
                {**expected_upload_receipt, "exclusive_bytes_proven": True},
                pretty=True,
            ),
        )
        response_path = root / "submit-response.json"
        if response_path.exists():
            response = _read_canonical_json_file(
                response_path, "coordinator submit response"
            )
            _required_run_id(response.get("run_id"), "submit response run_id")
            return response
        response = submit_databricks_run(workspace, payload)
        _required_run_id(response.get("run_id"), "submit response run_id")
        _write_or_require_exact(
            response_path, _canonical_json_bytes(response, pretty=True)
        )
        return dict(response)


def collect_publication_handoff_closure(
    workspace: DatabricksWorkspaceConfig,
    *,
    reservation_root: str | Path,
    request_authorization: PublicationHandoffClosureRequestAuthorization,
) -> PublicationHandoffRemoteClosureAuthorization:
    """Collect direct CPU status and compact result without touching ``/dbfs``."""

    if not isinstance(workspace, DatabricksWorkspaceConfig):
        raise TypeError("workspace has the wrong type")
    authorization = require_publication_handoff_closure_request_authorization(
        request_authorization
    )
    _require_handoff_controller_lease_root(authorization, reservation_root)
    root = _existing_local_reservation_root(
        reservation_root,
        request_authorization=authorization,
    )
    request = _read_canonical_json_file(root / "request.json", "closure request")
    payload = _read_canonical_json_file(
        root / "submit-payload.json", "closure submit payload"
    )
    response = _read_canonical_json_file(
        root / "submit-response.json", "closure submit response"
    )
    upload_receipt = _read_canonical_json_file(
        root / "request-upload.json", "closure request upload receipt"
    )
    _validate_closure_request(request)
    request_bytes = _canonical_json_bytes(request, pretty=True)
    expected_upload_receipt = {
        "dbfs_uri": request.get("request_uri"),
        "exclusive_bytes_proven": True,
        "file_sha256": sha256(request_bytes).hexdigest(),
        "size_bytes": len(request_bytes),
    }
    if upload_receipt != expected_upload_receipt:
        raise ValueError("reserved coordinator request upload receipt drift")
    if payload != _render_publication_handoff_closure_submit_payload(request):
        raise ValueError("reserved coordinator submit payload drift")
    run_id = _required_run_id(response.get("run_id"), "submit response run_id")
    terminal = get_databricks_run(workspace, run_id)
    control_plane_status_sha256 = _validate_coordinator_terminal_run(
        terminal,
        submit_payload=payload,
        expected_run_id=run_id,
        expected_stage=_required_stage(request.get("stage")),
    )
    remote_request_bytes = download_databricks_volume_file_bytes(
        workspace,
        _required_string(request, "request_uri"),
        max_bytes=PUBLICATION_HANDOFF_CLOSURE_REQUEST_BYTES_MAX,
    )
    if not hmac.compare_digest(remote_request_bytes, request_bytes):
        raise ValueError("remote coordinator request bytes drift from reservation")
    result_uri = _required_string(request, "result_uri")
    result_bytes = download_databricks_volume_file_bytes(
        workspace,
        result_uri,
        max_bytes=PUBLICATION_HANDOFF_CLOSURE_RESULT_BYTES_MAX,
    )
    result = _canonical_closed_record_from_bytes(result_bytes, "coordinator result")
    _validate_closure_result(result, request=request)
    if _required_string(_required_mapping(result, "coordinator"), "run_id") != run_id:
        raise ValueError("coordinator result belongs to another Databricks run")
    _write_or_require_exact(
        root / "runs-get.json", _canonical_json_bytes(terminal, pretty=True)
    )
    _write_or_require_exact(root / "coordinator-result.json", result_bytes)
    return PublicationHandoffRemoteClosureAuthorization(
        request=request,
        result=result,
        result_file_sha256=sha256(result_bytes).hexdigest(),
        coordinator_run_id=run_id,
        control_plane_status_sha256=control_plane_status_sha256,
        _issuer=_AUTHORIZATION_ISSUER,
    )


def require_q8_handoff_remote_closure_authorization(
    authorization: object,
    *,
    expected_output_root_uri: str,
    expected_execution_file_sha256: str,
    expected_input_bundle_sha256: str,
    expected_qualification_closed_record_sha256: str,
) -> PublicationHandoffRemoteClosureAuthorization:
    return _require_remote_closure_authorization(
        authorization,
        expected_stage=PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE,
        expected_output_root_uri=expected_output_root_uri,
        expected_execution_file_sha256=expected_execution_file_sha256,
        expected_input_bundle_sha256=expected_input_bundle_sha256,
        expected_qualification_closed_record_sha256=(
            expected_qualification_closed_record_sha256
        ),
    )


def require_q8_handoff_remote_closure_predecessor_authorization(
    authorization: object,
    *,
    expected_ledger_id: str,
    expected_ledger_path_sha256: str,
    expected_ledger_prefix: DatabricksLedgerPrefix,
    expected_input_bundle_sha256: str,
    expected_qualification_closed_record_sha256: str,
) -> PublicationHandoffRemoteClosureAuthorization:
    """Require the exact remote Q8 terminal closure used to open BF16."""

    if not isinstance(authorization, PublicationHandoffRemoteClosureAuthorization):
        raise TypeError(
            "BF16 predecessor requires "
            "PublicationHandoffRemoteClosureAuthorization"
        )
    resolved = require_q8_handoff_remote_closure_authorization(
        authorization,
        expected_output_root_uri=authorization.output_root_uri,
        expected_execution_file_sha256=authorization.execution_file_sha256,
        expected_input_bundle_sha256=expected_input_bundle_sha256,
        expected_qualification_closed_record_sha256=(
            expected_qualification_closed_record_sha256
        ),
    )
    if not isinstance(expected_ledger_prefix, DatabricksLedgerPrefix):
        raise TypeError("expected_ledger_prefix has the wrong type")
    if (
        resolved.ledger_id != _nonempty(expected_ledger_id, "expected_ledger_id")
        or not hmac.compare_digest(
            resolved.ledger_path_sha256,
            _require_sha256(
                expected_ledger_path_sha256,
                "expected_ledger_path_sha256",
            ),
        )
        or resolved.ledger_prefix != expected_ledger_prefix
    ):
        raise ValueError("remote Q8 closure ledger predecessor binding drift")
    return resolved


def require_bf16_handoff_remote_closure_authorization(
    authorization: object,
    *,
    expected_output_root_uri: str,
    expected_execution_file_sha256: str,
    expected_input_bundle_sha256: str,
    expected_qualification_closed_record_sha256: str,
) -> PublicationHandoffRemoteClosureAuthorization:
    return _require_remote_closure_authorization(
        authorization,
        expected_stage=PUBLICATION_HANDOFF_CLOSURE_BF16_STAGE,
        expected_output_root_uri=expected_output_root_uri,
        expected_execution_file_sha256=expected_execution_file_sha256,
        expected_input_bundle_sha256=expected_input_bundle_sha256,
        expected_qualification_closed_record_sha256=(
            expected_qualification_closed_record_sha256
        ),
    )


def run_publication_handoff_closure_coordinator(
    request: Mapping[str, Any],
    *,
    coordinator_run_id: str,
) -> dict[str, Any]:
    """Run or fully revalidate one mounted closure and emit its compact result."""

    _validate_closure_request(request)
    run_id = _required_run_id(coordinator_run_id, "coordinator_run_id")
    coordinator = _required_mapping(request, "coordinator")
    _verify_source_closure(coordinator)
    stage = _required_stage(request.get("stage"))
    root = _cluster_path(_required_string(request, "output_root_uri"))
    result_path = _cluster_path(_required_string(request, "result_uri"))
    plan_binding = _required_mapping(request, "plan")
    plan_path = _cluster_path(_required_string(plan_binding, "uri"))
    plan_bytes = _read_verified_file_bytes(
        plan_path,
        _required_string(plan_binding, "file_sha256"),
        "generation plan",
    )
    plan = _canonical_json_object_from_bytes(plan_bytes, "generation plan")
    if plan.get("closed_record_sha256") != plan_binding.get("closed_record_sha256"):
        raise ValueError("generation plan closure binding drift")
    prepared_root = _cluster_path(_required_string(request, "prepared_input_root_uri"))
    if not prepared_root.is_dir() or prepared_root.is_symlink():
        raise ValueError("prepared input root must be one mounted real directory")
    ledger_binding = _required_mapping(request, "ledger_snapshot")
    ledger = databricks_cluster_hour_ledger_from_record(
        _required_mapping(ledger_binding, "record")
    )
    if _canonical_sha256(databricks_cluster_hour_ledger_to_record(ledger)) != (
        ledger_binding.get("record_sha256")
    ):
        raise ValueError("ledger snapshot record SHA-256 drift")
    ledger_path_binding = _required_string(
        _required_mapping(request, "ledger_lineage"), "ledger_path_sha256"
    )
    tokenizer = load_main_latency_tokenizer()
    execution_contract = _required_mapping(request, "execution_contract")
    worker_evidence = _required_sequence(request, "worker_evidence")
    attempt_ids = {
        _required_int(
            _mapping(item, "worker evidence"), "worker_index"
        ): _required_string(_mapping(item, "worker evidence"), "attempt_id")
        for item in worker_evidence
    }
    producer_batch_prefix = databricks_ledger_prefix_from_record(
        _required_mapping(
            _required_mapping(request, "ledger_lineage"),
            "producer_batch_prefix",
        )
    )
    closed: Any
    if stage == PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE:
        q8_execution_config = _q8._execution_config_from_record(execution_contract)
        bindings = {
            _required_int(
                _mapping(item, "Q8 worker evidence"), "worker_index"
            ): _q8.PublicationLatencyHandoffDatabricksAttestationBinding(
                worker_index=_required_int(
                    _mapping(item, "Q8 worker evidence"), "worker_index"
                ),
                path=(
                    root
                    / _q8.PUBLICATION_LATENCY_HANDOFF_DATABRICKS_ATTESTATION_DIRECTORY
                    / f"worker-{_required_int(_mapping(item, 'Q8 worker evidence'), 'worker_index'):02d}.json"
                ),
                file_sha256=_required_string(
                    _mapping(item, "Q8 worker evidence"), "attestation_file_sha256"
                ),
                closed_record_sha256=_required_string(
                    _mapping(item, "Q8 worker evidence"),
                    "attestation_closed_record_sha256",
                ),
            )
            for item in worker_evidence
        }
        execution_path = root / _q8.PUBLICATION_LATENCY_HANDOFF_EXECUTION_FILENAME
        if result_path.exists() or execution_path.exists():
            closed = _q8._replay_closed_publication_latency_handoff_generation(
                plan,
                prepared_input_dir=prepared_root,
                durable_output_root=root,
                tokenizer=tokenizer,
                config=q8_execution_config,
                ledger_snapshot=ledger,
                ledger_path_sha256=ledger_path_binding,
                expected_producer_batch_prefix=producer_batch_prefix,
                attempt_ids_by_worker=attempt_ids,
                attestations_by_worker=bindings,
                _issuer=_q8._POST_CLOSE_REPLAY_ISSUER,
            )
        else:
            closed = _q8.close_publication_latency_handoff_generation_from_workers(
                plan,
                prepared_input_dir=prepared_root,
                durable_output_root=root,
                tokenizer=tokenizer,
                config=q8_execution_config,
                ledger_path=Path("/local_disk0/cachet-remote-ledger-snapshot.json"),
                attempt_ids_by_worker=attempt_ids,
                attestations_by_worker=bindings,
                _ledger_snapshot=ledger,
                _ledger_path_sha256=ledger_path_binding,
                _expected_producer_batch_prefix=producer_batch_prefix,
                _remote_ledger_issuer=_q8._REMOTE_CLOSURE_LEDGER_ISSUER,
            )
    else:
        bf16_execution_config = _bf16._execution_config_from_record(execution_contract)
        authorizations: dict[int, _bf16.PublicationBF16HandoffWorkerAuthorization] = {}
        for item in worker_evidence:
            evidence = _mapping(item, "BF16 worker evidence")
            worker_index = _required_int(evidence, "worker_index")
            binding = _bf16.PublicationBF16HandoffAttestationBinding(
                worker_index=worker_index,
                path=(
                    root
                    / _bf16.PUBLICATION_BF16_HANDOFF_ATTESTATION_DIRECTORY
                    / f"worker-{worker_index:02d}.json"
                ),
                file_sha256=_required_string(evidence, "attestation_file_sha256"),
                closed_record_sha256=_required_string(
                    evidence, "attestation_closed_record_sha256"
                ),
            )
            authorizations[worker_index] = (
                _bf16.PublicationBF16HandoffWorkerAuthorization(
                    binding=binding,
                    attempt_id=_required_string(evidence, "attempt_id"),
                    ledger_id=ledger.ledger_id,
                    ledger_path_sha256=ledger_path_binding,
                    producer_batch_prefix=producer_batch_prefix,
                    control_plane_status_sha256=_required_string(
                        evidence, "control_plane_status_sha256"
                    ),
                    _issuer=_bf16._WORKER_AUTHORIZATION_ISSUER,
                )
            )
        execution_path = root / _bf16.PUBLICATION_BF16_HANDOFF_EXECUTION_FILENAME
        if result_path.exists() or execution_path.exists():
            closed = _bf16._replay_closed_publication_bf16_handoff_generation(
                plan,
                prepared_input_dir=prepared_root,
                durable_output_root=root,
                tokenizer=tokenizer,
                config=bf16_execution_config,
                ledger_snapshot=ledger,
                ledger_path_sha256=ledger_path_binding,
                expected_producer_batch_prefix=producer_batch_prefix,
                attempt_ids_by_worker=attempt_ids,
                worker_authorizations=authorizations,
                _issuer=_bf16._POST_CLOSE_REPLAY_ISSUER,
            )
        else:
            closed = _bf16.close_publication_bf16_handoff_generation_from_workers(
                plan,
                prepared_input_dir=prepared_root,
                durable_output_root=root,
                tokenizer=tokenizer,
                config=bf16_execution_config,
                ledger_path=Path("/local_disk0/cachet-remote-ledger-snapshot.json"),
                attempt_ids_by_worker=attempt_ids,
                worker_authorizations=authorizations,
                _ledger_snapshot=ledger,
                _ledger_path_sha256=ledger_path_binding,
                _remote_ledger_issuer=_bf16._REMOTE_CLOSURE_LEDGER_ISSUER,
            )
    result = _build_compact_result(request, closed=closed, coordinator_run_id=run_id)
    result_bytes = _canonical_json_bytes(result, pretty=True)
    if len(result_bytes) > PUBLICATION_HANDOFF_CLOSURE_RESULT_BYTES_MAX:
        raise ValueError("compact coordinator result exceeds the Files API cap")
    if result_path.exists():
        if result_path.is_symlink():
            raise ValueError("existing coordinator result must not be a symlink")
        existing_bytes = result_path.read_bytes()
        existing = _canonical_closed_record_from_bytes(
            existing_bytes, "existing coordinator result"
        )
        _validate_closure_result(existing, request=request)
        if (
            _required_string(_required_mapping(existing, "coordinator"), "run_id")
            != run_id
        ):
            raise FileExistsError("existing coordinator result belongs to another run")
        if not hmac.compare_digest(existing_bytes, result_bytes):
            raise ValueError(
                "existing coordinator result differs from full post-close replay"
            )
        return existing
    result_path.parent.mkdir(parents=True, exist_ok=True)
    _write_or_require_exact(result_path, result_bytes)
    return result


def write_publication_handoff_closure_runner_script(path: str | Path) -> Path:
    destination = Path(path).expanduser().absolute()
    _write_or_require_exact(
        destination,
        PUBLICATION_HANDOFF_CLOSURE_RUNNER_SCRIPT.encode("utf-8"),
    )
    return destination


def _build_closure_request(
    *,
    stage: str,
    attempt_id: str | None,
    coordinator_config: PublicationHandoffClosureCoordinatorConfig,
    plan_uri: str,
    plan_file_sha256: str,
    plan_closed_record_sha256: str,
    input_bundle_sha256: str,
    qualified_artifact_pins: GPUQualificationArtifactPinsV2,
    prepared_input_root_uri: str,
    durable_output_root_uri: str,
    controller_singleton: Mapping[str, Any],
    execution_contract: Mapping[str, Any],
    ledger_path: str | Path,
    batch_authorization: Any,
    worker_evidence: Sequence[Mapping[str, Any]],
    expected_qualification_closed_record_sha256: str,
) -> dict[str, Any]:
    stage = _required_stage(stage)
    if not isinstance(coordinator_config, PublicationHandoffClosureCoordinatorConfig):
        raise TypeError("coordinator_config has the wrong type")
    if not isinstance(qualified_artifact_pins, GPUQualificationArtifactPinsV2):
        raise TypeError("qualified_artifact_pins must be native v2")
    plan_uri = _canonical_volume_file_uri(plan_uri, "plan_uri")
    prepared_uri = _canonical_volume_directory_uri(
        prepared_input_root_uri, "prepared_input_root_uri"
    )
    output_uri = _canonical_volume_directory_uri(
        durable_output_root_uri, "durable_output_root_uri"
    )
    singleton = dict(controller_singleton)
    _validate_handoff_closure_singleton(
        singleton,
        expected_stage=stage,
        expected_output_root_uri=output_uri,
    )
    canonical_attempt = _handoff_closure_attempt_id(singleton)
    if attempt_id is not None and _required_attempt_id(attempt_id) != canonical_attempt:
        raise ValueError("handoff closure attempt_id differs from singleton identity")
    attempt = canonical_attempt
    path_sha = databricks_ledger_path_sha256(ledger_path)
    if batch_authorization.ledger_path_sha256 != path_sha:
        raise ValueError("producer batch belongs to a different ledger path")
    evidence_attempt_ids = tuple(
        _required_string(_mapping(item, "worker evidence"), "attempt_id")
        for item in worker_evidence
    )
    if evidence_attempt_ids != tuple(batch_authorization.attempt_ids):
        raise ValueError("worker evidence does not match the authorized producer batch")
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    if ledger.ledger_id != batch_authorization.batch_prefix.ledger_id:
        raise ValueError("producer batch ledger identity drift")
    require_databricks_ledger_prefix(ledger, batch_authorization.batch_prefix)
    terminal_prefix = require_databricks_batch_terminal_closure(
        ledger,
        batch_authorization,
        require_complete_current_prefix=True,
    )
    request_root_uri = _handoff_closure_request_root_uri(output_uri, stage=stage)
    if coordinator_config.request_root_uri != request_root_uri:
        raise ValueError(
            "handoff closure request_root_uri differs from the bound output root"
        )
    result_uri = _handoff_closure_result_uri(output_uri, stage=stage)
    request_uri = _join_volume_uri(request_root_uri, "request.json")
    ledger_record = databricks_cluster_hour_ledger_to_record(ledger)
    request: dict[str, Any] = {
        "attempt_id": attempt,
        "closed_record_sha256": "",
        "coordinator": coordinator_config.to_record(),
        "controller_singleton": singleton,
        "execution_contract": dict(execution_contract),
        "execution_contract_sha256": _canonical_sha256(execution_contract),
        "expected_qualification_closed_record_sha256": _require_sha256(
            expected_qualification_closed_record_sha256,
            "expected_qualification_closed_record_sha256",
        ),
        "input_bundle_sha256": _require_sha256(
            input_bundle_sha256, "input_bundle_sha256"
        ),
        "ledger_lineage": {
            "ledger_id": ledger.ledger_id,
            "ledger_path_sha256": path_sha,
            "predecessor_prefix": batch_authorization.predecessor_prefix.to_record(),
            "producer_batch_prefix": batch_authorization.batch_prefix.to_record(),
            "terminal_prefix": terminal_prefix.to_record(),
        },
        "ledger_snapshot": {
            "record": ledger_record,
            "record_sha256": _canonical_sha256(ledger_record),
        },
        "output_root_uri": output_uri,
        "plan": {
            "closed_record_sha256": _require_sha256(
                plan_closed_record_sha256, "plan_closed_record_sha256"
            ),
            "file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
            "uri": plan_uri,
        },
        "prepared_input_root_uri": prepared_uri,
        "qualified_artifact_pins": qualified_artifact_pins.to_record(),
        "record_type": PUBLICATION_HANDOFF_CLOSURE_REQUEST_RECORD_TYPE,
        "request_uri": request_uri,
        "result_uri": result_uri,
        "schema_version": PUBLICATION_HANDOFF_CLOSURE_SCHEMA_VERSION,
        "stage": stage,
        "worker_evidence": [dict(item) for item in worker_evidence],
    }
    request["closed_record_sha256"] = _closed_record_sha256(request)
    _validate_closure_request(request)
    return request


def _validate_closure_request(record: Mapping[str, Any]) -> None:
    if (
        record.get("record_type") != PUBLICATION_HANDOFF_CLOSURE_REQUEST_RECORD_TYPE
        or record.get("schema_version") != PUBLICATION_HANDOFF_CLOSURE_SCHEMA_VERSION
        or record.get("closed_record_sha256") != _closed_record_sha256(record)
    ):
        raise ValueError("handoff closure request envelope is invalid")
    stage = _required_stage(record.get("stage"))
    output_root = _canonical_volume_directory_uri(
        _required_string(record, "output_root_uri"), "output_root_uri"
    )
    singleton = _required_mapping(record, "controller_singleton")
    _validate_handoff_closure_singleton(
        singleton,
        expected_stage=stage,
        expected_output_root_uri=output_root,
    )
    attempt_id = _required_attempt_id(record.get("attempt_id"))
    if attempt_id != _handoff_closure_attempt_id(singleton):
        raise ValueError("handoff closure request attempt identity drift")
    expected_result = _handoff_closure_result_uri(output_root, stage=stage)
    if record.get("result_uri") != expected_result:
        raise ValueError("coordinator result URI is not canonical")
    _canonical_volume_directory_uri(
        _required_string(record, "prepared_input_root_uri"),
        "prepared_input_root_uri",
    )
    plan = _required_mapping(record, "plan")
    _canonical_volume_file_uri(_required_string(plan, "uri"), "plan.uri")
    for field_name in ("file_sha256", "closed_record_sha256"):
        _require_sha256(_required_string(plan, field_name), f"plan.{field_name}")
    for field_name in (
        "input_bundle_sha256",
        "execution_contract_sha256",
        "expected_qualification_closed_record_sha256",
    ):
        _require_sha256(_required_string(record, field_name), field_name)
    if record.get("execution_contract_sha256") != _canonical_sha256(
        _required_mapping(record, "execution_contract")
    ):
        raise ValueError("execution contract digest drift")
    coordinator = _required_mapping(record, "coordinator")
    coordinator_config = _coordinator_config_from_record(coordinator)
    qualified_pins = _q8.gpu_qualification_artifact_pins_v2_from_record(
        _required_mapping(record, "qualified_artifact_pins")
    )
    expected_qualified_pins = {
        "cachet_source_tree_sha256": coordinator_config.cachet_source_tree_sha256,
        "input_bundle_sha256": record.get("input_bundle_sha256"),
        "package_wheel_sha256": coordinator_config.package_wheel_sha256,
        "patched_flashinfer_wheel_sha256": (
            coordinator_config.patched_flashinfer_wheel_sha256
        ),
        "patched_vllm_wheel_sha256": coordinator_config.patched_vllm_wheel_sha256,
        "runtime_closure_manifest_sha256": (
            coordinator_config.runtime_closure_manifest_sha256
        ),
        "runtime_lock_sha256": coordinator_config.runtime_lock_sha256,
    }
    if any(
        qualified_pins.to_record().get(name) != value
        for name, value in expected_qualified_pins.items()
    ):
        raise ValueError("closure request differs from native-v2 qualification pins")
    if qualified_pins.runner_sha256 == coordinator_config.runner_sha256:
        raise ValueError(
            "qualification runner must remain distinct from the closure runner"
        )
    expected_request_root = _handoff_closure_request_root_uri(
        output_root, stage=stage
    )
    if coordinator_config.request_root_uri != expected_request_root:
        raise ValueError("coordinator request root is not singleton-derived")
    expected_request = _join_volume_uri(expected_request_root, "request.json")
    if record.get("request_uri") != expected_request:
        raise ValueError("coordinator request URI is not canonical")
    ledger_binding = _required_mapping(record, "ledger_snapshot")
    ledger_record = _required_mapping(ledger_binding, "record")
    ledger = databricks_cluster_hour_ledger_from_record(ledger_record)
    if ledger_binding.get("record_sha256") != _canonical_sha256(ledger_record):
        raise ValueError("ledger snapshot digest drift")
    lineage = _required_mapping(record, "ledger_lineage")
    if lineage.get("ledger_id") != ledger.ledger_id:
        raise ValueError("ledger snapshot identity drift")
    _require_sha256(
        _required_string(lineage, "ledger_path_sha256"), "ledger_path_sha256"
    )
    predecessor = databricks_ledger_prefix_from_record(
        _required_mapping(lineage, "predecessor_prefix")
    )
    producer = databricks_ledger_prefix_from_record(
        _required_mapping(lineage, "producer_batch_prefix")
    )
    terminal = databricks_ledger_prefix_from_record(
        _required_mapping(lineage, "terminal_prefix")
    )
    for prefix in (predecessor, producer, terminal):
        if prefix.ledger_id != ledger.ledger_id:
            raise ValueError("closure request ledger prefix identity drift")
        require_databricks_ledger_prefix(ledger, prefix)
    if databricks_ledger_prefix(ledger) != terminal:
        raise ValueError("closure ledger snapshot is not the terminal prefix")
    evidence = _required_sequence(record, "worker_evidence")
    if len(evidence) != 16:
        raise ValueError("handoff closure requires sixteen worker evidence records")
    workers: set[int] = set()
    attempts: set[str] = set()
    for raw in evidence:
        item = _mapping(raw, "worker evidence")
        worker_index = _required_int(item, "worker_index")
        if not 0 <= worker_index < 16 or worker_index in workers:
            raise ValueError("worker evidence identity coverage drift")
        workers.add(worker_index)
        attempt = _nonempty(_required_string(item, "attempt_id"), "attempt_id")
        if attempt in attempts:
            raise ValueError("worker evidence attempt IDs must be unique")
        attempts.add(attempt)
        for name in (
            "attestation_file_sha256",
            "attestation_closed_record_sha256",
        ):
            _require_sha256(_required_string(item, name), name)
        if stage == PUBLICATION_HANDOFF_CLOSURE_BF16_STAGE:
            _require_sha256(
                _required_string(item, "control_plane_status_sha256"),
                "control_plane_status_sha256",
            )
    if workers != set(range(16)):
        raise ValueError("worker evidence must cover workers 0..15 exactly")
    if len(_canonical_json_bytes(record, pretty=True)) > (
        PUBLICATION_HANDOFF_CLOSURE_REQUEST_BYTES_MAX
    ):
        raise ValueError("handoff closure request exceeds the bounded Files API cap")


def _build_compact_result(
    request: Mapping[str, Any],
    *,
    closed: Any,
    coordinator_run_id: str,
) -> dict[str, Any]:
    stage = _required_stage(request.get("stage"))
    root = _cluster_path(_required_string(request, "output_root_uri"))
    if stage == PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE:
        execution_path = closed.execution_record_path
        execution_record = dict(closed.record)
        manifests = []
        for bundle in _q8._mapping_sequence(
            execution_record.get("bundles"), field_name="bundles"
        ):
            relative = _q8._required_string(bundle, "manifest_relative_path")
            path = root / PurePosixPath(relative)
            manifest = _handoff_artifacts.read_publication_latency_handoff_bundle(path)
            manifests.append(
                {
                    "closed_record_sha256": _q8._required_string(
                        manifest, "closed_record_sha256"
                    ),
                    "context_tokens": _q8._required_int(bundle, "context_tokens"),
                    "file_sha256": _file_sha256(path),
                    "portable_bundle_sha256": _q8._required_string(
                        manifest, "portable_bundle_sha256"
                    ),
                    "record": manifest,
                    "source_root_uri": _join_volume_uri(
                        _required_string(request, "output_root_uri"),
                        _q8._required_string(bundle, "source_root_relative_path"),
                    ),
                    "uri": _join_volume_uri(
                        _required_string(request, "output_root_uri"), relative
                    ),
                }
            )
    else:
        execution_path = closed.execution_record_path
        execution_record = dict(closed.record)
        manifest = dict(closed.manifest)
        manifests = [
            {
                "closed_record_sha256": _bf16._required_string(
                    manifest, "closed_record_sha256"
                ),
                "context_tokens": _bf16.PUBLICATION_BF16_HANDOFF_CONTEXT_TOKENS,
                "file_sha256": _file_sha256(closed.manifest_path),
                "portable_bundle_sha256": _bf16._required_string(
                    manifest, "portable_bundle_sha256"
                ),
                "record": manifest,
                "source_root_uri": _join_volume_uri(
                    _required_string(request, "output_root_uri"),
                    closed.source_root.relative_to(root).as_posix(),
                ),
                "uri": _join_volume_uri(
                    _required_string(request, "output_root_uri"),
                    closed.manifest_path.relative_to(root).as_posix(),
                ),
            }
        ]
    result: dict[str, Any] = {
        "closed_record_sha256": "",
        "coordinator": {
            "close_function": (
                "close_publication_latency_handoff_generation_from_workers"
                if stage == PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE
                else "close_publication_bf16_handoff_generation_from_workers"
            ),
            "node_type_id": PUBLICATION_HANDOFF_CLOSURE_NODE_TYPE_ID,
            "run_id": coordinator_run_id,
            "runner_sha256": PUBLICATION_HANDOFF_CLOSURE_RUNNER_SHA256,
            "tree_validation": "full_mounted_byte_replay",
        },
        "execution": {
            "closed_record_sha256": _required_string(
                execution_record, "closed_record_sha256"
            ),
            "file_sha256": _file_sha256(execution_path),
            "record": execution_record,
            "uri": _join_volume_uri(
                _required_string(request, "output_root_uri"),
                execution_path.relative_to(root).as_posix(),
            ),
        },
        "ledger_lineage": dict(_required_mapping(request, "ledger_lineage")),
        "manifests": manifests,
        "output_root_uri": _required_string(request, "output_root_uri"),
        "pins": {
            "coordinator": dict(_required_mapping(request, "coordinator")),
            "execution_contract_sha256": _required_string(
                request, "execution_contract_sha256"
            ),
            "input_bundle_sha256": _required_string(request, "input_bundle_sha256"),
            "plan": dict(_required_mapping(request, "plan")),
            "qualified_artifact_pins": dict(
                _required_mapping(request, "qualified_artifact_pins")
            ),
            "qualification_closed_record_sha256": _required_string(
                request, "expected_qualification_closed_record_sha256"
            ),
        },
        "record_type": PUBLICATION_HANDOFF_CLOSURE_RESULT_RECORD_TYPE,
        "request_closed_record_sha256": _required_string(
            request, "closed_record_sha256"
        ),
        "schema_version": PUBLICATION_HANDOFF_CLOSURE_SCHEMA_VERSION,
        "stage": stage,
    }
    result["closed_record_sha256"] = _closed_record_sha256(result)
    _validate_closure_result(result, request=request)
    return result


def _validate_closure_result(
    record: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> None:
    _validate_closure_request(request)
    if (
        record.get("record_type") != PUBLICATION_HANDOFF_CLOSURE_RESULT_RECORD_TYPE
        or record.get("schema_version") != PUBLICATION_HANDOFF_CLOSURE_SCHEMA_VERSION
        or record.get("closed_record_sha256") != _closed_record_sha256(record)
        or record.get("request_closed_record_sha256")
        != request.get("closed_record_sha256")
        or record.get("stage") != request.get("stage")
        or record.get("output_root_uri") != request.get("output_root_uri")
        or record.get("ledger_lineage") != request.get("ledger_lineage")
    ):
        raise ValueError("handoff closure result envelope/binding is invalid")
    stage = _required_stage(record.get("stage"))
    coordinator = _required_mapping(record, "coordinator")
    expected_close_function = (
        "close_publication_latency_handoff_generation_from_workers"
        if stage == PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE
        else "close_publication_bf16_handoff_generation_from_workers"
    )
    if (
        coordinator.get("close_function") != expected_close_function
        or coordinator.get("node_type_id") != PUBLICATION_HANDOFF_CLOSURE_NODE_TYPE_ID
        or coordinator.get("runner_sha256") != PUBLICATION_HANDOFF_CLOSURE_RUNNER_SHA256
        or coordinator.get("tree_validation") != "full_mounted_byte_replay"
    ):
        raise ValueError("coordinator result does not attest full mounted closure")
    _required_run_id(coordinator.get("run_id"), "coordinator run_id")
    pins = _required_mapping(record, "pins")
    expected_pins = {
        "coordinator": dict(_required_mapping(request, "coordinator")),
        "execution_contract_sha256": request.get("execution_contract_sha256"),
        "input_bundle_sha256": request.get("input_bundle_sha256"),
        "plan": dict(_required_mapping(request, "plan")),
        "qualified_artifact_pins": dict(
            _required_mapping(request, "qualified_artifact_pins")
        ),
        "qualification_closed_record_sha256": request.get(
            "expected_qualification_closed_record_sha256"
        ),
    }
    if dict(pins) != expected_pins:
        raise ValueError("coordinator result pin set drift")
    execution = _required_mapping(record, "execution")
    execution_record = _required_mapping(execution, "record")
    expected_execution_name = (
        _q8.PUBLICATION_LATENCY_HANDOFF_EXECUTION_FILENAME
        if stage == PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE
        else _bf16.PUBLICATION_BF16_HANDOFF_EXECUTION_FILENAME
    )
    if execution.get("uri") != _join_volume_uri(
        _required_string(request, "output_root_uri"), expected_execution_name
    ):
        raise ValueError("coordinator execution URI drift")
    if (
        execution.get("closed_record_sha256")
        != execution_record.get("closed_record_sha256")
        or execution.get("file_sha256")
        != sha256(_canonical_json_bytes(execution_record, pretty=True)).hexdigest()
        or execution_record.get("plan_closed_record_sha256")
        != _required_mapping(request, "plan").get("closed_record_sha256")
        or execution_record.get("input_bundle_sha256")
        != request.get("input_bundle_sha256")
        or _canonical_sha256(_required_mapping(execution_record, "execution_contract"))
        != request.get("execution_contract_sha256")
        or _required_mapping(execution_record, "generator_hardware").get(
            "qualification_closed_record_sha256"
        )
        != request.get("expected_qualification_closed_record_sha256")
    ):
        raise ValueError("coordinator execution record binding drift")
    if stage == PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE:
        if (
            execution_record.get("record_type")
            != _q8.PUBLICATION_LATENCY_HANDOFF_EXECUTION_RECORD_TYPE
            or execution_record.get("schema_version")
            != _q8.PUBLICATION_LATENCY_HANDOFF_EXECUTION_SCHEMA_VERSION
            or execution_record.get("execution_mode")
            != _q8.PUBLICATION_LATENCY_HANDOFF_EXECUTION_MODE_DISTRIBUTED
            or execution_record.get("closed_record_sha256")
            != _q8._closed_record_sha256(execution_record)
        ):
            raise ValueError("Q8 execution closed digest drift")
    elif (
        execution_record.get("record_type")
        != _bf16.PUBLICATION_BF16_HANDOFF_EXECUTION_RECORD_TYPE
        or execution_record.get("schema_version")
        != _bf16.PUBLICATION_BF16_HANDOFF_SCHEMA_VERSION
        or execution_record.get("execution_mode")
        != _bf16.PUBLICATION_BF16_HANDOFF_EXECUTION_MODE
        or execution_record.get("closed_record_sha256")
        != _bf16._closed_record_sha256(execution_record)
    ):
        raise ValueError("BF16 execution closed digest drift")
    accounting = _required_mapping(execution_record, "accounting")
    workers = [
        _mapping(item, "execution worker")
        for item in _required_sequence(execution_record, "workers")
    ]
    if (
        accounting.get("coordinator_gpu_hours") != 0.0
        or accounting.get("payload_copy_count_during_closure") != 0
        or accounting.get("worker_count") != 16
        or len(workers) != 16
        or [item.get("worker_index") for item in workers] != list(range(16))
    ):
        raise ValueError("coordinator execution accounting/worker coverage drift")
    reconciliation = _required_mapping(execution_record, "ledger_reconciliation")
    lineage = _required_mapping(request, "ledger_lineage")
    if reconciliation.get("ledger_id") != lineage.get("ledger_id"):
        raise ValueError("execution ledger reconciliation identity drift")
    evidence_by_worker = {
        _required_int(item, "worker_index"): item
        for item in (
            _mapping(raw, "worker evidence")
            for raw in _required_sequence(request, "worker_evidence")
        )
    }
    if stage == PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE:
        attempts = [
            _mapping(item, "Q8 ledger attempt")
            for item in _required_sequence(reconciliation, "attempts")
        ]
        if (
            len(attempts) != 16
            or reconciliation.get("attempts_sha256") != _canonical_sha256(attempts)
            or [item.get("worker_index") for item in attempts] != list(range(16))
            or any(
                item.get("verification_source") != "direct_databricks_runs_get"
                for item in attempts
            )
            or any(
                item.get("attempt_id") != evidence_by_worker[index].get("attempt_id")
                or item.get("attestation_file_sha256")
                != evidence_by_worker[index].get("attestation_file_sha256")
                or item.get("attestation_closed_record_sha256")
                != evidence_by_worker[index].get("attestation_closed_record_sha256")
                for index, item in enumerate(attempts)
            )
        ):
            raise ValueError("Q8 execution ledger closure is incomplete")
    else:
        attempts = [
            _mapping(item, "BF16 ledger attempt")
            for item in _required_sequence(reconciliation, "attempts")
        ]
        if (
            reconciliation.get("attempt_count") != 16
            or reconciliation.get("verification_source") != "direct_databricks_runs_get"
            or len(attempts) != 16
            or reconciliation.get("attempts_sha256") != _canonical_sha256(attempts)
            or [item.get("worker_index") for item in attempts] != list(range(16))
            or any(
                item.get("attempt_id") != evidence_by_worker[index].get("attempt_id")
                or item.get("attestation_closed_record_sha256")
                != evidence_by_worker[index].get("attestation_closed_record_sha256")
                for index, item in enumerate(attempts)
            )
        ):
            raise ValueError("BF16 execution ledger closure is incomplete")
    manifests = [
        _mapping(item, "manifest binding")
        for item in _required_sequence(record, "manifests")
    ]
    expected_contexts = (
        [8192, 16384, 32768]
        if stage == PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE
        else [16384]
    )
    if [item.get("context_tokens") for item in manifests] != expected_contexts:
        raise ValueError("coordinator manifest context coverage drift")
    if stage == PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE:
        execution_bundles = [
            _mapping(item, "Q8 execution bundle")
            for item in _required_sequence(execution_record, "bundles")
        ]
    else:
        execution_bundles = [_required_mapping(execution_record, "bundle")]
    if len(execution_bundles) != len(manifests):
        raise ValueError("execution/manifest bundle coverage drift")
    if stage == PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE and (
        execution_record.get("bundles_sha256") != _canonical_sha256(execution_bundles)
        or list(
            _required_sequence(
                _required_mapping(execution_record, "serving_reuse"),
                "context_bundles",
            )
        )
        != execution_bundles
    ):
        raise ValueError("Q8 execution bundle closure drift")
    output_root_uri = _required_string(request, "output_root_uri")
    for item, bundle in zip(manifests, execution_bundles, strict=True):
        manifest = _required_mapping(item, "record")
        validated = _handoff_artifacts._validated_bundle_record(manifest)
        if dict(validated) != dict(manifest):
            raise ValueError("embedded manifest normalization drift")
        if (
            item.get("closed_record_sha256") != manifest.get("closed_record_sha256")
            or item.get("portable_bundle_sha256")
            != manifest.get("portable_bundle_sha256")
            or item.get("file_sha256")
            != sha256(_canonical_json_bytes(manifest, pretty=True)).hexdigest()
            or manifest.get("input_bundle_sha256") != request.get("input_bundle_sha256")
            or manifest.get("context_tokens") != item.get("context_tokens")
            or bundle.get("context_tokens") != item.get("context_tokens")
            or bundle.get("closed_record_sha256") != item.get("closed_record_sha256")
            or bundle.get("portable_bundle_sha256")
            != item.get("portable_bundle_sha256")
        ):
            raise ValueError("embedded manifest binding drift")
        manifest_relative = _required_string(bundle, "manifest_relative_path")
        source_relative = _required_string(bundle, "source_root_relative_path")
        manifest_uri = _canonical_volume_file_uri(
            _required_string(item, "uri"), "manifest uri"
        )
        source_root_uri = _canonical_volume_directory_uri(
            _required_string(item, "source_root_uri"), "source root uri"
        )
        if (
            manifest_uri != _join_volume_uri(output_root_uri, manifest_relative)
            or source_root_uri != _join_volume_uri(output_root_uri, source_relative)
            or source_relative
            != (f"bundles/{item['context_tokens']}-{item['portable_bundle_sha256']}")
            or PurePosixPath(source_relative).name
            != f"{item['context_tokens']}-{item['portable_bundle_sha256']}"
        ):
            raise ValueError("manifest/source URI does not join execution bundle")
        if stage == PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE:
            expected_manifest_relative = (
                f"manifests/{item['context_tokens']}-"
                f"{item['portable_bundle_sha256']}.json"
            )
            if manifest_relative != expected_manifest_relative:
                raise ValueError("Q8 manifest path is not content addressed")
            _q8._validate_publication_manifest_contract(manifest)
        else:
            if (
                manifest_relative
                != f"manifests/{_bf16.PUBLICATION_BF16_HANDOFF_MANIFEST_FILENAME}"
                or bundle.get("manifest_file_sha256") != item.get("file_sha256")
            ):
                raise ValueError("BF16 execution manifest binding drift")
            _bf16._validate_bf16_manifest_contract(manifest)
    if len(_canonical_json_bytes(record, pretty=True)) > (
        PUBLICATION_HANDOFF_CLOSURE_RESULT_BYTES_MAX
    ):
        raise ValueError("compact coordinator result exceeds the Files API cap")


def _require_remote_closure_authorization(
    authorization: object,
    *,
    expected_stage: str,
    expected_output_root_uri: str,
    expected_execution_file_sha256: str,
    expected_input_bundle_sha256: str,
    expected_qualification_closed_record_sha256: str,
) -> PublicationHandoffRemoteClosureAuthorization:
    if not isinstance(authorization, PublicationHandoffRemoteClosureAuthorization):
        raise TypeError("remote handoff closure authorization has the wrong type")
    _validate_closure_result(
        dict(authorization.result_record),
        request=dict(authorization.request_record),
    )
    expected_root = _canonical_volume_directory_uri(
        expected_output_root_uri, "expected_output_root_uri"
    )
    pins = _required_mapping(authorization.result_record, "pins")
    if (
        authorization.stage != expected_stage
        or authorization.output_root_uri != expected_root
        or not hmac.compare_digest(
            authorization.execution_file_sha256,
            _require_sha256(
                expected_execution_file_sha256,
                "expected_execution_file_sha256",
            ),
        )
        or pins.get("input_bundle_sha256")
        != _require_sha256(expected_input_bundle_sha256, "expected_input_bundle_sha256")
        or pins.get("qualification_closed_record_sha256")
        != _require_sha256(
            expected_qualification_closed_record_sha256,
            "expected_qualification_closed_record_sha256",
        )
    ):
        raise ValueError("remote handoff closure authorization binding drift")
    expected_causal = _canonical_sha256(
        {
            "control_plane_status_sha256": authorization.control_plane_status_sha256,
            "coordinator_run_id": authorization.coordinator_run_id,
            "ledger_lineage": dict(
                _required_mapping(authorization.request_record, "ledger_lineage")
            ),
            "request_closed_record_sha256": authorization.request_closed_record_sha256,
            "result_closed_record_sha256": authorization.result_closed_record_sha256,
            "result_file_sha256": authorization.result_file_sha256,
        }
    )
    if expected_causal != authorization.causal_closure_sha256:
        raise ValueError("remote handoff closure causal binding drift")
    return authorization


def _validate_coordinator_terminal_run(
    terminal: Mapping[str, Any],
    *,
    submit_payload: Mapping[str, Any],
    expected_run_id: str,
    expected_stage: str,
) -> str:
    snapshot, canonical = canonical_databricks_submit_payload_snapshot(terminal)
    if _required_run_id(snapshot.get("run_id"), "terminal run_id") != expected_run_id:
        raise ValueError("coordinator terminal response belongs to another run")
    if (
        _required_run_id(
            snapshot.get("original_attempt_run_id"), "original_attempt_run_id"
        )
        != expected_run_id
    ):
        raise ValueError("coordinator did not finish on its original attempt")
    repairs = snapshot.get("repair_history")
    if repairs not in (None, []):
        raise ValueError("coordinator run repair is forbidden")
    raw_tasks = _required_sequence(snapshot, "tasks")
    raw_task = _mapping(raw_tasks[0], "terminal task") if len(raw_tasks) == 1 else {}
    submitted_tasks = _required_sequence(submit_payload, "tasks")
    submitted_task = (
        _mapping(submitted_tasks[0], "submitted task")
        if len(submitted_tasks) == 1
        else {}
    )
    observed_python_task = raw_task.get("spark_python_task")
    submitted_python_task = submitted_task.get("spark_python_task")
    if (
        not isinstance(observed_python_task, Mapping)
        or not isinstance(submitted_python_task, Mapping)
        or dict(observed_python_task) != dict(submitted_python_task)
    ):
        raise ValueError(
            "coordinator terminal spark_python_task differs from submitted task"
        )
    attempt_number = raw_task.get("attempt_number")
    task_run_id = _required_run_id(raw_task.get("run_id"), "terminal task run_id")
    if (
        len(raw_tasks) != 1
        or type(attempt_number) is not int
        or attempt_number != 0
        or task_run_id == expected_run_id
    ):
        raise ValueError(
            "coordinator must finish its distinct child task on attempt zero"
        )
    status = databricks_run_status_record(
        summarize_databricks_run(snapshot, submit_payload=submit_payload)
    )
    if status is None:
        raise ValueError("coordinator terminal response has no sanitized status")
    tasks = _required_sequence(status, "tasks")
    task = _mapping(tasks[0], "coordinator status task") if len(tasks) == 1 else {}
    expected_task_key = f"handoff_closure_{expected_stage}"
    if (
        status.get("terminal") is not True
        or status.get("succeeded") is not True
        or status.get("life_cycle_state") != "TERMINATED"
        or status.get("result_state") != "SUCCESS"
        or status.get("active_task_key") is not None
        or status.get("task_count") != 1
        or task.get("task_key") != expected_task_key
        or task.get("life_cycle_state") != "TERMINATED"
        or task.get("result_state") != "SUCCESS"
        or task.get("node_type_id") != PUBLICATION_HANDOFF_CLOSURE_NODE_TYPE_ID
        or task.get("driver_node_type_id") != PUBLICATION_HANDOFF_CLOSURE_NODE_TYPE_ID
    ):
        raise ValueError("coordinator run is not one successful c5d.4xlarge task")
    return sha256(canonical).hexdigest()


def _verify_source_closure(coordinator: Mapping[str, Any]) -> None:
    source_path = _cluster_path(_required_string(coordinator, "source_closure_uri"))
    content = _read_verified_file_bytes(
        source_path,
        _required_string(coordinator, "cachet_source_tree_sha256"),
        "source closure",
    )
    record = _canonical_json_object_from_bytes(content, "source closure")
    normalized = dict(record)
    normalized["closed_record_sha256"] = ""
    if (
        record.get("record_type") != _PUBLICATION_SOURCE_CLOSURE_V2_RECORD_TYPE
        or record.get("schema_version")
        != _PUBLICATION_SOURCE_CLOSURE_V2_SCHEMA_VERSION
        or record.get("closed_record_sha256") != _canonical_sha256(normalized)
        or _required_mapping(record, "git").get("commit")
        != coordinator.get("source_revision")
    ):
        raise ValueError("source closure identity drift")
    runtime = _required_mapping(record, "runtime")
    if dict(runtime) != _native_v2_source_closure_runtime_identity():
        raise ValueError("source closure runtime identity drift")
    if (
        _required_mapping(runtime, "base_lock").get("sha256")
        != coordinator.get("runtime_lock_sha256")
        or _required_mapping(runtime, "vllm").get("wheel_sha256")
        != coordinator.get("patched_vllm_wheel_sha256")
        or _required_mapping(runtime, "flashinfer").get("patched_wheel_sha256")
        != coordinator.get("patched_flashinfer_wheel_sha256")
        or _required_mapping(runtime, "runtime_closure").get("file_sha256")
        != coordinator.get("runtime_closure_manifest_sha256")
    ):
        raise ValueError("source closure runtime pins drift")
    wheels = [
        _mapping(item, "source closure file")
        for item in _required_sequence(record, "files")
        if _mapping(item, "source closure file").get("role") == "cachet_package_wheel"
    ]
    if len(wheels) != 1 or wheels[0].get("sha256") != coordinator.get(
        "package_wheel_sha256"
    ):
        raise ValueError("source closure package wheel pin drift")


def _native_v2_source_closure_runtime_identity() -> dict[str, Any]:
    return {
        "base_lock": {
            "byte_count": VLLM_RUNTIME_BASE_LOCK_SIZE,
            "distribution_count": VLLM_RUNTIME_BASE_LOCK_DISTRIBUTION_COUNT,
            "hash_count": VLLM_RUNTIME_BASE_LOCK_HASH_COUNT,
            "sha256": VLLM_RUNTIME_BASE_LOCK_SHA256,
        },
        "flashinfer": {
            "manifest_closed_record_sha256": (
                FLASHINFER_PATCHED_MANIFEST_CLOSED_RECORD_SHA256
            ),
            "manifest_file_byte_count": FLASHINFER_PATCHED_MANIFEST_SIZE,
            "manifest_file_sha256": FLASHINFER_PATCHED_MANIFEST_FILE_SHA256,
            "patched_wheel_byte_count": FLASHINFER_PATCHED_WHEEL_SIZE,
            "patched_wheel_sha256": FLASHINFER_PATCHED_WHEEL_SHA256,
            "pristine_wheel_byte_count": FLASHINFER_SOURCE_WHEEL_SIZE,
            "pristine_wheel_sha256": FLASHINFER_SOURCE_WHEEL_SHA256,
        },
        "input_bundle_sha256": GPU_QUALIFICATION_PUBLICATION_INPUT_BUNDLE_SHA256,
        "runtime_closure": {
            "closed_record_sha256": RUNTIME_ARTIFACT_CLOSURE_CLOSED_RECORD_SHA256,
            "file_byte_count": RUNTIME_ARTIFACT_CLOSURE_FILE_SIZE,
            "file_sha256": RUNTIME_ARTIFACT_CLOSURE_FILE_SHA256,
        },
        "source_lock": {
            "byte_count": VLLM_RUNTIME_SOURCE_LOCK_SIZE,
            "distribution_count": VLLM_RUNTIME_SOURCE_LOCK_DISTRIBUTION_COUNT,
            "hash_count": VLLM_RUNTIME_SOURCE_LOCK_HASH_COUNT,
            "sha256": VLLM_RUNTIME_LOCK_SHA256,
        },
        "vllm": {
            "manifest_file_byte_count": VLLM_PATCHED_MANIFEST_SIZE,
            "manifest_file_sha256": VLLM_PATCHED_MANIFEST_SHA256,
            "wheel_byte_count": VLLM_PATCHED_WHEEL_SIZE,
            "wheel_sha256": GPU_QUALIFICATION_PATCHED_WHEEL_SHA256,
        },
    }


def _revalidate_closed_tree(stage: str, root: Path) -> None:
    if stage == PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE:
        _q8.read_publication_latency_handoff_generation_result(root)
    else:
        _bf16.read_publication_bf16_handoff_generation_result(root)


def _reserve_closure_attempt(
    request_authorization: PublicationHandoffClosureRequestAuthorization,
    payload: Mapping[str, Any],
    *,
    reservation_root: str | Path,
) -> Path:
    authorization = require_publication_handoff_closure_request_authorization(
        request_authorization
    )
    request = dict(authorization.request_record)
    root = Path(reservation_root).expanduser().absolute()
    _require_handoff_controller_lease_root(authorization, root)
    _require_no_symlink_ancestors(root, include_leaf=True)
    if root.exists() and not root.is_dir():
        raise ValueError("reservation_root must be a directory")
    root.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_ancestors(root, include_leaf=True)
    request_bytes = _canonical_json_bytes(request, pretty=True)
    payload_bytes = _canonical_json_bytes(payload, pretty=True)
    reservation = {
        "attempt_id": request["attempt_id"],
        "batch_evidence_sha256": authorization.batch_evidence_sha256,
        "closed_record_sha256": "",
        "controller_lease_root_sha256": (
            authorization.controller_lease_root_sha256
        ),
        "qualified_artifact_pins_sha256": (
            authorization.qualified_artifact_pins_sha256
        ),
        "qualified_artifact_pins": dict(
            _required_mapping(request, "qualified_artifact_pins")
        ),
        "qualification_authorization_binding_sha256": (
            authorization.qualification_authorization_binding_sha256
        ),
        "record_type": PUBLICATION_HANDOFF_CLOSURE_RESERVATION_RECORD_TYPE,
        "request_authorization_sha256": authorization.authorization_sha256,
        "request_closed_record_sha256": request["closed_record_sha256"],
        "request_file_sha256": sha256(request_bytes).hexdigest(),
        "schema_version": PUBLICATION_HANDOFF_CLOSURE_SCHEMA_VERSION,
        "submit_payload_sha256": sha256(
            canonical_databricks_submit_payload_snapshot(payload)[1]
        ).hexdigest(),
    }
    reservation["closed_record_sha256"] = _closed_record_sha256(reservation)
    _write_or_require_exact(root / "request.json", request_bytes)
    _write_or_require_exact(root / "submit-payload.json", payload_bytes)
    _write_or_require_exact(
        root / "reservation.json", _canonical_json_bytes(reservation, pretty=True)
    )
    _fsync_directory(root)
    return root


def _existing_local_reservation_root(
    value: str | Path,
    *,
    request_authorization: PublicationHandoffClosureRequestAuthorization,
) -> Path:
    authorization = require_publication_handoff_closure_request_authorization(
        request_authorization
    )
    root = Path(value).expanduser().absolute()
    _require_no_symlink_ancestors(root, include_leaf=True)
    if not root.is_dir():
        raise ValueError("reservation_root must be an existing directory")
    reservation = _read_canonical_json_file(root / "reservation.json", "reservation")
    if (
        reservation.get("record_type")
        != PUBLICATION_HANDOFF_CLOSURE_RESERVATION_RECORD_TYPE
        or reservation.get("schema_version")
        != PUBLICATION_HANDOFF_CLOSURE_SCHEMA_VERSION
        or reservation.get("closed_record_sha256") != _closed_record_sha256(reservation)
    ):
        raise ValueError("coordinator reservation envelope drift")
    request_path = root / "request.json"
    payload_path = root / "submit-payload.json"
    _require_regular_file_no_follow(request_path, "reserved request")
    _require_regular_file_no_follow(payload_path, "reserved submit payload")
    request_bytes = request_path.read_bytes()
    payload_bytes = payload_path.read_bytes()
    request = _canonical_json_object_from_bytes(request_bytes, "reserved request")
    payload = _canonical_json_object_from_bytes(
        payload_bytes, "reserved submit payload"
    )
    _snapshot, canonical_payload = canonical_databricks_submit_payload_snapshot(payload)
    if (
        reservation.get("attempt_id") != request.get("attempt_id")
        or reservation.get("controller_lease_root_sha256")
        != authorization.controller_lease_root_sha256
        or reservation.get("batch_evidence_sha256")
        != authorization.batch_evidence_sha256
        or reservation.get("request_authorization_sha256")
        != authorization.authorization_sha256
        or reservation.get("qualified_artifact_pins_sha256")
        != authorization.qualified_artifact_pins_sha256
        or reservation.get("qualified_artifact_pins")
        != request.get("qualified_artifact_pins")
        or reservation.get("qualification_authorization_binding_sha256")
        != authorization.qualification_authorization_binding_sha256
        or reservation.get("request_closed_record_sha256")
        != request.get("closed_record_sha256")
        or reservation.get("request_file_sha256") != sha256(request_bytes).hexdigest()
        or reservation.get("submit_payload_sha256")
        != sha256(canonical_payload).hexdigest()
    ):
        raise ValueError("coordinator reservation source binding drift")
    if request != dict(authorization.request_record):
        raise ValueError("reserved request differs from live request authority")
    return root


def _require_handoff_controller_lease_root(
    authorization: PublicationHandoffClosureRequestAuthorization,
    value: str | Path,
) -> Path:
    root = Path(value).expanduser().absolute()
    if root != authorization.controller_lease_root or (
        _controller_path_sha256(
            root,
            domain="cachet.publication.handoff_closure.controller_lease.v2",
        )
        != authorization.controller_lease_root_sha256
    ):
        raise ValueError(
            "handoff closure reservation_root differs from singleton authority"
        )
    return root


def _coordinator_config_from_record(
    record: Mapping[str, Any],
) -> PublicationHandoffClosureCoordinatorConfig:
    config = PublicationHandoffClosureCoordinatorConfig(
        runner_python_file=_required_string(record, "runner_python_file"),
        package_wheel_uri=_required_string(record, "package_wheel_uri"),
        package_wheel_sha256=_required_string(record, "package_wheel_sha256"),
        runtime_lock_uri=_required_string(record, "runtime_lock_uri"),
        runtime_lock_sha256=_required_string(record, "runtime_lock_sha256"),
        patched_vllm_wheel_uri=_required_string(record, "patched_vllm_wheel_uri"),
        patched_vllm_wheel_sha256=_required_string(record, "patched_vllm_wheel_sha256"),
        patched_flashinfer_wheel_uri=_required_string(
            record, "patched_flashinfer_wheel_uri"
        ),
        patched_flashinfer_wheel_sha256=_required_string(
            record, "patched_flashinfer_wheel_sha256"
        ),
        runtime_closure_manifest_uri=_required_string(
            record, "runtime_closure_manifest_uri"
        ),
        runtime_closure_manifest_sha256=_required_string(
            record, "runtime_closure_manifest_sha256"
        ),
        source_closure_uri=_required_string(record, "source_closure_uri"),
        cachet_source_tree_sha256=_required_string(record, "cachet_source_tree_sha256"),
        request_root_uri=_required_string(record, "request_root_uri"),
        source_revision=_required_string(record, "source_revision"),
        single_user_name=_required_string(record, "single_user_name"),
        runtime_venv_dir=_required_string(record, "runtime_venv_dir"),
        runner_sha256=_required_string(record, "runner_sha256"),
        node_type_id=_required_string(record, "node_type_id"),
        spark_version=_required_string(record, "spark_version"),
        data_security_mode=_required_string(record, "data_security_mode"),
        timeout_seconds=_required_int(record, "timeout_seconds"),
        custom_tags=dict(_required_mapping(record, "custom_tags")),
    )
    if dict(record) != config.to_record():
        raise ValueError("coordinator config record keys or normalization drift")
    return config


def _cluster_path(uri: str) -> Path:
    canonical = _canonical_volume_uri(uri, field_name="volume URI", require_file=None)
    return Path("/Volumes") / canonical.removeprefix("dbfs:/Volumes/")


def _canonical_volume_file_uri(value: str, field_name: str) -> str:
    return _canonical_volume_uri(value, field_name=field_name, require_file=True)


def _canonical_volume_directory_uri(value: str, field_name: str) -> str:
    return _canonical_volume_uri(value, field_name=field_name, require_file=False)


def _canonical_volume_uri(
    value: str,
    *,
    field_name: str,
    require_file: bool | None,
) -> str:
    if not isinstance(value, str) or not value.startswith("dbfs:/Volumes/"):
        raise ValueError(f"{field_name} must be a canonical dbfs:/Volumes URI")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field_name} cannot contain control characters")
    if any(character in value for character in ("%", "?", "#", "\\")):
        raise ValueError(f"{field_name} cannot contain URL syntax or backslashes")
    raw = value.removeprefix("dbfs:")
    path = PurePosixPath(raw)
    if (
        path.as_posix() != raw
        or path.parts[:2] != ("/", "Volumes")
        or len(path.parts) < 5
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise ValueError(f"{field_name} is not canonical or confined")
    if require_file is True and len(path.parts) < 6:
        raise ValueError(f"{field_name} must name a file beneath a UC volume")
    if value.endswith("/"):
        raise ValueError(f"{field_name} cannot end with a slash")
    return f"dbfs:{path.as_posix()}"


def _join_volume_uri(root: str, relative: str) -> str:
    canonical_root = _canonical_volume_directory_uri(root, "volume root")
    relative_path = PurePosixPath(relative)
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise ValueError("relative volume path is not confined")
    return _canonical_volume_uri(
        f"{canonical_root}/{relative_path.as_posix()}",
        field_name="joined volume URI",
        require_file=None,
    )


def _read_verified_file_bytes(path: Path, expected_sha256: str, label: str) -> bytes:
    _require_regular_file_no_follow(path, label)
    content = path.read_bytes()
    if not hmac.compare_digest(
        sha256(content).hexdigest(),
        _require_sha256(expected_sha256, f"{label} SHA-256"),
    ):
        raise ValueError(f"{label} SHA-256 drift")
    return content


def _canonical_json_object_from_bytes(content: bytes, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be UTF-8 JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    record = dict(decoded)
    if content != _canonical_json_bytes(record, pretty=True):
        raise ValueError(f"{label} must be canonical newline-terminated JSON")
    return record


def _canonical_closed_record_from_bytes(content: bytes, label: str) -> dict[str, Any]:
    record = _canonical_json_object_from_bytes(content, label)
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError(f"{label} closed-record SHA-256 drift")
    return record


def _read_canonical_json_file(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file_no_follow(path, label)
    return _canonical_json_object_from_bytes(path.read_bytes(), label)


def _write_or_require_exact(path: Path, content: bytes) -> None:
    path = Path(path).expanduser().absolute()
    _require_no_symlink_ancestors(path, include_leaf=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_ancestors(path.parent, include_leaf=True)
    if path.exists():
        _require_regular_file_no_follow(path, "existing immutable file")
        if path.read_bytes() != content:
            raise FileExistsError(f"immutable file contains different bytes: {path}")
        return
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _require_no_symlink_ancestors(path: Path, *, include_leaf: bool) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    target = absolute if include_leaf else absolute.parent
    for candidate in reversed((target, *target.parents)):
        try:
            mode = os.lstat(candidate).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"path cannot traverse a symlink: {candidate}")


def _require_regular_file_no_follow(path: Path, label: str) -> None:
    _require_no_symlink_ancestors(path, include_leaf=True)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"{label} must be an existing regular file") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be an existing regular file")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_parameter_bytes(parameters: Sequence[str]) -> None:
    if not parameters or any(
        not isinstance(item, str) or not item for item in parameters
    ):
        raise TypeError("coordinator parameters must be non-empty strings")
    encoded = _canonical_json_bytes(list(parameters), pretty=False)
    if len(encoded) > PUBLICATION_HANDOFF_CLOSURE_PARAMETER_BYTES_MAX or any(
        len(item.encode("utf-8")) > PUBLICATION_HANDOFF_CLOSURE_PARAMETER_BYTES_MAX
        for item in parameters
    ):
        raise ValueError("coordinator parameters exceed the 9,500-byte safety bound")


def _required_stage(value: Any) -> str:
    if value not in PUBLICATION_HANDOFF_CLOSURE_STAGES:
        raise ValueError("handoff closure stage must be q8 or bf16")
    return cast(str, value)


def _required_attempt_id(value: Any) -> str:
    if not isinstance(value, str) or _ATTEMPT_ID_RE.fullmatch(value) is None:
        raise ValueError("attempt_id must be one safe lowercase identifier")
    return value


def _required_run_id(value: Any, field_name: str) -> str:
    if type(value) is int and value > 0:
        return str(value)
    if (
        isinstance(value, str)
        and value.isascii()
        and value.isdigit()
        and not value.startswith("0")
    ):
        return value
    raise ValueError(f"{field_name} must be a canonical positive decimal run ID")


def _required_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    return _mapping(value.get(field_name), field_name)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _required_sequence(value: Mapping[str, Any], field_name: str) -> list[Any]:
    raw = value.get(field_name)
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a list")
    return raw


def _required_string(value: Mapping[str, Any], field_name: str) -> str:
    return _nonempty(value.get(field_name), field_name)


def _nonempty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _required_int(value: Mapping[str, Any], field_name: str) -> int:
    raw = value.get(field_name)
    if type(raw) is not int:
        raise ValueError(f"{field_name} must be an integer")
    return raw


def _require_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _canonical_json_bytes(value: Any, *, pretty: bool) -> bytes:
    if pretty:
        return (
            json.dumps(
                value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n"
        ).encode("utf-8")
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return sha256(_canonical_json_bytes(value, pretty=False)).hexdigest()


def _closed_record_sha256(record: Mapping[str, Any]) -> str:
    normalized = dict(record)
    normalized["closed_record_sha256"] = ""
    return _canonical_sha256(normalized)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Close publication handoffs on CPU")
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render-submit")
    render.add_argument("--request", required=True)
    reserve = subparsers.add_parser("reserve-submit")
    reserve.add_argument("--request", required=True)
    reserve.add_argument("--reservation-root", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--reservation-root", required=True)
    run = subparsers.add_parser("run-coordinator")
    run.add_argument("--request-uri", required=True)
    run.add_argument("--request-file-sha256", required=True)
    run.add_argument("--expected-request-closed-record-sha256", required=True)
    run.add_argument("--coordinator-run-id", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in {"render-submit", "reserve-submit", "collect"}:
            raise TypeError(
                "controller launch/recovery requires an in-memory "
                "PublicationHandoffClosureRequestAuthorization; raw request files "
                "are deliberately nonauthorizing"
            )
        request_uri = _canonical_volume_file_uri(args.request_uri, "request_uri")
        raw = _read_verified_file_bytes(
            _cluster_path(request_uri),
            args.request_file_sha256,
            "coordinator request",
        )
        if len(raw) > PUBLICATION_HANDOFF_CLOSURE_REQUEST_BYTES_MAX:
            raise ValueError("coordinator request exceeds its byte cap")
        request = _canonical_closed_record_from_bytes(raw, "coordinator request")
        _validate_closure_request(request)
        if request.get("request_uri") != request_uri or request.get(
            "closed_record_sha256"
        ) != _require_sha256(
            args.expected_request_closed_record_sha256,
            "expected request closed-record SHA-256",
        ):
            raise ValueError("coordinator request parameter digest drift")
        result = run_publication_handoff_closure_coordinator(
            request,
            coordinator_run_id=args.coordinator_run_id,
        )
        print(
            json.dumps(
                {
                    "closed_record_sha256": result["closed_record_sha256"],
                    "ok": True,
                    "stage": result["stage"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"error": str(exc), "error_type": type(exc).__name__, "ok": False},
                sort_keys=True,
            )
        )
        return 1


__all__ = [
    "PUBLICATION_HANDOFF_CLOSURE_BF16_STAGE",
    "PUBLICATION_HANDOFF_CLOSURE_CONFIG_RECORD_TYPE",
    "PUBLICATION_HANDOFF_CLOSURE_NODE_TYPE_ID",
    "PUBLICATION_HANDOFF_CLOSURE_PARAMETER_BYTES_MAX",
    "PUBLICATION_HANDOFF_CLOSURE_Q8_STAGE",
    "PUBLICATION_HANDOFF_CLOSURE_REQUEST_BYTES_MAX",
    "PUBLICATION_HANDOFF_CLOSURE_REQUEST_RECORD_TYPE",
    "PUBLICATION_HANDOFF_CLOSURE_RESULT_BYTES_MAX",
    "PUBLICATION_HANDOFF_CLOSURE_RESULT_RECORD_TYPE",
    "PUBLICATION_HANDOFF_CLOSURE_RUNNER_FILENAME",
    "PUBLICATION_HANDOFF_CLOSURE_RUNNER_SCRIPT",
    "PUBLICATION_HANDOFF_CLOSURE_RUNNER_SHA256",
    "PUBLICATION_HANDOFF_CLOSURE_SPARK_VERSION",
    "PUBLICATION_HANDOFF_CLOSURE_TIMEOUT_SECONDS",
    "PublicationHandoffClosureRequestAuthorization",
    "PublicationHandoffClosureCoordinatorConfig",
    "PublicationHandoffRemoteClosureAuthorization",
    "build_bf16_handoff_closure_request",
    "build_q8_handoff_closure_request",
    "collect_publication_handoff_closure",
    "publication_handoff_closure_request_root_uri",
    "render_publication_handoff_closure_submit_payload",
    "require_bf16_handoff_remote_closure_authorization",
    "require_publication_handoff_closure_request_authorization",
    "require_q8_handoff_remote_closure_authorization",
    "require_q8_handoff_remote_closure_predecessor_authorization",
    "reserve_and_submit_publication_handoff_closure",
    "run_publication_handoff_closure_coordinator",
    "write_publication_handoff_closure_runner_script",
]


if __name__ == "__main__":  # pragma: no cover - exercised by Databricks.
    raise SystemExit(main())
