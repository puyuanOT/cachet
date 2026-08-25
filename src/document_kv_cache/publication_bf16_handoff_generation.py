"""Generate the governed BF16 handoff prerequisite for publication latency.

The auxiliary BF16 arm consumes one closed 16k bundle containing the exact
32-example suite for every publication dataset.  Generation is deliberately a
separate, capability-gated cloud phase: sixteen independent one-GPU L40S jobs
write content-addressed payloads, direct ``runs/get`` responses close their
ledger actuals, and a CPU coordinator creates the portable manifest without
copying KV payload bytes.
"""

from __future__ import annotations

import argparse
import gc
import hmac
import importlib
import json
import math
import os
import re
import shutil
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

if TYPE_CHECKING:
    from document_kv_cache.publication_handoff_closure_coordinator import (
        PublicationHandoffRemoteClosureAuthorization,
    )

from document_kv_cache._benchmark_datasets import _example_from_record
from document_kv_cache.artifact_identity import TokenContract, token_ids_digest
from document_kv_cache.benchmark_handoffs import (
    enrich_benchmark_records_with_handoffs,
    generate_benchmark_handoff_bundles,
)
from document_kv_cache.benchmarks import (
    DOCUMENT_KV_HANDOFF_JSON_PARAM,
    DOCUMENT_KV_PAYLOAD_URI_PARAM,
    SUPPORTED_V1_DATASETS,
    benchmark_cache_prefix_segments,
)
from document_kv_cache.databricks_job import (
    DEFAULT_DATABRICKS_DATA_SECURITY_MODE,
    DEFAULT_DATABRICKS_SPARK_VERSION,
)
from document_kv_cache.databricks_resource_ledger import (
    MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS,
    MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS,
    DatabricksClusterHourLedger,
    DatabricksClusterHourReservation,
    DatabricksBatchReservationAuthorization,
    DatabricksLedgerPrefix,
    DatabricksRunAttemptReservationRequest,
    canonical_databricks_submit_payload_snapshot,
    databricks_ledger_path_sha256,
    databricks_ledger_prefix,
    databricks_ledger_prefix_from_record,
    read_databricks_cluster_hour_ledger_json,
    replay_databricks_run_attempt_batch_authorization_json,
    record_databricks_verified_run_terminal_actual_json,
    reserve_databricks_run_attempt_json,
    reserve_databricks_run_attempt_batch_authorized_json,
    require_databricks_batch_reservation_authorization,
    require_databricks_batch_terminal_closure,
    require_databricks_ledger_prefix,
    require_databricks_publication_batch_admission,
)
from document_kv_cache.databricks_runs import (
    DatabricksURLOpener,
    DatabricksWorkspaceConfig,
    bind_databricks_run_idempotency_token,
    databricks_run_status_record,
    get_databricks_run,
    reserve_and_submit_databricks_run,
    require_databricks_run_idempotency_token,
    resume_pre_reserved_databricks_run,
    submit_pre_reserved_databricks_run,
    summarize_databricks_run,
)
from document_kv_cache.engine_protocol import (
    KVKeyPositionEncoding,
    KVLayout,
    KVPayloadAxisOrder,
    KVStorageLayout,
)
from document_kv_cache.gpu_qualification import (
    GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256,
    GPUQualificationArtifactPins,
    GPUQualificationSelection,
    canonical_gpu_qualification_json,
    validate_gpu_qualification_evidence_record,
)
from document_kv_cache.gpu_qualification_databricks import (
    GPUQualificationLaunchAuthorization,
    require_gpu_qualification_launch_authorization,
)
from document_kv_cache.main_latency_inputs import (
    MAIN_LATENCY_ADD_SPECIAL_TOKENS,
    MAIN_LATENCY_EXAMPLES_PER_DATASET,
    MAIN_LATENCY_PROVENANCE_FILENAME,
    MAIN_LATENCY_TOKENIZER_ID,
    MAIN_LATENCY_TOKENIZER_REVISION,
    MainLatencyTokenizer,
    PreparedMainLatencyInputs,
    load_main_latency_tokenizer,
    verify_main_latency_inputs,
)
from document_kv_cache.model_profiles import (
    QWEN3_4B_INSTRUCT_HF_MODEL_ID,
    QWEN3_4B_ROPE_ROTARY_DIM,
    QWEN3_4B_ROPE_THETA,
    layout_for_model,
)
from document_kv_cache.models import CacheGenerationMethod
from document_kv_cache.publication_campaign import (
    PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
    PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS,
    PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND,
    PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS,
)
from document_kv_cache.publication_handoff_artifacts import (
    StagedPublicationLatencyHandoffBundle,
    close_publication_latency_handoff_bundle,
    read_publication_latency_handoff_bundle,
    stage_publication_latency_handoff_bundle,
    validate_publication_latency_handoff_bundle,
    write_publication_latency_handoff_bundle,
)
from document_kv_cache.publication_latency_handoff_generation import (
    PUBLICATION_LATENCY_HANDOFF_GENERATOR_GPU_MODEL,
    PUBLICATION_LATENCY_HANDOFF_GENERATOR_HARDWARE_TARGET,
    PUBLICATION_LATENCY_HANDOFF_GENERATOR_MODEL_DTYPE,
    PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID,
    PUBLICATION_LATENCY_HANDOFF_GENERATOR_QUANTIZATION,
    PUBLICATION_LATENCY_HANDOFF_RUNNER_SCRIPT,
    PUBLICATION_LATENCY_HANDOFF_TRANSFORMERS_VERSION,
    PublicationLatencyGeneratorHardwareQualification,
    _post_close_jsonl_objects,
    _post_close_rebased_generated_row,
    _post_close_regular_file_inventory,
)
from document_kv_cache.serving_env import VLLM_RUNTIME_LOCK_SHA256
from document_kv_cache.storage import local_path
from document_kv_cache.transformers_generator import (
    CACHET_TRANSFORMERS_ADD_SPECIAL_TOKENS_ENV,
    CACHET_TRANSFORMERS_CACHE_AXIS_ORDER_ENV,
    CACHET_TRANSFORMERS_DEVICE_ENV,
    CACHET_TRANSFORMERS_DEVICE_MAP_ENV,
    CACHET_TRANSFORMERS_MODEL_ID_ENV,
    CACHET_TRANSFORMERS_MODEL_REVISION_ENV,
    CACHET_TRANSFORMERS_PRE_ROPE_ENV,
    CACHET_TRANSFORMERS_QUANTIZATION_CONFIG_JSON_ENV,
    CACHET_TRANSFORMERS_QUANTIZATION_ENV,
    CACHET_TRANSFORMERS_TOKENIZER_ID_ENV,
    CACHET_TRANSFORMERS_TOKENIZER_REVISION_ENV,
    CACHET_TRANSFORMERS_TORCH_DTYPE_ENV,
    CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV,
    build_pre_rope_transformers_kv_chunk_generator,
)
from document_kv_cache.workflow import KVChunkGenerator


PUBLICATION_BF16_HANDOFF_CONTEXT_TOKENS: Final = 16_384
PUBLICATION_BF16_HANDOFF_WORKER_COUNT: Final = PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS
PUBLICATION_BF16_HANDOFF_TASK_COUNT: Final = (
    len(SUPPORTED_V1_DATASETS) * PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
)
PUBLICATION_BF16_HANDOFF_TASK_TIMEOUT_SECONDS: Final = 5 * 60 * 60
PUBLICATION_BF16_HANDOFF_MAX_RESERVED_GPU_HOURS: Final = (
    PUBLICATION_BF16_HANDOFF_WORKER_COUNT
    * PUBLICATION_BF16_HANDOFF_TASK_TIMEOUT_SECONDS
    / 3600.0
)
PUBLICATION_BF16_HANDOFF_DTYPE: Final = "bfloat16"
PUBLICATION_BF16_HANDOFF_ARM_ID: Final = "vanilla"
PUBLICATION_BF16_HANDOFF_PLAN_RECORD_TYPE: Final = (
    "cachet.publication_bf16_handoff_generation_plan.v1"
)
PUBLICATION_BF16_HANDOFF_WORKER_PAYLOAD_RECORD_TYPE: Final = (
    "cachet.publication_bf16_handoff_worker_payload.v1"
)
PUBLICATION_BF16_HANDOFF_WORKER_RESULT_RECORD_TYPE: Final = (
    "cachet.publication_bf16_handoff_worker_result.v1"
)
PUBLICATION_BF16_HANDOFF_ATTESTATION_RECORD_TYPE: Final = (
    "cachet.publication_bf16_handoff_databricks_execution.v1"
)
PUBLICATION_BF16_HANDOFF_EXECUTION_RECORD_TYPE: Final = (
    "cachet.publication_bf16_handoff_generation_execution.v1"
)
PUBLICATION_BF16_HANDOFF_SCHEMA_VERSION: Final = 1
PUBLICATION_BF16_HANDOFF_EXECUTION_MODE: Final = "distributed_16x_l40s_qualified"
PUBLICATION_BF16_HANDOFF_EXECUTION_FILENAME: Final = (
    "publication-bf16-handoff-generation.execution.json"
)
PUBLICATION_BF16_HANDOFF_MANIFEST_FILENAME: Final = (
    "publication-bf16-handoff-16384.manifest.json"
)
PUBLICATION_BF16_HANDOFF_ATTESTATION_DIRECTORY: Final = "databricks-attestations"
PUBLICATION_BF16_HANDOFF_RUNNER_FILENAME: Final = (
    "publication_bf16_handoff_generation_runner.py"
)
PUBLICATION_BF16_HANDOFF_RUNNER_SCRIPT: Final = (
    PUBLICATION_LATENCY_HANDOFF_RUNNER_SCRIPT.replace(
        "publication_latency_handoff_generation",
        "publication_bf16_handoff_generation",
    ).replace(
        "latency handoff runner",
        "BF16 handoff runner",
    )
)
PUBLICATION_BF16_HANDOFF_RUNNER_SHA256: Final = sha256(
    PUBLICATION_BF16_HANDOFF_RUNNER_SCRIPT.encode("utf-8")
).hexdigest()

_PLAN_ORDER_DOMAIN = "cachet.publication.bf16_handoff.lpt.v1"
_SHA256_LENGTH = 64
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")
_SUBMISSION_AUTHORIZATION_ISSUER = object()
_WORKER_AUTHORIZATION_ISSUER = object()
_SERVING_AUTHORIZATION_ISSUER = object()
_REMOTE_CLOSURE_LEDGER_ISSUER = object()
_POST_CLOSE_REPLAY_ISSUER = object()


class PublicationBF16GeneratorFactory(Protocol):
    """Construct one persistent GPU-resident generator per worker."""

    def __call__(self, worker_index: int) -> KVChunkGenerator: ...


@dataclass(frozen=True, slots=True)
class PublicationBF16HandoffExecutionConfig:
    """Frozen model, tokenizer, BF16 layout, and generator implementation pins."""

    layout: KVLayout
    model_revision: str
    generator_version: str
    vllm_bitsandbytes_loader_source_sha256: str
    tokenizer_id: str = MAIN_LATENCY_TOKENIZER_ID
    tokenizer_revision: str = MAIN_LATENCY_TOKENIZER_REVISION
    generator_family: str = "transformers"
    generator_device_map: str = "auto"
    generator_quantization: str = PUBLICATION_LATENCY_HANDOFF_GENERATOR_QUANTIZATION
    generator_model_dtype: str = PUBLICATION_LATENCY_HANDOFF_GENERATOR_MODEL_DTYPE
    generator_cache_axis_order: str = "head_major"
    generator_trust_remote_code: bool = False
    generator_add_special_tokens: bool = False
    generator_quantization_config: Mapping[str, Any] = field(
        default_factory=lambda: {
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_storage": "uint8",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "load_in_4bit": True,
        }
    )
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    align_bytes: int = 4096

    def __post_init__(self) -> None:
        if not isinstance(self.layout, KVLayout):
            raise TypeError("layout must be KVLayout")
        self.layout.validate()
        if self.layout.dtype not in {"bf16", "bfloat16"}:
            raise ValueError("precision handoffs require BF16 KV")
        if (
            not self.layout.pre_rope
            or self.layout.key_position_encoding != KVKeyPositionEncoding.PRE_ROPE
        ):
            raise ValueError("precision handoffs require pre-RoPE keys")
        if (
            self.layout.shares_kv_storage
            or self.layout.storage_layout != KVStorageLayout.SEPARATE_KEY_VALUE
        ):
            raise ValueError("BF16 pre-RoPE handoffs require separate K/V storage")
        expected_layout = layout_for_model(
            QWEN3_4B_INSTRUCT_HF_MODEL_ID,
            dtype=PUBLICATION_BF16_HANDOFF_DTYPE,
            pre_rope=True,
            rope_theta=QWEN3_4B_ROPE_THETA,
            rope_rotary_dim=QWEN3_4B_ROPE_ROTARY_DIM,
            shares_kv_storage=False,
            storage_layout=KVStorageLayout.SEPARATE_KEY_VALUE,
            payload_axis_order=KVPayloadAxisOrder.TOKEN_MAJOR,
        )
        if self.layout != expected_layout:
            raise ValueError("BF16 layout must be the exact frozen Qwen3-4B layout")
        for field_name in (
            "model_revision",
            "generator_version",
            "tokenizer_id",
            "tokenizer_revision",
            "generator_family",
            "generator_device_map",
            "generator_quantization",
            "generator_model_dtype",
            "generator_cache_axis_order",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value == "unresolved":
                raise ValueError(f"{field_name} must be pinned")
        if self.tokenizer_id != MAIN_LATENCY_TOKENIZER_ID:
            raise ValueError("tokenizer_id must match the publication input bundle")
        if self.tokenizer_revision != MAIN_LATENCY_TOKENIZER_REVISION:
            raise ValueError("tokenizer_revision must match the publication inputs")
        if self.model_revision != MAIN_LATENCY_TOKENIZER_REVISION:
            raise ValueError("model_revision must match the frozen Qwen3 revision")
        if self.generator_version != PUBLICATION_LATENCY_HANDOFF_TRANSFORMERS_VERSION:
            raise ValueError("generator_version must match the frozen runtime lock")
        if self.vllm_bitsandbytes_loader_source_sha256 != (
            GPU_QUALIFICATION_BITSANDBYTES_LOADER_SHA256
        ):
            raise ValueError("vLLM BitsAndBytes loader source is not qualified")
        expected_quantization = {
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_storage": "uint8",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "load_in_4bit": True,
        }
        if (
            self.generator_family != "transformers"
            or self.generator_device_map != "auto"
            or self.generator_quantization
            != PUBLICATION_LATENCY_HANDOFF_GENERATOR_QUANTIZATION
            or self.generator_model_dtype != "bfloat16"
            or self.generator_cache_axis_order != "head_major"
            or self.generator_trust_remote_code is not False
            or self.generator_add_special_tokens is not False
            or dict(self.generator_quantization_config) != expected_quantization
        ):
            raise ValueError("BF16 generator configuration drift")
        for field_name in ("tensor_parallel_size", "pipeline_parallel_size"):
            if (
                type(getattr(self, field_name)) is not int
                or getattr(self, field_name) != 1
            ):
                raise ValueError(f"{field_name} must equal 1")
        if type(self.align_bytes) is not int or self.align_bytes != 4096:
            raise ValueError("align_bytes must equal the frozen 4096-byte alignment")
        object.__setattr__(
            self,
            "generator_quantization_config",
            MappingProxyType(dict(self.generator_quantization_config)),
        )


@dataclass(frozen=True, slots=True)
class DatabricksPublicationBF16HandoffJobConfig:
    """Exact bootstrap artifacts and one-task L40S job settings."""

    runner_python_file: str
    worker_payload_uri_template: str
    package_wheel_uri: str
    package_wheel_sha256: str
    runtime_lock_uri: str
    runtime_lock_sha256: str
    patched_vllm_wheel_uri: str
    patched_vllm_wheel_sha256: str
    source_revision: str
    cachet_source_tree_sha256: str
    runner_sha256: str = PUBLICATION_BF16_HANDOFF_RUNNER_SHA256
    runtime_venv_dir_template: str = (
        "/local_disk0/cachet-bf16-handoff-runtime-{worker_index}"
    )
    run_name: str = "cachet-vllm-0271-bf16-handoff-generation"
    run_timeout_seconds: int = PUBLICATION_BF16_HANDOFF_TASK_TIMEOUT_SECONDS
    task_max_retries: int = 0
    node_type_id: str = PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID
    spark_version: str = DEFAULT_DATABRICKS_SPARK_VERSION
    data_security_mode: str = DEFAULT_DATABRICKS_DATA_SECURITY_MODE
    single_user_name: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "runner_python_file",
            "worker_payload_uri_template",
            "package_wheel_uri",
            "runtime_lock_uri",
            "patched_vllm_wheel_uri",
            "runtime_venv_dir_template",
            "run_name",
            "spark_version",
            "data_security_mode",
        ):
            _nonempty_string(getattr(self, field_name), field_name)
        for field_name in (
            "package_wheel_sha256",
            "runtime_lock_sha256",
            "patched_vllm_wheel_sha256",
            "cachet_source_tree_sha256",
            "runner_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        _resolved_source_revision(self.source_revision)
        if "{worker_index}" not in self.worker_payload_uri_template or (
            "{worker_index}" not in self.runtime_venv_dir_template
        ):
            raise ValueError("worker URI and venv templates require {worker_index}")
        if self.runner_sha256 != PUBLICATION_BF16_HANDOFF_RUNNER_SHA256:
            raise ValueError("BF16 runner hash differs from reviewed source")
        if self.runtime_lock_sha256 != VLLM_RUNTIME_LOCK_SHA256:
            raise ValueError("runtime lock differs from the frozen publication lock")
        if self.run_timeout_seconds != PUBLICATION_BF16_HANDOFF_TASK_TIMEOUT_SECONDS:
            raise ValueError("BF16 producer timeout must equal five hours")
        if self.task_max_retries != 0:
            raise ValueError("BF16 producer retries must be disabled")
        if self.node_type_id != PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID:
            raise ValueError("BF16 producers require g6e.4xlarge L40S")
        if self.data_security_mode == "SINGLE_USER" and not self.single_user_name:
            raise ValueError("SINGLE_USER requires single_user_name")


@dataclass(frozen=True, slots=True)
class PublicationBF16HandoffAttestationBinding:
    worker_index: int
    path: Path
    file_sha256: str
    closed_record_sha256: str

    def __post_init__(self) -> None:
        _worker_index(self.worker_index)
        object.__setattr__(self, "path", Path(self.path).expanduser().absolute())
        _require_sha256(self.file_sha256, "file_sha256")
        _require_sha256(self.closed_record_sha256, "closed_record_sha256")


@dataclass(frozen=True, slots=True, init=False)
class PublicationBF16HandoffSubmissionAuthorization:
    """Ephemeral authority proving the durable exact-16 BF16 phase admission."""

    batch_authorization: DatabricksBatchReservationAuthorization
    phase_lease_root: Path
    phase_lease_root_sha256: str
    phase_lease_file_sha256: str
    phase_lease_closed_record_sha256: str
    batch_marker_file_sha256: str
    batch_marker_closed_record_sha256: str
    q8_remote_closure_binding_sha256: str
    _q8_remote_closure_binding_canonical_bytes: bytes
    durable_output_root: str

    def __init__(
        self,
        *,
        batch_authorization: DatabricksBatchReservationAuthorization,
        phase_lease_root: str | Path,
        q8_remote_closure_binding: Mapping[str, Any],
        durable_output_root: str,
        _issuer: object,
    ) -> None:
        if _issuer is not _SUBMISSION_AUTHORIZATION_ISSUER:
            raise TypeError(
                "BF16 submission authority requires the durable phase issuer"
            )
        if not isinstance(
            batch_authorization, DatabricksBatchReservationAuthorization
        ):
            raise TypeError("batch_authorization has the wrong type")
        root = Path(phase_lease_root).expanduser().absolute()
        binding = _json_object(
            _canonical_json_bytes(q8_remote_closure_binding, pretty=False),
            field_name="Q8 remote closure binding",
        )
        output_root = _normalized_bf16_durable_output_root(durable_output_root)
        lease, marker = _validate_bf16_submission_phase_files(
            root,
            batch_authorization,
            q8_remote_closure_binding=binding,
            durable_output_root=output_root,
        )
        root = root.resolve(strict=True)
        object.__setattr__(self, "batch_authorization", batch_authorization)
        object.__setattr__(self, "phase_lease_root", root)
        object.__setattr__(
            self,
            "phase_lease_root_sha256",
            _bf16_phase_lease_root_sha256(root),
        )
        object.__setattr__(
            self,
            "phase_lease_file_sha256",
            _file_sha256(root / "phase-lease.json"),
        )
        object.__setattr__(
            self,
            "phase_lease_closed_record_sha256",
            _required_string(lease, "closed_record_sha256"),
        )
        object.__setattr__(
            self,
            "batch_marker_file_sha256",
            _file_sha256(root / "batch-reserved.json"),
        )
        object.__setattr__(
            self,
            "batch_marker_closed_record_sha256",
            _required_string(marker, "closed_record_sha256"),
        )
        object.__setattr__(
            self,
            "q8_remote_closure_binding_sha256",
            _canonical_sha256(binding),
        )
        object.__setattr__(
            self,
            "_q8_remote_closure_binding_canonical_bytes",
            _canonical_json_bytes(binding, pretty=False),
        )
        object.__setattr__(self, "durable_output_root", output_root)

    @property
    def q8_remote_closure_binding(self) -> Mapping[str, Any]:
        return MappingProxyType(
            _json_object(
                self._q8_remote_closure_binding_canonical_bytes,
                field_name="authorized Q8 remote closure binding",
            )
        )


@dataclass(frozen=True, slots=True, init=False)
class PublicationBF16HandoffWorkerAuthorization:
    """Ephemeral authority issued only after one live direct ``runs/get`` join."""

    binding: PublicationBF16HandoffAttestationBinding
    attempt_id: str
    ledger_id: str
    ledger_path_sha256: str
    producer_batch_prefix: DatabricksLedgerPrefix
    control_plane_status_sha256: str

    def __init__(
        self,
        *,
        binding: PublicationBF16HandoffAttestationBinding,
        attempt_id: str,
        ledger_id: str,
        ledger_path_sha256: str,
        producer_batch_prefix: DatabricksLedgerPrefix,
        control_plane_status_sha256: str,
        _issuer: object,
    ) -> None:
        if _issuer is not _WORKER_AUTHORIZATION_ISSUER:
            raise TypeError("BF16 worker authority requires live runs/get collection")
        if not isinstance(binding, PublicationBF16HandoffAttestationBinding):
            raise TypeError("binding has the wrong type")
        object.__setattr__(self, "binding", binding)
        object.__setattr__(
            self, "attempt_id", _nonempty_string(attempt_id, "attempt_id")
        )
        object.__setattr__(self, "ledger_id", _nonempty_string(ledger_id, "ledger_id"))
        object.__setattr__(
            self,
            "ledger_path_sha256",
            _require_sha256(ledger_path_sha256, "ledger_path_sha256"),
        )
        if not isinstance(producer_batch_prefix, DatabricksLedgerPrefix):
            raise TypeError("producer_batch_prefix has the wrong type")
        if producer_batch_prefix.ledger_id != ledger_id:
            raise ValueError("BF16 worker batch prefix identity drift")
        object.__setattr__(self, "producer_batch_prefix", producer_batch_prefix)
        object.__setattr__(
            self,
            "control_plane_status_sha256",
            _require_sha256(
                control_plane_status_sha256,
                "control_plane_status_sha256",
            ),
        )


@dataclass(frozen=True, slots=True)
class PublicationBF16HandoffGenerationResult:
    root: Path
    source_root: Path
    manifest_path: Path
    execution_record_path: Path
    manifest: Mapping[str, Any]
    record: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "source_root", Path(self.source_root))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(
            self, "execution_record_path", Path(self.execution_record_path)
        )
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))
        object.__setattr__(self, "record", MappingProxyType(dict(self.record)))


@dataclass(frozen=True, slots=True, init=False)
class PublicationBF16HandoffServingAuthorization:
    """Non-record authority for serving from one fully reconciled BF16 result."""

    result_root: Path
    execution_file_sha256: str
    execution_closed_record_sha256: str
    manifest_file_sha256: str
    manifest_closed_record_sha256: str
    ledger_id: str
    ledger_path_sha256: str
    predecessor_prefix: DatabricksLedgerPrefix
    producer_batch_prefix: DatabricksLedgerPrefix
    ledger_prefix: DatabricksLedgerPrefix
    causal_closure_sha256: str

    def __init__(
        self,
        *,
        result: PublicationBF16HandoffGenerationResult,
        ledger_id: str,
        ledger_path_sha256: str,
        predecessor_prefix: DatabricksLedgerPrefix,
        producer_batch_prefix: DatabricksLedgerPrefix,
        ledger_prefix: DatabricksLedgerPrefix,
        causal_closure_sha256: str,
        _issuer: object,
    ) -> None:
        if _issuer is not _SERVING_AUTHORIZATION_ISSUER:
            raise TypeError(
                "BF16 serving authority requires live causal reconciliation"
            )
        if not isinstance(result, PublicationBF16HandoffGenerationResult):
            raise TypeError("result has the wrong type")
        object.__setattr__(self, "result_root", result.root.absolute())
        object.__setattr__(
            self,
            "execution_file_sha256",
            _file_sha256(result.execution_record_path),
        )
        object.__setattr__(
            self,
            "execution_closed_record_sha256",
            _required_string(result.record, "closed_record_sha256"),
        )
        object.__setattr__(
            self,
            "manifest_file_sha256",
            _file_sha256(result.manifest_path),
        )
        object.__setattr__(
            self,
            "manifest_closed_record_sha256",
            _required_string(result.manifest, "closed_record_sha256"),
        )
        object.__setattr__(self, "ledger_id", _nonempty_string(ledger_id, "ledger_id"))
        object.__setattr__(
            self,
            "ledger_path_sha256",
            _require_sha256(ledger_path_sha256, "ledger_path_sha256"),
        )
        if not isinstance(predecessor_prefix, DatabricksLedgerPrefix):
            raise TypeError("predecessor_prefix has the wrong type")
        if not isinstance(producer_batch_prefix, DatabricksLedgerPrefix):
            raise TypeError("producer_batch_prefix has the wrong type")
        if not isinstance(ledger_prefix, DatabricksLedgerPrefix):
            raise TypeError("ledger_prefix has the wrong type")
        if (
            predecessor_prefix.ledger_id != ledger_id
            or producer_batch_prefix.ledger_id != ledger_id
            or ledger_prefix.ledger_id != ledger_id
        ):
            raise ValueError("BF16 ledger prefix identity drift")
        object.__setattr__(self, "predecessor_prefix", predecessor_prefix)
        object.__setattr__(self, "producer_batch_prefix", producer_batch_prefix)
        object.__setattr__(self, "ledger_prefix", ledger_prefix)
        object.__setattr__(
            self,
            "causal_closure_sha256",
            _require_sha256(causal_closure_sha256, "causal_closure_sha256"),
        )


def build_publication_bf16_handoff_execution_config(
    *,
    vllm_bitsandbytes_loader_source_sha256: str,
    generator_version: str = PUBLICATION_LATENCY_HANDOFF_TRANSFORMERS_VERSION,
) -> PublicationBF16HandoffExecutionConfig:
    """Build the frozen Qwen3 BF16 pre-RoPE generation contract."""

    layout = layout_for_model(
        QWEN3_4B_INSTRUCT_HF_MODEL_ID,
        dtype=PUBLICATION_BF16_HANDOFF_DTYPE,
        pre_rope=True,
        rope_theta=QWEN3_4B_ROPE_THETA,
        rope_rotary_dim=QWEN3_4B_ROPE_ROTARY_DIM,
        shares_kv_storage=False,
        storage_layout=KVStorageLayout.SEPARATE_KEY_VALUE,
        payload_axis_order=KVPayloadAxisOrder.TOKEN_MAJOR,
    )
    return PublicationBF16HandoffExecutionConfig(
        layout=layout,
        model_revision=MAIN_LATENCY_TOKENIZER_REVISION,
        generator_version=generator_version,
        vllm_bitsandbytes_loader_source_sha256=(vllm_bitsandbytes_loader_source_sha256),
    )


def build_publication_bf16_handoff_generation_plan(
    prepared_input_dir: str | Path,
    *,
    plan_id: str,
    tokenizer: MainLatencyTokenizer,
    worker_count: int = PUBLICATION_BF16_HANDOFF_WORKER_COUNT,
    source_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Close the exact 128-item 16k suite into sixteen token-balanced shards."""

    _safe_id(plan_id, "plan_id")
    if type(worker_count) is not int or worker_count != (
        PUBLICATION_BF16_HANDOFF_WORKER_COUNT
    ):
        raise ValueError("publication BF16 generation requires exactly 16 workers")
    prepared = verify_main_latency_inputs(
        prepared_input_dir,
        tokenizer=tokenizer,
        source_paths=source_paths,
        examples_per_dataset=PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
    )
    items = _generation_items(prepared, tokenizer=tokenizer)
    assignments = _lpt_assignments(items, worker_count=worker_count)
    workers: list[dict[str, Any]] = []
    for worker_index_value, assignment in enumerate(assignments):
        ordered = sorted(
            assignment,
            key=lambda item: (
                cast(str, item["dataset"]),
                cast(int, item["row_index"]),
            ),
        )
        workers.append(
            {
                "cache_prefix_tokens": sum(
                    cast(int, item["cache_prefix_tokens"]) for item in ordered
                ),
                "input_token_slots": sum(
                    cast(int, item["input_token_slots"]) for item in ordered
                ),
                "item_count": len(ordered),
                "items": ordered,
                "items_sha256": _canonical_sha256(ordered),
                "persistent_generator_instances": 1,
                "worker_id": f"bf16-handoff-worker-{worker_index_value:02d}",
                "worker_index": worker_index_value,
            }
        )
    loads = [cast(int, worker["cache_prefix_tokens"]) for worker in workers]
    cache_prefix_tokens = sum(loads)
    record: dict[str, Any] = {
        "closed_record_sha256": "",
        "context_tokens": PUBLICATION_BF16_HANDOFF_CONTEXT_TOKENS,
        "coverage": {
            "cache_prefix_generation_tokens": cache_prefix_tokens,
            "dataset_count": len(SUPPORTED_V1_DATASETS),
            "datasets": list(SUPPORTED_V1_DATASETS),
            "examples_per_dataset": PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
            "input_token_slots": sum(
                cast(int, item["input_token_slots"]) for item in items
            ),
            "task_count": len(items),
            "task_ids_sha256": _canonical_sha256(
                sorted(cast(str, item["task_id"]) for item in items)
            ),
        },
        "generation_contract": {
            "cache_method": CacheGenerationMethod.VANILLA_PREFILL.value,
            "context_tokens": PUBLICATION_BF16_HANDOFF_CONTEXT_TOKENS,
            "kv_dtype": PUBLICATION_BF16_HANDOFF_DTYPE,
            "position_encoding": KVKeyPositionEncoding.PRE_ROPE.value,
            "regenerate_inside_timed_serving_jobs": False,
            "segment_per_document": True,
            "storage_layout": KVStorageLayout.SEPARATE_KEY_VALUE.value,
        },
        "input_bundle_sha256": prepared.bundle_sha256,
        "plan_id": plan_id,
        "record_type": PUBLICATION_BF16_HANDOFF_PLAN_RECORD_TYPE,
        "schema_version": PUBLICATION_BF16_HANDOFF_SCHEMA_VERSION,
        "sharding": {
            "algorithm": "deterministic_lpt_exact_cache_prefix_tokens_v1",
            "max_parallel_workers": PUBLICATION_BF16_HANDOFF_WORKER_COUNT,
            "max_worker_cache_prefix_tokens": max(loads),
            "min_worker_cache_prefix_tokens": min(loads),
            "worker_count": worker_count,
            "worker_imbalance_tokens": max(loads) - min(loads),
        },
        "workers": workers,
        "workers_sha256": _canonical_sha256(workers),
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    return record


def validate_publication_bf16_handoff_generation_plan(
    record: Mapping[str, Any],
    *,
    prepared_input_dir: str | Path,
    tokenizer: MainLatencyTokenizer,
    source_paths: Mapping[str, str | Path] | None = None,
) -> None:
    """Rebuild the plan from verified source inputs and require byte equality."""

    _validate_plan_envelope(record)
    expected = build_publication_bf16_handoff_generation_plan(
        prepared_input_dir,
        plan_id=_required_string(record, "plan_id"),
        tokenizer=tokenizer,
        worker_count=_required_int(
            _required_mapping(record, "sharding"), "worker_count"
        ),
        source_paths=source_paths,
    )
    if dict(record) != expected:
        raise ValueError("BF16 handoff plan does not match verified inputs")


def write_publication_bf16_handoff_generation_plan(
    record: Mapping[str, Any], path: str | Path
) -> Path:
    _validate_plan_envelope(record)
    destination = Path(path).expanduser().absolute()
    _write_json_exclusive(record, destination)
    _sync_directory(destination.parent)
    _sync_directory(destination.parent.parent)
    return destination


def read_publication_bf16_handoff_generation_plan(
    path: str | Path,
) -> dict[str, Any]:
    record = _read_canonical_json_file(path, "BF16 generation plan")
    _validate_plan_envelope(record)
    return record


def build_publication_bf16_handoff_worker_payloads(
    plan: Mapping[str, Any],
    *,
    plan_uri: str,
    plan_file_sha256: str,
    prepared_input_uri: str,
    prepared_provenance_file_sha256: str,
    prepared_provenance_closed_record_sha256: str,
    durable_output_root: str,
    local_work_root_template: str,
    source_revision: str,
    config: PublicationBF16HandoffExecutionConfig,
    hardware_qualification: PublicationLatencyGeneratorHardwareQualification,
) -> tuple[dict[str, Any], ...]:
    """Build record-only worker inputs; this function grants no launch authority."""

    _validate_plan_envelope(plan)
    _resolved_source_revision(source_revision)
    for field_name, value in (
        ("plan_uri", plan_uri),
        ("prepared_input_uri", prepared_input_uri),
        ("durable_output_root", durable_output_root),
        ("local_work_root_template", local_work_root_template),
    ):
        _nonempty_string(value, field_name)
    if "{worker_index}" not in local_work_root_template:
        raise ValueError("local_work_root_template requires {worker_index}")
    for field_name, value in (
        ("plan_file_sha256", plan_file_sha256),
        ("prepared_provenance_file_sha256", prepared_provenance_file_sha256),
        (
            "prepared_provenance_closed_record_sha256",
            prepared_provenance_closed_record_sha256,
        ),
    ):
        _require_sha256(value, field_name)
    if not isinstance(config, PublicationBF16HandoffExecutionConfig):
        raise TypeError("config has the wrong type")
    if not isinstance(
        hardware_qualification, PublicationLatencyGeneratorHardwareQualification
    ):
        raise TypeError("hardware_qualification has the wrong type")
    qualification = _hardware_qualification_record(hardware_qualification)
    workers = _mapping_sequence(plan.get("workers"), "workers")
    payloads: list[dict[str, Any]] = []
    for expected_index, worker in enumerate(workers):
        worker_index_value = _required_int(worker, "worker_index")
        if worker_index_value != expected_index:
            raise ValueError("plan worker indices must be contiguous")
        payload: dict[str, Any] = {
            "assignment": {
                "cache_prefix_tokens": _required_int(worker, "cache_prefix_tokens"),
                "input_token_slots": _required_int(worker, "input_token_slots"),
                "item_count": _required_int(worker, "item_count"),
                "items_sha256": _required_string(worker, "items_sha256"),
            },
            "closed_record_sha256": "",
            "durable_output_root": durable_output_root.rstrip("/"),
            "execution_contract": _execution_config_record(config),
            "generator_hardware_qualification": qualification,
            "input_bundle_sha256": _required_string(plan, "input_bundle_sha256"),
            "local_work_root": local_work_root_template.format(
                worker_index=f"{worker_index_value:02d}"
            ),
            "output_binding": {
                "partial_record_relative_path": (
                    f"worker-records/worker-{worker_index_value:02d}.jsonl"
                ),
                "result_relative_path": (
                    f"worker-results/worker-{worker_index_value:02d}.json"
                ),
                "worker_bundle_relative_root": (
                    f"pending/{PUBLICATION_BF16_HANDOFF_CONTEXT_TOKENS}/"
                    f"worker-{worker_index_value:02d}"
                ),
            },
            "plan": {
                "closed_record_sha256": _required_string(plan, "closed_record_sha256"),
                "file_sha256": plan_file_sha256,
                "uri": plan_uri,
            },
            "prepared_inputs": {
                "bundle_sha256": _required_string(plan, "input_bundle_sha256"),
                "provenance_closed_record_sha256": (
                    prepared_provenance_closed_record_sha256
                ),
                "provenance_file_sha256": prepared_provenance_file_sha256,
                "uri": prepared_input_uri,
            },
            "record_type": PUBLICATION_BF16_HANDOFF_WORKER_PAYLOAD_RECORD_TYPE,
            "schema_version": PUBLICATION_BF16_HANDOFF_SCHEMA_VERSION,
            "source": {
                "cachet_source_tree_sha256": (
                    hardware_qualification.expected_artifact_pins.cachet_source_tree_sha256
                ),
                "source_revision": source_revision,
            },
            "worker_id": _required_string(worker, "worker_id"),
            "worker_index": worker_index_value,
        }
        _validate_local_nvme_worker_root(
            _required_string(payload, "local_work_root"),
            worker_index=worker_index_value,
            field_name="local_work_root",
        )
        payload["closed_record_sha256"] = _closed_record_sha256(payload)
        payloads.append(payload)
    if (
        len(payloads) != PUBLICATION_BF16_HANDOFF_WORKER_COUNT
        or len({item["closed_record_sha256"] for item in payloads})
        != PUBLICATION_BF16_HANDOFF_WORKER_COUNT
    ):
        raise ValueError("BF16 worker payload closure is incomplete or colliding")
    return tuple(payloads)


def validate_publication_bf16_handoff_worker_payload(
    payload: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> None:
    _validate_plan_envelope(plan)
    if (
        payload.get("record_type")
        != (PUBLICATION_BF16_HANDOFF_WORKER_PAYLOAD_RECORD_TYPE)
        or payload.get("schema_version") != PUBLICATION_BF16_HANDOFF_SCHEMA_VERSION
    ):
        raise ValueError("BF16 worker payload envelope is invalid")
    if payload.get("closed_record_sha256") != _closed_record_sha256(payload):
        raise ValueError("BF16 worker payload closure is invalid")
    plan_binding = _required_mapping(payload, "plan")
    if plan_binding.get("closed_record_sha256") != plan.get(
        "closed_record_sha256"
    ) or payload.get("input_bundle_sha256") != plan.get("input_bundle_sha256"):
        raise ValueError("BF16 worker payload source binding drift")
    worker_index_value = _required_int(payload, "worker_index")
    workers = _mapping_sequence(plan.get("workers"), "workers")
    if not 0 <= worker_index_value < len(workers):
        raise ValueError("BF16 worker index is outside the plan")
    worker = workers[worker_index_value]
    expected_assignment = {
        "cache_prefix_tokens": worker.get("cache_prefix_tokens"),
        "input_token_slots": worker.get("input_token_slots"),
        "item_count": worker.get("item_count"),
        "items_sha256": worker.get("items_sha256"),
    }
    if dict(_required_mapping(payload, "assignment")) != expected_assignment:
        raise ValueError("BF16 worker assignment drift")
    if payload.get("worker_id") != worker.get("worker_id"):
        raise ValueError("BF16 worker identity drift")
    _validate_local_nvme_worker_root(
        _required_string(payload, "local_work_root"),
        worker_index=worker_index_value,
        field_name="local_work_root",
    )
    _execution_config_from_record(_required_mapping(payload, "execution_contract"))
    _validate_qualification_record(
        _required_mapping(payload, "generator_hardware_qualification")
    )
    source = _required_mapping(payload, "source")
    _resolved_source_revision(_required_string(source, "source_revision"))
    _require_sha256(
        _required_string(source, "cachet_source_tree_sha256"),
        "cachet_source_tree_sha256",
    )
    expected_output = {
        "partial_record_relative_path": (
            f"worker-records/worker-{worker_index_value:02d}.jsonl"
        ),
        "result_relative_path": (
            f"worker-results/worker-{worker_index_value:02d}.json"
        ),
        "worker_bundle_relative_root": (
            f"pending/{PUBLICATION_BF16_HANDOFF_CONTEXT_TOKENS}/"
            f"worker-{worker_index_value:02d}"
        ),
    }
    if dict(_required_mapping(payload, "output_binding")) != expected_output:
        raise ValueError("BF16 worker output binding drift")


def write_publication_bf16_handoff_worker_payloads(
    payloads: Sequence[Mapping[str, Any]], output_dir: str | Path
) -> tuple[Path, ...]:
    values = tuple(payloads)
    if len(values) != PUBLICATION_BF16_HANDOFF_WORKER_COUNT:
        raise ValueError("exactly sixteen BF16 worker payloads are required")
    root = Path(output_dir).expanduser().absolute()
    _require_fresh_path(root)
    root.mkdir(parents=True)
    paths: list[Path] = []
    for index, payload in enumerate(values):
        if _required_int(payload, "worker_index") != index:
            raise ValueError("BF16 payload order must be contiguous")
        path = root / f"bf16-handoff-worker-{index:02d}.json"
        _write_json_exclusive(payload, path)
        paths.append(path)
    _sync_directory(root)
    _sync_directory(root.parent)
    return tuple(paths)


def build_databricks_publication_bf16_handoff_submit_payloads(
    config: DatabricksPublicationBF16HandoffJobConfig,
    worker_payloads: Sequence[Mapping[str, Any]],
    *,
    ledger_path: str | Path,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    q8_handoff_remote_closure_authorization: PublicationHandoffRemoteClosureAuthorization,
) -> tuple[dict[str, Any], ...]:
    """Render sixteen capability-authorized, independent one-GPU submissions."""

    _require_q8_remote_closure_type(q8_handoff_remote_closure_authorization)
    if not isinstance(config, DatabricksPublicationBF16HandoffJobConfig):
        raise TypeError("config has the wrong type")
    payloads = tuple(worker_payloads)
    expected_input, expected_qualification = _bf16_remote_predecessor_pins(payloads)
    _require_matching_predecessor_ledger(
        ledger_path,
        qualification_launch_authorization,
        q8_handoff_remote_closure_authorization,
        expected_input_bundle_sha256=expected_input,
        expected_qualification_closed_record_sha256=expected_qualification,
    )
    if len(payloads) != PUBLICATION_BF16_HANDOFF_WORKER_COUNT:
        raise ValueError("production BF16 generation requires sixteen workers")
    submissions: list[dict[str, Any]] = []
    roots: set[str] = set()
    for index, payload in enumerate(payloads):
        _require_authorized_worker_payload(
            payload,
            qualification_launch_authorization,
            worker_index_value=index,
        )
        _validate_job_artifact_binding(config, payload)
        root = _required_string(payload, "durable_output_root")
        plan_sha256 = _required_string(
            _required_mapping(payload, "plan"), "closed_record_sha256"
        )
        if not _is_durable_plan_root(root, plan_sha256=plan_sha256):
            raise ValueError(
                "BF16 durable output root is not plan-bound DBFS/UC storage"
            )
        roots.add(root)
        worker_label = f"{index:02d}"
        worker_uri = config.worker_payload_uri_template.format(
            worker_index=worker_label
        )
        runtime_venv = config.runtime_venv_dir_template.format(
            worker_index=worker_label
        )
        _validate_local_nvme_worker_root(
            runtime_venv,
            worker_index=index,
            field_name="runtime_venv_dir",
        )
        task = {
            "max_retries": 0,
            "new_cluster": _l40s_cluster(config),
            "spark_python_task": {
                "parameters": [
                    "--runner-sha256",
                    config.runner_sha256,
                    "--package-wheel-uri",
                    config.package_wheel_uri,
                    "--package-wheel-sha256",
                    config.package_wheel_sha256,
                    "--runtime-lock-uri",
                    config.runtime_lock_uri,
                    "--runtime-lock-sha256",
                    config.runtime_lock_sha256,
                    "--patched-vllm-wheel-uri",
                    config.patched_vllm_wheel_uri,
                    "--patched-vllm-wheel-sha256",
                    config.patched_vllm_wheel_sha256,
                    "--runtime-venv-dir",
                    runtime_venv,
                    "run-worker",
                    "--worker-payload-json",
                    worker_uri,
                    "--expected-worker-payload-sha256",
                    _worker_payload_file_sha256(payload),
                ],
                "python_file": config.runner_python_file,
            },
            "task_key": f"bf16_handoff_worker_{index:02d}",
            "timeout_seconds": PUBLICATION_BF16_HANDOFF_TASK_TIMEOUT_SECONDS,
        }
        attempt_id = publication_bf16_handoff_worker_attempt_id(
            payload,
            worker_index=index,
        )
        submissions.append(
            bind_databricks_run_idempotency_token(
                {
                    "run_name": f"{config.run_name}-worker-{index:02d}",
                    "tasks": [task],
                    "timeout_seconds": PUBLICATION_BF16_HANDOFF_TASK_TIMEOUT_SECONDS,
                },
                attempt_id=attempt_id,
            )
        )
    if (
        len(roots) != 1
        or sum(cast(int, payload["timeout_seconds"]) for payload in submissions)
        / 3600.0
        != PUBLICATION_BF16_HANDOFF_MAX_RESERVED_GPU_HOURS
    ):
        raise ValueError("BF16 producer wave does not close one 80-GPU-hour root")
    return tuple(submissions)


def publication_bf16_handoff_worker_attempt_id(
    worker_payload: Mapping[str, Any],
    *,
    worker_index: int,
) -> str:
    """Return the sole publication attempt identity for one BF16 worker."""

    _worker_index(worker_index)
    if _required_int(worker_payload, "worker_index") != worker_index:
        raise ValueError("BF16 attempt worker index differs from its worker payload")
    plan_sha256 = _require_sha256(
        _required_mapping(worker_payload, "plan").get("closed_record_sha256"),
        "worker plan closed_record_sha256",
    )
    return f"publication-bf16/{plan_sha256[:20]}/worker-{worker_index:02d}"


def _require_matching_predecessor_ledger(
    ledger_path: str | Path,
    authorization: GPUQualificationLaunchAuthorization,
    q8_authorization: PublicationHandoffRemoteClosureAuthorization,
    *,
    expected_input_bundle_sha256: str,
    expected_qualification_closed_record_sha256: str,
    expected_q8_ledger_prefix: DatabricksLedgerPrefix | None = None,
) -> DatabricksClusterHourLedger:
    if not isinstance(authorization, GPUQualificationLaunchAuthorization):
        raise TypeError(
            "qualification_launch_authorization must be a "
            "GPUQualificationLaunchAuthorization"
        )
    from document_kv_cache.publication_handoff_closure_coordinator import (
        require_q8_handoff_remote_closure_predecessor_authorization,
    )

    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    ledger_path_sha256 = databricks_ledger_path_sha256(ledger_path)
    live_prefix = databricks_ledger_prefix(ledger)
    q8_prefix = (
        live_prefix
        if expected_q8_ledger_prefix is None
        else expected_q8_ledger_prefix
    )
    require_databricks_ledger_prefix(ledger, q8_prefix)
    q8_remote = require_q8_handoff_remote_closure_predecessor_authorization(
        q8_authorization,
        expected_ledger_id=ledger.ledger_id,
        expected_ledger_path_sha256=ledger_path_sha256,
        expected_ledger_prefix=q8_prefix,
        expected_input_bundle_sha256=expected_input_bundle_sha256,
        expected_qualification_closed_record_sha256=(
            expected_qualification_closed_record_sha256
        ),
    )
    if (
        ledger.ledger_id != authorization.ledger_id
        or ledger.ledger_id != q8_remote.ledger_id
    ):
        raise ValueError("BF16 ledger differs from predecessor authorities")
    if (
        ledger_path_sha256 != authorization.ledger_path_sha256
        or ledger_path_sha256 != q8_remote.ledger_path_sha256
    ):
        raise ValueError("BF16 ledger path differs from predecessor authorities")
    require_databricks_ledger_prefix(ledger, authorization.ledger_prefix)
    return ledger


def _bf16_remote_predecessor_pins(
    worker_payloads: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    values = tuple(worker_payloads)
    if not values:
        raise ValueError("BF16 predecessor binding requires worker payloads")
    bindings = {
        (
            _require_sha256(
                payload.get("input_bundle_sha256"),
                "input_bundle_sha256",
            ),
            _require_sha256(
                _required_mapping(
                    payload, "generator_hardware_qualification"
                ).get("evidence_closed_record_sha256"),
                "qualification evidence_closed_record_sha256",
            ),
        )
        for payload in values
    }
    if len(bindings) != 1:
        raise ValueError("BF16 workers disagree on Q8 predecessor input/qualification")
    return next(iter(bindings))


def _q8_remote_closure_binding(
    authorization: object,
    *,
    expected_input_bundle_sha256: str,
    expected_qualification_closed_record_sha256: str,
) -> dict[str, Any]:
    _require_q8_remote_closure_type(authorization)
    from document_kv_cache.publication_handoff_closure_coordinator import (
        PublicationHandoffRemoteClosureAuthorization,
    )

    if not isinstance(authorization, PublicationHandoffRemoteClosureAuthorization):
        raise AssertionError("Q8 remote closure type guard drift")
    return {
        "causal_closure_sha256": authorization.causal_closure_sha256,
        "execution_file_sha256": authorization.execution_file_sha256,
        "input_bundle_sha256": _require_sha256(
            expected_input_bundle_sha256, "expected_input_bundle_sha256"
        ),
        "ledger_id": authorization.ledger_id,
        "ledger_path_sha256": authorization.ledger_path_sha256,
        "ledger_prefix": authorization.ledger_prefix.to_record(),
        "qualification_closed_record_sha256": _require_sha256(
            expected_qualification_closed_record_sha256,
            "expected_qualification_closed_record_sha256",
        ),
        "request_closed_record_sha256": (
            authorization.request_closed_record_sha256
        ),
        "result_closed_record_sha256": authorization.result_closed_record_sha256,
        "result_file_sha256": authorization.result_file_sha256,
    }


def _require_q8_remote_closure_type(authorization: object) -> None:
    from document_kv_cache.publication_handoff_closure_coordinator import (
        PublicationHandoffRemoteClosureAuthorization,
    )

    if not isinstance(authorization, PublicationHandoffRemoteClosureAuthorization):
        raise TypeError(
            "BF16 predecessor requires "
            "PublicationHandoffRemoteClosureAuthorization"
        )


def _require_submission_q8_remote_closure(
    ledger_path: str | Path,
    authorization: object,
    submission_authorization: PublicationBF16HandoffSubmissionAuthorization,
) -> DatabricksBatchReservationAuthorization:
    from document_kv_cache.publication_handoff_closure_coordinator import (
        require_q8_handoff_remote_closure_predecessor_authorization,
    )

    _require_q8_remote_closure_type(authorization)
    batch = require_publication_bf16_handoff_submission_authorization(
        submission_authorization
    )
    binding = dict(submission_authorization.q8_remote_closure_binding)
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    require_databricks_ledger_prefix(ledger, batch.predecessor_prefix)
    require_databricks_ledger_prefix(ledger, batch.batch_prefix)
    remote = require_q8_handoff_remote_closure_predecessor_authorization(
        authorization,
        expected_ledger_id=ledger.ledger_id,
        expected_ledger_path_sha256=databricks_ledger_path_sha256(ledger_path),
        expected_ledger_prefix=batch.predecessor_prefix,
        expected_input_bundle_sha256=_required_string(
            binding, "input_bundle_sha256"
        ),
        expected_qualification_closed_record_sha256=_required_string(
            binding, "qualification_closed_record_sha256"
        ),
    )
    observed = _q8_remote_closure_binding(
        remote,
        expected_input_bundle_sha256=_required_string(
            binding, "input_bundle_sha256"
        ),
        expected_qualification_closed_record_sha256=_required_string(
            binding, "qualification_closed_record_sha256"
        ),
    )
    if (
        observed != binding
        or _canonical_sha256(binding)
        != submission_authorization.q8_remote_closure_binding_sha256
        or batch.ledger_path_sha256 != binding.get("ledger_path_sha256")
        or batch.predecessor_prefix.to_record() != binding.get("ledger_prefix")
    ):
        raise ValueError("BF16 phase lease Q8 remote closure binding drift")
    return batch


def reserve_publication_bf16_handoff_worker_attempt_json(
    ledger_path: str | Path,
    submit_payload: Mapping[str, Any],
    *,
    worker_payload: Mapping[str, Any],
    worker_index: int,
    attempt_id: str,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    q8_handoff_remote_closure_authorization: PublicationHandoffRemoteClosureAuthorization,
) -> DatabricksClusterHourLedger:
    """Capability-check and reserve one exact five-hour producer payload."""

    _require_q8_remote_closure_type(q8_handoff_remote_closure_authorization)
    expected_worker_sha256 = _worker_payload_file_sha256(worker_payload)
    _validate_launch_binding(
        submit_payload,
        worker_payload=worker_payload,
        worker_index_value=worker_index,
        authorization=qualification_launch_authorization,
        expected_worker_sha256=expected_worker_sha256,
    )
    path = Path(ledger_path)
    expected_input, expected_qualification = _bf16_remote_predecessor_pins(
        (worker_payload,)
    )
    _require_matching_predecessor_ledger(
        path,
        qualification_launch_authorization,
        q8_handoff_remote_closure_authorization,
        expected_input_bundle_sha256=expected_input,
        expected_qualification_closed_record_sha256=expected_qualification,
    )
    raise RuntimeError(
        "single-worker BF16 reservation is nonpublication; use the exact 16-worker wave"
    )
    return reserve_databricks_run_attempt_json(
        path,
        submit_payload,
        attempt_id=attempt_id,
        workload_id=f"publication-bf16-handoff-worker-{worker_index:02d}",
        reservation_validator=_reservation_validator(
            path,
            worker_payload=worker_payload,
            worker_index_value=worker_index,
            authorization=qualification_launch_authorization,
            q8_authorization=q8_handoff_remote_closure_authorization,
            expected_worker_sha256=expected_worker_sha256,
        ),
    )


def reserve_and_submit_publication_bf16_handoff_worker(
    workspace: DatabricksWorkspaceConfig,
    submit_payload: Mapping[str, Any],
    *,
    ledger_path: str | Path,
    worker_payload: Mapping[str, Any],
    worker_index: int,
    attempt_id: str,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    q8_handoff_remote_closure_authorization: PublicationHandoffRemoteClosureAuthorization,
    opener: DatabricksURLOpener | None = None,
) -> dict[str, Any]:
    """Reserve exact bytes, submit them, and durably receipt-bind the cloud run."""

    _require_q8_remote_closure_type(q8_handoff_remote_closure_authorization)
    expected_worker_sha256 = _worker_payload_file_sha256(worker_payload)
    _validate_launch_binding(
        submit_payload,
        worker_payload=worker_payload,
        worker_index_value=worker_index,
        authorization=qualification_launch_authorization,
        expected_worker_sha256=expected_worker_sha256,
    )
    path = Path(ledger_path)
    expected_input, expected_qualification = _bf16_remote_predecessor_pins(
        (worker_payload,)
    )
    _require_matching_predecessor_ledger(
        path,
        qualification_launch_authorization,
        q8_handoff_remote_closure_authorization,
        expected_input_bundle_sha256=expected_input,
        expected_qualification_closed_record_sha256=expected_qualification,
    )
    raise RuntimeError(
        "single-worker BF16 submission is nonpublication; use the exact 16-worker wave"
    )
    return reserve_and_submit_databricks_run(
        workspace,
        submit_payload,
        ledger_path=path,
        attempt_id=attempt_id,
        workload_id=f"publication-bf16-handoff-worker-{worker_index:02d}",
        reservation_validator=_reservation_validator(
            path,
            worker_payload=worker_payload,
            worker_index_value=worker_index,
            authorization=qualification_launch_authorization,
            q8_authorization=q8_handoff_remote_closure_authorization,
            expected_worker_sha256=expected_worker_sha256,
        ),
        opener=opener,
    )


def reserve_and_submit_publication_bf16_handoff_worker_wave(
    workspace: DatabricksWorkspaceConfig,
    submit_payloads: Sequence[Mapping[str, Any]],
    *,
    ledger_path: str | Path,
    worker_payloads: Sequence[Mapping[str, Any]],
    attempt_ids_by_worker: Mapping[int, str],
    phase_lease_root: str | Path,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    q8_handoff_remote_closure_authorization: PublicationHandoffRemoteClosureAuthorization,
    opener: DatabricksURLOpener | None = None,
) -> tuple[
    tuple[dict[str, Any], ...],
    PublicationBF16HandoffSubmissionAuthorization,
]:
    """Atomically admit all sixteen BF16 producers before the first POST."""

    _require_q8_remote_closure_type(q8_handoff_remote_closure_authorization)
    submissions = tuple(submit_payloads)
    workers = tuple(worker_payloads)
    indexes = tuple(range(PUBLICATION_BF16_HANDOFF_WORKER_COUNT))
    if len(submissions) != len(indexes) or len(workers) != len(indexes):
        raise ValueError("BF16 production wave requires exactly sixteen members")
    if set(attempt_ids_by_worker) != set(indexes):
        raise ValueError("BF16 attempt IDs must cover workers 0..15 exactly")
    durable_output_root = _common_bf16_durable_output_root(workers)
    expected_input, expected_qualification = _bf16_remote_predecessor_pins(workers)
    live = _require_matching_predecessor_ledger(
        ledger_path,
        qualification_launch_authorization,
        q8_handoff_remote_closure_authorization,
        expected_input_bundle_sha256=expected_input,
        expected_qualification_closed_record_sha256=expected_qualification,
    )
    predecessor = databricks_ledger_prefix(live)
    q8_remote_binding = _q8_remote_closure_binding(
        q8_handoff_remote_closure_authorization,
        expected_input_bundle_sha256=expected_input,
        expected_qualification_closed_record_sha256=expected_qualification,
    )
    attempts = tuple(attempt_ids_by_worker[index] for index in indexes)
    expected_attempts = tuple(
        publication_bf16_handoff_worker_attempt_id(workers[index], worker_index=index)
        for index in indexes
    )
    if attempts != expected_attempts:
        raise ValueError("BF16 attempt IDs differ from frozen worker identities")
    payload_digests: list[str] = []
    requests: list[DatabricksRunAttemptReservationRequest] = []
    for index, submit_payload, worker_payload in zip(
        indexes, submissions, workers, strict=True
    ):
        expected_worker_sha256 = _worker_payload_file_sha256(worker_payload)
        _validate_launch_binding(
            submit_payload,
            worker_payload=worker_payload,
            worker_index_value=index,
            authorization=qualification_launch_authorization,
            expected_worker_sha256=expected_worker_sha256,
        )
        require_databricks_run_idempotency_token(
            submit_payload,
            attempt_id=attempts[index],
        )
        _snapshot, canonical = canonical_databricks_submit_payload_snapshot(
            submit_payload
        )
        payload_digests.append(sha256(canonical).hexdigest())
        requests.append(
            DatabricksRunAttemptReservationRequest(
                attempt_id=attempts[index],
                workload_id=f"publication-bf16-handoff-worker-{index:02d}",
                submit_payload=submit_payload,
            )
        )

    lease_root = Path(phase_lease_root).expanduser().absolute()
    _require_fresh_path(lease_root)
    if not lease_root.parent.is_dir():
        raise ValueError("BF16 phase lease parent must already be a real directory")
    lease_root.mkdir()
    _sync_directory(lease_root.parent)
    lease = {
        "attempt_ids": list(attempts),
        "closed_record_sha256": "",
        "durable_output_root": durable_output_root,
        "ledger_path_sha256": q8_remote_binding["ledger_path_sha256"],
        "predecessor_prefix": predecessor.to_record(),
        "q8_remote_closure": q8_remote_binding,
        "record_type": "cachet.publication_bf16_handoff_phase_lease.v1",
        "submit_payload_sha256": payload_digests,
    }
    lease["closed_record_sha256"] = _closed_record_sha256(lease)
    _write_json_exclusive(lease, lease_root / "phase-lease.json")
    _sync_directory(lease_root)

    def validate_batch(
        batch_live: DatabricksClusterHourLedger,
        reservations: tuple[DatabricksClusterHourReservation, ...],
        snapshots: tuple[Mapping[str, Any], ...],
    ) -> None:
        if databricks_ledger_path_sha256(ledger_path) != (
            q8_remote_binding["ledger_path_sha256"]
        ):
            raise ValueError("BF16 batch ledger path binding drift")
        require_databricks_ledger_prefix(batch_live, predecessor)
        if len(reservations) != len(indexes) or len(snapshots) != len(indexes):
            raise ValueError("BF16 batch must contain exactly sixteen producers")
        for index, reservation, snapshot, worker_payload in zip(
            indexes, reservations, snapshots, workers, strict=True
        ):
            expected_worker_sha256 = _worker_payload_file_sha256(worker_payload)
            _validate_launch_binding(
                snapshot,
                worker_payload=worker_payload,
                worker_index_value=index,
                authorization=qualification_launch_authorization,
                expected_worker_sha256=expected_worker_sha256,
            )
            require_databricks_run_idempotency_token(
                snapshot,
                attempt_id=attempts[index],
            )
            if (
                reservation.attempt_id != attempts[index]
                or reservation.submit_payload_sha256 != payload_digests[index]
                or reservation.reserved_cluster_hours != 5.0
            ):
                raise ValueError("BF16 batch reservation member drift")
        proposed_hours = sum(item.reserved_cluster_hours for item in reservations)
        proposed_tasks = sum(len(item.task_timeout_seconds) for item in reservations)
        if batch_live.cap_cluster_hours != MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS:
            raise ValueError("BF16 generation requires the 1024-hour ledger")
        if (
            batch_live.active_reserved_task_count + proposed_tasks
            > PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS
        ):
            raise ValueError("BF16 wave exceeds the global 16-job concurrency cap")
        if (
            batch_live.active_reserved_cluster_hours + proposed_hours
            > MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS
        ):
            raise ValueError("BF16 wave exceeds the active 900-hour cap")
        if (
            batch_live.accounted_cluster_hours + proposed_hours
            > MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
            - PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS
        ):
            raise ValueError("BF16 wave consumes the 124-hour headroom")

    try:
        _batch_ledger, batch_authorization = (
            reserve_databricks_run_attempt_batch_authorized_json(
                ledger_path,
                tuple(requests),
                expected_predecessor_prefix=predecessor,
                batch_validator=validate_batch,
            )
        )
    except BaseException:
        if lease_root.is_dir() and not lease_root.is_symlink():
            for candidate in tuple(lease_root.iterdir()):
                if candidate.is_file() and not candidate.is_symlink():
                    candidate.unlink()
            if not any(lease_root.iterdir()):
                lease_root.rmdir()
                _sync_directory(lease_root.parent)
        raise
    require_databricks_batch_reservation_authorization(
        batch_authorization,
        expected_predecessor_prefix=predecessor,
        expected_attempt_ids=attempts,
        expected_submit_payload_sha256s=payload_digests,
    )
    batch_record = {
        "batch_prefix": batch_authorization.batch_prefix.to_record(),
        "closed_record_sha256": "",
        "record_type": "cachet.publication_bf16_handoff_batch_reserved.v1",
    }
    batch_record["closed_record_sha256"] = _closed_record_sha256(batch_record)
    _write_json_exclusive(batch_record, lease_root / "batch-reserved.json")
    _sync_directory(lease_root)
    submission_authorization = _issue_bf16_submission_authorization(
        batch_authorization,
        lease_root,
        q8_remote_closure_binding=q8_remote_binding,
        durable_output_root=durable_output_root,
    )

    responses: list[dict[str, Any]] = []
    for index, submit_payload in zip(indexes, submissions, strict=True):
        intent_path = lease_root / f"worker-{index:02d}.post-intent.json"
        intent = {
            "attempt_id": attempts[index],
            "batch_prefix": batch_authorization.batch_prefix.to_record(),
            "closed_record_sha256": "",
            "record_type": "cachet.publication_bf16_handoff_post_intent.v1",
            "submit_payload_sha256": payload_digests[index],
            "worker_index": index,
        }
        intent["closed_record_sha256"] = _closed_record_sha256(intent)
        _write_json_exclusive(intent, intent_path)
        _sync_directory(lease_root)
        response = submit_pre_reserved_databricks_run(
            workspace,
            submit_payload,
            ledger_path=ledger_path,
            attempt_id=attempts[index],
            batch_authorization=batch_authorization,
            opener=opener,
        )
        receipt_ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
        receipt = next(
            item
            for item in receipt_ledger.submission_receipts
            if item.attempt_id == attempts[index]
        )
        receipt_record = {
            "attempt_id": attempts[index],
            "closed_record_sha256": "",
            "record_type": "cachet.publication_bf16_handoff_submit_receipt.v1",
            "run_id": receipt.run_id,
            "submit_payload_sha256": receipt.submit_payload_sha256,
            "submit_response_sha256": receipt.submit_response_sha256,
            "worker_index": index,
        }
        receipt_record["closed_record_sha256"] = _closed_record_sha256(receipt_record)
        receipt_path = lease_root / f"worker-{index:02d}.receipt.json"
        _write_json_exclusive(receipt_record, receipt_path)
        intent_path.unlink()
        _sync_directory(lease_root)
        responses.append(response)
    return tuple(responses), submission_authorization


def resume_publication_bf16_handoff_worker_wave(
    workspace: DatabricksWorkspaceConfig,
    submit_payloads: Sequence[Mapping[str, Any]],
    *,
    ledger_path: str | Path,
    worker_payloads: Sequence[Mapping[str, Any]],
    attempt_ids_by_worker: Mapping[int, str],
    phase_lease_root: str | Path,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    q8_handoff_remote_closure_authorization: PublicationHandoffRemoteClosureAuthorization,
    opener: DatabricksURLOpener | None = None,
) -> tuple[
    tuple[dict[str, Any], ...],
    PublicationBF16HandoffSubmissionAuthorization,
]:
    """Resume the exact BF16 producer wave from its durable phase lease."""

    _require_q8_remote_closure_type(q8_handoff_remote_closure_authorization)
    submissions = tuple(submit_payloads)
    workers = tuple(worker_payloads)
    indexes = tuple(range(PUBLICATION_BF16_HANDOFF_WORKER_COUNT))
    if len(submissions) != len(indexes) or len(workers) != len(indexes):
        raise ValueError("BF16 production wave requires exactly sixteen members")
    if set(attempt_ids_by_worker) != set(indexes):
        raise ValueError("BF16 attempt IDs must cover workers 0..15 exactly")
    durable_output_root = _common_bf16_durable_output_root(workers)
    expected_input, expected_qualification = _bf16_remote_predecessor_pins(workers)
    q8_remote_binding = _q8_remote_closure_binding(
        q8_handoff_remote_closure_authorization,
        expected_input_bundle_sha256=expected_input,
        expected_qualification_closed_record_sha256=expected_qualification,
    )
    predecessor = databricks_ledger_prefix_from_record(
        _required_mapping(q8_remote_binding, "ledger_prefix")
    )
    live = _require_matching_predecessor_ledger(
        ledger_path,
        qualification_launch_authorization,
        q8_handoff_remote_closure_authorization,
        expected_input_bundle_sha256=expected_input,
        expected_qualification_closed_record_sha256=expected_qualification,
        expected_q8_ledger_prefix=predecessor,
    )
    require_databricks_ledger_prefix(live, predecessor)
    attempts = tuple(attempt_ids_by_worker[index] for index in indexes)
    if attempts != tuple(
        publication_bf16_handoff_worker_attempt_id(workers[index], worker_index=index)
        for index in indexes
    ):
        raise ValueError("BF16 attempt IDs differ from frozen worker identities")
    requests: list[DatabricksRunAttemptReservationRequest] = []
    payload_digests: list[str] = []
    for index, submit_payload, worker_payload in zip(
        indexes, submissions, workers, strict=True
    ):
        worker_sha256 = _worker_payload_file_sha256(worker_payload)
        _validate_launch_binding(
            submit_payload,
            worker_payload=worker_payload,
            worker_index_value=index,
            authorization=qualification_launch_authorization,
            expected_worker_sha256=worker_sha256,
        )
        require_databricks_run_idempotency_token(
            submit_payload, attempt_id=attempts[index]
        )
        _snapshot, canonical = canonical_databricks_submit_payload_snapshot(
            submit_payload
        )
        payload_digests.append(sha256(canonical).hexdigest())
        requests.append(
            DatabricksRunAttemptReservationRequest(
                attempt_id=attempts[index],
                workload_id=f"publication-bf16-handoff-worker-{index:02d}",
                submit_payload=submit_payload,
            )
        )
    lease_root = Path(phase_lease_root).expanduser().absolute()
    if not lease_root.is_dir() or lease_root.is_symlink():
        raise ValueError("BF16 resume requires the existing real phase lease")
    expected_lease: dict[str, Any] = {
        "attempt_ids": list(attempts),
        "closed_record_sha256": "",
        "durable_output_root": durable_output_root,
        "ledger_path_sha256": q8_remote_binding["ledger_path_sha256"],
        "predecessor_prefix": predecessor.to_record(),
        "q8_remote_closure": q8_remote_binding,
        "record_type": "cachet.publication_bf16_handoff_phase_lease.v1",
        "submit_payload_sha256": payload_digests,
    }
    expected_lease["closed_record_sha256"] = _closed_record_sha256(expected_lease)
    if (
        _read_canonical_json_file(lease_root / "phase-lease.json", "BF16 phase lease")
        != expected_lease
    ):
        raise ValueError("BF16 phase lease differs from the frozen wave")
    batch_authorization = replay_databricks_run_attempt_batch_authorization_json(
        ledger_path,
        tuple(requests),
        expected_predecessor_prefix=predecessor,
    )
    require_databricks_publication_batch_admission(live, batch_authorization)
    expected_batch: dict[str, Any] = {
        "batch_prefix": batch_authorization.batch_prefix.to_record(),
        "closed_record_sha256": "",
        "record_type": "cachet.publication_bf16_handoff_batch_reserved.v1",
    }
    expected_batch["closed_record_sha256"] = _closed_record_sha256(expected_batch)
    batch_path = lease_root / "batch-reserved.json"
    if batch_path.exists() or batch_path.is_symlink():
        if _read_canonical_json_file(batch_path, "BF16 batch marker") != expected_batch:
            raise ValueError("BF16 batch marker differs from the ledger batch")
    else:
        _write_json_exclusive(expected_batch, batch_path)
        _sync_directory(lease_root)
    submission_authorization = _issue_bf16_submission_authorization(
        batch_authorization,
        lease_root,
        q8_remote_closure_binding=q8_remote_binding,
        durable_output_root=durable_output_root,
    )
    responses: list[dict[str, Any]] = []
    for index, submit_payload in zip(indexes, submissions, strict=True):
        intent_path = lease_root / f"worker-{index:02d}.post-intent.json"
        receipt_path = lease_root / f"worker-{index:02d}.receipt.json"
        expected_intent: dict[str, Any] = {
            "attempt_id": attempts[index],
            "batch_prefix": batch_authorization.batch_prefix.to_record(),
            "closed_record_sha256": "",
            "record_type": "cachet.publication_bf16_handoff_post_intent.v1",
            "submit_payload_sha256": payload_digests[index],
            "worker_index": index,
        }
        expected_intent["closed_record_sha256"] = _closed_record_sha256(expected_intent)
        if intent_path.exists() or intent_path.is_symlink():
            if (
                _read_canonical_json_file(
                    intent_path, f"BF16 worker {index} post intent"
                )
                != expected_intent
            ):
                raise ValueError("BF16 post intent drift")
        elif not receipt_path.exists():
            _write_json_exclusive(expected_intent, intent_path)
            _sync_directory(lease_root)
        response = resume_pre_reserved_databricks_run(
            workspace,
            submit_payload,
            ledger_path=ledger_path,
            attempt_id=attempts[index],
            batch_authorization=batch_authorization,
            opener=opener,
        )
        ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
        receipt = next(
            item
            for item in ledger.submission_receipts
            if item.attempt_id == attempts[index]
        )
        expected_receipt: dict[str, Any] = {
            "attempt_id": attempts[index],
            "closed_record_sha256": "",
            "record_type": "cachet.publication_bf16_handoff_submit_receipt.v1",
            "run_id": receipt.run_id,
            "submit_payload_sha256": receipt.submit_payload_sha256,
            "submit_response_sha256": receipt.submit_response_sha256,
            "worker_index": index,
        }
        expected_receipt["closed_record_sha256"] = _closed_record_sha256(
            expected_receipt
        )
        if receipt_path.exists() or receipt_path.is_symlink():
            if (
                _read_canonical_json_file(receipt_path, f"BF16 worker {index} receipt")
                != expected_receipt
            ):
                raise ValueError("BF16 durable submit receipt drift")
        else:
            _write_json_exclusive(expected_receipt, receipt_path)
        if intent_path.is_file() and not intent_path.is_symlink():
            intent_path.unlink()
        _sync_directory(lease_root)
        responses.append(response)
    expected_names = {"phase-lease.json", "batch-reserved.json"} | {
        f"worker-{index:02d}.receipt.json" for index in indexes
    }
    if {item.name for item in lease_root.iterdir()} != expected_names:
        raise ValueError("resumed BF16 phase lease directory is not closed")
    return tuple(responses), submission_authorization


def require_publication_bf16_handoff_submission_authorization(
    authorization: object,
) -> DatabricksBatchReservationAuthorization:
    """Replay the BF16 phase lease and marker behind submission authority."""

    if not isinstance(authorization, PublicationBF16HandoffSubmissionAuthorization):
        raise TypeError(
            "BF16 publication collection requires "
            "PublicationBF16HandoffSubmissionAuthorization"
        )
    lease, marker = _validate_bf16_submission_phase_files(
        authorization.phase_lease_root,
        authorization.batch_authorization,
        q8_remote_closure_binding=authorization.q8_remote_closure_binding,
        durable_output_root=authorization.durable_output_root,
    )
    observed = (
        _bf16_phase_lease_root_sha256(authorization.phase_lease_root),
        _file_sha256(authorization.phase_lease_root / "phase-lease.json"),
        _required_string(lease, "closed_record_sha256"),
        _file_sha256(authorization.phase_lease_root / "batch-reserved.json"),
        _required_string(marker, "closed_record_sha256"),
        _canonical_sha256(_required_mapping(lease, "q8_remote_closure")),
    )
    expected = (
        authorization.phase_lease_root_sha256,
        authorization.phase_lease_file_sha256,
        authorization.phase_lease_closed_record_sha256,
        authorization.batch_marker_file_sha256,
        authorization.batch_marker_closed_record_sha256,
        authorization.q8_remote_closure_binding_sha256,
    )
    if observed != expected:
        raise ValueError("BF16 durable phase authorization evidence drift")
    return authorization.batch_authorization


def _issue_bf16_submission_authorization(
    batch_authorization: DatabricksBatchReservationAuthorization,
    phase_lease_root: Path,
    *,
    q8_remote_closure_binding: Mapping[str, Any],
    durable_output_root: str,
) -> PublicationBF16HandoffSubmissionAuthorization:
    return PublicationBF16HandoffSubmissionAuthorization(
        batch_authorization=batch_authorization,
        phase_lease_root=phase_lease_root,
        q8_remote_closure_binding=q8_remote_closure_binding,
        durable_output_root=durable_output_root,
        _issuer=_SUBMISSION_AUTHORIZATION_ISSUER,
    )


def _validate_bf16_submission_phase_files(
    phase_lease_root: Path,
    batch_authorization: DatabricksBatchReservationAuthorization,
    *,
    q8_remote_closure_binding: Mapping[str, Any],
    durable_output_root: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(batch_authorization, DatabricksBatchReservationAuthorization):
        raise TypeError("BF16 phase batch authorization has the wrong type")
    root = Path(phase_lease_root).expanduser().absolute()
    _reject_symlink_path(root, include_leaf=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("BF16 submission authority requires a real phase lease")
    lease = _read_canonical_json_file(root / "phase-lease.json", "BF16 phase lease")
    expected_lease: dict[str, Any] = {
        "attempt_ids": list(batch_authorization.attempt_ids),
        "closed_record_sha256": "",
        "durable_output_root": _normalized_bf16_durable_output_root(
            durable_output_root
        ),
        "ledger_path_sha256": batch_authorization.ledger_path_sha256,
        "predecessor_prefix": batch_authorization.predecessor_prefix.to_record(),
        "q8_remote_closure": dict(q8_remote_closure_binding),
        "record_type": "cachet.publication_bf16_handoff_phase_lease.v1",
        "submit_payload_sha256": list(
            batch_authorization.submit_payload_sha256s
        ),
    }
    expected_lease["closed_record_sha256"] = _closed_record_sha256(expected_lease)
    if lease != expected_lease:
        raise ValueError("BF16 phase lease differs from the authorized atomic batch")
    marker = _read_canonical_json_file(
        root / "batch-reserved.json", "BF16 batch marker"
    )
    expected_marker: dict[str, Any] = {
        "batch_prefix": batch_authorization.batch_prefix.to_record(),
        "closed_record_sha256": "",
        "record_type": "cachet.publication_bf16_handoff_batch_reserved.v1",
    }
    expected_marker["closed_record_sha256"] = _closed_record_sha256(
        expected_marker
    )
    if marker != expected_marker:
        raise ValueError("BF16 batch marker differs from the authorized atomic batch")
    return lease, marker


def _normalized_bf16_durable_output_root(value: object) -> str:
    if not isinstance(value, str) or not value.rstrip("/"):
        raise ValueError("BF16 durable_output_root must be a non-empty string")
    return value.rstrip("/")


def _common_bf16_durable_output_root(
    worker_payloads: Sequence[Mapping[str, Any]],
) -> str:
    roots = {
        _normalized_bf16_durable_output_root(
            _required_string(payload, "durable_output_root")
        )
        for payload in worker_payloads
    }
    if len(roots) != 1:
        raise ValueError("BF16 workers must share one durable_output_root")
    return next(iter(roots))


def _bf16_phase_lease_root_sha256(path: Path) -> str:
    root = Path(path).expanduser().absolute()
    _reject_symlink_path(root, include_leaf=True)
    canonical = root.resolve(strict=True)
    return _canonical_sha256(
        {
            "domain": "cachet.publication.bf16_handoff.phase_lease_path.v1",
            "resolved_absolute_path": str(canonical),
        }
    )


@dataclass(frozen=True, slots=True)
class _BF16WorkerBatch:
    records: tuple[dict[str, Any], ...]
    task_ids: tuple[str, ...]
    cache_prefix_tokens: int
    input_token_slots: int
    generation_seconds: float
    durable_sync_seconds: float


def _generation_items(
    prepared: PreparedMainLatencyInputs,
    *,
    tokenizer: MainLatencyTokenizer,
) -> list[dict[str, Any]]:
    if MAIN_LATENCY_EXAMPLES_PER_DATASET != PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET:
        raise ValueError("main-latency and campaign example counts diverged")
    artifacts = tuple(
        item
        for item in prepared.files
        if item.input_tokens_target == PUBLICATION_BF16_HANDOFF_CONTEXT_TOKENS
    )
    if len(artifacts) != len(SUPPORTED_V1_DATASETS) or {
        item.dataset for item in artifacts
    } != set(SUPPORTED_V1_DATASETS):
        raise ValueError("prepared input bundle lacks the exact four 16k files")
    items: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for artifact in sorted(artifacts, key=lambda item: item.dataset):
        records = _canonical_jsonl_records(artifact.jsonl_path)
        if len(records) != PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET:
            raise ValueError("each BF16 input file must contain exactly 32 rows")
        relative_path = artifact.jsonl_path.relative_to(prepared.output_dir).as_posix()
        for row_index, row in enumerate(records):
            dataset = _required_string(row, "dataset")
            example_id = _required_string(row, "example_id")
            if dataset != artifact.dataset:
                raise ValueError("prepared row dataset differs from its file")
            identity = (dataset, example_id)
            if identity in identities:
                raise ValueError("BF16 suite contains a duplicate example identity")
            identities.add(identity)
            contracts = _cache_prefix_segment_token_contracts(
                row,
                tokenizer=tokenizer,
            )
            cache_prefix_tokens = sum(
                cast(int, item["token_count"]) for item in contracts
            )
            task_identity = {
                "context_tokens": PUBLICATION_BF16_HANDOFF_CONTEXT_TOKENS,
                "dataset": dataset,
                "example_id": example_id,
            }
            item: dict[str, Any] = {
                "cache_prefix_tokens": cache_prefix_tokens,
                "context_tokens": PUBLICATION_BF16_HANDOFF_CONTEXT_TOKENS,
                "dataset": dataset,
                "example_id": example_id,
                "input_token_slots": PUBLICATION_BF16_HANDOFF_CONTEXT_TOKENS,
                "prepared_jsonl_relative_path": relative_path,
                "prepared_record_sha256": _canonical_sha256(row),
                "row_index": row_index,
                "segment_token_contracts": contracts,
                "segment_token_contracts_sha256": _canonical_sha256(contracts),
                "task_id": (
                    f"{PUBLICATION_BF16_HANDOFF_CONTEXT_TOKENS}-{dataset}-"
                    f"{_canonical_sha256(task_identity)[:20]}"
                ),
            }
            item["assignment_sha256"] = _canonical_sha256(
                {
                    "domain": _PLAN_ORDER_DOMAIN,
                    "input_bundle_sha256": prepared.bundle_sha256,
                    "item": item,
                }
            )
            items.append(item)
    if len(items) != PUBLICATION_BF16_HANDOFF_TASK_COUNT:
        raise ValueError("BF16 generation requires exactly 128 tasks")
    task_ids = [cast(str, item["task_id"]) for item in items]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("BF16 task identifiers collide")
    return items


def _lpt_assignments(
    items: Sequence[dict[str, Any]],
    *,
    worker_count: int,
) -> tuple[list[dict[str, Any]], ...]:
    assignments: list[list[dict[str, Any]]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    for item in sorted(
        items,
        key=lambda value: (
            -cast(int, value["cache_prefix_tokens"]),
            cast(str, value["assignment_sha256"]),
        ),
    ):
        worker_index_value = min(
            range(worker_count),
            key=lambda index: (loads[index], index),
        )
        assignments[worker_index_value].append(item)
        loads[worker_index_value] += cast(int, item["cache_prefix_tokens"])
    if any(not assignment for assignment in assignments):
        raise ValueError("BF16 LPT sharding produced an empty worker")
    return tuple(assignments)


def _plan_task_ids(record: Mapping[str, Any]) -> tuple[str, ...]:
    task_ids = tuple(
        _required_string(item, "task_id")
        for worker in _mapping_sequence(record.get("workers"), "workers")
        for item in _mapping_sequence(worker.get("items"), "worker.items")
    )
    if len(task_ids) != PUBLICATION_BF16_HANDOFF_TASK_COUNT:
        raise ValueError("BF16 plan task coverage is incomplete")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("BF16 plan task coverage contains duplicates")
    return task_ids


def _validate_plan_envelope(record: Mapping[str, Any]) -> None:
    if record.get("record_type") != PUBLICATION_BF16_HANDOFF_PLAN_RECORD_TYPE:
        raise ValueError("BF16 plan record_type is invalid")
    if record.get("schema_version") != PUBLICATION_BF16_HANDOFF_SCHEMA_VERSION:
        raise ValueError("BF16 plan schema_version is invalid")
    if record.get("context_tokens") != PUBLICATION_BF16_HANDOFF_CONTEXT_TOKENS:
        raise ValueError("BF16 plan must target exactly 16k")
    if record.get("closed_record_sha256") != _closed_record_sha256(record):
        raise ValueError("BF16 plan closure is invalid")
    workers = _mapping_sequence(record.get("workers"), "workers")
    if len(workers) != PUBLICATION_BF16_HANDOFF_WORKER_COUNT:
        raise ValueError("BF16 plan requires exactly sixteen workers")
    if record.get("workers_sha256") != _canonical_sha256(list(workers)):
        raise ValueError("BF16 worker-plan digest is invalid")
    _require_sha256(
        _required_string(record, "input_bundle_sha256"), "input_bundle_sha256"
    )
    _plan_task_ids(record)


def _hardware_qualification_record(
    value: PublicationLatencyGeneratorHardwareQualification,
) -> dict[str, Any]:
    selection = value.selection
    record = {
        "evidence_closed_record_sha256": _required_string(
            value.evidence_record,
            "closed_record_sha256",
        ),
        "evidence_file_sha256": value.evidence_file_sha256,
        "evidence_uri": value.evidence_uri,
        "expected_artifact_pins": value.expected_artifact_pins.to_record(),
        "expected_campaign_id": value.expected_campaign_id,
        "generation_artifacts_sha256": selection.generation_artifacts_sha256,
        "generation_databricks_node_type_id": (
            selection.generation_databricks_node_type_id
        ),
        "generation_hardware_id": selection.generation_hardware_id,
        "generation_prefix_tokens_per_second": (
            selection.generation_prefix_tokens_per_second
        ),
        "plan_closed_record_sha256": selection.plan_sha256,
        "plan_file_sha256": value.plan_file_sha256,
        "plan_uri": value.plan_uri,
    }
    _validate_qualification_record(record)
    return record


def _qualification_pins(record: Mapping[str, Any]) -> GPUQualificationArtifactPins:
    pins = _required_mapping(record, "expected_artifact_pins")
    return GPUQualificationArtifactPins(
        runtime_lock_sha256=_required_string(pins, "runtime_lock_sha256"),
        patched_vllm_wheel_sha256=_required_string(
            pins,
            "patched_vllm_wheel_sha256",
        ),
        package_wheel_sha256=_required_string(pins, "package_wheel_sha256"),
        cachet_source_tree_sha256=_required_string(
            pins,
            "cachet_source_tree_sha256",
        ),
        runner_sha256=_required_string(pins, "runner_sha256"),
        input_bundle_sha256=_required_string(pins, "input_bundle_sha256"),
    )


def _validate_qualification_record(record: Mapping[str, Any]) -> None:
    for name in (
        "evidence_closed_record_sha256",
        "evidence_file_sha256",
        "generation_artifacts_sha256",
        "plan_closed_record_sha256",
        "plan_file_sha256",
    ):
        _require_sha256(_required_string(record, name), name)
    for name in ("evidence_uri", "expected_campaign_id", "plan_uri"):
        _nonempty_string(_required_string(record, name), name)
    _qualification_pins(record)
    if record.get("generation_hardware_id") != (
        PUBLICATION_LATENCY_HANDOFF_GENERATOR_HARDWARE_TARGET
    ) or record.get("generation_databricks_node_type_id") != (
        PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID
    ):
        raise ValueError("BF16 producer qualification must select one L40S")
    if (
        _required_positive_number(
            record,
            "generation_prefix_tokens_per_second",
        )
        < PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
    ):
        raise ValueError("BF16 producer qualification is below 35 tokens/s")


def _require_authorized_worker_payload(
    payload: Mapping[str, Any],
    authorization: object,
    *,
    worker_index_value: int,
    expected_worker_sha256: str | None = None,
) -> GPUQualificationSelection:
    _worker_index(worker_index_value)
    if (
        payload.get("record_type")
        != PUBLICATION_BF16_HANDOFF_WORKER_PAYLOAD_RECORD_TYPE
        or payload.get("schema_version") != PUBLICATION_BF16_HANDOFF_SCHEMA_VERSION
        or payload.get("closed_record_sha256") != _closed_record_sha256(payload)
    ):
        raise ValueError("BF16 worker payload closure is invalid")
    if _required_int(payload, "worker_index") != worker_index_value:
        raise ValueError("BF16 worker payload identity differs")
    _validate_local_nvme_worker_root(
        _required_string(payload, "local_work_root"),
        worker_index=worker_index_value,
        field_name="local_work_root",
    )
    _execution_config_from_record(_required_mapping(payload, "execution_contract"))
    expected_output = {
        "partial_record_relative_path": (
            f"worker-records/worker-{worker_index_value:02d}.jsonl"
        ),
        "result_relative_path": f"worker-results/worker-{worker_index_value:02d}.json",
        "worker_bundle_relative_root": (
            f"pending/{PUBLICATION_BF16_HANDOFF_CONTEXT_TOKENS}/"
            f"worker-{worker_index_value:02d}"
        ),
    }
    if dict(_required_mapping(payload, "output_binding")) != expected_output:
        raise ValueError("BF16 worker output binding drift")
    observed_sha256 = _worker_payload_file_sha256(payload)
    if expected_worker_sha256 is not None and not hmac.compare_digest(
        observed_sha256,
        _require_sha256(expected_worker_sha256, "expected_worker_sha256"),
    ):
        raise ValueError("BF16 worker payload changed after authorization preflight")
    qualification = _required_mapping(
        payload,
        "generator_hardware_qualification",
    )
    _validate_qualification_record(qualification)
    selection = require_gpu_qualification_launch_authorization(
        authorization,
        expected_plan_sha256=_required_string(
            qualification,
            "plan_closed_record_sha256",
        ),
        expected_evidence_file_sha256=_required_string(
            qualification,
            "evidence_file_sha256",
        ),
    )
    if not isinstance(authorization, GPUQualificationLaunchAuthorization):
        raise TypeError("BF16 launch requires GPUQualificationLaunchAuthorization")
    if authorization.evidence_closed_record_sha256 != _required_string(
        qualification,
        "evidence_closed_record_sha256",
    ):
        raise ValueError("BF16 launch authority evidence closure differs")
    exact = {
        "generation_artifacts_sha256": selection.generation_artifacts_sha256,
        "generation_databricks_node_type_id": (
            selection.generation_databricks_node_type_id
        ),
        "generation_hardware_id": selection.generation_hardware_id,
        "generation_prefix_tokens_per_second": (
            selection.generation_prefix_tokens_per_second
        ),
        "plan_closed_record_sha256": selection.plan_sha256,
    }
    if any(qualification.get(key) != value for key, value in exact.items()):
        raise ValueError("BF16 launch authority selection differs from worker payload")
    pins = _qualification_pins(qualification)
    source = _required_mapping(payload, "source")
    _resolved_source_revision(_required_string(source, "source_revision"))
    if (
        payload.get("input_bundle_sha256") != pins.input_bundle_sha256
        or source.get("cachet_source_tree_sha256") != pins.cachet_source_tree_sha256
    ):
        raise ValueError("BF16 worker differs from qualified input/source artifacts")
    return selection


def _validate_job_artifact_binding(
    config: DatabricksPublicationBF16HandoffJobConfig,
    payload: Mapping[str, Any],
) -> None:
    qualification = _required_mapping(
        payload,
        "generator_hardware_qualification",
    )
    pins = _qualification_pins(qualification)
    source = _required_mapping(payload, "source")
    expected = {
        "package_wheel_sha256": config.package_wheel_sha256,
        "patched_vllm_wheel_sha256": config.patched_vllm_wheel_sha256,
        "runtime_lock_sha256": config.runtime_lock_sha256,
        "cachet_source_tree_sha256": config.cachet_source_tree_sha256,
    }
    observed = {
        "package_wheel_sha256": pins.package_wheel_sha256,
        "patched_vllm_wheel_sha256": pins.patched_vllm_wheel_sha256,
        "runtime_lock_sha256": pins.runtime_lock_sha256,
        "cachet_source_tree_sha256": pins.cachet_source_tree_sha256,
    }
    if expected != observed:
        raise ValueError("BF16 job artifacts differ from GPU qualification")
    if source.get("source_revision") != config.source_revision:
        raise ValueError("BF16 job source revision differs from worker payload")


def _worker_payload_file_sha256(payload: Mapping[str, Any]) -> str:
    return sha256(_canonical_json_bytes(payload, pretty=True)).hexdigest()


def _is_durable_plan_root(value: str, *, plan_sha256: str) -> bool:
    _require_sha256(plan_sha256, "plan_sha256")
    if value.startswith("dbfs:/"):
        normalized = value.removeprefix("dbfs:/")
    elif value.startswith("uc-volume:"):
        normalized = value.removeprefix("uc-volume:").lstrip("/")
    elif value.startswith("/Volumes/"):
        normalized = value.lstrip("/")
    else:
        return False
    parts = normalized.split("/")
    if len(parts) < 3 or any(not part or part in {".", ".."} for part in parts):
        return False
    if parts[0] == "Volumes" and len(parts) < 6:
        return False
    return parts[-2:] == ["publication-bf16-handoffs", plan_sha256]


def _l40s_cluster(config: DatabricksPublicationBF16HandoffJobConfig) -> dict[str, Any]:
    cluster: dict[str, Any] = {
        "aws_attributes": {"availability": "ON_DEMAND", "zone_id": "auto"},
        "custom_tags": {
            "ResourceClass": "SingleNode",
            "gpu_model": PUBLICATION_LATENCY_HANDOFF_GENERATOR_GPU_MODEL,
            "hardware_target": PUBLICATION_LATENCY_HANDOFF_GENERATOR_HARDWARE_TARGET,
            "purpose": "cachet-vllm-0271-bf16-handoff-generation",
        },
        "data_security_mode": config.data_security_mode,
        "driver_node_type_id": config.node_type_id,
        "node_type_id": config.node_type_id,
        "num_workers": 0,
        "spark_conf": {
            "spark.databricks.cluster.profile": "singleNode",
            "spark.master": "local[*]",
        },
        "spark_version": config.spark_version,
    }
    if config.data_security_mode == "SINGLE_USER":
        cluster["single_user_name"] = config.single_user_name
    return cluster


def _spark_parameter(payload: Mapping[str, Any], flag: str) -> str:
    task = _mapping_sequence(payload.get("tasks"), "tasks")
    if len(task) != 1:
        raise ValueError("BF16 submit payload must contain one task")
    parameters = _required_mapping(task[0], "spark_python_task").get("parameters")
    if (
        not isinstance(parameters, Sequence)
        or isinstance(parameters, (str, bytes, bytearray))
        or any(not isinstance(item, str) for item in parameters)
    ):
        raise ValueError("BF16 spark parameters are invalid")
    positions = [index for index, item in enumerate(parameters) if item == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(parameters):
        raise ValueError(f"BF16 spark parameters require exactly one {flag}")
    value = parameters[positions[0] + 1]
    if not isinstance(value, str) or not value:
        raise ValueError(f"BF16 spark parameter {flag} is empty")
    return value


def _validate_submit_payload(
    payload: Mapping[str, Any], *, worker_index_value: int
) -> None:
    _worker_index(worker_index_value)
    if payload.get("timeout_seconds") != PUBLICATION_BF16_HANDOFF_TASK_TIMEOUT_SECONDS:
        raise ValueError("BF16 run timeout must equal five hours")
    tasks = _mapping_sequence(payload.get("tasks"), "tasks")
    if len(tasks) != 1:
        raise ValueError("BF16 producer run must contain one task")
    task = tasks[0]
    if (
        task.get("task_key") != f"bf16_handoff_worker_{worker_index_value:02d}"
        or task.get("timeout_seconds") != PUBLICATION_BF16_HANDOFF_TASK_TIMEOUT_SECONDS
        or task.get("max_retries") != 0
    ):
        raise ValueError("BF16 task identity/timeout/retry contract is invalid")
    cluster = _required_mapping(task, "new_cluster")
    if (
        cluster.get("node_type_id")
        != PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID
        or cluster.get("driver_node_type_id")
        != PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID
        or cluster.get("num_workers") != 0
    ):
        raise ValueError("BF16 task must use one g6e.4xlarge L40S")


def _validate_launch_binding(
    submit_payload: Mapping[str, Any],
    *,
    worker_payload: Mapping[str, Any],
    worker_index_value: int,
    authorization: GPUQualificationLaunchAuthorization,
    expected_worker_sha256: str,
) -> None:
    _validate_submit_payload(submit_payload, worker_index_value=worker_index_value)
    _require_authorized_worker_payload(
        worker_payload,
        authorization,
        worker_index_value=worker_index_value,
        expected_worker_sha256=expected_worker_sha256,
    )
    qualification = _required_mapping(
        worker_payload,
        "generator_hardware_qualification",
    )
    pins = _qualification_pins(qualification)
    expected_parameters = {
        "--expected-worker-payload-sha256": expected_worker_sha256,
        "--package-wheel-sha256": pins.package_wheel_sha256,
        "--patched-vllm-wheel-sha256": pins.patched_vllm_wheel_sha256,
        "--runtime-lock-sha256": pins.runtime_lock_sha256,
    }
    for flag, expected in expected_parameters.items():
        if not hmac.compare_digest(_spark_parameter(submit_payload, flag), expected):
            raise ValueError(f"BF16 submit parameter {flag} differs from authority")


def _reservation_validator(
    ledger_path: Path,
    *,
    worker_payload: Mapping[str, Any],
    worker_index_value: int,
    authorization: GPUQualificationLaunchAuthorization,
    q8_authorization: PublicationHandoffRemoteClosureAuthorization,
    expected_worker_sha256: str,
) -> Callable[[DatabricksClusterHourReservation, Mapping[str, Any]], None]:
    def validate(
        reservation: DatabricksClusterHourReservation,
        snapshot: Mapping[str, Any],
    ) -> None:
        _validate_launch_binding(
            snapshot,
            worker_payload=worker_payload,
            worker_index_value=worker_index_value,
            authorization=authorization,
            expected_worker_sha256=expected_worker_sha256,
        )
        if reservation.reserved_cluster_hours != 5.0:
            raise ValueError("one BF16 producer must reserve five GPU-hours")
        ledger = _require_matching_predecessor_ledger(
            ledger_path,
            authorization,
            q8_authorization,
            expected_input_bundle_sha256=_required_string(
                worker_payload, "input_bundle_sha256"
            ),
            expected_qualification_closed_record_sha256=_required_string(
                _required_mapping(
                    worker_payload, "generator_hardware_qualification"
                ),
                "evidence_closed_record_sha256",
            ),
        )
        if ledger.cap_cluster_hours != MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS:
            raise ValueError("BF16 generation requires the 1024-hour ledger")
        if (
            ledger.active_reserved_task_count + len(reservation.task_timeout_seconds)
            > PUBLICATION_CAMPAIGN_MAX_PARALLEL_JOBS
        ):
            raise ValueError(
                "BF16 reservation exceeds the global 16-job concurrency cap"
            )
        if (
            ledger.active_reserved_cluster_hours + reservation.reserved_cluster_hours
            > MAX_DATABRICKS_ACTIVE_RESERVED_CLUSTER_HOURS
        ):
            raise ValueError("BF16 reservation exceeds the active 900-hour cap")
        if (
            ledger.accounted_cluster_hours + reservation.reserved_cluster_hours
            > MAX_DATABRICKS_AGGREGATE_CLUSTER_HOURS
            - PUBLICATION_CAMPAIGN_UNRESERVED_HEADROOM_HOURS
        ):
            raise ValueError("BF16 reservation consumes the 124-hour headroom")

    return validate


def _execution_config_record(
    config: PublicationBF16HandoffExecutionConfig,
) -> dict[str, Any]:
    layout = config.layout
    return {
        "align_bytes": config.align_bytes,
        "generator_add_special_tokens": config.generator_add_special_tokens,
        "generator_cache_axis_order": config.generator_cache_axis_order,
        "generator_device_map": config.generator_device_map,
        "generator_family": config.generator_family,
        "generator_model_dtype": config.generator_model_dtype,
        "generator_quantization": config.generator_quantization,
        "generator_quantization_config": dict(config.generator_quantization_config),
        "generator_trust_remote_code": config.generator_trust_remote_code,
        "generator_version": config.generator_version,
        "layout": {
            "block_size": layout.block_size,
            "bytes_per_token": layout.bytes_per_token,
            "dtype": layout.dtype,
            "head_size": layout.head_size,
            "key_position_encoding": _enum_string(layout.key_position_encoding),
            "kv_stride_bytes": layout.kv_stride_bytes,
            "layout_version": layout.layout_version,
            "lora_id": layout.lora_id,
            "model_id": layout.model_id,
            "num_kv_heads": layout.num_kv_heads,
            "num_layers": layout.num_layers,
            "num_query_heads": layout.num_query_heads,
            "payload_axis_order": _enum_string(layout.payload_axis_order),
            "pre_rope": layout.pre_rope,
            "rope_rotary_dim": layout.rope_rotary_dim,
            "rope_theta": layout.rope_theta,
            "shares_kv_storage": layout.shares_kv_storage,
            "storage_layout": _enum_string(layout.storage_layout),
        },
        "model_revision": config.model_revision,
        "pipeline_parallel_size": config.pipeline_parallel_size,
        "tensor_parallel_size": config.tensor_parallel_size,
        "tokenizer_id": config.tokenizer_id,
        "tokenizer_revision": config.tokenizer_revision,
        "vllm_bitsandbytes_loader_source_sha256": (
            config.vllm_bitsandbytes_loader_source_sha256
        ),
    }


def _execution_config_from_record(
    record: Mapping[str, Any],
) -> PublicationBF16HandoffExecutionConfig:
    layout = _required_mapping(record, "layout")
    return PublicationBF16HandoffExecutionConfig(
        layout=KVLayout(
            model_id=_required_string(layout, "model_id"),
            lora_id=_required_string(layout, "lora_id"),
            layout_version=_required_string(layout, "layout_version"),
            dtype=_required_string(layout, "dtype"),
            num_layers=_required_int(layout, "num_layers"),
            block_size=_required_int(layout, "block_size"),
            bytes_per_token=_required_int(layout, "bytes_per_token"),
            num_query_heads=_optional_int(layout, "num_query_heads"),
            num_kv_heads=_optional_int(layout, "num_kv_heads"),
            head_size=_optional_int(layout, "head_size"),
            kv_stride_bytes=_optional_int(layout, "kv_stride_bytes"),
            shares_kv_storage=_required_bool(layout, "shares_kv_storage"),
            storage_layout=_required_string(layout, "storage_layout"),
            payload_axis_order=_required_string(layout, "payload_axis_order"),
            pre_rope=_required_bool(layout, "pre_rope"),
            rope_theta=_optional_number(layout, "rope_theta"),
            rope_rotary_dim=_optional_int(layout, "rope_rotary_dim"),
            key_position_encoding=_required_string(
                layout,
                "key_position_encoding",
            ),
        ),
        model_revision=_required_string(record, "model_revision"),
        generator_version=_required_string(record, "generator_version"),
        vllm_bitsandbytes_loader_source_sha256=_required_string(
            record,
            "vllm_bitsandbytes_loader_source_sha256",
        ),
        tokenizer_id=_required_string(record, "tokenizer_id"),
        tokenizer_revision=_required_string(record, "tokenizer_revision"),
        generator_family=_required_string(record, "generator_family"),
        generator_device_map=_required_string(record, "generator_device_map"),
        generator_quantization=_required_string(record, "generator_quantization"),
        generator_model_dtype=_required_string(record, "generator_model_dtype"),
        generator_cache_axis_order=_required_string(
            record,
            "generator_cache_axis_order",
        ),
        generator_trust_remote_code=_required_bool(
            record,
            "generator_trust_remote_code",
        ),
        generator_add_special_tokens=_required_bool(
            record,
            "generator_add_special_tokens",
        ),
        generator_quantization_config=_required_mapping(
            record,
            "generator_quantization_config",
        ),
        tensor_parallel_size=_required_int(record, "tensor_parallel_size"),
        pipeline_parallel_size=_required_int(record, "pipeline_parallel_size"),
        align_bytes=_required_int(record, "align_bytes"),
    )


def _read_bound_closed_json(
    uri: str,
    *,
    file_sha256: str,
    closed_record_sha256: str,
    field_name: str,
) -> dict[str, Any]:
    path = Path(local_path(uri)).expanduser().absolute()
    _reject_symlink_path(path, include_leaf=True)
    content = path.read_bytes()
    if not hmac.compare_digest(
        sha256(content).hexdigest(),
        _require_sha256(file_sha256, f"{field_name} file_sha256"),
    ):
        raise ValueError(f"{field_name} file SHA-256 drift")
    record = _json_object(content, field_name=field_name)
    if content != _canonical_json_bytes(record, pretty=True):
        raise ValueError(f"{field_name} is not canonical JSON")
    if record.get("closed_record_sha256") != closed_record_sha256 or (
        record.get("closed_record_sha256") != _closed_record_sha256(record)
    ):
        raise ValueError(f"{field_name} closed-record binding drift")
    return record


def _read_bound_qualification_json(
    uri: str,
    *,
    file_sha256: str,
    closed_record_sha256: str,
    field_name: str,
) -> dict[str, Any]:
    path = Path(local_path(uri)).expanduser().absolute()
    _reject_symlink_path(path, include_leaf=True)
    content = path.read_bytes()
    if not hmac.compare_digest(
        sha256(content).hexdigest(),
        _require_sha256(file_sha256, f"{field_name} file_sha256"),
    ):
        raise ValueError(f"{field_name} file SHA-256 drift")
    record = _json_object(content, field_name=field_name)
    expected = (canonical_gpu_qualification_json(record) + "\n").encode("utf-8")
    if content != expected:
        raise ValueError(f"{field_name} is not canonical JSON")
    if record.get("closed_record_sha256") != closed_record_sha256:
        raise ValueError(f"{field_name} closure binding drift")
    return record


def _verify_bound_hardware_qualification_file(binding: Mapping[str, Any]) -> None:
    _validate_qualification_record(binding)
    evidence = _read_bound_qualification_json(
        _required_string(binding, "evidence_uri"),
        file_sha256=_required_string(binding, "evidence_file_sha256"),
        closed_record_sha256=_required_string(
            binding,
            "evidence_closed_record_sha256",
        ),
        field_name="GPU qualification evidence",
    )
    plan = _read_bound_qualification_json(
        _required_string(binding, "plan_uri"),
        file_sha256=_required_string(binding, "plan_file_sha256"),
        closed_record_sha256=_required_string(
            binding,
            "plan_closed_record_sha256",
        ),
        field_name="GPU qualification plan",
    )
    selection = validate_gpu_qualification_evidence_record(
        evidence,
        plan_record=plan,
        expected_campaign_id=_required_string(binding, "expected_campaign_id"),
        expected_artifact_pins=_qualification_pins(binding),
    )
    expected = {
        "generation_artifacts_sha256": selection.generation_artifacts_sha256,
        "generation_databricks_node_type_id": (
            selection.generation_databricks_node_type_id
        ),
        "generation_hardware_id": selection.generation_hardware_id,
        "generation_prefix_tokens_per_second": (
            selection.generation_prefix_tokens_per_second
        ),
        "plan_closed_record_sha256": selection.plan_sha256,
    }
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError("BF16 worker qualification selection binding drift")


def _prepared_rows_by_task(
    prepared: PreparedMainLatencyInputs,
) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for artifact in prepared.files:
        if artifact.input_tokens_target != PUBLICATION_BF16_HANDOFF_CONTEXT_TOKENS:
            continue
        for row_index, row in enumerate(_canonical_jsonl_records(artifact.jsonl_path)):
            key = (artifact.dataset, row_index)
            if key in rows:
                raise ValueError("BF16 prepared row key collision")
            rows[key] = row
    return rows


def _apply_production_generator_environment(
    config: PublicationBF16HandoffExecutionConfig,
) -> None:
    fixed = {
        CACHET_TRANSFORMERS_ADD_SPECIAL_TOKENS_ENV: (
            "1" if config.generator_add_special_tokens else "0"
        ),
        CACHET_TRANSFORMERS_CACHE_AXIS_ORDER_ENV: config.generator_cache_axis_order,
        CACHET_TRANSFORMERS_DEVICE_ENV: "cuda",
        CACHET_TRANSFORMERS_DEVICE_MAP_ENV: config.generator_device_map,
        CACHET_TRANSFORMERS_MODEL_ID_ENV: config.layout.model_id,
        CACHET_TRANSFORMERS_MODEL_REVISION_ENV: config.model_revision,
        CACHET_TRANSFORMERS_PRE_ROPE_ENV: "1",
        CACHET_TRANSFORMERS_QUANTIZATION_ENV: config.generator_quantization,
        CACHET_TRANSFORMERS_QUANTIZATION_CONFIG_JSON_ENV: json.dumps(
            dict(config.generator_quantization_config),
            sort_keys=True,
            separators=(",", ":"),
        ),
        CACHET_TRANSFORMERS_TOKENIZER_ID_ENV: config.tokenizer_id,
        CACHET_TRANSFORMERS_TOKENIZER_REVISION_ENV: config.tokenizer_revision,
        CACHET_TRANSFORMERS_TORCH_DTYPE_ENV: config.generator_model_dtype,
        CACHET_TRANSFORMERS_TRUST_REMOTE_CODE_ENV: (
            "1" if config.generator_trust_remote_code else "0"
        ),
    }
    for key, value in fixed.items():
        observed = os.environ.get(key)
        if observed is not None and observed != value:
            raise ValueError(f"runtime environment {key} conflicts with BF16 contract")
        os.environ[key] = value


def _verify_installed_vllm_bitsandbytes_loader_source(
    config: PublicationBF16HandoffExecutionConfig,
) -> None:
    module = importlib.import_module(
        "vllm.model_executor.model_loader.bitsandbytes_loader"
    )
    source = getattr(module, "__file__", None)
    if not isinstance(source, str) or not source:
        raise RuntimeError("vLLM BitsAndBytes loader has no source file")
    path = Path(source).resolve()
    if path.suffix == ".pyc" and path.with_suffix(".py").is_file():
        path = path.with_suffix(".py")
    if not hmac.compare_digest(
        _file_sha256(path), config.vllm_bitsandbytes_loader_source_sha256
    ):
        raise ValueError("vLLM BitsAndBytes loader source SHA-256 drift")


def _production_generator_factory(worker_index: int) -> KVChunkGenerator:
    del worker_index
    return build_pre_rope_transformers_kv_chunk_generator()


def _probe_single_l40s_hardware() -> Mapping[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - production dependency.
        raise RuntimeError(
            "torch is required to attest BF16 producer hardware"
        ) from exc
    if int(torch.cuda.device_count()) != 1:
        raise ValueError("BF16 producer requires exactly one visible GPU")
    properties = torch.cuda.get_device_properties(0)
    return {
        "cuda_device_count": 1,
        "cuda_device_name": str(torch.cuda.get_device_name(0)),
        "cuda_device_total_memory_bytes": int(properties.total_memory),
        "cuda_major": int(properties.major),
        "cuda_minor": int(properties.minor),
        "gpu_model": PUBLICATION_LATENCY_HANDOFF_GENERATOR_GPU_MODEL,
        "hardware_target": PUBLICATION_LATENCY_HANDOFF_GENERATOR_HARDWARE_TARGET,
        "node_type_id": PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID,
    }


def _validate_observed_l40s_hardware(record: Mapping[str, Any]) -> None:
    if (
        record.get("cuda_device_count") != 1
        or record.get("hardware_target")
        != PUBLICATION_LATENCY_HANDOFF_GENERATOR_HARDWARE_TARGET
        or record.get("gpu_model") != PUBLICATION_LATENCY_HANDOFF_GENERATOR_GPU_MODEL
        or record.get("node_type_id")
        != PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID
    ):
        raise ValueError("observed BF16 producer hardware is not one qualified L40S")
    if _required_int(record, "cuda_device_total_memory_bytes") <= 0:
        raise ValueError("observed L40S memory must be positive")


def _validate_generator(
    generator: object,
    *,
    config: PublicationBF16HandoffExecutionConfig,
) -> None:
    if not callable(getattr(generator, "generate", None)):
        raise TypeError("BF16 generator must implement generate")
    if getattr(generator, "pre_rope", None) is not True:
        raise ValueError("BF16 producer must capture pre-RoPE KV")
    expected_identity = {
        "cache_axis_order": config.generator_cache_axis_order,
        "generator_family": config.generator_family,
        "generator_version": config.generator_version,
        "model_id": config.layout.model_id,
        "model_revision": config.model_revision,
        "tokenizer_id": config.tokenizer_id,
        "tokenizer_revision": config.tokenizer_revision,
    }
    if any(
        getattr(generator, key, None) != value
        for key, value in expected_identity.items()
    ):
        raise ValueError("BF16 production generator identity differs from contract")
    resolved = getattr(generator, "config", None)
    expected_resolved = {
        "add_special_tokens": config.generator_add_special_tokens,
        "cache_axis_order": config.generator_cache_axis_order,
        "device": "cuda",
        "device_map": config.generator_device_map,
        "model_id": config.layout.model_id,
        "model_revision": config.model_revision,
        "quantization": config.generator_quantization,
        "tokenizer_revision": config.tokenizer_revision,
        "torch_dtype": config.generator_model_dtype,
        "trust_remote_code": config.generator_trust_remote_code,
    }
    if resolved is None or any(
        getattr(resolved, key, None) != value
        for key, value in expected_resolved.items()
    ):
        raise ValueError("BF16 production generator configuration drift")
    if (
        getattr(resolved, "resolved_tokenizer_id", None) != config.tokenizer_id
        or dict(getattr(resolved, "quantization_config", {}))
        != dict(config.generator_quantization_config)
        or dict(getattr(resolved, "model_kwargs", {}))
        or dict(getattr(resolved, "tokenizer_kwargs", {}))
    ):
        raise ValueError("BF16 production generator loader contract drift")


def _execute_worker_batch(
    worker: Mapping[str, Any],
    *,
    rows_by_task: Mapping[tuple[str, int], dict[str, Any]],
    work_root: Path,
    bundle_root: Path,
    generator: KVChunkGenerator,
    config: PublicationBF16HandoffExecutionConfig,
    clock: Callable[[], float],
) -> _BF16WorkerBatch:
    _worker_index(_required_int(worker, "worker_index"))
    items = _mapping_sequence(worker.get("items"), "worker.items")
    records: list[dict[str, Any]] = []
    task_ids: list[str] = []
    for item in items:
        key = (_required_string(item, "dataset"), _required_int(item, "row_index"))
        try:
            row = rows_by_task[key]
        except KeyError as exc:
            raise ValueError(f"planned BF16 input row is missing: {key}") from exc
        if _canonical_sha256(row) != _required_string(item, "prepared_record_sha256"):
            raise ValueError("planned BF16 input row changed before generation")
        if row.get("example_id") != item.get("example_id"):
            raise ValueError("planned BF16 input identity changed before generation")
        records.append(row)
        task_ids.append(_required_string(item, "task_id"))
    input_jsonl = work_root / "input.jsonl"
    _write_jsonl_exclusive(records, input_jsonl)
    generation_start = clock()
    result = generate_benchmark_handoff_bundles(
        input_jsonl,
        output_dir=bundle_root,
        generator=generator,
        layout=config.layout,
        backend="vllm",
        model_id=config.layout.model_id,
        lora_id=config.layout.lora_id,
        cache_method=CacheGenerationMethod.VANILLA_PREFILL,
        model_revision=config.model_revision,
        tokenizer_id=config.tokenizer_id,
        tokenizer_revision=config.tokenizer_revision,
        generator_family=config.generator_family,
        generator_version=config.generator_version,
        tensor_parallel_size=config.tensor_parallel_size,
        pipeline_parallel_size=config.pipeline_parallel_size,
        storage_layout=config.layout.storage_layout,
        segment_per_document=True,
        align_bytes=config.align_bytes,
    )
    shard_path = Path(local_path(result.shard_uri)).expanduser().absolute()
    _require_confined_path(bundle_root, shard_path, field_name="intermediate shard")
    if not shard_path.is_file() or shard_path.is_symlink():
        raise ValueError("BF16 intermediate shard is missing or unsafe")
    shard_path.unlink()
    enriched = enrich_benchmark_records_with_handoffs(
        records,
        result.manifest,
        arm_id=PUBLICATION_BF16_HANDOFF_ARM_ID,
    )
    expected_keys = {
        (_required_string(item, "dataset"), _required_string(item, "example_id"))
        for item in items
    }
    observed_keys = {
        (_required_string(row, "dataset"), _required_string(row, "example_id"))
        for row in enriched
    }
    if observed_keys != expected_keys or len(enriched) != len(expected_keys):
        raise ValueError("BF16 worker generated incomplete/duplicate coverage")
    expected_tokens = {
        (_required_string(item, "dataset"), _required_string(item, "example_id")): (
            _required_int(item, "cache_prefix_tokens")
        )
        for item in items
    }
    expected_contracts = {
        (_required_string(item, "dataset"), _required_string(item, "example_id")): [
            dict(contract)
            for contract in _mapping_sequence(
                item.get("segment_token_contracts"),
                "segment_token_contracts",
            )
        ]
        for item in items
    }
    for row in enriched:
        identity = (
            _required_string(row, "dataset"),
            _required_string(row, "example_id"),
        )
        if _handoff_total_tokens(row) != expected_tokens[identity]:
            raise ValueError("BF16 handoff token count differs from the plan")
        if _handoff_segment_token_contracts(row) != expected_contracts[identity]:
            raise ValueError("BF16 handoff token identities differ from the plan")
    generation_seconds = _positive_duration(clock() - generation_start)
    durable_start = clock()
    _sync_tree(bundle_root)
    durable_seconds = _nonnegative_duration(clock() - durable_start)
    return _BF16WorkerBatch(
        records=tuple(enriched),
        task_ids=tuple(task_ids),
        cache_prefix_tokens=sum(expected_tokens.values()),
        input_token_slots=sum(
            _required_int(item, "input_token_slots") for item in items
        ),
        generation_seconds=generation_seconds,
        durable_sync_seconds=durable_seconds,
    )


def run_publication_bf16_handoff_worker(
    worker_payload_json: str | Path,
    *,
    expected_worker_payload_sha256: str,
) -> dict[str, Any]:
    """Run one production shard; publication closure still requires cloud evidence."""

    expected = _require_sha256(
        expected_worker_payload_sha256,
        "expected_worker_payload_sha256",
    )
    payload_path = Path(local_path(str(worker_payload_json))).expanduser().absolute()
    _reject_symlink_path(payload_path, include_leaf=True)
    payload_bytes = payload_path.read_bytes()
    if not hmac.compare_digest(sha256(payload_bytes).hexdigest(), expected):
        raise ValueError("BF16 worker payload file SHA-256 differs")
    payload = _json_object(payload_bytes, field_name="BF16 worker payload")
    if payload_bytes != _canonical_json_bytes(payload, pretty=True):
        raise ValueError("BF16 worker payload is not canonical JSON")
    plan_binding = _required_mapping(payload, "plan")
    plan = _read_bound_closed_json(
        _required_string(plan_binding, "uri"),
        file_sha256=_required_string(plan_binding, "file_sha256"),
        closed_record_sha256=_required_string(plan_binding, "closed_record_sha256"),
        field_name="BF16 generation plan",
    )
    validate_publication_bf16_handoff_worker_payload(payload, plan=plan)
    prepared_binding = _required_mapping(payload, "prepared_inputs")
    prepared_root = (
        Path(local_path(_required_string(prepared_binding, "uri")))
        .expanduser()
        .absolute()
    )
    _reject_symlink_path(prepared_root, include_leaf=True)
    provenance_path = prepared_root / MAIN_LATENCY_PROVENANCE_FILENAME
    _verify_file_sha256(
        provenance_path,
        _required_string(prepared_binding, "provenance_file_sha256"),
        field_name="prepared input provenance",
    )
    provenance = _json_object(
        provenance_path.read_bytes(),
        field_name="prepared input provenance",
    )
    if provenance.get("closed_record_sha256") != prepared_binding.get(
        "provenance_closed_record_sha256"
    ):
        raise ValueError("prepared input provenance closure differs")
    qualification = _required_mapping(
        payload,
        "generator_hardware_qualification",
    )
    _verify_bound_hardware_qualification_file(qualification)
    tokenizer = load_main_latency_tokenizer()
    validate_publication_bf16_handoff_generation_plan(
        plan,
        prepared_input_dir=prepared_root,
        tokenizer=tokenizer,
    )
    prepared = verify_main_latency_inputs(
        prepared_root,
        tokenizer=tokenizer,
        examples_per_dataset=PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
    )
    if prepared.bundle_sha256 != payload.get("input_bundle_sha256"):
        raise ValueError("BF16 worker prepared input bundle differs")
    worker_index_value = _required_int(payload, "worker_index")
    worker = _mapping_sequence(plan.get("workers"), "workers")[worker_index_value]
    output_root = (
        Path(local_path(_required_string(payload, "durable_output_root")))
        .expanduser()
        .absolute()
    )
    _reject_symlink_path(output_root, include_leaf=True)
    output_root.mkdir(parents=True, exist_ok=True)
    _require_worker_output_fresh(payload, output_root=output_root)
    work_root = (
        Path(_required_string(payload, "local_work_root")).expanduser().absolute()
    )
    _reject_symlink_path(work_root, include_leaf=True)
    _require_fresh_path(work_root)
    work_root.mkdir(parents=True)
    config = _execution_config_from_record(
        _required_mapping(payload, "execution_contract")
    )
    _apply_production_generator_environment(config)
    _verify_installed_vllm_bitsandbytes_loader_source(config)
    observed_hardware = dict(_probe_single_l40s_hardware())
    _validate_observed_l40s_hardware(observed_hardware)
    rows_by_task = _prepared_rows_by_task(prepared)
    producer_start = _monotonic_clock()
    generator: KVChunkGenerator | None = None
    try:
        generator = _production_generator_factory(worker_index_value)
        _validate_generator(generator, config=config)
        binding = _required_mapping(payload, "output_binding")
        bundle_root = _confined_relative_path(
            output_root,
            _required_string(binding, "worker_bundle_relative_root"),
            field_name="worker bundle root",
        )
        batch = _execute_worker_batch(
            worker,
            rows_by_task=rows_by_task,
            work_root=work_root,
            bundle_root=bundle_root,
            generator=generator,
            config=config,
            clock=_monotonic_clock,
        )
        partial_path = _confined_relative_path(
            output_root,
            _required_string(binding, "partial_record_relative_path"),
            field_name="worker partial record",
        )
        _write_jsonl_exclusive(batch.records, partial_path)
        _sync_file(partial_path)
        _sync_directory(partial_path.parent)
        bundle_files = _bundle_file_records(output_root, bundle_root=bundle_root)
        producer_seconds = _positive_duration(_monotonic_clock() - producer_start)
        result: dict[str, Any] = {
            "accounting": {
                "cache_prefix_tokens": batch.cache_prefix_tokens,
                "durable_byte_count": sum(
                    cast(int, item["byte_count"]) for item in bundle_files
                )
                + partial_path.stat().st_size,
                "durable_sync_seconds": batch.durable_sync_seconds,
                "generation_seconds": batch.generation_seconds,
                "includes_generation_payload_hash_and_durable_sync": True,
                "input_token_slots": batch.input_token_slots,
                "producer_metered_seconds": producer_seconds,
            },
            "bundle_files": bundle_files,
            "bundle_files_sha256": _canonical_sha256(bundle_files),
            "closed_record_sha256": "",
            "execution_contract": _execution_config_record(config),
            "execution_mode": PUBLICATION_BF16_HANDOFF_EXECUTION_MODE,
            "generator_hardware": {
                "observed": observed_hardware,
                "qualification": dict(qualification),
            },
            "input_bundle_sha256": prepared.bundle_sha256,
            "partial_record_file": {
                "byte_count": partial_path.stat().st_size,
                "record_count": len(batch.records),
                "relative_path": partial_path.relative_to(output_root).as_posix(),
                "sha256": _file_sha256(partial_path),
            },
            "plan_closed_record_sha256": plan["closed_record_sha256"],
            "record_type": PUBLICATION_BF16_HANDOFF_WORKER_RESULT_RECORD_TYPE,
            "schema_version": PUBLICATION_BF16_HANDOFF_SCHEMA_VERSION,
            "task_ids": sorted(batch.task_ids),
            "task_ids_sha256": _canonical_sha256(sorted(batch.task_ids)),
            "worker_id": payload["worker_id"],
            "worker_index": worker_index_value,
        }
        result["closed_record_sha256"] = _closed_record_sha256(result)
        result_path = _confined_relative_path(
            output_root,
            _required_string(binding, "result_relative_path"),
            field_name="worker result",
        )
        _write_json_exclusive(result, result_path)
        _sync_file(result_path)
        _sync_directory(result_path.parent)
        _sync_directory(output_root)
        return result
    finally:
        if generator is not None:
            close = getattr(generator, "close", None)
            if callable(close):
                close()
        shutil.rmtree(work_root, ignore_errors=True)
        gc.collect()
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            pass
        else:
            torch.cuda.empty_cache()


def _require_worker_output_fresh(
    payload: Mapping[str, Any],
    *,
    output_root: Path,
) -> None:
    binding = _required_mapping(payload, "output_binding")
    paths = (
        _confined_relative_path(
            output_root,
            _required_string(binding, "worker_bundle_relative_root"),
            field_name="worker bundle root",
        ),
        _confined_relative_path(
            output_root,
            _required_string(binding, "partial_record_relative_path"),
            field_name="worker partial record",
        ),
        _confined_relative_path(
            output_root,
            _required_string(binding, "result_relative_path"),
            field_name="worker result",
        ),
    )
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"BF16 worker output is not fresh: {path}")


def _bundle_file_records(
    output_root: Path, *, bundle_root: Path
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(bundle_root.rglob("*")):
        if path.is_symlink():
            raise ValueError("BF16 worker bundle contains a symlink")
        if not path.is_file():
            continue
        records.append(
            {
                "byte_count": path.stat().st_size,
                "relative_path": path.relative_to(output_root).as_posix(),
                "sha256": _file_sha256(path),
            }
        )
    if not records:
        raise ValueError("BF16 worker bundle contains no durable files")
    return records


def _read_worker_result(path: Path) -> dict[str, Any]:
    _reject_symlink_path(path, include_leaf=True)
    content = path.read_bytes()
    record = _json_object(content, field_name="BF16 worker result")
    if content != _canonical_json_bytes(record, pretty=True):
        raise ValueError("BF16 worker result is not canonical JSON")
    if (
        record.get("record_type") != PUBLICATION_BF16_HANDOFF_WORKER_RESULT_RECORD_TYPE
        or record.get("schema_version") != PUBLICATION_BF16_HANDOFF_SCHEMA_VERSION
        or record.get("execution_mode") != PUBLICATION_BF16_HANDOFF_EXECUTION_MODE
        or record.get("closed_record_sha256") != _closed_record_sha256(record)
    ):
        raise ValueError("BF16 worker result envelope is invalid")
    return record


def collect_publication_bf16_handoff_worker_attestation(
    workspace: DatabricksWorkspaceConfig,
    submit_payload: Mapping[str, Any],
    submit_response: Mapping[str, Any],
    *,
    ledger_path: str | Path,
    durable_output_root: str | Path,
    worker_index: int,
    attempt_id: str,
    q8_handoff_remote_closure_authorization: PublicationHandoffRemoteClosureAuthorization,
    submission_authorization: PublicationBF16HandoffSubmissionAuthorization,
) -> PublicationBF16HandoffWorkerAuthorization:
    """Collect one direct ``runs/get`` response and causally close its ledger event."""

    _require_q8_remote_closure_type(q8_handoff_remote_closure_authorization)
    if not isinstance(workspace, DatabricksWorkspaceConfig):
        raise TypeError("workspace has the wrong type")
    batch_reservation_authorization = _require_submission_q8_remote_closure(
        ledger_path,
        q8_handoff_remote_closure_authorization,
        submission_authorization,
    )
    _worker_index(worker_index)
    _validate_submit_payload(submit_payload, worker_index_value=worker_index)
    response_snapshot, _ = canonical_databricks_submit_payload_snapshot(submit_response)
    parent_run_id = _databricks_cloud_id(
        response_snapshot.get("run_id"),
        field_name="submit response run_id",
    )
    terminal_run = get_databricks_run(workspace, parent_run_id)
    record = _build_databricks_attestation(
        submit_payload,
        submit_response,
        terminal_run,
        ledger_path=ledger_path,
        durable_output_root=durable_output_root,
        worker_index=worker_index,
        attempt_id=attempt_id,
        q8_handoff_remote_closure_authorization=q8_handoff_remote_closure_authorization,
        submission_authorization=submission_authorization,
    )
    root = Path(local_path(str(durable_output_root))).expanduser().absolute()
    binding = write_publication_bf16_handoff_attestation(
        record,
        root
        / PUBLICATION_BF16_HANDOFF_ATTESTATION_DIRECTORY
        / f"worker-{worker_index:02d}.json",
    )
    existing_ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    existing_actual = next(
        (
            item
            for item in existing_ledger.terminal_actuals
            if item.attempt_id == attempt_id
        ),
        None,
    )
    updated = (
        existing_ledger
        if existing_actual is not None
        else record_databricks_verified_run_terminal_actual_json(
            ledger_path,
            attempt_id=attempt_id,
            run_record=terminal_run,
        )
    )
    actual = next(
        item for item in updated.terminal_actuals if item.attempt_id == attempt_id
    )
    cloud = _required_mapping(record, "cloud_execution")
    attempt = _required_mapping(record, "attempt")
    if (
        actual.terminal_state != "succeeded"
        or actual.run_id != parent_run_id
        or actual.submit_payload_sha256 != attempt.get("submit_payload_sha256")
        or actual.control_plane_status_sha256
        != cloud.get("control_plane_status_sha256")
        or actual.actual_cluster_duration_seconds
        != cloud.get("actual_gpu_duration_seconds")
        or actual.verification_source != "direct_databricks_runs_get"
    ):
        raise ValueError("BF16 ledger terminal event differs from direct attestation")
    return PublicationBF16HandoffWorkerAuthorization(
        binding=binding,
        attempt_id=attempt_id,
        ledger_id=updated.ledger_id,
        ledger_path_sha256=batch_reservation_authorization.ledger_path_sha256,
        producer_batch_prefix=(
            submission_authorization.batch_authorization.batch_prefix
        ),
        control_plane_status_sha256=_required_string(
            cloud,
            "control_plane_status_sha256",
        ),
        _issuer=_WORKER_AUTHORIZATION_ISSUER,
    )


def _build_databricks_attestation(
    submit_payload: Mapping[str, Any],
    submit_response: Mapping[str, Any],
    terminal_run: Mapping[str, Any],
    *,
    ledger_path: str | Path,
    durable_output_root: str | Path,
    worker_index: int,
    attempt_id: str,
    q8_handoff_remote_closure_authorization: PublicationHandoffRemoteClosureAuthorization,
    submission_authorization: PublicationBF16HandoffSubmissionAuthorization,
) -> dict[str, Any]:
    snapshot, canonical_payload = canonical_databricks_submit_payload_snapshot(
        submit_payload
    )
    response_snapshot, canonical_response = (
        canonical_databricks_submit_payload_snapshot(submit_response)
    )
    terminal_snapshot, canonical_terminal = (
        canonical_databricks_submit_payload_snapshot(terminal_run)
    )
    _validate_submit_payload(snapshot, worker_index_value=worker_index)
    submit_payload_sha256 = sha256(canonical_payload).hexdigest()
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    batch_reservation_authorization = _require_submission_q8_remote_closure(
        ledger_path,
        q8_handoff_remote_closure_authorization,
        submission_authorization,
    )
    if (
        databricks_ledger_path_sha256(ledger_path)
        != batch_reservation_authorization.ledger_path_sha256
    ):
        raise ValueError("BF16 attestation batch/ledger authority drift")
    if attempt_id not in batch_reservation_authorization.attempt_ids:
        raise ValueError("BF16 attestation attempt is outside the atomic batch")
    batch_index = batch_reservation_authorization.attempt_ids.index(attempt_id)
    if (
        batch_reservation_authorization.submit_payload_sha256s[batch_index]
        != submit_payload_sha256
    ):
        raise ValueError("BF16 attestation payload differs from atomic batch")
    require_databricks_ledger_prefix(
        ledger, batch_reservation_authorization.batch_prefix
    )
    reservation = next(
        (item for item in ledger.reservations if item.attempt_id == attempt_id),
        None,
    )
    receipt = next(
        (item for item in ledger.submission_receipts if item.attempt_id == attempt_id),
        None,
    )
    workload_id = f"publication-bf16-handoff-worker-{worker_index:02d}"
    if (
        reservation is None
        or reservation.workload_id != workload_id
        or reservation.submit_payload_sha256 != submit_payload_sha256
        or reservation.reserved_cluster_hours != 5.0
        or receipt is None
        or receipt.submit_payload_sha256 != submit_payload_sha256
        or receipt.submit_response_sha256 != sha256(canonical_response).hexdigest()
    ):
        raise ValueError("BF16 attestation lacks an exact reservation/submit receipt")
    parent_run_id = _databricks_cloud_id(
        response_snapshot.get("run_id"),
        field_name="submit response run_id",
    )
    if receipt.run_id != parent_run_id:
        raise ValueError("BF16 submission receipt belongs to another run")
    if (
        _databricks_cloud_id(
            terminal_snapshot.get("run_id"),
            field_name="terminal run_id",
        )
        != parent_run_id
    ):
        raise ValueError("BF16 terminal response belongs to another run")
    original_attempt_run_id = _databricks_cloud_id(
        terminal_snapshot.get("original_attempt_run_id"),
        field_name="terminal original_attempt_run_id",
    )
    if original_attempt_run_id != parent_run_id:
        raise ValueError("BF16 terminal response is not the original attempt")
    raw_tasks = _mapping_sequence(terminal_snapshot.get("tasks"), "terminal tasks")
    if len(raw_tasks) != 1 or _required_int(raw_tasks[0], "attempt_number") != 0:
        raise ValueError("BF16 producer must finish its only task on attempt zero")
    repairs_raw = terminal_snapshot.get("repair_history")
    repairs = (
        ()
        if repairs_raw is None
        else _mapping_sequence(repairs_raw, "terminal repair_history")
    )
    if repairs:
        raise ValueError("BF16 producer must not use run repair")
    summarized = summarize_databricks_run(
        terminal_snapshot,
        submit_payload=snapshot,
    )
    status = databricks_run_status_record(summarized)
    if status is None:
        raise ValueError("BF16 terminal response has no sanitized status")
    _validate_terminal_status(
        status,
        worker_index=worker_index,
        expected_run_name=_required_string(snapshot, "run_name"),
        submit_payload_sha256=submit_payload_sha256,
    )
    if _databricks_cloud_id(status.get("run_id"), field_name="parent run_id") != (
        parent_run_id
    ):
        raise ValueError("BF16 terminal status belongs to another parent run")
    status_tasks = _mapping_sequence(status.get("tasks"), "status.tasks")
    task = status_tasks[0]
    task_run_id = _databricks_cloud_id(
        task.get("run_id"),
        field_name="terminal task run_id",
    )
    if task_run_id == parent_run_id:
        raise ValueError("BF16 parent and task run IDs must be distinct")
    cluster_id = _databricks_cloud_id(
        task.get("cluster_id"),
        field_name="terminal task cluster_id",
    )
    if (
        status.get("cluster_id") is not None
        and _databricks_cloud_id(
            status.get("cluster_id"),
            field_name="terminal parent cluster_id",
        )
        != cluster_id
    ):
        raise ValueError("BF16 parent and task cluster IDs differ")
    parent_start = _positive_epoch_millis(status, "start_time")
    parent_end = _positive_epoch_millis(status, "end_time")
    task_start = _positive_epoch_millis(task, "start_time")
    task_end = _positive_epoch_millis(task, "end_time")
    if not parent_start <= task_start < task_end <= parent_end:
        raise ValueError("BF16 parent/task timestamps are not causally nested")
    actual_seconds = (task_end - task_start) / 1000.0
    if actual_seconds > PUBLICATION_BF16_HANDOFF_TASK_TIMEOUT_SECONDS:
        raise ValueError("BF16 task duration exceeds five hours")
    root = Path(local_path(str(durable_output_root))).expanduser().absolute()
    result_relative = f"worker-results/worker-{worker_index:02d}.json"
    result_path = _confined_relative_path(
        root,
        result_relative,
        field_name="attested BF16 worker result",
    )
    worker_result = _read_worker_result(result_path)
    if worker_result.get("worker_index") != worker_index:
        raise ValueError("attested BF16 result belongs to another worker")
    record: dict[str, Any] = {
        "attempt": {
            "attempt_id": attempt_id,
            "ledger_id": ledger.ledger_id,
            "ledger_path_sha256": batch_reservation_authorization.ledger_path_sha256,
            "producer_batch_prefix": (
                batch_reservation_authorization.batch_prefix.to_record()
            ),
            "reserved_gpu_hours": reservation.reserved_cluster_hours,
            "submit_payload_sha256": submit_payload_sha256,
            "submit_response_sha256": receipt.submit_response_sha256,
            "worker_index": worker_index,
            "workload_id": workload_id,
        },
        "closed_record_sha256": "",
        "cloud_execution": {
            "actual_gpu_duration_seconds": actual_seconds,
            "attempt_number": 0,
            "cluster_id": cluster_id,
            "control_plane_status_sha256": sha256(canonical_terminal).hexdigest(),
            "life_cycle_state": _required_string(status, "life_cycle_state"),
            "original_attempt_run_id": original_attempt_run_id,
            "parent_end_time_epoch_ms": parent_end,
            "parent_run_id": parent_run_id,
            "parent_start_time_epoch_ms": parent_start,
            "repair_count": 0,
            "result_state": _required_string(status, "result_state"),
            "task_end_time_epoch_ms": task_end,
            "task_key": f"bf16_handoff_worker_{worker_index:02d}",
            "task_run_id": task_run_id,
            "task_start_time_epoch_ms": task_start,
            "terminal_state": "succeeded",
        },
        "record_type": PUBLICATION_BF16_HANDOFF_ATTESTATION_RECORD_TYPE,
        "schema_version": PUBLICATION_BF16_HANDOFF_SCHEMA_VERSION,
        "worker_result": {
            "closed_record_sha256": worker_result["closed_record_sha256"],
            "file_sha256": _file_sha256(result_path),
            "relative_path": result_relative,
        },
    }
    record["closed_record_sha256"] = _closed_record_sha256(record)
    _validate_attestation_record(record)
    return record


def _validate_terminal_status(
    status: Mapping[str, Any],
    *,
    worker_index: int,
    expected_run_name: str,
    submit_payload_sha256: str,
) -> None:
    task_key = f"bf16_handoff_worker_{worker_index:02d}"
    if (
        status.get("terminal") is not True
        or status.get("succeeded") is not True
        or status.get("life_cycle_state") != "TERMINATED"
        or status.get("result_state") != "SUCCESS"
        or status.get("active_task_key") is not None
        or status.get("task_count") != 1
        or status.get("run_name") != expected_run_name
    ):
        raise ValueError("BF16 Databricks status is not one successful task")
    tasks = _mapping_sequence(status.get("tasks"), "terminal status tasks")
    if len(tasks) != 1:
        raise ValueError("BF16 terminal status must contain one task")
    task = tasks[0]
    if (
        task.get("task_key") != task_key
        or task.get("life_cycle_state") != "TERMINATED"
        or task.get("result_state") != "SUCCESS"
        or task.get("node_type_id")
        != PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID
        or task.get("driver_node_type_id")
        != PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID
    ):
        raise ValueError("BF16 terminal task is not the exact L40S producer")
    submit = _required_mapping(status, "submit_payload")
    if (
        submit.get("sha256") != submit_payload_sha256
        or submit.get("run_name") != expected_run_name
        or submit.get("task_count") != 1
        or submit.get("task_keys") != [task_key]
        or submit.get("node_type_ids")
        != [PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID]
        or submit.get("driver_node_type_ids")
        != [PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID]
        or submit.get("single_node") is not True
    ):
        raise ValueError("BF16 terminal submit summary differs from reserved bytes")


def _validate_attestation_record(record: Mapping[str, Any]) -> None:
    if (
        record.get("record_type") != PUBLICATION_BF16_HANDOFF_ATTESTATION_RECORD_TYPE
        or record.get("schema_version") != PUBLICATION_BF16_HANDOFF_SCHEMA_VERSION
        or record.get("closed_record_sha256") != _closed_record_sha256(record)
    ):
        raise ValueError("BF16 cloud attestation envelope is invalid")
    attempt = _required_mapping(record, "attempt")
    worker_index = _required_int(attempt, "worker_index")
    _worker_index(worker_index)
    if (
        attempt.get("workload_id")
        != f"publication-bf16-handoff-worker-{worker_index:02d}"
        or attempt.get("reserved_gpu_hours") != 5.0
    ):
        raise ValueError("BF16 cloud attestation reservation identity is invalid")
    _require_sha256(
        _required_string(attempt, "ledger_path_sha256"), "ledger_path_sha256"
    )
    batch_prefix = databricks_ledger_prefix_from_record(
        _required_mapping(attempt, "producer_batch_prefix")
    )
    if batch_prefix.ledger_id != attempt.get("ledger_id"):
        raise ValueError("BF16 attested batch prefix identity drift")
    for name in ("submit_payload_sha256", "submit_response_sha256"):
        _require_sha256(_required_string(attempt, name), name)
    cloud = _required_mapping(record, "cloud_execution")
    if (
        cloud.get("attempt_number") != 0
        or cloud.get("repair_count") != 0
        or cloud.get("life_cycle_state") != "TERMINATED"
        or cloud.get("result_state") != "SUCCESS"
        or cloud.get("terminal_state") != "succeeded"
        or cloud.get("task_key") != f"bf16_handoff_worker_{worker_index:02d}"
        or cloud.get("original_attempt_run_id") != cloud.get("parent_run_id")
        or cloud.get("task_run_id") == cloud.get("parent_run_id")
    ):
        raise ValueError("BF16 cloud attestation terminal identity is invalid")
    for name in ("parent_run_id", "task_run_id", "cluster_id"):
        _databricks_cloud_id(cloud.get(name), field_name=name)
    _require_sha256(
        _required_string(cloud, "control_plane_status_sha256"),
        "control_plane_status_sha256",
    )
    parent_start = _positive_epoch_millis(cloud, "parent_start_time_epoch_ms")
    parent_end = _positive_epoch_millis(cloud, "parent_end_time_epoch_ms")
    task_start = _positive_epoch_millis(cloud, "task_start_time_epoch_ms")
    task_end = _positive_epoch_millis(cloud, "task_end_time_epoch_ms")
    if not parent_start <= task_start < task_end <= parent_end:
        raise ValueError("BF16 cloud attestation timestamps are invalid")
    duration = _required_positive_number(cloud, "actual_gpu_duration_seconds")
    if (
        not math.isclose(
            duration,
            (task_end - task_start) / 1000.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or duration > PUBLICATION_BF16_HANDOFF_TASK_TIMEOUT_SECONDS
    ):
        raise ValueError("BF16 cloud attestation duration is invalid")
    result = _required_mapping(record, "worker_result")
    if result.get("relative_path") != f"worker-results/worker-{worker_index:02d}.json":
        raise ValueError("BF16 attestation worker-result path is invalid")
    for name in ("closed_record_sha256", "file_sha256"):
        _require_sha256(_required_string(result, name), name)


def write_publication_bf16_handoff_attestation(
    record: Mapping[str, Any],
    path: str | Path,
) -> PublicationBF16HandoffAttestationBinding:
    """Durably publish or idempotently confirm one canonical attestation."""

    _validate_attestation_record(record)
    worker_index = _required_int(_required_mapping(record, "attempt"), "worker_index")
    destination = Path(path).expanduser().absolute()
    content = _canonical_json_bytes(record, pretty=True)
    _reject_symlink_path(destination, include_leaf=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != content:
            raise FileExistsError("BF16 attestation path contains different evidence")
    else:
        with destination.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    _sync_directory(destination.parent)
    _sync_directory(destination.parent.parent)
    return PublicationBF16HandoffAttestationBinding(
        worker_index=worker_index,
        path=destination,
        file_sha256=sha256(content).hexdigest(),
        closed_record_sha256=_required_string(record, "closed_record_sha256"),
    )


def read_publication_bf16_handoff_attestation(
    binding: PublicationBF16HandoffAttestationBinding,
    *,
    durable_output_root: str | Path,
) -> dict[str, Any]:
    if not isinstance(binding, PublicationBF16HandoffAttestationBinding):
        raise TypeError("binding has the wrong type")
    root = Path(local_path(str(durable_output_root))).expanduser().absolute()
    expected_path = (
        root
        / PUBLICATION_BF16_HANDOFF_ATTESTATION_DIRECTORY
        / f"worker-{binding.worker_index:02d}.json"
    )
    if binding.path != expected_path:
        raise ValueError("BF16 attestation path is not canonical")
    _reject_symlink_path(expected_path, include_leaf=True)
    content = expected_path.read_bytes()
    if not hmac.compare_digest(sha256(content).hexdigest(), binding.file_sha256):
        raise ValueError("BF16 attestation file SHA-256 drift")
    record = _json_object(content, field_name="BF16 cloud attestation")
    if content != _canonical_json_bytes(record, pretty=True):
        raise ValueError("BF16 cloud attestation is not canonical JSON")
    _validate_attestation_record(record)
    if record.get("closed_record_sha256") != binding.closed_record_sha256:
        raise ValueError("BF16 attestation closure binding drift")
    result_binding = _required_mapping(record, "worker_result")
    result_path = _confined_relative_path(
        root,
        _required_string(result_binding, "relative_path"),
        field_name="attested BF16 worker result",
    )
    result = _read_worker_result(result_path)
    if (
        result.get("worker_index") != binding.worker_index
        or result.get("closed_record_sha256")
        != result_binding.get("closed_record_sha256")
        or not hmac.compare_digest(
            _file_sha256(result_path),
            _required_string(result_binding, "file_sha256"),
        )
    ):
        raise ValueError("BF16 attestation worker-result binding drift")
    return record


def _ledger_reconciliation(
    ledger_path: str | Path,
    *,
    attempt_ids_by_worker: Mapping[int, str],
    durable_output_root: str | Path,
    worker_authorizations: Mapping[int, PublicationBF16HandoffWorkerAuthorization],
    _ledger: DatabricksClusterHourLedger | None = None,
    _ledger_path_sha256: str | None = None,
) -> dict[str, Any]:
    expected_workers = set(range(PUBLICATION_BF16_HANDOFF_WORKER_COUNT))
    if set(attempt_ids_by_worker) != expected_workers:
        raise ValueError("BF16 attempt mapping must cover workers 0..15 exactly")
    if set(worker_authorizations) != expected_workers:
        raise ValueError("BF16 worker authority must cover workers 0..15 exactly")
    attempt_ids = tuple(
        attempt_ids_by_worker[index] for index in sorted(expected_workers)
    )
    if any(not isinstance(item, str) or not item for item in attempt_ids) or (
        len(set(attempt_ids)) != PUBLICATION_BF16_HANDOFF_WORKER_COUNT
    ):
        raise ValueError("BF16 attempt identifiers must be sixteen unique strings")
    ledger = (
        read_databricks_cluster_hour_ledger_json(ledger_path)
        if _ledger is None
        else _ledger
    )
    ledger_path_binding = (
        databricks_ledger_path_sha256(ledger_path)
        if _ledger_path_sha256 is None
        else _require_sha256(_ledger_path_sha256, "ledger_path_sha256")
    )
    reservations = {item.attempt_id: item for item in ledger.reservations}
    receipts = {item.attempt_id: item for item in ledger.submission_receipts}
    actuals = {item.attempt_id: item for item in ledger.terminal_actuals}
    attempts: list[dict[str, Any]] = []
    parent_run_ids: set[str] = set()
    task_run_ids: set[str] = set()
    cluster_ids: set[str] = set()
    producer_batch_prefixes: set[str] = set()
    for worker_index in range(PUBLICATION_BF16_HANDOFF_WORKER_COUNT):
        attempt_id = attempt_ids_by_worker[worker_index]
        reservation = reservations.get(attempt_id)
        receipt = receipts.get(attempt_id)
        actual = actuals.get(attempt_id)
        if reservation is None or receipt is None or actual is None:
            raise ValueError("BF16 ledger reconciliation is incomplete")
        if (
            reservation.workload_id
            != f"publication-bf16-handoff-worker-{worker_index:02d}"
            or reservation.reserved_cluster_hours != 5.0
            or actual.terminal_state != "succeeded"
            or actual.verification_source != "direct_databricks_runs_get"
            or actual.actual_cluster_duration_seconds <= 0.0
            or actual.actual_cluster_duration_seconds
            > PUBLICATION_BF16_HANDOFF_TASK_TIMEOUT_SECONDS
        ):
            raise ValueError("BF16 ledger attempt identity/accounting is invalid")
        authority = worker_authorizations[worker_index]
        if not isinstance(authority, PublicationBF16HandoffWorkerAuthorization):
            raise TypeError("BF16 reconciliation requires live worker authorizations")
        require_databricks_ledger_prefix(ledger, authority.producer_batch_prefix)
        producer_batch_prefixes.add(authority.producer_batch_prefix.prefix_sha256)
        binding = authority.binding
        if binding.worker_index != worker_index:
            raise ValueError("BF16 attestation mapping contains the wrong worker")
        attestation = read_publication_bf16_handoff_attestation(
            binding,
            durable_output_root=durable_output_root,
        )
        attempt = _required_mapping(attestation, "attempt")
        cloud = _required_mapping(attestation, "cloud_execution")
        if (
            authority.attempt_id != attempt_id
            or authority.ledger_id != ledger.ledger_id
            or authority.ledger_path_sha256 != ledger_path_binding
            or authority.control_plane_status_sha256
            != cloud.get("control_plane_status_sha256")
            or attempt.get("attempt_id") != attempt_id
            or attempt.get("ledger_id") != ledger.ledger_id
            or attempt.get("ledger_path_sha256") != authority.ledger_path_sha256
            or attempt.get("producer_batch_prefix")
            != authority.producer_batch_prefix.to_record()
            or attempt.get("workload_id") != reservation.workload_id
            or attempt.get("submit_payload_sha256") != reservation.submit_payload_sha256
            or attempt.get("submit_response_sha256") != receipt.submit_response_sha256
            or receipt.run_id != cloud.get("parent_run_id")
            or actual.run_id != receipt.run_id
            or actual.submit_payload_sha256 != reservation.submit_payload_sha256
            or actual.control_plane_status_sha256
            != cloud.get("control_plane_status_sha256")
            or actual.actual_cluster_duration_seconds
            != cloud.get("actual_gpu_duration_seconds")
        ):
            raise ValueError(
                "BF16 cloud attestation differs from immutable ledger events"
            )
        parent_run_id = _required_string(cloud, "parent_run_id")
        task_run_id = _required_string(cloud, "task_run_id")
        cluster_id = _required_string(cloud, "cluster_id")
        parent_run_ids.add(parent_run_id)
        task_run_ids.add(task_run_id)
        cluster_ids.add(cluster_id)
        attempts.append(
            {
                "actual_gpu_duration_seconds": actual.actual_cluster_duration_seconds,
                "attempt_id": attempt_id,
                "attestation_closed_record_sha256": (
                    attestation["closed_record_sha256"]
                ),
                "cluster_id": cluster_id,
                "parent_run_id": parent_run_id,
                "submit_payload_sha256": reservation.submit_payload_sha256,
                "task_run_id": task_run_id,
                "worker_index": worker_index,
            }
        )
    if (
        len(parent_run_ids) != PUBLICATION_BF16_HANDOFF_WORKER_COUNT
        or len(task_run_ids) != PUBLICATION_BF16_HANDOFF_WORKER_COUNT
        or len(cluster_ids) != PUBLICATION_BF16_HANDOFF_WORKER_COUNT
        or parent_run_ids.intersection(task_run_ids)
    ):
        raise ValueError("BF16 cloud parent/task/cluster IDs are not globally unique")
    if len(producer_batch_prefixes) != 1:
        raise ValueError("BF16 worker attestations bind different atomic batches")
    return {
        "attempt_count": len(attempts),
        "attempts": attempts,
        "attempts_sha256": _canonical_sha256(attempts),
        "ledger_id": ledger.ledger_id,
        "parent_run_ids_sha256": _canonical_sha256(sorted(parent_run_ids)),
        "task_run_ids_sha256": _canonical_sha256(sorted(task_run_ids)),
        "cluster_ids_sha256": _canonical_sha256(sorted(cluster_ids)),
        "verification_source": "direct_databricks_runs_get",
    }


def publication_bf16_handoff_terminal_actual_gpu_seconds_from_ledger(
    ledger_path: str | Path,
    *,
    attempt_ids_by_worker: Mapping[int, str],
    durable_output_root: str | Path,
    worker_authorizations: Mapping[int, PublicationBF16HandoffWorkerAuthorization],
) -> dict[int, float]:
    """Return all sixteen direct-control-plane GPU durations after full joins."""

    record = _ledger_reconciliation(
        ledger_path,
        attempt_ids_by_worker=attempt_ids_by_worker,
        durable_output_root=durable_output_root,
        worker_authorizations=worker_authorizations,
    )
    return {
        _required_int(item, "worker_index"): _required_positive_number(
            item,
            "actual_gpu_duration_seconds",
        )
        for item in _mapping_sequence(record.get("attempts"), "attempts")
    }


def close_publication_bf16_handoff_generation_from_workers(
    plan: Mapping[str, Any],
    *,
    prepared_input_dir: str | Path,
    durable_output_root: str | Path,
    tokenizer: MainLatencyTokenizer,
    config: PublicationBF16HandoffExecutionConfig,
    ledger_path: str | Path,
    attempt_ids_by_worker: Mapping[int, str],
    worker_authorizations: Mapping[int, PublicationBF16HandoffWorkerAuthorization],
    source_paths: Mapping[str, str | Path] | None = None,
    _ledger_snapshot: DatabricksClusterHourLedger | None = None,
    _ledger_path_sha256: str | None = None,
    _remote_ledger_issuer: object | None = None,
) -> PublicationBF16HandoffGenerationResult:
    """Verify all sixteen producer closures and publish one 16k bundle by rename."""

    if (
        _ledger_snapshot is not None
        or _ledger_path_sha256 is not None
        or _remote_ledger_issuer is not None
    ):
        if _remote_ledger_issuer is not _REMOTE_CLOSURE_LEDGER_ISSUER:
            raise TypeError("remote ledger snapshot requires the coordinator issuer")
        if not isinstance(_ledger_snapshot, DatabricksClusterHourLedger) or (
            _ledger_path_sha256 is None
        ):
            raise ValueError("remote ledger snapshot and path digest are both required")

    validate_publication_bf16_handoff_generation_plan(
        plan,
        prepared_input_dir=prepared_input_dir,
        tokenizer=tokenizer,
        source_paths=source_paths,
    )
    workers = _mapping_sequence(plan.get("workers"), "workers")
    root = Path(local_path(str(durable_output_root))).expanduser().absolute()
    _reject_symlink_path(root, include_leaf=True)
    if not root.is_dir():
        raise ValueError("BF16 durable output root must be an existing real directory")
    for relative in (
        "bundles",
        "manifests",
        PUBLICATION_BF16_HANDOFF_EXECUTION_FILENAME,
    ):
        target = root / relative
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"BF16 coordinator output is not fresh: {target}")
    reconciliation = _ledger_reconciliation(
        ledger_path,
        attempt_ids_by_worker=attempt_ids_by_worker,
        durable_output_root=root,
        worker_authorizations=worker_authorizations,
        _ledger=_ledger_snapshot,
        _ledger_path_sha256=_ledger_path_sha256,
    )
    actuals = {
        _required_int(item, "worker_index"): _required_positive_number(
            item,
            "actual_gpu_duration_seconds",
        )
        for item in _mapping_sequence(reconciliation.get("attempts"), "attempts")
    }
    coordinator_start = _monotonic_clock()
    worker_results: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    all_task_ids: list[str] = []
    qualification_closures: set[str] = set()
    observed_hardware: set[str] = set()
    for worker_index, worker in enumerate(workers):
        result_path = root / "worker-results" / f"worker-{worker_index:02d}.json"
        result = _validate_worker_result(
            result_path,
            output_root=root,
            worker=worker,
            plan=plan,
            config=config,
        )
        accounting = _required_mapping(result, "accounting")
        metered = _required_positive_number(accounting, "producer_metered_seconds")
        if actuals[worker_index] + 1e-12 < metered:
            raise ValueError("BF16 terminal GPU actual is shorter than worker metering")
        hardware = _required_mapping(result, "generator_hardware")
        qualification = _required_mapping(hardware, "qualification")
        qualification_closures.add(
            _required_string(qualification, "evidence_closed_record_sha256")
        )
        observed_hardware.add(
            _canonical_sha256(_required_mapping(hardware, "observed"))
        )
        partial = _required_mapping(result, "partial_record_file")
        partial_path = _confined_relative_path(
            root,
            _required_string(partial, "relative_path"),
            field_name="BF16 partial record",
        )
        all_records.extend(_canonical_jsonl_records(partial_path))
        all_task_ids.extend(cast(list[str], result["task_ids"]))
        worker_results.append(result)
    if len(qualification_closures) != 1:
        raise ValueError("BF16 workers used different GPU qualification evidence")
    expected_task_ids = _plan_task_ids(plan)
    if Counter(all_task_ids) != Counter(expected_task_ids) or (
        len(set(all_task_ids)) != len(all_task_ids)
    ):
        raise ValueError("BF16 worker task coverage is incomplete or duplicated")
    expected_count = (
        len(SUPPORTED_V1_DATASETS) * PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
    )
    identities = [
        (_required_string(row, "dataset"), _required_string(row, "example_id"))
        for row in all_records
    ]
    if len(all_records) != expected_count or len(set(identities)) != expected_count:
        raise ValueError("BF16 enriched row coverage is incomplete or duplicated")

    pending_context = root / "pending" / str(PUBLICATION_BF16_HANDOFF_CONTEXT_TOKENS)
    dataset_paths: dict[str, Path] = {}
    for dataset in SUPPORTED_V1_DATASETS:
        rows = sorted(
            (row for row in all_records if row.get("dataset") == dataset),
            key=lambda row: _required_string(row, "example_id"),
        )
        if len(rows) != PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET:
            raise ValueError(f"BF16 coverage is incomplete for {dataset}")
        path = pending_context / "datasets" / f"{dataset}.jsonl"
        _write_handoff_dataset_jsonl_exclusive(rows, path)
        dataset_paths[dataset] = path
    _sync_tree(pending_context)
    manifest = close_publication_latency_handoff_bundle(
        pending_context,
        dataset_paths,
        context_tokens=PUBLICATION_BF16_HANDOFF_CONTEXT_TOKENS,
        input_bundle_sha256=_required_string(plan, "input_bundle_sha256"),
    )
    _validate_bf16_manifest_contract(manifest)
    portable_digest = _required_string(manifest, "portable_bundle_sha256")
    bundles_root = root / "bundles"
    manifests_root = root / "manifests"
    bundles_root.mkdir()
    manifests_root.mkdir()
    _sync_directory(bundles_root)
    _sync_directory(manifests_root)
    _sync_directory(root)
    source_root = bundles_root / f"16384-{portable_digest}"
    if source_root.exists() or source_root.is_symlink():
        raise ValueError("BF16 content-addressed bundle collision")
    os.rename(pending_context, source_root)
    _sync_directory(source_root.parent)
    validate_publication_latency_handoff_bundle(manifest, bundle_root=source_root)
    manifest_path = manifests_root / PUBLICATION_BF16_HANDOFF_MANIFEST_FILENAME
    write_publication_latency_handoff_bundle(manifest, manifest_path)
    _sync_file(manifest_path)
    pending = root / "pending"
    if any(pending.iterdir()):
        raise ValueError("BF16 pending tree contains unclosed output")
    pending.rmdir()
    _sync_tree(bundles_root)
    _sync_tree(manifests_root)
    coordinator_seconds = _positive_duration(_monotonic_clock() - coordinator_start)
    charged_seconds = sum(actuals.values())
    cache_prefix_tokens = _required_int(
        _required_mapping(plan, "coverage"),
        "cache_prefix_generation_tokens",
    )
    input_token_slots = _required_int(
        _required_mapping(plan, "coverage"),
        "input_token_slots",
    )
    tokens_per_gpu_second = cache_prefix_tokens / charged_seconds
    report_workers = [
        {
            "charged_gpu_seconds": actuals[index],
            "producer_metered_seconds": _required_mapping(result, "accounting")[
                "producer_metered_seconds"
            ],
            "result_closed_record_sha256": result["closed_record_sha256"],
            "result_relative_path": f"worker-results/worker-{index:02d}.json",
            "worker_index": index,
        }
        for index, result in enumerate(worker_results)
    ]
    report: dict[str, Any] = {
        "accounting": {
            "charged_gpu_hours": charged_seconds / 3600.0,
            "charged_gpu_seconds": charged_seconds,
            "coordinator_gpu_hours": 0.0,
            "coordinator_wall_seconds": coordinator_seconds,
            "cost_model": "sum_independent_one_gpu_worker_terminal_lifecycles",
            "durable_byte_count": _tree_byte_count(root),
            "end_to_end_cache_prefix_tokens_per_gpu_second": tokens_per_gpu_second,
            "end_to_end_input_token_slots_per_gpu_second": (
                input_token_slots / charged_seconds
            ),
            "end_to_end_wall_seconds": max(actuals.values()),
            "full_launch_min_tokens_per_gpu_second": (
                PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
            ),
            "full_launch_throughput_gate_passed": (
                tokens_per_gpu_second
                >= PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
            ),
            "includes_bootstrap_generation_hash_and_durable_write": True,
            "payload_copy_count_during_closure": 0,
            "worker_count": PUBLICATION_BF16_HANDOFF_WORKER_COUNT,
        },
        "bundle": {
            "closed_record_sha256": manifest["closed_record_sha256"],
            "context_tokens": PUBLICATION_BF16_HANDOFF_CONTEXT_TOKENS,
            "manifest_file_sha256": _file_sha256(manifest_path),
            "manifest_relative_path": manifest_path.relative_to(root).as_posix(),
            "portable_bundle_sha256": portable_digest,
            "source_root_relative_path": source_root.relative_to(root).as_posix(),
        },
        "closed_record_sha256": "",
        "coverage": {
            "cache_prefix_generation_tokens": cache_prefix_tokens,
            "dataset_count": len(SUPPORTED_V1_DATASETS),
            "examples_per_dataset": PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
            "generated_task_count": len(all_task_ids),
            "input_token_slots": input_token_slots,
            "task_ids_sha256": _canonical_sha256(sorted(all_task_ids)),
        },
        "execution_contract": _execution_config_record(config),
        "execution_mode": PUBLICATION_BF16_HANDOFF_EXECUTION_MODE,
        "generator_hardware": {
            "hardware_target": PUBLICATION_LATENCY_HANDOFF_GENERATOR_HARDWARE_TARGET,
            "node_type_id": PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID,
            "observed_hardware_identity_sha256": sorted(observed_hardware),
            "qualification_closed_record_sha256": next(iter(qualification_closures)),
        },
        "input_bundle_sha256": plan["input_bundle_sha256"],
        "ledger_reconciliation": reconciliation,
        "plan_closed_record_sha256": plan["closed_record_sha256"],
        "record_type": PUBLICATION_BF16_HANDOFF_EXECUTION_RECORD_TYPE,
        "schema_version": PUBLICATION_BF16_HANDOFF_SCHEMA_VERSION,
        "serving_reuse": {
            "regenerate_inside_timed_serving_jobs": False,
            "required_action": "validate_manifest_then_stage_content_addressed_bundle",
            "stage_target": "node_local_nvme",
        },
        "workers": report_workers,
    }
    report["closed_record_sha256"] = _closed_record_sha256(report)
    execution_path = root / PUBLICATION_BF16_HANDOFF_EXECUTION_FILENAME
    _write_json_exclusive(report, execution_path)
    _sync_file(execution_path)
    _sync_directory(root)
    return read_publication_bf16_handoff_generation_result(root)


def _replay_closed_publication_bf16_handoff_generation(
    plan: Mapping[str, Any],
    *,
    prepared_input_dir: str | Path,
    durable_output_root: str | Path,
    tokenizer: MainLatencyTokenizer,
    config: PublicationBF16HandoffExecutionConfig,
    ledger_snapshot: DatabricksClusterHourLedger,
    ledger_path_sha256: str,
    expected_producer_batch_prefix: DatabricksLedgerPrefix,
    attempt_ids_by_worker: Mapping[int, str],
    worker_authorizations: Mapping[int, PublicationBF16HandoffWorkerAuthorization],
    source_paths: Mapping[str, str | Path] | None = None,
    _issuer: object | None = None,
) -> PublicationBF16HandoffGenerationResult:
    """Reopen every raw BF16 producer byte after content-addressed rename."""

    if _issuer is not _POST_CLOSE_REPLAY_ISSUER:
        raise TypeError("post-close BF16 replay requires the coordinator issuer")
    if not isinstance(config, PublicationBF16HandoffExecutionConfig):
        raise TypeError("config must be a PublicationBF16HandoffExecutionConfig")
    if not isinstance(ledger_snapshot, DatabricksClusterHourLedger):
        raise TypeError("ledger_snapshot must be a DatabricksClusterHourLedger")
    if not isinstance(expected_producer_batch_prefix, DatabricksLedgerPrefix):
        raise TypeError("expected_producer_batch_prefix has the wrong type")
    path_digest = _require_sha256(ledger_path_sha256, "ledger_path_sha256")
    expected_workers = set(range(PUBLICATION_BF16_HANDOFF_WORKER_COUNT))
    if set(worker_authorizations) != expected_workers or any(
        authority.producer_batch_prefix != expected_producer_batch_prefix
        for authority in worker_authorizations.values()
    ):
        raise ValueError("post-close BF16 worker authorities bind the wrong batch")
    validate_publication_bf16_handoff_generation_plan(
        plan,
        prepared_input_dir=prepared_input_dir,
        tokenizer=tokenizer,
        source_paths=source_paths,
    )
    root = Path(local_path(str(durable_output_root))).expanduser().absolute()
    result = read_publication_bf16_handoff_generation_result(root)
    execution = result.record
    _require_exact_mapping_keys(
        execution,
        {
            "accounting",
            "bundle",
            "closed_record_sha256",
            "coverage",
            "execution_contract",
            "execution_mode",
            "generator_hardware",
            "input_bundle_sha256",
            "ledger_reconciliation",
            "plan_closed_record_sha256",
            "record_type",
            "schema_version",
            "serving_reuse",
            "workers",
        },
        label="post-close BF16 execution",
    )
    if (
        execution.get("execution_contract") != _execution_config_record(config)
        or execution.get("plan_closed_record_sha256")
        != plan.get("closed_record_sha256")
        or execution.get("input_bundle_sha256") != plan.get("input_bundle_sha256")
    ):
        raise ValueError("post-close BF16 plan/input/config binding drift")
    reconciliation = _ledger_reconciliation(
        Path("/local_disk0/cachet-post-close-ledger-snapshot.json"),
        attempt_ids_by_worker=attempt_ids_by_worker,
        durable_output_root=root,
        worker_authorizations=worker_authorizations,
        _ledger=ledger_snapshot,
        _ledger_path_sha256=path_digest,
    )
    if dict(_required_mapping(execution, "ledger_reconciliation")) != reconciliation:
        raise ValueError("post-close BF16 ledger reconciliation drift")
    attempts = _mapping_sequence(
        reconciliation.get("attempts"), "post-close BF16 reconciliation attempts"
    )
    attempt_by_worker = {
        _required_int(item, "worker_index"): item for item in attempts
    }
    if set(attempt_by_worker) != expected_workers:
        raise ValueError("post-close BF16 ledger worker coverage drift")

    bundle = _required_mapping(execution, "bundle")
    manifest_path = _confined_relative_path(
        root,
        _required_string(bundle, "manifest_relative_path"),
        field_name="post-close BF16 manifest",
    )
    source_root = _confined_relative_path(
        root,
        _required_string(bundle, "source_root_relative_path"),
        field_name="post-close BF16 source root",
    )
    manifest = read_publication_latency_handoff_bundle(manifest_path)
    manifest_file_sequence = _mapping_sequence(manifest.get("files"), "manifest.files")
    manifest_files = {
        _required_string(item, "relative_name"): item
        for item in manifest_file_sequence
    }
    if len(manifest_files) != len(manifest_file_sequence):
        raise ValueError("post-close BF16 manifest contains duplicate file names")
    expected_bundle_paths = {
        source_root / PurePosixPath(relative_name) for relative_name in manifest_files
    }
    manifest_entries: dict[tuple[str, str], Mapping[str, Any]] = {}
    final_rows: dict[str, tuple[dict[str, Any], ...]] = {}
    for dataset_record in _mapping_sequence(manifest.get("datasets"), "datasets"):
        dataset = _required_string(dataset_record, "dataset")
        dataset_path = source_root / PurePosixPath(
            _required_string(dataset_record, "relative_name")
        )
        final_rows[dataset] = _post_close_jsonl_objects(dataset_path)
        for entry in _mapping_sequence(dataset_record.get("entries"), "dataset.entries"):
            identity = (dataset, _required_string(entry, "example_id"))
            if identity in manifest_entries:
                raise ValueError("post-close BF16 manifest identity duplication")
            manifest_entries[identity] = entry

    workers = _mapping_sequence(plan.get("workers"), "workers")
    if len(workers) != PUBLICATION_BF16_HANDOFF_WORKER_COUNT:
        raise ValueError("post-close BF16 plan requires sixteen workers")
    expected_result_paths: set[Path] = set()
    expected_record_paths: set[Path] = set()
    expected_attestation_paths = {
        authority.binding.path.absolute()
        for authority in worker_authorizations.values()
    }
    observed_bundle_records: dict[str, Mapping[str, Any]] = {}
    worker_rows: list[dict[str, Any]] = []
    worker_results: list[dict[str, Any]] = []
    all_task_ids: list[str] = []
    qualification_records: list[Mapping[str, Any]] = []
    qualification_closures: set[str] = set()
    observed_hardware_sha256: set[str] = set()
    for worker_index, worker in enumerate(workers):
        if _required_int(worker, "worker_index") != worker_index:
            raise ValueError("post-close BF16 plan worker order drift")
        result_path = root / "worker-results" / f"worker-{worker_index:02d}.json"
        expected_result_paths.add(result_path)
        worker_result = _read_worker_result(result_path)
        _require_exact_mapping_keys(
            worker_result,
            {
                "accounting",
                "bundle_files",
                "bundle_files_sha256",
                "closed_record_sha256",
                "execution_contract",
                "execution_mode",
                "generator_hardware",
                "input_bundle_sha256",
                "partial_record_file",
                "plan_closed_record_sha256",
                "record_type",
                "schema_version",
                "task_ids",
                "task_ids_sha256",
                "worker_id",
                "worker_index",
            },
            label="post-close BF16 worker result",
        )
        if (
            worker_result.get("worker_index") != worker_index
            or worker_result.get("worker_id") != worker.get("worker_id")
            or worker_result.get("plan_closed_record_sha256")
            != plan.get("closed_record_sha256")
            or worker_result.get("input_bundle_sha256")
            != plan.get("input_bundle_sha256")
            or worker_result.get("execution_contract")
            != _execution_config_record(config)
        ):
            raise ValueError("post-close BF16 worker source/config binding drift")
        planned_items = _mapping_sequence(worker.get("items"), "worker.items")
        expected_task_ids = sorted(
            _required_string(item, "task_id") for item in planned_items
        )
        if worker_result.get("task_ids") != expected_task_ids or worker_result.get(
            "task_ids_sha256"
        ) != _canonical_sha256(expected_task_ids):
            raise ValueError("post-close BF16 worker task closure drift")
        all_task_ids.extend(expected_task_ids)

        bundle_files = _mapping_sequence(worker_result.get("bundle_files"), "bundle_files")
        if worker_result.get("bundle_files_sha256") != _canonical_sha256(bundle_files):
            raise ValueError("post-close BF16 worker bundle closure drift")
        worker_bundle_bytes = 0
        expected_prefix = f"pending/16384/worker-{worker_index:02d}/"
        for file_record in bundle_files:
            _require_exact_mapping_keys(
                file_record,
                {"byte_count", "relative_path", "sha256"},
                label="post-close BF16 worker bundle file",
            )
            preclose_relative = _required_string(file_record, "relative_path")
            if not preclose_relative.startswith(expected_prefix):
                raise ValueError("post-close BF16 worker bundle path drift")
            final_relative = preclose_relative.removeprefix("pending/16384/")
            if final_relative in observed_bundle_records:
                raise ValueError("post-close BF16 worker bundle path duplication")
            manifest_file = manifest_files.get(final_relative)
            expected_file_record = (
                {
                    "byte_count": manifest_file.get("byte_count"),
                    "relative_path": preclose_relative,
                    "sha256": manifest_file.get("sha256"),
                }
                if manifest_file is not None
                and manifest_file.get("role") in {"handoff_json", "payload"}
                else None
            )
            if expected_file_record != dict(file_record):
                raise ValueError(
                    "post-close BF16 worker file differs from finalized manifest"
                )
            observed_bundle_records[final_relative] = file_record
            worker_bundle_bytes += _required_int(file_record, "byte_count")

        partial = _required_mapping(worker_result, "partial_record_file")
        _require_exact_mapping_keys(
            partial,
            {"byte_count", "record_count", "relative_path", "sha256"},
            label="post-close BF16 worker record file",
        )
        expected_relative = f"worker-records/worker-{worker_index:02d}.jsonl"
        if partial.get("relative_path") != expected_relative:
            raise ValueError("post-close BF16 worker-record path drift")
        record_path = _confined_relative_path(
            root,
            expected_relative,
            field_name="post-close BF16 worker record",
        )
        expected_record_paths.add(record_path)
        _verify_file_record(record_path, partial)
        rows = _canonical_jsonl_records(record_path)
        expected_identities = {
            (_required_string(item, "dataset"), _required_string(item, "example_id"))
            for item in planned_items
        }
        observed_identities = {
            (_required_string(row, "dataset"), _required_string(row, "example_id"))
            for row in rows
        }
        if (
            len(rows) != len(planned_items)
            or partial.get("record_count") != len(planned_items)
            or observed_identities != expected_identities
        ):
            raise ValueError("post-close BF16 worker-record identity drift")
        plan_by_identity = {
            (_required_string(item, "dataset"), _required_string(item, "example_id")): item
            for item in planned_items
        }
        for row in rows:
            identity = (
                _required_string(row, "dataset"),
                _required_string(row, "example_id"),
            )
            manifest_entry = manifest_entries.get(identity)
            if manifest_entry is None:
                raise ValueError("post-close BF16 row is absent from the manifest")
            rebased = _post_close_rebased_generated_row(
                row,
                source_root=source_root,
                manifest_entry=manifest_entry,
                arm_id=PUBLICATION_BF16_HANDOFF_ARM_ID,
            )
            planned = plan_by_identity[identity]
            expected_contracts = [
                dict(item)
                for item in _mapping_sequence(
                    planned.get("segment_token_contracts"), "segment_token_contracts"
                )
            ]
            if (
                _handoff_total_tokens(rebased)
                != _required_int(planned, "cache_prefix_tokens")
                or _handoff_segment_token_contracts(rebased) != expected_contracts
            ):
                raise ValueError("post-close BF16 handoff differs from the exact plan")
            worker_rows.append(dict(row))

        worker_accounting = _required_mapping(worker_result, "accounting")
        _require_exact_mapping_keys(
            worker_accounting,
            {
                "cache_prefix_tokens",
                "durable_byte_count",
                "durable_sync_seconds",
                "generation_seconds",
                "includes_generation_payload_hash_and_durable_sync",
                "input_token_slots",
                "producer_metered_seconds",
            },
            label="post-close BF16 worker accounting",
        )
        _required_positive_number(worker_accounting, "generation_seconds")
        _nonnegative_duration(
            _required_number(worker_accounting, "durable_sync_seconds")
        )
        metered = _required_positive_number(
            worker_accounting, "producer_metered_seconds"
        )
        if (
            worker_accounting.get("cache_prefix_tokens")
            != sum(_required_int(item, "cache_prefix_tokens") for item in planned_items)
            or worker_accounting.get("input_token_slots")
            != sum(_required_int(item, "input_token_slots") for item in planned_items)
            or worker_accounting.get("durable_byte_count")
            != worker_bundle_bytes + _required_int(partial, "byte_count")
            or worker_accounting.get(
                "includes_generation_payload_hash_and_durable_sync"
            )
            is not True
        ):
            raise ValueError("post-close BF16 worker accounting drift")
        attempt = attempt_by_worker[worker_index]
        if (
            _required_positive_number(attempt, "actual_gpu_duration_seconds")
            + 1e-12
            < metered
        ):
            raise ValueError("post-close BF16 worker metering exceeds ledger actual")
        authority = worker_authorizations[worker_index]
        attestation = read_publication_bf16_handoff_attestation(
            authority.binding,
            durable_output_root=root,
        )
        attested_result = _required_mapping(attestation, "worker_result")
        if (
            attested_result.get("file_sha256") != _file_sha256(result_path)
            or attested_result.get("closed_record_sha256")
            != worker_result.get("closed_record_sha256")
        ):
            raise ValueError("post-close BF16 worker/attestation evidence drift")
        hardware = _required_mapping(worker_result, "generator_hardware")
        _require_exact_mapping_keys(
            hardware,
            {"observed", "qualification"},
            label="post-close BF16 worker hardware",
        )
        qualification = _required_mapping(hardware, "qualification")
        _validate_qualification_record(qualification)
        observed = _required_mapping(hardware, "observed")
        _validate_observed_l40s_hardware(observed)
        qualification_records.append(qualification)
        qualification_closures.add(
            _required_string(qualification, "evidence_closed_record_sha256")
        )
        observed_hardware_sha256.add(_canonical_sha256(observed))
        worker_results.append(worker_result)

    expected_worker_files = {
        relative_name: item
        for relative_name, item in manifest_files.items()
        if item.get("role") in {"handoff_json", "payload"}
    }
    if set(observed_bundle_records) != set(expected_worker_files):
        raise ValueError("post-close BF16 worker bundle inventory drift")
    for dataset in SUPPORTED_V1_DATASETS:
        expected_rows = tuple(
            sorted(
                (row for row in worker_rows if row.get("dataset") == dataset),
                key=lambda row: _required_string(row, "example_id"),
            )
        )
        if final_rows.get(dataset) != expected_rows:
            raise ValueError(
                "post-close BF16 worker records differ from finalized datasets"
            )

    result_paths = _post_close_regular_file_inventory(
        root / "worker-results", label="post-close BF16 worker-results"
    )
    record_paths = _post_close_regular_file_inventory(
        root / "worker-records", label="post-close BF16 worker-records"
    )
    attestation_paths = _post_close_regular_file_inventory(
        root / PUBLICATION_BF16_HANDOFF_ATTESTATION_DIRECTORY,
        label="post-close BF16 attestations",
    )
    manifest_paths = _post_close_regular_file_inventory(
        root / "manifests", label="post-close BF16 manifests"
    )
    bundle_paths = _post_close_regular_file_inventory(
        root / "bundles", label="post-close BF16 bundles"
    )
    if (
        result_paths != expected_result_paths
        or record_paths != expected_record_paths
        or attestation_paths != expected_attestation_paths
        or manifest_paths != {manifest_path}
        or bundle_paths != expected_bundle_paths
    ):
        raise ValueError("post-close BF16 durable file inventory drift")
    if len(qualification_closures) != 1 or len(
        {_canonical_sha256(item) for item in qualification_records}
    ) != 1:
        raise ValueError("post-close BF16 hardware qualification drift")
    _verify_bound_hardware_qualification_file(qualification_records[0])
    planned_task_ids = _plan_task_ids(plan)
    if Counter(all_task_ids) != Counter(planned_task_ids) or len(
        set(all_task_ids)
    ) != len(all_task_ids):
        raise ValueError("post-close BF16 task coverage drift")

    actuals = {
        index: _required_positive_number(
            attempt_by_worker[index], "actual_gpu_duration_seconds"
        )
        for index in range(PUBLICATION_BF16_HANDOFF_WORKER_COUNT)
    }
    expected_execution_workers = [
        {
            "charged_gpu_seconds": actuals[index],
            "producer_metered_seconds": _required_mapping(
                worker_results[index], "accounting"
            )["producer_metered_seconds"],
            "result_closed_record_sha256": worker_results[index][
                "closed_record_sha256"
            ],
            "result_relative_path": f"worker-results/worker-{index:02d}.json",
            "worker_index": index,
        }
        for index in range(PUBLICATION_BF16_HANDOFF_WORKER_COUNT)
    ]
    if list(_mapping_sequence(execution.get("workers"), "workers")) != (
        expected_execution_workers
    ):
        raise ValueError("post-close BF16 execution worker closure drift")
    cache_prefix_tokens = _required_int(
        _required_mapping(plan, "coverage"), "cache_prefix_generation_tokens"
    )
    input_token_slots = _required_int(
        _required_mapping(plan, "coverage"), "input_token_slots"
    )
    expected_coverage = {
        "cache_prefix_generation_tokens": cache_prefix_tokens,
        "dataset_count": len(SUPPORTED_V1_DATASETS),
        "examples_per_dataset": PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET,
        "generated_task_count": len(all_task_ids),
        "input_token_slots": input_token_slots,
        "task_ids_sha256": _canonical_sha256(sorted(all_task_ids)),
    }
    if dict(_required_mapping(execution, "coverage")) != expected_coverage:
        raise ValueError("post-close BF16 execution task closure drift")
    expected_hardware = {
        "hardware_target": PUBLICATION_LATENCY_HANDOFF_GENERATOR_HARDWARE_TARGET,
        "node_type_id": PUBLICATION_LATENCY_HANDOFF_GENERATOR_NODE_TYPE_ID,
        "observed_hardware_identity_sha256": sorted(observed_hardware_sha256),
        "qualification_closed_record_sha256": next(iter(qualification_closures)),
    }
    if dict(_required_mapping(execution, "generator_hardware")) != expected_hardware:
        raise ValueError("post-close BF16 execution hardware closure drift")
    charged_seconds = sum(actuals.values())
    accounting = _required_mapping(execution, "accounting")
    coordinator_seconds = _required_positive_number(
        accounting, "coordinator_wall_seconds"
    )
    durable_paths = (
        bundle_paths | manifest_paths | record_paths | result_paths | attestation_paths
    )
    expected_accounting = {
        "charged_gpu_hours": charged_seconds / 3600.0,
        "charged_gpu_seconds": charged_seconds,
        "coordinator_gpu_hours": 0.0,
        "coordinator_wall_seconds": coordinator_seconds,
        "cost_model": "sum_independent_one_gpu_worker_terminal_lifecycles",
        "durable_byte_count": sum(path.stat().st_size for path in durable_paths),
        "end_to_end_cache_prefix_tokens_per_gpu_second": (
            cache_prefix_tokens / charged_seconds
        ),
        "end_to_end_input_token_slots_per_gpu_second": (
            input_token_slots / charged_seconds
        ),
        "end_to_end_wall_seconds": max(actuals.values()),
        "full_launch_min_tokens_per_gpu_second": (
            PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
        ),
        "full_launch_throughput_gate_passed": (
            cache_prefix_tokens / charged_seconds
            >= PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
        ),
        "includes_bootstrap_generation_hash_and_durable_write": True,
        "payload_copy_count_during_closure": 0,
        "worker_count": PUBLICATION_BF16_HANDOFF_WORKER_COUNT,
    }
    if dict(accounting) != expected_accounting:
        raise ValueError("post-close BF16 execution accounting closure drift")
    if dict(_required_mapping(execution, "serving_reuse")) != {
        "regenerate_inside_timed_serving_jobs": False,
        "required_action": "validate_manifest_then_stage_content_addressed_bundle",
        "stage_target": "node_local_nvme",
    }:
        raise ValueError("post-close BF16 serving reuse closure drift")
    return result


def _validate_worker_result(
    path: Path,
    *,
    output_root: Path,
    worker: Mapping[str, Any],
    plan: Mapping[str, Any],
    config: PublicationBF16HandoffExecutionConfig,
) -> dict[str, Any]:
    result = _read_worker_result(path)
    worker_index = _required_int(worker, "worker_index")
    if (
        result.get("worker_index") != worker_index
        or result.get("worker_id") != worker.get("worker_id")
        or result.get("plan_closed_record_sha256") != plan.get("closed_record_sha256")
        or result.get("input_bundle_sha256") != plan.get("input_bundle_sha256")
        or dict(_required_mapping(result, "execution_contract"))
        != _execution_config_record(config)
    ):
        raise ValueError("BF16 worker result source/config binding drift")
    items = _mapping_sequence(worker.get("items"), "worker.items")
    expected_task_ids = sorted(_required_string(item, "task_id") for item in items)
    task_ids = result.get("task_ids")
    if task_ids != expected_task_ids or result.get("task_ids_sha256") != (
        _canonical_sha256(expected_task_ids)
    ):
        raise ValueError("BF16 worker result task closure drift")
    accounting = _required_mapping(result, "accounting")
    expected_tokens = sum(_required_int(item, "cache_prefix_tokens") for item in items)
    expected_slots = sum(_required_int(item, "input_token_slots") for item in items)
    if (
        accounting.get("cache_prefix_tokens") != expected_tokens
        or accounting.get("input_token_slots") != expected_slots
        or accounting.get("includes_generation_payload_hash_and_durable_sync")
        is not True
    ):
        raise ValueError("BF16 worker result accounting coverage drift")
    for name in ("generation_seconds", "producer_metered_seconds"):
        _required_positive_number(accounting, name)
    _nonnegative_duration(_required_number(accounting, "durable_sync_seconds"))
    if _required_int(accounting, "durable_byte_count") <= 0:
        raise ValueError("BF16 worker durable byte count must be positive")
    qualification = _required_mapping(
        _required_mapping(result, "generator_hardware"),
        "qualification",
    )
    _validate_qualification_record(qualification)
    _validate_observed_l40s_hardware(
        _required_mapping(_required_mapping(result, "generator_hardware"), "observed")
    )
    partial = _required_mapping(result, "partial_record_file")
    expected_partial = f"worker-records/worker-{worker_index:02d}.jsonl"
    if partial.get("relative_path") != expected_partial or partial.get(
        "record_count"
    ) != len(items):
        raise ValueError("BF16 worker partial-record binding drift")
    partial_path = _confined_relative_path(
        output_root,
        expected_partial,
        field_name="BF16 worker partial record",
    )
    _verify_file_record(partial_path, partial)
    if len(_canonical_jsonl_records(partial_path)) != len(items):
        raise ValueError("BF16 worker partial-record coverage drift")
    bundle_root = output_root / "pending" / "16384" / f"worker-{worker_index:02d}"
    expected_files = _bundle_file_records(output_root, bundle_root=bundle_root)
    observed_files = [
        dict(item)
        for item in _mapping_sequence(result.get("bundle_files"), "bundle_files")
    ]
    if observed_files != expected_files or result.get("bundle_files_sha256") != (
        _canonical_sha256(expected_files)
    ):
        raise ValueError("BF16 worker bundle file inventory drift")
    actual_bytes = sum(cast(int, item["byte_count"]) for item in expected_files) + (
        partial_path.stat().st_size
    )
    if accounting.get("durable_byte_count") != actual_bytes:
        raise ValueError("BF16 worker durable byte accounting drift")
    return result


def _validate_bf16_manifest_contract(manifest: Mapping[str, Any]) -> None:
    if manifest.get("context_tokens") != PUBLICATION_BF16_HANDOFF_CONTEXT_TOKENS:
        raise ValueError("BF16 manifest is not the exact 16k prerequisite")
    identity = _required_mapping(manifest, "identity")
    layout = _required_mapping(identity, "layout_identity")
    if layout.get("dtype") not in {"bf16", "bfloat16"}:
        raise ValueError("precision handoff manifest is not BF16")
    if (
        layout.get("model_id") != QWEN3_4B_INSTRUCT_HF_MODEL_ID
        or layout.get("pre_rope") is not True
        or layout.get("key_position_encoding") != KVKeyPositionEncoding.PRE_ROPE.value
        or layout.get("storage_layout") != KVStorageLayout.SEPARATE_KEY_VALUE.value
        or layout.get("rope_theta") != QWEN3_4B_ROPE_THETA
        or layout.get("rope_rotary_dim") != QWEN3_4B_ROPE_ROTARY_DIM
    ):
        raise ValueError("precision handoff layout is not exact Qwen3 pre-RoPE BF16")
    datasets = _mapping_sequence(manifest.get("datasets"), "datasets")
    if tuple(item.get("dataset") for item in datasets) != tuple(SUPPORTED_V1_DATASETS):
        raise ValueError("BF16 manifest dataset coverage/order is invalid")
    for dataset in datasets:
        entries = _mapping_sequence(dataset.get("entries"), "dataset.entries")
        if (
            dataset.get("row_count") != PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
            or len(entries) != PUBLICATION_CAMPAIGN_EXAMPLES_PER_DATASET
        ):
            raise ValueError("BF16 manifest requires 32 rows per dataset")
        for entry in entries:
            if (
                entry.get("cache_method") != CacheGenerationMethod.VANILLA_PREFILL.value
                or entry.get("transfer_scope")
                != f"arm_kv_transfer_params/{PUBLICATION_BF16_HANDOFF_ARM_ID}"
            ):
                raise ValueError("BF16 manifest contains a non-Vanilla handoff")


def read_publication_bf16_handoff_generation_result(
    durable_output_root: str | Path,
) -> PublicationBF16HandoffGenerationResult:
    """Re-authenticate the execution record, manifest, source tree, and accounting."""

    root = Path(local_path(str(durable_output_root))).expanduser().absolute()
    _reject_symlink_path(root, include_leaf=True)
    execution_path = root / PUBLICATION_BF16_HANDOFF_EXECUTION_FILENAME
    record = _read_canonical_json_file(execution_path, "BF16 generation execution")
    if (
        record.get("record_type") != PUBLICATION_BF16_HANDOFF_EXECUTION_RECORD_TYPE
        or record.get("schema_version") != PUBLICATION_BF16_HANDOFF_SCHEMA_VERSION
        or record.get("execution_mode") != PUBLICATION_BF16_HANDOFF_EXECUTION_MODE
        or record.get("closed_record_sha256") != _closed_record_sha256(record)
    ):
        raise ValueError("BF16 generation execution envelope is invalid")
    bundle = _required_mapping(record, "bundle")
    if bundle.get("context_tokens") != PUBLICATION_BF16_HANDOFF_CONTEXT_TOKENS:
        raise ValueError("BF16 execution bundle context is invalid")
    manifest_path = _confined_relative_path(
        root,
        _required_string(bundle, "manifest_relative_path"),
        field_name="BF16 manifest",
    )
    if manifest_path.name != PUBLICATION_BF16_HANDOFF_MANIFEST_FILENAME:
        raise ValueError("BF16 manifest path is not canonical")
    if not hmac.compare_digest(
        _file_sha256(manifest_path),
        _required_string(bundle, "manifest_file_sha256"),
    ):
        raise ValueError("BF16 manifest file SHA-256 drift")
    manifest = read_publication_latency_handoff_bundle(manifest_path)
    if (
        manifest.get("closed_record_sha256") != bundle.get("closed_record_sha256")
        or manifest.get("portable_bundle_sha256")
        != bundle.get("portable_bundle_sha256")
        or manifest.get("input_bundle_sha256") != record.get("input_bundle_sha256")
    ):
        raise ValueError("BF16 manifest execution binding drift")
    _validate_bf16_manifest_contract(manifest)
    source_root = _confined_relative_path(
        root,
        _required_string(bundle, "source_root_relative_path"),
        field_name="BF16 source root",
    )
    if source_root.name != f"16384-{manifest['portable_bundle_sha256']}":
        raise ValueError("BF16 source root is not content addressed")
    validate_publication_latency_handoff_bundle(manifest, bundle_root=source_root)
    _execution_config_from_record(_required_mapping(record, "execution_contract"))
    accounting = _required_mapping(record, "accounting")
    workers = _mapping_sequence(record.get("workers"), "workers")
    if len(workers) != PUBLICATION_BF16_HANDOFF_WORKER_COUNT:
        raise ValueError("BF16 execution must contain sixteen workers")
    durations = [
        _required_positive_number(worker, "charged_gpu_seconds") for worker in workers
    ]
    if (
        not math.isclose(
            _required_positive_number(accounting, "charged_gpu_seconds"),
            sum(durations),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _required_positive_number(accounting, "end_to_end_wall_seconds"),
            max(durations),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or accounting.get("charged_gpu_hours") != sum(durations) / 3600.0
        or accounting.get("coordinator_gpu_hours") != 0.0
        or accounting.get("payload_copy_count_during_closure") != 0
        or accounting.get("worker_count") != PUBLICATION_BF16_HANDOFF_WORKER_COUNT
        or accounting.get("full_launch_throughput_gate_passed") is not True
    ):
        raise ValueError("BF16 execution accounting is invalid")
    reconciliation = _required_mapping(record, "ledger_reconciliation")
    if (
        reconciliation.get("attempt_count") != PUBLICATION_BF16_HANDOFF_WORKER_COUNT
        or reconciliation.get("verification_source") != "direct_databricks_runs_get"
    ):
        raise ValueError("BF16 execution lacks full cloud/ledger reconciliation")
    return PublicationBF16HandoffGenerationResult(
        root=root,
        source_root=source_root,
        manifest_path=manifest_path,
        execution_record_path=execution_path,
        manifest=manifest,
        record=record,
    )


def authorize_publication_bf16_handoff_serving(
    result: PublicationBF16HandoffGenerationResult,
    *,
    ledger_path: str | Path,
    qualification_launch_authorization: GPUQualificationLaunchAuthorization,
    q8_handoff_remote_closure_authorization: PublicationHandoffRemoteClosureAuthorization,
    submission_authorization: PublicationBF16HandoffSubmissionAuthorization,
    attempt_ids_by_worker: Mapping[int, str],
    worker_authorizations: Mapping[int, PublicationBF16HandoffWorkerAuthorization],
) -> PublicationBF16HandoffServingAuthorization:
    """Issue non-record serving authority after replaying all live worker joins."""

    _require_q8_remote_closure_type(q8_handoff_remote_closure_authorization)
    if not isinstance(result, PublicationBF16HandoffGenerationResult):
        raise TypeError("result has the wrong type")
    verified = read_publication_bf16_handoff_generation_result(result.root)
    if verified.record.get("closed_record_sha256") != result.record.get(
        "closed_record_sha256"
    ) or verified.manifest.get("closed_record_sha256") != result.manifest.get(
        "closed_record_sha256"
    ):
        raise ValueError("BF16 generation result changed before authorization")
    hardware = _required_mapping(verified.record, "generator_hardware")
    batch_reservation_authorization = _require_submission_q8_remote_closure(
        ledger_path,
        q8_handoff_remote_closure_authorization,
        submission_authorization,
    )
    q8_prefix = batch_reservation_authorization.predecessor_prefix
    predecessor_ledger = _require_matching_predecessor_ledger(
        ledger_path,
        qualification_launch_authorization,
        q8_handoff_remote_closure_authorization,
        expected_input_bundle_sha256=_required_string(
            verified.record, "input_bundle_sha256"
        ),
        expected_qualification_closed_record_sha256=_required_string(
            hardware, "qualification_closed_record_sha256"
        ),
        expected_q8_ledger_prefix=q8_prefix,
    )
    if hardware.get("qualification_closed_record_sha256") != (
        qualification_launch_authorization.evidence_closed_record_sha256
    ):
        raise ValueError("BF16 result differs from GPU qualification authority")
    reconciliation = _ledger_reconciliation(
        ledger_path,
        attempt_ids_by_worker=attempt_ids_by_worker,
        durable_output_root=verified.root,
        worker_authorizations=worker_authorizations,
    )
    if dict(_required_mapping(verified.record, "ledger_reconciliation")) != (
        reconciliation
    ):
        raise ValueError("BF16 execution record differs from live causal replay")
    attempts = _mapping_sequence(
        reconciliation.get("attempts"), field_name="BF16 reconciliation attempts"
    )
    expected_attempt_ids = tuple(
        _required_string(item, "attempt_id") for item in attempts
    )
    expected_payload_digests = tuple(
        _required_string(item, "submit_payload_sha256") for item in attempts
    )
    batch_prefix = require_databricks_batch_reservation_authorization(
        batch_reservation_authorization,
        expected_predecessor_prefix=q8_prefix,
        expected_attempt_ids=expected_attempt_ids,
        expected_submit_payload_sha256s=expected_payload_digests,
    )
    require_databricks_ledger_prefix(predecessor_ledger, batch_prefix)
    final_ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    if databricks_ledger_path_sha256(ledger_path) != (
        batch_reservation_authorization.ledger_path_sha256
    ):
        raise ValueError("BF16 ledger path changed during serving authorization")
    require_databricks_ledger_prefix(final_ledger, q8_prefix)
    require_databricks_ledger_prefix(final_ledger, batch_prefix)
    final_reconciliation = _ledger_reconciliation(
        ledger_path,
        attempt_ids_by_worker=attempt_ids_by_worker,
        durable_output_root=verified.root,
        worker_authorizations=worker_authorizations,
        _ledger=final_ledger,
    )
    if final_reconciliation != reconciliation:
        raise ValueError("BF16 ledger changed during causal serving replay")
    ledger_prefix = require_databricks_batch_terminal_closure(
        final_ledger,
        batch_reservation_authorization,
        require_complete_current_prefix=True,
    )
    causal_closure = {
        "ledger_prefix": ledger_prefix.to_record(),
        "predecessor_prefix": q8_prefix.to_record(),
        "producer_batch_prefix": batch_prefix.to_record(),
        "reconciliation": reconciliation,
    }
    return PublicationBF16HandoffServingAuthorization(
        result=verified,
        ledger_id=final_ledger.ledger_id,
        ledger_path_sha256=batch_reservation_authorization.ledger_path_sha256,
        predecessor_prefix=q8_prefix,
        producer_batch_prefix=batch_prefix,
        ledger_prefix=ledger_prefix,
        causal_closure_sha256=_canonical_sha256(causal_closure),
        _issuer=_SERVING_AUTHORIZATION_ISSUER,
    )


def resolve_publication_bf16_handoff_bundle(
    authorization: object,
) -> PublicationBF16HandoffGenerationResult:
    """Resolve only a collector-issued, fully reconciled BF16 capability."""

    if not isinstance(authorization, PublicationBF16HandoffServingAuthorization):
        raise TypeError(
            "BF16 serving requires PublicationBF16HandoffServingAuthorization"
        )
    result = read_publication_bf16_handoff_generation_result(authorization.result_root)
    reconciliation = _required_mapping(result.record, "ledger_reconciliation")
    expected_causal_closure = _canonical_sha256(
        {
            "ledger_prefix": authorization.ledger_prefix.to_record(),
            "predecessor_prefix": authorization.predecessor_prefix.to_record(),
            "producer_batch_prefix": authorization.producer_batch_prefix.to_record(),
            "reconciliation": dict(reconciliation),
        }
    )
    if (
        _file_sha256(result.execution_record_path)
        != authorization.execution_file_sha256
        or result.record.get("closed_record_sha256")
        != authorization.execution_closed_record_sha256
        or _file_sha256(result.manifest_path) != authorization.manifest_file_sha256
        or result.manifest.get("closed_record_sha256")
        != authorization.manifest_closed_record_sha256
        or reconciliation.get("ledger_id") != authorization.ledger_id
        or authorization.ledger_prefix.ledger_id != authorization.ledger_id
        or authorization.producer_batch_prefix.ledger_id != authorization.ledger_id
        or authorization.predecessor_prefix.ledger_id != authorization.ledger_id
        or expected_causal_closure != authorization.causal_closure_sha256
    ):
        raise ValueError("BF16 serving authorization binding drift")
    return result


def require_publication_bf16_handoff_serving_authorization(
    authorization: object,
    *,
    expected_manifest_file_sha256: str,
    expected_manifest_closed_record_sha256: str,
    expected_input_bundle_sha256: str,
) -> PublicationBF16HandoffGenerationResult:
    """Fail closed at a serving-plan boundary against exact final artifact pins."""

    result = resolve_publication_bf16_handoff_bundle(authorization)
    if (
        not hmac.compare_digest(
            _file_sha256(result.manifest_path),
            _require_sha256(
                expected_manifest_file_sha256,
                "expected_manifest_file_sha256",
            ),
        )
        or result.manifest.get("closed_record_sha256")
        != _require_sha256(
            expected_manifest_closed_record_sha256,
            "expected_manifest_closed_record_sha256",
        )
        or result.manifest.get("input_bundle_sha256")
        != _require_sha256(
            expected_input_bundle_sha256,
            "expected_input_bundle_sha256",
        )
    ):
        raise ValueError("BF16 serving authorization differs from final artifact pins")
    return result


def stage_publication_bf16_handoff_bundle(
    authorization: PublicationBF16HandoffServingAuthorization,
    *,
    local_nvme_dir: str | Path,
) -> StagedPublicationLatencyHandoffBundle:
    """Resolve, verify, then atomically stage the BF16 bundle on node-local NVMe."""

    result = resolve_publication_bf16_handoff_bundle(authorization)
    return stage_publication_latency_handoff_bundle(
        result.manifest,
        source_root=result.source_root,
        local_nvme_dir=local_nvme_dir,
    )


def build_publication_bf16_handoff_resource_estimate(
    plan: Mapping[str, Any],
    *,
    config: PublicationBF16HandoffExecutionConfig,
) -> dict[str, Any]:
    """Return conservative storage and GPU-budget bounds without inventing results."""

    _validate_plan_envelope(plan)
    if not isinstance(config, PublicationBF16HandoffExecutionConfig):
        raise TypeError("config has the wrong type")
    workers = _mapping_sequence(plan.get("workers"), "workers")
    cache_tokens = _required_int(
        _required_mapping(plan, "coverage"),
        "cache_prefix_generation_tokens",
    )
    segment_count = sum(
        len(_mapping_sequence(item.get("segment_token_contracts"), "segments"))
        for worker in workers
        for item in _mapping_sequence(worker.get("items"), "worker.items")
    )
    logical_payload_bytes = cache_tokens * config.layout.bytes_per_token
    max_worker_tokens = max(
        _required_int(worker, "cache_prefix_tokens") for worker in workers
    )
    return {
        "cache_prefix_generation_tokens": cache_tokens,
        "logical_payload_bytes": logical_payload_bytes,
        "logical_payload_gib": logical_payload_bytes / (1024**3),
        "max_alignment_overhead_bytes": segment_count * (config.align_bytes - 1),
        "max_parallel_l40s_jobs": PUBLICATION_BF16_HANDOFF_WORKER_COUNT,
        "max_worker_generation_hours_at_35_tokens_per_second": (
            max_worker_tokens
            / PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
            / 3600.0
        ),
        "maximum_generation_gpu_hours_at_35_tokens_per_second": (
            cache_tokens
            / PUBLICATION_CAMPAIGN_MIN_GENERATION_TOKENS_PER_SECOND
            / 3600.0
        ),
        "no_retry": True,
        "reserved_gpu_hour_upper_bound": PUBLICATION_BF16_HANDOFF_MAX_RESERVED_GPU_HOURS,
        "segment_count": segment_count,
        "task_timeout_seconds": PUBLICATION_BF16_HANDOFF_TASK_TIMEOUT_SECONDS,
    }


def write_publication_bf16_handoff_runner_script(path: str | Path) -> Path:
    """Write the reviewed standalone BF16 bootstrap runner exactly once."""

    destination = Path(path).expanduser().absolute()
    _reject_symlink_path(destination, include_leaf=True)
    if destination.exists():
        raise FileExistsError(f"BF16 runner already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(PUBLICATION_BF16_HANDOFF_RUNNER_SCRIPT.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    if _file_sha256(destination) != PUBLICATION_BF16_HANDOFF_RUNNER_SHA256:
        raise RuntimeError("written BF16 runner hash differs from reviewed source")
    _sync_directory(destination.parent)
    return destination


def _cache_prefix_segment_token_contracts(
    row: Mapping[str, Any],
    *,
    tokenizer: MainLatencyTokenizer,
) -> list[dict[str, Any]]:
    example = _example_from_record(
        row,
        default_dataset=_required_string(row, "dataset"),
        record_index=1,
        require_dataset=True,
    )
    contracts: list[dict[str, Any]] = []
    for chunk_id, text in benchmark_cache_prefix_segments(example):
        values = tokenizer.encode(
            text,
            add_special_tokens=MAIN_LATENCY_ADD_SPECIAL_TOKENS,
        )
        if (
            isinstance(values, (str, bytes, bytearray))
            or not isinstance(values, Sequence)
            or not values
            or any(type(token_id) is not int or token_id < 0 for token_id in values)
        ):
            raise ValueError("BF16 input segment contains invalid token IDs")
        token_ids = tuple(values)
        contracts.append(
            {
                "chunk_id": chunk_id,
                "token_count": len(token_ids),
                "token_ids_digest": token_ids_digest(token_ids),
            }
        )
    if not contracts:
        raise ValueError("BF16 cache prefix must contain at least one segment")
    return contracts


def _read_generated_handoff(row: Mapping[str, Any]) -> dict[str, Any]:
    arms = _required_mapping(row, "arm_kv_transfer_params")
    params = _required_mapping(arms, PUBLICATION_BF16_HANDOFF_ARM_ID)
    handoff_path = (
        Path(local_path(_required_string(params, DOCUMENT_KV_HANDOFF_JSON_PARAM)))
        .expanduser()
        .absolute()
    )
    payload_path = (
        Path(local_path(_required_string(params, DOCUMENT_KV_PAYLOAD_URI_PARAM)))
        .expanduser()
        .absolute()
    )
    _reject_symlink_path(handoff_path, include_leaf=True)
    _reject_symlink_path(payload_path, include_leaf=True)
    if not handoff_path.is_file() or not payload_path.is_file():
        raise ValueError("BF16 generated handoff artifacts are missing")
    return _json_object(handoff_path.read_bytes(), field_name="generated BF16 handoff")


def _handoff_total_tokens(row: Mapping[str, Any]) -> int:
    return _required_int(
        _required_mapping(_read_generated_handoff(row), "handle"),
        "total_tokens",
    )


def _handoff_segment_token_contracts(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    handle = _required_mapping(_read_generated_handoff(row), "handle")
    segments = _mapping_sequence(handle.get("segments"), "handle.segments")
    contracts: list[dict[str, Any]] = []
    for segment in segments:
        contract = TokenContract.from_record(
            _required_mapping(segment, "token_contract")
        )
        if contract.token_count != _required_int(segment, "token_count"):
            raise ValueError("BF16 generated segment token count differs from contract")
        contracts.append(
            {
                "chunk_id": _required_string(segment, "chunk_id"),
                "token_count": contract.token_count,
                "token_ids_digest": contract.token_ids_digest,
            }
        )
    if not contracts:
        raise ValueError("BF16 generated handoff contains no segments")
    return contracts


def _canonical_jsonl_records(path: Path) -> tuple[dict[str, Any], ...]:
    _reject_symlink_path(path, include_leaf=True)
    content = path.read_bytes()
    if not content or not content.endswith(b"\n"):
        raise ValueError(f"JSONL must be non-empty and newline-terminated: {path}")
    records: list[dict[str, Any]] = []
    for line in content.splitlines():
        record = _json_object(line, field_name=f"JSONL row in {path.name}")
        if line + b"\n" != _canonical_json_bytes(record, pretty=False) + b"\n":
            raise ValueError(f"JSONL row is not canonical: {path}")
        records.append(record)
    return tuple(records)


def _write_jsonl_exclusive(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    values = tuple(records)
    if not values:
        raise ValueError("refusing to write an empty BF16 JSONL")
    _reject_symlink_path(path, include_leaf=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for record in values:
            handle.write(_canonical_json_bytes(record, pretty=False) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_handoff_dataset_jsonl_exclusive(
    records: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    _write_jsonl_exclusive(records, path)


def _write_json_exclusive(record: Mapping[str, Any], path: Path) -> None:
    _reject_symlink_path(path, include_leaf=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_json_bytes(record, pretty=True))
        handle.flush()
        os.fsync(handle.fileno())


def _read_canonical_json_file(path: str | Path, field_name: str) -> dict[str, Any]:
    source = Path(path).expanduser().absolute()
    _reject_symlink_path(source, include_leaf=True)
    content = source.read_bytes()
    record = _json_object(content, field_name=field_name)
    if content != _canonical_json_bytes(record, pretty=True):
        raise ValueError(f"{field_name} is not canonical JSON")
    return record


def _verify_file_sha256(path: Path, expected: str, *, field_name: str) -> None:
    _reject_symlink_path(path, include_leaf=True)
    if not path.is_file() or not hmac.compare_digest(
        _file_sha256(path),
        _require_sha256(expected, f"{field_name} SHA-256"),
    ):
        raise ValueError(f"{field_name} SHA-256 drift")


def _verify_file_record(path: Path, record: Mapping[str, Any]) -> None:
    _reject_symlink_path(path, include_leaf=True)
    if (
        not path.is_file()
        or path.stat().st_size != _required_int(record, "byte_count")
        or not hmac.compare_digest(
            _file_sha256(path),
            _required_string(record, "sha256"),
        )
    ):
        raise ValueError(f"BF16 durable file binding drift: {path}")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_tree(root: Path) -> None:
    _reject_symlink_path(root, include_leaf=True)
    directories: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"BF16 durable tree contains a symlink: {path}")
        if path.is_file():
            _sync_file(path)
        elif path.is_dir():
            directories.append(path)
    for directory in reversed(directories):
        _sync_directory(directory)
    _sync_directory(root)
    _sync_directory(root.parent)


def _tree_byte_count(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _require_fresh_path(path: Path) -> None:
    _reject_symlink_path(path, include_leaf=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to reuse BF16 output path: {path}")


def _reject_symlink_path(path: Path, *, include_leaf: bool) -> None:
    absolute = path.expanduser().absolute()
    candidates = list(reversed(absolute.parents))
    if include_leaf:
        candidates.append(absolute)
    for candidate in candidates:
        if candidate.is_symlink():
            raise ValueError(f"BF16 path traverses a symlink: {candidate}")


def _confined_relative_path(root: Path, value: str, *, field_name: str) -> Path:
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{field_name} must be a confined relative path")
    path = root.joinpath(*relative.parts).absolute()
    _require_confined_path(root, path, field_name=field_name)
    return path


def _require_confined_path(root: Path, path: Path, *, field_name: str) -> None:
    root_absolute = root.expanduser().absolute()
    candidate = path.expanduser().absolute()
    try:
        candidate.relative_to(root_absolute)
    except ValueError as exc:
        raise ValueError(f"{field_name} escapes the durable root") from exc
    _reject_symlink_path(candidate, include_leaf=True)


def _validate_local_nvme_worker_root(
    value: str,
    *,
    worker_index: int,
    field_name: str,
) -> str:
    _worker_index(worker_index)
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or len(path.parts) != 3
        or path.parts[1] != "local_disk0"
        or any(part in {"", ".", ".."} for part in path.parts)
        or not _SAFE_ID_RE.fullmatch(path.name)
        or f"{worker_index:02d}" not in path.name
    ):
        raise ValueError(
            f"{field_name} must be one worker-unique direct child of /local_disk0"
        )
    return value


def _safe_id(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a safe identifier")
    return value


def _resolved_source_revision(value: str) -> str:
    _nonempty_string(value, "source_revision")
    if value in {"HEAD", "main", "master", "unresolved"} or any(
        character.isspace() for character in value
    ):
        raise ValueError("source_revision must be an immutable resolved revision")
    return value


def _worker_index(value: int) -> int:
    if type(value) is not int or not 0 <= value < PUBLICATION_BF16_HANDOFF_WORKER_COUNT:
        raise ValueError("worker_index must be between 0 and 15")
    return value


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _required_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    observed = value.get(field_name)
    if not isinstance(observed, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return observed


def _require_exact_mapping_keys(
    value: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} must use a closed schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _mapping_sequence(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or any(not isinstance(item, Mapping) for item in value)
    ):
        raise ValueError(f"{field_name} must be an array of objects")
    return tuple(cast(Mapping[str, Any], item) for item in value)


def _required_string(value: Mapping[str, Any], field_name: str) -> str:
    return _nonempty_string(value.get(field_name), field_name)


def _required_int(value: Mapping[str, Any], field_name: str) -> int:
    observed = value.get(field_name)
    if type(observed) is not int:
        raise ValueError(f"{field_name} must be an integer")
    return observed


def _required_bool(value: Mapping[str, Any], field_name: str) -> bool:
    observed = value.get(field_name)
    if type(observed) is not bool:
        raise ValueError(f"{field_name} must be a boolean")
    return observed


def _optional_int(value: Mapping[str, Any], field_name: str) -> int | None:
    observed = value.get(field_name)
    if observed is None:
        return None
    if type(observed) is not int:
        raise ValueError(f"{field_name} must be an integer or null")
    return observed


def _optional_number(value: Mapping[str, Any], field_name: str) -> float | None:
    observed = value.get(field_name)
    if observed is None:
        return None
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        raise ValueError(f"{field_name} must be numeric or null")
    result = float(observed)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _required_number(value: Mapping[str, Any], field_name: str) -> float:
    result = _optional_number(value, field_name)
    if result is None:
        raise ValueError(f"{field_name} must be numeric")
    return result


def _required_positive_number(value: Mapping[str, Any], field_name: str) -> float:
    result = _required_number(value, field_name)
    if result <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return result


def _positive_epoch_millis(value: Mapping[str, Any], field_name: str) -> int:
    result = _required_int(value, field_name)
    if result <= 0:
        raise ValueError(f"{field_name} must be positive")
    return result


def _databricks_cloud_id(value: Any, *, field_name: str) -> str:
    if type(value) is int and value >= 0:
        return str(value)
    if (
        isinstance(value, str)
        and value
        and value == value.strip()
        and len(value) <= 256
        and not any(character.isspace() for character in value)
    ):
        return value
    raise ValueError(f"{field_name} must be a sanitized non-empty Databricks ID")


def _require_sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _positive_duration(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("measured duration must be finite and positive")
    return value


def _nonnegative_duration(value: float) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("measured duration must be finite and nonnegative")
    return value


def _enum_string(value: object) -> str:
    observed = getattr(value, "value", value)
    if not isinstance(observed, str) or not observed:
        raise ValueError("layout enum value must be a non-empty string")
    return observed


def _json_object(content: bytes, *, field_name: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return cast(dict[str, Any], value)


def _canonical_json_bytes(value: Any, *, pretty: bool) -> bytes:
    if pretty:
        return (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return sha256(_canonical_json_bytes(value, pretty=False)).hexdigest()


def _closed_record_sha256(record: Mapping[str, Any]) -> str:
    normalized = dict(record)
    normalized["closed_record_sha256"] = ""
    return _canonical_sha256(normalized)


def _monotonic_clock() -> float:
    return time.perf_counter()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publication BF16 handoff producer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("run-worker")
    worker.add_argument("--worker-payload-json", required=True)
    worker.add_argument("--expected-worker-payload-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "run-worker":
        result = run_publication_bf16_handoff_worker(
            args.worker_payload_json,
            expected_worker_payload_sha256=args.expected_worker_payload_sha256,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    raise ValueError(f"unsupported command: {args.command}")


__all__ = [
    "DatabricksPublicationBF16HandoffJobConfig",
    "PUBLICATION_BF16_HANDOFF_CONTEXT_TOKENS",
    "PUBLICATION_BF16_HANDOFF_EXECUTION_FILENAME",
    "PUBLICATION_BF16_HANDOFF_EXECUTION_MODE",
    "PUBLICATION_BF16_HANDOFF_MANIFEST_FILENAME",
    "PUBLICATION_BF16_HANDOFF_MAX_RESERVED_GPU_HOURS",
    "PUBLICATION_BF16_HANDOFF_RUNNER_SCRIPT",
    "PUBLICATION_BF16_HANDOFF_RUNNER_SHA256",
    "PUBLICATION_BF16_HANDOFF_TASK_TIMEOUT_SECONDS",
    "PUBLICATION_BF16_HANDOFF_WORKER_COUNT",
    "PublicationBF16HandoffAttestationBinding",
    "PublicationBF16HandoffExecutionConfig",
    "PublicationBF16HandoffGenerationResult",
    "PublicationBF16HandoffServingAuthorization",
    "PublicationBF16HandoffSubmissionAuthorization",
    "PublicationBF16HandoffWorkerAuthorization",
    "authorize_publication_bf16_handoff_serving",
    "build_databricks_publication_bf16_handoff_submit_payloads",
    "build_publication_bf16_handoff_execution_config",
    "build_publication_bf16_handoff_generation_plan",
    "build_publication_bf16_handoff_resource_estimate",
    "build_publication_bf16_handoff_worker_payloads",
    "close_publication_bf16_handoff_generation_from_workers",
    "collect_publication_bf16_handoff_worker_attestation",
    "publication_bf16_handoff_worker_attempt_id",
    "publication_bf16_handoff_terminal_actual_gpu_seconds_from_ledger",
    "read_publication_bf16_handoff_attestation",
    "read_publication_bf16_handoff_generation_plan",
    "read_publication_bf16_handoff_generation_result",
    "reserve_and_submit_publication_bf16_handoff_worker_wave",
    "resume_publication_bf16_handoff_worker_wave",
    "require_publication_bf16_handoff_serving_authorization",
    "require_publication_bf16_handoff_submission_authorization",
    "resolve_publication_bf16_handoff_bundle",
    "run_publication_bf16_handoff_worker",
    "stage_publication_bf16_handoff_bundle",
    "validate_publication_bf16_handoff_generation_plan",
    "validate_publication_bf16_handoff_worker_payload",
    "write_publication_bf16_handoff_attestation",
    "write_publication_bf16_handoff_generation_plan",
    "write_publication_bf16_handoff_runner_script",
    "write_publication_bf16_handoff_worker_payloads",
]


if __name__ == "__main__":  # pragma: no cover - exercised by Databricks runner.
    raise SystemExit(main())
