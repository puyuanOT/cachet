from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from document_kv_cache._benchmark_models import (
    BENCHMARK_ARM_ENVIRONMENT_FIELDS,
    _json_object_mapping,
    _validate_non_empty_string,
    _validate_positive_int,
)
from document_kv_cache.benchmarks import (
    BASELINE_PREFILL_ARM,
    CACHE_REUSE_ARM,
    DEFAULT_HARDWARE_TARGET,
    DEFAULT_V1_MODEL_ID,
    DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
    BenchmarkArm,
    BenchmarkOfflineCosts,
    require_runnable_cachet_benchmark_arm,
    validate_v1_dataset,
)
from document_kv_cache.methods import MethodRegistry, default_method_registry
from document_kv_cache.storage import local_path


def main(argv: Sequence[str] | None = None) -> int:
    from document_kv_cache.benchmark_runner import (
        DEFAULT_OPENAI_COMPLETIONS_ENDPOINT,
        PREFIX_CACHE_SALT_MODES,
        OpenAICompatibleBenchmarkConfig,
        benchmark_run_result_to_record,
        run_openai_compatible_v1_benchmark,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run the V1 document KV-cache benchmark against OpenAI-compatible "
            "vLLM/SGLang servers."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="DATASET=PATH",
        help=("Dataset JSONL path. Repeat for biography, hotpotqa, musique, and niah."),
    )
    parser.add_argument("--suite-id", default="v1-openai-compatible")
    parser.add_argument(
        "--base-url",
        required=True,
        help="Baseline server base URL, for example http://localhost:8000",
    )
    parser.add_argument(
        "--cache-base-url",
        help="Optional KV-aware cache server/proxy URL. Defaults to --base-url.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_OPENAI_COMPLETIONS_ENDPOINT,
        help="Completions endpoint appended to --base-url.",
    )
    parser.add_argument(
        "--cache-endpoint",
        help="Optional endpoint appended to --cache-base-url for the cache arm.",
    )
    parser.add_argument("--model-id", default=DEFAULT_V1_MODEL_ID)
    parser.add_argument("--hardware-target", default=DEFAULT_HARDWARE_TARGET)
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--request-parallelism",
        type=int,
        default=1,
        help=(
            "Maximum number of benchmark requests issued concurrently by the client."
        ),
    )
    parser.add_argument(
        "--publication-latency-schedule-json",
        help=(
            "Closed publication latency schedule JSON. Enables exact scheduled "
            "membership and identity-sticky lane execution."
        ),
    )
    parser.add_argument(
        "--publication-latency-expected-input-bundle-sha256",
        help=(
            "Expected verified main-latency input bundle SHA-256. Required with "
            "--publication-latency-schedule-json."
        ),
    )
    parser.add_argument(
        "--arm",
        action="append",
        choices=(BASELINE_PREFILL_ARM, CACHE_REUSE_ARM),
        help=(
            "Benchmark only this arm. Repeat to select multiple arms; omit to run "
            "baseline_prefill and document_kv_cache."
        ),
    )
    parser.add_argument(
        "--arm-spec-json",
        action="append",
        help=(
            "JSON object describing an arbitrary baseline, Cachet, upstream, or "
            "external arm. Repeat for N-way comparisons; mutually exclusive with "
            "--arm."
        ),
    )
    parser.add_argument(
        "--arm-spec-json-file",
        action="append",
        help=(
            "Path to a JSON arm-spec object or array of objects. Repeat for multiple "
            "files; mutually exclusive with --arm."
        ),
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--interleave-examples",
        action="store_true",
        help=(
            "Round-robin requests across examples so a request_parallelism=N wave "
            "draws from N distinct documents instead of repeating one example."
        ),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--generation-seed", type=int)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument(
        "--evidence-policy",
        choices=("smoke", "canary", "publication"),
        default="smoke",
    )
    parser.add_argument("--api-key")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument(
        "--cache-runtime-prompt",
        action="store_true",
        help=(
            "Send only the runtime suffix for the cache arm; requires a KV-aware "
            "proxy that binds cached prefixes."
        ),
    )
    parser.add_argument(
        "--server-usage",
        action="store_true",
        help=(
            "Record server usage.prompt_tokens in metadata when present; reported "
            "prompt_tokens still follow the logical/runtime prompt context."
        ),
    )
    parser.add_argument(
        "--baseline-extra-body-json",
        default="{}",
        help="JSON object merged into baseline requests.",
    )
    parser.add_argument(
        "--cache-extra-body-json",
        default="{}",
        help="JSON object merged into cache-arm requests.",
    )
    parser.add_argument(
        "--prefix-cache-salt-mode",
        choices=PREFIX_CACHE_SALT_MODES,
        default="static",
        help=(
            "How to apply cache_salt from extra-body JSON. 'static' sends it "
            "unchanged; 'per_request' derives a deterministic salt per "
            "dataset/example/arm/repeat."
        ),
    )
    parser.add_argument(
        "--no-isolate-arms",
        dest="isolate_arms",
        action="store_false",
        help=(
            "Interleave all arms through one shared concurrency pool instead of "
            "running each arm in its own phase. Off by default (arms are isolated) "
            "because co-scheduling the cache arm behind baseline full-prefill "
            "requests inflates the measured cache-arm TTFT."
        ),
    )
    parser.set_defaults(isolate_arms=True)
    parser.add_argument(
        "--output-json",
        help="Write the full benchmark result JSON to this path instead of stdout.",
    )
    parser.add_argument(
        "--evidence-gate-output-json",
        help=(
            "Write the selected smoke/canary/publication gate as a standalone JSON "
            "record."
        ),
    )
    parser.add_argument(
        "--artifact-identity-json",
        action="append",
        help=(
            "ArtifactIdentity JSON used by publication gating. Repeat for all "
            "artifacts."
        ),
    )
    parser.add_argument(
        "--cache-state-attestation-json",
        action="append",
        help=(
            "Cache-state attestation or vLLM telemetry JSON. Repeat for cold requests."
        ),
    )
    parser.add_argument("--model-revision", default="unresolved")
    parser.add_argument(
        "--canonical-model-id",
        default="",
        help="Canonical source model identity; defaults to --model-id.",
    )
    parser.add_argument("--tokenizer-id", default="unresolved")
    parser.add_argument("--tokenizer-revision", default="unresolved")
    parser.add_argument("--lora-id", default="base")
    parser.add_argument("--engine-id", default="unresolved")
    parser.add_argument("--engine-version", default="unresolved")
    parser.add_argument("--serving-platform", default="unresolved")
    parser.add_argument("--model-dtype", default="unresolved")
    parser.add_argument("--model-quantization", default="none")
    parser.add_argument("--runtime-kv-dtype", default="unresolved")
    parser.add_argument("--layout-version", default="unresolved")
    parser.add_argument("--payload-axis-order", default="unresolved")
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--key-position-encoding", default="unresolved")
    parser.add_argument("--rope-theta", type=float)
    parser.add_argument("--rope-rotary-dim", type=int)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--pipeline-parallel-size", type=int)
    parser.add_argument(
        "--package-revision",
        action="append",
        metavar="PACKAGE=REVISION",
    )
    parser.add_argument(
        "--prompt-template-version",
        default=DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
    )
    parser.add_argument("--input-tokens-target", type=int)
    parser.add_argument("--hardware-fingerprint", default="unresolved")
    parser.add_argument("--runtime-id", default="unresolved")
    parser.add_argument("--runtime-version", default="unresolved")
    parser.add_argument("--storage-identity", default="unresolved")
    parser.add_argument("--cache-state", default="unresolved")
    parser.add_argument("--complete-dataset-split", action="store_true")
    parser.add_argument(
        "--measurement-scope",
        action="append",
        choices=("latency", "quality", "resource"),
        help=(
            "Declared evidence scope. Repeat as needed; defaults to latency and "
            "quality."
        ),
    )
    parser.add_argument(
        "--comparison-mode",
        choices=("methods_same_setting", "single_method_setting_variation"),
        default="methods_same_setting",
    )
    parser.add_argument("--varied-setting", default="")
    parser.add_argument(
        "--reference-arm-id",
        default="",
        help=(
            "Explicit comparison reference arm. Required for same-method setting "
            "variations whose reference also uses cached physical inputs."
        ),
    )
    args = parser.parse_args(argv)

    try:
        arms, arm_base_urls, arm_endpoints, arm_extra_bodies = _arm_specs_from_cli(
            args.arm_spec_json or (),
            args.arm_spec_json_file or (),
        )
        config = OpenAICompatibleBenchmarkConfig(
            suite_id=args.suite_id,
            dataset_paths=_dataset_paths_from_cli(args.dataset),
            base_url=args.base_url,
            cache_base_url=args.cache_base_url,
            endpoint=args.endpoint,
            cache_endpoint=args.cache_endpoint,
            model_id=args.model_id,
            hardware_target=args.hardware_target,
            limit_per_dataset=args.limit_per_dataset,
            repeats=args.repeats,
            request_parallelism=args.request_parallelism,
            arm_ids=tuple(args.arm or ()),
            arms=arms,
            arm_base_urls=arm_base_urls,
            arm_endpoints=arm_endpoints,
            arm_extra_bodies=arm_extra_bodies,
            shuffle=args.shuffle,
            interleave_examples=args.interleave_examples,
            seed=args.seed,
            generation_seed=args.generation_seed,
            warmups=args.warmups,
            evidence_policy=args.evidence_policy,
            isolate_arms=args.isolate_arms,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
            stream=not args.no_stream,
            cache_runtime_prompt=args.cache_runtime_prompt,
            prompt_token_accounting=(
                "server_usage" if args.server_usage else "logical"
            ),
            baseline_extra_body=_json_object_option(
                args.baseline_extra_body_json,
                "--baseline-extra-body-json",
            ),
            cache_extra_body=_json_object_option(
                args.cache_extra_body_json,
                "--cache-extra-body-json",
            ),
            prefix_cache_salt_mode=args.prefix_cache_salt_mode,
            model_revision=args.model_revision,
            canonical_model_id=args.canonical_model_id,
            tokenizer_id=args.tokenizer_id,
            tokenizer_revision=args.tokenizer_revision,
            lora_id=args.lora_id,
            engine_id=args.engine_id,
            engine_version=args.engine_version,
            serving_platform=args.serving_platform,
            model_dtype=args.model_dtype,
            model_quantization=args.model_quantization,
            runtime_kv_dtype=args.runtime_kv_dtype,
            layout_version=args.layout_version,
            payload_axis_order=args.payload_axis_order,
            block_size=args.block_size,
            key_position_encoding=args.key_position_encoding,
            rope_theta=args.rope_theta,
            rope_rotary_dim=args.rope_rotary_dim,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            package_revisions=_named_revisions(args.package_revision or ()),
            prompt_template_version=args.prompt_template_version,
            input_tokens_target=args.input_tokens_target,
            hardware_fingerprint=args.hardware_fingerprint,
            runtime_id=args.runtime_id,
            runtime_version=args.runtime_version,
            storage_identity=args.storage_identity,
            cache_state=args.cache_state,
            complete_dataset_split=args.complete_dataset_split,
            measurement_scopes=tuple(args.measurement_scope or ("latency", "quality")),
            comparison_mode=args.comparison_mode,
            varied_setting=args.varied_setting,
            reference_arm_id=args.reference_arm_id,
            publication_latency_schedule_path=(
                args.publication_latency_schedule_json
            ),
            publication_latency_expected_input_bundle_sha256=(
                args.publication_latency_expected_input_bundle_sha256
            ),
        )
        result = run_openai_compatible_v1_benchmark(config)
        artifact_identities = _artifact_identities_from_files(
            args.artifact_identity_json or ()
        )
        cache_state_attestations = _cache_state_attestations_from_files(
            args.cache_state_attestation_json or ()
        )
        result_record = benchmark_run_result_to_record(
            result,
            artifact_identities=artifact_identities,
            cache_state_attestations=cache_state_attestations,
        )
        if args.output_json:
            output_path = local_path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            print(json.dumps(result_record, indent=2, sort_keys=True))
        if args.evidence_gate_output_json:
            gate_path = local_path(args.evidence_gate_output_json)
            gate_path.parent.mkdir(parents=True, exist_ok=True)
            gate_path.write_text(
                json.dumps(
                    result_record["evidence_gate"],
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    return 0 if not any(measurement.error for measurement in result.measurements) else 2


def _dataset_paths_from_cli(values: Sequence[str]) -> Mapping[str, Path]:
    dataset_paths: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--dataset must use DATASET=PATH")
        dataset, raw_path = value.split("=", 1)
        validate_v1_dataset(dataset)
        if not raw_path:
            raise ValueError(f"--dataset {dataset}=PATH must include a path")
        if dataset in dataset_paths:
            raise ValueError(f"Duplicate dataset path for {dataset!r}")
        dataset_paths[dataset] = local_path(raw_path)
    return dataset_paths


def _json_object_option(raw_json: str, option_name: str) -> Mapping[str, Any]:
    value = json.loads(raw_json)
    if not isinstance(value, Mapping):
        raise ValueError(f"{option_name} must decode to a JSON object")
    return value


def _arm_specs_from_cli(
    inline_specs: Sequence[str],
    spec_files: Sequence[str],
) -> tuple[
    tuple[BenchmarkArm, ...],
    Mapping[str, str],
    Mapping[str, str],
    Mapping[str, Mapping[str, Any]],
]:
    raw_specs: list[Mapping[str, Any]] = []
    for raw in inline_specs:
        raw_specs.append(_json_object_option(raw, "--arm-spec-json"))
    for raw_path in spec_files:
        path = local_path(raw_path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, Mapping) and "arms" in value:
            value = value["arms"]
        if isinstance(value, Mapping):
            raw_specs.append(value)
        elif isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            for index, item in enumerate(value):
                if not isinstance(item, Mapping):
                    raise ValueError(
                        f"--arm-spec-json-file {path} entry {index} must be an object"
                    )
                raw_specs.append(item)
        else:
            raise ValueError(
                f"--arm-spec-json-file {path} must contain an object or array of objects"
            )
    if not raw_specs:
        return (), {}, {}, {}
    return parse_benchmark_arm_specs(raw_specs)


def parse_benchmark_arm_specs(
    raw_specs: Sequence[Mapping[str, Any]],
    *,
    method_registry: MethodRegistry | None = None,
) -> tuple[
    tuple[BenchmarkArm, ...],
    Mapping[str, str],
    Mapping[str, str],
    Mapping[str, Mapping[str, Any]],
]:
    """Parse closed arbitrary-arm records used by CLI and orchestration tools."""

    registry = default_method_registry() if method_registry is None else method_registry
    if not isinstance(registry, MethodRegistry):
        raise TypeError("method_registry must be a MethodRegistry or None")
    arms: list[BenchmarkArm] = []
    base_urls: dict[str, str] = {}
    endpoints: dict[str, str] = {}
    extra_bodies: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_specs):
        if not isinstance(raw, Mapping):
            raise ValueError(f"arm_specs[{index}] must be an object")
        arm, base_url, endpoint, extra_body = _benchmark_arm_from_spec(
            raw,
            field_name=f"arm_specs[{index}]",
            method_registry=registry,
        )
        arms.append(arm)
        if base_url is not None:
            base_urls[arm.arm_id] = base_url
        if endpoint is not None:
            endpoints[arm.arm_id] = endpoint
        if extra_body is not None:
            extra_bodies[arm.arm_id] = extra_body
    if len({arm.arm_id for arm in arms}) != len(arms):
        raise ValueError("arm specs must not contain duplicate arm ids")
    return tuple(arms), base_urls, endpoints, extra_bodies


def _benchmark_arm_from_spec(
    raw: Mapping[str, Any],
    *,
    field_name: str,
    method_registry: MethodRegistry,
) -> tuple[BenchmarkArm, str | None, str | None, Mapping[str, Any] | None]:
    from document_kv_cache.benchmark_runner import validate_arm_extra_body_contract

    allowed = {
        "arm_id",
        "uses_cache",
        "description",
        "cache_method",
        "connector_mode",
        "variant_id",
        "implementation_kind",
        "method_version",
        "method_config_digest",
        "physical_transform_id",
        "physical_transform_version",
        "physical_transform_config_digest",
        "scorer_plugin_path",
        "offline_costs",
        "base_url",
        "endpoint",
        "extra_body",
        "source_revision",
        "checkpoint_identity",
        "setting_overrides",
        "runtime_environment_overrides",
        "requires_cachet_handoff",
    }
    unknown = set(raw).difference(allowed)
    if unknown:
        raise ValueError(f"{field_name} has unknown fields: {sorted(unknown)}")
    arm_id = raw.get("arm_id")
    description = raw.get("description")
    uses_cache = raw.get("uses_cache")
    _validate_non_empty_string(arm_id, f"{field_name}.arm_id")
    _validate_non_empty_string(description, f"{field_name}.description")
    assert isinstance(arm_id, str)
    assert isinstance(description, str)
    if type(uses_cache) is not bool:
        raise ValueError(f"{field_name}.uses_cache must be a boolean")
    raw_costs = raw.get("offline_costs", {})
    if not isinstance(raw_costs, Mapping):
        raise ValueError(f"{field_name}.offline_costs must be an object")
    cost_fields = {
        "training_seconds",
        "artifact_generation_seconds",
        "checkpoint_load_seconds",
        "artifact_bytes",
        "peak_memory_bytes",
    }
    unknown_costs = set(raw_costs).difference(cost_fields)
    if unknown_costs:
        raise ValueError(
            f"{field_name}.offline_costs has unknown fields: {sorted(unknown_costs)}"
        )
    runtime_environment_overrides = _json_object_mapping(
        raw.get("runtime_environment_overrides", {}),
        f"{field_name}.runtime_environment_overrides",
    )
    unknown_environment_fields = set(runtime_environment_overrides).difference(
        BENCHMARK_ARM_ENVIRONMENT_FIELDS
    )
    if unknown_environment_fields:
        raise ValueError(
            f"{field_name}.runtime_environment_overrides has unknown fields: "
            f"{sorted(unknown_environment_fields)}"
        )
    for environment_field, value in runtime_environment_overrides.items():
        if environment_field in {
            "block_size",
            "tensor_parallel_size",
            "pipeline_parallel_size",
        }:
            if value is not None:
                _validate_positive_int(
                    value,
                    f"{field_name}.runtime_environment_overrides.{environment_field}",
                )
        else:
            _validate_non_empty_string(
                value,
                f"{field_name}.runtime_environment_overrides.{environment_field}",
            )
    arm = BenchmarkArm(
        arm_id=arm_id,
        uses_cache=uses_cache,
        description=description,
        cache_method=_optional_spec_string(raw, "cache_method", field_name),
        connector_mode=_optional_spec_string(raw, "connector_mode", field_name),
        variant_id=_optional_spec_string(raw, "variant_id", field_name),
        implementation_kind=_optional_spec_string(
            raw,
            "implementation_kind",
            field_name,
        ),
        method_version=_optional_spec_string(raw, "method_version", field_name),
        method_config_digest=_optional_spec_string(
            raw,
            "method_config_digest",
            field_name,
        ),
        physical_transform_id=(
            _optional_spec_string(raw, "physical_transform_id", field_name)
            or "identity"
        ),
        physical_transform_version=(
            _optional_spec_string(raw, "physical_transform_version", field_name) or "1"
        ),
        physical_transform_config_digest=_optional_spec_string(
            raw,
            "physical_transform_config_digest",
            field_name,
        ),
        scorer_plugin_path=_optional_spec_string(
            raw,
            "scorer_plugin_path",
            field_name,
        ),
        offline_costs=BenchmarkOfflineCosts(**dict(raw_costs)),
        source_revision=_optional_spec_string(
            raw,
            "source_revision",
            field_name,
        ),
        checkpoint_identity=_optional_spec_string(
            raw,
            "checkpoint_identity",
            field_name,
        ),
        setting_overrides=_json_object_mapping(
            raw.get("setting_overrides", {}),
            f"{field_name}.setting_overrides",
        ),
        runtime_environment_overrides=runtime_environment_overrides,
        requires_cachet_handoff=raw.get("requires_cachet_handoff"),
    )
    require_runnable_cachet_benchmark_arm(
        arm,
        registry=method_registry,
    )
    base_url = _optional_spec_string_or_none(raw, "base_url", field_name)
    endpoint = _optional_spec_string_or_none(raw, "endpoint", field_name)
    raw_extra_body = raw.get("extra_body")
    extra_body = (
        None
        if raw_extra_body is None
        else _json_object_mapping(raw_extra_body, f"{field_name}.extra_body")
    )
    if extra_body is not None:
        validate_arm_extra_body_contract(extra_body, f"{field_name}.extra_body")
    return arm, base_url, endpoint, extra_body


def _optional_spec_string(
    raw: Mapping[str, Any],
    key: str,
    field_name: str,
) -> str:
    value = raw.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"{field_name}.{key} must be a string")
    return value


def _optional_spec_string_or_none(
    raw: Mapping[str, Any],
    key: str,
    field_name: str,
) -> str | None:
    if key not in raw:
        return None
    value = _optional_spec_string(raw, key, field_name)
    if not value:
        raise ValueError(f"{field_name}.{key} must be non-empty when provided")
    return value


def _named_revisions(values: Sequence[str]) -> tuple[tuple[str, str], ...]:
    revisions: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError("--package-revision must use PACKAGE=REVISION")
        name, revision = value.split("=", 1)
        if not name or not revision:
            raise ValueError("--package-revision must use non-empty PACKAGE=REVISION")
        revisions.append((name, revision))
    return tuple(revisions)


def _artifact_identities_from_files(paths: Sequence[str]) -> Mapping[str, Any]:
    from document_kv_cache.artifact_identity import ArtifactIdentity

    identities: dict[str, ArtifactIdentity] = {}
    for raw_path in paths:
        record = json.loads(local_path(raw_path).read_text(encoding="utf-8"))
        if not isinstance(record, Mapping):
            raise ValueError(
                f"artifact identity file {raw_path} must contain an object"
            )
        identity = ArtifactIdentity.from_record(record)
        if identity.artifact_id in identities:
            raise ValueError(f"duplicate artifact identity {identity.artifact_id}")
        identities[identity.artifact_id] = identity
    return identities


def _cache_state_attestations_from_files(paths: Sequence[str]) -> tuple[Any, ...]:
    from document_kv_cache.benchmark_gates import (
        CACHE_STATE_ATTESTATION_RECORD_TYPE,
        CacheStateAttestation,
        cache_state_attestation_from_vllm_telemetry,
    )

    attestations: list[CacheStateAttestation] = []
    for raw_path in paths:
        record = json.loads(local_path(raw_path).read_text(encoding="utf-8"))
        if not isinstance(record, Mapping):
            raise ValueError(
                f"cache-state attestation file {raw_path} must contain an object"
            )
        if record.get("record_type") == CACHE_STATE_ATTESTATION_RECORD_TYPE:
            values = {
                key: value
                for key, value in record.items()
                if key not in {"record_type", "cold_read_attested"}
            }
            attestation = CacheStateAttestation(**values)
        else:
            attestation = cache_state_attestation_from_vllm_telemetry(record)
        attestations.append(attestation)
    return tuple(attestations)


main.__module__ = "document_kv_cache.benchmark_runner"
parse_benchmark_arm_specs.__module__ = "document_kv_cache.benchmark_runner"
