from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from typing import Any

from document_kv_cache._benchmark_models import (
    BENCHMARK_ARM_ENVIRONMENT_FIELDS,
    BENCHMARK_SETTING_DIMENSION_FIELDS,
    BenchmarkArmEnvironment,
    BenchmarkArmManifest,
    BenchmarkExperimentManifest,
    BenchmarkManifestContext,
    BenchmarkScorerManifest,
    _decoding_config_digest,
    _validate_non_empty_string,
)
from document_kv_cache.benchmarks import (
    BenchmarkArm,
    BenchmarkExample,
    BenchmarkSuite,
    DatasetScorerRegistry,
    InferenceMeasurement,
    build_prompt_parts,
)


BENCHMARK_EXPERIMENT_MANIFEST_RECORD_TYPE = (
    "document_kv.benchmark_experiment_manifest.v1"
)
VARIES_BY_ARM = "varies_by_arm"


def _build_experiment_manifest(
    suite: BenchmarkSuite,
    *,
    arms: Sequence[BenchmarkArm],
    measurements: Sequence[InferenceMeasurement],
    scorer_registry: DatasetScorerRegistry,
    context: BenchmarkManifestContext,
    request_parallelism: int,
    repeats: int,
    warmups: int,
    isolate_arms: bool,
    shuffle: bool,
    seed: int | None,
    interleave_examples: bool,
    baseline_arm_id: str,
    request_customization_digests: Mapping[str, str],
) -> BenchmarkExperimentManifest:
    dataset_digests = tuple(
        (
            dataset,
            _sample_selection_digest(
                tuple(
                    example for example in suite.examples if example.dataset == dataset
                ),
                scorer_registry=scorer_registry,
            ),
        )
        for dataset in suite.datasets
    )
    sample_digest = _sha256_json(
        {
            "dataset_sample_digests": dataset_digests,
            "ordered_example_keys": [
                [example.dataset, example.example_id] for example in suite.examples
            ],
        }
    )
    scorer_identities = tuple(
        BenchmarkScorerManifest(
            dataset=dataset,
            scorer_id=scorer.scorer_id,
            version=scorer.version,
            plugin_path=scorer.plugin_path,
            publication_approved=scorer.publication_approved,
            metric_specs=scorer.metric_specs,
            prompt_plugin_path=scorer.prompt_plugin_path,
            prompt_template_version=scorer.prompt_template_version,
        )
        for dataset in suite.datasets
        for scorer in (scorer_registry.get(dataset),)
    )
    arm_manifests: list[BenchmarkArmManifest] = []
    allow_legacy_cache_params = (
        sum(1 for arm in arms if arm.requires_cachet_handoff) == 1
    )
    for arm in arms:
        request_customization_digest = request_customization_digests[arm.arm_id]
        runtime_environment = _build_arm_runtime_environment(
            suite,
            context=context,
            arm=arm,
            isolated_source=len(arms) == 1,
        )
        artifact_ids = tuple(
            sorted(
                {
                    measurement.artifact_id
                    for measurement in measurements
                    if measurement.arm_id == arm.arm_id and measurement.artifact_id
                }
            )
        )
        physical_config_digest = _sha256_json(
            {
                "arm_id": arm.arm_id,
                "implementation_kind": arm.implementation_kind,
                "method_id": arm.cache_method,
                "method_version": arm.method_version,
                "method_config_digest": arm.method_config_digest,
                "connector_mode": arm.connector_mode,
                "variant_id": arm.variant_id,
                "physical_transform_id": arm.physical_transform_id,
                "physical_transform_version": arm.physical_transform_version,
                "source_revision": arm.source_revision,
                "checkpoint_identity": arm.checkpoint_identity,
                "setting_overrides": dict(arm.setting_overrides),
                "runtime_environment_overrides": dict(
                    arm.runtime_environment_overrides
                ),
                "declared_physical_transform_config_digest": (
                    arm.physical_transform_config_digest
                ),
                "request_customization_digest": request_customization_digest,
                "per_example_transfer_params": [
                    {
                        "dataset": example.dataset,
                        "example_id": example.example_id,
                        "kv_transfer_params": (
                            _kv_transfer_params_for_arm(
                                example,
                                arm,
                                allow_legacy=allow_legacy_cache_params,
                            )
                            if arm.requires_cachet_handoff
                            else {}
                        ),
                    }
                    for example in suite.examples
                ],
            }
        )
        costs = arm.offline_costs
        arm_manifests.append(
            BenchmarkArmManifest(
                arm_id=arm.arm_id,
                implementation_kind=arm.implementation_kind,
                uses_cache=arm.uses_cache,
                method_id=arm.cache_method,
                method_version=arm.method_version,
                method_config_digest=arm.method_config_digest,
                artifact_ids=artifact_ids,
                variant_id=arm.variant_id,
                connector_mode=arm.connector_mode,
                physical_transform_id=arm.physical_transform_id,
                physical_transform_version=arm.physical_transform_version,
                declared_physical_transform_config_digest=(
                    arm.physical_transform_config_digest
                ),
                physical_transform_config_digest=physical_config_digest,
                request_customization_digest=request_customization_digest,
                scorer_plugin_path=arm.scorer_plugin_path,
                offline_training_seconds=costs.training_seconds,
                offline_artifact_generation_seconds=costs.artifact_generation_seconds,
                offline_checkpoint_load_seconds=costs.checkpoint_load_seconds,
                artifact_bytes=costs.artifact_bytes,
                offline_peak_memory_bytes=costs.peak_memory_bytes,
                source_revision=arm.source_revision,
                checkpoint_identity=arm.checkpoint_identity,
                setting_overrides=arm.setting_overrides,
                requires_cachet_handoff=bool(arm.requires_cachet_handoff),
                runtime_environment=runtime_environment,
            )
        )
    _validate_arm_runtime_environments(
        tuple(arm_manifests),
        comparison_mode=context.comparison_mode,
        varied_setting=context.varied_setting,
        reference_arm_id=baseline_arm_id,
    )
    decoding_config_digest = _decoding_config_digest(
        max_output_tokens=context.max_output_tokens,
        temperature=context.temperature,
        stream=context.stream,
        generation_seed=context.generation_seed,
        decode_settings=context.decode_settings,
    )
    if isolate_arms:
        order_mode = "arm_isolated"
    elif interleave_examples:
        order_mode = "example_interleaved"
    else:
        order_mode = "grouped"
    return BenchmarkExperimentManifest(
        experiment_id=suite.suite_id,
        baseline_arm_id=baseline_arm_id,
        comparison_mode=context.comparison_mode,
        varied_setting=context.varied_setting,
        sample_selection_digest=sample_digest,
        dataset_sample_digests=dataset_digests,
        datasets=tuple(suite.datasets),
        example_count=len(suite.examples),
        complete_dataset_split=context.complete_dataset_split,
        measurement_scopes=context.measurement_scopes,
        prompt_template_version=context.prompt_template_version,
        scorer_identities=scorer_identities,
        input_tokens_target=context.input_tokens_target,
        output_tokens_target=context.max_output_tokens,
        temperature=context.temperature,
        stream=context.stream,
        generation_seed=context.generation_seed,
        decode_settings=context.decode_settings,
        decoding_config_digest=decoding_config_digest,
        model_id=_legacy_environment_summary(
            arm_manifests,
            "served_model_id",
        ),
        model_revision=_legacy_environment_summary(
            arm_manifests,
            "model_revision",
        ),
        tokenizer_id=_legacy_environment_summary(
            arm_manifests,
            "tokenizer_id",
        ),
        tokenizer_revision=_legacy_environment_summary(
            arm_manifests,
            "tokenizer_revision",
        ),
        engine_id=_legacy_environment_summary(
            arm_manifests,
            "engine_id",
        ),
        engine_version=_legacy_environment_summary(
            arm_manifests,
            "engine_version",
        ),
        package_revisions=context.package_revisions,
        hardware_target=_legacy_environment_summary(
            arm_manifests,
            "hardware_target",
        ),
        hardware_fingerprint=_legacy_environment_summary(
            arm_manifests,
            "hardware_fingerprint",
        ),
        runtime_id=context.runtime_id,
        runtime_version=_legacy_environment_summary(
            arm_manifests,
            "runtime_version",
        ),
        storage_identity=_legacy_environment_summary(
            arm_manifests,
            "storage_identity",
        ),
        cache_state=_legacy_environment_summary(
            arm_manifests,
            "cache_state",
        ),
        request_parallelism=request_parallelism,
        repeats=repeats,
        warmups=warmups,
        isolate_arms=isolate_arms,
        order_mode=order_mode,
        shuffle=shuffle,
        benchmark_seed=seed,
        arms=tuple(arm_manifests),
        execution_isolation_mode=(
            "shared_process_sequential" if isolate_arms else "shared_process_concurrent"
        ),
    )


def _build_arm_runtime_environment(
    suite: BenchmarkSuite,
    *,
    context: BenchmarkManifestContext,
    arm: BenchmarkArm,
    isolated_source: bool,
) -> BenchmarkArmEnvironment:
    base = BenchmarkArmEnvironment(
        served_model_id=suite.model_id,
        canonical_model_id=context.canonical_model_id or suite.model_id,
        model_revision=context.model_revision,
        tokenizer_id=context.tokenizer_id,
        tokenizer_revision=context.tokenizer_revision,
        lora_id=context.lora_id,
        prompt_template_version=context.prompt_template_version,
        engine_id=context.engine_id,
        engine_version=context.engine_version,
        serving_platform=context.serving_platform,
        hardware_target=suite.hardware_target,
        hardware_fingerprint=context.hardware_fingerprint,
        model_dtype=context.model_dtype,
        model_quantization=context.model_quantization,
        runtime_kv_dtype=context.runtime_kv_dtype,
        layout_version=context.layout_version,
        payload_axis_order=context.payload_axis_order,
        block_size=context.block_size,
        key_position_encoding=context.key_position_encoding,
        rope_theta=context.rope_theta,
        rope_rotary_dim=context.rope_rotary_dim,
        tensor_parallel_size=context.tensor_parallel_size,
        pipeline_parallel_size=context.pipeline_parallel_size,
        runtime_version=context.runtime_version,
        storage_identity=context.storage_identity,
        cache_state=context.cache_state,
    )
    overrides = dict(arm.runtime_environment_overrides)
    unknown = set(overrides).difference(BENCHMARK_ARM_ENVIRONMENT_FIELDS)
    if unknown:
        raise ValueError(
            f"arm {arm.arm_id!r} runtime_environment_overrides has unknown fields: "
            f"{sorted(unknown)}"
        )
    if not overrides:
        return base
    candidate = replace(base, **overrides)
    if isolated_source and candidate != base:
        mismatches = sorted(
            field_name
            for field_name in overrides
            if getattr(candidate, field_name) != getattr(base, field_name)
        )
        raise ValueError(
            f"isolated arm {arm.arm_id!r} runtime_environment_overrides must "
            "equal its source manifest context; mismatched fields: "
            f"{mismatches}"
        )
    return candidate


def _runtime_environment_to_record(
    environment: BenchmarkArmEnvironment,
) -> dict[str, Any]:
    if not isinstance(environment, BenchmarkArmEnvironment):
        raise TypeError("environment must be BenchmarkArmEnvironment")
    return {
        field_name: getattr(environment, field_name)
        for field_name in BENCHMARK_ARM_ENVIRONMENT_FIELDS
    }


def _legacy_environment_summary(
    arms: Sequence[BenchmarkArmManifest],
    field_name: str,
) -> Any:
    values = {
        getattr(arm.runtime_environment, field_name)
        for arm in arms
    }
    if len(values) == 1:
        return next(iter(values))
    return VARIES_BY_ARM


def _runtime_environment_differences(
    left: BenchmarkArmEnvironment,
    right: BenchmarkArmEnvironment,
) -> set[str]:
    return {
        field_name
        for field_name in BENCHMARK_ARM_ENVIRONMENT_FIELDS
        if getattr(left, field_name) != getattr(right, field_name)
    }


def _validate_arm_runtime_environments(
    arms: Sequence[BenchmarkArmManifest],
    *,
    comparison_mode: str,
    varied_setting: str,
    reference_arm_id: str,
) -> None:
    if not arms:
        raise ValueError("benchmark arms must be non-empty")
    by_id = {arm.arm_id: arm for arm in arms}
    if reference_arm_id not in by_id:
        raise ValueError(
            f"reference_arm_id references unknown arm {reference_arm_id!r}"
        )
    reference = by_id[reference_arm_id]
    if comparison_mode == "methods_same_setting":
        method_owned_position_fields = {
            "key_position_encoding",
            "rope_theta",
            "rope_rotary_dim",
        }
        for arm in arms:
            differences = _runtime_environment_differences(
                reference.runtime_environment,
                arm.runtime_environment,
            )
            if arm.method_id != reference.method_id:
                differences.difference_update(method_owned_position_fields)
            if differences:
                raise ValueError(
                    "methods_same_setting requires identical actual runtime "
                    f"environments; arm {arm.arm_id!r} changes {sorted(differences)}"
                )
        return
    if comparison_mode != "single_method_setting_variation":
        raise ValueError("unsupported comparison_mode")
    if varied_setting not in BENCHMARK_ARM_ENVIRONMENT_FIELDS:
        raise ValueError(
            "varied_setting must name one typed runtime environment field; "
            f"got {varied_setting!r}"
        )
    actual_values: list[Any] = []
    for arm in arms:
        if set(arm.setting_overrides) != {varied_setting}:
            raise ValueError(
                f"arm {arm.arm_id!r} setting_overrides must contain exactly "
                f"{varied_setting!r}"
            )
        actual_value = getattr(arm.runtime_environment, varied_setting)
        if arm.setting_overrides[varied_setting] != actual_value:
            raise ValueError(
                f"arm {arm.arm_id!r} setting_overrides.{varied_setting} must "
                "equal its actual runtime environment"
            )
        actual_values.append(actual_value)
        if arm.arm_id == reference_arm_id:
            continue
        if (
            varied_setting != "serving_platform"
            and arm.request_customization_digest
            != reference.request_customization_digest
        ):
            raise ValueError(
                "single_method_setting_variation requires invariant static "
                "request customizations unless serving_platform is varied; "
                f"arm {arm.arm_id!r} changes request_customization_digest"
            )
        differences = _runtime_environment_differences(
            reference.runtime_environment,
            arm.runtime_environment,
        )
        allowed_differences = BENCHMARK_SETTING_DIMENSION_FIELDS.get(
            varied_setting,
            frozenset({varied_setting}),
        )
        unexpected_differences = differences.difference(allowed_differences)
        if varied_setting not in differences or unexpected_differences:
            raise ValueError(
                "single_method_setting_variation requires the declared setting "
                "dimension and only its typed dependent fields to differ; arm "
                f"{arm.arm_id!r} changes {sorted(differences)}"
            )
    if len(arms) > 1 and len({_sha256_json(value) for value in actual_values}) != len(
        arms
    ):
        raise ValueError(
            "setting-variation arms must have distinct actual varied setting values"
        )


def _sample_selection_digest(
    examples: Sequence[BenchmarkExample],
    *,
    scorer_registry: DatasetScorerRegistry,
) -> str:
    return _sha256_json(
        [
            {
                "dataset": example.dataset,
                "example_id": example.example_id,
                "logical_prompt": build_prompt_parts(
                    example,
                    scorer=scorer_registry.get(example.dataset),
                ).prefill_prompt,
                "query": example.query,
                "expected_answer": example.expected_answer,
                "references": list(example.references),
                "metadata": dict(example.metadata),
            }
            for example in examples
        ]
    )


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        _json_materialize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _json_materialize(value: Any) -> Any:
    """Recursively copy immutable evaluation values into plain JSON containers."""

    if isinstance(value, Mapping):
        return {key: _json_materialize(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray, memoryview),
    ):
        return [_json_materialize(item) for item in value]
    return value


def benchmark_experiment_manifest_to_record(
    manifest: BenchmarkExperimentManifest,
) -> dict[str, Any]:
    if not isinstance(manifest, BenchmarkExperimentManifest):
        raise TypeError("manifest must be BenchmarkExperimentManifest")
    record = {
        "record_type": BENCHMARK_EXPERIMENT_MANIFEST_RECORD_TYPE,
        "manifest_version": 1,
        "experiment_id": manifest.experiment_id,
        "comparison": {
            "mode": manifest.comparison_mode,
            "varied_setting": manifest.varied_setting or None,
            "baseline_arm_id": manifest.baseline_arm_id,
            "reference_arm_id": manifest.baseline_arm_id,
        },
        "logical_workload": {
            "sample_selection_digest": manifest.sample_selection_digest,
            "dataset_sample_digests": {
                dataset: digest for dataset, digest in manifest.dataset_sample_digests
            },
            "datasets": list(manifest.datasets),
            "example_count": manifest.example_count,
            "complete_dataset_split": manifest.complete_dataset_split,
            "measurement_scopes": list(manifest.measurement_scopes),
            "prompt_template_version": manifest.prompt_template_version,
            "input_tokens_target": manifest.input_tokens_target,
            "output_tokens_target": manifest.output_tokens_target,
            "scorers": [
                {
                    "dataset": scorer.dataset,
                    "scorer_id": scorer.scorer_id,
                    "version": scorer.version,
                    "plugin_path": scorer.plugin_path,
                    "publication_approved": scorer.publication_approved,
                    "metrics": [
                        {
                            "metric_name": metric.metric_name,
                            "direction": metric.direction,
                            "max_regression": metric.max_regression,
                        }
                        for metric in scorer.metric_specs
                    ],
                    "prompt_plugin_path": scorer.prompt_plugin_path or None,
                    "prompt_template_version": (scorer.prompt_template_version or None),
                }
                for scorer in manifest.scorer_identities
            ],
        },
        "decoding": {
            "max_output_tokens": manifest.output_tokens_target,
            "temperature": manifest.temperature,
            "stream": manifest.stream,
            "generation_seed": manifest.generation_seed,
            "settings": _json_materialize(manifest.decode_settings),
            "config_digest": manifest.decoding_config_digest,
        },
        "model_runtime": {
            "model_id": manifest.model_id,
            "model_revision": manifest.model_revision,
            "tokenizer_id": manifest.tokenizer_id,
            "tokenizer_revision": manifest.tokenizer_revision,
            "engine_id": manifest.engine_id,
            "engine_version": manifest.engine_version,
            "canonical_model_id": _legacy_environment_summary(
                manifest.arms,
                "canonical_model_id",
            ),
            "lora_id": _legacy_environment_summary(manifest.arms, "lora_id"),
            "prompt_template_version": _legacy_environment_summary(
                manifest.arms,
                "prompt_template_version",
            ),
            "serving_platform": _legacy_environment_summary(
                manifest.arms,
                "serving_platform",
            ),
            "model_dtype": _legacy_environment_summary(
                manifest.arms,
                "model_dtype",
            ),
            "model_quantization": _legacy_environment_summary(
                manifest.arms,
                "model_quantization",
            ),
            "runtime_kv_dtype": _legacy_environment_summary(
                manifest.arms,
                "runtime_kv_dtype",
            ),
            "layout_version": _legacy_environment_summary(
                manifest.arms,
                "layout_version",
            ),
            "payload_axis_order": _legacy_environment_summary(
                manifest.arms,
                "payload_axis_order",
            ),
            "block_size": _legacy_environment_summary(
                manifest.arms,
                "block_size",
            ),
            "key_position_encoding": _legacy_environment_summary(
                manifest.arms,
                "key_position_encoding",
            ),
            "rope_theta": _legacy_environment_summary(
                manifest.arms,
                "rope_theta",
            ),
            "rope_rotary_dim": _legacy_environment_summary(
                manifest.arms,
                "rope_rotary_dim",
            ),
            "tensor_parallel_size": _legacy_environment_summary(
                manifest.arms,
                "tensor_parallel_size",
            ),
            "pipeline_parallel_size": _legacy_environment_summary(
                manifest.arms,
                "pipeline_parallel_size",
            ),
            "package_revisions": dict(manifest.package_revisions),
        },
        "environment": {
            "hardware_target": manifest.hardware_target,
            "hardware_fingerprint": manifest.hardware_fingerprint,
            "runtime_id": manifest.runtime_id,
            "runtime_version": manifest.runtime_version,
            "storage_identity": manifest.storage_identity,
            "cache_state": manifest.cache_state,
        },
        "execution": {
            "request_parallelism": manifest.request_parallelism,
            "repeats": manifest.repeats,
            "warmups": manifest.warmups,
            "isolate_arms": manifest.isolate_arms,
            "order_mode": manifest.order_mode,
            "shuffle": manifest.shuffle,
            "benchmark_seed": manifest.benchmark_seed,
            "isolation_mode": manifest.execution_isolation_mode,
            "source_execution_ids": [
                {"arm_id": arm_id, "execution_id_digest": execution_id_digest}
                for arm_id, execution_id_digest in manifest.source_execution_ids
            ],
        },
        "arms": [
            {
                "arm_id": arm.arm_id,
                "implementation_kind": arm.implementation_kind,
                "uses_cache": arm.uses_cache,
                "requires_cachet_handoff": arm.requires_cachet_handoff,
                "method_id": arm.method_id,
                "method_version": arm.method_version,
                "method_config_digest": arm.method_config_digest or None,
                "artifact_ids": list(arm.artifact_ids),
                "variant_id": arm.variant_id,
                "connector_mode": arm.connector_mode,
                "physical_transform": {
                    "transform_id": arm.physical_transform_id,
                    "version": arm.physical_transform_version,
                    "declared_config_digest": (
                        arm.declared_physical_transform_config_digest or None
                    ),
                    "config_digest": arm.physical_transform_config_digest,
                },
                "request_customization": {
                    "config_digest": arm.request_customization_digest,
                },
                "scorer_plugin_path": arm.scorer_plugin_path or None,
                "source_revision": arm.source_revision or None,
                "checkpoint_identity": arm.checkpoint_identity or None,
                "setting_overrides": dict(arm.setting_overrides),
                "runtime_environment": _runtime_environment_to_record(
                    arm.runtime_environment
                ),
                "offline_costs": {
                    "training_seconds": arm.offline_training_seconds,
                    "artifact_generation_seconds": (
                        arm.offline_artifact_generation_seconds
                    ),
                    "checkpoint_load_seconds": arm.offline_checkpoint_load_seconds,
                    "artifact_bytes": arm.artifact_bytes,
                    "peak_memory_bytes": arm.offline_peak_memory_bytes,
                },
            }
            for arm in manifest.arms
        ],
        "has_unresolved_provenance": manifest.has_unresolved_provenance,
    }
    materialized = _json_materialize(record)
    if not isinstance(materialized, dict):  # pragma: no cover - defensive invariant.
        raise TypeError("benchmark manifest must serialize to an object")
    return materialized


def _resolve_reference_arm_id(
    arms: Sequence[BenchmarkArm],
    requested: str,
    *,
    comparison_mode: str,
) -> str:
    arm_ids = {arm.arm_id for arm in arms}
    if requested:
        if requested not in arm_ids:
            raise ValueError(f"reference_arm_id references unknown arm {requested!r}")
        return requested
    for arm in arms:
        if not arm.uses_cache:
            return arm.arm_id
    if len(arms) == 1:
        return arms[0].arm_id
    if comparison_mode == "single_method_setting_variation":
        raise ValueError(
            "single_method_setting_variation requires an explicit reference_arm_id"
        )
    raise ValueError(
        "multi-arm comparisons without a non-cache arm require an explicit "
        "reference_arm_id"
    )


def _validate_comparison_design(
    arms: Sequence[BenchmarkArm],
    *,
    comparison_mode: str,
    varied_setting: str,
    reference_arm_id: str,
) -> None:
    if not arms:
        raise ValueError("benchmark arms must be non-empty")
    arm_ids = {arm.arm_id for arm in arms}
    if reference_arm_id and reference_arm_id not in arm_ids:
        raise ValueError(
            f"reference_arm_id references unknown arm {reference_arm_id!r}"
        )
    if comparison_mode == "methods_same_setting":
        offenders = [arm.arm_id for arm in arms if arm.setting_overrides]
        if offenders:
            raise ValueError(
                "methods_same_setting arms must not declare setting_overrides: "
                f"{offenders}"
            )
        if varied_setting:
            raise ValueError(
                "varied_setting is only valid for setting-variation comparisons"
            )
        return
    if comparison_mode != "single_method_setting_variation":
        raise ValueError("unsupported comparison_mode")
    _validate_non_empty_string(varied_setting, "varied_setting")
    if varied_setting not in BENCHMARK_ARM_ENVIRONMENT_FIELDS:
        raise ValueError(
            "varied_setting must name one typed runtime environment field; "
            f"got {varied_setting!r}"
        )
    if not reference_arm_id:
        raise ValueError(
            "single_method_setting_variation requires an explicit reference_arm_id"
        )
    method_ids = {arm.cache_method for arm in arms}
    if "" in method_ids or len(method_ids) != 1:
        raise ValueError(
            "setting-variation comparisons require one shared non-empty cache_method"
        )
    by_id = {arm.arm_id: arm for arm in arms}
    reference = by_id[reference_arm_id]
    invariant_fields: tuple[str, ...] = (
        "implementation_kind",
        "uses_cache",
        "cache_method",
        "method_version",
        "method_config_digest",
        "connector_mode",
        "requires_cachet_handoff",
        "physical_transform_id",
        "physical_transform_version",
        "physical_transform_config_digest",
        "scorer_plugin_path",
        "source_revision",
    )
    if varied_setting != "model_quantization":
        invariant_fields = (*invariant_fields, "checkpoint_identity")
    for arm in arms:
        if arm.arm_id == reference_arm_id:
            continue
        differences = tuple(
            field_name
            for field_name in invariant_fields
            if getattr(arm, field_name) != getattr(reference, field_name)
        )
        if differences:
            raise ValueError(
                "single_method_setting_variation requires invariant method and "
                f"implementation fields; arm {arm.arm_id!r} changes "
                f"{list(differences)}"
            )
    values: list[str] = []
    for arm in arms:
        if set(arm.setting_overrides) != {varied_setting}:
            raise ValueError(
                f"arm {arm.arm_id!r} setting_overrides must contain exactly "
                f"{varied_setting!r}"
            )
        values.append(_sha256_json(arm.setting_overrides[varied_setting]))
    if len(arms) > 1 and len(set(values)) != len(values):
        raise ValueError(
            "setting-variation arms must declare distinct varied setting values"
        )


def _kv_transfer_params_for_arm(
    example: BenchmarkExample,
    arm: BenchmarkArm,
    *,
    allow_legacy: bool,
) -> Mapping[str, Any]:
    if not arm.requires_cachet_handoff:
        return {}
    if arm.arm_id in example.arm_kv_transfer_params:
        return example.arm_kv_transfer_params[arm.arm_id]
    if allow_legacy:
        return example.kv_transfer_params
    raise ValueError(
        f"example {example.dataset}:{example.example_id} is missing "
        f"arm_kv_transfer_params for cache arm {arm.arm_id!r}"
    )


benchmark_experiment_manifest_to_record.__module__ = (
    "document_kv_cache.benchmark_runner"
)
