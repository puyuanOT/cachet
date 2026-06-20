import json
import os
import pickle
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

import document_kv_cache.benchmark_plan as public_benchmark_plan
import restaurant_kv_serving.benchmark_plan as legacy_benchmark_plan
from document_kv_cache.benchmark_plan import (
    BenchmarkDatasetPath,
    BenchmarkPlanConfig,
    ENGINE_PROBE_TARGETS_RECORD_TYPE,
    ENGINE_PROBE_TARGETS_SCHEMA_VERSION,
    EngineProbePlanConfig,
    ReleaseBundlePlanConfig,
    ReleaseEvidencePlanConfig,
    StorageBenchmarkPlanConfig,
    benchmark_job_plan_to_record,
    build_v1_benchmark_plan,
    engine_probe_targets_to_record,
    main,
    write_engine_probe_targets_json,
    write_benchmark_job_plan_shell,
)
from document_kv_cache.databricks_engine_probe_job import read_databricks_engine_probe_targets_json
from document_kv_cache.engine_adapters import ServingBackend
from document_kv_cache.native_probe_factories import SGLANG_NATIVE_PROBE_FACTORY, VLLM_NATIVE_PROBE_FACTORY


REPO_ROOT = Path(__file__).resolve().parents[1]


def dataset_paths(tmp_path):
    return tuple(
        BenchmarkDatasetPath(
            dataset=dataset,
            raw_jsonl=str(tmp_path / "raw" / f"{dataset}.jsonl"),
            prepared_jsonl=str(tmp_path / "prepared" / f"{dataset}.jsonl"),
        )
        for dataset in ("biography", "hotpotqa", "musique", "niah")
    )


def planned_release_probe_cli_args(tmp_path):
    return [
        "--raw-dataset",
        f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
        "--prepared-dir",
        str(tmp_path / "prepared"),
        "--base-url",
        "http://localhost:8000",
        "--allow-partial",
        "--storage-benchmark-workspace-dir",
        "/local_disk0/document-kv-storage-benchmark",
        "--storage-benchmark-output-json",
        str(tmp_path / "storage.json"),
        "--storage-benchmark-uc-volume-root",
        "/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
        "--engine-probe-handoff-json",
        f"vllm={tmp_path / 'vllm-handoff.json'}",
        "--engine-probe-factory",
        "vllm=vllm_probe:factory",
        "--engine-probe-output-json",
        f"vllm={tmp_path / 'vllm-probe.json'}",
        "--engine-probe-actions-output-json",
        f"vllm={tmp_path / 'vllm-actions.json'}",
        "--engine-probe-handoff-json",
        f"sglang={tmp_path / 'sglang-handoff.json'}",
        "--engine-probe-factory",
        "sglang=sglang_probe:factory",
        "--engine-probe-output-json",
        f"sglang={tmp_path / 'sglang-probe.json'}",
        "--engine-probe-actions-output-json",
        f"sglang={tmp_path / 'sglang-actions.json'}",
        "--release-evidence-output-json",
        str(tmp_path / "release-evidence.json"),
    ]


def test_benchmark_plan_config_keeps_native_probe_field_positional_compatibility():
    field_names = [field.name for field in fields(BenchmarkPlanConfig)]

    assert field_names.index("native_probe_factories_output_json") < field_names.index(
        "repository_hygiene_output_json"
    )
    assert field_names.index("repository_hygiene_output_json") < field_names.index(
        "github_governance_output_json"
    )
    assert field_names.index("github_governance_output_json") < field_names.index(
        "release_preflight_output_json"
    )


def release_action_jsons(tmp_path):
    return (
        str(tmp_path / "vllm-actions.json"),
        str(tmp_path / "sglang-actions.json"),
    )


def strict_release_bundle_plan_config(tmp_path, *, bundle_overrides=None, config_overrides=None):
    bundle_values = {
        "output_dir": str(tmp_path / "release-bundle"),
        "output_json": str(tmp_path / "release-bundle-manifest.json"),
        "preflight_json": str(tmp_path / "release-inputs.json"),
        "plan_execution_jsons": (str(tmp_path / "plan-execution.json"),),
        "databricks_run_status_jsons": (str(tmp_path / "databricks-run-status.json"),),
        "package_wheel": str(tmp_path / "dist" / "document_kv_cache-0.2.0-py3-none-any.whl"),
        "pr_evidence_jsons": (str(tmp_path / "pr-evidence.json"),),
        "github_governance_json": str(tmp_path / "github-governance.json"),
        "repository_hygiene_json": str(tmp_path / "repository-hygiene.json"),
        "native_probe_factories_jsons": (str(tmp_path / "native-probe-factories.json"),),
        "require_complete_v1": True,
    }
    if bundle_overrides is not None:
        bundle_values.update(bundle_overrides)

    config_values = {
        "suite_id": "v1-g5",
        "dataset_paths": dataset_paths(tmp_path),
        "base_url": "http://localhost:8000",
        "benchmark_output_json": str(tmp_path / "results.json"),
        "storage_benchmark": StorageBenchmarkPlanConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json=str(tmp_path / "storage.json"),
            readers=("memory", "disk", "unity_catalog"),
            uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
        ),
        "release_evidence": ReleaseEvidencePlanConfig(
            output_json=str(tmp_path / "release-evidence.json"),
            engine_probe_jsons=(
                str(tmp_path / "vllm-probe.json"),
                str(tmp_path / "sglang-probe.json"),
            ),
            engine_actions_jsons=release_action_jsons(tmp_path),
        ),
        "release_bundle": ReleaseBundlePlanConfig(**bundle_values),
    }
    if config_overrides is not None:
        config_values.update(config_overrides)
    return BenchmarkPlanConfig(**config_values)


def test_build_v1_benchmark_plan_prepares_all_datasets_then_runs_benchmark(tmp_path):
    config = BenchmarkPlanConfig(
        suite_id="v1-g5",
        dataset_paths=dataset_paths(tmp_path),
        base_url="http://localhost:8000",
        cache_base_url="http://cache:8000",
        cache_runtime_prompt=True,
        limit_per_dataset=5,
        benchmark_output_json=str(tmp_path / "results.json"),
    )

    plan = build_v1_benchmark_plan(config)
    record = benchmark_job_plan_to_record(plan)

    assert record["plan_version"] == "v1"
    assert record["model_id"] == "qwen3:4b-instruct"
    assert record["hardware_target"] == "aws-g5"
    assert [command.name for command in plan.preparation_commands] == [
        "prepare-biography",
        "prepare-hotpotqa",
        "prepare-musique",
        "prepare-niah",
    ]
    assert plan.preparation_commands[0].argv[:4] == (
        "python",
        "-m",
        "document_kv_cache.dataset_prep",
        "--dataset",
    )
    assert "--limit" in plan.preparation_commands[0].argv
    assert plan.benchmark_command.name == "run-benchmark"
    assert "--cache-base-url" in plan.benchmark_command.argv
    assert "--cache-runtime-prompt" in plan.benchmark_command.argv
    assert "--dataset" in plan.benchmark_command.argv
    assert "biography=" + str(tmp_path / "prepared" / "biography.jsonl") in plan.benchmark_command.argv


def test_build_v1_benchmark_plan_can_append_storage_reader_benchmark(tmp_path):
    config = BenchmarkPlanConfig(
        suite_id="v1-g5",
        dataset_paths=dataset_paths(tmp_path),
        base_url="http://localhost:8000",
        benchmark_output_json=str(tmp_path / "results.json"),
        storage_benchmark=StorageBenchmarkPlanConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json="/Volumes/catalog/schema/volume/storage-result.json",
            benchmark_id="storage-g5",
            chunk_count=8,
            chunk_bytes=1024,
            repeats=3,
            parallelism=2,
            readers=("disk", "unity_catalog"),
            uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
        ),
    )

    plan = build_v1_benchmark_plan(config)
    record = benchmark_job_plan_to_record(plan)
    storage_command = plan.post_benchmark_commands[0]

    assert [command.name for command in plan.commands][-2:] == ["run-benchmark", "run-storage-benchmark"]
    assert record["storage_benchmark_output_json"] == "/Volumes/catalog/schema/volume/storage-result.json"
    assert storage_command.argv[:3] == ("python", "-m", "document_kv_cache.storage_benchmark")
    assert "--workspace-dir" in storage_command.argv
    assert "/local_disk0/document-kv-storage-benchmark" in storage_command.argv
    assert "--reader" in storage_command.argv
    assert "disk" in storage_command.argv
    assert "unity_catalog" in storage_command.argv
    assert "--uc-volume-root" in storage_command.argv
    assert "/Volumes/catalog/schema/volume/document-kv-storage-benchmark" in storage_command.argv


def test_build_v1_benchmark_plan_can_append_release_evidence_validation(tmp_path):
    config = BenchmarkPlanConfig(
        suite_id="v1-g5",
        dataset_paths=dataset_paths(tmp_path),
        base_url="http://localhost:8000",
        benchmark_output_json=str(tmp_path / "results.json"),
        storage_benchmark=StorageBenchmarkPlanConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json=str(tmp_path / "storage.json"),
            readers=("memory", "disk", "unity_catalog"),
            uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
        ),
        release_evidence=ReleaseEvidencePlanConfig(
            output_json=str(tmp_path / "release-evidence.json"),
            engine_probe_jsons=(
                str(tmp_path / "vllm-probe.json"),
                str(tmp_path / "sglang-probe.json"),
            ),
            engine_actions_jsons=release_action_jsons(tmp_path),
        ),
    )

    plan = build_v1_benchmark_plan(config)
    record = benchmark_job_plan_to_record(plan)
    release_command = plan.post_benchmark_commands[-1]

    assert [command.name for command in plan.commands][-3:] == [
        "run-benchmark",
        "run-storage-benchmark",
        "validate-release-evidence",
    ]
    assert record["release_evidence_output_json"] == str(tmp_path / "release-evidence.json")
    assert record["release_storage_benchmark_json"] == str(tmp_path / "storage.json")
    assert record["release_engine_probe_jsons"] == [
        str(tmp_path / "vllm-probe.json"),
        str(tmp_path / "sglang-probe.json"),
    ]
    assert record["release_engine_actions_jsons"] == [
        str(tmp_path / "vllm-actions.json"),
        str(tmp_path / "sglang-actions.json"),
    ]
    assert release_command.argv[:3] == ("python", "-m", "document_kv_cache.release_evidence")
    assert release_command.argv[release_command.argv.index("--v1-benchmark-json") + 1] == str(
        tmp_path / "results.json"
    )
    assert release_command.argv[release_command.argv.index("--storage-benchmark-json") + 1] == str(
        tmp_path / "storage.json"
    )
    assert release_command.argv[release_command.argv.index("--output-json") + 1] == str(
        tmp_path / "release-evidence.json"
    )
    assert release_command.argv.count("--engine-probe-json") == 2
    assert release_command.argv.count("--engine-actions-json") == 2
    assert str(tmp_path / "vllm-probe.json") in release_command.argv
    assert str(tmp_path / "sglang-probe.json") in release_command.argv
    assert str(tmp_path / "vllm-actions.json") in release_command.argv
    assert str(tmp_path / "sglang-actions.json") in release_command.argv


def test_build_v1_benchmark_plan_can_append_release_bundle_after_release_evidence(tmp_path):
    config = BenchmarkPlanConfig(
        suite_id="v1-g5",
        dataset_paths=dataset_paths(tmp_path),
        base_url="http://localhost:8000",
        benchmark_output_json=str(tmp_path / "results.json"),
        storage_benchmark=StorageBenchmarkPlanConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json=str(tmp_path / "storage.json"),
            readers=("memory", "disk", "unity_catalog"),
            uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
        ),
        release_evidence=ReleaseEvidencePlanConfig(
            output_json=str(tmp_path / "release-evidence.json"),
            engine_probe_jsons=(
                str(tmp_path / "vllm-probe.json"),
                str(tmp_path / "sglang-probe.json"),
            ),
            engine_actions_jsons=release_action_jsons(tmp_path),
        ),
        release_bundle=ReleaseBundlePlanConfig(
            output_dir=str(tmp_path / "release-bundle"),
            output_json=str(tmp_path / "release-bundle-manifest.json"),
            preflight_json=str(tmp_path / "release-inputs.json"),
            plan_execution_jsons=(str(tmp_path / "plan-execution.json"),),
            databricks_run_status_jsons=(str(tmp_path / "databricks-run-status.json"),),
            package_wheel=str(tmp_path / "dist" / "document_kv_cache-0.2.0-py3-none-any.whl"),
            pr_evidence_jsons=(str(tmp_path / "pr-evidence.json"),),
            github_governance_json=str(tmp_path / "github-governance.json"),
            repository_hygiene_json=str(tmp_path / "repository-hygiene.json"),
            native_probe_factories_jsons=(str(tmp_path / "native-probe-factories.json"),),
            overwrite=True,
            require_complete_v1=True,
        ),
    )

    plan = build_v1_benchmark_plan(config)
    record = benchmark_job_plan_to_record(plan)
    bundle_command = plan.post_benchmark_commands[-1]

    assert [command.name for command in plan.commands][-4:] == [
        "run-benchmark",
        "run-storage-benchmark",
        "validate-release-evidence",
        "build-release-bundle",
    ]
    assert record["release_bundle_output_dir"] == str(tmp_path / "release-bundle")
    assert record["release_bundle_output_json"] == str(tmp_path / "release-bundle-manifest.json")
    assert record["release_bundle"] == {
        "databricks_run_status_jsons": [str(tmp_path / "databricks-run-status.json")],
        "engine_probe_jsons": [
            str(tmp_path / "vllm-probe.json"),
            str(tmp_path / "sglang-probe.json"),
        ],
        "engine_actions_jsons": [
            str(tmp_path / "vllm-actions.json"),
            str(tmp_path / "sglang-actions.json"),
        ],
        "github_governance_json": str(tmp_path / "github-governance.json"),
        "native_probe_factories_jsons": [str(tmp_path / "native-probe-factories.json")],
        "output_dir": str(tmp_path / "release-bundle"),
        "output_json": str(tmp_path / "release-bundle-manifest.json"),
        "overwrite": True,
        "package_wheel": str(tmp_path / "dist" / "document_kv_cache-0.2.0-py3-none-any.whl"),
        "plan_execution_jsons": [str(tmp_path / "plan-execution.json")],
        "preflight_json": str(tmp_path / "release-inputs.json"),
        "pr_evidence_jsons": [str(tmp_path / "pr-evidence.json")],
        "release_evidence_json": str(tmp_path / "release-evidence.json"),
        "repository_hygiene_json": str(tmp_path / "repository-hygiene.json"),
        "require_complete_v1": True,
        "storage_benchmark_json": str(tmp_path / "storage.json"),
        "v1_benchmark_json": str(tmp_path / "results.json"),
    }
    assert bundle_command.argv[:3] == ("python", "-m", "document_kv_cache.release_bundle")
    assert bundle_command.argv[bundle_command.argv.index("--release-evidence-json") + 1] == str(
        tmp_path / "release-evidence.json"
    )
    assert bundle_command.argv[bundle_command.argv.index("--output-dir") + 1] == str(tmp_path / "release-bundle")
    assert bundle_command.argv[bundle_command.argv.index("--output-json") + 1] == str(
        tmp_path / "release-bundle-manifest.json"
    )
    assert bundle_command.argv.count("--engine-probe-json") == 2
    assert bundle_command.argv.count("--engine-actions-json") == 2
    assert "--preflight-json" in bundle_command.argv
    assert "--plan-execution-json" in bundle_command.argv
    assert "--databricks-run-status-json" in bundle_command.argv
    assert "--package-wheel" in bundle_command.argv
    assert "--pr-evidence-json" in bundle_command.argv
    assert "--github-governance-json" in bundle_command.argv
    assert "--repository-hygiene-json" in bundle_command.argv
    assert "--native-probe-factories-json" in bundle_command.argv
    assert "--require-complete-v1" in bundle_command.argv
    assert "--overwrite" in bundle_command.argv


def test_build_v1_benchmark_plan_omits_strict_release_bundle_flag_by_default(tmp_path):
    config = BenchmarkPlanConfig(
        suite_id="v1-g5",
        dataset_paths=dataset_paths(tmp_path),
        base_url="http://localhost:8000",
        benchmark_output_json=str(tmp_path / "results.json"),
        storage_benchmark=StorageBenchmarkPlanConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json=str(tmp_path / "storage.json"),
            readers=("memory", "disk", "unity_catalog"),
            uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
        ),
        release_evidence=ReleaseEvidencePlanConfig(
            output_json=str(tmp_path / "release-evidence.json"),
            engine_probe_jsons=(
                str(tmp_path / "vllm-probe.json"),
                str(tmp_path / "sglang-probe.json"),
            ),
            engine_actions_jsons=release_action_jsons(tmp_path),
        ),
        release_bundle=ReleaseBundlePlanConfig(
            output_dir=str(tmp_path / "release-bundle"),
            output_json=str(tmp_path / "release-bundle-manifest.json"),
        ),
    )

    plan = build_v1_benchmark_plan(config)
    record = benchmark_job_plan_to_record(plan)
    bundle_command = plan.post_benchmark_commands[-1]

    assert record["release_bundle"]["require_complete_v1"] is False
    assert "--require-complete-v1" not in bundle_command.argv


@pytest.mark.parametrize(
    ("bundle_overrides", "config_overrides", "expected_missing"),
    [
        ({"preflight_json": None}, None, "preflight sidecar"),
        ({"plan_execution_jsons": ()}, None, "benchmark plan execution sidecar"),
        ({"databricks_run_status_jsons": ()}, None, "Databricks run-status sidecar"),
        ({"package_wheel": None}, None, "tested package wheel"),
        ({"pr_evidence_jsons": ()}, None, "PR evidence sidecar"),
        ({"github_governance_json": None}, None, "GitHub governance sidecar"),
        ({"repository_hygiene_json": None}, None, "repository hygiene sidecar"),
        ({"native_probe_factories_jsons": ()}, None, "native probe factory diagnostics sidecar"),
    ],
)
def test_benchmark_plan_rejects_incomplete_strict_release_bundle(
    tmp_path,
    bundle_overrides,
    config_overrides,
    expected_missing,
):
    with pytest.raises(ValueError, match="strict V1 release bundle plans require") as error:
        strict_release_bundle_plan_config(
            tmp_path,
            bundle_overrides=bundle_overrides,
            config_overrides=config_overrides,
        )

    assert expected_missing in str(error.value)


def test_benchmark_plan_accepts_generated_strict_release_bundle_sidecars(tmp_path):
    config = strict_release_bundle_plan_config(
        tmp_path,
        bundle_overrides={
            "preflight_json": None,
            "github_governance_json": None,
            "repository_hygiene_json": None,
            "native_probe_factories_jsons": (),
        },
        config_overrides={
            "release_preflight_output_json": str(tmp_path / "release-inputs.json"),
            "github_governance_output_json": str(tmp_path / "github-governance.json"),
            "repository_hygiene_output_json": str(tmp_path / "repository-hygiene.json"),
            "native_probe_factories_output_json": str(tmp_path / "native-probe-factories.json"),
        },
    )

    record = benchmark_job_plan_to_record(build_v1_benchmark_plan(config))

    assert record["release_bundle"]["preflight_json"] == str(tmp_path / "release-inputs.json")
    assert record["release_bundle"]["github_governance_json"] == str(tmp_path / "github-governance.json")
    assert record["release_bundle"]["repository_hygiene_json"] == str(tmp_path / "repository-hygiene.json")
    assert record["release_bundle"]["native_probe_factories_jsons"] == [
        str(tmp_path / "native-probe-factories.json")
    ]


def test_build_v1_benchmark_plan_can_generate_and_bundle_github_governance(tmp_path):
    github_governance_json = str(tmp_path / "github-governance.json")
    config = BenchmarkPlanConfig(
        suite_id="v1-g5",
        dataset_paths=dataset_paths(tmp_path),
        base_url="http://localhost:8000",
        benchmark_output_json=str(tmp_path / "results.json"),
        storage_benchmark=StorageBenchmarkPlanConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json=str(tmp_path / "storage.json"),
            readers=("memory", "disk", "unity_catalog"),
            uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
        ),
        release_evidence=ReleaseEvidencePlanConfig(
            output_json=str(tmp_path / "release-evidence.json"),
            engine_probe_jsons=(
                str(tmp_path / "vllm-probe.json"),
                str(tmp_path / "sglang-probe.json"),
            ),
            engine_actions_jsons=release_action_jsons(tmp_path),
        ),
        release_bundle=ReleaseBundlePlanConfig(
            output_dir=str(tmp_path / "release-bundle"),
            output_json=str(tmp_path / "release-bundle-manifest.json"),
        ),
        github_governance_output_json=github_governance_json,
    )

    plan = build_v1_benchmark_plan(config)
    record = benchmark_job_plan_to_record(plan)
    governance_command = next(command for command in plan.commands if command.name == "inspect-github-governance")
    bundle_command = plan.post_benchmark_commands[-1]

    assert [command.name for command in plan.commands][-5:] == [
        "run-benchmark",
        "run-storage-benchmark",
        "inspect-github-governance",
        "validate-release-evidence",
        "build-release-bundle",
    ]
    assert record["github_governance_output_json"] == github_governance_json
    assert record["release_bundle"]["github_governance_json"] == github_governance_json
    assert governance_command.argv == (
        "python",
        "-m",
        "document_kv_cache.github_governance",
        "--output-json",
        github_governance_json,
    )
    assert bundle_command.argv.count("--github-governance-json") == 1
    assert bundle_command.argv[bundle_command.argv.index("--github-governance-json") + 1] == (
        github_governance_json
    )


def test_build_v1_benchmark_plan_allows_equivalent_generated_github_governance_bundle_path(tmp_path):
    github_governance_json = str(tmp_path / "github-governance.json")
    equivalent_github_governance_json = str(tmp_path / "subdir" / ".." / "github-governance.json")
    config = BenchmarkPlanConfig(
        suite_id="v1-g5",
        dataset_paths=dataset_paths(tmp_path),
        base_url="http://localhost:8000",
        benchmark_output_json=str(tmp_path / "results.json"),
        storage_benchmark=StorageBenchmarkPlanConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json=str(tmp_path / "storage.json"),
            readers=("memory", "disk", "unity_catalog"),
            uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
        ),
        release_evidence=ReleaseEvidencePlanConfig(
            output_json=str(tmp_path / "release-evidence.json"),
            engine_probe_jsons=(
                str(tmp_path / "vllm-probe.json"),
                str(tmp_path / "sglang-probe.json"),
            ),
            engine_actions_jsons=release_action_jsons(tmp_path),
        ),
        release_bundle=ReleaseBundlePlanConfig(
            output_dir=str(tmp_path / "release-bundle"),
            output_json=str(tmp_path / "release-bundle-manifest.json"),
            github_governance_json=equivalent_github_governance_json,
        ),
        github_governance_output_json=github_governance_json,
    )

    plan = build_v1_benchmark_plan(config)
    record = benchmark_job_plan_to_record(plan)
    bundle_command = plan.post_benchmark_commands[-1]

    assert record["release_bundle"]["github_governance_json"] == github_governance_json
    assert bundle_command.argv.count("--github-governance-json") == 1
    assert bundle_command.argv[bundle_command.argv.index("--github-governance-json") + 1] == (
        github_governance_json
    )


def test_benchmark_plan_rejects_conflicting_generated_github_governance_bundle_path(tmp_path):
    with pytest.raises(
        ValueError,
        match="release bundle github_governance_json must match github_governance_output_json",
    ):
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=dataset_paths(tmp_path),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
            storage_benchmark=StorageBenchmarkPlanConfig(
                workspace_dir="/local_disk0/document-kv-storage-benchmark",
                output_json=str(tmp_path / "storage.json"),
                readers=("memory", "disk", "unity_catalog"),
                uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
            ),
            release_evidence=ReleaseEvidencePlanConfig(
                output_json=str(tmp_path / "release-evidence.json"),
                engine_probe_jsons=(
                    str(tmp_path / "vllm-probe.json"),
                    str(tmp_path / "sglang-probe.json"),
                ),
                engine_actions_jsons=release_action_jsons(tmp_path),
            ),
            release_bundle=ReleaseBundlePlanConfig(
                output_dir=str(tmp_path / "release-bundle"),
                output_json=str(tmp_path / "release-bundle-manifest.json"),
                github_governance_json=str(tmp_path / "other-github-governance.json"),
            ),
            github_governance_output_json=str(tmp_path / "github-governance.json"),
        )


def test_build_v1_benchmark_plan_can_generate_and_bundle_repository_hygiene(tmp_path):
    repository_hygiene_json = str(tmp_path / "repository-hygiene.json")
    config = BenchmarkPlanConfig(
        suite_id="v1-g5",
        dataset_paths=dataset_paths(tmp_path),
        base_url="http://localhost:8000",
        benchmark_output_json=str(tmp_path / "results.json"),
        storage_benchmark=StorageBenchmarkPlanConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json=str(tmp_path / "storage.json"),
            readers=("memory", "disk", "unity_catalog"),
            uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
        ),
        release_evidence=ReleaseEvidencePlanConfig(
            output_json=str(tmp_path / "release-evidence.json"),
            engine_probe_jsons=(
                str(tmp_path / "vllm-probe.json"),
                str(tmp_path / "sglang-probe.json"),
            ),
            engine_actions_jsons=release_action_jsons(tmp_path),
        ),
        release_bundle=ReleaseBundlePlanConfig(
            output_dir=str(tmp_path / "release-bundle"),
            output_json=str(tmp_path / "release-bundle-manifest.json"),
        ),
        repository_hygiene_output_json=repository_hygiene_json,
    )

    plan = build_v1_benchmark_plan(config)
    record = benchmark_job_plan_to_record(plan)
    hygiene_command = next(command for command in plan.commands if command.name == "inspect-repository-hygiene")
    bundle_command = plan.post_benchmark_commands[-1]

    assert [command.name for command in plan.commands][-5:] == [
        "run-benchmark",
        "run-storage-benchmark",
        "inspect-repository-hygiene",
        "validate-release-evidence",
        "build-release-bundle",
    ]
    assert record["repository_hygiene_output_json"] == repository_hygiene_json
    assert record["release_bundle"]["repository_hygiene_json"] == repository_hygiene_json
    assert hygiene_command.argv == (
        "python",
        "-m",
        "document_kv_cache.repository_hygiene",
        "--repository-root",
        ".",
        "--output-json",
        repository_hygiene_json,
    )
    assert bundle_command.argv.count("--repository-hygiene-json") == 1
    assert bundle_command.argv[bundle_command.argv.index("--repository-hygiene-json") + 1] == (
        repository_hygiene_json
    )


def test_build_v1_benchmark_plan_allows_equivalent_generated_repository_hygiene_bundle_path(tmp_path):
    repository_hygiene_json = str(tmp_path / "repository-hygiene.json")
    equivalent_repository_hygiene_json = str(tmp_path / "subdir" / ".." / "repository-hygiene.json")
    config = BenchmarkPlanConfig(
        suite_id="v1-g5",
        dataset_paths=dataset_paths(tmp_path),
        base_url="http://localhost:8000",
        benchmark_output_json=str(tmp_path / "results.json"),
        storage_benchmark=StorageBenchmarkPlanConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json=str(tmp_path / "storage.json"),
            readers=("memory", "disk", "unity_catalog"),
            uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
        ),
        release_evidence=ReleaseEvidencePlanConfig(
            output_json=str(tmp_path / "release-evidence.json"),
            engine_probe_jsons=(
                str(tmp_path / "vllm-probe.json"),
                str(tmp_path / "sglang-probe.json"),
            ),
            engine_actions_jsons=release_action_jsons(tmp_path),
        ),
        release_bundle=ReleaseBundlePlanConfig(
            output_dir=str(tmp_path / "release-bundle"),
            output_json=str(tmp_path / "release-bundle-manifest.json"),
            repository_hygiene_json=equivalent_repository_hygiene_json,
        ),
        repository_hygiene_output_json=repository_hygiene_json,
    )

    plan = build_v1_benchmark_plan(config)
    record = benchmark_job_plan_to_record(plan)
    bundle_command = plan.post_benchmark_commands[-1]

    assert record["release_bundle"]["repository_hygiene_json"] == repository_hygiene_json
    assert bundle_command.argv.count("--repository-hygiene-json") == 1
    assert bundle_command.argv[bundle_command.argv.index("--repository-hygiene-json") + 1] == (
        repository_hygiene_json
    )


def test_benchmark_plan_rejects_conflicting_generated_repository_hygiene_bundle_path(tmp_path):
    with pytest.raises(
        ValueError,
        match="release bundle repository_hygiene_json must match repository_hygiene_output_json",
    ):
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=dataset_paths(tmp_path),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
            storage_benchmark=StorageBenchmarkPlanConfig(
                workspace_dir="/local_disk0/document-kv-storage-benchmark",
                output_json=str(tmp_path / "storage.json"),
                readers=("memory", "disk", "unity_catalog"),
                uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
            ),
            release_evidence=ReleaseEvidencePlanConfig(
                output_json=str(tmp_path / "release-evidence.json"),
                engine_probe_jsons=(
                    str(tmp_path / "vllm-probe.json"),
                    str(tmp_path / "sglang-probe.json"),
                ),
                engine_actions_jsons=release_action_jsons(tmp_path),
            ),
            release_bundle=ReleaseBundlePlanConfig(
                output_dir=str(tmp_path / "release-bundle"),
                output_json=str(tmp_path / "release-bundle-manifest.json"),
                repository_hygiene_json=str(tmp_path / "other-repository-hygiene.json"),
            ),
            repository_hygiene_output_json=str(tmp_path / "repository-hygiene.json"),
        )


def test_build_v1_benchmark_plan_can_generate_and_bundle_native_probe_factories(tmp_path):
    native_probe_factories_json = str(tmp_path / "native-probe-factories.json")
    config = BenchmarkPlanConfig(
        suite_id="v1-g5",
        dataset_paths=dataset_paths(tmp_path),
        base_url="http://localhost:8000",
        benchmark_output_json=str(tmp_path / "results.json"),
        storage_benchmark=StorageBenchmarkPlanConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json=str(tmp_path / "storage.json"),
            readers=("memory", "disk", "unity_catalog"),
            uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
        ),
        release_evidence=ReleaseEvidencePlanConfig(
            output_json=str(tmp_path / "release-evidence.json"),
            engine_probe_jsons=(
                str(tmp_path / "vllm-probe.json"),
                str(tmp_path / "sglang-probe.json"),
            ),
            engine_actions_jsons=release_action_jsons(tmp_path),
        ),
        release_bundle=ReleaseBundlePlanConfig(
            output_dir=str(tmp_path / "release-bundle"),
            output_json=str(tmp_path / "release-bundle-manifest.json"),
        ),
        native_probe_factories_output_json=native_probe_factories_json,
    )

    plan = build_v1_benchmark_plan(config)
    record = benchmark_job_plan_to_record(plan)
    native_command = next(command for command in plan.commands if command.name == "inspect-native-probe-factories")
    bundle_command = plan.post_benchmark_commands[-1]

    assert [command.name for command in plan.commands][-5:] == [
        "run-benchmark",
        "run-storage-benchmark",
        "inspect-native-probe-factories",
        "validate-release-evidence",
        "build-release-bundle",
    ]
    assert record["native_probe_factories_output_json"] == native_probe_factories_json
    assert record["release_bundle"]["native_probe_factories_jsons"] == [native_probe_factories_json]
    assert native_command.argv == (
        "python",
        "-m",
        "document_kv_cache.native_probe_factories",
        "--output-json",
        native_probe_factories_json,
    )
    assert bundle_command.argv[bundle_command.argv.index("--native-probe-factories-json") + 1] == (
        native_probe_factories_json
    )


def test_build_v1_benchmark_plan_dedupes_equivalent_native_probe_factories_paths(tmp_path):
    native_probe_factories_json = str(tmp_path / "native-probe-factories.json")
    equivalent_native_probe_factories_json = str(tmp_path / "subdir" / ".." / "native-probe-factories.json")
    config = BenchmarkPlanConfig(
        suite_id="v1-g5",
        dataset_paths=dataset_paths(tmp_path),
        base_url="http://localhost:8000",
        benchmark_output_json=str(tmp_path / "results.json"),
        storage_benchmark=StorageBenchmarkPlanConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json=str(tmp_path / "storage.json"),
            readers=("memory", "disk", "unity_catalog"),
            uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
        ),
        release_evidence=ReleaseEvidencePlanConfig(
            output_json=str(tmp_path / "release-evidence.json"),
            engine_probe_jsons=(
                str(tmp_path / "vllm-probe.json"),
                str(tmp_path / "sglang-probe.json"),
            ),
            engine_actions_jsons=release_action_jsons(tmp_path),
        ),
        release_bundle=ReleaseBundlePlanConfig(
            output_dir=str(tmp_path / "release-bundle"),
            output_json=str(tmp_path / "release-bundle-manifest.json"),
            native_probe_factories_jsons=(equivalent_native_probe_factories_json,),
        ),
        native_probe_factories_output_json=native_probe_factories_json,
    )

    plan = build_v1_benchmark_plan(config)
    record = benchmark_job_plan_to_record(plan)
    bundle_command = plan.post_benchmark_commands[-1]

    assert record["release_bundle"]["native_probe_factories_jsons"] == [native_probe_factories_json]
    assert bundle_command.argv.count("--native-probe-factories-json") == 1
    assert bundle_command.argv[bundle_command.argv.index("--native-probe-factories-json") + 1] == (
        native_probe_factories_json
    )


def test_build_v1_benchmark_plan_can_run_planned_engine_probes_before_release_validation(tmp_path):
    config = BenchmarkPlanConfig(
        suite_id="v1-g5",
        dataset_paths=dataset_paths(tmp_path),
        base_url="http://localhost:8000",
        benchmark_output_json=str(tmp_path / "results.json"),
        storage_benchmark=StorageBenchmarkPlanConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json=str(tmp_path / "storage.json"),
            readers=("memory", "disk", "unity_catalog"),
            uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
        ),
        engine_probes=(
            EngineProbePlanConfig(
                backend="vllm",
                handoff_json=str(tmp_path / "vllm-handoff.json"),
                probe_factory="vllm_probe:factory",
                output_json=str(tmp_path / "vllm-probe.json"),
                actions_output_json=str(tmp_path / "vllm-actions.json"),
                payload_uri=f"disk:{tmp_path / 'vllm.kv'}",
                metadata=("probe.source=plan",),
            ),
            EngineProbePlanConfig(
                backend="sglang",
                handoff_json=str(tmp_path / "sglang-handoff.json"),
                probe_factory="sglang_probe:factory",
                output_json=str(tmp_path / "sglang-probe.json"),
                actions_output_json=str(tmp_path / "sglang-actions.json"),
            ),
        ),
        release_evidence=ReleaseEvidencePlanConfig(
            output_json=str(tmp_path / "release-evidence.json"),
        ),
    )

    plan = build_v1_benchmark_plan(config)
    record = benchmark_job_plan_to_record(plan)
    command_names = [command.name for command in plan.commands]
    vllm_command = plan.post_benchmark_commands[1]
    release_command = plan.post_benchmark_commands[-1]

    assert command_names[-4:] == [
        "run-storage-benchmark",
        "run-vllm-engine-probe",
        "run-sglang-engine-probe",
        "validate-release-evidence",
    ]
    assert record["planned_engine_probes"] == [
        {
            "allow_non_native_probe": False,
            "backend": "vllm",
            "engine_version": None,
            "handoff_json": str(tmp_path / "vllm-handoff.json"),
            "metadata": ["probe.source=plan"],
            "output_json": str(tmp_path / "vllm-probe.json"),
            "actions_output_json": str(tmp_path / "vllm-actions.json"),
            "payload_uri": f"disk:{tmp_path / 'vllm.kv'}",
            "probe_factory": "vllm_probe:factory",
        },
        {
            "allow_non_native_probe": False,
            "backend": "sglang",
            "engine_version": None,
            "handoff_json": str(tmp_path / "sglang-handoff.json"),
            "metadata": [],
            "output_json": str(tmp_path / "sglang-probe.json"),
            "actions_output_json": str(tmp_path / "sglang-actions.json"),
            "payload_uri": None,
            "probe_factory": "sglang_probe:factory",
        },
    ]
    assert record["release_engine_probe_jsons"] == [
        str(tmp_path / "vllm-probe.json"),
        str(tmp_path / "sglang-probe.json"),
    ]
    assert record["release_engine_actions_jsons"] == [
        str(tmp_path / "vllm-actions.json"),
        str(tmp_path / "sglang-actions.json"),
    ]
    assert vllm_command.argv[:3] == ("python", "-m", "document_kv_cache.engine_probe")
    assert "--expected-backend" in vllm_command.argv
    assert vllm_command.argv[vllm_command.argv.index("--expected-backend") + 1] == "vllm"
    assert "--payload-uri" in vllm_command.argv
    assert vllm_command.argv[vllm_command.argv.index("--actions-output-json") + 1] == str(
        tmp_path / "vllm-actions.json"
    )
    assert "--metadata" in vllm_command.argv
    assert release_command.argv.count("--engine-probe-json") == 2
    assert release_command.argv.count("--engine-actions-json") == 2
    assert str(tmp_path / "sglang-probe.json") in release_command.argv
    assert str(tmp_path / "vllm-probe.json") in release_command.argv
    assert str(tmp_path / "sglang-actions.json") in release_command.argv
    assert str(tmp_path / "vllm-actions.json") in release_command.argv


def test_engine_probe_targets_record_can_feed_databricks_matrix_helper(tmp_path):
    probes = (
        EngineProbePlanConfig(
            backend="vllm",
            handoff_json=str(tmp_path / "vllm-handoff.json"),
            probe_factory="vllm_probe:factory",
            output_json=str(tmp_path / "vllm-probe.json"),
            actions_output_json=str(tmp_path / "vllm-actions.json"),
            payload_uri=f"disk:{tmp_path / 'vllm.kv'}",
            metadata=("probe.source=plan",),
        ),
        EngineProbePlanConfig(
            backend="sglang",
            handoff_json=str(tmp_path / "sglang-handoff.json"),
            probe_factory="sglang_probe:factory",
            output_json=str(tmp_path / "sglang-probe.json"),
            actions_output_json=str(tmp_path / "sglang-actions.json"),
        ),
    )
    record = engine_probe_targets_to_record(probes, release_safe=True)

    assert record == {
        "record_type": ENGINE_PROBE_TARGETS_RECORD_TYPE,
        "schema_version": ENGINE_PROBE_TARGETS_SCHEMA_VERSION,
        "release_safe": True,
        "probes": [
            {
                "allow_non_native_probe": False,
                "backend": "vllm",
                "handoff_json": str(tmp_path / "vllm-handoff.json"),
                "metadata": ["probe.source=plan"],
                "output_json": str(tmp_path / "vllm-probe.json"),
                "actions_output_json": str(tmp_path / "vllm-actions.json"),
                "payload_uri": f"disk:{tmp_path / 'vllm.kv'}",
                "probe_factory": "vllm_probe:factory",
            },
            {
                "allow_non_native_probe": False,
                "backend": "sglang",
                "handoff_json": str(tmp_path / "sglang-handoff.json"),
                "metadata": [],
                "output_json": str(tmp_path / "sglang-probe.json"),
                "actions_output_json": str(tmp_path / "sglang-actions.json"),
                "probe_factory": "sglang_probe:factory",
            },
        ],
    }

    target_path = tmp_path / "engine-probe-targets.json"
    plan = build_v1_benchmark_plan(
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=dataset_paths(tmp_path),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
            engine_probes=probes,
        )
    )
    write_engine_probe_targets_json(plan, target_path, release_safe=True)

    targets = read_databricks_engine_probe_targets_json(target_path)
    assert [target.expected_backend for target in targets] == [ServingBackend.VLLM, ServingBackend.SGLANG]
    assert [target.actions_output_json for target in targets] == [
        str(tmp_path / "vllm-actions.json"),
        str(tmp_path / "sglang-actions.json"),
    ]
    assert targets[0].metadata == ("probe.source=plan",)


def test_engine_probe_targets_release_safe_rejects_debug_or_incomplete_planned_probes(tmp_path):
    with pytest.raises(ValueError, match="exactly the release backends"):
        engine_probe_targets_to_record(
            (
                EngineProbePlanConfig(
                    backend="vllm",
                    handoff_json=str(tmp_path / "vllm-handoff.json"),
                    probe_factory="vllm_probe:factory",
                    output_json=str(tmp_path / "vllm-probe.json"),
                ),
            ),
            release_safe=True,
        )

    with pytest.raises(ValueError, match="release_evidence cannot consume planned debug engine probes"):
        engine_probe_targets_to_record(
            (
                EngineProbePlanConfig(
                    backend="vllm",
                    handoff_json=str(tmp_path / "vllm-handoff.json"),
                    probe_factory="vllm_probe:factory",
                    output_json=str(tmp_path / "vllm-probe.json"),
                    engine_version="debug-vllm",
                ),
                EngineProbePlanConfig(
                    backend="sglang",
                    handoff_json=str(tmp_path / "sglang-handoff.json"),
                    probe_factory="sglang_probe:factory",
                    output_json=str(tmp_path / "sglang-probe.json"),
                ),
            ),
            release_safe=True,
        )


def test_release_evidence_plan_can_use_existing_storage_benchmark_json(tmp_path):
    config = BenchmarkPlanConfig(
        suite_id="v1-g5",
        dataset_paths=dataset_paths(tmp_path),
        base_url="http://localhost:8000",
        benchmark_output_json=str(tmp_path / "results.json"),
        release_evidence=ReleaseEvidencePlanConfig(
            output_json=str(tmp_path / "release-evidence.json"),
            engine_probe_jsons=(
                str(tmp_path / "vllm-probe.json"),
                str(tmp_path / "sglang-probe.json"),
            ),
            engine_actions_jsons=release_action_jsons(tmp_path),
            storage_benchmark_json=str(tmp_path / "existing-storage.json"),
        ),
    )

    plan = build_v1_benchmark_plan(config)
    record = benchmark_job_plan_to_record(plan)
    release_argv = plan.post_benchmark_commands[-1].argv

    assert record["release_storage_benchmark_json"] == str(tmp_path / "existing-storage.json")
    assert record["release_engine_actions_jsons"] == [
        str(tmp_path / "vllm-actions.json"),
        str(tmp_path / "sglang-actions.json"),
    ]
    assert release_argv[release_argv.index("--storage-benchmark-json") + 1] == str(
        tmp_path / "existing-storage.json"
    )
    assert release_argv.count("--engine-actions-json") == 2


def test_benchmark_plan_requires_storage_artifact_for_release_evidence(tmp_path):
    with pytest.raises(ValueError, match="release_evidence requires"):
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=dataset_paths(tmp_path),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
            release_evidence=ReleaseEvidencePlanConfig(
                output_json=str(tmp_path / "release-evidence.json"),
                engine_probe_jsons=(str(tmp_path / "vllm-probe.json"),),
            ),
        )


def test_benchmark_plan_requires_explicit_release_probe_jsons_for_all_backends(tmp_path):
    with pytest.raises(ValueError, match="all release backends"):
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=dataset_paths(tmp_path),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
            release_evidence=ReleaseEvidencePlanConfig(
                output_json=str(tmp_path / "release-evidence.json"),
                engine_probe_jsons=(str(tmp_path / "vllm-probe.json"),),
                engine_actions_jsons=release_action_jsons(tmp_path),
                storage_benchmark_json=str(tmp_path / "existing-storage.json"),
            ),
        )


def test_release_evidence_plan_rejects_duplicate_explicit_probe_jsons(tmp_path):
    with pytest.raises(ValueError, match="engine_probe_jsons entries must be distinct"):
        ReleaseEvidencePlanConfig(
            output_json=str(tmp_path / "release-evidence.json"),
            engine_probe_jsons=(
                str(tmp_path / "probe.json"),
                str(tmp_path / "probe.json"),
            ),
            storage_benchmark_json=str(tmp_path / "storage.json"),
        )


def test_release_evidence_plan_rejects_equivalent_explicit_probe_json_paths(tmp_path):
    with pytest.raises(ValueError, match="engine_probe_jsons entries must be distinct"):
        ReleaseEvidencePlanConfig(
            output_json=str(tmp_path / "release-evidence.json"),
            engine_probe_jsons=(
                str(tmp_path / "probe.json"),
                str(tmp_path / "subdir" / ".." / "probe.json"),
            ),
            storage_benchmark_json=str(tmp_path / "storage.json"),
        )


def test_release_evidence_plan_rejects_local_uri_alias_probe_json_paths(tmp_path):
    with pytest.raises(ValueError, match="engine_probe_jsons entries must be distinct"):
        ReleaseEvidencePlanConfig(
            output_json=str(tmp_path / "release-evidence.json"),
            engine_probe_jsons=(
                str(tmp_path / "probe.json"),
                f"file:{tmp_path / 'probe.json'}",
            ),
            storage_benchmark_json=str(tmp_path / "storage.json"),
        )


def test_benchmark_plan_requires_release_evidence_before_release_bundle(tmp_path):
    with pytest.raises(ValueError, match="release_bundle requires release_evidence"):
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=dataset_paths(tmp_path),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
            release_bundle=ReleaseBundlePlanConfig(
                output_dir=str(tmp_path / "release-bundle"),
                output_json=str(tmp_path / "release-bundle-manifest.json"),
            ),
        )


def test_release_bundle_plan_rejects_non_boolean_strict_v1_flag(tmp_path):
    with pytest.raises(ValueError, match="release bundle require_complete_v1 must be boolean"):
        ReleaseBundlePlanConfig(
            output_dir=str(tmp_path / "release-bundle"),
            output_json=str(tmp_path / "release-bundle-manifest.json"),
            require_complete_v1=1,  # type: ignore[arg-type]
        )


def test_benchmark_plan_release_evidence_can_use_planned_engine_probe_outputs(tmp_path):
    config = BenchmarkPlanConfig(
        suite_id="v1-g5",
        dataset_paths=dataset_paths(tmp_path),
        base_url="http://localhost:8000",
        benchmark_output_json=str(tmp_path / "results.json"),
        release_evidence=ReleaseEvidencePlanConfig(
            output_json=str(tmp_path / "release-evidence.json"),
            storage_benchmark_json=str(tmp_path / "storage.json"),
        ),
        engine_probes=(
            EngineProbePlanConfig(
                backend="vllm",
                handoff_json=str(tmp_path / "vllm-handoff.json"),
                probe_factory="probe:factory",
                output_json=str(tmp_path / "vllm-probe.json"),
                actions_output_json=str(tmp_path / "vllm-actions.json"),
            ),
            EngineProbePlanConfig(
                backend="sglang",
                handoff_json=str(tmp_path / "sglang-handoff.json"),
                probe_factory="probe:factory",
                output_json=str(tmp_path / "sglang-probe.json"),
                actions_output_json=str(tmp_path / "sglang-actions.json"),
            ),
        ),
    )

    release_argv = build_v1_benchmark_plan(config).post_benchmark_commands[-1].argv

    assert str(tmp_path / "vllm-probe.json") in release_argv
    assert str(tmp_path / "sglang-probe.json") in release_argv
    assert str(tmp_path / "vllm-actions.json") in release_argv
    assert str(tmp_path / "sglang-actions.json") in release_argv


def test_benchmark_plan_requires_all_release_backends_when_using_planned_probe_outputs(tmp_path):
    with pytest.raises(ValueError, match="all release backends"):
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=dataset_paths(tmp_path),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
            release_evidence=ReleaseEvidencePlanConfig(
                output_json=str(tmp_path / "release-evidence.json"),
                storage_benchmark_json=str(tmp_path / "storage.json"),
            ),
            engine_probes=(
                EngineProbePlanConfig(
                    backend="vllm",
                    handoff_json=str(tmp_path / "handoff.json"),
                    probe_factory="probe:factory",
                    output_json=str(tmp_path / "probe.json"),
                    actions_output_json=str(tmp_path / "actions.json"),
                ),
            ),
        )


def test_benchmark_plan_rejects_generated_artifact_output_path_collisions(tmp_path):
    with pytest.raises(ValueError, match="output paths must be distinct"):
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=dataset_paths(tmp_path),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
            storage_benchmark=StorageBenchmarkPlanConfig(
                workspace_dir="/local_disk0/document-kv-storage-benchmark",
                output_json=str(tmp_path / "storage.json"),
                readers=("memory", "disk", "unity_catalog"),
                uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
            ),
            engine_probes=(
                EngineProbePlanConfig(
                    backend="vllm",
                    handoff_json=str(tmp_path / "vllm-handoff.json"),
                    probe_factory="probe:factory",
                    output_json=str(tmp_path / "release-evidence.json"),
                ),
                EngineProbePlanConfig(
                    backend="sglang",
                    handoff_json=str(tmp_path / "sglang-handoff.json"),
                    probe_factory="probe:factory",
                    output_json=str(tmp_path / "sglang-probe.json"),
                ),
            ),
            release_evidence=ReleaseEvidencePlanConfig(
                output_json=str(tmp_path / "release-evidence.json"),
            ),
        )


def test_benchmark_plan_rejects_release_bundle_output_path_collisions(tmp_path):
    with pytest.raises(ValueError, match="output paths must be distinct"):
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=dataset_paths(tmp_path),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
            storage_benchmark=StorageBenchmarkPlanConfig(
                workspace_dir="/local_disk0/document-kv-storage-benchmark",
                output_json=str(tmp_path / "storage.json"),
                readers=("memory", "disk", "unity_catalog"),
                uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
            ),
            release_evidence=ReleaseEvidencePlanConfig(
                output_json=str(tmp_path / "release-evidence.json"),
                engine_probe_jsons=(
                    str(tmp_path / "vllm-probe.json"),
                    str(tmp_path / "sglang-probe.json"),
                ),
                engine_actions_jsons=release_action_jsons(tmp_path),
            ),
            release_bundle=ReleaseBundlePlanConfig(
                output_dir=str(tmp_path / "release-bundle"),
                output_json=str(tmp_path / "release-evidence.json"),
            ),
        )


def test_benchmark_plan_rejects_release_bundle_output_dir_colliding_with_generated_file(tmp_path):
    with pytest.raises(ValueError, match="output paths must be distinct"):
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=dataset_paths(tmp_path),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
            storage_benchmark=StorageBenchmarkPlanConfig(
                workspace_dir="/local_disk0/document-kv-storage-benchmark",
                output_json=str(tmp_path / "storage.json"),
                readers=("memory", "disk", "unity_catalog"),
                uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
            ),
            release_evidence=ReleaseEvidencePlanConfig(
                output_json=str(tmp_path / "release-evidence.json"),
                engine_probe_jsons=(
                    str(tmp_path / "vllm-probe.json"),
                    str(tmp_path / "sglang-probe.json"),
                ),
                engine_actions_jsons=release_action_jsons(tmp_path),
            ),
            release_bundle=ReleaseBundlePlanConfig(
                output_dir=str(tmp_path / "release-evidence.json"),
                output_json=str(tmp_path / "release-bundle-manifest.json"),
            ),
        )


def test_benchmark_plan_rejects_native_probe_factories_output_path_collisions(tmp_path):
    with pytest.raises(ValueError, match="output paths must be distinct"):
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=dataset_paths(tmp_path),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
            storage_benchmark=StorageBenchmarkPlanConfig(
                workspace_dir="/local_disk0/document-kv-storage-benchmark",
                output_json=str(tmp_path / "storage.json"),
            ),
            native_probe_factories_output_json=str(tmp_path / "storage.json"),
        )


def test_build_v1_benchmark_plan_can_generate_and_bundle_release_preflight(tmp_path):
    release_preflight_json = str(tmp_path / "release-inputs.json")
    config = BenchmarkPlanConfig(
        suite_id="v1-g5",
        dataset_paths=dataset_paths(tmp_path),
        base_url="http://localhost:8000",
        benchmark_output_json=str(tmp_path / "results.json"),
        storage_benchmark=StorageBenchmarkPlanConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json=str(tmp_path / "storage.json"),
            readers=("memory", "disk", "unity_catalog"),
            uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
        ),
        release_evidence=ReleaseEvidencePlanConfig(
            output_json=str(tmp_path / "release-evidence.json"),
            engine_probe_jsons=(
                str(tmp_path / "vllm-probe.json"),
                str(tmp_path / "sglang-probe.json"),
            ),
            engine_actions_jsons=release_action_jsons(tmp_path),
        ),
        release_bundle=ReleaseBundlePlanConfig(
            output_dir=str(tmp_path / "release-bundle"),
            output_json=str(tmp_path / "release-bundle-manifest.json"),
        ),
        release_preflight_output_json=release_preflight_json,
    )

    plan = build_v1_benchmark_plan(config)
    record = benchmark_job_plan_to_record(plan)
    preflight_command = next(command for command in plan.commands if command.name == "preflight-release-evidence")
    bundle_command = plan.post_benchmark_commands[-1]

    assert [command.name for command in plan.commands][-5:] == [
        "run-benchmark",
        "run-storage-benchmark",
        "preflight-release-evidence",
        "validate-release-evidence",
        "build-release-bundle",
    ]
    assert record["release_preflight_output_json"] == release_preflight_json
    assert record["release_bundle"]["preflight_json"] == release_preflight_json
    assert preflight_command.argv[:3] == ("python", "-m", "document_kv_cache.release_evidence")
    assert "--preflight-only" in preflight_command.argv
    assert preflight_command.argv[preflight_command.argv.index("--preflight-output-json") + 1] == (
        release_preflight_json
    )
    assert "--output-json" not in preflight_command.argv
    assert preflight_command.argv.count("--engine-probe-json") == 2
    assert preflight_command.argv.count("--engine-actions-json") == 2
    assert bundle_command.argv.count("--preflight-json") == 1
    assert bundle_command.argv[bundle_command.argv.index("--preflight-json") + 1] == release_preflight_json


def test_build_v1_benchmark_plan_allows_equivalent_generated_release_preflight_bundle_path(tmp_path):
    release_preflight_json = str(tmp_path / "release-inputs.json")
    equivalent_release_preflight_json = str(tmp_path / "subdir" / ".." / "release-inputs.json")
    config = BenchmarkPlanConfig(
        suite_id="v1-g5",
        dataset_paths=dataset_paths(tmp_path),
        base_url="http://localhost:8000",
        benchmark_output_json=str(tmp_path / "results.json"),
        storage_benchmark=StorageBenchmarkPlanConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json=str(tmp_path / "storage.json"),
            readers=("memory", "disk", "unity_catalog"),
            uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
        ),
        release_evidence=ReleaseEvidencePlanConfig(
            output_json=str(tmp_path / "release-evidence.json"),
            engine_probe_jsons=(
                str(tmp_path / "vllm-probe.json"),
                str(tmp_path / "sglang-probe.json"),
            ),
            engine_actions_jsons=release_action_jsons(tmp_path),
        ),
        release_bundle=ReleaseBundlePlanConfig(
            output_dir=str(tmp_path / "release-bundle"),
            output_json=str(tmp_path / "release-bundle-manifest.json"),
            preflight_json=equivalent_release_preflight_json,
        ),
        release_preflight_output_json=release_preflight_json,
    )

    plan = build_v1_benchmark_plan(config)
    record = benchmark_job_plan_to_record(plan)
    bundle_command = plan.post_benchmark_commands[-1]

    assert record["release_bundle"]["preflight_json"] == release_preflight_json
    assert bundle_command.argv.count("--preflight-json") == 1
    assert bundle_command.argv[bundle_command.argv.index("--preflight-json") + 1] == release_preflight_json


def test_benchmark_plan_rejects_conflicting_generated_release_preflight_bundle_path(tmp_path):
    with pytest.raises(
        ValueError,
        match="release bundle preflight_json must match release_preflight_output_json",
    ):
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=dataset_paths(tmp_path),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
            storage_benchmark=StorageBenchmarkPlanConfig(
                workspace_dir="/local_disk0/document-kv-storage-benchmark",
                output_json=str(tmp_path / "storage.json"),
                readers=("memory", "disk", "unity_catalog"),
                uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
            ),
            release_evidence=ReleaseEvidencePlanConfig(
                output_json=str(tmp_path / "release-evidence.json"),
                engine_probe_jsons=(
                    str(tmp_path / "vllm-probe.json"),
                    str(tmp_path / "sglang-probe.json"),
                ),
                engine_actions_jsons=release_action_jsons(tmp_path),
            ),
            release_bundle=ReleaseBundlePlanConfig(
                output_dir=str(tmp_path / "release-bundle"),
                output_json=str(tmp_path / "release-bundle-manifest.json"),
                preflight_json=str(tmp_path / "other-release-inputs.json"),
            ),
            release_preflight_output_json=str(tmp_path / "release-inputs.json"),
        )


def test_benchmark_plan_release_preflight_requires_release_evidence(tmp_path):
    with pytest.raises(ValueError, match="release_preflight_output_json requires release_evidence"):
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=dataset_paths(tmp_path),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
            release_preflight_output_json=str(tmp_path / "release-inputs.json"),
        )


def test_benchmark_plan_rejects_release_preflight_output_path_collisions(tmp_path):
    with pytest.raises(ValueError, match="output paths must be distinct"):
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=dataset_paths(tmp_path),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
            storage_benchmark=StorageBenchmarkPlanConfig(
                workspace_dir="/local_disk0/document-kv-storage-benchmark",
                output_json=str(tmp_path / "storage.json"),
            ),
            release_evidence=ReleaseEvidencePlanConfig(
                output_json=str(tmp_path / "release-evidence.json"),
                engine_probe_jsons=(
                    str(tmp_path / "vllm-probe.json"),
                    str(tmp_path / "sglang-probe.json"),
                ),
                engine_actions_jsons=release_action_jsons(tmp_path),
            ),
            release_preflight_output_json=str(tmp_path / "storage.json"),
        )


def test_benchmark_plan_rejects_repository_hygiene_output_path_collisions(tmp_path):
    with pytest.raises(ValueError, match="output paths must be distinct"):
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=dataset_paths(tmp_path),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
            storage_benchmark=StorageBenchmarkPlanConfig(
                workspace_dir="/local_disk0/document-kv-storage-benchmark",
                output_json=str(tmp_path / "storage.json"),
            ),
            repository_hygiene_output_json=str(tmp_path / "storage.json"),
        )


def test_benchmark_plan_rejects_github_governance_output_path_collisions(tmp_path):
    with pytest.raises(ValueError, match="output paths must be distinct"):
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=dataset_paths(tmp_path),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
            storage_benchmark=StorageBenchmarkPlanConfig(
                workspace_dir="/local_disk0/document-kv-storage-benchmark",
                output_json=str(tmp_path / "storage.json"),
            ),
            github_governance_output_json=str(tmp_path / "storage.json"),
        )


def test_benchmark_plan_rejects_equivalent_generated_artifact_output_paths(tmp_path):
    with pytest.raises(ValueError, match="output paths must be distinct"):
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=dataset_paths(tmp_path),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
            storage_benchmark=StorageBenchmarkPlanConfig(
                workspace_dir="/local_disk0/document-kv-storage-benchmark",
                output_json=str(tmp_path / "subdir" / ".." / "results.json"),
            ),
        )


def test_benchmark_plan_rejects_local_uri_alias_generated_artifact_output_paths(tmp_path):
    with pytest.raises(ValueError, match="output paths must be distinct"):
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=dataset_paths(tmp_path),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
            storage_benchmark=StorageBenchmarkPlanConfig(
                workspace_dir="/local_disk0/document-kv-storage-benchmark",
                output_json=f"disk:{tmp_path / 'results.json'}",
            ),
        )


def test_engine_probe_plan_rejects_empty_metadata_keys(tmp_path):
    with pytest.raises(ValueError, match="metadata entries"):
        EngineProbePlanConfig(
            backend="vllm",
            handoff_json=str(tmp_path / "handoff.json"),
            probe_factory="probe:factory",
            output_json=str(tmp_path / "probe.json"),
            metadata=("=value",),
        )


def test_benchmark_plan_requires_release_storage_readers_for_planned_release_evidence(tmp_path):
    with pytest.raises(ValueError, match="release readers"):
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=dataset_paths(tmp_path),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
            storage_benchmark=StorageBenchmarkPlanConfig(
                workspace_dir="/local_disk0/document-kv-storage-benchmark",
                output_json=str(tmp_path / "storage.json"),
                readers=("disk",),
            ),
            release_evidence=ReleaseEvidencePlanConfig(
                output_json=str(tmp_path / "release-evidence.json"),
                engine_probe_jsons=(
                    str(tmp_path / "vllm-probe.json"),
                    str(tmp_path / "sglang-probe.json"),
                ),
                engine_actions_jsons=release_action_jsons(tmp_path),
            ),
        )


def test_benchmark_plan_accepts_release_storage_readers_in_any_order(tmp_path):
    config = BenchmarkPlanConfig(
        suite_id="v1-g5",
        dataset_paths=dataset_paths(tmp_path),
        base_url="http://localhost:8000",
        benchmark_output_json=str(tmp_path / "results.json"),
        storage_benchmark=StorageBenchmarkPlanConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json=str(tmp_path / "storage.json"),
            readers=("disk", "memory", "unity_catalog"),
            uc_volume_root="/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
        ),
        release_evidence=ReleaseEvidencePlanConfig(
            output_json=str(tmp_path / "release-evidence.json"),
            engine_probe_jsons=(
                str(tmp_path / "vllm-probe.json"),
                str(tmp_path / "sglang-probe.json"),
            ),
            engine_actions_jsons=release_action_jsons(tmp_path),
        ),
    )

    plan = build_v1_benchmark_plan(config)

    assert plan.post_benchmark_commands[-1].name == "validate-release-evidence"


def test_storage_benchmark_plan_config_validates_reader_selection(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        StorageBenchmarkPlanConfig(
            workspace_dir=str(tmp_path / "workspace"),
            output_json=str(tmp_path / "storage.json"),
            readers=("object-store",),
        )

    with pytest.raises(ValueError, match="uc_volume_root"):
        StorageBenchmarkPlanConfig(
            workspace_dir=str(tmp_path / "workspace"),
            output_json=str(tmp_path / "storage.json"),
            readers=("unity_catalog",),
        )

    with pytest.raises(ValueError, match="uc_volume_root"):
        StorageBenchmarkPlanConfig(
            workspace_dir=str(tmp_path / "workspace"),
            output_json=str(tmp_path / "storage.json"),
            readers=("unity_catalog",),
            uc_volume_root="",
        )


def test_storage_benchmark_plan_defaults_to_memory_and_disk_without_uc_root(tmp_path):
    config = BenchmarkPlanConfig(
        suite_id="v1-g5",
        dataset_paths=dataset_paths(tmp_path),
        base_url="http://localhost:8000",
        benchmark_output_json=str(tmp_path / "results.json"),
        storage_benchmark=StorageBenchmarkPlanConfig(
            workspace_dir="/local_disk0/document-kv-storage-benchmark",
            output_json=str(tmp_path / "storage.json"),
        ),
    )

    storage_argv = build_v1_benchmark_plan(config).post_benchmark_commands[0].argv

    assert storage_argv.count("--reader") == 2
    assert "memory" in storage_argv
    assert "disk" in storage_argv
    assert "unity_catalog" not in storage_argv


def test_benchmark_plan_requires_all_v1_datasets_by_default(tmp_path):
    with pytest.raises(ValueError, match="missing required V1 datasets"):
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=(
                BenchmarkDatasetPath(
                    dataset="biography",
                    raw_jsonl=str(tmp_path / "raw.jsonl"),
                    prepared_jsonl=str(tmp_path / "prepared.jsonl"),
                ),
            ),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
        )


def test_benchmark_plan_allows_partial_smoke_when_requested(tmp_path):
    config = BenchmarkPlanConfig(
        suite_id="smoke",
        dataset_paths=(
            BenchmarkDatasetPath(
                dataset="biography",
                raw_jsonl=str(tmp_path / "raw.jsonl"),
                prepared_jsonl=str(tmp_path / "prepared.jsonl"),
            ),
        ),
        base_url="http://localhost:8000",
        benchmark_output_json=str(tmp_path / "results.json"),
        require_all_v1_datasets=False,
    )

    plan = build_v1_benchmark_plan(config)

    assert len(plan.preparation_commands) == 1
    assert "biography=" + str(tmp_path / "prepared.jsonl") in plan.benchmark_command.argv


def test_write_benchmark_job_plan_shell_emits_executable_script(tmp_path):
    plan = build_v1_benchmark_plan(
        BenchmarkPlanConfig(
            suite_id="v1-g5",
            dataset_paths=dataset_paths(tmp_path),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
        )
    )
    path = tmp_path / "run.sh"

    write_benchmark_job_plan_shell(plan, path)

    content = path.read_text(encoding="utf-8")
    assert content.startswith("#!/usr/bin/env bash\nset -euo pipefail")
    assert "document_kv_cache.dataset_prep" in content
    assert "document_kv_cache.benchmark_runner" in content
    assert path.stat().st_mode & 0o111


def test_main_prints_plan_json_for_full_dataset_set(capsys, tmp_path):
    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--raw-dataset",
            f"hotpotqa={tmp_path / 'raw' / 'hotpotqa.jsonl'}",
            "--raw-dataset",
            f"musique={tmp_path / 'raw' / 'musique.jsonl'}",
            "--raw-dataset",
            f"niah={tmp_path / 'raw' / 'niah.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--benchmark-output-json",
            str(tmp_path / "results.json"),
        ]
    )

    assert exit_code == 0
    record = json.loads(capsys.readouterr().out)
    assert record["suite_id"] == "v1-openai-compatible"
    assert len(record["datasets"]) == 4
    assert record["commands"][0]["argv"][0] == sys.executable
    assert record["commands"][-1]["name"] == "run-benchmark"


def test_public_benchmark_plan_main_respects_document_namespace_monkeypatch(monkeypatch, capsys, tmp_path):
    original_legacy_build = legacy_benchmark_plan.build_v1_benchmark_plan

    def fake_build(config):
        assert config.suite_id == "v1-openai-compatible"
        return "public-hook-plan"

    def fake_record(plan):
        assert plan == "public-hook-plan"
        return {"ok": True, "source": "public-hook"}

    monkeypatch.setattr(public_benchmark_plan, "build_v1_benchmark_plan", fake_build)
    monkeypatch.setattr(public_benchmark_plan, "benchmark_job_plan_to_record", fake_record)

    exit_code = public_benchmark_plan.main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "source": "public-hook"}
    assert legacy_benchmark_plan.build_v1_benchmark_plan is original_legacy_build


def test_legacy_benchmark_plan_main_respects_legacy_namespace_monkeypatch(monkeypatch, capsys, tmp_path):
    original_public_build = public_benchmark_plan.build_v1_benchmark_plan

    def fake_build(config):
        assert config.suite_id == "v1-openai-compatible"
        return "legacy-hook-plan"

    def fake_record(plan):
        assert plan == "legacy-hook-plan"
        return {"ok": True, "source": "legacy-hook"}

    monkeypatch.setattr(legacy_benchmark_plan, "build_v1_benchmark_plan", fake_build)
    monkeypatch.setattr(legacy_benchmark_plan, "benchmark_job_plan_to_record", fake_record)

    exit_code = legacy_benchmark_plan.main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "source": "legacy-hook"}
    assert public_benchmark_plan.build_v1_benchmark_plan is original_public_build


def test_legacy_benchmark_plan_output_json_respects_legacy_writer_hook(monkeypatch, tmp_path):
    plan_json = tmp_path / "plan.json"

    def fake_build(config):
        return "legacy-hook-plan"

    def fake_write(plan, path):
        assert plan == "legacy-hook-plan"
        assert path == str(plan_json)
        plan_json.write_text(json.dumps({"source": "legacy-writer-hook"}) + "\n", encoding="utf-8")

    monkeypatch.setattr(legacy_benchmark_plan, "build_v1_benchmark_plan", fake_build)
    monkeypatch.setattr(legacy_benchmark_plan, "write_benchmark_job_plan_json", fake_write)

    exit_code = legacy_benchmark_plan.main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--plan-output-json",
            str(plan_json),
        ]
    )

    assert exit_code == 0
    assert json.loads(plan_json.read_text(encoding="utf-8")) == {"source": "legacy-writer-hook"}


def test_legacy_benchmark_plan_main_isolates_dataclass_method_globals(monkeypatch, tmp_path):
    plan_json = tmp_path / "plan.json"
    original_public_datasets = public_benchmark_plan.SUPPORTED_V1_DATASETS

    def validate_custom_dataset(dataset):
        if dataset != "custom":
            raise ValueError("expected custom dataset")

    monkeypatch.setattr(legacy_benchmark_plan, "SUPPORTED_V1_DATASETS", ("custom",))
    monkeypatch.setattr(legacy_benchmark_plan, "validate_v1_dataset", validate_custom_dataset)

    exit_code = legacy_benchmark_plan.main(
        [
            "--raw-dataset",
            f"custom={tmp_path / 'raw' / 'custom.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--plan-output-json",
            str(plan_json),
        ]
    )

    record = json.loads(plan_json.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert record["datasets"] == [
        {
            "dataset": "custom",
            "raw_jsonl": str(tmp_path / "raw" / "custom.jsonl"),
            "prepared_jsonl": str(tmp_path / "prepared" / "custom.jsonl"),
        }
    ]
    assert public_benchmark_plan.SUPPORTED_V1_DATASETS is original_public_datasets


def test_legacy_benchmark_plan_without_patches_returns_public_dataclass_instances(tmp_path):
    config = BenchmarkPlanConfig(
        suite_id="v1-g5",
        dataset_paths=dataset_paths(tmp_path),
        base_url="http://localhost:8000",
        benchmark_output_json=str(tmp_path / "results.json"),
    )

    plan = legacy_benchmark_plan.build_v1_benchmark_plan(config)
    repeated_plan = legacy_benchmark_plan.build_v1_benchmark_plan(config)

    assert type(plan) is legacy_benchmark_plan.BenchmarkJobPlan
    assert type(plan) is public_benchmark_plan.BenchmarkJobPlan
    assert plan == repeated_plan
    assert pickle.loads(pickle.dumps(plan)) == plan


def test_legacy_benchmark_plan_saved_original_wrapper_does_not_recurse(monkeypatch, tmp_path):
    config = BenchmarkPlanConfig(
        suite_id="v1-g5",
        dataset_paths=dataset_paths(tmp_path),
        base_url="http://localhost:8000",
        benchmark_output_json=str(tmp_path / "results.json"),
    )
    original_build = legacy_benchmark_plan.build_v1_benchmark_plan
    wrapped_calls = 0

    def wrapped_build(config):
        nonlocal wrapped_calls
        wrapped_calls += 1
        return original_build(config)

    monkeypatch.setattr(legacy_benchmark_plan, "build_v1_benchmark_plan", wrapped_build)

    direct_plan = original_build(config)
    assert type(direct_plan) is public_benchmark_plan.BenchmarkJobPlan
    assert wrapped_calls == 0

    exit_code = legacy_benchmark_plan.main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--plan-output-json",
            str(tmp_path / "plan.json"),
        ]
    )
    assert exit_code == 0
    assert wrapped_calls == 1


def test_legacy_benchmark_plan_import_order_does_not_capture_public_monkeypatch():
    script = """
import json
from pathlib import Path
import tempfile

import document_kv_cache.benchmark_plan as public_benchmark_plan

def public_validate_should_not_run(dataset):
    raise AssertionError(f"legacy imported patched public validator for {dataset}")

class FakeBenchmarkPlanConfig:
    def __init__(self, *args, **kwargs):
        raise AssertionError("legacy imported patched public config class")

public_benchmark_plan.SUPPORTED_V1_DATASETS = ("custom",)
public_benchmark_plan.validate_v1_dataset = public_validate_should_not_run
public_benchmark_plan.BenchmarkPlanConfig = FakeBenchmarkPlanConfig

import restaurant_kv_serving.benchmark_plan as legacy_benchmark_plan

assert legacy_benchmark_plan.BenchmarkPlanConfig is not FakeBenchmarkPlanConfig
assert legacy_benchmark_plan.SUPPORTED_V1_DATASETS == ("biography", "hotpotqa", "musique", "niah")
dataset_path = legacy_benchmark_plan.BenchmarkDatasetPath(
    dataset="biography",
    raw_jsonl="raw.jsonl",
    prepared_jsonl="prepared.jsonl",
)
assert dataset_path.dataset == "biography"

try:
    public_benchmark_plan.BenchmarkDatasetPath(
        dataset="biography",
        raw_jsonl="raw.jsonl",
        prepared_jsonl="prepared.jsonl",
    )
except AssertionError:
    pass
else:
    raise AssertionError("public monkeypatch was not installed")

with tempfile.TemporaryDirectory() as raw_tmp:
    tmp_path = Path(raw_tmp)
    plan_json = tmp_path / "plan.json"
    exit_code = legacy_benchmark_plan.main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--plan-output-json",
            str(plan_json),
        ]
    )
    assert exit_code == 0
    record = json.loads(plan_json.read_text(encoding="utf-8"))
    assert record["datasets"][0]["dataset"] == "biography"
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_benchmark_plan_document_module_owns_public_api():
    assert public_benchmark_plan.__all__ == [
        "PLAN_VERSION",
        "ENGINE_PROBE_TARGETS_RECORD_TYPE",
        "ENGINE_PROBE_TARGETS_SCHEMA_VERSION",
        "BenchmarkDatasetPath",
        "BenchmarkCommand",
        "StorageBenchmarkPlanConfig",
        "EngineProbePlanConfig",
        "ReleaseEvidencePlanConfig",
        "ReleaseBundlePlanConfig",
        "BenchmarkPlanConfig",
        "BenchmarkJobPlan",
        "build_v1_benchmark_plan",
        "benchmark_job_plan_to_record",
        "engine_probe_targets_to_record",
        "write_benchmark_job_plan_json",
        "write_benchmark_job_plan_shell",
        "write_engine_probe_targets_json",
        "main",
    ]
    assert public_benchmark_plan.BenchmarkPlanConfig.__module__ == "document_kv_cache.benchmark_plan"
    assert public_benchmark_plan.build_v1_benchmark_plan.__module__ == "document_kv_cache.benchmark_plan"
    assert public_benchmark_plan.main.__module__ == "document_kv_cache.benchmark_plan"
    assert not hasattr(legacy_benchmark_plan, "__all__")
    assert legacy_benchmark_plan.BenchmarkPlanConfig is public_benchmark_plan.BenchmarkPlanConfig
    assert legacy_benchmark_plan.build_v1_benchmark_plan.__module__ == "restaurant_kv_serving.benchmark_plan"
    assert legacy_benchmark_plan.main.__module__ == "restaurant_kv_serving.benchmark_plan"


def test_main_writes_json_and_shell_outputs(tmp_path):
    plan_json = tmp_path / "plan.json"
    plan_sh = tmp_path / "plan.sh"

    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--plan-output-json",
            str(plan_json),
            "--plan-output-sh",
            str(plan_sh),
        ]
    )

    assert exit_code == 0
    assert json.loads(plan_json.read_text(encoding="utf-8"))["datasets"][0]["dataset"] == "biography"
    assert "document_kv_cache.benchmark_runner" in plan_sh.read_text(encoding="utf-8")


def test_main_can_include_storage_benchmark_command(tmp_path):
    plan_json = tmp_path / "plan.json"

    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--storage-benchmark-workspace-dir",
            "/local_disk0/document-kv-storage-benchmark",
            "--storage-benchmark-output-json",
            str(tmp_path / "storage.json"),
            "--storage-benchmark-reader",
            "disk",
            "--plan-output-json",
            str(plan_json),
        ]
    )

    record = json.loads(plan_json.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert record["commands"][-1]["name"] == "run-storage-benchmark"
    assert record["storage_benchmark_output_json"] == str(tmp_path / "storage.json")
    assert record["commands"][-1]["argv"].count("--reader") == 1
    assert "disk" in record["commands"][-1]["argv"]


def test_main_can_include_release_evidence_validation_command(tmp_path):
    plan_json = tmp_path / "plan.json"

    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--storage-benchmark-workspace-dir",
            "/local_disk0/document-kv-storage-benchmark",
            "--storage-benchmark-output-json",
            str(tmp_path / "storage.json"),
            "--storage-benchmark-uc-volume-root",
            "/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
            "--release-evidence-output-json",
            str(tmp_path / "release-evidence.json"),
            "--release-engine-probe-json",
            str(tmp_path / "vllm-probe.json"),
            "--release-engine-probe-json",
            str(tmp_path / "sglang-probe.json"),
            "--release-engine-actions-json",
            str(tmp_path / "vllm-actions.json"),
            "--release-engine-actions-json",
            str(tmp_path / "sglang-actions.json"),
            "--plan-output-json",
            str(plan_json),
        ]
    )

    record = json.loads(plan_json.read_text(encoding="utf-8"))
    release_argv = record["commands"][-1]["argv"]

    assert exit_code == 0
    assert record["commands"][-1]["name"] == "validate-release-evidence"
    assert record["release_evidence_output_json"] == str(tmp_path / "release-evidence.json")
    assert record["release_storage_benchmark_json"] == str(tmp_path / "storage.json")
    assert record["release_engine_probe_jsons"] == [
        str(tmp_path / "vllm-probe.json"),
        str(tmp_path / "sglang-probe.json"),
    ]
    assert record["release_engine_actions_jsons"] == [
        str(tmp_path / "vllm-actions.json"),
        str(tmp_path / "sglang-actions.json"),
    ]
    assert release_argv[release_argv.index("--storage-benchmark-json") + 1] == str(tmp_path / "storage.json")
    assert release_argv.count("--engine-probe-json") == 2
    assert release_argv.count("--engine-actions-json") == 2


def test_main_can_include_planned_engine_probes_and_release_evidence_validation(tmp_path):
    plan_json = tmp_path / "plan.json"
    targets_json = tmp_path / "engine-probe-targets.json"

    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--storage-benchmark-workspace-dir",
            "/local_disk0/document-kv-storage-benchmark",
            "--storage-benchmark-output-json",
            str(tmp_path / "storage.json"),
            "--storage-benchmark-uc-volume-root",
            "/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
            "--engine-probe-handoff-json",
            f"vllm={tmp_path / 'vllm-handoff.json'}",
            "--engine-probe-factory",
            "vllm=vllm_probe:factory",
            "--engine-probe-output-json",
            f"vllm={tmp_path / 'vllm-probe.json'}",
            "--engine-probe-actions-output-json",
            f"vllm={tmp_path / 'vllm-actions.json'}",
            "--engine-probe-payload-uri",
            f"vllm=disk:{tmp_path / 'vllm.kv'}",
            "--engine-probe-metadata",
            "vllm=probe.source=cli",
            "--engine-probe-handoff-json",
            f"sglang={tmp_path / 'sglang-handoff.json'}",
            "--engine-probe-factory",
            "sglang=sglang_probe:factory",
            "--engine-probe-output-json",
            f"sglang={tmp_path / 'sglang-probe.json'}",
            "--engine-probe-actions-output-json",
            f"sglang={tmp_path / 'sglang-actions.json'}",
            "--release-evidence-output-json",
            str(tmp_path / "release-evidence.json"),
            "--plan-output-json",
            str(plan_json),
            "--engine-probe-targets-output-json",
            str(targets_json),
            "--engine-probe-targets-release-safe",
        ]
    )

    record = json.loads(plan_json.read_text(encoding="utf-8"))
    targets_record = json.loads(targets_json.read_text(encoding="utf-8"))
    command_names = [command["name"] for command in record["commands"]]
    vllm_argv = record["commands"][command_names.index("run-vllm-engine-probe")]["argv"]
    release_argv = record["commands"][-1]["argv"]

    assert exit_code == 0
    assert command_names[-4:] == [
        "run-storage-benchmark",
        "run-sglang-engine-probe",
        "run-vllm-engine-probe",
        "validate-release-evidence",
    ]
    assert record["release_engine_probe_jsons"] == [
        str(tmp_path / "sglang-probe.json"),
        str(tmp_path / "vllm-probe.json"),
    ]
    assert record["release_engine_actions_jsons"] == [
        str(tmp_path / "sglang-actions.json"),
        str(tmp_path / "vllm-actions.json"),
    ]
    assert record["planned_engine_probes"][0]["backend"] == "sglang"
    assert record["planned_engine_probes"][0]["actions_output_json"] == str(tmp_path / "sglang-actions.json")
    assert record["planned_engine_probes"][1]["metadata"] == ["probe.source=cli"]
    assert vllm_argv[vllm_argv.index("--expected-backend") + 1] == "vllm"
    assert vllm_argv[vllm_argv.index("--payload-uri") + 1] == f"disk:{tmp_path / 'vllm.kv'}"
    assert vllm_argv[vllm_argv.index("--actions-output-json") + 1] == str(tmp_path / "vllm-actions.json")
    assert release_argv.count("--engine-probe-json") == 2
    assert release_argv.count("--engine-actions-json") == 2
    assert str(tmp_path / "vllm-probe.json") in release_argv
    assert str(tmp_path / "sglang-probe.json") in release_argv
    assert str(tmp_path / "vllm-actions.json") in release_argv
    assert str(tmp_path / "sglang-actions.json") in release_argv
    assert targets_record["record_type"] == ENGINE_PROBE_TARGETS_RECORD_TYPE
    assert targets_record["release_safe"] is True
    assert [probe["backend"] for probe in targets_record["probes"]] == ["sglang", "vllm"]
    assert [probe["actions_output_json"] for probe in targets_record["probes"]] == [
        str(tmp_path / "sglang-actions.json"),
        str(tmp_path / "vllm-actions.json"),
    ]
    assert targets_record["probes"][1]["metadata"] == ["probe.source=cli"]
    assert read_databricks_engine_probe_targets_json(targets_json)[1].metadata == ("probe.source=cli",)


def test_main_can_fill_builtin_engine_probe_factories_for_planned_probes(tmp_path):
    plan_json = tmp_path / "plan.json"
    targets_json = tmp_path / "engine-probe-targets.json"

    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--storage-benchmark-workspace-dir",
            "/local_disk0/document-kv-storage-benchmark",
            "--storage-benchmark-output-json",
            str(tmp_path / "storage.json"),
            "--storage-benchmark-uc-volume-root",
            "/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
            "--engine-probe-handoff-json",
            f"vllm={tmp_path / 'vllm-handoff.json'}",
            "--engine-probe-output-json",
            f"vllm={tmp_path / 'vllm-probe.json'}",
            "--engine-probe-actions-output-json",
            f"vllm={tmp_path / 'vllm-actions.json'}",
            "--engine-probe-handoff-json",
            f"sglang={tmp_path / 'sglang-handoff.json'}",
            "--engine-probe-output-json",
            f"sglang={tmp_path / 'sglang-probe.json'}",
            "--engine-probe-actions-output-json",
            f"sglang={tmp_path / 'sglang-actions.json'}",
            "--engine-probe-use-builtin-factories",
            "--release-evidence-output-json",
            str(tmp_path / "release-evidence.json"),
            "--plan-output-json",
            str(plan_json),
            "--engine-probe-targets-output-json",
            str(targets_json),
            "--engine-probe-targets-release-safe",
        ]
    )

    record = json.loads(plan_json.read_text(encoding="utf-8"))
    targets_record = json.loads(targets_json.read_text(encoding="utf-8"))
    factories = {probe["backend"]: probe["probe_factory"] for probe in record["planned_engine_probes"]}

    assert exit_code == 0
    assert factories == {
        "sglang": SGLANG_NATIVE_PROBE_FACTORY,
        "vllm": VLLM_NATIVE_PROBE_FACTORY,
    }
    assert targets_record["release_safe"] is True
    assert {probe["probe_factory"] for probe in targets_record["probes"]} == {
        SGLANG_NATIVE_PROBE_FACTORY,
        VLLM_NATIVE_PROBE_FACTORY,
    }


def test_main_rejects_engine_probe_targets_output_without_planned_probes(capsys, tmp_path):
    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--engine-probe-targets-output-json",
            str(tmp_path / "engine-probe-targets.json"),
        ]
    )

    record = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert record["ok"] is False
    assert "engine_probe_targets require planned engine_probes" in record["error"]


def test_main_rejects_engine_probe_targets_output_path_collision(capsys, tmp_path):
    target_json = tmp_path / "engine-probe-targets.json"
    exit_code = main(
        [
            *planned_release_probe_cli_args(tmp_path),
            "--plan-output-json",
            str(target_json),
            "--engine-probe-targets-output-json",
            str(target_json),
        ]
    )

    record = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert record["ok"] is False
    assert "output paths must be distinct" in record["error"]
    assert "engine_probe_targets_output_json" in record["error"]


def test_main_rejects_release_evidence_from_planned_engine_version_debug_probe(capsys, tmp_path):
    exit_code = main(
        [
            *planned_release_probe_cli_args(tmp_path),
            "--engine-probe-engine-version",
            "vllm=debug-vllm",
            "--plan-output-json",
            str(tmp_path / "plan.json"),
        ]
    )

    record = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert record["ok"] is False
    assert "release_evidence cannot consume planned debug engine probes" in record["error"]
    assert "engine_version overrides" in record["error"]
    assert "vllm" in record["error"]


def test_main_rejects_release_evidence_from_planned_non_native_debug_probe(capsys, tmp_path):
    exit_code = main(
        [
            *planned_release_probe_cli_args(tmp_path),
            "--allow-non-native-engine-probe",
            "sglang",
            "--plan-output-json",
            str(tmp_path / "plan.json"),
        ]
    )

    record = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert record["ok"] is False
    assert "release_evidence cannot consume planned debug engine probes" in record["error"]
    assert "non-native debug probes" in record["error"]
    assert "sglang" in record["error"]


def test_main_allows_debug_planned_engine_probes_when_release_uses_explicit_probe_jsons(tmp_path):
    plan_json = tmp_path / "plan.json"

    exit_code = main(
        [
            *planned_release_probe_cli_args(tmp_path),
            "--engine-probe-engine-version",
            "vllm=debug-vllm",
            "--allow-non-native-engine-probe",
            "sglang",
            "--release-engine-probe-json",
            str(tmp_path / "native-vllm-probe.json"),
            "--release-engine-probe-json",
            str(tmp_path / "native-sglang-probe.json"),
            "--release-engine-actions-json",
            str(tmp_path / "native-vllm-actions.json"),
            "--release-engine-actions-json",
            str(tmp_path / "native-sglang-actions.json"),
            "--plan-output-json",
            str(plan_json),
        ]
    )

    record = json.loads(plan_json.read_text(encoding="utf-8"))
    release_argv = record["commands"][-1]["argv"]

    assert exit_code == 0
    assert record["planned_engine_probes"][1]["engine_version"] == "debug-vllm"
    assert record["planned_engine_probes"][0]["allow_non_native_probe"] is True
    assert str(tmp_path / "native-vllm-probe.json") in release_argv
    assert str(tmp_path / "native-sglang-probe.json") in release_argv
    assert str(tmp_path / "native-vllm-actions.json") in release_argv
    assert str(tmp_path / "native-sglang-actions.json") in release_argv
    assert str(tmp_path / "vllm-probe.json") not in release_argv
    assert str(tmp_path / "sglang-probe.json") not in release_argv
    assert str(tmp_path / "vllm-actions.json") not in release_argv
    assert str(tmp_path / "sglang-actions.json") not in release_argv


def test_main_rejects_explicit_release_probe_json_aliasing_planned_debug_probe(capsys, tmp_path):
    exit_code = main(
        [
            *planned_release_probe_cli_args(tmp_path),
            "--engine-probe-engine-version",
            "vllm=debug-vllm",
            "--release-engine-probe-json",
            str(tmp_path / "native-sglang-probe.json"),
            "--release-engine-probe-json",
            str(tmp_path / "vllm-probe.json"),
            "--release-engine-actions-json",
            str(tmp_path / "native-vllm-actions.json"),
            "--release-engine-actions-json",
            str(tmp_path / "native-sglang-actions.json"),
            "--plan-output-json",
            str(tmp_path / "plan.json"),
        ]
    )

    record = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert record["ok"] is False
    assert "explicit engine_probe_jsons" in record["error"]
    assert "planned debug engine probe outputs" in record["error"]
    assert "vllm=" in record["error"]


def test_main_allows_explicit_release_probe_json_aliasing_planned_native_probe(tmp_path):
    plan_json = tmp_path / "plan.json"

    exit_code = main(
        [
            *planned_release_probe_cli_args(tmp_path),
            "--release-engine-probe-json",
            str(tmp_path / "sglang-probe.json"),
            "--release-engine-probe-json",
            str(tmp_path / "vllm-probe.json"),
            "--release-engine-actions-json",
            str(tmp_path / "sglang-actions.json"),
            "--release-engine-actions-json",
            str(tmp_path / "vllm-actions.json"),
            "--plan-output-json",
            str(plan_json),
        ]
    )

    record = json.loads(plan_json.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert record["release_engine_probe_jsons"] == [
        str(tmp_path / "sglang-probe.json"),
        str(tmp_path / "vllm-probe.json"),
    ]
    assert record["release_engine_actions_jsons"] == [
        str(tmp_path / "sglang-actions.json"),
        str(tmp_path / "vllm-actions.json"),
    ]


def test_main_can_include_release_bundle_command(tmp_path):
    plan_json = tmp_path / "plan.json"

    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--storage-benchmark-workspace-dir",
            "/local_disk0/document-kv-storage-benchmark",
            "--storage-benchmark-output-json",
            str(tmp_path / "storage.json"),
            "--storage-benchmark-uc-volume-root",
            "/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
            "--release-evidence-output-json",
            str(tmp_path / "release-evidence.json"),
            "--release-preflight-output-json",
            str(tmp_path / "release-inputs.json"),
            "--release-engine-probe-json",
            str(tmp_path / "vllm-probe.json"),
            "--release-engine-probe-json",
            str(tmp_path / "sglang-probe.json"),
            "--release-engine-actions-json",
            str(tmp_path / "vllm-actions.json"),
            "--release-engine-actions-json",
            str(tmp_path / "sglang-actions.json"),
            "--release-bundle-output-dir",
            str(tmp_path / "release-bundle"),
            "--release-bundle-output-json",
            str(tmp_path / "release-bundle-manifest.json"),
            "--release-bundle-preflight-json",
            str(tmp_path / "release-inputs.json"),
            "--release-bundle-plan-execution-json",
            str(tmp_path / "plan-execution.json"),
            "--release-bundle-databricks-run-status-json",
            str(tmp_path / "databricks-run-status.json"),
            "--release-bundle-package-wheel",
            str(tmp_path / "dist" / "document_kv_cache-0.2.0-py3-none-any.whl"),
            "--release-bundle-pr-evidence-json",
            str(tmp_path / "pr-evidence.json"),
            "--release-bundle-github-governance-json",
            str(tmp_path / "github-governance.json"),
            "--github-governance-output-json",
            str(tmp_path / "github-governance.json"),
            "--release-bundle-repository-hygiene-json",
            str(tmp_path / "repository-hygiene.json"),
            "--repository-hygiene-output-json",
            str(tmp_path / "repository-hygiene.json"),
            "--release-bundle-native-probe-factories-json",
            str(tmp_path / "native-probe-factories.json"),
            "--native-probe-factories-output-json",
            str(tmp_path / "native-probe-factories.json"),
            "--release-bundle-require-complete-v1",
            "--release-bundle-overwrite",
            "--plan-output-json",
            str(plan_json),
        ]
    )

    record = json.loads(plan_json.read_text(encoding="utf-8"))
    bundle_argv = record["commands"][-1]["argv"]

    assert exit_code == 0
    assert [command["name"] for command in record["commands"]][-7:] == [
        "run-storage-benchmark",
        "inspect-github-governance",
        "inspect-repository-hygiene",
        "inspect-native-probe-factories",
        "preflight-release-evidence",
        "validate-release-evidence",
        "build-release-bundle",
    ]
    governance_argv = next(
        command["argv"] for command in record["commands"] if command["name"] == "inspect-github-governance"
    )
    hygiene_argv = next(
        command["argv"] for command in record["commands"] if command["name"] == "inspect-repository-hygiene"
    )
    native_argv = next(
        command["argv"] for command in record["commands"] if command["name"] == "inspect-native-probe-factories"
    )
    preflight_argv = next(
        command["argv"] for command in record["commands"] if command["name"] == "preflight-release-evidence"
    )
    assert record["release_bundle_output_dir"] == str(tmp_path / "release-bundle")
    assert record["github_governance_output_json"] == str(tmp_path / "github-governance.json")
    assert record["repository_hygiene_output_json"] == str(tmp_path / "repository-hygiene.json")
    assert record["native_probe_factories_output_json"] == str(tmp_path / "native-probe-factories.json")
    assert record["release_preflight_output_json"] == str(tmp_path / "release-inputs.json")
    assert record["release_bundle"]["release_evidence_json"] == str(tmp_path / "release-evidence.json")
    assert record["release_bundle"]["preflight_json"] == str(tmp_path / "release-inputs.json")
    assert record["release_bundle"]["engine_actions_jsons"] == [
        str(tmp_path / "vllm-actions.json"),
        str(tmp_path / "sglang-actions.json"),
    ]
    assert record["release_bundle"]["databricks_run_status_jsons"] == [
        str(tmp_path / "databricks-run-status.json")
    ]
    assert record["release_bundle"]["github_governance_json"] == str(tmp_path / "github-governance.json")
    assert record["release_bundle"]["repository_hygiene_json"] == str(tmp_path / "repository-hygiene.json")
    assert record["release_bundle"]["native_probe_factories_jsons"] == [
        str(tmp_path / "native-probe-factories.json")
    ]
    assert record["release_bundle"]["require_complete_v1"] is True
    assert governance_argv[:3] == [sys.executable, "-m", "document_kv_cache.github_governance"]
    assert governance_argv[governance_argv.index("--output-json") + 1] == str(tmp_path / "github-governance.json")
    assert hygiene_argv[:3] == [sys.executable, "-m", "document_kv_cache.repository_hygiene"]
    assert hygiene_argv[hygiene_argv.index("--repository-root") + 1] == "."
    assert hygiene_argv[hygiene_argv.index("--output-json") + 1] == str(tmp_path / "repository-hygiene.json")
    assert native_argv[:3] == [sys.executable, "-m", "document_kv_cache.native_probe_factories"]
    assert native_argv[native_argv.index("--output-json") + 1] == str(tmp_path / "native-probe-factories.json")
    assert preflight_argv[:3] == [sys.executable, "-m", "document_kv_cache.release_evidence"]
    assert "--preflight-only" in preflight_argv
    assert preflight_argv[preflight_argv.index("--preflight-output-json") + 1] == str(
        tmp_path / "release-inputs.json"
    )
    assert bundle_argv[:3] == [sys.executable, "-m", "document_kv_cache.release_bundle"]
    assert bundle_argv.count("--engine-probe-json") == 2
    assert bundle_argv.count("--engine-actions-json") == 2
    assert bundle_argv[bundle_argv.index("--output-dir") + 1] == str(tmp_path / "release-bundle")
    assert bundle_argv[bundle_argv.index("--preflight-json") + 1] == str(tmp_path / "release-inputs.json")
    assert bundle_argv.count("--preflight-json") == 1
    assert bundle_argv[bundle_argv.index("--github-governance-json") + 1] == str(
        tmp_path / "github-governance.json"
    )
    assert bundle_argv.count("--github-governance-json") == 1
    assert bundle_argv[bundle_argv.index("--repository-hygiene-json") + 1] == str(
        tmp_path / "repository-hygiene.json"
    )
    assert bundle_argv.count("--repository-hygiene-json") == 1
    assert bundle_argv[bundle_argv.index("--native-probe-factories-json") + 1] == str(
        tmp_path / "native-probe-factories.json"
    )
    assert "--require-complete-v1" in bundle_argv
    assert "--overwrite" in bundle_argv


def test_main_rejects_incomplete_strict_release_bundle_command(capsys, tmp_path):
    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--storage-benchmark-workspace-dir",
            "/local_disk0/document-kv-storage-benchmark",
            "--storage-benchmark-output-json",
            str(tmp_path / "storage.json"),
            "--storage-benchmark-uc-volume-root",
            "/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
            "--release-evidence-output-json",
            str(tmp_path / "release-evidence.json"),
            "--release-engine-probe-json",
            str(tmp_path / "vllm-probe.json"),
            "--release-engine-probe-json",
            str(tmp_path / "sglang-probe.json"),
            "--release-engine-actions-json",
            str(tmp_path / "vllm-actions.json"),
            "--release-engine-actions-json",
            str(tmp_path / "sglang-actions.json"),
            "--release-bundle-output-dir",
            str(tmp_path / "release-bundle"),
            "--release-bundle-require-complete-v1",
        ]
    )

    record = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert record["ok"] is False
    assert "strict V1 release bundle plans require" in record["error"]
    assert "preflight sidecar" in record["error"]
    assert "benchmark plan execution sidecar" in record["error"]
    assert "tested package wheel" in record["error"]


def test_main_rejects_plan_output_json_colliding_with_generated_artifact(capsys, tmp_path):
    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--benchmark-output-json",
            str(tmp_path / "results.json"),
            "--plan-output-json",
            str(tmp_path / "subdir" / ".." / "results.json"),
        ]
    )

    record = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert record["ok"] is False
    assert "plan_output_json" in record["error"]
    assert "benchmark_output_json" in record["error"]


def test_main_rejects_plan_output_sh_colliding_with_release_bundle_output_dir(capsys, tmp_path):
    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--storage-benchmark-workspace-dir",
            "/local_disk0/document-kv-storage-benchmark",
            "--storage-benchmark-output-json",
            str(tmp_path / "storage.json"),
            "--storage-benchmark-uc-volume-root",
            "/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
            "--release-evidence-output-json",
            str(tmp_path / "release-evidence.json"),
            "--release-engine-probe-json",
            str(tmp_path / "vllm-probe.json"),
            "--release-engine-probe-json",
            str(tmp_path / "sglang-probe.json"),
            "--release-engine-actions-json",
            str(tmp_path / "vllm-actions.json"),
            "--release-engine-actions-json",
            str(tmp_path / "sglang-actions.json"),
            "--release-bundle-output-dir",
            str(tmp_path / "release-bundle"),
            "--plan-output-sh",
            f"disk:{tmp_path / 'release-bundle'}",
        ]
    )

    record = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert record["ok"] is False
    assert "plan_output_sh" in record["error"]
    assert "release_bundle.output_dir" in record["error"]


def test_main_rejects_plan_output_json_and_shell_collisions(capsys, tmp_path):
    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--plan-output-json",
            str(tmp_path / "plan.json"),
            "--plan-output-sh",
            f"file:{tmp_path / 'plan.json'}",
        ]
    )

    record = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert record["ok"] is False
    assert "plan_output_json" in record["error"]
    assert "plan_output_sh" in record["error"]


def test_main_rejects_release_bundle_options_without_output_dir(capsys, tmp_path):
    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--release-bundle-package-wheel",
            str(tmp_path / "dist" / "document_kv_cache-0.2.0-py3-none-any.whl"),
        ]
    )

    record = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert record["ok"] is False
    assert "--release-bundle-output-dir" in record["error"]


def test_main_uses_all_storage_readers_when_uc_root_is_explicit(tmp_path):
    plan_json = tmp_path / "plan.json"

    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--storage-benchmark-workspace-dir",
            "/local_disk0/document-kv-storage-benchmark",
            "--storage-benchmark-uc-volume-root",
            "/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
            "--plan-output-json",
            str(plan_json),
        ]
    )

    storage_argv = json.loads(plan_json.read_text(encoding="utf-8"))["commands"][-1]["argv"]

    assert exit_code == 0
    assert storage_argv.count("--reader") == 3
    assert "unity_catalog" in storage_argv
    assert "/Volumes/catalog/schema/volume/document-kv-storage-benchmark" in storage_argv


def test_main_rejects_storage_options_without_workspace(capsys, tmp_path):
    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--storage-benchmark-uc-volume-root",
            "/Volumes/catalog/schema/volume/document-kv-storage-benchmark",
        ]
    )

    record = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert record["ok"] is False
    assert "--storage-benchmark-workspace-dir" in record["error"]


def test_main_rejects_release_evidence_options_without_output(capsys, tmp_path):
    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--release-engine-probe-json",
            str(tmp_path / "vllm-probe.json"),
        ]
    )

    record = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert record["ok"] is False
    assert "--release-evidence-output-json" in record["error"]


def test_main_rejects_release_evidence_without_engine_probe_json(capsys, tmp_path):
    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--release-evidence-output-json",
            str(tmp_path / "release-evidence.json"),
        ]
    )

    record = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert record["ok"] is False
    assert "--release-engine-probe-json" in record["error"]
    assert "planned engine probes" in record["error"]


def test_main_rejects_incomplete_planned_engine_probe_options(capsys, tmp_path):
    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--engine-probe-handoff-json",
            f"vllm={tmp_path / 'handoff.json'}",
            "--engine-probe-output-json",
            f"vllm={tmp_path / 'probe.json'}",
        ]
    )

    record = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert record["ok"] is False
    assert "--engine-probe-factory" in record["error"]


def test_main_rejects_builtin_engine_probe_factories_without_handoff(capsys, tmp_path):
    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--engine-probe-use-builtin-factories",
        ]
    )

    record = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert record["ok"] is False
    assert "--engine-probe-handoff-json" in record["error"]


def test_main_rejects_unplanned_explicit_factory_with_builtin_engine_probe_factories(capsys, tmp_path):
    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--engine-probe-handoff-json",
            f"vllm={tmp_path / 'handoff.json'}",
            "--engine-probe-output-json",
            f"vllm={tmp_path / 'probe.json'}",
            "--engine-probe-factory",
            "sglang=sglang_probe:factory",
            "--engine-probe-use-builtin-factories",
        ]
    )

    record = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert record["ok"] is False
    assert "--engine-probe-factory" in record["error"]
    assert "extra=['sglang']" in record["error"]


def test_main_rejects_engine_probe_options_without_handoff(capsys, tmp_path):
    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--engine-probe-factory",
            "vllm=probe:factory",
        ]
    )

    record = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert record["ok"] is False
    assert "--engine-probe-handoff-json" in record["error"]


def test_main_rejects_explicit_uc_reader_without_uc_root(capsys, tmp_path):
    exit_code = main(
        [
            "--raw-dataset",
            f"biography={tmp_path / 'raw' / 'biography.jsonl'}",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--base-url",
            "http://localhost:8000",
            "--allow-partial",
            "--storage-benchmark-workspace-dir",
            "/local_disk0/document-kv-storage-benchmark",
            "--storage-benchmark-reader",
            "unity_catalog",
        ]
    )

    record = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert record["ok"] is False
    assert "uc_volume_root" in record["error"]
