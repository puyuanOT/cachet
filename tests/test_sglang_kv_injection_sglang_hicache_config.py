import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

import sglang_kv_injection
from sglang_kv_injection.sglang_hicache_config import (
    DOCUMENT_KV_HICACHE_CONFIG_RECORD_TYPE,
    DOCUMENT_KV_HICACHE_CONFIG_SCHEMA_VERSION,
    SGLANG_HICACHE_DYNAMIC_BACKEND,
    main,
    sglang_hicache_cli_args,
    sglang_hicache_launch_config,
)
from sglang_kv_injection.sglang_dynamic_backend import (
    DOCUMENT_KV_HICACHE_BACKEND_CLASS,
    DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH,
    DOCUMENT_KV_HICACHE_PROVIDER_FACTORY,
    DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


def test_sglang_hicache_launch_config_builds_dynamic_backend_payload():
    config = sglang_hicache_launch_config(
        backend_name="document_kv",
        module_path="company_sglang_patch.document_kv_backend",
        class_name="DocumentKVHiCacheBackend",
        provider_factory="company_sglang_patch.providers:build_provider",
        extra_config={"tenant": "qa", "max_ready_queue": 16},
        page_size=64,
        hicache_ratio=2.5,
        hicache_size_gb=96,
        hicache_io_backend="kernel",
        hicache_mem_layout="page_first",
        hicache_storage_prefetch_policy="timeout",
        hicache_write_policy="write_through",
    )

    assert config["enable_hierarchical_cache"] is True
    assert config["hicache_storage_backend"] == SGLANG_HICACHE_DYNAMIC_BACKEND
    assert config["page_size"] == 64
    assert config["hicache_ratio"] == 2.5
    assert config["hicache_size"] == 96
    assert config["hicache_io_backend"] == "kernel"
    extra = json.loads(config["hicache_storage_backend_extra_config"])
    assert extra["backend_name"] == "document_kv"
    assert extra["module_path"] == "company_sglang_patch.document_kv_backend"
    assert extra["class_name"] == "DocumentKVHiCacheBackend"
    assert extra["hicache_storage_pass_prefix_keys"] is True
    assert extra["interface_v1"] is True
    assert extra[DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY] == (
        "company_sglang_patch.providers:build_provider"
    )
    assert extra["tenant"] == "qa"
    assert extra["max_ready_queue"] == 16
    assert extra["document_kv.record_type"] == DOCUMENT_KV_HICACHE_CONFIG_RECORD_TYPE
    assert extra["document_kv.schema_version"] == DOCUMENT_KV_HICACHE_CONFIG_SCHEMA_VERSION
    assert extra["document_kv.backend"] == "sglang"
    assert extra["document_kv.kv_injection_method"] == "runtime-prefix-cache-bind"
    assert extra["document_kv.engine_handoff_record_type"] == "document_kv.engine_adapter_request.v1"
    assert extra["document_kv.engine_handoff_schema_version"] == 2
    assert extra["document_kv.requires_native_runtime"] is True


def test_sglang_hicache_launch_config_defaults_to_builtin_page_provider():
    config = sglang_hicache_launch_config()
    extra = json.loads(config["hicache_storage_backend_extra_config"])

    assert extra[DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY] == DOCUMENT_KV_HICACHE_PROVIDER_FACTORY
    assert extra["hicache_storage_pass_prefix_keys"] is True
    assert extra["interface_v1"] is True


def test_sglang_hicache_cli_args_are_launch_server_ready():
    args = sglang_hicache_cli_args(
        provider_factory="company_sglang_patch.providers:build_provider",
        page_size=64,
    )

    assert args[0] == "--enable-hierarchical-cache"
    assert "--hicache-storage-backend" in args
    assert "--hicache-storage-backend-extra-config" in args
    assert "--page-size" in args
    assert "64" in args
    extra_index = args.index("--hicache-storage-backend-extra-config") + 1
    extra = json.loads(args[extra_index])
    assert extra["module_path"] == DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH
    assert extra["class_name"] == DOCUMENT_KV_HICACHE_BACKEND_CLASS
    assert extra["hicache_storage_pass_prefix_keys"] is True
    assert extra["interface_v1"] is True
    assert extra[DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY] == (
        "company_sglang_patch.providers:build_provider"
    )
    assert extra["document_kv.requires_native_runtime"] is True


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("backend_name", {"backend_name": "", "module_path": "mod", "class_name": "Backend"}),
        ("module_path", {"backend_name": "document_kv", "module_path": "", "class_name": "Backend"}),
        ("class_name", {"backend_name": "document_kv", "module_path": "mod", "class_name": " "}),
    ],
)
def test_sglang_hicache_launch_config_requires_backend_identity(field, kwargs):
    with pytest.raises(ValueError, match=field):
        sglang_hicache_launch_config(**kwargs)


def test_sglang_hicache_launch_config_rejects_reserved_extra_config_keys():
    with pytest.raises(ValueError, match="document_kv"):
        sglang_hicache_launch_config(
            backend_name="document_kv",
            module_path="company_sglang_patch.document_kv_backend",
            class_name="DocumentKVHiCacheBackend",
            extra_config={"document_kv.record_type": "override"},
        )


def test_sglang_hicache_launch_config_rejects_invalid_provider_factory():
    with pytest.raises(ValueError, match="module:attribute"):
        sglang_hicache_launch_config(provider_factory="not-a-factory")


@pytest.mark.parametrize("reserved_key", ["backend_name", "module_path", "class_name"])
def test_sglang_hicache_launch_config_rejects_dynamic_backend_identity_overrides(reserved_key):
    with pytest.raises(ValueError, match="dynamic backend identity"):
        sglang_hicache_launch_config(
            backend_name="document_kv",
            module_path="company_sglang_patch.document_kv_backend",
            class_name="DocumentKVHiCacheBackend",
            extra_config={reserved_key: "override"},
        )


def test_sglang_hicache_launch_config_rejects_non_json_extra_config():
    with pytest.raises(TypeError, match="JSON-serializable"):
        sglang_hicache_launch_config(
            backend_name="document_kv",
            module_path="company_sglang_patch.document_kv_backend",
            class_name="DocumentKVHiCacheBackend",
            extra_config={"bad": object()},
        )


@pytest.mark.parametrize("bad_number", [math.inf, -math.inf, math.nan])
def test_sglang_hicache_launch_config_rejects_non_finite_numbers(bad_number):
    with pytest.raises(ValueError, match="hicache_ratio"):
        sglang_hicache_launch_config(
            backend_name="document_kv",
            module_path="company_sglang_patch.document_kv_backend",
            class_name="DocumentKVHiCacheBackend",
            hicache_ratio=bad_number,
        )


def test_sglang_hicache_launch_config_rejects_non_standard_json_numbers_in_extra_config():
    with pytest.raises(TypeError, match="JSON-serializable"):
        sglang_hicache_launch_config(
            backend_name="document_kv",
            module_path="company_sglang_patch.document_kv_backend",
            class_name="DocumentKVHiCacheBackend",
            extra_config={"bad": math.nan},
        )


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("page_size", {"page_size": 0}),
        ("hicache_ratio", {"hicache_ratio": 0}),
        ("hicache_size", {"hicache_size_gb": -1}),
        ("hicache_io_backend", {"hicache_io_backend": ""}),
    ],
)
def test_sglang_hicache_launch_config_validates_optional_server_args(field, kwargs):
    with pytest.raises(ValueError, match=field):
        sglang_hicache_launch_config(
            backend_name="document_kv",
            module_path="company_sglang_patch.document_kv_backend",
            class_name="DocumentKVHiCacheBackend",
            **kwargs,
        )


def test_package_root_reexports_hicache_helpers():
    assert sglang_kv_injection.sglang_hicache_launch_config is sglang_hicache_launch_config
    assert sglang_kv_injection.sglang_hicache_cli_args is sglang_hicache_cli_args


def test_main_writes_hicache_config_sidecar(tmp_path):
    output_json = tmp_path / "sglang-launch-config.json"

    exit_code = main(
        [
            "--module-path",
            "company_sglang_patch.document_kv_backend",
            "--class-name",
            "DocumentKVHiCacheBackend",
            "--provider-factory",
            "company_sglang_patch.providers:build_provider",
            "--extra-config",
            "tenant=\"qa\"",
            "--extra-config",
            "max_ready_queue=16",
            "--page-size",
            "64",
            "--hicache-ratio",
            "2.5",
            "--hicache-size-gb",
            "96",
            "--output-json",
            str(output_json),
        ]
    )

    config = json.loads(output_json.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert config == sglang_hicache_launch_config(
        backend_name="document_kv",
        module_path="company_sglang_patch.document_kv_backend",
        class_name="DocumentKVHiCacheBackend",
        provider_factory="company_sglang_patch.providers:build_provider",
        extra_config={"tenant": "qa", "max_ready_queue": 16},
        page_size=64,
        hicache_ratio=2.5,
        hicache_size_gb=96,
    )


def test_main_prints_hicache_config_to_stdout(capsys):
    exit_code = main([])

    output = capsys.readouterr().out
    config = json.loads(output)
    extra = json.loads(config["hicache_storage_backend_extra_config"])

    assert exit_code == 0
    assert config["enable_hierarchical_cache"] is True
    assert config["hicache_storage_backend"] == SGLANG_HICACHE_DYNAMIC_BACKEND
    assert extra["backend_name"] == "document_kv"
    assert extra["module_path"] == DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH
    assert extra["class_name"] == DOCUMENT_KV_HICACHE_BACKEND_CLASS
    assert extra[DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY] == DOCUMENT_KV_HICACHE_PROVIDER_FACTORY


def test_main_reports_invalid_extra_config(capsys):
    exit_code = main(
        [
            "--module-path",
            "company_sglang_patch.document_kv_backend",
            "--extra-config",
            "tenant=not-json",
        ]
    )

    record = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert record["ok"] is False
    assert record["error_type"] == "ValueError"
    assert "valid JSON" in record["error"]
