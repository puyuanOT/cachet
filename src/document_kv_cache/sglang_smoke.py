"""Databricks-friendly SGLang live smoke for the Qwen3 Cachet path."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any
import urllib.error
import urllib.request

from document_kv_cache.benchmark_handoffs import load_benchmark_kv_chunk_generator
from document_kv_cache.benchmarks import (
    DEFAULT_HARDWARE_TARGET,
    DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
    benchmark_cache_request,
    benchmark_cache_source_document,
    validate_v1_hardware_target,
)
from document_kv_cache.engine_adapters import ServingBackend, build_engine_adapter_request, sglang_adapter_spec
from document_kv_cache.engine_probe import write_engine_adapter_handoff_bundle
from document_kv_cache.live_server import (
    LiveServerCheckConfig,
    build_live_server_check_request,
    live_check_kv_transfer_params,
    run_openai_compatible_live_check,
)
from document_kv_cache.manifest import InMemoryManifestStore
from document_kv_cache.model_profiles import QWEN3_4B_INSTRUCT_HF_MODEL_ID, layout_for_model
from document_kv_cache.openai_compatible import PromptTextMode
from document_kv_cache.serving_env import (
    SGLANG_DEPENDENCY_CONSTRAINTS,
    SGLANG_SERVING_ENVIRONMENT_PROFILE,
    SGLANG_VERSION,
)
from document_kv_cache.workflow import CacheBuildConfig, DocumentKVWorkflow
from sglang_kv_injection.hicache_keys import sglang_hicache_page_keys
from sglang_kv_injection.sglang_dynamic_backend import (
    DOCUMENT_KV_HICACHE_BACKEND_CLASS,
    DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH,
    DOCUMENT_KV_HICACHE_PAGE_STORE_URI_CONFIG_KEY,
    DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY,
    DocumentKVHiCacheBackend,
    NoOpDocumentKVHiCacheProvider,
)
from sglang_kv_injection.sglang_hicache_config import (
    sglang_hicache_cli_args,
    sglang_hicache_launch_config,
)
from sglang_kv_injection.sglang_request_metadata_bridge import (
    sglang_request_metadata_bridge_status_to_record,
)

HF_MODEL_ID = QWEN3_4B_INSTRUCT_HF_MODEL_ID
SERVED_MODEL_NAME = "qwen3:4b-instruct"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
DEFAULT_LOCAL_ROOT = Path("/local_disk0")
DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV = "DOCUMENT_KV_PACKAGE_INSTALL_SPEC"
DEFAULT_SGLANG_LIVE_HANDOFF_GENERATOR_FACTORY = (
    "document_kv_cache.transformers_generator:build_transformers_kv_chunk_generator"
)
DEFAULT_SGLANG_HICACHE_PAGE_SIZE = 1
LIVE_HANDOFF_CACHE_ARTIFACT_PREFIX = "cachet-live"
SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE = (
    "SGLang handoff-backed live smoke requires handoff_json or handoff_record "
    "plus explicit sglang_hicache_page_keys metadata so Cachet can bind the "
    "handoff to SGLang runtime HiCache page keys."
)
SGLANG_BASELINE_HANDOFF_FIELDS_UNSUPPORTED_MESSAGE = (
    "baseline-only SGLang smoke must not include handoff_json, handoff_record, handoff_record_json, "
    "payload_uri, request_id, sglang_hicache_page_keys, or generated live handoff settings"
)
SGLANG_GENERATED_HANDOFF_EXPLICIT_FIELDS_UNSUPPORTED_MESSAGE = (
    "generated SGLang live handoff must not be combined with explicit handoff_json, "
    "handoff_record, payload_uri, request_id, or sglang_hicache_page_keys"
)

__all__ = [
    "SGLANG_VERSION",
    "SGLANG_DEPENDENCY_CONSTRAINTS",
    "HF_MODEL_ID",
    "SERVED_MODEL_NAME",
    "SERVER_BASE_URL",
    "DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV",
    "DEFAULT_SGLANG_LIVE_HANDOFF_GENERATOR_FACTORY",
    "DEFAULT_SGLANG_HICACHE_PAGE_SIZE",
    "LIVE_HANDOFF_CACHE_ARTIFACT_PREFIX",
    "SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE",
    "SGLANG_BASELINE_HANDOFF_FIELDS_UNSUPPORTED_MESSAGE",
    "SGLANG_GENERATED_HANDOFF_EXPLICIT_FIELDS_UNSUPPORTED_MESSAGE",
    "SGLangLiveHandoffGenerationConfig",
    "SGLangSmokeBenchmarkConfig",
    "build_metadata",
    "build_sglang_hicache_provider_probe_record",
    "build_sglang_server_args",
    "dependency_constraints",
    "document_kv_package_install_spec",
    "install_document_kv_package",
    "install_sglang",
    "parse_args",
    "prepare_generated_live_handoff",
    "release_live_handoff_generation_resources",
    "run_sglang_live_smoke",
    "sglang_hicache_config_for_smoke",
    "sglang_live_kv_transfer_params",
    "write_json",
    "main",
]


@dataclass(frozen=True, slots=True)
class SGLangLiveHandoffGenerationConfig:
    """Optional generation settings for the SGLang live smoke cache arm."""

    output_dir: Path
    generator_factory: str = DEFAULT_SGLANG_LIVE_HANDOFF_GENERATOR_FACTORY
    dtype: str = "bfloat16"
    align_bytes: int = 4096
    page_size: int = DEFAULT_SGLANG_HICACHE_PAGE_SIZE
    timeout_seconds: float = 1800.0

    def __post_init__(self) -> None:
        if self.output_dir is None:
            raise ValueError("live_handoff_output_dir must be provided")
        if not isinstance(self.generator_factory, str) or not self.generator_factory.strip():
            raise ValueError("live_handoff_generator_factory must be non-empty")
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise ValueError("live_handoff_dtype must be non-empty")
        if isinstance(self.align_bytes, bool) or not isinstance(self.align_bytes, int) or self.align_bytes <= 0:
            raise ValueError("live_handoff_align_bytes must be a positive integer")
        if isinstance(self.page_size, bool) or not isinstance(self.page_size, int) or self.page_size <= 0:
            raise ValueError("sglang_hicache_page_size must be a positive integer")
        if self.timeout_seconds <= 0:
            raise ValueError("live_handoff_generation_timeout_seconds must be positive")
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    def to_metadata(self) -> dict[str, object]:
        return {
            "generator_factory": self.generator_factory,
            "output_dir": str(self.output_dir),
            "dtype": self.dtype,
            "align_bytes": self.align_bytes,
            "sglang_hicache_page_size": self.page_size,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True, slots=True)
class SGLangSmokeBenchmarkConfig:
    """Runtime configuration for a one-node Databricks SGLang live smoke."""

    benchmark_id: str
    output_dir: Path
    max_tokens: int = 32
    timeout_seconds: float = 240.0
    import_probe_timeout_seconds: float = 180.0
    server_start_timeout_seconds: float = 480.0
    local_root: Path = DEFAULT_LOCAL_ROOT
    server_host: str = SERVER_HOST
    server_port: int = SERVER_PORT
    client_host: str = SERVER_HOST
    context_length: int = 4096
    mem_fraction_static: float = 0.85
    hardware_target: str = DEFAULT_HARDWARE_TARGET
    stream: bool = True
    package_install_spec: str | None = None
    baseline_only: bool = False
    cache_prompt_text_mode: PromptTextMode = "runtime"
    handoff_json: str | None = None
    handoff_record: Mapping[str, Any] | None = None
    payload_uri: str | None = None
    request_id: str | None = None
    sglang_hicache_page_keys: Sequence[str] | None = None
    handoff_generation: SGLangLiveHandoffGenerationConfig | None = None
    hicache_page_store_uri: str | None = None
    hicache_page_size: int | None = None
    hicache_ratio: float | None = None
    hicache_size_gb: int | None = None
    hicache_io_backend: str | None = None
    hicache_mem_layout: str | None = None
    hicache_storage_prefetch_policy: str | None = None
    hicache_write_policy: str | None = None

    def __post_init__(self) -> None:
        if not self.benchmark_id:
            raise ValueError("benchmark_id must be non-empty")
        if self.output_dir is None:
            raise ValueError("output_dir must be provided")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.import_probe_timeout_seconds <= 0:
            raise ValueError("import_probe_timeout_seconds must be positive")
        if self.server_start_timeout_seconds <= 0:
            raise ValueError("server_start_timeout_seconds must be positive")
        if self.local_root is None:
            raise ValueError("local_root must be provided")
        if not self.server_host:
            raise ValueError("server_host must be non-empty")
        if not 0 < self.server_port < 65536:
            raise ValueError("server_port must be between 1 and 65535")
        if not self.client_host:
            raise ValueError("client_host must be non-empty")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if not 0 < self.mem_fraction_static <= 1:
            raise ValueError("mem_fraction_static must be in (0, 1]")
        if type(self.stream) is not bool:
            raise ValueError("stream must be a boolean")
        if type(self.baseline_only) is not bool:
            raise ValueError("baseline_only must be a boolean")
        if not isinstance(self.hardware_target, str) or not self.hardware_target.strip():
            raise ValueError("hardware_target must be non-empty")
        validate_v1_hardware_target(self.hardware_target)
        if self.package_install_spec is not None and not self.package_install_spec.strip():
            raise ValueError("package_install_spec must be non-empty when provided")
        if self.cache_prompt_text_mode not in {"logical", "runtime"}:
            raise ValueError("cache_prompt_text_mode must be 'logical' or 'runtime'")
        if self.handoff_json and self.handoff_record is not None:
            raise ValueError("SGLang smoke handoff params must use only one of handoff_json or handoff_record")
        if self.handoff_record is not None and not isinstance(self.handoff_record, Mapping):
            raise ValueError("handoff_record must be a JSON object")
        page_keys_provided = self.sglang_hicache_page_keys is not None
        page_keys = _string_tuple(self.sglang_hicache_page_keys or (), field_name="sglang_hicache_page_keys")
        has_handoff_fields = any(
            value is not None
            for value in (self.handoff_json, self.handoff_record, self.payload_uri, self.request_id)
        ) or page_keys_provided
        if self.handoff_generation is not None and not isinstance(
            self.handoff_generation, SGLangLiveHandoffGenerationConfig
        ):
            raise TypeError("handoff_generation must be a SGLangLiveHandoffGenerationConfig")
        if (
            self.hicache_page_size is not None
            and (isinstance(self.hicache_page_size, bool) or not isinstance(self.hicache_page_size, int))
        ):
            raise ValueError("hicache_page_size must be a positive integer when provided")
        if self.hicache_page_size is not None and self.hicache_page_size <= 0:
            raise ValueError("hicache_page_size must be a positive integer when provided")
        if (
            self.handoff_generation is not None
            and self.hicache_page_size is not None
            and self.hicache_page_size != self.handoff_generation.page_size
        ):
            raise ValueError("hicache_page_size must match live handoff generation page_size")
        if self.baseline_only:
            if has_handoff_fields or self.handoff_generation is not None:
                raise ValueError(SGLANG_BASELINE_HANDOFF_FIELDS_UNSUPPORTED_MESSAGE)
        else:
            if self.handoff_generation is not None:
                if has_handoff_fields:
                    raise ValueError(SGLANG_GENERATED_HANDOFF_EXPLICIT_FIELDS_UNSUPPORTED_MESSAGE)
            else:
                if self.handoff_json is None and self.handoff_record is None:
                    raise ValueError(SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE)
                if not page_keys:
                    raise ValueError(SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "local_root", Path(self.local_root))
        object.__setattr__(self, "sglang_hicache_page_keys", page_keys)
        if self.handoff_generation is not None and self.hicache_page_size is None:
            object.__setattr__(self, "hicache_page_size", self.handoff_generation.page_size)
        if self.handoff_record is not None:
            object.__setattr__(self, "handoff_record", dict(self.handoff_record))

    @property
    def local_dir(self) -> Path:
        return self.local_root / f"document-kv-sglang-smoke-{self.benchmark_id}"

    @property
    def hf_cache_dir(self) -> Path:
        return self.local_root / "hf-cache"

    @property
    def server_base_url(self) -> str:
        return f"http://{self.client_host}:{self.server_port}"

    @property
    def venv_dir(self) -> Path:
        return self.local_dir / "sglang-venv"

    @property
    def venv_python(self) -> Path:
        return self.venv_dir / "bin" / "python"

    @property
    def server_log_path(self) -> Path:
        return self.local_dir / "sglang-server.log"

    @property
    def server_log_copy_path(self) -> Path:
        return self.output_dir / "sglang-server.log"

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "metadata.json"

    @property
    def import_probe_path(self) -> Path:
        return self.output_dir / "sglang-import-probe.json"

    @property
    def launch_config_path(self) -> Path:
        return self.output_dir / "sglang-launch-config.json"

    @property
    def live_smoke_output_path(self) -> Path:
        return self.output_dir / "sglang-live-smoke.json"

    @property
    def live_handoff_generation_path(self) -> Path:
        return self.output_dir / "sglang-live-handoff-generation.json"


def run_sglang_live_smoke(config: SGLangSmokeBenchmarkConfig) -> None:
    """Create an isolated SGLang env, start Qwen3, and run strict live checks."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.local_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(config.hf_cache_dir)

    launch_config = sglang_hicache_config_for_smoke(config)
    write_json(config.launch_config_path, launch_config)
    metadata = build_metadata(config, launch_config=launch_config)
    write_json(config.metadata_path, metadata)

    create_venv(config.venv_dir)
    install_sglang(config.venv_python)
    install_document_kv_package(config.venv_python, document_kv_package_install_spec(config))
    metadata.update(installed_versions(config.venv_python))
    write_json(config.metadata_path, metadata)
    import_probe_record = probe_sglang_import(
        config.venv_python,
        config.import_probe_path,
        launch_config_path=config.launch_config_path,
        timeout_seconds=config.import_probe_timeout_seconds,
        env=server_env(config),
    )
    metadata.update(_metadata_from_import_probe(import_probe_record))

    runtime_config = prepare_generated_live_handoff(config)
    if runtime_config is not config:
        metadata["generated_live_handoff"] = True
        metadata["live_handoff_generation_path"] = str(config.live_handoff_generation_path)
        metadata["generated_live_handoff_json"] = runtime_config.handoff_json
        metadata["generated_live_handoff_payload_uri"] = runtime_config.payload_uri
        metadata["generated_live_handoff_page_keys"] = len(runtime_config.sglang_hicache_page_keys or ())
        write_json(config.metadata_path, metadata)

    metadata["sglang_server_local_log"] = str(runtime_config.server_log_path)
    metadata["sglang_server_log"] = str(runtime_config.server_log_copy_path)
    metadata["live_smoke_output_path"] = str(runtime_config.live_smoke_output_path)
    write_json(config.metadata_path, metadata)

    server = start_sglang_server(runtime_config, runtime_config.venv_python, runtime_config.server_log_path)
    try:
        wait_for_sglang_server(
            server,
            runtime_config.server_log_path,
            runtime_config,
            timeout_seconds=runtime_config.server_start_timeout_seconds,
        )
        copy_file_if_exists(runtime_config.server_log_path, runtime_config.server_log_copy_path)
        run_live_checks(runtime_config, import_probe_record=import_probe_record)
    finally:
        terminate_process(server)
        copy_file_if_exists(runtime_config.server_log_path, runtime_config.server_log_copy_path)


def build_metadata(
    config: SGLangSmokeBenchmarkConfig,
    *,
    launch_config: Mapping[str, Any] | None = None,
    import_probe_record: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "benchmark_id": config.benchmark_id,
        "hf_model_id": HF_MODEL_ID,
        "served_model_name": SERVED_MODEL_NAME,
        "sglang_version_requested": SGLANG_VERSION,
        "server_bind_host": config.server_host,
        "server_client_host": config.client_host,
        "server_base_url": config.server_base_url,
        "hf_home": str(config.hf_cache_dir),
        "sglang_python": str(config.venv_python),
        "dependency_constraints": dependency_constraints(),
        "context_length": config.context_length,
        "mem_fraction_static": config.mem_fraction_static,
        "hardware_target": config.hardware_target,
        "document_kv_package_install_spec": document_kv_package_install_spec(config),
        "baseline_only": config.baseline_only,
        "cache_arm_supported": not config.baseline_only,
        "cache_arm_blocker": None if not config.baseline_only else SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE,
        "live_request_metadata_bridge_required": True,
        "live_request_metadata_bridge_ok": False,
        "requires_kv_transfer_params": not config.baseline_only,
        "cache_prompt_text_mode": config.cache_prompt_text_mode,
        "kv_transfer_params_transport": "custom_params",
        "generates_live_handoff": config.handoff_generation is not None,
        "live_handoff_generation": (
            config.handoff_generation.to_metadata() if config.handoff_generation is not None else None
        ),
        "live_handoff_generation_path": str(config.live_handoff_generation_path)
        if config.handoff_generation is not None
        else None,
        "sglang_hicache_launch_config": dict(launch_config or sglang_hicache_config_for_smoke(config)),
    }
    metadata.update(_metadata_from_import_probe(import_probe_record))
    return metadata


def sglang_hicache_config_for_smoke(config: SGLangSmokeBenchmarkConfig) -> dict[str, Any]:
    extra_config: dict[str, Any] = {}
    if config.hicache_page_store_uri is not None:
        extra_config[DOCUMENT_KV_HICACHE_PAGE_STORE_URI_CONFIG_KEY] = config.hicache_page_store_uri
    return sglang_hicache_launch_config(
        extra_config=extra_config,
        page_size=config.hicache_page_size,
        hicache_ratio=config.hicache_ratio,
        hicache_size_gb=config.hicache_size_gb,
        hicache_io_backend=config.hicache_io_backend,
        hicache_mem_layout=config.hicache_mem_layout,
        hicache_storage_prefetch_policy=config.hicache_storage_prefetch_policy,
        hicache_write_policy=config.hicache_write_policy,
    )


def build_sglang_hicache_provider_probe_record(
    launch_config: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Instantiate the configured SGLang dynamic backend and verify provider wiring."""

    config = launch_config or sglang_hicache_launch_config()
    if not isinstance(config, Mapping):
        raise TypeError("SGLang HiCache launch config must be a mapping")
    if config.get("enable_hierarchical_cache") is not True:
        raise ValueError("enable_hierarchical_cache must be true")
    if config.get("hicache_storage_backend") != "dynamic":
        raise ValueError("hicache_storage_backend must be 'dynamic'")
    extra_config = _hicache_extra_config(config)
    provider_factory = extra_config.get(DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY)
    if not isinstance(provider_factory, str) or not provider_factory.strip():
        raise ValueError(
            f"{DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY} must be a non-empty module:attribute string"
        )
    if extra_config.get("document_kv.requires_native_runtime") is not True:
        raise ValueError("document_kv.requires_native_runtime must be true")

    backend = DocumentKVHiCacheBackend(
        SimpleNamespace(
            extra_config=extra_config,
            model_path=HF_MODEL_ID,
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
        ),
        {},
    )
    provider = backend.provider
    if isinstance(provider, NoOpDocumentKVHiCacheProvider):
        raise ValueError("SGLang smoke cannot run with NoOpDocumentKVHiCacheProvider")
    if getattr(provider, "document_kv_hicache_provider", False) is not True:
        raise TypeError("SGLang smoke requires a runtime-facing document KV HiCache provider")
    request_metadata_bridge = sglang_request_metadata_bridge_status_to_record(
        getattr(backend, "request_metadata_bridge_status", None)
    )

    return {
        "document_kv_hicache_provider_ok": True,
        "document_kv_provider_factory": provider_factory,
        "document_kv_provider_type": f"{type(provider).__module__}.{type(provider).__qualname__}",
        "document_kv_backend_type": f"{type(backend).__module__}.{type(backend).__qualname__}",
        "document_kv_request_metadata_bridge": request_metadata_bridge,
        "document_kv_request_metadata_bridge_ok": request_metadata_bridge.get("ok") is True,
        "document_kv_requires_native_runtime": True,
        "document_kv_hicache_backend_module": DOCUMENT_KV_HICACHE_BACKEND_MODULE_PATH,
        "document_kv_hicache_backend_class": DOCUMENT_KV_HICACHE_BACKEND_CLASS,
    }


def sglang_live_kv_transfer_params(config: SGLangSmokeBenchmarkConfig) -> dict[str, Any]:
    if config.baseline_only:
        return {}
    params = live_check_kv_transfer_params(
        handoff_json=config.handoff_json,
        handoff_record=config.handoff_record,
        request_id=config.request_id,
        payload_uri=config.payload_uri,
        sglang_hicache_page_keys=config.sglang_hicache_page_keys,
        expected_backend=ServingBackend.SGLANG,
    )
    return params


def prepare_generated_live_handoff(config: SGLangSmokeBenchmarkConfig) -> SGLangSmokeBenchmarkConfig:
    """Generate the SGLang live-smoke handoff and page keys when requested."""

    generation = config.handoff_generation
    if generation is None:
        return config
    if config.venv_python.exists():
        record = _generate_live_handoff_inputs_in_subprocess(config, generation)
    else:
        generation.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            record = _generate_live_handoff_inputs(config, generation)
        finally:
            release_live_handoff_generation_resources()
    write_json(config.live_handoff_generation_path, record)
    return _config_with_generated_live_handoff(config, record)


def _generate_live_handoff_inputs_in_subprocess(
    config: SGLangSmokeBenchmarkConfig,
    generation: SGLangLiveHandoffGenerationConfig,
) -> dict[str, object]:
    input_path = config.local_dir / "sglang-live-handoff-generation-worker-input.json"
    output_path = config.local_dir / "sglang-live-handoff-generation-worker-output.json"
    payload: dict[str, object] = {
        "benchmark_id": config.benchmark_id,
        "output_dir": str(config.output_dir),
        "local_root": str(config.local_root),
        "hardware_target": config.hardware_target,
        "handoff_generation": generation.to_metadata(),
    }
    write_json(input_path, payload)
    code = """
import json
import sys
from pathlib import Path

from document_kv_cache.sglang_smoke import (
    SGLangLiveHandoffGenerationConfig,
    SGLangSmokeBenchmarkConfig,
    _generate_live_handoff_inputs,
    release_live_handoff_generation_resources,
    write_json,
)

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
payload = json.loads(input_path.read_text(encoding="utf-8"))
generation_payload = payload["handoff_generation"]
generation = SGLangLiveHandoffGenerationConfig(
    output_dir=Path(generation_payload["output_dir"]),
    generator_factory=generation_payload["generator_factory"],
    dtype=generation_payload["dtype"],
    align_bytes=int(generation_payload["align_bytes"]),
    page_size=int(generation_payload["sglang_hicache_page_size"]),
    timeout_seconds=float(generation_payload["timeout_seconds"]),
)
config = SGLangSmokeBenchmarkConfig(
    benchmark_id=payload["benchmark_id"],
    output_dir=Path(payload["output_dir"]),
    local_root=Path(payload["local_root"]),
    hardware_target=payload["hardware_target"],
    handoff_generation=generation,
)
try:
    record = _generate_live_handoff_inputs(config, generation)
finally:
    release_live_handoff_generation_resources()
record["generator_python"] = sys.executable
write_json(output_path, record)
"""
    argv = [
        str(config.venv_python),
        "-c",
        code,
        str(input_path),
        str(output_path),
    ]
    print("+", " ".join([argv[0], "-c", "<sglang live handoff generation>", *argv[3:]]), flush=True)
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=generation.timeout_seconds,
            env=server_env(config),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"SGLang live handoff generation timed out after {generation.timeout_seconds:.1f}s; "
            f"stdout_tail={tail_text(exc.stdout)!r}; stderr_tail={tail_text(exc.stderr)!r}"
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"SGLang live handoff generation failed with return code {completed.returncode}; "
            f"stdout_tail={tail_text(completed.stdout)!r}; stderr_tail={tail_text(completed.stderr)!r}"
        )
    if not output_path.exists():
        raise RuntimeError(f"SGLang live handoff generation did not write {output_path}")
    record = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise RuntimeError(f"SGLang live handoff generation wrote invalid result {output_path}")
    if record.get("ok") is not True:
        raise RuntimeError(f"SGLang live handoff generation worker result was not ok in {output_path}")
    return record


def _generate_live_handoff_inputs(
    config: SGLangSmokeBenchmarkConfig,
    generation: SGLangLiveHandoffGenerationConfig,
) -> dict[str, object]:
    generation.output_dir.mkdir(parents=True, exist_ok=True)
    generator = load_benchmark_kv_chunk_generator(generation.generator_factory)
    layout = layout_for_model(
        SERVED_MODEL_NAME,
        dtype=generation.dtype,
        block_size=generation.page_size,
    )
    live_request = build_live_server_check_request(
        model_id=SERVED_MODEL_NAME,
        hardware_target=config.hardware_target,
        use_cache_arm=True,
    )
    source_document = benchmark_cache_source_document(
        live_request.example,
        prefix=LIVE_HANDOFF_CACHE_ARTIFACT_PREFIX,
    )
    source_chunk = source_document.chunks[0]
    page_keys = sglang_hicache_page_keys(
        _token_ids_for_generator(generator, source_chunk.text),
        page_size=generation.page_size,
    )
    if not page_keys:
        raise RuntimeError("SGLang live handoff generation produced no HiCache page keys")

    request_id = f"{LIVE_HANDOFF_CACHE_ARTIFACT_PREFIX}-{config.benchmark_id}"
    document_request = benchmark_cache_request(
        live_request.example,
        model_id=SERVED_MODEL_NAME,
        request_id=request_id,
        prefix=LIVE_HANDOFF_CACHE_ARTIFACT_PREFIX,
    )
    cache_config = CacheBuildConfig(
        model_id=layout.model_id,
        lora_id=layout.lora_id,
        prompt_template_version=DEFAULT_V1_PROMPT_TEMPLATE_VERSION,
        dtype=layout.dtype,
        layout_version=layout.layout_version,
        storage_layout=layout.storage_layout,
    )
    workflow = DocumentKVWorkflow.with_storage(manifest=InMemoryManifestStore())
    source_shard_uri = f"disk:{generation.output_dir / 'sglang-live-source.kvpack'}"
    cache_generation = workflow.generate_cache(
        documents=(source_document,),
        generator=generator,
        config=cache_config,
        shard_uri=source_shard_uri,
        align_bytes=generation.align_bytes,
    )
    ready = workflow.prepare_for_engine(
        document_request,
        layout=layout,
        metadata={
            "cachet.live_smoke": "sglang",
            "cachet.live_smoke.benchmark_id": config.benchmark_id,
        },
        cache_method=cache_generation.cache_method,
        training_artifacts=cache_generation.training_artifacts,
    )
    adapter_request = build_engine_adapter_request(ready, spec=sglang_adapter_spec())
    handoff_json = generation.output_dir / "sglang-live.handoff.json"
    payload_uri = f"disk:{generation.output_dir / 'sglang-live.kv'}"
    handoff_path, payload_path = write_engine_adapter_handoff_bundle(
        adapter_request,
        handoff_json,
        payload_uri=payload_uri,
        require_external_payload_uri=True,
    )
    return {
        "ok": True,
        "backend": ServingBackend.SGLANG.value,
        "benchmark_id": config.benchmark_id,
        "request_id": request_id,
        "handoff_json": str(handoff_path),
        "payload_uri": payload_uri,
        "payload_path": str(payload_path),
        "source_shard_uri": source_shard_uri,
        "sglang_hicache_page_keys": list(page_keys),
        "sglang_hicache_page_size": generation.page_size,
        "cache_prefix_chars": len(source_chunk.text),
        "cache_prefix_tokens": ready.handle.total_tokens,
        "generator_factory": generation.generator_factory,
        "output_dir": str(generation.output_dir),
        "dtype": generation.dtype,
        "align_bytes": generation.align_bytes,
    }


def release_live_handoff_generation_resources() -> None:
    """Release best-effort Transformers/Torch memory before SGLang starts."""

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    cuda = getattr(torch, "cuda", None)
    empty_cache = getattr(cuda, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def _config_with_generated_live_handoff(
    config: SGLangSmokeBenchmarkConfig,
    record: Mapping[str, object],
) -> SGLangSmokeBenchmarkConfig:
    if record.get("ok") is not True:
        raise RuntimeError("SGLang live handoff generation did not report ok=true")
    handoff_json = record.get("handoff_json")
    payload_uri = record.get("payload_uri")
    request_id = record.get("request_id")
    if not isinstance(handoff_json, str) or not handoff_json:
        raise RuntimeError("SGLang live handoff generation must return handoff_json")
    if not isinstance(payload_uri, str) or not payload_uri:
        raise RuntimeError("SGLang live handoff generation must return payload_uri")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError("SGLang live handoff generation must return request_id")
    page_keys = _string_tuple(
        record.get("sglang_hicache_page_keys"),
        field_name="sglang_hicache_page_keys",
    )
    if not page_keys:
        raise RuntimeError("SGLang live handoff generation must return at least one HiCache page key")
    return replace(
        config,
        handoff_json=handoff_json,
        handoff_record=None,
        payload_uri=payload_uri,
        request_id=request_id,
        sglang_hicache_page_keys=page_keys,
        handoff_generation=None,
    )


def _token_ids_for_generator(generator: object, text: str) -> tuple[int, ...]:
    tokenizer = getattr(generator, "tokenizer", None)
    if tokenizer is None:
        raise TypeError("SGLang live handoff generator must expose tokenizer")
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=bool(getattr(generator, "add_special_tokens", False)),
    )
    if isinstance(encoded, Mapping):
        input_ids = encoded.get("input_ids")
    else:
        input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None:
        raise ValueError("SGLang live handoff tokenizer output must include input_ids")
    return _flatten_token_ids(input_ids)


def _flatten_token_ids(value: object) -> tuple[int, ...]:
    flattened: list[int] = []
    _append_token_ids(flattened, value)
    if not flattened:
        raise ValueError("SGLang live handoff tokenizer must return at least one token id")
    return tuple(flattened)


def _append_token_ids(flattened: list[int], value: object) -> None:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    reshape = getattr(value, "reshape", None)
    if callable(reshape):
        try:
            value = reshape(-1)
        except TypeError:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError("SGLang live handoff token ids must be integers")
    if isinstance(value, Sequence):
        for item in value:
            _append_token_ids(flattened, item)
        return
    if type(value) is not int:
        raise ValueError("SGLang live handoff token ids must be integers")
    flattened.append(value)


def dependency_constraints() -> list[str]:
    return list(SGLANG_SERVING_ENVIRONMENT_PROFILE.dependency_constraints)


def document_kv_package_install_spec(config: SGLangSmokeBenchmarkConfig) -> str:
    if config.package_install_spec is not None:
        return _cluster_file_path(config.package_install_spec)
    env_value = os.environ.get(DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV)
    if env_value is not None:
        if not env_value.strip():
            raise ValueError(f"{DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV} must be non-empty when set")
        return _cluster_file_path(env_value)
    source_root = _source_checkout_root()
    if source_root is not None:
        return str(source_root)
    raise RuntimeError(
        "SGLang smoke benchmark requires a Cachet package install spec for the isolated SGLang environment; "
        f"set {DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV} or pass --package-install-spec"
    )


def run_live_checks(
    config: SGLangSmokeBenchmarkConfig,
    *,
    import_probe_record: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    baseline = run_openai_compatible_live_check(
        LiveServerCheckConfig(
            base_url=config.server_base_url,
            model_id=SERVED_MODEL_NAME,
            hardware_target=config.hardware_target,
            max_tokens=config.max_tokens,
            timeout_seconds=config.timeout_seconds,
            stream=config.stream,
            prompt_token_accounting="server_usage",
        )
    )
    baseline_record = baseline.to_record()
    baseline_record["label"] = "baseline_prefill"
    cache_record: dict[str, object] | None = None
    if not config.baseline_only:
        cache = run_openai_compatible_live_check(
            LiveServerCheckConfig(
                base_url=config.server_base_url,
                model_id=SERVED_MODEL_NAME,
                hardware_target=config.hardware_target,
                use_cache_arm=True,
                prompt_text_mode=config.cache_prompt_text_mode,
                prompt_token_accounting="server_usage",
                kv_transfer_params_transport="custom_params",
                kv_transfer_params=sglang_live_kv_transfer_params(config),
                max_tokens=config.max_tokens,
                timeout_seconds=config.timeout_seconds,
                stream=config.stream,
            )
        )
        cache_record = cache.to_record()
        cache_record["label"] = "document_kv_cache"

    issues = []
    if baseline_record.get("ok") is not True:
        issues.append("baseline live check failed")
    if not config.baseline_only and (cache_record is None or cache_record.get("ok") is not True):
        issues.append("cache-arm live check failed")
    import_probe_metadata = _metadata_from_import_probe(import_probe_record)
    if not config.baseline_only and import_probe_metadata["live_request_metadata_bridge_ok"] is not True:
        issues.append("request metadata bridge not verified")
    record = {
        "ok": not issues,
        "benchmark_id": config.benchmark_id,
        "engine": "sglang",
        "model_id": SERVED_MODEL_NAME,
        "hardware_target": config.hardware_target,
        "baseline_only": config.baseline_only,
        "cache_arm_supported": not config.baseline_only,
        "cache_arm_blocker": None if not config.baseline_only else SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE,
        "live_request_metadata_bridge_required": True,
        "live_request_metadata_bridge_ok": import_probe_metadata["live_request_metadata_bridge_ok"],
        "sglang_import_probe_ok": import_probe_metadata["sglang_import_probe_ok"],
        "document_kv_request_metadata_bridge": import_probe_metadata["document_kv_request_metadata_bridge"],
        "requires_kv_transfer_params": not config.baseline_only,
        "kv_transfer_params_transport": "custom_params",
        "cache_prompt_text_mode": config.cache_prompt_text_mode,
        "baseline": baseline_record,
        "cache": cache_record,
        "issues": issues,
    }
    write_json(config.live_smoke_output_path, record)
    if issues:
        raise RuntimeError(f"SGLang live smoke failed: {'; '.join(issues)}")
    return record


def build_sglang_server_args(config: SGLangSmokeBenchmarkConfig, python_executable: Path) -> list[str]:
    args = [
        str(python_executable),
        "-u",
        "-m",
        "sglang.launch_server",
        "--model-path",
        HF_MODEL_ID,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        config.server_host,
        "--port",
        str(config.server_port),
        "--context-length",
        str(config.context_length),
        "--mem-fraction-static",
        str(config.mem_fraction_static),
        "--trust-remote-code",
    ]
    extra_config: dict[str, Any] = {}
    if config.hicache_page_store_uri is not None:
        extra_config[DOCUMENT_KV_HICACHE_PAGE_STORE_URI_CONFIG_KEY] = config.hicache_page_store_uri
    args.extend(
        sglang_hicache_cli_args(
            extra_config=extra_config,
            page_size=config.hicache_page_size,
            hicache_ratio=config.hicache_ratio,
            hicache_size_gb=config.hicache_size_gb,
            hicache_io_backend=config.hicache_io_backend,
            hicache_mem_layout=config.hicache_mem_layout,
            hicache_storage_prefetch_policy=config.hicache_storage_prefetch_policy,
            hicache_write_policy=config.hicache_write_policy,
        )
    )
    return args


def start_sglang_server(
    config: SGLangSmokeBenchmarkConfig,
    python_executable: Path,
    log_path: Path,
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = build_sglang_server_args(config, python_executable)
    print("+", " ".join(argv), flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        return subprocess.Popen(argv, stdout=log_handle, stderr=subprocess.STDOUT, text=True, env=server_env(config))


def wait_for_sglang_server(
    server: subprocess.Popen,
    log_path: Path,
    config: SGLangSmokeBenchmarkConfig,
    *,
    timeout_seconds: float = 900.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    health_url = f"{config.server_base_url}/health"
    models_url = f"{config.server_base_url}/v1/models"
    last_model_error = ""
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"SGLang server exited with {server.returncode}; log tail:\n{tail(log_path)}")
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if 200 <= response.status < 300:
                    model_ids = fetch_served_model_ids(models_url)
                    if SERVED_MODEL_NAME in model_ids:
                        return
                    last_model_error = f"health OK but served models were {sorted(model_ids)!r}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError) as exc:
            last_model_error = str(exc)
        time.sleep(5)
    raise TimeoutError(
        f"Timed out waiting for SGLang model {SERVED_MODEL_NAME!r} at {config.server_base_url}; "
        f"last readiness error: {last_model_error}; log tail:\n{tail(log_path)}"
    )


def fetch_served_model_ids(models_url: str) -> set[str]:
    with urllib.request.urlopen(models_url, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {
        item["id"]
        for item in payload["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }


def install_sglang(python_executable: Path) -> None:
    run([str(python_executable), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([str(python_executable), "-m", "pip", "install", *dependency_constraints()])


def install_document_kv_package(python_executable: Path, install_spec: str) -> None:
    run([str(python_executable), "-m", "pip", "install", "--no-deps", install_spec])


def probe_sglang_import(
    python_executable: Path,
    output_path: Path,
    *,
    launch_config_path: Path,
    timeout_seconds: float,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    code = """
import importlib.metadata as md
import json
import sys
from pathlib import Path

import document_kv_cache
import sglang
import sglang.launch_server
import sglang_kv_injection.sglang_dynamic_backend as document_kv_sglang_backend
from document_kv_cache.sglang_smoke import build_sglang_hicache_provider_probe_record

launch_config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
payload = {
    "ok": True,
    "sglang_version": md.version("sglang"),
    "document_kv_cache_version": md.version("cachet-kv"),
    "document_kv_cache_module": document_kv_cache.__name__,
    "document_kv_hicache_backend_module": document_kv_sglang_backend.__name__,
}
try:
    import torch
except Exception as exc:
    payload["torch_import_error"] = str(exc)
else:
    payload["torch_version"] = torch.__version__
    payload["cuda_available"] = torch.cuda.is_available()
    payload["cuda_device_count"] = torch.cuda.device_count()
    if torch.cuda.is_available():
        payload["cuda_device_name"] = torch.cuda.get_device_name(0)
payload.update(build_sglang_hicache_provider_probe_record(launch_config))
print(json.dumps(payload, sort_keys=True), flush=True)
"""
    argv = [str(python_executable), "-c", code, str(launch_config_path)]
    print("+", " ".join([argv[0], "-c", "<sglang import probe>", argv[3]]), flush=True)
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env or os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        record = {
            "ok": False,
            "error_type": "TimeoutExpired",
            "error": f"SGLang import probe timed out after {timeout_seconds:.1f}s",
            "stdout_tail": tail_text(exc.stdout),
            "stderr_tail": tail_text(exc.stderr),
        }
        write_json(output_path, record)
        raise RuntimeError(record["error"]) from exc

    record = {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout_tail": tail_text(completed.stdout),
        "stderr_tail": tail_text(completed.stderr),
    }
    if completed.returncode == 0:
        record.update(last_json_object(completed.stdout))
        if record.get("document_kv_request_metadata_bridge_ok") is not True:
            record["ok"] = False
            record["error_type"] = "SGLangRequestMetadataBridgeUnavailable"
            record["error"] = "SGLang request metadata bridge did not report ok=true"
    write_json(output_path, record)
    if completed.returncode != 0:
        raise RuntimeError(f"SGLang import probe failed with return code {completed.returncode}")
    if record.get("ok") is not True:
        raise RuntimeError(str(record.get("error", "SGLang import probe failed release checks")))
    return record


def create_venv(venv_dir: Path) -> None:
    if venv_python(venv_dir).exists():
        return
    try:
        run([sys.executable, "-m", "venv", str(venv_dir)])
    except subprocess.CalledProcessError:
        run([sys.executable, "-m", "pip", "install", "virtualenv"])
        run([sys.executable, "-m", "virtualenv", str(venv_dir)])


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def installed_versions(python_executable: Path) -> dict[str, str]:
    return {
        "sglang_version_installed": installed_package_version(python_executable, "sglang"),
        "document_kv_cache_version_installed": installed_package_version(python_executable, "cachet-kv"),
        "torch_version_installed": installed_package_version(python_executable, "torch"),
    }


def installed_package_version(python_executable: Path, package_name: str) -> str:
    completed = subprocess.run(
        [str(python_executable), "-m", "pip", "show", package_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "unknown"
    for line in completed.stdout.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def run(argv: list[str]) -> None:
    print("+", " ".join(argv), flush=True)
    subprocess.run(argv, check=True)


def server_env(config: SGLangSmokeBenchmarkConfig) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HOME"] = str(config.hf_cache_dir)
    return env


def terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_file_if_exists(source_path: Path, target_path: Path) -> None:
    if source_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_path)


def last_json_object(text: str) -> dict[str, object]:
    for line in reversed(text.splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def tail_text(text: str | bytes | None, *, max_chars: int = 12000) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return text[-max_chars:]


def tail(path: Path, *, lines: int = 120) -> str:
    if not path.exists():
        return "<missing log>"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def parse_args(argv: list[str] | None = None) -> SGLangSmokeBenchmarkConfig:
    parser = argparse.ArgumentParser(description="Run a Qwen3/SGLang live Cachet smoke on Databricks g5/g6.")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--import-probe-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--server-start-timeout-seconds", type=float, default=480.0)
    parser.add_argument("--local-root", default=str(DEFAULT_LOCAL_ROOT))
    parser.add_argument("--server-host", default=SERVER_HOST)
    parser.add_argument("--server-port", type=int, default=SERVER_PORT)
    parser.add_argument("--client-host", default=SERVER_HOST)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--hardware-target", default=DEFAULT_HARDWARE_TARGET)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--package-install-spec")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--cache-prompt-text-mode", choices=("logical", "runtime"), default="runtime")
    parser.add_argument("--handoff-json")
    parser.add_argument("--handoff-record-json")
    parser.add_argument("--payload-uri")
    parser.add_argument("--request-id")
    parser.add_argument("--sglang-hicache-page-keys-json")
    parser.add_argument(
        "--generate-live-handoff",
        action="store_true",
        help="Generate the live SGLang handoff and HiCache page keys before starting the server.",
    )
    parser.add_argument("--live-handoff-output-dir")
    parser.add_argument(
        "--live-handoff-generator-factory",
        default=DEFAULT_SGLANG_LIVE_HANDOFF_GENERATOR_FACTORY,
    )
    parser.add_argument("--live-handoff-dtype", default="bfloat16")
    parser.add_argument("--live-handoff-align-bytes", type=int, default=4096)
    parser.add_argument("--sglang-hicache-page-size", type=int)
    parser.add_argument("--live-handoff-generation-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--hicache-page-store-uri")
    parser.add_argument("--hicache-ratio", type=float)
    parser.add_argument("--hicache-size-gb", type=int)
    parser.add_argument("--hicache-io-backend")
    parser.add_argument("--hicache-mem-layout")
    parser.add_argument("--hicache-storage-prefetch-policy")
    parser.add_argument("--hicache-write-policy")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    handoff_generation = _live_handoff_generation_from_args(args, output_dir)
    return SGLangSmokeBenchmarkConfig(
        benchmark_id=args.benchmark_id,
        output_dir=output_dir,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        import_probe_timeout_seconds=args.import_probe_timeout_seconds,
        server_start_timeout_seconds=args.server_start_timeout_seconds,
        local_root=Path(args.local_root),
        server_host=args.server_host,
        server_port=args.server_port,
        client_host=args.client_host,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
        hardware_target=args.hardware_target,
        stream=not args.no_stream,
        package_install_spec=args.package_install_spec,
        baseline_only=args.baseline_only,
        cache_prompt_text_mode=args.cache_prompt_text_mode,
        handoff_json=args.handoff_json,
        handoff_record=_json_object_option(args.handoff_record_json, "--handoff-record-json"),
        payload_uri=args.payload_uri,
        request_id=args.request_id,
        sglang_hicache_page_keys=_json_string_array_option(
            args.sglang_hicache_page_keys_json,
            "--sglang-hicache-page-keys-json",
        ),
        handoff_generation=handoff_generation,
        hicache_page_store_uri=args.hicache_page_store_uri,
        hicache_page_size=(
            handoff_generation.page_size if handoff_generation is not None else args.sglang_hicache_page_size
        ),
        hicache_ratio=args.hicache_ratio,
        hicache_size_gb=args.hicache_size_gb,
        hicache_io_backend=args.hicache_io_backend,
        hicache_mem_layout=args.hicache_mem_layout,
        hicache_storage_prefetch_policy=args.hicache_storage_prefetch_policy,
        hicache_write_policy=args.hicache_write_policy,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        run_sglang_live_smoke(parse_args(argv))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, sort_keys=True))
        return 1
    return 0


def _hicache_extra_config(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("hicache_storage_backend_extra_config")
    if isinstance(raw, str):
        decoded = json.loads(raw)
    else:
        decoded = raw
    if not isinstance(decoded, Mapping):
        raise ValueError("hicache_storage_backend_extra_config must decode to a JSON object")
    return dict(decoded)


def _json_object_option(value: str | None, option_name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    decoded = json.loads(value)
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{option_name} must decode to a JSON object")
    return decoded


def _metadata_from_import_probe(import_probe_record: Mapping[str, Any] | None) -> dict[str, object]:
    bridge_record = None
    if import_probe_record is not None:
        bridge_record = import_probe_record.get("document_kv_request_metadata_bridge")
    return {
        "sglang_import_probe_ok": import_probe_record is not None and import_probe_record.get("ok") is True,
        "live_request_metadata_bridge_ok": (
            import_probe_record is not None
            and import_probe_record.get("document_kv_request_metadata_bridge_ok") is True
        ),
        "document_kv_request_metadata_bridge": bridge_record if isinstance(bridge_record, Mapping) else None,
    }


def _json_string_array_option(value: str | None, option_name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    decoded = json.loads(value)
    return _string_tuple(decoded, field_name=option_name)


def _live_handoff_generation_from_args(
    args: argparse.Namespace,
    output_dir: Path,
) -> SGLangLiveHandoffGenerationConfig | None:
    if not args.generate_live_handoff:
        if args.live_handoff_output_dir is not None:
            raise ValueError("--live-handoff-output-dir requires --generate-live-handoff")
        return None
    return SGLangLiveHandoffGenerationConfig(
        output_dir=Path(args.live_handoff_output_dir) if args.live_handoff_output_dir else output_dir / "live-handoff",
        generator_factory=args.live_handoff_generator_factory,
        dtype=args.live_handoff_dtype,
        align_bytes=args.live_handoff_align_bytes,
        page_size=args.sglang_hicache_page_size or DEFAULT_SGLANG_HICACHE_PAGE_SIZE,
        timeout_seconds=args.live_handoff_generation_timeout_seconds,
    )


def _string_tuple(values: object, *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence of strings")
    if not isinstance(values, Sequence):
        raise ValueError(f"{field_name} must be a sequence of strings")
    items = tuple(values)
    for index, item in enumerate(items):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{field_name}[{index}] must be a non-empty string")
    return items


def _cluster_file_path(uri: str) -> str:
    if uri.startswith("dbfs:/"):
        return "/dbfs/" + uri.removeprefix("dbfs:/").lstrip("/")
    return uri


def _source_checkout_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "document_kv_cache").exists():
            return parent
    return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
