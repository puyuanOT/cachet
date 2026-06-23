from pathlib import Path
import json
import subprocess

import pytest

import document_kv_cache.sglang_smoke as public_sglang_smoke
from document_kv_cache.benchmarks import (
    DOCUMENT_KV_REQUEST_ID_PARAM,
)
from document_kv_cache.engine import EngineReadyRequest
from document_kv_cache.engine_adapters import build_engine_adapter_request, engine_adapter_request_to_record, sglang_adapter_spec
from document_kv_cache.engine_protocol import KVCacheHandle, KVLayout, KVSegment
from document_kv_cache.sglang_smoke import (
    DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV,
    HF_MODEL_ID,
    SGLANG_BASELINE_HANDOFF_FIELDS_UNSUPPORTED_MESSAGE,
    SERVED_MODEL_NAME,
    SGLANG_DEPENDENCY_CONSTRAINTS,
    SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE,
    SGLANG_VERSION,
    SGLangSmokeBenchmarkConfig,
    build_metadata,
    build_sglang_hicache_provider_probe_record,
    build_sglang_server_args,
    dependency_constraints,
    document_kv_package_install_spec,
    install_document_kv_package,
    install_sglang,
    parse_args,
    run_live_checks,
    sglang_hicache_config_for_smoke,
)
from sglang_kv_injection.sglang_dynamic_backend import (
    DOCUMENT_KV_HICACHE_PAGE_STORE_URI_CONFIG_KEY,
    DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def handoff_record(*, request_id: str, payload_uri: str) -> dict[str, object]:
    layout = KVLayout(
        model_id="tiny-test-model",
        lora_id="base",
        layout_version="standard-v1",
        dtype="int8",
        num_layers=1,
        block_size=2,
        bytes_per_token=4,
    )
    handle = KVCacheHandle(
        request_id=request_id,
        handle_uri=f"document-kv://{request_id}",
        layout=layout,
        segments=(KVSegment("doc-1", "document_static", "static", 0, 1, 0, 4),),
        total_tokens=1,
        total_bytes=4,
    )
    ready = EngineReadyRequest(handle=handle, payload=b"data", estimated_gpu_bytes=4)
    adapter_request = build_engine_adapter_request(ready, spec=sglang_adapter_spec())
    return engine_adapter_request_to_record(adapter_request, payload_uri=payload_uri)


def write_handoff_json(path: Path, *, request_id: str, payload_uri: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(handoff_record(request_id=request_id, payload_uri=payload_uri), sort_keys=True),
        encoding="utf-8",
    )


class FakeLiveResult:
    def __init__(self, *, ok: bool, request_id: str | None, prompt_text_mode: str, cache_arm: bool) -> None:
        self.ok = ok
        self.request_id = request_id
        self.prompt_text_mode = prompt_text_mode
        self.cache_arm = cache_arm

    def to_record(self):
        return {
            "ok": self.ok,
            "request_id": self.request_id,
            "prompt_text_mode": self.prompt_text_mode,
            "arm_id": "document_kv_cache" if self.cache_arm else "baseline_prefill",
        }


def test_dependency_constraints_match_pinned_sglang_stack():
    assert dependency_constraints() == list(SGLANG_DEPENDENCY_CONSTRAINTS)
    assert dependency_constraints() == ["sglang==0.5.10.post1"]
    assert SGLANG_VERSION == "0.5.10.post1"
    assert HF_MODEL_ID == "Qwen/Qwen3-4B-Instruct-2507"
    assert SERVED_MODEL_NAME == "qwen3:4b-instruct"


def test_sglang_smoke_rejects_cache_arm_until_request_to_hicache_binding_exists(tmp_path):
    with pytest.raises(ValueError) as exc:
        SGLangSmokeBenchmarkConfig(benchmark_id="sglang-1", output_dir=tmp_path / "out")

    assert str(exc.value) == SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE

    handoff_path = tmp_path / "handoffs" / "sglang-live.handoff.json"
    payload_uri = f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}"
    write_handoff_json(handoff_path, request_id="cachet-live-sglang-1", payload_uri=payload_uri)
    with pytest.raises(ValueError) as exc:
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            handoff_json=str(handoff_path),
            payload_uri=payload_uri,
            request_id="cachet-live-sglang-1",
        )

    assert str(exc.value) == SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE


def test_sglang_smoke_accepts_baseline_only_without_handoff_fields(tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
    )

    assert config.baseline_only is True
    assert config.local_dir == Path("/local_disk0/document-kv-sglang-smoke-sglang-1")


def test_sglang_smoke_rejects_handoff_fields_for_baseline_only(tmp_path):
    handoff_path = tmp_path / "handoffs" / "sglang-live.handoff.json"
    write_handoff_json(
        handoff_path,
        request_id="cachet-live-sglang-1",
        payload_uri=f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}",
    )

    with pytest.raises(ValueError) as exc:
        SGLangSmokeBenchmarkConfig(
            benchmark_id="sglang-1",
            output_dir=tmp_path / "out",
            baseline_only=True,
            handoff_json=str(handoff_path),
        )

    assert str(exc.value) == SGLANG_BASELINE_HANDOFF_FIELDS_UNSUPPORTED_MESSAGE


def test_document_kv_package_install_spec_prefers_config_then_env(monkeypatch, tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
        package_install_spec="dbfs:/tmp/cachet/cachet_kv.whl",
    )

    assert document_kv_package_install_spec(config) == "/dbfs/tmp/cachet/cachet_kv.whl"

    monkeypatch.setenv(DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV, "dbfs:/tmp/cachet/from-env.whl")
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
    )

    assert document_kv_package_install_spec(config) == "/dbfs/tmp/cachet/from-env.whl"


def test_document_kv_package_install_spec_falls_back_to_source_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv(DOCUMENT_KV_PACKAGE_INSTALL_SPEC_ENV, raising=False)
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
    )

    assert document_kv_package_install_spec(config) == str(REPO_ROOT)


def test_install_sglang_and_cachet_package_use_pinned_constraints(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(public_sglang_smoke, "run", lambda argv: calls.append(argv))
    python = tmp_path / "venv" / "bin" / "python"

    install_sglang(python)
    install_document_kv_package(python, "/tmp/cachet.whl")

    assert calls == [
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        [str(python), "-m", "pip", "install", *dependency_constraints()],
        [str(python), "-m", "pip", "install", "--no-deps", "/tmp/cachet.whl"],
    ]


def test_sglang_server_args_use_qwen3_and_hicache_backend(tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
        server_port=8123,
        context_length=8192,
        mem_fraction_static=0.72,
        hicache_page_store_uri=f"disk:{tmp_path / 'hicache-pages'}",
        hicache_size_gb=4,
        hicache_write_policy="write_through_selective",
    )

    args = build_sglang_server_args(config, tmp_path / "venv" / "bin" / "python")

    assert args[:4] == [str(tmp_path / "venv" / "bin" / "python"), "-u", "-m", "sglang.launch_server"]
    assert args[args.index("--model-path") + 1] == HF_MODEL_ID
    assert args[args.index("--served-model-name") + 1] == SERVED_MODEL_NAME
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--port") + 1] == "8123"
    assert args[args.index("--context-length") + 1] == "8192"
    assert args[args.index("--mem-fraction-static") + 1] == "0.72"
    assert "--enable-hierarchical-cache" in args
    assert args[args.index("--hicache-storage-backend") + 1] == "dynamic"
    assert args[args.index("--hicache-size") + 1] == "4"
    assert args[args.index("--hicache-write-policy") + 1] == "write_through_selective"
    extra_config = json.loads(args[args.index("--hicache-storage-backend-extra-config") + 1])
    assert extra_config[DOCUMENT_KV_HICACHE_PAGE_STORE_URI_CONFIG_KEY].endswith("/hicache-pages")
    assert extra_config[DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY]
    assert extra_config["document_kv.requires_native_runtime"] is True


def test_sglang_hicache_provider_probe_rejects_noop_launch_config():
    launch_config = sglang_hicache_config_for_smoke(
        SGLangSmokeBenchmarkConfig(benchmark_id="sglang-1", output_dir=Path("/tmp/out"), baseline_only=True)
    )
    extra_config = json.loads(launch_config["hicache_storage_backend_extra_config"])
    extra_config.pop(DOCUMENT_KV_HICACHE_PROVIDER_FACTORY_CONFIG_KEY)
    launch_config["hicache_storage_backend_extra_config"] = json.dumps(extra_config)

    with pytest.raises(ValueError, match="provider_factory"):
        build_sglang_hicache_provider_probe_record(launch_config)


def test_sglang_hicache_provider_probe_accepts_builtin_provider(tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
        hicache_page_store_uri=f"disk:{tmp_path / 'pages'}",
    )

    record = build_sglang_hicache_provider_probe_record(sglang_hicache_config_for_smoke(config))

    assert record["document_kv_hicache_provider_ok"] is True
    assert record["document_kv_requires_native_runtime"] is True
    assert record["document_kv_provider_type"].endswith("DocumentKVHiCachePageProvider")


def test_sglang_smoke_cli_rejects_handoff_cache_arm_before_launch(monkeypatch, capsys, tmp_path):
    handoff_path = tmp_path / "handoffs" / "sglang-live.handoff.json"
    payload_uri = f"disk:{tmp_path / 'payloads' / 'sglang-live.kv'}"
    write_handoff_json(handoff_path, request_id="cachet-live-sglang-1", payload_uri=payload_uri)

    def fail_if_called(_config):
        raise AssertionError("SGLang smoke must fail before server launch")

    monkeypatch.setattr(public_sglang_smoke, "run_sglang_live_smoke", fail_if_called)

    exit_code = public_sglang_smoke.main(
        [
            "--benchmark-id",
            "sglang-1",
            "--output-dir",
            str(tmp_path / "out"),
            "--handoff-json",
            str(handoff_path),
            "--payload-uri",
            payload_uri,
            "--request-id",
            "cachet-live-sglang-1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["error_type"] == "ValueError"
    assert payload["error"] == SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE


def test_build_metadata_records_custom_params_transport(tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
        hardware_target="aws-g5-a10g",
    )

    metadata = build_metadata(config)

    assert metadata["hardware_target"] == "aws-g5-a10g"
    assert metadata["kv_transfer_params_transport"] == "custom_params"
    assert metadata["cache_prompt_text_mode"] == "runtime"
    assert metadata["requires_kv_transfer_params"] is False
    assert metadata["cache_arm_supported"] is False
    assert metadata["cache_arm_blocker"] == SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE


def test_run_live_checks_runs_baseline_only_and_records_cache_arm_blocker(monkeypatch, tmp_path):
    config = SGLangSmokeBenchmarkConfig(
        benchmark_id="sglang-1",
        output_dir=tmp_path / "out",
        baseline_only=True,
    )
    seen_configs = []

    def fake_live_check(live_config):
        seen_configs.append(live_config)
        return FakeLiveResult(
            ok=True,
            request_id=live_config.kv_transfer_params.get(DOCUMENT_KV_REQUEST_ID_PARAM),
            prompt_text_mode=live_config.prompt_text_mode,
            cache_arm=live_config.use_cache_arm,
        )

    monkeypatch.setattr(public_sglang_smoke, "run_openai_compatible_live_check", fake_live_check)

    record = run_live_checks(config)

    assert record["ok"] is True
    assert len(seen_configs) == 1
    assert seen_configs[0].use_cache_arm is False
    assert seen_configs[0].prompt_text_mode == "logical"
    assert record["cache"] is None
    assert record["requires_kv_transfer_params"] is False
    assert record["cache_arm_supported"] is False
    assert record["cache_arm_blocker"] == SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE
    written = json.loads(config.live_smoke_output_path.read_text(encoding="utf-8"))
    assert written["cache"] is None


def test_parse_args_builds_baseline_only_config(tmp_path):
    config = parse_args(
        [
            "--benchmark-id",
            "sglang-1",
            "--output-dir",
            str(tmp_path / "out"),
            "--baseline-only",
            "--hardware-target",
            "aws-g5-a10g",
            "--no-stream",
        ]
    )

    assert config.baseline_only is True
    assert config.hardware_target == "aws-g5-a10g"
    assert config.stream is False


def test_probe_sglang_import_writes_timeout_artifact(monkeypatch, tmp_path):
    def timeout_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=3, output="partial out", stderr="partial err")

    monkeypatch.setattr(public_sglang_smoke.subprocess, "run", timeout_run)
    output_path = tmp_path / "probe.json"

    with pytest.raises(RuntimeError, match="timed out"):
        public_sglang_smoke.probe_sglang_import(
            tmp_path / "python",
            output_path,
            launch_config_path=tmp_path / "launch.json",
            timeout_seconds=3,
        )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["ok"] is False
    assert record["error_type"] == "TimeoutExpired"
    assert "partial out" in record["stdout_tail"]
    assert "partial err" in record["stderr_tail"]
