from pathlib import Path
import os
import subprocess
import sys
import urllib.error

import pytest

import document_kv_cache.vllm_smoke as public_vllm_smoke
import restaurant_kv_serving.vllm_smoke as legacy_vllm_smoke
from document_kv_cache.serving_env import VLLM_SERVING_ENVIRONMENT_PROFILE
from document_kv_cache.vllm_smoke import (
    FASTAPI_CONSTRAINT,
    HUGGINGFACE_HUB_CONSTRAINT,
    HF_MODEL_ID,
    NUMPY_CONSTRAINT,
    PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT,
    SERVED_MODEL_NAME,
    TOKENIZERS_CONSTRAINT,
    TRANSFORMERS_CONSTRAINT,
    VLLM_VERSION,
    VLLMSmokeBenchmarkConfig,
    build_benchmark_runner_args,
    build_metadata,
    build_vllm_server_args,
    dataset_args,
    dependency_constraints,
    parse_args,
    run_vllm_smoke_benchmark,
    smoke_dataset_records,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dependency_constraints_match_pinned_g5_vllm_stack():
    assert dependency_constraints() == list(VLLM_SERVING_ENVIRONMENT_PROFILE.dependency_constraints)
    assert all("==" in constraint for constraint in dependency_constraints())
    assert VLLM_VERSION == "0.23.0"
    assert TRANSFORMERS_CONSTRAINT == "transformers==5.12.1"
    assert HUGGINGFACE_HUB_CONSTRAINT == "huggingface-hub==1.20.1"
    assert TOKENIZERS_CONSTRAINT == "tokenizers==0.22.2"
    assert NUMPY_CONSTRAINT == "numpy==2.3.5"
    numpy_version = tuple(int(part) for part in NUMPY_CONSTRAINT.split("==", maxsplit=1)[1].split("."))
    assert (1, 25, 0) <= numpy_version < (2, 4, 0)
    assert FASTAPI_CONSTRAINT == "fastapi[standard]==0.136.0"
    fastapi_version = tuple(int(part) for part in FASTAPI_CONSTRAINT.split("==", maxsplit=1)[1].split("."))
    assert (0, 115, 0) <= fastapi_version < (0, 137, 0)
    assert PROMETHEUS_FASTAPI_INSTRUMENTATOR_CONSTRAINT == "prometheus-fastapi-instrumentator==8.0.0"
    assert HF_MODEL_ID == "Qwen/Qwen3-4B-Instruct-2507"
    assert SERVED_MODEL_NAME == "qwen3:4b-instruct"


def test_smoke_dataset_records_cover_v1_release_datasets():
    records = smoke_dataset_records()

    assert set(records) == {"biography", "hotpotqa", "musique", "niah"}
    assert records["biography"]["expected_answer"] == "Katherine Johnson"
    assert records["hotpotqa"]["expected_answer"] == "Paris"
    assert records["musique"]["expected_answer"] == "Ada Lovelace"
    assert records["niah"]["expected_answer"] == "cerulean lantern"
    assert all(record["documents"] for record in records.values())


def test_vllm_server_args_use_qwen3_instruct_and_g5_safe_limits(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        server_port=8123,
    )
    args = build_vllm_server_args(config, tmp_path / "venv" / "bin" / "python")

    assert args[:4] == [str(tmp_path / "venv" / "bin" / "python"), "-u", "-m", "vllm.entrypoints.openai.api_server"]
    assert args[args.index("--model") + 1] == HF_MODEL_ID
    assert args[args.index("--served-model-name") + 1] == SERVED_MODEL_NAME
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--port") + 1] == "8123"
    assert args[args.index("--dtype") + 1] == "bfloat16"
    assert args[args.index("--max-model-len") + 1] == "4096"
    assert args[args.index("--max-num-seqs") + 1] == "2"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.85"
    assert "--trust-remote-code" in args
    assert "--no-enable-log-requests" in args
    assert "--disable-log-requests" not in args


def test_benchmark_runner_args_include_all_smoke_datasets(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        max_tokens=32,
        timeout_seconds=240,
        local_root=tmp_path / "local",
        server_port=8123,
    )
    dataset_paths = {name: tmp_path / f"{name}.jsonl" for name in smoke_dataset_records()}

    args = build_benchmark_runner_args(config, dataset_paths)

    assert args[:3] == [sys.executable, "-m", "document_kv_cache.benchmark_runner"]
    assert args[args.index("--suite-id") + 1] == "smoke-1"
    assert args[args.index("--base-url") + 1] == "http://127.0.0.1:8123"
    assert args[args.index("--model-id") + 1] == SERVED_MODEL_NAME
    assert args[args.index("--hardware-target") + 1] == "aws-g6-l4"
    assert args[args.index("--output-json") + 1] == str(tmp_path / "out" / "v1-benchmark.json")
    assert "--server-usage" in args
    assert dataset_args(dataset_paths) == [
        "--dataset",
        f"biography={tmp_path / 'biography.jsonl'}",
        "--dataset",
        f"hotpotqa={tmp_path / 'hotpotqa.jsonl'}",
        "--dataset",
        f"musique={tmp_path / 'musique.jsonl'}",
        "--dataset",
        f"niah={tmp_path / 'niah.jsonl'}",
    ]


def test_metadata_records_reproducible_smoke_context(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
    )

    metadata = build_metadata(config)

    assert metadata["benchmark_id"] == "smoke-1"
    assert metadata["hf_model_id"] == HF_MODEL_ID
    assert metadata["served_model_name"] == SERVED_MODEL_NAME
    assert metadata["server_bind_host"] == "127.0.0.1"
    assert metadata["server_client_host"] == "127.0.0.1"
    assert metadata["server_base_url"] == "http://127.0.0.1:8000"
    assert metadata["hf_home"] == str(tmp_path / "local" / "hf-cache")
    assert metadata["vllm_python"] == str(tmp_path / "local" / "document-kv-vllm-smoke-smoke-1" / "vllm-venv" / "bin" / "python")
    assert metadata["dependency_constraints"] == dependency_constraints()


def test_parse_args_builds_config_with_overrides(tmp_path):
    config = parse_args(
        [
            "--benchmark-id",
            "smoke-1",
            "--output-dir",
            str(tmp_path / "out"),
            "--local-root",
            str(tmp_path / "local"),
            "--max-tokens",
            "16",
            "--timeout-seconds",
            "12.5",
            "--import-probe-timeout-seconds",
            "9",
            "--server-start-timeout-seconds",
            "30",
            "--server-host",
            "0.0.0.0",
            "--server-port",
            "8123",
            "--client-host",
            "127.0.0.1",
        ]
    )

    assert config == VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        max_tokens=16,
        timeout_seconds=12.5,
        import_probe_timeout_seconds=9,
        server_start_timeout_seconds=30,
        server_host="0.0.0.0",
        server_port=8123,
        client_host="127.0.0.1",
    )


def test_vllm_smoke_config_validates_before_runtime_setup(tmp_path):
    invalid_cases = [
        ({"benchmark_id": ""}, "benchmark_id must be non-empty"),
        ({"max_tokens": 0}, "max_tokens must be positive"),
        ({"timeout_seconds": 0}, "timeout_seconds must be positive"),
        ({"import_probe_timeout_seconds": 0}, "import_probe_timeout_seconds must be positive"),
        ({"server_start_timeout_seconds": 0}, "server_start_timeout_seconds must be positive"),
        ({"server_host": ""}, "server_host must be non-empty"),
        ({"server_port": 0}, "server_port must be between 1 and 65535"),
        ({"server_port": 65536}, "server_port must be between 1 and 65535"),
        ({"client_host": ""}, "client_host must be non-empty"),
    ]

    for overrides, message in invalid_cases:
        kwargs = {
            "benchmark_id": "smoke-1",
            "output_dir": tmp_path / "out",
            "local_root": tmp_path / "local",
        }
        kwargs.update(overrides)
        with pytest.raises(ValueError, match=message):
            VLLMSmokeBenchmarkConfig(**kwargs)


def test_parse_args_rejects_invalid_values_before_setup(tmp_path):
    with pytest.raises(ValueError, match="server_port must be between"):
        parse_args(
            [
                "--benchmark-id",
                "smoke-1",
                "--output-dir",
                str(tmp_path / "out"),
                "--server-port",
                "0",
            ]
        )


def test_server_base_url_uses_client_host_not_bind_host(tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        server_host="0.0.0.0",
        server_port=8123,
    )

    server_args = build_vllm_server_args(config, tmp_path / "venv" / "bin" / "python")

    assert server_args[server_args.index("--host") + 1] == "0.0.0.0"
    assert config.server_base_url == "http://127.0.0.1:8123"
    assert build_metadata(config)["server_bind_host"] == "0.0.0.0"
    assert build_metadata(config)["server_client_host"] == "127.0.0.1"


def test_server_env_forces_local_root_hf_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", "/slow-or-wrong")
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
    )

    env = legacy_vllm_smoke.server_env(config)

    assert env["HF_HOME"] == str(tmp_path / "local" / "hf-cache")
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert env["PYTHONUNBUFFERED"] == "1"


class _FakeServer:
    returncode = None

    def poll(self):
        return None


class _FakeResponse:
    def __init__(self, *, status=200, payload=b""):
        self.status = status
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self._payload


def test_wait_for_server_requires_expected_served_model(monkeypatch, tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        server_port=8123,
    )
    requested_urls = []

    def fake_urlopen(url, timeout):
        requested_urls.append(url)
        if url.endswith("/health"):
            return _FakeResponse(status=200)
        if url.endswith("/v1/models"):
            return _FakeResponse(
                status=200,
                payload=b'{"data":[{"id":"qwen3:4b-instruct"}]}',
            )
        raise urllib.error.URLError(f"unexpected url {url}")

    monkeypatch.setattr(legacy_vllm_smoke.urllib.request, "urlopen", fake_urlopen)

    legacy_vllm_smoke.wait_for_server(
        _FakeServer(),
        tmp_path / "missing.log",
        config,
        timeout_seconds=1,
    )

    assert requested_urls == [
        "http://127.0.0.1:8123/health",
        "http://127.0.0.1:8123/v1/models",
    ]


def test_run_vllm_smoke_benchmark_orchestrates_and_cleans_up(monkeypatch, tmp_path):
    calls = []
    fake_server = object()
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
        server_port=8123,
    )
    dataset_paths = {name: tmp_path / f"{name}.jsonl" for name in smoke_dataset_records()}

    monkeypatch.setattr(public_vllm_smoke, "create_venv", lambda path: calls.append(("create_venv", path)))
    monkeypatch.setattr(public_vllm_smoke, "install_vllm", lambda python: calls.append(("install_vllm", python)))
    monkeypatch.setattr(
        public_vllm_smoke,
        "installed_versions",
        lambda python: {"vllm_version_installed": "0.23.0", "transformers_version_installed": "5.12.1"},
    )
    monkeypatch.setattr(
        public_vllm_smoke,
        "probe_vllm_import",
        lambda python, output, *, timeout_seconds, env: calls.append(
            ("probe_vllm_import", python, output, timeout_seconds, env["HF_HOME"])
        ),
    )
    monkeypatch.setattr(
        public_vllm_smoke,
        "write_smoke_datasets",
        lambda local_dir: calls.append(("write_smoke_datasets", local_dir)) or dataset_paths,
    )
    monkeypatch.setattr(
        public_vllm_smoke,
        "start_vllm_server",
        lambda cfg, python, log_path: calls.append(("start_vllm_server", cfg.server_base_url, python, log_path))
        or fake_server,
    )
    monkeypatch.setattr(
        public_vllm_smoke,
        "wait_for_server",
        lambda server, log_path, cfg, *, timeout_seconds: calls.append(
            ("wait_for_server", server, log_path, cfg.server_base_url, timeout_seconds)
        ),
    )
    monkeypatch.setattr(public_vllm_smoke, "run", lambda argv: calls.append(("run", argv)))
    monkeypatch.setattr(public_vllm_smoke, "terminate_process", lambda server: calls.append(("terminate", server)))
    monkeypatch.setattr(
        public_vllm_smoke,
        "copy_file_if_exists",
        lambda source, target: calls.append(("copy", source, target)),
    )

    run_vllm_smoke_benchmark(config)

    assert calls == [
        ("create_venv", config.venv_dir),
        ("install_vllm", config.venv_python),
        (
            "probe_vllm_import",
            config.venv_python,
            config.import_probe_path,
            180.0,
            str(tmp_path / "local" / "hf-cache"),
        ),
        ("write_smoke_datasets", config.local_dir),
        ("start_vllm_server", "http://127.0.0.1:8123", config.venv_python, config.server_log_path),
        ("wait_for_server", fake_server, config.server_log_path, "http://127.0.0.1:8123", 480.0),
        ("copy", config.server_log_path, config.server_log_copy_path),
        ("run", build_benchmark_runner_args(config, dataset_paths)),
        ("terminate", fake_server),
        ("copy", config.server_log_path, config.server_log_copy_path),
    ]
    metadata = build_metadata(config)
    assert metadata["server_base_url"] == "http://127.0.0.1:8123"
    assert metadata["hf_home"] == str(tmp_path / "local" / "hf-cache")


def test_legacy_vllm_smoke_run_respects_legacy_helper_monkeypatch(monkeypatch, tmp_path):
    config = VLLMSmokeBenchmarkConfig(
        benchmark_id="smoke-1",
        output_dir=tmp_path / "out",
        local_root=tmp_path / "local",
    )

    def fake_create_venv(path):
        raise RuntimeError(f"legacy hook used for {path.name}")

    monkeypatch.setattr(legacy_vllm_smoke, "create_venv", fake_create_venv)

    with pytest.raises(RuntimeError, match="legacy hook used"):
        legacy_vllm_smoke.run_vllm_smoke_benchmark(config)


def test_legacy_vllm_smoke_direct_helper_respects_legacy_run_monkeypatch(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(legacy_vllm_smoke, "run", lambda argv: calls.append(argv))

    legacy_vllm_smoke.create_venv(tmp_path / "venv")

    assert calls == [[sys.executable, "-m", "venv", str(tmp_path / "venv")]]


def test_legacy_vllm_smoke_main_respects_legacy_run_monkeypatch(monkeypatch, tmp_path):
    called = {}

    def fake_run(config):
        called["config"] = config

    monkeypatch.setattr(legacy_vllm_smoke, "run_vllm_smoke_benchmark", fake_run)

    exit_code = legacy_vllm_smoke.main(
        [
            "--benchmark-id",
            "smoke-1",
            "--output-dir",
            str(tmp_path / "out"),
            "--local-root",
            str(tmp_path / "local"),
        ]
    )

    assert exit_code == 0
    assert called["config"].benchmark_id == "smoke-1"
    assert called["config"].output_dir == tmp_path / "out"
    assert called["config"].local_root == tmp_path / "local"


def test_legacy_vllm_smoke_module_execution_shows_help():
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    completed = subprocess.run(
        [sys.executable, "-m", "restaurant_kv_serving.vllm_smoke", "--help"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Run a Qwen3/vLLM V1 benchmark smoke on Databricks g6/L4." in completed.stdout


def test_public_vllm_smoke_main_respects_document_namespace_monkeypatch(monkeypatch, tmp_path):
    called = {}

    def fake_run(config):
        called["config"] = config

    monkeypatch.setattr(public_vllm_smoke, "run_vllm_smoke_benchmark", fake_run)

    exit_code = public_vllm_smoke.main(
        [
            "--benchmark-id",
            "smoke-1",
            "--output-dir",
            str(tmp_path / "out"),
            "--local-root",
            str(tmp_path / "local"),
        ]
    )

    assert exit_code == 0
    assert called["config"].benchmark_id == "smoke-1"
    assert called["config"].output_dir == tmp_path / "out"
    assert called["config"].local_root == tmp_path / "local"
