import json
import os
from pathlib import Path
import subprocess
import sys

from document_kv_cache.databricks_sglang_smoke_job import (
    DEFAULT_DATABRICKS_SGLANG_SMOKE_PURPOSE,
    DEFAULT_DATABRICKS_SGLANG_SMOKE_RUN_NAME,
    DEFAULT_DATABRICKS_SGLANG_SMOKE_TASK_KEY,
    DatabricksSGLangSmokeJobConfig,
    build_databricks_sglang_smoke_run_submit_payload,
    main,
    write_databricks_sglang_smoke_run_submit_json,
    write_databricks_sglang_smoke_runner_script,
)
from document_kv_cache.sglang_smoke import (
    DEFAULT_SGLANG_HICACHE_PAGE_SIZE,
    DEFAULT_SGLANG_HICACHE_STORAGE_PREFETCH_THRESHOLD,
    DEFAULT_SGLANG_LIVE_HANDOFF_GENERATOR_FACTORY,
    SGLANG_BASELINE_HANDOFF_FIELDS_UNSUPPORTED_MESSAGE,
    SGLANG_GENERATED_HANDOFF_EXPLICIT_FIELDS_UNSUPPORTED_MESSAGE,
    SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE,
)


WHEEL_URI = "/Volumes/catalog/schema/volume/wheels/cachet_kv-0.2.0-py3-none-any.whl"
SINGLE_USER_NAME = "user@example.com"
HANDOFF_JSON = "/Volumes/catalog/schema/volume/live/sglang-live.handoff.json"
PAGE_KEYS_JSON = '["page-a","page-b"]'


def test_build_databricks_sglang_smoke_payload_uses_single_node_g6_cluster():
    config = DatabricksSGLangSmokeJobConfig(
        benchmark_id="v1-sglang-smoke-001",
        output_dir="/Volumes/catalog/schema/volume/v1-sglang-smoke",
        runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
        node_type_id="g6.8xlarge",
        wheel_uri=WHEEL_URI,
        single_user_name=SINGLE_USER_NAME,
        max_tokens=48,
        timeout_seconds=300,
        import_probe_timeout_seconds=90,
        server_start_timeout_seconds=600,
        local_root="/local_disk0",
        server_host="0.0.0.0",
        server_port=8123,
        client_host="127.0.0.1",
        context_length=8192,
        mem_fraction_static=0.72,
        stream=False,
        baseline_only=True,
        hicache_page_store_uri="/local_disk0/cachet/sglang-hicache",
        hicache_size_gb=4,
        custom_tags={"team": "document-kv"},
    )

    payload = build_databricks_sglang_smoke_run_submit_payload(config)
    task = payload["tasks"][0]
    cluster = task["new_cluster"]

    assert payload["run_name"] == DEFAULT_DATABRICKS_SGLANG_SMOKE_RUN_NAME
    assert task["task_key"] == DEFAULT_DATABRICKS_SGLANG_SMOKE_TASK_KEY
    assert "libraries" not in task
    assert cluster["node_type_id"] == "g6.8xlarge"
    assert cluster["driver_node_type_id"] == "g6.8xlarge"
    assert cluster["data_security_mode"] == "SINGLE_USER"
    assert cluster["single_user_name"] == SINGLE_USER_NAME
    assert cluster["num_workers"] == 0
    assert cluster["custom_tags"]["ResourceClass"] == "SingleNode"
    assert cluster["custom_tags"]["purpose"] == DEFAULT_DATABRICKS_SGLANG_SMOKE_PURPOSE
    assert cluster["custom_tags"]["team"] == "document-kv"
    assert task["spark_python_task"] == {
        "python_file": "dbfs:/benchmarks/run_sglang_smoke.py",
        "parameters": [
            "--benchmark-id",
            "v1-sglang-smoke-001",
            "--output-dir",
            "/Volumes/catalog/schema/volume/v1-sglang-smoke",
            "--max-tokens",
            "48",
            "--timeout-seconds",
            "300",
            "--import-probe-timeout-seconds",
            "90",
            "--server-start-timeout-seconds",
            "600",
            "--local-root",
            "/local_disk0",
            "--server-host",
            "0.0.0.0",
            "--server-port",
            "8123",
            "--client-host",
            "127.0.0.1",
            "--context-length",
            "8192",
            "--mem-fraction-static",
            "0.72",
            "--hardware-target",
                "aws-g6-l4",
                "--cache-prompt-text-mode",
                "logical",
                "--no-stream",
            "--baseline-only",
            "--hicache-page-store-uri",
            "/local_disk0/cachet/sglang-hicache",
            "--hicache-size-gb",
            "4",
            "--hicache-storage-prefetch-threshold",
            str(DEFAULT_SGLANG_HICACHE_STORAGE_PREFETCH_THRESHOLD),
            "--package-wheel-uri",
            WHEEL_URI,
        ],
    }


def test_databricks_sglang_smoke_config_requires_handoff_and_page_keys_for_cache_arm():
    try:
        DatabricksSGLangSmokeJobConfig(
            benchmark_id="v1-sglang-smoke-001",
            output_dir="/Volumes/catalog/schema/volume/v1-sglang-smoke",
            runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
            single_user_name=SINGLE_USER_NAME,
        )
    except ValueError as exc:
        assert str(exc) == SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE
    else:
        raise AssertionError("expected missing handoff validation to fail")

    try:
        DatabricksSGLangSmokeJobConfig(
            benchmark_id="v1-sglang-smoke-001",
            output_dir="/Volumes/catalog/schema/volume/v1-sglang-smoke",
            runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
            single_user_name=SINGLE_USER_NAME,
            handoff_json=HANDOFF_JSON,
        )
    except ValueError as exc:
        assert str(exc) == SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE
    else:
        raise AssertionError("expected missing page-key validation to fail")

    config = DatabricksSGLangSmokeJobConfig(
        benchmark_id="v1-sglang-smoke-001",
        output_dir="/Volumes/catalog/schema/volume/v1-sglang-smoke",
        runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
        single_user_name=SINGLE_USER_NAME,
        handoff_json=HANDOFF_JSON,
        payload_uri="/Volumes/catalog/schema/volume/live/sglang-live.kv",
        request_id="cachet-live-sglang-1",
        sglang_hicache_page_keys_json=PAGE_KEYS_JSON,
    )

    parameters = build_databricks_sglang_smoke_run_submit_payload(config)["tasks"][0]["spark_python_task"][
        "parameters"
    ]
    assert "--baseline-only" not in parameters
    assert parameters[parameters.index("--handoff-json") + 1] == HANDOFF_JSON
    assert parameters[parameters.index("--sglang-hicache-page-keys-json") + 1] == PAGE_KEYS_JSON

    config = DatabricksSGLangSmokeJobConfig(
        benchmark_id="v1-sglang-baseline-001",
        output_dir="/Volumes/catalog/schema/volume/v1-sglang-baseline",
        runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
        single_user_name=SINGLE_USER_NAME,
        baseline_only=True,
    )

    parameters = build_databricks_sglang_smoke_run_submit_payload(config)["tasks"][0]["spark_python_task"][
        "parameters"
    ]
    assert "--baseline-only" in parameters
    assert "--handoff-json" not in parameters


def test_databricks_sglang_smoke_config_supports_generated_live_handoff_cache_arm():
    default_config = DatabricksSGLangSmokeJobConfig(
        benchmark_id="v1-sglang-generated-defaults",
        output_dir="/Volumes/catalog/schema/volume/v1-sglang-generated",
        runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
        single_user_name=SINGLE_USER_NAME,
        generate_live_handoff=True,
    )
    assert default_config.cache_prompt_text_mode == "logical"
    assert default_config.live_handoff_generator_factory == DEFAULT_SGLANG_LIVE_HANDOFF_GENERATOR_FACTORY
    assert default_config.sglang_hicache_page_size == DEFAULT_SGLANG_HICACHE_PAGE_SIZE
    assert (
        default_config.hicache_storage_prefetch_threshold
        == DEFAULT_SGLANG_HICACHE_STORAGE_PREFETCH_THRESHOLD
    )

    config = DatabricksSGLangSmokeJobConfig(
        benchmark_id="v1-sglang-generated-001",
        output_dir="/Volumes/catalog/schema/volume/v1-sglang-generated",
        runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
        node_type_id="g6.8xlarge",
        single_user_name=SINGLE_USER_NAME,
        generate_live_handoff=True,
        live_handoff_output_dir="/Volumes/catalog/schema/volume/v1-sglang-generated/live-handoff",
        live_handoff_generator_factory="module:factory",
        live_handoff_dtype="float16",
        live_handoff_align_bytes=8,
        sglang_hicache_page_size=2,
        live_handoff_generation_timeout_seconds=12.5,
        spark_env_vars={"CACHET_TRANSFORMERS_DEVICE": "cuda"},
    )

    payload = build_databricks_sglang_smoke_run_submit_payload(config)
    cluster = payload["tasks"][0]["new_cluster"]
    parameters = payload["tasks"][0]["spark_python_task"]["parameters"]

    assert cluster["node_type_id"] == "g6.8xlarge"
    assert cluster["spark_env_vars"] == {"CACHET_TRANSFORMERS_DEVICE": "cuda"}
    assert "--baseline-only" not in parameters
    assert "--handoff-json" not in parameters
    assert "--sglang-hicache-page-keys-json" not in parameters
    assert "--generate-live-handoff" in parameters
    assert parameters[parameters.index("--live-handoff-output-dir") + 1].endswith("/live-handoff")
    assert parameters[parameters.index("--live-handoff-generator-factory") + 1] == "module:factory"
    assert parameters[parameters.index("--live-handoff-dtype") + 1] == "float16"
    assert parameters[parameters.index("--live-handoff-align-bytes") + 1] == "8"
    assert parameters[parameters.index("--sglang-hicache-page-size") + 1] == "2"
    assert parameters[parameters.index("--live-handoff-generation-timeout-seconds") + 1] == "12.5"
    assert parameters[parameters.index("--hicache-storage-prefetch-threshold") + 1] == str(
        DEFAULT_SGLANG_HICACHE_STORAGE_PREFETCH_THRESHOLD
    )


def test_databricks_sglang_smoke_config_validates_cluster_and_runtime_fields():
    invalid_cases = [
        ({"context_length": 0, "baseline_only": True}, "context_length must be positive"),
        ({"mem_fraction_static": 0, "baseline_only": True}, "mem_fraction_static must be in"),
        ({"cache_prompt_text_mode": "full", "baseline_only": True}, "cache_prompt_text_mode"),
        (
            {"handoff_json": HANDOFF_JSON, "handoff_record_json": "{}", "baseline_only": True},
            "only one of handoff_json",
        ),
        ({"handoff_record_json": "[]", "baseline_only": True}, "handoff_record_json must decode"),
        (
            {"sglang_hicache_page_keys_json": PAGE_KEYS_JSON, "baseline_only": True},
            SGLANG_BASELINE_HANDOFF_FIELDS_UNSUPPORTED_MESSAGE,
        ),
        (
            {"sglang_hicache_page_keys_json": "[]", "baseline_only": True},
            SGLANG_BASELINE_HANDOFF_FIELDS_UNSUPPORTED_MESSAGE,
        ),
        (
            {"sglang_hicache_page_keys_json": '"page-a"', "baseline_only": False},
            "sglang_hicache_page_keys_json must decode",
        ),
        (
            {"sglang_hicache_page_keys_json": PAGE_KEYS_JSON, "baseline_only": False},
            SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE,
        ),
        (
            {"handoff_json": HANDOFF_JSON, "sglang_hicache_page_keys_json": "[]", "baseline_only": False},
            SGLANG_HANDOFF_BINDING_UNSUPPORTED_MESSAGE,
        ),
        (
            {"handoff_json": HANDOFF_JSON, "baseline_only": True},
            SGLANG_BASELINE_HANDOFF_FIELDS_UNSUPPORTED_MESSAGE,
        ),
        (
            {"generate_live_handoff": True, "baseline_only": True},
            SGLANG_BASELINE_HANDOFF_FIELDS_UNSUPPORTED_MESSAGE,
        ),
        (
            {"generate_live_handoff": True, "handoff_json": HANDOFF_JSON, "baseline_only": False},
            SGLANG_GENERATED_HANDOFF_EXPLICIT_FIELDS_UNSUPPORTED_MESSAGE,
        ),
        ({"generate_live_handoff": True, "live_handoff_generator_factory": "", "baseline_only": False}, "factory"),
        ({"generate_live_handoff": True, "live_handoff_align_bytes": 0, "baseline_only": False}, "align"),
        ({"generate_live_handoff": True, "sglang_hicache_page_size": 0, "baseline_only": False}, "page_size"),
        (
            {"hicache_storage_prefetch_threshold": 0, "baseline_only": True},
            "hicache_storage_prefetch_threshold",
        ),
        ({"spark_env_vars": {"DATABRICKS_TOKEN": "redacted"}, "baseline_only": True}, "looks secret-bearing"),
    ]

    for overrides, message in invalid_cases:
        kwargs = {
            "benchmark_id": "v1-sglang-smoke-001",
            "output_dir": "/Volumes/catalog/schema/volume/v1-sglang-smoke",
            "runner_python_file": "dbfs:/benchmarks/run_sglang_smoke.py",
            "single_user_name": SINGLE_USER_NAME,
            "baseline_only": True,
        }
        kwargs.update(overrides)
        try:
            DatabricksSGLangSmokeJobConfig(**kwargs)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"expected validation to fail for {overrides!r}")


def test_write_databricks_sglang_smoke_runner_script_imports_smoke_main(tmp_path):
    path = tmp_path / "run_sglang_smoke.py"

    write_databricks_sglang_smoke_runner_script(path)

    runner_text = path.read_text(encoding="utf-8")
    assert "--package-wheel-uri" in runner_text
    assert "DOCUMENT_KV_PACKAGE_INSTALL_SPEC" in runner_text
    assert "pip\", \"install\"" in runner_text
    assert "dbfs:/" in runner_text
    assert "document_kv_cache.sglang_smoke" in runner_text
    assert "if exit_code:" in runner_text


def test_generated_sglang_smoke_runner_installs_wheel_before_forwarding_args(tmp_path):
    runner_path = tmp_path / "run_sglang_smoke.py"
    pip_call_path = tmp_path / "pip-call.json"
    main_args_path = tmp_path / "main-args.json"
    events_path = tmp_path / "events.jsonl"
    package_dir = tmp_path / "document_kv_cache"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "sglang_smoke.py").write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "",
                "with open(os.environ['RUNNER_EVENTS_JSONL'], 'a', encoding='utf-8') as handle:",
                "    handle.write(json.dumps({'event': 'sglang_smoke_import'}) + '\\n')",
                "",
                "def main(argv=None):",
                "    with open(os.environ['RUNNER_EVENTS_JSONL'], 'a', encoding='utf-8') as handle:",
                "        handle.write(json.dumps({'event': 'main'}) + '\\n')",
                "    with open(os.environ['MAIN_ARGS_JSON'], 'w', encoding='utf-8') as handle:",
                "        json.dump({",
                "            'argv': argv,",
                "            'package_install_spec': os.environ.get('DOCUMENT_KV_PACKAGE_INSTALL_SPEC'),",
                "        }, handle)",
                "    return 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "sitecustomize.py").write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "import subprocess",
                "",
                "def _capture_check_call(argv):",
                "    with open(os.environ['RUNNER_EVENTS_JSONL'], 'a', encoding='utf-8') as handle:",
                "        handle.write(json.dumps({'event': 'pip_install'}) + '\\n')",
                "    with open(os.environ['PIP_CALL_JSON'], 'w', encoding='utf-8') as handle:",
                "        json.dump(argv, handle)",
                "    return 0",
                "",
                "subprocess.check_call = _capture_check_call",
                "",
            ]
        ),
        encoding="utf-8",
    )

    write_databricks_sglang_smoke_runner_script(runner_path)
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "PIP_CALL_JSON": str(pip_call_path),
        "MAIN_ARGS_JSON": str(main_args_path),
        "RUNNER_EVENTS_JSONL": str(events_path),
    }

    subprocess.run(
        [
            sys.executable,
            str(runner_path),
            "--package-wheel-uri",
            "dbfs:/tmp/cachet/cachet_kv-0.2.0-py3-none-any.whl",
            "--benchmark-id",
            "v1-sglang-smoke-001",
            "--output-dir",
            "/dbfs/tmp/cachet/output",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    pip_call = json.loads(pip_call_path.read_text(encoding="utf-8"))
    assert Path(pip_call[0]).resolve() == Path(sys.executable).resolve()
    assert pip_call[1:] == [
        "-m",
        "pip",
        "install",
        "/dbfs/tmp/cachet/cachet_kv-0.2.0-py3-none-any.whl",
    ]
    main_payload = json.loads(main_args_path.read_text(encoding="utf-8"))
    assert main_payload == {
        "argv": [
            "--benchmark-id",
            "v1-sglang-smoke-001",
            "--output-dir",
            "/dbfs/tmp/cachet/output",
        ],
        "package_install_spec": "/dbfs/tmp/cachet/cachet_kv-0.2.0-py3-none-any.whl",
    }
    events = [json.loads(line)["event"] for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert events == ["pip_install", "sglang_smoke_import", "main"]


def test_write_databricks_sglang_smoke_run_submit_json_writes_payload(tmp_path):
    path = tmp_path / "payload.json"

    write_databricks_sglang_smoke_run_submit_json(
        DatabricksSGLangSmokeJobConfig(
            benchmark_id="v1-sglang-smoke-001",
            output_dir="/Volumes/catalog/schema/volume/v1-sglang-smoke",
            runner_python_file="dbfs:/benchmarks/run_sglang_smoke.py",
            single_user_name=SINGLE_USER_NAME,
            baseline_only=True,
        ),
        path,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["tasks"][0]["task_key"] == DEFAULT_DATABRICKS_SGLANG_SMOKE_TASK_KEY


def test_main_writes_sglang_smoke_payload_and_runner_script(tmp_path):
    payload_path = tmp_path / "payload.json"
    runner_path = tmp_path / "run_sglang_smoke.py"

    exit_code = main(
        [
            "--benchmark-id",
            "v1-sglang-smoke-001",
            "--output-dir",
            "/Volumes/catalog/schema/volume/v1-sglang-smoke",
            "--runner-python-file",
            "dbfs:/benchmarks/run_sglang_smoke.py",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--wheel-uri",
            WHEEL_URI,
            "--baseline-only",
            "--spark-env-var",
            "CACHET_SGLANG_TRACE=1",
            "--output-json",
            str(payload_path),
            "--runner-script-output",
            str(runner_path),
        ]
    )

    assert exit_code == 0
    task = json.loads(payload_path.read_text(encoding="utf-8"))["tasks"][0]
    assert "libraries" not in task
    assert task["spark_python_task"]["parameters"][-2:] == ["--package-wheel-uri", WHEEL_URI]
    assert task["new_cluster"]["spark_env_vars"] == {"CACHET_SGLANG_TRACE": "1"}
    assert "sglang_smoke" in runner_path.read_text(encoding="utf-8")


def test_main_derives_sglang_smoke_node_type_from_g5_hardware_target(tmp_path):
    payload_path = tmp_path / "payload.json"

    exit_code = main(
        [
            "--benchmark-id",
            "v1-sglang-smoke-001",
            "--output-dir",
            "/Volumes/catalog/schema/volume/v1-sglang-smoke",
            "--runner-python-file",
            "dbfs:/benchmarks/run_sglang_smoke.py",
            "--hardware-target",
            "aws-g5-a10g",
            "--single-user-name",
            SINGLE_USER_NAME,
            "--baseline-only",
            "--output-json",
            str(payload_path),
        ]
    )

    task = json.loads(payload_path.read_text(encoding="utf-8"))["tasks"][0]
    cluster = task["new_cluster"]
    parameters = task["spark_python_task"]["parameters"]
    assert exit_code == 0
    assert cluster["node_type_id"] == "g5.8xlarge"
    assert cluster["driver_node_type_id"] == "g5.8xlarge"
    assert parameters[parameters.index("--hardware-target") + 1] == "aws-g5-a10g"
