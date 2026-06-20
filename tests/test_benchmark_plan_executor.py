import hashlib
import json
import os
import subprocess
import sys
from textwrap import dedent

import pytest

import document_kv_cache.benchmark_plan_executor as public_plan_executor
import restaurant_kv_serving.benchmark_plan_executor as legacy_plan_executor
from document_kv_cache.benchmark_plan_executor import (
    BENCHMARK_PLAN_EXECUTION_RECORD_TYPE,
    BENCHMARK_PLAN_SOURCE_RECORD_TYPE,
    benchmark_plan_source_payload_to_record,
    benchmark_plan_source_to_record,
    benchmark_command_results_to_record,
    execute_benchmark_job_plan,
    execute_benchmark_job_plan_json,
    main,
)
from restaurant_kv_serving.benchmark_plan_executor import _driver_path, _runtime_argv


def plan_for(argv):
    return {"commands": [{"name": "command-1", "argv": list(argv)}]}


def test_execute_benchmark_job_plan_dry_run_reports_skipped_commands():
    results = execute_benchmark_job_plan(plan_for((sys.executable, "-c", "raise SystemExit(3)")), dry_run=True)

    assert results[0].name == "command-1"
    assert results[0].returncode == 0
    assert results[0].skipped is True


def test_execute_benchmark_job_plan_runs_commands_in_order(tmp_path):
    output_path = tmp_path / "out.txt"
    plan = plan_for(
        (
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(output_path)!r}).write_text('ok', encoding='utf-8')",
        )
    )

    results = execute_benchmark_job_plan(plan)

    assert output_path.read_text(encoding="utf-8") == "ok"
    record = benchmark_command_results_to_record(results)
    assert record["record_type"] == BENCHMARK_PLAN_EXECUTION_RECORD_TYPE
    assert record["commands"][0]["returncode"] == 0


def test_benchmark_command_results_record_can_include_plan_source():
    plan_source = {"path": "dbfs:/benchmarks/plan.json", "sha256": "a" * 64}
    results = execute_benchmark_job_plan(plan_for((sys.executable, "-c", "pass")), dry_run=True)

    record = benchmark_command_results_to_record(results, plan_source=plan_source)

    assert record["record_type"] == BENCHMARK_PLAN_EXECUTION_RECORD_TYPE
    assert record["plan_source"] == plan_source


def test_benchmark_plan_source_record_hashes_driver_visible_plan_json(tmp_path):
    path = tmp_path / "plan.json"
    plan = {
        "plan_version": "v1",
        "suite_id": "suite-1",
        "model_id": "qwen3:4b-instruct",
        "hardware_target": "aws-g5",
        "commands": [{"name": "command-1", "argv": [sys.executable, "-c", "pass"]}],
    }
    payload = json.dumps(plan, sort_keys=True).encode("utf-8")
    path.write_bytes(payload)

    record = benchmark_plan_source_to_record(path)

    assert record == {
        "record_type": BENCHMARK_PLAN_SOURCE_RECORD_TYPE,
        "path": str(path),
        "driver_path": str(path),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "plan_version": "v1",
        "suite_id": "suite-1",
        "model_id": "qwen3:4b-instruct",
        "hardware_target": "aws-g5",
        "command_count": 1,
    }


def test_main_hashes_the_plan_payload_that_was_executed(tmp_path):
    plan_path = tmp_path / "plan.json"
    result_path = tmp_path / "result.json"
    original_plan = {
        "plan_version": "v1",
        "suite_id": "original-suite",
        "model_id": "qwen3:4b-instruct",
        "hardware_target": "aws-g5",
        "commands": [
            {
                "name": "rewrite-plan",
                "argv": [
                    sys.executable,
                    "-c",
                    (
                        "import json; "
                        f"open({str(plan_path)!r}, 'w', encoding='utf-8').write("
                        "json.dumps({'commands': [], 'suite_id': 'mutated-suite'}))"
                    ),
                ],
            }
        ],
    }
    original_payload = json.dumps(original_plan, sort_keys=True).encode("utf-8")
    plan_path.write_bytes(original_payload)

    exit_code = main(["--plan-json", str(plan_path), "--result-json", str(result_path)])

    assert exit_code == 0
    record = json.loads(result_path.read_text(encoding="utf-8"))
    assert record["plan_source"]["suite_id"] == "original-suite"
    assert record["plan_source"]["sha256"] == hashlib.sha256(original_payload).hexdigest()
    assert json.loads(plan_path.read_text(encoding="utf-8"))["suite_id"] == "mutated-suite"


def test_execute_benchmark_job_plan_records_nonzero_return_code_and_stops():
    results = execute_benchmark_job_plan(
        {
            "commands": [
                {"name": "fail", "argv": [sys.executable, "-c", "raise SystemExit(7)"]},
                {"name": "skip", "argv": [sys.executable, "-c", "raise SystemExit(0)"]},
            ]
        }
    )
    record = benchmark_command_results_to_record(results)

    assert len(results) == 1
    assert record["ok"] is False
    assert record["commands"][0]["returncode"] == 7
    assert record["commands"][0]["error"] is None


def test_execute_benchmark_job_plan_json_loads_plan_file(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan_for((sys.executable, "-c", "pass"))), encoding="utf-8")

    results = execute_benchmark_job_plan_json(path, dry_run=True)

    assert results[0].skipped is True


def test_execute_benchmark_job_plan_validates_command_records():
    with pytest.raises(ValueError, match="commands array"):
        execute_benchmark_job_plan({})

    with pytest.raises(ValueError, match="argv"):
        execute_benchmark_job_plan({"commands": [{"name": "bad", "argv": []}]})


def test_driver_path_normalizes_dbfs_uri_for_databricks_driver():
    assert _driver_path("dbfs:/benchmarks/v1-plan.json").as_posix() == "/dbfs/benchmarks/v1-plan.json"


def test_runtime_argv_resolves_plain_python_to_current_interpreter():
    assert _runtime_argv(("python", "-m", "document_kv_cache.dataset_prep")) == (
        sys.executable,
        "-m",
        "document_kv_cache.dataset_prep",
    )
    assert _runtime_argv(("/custom/python", "-m", "document_kv_cache.dataset_prep")) == (
        "/custom/python",
        "-m",
        "document_kv_cache.dataset_prep",
    )


def test_main_writes_result_json_for_dry_run(tmp_path):
    plan_path = tmp_path / "plan.json"
    result_path = tmp_path / "result.json"
    plan_path.write_text(json.dumps(plan_for((sys.executable, "-c", "pass"))), encoding="utf-8")

    exit_code = main(["--plan-json", str(plan_path), "--dry-run", "--result-json", str(result_path)])

    assert exit_code == 0
    record = json.loads(result_path.read_text(encoding="utf-8"))
    assert record["commands"][0]["skipped"] is True
    assert record["plan_source"]["record_type"] == BENCHMARK_PLAN_SOURCE_RECORD_TYPE
    assert record["plan_source"]["path"] == str(plan_path)


def test_main_writes_result_json_for_failed_command(tmp_path):
    plan_path = tmp_path / "plan.json"
    result_path = tmp_path / "result.json"
    plan_path.write_text(json.dumps(plan_for((sys.executable, "-c", "raise SystemExit(4)"))), encoding="utf-8")

    exit_code = main(["--plan-json", str(plan_path), "--result-json", str(result_path)])

    assert exit_code == 2
    record = json.loads(result_path.read_text(encoding="utf-8"))
    assert record["ok"] is False
    assert record["commands"][0]["returncode"] == 4
    assert record["plan_source"]["path"] == str(plan_path)


def test_main_creates_parent_directory_for_failed_command_result_json(tmp_path):
    plan_path = tmp_path / "plan.json"
    result_path = tmp_path / "nested" / "results" / "failed.json"
    plan_path.write_text(json.dumps(plan_for((sys.executable, "-c", "raise SystemExit(4)"))), encoding="utf-8")

    exit_code = main(["--plan-json", str(plan_path), "--result-json", str(result_path)])

    assert exit_code == 2
    assert json.loads(result_path.read_text(encoding="utf-8"))["commands"][0]["returncode"] == 4


def test_public_plan_executor_main_respects_document_namespace_monkeypatch(monkeypatch, capsys, tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan_for((sys.executable, "-c", "pass"))), encoding="utf-8")
    original_legacy_execute = legacy_plan_executor.execute_benchmark_job_plan_json

    def fake_execute_json(path, *, dry_run=False, cwd=None):
        assert str(path) == str(plan_path)
        return ("public-hook-result",)

    def fake_record(results):
        assert results == ("public-hook-result",)
        return {"ok": True, "source": "public-hook"}

    def fake_plan_source_payload_to_record(path, driver_path, payload):
        assert str(path) == str(plan_path)
        assert str(driver_path) == str(plan_path)
        assert payload == plan_path.read_bytes()
        return {"source": "public-plan-source-hook"}

    monkeypatch.setattr(public_plan_executor, "execute_benchmark_job_plan_json", fake_execute_json)
    monkeypatch.setattr(public_plan_executor, "benchmark_command_results_to_record", fake_record)
    monkeypatch.setattr(
        public_plan_executor,
        "benchmark_plan_source_payload_to_record",
        fake_plan_source_payload_to_record,
    )

    exit_code = public_plan_executor.main(["--plan-json", str(plan_path)])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "source": "public-hook",
        "plan_source": {"source": "public-plan-source-hook"},
    }
    assert legacy_plan_executor.execute_benchmark_job_plan_json is original_legacy_execute


def test_legacy_plan_executor_main_respects_legacy_namespace_monkeypatch(monkeypatch, capsys, tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan_for((sys.executable, "-c", "pass"))), encoding="utf-8")
    original_public_execute = public_plan_executor.execute_benchmark_job_plan_json

    def fake_execute_json(path, *, dry_run=False, cwd=None):
        assert str(path) == str(plan_path)
        return ("legacy-hook-result",)

    def fake_record(results):
        assert results == ("legacy-hook-result",)
        return {"ok": True, "source": "legacy-hook"}

    def fake_plan_source_payload_to_record(path, driver_path, payload):
        assert str(path) == str(plan_path)
        assert str(driver_path) == str(plan_path)
        assert payload == plan_path.read_bytes()
        return {"source": "legacy-plan-source-hook"}

    monkeypatch.setattr(legacy_plan_executor, "execute_benchmark_job_plan_json", fake_execute_json)
    monkeypatch.setattr(legacy_plan_executor, "benchmark_command_results_to_record", fake_record)
    monkeypatch.setattr(
        legacy_plan_executor,
        "benchmark_plan_source_payload_to_record",
        fake_plan_source_payload_to_record,
    )

    exit_code = legacy_plan_executor.main(["--plan-json", str(plan_path)])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "source": "legacy-hook",
        "plan_source": {"source": "legacy-plan-source-hook"},
    }
    assert public_plan_executor.execute_benchmark_job_plan_json is original_public_execute


def test_legacy_plan_executor_result_json_respects_legacy_record_hook(monkeypatch, tmp_path):
    plan_path = tmp_path / "plan.json"
    result_path = tmp_path / "result.json"
    plan_path.write_text(json.dumps(plan_for((sys.executable, "-c", "pass"))), encoding="utf-8")

    def fake_execute_json(path, *, dry_run=False, cwd=None):
        assert str(path) == str(plan_path)
        return ("legacy-hook-result",)

    def fake_record(results):
        assert results == ("legacy-hook-result",)
        return {"ok": True, "source": "legacy-result-json-hook"}

    def fake_plan_source_payload_to_record(path, driver_path, payload):
        return {"source": "legacy-plan-source-hook"}

    monkeypatch.setattr(legacy_plan_executor, "execute_benchmark_job_plan_json", fake_execute_json)
    monkeypatch.setattr(legacy_plan_executor, "benchmark_command_results_to_record", fake_record)
    monkeypatch.setattr(
        legacy_plan_executor,
        "benchmark_plan_source_payload_to_record",
        fake_plan_source_payload_to_record,
    )

    exit_code = legacy_plan_executor.main(["--plan-json", str(plan_path), "--result-json", str(result_path)])

    assert exit_code == 0
    assert json.loads(result_path.read_text(encoding="utf-8")) == {
        "ok": True,
        "source": "legacy-result-json-hook",
        "plan_source": {"source": "legacy-plan-source-hook"},
    }


def test_legacy_plan_executor_import_order_does_not_capture_public_monkeypatch():
    script = dedent(
        """
        import sys
        import document_kv_cache.benchmark_plan_executor as public_plan_executor

        def public_execute_should_not_run(*args, **kwargs):
            raise AssertionError("legacy imported a public monkeypatch as its default")

        public_plan_executor.execute_benchmark_job_plan = public_execute_should_not_run

        import restaurant_kv_serving.benchmark_plan_executor as legacy_plan_executor

        assert legacy_plan_executor.BenchmarkCommandResult is public_plan_executor.BenchmarkCommandResult
        results = legacy_plan_executor.execute_benchmark_job_plan(
            {"commands": [{"name": "command-1", "argv": [sys.executable, "-c", "pass"]}]},
            dry_run=True,
        )
        assert results[0].skipped is True
        try:
            public_plan_executor.execute_benchmark_job_plan({"commands": []})
        except AssertionError:
            pass
        else:
            raise AssertionError("public monkeypatch was not installed")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_benchmark_plan_executor_document_module_owns_public_api():
    assert public_plan_executor.__all__ == [
        "BENCHMARK_PLAN_EXECUTION_RECORD_TYPE",
        "BENCHMARK_PLAN_SOURCE_RECORD_TYPE",
        "BenchmarkCommandResult",
        "execute_benchmark_job_plan",
        "execute_benchmark_job_plan_json",
        "benchmark_command_results_to_record",
        "benchmark_plan_source_to_record",
        "benchmark_plan_source_payload_to_record",
        "write_benchmark_command_results_json",
        "main",
    ]
    assert public_plan_executor.BenchmarkCommandResult.__module__ == "document_kv_cache.benchmark_plan_executor"
    assert public_plan_executor.execute_benchmark_job_plan.__module__ == "document_kv_cache.benchmark_plan_executor"
    assert public_plan_executor.main.__module__ == "document_kv_cache.benchmark_plan_executor"
    assert not hasattr(legacy_plan_executor, "__all__")
    assert legacy_plan_executor.BenchmarkCommandResult is public_plan_executor.BenchmarkCommandResult
    assert legacy_plan_executor.execute_benchmark_job_plan.__module__ == "restaurant_kv_serving.benchmark_plan_executor"
    assert (
        legacy_plan_executor.benchmark_command_results_to_record.__module__
        == "restaurant_kv_serving.benchmark_plan_executor"
    )
    assert (
        legacy_plan_executor.benchmark_plan_source_payload_to_record.__module__
        == "restaurant_kv_serving.benchmark_plan_executor"
    )
    assert legacy_plan_executor.main.__module__ == "restaurant_kv_serving.benchmark_plan_executor"
