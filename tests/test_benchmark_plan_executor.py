import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
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
    write_benchmark_command_results_json,
)
from document_kv_cache.benchmark_plan import (
    BenchmarkDatasetPath,
    BenchmarkPlanConfig,
    benchmark_job_plan_to_record,
    build_v1_benchmark_plan,
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


def test_write_benchmark_command_results_json_can_include_plan_source(tmp_path):
    output_path = tmp_path / "nested" / "result.json"
    plan_source = {"path": "dbfs:/benchmarks/plan.json", "sha256": "b" * 64}
    results = execute_benchmark_job_plan(plan_for((sys.executable, "-c", "pass")), dry_run=True)

    write_benchmark_command_results_json(results, output_path, plan_source=plan_source)

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["record_type"] == BENCHMARK_PLAN_EXECUTION_RECORD_TYPE
    assert record["plan_source"] == plan_source


def test_write_benchmark_command_results_json_preserves_old_serializer_hooks(monkeypatch, tmp_path):
    output_path = tmp_path / "result.json"
    plan_source = {"path": "dbfs:/benchmarks/plan.json", "sha256": "c" * 64}
    results = execute_benchmark_job_plan(plan_for((sys.executable, "-c", "pass")), dry_run=True)

    def fake_record(observed_results):
        assert observed_results == results
        return {"ok": True, "source": "old-serializer-hook"}

    monkeypatch.setattr(public_plan_executor, "benchmark_command_results_to_record", fake_record)

    public_plan_executor.write_benchmark_command_results_json(results, output_path, plan_source=plan_source)

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "ok": True,
        "source": "old-serializer-hook",
        "plan_source": plan_source,
    }


def test_benchmark_plan_source_record_hashes_driver_visible_plan_json(tmp_path):
    path = tmp_path / "plan.json"
    plan = {
        "plan_version": "v1",
        "suite_id": "suite-1",
        "model_id": "qwen3:4b-instruct",
        "hardware_target": "aws-g6-l4",
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
        "hardware_target": "aws-g6-l4",
        "command_count": 1,
    }


def test_benchmark_plan_source_record_summarizes_handoff_generation_commands(tmp_path):
    path = tmp_path / "plan.json"
    plan = {
        "plan_version": "v1",
        "suite_id": "suite-1",
        "model_id": "qwen3:4b-instruct",
        "hardware_target": "aws-g6-l4",
        "commands": [
            {"name": "prepare-biography", "argv": [sys.executable, "-c", "pass"]},
            {"name": "generate-biography-handoff-bundles", "argv": [sys.executable, "-c", "pass"]},
            {"name": "enrich-biography-handoffs", "argv": [sys.executable, "-c", "pass"]},
            {"name": "generate-hotpotqa-handoff-bundles", "argv": [sys.executable, "-c", "pass"]},
            {"name": "enrich-hotpotqa-handoffs", "argv": [sys.executable, "-c", "pass"]},
            {"name": "generate-unknown-handoff-bundles", "argv": [sys.executable, "-c", "pass"]},
            {"name": "generate-biography-handoff-bundles", "argv": [sys.executable, "-c", "pass"]},
            {"name": "run-benchmark", "argv": [sys.executable, "-c", "pass"]},
        ],
    }
    payload = json.dumps(plan, sort_keys=True).encode("utf-8")

    record = benchmark_plan_source_payload_to_record(str(path), path, payload)

    assert record["command_count"] == 8
    assert record["benchmark_handoff_generation_datasets"] == ["biography", "hotpotqa"]
    assert record["benchmark_handoff_enrichment_datasets"] == ["biography", "hotpotqa"]


def test_main_hashes_the_plan_payload_that_was_executed(tmp_path):
    plan_path = tmp_path / "plan.json"
    result_path = tmp_path / "result.json"
    original_plan = {
        "plan_version": "v1",
        "suite_id": "original-suite",
        "model_id": "qwen3:4b-instruct",
        "hardware_target": "aws-g6-l4",
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


def test_execute_benchmark_job_plan_accepts_generated_benchmark_plan_record(tmp_path):
    plan = build_v1_benchmark_plan(
        BenchmarkPlanConfig(
            suite_id="v1-g6-l4",
            dataset_paths=tuple(
                BenchmarkDatasetPath(
                    dataset=dataset,
                    raw_jsonl=str(tmp_path / "raw" / f"{dataset}.jsonl"),
                    prepared_jsonl=str(tmp_path / "prepared" / f"{dataset}.jsonl"),
                )
                for dataset in ("biography", "hotpotqa", "musique", "niah")
            ),
            base_url="http://localhost:8000",
            benchmark_output_json=str(tmp_path / "results.json"),
        )
    )
    record = benchmark_job_plan_to_record(plan)

    results = execute_benchmark_job_plan(record, dry_run=True)

    assert len(results) == len(record["commands"])
    assert all(result.skipped for result in results)


def test_execute_benchmark_job_plan_validates_command_records():
    with pytest.raises(ValueError, match="commands array"):
        execute_benchmark_job_plan({})

    with pytest.raises(ValueError, match="argv"):
        execute_benchmark_job_plan({"commands": [{"name": "bad", "argv": []}]})


def test_execute_benchmark_job_plan_rejects_unsupported_plan_keys():
    plan = {**plan_for((sys.executable, "-c", "pass")), "debug": True}

    with pytest.raises(ValueError, match=r"Benchmark plan JSON has unsupported keys: \['debug'\]"):
        execute_benchmark_job_plan(plan, dry_run=True)


def test_execute_benchmark_job_plan_rejects_unsupported_command_keys():
    plan = {"commands": [{"name": "command-1", "argv": [sys.executable, "-c", "pass"], "debug": True}]}

    with pytest.raises(ValueError, match=r"commands\[0\] has unsupported keys: \['debug'\]"):
        execute_benchmark_job_plan(plan, dry_run=True)


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


def test_public_plan_executor_result_json_preserves_old_writer_hook(monkeypatch, tmp_path):
    plan_path = tmp_path / "plan.json"
    result_path = tmp_path / "result.json"
    plan_path.write_text(json.dumps(plan_for((sys.executable, "-c", "pass"))), encoding="utf-8")

    def fake_writer(results, path):
        assert results[0].ok is True
        Path(path).write_text(json.dumps({"ok": True, "source": "public-writer-hook"}), encoding="utf-8")

    monkeypatch.setattr(public_plan_executor, "write_benchmark_command_results_json", fake_writer)

    exit_code = public_plan_executor.main(
        ["--plan-json", str(plan_path), "--dry-run", "--result-json", str(result_path)]
    )

    assert exit_code == 0
    record = json.loads(result_path.read_text(encoding="utf-8"))
    assert record["source"] == "public-writer-hook"
    assert record["plan_source"]["path"] == str(plan_path)


def test_legacy_plan_executor_result_json_preserves_old_writer_hook(monkeypatch, tmp_path):
    plan_path = tmp_path / "plan.json"
    result_path = tmp_path / "result.json"
    plan_path.write_text(json.dumps(plan_for((sys.executable, "-c", "pass"))), encoding="utf-8")

    def fake_plan_source_payload_to_record(path, driver_path, payload):
        return {"source": "legacy-plan-source-hook"}

    def fake_writer(results, path):
        assert results[0].ok is True
        Path(path).write_text(json.dumps({"ok": True, "source": "legacy-writer-hook"}), encoding="utf-8")

    monkeypatch.setattr(
        legacy_plan_executor,
        "benchmark_plan_source_payload_to_record",
        fake_plan_source_payload_to_record,
    )
    monkeypatch.setattr(legacy_plan_executor, "write_benchmark_command_results_json", fake_writer)

    exit_code = legacy_plan_executor.main(
        ["--plan-json", str(plan_path), "--dry-run", "--result-json", str(result_path)]
    )

    assert exit_code == 0
    assert json.loads(result_path.read_text(encoding="utf-8")) == {
        "ok": True,
        "source": "legacy-writer-hook",
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


def test_legacy_plan_executor_uses_source_result_class_when_public_class_is_replaced_before_import():
    script = dedent(
        """
        import sys
        import document_kv_cache.benchmark_plan_executor as public_plan_executor

        class FakeBenchmarkCommandResult:
            def __init__(*args, **kwargs):
                raise AssertionError("legacy imported patched public result class")

        public_plan_executor.BenchmarkCommandResult = FakeBenchmarkCommandResult

        import restaurant_kv_serving.benchmark_plan_executor as legacy_plan_executor

        assert legacy_plan_executor.BenchmarkCommandResult is not FakeBenchmarkCommandResult
        result = legacy_plan_executor.BenchmarkCommandResult(
            name="command-1",
            argv=(sys.executable, "-c", "pass"),
            returncode=0,
        )
        assert result.ok is True
        assert type(result) is legacy_plan_executor.BenchmarkCommandResult
        results = legacy_plan_executor.execute_benchmark_job_plan(
            {"commands": [{"name": "command-1", "argv": [sys.executable, "-c", "pass"]}]},
            dry_run=True,
        )
        assert type(results[0]) is legacy_plan_executor.BenchmarkCommandResult
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


def test_legacy_plan_executor_uses_source_result_class_when_public_class_is_mutated_before_import():
    script = dedent(
        """
        import sys
        import document_kv_cache.benchmark_plan_executor as public_plan_executor

        public_plan_executor.BenchmarkCommandResult.ok = property(lambda self: False)

        import restaurant_kv_serving.benchmark_plan_executor as legacy_plan_executor

        result = legacy_plan_executor.BenchmarkCommandResult(
            name="command-1",
            argv=(sys.executable, "-c", "pass"),
            returncode=0,
        )
        assert result.ok is True
        results = legacy_plan_executor.execute_benchmark_job_plan(
            {"commands": [{"name": "command-1", "argv": [sys.executable, "-c", "pass"]}]},
            dry_run=True,
        )
        assert results[0].ok is True
        assert public_plan_executor.BenchmarkCommandResult(
            name="command-1",
            argv=(sys.executable, "-c", "pass"),
            returncode=0,
        ).ok is False
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


def test_legacy_plan_executor_uses_source_result_class_when_public_slot_descriptor_is_mutated_before_import():
    script = dedent(
        """
        import sys
        import document_kv_cache.benchmark_plan_executor as public_plan_executor

        class ForeignResult:
            __slots__ = ("returncode",)

        public_plan_executor.BenchmarkCommandResult.returncode = ForeignResult.returncode

        import restaurant_kv_serving.benchmark_plan_executor as legacy_plan_executor

        result = legacy_plan_executor.BenchmarkCommandResult(
            name="command-1",
            argv=(sys.executable, "-c", "pass"),
            returncode=0,
        )
        assert result.ok is True
        results = legacy_plan_executor.execute_benchmark_job_plan(
            {"commands": [{"name": "command-1", "argv": [sys.executable, "-c", "pass"]}]},
            dry_run=True,
        )
        assert type(results[0]) is legacy_plan_executor.BenchmarkCommandResult
        assert results[0].ok is True
        try:
            public_plan_executor.BenchmarkCommandResult(
                name="command-1",
                argv=(sys.executable, "-c", "pass"),
                returncode=0,
            )
        except TypeError:
            pass
        else:
            raise AssertionError("public slot descriptor mutation did not affect construction")
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
