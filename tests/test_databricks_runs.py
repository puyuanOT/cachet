import json
import os
import pickle
import subprocess
import sys
import urllib.error

import pytest

import document_kv_cache.databricks_runs as public_databricks_runs
import restaurant_kv_serving.databricks_runs as legacy_databricks_runs
from document_kv_cache.databricks_runs import (
    DEFAULT_DATABRICKS_HOST_ENV,
    DEFAULT_DATABRICKS_TOKEN_ENV,
    DATABRICKS_RUN_STATUS_RECORD_TYPE,
    DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE,
    DatabricksWorkspaceConfig,
    databricks_run_status_record,
    databricks_run_status_sidecar_issues,
    databricks_workspace_config_from_env,
    get_databricks_run,
    main,
    read_databricks_run_submit_payload,
    submit_databricks_run,
    summarize_databricks_run,
    summarize_databricks_run_submit_payload,
    validate_databricks_run_status_sidecar,
    write_databricks_run_response_json,
)


def test_workspace_config_from_env_normalizes_host_and_hides_token_in_repr():
    config = databricks_workspace_config_from_env(
        environ={
            DEFAULT_DATABRICKS_HOST_ENV: "https://dbc.example.cloud.databricks.com/",
            DEFAULT_DATABRICKS_TOKEN_ENV: "secret-token",
        },
        timeout_seconds=12,
    )

    assert config.normalized_host == "https://dbc.example.cloud.databricks.com"
    assert config.timeout_seconds == 12
    assert "secret-token" not in repr(config)


def test_workspace_config_from_env_requires_host_and_token():
    with pytest.raises(ValueError, match=DEFAULT_DATABRICKS_HOST_ENV):
        databricks_workspace_config_from_env(environ={DEFAULT_DATABRICKS_TOKEN_ENV: "token"})

    with pytest.raises(ValueError, match=DEFAULT_DATABRICKS_TOKEN_ENV):
        databricks_workspace_config_from_env(environ={DEFAULT_DATABRICKS_HOST_ENV: "https://dbc.example"})


def test_submit_databricks_run_posts_payload_with_bearer_token():
    opener = _FakeOpener({"run_id": 123, "number_in_job": 1})
    config = DatabricksWorkspaceConfig("https://dbc.example/", "secret-token", timeout_seconds=9)

    response = submit_databricks_run(config, {"run_name": "document-kv-vllm-smoke"}, opener=opener)

    assert response == {"run_id": 123, "number_in_job": 1}
    request = opener.requests[0]
    assert request.full_url == "https://dbc.example/api/2.1/jobs/runs/submit"
    assert request.get_method() == "POST"
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert json.loads(request.data.decode("utf-8")) == {"run_name": "document-kv-vllm-smoke"}
    assert opener.timeouts == [9]


def test_get_databricks_run_fetches_run_by_id():
    opener = _FakeOpener({"run_id": 123, "state": {"life_cycle_state": "TERMINATED"}})
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")

    response = get_databricks_run(config, 123, opener=opener)

    assert response["state"]["life_cycle_state"] == "TERMINATED"
    request = opener.requests[0]
    assert request.full_url == "https://dbc.example/api/2.1/jobs/runs/get?run_id=123"
    assert request.get_method() == "GET"
    assert request.data is None


def test_summarize_databricks_run_extracts_run_and_task_state():
    summary = summarize_databricks_run(
        {
            "run_id": 123,
            "run_name": "document-kv-v1-benchmark",
            "run_page_url": "https://dbc.example/#job/123",
            "state": {
                "life_cycle_state": "RUNNING",
                "state_message": "task is running",
            },
            "start_time": 1000,
            "cluster_instance": {"cluster_id": "cluster-main", "spark_context_id": "ignored"},
            "tasks": [
                {
                    "task_key": "prepare",
                    "run_id": 124,
                    "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
                    "cluster_instance": {"cluster_id": "cluster-prepare"},
                    "start_time": 1001,
                    "end_time": 1002,
                },
                {
                    "task_key": "benchmark",
                    "run_id": 125,
                    "state": {"life_cycle_state": "RUNNING", "state_message": "working"},
                    "cluster_instance": {"cluster_id": "cluster-benchmark"},
                    "start_time": 1003,
                },
            ],
        }
    )

    assert summary == {
        "record_type": DATABRICKS_RUN_STATUS_RECORD_TYPE,
        "run_id": 123,
        "run_name": "document-kv-v1-benchmark",
        "run_page_url": "https://dbc.example/#job/123",
        "life_cycle_state": "RUNNING",
        "result_state": None,
        "state_message": "task is running",
        "start_time": 1000,
        "end_time": None,
        "terminal": False,
        "succeeded": False,
        "active_task_key": "benchmark",
        "task_count": 2,
        "tasks": [
            {
                "task_key": "prepare",
                "run_id": 124,
                "life_cycle_state": "TERMINATED",
                "result_state": "SUCCESS",
                "state_message": None,
                "cluster_id": "cluster-prepare",
                "start_time": 1001,
                "end_time": 1002,
            },
            {
                "task_key": "benchmark",
                "run_id": 125,
                "life_cycle_state": "RUNNING",
                "result_state": None,
                "state_message": "working",
                "cluster_id": "cluster-benchmark",
                "start_time": 1003,
                "end_time": None,
            },
        ],
        "cluster_id": "cluster-main",
    }


def test_summarize_databricks_run_marks_successful_terminal_run():
    summary = summarize_databricks_run(
        {
            "run_id": 123,
            "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
        }
    )

    assert summary["terminal"] is True
    assert summary["succeeded"] is True
    assert summary["task_count"] == 0


def test_summarize_databricks_run_can_attach_submit_payload_provenance():
    summary = summarize_databricks_run(
        {
            "run_id": 123,
            "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
            "tasks": [
                {
                    "task_key": "run-benchmark",
                    "run_id": 124,
                    "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
                }
            ],
        },
        submit_payload=_single_node_g5_submit_payload(),
        submit_payload_path="/Volumes/catalog/schema/volume/payload.json",
    )

    submit_payload = summary["submit_payload"]
    assert submit_payload["record_type"] == DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE
    assert submit_payload["source_path"] == "/Volumes/catalog/schema/volume/payload.json"
    assert len(submit_payload["sha256"]) == 64
    assert submit_payload["single_node"] is True
    assert submit_payload["aws_single_node_gpu_type"] is True
    assert submit_payload["aws_g5_node_type"] is True
    assert submit_payload["node_type_ids"] == ["g5.4xlarge"]
    assert submit_payload["data_security_modes"] == ["SINGLE_USER"]
    assert submit_payload["task_keys"] == ["run-benchmark"]


def test_summarize_databricks_run_accepts_g6_l4_submit_payload_provenance():
    payload = _single_node_g5_submit_payload()
    cluster = payload["tasks"][0]["new_cluster"]
    cluster["node_type_id"] = "g6.8xlarge"
    cluster["driver_node_type_id"] = "g6.8xlarge"

    summary = summarize_databricks_run(
        {
            "run_id": 123,
            "run_name": "document-kv-v1",
            "run_page_url": "https://dbc.example/#job/123/run/123",
            "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
            "tasks": [
                {
                    "task_key": "run-benchmark",
                    "run_id": 124,
                    "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
                }
            ],
        },
        submit_payload=payload,
        submit_payload_path="/Volumes/catalog/schema/volume/payload.json",
    )

    assert summary["submit_payload"]["aws_single_node_gpu_type"] is True
    assert summary["submit_payload"]["aws_g5_node_type"] is True
    assert summary["submit_payload"]["node_type_ids"] == ["g6.8xlarge"]
    assert databricks_run_status_sidecar_issues(summary) == ()


def test_databricks_run_status_sidecar_validation_accepts_direct_and_wrapped_records():
    status_record = _valid_databricks_run_status_record()
    wrapped_record = {"ok": True, "action": "get", "summary": status_record}

    assert databricks_run_status_record(status_record) is status_record
    assert databricks_run_status_record(wrapped_record) is status_record
    assert databricks_run_status_sidecar_issues(status_record) == ()
    assert databricks_run_status_sidecar_issues(wrapped_record) == ()
    validate_databricks_run_status_sidecar(status_record)
    validate_databricks_run_status_sidecar(wrapped_record)


def test_databricks_run_status_sidecar_validation_reports_release_readiness_issues():
    status_record = _valid_databricks_run_status_record()
    bad_record = {
        **status_record,
        "response": {"raw": True},
        "succeeded": False,
        "result_state": "FAILED",
    }

    issues = databricks_run_status_sidecar_issues(bad_record)

    assert "Databricks run status sidecar must not include the raw Jobs API response" in issues
    assert "Databricks run status sidecar succeeded must be true" in issues
    assert "Databricks run status sidecar result_state must be 'SUCCESS'" in issues
    assert any("unsupported keys" in issue and "response" in issue for issue in issues)
    with pytest.raises(ValueError, match="succeeded must be true"):
        validate_databricks_run_status_sidecar(bad_record)


def test_databricks_run_status_sidecar_validation_requires_submit_payload():
    status_record = _valid_databricks_run_status_record()
    bad_record = dict(status_record)
    del bad_record["submit_payload"]

    assert databricks_run_status_sidecar_issues(bad_record) == (
        "Databricks run status sidecar submit_payload must be an object",
    )


def test_databricks_run_status_sidecar_validation_requires_null_active_task_key_for_success():
    status_record = _valid_databricks_run_status_record()
    bad_record = {**status_record, "active_task_key": "run-benchmark"}

    issues = databricks_run_status_sidecar_issues(bad_record)

    assert (
        "Databricks run status sidecar active_task_key must be null for successful terminal runs"
        in issues
    )


def test_databricks_run_status_sidecar_validation_rejects_unsupported_gpu_or_mismatched_payload():
    status_record = _valid_databricks_run_status_record()
    submit_payload = json.loads(json.dumps(status_record["submit_payload"]))
    submit_payload["aws_single_node_gpu_type"] = False
    submit_payload["aws_g5_node_type"] = False
    submit_payload["tasks"][0]["task_key"] = "different-task"
    submit_payload["tasks"][0]["node_type_id"] = "g6e.8xlarge"
    submit_payload["task_keys"] = ["different-task"]
    bad_record = {**status_record, "submit_payload": submit_payload}

    issues = databricks_run_status_sidecar_issues(bad_record)

    assert "Databricks run status sidecar submit_payload.aws_single_node_gpu_type must be true" in issues
    assert (
        "Databricks run status sidecar submit_payload.tasks[0].node_type_id must be an AWS g5/g6 node type"
        in issues
    )
    assert "Databricks run status sidecar submit_payload.task_keys must match status task keys" in issues


def test_databricks_run_status_sidecar_validation_matches_submit_payload_run_name():
    status_record = _valid_databricks_run_status_record()
    submit_payload = json.loads(json.dumps(status_record["submit_payload"]))
    submit_payload["run_name"] = "document-kv-stale-run"
    bad_record = {**status_record, "submit_payload": submit_payload}

    issues = databricks_run_status_sidecar_issues(bad_record)

    assert "Databricks run status sidecar submit_payload.run_name must match run_name" in issues


@pytest.mark.parametrize("purpose", [None, ""])
def test_databricks_run_status_sidecar_validation_requires_submit_payload_task_purpose(purpose):
    status_record = _valid_databricks_run_status_record()
    submit_payload = json.loads(json.dumps(status_record["submit_payload"]))
    submit_payload["tasks"][0]["purpose"] = purpose
    bad_record = {**status_record, "submit_payload": submit_payload}

    issues = databricks_run_status_sidecar_issues(bad_record)

    assert (
        "Databricks run status sidecar submit_payload.tasks[0].purpose must be a non-empty string"
        in issues
    )


@pytest.mark.parametrize(
    ("summary_field", "bad_values"),
    [
        ("node_type_ids", ["g5.12xlarge"]),
        ("driver_node_type_ids", ["g5.12xlarge"]),
        ("spark_versions", ["15.3.x-gpu-ml-scala2.12"]),
        ("data_security_modes", ["SINGLE_USER", "USER_ISOLATION"]),
    ],
)
def test_databricks_run_status_sidecar_validation_matches_submit_payload_summary_arrays(
    summary_field,
    bad_values,
):
    status_record = _valid_databricks_run_status_record()
    submit_payload = json.loads(json.dumps(status_record["submit_payload"]))
    submit_payload[summary_field] = bad_values
    bad_record = {**status_record, "submit_payload": submit_payload}

    issues = databricks_run_status_sidecar_issues(bad_record)

    assert (
        f"Databricks run status sidecar submit_payload.{summary_field} must match submit_payload.tasks"
        in issues
    )


def test_summarize_databricks_run_submit_payload_reports_unsupported_gpu_multi_node_payload():
    payload = _single_node_g5_submit_payload()
    payload["tasks"][0]["new_cluster"]["node_type_id"] = "g6e.4xlarge"
    payload["tasks"][0]["new_cluster"]["num_workers"] = 1

    summary = summarize_databricks_run_submit_payload(payload)

    assert summary["record_type"] == DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE
    assert summary["single_node"] is False
    assert summary["aws_single_node_gpu_type"] is False
    assert summary["aws_g5_node_type"] is False


def test_databricks_http_errors_are_sanitized():
    opener = _HTTPErrorOpener(
        urllib.error.HTTPError(
            "https://dbc.example/api/2.1/jobs/runs/get?run_id=123",
            403,
            "Forbidden",
            {},
            _BytesFile(b'{"error_code":"PERMISSION_DENIED","message":"not allowed"}'),
        )
    )
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")

    with pytest.raises(RuntimeError, match="HTTP 403: not allowed") as excinfo:
        get_databricks_run(config, 123, opener=opener)

    assert "secret-token" not in str(excinfo.value)


def test_databricks_http_error_body_echoed_credentials_are_redacted():
    opener = _HTTPErrorOpener(
        urllib.error.HTTPError(
            "https://dbc.example/api/2.1/jobs/runs/get?run_id=123",
            403,
            "Forbidden",
            {},
            _BytesFile(b'{"message":"Authorization: Bearer secret-token; token=secret-token"}'),
        )
    )
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")

    with pytest.raises(RuntimeError) as excinfo:
        get_databricks_run(config, 123, opener=opener)

    error = str(excinfo.value)
    assert "HTTP 403" in error
    assert "secret-token" not in error
    assert "Bearer [REDACTED]" in error
    assert "token=[REDACTED]" in error


def test_read_and_write_databricks_run_json_helpers(tmp_path):
    payload_path = tmp_path / "payload.json"
    response_path = tmp_path / "response.json"
    payload_path.write_text('{"run_name":"document-kv-vllm-smoke"}', encoding="utf-8")

    assert read_databricks_run_submit_payload(payload_path) == {"run_name": "document-kv-vllm-smoke"}

    write_databricks_run_response_json({"ok": True, "response": {"run_id": 123}}, response_path)

    assert json.loads(response_path.read_text(encoding="utf-8")) == {"ok": True, "response": {"run_id": 123}}


def test_read_databricks_run_submit_payload_rejects_non_object(tmp_path):
    payload_path = tmp_path / "payload.json"
    payload_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        read_databricks_run_submit_payload(payload_path)


def test_main_submit_writes_response_json(monkeypatch, tmp_path):
    payload_path = tmp_path / "payload.json"
    output_path = tmp_path / "response.json"
    payload_path.write_text('{"run_name":"document-kv-vllm-smoke"}', encoding="utf-8")

    monkeypatch.setenv(DEFAULT_DATABRICKS_HOST_ENV, "https://dbc.example")
    monkeypatch.setenv(DEFAULT_DATABRICKS_TOKEN_ENV, "secret-token")
    monkeypatch.setattr(
        legacy_databricks_runs,
        "submit_databricks_run",
        lambda config, payload: {"run_id": 123, "payload": payload, "host": config.normalized_host},
    )

    exit_code = legacy_databricks_runs.main(
        [
            "--output-json",
            str(output_path),
            "submit",
            "--payload-json",
            str(payload_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "ok": True,
        "action": "submit",
        "response": {
            "run_id": 123,
            "payload": {"run_name": "document-kv-vllm-smoke"},
            "host": "https://dbc.example",
        },
    }


def test_main_get_can_write_summary(monkeypatch, tmp_path):
    output_path = tmp_path / "response.json"
    raw_secret = "do-not-write-me"
    monkeypatch.setenv(DEFAULT_DATABRICKS_HOST_ENV, "https://dbc.example")
    monkeypatch.setenv(DEFAULT_DATABRICKS_TOKEN_ENV, "secret-token")
    monkeypatch.setattr(
        legacy_databricks_runs,
        "get_databricks_run",
        lambda config, run_id: {
            "run_id": int(run_id),
            "tasks": [{"notebook_task": {"base_parameters": {"token": raw_secret}}}],
            "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
        },
    )

    exit_code = legacy_databricks_runs.main(
        [
            "--output-json",
            str(output_path),
            "get",
            "--run-id",
            "123",
            "--summary",
        ]
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "response" not in record
    assert raw_secret not in output_path.read_text(encoding="utf-8")
    assert record["summary"]["record_type"] == DATABRICKS_RUN_STATUS_RECORD_TYPE
    assert record["summary"]["succeeded"] is True


def test_main_get_summary_can_include_submit_payload_provenance(monkeypatch, tmp_path):
    output_path = tmp_path / "response.json"
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(_single_node_g5_submit_payload()), encoding="utf-8")
    monkeypatch.setenv(DEFAULT_DATABRICKS_HOST_ENV, "https://dbc.example")
    monkeypatch.setenv(DEFAULT_DATABRICKS_TOKEN_ENV, "secret-token")
    monkeypatch.setattr(
        legacy_databricks_runs,
        "get_databricks_run",
        lambda config, run_id: {
            "run_id": int(run_id),
            "tasks": [
                {
                    "task_key": "run-benchmark",
                    "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
                }
            ],
            "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
        },
    )

    exit_code = legacy_databricks_runs.main(
        [
            "--output-json",
            str(output_path),
            "get",
            "--run-id",
            "123",
            "--summary",
            "--submit-payload-json",
            str(payload_path),
        ]
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "response" not in record
    assert record["summary"]["submit_payload"]["source_path"] == str(payload_path)
    assert record["summary"]["submit_payload"]["single_node"] is True
    assert record["summary"]["submit_payload"]["aws_g5_node_type"] is True
    assert record["summary"]["submit_payload"]["aws_single_node_gpu_type"] is True


def test_main_get_summary_can_include_raw_response_when_requested(monkeypatch, tmp_path):
    output_path = tmp_path / "response.json"
    monkeypatch.setenv(DEFAULT_DATABRICKS_HOST_ENV, "https://dbc.example")
    monkeypatch.setenv(DEFAULT_DATABRICKS_TOKEN_ENV, "secret-token")
    monkeypatch.setattr(
        legacy_databricks_runs,
        "get_databricks_run",
        lambda config, run_id: {
            "run_id": int(run_id),
            "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
        },
    )

    exit_code = legacy_databricks_runs.main(
        [
            "--output-json",
            str(output_path),
            "get",
            "--run-id",
            "123",
            "--summary",
            "--include-response",
        ]
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert record["response"]["run_id"] == 123
    assert record["summary"]["succeeded"] is True


def test_main_output_json_write_failure_falls_back_to_stdout(monkeypatch, tmp_path, capsys):
    payload_path = tmp_path / "payload.json"
    output_path = tmp_path / "missing-parent" / "response.json"
    payload_path.write_text('{"run_name":"document-kv-vllm-smoke"}', encoding="utf-8")

    monkeypatch.setenv(DEFAULT_DATABRICKS_HOST_ENV, "https://dbc.example")
    monkeypatch.setenv(DEFAULT_DATABRICKS_TOKEN_ENV, "secret-token")
    monkeypatch.setattr(legacy_databricks_runs, "submit_databricks_run", lambda config, payload: {"run_id": 123})

    exit_code = legacy_databricks_runs.main(
        [
            "--output-json",
            str(output_path),
            "submit",
            "--payload-json",
            str(payload_path),
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert result["ok"] is False
    assert result["error_type"] == "FileNotFoundError"
    assert result["output_json_error_type"] == "FileNotFoundError"


def test_main_output_json_redacts_databricks_http_error_body(monkeypatch, tmp_path):
    output_path = tmp_path / "response.json"
    opener = _HTTPErrorOpener(
        urllib.error.HTTPError(
            "https://dbc.example/api/2.1/jobs/runs/get?run_id=123",
            401,
            "Unauthorized",
            {},
            _BytesFile(b'{"message":"Authorization: Bearer secret-token; echoed secret-token"}'),
        )
    )

    monkeypatch.setenv(DEFAULT_DATABRICKS_HOST_ENV, "https://dbc.example")
    monkeypatch.setenv(DEFAULT_DATABRICKS_TOKEN_ENV, "secret-token")
    monkeypatch.setattr(
        legacy_databricks_runs,
        "get_databricks_run",
        lambda config, run_id: legacy_databricks_runs._databricks_api_json(
            config,
            "GET",
            f"/api/2.1/jobs/runs/get?run_id={run_id}",
            opener=opener,
        ),
    )

    exit_code = legacy_databricks_runs.main(
        [
            "--output-json",
            str(output_path),
            "get",
            "--run-id",
            "123",
        ]
    )

    output = output_path.read_text(encoding="utf-8")
    record = json.loads(output)
    assert exit_code == 1
    assert record["error_type"] == "RuntimeError"
    assert "HTTP 401" in record["error"]
    assert "secret-token" not in output
    assert "Bearer [REDACTED]" in record["error"]


def test_public_databricks_runs_main_respects_document_namespace_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "response.json"
    original_legacy_get = legacy_databricks_runs.get_databricks_run

    def fake_get(config, run_id):
        assert run_id == "123"
        return {"run_id": 123, "source": "public-hook"}

    def fake_summary(run, **kwargs):
        assert run == {"run_id": 123, "source": "public-hook"}
        assert kwargs == {"submit_payload": None, "submit_payload_path": None}
        return {"record_type": DATABRICKS_RUN_STATUS_RECORD_TYPE, "source": "summary-hook"}

    monkeypatch.setattr(public_databricks_runs, "get_databricks_run", fake_get)
    monkeypatch.setattr(public_databricks_runs, "summarize_databricks_run", fake_summary)
    monkeypatch.setattr(
        public_databricks_runs,
        "databricks_workspace_config_from_env",
        lambda **kwargs: DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
    )

    exit_code = public_databricks_runs.main(
        [
            "--output-json",
            str(output_path),
            "get",
            "--run-id",
            "123",
            "--summary",
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "ok": True,
        "action": "get",
        "summary": {"record_type": DATABRICKS_RUN_STATUS_RECORD_TYPE, "source": "summary-hook"},
    }
    assert legacy_databricks_runs.get_databricks_run is original_legacy_get


def test_legacy_databricks_runs_main_respects_legacy_namespace_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "response.json"
    original_public_get = public_databricks_runs.get_databricks_run

    def fake_get(config, run_id):
        assert run_id == "123"
        assert isinstance(config, legacy_databricks_runs.DatabricksWorkspaceConfig)
        return {"run_id": 123, "source": "legacy-hook"}

    def fake_summary(run, **kwargs):
        assert run == {"run_id": 123, "source": "legacy-hook"}
        assert kwargs == {"submit_payload": None, "submit_payload_path": None}
        return {"record_type": DATABRICKS_RUN_STATUS_RECORD_TYPE, "source": "legacy-summary-hook"}

    monkeypatch.setattr(legacy_databricks_runs, "get_databricks_run", fake_get)
    monkeypatch.setattr(legacy_databricks_runs, "summarize_databricks_run", fake_summary)
    monkeypatch.setattr(
        legacy_databricks_runs,
        "databricks_workspace_config_from_env",
        lambda **kwargs: legacy_databricks_runs.DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
    )

    exit_code = legacy_databricks_runs.main(
        [
            "--output-json",
            str(output_path),
            "get",
            "--run-id",
            "123",
            "--summary",
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "ok": True,
        "action": "get",
        "summary": {"record_type": DATABRICKS_RUN_STATUS_RECORD_TYPE, "source": "legacy-summary-hook"},
    }
    assert public_databricks_runs.get_databricks_run is original_public_get


def test_legacy_databricks_runs_ignores_document_namespace_monkeypatch(monkeypatch, tmp_path):
    output_path = tmp_path / "response.json"

    def public_get_should_not_run(config, run_id):  # pragma: no cover - defensive assertion
        raise AssertionError("legacy main should not use document namespace monkeypatches")

    monkeypatch.setattr(public_databricks_runs, "get_databricks_run", public_get_should_not_run)
    monkeypatch.setattr(legacy_databricks_runs, "get_databricks_run", lambda config, run_id: {"run_id": int(run_id)})
    monkeypatch.setattr(
        legacy_databricks_runs,
        "databricks_workspace_config_from_env",
        lambda **kwargs: legacy_databricks_runs.DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
    )

    exit_code = legacy_databricks_runs.main(
        [
            "--output-json",
            str(output_path),
            "get",
            "--run-id",
            "123",
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["response"] == {"run_id": 123}


def test_legacy_databricks_runs_private_summary_hook_is_isolated(monkeypatch):
    run = {
        "run_id": 123,
        "state": {"life_cycle_state": "RUNNING"},
        "tasks": [{"task_key": "real-task", "state": {"life_cycle_state": "RUNNING"}}],
    }

    monkeypatch.setattr(
        legacy_databricks_runs,
        "_task_summary",
        lambda task: {
            "task_key": "patched-task",
            "life_cycle_state": "RUNNING",
            "result_state": None,
            "state_message": None,
            "cluster_id": None,
            "start_time": None,
            "end_time": None,
        },
    )

    legacy_summary = legacy_databricks_runs.summarize_databricks_run(run)
    public_summary = public_databricks_runs.summarize_databricks_run(run)

    assert legacy_summary["active_task_key"] == "patched-task"
    assert legacy_summary["tasks"][0]["task_key"] == "patched-task"
    assert public_summary["active_task_key"] == "real-task"
    assert public_summary["tasks"][0]["task_key"] == "real-task"


def test_legacy_databricks_runs_ignores_preimport_document_env_helper_monkeypatch():
    env = {
        **os.environ,
        "PYTHONPATH": "src",
    }
    script = """
import json
import document_kv_cache.databricks_runs as public_runs

def broken_env_helper(**kwargs):
    raise RuntimeError("unexpected import-order env helper")

public_runs.databricks_workspace_config_from_env = broken_env_helper

import restaurant_kv_serving.databricks_runs as legacy_runs

config = legacy_runs.databricks_workspace_config_from_env(
    environ={
        "DATABRICKS_HOST": "https://dbc.example/",
        "DATABRICKS_TOKEN": "secret-token",
    },
)
print(json.dumps({"host": config.normalized_host, "module": type(config).__module__}))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "host": "https://dbc.example",
        "module": "restaurant_kv_serving.databricks_runs",
    }


def test_legacy_databricks_runs_uses_source_config_base_when_public_class_is_replaced_before_import():
    env = {
        **os.environ,
        "PYTHONPATH": "src",
    }
    script = """
import json
import document_kv_cache.databricks_runs as public_runs

public_runs.DatabricksWorkspaceConfig = object

import restaurant_kv_serving.databricks_runs as legacy_runs

config = legacy_runs.DatabricksWorkspaceConfig("https://dbc.example/", "secret-token")
print(json.dumps({"host": config.normalized_host, "module": type(config).__module__}))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "host": "https://dbc.example",
        "module": "restaurant_kv_serving.databricks_runs",
    }


def test_databricks_runs_reexports_document_owned_api_with_legacy_subclass():
    assert public_databricks_runs.DatabricksWorkspaceConfig.__module__ == "document_kv_cache.databricks_runs"
    assert legacy_databricks_runs.DatabricksWorkspaceConfig.__module__ == "restaurant_kv_serving.databricks_runs"
    assert issubclass(legacy_databricks_runs.DatabricksWorkspaceConfig, public_databricks_runs.DatabricksWorkspaceConfig)
    assert legacy_databricks_runs.DatabricksHTTPResponse is public_databricks_runs.DatabricksHTTPResponse
    assert legacy_databricks_runs.DatabricksURLOpener is public_databricks_runs.DatabricksURLOpener
    assert public_databricks_runs.summarize_databricks_run.__module__ == "document_kv_cache.databricks_runs"
    assert legacy_databricks_runs.summarize_databricks_run.__module__ == "restaurant_kv_serving.databricks_runs"
    assert set(public_databricks_runs.__all__) < set(legacy_databricks_runs.__all__)
    assert "urllib" not in public_databricks_runs.__all__
    assert "urllib" in legacy_databricks_runs.__all__


def test_legacy_workspace_config_factory_returns_picklable_legacy_config():
    config = legacy_databricks_runs.databricks_workspace_config_from_env(
        environ={
            DEFAULT_DATABRICKS_HOST_ENV: "https://dbc.example/",
            DEFAULT_DATABRICKS_TOKEN_ENV: "secret-token",
        }
    )

    round_tripped = pickle.loads(pickle.dumps(config))

    assert type(config) is legacy_databricks_runs.DatabricksWorkspaceConfig
    assert isinstance(config, public_databricks_runs.DatabricksWorkspaceConfig)
    assert type(round_tripped) is legacy_databricks_runs.DatabricksWorkspaceConfig
    assert round_tripped.normalized_host == "https://dbc.example"
    assert "secret-token" not in repr(round_tripped)
    assert not hasattr(config, "__dict__")


def test_databricks_runs_star_import_surfaces_are_stable():
    expected_legacy_exports = {
        "DEFAULT_DATABRICKS_HOST_ENV",
        "DEFAULT_DATABRICKS_TOKEN_ENV",
        "DEFAULT_DATABRICKS_TIMEOUT_SECONDS",
        "DATABRICKS_RUN_STATUS_RECORD_TYPE",
        "DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE",
        "DATABRICKS_TERMINAL_LIFE_CYCLE_STATES",
        "DatabricksHTTPResponse",
        "DatabricksURLOpener",
        "DatabricksWorkspaceConfig",
        "databricks_workspace_config_from_env",
        "submit_databricks_run",
        "get_databricks_run",
        "summarize_databricks_run",
        "summarize_databricks_run_submit_payload",
        "databricks_run_status_record",
        "databricks_run_status_sidecar_issues",
        "validate_databricks_run_status_sidecar",
        "write_databricks_run_response_json",
        "read_databricks_run_submit_payload",
        "main",
        "argparse",
        "hashlib",
        "Mapping",
        "Sequence",
        "dataclass",
        "field",
        "json",
        "os",
        "Path",
        "Any",
        "Protocol",
        "urllib",
    }
    public_namespace: dict[str, object] = {}
    legacy_namespace: dict[str, object] = {}

    exec("from document_kv_cache.databricks_runs import *", public_namespace)
    exec("from restaurant_kv_serving.databricks_runs import *", legacy_namespace)

    assert set(public_databricks_runs.__all__) == set(public_namespace) - {"__builtins__"}
    assert set(legacy_databricks_runs.__all__) == expected_legacy_exports
    assert set(legacy_databricks_runs.__all__) == set(legacy_namespace) - {"__builtins__"}
    assert "RLock" not in legacy_namespace
    assert "urllib" not in public_namespace


def test_legacy_databricks_runs_module_execution_help():
    result = subprocess.run(
        [sys.executable, "-m", "restaurant_kv_serving.databricks_runs", "--help"],
        capture_output=True,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Submit or inspect Databricks runs" in result.stdout


class _FakeResponse:
    status = 200

    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


class _FakeOpener:
    def __init__(self, payload):
        self._payload = payload
        self.requests = []
        self.timeouts = []

    def __call__(self, request, *, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        return _FakeResponse(self._payload)


class _HTTPErrorOpener:
    def __init__(self, error):
        self._error = error

    def __call__(self, request, *, timeout):
        raise self._error


class _BytesFile:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload

    def close(self):
        pass


def _valid_databricks_run_status_record():
    return summarize_databricks_run(
        {
            "run_id": 123,
            "run_name": "document-kv-v1",
            "run_page_url": "https://dbc.example/#job/123/run/123",
            "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
            "start_time": 1000,
            "end_time": 2000,
            "cluster_instance": {"cluster_id": "cluster-main"},
            "tasks": [
                {
                    "task_key": "run-benchmark",
                    "run_id": 124,
                    "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
                    "cluster_instance": {"cluster_id": "cluster-task"},
                    "start_time": 1001,
                    "end_time": 1999,
                }
            ],
        },
        submit_payload=_single_node_g5_submit_payload(),
        submit_payload_path="/Volumes/catalog/schema/volume/databricks-run-submit.json",
    )


def _single_node_g5_submit_payload():
    return {
        "run_name": "document-kv-v1",
        "tasks": [
            {
                "task_key": "run-benchmark",
                "new_cluster": {
                    "spark_version": "15.4.x-gpu-ml-scala2.12",
                    "node_type_id": "g5.4xlarge",
                    "driver_node_type_id": "g5.4xlarge",
                    "num_workers": 0,
                    "data_security_mode": "SINGLE_USER",
                    "custom_tags": {
                        "ResourceClass": "SingleNode",
                        "purpose": "document-kv-benchmark",
                    },
                },
            }
        ],
    }
