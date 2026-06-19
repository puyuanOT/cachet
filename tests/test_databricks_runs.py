import json
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
    databricks_workspace_config_from_env,
    get_databricks_run,
    main,
    read_databricks_run_submit_payload,
    submit_databricks_run,
    summarize_databricks_run,
    summarize_databricks_run_submit_payload,
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
    assert submit_payload["aws_g5_node_type"] is True
    assert submit_payload["node_type_ids"] == ["g5.4xlarge"]
    assert submit_payload["data_security_modes"] == ["SINGLE_USER"]
    assert submit_payload["task_keys"] == ["run-benchmark"]


def test_summarize_databricks_run_submit_payload_reports_non_g5_multi_node_payload():
    payload = _single_node_g5_submit_payload()
    payload["tasks"][0]["new_cluster"]["node_type_id"] = "g6.4xlarge"
    payload["tasks"][0]["new_cluster"]["num_workers"] = 1

    summary = summarize_databricks_run_submit_payload(payload)

    assert summary["record_type"] == DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE
    assert summary["single_node"] is False
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
