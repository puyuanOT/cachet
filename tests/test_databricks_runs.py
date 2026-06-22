import hashlib
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import types
import urllib.error

import pytest

import document_kv_cache.databricks_runs as public_databricks_runs
import restaurant_kv_serving.databricks_runs as legacy_databricks_runs
from document_kv_cache._hardware_targets import (
    HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES,
    SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES,
)
from document_kv_cache.databricks_runs import (
    DEFAULT_DATABRICKS_CONFIG_FILE,
    DEFAULT_DATABRICKS_HOST_ENV,
    DEFAULT_DATABRICKS_TOKEN_ENV,
    DATABRICKS_PROFILE_AUTH_MODES,
    DATABRICKS_AUTH_CHECK_RECORD_TYPE,
    DATABRICKS_DBFS_PUT_MAX_CONTENT_BYTES,
    DATABRICKS_RUN_STATUS_RECORD_TYPE,
    DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE,
    DatabricksWorkspaceConfig,
    check_databricks_auth,
    databricks_run_status_record,
    databricks_run_status_sidecar_issues,
    databricks_workspace_config_from_env,
    databricks_workspace_config_from_profile,
    databricks_workspace_config_from_sdk_profile,
    get_databricks_run,
    main,
    plan_databricks_stage_and_submit,
    put_databricks_dbfs_file,
    read_databricks_run_submit_payload,
    stage_and_submit_databricks_run,
    submit_databricks_run,
    summarize_databricks_run,
    summarize_databricks_run_submit_payload,
    validate_databricks_run_status_sidecar,
    write_databricks_run_response_json,
)


def test_databricks_run_status_uses_shared_hardware_target_prefixes():
    assert (
        public_databricks_runs._SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES
        == SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES
    )
    assert (
        public_databricks_runs._HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES
        == HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES
    )
    assert DATABRICKS_PROFILE_AUTH_MODES == ("auto", "static", "sdk")
    assert public_databricks_runs.DATABRICKS_PROFILE_AUTH_MODES == DATABRICKS_PROFILE_AUTH_MODES


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


def test_workspace_config_from_profile_reads_databricks_cli_config(tmp_path):
    config_path = tmp_path / ".databrickscfg"
    config_path.write_text(
        "[QA]\n"
        "host = https://dbc.example.cloud.databricks.com/\n"
        "token = secret-token\n",
        encoding="utf-8",
    )

    config = databricks_workspace_config_from_profile(
        "QA",
        config_file=config_path,
        timeout_seconds=17,
    )

    assert config.normalized_host == "https://dbc.example.cloud.databricks.com"
    assert config.timeout_seconds == 17
    assert "secret-token" not in repr(config)


def test_workspace_config_from_profile_supports_default_section(tmp_path):
    config_path = tmp_path / ".databrickscfg"
    config_path.write_text(
        "[DEFAULT]\n"
        "host = https://dbc.example.cloud.databricks.com\n"
        "token = default-secret-token\n",
        encoding="utf-8",
    )

    config = databricks_workspace_config_from_profile("DEFAULT", config_file=config_path)

    assert config.normalized_host == "https://dbc.example.cloud.databricks.com"
    assert "default-secret-token" not in repr(config)


def test_workspace_config_from_profile_does_not_inherit_default_credentials(tmp_path):
    config_path = tmp_path / ".databrickscfg"
    config_path.write_text(
        "[DEFAULT]\n"
        "host = https://default.example.cloud.databricks.com\n"
        "token = default-secret-token\n"
        "[MISSING_HOST]\n"
        "token = profile-secret-token\n"
        "[MISSING_TOKEN]\n"
        "host = https://profile.example.cloud.databricks.com\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing host"):
        databricks_workspace_config_from_profile("MISSING_HOST", config_file=config_path)

    with pytest.raises(ValueError, match="missing token"):
        databricks_workspace_config_from_profile("MISSING_TOKEN", config_file=config_path)

    config = databricks_workspace_config_from_profile("DEFAULT", config_file=config_path)

    assert config.normalized_host == "https://default.example.cloud.databricks.com"
    assert "default-secret-token" not in repr(config)


def test_workspace_config_from_profile_validates_profile_file_host_and_token(tmp_path):
    missing_config = tmp_path / "missing.cfg"
    with pytest.raises(ValueError, match="was not found"):
        databricks_workspace_config_from_profile("QA", config_file=missing_config)

    config_path = tmp_path / ".databrickscfg"
    config_path.write_text("[QA]\nhost = https://dbc.example\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing token"):
        databricks_workspace_config_from_profile("QA", config_file=config_path)

    config_path.write_text("[QA]\ntoken = secret-token\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing host"):
        databricks_workspace_config_from_profile("QA", config_file=config_path)

    with pytest.raises(ValueError, match="was not found"):
        databricks_workspace_config_from_profile("OTHER", config_file=config_path)

    with pytest.raises(ValueError, match="profile must be"):
        databricks_workspace_config_from_profile("", config_file=config_path)


def test_workspace_config_from_profile_uses_sdk_for_oauth_profile(tmp_path, monkeypatch):
    config_path = tmp_path / ".databrickscfg"
    config_path.write_text(
        "[QA_OAUTH]\n"
        "host = https://profile.example.cloud.databricks.com\n"
        "auth_type = databricks-cli\n"
        "workspace_id = 123456\n",
        encoding="utf-8",
    )
    calls = []

    class FakeSdkConfig:
        host = "https://resolved.example.cloud.databricks.com/"

        def authenticate(self):
            return {"Authorization": "Bearer oauth-secret-token"}

    def fake_sdk_config(profile, *, config_file, timeout_seconds):
        calls.append((profile, config_file, timeout_seconds))
        return FakeSdkConfig()

    monkeypatch.setattr(public_databricks_runs, "_databricks_sdk_config", fake_sdk_config)

    config = databricks_workspace_config_from_profile(
        "QA_OAUTH",
        config_file=config_path,
        timeout_seconds=11,
    )

    assert calls == [("QA_OAUTH", config_path, 11)]
    assert config.normalized_host == "https://resolved.example.cloud.databricks.com"
    assert config.timeout_seconds == 11
    assert config.token == "oauth-secret-token"
    assert "oauth-secret-token" not in repr(config)


def test_workspace_config_from_profile_can_force_sdk_when_profile_has_token(tmp_path, monkeypatch):
    config_path = tmp_path / ".databrickscfg"
    config_path.write_text(
        "[QA_OAUTH]\n"
        "host = https://profile.example.cloud.databricks.com\n"
        "token = stale-static-token\n"
        "auth_type = databricks-cli\n",
        encoding="utf-8",
    )
    calls = []

    class FakeSdkConfig:
        host = "https://refreshed.example.cloud.databricks.com/"

        def authenticate(self):
            return {"Authorization": "Bearer refreshed-oauth-token"}

    def fake_sdk_config(profile, *, config_file, timeout_seconds):
        calls.append((profile, config_file, timeout_seconds))
        return FakeSdkConfig()

    monkeypatch.setattr(public_databricks_runs, "_databricks_sdk_config", fake_sdk_config)

    config = databricks_workspace_config_from_profile(
        "QA_OAUTH",
        config_file=config_path,
        timeout_seconds=23,
        profile_auth_mode="sdk",
    )

    assert calls == [("QA_OAUTH", config_path, 23)]
    assert config.normalized_host == "https://refreshed.example.cloud.databricks.com"
    assert config.timeout_seconds == 23
    assert config.token == "refreshed-oauth-token"


def test_workspace_config_from_profile_static_mode_does_not_fall_back_to_sdk(tmp_path, monkeypatch):
    config_path = tmp_path / ".databrickscfg"
    config_path.write_text(
        "[QA_OAUTH]\n"
        "host = https://profile.example.cloud.databricks.com\n"
        "auth_type = databricks-cli\n",
        encoding="utf-8",
    )

    def sdk_should_not_run(*args, **kwargs):
        raise AssertionError("SDK auth should not run in static profile auth mode")

    monkeypatch.setattr(public_databricks_runs, "_databricks_sdk_config", sdk_should_not_run)

    with pytest.raises(ValueError, match="missing token"):
        databricks_workspace_config_from_profile(
            "QA_OAUTH",
            config_file=config_path,
            profile_auth_mode="static",
        )


def test_workspace_config_from_profile_rejects_unknown_profile_auth_mode(tmp_path):
    config_path = tmp_path / ".databrickscfg"
    config_path.write_text(
        "[QA]\n"
        "host = https://dbc.example.cloud.databricks.com\n"
        "token = profile-secret-token\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="profile_auth_mode"):
        databricks_workspace_config_from_profile(
            "QA",
            config_file=config_path,
            profile_auth_mode="refresh",
        )


def test_workspace_config_from_sdk_profile_requires_bearer_authorization(tmp_path, monkeypatch):
    config_path = tmp_path / ".databrickscfg"
    config_path.write_text(
        "[QA_OAUTH]\n"
        "host = https://profile.example.cloud.databricks.com\n"
        "auth_type = databricks-cli\n",
        encoding="utf-8",
    )

    class FakeSdkConfig:
        host = "https://profile.example.cloud.databricks.com"

        def authenticate(self):
            return {"Authorization": "Basic not-supported"}

    monkeypatch.setattr(
        public_databricks_runs,
        "_databricks_sdk_config",
        lambda *args, **kwargs: FakeSdkConfig(),
    )

    with pytest.raises(ValueError, match="Bearer Authorization"):
        databricks_workspace_config_from_sdk_profile("QA_OAUTH", config_file=config_path)


def test_workspace_config_from_sdk_profile_redacts_sdk_load_errors(tmp_path, monkeypatch):
    config_path = tmp_path / ".databrickscfg"
    config_path.write_text(
        "[QA_OAUTH]\n"
        "host = https://profile.example.cloud.databricks.com\n"
        "auth_type = databricks-cli\n",
        encoding="utf-8",
    )

    def raise_sdk_error(*args, **kwargs):
        raise ValueError("Bearer sdk-secret-token")

    monkeypatch.setattr(public_databricks_runs, "_databricks_sdk_config", raise_sdk_error)

    with pytest.raises(ValueError) as exc_info:
        databricks_workspace_config_from_sdk_profile("QA_OAUTH", config_file=config_path)

    message = str(exc_info.value)
    assert "sdk-secret-token" not in message
    assert "Bearer [REDACTED]" in message


def test_workspace_config_from_sdk_profile_redacts_authenticate_errors(tmp_path, monkeypatch):
    config_path = tmp_path / ".databrickscfg"
    config_path.write_text(
        "[QA_OAUTH]\n"
        "host = https://profile.example.cloud.databricks.com\n"
        "auth_type = databricks-cli\n",
        encoding="utf-8",
    )

    class FakeSdkConfig:
        host = "https://profile.example.cloud.databricks.com"

        def authenticate(self):
            raise RuntimeError("Bearer refresh-secret-token")

    monkeypatch.setattr(
        public_databricks_runs,
        "_databricks_sdk_config",
        lambda *args, **kwargs: FakeSdkConfig(),
    )

    with pytest.raises(ValueError) as exc_info:
        databricks_workspace_config_from_sdk_profile("QA_OAUTH", config_file=config_path)

    message = str(exc_info.value)
    assert "refresh-secret-token" not in message
    assert "could not authenticate: Bearer [REDACTED]" in message


def test_sdk_profile_config_ignores_ambient_databricks_auth_env(monkeypatch, tmp_path):
    config_path = tmp_path / ".databrickscfg"
    config_path.write_text(
        "[QA_OAUTH]\n"
        "host = https://profile.example.cloud.databricks.com\n"
        "auth_type = databricks-cli\n",
        encoding="utf-8",
    )
    observed_env = []

    class FakeSdkAttribute:
        def __init__(self, name, env=None, auth=None, env_aliases=()):
            self.name = name
            self.env = env
            self.auth = auth
            self.env_aliases = env_aliases

    class FakeSdkConfig:
        def __init__(self, **kwargs):
            observed_env.append(
                {
                    "DATABRICKS_AUTH_TYPE": os.environ.get("DATABRICKS_AUTH_TYPE"),
                    "DATABRICKS_CLI_PATH": os.environ.get("DATABRICKS_CLI_PATH"),
                    "DATABRICKS_CONFIG_FILE": os.environ.get("DATABRICKS_CONFIG_FILE"),
                    "DATABRICKS_CONFIG_PROFILE": os.environ.get("DATABRICKS_CONFIG_PROFILE"),
                    "DATABRICKS_HOST": os.environ.get("DATABRICKS_HOST"),
                    "DATABRICKS_TOKEN": os.environ.get("DATABRICKS_TOKEN"),
                }
            )
            self.kwargs = kwargs
            self.host = "https://profile.example.cloud.databricks.com"

        @classmethod
        def attributes(cls):
            return [
                FakeSdkAttribute("auth_type", "DATABRICKS_AUTH_TYPE"),
                FakeSdkAttribute("config_file", "DATABRICKS_CONFIG_FILE"),
                FakeSdkAttribute("databricks_cli_path", "DATABRICKS_CLI_PATH"),
                FakeSdkAttribute("host", "DATABRICKS_HOST"),
                FakeSdkAttribute("profile", "DATABRICKS_CONFIG_PROFILE"),
                FakeSdkAttribute("token", "DATABRICKS_TOKEN", auth="pat"),
            ]

        def authenticate(self):
            return {"Authorization": "Bearer profile-oauth-token"}

    databricks_module = types.ModuleType("databricks")
    sdk_module = types.ModuleType("databricks.sdk")
    core_module = types.ModuleType("databricks.sdk.core")
    core_module.Config = FakeSdkConfig
    databricks_module.sdk = sdk_module
    sdk_module.core = core_module
    monkeypatch.setitem(sys.modules, "databricks", databricks_module)
    monkeypatch.setitem(sys.modules, "databricks.sdk", sdk_module)
    monkeypatch.setitem(sys.modules, "databricks.sdk.core", core_module)
    monkeypatch.setenv("DATABRICKS_AUTH_TYPE", "pat")
    monkeypatch.setenv("DATABRICKS_CLI_PATH", "/custom/databricks")
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", "/ambient/.databrickscfg")
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "AMBIENT")
    monkeypatch.setenv("DATABRICKS_HOST", "https://ambient.example.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "ambient-secret-token")

    config = databricks_workspace_config_from_profile("QA_OAUTH", config_file=config_path)

    assert config.normalized_host == "https://profile.example.cloud.databricks.com"
    assert config.token == "profile-oauth-token"
    assert observed_env == [
        {
            "DATABRICKS_AUTH_TYPE": None,
            "DATABRICKS_CLI_PATH": "/custom/databricks",
            "DATABRICKS_CONFIG_FILE": None,
            "DATABRICKS_CONFIG_PROFILE": None,
            "DATABRICKS_HOST": None,
            "DATABRICKS_TOKEN": None,
        }
    ]
    assert os.environ["DATABRICKS_AUTH_TYPE"] == "pat"
    assert os.environ["DATABRICKS_CONFIG_FILE"] == "/ambient/.databrickscfg"
    assert os.environ["DATABRICKS_CONFIG_PROFILE"] == "AMBIENT"
    assert os.environ["DATABRICKS_HOST"] == "https://ambient.example.cloud.databricks.com"
    assert os.environ["DATABRICKS_TOKEN"] == "ambient-secret-token"


def test_check_databricks_auth_calls_identity_endpoint_without_user_pii():
    opener = _FakeOpener(
        {
            "id": "123",
            "userName": "person@example.com",
            "displayName": "Person Example",
        }
    )
    config = DatabricksWorkspaceConfig("https://dbc.example/", "secret-token", timeout_seconds=9)

    record = check_databricks_auth(config, opener=opener)

    assert record == {
        "record_type": DATABRICKS_AUTH_CHECK_RECORD_TYPE,
        "authenticated": True,
        "endpoint": "/api/2.0/preview/scim/v2/Me",
        "http_status": 200,
        "workspace_host_sha256": hashlib.sha256(b"https://dbc.example").hexdigest(),
        "response_keys": ["displayName", "id", "userName"],
    }
    request = opener.requests[0]
    assert request.full_url == "https://dbc.example/api/2.0/preview/scim/v2/Me"
    assert request.get_method() == "GET"
    assert request.data is None
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert opener.timeouts == [9]
    serialized = json.dumps(record, sort_keys=True)
    assert "person@example.com" not in serialized
    assert "Person Example" not in serialized
    assert "secret-token" not in serialized


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


def test_put_databricks_dbfs_file_posts_base64_payload_with_bearer_token(tmp_path):
    local_path = tmp_path / "cachet.whl"
    local_path.write_bytes(b"wheel-bytes")
    opener = _FakeOpener({})
    config = DatabricksWorkspaceConfig("https://dbc.example/", "secret-token", timeout_seconds=9)

    response = put_databricks_dbfs_file(
        config,
        local_path,
        "dbfs:/FileStore/cachet/cachet.whl",
        overwrite=True,
        opener=opener,
    )

    assert response == {}
    request = opener.requests[0]
    assert request.full_url == "https://dbc.example/api/2.0/dbfs/put"
    assert request.get_method() == "POST"
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert json.loads(request.data.decode("utf-8")) == {
        "path": "/FileStore/cachet/cachet.whl",
        "contents": "d2hlZWwtYnl0ZXM=",
        "overwrite": True,
    }
    assert opener.timeouts == [9]


def test_put_databricks_dbfs_file_accepts_absolute_dbfs_api_path(tmp_path):
    local_path = tmp_path / "runner.py"
    local_path.write_text("print('ok')\n", encoding="utf-8")
    opener = _FakeOpener({})
    config = DatabricksWorkspaceConfig("https://dbc.example/", "secret-token")

    put_databricks_dbfs_file(config, local_path, "/FileStore/cachet/runner.py", opener=opener)

    request = opener.requests[0]
    assert json.loads(request.data.decode("utf-8"))["path"] == "/FileStore/cachet/runner.py"


def test_put_databricks_dbfs_file_rejects_relative_dbfs_path_before_network(tmp_path):
    local_path = tmp_path / "runner.py"
    local_path.write_text("print('ok')\n", encoding="utf-8")
    opener = _FakeOpener({})
    config = DatabricksWorkspaceConfig("https://dbc.example/", "secret-token")

    with pytest.raises(ValueError, match="absolute DBFS path"):
        put_databricks_dbfs_file(config, local_path, "FileStore/cachet/runner.py", opener=opener)

    assert opener.requests == []


def test_put_databricks_dbfs_file_rejects_large_base64_put_payload(tmp_path):
    local_path = tmp_path / "large.whl"
    local_path.write_bytes(b"x" * DATABRICKS_DBFS_PUT_MAX_CONTENT_BYTES)
    opener = _FakeOpener({})
    config = DatabricksWorkspaceConfig("https://dbc.example/", "secret-token")

    with pytest.raises(ValueError, match="base64 bytes"):
        put_databricks_dbfs_file(config, local_path, "dbfs:/FileStore/cachet/large.whl", opener=opener)

    assert opener.requests == []


def test_stage_and_submit_databricks_run_uploads_artifacts_then_submits_payload(tmp_path):
    runner_path = tmp_path / "run_engine_probe.py"
    wheel_path = tmp_path / "document_kv_cache-0.2.0-py3-none-any.whl"
    runner_path.write_text("print('cachet')\n", encoding="utf-8")
    wheel_path.write_bytes(b"wheel-bytes")
    payload = _dbfs_artifact_submit_payload()
    opener = _SequentialOpener(({}, {}, {"run_id": 123}))
    config = DatabricksWorkspaceConfig("https://dbc.example/", "secret-token", timeout_seconds=9)

    record = stage_and_submit_databricks_run(
        config,
        payload,
        (
            (runner_path, "dbfs:/cachet/run_engine_probe.py"),
            (wheel_path, "/cachet/document_kv_cache-0.2.0-py3-none-any.whl"),
        ),
        overwrite=True,
        require_payload_dbfs_artifacts=True,
        opener=opener,
    )

    assert record["ok"] is True
    assert record["action"] == "stage-and-submit"
    assert record["response"] == {"run_id": 123}
    assert [request.full_url for request in opener.requests] == [
        "https://dbc.example/api/2.0/dbfs/put",
        "https://dbc.example/api/2.0/dbfs/put",
        "https://dbc.example/api/2.1/jobs/runs/submit",
    ]
    assert [request.get_method() for request in opener.requests] == ["POST", "POST", "POST"]
    assert json.loads(opener.requests[0].data.decode("utf-8"))["path"] == "/cachet/run_engine_probe.py"
    assert json.loads(opener.requests[1].data.decode("utf-8"))["path"] == (
        "/cachet/document_kv_cache-0.2.0-py3-none-any.whl"
    )
    assert json.loads(opener.requests[2].data.decode("utf-8")) == payload
    assert opener.timeouts == [9, 9, 9]
    assert [upload["artifact"]["dbfs_path"] for upload in record["artifact_uploads"]] == [
        "dbfs:/cachet/run_engine_probe.py",
        "dbfs:/cachet/document_kv_cache-0.2.0-py3-none-any.whl",
    ]


def test_stage_and_submit_databricks_run_can_preflight_auth_before_uploads(tmp_path):
    runner_path = tmp_path / "run_engine_probe.py"
    wheel_path = tmp_path / "document_kv_cache-0.2.0-py3-none-any.whl"
    runner_path.write_text("print('cachet')\n", encoding="utf-8")
    wheel_path.write_bytes(b"wheel-bytes")
    payload = _dbfs_artifact_submit_payload()
    opener = _SequentialOpener(
        (
            {"userName": "person@example.com", "id": "abc"},
            {},
            {},
            {"run_id": 123},
        )
    )
    config = DatabricksWorkspaceConfig("https://dbc.example/", "secret-token", timeout_seconds=9)

    record = stage_and_submit_databricks_run(
        config,
        payload,
        (
            (runner_path, "dbfs:/cachet/run_engine_probe.py"),
            (wheel_path, "dbfs:/cachet/document_kv_cache-0.2.0-py3-none-any.whl"),
        ),
        overwrite=True,
        require_payload_dbfs_artifacts=True,
        preflight_auth_check=True,
        opener=opener,
    )

    assert record["ok"] is True
    assert record["auth"]["record_type"] == DATABRICKS_AUTH_CHECK_RECORD_TYPE
    assert record["auth"]["response_keys"] == ["id", "userName"]
    assert record["response"] == {"run_id": 123}
    assert [request.full_url for request in opener.requests] == [
        "https://dbc.example/api/2.0/preview/scim/v2/Me",
        "https://dbc.example/api/2.0/dbfs/put",
        "https://dbc.example/api/2.0/dbfs/put",
        "https://dbc.example/api/2.1/jobs/runs/submit",
    ]
    assert [request.get_method() for request in opener.requests] == ["GET", "POST", "POST", "POST"]
    serialized = json.dumps(record, sort_keys=True)
    assert "person@example.com" not in serialized
    assert "secret-token" not in serialized


def test_stage_and_submit_databricks_run_stops_on_failed_preflight_auth(tmp_path):
    runner_path = tmp_path / "run_engine_probe.py"
    wheel_path = tmp_path / "document_kv_cache-0.2.0-py3-none-any.whl"
    runner_path.write_text("print('cachet')\n", encoding="utf-8")
    wheel_path.write_bytes(b"wheel-bytes")
    error = urllib.error.HTTPError(
        "https://dbc.example/api/2.0/preview/scim/v2/Me",
        403,
        "Forbidden",
        {},
        _BytesFile(b'{"message":"Authorization: Bearer secret-token"}'),
    )
    opener = _RecordingHTTPErrorOpener(error)
    config = DatabricksWorkspaceConfig("https://dbc.example/", "secret-token")

    with pytest.raises(RuntimeError) as exc_info:
        stage_and_submit_databricks_run(
            config,
            _dbfs_artifact_submit_payload(),
            (
                (runner_path, "dbfs:/cachet/run_engine_probe.py"),
                (wheel_path, "dbfs:/cachet/document_kv_cache-0.2.0-py3-none-any.whl"),
            ),
            require_payload_dbfs_artifacts=True,
            preflight_auth_check=True,
            opener=opener,
        )

    assert "secret-token" not in str(exc_info.value)
    assert "Bearer [REDACTED]" in str(exc_info.value)
    assert [request.full_url for request in opener.requests] == [
        "https://dbc.example/api/2.0/preview/scim/v2/Me"
    ]


def test_stage_and_submit_databricks_run_rejects_unstaged_payload_dbfs_uri_before_network(tmp_path):
    runner_path = tmp_path / "run_engine_probe.py"
    runner_path.write_text("print('cachet')\n", encoding="utf-8")
    opener = _FakeOpener({})
    config = DatabricksWorkspaceConfig("https://dbc.example/", "secret-token")

    with pytest.raises(ValueError, match="without staged artifacts"):
        stage_and_submit_databricks_run(
            config,
            _dbfs_artifact_submit_payload(),
            ((runner_path, "dbfs:/cachet/run_engine_probe.py"),),
            require_payload_dbfs_artifacts=True,
            opener=opener,
        )

    assert opener.requests == []


def test_stage_and_submit_databricks_run_validates_all_artifacts_before_upload(tmp_path):
    runner_path = tmp_path / "run_engine_probe.py"
    missing_wheel_path = tmp_path / "missing.whl"
    runner_path.write_text("print('cachet')\n", encoding="utf-8")
    opener = _FakeOpener({})
    config = DatabricksWorkspaceConfig("https://dbc.example/", "secret-token")

    with pytest.raises(ValueError, match="local_path must be an existing file"):
        stage_and_submit_databricks_run(
            config,
            _dbfs_artifact_submit_payload(),
            (
                (runner_path, "dbfs:/cachet/run_engine_probe.py"),
                (missing_wheel_path, "dbfs:/cachet/document_kv_cache-0.2.0-py3-none-any.whl"),
            ),
            require_payload_dbfs_artifacts=True,
            opener=opener,
        )

    assert opener.requests == []


def test_plan_databricks_stage_and_submit_validates_artifacts_without_network(tmp_path):
    runner_path = tmp_path / "run_engine_probe.py"
    wheel_path = tmp_path / "document_kv_cache-0.2.0-py3-none-any.whl"
    runner_path.write_text("print('cachet')\n", encoding="utf-8")
    wheel_path.write_bytes(b"wheel-bytes")

    record = plan_databricks_stage_and_submit(
        _dbfs_artifact_submit_payload(),
        (
            (runner_path, "dbfs:/cachet/run_engine_probe.py"),
            (wheel_path, "dbfs:/cachet/document_kv_cache-0.2.0-py3-none-any.whl"),
        ),
        overwrite=True,
        require_payload_dbfs_artifacts=True,
        submit_payload_path="/tmp/payload.json",
    )

    assert record["ok"] is True
    assert record["action"] == "stage-and-submit-plan"
    assert "response" not in record
    assert record["submit_payload"]["source_path"] == "/tmp/payload.json"
    assert record["submit_payload"]["task_keys"] == ["document_kv_engine_probe"]
    assert [upload["artifact"]["dbfs_path"] for upload in record["artifact_uploads"]] == [
        "dbfs:/cachet/run_engine_probe.py",
        "dbfs:/cachet/document_kv_cache-0.2.0-py3-none-any.whl",
    ]
    assert record["artifact_uploads"][0]["upload_request"] == {
        "path": "/cachet/run_engine_probe.py",
        "overwrite": True,
        "contents_base64_bytes": len("cHJpbnQoJ2NhY2hldCcpCg=="),
    }


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
                "spark_env_keys": [],
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
                "spark_env_keys": [],
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
    payload = _single_node_g5_submit_payload()
    payload["tasks"][0]["new_cluster"]["spark_env_vars"] = {
        "CACHET_TRANSFORMERS_DEVICE": "cuda",
        "CACHET_TRANSFORMERS_TORCH_DTYPE": "bfloat16",
    }

    summary = summarize_databricks_run(
        {
            "run_id": 123,
            "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
            "tasks": [
                {
                    "task_key": "run-benchmark",
                    "run_id": 124,
                    "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
                    "new_cluster": payload["tasks"][0]["new_cluster"],
                }
            ],
        },
        submit_payload=payload,
        submit_payload_path="/Volumes/catalog/schema/volume/payload.json",
    )

    submit_payload = summary["submit_payload"]
    assert submit_payload["record_type"] == DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE
    assert submit_payload["source_path"] == "/Volumes/catalog/schema/volume/payload.json"
    assert len(submit_payload["sha256"]) == 64
    assert submit_payload["single_node"] is True
    assert submit_payload["aws_single_node_gpu_type"] is True
    assert submit_payload["aws_g5_node_type"] is True
    assert submit_payload["node_type_ids"] == ["g6.4xlarge"]
    assert submit_payload["hardware_targets"] == ["aws-g6-l4"]
    assert submit_payload["data_security_modes"] == ["SINGLE_USER"]
    assert submit_payload["task_keys"] == ["run-benchmark"]
    assert submit_payload["spark_env_keys"] == [
        "CACHET_TRANSFORMERS_DEVICE",
        "CACHET_TRANSFORMERS_TORCH_DTYPE",
    ]
    assert submit_payload["tasks"][0]["spark_env_keys"] == [
        "CACHET_TRANSFORMERS_DEVICE",
        "CACHET_TRANSFORMERS_TORCH_DTYPE",
    ]
    assert summary["tasks"][0]["spark_env_keys"] == [
        "CACHET_TRANSFORMERS_DEVICE",
        "CACHET_TRANSFORMERS_TORCH_DTYPE",
    ]
    serialized_summary = json.dumps(submit_payload, sort_keys=True)
    assert "cuda" not in serialized_summary
    assert "bfloat16" not in serialized_summary
    assert databricks_run_status_sidecar_issues(summary) == ()


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
    assert summary["submit_payload"]["hardware_targets"] == ["aws-g6-l4"]
    assert databricks_run_status_sidecar_issues(summary) == ()
    assert databricks_run_status_sidecar_issues(summary, expected_hardware_target="aws-g6-l4") == ()


def test_databricks_run_status_sidecar_validation_rejects_expected_hardware_target_mismatch():
    status_record = _valid_databricks_run_status_record()

    issues = databricks_run_status_sidecar_issues(
        status_record,
        expected_hardware_target="aws-g5-a10g",
    )

    assert (
        "Databricks run status sidecar submit_payload.tasks[0].node_type_id must match "
        "hardware_target 'aws-g5-a10g'"
        in issues
    )
    assert (
        "Databricks run status sidecar submit_payload.tasks[0].driver_node_type_id must match "
        "hardware_target 'aws-g5-a10g'"
        in issues
    )


def test_validate_databricks_run_status_sidecar_honors_expected_hardware_target():
    status_record = _valid_databricks_run_status_record()

    validate_databricks_run_status_sidecar(status_record, expected_hardware_target="aws-g6-l4")
    with pytest.raises(ValueError, match=r"hardware_target 'aws-g5-a10g'"):
        validate_databricks_run_status_sidecar(status_record, expected_hardware_target="aws-g5-a10g")


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
        "Databricks run status sidecar submit_payload.tasks[0].node_type_id must be a supported V1 AWS GPU node type"
        in issues
    )
    assert "Databricks run status sidecar submit_payload.task_keys must match status task keys" in issues


def test_databricks_run_status_sidecar_validation_rejects_contradictory_gpu_flags():
    status_record = _valid_databricks_run_status_record()
    submit_payload = json.loads(json.dumps(status_record["submit_payload"]))
    submit_payload["aws_single_node_gpu_type"] = True
    submit_payload["aws_g5_node_type"] = False
    bad_record = {**status_record, "submit_payload": submit_payload}

    issues = databricks_run_status_sidecar_issues(bad_record)

    assert "Databricks run status sidecar submit_payload.aws_single_node_gpu_type must be true" in issues
    assert (
        "Databricks run status sidecar submit_payload.aws_single_node_gpu_type and aws_g5_node_type must match"
        in issues
    )


def test_databricks_run_status_sidecar_validation_accepts_legacy_gpu_flag_only():
    status_record = _valid_databricks_run_status_record()
    submit_payload = json.loads(json.dumps(status_record["submit_payload"]))
    del submit_payload["aws_single_node_gpu_type"]
    bad_record = {**status_record, "submit_payload": submit_payload}

    assert databricks_run_status_sidecar_issues(bad_record, expected_hardware_target="aws-g6-l4") == ()
    validate_databricks_run_status_sidecar(bad_record, expected_hardware_target="aws-g6-l4")


def test_databricks_run_status_sidecar_validation_accepts_missing_hardware_targets_for_legacy_sidecars():
    status_record = _valid_databricks_run_status_record()
    submit_payload = json.loads(json.dumps(status_record["submit_payload"]))
    del submit_payload["hardware_targets"]
    legacy_record = {**status_record, "submit_payload": submit_payload}

    assert databricks_run_status_sidecar_issues(legacy_record, expected_hardware_target="aws-g6-l4") == ()
    validate_databricks_run_status_sidecar(legacy_record, expected_hardware_target="aws-g6-l4")


def test_databricks_run_status_sidecar_validation_rejects_null_hardware_targets():
    status_record = _valid_databricks_run_status_record()
    submit_payload = json.loads(json.dumps(status_record["submit_payload"]))
    submit_payload["hardware_targets"] = None
    bad_record = {**status_record, "submit_payload": submit_payload}

    issues = databricks_run_status_sidecar_issues(bad_record)

    assert (
        "Databricks run status sidecar submit_payload.hardware_targets must be an array of non-empty strings"
        in issues
    )


def test_databricks_run_status_sidecar_validation_accepts_g5_hardware_target():
    status_record = _valid_databricks_run_status_record()
    submit_payload = json.loads(json.dumps(status_record["submit_payload"]))
    submit_payload["tasks"][0]["node_type_id"] = "g5.8xlarge"
    submit_payload["tasks"][0]["driver_node_type_id"] = "g5.8xlarge"
    submit_payload["node_type_ids"] = ["g5.8xlarge"]
    submit_payload["driver_node_type_ids"] = ["g5.8xlarge"]
    submit_payload["hardware_targets"] = ["aws-g5-a10g"]
    g5_record = {**status_record, "submit_payload": submit_payload}

    assert databricks_run_status_sidecar_issues(g5_record, expected_hardware_target="aws-g5-a10g") == ()
    validate_databricks_run_status_sidecar(g5_record, expected_hardware_target="aws-g5-a10g")
    assert any(
        "hardware_target 'aws-g6-l4'" in issue
        for issue in databricks_run_status_sidecar_issues(g5_record, expected_hardware_target="aws-g6-l4")
    )


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


def test_databricks_run_status_sidecar_validation_matches_submit_payload_hardware_targets():
    status_record = _valid_databricks_run_status_record()
    submit_payload = json.loads(json.dumps(status_record["submit_payload"]))
    submit_payload["hardware_targets"] = ["aws-g5-a10g"]
    bad_record = {**status_record, "submit_payload": submit_payload}

    issues = databricks_run_status_sidecar_issues(bad_record)

    assert (
        "Databricks run status sidecar submit_payload.hardware_targets must match submit_payload.tasks"
        in issues
    )


@pytest.mark.parametrize(
    ("summary_field", "bad_values"),
    [
        ("node_type_ids", ["g6.12xlarge"]),
        ("driver_node_type_ids", ["g6.12xlarge"]),
        ("hardware_targets", ["aws-g5-a10g"]),
        ("spark_versions", ["15.3.x-gpu-ml-scala2.12"]),
        ("spark_env_keys", ["CACHET_TRANSFORMERS_DEVICE"]),
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


def test_databricks_run_status_sidecar_validation_rejects_malformed_spark_env_keys():
    status_record = _valid_databricks_run_status_record()
    submit_payload = json.loads(json.dumps(status_record["submit_payload"]))
    submit_payload["spark_env_keys"] = ["CACHET_TRANSFORMERS_DEVICE", "DATABRICKS_TOKEN"]
    submit_payload["tasks"][0]["spark_env_keys"] = ["CACHET_TRANSFORMERS_DEVICE", "DATABRICKS_TOKEN"]
    bad_record = {**status_record, "submit_payload": submit_payload}

    issues = databricks_run_status_sidecar_issues(bad_record)

    assert (
        "Databricks run status sidecar submit_payload.spark_env_keys contains secret-looking "
        "environment variable name 'DATABRICKS_TOKEN'"
        in issues
    )
    assert (
        "Databricks run status sidecar submit_payload.tasks[0].spark_env_keys contains "
        "secret-looking environment variable name 'DATABRICKS_TOKEN'"
        in issues
    )


def test_summarize_databricks_run_redacts_token_pattern_spark_env_keys_before_serializing():
    token_like_key = "dapi" + ("0" * 32)
    payload = _single_node_g5_submit_payload()
    payload["tasks"][0]["new_cluster"]["spark_env_vars"] = {token_like_key: "not-serialized"}

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
                    "new_cluster": payload["tasks"][0]["new_cluster"],
                }
            ],
        },
        submit_payload=payload,
        submit_payload_path="/Volumes/catalog/schema/volume/payload.json",
    )

    serialized_summary = json.dumps(summary, sort_keys=True)
    assert token_like_key not in serialized_summary
    assert "not-serialized" not in serialized_summary
    assert summary["tasks"][0]["spark_env_keys"] == ["[REDACTED_DATABRICKS_TOKEN_KEY]"]
    assert summary["submit_payload"]["tasks"][0]["spark_env_keys"] == [
        "[REDACTED_DATABRICKS_TOKEN_KEY]"
    ]
    assert (
        "Databricks run status sidecar tasks[0].spark_env_keys contains redacted "
        "Databricks token-pattern environment variable name"
        in databricks_run_status_sidecar_issues(summary)
    )


def test_databricks_run_status_sidecar_validation_rejects_stale_spark_env_key_claims():
    status_record = _valid_databricks_run_status_record()
    submit_payload = json.loads(json.dumps(status_record["submit_payload"]))
    submit_payload["spark_env_keys"] = ["CACHET_TRANSFORMERS_DEVICE"]
    submit_payload["tasks"][0]["spark_env_keys"] = ["CACHET_TRANSFORMERS_DEVICE"]
    bad_record = {**status_record, "submit_payload": submit_payload}

    issues = databricks_run_status_sidecar_issues(bad_record)

    assert (
        "Databricks run status sidecar submit_payload.spark_env_keys must match submit_payload.tasks"
        not in issues
    )
    assert (
        "Databricks run status sidecar submit_payload.tasks spark_env_keys must match run task "
        "'run-benchmark' spark_env_keys"
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


def test_main_submit_can_use_databricks_profile_without_env(monkeypatch, tmp_path):
    payload_path = tmp_path / "payload.json"
    output_path = tmp_path / "response.json"
    config_path = tmp_path / ".databrickscfg"
    payload_path.write_text('{"run_name":"cachet-profile-smoke"}', encoding="utf-8")
    config_path.write_text(
        "[QA]\n"
        "host = https://dbc.example.cloud.databricks.com/\n"
        "token = profile-secret-token\n",
        encoding="utf-8",
    )

    monkeypatch.delenv(DEFAULT_DATABRICKS_HOST_ENV, raising=False)
    monkeypatch.delenv(DEFAULT_DATABRICKS_TOKEN_ENV, raising=False)
    monkeypatch.setattr(
        legacy_databricks_runs,
        "submit_databricks_run",
        lambda config, payload: {"run_id": 456, "host": config.normalized_host, "payload": payload},
    )

    exit_code = legacy_databricks_runs.main(
        [
            "--profile",
            "QA",
            "--config-file",
            str(config_path),
            "--output-json",
            str(output_path),
            "submit",
            "--payload-json",
            str(payload_path),
        ]
    )

    output = output_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "profile-secret-token" not in output
    assert json.loads(output) == {
        "ok": True,
        "action": "submit",
        "response": {
            "run_id": 456,
            "host": "https://dbc.example.cloud.databricks.com",
            "payload": {"run_name": "cachet-profile-smoke"},
        },
    }


def test_main_submit_can_force_sdk_profile_auth(monkeypatch, tmp_path):
    payload_path = tmp_path / "payload.json"
    output_path = tmp_path / "response.json"
    config_path = tmp_path / ".databrickscfg"
    payload_path.write_text('{"run_name":"cachet-profile-smoke"}', encoding="utf-8")
    config_path.write_text(
        "[QA_OAUTH]\n"
        "host = https://dbc.example.cloud.databricks.com/\n"
        "token = stale-static-token\n"
        "auth_type = databricks-cli\n",
        encoding="utf-8",
    )

    def fake_sdk_profile(profile, *, config_file, timeout_seconds):
        assert profile == "QA_OAUTH"
        assert Path(config_file) == config_path
        assert timeout_seconds == 60.0
        return DatabricksWorkspaceConfig(
            "https://sdk-resolved.example.cloud.databricks.com",
            "refreshed-oauth-token",
            timeout_seconds=timeout_seconds,
        )

    def fake_submit(config, payload):
        assert config.normalized_host == "https://sdk-resolved.example.cloud.databricks.com"
        assert config.token == "refreshed-oauth-token"
        return {"run_id": 789, "host": config.normalized_host, "payload": payload}

    monkeypatch.delenv(DEFAULT_DATABRICKS_HOST_ENV, raising=False)
    monkeypatch.delenv(DEFAULT_DATABRICKS_TOKEN_ENV, raising=False)
    monkeypatch.setattr(legacy_databricks_runs, "databricks_workspace_config_from_sdk_profile", fake_sdk_profile)
    monkeypatch.setattr(legacy_databricks_runs, "submit_databricks_run", fake_submit)

    exit_code = legacy_databricks_runs.main(
        [
            "--profile",
            "QA_OAUTH",
            "--profile-auth-mode",
            "sdk",
            "--config-file",
            str(config_path),
            "--output-json",
            str(output_path),
            "submit",
            "--payload-json",
            str(payload_path),
        ]
    )

    output = output_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "stale-static-token" not in output
    assert "refreshed-oauth-token" not in output
    assert json.loads(output) == {
        "ok": True,
        "action": "submit",
        "response": {
            "run_id": 789,
            "host": "https://sdk-resolved.example.cloud.databricks.com",
            "payload": {"run_name": "cachet-profile-smoke"},
        },
    }


def test_main_auth_check_writes_sanitized_record(monkeypatch, tmp_path):
    output_path = tmp_path / "auth-check.json"
    raw_secret = "secret-token"
    monkeypatch.setenv(DEFAULT_DATABRICKS_HOST_ENV, "https://dbc.example/")
    monkeypatch.setenv(DEFAULT_DATABRICKS_TOKEN_ENV, raw_secret)

    def fake_auth_check(config):
        assert config.normalized_host == "https://dbc.example"
        assert config.token == raw_secret
        return {
            "record_type": DATABRICKS_AUTH_CHECK_RECORD_TYPE,
            "authenticated": True,
            "endpoint": "/api/2.0/preview/scim/v2/Me",
            "http_status": 200,
            "workspace_host_sha256": hashlib.sha256(b"https://dbc.example").hexdigest(),
            "response_keys": ["id", "userName"],
        }

    monkeypatch.setattr(legacy_databricks_runs, "check_databricks_auth", fake_auth_check)

    exit_code = legacy_databricks_runs.main(
        [
            "--output-json",
            str(output_path),
            "auth-check",
        ]
    )

    output = output_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert raw_secret not in output
    assert json.loads(output) == {
        "ok": True,
        "action": "auth-check",
        "auth": {
            "record_type": DATABRICKS_AUTH_CHECK_RECORD_TYPE,
            "authenticated": True,
            "endpoint": "/api/2.0/preview/scim/v2/Me",
            "http_status": 200,
            "workspace_host_sha256": hashlib.sha256(b"https://dbc.example").hexdigest(),
            "response_keys": ["id", "userName"],
        },
    }


def test_main_config_file_requires_profile(tmp_path):
    payload_path = tmp_path / "payload.json"
    output_path = tmp_path / "response.json"
    payload_path.write_text('{"run_name":"cachet-profile-smoke"}', encoding="utf-8")

    exit_code = legacy_databricks_runs.main(
        [
            "--config-file",
            str(tmp_path / ".databrickscfg"),
            "--output-json",
            str(output_path),
            "submit",
            "--payload-json",
            str(payload_path),
        ]
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert record["error_type"] == "ValueError"
    assert "--config-file requires --profile" in record["error"]


def test_main_profile_auth_mode_requires_profile(tmp_path):
    output_path = tmp_path / "response.json"

    exit_code = legacy_databricks_runs.main(
        [
            "--profile-auth-mode",
            "sdk",
            "--output-json",
            str(output_path),
            "auth-check",
        ]
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert record["error_type"] == "ValueError"
    assert "--profile-auth-mode requires --profile" in record["error"]


def test_main_put_dbfs_file_writes_sanitized_artifact_record(monkeypatch, tmp_path):
    local_path = tmp_path / "run_engine_probe.py"
    output_path = tmp_path / "upload.json"
    local_path.write_text("print('cachet')\n", encoding="utf-8")
    raw_secret = "secret-token"

    def fake_api_json(config, method, path_and_query, *, opener, payload=None):
        assert config.normalized_host == "https://dbc.example"
        assert method == "POST"
        assert path_and_query == "/api/2.0/dbfs/put"
        assert payload["path"] == "/FileStore/cachet/run_engine_probe.py"
        assert payload["overwrite"] is True
        return {"status": "ok"}

    monkeypatch.setenv(DEFAULT_DATABRICKS_HOST_ENV, "https://dbc.example")
    monkeypatch.setenv(DEFAULT_DATABRICKS_TOKEN_ENV, raw_secret)
    monkeypatch.setattr(legacy_databricks_runs, "_databricks_api_json", fake_api_json)

    exit_code = legacy_databricks_runs.main(
        [
            "--output-json",
            str(output_path),
            "put-dbfs-file",
            "--local-path",
            str(local_path),
            "--dbfs-path",
            "dbfs:/FileStore/cachet/run_engine_probe.py",
            "--overwrite",
        ]
    )

    output = output_path.read_text(encoding="utf-8")
    record = json.loads(output)
    assert exit_code == 0
    assert raw_secret not in output
    assert record == {
        "ok": True,
        "action": "put-dbfs-file",
        "response": {"status": "ok"},
        "artifact": {
            "local_path": str(local_path),
            "dbfs_path": "dbfs:/FileStore/cachet/run_engine_probe.py",
            "dbfs_api_path": "/FileStore/cachet/run_engine_probe.py",
            "bytes": len("print('cachet')\n".encode("utf-8")),
            "sha256": hashlib.sha256(b"print('cachet')\n").hexdigest(),
        },
    }


def test_main_put_dbfs_file_artifact_record_uses_uploaded_bytes(monkeypatch, tmp_path):
    local_path = tmp_path / "run_engine_probe.py"
    output_path = tmp_path / "upload.json"
    uploaded_bytes = b"print('before')\n"
    changed_bytes = b"print('after')\n"
    local_path.write_bytes(uploaded_bytes)

    def fake_api_json(config, method, path_and_query, *, opener, payload=None):
        assert json.loads(json.dumps(payload))["contents"] == "cHJpbnQoJ2JlZm9yZScpCg=="
        local_path.write_bytes(changed_bytes)
        return {"status": "ok"}

    monkeypatch.setenv(DEFAULT_DATABRICKS_HOST_ENV, "https://dbc.example")
    monkeypatch.setenv(DEFAULT_DATABRICKS_TOKEN_ENV, "secret-token")
    monkeypatch.setattr(legacy_databricks_runs, "_databricks_api_json", fake_api_json)

    exit_code = legacy_databricks_runs.main(
        [
            "--output-json",
            str(output_path),
            "put-dbfs-file",
            "--local-path",
            str(local_path),
            "--dbfs-path",
            "dbfs:/FileStore/cachet/run_engine_probe.py",
        ]
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert record["artifact"]["bytes"] == len(uploaded_bytes)
    assert record["artifact"]["sha256"] == hashlib.sha256(uploaded_bytes).hexdigest()
    assert record["artifact"]["sha256"] != hashlib.sha256(changed_bytes).hexdigest()


def test_main_stage_and_submit_writes_sanitized_artifact_and_submit_record(monkeypatch, tmp_path):
    runner_path = tmp_path / "run_engine_probe.py"
    wheel_path = tmp_path / "document_kv_cache-0.2.0-py3-none-any.whl"
    payload_path = tmp_path / "payload.json"
    output_path = tmp_path / "stage-submit.json"
    runner_path.write_text("print('cachet')\n", encoding="utf-8")
    wheel_path.write_bytes(b"wheel-bytes")
    payload_path.write_text(json.dumps(_dbfs_artifact_submit_payload()), encoding="utf-8")
    raw_secret = "secret-token"

    responses = {
        "/api/2.0/dbfs/put": [{}, {}],
        "/api/2.1/jobs/runs/submit": [{"run_id": 123}],
    }

    def fake_api_json(config, method, path_and_query, *, opener, payload=None):
        assert config.normalized_host == "https://dbc.example"
        assert method == "POST"
        return responses[path_and_query].pop(0)

    monkeypatch.setenv(DEFAULT_DATABRICKS_HOST_ENV, "https://dbc.example")
    monkeypatch.setenv(DEFAULT_DATABRICKS_TOKEN_ENV, raw_secret)
    monkeypatch.setattr(legacy_databricks_runs, "_databricks_api_json", fake_api_json)

    exit_code = legacy_databricks_runs.main(
        [
            "--output-json",
            str(output_path),
            "stage-and-submit",
            "--payload-json",
            str(payload_path),
            "--artifact",
            f"{runner_path}=dbfs:/cachet/run_engine_probe.py",
            "--artifact",
            f"{wheel_path}=dbfs:/cachet/document_kv_cache-0.2.0-py3-none-any.whl",
            "--overwrite",
            "--require-payload-dbfs-artifacts",
        ]
    )

    output = output_path.read_text(encoding="utf-8")
    record = json.loads(output)
    assert exit_code == 0
    assert raw_secret not in output
    assert record["action"] == "stage-and-submit"
    assert record["response"] == {"run_id": 123}
    assert [upload["artifact"]["dbfs_path"] for upload in record["artifact_uploads"]] == [
        "dbfs:/cachet/run_engine_probe.py",
        "dbfs:/cachet/document_kv_cache-0.2.0-py3-none-any.whl",
    ]
    assert responses == {"/api/2.0/dbfs/put": [], "/api/2.1/jobs/runs/submit": []}


def test_main_stage_and_submit_forwards_preflight_auth_check(monkeypatch, tmp_path):
    runner_path = tmp_path / "run_engine_probe.py"
    wheel_path = tmp_path / "document_kv_cache-0.2.0-py3-none-any.whl"
    payload_path = tmp_path / "payload.json"
    output_path = tmp_path / "stage-submit.json"
    runner_path.write_text("print('cachet')\n", encoding="utf-8")
    wheel_path.write_bytes(b"wheel-bytes")
    payload = _dbfs_artifact_submit_payload()
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    seen = {}

    def fake_stage_and_submit(
        config,
        received_payload,
        artifacts,
        *,
        overwrite=False,
        require_payload_dbfs_artifacts=False,
        preflight_auth_check=False,
    ):
        seen["host"] = config.normalized_host
        seen["payload"] = received_payload
        seen["artifacts"] = tuple((str(local_path), dbfs_path) for local_path, dbfs_path in artifacts)
        seen["overwrite"] = overwrite
        seen["require_payload_dbfs_artifacts"] = require_payload_dbfs_artifacts
        seen["preflight_auth_check"] = preflight_auth_check
        return {
            "ok": True,
            "action": "stage-and-submit",
            "auth": {"record_type": DATABRICKS_AUTH_CHECK_RECORD_TYPE},
            "response": {"run_id": 123},
            "artifact_uploads": [],
        }

    monkeypatch.setenv(DEFAULT_DATABRICKS_HOST_ENV, "https://dbc.example")
    monkeypatch.setenv(DEFAULT_DATABRICKS_TOKEN_ENV, "secret-token")
    monkeypatch.setattr(legacy_databricks_runs, "stage_and_submit_databricks_run", fake_stage_and_submit)

    exit_code = legacy_databricks_runs.main(
        [
            "--output-json",
            str(output_path),
            "stage-and-submit",
            "--payload-json",
            str(payload_path),
            "--artifact",
            f"{runner_path}=dbfs:/cachet/run_engine_probe.py",
            "--artifact",
            f"{wheel_path}=dbfs:/cachet/document_kv_cache-0.2.0-py3-none-any.whl",
            "--overwrite",
            "--require-payload-dbfs-artifacts",
            "--preflight-auth-check",
        ]
    )

    assert exit_code == 0
    assert seen == {
        "host": "https://dbc.example",
        "payload": payload,
        "artifacts": (
            (str(runner_path), "dbfs:/cachet/run_engine_probe.py"),
            (str(wheel_path), "dbfs:/cachet/document_kv_cache-0.2.0-py3-none-any.whl"),
        ),
        "overwrite": True,
        "require_payload_dbfs_artifacts": True,
        "preflight_auth_check": True,
    }
    assert json.loads(output_path.read_text(encoding="utf-8"))["auth"] == {
        "record_type": DATABRICKS_AUTH_CHECK_RECORD_TYPE
    }


def test_main_stage_and_submit_dry_run_writes_plan_without_databricks_env(monkeypatch, tmp_path):
    runner_path = tmp_path / "run_engine_probe.py"
    wheel_path = tmp_path / "document_kv_cache-0.2.0-py3-none-any.whl"
    payload_path = tmp_path / "payload.json"
    output_path = tmp_path / "stage-submit-plan.json"
    runner_path.write_text("print('cachet')\n", encoding="utf-8")
    wheel_path.write_bytes(b"wheel-bytes")
    payload_path.write_text(json.dumps(_dbfs_artifact_submit_payload()), encoding="utf-8")
    monkeypatch.delenv(DEFAULT_DATABRICKS_HOST_ENV, raising=False)
    monkeypatch.delenv(DEFAULT_DATABRICKS_TOKEN_ENV, raising=False)

    exit_code = legacy_databricks_runs.main(
        [
            "--output-json",
            str(output_path),
            "stage-and-submit",
            "--payload-json",
            str(payload_path),
            "--artifact",
            f"{runner_path}=dbfs:/cachet/run_engine_probe.py",
            "--artifact",
            f"{wheel_path}=dbfs:/cachet/document_kv_cache-0.2.0-py3-none-any.whl",
            "--require-payload-dbfs-artifacts",
            "--dry-run",
        ]
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert record["action"] == "stage-and-submit-plan"
    assert "response" not in record
    assert record["submit_payload"]["source_path"] == str(payload_path)
    assert [upload["artifact"]["dbfs_path"] for upload in record["artifact_uploads"]] == [
        "dbfs:/cachet/run_engine_probe.py",
        "dbfs:/cachet/document_kv_cache-0.2.0-py3-none-any.whl",
    ]


def test_main_payload_summary_requires_no_credentials(monkeypatch, tmp_path):
    output_path = tmp_path / "payload-summary.json"
    payload_path = tmp_path / "payload.json"
    payload = _single_node_g5_submit_payload()
    payload["tasks"][0]["new_cluster"]["spark_env_vars"] = {
        "DOCUMENT_KV_VLLM_NATIVE_PROBE_FACTORY": "vllm_kv_injection.probe:build_native_connector_probe"
    }
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.delenv(DEFAULT_DATABRICKS_HOST_ENV, raising=False)
    monkeypatch.delenv(DEFAULT_DATABRICKS_TOKEN_ENV, raising=False)

    exit_code = legacy_databricks_runs.main(
        [
            "--output-json",
            str(output_path),
            "payload-summary",
            "--payload-json",
            str(payload_path),
            "--expected-hardware-target",
            "aws-g6-l4",
        ]
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert record["action"] == "payload-summary"
    assert "response" not in record
    assert record["summary"]["record_type"] == DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE
    assert record["summary"]["source_path"] == str(payload_path)
    assert record["summary"]["hardware_targets"] == ["aws-g6-l4"]
    assert record["summary"]["spark_env_keys"] == ["DOCUMENT_KV_VLLM_NATIVE_PROBE_FACTORY"]


def test_main_payload_summary_rejects_unexpected_hardware_target_before_auth(monkeypatch, tmp_path):
    output_path = tmp_path / "payload-summary.json"
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(_single_node_g5_submit_payload()), encoding="utf-8")
    monkeypatch.delenv(DEFAULT_DATABRICKS_HOST_ENV, raising=False)
    monkeypatch.delenv(DEFAULT_DATABRICKS_TOKEN_ENV, raising=False)

    exit_code = legacy_databricks_runs.main(
        [
            "--output-json",
            str(output_path),
            "payload-summary",
            "--payload-json",
            str(payload_path),
            "--expected-hardware-target",
            "aws-g5-a10g",
        ]
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert record["ok"] is False
    assert record["error_type"] == "ValueError"
    assert "hardware_target 'aws-g5-a10g'" in record["error"]
    assert DEFAULT_DATABRICKS_HOST_ENV not in record["error"]
    assert DEFAULT_DATABRICKS_TOKEN_ENV not in record["error"]


def test_main_payload_summary_rejects_malformed_raw_task_array_before_auth(monkeypatch, tmp_path):
    output_path = tmp_path / "payload-summary.json"
    payload_path = tmp_path / "payload.json"
    payload = _single_node_g5_submit_payload()
    payload["tasks"].append(42)
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.delenv(DEFAULT_DATABRICKS_HOST_ENV, raising=False)
    monkeypatch.delenv(DEFAULT_DATABRICKS_TOKEN_ENV, raising=False)

    exit_code = legacy_databricks_runs.main(
        [
            "--output-json",
            str(output_path),
            "payload-summary",
            "--payload-json",
            str(payload_path),
            "--expected-hardware-target",
            "aws-g6-l4",
        ]
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert record["ok"] is False
    assert record["error_type"] == "ValueError"
    assert "tasks must contain only objects" in record["error"]
    assert "invalid task indices: 1" in record["error"]
    assert DEFAULT_DATABRICKS_HOST_ENV not in record["error"]
    assert DEFAULT_DATABRICKS_TOKEN_ENV not in record["error"]


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
                    "run_id": 124,
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


def test_main_get_summary_can_validate_expected_hardware_target(monkeypatch, tmp_path):
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
            "run_name": "document-kv-v1",
            "tasks": [
                {
                    "task_key": "run-benchmark",
                    "run_id": 124,
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
            "--expected-hardware-target",
            "aws-g6-l4",
        ]
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert record["summary"]["submit_payload"]["hardware_targets"] == ["aws-g6-l4"]


def test_main_get_summary_rejects_unexpected_hardware_target(monkeypatch, tmp_path):
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
            "run_name": "document-kv-v1",
            "tasks": [
                {
                    "task_key": "run-benchmark",
                    "run_id": 124,
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
            "--expected-hardware-target",
            "aws-g5-a10g",
        ]
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert record["ok"] is False
    assert record["error_type"] == "ValueError"
    assert "hardware_target 'aws-g5-a10g'" in record["error"]
    assert "response" not in record


def test_main_get_expected_hardware_target_requires_submit_payload(tmp_path):
    output_path = tmp_path / "response.json"

    exit_code = legacy_databricks_runs.main(
        [
            "--output-json",
            str(output_path),
            "get",
            "--run-id",
            "123",
            "--summary",
            "--expected-hardware-target",
            "aws-g6-l4",
        ]
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert record["ok"] is False
    assert record["error"] == "--expected-hardware-target requires --submit-payload-json"


def test_main_get_expected_hardware_target_requires_summary(tmp_path):
    output_path = tmp_path / "response.json"
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(_single_node_g5_submit_payload()), encoding="utf-8")

    exit_code = legacy_databricks_runs.main(
        [
            "--output-json",
            str(output_path),
            "get",
            "--run-id",
            "123",
            "--submit-payload-json",
            str(payload_path),
            "--expected-hardware-target",
            "aws-g6-l4",
        ]
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert record["ok"] is False
    assert record["error"] == "--expected-hardware-target requires --summary"


def test_main_get_expected_hardware_target_rejects_include_response(monkeypatch, tmp_path):
    output_path = tmp_path / "response.json"
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(_single_node_g5_submit_payload()), encoding="utf-8")
    raw_secret = "do-not-write-raw-response"
    monkeypatch.setenv(DEFAULT_DATABRICKS_HOST_ENV, "https://dbc.example")
    monkeypatch.setenv(DEFAULT_DATABRICKS_TOKEN_ENV, "secret-token")
    monkeypatch.setattr(
        legacy_databricks_runs,
        "get_databricks_run",
        lambda config, run_id: {
            "run_id": int(run_id),
            "raw_secret": raw_secret,
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
            "--expected-hardware-target",
            "aws-g6-l4",
            "--include-response",
        ]
    )

    record_text = output_path.read_text(encoding="utf-8")
    record = json.loads(record_text)
    assert exit_code == 1
    assert record["ok"] is False
    assert record["error"] == "--expected-hardware-target cannot be combined with --include-response"
    assert raw_secret not in record_text


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


def test_legacy_databricks_runs_uses_source_dbfs_limit_when_public_constant_is_replaced_before_import():
    env = {
        **os.environ,
        "PYTHONPATH": "src",
    }
    script = """
import json
import document_kv_cache.databricks_runs as public_runs

public_runs.DATABRICKS_DBFS_PUT_MAX_CONTENT_BYTES = 7

import restaurant_kv_serving.databricks_runs as legacy_runs

print(json.dumps({
    "document_limit": public_runs.DATABRICKS_DBFS_PUT_MAX_CONTENT_BYTES,
    "legacy_limit": legacy_runs.DATABRICKS_DBFS_PUT_MAX_CONTENT_BYTES,
}))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "document_limit": 7,
        "legacy_limit": DATABRICKS_DBFS_PUT_MAX_CONTENT_BYTES,
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


def test_legacy_databricks_runs_forwards_expected_hardware_target_keyword():
    status_record = _valid_databricks_run_status_record()

    legacy_issues = legacy_databricks_runs.databricks_run_status_sidecar_issues(
        status_record,
        expected_hardware_target="aws-g5-a10g",
    )
    public_issues = public_databricks_runs.databricks_run_status_sidecar_issues(
        status_record,
        expected_hardware_target="aws-g5-a10g",
    )

    assert legacy_issues == public_issues
    assert any("hardware_target 'aws-g5-a10g'" in issue for issue in legacy_issues)
    with pytest.raises(ValueError, match=r"hardware_target 'aws-g5-a10g'"):
        legacy_databricks_runs.validate_databricks_run_status_sidecar(
            status_record,
            expected_hardware_target="aws-g5-a10g",
        )


def test_legacy_databricks_runs_private_g5_gpu_shim_uses_generic_v1_gpu_check():
    assert legacy_databricks_runs._is_aws_g5_node_type("g5.8xlarge") is True
    assert legacy_databricks_runs._is_aws_g5_node_type("g6.8xlarge") is True
    assert legacy_databricks_runs._is_aws_g5_node_type("g6e.8xlarge") is False
    assert (
        legacy_databricks_runs._is_aws_g5_node_type("g6.8xlarge")
        is public_databricks_runs._is_supported_aws_single_node_gpu_type("g6.8xlarge")
    )


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


def test_legacy_workspace_profile_config_factory_returns_picklable_legacy_config(tmp_path):
    config_path = tmp_path / ".databrickscfg"
    config_path.write_text(
        "[QA]\n"
        "host = https://dbc.example/\n"
        "token = secret-token\n",
        encoding="utf-8",
    )

    config = legacy_databricks_runs.databricks_workspace_config_from_profile(
        "QA",
        config_file=config_path,
    )

    round_tripped = pickle.loads(pickle.dumps(config))
    assert type(config) is legacy_databricks_runs.DatabricksWorkspaceConfig
    assert isinstance(config, public_databricks_runs.DatabricksWorkspaceConfig)
    assert type(round_tripped) is legacy_databricks_runs.DatabricksWorkspaceConfig
    assert round_tripped.normalized_host == "https://dbc.example"
    assert "secret-token" not in repr(round_tripped)


def test_databricks_runs_star_import_surfaces_are_stable():
    expected_legacy_exports = {
        "DEFAULT_DATABRICKS_CONFIG_FILE",
        "DEFAULT_DATABRICKS_HOST_ENV",
        "DEFAULT_DATABRICKS_TOKEN_ENV",
        "DEFAULT_DATABRICKS_TIMEOUT_SECONDS",
        "DATABRICKS_AUTH_CHECK_RECORD_TYPE",
        "DATABRICKS_PROFILE_AUTH_MODES",
        "DATABRICKS_DBFS_PUT_MAX_CONTENT_BYTES",
        "DATABRICKS_RUN_STATUS_RECORD_TYPE",
        "DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE",
        "DATABRICKS_TERMINAL_LIFE_CYCLE_STATES",
        "DatabricksHTTPResponse",
        "DatabricksURLOpener",
        "DatabricksWorkspaceConfig",
        "databricks_workspace_config_from_env",
        "databricks_workspace_config_from_profile",
        "databricks_workspace_config_from_sdk_profile",
        "check_databricks_auth",
        "submit_databricks_run",
        "get_databricks_run",
        "put_databricks_dbfs_file",
        "plan_databricks_stage_and_submit",
        "stage_and_submit_databricks_run",
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
    assert "Submit, inspect, or stage Databricks artifacts" in result.stdout


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


class _SequentialOpener:
    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.requests = []
        self.timeouts = []

    def __call__(self, request, *, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        return _FakeResponse(self._payloads.pop(0))


class _HTTPErrorOpener:
    def __init__(self, error):
        self._error = error

    def __call__(self, request, *, timeout):
        raise self._error


class _RecordingHTTPErrorOpener:
    def __init__(self, error):
        self._error = error
        self.requests = []
        self.timeouts = []

    def __call__(self, request, *, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
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
                    "node_type_id": "g6.4xlarge",
                    "driver_node_type_id": "g6.4xlarge",
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


def _dbfs_artifact_submit_payload():
    return {
        "run_name": "document-kv-engine-probe",
        "tasks": [
            {
                "task_key": "document_kv_engine_probe",
                "spark_python_task": {
                    "python_file": "dbfs:/cachet/run_engine_probe.py",
                    "parameters": [
                        "--package-wheel-uri",
                        "dbfs:/cachet/document_kv_cache-0.2.0-py3-none-any.whl",
                    ],
                },
            }
        ],
    }
