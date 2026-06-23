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
    wheel_path = tmp_path / "cachet_kv-0.2.0-py3-none-any.whl"
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
            (wheel_path, "/cachet/cachet_kv-0.2.0-py3-none-any.whl"),
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
        "/cachet/cachet_kv-0.2.0-py3-none-any.whl"
    )
    assert json.loads(opener.requests[2].data.decode("utf-8")) == payload
    assert opener.timeouts == [9, 9, 9]
    assert [upload["artifact"]["dbfs_path"] for upload in record["artifact_uploads"]] == [
        "dbfs:/cachet/run_engine_probe.py",
        "dbfs:/cachet/cachet_kv-0.2.0-py3-none-any.whl",
    ]


def test_stage_and_submit_databricks_run_can_preflight_auth_before_uploads(tmp_path):
    runner_path = tmp_path / "run_engine_probe.py"
    wheel_path = tmp_path / "cachet_kv-0.2.0-py3-none-any.whl"
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
            (wheel_path, "dbfs:/cachet/cachet_kv-0.2.0-py3-none-any.whl"),
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
    wheel_path = tmp_path / "cachet_kv-0.2.0-py3-none-any.whl"
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
                (wheel_path, "dbfs:/cachet/cachet_kv-0.2.0-py3-none-any.whl"),
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


def test_plan_stage_and_submit_can_require_only_staged_payload_artifacts(tmp_path):
    runner_path = tmp_path / "run_engine_probe.py"
    wheel_path = tmp_path / "cachet_kv-0.2.0-py3-none-any.whl"
    launch_config_path = tmp_path / "sglang-launch-config.json"
    runner_path.write_text("print('cachet')\n", encoding="utf-8")
    wheel_path.write_bytes(b"wheel-bytes")
    launch_config_path.write_text('{"hicache_storage_backend":"dynamic"}\n', encoding="utf-8")

    record = plan_databricks_stage_and_submit(
        _generated_native_probe_submit_payload(),
        (
            (runner_path, "dbfs:/benchmarks/cachet/run_engine_probe.py"),
            (wheel_path, "dbfs:/benchmarks/cachet/cachet_kv-0.2.0-py3-none-any.whl"),
            (launch_config_path, "dbfs:/benchmarks/cachet/sglang-launch-config.json"),
        ),
        require_payload_staged_dbfs_artifacts=True,
    )

    assert record["ok"] is True
    assert [upload["artifact"]["dbfs_path"] for upload in record["artifact_uploads"]] == [
        "dbfs:/benchmarks/cachet/run_engine_probe.py",
        "dbfs:/benchmarks/cachet/cachet_kv-0.2.0-py3-none-any.whl",
        "dbfs:/benchmarks/cachet/sglang-launch-config.json",
    ]


def test_strict_payload_dbfs_artifact_check_still_rejects_generated_probe_outputs(tmp_path):
    runner_path = tmp_path / "run_engine_probe.py"
    wheel_path = tmp_path / "cachet_kv-0.2.0-py3-none-any.whl"
    launch_config_path = tmp_path / "sglang-launch-config.json"
    runner_path.write_text("print('cachet')\n", encoding="utf-8")
    wheel_path.write_bytes(b"wheel-bytes")
    launch_config_path.write_text('{"hicache_storage_backend":"dynamic"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="sglang-fixture"):
        plan_databricks_stage_and_submit(
            _generated_native_probe_submit_payload(),
            (
                (runner_path, "dbfs:/benchmarks/cachet/run_engine_probe.py"),
                (wheel_path, "dbfs:/benchmarks/cachet/cachet_kv-0.2.0-py3-none-any.whl"),
                (launch_config_path, "dbfs:/benchmarks/cachet/sglang-launch-config.json"),
            ),
            require_payload_dbfs_artifacts=True,
        )


def test_stage_and_submit_rejects_missing_staged_payload_artifact_before_network(tmp_path):
    runner_path = tmp_path / "run_engine_probe.py"
    wheel_path = tmp_path / "cachet_kv-0.2.0-py3-none-any.whl"
    runner_path.write_text("print('cachet')\n", encoding="utf-8")
    wheel_path.write_bytes(b"wheel-bytes")
    opener = _FakeOpener({})
    config = DatabricksWorkspaceConfig("https://dbc.example/", "secret-token")

    with pytest.raises(ValueError, match="sglang-launch-config.json"):
        stage_and_submit_databricks_run(
            config,
            _generated_native_probe_submit_payload(),
            (
                (runner_path, "dbfs:/benchmarks/cachet/run_engine_probe.py"),
                (wheel_path, "dbfs:/benchmarks/cachet/cachet_kv-0.2.0-py3-none-any.whl"),
            ),
            require_payload_staged_dbfs_artifacts=True,
            opener=opener,
        )

    assert opener.requests == []


def test_stage_and_submit_requires_non_fixture_engine_probe_inputs(tmp_path):
    runner_path = tmp_path / "run_engine_probe.py"
    wheel_path = tmp_path / "cachet_kv-0.2.0-py3-none-any.whl"
    handoff_path = tmp_path / "request.handoff.json"
    payload_path = tmp_path / "request.payload.kv"
    layer_names_path = tmp_path / "vllm-layer-names.json"
    runner_path.write_text("print('cachet')\n", encoding="utf-8")
    wheel_path.write_bytes(b"wheel-bytes")
    handoff_path.write_text('{"record_type":"handoff"}\n', encoding="utf-8")
    payload_path.write_bytes(b"payload")
    layer_names_path.write_text('["layer.0"]\n', encoding="utf-8")

    record = plan_databricks_stage_and_submit(
        _non_fixture_engine_probe_submit_payload(),
        (
            (runner_path, "dbfs:/benchmarks/cachet/run_engine_probe.py"),
            (wheel_path, "dbfs:/benchmarks/cachet/cachet_kv-0.2.0-py3-none-any.whl"),
            (handoff_path, "dbfs:/benchmarks/cachet/request.handoff.json"),
            (payload_path, "dbfs:/benchmarks/cachet/request.payload.kv"),
            (layer_names_path, "dbfs:/benchmarks/cachet/vllm-layer-names.json"),
        ),
        require_payload_staged_dbfs_artifacts=True,
    )

    assert record["ok"] is True
    assert [upload["artifact"]["dbfs_path"] for upload in record["artifact_uploads"]] == [
        "dbfs:/benchmarks/cachet/run_engine_probe.py",
        "dbfs:/benchmarks/cachet/cachet_kv-0.2.0-py3-none-any.whl",
        "dbfs:/benchmarks/cachet/request.handoff.json",
        "dbfs:/benchmarks/cachet/request.payload.kv",
        "dbfs:/benchmarks/cachet/vllm-layer-names.json",
    ]


def test_stage_and_submit_rejects_missing_non_fixture_engine_probe_inputs(tmp_path):
    runner_path = tmp_path / "run_engine_probe.py"
    wheel_path = tmp_path / "cachet_kv-0.2.0-py3-none-any.whl"
    runner_path.write_text("print('cachet')\n", encoding="utf-8")
    wheel_path.write_bytes(b"wheel-bytes")

    with pytest.raises(ValueError, match="request.handoff.json"):
        plan_databricks_stage_and_submit(
            _non_fixture_engine_probe_submit_payload(),
            (
                (runner_path, "dbfs:/benchmarks/cachet/run_engine_probe.py"),
                (wheel_path, "dbfs:/benchmarks/cachet/cachet_kv-0.2.0-py3-none-any.whl"),
            ),
            require_payload_staged_dbfs_artifacts=True,
        )


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
                (missing_wheel_path, "dbfs:/cachet/cachet_kv-0.2.0-py3-none-any.whl"),
            ),
            require_payload_dbfs_artifacts=True,
            opener=opener,
        )

    assert opener.requests == []


def test_plan_databricks_stage_and_submit_validates_artifacts_without_network(tmp_path):
    runner_path = tmp_path / "run_engine_probe.py"
    wheel_path = tmp_path / "cachet_kv-0.2.0-py3-none-any.whl"
    runner_path.write_text("print('cachet')\n", encoding="utf-8")
    wheel_path.write_bytes(b"wheel-bytes")

    record = plan_databricks_stage_and_submit(
        _dbfs_artifact_submit_payload(),
        (
            (runner_path, "dbfs:/cachet/run_engine_probe.py"),
            (wheel_path, "dbfs:/cachet/cachet_kv-0.2.0-py3-none-any.whl"),
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
        "dbfs:/cachet/cachet_kv-0.2.0-py3-none-any.whl",
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
                    "new_cluster": {
                        "node_type_id": "g6.8xlarge",
                        "driver_node_type_id": "g6.8xlarge",
                    },
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
    assert summary["tasks"][0]["node_type_id"] == "g6.8xlarge"
    assert summary["tasks"][0]["driver_node_type_id"] == "g6.8xlarge"
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


def test_databricks_run_status_sidecar_validation_can_require_exact_node_type():
    status_record = _valid_databricks_run_status_record()

    assert databricks_run_status_sidecar_issues(
        status_record,
        expected_hardware_target="aws-g6-l4",
        expected_node_type_id="g6.8xlarge",
    ) == (
        "Databricks run status sidecar tasks[0].node_type_id must be present for "
        "node_type_id 'g6.8xlarge' validation",
        "Databricks run status sidecar tasks[0].driver_node_type_id must be present for "
        "node_type_id 'g6.8xlarge' validation",
        "Databricks run status sidecar submit_payload.tasks[0].node_type_id must be "
        "node_type_id 'g6.8xlarge'",
        "Databricks run status sidecar submit_payload.tasks[0].driver_node_type_id must be "
        "node_type_id 'g6.8xlarge'",
    )

    payload = _single_node_g5_submit_payload()
    cluster = payload["tasks"][0]["new_cluster"]
    cluster["node_type_id"] = "g6.8xlarge"
    cluster["driver_node_type_id"] = "g6.8xlarge"
    exact_record = summarize_databricks_run(
        {
            "run_id": 123,
            "run_name": "document-kv-v1",
            "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
            "tasks": [
                {
                    "task_key": "run-benchmark",
                    "run_id": 124,
                    "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
                    "new_cluster": {
                        "node_type_id": "g6.8xlarge",
                        "driver_node_type_id": "g6.8xlarge",
                    },
                }
            ],
        },
        submit_payload=payload,
        submit_payload_path="/Volumes/catalog/schema/volume/databricks-run-submit.json",
    )

    validate_databricks_run_status_sidecar(
        exact_record,
        expected_hardware_target="aws-g6-l4",
        expected_node_type_id="g6.8xlarge",
    )


def test_databricks_run_status_sidecar_validation_rejects_live_node_type_mismatch():
    payload = _single_node_g5_submit_payload()
    payload_cluster = payload["tasks"][0]["new_cluster"]
    payload_cluster["node_type_id"] = "g6.8xlarge"
    payload_cluster["driver_node_type_id"] = "g6.8xlarge"
    status_record = summarize_databricks_run(
        {
            "run_id": 123,
            "run_name": "document-kv-v1",
            "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
            "tasks": [
                {
                    "task_key": "run-benchmark",
                    "run_id": 124,
                    "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
                    "new_cluster": {
                        "node_type_id": "g6e.8xlarge",
                        "driver_node_type_id": "g6e.8xlarge",
                    },
                }
            ],
        },
        submit_payload=payload,
        submit_payload_path="/Volumes/catalog/schema/volume/databricks-run-submit.json",
    )

    issues = databricks_run_status_sidecar_issues(
        status_record,
        expected_hardware_target="aws-g6-l4",
        expected_node_type_id="g6.8xlarge",
    )

    assert (
        "Databricks run status sidecar tasks[0].node_type_id must be a supported V1 AWS GPU node type"
        in issues
    )
    assert (
        "Databricks run status sidecar tasks[0].driver_node_type_id must be a supported V1 AWS GPU node type"
        in issues
    )


def test_databricks_run_status_sidecar_validation_requires_live_node_evidence_for_exact_node_type():
    payload = _single_node_g5_submit_payload()
    payload_cluster = payload["tasks"][0]["new_cluster"]
    payload_cluster["node_type_id"] = "g6.8xlarge"
    payload_cluster["driver_node_type_id"] = "g6.8xlarge"
    status_record = summarize_databricks_run(
        {
            "run_id": 123,
            "run_name": "document-kv-v1",
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
        submit_payload_path="/Volumes/catalog/schema/volume/databricks-run-submit.json",
    )

    issues = databricks_run_status_sidecar_issues(
        status_record,
        expected_hardware_target="aws-g6-l4",
        expected_node_type_id="g6.8xlarge",
    )

    assert (
        "Databricks run status sidecar tasks[0].node_type_id must be present for "
        "node_type_id 'g6.8xlarge' validation"
        in issues
    )
    assert (
        "Databricks run status sidecar tasks[0].driver_node_type_id must be present for "
        "node_type_id 'g6.8xlarge' validation"
        in issues
    )


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
                        "dbfs:/cachet/cachet_kv-0.2.0-py3-none-any.whl",
                    ],
                },
            }
        ],
    }


def _generated_native_probe_submit_payload():
    return {
        "run_name": "document-kv-engine-probe",
        "tasks": [
            {
                "task_key": "document_kv_engine_probe",
                "spark_python_task": {
                    "python_file": "dbfs:/benchmarks/cachet/run_engine_probe.py",
                    "parameters": [
                        "--fixture-output-dir",
                        "dbfs:/benchmarks/cachet/probes/sglang-fixture",
                        "--fixture-backend",
                        "sglang",
                        "--sglang-runtime-preflight-output-json",
                        "dbfs:/benchmarks/cachet/probes/sglang-fixture/sglang-runtime-preflight.json",
                        "--sglang-runtime-preflight-launch-config-json",
                        "dbfs:/benchmarks/cachet/sglang-launch-config.json",
                        "--handoff-json",
                        "dbfs:/benchmarks/cachet/probes/sglang-fixture/qwen3-v1-fixture.handoff.json",
                        "--probe-factory",
                        "document_kv_cache.native_probe_factories:sglang_native_probe_factory",
                        "--output-json",
                        "dbfs:/benchmarks/cachet/probes/sglang-engine-probe.json",
                        "--expected-backend",
                        "sglang",
                        "--package-wheel-uri",
                        "dbfs:/benchmarks/cachet/cachet_kv-0.2.0-py3-none-any.whl",
                    ],
                },
            }
        ],
    }


def _non_fixture_engine_probe_submit_payload():
    return {
        "run_name": "document-kv-engine-probe",
        "tasks": [
            {
                "task_key": "document_kv_engine_probe",
                "spark_python_task": {
                    "python_file": "dbfs:/benchmarks/cachet/run_engine_probe.py",
                    "parameters": [
                        "--vllm-runtime-preflight-output-json",
                        "dbfs:/benchmarks/cachet/vllm-runtime-preflight.json",
                        "--vllm-runtime-preflight-layer-names-json",
                        "dbfs:/benchmarks/cachet/vllm-layer-names.json",
                        "--handoff-json",
                        "dbfs:/benchmarks/cachet/request.handoff.json",
                        "--probe-factory",
                        "document_kv_cache.native_probe_factories:vllm_native_probe_factory",
                        "--output-json",
                        "dbfs:/benchmarks/cachet/vllm-engine-probe.json",
                        "--expected-backend",
                        "vllm",
                        "--payload-uri",
                        "dbfs:/benchmarks/cachet/request.payload.kv",
                        "--package-wheel-uri",
                        "dbfs:/benchmarks/cachet/cachet_kv-0.2.0-py3-none-any.whl",
                    ],
                },
            }
        ],
    }
