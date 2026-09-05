import hashlib
import io
import json
import os
import sys
import traceback
import types
import urllib.error
import urllib.response
from email.message import Message
from concurrent.futures import ThreadPoolExecutor

import pytest

import document_kv_cache.databricks_runs as public_databricks_runs
from document_kv_cache._hardware_targets import (
    HARDWARE_TARGET_AWS_SINGLE_NODE_GPU_PREFIXES,
    SUPPORTED_AWS_SINGLE_NODE_GPU_PREFIXES,
)
from document_kv_cache.databricks_runs import (
    DEFAULT_DATABRICKS_HOST_ENV,
    DEFAULT_DATABRICKS_TOKEN_ENV,
    DATABRICKS_ACTIVE_RUNS_MAX_ENTRIES,
    DATABRICKS_ACTIVE_RUNS_MAX_PAGES,
    DATABRICKS_API_PAGE_MAX_BYTES,
    DATABRICKS_API_PAGE_TOKEN_MAX_BYTES,
    DATABRICKS_NODE_TYPES_MAX_ENTRIES,
    DATABRICKS_PROFILE_AUTH_MODES,
    DATABRICKS_AUTH_CHECK_RECORD_TYPE,
    DATABRICKS_DBFS_PUT_MAX_CONTENT_BYTES,
    DATABRICKS_VOLUME_DIRECTORY_MAX_ENTRIES,
    DATABRICKS_VOLUME_DIRECTORY_MAX_PAGES,
    DATABRICKS_VOLUME_DIRECTORY_MAX_PAGE_TOKEN_BYTES,
    DATABRICKS_VOLUME_FILE_MAX_DOWNLOAD_BYTES,
    DATABRICKS_VOLUME_FILE_MAX_STREAM_BYTES,
    DATABRICKS_VOLUME_FILE_MAX_UPLOAD_BYTES,
    DATABRICKS_RUN_STATUS_RECORD_TYPE,
    DATABRICKS_RUN_SUBMIT_PAYLOAD_RECORD_TYPE,
    DatabricksWorkspaceConfig,
    bind_databricks_run_idempotency_token,
    check_databricks_auth,
    create_databricks_volume_directory_idempotent,
    databricks_run_status_record,
    databricks_run_status_sidecar_issues,
    databricks_workspace_config_from_env,
    databricks_workspace_config_from_profile,
    databricks_workspace_config_from_sdk_profile,
    download_databricks_volume_file_bytes,
    get_databricks_volume_file_metadata,
    get_databricks_run,
    get_databricks_run_output,
    list_active_databricks_runs,
    list_databricks_node_types,
    list_databricks_volume_directory,
    plan_databricks_stage_and_submit,
    put_databricks_dbfs_file,
    read_databricks_run_submit_payload,
    require_databricks_current_user_name,
    recover_pre_reserved_databricks_run,
    reserve_and_submit_databricks_run,
    reserve_and_submit_databricks_run_json,
    stage_and_submit_databricks_run,
    submit_databricks_run,
    submit_pre_reserved_databricks_run,
    stream_databricks_volume_file_sha256,
    summarize_databricks_run,
    summarize_databricks_run_submit_payload,
    validate_databricks_run_status_sidecar,
    upload_databricks_volume_file_bytes_exclusive,
    upload_databricks_volume_file_path_exclusive,
    write_databricks_run_response_json,
)
from document_kv_cache.databricks_resource_ledger import (
    DatabricksRunAttemptReservationRequest,
    create_databricks_cluster_hour_ledger_json,
    databricks_ledger_prefix,
    read_databricks_cluster_hour_ledger_json,
    reserve_databricks_run_attempt_batch_authorized_json,
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


def test_databricks_no_redirect_handler_blocks_every_redirect_method_and_status():
    for status_code in (301, 302, 303, 307, 308):
        for method in ("GET", "HEAD", "POST", "PUT"):
            transport = _RedirectingHTTPHandler(status_code)
            opener = urllib.request.build_opener(
                public_databricks_runs._DatabricksNoRedirectHandler(),
                transport,
            )
            request = urllib.request.Request(
                "http://workspace.example/api/2.0/test",
                data=b"{}" if method in {"POST", "PUT"} else None,
                method=method,
                headers={"Authorization": "Bearer secret-token"},
            )

            with pytest.raises(urllib.error.HTTPError) as excinfo:
                opener.open(request, timeout=1)

            assert excinfo.value.code == status_code
            assert len(transport.requests) == 1
            assert transport.requests[0].full_url.startswith(
                "http://workspace.example/"
            )
            assert transport.requests[0].headers["Authorization"] == (
                "Bearer secret-token"
            )


def test_all_authenticated_databricks_defaults_reject_redirects_without_following(
    tmp_path,
    monkeypatch,
):
    real_build_opener = urllib.request.build_opener
    active_transport = None

    def controlled_build_opener(*handlers):
        assert active_transport is not None
        return real_build_opener(*handlers, active_transport)

    monkeypatch.setattr(
        public_databricks_runs.urllib.request,
        "build_opener",
        controlled_build_opener,
    )
    config = DatabricksWorkspaceConfig("http://workspace.example", "secret-token")
    source = tmp_path / "package.whl"
    source.write_bytes(b"package")
    source_sha256 = hashlib.sha256(b"package").hexdigest()
    volume_file = "dbfs:/Volumes/c/s/v/package.whl"
    volume_directory = "dbfs:/Volumes/c/s/v/runtime"
    cases = (
        ("GET", lambda: check_databricks_auth(config)),
        (
            "GET",
            lambda: require_databricks_current_user_name(
                config,
                expected_user_name="person@example.com",
            ),
        ),
        ("POST", lambda: submit_databricks_run(config, {"run_name": "run"})),
        ("GET", lambda: get_databricks_run(config, 1)),
        ("GET", lambda: get_databricks_run_output(config, 1)),
        ("GET", lambda: list_active_databricks_runs(config)),
        ("GET", lambda: list_databricks_node_types(config)),
        (
            "GET",
            lambda: download_databricks_volume_file_bytes(config, volume_file),
        ),
        (
            "GET",
            lambda: stream_databricks_volume_file_sha256(config, volume_file),
        ),
        (
            "HEAD",
            lambda: get_databricks_volume_file_metadata(config, volume_file),
        ),
        (
            "GET",
            lambda: list_databricks_volume_directory(config, volume_directory),
        ),
        (
            "PUT",
            lambda: upload_databricks_volume_file_bytes_exclusive(
                config,
                volume_file,
                b"package",
            ),
        ),
        (
            "PUT",
            lambda: upload_databricks_volume_file_path_exclusive(
                config,
                volume_file,
                source,
                expected_sha256=source_sha256,
                expected_size=7,
            ),
        ),
        (
            "PUT",
            lambda: create_databricks_volume_directory_idempotent(
                config,
                volume_directory,
            ),
        ),
        (
            "POST",
            lambda: put_databricks_dbfs_file(
                config,
                source,
                "dbfs:/FileStore/package.whl",
            ),
        ),
        (
            "HEAD",
            lambda: public_databricks_runs._prove_databricks_volume_directory_exists(
                config,
                "/Volumes/c/s/v/runtime",
                opener=None,
            ),
        ),
    )

    for method, action in cases:
        active_transport = _RedirectingHTTPHandler(302)
        with pytest.raises(RuntimeError) as excinfo:
            action()
        formatted = "".join(
            traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb)
        )
        assert "secret-token" not in formatted
        assert len(active_transport.requests) == 1
        assert active_transport.requests[0].get_method() == method
        assert active_transport.requests[0].full_url.startswith(
            "http://workspace.example/"
        )


def test_databricks_json_response_is_status_and_content_length_closed():
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")
    payload = b'{"id":"123"}'
    exact = _StreamingBinaryOpener(
        payload,
        status=200,
        headers={"content-length": str(len(payload))},
    )
    assert check_databricks_auth(config, opener=exact)["http_status"] == 200
    assert exact.response.read_limits == [len(payload), 1]

    chunked = _StreamingBinaryOpener(
        payload,
        status=200,
        headers={"transfer-encoding": "chunked"},
    )
    assert check_databricks_auth(config, opener=chunked)["http_status"] == 200

    for status in (201, 204, 301, 302, 303, 307, 308, 200.0, True, None):
        wrong_status = _BinaryOpener(b"not-json", status=status)
        with pytest.raises(RuntimeError, match="unexpected HTTP status"):
            check_databricks_auth(config, opener=wrong_status)
        assert wrong_status.response.read_limits == []

    cases = (
        (
            _StreamingBinaryOpener(
                payload,
                headers={"content-length": str(len(payload) + 1)},
            ),
            "ended before content-length",
        ),
        (
            _StreamingBinaryOpener(
                payload,
                headers={"content-length": str(len(payload) - 1)},
            ),
            "bytes beyond content-length",
        ),
        (
            _BinaryOpener(
                payload,
                headers={
                    "content-length": str(len(payload)),
                    "transfer-encoding": "chunked",
                },
            ),
            "cannot combine content-length",
        ),
        (
            _BinaryOpener(
                b"",
                headers={
                    "content-length": str(DATABRICKS_API_PAGE_MAX_BYTES + 1)
                },
            ),
            "content-length exceeds.*byte cap",
        ),
        (
            _BinaryOpener(
                payload,
                headers={"content-length": "1"},
                oversized_reads=True,
            ),
            "chunk byte cap",
        ),
    )
    for malformed, error_match in cases:
        with pytest.raises(RuntimeError, match=error_match):
            check_databricks_auth(config, opener=malformed)


def test_current_user_binding_requires_exact_active_single_user_without_pii():
    opener = _FakeOpener(
        {
            "active": True,
            "id": "123",
            "userName": "person@example.com",
        }
    )
    config = DatabricksWorkspaceConfig("https://dbc.example/", "secret-token")

    record = require_databricks_current_user_name(
        config,
        expected_user_name="person@example.com",
        opener=opener,
    )

    assert record == {
        "record_type": "document_kv.databricks_current_user_binding.v1",
        "authenticated": True,
        "endpoint": "/api/2.0/preview/scim/v2/Me",
        "http_status": 200,
        "user_name_sha256": hashlib.sha256(b"person@example.com").hexdigest(),
        "workspace_host_sha256": hashlib.sha256(b"https://dbc.example").hexdigest(),
    }
    serialized = json.dumps(record, sort_keys=True)
    assert "person@example.com" not in serialized
    assert "secret-token" not in serialized


@pytest.mark.parametrize(
    ("response", "expected_user_name", "error"),
    [
        ({"active": True}, "person@example.com", "lacks a normalized userName"),
        (
            {"active": False, "userName": "person@example.com"},
            "person@example.com",
            "not explicitly active",
        ),
        (
            {"userName": "person@example.com"},
            "person@example.com",
            "not explicitly active",
        ),
        (
            {"active": None, "userName": "person@example.com"},
            "person@example.com",
            "not explicitly active",
        ),
        (
            {"active": 0, "userName": "person@example.com"},
            "person@example.com",
            "not explicitly active",
        ),
        (
            {"active": 1, "userName": "person@example.com"},
            "person@example.com",
            "not explicitly active",
        ),
        (
            {"active": "true", "userName": "person@example.com"},
            "person@example.com",
            "not explicitly active",
        ),
        (
            {"active": True, "userName": "other@example.com"},
            "person@example.com",
            "differs from single_user_name",
        ),
    ],
)
def test_current_user_binding_rejects_missing_inactive_or_drifted_identity(
    response,
    expected_user_name,
    error,
):
    with pytest.raises(ValueError, match=error):
        require_databricks_current_user_name(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            expected_user_name=expected_user_name,
            opener=_FakeOpener(response),
        )


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


def test_reserved_submit_posts_exact_digest_and_resists_payload_file_mutation(
    tmp_path,
):
    payload_path = tmp_path / "submit.json"
    ledger_path = tmp_path / "cluster-hours.json"
    original_payload = _bounded_submit_payload(run_name="original-run")
    payload_path.write_text(json.dumps(original_payload), encoding="utf-8")
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="representative-canary-2026",
    )
    opener = _FakeOpener({"run_id": 123})
    config = DatabricksWorkspaceConfig(
        "https://dbc.example/",
        "secret-token",
        timeout_seconds=9,
    )

    def mutate_file_after_snapshot(_reservation, snapshot):
        assert snapshot["run_name"] == "original-run"
        payload_path.write_text(
            json.dumps(_bounded_submit_payload(run_name="mutated-run")),
            encoding="utf-8",
        )

    response = reserve_and_submit_databricks_run_json(
        config,
        payload_path,
        ledger_path=ledger_path,
        attempt_id="attempt-001",
        workload_id="vllm-8k-baseline",
        reservation_validator=mutate_file_after_snapshot,
        opener=opener,
    )

    assert response == {"run_id": 123}
    assert len(opener.requests) == 1
    request = opener.requests[0]
    assert json.loads(request.data.decode("utf-8")) == original_payload
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    reservation = ledger.reservations[0]
    assert reservation.submit_payload_sha256 == hashlib.sha256(request.data).hexdigest()
    assert len(ledger.submission_receipts) == 1
    assert ledger.submission_receipts[0].attempt_id == "attempt-001"
    assert ledger.submission_receipts[0].run_id == "123"
    assert ledger.submission_receipts[0].submit_payload_sha256 == (
        reservation.submit_payload_sha256
    )
    assert json.loads(payload_path.read_text(encoding="utf-8"))["run_name"] == (
        "mutated-run"
    )


def test_reserved_submit_rejects_over_cap_without_calling_opener(tmp_path):
    ledger_path = tmp_path / "cluster-hours.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="representative-canary-2026",
        cap_cluster_hours=4.0,
    )
    config = DatabricksWorkspaceConfig("https://dbc.example/", "secret-token")
    first_opener = _FakeOpener({"run_id": 123})
    reserve_and_submit_databricks_run(
        config,
        _bounded_submit_payload(),
        ledger_path=ledger_path,
        attempt_id="attempt-001",
        workload_id="vllm-8k-baseline",
        opener=first_opener,
    )
    blocked_opener = _FakeOpener({"run_id": 124})

    with pytest.raises(ValueError, match="would exceed.*cluster-hour cap"):
        reserve_and_submit_databricks_run(
            config,
            _bounded_submit_payload(),
            ledger_path=ledger_path,
            attempt_id="attempt-002",
            workload_id="vllm-8k-baseline",
            opener=blocked_opener,
        )

    assert blocked_opener.requests == []
    assert len(read_databricks_cluster_hour_ledger_json(ledger_path).reservations) == 1


def test_failed_reserved_submit_conservatively_keeps_reservation(tmp_path):
    ledger_path = tmp_path / "cluster-hours.json"
    create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="representative-canary-2026",
    )
    config = DatabricksWorkspaceConfig("https://dbc.example/", "secret-token")
    error = urllib.error.HTTPError(
        "https://dbc.example/api/2.1/jobs/runs/submit",
        503,
        "unavailable",
        {},
        _BytesFile(b'{"message":"temporarily unavailable"}'),
    )
    opener = _RecordingHTTPErrorOpener(error)

    with pytest.raises(RuntimeError, match="HTTP 503"):
        reserve_and_submit_databricks_run(
            config,
            _bounded_submit_payload(),
            ledger_path=ledger_path,
            attempt_id="attempt-failed",
            workload_id="vllm-8k-baseline",
            opener=opener,
        )

    assert len(opener.requests) == 1
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    assert [item.attempt_id for item in ledger.reservations] == ["attempt-failed"]
    assert ledger.active_reserved_cluster_hours == 4.0
    assert ledger.submission_receipts == ()
    assert ledger.terminal_actuals == ()


def test_reserve_and_submit_cli_uses_coupled_json_entrypoint(
    tmp_path,
    monkeypatch,
    capsys,
):
    payload_path = tmp_path / "submit.json"
    ledger_path = tmp_path / "cluster-hours.json"
    payload_path.write_text(json.dumps(_bounded_submit_payload()), encoding="utf-8")
    captured = {}

    def fake_reserved_submit(config, path, **kwargs):
        captured.update(
            {
                "host": config.normalized_host,
                "payload_path": path,
                **kwargs,
            }
        )
        return {"run_id": 321}

    monkeypatch.setattr(
        public_databricks_runs,
        "reserve_and_submit_databricks_run_json",
        fake_reserved_submit,
    )
    monkeypatch.setenv(DEFAULT_DATABRICKS_HOST_ENV, "https://dbc.example/")
    monkeypatch.setenv(DEFAULT_DATABRICKS_TOKEN_ENV, "secret-token")

    exit_code = public_databricks_runs.main(
        [
            "reserve-and-submit",
            "--payload-json",
            str(payload_path),
            "--ledger-json",
            str(ledger_path),
            "--attempt-id",
            "attempt-001",
            "--workload-id",
            "g6-vllm-8k-64-baseline",
            "--representative-canary",
        ]
    )

    assert exit_code == 0
    reservation_validator = captured.pop("reservation_validator")
    assert callable(reservation_validator)
    assert (
        reservation_validator.__name__ == "validate_representative_canary_reservation"
    )
    assert captured == {
        "host": "https://dbc.example",
        "payload_path": str(payload_path),
        "ledger_path": str(ledger_path),
        "attempt_id": "attempt-001",
        "workload_id": "g6-vllm-8k-64-baseline",
    }
    assert json.loads(capsys.readouterr().out)["response"] == {"run_id": 321}


@pytest.mark.parametrize(
    ("workload_id", "extra_args", "error_match"),
    [
        (
            "g6-vllm-8k-64-baseline",
            (),
            "representative canary workload_id requires --representative-canary",
        ),
        (
            "unknown-canary-workload",
            ("--representative-canary",),
            "--representative-canary requires a workload_id from the exact ",
        ),
    ],
)
def test_reserve_and_submit_cli_fails_closed_for_representative_flag_mismatch(
    tmp_path,
    monkeypatch,
    capsys,
    workload_id,
    extra_args,
    error_match,
):
    payload_path = tmp_path / "submit.json"
    ledger_path = tmp_path / "cluster-hours.json"
    payload_path.write_text(json.dumps(_bounded_submit_payload()), encoding="utf-8")
    calls = []

    def fake_reserved_submit(*args, **kwargs):
        calls.append((args, kwargs))
        return {"run_id": 321}

    monkeypatch.setattr(
        public_databricks_runs,
        "reserve_and_submit_databricks_run_json",
        fake_reserved_submit,
    )
    monkeypatch.setenv(DEFAULT_DATABRICKS_HOST_ENV, "https://dbc.example/")
    monkeypatch.setenv(DEFAULT_DATABRICKS_TOKEN_ENV, "secret-token")

    exit_code = public_databricks_runs.main(
        [
            "reserve-and-submit",
            "--payload-json",
            str(payload_path),
            "--ledger-json",
            str(ledger_path),
            "--attempt-id",
            "attempt-001",
            "--workload-id",
            workload_id,
            *extra_args,
        ]
    )

    assert exit_code == 1
    assert calls == []
    assert error_match in json.loads(capsys.readouterr().out)["error"]


def test_get_databricks_run_fetches_run_by_id():
    opener = _FakeOpener({"run_id": 123, "state": {"life_cycle_state": "TERMINATED"}})
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")

    response = get_databricks_run(config, 123, opener=opener)

    assert response["state"]["life_cycle_state"] == "TERMINATED"
    request = opener.requests[0]
    assert request.full_url == "https://dbc.example/api/2.1/jobs/runs/get?run_id=123"
    assert request.get_method() == "GET"
    assert request.data is None


def test_get_databricks_run_output_fetches_child_run_by_id():
    opener = _BinaryOpener(
        json.dumps(
            {
            "error": "NameError: name '__file__' is not defined",
            "metadata": {"run_id": 456},
            }
        ).encode()
    )
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")

    response = get_databricks_run_output(config, "456", opener=opener)

    assert response["metadata"]["run_id"] == 456
    request = opener.requests[0]
    assert request.full_url == (
        "https://dbc.example/api/2.1/jobs/runs/get-output?run_id=456"
    )
    assert request.get_method() == "GET"
    assert request.data is None
    assert opener.response.read_limits == [
        public_databricks_runs._DATABRICKS_VOLUME_FILE_STREAM_CHUNK_BYTES,
        public_databricks_runs._DATABRICKS_VOLUME_FILE_STREAM_CHUNK_BYTES,
    ]


def test_download_databricks_volume_file_bytes_uses_authenticated_files_api():
    expected = b'{"ok":true}\n'
    opener = _BinaryOpener(expected)
    config = DatabricksWorkspaceConfig(
        "https://dbc.example/", "secret-token", timeout_seconds=9
    )

    content = download_databricks_volume_file_bytes(
        config,
        "dbfs:/Volumes/catalog/schema/volume/results/result file.json",
        max_bytes=len(expected),
        opener=opener,
    )

    assert content == expected
    request = opener.requests[0]
    assert request.full_url == (
        "https://dbc.example/api/2.0/fs/files/Volumes/catalog/schema/volume/"
        "results/result%20file.json"
    )
    assert request.get_method() == "GET"
    assert request.data is None
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert request.headers["Accept"] == "application/octet-stream"
    assert opener.timeouts == [9]
    assert opener.response.read_limits == [len(expected) + 1, 1]


@pytest.mark.parametrize(
    "uri",
    (
        "",
        "/Volumes/catalog/schema/volume/result.json",
        "dbfs:/FileStore/result.json",
        "dbfs:/Volumes/catalog/schema/volume",
        "dbfs:/Volumes/catalog/schema/volume//result.json",
        "dbfs:/Volumes/catalog/schema/volume/../result.json",
        "dbfs:/Volumes/catalog/schema/volume/%2e%2e/result.json",
        "dbfs:/Volumes/catalog/schema/volume/result.json?download=true",
        "dbfs:/Volumes/catalog/schema/volume/result.json#fragment",
    ),
)
def test_download_databricks_volume_file_bytes_rejects_unsafe_uri(uri):
    opener = _BinaryOpener(b"unused")

    with pytest.raises(ValueError, match="dbfs_uri|dbfs:/Volumes|canonical"):
        download_databricks_volume_file_bytes(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            uri,
            opener=opener,
        )

    assert opener.requests == []


@pytest.mark.parametrize(
    "max_bytes",
    (False, 0, -1, DATABRICKS_VOLUME_FILE_MAX_DOWNLOAD_BYTES + 1),
)
def test_download_databricks_volume_file_bytes_rejects_invalid_size_cap(max_bytes):
    opener = _BinaryOpener(b"unused")

    with pytest.raises(ValueError, match="max_bytes"):
        download_databricks_volume_file_bytes(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            "dbfs:/Volumes/catalog/schema/volume/result.json",
            max_bytes=max_bytes,
            opener=opener,
        )

    assert opener.requests == []


def test_download_databricks_volume_file_bytes_rejects_oversize_response():
    opener = _BinaryOpener(b"12345")

    with pytest.raises(RuntimeError, match="exceeds the controller byte cap"):
        download_databricks_volume_file_bytes(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            "dbfs:/Volumes/catalog/schema/volume/result.json",
            max_bytes=4,
            opener=opener,
        )

    assert opener.response.read_limits == [5]


def test_download_databricks_volume_file_bytes_sanitizes_http_errors():
    error_body = _BytesFile(
        b'{"message":"Authorization: Bearer secret-token; token=secret-token"}'
    )
    error = urllib.error.HTTPError(
        "https://dbc.example/api/2.0/fs/files/Volumes/c/s/v/result.json",
        403,
        "Forbidden",
        {},
        error_body,
    )

    with pytest.raises(RuntimeError, match="HTTP 403") as excinfo:
        download_databricks_volume_file_bytes(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            "dbfs:/Volumes/c/s/v/result.json",
            opener=_HTTPErrorOpener(error),
        )

    assert "secret-token" not in str(excinfo.value)
    assert error_body.read_limits == [
        public_databricks_runs._DATABRICKS_ERROR_BODY_MAX_BYTES + 1
    ]


def test_stream_volume_file_sha256_is_authenticated_bounded_and_unbuffered():
    chunk_size = public_databricks_runs._DATABRICKS_VOLUME_FILE_STREAM_CHUNK_BYTES
    content = b"a" * chunk_size + b"b" * chunk_size + b"tail"
    opener = _StreamingBinaryOpener(content)
    config = DatabricksWorkspaceConfig(
        "https://dbc.example/", "secret-token", timeout_seconds=13
    )

    record = stream_databricks_volume_file_sha256(
        config,
        "dbfs:/Volumes/catalog/schema/volume/runtime/package wheel.whl",
        max_bytes=len(content),
        opener=opener,
    )

    assert record == {
        "dbfs_uri": "dbfs:/Volumes/catalog/schema/volume/runtime/package wheel.whl",
        "file_sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }
    request = opener.requests[0]
    assert request.full_url.endswith("/runtime/package%20wheel.whl")
    assert request.get_method() == "GET"
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert request.headers["Accept"] == "application/octet-stream"
    assert request.headers["Accept-encoding"] == "identity"
    assert opener.timeouts == [13]
    assert opener.response.read_limits == [chunk_size, chunk_size, 4, 1]
    assert max(opener.response.read_limits) == chunk_size


@pytest.mark.parametrize(
    ("headers", "error_match"),
    (
        ({}, "content-length.*missing or invalid"),
        ({"content-length": "not-an-int"}, "content-length.*missing or invalid"),
        ({"content-length": "5"}, "content-length exceeds.*byte cap"),
        (
            {"content-length": "4", "transfer-encoding": "chunked"},
            "transfer-encoding is unexpected",
        ),
        (
            {"content-length": "4", "content-encoding": "gzip"},
            "content-encoding is not identity",
        ),
    ),
)
def test_stream_volume_file_sha256_rejects_content_length_before_body(
    headers,
    error_match,
):
    opener = _StreamingBinaryOpener(b"12345", headers=headers)

    with pytest.raises(RuntimeError, match=error_match):
        stream_databricks_volume_file_sha256(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            "dbfs:/Volumes/catalog/schema/volume/runtime/package.whl",
            max_bytes=4,
            opener=opener,
        )

    assert opener.response.read_limits == []


def test_stream_volume_file_sha256_rejects_short_trailing_and_oversized_chunks():
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")
    uri = "dbfs:/Volumes/catalog/schema/volume/runtime/package.whl"

    short = _StreamingBinaryOpener(b"1234", headers={"content-length": "5"})
    with pytest.raises(RuntimeError, match="ended before content-length"):
        stream_databricks_volume_file_sha256(config, uri, opener=short)

    trailing = _StreamingBinaryOpener(b"12345", headers={"content-length": "4"})
    with pytest.raises(RuntimeError, match="bytes beyond content-length"):
        stream_databricks_volume_file_sha256(config, uri, opener=trailing)

    oversized_chunk = _BinaryOpener(
        b"12",
        headers={"content-length": "1"},
        oversized_reads=True,
    )
    with pytest.raises(RuntimeError, match="chunk byte cap"):
        stream_databricks_volume_file_sha256(config, uri, opener=oversized_chunk)


def test_stream_volume_file_sha256_caps_inputs_status_and_redacts_errors():
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")
    uri = "dbfs:/Volumes/catalog/schema/volume/runtime/package.whl"
    opener = _StreamingBinaryOpener(b"")
    for invalid_cap in (False, 0, DATABRICKS_VOLUME_FILE_MAX_STREAM_BYTES + 1):
        with pytest.raises(ValueError, match="max_bytes"):
            stream_databricks_volume_file_sha256(
                config,
                uri,
                max_bytes=invalid_cap,
                opener=opener,
            )
    with pytest.raises(ValueError, match="canonical"):
        stream_databricks_volume_file_sha256(
            config,
            "dbfs:/Volumes/catalog/schema/volume/runtime/../package.whl",
            opener=opener,
        )
    assert opener.requests == []

    with pytest.raises(RuntimeError, match="unexpected HTTP status"):
        stream_databricks_volume_file_sha256(
            config,
            uri,
            opener=_StreamingBinaryOpener(b"", status=206),
        )
    with pytest.raises(RuntimeError, match="unexpected HTTP status"):
        stream_databricks_volume_file_sha256(
            config,
            uri,
            opener=_StreamingBinaryOpener(b"", status=200.0),
        )

    error_body = _BytesFile(b'{"message":"Bearer secret-token"}')
    error = urllib.error.HTTPError(
        "https://dbc.example/api/2.0/fs/files/Volumes/c/s/v/package.whl",
        403,
        "Forbidden",
        {},
        error_body,
    )
    with pytest.raises(RuntimeError, match="HTTP 403") as excinfo:
        stream_databricks_volume_file_sha256(
            config,
            uri,
            opener=_HTTPErrorOpener(error),
        )
    assert "secret-token" not in str(excinfo.value)
    assert error_body.read_limits == [
        public_databricks_runs._DATABRICKS_ERROR_BODY_MAX_BYTES + 1
    ]


def test_list_active_runs_is_paginated_bounded_sorted_and_sanitized():
    opener = _SequentialBinaryOpener(
        [
            {
                "has_more": True,
                "next_page_token": "opaque+/= token",
                "runs": [
                    {
                        "run_id": 22,
                        "run_name": "second",
                        "state": {
                            "life_cycle_state": "RUNNING",
                            "state_message": "Bearer secret-token",
                        },
                    }
                ],
            },
            {
                "has_more": False,
                "prev_page_token": "previous",
                "runs": [
                    {
                        "run_id": 11,
                        "run_name": "first",
                        "state": {"life_cycle_state": "QUEUED"},
                    }
                ],
            },
        ]
    )
    config = DatabricksWorkspaceConfig(
        "https://dbc.example/", "secret-token", timeout_seconds=17
    )

    runs = list_active_databricks_runs(config, opener=opener)

    assert runs == (
        {"life_cycle_state": "QUEUED", "run_id": 11},
        {"life_cycle_state": "RUNNING", "run_id": 22},
    )
    assert "secret-token" not in json.dumps(runs)
    assert opener.requests[0].full_url.endswith(
        "/api/2.1/jobs/runs/list?active_only=true&limit=20"
    )
    assert opener.requests[1].full_url.endswith(
        "active_only=true&limit=20&page_token=opaque%2B%2F%3D+token"
    )
    assert all(
        request.headers["Authorization"] == "Bearer secret-token"
        for request in opener.requests
    )
    assert opener.timeouts == [17, 17]


def test_list_active_runs_empty_snapshot_and_entry_cap_fail_closed():
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")
    assert (
        list_active_databricks_runs(
            config,
            opener=_SequentialBinaryOpener([{}]),
        )
        == ()
    )

    invalid_cap_opener = _SequentialBinaryOpener([])
    for invalid_cap in (False, 0, DATABRICKS_ACTIVE_RUNS_MAX_ENTRIES + 1):
        with pytest.raises(ValueError, match="max_runs"):
            list_active_databricks_runs(
                config,
                max_runs=invalid_cap,
                opener=invalid_cap_opener,
            )
    assert invalid_cap_opener.requests == []

    too_many = _SequentialBinaryOpener(
        [
            {
                "runs": [
                    {"run_id": 1, "state": {"life_cycle_state": "RUNNING"}},
                    {"run_id": 2, "state": {"life_cycle_state": "RUNNING"}},
                ]
            }
        ]
    )
    with pytest.raises(RuntimeError, match="page entry cap"):
        list_active_databricks_runs(config, max_runs=1, opener=too_many)


def test_list_active_runs_rejects_duplicate_cycle_oversized_token_and_page_dos():
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")
    run = {"run_id": 1, "state": {"life_cycle_state": "RUNNING"}}
    duplicate = _SequentialBinaryOpener(
        [
            {"runs": [run], "next_page_token": "next"},
            {"runs": [run]},
        ]
    )
    with pytest.raises(RuntimeError, match="duplicates run_id"):
        list_active_databricks_runs(config, opener=duplicate)

    cycle = _SequentialBinaryOpener(
        [
            {"runs": [], "next_page_token": "next"},
            {"runs": [], "next_page_token": "next"},
        ]
    )
    with pytest.raises(RuntimeError, match="token repeated"):
        list_active_databricks_runs(config, opener=cycle)

    oversized_token = _SequentialBinaryOpener(
        [
            {
                "runs": [],
                "next_page_token": "x" * (DATABRICKS_API_PAGE_TOKEN_MAX_BYTES + 1),
            }
        ]
    )
    with pytest.raises(RuntimeError, match="token byte cap"):
        list_active_databricks_runs(config, opener=oversized_token)
    assert len(oversized_token.requests) == 1

    page_dos = _SequentialBinaryOpener(
        [
            {"runs": [], "next_page_token": f"unique-{index}"}
            for index in range(DATABRICKS_ACTIVE_RUNS_MAX_PAGES)
        ]
    )
    with pytest.raises(RuntimeError, match="page cap"):
        list_active_databricks_runs(config, opener=page_dos)
    assert len(page_dos.requests) == DATABRICKS_ACTIVE_RUNS_MAX_PAGES


def test_list_active_runs_caps_response_bytes_and_redacts_http_error():
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")
    with pytest.raises(RuntimeError, match="duplicate key"):
        list_active_databricks_runs(
            config,
            opener=_BinaryOpener(b'{"runs":[],"runs":[]}'),
        )

    with pytest.raises(RuntimeError, match="secret-like text") as secret_exc:
        list_active_databricks_runs(
            config,
            opener=_SequentialBinaryOpener(
                [
                    {
                        "runs": [
                            {
                                "run_id": 1,
                                "state": {"life_cycle_state": "secret-token"},
                            }
                        ]
                    }
                ]
            ),
        )
    assert "secret-token" not in str(secret_exc.value)

    with pytest.raises(RuntimeError, match="response exceeds.*byte cap"):
        list_active_databricks_runs(
            config,
            opener=_BinaryOpener(b"x" * (DATABRICKS_API_PAGE_MAX_BYTES + 1)),
        )

    error = urllib.error.HTTPError(
        "https://dbc.example/api/2.1/jobs/runs/list",
        403,
        "Forbidden",
        {},
        _BytesFile(b'{"message":"Bearer secret-token"}'),
    )
    with pytest.raises(RuntimeError, match="HTTP 403") as excinfo:
        list_active_databricks_runs(config, opener=_HTTPErrorOpener(error))
    assert "secret-token" not in str(excinfo.value)


def test_list_node_types_is_authenticated_bounded_sorted_and_sanitized():
    opener = _SequentialBinaryOpener(
        [
            {
                "node_types": [
                    {
                        "node_type_id": "g6e.4xlarge",
                        "description": "Bearer secret-token",
                    },
                    {"node_type_id": "g5.8xlarge", "memory_mb": 131072},
                    {"node_type_id": "g6.8xlarge", "num_cores": 32},
                ],
                "success": {},
            }
        ]
    )
    config = DatabricksWorkspaceConfig(
        "https://dbc.example/", "secret-token", timeout_seconds=19
    )

    node_types = list_databricks_node_types(config, opener=opener)

    assert node_types == (
        {"node_type_id": "g5.8xlarge"},
        {"node_type_id": "g6.8xlarge"},
        {"node_type_id": "g6e.4xlarge"},
    )
    assert "secret-token" not in json.dumps(node_types)
    request = opener.requests[0]
    assert request.full_url == ("https://dbc.example/api/2.0/clusters/list-node-types")
    assert request.get_method() == "GET"
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert opener.timeouts == [19]


def test_list_node_types_rejects_caps_schema_duplicates_and_invalid_ids():
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")
    invalid_cap_opener = _SequentialBinaryOpener([])
    for invalid_cap in (False, 0, DATABRICKS_NODE_TYPES_MAX_ENTRIES + 1):
        with pytest.raises(ValueError, match="max_node_types"):
            list_databricks_node_types(
                config,
                max_node_types=invalid_cap,
                opener=invalid_cap_opener,
            )
    assert invalid_cap_opener.requests == []

    cases = (
        ({"node_types": [], "unexpected": True}, "schema drift"),
        ({"node_types": [], "success": {"unexpected": True}}, "success marker"),
        ({"node_types": [{"node_type_id": " bad "}]}, "node_type_id.*invalid"),
        (
            {"node_types": [{"node_type_id": "secret-token"}]},
            "secret-like text",
        ),
        (
            {
                "node_types": [
                    {"node_type_id": "g5.8xlarge"},
                    {"node_type_id": "g5.8xlarge"},
                ]
            },
            "duplicates node_type_id",
        ),
    )
    for response, error_match in cases:
        with pytest.raises(RuntimeError, match=error_match):
            list_databricks_node_types(
                config,
                opener=_SequentialBinaryOpener([response]),
            )

    with pytest.raises(RuntimeError, match="entry cap"):
        list_databricks_node_types(
            config,
            max_node_types=1,
            opener=_SequentialBinaryOpener(
                [
                    {
                        "node_types": [
                            {"node_type_id": "g5.8xlarge"},
                            {"node_type_id": "g6.8xlarge"},
                        ]
                    }
                ]
            ),
        )


def test_list_databricks_volume_directory_is_authenticated_paginated_and_sorted():
    opener = _SequentialBinaryOpener(
        [
            {
                "contents": [
                    {
                        "file_size": 9,
                        "is_directory": False,
                        "last_modified": 12,
                        "name": "z.json",
                        "path": "/Volumes/catalog/schema/volume/results/z.json",
                    }
                ],
                "next_page_token": "opaque+/= token",
            },
            {
                "contents": [
                    {
                        "is_directory": True,
                        "name": "a",
                        "path": "/Volumes/catalog/schema/volume/results/a/",
                    }
                ]
            },
        ]
    )
    config = DatabricksWorkspaceConfig(
        "https://dbc.example/", "secret-token", timeout_seconds=11
    )

    entries = list_databricks_volume_directory(
        config,
        "dbfs:/Volumes/catalog/schema/volume/results",
        opener=opener,
    )

    assert [entry["name"] for entry in entries] == ["a", "z.json"]
    assert entries[0] == {
        "is_directory": True,
        "name": "a",
        "path": "/Volumes/catalog/schema/volume/results/a/",
    }
    assert entries[1] == {
        "file_size": 9,
        "is_directory": False,
        "last_modified": 12,
        "name": "z.json",
        "path": "/Volumes/catalog/schema/volume/results/z.json",
    }
    assert opener.requests[0].full_url.endswith(
        "/directories/Volumes/catalog/schema/volume/results?page_size=1000"
    )
    assert opener.requests[1].full_url.endswith(
        "?page_size=1000&page_token=opaque%2B%2F%3D+token"
    )
    assert all(
        request.headers["Authorization"] == "Bearer secret-token"
        for request in opener.requests
    )
    assert opener.timeouts == [11, 11]


def test_list_databricks_volume_directory_accepts_empty_page_forms():
    config = DatabricksWorkspaceConfig(
        "https://dbc.example", "secret-token", timeout_seconds=13
    )
    for payload in (b"{}", {"contents": []}):
        opener = _SequentialBinaryOpener([payload])

        assert list_databricks_volume_directory(
            config,
            "dbfs:/Volumes/catalog/schema/volume/results",
            opener=opener,
        ) == ()

        assert len(opener.requests) == 1
        assert opener.requests[0].headers["Authorization"] == "Bearer secret-token"
        assert opener.timeouts == [13]


def test_list_databricks_volume_directory_rejects_empty_mapping_after_page():
    parent = "/Volumes/catalog/schema/volume/results"
    entry = {
        "file_size": 1,
        "is_directory": False,
        "last_modified": 1,
        "name": "result.json",
        "path": parent + "/result.json",
    }
    for first_page_contents in ([], [entry]):
        opener = _SequentialBinaryOpener(
            [
                {"contents": first_page_contents, "next_page_token": "next"},
                b"{}",
            ]
        )

        with pytest.raises(RuntimeError, match="schema drift"):
            list_databricks_volume_directory(
                DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
                "dbfs:" + parent,
                opener=opener,
            )

        assert len(opener.requests) == 2
        assert opener.requests[1].full_url.endswith(
            "?page_size=1000&page_token=next"
        )


def test_list_databricks_volume_directory_rejects_nonempty_missing_contents():
    malformed_pages = (
        {"next_page_token": "next"},
        {"unexpected": "Bearer secret-token"},
        {"contents": None},
        {"contents": {}},
        {"contents": "[]"},
        {"contents": False},
        {"contents": 0},
        {"contents": [], "unexpected": True},
    )
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")

    for page in malformed_pages:
        with pytest.raises(
            RuntimeError,
            match="schema drift|contents must be an array",
        ) as exc_info:
            list_databricks_volume_directory(
                config,
                "dbfs:/Volumes/catalog/schema/volume/results",
                opener=_SequentialBinaryOpener([page]),
            )

        formatted = "".join(
            traceback.format_exception(
                type(exc_info.value),
                exc_info.value,
                exc_info.value.__traceback__,
            )
        )
        assert "secret-token" not in formatted


@pytest.mark.parametrize(
    "entry",
    (
        {
            "is_directory": True,
            "last_modified": 1,
            "name": "directory",
            "path": "/Volumes/catalog/schema/volume/results/directory/",
        },
        {
            "is_directory": True,
            "name": "directory",
            "path": "/Volumes/catalog/schema/volume/results/directory",
        },
        {
            "file_size": 1,
            "is_directory": False,
            "last_modified": 1,
            "name": "file.json",
            "path": "/Volumes/catalog/schema/volume/results/file.json/",
        },
    ),
)
def test_list_databricks_volume_directory_rejects_cross_kind_entry_shapes(entry):
    opener = _SequentialBinaryOpener([{"contents": [entry]}])

    with pytest.raises(RuntimeError, match="schema drift|path kind"):
        list_databricks_volume_directory(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            "dbfs:/Volumes/catalog/schema/volume/results",
            opener=opener,
        )


@pytest.mark.parametrize(
    "uri",
    (
        "",
        "/Volumes/catalog/schema/volume/results",
        "dbfs:/Volumes/catalog/schema",
        "dbfs:/Volumes/catalog/schema/volume/",
        "dbfs:/Volumes/catalog/schema/volume/results/../nested",
        "dbfs:/Volumes/catalog/schema/volume/results%2Fnested",
    ),
)
def test_list_databricks_volume_directory_rejects_unsafe_uri(uri):
    opener = _SequentialBinaryOpener([])

    with pytest.raises(ValueError, match="dbfs_uri|dbfs:/Volumes|canonical"):
        list_databricks_volume_directory(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            uri,
            opener=opener,
        )

    assert opener.requests == []


def test_list_databricks_volume_directory_rejects_tamper_cap_and_token_cycle():
    parent = "/Volumes/catalog/schema/volume/results"
    malformed = _SequentialBinaryOpener(
        [
            {
                "contents": [
                    {
                        "file_size": 1,
                        "is_directory": False,
                        "last_modified": 1,
                        "name": "x",
                        "path": parent + "/nested/x",
                    }
                ]
            }
        ]
    )
    with pytest.raises(RuntimeError, match="direct child"):
        list_databricks_volume_directory(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            "dbfs:" + parent,
            opener=malformed,
        )

    entry = {
        "file_size": 1,
        "is_directory": False,
        "last_modified": 1,
        "name": "x",
        "path": parent + "/x",
    }
    capped = _SequentialBinaryOpener([{"contents": [entry]}])
    with pytest.raises(RuntimeError, match="entry cap"):
        list_databricks_volume_directory(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            "dbfs:" + parent,
            max_entries=0,
            opener=capped,
        )
    with pytest.raises(ValueError, match="max_entries"):
        list_databricks_volume_directory(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            "dbfs:" + parent,
            max_entries=DATABRICKS_VOLUME_DIRECTORY_MAX_ENTRIES + 1,
            opener=_SequentialBinaryOpener([]),
        )

    cycle = _SequentialBinaryOpener(
        [
            {"contents": [], "next_page_token": "next"},
            {"contents": [], "next_page_token": "next"},
        ]
    )
    with pytest.raises(RuntimeError, match="token repeated"):
        list_databricks_volume_directory(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            "dbfs:" + parent,
            opener=cycle,
        )


def test_list_databricks_volume_directory_caps_unique_empty_pages():
    opener = _SequentialBinaryOpener(
        [
            {"contents": [], "next_page_token": f"unique-{index}"}
            for index in range(DATABRICKS_VOLUME_DIRECTORY_MAX_PAGES)
        ]
    )

    with pytest.raises(RuntimeError, match="page cap"):
        list_databricks_volume_directory(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            "dbfs:/Volumes/catalog/schema/volume/results",
            opener=opener,
        )

    assert len(opener.requests) == DATABRICKS_VOLUME_DIRECTORY_MAX_PAGES


def test_list_databricks_volume_directory_caps_page_token_before_next_request():
    opener = _SequentialBinaryOpener(
        [
            {
                "contents": [],
                "next_page_token": "x"
                * (DATABRICKS_VOLUME_DIRECTORY_MAX_PAGE_TOKEN_BYTES + 1),
            }
        ]
    )

    with pytest.raises(RuntimeError, match="next_page_token.*byte cap"):
        list_databricks_volume_directory(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            "dbfs:/Volumes/catalog/schema/volume/results",
            opener=opener,
        )

    assert len(opener.requests) == 1


def test_get_databricks_volume_file_metadata_uses_authenticated_head_and_cap():
    opener = _BinaryOpener(
        b"",
        headers={
            "content-length": "17",
            "content-type": "application/json",
            "last-modified": "Mon, 24 Aug 2026 12:34:56 GMT",
        },
    )
    config = DatabricksWorkspaceConfig(
        "https://dbc.example/", "secret-token", timeout_seconds=7
    )

    metadata = get_databricks_volume_file_metadata(
        config,
        "dbfs:/Volumes/catalog/schema/volume/control/request file.json",
        max_bytes=17,
        opener=opener,
    )

    assert metadata["content_length"] == 17
    assert metadata["content_type"] == "application/json"
    request = opener.requests[0]
    assert request.get_method() == "HEAD"
    assert request.full_url.endswith("/control/request%20file.json")
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert opener.response.read_limits == [1]


def test_get_databricks_volume_file_metadata_rejects_unsafe_oversize_and_redacts():
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")
    opener = _BinaryOpener(b"", headers={"content-length": "0"})
    with pytest.raises(ValueError, match="canonical"):
        get_databricks_volume_file_metadata(
            config,
            "dbfs:/Volumes/c/s/v/../request.json",
            opener=opener,
        )
    assert opener.requests == []

    with pytest.raises(RuntimeError, match="byte cap"):
        get_databricks_volume_file_metadata(
            config,
            "dbfs:/Volumes/c/s/v/request.json",
            max_bytes=4,
            opener=_BinaryOpener(b"", headers={"content-length": "5"}),
        )

    error = urllib.error.HTTPError(
        "https://dbc.example/api/2.0/fs/files/Volumes/c/s/v/request.json",
        403,
        "Forbidden",
        {},
        _BytesFile(b'{"message":"Bearer secret-token"}'),
    )
    with pytest.raises(RuntimeError, match="HTTP 403") as excinfo:
        get_databricks_volume_file_metadata(
            config,
            "dbfs:/Volumes/c/s/v/request.json",
            opener=_HTTPErrorOpener(error),
        )
    assert "secret-token" not in str(excinfo.value)


def test_upload_databricks_volume_file_bytes_exclusive_uses_raw_no_overwrite_put():
    content = b'{"canonical":true}\n'
    opener = _BinaryOpener(b"", status=204)
    config = DatabricksWorkspaceConfig(
        "https://dbc.example/", "secret-token", timeout_seconds=8
    )

    record = upload_databricks_volume_file_bytes_exclusive(
        config,
        "dbfs:/Volumes/catalog/schema/volume/control/request file.json",
        content,
        opener=opener,
    )

    assert record["created"] is True
    assert record["file_sha256"] == hashlib.sha256(content).hexdigest()
    request = opener.requests[0]
    assert request.get_method() == "PUT"
    assert request.full_url.endswith("request%20file.json?overwrite=false")
    assert request.data == content
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert request.headers["Content-type"] == "application/octet-stream"


@pytest.mark.parametrize("failure", ("conflict", "lost-response"))
def test_upload_databricks_volume_file_bytes_exclusive_replays_identical_bytes(
    failure,
):
    content = b"canonical-request"
    if failure == "conflict":
        error = urllib.error.HTTPError(
            "https://dbc.example/api/2.0/fs/files/Volumes/c/s/v/request.json",
            409,
            "Conflict",
            {},
            _BytesFile(b'{"error_code":"RESOURCE_ALREADY_EXISTS"}'),
        )
        put_opener = _HTTPErrorOpener(error)
    else:
        put_opener = _ExceptionOpener(TimeoutError("accepted response lost"))
    readback = _BinaryOpener(content)

    record = upload_databricks_volume_file_bytes_exclusive(
        DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
        "dbfs:/Volumes/c/s/v/request.json",
        content,
        opener=put_opener,
        readback_opener=readback,
    )

    assert record["created"] is False
    assert readback.requests[0].get_method() == "GET"


def test_upload_databricks_volume_file_bytes_exclusive_rejects_conflict_and_caps():
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")
    error = urllib.error.HTTPError(
        "https://dbc.example/api/2.0/fs/files/Volumes/c/s/v/request.json",
        409,
        "Conflict",
        {},
        _BytesFile(b'{"error_code":"RESOURCE_ALREADY_EXISTS"}'),
    )
    with pytest.raises(RuntimeError, match="different existing bytes"):
        upload_databricks_volume_file_bytes_exclusive(
            config,
            "dbfs:/Volumes/c/s/v/request.json",
            b"wanted",
            opener=_HTTPErrorOpener(error),
            readback_opener=_BinaryOpener(b"different"),
        )

    opener = _BinaryOpener(b"", status=204)
    with pytest.raises(ValueError, match="canonical"):
        upload_databricks_volume_file_bytes_exclusive(
            config,
            "dbfs:/Volumes/c/s/v/../request.json",
            b"request",
            opener=opener,
        )
    with pytest.raises(ValueError, match="upload cap"):
        upload_databricks_volume_file_bytes_exclusive(
            config,
            "dbfs:/Volumes/c/s/v/request.json",
            b"12345",
            max_bytes=4,
            opener=opener,
        )
    with pytest.raises(ValueError, match="max_bytes"):
        upload_databricks_volume_file_bytes_exclusive(
            config,
            "dbfs:/Volumes/c/s/v/request.json",
            b"request",
            max_bytes=DATABRICKS_VOLUME_FILE_MAX_UPLOAD_BYTES + 1,
            opener=opener,
        )
    assert opener.requests == []


def test_upload_databricks_volume_file_bytes_exclusive_redacts_http_errors():
    error = urllib.error.HTTPError(
        "https://dbc.example/api/2.0/fs/files/Volumes/c/s/v/request.json",
        403,
        "Forbidden",
        {},
        _BytesFile(b'{"message":"Bearer secret-token token=secret-token"}'),
    )

    with pytest.raises(RuntimeError, match="HTTP 403") as excinfo:
        upload_databricks_volume_file_bytes_exclusive(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            "dbfs:/Volumes/c/s/v/request.json",
            b"request",
            opener=_HTTPErrorOpener(error),
        )

    assert "secret-token" not in str(excinfo.value)


def test_upload_databricks_volume_file_path_exclusive_streams_and_proves_remote(
    tmp_path,
):
    content = b"verified-local-upload" * 100_000
    source = tmp_path / "runtime wheel.whl"
    source.write_bytes(content)
    upload = _ConsumingUploadOpener()
    readback = _StreamingBinaryOpener(content)
    config = DatabricksWorkspaceConfig(
        "https://dbc.example/", "secret-token", timeout_seconds=19
    )
    uri = "dbfs:/Volumes/catalog/schema/volume/runtime/runtime wheel.whl"

    record = upload_databricks_volume_file_path_exclusive(
        config,
        uri,
        source,
        expected_sha256=hashlib.sha256(content).hexdigest(),
        expected_size=len(content),
        opener=upload,
        readback_opener=readback,
    )

    assert record == {
        "created": True,
        "dbfs_uri": uri,
        "file_sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }
    request = upload.requests[0]
    assert request.get_method() == "PUT"
    assert request.full_url.endswith(
        "/runtime/runtime%20wheel.whl?overwrite=false"
    )
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert request.headers["Content-length"] == str(len(content))
    assert request.headers["Content-type"] == "application/octet-stream"
    assert not isinstance(request.data, bytes)
    assert upload.sha256 == hashlib.sha256(content).hexdigest()
    assert upload.total_bytes == len(content)
    assert max(upload.chunk_sizes) <= (
        public_databricks_runs._DATABRICKS_VOLUME_FILE_STREAM_CHUNK_BYTES
    )
    assert upload.timeouts == [19]
    assert readback.requests[0].get_method() == "GET"


def test_upload_databricks_volume_file_path_exclusive_streams_83113106_sparse_bytes(
    tmp_path,
):
    size_bytes = 83_113_106
    source = tmp_path / "flashinfer-python.whl"
    with source.open("wb") as stream:
        stream.seek(size_bytes - 1)
        stream.write(b"\0")
    expected_sha256 = _repeated_zero_sha256(size_bytes)
    upload = _ConsumingUploadOpener()
    readback = _GeneratedZeroStreamingOpener(size_bytes)

    record = upload_databricks_volume_file_path_exclusive(
        DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
        "dbfs:/Volumes/c/s/v/runtime/flashinfer-python.whl",
        source,
        expected_sha256=expected_sha256,
        expected_size=size_bytes,
        opener=upload,
        readback_opener=readback,
    )

    assert record["size_bytes"] == size_bytes
    assert record["file_sha256"] == expected_sha256
    assert upload.total_bytes == size_bytes
    assert upload.sha256 == expected_sha256
    assert max(upload.chunk_sizes) <= 1024 * 1024
    assert max(readback.response.read_limits) <= 1024 * 1024


def test_upload_databricks_volume_file_path_exclusive_rejects_caps_and_bad_pins(
    tmp_path,
):
    source = tmp_path / "package.whl"
    source.write_bytes(b"package")
    digest = hashlib.sha256(b"package").hexdigest()
    uri = "dbfs:/Volumes/c/s/v/package.whl"
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")
    upload = _ConsumingUploadOpener()

    cases = (
        {"expected_sha256": "A" * 64, "expected_size": 7},
        {"expected_sha256": digest, "expected_size": False},
        {"expected_sha256": digest, "expected_size": 8},
        {"expected_sha256": "0" * 64, "expected_size": 7},
        {
            "expected_sha256": digest,
            "expected_size": 7,
            "max_bytes": DATABRICKS_VOLUME_FILE_MAX_STREAM_BYTES + 1,
        },
        {"expected_sha256": digest, "expected_size": 7, "max_bytes": 6},
    )
    for arguments in cases:
        with pytest.raises(ValueError, match="expected|max_bytes|SHA-256|size"):
            upload_databricks_volume_file_path_exclusive(
                config,
                uri,
                source,
                opener=upload,
                **arguments,
            )
    assert upload.requests == []


def test_upload_databricks_volume_file_path_exclusive_rejects_unsafe_sources(
    tmp_path,
    monkeypatch,
):
    content = b"package"
    digest = hashlib.sha256(content).hexdigest()
    source = tmp_path / "package.whl"
    source.write_bytes(content)
    symlink = tmp_path / "symlink.whl"
    symlink.symlink_to(source)
    hardlink = tmp_path / "hardlink.whl"
    os.link(source, hardlink)
    fifo = tmp_path / "package.fifo"
    os.mkfifo(fifo)
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")
    uri = "dbfs:/Volumes/c/s/v/package.whl"
    upload = _ConsumingUploadOpener()

    for unsafe in (source.name, symlink, hardlink, fifo):
        with pytest.raises(ValueError, match="canonical|symbolic|hard link|regular"):
            upload_databricks_volume_file_path_exclusive(
                config,
                uri,
                unsafe,
                expected_sha256=digest,
                expected_size=len(content),
                opener=upload,
            )
    os.unlink(hardlink)
    real_uid = os.getuid()
    monkeypatch.setattr(
        public_databricks_runs.os, "getuid", lambda: real_uid + 1
    )
    with pytest.raises(ValueError, match="owned by the current user"):
        upload_databricks_volume_file_path_exclusive(
            config,
            uri,
            source,
            expected_sha256=digest,
            expected_size=len(content),
            opener=upload,
        )
    assert upload.requests == []


def test_upload_databricks_volume_file_path_exclusive_rejects_replacement(
    tmp_path,
):
    content = b"original-package"
    source = tmp_path / "package.whl"
    source.write_bytes(content)

    def replace_source():
        source.rename(tmp_path / "original.whl")
        source.write_bytes(content)

    upload = _ConsumingUploadOpener(before_consume=replace_source)
    with pytest.raises(RuntimeError, match="source identity drifted"):
        upload_databricks_volume_file_path_exclusive(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            "dbfs:/Volumes/c/s/v/package.whl",
            source,
            expected_sha256=hashlib.sha256(content).hexdigest(),
            expected_size=len(content),
            opener=upload,
        )
    assert upload.chunk_sizes == []


def test_upload_databricks_volume_file_path_exclusive_rejects_midstream_mutation(
    tmp_path,
):
    chunk_size = public_databricks_runs._DATABRICKS_VOLUME_FILE_STREAM_CHUNK_BYTES
    content = b"a" * (chunk_size + 1)
    source = tmp_path / "package.whl"
    source.write_bytes(content)

    def mutate_source(_chunk_count):
        with source.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            stream.write(b"b")

    upload = _ConsumingUploadOpener(after_chunk=mutate_source)
    with pytest.raises(RuntimeError, match="source identity drifted"):
        upload_databricks_volume_file_path_exclusive(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            "dbfs:/Volumes/c/s/v/package.whl",
            source,
            expected_sha256=hashlib.sha256(content).hexdigest(),
            expected_size=len(content),
            opener=upload,
        )
    assert len(upload.chunk_sizes) == 1


@pytest.mark.parametrize(
    ("failure_mode", "error_match"),
    (("short", "ended before"), ("extra", "beyond its verified size")),
)
def test_upload_databricks_volume_file_path_exclusive_rejects_read_drift(
    tmp_path,
    monkeypatch,
    failure_mode,
    error_match,
):
    content = b"abc"
    source = tmp_path / "package.whl"
    source.write_bytes(content)
    real_read = os.read
    call_count = 0

    def drifting_read(file_descriptor, amount):
        nonlocal call_count
        call_count += 1
        if failure_mode == "short" and call_count == 3:
            return b""
        if failure_mode == "extra" and call_count == 4:
            return b"x"
        return real_read(file_descriptor, amount)

    monkeypatch.setattr(public_databricks_runs.os, "read", drifting_read)
    with pytest.raises(RuntimeError, match=error_match):
        upload_databricks_volume_file_path_exclusive(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            "dbfs:/Volumes/c/s/v/package.whl",
            source,
            expected_sha256=hashlib.sha256(content).hexdigest(),
            expected_size=len(content),
            opener=_ConsumingUploadOpener(),
        )


@pytest.mark.parametrize("failure", (400, 409, "timeout"))
def test_upload_databricks_volume_file_path_exclusive_proves_replay(
    tmp_path,
    failure,
):
    content = b"canonical-package"
    source = tmp_path / "package.whl"
    source.write_bytes(content)
    if isinstance(failure, int):
        error = urllib.error.HTTPError(
            "https://dbc.example/api/2.0/fs/files/Volumes/c/s/v/package.whl",
            failure,
            "Conflict",
            {},
            _BytesFile(b'{"error_code":"RESOURCE_ALREADY_EXISTS"}'),
        )
        upload = _RecordingHTTPErrorOpener(error)
    else:
        upload = _ExceptionOpener(TimeoutError("accepted response lost"))

    record = upload_databricks_volume_file_path_exclusive(
        DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
        "dbfs:/Volumes/c/s/v/package.whl",
        source,
        expected_sha256=hashlib.sha256(content).hexdigest(),
        expected_size=len(content),
        opener=upload,
        readback_opener=_StreamingBinaryOpener(content),
    )

    assert record["created"] is False
    assert record["file_sha256"] == hashlib.sha256(content).hexdigest()


def test_upload_databricks_volume_file_path_exclusive_rejects_wrong_readback(
    tmp_path,
):
    content = b"canonical-package"
    source = tmp_path / "package.whl"
    source.write_bytes(content)
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")
    uri = "dbfs:/Volumes/c/s/v/package.whl"
    digest = hashlib.sha256(content).hexdigest()
    conflict = urllib.error.HTTPError(
        "https://dbc.example/api/2.0/fs/files/Volumes/c/s/v/package.whl",
        409,
        "Conflict",
        {},
        _BytesFile(b'{"error_code":"RESOURCE_ALREADY_EXISTS"}'),
    )
    with pytest.raises(RuntimeError, match="different remote file"):
        upload_databricks_volume_file_path_exclusive(
            config,
            uri,
            source,
            expected_sha256=digest,
            expected_size=len(content),
            opener=_HTTPErrorOpener(conflict),
            readback_opener=_StreamingBinaryOpener(b"different-package"),
        )
    with pytest.raises(RuntimeError, match="post-PUT readback"):
        upload_databricks_volume_file_path_exclusive(
            config,
            uri,
            source,
            expected_sha256=digest,
            expected_size=len(content),
            opener=_ConsumingUploadOpener(),
            readback_opener=_StreamingBinaryOpener(b"different-package"),
        )
    with pytest.raises(RuntimeError, match="post-PUT readback"):
        upload_databricks_volume_file_path_exclusive(
            config,
            uri,
            source,
            expected_sha256=digest,
            expected_size=len(content),
            opener=_ConsumingUploadOpener(),
            readback_opener=_StreamingBinaryOpener(content + b"x"),
        )
    missing = urllib.error.HTTPError(
        "https://dbc.example/api/2.0/fs/files/Volumes/c/s/v/package.whl",
        404,
        "Not Found",
        {},
        _BytesFile(b'{"error_code":"RESOURCE_DOES_NOT_EXIST"}'),
    )
    with pytest.raises(RuntimeError, match="readback did not prove replay"):
        upload_databricks_volume_file_path_exclusive(
            config,
            uri,
            source,
            expected_sha256=digest,
            expected_size=len(content),
            opener=_ExceptionOpener(TimeoutError("lost")),
            readback_opener=_HTTPErrorOpener(missing),
        )
    raw_conflict_record = upload_databricks_volume_file_path_exclusive(
        config,
        uri,
        source,
        expected_sha256=digest,
        expected_size=len(content),
        opener=_ConsumingUploadOpener(response_status=409),
        readback_opener=_StreamingBinaryOpener(content),
    )
    assert raw_conflict_record["created"] is False


@pytest.mark.parametrize(
    ("status", "payload", "headers", "error_match"),
    (
        (200, b"", {}, "unexpected HTTP status"),
        (204.0, b"", {}, "unexpected HTTP status"),
        (400.0, b"", {}, "unexpected HTTP status"),
        (204, b"unexpected", {}, "body is not empty"),
        (204, b"", {"content-length": "1"}, "content-length is not zero"),
        (204, b"", {"content-encoding": "gzip"}, "content-encoding"),
        (204, b"", {"transfer-encoding": "chunked"}, "transfer-encoding"),
    ),
)
def test_upload_databricks_volume_file_path_exclusive_rejects_response_anomalies(
    tmp_path,
    status,
    payload,
    headers,
    error_match,
):
    content = b"canonical-package"
    source = tmp_path / "package.whl"
    source.write_bytes(content)
    with pytest.raises(RuntimeError, match=error_match):
        upload_databricks_volume_file_path_exclusive(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            "dbfs:/Volumes/c/s/v/package.whl",
            source,
            expected_sha256=hashlib.sha256(content).hexdigest(),
            expected_size=len(content),
            opener=_ConsumingUploadOpener(
                response_payload=payload,
                response_status=status,
                response_headers=headers,
            ),
        )


def test_upload_databricks_volume_file_path_exclusive_redacts_errors(tmp_path):
    content = b"canonical-package"
    source = tmp_path / "package.whl"
    source.write_bytes(content)
    error = urllib.error.HTTPError(
        "https://dbc.example/api/2.0/fs/files/Volumes/c/s/v/package.whl",
        403,
        "Forbidden",
        {},
        _BytesFile(b'{"message":"Bearer secret-token token=secret-token"}'),
    )
    with pytest.raises(RuntimeError, match="HTTP 403") as excinfo:
        upload_databricks_volume_file_path_exclusive(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            "dbfs:/Volumes/c/s/v/package.whl",
            source,
            expected_sha256=hashlib.sha256(content).hexdigest(),
            expected_size=len(content),
            opener=_HTTPErrorOpener(error),
        )
    assert "secret-token" not in str(excinfo.value)
    assert "secret-token" not in "".join(
        traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb)
    )

    with pytest.raises(RuntimeError, match="exclusive path upload failed") as excinfo:
        upload_databricks_volume_file_path_exclusive(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            "dbfs:/Volumes/c/s/v/package.whl",
            source,
            expected_sha256=hashlib.sha256(content).hexdigest(),
            expected_size=len(content),
            opener=_ExceptionOpener(RuntimeError("Bearer secret-token")),
        )
    assert "secret-token" not in "".join(
        traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb)
    )

    with pytest.raises(RuntimeError, match="readback did not prove replay") as excinfo:
        upload_databricks_volume_file_path_exclusive(
            DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
            "dbfs:/Volumes/c/s/v/package.whl",
            source,
            expected_sha256=hashlib.sha256(content).hexdigest(),
            expected_size=len(content),
            opener=_ExceptionOpener(
                urllib.error.URLError("Bearer secret-token")
            ),
            readback_opener=_ExceptionOpener(
                urllib.error.URLError("token=secret-token")
            ),
        )
    assert "secret-token" not in "".join(
        traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb)
    )


def test_create_databricks_volume_directory_idempotent_uses_official_put():
    opener = _BinaryOpener(
        b"",
        status=204,
        headers={"content-length": "0"},
    )
    config = DatabricksWorkspaceConfig(
        "https://dbc.example/", "secret-token", timeout_seconds=23
    )
    uri = "dbfs:/Volumes/catalog/schema/volume/runtime/package wheels"

    record = create_databricks_volume_directory_idempotent(
        config,
        uri,
        opener=opener,
    )

    assert record == {"dbfs_uri": uri}
    request = opener.requests[0]
    assert request.get_method() == "PUT"
    assert request.full_url.endswith(
        "/api/2.0/fs/directories/Volumes/catalog/schema/volume/runtime/"
        "package%20wheels"
    )
    assert request.data is None
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert request.headers["Content-length"] == "0"
    assert opener.timeouts == [23]


@pytest.mark.parametrize("failure", (400, 409, "timeout"))
def test_create_databricks_volume_directory_idempotent_proves_uncertain_outcome(
    failure,
):
    if isinstance(failure, int):
        error = urllib.error.HTTPError(
            "https://dbc.example/api/2.0/fs/directories/Volumes/c/s/v/runtime",
            failure,
            "Conflict",
            {},
            _BytesFile(b'{"error_code":"RESOURCE_ALREADY_EXISTS"}'),
        )
        put = _HTTPErrorOpener(error)
    else:
        put = _ExceptionOpener(TimeoutError("accepted response lost"))
    proof = _BinaryOpener(
        b"",
        status=200,
        headers={"content-length": "0"},
    )
    uri = "dbfs:/Volumes/c/s/v/runtime/nested"

    assert create_databricks_volume_directory_idempotent(
        DatabricksWorkspaceConfig("https://dbc.example", "secret-token"),
        uri,
        opener=put,
        proof_opener=proof,
    ) == {"dbfs_uri": uri}
    assert proof.requests[0].get_method() == "HEAD"
    assert proof.requests[0].headers["Authorization"] == "Bearer secret-token"
    assert proof.requests[0].full_url.endswith("/Volumes/c/s/v/runtime/nested")


def test_create_databricks_volume_directory_idempotent_rejects_anomalies_and_redacts():
    config = DatabricksWorkspaceConfig("https://dbc.example", "secret-token")
    opener = _BinaryOpener(b"", status=204)
    with pytest.raises(ValueError, match="canonical"):
        create_databricks_volume_directory_idempotent(
            config,
            "dbfs:/Volumes/c/s/v/runtime/../escape",
            opener=opener,
        )
    assert opener.requests == []

    with pytest.raises(RuntimeError, match="body is not empty"):
        create_databricks_volume_directory_idempotent(
            config,
            "dbfs:/Volumes/c/s/v/runtime",
            opener=_BinaryOpener(b"unexpected", status=204),
        )

    with pytest.raises(RuntimeError, match="unexpected HTTP status"):
        create_databricks_volume_directory_idempotent(
            config,
            "dbfs:/Volumes/c/s/v/runtime",
            opener=_BinaryOpener(b"", status=204.0),
        )

    assert create_databricks_volume_directory_idempotent(
        config,
        "dbfs:/Volumes/c/s/v/runtime",
        opener=_BinaryOpener(b"conflict", status=409),
        proof_opener=_BinaryOpener(
            b"",
            status=200,
            headers={"content-length": "0"},
        ),
    ) == {"dbfs_uri": "dbfs:/Volumes/c/s/v/runtime"}

    error = urllib.error.HTTPError(
        "https://dbc.example/api/2.0/fs/directories/Volumes/c/s/v/runtime",
        403,
        "Forbidden",
        {},
        _BytesFile(b'{"message":"Bearer secret-token token=secret-token"}'),
    )
    with pytest.raises(RuntimeError, match="HTTP 403") as excinfo:
        create_databricks_volume_directory_idempotent(
            config,
            "dbfs:/Volumes/c/s/v/runtime",
            opener=_HTTPErrorOpener(error),
        )
    assert "secret-token" not in str(excinfo.value)
    assert "secret-token" not in "".join(
        traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb)
    )

    with pytest.raises(RuntimeError, match="directory create failed") as excinfo:
        create_databricks_volume_directory_idempotent(
            config,
            "dbfs:/Volumes/c/s/v/runtime",
            opener=_ExceptionOpener(RuntimeError("Bearer secret-token")),
        )
    assert "secret-token" not in "".join(
        traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb)
    )

    with pytest.raises(RuntimeError, match="existence proof failed") as excinfo:
        create_databricks_volume_directory_idempotent(
            config,
            "dbfs:/Volumes/c/s/v/runtime",
            opener=_ExceptionOpener(
                urllib.error.URLError("Bearer secret-token")
            ),
            proof_opener=_ExceptionOpener(
                RuntimeError("token=secret-token")
            ),
        )
    assert "secret-token" not in "".join(
        traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb)
    )


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
    assert "secret-token" not in "".join(
        traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb)
    )


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
    assert "secret-token" not in "".join(
        traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb)
    )


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
        self._payload = json.dumps(payload).encode("utf-8")
        self._offset = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, amt=-1):
        end = (
            len(self._payload)
            if amt < 0
            else min(len(self._payload), self._offset + amt)
        )
        chunk = self._payload[self._offset : end]
        self._offset = end
        return chunk


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


class _BinaryResponse:
    def __init__(
        self,
        payload,
        *,
        status=200,
        headers=None,
        oversized_reads=False,
    ):
        self._payload = payload
        self._offset = 0
        self._oversized_reads = oversized_reads
        self.status = status
        self.headers = {} if headers is None else dict(headers)
        self.read_limits = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, amt=-1):
        self.read_limits.append(amt)
        if self._oversized_reads:
            return self._payload
        end = (
            len(self._payload)
            if amt < 0
            else min(len(self._payload), self._offset + amt)
        )
        chunk = self._payload[self._offset : end]
        self._offset = end
        return chunk


class _BinaryOpener:
    def __init__(
        self,
        payload,
        *,
        status=200,
        headers=None,
        oversized_reads=False,
    ):
        self.response = _BinaryResponse(
            payload,
            status=status,
            headers=headers,
            oversized_reads=oversized_reads,
        )
        self.requests = []
        self.timeouts = []

    def __call__(self, request, *, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        return self.response


class _StreamingBinaryResponse:
    def __init__(self, payload, *, status=200, headers=None):
        self._payload = payload
        self._offset = 0
        self.status = status
        self.headers = (
            {"content-length": str(len(payload))} if headers is None else dict(headers)
        )
        self.read_limits = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, amt=-1):
        self.read_limits.append(amt)
        if amt < 0:
            end = len(self._payload)
        else:
            end = min(len(self._payload), self._offset + amt)
        chunk = self._payload[self._offset : end]
        self._offset = end
        return chunk


class _StreamingBinaryOpener:
    def __init__(self, payload, *, status=200, headers=None):
        self.response = _StreamingBinaryResponse(
            payload,
            status=status,
            headers=headers,
        )
        self.requests = []
        self.timeouts = []

    def __call__(self, request, *, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        return self.response


class _ConsumingUploadOpener:
    def __init__(
        self,
        *,
        response_payload=b"",
        response_status=204,
        response_headers=None,
        before_consume=None,
        after_chunk=None,
    ):
        self._response_payload = response_payload
        self._response_status = response_status
        self._response_headers = response_headers
        self._before_consume = before_consume
        self._after_chunk = after_chunk
        self.requests = []
        self.timeouts = []
        self.chunk_sizes = []
        self.total_bytes = 0
        self.sha256 = hashlib.sha256(b"").hexdigest()

    def __call__(self, request, *, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        if self._before_consume is not None:
            self._before_consume()
        digest = hashlib.sha256()
        for chunk_count, chunk in enumerate(request.data, start=1):
            assert type(chunk) is bytes
            self.chunk_sizes.append(len(chunk))
            self.total_bytes += len(chunk)
            digest.update(chunk)
            if self._after_chunk is not None:
                self._after_chunk(chunk_count)
        self.sha256 = digest.hexdigest()
        return _BinaryResponse(
            self._response_payload,
            status=self._response_status,
            headers=self._response_headers,
        )


class _GeneratedZeroStreamingResponse:
    def __init__(self, size_bytes):
        self._size_bytes = size_bytes
        self._offset = 0
        self.status = 200
        self.headers = {"content-length": str(size_bytes)}
        self.read_limits = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, amt=-1):
        self.read_limits.append(amt)
        remaining = self._size_bytes - self._offset
        read_size = remaining if amt < 0 else min(remaining, amt)
        self._offset += read_size
        return b"\0" * read_size


class _GeneratedZeroStreamingOpener:
    def __init__(self, size_bytes):
        self.response = _GeneratedZeroStreamingResponse(size_bytes)
        self.requests = []
        self.timeouts = []

    def __call__(self, request, *, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        return self.response


class _RedirectingHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, status_code, redirect_url="http://attacker.example/steal"):
        super().__init__()
        self._status_code = status_code
        self._redirect_url = redirect_url
        self.requests = []

    def http_open(self, request):
        self.requests.append(request)
        if len(self.requests) > 1:
            response = urllib.response.addinfourl(
                io.BytesIO(b'{"attacker":true}'),
                Message(),
                request.full_url,
                code=200,
            )
            response.msg = "OK"
            return response
        headers = Message()
        headers["Location"] = self._redirect_url
        response = urllib.response.addinfourl(
            io.BytesIO(b'{"redirect":true}'),
            headers,
            request.full_url,
            code=self._status_code,
        )
        response.msg = "Bearer secret-token redirect"
        return response


def _repeated_zero_sha256(size_bytes):
    digest = hashlib.sha256()
    chunk = b"\0" * (1024 * 1024)
    remaining = size_bytes
    while remaining:
        read_size = min(len(chunk), remaining)
        digest.update(chunk[:read_size])
        remaining -= read_size
    return digest.hexdigest()


class _SequentialBinaryOpener:
    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.requests = []
        self.timeouts = []

    def __call__(self, request, *, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        payload = self._payloads.pop(0)
        if not isinstance(payload, bytes):
            payload = json.dumps(payload).encode("utf-8")
        return _BinaryResponse(payload)


class _HTTPErrorOpener:
    def __init__(self, error):
        self._error = error

    def __call__(self, request, *, timeout):
        raise self._error


class _ExceptionOpener:
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
        self.read_limits = []

    def read(self, amt=-1):
        self.read_limits.append(amt)
        return self._payload

    def close(self):
        pass


def _pre_reserved_idempotency_case(tmp_path, *, attempt_id="idempotent-attempt"):
    ledger_path = tmp_path / "idempotency-ledger.json"
    opening = create_databricks_cluster_hour_ledger_json(
        ledger_path,
        ledger_id="idempotency-ledger",
    )
    payload = bind_databricks_run_idempotency_token(
        _bounded_submit_payload(),
        attempt_id=attempt_id,
    )
    _ledger, authorization = reserve_databricks_run_attempt_batch_authorized_json(
        ledger_path,
        (
            DatabricksRunAttemptReservationRequest(
                attempt_id=attempt_id,
                workload_id="idempotency-recovery-test",
                submit_payload=payload,
            ),
        ),
        expected_predecessor_prefix=databricks_ledger_prefix(opening),
    )
    config = DatabricksWorkspaceConfig(
        "https://dbc.example.cloud.databricks.com",
        "secret-token",
    )
    return config, ledger_path, payload, authorization


class _IdempotentSubmitService:
    def __init__(self, *, fail_mode):
        self.fail_mode = fail_mode
        self.calls = 0
        self.runs_by_token = {}

    def __call__(self, request, *, timeout):
        assert timeout > 0
        self.calls += 1
        payload = json.loads(request.data)
        token = payload["idempotency_token"]
        if self.calls == 1 and self.fail_mode == "accepted_timeout":
            self.runs_by_token[token] = 701
            raise TimeoutError("response was lost after acceptance")
        if self.calls == 1 and self.fail_mode == "preaccept_failure":
            raise ConnectionError("request failed before acceptance")
        run_id = self.runs_by_token.setdefault(token, 702)
        return _FakeResponse({"run_id": run_id})


def test_pre_reserved_recovery_reuses_accepted_idempotent_run(tmp_path):
    config, ledger_path, payload, authorization = _pre_reserved_idempotency_case(
        tmp_path
    )
    service = _IdempotentSubmitService(fail_mode="accepted_timeout")

    with pytest.raises(TimeoutError, match="lost after acceptance"):
        submit_pre_reserved_databricks_run(
            config,
            payload,
            ledger_path=ledger_path,
            attempt_id="idempotent-attempt",
            batch_authorization=authorization,
            opener=service,
        )
    recovered = recover_pre_reserved_databricks_run(
        config,
        payload,
        ledger_path=ledger_path,
        attempt_id="idempotent-attempt",
        batch_authorization=authorization,
        opener=service,
    )

    assert recovered == {"run_id": "701"}
    ledger = read_databricks_cluster_hour_ledger_json(ledger_path)
    assert [item.run_id for item in ledger.submission_receipts] == ["701"]
    assert service.calls == 2


def test_pre_reserved_recovery_after_preaccept_failure_creates_one_run(tmp_path):
    config, ledger_path, payload, authorization = _pre_reserved_idempotency_case(
        tmp_path
    )
    service = _IdempotentSubmitService(fail_mode="preaccept_failure")

    with pytest.raises(ConnectionError, match="before acceptance"):
        submit_pre_reserved_databricks_run(
            config,
            payload,
            ledger_path=ledger_path,
            attempt_id="idempotent-attempt",
            batch_authorization=authorization,
            opener=service,
        )
    assert service.runs_by_token == {}
    recovered = recover_pre_reserved_databricks_run(
        config,
        payload,
        ledger_path=ledger_path,
        attempt_id="idempotent-attempt",
        batch_authorization=authorization,
        opener=service,
    )

    assert recovered == {"run_id": "702"}
    assert len(service.runs_by_token) == 1


def test_pre_reserved_recovery_rejects_token_or_payload_drift_before_post(tmp_path):
    config, ledger_path, payload, authorization = _pre_reserved_idempotency_case(
        tmp_path
    )
    service = _IdempotentSubmitService(fail_mode="accepted_timeout")
    with pytest.raises(TimeoutError):
        submit_pre_reserved_databricks_run(
            config,
            payload,
            ledger_path=ledger_path,
            attempt_id="idempotent-attempt",
            batch_authorization=authorization,
            opener=service,
        )
    calls = service.calls
    missing_token = dict(payload)
    missing_token.pop("idempotency_token")
    with pytest.raises(ValueError, match="lacks idempotency_token"):
        recover_pre_reserved_databricks_run(
            config,
            missing_token,
            ledger_path=ledger_path,
            attempt_id="idempotent-attempt",
            batch_authorization=authorization,
            opener=service,
        )
    changed = dict(payload)
    changed["run_name"] = "changed-wire-bytes"
    with pytest.raises(ValueError, match="idempotency token drift"):
        recover_pre_reserved_databricks_run(
            config,
            changed,
            ledger_path=ledger_path,
            attempt_id="idempotent-attempt",
            batch_authorization=authorization,
            opener=service,
        )
    assert service.calls == calls


def test_concurrent_pre_reserved_recovery_posts_once_and_returns_one_run(tmp_path):
    config, ledger_path, payload, authorization = _pre_reserved_idempotency_case(
        tmp_path
    )
    service = _IdempotentSubmitService(fail_mode="accepted_timeout")
    with pytest.raises(TimeoutError):
        submit_pre_reserved_databricks_run(
            config,
            payload,
            ledger_path=ledger_path,
            attempt_id="idempotent-attempt",
            batch_authorization=authorization,
            opener=service,
        )

    def recover():
        return recover_pre_reserved_databricks_run(
            config,
            payload,
            ledger_path=ledger_path,
            attempt_id="idempotent-attempt",
            batch_authorization=authorization,
            opener=service,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: recover(), range(2)))

    assert results == [{"run_id": "701"}, {"run_id": "701"}]
    assert service.calls == 2
    assert len(read_databricks_cluster_hour_ledger_json(ledger_path).submission_receipts) == 1


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


def _bounded_submit_payload(*, run_name="representative-canary"):
    return {
        "run_name": run_name,
        "timeout_seconds": 14400,
        "tasks": [
            {
                "task_key": "representative-canary",
                "timeout_seconds": 14400,
                "max_retries": 0,
                "new_cluster": {
                    "spark_version": "15.4.x-gpu-ml-scala2.12",
                    "node_type_id": "g6.8xlarge",
                    "driver_node_type_id": "g6.8xlarge",
                    "num_workers": 0,
                },
                "spark_python_task": {
                    "python_file": "dbfs:/cachet/run.py",
                    "parameters": [],
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
