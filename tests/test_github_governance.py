import json
import os
import subprocess
import sys
import urllib.error

import pytest

from document_kv_cache.github_governance import (
    DEFAULT_GITHUB_REPOSITORY_ENV,
    DEFAULT_GITHUB_TOKEN_ENV,
    GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE,
    GitHubRepositoryConfig,
    github_repository_config_from_env,
    main,
    summarize_github_repository_governance,
    write_github_repository_governance_json,
)


def test_github_repository_config_from_env_hides_token_in_repr():
    config = github_repository_config_from_env(
        environ={
            DEFAULT_GITHUB_REPOSITORY_ENV: "owner/document-kv-cache",
            DEFAULT_GITHUB_TOKEN_ENV: "secret-token",
        },
        timeout_seconds=9,
    )

    assert config.repository == "owner/document-kv-cache"
    assert config.normalized_api_base_url == "https://api.github.com"
    assert config.timeout_seconds == 9
    assert "secret-token" not in repr(config)


def test_github_repository_config_validates_repository_and_token():
    with pytest.raises(ValueError, match=DEFAULT_GITHUB_REPOSITORY_ENV):
        github_repository_config_from_env(environ={DEFAULT_GITHUB_TOKEN_ENV: "token"})

    with pytest.raises(ValueError, match=DEFAULT_GITHUB_TOKEN_ENV):
        github_repository_config_from_env(environ={DEFAULT_GITHUB_REPOSITORY_ENV: "owner/repo"})

    with pytest.raises(ValueError, match="OWNER/REPO"):
        GitHubRepositoryConfig("owner-only", "token")


def test_summarize_github_repository_governance_reports_release_ready_repo():
    opener = _FakeGitHubOpener(
        {
            "/repos/owner/document-kv-cache": _repository(private=False),
            "/repos/owner/document-kv-cache/branches/main/protection": _branch_protection(),
        }
    )
    config = GitHubRepositoryConfig("owner/document-kv-cache", "secret-token", timeout_seconds=11)

    summary = summarize_github_repository_governance(config, opener=opener)

    assert summary == {
        "record_type": GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE,
        "ok": True,
        "repository": "owner/document-kv-cache",
        "default_branch": "main",
        "branch": "main",
        "private": False,
        "visibility": "public",
        "archived": False,
        "disabled": False,
        "description": "Cachet: document KV-cache orchestration.",
        "homepage": "https://github.com/owner/document-kv-cache",
        "topics": ["cachet", "kv-cache"],
        "branch_protection": {
            "enabled": True,
            "required_status_checks": {
                "strict": True,
                "contexts": ["Test and build"],
            },
            "required_pull_request_reviews": {
                "dismiss_stale_reviews": True,
                "require_last_push_approval": True,
                "required_approving_review_count": 1,
            },
            "required_linear_history": True,
            "required_conversation_resolution": True,
            "enforce_admins": True,
            "allow_force_pushes": False,
            "allow_deletions": False,
        },
        "issues": [],
    }
    assert [request.full_url for request in opener.requests] == [
        "https://api.github.com/repos/owner/document-kv-cache",
        "https://api.github.com/repos/owner/document-kv-cache/branches/main/protection",
    ]
    assert opener.requests[0].headers["Authorization"] == "Bearer secret-token"
    assert opener.timeouts == [11, 11]


def test_summarize_github_repository_governance_fails_closed_for_private_unprotected_repo():
    opener = _FakeGitHubOpener(
        {
            "/repos/owner/document-kv-cache": _repository(private=True),
            "/repos/owner/document-kv-cache/branches/main/protection": urllib.error.HTTPError(
                "https://api.github.com/repos/owner/document-kv-cache/branches/main/protection",
                403,
                "Forbidden",
                {},
                _BytesFile(b'{"message":"Upgrade to GitHub Pro or make this repository public"}'),
            ),
        }
    )
    config = GitHubRepositoryConfig("owner/document-kv-cache", "secret-token")

    summary = summarize_github_repository_governance(config, opener=opener)

    assert summary["ok"] is False
    assert summary["private"] is True
    assert summary["branch_protection"]["enabled"] is False
    assert summary["branch_protection"]["error_status_code"] == 403
    assert summary["issues"] == [
        "repository must be public before open-source release",
        "repository visibility must be public before open-source release",
        "main branch protection must be enabled",
    ]


def test_summarize_github_repository_governance_requires_public_visibility():
    repository = _repository(private=False)
    repository["visibility"] = "internal"
    opener = _FakeGitHubOpener(
        {
            "/repos/owner/document-kv-cache": repository,
            "/repos/owner/document-kv-cache/branches/main/protection": _branch_protection(),
        }
    )
    config = GitHubRepositoryConfig("owner/document-kv-cache", "secret-token")

    summary = summarize_github_repository_governance(config, opener=opener)

    assert summary["ok"] is False
    assert summary["private"] is False
    assert summary["visibility"] == "internal"
    assert summary["issues"] == [
        "repository visibility must be public before open-source release",
    ]


def test_github_http_errors_are_sanitized():
    opener = _FakeGitHubOpener(
        {
            "/repos/owner/document-kv-cache": urllib.error.HTTPError(
                "https://api.github.com/repos/owner/document-kv-cache",
                401,
                "Unauthorized",
                {},
                _BytesFile(b'{"message":"Authorization: Bearer secret-token; token=secret-token"}'),
            ),
        }
    )
    config = GitHubRepositoryConfig("owner/document-kv-cache", "secret-token")

    with pytest.raises(RuntimeError) as excinfo:
        summarize_github_repository_governance(config, opener=opener)

    error = str(excinfo.value)
    assert "HTTP 401" in error
    assert "secret-token" not in error
    assert "Bearer [REDACTED]" in error
    assert "token=[REDACTED]" in error


def test_main_writes_release_readiness_summary(monkeypatch, tmp_path):
    output_path = tmp_path / "github-governance.json"
    monkeypatch.setenv(DEFAULT_GITHUB_TOKEN_ENV, "secret-token")
    monkeypatch.setattr(
        "document_kv_cache.github_governance.summarize_github_repository_governance",
        lambda config: {
            "record_type": GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE,
            "ok": True,
            "repository": config.repository,
            "issues": [],
        },
    )

    exit_code = main(
        [
            "--repository",
            "owner/document-kv-cache",
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "ok": True,
        "summary": {
            "record_type": GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE,
            "ok": True,
            "repository": "owner/document-kv-cache",
            "issues": [],
        },
    }


def test_main_returns_nonzero_when_governance_summary_is_not_release_ready(monkeypatch, capsys):
    monkeypatch.setenv(DEFAULT_GITHUB_REPOSITORY_ENV, "owner/document-kv-cache")
    monkeypatch.setenv(DEFAULT_GITHUB_TOKEN_ENV, "secret-token")
    monkeypatch.setattr(
        "document_kv_cache.github_governance.summarize_github_repository_governance",
        lambda config: {
            "record_type": GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE,
            "ok": False,
            "repository": config.repository,
            "issues": ["main branch protection must be enabled"],
        },
    )

    exit_code = main([])

    assert exit_code == 2
    assert json.loads(capsys.readouterr().out)["summary"]["issues"] == [
        "main branch protection must be enabled",
    ]


def test_write_github_repository_governance_json(tmp_path):
    output_path = tmp_path / "nested" / "github-governance.json"
    output_path.parent.mkdir()

    write_github_repository_governance_json({"ok": True}, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"ok": True}


def test_github_governance_module_executes_with_python_m():
    pythonpath = os.pathsep.join(["src", os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-m", "document_kv_cache.github_governance", "--repository", "owner/repo"],
        env={**os.environ, "PYTHONPATH": pythonpath, DEFAULT_GITHUB_TOKEN_ENV: ""},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert json.loads(completed.stdout)["error_type"] == "ValueError"


class _FakeGitHubOpener:
    def __init__(self, responses):
        self.responses = responses
        self.requests = []
        self.timeouts = []

    def __call__(self, request, *, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        path = request.full_url.removeprefix("https://api.github.com")
        response = self.responses[path]
        if isinstance(response, urllib.error.HTTPError):
            raise response
        return _FakeResponse(response)


class _FakeResponse:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class _BytesFile:
    def __init__(self, payload: bytes):
        self.payload = payload

    def read(self):
        return self.payload

    def close(self):
        pass


def _repository(*, private: bool):
    return {
        "full_name": "owner/document-kv-cache",
        "default_branch": "main",
        "private": private,
        "visibility": "private" if private else "public",
        "archived": False,
        "disabled": False,
        "description": "Cachet: document KV-cache orchestration.",
        "homepage": "https://github.com/owner/document-kv-cache",
        "topics": ["kv-cache", "cachet"],
    }


def _branch_protection():
    return {
        "required_status_checks": {
            "strict": True,
            "contexts": ["Test and build"],
        },
        "required_pull_request_reviews": {
            "dismiss_stale_reviews": True,
            "require_last_push_approval": True,
            "required_approving_review_count": 1,
        },
        "required_linear_history": {"enabled": True},
        "required_conversation_resolution": {"enabled": True},
        "enforce_admins": {"enabled": True},
        "allow_force_pushes": {"enabled": False},
        "allow_deletions": {"enabled": False},
    }
