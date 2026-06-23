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
            DEFAULT_GITHUB_REPOSITORY_ENV: "owner/cachet",
            DEFAULT_GITHUB_TOKEN_ENV: "secret-token",
        },
        timeout_seconds=9,
    )

    assert config.repository == "owner/cachet"
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
            "/repos/owner/cachet": _repository(private=False),
            "/repos/owner/cachet/branches/main/protection": _branch_protection(),
            "/repos/owner/cachet/pulls?state=open&per_page=100": [],
        }
    )
    config = GitHubRepositoryConfig("owner/cachet", "secret-token", timeout_seconds=11)

    summary = summarize_github_repository_governance(config, opener=opener)

    assert summary == {
        "record_type": GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE,
        "ok": True,
        "repository": "owner/cachet",
        "default_branch": "main",
        "branch": "main",
        "private": False,
        "visibility": "public",
        "archived": False,
        "disabled": False,
        "description": "Cachet: document KV-cache orchestration.",
        "homepage": "https://github.com/owner/cachet",
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
        "merge_settings": {
            "allow_squash_merge": True,
            "allow_rebase_merge": True,
            "allow_merge_commit": False,
            "allow_auto_merge": True,
            "delete_branch_on_merge": True,
        },
        "open_pull_requests": {
            "checked": True,
            "total_count": 0,
            "allowed_numbers": [],
            "allowed_count": 0,
            "allowed": [],
            "unexpected_count": 0,
            "unexpected": [],
            "truncated": False,
        },
        "issues": [],
    }
    assert [request.full_url for request in opener.requests] == [
        "https://api.github.com/repos/owner/cachet",
        "https://api.github.com/repos/owner/cachet/branches/main/protection",
        "https://api.github.com/repos/owner/cachet/pulls?state=open&per_page=100",
    ]
    assert opener.requests[0].headers["Authorization"] == "Bearer secret-token"
    assert opener.timeouts == [11, 11, 11]


def test_summarize_github_repository_governance_fails_closed_for_private_unprotected_repo():
    opener = _FakeGitHubOpener(
        {
            "/repos/owner/cachet": _repository(private=True),
            "/repos/owner/cachet/branches/main/protection": urllib.error.HTTPError(
                "https://api.github.com/repos/owner/cachet/branches/main/protection",
                403,
                "Forbidden",
                {},
                _BytesFile(b'{"message":"Upgrade to GitHub Pro or make this repository public"}'),
            ),
            "/repos/owner/cachet/pulls?state=open&per_page=100": [],
        }
    )
    config = GitHubRepositoryConfig("owner/cachet", "secret-token")

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
            "/repos/owner/cachet": repository,
            "/repos/owner/cachet/branches/main/protection": _branch_protection(),
            "/repos/owner/cachet/pulls?state=open&per_page=100": [],
        }
    )
    config = GitHubRepositoryConfig("owner/cachet", "secret-token")

    summary = summarize_github_repository_governance(config, opener=opener)

    assert summary["ok"] is False
    assert summary["private"] is False
    assert summary["visibility"] == "internal"
    assert summary["issues"] == [
        "repository visibility must be public before open-source release",
    ]


def test_summarize_github_repository_governance_requires_cachet_repository_branding():
    repository = _repository(private=False)
    repository["description"] = "Document KV-cache orchestration."
    repository["homepage"] = "https://github.com/owner/doc-kv"
    repository["topics"] = ["long-context"]
    opener = _FakeGitHubOpener(
        {
            "/repos/owner/cachet": repository,
            "/repos/owner/cachet/branches/main/protection": _branch_protection(),
            "/repos/owner/cachet/pulls?state=open&per_page=100": [],
        }
    )
    config = GitHubRepositoryConfig("owner/cachet", "secret-token")

    summary = summarize_github_repository_governance(config, opener=opener)

    assert summary["ok"] is False
    assert summary["description"] == "Document KV-cache orchestration."
    assert summary["topics"] == ["long-context"]
    assert summary["issues"] == [
        "repository description must mention Cachet before open-source release",
        "repository homepage must mention Cachet before open-source release",
        "repository topics must include: cachet, kv-cache",
    ]


def test_summarize_github_repository_governance_requires_cachet_repository_name():
    repository = _repository(private=False)
    repository["name"] = "doc-kv"
    repository["full_name"] = "owner/doc-kv"
    opener = _FakeGitHubOpener(
        {
            "/repos/owner/doc-kv": repository,
            "/repos/owner/doc-kv/branches/main/protection": _branch_protection(),
            "/repos/owner/doc-kv/pulls?state=open&per_page=100": [],
        }
    )
    config = GitHubRepositoryConfig("owner/doc-kv", "secret-token")

    summary = summarize_github_repository_governance(config, opener=opener)

    assert summary["ok"] is False
    assert summary["repository"] == "owner/doc-kv"
    assert summary["issues"] == [
        "repository name must be 'cachet' before open-source release",
    ]


def test_summarize_github_repository_governance_requires_admin_branch_protection():
    protection = _branch_protection()
    protection["enforce_admins"] = {"enabled": False}
    opener = _FakeGitHubOpener(
        {
            "/repos/owner/cachet": _repository(private=False),
            "/repos/owner/cachet/branches/main/protection": protection,
            "/repos/owner/cachet/pulls?state=open&per_page=100": [],
        }
    )
    config = GitHubRepositoryConfig("owner/cachet", "secret-token")

    summary = summarize_github_repository_governance(config, opener=opener)

    assert summary["ok"] is False
    assert summary["branch_protection"]["enforce_admins"] is False
    assert summary["issues"] == [
        "branch protection must apply to administrators",
    ]


def test_summarize_github_repository_governance_requires_pr_merge_hygiene():
    repository = _repository(private=False)
    repository["allow_squash_merge"] = False
    repository["allow_rebase_merge"] = False
    repository["delete_branch_on_merge"] = False
    opener = _FakeGitHubOpener(
        {
            "/repos/owner/cachet": repository,
            "/repos/owner/cachet/branches/main/protection": _branch_protection(),
            "/repos/owner/cachet/pulls?state=open&per_page=100": [],
        }
    )
    config = GitHubRepositoryConfig("owner/cachet", "secret-token")

    summary = summarize_github_repository_governance(config, opener=opener)

    assert summary["ok"] is False
    assert summary["merge_settings"] == {
        "allow_squash_merge": False,
        "allow_rebase_merge": False,
        "allow_merge_commit": False,
        "allow_auto_merge": True,
        "delete_branch_on_merge": False,
    }
    assert summary["issues"] == [
        "repository must allow squash or rebase merging",
        "repository must delete head branches after merge",
    ]


def test_summarize_github_repository_governance_requires_auto_merge():
    repository = _repository(private=False)
    repository["allow_auto_merge"] = False
    opener = _FakeGitHubOpener(
        {
            "/repos/owner/cachet": repository,
            "/repos/owner/cachet/branches/main/protection": _branch_protection(),
            "/repos/owner/cachet/pulls?state=open&per_page=100": [],
        }
    )
    config = GitHubRepositoryConfig("owner/cachet", "secret-token")

    summary = summarize_github_repository_governance(config, opener=opener)

    assert summary["ok"] is False
    assert summary["merge_settings"]["allow_auto_merge"] is False
    assert summary["issues"] == [
        "repository must enable GitHub auto-merge",
    ]


def test_summarize_github_repository_governance_rejects_unexpected_open_pull_requests():
    opener = _FakeGitHubOpener(
        {
            "/repos/owner/cachet": _repository(private=False),
            "/repos/owner/cachet/branches/main/protection": _branch_protection(),
            "/repos/owner/cachet/pulls?state=open&per_page=100": [
                _pull_request(72, title="Stale experiment branch", head_ref="experiment", draft=True),
                _pull_request(73, title="Ready release guard", head_ref="release-guard", draft=False),
            ],
        }
    )
    config = GitHubRepositoryConfig("owner/cachet", "secret-token")

    summary = summarize_github_repository_governance(config, opener=opener)

    assert summary["ok"] is False
    assert summary["open_pull_requests"] == {
        "checked": True,
        "total_count": 2,
        "allowed_numbers": [],
        "allowed_count": 0,
        "allowed": [],
        "unexpected_count": 2,
        "unexpected": [
            {
                "number": 72,
                "title": "Stale experiment branch",
                "draft": True,
                "html_url": "https://github.com/owner/cachet/pull/72",
                "head_ref": "experiment",
                "base_ref": "main",
            },
            {
                "number": 73,
                "title": "Ready release guard",
                "draft": False,
                "html_url": "https://github.com/owner/cachet/pull/73",
                "head_ref": "release-guard",
                "base_ref": "main",
            },
        ],
        "truncated": False,
    }
    assert summary["issues"] == [
        "repository must not have unexpected open pull requests: #72, #73",
    ]


def test_summarize_github_repository_governance_allows_current_open_pull_request():
    opener = _FakeGitHubOpener(
        {
            "/repos/owner/cachet": _repository(private=False),
            "/repos/owner/cachet/branches/main/protection": _branch_protection(),
            "/repos/owner/cachet/pulls?state=open&per_page=100": [
                _pull_request(73, title="Ready release guard", head_ref="release-guard", draft=False),
            ],
        }
    )
    config = GitHubRepositoryConfig("owner/cachet", "secret-token")

    summary = summarize_github_repository_governance(
        config,
        allowed_open_pull_request_numbers=(73,),
        opener=opener,
    )

    assert summary["ok"] is True
    assert summary["open_pull_requests"] == {
        "checked": True,
        "total_count": 1,
        "allowed_numbers": [73],
        "allowed_count": 1,
        "allowed": [
            {
                "number": 73,
                "title": "Ready release guard",
                "draft": False,
                "html_url": "https://github.com/owner/cachet/pull/73",
                "head_ref": "release-guard",
                "base_ref": "main",
            },
        ],
        "unexpected_count": 0,
        "unexpected": [],
        "truncated": False,
    }
    assert summary["issues"] == []


def test_github_http_errors_are_sanitized():
    opener = _FakeGitHubOpener(
        {
            "/repos/owner/cachet": urllib.error.HTTPError(
                "https://api.github.com/repos/owner/cachet",
                401,
                "Unauthorized",
                {},
                _BytesFile(b'{"message":"Authorization: Bearer secret-token; token=secret-token"}'),
            ),
        }
    )
    config = GitHubRepositoryConfig("owner/cachet", "secret-token")

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
        "document_kv_cache.github_governance._github_token_from_gh_auth",
        lambda *, timeout_seconds: (_ for _ in ()).throw(AssertionError("gh fallback should not run")),
    )
    captured_allowed_numbers = []
    captured_token = []

    def fake_summary(config, *, allowed_open_pull_request_numbers=()):
        captured_allowed_numbers.extend(allowed_open_pull_request_numbers)
        captured_token.append(config.token)
        return {
            "record_type": GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE,
            "ok": True,
            "repository": config.repository,
            "issues": [],
        }

    monkeypatch.setattr("document_kv_cache.github_governance.summarize_github_repository_governance", fake_summary)

    exit_code = main(
        [
            "--repository",
            "owner/cachet",
            "--allow-open-pull-request-number",
            "73",
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert captured_allowed_numbers == [73]
    assert captured_token == ["secret-token"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "ok": True,
        "summary": {
            "record_type": GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE,
            "ok": True,
            "repository": "owner/cachet",
            "issues": [],
        },
    }


def test_main_falls_back_to_gh_auth_token_when_token_env_is_missing(monkeypatch, tmp_path):
    output_path = tmp_path / "github-governance.json"
    monkeypatch.delenv(DEFAULT_GITHUB_TOKEN_ENV, raising=False)
    captured_token = []
    captured_timeout = []

    def fake_gh_auth_token(*, timeout_seconds):
        captured_timeout.append(timeout_seconds)
        return "gh-cli-token"

    def fake_summary(config, *, allowed_open_pull_request_numbers=()):
        captured_token.append(config.token)
        return {
            "record_type": GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE,
            "ok": True,
            "repository": config.repository,
            "issues": [],
        }

    monkeypatch.setattr("document_kv_cache.github_governance._github_token_from_gh_auth", fake_gh_auth_token)
    monkeypatch.setattr("document_kv_cache.github_governance.summarize_github_repository_governance", fake_summary)

    exit_code = main(
        [
            "--repository",
            "owner/cachet",
            "--gh-auth-token-timeout-seconds",
            "3.5",
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert captured_timeout == [3.5]
    assert captured_token == ["gh-cli-token"]
    assert json.loads(output_path.read_text(encoding="utf-8"))["summary"]["repository"] == (
        "owner/cachet"
    )


def test_main_can_disable_gh_auth_token_fallback(monkeypatch, tmp_path):
    output_path = tmp_path / "github-governance.json"
    monkeypatch.delenv(DEFAULT_GITHUB_TOKEN_ENV, raising=False)
    monkeypatch.setattr(
        "document_kv_cache.github_governance._github_token_from_gh_auth",
        lambda *, timeout_seconds: (_ for _ in ()).throw(AssertionError("gh fallback should not run")),
    )

    exit_code = main(
        [
            "--repository",
            "owner/cachet",
            "--no-gh-auth-token-fallback",
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 1
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["error_type"] == "ValueError"
    assert record["error"] == f"{DEFAULT_GITHUB_TOKEN_ENV} must be set"


def test_main_returns_nonzero_when_governance_summary_is_not_release_ready(monkeypatch, capsys):
    monkeypatch.setenv(DEFAULT_GITHUB_REPOSITORY_ENV, "owner/cachet")
    monkeypatch.setenv(DEFAULT_GITHUB_TOKEN_ENV, "secret-token")
    monkeypatch.setattr(
        "document_kv_cache.github_governance.summarize_github_repository_governance",
        lambda config, *, allowed_open_pull_request_numbers=(): {
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
        [
            sys.executable,
            "-m",
            "document_kv_cache.github_governance",
            "--repository",
            "owner/repo",
            "--no-gh-auth-token-fallback",
        ],
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
        "name": "cachet",
        "full_name": "owner/cachet",
        "default_branch": "main",
        "private": private,
        "visibility": "private" if private else "public",
        "archived": False,
        "disabled": False,
        "description": "Cachet: document KV-cache orchestration.",
        "homepage": "https://github.com/owner/cachet",
        "topics": ["kv-cache", "cachet"],
        "allow_squash_merge": True,
        "allow_rebase_merge": True,
        "allow_merge_commit": False,
        "allow_auto_merge": True,
        "delete_branch_on_merge": True,
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


def _pull_request(number: int, *, title: str, head_ref: str, draft: bool):
    return {
        "number": number,
        "title": title,
        "draft": draft,
        "html_url": f"https://github.com/owner/cachet/pull/{number}",
        "head": {"ref": head_ref},
        "base": {"ref": "main"},
    }
