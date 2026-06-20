"""Inspect GitHub repository governance settings for release readiness."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any, Protocol
import urllib.error
import urllib.parse
import urllib.request


__all__ = [
    "DEFAULT_GITHUB_API_BASE_URL",
    "DEFAULT_GITHUB_BRANCH",
    "DEFAULT_GITHUB_REPOSITORY_ENV",
    "DEFAULT_GITHUB_TIMEOUT_SECONDS",
    "DEFAULT_GITHUB_TOKEN_ENV",
    "GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE",
    "GitHubRepositoryConfig",
    "github_repository_config_from_env",
    "summarize_github_repository_governance",
    "write_github_repository_governance_json",
    "main",
]

DEFAULT_GITHUB_API_BASE_URL = "https://api.github.com"
DEFAULT_GITHUB_BRANCH = "main"
DEFAULT_GITHUB_REPOSITORY_ENV = "GITHUB_REPOSITORY"
DEFAULT_GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
DEFAULT_GITHUB_TIMEOUT_SECONDS = 60.0
GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE = "document_kv.github_repository_governance.v1"
REQUIRED_CI_STATUS_CHECK = "Test and build"


class GitHubHTTPResponse(Protocol):
    status: int

    def __enter__(self) -> "GitHubHTTPResponse": ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool: ...

    def read(self) -> bytes: ...


class GitHubURLOpener(Protocol):
    def __call__(self, request: urllib.request.Request, *, timeout: float) -> GitHubHTTPResponse: ...


@dataclass(frozen=True, slots=True)
class GitHubRepositoryConfig:
    repository: str
    token: str = field(repr=False)
    branch: str = DEFAULT_GITHUB_BRANCH
    api_base_url: str = DEFAULT_GITHUB_API_BASE_URL
    timeout_seconds: float = DEFAULT_GITHUB_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not _valid_repository(self.repository):
            raise ValueError("repository must be in OWNER/REPO form")
        if not self.token:
            raise ValueError("token must be non-empty")
        if not self.branch:
            raise ValueError("branch must be non-empty")
        if not self.api_base_url:
            raise ValueError("api_base_url must be non-empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    @property
    def normalized_api_base_url(self) -> str:
        return self.api_base_url.rstrip("/")


def github_repository_config_from_env(
    *,
    repository_env: str = DEFAULT_GITHUB_REPOSITORY_ENV,
    token_env: str = DEFAULT_GITHUB_TOKEN_ENV,
    branch: str = DEFAULT_GITHUB_BRANCH,
    api_base_url: str = DEFAULT_GITHUB_API_BASE_URL,
    timeout_seconds: float = DEFAULT_GITHUB_TIMEOUT_SECONDS,
    environ: Mapping[str, str] | None = None,
) -> GitHubRepositoryConfig:
    env = os.environ if environ is None else environ
    repository = env.get(repository_env, "")
    token = env.get(token_env, "")
    if not repository:
        raise ValueError(f"{repository_env} must be set")
    if not token:
        raise ValueError(f"{token_env} must be set")
    return GitHubRepositoryConfig(
        repository=repository,
        token=token,
        branch=branch,
        api_base_url=api_base_url,
        timeout_seconds=timeout_seconds,
    )


def summarize_github_repository_governance(
    config: GitHubRepositoryConfig,
    *,
    allowed_open_pull_request_numbers: Sequence[int] = (),
    opener: GitHubURLOpener = urllib.request.urlopen,
) -> dict[str, Any]:
    repository = _github_api_json(config, f"/repos/{_quote_path(config.repository)}", opener=opener)
    protection = _github_branch_protection_summary(config, opener=opener)
    open_pull_requests = _github_open_pull_requests_summary(
        config,
        allowed_open_pull_request_numbers=allowed_open_pull_request_numbers,
        opener=opener,
    )
    issues = _release_readiness_issues(repository, protection, open_pull_requests)
    return {
        "record_type": GITHUB_REPOSITORY_GOVERNANCE_RECORD_TYPE,
        "ok": not issues,
        "repository": repository.get("full_name") or config.repository,
        "default_branch": repository.get("default_branch"),
        "branch": config.branch,
        "private": repository.get("private"),
        "visibility": repository.get("visibility"),
        "archived": repository.get("archived"),
        "disabled": repository.get("disabled"),
        "description": repository.get("description"),
        "homepage": repository.get("homepage"),
        "topics": _sorted_texts(repository.get("topics")),
        "branch_protection": protection,
        "open_pull_requests": open_pull_requests,
        "issues": issues,
    }


def write_github_repository_governance_json(record: Mapping[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _github_branch_protection_summary(
    config: GitHubRepositoryConfig,
    *,
    opener: GitHubURLOpener,
) -> dict[str, Any]:
    path = f"/repos/{_quote_path(config.repository)}/branches/{urllib.parse.quote(config.branch, safe='')}/protection"
    try:
        protection = _github_api_json(config, path, opener=opener)
    except RuntimeError as exc:
        status_code = getattr(exc, "status_code", None)
        return {
            "enabled": False,
            "error": str(exc),
            "error_status_code": status_code,
        }
    required_status_checks = _mapping(protection.get("required_status_checks"))
    required_pull_request_reviews = _mapping(protection.get("required_pull_request_reviews"))
    return {
        "enabled": True,
        "required_status_checks": {
            "strict": required_status_checks.get("strict"),
            "contexts": _sorted_texts(required_status_checks.get("contexts")),
        },
        "required_pull_request_reviews": {
            "dismiss_stale_reviews": required_pull_request_reviews.get("dismiss_stale_reviews"),
            "require_last_push_approval": required_pull_request_reviews.get("require_last_push_approval"),
            "required_approving_review_count": required_pull_request_reviews.get("required_approving_review_count"),
        },
        "required_linear_history": _enabled_flag(protection.get("required_linear_history")),
        "required_conversation_resolution": _enabled_flag(protection.get("required_conversation_resolution")),
        "enforce_admins": _enabled_flag(protection.get("enforce_admins")),
        "allow_force_pushes": _enabled_flag(protection.get("allow_force_pushes")),
        "allow_deletions": _enabled_flag(protection.get("allow_deletions")),
    }


def _github_open_pull_requests_summary(
    config: GitHubRepositoryConfig,
    *,
    allowed_open_pull_request_numbers: Sequence[int],
    opener: GitHubURLOpener,
) -> dict[str, Any]:
    allowed_numbers = _positive_pull_request_numbers(allowed_open_pull_request_numbers)
    path = f"/repos/{_quote_path(config.repository)}/pulls?state=open&per_page=100"
    try:
        open_pull_requests = _github_api_json_array(config, path, opener=opener)
    except RuntimeError as exc:
        status_code = getattr(exc, "status_code", None)
        return {
            "checked": False,
            "error": str(exc),
            "error_status_code": status_code,
            "allowed_numbers": allowed_numbers,
        }
    total_count = len(open_pull_requests)
    unexpected = [
        _pull_request_summary(pull_request)
        for pull_request in open_pull_requests
        if _pull_request_number(pull_request) not in allowed_numbers
    ]
    return {
        "checked": True,
        "total_count": total_count,
        "allowed_numbers": allowed_numbers,
        "unexpected_count": len(unexpected),
        "unexpected": unexpected,
        "truncated": total_count >= 100,
    }


def _github_api_json(
    config: GitHubRepositoryConfig,
    path: str,
    *,
    opener: GitHubURLOpener,
) -> dict[str, Any]:
    parsed = _github_api_json_value(config, path, opener=opener)
    if not isinstance(parsed, dict):
        raise RuntimeError("GitHub response JSON must be an object")
    return parsed


def _github_api_json_array(
    config: GitHubRepositoryConfig,
    path: str,
    *,
    opener: GitHubURLOpener,
) -> list[Any]:
    parsed = _github_api_json_value(config, path, opener=opener)
    if not isinstance(parsed, list):
        raise RuntimeError("GitHub response JSON must be an array")
    return parsed


def _github_api_json_value(
    config: GitHubRepositoryConfig,
    path: str,
    *,
    opener: GitHubURLOpener,
) -> Any:
    request = urllib.request.Request(
        f"{config.normalized_api_base_url}{path}",
        method="GET",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {config.token}",
        },
    )
    try:
        with opener(request, timeout=config.timeout_seconds) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        error = RuntimeError(_format_github_http_error(exc.code, body, token=config.token))
        setattr(error, "status_code", exc.code)
        raise error from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GitHub request failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("GitHub response was not valid JSON") from exc
    return parsed


def _release_readiness_issues(
    repository: Mapping[str, Any],
    protection: Mapping[str, Any],
    open_pull_requests: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []
    if repository.get("private") is True:
        issues.append("repository must be public before open-source release")
    if repository.get("visibility") != "public":
        issues.append("repository visibility must be public before open-source release")
    if repository.get("archived") is True:
        issues.append("repository must not be archived")
    if repository.get("disabled") is True:
        issues.append("repository must not be disabled")
    if protection.get("enabled") is not True:
        issues.append("main branch protection must be enabled")
        return issues
    status_checks = _mapping(protection.get("required_status_checks"))
    pull_request_reviews = _mapping(protection.get("required_pull_request_reviews"))
    if status_checks.get("strict") is not True:
        issues.append("branch protection must require an up-to-date status check")
    if REQUIRED_CI_STATUS_CHECK not in _sorted_texts(status_checks.get("contexts")):
        issues.append(f"branch protection must require the {REQUIRED_CI_STATUS_CHECK!r} status check")
    if pull_request_reviews.get("dismiss_stale_reviews") is not True:
        issues.append("branch protection must dismiss stale pull-request approvals")
    if pull_request_reviews.get("require_last_push_approval") is not True:
        issues.append("branch protection must require approval after the last push")
    if pull_request_reviews.get("required_approving_review_count") != 1:
        issues.append("branch protection must require exactly one approving review")
    if protection.get("required_linear_history") is not True:
        issues.append("branch protection must require linear history")
    if protection.get("required_conversation_resolution") is not True:
        issues.append("branch protection must require conversation resolution")
    if protection.get("enforce_admins") is not True:
        issues.append("branch protection must apply to administrators")
    if protection.get("allow_force_pushes") is not False:
        issues.append("branch protection must block force-pushes")
    if protection.get("allow_deletions") is not False:
        issues.append("branch protection must block branch deletion")
    if open_pull_requests.get("checked") is not True:
        issues.append("open pull request inspection must succeed")
    elif open_pull_requests.get("unexpected_count") != 0:
        issues.append(
            "repository must not have unexpected open pull requests: "
            + ", ".join(
                f"#{number}"
                for number in _pull_request_numbers_from_summary(open_pull_requests.get("unexpected"))
            )
        )
    if open_pull_requests.get("truncated") is True:
        issues.append("open pull request inspection must not be truncated")
    return issues


def _format_github_http_error(status_code: int, body: str, *, token: str | None = None) -> str:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = {}
    if isinstance(parsed, Mapping):
        message = parsed.get("message") or body
    else:
        message = body
    return f"GitHub request failed with HTTP {status_code}: {_redact_secret_text(str(message), token=token)}"


def _redact_secret_text(text: str, *, token: str | None = None) -> str:
    redacted = text.replace(token, "[REDACTED]") if token else text
    return re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/\-=]+", r"\1[REDACTED]", redacted)


def _valid_repository(value: str) -> bool:
    parts = value.split("/")
    return len(parts) == 2 and all(part for part in parts)


def _positive_pull_request_numbers(values: Sequence[int]) -> list[int]:
    return sorted(set(number for number in values if isinstance(number, int) and number > 0))


def _pull_request_summary(value: Any) -> dict[str, Any]:
    pull_request = _mapping(value)
    head = _mapping(pull_request.get("head"))
    base = _mapping(pull_request.get("base"))
    return {
        "number": _pull_request_number(pull_request),
        "title": pull_request.get("title"),
        "draft": pull_request.get("draft"),
        "html_url": pull_request.get("html_url"),
        "head_ref": head.get("ref"),
        "base_ref": base.get("ref"),
    }


def _pull_request_number(value: Mapping[str, Any]) -> int | None:
    number = value.get("number")
    return number if isinstance(number, int) and number > 0 else None


def _pull_request_numbers_from_summary(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return sorted(number for item in value if (number := _pull_request_number(_mapping(item))) is not None)


def _quote_path(value: str) -> str:
    return "/".join(urllib.parse.quote(part, safe="") for part in value.split("/"))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _enabled_flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("enabled"), bool):
        return value["enabled"]
    return None


def _sorted_texts(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return sorted(item for item in value if isinstance(item, str) and item)


def _success_record(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {"ok": bool(summary.get("ok")), "summary": dict(summary)}


def _write_or_print(record: Mapping[str, Any], output_json: str | None) -> None:
    if output_json:
        write_github_repository_governance_json(record, output_json)
    else:
        print(json.dumps(record, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect GitHub repository governance readiness.")
    parser.add_argument("--repository", help="Repository in OWNER/REPO form. Defaults to $GITHUB_REPOSITORY.")
    parser.add_argument("--repository-env", default=DEFAULT_GITHUB_REPOSITORY_ENV)
    parser.add_argument("--token-env", default=DEFAULT_GITHUB_TOKEN_ENV)
    parser.add_argument("--branch", default=DEFAULT_GITHUB_BRANCH)
    parser.add_argument("--api-base-url", default=DEFAULT_GITHUB_API_BASE_URL)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_GITHUB_TIMEOUT_SECONDS)
    parser.add_argument(
        "--allow-open-pull-request-number",
        action="append",
        type=int,
        default=[],
        help=(
            "Allow this open pull-request number when deciding release readiness. "
            "Repeat for a release sidecar produced before the current PR is merged."
        ),
    )
    parser.add_argument("--output-json", help="Write the governance status JSON to this path instead of stdout.")
    args = parser.parse_args(argv)

    try:
        if args.repository:
            token = os.environ.get(args.token_env, "")
            if not token:
                raise ValueError(f"{args.token_env} must be set")
            config = GitHubRepositoryConfig(
                repository=args.repository,
                token=token,
                branch=args.branch,
                api_base_url=args.api_base_url,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            config = github_repository_config_from_env(
                repository_env=args.repository_env,
                token_env=args.token_env,
                branch=args.branch,
                api_base_url=args.api_base_url,
                timeout_seconds=args.timeout_seconds,
            )
        summary = summarize_github_repository_governance(
            config,
            allowed_open_pull_request_numbers=args.allow_open_pull_request_number,
        )
        record = _success_record(summary)
        _write_or_print(record, args.output_json)
    except Exception as exc:
        record = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        _write_or_print(record, args.output_json)
        return 1
    return 0 if record["ok"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
