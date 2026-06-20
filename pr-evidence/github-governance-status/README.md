# GitHub Governance Status PR Evidence

This slice adds a report-only GitHub governance helper for release readiness.
It records repository visibility, `main` branch protection, required CI status
checks, pull-request review settings, and any fail-closed release issues without
mutating GitHub settings or persisting credentials.

Verification:

- `poetry run pytest tests/test_github_governance.py tests/test_public_package.py tests/test_project_governance.py -q`
- `poetry run pytest`
- `poetry check`
- `git diff --check`
- `poetry build`
- repository secret-pattern scan, with only existing fake LangSmith detector
  fixtures matching

GPT-5.5/high review found one P1 fail-open case where GitHub `visibility:
internal` could pass. The helper now also requires `visibility == "public"`,
the regression test covers that case, and the re-review approved.
