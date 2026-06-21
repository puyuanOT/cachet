# GitHub Metadata

This folder contains repository-level contribution automation and templates.

- `main-branch-protection.json` is the operator-ready GitHub branch protection
  payload for `main`. Apply it before release publication so direct pushes,
  force-pushes, and branch deletion are blocked, pull-request review is
  required, stale approvals are dismissed, conversations must be resolved, and
  the CI `Test and build` check must pass against an up-to-date branch.
- `pull_request_template.md` captures the required PR description, Refactor
  skill evidence, GPT-5.5 review evidence, and test or benchmark evidence for
  each logical development slice.
- `workflows/ci.yml` runs the automated pull-request quality gate and repeats
  the same safety check on pushes to `main`.

Apply the branch-protection payload with a token that has repository
administration rights:

```bash
curl -L -X PUT \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/OWNER/cachet/branches/main/protection \
  --data @.github/main-branch-protection.json
```

GitHub may require the repository to be public or the owner account to have a
plan that supports private-repository branch protection before this API accepts
the payload. Until that protection is active, direct pushes to `main` remain a
process violation even though GitHub cannot reject them automatically.
Because the payload enables `required_linear_history`, repository settings must
also keep squash or rebase merging enabled, enable GitHub auto-merge, and delete
head branches after merge before applying the payload.
