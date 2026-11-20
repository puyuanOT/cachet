# Release And Governance

Maintainer reference for validating release evidence, assembling checksummed
release bundles, and capturing GitHub governance and repository hygiene
sidecars. The release sequence is in
[`../maintainer-release-checklist.md`](../maintainer-release-checklist.md), the
evidence storage rules are in
[`../../evidence-policy.md`](../../evidence-policy.md), and the V1 audit map is
[`docs/v1-requirements-matrix.md`](../../v1-requirements-matrix.md).

## Release Evidence

After the V1 benchmark, storage benchmark, and native vLLM/SGLang probe runs
complete, validate the collected artifacts. Add `--preflight-only` with
`--preflight-output-json` to check which inputs are present before strict
validation.

```bash
python -m document_kv_cache.release_evidence \
  --v1-benchmark-json v1-results.json \
  --storage-benchmark-json storage-benchmark.json \
  --engine-probe-json vllm-probe.json \
  --engine-probe-json sglang-probe.json \
  --engine-actions-json vllm-connector-actions.json \
  --engine-actions-json sglang-connector-actions.json \
  --output-json release-evidence.json
```

The command exits `0` only when all of the following hold:

- The V1 benchmark is a `document_kv.benchmark_run.v1` record for Qwen3 on
  supported AWS g6/L4 hardware, comparison rows have finite quality deltas and
  positive finite speedups, and successful rows and summaries carry positive
  prompt-token, completion-token, and throughput values, so zero-token or
  summary-only stubs fail. Baseline rows must use the logical prompt without
  KV transfer params; cache rows may use the logical or runtime prompt mode
  but must attach them.
- The storage benchmark has strict Memory + Disk + real UC Volume evidence.
- Exactly one native probe record and one connector-action descriptor exist
  per backend. Each probe reports the pinned runtime engine version and the
  serving-engine package metadata from the `document_kv_cache.serving_env`
  profile; each descriptor validates against the reserve/copy/bind/release
  schema and the one-byte Qwen3 GQA layout. The descriptor must describe the
  same request, block count, copied tokens, bytes, and segments, payload mode,
  and layout as the probe it audits.

The output lists `artifact_sources` with `size_bytes` and `sha256` for every
evaluated file. Release bundles re-run validation and require those
fingerprints to match the bundled payloads when present. The preflight sidecar
reports malformed JSON, missing paths, and wrong role or record-type
assignments separately (for example, a connector-actions record passed as
`--engine-probe-json` appears under `invalid_record_type_paths`).

## Release Bundle

Build the release bundle as its own command after plan execution:

```bash
python -m document_kv_cache.release_bundle \
  --v1-benchmark-json v1-results.json \
  --sglang-live-v1-benchmark-json sglang-live-benchmark.json \
  --compatibility-benchmark-json g5-v1-results.json \
  --storage-benchmark-json storage-benchmark.json \
  --engine-probe-json vllm-probe.json \
  --engine-probe-json sglang-probe.json \
  --engine-actions-json vllm-connector-actions.json \
  --engine-actions-json sglang-connector-actions.json \
  --engine-launch-config-json vllm-launch-config.json \
  --engine-launch-config-json sglang-launch-config.json \
  --release-evidence-json release-evidence.json \
  --preflight-json release-inputs.json \
  --plan-execution-json plan-execution.json \
  --databricks-run-status-json databricks-run-status-benchmark.json \
  --databricks-run-status-json databricks-run-status-storage.json \
  --databricks-run-status-json databricks-run-status-vllm-engine-probe.json \
  --databricks-run-status-json databricks-run-status-sglang-engine-probe.json \
  --compatibility-databricks-run-status-json databricks-run-status-g5-benchmark.json \
  --package-wheel dist/cachet_kv-0.2.0-py3-none-any.whl \
  --pr-evidence-json pr-evidence.json \
  --dependency-freshness-json docs/release-ops/evidence/dependency-freshness/current/dependency-freshness-evidence.json \
  --requirements-matrix-md docs/v1-requirements-matrix.md \
  --github-governance-json github-governance.json \
  --repository-hygiene-json repository-hygiene.json \
  --native-probe-factories-json native-probe-factories.json \
  --require-complete-v1 \
  --output-dir document-kv-release-bundle \
  --output-json release-bundle-manifest.json
```

The bundle holds byte-for-byte copies plus a `manifest.json` with the original
path, bundled path, record type, backend where applicable, package name/version
for wheel artifacts, size, and SHA-256 for every artifact.

- `--require-complete-v1` refuses to build unless release evidence, preflight,
  the SGLang live V1 benchmark, vLLM/SGLang probes, connector actions, and
  launch configs, plan execution, Databricks run-status sidecars, the tested
  wheel, PR evidence, dependency freshness, the V1 requirements matrix, GitHub
  governance, repository hygiene, and native-probe factory diagnostics are all
  present. It needs exactly one run-status sidecar for the benchmark and
  storage purposes plus one vLLM and one SGLang engine-probe status, split
  across backend runs or grouped in a matrix run, and the factory diagnostics
  must report both built-in factories as supported.
- `--compatibility-benchmark-json` carries non-default V1 evidence such as
  `aws-g5-a10g` runs, validated against the same storage, probe, and action
  sidecars; the g6/L4 release target is rejected in that role.
  `--compatibility-databricks-run-status-json` carries the matching successful
  status, must match a bundled compatibility benchmark's hardware target, and
  is required for each compatibility target in strict bundles.
- Run-status sidecars are compact `databricks_runs get --summary` outputs (or
  their inner `summary` records). They must be terminal, successful, free of
  active task keys, and attached to a hashed single-node AWS g6/L4
  `SINGLE_USER` submit-payload summary whose task summaries carry non-empty
  `purpose` tags and whose summary arrays match the task summaries. Raw Jobs
  API responses from `--include-response` are rejected.
- Release-evidence, preflight, plan-execution (including each command and the
  embedded plan-source record), and PR evidence sidecars are closed schemas.
- PR evidence must be a `document_kv.pr_evidence.v1` record with the pull
  request number and canonical owner/repo URL, Refactor-skill evidence, a
  completed GPT-5.5 review, and every GPT-5.5 finding resolved. With a
  governance sidecar bundled, each PR URL must point at the governed
  repository.
- The wheel must be a pure Python `py3-none-any` `cachet-kv` wheel whose
  filename, root `.dist-info` directory, and `METADATA` describe the same
  package identity, with a `RECORD` that lists the payload and metadata with
  matching hashes and sizes and no duplicate zip members. Its top-level entries
  must be limited to the `cachet`, `document_kv_cache`, `vllm_kv_injection`,
  and `sglang_kv_injection` packages and that `.dist-info` directory, so
  `.data/purelib` and `.data/platlib` package trees are rejected, and its
  console scripts must exactly match `[project.scripts]`. The manifest records
  the normalized package name and package version for the copied wheel. Strict
  bundles also require the wheel version to match the current project version
  from `pyproject.toml` or installed package metadata, and the wheel must
  carry `py.typed` for `cachet` and `document_kv_cache`.

## GitHub Governance

Capture repository governance as its own sidecar. The token comes from
`GITHUB_TOKEN`, or from `gh auth token` for local runs unless
`--no-gh-auth-token-fallback` is set; use an explicit `GITHUB_TOKEN` in CI and
Databricks jobs.

```bash
python -m document_kv_cache.github_governance \
  --repository OWNER/cachet \
  --output-json github-governance.json
```

It exits non-zero until the repository is public and named `cachet`, its
description (and any homepage URL) mentions Cachet, its topics include `cachet`
and `kv-cache`, the required `Test and build` protection is active and applies
to administrators, merge settings allow squash or rebase merges, auto-merge is
enabled, merged branches are deleted automatically, and no unexpected pull
requests are open. Inside the current release PR, pass
`--allow-open-pull-request-number <PR_NUMBER>` so that PR is recorded but not
treated as stale. Regenerate governance sidecars whenever the governance
schema changes; strict bundles reject sidecars without the merge-settings
block.

Apply the branch-protection payload from a maintainer shell with repository
administration rights:

```bash
curl -L -X PUT \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/OWNER/cachet/branches/main/protection \
  --data @.github/main-branch-protection.json
```

GitHub may require a public repository or a plan that supports
private-repository branch protection before it accepts the payload; until
protection is active, the sidecar records `ok=false` and direct pushes to
`main` remain a process violation even though GitHub cannot reject them.
Because the payload enables `required_linear_history`, repository settings
must keep squash or rebase merging enabled, enable GitHub auto-merge, and
delete head branches after merge before applying it.

## Repository Hygiene

```bash
python -m document_kv_cache.repository_hygiene \
  --repository-root . \
  --output-json repository-hygiene.json
```

The sidecar exits non-zero until every required `.gitignore` pattern is
present, no generated, build, cache, notebook-checkpoint, `databricks-runs/`,
or secret-like path is tracked or visible as untracked, every non-generated
directory has a `README.md` or package docstring, and no tracked file differs
from `HEAD`.
