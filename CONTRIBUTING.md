# Contributing

Document KV Cache is developed through small, reviewable pull requests. Direct
pushes to `main` are not part of the project workflow. Before release
publication, repository maintainers must apply `.github/main-branch-protection.json`
to the `main` branch so GitHub requires pull requests, one approving review,
resolved conversations, an up-to-date `Test and build` status check, linear
history, and blocks force-pushes or branch deletion. If GitHub rejects that
payload for a private repository, make the repository public or upgrade the
owner plan before treating the release workflow as enforced.

Each PR should include:

- what changed and why it is needed
- the affected package, storage, serving, benchmark, or documentation boundary
- test and benchmark evidence, or a clear note when a benchmark is not relevant
- confirmation that the Refactor skill was applied during the slice
- GPT-5.5 review findings and the follow-up fixes, or an explicit clean review

Use `document-kv-pr-evidence` to emit a JSON sidecar that captures the same
traceability fields and fails closed when the Refactor skill or GPT-5.5 review
gate is missing.

Keep PRs focused. A useful slice should move one architectural requirement
forward without mixing unrelated storage, serving, benchmark, and documentation
changes. When a change touches runtime cache layout, model metadata, storage
contracts, or public imports, include focused regression tests before broader
suite verification.

Serving changes must integrate with established engines such as vLLM or SGLang.
Do not add a proprietary request scheduler, decoder, or custom serving solver to
this package; keep engine-specific code at the handoff/adapter boundary.

Credentials must stay outside the repository. Use environment variables or the
target platform's secret store for Databricks tokens, OpenAI keys, private keys,
and service credentials. Do not paste secrets into benchmark records, release
evidence, PR evidence, notebooks, or generated Databricks runner scripts. Local
`.env*`, PEM/key files, and logs are ignored so developers can test locally
without making secret-bearing files easy to commit.

Generated artifacts such as wheels, caches, and local benchmark outputs should
not be committed. The source tree should remain reproducible from `pyproject.toml`
and the documented benchmark/job commands.
