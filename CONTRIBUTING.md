# Contributing

Thanks for helping improve Cachet. You do not need Databricks, GPUs, internal
release evidence, or maintainer-only review tooling to open useful issues or
pull requests.

## Good First Contributions

Good first contributions include:

- documentation fixes;
- runnable examples;
- clearer error messages;
- small validation tests;
- issue reproduction cases;
- improvements to the local quickstart.

See [`ROADMAP.md`](ROADMAP.md) for current project direction.

## Open An Issue

Use the GitHub issue templates for bugs, feature requests, or good-first-issue
ideas. Please include:

- what you tried;
- what happened;
- what you expected;
- your Cachet and Python versions;
- the smallest sanitized example that reproduces the issue.

Do not paste credentials, tokens, private data, raw service responses, or
customer documents into issues.

## Open A Pull Request

1. Fork the repository and create a branch.
2. Keep the change focused.
3. Add or update tests when behavior changes.
4. Run the narrowest useful tests locally.
5. Explain what changed and why in the PR description.

Useful local checks:

```bash
python -m cachet.quickstart_local
python examples/quickstart_local.py
poetry run pytest tests/test_project_governance.py -q
poetry run pytest -q
poetry check --lock
```

If your change touches only docs, examples, or packaging, say so in the PR and
list the checks you ran.

## Serving Changes

Serving changes should integrate with established engines such as vLLM or
SGLang. Cachet should stay at the document KV preparation and handoff boundary;
it should not grow its own scheduler, decoder, or custom serving engine.

## Secrets And Generated Files

Keep credentials outside the repository. Use environment variables or the
target platform's secret store for Databricks tokens, OpenAI keys, private keys,
and service credentials.

Do not commit:

- `.env` files;
- PEM/key files;
- logs;
- generated wheels;
- raw benchmark outputs;
- raw service responses;
- local Databricks run payloads.

Local scratch output belongs under ignored directories such as
`databricks-runs/`.

## Maintainer-Only Release Gates

Maintainers run additional release checks, including PR traceability sidecars,
repository hygiene, GitHub governance, release-bundle validation, and benchmark
publication. Those gates are documented under
[`docs/release-ops/`](docs/release-ops/README.md). External contributors do not
need to produce those artifacts before opening a PR.
