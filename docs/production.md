# Production

This page is for integrators moving from the local quickstart to real serving.
It intentionally avoids release-bundle and PR-evidence details; release
operators should use [`release-ops/README.md`](release-ops/README.md).

## Production Checklist

1. Choose the model and layout.
2. Generate real KV payloads for stable document chunks.
3. Store packed shards in a location visible to serving workers.
4. Store manifest rows in a durable table or service.
5. Materialize requests with `DocumentKVWorkflow`.
6. Hand the payload and metadata to vLLM, SGLang, or a custom adapter.
7. Compare against a no-cache baseline before treating speedups as real.

## vLLM

Cachet ships vLLM adapter modules in the same `cachet-kv` distribution. The
serving environment still installs and runs vLLM. The native path is:

```text
Cachet handoff -> vLLM KV transfer params -> Cachet vLLM provider -> vLLM paged KV blocks
```

Start with [`native-engine-integration.md`](native-engine-integration.md) when
you are wiring real vLLM block allocation and load behavior.

## SGLang

Cachet ships SGLang adapter modules for HiCache-style handoff metadata. The
serving environment still installs and runs SGLang. The native path is:

```text
Cachet handoff -> SGLang request metadata -> Cachet HiCache provider -> SGLang prefix binding
```

The current SGLang evidence validates live cache-hit correctness and quality.
Treat performance results separately from correctness until your prompt lengths,
cache-hit sizes, and runtime configuration match your workload.

## Databricks

Databricks is one supported production and benchmark environment, not a
requirement for using Cachet. Use it when you need managed GPU jobs, Unity
Catalog Volumes, or reproducible benchmark runs on the target hardware.

Databricks job templates live under [`../databricks/`](../databricks/). The
public quickstart does not require them.

## Stable User Commands

Most production users start with Python. The local example module is stable for
new users:

```bash
python -m cachet.quickstart_local
```

The user-facing CLI that is useful outside release operations is:

```bash
cachet-engine-launch-config --help
```

Maintainer-only commands for release bundles, PR evidence, repository hygiene,
GitHub governance, Databricks job generation, and benchmark publication are
documented in [`release-ops/README.md`](release-ops/README.md).

## Benchmarks

Use [`../benchmarks/current/README.md`](../benchmarks/current/) for the current
human-readable benchmark summary. Keep raw run output out of the source tree;
promote only compact, sanitized reports when they support a public claim.
