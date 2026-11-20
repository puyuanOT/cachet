# Maintainer Reference

Cachet turns long, repeated document context into reusable KV-cache payloads
that vLLM and SGLang load through their native connectors. This reference is
for maintainers and release operators; contributors start with
[`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`AGENTS.md`](../../AGENTS.md).

| Topic | Scope |
| --- | --- |
| [Core library](maintainer/core-library.md) | Manifest and storage model, readers, tiered cache, workflow API, model profiles |
| [Serving engines](maintainer/serving-engines.md) | Engine handoff records, connector actions, native probes, launch configs, live and serving smokes |
| [Benchmarks](maintainer/benchmarks.md) | V1 benchmark contract, handoff bundles, command plans, dataset prep, runner, storage-reader benchmark |
| [Databricks](maintainer/databricks.md) | Single-node GPU jobs, engine-probe targets, staging, submission, run status |
| [Release and governance](maintainer/release-and-governance.md) | Release evidence, PR evidence, release bundles, GitHub governance, repository hygiene |
| [Development](maintainer/development.md) | Dependency pins, isolated runtimes, dependency freshness, CI, repository rules, license |

[`docs/v1-requirements-matrix.md`](../v1-requirements-matrix.md) is the audit
map for V1 requirements and remaining gates. The vLLM 0.30.0 campaign
machinery (resource ledger, GPU qualification, latency, full-score, and
finalizer stages, source freeze, LMCache and hybrid launchers, representative
canaries, the Quality v2 job runner, the controller semantic lock, and the full
offline validators) is preserved at the `vllm-0300-campaign-archive` tag.

## Remaining V1 Work

- Build a new strict release bundle exclusively from source-bound artifacts,
  Databricks attestations, results, and a passing publication gate. Native
  engine block managers remain owned by vLLM and SGLang rather than Cachet.
- Keep GitHub governance release-ready before each public release. Current
  governance evidence is green: the repository is public, `allow_auto_merge`
  reports enabled, merged head branches are deleted automatically, `main`
  branch protection requires the `Test and build` check, branch protection
  applies to administrators, force-pushes/deletions are blocked, and no
  unexpected pull requests are open.
- Continue running connector action descriptors validation against native
  engine block managers in vLLM and SGLang whenever adapter contracts, runtime
  pins, or launch configs change.
- Keep serving integrations inside established engines; do not add a proprietary scheduler or custom solver.
