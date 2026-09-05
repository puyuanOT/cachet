# Repository Map

This repository is organized around Cachet as one project and one distribution
package, with the public Cachet facade, canonical implementation modules,
engine adapters, benchmarks, and release evidence in the same place. Use this
map to find the right entry point before opening generated or machine-readable
folders.

| Path | Audience | Purpose |
| --- | --- | --- |
| [`../README.md`](../README.md) | Users and integrators | Product overview, install, local quickstart, and next-step links |
| [`getting-started.md`](getting-started.md) | New users | Beginner path for the local example and first Cachet workflow |
| [`concepts.md`](concepts.md) | New users and integrators | Glossary for KV cache, materialization, handoff, chunks, manifests, vLLM, and SGLang |
| [`production.md`](production.md) | Integrators | Production path for vLLM, SGLang, and managed-cloud deployment |
| [`../src/cachet/`](../src/cachet/) | New Cachet users | Cachet-branded public import facade and `cachet-*` CLI entry points |
| [`../src/document_kv_cache/`](../src/document_kv_cache/) | Maintainers | Canonical implementation modules behind the Cachet facade |
| [`../src/vllm_kv_injection/`](../src/vllm_kv_injection/) | Serving maintainers | Vendored vLLM native KV-transfer adapter and probe support |
| [`../src/sglang_kv_injection/`](../src/sglang_kv_injection/) | Serving maintainers | Vendored SGLang HiCache adapter, metadata bridge, and probe support |
| [`../benchmarks/`](../benchmarks/) | Readers citing results | Human-readable benchmark reports plus compact sanitized JSON records |
| [`release-ops/evidence/`](release-ops/evidence/) | Release operators | Durable non-benchmark release-governance evidence, such as dependency freshness |
| [`release-ops/pr-evidence/`](release-ops/pr-evidence/) | PR reviewers and release auditors | Machine-checkable PR traceability sidecars |
| [`release-ops/`](release-ops/) | Release operators | Maintainer-only release, audit, governance, and historical reference material |
| [`../databricks/`](../databricks/) | Databricks operators | Job templates and bundle fragments used to stage managed runs |
| `../databricks-runs/` | Local operators only | Ignored scratch output for generated payloads, task status, and logs |
| [`../tests/`](../tests/) | Maintainers | Unit, contract, governance, and release-bundle validation tests |
| [`./`](./) | Maintainers and release operators | Architecture, migration, native integration, release readiness, and repository policy docs |

## Reading Paths

For a product overview, start with [`../README.md`](../README.md), then open
[`getting-started.md`](getting-started.md) and [`concepts.md`](concepts.md).
Use [`../benchmarks/README.md`](../benchmarks/README.md) only
when you want the current benchmark answer.

For serving-engine work, start with
[`native-engine-integration.md`](native-engine-integration.md), then inspect
the vLLM and SGLang adapter READMEs under `src/`.

For release readiness, start with [`release-ops/README.md`](release-ops/README.md),
then use [`v1-requirements-matrix.md`](v1-requirements-matrix.md),
[`../benchmarks/README.md`](../benchmarks/README.md),
[`release-ops/evidence/README.md`](release-ops/evidence/README.md), and
[`release-ops/pr-evidence/README.md`](release-ops/pr-evidence/README.md) to
follow each artifact family.

For repository hygiene, read [`evidence-policy.md`](evidence-policy.md) before
adding a new machine-readable artifact or a new output directory.
