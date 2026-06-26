# Cachet Benchmarks

This directory is the public benchmark appendix for Cachet. It is table-first:
the current page starts with the intended main result table, then shows focused
ablation tables and links to existing evidence.

Start with [`current/`](current/). It is the paper-style benchmark page for
speed, quality, resource, and coverage questions.

## Directory Layout

| Folder | Purpose |
| --- | --- |
| [`current/`](current/) | Canonical paper-style benchmark appendix with one main performance table and ablation tables |
| [`appendix/existing-results/`](appendix/existing-results/) | Committed prior/current evidence that does not match the fixed main-table configuration |
| [`databricks/`](databricks/) | Sanitized Databricks audit mirrors kept at stable paths for release evidence |
| [`_template/`](_template/) | Required table shape for future public benchmark result folders |
| [`vllm/`](vllm/) | Short redirect/index page for vLLM appendix evidence |
| [`sglang/`](sglang/) | Short redirect/index page for SGLang appendix evidence |
| [`storage/`](storage/) | Short redirect/index page for storage-reader appendix evidence |
| [`native-engine/`](native-engine/) | Short redirect/index page for native connector appendix evidence |

## Main Table Contract

The main benchmark table in [`current/`](current/) is intentionally fixed to
one comparable configuration:

| Field | Fixed value |
| --- | --- |
| Model | Qwen3-4B-Instruct |
| Serving engine | vLLM |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Request parallelism | 8 requests in flight |
| Output length | Emit 256 tokens |
| Input contexts | 8k, 16k, 32k |
| Cache location for Cachet methods | Local disk, not Unity Catalog |
| Methods | Baseline, Cachet + vanilla KV, Cachet + KV Packet |

Blank numeric cells in the main and ablation tables mean not measured yet; not
zero. KV Packet rows are present so the table captures the intended
method comparison, but they are marked `not implemented yet`.

## Existing Evidence Boundary

Current committed vLLM, SGLang, storage, and native-probe results are preserved
under [`appendix/existing-results/`](appendix/existing-results/). They remain
auditable evidence, but they do not populate the main table because they used
different prompt-token ranges, repeat counts, output lengths, cache/storage
assumptions, hardware, or serving paths.

The [`databricks/`](databricks/) folder remains unmoved because release-evidence
JSON records refer to those paths. Do not put raw Databricks Jobs API
responses, credentials, package wheels, driver logs, generated datasets, prompt
payload blobs, or local scratch output in this directory.

Historical failed SGLang smoke attempts and superseded readiness artifacts live
under [`../docs/release-ops/benchmark-archive/`](../docs/release-ops/benchmark-archive/).
