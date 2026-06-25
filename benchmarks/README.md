# Cachet Benchmarks

This directory is the public benchmark surface for Cachet. It answers three
questions before exposing audit details:

- how Cachet compares with the no-cache prefill baseline;
- which models, hardware targets, cache methods, and serving engines are
  covered;
- which results are performance benchmarks versus integration or readiness
  evidence.

Start with [`current/`](current/) for the shortest answer.

## Headline Results

| Serving platform | Model | Hardware | Method | Cachet vs baseline | Status |
| --- | --- | --- | --- | --- | --- |
| vLLM | Qwen3 4B Instruct | AWS g6/L4, `g6.8xlarge` | Vanilla external KV | 5.27x-6.97x TTFT speedup; 1.74x-2.25x time-to-completion speedup; unchanged quality | Published benchmark |
| vLLM | Qwen3 4B Instruct | AWS g5/A10G, `g5.8xlarge` | Vanilla external KV | 4.66x-6.04x TTFT speedup; 2.04x-2.67x time-to-completion speedup; unchanged quality | Compatibility benchmark |
| SGLang | Qwen3 4B Instruct | AWS g6/L4, `g6.8xlarge` | Vanilla external KV via HiCache | 8/8 Cachet-backed cache-hit validations and unchanged quality, but slower than baseline on short prepared prompts | Integration benchmark; not a speedup claim |
| Storage readers | n/a | AWS g6/L4, `g6.8xlarge` | Memory, disk, Unity Catalog | Zero reader errors; Memory 6531.4 MiB/s, Disk 6214.4 MiB/s, Unity Catalog 1148.0 MiB/s | Storage benchmark |
| Native engine probes | Qwen3 4B layout probe | AWS g6/L4, `g6.8xlarge` | Vanilla external KV handoff | vLLM and SGLang provider-backed native connector probes copied 48 tokens | Integration evidence, not latency evidence |

## Benchmark Matrix

| Folder | Audience | What It Shows |
| --- | --- | --- |
| [`current/`](current/) | Users comparing Cachet with baselines | Current performance, method, model, hardware, and serving-platform summary |
| [`vllm/`](vllm/) | vLLM users | Full no-cache-prefill versus Cachet vanilla-KV latency and quality benchmarks |
| [`sglang/`](sglang/) | SGLang users and maintainers | Current SGLang live benchmark status, including validated cache hits and no current speedup |
| [`storage/`](storage/) | Integrators choosing storage backends | Memory, disk, and Unity Catalog reader throughput and latency |
| [`native-engine/`](native-engine/) | Serving maintainers | vLLM/SGLang native connector evidence against engine-owned KV block managers |
| [`databricks/`](databricks/) | Release operators and auditors | Sanitized QA run-status mirrors for the public benchmark folders |
| [`_template/`](_template/) | Maintainers adding results | Stable report template for new public benchmark folders |

Use public benchmark folder names such as `qwen3-4b-g6-l4-vanilla-kv` instead of dates.
The folder name should tell a reader the serving platform, model,
hardware, and method before they open a file. Databricks run IDs, task IDs, and
audit sidecars remain available inside the result details or under
[`databricks/`](databricks/), but they are not the primary navigation model.

## Methods

| Method | Current Support | Benchmark Status |
| --- | --- | --- |
| Vanilla external KV cache | Supported today | Benchmarked on vLLM; validated on SGLang with HiCache-backed cache hits |
| KV Packet | Planned next | Not benchmarked yet |
| Adapter-trained or learned KV methods | API leaves room for future adapters | Not benchmarked yet |

Only cite a method as supported when the matching benchmark or integration
folder says so. Planned methods should appear in this table, not as performance
claims.

## Evidence Boundary

The public folders contain concise `README.md` summaries plus sanitized JSON
records needed to audit each claim. They intentionally exclude raw Databricks
Jobs API responses, credentials, package wheels, driver logs, generated
datasets, prompt payload blobs, and local scratch output.

Historical failed SGLang smoke attempts and superseded readiness artifacts live
under [`docs/release-ops/benchmark-archive/`](../docs/release-ops/benchmark-archive/)
instead of the public benchmark path.
