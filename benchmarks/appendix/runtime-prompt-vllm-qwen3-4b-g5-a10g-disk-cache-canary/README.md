# Runtime-Prompt vLLM Canary: Qwen3 4B on g5/A10G

This folder records a failed Databricks canary for the primary Cachet + vanilla
KV latency row. The canary used the fixed primary-table environment with
`--benchmark-cache-runtime-prompt`, which makes Cachet requests send only the
uncached runtime suffix while passing the KV handoff out of band.

## Table Configuration

| Field | Value |
| --- | --- |
| Model | Qwen3-4B-Instruct, `qwen3:4b-instruct` |
| Serving engine | vLLM |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Request parallelism | 8 requests in flight |
| Output length for TTC | Forced 256-token decode |
| Input context | 8k tokens |
| Dataset inputs | Biography, HotpotQA, MusiQue, NIAH synthetic prepared examples |
| Cachet method | Cachet + vanilla KV |
| Cache residency | Local disk under `/local_disk0`, not Unity Catalog |
| Cache prompt mode | Runtime suffix only |
| Source commit | `1369e48` |
| DBFS staging root | `dbfs:/benchmarks/cachet/primary-table-runtime-prompt-1369e48-20260627_020433` |

## Result

| Method | Input context | Successful requests | Failed requests | Result |
| --- | ---: | ---: | ---: | --- |
| Cachet + vanilla KV | 8k | 0 | 32 | vLLM rejected runtime-prompt mode before any successful measurement |

The vLLM server log reports:

```text
ValueError: Document KV vLLM loads require the full logical prompt; document_kv.prompt_text_mode='runtime' cannot be used
```

## Interpretation

This canary confirms that the current vLLM native provider cannot use
suffix-only runtime prompt text. Current primary Cachet + vanilla KV rows
therefore use vLLM's logical-prompt external-prefix path: Cachet binds the raw
KV handoff out of band, vLLM matches the cached prefix against the logical
prompt, and the provider loads local-disk KV blocks for that cached prefix.

## Provenance

| File | Contents |
| --- | --- |
| [`failure_summary.json`](failure_summary.json) | Sanitized run configuration, Databricks run ids, and failure interpretation |
