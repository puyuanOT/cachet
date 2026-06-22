# vLLM Smoke Bundle Template

This packaged template mirrors `databricks/vllm-smoke/databricks.yml` from the
repository. It defines only the self-contained vLLM smoke job, so users can
validate the target AWS g6/L4 Databricks runtime before preparing full V1 benchmark
datasets or plan artifacts.

The generated runner installs the uploaded Cachet wheel into both the
Databricks driver process and the isolated vLLM environment. vLLM starts with
Cachet's `DocumentKVConnector` `KVTransferConfig`, so prepared inputs carrying
`kv_transfer_params` exercise the provider-backed cache path instead of a stock
prefill-only server. The runner writes `vllm-import-probe.json` before model
startup, and that probe fails unless the same `KVTransferConfig` resolves to a
native document-KV provider factory.

The same runner can be configured for prepared V1 benchmark JSONL files by
supplying one `--dataset DATASET=PATH` flag for each required dataset plus the
server sizing flags (`--max-model-len`, `--max-num-seqs`, and
`--gpu-memory-utilization`). In prepared mode the generated metadata records the
dataset paths, sizing values, Cachet package install spec, and vLLM
`KVTransferConfig`, which separates long-context benchmark evidence from the
built-in smoke examples.
