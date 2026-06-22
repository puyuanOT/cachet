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
supplying one enriched `--dataset DATASET=PATH` flag for each required dataset
plus the server sizing flags (`--max-model-len`, `--max-num-seqs`, and
`--gpu-memory-utilization`). In prepared mode every row must carry Cachet
`kv_transfer_params`; the runner writes `prepared-handoff-coverage.json` before
model startup and sends the full logical prompt plus `kv_transfer_params` for
the cache arm. Native vLLM needs the logical prefix token positions to allocate
external KV blocks, so suffix-only cache prompts are not used by this smoke. The
generated metadata records the dataset paths, sizing values, Cachet package
install spec, and vLLM `KVTransferConfig`, which separates long-context
benchmark evidence from the built-in smoke examples.
