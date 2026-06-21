# vLLM Smoke Bundle Template

This packaged template mirrors `databricks/vllm-smoke/databricks.yml` from the
repository. It defines only the self-contained vLLM smoke job, so users can
validate the target AWS g6/L4 Databricks runtime before preparing full V1 benchmark
datasets or plan artifacts.

The same runner can be configured for prepared V1 benchmark JSONL files by
supplying one `--dataset DATASET=PATH` flag for each required dataset plus the
server sizing flags (`--max-model-len`, `--max-num-seqs`, and
`--gpu-memory-utilization`). In prepared mode the generated metadata records the
dataset paths and sizing values, which separates long-context benchmark evidence
from the built-in smoke examples.
