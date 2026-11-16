# vLLM 0.30.0 Auxiliary Latency Re-run

This directory contains the sealed, plan-coherent re-run of the vLLM 0.30.0
campaign's four auxiliary latency contrasts (RAM and Unity Catalog against
Disk, BF16 and A10G against the 16k, concurrency-4 Vanilla reference).

- [`auxiliary-latency-rerun.json`](auxiliary-latency-rerun.json) is the sealed
  record with closed-record SHA-256
  `18a51fc96e21af35960e6532fb2b0533d89279f22912b9bb5a9c04ce842af201`.

All 30 jobs ran under execution plan
`219c45b80decdc37b8c66f41379d0430d85bc5825c6809badb67e68ee0c1153a`, one package wheel, and one
source tree, five deployment blocks of six cells each. The record binds each
job's sealed result digest, the paired comparisons and their coherence checks,
the per-cell descriptive statistics, and the paired crossed-bootstrap
estimates. `block-04-hardware-a10g` ran on attempt 2 after attempt 1 ended `FAILED` with no result. Each job's attempts, with their Databricks run ids and result states, are in the record.

Raw per-request results, server and connector logs, prompts, and Databricks
responses are intentionally excluded.
