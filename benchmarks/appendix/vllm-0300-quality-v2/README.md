# vLLM 0.30.0 Quality Benchmark v2

This directory contains the sealed quality-v2 report for the vLLM 0.30.0
campaign.

- [`quality-v2-report.json`](quality-v2-report.json) is the sealed report with
  closed-record SHA-256
  `86e6707aa056ab8f36306584e509e3fa391dd6946e0ad94dfe0e49aa8fcff677`.

The report pairs full prefill and Vanilla KV outputs by example over every
selected HotpotQA (7,405), MuSiQue (2,417), and NIAH (1,000) example plus a
2,000-example Biography diagnostic subset. The examples were served in 21
single-GPU chunks, each one producer pass and one vLLM 0.30.0 boot serving both
arms. Two chunks (`hotpotqa-04` and `niah-07`) lost their first cluster to AWS
capacity before producing a result and ran again unchanged; the report records
the attempt and output file digests of every chunk.

Raw per-example outputs, server logs, prompts, and Databricks responses are
intentionally excluded.
