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

## Inputs

The inputs derive from the campaign's pinned full-score natural datasets
(`_FULL_SCORE_PUBLICATION_SOURCE_RECORDS` in
`src/document_kv_cache/full_score_execution.py` at the
`vllm-0300-campaign-archive` tag):

| Dataset | Rule | Examples | Input JSONL SHA-256 |
| --- | --- | ---: | --- |
| HotpotQA | Every source example, unchanged | 7,405 | `c871d8536c48e8db5e39e2215ae5e9eb37260dbc4eb871816e2049f1863ae2dc` |
| MuSiQue | Every source example, unchanged | 2,417 | `f1404c3ccef4b1df8954525efc95f09ce26e4cbf39747372c06f452fd9d13c6a` |
| NIAH | `build_quality_v2_niah_records(paragraphs, item_count=1000)` over every HotpotQA and MuSiQue source paragraph in file order, counting tokens with the Qwen3 tokenizer at revision `cdbee75f17c01a7cc42f958dc650907174af0554` | 1,000 | `d6c2c9d988e1417c1434cc565602a4b2c4334f87808e96e7eed98caad69f2c39` |
| Biography | `random.Random(f"{QUALITY_V2_NIAH_SEED}:biography-diagnostic").sample(source, 2000)`, sorted by `example_id` | 2,000 | `b18adf6bd29b9776465c34ddccbab02e2e0242544a9b26e76edf404cabcac1e0` |

Each input JSONL holds one `json.dumps(record, ensure_ascii=False,
sort_keys=True)` line per example. HotpotQA and MuSiQue examples carry up to 10
and 20 documents; every NIAH and Biography example carries exactly one.

Raw per-example outputs, server logs, prompts, and Databricks responses are
intentionally excluded.
