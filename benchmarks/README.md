# Cachet Benchmarks

This directory is the public benchmark surface for Cachet.

<!-- cachet:vllm-0271-publication-table:status:begin -->
> **Status: vLLM 0.30.0 campaign published.** The sanitized campaign report and its report-bound publication gate passed. The gate is an integrity gate: it checks evidence bindings and fixed coverage counts, sets no quality or latency threshold, and latency results are estimation-only.
>
> Coverage: 115 latency jobs; 83,653 distinct full-score examples; 101,573 Vanilla KV cache requests counted by the gate (17,920 latency and 83,653 full-score), of which 16,640 latency requests are cold-read attested.
>
> Only latency requests carry cold-read attestations. The design requires 15,360 to 16,640 of the 17,920: the 1,280 RAM-storage requests are served from the in-memory payload cache and never read cold, and the 1,280 Unity Catalog requests load through the mounted path and need not attest one.
<!-- cachet:vllm-0271-publication-table:status:end -->

<!-- cachet:vllm-0271-publication-table:campaign-disclosures:begin -->
> **Campaign ledger disclosures.** Rendered from the sealed report's Databricks cluster-hour ledger projection and latency source fields. They change no metric on this page; they record the campaign's budget, retried and abandoned work, and how latency jobs are bound to the ledger.

| Disclosure | Report value | Meaning |
| --- | --- | --- |
| Cluster-hour budget | Raised from 1,024 h to 2,024 h after 166 reservations, 119 receipts, and 166 terminal actuals | Budget extensions recorded in the campaign ledger; every ledger prefix before a raise keeps its earlier cap |
| Final ledger accounting | 1,345.64 h accounted; 678.36 h remaining of a 2,024 h cap; 0 active reserved tasks | The ledger closed with nothing reserved and at least 124 h unreserved |
| GPU qualification capacity retries | 0 attempts, 0.00 h | Qualification jobs relaunched after a Databricks capacity failure |
| Full-score capacity retries | 37 attempts, 132.51 h | Phases relaunched after a Databricks capacity failure; the 57 full-score ledger attempts after the latency prefix cover the 20 phases, and only each phase's final attempt is scored |
| Abandoned full-score attempts | 74 attempts, 368.23 h; 11 closed consumer phases | Earlier full-score chains the campaign restarted over before the latency prefix; the waves their consumer phases scored are not used in this report |
| Latency recovered balance | 47 seeds, 554.99 h | Terminal-only ledger seeds restoring hours spent before the ledger was recreated |
| Latency jobs with their own ledger receipt | 70 of 115 | Receipt recorded when the job was submitted |
| Latency jobs with a recovered receipt | 4 of 115 | Finished before the ledger was recreated; receipt re-recorded by run ID under a timeout-only stand-in payload |
| Latency jobs bound only by the recovered balance | 41 of 115 | Finished before the ledger was recreated and have no ledger record; accepted only because the recovered-balance seeds cover the spent hours |
| Latency launch plans | 34 execution plans; 54 superseded attempts | Jobs are joined across the plans they ran under (capacity retries and reopened cells); superseded attempts are not used. Plan-set SHA-256 `f248a29c8a198a3c5973da9f235c0a9db4510a838585bcaa9565477434bfd366` |
<!-- cachet:vllm-0271-publication-table:campaign-disclosures:end -->

The governed tables on this page are rendered byte-for-byte from the sealed
campaign report under [`appendix/vllm-0271-publication-v1/`](appendix/vllm-0271-publication-v1/).
The campaign ran on vLLM 0.30.0; its identifier keeps the name of the frozen
0.27.1 design it was ported from. Two later, separately sealed measurements
complete it:

- [Quality Benchmark v2](#quality-benchmark-v2) replaces the campaign's score
  table as the measure of answer quality. The v1 full-score protocol did not
  measure answer quality (see the warning on that table).
- The [Auxiliary Latency Re-run](#auxiliary-latency-re-run) repeats the four
  auxiliary contrasts (storage, BF16, A10G) with each block's reference and
  treatments under one launch plan, uniform untimed warmup, and measured cold
  reads. The campaign's auxiliary rows joined cells across launch plans.

## Frozen Campaign Configuration

The closed design is implemented by
[`document_kv_cache.publication_campaign`](https://github.com/puyuanOT/cachet/blob/vllm-0300-campaign-archive/src/document_kv_cache/publication_campaign.py)
at the `vllm-0300-campaign-archive` tag, which preserves the campaign machinery
that is not part of the package. The source snapshot and generated campaign
record must be frozen before any production job is submitted.

| Field | Frozen value |
| --- | --- |
| Campaign | `vllm-0271-publication-v1` |
| Model | `Qwen/Qwen3-4B-Instruct-2507`, served as `qwen3:4b-instruct` |
| Model weights | bitsandbytes 4-bit runtime weights; the model identifier is not itself a prequantized 4-bit checkpoint |
| Serving engine | vLLM `0.30.0`, ported from the frozen vLLM `0.27.1` design |
| Core serving hardware | AWS g6/L4, `g6.8xlarge` |
| Handoff generation hardware | AWS g6e/L40S was the designed sole handoff generator, and generated bundles are reused by timed serving jobs. Bundles later lost from the campaign volume were regenerated on L4 with the same pinned generator and runtime (same files and sizes, not byte-identical payloads). Of the 70 latency jobs that serve a handoff bundle, 23 served an L4-regenerated bundle (all fifteen 8k Vanilla jobs, six 32k Vanilla jobs, and two BF16 jobs), 18 served a byte-identical restored copy of an L40S bundle, and the other 29 served their original L40S bundles |
| Default document/runtime KV | Q8, `fp8_e5m2`, pre-RoPE keys with absolute-position injection |
| Input contexts | 8k, 16k, and 32k prepared prompts |
| Closed-loop request concurrency | 1, 2, and 4; zero think time |
| Dataset coverage | Biography, HotpotQA, MusiQue, and NIAH |
| Examples | 32 distinct examples per dataset |
| Repeats | 2 per example within each deployment block |
| Deployment blocks | 5 matched fresh-cluster blocks |
| Requests per method/context/concurrency cell and block | 256 |
| Latency decode | Forced 256-token decode; exact decoding pins come from the closed latency execution record |
| Full-score decode | Natural EOS with a 64-token maximum, temperature 0, closed-loop concurrency 4, one paired pass per method, and no prompt padding or tokenizer truncation |
| No-retry job timeouts | 8k c1/c2/c4: 6/4/4h; 16k: 8/6/4h; 32k: 12/8/4h; all c4 auxiliary jobs: 4h |
| Experimental units | Core Baseline/Vanilla pair; Disk/RAM/UC trio; 16k-c4 core pair plus BF16/A10G four-job wave |
| Latency analysis | Paired crossed bootstrap over deployment blocks and shared example identities; no post-hoc cell significance |
| Full-score analysis | Per-example paired deltas with pointwise 95% paired-example bootstrap intervals, 20,000 draws, and dataset stratification |
| Full-score scope | One complete paired pass over every selected dataset row; no padding, truncation, sampling replacement, or answer-quality preservation gate |
| Budget | 1,024 aggregate GPU-hours, 900 active reserved hours, 124 hours unreserved headroom, at most 16 parallel jobs; raised to 2,024 GPU-hours during the campaign (see the ledger disclosures above) |
| Retained ledger opening | 71.390128 reconciled GPU-hours; exact 236/98/236 post-migration append-only prefix is campaign-bound; zero active reservations |
| Frozen generation workload | 72,871,510 cache-prefix tokens across Q8 latency, BF16 latency, and complete full-score handoffs; 578.345317 GPU-hours at the 35 token/GPU-s gate |

Retained legacy-v1 qualification records keep their historical cross-hardware
raw-byte identity rule and are never reinterpreted. Fresh native-v2
qualification uses a distinct repeat-aware sentinel: it requires same-hardware
fresh-load byte reproducibility on L4 and L40S, cross-hardware
logical/token/layout/size equivalence without raw-digest equality, and an
independent L40S throughput pass.

`Baseline` sends the complete logical prompt to vLLM and computes all KV at
request time. `Vanilla KV` reuses independently generated document KV, stores
keys pre-RoPE, assembles documents in logical order, applies absolute positions
at injection, and measures storage-to-GPU hydration inside TTFT. Vanilla is not
required to preserve Baseline answer quality: the full evaluation is intended
to measure the quality loss, especially for multi-document QA.

KV Packet, CacheBlend, and InfoFlow KV have no executable Cachet benchmark path
in this campaign. Their cells remain explicitly unsupported rather than being
estimated from another method.

## Main Latency And Resource Table

Latency values are reported in seconds. Memory values are rendered as GiB
together with their exact byte counts. Each implemented row represents five
matched deployment blocks; no pre-reset measurement is carried forward.

<!-- cachet:vllm-0271-publication-table:core-latency:begin -->
| Method | Input context | Concurrency setting | P50 TTFT (s) | P95 TTFT (s) | P50 TTC (s, 256 toks) | P95 TTC (s, 256 toks) | P50 decode tok/s | Configured closed-loop concurrency | Peak GPU process memory | Peak host memory | Peak process-tree RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 8k | 1 | 2.6135 | 2.6457 | 15.8455 | 16.1795 | 19.34 | 1 | 0.000 GiB (0 bytes) | 29.745 GiB (31,938,039,808 bytes) | 3.779 GiB (4,057,989,120 bytes) |
| Vanilla&nbsp;KV | 8k | 1 | 0.9450 | 1.0265 | 14.2516 | 14.6051 | 19.24 | 1 | 0.000 GiB (0 bytes) | 29.970 GiB (32,180,011,008 bytes) | 4.392 GiB (4,716,294,144 bytes) |
| Baseline | 8k | 2 | 3.0486 | 3.6896 | 21.6681 | 22.1533 | 13.88 | 2 | 0.000 GiB (0 bytes) | 29.196 GiB (31,349,207,040 bytes) | 3.793 GiB (4,073,091,072 bytes) |
| Vanilla&nbsp;KV | 8k | 2 | 1.8369 | 1.9489 | 18.4292 | 18.9054 | 15.43 | 2 | 0.000 GiB (0 bytes) | 30.266 GiB (32,497,545,216 bytes) | 4.428 GiB (4,754,821,120 bytes) |
| Baseline | 8k | 4 | 4.2330 | 4.8973 | 27.3916 | 27.7110 | 11.12 | 4 | 0.000 GiB (0 bytes) | 29.936 GiB (32,143,495,168 bytes) | 3.806 GiB (4,087,037,952 bytes) |
| Vanilla&nbsp;KV | 8k | 4 | 3.3956 | 3.6025 | 20.3097 | 20.5579 | 15.20 | 4 | 0.000 GiB (0 bytes) | 29.376 GiB (31,542,415,360 bytes) | 4.445 GiB (4,772,397,056 bytes) |
| Baseline | 16k | 1 | 7.5287 | 7.9439 | 20.8174 | 21.3683 | 19.40 | 1 | 0.000 GiB (0 bytes) | 29.011 GiB (31,150,710,784 bytes) | 3.793 GiB (4,072,595,456 bytes) |
| Vanilla&nbsp;KV | 16k | 1 | 1.7770 | 1.9079 | 15.1499 | 15.5268 | 19.17 | 1 | 0.000 GiB (0 bytes) | 30.471 GiB (32,717,807,616 bytes) | 4.981 GiB (5,348,839,424 bytes) |
| Baseline | 16k | 2 | 8.5575 | 10.5078 | 32.1174 | 32.9640 | 10.92 | 2 | 0.000 GiB (0 bytes) | 30.300 GiB (32,533,970,944 bytes) | 3.849 GiB (4,132,691,968 bytes) |
| Vanilla&nbsp;KV | 16k | 2 | 3.3996 | 3.5529 | 20.6281 | 20.9221 | 14.89 | 2 | 0.000 GiB (0 bytes) | 31.540 GiB (33,865,486,336 bytes) | 5.000 GiB (5,368,848,384 bytes) |
| Baseline | 16k | 4 | 10.2025 | 11.1110 | 49.3268 | 49.8144 | 6.56 | 4 | 0.000 GiB (0 bytes) | 27.820 GiB (29,871,738,880 bytes) | 3.826 GiB (4,107,673,600 bytes) |
| Vanilla&nbsp;KV | 16k | 4 | 6.6893 | 6.8766 | 26.5280 | 26.8459 | 12.92 | 4 | 0.000 GiB (0 bytes) | 31.208 GiB (33,509,347,328 bytes) | 5.019 GiB (5,389,053,952 bytes) |
| Baseline | 32k | 1 | 24.9313 | 25.6148 | 38.1711 | 38.9802 | 19.25 | 1 | 0.000 GiB (0 bytes) | 30.607 GiB (32,864,153,600 bytes) | 3.831 GiB (4,113,100,800 bytes) |
| Vanilla&nbsp;KV | 32k | 1 | 3.3045 | 3.4823 | 16.5293 | 16.8730 | 19.40 | 1 | 0.000 GiB (0 bytes) | 31.377 GiB (33,690,509,312 bytes) | 6.112 GiB (6,562,750,464 bytes) |
| Baseline | 32k | 2 | 26.7325 | 33.5432 | 69.1110 | 71.6018 | 6.06 | 2 | 0.000 GiB (0 bytes) | 32.578 GiB (34,980,614,144 bytes) | 3.842 GiB (4,125,671,424 bytes) |
| Vanilla&nbsp;KV | 32k | 2 | 6.6919 | 6.9381 | 26.9123 | 27.7156 | 12.51 | 2 | 0.000 GiB (0 bytes) | 31.176 GiB (33,475,280,896 bytes) | 6.118 GiB (6,569,119,744 bytes) |
| Baseline | 32k | 4 | 30.7658 | 31.9454 | 123.6072 | 126.4093 | 2.74 | 4 | 0.000 GiB (0 bytes) | 31.912 GiB (34,265,657,344 bytes) | 3.839 GiB (4,121,980,928 bytes) |
| Vanilla&nbsp;KV | 32k | 4 | 13.0823 | 13.3619 | 38.3251 | 38.7688 | 10.13 | 4 | 0.000 GiB (0 bytes) | 30.982 GiB (33,266,434,048 bytes) | 6.154 GiB (6,607,933,440 bytes) |
<!-- cachet:vllm-0271-publication-table:core-latency:end -->

## Paired Latency Estimands

Each estimand is computed from five matched deployment blocks. A speedup is
`reference latency / treatment latency`, so values above 1 mean that the named
treatment is faster. The table reports estimation with pointwise 95% paired
crossed-bootstrap intervals; it is not a grid of post-hoc significance
tests.

The intervals describe uncertainty over deployments and shared example
identities. Each bootstrap draw samples five matched blocks and independently
samples example identities within each fixed dataset stratum. The same example
multiset is used across all selected blocks, with paired request repeats kept
together. These are approximate intervals; five deployments do not establish
exact 95% coverage, and the storage ablation has only two identities per dataset.

<!-- cachet:vllm-0271-publication-table:latency-estimands:begin -->
| Treatment vs reference | Setting | TTFT geometric speedup | TTFT 95% CI | TTC geometric speedup | TTC 95% CI | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Vanilla KV vs Baseline | 8k, concurrency 1 | 2.7349 | [2.6724, 2.7867] | 1.1112 | [1.0997, 1.1232] | Five matched blocks |
| Vanilla KV vs Baseline | 8k, concurrency 2 | 1.7180 | [1.6743, 1.7653] | 1.1738 | [1.1602, 1.1866] | Five matched blocks |
| Vanilla KV vs Baseline | 8k, concurrency 4 | 1.1675 | [1.1259, 1.2073] | 1.3408 | [1.3296, 1.3505] | Five matched blocks |
| Vanilla KV vs Baseline | 16k, concurrency 1 | 4.2214 | [4.1462, 4.3233] | 1.3714 | [1.3434, 1.3975] | Five matched blocks |
| Vanilla KV vs Baseline | 16k, concurrency 2 | 2.5492 | [2.4654, 2.6362] | 1.5553 | [1.5361, 1.5789] | Five matched blocks |
| Vanilla KV vs Baseline | 16k, concurrency 4 | 1.4748 | [1.4485, 1.5025] | 1.8546 | [1.8424, 1.8672] | Five matched blocks |
| Vanilla KV vs Baseline | 32k, concurrency 1 | 7.4920 | [7.3715, 7.6187] | 2.3153 | [2.2874, 2.3496] | Five matched blocks |
| Vanilla KV vs Baseline | 32k, concurrency 2 | 4.3192 | [4.2356, 4.4057] | 2.5631 | [2.5261, 2.5913] | Five matched blocks |
| Vanilla KV vs Baseline | 32k, concurrency 4 | 2.2942 | [2.2370, 2.3562] | 3.2314 | [3.1800, 3.2870] | Five matched blocks |
| BF16 payload/runtime KV vs Q8 | 16k, concurrency 4, L4/NVMe | 0.5388 | [0.5221, 0.5557] | 0.7282 | [0.7207, 0.7359] | Five matched blocks |
| RAM vs Disk | 16k, concurrency 4, L4 | 1.6704 | [1.6523, 1.6989] | 1.1132 | [1.1085, 1.1185] | Five matched blocks |
| Unity Catalog vs Disk | 16k, concurrency 4, L4 | 0.5754 | [0.5585, 0.5927] | 0.8410 | [0.8337, 0.8483] | Five matched blocks |
| A10G vs L4 | 16k, concurrency 4, local NVMe | 0.9556 | [0.9356, 0.9677] | 1.0671 | [1.0584, 1.0746] | Five matched blocks |
<!-- cachet:vllm-0271-publication-table:latency-estimands:end -->

## Unsupported Method Status

| Method | Latency/resource result | Full-dataset score | Reason |
| --- | --- | --- | --- |
| [KV&nbsp;Packet](https://arxiv.org/abs/2604.13226) | N/A (method not implemented) | N/A (method not implemented) | No pinned executable integration or Cachet artifact/serving contract |
| CacheBlend | N/A (method not implemented) | N/A (method not implemented) | Selective cross-document recomputation is not implemented |
| InfoFlow&nbsp;KV | N/A (method not implemented) | N/A (method not implemented) | Cross-document information-flow method is not implemented |
| LMCache | N/A (runtime not qualified) | N/A (runtime not qualified) | No combined hash-locked vLLM 0.27.1/LMCache runtime or GPU qualification |
| Multi connector | N/A (runtime not qualified) | N/A (runtime not qualified) | No combined hash-locked vLLM 0.27.1/Multi runtime or GPU qualification |

LMCache and Multi retain smoke implementations. Their vLLM 0.27.1 canary and
publication paths require a combined hash-locked runtime and GPU qualification.

## Benchmark Dataset Score Table

> **These v1 scores do not measure answer quality.** The v1 protocol sent raw
> completions without the model's chat template, stopped decoding at 64 tokens,
> and accepted only an output that was exactly one `<final_answer>` block. The
> instruct model echoed instruction text before answering, so the strict parser
> rejected most outputs of both methods (see the `extraneous_text` counts) and
> both columns sit near zero. The table is kept because it is part of the sealed
> report. Use [Quality Benchmark v2](#quality-benchmark-v2) for answer quality.

The reset requires a complete paired evaluation over every selected row in the
four implemented datasets. The earlier small diagnostic is not retained or
substituted for this table. Full-score values use one paired pass per method
over all 83,653 natural-length examples, with no padding or tokenizer
truncation. Requests run at closed-loop concurrency 4, temperature 0, natural
EOS enabled, and a 64-token maximum. Parser-status cells report the complete
counts for `ok`, `missing_block`, `multiple_or_malformed_blocks`,
`extraneous_text`, `nested_block`, and `empty_answer`, including explicit zeros for unobserved
states. Invalid or malformed final-answer parses receive zero
and remain in denominators. Scores, deltas, and confidence limits are unit-scale
fractions rendered to six decimal places; scientific notation is used if a
nonzero value would otherwise round to zero. Intervals are pointwise 95%
paired-example bootstrap intervals with 20,000 draws, stratified by dataset;
NIAH is additionally reported over its frozen nine-cell grid.

<!-- cachet:vllm-0271-publication-table:dataset-scores:begin -->
| Dataset | Governed metric | n | Baseline | Baseline parser-status counts | Vanilla KV | Vanilla parser-status counts | Vanilla − Baseline | Paired 95% CI |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: | ---: |
| Biography | Normalized-title exact match | 72831 | 0.001291 | ok=9196; missing_block=165; multiple_or_malformed_blocks=525; extraneous_text=62945; nested_block=0; empty_answer=0 | 0.001483 | ok=9445; missing_block=67; multiple_or_malformed_blocks=399; extraneous_text=62920; nested_block=0; empty_answer=0 | 0.000192 | [-0.000151, 0.000522] |
| HotpotQA | Answer exact match | 7405 | 0.054693 | ok=866; missing_block=230; multiple_or_malformed_blocks=2556; extraneous_text=3753; nested_block=0; empty_answer=0 | 0.000270 | ok=5; missing_block=433; multiple_or_malformed_blocks=524; extraneous_text=6443; nested_block=0; empty_answer=0 | -0.054423 | [-0.059689, -0.049291] |
| HotpotQA | Answer F1 | 7405 | 0.073818 | ok=866; missing_block=230; multiple_or_malformed_blocks=2556; extraneous_text=3753; nested_block=0; empty_answer=0 | 0.000304 | ok=5; missing_block=433; multiple_or_malformed_blocks=524; extraneous_text=6443; nested_block=0; empty_answer=0 | -0.073514 | [-0.079105, -0.067895] |
| MusiQue | Official answer exact match, alias-max | 2417 | 0.021514 | ok=241; missing_block=129; multiple_or_malformed_blocks=270; extraneous_text=1777; nested_block=0; empty_answer=0 | 0.000000 | ok=1; missing_block=201; multiple_or_malformed_blocks=185; extraneous_text=2030; nested_block=0; empty_answer=0 | -0.021514 | [-0.027307, -0.016136] |
| MusiQue | Official answer F1, alias-max | 2417 | 0.033233 | ok=241; missing_block=129; multiple_or_malformed_blocks=270; extraneous_text=1777; nested_block=0; empty_answer=0 | 0.000000 | ok=1; missing_block=201; multiple_or_malformed_blocks=185; extraneous_text=2030; nested_block=0; empty_answer=0 | -0.033233 | [-0.039884, -0.026839] |
| NIAH | Exact-value overall accuracy | 1000 | 0.009000 | ok=19; missing_block=52; multiple_or_malformed_blocks=235; extraneous_text=694; nested_block=0; empty_answer=0 | 0.000000 | ok=0; missing_block=2; multiple_or_malformed_blocks=121; extraneous_text=877; nested_block=0; empty_answer=0 | -0.009000 | [-0.015000, -0.004000] |
| LongBench v2 | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) |
| RULER | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) | N/A (runner not implemented) |
<!-- cachet:vllm-0271-publication-table:dataset-scores:end -->

NIAH is additionally reported as the full 3-by-3 grid below; every cell is a
governed exact-value accuracy over its bound source examples.

<!-- cachet:vllm-0271-publication-table:niah-grid:begin -->
| Context | Needle position | n | Baseline accuracy | Vanilla KV accuracy | Vanilla − Baseline | Paired 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8k | 10% | 112 | 0.000000 | 0.000000 | 0.000000 | [0.000000, 0.000000] |
| 8k | 50% | 111 | 0.000000 | 0.000000 | 0.000000 | [0.000000, 0.000000] |
| 8k | 90% | 111 | 0.000000 | 0.000000 | 0.000000 | [0.000000, 0.000000] |
| 16k | 10% | 111 | 0.027027 | 0.000000 | -0.027027 | [-0.063063, 0.000000] |
| 16k | 50% | 111 | 0.000000 | 0.000000 | 0.000000 | [0.000000, 0.000000] |
| 16k | 90% | 111 | 0.009009 | 0.000000 | -0.009009 | [-0.027027, 0.000000] |
| 32k | 10% | 111 | 0.009009 | 0.000000 | -0.009009 | [-0.027027, 0.000000] |
| 32k | 50% | 111 | 0.018018 | 0.000000 | -0.018018 | [-0.045045, 0.000000] |
| 32k | 90% | 111 | 0.018018 | 0.000000 | -0.018018 | [-0.045045, 0.000000] |
<!-- cachet:vllm-0271-publication-table:niah-grid:end -->

## Quality Benchmark v2

Quality v2 re-measures answer quality for the same model, runtime, datasets,
answer normalisation, and per-document Vanilla KV method, changing only what
the v1 audit showed to be broken:

- each prompt is wrapped in the Qwen3 chat template as a single user turn;
- decoding may run to 256 tokens at temperature 0;
- answers come from the versioned tolerant parser `cachet.first_final_answer`
  v2 (the first closed `<final_answer>` block that is not the echoed
  placeholder); strict whole-output compliance is reported separately;
- Biography references and predictions are compared without WikiBio
  `-lrb- … -rrb-` disambiguators (WikiBio's tokenised parentheses) or other
  parenthesised text, and Biography
  is diagnostic only: its
  title exact match is not a reliable quality measure (see below);
- NIAH uses natural-text haystacks (HotpotQA and MuSiQue context paragraphs)
  with a numeric needle at a paragraph boundary, over the same 3-by-3 grid.

Every selected HotpotQA, MuSiQue, and NIAH example is scored, plus a fixed
2,000-example Biography diagnostic subset. Each example runs once with full
prefill and once with the Vanilla KV cache on the same vLLM boot, and the two
outputs are paired by example. Intervals are 95% percentile bootstrap
intervals over paired examples. The protocol fixes a non-inferiority margin
before scoring: on each headline dataset's primary metric, the lower end of
the Vanilla − full prefill interval must be at least −0.02. A dataset that
does not meet the margin is reported as such; nothing is dropped or retried
to change a verdict. The sealed report is
[`appendix/vllm-0300-quality-v2/quality-v2-report.json`](appendix/vllm-0300-quality-v2/).

<!-- cachet:vllm-0300-quality-v2:begin -->
Report `closed_record_sha256`: `86e6707aa056ab8f36306584e509e3fa391dd6946e0ad94dfe0e49aa8fcff677`. Protocol `cachet-vllm-0.30.0-quality-v2`, parser `cachet.first_final_answer` v2, 10,000 paired bootstrap draws.

| Dataset | Role | Metric | Pairs | Full prefill | Vanilla KV | Vanilla − full prefill | Paired 95% CI | Non-inferiority (CI low ≥ −0.02) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| HotpotQA | Headline, primary | Exact match | 7,405 | 0.462 | 0.187 | -0.275 | [-0.287, -0.263] | Does not meet |
| HotpotQA | Headline, secondary | Answer F1 | 7,405 | 0.604 | 0.266 | -0.338 | [-0.350, -0.327] | Not assessed |
| MuSiQue | Headline, primary | Answer exact match, alias-max | 2,417 | 0.182 | 0.008 | -0.174 | [-0.190, -0.158] | Does not meet |
| MuSiQue | Headline, secondary | Answer F1, alias-max | 2,417 | 0.265 | 0.026 | -0.240 | [-0.256, -0.223] | Not assessed |
| NIAH | Headline, primary | Exact-value accuracy | 1,000 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | Meets |
| Biography | Diagnostic | Exact match | 2,000 | 0.076 | 0.078 | +0.002 | [-0.001, +0.005] | Not assessed |

| Dataset | Strict whole-output format (full / Vanilla) | Answer extracted (full / Vanilla) | Reached 256-token cap (full / Vanilla) | Mean completion tokens (full / Vanilla) | Document markup echoed (full / Vanilla) | Request errors (full / Vanilla) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HotpotQA | 98.5% / 85.1% | 98.6% / 95.7% | 0.0% / 1.7% | 17.3 / 35.7 | 0.0% / 0.3% | 0 / 0 |
| MuSiQue | 98.6% / 9.8% | 98.8% / 82.5% | 0.2% / 12.5% | 25.7 / 129.4 | 0.0% / 0.5% | 0 / 0 |
| NIAH | 100.0% / 100.0% | 100.0% / 100.0% | 0.0% / 0.0% | 16.0 / 16.0 | 0.0% / 0.0% | 0 / 0 |
| Biography | 99.9% / 99.9% | 99.9% / 99.9% | 0.0% / 0.0% | 14.1 / 13.7 | 0.0% / 0.0% | 0 / 0 |

| NIAH cell | Pairs | Full prefill accuracy | Vanilla KV accuracy | Vanilla − full prefill |
| --- | ---: | ---: | ---: | ---: |
| 8k, needle at 10% | 112 | 1.000 | 1.000 | +0.000 |
| 8k, needle at 50% | 111 | 1.000 | 1.000 | +0.000 |
| 8k, needle at 90% | 111 | 1.000 | 1.000 | +0.000 |
| 16k, needle at 10% | 111 | 1.000 | 1.000 | +0.000 |
| 16k, needle at 50% | 111 | 1.000 | 1.000 | +0.000 |
| 16k, needle at 90% | 111 | 1.000 | 1.000 | +0.000 |
| 32k, needle at 10% | 111 | 1.000 | 1.000 | +0.000 |
| 32k, needle at 50% | 111 | 1.000 | 1.000 | +0.000 |
| 32k, needle at 90% | 111 | 1.000 | 1.000 | +0.000 |

| Biography reference | Pairs | Full prefill exact match | Vanilla KV exact match |
| --- | ---: | ---: | ---: |
| Had a WikiBio `-lrb- … -rrb-` disambiguator | 220 | 0.582 | 0.577 |
| Plain title | 1,780 | 0.013 | 0.016 |
<!-- cachet:vllm-0300-quality-v2:end -->

Vanilla KV does not meet the margin on either multi-hop dataset. HotpotQA
exact match falls from 0.462 to 0.187 and MuSiQue from 0.182 to 0.008, with
intervals far below −0.02. Vanilla prefills each document on its own, so no
document attends to another, and these are the two datasets whose answers
bridge facts across documents (HotpotQA examples carry up to 10 documents and
MuSiQue up to 20). The failures differ by dataset. Ignoring case, an
extracted answer counts as an abstention when it begins with the words
"unknown", "none", "do not know", "don't know", "I do not know", "I don't
know", "cannot", "can't", "it is not possible", "it is impossible", "it is
unclear", "not enough", "insufficient", or "there is no", or when it contains
"do not", "does not", or "did not" followed by provide, mention, specify,
state, contain, include, say, or indicate; "not" followed by provided,
mentioned, specified, stated, given, or available; or "no information" or
"no mention". By that rule, 4,177 of HotpotQA's 6,017
Vanilla misses (69%) are wrong answers, 1,519 (25%) abstain, and 321 (5%) have
no extractable answer. On MuSiQue, Vanilla mostly declines: of its 2,397
misses, 1,420 (59%) abstain, 423 (18%) have no extractable answer, and 554
(23%) are wrong answers; full prefill abstains on 729 (37%) of its own 1,976
misses. The split is approximate, because some answers deny the question's
premise ("not applicable", "there is a mistake") rather than decline. Only
9.8% of Vanilla's MuSiQue outputs meet the strict whole-output format, against
98.6% with full prefill, and they run about five times longer.

NIAH and Biography examples each carry exactly one document, so per-document
caching removes no cross-document attention there. Their results test whether
independently generated KV survives transfer, Q8 storage, and absolute-position
injection, not whether multi-document answers survive separate caching. On
that test NIAH single-needle retrieval is identical at every length and depth.
Both arms answer all 1,000 items correctly, so NIAH is at its ceiling and
cannot show a small loss.

The Biography split comes from its source data. WikiBio writes parentheses
as the tokens `-lrb-` and `-rrb-`, and the dataset masks each subject by
replacing every exact occurrence of the article title with "the subject". A
disambiguated title such as `name -lrb- occupation -rrb-` never occurs in the
text, so nothing is masked, and the title without its disambiguator still
appears as a whole name in 149 of these 220 examples (against 1 of 1,780 plain
titles). Nearly every disambiguated-title match comes from those visible
names: 127 of 128 with full prefill and 126 of 127 with Vanilla KV. Plain
titles match 24 and 29 times, 23 and 28 of them where the title does not
appear; the name can still appear in another form, such as with a middle
name. Both arms behave alike on this split.

## Ablation Tables

Every implemented ablation belongs to the governed vLLM 0.30.0 campaign.
Precision and hardware reuse the exact 16k, concurrency-4 core Vanilla control
from the same deployment block. Storage uses a dedicated matched Disk/RAM/Unity
Catalog trio per block: two examples per dataset, 32 repeats, 256 requests, and
concurrency 4.
Those two identities are the lowest domain-separated SHA-256 ranks within the
canonical 32-example dataset domain; callers cannot choose a favorable subset.

### Document KV Precision

<!-- cachet:vllm-0271-publication-table:precision:begin -->
| Document KV payload/runtime KV | P50 TTFT (s) | P50 TTC (s) | Status |
| --- | ---: | ---: | --- |
| Q8 / `fp8_e5m2` | 6.6893 | 26.5280 | Core 16k, concurrency-4 Vanilla anchor |
| bf16 / bf16 | 12.5096 | 36.4467 | Implemented five-block refresh cell |
| Packed Q4 | N/A (not implemented) | N/A (not implemented) | N/A (not implemented); no packed-Q4 serving contract |
<!-- cachet:vllm-0271-publication-table:precision:end -->

### Storage Tier

<!-- cachet:vllm-0271-publication-table:storage:begin -->
| Storage tier | P50 TTFT (s) | P50 TTC (s) | Status |
| --- | ---: | ---: | --- |
| Local NVMe disk | 6.7933 | 26.5914 | Five-block control; cold reads attested, page-cache residency unmeasured |
| RAM | 4.0985 | 23.9506 | Prewarmed 16-GiB payload cache; priming also warmed the serving process |
| Unity Catalog mounted path | 11.5882 | 31.3847 | OS eviction requested, residency unmeasured; backend cache unproven |
| Hybrid RAM/disk/Unity Catalog | N/A (not implemented) | N/A (not implemented) | N/A (not implemented); combined serving policy unsupported |
<!-- cachet:vllm-0271-publication-table:storage:end -->

### Hardware

<!-- cachet:vllm-0271-publication-table:hardware:begin -->
| Hardware | P50 TTFT (s) | P50 TTC (s) | Status |
| --- | ---: | ---: | --- |
| AWS g6/L4, `g6.8xlarge` | 6.6893 | 26.5280 | Core 16k, concurrency-4 Vanilla anchor |
| AWS g5/A10G, `g5.8xlarge` | 6.9493 | 24.8457 | Implemented five-block compatibility refresh cell; qualified vLLM `gpu_memory_utilization=0.85` against an A10G-only 1.75 GiB peak-headroom gate (other GPUs: 2 GiB) |
<!-- cachet:vllm-0271-publication-table:hardware:end -->

### Auxiliary Latency Re-run

The campaign's auxiliary rows above have three measurement defects, found in
review after the report was sealed:

- the latency collection joined jobs from 34 launch plans, so an auxiliary
  cell and the core reference it is divided by could come from different
  plans (an audit of the raw job records, which are not published, found
  that 19 of the 45 core pairs also mixed plans and that restricting the core
  estimands to same-plan pairs moves them by at most 0.05×);
- no arm ran a warmup except RAM, whose 16 priming requests also warmed the
  serving process, so RAM started warm while Disk and Unity Catalog started
  cold; and
- cold reads were attested without measuring page-cache residency, and the
  BF16 job ran at `gpu_memory_utilization=0.8` while its records said 0.9.

The re-run repeats the 16k, concurrency-4 core Vanilla reference and all five
auxiliary cells in each of five blocks: 30 jobs under one launch plan, one
wheel, and one source tree. Every arm runs the same four untimed, uniquely
cache-salted warmup requests before timing. The RAM arm still sends its 16
payload-cache priming requests first, because a populated payload cache is the
treatment it measures; after the shared warmup, every arm starts timing with a
warm serving process. Cold reads are attested only when
`mincore` reports zero resident pages after eviction; this measures the local
page cache only, so the Unity Catalog backend cache remains unproven. The BF16
runtime records the 0.8 setting it runs at. The estimands use the campaign's paired
crossed bootstrap, with each contrast divided by its own re-run control. The
sealed record is
[`appendix/vllm-0300-auxiliary-latency-rerun/auxiliary-latency-rerun.json`](appendix/vllm-0300-auxiliary-latency-rerun/).

Files went missing from the campaign volume while the re-run was being
prepared. Sixteen latency input files were rebuilt and matched their recorded
SHA-256 digests byte for byte. The restored 16k Q8 handoff bundle could not be
rebuilt that way, so every Q8 cell in the re-run serves a 16k Q8 bundle
regenerated on L4. This is the same regeneration path that produced the
campaign's L4-regenerated bundles (served to all fifteen 8k jobs, six 32k jobs,
and two BF16 jobs): the same 128 files with the same payload sizes, but not
byte-identical payloads. A first launch of the re-run, stopped
when the bundle loss surfaced, is superseded and none of its results are used.

<!-- cachet:vllm-0300-auxiliary-latency-rerun:begin -->
Re-run `closed_record_sha256`: `18a51fc96e21af35960e6532fb2b0533d89279f22912b9bb5a9c04ce842af201`; execution plan `219c45b80decdc37b8c66f41379d0430d85bc5825c6809badb67e68ee0c1153a`. All 30 jobs share one campaign, source revision and tree, package wheel, runtime artifact closure, qualification pins and campaign sources, each runs its cell's frozen runtime, and each serves the handoff its plan binds to it. Every compared pair runs the same input files and request schedule and differs only in the treatment's design (the RAM payload cache and the Unity Catalog handoff stage, each with its cache telemetry policy; the BF16 handoff bundle, KV dtype and GPU memory utilization; the A10G node and GPU memory utilization), its zone, and its per-job output and stage paths. 1 of them ran on a later attempt of the same plan after every earlier attempt ended without a result: `block-04-hardware-a10g` (attempt 2; attempt 1 ended `FAILED`).

| Treatment vs reference | Setting | TTFT speedup, campaign | TTFT 95% CI, campaign | TTFT speedup, re-run | TTFT 95% CI, re-run | TTC speedup, campaign | TTC 95% CI, campaign | TTC speedup, re-run | TTC 95% CI, re-run |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 payload/runtime KV vs Q8 | 16k, concurrency 4, L4/NVMe | 0.5388 | [0.5221, 0.5557] | 0.5495 | [0.5307, 0.5842] | 0.7282 | [0.7207, 0.7359] | 0.7346 | [0.7295, 0.7418] |
| RAM vs Disk | 16k, concurrency 4, L4 | 1.6704 | [1.6523, 1.6989] | 1.6958 | [1.6661, 1.7275] | 1.1132 | [1.1085, 1.1185] | 1.1162 | [1.1112, 1.1206] |
| Unity Catalog vs Disk | 16k, concurrency 4, L4 | 0.5754 | [0.5585, 0.5927] | 0.6027 | [0.5877, 0.6183] | 0.8410 | [0.8337, 0.8483] | 0.8587 | [0.8520, 0.8653] |
| A10G vs L4 | 16k, concurrency 4, local NVMe | 0.9556 | [0.9356, 0.9677] | 0.9508 | [0.9209, 1.0087] | 1.0671 | [1.0584, 1.0746] | 1.0670 | [1.0564, 1.0786] |

| Re-run cell | Node | GPU memory utilization | P50 TTFT (s) | P95 TTFT (s) | P50 TTC (s) | Connector loads | Cold-read attestations | Payload-cache hits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16k-c4-vanilla | `g6.8xlarge` | 0.9 | 6.4916 | 7.5494 | 26.4159 | 1280 | 1280 | 0 |
| storage-disk | `g6.8xlarge` | 0.9 | 6.8776 | 7.2199 | 26.8575 | 1280 | 1280 | 0 |
| storage-ram | `g6.8xlarge` | 0.9 | 4.1039 | 4.1368 | 24.0584 | 1280 | 0 | 1280 |
| storage-uc | `g6.8xlarge` | 0.9 | 11.4368 | 12.4040 | 31.2418 | 1280 | 1280 | 0 |
| precision-bf16 | `g6.8xlarge` | 0.8 | 12.1343 | 12.3122 | 36.1443 | 1280 | 1280 | 0 |
| hardware-a10g | `g5.8xlarge` | 0.85 | 7.0004 | 7.1971 | 24.9018 | 1280 | 1280 | 0 |
<!-- cachet:vllm-0300-auxiliary-latency-rerun:end -->

The re-run reproduces every campaign contrast closely: the four TTFT speedups
move by at most 0.03× and the TTC speedups by at most 0.02×, so the mixed
plans, uneven warmup, and unmeasured cold reads did not materially change the
campaign's auxiliary rows. RAM remains the only faster storage tier (1.70×
TTFT and 1.12× TTC against Disk) and Unity Catalog stays slower (0.60× and
0.86×). BF16 payloads, twice the bytes of Q8, take about 1.8 times as long to
first token (0.55×). The A10G node matches the L4 within noise on TTFT (0.95×,
interval 0.92–1.01) and completes requests about 7% faster end to end.

### Serving Platform

<!-- cachet:vllm-0271-publication-table:platform:begin -->
| Serving platform | Result | Status |
| --- | --- | --- |
| vLLM 0.30.0 | Governed campaign published | Report-bound integrity gate passed (no quality threshold) |
| SGLang | N/A (Q8 pre-RoPE serving path not implemented) | Unsupported for the main campaign |
<!-- cachet:vllm-0271-publication-table:platform:end -->

### Resource And Cache Telemetry

The final report renders all 23 governed descriptive cells with aggregate GPU,
host, process-tree, connector-load, backend-read, cold-read, eviction,
mounted-path, payload-cache, and storage-materialization telemetry.

<!-- cachet:vllm-0271-publication-table:resource-cache:begin -->
| Cell | Mean GPU util (%) | Peak GPU util (%) | GPU util samples | Peak GPU process memory | Peak host memory | Peak process-tree RSS | Connector loads | Backend bytes read / expected | Cold-read attestations | Evictions requested / succeeded | Mounted-path loads | Payload-cache hits / misses | Storage materializations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline 8k c1 | 43.07 | 100.00 | 20282 | 0.000 GiB (0 bytes) | 29.745 GiB (31,938,039,808 bytes) | 3.779 GiB (4,057,989,120 bytes) | 0 | 0.000 GiB (0 bytes) / 0.000 GiB (0 bytes) | 0 | 0 / 0 | 0 | 0 / 0 | 0 |
| Vanilla&nbsp;KV 8k c1 | 30.84 | 96.00 | 18253 | 0.000 GiB (0 bytes) | 29.970 GiB (32,180,011,008 bytes) | 4.392 GiB (4,716,294,144 bytes) | 1280 | 716.323 GiB (769,145,978,880 bytes) / 716.323 GiB (769,145,978,880 bytes) | 1280 | 1280 / 1280 | 0 | 0 / 0 | 0 |
| Baseline 8k c2 | 93.03 | 100.00 | 13855 | 0.000 GiB (0 bytes) | 29.196 GiB (31,349,207,040 bytes) | 3.793 GiB (4,073,091,072 bytes) | 0 | 0.000 GiB (0 bytes) / 0.000 GiB (0 bytes) | 0 | 0 / 0 | 0 | 0 / 0 | 0 |
| Vanilla&nbsp;KV 8k c2 | 83.60 | 100.00 | 11803 | 0.000 GiB (0 bytes) | 30.266 GiB (32,497,545,216 bytes) | 4.428 GiB (4,754,821,120 bytes) | 1280 | 716.323 GiB (769,145,978,880 bytes) / 716.323 GiB (769,145,978,880 bytes) | 1280 | 1280 / 1280 | 0 | 0 / 0 | 0 |
| Baseline 8k c4 | 99.87 | 100.00 | 8718 | 0.000 GiB (0 bytes) | 29.936 GiB (32,143,495,168 bytes) | 3.806 GiB (4,087,037,952 bytes) | 0 | 0.000 GiB (0 bytes) / 0.000 GiB (0 bytes) | 0 | 0 / 0 | 0 | 0 / 0 | 0 |
| Vanilla&nbsp;KV 8k c4 | 86.23 | 100.00 | 6499 | 0.000 GiB (0 bytes) | 29.376 GiB (31,542,415,360 bytes) | 4.445 GiB (4,772,397,056 bytes) | 1280 | 716.323 GiB (769,145,978,880 bytes) / 716.323 GiB (769,145,978,880 bytes) | 1280 | 1280 / 1280 | 0 | 0 / 0 | 0 |
| Baseline 16k c1 | 59.42 | 100.00 | 26638 | 0.000 GiB (0 bytes) | 29.011 GiB (31,150,710,784 bytes) | 3.793 GiB (4,072,595,456 bytes) | 0 | 0.000 GiB (0 bytes) / 0.000 GiB (0 bytes) | 0 | 0 / 0 | 0 | 0 / 0 | 0 |
| Vanilla&nbsp;KV 16k c1 | 34.56 | 100.00 | 19424 | 0.000 GiB (0 bytes) | 30.471 GiB (32,717,807,616 bytes) | 4.981 GiB (5,348,839,424 bytes) | 1280 | 1436.323 GiB (1,542,240,092,160 bytes) / 1436.323 GiB (1,542,240,092,160 bytes) | 1280 | 1280 / 1280 | 0 | 0 / 0 | 0 |
| Baseline 16k c2 | 99.90 | 100.00 | 20570 | 0.000 GiB (0 bytes) | 30.300 GiB (32,533,970,944 bytes) | 3.849 GiB (4,132,691,968 bytes) | 0 | 0.000 GiB (0 bytes) / 0.000 GiB (0 bytes) | 0 | 0 / 0 | 0 | 0 / 0 | 0 |
| Vanilla&nbsp;KV 16k c2 | 87.34 | 100.00 | 13224 | 0.000 GiB (0 bytes) | 31.540 GiB (33,865,486,336 bytes) | 5.000 GiB (5,368,848,384 bytes) | 1280 | 1436.323 GiB (1,542,240,092,160 bytes) / 1436.323 GiB (1,542,240,092,160 bytes) | 1280 | 1280 / 1280 | 0 | 0 / 0 | 0 |
| Baseline 16k c4 | 99.97 | 100.00 | 15759 | 0.000 GiB (0 bytes) | 27.820 GiB (29,871,738,880 bytes) | 3.826 GiB (4,107,673,600 bytes) | 0 | 0.000 GiB (0 bytes) / 0.000 GiB (0 bytes) | 0 | 0 / 0 | 0 | 0 / 0 | 0 |
| Vanilla&nbsp;KV 16k c4 | 80.45 | 100.00 | 8489 | 0.000 GiB (0 bytes) | 31.208 GiB (33,509,347,328 bytes) | 5.019 GiB (5,389,053,952 bytes) | 1280 | 1436.323 GiB (1,542,240,092,160 bytes) / 1436.323 GiB (1,542,240,092,160 bytes) | 1280 | 1280 / 1280 | 0 | 0 / 0 | 0 |
| Baseline 32k c1 | 80.89 | 100.00 | 48945 | 0.000 GiB (0 bytes) | 30.607 GiB (32,864,153,600 bytes) | 3.831 GiB (4,113,100,800 bytes) | 0 | 0.000 GiB (0 bytes) / 0.000 GiB (0 bytes) | 0 | 0 / 0 | 0 | 0 / 0 | 0 |
| Vanilla&nbsp;KV 32k c1 | 41.00 | 100.00 | 21143 | 0.000 GiB (0 bytes) | 31.377 GiB (33,690,509,312 bytes) | 6.112 GiB (6,562,750,464 bytes) | 1280 | 2876.323 GiB (3,088,428,318,720 bytes) / 2876.323 GiB (3,088,428,318,720 bytes) | 1280 | 1280 / 1280 | 0 | 0 / 0 | 0 |
| Baseline 32k c2 | 99.93 | 100.00 | 44441 | 0.000 GiB (0 bytes) | 32.578 GiB (34,980,614,144 bytes) | 3.842 GiB (4,125,671,424 bytes) | 0 | 0.000 GiB (0 bytes) / 0.000 GiB (0 bytes) | 0 | 0 / 0 | 0 | 0 / 0 | 0 |
| Vanilla&nbsp;KV 32k c2 | 80.16 | 100.00 | 17335 | 0.000 GiB (0 bytes) | 31.176 GiB (33,475,280,896 bytes) | 6.118 GiB (6,569,119,744 bytes) | 1280 | 2876.323 GiB (3,088,428,318,720 bytes) / 2876.323 GiB (3,088,428,318,720 bytes) | 1280 | 1280 / 1280 | 0 | 0 / 0 | 0 |
| Baseline 32k c4 | 99.98 | 100.00 | 39586 | 0.000 GiB (0 bytes) | 31.912 GiB (34,265,657,344 bytes) | 3.839 GiB (4,121,980,928 bytes) | 0 | 0.000 GiB (0 bytes) / 0.000 GiB (0 bytes) | 0 | 0 / 0 | 0 | 0 / 0 | 0 |
| Vanilla&nbsp;KV 32k c4 | 72.80 | 100.00 | 12226 | 0.000 GiB (0 bytes) | 30.982 GiB (33,266,434,048 bytes) | 6.154 GiB (6,607,933,440 bytes) | 1280 | 2876.323 GiB (3,088,428,318,720 bytes) / 2876.323 GiB (3,088,428,318,720 bytes) | 1280 | 1280 / 1280 | 0 | 0 / 0 | 0 |
| BF16 / bf16, 16k c4 L4/NVMe | 71.41 | 100.00 | 11655 | 0.000 GiB (0 bytes) | 31.108 GiB (33,401,925,632 bytes) | 6.114 GiB (6,565,130,240 bytes) | 1280 | 2872.646 GiB (3,084,480,184,320 bytes) / 2872.646 GiB (3,084,480,184,320 bytes) | 1280 | 1280 / 1280 | 0 | 0 / 0 | 0 |
| Disk, 16k c4 L4 | 79.87 | 100.00 | 8530 | 0.000 GiB (0 bytes) | 31.375 GiB (33,688,494,080 bytes) | 5.021 GiB (5,390,913,536 bytes) | 1280 | 1436.089 GiB (1,541,988,679,680 bytes) / 1436.089 GiB (1,541,988,679,680 bytes) | 1280 | 1280 / 1280 | 0 | 0 / 0 | 0 |
| RAM, 16k c4 L4 | 88.59 | 100.00 | 7662 | 0.000 GiB (0 bytes) | 38.060 GiB (40,866,623,488 bytes) | 12.840 GiB (13,786,701,824 bytes) | 1280 | 0.000 GiB (0 bytes) / 0.000 GiB (0 bytes) | 0 | 0 / 0 | 0 | 1280 / 0 | 0 |
| Unity Catalog, 16k c4 L4 | 67.09 | 100.00 | 10153 | 0.000 GiB (0 bytes) | 32.150 GiB (34,520,612,864 bytes) | 5.019 GiB (5,388,795,904 bytes) | 1280 | 1436.089 GiB (1,541,988,679,680 bytes) / 1436.089 GiB (1,541,988,679,680 bytes) | 1280 | 1280 / 1280 | 1280 | 0 / 0 | 0 |
| A10G, 16k c4 local NVMe | 58.23 | 100.00 | 7957 | 0.000 GiB (0 bytes) | 30.515 GiB (32,765,194,240 bytes) | 5.010 GiB (5,379,813,376 bytes) | 1280 | 1436.323 GiB (1,542,240,092,160 bytes) / 1436.323 GiB (1,542,240,092,160 bytes) | 1280 | 1280 / 1280 | 0 | 0 / 0 | 0 |
<!-- cachet:vllm-0271-publication-table:resource-cache:end -->

## Evidence And Publication Gate

The appendix accepts only the exact sanitized, content-addressed report/gate
pair under `appendix/vllm-0271-publication-v1/`. Raw Jobs API responses,
credentials, wheels, logs, generated datasets, prompt payloads, and local
scratch output do not belong here.

The frozen vLLM 0.27.1 design required all of the following:

- the frozen campaign source and wheel identity;
- one closed campaign record and exact input provenance;
- qualified L4/A10G/L40S runtime artifacts as applicable;
- 16 independent Q8 and 16 independent BF16 no-retry L40S
  handoff-generation attestations;
- 23 attempt-zero, no-retry CPU coordinator attestations (two Q8/BF16 tree
  closers, one latency-source closer, and 20 full-score ready/evidence closers),
  each proving zero GPU-ledger mutation;
- matched fresh-cluster latency execution records for all planned cells;
- the corrected complete paired score execution record;
- resource, cache-state, and Databricks control-plane attestations; and
- one sealed, sanitized campaign report that reaggregates both branches,
  binds the exact 115-job latency closure and 160-shard full-score closure,
  and projects the published tables; and
- its exact passing `document_kv.benchmark_publication_gate.v1` pair, whose
  `benchmark_payload_digest` equals the campaign report's
  `closed_record_sha256`.

The vLLM 0.30.0 finalizer that produced the published
`cachet.vllm_0300_publication_report.v1` replays the latency collection, GPU
qualification, and final full-score consumer authorities. It does not replay
the L40S handoff-generation, Q8/BF16 tree-closure, or latency-source-closure
attestations: after the campaign
volume lost those trees, each latency job record instead carries its own
handoff provenance (distributed L40S generation, a closed BF16 bundle, a
byte-identical restored bundle, or an L4-regenerated bundle), and the
collection verifies it per job.

The isolated one-arm latency jobs are component evidence under the `smoke`
policy. They make no standalone comparative claim. Publication is authorized
only by the final campaign-level report/gate pair after all Baseline and Vanilla
components have been recomputed and the shared GPU ledger closes with no active
reservation and the protected 124-hour headroom intact.

The quality v2 report and the auxiliary latency re-run record are sealed
separately, each with its own `closed_record_sha256`, in
`appendix/vllm-0300-quality-v2/` and
`appendix/vllm-0300-auxiliary-latency-rerun/`. They do not alter the campaign
report or gate. Their README regions above are rendered from those records.

[`verify_sealed_records.py`](verify_sealed_records.py) is a standard-library
checker: it recomputes each record's `closed_record_sha256`, checks the pinned
record and rendered-appendix bytes, confirms that the gate binds the report,
and compares every governed README region byte-for-byte with its sealed
rendering. The renderers and the full record validators run at the
`vllm-0300-campaign-archive` tag.

## Directory Layout

| Folder | Purpose |
| --- | --- |
| [`appendix/`](appendix/) | Sealed campaign report/gate pair, the quality v2 report, and the auxiliary latency re-run record |
| [`databricks/`](databricks/) | Databricks evidence-handling policy; no run results are currently published |
| [`native-engine/`](native-engine/) | Native integration scope; not a latency result folder |
| [`sglang/`](sglang/) | Explicit unsupported/pending SGLang benchmark status |
| [`storage/`](storage/) | Storage-ablation campaign status |
| [`vllm/`](vllm/) | vLLM 0.30.0 campaign index |
| [`_template/`](_template/) | Required report shape for future sanitized evidence |
