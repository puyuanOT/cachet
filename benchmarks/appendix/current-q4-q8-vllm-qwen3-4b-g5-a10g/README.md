# Current Q4/Q8 vLLM Benchmark Evidence

This appendix records the current public benchmark protocol: Qwen3-4B-Instruct
with 4-bit model weights, Q8 document KV, shared GPU prefix references, local
disk handoff bundles, and vLLM on `g5.8xlarge`.

## Configuration

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen3-4B-Instruct-2507` served as `qwen3:4b-instruct` |
| Serving engine | vLLM `0.23.0` |
| Hardware | AWS g5/A10G, `g5.8xlarge` |
| Model weights | vLLM `--quantization bitsandbytes` |
| vLLM KV cache dtype | `fp8_e5m2` |
| Default Cachet document KV dtype | `fp8_e5m2` |
| Cachet residency | Local disk handoff bundles, prewarmed into vLLM prefix cache |
| Prefix sharing | `--enable-prefix-caching`, static `cache_salt`, prewarm before measurement |
| Request parallelism | 8 requests in flight |
| Output length | Forced 256 tokens with `ignore_eos=true` |
| Datasets | Biography, HotpotQA, MusiQue, NIAH; one prepared row per dataset, eight repeats |

## Current Latency And Resource Results

Latency values are seconds. Percentiles are computed over 32 successful
request-level measurements per complete row. `P50 decode tok/s` is computed as
`completion_tokens / (TTC - TTFT)`. `Max observed GPU KV-pool use` comes from
vLLM server-log telemetry over a 16.32 GiB KV pool; it is not a full
process-level peak GPU memory sample.

| Method | Input context | Document KV payload | P50 TTFT | P95 TTFT | P50 TTC (256 tokens) | P95 TTC (256 tokens) | P50 decode tok/s | Observed parallelism | Max observed GPU KV-pool use |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Baseline, no precomputed KV | 8k | N/A | 2.166 | 2.737 | 14.507 | 15.080 | 20.742 | 8 running / 0 waiting | 0.70 GiB (4.3%) |
| Baseline, no precomputed KV | 16k | N/A | 6.047 | 6.644 | 20.728 | 21.324 | 17.437 | 8 running / up to 7 waiting | 1.26 GiB (7.7%) |
| Baseline, no precomputed KV | 32k | N/A | 16.622 | 16.971 | 35.361 | 35.714 | 13.659 | 8 running / up to 7 waiting | 2.28 GiB (14.0%) |
| Cachet + vanilla KV | 8k | Q8 (`fp8_e5m2`) | 0.389 | 0.422 | 12.730 | 12.752 | 20.732 | 8 running / 0 waiting | 0.69 GiB (4.2%) |
| Cachet + vanilla KV | 16k | Q8 (`fp8_e5m2`) | 0.432 | 0.595 | 15.196 | 15.468 | 17.407 | 8 running / 0 waiting | 1.26 GiB (7.7%) |
| Cachet + vanilla KV | 32k | Q8 (`fp8_e5m2`) | 0.677 | 1.122 | 19.705 | 20.000 | 13.487 | 8 running / 0 waiting | 2.27 GiB (13.9%) |

Each completed row had 32 successful measurements and zero request errors. For
the Cachet Q8 rows, connector telemetry recorded four `load_request` events per
run, all from prewarm, and no Cachet KV loads during measured requests.

## Prepared Dataset Smoke Results

Each cell is `answer-found / strict EM`. These values are not official
Biography, HotpotQA, MusiQue, or NIAH benchmark scores. The suite has one
prepared row per dataset/context and eight repeats; answer-found is a
containment check, while strict EM is 0.00 because forced-256 outputs are
verbose.

| Method | Input context | Prepared examples per dataset | Repeats per example | Biography | HotpotQA | MusiQue | NIAH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, no precomputed KV | 8k | 1 | 8 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| Baseline, no precomputed KV | 16k | 1 | 8 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| Baseline, no precomputed KV | 32k | 1 | 8 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| Cachet + vanilla KV | 8k | 1 | 8 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| Cachet + vanilla KV | 16k | 1 | 8 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| Cachet + vanilla KV | 32k | 1 | 8 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |

## Document KV Precision Evidence

The bf16 document-KV ablation is a quality failure under the FP8 runtime KV
setting: the outputs did not contain the expected answers, so it is not used as
the default Cachet row.

| Document KV payload | Input context | P50 TTFT | P95 TTFT | P50 TTC (256 tokens) | P95 TTC (256 tokens) | P50 decode tok/s | Answer-found / strict EM | Cache footprint | Max observed GPU KV-pool use | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bf16 | 16k | 0.539 | 0.646 | 15.224 | 15.336 | 17.425 | 0.00 / 0.00 | 9.83 GB | 1.29 GiB (7.9%) | Quality failure under FP8 runtime KV |
| Q8 (`fp8_e5m2`) | 16k | 0.432 | 0.595 | 15.196 | 15.468 | 17.407 | 1.00 / 0.00 | 4.92 GB | 1.26 GiB (7.7%) | Default document KV precision |
| Q4 packed | 16k |  |  |  |  |  |  |  |  | Implementation pending |

## Databricks Provenance

| Row | Databricks parent run | Task run | DBFS output |
| --- | --- | --- | --- |
| Cachet Q8, 8k | `397306394664215` | `32308763644415` | `dbfs:/benchmarks/cachet/q4-q8-shared-current-20260627_174450/runs/8k-cachet-q4mat-prewarm-e5m2-current-limit1` |
| Cachet Q8, 32k | `353401098635093` | `509176024738855` | `dbfs:/benchmarks/cachet/q4-q8-shared-current-20260627_174450/runs/32k-cachet-q4mat-prewarm-e5m2-current-limit1` |
| Baseline, 8k | `777052874361660` | `1094055508701915` | `dbfs:/benchmarks/cachet/q4-q8-default-current-20260627_220945/runs/q4-q8-default-8k-baseline` |
| Baseline, 16k | `1108385130287829` | `1019661460470812` | `dbfs:/benchmarks/cachet/q4-q8-default-current-20260627_220945/runs/q4-q8-default-16k-baseline` |
| Baseline, 32k | `221269968259600` | `1081383537088909` | `dbfs:/benchmarks/cachet/q4-q8-default-current-20260627_220945/runs/q4-q8-default-32k-baseline` |
| Cachet Q8, 16k | `126174968860307` | `289332644191985` | `dbfs:/benchmarks/cachet/q4-q8-default-current-20260627_220945/runs/q4-q8-default-16k-cachet-q8` |
| Cachet bf16 document KV, 16k | `745434681832900` | `105007045561087` | `dbfs:/benchmarks/cachet/q4-q8-default-current-20260627_220945/runs/q4-q8-ablation-16k-cachet-bf16-doc-kv` |

The completed Q8 server logs report `load_format=bitsandbytes`,
`quantization=bitsandbytes`, `kv_cache_dtype=fp8_e5m2`,
`enable_prefix_caching=True`, 16.32 GiB available KV-cache memory, 237,728 GPU
KV-cache tokens, and 7.25x maximum concurrency for 32,768-token requests.

The 16k Q8 handoff payloads occupied 4.92 GB across the four prepared examples;
the 16k bf16 handoff payloads occupied 9.83 GB across the same four examples.
Peak GPU process memory, GPU utilization, CPU RSS, and host RAM were not sampled.

## Q4 Document KV Status

Packed-Q4 document KV is not included as a measured row yet. The current vLLM
integration can inject into FP8 runtime KV pages, but the Cachet payload layout
does not yet define a packed 4-bit document-KV format with dequantization or
native serving-engine Q4 KV pages.
