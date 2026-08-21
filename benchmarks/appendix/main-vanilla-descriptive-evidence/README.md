# Main Vanilla descriptive evidence

This folder contains the compact, sanitized evidence behind the current
Q4-weight/Q8-KV benchmark tables. The measurements are useful descriptive
engineering results and **nonpublication evidence**. The main
Baseline-versus-Vanilla comparison does not pass Cachet's canonical canary
gate.

The Baseline and Vanilla jobs were deliberately isolated on fresh single-node
clusters. That makes the raw Baseline record structurally fail the repository's
generic canary gate: it contains no cache arm, and run-level resource telemetry
is hash-bound outside the per-arm resource schema. The table pairs therefore
remain descriptive even though their inputs, runtime settings, source, wheel,
results, and external telemetry were strictly validated. This folder does not
claim a canonical canary or publication-gate pass.

## Frozen protocol

| Field | Value |
| --- | --- |
| Model / revision | `Qwen/Qwen3-4B-Instruct-2507` / `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Weight / KV precision | 4-bit bitsandbytes weights; Q8 (`fp8_e5m2`) document and runtime KV unless an ablation says otherwise |
| Engine | vLLM `0.23.0`, `TRITON_ATTN` |
| Logical inputs | Biography, HotpotQA, MusiQue, NIAH; 2 examples per dataset |
| Repetition / concurrency | 32 repeats per example; 256 requests per job; request parallelism 4 |
| Decode | Forced 256 tokens, temperature 0, per-request prefix-cache salt |
| Main hardware | AWS g6/L4, `g6.8xlarge`; isolated single-node jobs |
| Main storage boundary | Local-NVMe-to-GPU hydration occurs inside Vanilla TTFT; OS page-cache eviction is requested per load |
| Source revision | `38919a6b64681d647868696ccf7d6b736ec29e2b` |
| Wheel SHA-256 | `74038cef655805add05688e307aa92596e2b6236d949d001e94964534af3e9af` |
| Input bundle SHA-256 | `832c1e4fbb8371d93ee7eb08d3b727fffbcbbb14831e88c996a3bd9e30250896` |
| Input provenance SHA-256 | `6ebf23b59ce994b69355504c71a3214863abf7c7d2bd356733364996c43f363d` |
| Closed input record SHA-256 | `10ead870911804c7b41b6b083f84d1d680377bc9d5f5e8fd2de06e8787784afd` |
| Sanitized record | [`evidence.json`](evidence.json) |
| Evidence JSON SHA-256 | `3306f9a17680b9f6689dffa5880f14c80dbb339976e7a08458eb4bf69fecab27` |

The eight logical example identities are reused across 8k, 16k, and 32k.
Main method pairs are matched within each context length. The 8k and 16k jobs
use `gpu_memory_utilization=0.90`; the 32k pair uses `0.70`, so cross-context
latency trends are descriptive rather than an identical-memory-configuration
scaling experiment.

## Scope and limitations

- Main latency percentiles contain 256 successful request-level measurements
  per isolated job. P50 decode throughput is the median per-request
  `completion_tokens / (TTC - TTFT)` value.
- RAM storage results use a 16 GiB prewarmed payload cache, 16 premeasurement
  requests (8 populate + 8 verify), 256 measured payload-cache hits, and zero
  measured backend reads.
- Unity Catalog uses a mounted UC path. OS page-cache eviction was requested and
  succeeded, but the backend cache state is unproven; it is not labeled strict
  cold-UC evidence.
- The hardware comparison is descriptive because the g6/L4 and g5/A10G node
  types have different local-disk topologies in addition to different GPUs.
- Precision is a coupled end-to-end document-payload **and runtime-KV**
  comparison. It is not a payload-only ablation.
- The separate score diagnostic uses five examples per dataset and natural EOS.
  It is not substituted into the full-dataset score table and supports no
  accuracy, significance, superiority, or publication claim.
