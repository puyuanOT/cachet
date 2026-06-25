# SGLang Benchmarks

These folders track Cachet's SGLang status. SGLang integration now validates
Cachet-backed external KV hits through HiCache, but the current short-prompt
benchmarks do not show a speedup.

## Results

| Result | Model | Hardware | Method | What Passed | Performance |
| --- | --- | --- | --- | --- | --- |
| [`qwen3-4b-g6-l4-vanilla-kv-prepared/`](qwen3-4b-g6-l4-vanilla-kv-prepared/) | Qwen3 4B Instruct | AWS g6/L4, `g6.8xlarge` | Vanilla external KV through SGLang HiCache | 16 live measurements, 8/8 Cachet cache-hit validations, quality delta `0.0` | Cachet arm was slower on the short prepared prompts |
| [`qwen3-4b-g6-l4-vanilla-kv-synthetic-niah/`](qwen3-4b-g6-l4-vanilla-kv-synthetic-niah/) | Qwen3 4B Instruct | AWS g6/L4, `g6.8xlarge` | Vanilla external KV through SGLang HiCache | Two repeated Cachet-backed cache hits with 175 cached tokens and unchanged quality | Tiny-prompt scoped evidence; no speedup |

## Current Interpretation

SGLang support is a correctness and integration result today:

- Cachet can generate handoffs with SGLang HiCache page-key metadata.
- The live request metadata bridge reaches SGLang's HiCache storage path.
- Cachet-backed cache-arm requests report validated cached tokens.
- Quality gates pass on the prepared and synthetic successful runs.

It is not yet a performance win. Treat the SGLang results as proof that the
serving path works, not as a reason to expect lower latency on every prompt.

## Historical Runs

Failed and superseded SGLang smoke attempts are preserved under
[`../../docs/release-ops/benchmark-archive/sglang-smoke/`](../../docs/release-ops/benchmark-archive/sglang-smoke/).
They are useful for maintainers debugging SGLang integration, but they are not
public benchmark results.

## Method Roadmap

| Method | Status |
| --- | --- |
| Vanilla external KV cache | Integrated and benchmarked for correctness/cache hits |
| KV Packet | Planned; no public SGLang result yet |
| Adapter-trained or learned KV methods | Not benchmarked |
