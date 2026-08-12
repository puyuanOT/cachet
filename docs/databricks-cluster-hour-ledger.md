# Databricks Cluster-Hour Guard

Representative canaries use a persistent, credential-free ledger with a hard
aggregate cap of 120 cluster-hours. The fixed ten-job sequence reserves at most
40 hours on its first pass because every task is bounded to four hours. Manual
retry attempts are new reservations: a second complete pass reaches 80 hours,
and a third reaches 120 hours.

Generate the Databricks `runs/submit` payload first. Representative vLLM and
SGLang jobs require exactly 14,400 seconds at both the run and task levels and
task `max_retries: 0`. Generic engine-probe and storage jobs remain configurable
from 1 through 14,400 seconds, also with zero task retries.

Build and stage Cachet before generating a representative payload. The wheel is
content-addressed: compute the SHA-256 of the exact bytes, put that lowercase
64-character digest in the persistent Databricks path, and pass the same digest
with `--wheel-sha256`. Both representative job builders reject a missing digest,
a non-persistent URI, or a URI whose path does not contain the digest. Their
bootstrap runners hash the staged bytes before `pip install`; a mismatch stops
the task. For example:

```bash
python -m build --wheel
WHEEL_FILE=dist/cachet_kv-0.2.0-py3-none-any.whl
WHEEL_SHA256=$(shasum -a 256 "$WHEEL_FILE" | awk '{print $1}')
WHEEL_URI="dbfs:/cachet/wheels/${WHEEL_SHA256}/$(basename "$WHEEL_FILE")"
databricks fs cp "$WHEEL_FILE" "$WHEEL_URI" --overwrite
```

The job runner is also content-addressed. Its persistent URI must contain the
manifest's exact SHA-256 as a path component and use the platform-specific
canonical basename (`run_vllm_smoke.py` or `run_sglang_smoke.py`). A same-name
script at any other path is rejected. A complete SGLang representative
payload-generation example is:

```bash
RUNNER_SHA256=6e3b5cd79181828bcb515e210fea46e6aa75b7636c2a3bf8e19775f5026bc1de
RUNNER_URI="dbfs:/cachet/runners/${RUNNER_SHA256}/run_sglang_smoke.py"
cachet-sglang-smoke-databricks-job \
  --benchmark-id g6-sglang-4k-32-paired-smoke \
  --output-dir /Volumes/catalog/schema/volume/g6-sglang-4k-32-paired-smoke \
  --runner-python-file "$RUNNER_URI" \
  --runner-script-output .cachet/run_sglang_smoke.py \
  --run-timeout-seconds 14400 \
  --hardware-target aws-g6-l4 \
  --node-type-id g6.8xlarge \
  --single-user-name "$DATABRICKS_SINGLE_USER_NAME" \
  --wheel-uri "$WHEEL_URI" \
  --wheel-sha256 "$WHEEL_SHA256" \
  --model-revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --tokenizer-revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --representative-canary \
  --representative-workload-profile sglang-4k-32-v1 \
  --context-length 4096 \
  --max-tokens 32 \
  --mem-fraction-static 0.85 \
  --live-benchmark-repeats 2 \
  --sglang-attention-backend triton \
  --sglang-sampling-backend pytorch \
  --sglang-enable-deterministic-inference \
  --generate-live-handoff \
  --live-handoff-output-dir /local_disk0/g6-sglang-4k-32-paired-smoke/handoffs \
  --output-json .cachet/g6-sglang-4k-32-paired-smoke-submit.json
test "$(shasum -a 256 .cachet/run_sglang_smoke.py | awk '{print $1}')" = \
  "$RUNNER_SHA256"
databricks fs cp .cachet/run_sglang_smoke.py "$RUNNER_URI" --overwrite
```

The source tree tests bind each manifest runner digest directly to the UTF-8
bytes of its embedded runner-script constant, so a runner edit requires an
explicit manifest digest update. Stage the verified script before submission.
The coupled representative validator rejects extra runner flags, duplicate
singleton flags, alternate package-install sources, and hidden Spark environment
overrides; the verified wheel is the only Cachet install source.

The ordered first wave is closed and versioned:

| Order | Workload ID | Requirement | Profile | Platform / node | Arm |
| ---: | --- | --- | --- | --- | --- |
| 1 | `g6-vllm-8k-64-baseline` | required | `vllm-8k-64-v1` | vLLM / `g6.8xlarge` | baseline |
| 2 | `g6-vllm-8k-64-full-prefix` | required | `vllm-8k-64-v1` | vLLM / `g6.8xlarge` | full prefix |
| 3 | `g6-vllm-8k-64-vanilla` | required | `vllm-8k-64-v1` | vLLM / `g6.8xlarge` | vanilla per-document |
| 4 | `g6-vllm-16k-256-baseline` | required | `vllm-16k-256-v1` | vLLM / `g6.8xlarge` | baseline |
| 5 | `g6-vllm-16k-256-full-prefix` | required | `vllm-16k-256-v1` | vLLM / `g6.8xlarge` | full prefix |
| 6 | `g6-vllm-16k-256-vanilla` | required | `vllm-16k-256-v1` | vLLM / `g6.8xlarge` | vanilla per-document |
| 7 | `g6-sglang-4k-32-paired-smoke` | required | `sglang-4k-32-v1` | SGLang / `g6.8xlarge` | paired baseline/cache smoke |
| 8 | `g5-vllm-8k-64-baseline` | best effort | `vllm-8k-64-v1` | vLLM / `g5.8xlarge` | baseline |
| 9 | `g5-vllm-8k-64-full-prefix` | best effort | `vllm-8k-64-v1` | vLLM / `g5.8xlarge` | full prefix |
| 10 | `g5-vllm-8k-64-vanilla` | best effort | `vllm-8k-64-v1` | vLLM / `g5.8xlarge` | vanilla per-document |

All ten bind Qwen3-4B-Instruct revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. The vLLM jobs use exactly one
prepared HotpotQA input, BF16 model/KV, `max_num_seqs=2`, and GPU utilization
`0.85`; cache arms also bind their method-owned handoff topology under
`/local_disk0`. Serving dependency pins are part of the manifest contract.

Each isolated vLLM job keeps its unique workload ID, while the three arms for
one hardware/profile group share a comparison suite ID. The payload passes
Databricks' `{{task.run_id}}` dynamic value as the physical runtime ID, so
retries remain distinct without rewriting evidence. Representative provenance
also binds the exact node CPU/RAM/GPU/local-disk geometry, DBR runtime, cold
cache protocol, and the SHA-256 of the wheel that the bootstrap verified before
installation.

Initialize one ledger for the canary sequence:

```bash
cachet-databricks-resource-ledger init \
  --ledger-json .cachet/representative-canary-cluster-hours.json \
  --ledger-id representative-canary-2026-08
```

Representative submissions must use the coupled command. It reads the payload
once, reserves its worst-case task time under the ledger lock, and posts the
exact canonical bytes whose digest was reserved:

```bash
cachet-databricks-runs reserve-and-submit \
  --payload-json .cachet/vllm-baseline-submit.json \
  --ledger-json .cachet/representative-canary-cluster-hours.json \
  --attempt-id vllm-baseline-attempt-01 \
  --workload-id g6-vllm-8k-64-baseline \
  --representative-canary
```

Do not split a representative reservation and submission into two commands.
The `--representative-canary` path binds both the workload ID and payload to the
ordered ten-workload manifest. The coupled path prevents a payload-file edit
between accounting and the Jobs API request. An over-cap or invalid reservation
performs no POST. If the POST fails, its reservation deliberately remains active
until terminal information is available or the operator accounts for the failed
attempt.

The ledger persists the payload SHA-256, bounded task timeouts, and safe
identifiers—not the payload, credentials, workspace URL, response, or task
output. Attempt IDs are unique. A failed submission still consumes its
reservation until it is explicitly recorded terminal.

When Databricks reports a terminal state, record the aggregate actual cluster
duration. For a multi-task payload, sum the task cluster durations rather than
using the overall wall-clock interval:

```bash
cachet-databricks-resource-ledger terminal \
  --ledger-json .cachet/representative-canary-cluster-hours.json \
  --attempt-id vllm-baseline-attempt-01 \
  --terminal-state succeeded \
  --actual-cluster-duration-seconds 2712
```

Terminal reconciliation replaces that attempt's active worst-case reservation
with its actual duration. It never releases more than the difference between
the reservation and the recorded actual. Unknown attempts, duplicate terminal
records, durations above the reservation, malformed payloads, and reservations
that would exceed the cap fail closed.

The ledger uses closed JSON schemas, immutable event objects, an exclusive file
lock for read-modify-write operations, and atomic replacement. Keep the ledger
file with the canary evidence so repeated manual attempts share the same budget.
