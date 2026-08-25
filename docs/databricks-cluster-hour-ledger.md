# Databricks Cluster-Hour Guard

The generic credential-free ledger supports an explicit hard aggregate cap of
up to 1,024 cluster-hours. Publication campaigns must set their approved cap in
the immutable ledger before the first reservation; an over-cap reservation
performs no submission. Active worst-case reservations are additionally capped
at 900 hours, preserving at least 124 hours of unreserved campaign headroom.
When an approved campaign raises a prior cap, use the fail-closed `raise-cap`
operation on the existing quiescent ledger. It preserves every reservation and
terminal actual; starting a new zero-balance ledger would incorrectly discard
already consumed GPU-hours.

Publication reservations must use the same ledger ID carried by the live GPU
qualification capability. While holding that ledger's lock, each workload also
keeps the total task count across all nonterminal reservations at or below the
campaign-wide 16-job concurrency cap.

Representative canaries retain a narrower, persistent hard aggregate cap of
120 cluster-hours. The fixed ten-job sequence reserves at most
40 hours on its first pass because every task is bounded to four hours. Manual
retry attempts are new reservations: a second complete pass reaches 80 hours,
and a third reaches 120 hours.

Generate the Databricks `runs/submit` payload first. Representative vLLM and
SGLang jobs require exactly 14,400 seconds at both the run and task levels and
task `max_retries: 0`. Generic engine-probe and storage jobs remain configurable
from 1 through 43,200 seconds, also with zero task retries; their individual
campaign contracts may impose narrower limits.

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

## vLLM 0.27.1 publication campaign

The publication campaign has a separate closed design record. Generate it
before producing any submit payloads:

```bash
python -m document_kv_cache.publication_campaign \
  --campaign-id vllm-0271-publication-v1 \
  --campaign-ledger-id representative-canary-823bd9d82a5c1730 \
  --campaign-ledger-json databricks-runs/vllm-0271-publication-prep/cluster-hours.json \
  --output-json .cachet/vllm-0271-publication-campaign.json
```

The command reads the migrated, quiescent ledger and closes its exact ordered
append-only prefix into the campaign record. Matching only the ledger ID is not
sufficient: a fresh ledger with the same ID would erase the retained opening
balance and is rejected. The record also binds a privacy-safe SHA-256 of the
canonical symlink-free ledger path, preventing two copied ledgers from funding
divergent branches. Every later phase must extend the immediately prior
authorized prefix and acquire one atomic whole-wave lease before its first
submission.

The retained opening is now the exact 208-reservation, 70-receipt,
208-terminal prefix. It preserves the earlier 124/0/124 history, the 14/0/14
pre-run rejection, and the intermediate 138/0/138, 152/14/152, 166/28/166,
180/42/180, and 194/56/194 prefixes. The rejected plan SHA-256,
rejection-evidence file SHA-256, HTTP 400 status, 18,292-byte observed
parameters JSON versus the 10,000-byte server limit, and zero observed active
runs remain bound in `analysis.opening_ledger_provenance`.

The first live 14-job qualification batch is a second explicit provenance hop.
All 14 submissions received run identities, but `NONE` data-security mode could
not resolve the Unity Catalog Volume bootstrap. Seven runs failed and the other
seven were canceled after those failures. The exact intermediate prefix, batch
plan SHA-256, direct `runs/get` reconciliation-manifest closure, 14/14/14
transition, terminal-state counts, and 1.599130277778 actual GPU-hours are bound
in the campaign record.

The subsequent 14-job `SINGLE_USER` batch is retained as a third explicit
provenance hop. Plan
`2cf4ef1092a435c1e713f2a94115021ea7069ab6295d18ce5fcb5d4a479ce997`
used reviewed runner
`f5ee833621428d630df1a59952a485d4ac55cabf987186d98a40274a2cf8a958`,
whose bootstrap referenced undefined `__file__` under Databricks
`spark_python_task` execution. All 14 runs were created and all 14 tasks failed
with `INTERNAL_ERROR` before package installation. Direct `runs/get` and
`runs/get-output` evidence closes as
`8c7623aa2618066ea0ccedcba1d35a340308da04aaa040f89364bc4ea3d1b71c`
in the exact manifest file
`1d0246ece1d6f844420d22a26b729d3f0d971ca0b30c0bf1ef0b5a84dcf6f360`.
Its 4,585.718 terminal cluster-seconds add 1.273810555556 GPU-hours.

The next 14-job `SINGLE_USER` batch is retained as a fourth provenance hop.
Plan
`d6f7619f6a70311fac571b31bedc7974e756a1679218cf63b76a7e7ceb91ebec`
used reviewed runner
`04cfe3a16200f011710317d829b7c52c0e4ca12f95fd8d277c949e7d6856d5b0`.
All 14 runs were created, but the qualification bootstrap could not resolve
Databricks cluster identity. Every task returned exactly
`RuntimeError: Databricks cluster identity is unavailable; expected
DATABRICKS_CLUSTER_ID or DB_CLUSTER_ID` and failed at attempt zero without a
repair. The five-key `runs/get-output` records and parent `runs/get` records
close in manifest
`fbb1fd4250b3fc62b58778047b12fe3775e6cffbc8641b38a00c721a9d4c768d`
whose exact file SHA-256 is
`06c527102283bb379ecb26a345e76467d7e1614771d9a3c8313e9ebe6d941cf9`.
Its 4,564.259 terminal cluster-seconds add 1.267849722222 GPU-hours. Offline
reconciliation produced the exact 180/42/180 prefix
`376114c27f35725bab5418969d28a77d4a3600dba44d049b597512142856d86f`
and canonical ledger file SHA-256
`f76cce3b68417f8d14a5e030d9eacaef3e61d17f123a2a2b5d38be5428a89b94`.

The fifth 14-job `SINGLE_USER` batch is retained as the runtime-lock-index
failure provenance hop. Plan
`f991036176d59df70f0e339be4eb4a67a7c03a51536f62bf440df1ac72fd0e33`
used the same reviewed runner
`04cfe3a16200f011710317d829b7c52c0e4ca12f95fd8d277c949e7d6856d5b0`.
All 14 tasks failed with `INTERNAL_ERROR` before package installation because
exactly `pip requirements-file index precedence omitted the PyTorch CU129
index and prevented hash-locked torch resolution`. Every complete
`runs/get-output` record uses exactly the five logged keys `error`,
`error_trace`, `logs`, `logs_truncated`, and `metadata`; the path-normalized
error SHA-256 is
`7544cab6366fc1813af8d04da00a8a1f76f1098e3b06c738d8ff8ddd392ae235`.
The sealed evidence tree contains exactly 29 regular files and 1,564,133 bytes
and has SHA-256
`5016ed50001b77b77f329e858c01b1a65c5e927f1c55eec7fbc01208d8f25886`.
It closes in manifest
`2ee650e0e05ea059bd9f552d6975149c05cbda6dc8d3a715a73594913f078b29`
whose exact file SHA-256 is
`e0f56f1250c4ce213d1a8ba0384ccdad1a1b38fb964c1b6bfcf5729006150455`.
Its 7,754.755 terminal cluster-seconds add 2.154098611111 GPU-hours. The
predicted terminal prefix and final offline-reconciled prefix are both the
exact 194/56/194 prefix
`381ed88dfca75a17cf11b09b7e3dedb435328e518e8f1f0f0d9591be27796f26`.
The predecessor ledger file SHA-256 is
`1ac7ee076d2a5aa3b12bfd18d3cb6f8843aa9f8f7b8e07686c519869985a6916`.

The sixth 14-job `SINGLE_USER` batch is retained as the site-packages-path
failure provenance hop. Plan
`be4cb0e80e17c99d9c4bd8abb89b24efb6e1202072fb734c739d322812218c9c`
is stored in the exact file
`c63521b29233addc1c5ab4435dfa0d639135765bce7a54298c0b0b1200741651`
and used reviewed runner
`ca93baeda09f3df050b0dad3b8f3091c0f74235c426bd66555b67bd4b6eeafbc`.
All fourteen hash-locked qualification runtimes installed and verified, then
failed before sentinel worker launch because the site-packages read-only
freezer rejected a nonexistent Debian local dist-packages scheme path reported
by `site.getsitepackages()`. The path-normalized error SHA-256 is
`8937fb907ae789c647754b2bbe9dbc4d9e167b67b8e437613260373b658c0da3`.
The sealed evidence tree contains exactly 29 regular files and 1,945,499 bytes
and has SHA-256
`2c555ea534fc3d41d3bc998fcaff8f07aedf42e1872200e39f9ed46796081607`.
It closes in manifest
`a685849f6446063bdd5b220cd3ac5218c6e49a1e2d8487acac36316537b35eb7`
whose exact file SHA-256 is
`2996e67b6c6305544c11231266500dcb9c53aa2bbc701fa6d6e626299c2ab06e`.
Its 11,498.35 terminal cluster-seconds add 3.193986111111 GPU-hours. The
predicted terminal prefix and final offline-reconciled prefix are both the
exact 208/70/208 prefix
`a71cee32c1ae056d7db7c72c70fa72bcf5622d8a3ae6d72590c4435bb9db4af9`.
The canonical ledger file SHA-256 is
`fd0b6774928f77166657c8d35652e4d557f6708552d88c7c6725fc42d7723e87`.
The reconciled opening balance is therefore 64.483036 GPU-hours, with zero
active reservations and 959.516964 hours remaining under the 1,024-hour
aggregate cap.

Every publication `runs/submit` payload also carries a package-derived,
64-character Databricks idempotency token bound to its attempt identity and
canonical payload bytes. A durable pre-POST claim prevents concurrent local
submission, while an accepted request whose response was lost may be recovered
only by replaying those exact bytes and token; Databricks then returns the same
run identity. Payload or token drift remains fail-closed.

The record freezes five deployment blocks, matched Baseline/Vanilla jobs at
8k/16k/32k and closed-loop concurrency 1/2/4, 32 examples per dataset, two
repeats, and 256 requests per cell. The core factorial contains 90 isolated
jobs. Twenty precision, RAM/UC, and A10G treatment jobs are joined by five fresh
Disk controls, for 115 latency jobs total. Each block's Disk/RAM/UC trio uses a
separate capacity-safe storage schedule with two examples per dataset and 32
repeats (256 requests); the main factorial remains 32 examples per dataset and
two repeats. Each storage trio launches in one matched wave; each 16k/c4 core
Baseline/Vanilla pair launches in the same wave as its BF16 and A10G treatments.
No job retries. Core timeout hours by c1/c2/c4 are 6/4/4 at 8k, 8/6/4 at 16k,
and 12/8/4 at 32k; every c4 auxiliary job has a four-hour timeout. These
condition-specific reservations replace the old universal four-hour assumption,
while the generic Jobs API validator permits at most 43,200 seconds. The same
record requires one complete paired score pass
per method over the four implemented datasets, without padding, truncation, or
an answer-quality preservation gate. It also binds the 1,024-hour aggregate
cap, 900-hour active-reservation guard, 124-hour headroom, 16-job parallel
limit, and the 35-token/s effective generation gate required before the
complete score campaign may launch.

The budget also charges the latency handoff build that precedes timed serving:
128 identities are generated once at each of 8k, 16k, and 32k. The exact frozen
cache-prefix workload is 7,323,967 generated tokens (7,340,032 conservative
input-token slots), so the 35-token/GPU-s gate permits at most 58.1268 actual
GPU-hours. Sixteen separately submitted, no-retry L40S producer jobs reserve at
most 80 GPU-hours (five hours each). The exact 16-member wave is reserved
atomically before any POST, then each attempt is reconciled from its own
terminal control-plane record. Charged time is the sum of their one-GPU terminal lifecycles,
including bootstrap, generation, hashing, and durable writes. A CPU coordinator
then verifies every file and closes the shared-root bundles using metadata
renames, with zero payload copies and zero coordinator GPU charge. L40S is
eligible only after the sealed L4/L40S artifact-equivalence and >=35-token/s
qualification passes. Timed L4/A10G serving jobs stage the closed 8k/16k/32k
bundle they need to node-local NVMe and never regenerate it.

The auxiliary precision arm has a separate prerequisite with the same launch
discipline: 128 exact 16k rows are sharded across 16 independent no-retry L40S
jobs, each bounded to five hours (80 reserved GPU-hours for the wave). Its BF16
pre-RoPE Qwen3 layout uses 147,456 logical payload bytes per cached prefix
token. The plan-bound resource estimator reports the actual token-derived
storage and GPU-hour estimates; the absolute 16k input-slot envelope is 288 GiB
of payload data before small JSON metadata (actual cache prefixes are smaller),
and the measured ledger charge and durable byte count replace those estimates
after closure. Publication use requires all 16 direct `runs/get` terminal
attestations, unique parent/task/cluster IDs, attempt zero with no repair, and a
content-addressed manifest that stages to node-local NVMe without regeneration.

Large durable trees are verified where they are mounted; they are never mirrored
to the Mac controller. The closed control plane therefore adds exactly 23
single-node, no-retry `c5d.4xlarge` CPU jobs on
`15.4.x-cpu-ml-scala2.12`: two 12-hour-bounded Q8/BF16 handoff closers, one
two-hour-bounded latency-source closer, and 20 two-hour-bounded full-score
ready/evidence closers (producer plus consumer for each of ten waves). Their
66 CPU-node-hour timeout envelope is recorded separately: all jobs declare zero
GPU tasks, do not reserve or mutate the 1,024-hour GPU ledger, and must preserve
the immediately preceding GPU-ledger prefix byte-for-byte. The Mac accepts only
compact issuer capabilities after direct attempt-zero `runs/get` evidence and
authenticated, bounded Unity Catalog Files API transport.

The closed budget inventory is intentionally more explicit than a single
timeout sum:

| Phase | Frozen workload | Timeout reservation upper bound |
| --- | ---: | ---: |
| GPU qualification | 14 independent jobs | 56 GPU-hours |
| Latency Q8 handoff generation | 7,323,967 cache-prefix tokens; 16 producers | 80 GPU-hours |
| Latency BF16 handoff generation | 2,091,797 cache-prefix tokens; 16 producers | 80 GPU-hours |
| Timed latency/resource campaign | 65 jobs at 4h, 20 at 6h, 20 at 8h, 10 at 12h | 660 GPU-hours |
| Full-score Q8 producer phases | 160 shards in ten 16-task phases | 960 GPU-hours |
| Full-score paired consumer phases | 160 shards in ten 16-task phases | 960 GPU-hours |
| Remote closure control plane | 23 single-node CPU jobs | 66 CPU-node-hours; 0 GPU-hours |

The timeout upper bounds are safety envelopes, not a claim that all phases can
consume them. Qualification, handoff generation, and timed latency are admitted
wave by wave from reconciled terminal actuals and hard cap/headroom checks. The
governed live-P90 projection applies to each producer and consumer phase of
every nonzero-indexed full-score wave (waves 1–9); both wave-zero phases use
hard cap/headroom admission. The three generation workloads contain 72,871,510
cache-prefix tokens; at the required 35 effective
tokens/GPU-second their combined allowance is 578.345317 GPU-hours. The
retained ledger opens the reset with 64.483036 terminal GPU-hours, leaving only
257.171646 hours inside the protected 900-hour envelope for qualification,
timed latency, and full-score consumers at that generation threshold.

The BF16 prerequisite closes 308,448,018,432 payload bytes (287.264603 GiB);
its conservative absolute 16k slot envelope is 288 GiB. The full-score program
is frozen to 83,653 examples, 160 shards, 63,455,746 cache-prefix tokens, and
66,448,937 natural-prompt tokens. Its content-addressed inventory, shard plan,
and execution plan are part of the campaign record, so a sampled or reordered
subcampaign cannot spend against the publication budget.
