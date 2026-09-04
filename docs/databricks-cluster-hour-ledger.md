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

The retained opening is now the exact 236-reservation, 98-receipt,
236-terminal prefix. It preserves the earlier 124/0/124 history, the 14/0/14
pre-run rejection, and the intermediate 138/0/138, 152/14/152, 166/28/166,
180/42/180, 194/56/194, 208/70/208, and 222/84/222 prefixes. The rejected plan SHA-256,
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
The pre-c0bede canonical ledger file SHA-256 is
`fd0b6774928f77166657c8d35652e4d557f6708552d88c7c6725fc42d7723e87`.

The seventh 14-job `SINGLE_USER` batch is retained as the split runtime
observation and opaque worker-subprocess failure provenance hop. Plan
`c0bede45ea211798c9a5eb31010a91074ded70e370f8ea4fcbeb59b3b9f95598`
is stored in the exact file
`fe59e32c44ab50f91bae5114a587268d44ebb9acfba74500aedb66158e2541b7`
and used reviewed runner
`ca93baeda09f3df050b0dad3b8f3091c0f74235c426bd66555b67bd4b6eeafbc`.
The exact frozen reason is: all fourteen hash-locked qualification runtimes
installed and verified; the two packed-page-roundtrip workers returned
measurements before post-success runtime observation rejected the
virtualenv-created runtime/bin/python symlink, while the other twelve
sentinel-worker subprocesses exited nonzero and the reviewed launcher did not
surface their captured child stdout/stderr, so their underlying worker causes
remain unknown. The two observer failures have path-normalized error SHA-256
`3662915979987aef1fe4bcf9e0e62f06c67992ee73da679e44f6b6a261e634f5`;
the twelve opaque worker-subprocess errors have normalized SHA-256
`3f1ddd73298cd46347cf57b84d6cf22f7d6e98802b50ded9457d7a999563786b`.
Every job's raw error SHA-256 and UTF-8 byte count is separately source-pinned.
The sealed evidence tree contains exactly 29 regular files and 1,828,218 bytes
and has SHA-256
`bb6636f3b9bdf5afae0b7d1beb97f5f3192017ba5b04abb651f2a389889aa57f`.
It closes in manifest
`6c4cca0ec4fbcf4ccb434573f965eeb8022909ce5bdd6afdf31d61085807fa9b`
whose exact file SHA-256 is
`53fd4b076a642101790d21ebbc03b1eb7e609428c2ccd7eafb8cbad5a9a3a112`.
Its 12,410.279 terminal cluster-seconds add 3.447299722222222 GPU-hours.
The predicted terminal prefix and final offline-reconciled prefix are both the
exact 222/84/222 prefix
`22ac65492fa0871f528552cfcae0bd6332b1429cd9fc2e92c373c5e534202d4a`.
The canonical ledger file SHA-256 is
`38677fff866e0a7268398c4b616b4be968df3a8191381db74ebd8fcb71af50ef`.
The pre-c0bede campaign file
`c805c303a92dba3fdd0390699c757974c1f738ebc4c553bb651618cb27bf8056`
and its closed record
`1f1682a99e69ad691dfab68a85cc9555eff4daea437d5095d93410af2430c490`
remain immutable provenance. The reconciled opening balance is therefore
67.930336 GPU-hours, with zero active reservations and 956.069664 hours
remaining under the 1,024-hour aggregate cap.

The eighth 14-job `SINGLE_USER` batch is retained as the mixed sentinel and
result-validation failure provenance hop. Plan
`694441bffc253141156f9c808666112d39bb5829d22825d1d88c93ab47a5e830`
is stored in the exact file
`e19e9b173ad8e2705d11cfbd637aa3702a98e37d827e7d1489460c1462c5a649`
and used reviewed runner
`ca93baeda09f3df050b0dad3b8f3091c0f74235c426bd66555b67bd4b6eeafbc`.
All fourteen parent runs reached `INTERNAL_ERROR`/`FAILED`; their sole
attempt-zero tasks reached `TERMINATED`/`FAILED` without repairs. The exact
failure split is two post-measurement vLLM version-contract mismatches, two
forced-runtime-handoff unresolved-native-object failures, eight sentinel
layout-conflict failures, and two auto-backend FlashInfer `array.array`
TypeError engine-initialization failures. Every raw error SHA-256 and UTF-8 byte
count is source-pinned, and all four job sets are disjoint and exhaustive.

The five-key `runs/get-output` records all have `logs_truncated=false`. Their
sealed evidence tree contains exactly 29 regular files and 2,094,892 bytes with
SHA-256
`7455fa1e30356bb79ccb75a8dbe24df32f33a365141505e0270eb13c7f39b71d`.
It closes in manifest
`13ad4eabd10bde1b5c7e0aa7b9721dd3bd8fbe57f6c20204093749df8d84954f`,
whose exact file SHA-256 is
`a6e0c985d64b0072776dd1247094600d81b885cfe4a3fb0f6418e8b811134304`.
The submit closure separately binds 16 files and 22,468 bytes with tree SHA-256
`81817e833e6878ff5bfd45fff2a94ffafb341d7acecc6e4f7212d268646f8f72`,
including phase lease `ea9f9ec3c415001ef0cf65e9d3673950f7a49d7a2733eb76bf2668bdf7d80344`
and batch marker `b4f2cb8ea2c1f637c31b3745217670622657845f3a893ea2e7a5f78025c125d8`.
The exact ledger lineage is 222/84/222
`22ac65492fa0871f528552cfcae0bd6332b1429cd9fc2e92c373c5e534202d4a`,
236/84/222
`92cf13248ab854e5e1c789d94ac60c20fa77c2a9bf67c83d326ef0bef5603de4`,
236/98/222
`7c83650851e5b169adb85961226745d3082fecc9ae9c007ee84606f7b1329b07`,
then 236/98/236
`07b9663e42c2dd8040f689d08fabdd6d7eefaf25f8f1decedc23af683e0011c7`.
The 12,455.252 terminal cluster-seconds add 3.459792222222222 GPU-hours.
The canonical ledger file SHA-256 is
`784a43eafec2f6d6086b4258959b308043e183f361218463be14dea3702bd62d`.
The predecessor campaign remains immutable at file SHA-256
`eb306f9a8be50730bfef81121c2a83ebec7e50e89386addb7f77ce6001bcd85f`
and closed record
`5f90b531b30ac6f4b29e0151d688a005b0377b205ca39645376d7d43aef5e305`.
The refreshed campaign closes as
`2d35875107c709d71e6f558d2a029afb53ee371d851e83a18fe0d194f6fc0e0c`
in canonical file
`353b8b3e77eca5347901232709a40c45a0f996be4fc6f25ed55511d38457dc85`.
The immutable campaign-opening snapshot records 71.390128 GPU-hours, zero active
reservations, and 952.609872 hours remaining under the 1,024-hour aggregate cap
at the exact 236/98/236 prefix. It is historical plan authority, not a claim
about the live ledger; every successor stage and final publication record must
bind its own later ledger prefix.

The reviewed qualification-v2 opening extends that same ordered ledger to
265/127/265 at prefix
`e3aaca37d5e01cbb5060800ef2e3e115e048fc35c7e1ae74539d0085c7b5c8e1`
and 77.50443361111115 terminal GPU-hours. The latest reconciled quiescent GPU
predecessor in this repository extends it again to 279/141/279 at prefix
`7bdfab96021910df7a06ac1cf87604eefe7c1f4181f49a242212f699c443ca1a`
and 81.16875222222224 terminal GPU-hours. These are immutable successor
snapshots, not permission to reuse a stale controller freeze; a new freeze must
bind the then-current complete predecessor.

The later native-v2 14-job batch is retained as a cross-hardware raw-byte
identity gate failure. Its plan is
`7f8b82e271794501a86f61134f73c47519d9fa2d7f1d3d1202ee5b10e0d3653a`,
its source closure is
`59f2d25e4cb75adc13189ecd9c8617e5c29eb1e965ef8360e44cb32ce1dbfc47`,
and its phase batch is
`7efac847c9bfe3f42d9a2bf289035a7c10454276fb350aeb727b37034b0c1a6d`.
All fourteen attempt-zero, no-retry jobs reached terminal success. Their
3.5296394444444443 GPU-hours closed the exact 428/290/428 ledger prefix
`7e0d3fedba07a6ed0f7dd4ef23d4f0c82912626043586e54547a59016a195222`
at 115.43377555555554 terminal GPU-hours. Terminal job success did not
constitute global qualification success: collection correctly failed closed
with `ValueError: L4 and L40S generation artifacts are not byte-identical`.
The collector failure record closes as
`e103ca7dc7bbb630404013873a0c9a7909d6ac8f052ed5acf42050dd106930da`;
its exact file SHA-256 is
`81d597ef57c85d109cad7f19c23da849566a39e581dd48d3113a49622dc1926b`.

The L4 generation result closes as
`9d120142ff462d92b86ccd694c31acb0b19690dd682185d196b8bcae22c20bc4`
and measured 922.8792245248601 tokens/s. The L40S result closes as
`8e64f0f98fe88f97391cb182ed9bc60229ee9cf8798762856a938c4375d01f7c`
and measured 949.9742346119579 tokens/s. Zero of twelve corresponding raw
artifact SHA-256 values matched across L4 and L40S, while all twelve records
matched after omitting only `raw_artifact_sha256`: dataset/example identity,
input target, cache-prefix token count and token digest, segment order/counts
and token digests, and raw byte length were equal. The native-v2 envelope had
inherited the legacy-v1 aggregate byte-identity rule, so this batch remains
`FAILED`. It must not be relabeled, resealed as passing evidence, or admitted
under a successor contract.

Two subsequent one-job repeats are retained only as diagnostic,
non-authorizing evidence; they neither repair that failed batch nor grant
publication qualification. L40S run `506950471100618` reproduced all twelve
original L40S raw artifact digests from an independent fresh load and measured
946.4833749604393 tokens/s. Its result closes as
`26616b95cfe5efab2527e28eade5fad9dedb48f12e76c7eb41e5bd7f4ab081ca`,
its terminal evidence closes as
`51c72161bb2dff2d359ce03419a47f7415a7d1bd1659a2d6444450eba717bcdd`,
and its 1,204.212 terminal seconds add 0.3345033333333333 GPU-hours at the
exact 429/291/429 prefix
`2ef0f0d30b40b164fd157627ed55ae881b15e28fdb9143ea35a3b599b334feb4`.
L4 run `71783401971590` likewise reproduced all twelve original L4 raw
artifact digests and measured 920.6639135510926 tokens/s. Its result closes as
`e78d2e5bc7b6cdc9e217f7d6cb821bbd763e8b7e290d1037d4a95b4308bbd2cf`,
its terminal evidence closes as
`19e8a6a6c5bbfeb8736d8fe61204262a6a367bf37a30cae0e6d7b15e5c26a9ab`,
and its 1,271.03 terminal seconds add 0.3530638888888889 GPU-hours. The final
diagnostic prefix is the exact 430/292/430 prefix
`116251d3ca5fce37ce5749565e1059fdf65b30ce17fd12ebc50b877835f9772b`,
with 116.12134277777776 terminal GPU-hours, zero active reservations, and
907.8786572222223 hours remaining under the cap. These diagnostics support a
fresh repeat-aware native-v2 protocol; they cannot be combined with the old
fourteen results to manufacture a pass.

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
terminal control-plane record. Charged time is the sum of their one-GPU
terminal lifecycles, including bootstrap, generation, hashing, and durable
writes. A CPU coordinator then verifies every file and closes the shared-root
bundles using metadata renames, with zero payload copies and zero coordinator
GPU charge. L40S is the sole publication handoff generator. Fresh native-v2
eligibility requires same-hardware fresh-load byte reproducibility separately
on L4 and L40S, cross-hardware logical/token/layout/size equivalence without
raw-digest equality, exact raw artifact length of 73,728 bytes per cache-prefix
token, and L40S throughput of at least 35 tokens/GPU-second. Timed L4/A10G
serving jobs stage the closed 8k/16k/32k bundle they need to node-local NVMe and
never regenerate it.

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
campaign-opening snapshot records 71.390128 terminal GPU-hours. At that
historical prefix, this left only 250.264554 hours inside the protected 900-hour
envelope for qualification, timed latency, and full-score consumers at that
generation threshold.

The BF16 prerequisite closes 308,448,018,432 payload bytes (287.264603 GiB);
its conservative absolute 16k slot envelope is 288 GiB. The full-score program
is frozen to 83,653 examples, 160 shards, 63,455,746 cache-prefix tokens, and
66,448,937 natural-prompt tokens. Its content-addressed inventory, shard plan,
and execution plan are part of the campaign record, so a sampled or reordered
subcampaign cannot spend against the publication budget.
