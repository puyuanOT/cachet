# Test Fixtures

This directory contains immutable compatibility fixtures used by contract tests.

- `engine_adapter_handoff_v2.json` is a literal legacy handoff record. Tests use
  it only to prove that schema-v2 raw-KV records require explicit compatibility
  opt-in; new writers emit the current strict schema.

- `vllm_0271_runtime_closure.json` is the immutable package/lock metadata
  fixture pinned by the runtime parser. It contains no workspace paths,
  credentials, prompts, measurements, or GPU qualification success. Tests
  verify its original file/record hashes and reject changed bytes without
  requiring the workstation's ignored artifact directory.

- `publication_campaign_pre_mixed_sentinel.json` is the original vLLM 0.27.1
  predecessor protocol and cost-provenance compatibility fixture. Its original
  byte count and hashes are checked when testing preservation of campaign
  history. It contains no prompts, answers, result measurements, credentials,
  or workspace artifact locations and is not current publication evidence.
