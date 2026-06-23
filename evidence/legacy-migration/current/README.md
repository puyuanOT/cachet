# Current Legacy Migration Evidence

This directory contains the current generated
`document_kv.legacy_compatibility_migration.v1` evidence for the remaining
source-only legacy compatibility cleanup gate.

The evidence is generated from `legacy-migration-scan-config.json`, which scans
the release, benchmark, storage, native-probe, and smoke runner surfaces that
feed the current Databricks QA evidence. The generated record confirms those
checked runners use Cachet or `document_kv_cache` surfaces and do not reference
the legacy import package or legacy console-script prefix.

Files:

- `legacy-migration-scan-config.json`: scan inputs and evidence URIs.
- `legacy-migration-evidence.json`: generated migration evidence.
- `legacy-migration-validation.json`: validation summary for the generated
  evidence record.

This is not the benchmark report directory. Human-readable benchmark summaries
and sanitized benchmark JSON live under `benchmarks/`, starting with
`benchmarks/databricks/CURRENT.md`; the Databricks run-specific folders there
are the durable standalone benchmark artifacts.

Regenerate from the repository root with:

```bash
python -m document_kv_cache.legacy_compatibility \
  --scan-config-json evidence/legacy-migration/current/legacy-migration-scan-config.json \
  --scan-base-dir . \
  --output-json evidence/legacy-migration/current/legacy-migration-evidence.json
```
