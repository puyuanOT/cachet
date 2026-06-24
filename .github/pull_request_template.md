# What Changed

<!-- Summarize the concrete code, documentation, benchmark, or packaging changes. -->

# Why

<!-- Explain the user problem, bug, or project goal this PR advances. -->

# Scope

<!-- Name the touched boundaries, such as docs, examples, storage, materialization, vLLM/SGLang integration, benchmarks, or packaging. -->

# Verification

<!-- List exact tests, builds, example runs, or explain why a benchmark is not relevant. -->

# Contributor Checklist

- [ ] Public API changes are documented.
- [ ] New examples or user-visible docs are runnable or clearly marked as illustrative.
- [ ] Every new folder has a README or package docstring.
- [ ] Storage, manifest, or KV-layout format changes include regression tests.
- [ ] Serving changes stay inside established engine integrations such as vLLM/SGLang, with no proprietary scheduler or custom solver.
- [ ] Benchmark or serving changes compare against the no-cache prefill baseline when relevant.
- [ ] Generated artifacts, credentials, logs, and local run output are excluded from the commit.

# Maintainer Notes

<!-- Maintainers may attach release-audit sidecars or internal review notes after the public PR is ready. -->
