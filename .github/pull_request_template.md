# What Changed

<!-- Summarize the concrete code, documentation, benchmark, or packaging changes. -->

# Why

<!-- Explain which project requirement this PR advances and why the change belongs in this slice. -->

# Scope

<!-- Name the touched boundaries, such as storage, materialization, model layout, vLLM/SGLang integration, benchmark, docs, or packaging. -->

# Verification

<!-- List exact tests, builds, benchmark runs, or explain why a benchmark is not relevant. -->

# AI Review

## Refactor Skill Evidence

<!-- Note how the Refactor skill was applied, especially what boundary was kept stable. -->

## GPT-5.5 Review Evidence

<!-- Paste the GPT-5.5 findings and the fixes made, or state that the review returned no findings. -->

<!-- Attach the JSON sidecar emitted by document-kv-pr-evidence. -->

- [ ] Refactor skill applied during this PR slice.
- [ ] GPT-5.5 review completed.
- [ ] Review findings were fixed, or the review was clean.

# Checklist

- [ ] Public API changes are documented.
- [ ] Every new folder has a README or package docstring.
- [ ] Storage, manifest, or KV-layout format changes include regression tests.
- [ ] Serving changes stay inside established engine integrations such as vLLM/SGLang, with no proprietary scheduler or custom solver.
- [ ] Benchmark or serving changes compare against the no-cache prefill baseline when relevant.
- [ ] Generated artifacts are excluded from the commit.
