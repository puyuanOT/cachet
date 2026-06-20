# Strict PR Evidence Validation Summaries

This PR-evidence sidecar covers the directory-validation hardening slice for
PR-evidence validation-summary records.

The slice makes recursive PR-evidence directory validation skip only clean
closed-schema validation summaries, so raw or debug-augmented summaries are
reported instead of silently ignored.
