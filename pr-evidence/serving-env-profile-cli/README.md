# Serving Environment Profile CLI

This PR-evidence sidecar covers the serving-environment diagnostics slice that
makes pinned vLLM/SGLang install profiles available as standalone JSON records.

The slice adds `document-kv-serving-env` and legacy `restaurant-kv-serving-env`
entry points, plus `python -m` execution for both namespaces, so release jobs can
capture the isolated serving dependency contract next to native probe and
release-bundle evidence.
