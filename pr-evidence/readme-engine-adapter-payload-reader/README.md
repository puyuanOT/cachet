# README Engine Adapter Payload Reader

This PR makes the public serving-engine handoff recipe use the package-owned
`read_engine_adapter_payload` helper instead of an undefined placeholder. The
README now shows connector authors how to validate the handoff record, load the
external payload bytes, view merged or segmented payload spans, and derive the
vLLM/SGLang connector actions from public `document_kv_cache` APIs.

Verification:

- `poetry run pytest tests/test_project_governance.py::test_readme_engine_adapter_handoff_example_uses_public_payload_reader tests/test_project_governance.py::test_readme_documents_cachet_brand_and_scope -q`
- `poetry run pytest tests/test_project_governance.py tests/test_public_package.py tests/test_engine_probe.py tests/test_engine_adapters.py -q`
- `python -m compileall -q src/document_kv_cache src/restaurant_kv_serving`
- `git diff --check`
- `poetry run pytest -q`

GPT-5.5 reviewed the diff, ran the focused README governance test, checked the
README imports against public exports, and approved with no findings.
