# README Model Profile Definition Example

This PR clarifies the public model-expansion recipe for future Qwen3.5,
MiniMax, and adapter-trained KV Packet style integrations. The README example
now uses a schematic MQA-style `KVModelProfile`, wraps it in a portable
`ModelProfileDefinition`, registers it with `with_definition`, and writes the
profile JSON artifact that external model bundles can ship.

The example explicitly says real integrations must replace the illustrative
geometry with measured values from the target model.

Verification:

- `poetry run pytest tests/test_project_governance.py::test_readme_model_profile_example_uses_portable_definition_artifact tests/test_model_profiles.py -q`
- `poetry run pytest tests/test_project_governance.py tests/test_public_package.py tests/test_model_profiles.py -q`
- `python -m compileall -q src/document_kv_cache src/restaurant_kv_serving`
- `git diff --check`
- `poetry run pytest -q`

GPT-5.5 reviewed the diff, ran the focused governance test, executed the README
code fence in a temporary directory, and approved with no findings.
