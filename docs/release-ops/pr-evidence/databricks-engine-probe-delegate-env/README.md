# Databricks engine-probe delegate env

This PR wires benchmark-plan and Databricks engine-probe target JSON through to
backend-specific native probe delegate environment variables. It lets managed
AWS g5 Databricks runs keep stable Cachet built-in vLLM/SGLang probe factory
paths while injecting downstream native adapter factories through the cluster
`spark_env_vars` expected by `document_kv_cache.native_probe_factories`.

Verification is recorded in `pr-evidence.json`. GPT-5.5 reviewer Gauss the 3rd
approved the branch with no findings.
