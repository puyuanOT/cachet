# Runtime KV Offload Probe

This folder contains the tiny Databricks runner used to verify Cachet's
platform-native runtime KV offload launch config and hierarchical document-KV
persistence behavior from an installed wheel.

The runner installs the supplied `cachet-kv` wheel, then delegates to
`document_kv_cache.runtime_kv_offload_probe`. It writes a compact JSON evidence
record containing:

- vLLM native `OffloadingConnector` launch config shape;
- SGLang HiCache launch arguments;
- synthetic Cachet document-KV hierarchy reads across cold storage, CPU RAM,
  and local-disk promotion/eviction;
- optional runtime package/import checks when strict flags are enabled.

Example:

```bash
databricks fs cp runner.py dbfs:/benchmarks/cachet/runtime-kv-offload/runner.py --overwrite

databricks jobs submit --json '{
  "run_name": "cachet-runtime-kv-offload-probe",
  "tasks": [
    {
      "task_key": "runtime_kv_offload_probe",
      "new_cluster": {
        "spark_version": "15.4.x-gpu-ml-scala2.12",
        "node_type_id": "g5.8xlarge",
        "driver_node_type_id": "g5.8xlarge",
        "num_workers": 0,
        "spark_conf": {
          "spark.master": "local[*]",
          "spark.databricks.cluster.profile": "singleNode"
        },
        "custom_tags": {
          "ResourceClass": "SingleNode",
          "purpose": "cachet-runtime-kv-offload-probe"
        },
        "data_security_mode": "SINGLE_USER",
        "single_user_name": "user@example.com"
      },
      "spark_python_task": {
        "python_file": "dbfs:/benchmarks/cachet/runtime-kv-offload/runner.py",
        "parameters": [
          "--package-wheel-uri",
          "dbfs:/benchmarks/cachet/runtime-kv-offload/cachet_kv-0.2.0-py3-none-any.whl",
          "--work-dir",
          "/local_disk0/cachet-runtime-kv-offload-probe",
          "--output-json",
          "/dbfs/benchmarks/cachet/runtime-kv-offload/probe.json"
        ]
      }
    }
  ]
}'
```
