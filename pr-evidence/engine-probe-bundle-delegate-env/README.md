# Engine-probe bundle delegate env

This PR adds optional backend-specific native probe delegate variables to the
standalone Databricks Asset Bundle engine-probe template and its packaged copy.
The variables map to the environment names consumed by Cachet's built-in native
vLLM/SGLang probe factories so bundle users can run built-in stable factory
paths with downstream native block-manager adapters.

The PR evidence JSON is generated after GPT-5.5 review completes.
