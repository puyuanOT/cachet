# Strict V1 engine-probe completeness

This PR makes strict V1 release-bundle mode explicitly require both native
vLLM and SGLang engine-probe sidecars. The change keeps release publishing
aligned with the documented complete artifact set and reports missing probe
artifacts before bundle copying starts.

The PR evidence JSON is finalized after GPT-5.5 review completes.
