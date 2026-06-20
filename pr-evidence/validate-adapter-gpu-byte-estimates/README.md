Validate Adapter GPU Byte Estimates

This PR hardens direct serving-adapter dataclass construction so invalid
`estimated_gpu_bytes` values are rejected consistently with serialized engine
handoff records.
