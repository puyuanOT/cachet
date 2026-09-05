# Storage Benchmark Index

The vLLM 0.27.1 campaign defines the implemented Vanilla KV storage comparison
as a separate matched Disk/RAM/Unity Catalog trio in each of five deployment
blocks. Every cell uses the same capacity-safe 16k schedule: two
examples per dataset, 32 repeats, 256 requests, and concurrency 4. Disk is a
fresh strict-cold local-NVMe control. The RAM cell requires proof that a 16-GiB
provider payload cache was populated and verified before all measured hits. The
Unity Catalog cell requires proof of mounted-path loads, successful OS-eviction
requests, and exact bytes while retaining the honest `backend cache unproven`
label. The combined hybrid policy remains unsupported.

No prior storage number is carried forward. See the [benchmark root](../) for
the canonical current campaign state and governed table.
