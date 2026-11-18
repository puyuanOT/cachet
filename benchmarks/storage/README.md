# Storage Benchmark Index

The campaign was designed and frozen for vLLM 0.27.1 and ran on vLLM 0.30.0.
The vLLM 0.27.1 campaign defines the implemented Vanilla KV storage comparison
as a separate matched Disk/RAM/Unity Catalog trio in each of five deployment
blocks. Every cell uses the same capacity-safe 16k schedule: two
examples per dataset, 32 repeats, 256 requests, and concurrency 4. Disk is a
fresh local-NVMe control; the campaign attested its cold reads by successful
eviction requests without measuring page-cache residency. The RAM cell requires
proof that a 16-GiB provider payload cache was populated and verified before
all measured hits; those priming requests also warmed the serving process. The
Unity Catalog cell requires proof of mounted-path loads, successful OS-eviction
requests (again without measured residency), and exact bytes while retaining
the honest `backend cache unproven` label. The combined hybrid policy remains
unsupported.

The campaign's RAM cell was primed while Disk and Unity Catalog ran no warmup,
so the trio was re-run with a uniform untimed warmup and cold reads measured
against the local page cache (the Unity Catalog backend cache remains
unproven); see the
[Auxiliary Latency Re-run](../README.md#auxiliary-latency-re-run).

No prior storage number is carried forward. See the [benchmark root](../) for
the canonical current campaign state and governed table.
