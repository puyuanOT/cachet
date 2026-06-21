# Validate Storage Result Volume

This PR tightens release evidence checks for storage benchmark rows. Each reader
row must now match the benchmark trace fields:

- `total_reads == chunk_count * repeats`
- `total_bytes == chunk_count * chunk_bytes * repeats`
- row `parallelism ==` the storage benchmark `parallelism`

The focused regression test mutates memory, disk, and Unity Catalog rows
independently so inconsistent hand-written or stale storage artifacts cannot pass
the release gate.
