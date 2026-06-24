# Validate Measurement Quality Booleans

This PR tightens strict V1 release evidence for raw benchmark measurements.

Benchmark records emitted by the runner include both `exact_match` and `answer_found` quality flags on each measurement. Release evidence already validated `answer_found` when present; this slice makes the contract symmetric by validating `exact_match` the same way.

The validation remains optional at the raw measurement level so records with `None` quality flags, or older records where summary rows carry the quality evidence, do not fail solely because the per-measurement flag is absent.
