# Harden V1 Evidence Identities

This PR tightens release-evidence validation for V1 benchmark records.

It makes malformed or duplicate benchmark identities explicit validation failures instead of letting them count toward required V1 coverage. The touched validator paths are:

- `measurements`: supported dataset, supported arm, and duplicate dataset/arm rows.
- `report_rows`: supported dataset, supported arm, and duplicate dataset/arm rows.
- `comparisons`: supported dataset and duplicate per-dataset comparisons.

The change is intentionally scoped to validation behavior for invalid evidence records. Existing valid release evidence remains accepted by the full release-evidence test suite.
