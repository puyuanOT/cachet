# PEP 561 Typed Package Markers

This PR-evidence sidecar covers the type-distribution slice that makes the
public document package and legacy compatibility package advertise inline type
annotations to downstream type checkers.

The slice adds `py.typed` markers to both packages, includes them in sdist and
wheel artifacts, documents installed-wheel typing, and verifies the metadata and
marker files through focused package/governance tests plus built-wheel
inspection.

