# PEP 621 Packaging Metadata

This PR-evidence sidecar covers the packaging modernization slice that moves
publish-facing metadata out of deprecated Poetry-only fields and into standard
PEP 621 sections.

The slice keeps Poetry as the build backend, leaves package include rules under
`[tool.poetry]`, and moves metadata, optional dependencies, and console scripts
to `[project]`, `[project.optional-dependencies]`, and `[project.scripts]`.

