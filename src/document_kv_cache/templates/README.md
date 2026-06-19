# Packaged Templates

This folder contains template resources shipped inside the `document-kv-cache`
wheel. They are packaged as importable resources, not executable runtime
modules; the `document-kv-templates` CLI lists, shows, and extracts these files
for teams that want repository-free Databricks job scaffolding.

- `databricks/` contains Databricks Asset Bundle templates and their local
  usage notes.

Keep this tree workspace-neutral. Template files must not embed workspace URLs,
tokens, catalog names, or user-specific upload paths.
