# Terminal Databricks Status Evidence

This folder contains PR evidence for hardening release-oriented Databricks
run-status sidecars. The change rejects successful terminal status records that
still report an active task pointer, keeping compact release evidence aligned
with the terminal run state emitted by `databricks_runs get --summary`.
