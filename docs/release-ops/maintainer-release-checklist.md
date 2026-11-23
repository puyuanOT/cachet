# Maintainer Release Checklist

This checklist is for Cachet maintainers. It is not required knowledge for
external contributors opening issues or ordinary pull requests.

Before a public release:

- confirm the full test suite passes;
- confirm repository hygiene is clean;
- refresh dependency freshness evidence when dependency pins change;
- refresh benchmark evidence when runtime behavior, model layout, or connector
  behavior changes;
- validate release evidence and strict release bundles;
- generate the release PR evidence sidecar with `cachet-pr-evidence` and pass
  it to the strict release bundle;
- confirm GitHub governance and branch protection settings are restored;
- publish release notes and attach any large audit artifacts outside the source
  tree when appropriate.

Internal PR workflow gates such as Refactor-skill evidence and model review
belong here and in the release PR evidence sidecar, not in the beginner
contribution path.

