# Release Manifest Wheel Identity

This PR-evidence sidecar covers the release-bundle manifest slice that records
package wheel identity in addition to copied artifact checksums.

The release manifest now includes the normalized package name and package
version for `package_wheel` artifacts, making the tested Cachet wheel auditable
from the handoff manifest without reopening the wheel archive.
