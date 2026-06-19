# Engine Probe Serving Profile Metadata

This PR-evidence sidecar covers the release-gate slice that ties native engine
probe artifacts to the pinned serving-engine profiles used by the package.

The slice adds runner-owned metadata for the expected serving package and
package version, exports those constants from the public package, and makes
release evidence reject probe records whose metadata or runtime-reported engine
version does not match the pinned vLLM/SGLang profile.

