Immutable Request Chunk Maps

This PR makes document and legacy request chunk maps stable after construction.
Caller-owned dicts and lists are copied into `FrozenDocumentChunkMap` with tuple
chunk ids, preserving JSON/copy/pickle compatibility while rejecting mutation.
