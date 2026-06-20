Confine Disk Range Reader Root

This PR hardens disk-backed KV shard loading so relative shard paths under an
explicit root cannot traverse outside that root.
