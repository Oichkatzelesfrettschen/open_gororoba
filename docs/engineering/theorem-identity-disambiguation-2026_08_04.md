---
description: Typed theorem identities and explicit claim links for the canonical control plane
last_verified: 2026-08-04
evidence_class: executed-control-plane-repair
---

# Theorem identity disambiguation

The control plane separates canonical claim IDs from Rocq theorem identities. A proof filename such as C1635_SedenionDriverSemantics.v retains its legacy name and path, but it does not imply claim C-1635.

Stable theorem IDs use the THM-* namespace. Canonical claims use the C-* namespace. theorem_claim_links records a relation only when the claim explicitly names the proof path in formal_proof, where_stated, or status_note. Numeric prefixes serve only as reservation signals for future claim allocation.

The four collision rows remain legacy theorem aliases until the binding specification creates fresh formal proposition claims. The new formal claims remain separate from the Ward and energy successor claims C-1635 through C-1638.

The validator rejects a numeric-prefix-only relation, rejects an unresolved claim-like theorem identity, checks proof paths, checks explicit relation parity with the generated theorem projection, and checks theorem evidence foreign keys. The successor allocator skips numeric identifiers reserved by theorem aliases and pending transition specifications.

The append-only theorem identity event records the binding specification hash, stable theorem IDs, and fresh claim IDs. Exact replay returns the original mapping. Changed content under an existing binding key fails.

Declare the identity phase complete only after the four formal propositions have explicit THM-* identities, fresh canonical claim rows, exact theorem links, resolved evidence references, regenerated views, and passing negative controls.
