---
description: retained local governance and Rust gate result for the evidence-ledger batch
last_verified: 2026-08-01
evidence_class: executed_gate
---

# Local governance gate evidence

The isolated evidence-ledger worktree passes the repository local gate after
the Markdown registry generators restore their authority and regeneration
headers. The gate runs against `origin/main`, detects 17 changed files, and
selects `gororoba_cli_data` as the direct Rust scope.

## Executed result

| Gate surface | Result | Observed evidence |
| --- | --- | --- |
| Cache and host routing | pass | 6 physical cores, worker budget 6, cache usage 4906 MB |
| Shared check suite | pass | ANSI check, terminology gate over 8767 files, and report-write policy |
| Rust clippy | pass | `gororoba_cli_data` direct scope with `-D warnings` |
| Rust nextest | pass | 19 library tests passed, nextest run `c8e02897-3c5d-4f73-8296-68f28abf92b8` |
| Markdown inventory | pass | registered=128, on_disk=128, owner entries=128 |
| Governance gate | pass | schema signatures, crossrefs, aliases, authority, headers, and removal policy |

## Broader registry acceptance boundary

The broader read-only acceptance target does not pass in this checkout. Its
first semantic-atoms verifier reports a canonical claim-set mismatch and a
proof-skeleton population of 1417 against 1448 canonical claims. The target
stops before the evidence-provenance, integrity-resolution, and
execution-planning verifiers. This result is retained as an existing registry
acceptance boundary; the scoped local gate above remains the gate used for this
batch.

| Command | Result | Observed evidence |
| --- | --- | --- |
| `make registry-acceptance-gate-readonly` | fail | `semantic-atoms`: claims_atoms claim set mismatch; proof_skeletons=1417 < canonical claims=1448 |

## Replay commands

Run from the repository root with the pinned toolchain and a worktree-local
target directory:

```bash
make gate-local
make governance-gate-readonly
```

The scoped Rust lane uses library tests by default. The CI lane must run the
full integration-test kind before treating this local result as a complete
workspace regression result.

## Closure boundary

This record closes the P0 local-gate substitute row for the current batch. It
does not claim that the broader registry acceptance target passes, or that
every historical experiment, proof axiom, or structural debt row is resolved.
Those boundaries remain in the active roadmap and the P1/P2 evidence records.
