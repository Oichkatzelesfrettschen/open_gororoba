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

## Prior registry acceptance boundary

The pre-refresh read-only acceptance target stopped at semantic-atoms because
the generated claim and proof-skeleton lanes were stale. It reported 1417
proof skeletons against 1448 canonical claims. This historical result remains
retained as the reason for the generated-lane reconciliation in this batch.

## Current registry and full-audit result

The 2026-08-01 reconciliation regenerates the semantic, evidence-provenance,
integrity-resolution, and execution-planning views from the canonical SQLite
control plane. The registry acceptance target now passes.

| Gate surface | Result | Observed evidence |
| --- | --- | --- |
| Semantic atoms | pass | claims=1448, edges=5370, equations=450, symbols=870, proof_skeletons=1454 |
| Evidence provenance | pass | derivations=4358, bibliography_normalized=442, provenance_records=276, narrative_paragraphs=1992 |
| Integrity resolution | pass | conflict markers=108, lacunae=186 |
| Execution planning | pass | experiments=232, lineages=232, edges=1113, workstreams=13, todo=65, actions=38 |
| Registry acceptance | pass | semantic atoms, evidence provenance, integrity resolution, execution planning, crossrefs, aliases, inventory, and owner-map checks exit zero |

The full keep-going audit writes `reports/gates/2026-08-01/212938/`. The
registry step and workspace check pass. The Rust regression step remains open:
6104 tests run with 6102 passed, 2 failed, and 48 skipped. Both failures are
missing-input failures for these retained intake paths:

| Missing input | Failing tests | Intake condition |
| --- | --- | --- |
| `data/external/pdg_2025/mass_subset.csv` | `c068_subset_match_uses_full_eigenlevel_count` | Admit the CSV with source provenance and a content hash |
| `data/external/nanograv_15yr_freespectrum.csv` | `c070_quantile_curve_has_expected_length` | Admit the CSV with source provenance and a content hash |

| Command | Result | Observed evidence |
| --- | --- | --- |
| `make registry-acceptance-gate-readonly` | pass | Current generated lanes and governance checks exit zero |
| `make gate-audit` | partial | `gate-ci-registry` and `workspace-check` exit zero; `gate-ci-rust` exits 2 on the two missing external inputs |

## Replay commands

Run from the repository root with the pinned toolchain and a worktree-local
target directory:

```bash
make gate-local
make governance-gate-readonly
make registry-acceptance-gate-readonly
make gate-audit
```

The scoped Rust lane uses library tests by default. The CI lane must run the
full integration-test kind before treating this local result as a complete
workspace regression result.

## Closure boundary

This record closes the P0 local-gate substitute row for the current batch. It
records the current registry acceptance pass and the explicit T-051 Rust input
boundary. It does not claim that the two missing external datasets or every
historical experiment, proof axiom, or structural debt row is resolved. Those
boundaries remain in the active roadmap and the P1/P2 evidence records.
