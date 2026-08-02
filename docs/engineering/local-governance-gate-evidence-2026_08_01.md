---
description: retained local governance and Rust gate result for the evidence-ledger batch
last_verified: 2026-08-01
evidence_class: executed_gate
---

# Local governance gate evidence

The isolated evidence-ledger worktree passes the registry, governance, full
nextest, and workspace-check lanes. The keep-going audit remains nonzero only
in `integrity-rust`, where the registry checker reports bounded baseline drift
outside the external-input admission change.

## Executed result

| Gate surface | Result | Observed evidence |
| --- | --- | --- |
| Cache and host routing | pass | 6 physical cores, worker budget 6, cache usage 4906 MB |
| Shared check suite | pass | ANSI check, terminology gate over 8767 files, and report-write policy |
| Rust clippy | pass | `gororoba_cli_data` direct scope with `-D warnings` |
| Rust nextest | pass | 19 library tests passed, nextest run `c8e02897-3c5d-4f73-8296-68f28abf92b8` |
| Markdown inventory | pass | registered=129, on_disk=129, owner entries=129 |
| Governance gate | pass | schema signatures, crossrefs, aliases, authority, headers, and removal policy |
| Registry mirror freshness | pass | Generated mirrors are fresh; claim-ticket verification skips 15 ignored local paths |

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

The full keep-going audit writes `reports/gates/2026-08-01/233501/`. The
registry step, standard and heavy nextest lanes, and workspace check pass. The
Rust regression step completes all tests, then remains nonzero in
`integrity-rust` because the registry checker reports three bounded drift
classes:

| Surface or drift class | Observed evidence | Required follow-up |
| --- | --- | --- |
| Standard nextest | 6105 passed, 48 skipped | No missing-input failures remain |
| Heavy nextest | 1346 passed, 68 skipped | No missing-input failures remain |
| Claim identity baseline | 46 identity-gap claims, `C-1551` through `C-1596` | Reconcile the registry-check baseline |
| Binary registry | `experiment-manifest` and `rocq-project-audit` are absent from `binaries.toml` | Register both Cargo binaries through the canonical registry workflow |
| Experiment count | `project.toml` declares 228; `experiments.toml` contains 232 | Reconcile the canonical project count |

| Command | Result | Observed evidence |
| --- | --- | --- |
| `make registry-acceptance-gate-readonly` | pass | Current generated lanes and governance checks exit zero |
| standard nextest | pass | 6105 tests passed, 48 skipped |
| heavy nextest | pass | 1346 tests passed, 68 skipped |
| `workspace-check` | pass | `cargo check --workspace --tests` exits zero |
| `make gate-audit` | partial | `gate-ci-registry` and `workspace-check` exit zero; `gate-ci-rust` exits 2 in `integrity-rust` on the bounded drift classes above |

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

This record closes the two T-051 external-input failure boundaries. The PDG
mass subset and NANOGrav free-spectrum files have retained source contracts,
content hashes, and replay notes, while production file-backed audits retain
the external source boundary and unit tests use deterministic injected rows.
T-051 remains open because `integrity-rust` does not exit zero until the
identity-gap, binary-registry, and experiment-count drift is reconciled. This
record does not claim that T-058, WS-OPTICS-GR-001, or every historical
experiment, proof axiom, or structural debt row is resolved.
