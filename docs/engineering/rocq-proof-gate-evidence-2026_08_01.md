---
description: Retained Rocq proof gate and project-file completeness evidence
last_verified: 2026-08-01
evidence_class: formal-verification-evidence
---

# Rocq proof gate evidence

The pinned Rocq 9.1.1 toolchain passes both interface and body compilation in
the isolated worktree. The proof result is strong for compilation and explicit
about the remaining project-file and axiom boundaries.

## Gate results

| Check | Result | Replay command |
| --- | --- | --- |
| Interface compilation | PASS | `make -C proofs vos` |
| Body compilation | PASS | `make -C proofs vok` |
| Rocq version | `9.1.1`, OCaml `5.5.0` | `rocq --version` |
| Rust toolchain | `1.97.0` | `rust-toolchain.toml` |
| Executable `Admitted.` commands | 0 | `rg -n '^Admitted\\.$' proofs/theories proofs/verified` |
| `C1467_XORSignCocycle.v` placeholder markers | 0 | `rg -n 'PLACEHOLDER' proofs/verified/C1467_XORSignCocycle.v` |

The `Zero Admitted` phrases that occur inside comments do not constitute Rocq
commands. The executable scan above remains the closure check for admitted
proof bodies.

## Project-file boundary

The checkout contains 300 `.v` files. The intended source directories contain
297 files: 135 under `proofs/theories` and 162 under `proofs/verified`. The
Rocq project file lists 296 source files. The bidirectional difference is
`proofs/theories/FP24Representable.v`: `_CoqProject` lists it, while
`_RocqProject` does not. The three extraction files are intentionally outside
the source project set.

This is a project-coverage discrepancy, not a failed proof gate. The next
formal-lane action is to reconcile `_RocqProject` with the intended source set
or record why the FP24 file remains outside the active Rocq project.

## Axiom boundary

The source scan finds 169 top-level `Axiom` or `Parameter` declarations across
the theories and verified directories. These declarations represent explicit
model assumptions and abstract interfaces; they do not become proved facts
because `vos` and `vok` pass. The existing axiom disposition report remains the
source for the detailed assumption classification.

The research-quality claim is therefore bounded: the compiled theorem bodies
are replayable under the pinned toolchain, while theorem strength still depends
on the declared axioms and parameters. A future closure record carries the
axiom disposition hash, project-file parity result, and both gate outputs.

## Registry boundary

The theorem registry does not have a standalone `registry/theorems.toml` source
file. The canonical theorem table is the `theorems` table in
`registry/canonical/control_plane.sqlite3`. The generated compatibility view is
`registry/markdown_export/theorems_registry_mirror.rs`, and its rows are loaded
from `_RocqProject` by the provenance indexer. The retained gate note and the
canonical theorem rows therefore expose related facts through separate records;
they do not yet share one typed evidence record with a common hash. That is the
remaining `p1-formal-evidence-registry` boundary.
