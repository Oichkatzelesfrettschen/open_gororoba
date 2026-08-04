---
description: Typed theorem identity disambiguation and post-transition falsification frontier
last_verified: 2026-08-04
evidence_class: control-plane-and-frontier
---

# Claim theorem identity and falsification frontier

The identity repair separates canonical claim IDs from theorem IDs and refreshes the research frontier from the canonical SQLite database. The repair does not increase scientific coverage by itself. It makes the ledger able to state which formal proposition, implementation predicate, or phenomenological mapping a result addresses.

## Identity boundary

The Rocq proof filename remains a legacy name and repository path. The stable theorem identity is the typed formal namespace. The canonical claim is a separate row linked through an explicit `formal_proposition` relation.

| Legacy theorem name | Stable theorem ID | Actual formal claim |
| --- | --- | --- |
| `C1635_SedenionDriverSemantics` | `THM-SASS-DRIVER-001` | `C-1649` |
| `C1636_Cariow2013SedenionSchedule` | `THM-CARIOW-SEDENION-SCHEDULE-001` | `C-1646` |
| `C1637_R300SedenionZeroDivisor` | `THM-R300-SEDENION-ZERO-DIVISOR-001` | `C-1648` |
| `C1638_OctonionDowncastNoZeroDivisors` | `THM-OCTONION-DOWNCAST-ZERO-DIVISORS-001` | `C-1647` |

The canonical successors retain their unrelated scientific meanings:

| Canonical claim | Meaning |
| --- | --- |
| `C-1635` | tensor electromagnetic Ward conformance |
| `C-1636` | source gravitational Ward identity |
| `C-1637` | tensor gravitational Ward conformance |
| `C-1638` | energy constraint with declared tolerance |

The proof linker accepts only an explicit `formal_proof`, `where_stated`, or `status_note` path. A `C####` or `C_` theorem prefix never creates a claim link. Claim-like theorem aliases remain reserved for successor allocation and must be explicit aliases or explicit links.

## Refreshed ledger state

The frontier generator reads `registry/canonical/control_plane.sqlite3` read-only and writes [post-transition-falsification-frontier.toml](../../data/output/audit/2026-08-04/post-transition-falsification-frontier.toml). The snapshot records database hash `e0b6998368760da921d16f390a9f579c409711e10c3261128276167d296d832c` at generation time. Registration of the final validation artifact changes the database hash to `89b45ecaa86da07110a38e2158c3d3e568fed3d22deecc609f0e4cd1a9e42904` without changing the counted claims, theorem identities, or transition events.

| Measure | Count |
| --- | ---: |
| Claims | 1,464 |
| Stable theorem identities | 162 |
| Append-only transition events | 4 |
| Compound claims | 374 |
| False-confidence queue entries | 1,435 |
| Open-adjudication queue entries | 223 |
| Verified rows | 1,191 |

The queue counts are triage surfaces, not verdicts. The classifier identifies status/evidence contradictions, non-independent oracle language, compound epistemic layers, and ready discriminators. A queue entry requires the class-specific falsifier before any claim transition.

The three-layer invariant stays explicit:

1. A source proposition records what the paper, theorem, or cited equation asserts.
2. An implementation-conformance claim records whether code implements that proposition.
3. A phenomenological-mapping claim records whether model parameters or observables support the physical interpretation.

Evidence at one layer never promotes a claim at another layer. Compound rows remain marked for future splitting rather than being silently promoted.

## Bounded scientific tranches

The frontier emits exact declared IDs instead of selecting tranches from broad keyword matches.

| Tranche | Claims | Discriminator | Controls and blockers |
| --- | --- | --- | --- |
| P2A source-faithful channel semantics | `C-848`, `C-849`, `C-850`, `C-1640` through `C-1644` | Complex channel amplitudes with separate scattering, extinction, absorption, unitarity, reciprocity, time reversal, and passive-loss predicates | Preserve `C-851`; block `C-864`, `C-867`, `C-1638`, and `C-1639` |
| P2B held-out Mie and TCMT reproduction | `C-849`, `C-850` | Frozen training parameters and held-out complex channel amplitudes with phase and magnitude errors separate | Do not promote the comparison claims during oracle repair |
| SFWM source reproduction | `C-832`, `C-834`, `C-839` | Separate paper-calibrated, Sellmeier-derived, direct SFWM, cascaded SHG plus SPDC, and total detected infrared quantities | Preserve `C-833` as the Sellmeier-ordering control |

The P2A list is a planning output only. This run does not modify optics code or begin P2.

## Instrumented path capture

The retained call map covers the typed CLI, identity transaction, reindex/import path, proof linker, exporter, frontier generator, and canonical SQLite checks. The focused source list contains 11 Rust files.

| Tool | Result | Interpretation |
| --- | --- | --- |
| Universal Ctags 6.2.1 | 1,206 symbols, exit 0 | Lexical symbol inventory |
| Cscope 15.9 | index created, exit 0 | Lexical cross-reference only |
| GNU Cflow 1.8 | one `main()` output, exit 0 with Rust redefinition diagnostics | Not a semantic Rust call graph |
| Rust unit tests | 20 passed, 0 failed | Typed transition, identity, rollback, allocation, and explicit-link controls |

The full capture and limits are retained in [claim-theorem-identity-call-map.toml](../../data/output/audit/2026-08-04/claim-theorem-identity-call-map.toml). The load-bearing execution path is cross-checked by Rust compilation, targeted CLI execution, explicit SQLite foreign keys, and generated-view verification.

## Load-bearing findings

The numeric collision is a namespace failure, not a proof failure. A filename prefix compressed two independent identity systems into one token and made an incorrect edge possible even though the current export happened to leave the rows unlinked.

The reindex failure is a preservation failure at a replacement boundary. Replacing compatibility experiment rows must preserve experiments referenced by append-only transition events or the importer destroys the evidence graph it is supposed to regenerate.

The frontier refresh exposes a measurement correction. A larger `Verified` population can result from adding formal successor rows while the falsification surface remains mostly untested. Epistemic repair therefore requires queue coverage, independent falsifiers, compound-claim splits, and retained unresolved assumptions in addition to status counts.

Declare the identity phase complete only after every theorem has a stable typed identity, every claim-like prefix is explicitly linked or explicitly aliased, successor allocation skips reserved identifiers deterministically, the four formal surfaces link to `C-1646` through `C-1649`, the refreshed queues come from the post-transition database, and the P1 manifest and transition events remain byte- and semantically intact.
