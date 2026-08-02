---
description: Active evidence ledger and document authority crosswalk for P0-P2 repository work
last_verified: 2026-08-01
evidence_class: active-operating-contract
---

# Evidence ledger operating contract

This document is the active index for the P0-P2 repository buildout. It records
which artifact owns each decision, which generated views mirror that authority,
and which command proves each transition. Historical reports retain their
original claims and remain outside this ledger's rewrite scope.

## Authority crosswalk

| Surface | Authority | Derived views | Mutation path |
| --- | --- | --- | --- |
| Planning rows | `registry/canonical/control_plane.sqlite3` | `registry/roadmap.toml`, `registry/todo.toml`, `registry/next_actions.toml`, `docs/ROADMAP.md`, `docs/NEXT_ACTIONS.md` | `gororoba-db planning ...` |
| Debt taxonomy and execution order | `plans/repo_debt_taxonomy_roadmap_2026_06_04.toml` | active roadmap summaries and this ledger | `apply_patch`, followed by the named verification gates |
| Markdown ownership | `registry/markdown_owner_map.toml` | owner-map checks and governance policy inputs | `markdown-registry register` or an explicit owner-map review |
| Markdown content inventory | on-disk Markdown plus owner-map coverage | `registry/knowledge_sources.toml` | `make registry-knowledge` |
| Markdown lifecycle policy | `registry/knowledge_sources.toml` plus header evidence | `registry/markdown_governance.toml` | `make registry-governance` |
| Document search index | `registry/canonical/control_plane.sqlite3` | `documents` and `document_search` tables | `provenance index --knowledge-sources registry/knowledge_sources.toml` |
| Formal proof status | `proofs/`, `_RocqProject`, retained proof reports | proof gate output and dated evidence notes | `make rocq-project-check`, `make -C proofs vos`, and `make -C proofs vok` |

Generated TOML and Markdown views are read-only compatibility artifacts. A
generated view never becomes a manual source because a consumer reads it.

## Evidence contract

Every active frontier row carries these fields in the debt roadmap:

| Field | Required meaning |
| --- | --- |
| `row_id` | Stable mechanism-first identifier |
| `status` | `open`, `partial`, `blocked`, or `closed` |
| `owner_surface` | Exact source, registry, code, or host boundary |
| `claim` | One falsifiable engineering statement |
| `evidence_refs` | Paths, commands, hashes, or retained outputs |
| `falsifier` | Observation that invalidates the claim |
| `next_action` | Smallest reproducible action that changes the state |
| `closure_condition` | Observable condition for `closed` |

The ledger distinguishes three states. `Verified` has a retained artifact and a
passing check. `Partial` has evidence for one boundary but an explicit open
boundary. `Blocked` names the searched surface and the intake condition; it does
not claim global absence.

## P0: control-plane and gate closure

| Row | State | Evidence surface | Closure condition |
| --- | --- | --- | --- |
| `p0-document-index` | closed | `registry/knowledge_sources.toml`, `registry/markdown_governance.toml`, provenance document tables | Fresh generators emit 128 rows, provenance reindex stores 128 documents, and parity plus governance checks agree on the same document set |
| `p0-planning-row-reconciliation` | partial | SQLite planning tables and generated planning exports | Obsolete PDS-SBN rows close through the typed CLI; live Voyager and optics rows retain their real open boundaries |
| `p0-local-gate-substitute` | closed | `Makefile`, `.githooks/pre-push`, local gate output, retained gate evidence | The local gate reproduces the required governance, registry, ASCII, terminology, and Rust checks with retained output |

## P1: scientific and formal evidence closure

| Row | State | Evidence surface | Closure condition |
| --- | --- | --- | --- |
| `p1-voyager-bartol-amda-comparator` | partial | Voyager finding document, source manifest, and exact downloaded or bounded-missing data | Bartol and AMDA use the same time basis and units, or the missing AMDA boundary remains explicitly blocked |
| `p1-rocq-project-completeness` | partial | `_RocqProject`, `rocq-project-audit`, pinned `vos` and `vok` outputs | Project parity and both gates pass; every remaining axiom or parameter has a named disposition with a falsifier |
| `p1-formal-evidence-registry` | partial | formal proof field schema and proof evidence outputs | Proof counts, theorem status, assumptions, and gate results share one dated evidence record |

## P2: reproducibility and structural debt

| Row | State | Evidence surface | Closure condition |
| --- | --- | --- | --- |
| `p2-randomness-run-manifests` | partial | `rand` call sites, seed policy, experiment outputs, manifest schema, and `experiment-manifest` | New manifests record seed policy, generator, toolchain, feature set, hardware, and input/output hashes; legacy rows remain explicitly unclassified |
| `p2-materials-source-contract` | partial | `materials_core`, `materials_data`, build scripts, and source data | The current generated-data boundary and bounded absent paths are recorded; table hashes and parity evidence remain open |
| `p2-structural-debt-slices` | open | largest-file census, CPD output, and crate ownership boundaries | Each split preserves API behavior, tests the boundary, and records before/after measurements |

## Reproducible gate sequence

Run commands from the repository root with the pinned Rust toolchain and a
worktree-local target directory.

```bash
CARGO_TARGET_DIR=.cache/cli-target cargo run -q -p gororoba_cli_data --bin markdown-registry -- build-knowledge-sources
CARGO_TARGET_DIR=.cache/cli-target cargo run -q -p gororoba_cli_data --bin markdown-registry -- build-governance
CARGO_TARGET_DIR=.cache/cli-target cargo run -q -p gororoba_cli_data --bin provenance -- index --knowledge-sources registry/knowledge_sources.toml
make registry-verify-mirrors
make registry-acceptance-gate-readonly
make rocq-project-check
make -C proofs vos
make -C proofs vok
```

The sequence records generated deltas before a commit. A failed gate remains an
open ledger row until its root cause and retained evidence are named.

## Retention and review cadence

| Artifact class | Policy |
| --- | --- |
| Raw external captures and hashes | Retain verbatim; add a manifest rather than rewriting bytes |
| Generated registry views | Regenerate from the canonical source and review the diff |
| Historical reports | Preserve body and provenance; add a new dated finding for new evidence |
| Temporary build caches | Keep only inside ignored worktree cache paths; remove only after scope confirmation |
| Open planning rows | Update through the typed SQLite CLI and verify the generated exports |

The next review starts from the live checkout, branch, toolchain, and generated
hashes. It does not infer closure from a stale title, an old PR, or a missing
remote workflow run.
