---
description: Bounded inventory and ownership boundary for ignored Markdown artifacts in the open_gororoba checkout
last_verified: 2026-08-01
evidence_class: retained-local-inventory
status: active
---

# Markdown governance ignored boundary

The canonical checkout contains 136 ignored Markdown files that are absent
from the Git reproducible source surface. The owner map contains 129 entries,
which matches the 129 tracked Markdown files. The 136 ignored files therefore
do not enter `registry/markdown_owner_map.toml` in this change: registering a
local-only file would create a stale owner entry in an isolated worktree.

The Markdown registry uses Git's tracked plus unignored working-tree surface
as its input. This preserves the owner-map invariant for reproducible files
while keeping private caches, generated mirrors, and acquisition captures out
of the governance gate. The local files remain available for evidence review
and are not deleted by this policy.

The claim-ticket mirror verifier applies the same boundary. It checks
Git-governed ticket paths and skips 15 ignored local ticket paths, including
the ignored index, with an explicit gate message. The registry still records
those source contracts for a future ticket-mirror admission or retirement
decision.

## Inventory by ownership boundary

| Scope | Count | Classification | Disposition |
| --- | ---: | --- | --- |
| `.github/copilot-instructions.md` | 1 | Local agent overlay | Preserve locally; exclude from project corpus |
| `REQUIREMENTS.md` | 1 | Generated root mirror | Preserve locally; source is `registry/requirements.toml` and `registry/requirements_narrative.toml` |
| `archive/root_cleanup_2026-04-03/**/.gemini/plans/*.md` | 2 | Superseded agent plans | Retain as archive evidence; exclude from active corpus |
| `crates/brown_1972/*.md` | 4 | Legacy research notes | Candidate for a separate Brown documentation admission after ASCII cleanup and ownership review |
| `data/benchmarks/parity_report.md` | 1 | Local benchmark capture | Retain pending host, tool, and input provenance |
| `docs/*.md` legacy root notes | 17 | Project narratives, roadmaps, and task mirrors | Candidate for separate source admission or explicit retirement |
| `docs/book/src/**/*.md` | 28 | mdBook source pages | Candidate for separate source admission; `registry/book_docs.toml` already names this lane |
| `docs/monograph/*.md` | 1 | Research monograph | Candidate for separate source admission after encoding review |
| `docs/reports/**/*.md` | 14 | Research packets and dossiers | Candidate for separate source admission or archive disposition |
| `docs/research/*.md` | 2 | Research narratives | Candidate for separate source admission or archive disposition |
| `docs/tickets/*.md` | 14 | Generated and legacy ticket mirrors | Reconcile against canonical registry before admission |
| `reports/acquisition_sessions/**/checklist.md` | 49 | Acquisition-session evidence | Retain as external evidence; exclude from active Markdown corpus |
| `reports/cayley_dickson_cache_audit_2026_03_25.md` | 1 | External cache audit | Retain as evidence; exclude until its cache boundary is reproducible |
| `reports/profiling/root_legacy_perf_capture_triage_2026_04_04.md` | 1 | Profiling capture triage | Retain pending benchmark provenance |
| **Total** | **136** | **Bounded local inventory** | **No ignored file is deleted** |

## Reproduction and falsifiers

Run the cached registry binary from the repository root to re-enumerate the
local boundary:

```bash
.cache/gate-target/gate-tools/markdown-registry verify-inventory-toml-first
```

The expected local result before the Git-boundary change is 136
`UNREGISTERED` lines. A clean isolated worktree has zero such lines because
these files are not part of the Git source surface. The boundary is false if a
tracked Markdown file is omitted, an unignored Markdown file is omitted, or an
owner-map entry points at a path absent from the worktree.

The next admission unit is the mdBook lane. It requires adding the source
pages, updating the owner map through `markdown-registry register`, regenerating
the book and knowledge-source compatibility views, and verifying the result in
a clean worktree. This ledger keeps that work separate from local acquisition
and profiling residue.
