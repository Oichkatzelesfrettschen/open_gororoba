# Next Actions Registry Mirror

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: TOML registry files under registry/ -->

Authoritative source: `registry/next_actions.toml`.

- Updated: 2026-02-09
- Status: `active`

## Priority Queue

### NA-001 (P0): Complete TOML centralization for markdown mirrors

- Status: `in_progress`
- Description: Keep TOML-first governance: preserve mirror freshness gates, retire remaining narrative overlays where feasible, and keep legacy ingest explicit.
- References:
  - `registry/knowledge_sources.toml`
  - `registry/knowledge/documents.toml`
  - `registry/knowledge_migration_plan.toml`
  - `src/verification/verify_registry_mirror_freshness.py`

### NA-002 (P1): Resolve multiplication coupling rho(b) in GL(8,Z)

- Status: `done`
- Description: REFUTED (Sprint 16). Only identity element b=0 yields zero residual in rank-adaptive reduced-subspace algorithm. C-466 refuted.
- References:
  - `docs/NEXT_ACTIONS.md`
  - `docs/ROADMAP.md`

### NA-003 (P1): Primary-source citation sweep for remaining claims

- Status: `open`
- Description: Append first-party citations and WHERE STATED coverage for outstanding claims.
- References:
  - `registry/claims.toml`
  - `docs/BIBLIOGRAPHY.md`
  - `docs/CLAIMS_TASKS.md`

### NA-004 (P2): Paper-ready LaTeX pipeline consolidation

- Status: `open`
- Description: Drive manuscript generation from verified TOML registries with reproducible sections.
- References:
  - `docs/latex/`
  - `crates/gororoba_cli/src/bin/generate_latex.rs`
  - `registry/claims.toml`
  - `registry/insights.toml`
