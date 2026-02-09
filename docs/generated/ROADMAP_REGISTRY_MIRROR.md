# Roadmap Registry Mirror

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: TOML registry files under registry/ -->
<!-- Generated at: 2026-02-09T08:38:02Z -->

Authoritative source: `registry/roadmap.toml`.

- Consolidated date: 2026-02-06
- Source markdown: `docs/ROADMAP.md`
- Status: `active`

## Companion Docs

- `docs/TODO.md`
- `docs/NEXT_ACTIONS.md`

## Workstreams

### WS-REGISTRY-001: Documentation Registry Evolution

- Priority: `high`
- Status: `in_progress`
- Description: Move markdown operational docs to TOML-first authoritative registries with generated markdown mirrors.
- Primary outputs:
  - `registry/knowledge_sources.toml`
  - `registry/knowledge/documents.toml`
  - `registry/roadmap.toml`
  - `registry/todo.toml`
  - `registry/next_actions.toml`
  - `registry/requirements.toml`

### WS-QUALITY-001: Warnings-As-Errors Discipline

- Priority: `high`
- Status: `active`
- Description: All Python and Rust checks run with warnings treated as failures.
- Primary outputs:
  - `Makefile`
  - `docs/REQUIREMENTS.md`
  - `AGENTS.md`

### WS-CLAIMS-001: Claims Evidence Governance

- Priority: `high`
- Status: `active`
- Description: Maintain claim lifecycle with TOML registry authority and markdown mirrors.
- Primary outputs:
  - `registry/claims.toml`
  - `docs/CLAIMS_EVIDENCE_MATRIX.md`
  - `docs/CLAIMS_TASKS.md`
