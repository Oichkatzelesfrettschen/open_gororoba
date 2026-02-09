# TODO Registry Mirror

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: TOML registry files under registry/ -->

Authoritative source: `registry/todo.toml`.

- Updated: 2026-02-09
- Sprint: S13 (TOML Centralization and Markdown Governance)
- Sprint status: `in_progress`

## Tasks

### S13-T01: Build markdown knowledge index

- Status: `done`
- Evidence:
  - `src/scripts/analysis/build_knowledge_sources_registry.py`
  - `registry/knowledge_sources.toml`

### S13-T02: Create raw TOML capture for non-generated markdown docs

- Status: `done`
- Evidence:
  - `src/scripts/analysis/migrate_markdown_corpus_to_toml.py`
  - `registry/knowledge/documents.toml`

### S13-T03: Define authoritative operational TOML registries

- Status: `in_progress`
- Evidence:
  - `registry/roadmap.toml`
  - `registry/todo.toml`
  - `registry/next_actions.toml`
  - `registry/requirements.toml`

### S13-T04: Add TOML-first export path for insights markdown mirror

- Status: `pending`
- Evidence:
  - `registry/insights.toml`
  - `docs/INSIGHTS.md`

### S13-T05: Add TOML-first export path for experiments markdown mirror

- Status: `pending`
- Evidence:
  - `registry/experiments.toml`
  - `docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md`

### S13-T06: Run registry and quality gates after registry schema changes

- Status: `pending`
- Evidence:
  - `make registry`
  - `cargo run --release --bin registry-check`
  - `python3 bin/ascii_check.py --check`
