# Three-Layer Registry Architecture

> Canonical reference for the SQLite-first architecture with TOML compatibility artifacts.

## Overview

The registry uses a three-layer architecture where `.cache/registry.sqlite3`
is the canonical source of truth, and files under `registry/` are maintained as
compatibility and migration artifacts.

```
LAYER 1: CANONICAL (SQLite)
    |
    v  `make registry-build`
LAYER 2: COMPATIBILITY (legacy TOML imports plus DB-backed compatibility exports)
    |
    v  `gororoba-db` CLI
LAYER 3: QUERY (claims, insights, experiments, xref, audit, search)
```

## Layer 1: Canonical Registry Database

All canonical values live in `.cache/registry.sqlite3`. The CLI and tooling
interact with this database as the source of truth for queries, checks, and
reproducible outputs.

| Component | Why it matters |
|-----------|----------------|
| Canonical store | SQLite file with signed migrations and deterministic schema |
| Verification target | `make governance-gate` verifies source integrity and references |
| Query performance | FTS5 and indexed relational paths for fast interactive use |

## Layer 2: Compatibility TOML Layer

Compatibility TOML files in `registry/` are managed as a controlled migration
surface and interoperability layer. `registry/source_manifest.toml` tracks those
files for rebuild and export consistency.

| Category | Files | Examples |
|----------|-------|---------|
| Core research | 6 | claims.toml, insights.toml, experiments.toml, binaries.toml |
| Evidence graph | 6 | bibliography.toml, claims_evidence_edges.toml, lacunae.toml |
| Data governance | 7 | artifact_source_of_truth.toml, data_governance.toml |
| Narrative content | 6 | research_narratives.toml, book_docs.toml |
| Project config | 7 | roadmap.toml, terminology_standards.toml |
| Infrastructure | 2 | agents_contract.toml, mcp_server_matrix.toml |
| Governance lock | 1 | schema_signatures.toml |

The build step still ingests compatibility TOMLs where migration is incomplete,
but DB-backed lanes now round-trip through SQLite before their compatibility
exports are rewritten:

1. Reads `registry/source_manifest.toml` for compatible descriptors
2. Deletes any existing `.cache/registry.sqlite3`
3. Creates a fresh DB with all 13 migrations
4. Sets `PRAGMA journal_mode=WAL` for concurrent read safety
5. Ingests compatibility TOML inputs into canonical tables
6. Rewrites DB-backed planning compatibility exports (`roadmap.toml`, `todo.toml`, `next_actions.toml`)
7. Builds FTS5 full-text indexes on claims, insights, bibliography
8. Builds crossref join tables (claim-experiment, claim-insight)
9. Records build metadata (timestamp, source count)

Makefile prerequisites keep the DB refresh tied to compatibility changes, while
queries continue to run directly against SQLite.

## Layer 3: Query CLI

```bash
# Build
gororoba-db build              # Create .cache/registry.sqlite3
gororoba-db build --verify     # Build + verify integrity

# Claims
gororoba-db claims list --status Verified
gororoba-db claims show C-1234
gororoba-db claims search "ultrametric"
gororoba-db claims unlinked

# Insights
gororoba-db insights list
gororoba-db insights search "Fourier"

# Experiments
gororoba-db experiments list --status active

# Cross-references
gororoba-db xref dangling      # Find broken references
gororoba-db xref unlinked      # Claims with no links
gororoba-db xref coverage      # Coverage summary

# Full-text search
gororoba-db search "algebraic structure"

# Audit
gororoba-db audit signatures   # Verify schema hashes
gororoba-db audit crossrefs    # Check referential integrity

# Statistics
gororoba-db stats
gororoba-db schema
```

## Database Schema

13 migrations in `db/migrations/`:

| Layer | Tables | Migration |
|-------|--------|-----------|
| Provenance | artifacts, documents, lanes, mirrors | 0001 |
| Control Plane | claims, insights, experiments_cp, binaries_cp, theorems | 0002-0004 |
| Downloads | download_jobs, download_attempts, download_campaigns | 0005-0008 |
| External Sources | external_source_contracts, external_source_dossiers | 0009 |
| Knowledge | equation_atoms, proof_skeletons, derivation_steps | 0010 |
| Planning | roadmap_items, todo_items, next_action_items | 0010, 0013 |
| Narratives | research_narratives (+ FTS5 search) | 0010 |
| FTS5 + Crossrefs | claims_fts, insights_fts, bibliography_fts, evidence_edges, crossref tables | 0011 |
| Literature Verification | literature_verification_runs, literature_verification_results, literature_novelty_similar_papers | 0012 |

## Governance Gate

The governance gate validates the canonical SQLite data plus TOML compatibility
invariants.

```bash
make governance-gate    # Schema signatures, crossrefs, labels, etc.
```

The `schema_signatures.toml` file remains the canonical governance snapshot committed
to git. It contains content and schema hashes for governance validation.
Regenerate with `make integrity-resolution`.

## Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Canonical source | SQLite 3 (WAL mode) | Fast queries, transactional consistency |
| Compatibility export | TOML | Human-readable, review-friendly, git-mergeable |
| Rust bindings | rusqlite 0.38 | Embedded SQLite, zero external deps |
| Migrations | rusqlite_migration 2.4 | Sequential SQL files |
| CLI | clap 4 (derive) | Subcommand dispatch |
| Full-text search | SQLite FTS5 | Built-in, BM25 ranking |
| Hashing | blake3 | Fast content fingerprinting |

## Workflow

To modify the registry:

```bash
# 1. Update compatibility TOMLs as needed (migration path)
vim registry/claims.toml

# 2. Regenerate schema signatures (if governance gate requires it)
make integrity-resolution

# 3. Rebuild canonical DB and compatibility artifacts (automatic via Make prerequisites)
make registry-build

# 4. Query to verify
gororoba-db claims show C-1375
gororoba-db xref dangling

# 5. Run governance gate before push
make governance-gate
```
