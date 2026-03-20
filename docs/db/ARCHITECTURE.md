# Three-Layer Registry Architecture

> Canonical reference for the TOML-source, SQLite-build, CLI-query architecture.

## Overview

The registry uses a three-layer architecture where **TOML files are the single
source of truth**, SQLite is a derived build artifact, and the `gororoba-db` CLI
provides fast querying and auditing.

```
LAYER 1: SOURCE (36 TOML files, human-edited, git-tracked)
    |
    v  `make registry-build`
LAYER 2: BUILD (.cache/registry.sqlite3, .gitignore'd, deterministic)
    |
    v  `gororoba-db` CLI
LAYER 3: QUERY (claims, insights, experiments, xref, audit, search)
```

## Layer 1: Source TOML Files

All 36 source files are listed in `registry/source_manifest.toml` with their
roles and target tables. Edit these files directly; they are the authoritative
record.

| Category | Files | Examples |
|----------|-------|---------|
| Core research | 6 | claims.toml, insights.toml, experiments.toml, binaries.toml |
| Evidence graph | 6 | bibliography.toml, claims_evidence_edges.toml, lacunae.toml |
| Data governance | 7 | artifact_source_of_truth.toml, data_governance.toml |
| Narrative content | 6 | research_narratives.toml, book_docs.toml |
| Project config | 7 | roadmap.toml, terminology_standards.toml |
| Infrastructure | 2 | agents_contract.toml, mcp_server_matrix.toml |
| Governance lock | 1 | schema_signatures.toml (derived but committed) |

## Layer 2: Build

The derived SQLite database is created deterministically from Layer 1 sources:

```bash
make registry-build          # Prerequisite-guarded (no-op if sources unchanged)
make registry-build-verify   # Build + verify crossrefs
```

The build step:
1. Reads `registry/source_manifest.toml` for the file list
2. Deletes any existing `.cache/registry.sqlite3`
3. Creates a fresh DB with all 11 migrations
4. Sets `PRAGMA journal_mode=WAL` for concurrent read safety
5. Ingests all 36 source TOML files into normalized tables
6. Builds FTS5 full-text indexes on claims, insights, bibliography
7. Builds crossref join tables (claim-experiment, claim-insight)
8. Records build metadata (timestamp, source count)

The Makefile uses proper prerequisites so the DB only rebuilds when a source
TOML file changes. Agents running queries hit the cached DB without triggering
rebuilds.

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

11 migrations in `db/migrations/`:

| Layer | Tables | Migration |
|-------|--------|-----------|
| Provenance | artifacts, documents, lanes, mirrors | 0001 |
| Control Plane | claims, insights, experiments_cp, binaries_cp, theorems | 0002-0004 |
| Downloads | download_jobs, download_attempts, download_campaigns | 0005-0008 |
| External Sources | external_source_contracts, external_source_dossiers | 0009 |
| Knowledge | equation_atoms, proof_skeletons, derivation_steps | 0010 |
| Planning | roadmap_items, todo_items, next_action_items | 0010 |
| Narratives | research_narratives (+ FTS5 search) | 0010 |
| FTS5 + Crossrefs | claims_fts, insights_fts, bibliography_fts, evidence_edges, crossref tables | 0011 |

## Governance Gate

The governance gate validates source TOML files directly (no SQLite required):

```bash
make governance-gate    # Schema signatures, crossrefs, labels, etc.
```

The `schema_signatures.toml` file is the one derived file committed to git.
It contains content and schema hashes for governance validation. Regenerate
with `make integrity-resolution`.

## Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Source format | TOML | Human-readable, git-mergeable, diffable |
| Build artifact | SQLite 3 (WAL mode) | Fast queries, FTS5, concurrent reads |
| Rust bindings | rusqlite 0.38 | Embedded SQLite, zero external deps |
| Migrations | rusqlite_migration 2.4 | Sequential SQL files |
| CLI | clap 4 (derive) | Subcommand dispatch |
| Full-text search | SQLite FTS5 | Built-in, BM25 ranking |
| Hashing | blake3 | Fast content fingerprinting |

## Workflow

To modify the registry:

```bash
# 1. Edit source TOML files directly
vim registry/claims.toml

# 2. Regenerate schema signatures (if governance gate requires it)
make integrity-resolution

# 3. Rebuild derived DB (automatic via Make prerequisites)
make registry-build

# 4. Query to verify
gororoba-db claims show C-1375
gororoba-db xref dangling

# 5. Run governance gate before push
make governance-gate
```
