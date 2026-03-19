# SQLite Source-of-Truth Architecture

> Canonical reference for the database-first design of open\_gororoba.

## Overview

The `registry/canonical/control_plane.sqlite3` database is the **single
authoritative source of truth** for all structured metadata in the project.
TOML files in `registry/` are **read-only compatibility exports** that can be
regenerated at any time from the database.  The `gororoba-db` CLI provides a
unified Rust-native entrypoint for querying, importing, exporting, auditing,
and managing the database.

```
┌────────────────────────────────────────────────────────────┐
│          registry/canonical/control_plane.sqlite3           │
│                     SOURCE OF TRUTH                        │
│  10 migrations · 35+ tables · FTS5 full-text search        │
└────────────────────┬───────────────────────────────────────┘
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              ▼
 TOML exports   gororoba-db CLI  provenance CLI
 (read-only)    (query/import)   (index/export)
```

## Database Layers

| Layer | Tables | Migration | Status |
|-------|--------|-----------|--------|
| **Provenance** | `artifacts`, `documents`, `citations`, `links`, `artifact_links`, `artifact_paths`, `mirror_observations`, `lane_assignments`, `export_runs`, `ingest_fingerprints` | 0001 | Migrated |
| **Control Plane** | `claims`, `insights`, `experiments_cp`, `binaries_cp`, `theorems`, `control_plane_runs`, `control_plane_meta`, `registry_snapshots` | 0002–0004 | Migrated |
| **Downloads** | `download_jobs`, `download_attempts`, `download_campaigns`, `download_campaign_jobs` | 0005–0008 | Migrated |
| **External Sources** | `external_source_contracts`, `external_source_dossiers` (+ meta/values) | 0009 | Migrated |
| **Knowledge** | `equation_atoms`, `proof_atoms`, `proof_skeletons`, `derivation_steps` | 0010 | Pending import |
| **Planning** | `roadmap_items`, `todo_items`, `next_action_items` | 0010 | Pending import |
| **Narratives** | `research_narratives`, `research_narrative_search` (FTS5) | 0010 | Pending import |
| **Notebooks** | `notebook_sessions` | 0010 | New |
| **Manifest** | `source_of_truth_manifest` | 0010 | Bootstrapped |

## What Belongs in the Database

### ✅ MUST be in the database (authoritative)

| Data | Reason |
|------|--------|
| Claims (C-001 … C-1300) | Formal verification status, proof links, evidence chains |
| Insights (I-001 … I-182) | Cross-domain research discoveries, claim references |
| Experiments (E-001 … E-200) | Reproducibility metadata, run commands, SHA256 hashes |
| Binaries (364 entries) | Binary-to-experiment mapping, crate sourcing |
| Theorems (144 entries) | Formal proofs, Rocq verification status |
| Artifacts (3236 entries) | Provenance index, mirror tracking, canonical URLs |
| External source contracts | Data governance, retrieval policies, deadlines |
| Download jobs/campaigns | Pipeline state, attempt tracking, failure classification |
| Equation atoms | Mathematical knowledge graph, derivation links |
| Proof skeletons / derivation steps | Formal proof structure, step dependencies |
| Roadmap / todo / next actions | Project planning with dependency graphs |
| Research narratives | Full-text searchable research documents |
| Notebook sessions | Interactive analysis session metadata |

### ❌ MUST NOT be in the database

| Data | Location | Reason |
|------|----------|--------|
| Raw data files (CSV, HDF5, FITS) | `data/` | Binary blobs, too large, filesystem-native |
| Compiled artifacts (`target/`, `*.vo`) | Build output | Ephemeral, reproducible from source |
| Credentials, secrets, API keys | `.env` files | Security, never committed |
| Binary executables | Build output | Compiled from source code |
| Git history | `.git/` | Version control is its own database |
| LaTeX build outputs | `docs/latex/out/` | Generated, ephemeral |

### 🗄️ Legacy items (already in `archive/`)

| Data | Archive Location |
|------|-----------------|
| Pantheon/PhysicsForge migration | `archive/registry/pantheon_physicsforge/` |
| Wave 4–6 phase plans | `archive/registry/wave_phase_plans/` |
| 8086 instruction cycle CSVs | `archive/8086_legacy/` |
| Retired external placeholders | `archive/external_legacy_placeholders/` |
| Non-reproducible snapshots | `archive/external_nonreproducible_snapshots/` |

### 🔄 Legitimate items handled differently

| Data | Location | Approach |
|------|----------|----------|
| Formal proofs (`.v`, `.lean`) | `proofs/` | Filesystem + DB cross-references (`theorems.proof_path`) |
| LaTeX sources | `docs/latex/` | Filesystem + DB claim/experiment references |
| Makefile targets | `Makefile` | Orchestration layer referencing `gororoba-db` and `provenance` |
| Python scripts | `src/scripts/` | Legacy generators being replaced by Rust binaries |

## CLI Entrypoints

### `gororoba-db` (new, lightweight)

The primary user-facing tool for database interaction:

```bash
# Database overview
gororoba-db stats
gororoba-db schema
gororoba-db audit

# Import data from TOML into SQLite
gororoba-db import-planning
gororoba-db import-knowledge
gororoba-db import-narratives

# Export from SQLite
gororoba-db export-planning --table roadmap --format json

# Query
gororoba-db query roadmap --status active
gororoba-db query todo --status open --limit 10

# Full-text search
gororoba-db search "algebraic structure"

# Legacy analysis
gororoba-db archive-legacy

# Notebook integration
gororoba-db notebook-info
gororoba-db notebooks list
gororoba-db notebooks create --title "Analysis Session"
```

### `provenance` (existing, operator-focused)

Heavy-duty operator CLI for bulk index/export/verify cycles:

```bash
provenance index-control-plane     # Import from TOML → SQLite
provenance export-control-plane    # Export SQLite → TOML compatibility
provenance verify-control-plane    # Validate invariants
provenance query claim C-001       # Query individual entities
provenance doctor                  # Health report
```

## Jupyter / evcxr Integration

The `evcxr_jupyter` crate provides a Rust kernel for Jupyter notebooks.
Within a notebook, workspace crates can be loaded interactively:

```rust
:dep provenance_store = { path = "crates/provenance_store" }
:dep cd_kernel = { path = "crates/cd_kernel" }

use provenance_store::ProvenanceStore;
let store = ProvenanceStore::open(
    std::path::Path::new("registry/canonical/control_plane.sqlite3")
).unwrap();

// Query the source of truth directly
let stats = store.source_of_truth_stats().unwrap();
for (table, cat, count, meta) in &stats {
    println!("{table}: {count} rows [{cat}]");
}
```

Install with:
```bash
cargo install --locked evcxr_jupyter
evcxr_jupyter --install
jupyter notebook  # Select "Rust" kernel
```

## Migration Workflow

To bring knowledge and planning data into SQLite:

```bash
# 1. Import planning data (roadmap, todo, next-actions)
gororoba-db import-planning

# 2. Import knowledge base (equations, proofs, derivations)
gororoba-db import-knowledge

# 3. Import research narratives
gororoba-db import-narratives

# 4. Verify
gororoba-db stats
gororoba-db audit
```

## Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Database | SQLite 3 (bundled) | Zero-config, single-file, embedded, cross-platform |
| Rust bindings | `rusqlite` 0.38 | Thin ergonomic wrapper, sync, bundled SQLite |
| Migrations | `rusqlite_migration` 2.4 | Sequential numbered SQL files in `db/migrations/` |
| CLI framework | `clap` 4.5 (derive) | Standard Rust CLI with subcommands |
| Serialization | `serde` + `toml` + `serde_json` | Multi-format import/export |
| Hashing | `blake3` | Fast cryptographic hashing for ingest fingerprints |
| Full-text search | SQLite FTS5 | Built-in, no external dependencies |
| Notebooks | `evcxr_jupyter` | Rust REPL + Jupyter kernel for interactive analysis |
