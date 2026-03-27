# open\_gororoba

A pure-Rust computational physics workspace exploring Cayley-Dickson
algebraic structures, dark-matter phenomenology, lattice-Boltzmann
turbulence, formal verification, and multi-scale observational data
analysis.

## Quick start

```bash
# Prerequisites: Rust nightly (pinned via rust-toolchain.toml)
git clone https://github.com/Oichkatzelesfrettschen/open_gororoba.git
cd open_gororoba

# Build the entire workspace
cargo build --workspace

# Run the quality gates
make rust-clippy          # Clippy with warnings-as-errors
make rust-semver-check    # Public API SemVer compliance
cargo test --workspace    # Full test suite (~3700 tests)
```

## Project layout

```
open_gororoba/
  crates/              54 Rust crates (libraries + CLI binaries)
  registry/            TOML registries (claims, insights, experiments, binaries)
  registry/canonical/  SQLite source-of-truth database (control_plane.sqlite3)
  proofs/              193 Rocq (Coq 9.1.1) formal proofs
  data/                External datasets and computed results (gitignored)
  db/migrations/       10 SQLite migration scripts
  docs/                Synthesis documents and architecture references
  Makefile             Build, test, governance, and CI targets
```

## Registry and claims system

The project tracks every conjecture, measurement, and computation as a
**claim** with a unique ID (C-001 through C-1300+). Each claim has a
status (`verified`, `falsified`, `open`), a source location in the
codebase, and optionally a formal proof in Rocq.

| Registry | Entries | Purpose |
|----------|---------|---------|
| `registry/claims.toml` | 1300+ | Conjectures, theorems, measurements |
| `registry/insights.toml` | 182 | Cross-domain research discoveries |
| `registry/experiments.toml` | 200 | Reproducible experiment definitions |
| `registry/binaries.toml` | 261+ | CLI binary inventory |

### SQLite source of truth

The `registry/canonical/control_plane.sqlite3` database is the
**authoritative source of truth** for all structured metadata. TOML
files are read-only compatibility exports. The database provides:

- 35+ tables across 10 migrations
- FTS5 full-text search over research narratives
- Provenance tracking for all artifacts
- Planning tables (roadmap, todo, next-actions) with dependency graphs
- Knowledge graph (equation atoms, proof skeletons, derivation steps)

See [docs/db/ARCHITECTURE.md](docs/db/ARCHITECTURE.md) for the full
schema and migration workflow.

### CLI tools

```bash
# Database overview and audit
cargo run -p gororoba_db --bin gororoba-db -- stats
cargo run -p gororoba_db --bin gororoba-db -- audit

# Import planning data from TOML into SQLite
cargo run -p gororoba_db --bin gororoba-db -- import-planning

# Full-text search across research narratives
cargo run -p gororoba_db --bin gororoba-db -- search "algebraic structure"

# Provenance indexing and verification
cargo run -p gororoba_cli_provenance --bin provenance -- doctor
```

## Crate ecosystem

The workspace is organized into domain-specific crates:

### Core libraries

| Crate | Purpose |
|-------|---------|
| `cd_kernel` | Cayley-Dickson algebra engine (signs, zero divisors, SIMD) |
| `algebra_analysis` | Spectral analysis, obstruction theory |
| `algebra_experimental` | Experimental algebraic structures (SU(5), braiding, imbalance) |
| `gororoba_algebra` | Lie algebras, physics bridges, GPU dispatch |
| `verified_core` | Formally verified computations (unified action, spectral dim) |
| `cosmology_core` | NFW halos, harmonic stacking, DC14 profiles |
| `quantum_core` | Quantum field manifolds, renormalization |
| `gr_core` | General relativity, ADM formalism, NanoGrav fitting |
| `stats_core` | Bootstrap CI, mutual information, quantile regression |
| `tensor_core` | Tensor-train cross approximation |
| `spectral_core` | Surrogate models, spectral methods |
| `lbm_3d` / `lbm_3d_cuda` | Lattice-Boltzmann 3D (CPU + CUDA MRT) |

### Data and provenance

| Crate | Purpose |
|-------|---------|
| `data_core` | Catalog loaders (MaNGA, LoTSS, FITS, VOTable) |
| `provenance_store` | SQLite-backed artifact and metadata store |
| `provenance_core` | Provenance data model |
| `provenance_ops` | Data ingest pipelines (ORIX, HEASARC) |
| `gororoba_db` | `gororoba-db` CLI for database operations |
| `manga_experiments` | MaNGA dark-matter hypothesis experiments |

### CLI binaries

| Crate | Binaries |
|-------|----------|
| `gororoba_cli_data` | NanoGrav timing, LoTSS fetch, entropy audit |
| `gororoba_cli_physics` | Harmonic halo curves, crystal integrals, CHSH sweeps |
| `gororoba_cli_algebra` | Spectral flow, majorana braiding |
| `gororoba_cli_quantum` | Fracton codes, pseudospectrum, absorber Pareto |
| `gororoba_cli_governance` | Registry integrity resolution |

## Formal verification

193 Rocq theories in `proofs/` verify algebraic identities, norm
bounds, and obstruction invariants. Build with:

```bash
make -C proofs vos   # Interface-only compilation (fast)
make -C proofs vok   # Full parallel body verification
```

Key tactics: `ring_simplify; lra` for concrete norm proofs,
`cbv [whitelist]` for Cayley-Dickson dimension >= 8 (avoids OOM from
`simpl`).

## Quality gates

| Gate | Command | Enforced |
|------|---------|----------|
| Clippy | `make rust-clippy` | Warnings-as-errors (`-D warnings`) |
| SemVer | `make rust-semver-check` | Public API compatibility |
| Tests | `cargo test --workspace` | ~3700 tests, 0 tolerance |
| Character | `make ansi-check` | Emoji-blocking UTF-8 |
| Terminology | `make terminology-gate` | 8 banned patterns |
| Governance | `make governance-gate` | Full registry + integrity |
| Pre-push | `make pre-push-gate` | Scoped clippy + test + governance |

## Toolchain

- **Rust**: nightly-2026-03-05 (pinned via `rust-toolchain.toml`)
- **Edition**: 2024
- **Build**: Cranelift backend for dev (opt-level 2), LLVM for release
- **CUDA**: Optional via `cudarc 0.19.1` (feature-gated)
- **Formal proofs**: Rocq 9.1.1

## License

GPL-2.0-only
