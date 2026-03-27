# open\_gororoba

A pure-Rust computational physics workspace bridging Cayley-Dickson
algebra, heliospheric plasma dynamics, lattice-Boltzmann turbulence,
GPU-accelerated simulations, formal verification, and multi-spacecraft
observational analysis -- from quaternions to 16384-dimensional algebras,
from 0.05 AU (Parker Solar Probe) to 137 AU (Voyager interstellar).

## Quick start

```bash
# Prerequisites: Rust nightly (pinned via rust-toolchain.toml)
git clone https://github.com/Oichkatzelesfrettschen/open_gororoba.git
cd open_gororoba

# Build the entire workspace
cargo build --workspace

# Run the quality gates
make governance-gate       # Full 7-check registry + integrity gate
make rust-clippy           # Clippy with warnings-as-errors
cargo test --workspace     # Full test suite
```

## Project layout

```
open_gororoba/
  crates/              68 Rust crates (libraries + CLI binaries)
  registry/            TOML registries (claims, insights, experiments, binaries)
  registry/canonical/  SQLite source-of-truth database (control_plane.sqlite3)
  proofs/              262 Rocq 9.1.1 formal proofs (105 verified theories)
  data/                External datasets and computed results (LFS-tracked)
  db/migrations/       12 SQLite migration scripts
  docs/                Synthesis documents, evidence notes, architecture references
  Makefile             Build, test, governance, and CI targets
```

## Registry and claims system

The project tracks every conjecture, measurement, and computation as a
**claim** with a unique ID (C-001 through C-1551). Each claim has a
status (`Verified`, `Refuted`, `Provisional`), a source location in
the codebase, and optionally a formal proof in Rocq.

| Registry | Entries | Purpose |
|----------|---------|---------|
| `registry/claims.toml` | 1412 | Conjectures, theorems, measurements |
| `registry/insights.toml` | 182 | Cross-domain research discoveries |
| `registry/experiments.toml` | 210 | Reproducible experiment definitions |
| `registry/binaries.toml` | 377 | CLI binary inventory |

### SQLite source of truth

The `registry/canonical/control_plane.sqlite3` database is the
**authoritative source of truth** for all structured metadata. TOML
files are read-only compatibility exports. The database provides:

- 73 tables across 12 migrations
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
| `cd_kernel` | Cayley-Dickson algebra engine (signs, zero divisors, AVX2 SIMD) |
| `algebra_analysis` | Box-kite alignment, spectral analysis, obstruction theory |
| `algebra_experimental` | Experimental algebraic structures (SU(5), braiding, imbalance) |
| `gororoba_algebra` | Lie algebras, Golay code, Leech lattice, GPU dispatch |
| `verified_core` | Formally verified computations (unified action, spectral dim) |
| `cosmology_core` | NFW halos, harmonic stacking, DC14 profiles |
| `quantum_core` | Quantum field manifolds, renormalization |
| `gr_core` | General relativity, ADM formalism, NanoGrav fitting |
| `stats_core` | Bootstrap CI, mutual information, quantile regression |
| `tensor_core` | Tensor-train cross approximation |
| `spectral_core` | Surrogate models, spectral methods |
| `lbm_3d` / `lbm_3d_cuda` | Lattice-Boltzmann 3D (CPU + CUDA MRT + Vulkan) |

### Data and provenance

| Crate | Purpose |
|-------|---------|
| `data_core` | Catalog loaders (MaNGA, LoTSS, FITS, VOTable, OMNI, Ulysses) |
| `provenance_store` | SQLite-backed artifact and metadata store |
| `provenance_core` | Provenance data model |
| `provenance_ops` | Data ingest pipelines (ORIX, HEASARC, AMDA HAPI) |
| `gororoba_db` | `gororoba-db` CLI for database operations |
| `lit_search` | 17-source academic literature search and deduplication |

### CLI binaries (377 registered)

| Crate | Representative binaries |
|-------|------------------------|
| `gororoba_cli_data` | NanoGrav timing, LoTSS fetch, CD cache, entropy audit |
| `gororoba_cli_physics` | Heliosphere pipeline (15 bins), box-kite alignment, LBM sims |
| `gororoba_cli_algebra` | Spectral flow, Majorana braiding, ZD resonance |
| `gororoba_cli_quantum` | Fracton codes, pseudospectrum, absorber Pareto |
| `gororoba_cli_governance` | Registry integrity resolution, markdown registry |

## Heliosphere research pipeline

The project includes a complete multi-spacecraft heliosphere analysis
pipeline spanning 13 missions (ACE, Cassini, Helios 1/2, IBEX, IMP 8,
Juno, New Horizons, OMNI, Parker Solar Probe, Solar Orbiter, STEREO-A,
Ulysses, Voyager 1/2, WIND) from 0.05 to 157 AU:

1. **Feature cube**: Unified 16D hourly records from all missions
2. **Magnetic Takens embedding**: 4-step delay into sedenion (16D) space
3. **Quench scan**: Associator norm mapped across (r, latitude) bins
4. **Box-kite alignment**: GPU-accelerated PSL(2,7) permutation scan
5. **Null audit**: Temporal-shuffle and channel-permutation deconfounding
6. **Backend parity**: CPU/CUDA/Vulkan numerical equivalence verified

See `docs/reports/RC1_EVIDENCE_NOTE.md` for the frozen evidence envelope.

## Formal verification

262 Rocq files in `proofs/` (105 verified theories) verify algebraic
identities, norm bounds, and obstruction invariants. Build with:

```bash
make -C proofs vos   # Interface-only compilation (fast)
make -C proofs vok   # Full parallel body verification
```

Key tactics: `ring_simplify; lra` for concrete norm proofs,
`cbv [whitelist]` for Cayley-Dickson dimension >= 8 (avoids OOM from
`simpl`). Tower proofs use `rewrite sed_mul_scale_left` for O(seconds)
compilation vs O(hours) for monolithic `ring`.

## Quality gates

| Gate | Command | Enforced |
|------|---------|----------|
| Clippy | `make rust-clippy` | Warnings-as-errors (`-D warnings`) |
| SemVer | `make rust-semver-check` | Public API compatibility |
| Tests | `cargo test --workspace` | Full suite, 0 tolerance |
| ASCII | `make ansi-check` | No Unicode/emoji in source |
| Terminology | `make terminology-gate` | 8 banned patterns |
| Governance | `make governance-gate` | 7-check registry + integrity |
| Pre-push | `make pre-push-gate` | Scoped clippy + test + governance |

## Toolchain

- **Rust**: nightly-2026-03-05 (pinned via `rust-toolchain.toml`)
- **Edition**: 2024
- **Build**: Cranelift backend for dev (opt-level 2), LLVM for release
- **GPU**: CUDA via `cudarc 0.19.1`, Vulkan via `ash` (feature-gated)
- **Formal proofs**: Rocq 9.1.1
- **Allocator**: mimalloc (workspace default)

## License

GPL-2.0-only
