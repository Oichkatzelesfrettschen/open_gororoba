# open\_gororoba

Rust-first computational physics research platform linking Cayley-Dickson
algebra, lattice Boltzmann fluid dynamics, topological data analysis, and
cosmological observables.

## Overview

open\_gororoba is a scientific workspace organized around four interconnected
thesis pipelines that explore how algebraic structure in hypercomplex number
systems (octonions, sedenions, and beyond) influences physical dynamics at
multiple scales.  The repository maintains over 860 scientific claims with
evidence-linked verification, 4400+ automated tests, and a TOML-first
registry model where structured data is the source of truth.

### Four Thesis Pipelines

1. **Vacuum Frustration** -- Cayley-Dickson algebraic structure perturbs
   lattice transport behavior.  The 3/8 frustration attractor emerges from
   zero-divisor topology in dimension >= 16.

2. **Filtration Cascade** -- Persistent homology traces topological phase
   behavior in algebraically modified flow fields.  Vietoris-Rips and
   cubical complexes detect phase transitions invisible to spectral methods.

3. **Neural Homotopy** -- Neural models learn the residual structure between
   analytic predictions and simulated dynamics, providing a data-driven
   correction layer.

4. **Synthesis** -- Combined outputs from the first three pipelines are
   compared against cosmological observables (SNIa, BAO, CMB) and
   experimental data (ALICE QGP, materials spectroscopy).

## Crate Architecture

The workspace is organized into four tiers with strict dependency ordering:

### Tier 1: Foundation

Core numerical and algorithmic crates with no internal tier dependencies.

| Crate | Purpose |
|-------|---------|
| `algebra_core` | Cayley-Dickson algebras, Clifford algebras, zero-divisor analysis |
| `stats_core` | Statistical distributions, bootstrap CI, hypothesis testing |
| `spectral_core` | FFT, wavelet transforms, spectral density estimation |
| `lbm_core` | Lattice Boltzmann method primitives (D2Q9, D3Q19, D3Q27) |
| `data_core` | Data loading, CSV/HDF5 I/O, dataset management |
| `control_core` | Feedback control, PID, adaptive parameter tuning |
| `optics_core` | Thin-film optics, Fresnel/TMM, Bessel functions, Mie theory |
| `scrolls_core` | Structured document parsing and scrolls pipeline |
| `docpipe` | Documentation pipeline and registry emission |

### Tier 2: Domain Physics

Physics and domain models built on foundation crates.

| Crate | Purpose |
|-------|---------|
| `gr_core` | General relativity, PPN constraints, photon-graviton mixing |
| `cosmology_core` | Friedmann equations, SNIa fitting, BAO analysis |
| `materials_core` | Drude-Lorentz models, metamaterial absorbers |
| `quantum_core` | Quantum gates, entanglement measures, decoherence |
| `snia_core` | Type Ia supernovae standardization and Hubble residuals |
| `spin_tomography_core` | Two-qubit state tomography, Bloch decomposition |
| `qgp_scaling` | Quark-gluon plasma scaling relations |

### Tier 3: Synthesis

Cross-domain synthesis and high-fidelity simulation lanes.

| Crate | Purpose |
|-------|---------|
| `vacuum_frustration` | Frustration density measurement, zero-divisor attractor |
| `lattice_filtration` | Persistent homology on algebraically modified fields |
| `neural_homotopy` | Neural residual learning (burn framework) |
| `lbm_3d` | CPU 3D lattice Boltzmann solver |
| `lbm_3d_cuda` | CUDA-accelerated 3D LBM (feature-gated) |
| `lbm_vulkan` | Vulkan compute LBM (SPIR-V shaders) |
| `cosmic_scheduler` | Multi-scale simulation orchestration |
| `cd_spin_bridge` | Bridge: vacuum frustration <-> spin tomography |

### Tier 4: Orchestration

Execution surface and integration boundaries.

| Crate | Purpose |
|-------|---------|
| `gororoba_engine` | Pipeline orchestration, adaptive GPU dispatch |
| `gororoba_cli` | 155 CLI binaries for analysis, verification, simulation |
| `gororoba_py` | Python bindings via PyO3 |
| `gororoba_contracts` | Cross-crate trait contracts |

## Building

Requires Rust nightly toolchain.

```sh
# Full workspace build
cargo build --workspace -j$(nproc)

# With GPU support (requires CUDA toolkit)
cargo build --workspace --features gpu -j$(nproc)

# Release build
cargo build --release --workspace -j$(nproc)
```

## Testing

```sh
# Run all workspace tests
cargo test --workspace -j$(nproc)

# Focused crate test (fast iteration)
cargo test -p algebra_core

# Clippy with warnings-as-errors
cargo clippy --workspace --all-targets -j$(nproc) -- -D warnings

# Full quality gate (clippy + tests)
cargo clippy --workspace -j$(nproc) -- -D warnings && cargo test --workspace -j$(nproc)
```

## Registry Model

Scientific claims, experiments, insights, and project metadata live in
`registry/*.toml` files.  These are the **source of truth**; markdown files
under `docs/` are generated mirrors.

| Registry | Purpose |
|----------|---------|
| `registry/claims.toml` | Scientific claims with status and evidence links |
| `registry/experiments.toml` | Experiment catalog and methodological records |
| `registry/insights.toml` | Insight catalog and synthesis observations |
| `registry/roadmap.toml` | Phase/sprint roadmap and workstream tracking |
| `registry/binaries.toml` | CLI binary inventory and descriptions |
| `registry/project.toml` | Project-level metadata and aggregate counts |
| `registry/schema_signatures.toml` | SHA-256 integrity hashes for all registries |

## Project Layout

```
crates/          Rust workspace crates (28 members)
src/             Python package code, verification scripts, analysis helpers
tests/           Python pytest suites and integration checks
registry/        Canonical TOML records (source of truth)
docs/            Generated markdown mirrors and documentation
data/            Artifacts, datasets, and experiment outputs
apps/            Studio and app assets
```

## License

MIT
