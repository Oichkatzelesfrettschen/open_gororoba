# Open Gororoba

Research workbench for Cayley-Dickson algebra and physics applications.
GPL-2.0-only. Rust 2024 edition workspace.

## Project Structure

```
open_gororoba/
  crates/                   # 27 Rust workspace members
    algebra_core/           # CD algebras, box-kites, Lie algebras, E7/E8
    cosmology_core/         # Gravastar, bypass models, TOV solver
    data_core/              # Data providers, catalogs (Wow!, GWTC, FRB, SDSS)
    gororoba_cli/           # 140 binaries (experiment runners, analysis tools)
    gororoba_engine/        # Synthesis engine (4-thesis pipeline)
    gr_core/                # General relativity (acoustic metric, Hawking)
    lbm_core/               # Lattice Boltzmann (D3Q19, turbulence)
    lbm_3d/                 # 3D LBM CPU implementation
    lbm_3d_cuda/            # CUDA BF16 LBM kernels (cudarc 0.19.1)
    lbm_vulkan/             # Vulkan compute LBM (wgpu)
    materials_core/         # Optical database (30 materials), Casimir, KK
    neural_homotopy/        # Burn ML correction tensor (MLP 256-128-64-16)
    optics_core/            # GRIN raytracing, TCMT, absorber benchmarks
    quantum_core/           # PEPS tensor networks, GPU contraction
    spectral_core/          # Ghost spectral analysis, wavelets, CHSH bridge
    stats_core/             # Homology, VR persistence, ultrametrics, bootstrap
    vacuum_frustration/     # ZD resonance, algebraic lensing, Kubo coupling
    ...                     # + 10 more (snia_core, scrolls_core, etc.)
  registry/                 # 131 TOML registries (source of truth)
    claims.toml             # 818 claims (0 Proposed, all adjudicated)
    experiments.toml        # 70 experiments (68 completed, 2 deferred)
    insights.toml           # 95 insights
    binaries.toml           # 136 registered binaries
    project.toml            # Counters and sprint history
    roadmap.toml            # 11 workstreams (8 done, 3 active)
    bibliography_normalized.toml  # 214+ BIB entries with DOIs
    ...                     # + 120 more (claims_atoms, lacunae, etc.)
  data/                     # Runtime data (mostly gitignored)
    csv/                    # 347 claim-evidence CSVs (tracked)
    external/               # Downloaded datasets (gitignored, ~20K files)
    artifacts/              # Generated images/videos (gitignored)
  docs/
    latex/                  # LLM-Scaffold paper (118pp, 0 LaTeX errors)
    external_sources/       # Provenance documentation
    book/                   # mdbook chapters
  src/
    ghost_stats/            # Python bridge (7 scripts: Lomb-Scargle, IAAFT, etc.)
  apps/
    gororoba_studio/        # Web UI for experiment monitoring
```

## Build

```sh
cargo build --workspace -j$(nproc)           # debug
cargo build --workspace --release -j$(nproc) # release
cargo test --workspace -j$(nproc)            # 4084 tests
cargo clippy --workspace -j$(nproc)          # warnings-as-errors
```

## Key Binaries

| Binary | Purpose |
|--------|---------|
| `rho-ghost-fft` | Ghost spectral analysis (csv, batch, rigorous, bootstrap) |
| `claims-consolidate` | Claim lifecycle management (analyze, normalize, merge) |
| `evidence-package` | Reproducibility bundle with checksums |
| `gororoba-studio` | Web dashboard for experiment telemetry |
| `generate-latex` | Paper + appendix generation from TOML |
| `ghost-spectral-audit` | Multi-method falsification (7 statistical tests) |
| `demo-optical-titanates` | Optical database thesis demonstrations |
| `cd-tower-report` | Comprehensive CD algebra tower analysis |

## Registry (TOML-First)

All project metadata lives in `registry/*.toml`. Markdown files are
generated mirrors. Edit TOML, not markdown.

Key registries:
- `claims.toml` -- 818 claims with status, evidence, verify/refute criteria
- `experiments.toml` -- 70 experiments with run commands and SHA-256 checksums
- `roadmap.toml` -- 11 workstreams tracking project phases
- `lacunae.toml` -- 175 tracked gaps (0 open)
- `bibliography_normalized.toml` -- 214+ references with DOIs

## Gitignored (Reproducible)

These directories are gitignored but regenerable:
- `data/external/` -- Downloaded datasets (~7 GB, via `dataset-fetch-all`)
- `data/artifacts/images/` -- Generated plots (via experiment binaries)
- `data/h5/` -- HDF5 simulation snapshots
- `data/thesis_lab*/` -- Thesis experiment outputs
- `data/kubo_transport/` -- Kubo coupling results
- `target/` -- Rust build artifacts
- `docs/latex/out/` -- LaTeX PDF output

## Four Theses Framework

The project tests four falsifiable theses about Cayley-Dickson algebras:

1. **T1 (Scalar-TOV Correlation)**: CD scalar frustration correlates with
   TOV stellar structure observables (r=-1.0, Cassini gate PASS)
2. **T2 (Non-Newtonian Viscosity)**: CD associator drives BGK shear
   thickening in lattice Boltzmann (ratio=1.254 at 64^3)
3. **T3 (Neural Homotopy Correction)**: Burn MLP reduces pentagon
   violation 78% (2.50 -> 0.547)
4. **T4 (Latency Law)**: Sedenion-keyed 3D toroidal walk follows
   inverse-square power law (R^2=0.995, gamma=-2.41)

All four theses pass simultaneously via the synthesis engine (E-033).

## Sprint History

See `registry/project.toml` for per-sprint summaries (S8 through S55).
Current: Sprint 55 (Taxonomy Verification + Repo Hygiene).
