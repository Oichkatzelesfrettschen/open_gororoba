//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! # open_gororoba
//!
//! A research workbench exploring whether algebraic structure in Cayley-Dickson
//! algebras (real -> complex -> quaternion -> octonion -> sedenion -> ...) can
//! explain or predict phenomena in quantum gravity, particle physics, and cosmology.
//!
//! The canonical documentation path is now:
//!
//! - crate rustdoc for API and module behavior
//! - this mdBook for architecture and navigation
//! - `registry/canonical/control_plane.sqlite3` for claims, experiments, theorem
//!   inventory, and evidence indexing
//! - `registry/*.toml` as generated compatibility exports for legacy tooling and
//!   lightweight browsing
//!
//! The LaTeX tree under `docs/latex/` is publication output, not the primary
//! source for current repo truth.
//!
//! ## Repo visual map
//!
//! ![Repo scope dashboard](./assets/repo_scope_dashboard_3160x2820.png)
//!
//! The scope dashboard is generated from live Cargo metadata plus canonical
//! registry state. The mdBook copies live under `docs/book/src/assets/`, while the
//! source PNGs and CSV summaries live under `data/artifacts/images/` and `data/csv/`.
//!
//! ## Scientific plates
//!
//! ![E-183 mass-phase manifold](./assets/science_e183_phase_plate_3160x2820.png)
//!
//! Generated companions:
//!
//! - `data/artifacts/images/science_gravastar_stability_plate_3160x2820.png`
//! - `data/artifacts/images/science_pathion_zero_divisor_interaction_graph_3160x2820.png`
//!
//! These plates are driven by live generated result lanes rather than repo
//! metadata:
//!
//! - `data/results/e183/*` for the MaNGA mass-phase field and cross-algebra correlation analysis.
//! - `data/csv/gravastar_radial_stability.csv`, `data/csv/gravastar_ligo_mass_sweep.csv`,
//!   and `data/csv/genesis_gravastar_bridge.csv` for the radial instability field and stable branch distribution.
//! - `data/csv/pathion_zd_edges.csv` and `data/csv/sedenion_zd_edges.csv` for the zero-divisor interaction graphs.
//! - `data/csv/sedenion_mass_spectrum.csv`, `data/csv/pathion_coupling_sweep.csv`,
//!   `data/csv/pathion_sink_compare.csv`, and `data/csv/sedenion_field_metrics_3D.csv`
//!   for the mass spectrum, coupling response, damping trajectory, and 3D field relaxation summaries.
//!
//! ## Canonical Inventory
//!
//! For current counts and inventory, prefer the Rust workspace and canonical
//! SQLite control plane over hard-coded narrative snapshots:
//!
//! - `crates/` for the active workspace members and public APIs
//! - `registry/project.toml` for synchronized top-level counters
//! - `registry/canonical/control_plane.sqlite3` for tracked claims, insights,
//!   experiments, binaries, and theorem rows
//! - `registry/claims.toml`, `registry/insights.toml`, `registry/experiments.toml`,
//!   and `registry/binaries.toml` as generated compatibility exports
//! - `papers/MANIFEST.toml` for the paper corpus
//!
//! ## Quick Start
//!
//! ```textsh
//! # Build everything
//! cargo build --workspace -j$(nproc)
//!
//! # Generate crate docs
//! cargo doc --workspace --no-deps
//!
//! # Build the web book
//! mdbook build docs/book
//!
//! # Run tests
//! cargo test --workspace -j$(nproc)
//!
//! # Run clippy (warnings-as-errors)
//! cargo clippy --workspace -j$(nproc) -- -D warnings
//!
//! # Check registry integrity
//! cargo run --release --bin registry-check
//!
//! # Synchronize project counters from registry sources
//! cargo run --release --bin project-counter-sync
//!
//! # Extract a paper to TOML
//! cargo run --release --bin extract-papers -- --only demarrais-2000-math0011260
//! ```text
//!
//! ## Codebase Structure
//!
//! ```text
//! crates/
//!   gororoba_algebra/ Cayley-Dickson facade, Clifford, wheels, p-adic, groups
//!   cd_kernel/        Pure Cayley-Dickson arithmetic kernel
//!   cosmology_core/   FLRW, bouncing models, dark energy
//!   gr_core/          Schwarzschild, Kerr, Novikov-Thorne, spectral bands
//!   materials_core/   Periodic table, metamaterial models
//!   optics_core/      GRIN lenses, ray tracing
//!   quantum_core/     Casimir, tensor networks, Grover
//!   stats_core/       Ultrametric, dip test, bootstrap, GPU
//!   spectral_core/    Fractional PDE, negative dimensions
//!   lbm_core/         Lattice Boltzmann
//!   control_core/     Control theory
//!   data_core/        Data loading, benchmarks
//!   docpipe/          PDF extraction (pdfium-render primary)
//!   gororoba_py/      PyO3 bindings (thin wrappers)
//!   gororoba_cli*/    Domain-specific operator surfaces
//! registry/           SQLite control plane plus generated compatibility registries
//! papers/             PDF collection + TOML extractions
//! docs/               Documentation and tracking
//! ```text
//!
//! Start with [Canonical Sources](./architecture/canonical-sources.md) if you are
//! deciding which doc surface to trust.
//!
