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
//! # Binary Catalog
//!
//! The `gororoba_cli` crate provides 40 binaries. Each binary is indexed in the
//! canonical SQLite control plane and exported into `registry/binaries.toml` as a
//! compatibility view.
//!
//! ## Experiment binaries
//!
//! | Binary | Experiment | Description |
//! |--------|------------|-------------|
//! | motif-census | E-001 | Cayley-Dickson motif census across doublings |
//! | multi-dataset-ultrametric | E-002 | GPU-accelerated multi-dataset ultrametric sweep |
//! | real-cosmo-fit | E-003 | Joint Pantheon+ / DESI BAO cosmological fit |
//! | kerr-shadow | E-004 | Bardeen shadow boundary for Kerr black holes |
//! | zd-search | E-005 | Zero-divisor graph search and invariants |
//! | gravastar-sweep | E-006 | TOV parameter sweep for gravastar models |
//! | tensor-network | E-007 | Tensor network simulator (Bell/GHZ/PEPS) |
//! | mass-clumping | E-008 | GWTC-3 mass distribution dip test |
//! | neg-dim-eigen | E-009 | Negative-dimension eigenvalue convergence |
//! | materials-baseline | E-010 | Materials science regression baselines |
//!
//! ## Infrastructure binaries
//!
//! | Binary | Description |
//! |--------|-------------|
//! | registry-check | Validate canonical control-plane integrity and export freshness |
//! | project-counter-sync | Synchronize top-level counters from the canonical control plane |
//! | migrate-claims | Legacy/bootstrap import from markdown claims mirror into SQLite compatibility exports |
//! | migrate-insights | Legacy/bootstrap import from markdown insights mirror into SQLite compatibility exports |
//! | generate-latex | Publication export: generate LaTeX appendices from canonical control-plane data |
//! | registry-emit | Export derived web and publication views; `control-plane-docs` is the standard refresh path for claims, insights, experiments, and theorems |
//! | extract-papers | Batch PDF extraction to structured TOML |
//! | export-hdf5 | Export computed data to HDF5 (feature-gated) |
//! | fetch-chime-frb | Download CHIME/FRB catalog data |
//! | fetch-datasets | Download all external astrophysical datasets |
//! | claims-audit | Audit claims evidence matrix |
//! | claims-verify | Verify claims against test suite |
//!
//! ## Domain-specific binaries
//!
//! | Binary | Description |
//! |--------|-------------|
//! | grin-trace | GRIN lens ray-tracing |
//! | mera-entropy | MERA tensor network entanglement entropy |
//! | frac-laplacian | Fractional Laplacian eigenvalue computation |
//! | lbm-poiseuille | Lattice Boltzmann Poiseuille flow |
//! | bounce-cosmology | Bouncing cosmology simulation |
//! | frac-schrodinger | Fractional Schrodinger equation solver |
//! | oct-field | Octonionic field theory computation |
//! | harper-chern | Harper-Hofstadter Chern number |
//! | tcmt-sweep | Temporal coupled-mode theory sweep |
//! | ema-calc | Exponential moving average calculator |
//!
//! ## Data analysis binaries
//!
//! | Binary | Description |
//! |--------|-------------|
//! | frb-ultrametric | FRB catalog ultrametric analysis |
//! | gw-merger-tree | Gravitational wave merger tree dendrograms |
//! | frb-cascades | FRB temporal cascade (SOC) analysis |
//! | dm-ultrametric | Dispersion measure ultrametric analysis |
//! | baire-compact | Baire metric compactness analysis |
//! | cosmic-dendrogram | Cosmic-scale dendrogram hierarchy |
//!
