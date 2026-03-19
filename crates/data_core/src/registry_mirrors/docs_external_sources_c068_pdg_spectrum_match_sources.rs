//! # C068 PDG Spectrum Match Sources
//!
//! ## Claim
//!
//! `C-068`: "Sedenion 84-ZD interaction matrix eigenvalue spectrum matches PDG particle masses."
//!
//! ## Official Sources
//!
//! 1. Particle Data Group landing page (current update surface):
//!    - https://pdg.lbl.gov/2025/
//! 2. Review of Particle Physics update surface:
//!    - https://pdg.lbl.gov/2025/reviews/contents_sports.html
//! 3. Summary tables and listings entry point:
//!    - https://pdg.lbl.gov/2025/listings/contents_listings.html
//!
//! ## Local Reproducible Sources
//!
//! 1. Cached PDG 2025 source surfaces and distilled mass table:
//!    - `data/external/pdg_2025/index.html`
//!    - `data/external/pdg_2025/contents_listings.html`
//!    - `data/external/pdg_2025/contents_tables.html`
//!    - `data/external/pdg_2025/rpp2025-sum-leptons.pdf`
//!    - `data/external/pdg_2025/rpp2025-sum-quarks.pdf`
//!    - `data/external/pdg_2025/rpp2025-sum-gauge-higgs-bosons.pdf`
//!    - `data/external/pdg_2025/mass_subset.csv`
//! 2. Standard 84-ZD partner graph construction:
//!    - `crates/algebra_analysis/src/reggiani.rs`
//! 3. Rust parser for the distilled PDG table:
//!    - `crates/data_core/src/catalogs/pdg.rs`
//! 4. New deterministic blind-baseline audit:
//!    - `crates/gororoba_cli_data/src/bin/cd_pattern_baseline_audit.rs`
//! 5. Audit artifact:
//!    - `data/output/claims_falsification/cd_pattern_baseline_audit.toml`
//! 6. Registry linkage:
//!    - `registry/claims.toml`
//!    - `registry/experiments.toml`
//!    - `registry/claims_tasks.toml`
//!
//! ## Scope Note
//!
//! The repo no longer treats visual or label-substitution resemblance as evidence. The current falsifier requires:
//!
//! - exact spectrum reconstruction from the standard 84-ZD graph,
//! - blind subset matching against a fixed PDG mass ladder, and
//! - representation sensitivity checks across linear vs log mass scales.
//!
//! That standard is implemented in `cd-pattern-baseline-audit`.
//!
