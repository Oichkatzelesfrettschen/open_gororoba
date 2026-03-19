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
//! # Insights
//!
//! Insights are authored canonically in the SQLite control plane at
//! `registry/canonical/control_plane.sqlite3`. The compatibility file
//! `registry/insights.toml` and rendered Markdown views are exports.
//!
//! Each insight records a significant finding or interpretation from the
//! computational census.
//!
//! ## Insight catalog
//!
//! | ID | Title | Status | Sprint |
//! |----|-------|--------|--------|
//! | I-001 | Macquart Relation Fills the Comoving Distance Gap | Verified | S6 |
//! | I-002 | Ultrametric Structure Lives in Representations, Not Scalars | Verified | S6 |
//! | I-003 | Existing Rust Crate Ecosystem for Cosmological Analysis | Verified | S6 |
//! | I-004 | Kodama Dendrogram and Real Observational Cosmology | Verified | S6 |
//! | I-005 | Ultrametric Structure is Radio-Transient-Specific | Superseded | S6 |
//! | I-006 | Motif Census Scaling Laws (dim=16..256) | Verified | S6 |
//! | I-007 | Kerr Geodesic Integrator Verification Summary | Verified | S6 |
//! | I-008 | Cross-Domain Ultrametric Analysis (5K Subsampling) | Superseded | S7 |
//! | I-009 | Elliptic Integral Crate Eliminates Carlson Port | Verified | S6 |
//! | I-010 | nalgebra 0.33/0.34 Version Split Blocks Autodiff | Open | S6 |
//! | I-011 | GPU Ultrametric Sweep (9 catalogs) | Verified | S7 |
//! | I-012 | The Pathion Cubic Anomaly | Verified | S8 |
//! | I-013 | The Hierarchy Fingerprint Theorem | Verified | S7 |
//! | I-014 | Cayley-Dickson External Data Cross-Validation | Cross-validated | S8 |
//! | I-015 | Monograph Theses Verification (Lattice Codebook) | Verified | S9 |
//! | I-016 | De Marrais Emanation Architecture | Verified | S10 |
//!
//! ## Supersession chain
//!
//! I-005 and I-008 are marked **superseded** because the GPU sweep (I-011) with
//! 10 million triples revealed that 5K subsampling destroyed the Hipparcos galactic
//! spatial signal, making the earlier "radio-transient-specific" conclusion too narrow.
//!
//! ## Validation and export
//!
//! Use the canonical DB flow first:
//!
//! ```sh
//! cargo run --release -p gororoba_cli_provenance --bin provenance -- \
//!   --db registry/canonical/control_plane.sqlite3 \
//!   import-legacy-control-plane
//!
//! cargo run --release -p gororoba_cli_provenance --bin provenance -- \
//!   --db registry/canonical/control_plane.sqlite3 \
//!   export-control-plane
//!
//! cargo run --release -p gororoba_cli_provenance --bin provenance -- \
//!   --db registry/canonical/control_plane.sqlite3 \
//!   verify-control-plane
//! ```
//!
//! SQLite-first inspection example:
//!
//! ```sh
//! cargo run --release -p gororoba_cli_provenance --bin provenance -- \
//!   --db registry/canonical/control_plane.sqlite3 \
//!   query insight I-001
//! ```
//!
//! The `generate-latex` binary still produces `docs/latex/insights_appendix.tex`
//! for publication packaging. That appendix is a rendered export, not a
//! canonical source.
//!
