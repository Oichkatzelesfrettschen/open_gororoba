//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/docs_root_narratives.toml -->
//!
//! # Source Infrastructure Best Practices (Toml-First, Rust-First)
//!
//! ## Scope
//!
//! This repo uses `registry/artifact_source_of_truth.toml` as the single authoritative artifact ledger.
//! All scoped lane files are generated projections and must be reproducible from the master.
//!
//! ## Practical standard for this repository
//!
//! 1. One authoritative master.
//! - Master: `registry/artifact_source_of_truth.toml`
//! - Rule: no hand edits in projected lane files.
//!
//! 2. Deterministic projections.
//! - Lanes: `registry/source_lanes/*.toml`
//! - Generator: `cargo run -p gororoba_cli_data --bin provenance -- export`
//! - Verifier: `cargo run -p gororoba_cli_data --bin provenance -- verify`
//!
//! 3. Provenance retention.
//! - Keep both working and non-working mirrors.
//! - Never discard blocked links; classify and track manual intervention.
//!
//! 4. Reproducible verification gates.
//! - Always run rebuild + verify in sequence.
//! - Keep generated audit artifacts in `data/external/intake/...` and `reports/...`.
//!
//! 5. Rust-first control plane.
//! - Registry synthesis, audit, and reconciliation run through Rust CLI surfaces.
//! - Generated mirrors must be reproducible from committed TOML and markdown inputs.
//! - Keep generated outputs deterministic, ASCII-safe, and file-contract stable.
//!
//! ## External standards used as reference points
//!
//! - W3C PROV overview: https://www.w3.org/TR/prov-overview/
//! - FAIR principles paper: https://www.nature.com/articles/sdata201618
//! - Joint declaration of data citation principles: https://doi.org/10.25490/a97f-egyk
//! - DataCite Metadata Schema 4.5: https://schema.datacite.org/meta/kernel-4.5/
//! - GitHub citation files (`CITATION.cff`): https://docs.github.com/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files
//! - OpenLineage docs: https://openlineage.io/docs/
//!
//! ## Canonical workflow
//!
//! ```ignore
//! cargo run -p gororoba_cli_data --bin provenance -- export
//! cargo run -p gororoba_cli_data --bin provenance -- index
//! cargo run -p gororoba_cli_data --bin provenance -- link-audit
//! cargo run -p gororoba_cli_data --bin provenance -- recover
//! cargo run -p gororoba_cli_data --bin provenance -- verify
//! ```ignore
//!
