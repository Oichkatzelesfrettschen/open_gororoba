//! Re-export integrity tests for the source_provenance module (PH-3 B3, Layer 1).
//!
//! WHY: `source_provenance` lives in gororoba_cli_data/src/source_provenance.rs but is
//! accessed through two hops: the file is included into provenance_ops via a `#[path]`
//! annotation, then re-exported back into gororoba_cli_data as
//! `pub use provenance_ops::source_provenance`. If either hop breaks (rename, removal,
//! visibility change), this file fails to compile -- catching the regression at the facade.
//!
//! These tests verify the public API surface as seen by a downstream consumer of
//! gororoba_cli_data, not by code internal to the crate.

#[allow(unused_imports)]
use gororoba_cli_data::source_provenance::{
    BuildSummary, SourceInfrastructureSummary, VerifySummary, default_repo_root,
};

/// Verify that default_repo_root() returns a path that looks like the workspace root.
///
/// WHY: The function derives the root from the crate's CARGO_MANIFEST_DIR at compile time.
/// If the crate is moved or the constant is changed, the path would point to a
/// nonexistent directory -- this test catches that immediately.
#[test]
fn source_provenance_default_repo_root_is_workspace() {
    let root = default_repo_root();
    assert!(
        root.exists(),
        "default_repo_root() returned nonexistent path: {root:?}"
    );
    assert!(
        root.join("Cargo.toml").exists(),
        "default_repo_root() must contain Cargo.toml, got: {root:?}"
    );
    assert!(
        root.join("Cargo.lock").exists(),
        "default_repo_root() must contain Cargo.lock (workspace), got: {root:?}"
    );
}

/// Verify that BuildSummary is constructable through the re-export path.
#[test]
fn build_summary_constructable_via_reexport() {
    let _summary = BuildSummary::default();
}

/// Verify that VerifySummary is constructable through the re-export path.
#[test]
fn verify_summary_constructable_via_reexport() {
    let _summary = VerifySummary::default();
}

/// Verify that SourceInfrastructureSummary is constructable through the re-export path.
#[test]
fn source_infrastructure_summary_constructable_via_reexport() {
    let _summary = SourceInfrastructureSummary::default();
}
