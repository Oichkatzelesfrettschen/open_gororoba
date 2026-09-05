//! Public facade compatibility for provenance_ops::source_provenance.
//!
//! Downstream users retain the gororoba_cli_data re-export while the
//! implementation and its tests belong to provenance_ops.

#[allow(unused_imports)]
use gororoba_cli_data::source_provenance::{
    BuildSummary, SourceInfrastructureSummary, VerifySummary, default_repo_root,
};

/// The public facade resolves the workspace through repo_root.
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
