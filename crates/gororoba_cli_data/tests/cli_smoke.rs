//! CLI smoke tests: verify that representative binaries can parse --help.
//!
//! WHY: After extracting nanograv modules to data_core (PH-3 B2b) and
//! completing the three-layer facade test suite (PH-3 B3), these tests confirm
//! that binary initialization (clap parser construction, static imports,
//! link-time symbol resolution) still works end-to-end for every domain.
//!
//! HOW: cargo test sets CARGO_BIN_EXE_<name> for each [[bin]] in the crate.
//! We invoke --help and assert exit code 0 -- this exercises the full
//! clap derive macro expansion and proves the binary is linkable.
//!
//! COVERAGE: One or two binaries per domain category, chosen to maximize
//! breadth of link-time symbol resolution across the dependency graph.
//! Full enumeration of all ~100 binaries is omitted to keep test-suite build
//! times reasonable; the selected set catches regressions in the major crate
//! entry points (clap, governance, nanograv, registry, data, publishing).

use std::process::Command;

/// Run `<binary> <lane...> --help` and assert it exits successfully with
/// non-empty stdout.
fn assert_help_succeeds_for(bin_env: &str, lane: &[&str]) {
    let bin_path = std::env::var(bin_env).unwrap_or_else(|_| {
        panic!("env var {bin_env} not set -- is the binary declared in Cargo.toml?")
    });
    let output = Command::new(&bin_path)
        .args(lane)
        .arg("--help")
        .output()
        .unwrap_or_else(|err| panic!("failed to run {bin_path}: {err}"));
    let invocation = if lane.is_empty() {
        bin_env.to_string()
    } else {
        format!("{bin_env} {}", lane.join(" "))
    };
    assert!(
        output.status.success(),
        "{invocation} --help exited with {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.is_empty(), "{invocation} --help produced no stdout");
}

fn assert_help_succeeds(bin_env: &str) {
    assert_help_succeeds_for(bin_env, &[]);
}

/// A lane inside a collapsed dispatcher. Reaching the lane's own clap parser
/// takes the subcommand name first, so this proves the same link-time and
/// derive-expansion invariant the flat-binary form used to prove.
fn assert_lane_help_succeeds(bin_env: &str, lane: &str) {
    assert_help_succeeds_for(bin_env, &[lane]);
}

// =============================================================================
// Claims domain (3 binaries)
// =============================================================================

#[test]
fn smoke_claims_audit_help() {
    assert_help_succeeds("CARGO_BIN_EXE_claims-audit");
}

#[test]
fn smoke_claims_verify_help() {
    assert_help_succeeds("CARGO_BIN_EXE_claims-verify");
}

#[test]
fn smoke_claims_consolidate_help() {
    assert_help_succeeds("CARGO_BIN_EXE_claims-consolidate");
}

// =============================================================================
// Registry domain (3 binaries)
// =============================================================================

#[test]
fn smoke_registry_check_help() {
    assert_help_succeeds("CARGO_BIN_EXE_registry-check");
}

#[test]
fn smoke_registry_emit_help() {
    assert_help_succeeds("CARGO_BIN_EXE_registry-emit");
}

#[test]
fn smoke_project_counter_sync_help() {
    assert_help_succeeds("CARGO_BIN_EXE_project-counter-sync");
}

// =============================================================================
// Governance domain (5 binaries)
// =============================================================================

#[test]
fn smoke_registry_integrity_help() {
    assert_help_succeeds("CARGO_BIN_EXE_registry-integrity");
}

#[test]
fn smoke_governance_verify_help() {
    assert_help_succeeds("CARGO_BIN_EXE_governance-verify");
}

#[test]
fn smoke_data_governance_gate_help() {
    assert_help_succeeds("CARGO_BIN_EXE_data-governance-gate");
}

#[test]
fn smoke_verify_registry_mirror_freshness_help() {
    assert_help_succeeds("CARGO_BIN_EXE_verify-registry-mirror-freshness");
}

#[test]
fn smoke_agents_render_help() {
    assert_help_succeeds("CARGO_BIN_EXE_agents-render");
}

// =============================================================================
// Data / fetch domain (3 binaries)
// =============================================================================

#[test]
fn smoke_lotss_fetch_help() {
    assert_help_succeeds("CARGO_BIN_EXE_lotss-fetch");
}

#[test]
fn smoke_fetch_datasets_help() {
    assert_help_succeeds("CARGO_BIN_EXE_fetch-datasets");
}

#[test]
fn smoke_heliosphere_stage_help() {
    assert_help_succeeds("CARGO_BIN_EXE_heliosphere-stage");
}

// =============================================================================
// Migration domain (2 binaries)
// =============================================================================

#[test]
fn smoke_migrate_claims_help() {
    assert_help_succeeds("CARGO_BIN_EXE_migrate-claims");
}

#[test]
fn smoke_migrate_insights_help() {
    assert_help_succeeds("CARGO_BIN_EXE_migrate-insights");
}

// =============================================================================
// Publishing domain (2 binaries)
// paper-build uses a hand-rolled subcommand dispatch that rejects --help;
// extract-papers is a standard clap binary from the same domain.
// =============================================================================

#[test]
fn smoke_generate_latex_help() {
    assert_help_succeeds("CARGO_BIN_EXE_generate-latex");
}

#[test]
fn smoke_extract_papers_help() {
    assert_help_succeeds("CARGO_BIN_EXE_extract-papers");
}

// =============================================================================
// Spectral / quantum chaos domain (2 binaries)
// =============================================================================

#[test]
fn smoke_zd_spectral_dimension_help() {
    assert_lane_help_succeeds("CARGO_BIN_EXE_zd-spectral", "dimension");
}

#[test]
fn smoke_zd_quantum_chaos_help() {
    assert_lane_help_succeeds("CARGO_BIN_EXE_zd-spectral", "quantum-chaos");
}

// =============================================================================
// Gates / analysis domain (3 binaries)
// rho-trace-gate and simulation-trace-gate open their first argument as a file
// path (no clap); third-party-source-verifier uses clap and covers the domain.
// =============================================================================

#[test]
fn smoke_third_party_source_verifier_help() {
    assert_help_succeeds("CARGO_BIN_EXE_third-party-source-verifier");
}

#[test]
fn smoke_evidence_package_help() {
    assert_help_succeeds("CARGO_BIN_EXE_evidence-package");
}

#[test]
fn smoke_falsification_audit_help() {
    assert_help_succeeds("CARGO_BIN_EXE_falsification-audit");
}

// =============================================================================
// Nanograv domain (3 binaries -- exercises re-export path from data_core)
// =============================================================================

#[test]
fn smoke_nanograv_propagation_audit_help() {
    assert_lane_help_succeeds("CARGO_BIN_EXE_nanograv", "propagation-audit");
}

#[test]
fn smoke_nanograv_timing_inventory_help() {
    assert_lane_help_succeeds("CARGO_BIN_EXE_nanograv", "timing-inventory");
}

#[test]
fn smoke_nanograv_timing_refit_preflight_help() {
    assert_lane_help_succeeds("CARGO_BIN_EXE_nanograv", "timing-refit-preflight");
}

// =============================================================================
// Inventory / knowledge domain (2 binaries)
// =============================================================================

#[test]
fn smoke_knowledge_atoms_help() {
    assert_help_succeeds("CARGO_BIN_EXE_knowledge-atoms");
}

#[test]
fn smoke_workspace_routing_help() {
    assert_help_succeeds("CARGO_BIN_EXE_workspace-routing");
}
