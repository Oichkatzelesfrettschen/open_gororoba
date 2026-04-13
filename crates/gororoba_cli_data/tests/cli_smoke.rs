//! CLI smoke tests: verify that representative binaries can parse --help.
//!
//! WHY: After extracting nanograv modules to data_core (PH-3 B2b) and
//! stripping heavy deps from the CLI surface, these tests confirm that
//! binary initialization (clap parser construction, static imports,
//! link-time symbol resolution) still works end-to-end.
//!
//! HOW: cargo test sets CARGO_BIN_EXE_<name> for each [[bin]] in the crate.
//! We invoke --help and assert exit code 0 -- this exercises the full
//! clap derive macro expansion and proves the binary is linkable.

use std::process::Command;

/// Run `<binary> --help` and assert it exits successfully.
fn assert_help_succeeds(bin_env: &str) {
    let bin_path = std::env::var(bin_env).unwrap_or_else(|_| {
        panic!(
            "env var {bin_env} not set -- is the binary declared in Cargo.toml?"
        )
    });
    let output = Command::new(&bin_path)
        .arg("--help")
        .output()
        .unwrap_or_else(|err| panic!("failed to run {bin_path}: {err}"));
    assert!(
        output.status.success(),
        "{bin_env} --help exited with {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.is_empty(),
        "{bin_env} --help produced no stdout"
    );
}

// -- Governance domain --------------------------------------------------------

#[test]
fn smoke_claims_audit_help() {
    assert_help_succeeds("CARGO_BIN_EXE_claims-audit");
}

#[test]
fn smoke_claims_verify_help() {
    assert_help_succeeds("CARGO_BIN_EXE_claims-verify");
}

#[test]
fn smoke_registry_check_help() {
    assert_help_succeeds("CARGO_BIN_EXE_registry-check");
}

// -- Data/fetch domain --------------------------------------------------------

#[test]
fn smoke_lotss_fetch_help() {
    assert_help_succeeds("CARGO_BIN_EXE_lotss-fetch");
}

// -- Governance gate tool -----------------------------------------------------

#[test]
fn smoke_integrity_resolution_help() {
    assert_help_succeeds("CARGO_BIN_EXE_integrity-resolution");
}

// -- Nanograv domain (exercises re-export path) -------------------------------

#[test]
fn smoke_nanograv_propagation_audit_help() {
    assert_help_succeeds("CARGO_BIN_EXE_nanograv-propagation-audit");
}
