// SPDX-License-Identifier: GPL-2.0-or-later
//
// The deprecated `export` alias names its replacement and refuses to run.

use std::{path::PathBuf, process::Command};

/// Cargo defines CARGO_BIN_EXE_provenance when it builds the test through
/// `cargo test`; `clippy --all-targets` compiles the test without building the
/// binaries, so the path is resolved from the test executable's own directory
/// as a fallback.
fn provenance_bin() -> PathBuf {
    if let Some(path) = option_env!("CARGO_BIN_EXE_provenance") {
        return PathBuf::from(path);
    }
    let mut dir = std::env::current_exe().expect("current exe");
    dir.pop();
    if dir.ends_with("deps") {
        dir.pop();
    }
    dir.join("provenance")
}

#[test]
fn deprecated_export_alias_names_the_scan_subcommand_and_exits_nonzero() {
    let output = Command::new(provenance_bin())
        .args(["export"])
        .output()
        .expect("run provenance");
    assert!(!output.status.success(), "the alias must exit nonzero");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("export-artifact-scan"),
        "stderr must name the replacement: {stderr}"
    );
}

#[test]
fn export_artifact_scan_help_states_that_it_reads_host_filesystem_state() {
    let output = Command::new(provenance_bin())
        .args(["export-artifact-scan", "--help"])
        .output()
        .expect("run provenance");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("host filesystem state"), "{stdout}");
    assert!(stdout.contains("--allow-shrink"), "{stdout}");
}
