// SPDX-License-Identifier: GPL-2.0-or-later
//
// The deprecated `export` alias names its replacement and refuses to run.

use std::process::Command;

#[test]
fn deprecated_export_alias_names_the_scan_subcommand_and_exits_nonzero() {
    let output = Command::new(env!("CARGO_BIN_EXE_provenance"))
        .args(["export", "--help"])
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
    let output = Command::new(env!("CARGO_BIN_EXE_provenance"))
        .args(["export-artifact-scan", "--help"])
        .output()
        .expect("run provenance");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("host filesystem state"), "{stdout}");
    assert!(stdout.contains("--allow-shrink"), "{stdout}");
}
