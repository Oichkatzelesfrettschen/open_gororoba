//! Smoke tests for binaries owned by the governance CLI package.

use std::process::Command;

#[test]
fn registry_integrity_help_succeeds() {
    let binary = env!("CARGO_BIN_EXE_registry-integrity");
    let output = Command::new(binary)
        .arg("--help")
        .output()
        .expect("execute registry-integrity --help");
    assert!(
        output.status.success(),
        "registry-integrity --help exited with {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !output.stdout.is_empty(),
        "registry-integrity --help produced no stdout"
    );
}
