//! gororoba-doctor: Environment and Dependency Diagnostics.
//!
//! Checks for required system binaries and workspace configuration.
//!
//! Migrated from bin/doctor.py.

use anyhow::Result;
use std::process::Command;
use std::path::Path;

struct Check {
    name: &'static str,
    cmd: &'static str,
    args: &'static [&'static str],
}

const BINARY_CHECKS: &[Check] = &[
    Check { name: "rustc", cmd: "rustc", args: &["--version"] },
    Check { name: "cargo", cmd: "cargo", args: &["--version"] },
    Check { name: "docker", cmd: "docker", args: &["--version"] },
    Check { name: "coqc", cmd: "coqc", args: &["--version"] },
    Check { name: "latexmk", cmd: "latexmk", args: &["--version"] },
];

fn main() -> Result<()> {
    println!("--- MaNGA IFU Null Experiment [DOCTOR] ---");

    println!("\nSystem Binaries:");
    for check in BINARY_CHECKS {
        let status = Command::new(check.cmd)
            .args(check.args)
            .output();
        
        match status {
            Ok(output) if output.status.success() => {
                let version = String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .next()
                    .unwrap_or("unknown")
                    .to_string();
                println!("  [OK]   {:<10} ({})", check.name, version.trim());
            }
            _ => {
                println!("  [FAIL] {:<10} (MISSING or FAILED)", check.name);
            }
        }
    }

    println!("\nWorkspace Checks:");
    let paths = ["Cargo.toml", "Cargo.lock", "agents.toml", "deny.toml"];
    for p in &paths {
        if Path::new(p).exists() {
            println!("  [OK]   {}", p);
        } else {
            println!("  [FAIL] {} (MISSING)", p);
        }
    }

    println!("\nNext steps:");
    println!("  - Core: `make test` or `cargo test`.");
    println!("  - Docs: `cargo run -p xtask -- db-docs`.");
    
    Ok(())
}
