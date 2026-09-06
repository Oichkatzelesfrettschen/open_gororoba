//! pre_push_hook: Pure Rust port of `.githooks/pre-push`.
//! Runs `make validate-local`.

use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("[pre-push] running make validate-local");

    let status = Command::new("make").arg("validate-local").status()?;

    if !status.success() {
        eprintln!("Pre-push validation failed: make validate-local exited with error.");
        std::process::exit(status.code().unwrap_or(1));
    }

    Ok(())
}
