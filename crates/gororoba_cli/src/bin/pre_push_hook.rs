//! pre_push_hook: Pure Rust port of `.githooks/pre-push`.
//! Runs `makew gate-local` and `git-lfs pre-push`.

use std::{env, process::Command};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("[pre-push] running ./makew gate-local");

    // Runs the ./makew gate-local script directly.
    // TODO: call the gate-local cargo xtask subcommand once the makew logic is ported.
    let status = Command::new("./makew").arg("gate-local").status()?;

    if !status.success() {
        eprintln!("Pre-push hook failed: makew gate-local exited with error.");
        std::process::exit(status.code().unwrap_or(1));
    }

    // git-lfs pre-push
    if Command::new("git-lfs").arg("--version").output().is_ok() {
        let args: Vec<String> = env::args().collect();
        let mut child = Command::new("git-lfs")
            .arg("pre-push")
            .args(&args[1..])
            .spawn()?;
        let status = child.wait()?;
        if !status.success() {
            std::process::exit(status.code().unwrap_or(1));
        }
    }

    Ok(())
}
