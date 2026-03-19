//! makew: Pure Rust port of the `makew` shell script.
//! Unsets GNU make environment variables and executes `make`.

use std::env;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::new("make");
    
    // GNU make reads MAKEFLAGS from the environment.
    // Strip inherited make state here and let the repo's own concurrency policy take over.
    cmd.env_remove("MAKEFLAGS");
    cmd.env_remove("MFLAGS");
    cmd.env_remove("CARGO_MAKEFLAGS");
    cmd.env_remove("GNUMAKEFLAGS");
    cmd.env_remove("MAKELEVEL");

    let args: Vec<String> = env::args().skip(1).collect();
    cmd.args(args);

    let status = cmd.status()?;
    std::process::exit(status.code().unwrap_or(1));
}
