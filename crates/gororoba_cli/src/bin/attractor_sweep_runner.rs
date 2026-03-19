//! attractor_sweep_runner: Pure Rust port of `src/scripts/analysis/c590_build_attractor_artifacts.sh`.
//! Orchestrates the attractor sweep experiments.

use std::env;
use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = Path::new("data/csv");
    fs::create_dir_all(out_dir)?;

    let runtime_csv = out_dir.join("c590_attractor_runtime_baseline.csv");
    let sweep_csv = out_dir.join("c590_attractor_ratio_sweep.csv");

    let fast_mode = env::var("C590_FAST_MODE").unwrap_or_else(|_| "0".to_string()) == "1";
    
    let runtime_dims = env::var("C590_RUNTIME_BASELINE_DIMS").unwrap_or_else(|_| "128,256".to_string());
    let mut sweep_dims = env::var("C590_SWEEP_DIMS").unwrap_or_else(|_| {
        if fast_mode {
            "16,32,64,128,256".to_string()
        } else {
            "16,32,64,128,256,512".to_string()
        }
    });
    
    let timeout_s = env::var("C590_TIMEOUT_S").unwrap_or_else(|_| {
        if fast_mode {
            "90".to_string()
        } else {
            "0".to_string()
        }
    });

    if env::var("C590_INCLUDE_DIM512").unwrap_or_else(|_| "0".to_string()) == "1" && !sweep_dims.contains("512") {
        sweep_dims.push_str(",512");
    }
    if env::var("C590_INCLUDE_DIM1024").unwrap_or_else(|_| "0".to_string()) == "1" && !sweep_dims.contains("1024") {
        sweep_dims.push_str(",1024");
    }

    println!("C590 runtime dims: {}", runtime_dims);
    println!("C590 sweep dims:   {}", sweep_dims);
    println!("C590 timeout(s):   {}", timeout_s);

    let run_cmd = |args: Vec<&str>, envs: Vec<(&str, &str)>| -> Result<(), Box<dyn std::error::Error>> {
        let mut cmd = if timeout_s != "0" {
            let mut c = Command::new("timeout");
            c.arg(format!("{}s", timeout_s));
            c.arg("cargo");
            c
        } else {
            Command::new("cargo")
        };

        cmd.args(args);
        for (k, v) in envs {
            cmd.env(k, v);
        }
        
        let status = cmd.status()?;
        if !status.success() {
            println!("Command failed with status: {:?}", status);
        }
        Ok(())
    };

    let target_dir = "/tmp/codex_target_algebra";

    // Debug Baseline
    run_cmd(
        vec!["run", "-q", "--bin", "c590_attractor_sweep", "--", "--dims", &runtime_dims, "--profile-tag", "debug", "--output", runtime_csv.to_str().unwrap()],
        vec![("CARGO_TARGET_DIR", target_dir)]
    )?;

    // Release Baseline
    run_cmd(
        vec!["run", "-q", "--release", "--bin", "c590_attractor_sweep", "--", "--dims", &runtime_dims, "--profile-tag", "release", "--output", runtime_csv.to_str().unwrap(), "--append"],
        vec![("CARGO_TARGET_DIR", target_dir)]
    )?;

    // Full Sweep
    run_cmd(
        vec!["run", "-q", "--release", "--bin", "c590_attractor_sweep", "--", "--dims", &sweep_dims, "--profile-tag", "release", "--output", sweep_csv.to_str().unwrap()],
        vec![("CARGO_TARGET_DIR", target_dir)]
    )?;

    println!("Wrote {}", runtime_csv.display());
    println!("Wrote {}", sweep_csv.display());

    Ok(())
}
