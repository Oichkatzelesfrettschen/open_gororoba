//! Rust port of `run_reynolds_sweep.sh`.
//! Executes the Reynolds Independence Sweep experiments.

use std::{
    fs::File,
    io::Write,
    process::{Command, Stdio},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting Reynolds Independence Sweep...");

    let experiments = vec![
        (8, 25000, "warp_exp_c_8.log"),
        (16, 25000, "warp_exp_c_16.log"),
        (32, 10000, "warp_exp_c_32.log"),
    ];

    for (size, steps, log_file) in experiments {
        println!("Running Size {}^3...", size);

        let mut child = Command::new("cargo")
            .args(&[
                "run",
                "--release",
                "--bin",
                "warp-gpu-experiment",
                "--",
                "--experiment",
                "C",
                "--size",
                &size.to_string(),
                "--steps",
                &steps.to_string(),
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let output = child.wait_with_output()?;

        let mut file = File::create(log_file)?;
        file.write_all(&output.stdout)?;
        file.write_all(&output.stderr)?;

        if !output.status.success() {
            eprintln!("Warning: Size {} sweep exited with an error.", size);
        }
    }

    println!("Sweep Complete.");
    Ok(())
}
