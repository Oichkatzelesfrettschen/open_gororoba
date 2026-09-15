//! Compute a worker count from the CPUs available to the process.

use std::process::ExitCode;

fn detected_worker_budget() -> Result<usize, String> {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .map_err(|error| format!("failed to detect available CPUs: {error}"))
}

fn run() -> Result<(), String> {
    if std::env::args_os().nth(1).is_some() {
        return Err("usage: detect-worker-budget".to_owned());
    }
    println!("{}", detected_worker_budget()?);
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("detect-worker-budget: {error}");
            ExitCode::from(2)
        }
    }
}
