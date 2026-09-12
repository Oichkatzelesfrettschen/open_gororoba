//! Compute a worker count from the CPUs available to the process.

use std::env;
use std::process::ExitCode;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BudgetMode {
    Local,
    Ci,
}

impl BudgetMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "local" => Ok(Self::Local),
            "ci" => Ok(Self::Ci),
            _ => Err(format!("unknown worker-budget mode {value:?}")),
        }
    }
}

fn parse_positive(value: &str, description: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{description} must be a positive integer, got {value:?}"))?;
    if parsed == 0 {
        return Err(format!(
            "{description} must be a positive integer, got {value:?}"
        ));
    }
    Ok(parsed)
}

fn environment_positive(name: &str, description: &str) -> Result<Option<usize>, String> {
    match env::var(name) {
        Ok(value) => parse_positive(&value, description).map(Some),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(_)) => Err(format!("{name} must contain UTF-8 text")),
    }
}

fn worker_budget(mode: BudgetMode, logical_cpus: usize) -> usize {
    match mode {
        BudgetMode::Local => (logical_cpus / 2).max(1).min(2),
        BudgetMode::Ci => logical_cpus.max(1),
    }
}

fn detected_worker_budget(mode: BudgetMode) -> Result<usize, String> {
    let logical_cpus = environment_positive(
        "GOROROBA_WORKER_TEST_CPUS",
        "worker CPU count override",
    )?
    .unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(2)
    });

    Ok(worker_budget(mode, logical_cpus))
}

fn self_test() -> Result<(), String> {
    let cases = [
        (BudgetMode::Local, 1, 1),
        (BudgetMode::Local, 128, 2),
        (BudgetMode::Ci, 4, 4),
        (BudgetMode::Ci, 128, 128),
    ];
    for (mode, logical_cpus, expected) in cases {
        let observed = worker_budget(mode, logical_cpus);
        if observed != expected {
            return Err(format!(
                "worker-budget self-test failed: mode={mode:?} cpus={logical_cpus} expected={expected} observed={observed}"
            ));
        }
    }
    if BudgetMode::parse("invalid").is_ok() {
        return Err("worker-budget self-test accepted an invalid mode".to_owned());
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let operation = arguments.next().ok_or_else(|| {
        "usage: detect-worker-budget <local|ci|--self-test>".to_owned()
    })?;
    if arguments.next().is_some() {
        return Err("detect-worker-budget accepts exactly one argument".to_owned());
    }
    if operation == "--self-test" {
        self_test()?;
        return Ok(());
    }
    println!("{}", detected_worker_budget(BudgetMode::parse(&operation)?)?);
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

#[cfg(test)]
mod tests {
    use super::{BudgetMode, worker_budget};

    #[test]
    fn local_budget_reserves_capacity() {
        assert_eq!(worker_budget(BudgetMode::Local, 128), 2);
    }

    #[test]
    fn ci_budget_uses_all_available_cpus() {
        assert_eq!(worker_budget(BudgetMode::Ci, 128), 128);
    }
}
