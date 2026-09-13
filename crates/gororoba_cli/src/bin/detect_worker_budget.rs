//! Compute a worker count from the CPUs available to the process.

use std::env;
use std::process::ExitCode;

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

fn worker_budget(logical_cpus: usize) -> usize {
    logical_cpus.max(1)
}

fn detected_worker_budget() -> Result<usize, String> {
    let logical_cpus = match environment_positive(
        "GOROROBA_WORKER_TEST_CPUS",
        "worker CPU count override",
    )? {
        Some(logical_cpus) => logical_cpus,
        None => std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .map_err(|error| format!("failed to detect available CPUs: {error}"))?,
    };

    Ok(worker_budget(logical_cpus))
}

fn self_test() -> Result<(), String> {
    let cases = [(1, 1), (4, 4), (128, 128)];
    for (logical_cpus, expected) in cases {
        let observed = worker_budget(logical_cpus);
        if observed != expected {
            return Err(format!(
                "worker-budget self-test failed: cpus={logical_cpus} expected={expected} observed={observed}"
            ));
        }
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let operation = arguments.next();
    if arguments.next().is_some() {
        return Err("usage: detect-worker-budget [--self-test]".to_owned());
    }
    match operation.as_deref() {
        None => println!("{}", detected_worker_budget()?),
        Some("--self-test") => self_test()?,
        Some(_) => return Err("usage: detect-worker-budget [--self-test]".to_owned()),
    }
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
    use super::worker_budget;

    #[test]
    fn budget_uses_all_available_cpus() {
        assert_eq!(worker_budget(128), 128);
    }
}
