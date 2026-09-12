//! Compute a bounded worker count from CPU, memory, and cgroup limits.

use std::env;
use std::fs;
use std::process::ExitCode;

const MEMORY_MIB_PER_WORKER: usize = 2_048;

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

fn proc_available_memory_mib() -> Option<usize> {
    let contents = fs::read_to_string("/proc/meminfo").ok()?;
    let kilobytes = contents.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        (fields.next()? == "MemAvailable:")
            .then(|| fields.next()?.parse::<usize>().ok())
            .flatten()
    })?;
    Some(kilobytes / 1_024)
}

fn cgroup_available_memory_mib() -> Option<usize> {
    let maximum = fs::read_to_string("/sys/fs/cgroup/memory.max")
        .ok()?
        .trim()
        .parse::<usize>()
        .ok()?;
    let current = fs::read_to_string("/sys/fs/cgroup/memory.current")
        .ok()?
        .trim()
        .parse::<usize>()
        .ok()?;
    maximum
        .checked_sub(current)
        .map(|available_bytes| available_bytes / 1_048_576)
}

fn worker_budget(mode: BudgetMode, logical_cpus: usize, available_memory_mib: usize) -> usize {
    let cpu_budget = (logical_cpus / 2).max(1);
    let memory_budget = (available_memory_mib / MEMORY_MIB_PER_WORKER).max(1);
    let bounded = cpu_budget.min(memory_budget);
    match mode {
        BudgetMode::Local => bounded.min(2),
        BudgetMode::Ci => bounded,
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

    let memory_override = environment_positive(
        "GOROROBA_WORKER_TEST_MEMORY_MB",
        "available-memory override",
    )?;
    let available_memory_mib = memory_override.unwrap_or_else(|| {
        match (
            proc_available_memory_mib(),
            cgroup_available_memory_mib(),
        ) {
            (Some(host), Some(cgroup)) => host.min(cgroup),
            (Some(host), None) => host,
            (None, Some(cgroup)) => cgroup,
            (None, None) => MEMORY_MIB_PER_WORKER,
        }
    });

    Ok(worker_budget(mode, logical_cpus, available_memory_mib))
}

fn self_test() -> Result<(), String> {
    let cases = [
        (BudgetMode::Local, 1, 65_536, 1),
        (BudgetMode::Local, 128, 65_536, 2),
        (BudgetMode::Ci, 128, 8_192, 4),
        (BudgetMode::Ci, 2, 1, 1),
    ];
    for (mode, logical_cpus, available_memory_mib, expected) in cases {
        let observed = worker_budget(mode, logical_cpus, available_memory_mib);
        if observed != expected {
            return Err(format!(
                "worker-budget self-test failed: mode={mode:?} cpus={logical_cpus} memory_mib={available_memory_mib} expected={expected} observed={observed}"
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
        assert_eq!(worker_budget(BudgetMode::Local, 128, 65_536), 2);
    }

    #[test]
    fn ci_budget_obeys_memory_limit() {
        assert_eq!(worker_budget(BudgetMode::Ci, 128, 8_192), 4);
    }
}
