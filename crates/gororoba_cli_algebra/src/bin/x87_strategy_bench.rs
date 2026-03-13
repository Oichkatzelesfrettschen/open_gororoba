//! Benchmark multicore x87 and AVX2 reduction strategies on pinned physical cores.

use std::{
    fs,
    hint::black_box,
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::Context;
use clap::Parser;
use csv::Writer;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use cd_kernel::{
    ParallelReductionStrategy, PhysicalCorePlan, avx2_dot, avx2_sum, parallel_dot, parallel_sum,
    x87_dot, x87_sum,
};

#[derive(Parser, Debug)]
#[command(name = "x87-strategy-bench")]
#[command(about = "Benchmark pinned multicore x87, AVX2, and hybrid reduction strategies")]
struct Args {
    /// Elements per benchmark dataset.
    #[arg(long, default_value = "1048576")]
    len: usize,

    /// Repetitions per strategy per dataset.
    #[arg(long, default_value = "7")]
    repeats: usize,

    /// Optional worker cap. Defaults to one worker per physical core.
    #[arg(long)]
    workers: Option<usize>,

    /// Explicit worker-count sweep, for example `--worker-counts 1,2,4,6`.
    #[arg(long, value_delimiter = ',')]
    worker_counts: Vec<usize>,

    /// RNG seed for reproducible random datasets.
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Output CSV path.
    #[arg(long, default_value = "data/csv/x87_strategy_bench.csv")]
    output: PathBuf,

    /// Optional Markdown summary path.
    #[arg(long)]
    summary: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy)]
enum WorkloadKind {
    Sum,
    Dot,
}

#[derive(Debug)]
enum Workload {
    Sum {
        name: &'static str,
        data: Vec<f64>,
    },
    Dot {
        name: &'static str,
        left: Vec<f64>,
        right: Vec<f64>,
    },
}

#[derive(Debug)]
struct Measurement {
    result: f64,
    median: Duration,
    best: Duration,
    worst: Duration,
}

#[derive(Debug)]
struct RunContext {
    len: usize,
    repeats: usize,
    seed: u64,
    worker_counts: Vec<usize>,
    detected_workers: usize,
    hostname: String,
    cpu_model: String,
}

#[derive(Debug)]
struct CsvRow {
    workload: &'static str,
    kind: WorkloadKind,
    strategy: &'static str,
    len: usize,
    workers: usize,
    core_ids: String,
    result: f64,
    reference: f64,
    abs_error: f64,
    ulp_error: u64,
    median_ns: u128,
    best_ns: u128,
    worst_ns: u128,
    throughput_melems_per_s: f64,
}

#[derive(Debug, Clone, Copy)]
struct RowTemplate<'a> {
    workload: &'static str,
    kind: WorkloadKind,
    len: usize,
    workers: usize,
    core_ids: &'a str,
    reference: f64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.len < 16 {
        anyhow::bail!("--len must be at least 16");
    }
    if args.repeats == 0 {
        anyhow::bail!("--repeats must be greater than zero");
    }

    let detected_plan = PhysicalCorePlan::pinned(None);
    let worker_counts = select_worker_counts(&args, detected_plan.worker_count());
    let context = build_run_context(&args, detected_plan.worker_count(), worker_counts.clone());
    let workloads = build_workloads(args.len, args.seed);

    println!(
        "Host {} on {}; detected {} physical-core workers; running worker counts {:?}",
        context.hostname, context.cpu_model, context.detected_workers, context.worker_counts
    );

    let mut rows = Vec::new();
    for workload in &workloads {
        rows.extend(run_serial_rows(workload, args.repeats));
        for &worker_count in &worker_counts {
            let plan = PhysicalCorePlan::pinned(Some(worker_count));
            let core_ids = plan
                .core_ids()
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(";");
            rows.extend(run_parallel_rows(
                workload,
                args.repeats,
                &plan,
                worker_count,
                &core_ids,
            ));
        }
    }

    print_summary(&rows);
    write_csv(&args.output, &context, &rows)?;
    if let Some(summary) = &args.summary {
        write_markdown_summary(summary, &context, &rows)?;
        println!("Wrote benchmark summary to {}", summary.display());
    }
    println!("Wrote benchmark table to {}", args.output.display());
    Ok(())
}

fn select_worker_counts(args: &Args, detected_workers: usize) -> Vec<usize> {
    let mut counts = if args.worker_counts.is_empty() {
        vec![args.workers.unwrap_or(detected_workers)]
    } else {
        args.worker_counts.clone()
    };

    counts.retain(|count| *count > 0);
    for count in &mut counts {
        *count = (*count).min(detected_workers.max(1));
    }
    counts.sort_unstable();
    counts.dedup();
    if counts.is_empty() {
        vec![detected_workers.max(1)]
    } else {
        counts
    }
}

fn build_run_context(
    args: &Args,
    detected_workers: usize,
    worker_counts: Vec<usize>,
) -> RunContext {
    RunContext {
        len: args.len,
        repeats: args.repeats,
        seed: args.seed,
        worker_counts,
        detected_workers,
        hostname: detect_hostname(),
        cpu_model: detect_cpu_model(),
    }
}

fn build_workloads(len: usize, seed: u64) -> Vec<Workload> {
    let len_even = if len.is_multiple_of(2) { len } else { len + 1 };
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let positive_sum = (0..len)
        .map(|i| ((i % 1024) as f64 + 1.0) * 0.000_976_562_5)
        .collect();
    let cancellation_sum = (0..len)
        .map(|i| {
            if i % 2 == 0 {
                1.0e8
            } else {
                -(1.0e8 + (i as f64) * 1.0e-6)
            }
        })
        .collect();
    let random_left = (0..len).map(|_| rng.gen_range(-1.0_f64..1.0_f64)).collect();
    let random_right = (0..len).map(|_| rng.gen_range(-1.0_f64..1.0_f64)).collect();
    let ill_left = vec![1.0e9; len_even];
    let half = len_even / 2;
    let correction = 1.0 / (half as f64 * 1.0e9);
    let ill_right = (0..len_even)
        .map(|i| {
            if i < half {
                1.0e9
            } else {
                -(1.0e9 - correction)
            }
        })
        .collect();

    vec![
        Workload::Sum {
            name: "sum_positive",
            data: positive_sum,
        },
        Workload::Sum {
            name: "sum_cancellation",
            data: cancellation_sum,
        },
        Workload::Dot {
            name: "dot_random",
            left: random_left,
            right: random_right,
        },
        Workload::Dot {
            name: "dot_ill_conditioned",
            left: ill_left,
            right: ill_right,
        },
    ]
}

fn run_serial_rows(workload: &Workload, repeats: usize) -> Vec<CsvRow> {
    match workload {
        Workload::Sum { name, data } => run_sum_serial_rows(name, data, repeats),
        Workload::Dot { name, left, right } => run_dot_serial_rows(name, left, right, repeats),
    }
}

fn run_parallel_rows(
    workload: &Workload,
    repeats: usize,
    plan: &PhysicalCorePlan,
    worker_count: usize,
    core_ids: &str,
) -> Vec<CsvRow> {
    match workload {
        Workload::Sum { name, data } => {
            run_sum_parallel_rows(name, data, repeats, plan, worker_count, core_ids)
        }
        Workload::Dot { name, left, right } => {
            run_dot_parallel_rows(name, left, right, repeats, plan, worker_count, core_ids)
        }
    }
}

fn run_sum_serial_rows(workload: &'static str, data: &[f64], repeats: usize) -> Vec<CsvRow> {
    let reference = x87_sum(data);
    let template = RowTemplate {
        workload,
        kind: WorkloadKind::Sum,
        len: data.len(),
        workers: 1,
        core_ids: "serial",
        reference,
    };

    [
        (
            "serial_naive",
            measure(repeats, || data.iter().copied().sum()),
        ),
        ("serial_kahan", measure(repeats, || kahan_sum(data))),
        ("serial_x87", measure(repeats, || x87_sum(data))),
        ("serial_avx2", measure(repeats, || avx2_sum(data))),
    ]
    .into_iter()
    .map(|(strategy, measurement)| make_row(template, strategy, measurement))
    .collect()
}

fn run_sum_parallel_rows(
    workload: &'static str,
    data: &[f64],
    repeats: usize,
    plan: &PhysicalCorePlan,
    worker_count: usize,
    core_ids: &str,
) -> Vec<CsvRow> {
    let reference = x87_sum(data);
    let template = RowTemplate {
        workload,
        kind: WorkloadKind::Sum,
        len: data.len(),
        workers: worker_count,
        core_ids,
        reference,
    };

    [
        (
            ParallelReductionStrategy::X87PerChunk.label(),
            measure(repeats, || {
                parallel_sum(data, ParallelReductionStrategy::X87PerChunk, plan)
            }),
        ),
        (
            ParallelReductionStrategy::Avx2PerChunk.label(),
            measure(repeats, || {
                parallel_sum(data, ParallelReductionStrategy::Avx2PerChunk, plan)
            }),
        ),
        (
            ParallelReductionStrategy::Avx2PerChunkX87Final.label(),
            measure(repeats, || {
                parallel_sum(data, ParallelReductionStrategy::Avx2PerChunkX87Final, plan)
            }),
        ),
    ]
    .into_iter()
    .map(|(strategy, measurement)| make_row(template, strategy, measurement))
    .collect()
}

fn run_dot_serial_rows(
    workload: &'static str,
    left: &[f64],
    right: &[f64],
    repeats: usize,
) -> Vec<CsvRow> {
    let reference = x87_dot(left, right);
    let template = RowTemplate {
        workload,
        kind: WorkloadKind::Dot,
        len: left.len(),
        workers: 1,
        core_ids: "serial",
        reference,
    };

    [
        (
            "serial_naive",
            measure(repeats, || {
                left.iter()
                    .zip(right.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
            }),
        ),
        ("serial_kahan", measure(repeats, || kahan_dot(left, right))),
        ("serial_x87", measure(repeats, || x87_dot(left, right))),
        ("serial_avx2", measure(repeats, || avx2_dot(left, right))),
    ]
    .into_iter()
    .map(|(strategy, measurement)| make_row(template, strategy, measurement))
    .collect()
}

fn run_dot_parallel_rows(
    workload: &'static str,
    left: &[f64],
    right: &[f64],
    repeats: usize,
    plan: &PhysicalCorePlan,
    worker_count: usize,
    core_ids: &str,
) -> Vec<CsvRow> {
    let reference = x87_dot(left, right);
    let template = RowTemplate {
        workload,
        kind: WorkloadKind::Dot,
        len: left.len(),
        workers: worker_count,
        core_ids,
        reference,
    };

    [
        (
            ParallelReductionStrategy::X87PerChunk.label(),
            measure(repeats, || {
                parallel_dot(left, right, ParallelReductionStrategy::X87PerChunk, plan)
            }),
        ),
        (
            ParallelReductionStrategy::Avx2PerChunk.label(),
            measure(repeats, || {
                parallel_dot(left, right, ParallelReductionStrategy::Avx2PerChunk, plan)
            }),
        ),
        (
            ParallelReductionStrategy::Avx2PerChunkX87Final.label(),
            measure(repeats, || {
                parallel_dot(
                    left,
                    right,
                    ParallelReductionStrategy::Avx2PerChunkX87Final,
                    plan,
                )
            }),
        ),
    ]
    .into_iter()
    .map(|(strategy, measurement)| make_row(template, strategy, measurement))
    .collect()
}

fn make_row(template: RowTemplate<'_>, strategy: &'static str, measurement: Measurement) -> CsvRow {
    CsvRow {
        workload: template.workload,
        kind: template.kind,
        strategy,
        len: template.len,
        workers: template.workers,
        core_ids: template.core_ids.to_string(),
        result: measurement.result,
        reference: template.reference,
        abs_error: (measurement.result - template.reference).abs(),
        ulp_error: ulp_diff(measurement.result, template.reference),
        median_ns: measurement.median.as_nanos(),
        best_ns: measurement.best.as_nanos(),
        worst_ns: measurement.worst.as_nanos(),
        throughput_melems_per_s: template.len as f64 / measurement.median.as_secs_f64() / 1.0e6,
    }
}

fn measure<F>(repeats: usize, mut f: F) -> Measurement
where
    F: FnMut() -> f64,
{
    let mut durations = Vec::with_capacity(repeats);
    let mut result = 0.0_f64;
    for _ in 0..repeats {
        let start = Instant::now();
        result = black_box(f());
        durations.push(start.elapsed());
    }
    durations.sort_unstable();
    Measurement {
        result,
        median: durations[durations.len() / 2],
        best: durations[0],
        worst: durations[durations.len() - 1],
    }
}

fn kahan_sum(data: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64;
    for &x in data {
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

fn kahan_dot(left: &[f64], right: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64;
    for (&a, &b) in left.iter().zip(right.iter()) {
        let prod = a * b;
        let y = prod - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

fn ulp_diff(left: f64, right: f64) -> u64 {
    if !left.is_finite() || !right.is_finite() {
        return u64::MAX;
    }
    ordered_bits(left).abs_diff(ordered_bits(right))
}

fn ordered_bits(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits & (1_u64 << 63) != 0 {
        !bits
    } else {
        bits | (1_u64 << 63)
    }
}

fn print_summary(rows: &[CsvRow]) {
    println!(
        "{:<22} {:<4} {:<27} {:>7} {:>12} {:>12} {:>12}",
        "workload", "kind", "strategy", "workers", "median_ms", "abs_err", "ulp_err"
    );
    for row in rows {
        println!(
            "{:<22} {:<4} {:<27} {:>7} {:>12.3} {:>12.5e} {:>12}{}",
            row.workload,
            match row.kind {
                WorkloadKind::Sum => "sum",
                WorkloadKind::Dot => "dot",
            },
            row.strategy,
            row.workers,
            row.median_ns as f64 / 1.0e6,
            row.abs_error,
            row.ulp_error,
            if row.is_unstable() { "  unstable" } else { "" },
        );
    }
}

fn write_csv(path: &PathBuf, context: &RunContext, rows: &[CsvRow]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut writer = Writer::from_path(path)?;
    writer.write_record([
        "hostname",
        "cpu_model",
        "detected_workers",
        "requested_repeats",
        "seed",
        "worker_sweep",
        "workload",
        "kind",
        "strategy",
        "len",
        "workers",
        "core_ids",
        "result",
        "reference",
        "abs_error",
        "ulp_error",
        "median_ns",
        "best_ns",
        "worst_ns",
        "unstable",
        "throughput_melems_per_s",
    ])?;

    for row in rows {
        writer.write_record([
            context.hostname.clone(),
            context.cpu_model.clone(),
            context.detected_workers.to_string(),
            context.repeats.to_string(),
            context.seed.to_string(),
            format_worker_counts(&context.worker_counts),
            row.workload.to_string(),
            match row.kind {
                WorkloadKind::Sum => "sum".to_string(),
                WorkloadKind::Dot => "dot".to_string(),
            },
            row.strategy.to_string(),
            row.len.to_string(),
            row.workers.to_string(),
            row.core_ids.clone(),
            format!("{:.17e}", row.result),
            format!("{:.17e}", row.reference),
            format!("{:.17e}", row.abs_error),
            row.ulp_error.to_string(),
            row.median_ns.to_string(),
            row.best_ns.to_string(),
            row.worst_ns.to_string(),
            row.is_unstable().to_string(),
            format!("{:.6}", row.throughput_melems_per_s),
        ])?;
    }

    writer.flush()?;
    Ok(())
}

fn write_markdown_summary(
    path: &PathBuf,
    context: &RunContext,
    rows: &[CsvRow],
) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut out = String::new();
    out.push_str("# x87 Strategy Benchmark Summary\n\n");
    out.push_str("## Run Context\n\n");
    out.push_str(&format!("- Host: `{}`\n", context.hostname));
    out.push_str(&format!("- CPU: `{}`\n", context.cpu_model));
    out.push_str(&format!("- Problem size: `len={}`\n", context.len));
    out.push_str(&format!("- Repeats per row: `{}`\n", context.repeats));
    out.push_str(&format!("- RNG seed: `{}`\n", context.seed));
    out.push_str(&format!(
        "- Detected physical-core workers: `{}`\n",
        context.detected_workers
    ));
    out.push_str(&format!(
        "- Worker sweep: `{}`\n",
        format_worker_counts(&context.worker_counts)
    ));
    out.push_str(
        "- Stability heuristic: rows are marked unstable when `worst_ns > 5 * median_ns` or `best_ns * 2 < median_ns`.\n\n",
    );

    let mut workloads = rows.iter().map(|row| row.workload).collect::<Vec<_>>();
    workloads.sort_unstable();
    workloads.dedup();

    for workload in workloads {
        let workload_rows = rows
            .iter()
            .filter(|row| row.workload == workload)
            .collect::<Vec<_>>();
        let baseline_ns = workload_rows
            .iter()
            .find(|row| row.strategy == "serial_x87")
            .context("missing serial_x87 baseline row")?
            .median_ns;

        out.push_str(&format!("## {workload}\n\n"));
        out.push_str(&recommendation_block(&workload_rows));
        out.push('\n');
        out.push_str(
            "| Strategy | Workers | Median ms | Speedup vs serial x87 | Abs error | ULP error | Stable |\n",
        );
        out.push_str("|---|---:|---:|---:|---:|---:|---|\n");
        for row in workload_rows {
            let speedup = baseline_ns as f64 / row.median_ns as f64;
            out.push_str(&format!(
                "| {} | {} | {:.3} | {:.3} | {:.5e} | {} | {} |\n",
                row.strategy,
                row.workers,
                row.median_ns as f64 / 1.0e6,
                speedup,
                row.abs_error,
                row.ulp_error,
                if row.is_unstable() { "no" } else { "yes" },
            ));
        }
        out.push('\n');
    }

    fs::write(path, out)?;
    Ok(())
}

fn recommendation_block(rows: &[&CsvRow]) -> String {
    let fastest_overall = fastest_prefer_stable(rows, |_| true);
    let fastest_exact = fastest_prefer_stable(rows, |row| row.ulp_error == 0);
    let fastest_parallel_exact =
        fastest_prefer_stable(rows, |row| row.workers > 1 && row.ulp_error == 0);
    let fastest_parallel_near =
        fastest_prefer_stable(rows, |row| row.workers > 1 && row.ulp_error <= 1);

    let mut out = String::new();
    out.push_str("Recommendations:\n\n");
    out.push_str(&format_recommendation("Fastest overall", fastest_overall));
    out.push_str(&format_recommendation("Fastest exact", fastest_exact));
    out.push_str(&format_recommendation(
        "Fastest exact parallel lane",
        fastest_parallel_exact,
    ));
    out.push_str(&format_recommendation(
        "Fastest <=1 ULP parallel lane",
        fastest_parallel_near,
    ));
    out
}

fn fastest_prefer_stable<'a, F>(rows: &[&'a CsvRow], mut predicate: F) -> Option<&'a CsvRow>
where
    F: FnMut(&CsvRow) -> bool,
{
    rows.iter()
        .copied()
        .filter(|row| predicate(row) && !row.is_unstable())
        .min_by_key(|row| row.median_ns)
        .or_else(|| {
            rows.iter()
                .copied()
                .filter(|row| predicate(row))
                .min_by_key(|row| row.median_ns)
        })
}

impl CsvRow {
    fn is_unstable(&self) -> bool {
        let median = self.median_ns.max(1);
        self.worst_ns > median.saturating_mul(5) || self.best_ns.saturating_mul(2) < median
    }
}

fn format_recommendation(label: &str, row: Option<&CsvRow>) -> String {
    match row {
        Some(row) => format!(
            "- {}: `{}` with {} worker(s), {:.3} ms, abs_err {:.5e}, ulp {}{}\n",
            label,
            row.strategy,
            row.workers,
            row.median_ns as f64 / 1.0e6,
            row.abs_error,
            row.ulp_error,
            if row.is_unstable() { " [unstable]" } else { "" },
        ),
        None => format!("- {}: none\n", label),
    }
}

fn format_worker_counts(worker_counts: &[usize]) -> String {
    worker_counts
        .iter()
        .map(|count| count.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn detect_hostname() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            fs::read_to_string("/etc/hostname")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| "unknown-host".to_string())
}

fn detect_cpu_model() -> String {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|cpuinfo| {
            cpuinfo.lines().find_map(|line| {
                line.strip_prefix("model name\t: ")
                    .map(|value| value.trim().to_string())
            })
        })
        .or_else(|| {
            fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
                .ok()
                .map(|value| format!("unknown-cpu (governor {})", value.trim()))
        })
        .unwrap_or_else(|| "unknown-cpu".to_string())
}

#[cfg(test)]
mod tests {
    use super::{Args, CsvRow, WorkloadKind, select_worker_counts};

    #[test]
    fn worker_counts_are_sorted_deduped_and_capped() {
        let args = Args {
            len: 1024,
            repeats: 3,
            workers: None,
            worker_counts: vec![6, 2, 8, 2, 0, 4],
            seed: 42,
            output: "ignored.csv".into(),
            summary: None,
        };

        let counts = select_worker_counts(&args, 4);
        assert_eq!(counts, vec![2, 4]);
    }

    #[test]
    fn instability_heuristic_flags_large_spread() {
        let stable = CsvRow {
            workload: "stable",
            kind: WorkloadKind::Sum,
            strategy: "serial_x87",
            len: 1024,
            workers: 1,
            core_ids: "serial".to_string(),
            result: 1.0,
            reference: 1.0,
            abs_error: 0.0,
            ulp_error: 0,
            median_ns: 100,
            best_ns: 90,
            worst_ns: 300,
            throughput_melems_per_s: 1.0,
        };
        assert!(!stable.is_unstable());

        let unstable = CsvRow {
            worst_ns: 600,
            ..stable
        };

        assert!(unstable.is_unstable());
    }
}
