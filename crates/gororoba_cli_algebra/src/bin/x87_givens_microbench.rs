//! Measure the actual composed x87 Givens/transcendental micro-kernels.

#![cfg(target_arch = "x86_64")]

use std::{
    collections::BTreeSet,
    fs,
    hint::black_box,
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::Context;
use clap::Parser;
use csv::Writer;

use cd_kernel::{
    Ext80, atan2_ext80, givens_sincos_ext80, givens_sincos_f64, sincos_ext80, x87_atan2_sincos,
    x87_givens_diagonal_update, x87_givens_sincos,
};

#[derive(Parser, Debug)]
#[command(name = "x87-givens-microbench")]
#[command(about = "Benchmark the composed x87 Givens/transcendental micro-kernels")]
struct Args {
    /// Inner-loop iterations per repeat.
    #[arg(long, default_value = "200000")]
    iterations: usize,

    /// Repetitions per kernel/case.
    #[arg(long, default_value = "9")]
    repeats: usize,

    /// Output CSV path.
    #[arg(long, default_value = "reports/benchmarks/x87_givens_microbench.csv")]
    output: PathBuf,

    /// Optional comma-separated case filter.
    #[arg(long)]
    cases: Option<String>,

    /// Optional comma-separated kernel filter.
    #[arg(long)]
    kernels: Option<String>,

    /// Optional Markdown summary path.
    #[arg(long)]
    summary: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug)]
struct GivensCase {
    name: &'static str,
    y: f64,
    x: f64,
    app: f64,
    apq: f64,
    aqq: f64,
}

#[derive(Clone, Copy, Debug)]
struct Measurement {
    checksum: f64,
    median: Duration,
    best: Duration,
    worst: Duration,
}

#[derive(Clone, Debug)]
struct CsvRow {
    case: &'static str,
    kernel: &'static str,
    iterations: usize,
    repeats: usize,
    checksum: f64,
    median_ns: u128,
    best_ns: u128,
    worst_ns: u128,
    ns_per_call: f64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.iterations == 0 {
        anyhow::bail!("--iterations must be greater than zero");
    }
    if args.repeats == 0 {
        anyhow::bail!("--repeats must be greater than zero");
    }

    let requested_cases = parse_filter(args.cases.as_deref());
    let requested_kernels = parse_filter(args.kernels.as_deref());
    validate_requested(
        "case",
        &requested_cases,
        givens_cases().iter().map(|case| case.name),
    )?;
    validate_requested("kernel", &requested_kernels, kernel_names())?;

    let cases = filter_cases(&givens_cases(), &requested_cases);
    if cases.is_empty() {
        anyhow::bail!("selected case filter produced no benchmark cases");
    }
    let mut rows = Vec::new();
    for case in &cases {
        rows.extend(benchmark_case(
            *case,
            args.iterations,
            args.repeats,
            &requested_kernels,
        ));
    }
    if rows.is_empty() {
        anyhow::bail!("selected kernel filter produced no benchmark rows");
    }

    print_summary(&rows);
    write_csv(&args.output, &rows)?;
    if let Some(summary) = &args.summary {
        write_markdown_summary(summary, &rows)?;
        println!("Wrote x87 microbench summary to {}", summary.display());
    }
    println!("Wrote x87 microbench table to {}", args.output.display());
    Ok(())
}

fn givens_cases() -> Vec<GivensCase> {
    vec![
        GivensCase {
            name: "quarter_pi",
            y: 1.0,
            x: 1.0,
            app: 10.0,
            apq: 2.5,
            aqq: 6.0,
        },
        GivensCase {
            name: "obstruction_like",
            y: 96.0,
            x: 384.0,
            app: 312.0,
            apq: 48.0,
            aqq: -72.0,
        },
        GivensCase {
            name: "mixed_signs",
            y: -14.0,
            x: 5.5,
            app: -40.0,
            apq: -7.0,
            aqq: 12.0,
        },
        GivensCase {
            name: "tiny_angle",
            y: 1.0e-12,
            x: 1.0e3,
            app: 1.0e6,
            apq: 5.0e-4,
            aqq: 9.999e5,
        },
    ]
}

fn benchmark_case(
    case: GivensCase,
    iterations: usize,
    repeats: usize,
    requested_kernels: &BTreeSet<String>,
) -> Vec<CsvRow> {
    let ext_y = Ext80::from_f64(case.y);
    let ext_x = Ext80::from_f64(case.x);
    let precomputed = givens_sincos_f64(case.y, case.x).value;
    let (sin_t, cos_t) = precomputed;

    [
        (
            "x87_atan2_sincos",
            measure(iterations, repeats, || {
                let (sin_v, cos_v) = x87_atan2_sincos(black_box(case.y), black_box(case.x));
                black_box(sin_v + cos_v)
            }),
        ),
        (
            "givens_sincos_f64",
            measure(iterations, repeats, || {
                let status = givens_sincos_f64(black_box(case.y), black_box(case.x));
                black_box(status.value.0 + status.value.1 + status.status.0 as f64)
            }),
        ),
        (
            "x87_givens_sincos",
            measure(iterations, repeats, || {
                let (sin_v, cos_v) = x87_givens_sincos(black_box(case.y), black_box(case.x));
                black_box(sin_v + cos_v)
            }),
        ),
        (
            "atan2_ext80",
            measure(iterations, repeats, || {
                let angle = atan2_ext80(black_box(ext_y), black_box(ext_x));
                black_box(angle.value.to_f64() + angle.status.0 as f64)
            }),
        ),
        (
            "atan2_half_sincos_ext80",
            measure(iterations, repeats, || {
                let angle = atan2_ext80(black_box(ext_y), black_box(ext_x));
                let half = angle.value.scale_pow2(-1);
                let trig = sincos_ext80(black_box(half));
                let (sin_v, cos_v) = trig.value;
                black_box(
                    sin_v.to_f64() + cos_v.to_f64() + angle.status.0 as f64 + trig.status.0 as f64,
                )
            }),
        ),
        (
            "givens_sincos_ext80",
            measure(iterations, repeats, || {
                let result = givens_sincos_ext80(black_box(ext_y), black_box(ext_x));
                black_box(
                    result.value.sin.to_f64() + result.value.cos.to_f64() + result.status.0 as f64,
                )
            }),
        ),
        (
            "x87_givens_diagonal_update",
            measure(iterations, repeats, || {
                let (pp, qq) = x87_givens_diagonal_update(
                    black_box(sin_t),
                    black_box(cos_t),
                    black_box(case.app),
                    black_box(case.apq),
                    black_box(case.aqq),
                );
                black_box(pp + qq)
            }),
        ),
    ]
    .into_iter()
    .filter(|(kernel, _)| matches_filter(requested_kernels, kernel))
    .map(|(kernel, measurement)| CsvRow {
        case: case.name,
        kernel,
        iterations,
        repeats,
        checksum: measurement.checksum,
        median_ns: measurement.median.as_nanos(),
        best_ns: measurement.best.as_nanos(),
        worst_ns: measurement.worst.as_nanos(),
        ns_per_call: measurement.median.as_secs_f64() * 1.0e9 / iterations as f64,
    })
    .collect()
}

fn parse_filter(raw: Option<&str>) -> BTreeSet<String> {
    raw.into_iter()
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn filter_cases(cases: &[GivensCase], requested_cases: &BTreeSet<String>) -> Vec<GivensCase> {
    cases
        .iter()
        .copied()
        .filter(|case| matches_filter(requested_cases, case.name))
        .collect()
}

fn matches_filter(requested: &BTreeSet<String>, candidate: &str) -> bool {
    requested.is_empty() || requested.contains(candidate)
}

fn validate_requested<'a>(
    label: &str,
    requested: &BTreeSet<String>,
    available: impl Iterator<Item = &'a str>,
) -> anyhow::Result<()> {
    if requested.is_empty() {
        return Ok(());
    }
    let available: BTreeSet<&str> = available.collect();
    let missing: Vec<&str> = requested
        .iter()
        .map(String::as_str)
        .filter(|value| !available.contains(value))
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    anyhow::bail!("unknown {} filter value(s): {}", label, missing.join(", "));
}

fn kernel_names() -> impl Iterator<Item = &'static str> {
    [
        "x87_atan2_sincos",
        "givens_sincos_f64",
        "x87_givens_sincos",
        "atan2_ext80",
        "atan2_half_sincos_ext80",
        "givens_sincos_ext80",
        "x87_givens_diagonal_update",
    ]
    .into_iter()
}

fn measure<F>(iterations: usize, repeats: usize, mut f: F) -> Measurement
where
    F: FnMut() -> f64,
{
    let mut durations = Vec::with_capacity(repeats);
    let mut checksum = 0.0_f64;
    for _ in 0..repeats {
        let start = Instant::now();
        let mut sink = 0.0_f64;
        for _ in 0..iterations {
            sink += black_box(f());
        }
        checksum = black_box(sink);
        durations.push(start.elapsed());
    }
    durations.sort_unstable();
    Measurement {
        checksum,
        median: durations[durations.len() / 2],
        best: durations[0],
        worst: durations[durations.len() - 1],
    }
}

fn print_summary(rows: &[CsvRow]) {
    println!(
        "{:<18} {:<28} {:>12} {:>14}",
        "case", "kernel", "median_ms", "ns_per_call"
    );
    for row in rows {
        println!(
            "{:<18} {:<28} {:>12.3} {:>14.3}",
            row.case,
            row.kernel,
            row.median_ns as f64 / 1.0e6,
            row.ns_per_call,
        );
    }
}

fn write_csv(path: &PathBuf, rows: &[CsvRow]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut writer = Writer::from_path(path)?;
    writer.write_record([
        "case",
        "kernel",
        "iterations",
        "repeats",
        "checksum",
        "median_ns",
        "best_ns",
        "worst_ns",
        "ns_per_call",
    ])?;
    for row in rows {
        writer.write_record([
            row.case,
            row.kernel,
            &row.iterations.to_string(),
            &row.repeats.to_string(),
            &format!("{:.17e}", row.checksum),
            &row.median_ns.to_string(),
            &row.best_ns.to_string(),
            &row.worst_ns.to_string(),
            &format!("{:.6}", row.ns_per_call),
        ])?;
    }
    writer.flush().context("flush x87 microbench CSV")
}

fn write_markdown_summary(path: &PathBuf, rows: &[CsvRow]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut markdown = String::new();
    markdown.push_str("# x87 Givens Microbench\n\n");
    markdown.push_str(
        "Median timings for the actual composed x87 Givens/transcendental helpers used by the current backend.\n\n",
    );

    for case in rows
        .iter()
        .map(|row| row.case)
        .collect::<BTreeSet<&str>>()
        .into_iter()
    {
        markdown.push_str(&format!("## {}\n\n", case));
        markdown.push_str("| kernel | median_ns | ns_per_call |\n");
        markdown.push_str("| --- | ---: | ---: |\n");
        for row in rows.iter().filter(|row| row.case == case) {
            markdown.push_str(&format!(
                "| {} | {} | {:.3} |\n",
                row.kernel, row.median_ns, row.ns_per_call
            ));
        }
        markdown.push('\n');
    }

    markdown.push_str("## Notes\n\n");
    markdown.push_str("- `givens_sincos_f64` and `givens_sincos_ext80` measure the composed half-angle Givens path the current Jacobi backend actually uses.\n");
    markdown.push_str("- `x87_atan2_sincos` measures the older full-angle composition without the half-angle step.\n");
    markdown.push_str("- `x87_givens_diagonal_update` isolates the 2x2 polynomial update so the transcendental path can be compared against the update path directly.\n");

    fs::write(path, markdown).with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::{filter_cases, givens_cases, kernel_names, parse_filter, validate_requested};

    #[test]
    fn parse_filter_trims_and_skips_empty_entries() {
        let filter = parse_filter(Some(" quarter_pi, ,mixed_signs , quarter_pi "));
        assert_eq!(filter.len(), 2);
        assert!(filter.contains("quarter_pi"));
        assert!(filter.contains("mixed_signs"));
    }

    #[test]
    fn filter_cases_keeps_only_requested_cases() {
        let requested = parse_filter(Some("mixed_signs,tiny_angle"));
        let cases = filter_cases(&givens_cases(), &requested);
        let names: Vec<&str> = cases.iter().map(|case| case.name).collect();
        assert_eq!(names, vec!["mixed_signs", "tiny_angle"]);
    }

    #[test]
    fn validate_requested_rejects_unknown_kernel_names() {
        let requested = parse_filter(Some("x87_atan2_sincos,missing_kernel"));
        let err = validate_requested("kernel", &requested, kernel_names()).unwrap_err();
        assert!(
            err.to_string().contains("missing_kernel"),
            "expected unknown kernel in error, got: {err}"
        );
    }
}
