//! Sweep block-Jacobi prototypes against current full-spectrum backends.

mod jacobi_benchmark_cases;

use std::{fs, path::PathBuf, time::Instant};

use algebra_analysis::{
    block_jacobi::symmetric_eigenvalues_block_jacobi, dd_jacobi, reference_jacobi,
};
use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use csv::Writer;
use jacobi_benchmark_cases::{MatrixFamily, build_case, select_families};

#[derive(Parser, Debug)]
#[command(name = "block-jacobi-backend-sweep")]
#[command(about = "Benchmark block-Jacobi prototypes with block sizes 2 and 4")]
struct Args {
    #[arg(long, value_delimiter = ',', default_values_t = vec![8usize, 16, 24, 32])]
    sizes: Vec<usize>,
    #[arg(long, default_value = "3")]
    repeats: usize,
    #[arg(long, value_delimiter = ',')]
    families: Vec<MatrixFamily>,
    #[arg(long, value_delimiter = ',')]
    solvers: Vec<SolverKind>,
    #[arg(
        long,
        default_value = "reports/benchmarks/block_jacobi_backend_sweep.csv"
    )]
    output: PathBuf,
    #[arg(long)]
    summary: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "snake_case")]
enum SolverKind {
    ReferenceF64,
    DoubleDouble,
    X87,
    BlockJacobiB2,
    BlockJacobiB4,
}

impl SolverKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::ReferenceF64 => "reference_f64",
            Self::DoubleDouble => "double_double",
            Self::X87 => "x87",
            Self::BlockJacobiB2 => "block_jacobi_b2",
            Self::BlockJacobiB4 => "block_jacobi_b4",
        }
    }

    fn defaults() -> &'static [Self] {
        &[
            Self::ReferenceF64,
            Self::DoubleDouble,
            Self::X87,
            Self::BlockJacobiB2,
            Self::BlockJacobiB4,
        ]
    }
}

#[derive(Debug)]
struct Row {
    family: &'static str,
    size: usize,
    solver: &'static str,
    median_ns: u128,
    max_abs_error: f64,
    rms_abs_error: f64,
    status: &'static str,
    error_message: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.repeats > 0, "--repeats must be greater than zero");
    let families = select_families(&args.families);
    let solvers = select_solvers(&args.solvers);
    let mut sizes = args.sizes;
    sizes.retain(|&n| n >= 2);
    sizes.sort_unstable();
    sizes.dedup();
    ensure!(!sizes.is_empty(), "need at least one matrix size >= 2");

    let mut rows = Vec::new();
    for family in families {
        for size in &sizes {
            let case = build_case(family, *size)?;
            for solver in &solvers {
                rows.push(run_row(&case, *solver, args.repeats));
            }
        }
    }

    write_csv(&args.output, &rows)?;
    if let Some(summary) = &args.summary {
        write_summary(summary, &rows)?;
    }
    Ok(())
}

fn select_solvers(requested: &[SolverKind]) -> Vec<SolverKind> {
    let mut solvers = if requested.is_empty() {
        SolverKind::defaults().to_vec()
    } else {
        requested.to_vec()
    };
    solvers.sort_by_key(|solver| solver.as_str());
    solvers.dedup();
    solvers
}

fn run_row(case: &jacobi_benchmark_cases::MatrixCase, solver: SolverKind, repeats: usize) -> Row {
    if solver == SolverKind::X87 && !cfg!(target_arch = "x86_64") {
        return Row {
            family: case.family_name,
            size: case.matrix.len(),
            solver: solver.as_str(),
            median_ns: 0,
            max_abs_error: f64::NAN,
            rms_abs_error: f64::NAN,
            status: "unavailable",
            error_message: "x87 backend unavailable on non-x86_64 target".to_string(),
        };
    }

    let mut durations = Vec::with_capacity(repeats);
    let mut last = Vec::new();
    for _ in 0..repeats {
        let started = Instant::now();
        match run_solver(&case.matrix, solver) {
            Ok(eigs) => {
                durations.push(started.elapsed());
                last = eigs;
            }
            Err(error) => {
                return Row {
                    family: case.family_name,
                    size: case.matrix.len(),
                    solver: solver.as_str(),
                    median_ns: 0,
                    max_abs_error: f64::NAN,
                    rms_abs_error: f64::NAN,
                    status: "failed",
                    error_message: error.to_string(),
                };
            }
        }
    }

    durations.sort_unstable();
    let median = durations[durations.len() / 2].as_nanos();
    let (max_abs_error, rms_abs_error) = spectrum_error(&last, &case.expected_spectrum);
    Row {
        family: case.family_name,
        size: case.matrix.len(),
        solver: solver.as_str(),
        median_ns: median,
        max_abs_error,
        rms_abs_error,
        status: "ok",
        error_message: String::new(),
    }
}

fn run_solver(matrix: &[Vec<f64>], solver: SolverKind) -> Result<Vec<f64>> {
    let eigs = match solver {
        SolverKind::ReferenceF64 => reference_jacobi::symmetric_eigenvalues_f64(matrix)?,
        SolverKind::DoubleDouble => dd_jacobi::symmetric_eigenvalues_dd(matrix)?,
        SolverKind::X87 => {
            #[cfg(target_arch = "x86_64")]
            {
                let flat = matrix.iter().flatten().copied().collect::<Vec<f64>>();
                algebra_analysis::x87_jacobi::symmetric_eigenvalues_x87(
                    &flat,
                    matrix.len(),
                    100 * matrix.len() * matrix.len(),
                    1.0e-15,
                )?
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                anyhow::bail!("x87 backend unavailable on non-x86_64 target");
            }
        }
        SolverKind::BlockJacobiB2 => symmetric_eigenvalues_block_jacobi(matrix, 2, 16, 1.0e-10)?,
        SolverKind::BlockJacobiB4 => symmetric_eigenvalues_block_jacobi(matrix, 4, 16, 1.0e-10)?,
    };
    Ok(eigs)
}

fn spectrum_error(actual: &[f64], expected: &[f64]) -> (f64, f64) {
    let mut actual_sorted = actual.to_vec();
    let mut expected_sorted = expected.to_vec();
    sort_by_abs_desc(&mut actual_sorted);
    sort_by_abs_desc(&mut expected_sorted);

    let mut max_abs_error = 0.0_f64;
    let mut sq_sum = 0.0_f64;
    let count = actual_sorted.len().min(expected_sorted.len()).max(1);
    for (&lhs, &rhs) in actual_sorted.iter().zip(expected_sorted.iter()) {
        let err = (lhs - rhs).abs();
        max_abs_error = max_abs_error.max(err);
        sq_sum += err * err;
    }
    (max_abs_error, (sq_sum / count as f64).sqrt())
}

fn sort_by_abs_desc(values: &mut [f64]) {
    values.sort_by(|lhs, rhs| {
        rhs.abs()
            .total_cmp(&lhs.abs())
            .then_with(|| rhs.total_cmp(lhs))
    });
}

fn write_csv(path: &PathBuf, rows: &[Row]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut writer =
        Writer::from_path(path).with_context(|| format!("opening {}", path.display()))?;
    writer.write_record([
        "family",
        "size",
        "solver",
        "median_ns",
        "max_abs_error",
        "rms_abs_error",
        "status",
        "error_message",
    ])?;
    for row in rows {
        writer.write_record([
            row.family.to_string(),
            row.size.to_string(),
            row.solver.to_string(),
            row.median_ns.to_string(),
            row.max_abs_error.to_string(),
            row.rms_abs_error.to_string(),
            row.status.to_string(),
            row.error_message.clone(),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_summary(path: &PathBuf, rows: &[Row]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut out = String::new();
    out.push_str("# Block Jacobi Backend Sweep\n\n");
    out.push_str("| family | size | fastest solver | lowest max abs error |\n");
    out.push_str("| --- | ---: | --- | --- |\n");

    let mut keys: Vec<(&str, usize)> = rows.iter().map(|row| (row.family, row.size)).collect();
    keys.sort_unstable();
    keys.dedup();
    for (family, size) in keys {
        let family_rows: Vec<&Row> = rows
            .iter()
            .filter(|row| row.family == family && row.size == size && row.status == "ok")
            .collect();
        let fastest = family_rows
            .iter()
            .min_by_key(|row| row.median_ns)
            .map(|row| row.solver)
            .unwrap_or("none");
        let most_accurate = family_rows
            .iter()
            .min_by(|lhs, rhs| lhs.max_abs_error.total_cmp(&rhs.max_abs_error))
            .map(|row| row.solver)
            .unwrap_or("none");
        out.push_str(&format!(
            "| {family} | {size} | {fastest} | {most_accurate} |\n"
        ));
    }

    fs::write(path, out).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}
