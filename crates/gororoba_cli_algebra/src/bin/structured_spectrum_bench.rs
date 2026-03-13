//! Benchmark exact structured deflation on obstruction-like matrices.

mod jacobi_benchmark_cases;

use std::{fs, path::PathBuf, time::Instant};

use algebra_analysis::{
    reference_jacobi,
    spectrum_solvers::{
        deflate_isolated_zero_modes, exploratory_histogram_partition,
        histogram_projected_reduction, validated_quotient_reduction,
    },
};
use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use csv::Writer;
use jacobi_benchmark_cases::{MatrixFamily, build_case, select_families};

#[derive(Parser, Debug)]
#[command(name = "structured-spectrum-bench")]
#[command(about = "Benchmark exact structured reduction on obstruction-like matrices")]
struct Args {
    #[arg(long, value_delimiter = ',', default_values_t = vec![16usize, 32, 64])]
    sizes: Vec<usize>,
    #[arg(long, default_value = "3")]
    repeats: usize,
    #[arg(
        long,
        value_delimiter = ',',
        default_values_t = vec![
            MatrixFamily::QuantizedObstructionGraph,
            MatrixFamily::QuantizedShellPermutation,
            MatrixFamily::RealObstruction
        ]
    )]
    families: Vec<MatrixFamily>,
    #[arg(long, value_delimiter = ',')]
    solvers: Vec<SolverKind>,
    #[arg(
        long,
        default_value = "reports/benchmarks/structured_spectrum_bench.csv"
    )]
    output: PathBuf,
    #[arg(long)]
    summary: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "snake_case")]
enum SolverKind {
    ReferenceFull,
    StructuredDeflatedReference,
    HistogramProjectedReference,
}

impl SolverKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::ReferenceFull => "reference_full",
            Self::StructuredDeflatedReference => "structured_deflated_reference",
            Self::HistogramProjectedReference => "histogram_projected_reference",
        }
    }

    fn defaults() -> &'static [Self] {
        &[
            Self::ReferenceFull,
            Self::StructuredDeflatedReference,
            Self::HistogramProjectedReference,
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
    reduced_order: usize,
    deflated_zero_modes: usize,
    validated_quotient_cells: usize,
    exploratory_histogram_cells: usize,
    projected_order: usize,
}

#[derive(Debug, Clone, Copy)]
struct ReductionMetrics {
    reduced_order: usize,
    deflated_zero_modes: usize,
    validated_quotient_cells: usize,
    exploratory_histogram_cells: usize,
    projected_order: usize,
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
        for &size in &sizes {
            let case = match build_case(family, size) {
                Ok(case) => case,
                Err(_) if family == MatrixFamily::RealObstruction && !size.is_power_of_two() => {
                    continue;
                }
                Err(error) => return Err(error),
            };
            let reduction = deflate_isolated_zero_modes(&case.matrix, 1.0e-12)?;
            let metrics = ReductionMetrics {
                reduced_order: reduction.reduced_matrix.len(),
                deflated_zero_modes: reduction.deflated_zero_modes,
                validated_quotient_cells: validated_quotient_reduction(&case.matrix, 1.0e-12)?
                    .map(|candidate| candidate.partition.len())
                    .unwrap_or(0),
                exploratory_histogram_cells: exploratory_histogram_partition(
                    &case.matrix,
                    1.0e-12,
                )?
                .map(|candidate| candidate.partition.len())
                .unwrap_or(0),
                projected_order: histogram_projected_reduction(&reduction.reduced_matrix, 1.0e-12)?
                    .map(|candidate| candidate.partition.len())
                    .unwrap_or(0),
            };
            for solver in &solvers {
                rows.push(run_row(&case, metrics, *solver, args.repeats)?);
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

fn run_row(
    case: &jacobi_benchmark_cases::MatrixCase,
    metrics: ReductionMetrics,
    solver: SolverKind,
    repeats: usize,
) -> Result<Row> {
    let mut times = Vec::with_capacity(repeats);
    let mut last = Vec::new();
    for _ in 0..repeats {
        let started = Instant::now();
        let eigs = run_solver(&case.matrix, solver)?;
        times.push(started.elapsed());
        last = eigs;
    }
    times.sort_unstable();
    let (max_abs_error, rms_abs_error) = spectrum_error(&last, &case.expected_spectrum);
    Ok(Row {
        family: case.family_name,
        size: case.matrix.len(),
        solver: solver.as_str(),
        median_ns: times[times.len() / 2].as_nanos(),
        max_abs_error,
        rms_abs_error,
        reduced_order: metrics.reduced_order,
        deflated_zero_modes: metrics.deflated_zero_modes,
        validated_quotient_cells: metrics.validated_quotient_cells,
        exploratory_histogram_cells: metrics.exploratory_histogram_cells,
        projected_order: metrics.projected_order,
    })
}

fn run_solver(matrix: &[Vec<f64>], solver: SolverKind) -> Result<Vec<f64>> {
    match solver {
        SolverKind::ReferenceFull => Ok(reference_jacobi::symmetric_eigenvalues_f64(matrix)?),
        SolverKind::StructuredDeflatedReference => {
            let reduction = deflate_isolated_zero_modes(matrix, 1.0e-12)?;
            let mut eigs = if reduction.reduced_matrix.is_empty() {
                Vec::new()
            } else {
                reference_jacobi::symmetric_eigenvalues_f64(&reduction.reduced_matrix)?
            };
            eigs.extend(std::iter::repeat_n(0.0, reduction.deflated_zero_modes));
            eigs.sort_by(|lhs, rhs| {
                rhs.abs()
                    .total_cmp(&lhs.abs())
                    .then_with(|| rhs.total_cmp(lhs))
            });
            Ok(eigs)
        }
        SolverKind::HistogramProjectedReference => {
            let reduction = deflate_isolated_zero_modes(matrix, 1.0e-12)?;
            let reduced_order = reduction.reduced_matrix.len();
            let Some(projected) =
                histogram_projected_reduction(&reduction.reduced_matrix, 1.0e-12)?
            else {
                let mut eigs = if reduction.reduced_matrix.is_empty() {
                    Vec::new()
                } else {
                    reference_jacobi::symmetric_eigenvalues_f64(&reduction.reduced_matrix)?
                };
                eigs.extend(std::iter::repeat_n(0.0, reduction.deflated_zero_modes));
                eigs.sort_by(|lhs, rhs| {
                    rhs.abs()
                        .total_cmp(&lhs.abs())
                        .then_with(|| rhs.total_cmp(lhs))
                });
                return Ok(eigs);
            };
            let mut eigs = if projected.projected_matrix.is_empty() {
                Vec::new()
            } else {
                reference_jacobi::symmetric_eigenvalues_f64(&projected.projected_matrix)?
            };
            eigs.extend(std::iter::repeat_n(
                0.0,
                reduced_order.saturating_sub(projected.partition.len())
                    + reduction.deflated_zero_modes,
            ));
            eigs.sort_by(|lhs, rhs| {
                rhs.abs()
                    .total_cmp(&lhs.abs())
                    .then_with(|| rhs.total_cmp(lhs))
            });
            Ok(eigs)
        }
    }
}

fn spectrum_error(actual: &[f64], expected: &[f64]) -> (f64, f64) {
    let mut max_abs_error = 0.0_f64;
    let mut sq_sum = 0.0_f64;
    let count = actual.len().min(expected.len()).max(1);
    for (&lhs, &rhs) in actual.iter().zip(expected.iter()) {
        let err = (lhs - rhs).abs();
        max_abs_error = max_abs_error.max(err);
        sq_sum += err * err;
    }
    (max_abs_error, (sq_sum / count as f64).sqrt())
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
        "reduced_order",
        "deflated_zero_modes",
        "validated_quotient_cells",
        "exploratory_histogram_cells",
        "projected_order",
    ])?;
    for row in rows {
        writer.write_record([
            row.family.to_string(),
            row.size.to_string(),
            row.solver.to_string(),
            row.median_ns.to_string(),
            row.max_abs_error.to_string(),
            row.rms_abs_error.to_string(),
            row.reduced_order.to_string(),
            row.deflated_zero_modes.to_string(),
            row.validated_quotient_cells.to_string(),
            row.exploratory_histogram_cells.to_string(),
            row.projected_order.to_string(),
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
    out.push_str("# Structured Spectrum Bench\n\n");
    out.push_str("| family | size | fastest solver | lowest max abs error | deflated zero modes | validated quotient cells | exploratory histogram cells | projected order |\n");
    out.push_str("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |\n");
    let mut keys: Vec<(&str, usize)> = rows.iter().map(|row| (row.family, row.size)).collect();
    keys.sort_unstable();
    keys.dedup();
    for (family, size) in keys {
        let group: Vec<&Row> = rows
            .iter()
            .filter(|row| row.family == family && row.size == size)
            .collect();
        let fastest = group
            .iter()
            .min_by_key(|row| row.median_ns)
            .map(|row| row.solver)
            .unwrap_or("none");
        let lowest_error = group
            .iter()
            .min_by(|lhs, rhs| lhs.max_abs_error.total_cmp(&rhs.max_abs_error))
            .map(|row| row.solver)
            .unwrap_or("none");
        let deflated = group
            .first()
            .map(|row| row.deflated_zero_modes)
            .unwrap_or(0);
        let validated = group
            .first()
            .map(|row| row.validated_quotient_cells)
            .unwrap_or(0);
        let exploratory = group
            .first()
            .map(|row| row.exploratory_histogram_cells)
            .unwrap_or(0);
        let projected_order = group.first().map(|row| row.projected_order).unwrap_or(0);
        out.push_str(&format!(
            "| {family} | {size} | {fastest} | {lowest_error} | {deflated} | {validated} | {exploratory} | {projected_order} |\n"
        ));
    }
    fs::write(path, out).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}
