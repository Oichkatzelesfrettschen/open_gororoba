//! Benchmark partial-spectrum lanes against current full-spectrum Jacobi solves.

mod jacobi_benchmark_cases;

use std::{fs, path::PathBuf, time::Instant};

use algebra_analysis::{
    precision_policy::{
        JacobiDispatchInput, SpectrumDispatchDecision, SpectrumObjective, SpectrumSolverFamily,
        choose_jacobi_backend,
    },
    reference_jacobi,
    spectrum_solvers::{solve_spectrum, solve_with_decision},
};
use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use csv::Writer;
use jacobi_benchmark_cases::{MatrixFamily, build_case, select_families};

#[derive(Parser, Debug)]
#[command(name = "partial-spectrum-bench")]
#[command(about = "Benchmark partial-spectrum prototypes for k = 1,2,4")]
struct Args {
    #[arg(long, value_delimiter = ',', default_values_t = vec![16usize, 32, 64])]
    sizes: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_values_t = vec![1usize, 2, 4])]
    k_values: Vec<usize>,
    #[arg(long, default_value = "3")]
    repeats: usize,
    #[arg(long, value_delimiter = ',')]
    families: Vec<MatrixFamily>,
    #[arg(long, value_delimiter = ',')]
    objectives: Vec<ObjectiveKind>,
    #[arg(long, default_value = "reports/benchmarks/partial_spectrum_bench.csv")]
    output: PathBuf,
    #[arg(long)]
    summary: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "snake_case")]
enum ObjectiveKind {
    LargestAbs,
    SmallestAbs,
}

impl ObjectiveKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::LargestAbs => "largest_abs",
            Self::SmallestAbs => "smallest_abs",
        }
    }

    fn defaults() -> &'static [Self] {
        &[Self::LargestAbs, Self::SmallestAbs]
    }
}

#[derive(Debug)]
struct Row {
    family: &'static str,
    size: usize,
    objective: &'static str,
    k: usize,
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
    let objectives = select_objectives(&args.objectives);
    let mut sizes = args.sizes;
    sizes.retain(|&n| n >= 2);
    sizes.sort_unstable();
    sizes.dedup();
    let mut k_values = args.k_values;
    k_values.retain(|&k| k > 0);
    k_values.sort_unstable();
    k_values.dedup();
    ensure!(!sizes.is_empty(), "need at least one matrix size >= 2");
    ensure!(!k_values.is_empty(), "need at least one k > 0");

    let mut rows = Vec::new();
    for family in families {
        for size in &sizes {
            let case = build_case(family, *size)?;
            for objective_kind in &objectives {
                for &k in &k_values {
                    let objective = match objective_kind {
                        ObjectiveKind::LargestAbs => SpectrumObjective::LargestAbs { k },
                        ObjectiveKind::SmallestAbs => SpectrumObjective::SmallestAbs { k },
                    };
                    rows.push(run_row(
                        &case,
                        objective,
                        objective_kind.as_str(),
                        k,
                        SolverVariant::ReferenceFullTruncated,
                        args.repeats,
                    ));
                    rows.push(run_row(
                        &case,
                        objective,
                        objective_kind.as_str(),
                        k,
                        SolverVariant::PolicyFullTruncated,
                        args.repeats,
                    ));
                    rows.push(run_row(
                        &case,
                        objective,
                        objective_kind.as_str(),
                        k,
                        SolverVariant::PartialSubspace,
                        args.repeats,
                    ));
                }
            }
        }
    }

    write_csv(&args.output, &rows)?;
    if let Some(summary) = &args.summary {
        write_summary(summary, &rows)?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum SolverVariant {
    ReferenceFullTruncated,
    PolicyFullTruncated,
    PartialSubspace,
}

impl SolverVariant {
    fn as_str(self) -> &'static str {
        match self {
            Self::ReferenceFullTruncated => "reference_full_truncated",
            Self::PolicyFullTruncated => "policy_full_truncated",
            Self::PartialSubspace => "partial_subspace",
        }
    }
}

fn select_objectives(requested: &[ObjectiveKind]) -> Vec<ObjectiveKind> {
    let mut objectives = if requested.is_empty() {
        ObjectiveKind::defaults().to_vec()
    } else {
        requested.to_vec()
    };
    objectives.sort_by_key(|objective| objective.as_str());
    objectives.dedup();
    objectives
}

fn run_row(
    case: &jacobi_benchmark_cases::MatrixCase,
    objective: SpectrumObjective,
    objective_name: &'static str,
    k: usize,
    solver: SolverVariant,
    repeats: usize,
) -> Row {
    let expected = truncate_expected(&case.expected_spectrum, objective, k);
    let mut times = Vec::with_capacity(repeats);
    let mut last = Vec::new();
    for _ in 0..repeats {
        let started = Instant::now();
        match run_solver(case, objective, solver) {
            Ok(values) => {
                times.push(started.elapsed());
                last = values;
            }
            Err(error) => {
                return Row {
                    family: case.family_name,
                    size: case.matrix.len(),
                    objective: objective_name,
                    k,
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
    times.sort_unstable();
    let median_ns = times[times.len() / 2].as_nanos();
    let (max_abs_error, rms_abs_error) = spectrum_error(&last, &expected);
    Row {
        family: case.family_name,
        size: case.matrix.len(),
        objective: objective_name,
        k,
        solver: solver.as_str(),
        median_ns,
        max_abs_error,
        rms_abs_error,
        status: "ok",
        error_message: String::new(),
    }
}

fn run_solver(
    case: &jacobi_benchmark_cases::MatrixCase,
    objective: SpectrumObjective,
    solver: SolverVariant,
) -> Result<Vec<f64>> {
    match solver {
        SolverVariant::ReferenceFullTruncated => {
            let mut eigs = reference_jacobi::symmetric_eigenvalues_f64(&case.matrix)?;
            truncate_to_objective(&mut eigs, objective);
            Ok(eigs)
        }
        SolverVariant::PolicyFullTruncated => {
            let backend =
                choose_jacobi_backend(JacobiDispatchInput::obstruction_spectrum(case.matrix.len()))
                    .backend;
            let decision = SpectrumDispatchDecision {
                solver_family: SpectrumSolverFamily::FullJacobi,
                backend,
                reason: "policy full-spectrum baseline",
            };
            let mut eigs = solve_with_decision(
                &case.matrix,
                SpectrumObjective::FullSpectrum,
                decision,
                1.0e-12,
            )?;
            truncate_to_objective(&mut eigs, objective);
            Ok(eigs)
        }
        SolverVariant::PartialSubspace => {
            let input =
                algebra_analysis::precision_policy::SpectrumDispatchInput::obstruction_extremal(
                    case.matrix.len(),
                    objective,
                    case.structure_hints,
                );
            Ok(solve_spectrum(&case.matrix, input, 1.0e-10)?)
        }
    }
}

fn truncate_expected(expected: &[f64], objective: SpectrumObjective, k: usize) -> Vec<f64> {
    let mut values = expected.to_vec();
    truncate_to_objective(&mut values, objective);
    values.truncate(k.min(values.len()));
    values
}

fn truncate_to_objective(values: &mut Vec<f64>, objective: SpectrumObjective) {
    values.sort_by(|lhs, rhs| match objective {
        SpectrumObjective::LargestAbs { .. } => rhs
            .abs()
            .total_cmp(&lhs.abs())
            .then_with(|| rhs.total_cmp(lhs)),
        SpectrumObjective::SmallestAbs { .. } => lhs
            .abs()
            .total_cmp(&rhs.abs())
            .then_with(|| lhs.total_cmp(rhs)),
        SpectrumObjective::FullSpectrum => rhs
            .abs()
            .total_cmp(&lhs.abs())
            .then_with(|| rhs.total_cmp(lhs)),
    });
    let keep = objective.requested_count().min(values.len());
    values.truncate(keep);
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
        "objective",
        "k",
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
            row.objective.to_string(),
            row.k.to_string(),
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
    out.push_str("# Partial Spectrum Bench\n\n");
    out.push_str("| family | size | objective | k | fastest solver | lowest max abs error |\n");
    out.push_str("| --- | ---: | --- | ---: | --- | --- |\n");
    let mut keys: Vec<(&str, usize, &str, usize)> = rows
        .iter()
        .map(|row| (row.family, row.size, row.objective, row.k))
        .collect();
    keys.sort_unstable();
    keys.dedup();
    for (family, size, objective, k) in keys {
        let group: Vec<&Row> = rows
            .iter()
            .filter(|row| {
                row.family == family
                    && row.size == size
                    && row.objective == objective
                    && row.k == k
                    && row.status == "ok"
            })
            .collect();
        let fastest = group
            .iter()
            .min_by_key(|row| row.median_ns)
            .map(|row| row.solver)
            .unwrap_or("none");
        let accurate = group
            .iter()
            .min_by(|lhs, rhs| lhs.max_abs_error.total_cmp(&rhs.max_abs_error))
            .map(|row| row.solver)
            .unwrap_or("none");
        out.push_str(&format!(
            "| {family} | {size} | {objective} | {k} | {fastest} | {accurate} |\n"
        ));
    }
    fs::write(path, out).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}
