//! Sweep Jacobi solver backends across deterministic matrix families and sizes.

mod jacobi_benchmark_cases;

use std::{
    cmp::Ordering,
    fs,
    hint::black_box,
    path::PathBuf,
    time::{Duration, Instant},
};

use algebra_analysis::{
    precision_policy::{
        JacobiBackend, JacobiDispatchInput, MatrixWorkloadClass, choose_jacobi_backend,
    },
    reference_jacobi,
};
use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use csv::Writer;
use jacobi_benchmark_cases::{MatrixFamily, build_case, select_families};

#[derive(Parser, Debug)]
#[command(name = "jacobi-backend-sweep")]
#[command(
    about = "Benchmark x87, double-double, and reference Jacobi backends over deterministic matrix families"
)]
struct Args {
    /// Matrix sizes to benchmark.
    #[arg(
        long,
        value_delimiter = ',',
        default_values_t = vec![4usize, 8, 16, 24, 32]
    )]
    sizes: Vec<usize>,

    /// Repetitions per backend and matrix size.
    #[arg(long, default_value = "5")]
    repeats: usize,

    /// Optional family subset.
    #[arg(long, value_delimiter = ',')]
    families: Vec<MatrixFamily>,

    /// Optional backend subset.
    #[arg(long, value_delimiter = ',')]
    backends: Vec<BackendKind>,

    /// Output CSV path.
    #[arg(long, default_value = "reports/benchmarks/jacobi_backend_sweep.csv")]
    output: PathBuf,

    /// Optional Markdown summary path.
    #[arg(long)]
    summary: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy)]
enum BackendStatus {
    Ok,
    Failed,
    Unavailable,
}

impl BackendStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Failed => "failed",
            Self::Unavailable => "unavailable",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "snake_case")]
enum BackendKind {
    ReferenceF64,
    DoubleDouble,
    X87,
}

impl BackendKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::ReferenceF64 => "reference_f64",
            Self::DoubleDouble => "double_double",
            Self::X87 => "x87",
        }
    }

    fn defaults() -> &'static [Self] {
        &[Self::ReferenceF64, Self::DoubleDouble, Self::X87]
    }
}

#[derive(Debug)]
struct Measurement {
    median: Duration,
    best: Duration,
    worst: Duration,
}

#[derive(Debug)]
struct Row {
    family: &'static str,
    size: usize,
    workload_class: &'static str,
    symmetric: bool,
    nonnegative: bool,
    zero_diagonal: bool,
    isolated_zero_modes: bool,
    quantized_value_ladder: bool,
    equitable_partition_candidate: bool,
    backend: &'static str,
    status: &'static str,
    selected_by_default_policy: bool,
    policy_backend: &'static str,
    median_ns: u128,
    best_ns: u128,
    worst_ns: u128,
    max_abs_error: f64,
    rms_abs_error: f64,
    error_message: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.repeats > 0, "--repeats must be greater than zero");
    ensure!(!args.sizes.is_empty(), "--sizes must not be empty");

    let mut sizes = args.sizes;
    sizes.retain(|&n| n >= 2);
    sizes.sort_unstable();
    sizes.dedup();
    ensure!(!sizes.is_empty(), "need at least one matrix size >= 2");

    let families = select_families(&args.families);
    let backends = select_backends(&args.backends);
    let mut rows = Vec::new();

    for family in families {
        for &size in &sizes {
            let case = build_case(family, size)?;
            let decision = choose_jacobi_backend(JacobiDispatchInput::obstruction_spectrum(size));
            let policy_backend = map_policy_backend(decision.backend);

            for backend in &backends {
                rows.push(run_backend_row(
                    &case,
                    size,
                    *backend,
                    policy_backend,
                    args.repeats,
                ));
            }
        }
    }

    print_summary(&rows);
    write_csv(&args.output, &rows)?;
    if let Some(summary) = &args.summary {
        write_markdown_summary(summary, &rows)?;
        println!("Wrote Jacobi backend summary to {}", summary.display());
    }
    println!("Wrote Jacobi backend sweep to {}", args.output.display());
    Ok(())
}

fn select_backends(requested: &[BackendKind]) -> Vec<BackendKind> {
    let mut backends = if requested.is_empty() {
        BackendKind::defaults().to_vec()
    } else {
        requested.to_vec()
    };
    backends.sort_by_key(|backend| backend.as_str());
    backends.dedup();
    backends
}

fn run_backend_row(
    case: &jacobi_benchmark_cases::MatrixCase,
    size: usize,
    backend: BackendKind,
    policy_backend: BackendKind,
    repeats: usize,
) -> Row {
    let selected_by_default_policy = backend == policy_backend;

    if backend == BackendKind::X87 && !cfg!(target_arch = "x86_64") {
        return Row {
            family: case.family_name,
            size,
            workload_class: workload_class_str(case.workload_class),
            symmetric: case.structure_hints.symmetric,
            nonnegative: case.structure_hints.nonnegative,
            zero_diagonal: case.structure_hints.zero_diagonal,
            isolated_zero_modes: case.structure_hints.isolated_zero_modes,
            quantized_value_ladder: case.structure_hints.quantized_value_ladder,
            equitable_partition_candidate: case.structure_hints.equitable_partition_candidate,
            backend: backend.as_str(),
            status: BackendStatus::Unavailable.as_str(),
            selected_by_default_policy,
            policy_backend: policy_backend.as_str(),
            median_ns: 0,
            best_ns: 0,
            worst_ns: 0,
            max_abs_error: f64::NAN,
            rms_abs_error: f64::NAN,
            error_message: "x87 backend unavailable on non-x86_64 target".to_string(),
        };
    }

    match measure_backend(&case.matrix, &case.expected_spectrum, backend, repeats) {
        Ok((measurement, max_abs_error, rms_abs_error)) => Row {
            family: case.family_name,
            size,
            workload_class: workload_class_str(case.workload_class),
            symmetric: case.structure_hints.symmetric,
            nonnegative: case.structure_hints.nonnegative,
            zero_diagonal: case.structure_hints.zero_diagonal,
            isolated_zero_modes: case.structure_hints.isolated_zero_modes,
            quantized_value_ladder: case.structure_hints.quantized_value_ladder,
            equitable_partition_candidate: case.structure_hints.equitable_partition_candidate,
            backend: backend.as_str(),
            status: BackendStatus::Ok.as_str(),
            selected_by_default_policy,
            policy_backend: policy_backend.as_str(),
            median_ns: measurement.median.as_nanos(),
            best_ns: measurement.best.as_nanos(),
            worst_ns: measurement.worst.as_nanos(),
            max_abs_error,
            rms_abs_error,
            error_message: String::new(),
        },
        Err(error) => Row {
            family: case.family_name,
            size,
            workload_class: workload_class_str(case.workload_class),
            symmetric: case.structure_hints.symmetric,
            nonnegative: case.structure_hints.nonnegative,
            zero_diagonal: case.structure_hints.zero_diagonal,
            isolated_zero_modes: case.structure_hints.isolated_zero_modes,
            quantized_value_ladder: case.structure_hints.quantized_value_ladder,
            equitable_partition_candidate: case.structure_hints.equitable_partition_candidate,
            backend: backend.as_str(),
            status: BackendStatus::Failed.as_str(),
            selected_by_default_policy,
            policy_backend: policy_backend.as_str(),
            median_ns: 0,
            best_ns: 0,
            worst_ns: 0,
            max_abs_error: f64::NAN,
            rms_abs_error: f64::NAN,
            error_message: error.to_string(),
        },
    }
}

fn measure_backend(
    matrix: &[Vec<f64>],
    expected: &[f64],
    backend: BackendKind,
    repeats: usize,
) -> Result<(Measurement, f64, f64)> {
    let mut durations = Vec::with_capacity(repeats);
    let mut last_eigs = Vec::new();

    for _ in 0..repeats {
        let started = Instant::now();
        let eigs = run_backend(matrix, backend)?;
        durations.push(started.elapsed());
        last_eigs = eigs;
    }

    durations.sort_unstable();
    let measurement = Measurement {
        median: durations[durations.len() / 2],
        best: *durations.first().unwrap_or(&Duration::ZERO),
        worst: *durations.last().unwrap_or(&Duration::ZERO),
    };
    let (max_abs_error, rms_abs_error) = spectrum_error(&last_eigs, expected);
    Ok((measurement, max_abs_error, rms_abs_error))
}

fn run_backend(matrix: &[Vec<f64>], backend: BackendKind) -> Result<Vec<f64>> {
    let eigs = match backend {
        BackendKind::ReferenceF64 => {
            reference_jacobi::symmetric_eigenvalues_f64(black_box(matrix))?
        }
        BackendKind::DoubleDouble => {
            algebra_analysis::dd_jacobi::symmetric_eigenvalues_dd(black_box(matrix))?
        }
        BackendKind::X87 => {
            #[cfg(target_arch = "x86_64")]
            {
                let n = matrix.len();
                let flat = flatten_square_matrix(matrix);
                algebra_analysis::x87_jacobi::symmetric_eigenvalues_x87(
                    black_box(&flat),
                    n,
                    recommended_jacobi_iterations(n),
                    1.0e-15,
                )?
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                anyhow::bail!("x87 backend unavailable on non-x86_64 target");
            }
        }
    };
    Ok(eigs)
}

fn flatten_square_matrix(matrix: &[Vec<f64>]) -> Vec<f64> {
    let n = matrix.len();
    let mut flat = Vec::with_capacity(n * n);
    for row in matrix {
        flat.extend_from_slice(row);
    }
    flat
}

fn recommended_jacobi_iterations(n: usize) -> usize {
    100usize.saturating_mul(n).saturating_mul(n)
}

fn spectrum_error(actual: &[f64], expected: &[f64]) -> (f64, f64) {
    let mut actual_sorted = actual.to_vec();
    let mut expected_sorted = expected.to_vec();
    sort_by_abs_desc(&mut actual_sorted);
    sort_by_abs_desc(&mut expected_sorted);

    let mut max_abs_error = 0.0_f64;
    let mut sq_sum = 0.0;
    let count = actual_sorted.len().min(expected_sorted.len()).max(1);
    for (&lhs, &rhs) in actual_sorted.iter().zip(expected_sorted.iter()) {
        let err = (lhs - rhs).abs();
        max_abs_error = max_abs_error.max(err);
        sq_sum += err * err;
    }
    (max_abs_error, (sq_sum / count as f64).sqrt())
}

fn sort_by_abs_desc(values: &mut [f64]) {
    values.sort_by(|lhs, rhs| cmp_by_abs_desc(*lhs, *rhs));
}

fn cmp_by_abs_desc(lhs: f64, rhs: f64) -> Ordering {
    rhs.abs()
        .total_cmp(&lhs.abs())
        .then_with(|| rhs.total_cmp(&lhs))
}

fn cmp_error(lhs: f64, rhs: f64) -> Ordering {
    match (lhs.is_nan(), rhs.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => lhs.total_cmp(&rhs),
    }
}

fn map_policy_backend(backend: JacobiBackend) -> BackendKind {
    match backend {
        JacobiBackend::X87 => BackendKind::X87,
        JacobiBackend::DoubleDouble => BackendKind::DoubleDouble,
        JacobiBackend::ReferenceF64 => BackendKind::ReferenceF64,
    }
}

fn workload_class_str(class: MatrixWorkloadClass) -> &'static str {
    match class {
        MatrixWorkloadClass::FullSpectrumDense => "full_spectrum_dense",
        MatrixWorkloadClass::FewExtremal => "few_extremal",
        MatrixWorkloadClass::ObstructionStructured => "obstruction_structured",
    }
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
        "workload_class",
        "symmetric",
        "nonnegative",
        "zero_diagonal",
        "isolated_zero_modes",
        "quantized_value_ladder",
        "equitable_partition_candidate",
        "backend",
        "status",
        "selected_by_default_policy",
        "policy_backend",
        "median_ns",
        "best_ns",
        "worst_ns",
        "max_abs_error",
        "rms_abs_error",
        "error_message",
    ])?;

    for row in rows {
        writer.write_record([
            row.family.to_string(),
            row.size.to_string(),
            row.workload_class.to_string(),
            row.symmetric.to_string(),
            row.nonnegative.to_string(),
            row.zero_diagonal.to_string(),
            row.isolated_zero_modes.to_string(),
            row.quantized_value_ladder.to_string(),
            row.equitable_partition_candidate.to_string(),
            row.backend.to_string(),
            row.status.to_string(),
            row.selected_by_default_policy.to_string(),
            row.policy_backend.to_string(),
            row.median_ns.to_string(),
            row.best_ns.to_string(),
            row.worst_ns.to_string(),
            row.max_abs_error.to_string(),
            row.rms_abs_error.to_string(),
            row.error_message.clone(),
        ])?;
    }

    writer.flush()?;
    Ok(())
}

fn write_markdown_summary(path: &PathBuf, rows: &[Row]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }

    let mut out = String::new();
    out.push_str("# Jacobi Backend Sweep\n\n");
    out.push_str("| family | size | workload class | default policy | fastest successful backend | lowest max abs error |\n");
    out.push_str("| --- | ---: | --- | --- | --- | --- |\n");

    let mut keys: Vec<(&str, usize)> = rows.iter().map(|row| (row.family, row.size)).collect();
    keys.sort_unstable();
    keys.dedup();

    for (family, size) in keys {
        let family_rows: Vec<&Row> = rows
            .iter()
            .filter(|row| row.family == family && row.size == size)
            .collect();
        let workload_class = family_rows
            .first()
            .map(|row| row.workload_class)
            .unwrap_or("unknown");
        let default_policy = family_rows
            .iter()
            .find(|row| row.selected_by_default_policy)
            .map(|row| row.backend)
            .unwrap_or("unknown");
        let fastest = family_rows
            .iter()
            .filter(|row| row.status == BackendStatus::Ok.as_str())
            .min_by_key(|row| row.median_ns)
            .map(|row| row.backend)
            .unwrap_or("none");
        let most_accurate = family_rows
            .iter()
            .filter(|row| row.status == BackendStatus::Ok.as_str())
            .min_by(|lhs, rhs| cmp_error(lhs.max_abs_error, rhs.max_abs_error))
            .map(|row| row.backend)
            .unwrap_or("none");
        out.push_str(&format!(
            "| {family} | {size} | {workload_class} | {default_policy} | {fastest} | {most_accurate} |\n"
        ));
    }

    out.push_str("\n## Rows\n\n");
    out.push_str("| family | size | workload | backend | status | selected | symmetric | nonnegative | zero_diag | zero_modes | quantized | equitable | median ns | max abs error | rms abs error |\n");
    out.push_str("| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |\n");
    for row in rows {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.6e} | {:.6e} |\n",
            row.family,
            row.size,
            row.workload_class,
            row.backend,
            row.status,
            row.selected_by_default_policy,
            row.symmetric,
            row.nonnegative,
            row.zero_diagonal,
            row.isolated_zero_modes,
            row.quantized_value_ladder,
            row.equitable_partition_candidate,
            row.median_ns,
            row.max_abs_error,
            row.rms_abs_error,
        ));
    }

    fs::write(path, out).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn print_summary(rows: &[Row]) {
    println!(
        "{:<20} {:>4} {:<24} {:<16} {:<10} {:>12} {:>14}",
        "family", "size", "workload", "backend", "status", "median_ms", "max_abs_err"
    );
    for row in rows {
        let median_ms = row.median_ns as f64 / 1_000_000.0;
        println!(
            "{:<20} {:>4} {:<24} {:<16} {:<10} {:>12.3} {:>14.6e}",
            row.family,
            row.size,
            row.workload_class,
            row.backend,
            row.status,
            median_ms,
            row.max_abs_error
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BackendKind, cmp_by_abs_desc, cmp_error, select_backends, sort_by_abs_desc,
        workload_class_str,
    };
    use crate::jacobi_benchmark_cases::{MatrixFamily, build_case, select_families};
    use algebra_analysis::precision_policy::MatrixWorkloadClass;
    use std::cmp::Ordering;

    #[test]
    fn abs_sort_handles_equal_magnitudes_and_signs() {
        let mut values = vec![1.0, -1.0, 0.0, 2.0, -2.0];
        sort_by_abs_desc(&mut values);

        assert_eq!(values, vec![2.0, -2.0, 1.0, -1.0, 0.0]);
    }

    #[test]
    fn abs_sort_handles_nan_without_panicking() {
        let mut values = vec![f64::NAN, -3.0, 3.0, 0.0];
        sort_by_abs_desc(&mut values);

        assert!(values[0].is_nan());
        assert_eq!(values[1], 3.0);
        assert_eq!(values[2], -3.0);
        assert_eq!(values[3], 0.0);
    }

    #[test]
    fn error_comparison_pushes_nan_last() {
        assert_eq!(cmp_error(1.0, f64::NAN), Ordering::Less);
        assert_eq!(cmp_error(f64::NAN, 1.0), Ordering::Greater);
        assert_eq!(cmp_error(f64::NAN, f64::NAN), Ordering::Equal);
    }

    #[test]
    fn abs_comparison_is_total_for_tied_magnitudes() {
        assert_eq!(cmp_by_abs_desc(2.0, -2.0), Ordering::Less);
        assert_eq!(cmp_by_abs_desc(-2.0, 2.0), Ordering::Greater);
        assert_eq!(cmp_by_abs_desc(1.5, 1.5), Ordering::Equal);
    }

    #[test]
    fn quantized_obstruction_family_has_zero_diagonal_and_identity_row() {
        let case = build_case(MatrixFamily::QuantizedObstructionGraph, 8).unwrap();
        for (i, row) in case.matrix.iter().enumerate() {
            assert_eq!(row[i], 0.0);
            assert_eq!(case.matrix[0][i], 0.0);
            assert_eq!(row[0], 0.0);
        }
    }

    #[test]
    fn quantized_shell_family_has_zero_diagonal_and_identity_row() {
        let case = build_case(MatrixFamily::QuantizedShellPermutation, 8).unwrap();
        for (i, row) in case.matrix.iter().enumerate() {
            assert_eq!(row[i], 0.0);
            assert_eq!(case.matrix[0][i], 0.0);
            assert_eq!(row[0], 0.0);
        }
    }

    #[test]
    fn family_selection_defaults_and_deduplicates() {
        let families = select_families(&[
            MatrixFamily::ClusteredPairs,
            MatrixFamily::KnownSpectrum,
            MatrixFamily::KnownSpectrum,
        ]);

        assert_eq!(families.len(), 2);
        assert_eq!(families[0], MatrixFamily::ClusteredPairs);
        assert_eq!(families[1], MatrixFamily::KnownSpectrum);
    }

    #[test]
    fn backend_selection_defaults_and_deduplicates() {
        let backends = select_backends(&[
            BackendKind::DoubleDouble,
            BackendKind::ReferenceF64,
            BackendKind::ReferenceF64,
        ]);

        assert_eq!(backends.len(), 2);
        assert_eq!(backends[0], BackendKind::DoubleDouble);
        assert_eq!(backends[1], BackendKind::ReferenceF64);
    }

    #[test]
    fn workload_class_string_is_stable() {
        assert_eq!(
            workload_class_str(MatrixWorkloadClass::ObstructionStructured),
            "obstruction_structured"
        );
    }
}
