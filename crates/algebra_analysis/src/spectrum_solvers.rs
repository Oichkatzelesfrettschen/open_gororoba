//! Shared solver-family helpers for symmetric spectra.

use std::collections::BTreeMap;

use crate::{
    block_jacobi, dd_jacobi, partial_spectrum,
    precision_policy::{
        JacobiBackend, MatrixStructureHints, SpectrumDispatchDecision, SpectrumDispatchInput,
        SpectrumObjective, SpectrumSolverFamily, choose_spectrum_solver,
    },
    reference_jacobi,
};
use cd_kernel::error::{AlgebraError, AlgebraResult};

/// Result of exact isolated-zero-mode deflation.
#[derive(Debug, Clone)]
pub struct StructuredReduction {
    /// Reduced matrix with isolated zero modes removed.
    pub reduced_matrix: Vec<Vec<f64>>,
    /// Indices retained in the reduced matrix.
    pub retained_indices: Vec<usize>,
    /// Number of exact zero modes removed.
    pub deflated_zero_modes: usize,
}

/// Validated equitable-partition style reduction candidate.
#[derive(Debug, Clone)]
pub struct ValidatedQuotientReduction {
    /// Partition cells.
    pub partition: Vec<Vec<usize>>,
    /// Quotient matrix built from constant row-to-cell sums.
    pub quotient_matrix: Vec<Vec<f64>>,
}

/// Benchmark-only exploratory partition seed count from row histograms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExploratoryHistogramPartition {
    /// Partition cells induced by bucketed row histograms.
    pub partition: Vec<Vec<usize>>,
}

/// Benchmark-only projection onto the histogram cell-constant subspace.
#[derive(Debug, Clone)]
pub struct HistogramProjectedReduction {
    /// Partition cells used for the projection basis.
    pub partition: Vec<Vec<usize>>,
    /// Symmetric projected matrix on the cell-constant subspace.
    pub projected_matrix: Vec<Vec<f64>>,
}

/// Classify structural hints for a symmetric matrix.
pub fn classify_symmetric_matrix(
    matrix: &[Vec<f64>],
    tolerance: f64,
) -> AlgebraResult<MatrixStructureHints> {
    let n = matrix.len();
    for row in matrix {
        if row.len() != n {
            return Err(AlgebraError::DimensionMismatch {
                expected: n,
                found: row.len(),
            });
        }
    }

    let mut symmetric = true;
    let mut nonnegative = true;
    let mut zero_diagonal = true;
    let mut nonzero_values = Vec::new();

    for (i, row) in matrix.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            if value < -tolerance {
                nonnegative = false;
            }
            if i == j && value.abs() > tolerance {
                zero_diagonal = false;
            }
            if i < j && (value - matrix[j][i]).abs() > tolerance {
                symmetric = false;
            }
            if value.abs() > tolerance {
                nonzero_values.push(value.abs());
            }
        }
    }

    let isolated_zero_modes = !find_isolated_zero_mode_indices(matrix, tolerance).is_empty();
    let quantized_value_ladder = is_quantized_ladder(&nonzero_values, tolerance);
    let equitable_partition_candidate = detect_partition_seed_count(matrix, tolerance) < n;

    Ok(MatrixStructureHints {
        symmetric,
        nonnegative,
        zero_diagonal,
        isolated_zero_modes,
        quantized_value_ladder,
        equitable_partition_candidate,
    })
}

/// Deflate isolated exact or near-exact zero modes.
pub fn deflate_isolated_zero_modes(
    matrix: &[Vec<f64>],
    tolerance: f64,
) -> AlgebraResult<StructuredReduction> {
    let n = matrix.len();
    for row in matrix {
        if row.len() != n {
            return Err(AlgebraError::DimensionMismatch {
                expected: n,
                found: row.len(),
            });
        }
    }

    let zero_indices = find_isolated_zero_mode_indices(matrix, tolerance);
    let retained_indices: Vec<usize> = (0..n).filter(|idx| !zero_indices.contains(idx)).collect();
    let reduced_matrix = retained_indices
        .iter()
        .map(|&i| {
            retained_indices
                .iter()
                .map(|&j| matrix[i][j])
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<_>>();

    Ok(StructuredReduction {
        reduced_matrix,
        retained_indices,
        deflated_zero_modes: zero_indices.len(),
    })
}

/// Detect and validate a quotient-style reduction candidate.
pub fn validated_quotient_reduction(
    matrix: &[Vec<f64>],
    tolerance: f64,
) -> AlgebraResult<Option<ValidatedQuotientReduction>> {
    let partition = partition_by_row_pattern(matrix, tolerance)?;
    if partition.len() <= 1 || partition.len() >= matrix.len() {
        return Ok(None);
    }

    let mut quotient = vec![vec![0.0; partition.len()]; partition.len()];
    for (cell_idx, cell) in partition.iter().enumerate() {
        let representative = cell[0];
        for (target_idx, target) in partition.iter().enumerate() {
            let reference_sum: f64 = target.iter().map(|&j| matrix[representative][j]).sum();
            for &row in cell {
                let row_sum: f64 = target.iter().map(|&j| matrix[row][j]).sum();
                if (row_sum - reference_sum).abs() > tolerance {
                    return Ok(None);
                }
            }
            quotient[cell_idx][target_idx] = reference_sum;
        }
    }

    Ok(Some(ValidatedQuotientReduction {
        partition,
        quotient_matrix: quotient,
    }))
}

/// Detect coarse row-histogram cells as a benchmark-only structure signal.
pub fn exploratory_histogram_partition(
    matrix: &[Vec<f64>],
    tolerance: f64,
) -> AlgebraResult<Option<ExploratoryHistogramPartition>> {
    let partition = partition_by_row_histogram(matrix, tolerance)?;
    if partition.len() <= 1 || partition.len() >= matrix.len() {
        return Ok(None);
    }
    Ok(Some(ExploratoryHistogramPartition { partition }))
}

/// Build a benchmark-only symmetric projection onto histogram cell-constant vectors.
pub fn histogram_projected_reduction(
    matrix: &[Vec<f64>],
    tolerance: f64,
) -> AlgebraResult<Option<HistogramProjectedReduction>> {
    let Some(partition) = exploratory_histogram_partition(matrix, tolerance)? else {
        return Ok(None);
    };

    let cell_count = partition.partition.len();
    let mut projected = vec![vec![0.0; cell_count]; cell_count];
    for (i, left_cell) in partition.partition.iter().enumerate() {
        for (j, right_cell) in partition.partition.iter().enumerate().skip(i) {
            let sum = left_cell
                .iter()
                .flat_map(|&lhs| right_cell.iter().map(move |&rhs| matrix[lhs][rhs]))
                .sum::<f64>();
            let scale = ((left_cell.len() * right_cell.len()) as f64).sqrt();
            let value = if scale > 0.0 { sum / scale } else { 0.0 };
            projected[i][j] = value;
            projected[j][i] = value;
        }
    }

    Ok(Some(HistogramProjectedReduction {
        partition: partition.partition,
        projected_matrix: projected,
    }))
}

/// Solve a symmetric spectrum request through the current solver-family layer.
pub fn solve_spectrum(
    matrix: &[Vec<f64>],
    input: SpectrumDispatchInput,
    tolerance: f64,
) -> AlgebraResult<Vec<f64>> {
    let decision = choose_spectrum_solver(input);
    solve_with_decision(matrix, input.objective, decision, tolerance)
}

/// Solve with an already selected solver-family decision.
pub fn solve_with_decision(
    matrix: &[Vec<f64>],
    objective: SpectrumObjective,
    decision: SpectrumDispatchDecision,
    tolerance: f64,
) -> AlgebraResult<Vec<f64>> {
    match decision.solver_family {
        SpectrumSolverFamily::FullJacobi => solve_full_jacobi_backend(matrix, decision.backend),
        SpectrumSolverFamily::StructuredReducedJacobi => {
            let reduction = deflate_isolated_zero_modes(matrix, tolerance)?;
            let mut eigs = if reduction.reduced_matrix.is_empty() {
                Vec::new()
            } else {
                solve_full_jacobi_backend(&reduction.reduced_matrix, decision.backend)?
            };
            eigs.extend(std::iter::repeat_n(0.0, reduction.deflated_zero_modes));
            eigs.sort_by(|lhs, rhs| {
                rhs.abs()
                    .total_cmp(&lhs.abs())
                    .then_with(|| rhs.total_cmp(lhs))
            });
            Ok(eigs)
        }
        SpectrumSolverFamily::PartialSubspaceIteration => {
            partial_spectrum::symmetric_extremal_eigenvalues(matrix, objective, 64, tolerance)
        }
        SpectrumSolverFamily::BlockJacobi => {
            block_jacobi::symmetric_eigenvalues_block_jacobi(matrix, 2, 16, tolerance)
        }
    }
}

fn solve_full_jacobi_backend(
    matrix: &[Vec<f64>],
    backend: JacobiBackend,
) -> AlgebraResult<Vec<f64>> {
    match backend {
        JacobiBackend::ReferenceF64 => reference_jacobi::symmetric_eigenvalues_f64(matrix),
        JacobiBackend::DoubleDouble => dd_jacobi::symmetric_eigenvalues_dd(matrix),
        JacobiBackend::X87 => {
            #[cfg(target_arch = "x86_64")]
            {
                let n = matrix.len();
                let flat = matrix.iter().flatten().copied().collect::<Vec<f64>>();
                crate::x87_jacobi::symmetric_eigenvalues_x87(&flat, n, 100 * n * n, 1.0e-15)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                Err(AlgebraError::NumericalError(
                    "x87 backend unavailable on non-x86_64 target".to_string(),
                ))
            }
        }
    }
}

fn find_isolated_zero_mode_indices(matrix: &[Vec<f64>], tolerance: f64) -> Vec<usize> {
    let mut indices = Vec::new();
    'rows: for (i, row) in matrix.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            if value.abs() > tolerance || matrix[j][i].abs() > tolerance {
                continue 'rows;
            }
        }
        indices.push(i);
    }
    indices
}

fn is_quantized_ladder(values: &[f64], tolerance: f64) -> bool {
    if values.is_empty() {
        return false;
    }
    let mut bins = Vec::<f64>::new();
    for &value in values {
        if !bins
            .iter()
            .any(|existing| (existing - value).abs() <= tolerance)
        {
            bins.push(value);
        }
    }
    bins.len() <= 8
}

fn detect_partition_seed_count(matrix: &[Vec<f64>], tolerance: f64) -> usize {
    match partition_by_row_histogram(matrix, tolerance) {
        Ok(partition) => partition.len(),
        Err(_) => matrix.len(),
    }
}

fn partition_by_row_histogram(
    matrix: &[Vec<f64>],
    tolerance: f64,
) -> AlgebraResult<Vec<Vec<usize>>> {
    let n = matrix.len();
    for row in matrix {
        if row.len() != n {
            return Err(AlgebraError::DimensionMismatch {
                expected: n,
                found: row.len(),
            });
        }
    }

    let scale = tolerance.max(1.0e-12);
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (row_idx, row) in matrix.iter().enumerate() {
        let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
        let mut row_sum = 0.0_f64;
        let mut zero_count = 0usize;
        for &value in row {
            row_sum += value;
            if value.abs() <= tolerance {
                zero_count += 1;
                continue;
            }
            let bucket = (value / scale).round() as i64;
            *counts.entry(bucket).or_insert(0) += 1;
        }
        let mut signature = format!(
            "sum:{}|zero:{}",
            (row_sum / scale).round() as i64,
            zero_count
        );
        for (bucket, count) in counts {
            signature.push_str(&format!("|{bucket}:{count}"));
        }
        groups.entry(signature).or_default().push(row_idx);
    }

    Ok(groups.into_values().collect())
}

fn partition_by_row_pattern(matrix: &[Vec<f64>], tolerance: f64) -> AlgebraResult<Vec<Vec<usize>>> {
    let n = matrix.len();
    for row in matrix {
        if row.len() != n {
            return Err(AlgebraError::DimensionMismatch {
                expected: n,
                found: row.len(),
            });
        }
    }

    let scale = tolerance.max(1.0e-12);
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (row_idx, row) in matrix.iter().enumerate() {
        let signature = row
            .iter()
            .map(|&value| {
                if value.abs() <= tolerance {
                    0_i64
                } else {
                    (value / scale).round() as i64
                }
            })
            .map(|bucket| bucket.to_string())
            .collect::<Vec<_>>()
            .join(",");
        groups.entry(signature).or_default().push(row_idx);
    }

    Ok(groups.into_values().collect())
}

#[cfg(test)]
mod tests {
    use super::{
        classify_symmetric_matrix, deflate_isolated_zero_modes, exploratory_histogram_partition,
        histogram_projected_reduction, validated_quotient_reduction,
    };

    #[test]
    fn classifier_detects_zero_modes_and_quantized_ladder() {
        let matrix = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 16.0, 48.0],
            vec![0.0, 16.0, 0.0, 48.0],
            vec![0.0, 48.0, 48.0, 0.0],
        ];

        let hints = classify_symmetric_matrix(&matrix, 1.0e-12).unwrap();
        assert!(hints.symmetric);
        assert!(hints.nonnegative);
        assert!(hints.zero_diagonal);
        assert!(hints.isolated_zero_modes);
        assert!(hints.quantized_value_ladder);
    }

    #[test]
    fn exact_zero_mode_deflation_removes_empty_row_and_column() {
        let matrix = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 2.0, 1.0],
            vec![0.0, 1.0, 3.0],
        ];

        let reduction = deflate_isolated_zero_modes(&matrix, 1.0e-12).unwrap();
        assert_eq!(reduction.deflated_zero_modes, 1);
        assert_eq!(reduction.retained_indices, vec![1, 2]);
        assert_eq!(reduction.reduced_matrix.len(), 2);
    }

    #[test]
    fn quotient_reduction_validates_repeated_row_classes() {
        let matrix = vec![
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0],
            vec![1.0, 1.0, 0.0, 0.0],
        ];

        let quotient = validated_quotient_reduction(&matrix, 1.0e-12).unwrap();
        assert!(quotient.is_some());
        let quotient = quotient.unwrap();
        assert_eq!(quotient.partition.len(), 2);
        assert_eq!(quotient.quotient_matrix.len(), 2);
    }

    #[test]
    fn histogram_partition_detects_coarse_structure_even_without_exact_quotient() {
        let matrix = vec![
            vec![0.0, 1.0, 2.0, 2.0],
            vec![1.0, 0.0, 2.0, 2.0],
            vec![2.0, 2.0, 0.0, 3.0],
            vec![2.0, 2.0, 3.0, 0.0],
        ];

        let exploratory = exploratory_histogram_partition(&matrix, 1.0e-12).unwrap();
        assert!(exploratory.is_some());
        assert_eq!(exploratory.unwrap().partition.len(), 2);
    }

    #[test]
    fn histogram_projection_builds_symmetric_cell_constant_matrix() {
        let matrix = vec![
            vec![0.0, 1.0, 2.0, 2.0],
            vec![1.0, 0.0, 2.0, 2.0],
            vec![2.0, 2.0, 0.0, 3.0],
            vec![2.0, 2.0, 3.0, 0.0],
        ];

        let projected = histogram_projected_reduction(&matrix, 1.0e-12).unwrap();
        assert!(projected.is_some());
        let projected = projected.unwrap();
        assert_eq!(projected.partition.len(), 2);
        assert_eq!(projected.projected_matrix.len(), 2);
        assert!(
            (projected.projected_matrix[0][1] - projected.projected_matrix[1][0]).abs() <= 1.0e-12
        );
    }
}
