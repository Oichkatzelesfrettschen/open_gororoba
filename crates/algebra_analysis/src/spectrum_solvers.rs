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

/// Benchmark-only full partition-adapted basis transform and coupling analysis.
#[derive(Debug, Clone)]
pub struct HistogramAdaptedBasisReduction {
    /// Partition cells used to build the adapted basis.
    pub partition: Vec<Vec<usize>>,
    /// Full symmetric matrix in the partition-adapted orthonormal basis.
    pub transformed_matrix: Vec<Vec<f64>>,
    /// Number of coarse cell-constant basis vectors.
    pub coarse_dim: usize,
    /// Frobenius-energy ratio of centered-vs-centered couplings across distinct cells.
    pub centered_cross_cell_fro_ratio: f64,
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

/// Build the full partition-adapted orthonormal basis and transformed matrix.
pub fn histogram_adapted_basis_reduction(
    matrix: &[Vec<f64>],
    tolerance: f64,
) -> AlgebraResult<Option<HistogramAdaptedBasisReduction>> {
    let Some(partition) = exploratory_histogram_partition(matrix, tolerance)? else {
        return Ok(None);
    };

    let total_dim = matrix.len();
    let basis = build_partition_basis(&partition.partition, total_dim);
    let basis_dim = basis.len();
    let mut transformed = vec![vec![0.0; basis_dim]; basis_dim];
    let mut total_fro_sq = 0.0_f64;
    let mut centered_cross_cell_fro_sq = 0.0_f64;

    for i in 0..basis_dim {
        for j in i..basis_dim {
            let value = bilinear_form(&basis[i].1, matrix, &basis[j].1);
            transformed[i][j] = value;
            transformed[j][i] = value;
            let weight = if i == j { 1.0 } else { 2.0 };
            let energy = weight * value * value;
            total_fro_sq += energy;
            if matches!(
                (&basis[i].0, &basis[j].0),
                (PartitionBasisRole::Centered(lhs), PartitionBasisRole::Centered(rhs)) if lhs != rhs
            ) {
                centered_cross_cell_fro_sq += energy;
            }
        }
    }

    Ok(Some(HistogramAdaptedBasisReduction {
        partition: partition.partition.clone(),
        transformed_matrix: transformed,
        coarse_dim: partition.partition.len(),
        centered_cross_cell_fro_ratio: if total_fro_sq > 0.0 {
            centered_cross_cell_fro_sq / total_fro_sq
        } else {
            0.0
        },
    }))
}

/// Benchmark-only two-level lifted spectrum from histogram cells.
pub fn histogram_two_level_spectrum(
    matrix: &[Vec<f64>],
    tolerance: f64,
) -> AlgebraResult<Option<Vec<f64>>> {
    let reduction = deflate_isolated_zero_modes(matrix, tolerance)?;
    if reduction.reduced_matrix.is_empty() {
        return Ok(Some(vec![0.0; reduction.deflated_zero_modes]));
    }

    let Some(projected) = histogram_projected_reduction(&reduction.reduced_matrix, tolerance)?
    else {
        return Ok(None);
    };

    let mut eigs = reference_jacobi::symmetric_eigenvalues_f64(&projected.projected_matrix)?;
    for cell in &projected.partition {
        if cell.len() <= 1 {
            continue;
        }
        let centered_block = restrict_centered_cell_block(&reduction.reduced_matrix, cell);
        if centered_block.is_empty() {
            continue;
        }
        eigs.extend(reference_jacobi::symmetric_eigenvalues_f64(
            &centered_block,
        )?);
    }
    eigs.extend(std::iter::repeat_n(0.0, reduction.deflated_zero_modes));
    eigs.sort_by(|lhs, rhs| {
        rhs.abs()
            .total_cmp(&lhs.abs())
            .then_with(|| rhs.total_cmp(lhs))
    });
    Ok(Some(eigs))
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

fn restrict_centered_cell_block(matrix: &[Vec<f64>], cell: &[usize]) -> Vec<Vec<f64>> {
    let basis = centered_basis(cell.len());
    if basis.is_empty() {
        return Vec::new();
    }

    let mut restricted = vec![vec![0.0; basis.len()]; basis.len()];
    for (i, lhs) in basis.iter().enumerate() {
        for (j, rhs) in basis.iter().enumerate().skip(i) {
            let mut value = 0.0_f64;
            for (local_p, &global_p) in cell.iter().enumerate() {
                for (local_q, &global_q) in cell.iter().enumerate() {
                    value += lhs[local_p] * matrix[global_p][global_q] * rhs[local_q];
                }
            }
            restricted[i][j] = value;
            restricted[j][i] = value;
        }
    }
    restricted
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PartitionBasisRole {
    Coarse(usize),
    Centered(usize),
}

fn build_partition_basis(
    partition: &[Vec<usize>],
    total_dim: usize,
) -> Vec<(PartitionBasisRole, Vec<f64>)> {
    let mut basis = Vec::new();
    for (cell_idx, cell) in partition.iter().enumerate() {
        let scale = (cell.len() as f64).sqrt();
        let mut coarse = vec![0.0_f64; total_dim];
        for &global_idx in cell {
            coarse[global_idx] = 1.0 / scale;
        }
        basis.push((PartitionBasisRole::Coarse(cell_idx), coarse));
    }

    for (cell_idx, cell) in partition.iter().enumerate() {
        for local_basis in centered_basis(cell.len()) {
            let mut embedded = vec![0.0_f64; total_dim];
            for (local_idx, &global_idx) in cell.iter().enumerate() {
                embedded[global_idx] = local_basis[local_idx];
            }
            basis.push((PartitionBasisRole::Centered(cell_idx), embedded));
        }
    }

    basis
}

fn centered_basis(cell_size: usize) -> Vec<Vec<f64>> {
    if cell_size <= 1 {
        return Vec::new();
    }

    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(cell_size - 1);
    for axis in 0..(cell_size - 1) {
        let mut vector = vec![0.0_f64; cell_size];
        vector[axis] = 1.0;
        vector[cell_size - 1] = -1.0;

        for existing in &basis {
            let dot = dot_product(&vector, existing);
            for (value, &component) in vector.iter_mut().zip(existing.iter()) {
                *value -= dot * component;
            }
        }

        let norm = dot_product(&vector, &vector).sqrt();
        if norm > 0.0 {
            for value in &mut vector {
                *value /= norm;
            }
            basis.push(vector);
        }
    }

    basis
}

fn dot_product(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
}

fn bilinear_form(lhs: &[f64], matrix: &[Vec<f64>], rhs: &[f64]) -> f64 {
    lhs.iter()
        .enumerate()
        .map(|(i, &lhs_value)| {
            if lhs_value == 0.0 {
                0.0
            } else {
                rhs.iter()
                    .enumerate()
                    .map(|(j, &rhs_value)| lhs_value * matrix[i][j] * rhs_value)
                    .sum::<f64>()
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::{
        classify_symmetric_matrix, deflate_isolated_zero_modes, exploratory_histogram_partition,
        histogram_adapted_basis_reduction, histogram_projected_reduction,
        histogram_two_level_spectrum, validated_quotient_reduction,
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

    #[test]
    fn histogram_two_level_spectrum_preserves_matrix_dimension() {
        let matrix = vec![
            vec![0.0, 1.0, 2.0, 2.0],
            vec![1.0, 0.0, 2.0, 2.0],
            vec![2.0, 2.0, 0.0, 3.0],
            vec![2.0, 2.0, 3.0, 0.0],
        ];

        let eigs = histogram_two_level_spectrum(&matrix, 1.0e-12).unwrap();
        assert!(eigs.is_some());
        assert_eq!(eigs.unwrap().len(), matrix.len());
    }

    #[test]
    fn histogram_adapted_basis_reduction_preserves_dimension_and_reports_coupling() {
        let matrix = vec![
            vec![0.0, 1.0, 2.0, 2.0],
            vec![1.0, 0.0, 2.0, 2.0],
            vec![2.0, 2.0, 0.0, 3.0],
            vec![2.0, 2.0, 3.0, 0.0],
        ];

        let adapted = histogram_adapted_basis_reduction(&matrix, 1.0e-12).unwrap();
        assert!(adapted.is_some());
        let adapted = adapted.unwrap();
        assert_eq!(adapted.partition.len(), 2);
        assert_eq!(adapted.transformed_matrix.len(), matrix.len());
        assert!(adapted.centered_cross_cell_fro_ratio >= 0.0);
        assert!(adapted.centered_cross_cell_fro_ratio <= 1.0);
    }
}

/// Compute sorted eigenvalues of a symmetric matrix using nalgebra's dense QR solver.
///
/// WHY: Several benchmark/test binaries use nalgebra as a ground-truth reference for
/// verifying Jacobi solvers.  Centralising the call here removes the direct nalgebra
/// dep from those binaries; all nalgebra types are hidden behind this API.
///
/// The returned vector is ordered abs-descending (largest |eigenvalue| first); ties
/// are broken by value descending.  Empty input returns an empty vec.
pub fn symmetric_eigenvalues_sorted(matrix: &[Vec<f64>]) -> Vec<f64> {
    use nalgebra::DMatrix;
    let n = matrix.len();
    if n == 0 {
        return vec![];
    }
    let dense = DMatrix::from_fn(n, n, |i, j| matrix[i][j]);
    let mut eigs: Vec<f64> = dense.symmetric_eigenvalues().iter().copied().collect();
    eigs.sort_by(|lhs, rhs| {
        rhs.abs()
            .total_cmp(&lhs.abs())
            .then_with(|| rhs.total_cmp(lhs))
    });
    eigs
}
