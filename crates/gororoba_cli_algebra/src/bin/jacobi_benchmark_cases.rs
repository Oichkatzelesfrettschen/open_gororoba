use algebra_analysis::{
    homotopy_algebra::SedenionAInfinity,
    precision_policy::{MatrixStructureHints, MatrixWorkloadClass},
    spectrum_solvers::{classify_symmetric_matrix, symmetric_eigenvalues_sorted},
};
use anyhow::{Context, Result, ensure};
use clap::ValueEnum;
use std::fmt;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MatrixCase {
    pub family_name: &'static str,
    pub matrix: Vec<Vec<f64>>,
    pub expected_spectrum: Vec<f64>,
    pub workload_class: MatrixWorkloadClass,
    pub structure_hints: MatrixStructureHints,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "snake_case")]
pub enum MatrixFamily {
    KnownSpectrum,
    ClusteredPairs,
    GeometricDecay,
    SpikedTail,
    ObstructionPlateau,
    IdentityNullspacePlateau,
    QuantizedObstructionGraph,
    QuantizedShellPermutation,
    RealObstruction,
}

impl MatrixFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::KnownSpectrum => "known_spectrum",
            Self::ClusteredPairs => "clustered_pairs",
            Self::GeometricDecay => "geometric_decay",
            Self::SpikedTail => "spiked_tail",
            Self::ObstructionPlateau => "obstruction_plateau",
            Self::IdentityNullspacePlateau => "identity_nullspace_plateau",
            Self::QuantizedObstructionGraph => "quantized_obstruction_graph",
            Self::QuantizedShellPermutation => "quantized_shell_permutation",
            Self::RealObstruction => "real_obstruction",
        }
    }

    pub fn defaults() -> &'static [Self] {
        &[
            Self::KnownSpectrum,
            Self::ClusteredPairs,
            Self::GeometricDecay,
            Self::SpikedTail,
            Self::ObstructionPlateau,
            Self::IdentityNullspacePlateau,
            Self::QuantizedObstructionGraph,
            Self::QuantizedShellPermutation,
        ]
    }

    pub fn workload_class(self) -> MatrixWorkloadClass {
        match self {
            Self::KnownSpectrum
            | Self::ClusteredPairs
            | Self::GeometricDecay
            | Self::SpikedTail => MatrixWorkloadClass::FullSpectrumDense,
            Self::ObstructionPlateau
            | Self::IdentityNullspacePlateau
            | Self::QuantizedObstructionGraph
            | Self::QuantizedShellPermutation
            | Self::RealObstruction => MatrixWorkloadClass::ObstructionStructured,
        }
    }
}

impl fmt::Display for MatrixFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

pub fn select_families(requested: &[MatrixFamily]) -> Vec<MatrixFamily> {
    let mut families = if requested.is_empty() {
        MatrixFamily::defaults().to_vec()
    } else {
        requested.to_vec()
    };
    families.sort_by_key(|family| family.as_str());
    families.dedup();
    families
}

pub fn build_case(family: MatrixFamily, size: usize) -> Result<MatrixCase> {
    let matrix = match family {
        MatrixFamily::QuantizedObstructionGraph => build_quantized_obstruction_graph(size),
        MatrixFamily::QuantizedShellPermutation => build_quantized_shell_permutation(size),
        MatrixFamily::RealObstruction => {
            ensure!(
                size >= 8 && size.is_power_of_two(),
                "real obstruction family requires power-of-two size >= 8"
            );
            SedenionAInfinity::new(size).obstruction_matrix()
        }
        _ => householder_known_spectrum(&build_proxy_diagonal(family, size)),
    };
    let expected_spectrum = symmetric_eigenvalues_sorted(&matrix);
    let structure_hints = classify_symmetric_matrix(&matrix, 1.0e-12)
        .with_context(|| format!("classifying {}", family.as_str()))?;

    Ok(MatrixCase {
        family_name: family.as_str(),
        matrix,
        expected_spectrum,
        workload_class: family.workload_class(),
        structure_hints,
    })
}

fn build_proxy_diagonal(family: MatrixFamily, size: usize) -> Vec<f64> {
    let mut diag = match family {
        MatrixFamily::KnownSpectrum => (0..size).map(|i| (i + 1) as f64).collect::<Vec<_>>(),
        MatrixFamily::ClusteredPairs => (0..size)
            .map(|i| {
                let base = 1.0 + (i / 2) as f64;
                if i % 2 == 0 { base } else { base + 1.0e-6 }
            })
            .collect::<Vec<_>>(),
        MatrixFamily::GeometricDecay => (0..size)
            .map(|i| 2f64.powf(-0.5 * i as f64))
            .collect::<Vec<_>>(),
        MatrixFamily::SpikedTail => (0..size)
            .map(|i| match i {
                0 => size as f64,
                1 => (size as f64) * 0.5,
                2 => (size as f64) * 0.25,
                _ => 1.0e-3 * (size - i) as f64,
            })
            .collect::<Vec<_>>(),
        MatrixFamily::ObstructionPlateau => {
            let bulk = (size as f64) * 0.8;
            let tail = (size as f64) * 0.16;
            let mut values = Vec::with_capacity(size);
            values.push((size as f64) * 3.1);
            if size > 1 {
                values.push((size as f64) * 0.9);
            }
            while values.len() + 1 < size {
                if values.len() < 2 + size / 2 {
                    values.push(-bulk);
                } else {
                    values.push(-tail);
                }
            }
            while values.len() < size {
                values.push(0.0);
            }
            values
        }
        MatrixFamily::IdentityNullspacePlateau => {
            let mut values = Vec::with_capacity(size);
            values.push((size as f64) * 2.0);
            for i in 1..size {
                if i + 1 == size {
                    values.push(0.0);
                } else if i <= size / 3 {
                    values.push((size as f64) * 0.5);
                } else {
                    values.push(-(size as f64) * 0.35);
                }
            }
            values
        }
        MatrixFamily::QuantizedObstructionGraph
        | MatrixFamily::QuantizedShellPermutation
        | MatrixFamily::RealObstruction => unreachable!("handled above"),
    };
    sort_by_abs_desc(&mut diag);
    diag
}

fn householder_known_spectrum(diagonal: &[f64]) -> Vec<Vec<f64>> {
    let n = diagonal.len();
    let inv_n = 1.0 / n as f64;
    let mut matrix = vec![vec![0.0; n]; n];
    for (i, row) in matrix.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let mut value = 0.0;
            for (k, &lambda) in diagonal.iter().enumerate() {
                let h_ik = if i == k { 1.0 } else { 0.0 } - 2.0 * inv_n;
                let h_jk = if j == k { 1.0 } else { 0.0 } - 2.0 * inv_n;
                value += h_ik * lambda * h_jk;
            }
            *cell = value;
        }
    }
    matrix
}

fn build_quantized_obstruction_graph(size: usize) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; size]; size];
    if size <= 1 {
        return matrix;
    }

    for i in 0..size {
        let (head, tail) = matrix.split_at_mut(i + 1);
        let row = &mut head[i];
        for (offset, other_row) in tail.iter_mut().enumerate() {
            let j = i + 1 + offset;
            let value = quantized_obstruction_weight(i, j);
            row[j] = value;
            other_row[i] = value;
        }
    }

    matrix
}

fn build_quantized_shell_permutation(size: usize) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; size]; size];
    if size <= 2 {
        return matrix;
    }

    for i in 0..size {
        let (head, tail) = matrix.split_at_mut(i + 1);
        let row = &mut head[i];
        for (offset, other_row) in tail.iter_mut().enumerate() {
            let j = i + 1 + offset;
            let value = quantized_shell_weight(i, j, size);
            row[j] = value;
            other_row[i] = value;
        }
    }

    matrix
}

fn quantized_obstruction_weight(i: usize, j: usize) -> f64 {
    if i == j || i == 0 || j == 0 {
        return 0.0;
    }

    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
    if lo == 1 {
        return if hi.is_multiple_of(2) { 112.0 } else { 80.0 };
    }
    if lo == 2 {
        return if hi.is_multiple_of(3) { 80.0 } else { 48.0 };
    }

    let bucket = (lo + hi) % 4;
    16.0 * (2 * bucket + 1) as f64
}

fn quantized_shell_weight(i: usize, j: usize, size: usize) -> f64 {
    if i == j || i == 0 || j == 0 {
        return 0.0;
    }

    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
    if lo == 1 {
        return match hi % 4 {
            0 => 112.0,
            1 => 80.0,
            2 => 48.0,
            _ => 16.0,
        };
    }

    let span = size.saturating_sub(1).max(1);
    let mut dist = hi - lo;
    dist = dist.min(span - dist);

    match dist % 4 {
        0 => 16.0,
        1 => 48.0,
        2 => 80.0,
        _ => 112.0,
    }
}

fn sort_by_abs_desc(values: &mut [f64]) {
    values.sort_by(|lhs, rhs| {
        rhs.abs()
            .total_cmp(&lhs.abs())
            .then_with(|| rhs.total_cmp(lhs))
    });
}

#[cfg(test)]
mod tests {
    use super::{MatrixFamily, build_case};

    #[test]
    fn real_obstruction_case_requires_power_of_two_size() {
        let error = build_case(MatrixFamily::RealObstruction, 12).unwrap_err();
        assert!(error.to_string().contains("power-of-two"));
    }

    #[test]
    fn quantized_obstruction_case_has_metadata() {
        let case = build_case(MatrixFamily::QuantizedObstructionGraph, 8).unwrap();
        assert!(case.structure_hints.symmetric);
        assert!(case.structure_hints.zero_diagonal);
        assert!(case.structure_hints.quantized_value_ladder);
    }
}
