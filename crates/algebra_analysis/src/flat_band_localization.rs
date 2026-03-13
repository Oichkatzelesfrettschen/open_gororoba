//! Flat band fraction scaling across Cayley-Dickson dimensions.
//!
//! Measures how the flat band fraction (fraction of zero eigenvalues
//! in the ZD motif adjacency spectrum) scales with CD dimension.
//!
//! If flat band fraction is preserved across doublings, this provides
//! algebraic justification for CD dark matter localization: the internal
//! algebraic structure acts as a frustrated lattice that prevents
//! dispersion.

use crate::crystal_bands::dim_spectrum_summary;

/// Result of flat band fraction scaling analysis.
#[derive(Debug, Clone)]
pub struct FlatBandScaling {
    /// Dimensions analyzed.
    pub dims: Vec<usize>,
    /// Measured flat band fractions at each dimension.
    pub fractions: Vec<f64>,
    /// Fit type: "constant" if all values within tolerance of each other,
    /// or "varying" if they differ significantly.
    pub fit_type: String,
    /// For "constant": the mean value. For "varying": the range (max - min).
    pub fit_value: f64,
    /// Mean absolute deviation from the fit.
    pub residual: f64,
}

/// Measure flat band fraction at each dimension and characterize scaling.
///
/// If all fractions are within `tol` of each other, the scaling is "constant"
/// (implying dimension-independent localization strength). Otherwise "varying".
pub fn flat_band_scaling(dims: &[usize], tol: f64) -> FlatBandScaling {
    let mut fractions = Vec::with_capacity(dims.len());

    for &dim in dims {
        let summary = dim_spectrum_summary(dim, tol);
        fractions.push(summary.flat_band_fraction);
    }

    let mean = if fractions.is_empty() {
        0.0
    } else {
        fractions.iter().sum::<f64>() / fractions.len() as f64
    };

    let mad = if fractions.is_empty() {
        0.0
    } else {
        fractions.iter().map(|f| (f - mean).abs()).sum::<f64>() / fractions.len() as f64
    };

    let range = fractions.iter().copied().fold(f64::NAN, f64::max)
        - fractions.iter().copied().fold(f64::NAN, f64::min);

    let (fit_type, fit_value) = if range.is_nan() || range < tol {
        ("constant".to_string(), mean)
    } else {
        ("varying".to_string(), range)
    };

    FlatBandScaling {
        dims: dims.to_vec(),
        fractions,
        fit_type,
        fit_value,
        residual: mad,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d16_d32_scaling() {
        let result = flat_band_scaling(&[16, 32], 0.01);
        assert_eq!(result.dims.len(), 2);
        assert_eq!(result.fractions.len(), 2);
        // Both D=16 and D=32 have flat band fraction = 0.5
        for (dim, &fbf) in result.dims.iter().zip(result.fractions.iter()) {
            assert!(
                (fbf - 0.5).abs() < 0.05,
                "D={dim} flat band fraction = {fbf}, expected ~0.5"
            );
        }
        assert_eq!(result.fit_type, "constant");
        eprintln!(
            "Scaling: type={}, value={:.4}, residual={:.6}",
            result.fit_type, result.fit_value, result.residual
        );
    }

    #[test]
    fn test_empty() {
        let result = flat_band_scaling(&[], 0.01);
        assert!(result.dims.is_empty());
        assert!(result.fractions.is_empty());
    }

    /// D=64 flat band fraction should be ~0.5, extending the pattern
    /// from D=16 (42/84 = 0.5) and D=32 (105/210 = 0.5).
    ///
    /// This test is expensive (O(dim^3) eigenvalue computation) so it
    /// is gated behind #[ignore]. Run with:
    ///   cargo test -p algebra_analysis -- --ignored test_d64_flat_band
    #[test]
    #[ignore]
    fn test_d64_flat_band_fraction() {
        let result = flat_band_scaling(&[64], 0.01);
        assert_eq!(result.dims.len(), 1);
        let fbf = result.fractions[0];
        assert!(
            (fbf - 0.5).abs() < 0.05,
            "D=64 flat band fraction = {fbf}, expected ~0.5 (within 0.05)"
        );
        eprintln!("D=64 flat band fraction: {fbf:.6}");
    }

    /// Verify scaling holds across D=16, D=32, D=64.
    #[test]
    #[ignore]
    fn test_d16_d32_d64_scaling() {
        let result = flat_band_scaling(&[16, 32, 64], 0.01);
        assert_eq!(result.dims.len(), 3);
        for (dim, &fbf) in result.dims.iter().zip(result.fractions.iter()) {
            assert!(
                (fbf - 0.5).abs() < 0.05,
                "D={dim} flat band fraction = {fbf}, expected ~0.5"
            );
        }
        assert_eq!(
            result.fit_type, "constant",
            "Scaling should be constant across D=16/32/64, got: {} (range={:.4})",
            result.fit_type, result.fit_value
        );
        eprintln!(
            "Scaling: type={}, value={:.4}, residual={:.6}",
            result.fit_type, result.fit_value, result.residual
        );
    }
}
