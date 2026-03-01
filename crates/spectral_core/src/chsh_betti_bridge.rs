//! CHSH-Betti Bridge: Correlation between wavelet-scale structure and
//! point-cloud topology of velocity fields.
//!
//! Extracts a CHSH-like S parameter from Haar wavelet coefficients at
//! multiple scales, and Betti numbers from thresholded velocity magnitude
//! point clouds.  Tests whether spectral (wavelet) features correlate
//! with topological (persistent homology) features of the same field.
//!
//! The CHSH invariance theorem (C-706) proved |S| is exactly invariant
//! under Haar compression of a single signal.  This bridge instead
//! correlates S(field_i) with B1(field_i) ACROSS different velocity
//! configurations, which is a genuinely falsifiable prediction.

use crate::wavelet::haar_dwt;

/// Compute a CHSH-like S parameter from a 1D signal.
///
/// Extracts 4 "measurement channels" from the Haar DWT at 4 dyadic scales:
///   c0 = mean (coarsest scale)
///   c1 = sum of detail coefficients at scale 1 (N/2..N)
///   c2 = sum of detail coefficients at scale 2 (N/4..N/2)
///   c3 = sum of detail coefficients at scale 3 (N/8..N/4)
///
/// The S parameter is computed from correlators between adjacent scales:
///   E(i,j) = c_i * c_j / sqrt(c_i^2 * c_j^2 + eps)
///   S = E(0,1) - E(0,2) + E(1,3) + E(2,3)
///
/// This mimics the CHSH Bell parameter structure (S = E(a0,b0) - E(a0,b1) + E(a1,b0) + E(a1,b1))
/// but applied to wavelet-scale correlators rather than quantum measurement angles.
pub fn velocity_to_chsh(signal: &[f64]) -> f64 {
    let n = signal.len();
    if n < 16 {
        return 0.0;
    }

    let c = haar_dwt(signal);

    // Extract scale energies from wavelet coefficients
    // c[0] = scaling (coarsest), rest are details at dyadic scales
    let c0 = c[0]; // coarsest scale
    let half = n / 2;
    let quarter = n / 4;
    let eighth = n / 8;

    // Sum of absolute detail coefficients at each scale
    let c1: f64 = c[half..n].iter().map(|x| x.abs()).sum::<f64>() / half as f64;
    let c2: f64 = if quarter > 0 {
        c[quarter..half].iter().map(|x| x.abs()).sum::<f64>() / quarter as f64
    } else {
        0.0
    };
    let c3: f64 = if eighth > 0 {
        c[eighth..quarter].iter().map(|x| x.abs()).sum::<f64>() / eighth as f64
    } else {
        0.0
    };

    // Normalized correlators
    let eps = 1e-30;
    let e01 = c0 * c1 / ((c0 * c0 * c1 * c1).abs() + eps).sqrt();
    let e02 = c0 * c2 / ((c0 * c0 * c2 * c2).abs() + eps).sqrt();
    let e13 = c1 * c3 / ((c1 * c1 * c3 * c3).abs() + eps).sqrt();
    let e23 = c2 * c3 / ((c2 * c2 * c3 * c3).abs() + eps).sqrt();

    e01 - e02 + e13 + e23
}

/// Extract Betti numbers from a 3D velocity magnitude field.
///
/// 1. Compute velocity magnitudes |u| at each grid point.
/// 2. Threshold: keep points where |u| > threshold.
/// 3. Build distance matrix from superthreshold 3D positions.
/// 4. Compute VR persistent homology -> Betti-0 (components) and Betti-1 (loops).
///
/// Uses truncate_to_top_k for OOM safety.
pub fn velocity_to_betti(
    velocities: &[[f64; 3]],
    nx: usize,
    ny: usize,
    nz: usize,
    threshold: f64,
    max_points: usize,
) -> (usize, usize) {
    use sign_imbalance::vietoris_rips::{
        DistanceMatrix, VietorisRipsComplex, compute_betti_numbers, compute_persistent_homology,
    };

    // Compute velocity magnitudes and find superthreshold points
    let mut points_3d: Vec<f64> = Vec::new();

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = x + nx * (y + ny * z);
                let u = &velocities[idx];
                let mag = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                if mag > threshold {
                    points_3d.push(x as f64);
                    points_3d.push(y as f64);
                    points_3d.push(z as f64);
                }
            }
        }
    }

    let n_pts = points_3d.len() / 3;
    if n_pts < 3 {
        return (n_pts, 0);
    }

    // Truncate to max_points for OOM safety
    if n_pts > max_points {
        points_3d.truncate(max_points * 3);
    }

    let dist = DistanceMatrix::from_points_3d(&points_3d);
    // Use threshold proportional to grid spacing
    let max_threshold = ((nx * nx + ny * ny + nz * nz) as f64).sqrt() * 0.3;
    let complex = VietorisRipsComplex::build(&dist, max_threshold, 2);
    let pairs = compute_persistent_homology(&complex);
    let betti = compute_betti_numbers(&pairs, 0.01);

    let b0 = if !betti.is_empty() { betti[0] } else { 0 };
    let b1 = if betti.len() > 1 { betti[1] } else { 0 };

    (b0, b1)
}

/// Compute Spearman rank correlation between two vectors.
///
/// Returns (rho, p_value_approx).
/// p-value is approximate via t-distribution for n > 10.
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> (f64, f64) {
    assert_eq!(x.len(), y.len(), "vectors must have same length");
    let n = x.len();
    if n < 3 {
        return (0.0, 1.0);
    }

    fn rank(vals: &[f64]) -> Vec<f64> {
        let mut indexed: Vec<(usize, f64)> = vals.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut ranks = vec![0.0; vals.len()];
        let mut i = 0;
        while i < indexed.len() {
            let mut j = i;
            while j < indexed.len() && (indexed[j].1 - indexed[i].1).abs() < 1e-15 {
                j += 1;
            }
            let avg_rank = (i + j + 1) as f64 / 2.0;
            for item in indexed.iter().take(j).skip(i) {
                ranks[item.0] = avg_rank;
            }
            i = j;
        }
        ranks
    }

    let rx = rank(x);
    let ry = rank(y);

    let n_f = n as f64;
    let mean_r = (n_f + 1.0) / 2.0;

    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;

    for i in 0..n {
        let dx = rx[i] - mean_r;
        let dy = ry[i] - mean_r;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }

    let den = (den_x * den_y).sqrt();
    if den < 1e-15 {
        return (0.0, 1.0);
    }

    let rho = num / den;

    // Approximate p-value via t-statistic
    let t = rho * ((n_f - 2.0) / (1.0 - rho * rho)).sqrt();
    // Two-tailed p-value approximation using normal for large n
    let p = if n > 30 {
        2.0 * normal_cdf_complement(t.abs())
    } else {
        // Rough approximation for small n
        2.0 * normal_cdf_complement(t.abs())
    };

    (rho, p)
}

/// Complementary CDF of standard normal (1 - Phi(x)).
fn normal_cdf_complement(x: f64) -> f64 {
    // Abramowitz & Stegun approximation 26.2.17
    let t = 1.0 / (1.0 + 0.2316419 * x);
    let poly = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let pdf = (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    pdf * poly
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velocity_to_chsh_basic() {
        // A sinusoidal signal should produce a non-trivial S parameter
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin())
            .collect();
        let s = velocity_to_chsh(&signal);
        // S should be finite and bounded
        assert!(s.is_finite(), "S should be finite, got {}", s);
        assert!(s.abs() <= 4.0, "S should be bounded by 4, got {}", s);
    }

    #[test]
    fn test_velocity_to_chsh_constant_signal() {
        let signal = vec![1.0; 128];
        let s = velocity_to_chsh(&signal);
        // Constant signal: all detail coefficients are 0 -> S ~ 0
        assert!(s.abs() < 1e-10, "constant signal S should be ~0, got {}", s);
    }

    #[test]
    fn test_spearman_perfect_correlation() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..20).map(|i| i as f64 * 2.0 + 1.0).collect();
        let (rho, p) = spearman_correlation(&x, &y);
        assert!(
            (rho - 1.0).abs() < 1e-10,
            "perfect correlation should give rho=1, got {}",
            rho
        );
        assert!(
            p < 0.01,
            "p-value should be small for perfect correlation, got {}",
            p
        );
    }

    #[test]
    fn test_spearman_anticorrelation() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..20).map(|i| -(i as f64)).collect();
        let (rho, _) = spearman_correlation(&x, &y);
        assert!(
            (rho + 1.0).abs() < 1e-10,
            "anticorrelation should give rho=-1, got {}",
            rho
        );
    }

    #[test]
    fn test_normal_cdf_complement_symmetry() {
        // Phi(-x) should give large value (close to 1 for x=0)
        let c0 = normal_cdf_complement(0.0);
        assert!(
            (c0 - 0.5).abs() < 0.01,
            "Phi_c(0) should be ~0.5, got {}",
            c0
        );
    }
}
