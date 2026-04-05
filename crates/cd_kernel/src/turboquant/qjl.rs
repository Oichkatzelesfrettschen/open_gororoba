//! Quantized Johnson-Lindenstrauss sign sketches (QJL).
//!
//! Stage 2 of TurboQuant: project the quantization residual through a random
//! Gaussian matrix and store only the sign bit.  The asymmetric inner product
//! estimator is unbiased with correction scale sqrt(pi/2) / m.
//!
//! Reference: Zandieh et al., "QJL: 1-Bit Quantized JL Transform for KV Cache
//! Quantization with Zero Overhead" (AAAI 2025, arXiv 2406.03482).

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

/// Generate a random projection matrix S of shape (m, d) with i.i.d. N(0,1).
///
/// From `turboquant.py:generate_qjl_matrix`.  Default m = d.
/// Returns flat row-major array of length m * d.
pub fn generate_projection_matrix(d: usize, m: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let normal = StandardNormal;
    (0..m * d).map(|_| normal.sample(&mut rng)).collect()
}

/// Sign-quantize a residual vector.
///
/// Algorithm (from `turboquant.py:TurboQuantProd.quantize`):
///   1. projected = residual @ S^T   (project into m-dim space)
///   2. signs = sign(projected)       (quantize to +/-1)
///   3. map zeros to +1
///
/// `s_matrix` is (m, d) row-major.
/// `residual` has length d.
/// `signs_out` has length m, filled with +1 or -1 as i8.
/// `residual_norm` is set to ||residual||.
pub fn sign_quantize(
    residual: &[f64],
    s_matrix: &[f64],
    d: usize,
    m: usize,
    signs_out: &mut [i8],
    residual_norm: &mut f64,
) {
    debug_assert_eq!(residual.len(), d);
    debug_assert_eq!(s_matrix.len(), m * d);
    debug_assert_eq!(signs_out.len(), m);

    // Compute residual norm
    *residual_norm = residual.iter().map(|&v| v * v).sum::<f64>().sqrt();

    // Project and take sign
    for (j, sign_out) in signs_out.iter_mut().enumerate() {
        let mut dot = 0.0;
        let row_start = j * d;
        for i in 0..d {
            dot += s_matrix[row_start + i] * residual[i];
        }
        // sign(x): +1 if x >= 0, -1 if x < 0 (zeros map to +1)
        *sign_out = if dot >= 0.0 { 1 } else { -1 };
    }
}

/// Asymmetric inner product estimator: <query, original> ~= Term1 + Term2.
///
/// Algorithm (from `turboquant.py:TurboQuantProd.inner_product`):
///   Term1 = <query, x_mse>           (dot with MSE reconstruction)
///   Term2 = ||r|| * sqrt(pi/2) / m * <S @ query, signs>
///
/// `query` has length d.
/// `x_mse` has length d (MSE reconstruction of original).
/// `s_matrix` is (m, d) row-major (the QJL projection matrix).
/// `signs` has length m (i8, +/-1).
/// `residual_norm` is ||original - x_mse||.
///
/// Returns the estimated inner product <query, original>.
pub fn asymmetric_inner_product(
    query: &[f64],
    x_mse: &[f64],
    s_matrix: &[f64],
    signs: &[i8],
    residual_norm: f64,
    d: usize,
    m: usize,
) -> f64 {
    debug_assert_eq!(query.len(), d);
    debug_assert_eq!(x_mse.len(), d);
    debug_assert_eq!(s_matrix.len(), m * d);
    debug_assert_eq!(signs.len(), m);

    // Term 1: <query, x_mse>
    let term1: f64 = query.iter().zip(x_mse.iter()).map(|(q, x)| q * x).sum();

    // Term 2: ||r|| * sqrt(pi/2) / m * <S @ query, signs>
    // Compute S @ query (project query into m-dim space)
    let mut s_query_dot_signs = 0.0;
    for (j, &sign) in signs.iter().enumerate() {
        let mut proj = 0.0;
        let row_start = j * d;
        for i in 0..d {
            proj += s_matrix[row_start + i] * query[i];
        }
        s_query_dot_signs += proj * sign as f64;
    }

    let correction_scale = (std::f64::consts::FRAC_PI_2).sqrt() / m as f64;
    let term2 = residual_norm * correction_scale * s_query_dot_signs;

    term1 + term2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projection_matrix_shape() {
        let s = generate_projection_matrix(128, 128, 42);
        assert_eq!(s.len(), 128 * 128);
    }

    #[test]
    fn test_sign_quantize_values() {
        let d = 8;
        let m = 8;
        let residual = vec![0.1, -0.2, 0.3, -0.1, 0.05, -0.15, 0.25, -0.05];
        // Identity-like projection for easy verification
        let mut s = vec![0.0; m * d];
        for i in 0..m.min(d) {
            s[i * d + i] = 1.0;
        }
        let mut signs = vec![0i8; m];
        let mut norm = 0.0;
        sign_quantize(&residual, &s, d, m, &mut signs, &mut norm);
        assert!(norm > 0.0);
        // With identity projection, sign(projected) = sign(residual)
        assert_eq!(signs[0], 1); // 0.1 >= 0
        assert_eq!(signs[1], -1); // -0.2 < 0
        assert_eq!(signs[2], 1); // 0.3 >= 0
        assert_eq!(signs[3], -1); // -0.1 < 0
    }

    #[test]
    fn test_asymmetric_ip_unbiased() {
        // Statistical test: over many random vectors, the estimator
        // should converge to the true inner product.
        let d = 64;
        let m = d;
        let s_matrix = generate_projection_matrix(d, m, 42);

        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let normal = StandardNormal;

        let n_trials = 2000;
        let mut total_error = 0.0;

        for trial in 0..n_trials {
            // Random original vector
            let x: Vec<f64> = (0..d).map(|_| normal.sample(&mut rng)).collect();
            // Random query
            let q: Vec<f64> = (0..d).map(|_| normal.sample(&mut rng)).collect();

            // True inner product
            let true_ip: f64 = x.iter().zip(q.iter()).map(|(a, b)| a * b).sum();

            // Simulate MSE quantization (add small noise)
            let x_mse: Vec<f64> = x
                .iter()
                .map(|&v| v + 0.01 * (trial as f64 * 0.1).sin())
                .collect();
            let residual: Vec<f64> = x.iter().zip(x_mse.iter()).map(|(a, b)| a - b).collect();

            let mut signs = vec![0i8; m];
            let mut r_norm = 0.0;
            sign_quantize(&residual, &s_matrix, d, m, &mut signs, &mut r_norm);

            let estimated_ip =
                asymmetric_inner_product(&q, &x_mse, &s_matrix, &signs, r_norm, d, m);

            total_error += estimated_ip - true_ip;
        }

        let mean_error = total_error / n_trials as f64;
        // Unbiased: mean error should be near zero
        assert!(
            mean_error.abs() < 0.5,
            "QJL estimator biased: mean error = {} over {} trials",
            mean_error,
            n_trials
        );
    }
}
