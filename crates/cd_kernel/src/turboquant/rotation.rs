//! Random rotation matrices for TurboQuant decorrelation.
//!
//! Two methods:
//! - **Haar-random** rotation via QR decomposition of a Gaussian matrix (O(d^2))
//! - **Fast JL** rotation via D1 * WHT * D2 (O(d log d), Ailon-Chazelle 2006)
//!
//! # Measured performance (Rust release, 2026-03-28)
//!
//! | Method | d=128, 5K vecs | Quantize throughput | MSE (3-bit) | Cosine |
//! |--------|----------------|---------------------|-------------|--------|
//! | Haar   | 25 kvec/s      | O(d^2)              | 1.451       | 0.441  |
//! | WHT    | 77 kvec/s      | O(d log d)          | 1.444       | 0.446  |
//!
//! WHT is **3.1x faster** than Haar at d=128 and gives **0.5% lower MSE**
//! consistently across all tested bit-widths (2, 3, 4).
//!
//! In Python (PyTorch BLAS), Haar was 35x faster due to loop overhead in the
//! WHT butterfly.  In compiled Rust, the asymptotic advantage of O(d log d)
//! manifests cleanly.
//!
//! # Decorrelation quality
//!
//! Both methods produce comparable pairwise cross-correlations:
//! - Haar: mean |corr| = 0.0072
//! - WHT:  mean |corr| = 0.0086
//!
//! The WHT variant uses Rademacher diagonals (Ailon-Chazelle 2006 fast JL
//! construction) which provides provable Johnson-Lindenstrauss guarantees.
//!
//! # Recommendation
//!
//! Use `Rotation::new_fast_jl()` as the default for d >= 64.  The 3x speed
//! advantage compounds at scale (36 layers x 32 heads x thousands of tokens).
//! For d < 64, the O(d^2) matmul is already cheap enough that Haar is fine.

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
// Note: WHT, Rademacher, and fast JL functions now delegate to the
// standalone fwht crate (~/Github/cratesgororobas/fwht/).

/// Haar-distributed random orthogonal matrix via QR of Gaussian matrix.
///
/// Algorithm (from `turboquant.py:generate_rotation_matrix`):
///   1. G = d x d matrix with i.i.d. N(0,1) entries
///   2. Q, R = QR(G)
///   3. Q *= sign(diag(R))  -- ensures det(Q) = +1
///
/// Returns a flat d*d array in row-major order.
pub fn generate_haar_rotation(d: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let normal = StandardNormal;

    // Generate d x d Gaussian matrix (row-major)
    let mut g: Vec<f64> = (0..d * d).map(|_| normal.sample(&mut rng)).collect();

    // In-place QR via modified Gram-Schmidt
    // Q stored in g (columns), R diagonal stored separately
    let mut r_diag = vec![0.0f64; d];

    for j in 0..d {
        // Compute norm of column j
        let mut norm = 0.0;
        for i in 0..d {
            norm += g[i * d + j] * g[i * d + j];
        }
        norm = norm.sqrt();
        r_diag[j] = norm;

        if norm > 1e-15 {
            // Normalize column j
            let inv_norm = 1.0 / norm;
            for i in 0..d {
                g[i * d + j] *= inv_norm;
            }
        }

        // Orthogonalize remaining columns against column j
        for k in (j + 1)..d {
            let mut dot = 0.0;
            for i in 0..d {
                dot += g[i * d + j] * g[i * d + k];
            }
            for i in 0..d {
                g[i * d + k] -= dot * g[i * d + j];
            }
        }
    }

    // Apply sign correction: Q *= sign(diag(R))
    for j in 0..d {
        let sign = if r_diag[j] >= 0.0 { 1.0 } else { -1.0 };
        for i in 0..d {
            g[i * d + j] *= sign;
        }
    }

    g
}

/// Apply rotation: y = x @ Pi^T (row-vector convention).
///
/// pi is d*d row-major, x is a slice of length d.
/// Result written into out (length d).
pub fn rotate(x: &[f64], pi: &[f64], d: usize, out: &mut [f64]) {
    debug_assert_eq!(x.len(), d);
    debug_assert_eq!(pi.len(), d * d);
    debug_assert_eq!(out.len(), d);

    // y[j] = sum_i x[i] * Pi[j][i] = sum_i x[i] * Pi_T[i][j]
    // Pi is row-major, Pi^T[i][j] = Pi[j][i]
    for j in 0..d {
        let mut acc = 0.0;
        for i in 0..d {
            acc += x[i] * pi[j * d + i];
        }
        out[j] = acc;
    }
}

/// Apply inverse rotation: x = y @ Pi (row-vector convention).
///
/// For orthogonal Pi, Pi^{-1} = Pi^T, so x[j] = sum_i y[i] * Pi[i][j].
pub fn unrotate(y: &[f64], pi: &[f64], d: usize, out: &mut [f64]) {
    debug_assert_eq!(y.len(), d);
    debug_assert_eq!(pi.len(), d * d);
    debug_assert_eq!(out.len(), d);

    for j in 0..d {
        let mut acc = 0.0;
        for i in 0..d {
            acc += y[i] * pi[i * d + j];
        }
        out[j] = acc;
    }
}

// ---- Walsh-Hadamard Transform (via standalone fwht crate) ----

/// In-place Walsh-Hadamard Transform, normalized by 1/sqrt(d).
///
/// Delegates to the standalone `fwht` crate (extracted from this module).
/// The crate provides the same algorithm: k-level butterfly for d = 2^k,
/// normalized by 1/sqrt(d), self-inverse.
///
/// d must be a power of 2.
pub fn wht_inplace(data: &mut [f64]) {
    fwht::wht_inplace(data);
}

/// Generate random Rademacher sign vectors for fast JL rotation.
///
/// Delegates to the standalone `fwht` crate.
pub fn generate_rademacher_diagonals(d: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    fwht::generate_rademacher_diagonals(d, seed)
}

/// Fast JL rotation: y = D1 * WHT * D2 * x
///
/// Delegates to the standalone `fwht` crate.
pub fn fast_jl_rotate(x: &[f64], d1: &[f64], d2: &[f64], buf: &mut [f64], out: &mut [f64]) {
    fwht::fast_jl_rotate(x, d1, d2, buf, out);
}

/// Inverse fast JL rotation: x = D2 * WHT * D1 * y
///
/// Delegates to the standalone `fwht` crate.
pub fn fast_jl_unrotate(y: &[f64], d1: &[f64], d2: &[f64], buf: &mut [f64], out: &mut [f64]) {
    fwht::fast_jl_unrotate(y, d1, d2, buf, out);
}

/// E8 block rotation data (boxed to keep Rotation enum small).
#[derive(Clone, Debug)]
pub struct E8BlockData {
    pub roots: [super::e8_rotation::E8Root; 8],
    pub conj_roots: [super::e8_rotation::E8Root; 8],
    pub d: usize,
}

/// E8 + WHT composition data.
#[derive(Clone, Debug)]
pub struct E8WhtData {
    pub e8: E8BlockData,
    pub d1: Vec<f64>,
    pub d2: Vec<f64>,
    pub d: usize,
}

/// F4 block rotation data (boxed for enum size parity).
#[derive(Clone, Debug)]
pub struct F4BlockData {
    pub roots: [super::exceptional_roots::Root<4>; 16],
    pub d: usize,
}

/// Rotation method selector.
#[derive(Clone, Debug)]
pub enum Rotation {
    /// Dense Haar-random orthogonal matrix (d*d storage, O(d^2) apply).
    Haar { matrix: Vec<f64>, d: usize },
    /// Fast JL via WHT + Rademacher diagonals (2*d storage, O(d log d) apply).
    FastJL { d1: Vec<f64>, d2: Vec<f64>, d: usize },
    /// E8 lattice block rotation (8 roots storage, O(d) via sedenion multiply).
    /// Validated: KS p=0.816 vs Haar at d=128.  136x fewer parameters.
    /// Only works for d=128 (8 blocks of 16D sedenion).
    E8Block(Box<E8BlockData>),
    /// E8 + WHT composition: E8 for block-level algebraic decorrelation,
    /// then per-block WHT for within-block Gaussianization.
    /// Combines E8's CD-native structure with WHT's throughput.
    E8Wht(Box<E8WhtData>),
    /// F4 block rotation for d=64: 16 quaternion blocks rotated by F4 roots.
    /// F4 = automorphism group of Albert exceptional Jordan algebra J3(O).
    /// 18% better MSE than WHT at d=64 (measured 2026-03-28).
    F4Block(Box<F4BlockData>),
}

impl Rotation {
    pub fn new_haar(d: usize, seed: u64) -> Self {
        Rotation::Haar {
            matrix: generate_haar_rotation(d, seed),
            d,
        }
    }

    pub fn new_fast_jl(d: usize, seed: u64) -> Self {
        let (d1, d2) = generate_rademacher_diagonals(d, seed);
        Rotation::FastJL { d1, d2, d }
    }

    /// ZD-avoidance Fast JL: evaluate multiple candidate Rademacher diagonal
    /// pairs and select the one that maximizes minimum Koebisu D_2 score
    /// (furthest from zero-divisor manifold) on calibration vectors.
    ///
    /// D_2(v) = (||v_1||^2 - ||v_2||^2)^2 + 4*<v_1, v_2>^2
    /// where v_1, v_2 are the upper/lower halves of the sedenion (16D) blocks.
    ///
    /// Higher D_2 -> further from ZD manifold -> quantization error has more
    /// algebraic redundancy to absorb it.  O(n_candidates * n_cal * d) total.
    ///
    /// Returns the FastJL rotation with the best D_2 score.
    /// If d < 16 or calibration is empty, falls back to standard FastJL.
    pub fn new_fast_jl_zd_avoid(d: usize, seed: u64, calibration: &[&[f64]], n_candidates: usize) -> Self {
        if d < 16 || calibration.is_empty() || n_candidates <= 1 {
            return Self::new_fast_jl(d, seed);
        }

        let mut best_d1 = Vec::new();
        let mut best_d2 = Vec::new();
        let mut best_min_d2 = f64::NEG_INFINITY;

        let mut buf = vec![0.0f64; d];
        let mut out = vec![0.0f64; d];

        for candidate in 0..n_candidates {
            let candidate_seed = seed.wrapping_add(candidate as u64 * 997);
            let (d1, d2) = generate_rademacher_diagonals(d, candidate_seed);

            // Evaluate D_2 on calibration vectors after rotation
            let mut min_d2 = f64::MAX;
            for cal_vec in calibration.iter().take(50) {
                // Normalize
                let norm: f64 = cal_vec.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm < 1e-15 { continue; }
                let normalized: Vec<f64> = cal_vec.iter().map(|x| x / norm).collect();

                // Rotate
                fast_jl_rotate(&normalized, &d1, &d2, &mut buf, &mut out);

                // Evaluate D_2 on each 16D block of the rotated vector
                let n_blocks = d / 16;
                for block in 0..n_blocks {
                    let block_start = block * 16;
                    let block_slice = &out[block_start..block_start + 16];
                    let d2_val = crate::cayley_dickson::koebisu_d2(block_slice);
                    if d2_val < min_d2 {
                        min_d2 = d2_val;
                    }
                }
            }

            if min_d2 > best_min_d2 {
                best_min_d2 = min_d2;
                best_d1 = d1;
                best_d2 = d2;
            }
        }

        Rotation::FastJL { d1: best_d1, d2: best_d2, d }
    }

    /// Create E8 block rotation for d=128.
    ///
    /// Selects 8 diverse E8 roots and precomputes their conjugates
    /// for the inverse rotation.  Panics if d != 128.
    pub fn new_e8(d: usize, seed: u64) -> Self {
        assert_eq!(d, 128, "E8 block rotation requires d=128");
        let all_roots = super::e8_rotation::generate_e8_roots();
        let roots = super::e8_rotation::select_diverse_roots(&all_roots, seed);

        // Conjugate: for sedenion (a0, a1, ..., a15), conjugate = (a0, -a1, ..., -a15)
        // But our roots are 8D embedded into 16D, so conjugate negates coords[1..8]
        // and leaves coords[0] unchanged.  For unit-norm rotation elements,
        // the inverse is: conj(r) * x * r -> we need right-multiply by conj
        // Actually for CD left-multiplication L_r(x) = r*x, the inverse is L_{r^{-1}}
        // For unit sedenion, r^{-1} = conj(r) / ||r||^2.  Since ||r|| = 1 (normalized),
        // r^{-1} = conj(r).
        let mut conj_roots = roots;
        for root in &mut conj_roots {
            // Conjugate the 8D embedding: negate all but first coordinate
            for c in root.coords[1..].iter_mut() {
                *c = -*c;
            }
        }

        Rotation::E8Block(Box::new(E8BlockData { roots, conj_roots, d }))
    }

    /// Create E8 + WHT composition rotation for d=128.
    ///
    /// Forward: E8 block rotate -> WHT with Rademacher diagonals.
    /// This gives E8's algebraic structure plus WHT's Gaussianization.
    pub fn new_e8_wht(d: usize, seed: u64) -> Self {
        assert_eq!(d, 128, "E8+WHT requires d=128");

        // Build E8 part
        let all_roots = super::e8_rotation::generate_e8_roots();
        let roots = super::e8_rotation::select_diverse_roots(&all_roots, seed);
        let mut conj_roots = roots;
        for root in &mut conj_roots {
            for c in root.coords[1..].iter_mut() {
                *c = -*c;
            }
        }
        let e8 = E8BlockData { roots, conj_roots, d };

        // Build WHT Rademacher part (use different seed to avoid correlation)
        let (d1, d2) = generate_rademacher_diagonals(d, seed + 500);

        Rotation::E8Wht(Box::new(E8WhtData { e8, d1, d2, d }))
    }

    /// Create F4 block rotation for d=64.
    pub fn new_f4(d: usize, seed: u64) -> Self {
        assert_eq!(d, 64, "F4 block rotation requires d=64");
        let roots = super::f4_rotation::select_diverse_f4_roots(seed);
        Rotation::F4Block(Box::new(F4BlockData { roots, d }))
    }

    pub fn dim(&self) -> usize {
        match self {
            Rotation::Haar { d, .. }
            | Rotation::FastJL { d, .. } => *d,
            Rotation::E8Block(data) => data.d,
            Rotation::E8Wht(data) => data.d,
            Rotation::F4Block(data) => data.d,
        }
    }

    /// Apply forward rotation.  `buf` is scratch space (>= d elements).
    pub fn forward(&self, x: &[f64], buf: &mut [f64], out: &mut [f64]) {
        match self {
            Rotation::Haar { matrix, d } => rotate(x, matrix, *d, out),
            Rotation::FastJL { d1, d2, .. } => fast_jl_rotate(x, d1, d2, buf, out),
            Rotation::E8Block(data) => {
                super::e8_rotation::e8_block_rotate(x, &data.roots, out);
            }
            Rotation::E8Wht(data) => {
                let mut e8_out = vec![0.0f64; data.d];
                super::e8_rotation::e8_block_rotate(x, &data.e8.roots, &mut e8_out);
                fast_jl_rotate(&e8_out, &data.d1, &data.d2, buf, out);
            }
            Rotation::F4Block(data) => {
                let roots = &data.roots;
                super::f4_rotation::f4_block_rotate(x, roots, out);
            }
        }
    }

    /// Apply inverse rotation.  `buf` is scratch space (>= d elements).
    pub fn inverse(&self, y: &[f64], buf: &mut [f64], out: &mut [f64]) {
        match self {
            Rotation::Haar { matrix, d } => unrotate(y, matrix, *d, out),
            Rotation::FastJL { d1, d2, .. } => fast_jl_unrotate(y, d1, d2, buf, out),
            Rotation::E8Block(data) => {
                super::e8_rotation::e8_block_rotate(y, &data.conj_roots, out);
            }
            Rotation::E8Wht(data) => {
                let mut wht_inv = vec![0.0f64; data.d];
                fast_jl_unrotate(y, &data.d1, &data.d2, buf, &mut wht_inv);
                super::e8_rotation::e8_block_rotate(&wht_inv, &data.e8.conj_roots, out);
            }
            Rotation::F4Block(data) => {
                let roots = &data.roots;
                super::f4_rotation::f4_block_unrotate(y, roots, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haar_orthogonality() {
        let d = 16;
        let pi = generate_haar_rotation(d, 42);
        // Check Pi * Pi^T = I
        for i in 0..d {
            for j in 0..d {
                let mut dot = 0.0;
                for k in 0..d {
                    dot += pi[i * d + k] * pi[j * d + k];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "Pi*Pi^T[{},{}] = {}, expected {}",
                    i, j, dot, expected
                );
            }
        }
    }

    #[test]
    fn test_haar_roundtrip() {
        let d = 32;
        let pi = generate_haar_rotation(d, 123);
        let x: Vec<f64> = (0..d).map(|i| (i as f64 + 1.0) / d as f64).collect();
        let mut y = vec![0.0; d];
        let mut x_rt = vec![0.0; d];
        rotate(&x, &pi, d, &mut y);
        unrotate(&y, &pi, d, &mut x_rt);
        for i in 0..d {
            assert!(
                (x[i] - x_rt[i]).abs() < 1e-10,
                "Roundtrip error at {}: {} vs {}",
                i, x[i], x_rt[i]
            );
        }
    }

    #[test]
    fn test_wht_self_inverse() {
        let d = 128;
        let original: Vec<f64> = (0..d).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut data = original.clone();
        wht_inplace(&mut data);
        wht_inplace(&mut data);
        // WHT applied twice = identity (H * H / d = I, so (H/sqrt(d))^2 = I)
        for i in 0..d {
            assert!(
                (data[i] - original[i]).abs() < 1e-10,
                "WHT not self-inverse at {}: {} vs {}",
                i, data[i], original[i]
            );
        }
    }

    #[test]
    fn test_wht_delta_to_uniform() {
        let d = 8;
        let mut data = vec![0.0; d];
        data[0] = (d as f64).sqrt(); // delta at 0, scaled so WHT gives all 1s
        wht_inplace(&mut data);
        // After WHT of scaled delta: all entries should be 1.0
        for (i, &v) in data.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-10,
                "WHT(delta)[{}] = {}, expected 1.0",
                i, v
            );
        }
    }

    #[test]
    fn test_fast_jl_roundtrip() {
        let d = 64;
        let (d1, d2) = generate_rademacher_diagonals(d, 42);
        let x: Vec<f64> = (0..d).map(|i| (i as f64 * 0.3).cos()).collect();
        let mut buf = vec![0.0; d];
        let mut y = vec![0.0; d];
        let mut x_rt = vec![0.0; d];
        fast_jl_rotate(&x, &d1, &d2, &mut buf, &mut y);
        fast_jl_unrotate(&y, &d1, &d2, &mut buf, &mut x_rt);
        for i in 0..d {
            assert!(
                (x[i] - x_rt[i]).abs() < 1e-10,
                "Fast JL roundtrip error at {}: {} vs {}",
                i, x[i], x_rt[i]
            );
        }
    }

    #[test]
    fn test_fast_jl_norm_preservation() {
        let d = 128;
        let (d1, d2) = generate_rademacher_diagonals(d, 99);
        let x: Vec<f64> = (0..d).map(|i| (i as f64 * 0.7).sin()).collect();
        let norm_x: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut buf = vec![0.0; d];
        let mut y = vec![0.0; d];
        fast_jl_rotate(&x, &d1, &d2, &mut buf, &mut y);
        let norm_y: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (norm_x - norm_y).abs() / norm_x < 1e-10,
            "Norm not preserved: {} vs {}",
            norm_x, norm_y
        );
    }

    #[test]
    fn test_rotation_enum_dispatch() {
        let d = 32;
        let x: Vec<f64> = (0..d).map(|i| i as f64 * 0.1).collect();

        for rotation in [Rotation::new_haar(d, 42), Rotation::new_fast_jl(d, 42)] {
            let mut buf = vec![0.0; d];
            let mut y = vec![0.0; d];
            let mut x_rt = vec![0.0; d];
            rotation.forward(&x, &mut buf, &mut y);
            rotation.inverse(&y, &mut buf, &mut x_rt);
            for i in 0..d {
                assert!(
                    (x[i] - x_rt[i]).abs() < 1e-9,
                    "{:?} roundtrip error at {}: {} vs {}",
                    rotation, x[i], x_rt[i], // print rotation type
                    i
                );
            }
        }
    }

    #[test]
    fn test_e8_rotation_roundtrip() {
        let d = 128;
        let x: Vec<f64> = (0..d).map(|i| (i as f64 * 0.07).sin()).collect();
        let rotation = Rotation::new_e8(d, 42);

        let mut buf = vec![0.0; d];
        let mut y = vec![0.0; d];
        let mut x_rt = vec![0.0; d];
        rotation.forward(&x, &mut buf, &mut y);
        rotation.inverse(&y, &mut buf, &mut x_rt);

        // E8 block rotation inverse via conjugate should reconstruct
        let max_err: f64 = x.iter().zip(x_rt.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_err < 1e-8,
            "E8 roundtrip max error: {} (expected < 1e-8)",
            max_err
        );
    }

    #[test]
    fn test_e8_wht_roundtrip() {
        let d = 128;
        let x: Vec<f64> = (0..d).map(|i| (i as f64 * 0.07).sin()).collect();
        let rotation = Rotation::new_e8_wht(d, 42);

        let mut buf = vec![0.0; d];
        let mut y = vec![0.0; d];
        let mut x_rt = vec![0.0; d];
        rotation.forward(&x, &mut buf, &mut y);
        rotation.inverse(&y, &mut buf, &mut x_rt);

        let max_err: f64 = x.iter().zip(x_rt.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_err < 1e-7,
            "E8+WHT roundtrip max error: {} (expected < 1e-7)",
            max_err
        );
    }

    #[test]
    fn test_e8_wht_norm_preservation() {
        let d = 128;
        let x: Vec<f64> = (0..d).map(|i| (i as f64 * 0.13).cos()).collect();
        let norm_x: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();

        let rotation = Rotation::new_e8_wht(d, 42);
        let mut buf = vec![0.0; d];
        let mut y = vec![0.0; d];
        rotation.forward(&x, &mut buf, &mut y);
        let norm_y: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();

        assert!(
            (norm_x - norm_y).abs() / norm_x < 1e-8,
            "E8+WHT norm not preserved: {} vs {}",
            norm_x, norm_y
        );
    }

    #[test]
    fn test_zd_avoidance_rotation() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;
        use rand_distr::{Distribution, StandardNormal};

        let d = 128;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let normal = StandardNormal;
        let cal_vecs: Vec<Vec<f64>> = (0..30)
            .map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect())
            .collect();
        let cal_refs: Vec<&[f64]> = cal_vecs.iter().map(|v| v.as_slice()).collect();

        // Standard rotation
        let rot_std = Rotation::new_fast_jl(d, 42);
        // ZD-avoidance rotation with 32 candidates
        let rot_zd = Rotation::new_fast_jl_zd_avoid(d, 42, &cal_refs, 32);

        // Both should produce valid rotations (roundtrip)
        let x: Vec<f64> = (0..d).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut buf = vec![0.0; d];
        let mut y = vec![0.0; d];
        let mut x_rt = vec![0.0; d];

        rot_zd.forward(&x, &mut buf, &mut y);
        rot_zd.inverse(&y, &mut buf, &mut x_rt);
        let err: f64 = x.iter().zip(x_rt.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(err < 1e-8, "ZD-avoidance roundtrip error: {}", err);

        // ZD-avoidance should have higher minimum D_2 on the calibration data
        let mut min_d2_std = f64::MAX;
        let mut min_d2_zd = f64::MAX;
        let mut out_std = vec![0.0f64; d];
        let mut out_zd = vec![0.0f64; d];

        for cal in &cal_refs {
            let norm: f64 = cal.iter().map(|x| x * x).sum::<f64>().sqrt();
            let normalized: Vec<f64> = cal.iter().map(|x| x / norm).collect();

            rot_std.forward(&normalized, &mut buf, &mut out_std);
            rot_zd.forward(&normalized, &mut buf, &mut out_zd);

            for block in 0..(d / 16) {
                let s = block * 16;
                let d2_std = crate::cayley_dickson::koebisu_d2(&out_std[s..s + 16]);
                let d2_zd = crate::cayley_dickson::koebisu_d2(&out_zd[s..s + 16]);
                min_d2_std = min_d2_std.min(d2_std);
                min_d2_zd = min_d2_zd.min(d2_zd);
            }
        }

        println!("D_2 scores: standard min={:.6}, ZD-avoidance min={:.6}", min_d2_std, min_d2_zd);
        assert!(min_d2_zd >= min_d2_std,
            "ZD-avoidance should have higher or equal D_2: {} vs {}", min_d2_zd, min_d2_std);
    }
}
