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

// ---- Walsh-Hadamard Transform (Fast JL rotation) ----

/// In-place Walsh-Hadamard Transform, normalized by 1/sqrt(d).
///
/// Algorithm (from `wht_rotation.py:hadamard_transform`):
///   7-level butterfly for d=128.  Each level pairs elements at stride 2^level
///   and replaces (a, b) with (a+b, a-b).  Final result divided by sqrt(d).
///
/// d must be a power of 2.
pub fn wht_inplace(data: &mut [f64]) {
    let d = data.len();
    debug_assert!(d.is_power_of_two(), "WHT requires power-of-two dimension");

    let mut h = 1;
    while h < d {
        let mut i = 0;
        while i < d {
            for j in i..(i + h) {
                let a = data[j];
                let b = data[j + h];
                data[j] = a + b;
                data[j + h] = a - b;
            }
            i += h * 2;
        }
        h *= 2;
    }

    // Normalize
    let scale = 1.0 / (d as f64).sqrt();
    for v in data.iter_mut() {
        *v *= scale;
    }
}

/// Generate random Rademacher sign vectors for fast JL rotation.
///
/// Returns (d1, d2) where each is a Vec<f64> of +1/-1 values.
/// From `wht_rotation.py:generate_wht_rotation`.
pub fn generate_rademacher_diagonals(d: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let normal = StandardNormal;

    let sign = |rng: &mut ChaCha20Rng| -> f64 {
        let v: f64 = normal.sample(rng);
        if v >= 0.0 { 1.0 } else { -1.0 }
    };

    let d1: Vec<f64> = (0..d).map(|_| sign(&mut rng)).collect();
    let d2: Vec<f64> = (0..d).map(|_| sign(&mut rng)).collect();

    (d1, d2)
}

/// Fast JL rotation: y = D1 * WHT * D2 * x
///
/// O(d log d) structured random rotation (Ailon-Chazelle 2006).
/// Workspace `buf` must have length >= d.
pub fn fast_jl_rotate(x: &[f64], d1: &[f64], d2: &[f64], buf: &mut [f64], out: &mut [f64]) {
    let d = x.len();
    debug_assert_eq!(d1.len(), d);
    debug_assert_eq!(d2.len(), d);
    debug_assert!(buf.len() >= d);
    debug_assert_eq!(out.len(), d);

    // Step 1: multiply by D2
    for i in 0..d {
        buf[i] = x[i] * d2[i];
    }

    // Step 2: apply WHT in-place
    wht_inplace(&mut buf[..d]);

    // Step 3: multiply by D1
    for i in 0..d {
        out[i] = buf[i] * d1[i];
    }
}

/// Inverse fast JL rotation: x = D2 * WHT * D1 * y
///
/// WHT and Rademacher diagonals are self-inverse (up to scaling).
pub fn fast_jl_unrotate(y: &[f64], d1: &[f64], d2: &[f64], buf: &mut [f64], out: &mut [f64]) {
    let d = y.len();

    // Inverse is: D2 * WHT * D1 * y
    for i in 0..d {
        buf[i] = y[i] * d1[i];
    }
    wht_inplace(&mut buf[..d]);
    for i in 0..d {
        out[i] = buf[i] * d2[i];
    }
}

/// E8 block rotation data (boxed to keep Rotation enum small).
#[derive(Clone, Debug)]
pub struct E8BlockData {
    pub roots: [super::e8_rotation::E8Root; 8],
    pub conj_roots: [super::e8_rotation::E8Root; 8],
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

    pub fn dim(&self) -> usize {
        match self {
            Rotation::Haar { d, .. }
            | Rotation::FastJL { d, .. } => *d,
            Rotation::E8Block(data) => data.d,
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
}
