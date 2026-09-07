//! Normalized Walsh-Hadamard transforms and structured random rotations.
//!
//! # Quick Start
//!
//! ```
//! use fwht::{wht_inplace, fast_jl_rotate};
//!
//! // Basic WHT (in-place, O(d log d))
//! let mut data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
//! wht_inplace(&mut data);
//! // The transformed delta function has entries 1/sqrt(8).
//!
//! // Structured random rotation
//! let x = vec![1.0, 2.0, 3.0, 4.0];
//! let (d1, d2) = fwht::generate_rademacher_diagonals(4, 42);
//! let mut buf = vec![0.0; 4];
//! let mut out = vec![0.0; 4];
//! fast_jl_rotate(&x, &d1, &d2, &mut buf, &mut out);
//! // out contains the structured random rotation of x.
//! ```
//!
//! # Algorithm
//!
//! The WHT at dimension d = 2^k is a k-level butterfly network:
//! ```text
//! Level 0: pairs at stride 1   (a,b) -> (a+b, a-b)
//! Level 1: pairs at stride 2   (a,b) -> (a+b, a-b)
//! ...
//! Level k-1: pairs at stride 2^(k-1)
//! ```
//! Total: d * k / 2 butterflies = O(d log d).
//! Normalized by 1/sqrt(d), the transform is self-inverse: WHT(WHT(x)) = x.
//!
//! The fast JL rotation composes: `y = D1 * WHT * D2 * x`
//! where D1, D2 are random Rademacher diagonal matrices (+/-1 entries).
//! The composition preserves dimension and Euclidean norm in exact arithmetic.
//! A dimension-reducing projection and its Johnson-Lindenstrauss bound are
//! separate from the rotation implemented here.
//!
//! # References
//!
//! - Ailon & Chazelle, "Approximate nearest neighbors and the fast
//!   Johnson-Lindenstrauss transform," STOC 2006.

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

/// In-place Walsh-Hadamard Transform, normalized by 1/sqrt(d).
///
/// `data` must have power-of-two length. The transform is self-inverse:
/// applying it twice returns the original data.
///
/// # Panics
/// Panics if `data.len()` is not a power of 2.
///
/// # Example
/// ```
/// let mut data = vec![1.0, 0.0, 0.0, 0.0];
/// fwht::wht_inplace(&mut data);
/// assert!((data[0] - 0.5).abs() < 1e-10); // 1/sqrt(4) = 0.5
/// ```
pub fn wht_inplace(data: &mut [f64]) {
    let d = data.len();
    assert!(
        d.is_power_of_two(),
        "WHT requires power-of-two dimension, got {}",
        d
    );

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

    let scale = 1.0 / (d as f64).sqrt();
    for v in data.iter_mut() {
        *v *= scale;
    }
}

/// Generate random Rademacher sign vectors for fast JL rotation.
///
/// Returns (d1, d2) where each entry is +1.0 or -1.0.
/// The seed determines the random sequence (deterministic for reproducibility).
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
/// O(d log d) structured random rotation with unchanged dimension.
///
/// # Parameters
/// - `x`: input vector (length d, must be power of 2)
/// - `d1`, `d2`: Rademacher sign diagonals (from `generate_rademacher_diagonals`)
/// - `buf`: scratch workspace (length >= d)
/// - `out`: output vector (length d)
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
/// WHT and Rademacher diagonals are self-inverse.
pub fn fast_jl_unrotate(y: &[f64], d1: &[f64], d2: &[f64], buf: &mut [f64], out: &mut [f64]) {
    let d = y.len();
    for i in 0..d {
        buf[i] = y[i] * d1[i];
    }
    wht_inplace(&mut buf[..d]);
    for i in 0..d {
        out[i] = buf[i] * d2[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wht_self_inverse() {
        let original: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut data = original.clone();
        wht_inplace(&mut data);
        wht_inplace(&mut data);
        for (i, (&a, &b)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "Not self-inverse at {}: {} vs {}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_wht_delta_to_uniform() {
        let d = 8;
        let mut data = vec![0.0; d];
        data[0] = (d as f64).sqrt();
        wht_inplace(&mut data);
        for (i, &v) in data.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-10,
                "WHT(delta)[{}] = {}, expected 1.0",
                i,
                v
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
            assert!((x[i] - x_rt[i]).abs() < 1e-10, "Roundtrip error at {}", i);
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
            "Norm not preserved"
        );
    }

    #[test]
    fn test_various_dimensions() {
        for k in 1..=10 {
            let d = 1 << k; // 2, 4, 8, ..., 1024
            let mut data: Vec<f64> = (0..d).map(|i| i as f64).collect();
            let original = data.clone();
            wht_inplace(&mut data);
            wht_inplace(&mut data);
            for (i, (&a, &b)) in original.iter().zip(data.iter()).enumerate() {
                assert!((a - b).abs() < 1e-8, "d={}, idx={}: {} vs {}", d, i, a, b);
            }
        }
    }
}
