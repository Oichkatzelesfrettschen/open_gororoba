//! F4-block rotation for d=64: quaternion-level algebraic decorrelation.
//!
//! F4 is the automorphism group of the Albert exceptional Jordan algebra
//! J3(O) -- the 27-dimensional algebra of 3x3 Hermitian octonionic matrices.
//! Its 48 roots live in R^4, corresponding to quaternion rotations.
//!
//! At d=64 = 16 blocks of 4D, each block is a quaternion that can be
//! rotated by left-multiplication with a unit quaternion derived from
//! an F4 root.  This gives algebraically structured decorrelation at
//! the quaternion level, analogous to E8's sedenion-level structure.
//!
//! # Connection to CD tower
//!
//! - F4 roots (R^4) -> quaternion (level 2) blocks
//! - E8 roots (R^8) -> sedenion (level 4) blocks via octonion embedding
//!
//! The F4 rotation preserves all properties that quaternion multiplication
//! preserves: norm, associativity within the block.

use super::exceptional_roots::{generate_f4_roots, Root};

/// Select diverse F4 roots for d=64 (16 quaternion blocks).
///
/// Greedy max-angular-distance selection, same algorithm as E8.
pub fn select_diverse_f4_roots(seed: u64) -> [Root<4>; 16] {
    let all_roots = generate_f4_roots();
    let n = all_roots.len();
    let start_idx = (seed as usize) % n;
    let mut selected_indices = vec![start_idx];

    for _ in 1..16 {
        let mut best_idx = 0;
        let mut best_min_dist = f64::MIN;

        for (i, root) in all_roots.iter().enumerate() {
            if selected_indices.contains(&i) {
                continue;
            }
            let min_dist = selected_indices
                .iter()
                .map(|&si| 1.0 - root.dot(&all_roots[si]).abs() / (root.norm_sq().sqrt() * all_roots[si].norm_sq().sqrt()))
                .fold(f64::MAX, f64::min);

            if min_dist > best_min_dist {
                best_min_dist = min_dist;
                best_idx = i;
            }
        }
        selected_indices.push(best_idx);
    }

    let mut result = [Root { coords: [0.0; 4] }; 16];
    for (i, &idx) in selected_indices.iter().enumerate() {
        result[i] = all_roots[idx];
    }
    result
}

/// Apply F4-block rotation to a 64D vector.
///
/// Decomposes into 16 blocks of 4D quaternion.  Each block is rotated
/// by left-multiplication with a unit quaternion derived from the F4 root.
///
/// Quaternion multiplication: (a0, a1, a2, a3) * (b0, b1, b2, b3) =
///   (a0*b0 - a1*b1 - a2*b2 - a3*b3,
///    a0*b1 + a1*b0 + a2*b3 - a3*b2,
///    a0*b2 - a1*b3 + a2*b0 + a3*b1,
///    a0*b3 + a1*b2 - a2*b1 + a3*b0)
pub fn f4_block_rotate(v: &[f64], roots: &[Root<4>; 16], out: &mut [f64]) {
    assert_eq!(v.len(), 64);
    assert_eq!(out.len(), 64);

    for (b, root) in roots.iter().enumerate() {
        let offset = b * 4;

        // Extract block
        let b0 = v[offset];
        let b1 = v[offset + 1];
        let b2 = v[offset + 2];
        let b3 = v[offset + 3];

        // Normalize root to unit quaternion
        let norm = root.norm_sq().sqrt();
        let (a0, a1, a2, a3) = if norm > 1e-15 {
            let inv = 1.0 / norm;
            (root.coords[0] * inv, root.coords[1] * inv,
             root.coords[2] * inv, root.coords[3] * inv)
        } else {
            (1.0, 0.0, 0.0, 0.0) // identity quaternion
        };

        // Quaternion left-multiplication
        out[offset]     = a0*b0 - a1*b1 - a2*b2 - a3*b3;
        out[offset + 1] = a0*b1 + a1*b0 + a2*b3 - a3*b2;
        out[offset + 2] = a0*b2 - a1*b3 + a2*b0 + a3*b1;
        out[offset + 3] = a0*b3 + a1*b2 - a2*b1 + a3*b0;
    }
}

/// Inverse F4-block rotation using conjugate quaternions.
///
/// For unit quaternion q, q^{-1} = q* = (q0, -q1, -q2, -q3).
pub fn f4_block_unrotate(v: &[f64], roots: &[Root<4>; 16], out: &mut [f64]) {
    assert_eq!(v.len(), 64);
    assert_eq!(out.len(), 64);

    for (b, root) in roots.iter().enumerate() {
        let offset = b * 4;

        let b0 = v[offset];
        let b1 = v[offset + 1];
        let b2 = v[offset + 2];
        let b3 = v[offset + 3];

        // Conjugate quaternion (negate imaginary parts)
        let norm = root.norm_sq().sqrt();
        let (a0, a1, a2, a3) = if norm > 1e-15 {
            let inv = 1.0 / norm;
            (root.coords[0] * inv, -root.coords[1] * inv,
             -root.coords[2] * inv, -root.coords[3] * inv)
        } else {
            (1.0, 0.0, 0.0, 0.0)
        };

        out[offset]     = a0*b0 - a1*b1 - a2*b2 - a3*b3;
        out[offset + 1] = a0*b1 + a1*b0 + a2*b3 - a3*b2;
        out[offset + 2] = a0*b2 - a1*b3 + a2*b0 + a3*b1;
        out[offset + 3] = a0*b3 + a1*b2 - a2*b1 + a3*b0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f4_roundtrip() {
        let roots = select_diverse_f4_roots(42);
        let v: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut rotated = vec![0.0f64; 64];
        let mut recovered = vec![0.0f64; 64];

        f4_block_rotate(&v, &roots, &mut rotated);
        f4_block_unrotate(&rotated, &roots, &mut recovered);

        let max_err: f64 = v.iter().zip(recovered.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        assert!(max_err < 1e-10, "F4 roundtrip error: {}", max_err);
    }

    #[test]
    fn test_f4_norm_preservation() {
        let roots = select_diverse_f4_roots(42);
        let v: Vec<f64> = (0..64).map(|i| (i as f64 * 0.13).cos()).collect();
        let norm_v: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();

        let mut rotated = vec![0.0f64; 64];
        f4_block_rotate(&v, &roots, &mut rotated);
        let norm_r: f64 = rotated.iter().map(|x| x * x).sum::<f64>().sqrt();

        assert!(
            (norm_v - norm_r).abs() / norm_v < 1e-10,
            "F4 norm not preserved: {} vs {}", norm_v, norm_r
        );
    }

    #[test]
    fn test_f4_changes_vector() {
        let roots = select_diverse_f4_roots(42);
        let v: Vec<f64> = (0..64).map(|i| (i as f64 * 0.3).cos()).collect();
        let mut rotated = vec![0.0f64; 64];
        f4_block_rotate(&v, &roots, &mut rotated);

        let diff: f64 = v.iter().zip(rotated.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6, "F4 rotation should change the vector");
    }

    #[test]
    fn test_f4_as_pipeline_rotation() {
        // Test F4 rotation through the TurboQuantMSE pipeline at d=64
        use crate::turboquant::pipeline::TurboQuantMSE;

        let d = 64;
        let bits = 3;
        // Use WHT (F4 rotation not yet wired into Rotation enum for pipeline,
        // but we can test the rotation quality directly)
        let tq = TurboQuantMSE::new(d, bits, 42, true);
        let v: Vec<f64> = (0..d).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut buf = vec![0.0f64; 2 * d];
        let comp = tq.quantize(&v, &mut buf);
        let mut recon = vec![0.0f64; d];
        tq.dequantize(&comp, &mut buf, &mut recon);

        let mse_wht: f64 = v.iter().zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;

        // Now test with F4 rotation manually
        let roots = select_diverse_f4_roots(42);
        let mut rotated = vec![0.0f64; d];
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let v_unit: Vec<f64> = v.iter().map(|x| x / norm).collect();
        f4_block_rotate(&v_unit, &roots, &mut rotated);

        // Quantize the rotated vector using the codebook
        let codebook = crate::lloyd_max::get_codebook(d, bits);
        let indices: Vec<u8> = rotated.iter().map(|&val| {
            let mut idx = 0u8;
            for &b in codebook.boundaries.iter() {
                if val > b as f64 { idx += 1; } else { break; }
            }
            idx
        }).collect();

        // Dequantize in rotated space
        let recon_rotated: Vec<f64> = indices.iter()
            .map(|&idx| codebook.centroids[idx as usize] as f64).collect();

        // Unrotate
        let mut recon_f4 = vec![0.0f64; d];
        f4_block_unrotate(&recon_rotated, &roots, &mut recon_f4);

        // Denormalize
        for x in recon_f4.iter_mut() { *x *= norm; }

        let mse_f4: f64 = v.iter().zip(recon_f4.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;

        println!("d=64, 3-bit: WHT MSE={:.6}, F4 MSE={:.6}, ratio={:.3}",
            mse_wht, mse_f4, mse_f4 / mse_wht);

        // F4 should be comparable to WHT (within 2x)
        assert!(mse_f4 < mse_wht * 2.0,
            "F4 MSE much worse than WHT: {} vs {}", mse_f4, mse_wht);
    }
}
