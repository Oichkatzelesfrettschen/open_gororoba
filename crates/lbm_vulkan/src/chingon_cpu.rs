//! CPU reference for the Chingon AVT bilinear tensor contraction.
//!
//! WHY: The Vulkan, CUDA, and cubecl chingon paths need a deterministic
//! oracle. The contraction is:
//!
//!   `force[m] += alpha * v[i] * v[j] * sign * inv_n_viol`
//!
//! for each bit-packed violation `(i, j, m, sign)`. The CPU reference
//! evaluates the contraction exactly in serial f64 (then narrowed to
//! f32 for the GPU-output comparison) so we can validate any GPU path
//! at < 1 ULP per partial-sum step.
//!
//! HOW: Mirrors the algorithm of `shaders/chingon_avt.wgsl` line-for-line
//! except for the atomicAdd -- the CPU sums in `f64` accumulators per
//! `m` slot, then casts back to `f32` at the end. This minimises the
//! numerical drift between the reference and any GPU implementation
//! that uses 32-bit accumulators.

#[cfg(test)]
use crate::chingon_vulkan::pack_violation;

/// Per-axis state for the chingon contraction.
#[derive(Debug, Clone)]
pub struct ChingonInputs {
    /// State vector v in R^dim. Length must equal `dim`.
    pub v_nd: Vec<f32>,
    /// Bit-packed violation triples (i, j, m, sign_positive) using
    /// `pack_violation()` with the agreed `index_bits` for the chosen `dim`.
    pub packed_avt: Vec<u32>,
    /// Coupling constant in front of every contribution.
    pub alpha: f32,
    /// Per-violation scale factor (typically `1 / n_violations`).
    pub inv_n_viol: f32,
    /// Total number of bits used to encode each index. Must equal
    /// `ceil(log2(dim))` so that the 4 fields fit in a u32.
    pub index_bits: u32,
    /// Dimension of v (and of the output force vector).
    pub dim: u32,
}

/// Compute the chingon AVT contraction on CPU. Returns a fresh
/// `force` vector of length `dim` with the contraction result.
///
/// # Algorithm
///
/// For each violation `v` in `packed_avt`:
///   1. unpack `(m, j, i, sign_positive) = packed[v]`
///   2. `sign_val = +2.0` if `sign_positive` else `-2.0`
///   3. `contrib = alpha * v[i] * v[j] * sign_val * inv_n_viol`
///   4. `force[m] += contrib`
///
/// # Panics
/// Panics if any unpacked index is out of `[0, dim)` (the GPU shaders
/// trust the host to respect this bound, so the CPU reference enforces
/// it for matching error semantics).
pub fn chingon_contract_cpu(inputs: &ChingonInputs) -> Vec<f32> {
    assert!(inputs.dim as usize == inputs.v_nd.len());
    let dim = inputs.dim as usize;
    let mask = (1u32 << inputs.index_bits) - 1;
    let mut accum = vec![0.0_f64; dim];
    for &packed in inputs.packed_avt.iter() {
        let m_idx = (packed & mask) as usize;
        let j_idx = ((packed >> inputs.index_bits) & mask) as usize;
        let i_idx = ((packed >> (2 * inputs.index_bits)) & mask) as usize;
        let sign_positive = (packed >> (3 * inputs.index_bits)) & 1 == 1;
        let sign_val: f64 = if sign_positive { 2.0 } else { -2.0 };

        let vi = inputs.v_nd[i_idx] as f64;
        let vj = inputs.v_nd[j_idx] as f64;
        let alpha = inputs.alpha as f64;
        let inv_n = inputs.inv_n_viol as f64;
        accum[m_idx] += alpha * vi * vj * sign_val * inv_n;
    }
    accum.into_iter().map(|x| x as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial: zero violations leaves the force vector at zero.
    #[test]
    fn empty_violations_gives_zero_force() {
        let inputs = ChingonInputs {
            v_nd: vec![1.0; 8],
            packed_avt: vec![],
            alpha: 1.0,
            inv_n_viol: 1.0,
            index_bits: 3,
            dim: 8,
        };
        let force = chingon_contract_cpu(&inputs);
        assert_eq!(force, vec![0.0; 8]);
    }

    /// Single +violation with i==j==m==0 yields force[0] = alpha * v[0]^2 * 2.
    #[test]
    fn single_diagonal_positive_violation() {
        let inputs = ChingonInputs {
            v_nd: vec![3.0, 0.0, 0.0, 0.0],
            packed_avt: vec![pack_violation(0, 0, 0, true, 2)],
            alpha: 0.5,
            inv_n_viol: 1.0,
            index_bits: 2,
            dim: 4,
        };
        let force = chingon_contract_cpu(&inputs);
        // 0.5 * 3 * 3 * 2 * 1 = 9.0
        assert_eq!(force[0], 9.0);
        for f in force.iter().take(4).skip(1) {
            assert_eq!(*f, 0.0);
        }
    }

    /// Single -violation gives the negative.
    #[test]
    fn single_diagonal_negative_violation() {
        let inputs = ChingonInputs {
            v_nd: vec![3.0, 0.0, 0.0, 0.0],
            packed_avt: vec![pack_violation(0, 0, 0, false, 2)],
            alpha: 0.5,
            inv_n_viol: 1.0,
            index_bits: 2,
            dim: 4,
        };
        let force = chingon_contract_cpu(&inputs);
        // 0.5 * 3 * 3 * (-2) * 1 = -9.0
        assert_eq!(force[0], -9.0);
    }

    /// inv_n_viol scales every contribution uniformly.
    #[test]
    fn inv_n_viol_scales_uniformly() {
        let inputs = ChingonInputs {
            v_nd: vec![1.0, 1.0, 1.0, 1.0],
            packed_avt: vec![
                pack_violation(0, 1, 2, true, 2),
                pack_violation(1, 0, 2, true, 2),
            ],
            alpha: 1.0,
            inv_n_viol: 0.5,
            index_bits: 2,
            dim: 4,
        };
        let force = chingon_contract_cpu(&inputs);
        // Both contributions go to force[2]:
        //   1 * 1 * 1 * 2 * 0.5 = 1.0 each -> 2.0 total
        assert_eq!(force[2], 2.0);
    }
}
