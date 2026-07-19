//! Equal-receptive-field controls for the staple-associator detector.
//!
//! The staple associator at position k consumes rows k..k+5 (three
//! overlapping 4-lag staples) and is, by the coefficient census
//! (C-1630), the norm of a fixed 1848-term sparse cubic filter with
//! coefficients in {-2, +2}. Any fair mechanism comparison therefore
//! needs controls with the SAME six-sample receptive field, and --
//! for the tensor controls -- the same sparsity pattern, coefficient
//! magnitude, normalization, and arithmetic budget. The decisive
//! question: does the Cayley-Dickson coefficient pattern outperform
//! generic six-sample cubic temporal geometry, or does any cubic
//! filter with this support do as well?
//!
//! Two control families live here:
//!
//! Tensor controls -- `SparseCubicTensor` evaluates sum_{i,j,k}
//! c_{ijk} a_i b_j c_k e_{i XOR j XOR k} over staple triples with the
//! associator's own normalization. `from_associator` extracts the
//! exact CD tensor (and reproduces `joint_associator_norms` to
//! floating-point identity); `sign_scrambled` keeps the support and
//! |c| = 2 but redraws every sign from a seeded stream, destroying the
//! CD sign coherence while preserving everything else.
//!
//! Classical six-sample baselines -- cumulative rotation, maximum
//! rotation, maximum PVI (partial variance of increments, normalized
//! per file by the RMS lag-1 increment), and the maximum Gram
//! determinant volume |det(dB_i, dB_{i+1}, dB_{i+2})| over the three
//! consecutive increment triples, which is cubic in the increments and
//! is the natural "generic cubic geometry" statistic.
//!
//! Channel-axis probes permute (Bx, By, Bz) BEFORE embedding via
//! `permute_channels`; Cayley-Dickson multiplication is not SO(3)
//! equivariant, so a detector edge that survives axis permutation is
//! intrinsic while one that vanishes is coordinate-encoded.

use cd_kernel::mult_table::CdMultTable;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::staple_associator::STAPLE_DIM;

/// Sparse rank-3 tensor over the 16-component staple space: terms are
/// (i, j, k, coefficient), output component is i XOR j XOR k.
pub struct SparseCubicTensor {
    terms: Vec<(u8, u8, u8, i8)>,
}

impl SparseCubicTensor {
    /// Extract the exact basis-associator tensor from the SHA-verified
    /// multiplication table: coefficient of [e_i, e_j, e_k] is
    /// s(i,j)s(i^j,k) - s(j,k)s(i,j^k), nonzero on 1848 of 4096 triples.
    pub fn from_associator(table: &CdMultTable) -> Self {
        let mut terms = Vec::with_capacity(1848);
        for i in 0..STAPLE_DIM {
            for j in 0..STAPLE_DIM {
                for k in 0..STAPLE_DIM {
                    let (s_ij, ij) = table.multiply_basis(i, j);
                    let (s_ij_k, _) = table.multiply_basis(ij, k);
                    let (s_jk, jk) = table.multiply_basis(j, k);
                    let (s_i_jk, _) = table.multiply_basis(i, jk);
                    let c = i32::from(s_ij) * i32::from(s_ij_k) - i32::from(s_jk) * i32::from(s_i_jk);
                    if c != 0 {
                        terms.push((i as u8, j as u8, k as u8, c as i8));
                    }
                }
            }
        }
        Self { terms }
    }

    /// Same support and |coefficient| = 2, every sign redrawn from a
    /// seeded ChaCha8 stream: the CD sign coherence is destroyed while
    /// sparsity, magnitude, receptive field, normalization, and
    /// arithmetic budget stay fixed.
    pub fn sign_scrambled(&self, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let terms = self
            .terms
            .iter()
            .map(|&(i, j, k, c)| {
                let sign: i8 = if rng.random_range(0..2) == 0 { 1 } else { -1 };
                (i, j, k, c.abs() * sign)
            })
            .collect();
        Self { terms }
    }

    /// Number of nonzero terms.
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// Norm of the tensor contraction over one staple triple, divided
    /// by |a||b||c| + 1e-30 -- the associator's own normalization.
    pub fn normalized_score(
        &self,
        a: &[f64; STAPLE_DIM],
        b: &[f64; STAPLE_DIM],
        c: &[f64; STAPLE_DIM],
    ) -> f64 {
        let mut out = [0.0_f64; STAPLE_DIM];
        for &(i, j, k, coeff) in &self.terms {
            let prod = a[i as usize] * b[j as usize] * c[k as usize];
            if prod != 0.0 {
                out[(i ^ j ^ k) as usize] += f64::from(coeff) * prod;
            }
        }
        let raw = out.iter().map(|x| x * x).sum::<f64>().sqrt();
        let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nc = c.iter().map(|x| x * x).sum::<f64>().sqrt();
        raw / (na * nb * nc + 1e-30)
    }

    /// Scores over consecutive staple triples, aligned with
    /// `joint_associator_norms`.
    pub fn scores(&self, staples: &[[f64; STAPLE_DIM]]) -> Vec<f64> {
        if staples.len() < 3 {
            return Vec::new();
        }
        (0..staples.len() - 2)
            .map(|k| self.normalized_score(&staples[k], &staples[k + 1], &staples[k + 2]))
            .collect()
    }
}

/// Apply a channel permutation to (Bx, By, Bz) rows before embedding.
/// The magnitude channel is permutation-invariant, so this probes the
/// detector's dependence on the physical axis-to-basis assignment.
pub fn permute_channels(rows: &[[f64; 3]], perm: [usize; 3]) -> Vec<[f64; 3]> {
    rows.iter()
        .map(|r| [r[perm[0]], r[perm[1]], r[perm[2]]])
        .collect()
}

/// Angle between consecutive field vectors: acos of the clamped cosine,
/// zero when either vector vanishes.
fn pair_angle(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let na = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let nb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    let cosv = ((a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb)).clamp(-1.0, 1.0);
    cosv.acos()
}

/// Classical six-sample window statistics aligned with the associator:
/// entry k consumes rows k..k+5, matching the associator's receptive
/// field exactly. Output length is rows.len() - 5.
pub struct SixSampleBaselines {
    pub cum_rotation: Vec<f64>,
    pub max_rotation: Vec<f64>,
    pub max_pvi: Vec<f64>,
    pub max_gram_volume: Vec<f64>,
}

/// Compute all four classical baselines in one pass. PVI normalizes
/// each lag-1 increment by the file-level RMS increment (Greco &
/// Matthaeus convention applied per daily file, matching the
/// associator's per-file locality).
pub fn six_sample_baselines(rows: &[[f64; 3]]) -> SixSampleBaselines {
    let n = rows.len();
    if n < 6 {
        return SixSampleBaselines {
            cum_rotation: Vec::new(),
            max_rotation: Vec::new(),
            max_pvi: Vec::new(),
            max_gram_volume: Vec::new(),
        };
    }
    // Lag-1 increments and their file-level RMS for PVI normalization.
    let inc: Vec<[f64; 3]> = (1..n)
        .map(|i| {
            [
                rows[i][0] - rows[i - 1][0],
                rows[i][1] - rows[i - 1][1],
                rows[i][2] - rows[i - 1][2],
            ]
        })
        .collect();
    let inc_norm: Vec<f64> = inc
        .iter()
        .map(|d| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt())
        .collect();
    let rms = (inc_norm.iter().map(|x| x * x).sum::<f64>() / inc_norm.len() as f64).sqrt();
    let pvi_denom = rms + 1e-30;

    let angles: Vec<f64> = (1..n).map(|i| pair_angle(&rows[i - 1], &rows[i])).collect();

    let windows = n - 5;
    let mut cum_rotation = Vec::with_capacity(windows);
    let mut max_rotation = Vec::with_capacity(windows);
    let mut max_pvi = Vec::with_capacity(windows);
    let mut max_gram_volume = Vec::with_capacity(windows);
    for k in 0..windows {
        // Window rows k..k+5 own increments/angles k..k+4 (five pairs).
        let a = &angles[k..k + 5];
        cum_rotation.push(a.iter().sum::<f64>());
        max_rotation.push(a.iter().copied().fold(0.0_f64, f64::max));
        max_pvi.push(
            inc_norm[k..k + 5]
                .iter()
                .map(|x| x / pvi_denom)
                .fold(0.0_f64, f64::max),
        );
        // Three consecutive increment triples fit in five increments;
        // |det| is the parallelepiped volume, cubic in the increments.
        let mut vol = 0.0_f64;
        for t in k..k + 3 {
            let (d0, d1, d2) = (&inc[t], &inc[t + 1], &inc[t + 2]);
            let det = d0[0] * (d1[1] * d2[2] - d1[2] * d2[1])
                - d0[1] * (d1[0] * d2[2] - d1[2] * d2[0])
                + d0[2] * (d1[0] * d2[1] - d1[1] * d2[0]);
            vol = vol.max(det.abs());
        }
        max_gram_volume.push(vol);
    }
    SixSampleBaselines {
        cum_rotation,
        max_rotation,
        max_pvi,
        max_gram_volume,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::staple_associator::{joint_associator_norms, staple_embedding};

    fn synthetic_rows(n: usize) -> Vec<[f64; 3]> {
        (0..n)
            .map(|i| {
                let t = i as f64 * 0.37;
                [t.sin() * 3.0 + 1.0, (t * 1.7).cos() * 2.0, t * 0.1 - 0.5]
            })
            .collect()
    }

    #[test]
    fn associator_tensor_reproduces_joint_associator_norms() {
        let table = CdMultTable::generate(STAPLE_DIM);
        let tensor = SparseCubicTensor::from_associator(&table);
        assert_eq!(tensor.term_count(), 1848);
        let staples = staple_embedding(&synthetic_rows(24));
        let via_tensor = tensor.scores(&staples);
        let via_table = joint_associator_norms(&staples, true);
        assert_eq!(via_tensor.len(), via_table.len());
        for (t, m) in via_tensor.iter().zip(&via_table) {
            assert!(
                (t - m).abs() <= 1e-12 * m.abs().max(1.0),
                "tensor {t} vs mult-table {m}"
            );
        }
    }

    #[test]
    fn sign_scrambled_keeps_support_changes_scores() {
        let table = CdMultTable::generate(STAPLE_DIM);
        let tensor = SparseCubicTensor::from_associator(&table);
        let scrambled = tensor.sign_scrambled(7);
        assert_eq!(scrambled.term_count(), 1848);
        let staples = staple_embedding(&synthetic_rows(24));
        let a = tensor.scores(&staples);
        let b = scrambled.scores(&staples);
        let diff: f64 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff > 1e-6, "scrambling the signs changes the scores");
    }

    #[test]
    fn identity_permutation_is_identity() {
        let rows = synthetic_rows(12);
        let same = permute_channels(&rows, [0, 1, 2]);
        assert_eq!(rows, same[..]);
        let swapped = permute_channels(&rows, [1, 0, 2]);
        assert!(rows.iter().zip(&swapped).any(|(a, b)| a != b));
    }

    #[test]
    fn baselines_align_with_associator_length() {
        let rows = synthetic_rows(40);
        let staples = staple_embedding(&rows);
        let assoc = joint_associator_norms(&staples, true);
        let base = six_sample_baselines(&rows);
        assert_eq!(base.cum_rotation.len(), assoc.len());
        assert_eq!(base.max_pvi.len(), assoc.len());
        assert_eq!(base.max_gram_volume.len(), assoc.len());
        assert!(base.cum_rotation.iter().all(|&x| (0.0..=5.0 * std::f64::consts::PI).contains(&x)));
        assert!(base.max_rotation.iter().zip(&base.cum_rotation).all(|(m, c)| m <= c));
    }
}
