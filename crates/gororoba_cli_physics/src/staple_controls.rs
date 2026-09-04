//! Temporal-window controls for the staple-associator detector.
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
//! Classical window baselines -- cumulative rotation, maximum
//! rotation, maximum PVI (partial variance of increments, normalized
//! per file by the RMS lag-1 increment), and the maximum Gram
//! determinant volume |det(dB_i, dB_{i+1}, dB_{i+2})| over the three
//! consecutive increment triples, which is cubic in the increments and
//! is the natural "generic cubic geometry" statistic. PVI's full-file
//! normalization extends its dependency beyond the six-sample window;
//! score alignment alone therefore establishes only numerator support.
//!
//! Channel-axis probes permute (Bx, By, Bz) BEFORE embedding via
//! `permute_channels`; Cayley-Dickson multiplication is not SO(3)
//! equivariant, so a detector edge that survives axis permutation is
//! intrinsic while one that vanishes is coordinate-encoded.

use cd_kernel::mult_table::CdMultTable;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::staple_associator::STAPLE_DIM;

/// A sign function sigma(i,j) in {+1,-1} on the 16-element basis: the twist
/// of a unital XOR-graded algebra, e_i e_j = sigma(i,j) e_{i XOR j}.
pub type Twist = [[i8; STAPLE_DIM]; STAPLE_DIM];

/// The Cayley-Dickson twist read off the SHA-verified multiplication
/// table, so a random-twist ladder has the true sedenion twist at rung
/// zero in the same representation.
pub fn cd_twist(table: &CdMultTable) -> Twist {
    let mut sigma = [[0i8; STAPLE_DIM]; STAPLE_DIM];
    for (i, row) in sigma.iter_mut().enumerate() {
        for (j, entry) in row.iter_mut().enumerate() {
            let (sign, index) = table.multiply_basis(i, j);
            assert_eq!(
                index,
                i ^ j,
                "the CD product of e_{i} and e_{j} is XOR-graded"
            );
            *entry = sign;
        }
    }
    sigma
}

/// A uniformly random unital twist: sigma(0,j) = sigma(i,0) = +1 and every
/// other sign is an independent fair draw from the seeded stream.
pub fn random_unital_twist(rng: &mut ChaCha8Rng) -> Twist {
    let mut sigma = [[1i8; STAPLE_DIM]; STAPLE_DIM];
    for row in sigma.iter_mut().skip(1) {
        for entry in row.iter_mut().skip(1) {
            *entry = if rng.random_range(0..2) == 0 { 1 } else { -1 };
        }
    }
    sigma
}

/// Row-major sha256 of the twist as bytes 0x01 / 0xff, so a draw can be
/// named and reproduced from its hash alone.
pub fn twist_sha256(sigma: &Twist) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for row in sigma {
        for &entry in row {
            hasher.update([entry as u8]);
        }
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

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
                    let c =
                        i32::from(s_ij) * i32::from(s_ij_k) - i32::from(s_jk) * i32::from(s_i_jk);
                    if c != 0 {
                        terms.push((i as u8, j as u8, k as u8, c as i8));
                    }
                }
            }
        }
        Self { terms }
    }

    /// The associator tensor of the unital XOR-graded algebra with
    /// multiplication e_i e_j = sigma(i,j) e_{i XOR j}: the coefficient of
    /// [e_i, e_j, e_k] is sigma(i,j) sigma(i^j,k) - sigma(j,k) sigma(i,j^k),
    /// the same formula `from_associator` evaluates on the Cayley-Dickson
    /// sign table. `from_twist(&cd_twist(table))` therefore reproduces
    /// `from_associator(table)` term for term, and a random twist yields a
    /// genuine (in general non-alternative) algebra with its own zero
    /// pattern, one rung above the sign scramble that keeps the CD support
    /// but is no algebra at all.
    pub fn from_twist(sigma: &Twist) -> Self {
        let mut terms = Vec::with_capacity(2048);
        for i in 0..STAPLE_DIM {
            for j in 0..STAPLE_DIM {
                for k in 0..STAPLE_DIM {
                    let ij = i ^ j;
                    let jk = j ^ k;
                    let c = i32::from(sigma[i][j]) * i32::from(sigma[ij][k])
                        - i32::from(sigma[j][k]) * i32::from(sigma[i][jk]);
                    if c != 0 {
                        terms.push((i as u8, j as u8, k as u8, c as i8));
                    }
                }
            }
        }
        Self { terms }
    }

    /// Positive and negative coefficient counts over the support.
    pub fn sign_counts(&self) -> (usize, usize) {
        let positive = self.terms.iter().filter(|t| t.3 > 0).count();
        (positive, self.terms.len() - positive)
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

/// Window statistics aligned with the associator. Entry k uses rows
/// k..k+5 for rotation, Gram volume, and the PVI numerator. PVI also
/// uses the RMS increment across all input rows as calibration context.
/// Output length is rows.len().saturating_sub(5).
pub struct SixSampleBaselines {
    pub cum_rotation: Vec<f64>,
    pub max_rotation: Vec<f64>,
    pub max_pvi: Vec<f64>,
    pub max_gram_volume: Vec<f64>,
}

/// Compute all four classical baselines in one pass. PVI normalizes
/// each lag-1 increment by the file-level RMS increment (Greco &
/// Matthaeus convention applied per daily file). PVI therefore depends
/// on samples outside its six-sample numerator window, including later
/// samples; its score describes an offline daily-file measurement.
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
    fn from_twist_of_the_cd_twist_reproduces_from_associator() {
        let table = CdMultTable::generate(STAPLE_DIM);
        let cd = SparseCubicTensor::from_associator(&table);
        let sigma = cd_twist(&table);
        let twisted = SparseCubicTensor::from_twist(&sigma);
        assert_eq!(twisted.terms, cd.terms);
        assert_eq!(cd.term_count(), 1848);
        assert_eq!(sigma[0][5], 1);
        assert_eq!(sigma[5][0], 1);
    }

    #[test]
    fn random_twist_is_unital_and_yields_a_different_algebra() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let sigma = random_unital_twist(&mut rng);
        assert!(sigma[0].iter().all(|&s| s == 1));
        assert!(sigma.iter().all(|row| row[0] == 1));
        let t = SparseCubicTensor::from_twist(&sigma);
        assert!(t.term_count() > 0);
        let table = CdMultTable::generate(STAPLE_DIM);
        let cd = SparseCubicTensor::from_associator(&table);
        assert_ne!(t.terms, cd.terms);
        let (pos, neg) = t.sign_counts();
        assert_eq!(pos + neg, t.term_count());
        let again = random_unital_twist(&mut ChaCha8Rng::seed_from_u64(7));
        assert_eq!(twist_sha256(&again), twist_sha256(&sigma));
        assert_ne!(twist_sha256(&sigma), twist_sha256(&cd_twist(&table)));
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
    fn pvi_calibration_uses_increments_outside_the_scored_window() {
        let original_rows: Vec<[f64; 3]> =
            (1..=7).map(|value| [f64::from(value), 0.0, 0.0]).collect();
        let mut changed_rows = original_rows.clone();
        changed_rows[6] = [106.0, 0.0, 0.0];
        assert_eq!(original_rows[..6], changed_rows[..6]);

        let original = six_sample_baselines(&original_rows);
        let changed = six_sample_baselines(&changed_rows);
        assert_eq!(original.max_pvi[0], 1.0);
        let expected_pvi = (6.0_f64 / 10005.0).sqrt();
        assert!((changed.max_pvi[0] - expected_pvi).abs() < 1e-14);
        assert!(changed.max_pvi[0] < original.max_pvi[0]);
    }

    #[test]
    fn local_window_statistics_ignore_suffix_changes() {
        let original_rows = synthetic_rows(12);
        let mut changed_rows = original_rows.clone();
        for row in &mut changed_rows[6..] {
            *row = [106.0, -42.0, 17.0];
        }
        let original = six_sample_baselines(&original_rows);
        let changed = six_sample_baselines(&changed_rows);
        assert_eq!(original.cum_rotation[0], changed.cum_rotation[0]);
        assert_eq!(original.max_rotation[0], changed.max_rotation[0]);
        assert_eq!(original.max_gram_volume[0], changed.max_gram_volume[0]);
        let original_associator = joint_associator_norms(&staple_embedding(&original_rows), true);
        let changed_associator = joint_associator_norms(&staple_embedding(&changed_rows), true);
        assert_eq!(original_associator[0], changed_associator[0]);
        assert_ne!(original.max_pvi[0], changed.max_pvi[0]);
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
        assert!(
            base.cum_rotation
                .iter()
                .all(|&x| (0.0..=5.0 * std::f64::consts::PI).contains(&x))
        );
        assert!(
            base.max_rotation
                .iter()
                .zip(&base.cum_rotation)
                .all(|(m, c)| m <= c)
        );
    }
}
