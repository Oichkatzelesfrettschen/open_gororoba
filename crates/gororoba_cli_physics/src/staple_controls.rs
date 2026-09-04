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

/// The 35 lines of PG(3, 2), ordered canonically: each line is (a, b, c)
/// with 1 <= a < b < c <= 15 and a ^ b = c.
pub const PG32_LINE_COUNT: usize = 35;

/// Enumerate all 35 projective lines in PG(3, 2) = ((Z_2)^4 \ {0}) / ~ .
pub fn pg32_lines() -> [(usize, usize, usize); PG32_LINE_COUNT] {
    let mut lines = [(0, 0, 0); PG32_LINE_COUNT];
    let mut count = 0;
    for a in 1..STAPLE_DIM {
        for b in (a + 1)..STAPLE_DIM {
            let c = a ^ b;
            if b < c {
                lines[count] = (a, b, c);
                count += 1;
            }
        }
    }
    assert_eq!(count, PG32_LINE_COUNT);
    lines
}

/// The 15 Fano planes of PG(3, 2): kernels of the 15 nonzero linear forms
/// on (Z_2)^4, each with 7 nonzero points.
pub fn pg32_planes() -> [[usize; 7]; 15] {
    let mut planes = [[0usize; 7]; 15];
    for (slot, func) in (1u32..16).enumerate() {
        let mut n = 0;
        for x in 1..16u32 {
            if (func & x).count_ones() % 2 == 0 {
                planes[slot][n] = x as usize;
                n += 1;
            }
        }
        assert_eq!(n, 7);
    }
    planes
}

/// Associator support of `sigma` restricted to one Fano plane.
/// Cayley-Dickson sedenions put this at 168 on exactly eight of the fifteen
/// planes (those planes are octonion subalgebras).
pub fn plane_associator_terms(sigma: &Twist, plane: &[usize; 7]) -> usize {
    let mut terms = 0;
    for &i in plane {
        for &j in plane {
            for &k in plane {
                let c = i32::from(sigma[i][j]) * i32::from(sigma[i ^ j][k])
                    - i32::from(sigma[j][k]) * i32::from(sigma[i][j ^ k]);
                if c != 0 {
                    terms += 1;
                }
            }
        }
    }
    terms
}

/// How many of the 15 planes are octonion (168 associator terms) for `sigma`.
pub fn octonion_plane_count(sigma: &Twist) -> usize {
    pg32_planes()
        .iter()
        .filter(|plane| plane_associator_terms(sigma, plane) == 168)
        .count()
}

/// Number of the 15 planes that contain the line and are octonion for `sigma`.
pub fn line_octonion_incidence(sigma: &Twist, line: (usize, usize, usize)) -> usize {
    let (a, b, c) = line;
    pg32_planes()
        .iter()
        .filter(|plane| {
            plane.contains(&a)
                && plane.contains(&b)
                && plane.contains(&c)
                && plane_associator_terms(sigma, plane) == 168
        })
        .count()
}

/// Build a basis-alternative, anticommutative unital twist from orientation
/// signs on the 35 lines of PG(3, 2).
///
/// For each line (a, b, c) with a ^ b = c, sign s in {+1, -1}:
/// cyclic pairs: sigma(a, b) = sigma(b, c) = sigma(c, a) = s
/// anticyclic:   sigma(b, a) = sigma(c, b) = sigma(a, c) = -s
/// imaginary squares: sigma(i, i) = -1
/// identity row/col: sigma(0, i) = sigma(i, 0) = +1.
///
/// This construction guarantees that every pair of basis elements generates
/// an associative (quaternionic or sub-quaternionic) algebra: [e_i, e_i, e_j] = 0,
/// [e_i, e_j, e_j] = 0, and [e_i, e_j, e_i] = 0 identically for all i, j.
pub fn twist_from_line_orientations(
    lines: &[(usize, usize, usize); PG32_LINE_COUNT],
    signs: &[i8; PG32_LINE_COUNT],
) -> Twist {
    let mut sigma = [[1i8; STAPLE_DIM]; STAPLE_DIM];
    for (i, row) in sigma.iter_mut().enumerate().skip(1) {
        row[i] = -1;
    }
    for (&(a, b, c), &s) in lines.iter().zip(signs.iter()) {
        sigma[a][b] = s;
        sigma[b][c] = s;
        sigma[c][a] = s;
        sigma[b][a] = -s;
        sigma[c][b] = -s;
        sigma[a][c] = -s;
    }
    sigma
}

/// Extract line orientations from an existing basis-alternative twist (e.g. CD).
pub fn extract_line_orientations(
    sigma: &Twist,
    lines: &[(usize, usize, usize); PG32_LINE_COUNT],
) -> [i8; PG32_LINE_COUNT] {
    let mut signs = [0i8; PG32_LINE_COUNT];
    for (k, &(a, b, _c)) in lines.iter().enumerate() {
        signs[k] = sigma[a][b];
    }
    signs
}

/// A uniformly random basis-alternative twist: fair draw of the 35 line signs.
pub fn random_alternative_twist(
    lines: &[(usize, usize, usize); PG32_LINE_COUNT],
    rng: &mut ChaCha8Rng,
) -> Twist {
    let mut signs = [1i8; PG32_LINE_COUNT];
    for s in signs.iter_mut() {
        *s = if rng.random_range(0..2) == 0 { 1 } else { -1 };
    }
    twist_from_line_orientations(lines, &signs)
}

/// Generate a uniformly random invertible 4x4 matrix over GF(2) (an element of GL(4, Z_2)).
/// Group order is (16-1)(16-2)(16-4)(16-8) = 15 * 14 * 12 * 8 = 20,160.
pub fn random_gl4_z2(rng: &mut ChaCha8Rng) -> [u8; 4] {
    let r0 = rng.random_range(1..16) as u8;
    let mut r1 = rng.random_range(1..16) as u8;
    while r1 == r0 {
        r1 = rng.random_range(1..16) as u8;
    }
    let span2 = [0, r0, r1, r0 ^ r1];
    let mut r2 = rng.random_range(1..16) as u8;
    while span2.contains(&r2) {
        r2 = rng.random_range(1..16) as u8;
    }
    let mut span3 = [0u8; 8];
    for (idx, &x) in span2.iter().enumerate() {
        span3[idx] = x;
        span3[idx + 4] = x ^ r2;
    }
    let mut r3 = rng.random_range(1..16) as u8;
    while span3.contains(&r3) {
        r3 = rng.random_range(1..16) as u8;
    }
    [r0, r1, r2, r3]
}

/// Apply a GL(4, Z_2) matrix to an index in 0..15.
pub fn apply_gl4(matrix: &[u8; 4], v: usize) -> usize {
    let mut out = 0;
    for (bit, &row) in matrix.iter().enumerate() {
        let dot = (row as usize & v).count_ones() % 2;
        out |= dot << bit;
    }
    out as usize
}

/// Construct an algebra-isomorphic relabeling of the Cayley-Dickson twist.
///
/// Under any linear automorphism pi in GL(4, Z_2) and signs s in {+1, -1}^16 (s[0]=+1):
/// sigma'(i, j) = s[i] * s[j] * s[i ^ j] * sigma0(pi(i), pi(j)).
///
/// The map F: e_i -> s[i] * e_{pi(i)} is an algebra isomorphism from (A, sigma')
/// to (A, sigma0). Its associator tensor has identically 1848 terms (924 positive,
/// 924 negative) with coefficients in {-2, +2}.
pub fn isomorphic_twist(sigma0: &Twist, matrix: &[u8; 4], signs: &[i8; STAPLE_DIM]) -> Twist {
    let mut perm = [0usize; STAPLE_DIM];
    for (i, p) in perm.iter_mut().enumerate() {
        *p = apply_gl4(matrix, i);
    }
    let mut sigma = [[1i8; STAPLE_DIM]; STAPLE_DIM];
    for i in 0..STAPLE_DIM {
        for j in 0..STAPLE_DIM {
            let pi_i = perm[i];
            let pi_j = perm[j];
            let raw = sigma0[pi_i][pi_j];
            let s = signs[i] * signs[j] * signs[i ^ j];
            sigma[i][j] = raw * s;
        }
    }
    sigma
}

/// Random draw from the algebra-isomorphic relabeling orbit.
pub fn random_isomorphic_twist(sigma0: &Twist, rng: &mut ChaCha8Rng) -> Twist {
    let matrix = random_gl4_z2(rng);
    let mut signs = [1i8; STAPLE_DIM];
    for s in signs.iter_mut().skip(1) {
        *s = if rng.random_range(0..2) == 0 { 1 } else { -1 };
    }
    isomorphic_twist(sigma0, &matrix, &signs)
}

/// Generate a uniformly random normalized 2-cocycle on (Z_2)^4:
/// sigma(i, j) = (-1)^{i^T B j} * tau(i) * tau(j) * tau(i ^ j)
/// where B is a random 4x4 matrix over GF(2) and tau: {0..15} -> {+1, -1}
/// with tau(0) = +1.
///
/// A normalized 2-cocycle defines an associative twisted group algebra,
/// so its associator tensor is identically zero (0 terms).
pub fn random_normalized_cocycle(rng: &mut ChaCha8Rng) -> Twist {
    let mut b = [[0u8; 4]; 4];
    for row in b.iter_mut() {
        for entry in row.iter_mut() {
            *entry = rng.random_range(0..2) as u8;
        }
    }
    let mut tau = [1i8; STAPLE_DIM];
    for entry in tau.iter_mut().skip(1) {
        *entry = if rng.random_range(0..2) == 0 { 1 } else { -1 };
    }
    let mut sigma = [[1i8; STAPLE_DIM]; STAPLE_DIM];
    for i in 0..STAPLE_DIM {
        for j in 0..STAPLE_DIM {
            let mut quad = 0u32;
            for (p, row) in b.iter().enumerate() {
                if ((i >> p) & 1) != 0 {
                    for (q, &entry) in row.iter().enumerate() {
                        if ((j >> q) & 1) != 0 && entry == 1 {
                            quad ^= 1;
                        }
                    }
                }
            }
            let sign_b: i8 = if quad == 1 { -1 } else { 1 };
            sigma[i][j] = sign_b * tau[i] * tau[j] * tau[i ^ j];
        }
    }
    sigma
}

/// Where a canonical score sits relative to an ensemble's central 95 percent interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnsembleIntervalPosition {
    Below,
    Inside,
    Above,
}

/// Compare `canonical` to `[p2_5, p97_5]`. Equality on a fence sits inside.
pub fn ensemble_interval_position(
    canonical: f64,
    p2_5: f64,
    p97_5: f64,
) -> EnsembleIntervalPosition {
    if canonical > p97_5 {
        EnsembleIntervalPosition::Above
    } else if canonical < p2_5 {
        EnsembleIntervalPosition::Below
    } else {
        EnsembleIntervalPosition::Inside
    }
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

    /// Read-only access to the non-zero terms (i, j, k, coeff).
    pub fn terms(&self) -> &[(u8, u8, u8, i8)] {
        &self.terms
    }

    /// Score a single staple triple with a precomputed inverse denominator.
    pub fn score_triple_precomputed(
        &self,
        a: &[f64; STAPLE_DIM],
        b: &[f64; STAPLE_DIM],
        c: &[f64; STAPLE_DIM],
        inv_denom: f64,
    ) -> f64 {
        if self.terms.is_empty() {
            return 0.0;
        }
        let mut out = [0.0_f64; STAPLE_DIM];
        for &(i, j, k, coeff) in &self.terms {
            let prod = a[i as usize] * b[j as usize] * c[k as usize];
            if prod != 0.0 {
                out[(i ^ j ^ k) as usize] += f64::from(coeff) * prod;
            }
        }
        let raw_sq: f64 = out.iter().map(|x| x * x).sum();
        raw_sq.sqrt() * inv_denom
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

    #[test]
    fn score_triple_precomputed_matches_normalized_score() {
        let table = CdMultTable::generate(STAPLE_DIM);
        let cd = SparseCubicTensor::from_associator(&table);
        let rows = synthetic_rows(10);
        let staples = staple_embedding(&rows);
        let a = &staples[0];
        let b = &staples[1];
        let c = &staples[2];
        let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nc = c.iter().map(|x| x * x).sum::<f64>().sqrt();
        let inv_denom = 1.0 / (na * nb * nc + 1e-30);
        let s1 = cd.normalized_score(a, b, c);
        let s2 = cd.score_triple_precomputed(a, b, c, inv_denom);
        assert!((s1 - s2).abs() <= 1e-12 * s1.abs().max(1.0));
    }

    #[test]
    fn pg32_lines_reproduce_cd_twist() {
        let lines = pg32_lines();
        assert_eq!(lines.len(), 35);
        let table = CdMultTable::generate(STAPLE_DIM);
        let cd = cd_twist(&table);
        let signs = extract_line_orientations(&cd, &lines);
        let reconstructed = twist_from_line_orientations(&lines, &signs);
        assert_eq!(reconstructed, cd);
    }

    #[test]
    fn normalized_cocycles_have_zero_associator() {
        let mut rng = ChaCha8Rng::seed_from_u64(12345);
        for _ in 0..20 {
            let cocycle = random_normalized_cocycle(&mut rng);
            // Verify unital
            for (i, row) in cocycle.iter().enumerate() {
                assert_eq!(cocycle[0][i], 1);
                assert_eq!(row[0], 1);
            }
            let tensor = SparseCubicTensor::from_twist(&cocycle);
            assert_eq!(
                tensor.term_count(),
                0,
                "any 2-cocycle must have vanishing associator"
            );
        }
    }

    #[test]
    fn isomorphic_twists_preserve_terms_and_balance() {
        let table = CdMultTable::generate(STAPLE_DIM);
        let cd = cd_twist(&table);
        let mut rng = ChaCha8Rng::seed_from_u64(54321);
        for _ in 0..10 {
            let iso = random_isomorphic_twist(&cd, &mut rng);
            let tensor = SparseCubicTensor::from_twist(&iso);
            assert_eq!(
                tensor.term_count(),
                1848,
                "isomorphic twists must have 1848 terms"
            );
            let (pos, neg) = tensor.sign_counts();
            assert_eq!(pos, 924);
            assert_eq!(neg, 924);
        }
    }

    #[test]
    fn random_alternative_twists_satisfy_basis_alternativity() {
        let lines = pg32_lines();
        let mut rng = ChaCha8Rng::seed_from_u64(999);
        for _ in 0..5 {
            let sigma = random_alternative_twist(&lines, &mut rng);
            for i in 1..STAPLE_DIM {
                assert_eq!(sigma[i][i], -1);
                for j in 1..STAPLE_DIM {
                    if i != j {
                        assert_eq!(sigma[i][j], -sigma[j][i], "anticommutative");
                        // Left alternativity [e_i, e_i, e_j] = 0 <=> sigma(i, i ^ j) == -sigma(i, j)
                        assert_eq!(sigma[i][i ^ j], -sigma[i][j], "left alternative on basis");
                    }
                }
            }
        }
    }

    #[test]
    fn pg32_hyperplane_octonion_count_is_eight() {
        let table = CdMultTable::generate(STAPLE_DIM);
        let cd = cd_twist(&table);
        assert_eq!(octonion_plane_count(&cd), 8);
    }

    #[test]
    fn alternative_twist_term_counts_are_1080_plus_96k() {
        let lines = pg32_lines();
        let table = CdMultTable::generate(STAPLE_DIM);
        let cd = cd_twist(&table);
        let cd_n = SparseCubicTensor::from_twist(&cd).term_count();
        assert_eq!(cd_n, 1848);
        assert_eq!((cd_n - 1080) % 96, 0);
        assert_eq!((cd_n - 1080) / 96, 8);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..128 {
            let sigma = random_alternative_twist(&lines, &mut rng);
            let n = SparseCubicTensor::from_twist(&sigma).term_count();
            assert_eq!(n % 24, 0);
            assert!((1080..=1656).contains(&n) || n == 1848);
            assert_eq!((n - 1080) % 96, 0, "term count {n} is not 1080 + 96 k");
        }
    }

    #[test]
    fn hamming1_term_drop_tracks_octonion_incidence() {
        let lines = pg32_lines();
        let table = CdMultTable::generate(STAPLE_DIM);
        let cd = cd_twist(&table);
        let cd_signs = extract_line_orientations(&cd, &lines);
        let mut drop_by_inc = [Vec::<usize>::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut inc3_points = std::collections::BTreeSet::new();
        let mut inc3_lines = 0usize;
        for (k, &line) in lines.iter().enumerate() {
            let inc = line_octonion_incidence(&cd, line);
            let mut signs = cd_signs;
            signs[k] = -signs[k];
            let n = SparseCubicTensor::from_twist(&twist_from_line_orientations(&lines, &signs))
                .term_count();
            let drop = 1848 - n;
            drop_by_inc[inc].push(drop);
            if inc == 3 {
                inc3_lines += 1;
                inc3_points.insert(line.0);
                inc3_points.insert(line.1);
                inc3_points.insert(line.2);
            }
        }
        assert_eq!(inc3_lines, 7);
        assert_eq!(inc3_points.len(), 15);
        let inc3: Vec<(usize, usize, usize)> = lines
            .iter()
            .copied()
            .filter(|line| line_octonion_incidence(&cd, *line) == 3)
            .collect();
        let mut common: Vec<usize> = vec![inc3[0].0, inc3[0].1, inc3[0].2];
        for &(a, b, c) in &inc3 {
            common.retain(|p| *p == a || *p == b || *p == c);
        }
        assert_eq!(common, vec![8], "288-drop lines are the pencil through e_8");
        assert_eq!(8 / 4, 2, "e_8 is lag 2 in the staple packing");
        assert_eq!(8 % 4, 0, "e_8 is channel 0 (Bx) in the staple packing");
        for (inc, drops) in drop_by_inc.iter().enumerate() {
            if drops.is_empty() {
                continue;
            }
            let lo = *drops.iter().min().unwrap();
            let hi = *drops.iter().max().unwrap();
            assert_eq!(
                lo, hi,
                "Hamming-1 drop is not constant on octonion-incidence {inc}: {drops:?}"
            );
        }
        assert!(
            !drop_by_inc[0].is_empty() || !drop_by_inc[1].is_empty(),
            "expected some lines off the octonion planes"
        );
    }

    #[test]
    fn reconstructed_cd_has_1848_terms_random_alternative_twists_do_not() {
        let lines = pg32_lines();
        let table = CdMultTable::generate(STAPLE_DIM);
        let cd = cd_twist(&table);
        let reconstructed =
            twist_from_line_orientations(&lines, &extract_line_orientations(&cd, &lines));
        let cd_tensor = SparseCubicTensor::from_twist(&reconstructed);
        assert_eq!(cd_tensor.term_count(), 1848);
        let (cd_pos, cd_neg) = cd_tensor.sign_counts();
        assert_eq!((cd_pos, cd_neg), (924, 924));

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut max_terms = 0;
        for _ in 0..64 {
            let sigma = random_alternative_twist(&lines, &mut rng);
            let tensor = SparseCubicTensor::from_twist(&sigma);
            let (pos, neg) = tensor.sign_counts();
            assert_eq!(pos, neg, "35-line twists keep sign balance");
            assert!(tensor.term_count() < 1848);
            max_terms = max_terms.max(tensor.term_count());
        }
        assert!(max_terms >= 1080);
    }

    #[test]
    fn cd_1848_term_orientation_is_isolated_at_hamming_one() {
        let lines = pg32_lines();
        let table = CdMultTable::generate(STAPLE_DIM);
        let cd_signs = extract_line_orientations(&cd_twist(&table), &lines);
        let cd_terms =
            SparseCubicTensor::from_twist(&twist_from_line_orientations(&lines, &cd_signs))
                .term_count();
        assert_eq!(cd_terms, 1848);

        let mut hamming1_max = 0usize;
        let mut hamming1_at_1848 = 0usize;
        for k in 0..PG32_LINE_COUNT {
            let mut signs = cd_signs;
            signs[k] = -signs[k];
            let n = SparseCubicTensor::from_twist(&twist_from_line_orientations(&lines, &signs))
                .term_count();
            hamming1_max = hamming1_max.max(n);
            if n == 1848 {
                hamming1_at_1848 += 1;
            }
        }
        assert_eq!(
            hamming1_at_1848, 0,
            "a Hamming-1 neighbor of the CD orientation has 1848 terms"
        );
        assert!(
            hamming1_max < 1848,
            "Hamming-1 neighbor term-count max is {hamming1_max}"
        );

        let mut hamming2_at_1848 = 0usize;
        for a in 0..PG32_LINE_COUNT {
            for b in (a + 1)..PG32_LINE_COUNT {
                let mut signs = cd_signs;
                signs[a] = -signs[a];
                signs[b] = -signs[b];
                let n =
                    SparseCubicTensor::from_twist(&twist_from_line_orientations(&lines, &signs))
                        .term_count();
                if n == 1848 {
                    hamming2_at_1848 += 1;
                }
            }
        }
        assert_eq!(
            hamming2_at_1848, 0,
            "a Hamming-2 neighbor of the CD orientation has 1848 terms"
        );
    }

    #[test]
    fn ensemble_interval_position_splits_on_the_fences() {
        assert_eq!(
            ensemble_interval_position(0.8274, 0.8280, 0.8373),
            EnsembleIntervalPosition::Below
        );
        assert_eq!(
            ensemble_interval_position(0.8300, 0.8280, 0.8373),
            EnsembleIntervalPosition::Inside
        );
        assert_eq!(
            ensemble_interval_position(0.8400, 0.8280, 0.8373),
            EnsembleIntervalPosition::Above
        );
        assert_eq!(
            ensemble_interval_position(0.8280, 0.8280, 0.8373),
            EnsembleIntervalPosition::Inside
        );
    }
}
