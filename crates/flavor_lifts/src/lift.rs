//! Flavor lift trait and implementations.
//!
//! # The lift problem
//!
//! A `FlavorLift` maps a perturbation vector from the 42D assessor space
//! (or its V_6 complement) into a symmetric 3x3 mass matrix.  This is the
//! architectural layer between algebraic family geometry and physical
//! mixing matrices.
//!
//! The central theorem from Epic B (C-1502): V_6 is a **psi-eigenspace**
//! with eigenvalue -0.2215 (6-fold degenerate scalar).  This means V_6
//! carries a trivial-like S_3 representation, while Sym_3(R) carries the
//! natural permutation representation.  By Schur's lemma, no S_3-equivariant
//! map from a scalar source to a permutation target exists -- explaining
//! why TensorElementLift WORKS by being non-equivariant.
//!
//! # Implementation history (negative-result ladder)
//!
//! | Lift | Effective DOFs | Result | Claim |
//! |------|---------------|--------|-------|
//! | [`AssessorToFlavorMap`] (12/12/6 partition) | 3 | FAIL: collinear gradients | C-1475 |
//! | [`DirectOffDiagonalLift`] (3 blocks of 14) | 3 | FAIL: rank-2 Jacobian lock | C-1476 |
//! | [`PsiEquivariantLift`] (psi orbit weights) | 3 | FAIL: weak gradients | C-1477 |
//! | [`TensorElementLift`] (6 blocks of 7) | **6** | **SUCCESS** | C-1478 |
//!
//! The pattern: all 42->3 lifts hit the rank-2 lock because 3 DOFs cannot
//! independently steer 3 mixing angles (g_12 lies 100% in span{g_13, g_23}).
//! The 42->6 lift breaks this lock by preserving 6 independent DOFs matching
//! the dimension of V_6 exactly.

/// Core trait: map an assessor-space perturbation into a mass matrix.
pub trait FlavorLift {
    /// Apply the perturbation vector `v` (assessor space) to the
    /// mass matrix `m`. The perturbation must be symmetric.
    fn lift(&self, v: &[f64], m: &mut faer::Mat<f64>);
}

/// Assessor-to-flavor map using subalgebra overlap partition.
///
/// For assessor (low, high) with low in 1..7, high in 9..15:
/// - Solar (1-2): low in 4..7 (O1-only) AND high in 9..11 (O2)
/// - Reactor (1-3): low in 4..7 (O1-only) AND high in 12..15 (O3)
/// - Atmospheric (2-3): low in 1..3 (shared quaternion) AND high in 9..11 (O2)
pub struct AssessorToFlavorMap {
    /// Assessor indices connecting generation 1 and 2 (solar channel).
    pub gen_12_indices: Vec<usize>,
    /// Assessor indices connecting generation 1 and 3 (reactor channel).
    pub gen_13_indices: Vec<usize>,
    /// Assessor indices connecting generation 2 and 3 (atmospheric channel).
    pub gen_23_indices: Vec<usize>,
}

impl AssessorToFlavorMap {
    /// Default partition based on subalgebra overlap structure.
    pub fn default_partition(assessors: &[(usize, usize)]) -> Self {
        let mut gen_12 = Vec::new();
        let mut gen_13 = Vec::new();
        let mut gen_23 = Vec::new();

        for (a_idx, &(low, high)) in assessors.iter().enumerate() {
            let low_in_o1_only = (4..=7).contains(&low);
            let high_in_o2 = (9..=11).contains(&high);
            let high_in_o3 = (12..=15).contains(&high);

            if low_in_o1_only && high_in_o2 {
                gen_12.push(a_idx);
            } else if low_in_o1_only && high_in_o3 {
                gen_13.push(a_idx);
            } else if high_in_o2 && (1..=3).contains(&low) {
                gen_23.push(a_idx);
            }
        }

        Self { gen_12_indices: gen_12, gen_13_indices: gen_13, gen_23_indices: gen_23 }
    }

    /// Map a 42D assessor vector to a symmetric 3x3 generation matrix.
    ///
    /// The diagonal is zero (no self-coupling). Off-diagonal (i,j) is the
    /// sum of assessor components for that generation pair.
    pub fn to_generation_matrix(&self, v: &[f64]) -> faer::Mat<f64> {
        let f_12: f64 = self.gen_12_indices.iter().map(|&i| v[i]).sum();
        let f_13: f64 = self.gen_13_indices.iter().map(|&i| v[i]).sum();
        let f_23: f64 = self.gen_23_indices.iter().map(|&i| v[i]).sum();

        let mut m = faer::Mat::<f64>::zeros(3, 3);
        m.write(0, 1, f_12);
        m.write(1, 0, f_12);
        m.write(0, 2, f_13);
        m.write(2, 0, f_13);
        m.write(1, 2, f_23);
        m.write(2, 1, f_23);
        m
    }
}

impl FlavorLift for AssessorToFlavorMap {
    fn lift(&self, v: &[f64], m: &mut faer::Mat<f64>) {
        let delta = self.to_generation_matrix(v);
        for i in 0..3 {
            for j in 0..3 {
                let sym = (delta.read(i, j) + delta.read(j, i)) / 2.0;
                m.write(i, j, m.read(i, j) + sym);
            }
        }
    }
}

/// Direct off-diagonal injection: bypasses the assessor partition entirely.
///
/// Collapses the 42D perturbation into 3 independent off-diagonal coupling
/// strengths by partitioning into 3 blocks of 14. Each block's sum drives
/// one off-diagonal element of the mass matrix directly.
///
/// This targets eigenvector rotation (off-diagonal torque) instead of
/// eigenvalue shifting (diagonal trace). However, it still reduces 42D to
/// only 3 effective DOFs, which means it hits the same rank-2 Jacobian
/// lock as the partition-based lift.
pub struct DirectOffDiagonalLift;

impl FlavorLift for DirectOffDiagonalLift {
    fn lift(&self, v: &[f64], m: &mut faer::Mat<f64>) {
        let n = v.len().min(42);
        let block_size = n / 3;
        let torque_12: f64 = v[..block_size].iter().sum();
        let torque_13: f64 = v[block_size..2 * block_size].iter().sum();
        let torque_23: f64 = v[2 * block_size..n].iter().sum();

        m.write(0, 1, m.read(0, 1) + torque_12);
        m.write(1, 0, m.read(1, 0) + torque_12);
        m.write(0, 2, m.read(0, 2) + torque_13);
        m.write(2, 0, m.read(2, 0) + torque_13);
        m.write(1, 2, m.read(1, 2) + torque_23);
        m.write(2, 1, m.read(2, 1) + torque_23);
    }
}

/// Psi-equivariant lift: maps assessors to generations using the Gourlay psi
/// automorphism's orbit structure.
///
/// For each assessor pair (low, high), embeds it as a 16D unit vector,
/// applies psi, and classifies the orbit:
/// - Fixed points (psi(v) = v): contribute to diagonal (trace)
/// - 3-cycle orbits: contribute to off-diagonal via the circulant structure
///
/// This uses the S_3 generator directly rather than hand-crafted index bins.
/// However, the current psi linearization is known to be not faithful
/// (psi^3 != I on V_6), so gradients are weak. This is a strong provisional
/// result, not yet the final S_3-equivariant lift.
pub struct PsiEquivariantLift {
    /// For each assessor index: (off-diag contribution weights [f_12, f_13, f_23])
    weights: Vec<[f64; 3]>,
}

impl PsiEquivariantLift {
    /// Build the psi-orbit classification for the given assessor pairs.
    pub fn from_assessors(assessors: &[(usize, usize)]) -> Self {
        use cd_kernel::gourlay_psi;

        let mut weights = Vec::with_capacity(assessors.len());

        for &(low, high) in assessors {
            let mut v = [0.0_f64; 16];
            v[low] = 1.0;
            v[high] = 1.0;

            let psi_v = gourlay_psi(&v);
            let self_overlap: f64 = v.iter().zip(psi_v.iter()).map(|(a, b)| a * b).sum();

            let psi2_v = gourlay_psi(&psi_v);
            let cross_overlap: f64 = v.iter().zip(psi2_v.iter()).map(|(a, b)| a * b).sum();

            let norm = (self_overlap.abs() + cross_overlap.abs()).max(1e-15);
            let w_12 = -self_overlap / norm;
            let w_13 = -cross_overlap / norm;
            let w_23 = 1.0 - w_12.abs() - w_13.abs();

            weights.push([w_12, w_13, w_23]);
        }

        Self { weights }
    }
}

impl FlavorLift for PsiEquivariantLift {
    fn lift(&self, v: &[f64], m: &mut faer::Mat<f64>) {
        let mut f_12 = 0.0_f64;
        let mut f_13 = 0.0_f64;
        let mut f_23 = 0.0_f64;

        for (idx, &val) in v.iter().enumerate() {
            if idx >= self.weights.len() { break; }
            f_12 += val * self.weights[idx][0];
            f_13 += val * self.weights[idx][1];
            f_23 += val * self.weights[idx][2];
        }

        m.write(0, 1, m.read(0, 1) + f_12);
        m.write(1, 0, m.read(1, 0) + f_12);
        m.write(0, 2, m.read(0, 2) + f_13);
        m.write(2, 0, m.read(2, 0) + f_13);
        m.write(1, 2, m.read(1, 2) + f_23);
        m.write(2, 1, m.read(2, 1) + f_23);
    }
}

/// Tensor element lift: maps 42 assessors into 6 blocks of 7, each driving
/// one independent element of the symmetric 3x3 mass matrix.
///
/// This is the minimal successful lift (C-1478) that breaks the rank-2
/// Jacobian lock. It injects into all 6 independent elements of Herm_3
/// (3 diagonal + 3 off-diagonal), matching the 6D V_6 exactly.
///
/// **Scope**: The specific 7-assessor block assignment is a PROJECT-SPECIFIC
/// heuristic, not yet derived from the algebra. Whether this lift lies in
/// Hom_{S_3}(V_6, Sym_3) is the central open question (Epic B).
///
/// Block assignment (by natural assessor ordering):
///   assessors  0-6  -> M_11 (diagonal, gen 1)
///   assessors  7-13 -> M_22 (diagonal, gen 2)
///   assessors 14-20 -> M_33 (diagonal, gen 3)
///   assessors 21-27 -> M_12 (off-diagonal, solar)
///   assessors 28-34 -> M_13 (off-diagonal, reactor)
///   assessors 35-41 -> M_23 (off-diagonal, atmospheric)
pub struct TensorElementLift;

impl FlavorLift for TensorElementLift {
    fn lift(&self, v: &[f64], m: &mut faer::Mat<f64>) {
        let n = v.len().min(42);
        let block = 7_usize;

        let sum_block = |start: usize| -> f64 {
            let end = (start + block).min(n);
            v[start..end].iter().sum()
        };

        // Diagonal elements
        m.write(0, 0, m.read(0, 0) + sum_block(0));
        m.write(1, 1, m.read(1, 1) + sum_block(block));
        m.write(2, 2, m.read(2, 2) + sum_block(2 * block));

        // Off-diagonal elements (symmetric injection)
        let m12 = sum_block(3 * block);
        let m13 = sum_block(4 * block);
        let m23 = sum_block(5 * block);

        m.write(0, 1, m.read(0, 1) + m12);
        m.write(1, 0, m.read(1, 0) + m12);
        m.write(0, 2, m.read(0, 2) + m13);
        m.write(2, 0, m.read(2, 0) + m13);
        m.write(1, 2, m.read(1, 2) + m23);
        m.write(2, 1, m.read(2, 1) + m23);
    }
}

/// Apply a V_6 perturbation to a mass matrix using a pluggable lift.
///
/// For each V_6 direction k with coefficient beta[k], computes the combined
/// assessor-space vector, then delegates to the `FlavorLift` implementation
/// to map it into the mass matrix.
///
/// Invariant: beta = [0; 6] leaves m unchanged.
pub fn apply_v6_perturbation(
    m: &mut faer::Mat<f64>,
    v6_basis: &nalgebra::DMatrix<f64>,
    beta: &[f64; 6],
    flavor_lift: &dyn FlavorLift,
) {
    let n_basis = v6_basis.nrows().min(6);
    let n_cols = v6_basis.ncols();

    let mut v_combined = vec![0.0_f64; n_cols];
    for k in 0..n_basis {
        if beta[k].abs() < 1e-15 { continue; }
        for col in 0..n_cols {
            v_combined[col] += beta[k] * v6_basis[(k, col)];
        }
    }

    flavor_lift.lift(&v_combined, m);
}
