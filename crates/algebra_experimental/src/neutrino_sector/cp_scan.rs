use super::hermitian::{C2, cconj, cmul, hermitian_3x3_eig_hybrid};
use flavor_lifts::{FlavorLift, Pdg2024, apply_v6_perturbation};
use nalgebra::ComplexField;

// ---------------------------------------------------------------------------
// Parameterized CP scan point evaluator
// ---------------------------------------------------------------------------
//
// This section implements the parameterized evaluation pipeline for the
// CP violation scan.  The design separates three concerns:
//
//   CpScanResult   -- pure data, the 5 physical observables
//   CpScanContext   -- immutable algebraic data shared across all grid points
//   CpScanBuffers   -- mutable pre-allocated matrices, one per thread
//   evaluate_cp_scan_point() -- the pipeline itself
//
// The separation is motivated by the same principle Lions uses for malloc's
// "coremap" and "swapmap" (Lions 5.1): the resource layout is fixed, but
// the allocator state is per-invocation.  Here, the algebraic "layout"
// (mass matrices, V_6 basis, permutation) is fixed per scan, while the
// eigendecomposition working memory is per-thread mutable state.

/// The five physical observables from a single CP scan point.
///
/// All angles are in **degrees** (not radians), following the PDG
/// convention.  `j_cp` is the rephasing-invariant Jarlskog:
///
/// ```text
/// J = Im(U_e1 * U_mu2 * conj(U_e2) * conj(U_mu1))
/// ```
///
/// `delta_cp` is `arg(-U_e3)` in degrees -- this is convention-dependent.
/// For the rephasing-invariant delta, compute `arg(Jarlskog quartet)`
/// separately (see the rephasing analysis in [`test_cp_violation_joint_3d_scan`]).
///
/// # Concrete values at the optimum (C-1497, AMENDED)
///
/// ```text
/// theta_12 = 32.84,  theta_13 = 8.58,  theta_23 = 49.48
/// j_cp     = 3.33e-2 = J_max (kinematic maximum, delta~90)
/// delta_cp = 97.9 deg (arg(-U_e3)), 92.8 deg (invariant)
/// PDG |J|  = 8.6e-3 (non-maximal, delta=195).  Ratio 3.9x.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CpScanResult {
    /// Solar mixing angle (PDG: 33.41 +/- 0.72 deg).
    pub theta_12: f64,
    /// Reactor mixing angle (PDG: 8.54 +/- 0.12 deg).
    pub theta_13: f64,
    /// Atmospheric mixing angle (PDG: 49.0 +/- 1.3 deg).
    pub theta_23: f64,
    /// Jarlskog invariant -- kinematic maximum J_max for these angles.
    /// PDG measured |J| ~ 8.6e-3 (non-maximal, sin(delta) ~ 0.26).
    pub j_cp: f64,
    /// CP phase from arg(-U_e3), in degrees (convention-dependent).
    pub delta_cp: f64,
    /// Rephasing-invariant delta_CP from atan2(sin_delta, cos_delta),
    /// using Jarlskog for sin and |U_mu1|^2 identity for cos.
    /// In degrees. This is the preferred observable for PDG comparison.
    pub delta_cp_invariant: f64,
}

/// Extract delta_CP using rephasing-invariant observables (PDG convention).
///
/// Uses sin(delta) from the Jarlskog invariant and cos(delta) from the
/// |U_mu1|^2 unitarity relation, combined via atan2 for quadrant resolution.
///
/// This avoids the convention-dependence of arg(-U_e3), which depends on
/// individual matrix element phases rather than physical observables.
pub fn extract_delta_cp_invariant(u_moduli: &[[f64; 3]; 3], j_cp: f64) -> f64 {
    let s13 = u_moduli[0][2];
    let c13 = (1.0 - s13 * s13).max(0.0).sqrt();
    if c13 < 1e-15 {
        return 0.0;
    }

    let s12 = u_moduli[0][1] / c13;
    let c12 = u_moduli[0][0] / c13;
    let s23 = u_moduli[1][2] / c13;
    let c23 = u_moduli[2][2] / c13;

    // sin(delta) = J / (s12*c12*s23*c23*s13*c13^2)
    let denom = s12 * c12 * s23 * c23 * s13 * c13 * c13;
    let sin_delta = if denom.abs() > 1e-15 {
        j_cp / denom
    } else {
        0.0
    };

    // cos(delta) from |U_mu1|^2 = s12^2*c23^2 + c12^2*s23^2*s13^2
    //                              + 2*s12*c12*s23*c23*s13*cos(delta)
    let u_mu1_sq = u_moduli[1][0] * u_moduli[1][0];
    let expected_no_cp = s12 * s12 * c23 * c23 + c12 * c12 * s23 * s23 * s13 * s13;
    let cos_denom = 2.0 * s12 * c12 * s23 * c23 * s13;
    let cos_delta = if cos_denom.abs() > 1e-15 {
        (u_mu1_sq - expected_no_cp) / cos_denom
    } else {
        1.0
    };

    sin_delta.atan2(cos_delta).to_degrees()
}

/// Immutable algebraic context for a CP violation scan.
///
/// # Algebraic origin of each field
///
/// ```text
/// m_nu_real   : 3x3 symmetric, from sedenion associator friction
///               [a, x, b] with selectors (e_7, e_8), coupling alpha_nu=1.35.
///               Constructed by construct_pmns_matrices_two_param().
///
/// m_ch_real   : 3x3 symmetric, from selectors (e_11, e_12), alpha_ch=3.00.
///               Same construction, different subalgebra embedding.
///
/// v6_basis    : nalgebra DMatrix (6 x 42), the top-6 singular vectors of
///               the assessor-space Jacobian.  Spans the directions in the
///               42D assessor space that most efficiently steer the mixing
///               angles.  Computed by extract_v6_basis().
///
/// u_solar     : [f64; 6], the direction in V_6 that moves theta_12 while
///               preserving theta_13.  From compute_constrained_solar_direction().
///
/// u_atmo      : [f64; 6], the direction in V_6 that moves theta_23 while
///               preserving theta_13 and staying orthogonal to u_solar.
///               From compute_constrained_atmospheric_direction().
///
/// lift        : &dyn FlavorLift, maps 42D assessor vectors to 3x3 mass
///               matrix perturbations.  Currently TensorElementLift (the
///               S_3 intertwiner).  This is the ONLY non-algebraically-
///               canonical component -- see C-1489 for the no-equivariance
///               proof.
///
/// perm_u/d    : [usize; 3], eigenvalue permutation indices calibrated
///               against the real mass matrix baseline.  Ensures that
///               eigenvalue 0 maps to the lightest mass eigenstate
///               consistently across the scan.
/// ```
///
/// # Edge cases and invariants
///
/// - **perm stability**: The permutation is computed ONCE from the real
///   (alpha_CP=0) baseline.  If alpha_CP is so large that eigenvalues
///   cross (level repulsion), the permutation becomes invalid and angles
///   will be nonsensical.  The 2% acceptance filter catches this.
///
/// - **v6_basis rank**: The V_6 extraction can produce fewer than 6
///   significant singular values if the assessor-space Jacobian is
///   rank-deficient.  In practice, n_basis = min(nrows, 6) = 6 always.
///
/// - **lift non-uniqueness**: TensorElementLift is response-fitted, not
///   algebraically canonical.  A different FlavorLift implementation
///   would change the optimal (t_sol, t_atm) but NOT the existence
///   of CP violation (which comes from the J_k phase structure).
///
/// # Thread safety
///
/// All fields are shared references (`&'a`), so `CpScanContext` is
/// `Send + Sync` by construction.  It can be shared across rayon
/// threads without cloning.
pub struct CpScanContext<'a> {
    /// Neutrino mass matrix (3x3 real symmetric, from associator friction).
    pub m_nu_real: &'a faer::Mat<f64>,
    /// Charged lepton mass matrix (3x3 real symmetric).
    pub m_ch_real: &'a faer::Mat<f64>,
    /// Top-6 V_6 singular vectors (6 x 42, from assessor-space Jacobian).
    pub v6_basis: &'a nalgebra::DMatrix<f64>,
    /// Constrained solar direction in V_6 space.
    pub u_solar: &'a [f64; 6],
    /// Constrained atmospheric direction in V_6 space.
    pub u_atmo: &'a [f64; 6],
    /// The 42D -> 3x3 flavor lift (currently TensorElementLift).
    pub lift: &'a dyn FlavorLift,
    /// Eigenvalue permutation for charged leptons.
    pub perm_u: [usize; 3],
    /// Eigenvalue permutation for neutrinos.
    pub perm_d: [usize; 3],
}

/// Mutable pre-allocated buffers for the eigendecomposition loop.
///
/// Owns two 3x3 complex Hermitian faer matrices that are overwritten
/// (not re-allocated) at each scan point.  This eliminates ~1200
/// `Mat::zeros(3, 3)` heap allocations per k-embedding in the scan.
///
/// # Why separate from CpScanContext
///
/// `faer::Mat` is not `Sync` (it uses internal mutability for the
/// eigendecomposition working buffer).  By separating the mutable
/// buffers into their own struct, each rayon thread creates its own
/// `CpScanBuffers` while sharing a single `&CpScanContext`.
///
/// # Memory layout
///
/// Each `Mat<c64>::zeros(3, 3)` allocates 9 * 16 = 144 bytes on the
/// heap (9 complex f64s).  Two buffers = 288 bytes per thread.
/// This is allocated ONCE per k-embedding, not per grid point.
pub struct CpScanBuffers {
    /// Pre-allocated neutrino mass matrix (overwritten each point).
    pub m_nu: faer::Mat<faer::c64>,
    /// Pre-allocated charged lepton mass matrix (overwritten each point).
    pub m_ch: faer::Mat<faer::c64>,
}

impl Default for CpScanBuffers {
    fn default() -> Self {
        Self {
            m_nu: faer::Mat::<faer::c64>::zeros(3, 3),
            m_ch: faer::Mat::<faer::c64>::zeros(3, 3),
        }
    }
}

impl CpScanBuffers {
    /// Create a new buffer pair.  Equivalent to `Default::default()`.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Evaluate a single CP scan point in the 3D (alpha_CP, t_sol, t_atm) space.
///
/// # Mathematical pipeline
///
/// The pipeline has 5 steps, each corresponding to a physical operation:
///
/// ```text
/// Step 1: (t_sol, t_atm) --[u_solar, u_atmo]--> beta[6]
///         Linear combination in V_6 constrained-direction space.
///         beta[k] = t_sol * u_solar[k] + t_atm * u_atmo[k]
///
/// Step 2: beta --[apply_v6_perturbation]--> M_nu_pert (3x3 real)
///         The lift maps the 6D beta into a 42D assessor vector,
///         which is then injected into the mass matrix.
///         Symmetrization: M = (M + M^T) / 2.
///
/// Step 3: (M_nu_pert, alpha_CP, phi) --> M_nu_complex (3x3 Hermitian)
///         Phase injection: M[i][j] = |M_pert[i][j]| * exp(i * alpha * phi[i][j])
///         Hermiticity: M[j][i] = conj(M[i][j]).  Diagonal stays real.
///
/// Step 4: (M_ch, M_nu_complex) --[eigendecomp]--> (U_ch, U_nu)
///         Hermitian eigendecomposition via faer (iterative QR).
///         U_PMNS = U_ch^dag * U_nu.
///
/// Step 5: U_PMNS --[perm + extraction]--> CpScanResult
///         Apply perm_u/perm_d, extract |U_ij| -> angles,
///         Jarlskog quartet -> J_CP, arg(-U_e3) -> delta_CP.
/// ```
///
/// # Why phi is a separate argument (not in CpScanContext)
///
/// The phase matrix `phi[i][j]` depends on the k-embedding (which
/// complex structure J_k is used).  The context is k-independent;
/// phi changes per k.  This allows a single context to be reused
/// across all 7 k-embeddings in the parallel scan.
///
/// # Performance
///
/// ~1.5us per call with pre-allocated buffers (dominated by the
/// 3x3 complex eigendecomposition, ~0.7us per matrix * 2 matrices).
/// Without pre-allocation: ~5us per call (dominated by 2 * Mat::zeros).
///
/// # Callers
///
/// - [`test_cp_violation_joint_3d_scan`]: coarse + fine inner loops
/// - Future Nelder-Mead objective via argmin (planned, C-1497 follow-up)
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn evaluate_cp_scan_point(
    alpha_cp: f64,
    t_sol: f64,
    t_atm: f64,
    phi: &[[f64; 3]; 3],
    ctx: &CpScanContext<'_>,
    bufs: &mut CpScanBuffers,
) -> CpScanResult {
    // Step 1: beta from (t_sol, t_atm)
    let mut beta = [0.0_f64; 6];
    for k in 0..6 {
        beta[k] = t_sol * ctx.u_solar[k] + t_atm * ctx.u_atmo[k];
    }

    // Step 2: perturb real mass matrix
    let mut m_nu_pert = ctx.m_nu_real.clone();
    apply_v6_perturbation(&mut m_nu_pert, ctx.v6_basis, &beta, ctx.lift);
    let m_nu_pert = (&m_nu_pert + m_nu_pert.transpose()) * faer::Scale(0.5);

    // Step 3: fill pre-allocated complex Hermitian matrices
    for i in 0..3 {
        bufs.m_nu[(i, i)] = faer::c64::new(m_nu_pert[(i, i)], 0.0);
        bufs.m_ch[(i, i)] = faer::c64::new(ctx.m_ch_real[(i, i)], 0.0);
        for j in (i + 1)..3 {
            let phase = alpha_cp * phi[i][j];
            let mag = m_nu_pert[(i, j)];
            bufs.m_nu[(i, j)] = faer::c64::new(mag * phase.cos(), mag * phase.sin());
            bufs.m_nu[(j, i)] = faer::c64::new(mag * phase.cos(), -mag * phase.sin());
            bufs.m_ch[(i, j)] = faer::c64::new(ctx.m_ch_real[(i, j)], 0.0);
            bufs.m_ch[(j, i)] = faer::c64::new(ctx.m_ch_real[(j, i)], 0.0);
        }
    }

    // Step 4: eigendecompose
    let eig_ch = bufs.m_ch.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let eig_nu = bufs.m_nu.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let u_pmns = eig_ch.U().adjoint() * eig_nu.U();

    // Step 5: extract with permutation (no allocation)
    let u_at = |i: usize, j: usize| -> faer::c64 { u_pmns[(ctx.perm_u[i], ctx.perm_d[j])] };

    let u_e3_abs = u_at(0, 2).abs();
    let theta_13 = u_e3_abs.min(1.0).asin().to_degrees();
    let cos_13 = theta_13.to_radians().cos();

    let theta_12 = if cos_13 > 1e-15 {
        (u_at(0, 1).abs() / cos_13).min(1.0).asin().to_degrees()
    } else {
        0.0
    };

    let theta_23 = if cos_13 > 1e-15 {
        (u_at(1, 2).abs() / cos_13).min(1.0).asin().to_degrees()
    } else {
        0.0
    };

    let j_cp = (u_at(0, 0) * u_at(1, 1) * u_at(0, 1).conj() * u_at(1, 0).conj()).im;

    let delta_cp = (-u_at(0, 2)).arg().to_degrees();

    // Rephasing-invariant delta_CP via moduli + Jarlskog
    let u_moduli = {
        let mut m = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = u_at(i, j).abs();
            }
        }
        m
    };
    let delta_cp_invariant = extract_delta_cp_invariant(&u_moduli, j_cp);

    CpScanResult {
        theta_12,
        theta_13,
        theta_23,
        j_cp,
        delta_cp,
        delta_cp_invariant,
    }
}

// ---------------------------------------------------------------------------
/// Evaluate a single CP scan point using the stack-based Cardano eigensolver.
///
/// Identical to [`evaluate_cp_scan_point`] but uses [`hermitian_3x3_eig_hybrid`]
/// instead of faer's heap-allocated eigendecomposition. This eliminates ALL
/// heap allocation in the inner scan loop.
///
/// The mass matrices are read from `ctx.m_nu_real` and `ctx.m_ch_real` (faer Mats)
/// and converted to stack-allocated `[[C2; 3]; 3]` arrays for the Cardano solver.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn evaluate_cp_scan_point_cardano(
    alpha_cp: f64,
    t_sol: f64,
    t_atm: f64,
    phi: &[[f64; 3]; 3],
    ctx: &CpScanContext<'_>,
) -> CpScanResult {
    // Step 1: beta from (t_sol, t_atm)
    let mut beta = [0.0_f64; 6];
    for k in 0..6 {
        beta[k] = t_sol * ctx.u_solar[k] + t_atm * ctx.u_atmo[k];
    }

    // Step 2: perturb real mass matrix
    let mut m_nu_pert = ctx.m_nu_real.clone();
    apply_v6_perturbation(&mut m_nu_pert, ctx.v6_basis, &beta, ctx.lift);
    let m_nu_pert = (&m_nu_pert + m_nu_pert.transpose()) * faer::Scale(0.5);

    // Step 3: build stack-allocated 3x3 complex Hermitian matrices
    let mut h_nu: [[C2; 3]; 3] = [[(0.0, 0.0); 3]; 3];
    let mut h_ch: [[C2; 3]; 3] = [[(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        h_nu[i][i] = (m_nu_pert[(i, i)], 0.0);
        h_ch[i][i] = (ctx.m_ch_real[(i, i)], 0.0);
        for j in (i + 1)..3 {
            let phase = alpha_cp * phi[i][j];
            let mag = m_nu_pert[(i, j)];
            h_nu[i][j] = (mag * phase.cos(), mag * phase.sin());
            h_nu[j][i] = (mag * phase.cos(), -mag * phase.sin());
            h_ch[i][j] = (ctx.m_ch_real[(i, j)], 0.0);
            h_ch[j][i] = (ctx.m_ch_real[(j, i)], 0.0);
        }
    }

    // Step 4: Cardano eigendecompose
    let (_evals_ch, u_ch) = hermitian_3x3_eig_hybrid(&h_ch);
    let (_evals_nu, u_nu) = hermitian_3x3_eig_hybrid(&h_nu);

    // U_PMNS = U_ch^dag * U_nu
    let mut u_pmns = [[(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = (0.0_f64, 0.0_f64);
            for k in 0..3 {
                let a = cconj(u_ch[k][i]);
                let b = u_nu[k][j];
                s.0 += a.0 * b.0 - a.1 * b.1;
                s.1 += a.0 * b.1 + a.1 * b.0;
            }
            u_pmns[i][j] = s;
        }
    }

    // Step 5: apply permutation and extract
    let u_at = |i: usize, j: usize| -> C2 { u_pmns[ctx.perm_u[i]][ctx.perm_d[j]] };

    let u_e3 = u_at(0, 2);
    let u_e3_abs = (u_e3.0 * u_e3.0 + u_e3.1 * u_e3.1).sqrt();
    let theta_13 = u_e3_abs.min(1.0).asin().to_degrees();
    let cos_13 = theta_13.to_radians().cos();

    let theta_12 = if cos_13 > 1e-15 {
        let u_e2 = u_at(0, 1);
        let u_e2_abs = (u_e2.0 * u_e2.0 + u_e2.1 * u_e2.1).sqrt();
        (u_e2_abs / cos_13).min(1.0).asin().to_degrees()
    } else {
        0.0
    };

    let theta_23 = if cos_13 > 1e-15 {
        let u_mu3 = u_at(1, 2);
        let u_mu3_abs = (u_mu3.0 * u_mu3.0 + u_mu3.1 * u_mu3.1).sqrt();
        (u_mu3_abs / cos_13).min(1.0).asin().to_degrees()
    } else {
        0.0
    };

    let j_cp = cmul(
        cmul(u_at(0, 0), u_at(1, 1)),
        cmul(cconj(u_at(0, 1)), cconj(u_at(1, 0))),
    )
    .1;

    let neg_ue3 = (-u_e3.0, -u_e3.1);
    let delta_cp = neg_ue3.1.atan2(neg_ue3.0).to_degrees();

    let u_moduli = {
        let mut m = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let u = u_at(i, j);
                m[i][j] = (u.0 * u.0 + u.1 * u.1).sqrt();
            }
        }
        m
    };
    let delta_cp_invariant = extract_delta_cp_invariant(&u_moduli, j_cp);

    CpScanResult {
        theta_12,
        theta_13,
        theta_23,
        j_cp,
        delta_cp,
        delta_cp_invariant,
    }
}

// Nelder-Mead refinement of CP scan (argmin)
// ---------------------------------------------------------------------------

/// Cost function for argmin Nelder-Mead optimization of the CP scan.
///
/// The parameter vector is `[alpha_CP, t_sol, t_atm]` (k is frozen from
/// the grid search).  The cost function penalizes angle deviation from PDG
/// and rewards large |J_CP|.
pub struct CpNelderMeadCost<'a> {
    pub ctx: &'a CpScanContext<'a>,
    pub phi: [[f64; 3]; 3],
    pub bounds: [(f64, f64); 3],
    /// If true, pure prediction mode: cost = -|J_CP| (no angle penalty).
    pub prediction_mode: bool,
    /// Weight for mass-ratio penalty (r = dm21^2/dm31^2). 0.0 = no penalty.
    pub r_penalty_weight: f64,
}

impl<'a> argmin::core::CostFunction for CpNelderMeadCost<'a> {
    type Param = Vec<f64>;
    type Output = f64;

    // PMNS scan trait method: indexed access to the param Vec<f64>
    // (param[0..3]) is the canonical argmin pattern; iter().enumerate()
    // would only complicate the structured destructuring downstream.
    #[allow(clippy::needless_range_loop)]
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let alpha_cp = param[0].clamp(self.bounds[0].0, self.bounds[0].1);
        let t_sol = param[1].clamp(self.bounds[1].0, self.bounds[1].1);
        let t_atm = param[2].clamp(self.bounds[2].0, self.bounds[2].1);

        let r = evaluate_cp_scan_point_cardano(alpha_cp, t_sol, t_atm, &self.phi, self.ctx);

        if self.prediction_mode {
            return Ok(-r.j_cp.abs());
        }

        let pdg = Pdg2024::default();
        let err_12 = ((r.theta_12 - pdg.theta_12_deg) / pdg.theta_12_err).powi(2);
        let err_13 = ((r.theta_13 - pdg.theta_13_deg) / pdg.theta_13_err).powi(2);
        let err_23 = ((r.theta_23 - pdg.theta_23_deg) / pdg.theta_23_err).powi(2);
        let chi2_angles = err_12 + err_13 + err_23;

        let mut cost = chi2_angles - 100.0 * r.j_cp.abs();

        // Optional mass-ratio penalty
        if self.r_penalty_weight > 0.0 {
            // Compute eigenvalues via Cardano to get r
            let mut beta = [0.0_f64; 6];
            for k in 0..6 {
                beta[k] = t_sol * self.ctx.u_solar[k] + t_atm * self.ctx.u_atmo[k];
            }
            let mut m_nu = self.ctx.m_nu_real.clone();
            apply_v6_perturbation(&mut m_nu, self.ctx.v6_basis, &beta, self.ctx.lift);
            let m_nu = (&m_nu + m_nu.transpose()) * faer::Scale(0.5);
            let mut h_nu: [[C2; 3]; 3] = [[(0.0, 0.0); 3]; 3];
            for i in 0..3 {
                h_nu[i][i] = (m_nu[(i, i)], 0.0);
                for j in (i + 1)..3 {
                    let phase = alpha_cp * self.phi[i][j];
                    let mag = m_nu[(i, j)];
                    h_nu[i][j] = (mag * phase.cos(), mag * phase.sin());
                    h_nu[j][i] = (mag * phase.cos(), -mag * phase.sin());
                }
            }
            let (evals, _) = hermitian_3x3_eig_hybrid(&h_nu);
            let mut ev: [f64; 3] = [evals[0].abs(), evals[1].abs(), evals[2].abs()];
            ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let dm21 = ev[1] * ev[1] - ev[0] * ev[0];
            let dm31 = ev[2] * ev[2] - ev[0] * ev[0];
            let r_mass = if dm31.abs() > 1e-30 { dm21 / dm31 } else { 0.0 };
            let chi2_r = ((r_mass - 0.0307) / 0.001).powi(2);
            cost += self.r_penalty_weight * chi2_r;
        }

        Ok(cost)
    }
}

/// Run Nelder-Mead refinement starting from the grid-best point.
///
/// Freezes k (discrete), optimizes `(alpha_CP, t_sol, t_atm)` continuously.
/// Uses multi-start: grid-best + 3 random perturbations.
///
/// Returns the best `CpScanResult` found across all starts, along with the
/// optimized parameters `(alpha_CP, t_sol, t_atm)`.
pub fn refine_cp_nelder_mead(
    ctx: &CpScanContext<'_>,
    phi: &[[f64; 3]; 3],
    alpha0: f64,
    t_sol0: f64,
    t_atm0: f64,
    prediction_mode: bool,
) -> (CpScanResult, [f64; 3]) {
    refine_cp_nelder_mead_r(ctx, phi, alpha0, t_sol0, t_atm0, prediction_mode, 0.0)
}

/// Nelder-Mead with optional mass-ratio penalty.
///
/// `r_penalty_weight`: 0.0 = no r penalty (original behavior).
/// Positive values add chi2_r = weight * ((r - 0.0307) / 0.001)^2 to the cost.
pub fn refine_cp_nelder_mead_r(
    ctx: &CpScanContext<'_>,
    phi: &[[f64; 3]; 3],
    alpha0: f64,
    t_sol0: f64,
    t_atm0: f64,
    prediction_mode: bool,
    r_penalty_weight: f64,
) -> (CpScanResult, [f64; 3]) {
    use argmin::{
        core::{Executor, State},
        solver::neldermead::NelderMead,
    };

    let bounds = [
        (0.01, 1.0),
        (t_sol0 - 2.0, t_sol0 + 4.0),
        (t_atm0 - 2.0, t_atm0 + 8.0),
    ];

    let build_simplex = |x0: &[f64; 3]| -> Vec<Vec<f64>> {
        let steps = [0.05, 0.2, 0.5];
        let mut simplex = Vec::with_capacity(4);
        simplex.push(vec![x0[0], x0[1], x0[2]]);
        for i in 0..3 {
            let mut v = vec![x0[0], x0[1], x0[2]];
            v[i] += steps[i];
            for (j, &(lo, hi)) in bounds.iter().enumerate() {
                v[j] = v[j].clamp(lo, hi);
            }
            simplex.push(v);
        }
        simplex
    };

    // Multi-start: grid-best + 3 perturbations
    let starts: [[f64; 3]; 4] = [
        [alpha0, t_sol0, t_atm0],
        [alpha0 * 1.2, t_sol0 + 0.3, t_atm0 - 0.5],
        [alpha0 * 0.8, t_sol0 - 0.3, t_atm0 + 0.5],
        [(alpha0 + 0.1).min(1.0), t_sol0 + 0.5, t_atm0 + 1.0],
    ];

    let mut best_cost = f64::MAX;
    let mut best_params = [alpha0, t_sol0, t_atm0];

    for start in &starts {
        let simplex = build_simplex(start);
        let solver = match NelderMead::new(simplex).with_sd_tolerance(1e-8) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let problem = CpNelderMeadCost {
            ctx,
            phi: *phi,
            bounds,
            prediction_mode,
            r_penalty_weight,
        };
        let run = Executor::new(problem, solver)
            .configure(|state| state.max_iters(500))
            .run();
        if let Ok(result) = run {
            let state = result.state();
            if let Some(param) = state.get_best_param() {
                let cost = state.get_best_cost();
                if cost < best_cost {
                    best_cost = cost;
                    best_params = [
                        param[0].clamp(bounds[0].0, bounds[0].1),
                        param[1].clamp(bounds[1].0, bounds[1].1),
                        param[2].clamp(bounds[2].0, bounds[2].1),
                    ];
                }
            }
        }
    }

    let mut bufs = CpScanBuffers::new();
    let result = evaluate_cp_scan_point(
        best_params[0],
        best_params[1],
        best_params[2],
        phi,
        ctx,
        &mut bufs,
    );
    (result, best_params)
}
