//! Brans-Dicke scalar-tensor field equations and junction conditions.
//!
//! This module provides the field-theoretic infrastructure for analyzing
//! scalar-tensor gravity theories, focusing on Brans-Dicke (BD) theory
//! as the simplest prototype.
//!
//! # Key concepts
//!
//! - **Conformal transformation**: BD Jordan frame -> Einstein frame via g_tilde = phi * g
//! - **Fifth force**: scalar-mediated Yukawa coupling with strength alpha = 1/sqrt(2*omega+3)
//! - **Junction conditions**: Israel-Darmois matching across thin shells, extended for scalar
//! - **Bianchi identity violation**: non-dynamical kappa(x) breaks covariant conservation
//!
//! # References
//! - Brans, C. & Dicke, R. H. (1961), Phys. Rev. 124, 925
//! - Rodal, J. (2025), arXiv:2507.09724v2, Eq. 2, 5, 13
//! - Fujii, Y. & Maeda, K. (2003), The Scalar-Tensor Theory of Gravitation
//! - Will, C. M. (2014), Living Rev. Relativity 17, 4

/// Parameters for Brans-Dicke scalar-tensor theory.
#[derive(Clone, Copy, Debug)]
pub struct BransDickeParams {
    /// BD coupling constant (dimensionless).
    /// GR is recovered in the limit omega -> infinity.
    pub omega: f64,
    /// Background scalar field VEV (conventionally 1.0).
    pub phi_0: f64,
    /// Scalar potential V(phi) evaluated at phi_0.
    /// For massless BD, this is zero.
    pub v_potential: f64,
}

impl Default for BransDickeParams {
    fn default() -> Self {
        Self {
            omega: 50_000.0,
            phi_0: 1.0,
            v_potential: 0.0,
        }
    }
}

/// Junction condition data at a thin-shell boundary.
///
/// Encodes the Israel-Darmois matching conditions extended for the
/// scalar degree of freedom in BD theory.
#[derive(Clone, Copy, Debug)]
pub struct JunctionCondition {
    /// Jump in extrinsic curvature trace `[K]`.
    pub delta_k_trace: f64,
    /// Surface stress-energy trace S = h^{ij} S_{ij}.
    pub surface_stress_trace: f64,
    /// Normal derivative jump of scalar field `[d_perp phi]`.
    pub scalar_normal_jump: f64,
}

/// Conformal factor for Jordan -> Einstein frame transformation.
///
/// g_tilde_{mu nu} = phi * g_{mu nu}
///
/// At phi = 1 (GR vacuum), the frames coincide.
pub fn conformal_factor(phi: f64) -> f64 {
    phi
}

/// Fifth-force coupling strength in Brans-Dicke theory.
///
/// alpha = 1 / sqrt(2*omega + 3)
///
/// This is the ratio of the scalar-mediated force to Newtonian gravity.
/// For omega = 43,000 (Cassini bound), alpha ~ 0.0034.
/// For omega -> infinity (GR), alpha -> 0.
pub fn fifth_force_alpha(omega: f64) -> f64 {
    assert!(
        2.0 * omega + 3.0 > 0.0,
        "2*omega+3 must be positive for real alpha"
    );
    1.0 / (2.0 * omega + 3.0).sqrt()
}

/// Junction scalar normal derivative jump from surface stress.
///
/// In BD theory, the scalar field equation across a thin shell gives:
///   `[d_perp phi]` = -(8*pi / (3 + 2*omega)) * S
///
/// where S is the trace of the surface stress-energy tensor.
/// Rodal (2025) Eq. 13.
pub fn junction_scalar_jump(surface_stress_trace: f64, omega: f64) -> f64 {
    let denom = 3.0 + 2.0 * omega;
    assert!(denom.abs() > f64::EPSILON, "3 + 2*omega must be nonzero");
    -(8.0 * std::f64::consts::PI / denom) * surface_stress_trace
}

/// Bianchi identity violation magnitude for non-dynamical kappa(x).
///
/// If the gravitational coupling kappa is promoted to a spatial function
/// kappa(x) WITHOUT making it a dynamical field (i.e., without adding
/// kinetic and potential terms), then the contracted Bianchi identity
/// forces:
///   nabla_mu T^{mu nu} = (T / (2*kappa)) * nabla^nu kappa
///
/// which is nonzero whenever both T (stress-energy trace) and grad(kappa)
/// are nonzero. This violates local energy-momentum conservation.
/// Rodal (2025) Eq. 2.
///
/// Returns the magnitude of the violation: |grad_kappa * T / (2 * kappa)|.
/// A nonzero return means the theory is internally inconsistent.
pub fn bianchi_violation_magnitude(grad_kappa: f64, t_trace: f64, kappa: f64) -> f64 {
    assert!(kappa.abs() > f64::EPSILON, "kappa must be nonzero");
    (grad_kappa * t_trace / (2.0 * kappa)).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conformal_factor_identity_at_phi_1() {
        assert!((conformal_factor(1.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fifth_force_alpha_decreasing_with_omega() {
        let a100 = fifth_force_alpha(100.0);
        let a1000 = fifth_force_alpha(1000.0);
        let a10000 = fifth_force_alpha(10_000.0);
        assert!(a100 > a1000, "alpha should decrease with omega");
        assert!(a1000 > a10000, "alpha should decrease with omega");
    }

    #[test]
    fn test_fifth_force_alpha_gr_limit() {
        let alpha = fifth_force_alpha(1e12);
        assert!(alpha < 1e-6, "GR limit: alpha={alpha:.3e}, expected ~0");
    }

    #[test]
    fn test_junction_scalar_jump_zero_for_vacuum() {
        // Zero surface stress => zero scalar jump
        let jump = junction_scalar_jump(0.0, 50_000.0);
        assert!(
            jump.abs() < f64::EPSILON,
            "Vacuum shell should have zero scalar jump"
        );
    }

    #[test]
    fn test_bianchi_violation_nonzero_for_varying_kappa() {
        // Non-dynamical kappa(x) with nonzero gradient and nonzero T
        let violation = bianchi_violation_magnitude(0.1, -1e10, 1.0);
        assert!(
            violation > 0.0,
            "Varying kappa with nonzero T must violate Bianchi"
        );

        // Zero gradient => no violation
        let no_violation = bianchi_violation_magnitude(0.0, -1e10, 1.0);
        assert!(
            no_violation < f64::EPSILON,
            "Constant kappa should not violate Bianchi"
        );
    }
}
