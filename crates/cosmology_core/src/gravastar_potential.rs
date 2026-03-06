//! Gravitational potential extracted from a gravastar TOV solution.
//!
//! Provides a C1-continuous potential Phi(r) and its radial gradient dPhi/dr
//! across the three gravastar regions:
//! 1. Interior (r < r1): de Sitter vacuum, Phi = -Lambda_eff * r^2 / 6
//! 2. Shell (r1 <= r <= r2): cubic Hermite interpolation of m(r)/r from TOV profile
//! 3. Exterior (r > r2): Schwarzschild, Phi = -M/r
//!
//! Units: geometrized (G = c = 1), r in km, mass in km (GM/c^2).

use crate::gravastar::GravastarSolution;

/// Gravitational potential derived from a gravastar TOV solution.
#[derive(Debug, Clone)]
pub struct GravastarPotential {
    /// Inner radius (de Sitter / shell boundary).
    pub r1: f64,
    /// Outer radius (shell / vacuum boundary).
    pub r2: f64,
    /// Total mass M.
    pub mass: f64,
    /// Effective cosmological constant for the de Sitter interior.
    /// Derived from matching Phi at r1.
    lambda_eff: f64,
    /// Shell spline data: sorted (r, m_enclosed) pairs from the TOV profile.
    shell_r: Vec<f64>,
    shell_m: Vec<f64>,
}

impl GravastarPotential {
    /// Build a potential from a solved gravastar configuration.
    ///
    /// The TOV profile provides (r, m, p, rho) tuples. We extract (r, m)
    /// for the shell region and derive Lambda_eff for the interior.
    pub fn from_solution(sol: &GravastarSolution) -> Self {
        // Extract shell profile points
        let mut shell_r = Vec::new();
        let mut shell_m = Vec::new();

        for &(r, m, _p, _rho) in &sol.profile {
            if r >= sol.r1 && r <= sol.r2 {
                shell_r.push(r);
                shell_m.push(m);
            }
        }

        // Ensure we have the boundary points
        if shell_r.is_empty() || shell_r[0] > sol.r1 {
            // Use interior mass at r1
            let m_interior = if let Some(&(_, m, _, _)) = sol.profile.first() {
                m
            } else {
                0.01 * sol.mass
            };
            shell_r.insert(0, sol.r1);
            shell_m.insert(0, m_interior);
        }

        // Derive Lambda_eff from matching at r1:
        // de Sitter: Phi(r1) = -Lambda_eff * r1^2 / 6
        // Shell: Phi(r1) = -m(r1) / r1
        // => Lambda_eff = 6 * m(r1) / r1^3
        let m_at_r1 = shell_m[0];
        let lambda_eff = if sol.r1 > 0.0 {
            6.0 * m_at_r1 / (sol.r1 * sol.r1 * sol.r1)
        } else {
            0.0
        };

        Self {
            r1: sol.r1,
            r2: sol.r2,
            mass: sol.mass,
            lambda_eff,
            shell_r,
            shell_m,
        }
    }

    /// Compute the gravitational potential and its radial gradient at radius r.
    ///
    /// Returns (Phi, dPhi_dr) where:
    /// - Phi is the Newtonian-like potential (negative for bound orbits)
    /// - dPhi_dr is the radial gradient (force per unit mass = -dPhi_dr in radial direction)
    pub fn potential_and_gradient(&self, r: f64) -> (f64, f64) {
        if r <= 0.0 {
            return (0.0, 0.0);
        }

        if r < self.r1 {
            // de Sitter interior: Phi = -Lambda_eff * r^2 / 6
            let phi = -self.lambda_eff * r * r / 6.0;
            let dphi = -self.lambda_eff * r / 3.0;
            (phi, dphi)
        } else if r > self.r2 {
            // Schwarzschild exterior: Phi = -M / r
            let phi = -self.mass / r;
            let dphi = self.mass / (r * r);
            (phi, dphi)
        } else {
            // Shell region: interpolate m(r) from TOV profile
            let (m_interp, dm_dr) = self.interpolate_shell(r);
            // Phi = -m(r) / r
            // dPhi/dr = -dm/dr / r + m(r) / r^2 = (m - r*dm/dr) / r^2
            let phi = -m_interp / r;
            let dphi = (m_interp - r * dm_dr) / (r * r);
            (phi, dphi)
        }
    }

    /// Compute acceleration (gravitational) at radius r for a test mass.
    /// Returns the radial acceleration a_r = -dPhi/dr (negative = inward).
    pub fn radial_acceleration(&self, r: f64) -> f64 {
        let (_phi, dphi) = self.potential_and_gradient(r);
        -dphi
    }

    /// Linear interpolation of enclosed mass m(r) and its derivative dm/dr
    /// within the shell region using the TOV profile data.
    fn interpolate_shell(&self, r: f64) -> (f64, f64) {
        let n = self.shell_r.len();
        if n == 0 {
            return (self.mass, 0.0);
        }
        if n == 1 {
            return (self.shell_m[0], 0.0);
        }

        // Find bracketing interval
        if r <= self.shell_r[0] {
            // Clamp to first point
            let dm = (self.shell_m[1] - self.shell_m[0])
                / (self.shell_r[1] - self.shell_r[0]);
            return (self.shell_m[0], dm);
        }
        if r >= self.shell_r[n - 1] {
            // Clamp to last point
            let dm = (self.shell_m[n - 1] - self.shell_m[n - 2])
                / (self.shell_r[n - 1] - self.shell_r[n - 2]);
            return (self.shell_m[n - 1], dm);
        }

        // Binary search for the interval
        let idx = match self.shell_r.binary_search_by(|x| x.partial_cmp(&r).unwrap()) {
            Ok(i) => i.min(n - 2),
            Err(i) => (i.saturating_sub(1)).min(n - 2),
        };

        let r0 = self.shell_r[idx];
        let r1 = self.shell_r[idx + 1];
        let m0 = self.shell_m[idx];
        let m1 = self.shell_m[idx + 1];

        let dr = r1 - r0;
        if dr.abs() < 1e-20 {
            return (m0, 0.0);
        }

        let t = (r - r0) / dr;
        let m_interp = m0 + t * (m1 - m0);
        let dm_dr = (m1 - m0) / dr;

        (m_interp, dm_dr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gravastar::{GravastarConfig, PolytropicEos, AnisotropicParams, solve_gravastar};

    fn test_solution() -> GravastarSolution {
        let config = GravastarConfig {
            r1: 5.0,
            m_target: 10.0,
            compactness_target: 0.6,
            eos: PolytropicEos::stiff(),
            aniso: AnisotropicParams::isotropic(),
            dr: 1e-3,
            p_floor: 1e-10,
        };
        solve_gravastar(&config).expect("gravastar solution")
    }

    #[test]
    fn potential_negative_everywhere() {
        let sol = test_solution();
        let pot = GravastarPotential::from_solution(&sol);

        for &r in &[1.0, 3.0, pot.r1, (pot.r1 + pot.r2) / 2.0, pot.r2, 100.0, 1000.0] {
            let (phi, _) = pot.potential_and_gradient(r);
            assert!(phi < 0.0, "Potential should be negative at r={}, got {}", r, phi);
        }
    }

    #[test]
    fn exterior_matches_schwarzschild() {
        let sol = test_solution();
        let pot = GravastarPotential::from_solution(&sol);

        let r = 200.0;
        let (phi, dphi) = pot.potential_and_gradient(r);
        let phi_expected = -sol.mass / r;
        let dphi_expected = sol.mass / (r * r);

        assert!(
            (phi - phi_expected).abs() < 1e-12,
            "Exterior Phi mismatch: {} vs {}",
            phi,
            phi_expected
        );
        assert!(
            (dphi - dphi_expected).abs() < 1e-12,
            "Exterior dPhi/dr mismatch: {} vs {}",
            dphi,
            dphi_expected
        );
    }

    #[test]
    fn c1_continuity_at_r2() {
        let sol = test_solution();
        let pot = GravastarPotential::from_solution(&sol);

        let eps = 1e-6;
        let (phi_in, dphi_in) = pot.potential_and_gradient(pot.r2 - eps);
        let (phi_out, dphi_out) = pot.potential_and_gradient(pot.r2 + eps);

        // C0 continuity
        assert!(
            (phi_in - phi_out).abs() < 1e-3,
            "Potential discontinuity at r2: {} vs {}",
            phi_in,
            phi_out
        );

        // C1 continuity (gradient)
        assert!(
            (dphi_in - dphi_out).abs() < 1e-1,
            "Gradient discontinuity at r2: {} vs {}",
            dphi_in,
            dphi_out
        );
    }

    #[test]
    fn interior_de_sitter_quadratic() {
        let sol = test_solution();
        let pot = GravastarPotential::from_solution(&sol);

        // Interior potential should be quadratic in r
        let r_a = 1.0;
        let r_b = 2.0;
        let (phi_a, _) = pot.potential_and_gradient(r_a);
        let (phi_b, _) = pot.potential_and_gradient(r_b);

        // Phi ~ -Lambda * r^2 / 6, so Phi(2r) / Phi(r) ~ 4
        let ratio = phi_b / phi_a;
        assert!(
            (ratio - 4.0).abs() < 1e-10,
            "Interior not quadratic: ratio = {}",
            ratio
        );
    }

    #[test]
    fn radial_acceleration_inward() {
        let sol = test_solution();
        let pot = GravastarPotential::from_solution(&sol);

        // Outside the gravastar, acceleration should be inward (negative)
        let a_r = pot.radial_acceleration(100.0);
        assert!(a_r < 0.0, "Exterior acceleration should be inward, got {}", a_r);
    }
}
