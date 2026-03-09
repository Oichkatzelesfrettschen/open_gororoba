//! Force-field approximation (FFA) for heliospheric cosmic ray modulation.
//!
//! The Gleeson-Axford (1968) force-field approximation reduces the full 3D
//! Parker Transport Equation to a single scalar parameter phi (GV), the
//! modulation potential. It is exact when diffusion is spherically symmetric
//! and the solar wind is radial.
//!
//! J(r, E_kin) = J_ISM(E_kin + phi) * (E_kin^2 + 2*m*E_kin) / ((E_kin+phi)^2 + 2*m*(E_kin+phi))
//!
//! phi = integral from r to R_B of V_sw(r') / (3 * kappa_rr(r', R)) dr'
//!
//! Reference: Gleeson & Axford (1968) ApJ 154, 1011.
//! Fisk (1971) extended this to non-spherical geometries.

use gauss_quad::GaussLegendre;

/// Force-field proxy for fast diagnostic modulation calculations.
pub struct ForceFieldProxy {
    /// Outer boundary radius (AU) -- typically Voyager termination shock ~157 AU.
    pub r_boundary_au: f64,
    /// Particle rest mass (GeV/c^2). Use 0.938272 for protons.
    pub particle_mass_gev: f64,
}

impl ForceFieldProxy {
    /// Create a new force-field proxy.
    pub fn new(r_boundary_au: f64, particle_mass_gev: f64) -> Self {
        Self {
            r_boundary_au,
            particle_mass_gev,
        }
    }

    /// Compute the modulation potential phi (GV) at heliocentric radius r_au.
    ///
    /// phi(r) = integral from r to R_B of V_sw(r') / (3 * kappa_rr(r', R)) dr'
    ///
    /// Uses 32-point Gauss-Legendre quadrature for the radial integral.
    /// The integrand is V_sw(r) / (3 * kappa_rr(r)), where kappa_rr is the
    /// radial component of the diffusion tensor.
    pub fn modulation_potential(
        &self,
        r_au: f64,
        v_sw_profile: &dyn Fn(f64) -> f64,
        kappa_rr_profile: &dyn Fn(f64) -> f64,
    ) -> f64 {
        let gl = GaussLegendre::new(32).expect("GaussLegendre init failed");
        // Integrate from r_au to r_boundary_au
        gl.integrate(r_au, self.r_boundary_au, |r| {
            let v = v_sw_profile(r);
            let k = kappa_rr_profile(r).max(1e-30);
            v / (3.0 * k)
        })
    }

    /// Apply the FGA spectrum transformation.
    ///
    /// Returns the differential intensity at energy E_kin and modulation phi,
    /// given the local interstellar spectrum (LIS) value j_ism at E_kin + phi.
    ///
    /// J(r, E) = j_ism(E+phi) * (E^2 + 2*m*E) / ((E+phi)^2 + 2*m*(E+phi))
    pub fn modulate_spectrum(&self, phi_gev: f64, e_kin_gev: f64, j_ism: f64) -> f64 {
        let m = self.particle_mass_gev;
        let e = e_kin_gev;
        let e_phi = e + phi_gev;
        if e_phi <= 0.0 || e <= 0.0 {
            return 0.0;
        }
        let numerator = e * e + 2.0 * m * e;
        let denominator = e_phi * e_phi + 2.0 * m * e_phi;
        if denominator < 1e-30 {
            return 0.0;
        }
        j_ism * numerator / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_decreases_with_radius() {
        // phi(r) should decrease with r (integral shrinks as lower limit increases)
        let proxy = ForceFieldProxy::new(157.0, 0.938272);
        let v_sw = |_r: f64| 400.0; // km/s constant
        let kappa_rr = |r: f64| 4.5e-5 * r; // AU^2/s, grows with r
        let phi_inner = proxy.modulation_potential(1.0, &v_sw, &kappa_rr);
        let phi_outer = proxy.modulation_potential(50.0, &v_sw, &kappa_rr);
        assert!(
            phi_inner > phi_outer,
            "phi should decrease with radius: phi_inner={phi_inner:.4}, phi_outer={phi_outer:.4}"
        );
    }

    #[test]
    fn test_modulated_spectrum_recovers_ism_at_zero_phi() {
        let proxy = ForceFieldProxy::new(157.0, 0.938272);
        let j_ism = 1.23_f64;
        let result = proxy.modulate_spectrum(0.0, 1.0, j_ism);
        // At phi=0: J = j_ism * (E^2 + 2mE) / (E^2 + 2mE) = j_ism
        assert!(
            (result - j_ism).abs() < 1e-10,
            "FGA should recover LIS at phi=0"
        );
    }

    #[test]
    fn test_modulated_spectrum_suppressed_at_large_phi() {
        let proxy = ForceFieldProxy::new(157.0, 0.938272);
        let j_ism = 1.0_f64;
        // Large phi suppresses low-energy particles
        let j_mod = proxy.modulate_spectrum(1.0, 0.1, j_ism);
        assert!(j_mod < j_ism, "large phi should suppress spectrum");
    }
}
