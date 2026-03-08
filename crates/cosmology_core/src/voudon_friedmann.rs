//! Voudon-Friedmann Cosmology.
//!
//! Implements the modified Friedmann equations with Voudon Algebraic Pressure.
//! The Voudon pressure term arises from the statistical frustration density
//! of the 256D Voudon algebra, acting as a large-scale smoothing field.

use crate::flrw::FlatLCDM;

/// Modified FLRW cosmology with Voudon Algebraic Pressure.
pub struct VoudonFriedmann {
    pub base: FlatLCDM,
    /// Global Mean Frustration Density (Phi) from 256D algebra.
    pub phi_voudon: f64,
    /// Physical coupling constant for algebraic pressure.
    pub alpha_voudon: f64,
}

impl VoudonFriedmann {
    pub fn new(base: FlatLCDM, phi_voudon: f64, alpha_voudon: f64) -> Self {
        Self {
            base,
            phi_voudon,
            alpha_voudon,
        }
    }

    /// Modified dimensionless Hubble parameter E(z) including Voudon pressure.
    ///
    /// H^2 = H0^2 * [ Omega_m(1+z)^3 + Omega_Lambda + P_voudon(z) ]
    /// We model Voudon pressure as a stiff-fluid-like term that decays slowly,
    /// representing the persistent algebraic vacuum energy.
    pub fn e_z(&self, z: f64) -> f64 {
        let lcdm_part = self.base.omega_m * (1.0 + z).powi(3) + self.base.omega_lambda();

        // Voudon Pressure Term: alpha * Phi * (1+z)^n
        // For large scale homogeneity, n=0 (constant background pressure) or n=2.
        let voudon_part = self.alpha_voudon * self.phi_voudon;

        (lcdm_part + voudon_part).sqrt()
    }

    /// Modified Hubble parameter H(z) [km/s/Mpc].
    pub fn hubble(&self, z: f64) -> f64 {
        self.base.h0 * self.e_z(z)
    }

    /// Computes the 'Smoothing Horizon' - the scale at which Voudon pressure
    /// dominates over local gravitational variance.
    pub fn smoothing_scale_mpc(&self) -> f64 {
        // Phenomenological estimate: scale ~ c / sqrt(G * P_voudon)
        // Normalized to the Hubble length.
        self.base.hubble_length() * self.alpha_voudon.sqrt()
    }
}
