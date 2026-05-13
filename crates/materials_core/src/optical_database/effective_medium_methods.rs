//! Effective-medium and dielectric-contrast methods on `DrudeLorentzParams`.
//!
//! Extracted from the monolithic `optical_database.rs` impl block as part of
//! the #138 PH-MOD split. These 4 methods cover composite-material optics:
//!
//! - `maxwell_garnett_mix`: spherical inclusions in a host (asymmetric mix).
//! - `bruggeman_mix`: symmetric two-component effective medium.
//! - `dielectric_contrast`: Delta = (eps_1 - eps_2)/(eps_1 + eps_2),
//!   appears in Clausius-Mossotti, Casimir proximity corrections, and
//!   van der Waals coefficient expressions.
//! - `plasma_screening_ratio`: omega_p_bare / omega_p_screened, a measure
//!   of how much interband transitions screen the free-electron plasmon.
//!
//! Maxwell-Garnett and Bruggeman delegate to the canonical implementations
//! in `crate::effective_medium`, which centralise the iterative root-finding
//! used by the symmetric Bruggeman branch. References: Markel (2016) JOSA A
//! 33, 1244 (Maxwell-Garnett review); Bruggeman (1935) Ann. Phys. 416, 636.

use num_complex::Complex64;

use super::DrudeLorentzParams;

impl DrudeLorentzParams {
    /// Maxwell-Garnett effective medium approximation at a given frequency.
    /// Treats `self` as the host medium and `inclusion` as spherical
    /// inclusions with volume fraction `fill_fraction`. Returns the effective
    /// dielectric function of the composite.
    pub fn maxwell_garnett_mix(
        &self,
        inclusion: &DrudeLorentzParams,
        fill_fraction: f64,
        omega: f64,
    ) -> Complex64 {
        let eps_host = self.epsilon(omega);
        let eps_inc = inclusion.epsilon(omega);
        crate::effective_medium::maxwell_garnett(eps_host, eps_inc, fill_fraction)
    }

    /// Bruggeman self-consistent effective medium at a given frequency.
    /// Treats the two materials symmetrically (no host/inclusion distinction).
    /// Volume fraction `fill_fraction` refers to `self`; `other` occupies
    /// the remainder `(1 - fill_fraction)`.
    pub fn bruggeman_mix(
        &self,
        other: &DrudeLorentzParams,
        fill_fraction: f64,
        omega: f64,
    ) -> Complex64 {
        let eps_1 = self.epsilon(omega);
        let eps_2 = other.epsilon(omega);
        crate::effective_medium::bruggeman(eps_1, eps_2, fill_fraction)
    }

    /// Dielectric contrast factor between two materials:
    /// `Delta = (eps_1 - eps_2)/(eps_1 + eps_2)`. Appears in the
    /// Clausius-Mossotti relation, Casimir proximity corrections, and
    /// van der Waals interaction coefficients. `|Delta| = 1` for perfect
    /// metal vs vacuum; `|Delta| ~ 0` for index-matched media.
    pub fn dielectric_contrast(&self, other: &DrudeLorentzParams, omega: f64) -> Complex64 {
        let eps_1 = self.epsilon(omega);
        let eps_2 = other.epsilon(omega);
        (eps_1 - eps_2) / (eps_1 + eps_2)
    }

    /// Screening ratio: bare plasma frequency divided by screened plasma
    /// frequency. Quantifies how much interband transitions screen the
    /// free-electron response. For free-electron metals (no interband),
    /// ratio ~ 1. For d-band metals (gold, copper), ratio > 1.
    ///
    /// Returns `None` if no Drude term is present or no screened-plasma
    /// crossing can be located in the 0.5 - 30 eV scan window.
    pub fn plasma_screening_ratio(&self) -> Option<f64> {
        let omega_p_bare = if let Some(ext) = &self.extended_drude {
            ext.omega_p_ev
        } else if let Some(drude) = &self.drude {
            drude.omega_p_ev
        } else {
            return None;
        };
        let screened = self.screened_plasma_ev(0.5, 30.0)?;
        Some(omega_p_bare / screened)
    }
}
