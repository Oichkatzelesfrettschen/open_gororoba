//! Fresnel reflection + angular optics methods on `DrudeLorentzParams`.
//!
//! Extracted from `optical_database.rs` as part of the #138 PH-MOD split.
//! Four methods covering the air/material interface at oblique incidence:
//!
//! - `fresnel_rs`: s-polarization amplitude reflection coefficient.
//! - `fresnel_rp`: p-polarization amplitude reflection coefficient.
//! - `brewster_angle`: `theta_B = atan(n2/n1)` for low-loss dielectrics.
//! - `reflectance_angular`: `(R_s, R_p) = (|r_s|^2, |r_p|^2)`.
//!
//! Reference: Born & Wolf (1999) "Principles of Optics" 7th ed. Ch. 1.

use num_complex::Complex64;

use super::DrudeLorentzParams;

impl DrudeLorentzParams {
    /// Fresnel s-polarization amplitude reflection coefficient.
    /// `r_s = (n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t)` with
    /// `n1` the (real) incident medium and `n2 = sqrt(eps)` from this
    /// material. `theta_i` is the incidence angle in radians.
    pub fn fresnel_rs(&self, omega: f64, theta_i: f64, n_incident: f64) -> Complex64 {
        let n2 = self.refractive_index(omega);
        let cos_i = theta_i.cos();
        let sin_i = theta_i.sin();
        let sin_t_sq = Complex64::new(n_incident * n_incident * sin_i * sin_i, 0.0) / (n2 * n2);
        let cos_t = (Complex64::new(1.0, 0.0) - sin_t_sq).sqrt();
        let n1_cos_i = Complex64::new(n_incident * cos_i, 0.0);
        let n2_cos_t = n2 * cos_t;
        (n1_cos_i - n2_cos_t) / (n1_cos_i + n2_cos_t)
    }

    /// Fresnel p-polarization amplitude reflection coefficient.
    /// `r_p = (n2*cos_i - n1*cos_t) / (n2*cos_i + n1*cos_t)`.
    pub fn fresnel_rp(&self, omega: f64, theta_i: f64, n_incident: f64) -> Complex64 {
        let n2 = self.refractive_index(omega);
        let cos_i = theta_i.cos();
        let sin_i = theta_i.sin();
        let sin_t_sq = Complex64::new(n_incident * n_incident * sin_i * sin_i, 0.0) / (n2 * n2);
        let cos_t = (Complex64::new(1.0, 0.0) - sin_t_sq).sqrt();
        let n2_cos_i = n2 * cos_i;
        let n1_cos_t = Complex64::new(n_incident, 0.0) * cos_t;
        (n2_cos_i - n1_cos_t) / (n2_cos_i + n1_cos_t)
    }

    /// Brewster angle in radians for p-polarized light.
    /// `theta_B = atan(n2/n1)` for non-absorbing dielectrics. Returns
    /// `None` if the material is absorbing (`Im[n] > 0.01 * Re[n]`)
    /// because the pseudo-Brewster angle in absorbing media requires
    /// numerical search.
    pub fn brewster_angle(&self, omega: f64, n_incident: f64) -> Option<f64> {
        let n = self.refractive_index(omega);
        if n.im > 0.01 * n.re {
            return None;
        }
        Some((n.re / n_incident).atan())
    }

    /// Reflectance at arbitrary angle (intensity, not amplitude).
    /// Returns `(R_s, R_p) = (|r_s|^2, |r_p|^2)`.
    pub fn reflectance_angular(&self, omega: f64, theta_i: f64, n_incident: f64) -> (f64, f64) {
        let rs = self.fresnel_rs(omega, theta_i, n_incident);
        let rp = self.fresnel_rp(omega, theta_i, n_incident);
        (rs.norm_sqr(), rp.norm_sqr())
    }
}
