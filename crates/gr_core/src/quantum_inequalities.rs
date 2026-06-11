//! Ford-Roman quantum inequality bounds and energy condition analysis.
//!
//! Quantum field theory in curved spacetime allows pointwise negative energy
//! densities (Casimir effect, squeezed states), but the Ford-Roman quantum
//! inequalities constrain the magnitude and duration of negative energy.
//!
//! This module provides:
//! - 4D quantum inequality bound (Ford & Roman 1996)
//! - Lorentzian sampling weight function
//! - Alcubierre warp bubble energy density and total energy
//! - Pfenning-Ford analysis showing warp bubbles require >M_universe energy
//! - Classical energy condition checkers (WEC, NEC, SEC, DEC)
//! - Stateful negative energy budget accumulator
//!
//! # Key results
//!
//! Ford-Roman 4D bound (Minkowski background, massless scalar):
//!   (tau_0/pi) integral <T_00> f(t) dt >= -3 / (32 pi^2 tau_0^4)
//!
//! where f(t) = (tau_0/pi) / (t^2 + tau_0^2) is a Lorentzian sampling
//! function of characteristic width tau_0.
//!
//! Pfenning-Ford (1997) applied this to the Alcubierre metric and showed:
//!   |E_warp| ~ v_s^2 R^2 / (G delta)  >> M_universe c^2
//! for any macroscopic bubble (v_s >= c, R >= 1 m).
//!
//! # References
//! - Ford, L. H. & Roman, T. A. (1996): Phys. Rev. D 53, 5496
//! - Pfenning, M. J. & Ford, L. H. (1997): Class. Quantum Grav. 14, 1743
//! - Alcubierre, M. (1994): Class. Quantum Grav. 11, L73
//! - Ford, L. H. (1978): Proc. R. Soc. London A 364, 227

use crate::constants::{C_CGS, G_CGS, L_PLANCK_CGS};
use std::f64::consts::PI;

// ============================================================================
// Constants
// ============================================================================

/// Mass of the observable universe `[g]`.
///
/// Estimated from critical density rho_c ~ 9.5e-30 g/cm^3 times the
/// comoving Hubble volume V_H ~ 4/3 pi (4.4e28 cm)^3.
/// Conventional value ~ 1.5e56 g.
pub const M_OBSERVABLE_UNIVERSE_G: f64 = 1.5e56;

// ============================================================================
// Quantum inequality bounds
// ============================================================================

/// Ford-Roman lower bound on sampled energy density in 4D Minkowski spacetime.
///
/// For a Lorentzian sampling function of width tau_0 (in seconds):
///   bound = -3 / (32 pi^2 tau_0^4)
///
/// Units: natural (hbar = c = 1).  To convert to CGS energy density
/// [erg/cm^3], multiply by hbar*c / l_P^4 or use the CGS formula directly.
///
/// The bound says that the time-averaged energy density, weighted by the
/// Lorentzian, cannot be more negative than this value.
pub fn quantum_inequality_bound_4d(tau_0: f64) -> f64 {
    assert!(tau_0 > 0.0, "sampling time must be positive");
    -3.0 / (32.0 * PI * PI * tau_0.powi(4))
}

/// Lorentzian sampling function f(t; tau_0).
///
/// f(t) = (tau_0 / pi) / (t^2 + tau_0^2)
///
/// Normalized: integral_{-inf}^{inf} f(t) dt = 1.
/// The characteristic width is tau_0 (half-width at half-maximum).
pub fn lorentzian_sampling(t: f64, tau_0: f64) -> f64 {
    assert!(tau_0 > 0.0, "sampling time must be positive");
    (tau_0 / PI) / (t * t + tau_0 * tau_0)
}

/// Check whether a given energy density over duration tau_0 satisfies
/// the Ford-Roman quantum inequality.
///
/// Returns true if the energy density is above (less negative than) the
/// QI bound, meaning no QI violation.
///
/// # Arguments
/// * `energy_density` - Time-averaged <T_00> in natural units (hbar=c=1).
/// * `duration` - Sampling time tau_0 in natural time units.
pub fn check_qi_satisfied(energy_density: f64, duration: f64) -> bool {
    energy_density >= quantum_inequality_bound_4d(duration)
}

// ============================================================================
// Warp bubble energy analysis
// ============================================================================

/// Energy density T_00 for the Alcubierre warp metric.
///
/// T_00 = -(v_s^2 / (32 pi G)) * (df/dr_s)^2 * (y^2 + z^2) / r_s^2
///
/// This is always non-positive (negative or zero), violating the WEC.
///
/// # Arguments
/// * `v_s` - Warp bubble velocity `[cm/s]`.
/// * `r_s` - Distance from bubble center `[cm]`.  Must be > 0.
/// * `y` - Transverse coordinate y `[cm]`.
/// * `z` - Transverse coordinate z `[cm]`.
/// * `df_dr` - Derivative of the shape function f(r_s) w.r.t. r_s `[1/cm]`.
///
/// # Returns
/// Energy density in CGS [erg/cm^3].  Always <= 0.
pub fn alcubierre_energy_density(v_s: f64, r_s: f64, y: f64, z: f64, df_dr: f64) -> f64 {
    assert!(r_s > 0.0, "r_s must be positive");
    let transverse_sq = y * y + z * z;
    -(v_s * v_s) / (32.0 * PI * G_CGS) * df_dr * df_dr * transverse_sq / (r_s * r_s)
}

/// Total negative energy required for an Alcubierre warp bubble.
///
/// Thin-wall approximation (Pfenning & Ford 1997, Eq. 4.15):
///   E = -(c^4 / (4G)) * (v_s/c)^2 * R^2 / delta
///
/// where R is the bubble radius and delta is the wall thickness.
/// This is always negative, and |E| grows as v_s^2 and R^2.
///
/// # Arguments
/// * `v_s` - Bubble velocity `[cm/s]`.
/// * `r_bubble` - Bubble radius `[cm]`.
/// * `delta` - Wall thickness `[cm]`.  Must be > 0.
///
/// # Returns
/// Total energy in erg (negative).
pub fn warp_bubble_total_energy(v_s: f64, r_bubble: f64, delta: f64) -> f64 {
    assert!(delta > 0.0, "wall thickness must be positive");
    let beta = v_s / C_CGS;
    -(C_CGS.powi(4) / (4.0 * G_CGS)) * beta * beta * r_bubble * r_bubble / delta
}

/// Minimum wall thickness set by the quantum inequality.
///
/// The QI forces the wall thickness delta to be at least of order the
/// Planck length.  A thinner wall would require more negative energy
/// than QFT allows.
///
/// delta_min ~ (v_s / c)^{-1/3} * l_P  (parametric estimate)
///
/// At v_s = c, delta_min ~ l_P.  At v_s >> c, wall can be thinner
/// but is always >> 0.
///
/// # Arguments
/// * `v_s` - Bubble velocity `[cm/s]`.
///
/// # Returns
/// Minimum wall thickness in cm.
pub fn warp_wall_thickness_bound(v_s: f64) -> f64 {
    assert!(v_s > 0.0, "velocity must be positive");
    let ratio = v_s / C_CGS;
    if ratio < 1e-30 {
        return L_PLANCK_CGS;
    }
    L_PLANCK_CGS / ratio.cbrt()
}

/// Ratio |E_warp| / (M_universe * c^2).
///
/// Pfenning & Ford (1997) showed that for v_s = 10c, R = 100 m,
/// delta = 100 l_P, this ratio is ~ O(10^6) -- the warp drive would
/// need millions of times the total mass-energy of the observable universe.
///
/// # Arguments
/// * `v_s` - Bubble velocity `[cm/s]`.
/// * `r_bubble` - Bubble radius `[cm]`.
/// * `delta` - Wall thickness `[cm]`.
///
/// # Returns
/// Dimensionless ratio |E_warp| / (M_universe c^2).
pub fn pfenning_ford_energy_ratio(v_s: f64, r_bubble: f64, delta: f64) -> f64 {
    let e_warp = warp_bubble_total_energy(v_s, r_bubble, delta).abs();
    let e_universe = M_OBSERVABLE_UNIVERSE_G * C_CGS * C_CGS;
    e_warp / e_universe
}

// ============================================================================
// Negative energy budget accumulator
// ============================================================================

/// Stateful accumulator for tracking negative energy density over time.
///
/// Uses Lorentzian-weighted sampling to accumulate <T_00>(t) * f(t)
/// and checks whether the Ford-Roman QI is satisfied.
#[derive(Clone, Debug)]
pub struct NegativeEnergyBudget {
    /// Sampling timescale tau_0 (seconds in natural units).
    pub tau_0: f64,
    /// Running Lorentzian-weighted integral of energy density.
    pub integrated_density: f64,
    /// Number of samples added.
    pub n_samples: u64,
}

impl NegativeEnergyBudget {
    /// Create a new budget tracker with sampling time tau_0.
    pub fn new(tau_0: f64) -> Self {
        assert!(tau_0 > 0.0, "sampling time must be positive");
        Self {
            tau_0,
            integrated_density: 0.0,
            n_samples: 0,
        }
    }

    /// Add a sample of energy density rho at time t.
    ///
    /// The sample is weighted by the Lorentzian f(t; tau_0).
    pub fn add_sample(&mut self, t: f64, rho: f64) {
        let weight = lorentzian_sampling(t, self.tau_0);
        self.integrated_density += rho * weight;
        self.n_samples += 1;
    }

    /// Check whether the accumulated energy satisfies the QI bound.
    pub fn is_qi_satisfied(&self) -> bool {
        self.integrated_density >= quantum_inequality_bound_4d(self.tau_0)
    }

    /// Magnitude of QI violation (positive if violated, zero if satisfied).
    pub fn violation_magnitude(&self) -> f64 {
        let bound = quantum_inequality_bound_4d(self.tau_0);
        if self.integrated_density < bound {
            (bound - self.integrated_density).abs()
        } else {
            0.0
        }
    }
}

// ============================================================================
// Classical energy conditions
// ============================================================================

/// Weak Energy Condition: rho >= 0 AND rho + p >= 0.
///
/// Physical meaning: every timelike observer measures non-negative energy.
/// Violated by: Casimir effect, squeezed vacuum states.
pub fn weak_energy_condition(rho: f64, p: f64) -> bool {
    rho >= 0.0 && rho + p >= 0.0
}

/// Null Energy Condition: rho + p >= 0.
///
/// Physical meaning: energy density measured by null observers is non-negative.
/// Weakest of the classical conditions.  Violated by warp drives and
/// traversable wormholes.
pub fn null_energy_condition(rho: f64, p: f64) -> bool {
    rho + p >= 0.0
}

/// Strong Energy Condition: rho + 3p >= 0 AND rho + p >= 0.
///
/// Physical meaning: gravity is always attractive (focusing).
/// Violated by: dark energy (w < -1/3), inflation.
pub fn strong_energy_condition(rho: f64, p: f64) -> bool {
    rho + 3.0 * p >= 0.0 && rho + p >= 0.0
}

/// Dominant Energy Condition: rho >= |p|.
///
/// Physical meaning: energy flux is causal (does not exceed speed of light).
/// Implies WEC.
pub fn dominant_energy_condition(rho: f64, p: f64) -> bool {
    rho >= p.abs()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    // -- Quantum inequality bound --

    #[test]
    fn qi_bound_is_negative() {
        let bound = quantum_inequality_bound_4d(1.0);
        assert!(bound < 0.0, "QI bound must be negative");
    }

    #[test]
    fn qi_bound_scales_as_tau_minus_4() {
        let b1 = quantum_inequality_bound_4d(1.0);
        let b2 = quantum_inequality_bound_4d(2.0);
        // b(2*tau) / b(tau) = (tau/(2*tau))^4 = 1/16
        let ratio = b2 / b1;
        assert!(
            (ratio - 1.0 / 16.0).abs() < TOL,
            "QI bound should scale as tau^-4: ratio = {ratio}"
        );
    }

    #[test]
    fn qi_bound_more_restrictive_at_longer_times() {
        // Longer sampling -> smaller |bound| -> less negative energy allowed
        let b_short = quantum_inequality_bound_4d(0.01);
        let b_long = quantum_inequality_bound_4d(1.0);
        assert!(b_short < b_long, "shorter tau allows more negative energy");
    }

    // -- Lorentzian sampling --

    #[test]
    fn lorentzian_peak_at_zero() {
        let peak = lorentzian_sampling(0.0, 1.0);
        let expected = 1.0 / PI; // tau_0/pi / (0 + tau_0^2) = 1/(pi * tau_0) at tau_0=1
        assert!((peak - expected).abs() < TOL);
    }

    #[test]
    fn lorentzian_is_symmetric() {
        let tau = 2.0;
        let f_pos = lorentzian_sampling(3.0, tau);
        let f_neg = lorentzian_sampling(-3.0, tau);
        assert!((f_pos - f_neg).abs() < TOL);
    }

    #[test]
    fn lorentzian_approximate_normalization() {
        // Numerical check: integral from -100*tau to 100*tau should be ~1
        let tau = 1.0;
        let n = 10_000;
        let dt = 200.0 * tau / n as f64;
        let mut integral = 0.0;
        for i in 0..n {
            let t = -100.0 * tau + (i as f64 + 0.5) * dt;
            integral += lorentzian_sampling(t, tau) * dt;
        }
        assert!(
            (integral - 1.0).abs() < 0.01,
            "Lorentzian integral = {integral}, expected ~1.0"
        );
    }

    // -- QI check --

    #[test]
    fn check_qi_positive_energy_always_passes() {
        assert!(check_qi_satisfied(1.0, 1.0));
        assert!(check_qi_satisfied(0.0, 1.0));
    }

    #[test]
    fn check_qi_exactly_at_bound() {
        let tau = 1.0;
        let bound = quantum_inequality_bound_4d(tau);
        assert!(check_qi_satisfied(bound, tau));
    }

    #[test]
    fn check_qi_violation() {
        let tau = 1.0;
        let bound = quantum_inequality_bound_4d(tau);
        assert!(!check_qi_satisfied(bound - 1.0, tau));
    }

    // -- Warp bubble energy --

    #[test]
    fn alcubierre_energy_density_is_nonpositive() {
        let rho = alcubierre_energy_density(C_CGS, 1e5, 50.0, 50.0, 1.0);
        assert!(rho <= 0.0, "Alcubierre T_00 must be <= 0");
    }

    #[test]
    fn alcubierre_energy_zero_on_axis() {
        // On axis (y=z=0), the transverse factor vanishes
        let rho = alcubierre_energy_density(C_CGS, 1e5, 0.0, 0.0, 1.0);
        assert!(rho.abs() < TOL, "T_00 on axis should be zero");
    }

    #[test]
    fn warp_energy_is_negative() {
        let e = warp_bubble_total_energy(C_CGS, 1e4, L_PLANCK_CGS * 100.0);
        assert!(e < 0.0, "warp bubble energy must be negative");
    }

    #[test]
    fn warp_energy_scales_as_v_squared() {
        let delta = L_PLANCK_CGS * 100.0;
        let r = 1e4;
        let e1 = warp_bubble_total_energy(C_CGS, r, delta);
        let e2 = warp_bubble_total_energy(2.0 * C_CGS, r, delta);
        let ratio = e2 / e1;
        assert!(
            (ratio - 4.0).abs() < TOL,
            "E should scale as v^2: ratio = {ratio}"
        );
    }

    #[test]
    fn warp_energy_scales_as_r_squared() {
        let delta = L_PLANCK_CGS * 100.0;
        let v = C_CGS;
        let e1 = warp_bubble_total_energy(v, 1e4, delta);
        let e2 = warp_bubble_total_energy(v, 2e4, delta);
        let ratio = e2 / e1;
        assert!(
            (ratio - 4.0).abs() < TOL,
            "E should scale as R^2: ratio = {ratio}"
        );
    }

    #[test]
    fn pfenning_ford_10c_100m_exceeds_universe() {
        // v_s = 10c, R = 100 m = 1e4 cm, delta = 100 l_P
        let v_s = 10.0 * C_CGS;
        let r = 1e4; // 100 meters in cm
        let delta = 100.0 * L_PLANCK_CGS;
        let ratio = pfenning_ford_energy_ratio(v_s, r, delta);
        assert!(
            ratio > 1.0,
            "10c/100m warp requires more than M_universe: ratio = {ratio}"
        );
    }

    #[test]
    fn pfenning_ford_ratio_enormous() {
        // The ratio should be astronomically large for realistic parameters.
        // Pfenning & Ford (1997) show |E_warp|/E_universe ~ O(10^6) or more
        // for v_s=10c, R=100m, delta=100 l_P.  With the exact formula
        // E = (c^4/(4G)) * beta^2 * R^2 / delta, we get ratio ~ 10^12.
        let v_s = 10.0 * C_CGS;
        let r = 1e4;
        let delta = 100.0 * L_PLANCK_CGS;
        let ratio = pfenning_ford_energy_ratio(v_s, r, delta);
        assert!(ratio > 1e6, "energy ratio should be enormous: {ratio}");
    }

    #[test]
    fn warp_wall_thickness_at_c() {
        let delta = warp_wall_thickness_bound(C_CGS);
        // At v=c, ratio=1, so delta ~ l_P
        assert!(
            (delta / L_PLANCK_CGS - 1.0).abs() < 0.01,
            "wall thickness at c should be ~ l_P: {delta}"
        );
    }

    // -- Negative energy budget --

    #[test]
    fn budget_positive_energy_satisfies_qi() {
        let mut budget = NegativeEnergyBudget::new(1.0);
        budget.add_sample(0.0, 100.0);
        assert!(budget.is_qi_satisfied());
        assert!(budget.violation_magnitude() < TOL);
    }

    #[test]
    fn budget_tracks_samples() {
        let mut budget = NegativeEnergyBudget::new(1.0);
        budget.add_sample(0.0, 1.0);
        budget.add_sample(0.5, -0.5);
        assert_eq!(budget.n_samples, 2);
    }

    // -- Energy conditions --

    #[test]
    fn wec_normal_matter() {
        // Normal dust: rho > 0, p = 0
        assert!(weak_energy_condition(1.0, 0.0));
        // Radiation: rho > 0, p = rho/3
        assert!(weak_energy_condition(3.0, 1.0));
    }

    #[test]
    fn wec_violated_by_negative_energy() {
        assert!(!weak_energy_condition(-1.0, 0.0));
    }

    #[test]
    fn nec_weakest_condition() {
        // NEC only requires rho + p >= 0; negative rho is OK if p compensates
        assert!(null_energy_condition(-0.5, 1.0));
    }

    #[test]
    fn nec_violated_by_phantom_energy() {
        // Phantom energy: w < -1, so rho + p < 0
        assert!(!null_energy_condition(1.0, -2.0));
    }

    #[test]
    fn sec_violated_by_dark_energy() {
        // Dark energy with w = -1: rho + 3p = rho - 3*rho = -2*rho < 0
        let rho = 1.0;
        let p = -rho; // w = -1
        assert!(!strong_energy_condition(rho, p));
    }

    #[test]
    fn sec_satisfied_by_radiation() {
        let rho = 3.0;
        let p = 1.0; // w = 1/3
        assert!(strong_energy_condition(rho, p));
    }

    #[test]
    fn dec_satisfied_by_dust() {
        assert!(dominant_energy_condition(1.0, 0.0));
    }

    #[test]
    fn dec_violated_by_stiff_matter() {
        // Stiff: p = rho (allowed), but p > rho violates DEC
        assert!(!dominant_energy_condition(1.0, 2.0));
    }

    #[test]
    fn dec_implies_wec() {
        // For any (rho, p) where DEC holds, WEC should also hold
        let cases = [(1.0, 0.0), (2.0, 1.0), (5.0, -3.0), (1.0, -0.5)];
        for (rho, p) in cases {
            if dominant_energy_condition(rho, p) {
                assert!(
                    weak_energy_condition(rho, p),
                    "DEC implies WEC: failed at rho={rho}, p={p}"
                );
            }
        }
    }
}
