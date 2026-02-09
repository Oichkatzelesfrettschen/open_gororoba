//! Gravitational wave physics for compact binary systems.
//!
//! Implements the quadrupole formula and post-Newtonian corrections for
//! gravitational wave strain from inspiraling binaries.
//!
//! Key formulas:
//!   Chirp mass:       M_c = (m1 m2)^{3/5} / (m1+m2)^{1/5}
//!   Strain amplitude: h = (4/D) (G M_c/c^2)^{5/3} (pi f/c)^{2/3}
//!   Frequency evolution: df/dt = (96/5) pi^{8/3} (G M_c/c^3)^{5/3} f^{11/3}
//!   Time to coalescence: tau = (5/256) (G M_c/c^3)^{-5/3} (pi f)^{-8/3}
//!
//! References:
//!   - Peters & Mathews (1963): Phys. Rev. 131, 435
//!   - Peters (1964): Phys. Rev. 136, B1224
//!   - Blanchet (2014): Living Reviews in Relativity
//!   - Berti, Cardoso, Starinets (2009): Class. Quantum Grav. 26, 163001
//!   - LIGO Scientific Collaboration (2016): Phys. Rev. Lett. 116, 061102

use crate::constants::*;
use std::f64::consts::PI;

// ============================================================================
// Binary system parameters
// ============================================================================

/// Parameters for a compact binary system.
#[derive(Debug, Clone, Copy)]
pub struct BinarySystem {
    /// Primary mass [g]
    pub m1: f64,
    /// Secondary mass [g]
    pub m2: f64,
    /// Primary dimensionless spin (0 to 1)
    pub chi1: f64,
    /// Secondary dimensionless spin (0 to 1)
    pub chi2: f64,
    /// Luminosity distance [cm]
    pub distance: f64,
    /// Orbital inclination [rad] (0 = face-on)
    pub inclination: f64,
}

impl BinarySystem {
    /// Total mass [g].
    pub fn m_total(&self) -> f64 {
        self.m1 + self.m2
    }

    /// Reduced mass [g].
    pub fn mu(&self) -> f64 {
        self.m1 * self.m2 / self.m_total()
    }

    /// Symmetric mass ratio eta = mu/M (0 < eta <= 1/4).
    pub fn eta(&self) -> f64 {
        self.mu() / self.m_total()
    }

    /// Mass ratio q = m2/m1 (q <= 1 by convention, m1 >= m2).
    pub fn q(&self) -> f64 {
        self.m2 / self.m1
    }

    /// Chirp mass [g].
    pub fn chirp_mass(&self) -> f64 {
        chirp_mass(self.m1, self.m2)
    }

    /// Create a binary neutron star system from solar masses and Mpc distance.
    pub fn bns(m1_solar: f64, m2_solar: f64, d_mpc: f64) -> Self {
        Self {
            m1: m1_solar * M_SUN_CGS,
            m2: m2_solar * M_SUN_CGS,
            chi1: 0.0,
            chi2: 0.0,
            distance: d_mpc * MPC_CGS,
            inclination: 0.0,
        }
    }

    /// Create a binary black hole system from solar masses and Mpc distance.
    pub fn bbh(m1_solar: f64, m2_solar: f64, chi1: f64, chi2: f64, d_mpc: f64) -> Self {
        Self {
            m1: m1_solar * M_SUN_CGS,
            m2: m2_solar * M_SUN_CGS,
            chi1,
            chi2,
            distance: d_mpc * MPC_CGS,
            inclination: 0.0,
        }
    }
}

// ============================================================================
// Chirp mass
// ============================================================================

/// Chirp mass M_c = M_total * eta^{3/5} [g].
///
/// The chirp mass is the primary parameter measurable from GW signals.
/// For equal-mass systems, M_c = M_total / 2^{1/5}.
pub fn chirp_mass(m1: f64, m2: f64) -> f64 {
    let m_total = m1 + m2;
    let eta = (m1 * m2) / (m_total * m_total);
    m_total * eta.powf(0.6)
}

/// Chirp mass in geometric units [s].
///
/// M_c^{geo} = G M_c / c^3
fn chirp_mass_geometric(m_c: f64) -> f64 {
    G_CGS * m_c / (C_CGS * C_CGS * C_CGS)
}

// ============================================================================
// Strain amplitude
// ============================================================================

/// Leading-order (Newtonian) GW strain amplitude.
///
/// h_0 = (4/D) (G M_c / c^2)^{5/3} (pi f / c)^{2/3}
///
/// This is the "sky-averaged" strain for circular orbits.
pub fn strain_amplitude(m_c: f64, f: f64, distance: f64) -> f64 {
    if distance <= 0.0 || f <= 0.0 {
        return 0.0;
    }
    let gmc_c2 = G_CGS * m_c / (C_CGS * C_CGS);
    let factor1 = gmc_c2.powf(5.0 / 3.0);
    let factor2 = (PI * f / C_CGS).powf(2.0 / 3.0);
    (4.0 / distance) * factor1 * factor2
}

/// Plus and cross GW polarizations.
///
/// h_+ = h_0 (1 + cos^2 i) / 2 * cos(2 Phi)
/// h_x = h_0 cos(i) * sin(2 Phi)
pub fn polarizations(h0: f64, inclination: f64, phase: f64) -> (f64, f64) {
    let cos_i = inclination.cos();
    let h_plus = h0 * (1.0 + cos_i * cos_i) / 2.0 * (2.0 * phase).cos();
    let h_cross = h0 * cos_i * (2.0 * phase).sin();
    (h_plus, h_cross)
}

/// 1PN-corrected strain amplitude.
///
/// h = h_0 * [1 + (55/48 - 55 eta/16) x]
///
/// where x = (pi M_total f / c^3)^{2/3} is the PN expansion parameter.
pub fn strain_amplitude_1pn(m_c: f64, eta: f64, f: f64, distance: f64) -> f64 {
    let h0 = strain_amplitude(m_c, f, distance);
    let m_total = m_c / eta.powf(0.6);
    let m_geo = G_CGS * m_total / (C_CGS * C_CGS * C_CGS);
    let x = (PI * m_geo * f).powf(2.0 / 3.0);
    let correction = 1.0 + (55.0 / 48.0 - 55.0 * eta / 16.0) * x;
    h0 * correction
}

// ============================================================================
// Frequency evolution
// ============================================================================

/// GW frequency derivative (chirp rate) [Hz/s].
///
/// df/dt = (96/5) pi^{8/3} (G M_c/c^3)^{5/3} f^{11/3}
pub fn frequency_derivative(m_c: f64, f: f64) -> f64 {
    let m_c_geo = chirp_mass_geometric(m_c);
    (96.0 / 5.0) * PI.powf(8.0 / 3.0) * m_c_geo.powf(5.0 / 3.0) * f.powf(11.0 / 3.0)
}

/// Time to coalescence [s].
///
/// tau = (5/256) (G M_c/c^3)^{-5/3} (pi f)^{-8/3}
pub fn time_to_coalescence(m_c: f64, f: f64) -> f64 {
    if f <= 0.0 {
        return f64::INFINITY;
    }
    let m_c_geo = chirp_mass_geometric(m_c);
    (5.0 / 256.0) * m_c_geo.powf(-5.0 / 3.0) * (PI * f).powf(-8.0 / 3.0)
}

/// Orbital separation from Kepler's third law [cm].
///
/// a^3 = G M_total / (4 pi^2 f_orb^2) where f_GW = 2 f_orb.
pub fn orbital_separation(m_total: f64, f: f64) -> f64 {
    if f <= 0.0 {
        return f64::INFINITY;
    }
    let f_orb = f / 2.0;
    let a_cubed = G_CGS * m_total / (4.0 * PI * PI * f_orb * f_orb);
    a_cubed.cbrt()
}

/// GW frequency at ISCO [Hz].
///
/// f_ISCO = 1 / (pi M_geo * (r_ISCO/M)^{3/2})
///
/// For Schwarzschild (r_ISCO = 6M): f_ISCO = c^3 / (6^{3/2} pi G M).
pub fn frequency_isco(m_total: f64, r_isco_over_m: f64) -> f64 {
    let m_geo = G_CGS * m_total / (C_CGS * C_CGS * C_CGS);
    1.0 / (PI * m_geo * r_isco_over_m.powf(1.5))
}

// ============================================================================
// Post-Newtonian phase
// ============================================================================

/// GW phase with 2.5PN corrections [rad].
///
/// Includes PN coefficients from Blanchet (2014).
pub fn phase_2p5pn(m_c: f64, eta: f64, f: f64, t_c: f64, phi_c: f64) -> f64 {
    let m_total = m_c / eta.powf(0.6);
    let m_geo = G_CGS * m_total / (C_CGS * C_CGS * C_CGS);

    let v = (PI * m_geo * f).cbrt();
    let v2 = v * v;
    let v3 = v2 * v;

    let eta2 = eta * eta;

    // PN corrections to the phase
    let psi_n = 1.0; // Newtonian
    let psi_1pn = (3715.0 / 756.0 + 55.0 * eta / 9.0) * v2;
    let psi_15pn = -16.0 * PI * v3;
    let psi_2pn =
        (15_293_365.0 / 508_032.0 + 27_145.0 * eta / 504.0 + 3085.0 * eta2 / 72.0) * v2 * v2;
    let psi_25pn = PI * (38_645.0 / 756.0 - 65.0 * eta / 9.0) * (1.0 + 3.0 * v.ln()) * v3 * v2;

    let pn_sum = psi_n + psi_1pn + psi_15pn + psi_2pn + psi_25pn;
    let psi_leading = (3.0 / 128.0) / (PI * m_geo * f).powf(5.0 / 3.0);

    2.0 * PI * f * t_c - phi_c - PI / 4.0 + psi_leading * pn_sum
}

// ============================================================================
// Ringdown (quasi-normal modes)
// ============================================================================

/// Dominant (l=2, m=2) QNM frequency for Schwarzschild BH [Hz].
///
/// omega_22 = 0.3737 / M_geo (geometric units)
/// f_22 = omega_22 / (2 pi)
pub fn qnm_frequency_schwarzschild(m_final: f64) -> f64 {
    let m_geo = G_CGS * m_final / (C_CGS * C_CGS * C_CGS);
    0.3737 / (2.0 * PI * m_geo)
}

/// QNM damping time for Schwarzschild BH [s].
///
/// tau_22 = M_geo / 0.0890
pub fn qnm_damping_time_schwarzschild(m_final: f64) -> f64 {
    let m_geo = G_CGS * m_final / (C_CGS * C_CGS * C_CGS);
    m_geo / 0.0890
}

/// Ringdown strain at time t after merger.
///
/// h(t) = A exp(-t/tau) cos(omega t + phi)
pub fn ringdown_strain(amplitude: f64, omega_qnm: f64, tau: f64, t: f64, phi: f64) -> f64 {
    if t < 0.0 {
        return 0.0;
    }
    amplitude * (-t / tau).exp() * (omega_qnm * t + phi).cos()
}

// ============================================================================
// Energy and luminosity
// ============================================================================

/// GW luminosity [erg/s].
///
/// L_GW = (32/5) (c^5/G) eta^2 (M omega)^{10/3}
///
/// where omega = pi f is the angular GW frequency.
pub fn luminosity(m_total: f64, eta: f64, f: f64) -> f64 {
    let m_geo = G_CGS * m_total / (C_CGS * C_CGS * C_CGS);
    let omega = PI * f;
    let m_omega = m_geo * omega;
    let c5_g = C_CGS.powi(5) / G_CGS;
    (32.0 / 5.0) * c5_g * eta * eta * m_omega.powf(10.0 / 3.0)
}

/// Total energy radiated in GWs [erg].
///
/// E_rad = epsilon * M c^2 where epsilon ~ 0.0559 eta^2 (NR fit).
pub fn energy_radiated(m_total: f64, eta: f64) -> f64 {
    let epsilon = 0.0559 * eta * eta;
    epsilon * m_total * C_CGS * C_CGS
}

/// Characteristic strain for detector sensitivity plots.
///
/// h_c = h * sqrt(N_cycles) where N_cycles = f * tau.
pub fn characteristic_strain(m_c: f64, f: f64, distance: f64) -> f64 {
    let h0 = strain_amplitude(m_c, f, distance);
    let tau = time_to_coalescence(m_c, f);
    let n_cycles = f * tau;
    h0 * n_cycles.sqrt()
}

// ============================================================================
// Waveform generation
// ============================================================================

/// A single point in a gravitational waveform.
#[derive(Debug, Clone, Copy)]
pub struct WaveformPoint {
    /// Time [s]
    pub t: f64,
    /// GW frequency [Hz]
    pub f: f64,
    /// Plus polarization strain
    pub h_plus: f64,
    /// Cross polarization strain
    pub h_cross: f64,
    /// Orbital phase [rad]
    pub phase: f64,
}

/// Generate an inspiral waveform from f_low to f_high.
///
/// Integrates frequency evolution with Euler stepping and computes
/// 1PN-corrected strain at each time step.
///
/// Returns early if > 10M points (safety limit).
pub fn generate_inspiral_waveform(
    binary: &BinarySystem,
    f_low: f64,
    f_high: f64,
    dt: f64,
) -> Vec<WaveformPoint> {
    let m_c = binary.chirp_mass();
    let eta = binary.eta();
    let mut waveform = Vec::new();

    let mut f = f_low;
    let mut t = 0.0;
    let mut phase = 0.0;

    while f < f_high && f > 0.0 {
        let h0 = strain_amplitude_1pn(m_c, eta, f, binary.distance);
        let (h_plus, h_cross) = polarizations(h0, binary.inclination, phase);

        waveform.push(WaveformPoint {
            t,
            f,
            h_plus,
            h_cross,
            phase,
        });

        let df_dt = frequency_derivative(m_c, f);
        f += df_dt * dt;
        phase += PI * f * dt;
        t += dt;

        if waveform.len() > 10_000_000 {
            break;
        }
    }

    waveform
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    // -- Chirp mass --

    #[test]
    fn test_chirp_mass_equal_mass() {
        // For m1 = m2 = m: M_c = 2m * (1/4)^{3/5} = 2m * 0.4353 = 0.8706m
        let m = M_SUN_CGS;
        let mc = chirp_mass(m, m);
        let expected = 2.0 * m * 0.25_f64.powf(0.6);
        assert!((mc - expected).abs() / expected < TOL);
    }

    #[test]
    fn test_chirp_mass_asymmetric() {
        // 10 + 1 M_sun: M_c = 11 * (10/121)^{3/5} M_sun
        let mc = chirp_mass(10.0 * M_SUN_CGS, M_SUN_CGS);
        assert!(mc > 0.0 && mc < 11.0 * M_SUN_CGS);
    }

    #[test]
    fn test_chirp_mass_symmetric() {
        // Symmetric: chirp_mass(m1,m2) = chirp_mass(m2,m1)
        let mc1 = chirp_mass(10.0, 20.0);
        let mc2 = chirp_mass(20.0, 10.0);
        assert!((mc1 - mc2).abs() < TOL);
    }

    // -- Binary system --

    #[test]
    fn test_binary_eta_equal_mass() {
        let b = BinarySystem::bns(1.4, 1.4, 40.0);
        assert!((b.eta() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_binary_bns_distance() {
        let b = BinarySystem::bns(1.4, 1.4, 40.0);
        assert!((b.distance - 40.0 * MPC_CGS).abs() < 1.0);
    }

    // -- Strain --

    #[test]
    fn test_strain_gw150914() {
        // GW150914: M_c ~ 28.3 M_sun, f ~ 100 Hz, D ~ 410 Mpc
        // h ~ 1e-21 (order of magnitude)
        let mc = 28.3 * M_SUN_CGS;
        let d = 410.0 * MPC_CGS;
        let h = strain_amplitude(mc, 100.0, d);
        assert!(h > 1e-22 && h < 1e-20, "h = {h}");
    }

    #[test]
    fn test_strain_inversely_proportional_to_distance() {
        let mc = 10.0 * M_SUN_CGS;
        let h1 = strain_amplitude(mc, 100.0, 100.0 * MPC_CGS);
        let h2 = strain_amplitude(mc, 100.0, 200.0 * MPC_CGS);
        assert!((h1 / h2 - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_polarizations_face_on() {
        // Face-on (i=0): h_+ = h_0 cos(2*phase), h_x = h_0 sin(2*phase)
        let (hp, hx) = polarizations(1.0, 0.0, 0.0);
        assert!((hp - 1.0).abs() < TOL);
        assert!(hx.abs() < TOL);
    }

    #[test]
    fn test_polarizations_edge_on() {
        // Edge-on (i=pi/2): h_+ = h_0/2, h_x = 0
        let (hp, hx) = polarizations(1.0, PI / 2.0, 0.0);
        assert!((hp - 0.5).abs() < TOL);
        assert!(hx.abs() < 1e-15);
    }

    // -- Frequency evolution --

    #[test]
    fn test_frequency_derivative_positive() {
        let df = frequency_derivative(10.0 * M_SUN_CGS, 100.0);
        assert!(df > 0.0, "frequency should increase (inspiral)");
    }

    #[test]
    fn test_frequency_derivative_scales_with_f() {
        // df/dt ~ f^{11/3}
        let df1 = frequency_derivative(10.0 * M_SUN_CGS, 10.0);
        let df2 = frequency_derivative(10.0 * M_SUN_CGS, 20.0);
        let expected_ratio = 2.0_f64.powf(11.0 / 3.0);
        let ratio = df2 / df1;
        assert!(
            (ratio - expected_ratio).abs() / expected_ratio < 0.01,
            "ratio = {ratio}, expected {expected_ratio}"
        );
    }

    #[test]
    fn test_time_to_coalescence_gw150914() {
        // GW150914 entered LIGO band at ~35 Hz, ~0.2 s before merger
        let mc = 28.3 * M_SUN_CGS;
        let tau = time_to_coalescence(mc, 35.0);
        // Should be order of seconds
        assert!(tau > 0.01 && tau < 10.0, "tau = {tau} s");
    }

    #[test]
    fn test_time_to_coalescence_decreases_with_f() {
        let mc = 10.0 * M_SUN_CGS;
        let tau1 = time_to_coalescence(mc, 10.0);
        let tau2 = time_to_coalescence(mc, 100.0);
        assert!(tau2 < tau1);
    }

    // -- Orbital separation --

    #[test]
    fn test_orbital_separation_kepler() {
        // Earth-Sun: f_orb ~ 1/(365.25*86400) Hz, a ~ 1 AU
        let m = M_SUN_CGS;
        let f = 2.0 / (365.25 * 86400.0); // f_GW = 2 * f_orb
        let a = orbital_separation(m, f);
        let au = 1.496e13; // AU in cm
        assert!((a - au).abs() / au < 0.01, "a = {a}, expected {au}");
    }

    // -- ISCO frequency --

    #[test]
    fn test_isco_frequency_10_solar() {
        // f_ISCO for 10 M_sun Schwarzschild ~ 220 Hz
        let f = frequency_isco(10.0 * M_SUN_CGS, 6.0);
        assert!(f > 100.0 && f < 500.0, "f_ISCO = {f}");
    }

    #[test]
    fn test_isco_frequency_inversely_proportional_to_mass() {
        let f1 = frequency_isco(10.0 * M_SUN_CGS, 6.0);
        let f2 = frequency_isco(20.0 * M_SUN_CGS, 6.0);
        assert!((f1 / f2 - 2.0).abs() < 0.01);
    }

    // -- QNM --

    #[test]
    fn test_qnm_frequency_10_solar() {
        // For 10 M_sun: f_QNM ~ 1200 Hz
        let f = qnm_frequency_schwarzschild(10.0 * M_SUN_CGS);
        assert!(f > 500.0 && f < 3000.0, "f_QNM = {f}");
    }

    #[test]
    fn test_qnm_above_isco() {
        // Ringdown frequency should be above ISCO frequency
        let m = 10.0 * M_SUN_CGS;
        let f_isco = frequency_isco(m, 6.0);
        let f_qnm = qnm_frequency_schwarzschild(m);
        assert!(f_qnm > f_isco, "QNM should be above ISCO");
    }

    #[test]
    fn test_qnm_damping_time_positive() {
        let tau = qnm_damping_time_schwarzschild(10.0 * M_SUN_CGS);
        assert!(tau > 0.0 && tau.is_finite());
    }

    #[test]
    fn test_ringdown_decays() {
        let h0 = ringdown_strain(1.0, 1000.0, 0.01, 0.0, 0.0);
        let h1 = ringdown_strain(1.0, 1000.0, 0.01, 0.1, 0.0);
        assert!(h1.abs() < h0.abs(), "ringdown should decay");
    }

    #[test]
    fn test_ringdown_before_merger_zero() {
        let h = ringdown_strain(1.0, 1000.0, 0.01, -1.0, 0.0);
        assert!((h).abs() < TOL);
    }

    // -- Energy --

    #[test]
    fn test_luminosity_positive() {
        let l = luminosity(20.0 * M_SUN_CGS, 0.25, 100.0);
        assert!(l > 0.0);
    }

    #[test]
    fn test_energy_radiated_order_of_magnitude() {
        // Equal mass 30+30 M_sun: E ~ 0.05 * eta^2 * M c^2
        // eta = 0.25, so epsilon ~ 0.0559 * 0.0625 = 3.49e-3
        // E ~ 3.5e-3 * 60 * M_sun * c^2 ~ 3.5e-3 * 60 * 1.8e54 ~ 3.8e53 erg
        let e = energy_radiated(60.0 * M_SUN_CGS, 0.25);
        assert!(e > 1e52 && e < 1e55, "E = {e}");
    }

    // -- Characteristic strain --

    #[test]
    fn test_characteristic_strain_increases_with_lower_f() {
        // At lower frequency, more cycles -> higher h_c
        let mc = 10.0 * M_SUN_CGS;
        let d = 100.0 * MPC_CGS;
        let hc1 = characteristic_strain(mc, 10.0, d);
        let hc2 = characteristic_strain(mc, 100.0, d);
        assert!(hc1 > hc2, "lower f has more cycles");
    }

    // -- Waveform --

    #[test]
    fn test_waveform_frequency_increases() {
        let binary = BinarySystem::bbh(30.0, 30.0, 0.0, 0.0, 400.0);
        let wf = generate_inspiral_waveform(&binary, 20.0, 100.0, 1e-4);
        assert!(!wf.is_empty(), "waveform should not be empty");
        assert!(wf.last().unwrap().f > wf.first().unwrap().f);
    }

    #[test]
    fn test_waveform_phase_increases() {
        let binary = BinarySystem::bns(1.4, 1.4, 40.0);
        let wf = generate_inspiral_waveform(&binary, 30.0, 50.0, 1e-3);
        assert!(!wf.is_empty());
        assert!(wf.last().unwrap().phase > wf.first().unwrap().phase);
    }
}
