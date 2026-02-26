//! Parameterized Post-Newtonian (PPN) experimental constraints.
//!
//! This module encodes experimental bounds from Solar System tests,
//! pulsar timing, torsion-balance experiments, and gravitational wave
//! observations that collectively constrain scalar-tensor gravity.
//!
//! Each constant is sourced from published experimental results and
//! compiled following Rodal (2025), arXiv:2507.09724v2, Table 1,
//! supplemented by Will (2014) Table 4 for the complete PPN parameter set.
//!
//! # PPN parameter set (Will 2014, Table 2)
//!
//! The 10 PPN parameters are: gamma, beta, xi, alpha_1, alpha_2, alpha_3,
//! zeta_1, zeta_2, zeta_3, zeta_4. In GR, gamma = beta = 1 and all others
//! vanish. In Brans-Dicke theory, beta = 1, gamma = (1+omega)/(2+omega),
//! and all preferred-frame / conservation-law parameters vanish identically.
//!
//! # Functions
//!
//! - Complete PPN parameter computation for Brans-Dicke theory (all 10)
//! - Constraint gate functions returning bool (true = constraint satisfied)
//! - Composite constraint report over all bounds
//! - Perihelion precession rate for arbitrary PPN gamma, beta
//! - Gravitational dipole radiation power for scalar-tensor theories
//! - Graviton mass / Compton wavelength calculations
//!
//! # References
//! - Rodal, J. (2025), arXiv:2507.09724v2
//! - Will, C. M. (2014), Living Rev. Relativity 17, 4, Tables 2-4
//! - Brans, C. & Dicke, R. H. (1961), Phys. Rev. 124, 925
//! - Bertotti, B. et al. (2003), Nature 425, 374
//! - Williams, J. G. et al. (2004), Phys. Rev. Lett. 93, 261101
//! - Archibald, A. M. et al. (2018), Nature 559, 73
//! - Touboul, P. et al. (2022), Phys. Rev. Lett. 129, 121102
//! - Hoyle, C. D. et al. (2004), Phys. Rev. D 70, 042004
//! - Lee, J. G. et al. (2020), Phys. Rev. Lett. 124, 101101
//! - Abbott, B. P. et al. (2017), Phys. Rev. Lett. 119, 161101
//! - Park, R. S. et al. (2017), AJ 153, 121 (MESSENGER Mercury perihelion)
//! - Lambert, S. & Le Poncin-Lafitte, C. (2011), A&A 529, A70 (VLBI gamma)
//! - Kramer, M. et al. (2021), Phys. Rev. X 11, 041050 (double pulsar)
//! - Weisberg, J. M. & Huang, Y. (2016), ApJ 829, 55 (Hulse-Taylor)
//! - Everitt, C. W. F. et al. (2011), Phys. Rev. Lett. 106, 221101 (GP-B)
//! - Shao, L. et al. (2013), Class. Quantum Grav. 30, 165019 (alpha_2)
//! - Abbott, R. et al. (2022), Phys. Rev. D 105, 122001 (GWTC-3 graviton mass)
//! - NANOGrav Collaboration (2023), ApJL 951, L11 (graviton mass from PTA)

/// Cassini Shapiro delay bound on PPN gamma deviation.
///
/// |gamma - 1| < 2.3e-5 [Bertotti et al. 2003, Nature 425, 374].
pub const CASSINI_GAMMA_DEVIATION_BOUND: f64 = 2.3e-5;

/// Cassini-derived lower bound on Brans-Dicke omega parameter.
///
/// For BD theory, gamma = (1 + omega) / (2 + omega), so
/// |gamma - 1| = 1 / (2 + omega) < 2.3e-5 => omega > 1/2.3e-5 - 2 ~ 43476.
/// Conservatively rounded to 43_000.
pub const CASSINI_OMEGA_BD_LOWER: f64 = 43_000.0;

/// Lunar Laser Ranging (LLR) Nordtvedt parameter bound.
///
/// |eta_N| < 5e-4 [Williams et al. 2004, Phys. Rev. Lett. 93, 261101].
pub const LLR_NORDTVEDT_BOUND: f64 = 5e-4;

/// PSR J0337+1715 triple-system Nordtvedt parameter bound.
///
/// |eta_N| < 2e-6 [Archibald et al. 2018, Nature 559, 73].
/// This is ~250x more stringent than LLR.
pub const PULSAR_NORDTVEDT_BOUND: f64 = 2e-6;

/// MICROSCOPE Weak Equivalence Principle (WEP) Eotvos parameter bound.
///
/// |eta| < 1.1e-15 [Touboul et al. 2022, Phys. Rev. Lett. 129, 121102].
pub const MICROSCOPE_EOTVOS_BOUND: f64 = 1.1e-15;

/// Torsion-balance Yukawa fifth-force bound at lambda ~ 1 cm.
///
/// |alpha| < 1.8e-2 [Hoyle et al. 2004, Phys. Rev. D 70, 042004].
pub const HOYLE_YUKAWA_ALPHA_BOUND: f64 = 1.8e-2;

/// Torsion-balance Yukawa range bound for alpha = 1.
///
/// lambda < 38.6 um [Lee et al. 2020, Phys. Rev. Lett. 124, 101101].
pub const LEE_YUKAWA_LAMBDA_BOUND_UM: f64 = 38.6;

/// GW170817 gravitational wave speed bound.
///
/// |c_g/c - 1| < 3e-15 [Abbott et al. 2017, Phys. Rev. Lett. 119, 161101].
pub const GW_SPEED_BOUND: f64 = 3e-15;

/// Pulsar timing bound on time variation of G.
///
/// |Gdot/G| < 1e-13 yr^-1 [Archibald et al. 2018, Nature 559, 73].
pub const GDOT_OVER_G_BOUND: f64 = 1e-13;

// ---------------------------------------------------------------------------
// Extended experimental bounds (Will 2014 Table 4 + post-2014 updates)
// ---------------------------------------------------------------------------

/// VLBI light deflection bound on PPN gamma.
///
/// |gamma - 1| < 1.2e-4 [Lambert & Le Poncin-Lafitte 2011, A&A 529, A70].
/// Less constraining than Cassini (2.3e-5) but from a different physical
/// effect (light deflection vs Shapiro delay), important for systematic
/// cross-checks.
pub const VLBI_GAMMA_DEVIATION_BOUND: f64 = 1.2e-4;

/// Mercury perihelion precession bound on PPN beta.
///
/// |beta - 1| < 8e-5 when combined with Cassini gamma.
/// [Park et al. 2017, AJ 153, 121], using MESSENGER ranging data.
/// The perihelion rate depends on (2 + 2*gamma - beta)/3 * GR_rate.
pub const MERCURY_BETA_DEVIATION_BOUND: f64 = 8e-5;

/// Preferred-frame parameter alpha_1 bound.
///
/// |alpha_1| < 1e-4 [Will 2014, Table 4; from orbital polarization of
/// binary pulsars, Damour & Esposito-Farese 1992].
pub const ALPHA1_BOUND: f64 = 1e-4;

/// Preferred-frame parameter alpha_2 bound.
///
/// |alpha_2| < 2e-9 [Shao et al. 2013, Class. Quantum Grav. 30, 165019].
/// From non-detection of spin precession of solitary pulsars around
/// the preferred-frame direction.
pub const ALPHA2_BOUND: f64 = 2e-9;

/// Preferred-frame parameter alpha_3 bound.
///
/// |alpha_3| < 4e-20 [Will 2014, Table 4; from pulsar spin-down statistics].
/// This also tests conservation of total momentum.
pub const ALPHA3_BOUND: f64 = 4e-20;

/// Whitehead parameter xi bound.
///
/// |xi| < 3.9e-9 [Shao et al. 2013, from solitary pulsar profiles].
pub const XI_BOUND: f64 = 3.9e-9;

/// Gravity Probe B geodetic precession measurement.
///
/// Geodetic drift rate = -6601.8 +/- 18.3 mas/yr vs GR prediction -6606.1 mas/yr.
/// This constrains the combination (1 + gamma)/2 to 0.25% precision.
/// [Everitt et al. 2011, Phys. Rev. Lett. 106, 221101].
pub const GPB_GEODETIC_PRECISION: f64 = 0.0028;

/// Double pulsar orbital decay agreement with GR quadrupole formula.
///
/// Observed/predicted ratio = 0.999963(+63,-63) for PSR J0737-3039A/B.
/// [Kramer et al. 2021, Phys. Rev. X 11, 041050].
/// Constrains dipole radiation: any scalar dipole contribution must
/// be < 3.7e-5 of the GR quadrupole flux.
pub const DOUBLE_PULSAR_PDOT_AGREEMENT: f64 = 3.7e-5;

/// Hulse-Taylor pulsar orbital decay agreement with GR.
///
/// Observed/predicted ratio = 0.997 +/- 0.002 for PSR B1913+16.
/// [Weisberg & Huang 2016, ApJ 829, 55].
pub const HULSE_TAYLOR_PDOT_RATIO: f64 = 0.997;

/// LIGO/Virgo O3 graviton mass upper bound [eV/c^2].
///
/// m_g < 1.27e-23 eV/c^2 [Abbott et al. 2022, Phys. Rev. D 105, 122001].
/// From modified dispersion relation analysis of GWTC-3 events.
pub const LIGO_O3_GRAVITON_MASS_EV: f64 = 1.27e-23;

/// NANOGrav 15-year graviton mass upper bound [eV].
///
/// m_g < 9.1e-31 eV [NANOGrav Collaboration 2023, ApJL 951, L11].
/// From gravitational wave background dispersion. 8 orders of magnitude
/// more stringent than LIGO but from very different frequency band.
pub const NANOGRAV_GRAVITON_MASS_EV: f64 = 9.1e-31;

/// Gravitational constant in SI units [m^3 kg^-1 s^-2].
const G_SI: f64 = 6.674_30e-11;

/// Speed of light [m/s].
const C_SI: f64 = 2.997_924_58e8;

/// Reduced Planck constant [J s].
const HBAR_SI: f64 = 1.054_571_817e-34;

// ---------------------------------------------------------------------------
// PPN parameter functions for Brans-Dicke theory
// ---------------------------------------------------------------------------

/// PPN gamma parameter for Brans-Dicke theory.
///
/// gamma = (1 + omega) / (2 + omega)
///
/// In GR (omega -> infinity), gamma -> 1.
pub fn ppn_gamma_bd(omega: f64) -> f64 {
    (1.0 + omega) / (2.0 + omega)
}

/// PPN beta parameter for Brans-Dicke theory.
///
/// beta = 1 exactly for all omega (no preferred-frame effects in BD).
pub fn ppn_beta_bd() -> f64 {
    1.0
}

/// Nordtvedt parameter for Brans-Dicke theory.
///
/// eta_N = 4*beta - gamma - 3 = 4(1) - (1+w)/(2+w) - 3 = 1/(2+omega)
///
/// This measures the strength of gravitational self-energy contributions
/// to the inertial mass (Strong Equivalence Principle violation).
pub fn nordtvedt_parameter_bd(omega: f64) -> f64 {
    1.0 / (2.0 + omega)
}

/// Effective gravitational coupling for Brans-Dicke theory.
///
/// kappa_eff = (G / phi) * (2*omega + 4) / (2*omega + 3)
///
/// In GR limit (phi=1, omega->inf), kappa_eff -> G.
/// Rodal (2025) Eq. 5.
pub fn kappa_eff_bd(phi: f64, omega: f64) -> f64 {
    assert!(phi > 0.0, "scalar field phi must be positive");
    (G_SI / phi) * (2.0 * omega + 4.0) / (2.0 * omega + 3.0)
}

// ---------------------------------------------------------------------------
// Full PPN parameter set for Brans-Dicke (Will 2014, Table 3)
// ---------------------------------------------------------------------------

/// PPN xi (Whitehead) parameter for Brans-Dicke theory.
///
/// xi = 0 identically in BD. This parameter controls preferred-location
/// effects related to anisotropy of space-time.
pub fn ppn_xi_bd() -> f64 {
    0.0
}

/// PPN alpha_1 parameter for Brans-Dicke theory.
///
/// alpha_1 = 0 identically in BD. Measures preferred-frame effects in
/// orbital dynamics (analogous to a cosmic aether wind).
pub fn ppn_alpha1_bd() -> f64 {
    0.0
}

/// PPN alpha_2 parameter for Brans-Dicke theory.
///
/// alpha_2 = 0 identically in BD. Measures preferred-frame effects in
/// spin precession (analogous to spin-orbit coupling with a cosmic frame).
pub fn ppn_alpha2_bd() -> f64 {
    0.0
}

/// PPN alpha_3 parameter for Brans-Dicke theory.
///
/// alpha_3 = 0 identically in BD. Nonzero would violate conservation
/// of total momentum.
pub fn ppn_alpha3_bd() -> f64 {
    0.0
}

/// PPN zeta_1 parameter for Brans-Dicke theory.
///
/// zeta_1 = 0 identically in BD. Nonzero would violate conservation
/// of total momentum via different coupling.
pub fn ppn_zeta1_bd() -> f64 {
    0.0
}

/// PPN zeta_2 parameter for Brans-Dicke theory.
///
/// zeta_2 = 0 identically in BD.
pub fn ppn_zeta2_bd() -> f64 {
    0.0
}

/// PPN zeta_3 parameter for Brans-Dicke theory.
///
/// zeta_3 = 0 identically in BD.
pub fn ppn_zeta3_bd() -> f64 {
    0.0
}

/// PPN zeta_4 parameter for Brans-Dicke theory.
///
/// zeta_4 = 0 identically in BD.
pub fn ppn_zeta4_bd() -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Perihelion precession
// ---------------------------------------------------------------------------

/// Perihelion precession rate per orbit in PPN formalism [radians/orbit].
///
/// delta_phi = (6 * pi * G * M) / (a * c^2 * (1 - e^2))
///             * (2 + 2*gamma - beta) / 3
///
/// In GR (gamma=1, beta=1), the factor (2+2*gamma-beta)/3 = 1.
///
/// # Arguments
/// * `semi_major_m` - Semi-major axis in meters.
/// * `eccentricity` - Orbital eccentricity (0 < e < 1).
/// * `central_mass_kg` - Central body mass in kg.
/// * `gamma` - PPN gamma parameter.
/// * `beta` - PPN beta parameter.
pub fn perihelion_precession_per_orbit(
    semi_major_m: f64,
    eccentricity: f64,
    central_mass_kg: f64,
    gamma: f64,
    beta: f64,
) -> f64 {
    assert!(semi_major_m > 0.0, "semi-major axis must be positive");
    assert!(
        (0.0..1.0).contains(&eccentricity),
        "eccentricity must be in [0, 1)"
    );
    let ppn_factor = (2.0 + 2.0 * gamma - beta) / 3.0;
    let gr_precession = 6.0 * std::f64::consts::PI * G_SI * central_mass_kg
        / (semi_major_m * C_SI * C_SI * (1.0 - eccentricity * eccentricity));
    gr_precession * ppn_factor
}

// ---------------------------------------------------------------------------
// Gravitational dipole radiation
// ---------------------------------------------------------------------------

/// Gravitational dipole radiation power for a binary in scalar-tensor theory.
///
/// P_dipole = -(2/3) * G * mu * m * (2*pi*f)^2 * (s_1 - s_2)^2
///            / (a * (1 - e^2))
///
/// where s_1, s_2 are the scalar charges (sensitivities) of each body.
/// In GR (no scalar field), s_1 = s_2 = 0 and P_dipole = 0.
/// In BD theory, s = -1/(2*(3 + 2*omega)) for weakly self-gravitating bodies.
///
/// # Arguments
/// * `sensitivity_diff` - |s_1 - s_2|, the difference in scalar charges.
/// * `orbital_freq_hz` - Orbital frequency in Hz.
/// * `reduced_mass_kg` - Reduced mass mu = m1*m2/(m1+m2) in kg.
/// * `total_mass_kg` - Total mass m = m1 + m2 in kg.
/// * `semi_major_m` - Semi-major axis in meters.
/// * `eccentricity` - Orbital eccentricity.
pub fn dipole_radiation_power(
    sensitivity_diff: f64,
    orbital_freq_hz: f64,
    reduced_mass_kg: f64,
    total_mass_kg: f64,
    semi_major_m: f64,
    eccentricity: f64,
) -> f64 {
    assert!(semi_major_m > 0.0, "semi-major axis must be positive");
    let omega_orb = 2.0 * std::f64::consts::PI * orbital_freq_hz;
    let ecc_factor = 1.0 - eccentricity * eccentricity;
    -(2.0 / 3.0)
        * G_SI
        * reduced_mass_kg
        * total_mass_kg
        * omega_orb
        * omega_orb
        * sensitivity_diff
        * sensitivity_diff
        / (semi_major_m * ecc_factor)
}

/// Scalar charge (sensitivity) for a weakly self-gravitating body in BD.
///
/// s = -1 / (2 * (3 + 2*omega))
///
/// For neutron stars, the sensitivity can be much larger (strong-field
/// effects), but this weak-field value serves as a lower bound.
pub fn bd_scalar_sensitivity(omega: f64) -> f64 {
    -1.0 / (2.0 * (3.0 + 2.0 * omega))
}

// ---------------------------------------------------------------------------
// Graviton mass and dispersion
// ---------------------------------------------------------------------------

/// Compton wavelength of a massive graviton [meters].
///
/// lambda_g = hbar / (m_g * c)
///
/// For m_g = 1.27e-23 eV/c^2 (LIGO O3): lambda_g ~ 1.6e16 m
/// For m_g = 9.1e-31 eV (NANOGrav):      lambda_g ~ 2.2e23 m
///
/// # Arguments
/// * `graviton_mass_ev` - Graviton mass in eV/c^2.
pub fn graviton_compton_wavelength(graviton_mass_ev: f64) -> f64 {
    assert!(graviton_mass_ev > 0.0, "graviton mass must be positive");
    let mass_kg = graviton_mass_ev * 1.602_176_634e-19 / (C_SI * C_SI);
    HBAR_SI / (mass_kg * C_SI)
}

/// Massive graviton dispersion relation: group velocity.
///
/// v_g / c = sqrt(1 - (m_g * c^2 / E)^2)
///
/// For gravitational waves with energy E >> m_g*c^2, v_g -> c.
/// The velocity deficit is:
///   1 - v_g/c ~ (1/2) * (m_g * c^2 / E)^2
///
/// # Arguments
/// * `graviton_mass_ev` - Graviton mass in eV/c^2.
/// * `gw_energy_ev` - Gravitational wave energy in eV (= h * f).
pub fn massive_graviton_velocity_ratio(graviton_mass_ev: f64, gw_energy_ev: f64) -> f64 {
    assert!(gw_energy_ev > 0.0, "GW energy must be positive");
    let mass_ratio = graviton_mass_ev / gw_energy_ev;
    let r2 = mass_ratio * mass_ratio;
    if r2 >= 1.0 {
        return 0.0;
    }
    (1.0 - r2).sqrt()
}

/// Yukawa fifth-force potential between two masses.
///
/// V(r) = -G * m1 * m2 / r * (1 + alpha * exp(-r / lambda))
///
/// When alpha = 0, reduces to Newtonian gravity.
pub fn yukawa_potential(r: f64, m1: f64, m2: f64, alpha: f64, lambda: f64) -> f64 {
    assert!(r > 0.0, "distance r must be positive");
    assert!(lambda > 0.0, "range lambda must be positive");
    -G_SI * m1 * m2 / r * (1.0 + alpha * (-r / lambda).exp())
}

// ---------------------------------------------------------------------------
// Constraint gate functions
// ---------------------------------------------------------------------------

/// Returns true if the Cassini Shapiro delay bound is satisfied.
///
/// Requires |gamma - 1| < 2.3e-5, i.e. omega > ~43,000.
pub fn satisfies_cassini_gamma(omega: f64) -> bool {
    let gamma = ppn_gamma_bd(omega);
    (gamma - 1.0).abs() < CASSINI_GAMMA_DEVIATION_BOUND
}

/// Returns true if the Lunar Laser Ranging Nordtvedt bound is satisfied.
///
/// Requires |eta_N| < 5e-4.
pub fn satisfies_llr_nordtvedt(omega: f64) -> bool {
    nordtvedt_parameter_bd(omega).abs() < LLR_NORDTVEDT_BOUND
}

/// Returns true if the PSR J0337+1715 pulsar Nordtvedt bound is satisfied.
///
/// Requires |eta_N| < 2e-6, the most stringent Nordtvedt constraint.
pub fn satisfies_pulsar_nordtvedt(omega: f64) -> bool {
    nordtvedt_parameter_bd(omega).abs() < PULSAR_NORDTVEDT_BOUND
}

/// Returns true if the GW170817 gravitational wave speed bound is satisfied.
///
/// In Brans-Dicke theory, tensor gravitational waves propagate at c
/// (the scalar mode has a different speed, but the tensor mode is exact).
/// This gate always passes for BD, but is included for completeness
/// when checking non-BD scalar-tensor theories.
pub fn satisfies_gw_speed_bd() -> bool {
    // BD tensor modes propagate at c exactly.
    // |c_g/c - 1| = 0 < 3e-15.
    true
}

/// Returns true if the VLBI light-deflection gamma bound is satisfied.
///
/// Requires |gamma - 1| < 1.2e-4. Less stringent than Cassini but from
/// a different physical effect (gravitational light deflection measured
/// by radio telescope baselines vs Shapiro delay).
pub fn satisfies_vlbi_gamma(omega: f64) -> bool {
    let gamma = ppn_gamma_bd(omega);
    (gamma - 1.0).abs() < VLBI_GAMMA_DEVIATION_BOUND
}

/// Returns true if the Mercury perihelion beta bound is satisfied.
///
/// For BD, beta = 1 exactly, so this always passes. Included for
/// completeness and for use with general scalar-tensor theories where
/// beta != 1.
pub fn satisfies_mercury_beta_bd() -> bool {
    (ppn_beta_bd() - 1.0).abs() < MERCURY_BETA_DEVIATION_BOUND
}

/// Returns true if the double pulsar dipole radiation bound is satisfied.
///
/// In BD theory, both neutron stars have the same weak-field sensitivity
/// s = -1/(2*(3+2*omega)), so s_1 - s_2 = 0 for equal-composition bodies.
/// The bound is that any anomalous orbital decay from dipole radiation
/// must be < 3.7e-5 of the GR quadrupole flux.
///
/// For BD, this always passes because dipole radiation is identically
/// zero for identical-composition bodies. For asymmetric systems
/// (NS-WD binaries), this provides a real constraint.
pub fn satisfies_dipole_radiation_bd() -> bool {
    true
}

/// Returns true if the Gravity Probe B geodetic precession constraint is satisfied.
///
/// GP-B measured geodetic precession to 0.28% precision (Everitt et al. 2011).
/// The precession rate depends on (1+gamma)/2, so for BD:
///   (1 + gamma)/2 = (1 + (1+w)/(2+w))/2 = (3 + 2w)/(2(2+w))
///
/// Deviation from GR (where the factor is 1):
///   |(1+gamma)/2 - 1| = |1/(2(2+w))| < 0.0028
///   => omega > 1/(2*0.0028) - 2 ~ 177
pub fn satisfies_gpb_geodetic(omega: f64) -> bool {
    let gamma = ppn_gamma_bd(omega);
    let factor = (1.0 + gamma) / 2.0;
    (factor - 1.0).abs() < GPB_GEODETIC_PRECISION
}

/// Returns true if the preferred-frame alpha_1 bound is satisfied.
///
/// In BD, alpha_1 = 0 identically. For general scalar-tensor theories,
/// orbital polarization of binary pulsars constrains |alpha_1| < 1e-4.
pub fn satisfies_alpha1_bd() -> bool {
    ppn_alpha1_bd().abs() < ALPHA1_BOUND
}

/// Returns true if the preferred-frame alpha_2 bound is satisfied.
///
/// In BD, alpha_2 = 0 identically. Solitary pulsar spin precession
/// constrains |alpha_2| < 2e-9 (Shao et al. 2013).
pub fn satisfies_alpha2_bd() -> bool {
    ppn_alpha2_bd().abs() < ALPHA2_BOUND
}

/// Returns true if the preferred-frame alpha_3 bound is satisfied.
///
/// In BD, alpha_3 = 0 identically. Pulsar spin-down statistics
/// constrain |alpha_3| < 4e-20 (Will 2014). Also tests momentum conservation.
pub fn satisfies_alpha3_bd() -> bool {
    ppn_alpha3_bd().abs() < ALPHA3_BOUND
}

/// Returns true if the Whitehead xi bound is satisfied.
///
/// In BD, xi = 0 identically. Solitary pulsar profiles constrain
/// |xi| < 3.9e-9 (Shao et al. 2013).
pub fn satisfies_xi_bd() -> bool {
    ppn_xi_bd().abs() < XI_BOUND
}

/// Returns true if the MICROSCOPE Weak Equivalence Principle bound is satisfied.
///
/// MICROSCOPE constrains the Eotvos parameter |eta| < 1.1e-15 (Touboul et al. 2022).
/// In BD theory, the WEP violation for test bodies (not self-gravitating) is
/// identically zero because BD only violates the Strong Equivalence Principle.
/// WEP violation requires composition-dependent coupling, absent in canonical BD.
pub fn satisfies_microscope_wep_bd() -> bool {
    // BD does not violate WEP for test bodies (only SEP).
    true
}

/// Composite constraint report across all PPN bounds.
///
/// 13 independent gates covering:
/// - Metric parameters: gamma (Cassini, VLBI), beta (Mercury perihelion)
/// - Nordtvedt / SEP: LLR, PSR J0337+1715 triple system
/// - Preferred-frame: alpha_1, alpha_2, alpha_3
/// - Preferred-location: xi (Whitehead)
/// - GW propagation: tensor wave speed (GW170817)
/// - Radiation: dipole (double pulsar)
/// - Geodetic precession: GP-B
/// - WEP: MICROSCOPE
#[derive(Debug, Clone, Copy)]
pub struct PPNConstraintReport {
    /// Cassini Shapiro delay: |gamma-1| < 2.3e-5
    pub cassini: bool,
    /// Lunar Laser Ranging: |eta_N| < 5e-4
    pub llr: bool,
    /// PSR J0337+1715 triple system: |eta_N| < 2e-6
    pub pulsar: bool,
    /// GW170817 tensor wave speed = c
    pub gw_speed: bool,
    /// VLBI light deflection: |gamma-1| < 1.2e-4
    pub vlbi: bool,
    /// Mercury perihelion: |beta-1| < 8e-5
    pub mercury_beta: bool,
    /// Double pulsar dipole radiation: anomalous Pdot < 3.7e-5
    pub dipole: bool,
    /// Gravity Probe B geodetic precession: 0.28% precision
    pub gpb_geodetic: bool,
    /// Preferred-frame alpha_1: |alpha_1| < 1e-4
    pub alpha1: bool,
    /// Preferred-frame alpha_2: |alpha_2| < 2e-9
    pub alpha2: bool,
    /// Preferred-frame alpha_3: |alpha_3| < 4e-20
    pub alpha3: bool,
    /// Whitehead xi: |xi| < 3.9e-9
    pub xi: bool,
    /// MICROSCOPE WEP: |eta| < 1.1e-15
    pub microscope_wep: bool,
    /// Number of constraints satisfied out of total
    pub n_satisfied: usize,
    /// Total number of constraints checked
    pub n_total: usize,
}

impl PPNConstraintReport {
    /// Returns true if all constraints are satisfied.
    pub fn all_pass(&self) -> bool {
        self.n_satisfied == self.n_total
    }
}

/// Check all PPN constraints simultaneously for a given Brans-Dicke omega.
///
/// Returns a report summarizing which bounds are satisfied.
/// Checks 13 independent constraints covering gamma, beta, Nordtvedt,
/// GW speed, VLBI deflection, Mercury perihelion, dipole radiation,
/// GP-B geodetic, preferred-frame (alpha_1/2/3, xi), and MICROSCOPE WEP.
pub fn check_all_ppn_constraints(omega: f64) -> PPNConstraintReport {
    let cassini = satisfies_cassini_gamma(omega);
    let llr = satisfies_llr_nordtvedt(omega);
    let pulsar = satisfies_pulsar_nordtvedt(omega);
    let gw_speed = satisfies_gw_speed_bd();
    let vlbi = satisfies_vlbi_gamma(omega);
    let mercury_beta = satisfies_mercury_beta_bd();
    let dipole = satisfies_dipole_radiation_bd();
    let gpb_geodetic = satisfies_gpb_geodetic(omega);
    let alpha1 = satisfies_alpha1_bd();
    let alpha2 = satisfies_alpha2_bd();
    let alpha3 = satisfies_alpha3_bd();
    let xi = satisfies_xi_bd();
    let microscope_wep = satisfies_microscope_wep_bd();

    let gates = [
        cassini,
        llr,
        pulsar,
        gw_speed,
        vlbi,
        mercury_beta,
        dipole,
        gpb_geodetic,
        alpha1,
        alpha2,
        alpha3,
        xi,
        microscope_wep,
    ];
    let n_satisfied = gates.iter().filter(|&&b| b).count();

    PPNConstraintReport {
        cassini,
        llr,
        pulsar,
        gw_speed,
        vlbi,
        mercury_beta,
        dipole,
        gpb_geodetic,
        alpha1,
        alpha2,
        alpha3,
        xi,
        microscope_wep,
        n_satisfied,
        n_total: 13,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppn_gamma_gr_limit() {
        // omega -> infinity => gamma -> 1
        let gamma = ppn_gamma_bd(1e12);
        assert!(
            (gamma - 1.0).abs() < 1e-11,
            "GR limit: gamma={gamma}, expected ~1.0"
        );
    }

    #[test]
    fn test_ppn_gamma_cassini_boundary() {
        // At omega = 43_000, |gamma - 1| = 1/(2+43000) = 1/43002 ~ 2.326e-5
        // This is ABOVE the 2.3e-5 bound, so it should barely fail
        let gamma = ppn_gamma_bd(43_000.0);
        let dev = (gamma - 1.0).abs();
        // dev ~ 2.326e-5 > 2.3e-5, so Cassini gate should fail at exactly 43k
        assert!(
            dev > 2.3e-5,
            "omega=43000 should be marginal: dev={dev:.6e}"
        );
        // At omega = 43_480 (derived from 1/2.3e-5 - 2), it should pass
        let gamma2 = ppn_gamma_bd(43_480.0);
        let dev2 = (gamma2 - 1.0).abs();
        assert!(
            dev2 < CASSINI_GAMMA_DEVIATION_BOUND,
            "omega=43480 should satisfy Cassini: dev={dev2:.6e}"
        );
    }

    #[test]
    fn test_ppn_beta_always_one() {
        assert!((ppn_beta_bd() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_nordtvedt_pulsar_boundary() {
        // |eta_N| < 2e-6 => 1/(2+omega) < 2e-6 => omega > 500_000 - 2
        let omega_min = 1.0 / PULSAR_NORDTVEDT_BOUND - 2.0;
        assert!(
            !satisfies_pulsar_nordtvedt(omega_min - 100.0),
            "Below threshold should fail"
        );
        assert!(
            satisfies_pulsar_nordtvedt(omega_min + 100.0),
            "Above threshold should pass"
        );
    }

    #[test]
    fn test_kappa_eff_gr_limit() {
        // phi=1, omega->inf => kappa_eff -> G
        let kappa = kappa_eff_bd(1.0, 1e12);
        let ratio = kappa / G_SI;
        assert!(
            (ratio - 1.0).abs() < 1e-10,
            "GR limit: kappa/G = {ratio}, expected ~1.0"
        );
    }

    #[test]
    fn test_yukawa_reduces_to_newton() {
        // alpha = 0 => V = -G*m1*m2/r (pure Newton)
        let v = yukawa_potential(1.0, 1.0, 1.0, 0.0, 1.0);
        let v_newton = -G_SI;
        assert!(
            (v - v_newton).abs() < 1e-30,
            "alpha=0 should give Newton: V={v}, expected {v_newton}"
        );
    }

    #[test]
    fn test_all_constraints_pass_at_large_omega() {
        let report = check_all_ppn_constraints(1e6);
        assert!(report.cassini, "Cassini should pass at omega=1e6");
        assert!(report.llr, "LLR should pass at omega=1e6");
        assert!(report.pulsar, "Pulsar should pass at omega=1e6");
        assert!(report.gw_speed, "GW speed should pass (always for BD)");
        assert!(report.vlbi, "VLBI should pass at omega=1e6");
        assert!(report.mercury_beta, "Mercury beta should pass (BD: beta=1)");
        assert!(report.dipole, "Dipole should pass (BD: equal bodies)");
        assert!(
            report.gpb_geodetic,
            "GP-B geodetic should pass at omega=1e6"
        );
        assert!(report.alpha1, "alpha_1 should pass (BD: alpha_1=0)");
        assert!(report.alpha2, "alpha_2 should pass (BD: alpha_2=0)");
        assert!(report.alpha3, "alpha_3 should pass (BD: alpha_3=0)");
        assert!(report.xi, "xi should pass (BD: xi=0)");
        assert!(
            report.microscope_wep,
            "MICROSCOPE WEP should pass (BD: no WEP violation)"
        );
        assert!(
            report.all_pass(),
            "All 13 constraints should pass at omega=1e6"
        );
        assert_eq!(report.n_satisfied, 13);
    }

    #[test]
    fn test_all_constraints_fail_at_small_omega() {
        let report = check_all_ppn_constraints(10.0);
        assert!(!report.cassini, "Cassini should fail at omega=10");
        assert!(!report.llr, "LLR should fail at omega=10");
        assert!(!report.pulsar, "Pulsar should fail at omega=10");
        // GW speed, mercury beta, dipole, alpha_1/2/3, xi, MICROSCOPE always pass for BD
        assert!(report.gw_speed, "GW speed always passes for BD");
        // VLBI: |gamma-1| at omega=10 is 1/12 = 0.083, much > 1.2e-4
        assert!(!report.vlbi, "VLBI should fail at omega=10");
        assert!(report.mercury_beta, "Mercury beta always passes for BD");
        assert!(report.dipole, "Dipole always passes for BD");
        // GP-B: |(1+gamma)/2 - 1| = 1/(2*12) = 0.042, > 0.0028
        assert!(!report.gpb_geodetic, "GP-B should fail at omega=10");
        assert!(report.alpha1, "alpha_1 always passes for BD");
        assert!(report.alpha2, "alpha_2 always passes for BD");
        assert!(report.alpha3, "alpha_3 always passes for BD");
        assert!(report.xi, "xi always passes for BD");
        assert!(report.microscope_wep, "MICROSCOPE always passes for BD");
        assert!(!report.all_pass(), "Not all constraints pass at omega=10");
        // Passes: gw_speed + mercury_beta + dipole + alpha_1 + alpha_2 + alpha_3 + xi + microscope = 8
        assert_eq!(report.n_satisfied, 8);
    }

    #[test]
    fn test_gw_speed_constraint_bd() {
        // BD tensor modes always propagate at c
        assert!(satisfies_gw_speed_bd());
    }

    // -- Full PPN parameter set tests --

    #[test]
    fn test_bd_preferred_frame_params_vanish() {
        // All preferred-frame and conservation-law parameters are zero in BD
        assert!(ppn_xi_bd().abs() < f64::EPSILON, "xi must be 0 in BD");
        assert!(
            ppn_alpha1_bd().abs() < f64::EPSILON,
            "alpha_1 must be 0 in BD"
        );
        assert!(
            ppn_alpha2_bd().abs() < f64::EPSILON,
            "alpha_2 must be 0 in BD"
        );
        assert!(
            ppn_alpha3_bd().abs() < f64::EPSILON,
            "alpha_3 must be 0 in BD"
        );
        assert!(
            ppn_zeta1_bd().abs() < f64::EPSILON,
            "zeta_1 must be 0 in BD"
        );
        assert!(
            ppn_zeta2_bd().abs() < f64::EPSILON,
            "zeta_2 must be 0 in BD"
        );
        assert!(
            ppn_zeta3_bd().abs() < f64::EPSILON,
            "zeta_3 must be 0 in BD"
        );
        assert!(
            ppn_zeta4_bd().abs() < f64::EPSILON,
            "zeta_4 must be 0 in BD"
        );
    }

    #[test]
    fn test_bd_params_satisfy_experimental_bounds() {
        // BD vanishing parameters must be well within experimental bounds
        assert!(ppn_xi_bd().abs() < XI_BOUND);
        assert!(ppn_alpha1_bd().abs() < ALPHA1_BOUND);
        assert!(ppn_alpha2_bd().abs() < ALPHA2_BOUND);
        assert!(ppn_alpha3_bd().abs() < ALPHA3_BOUND);
    }

    // -- Perihelion precession tests --

    #[test]
    fn test_perihelion_precession_gr_mercury() {
        // Mercury: a = 5.791e10 m, e = 0.2056, M_sun = 1.989e30 kg
        let a = 5.791e10;
        let e = 0.2056;
        let m = 1.989e30;
        let dp = perihelion_precession_per_orbit(a, e, m, 1.0, 1.0);
        // GR prediction: 42.98 arcsec/century = 5.019e-7 rad/orbit
        // (Mercury orbital period = 87.97 days, 100 yr / 87.97 d = 415 orbits)
        let arcsec_per_century = dp * 415.0 * 206265.0;
        assert!(
            (arcsec_per_century - 42.98).abs() < 1.0,
            "Mercury GR precession ~ 43 arcsec/century, got {arcsec_per_century:.2}"
        );
    }

    #[test]
    fn test_perihelion_ppn_factor() {
        let a = 1e11;
        let e = 0.2;
        let m = 2e30;
        // GR: (2+2*1-1)/3 = 1.0
        let gr = perihelion_precession_per_orbit(a, e, m, 1.0, 1.0);
        // BD with gamma=0.999: (2+2*0.999-1)/3 = (2.998)/3 = 0.99933
        let bd = perihelion_precession_per_orbit(a, e, m, 0.999, 1.0);
        let ratio = bd / gr;
        assert!((ratio - 0.999_33).abs() < 1e-4, "PPN factor ratio: {ratio}");
    }

    // -- Dipole radiation tests --

    #[test]
    fn test_dipole_radiation_zero_for_equal_sensitivities() {
        let p = dipole_radiation_power(0.0, 1e-4, 1e30, 3e30, 1e9, 0.1);
        assert!(
            p.abs() < f64::EPSILON,
            "Equal sensitivities -> zero dipole radiation"
        );
    }

    #[test]
    fn test_dipole_radiation_nonzero_for_different_sensitivities() {
        let p = dipole_radiation_power(0.01, 1e-4, 1e30, 3e30, 1e9, 0.1);
        assert!(
            p < 0.0,
            "Dipole radiation power should be negative (energy loss)"
        );
    }

    #[test]
    fn test_bd_scalar_sensitivity_decreases_with_omega() {
        let s100 = bd_scalar_sensitivity(100.0).abs();
        let s10000 = bd_scalar_sensitivity(10_000.0).abs();
        assert!(
            s100 > s10000,
            "Sensitivity should decrease with omega: |s(100)|={s100}, |s(10000)|={s10000}"
        );
    }

    #[test]
    fn test_bd_scalar_sensitivity_gr_limit() {
        let s = bd_scalar_sensitivity(1e12);
        assert!(
            s.abs() < 1e-12,
            "GR limit: sensitivity should vanish, got {s}"
        );
    }

    // -- Graviton mass tests --

    #[test]
    fn test_graviton_compton_wavelength_ligo() {
        let lambda = graviton_compton_wavelength(LIGO_O3_GRAVITON_MASS_EV);
        // Should be ~ 1.6e16 m (about 1.7 light-years)
        assert!(
            lambda > 1e15 && lambda < 1e18,
            "LIGO graviton Compton wavelength ~ 1e16 m, got {lambda:.2e}"
        );
    }

    #[test]
    fn test_graviton_compton_wavelength_nanograv() {
        let lambda = graviton_compton_wavelength(NANOGRAV_GRAVITON_MASS_EV);
        // Should be ~ 2.2e23 m (about 7 kpc)
        assert!(
            lambda > 1e22 && lambda < 1e25,
            "NANOGrav graviton Compton wavelength ~ 1e23 m, got {lambda:.2e}"
        );
    }

    #[test]
    fn test_massive_graviton_velocity_high_energy() {
        // At E >> m_g*c^2, velocity -> c
        let v_ratio = massive_graviton_velocity_ratio(1e-23, 1e10);
        assert!(
            (v_ratio - 1.0).abs() < 1e-30,
            "High energy: v_g/c should be ~1, got {v_ratio}"
        );
    }

    #[test]
    fn test_massive_graviton_velocity_low_energy() {
        // At E = m_g*c^2, velocity = 0
        let v_ratio = massive_graviton_velocity_ratio(1.0, 1.0);
        assert!(
            v_ratio.abs() < f64::EPSILON,
            "At threshold: v_g/c should be 0, got {v_ratio}"
        );
    }

    // -- VLBI and Mercury beta gate tests --

    #[test]
    fn test_vlbi_gamma_less_stringent_than_cassini() {
        const { assert!(VLBI_GAMMA_DEVIATION_BOUND > CASSINI_GAMMA_DEVIATION_BOUND) };
    }

    #[test]
    fn test_vlbi_passes_where_cassini_fails() {
        // omega = 10000: |gamma-1| = 1/10002 ~ 1e-4, fails Cassini but passes VLBI
        assert!(!satisfies_cassini_gamma(10_000.0));
        assert!(satisfies_vlbi_gamma(10_000.0));
    }

    #[test]
    fn test_mercury_beta_always_passes_for_bd() {
        // BD has beta = 1 exactly
        assert!(satisfies_mercury_beta_bd());
    }

    #[test]
    fn test_dipole_radiation_always_passes_for_bd() {
        assert!(satisfies_dipole_radiation_bd());
    }

    // -- Constraint hierarchy test --

    #[test]
    fn test_constraint_hierarchy() {
        // The constraint hierarchy for BD omega should be:
        // 1. VLBI gamma passes first (weakest gamma constraint)
        // 2. LLR Nordtvedt passes next
        // 3. Cassini gamma passes
        // 4. Pulsar Nordtvedt passes last (most stringent)
        let omega_vlbi = 1.0 / VLBI_GAMMA_DEVIATION_BOUND; // ~8333
        let omega_llr = 1.0 / LLR_NORDTVEDT_BOUND; // ~2000
        let omega_cassini = 1.0 / CASSINI_GAMMA_DEVIATION_BOUND; // ~43478
        let omega_pulsar = 1.0 / PULSAR_NORDTVEDT_BOUND; // ~500000

        // Verify ordering: VLBI and LLR are close but different parameters
        // The key hierarchy for gamma: VLBI < Cassini
        // The key hierarchy for eta_N: LLR < Pulsar
        const { assert!(VLBI_GAMMA_DEVIATION_BOUND > CASSINI_GAMMA_DEVIATION_BOUND) };
        const { assert!(LLR_NORDTVEDT_BOUND > PULSAR_NORDTVEDT_BOUND) };

        // Omega thresholds for passing
        assert!(
            omega_vlbi < omega_cassini,
            "VLBI requires smaller omega than Cassini"
        );
        assert!(
            omega_llr < omega_pulsar,
            "LLR requires smaller omega than pulsar"
        );
        assert!(
            omega_cassini < omega_pulsar,
            "Cassini requires smaller omega than pulsar"
        );
    }

    // -- New gate tests (Sprint 56) --

    #[test]
    fn test_gpb_geodetic_passes_at_large_omega() {
        // At omega = 1e6, deviation is 1/(2*1e6) ~ 5e-7 << 0.0028
        assert!(satisfies_gpb_geodetic(1e6));
    }

    #[test]
    fn test_gpb_geodetic_fails_at_small_omega() {
        // At omega = 10, |(1+gamma)/2 - 1| = 1/(2*12) = 0.042 > 0.0028
        assert!(!satisfies_gpb_geodetic(10.0));
    }

    #[test]
    fn test_gpb_geodetic_boundary() {
        // Threshold: 1/(2*(2+w)) < 0.0028 => w > 1/(2*0.0028) - 2 = 176.7
        assert!(!satisfies_gpb_geodetic(170.0));
        assert!(satisfies_gpb_geodetic(200.0));
    }

    #[test]
    fn test_preferred_frame_gates_always_pass_for_bd() {
        // BD has all preferred-frame params = 0
        assert!(satisfies_alpha1_bd());
        assert!(satisfies_alpha2_bd());
        assert!(satisfies_alpha3_bd());
        assert!(satisfies_xi_bd());
    }

    #[test]
    fn test_microscope_wep_always_passes_for_bd() {
        // BD does not violate WEP for test bodies
        assert!(satisfies_microscope_wep_bd());
    }

    #[test]
    fn test_report_has_13_gates() {
        let report = check_all_ppn_constraints(1e6);
        assert_eq!(report.n_total, 13, "Should have 13 constraint gates");
    }

    #[test]
    fn test_gpb_in_constraint_hierarchy() {
        // GP-B geodetic is less stringent than Cassini on gamma-related quantities.
        // GP-B requires omega > ~177, Cassini requires omega > ~43478.
        let omega_gpb = 1.0 / (2.0 * GPB_GEODETIC_PRECISION) - 2.0; // ~176.7
        let omega_cassini_approx = 1.0 / CASSINI_GAMMA_DEVIATION_BOUND - 2.0; // ~43476
        assert!(
            omega_gpb < omega_cassini_approx,
            "GP-B threshold ({omega_gpb:.0}) < Cassini threshold ({omega_cassini_approx:.0})"
        );
    }
}
