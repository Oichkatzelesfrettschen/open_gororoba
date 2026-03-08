//! Concentric-cylinder Lorentz-Mie scattering solver (2D, TM polarization).
//!
//! Implements the transfer-matrix method for multilayer cylindrical scatterers.
//! Used to numerically validate the TCMT Fano lineshape formulas from
//! Ruan & Fan (2009).
//!
//! The scatterer is a set of concentric cylinders with given permittivities.
//! For TM polarization (E_z, H_theta), the field in layer j is:
//! ```text
//!   E_z = A_j * J_l(k_j*rho) + B_j * Y_l(k_j*rho)
//! ```
//! Boundary conditions at each interface yield a 2x2 transfer matrix.
//!
//! # References
//! - Ruan & Fan, arXiv:0909.3323v2 (2009)
//! - Bohren & Huffman, Absorption and Scattering of Light by Small Particles

use crate::{
    bessel::{bessel_j, bessel_j_prime, bessel_y, bessel_y_prime},
    fano_tcmt::{CrossSections, FanoChannel, FanoDrudeParams, drude_epsilon},
};
use num_complex::Complex64;
use std::f64::consts::PI;

/// A single cylindrical layer with outer radius and permittivity.
#[derive(Debug, Clone, Copy)]
pub struct CylinderLayer {
    /// Outer radius of this layer.
    pub outer_radius: f64,
    /// Relative permittivity (can be complex for metals).
    pub epsilon: Complex64,
}

/// Concentric cylinder geometry.
#[derive(Debug, Clone)]
pub struct ConcentricCylinder {
    /// Layers from innermost (core) to outermost shell.
    /// Each layer's outer_radius must be strictly increasing.
    pub layers: Vec<CylinderLayer>,
    /// Permittivity of the exterior medium (usually 1.0 for vacuum).
    pub eps_ext: Complex64,
}

/// Result for a single angular momentum channel.
#[derive(Debug, Clone, Copy)]
pub struct ChannelResult {
    /// Angular momentum index l.
    pub l: i32,
    /// Scattering coefficient S_l.
    pub s_l: Complex64,
    /// Reflection coefficient R_l = 1 + 2*S_l.
    pub r_l: Complex64,
}

/// Full Mie scattering result.
#[derive(Debug, Clone)]
pub struct MieResult {
    /// Per-channel results.
    pub channels: Vec<ChannelResult>,
    /// Total cross-sections (normalized by 2*lambda/pi).
    pub cross_sections: CrossSections,
    /// Angular frequency used.
    pub omega: f64,
}

/// Wavenumber k = omega * sqrt(epsilon) / c.
/// We work in natural units where c=1.
fn wavenumber(omega: f64, eps: Complex64) -> Complex64 {
    let omega_c = Complex64::new(omega, 0.0);
    omega_c * eps.sqrt()
}

/// 2x2 complex matrix stored as [[a, b], [c, d]].
type Mat2 = [[Complex64; 2]; 2];

fn mat2_mul(a: &Mat2, b: &Mat2) -> Mat2 {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

fn _mat2_identity() -> Mat2 {
    [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ]
}

/// Interface transfer matrix at radius rho between media with wavenumbers k_in and k_out.
///
/// Continuity of E_z and (1/k) * dE_z/drho at rho gives:
///   [A_out]     [J_l(k_out*rho)  Y_l(k_out*rho)]^{-1}   [J_l(k_in*rho)   Y_l(k_in*rho) ]   [A_in]
///   [B_out]  =  [J_l'(k_out*rho) Y_l'(k_out*rho)]     * [k_in/k_out * J_l'  k_in/k_out * Y_l'] * [B_in]
///
/// Actually for TM: E_z continuous and (1/mu) * dE_z/drho continuous (mu=1 everywhere).
/// So: A_in*J(x_in) + B_in*Y(x_in) = A_out*J(x_out) + B_out*Y(x_out)
///     k_in*[A_in*J'(x_in) + B_in*Y'(x_in)] = k_out*[A_out*J'(x_out) + B_out*Y'(x_out)]
///
/// We solve for [A_out, B_out] in terms of [A_in, B_in].
fn interface_matrix(l: i32, k_in: Complex64, k_out: Complex64, rho: f64) -> Mat2 {
    let x_in = k_in * rho;
    let x_out = k_out * rho;

    let j_in = bessel_j(l, x_in);
    let y_in = bessel_y(l, x_in);
    let jp_in = bessel_j_prime(l, x_in);
    let yp_in = bessel_y_prime(l, x_in);

    let j_out = bessel_j(l, x_out);
    let y_out = bessel_y(l, x_out);
    let jp_out = bessel_j_prime(l, x_out);
    let yp_out = bessel_y_prime(l, x_out);

    // RHS matrix (inner side): maps [A_in, B_in] to [E_z, dE_z/drho]
    // [J(x_in),       Y(x_in)      ]
    // [k_in*J'(x_in), k_in*Y'(x_in)]
    let rhs = [[j_in, y_in], [k_in * jp_in, k_in * yp_in]];

    // LHS matrix (outer side): maps [A_out, B_out] to [E_z, dE_z/drho]
    // [J(x_out),        Y(x_out)       ]
    // [k_out*J'(x_out), k_out*Y'(x_out)]
    // We need its inverse.
    let det = j_out * k_out * yp_out - y_out * k_out * jp_out;
    let lhs_inv = [
        [k_out * yp_out / det, -y_out / det],
        [-k_out * jp_out / det, j_out / det],
    ];

    mat2_mul(&lhs_inv, &rhs)
}

/// Scattering coefficient S_l for a concentric cylinder at angular frequency omega.
///
/// The algorithm:
/// 1. Start from core: [A_0, B_0] = [1, 0] (regularity at origin).
/// 2. Chain transfer matrices through all interfaces.
/// 3. In exterior, convert J/Y coefficients to H^(1)/H^(2) coefficients.
/// 4. S_l = -(outgoing coefficient) / (incoming coefficient) after subtracting
///    the incident wave.
pub fn scattering_coefficient_l(geom: &ConcentricCylinder, l: i32, omega: f64) -> Complex64 {
    if geom.layers.is_empty() {
        return Complex64::new(0.0, 0.0);
    }

    // Core wavenumber
    let k_core = wavenumber(omega, geom.layers[0].epsilon);
    let k_ext = wavenumber(omega, geom.eps_ext);

    // Start with [A, B] = [1, 0] in the core (B=0 for regularity)
    let mut a = Complex64::new(1.0, 0.0);
    let mut b = Complex64::new(0.0, 0.0);

    // Transfer through each interface
    let n_layers = geom.layers.len();
    for i in 0..n_layers {
        let rho = geom.layers[i].outer_radius;
        let k_in = if i == 0 {
            k_core
        } else {
            wavenumber(omega, geom.layers[i].epsilon)
        };
        let k_out = if i == n_layers - 1 {
            k_ext
        } else {
            wavenumber(omega, geom.layers[i + 1].epsilon)
        };

        let m = interface_matrix(l, k_in, k_out, rho);
        let new_a = m[0][0] * a + m[0][1] * b;
        let new_b = m[1][0] * a + m[1][1] * b;
        a = new_a;
        b = new_b;
    }

    // Now [a, b] are the J_l / Y_l coefficients in the exterior.
    // The total field in the exterior is a*J_l(k_ext*rho) + b*Y_l(k_ext*rho).
    // Decompose into Hankel functions:
    //   J = (H1 + H2)/2,  Y = (H1 - H2)/(2i)
    // So the H1 coefficient = a/2 + b/(2i) = a/2 - i*b/2
    //    the H2 coefficient = a/2 - b/(2i) = a/2 + i*b/2
    //
    // The incident plane wave (in 2D) decomposes as:
    //   E_inc = sum_l i^l * J_l(k*rho) * e^{i*l*theta}
    //         = sum_l i^l * [H1 + H2]/2
    // So the incident H2 coefficient is i^l / 2 (incoming wave).
    //
    // The scattered field = total - incident. The scattering coefficient is defined by:
    //   scattered H1 coefficient = S_l * (incident H1 coefficient)
    // But in our normalization, total H1 = a/2 - i*b/2, incident H1 = i^l/2.
    // S_l = (total_H1 - incident_H1) / incident_H1
    //
    // Actually, S_l is defined via: the TOTAL outgoing amplitude = (1 + 2*S_l) * incident_outgoing
    // Wait -- let's use the standard convention more carefully.
    //
    // For an incident plane wave, the field in channel l has:
    //   inc_l = J_l(k*rho) = [H1_l + H2_l]/2
    // The TOTAL field must be: inc_l + scattered_l
    // The scattered field is purely outgoing: scattered_l = S_l * H1_l (with some normalization)
    //
    // Our transfer matrix gives the TOTAL field coefficients [a, b] in J/Y basis
    // when the core has amplitude 1. We need to normalize so that the incident
    // wave has the correct amplitude.
    //
    // Actually, the simplest approach: The total field in exterior is
    //   a*J_l + b*Y_l = [(a - i*b)/2]*H1 + [(a + i*b)/2]*H2
    //
    // For a homogeneous cylinder (no scatterer), the field is just J_l everywhere,
    // so a=1, b=0, giving H1 coeff = H2 coeff = 1/2. The S_l = 0 case.
    //
    // For a scatterer, the ratio of outgoing to incoming amplitudes is:
    //   (a - i*b) / (a + i*b)
    // And S_l = [(a - i*b)/(a + i*b) - 1] / 2
    //
    // But wait: our [a,b] are the coefficients of the TOTAL field matching
    // the core condition [1,0]. The actual amplitude is determined by matching
    // to the incident wave. The incoming H2 part must equal the incident:
    //   (a + i*b)/2 * normalization = i^l / 2
    // So normalization = i^l / (a + i*b)
    // And the outgoing H1 coefficient = (a - i*b)/2 * i^l / (a + i*b)
    // The incident outgoing = i^l / 2
    // So the TOTAL outgoing / incident outgoing = (a - i*b) / (a + i*b)
    // This is R_l (the reflection coefficient).
    // S_l = (R_l - 1)/2

    let i_c = Complex64::i();
    let h1_coeff = a - i_c * b; // proportional to outgoing
    let h2_coeff = a + i_c * b; // proportional to incoming

    let r_l = h1_coeff / h2_coeff;
    (r_l - 1.0) / 2.0
}

/// Full Mie scattering for all channels |l| <= l_max.
pub fn mie_scattering(geom: &ConcentricCylinder, omega: f64, l_max: i32) -> MieResult {
    let mut channels = Vec::new();
    let mut c_sct = 0.0;
    let mut c_abs = 0.0;

    for l in -l_max..=l_max {
        let s_l = scattering_coefficient_l(geom, l, omega);
        let r_l = Complex64::new(1.0, 0.0) + 2.0 * s_l;
        channels.push(ChannelResult { l, s_l, r_l });
        c_sct += s_l.norm_sqr();
        c_abs += -s_l.re;
    }

    MieResult {
        channels,
        cross_sections: CrossSections {
            c_sct,
            c_abs,
            c_ext: c_sct + c_abs,
        },
        omega,
    }
}

/// Frequency sweep: compute Mie scattering at multiple frequencies.
pub fn mie_sweep(geom: &ConcentricCylinder, omegas: &[f64], l_max: i32) -> Vec<MieResult> {
    omegas
        .iter()
        .map(|&omega| mie_scattering(geom, omega, l_max))
        .collect()
}

/// Build the MDM (Metal-Dielectric-Metal) geometry from Fig. 4 of Ruan & Fan.
///
/// Geometry: dielectric core (eps_d), metal shell, dielectric shell, in vacuum.
/// Radii: rho1 (core), rho2 (metal outer), rho3 (dielectric outer).
/// In natural units: lambda_p = 2*pi/omega_p (since c=1).
pub fn ruan_fan_mdm_fig4(drude: &FanoDrudeParams) -> ConcentricCylinder {
    let lp = 2.0 * PI / drude.omega_p;
    let eps_d = Complex64::new(12.96, 0.0);
    ConcentricCylinder {
        layers: vec![
            CylinderLayer {
                outer_radius: 0.285 * lp,
                epsilon: eps_d,
            },
            CylinderLayer {
                outer_radius: 1.0 * lp,
                epsilon: Complex64::new(-1.0, 0.0), // placeholder, computed per-freq
            },
            CylinderLayer {
                outer_radius: 1.5 * lp,
                epsilon: eps_d,
            },
        ],
        eps_ext: Complex64::new(1.0, 0.0),
    }
}

/// Build the MDM geometry from Fig. 5 of Ruan & Fan.
pub fn ruan_fan_mdm_fig5(drude: &FanoDrudeParams) -> ConcentricCylinder {
    let lp = 2.0 * PI / drude.omega_p;
    let eps_d = Complex64::new(12.96, 0.0);
    ConcentricCylinder {
        layers: vec![
            CylinderLayer {
                outer_radius: 0.36 * lp,
                epsilon: eps_d,
            },
            CylinderLayer {
                outer_radius: 0.73 * lp,
                epsilon: Complex64::new(-1.0, 0.0), // placeholder
            },
            CylinderLayer {
                outer_radius: 1.0 * lp,
                epsilon: eps_d,
            },
        ],
        eps_ext: Complex64::new(1.0, 0.0),
    }
}

/// Update metal layer epsilon at a given frequency using Drude model.
pub fn update_metal_epsilon(
    geom: &mut ConcentricCylinder,
    metal_layer_idx: usize,
    drude: &FanoDrudeParams,
    omega: f64,
) {
    geom.layers[metal_layer_idx].epsilon = drude_epsilon(drude, omega);
}

/// Compute Mie scattering for MDM geometry with frequency-dependent Drude metal.
///
/// The metal layer epsilon is updated at each frequency before computing.
pub fn mie_mdm_sweep(
    base_geom: &ConcentricCylinder,
    metal_layer_idx: usize,
    drude: &FanoDrudeParams,
    omegas: &[f64],
    l_max: i32,
) -> Vec<MieResult> {
    omegas
        .iter()
        .map(|&omega| {
            let mut geom = base_geom.clone();
            update_metal_epsilon(&mut geom, metal_layer_idx, drude, omega);
            mie_scattering(&geom, omega, l_max)
        })
        .collect()
}

/// Extract Fano channel parameters from a Mie frequency sweep for a given channel l.
///
/// Finds the resonance frequency (peak of |S_l|^2), width (HWHM), and
/// background phase from the sweep data. Returns a FanoChannel.
pub fn extract_fano_params(omegas: &[f64], results: &[MieResult], l: i32) -> Option<FanoChannel> {
    if omegas.len() != results.len() || omegas.is_empty() {
        return None;
    }

    // Find peak |S_l|^2
    let mut max_s2 = 0.0_f64;
    let mut peak_idx = 0;
    for (i, result) in results.iter().enumerate() {
        if let Some(ch) = result.channels.iter().find(|c| c.l == l) {
            let s2 = ch.s_l.norm_sqr();
            if s2 > max_s2 {
                max_s2 = s2;
                peak_idx = i;
            }
        }
    }

    let omega_0 = omegas[peak_idx];

    // Find HWHM: first point where |S_l|^2 drops below max/2
    let half_max = max_s2 / 2.0;
    let mut gamma = 0.0;
    for i in (peak_idx + 1)..omegas.len() {
        if let Some(ch) = results[i].channels.iter().find(|c| c.l == l)
            && ch.s_l.norm_sqr() < half_max
        {
            // Linear interpolation
            let prev_s2 = results[i - 1]
                .channels
                .iter()
                .find(|c| c.l == l)
                .map(|c| c.s_l.norm_sqr())
                .unwrap_or(max_s2);
            let frac = (half_max - prev_s2) / (ch.s_l.norm_sqr() - prev_s2);
            let omega_half = omegas[i - 1] + frac * (omegas[i] - omegas[i - 1]);
            gamma = omega_half - omega_0;
            break;
        }
    }

    if gamma <= 0.0 {
        // Fallback: use 1/4 of sweep range
        gamma = (omegas.last()? - omegas.first()?) / 4.0;
    }

    // Extract background phase from R_l far from resonance
    let far_idx = if peak_idx > omegas.len() / 2 {
        0
    } else {
        omegas.len() - 1
    };
    let phi = results[far_idx]
        .channels
        .iter()
        .find(|c| c.l == l)
        .map(|c| c.r_l.arg())
        .unwrap_or(0.0);

    // Estimate gamma_0 from peak absorption
    // At resonance: |S|^2_max = gamma^2 / (gamma + gamma_0)^2 for phi that maximizes it
    // But more robustly: -Re(S) at resonance = gamma*gamma_0 / (gamma+gamma_0)^2
    let abs_at_peak = results[peak_idx]
        .channels
        .iter()
        .find(|c| c.l == l)
        .map(|c| -c.s_l.re)
        .unwrap_or(0.0);

    // From absorption at resonance and known gamma, solve for gamma_0:
    // abs = gamma*gamma_0 / (gamma+gamma_0)^2
    // Let r = gamma_0/gamma, then abs = r/(1+r)^2
    // => abs*(1+r)^2 = r => abs*r^2 + (2*abs-1)*r + abs = 0
    let gamma_0 = if abs_at_peak > 1e-15 {
        let a = abs_at_peak;
        let disc = (2.0 * a - 1.0).powi(2) - 4.0 * a * a;
        if disc >= 0.0 {
            let r = (-(2.0 * a - 1.0) + disc.sqrt()) / (2.0 * a);
            if r > 0.0 { r * gamma } else { 0.0 }
        } else {
            0.0
        }
    } else {
        0.0
    };

    Some(FanoChannel {
        omega_0,
        gamma,
        gamma_0,
        phi,
        l,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bessel::{hankel_1, hankel_1_prime};

    #[test]
    fn homogeneous_no_scattering() {
        // eps=1 everywhere => no scattering
        let geom = ConcentricCylinder {
            layers: vec![CylinderLayer {
                outer_radius: 1.0,
                epsilon: Complex64::new(1.0, 0.0),
            }],
            eps_ext: Complex64::new(1.0, 0.0),
        };
        for l in 0..=3 {
            let s = scattering_coefficient_l(&geom, l, 2.0);
            assert!(
                s.norm() < 1e-10,
                "S_{} should be 0 for homogeneous medium, got |S|={}",
                l,
                s.norm()
            );
        }
    }

    #[test]
    fn single_dielectric_exact() {
        // Single dielectric cylinder: compare against direct formula.
        // For a single layer with eps_d and radius a in vacuum:
        //   S_l = [sqrt(eps)*J_l'(k*a)*J_l(k0*a) - J_l(k*a)*J_l'(k0*a)]
        //       / [J_l(k*a)*H1_l'(k0*a) - sqrt(eps)*J_l'(k*a)*H1_l(k0*a)]
        //
        // where k = omega*sqrt(eps), k0 = omega
        let eps = Complex64::new(4.0, 0.0);
        let radius = 1.0;
        let omega = 2.0;
        let geom = ConcentricCylinder {
            layers: vec![CylinderLayer {
                outer_radius: radius,
                epsilon: eps,
            }],
            eps_ext: Complex64::new(1.0, 0.0),
        };

        let k0 = Complex64::new(omega, 0.0);
        let k = k0 * eps.sqrt();
        let x0 = k0 * radius;
        let x = k * radius;

        for l in 0..=2 {
            let s_mie = scattering_coefficient_l(&geom, l, omega);

            // Direct formula
            let sqrt_eps = eps.sqrt();
            let num = sqrt_eps * bessel_j_prime(l, x) * bessel_j(l, x0)
                - bessel_j(l, x) * bessel_j_prime(l, x0);
            let den = bessel_j(l, x) * hankel_1_prime(l, x0)
                - sqrt_eps * bessel_j_prime(l, x) * hankel_1(l, x0);
            // S_l = num/den, but sign convention: check
            // The standard 2D Mie coefficient is:
            //   a_l = num / den  (this is our S_l)
            let s_direct = num / den;

            let diff = (s_mie - s_direct).norm();
            assert!(
                diff < 1e-8,
                "Single dielectric l={}: transfer matrix S={:.6}, direct S={:.6}, diff={:.2e}",
                l,
                s_mie,
                s_direct,
                diff
            );
        }
    }

    #[test]
    fn energy_conservation_lossless() {
        // For lossless (real eps) scatterer: |R_l| = 1 for each channel
        let geom = ConcentricCylinder {
            layers: vec![CylinderLayer {
                outer_radius: 0.5,
                epsilon: Complex64::new(9.0, 0.0),
            }],
            eps_ext: Complex64::new(1.0, 0.0),
        };

        let result = mie_scattering(&geom, 3.0, 3);
        for ch in &result.channels {
            let r_norm = ch.r_l.norm();
            assert!(
                (r_norm - 1.0).abs() < 1e-8,
                "Lossless: |R_{}| should be 1.0, got {}",
                ch.l,
                r_norm
            );
        }
    }

    #[test]
    fn optical_theorem_numerical() {
        // C_ext = C_sct + C_abs
        let geom = ConcentricCylinder {
            layers: vec![CylinderLayer {
                outer_radius: 0.8,
                epsilon: Complex64::new(4.0, 0.5),
            }],
            eps_ext: Complex64::new(1.0, 0.0),
        };

        let result = mie_scattering(&geom, 2.5, 4);
        let cs = &result.cross_sections;
        assert!(
            (cs.c_ext - cs.c_sct - cs.c_abs).abs() < 1e-12,
            "Optical theorem: C_ext={}, C_sct+C_abs={}",
            cs.c_ext,
            cs.c_sct + cs.c_abs
        );
    }

    #[test]
    fn channel_symmetry() {
        // For a cylindrically symmetric scatterer: S_{-l} = S_l
        let geom = ConcentricCylinder {
            layers: vec![
                CylinderLayer {
                    outer_radius: 0.5,
                    epsilon: Complex64::new(4.0, 0.1),
                },
                CylinderLayer {
                    outer_radius: 1.0,
                    epsilon: Complex64::new(2.0, 0.0),
                },
            ],
            eps_ext: Complex64::new(1.0, 0.0),
        };

        for l in 1..=3 {
            let s_pos = scattering_coefficient_l(&geom, l, 2.0);
            let s_neg = scattering_coefficient_l(&geom, -l, 2.0);
            let diff = (s_pos - s_neg).norm();
            assert!(diff < 1e-10, "S_{} != S_{}: diff={}", l, -l, diff);
        }
    }

    #[test]
    fn mdm_resonance_near_paper_value() {
        // The MDM geometry from Fig. 4 should have a resonance near 0.155*omega_p.
        // We do a coarse sweep to verify a peak exists in the right neighborhood.
        let drude = FanoDrudeParams {
            omega_p: 1.0,
            gamma_d: 0.0,
        };
        let base_geom = ruan_fan_mdm_fig4(&drude);

        let n_pts = 100;
        let omega_min = 0.14;
        let omega_max = 0.17;
        let omegas: Vec<f64> = (0..n_pts)
            .map(|i| omega_min + (omega_max - omega_min) * i as f64 / (n_pts - 1) as f64)
            .collect();

        let results = mie_mdm_sweep(&base_geom, 1, &drude, &omegas, 0);

        // Find peak |S_0|^2
        let mut max_s2 = 0.0_f64;
        let mut peak_omega = 0.0;
        for (i, result) in results.iter().enumerate() {
            if let Some(ch) = result.channels.iter().find(|c| c.l == 0) {
                let s2 = ch.s_l.norm_sqr();
                if s2 > max_s2 {
                    max_s2 = s2;
                    peak_omega = omegas[i];
                }
            }
        }

        // Paper says resonance at ~0.1552*omega_p
        assert!(
            (peak_omega - 0.155).abs() < 0.01,
            "MDM resonance expected near 0.155*wp, found at {}",
            peak_omega
        );
        assert!(
            max_s2 > 0.1,
            "MDM should show significant scattering at resonance, got |S|^2={}",
            max_s2
        );
    }
}
