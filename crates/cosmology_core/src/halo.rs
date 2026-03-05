//! Ultra-diffuse galaxy (UDG) halo kinematics: Press-Schechter mass function
//! and NFW velocity dispersion fitting using CDG-2 (Li et al. 2025) constraints.
//!
//! # Fifth Observational Pillar
//!
//! CDG-2 is an ultra-diffuse galaxy in the Perseus cluster (z = 0.0179) with
//! at least 99.94% dark matter fraction (Li et al. 2025, arXiv:2506.15644), identified
//! via its globular cluster system. This module provides:
//!
//! 1. Press-Schechter halo mass function using the orthoplex growth factor
//! 2. NFW halo profile and Jeans-equation velocity dispersion
//! 3. UDG abundance ratio: orthoplex/LambdaCDM prediction in Perseus-like environments
//!
//! # Conventions
//!
//! All computations use h = 1 units (k in Mpc^{-1}, lengths in Mpc or kpc):
//! - Mean matter density: rho_bar_0 = Omega_m * 2.775 * 10^1^1 Msun Mpc^{-3}
//! - Critical density: rho_crit,0 = 277.5 Msun kpc^{-3}  (= 2.775 * 10^1^1 Msun Mpc^{-3})
//! - BBKS shape parameter: q = k / Omega_m  (k in Mpc^{-1}, h = 1)
//! - Normalization sphere: R_8 = 8 Mpc  (= 8 h^{-1} Mpc at h = 1)
//!
//! # References
//! - Press & Schechter (1974), ApJ 187, 425
//! - Bardeen, Bond, Kaiser & Szalay (1986), ApJ 304, 15 (BBKS transfer function)
//! - Navarro, Frenk & White (1997), ApJ 490, 493 (NFW profile)
//! - Lokas & Mamon (2001), MNRAS 321, 155 (NFW Jeans solution)
//! - Li et al. (2025), arXiv:2506.15644 (CDG-2 discovery)

use crate::{
    bounce::hubble_e_lcdm,
    gl_integrate,
    observational::compute_growth_batch,
    orthoplex_diffusion::hubble_e_orthoplex,
};

// ---------------------------------------------------------------------------
// Observational constants (CDG-2 / Perseus cluster)
// ---------------------------------------------------------------------------

/// Spherical collapse threshold (linear perturbation theory).
pub const DELTA_C: f64 = 1.686;

/// Perseus cluster redshift (Struble & Rood 1999; Hitomi Collaboration 2016).
pub const PERSEUS_Z: f64 = 0.0179;

/// CDG-2 stellar mass (L_V = 6.2 +- 3.0 * 10^6 Lsun, assuming M/L_V ~ 1).
///
/// Reference: Li et al. (2025), arXiv:2506.15644, Table 1.
pub const CDG2_STELLAR_MASS_SOLAR: f64 = 6.2e6;

/// Minimum dark matter fraction of CDG-2 (>= 99.94%).
///
/// Derived from the upper limit on the dynamical mass-to-light ratio.
/// Reference: Li et al. (2025), arXiv:2506.15644.
pub const CDG2_DM_FRACTION_MIN: f64 = 0.9994;

// ---------------------------------------------------------------------------
// Physical constants (h = 1 units)
// ---------------------------------------------------------------------------

/// Gravitational constant G in kpc (km/s)^2 Msun^{-1}.
///
/// Derived from G = 6.674 * 10^{-1}^1 m^3 kg^{-1} s^-^2,
/// 1 Msun = 1.989 * 10^3^0 kg, 1 kpc = 3.086 * 10^1^9 m.
const G_KPC_KM_S: f64 = 4.302e-6;

/// Present-day critical density (h = 1) in Msun kpc^{-3}.
///
/// rho_crit,0 = 3H_0^2/(8piG) = 2.775 * 10^1^1 Msun Mpc^{-3} / (10^3)^3 at h = 1.
const RHO_CRIT_KPC: f64 = 277.5;

/// Present-day critical density (h = 1) in Msun Mpc^{-3}.
const RHO_CRIT_MPC: f64 = 2.775e11;

// ---------------------------------------------------------------------------
// BBKS transfer function
// ---------------------------------------------------------------------------

/// BBKS (1986) CDM transfer function T(q).
///
/// T(q) = ln(1 + 2.34q)/(2.34q) * [1 + 3.89q + (16.1q)^2 + (5.46q)^3 + (6.71q)^4]^{-1/4}
///
/// Shape parameter q = k/Omega_m with k in Mpc^{-1} (h = 1 convention).
/// T -> 1 for small k (large scales), suppressed for large k (small scales).
pub fn bbks_transfer(k_mpc: f64, omega_m: f64) -> f64 {
    if omega_m <= 0.0 || k_mpc <= 0.0 {
        return 1.0;
    }
    let q = k_mpc / omega_m;
    let q234 = 2.34 * q;
    // ln(1 + 2.34q)/(2.34q) -> 1 as q -> 0
    let ln_term = if q234 < 1e-6 {
        1.0 - 0.5 * q234 + q234 * q234 / 3.0
    } else {
        (1.0 + q234).ln() / q234
    };
    let poly = 1.0
        + 3.89 * q
        + (16.1 * q).powi(2)
        + (5.46 * q).powi(3)
        + (6.71 * q).powi(4);
    ln_term * poly.powf(-0.25)
}

// ---------------------------------------------------------------------------
// Top-hat window function in Fourier space
// ---------------------------------------------------------------------------

/// Real-space top-hat window function W(x) = 3[sin(x) - x cos(x)] / x^3.
///
/// Uses the Taylor expansion 1 - x^2/10 for |x| < 10^{-3} to avoid cancellation.
#[inline]
fn top_hat_window(x: f64) -> f64 {
    if x.abs() < 1e-3 {
        1.0 - x * x / 10.0
    } else {
        3.0 * (x.sin() - x * x.cos()) / (x * x * x)
    }
}

// ---------------------------------------------------------------------------
// sigma(M): RMS mass fluctuation
// ---------------------------------------------------------------------------

/// Unnormalized sigma^2 integral I(R) over all k for a top-hat sphere of radius R (Mpc).
///
/// I(R) = integral_0^inf k^(ns+2) T^2(k) W^2(kR) dk
///
/// Evaluated via Gauss-Legendre (100 nodes) in log-k space to handle the
/// wide dynamic range: k = exp(u), dk = k du gives
///   I(R) = integral k^(ns+3) T^2(k) W^2(kR) du
fn sigma_sq_integral(r_mpc: f64, omega_m: f64, ns: f64) -> f64 {
    let u_min = 1e-5_f64.ln(); // k_min = 10^-^5 Mpc^{-1}
    let u_max = 1e5_f64.ln();  // k_max = 10^5 Mpc^{-1}
    gl_integrate(
        |u| {
            let k = u.exp();
            let t = bbks_transfer(k, omega_m);
            let w = top_hat_window(k * r_mpc);
            k.powf(ns + 3.0) * t * t * w * w
        },
        u_min,
        u_max,
        100,
    )
}

/// Comoving radius (Mpc) of the top-hat sphere enclosing mass M (Msun).
///
/// R = (3M / 4pi rho_bar_0)^{1/3}  with rho_bar_0 = Omega_m * 2.775 * 10^1^1 Msun Mpc^{-3}.
#[inline]
fn mass_to_radius_mpc(mass_solar: f64, omega_m: f64) -> f64 {
    let rho_bar = omega_m * RHO_CRIT_MPC;
    (3.0 * mass_solar / (4.0 * std::f64::consts::PI * rho_bar)).cbrt()
}

/// RMS mass fluctuation sigma(M) at z = 0, normalized so sigma(R = 8 Mpc) = sigma_8.
///
/// sigma^2(M) = sigma_8^2 * I(R(M)) / I(8 Mpc)
///
/// where I(R) = integral_0^inf k^(ns+2) T^2(k) W^2(kR) dk uses the BBKS transfer function
/// with shape parameter q = k / Omega_m (k in Mpc^{-1}, h = 1).
///
/// # Parameters
/// - `mass_solar`: Halo mass in Msun
/// - `omega_m`: Dimensionless matter density Omega_m
/// - `sigma8`: sigma_8 normalization at z = 0
/// - `ns`: Primordial power spectral index
///
/// # Returns
/// sigma(M) at z = 0 (dimensionless)
pub fn sigma_mass(mass_solar: f64, omega_m: f64, sigma8: f64, ns: f64) -> f64 {
    if mass_solar <= 0.0
        || omega_m <= 0.0
        || sigma8 < 0.0
        || !mass_solar.is_finite()
        || !omega_m.is_finite()
        || !sigma8.is_finite()
        || !ns.is_finite()
    {
        return 0.0;
    }
    const R8: f64 = 8.0; // Mpc (= 8 h^{-1} Mpc at h = 1)
    let i_r8 = sigma_sq_integral(R8, omega_m, ns);
    if i_r8 <= 0.0 {
        return sigma8;
    }
    sigma_mass_inner(mass_solar, omega_m, sigma8, ns, i_r8)
}

/// Internal helper: sigma(M) given precomputed `i_r8 = I(8 Mpc)`.
///
/// # Parameters
/// - `mass_solar`: Halo mass in Msun
/// - `omega_m`: Dimensionless matter density Omega_m
/// - `sigma8`: sigma_8 normalization at z = 0
/// - `ns`: Primordial spectral index
/// - `i_r8`: Precomputed unnormalized power integral I(R_8 = 8 Mpc, omega_m, n_s).
///   Pass the value returned by `sigma_sq_integral(8.0, omega_m, ns)`.
///   Providing this avoids recomputing the expensive GL integration every call
///   when multiple masses are evaluated under the same cosmology.
#[inline]
fn sigma_mass_inner(mass_solar: f64, omega_m: f64, sigma8: f64, ns: f64, i_r8: f64) -> f64 {
    let r = mass_to_radius_mpc(mass_solar, omega_m);
    let i_r = sigma_sq_integral(r, omega_m, ns);
    sigma8 * (i_r / i_r8).sqrt()
}

// ---------------------------------------------------------------------------
// Linear growth factor
// ---------------------------------------------------------------------------

/// Linear growth factor D(z)/D(0) for a given cosmological E(z) function.
///
/// Wraps [`compute_growth_batch`] to expose a single-redshift interface.
/// The RK4 growth ODE integrator uses the provided `e_func` for E(z).
///
/// Returns D(z)/D(0) in (0, 1] for z >= 0.
pub fn linear_growth_factor<F: Fn(f64) -> f64>(z: f64, omega_m: f64, e_func: &F) -> f64 {
    let results = compute_growth_batch(omega_m, e_func, &[z], 1.0);
    results[0].0
}

// ---------------------------------------------------------------------------
// Press-Schechter mass function
// ---------------------------------------------------------------------------

/// Press-Schechter halo mass function dn/dM at redshift z.
///
/// dn/dM = sqrt(2/pi) * (rho_bar/M) * (delta_c/sigma^2) * |dsigma/dM| * exp(-delta_c^2/2sigma^2)
///
/// where sigma = sigma(M) * D(z)/D(0) is the linearly evolved mass fluctuation,
/// delta_c = 1.686 is the spherical collapse threshold, and rho_bar is the mean
/// comoving matter density.
///
/// The derivative |dsigma/dM| is computed by central finite difference with
/// a 1% mass step.
///
/// # Parameters
/// - `mass_solar`: Halo mass in Msun
/// - `z`: Redshift
/// - `omega_m`: Dimensionless matter density Omega_m
/// - `sigma8`: sigma_8 normalization at z = 0
/// - `ns`: Primordial power spectral index
/// - `e_func`: E(z) = H(z)/H_0 for the dark energy model
///
/// # Returns
/// dn/dM in Mpc^{-3} Msun^{-1} (comoving)
pub fn press_schechter_mass_function<F: Fn(f64) -> f64>(
    mass_solar: f64,
    z: f64,
    omega_m: f64,
    sigma8: f64,
    ns: f64,
    e_func: &F,
) -> f64 {
    if mass_solar <= 0.0 || omega_m <= 0.0 {
        return 0.0;
    }

    let d_z = linear_growth_factor(z, omega_m, e_func);
    let i_r8 = sigma_sq_integral(8.0, omega_m, ns);
    if i_r8 <= 0.0 {
        return 0.0;
    }
    press_schechter_inner(mass_solar, d_z, omega_m, sigma8, ns, i_r8)
}

/// Internal helper: PS dn/dM given precomputed growth factor `d_z = D(z)/D(0)`
/// and normalization `i_r8 = I(8 Mpc)`.
///
/// Separating the pre-computation from the mass evaluation avoids re-running
/// the RK4 growth ODE and the R_8 normalization integral for every quadrature
/// node inside `udg_abundance_ratio`.
///
/// # Parameters
/// - `mass_solar`: Halo mass in Msun
/// - `d_z`: Precomputed linear growth factor D(z)/D(0)
/// - `omega_m`: Dimensionless matter density Omega_m
/// - `sigma8`: sigma_8 normalization at z = 0
/// - `ns`: Primordial spectral index
/// - `i_r8`: Precomputed normalization integral I(R_8 = 8 Mpc, omega_m, n_s)
fn press_schechter_inner(
    mass_solar: f64,
    d_z: f64,
    omega_m: f64,
    sigma8: f64,
    ns: f64,
    i_r8: f64,
) -> f64 {
    if mass_solar <= 0.0 {
        return 0.0;
    }

    let sigma_m0 = sigma_mass_inner(mass_solar, omega_m, sigma8, ns, i_r8);
    let sigma_m = sigma_m0 * d_z;
    if sigma_m <= 0.0 {
        return 0.0;
    }

    // Central-difference derivative dsigma/dM (1% mass perturbation)
    let eps = 0.01;
    let sigma_plus =
        sigma_mass_inner(mass_solar * (1.0 + eps), omega_m, sigma8, ns, i_r8) * d_z;
    let sigma_minus =
        sigma_mass_inner(mass_solar * (1.0 - eps), omega_m, sigma8, ns, i_r8) * d_z;
    let dsigma_dm = (sigma_plus - sigma_minus) / (2.0 * eps * mass_solar);

    // Mean comoving matter density (Msun Mpc^{-3})
    let rho_bar = omega_m * RHO_CRIT_MPC;

    // PS formula
    let nu = DELTA_C / sigma_m;
    let exp_term = (-0.5 * nu * nu).exp();

    (2.0_f64 / std::f64::consts::PI).sqrt()
        * (rho_bar / mass_solar)
        * (DELTA_C / (sigma_m * sigma_m))
        * dsigma_dm.abs()
        * exp_term
}

/// Press-Schechter mass function using the orthoplex dark energy model.
///
/// Identical to [`press_schechter_mass_function`] but uses
/// [`hubble_e_orthoplex`] for the growth factor integration. The orthoplex
/// dark energy changes D(z) and hence predicts different halo abundances.
///
/// # Parameters
/// - `mass_solar`: Halo mass in Msun
/// - `z`: Redshift
/// - `omega_m`: Dimensionless matter density Omega_m
/// - `sigma8`: sigma_8 normalization at z = 0
/// - `ns`: Primordial power spectral index
/// - `k`: Number of parts in K_{2,2,...,2} orthoplex graph
/// - `alpha`: Power-law exponent for diffusion scale mapping t(z) = t_0/(1+z)^alpha
/// - `beta`: Coupling constant w = -1 + beta d_s
/// - `t_0`: Present-day diffusion scale
///
/// # Returns
/// dn/dM in Mpc^{-3} Msun^{-1} (comoving)
#[allow(clippy::too_many_arguments)]
pub fn press_schechter_orthoplex(
    mass_solar: f64,
    z: f64,
    omega_m: f64,
    sigma8: f64,
    ns: f64,
    k: usize,
    alpha: f64,
    beta: f64,
    t_0: f64,
) -> f64 {
    let e_func = |zp: f64| hubble_e_orthoplex(zp, omega_m, k, alpha, beta, t_0);
    press_schechter_mass_function(mass_solar, z, omega_m, sigma8, ns, &e_func)
}

// ---------------------------------------------------------------------------
// UDG abundance ratio
// ---------------------------------------------------------------------------

/// Ratio of UDG-scale halo number density: orthoplex / LambdaCDM.
///
/// Integrates dn/dM over [m_lo, m_hi] in log-mass space for both the
/// orthoplex dark energy model and LambdaCDM, then returns their ratio.
///
/// Dense environments like Perseus (z ~ 0.0179) may show measurably different
/// UDG abundances between dark energy models once the mass function is
/// integrated over the UDG host halo mass range (~10^8--10^1^0 Msun).
///
/// # Parameters
/// - `z`: Cluster redshift (use `PERSEUS_Z` for CDG-2 / Perseus)
/// - `omega_m`: Dimensionless matter density Omega_m
/// - `sigma8`: sigma_8 normalization at z = 0
/// - `ns`: Primordial power spectral index
/// - `m_lo`, `m_hi`: Integration mass limits in Msun
/// - `k`, `alpha`, `beta`, `t_0`: Orthoplex dark energy parameters
///
/// # Returns
/// n_orthoplex / n_LambdaCDM (ratio > 1 means more UDGs predicted by orthoplex)
#[allow(clippy::too_many_arguments)]
pub fn udg_abundance_ratio(
    z: f64,
    omega_m: f64,
    sigma8: f64,
    ns: f64,
    m_lo: f64,
    m_hi: f64,
    k: usize,
    alpha: f64,
    beta: f64,
    t_0: f64,
) -> f64 {
    if m_lo >= m_hi || m_lo <= 0.0 {
        return 1.0;
    }

    // Precompute the R_8 normalization integral once --- shared by both models
    // since it depends only on omega_m and ns (not on the dark energy model).
    let i_r8 = sigma_sq_integral(8.0, omega_m, ns);
    if i_r8 <= 0.0 {
        return 1.0;
    }

    // Integrate in log-mass space: M = exp(u), dM = M du
    let u_lo = m_lo.ln();
    let u_hi = m_hi.ln();

    let e_ortho = |zp: f64| hubble_e_orthoplex(zp, omega_m, k, alpha, beta, t_0);
    let e_lcdm = |zp: f64| hubble_e_lcdm(zp, omega_m);

    // Precompute growth factors (one RK4 sweep per model, not one per GL node)
    let d_ortho = linear_growth_factor(z, omega_m, &e_ortho);
    let d_lcdm = linear_growth_factor(z, omega_m, &e_lcdm);

    let n_ortho = gl_integrate(
        |u| {
            let m = u.exp();
            press_schechter_inner(m, d_ortho, omega_m, sigma8, ns, i_r8) * m
        },
        u_lo,
        u_hi,
        50,
    );

    let n_lcdm = gl_integrate(
        |u| {
            let m = u.exp();
            press_schechter_inner(m, d_lcdm, omega_m, sigma8, ns, i_r8) * m
        },
        u_lo,
        u_hi,
        50,
    );

    if n_lcdm <= 0.0 {
        return 1.0;
    }
    n_ortho / n_lcdm
}

// ---------------------------------------------------------------------------
// NFW profile
// ---------------------------------------------------------------------------

/// NFW virial radius r_2_0_0 (kpc) from M_2_0_0 (Msun).
///
/// r_2_0_0 = (3 M_2_0_0 / (4pi * 200 rho_crit))^{1/3}  with rho_crit = 277.5 Msun kpc^{-3}.
#[inline]
fn nfw_r200_kpc(m200_solar: f64) -> f64 {
    (3.0 * m200_solar / (4.0 * std::f64::consts::PI * 200.0 * RHO_CRIT_KPC)).cbrt()
}

/// NFW characteristic density rho_s (Msun kpc^{-3}).
///
/// rho_s = 200 rho_crit c^3 / [3 (ln(1+c) - c/(1+c))]
#[inline]
fn nfw_rho_s(c200: f64) -> f64 {
    let gc = (1.0 + c200).ln() - c200 / (1.0 + c200);
    // Guard against pathological cases where gc is non-finite or too close to zero,
    // which would otherwise produce inf/NaN in the division below.
    if !gc.is_finite() || gc.abs() < 1.0e-12 {
        return 0.0;
    }
    200.0 * RHO_CRIT_KPC * c200 * c200 * c200 / (3.0 * gc)
}

/// Internal: NFW density rho(r) given precomputed scale radius r_s and
/// characteristic density rho_s.  Avoids recomputing r_s and rho_s on each
/// call inside the Jeans quadrature integrand.
#[inline]
fn nfw_density_rs(r_kpc: f64, r_s: f64, rho_s: f64) -> f64 {
    let x = r_kpc / r_s;
    rho_s / (x * (1.0 + x) * (1.0 + x))
}

/// Internal: NFW enclosed mass M(r) given precomputed r_s and rho_s.
#[inline]
fn nfw_enclosed_mass_rs(r_kpc: f64, r_s: f64, rho_s: f64) -> f64 {
    let x = r_kpc / r_s;
    4.0 * std::f64::consts::PI * rho_s * r_s * r_s * r_s * ((1.0 + x).ln() - x / (1.0 + x))
}

/// NFW density profile rho(r) in Msun kpc^{-3}.
///
/// rho(r) = rho_s / [(r/r_s)(1 + r/r_s)^2]
///
/// where r_s = r_2_0_0/c_2_0_0 and rho_s is set by the mass normalization.
///
/// # Parameters
/// - `r_kpc`: Radius in kpc
/// - `m200_solar`: Virial mass M_2_0_0 in Msun
/// - `c200`: Concentration parameter c_2_0_0
pub fn nfw_density(r_kpc: f64, m200_solar: f64, c200: f64) -> f64 {
    if r_kpc <= 0.0 || m200_solar <= 0.0 || c200 <= 0.0 {
        return 0.0;
    }
    let r_s = nfw_r200_kpc(m200_solar) / c200;
    let rho_s = nfw_rho_s(c200);
    nfw_density_rs(r_kpc, r_s, rho_s)
}

/// NFW enclosed mass M(r) in Msun.
///
/// M(r) = 4pi rho_s r_s^3 [ln(1 + r/r_s) - (r/r_s)/(1 + r/r_s)]
///
/// At r = r_2_0_0: M(r_2_0_0) = M_2_0_0 exactly by construction.
///
/// # Parameters
/// - `r_kpc`: Radius in kpc
/// - `m200_solar`: Virial mass M_2_0_0 in Msun
/// - `c200`: Concentration parameter c_2_0_0
pub fn nfw_enclosed_mass(r_kpc: f64, m200_solar: f64, c200: f64) -> f64 {
    if r_kpc <= 0.0 || m200_solar <= 0.0 || c200 <= 0.0 {
        return 0.0;
    }
    let r_s = nfw_r200_kpc(m200_solar) / c200;
    let rho_s = nfw_rho_s(c200);
    nfw_enclosed_mass_rs(r_kpc, r_s, rho_s)
}

/// Isotropic velocity dispersion sigma_r(r) (km/s) from the Jeans equation.
///
/// Solves the isotropic, spherical Jeans equation numerically:
///
///   sigma^2_r(r) = (1/rho(r)) integral_r^{r_max} rho(r') G M(r') / r'^2 dr'
///
/// The log substitution r' = r exp(u) removes the near-center singularity,
/// giving an integrand rho(r') G M(r') / r' that is smooth on [0, ln(r_max/r)].
///
/// Upper limit r_max = 10 r_2_0_0 (truncation bias < 1% for typical NFW halos).
///
/// # Parameters
/// - `r_kpc`: Projected radius in kpc
/// - `m200_solar`: Virial mass M_2_0_0 in Msun
/// - `c200`: Concentration parameter c_2_0_0
///
/// # Returns
/// Radial velocity dispersion sigma_r in km/s
pub fn nfw_velocity_dispersion(r_kpc: f64, m200_solar: f64, c200: f64) -> f64 {
    if r_kpc <= 0.0 || m200_solar <= 0.0 || c200 <= 0.0 {
        return 0.0;
    }

    // Precompute r_s and rho_s once --- reused for every GL quadrature node.
    let r_200 = nfw_r200_kpc(m200_solar);
    let r_s = r_200 / c200;
    let rho_s = nfw_rho_s(c200);

    let rho_r = nfw_density_rs(r_kpc, r_s, rho_s);
    if rho_r <= 0.0 {
        return 0.0;
    }

    let r_max = 10.0 * r_200;

    // Log substitution r' = r_kpc * exp(u), dr' = r' du:
    //   integral_r^{r_max} rho(r') G M(r') / r'^2 dr' = integral_0^{u_max} rho(r') G M(r') / r' du
    let u_max = (r_max / r_kpc).ln();

    let integral = gl_integrate(
        |u| {
            let rp = r_kpc * u.exp();
            let rho_p = nfw_density_rs(rp, r_s, rho_s);
            let m_p = nfw_enclosed_mass_rs(rp, r_s, rho_s);
            // Units: (km/s)^2 kpc Msun^{-1} * Msun * Msun kpc^{-3} / kpc = (km/s)^2 Msun kpc^{-3}
            G_KPC_KM_S * m_p * rho_p / rp
        },
        0.0,
        u_max,
        100,
    );

    // sigma^2_r = integral / rho(r)  [(km/s)^2 Msun kpc^{-3} / (Msun kpc^{-3}) = (km/s)^2]
    (integral / rho_r).max(0.0).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const OMEGA_M: f64 = 0.315;
    const SIGMA8: f64 = 0.811;
    const NS: f64 = 0.965;

    #[test]
    fn test_constants_physical_values() {
        assert_eq!(DELTA_C, 1.686);
        assert_eq!(PERSEUS_Z, 0.0179);
        assert_eq!(CDG2_STELLAR_MASS_SOLAR, 6.2e6);
        assert_eq!(CDG2_DM_FRACTION_MIN, 0.9994);
    }

    #[test]
    fn test_bbks_transfer_low_k_approaches_one() {
        // T(q) -> 1 as k -> 0 (large scales, no suppression)
        let t = bbks_transfer(1e-6, 0.3);
        assert_relative_eq!(t, 1.0, max_relative = 0.01);
    }

    #[test]
    fn test_bbks_transfer_decreases_with_k() {
        // T(k) should decrease with k (small-scale power suppression)
        let t1 = bbks_transfer(0.1, 0.3);
        let t2 = bbks_transfer(1.0, 0.3);
        let t3 = bbks_transfer(10.0, 0.3);
        assert!(t1 > t2, "T(0.1) > T(1.0)");
        assert!(t2 > t3, "T(1.0) > T(10.0)");
    }

    #[test]
    fn test_top_hat_window_small_x_approaches_one() {
        let w = top_hat_window(1e-6);
        assert_relative_eq!(w, 1.0, max_relative = 1e-8);
    }

    #[test]
    fn test_top_hat_window_continuity_at_transition() {
        // Check continuity near the Taylor/exact boundary (x ~ 1e-3)
        let w_taylor = top_hat_window(9e-4);
        let w_exact = top_hat_window(1.1e-3);
        assert_relative_eq!(w_taylor, w_exact, max_relative = 1e-4);
    }

    #[test]
    fn test_top_hat_window_negative_input_is_finite() {
        // Negative x is not physically expected but the function must not diverge.
        let w = top_hat_window(-1e-4);
        assert!(w.is_finite(), "W(-1e-4) must be finite");
    }

    #[test]
    fn test_sigma_mass_invalid_inputs_return_zero() {
        // Non-physical inputs must not panic or return NaN.
        assert_eq!(sigma_mass(0.0, OMEGA_M, SIGMA8, NS), 0.0);
        assert_eq!(sigma_mass(-1e9, OMEGA_M, SIGMA8, NS), 0.0);
        assert_eq!(sigma_mass(1e9, 0.0, SIGMA8, NS), 0.0);
        assert_eq!(sigma_mass(1e9, -0.3, SIGMA8, NS), 0.0);
        assert_eq!(sigma_mass(f64::INFINITY, OMEGA_M, SIGMA8, NS), 0.0);
        assert_eq!(sigma_mass(f64::NAN, OMEGA_M, SIGMA8, NS), 0.0);
    }

    #[test]
    fn test_nfw_rho_s_tiny_c_returns_zero() {
        // Very small c200 makes gc ~ 0; nfw_rho_s must return 0.0 (no NaN/inf).
        let rho = nfw_density(1.0, 1e12, 1e-15);
        assert_eq!(rho, 0.0);
        let m = nfw_enclosed_mass(1.0, 1e12, 1e-15);
        assert_eq!(m, 0.0);
    }

    #[test]
    fn test_sigma_mass_r8_normalization() {
        // sigma(M_8) must equal sigma_8 where M_8 is the mass within R_8 = 8 Mpc
        let rho_bar = OMEGA_M * RHO_CRIT_MPC;
        let m8 = (4.0 / 3.0) * std::f64::consts::PI * 8.0_f64.powi(3) * rho_bar;
        let s = sigma_mass(m8, OMEGA_M, SIGMA8, NS);
        assert_relative_eq!(s, SIGMA8, max_relative = 1e-3);
    }

    #[test]
    fn test_sigma_mass_decreases_with_mass() {
        // sigma(M) is a decreasing function of M
        let s1 = sigma_mass(1e8, OMEGA_M, SIGMA8, NS);
        let s2 = sigma_mass(1e10, OMEGA_M, SIGMA8, NS);
        let s3 = sigma_mass(1e12, OMEGA_M, SIGMA8, NS);
        assert!(s1 > s2, "sigma(10^8) > sigma(10^1^0)");
        assert!(s2 > s3, "sigma(10^1^0) > sigma(10^1^2)");
    }

    #[test]
    fn test_sigma_mass_udg_range_positive() {
        // CDG-2 mass range (10^8--10^1^0 Msun) must give positive sigma
        let s = sigma_mass(1e9, OMEGA_M, SIGMA8, NS);
        assert!(s > 0.0, "sigma(10^9 Msun) must be positive");
        assert!(s.is_finite(), "sigma(10^9 Msun) must be finite");
    }

    #[test]
    fn test_linear_growth_factor_z0_equals_one() {
        // D(0)/D(0) = 1 by definition
        let e_lcdm = |z: f64| hubble_e_lcdm(z, OMEGA_M);
        let d = linear_growth_factor(0.0, OMEGA_M, &e_lcdm);
        assert_relative_eq!(d, 1.0, max_relative = 1e-3);
    }

    #[test]
    fn test_linear_growth_factor_decreases_with_z() {
        // Structure grows toward today: D(z)/D(0) < 1 for z > 0
        let e_lcdm = |z: f64| hubble_e_lcdm(z, OMEGA_M);
        let d05 = linear_growth_factor(0.5, OMEGA_M, &e_lcdm);
        let d10 = linear_growth_factor(1.0, OMEGA_M, &e_lcdm);
        assert!(d05 > d10, "D(0.5) > D(1.0)");
        assert!(d05 < 1.0, "D(0.5)/D(0) < 1");
    }

    #[test]
    fn test_press_schechter_mass_function_positive() {
        let e_lcdm = |z: f64| hubble_e_lcdm(z, OMEGA_M);
        let dn_dm = press_schechter_mass_function(1e12, 0.0, OMEGA_M, SIGMA8, NS, &e_lcdm);
        assert!(dn_dm > 0.0, "PS mass function must be positive at 10^1^2 Msun");
        assert!(dn_dm.is_finite(), "PS mass function must be finite");
    }

    #[test]
    fn test_press_schechter_decreases_with_mass() {
        // Cluster-mass halos are rarer than group-mass halos
        let e_lcdm = |z: f64| hubble_e_lcdm(z, OMEGA_M);
        let dn11 = press_schechter_mass_function(1e11, 0.0, OMEGA_M, SIGMA8, NS, &e_lcdm);
        let dn13 = press_schechter_mass_function(1e13, 0.0, OMEGA_M, SIGMA8, NS, &e_lcdm);
        assert!(dn11 > dn13, "PS: dn/dM(10^1^1) > dn/dM(10^1^3)");
    }

    #[test]
    fn test_press_schechter_orthoplex_beta_zero_matches_lcdm() {
        // With beta = 0 the orthoplex model reduces to LambdaCDM
        let e_lcdm = |z: f64| hubble_e_lcdm(z, OMEGA_M);
        let dn_lcdm = press_schechter_mass_function(
            1e12, PERSEUS_Z, OMEGA_M, SIGMA8, NS, &e_lcdm,
        );
        let dn_ortho =
            press_schechter_orthoplex(1e12, PERSEUS_Z, OMEGA_M, SIGMA8, NS, 3, 1.0, 0.0, 1.0);
        assert_relative_eq!(dn_lcdm, dn_ortho, max_relative = 0.01);
    }

    #[test]
    fn test_udg_abundance_ratio_beta_zero_near_unity() {
        // beta = 0 orthoplex = LambdaCDM, so ratio ~ 1
        let ratio = udg_abundance_ratio(
            PERSEUS_Z, OMEGA_M, SIGMA8, NS, 1e8, 1e10, 3, 1.0, 0.0, 1.0,
        );
        assert_relative_eq!(ratio, 1.0, max_relative = 0.02);
    }

    #[test]
    fn test_udg_abundance_ratio_finite() {
        let ratio =
            udg_abundance_ratio(PERSEUS_Z, OMEGA_M, SIGMA8, NS, 1e8, 1e10, 3, 1.0, 0.05, 1.0);
        assert!(ratio.is_finite(), "Ratio must be finite");
        assert!(ratio > 0.0, "Ratio must be positive");
    }

    #[test]
    fn test_nfw_density_positive_and_finite() {
        let rho = nfw_density(10.0, 1e11, 10.0);
        assert!(rho > 0.0, "NFW density must be positive");
        assert!(rho.is_finite(), "NFW density must be finite");
    }

    #[test]
    fn test_nfw_density_decreases_with_r() {
        let rho1 = nfw_density(1.0, 1e11, 10.0);
        let rho2 = nfw_density(10.0, 1e11, 10.0);
        let rho3 = nfw_density(100.0, 1e11, 10.0);
        assert!(rho1 > rho2, "rho_NFW(1 kpc) > rho_NFW(10 kpc)");
        assert!(rho2 > rho3, "rho_NFW(10 kpc) > rho_NFW(100 kpc)");
    }

    #[test]
    fn test_nfw_density_zero_outside_domain() {
        assert_eq!(nfw_density(0.0, 1e11, 10.0), 0.0);
        assert_eq!(nfw_density(10.0, 0.0, 10.0), 0.0);
        assert_eq!(nfw_density(10.0, 1e11, 0.0), 0.0);
    }

    #[test]
    fn test_nfw_enclosed_mass_increases_with_r() {
        let m1 = nfw_enclosed_mass(5.0, 1e11, 10.0);
        let m2 = nfw_enclosed_mass(50.0, 1e11, 10.0);
        let m3 = nfw_enclosed_mass(500.0, 1e11, 10.0);
        assert!(m1 < m2, "M_enc(5) < M_enc(50)");
        assert!(m2 < m3, "M_enc(50) < M_enc(500)");
    }

    #[test]
    fn test_nfw_enclosed_mass_equals_m200_at_r200() {
        // M(r_2_0_0) = M_2_0_0 by construction
        let m200 = 1e11_f64;
        let c200 = 10.0_f64;
        let r200 = nfw_r200_kpc(m200);
        let m_enc = nfw_enclosed_mass(r200, m200, c200);
        assert_relative_eq!(m_enc, m200, max_relative = 1e-6);
    }

    #[test]
    fn test_nfw_velocity_dispersion_positive_and_finite() {
        let sigma = nfw_velocity_dispersion(10.0, 1e11, 10.0);
        assert!(sigma > 0.0, "sigma_r must be positive");
        assert!(sigma.is_finite(), "sigma_r must be finite");
    }

    #[test]
    fn test_nfw_velocity_dispersion_udg_scale() {
        // UDG-scale halo (10^1^0 Msun, c = 15): expect sigma ~ 10--100 km/s
        let sigma = nfw_velocity_dispersion(5.0, 1e10, 15.0);
        assert!(sigma > 1.0, "sigma_r(UDG) > 1 km/s");
        assert!(sigma < 500.0, "sigma_r(UDG) < 500 km/s");
    }

    #[test]
    fn test_nfw_velocity_dispersion_zero_outside_domain() {
        assert_eq!(nfw_velocity_dispersion(0.0, 1e11, 10.0), 0.0);
        assert_eq!(nfw_velocity_dispersion(10.0, 0.0, 10.0), 0.0);
        assert_eq!(nfw_velocity_dispersion(10.0, 1e11, 0.0), 0.0);
    }
}
