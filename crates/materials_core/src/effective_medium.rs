//! Effective Medium Theory for Metamaterials.
//!
//! Provides:
//! - Maxwell-Garnett mixing formula (dilute inclusions in host)
//! - Bruggeman self-consistent EMA (arbitrary fill fractions)
//! - Drude-Lorentz dielectric model with Kramers-Kronig consistency
//! - Transfer Matrix Method (TMM) for multilayer thin-film optics
//! - Kramers-Kronig Hilbert-transform consistency check
//!
//! All permittivities are complex: eps = eps' + i*eps''.
//!
//! # Literature
//! - Sihvola (1999), "Electromagnetic Mixing Formulas", IEE
//! - Pozar (2012), "Microwave Engineering", 4th ed., Wiley
//! - Born & Wolf (2019), "Principles of Optics", 7th ed., Cambridge

use num_complex::Complex64;
use std::f64::consts::PI;

/// Result from Kramers-Kronig consistency check.
#[derive(Debug, Clone)]
pub struct KramersKronigResult {
    /// Reconstructed component
    pub reconstructed: Vec<f64>,
    /// Maximum relative error between actual and reconstructed
    pub max_rel_error: f64,
}

/// Result from Transfer Matrix Method calculation.
#[derive(Debug, Clone)]
pub struct TmmResult {
    /// Complex reflection coefficient
    pub r: Complex64,
    /// Reflectance |r|^2
    pub reflectance: f64,
    /// Complex transmission coefficient (if computed)
    pub t: Option<Complex64>,
    /// Transmittance |t|^2 (if computed)
    pub transmittance: Option<f64>,
}

/// Lorentz oscillator parameters.
#[derive(Debug, Clone, Copy)]
pub struct LorentzOscillator {
    /// Oscillator strength
    pub strength: f64,
    /// Resonance frequency
    pub omega_0: f64,
    /// Damping rate
    pub gamma: f64,
}

// ---------------------------------------------------------------------------
// 1. Maxwell-Garnett Mixing
// ---------------------------------------------------------------------------

/// Maxwell-Garnett effective permittivity for spherical inclusions.
///
/// eps_eff = eps_host * (1 + 2*f*beta) / (1 - f*beta)
/// where beta = (eps_inc - eps_host) / (eps_inc + 2*eps_host)
///
/// # Arguments
/// * `eps_host` - Host medium permittivity
/// * `eps_inc` - Inclusion permittivity
/// * `f` - Volume fraction of inclusions (0 < f < 1)
///
/// # Returns
/// Effective permittivity
pub fn maxwell_garnett(eps_host: Complex64, eps_inc: Complex64, f: f64) -> Complex64 {
    let beta = (eps_inc - eps_host) / (eps_inc + 2.0 * eps_host);
    eps_host * (1.0 + 2.0 * f * beta) / (1.0 - f * beta)
}

/// Maxwell-Garnett for array of permittivities.
pub fn maxwell_garnett_array(
    eps_host: &[Complex64],
    eps_inc: &[Complex64],
    f: f64,
) -> Vec<Complex64> {
    eps_host
        .iter()
        .zip(eps_inc.iter())
        .map(|(&eh, &ei)| maxwell_garnett(eh, ei, f))
        .collect()
}

// ---------------------------------------------------------------------------
// 2. Bruggeman Self-Consistent EMA
// ---------------------------------------------------------------------------

/// Bruggeman effective medium approximation (self-consistent).
///
/// Solves: f*(eps_1 - eps_eff)/(eps_1 + 2*eps_eff)
///         + (1-f)*(eps_2 - eps_eff)/(eps_2 + 2*eps_eff) = 0
///
/// This gives a quadratic: 2*eps_eff^2 + B*eps_eff + C = 0
/// where B = -[(3f-1)*eps_1 + (2-3f)*eps_2], C = -eps_1*eps_2.
///
/// # Arguments
/// * `eps_1` - Permittivity of component 1
/// * `eps_2` - Permittivity of component 2
/// * `f` - Volume fraction of component 1
///
/// # Returns
/// Effective permittivity (causal root with positive imaginary part)
pub fn bruggeman(eps_1: Complex64, eps_2: Complex64, f: f64) -> Complex64 {
    let a = Complex64::new(2.0, 0.0);
    let b = -((3.0 * f - 1.0) * eps_1 + (2.0 - 3.0 * f) * eps_2);
    let c = -eps_1 * eps_2;

    let disc = b * b - 4.0 * a * c;
    let sqrt_disc = disc.sqrt();

    let root1 = (-b + sqrt_disc) / (2.0 * a);
    let root2 = (-b - sqrt_disc) / (2.0 * a);

    // Choose causal root (positive imaginary part for lossy media)
    if root1.im >= 0.0 && root1.re > 0.0 {
        root1
    } else if root2.im >= 0.0 && root2.re > 0.0 {
        root2
    } else if root1.re > 0.0 {
        root1
    } else {
        root2
    }
}

/// Bruggeman for arrays of permittivities.
pub fn bruggeman_array(eps_1: &[Complex64], eps_2: &[Complex64], f: f64) -> Vec<Complex64> {
    eps_1
        .iter()
        .zip(eps_2.iter())
        .map(|(&e1, &e2)| bruggeman(e1, e2, f))
        .collect()
}

// ---------------------------------------------------------------------------
// 3. Drude-Lorentz Dielectric Model
// ---------------------------------------------------------------------------

/// Drude-Lorentz dielectric function.
///
/// eps(omega) = eps_inf - omega_p^2 / (omega^2 + i*gamma_d*omega)
///              + sum_j S_j * omega_j^2 / (omega_j^2 - omega^2 - i*gamma_j*omega)
///
/// # Arguments
/// * `omega` - Frequency array
/// * `eps_inf` - High-frequency dielectric constant
/// * `omega_p` - Plasma frequency (Drude term)
/// * `gamma_d` - Drude damping rate
/// * `oscillators` - Lorentz oscillator parameters
pub fn drude_lorentz(
    omega: &[f64],
    eps_inf: f64,
    omega_p: f64,
    gamma_d: f64,
    oscillators: &[LorentzOscillator],
) -> Vec<Complex64> {
    omega
        .iter()
        .map(|&w| {
            let mut eps = Complex64::new(eps_inf, 0.0);

            // Drude term
            if omega_p > 0.0 {
                let denom = Complex64::new(w * w, gamma_d * w);
                if denom.norm() > 1e-30 {
                    eps -= omega_p * omega_p / denom;
                }
            }

            // Lorentz oscillators
            for osc in oscillators {
                let denom = Complex64::new(osc.omega_0 * osc.omega_0 - w * w, -osc.gamma * w);
                if denom.norm() > 1e-30 {
                    eps += osc.strength * osc.omega_0 * osc.omega_0 / denom;
                }
            }

            eps
        })
        .collect()
}

/// Drude-only dielectric (no oscillators).
pub fn drude(omega: &[f64], eps_inf: f64, omega_p: f64, gamma: f64) -> Vec<Complex64> {
    drude_lorentz(omega, eps_inf, omega_p, gamma, &[])
}

// ---------------------------------------------------------------------------
// 4. Kramers-Kronig Consistency Check
// ---------------------------------------------------------------------------

/// Verify Kramers-Kronig consistency of a dielectric function via FFT Hilbert transform.
///
/// Uses:
///   chi'(w) = -H[chi''](w)  (real part from imaginary)
///   chi''(w) = H[chi'](w)   (imaginary part from real)
///
/// # Arguments
/// * `eps` - Dielectric function array (positive frequencies only)
/// * `check_real` - If true, reconstruct eps' from eps''; else reconstruct eps'' from eps'
/// * `eps_inf` - High-frequency permittivity (default: last real part)
pub fn kramers_kronig_check(
    eps: &[Complex64],
    check_real: bool,
    eps_inf: Option<f64>,
) -> KramersKronigResult {
    use rustfft::FftPlanner;

    let n = eps.len();
    let eps_inf = eps_inf.unwrap_or(eps[n - 1].re);

    let (reconstructed, actual) = if check_real {
        // Reconstruct eps' from eps'' via chi' = -H[chi'']
        let chi_imag: Vec<f64> = eps.iter().map(|e| e.im).collect();

        // Odd-extend to negative frequencies: chi''(-w) = -chi''(w)
        let m = 2 * n;
        let mut signal: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); m];
        for i in 0..n {
            signal[i] = Complex64::new(-chi_imag[n - 1 - i], 0.0);
            signal[n + i] = Complex64::new(chi_imag[i], 0.0);
        }

        // FFT Hilbert: H[f] = IFFT[-i * sgn(freq) * FFT[f]]
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(m);
        let ifft = planner.plan_fft_inverse(m);

        fft.process(&mut signal);

        for (i, s) in signal.iter_mut().enumerate() {
            let freq_idx = if i <= m / 2 {
                i as i64
            } else {
                i as i64 - m as i64
            };
            let sgn = if freq_idx > 0 {
                1.0
            } else if freq_idx < 0 {
                -1.0
            } else {
                0.0
            };
            *s *= Complex64::new(0.0, -sgn);
        }

        ifft.process(&mut signal);
        let scale = 1.0 / m as f64;

        let reconstructed: Vec<f64> = (0..n).map(|i| eps_inf - signal[n + i].re * scale).collect();
        let actual: Vec<f64> = eps.iter().map(|e| e.re).collect();

        (reconstructed, actual)
    } else {
        // Reconstruct eps'' from eps' via chi'' = H[chi']
        let chi_real: Vec<f64> = eps.iter().map(|e| e.re - eps_inf).collect();

        // Even-extend to negative frequencies: chi'(-w) = chi'(w)
        let m = 2 * n;
        let mut signal: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); m];
        for i in 0..n {
            signal[i] = Complex64::new(chi_real[n - 1 - i], 0.0);
            signal[n + i] = Complex64::new(chi_real[i], 0.0);
        }

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(m);
        let ifft = planner.plan_fft_inverse(m);

        fft.process(&mut signal);

        for (i, s) in signal.iter_mut().enumerate() {
            let freq_idx = if i <= m / 2 {
                i as i64
            } else {
                i as i64 - m as i64
            };
            let sgn = if freq_idx > 0 {
                1.0
            } else if freq_idx < 0 {
                -1.0
            } else {
                0.0
            };
            *s *= Complex64::new(0.0, -sgn);
        }

        ifft.process(&mut signal);
        let scale = 1.0 / m as f64;

        let reconstructed: Vec<f64> = (0..n).map(|i| signal[n + i].re * scale).collect();
        let actual: Vec<f64> = eps.iter().map(|e| e.im).collect();

        (reconstructed, actual)
    };

    // Compute relative error
    let max_actual = actual.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let threshold = 0.01 * max_actual;

    let max_rel_error = actual
        .iter()
        .zip(reconstructed.iter())
        .filter(|(a, _)| a.abs() > threshold)
        .map(|(a, r)| (a - r).abs() / a.abs())
        .fold(0.0_f64, f64::max);

    KramersKronigResult {
        reconstructed,
        max_rel_error,
    }
}

// ---------------------------------------------------------------------------
// 5. Transfer Matrix Method (TMM) for Multilayer Optics
// ---------------------------------------------------------------------------

/// Transfer Matrix Method for a multilayer thin-film stack.
///
/// Uses the amplitude transfer matrix approach (Byrnes, arXiv:1603.02720)
/// which correctly handles absorbing layers. Each layer contributes a
/// propagation matrix and an interface matrix, composed into a system
/// transfer matrix relating forward/backward wave amplitudes.
///
/// # Arguments
/// * `n_layers` - Refractive indices [n_incidence, n_1, n_2, ..., n_substrate]
/// * `d_layers` - Thicknesses of intermediate layers [d_1, d_2, ...]
/// * `wavelength` - Free-space wavelength (same units as d_layers)
/// * `theta_i` - Angle of incidence (radians)
/// * `s_polarized` - True for s (TE), false for p (TM)
pub fn tmm_reflection(
    n_layers: &[Complex64],
    d_layers: &[f64],
    wavelength: f64,
    theta_i: f64,
    s_polarized: bool,
) -> TmmResult {
    let n_inc = n_layers[0];
    let num_layers = n_layers.len();

    // Snell's law: compute cos(theta) for each layer
    let sin_theta_i = theta_i.sin();
    let cos_theta: Vec<Complex64> = n_layers
        .iter()
        .map(|&n| {
            let sin_t = n_inc * sin_theta_i / n;
            let cos_t = (Complex64::new(1.0, 0.0) - sin_t * sin_t).sqrt();
            // Choose branch with positive imaginary part (forward-propagating)
            if cos_t.im < 0.0 { -cos_t } else { cos_t }
        })
        .collect();

    // Fresnel interface coefficients for each adjacent pair
    let fresnel = |i: usize, j: usize| -> (Complex64, Complex64) {
        let (ni, nj) = (n_layers[i], n_layers[j]);
        let (ci, cj) = (cos_theta[i], cos_theta[j]);
        if s_polarized {
            let eta_i = ni * ci;
            let eta_j = nj * cj;
            let r = (eta_i - eta_j) / (eta_i + eta_j);
            let t = 2.0 * eta_i / (eta_i + eta_j);
            (r, t)
        } else {
            // p-pol: use n/cos(theta)
            let eta_i = ni / ci;
            let eta_j = nj / cj;
            let r = (eta_i - eta_j) / (eta_i + eta_j);
            let t = 2.0 * eta_i / (eta_i + eta_j);
            (r, t)
        }
    };

    // Phase accumulated traversing each layer
    let delta: Vec<Complex64> = (0..num_layers)
        .map(|j| {
            if j == 0 || j == num_layers - 1 {
                Complex64::new(0.0, 0.0)
            } else {
                let d_j = d_layers[j - 1];
                2.0 * PI * n_layers[j] * cos_theta[j] * d_j / wavelength
            }
        })
        .collect();

    // Build system transfer matrix Mtilde (amplitude-based).
    // For each intermediate layer j (1..N-1):
    //   M_j = (1/t_{j,j+1}) * P_j * I_{j,j+1}
    // where P_j = [[exp(-i*delta_j), 0], [0, exp(i*delta_j)]]
    //       I_{j,j+1} = [[1, r_{j,j+1}], [r_{j,j+1}, 1]]
    //
    // Mtilde = (1/t_{0,1}) * I_{0,1} * M_1 * M_2 * ... * M_{N-2}
    let one = Complex64::new(1.0, 0.0);
    let i_unit = Complex64::new(0.0, 1.0);

    // Start with the first interface (incidence -> layer 1)
    let (r01, t01) = fresnel(0, 1);
    let mut mtilde = [[one / t01, r01 / t01], [r01 / t01, one / t01]];

    // Multiply in each intermediate layer + next interface
    #[allow(clippy::needless_range_loop)] // j indexes delta[] AND feeds fresnel(j, j+1)
    for j in 1..(num_layers - 1) {
        let (r_jk, t_jk) = fresnel(j, j + 1);
        let exp_neg = (-i_unit * delta[j]).exp();
        let exp_pos = (i_unit * delta[j]).exp();

        // M_j = (1/t_{j,j+1}) * P_j * I_{j,j+1}
        // = (1/t) * [[exp(-id)*1 + 0*r, exp(-id)*r + 0*1],
        //            [0*1 + exp(id)*r,   0*r + exp(id)*1 ]]
        // = (1/t) * [[exp(-id),   exp(-id)*r],
        //            [exp(id)*r,  exp(id)   ]]
        let inv_t = one / t_jk;
        let layer = [
            [inv_t * exp_neg, inv_t * exp_neg * r_jk],
            [inv_t * exp_pos * r_jk, inv_t * exp_pos],
        ];

        // Matrix multiply: mtilde = mtilde * layer
        let new_m = [
            [
                mtilde[0][0] * layer[0][0] + mtilde[0][1] * layer[1][0],
                mtilde[0][0] * layer[0][1] + mtilde[0][1] * layer[1][1],
            ],
            [
                mtilde[1][0] * layer[0][0] + mtilde[1][1] * layer[1][0],
                mtilde[1][0] * layer[0][1] + mtilde[1][1] * layer[1][1],
            ],
        ];
        mtilde = new_m;
    }

    // If there are no intermediate layers (bare interface), mtilde is already
    // the interface matrix for (0,1) which in that case is (0, substrate).

    // Extract r and t from the system matrix
    let r = mtilde[1][0] / mtilde[0][0];
    let reflectance = r.norm_sqr();

    let t = one / mtilde[0][0];

    // Transmittance: T = |t|^2 * Re(n_sub*cos_sub) / Re(n_inc*cos_inc)
    let flux_inc = if s_polarized {
        (n_layers[0] * cos_theta[0]).re
    } else {
        (n_layers[0] * cos_theta[0].conj()).re
    };
    let flux_sub = if s_polarized {
        (n_layers[num_layers - 1] * cos_theta[num_layers - 1]).re
    } else {
        (n_layers[num_layers - 1] * cos_theta[num_layers - 1].conj()).re
    };
    let transmittance = if flux_inc.abs() < 1e-30 {
        0.0
    } else {
        (t.norm_sqr() * flux_sub / flux_inc).max(0.0)
    };

    TmmResult {
        r,
        reflectance,
        t: Some(t),
        transmittance: Some(transmittance),
    }
}

/// TMM at multiple wavelengths.
pub fn tmm_spectrum(
    n_layers_fn: impl Fn(f64) -> Vec<Complex64>,
    d_layers: &[f64],
    wavelengths: &[f64],
    theta_i: f64,
    s_polarized: bool,
) -> Vec<TmmResult> {
    wavelengths
        .iter()
        .map(|&wl| {
            let n_layers = n_layers_fn(wl);
            tmm_reflection(&n_layers, d_layers, wl, theta_i, s_polarized)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_maxwell_garnett_dilute_limit() {
        let eps_host = Complex64::new(1.0, 0.0);
        let eps_inc = Complex64::new(4.0, 0.0);
        let f = 0.01;
        let eps_eff = maxwell_garnett(eps_host, eps_inc, f);
        // Dilute limit: eps_eff ~ eps_host * (1 + 2*f*(eps_inc - eps_host)/(eps_inc + 2*eps_host))
        assert!(eps_eff.re > 1.0 && eps_eff.re < 4.0);
    }

    #[test]
    fn test_maxwell_garnett_zero_fraction() {
        let eps_host = Complex64::new(2.0, 0.0);
        let eps_inc = Complex64::new(10.0, 0.0);
        let eps_eff = maxwell_garnett(eps_host, eps_inc, 0.0);
        assert_relative_eq!(eps_eff.re, eps_host.re, epsilon = 1e-10);
    }

    #[test]
    fn test_bruggeman_symmetric() {
        let eps_1 = Complex64::new(1.0, 0.0);
        let eps_2 = Complex64::new(4.0, 0.0);
        let eps_half = bruggeman(eps_1, eps_2, 0.5);
        // At f=0.5, should be between eps_1 and eps_2
        assert!(eps_half.re > 1.0 && eps_half.re < 4.0);
    }

    #[test]
    fn test_bruggeman_limits() {
        let eps_1 = Complex64::new(2.0, 0.0);
        let eps_2 = Complex64::new(3.0, 0.0);
        // f=0 should give eps_2, f=1 should give eps_1
        let eps_f0 = bruggeman(eps_1, eps_2, 0.0);
        let eps_f1 = bruggeman(eps_1, eps_2, 1.0);
        assert_relative_eq!(eps_f0.re, eps_2.re, epsilon = 0.1);
        assert_relative_eq!(eps_f1.re, eps_1.re, epsilon = 0.1);
    }

    #[test]
    fn test_drude_high_frequency_limit() {
        let omega: Vec<f64> = (1..100).map(|i| i as f64 * 0.1).collect();
        let eps = drude(&omega, 1.0, 1.0, 0.1);
        // At high frequency, eps -> eps_inf
        let last = eps.last().unwrap();
        assert!(last.re > 0.9 && last.re < 1.1);
    }

    #[test]
    fn test_drude_low_frequency_negative() {
        let omega = vec![0.1, 0.2, 0.3];
        let eps = drude(&omega, 1.0, 10.0, 0.1); // Large omega_p
        // At low frequency, Drude gives negative eps
        assert!(eps[0].re < 0.0);
    }

    #[test]
    fn test_lorentz_resonance_peak() {
        let omega: Vec<f64> = (1..100).map(|i| i as f64 * 0.1).collect();
        let osc = LorentzOscillator {
            strength: 1.0,
            omega_0: 5.0,
            gamma: 0.5,
        };
        let eps = drude_lorentz(&omega, 1.0, 0.0, 0.0, &[osc]);
        // Near resonance (omega ~ 5), eps.im magnitude should be significant
        let im_max_magnitude = eps
            .iter()
            .enumerate()
            .filter(|(i, _)| omega[*i] > 4.0 && omega[*i] < 6.0)
            .map(|(_, e)| e.im.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            im_max_magnitude > 0.01,
            "Im(eps) magnitude = {}",
            im_max_magnitude
        );
    }

    #[test]
    fn test_tmm_no_layers_fresnel() {
        // Interface between air (n=1) and glass (n=1.5) at normal incidence
        let n_layers = vec![Complex64::new(1.0, 0.0), Complex64::new(1.5, 0.0)];
        let d_layers: Vec<f64> = vec![];
        let result = tmm_reflection(&n_layers, &d_layers, 500.0, 0.0, true);
        // Fresnel: r = (n1 - n2) / (n1 + n2) = (1 - 1.5) / (1 + 1.5) = -0.2
        // R = 0.04
        assert_relative_eq!(result.reflectance, 0.04, epsilon = 0.01);
    }

    #[test]
    fn test_tmm_quarter_wave() {
        // Quarter-wave antireflection coating
        let n_inc = Complex64::new(1.0, 0.0);
        let n_ar = Complex64::new(1.38, 0.0); // MgF2 approximately
        let n_sub = Complex64::new(1.5, 0.0);
        let wavelength = 550.0; // nm
        let d_ar = wavelength / (4.0 * n_ar.re); // Quarter-wave thickness

        let n_layers = vec![n_inc, n_ar, n_sub];
        let d_layers = vec![d_ar];
        let result = tmm_reflection(&n_layers, &d_layers, wavelength, 0.0, true);

        // Quarter-wave AR should reduce reflectance significantly
        // For perfect AR: n_ar = sqrt(n_inc * n_sub) ~ 1.22, so 1.38 is close
        assert!(result.reflectance < 0.02);
    }

    #[test]
    fn test_tmm_energy_conservation_lossless() {
        // Air/glass: lossless, so R + T must equal 1.0
        let n_inc = Complex64::new(1.0, 0.0);
        let n_glass = Complex64::new(1.5, 0.0);
        let n_sub = Complex64::new(1.0, 0.0);
        let wavelength = 550.0;
        let d = wavelength / (4.0 * n_glass.re);
        let result = tmm_reflection(&[n_inc, n_glass, n_sub], &[d], wavelength, 0.0, true);
        let t = result.transmittance.unwrap();
        let sum = result.reflectance + t;
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tmm_energy_conservation_lossy() {
        // Lossy slab: R + T < 1.0, absorptance A > 0
        let n_inc = Complex64::new(1.0, 0.0);
        let n_lossy = Complex64::new(2.0, 0.5); // absorbing medium
        let n_sub = Complex64::new(1.0, 0.0);
        let wavelength = 550.0;
        let d = 100.0; // 100 nm slab
        let result = tmm_reflection(&[n_inc, n_lossy, n_sub], &[d], wavelength, 0.0, true);
        // Cross-check via Fresnel-Airy formula with exponentials (correct for lossy media).
        // For n1=n3=1, n2 complex, normal incidence, s-pol:
        //   r12 = (n1 - n2) / (n1 + n2), t12 = 2*n1/(n1+n2)
        //   r23 = (n2 - n3) / (n2 + n3), t23 = 2*n2/(n2+n3)
        //   delta = 2*pi*n2*d/lambda
        //   r = (r12 + r23 * exp(2i*delta)) / (1 + r12*r23*exp(2i*delta))
        //   t = (t12 * t23 * exp(i*delta)) / (1 + r12*r23*exp(2i*delta))
        let n1 = Complex64::new(1.0, 0.0);
        let n2 = Complex64::new(2.0, 0.5);
        let n3 = Complex64::new(1.0, 0.0);
        let i_unit = Complex64::new(0.0, 1.0);
        let delta = 2.0 * std::f64::consts::PI * n2 * 100.0 / 550.0;
        let r12 = (n1 - n2) / (n1 + n2);
        let r23 = (n2 - n3) / (n2 + n3);
        let t12 = 2.0 * n1 / (n1 + n2);
        let t23 = 2.0 * n2 / (n2 + n3);
        let exp2d = (2.0 * i_unit * delta).exp();
        let exp1d = (i_unit * delta).exp();
        let r_check = (r12 + r23 * exp2d) / (Complex64::new(1.0, 0.0) + r12 * r23 * exp2d);
        let t_check = (t12 * t23 * exp1d) / (Complex64::new(1.0, 0.0) + r12 * r23 * exp2d);
        let r_check_sq = r_check.norm_sqr();
        let t_check_sq = t_check.norm_sqr() * (n3.re / n1.re);
        assert_relative_eq!(result.reflectance, r_check_sq, epsilon = 1e-10);
        assert_relative_eq!(result.transmittance.unwrap(), t_check_sq, epsilon = 1e-10);

        let t_val = result.transmittance.unwrap();
        let a = 1.0 - result.reflectance - t_val;
        assert!(a > 0.0, "absorptance should be positive, got {}", a);
        assert!(a < 1.0, "absorptance should be < 1, got {}", a);
        assert!(result.reflectance + t_val < 1.0);
    }

    #[test]
    fn test_tmm_quarter_wave_transmittance() {
        // Quarter-wave AR coating: T should be high (> 0.95)
        let n_inc = Complex64::new(1.0, 0.0);
        let n_ar = Complex64::new(1.38, 0.0);
        let n_sub = Complex64::new(1.5, 0.0);
        let wavelength = 550.0;
        let d_ar = wavelength / (4.0 * n_ar.re);
        let result = tmm_reflection(&[n_inc, n_ar, n_sub], &[d_ar], wavelength, 0.0, true);
        let t = result.transmittance.unwrap();
        assert!(t > 0.95, "T should be > 0.95, got {}", t);
        // Energy conservation
        let sum = result.reflectance + t;
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tmm_no_layers_fresnel_transmittance() {
        // Bare air/glass interface: T = 1 - R
        let n_layers = vec![Complex64::new(1.0, 0.0), Complex64::new(1.5, 0.0)];
        let result = tmm_reflection(&n_layers, &[], 500.0, 0.0, true);
        let t = result.transmittance.unwrap();
        assert_relative_eq!(result.reflectance + t, 1.0, epsilon = 1e-10);
        // Fresnel T = 4*n1*n2/(n1+n2)^2 = 4*1.5/6.25 = 0.96
        assert_relative_eq!(t, 0.96, epsilon = 0.01);
    }

    #[test]
    fn test_kramers_kronig_runs() {
        // Verify KK check runs and produces reasonable output
        let omega: Vec<f64> = (1..100).map(|i| i as f64 * 0.1).collect();
        let eps = drude(&omega, 1.0, 2.0, 0.5);

        // Check real from imaginary
        let kk_result = kramers_kronig_check(&eps, true, Some(1.0));

        // Verify we get output of correct length
        assert_eq!(kk_result.reconstructed.len(), eps.len());

        // Verify max_rel_error is finite
        assert!(kk_result.max_rel_error.is_finite());

        // Note: FFT-based KK has significant boundary errors; for rigorous checks,
        // use principal-value integration or windowing. This test verifies the
        // algorithm runs and produces finite results.
    }
}
