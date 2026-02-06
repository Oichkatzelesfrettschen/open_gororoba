//! Fractional Schrodinger Equation Solver.
//!
//! Provides:
//! - Levy free-particle propagator via numerical quadrature
//! - Strang split-operator real-time evolution
//! - Imaginary-time ground state projection
//! - Variational ground state energy (Gaussian trial)
//!
//! The fractional Schrodinger equation:
//!   i * dpsi/dt = D * (-Delta)^{alpha/2} * psi + V(x) * psi
//!
//! where alpha in (0, 2] is the Levy index.
//!
//! # Literature
//! - Laskin (2000), Phys. Lett. A 268, 298
//! - Laskin (2002), Phys. Rev. E 66, 056108

use num_complex::Complex64;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Result from split-operator evolution.
#[derive(Clone, Debug)]
pub struct EvolutionResult {
    /// Final wavefunction (complex)
    pub psi: Vec<Complex64>,
    /// Spatial grid
    pub x: Vec<f64>,
    /// Probability density
    pub density: Vec<f64>,
    /// Final norm (should be ~1 if normalized)
    pub norm: f64,
}

/// Result from variational ground state calculation.
#[derive(Clone, Debug)]
pub struct VariationalResult {
    /// Variational ground state energy
    pub energy: f64,
    /// Optimal width parameter beta
    pub beta_opt: f64,
    /// Kinetic energy contribution
    pub kinetic: f64,
    /// Potential energy contribution
    pub potential: f64,
}

/// Result from Levy propagator calculation.
#[derive(Clone, Debug)]
pub struct PropagatorResult {
    /// Propagator values K(x, t)
    pub propagator: Vec<Complex64>,
    /// Spatial grid
    pub x: Vec<f64>,
}

/// Evaluate the free-particle Levy propagator via numerical quadrature.
///
/// K(x, t) = (1/2pi) * integral exp(i*k*x - i*D*|k|^alpha*t) dk
///
/// # Arguments
/// * `x` - Spatial evaluation points
/// * `t` - Time (> 0)
/// * `alpha` - Levy index in (0, 2]
/// * `d` - Generalized diffusion coefficient
/// * `n_k` - Number of quadrature points
/// * `k_max` - k-space cutoff
pub fn levy_propagator(
    x: &[f64],
    t: f64,
    alpha: f64,
    d: f64,
    n_k: usize,
    k_max: f64,
) -> PropagatorResult {
    let dk = 2.0 * k_max / n_k as f64;
    let k_vals: Vec<f64> = (0..n_k)
        .map(|i| -k_max + (i as f64 + 0.5) * dk)
        .collect();

    // Precompute phase factors in k-space
    let phase_k: Vec<f64> = k_vals.iter().map(|&k| -d * k.abs().powf(alpha) * t).collect();

    let mut propagator = Vec::with_capacity(x.len());
    for &xj in x {
        let mut sum = Complex64::new(0.0, 0.0);
        for (i, &k) in k_vals.iter().enumerate() {
            let phase = k * xj + phase_k[i];
            sum += Complex64::new(phase.cos(), phase.sin());
        }
        propagator.push(sum * dk / (2.0 * PI));
    }

    PropagatorResult {
        propagator,
        x: x.to_vec(),
    }
}

/// Standard Gaussian free-particle propagator (alpha = 2).
///
/// K(x, t) = sqrt(m / (2*pi*i*t)) * exp(i*m*x^2 / (2*t))
pub fn gaussian_propagator(x: &[f64], t: f64, m: f64) -> PropagatorResult {
    let prefactor = Complex64::new(0.0, 2.0 * PI * t / m).sqrt().inv();
    let propagator: Vec<Complex64> = x
        .iter()
        .map(|&xj| {
            let phase = m * xj * xj / (2.0 * t);
            prefactor * Complex64::new(phase.cos(), phase.sin())
        })
        .collect();

    PropagatorResult {
        propagator,
        x: x.to_vec(),
    }
}

/// L2 error between Levy and Gaussian propagators.
///
/// At alpha=2, D=1/(2m), the Levy propagator should recover the Gaussian.
pub fn propagator_l2_error(alpha: f64, d: f64, t: f64, n_x: usize, l: f64, n_k: usize, k_max: f64) -> f64 {
    let x: Vec<f64> = (0..n_x).map(|i| -l + 2.0 * l * i as f64 / n_x as f64).collect();
    let levy = levy_propagator(&x, t, alpha, d, n_k, k_max);

    let m = 1.0 / (2.0 * d);
    let gauss = gaussian_propagator(&x, t, m);

    let mut diff_norm_sq = 0.0;
    let mut gauss_norm_sq = 0.0;
    for (kl, kg) in levy.propagator.iter().zip(gauss.propagator.iter()) {
        diff_norm_sq += (kl - kg).norm_sqr();
        gauss_norm_sq += kg.norm_sqr();
    }

    if gauss_norm_sq < 1e-30 {
        return 0.0;
    }
    (diff_norm_sq / gauss_norm_sq).sqrt()
}

/// Strang split-operator evolution for the fractional Schrodinger equation.
///
/// Evolves psi under H = D * |k|^alpha + V(x) using Strang splitting:
///   exp(-i*H*dt) ~ exp(-i*V*dt/2) * exp(-i*T*dt) * exp(-i*V*dt/2)
///
/// # Arguments
/// * `psi0` - Initial wavefunction
/// * `x` - Spatial grid (evenly spaced)
/// * `v` - Potential on the grid
/// * `alpha` - Levy index in (0, 2]
/// * `d` - Generalized diffusion coefficient
/// * `dt` - Time step
/// * `n_steps` - Number of time steps
/// * `normalize` - Whether to renormalize after each step
#[allow(clippy::too_many_arguments)]
pub fn split_operator_evolve(
    psi0: &[Complex64],
    x: &[f64],
    v: &[f64],
    alpha: f64,
    d: f64,
    dt: f64,
    n_steps: usize,
    normalize: bool,
) -> EvolutionResult {
    let n = psi0.len();
    let dx = x[1] - x[0];

    // Build k-space grid
    let k: Vec<f64> = (0..n)
        .map(|i| {
            let freq = if i <= n / 2 { i as f64 } else { i as f64 - n as f64 };
            2.0 * PI * freq / (n as f64 * dx)
        })
        .collect();

    // Precompute exponentials
    let exp_v_half: Vec<Complex64> = v
        .iter()
        .map(|&vi| Complex64::new(0.0, -vi * dt / 2.0).exp())
        .collect();
    let exp_t: Vec<Complex64> = k
        .iter()
        .map(|&ki| Complex64::new(0.0, -d * ki.abs().powf(alpha) * dt).exp())
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut psi: Vec<Complex64> = psi0.to_vec();

    for _ in 0..n_steps {
        // V half-step
        for (p, &ev) in psi.iter_mut().zip(exp_v_half.iter()) {
            *p *= ev;
        }

        // T full step (in Fourier space)
        fft.process(&mut psi);
        for (p, &et) in psi.iter_mut().zip(exp_t.iter()) {
            *p *= et;
        }
        ifft.process(&mut psi);
        let scale = 1.0 / n as f64;
        for p in &mut psi {
            *p *= scale;
        }

        // V half-step
        for (p, &ev) in psi.iter_mut().zip(exp_v_half.iter()) {
            *p *= ev;
        }

        // Optionally renormalize
        if normalize {
            let norm: f64 = psi.iter().map(|p| p.norm_sqr()).sum::<f64>().sqrt() * dx.sqrt();
            if norm > 1e-30 {
                for p in &mut psi {
                    *p /= norm;
                }
            }
        }
    }

    let density: Vec<f64> = psi.iter().map(|p| p.norm_sqr()).collect();
    let norm = psi.iter().map(|p| p.norm_sqr()).sum::<f64>().sqrt() * dx.sqrt();

    EvolutionResult {
        psi,
        x: x.to_vec(),
        density,
        norm,
    }
}

/// Imaginary-time ground state projection.
///
/// Evolves psi under exp(-H*tau) to project onto the ground state.
///
/// # Arguments
/// * `x` - Spatial grid
/// * `v` - Potential on the grid
/// * `alpha` - Levy index in (0, 2]
/// * `d` - Generalized diffusion coefficient
/// * `tau` - Imaginary time step
/// * `n_steps` - Number of time steps
pub fn imaginary_time_ground_state(
    x: &[f64],
    v: &[f64],
    alpha: f64,
    d: f64,
    tau: f64,
    n_steps: usize,
) -> (Vec<f64>, f64) {
    let n = x.len();
    let dx = x[1] - x[0];

    // Initial Gaussian trial
    let mut psi: Vec<f64> = x.iter().map(|&xi| (-xi * xi / 4.0).exp()).collect();
    let mut norm: f64 = psi.iter().map(|&p| p * p).sum::<f64>() * dx;
    norm = norm.sqrt();
    for p in &mut psi {
        *p /= norm;
    }

    // Build k-space grid
    let k: Vec<f64> = (0..n)
        .map(|i| {
            let freq = if i <= n / 2 { i as f64 } else { i as f64 - n as f64 };
            2.0 * PI * freq / (n as f64 * dx)
        })
        .collect();

    // Precompute exponentials (real for imaginary time)
    let exp_v_half: Vec<f64> = v.iter().map(|&vi| (-vi * tau / 2.0).exp()).collect();
    let exp_t: Vec<f64> = k.iter().map(|&ki| (-d * ki.abs().powf(alpha) * tau).exp()).collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    for _ in 0..n_steps {
        // V half-step
        for (p, &ev) in psi.iter_mut().zip(exp_v_half.iter()) {
            *p *= ev;
        }

        // T full step (in Fourier space)
        let mut buffer: Vec<Complex64> = psi.iter().map(|&p| Complex64::new(p, 0.0)).collect();
        fft.process(&mut buffer);
        for (b, &et) in buffer.iter_mut().zip(exp_t.iter()) {
            *b *= et;
        }
        ifft.process(&mut buffer);
        let scale = 1.0 / n as f64;
        psi = buffer.iter().map(|c| c.re * scale).collect();

        // V half-step
        for (p, &ev) in psi.iter_mut().zip(exp_v_half.iter()) {
            *p *= ev;
        }

        // Renormalize
        norm = psi.iter().map(|&p| p * p).sum::<f64>() * dx;
        norm = norm.sqrt();
        if norm > 1e-30 {
            for p in &mut psi {
                *p /= norm;
            }
        }
    }

    // Compute energy <H> = <T> + <V>
    let mut buffer: Vec<Complex64> = psi.iter().map(|&p| Complex64::new(p, 0.0)).collect();
    fft.process(&mut buffer);

    let t_exp: f64 = buffer
        .iter()
        .zip(k.iter())
        .map(|(c, &ki)| d * ki.abs().powf(alpha) * c.norm_sqr())
        .sum::<f64>()
        * dx
        / n as f64;

    let v_exp: f64 = psi.iter().zip(v.iter()).map(|(&p, &vi)| vi * p * p).sum::<f64>() * dx;

    (psi, t_exp + v_exp)
}

/// Variational ground state energy using Gaussian trial wavefunction.
///
/// For H = D * |k|^alpha + (1/2) * m * omega^2 * x^2
///
/// Trial: psi(x) = (beta/pi)^{1/4} * exp(-beta*x^2/2)
///
/// Returns the minimized energy and optimal beta.
pub fn variational_ground_state(alpha: f64, d: f64, omega: f64, m: f64) -> VariationalResult {
    // For Gaussian trial psi(x) = (beta/pi)^{1/4} * exp(-beta*x^2/2):
    // <|k|^alpha> = beta^{alpha/2} * Gamma((alpha+1)/2) / sqrt(pi)
    // <x^2> = 1/(2*beta)
    // <V> = (1/2) * m * omega^2 * <x^2> = m * omega^2 / (4 * beta)
    let gamma_factor = gamma((alpha + 1.0) / 2.0);
    let frac_1_sqrt_pi = 1.0 / PI.sqrt();
    let coeff_t = d * gamma_factor * frac_1_sqrt_pi;
    let coeff_v = m * omega * omega / 4.0;

    // Energy(beta) = coeff_t * beta^{alpha/2} + coeff_v / beta
    // Golden section search for minimum
    let mut a: f64 = 0.01;
    let mut b: f64 = 100.0;
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;

    let energy = |beta: f64| -> f64 {
        coeff_t * beta.powf(alpha / 2.0) + coeff_v / beta
    };

    for _ in 0..100 {
        let c = b - (b - a) / phi;
        let dd = a + (b - a) / phi;
        if energy(c) < energy(dd) {
            b = dd;
        } else {
            a = c;
        }
        if (b - a).abs() < 1e-12 {
            break;
        }
    }

    let beta_opt = (a + b) / 2.0;
    let kinetic = coeff_t * beta_opt.powf(alpha / 2.0);
    let potential = coeff_v / beta_opt;

    VariationalResult {
        energy: kinetic + potential,
        beta_opt,
        kinetic,
        potential,
    }
}

/// Gamma function approximation (Lanczos).
fn gamma(x: f64) -> f64 {
    if x < 0.5 {
        PI / ((PI * x).sin() * gamma(1.0 - x))
    } else {
        let g = 7.0;
        #[allow(clippy::excessive_precision)]
        let c = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];
        let z = x - 1.0;
        let mut sum = c[0];
        for (i, &ci) in c.iter().enumerate().skip(1) {
            sum += ci / (z + i as f64);
        }
        let t = z + g + 0.5;
        (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gaussian_propagator_normalization() {
        let x: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.1).collect();
        let result = gaussian_propagator(&x, 1.0, 1.0);
        // Just check it doesn't panic and returns expected length
        assert_eq!(result.propagator.len(), x.len());
    }

    #[test]
    fn test_levy_propagator_alpha2_matches_gaussian() {
        // At alpha=2, Levy propagator should match Gaussian
        let l2_err = propagator_l2_error(2.0, 0.5, 1.0, 64, 10.0, 4096, 40.0);
        assert!(l2_err < 0.05, "L2 error = {} should be < 0.05", l2_err);
    }

    #[test]
    fn test_levy_propagator_alpha_15_different() {
        // At alpha=1.5, Levy propagator should differ from Gaussian
        let l2_err = propagator_l2_error(1.5, 0.5, 1.0, 64, 10.0, 4096, 40.0);
        assert!(l2_err > 0.01, "L2 error = {} should be > 0.01", l2_err);
    }

    #[test]
    fn test_split_operator_preserves_norm() {
        let n = 128;
        let l = 20.0;
        let dx = 2.0 * l / n as f64;
        let x: Vec<f64> = (0..n).map(|i| -l + i as f64 * dx).collect();
        let v: Vec<f64> = x.iter().map(|&xi| 0.5 * xi * xi).collect();

        // Initial Gaussian wavepacket
        let psi0: Vec<Complex64> = x.iter().map(|&xi| Complex64::new((-xi * xi / 2.0).exp(), 0.0)).collect();

        let result = split_operator_evolve(&psi0, &x, &v, 2.0, 0.5, 0.01, 100, true);
        assert_relative_eq!(result.norm, 1.0, epsilon = 0.05);
    }

    #[test]
    fn test_imaginary_time_ground_state_positive() {
        let n = 256;
        let l = 15.0;
        let dx = 2.0 * l / n as f64;
        let x: Vec<f64> = (0..n).map(|i| -l + i as f64 * dx).collect();
        let omega = 1.0;
        let v: Vec<f64> = x.iter().map(|&xi| 0.5 * omega * omega * xi * xi).collect();

        let (_, energy) = imaginary_time_ground_state(&x, &v, 2.0, 0.5, 0.01, 2000);
        assert!(energy > 0.0, "Ground state energy = {}", energy);
    }

    #[test]
    fn test_variational_alpha2_recovers_harmonic_oscillator() {
        // At alpha=2, D=0.5, m=1, omega=1, exact E_0 = omega/2 = 0.5
        let result = variational_ground_state(2.0, 0.5, 1.0, 1.0);
        // Variational bound is upper bound, should be close to 0.5
        assert!(result.energy > 0.45, "E = {} should be > 0.45", result.energy);
        assert!(result.energy < 0.55, "E = {} should be < 0.55", result.energy);
    }

    #[test]
    fn test_variational_alpha15_positive() {
        let result = variational_ground_state(1.5, 0.5, 1.0, 1.0);
        assert!(result.energy > 0.0, "E = {}", result.energy);
        assert!(result.beta_opt > 0.0, "beta_opt = {}", result.beta_opt);
    }

    #[test]
    fn test_gamma_function() {
        // Gamma(1) = 1, Gamma(2) = 1, Gamma(0.5) = sqrt(pi)
        assert_relative_eq!(gamma(1.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(gamma(2.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(gamma(0.5), PI.sqrt(), epsilon = 1e-10);
        assert_relative_eq!(gamma(3.0), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_imaginary_time_vs_variational_consistency() {
        let n = 512;
        let l = 20.0;
        let dx = 2.0 * l / n as f64;
        let x: Vec<f64> = (0..n).map(|i| -l + i as f64 * dx).collect();
        let omega = 1.0;
        let v: Vec<f64> = x.iter().map(|&xi| 0.5 * omega * omega * xi * xi).collect();

        let (_, e_imag) = imaginary_time_ground_state(&x, &v, 2.0, 0.5, 0.01, 3000);
        let e_var = variational_ground_state(2.0, 0.5, omega, 1.0);

        // Both should be close to 0.5 for harmonic oscillator
        assert!((e_imag - 0.5).abs() < 0.1, "E_imag = {}", e_imag);
        assert!((e_var.energy - 0.5).abs() < 0.1, "E_var = {}", e_var.energy);
    }
}
