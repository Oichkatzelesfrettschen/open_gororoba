//! Gross-Pitaevskii equation solver for BEC dynamics.
//!
//! Extends the split-operator method from `fractional_schrodinger` with
//! the nonlinear |psi|^2 mean-field potential for weakly-interacting
//! Bose-Einstein condensates.
//!
//! # Equation
//! i hbar dpsi/dt = (-hbar^2/(2m) nabla^2 + V_ext + g|psi|^2) psi
//!
//! In natural units (hbar = m = 1):
//! i dpsi/dt = (-1/2 nabla^2 + V + g|psi|^2) psi
//!
//! # Method
//! Strang splitting: V_eff half -> T full -> V_eff half
//! where V_eff(x) = V(x) + g*|psi(x)|^2 is updated each step.
//!
//! # Literature
//! - Gross (1961): Structure of a quantized vortex in boson systems
//! - Pitaevskii (1961): Vortex lines in an imperfect Bose gas
//! - Bao, Jaksch & Markowich (2003): Numerical solution of GPE

use num_complex::Complex64;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Configuration for the GPE solver.
#[derive(Debug, Clone)]
pub struct GpeConfig {
    /// Spatial grid points.
    pub x: Vec<f64>,
    /// External potential V(x).
    pub v_ext: Vec<f64>,
    /// Interaction strength g = 4*pi*a_s*N/L^(d-1) in natural units.
    pub g: f64,
    /// Time step.
    pub dt: f64,
}

/// Result of GPE evolution.
#[derive(Debug, Clone)]
pub struct GpeResult {
    /// Final wavefunction.
    pub psi: Vec<Complex64>,
    /// Spatial grid.
    pub x: Vec<f64>,
    /// Density |psi|^2.
    pub density: Vec<f64>,
    /// L2 norm of psi (should be ~1 for normalized).
    pub norm: f64,
    /// Total energy <H>.
    pub energy: f64,
}

/// Compute the total energy <psi|H|psi> of the GPE state.
///
/// E = int [ (hbar^2/2m)|grad(psi)|^2 + V|psi|^2 + g/2 |psi|^4 ] dx
///
/// In natural units: E = int [ 1/2 |grad(psi)|^2 + V|psi|^2 + g/2 |psi|^4 ] dx
fn compute_energy(psi: &[Complex64], x: &[f64], v: &[f64], g: f64) -> f64 {
    let n = psi.len();
    let dx = x[1] - x[0];

    // Kinetic energy via finite differences: -1/2 * psi* * d^2 psi/dx^2
    let mut e_kin = 0.0;
    for i in 1..n - 1 {
        let d2psi = (psi[i + 1] - 2.0 * psi[i] + psi[i - 1]) / (dx * dx);
        e_kin += (-0.5 * psi[i].conj() * d2psi).re;
    }
    e_kin *= dx;

    // Potential energy: int V |psi|^2 dx
    let e_pot: f64 = psi
        .iter()
        .zip(v.iter())
        .map(|(p, &vi)| vi * p.norm_sqr())
        .sum::<f64>()
        * dx;

    // Interaction energy: g/2 int |psi|^4 dx
    let e_int: f64 = psi.iter().map(|p| p.norm_sqr().powi(2)).sum::<f64>() * (g / 2.0) * dx;

    e_kin + e_pot + e_int
}

/// Evolve the GPE using split-operator Strang splitting.
///
/// At each step, the effective potential includes the nonlinear term:
/// V_eff(x) = V_ext(x) + g * |psi(x)|^2
///
/// # Arguments
/// * `psi0` - Initial wavefunction
/// * `cfg` - Solver configuration
/// * `n_steps` - Number of time steps
/// * `normalize` - Whether to renormalize each step
pub fn gpe_evolve(
    psi0: &[Complex64],
    cfg: &GpeConfig,
    n_steps: usize,
    normalize: bool,
) -> GpeResult {
    let n = psi0.len();
    let dx = cfg.x[1] - cfg.x[0];

    // Build k-space grid
    let k: Vec<f64> = (0..n)
        .map(|i| {
            let freq = if i <= n / 2 {
                i as f64
            } else {
                i as f64 - n as f64
            };
            2.0 * PI * freq / (n as f64 * dx)
        })
        .collect();

    // Kinetic exponential: exp(-i * k^2/2 * dt)
    let exp_t: Vec<Complex64> = k
        .iter()
        .map(|&ki| Complex64::new(0.0, -0.5 * ki * ki * cfg.dt).exp())
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut psi = psi0.to_vec();

    for _ in 0..n_steps {
        // Build effective potential: V_ext + g*|psi|^2
        // V_eff half-step
        for (p, &vi) in psi.iter_mut().zip(cfg.v_ext.iter()) {
            let v_eff = vi + cfg.g * p.norm_sqr();
            *p *= Complex64::new(0.0, -v_eff * cfg.dt / 2.0).exp();
        }

        // Kinetic full step in Fourier space
        fft.process(&mut psi);
        for (p, &et) in psi.iter_mut().zip(exp_t.iter()) {
            *p *= et;
        }
        ifft.process(&mut psi);
        let scale = 1.0 / n as f64;
        for p in &mut psi {
            *p *= scale;
        }

        // V_eff half-step (recompute with updated |psi|^2)
        for (p, &vi) in psi.iter_mut().zip(cfg.v_ext.iter()) {
            let v_eff = vi + cfg.g * p.norm_sqr();
            *p *= Complex64::new(0.0, -v_eff * cfg.dt / 2.0).exp();
        }

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
    let energy = compute_energy(&psi, &cfg.x, &cfg.v_ext, cfg.g);

    GpeResult {
        psi,
        x: cfg.x.clone(),
        density,
        norm,
        energy,
    }
}

/// Find the GPE ground state via imaginary-time evolution.
///
/// Replaces t -> -i*tau so the propagator becomes exp(-H*tau),
/// which projects onto the ground state. Renormalization is mandatory.
pub fn gpe_ground_state(cfg: &GpeConfig, tau: f64, n_steps: usize) -> GpeResult {
    let n = cfg.x.len();
    let dx = cfg.x[1] - cfg.x[0];
    let l = (n as f64) * dx;

    // Initial Gaussian guess
    let sigma = l / 8.0;
    let center = cfg.x[n / 2];
    let mut psi: Vec<Complex64> = cfg
        .x
        .iter()
        .map(|&xi| {
            let arg = -((xi - center).powi(2)) / (2.0 * sigma * sigma);
            Complex64::new(arg.exp(), 0.0)
        })
        .collect();

    // Normalize
    let norm0: f64 = psi.iter().map(|p| p.norm_sqr()).sum::<f64>().sqrt() * dx.sqrt();
    for p in &mut psi {
        *p /= norm0;
    }

    // Imaginary-time config: dt -> -i*tau means we use real exponentials
    // exp(-V_eff * tau/2) and exp(-k^2/2 * tau) instead of complex ones.
    let k: Vec<f64> = (0..n)
        .map(|i| {
            let freq = if i <= n / 2 {
                i as f64
            } else {
                i as f64 - n as f64
            };
            2.0 * PI * freq / (n as f64 * dx)
        })
        .collect();

    let exp_t: Vec<f64> = k.iter().map(|&ki| (-0.5 * ki * ki * tau).exp()).collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    for _ in 0..n_steps {
        // V_eff half-step (real exponential for imaginary time)
        for (p, &vi) in psi.iter_mut().zip(cfg.v_ext.iter()) {
            let v_eff = vi + cfg.g * p.norm_sqr();
            *p *= (-v_eff * tau / 2.0).exp();
        }

        // Kinetic full step
        fft.process(&mut psi);
        for (p, &et) in psi.iter_mut().zip(exp_t.iter()) {
            *p *= et;
        }
        ifft.process(&mut psi);
        let scale = 1.0 / n as f64;
        for p in &mut psi {
            *p *= scale;
        }

        // V_eff half-step
        for (p, &vi) in psi.iter_mut().zip(cfg.v_ext.iter()) {
            let v_eff = vi + cfg.g * p.norm_sqr();
            *p *= (-v_eff * tau / 2.0).exp();
        }

        // Renormalize (mandatory for imaginary time)
        let norm: f64 = psi.iter().map(|p| p.norm_sqr()).sum::<f64>().sqrt() * dx.sqrt();
        if norm > 1e-30 {
            for p in &mut psi {
                *p /= norm;
            }
        }
    }

    let density: Vec<f64> = psi.iter().map(|p| p.norm_sqr()).collect();
    let norm = psi.iter().map(|p| p.norm_sqr()).sum::<f64>().sqrt() * dx.sqrt();
    let energy = compute_energy(&psi, &cfg.x, &cfg.v_ext, cfg.g);

    GpeResult {
        psi,
        x: cfg.x.clone(),
        density,
        norm,
        energy,
    }
}

/// Create a harmonic potential V(x) = 0.5 * omega^2 * x^2.
pub fn harmonic_potential(x: &[f64], omega: f64) -> Vec<f64> {
    x.iter().map(|&xi| 0.5 * omega * omega * xi * xi).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_grid(n: usize, l: f64) -> Vec<f64> {
        let dx = l / n as f64;
        (0..n).map(|i| -l / 2.0 + (i as f64 + 0.5) * dx).collect()
    }

    #[test]
    fn test_gpe_ground_state_norm() {
        // Ground state should be normalized to 1
        let n = 256;
        let l = 20.0;
        let x = test_grid(n, l);
        let v = harmonic_potential(&x, 1.0);
        let cfg = GpeConfig {
            x,
            v_ext: v,
            g: 10.0,
            dt: 0.01,
        };
        let result = gpe_ground_state(&cfg, 0.01, 5000);
        eprintln!("GPE ground state norm = {:.6}", result.norm);
        assert!(
            (result.norm - 1.0).abs() < 0.01,
            "ground state norm={}, expected 1.0",
            result.norm
        );
    }

    #[test]
    fn test_gpe_ground_state_energy() {
        // With g=0, should recover harmonic oscillator ground state E = omega/2
        let n = 256;
        let l = 20.0;
        let x = test_grid(n, l);
        let omega = 1.0;
        let v = harmonic_potential(&x, omega);
        let cfg = GpeConfig {
            x,
            v_ext: v,
            g: 0.0, // Linear case
            dt: 0.01,
        };
        let result = gpe_ground_state(&cfg, 0.01, 5000);
        let expected_energy = omega / 2.0; // hbar*omega/2 in natural units
        eprintln!(
            "Linear GPE ground state energy = {:.6}, expected = {:.6}",
            result.energy, expected_energy
        );
        assert!(
            (result.energy - expected_energy).abs() < 0.1,
            "energy={}, expected {}",
            result.energy,
            expected_energy
        );
    }

    #[test]
    fn test_gpe_interaction_raises_energy() {
        // Repulsive interaction (g > 0) should raise the energy above omega/2
        let n = 256;
        let l = 20.0;
        let x = test_grid(n, l);
        let v = harmonic_potential(&x, 1.0);

        let cfg_linear = GpeConfig {
            x: x.clone(),
            v_ext: v.clone(),
            g: 0.0,
            dt: 0.01,
        };
        let cfg_nonlinear = GpeConfig {
            x,
            v_ext: v,
            g: 50.0,
            dt: 0.01,
        };

        let e_linear = gpe_ground_state(&cfg_linear, 0.01, 5000).energy;
        let e_nonlinear = gpe_ground_state(&cfg_nonlinear, 0.01, 5000).energy;

        eprintln!(
            "E_linear = {:.4}, E_nonlinear = {:.4}",
            e_linear, e_nonlinear
        );
        assert!(
            e_nonlinear > e_linear,
            "repulsive interaction should raise energy: {} vs {}",
            e_nonlinear,
            e_linear
        );
    }

    #[test]
    fn test_gpe_evolve_norm_preservation() {
        // Real-time evolution should preserve norm (without normalization)
        let n = 128;
        let l = 20.0;
        let x = test_grid(n, l);
        let dx = l / n as f64;
        let v = harmonic_potential(&x, 1.0);
        let cfg = GpeConfig {
            x: x.clone(),
            v_ext: v,
            g: 5.0,
            dt: 0.005,
        };

        // Start with normalized Gaussian
        let sigma = 1.0;
        let psi0: Vec<Complex64> = x
            .iter()
            .map(|&xi| Complex64::new((-xi * xi / (2.0 * sigma * sigma)).exp(), 0.0))
            .collect();
        let norm0: f64 = psi0.iter().map(|p| p.norm_sqr()).sum::<f64>().sqrt() * dx.sqrt();
        let psi0: Vec<Complex64> = psi0.iter().map(|p| p / norm0).collect();

        let result = gpe_evolve(&psi0, &cfg, 200, false);
        eprintln!("Norm after 200 steps = {:.8}", result.norm);
        // Strang splitting preserves norm to high accuracy for small dt
        assert!(
            (result.norm - 1.0).abs() < 0.05,
            "norm should be preserved: {}",
            result.norm
        );
    }

    #[test]
    fn test_gpe_ground_state_density_centered() {
        // Ground state density should be centered on the trap
        let n = 256;
        let l = 20.0;
        let x = test_grid(n, l);
        let v = harmonic_potential(&x, 1.0);
        let cfg = GpeConfig {
            x: x.clone(),
            v_ext: v,
            g: 10.0,
            dt: 0.01,
        };
        let result = gpe_ground_state(&cfg, 0.01, 5000);

        // Find peak density position
        let (peak_idx, _) = result
            .density
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        let peak_x = x[peak_idx];
        eprintln!("Peak density at x = {:.3}", peak_x);
        assert!(
            peak_x.abs() < 1.0,
            "peak should be near center, got x={}",
            peak_x
        );
    }
}
