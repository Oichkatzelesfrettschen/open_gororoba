//! Fractional Schrodinger absorber with Pareto-optimal parameter sweep.
//!
//! Implements a split-step FFT solver for the fractional Schrodinger equation
//! with polynomial absorbing boundary layers. The Pareto sweep identifies
//! configurations that minimize both edge mass leakage and interior wavefunction
//! distortion simultaneously.
//!
//! # References
//! - Laskin (2000): Fractional quantum mechanics, Phys. Rev. E 62, 3135
//! - Muga et al. (2004): Complex absorbing potentials, Phys. Rep. 395, 357

use num_complex::Complex64;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Trapezoidal integration of a uniformly sampled function.
pub fn integrate_trapezoidal(y: &[f64], dx: f64) -> f64 {
    if y.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..(y.len() - 1) {
        sum += (y[i] + y[i + 1]) * 0.5 * dx;
    }
    sum
}

/// Configuration for the fractional Schrodinger split-step solver.
pub struct FractionalSchrodingerConfig {
    /// Number of spatial grid points.
    pub n: usize,
    /// Domain half-width (spatial domain is [-l_domain, +l_domain]).
    pub l_domain: f64,
    /// Fractional exponent (1 < alpha <= 2; alpha=2 is standard QM).
    pub alpha: f64,
    /// Diffusion coefficient D_alpha.
    pub d_alpha: f64,
    /// Time step.
    pub dt: f64,
    /// Total number of time steps.
    pub steps: usize,
}

/// Initial Gaussian wave packet parameters.
pub struct WavePacketConfig {
    /// Center position.
    pub x0: f64,
    /// Central wavenumber.
    pub k0: f64,
    /// Width (standard deviation).
    pub sigma: f64,
}

/// Absorbing boundary layer parameters.
pub struct AbsorberParams {
    /// Absorption strength.
    pub eta: f64,
    /// Polynomial order of the absorbing mask.
    pub m_order: i32,
    /// Onset position (absorber activates for |x| > xc).
    pub xc: f64,
}

/// A point on the Pareto frontier (edge mass vs interior distortion).
pub struct ParetoPoint {
    /// Residual probability density beyond the absorber onset + delta.
    pub m_edge: f64,
    /// L2 norm of interior wavefunction distortion relative to free evolution.
    pub e_int: f64,
    /// Polynomial order used.
    pub m: i32,
    /// Absorption strength used.
    pub eta: f64,
    /// Onset position used.
    pub xc: f64,
}

/// Evolve a Gaussian wave packet under the fractional Schrodinger equation,
/// optionally with an absorbing boundary layer.
///
/// Returns the final wavefunction psi(x) as a Vec<Complex64>.
pub fn fractional_schrodinger_evolve(
    cfg: &FractionalSchrodingerConfig,
    wp: &WavePacketConfig,
    absorber: Option<&AbsorberParams>,
) -> Vec<Complex64> {
    let n = cfg.n;
    let dx = (2.0 * cfg.l_domain) / n as f64;

    let x: Vec<f64> = (0..n).map(|i| -cfg.l_domain + i as f64 * dx).collect();

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

    // Initial wave packet
    let mut psi: Vec<Complex64> = x
        .iter()
        .map(|&xi| {
            let amp = (-0.5 * ((xi - wp.x0) / wp.sigma).powi(2)).exp();
            let phase = wp.k0 * xi;
            Complex64::new(amp * phase.cos(), amp * phase.sin())
        })
        .collect();

    // Normalize
    let density: Vec<f64> = psi.iter().map(|p| p.norm_sqr()).collect();
    let norm = integrate_trapezoidal(&density, dx).sqrt();
    for p in &mut psi {
        *p /= norm;
    }

    // Kinetic phase factor in k-space
    let phase_k: Vec<Complex64> = k
        .iter()
        .map(|&ki| {
            let phase = -cfg.d_alpha * ki.abs().powf(cfg.alpha) * cfg.dt;
            Complex64::new(phase.cos(), phase.sin())
        })
        .collect();

    // Absorber mask in x-space
    let absorb: Vec<f64> = match absorber {
        Some(abs) => x
            .iter()
            .map(|&xi| {
                if xi.abs() > abs.xc {
                    (-abs.eta * (xi.abs() - abs.xc).powi(abs.m_order)).exp()
                } else {
                    1.0
                }
            })
            .collect(),
        None => vec![1.0; n],
    };

    // Split-step FFT time evolution
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);
    let scale = 1.0 / n as f64;

    for _ in 0..cfg.steps {
        fft.process(&mut psi);
        for (p, pk) in psi.iter_mut().zip(phase_k.iter()) {
            *p *= pk;
        }
        ifft.process(&mut psi);
        for (p, &ab) in psi.iter_mut().zip(absorb.iter()) {
            *p *= scale * ab;
        }
    }

    psi
}

/// Run a Pareto-optimal parameter sweep over absorber configurations.
///
/// For each combination of (m_order, eta, xc), evolves the wavefunction with
/// and without the absorber, then computes edge mass and interior distortion
/// relative to free evolution.
///
/// `delta` is the buffer zone width: edge mass is measured for |x| > xc + delta,
/// interior distortion for |x| < xc - delta.
pub fn pareto_sweep(
    cfg: &FractionalSchrodingerConfig,
    wp: &WavePacketConfig,
    orders: &[i32],
    etas: &[f64],
    xcs: &[f64],
    delta: f64,
) -> Vec<ParetoPoint> {
    let n = cfg.n;
    let dx = (2.0 * cfg.l_domain) / n as f64;
    let x: Vec<f64> = (0..n).map(|i| -cfg.l_domain + i as f64 * dx).collect();

    // Free evolution baseline (no absorber)
    let psi_free = fractional_schrodinger_evolve(cfg, wp, None);

    let mut points = Vec::with_capacity(orders.len() * etas.len() * xcs.len());

    for &m in orders {
        for &eta in etas {
            for &xc in xcs {
                let abs = AbsorberParams {
                    eta,
                    m_order: m,
                    xc,
                };
                let psi_abs = fractional_schrodinger_evolve(cfg, wp, Some(&abs));

                // Edge mass: probability beyond xc + delta
                let edge_density: Vec<f64> = (0..n)
                    .filter(|&i| x[i].abs() > xc + delta)
                    .map(|i| psi_abs[i].norm_sqr())
                    .collect();
                let m_edge = integrate_trapezoidal(&edge_density, dx);

                // Interior distortion: L2 norm of difference inside xc - delta
                let int_diff_sq: Vec<f64> = (0..n)
                    .filter(|&i| x[i].abs() < xc - delta)
                    .map(|i| (psi_abs[i] - psi_free[i]).norm_sqr())
                    .collect();
                let e_int = integrate_trapezoidal(&int_diff_sq, dx).sqrt();

                points.push(ParetoPoint {
                    m_edge,
                    e_int,
                    m,
                    eta,
                    xc,
                });
            }
        }
    }

    points
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrate_trapezoidal_constant() {
        let y = vec![2.0; 100];
        let result = integrate_trapezoidal(&y, 0.1);
        assert!((result - 19.8).abs() < 1e-10);
    }

    #[test]
    fn test_free_evolution_preserves_norm() {
        let cfg = FractionalSchrodingerConfig {
            n: 256,
            l_domain: 50.0,
            alpha: 2.0,
            d_alpha: 0.5,
            dt: 0.1,
            steps: 10,
        };
        let wp = WavePacketConfig {
            x0: 0.0,
            k0: 0.0,
            sigma: 5.0,
        };
        let psi = fractional_schrodinger_evolve(&cfg, &wp, None);
        let dx = (2.0 * cfg.l_domain) / cfg.n as f64;
        let density: Vec<f64> = psi.iter().map(|p| p.norm_sqr()).collect();
        let norm = integrate_trapezoidal(&density, dx);
        assert!(
            (norm - 1.0).abs() < 0.02,
            "norm should be ~1.0, got {}",
            norm
        );
    }

    #[test]
    fn test_absorber_reduces_edge_mass() {
        let cfg = FractionalSchrodingerConfig {
            n: 256,
            l_domain: 50.0,
            alpha: 1.5,
            d_alpha: 0.5,
            dt: 0.25,
            steps: 40,
        };
        let wp = WavePacketConfig {
            x0: -30.0,
            k0: 1.0,
            sigma: 5.0,
        };
        let points = pareto_sweep(&cfg, &wp, &[4], &[1e-3], &[30.0], 5.0);
        assert_eq!(points.len(), 1);
        assert!(
            points[0].m_edge < 0.5,
            "absorber should reduce edge mass, got {}",
            points[0].m_edge
        );
    }
}
