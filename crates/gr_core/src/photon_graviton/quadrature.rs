//! Integration utilities for worldline proper-time integrals.
//!
//! Uses Gauss-Legendre quadrature from `gauss_quad` for both the proper-time
//! variable T in `[t_min, t_max]` and the worldline modulus u in `[0, 1]`.
//!
//! Strategy: the exp(-m^2 T) damping makes the proper-time integral converge
//! rapidly. With t_max = 30/m^2, truncation error is exp(-30) < 1e-13.
//! GL with 128 nodes gives exponential convergence for smooth integrands.

use gauss_quad::GaussLegendre;
use num_complex::Complex64;
use std::num::NonZeroUsize;

/// Configuration for numerical quadrature.
#[derive(Debug, Clone, Copy)]
pub struct QuadratureConfig {
    /// Number of GL nodes for the u-integral `[0, 1]`.
    pub n_u: usize,
    /// Number of GL nodes for the proper-time integral.
    pub n_t: usize,
    /// Upper cutoff for proper-time (in units of 1/m^2). Default: 30.0.
    pub t_max: f64,
    /// Lower cutoff for proper-time. Default: 1e-8.
    pub t_min: f64,
}

impl Default for QuadratureConfig {
    fn default() -> Self {
        Self {
            n_u: 64,
            n_t: 128,
            t_max: 30.0,
            t_min: 1e-8,
        }
    }
}

impl QuadratureConfig {
    /// High-accuracy configuration (for Ward identity checks).
    pub fn high_accuracy() -> Self {
        Self {
            n_u: 96,
            n_t: 192,
            t_max: 40.0,
            t_min: 1e-10,
        }
    }

    /// Fast configuration (for exploratory computations).
    pub fn fast() -> Self {
        Self {
            n_u: 32,
            n_t: 64,
            t_max: 20.0,
            t_min: 1e-6,
        }
    }
}

/// Gauss-Legendre quadrature of a real function over `[a, b]`.
pub fn gl_integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, degree: usize) -> f64 {
    let quad =
        GaussLegendre::new(NonZeroUsize::new(degree).expect("quadrature degree must be non-zero"));
    quad.integrate(a, b, f)
}

/// Gauss-Legendre quadrature of a complex function over `[a, b]`.
pub fn gl_integrate_complex<F: Fn(f64) -> Complex64>(
    f: F,
    a: f64,
    b: f64,
    degree: usize,
) -> Complex64 {
    let quad =
        GaussLegendre::new(NonZeroUsize::new(degree).expect("quadrature degree must be non-zero"));
    // GaussLegendre only integrates f64, so we split into real and imaginary.
    let pairs = quad.as_node_weight_pairs();
    let half_len = 0.5 * (b - a);
    let mid = 0.5 * (a + b);
    let mut sum = Complex64::new(0.0, 0.0);
    for &(node, weight) in pairs {
        let x = mid + half_len * node;
        sum += f(x) * (half_len * weight);
    }
    sum
}

/// Proper-time integral: integral_t_min^t_max dT / T^power * exp(-m^2 * T) * f(T).
///
/// The mass_sq parameter is m^2 in natural units (= 1 for electrons).
/// The 1/T^power and exp(-m^2 T) factors are included in the measure.
pub fn proper_time_integral<F: Fn(f64) -> Complex64>(
    f: F,
    mass_sq: f64,
    power: f64,
    config: &QuadratureConfig,
) -> Complex64 {
    gl_integrate_complex(
        |t| {
            let measure = t.powf(-power) * (-mass_sq * t).exp();
            f(t) * measure
        },
        config.t_min,
        config.t_max,
        config.n_t,
    )
}

/// Double integral over proper-time T and worldline modulus u in `[0, 1]`:
///
///   integral dT/T^power * exp(-m^2 T) * integral_0^1 du * f(T, u)
pub fn double_integral<F: Fn(f64, f64) -> Complex64>(
    f: F,
    mass_sq: f64,
    power: f64,
    config: &QuadratureConfig,
) -> Complex64 {
    let quad_u = GaussLegendre::new(
        NonZeroUsize::new(config.n_u).expect("u-quadrature degree must be non-zero"),
    );
    let u_pairs: Vec<(f64, f64)> = quad_u.as_node_weight_pairs().to_vec();

    // Outer integral over T
    gl_integrate_complex(
        |t| {
            let measure = t.powf(-power) * (-mass_sq * t).exp();
            // Inner integral over u in [0, 1]
            let mut u_sum = Complex64::new(0.0, 0.0);
            for &(node, weight) in &u_pairs {
                // Map [-1, 1] -> [0, 1]: u = 0.5 + 0.5 * node
                let u = 0.5 + 0.5 * node;
                u_sum += f(t, u) * (0.5 * weight);
            }
            u_sum * measure
        },
        config.t_min,
        config.t_max,
        config.n_t,
    )
}

/// Estimate error by comparing N-point and N/2-point quadrature.
pub fn estimate_error<F: Fn(f64) -> Complex64 + Copy>(
    f: F,
    a: f64,
    b: f64,
    degree: usize,
) -> (Complex64, f64) {
    let fine = gl_integrate_complex(f, a, b, degree);
    let coarse = gl_integrate_complex(f, a, b, degree / 2);
    let err = (fine - coarse).norm();
    (fine, err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_gl_integrate_polynomial() {
        // integral_0^1 x^3 dx = 1/4
        let result = gl_integrate(|x| x * x * x, 0.0, 1.0, 32);
        assert!(
            (result - 0.25).abs() < 1e-14,
            "integral x^3 = {result}, expected 0.25"
        );
    }

    #[test]
    fn test_gl_integrate_sine() {
        // integral_0^pi sin(x) dx = 2
        let result = gl_integrate(|x| x.sin(), 0.0, PI, 32);
        assert!(
            (result - 2.0).abs() < 1e-12,
            "integral sin(x) = {result}, expected 2"
        );
    }

    #[test]
    fn test_gl_integrate_complex_exp() {
        // integral_0^1 exp(i*x) dx = (exp(i) - 1) / i
        let result = gl_integrate_complex(|x| Complex64::new(0.0, x).exp(), 0.0, 1.0, 64);
        let expected =
            (Complex64::new(0.0, 1.0).exp() - Complex64::new(1.0, 0.0)) / Complex64::new(0.0, 1.0);
        assert!(
            (result - expected).norm() < 1e-12,
            "integral exp(ix) = {result}, expected {expected}"
        );
    }

    #[test]
    fn test_proper_time_integral_exponential() {
        // integral_0^inf dT * exp(-T) = 1, approximated by [t_min, t_max]
        let config = QuadratureConfig::default();
        let result = proper_time_integral(
            |_t| Complex64::new(1.0, 0.0),
            1.0, // m^2 = 1
            0.0, // power = 0 (no 1/T factor)
            &config,
        );
        // Should be close to 1 - exp(-30) - (1 - exp(-t_min)) ~ 1
        assert!(
            (result.re - 1.0).abs() < 1e-6,
            "integral exp(-T) = {}, expected ~1",
            result.re
        );
        assert!(result.im.abs() < 1e-14);
    }

    #[test]
    fn test_double_integral_product() {
        // integral dT exp(-T) * integral_0^1 du * u = (1/2) * 1 = 0.5
        let config = QuadratureConfig::default();
        let result = double_integral(|_t, u| Complex64::new(u, 0.0), 1.0, 0.0, &config);
        assert!(
            (result.re - 0.5).abs() < 1e-5,
            "double integral = {}, expected ~0.5",
            result.re
        );
    }

    #[test]
    fn test_estimate_error_convergence() {
        let (val, err) = estimate_error(|x| Complex64::new(x.sin(), 0.0), 0.0, PI, 64);
        assert!((val.re - 2.0).abs() < 1e-12);
        assert!(err < 1e-10, "error estimate = {err}");
    }

    #[test]
    fn test_quadrature_configs() {
        let def = QuadratureConfig::default();
        let hi = QuadratureConfig::high_accuracy();
        let fast = QuadratureConfig::fast();
        assert!(hi.n_t > def.n_t);
        assert!(fast.n_t < def.n_t);
    }
}
