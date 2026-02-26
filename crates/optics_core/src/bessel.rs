//! Bessel functions for complex argument and integer order.
//!
//! Provides J_n, Y_n, H_n^(1), H_n^(2) and their derivatives, needed for
//! cylindrical Lorentz-Mie scattering. Hand-rolled because no suitable
//! pure-Rust crate exists (scilib is GPL-3.0, complex-bessel-rs needs gfortran).
//!
//! Algorithms:
//! - J_n: ascending power series (DLMF 10.2.2), converges for all z
//! - Y_n: integer-order limiting form (DLMF 10.2.3) with digamma
//! - Derivatives via recurrence: f_n'(z) = f_{n-1}(z) - (n/z)*f_n(z)
//!
//! # References
//! - NIST DLMF Chapter 10: <https://dlmf.nist.gov/10>
//! - Abramowitz & Stegun, Ch. 9

use num_complex::Complex64;
use std::f64::consts::PI;

/// Euler-Mascheroni constant.
const EULER_GAMMA: f64 = 0.5772156649015329;

/// Maximum number of terms in the power series.
const MAX_TERMS: usize = 200;

/// Convergence tolerance for the power series.
const TOL: f64 = 1e-15;

/// Bessel function of the first kind J_n(z) for integer order n >= 0.
///
/// Uses the ascending power series (DLMF 10.2.2):
///   J_n(z) = (z/2)^n * sum_{k=0}^inf (-z^2/4)^k / (k! * (n+k)!)
pub fn bessel_j(n: i32, z: Complex64) -> Complex64 {
    let n_abs = n.unsigned_abs() as usize;

    // Handle negative orders: J_{-n}(z) = (-1)^n * J_n(z)
    if n < 0 {
        let sign = if n_abs.is_multiple_of(2) { 1.0 } else { -1.0 };
        return Complex64::new(sign, 0.0) * bessel_j(n_abs as i32, z);
    }

    let z_half = z / 2.0;
    let neg_z2_4 = -(z * z) / 4.0;

    // Compute (z/2)^n
    let mut prefix = Complex64::new(1.0, 0.0);
    for _ in 0..n_abs {
        prefix *= z_half;
    }

    // Sum the series: sum_{k=0}^inf (-z^2/4)^k / (k! * (n+k)!)
    let mut sum = Complex64::new(1.0, 0.0); // k=0 term is 1/(0! * n!) but we factor n! out
    let mut term = Complex64::new(1.0, 0.0);
    // We accumulate: term_k = (-z^2/4)^k / (k! * (n+k)!) * n!
    // So term_{k+1}/term_k = (-z^2/4) / ((k+1) * (n+k+1))
    for k in 0..MAX_TERMS {
        let next_term = term * neg_z2_4 / ((k + 1) as f64 * (n_abs + k + 1) as f64);
        sum += next_term;
        if next_term.norm() < TOL * sum.norm() && k > 0 {
            break;
        }
        term = next_term;
    }

    // Divide by n!
    let mut n_factorial = 1.0_f64;
    for i in 1..=n_abs {
        n_factorial *= i as f64;
    }

    prefix * sum / n_factorial
}

/// Digamma function psi(n) for positive integer n.
/// psi(1) = -gamma, psi(n+1) = psi(n) + 1/n.
fn digamma_int(n: usize) -> f64 {
    let mut result = -EULER_GAMMA;
    for k in 1..n {
        result += 1.0 / k as f64;
    }
    result
}

/// Bessel function of the second kind Y_n(z) for integer order n >= 0.
///
/// Uses the integer-order limiting form (DLMF 10.2.3):
///   Y_n(z) = (2/pi)*[ln(z/2) + gamma]*J_n(z)
///          - (1/pi)*(z/2)^{-n} * sum_{k=0}^{n-1} (n-k-1)!/k! * (z^2/4)^k
///          - (1/pi)*(z/2)^n * sum_{k=0}^inf (-1)^k [psi(k+1)+psi(n+k+1)] * (z^2/4)^k / (k!*(n+k)!)
pub fn bessel_y(n: i32, z: Complex64) -> Complex64 {
    let n_abs = n.unsigned_abs() as usize;

    // Handle negative orders: Y_{-n}(z) = (-1)^n * Y_n(z)
    if n < 0 {
        let sign = if n_abs.is_multiple_of(2) { 1.0 } else { -1.0 };
        return Complex64::new(sign, 0.0) * bessel_y(n_abs as i32, z);
    }

    let z_half = z / 2.0;
    let z2_4 = z * z / 4.0;
    let inv_pi = 1.0 / PI;

    let jn = bessel_j(n_abs as i32, z);
    let ln_z_half = z_half.ln();

    // Term 1: (2/pi) * ln(z/2) * J_n(z)   [DLMF 10.2.4]
    // Note: the Euler-Mascheroni gamma is NOT in term1; it comes from
    // psi(1) = -gamma in the digamma series (term3).
    let term1 = (2.0 * inv_pi) * ln_z_half * jn;

    // Term 2: -(1/pi) * (z/2)^{-n} * sum_{k=0}^{n-1} (n-k-1)!/k! * (z^2/4)^k
    let mut term2 = Complex64::new(0.0, 0.0);
    if n_abs > 0 {
        let mut z_half_neg_n = Complex64::new(1.0, 0.0);
        for _ in 0..n_abs {
            z_half_neg_n /= z_half;
        }

        let mut sum2 = Complex64::new(0.0, 0.0);
        let mut z2_4_pow_k = Complex64::new(1.0, 0.0); // (z^2/4)^0 = 1
        let mut k_factorial = 1.0_f64;
        for k in 0..n_abs {
            // (n-k-1)!
            let mut nmk1_fac = 1.0_f64;
            for i in 1..=(n_abs - k - 1) {
                nmk1_fac *= i as f64;
            }
            sum2 += z2_4_pow_k * (nmk1_fac / k_factorial);
            z2_4_pow_k *= z2_4;
            k_factorial *= (k + 1) as f64;
        }
        term2 = -inv_pi * z_half_neg_n * sum2;
    }

    // Term 3: -(1/pi) * (z/2)^n * sum_{k=0}^inf (-1)^k [psi(k+1)+psi(n+k+1)] * (z^2/4)^k / (k!*(n+k)!)
    let mut z_half_n = Complex64::new(1.0, 0.0);
    for _ in 0..n_abs {
        z_half_n *= z_half;
    }

    let mut sum3 = Complex64::new(0.0, 0.0);
    let mut neg1_k_z2_4_k = Complex64::new(1.0, 0.0); // (-1)^0 * (z^2/4)^0
    let mut k_factorial = 1.0_f64;
    let mut nk_factorial = 1.0_f64;
    for i in 1..=n_abs {
        nk_factorial *= i as f64;
    }

    for k in 0..MAX_TERMS {
        if k > 0 {
            neg1_k_z2_4_k *= -z2_4;
            k_factorial *= k as f64;
            nk_factorial *= (n_abs + k) as f64;
        }
        let psi_sum = digamma_int(k + 1) + digamma_int(n_abs + k + 1);
        let this_term = neg1_k_z2_4_k * psi_sum / (k_factorial * nk_factorial);
        sum3 += this_term;
        if k > 2 && this_term.norm() < TOL * sum3.norm() {
            break;
        }
    }
    let term3 = -inv_pi * z_half_n * sum3;

    term1 + term2 + term3
}

/// Hankel function of the first kind: H_n^(1)(z) = J_n(z) + i*Y_n(z).
pub fn hankel_1(n: i32, z: Complex64) -> Complex64 {
    bessel_j(n, z) + Complex64::i() * bessel_y(n, z)
}

/// Hankel function of the second kind: H_n^(2)(z) = J_n(z) - i*Y_n(z).
pub fn hankel_2(n: i32, z: Complex64) -> Complex64 {
    bessel_j(n, z) - Complex64::i() * bessel_y(n, z)
}

/// Derivative of a Bessel-family function via recurrence:
///   f_n'(z) = f_{n-1}(z) - (n/z) * f_n(z)
///
/// The `f` argument must evaluate the same family (J, Y, H1, or H2).
pub fn bessel_derivative<F>(f: F, n: i32, z: Complex64) -> Complex64
where
    F: Fn(i32, Complex64) -> Complex64,
{
    f(n - 1, z) - Complex64::new(n as f64, 0.0) / z * f(n, z)
}

/// Derivative of J_n(z).
pub fn bessel_j_prime(n: i32, z: Complex64) -> Complex64 {
    bessel_derivative(bessel_j, n, z)
}

/// Derivative of Y_n(z).
pub fn bessel_y_prime(n: i32, z: Complex64) -> Complex64 {
    bessel_derivative(bessel_y, n, z)
}

/// Derivative of H_n^(1)(z).
pub fn hankel_1_prime(n: i32, z: Complex64) -> Complex64 {
    bessel_derivative(hankel_1, n, z)
}

/// Derivative of H_n^(2)(z).
pub fn hankel_2_prime(n: i32, z: Complex64) -> Complex64 {
    bessel_derivative(hankel_2, n, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_complex_approx(got: Complex64, expected: Complex64, tol: f64, msg: &str) {
        let diff = (got - expected).norm();
        assert!(
            diff < tol,
            "{}: got ({:.10}, {:.10}), expected ({:.10}, {:.10}), diff={:.2e}",
            msg,
            got.re,
            got.im,
            expected.re,
            expected.im,
            diff
        );
    }

    #[test]
    fn bessel_j_zero_arg() {
        let z = Complex64::new(0.0, 0.0);
        // J_0(0) = 1
        // We need special handling: the series gives 1.0 for n=0
        // For n=0, prefix=(z/2)^0=1, sum starts at 1, n!=1 => J_0(0)=1
        // But z=0 means all k>0 terms are 0
        assert_complex_approx(bessel_j(0, z), Complex64::new(1.0, 0.0), 1e-14, "J_0(0)=1");
        // J_n(0) = 0 for n > 0
        for n in 1..=5 {
            assert_complex_approx(
                bessel_j(n, z),
                Complex64::new(0.0, 0.0),
                1e-14,
                &format!("J_{}(0)=0", n),
            );
        }
    }

    #[test]
    fn bessel_j_real_values() {
        // DLMF reference values
        let z = Complex64::new(1.0, 0.0);
        assert_complex_approx(
            bessel_j(0, z),
            Complex64::new(0.7651976865579666, 0.0),
            1e-12,
            "J_0(1)",
        );
        assert_complex_approx(
            bessel_j(1, z),
            Complex64::new(0.4400505857449335, 0.0),
            1e-12,
            "J_1(1)",
        );

        let z5 = Complex64::new(5.0, 0.0);
        assert_complex_approx(
            bessel_j(0, z5),
            Complex64::new(-0.1775967713143383, 0.0),
            1e-12,
            "J_0(5)",
        );
    }

    #[test]
    fn bessel_j_complex_arg() {
        // J_0(1+i): reference from scipy.special.jv(0, 1+1j)
        let z = Complex64::new(1.0, 1.0);
        let expected = Complex64::new(0.9376084768060293, -0.4965299476091221);
        assert_complex_approx(bessel_j(0, z), expected, 1e-10, "J_0(1+i)");
    }

    #[test]
    fn bessel_y_real_values() {
        let z = Complex64::new(1.0, 0.0);
        assert_complex_approx(
            bessel_y(0, z),
            Complex64::new(0.08825696421567696, 0.0),
            1e-10,
            "Y_0(1)",
        );
        assert_complex_approx(
            bessel_y(1, z),
            Complex64::new(-0.7812128213002887, 0.0),
            1e-10,
            "Y_1(1)",
        );
    }

    #[test]
    fn hankel_sum_identity() {
        // H^(1)_n + H^(2)_n = 2*J_n
        for n in 0..=4 {
            let z = Complex64::new(3.0, 0.5);
            let h1 = hankel_1(n, z);
            let h2 = hankel_2(n, z);
            let j = bessel_j(n, z);
            assert_complex_approx(h1 + h2, 2.0 * j, 1e-10, &format!("H1+H2=2J for n={}", n));
        }
    }

    #[test]
    fn wronskian_identity() {
        // J_n(z)*Y_n'(z) - J_n'(z)*Y_n(z) = 2/(pi*z)
        for n in 0..=3 {
            let z = Complex64::new(2.5, 0.3);
            let jn = bessel_j(n, z);
            let yn = bessel_y(n, z);
            let jnp = bessel_j_prime(n, z);
            let ynp = bessel_y_prime(n, z);
            let wronskian = jn * ynp - jnp * yn;
            let expected = Complex64::new(2.0, 0.0) / (PI * z);
            assert_complex_approx(wronskian, expected, 1e-8, &format!("Wronskian for n={}", n));
        }
    }

    #[test]
    fn recurrence_relation() {
        // z*J_n'(z) = n*J_n(z) - z*J_{n+1}(z)
        for n in 1..=4 {
            let z = Complex64::new(3.0, 1.0);
            let lhs = z * bessel_j_prime(n, z);
            let rhs = Complex64::new(n as f64, 0.0) * bessel_j(n, z) - z * bessel_j(n + 1, z);
            assert_complex_approx(lhs, rhs, 1e-10, &format!("recurrence for n={}", n));
        }
    }

    #[test]
    fn negative_order_symmetry() {
        // J_{-n}(z) = (-1)^n * J_n(z)
        for n in 1..=5 {
            let z = Complex64::new(2.0, 0.7);
            let j_neg = bessel_j(-n, z);
            let sign = if (n as u32).is_multiple_of(2) { 1.0 } else { -1.0 };
            let expected = Complex64::new(sign, 0.0) * bessel_j(n, z);
            assert_complex_approx(
                j_neg,
                expected,
                1e-12,
                &format!("J_{{-{}}} = (-1)^{} J_{}", n, n, n),
            );
        }
    }
}
