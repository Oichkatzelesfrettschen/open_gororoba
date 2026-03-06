//! Complex quartic polynomial solver via Ferrari's method.
//!
//! Solves z^4 + a z^3 + b z^2 + c z + d = 0 over C.
//! Uses the resolvent cubic -> depressed quartic factorization.

use num_complex::Complex64;

/// Solve z^4 + a*z^3 + b*z^2 + c*z + d = 0 over C.
///
/// Returns 4 roots (with multiplicity).
pub fn solve_quartic(a: Complex64, b: Complex64, c: Complex64, d: Complex64) -> [Complex64; 4] {
    // Depressed quartic: substitute z = w - a/4
    let a2 = a * a;
    let p = b - 3.0 * a2 / 8.0;
    let q = c - a * b / 2.0 + a2 * a / 8.0;
    let r = d - a * c / 4.0 + a2 * b / 16.0 - 3.0 * a2 * a2 / 256.0;

    let roots = if q.norm() < 1e-14 {
        // Biquadratic: w^4 + p*w^2 + r = 0
        let disc = (p * p - 4.0 * r).sqrt();
        let u1 = (-p + disc) / 2.0;
        let u2 = (-p - disc) / 2.0;
        [u1.sqrt(), -u1.sqrt(), u2.sqrt(), -u2.sqrt()]
    } else {
        ferrari_depressed(p, q, r)
    };

    let shift = -a / 4.0;
    [
        roots[0] + shift,
        roots[1] + shift,
        roots[2] + shift,
        roots[3] + shift,
    ]
}

/// Ferrari's method for the depressed quartic w^4 + p*w^2 + q*w + r = 0.
fn ferrari_depressed(p: Complex64, q: Complex64, r: Complex64) -> [Complex64; 4] {
    // Resolvent cubic: m^3 + p*m^2 + (p^2/4 - r)*m - q^2/8 = 0
    let rc_a = p;
    let rc_b = p * p / 4.0 - r;
    let rc_c = -q * q / 8.0;

    let cubic_roots = solve_cubic_cardano(rc_a, rc_b, rc_c);

    // Pick the resolvent root with largest norm (best conditioning).
    let m = cubic_roots
        .iter()
        .max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
        .unwrap_or(cubic_roots[0]);

    let sqrt_2m = (2.0 * m).sqrt();

    if sqrt_2m.norm() < 1e-15 {
        // All resolvent roots near zero -- quartic is nearly w^4 + r = 0
        let fourth_root = (-r).sqrt().sqrt();
        let i_unit = Complex64::new(0.0, 1.0);
        return [
            fourth_root,
            i_unit * fourth_root,
            -fourth_root,
            -i_unit * fourth_root,
        ];
    }

    let half_p_plus_m = m + p / 2.0;
    let q_over_2s = q / (2.0 * sqrt_2m);

    let c1 = half_p_plus_m + q_over_2s;
    let c2 = half_p_plus_m - q_over_2s;

    // Solve two quadratics: w^2 + sqrt(2m)*w + c1 = 0 and w^2 - sqrt(2m)*w + c2 = 0
    let disc1 = (sqrt_2m * sqrt_2m - 4.0 * c1).sqrt();
    let disc2 = (sqrt_2m * sqrt_2m - 4.0 * c2).sqrt();

    [
        (-sqrt_2m + disc1) / 2.0,
        (-sqrt_2m - disc1) / 2.0,
        (sqrt_2m + disc2) / 2.0,
        (sqrt_2m - disc2) / 2.0,
    ]
}

/// Solve m^3 + a*m^2 + b*m + c = 0 via Cardano's formula.
fn solve_cubic_cardano(a: Complex64, b: Complex64, c: Complex64) -> [Complex64; 3] {
    // Depress: m = t - a/3
    let p = b - a * a / 3.0;
    let q = c - a * b / 3.0 + 2.0 * a * a * a / 27.0;

    let disc = q * q / 4.0 + p * p * p / 27.0;
    let sqrt_disc = disc.sqrt();

    let u = (-q / 2.0 + sqrt_disc).cbrt();
    let v = (-q / 2.0 - sqrt_disc).cbrt();

    let shift = -a / 3.0;
    let omega = Complex64::new(-0.5, 0.75_f64.sqrt());
    let omega2 = omega * omega;

    [
        u + v + shift,
        omega * u + omega2 * v + shift,
        omega2 * u + omega * v + shift,
    ]
}

/// Kerr radial potential coefficients.
///
/// R(r) = r^4 + (a^2 - xi^2 - eta)*r^2 + 2*M*[eta + (xi - a)^2]*r - a^2*eta
///
/// Returns (A, B, C, D) for r^4 + A*r^3 + B*r^2 + C*r + D = 0.
pub fn kerr_radial_potential_coefficients(
    mass: Complex64,
    a: Complex64,
    xi: Complex64,
    eta: Complex64,
) -> (Complex64, Complex64, Complex64, Complex64) {
    let a_coeff = Complex64::new(0.0, 0.0); // r^3 coefficient is 0 for Kerr
    let b_coeff = a * a - xi * xi - eta;
    let xi_minus_a = xi - a;
    let c_coeff = 2.0 * mass * (eta + xi_minus_a * xi_minus_a);
    let d_coeff = -a * a * eta;
    (a_coeff, b_coeff, c_coeff, d_coeff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn sort_by_real(roots: &mut [Complex64; 4]) {
        roots.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap_or(std::cmp::Ordering::Equal));
    }

    #[test]
    fn quartic_known_roots() {
        // (z - 1)(z - 2)(z - 3)(z - 4) = z^4 - 10z^3 + 35z^2 - 50z + 24
        let a = Complex64::new(-10.0, 0.0);
        let b = Complex64::new(35.0, 0.0);
        let c = Complex64::new(-50.0, 0.0);
        let d = Complex64::new(24.0, 0.0);

        let mut roots = solve_quartic(a, b, c, d);
        sort_by_real(&mut roots);

        for (i, expected) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            assert_relative_eq!(roots[i].re, *expected, epsilon = 1e-6);
            assert!(roots[i].im.abs() < 1e-6);
        }
    }

    #[test]
    fn quartic_biquadratic() {
        // z^4 - 5z^2 + 4 = (z^2 - 4)(z^2 - 1) -> roots: -2, -1, 1, 2
        let a = Complex64::new(0.0, 0.0);
        let b = Complex64::new(-5.0, 0.0);
        let c = Complex64::new(0.0, 0.0);
        let d = Complex64::new(4.0, 0.0);

        let mut roots = solve_quartic(a, b, c, d);
        sort_by_real(&mut roots);

        for (i, expected) in [-2.0, -1.0, 1.0, 2.0].iter().enumerate() {
            assert_relative_eq!(roots[i].re, *expected, epsilon = 1e-6);
            assert!(roots[i].im.abs() < 1e-6);
        }
    }

    #[test]
    fn quartic_unity_roots() {
        // z^4 + 1 = 0 -> roots at e^{i*pi*(2k+1)/4} for k=0,1,2,3
        let a = Complex64::new(0.0, 0.0);
        let b = Complex64::new(0.0, 0.0);
        let c = Complex64::new(0.0, 0.0);
        let d = Complex64::new(1.0, 0.0);

        let roots = solve_quartic(a, b, c, d);

        for r in &roots {
            assert_relative_eq!(r.norm(), 1.0, epsilon = 1e-6);
            let val = r.powi(4) + 1.0;
            assert!(val.norm() < 1e-5, "residual too large: {}", val.norm());
        }
    }

    #[test]
    fn kerr_schwarzschild_limit() {
        // a=0: R(r) = r^4 - (xi^2+eta)*r^2 + 2M*(eta+xi^2)*r
        // = r * [r^3 - (xi^2+eta)*r + 2M*(eta+xi^2)]
        // One root is always r=0.
        let m = Complex64::new(1.0, 0.0);
        let a = Complex64::new(0.0, 0.0);
        let xi = Complex64::new(5.0, 0.0);
        let eta = Complex64::new(10.0, 0.0);

        let (ca, cb, cc, cd) = kerr_radial_potential_coefficients(m, a, xi, eta);
        let roots = solve_quartic(ca, cb, cc, cd);

        let min_norm = roots.iter().map(|r| r.norm()).fold(f64::INFINITY, f64::min);
        assert!(min_norm < 1e-4, "Schwarzschild should have root near 0: {min_norm}");
    }
}
