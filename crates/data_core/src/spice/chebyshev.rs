use wide::f64x4;

/// Evaluates Chebyshev polynomials for X, Y, Z components simultaneously using `wide::f64x4`.
///
/// `x`: The normalized time parameter in [-1, 1].
/// `coeffs`: A slice of `f64x4`, where each `f64x4` contains the i-th coefficient for [X, Y, Z, 0.0].
///
/// Returns `[X, Y, Z, _]` evaluated at `x`.
pub fn evaluate_chebyshev_position(x: f64, coeffs: &[f64x4]) -> f64x4 {
    let mut d1 = f64x4::splat(0.0);
    let mut d2 = f64x4::splat(0.0);

    let x2 = f64x4::splat(x * 2.0);
    let x_simd = f64x4::splat(x);

    // Clenshaw's recurrence
    for c in coeffs.iter().rev().take(coeffs.len() - 1) {
        let temp = d1;
        d1 = x2 * d1 - d2 + *c;
        d2 = temp;
    }

    x_simd * d1 - d2 + coeffs[0]
}

/// Evaluates the derivative of Chebyshev polynomials for X, Y, Z components simultaneously.
/// This yields the velocity vectors.
///
/// `x`: The normalized time parameter in [-1, 1].
/// `coeffs`: A slice of `f64x4` containing the position coefficients.
///
/// Returns `[dX/dx, dY/dx, dZ/dx, _]` evaluated at `x`.
/// Note: To get physical velocity, this must be divided by the time radius of the segment.
pub fn evaluate_chebyshev_velocity(x: f64, coeffs: &[f64x4]) -> f64x4 {
    // Derivative coefficients recurrence for Clenshaw
    // We compute the sum C_i * T'_i(x)
    // T'_0(x) = 0
    // T'_1(x) = 1
    // T'_2(x) = 4x
    // In general, to sum a_i T'_i(x), we can differentiate the recurrence relation.
    // Or we can just compute the derivative coefficients first.
    // D_i = 2 * i * C_i + D_{i+2}

    let n = coeffs.len();
    if n < 2 {
        return f64x4::splat(0.0);
    }

    let mut deriv_coeffs = vec![f64x4::splat(0.0); n - 1];

    // Compute derivative coefficients
    // c'_i = c'_{i+2} + 2(i+1) c_{i+1}
    let mut c_plus_2 = f64x4::splat(0.0);
    let mut c_plus_1 = f64x4::splat(0.0);

    for i in (0..n - 1).rev() {
        let c_i = c_plus_2 + coeffs[i + 1] * f64x4::splat(2.0 * (i + 1) as f64);
        deriv_coeffs[i] = c_i;
        c_plus_2 = c_plus_1;
        c_plus_1 = c_i;
    }

    // Note: deriv_coeffs are now coefficients for standard Chebyshev polynomials, but
    // there's a factor of 1/2 for the 0th term in standard convention.
    deriv_coeffs[0] *= f64x4::splat(0.5);

    evaluate_chebyshev_position(x, &deriv_coeffs)
}
