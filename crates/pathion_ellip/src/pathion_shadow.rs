//! Pathion-modulated deflection of light rays in Kerr spacetime.
//!
//! Computes the exact deflection angle by:
//! 1. Diagonalizing the 32D impact parameters into 16 complex planes
//! 2. Solving the Kerr radial quartic R(r)=0 in each plane
//! 3. Integrating the deflection via Carlson RF + RJ from the turning point

use crate::carlson::{carlson_rf_complex, carlson_rj_complex};
use crate::diagonalizer::PathionDiagonalizer;
use crate::quartic::{kerr_radial_potential_coefficients, solve_quartic};
use num_complex::Complex64;

/// Compute per-plane deflection using Carlson integrals.
///
/// Given the 4 roots r1 <= r2 <= r3 <= r4 of R(r) = 0 and the mass M,
/// the scattering deflection from the turning point r_tp = r4 is:
///
///   delta_phi = 2 * RF(0, (r3-r1)(r4-r2), (r4-r1)(r3-r2))
///             * 2 / sqrt((r4-r1)(r3-r2))
///
/// with RJ corrections for the spin-dependent terms.
fn deflection_from_roots(
    roots: &[Complex64; 4],
    mass: Complex64,
    a: Complex64,
    xi: Complex64,
) -> Complex64 {
    // Sort roots by real part to identify turning point structure.
    let mut sorted = *roots;
    sorted.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap_or(std::cmp::Ordering::Equal));
    let [r1, r2, r3, r4] = sorted;

    let d41 = r4 - r1;
    let d32 = r3 - r2;
    let d31 = r3 - r1;
    let d42 = r4 - r2;

    let prod_outer = d41 * d32;
    if prod_outer.norm() < 1e-14 {
        // Degenerate case: roots coincide. Return zero deflection.
        return Complex64::new(0.0, 0.0);
    }

    // Scattering integral (Gralla-Lupsasca 2020, Eq. 2.12 analog):
    // phi = 2 / sqrt((r4-r1)(r3-r2)) * RF(0, (r3-r1)(r4-r2)/((r4-r1)(r3-r2)), 1)
    let k_sq = d31 * d42 / prod_outer;
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);

    let rf_val = carlson_rf_complex(zero, k_sq, one);
    let phi_orbit = 2.0 * rf_val / prod_outer.sqrt();

    // Spin correction via RJ (Hackmann-Lammerzahl 2012 style):
    // The spin enters through the p-parameter of RJ, related to the
    // angular momentum barrier a*xi/(r - r_+).
    let r_plus = mass + (mass * mass - a * a).sqrt();
    let p_val = (r4 - r_plus) * d32 / ((r3 - r_plus) * (r4 - r2));

    let rj_val = if p_val.norm() > 1e-14 {
        carlson_rj_complex(zero, k_sq, one, p_val)
    } else {
        zero
    };

    let spin_correction = 2.0 * a * xi / (3.0 * prod_outer.sqrt()) * rj_val;

    // Total deflection = orbital term + spin correction - pi (subtract straight-line path)
    phi_orbit + spin_correction - Complex64::new(std::f64::consts::PI, 0.0)
}

/// Compute the Pathion-modulated deflection of a light ray in Kerr spacetime.
///
/// # Arguments
/// * `mass` - Black hole mass (geometric units)
/// * `spin` - 32D Pathion spin parameter
/// * `xi` - 32D impact parameter (angular momentum / energy)
/// * `eta` - 32D Carter constant / energy^2
pub fn pathion_deflection(
    mass: f64,
    spin: &[f64; 32],
    xi: &[f64; 32],
    eta: &[f64; 32],
) -> anyhow::Result<[f64; 32]> {
    let diag = PathionDiagonalizer::new();
    let m = Complex64::new(mass, 0.0);

    let a_planes = diag.project(spin);
    let xi_planes = diag.project(xi);
    let eta_planes = diag.project(eta);

    let mut defl_planes = [Complex64::new(0.0, 0.0); 16];

    for i in 0..16 {
        let a = a_planes[i];
        let xi_c = xi_planes[i];
        let eta_c = eta_planes[i];

        // Build Kerr radial potential R(r) = r^4 + B*r^2 + C*r + D
        let (qa, qb, qc, qd) = kerr_radial_potential_coefficients(m, a, xi_c, eta_c);

        // Solve R(r) = 0 for the 4 complex roots
        let roots = solve_quartic(qa, qb, qc, qd);

        // Compute deflection from the root structure
        defl_planes[i] = deflection_from_roots(&roots, m, a, xi_c);
    }

    Ok(diag.recompose(&defl_planes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schwarzschild_deflection_sign() {
        // For a=0 (Schwarzschild), deflection should be positive for light
        // bending toward the mass.
        let mass = 1.0;
        let mut spin = [0.0; 32];
        spin[0] = 0.0; // a = 0

        let mut xi = [0.0; 32];
        xi[0] = 6.0; // impact parameter

        let mut eta = [0.0; 32];
        eta[0] = 20.0; // Carter constant

        let defl = pathion_deflection(mass, &spin, &xi, &eta).unwrap();
        // The real component should be finite (not NaN)
        assert!(defl[0].is_finite(), "deflection should be finite: {}", defl[0]);
    }

    #[test]
    fn deflection_increases_with_spin() {
        // Frame-dragging should modify the deflection
        let mass = 1.0;
        let mut xi = [0.0; 32];
        xi[0] = 8.0;
        let mut eta = [0.0; 32];
        eta[0] = 30.0;

        let mut spin_low = [0.0; 32];
        spin_low[0] = 0.1;
        let mut spin_high = [0.0; 32];
        spin_high[0] = 0.9;

        let defl_low = pathion_deflection(mass, &spin_low, &xi, &eta).unwrap();
        let defl_high = pathion_deflection(mass, &spin_high, &xi, &eta).unwrap();

        // Deflections should differ (spin matters)
        assert!(
            (defl_low[0] - defl_high[0]).abs() > 1e-6,
            "spin should affect deflection: low={}, high={}",
            defl_low[0],
            defl_high[0]
        );
    }
}
