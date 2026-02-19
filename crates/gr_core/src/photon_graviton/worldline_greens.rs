//! Worldline Green's functions on the circle.
//!
//! These are the fundamental building blocks for one-loop amplitudes in the
//! worldline formalism. The worldline is parametrized by tau in [0, T] with
//! periodic boundary conditions. The Green's function G_B(tau_1, tau_2) is
//! the propagator on this circle, modified by the constant background field.
//!
//! For a pure magnetic field B along z, the 4x4 matrix G_B decomposes into
//! scalar coefficient functions times projectors g+, g-, r+ (Eq B.15 of
//! Part 3). We implement only the scalar coefficients; the projector algebra
//! is handled in the diagram modules.
//!
//! All formulas from Appendix B of Bastianelli et al. (2026), arXiv:2601.23279.
//! Variable names follow the paper: z = eBT, u = (tau_1 - tau_2)/T, v = 1-2u.

/// Bosonic coefficient SB12: sinh(z*v) / sinh(z).
///
/// This is the antisymmetric (in the field plane) part of the bosonic
/// worldline Green's function. At z=0 it reduces to v = 1-2u.
///
/// # Arguments
/// * `z` -- dimensionless field parameter eB*T
/// * `u` -- worldline modulus in [0, 1], with v = 1-2u
pub fn sb12(z: f64, u: f64) -> f64 {
    let v = 1.0 - 2.0 * u;
    if z.abs() < 1e-10 {
        // Free field limit: SB12 -> v
        v
    } else if z.abs() < 0.1 {
        // Taylor expansion to avoid cancellation:
        // sinh(z*v)/sinh(z) = v + v*(v^2-1)*z^2/6 + ...
        let z2 = z * z;
        let v2 = v * v;
        v * (1.0 + (v2 - 1.0) * z2 / 6.0 + (v2 - 1.0) * (3.0 * v2 - 7.0) * z2 * z2 / 360.0)
    } else {
        (z * v).sinh() / z.sinh()
    }
}

/// Bosonic coefficient AB12: cosh(z*v) / sinh(z) - 1/z.
///
/// This is the symmetric (in the field plane) part. At z=0, AB12 = 0
/// (the field correction vanishes). The subtraction of 1/z removes the
/// free-field singularity.
///
/// # Arguments
/// * `z` -- dimensionless field parameter eB*T (must be > 0 for the 1/z term)
/// * `u` -- worldline modulus in [0, 1]
pub fn ab12(z: f64, u: f64) -> f64 {
    let v = 1.0 - 2.0 * u;
    if z.abs() < 1e-10 {
        // Free field: AB12 -> 0
        0.0
    } else if z.abs() < 0.1 {
        // Taylor: cosh(zv)/sinh(z) - 1/z
        // = (1 + z^2 v^2/2 + ...) / (z + z^3/6 + ...) - 1/z
        // = (1/z)(1 + z^2 v^2/2)(1 - z^2/6 + ...) - 1/z
        // = z*(v^2 - 1/3)/2 + O(z^3)
        // More precisely:
        // = z/2 * (v^2 - 1/3) + z^3/24 * (v^4 - 10*v^2/3 + 7/15) + ...
        let z2 = z * z;
        let v2 = v * v;
        z * (v2 - 1.0 / 3.0) / 2.0
            + z * z2 * (v2 * v2 - 10.0 * v2 / 3.0 + 7.0 / 15.0) / 24.0
    } else {
        (z * v).cosh() / z.sinh() - 1.0 / z
    }
}

/// Coincidence limit of AB12: coth(z) - 1/z = AB12(z, 0).
///
/// This appears in the tadpole diagram and Green's function at coincident points.
pub fn ab_coincidence(z: f64) -> f64 {
    super::special_functions::coth_minus_inv(z)
}

/// Fermionic coefficient SF12: cosh(z*v) / cosh(z).
///
/// Analogous to SB12 but for the Dirac spinor Green's function GF
/// (Eq B.48 of Part 3). At z=0, SF12 = 1.
///
/// # Arguments
/// * `z` -- dimensionless field parameter eB*T
/// * `u` -- worldline modulus in [0, 1]
pub fn sf12(z: f64, u: f64) -> f64 {
    let v = 1.0 - 2.0 * u;
    if z.abs() < 1e-10 {
        1.0
    } else if z.abs() < 0.1 {
        let z2 = z * z;
        let v2 = v * v;
        // cosh(zv)/cosh(z) = 1 + z^2*(v^2-1)/2 + O(z^4)
        1.0 + z2 * (v2 - 1.0) / 2.0
            + z2 * z2 * (v2 * v2 - 6.0 * v2 + 5.0) / 24.0
    } else {
        (z * v).cosh() / z.cosh()
    }
}

/// Fermionic coefficient AF12: sinh(z*v) / cosh(z).
///
/// Eq B.49 of Part 3. At z=0, AF12 = v = 1-2u.
pub fn af12(z: f64, u: f64) -> f64 {
    let v = 1.0 - 2.0 * u;
    if z.abs() < 1e-10 {
        v
    } else if z.abs() < 0.1 {
        let z2 = z * z;
        let v2 = v * v;
        v * (1.0 + z2 * (v2 / 6.0 - 0.5) + z2 * z2 * (v2 * v2 / 120.0 - v2 / 12.0 + 5.0 / 24.0))
    } else {
        (z * v).sinh() / z.cosh()
    }
}

// ---------------------------------------------------------------------------
// Derivatives of the Green's function (dot = d/d(tau_1))
// ---------------------------------------------------------------------------

/// d/du SB12 = 2*z*cosh(z*v) / sinh(z) (chain rule: dv/du = -2, d/dv sinh(zv) = z cosh(zv)).
///
/// Normalized: (1/T) * dG_B/d(tau_1) in the field plane involves SB12_dot / T.
/// Here we return the u-derivative of SB12 (without the T factor).
pub fn sb12_dot(z: f64, u: f64) -> f64 {
    let v = 1.0 - 2.0 * u;
    if z.abs() < 1e-10 {
        -2.0
    } else if z.abs() < 0.1 {
        let z2 = z * z;
        let v2 = v * v;
        -2.0 * (1.0 + (3.0 * v2 - 1.0) * z2 / 6.0)
    } else {
        -2.0 * z * (z * v).cosh() / z.sinh()
    }
}

/// d/du AB12 = -2*z*sinh(z*v) / sinh(z) = -2*z*SB12.
pub fn ab12_dot(z: f64, u: f64) -> f64 {
    let v = 1.0 - 2.0 * u;
    if z.abs() < 1e-10 {
        -2.0 * v
    } else {
        -2.0 * z * (z * v).sinh() / z.sinh()
    }
}

// ---------------------------------------------------------------------------
// Exponential factor in the amplitude integrand
// ---------------------------------------------------------------------------

/// The bilinear form k . Phi_B . k appearing in the exponential of the
/// vacuum polarization integrand.
///
/// For a pure magnetic field along z with photon 4-momentum
/// k = omega * (sin(theta), 0, cos(theta), i) [Euclidean]:
///
/// k . Phi_B . k = omega^2 * sin^2(theta) * (u(1-u) - SB12^2/(4z)) * T
///
/// where Phi_B is constructed from the worldline Green's function.
///
/// # Arguments
/// * `z` -- eB*T
/// * `u` -- worldline modulus
/// * `omega` -- photon energy (natural units)
/// * `theta` -- angle between k and B
/// * `proper_time` -- T
pub fn exp_factor(z: f64, u: f64, omega: f64, theta: f64, proper_time: f64) -> f64 {
    let sin2 = theta.sin().powi(2);
    let sb = sb12(z, u);
    let free_part = u * (1.0 - u);
    // In the field plane: the correction replaces u(1-u) with the field-dependent version
    let field_part = if z.abs() < 1e-10 {
        free_part
    } else {
        // From Eq B.38: the exponent involves (GB12_free - GB12_field) projected
        // For pure magnetic: this becomes u(1-u) - [SB12 correction terms]
        // The key combination from the paper (Eq 3.18):
        // k.Phi.k = T * omega^2 * sin^2(theta) * [u(1-u) + (SB12^2 - v^2)/(4z)]
        let v = 1.0 - 2.0 * u;
        free_part + (sb * sb - v * v) / (4.0 * z)
    };
    omega * omega * sin2 * field_part * proper_time
}

// ---------------------------------------------------------------------------
// Weak-field expansions for cross-validation
// ---------------------------------------------------------------------------

/// Weak-field expansion of SB12 to O(z^4).
///
/// SB12 = v + v(v^2-1)*z^2/6 + v(v^2-1)(3v^2-7)*z^4/360 + O(z^6)
pub fn sb12_weak(z: f64, u: f64) -> f64 {
    let v = 1.0 - 2.0 * u;
    let z2 = z * z;
    let v2 = v * v;
    v + v * (v2 - 1.0) * z2 / 6.0 + v * (v2 - 1.0) * (3.0 * v2 - 7.0) * z2 * z2 / 360.0
}

/// Weak-field expansion of AB12 to O(z^3).
///
/// AB12 = z*(v^2 - 1/3)/2 + O(z^3)
pub fn ab12_weak(z: f64, u: f64) -> f64 {
    let v = 1.0 - 2.0 * u;
    let v2 = v * v;
    z * (v2 - 1.0 / 3.0) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Free-field limits --

    #[test]
    fn test_sb12_free_field() {
        // At z=0, SB12(u) = 1 - 2u
        for &u in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let result = sb12(0.0, u);
            let expected = 1.0 - 2.0 * u;
            assert!(
                (result - expected).abs() < 1e-14,
                "SB12(0, {u}) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_ab12_free_field() {
        // At z=0, AB12 = 0
        for &u in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let result = ab12(0.0, u);
            assert!(
                result.abs() < 1e-14,
                "AB12(0, {u}) = {result}, expected 0"
            );
        }
    }

    #[test]
    fn test_sf12_free_field() {
        // At z=0, SF12 = 1
        for &u in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let result = sf12(0.0, u);
            assert!(
                (result - 1.0).abs() < 1e-14,
                "SF12(0, {u}) = {result}, expected 1"
            );
        }
    }

    #[test]
    fn test_af12_free_field() {
        // At z=0, AF12 = v = 1-2u
        for &u in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let result = af12(0.0, u);
            let expected = 1.0 - 2.0 * u;
            assert!(
                (result - expected).abs() < 1e-14,
                "AF12(0, {u}) = {result}, expected {expected}"
            );
        }
    }

    // -- Symmetry tests --

    #[test]
    fn test_sb12_symmetry() {
        // SB12(z, u) should be antisymmetric under u -> 1-u (since v -> -v and sinh is odd)
        // Actually SB12(z, 1-u) = -SB12(z, u) because v -> -v
        let z = 1.5;
        for &u in &[0.1, 0.25, 0.4] {
            let forward = sb12(z, u);
            let backward = sb12(z, 1.0 - u);
            assert!(
                (forward + backward).abs() < 1e-12,
                "SB12 antisymmetry: SB12({z},{u}) = {forward}, SB12({z},{}) = {backward}",
                1.0 - u
            );
        }
    }

    #[test]
    fn test_sb12_midpoint() {
        // At u=0.5, v=0, so SB12 = sinh(0)/sinh(z) = 0
        let result = sb12(2.0, 0.5);
        assert!(
            result.abs() < 1e-14,
            "SB12(2, 0.5) = {result}, expected 0"
        );
    }

    // -- Weak-field cross-validation --

    #[test]
    fn test_sb12_weak_field_agreement() {
        // For z = 0.01, exact and Taylor should agree to 12+ digits
        let z = 0.01;
        for &u in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let exact = sb12(z, u);
            let weak = sb12_weak(z, u);
            let diff = (exact - weak).abs();
            assert!(
                diff < 1e-12,
                "SB12 weak-field: exact={exact}, weak={weak}, diff={diff} at u={u}"
            );
        }
    }

    #[test]
    fn test_ab12_weak_field_agreement() {
        // ab12_weak only has O(z) + O(z^3) terms. At z=0.01 the missing O(z^3) term
        // gives a relative error of O(z^2) ~ 1e-4.
        let z = 0.01;
        for &u in &[0.1, 0.3, 0.7, 0.9] {
            let exact = ab12(z, u);
            let weak = ab12_weak(z, u);
            let rel_diff = if exact.abs() > 1e-20 {
                (exact - weak).abs() / exact.abs()
            } else {
                (exact - weak).abs()
            };
            assert!(
                rel_diff < 1e-3,
                "AB12 weak-field: exact={exact}, weak={weak}, rel_diff={rel_diff} at u={u}"
            );
        }
    }

    // -- Numerical derivative vs analytic --

    #[test]
    fn test_sb12_dot_numerical() {
        let z = 1.5;
        let u = 0.3;
        let h = 1e-6;
        let numerical = (sb12(z, u + h) - sb12(z, u - h)) / (2.0 * h);
        let analytic = sb12_dot(z, u);
        assert!(
            (numerical - analytic).abs() / analytic.abs() < 1e-6,
            "SB12_dot: numerical={numerical}, analytic={analytic}"
        );
    }

    #[test]
    fn test_ab12_dot_numerical() {
        let z = 1.5;
        let u = 0.3;
        let h = 1e-6;
        let numerical = (ab12(z, u + h) - ab12(z, u - h)) / (2.0 * h);
        let analytic = ab12_dot(z, u);
        assert!(
            (numerical - analytic).abs() / analytic.abs() < 1e-5,
            "AB12_dot: numerical={numerical}, analytic={analytic}"
        );
    }

    // -- Determinant consistency --

    #[test]
    fn test_sb12_at_boundary() {
        // SB12(z, 0) = sinh(z)/sinh(z) = 1
        let result = sb12(2.0, 0.0);
        assert!(
            (result - 1.0).abs() < 1e-14,
            "SB12(z, 0) = {result}, expected 1"
        );
    }

    #[test]
    fn test_sf12_even_symmetry() {
        // SF12(z, u) = cosh(zv)/cosh(z) -- even in v, so SF12(z, u) = SF12(z, 1-u)
        let z = 1.5;
        for &u in &[0.1, 0.25, 0.4] {
            let forward = sf12(z, u);
            let backward = sf12(z, 1.0 - u);
            assert!(
                (forward - backward).abs() < 1e-12,
                "SF12 symmetry: SF12({z},{u}) = {forward}, SF12({z},{}) = {backward}",
                1.0 - u
            );
        }
    }

    // -- Taylor crossover smoothness --

    #[test]
    fn test_sb12_taylor_crossover() {
        let u = 0.3;
        let z_below = 0.099;
        let z_above = 0.101;
        let below = sb12(z_below, u);
        let above = sb12(z_above, u);
        // Linear interpolation at z=0.1
        let interp = 0.5 * (below + above);
        let mid = sb12(0.1, u);
        assert!(
            (mid - interp).abs() < 1e-6,
            "SB12 crossover: below={below}, mid={mid}, above={above}, interp={interp}"
        );
    }

    #[test]
    fn test_exp_factor_zero_field() {
        // At z=0 (no field), exp_factor = omega^2 * sin^2(theta) * u(1-u) * T
        let val = exp_factor(0.0, 0.3, 1.0, 0.5, 2.0);
        let expected = 1.0 * 0.5_f64.sin().powi(2) * 0.3 * 0.7 * 2.0;
        assert!(
            (val - expected).abs() < 1e-12,
            "exp_factor(0) = {val}, expected {expected}"
        );
    }
}
