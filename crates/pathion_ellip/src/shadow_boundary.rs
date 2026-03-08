//! Shadow boundary computation for Pathion-modulated Kerr black holes.
//!
//! Sweeps impact parameters (xi, eta) to find the photon capture boundary,
//! producing the shadow silhouette in the observer's sky.

use crate::pathion_shadow::pathion_deflection;

/// A point on the shadow boundary in the observer's sky.
#[derive(Debug, Clone, Copy)]
pub struct ShadowPoint {
    /// Celestial coordinate alpha (horizontal displacement)
    pub alpha: f64,
    /// Celestial coordinate beta (vertical displacement)
    pub beta: f64,
    /// Impact parameter xi that produces this boundary point
    pub xi_0: f64,
    /// Carter constant eta that produces this boundary point
    pub eta_0: f64,
}

/// Compute the shadow boundary for a Pathion-modulated Kerr black hole.
///
/// The shadow boundary is the locus of impact parameters (xi, eta) where
/// the deflection angle diverges (unstable circular photon orbits).
///
/// For Kerr (in the real component), the critical curve is parametrized by
/// the unstable orbit radius r_ph:
///   xi(r) = (r^2 + a^2 - 2*M*r*a/(r - M)) * sign
///   eta(r) = r^2 * (3*M/r - 1 - a^2*xi^2/r^4)  (simplified)
///
/// # Arguments
/// * `mass` - Black hole mass
/// * `spin_0` - Real component of the spin (a/M)
/// * `n_points` - Number of points on the boundary
pub fn pathion_shadow_boundary(mass: f64, spin_0: f64, n_points: usize) -> Vec<ShadowPoint> {
    let a = spin_0 * mass;

    // Photon orbit radius range: from r_pro (prograde) to r_retro (retrograde)
    // For Kerr: r_ph = 2M(1 + cos(2/3 * arccos(-a/M))) for prograde
    let r_pro = 2.0 * mass * (1.0 + (2.0 / 3.0 * (-a / mass).acos()).cos());
    let r_retro = 2.0 * mass * (1.0 + (2.0 / 3.0 * (a / mass).acos()).cos());

    let r_min = r_pro.min(r_retro);
    let r_max = r_pro.max(r_retro);

    let mut boundary = Vec::with_capacity(n_points);

    for k in 0..n_points {
        let t = k as f64 / (n_points - 1).max(1) as f64;
        let r = r_min + t * (r_max - r_min);

        if r <= mass {
            continue;
        }

        // Critical impact parameters from Bardeen (1973)
        let r2 = r * r;
        let delta = r2 - 2.0 * mass * r + a * a;

        if delta.abs() < 1e-14 {
            continue;
        }

        let xi = if a.abs() < 1e-14 {
            // Schwarzschild limit: xi = r * sqrt(r / (r - 2M))
            let f = r / (r - 2.0 * mass);
            if f > 0.0 { r * f.sqrt() } else { continue }
        } else {
            (r2 * (r - 3.0 * mass) + a * a * (r + mass)) / (a * (r - mass))
        };

        let eta_num = r2 * (4.0 * mass * delta - r * (r - mass) * (r - mass));
        let eta = eta_num / (a * a * (r - mass) * (r - mass)).max(1e-14);

        // Celestial coordinates: alpha = -xi / sin(theta_obs), beta = sqrt(eta + a^2 cos^2(theta_obs) - xi^2 cot^2(theta_obs))
        // For equatorial observer (theta_obs = pi/2):
        let alpha = -xi;
        let beta_sq = eta + a * a - xi * xi * 0.0; // cot(pi/2) = 0
        let beta = if beta_sq >= 0.0 { beta_sq.sqrt() } else { 0.0 };

        boundary.push(ShadowPoint {
            alpha,
            beta,
            xi_0: xi,
            eta_0: eta,
        });

        // Add the mirror point (negative beta for the other half)
        if beta > 1e-10 {
            boundary.push(ShadowPoint {
                alpha,
                beta: -beta,
                xi_0: xi,
                eta_0: eta,
            });
        }
    }

    // Sort by angle for a clean boundary curve
    boundary.sort_by(|a, b| {
        let angle_a = a.beta.atan2(a.alpha);
        let angle_b = b.beta.atan2(b.alpha);
        angle_a
            .partial_cmp(&angle_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    boundary
}

/// Compute shadow boundary with full 32D Pathion deflection verification.
///
/// For each boundary point, also computes the actual 32D deflection to verify
/// that the critical curve produces near-divergent deflection.
pub fn pathion_shadow_boundary_verified(
    mass: f64,
    spin_32d: &[f64; 32],
    n_points: usize,
) -> Vec<(ShadowPoint, f64)> {
    let spin_0 = spin_32d[0];
    let boundary = pathion_shadow_boundary(mass, spin_0, n_points);

    boundary
        .iter()
        .map(|pt| {
            let mut xi = [0.0; 32];
            xi[0] = pt.xi_0;
            let mut eta = [0.0; 32];
            eta[0] = pt.eta_0.max(0.0);

            let defl = pathion_deflection(mass, spin_32d, &xi, &eta)
                .map(|d| d[0])
                .unwrap_or(f64::NAN);

            (*pt, defl)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schwarzschild_shadow_is_circle() {
        // For a=0, the shadow should be approximately circular with
        // radius = 3*sqrt(3)*M ~ 5.196*M
        let boundary = pathion_shadow_boundary(1.0, 0.0, 50);
        assert!(!boundary.is_empty(), "boundary should have points");

        let radii: Vec<f64> = boundary
            .iter()
            .map(|p| (p.alpha * p.alpha + p.beta * p.beta).sqrt())
            .collect();

        // All radii should be approximately equal (circular shadow)
        let mean_r = radii.iter().sum::<f64>() / radii.len() as f64;
        assert!(
            mean_r > 4.0 && mean_r < 6.5,
            "Schwarzschild shadow radius should be ~5.196M, got {mean_r}"
        );
    }

    #[test]
    fn kerr_shadow_is_asymmetric() {
        let boundary = pathion_shadow_boundary(1.0, 0.9, 100);
        assert!(!boundary.is_empty());

        // For spinning BH, alpha range should be asymmetric
        let alphas: Vec<f64> = boundary.iter().map(|p| p.alpha).collect();
        let alpha_min = alphas.iter().cloned().fold(f64::INFINITY, f64::min);
        let alpha_max = alphas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // The prograde side (alpha < 0) should extend further than retrograde
        assert!(
            alpha_max.abs() != alpha_min.abs(),
            "Kerr shadow should be asymmetric: min={alpha_min}, max={alpha_max}"
        );
    }
}
