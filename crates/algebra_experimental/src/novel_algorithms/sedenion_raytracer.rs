//! Non-Associative Geodesic Raytracing
//!
//! Extends general relativity into a 16-D metric tensor space where the metric
//! itself is sedenion-valued. This algorithm traces "rays" (geodesics) through
//! a manifold that contains Zero-Divisor (ZD) singularities.
//!
//! When a ray enters a ZD region, its momentum tensor undergoes non-associative
//! fracturing--the light path splits deterministically based on the failure of
//! the alternative property.

use cd_kernel::cayley_dickson::cd_multiply;

/// A space-time position and momentum in 16D coordinates.
#[derive(Debug, Clone)]
pub struct RayState {
    pub position: [f64; 16],
    pub momentum: [f64; 16],
}

/// **Non-Associative Geodesic Step**
/// Advances the ray by one time-step `dt`. The Christoffel symbol equivalent
/// is defined by the Sedenion multiplication of the local metric field with the momentum.
pub fn non_associative_geodesic_step(
    ray: &mut RayState,
    local_metric: &[f64; 16],
    dt: f64,
) -> Option<RayState> {
    // 1. Advance position: dx = p * dt
    for i in 0..16 {
        ray.position[i] += ray.momentum[i] * dt;
    }

    // 2. Compute non-associative curvature: dp = (Metric * Momentum) * Momentum
    let p1: [f64; 16] = cd_multiply(local_metric, &ray.momentum).try_into().unwrap();
    let curvature_force: [f64; 16] = cd_multiply(&p1, &ray.momentum).try_into().unwrap();

    // 3. Measure alternativity failure (Fracture condition)
    // If the space was associative/alternative, (M*P)*P == M*(P*P).
    let pp: [f64; 16] = cd_multiply(&ray.momentum, &ray.momentum)
        .try_into()
        .unwrap();
    let alt_curvature: [f64; 16] = cd_multiply(local_metric, &pp).try_into().unwrap();

    let mut fracture_magnitude: f64 = 0.0;
    for i in 0..16 {
        fracture_magnitude += (curvature_force[i] - alt_curvature[i]).powi(2);
    }
    fracture_magnitude = fracture_magnitude.sqrt();

    // 4. Update primary momentum
    for (m, &cf) in ray.momentum.iter_mut().zip(curvature_force.iter()) {
        *m += cf * dt;
    }

    // 5. If the alternativity failure is high (we hit a ZD singularity), the ray fractures.
    if fracture_magnitude > 1e-3 {
        let mut fractured_ray = ray.clone();
        for (m, &ac) in fractured_ray.momentum.iter_mut().zip(alt_curvature.iter()) {
            // The secondary ray takes the alternative path's momentum
            *m = ac * dt;
        }
        return Some(fractured_ray);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geodesic_fracture() {
        let mut ray = RayState {
            position: [0.0; 16],
            momentum: [0.1; 16],
        };
        // Inject a ZD-inducing momentum
        ray.momentum[15] = 1.0;
        ray.momentum[4] = -1.0;

        let mut metric = [0.0; 16];
        metric[1] = 1.0;
        metric[10] = 1.0;

        let fractured = non_associative_geodesic_step(&mut ray, &metric, 0.01);

        // The non-alternative metric should cause the ray to spawn a secondary path
        assert!(fractured.is_some());
    }
}
