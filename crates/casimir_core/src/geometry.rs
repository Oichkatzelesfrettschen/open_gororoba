//! Casimir geometry definitions.
//!
//! Defines the `CasimirGeometry` trait and implementations for standard
//! geometries used in Casimir energy calculations:
//! - Parallel plates (canonical validation case)
//! - Solid sphere (Casimir-Polder)
//! - Infinite cylinder
//! - Sphere inside cylinder (toroidal Casimir region)
//! - Pillar in cavity (Alcubierre-like pattern, White 2021)
//!
//! The trait method `intersects_count` determines how many boundary surfaces
//! a scaled worldline loop crosses, which enters the Theta_Sigma function
//! in the path integral.

/// Trait for Casimir boundary geometries.
///
/// Implementations define a closed surface (or set of surfaces) in 3D.
/// The worldline numerics check how many surfaces each scaled loop intersects.
pub trait CasimirGeometry: Send + Sync {
    /// Count how many boundary surfaces the loop (centered at `center` and
    /// scaled by `scale`) intersects.
    ///
    /// A loop "intersects" a surface if any of its line segments crosses
    /// from one side to the other.
    ///
    /// # Arguments
    /// * `loop_points` - Points of the scaled loop (already at proper time T).
    /// * `center` - The point x around which the loop is centered.
    /// * `scale` - The sqrt(T) scale factor (for reference; points are pre-scaled).
    fn intersects_count(
        &self,
        loop_points: &[[f64; 3]],
        center: [f64; 3],
        scale: f64,
    ) -> usize;

    /// Human-readable name.
    fn name(&self) -> &str;
}

/// Two infinite parallel plates at z = 0 and z = a.
///
/// The canonical Casimir geometry. Analytical result:
/// E/A = -pi^2 / (720 a^3)
#[derive(Debug, Clone)]
pub struct ParallelPlates {
    /// Plate separation [same units as loop coordinates].
    pub separation: f64,
}

impl CasimirGeometry for ParallelPlates {
    fn intersects_count(
        &self,
        loop_points: &[[f64; 3]],
        center: [f64; 3],
        _scale: f64,
    ) -> usize {
        let a = self.separation;
        let mut count = 0;

        for i in 0..loop_points.len() {
            let j = (i + 1) % loop_points.len();
            // Absolute z positions
            let z1 = center[2] + loop_points[i][2];
            let z2 = center[2] + loop_points[j][2];

            // Check crossing of z = 0 plane
            if (z1 < 0.0 && z2 >= 0.0) || (z1 >= 0.0 && z2 < 0.0) {
                count += 1;
            }
            // Check crossing of z = a plane
            if (z1 < a && z2 >= a) || (z1 >= a && z2 < a) {
                count += 1;
            }
        }

        count
    }

    fn name(&self) -> &str {
        "parallel_plates"
    }
}

/// Solid sphere boundary.
#[derive(Debug, Clone)]
pub struct SolidSphere {
    pub center: [f64; 3],
    pub radius: f64,
}

impl CasimirGeometry for SolidSphere {
    fn intersects_count(
        &self,
        loop_points: &[[f64; 3]],
        center: [f64; 3],
        _scale: f64,
    ) -> usize {
        let r2 = self.radius * self.radius;
        let mut count = 0;

        for i in 0..loop_points.len() {
            let j = (i + 1) % loop_points.len();

            let dx1 = center[0] + loop_points[i][0] - self.center[0];
            let dy1 = center[1] + loop_points[i][1] - self.center[1];
            let dz1 = center[2] + loop_points[i][2] - self.center[2];
            let inside1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1 < r2;

            let dx2 = center[0] + loop_points[j][0] - self.center[0];
            let dy2 = center[1] + loop_points[j][1] - self.center[1];
            let dz2 = center[2] + loop_points[j][2] - self.center[2];
            let inside2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2 < r2;

            if inside1 != inside2 {
                count += 1;
            }
        }

        count
    }

    fn name(&self) -> &str {
        "solid_sphere"
    }
}

/// Infinite cylinder along the z-axis.
#[derive(Debug, Clone)]
pub struct InfiniteCylinder {
    pub radius: f64,
}

impl CasimirGeometry for InfiniteCylinder {
    fn intersects_count(
        &self,
        loop_points: &[[f64; 3]],
        center: [f64; 3],
        _scale: f64,
    ) -> usize {
        let r2 = self.radius * self.radius;
        let mut count = 0;

        for i in 0..loop_points.len() {
            let j = (i + 1) % loop_points.len();

            let x1 = center[0] + loop_points[i][0];
            let y1 = center[1] + loop_points[i][1];
            let rho1_sq = x1 * x1 + y1 * y1;
            let inside1 = rho1_sq < r2;

            let x2 = center[0] + loop_points[j][0];
            let y2 = center[1] + loop_points[j][1];
            let rho2_sq = x2 * x2 + y2 * y2;
            let inside2 = rho2_sq < r2;

            if inside1 != inside2 {
                count += 1;
            }
        }

        count
    }

    fn name(&self) -> &str {
        "infinite_cylinder"
    }
}

/// Sphere inside a cylinder (White 2021 geometry for toroidal Casimir energy).
#[derive(Debug, Clone)]
pub struct SphereInCylinder {
    pub sphere_radius: f64,
    pub cylinder_radius: f64,
}

impl CasimirGeometry for SphereInCylinder {
    fn intersects_count(
        &self,
        loop_points: &[[f64; 3]],
        center: [f64; 3],
        _scale: f64,
    ) -> usize {
        let rs2 = self.sphere_radius * self.sphere_radius;
        let rc2 = self.cylinder_radius * self.cylinder_radius;
        let mut count = 0;

        for i in 0..loop_points.len() {
            let j = (i + 1) % loop_points.len();

            let p1 = [
                center[0] + loop_points[i][0],
                center[1] + loop_points[i][1],
                center[2] + loop_points[i][2],
            ];
            let p2 = [
                center[0] + loop_points[j][0],
                center[1] + loop_points[j][1],
                center[2] + loop_points[j][2],
            ];

            // Sphere crossing
            let sr1 = p1[0] * p1[0] + p1[1] * p1[1] + p1[2] * p1[2];
            let sr2 = p2[0] * p2[0] + p2[1] * p2[1] + p2[2] * p2[2];
            if (sr1 < rs2) != (sr2 < rs2) {
                count += 1;
            }

            // Cylinder crossing (rho^2 = x^2 + y^2)
            let cr1 = p1[0] * p1[0] + p1[1] * p1[1];
            let cr2 = p2[0] * p2[0] + p2[1] * p2[1];
            if (cr1 < rc2) != (cr2 < rc2) {
                count += 1;
            }
        }

        count
    }

    fn name(&self) -> &str {
        "sphere_in_cylinder"
    }
}

/// Pillar in a plate cavity (White 2021 geometry producing Alcubierre-like pattern).
///
/// Two plates at z = 0 and z = a, plus a cylindrical pillar of radius r_p
/// along the z-axis connecting them.
#[derive(Debug, Clone)]
pub struct PillarInCavity {
    pub plate_separation: f64,
    pub pillar_radius: f64,
}

impl CasimirGeometry for PillarInCavity {
    fn intersects_count(
        &self,
        loop_points: &[[f64; 3]],
        center: [f64; 3],
        _scale: f64,
    ) -> usize {
        let a = self.plate_separation;
        let rp2 = self.pillar_radius * self.pillar_radius;
        let mut count = 0;

        for i in 0..loop_points.len() {
            let j = (i + 1) % loop_points.len();

            let p1 = [
                center[0] + loop_points[i][0],
                center[1] + loop_points[i][1],
                center[2] + loop_points[i][2],
            ];
            let p2 = [
                center[0] + loop_points[j][0],
                center[1] + loop_points[j][1],
                center[2] + loop_points[j][2],
            ];

            // Plate crossings
            if (p1[2] < 0.0 && p2[2] >= 0.0) || (p1[2] >= 0.0 && p2[2] < 0.0) {
                count += 1;
            }
            if (p1[2] < a && p2[2] >= a) || (p1[2] >= a && p2[2] < a) {
                count += 1;
            }

            // Pillar crossing (cylinder along z)
            let rho1 = p1[0] * p1[0] + p1[1] * p1[1];
            let rho2 = p2[0] * p2[0] + p2[1] * p2[1];
            if (rho1 < rp2) != (rho2 < rp2) {
                count += 1;
            }
        }

        count
    }

    fn name(&self) -> &str {
        "pillar_in_cavity"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plates_no_intersection_inside() {
        let plates = ParallelPlates { separation: 1.0 };
        // A tiny loop entirely between the plates should not intersect
        let points = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
        let count = plates.intersects_count(&points, [0.0, 0.0, 0.5], 1.0);
        assert_eq!(count, 0, "tiny loop inside should not intersect plates");
    }

    #[test]
    fn test_plates_intersection_crossing() {
        let plates = ParallelPlates { separation: 1.0 };
        // A loop that crosses z=0
        let points = vec![[0.0, 0.0, -0.6], [0.0, 0.0, 0.6], [0.01, 0.0, 0.0]];
        let center = [0.0, 0.0, 0.0]; // center at z=0
        let count = plates.intersects_count(&points, center, 1.0);
        assert!(count > 0, "loop crossing z=0 should intersect: {count}");
    }

    #[test]
    fn test_sphere_trivially_outside() {
        let sphere = SolidSphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        // All loop points far away
        let points = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
        let count = sphere.intersects_count(&points, [10.0, 10.0, 10.0], 1.0);
        assert_eq!(count, 0, "loop far from sphere should not intersect");
    }

    #[test]
    fn test_cylinder_crossing() {
        let cyl = InfiniteCylinder { radius: 1.0 };
        // Loop that straddles the cylinder boundary
        let points = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
        let center = [0.0, 0.0, 0.0]; // center at origin
        let count = cyl.intersects_count(&points, center, 1.0);
        assert!(count > 0, "loop crossing cylinder boundary should intersect");
    }
}
