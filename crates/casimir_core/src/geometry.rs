//! Casimir geometry definitions.
//!
//! Defines the `CasimirGeometry` trait and implementations for standard
//! geometries used in Casimir energy calculations:
//! - Parallel plates (canonical validation case)
//! - Solid sphere (boundary crossing diagnostics)
//! - Infinite cylinder
//! - Sphere inside cylinder (disconnected two-body interaction)
//! - Pillar in cavity (connected boundary diagnostics)
//!
//! Crossing counts describe geometry only. Two-body interaction subtraction
//! uses distinct body identities through interaction_hit_mask; repeated
//! crossings of a single body carry zero interaction weight.

/// Geometry errors in the disconnected two-body interaction contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionGeometryError {
    /// The geometry lacks an explicit pair of disconnected bodies.
    UnsupportedGeometry,
    /// Body dimensions are nonfinite, nonpositive, touching, or overlapping.
    InvalidGeometry,
    /// Finite coordinates exceed the representable range of the geometric predicate.
    NumericalRange,
}

impl std::fmt::Display for InteractionGeometryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::UnsupportedGeometry => {
                "geometry requires an explicit disconnected two-body interaction"
            }
            Self::NumericalRange => "radial geometry predicate exceeds floating-point range",
            Self::InvalidGeometry => {
                "two-body geometry dimensions must be finite, positive, and disjoint"
            }
        })
    }
}
impl std::error::Error for InteractionGeometryError {}

/// Trait for Casimir boundary geometries.
///
/// Implementations define a closed surface (or set of surfaces) in 3D.
/// Diagnostic crossing counts remain separate from distinct-body interaction masks.
pub trait CasimirGeometry: Send + Sync {
    /// Count polygon segment crossing events for boundary diagnostics.
    ///
    /// Repeated crossings of one surface contribute repeatedly. The count
    /// therefore cannot substitute for a distinct-body interaction mask.
    ///
    /// A loop "intersects" a surface if any of its line segments crosses
    /// from one side to the other.
    ///
    /// # Arguments
    /// * `loop_points` - Points of the scaled loop (already at proper time T).
    /// * `center` - The point x around which the loop is centered.
    /// * `scale` - The sqrt(T) scale factor (for reference; points are pre-scaled).
    fn intersects_count(&self, loop_points: &[[f64; 3]], center: [f64; 3], scale: f64) -> usize;

    /// Validate the declared disconnected two-body interaction.
    ///
    /// A crossing-count implementation alone cannot establish body identity.
    fn validate_interaction(&self) -> Result<(), InteractionGeometryError> {
        Err(InteractionGeometryError::UnsupportedGeometry)
    }

    /// Return distinct body hits for a polygon with an implicit closing edge.
    ///
    /// Coordinates are already scaled; center supplies the spatial translation.
    fn interaction_hit_mask(
        &self,
        _loop_points: &[[f64; 3]],
        _center: [f64; 3],
    ) -> Result<[bool; 2], InteractionGeometryError> {
        Err(InteractionGeometryError::UnsupportedGeometry)
    }

    /// Human-readable name.
    fn name(&self) -> &str;
}

/// Two infinite parallel plates at z = 0 and z = a.
///
/// The canonical Casimir geometry. Analytical result:
/// Real-scalar Dirichlet E/A = -pi^2 / (1440 a^3).
#[derive(Debug, Clone)]
pub struct ParallelPlates {
    /// Plate separation [same units as loop coordinates].
    pub separation: f64,
}

impl CasimirGeometry for ParallelPlates {
    fn validate_interaction(&self) -> Result<(), InteractionGeometryError> {
        if self.separation.is_finite() && self.separation > 0.0 {
            Ok(())
        } else {
            Err(InteractionGeometryError::InvalidGeometry)
        }
    }

    fn interaction_hit_mask(
        &self,
        loop_points: &[[f64; 3]],
        center: [f64; 3],
    ) -> Result<[bool; 2], InteractionGeometryError> {
        self.validate_interaction()?;
        validate_polygon_coordinates(loop_points, center)?;
        if loop_points.is_empty() {
            return Ok([false; 2]);
        }
        let minimum = loop_points
            .iter()
            .map(|point| point[2] + center[2])
            .fold(f64::INFINITY, f64::min);
        let maximum = loop_points
            .iter()
            .map(|point| point[2] + center[2])
            .fold(f64::NEG_INFINITY, f64::max);
        Ok([
            minimum <= 0.0 && maximum >= 0.0,
            minimum <= self.separation && maximum >= self.separation,
        ])
    }

    fn intersects_count(&self, loop_points: &[[f64; 3]], center: [f64; 3], _scale: f64) -> usize {
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
    fn intersects_count(&self, loop_points: &[[f64; 3]], center: [f64; 3], _scale: f64) -> usize {
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
    fn intersects_count(&self, loop_points: &[[f64; 3]], center: [f64; 3], _scale: f64) -> usize {
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

/// Concentric sphere and cylindrical shell, with strict radial separation.
#[derive(Debug, Clone)]
pub struct SphereInCylinder {
    pub sphere_radius: f64,
    pub cylinder_radius: f64,
}

impl CasimirGeometry for SphereInCylinder {
    fn validate_interaction(&self) -> Result<(), InteractionGeometryError> {
        if self.sphere_radius.is_finite()
            && self.cylinder_radius.is_finite()
            && self.sphere_radius > 0.0
            && self.sphere_radius < self.cylinder_radius
        {
            Ok(())
        } else {
            Err(InteractionGeometryError::InvalidGeometry)
        }
    }

    fn interaction_hit_mask(
        &self,
        loop_points: &[[f64; 3]],
        center: [f64; 3],
    ) -> Result<[bool; 2], InteractionGeometryError> {
        self.validate_interaction()?;
        validate_polygon_coordinates(loop_points, center)?;
        let mut hits = [false; 2];
        for vertex in 0..loop_points.len() {
            let next = (vertex + 1) % loop_points.len();
            let start =
                std::array::from_fn(|dimension| center[dimension] + loop_points[vertex][dimension]);
            let end =
                std::array::from_fn(|dimension| center[dimension] + loop_points[next][dimension]);
            hits[0] |= segment_hits_radial_surface(start, end, self.sphere_radius, 3)?;
            hits[1] |= segment_hits_radial_surface(start, end, self.cylinder_radius, 2)?;
        }
        Ok(hits)
    }

    fn intersects_count(&self, loop_points: &[[f64; 3]], center: [f64; 3], _scale: f64) -> usize {
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

/// Pillar touching both plates; interaction subtraction requires other models.
///
/// Two plates at z = 0 and z = a, plus a cylindrical pillar of radius r_p
/// along the z-axis connecting them.
#[derive(Debug, Clone)]
pub struct PillarInCavity {
    pub plate_separation: f64,
    pub pillar_radius: f64,
}

impl CasimirGeometry for PillarInCavity {
    fn intersects_count(&self, loop_points: &[[f64; 3]], center: [f64; 3], _scale: f64) -> usize {
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

fn validate_polygon_coordinates(
    points: &[[f64; 3]],
    center: [f64; 3],
) -> Result<(), InteractionGeometryError> {
    if center.iter().any(|coordinate| !coordinate.is_finite())
        || points.iter().any(|point| {
            point.iter().zip(center).any(|(coordinate, translation)| {
                !coordinate.is_finite() || !(coordinate + translation).is_finite()
            })
        })
    {
        Err(InteractionGeometryError::NumericalRange)
    } else {
        Ok(())
    }
}

fn geometric_product(first: f64, second: f64) -> Result<f64, InteractionGeometryError> {
    let product = first * second;
    if !product.is_finite() || (first != 0.0 && second != 0.0 && product == 0.0) {
        Err(InteractionGeometryError::NumericalRange)
    } else {
        Ok(product)
    }
}

// The closest segment point catches enter-and-exit crossings whose endpoints
// both lie outside. Normalizing lengths before squaring preserves geometrically
// identical huge and tiny configurations. Lost nonzero terms produce an error
// instead of a false surface hit; ordinary floating-point roundoff remains.
fn segment_hits_radial_surface(
    start: [f64; 3],
    end: [f64; 3],
    radius: f64,
    dimensions: usize,
) -> Result<bool, InteractionGeometryError> {
    let length_scale = start[..dimensions]
        .iter()
        .chain(&end[..dimensions])
        .map(|coordinate| coordinate.abs())
        .fold(radius, f64::max);
    let normalize = |coordinate: f64| {
        let normalized = coordinate / length_scale;
        if coordinate != 0.0 && normalized == 0.0 {
            Err(InteractionGeometryError::NumericalRange)
        } else {
            Ok(normalized)
        }
    };
    let normalized_radius = normalize(radius)?;
    let radius_squared = geometric_product(normalized_radius, normalized_radius)?;
    let mut normalized_start = [0.0; 3];
    let mut displacements = [0.0; 3];
    let mut norm_start = 0.0;
    let mut norm_end = 0.0;
    let mut length_squared = 0.0;
    let mut projection = 0.0;
    for dimension in 0..dimensions {
        let first = normalize(start[dimension])?;
        let second = normalize(end[dimension])?;
        let displacement = second - first;
        if start[dimension] != end[dimension] && displacement == 0.0 {
            return Err(InteractionGeometryError::NumericalRange);
        }
        normalized_start[dimension] = first;
        displacements[dimension] = displacement;
        norm_start += geometric_product(first, first)?;
        norm_end += geometric_product(second, second)?;
        length_squared += geometric_product(displacement, displacement)?;
        projection += geometric_product(first, displacement)?;
    }
    let fraction = if length_squared > 0.0 {
        (-projection / length_squared).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut minimum_squared = 0.0;
    for dimension in 0..dimensions {
        let closest_coordinate =
            normalized_start[dimension] + geometric_product(fraction, displacements[dimension])?;
        minimum_squared += geometric_product(closest_coordinate, closest_coordinate)?;
    }
    Ok(minimum_squared <= radius_squared && norm_start.max(norm_end) >= radius_squared)
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
        assert!(
            count > 0,
            "loop crossing cylinder boundary should intersect"
        );
    }

    #[test]
    fn interaction_excludes_repeated_hits_of_one_plate() {
        let geometry = ParallelPlates { separation: 1.0 };
        let polygon = [
            [0.0, 0.0, -0.2],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, -0.1],
            [0.0, 0.0, 0.1],
        ];
        assert!(geometry.intersects_count(&polygon, [0.0; 3], 1.0) >= 2);
        assert_eq!(
            geometry.interaction_hit_mask(&polygon, [0.0; 3]),
            Ok([true, false])
        );
        let spanning = [[0.0, 0.0, -0.2], [0.0, 0.0, 1.2], [0.0; 3]];
        assert_eq!(
            geometry.interaction_hit_mask(&spanning, [0.0; 3]),
            Ok([true, true])
        );
    }

    #[test]
    fn sphere_cylinder_uses_distinct_bodies_and_segment_intersections() {
        let geometry = SphereInCylinder {
            sphere_radius: 1.0,
            cylinder_radius: 3.0,
        };
        let sphere_only = [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        assert_eq!(
            geometry.interaction_hit_mask(&sphere_only, [0.0; 3]),
            Ok([true, false])
        );
        let both = [[-4.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]];
        assert_eq!(
            geometry.interaction_hit_mask(&both, [0.0; 3]),
            Ok([true, true])
        );
        assert_eq!(
            SolidSphere {
                center: [0.0; 3],
                radius: 1.0
            }
            .validate_interaction(),
            Err(InteractionGeometryError::UnsupportedGeometry)
        );
        assert!(
            SphereInCylinder {
                sphere_radius: 3.0,
                cylinder_radius: 3.0
            }
            .validate_interaction()
            .is_err()
        );
    }

    #[test]
    fn radial_hits_preserve_huge_and_tiny_length_scales() {
        for length_scale in [1e200, 1e-200] {
            let geometry = SphereInCylinder {
                sphere_radius: length_scale,
                cylinder_radius: 2.0 * length_scale,
            };
            let point_polygon = [[0.0; 3]; 3];
            assert_eq!(
                geometry.interaction_hit_mask(&point_polygon, [10.0 * length_scale, 0.0, 0.0]),
                Ok([false, false])
            );
            assert_eq!(
                geometry.interaction_hit_mask(&point_polygon, [0.0; 3]),
                Ok([false, false])
            );
            let spanning = [
                [-4.0 * length_scale, 0.0, 0.0],
                [4.0 * length_scale, 0.0, 0.0],
                [0.0, 4.0 * length_scale, 0.0],
            ];
            assert_eq!(
                geometry.interaction_hit_mask(&spanning, [0.0; 3]),
                Ok([true, true])
            );
        }
    }
}
