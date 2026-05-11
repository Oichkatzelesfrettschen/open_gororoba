//! Geometry primitives for the Casimir sphere-plate-sphere system.
//!
//! Defines `Sphere`, `Plate`, and the `SpherePlateSphere` three-terminal
//! configuration with positional helpers (gap accessors, micrometer
//! conveniences). These types carry no force-calculation logic on their own
//! -- they are consumed by the PFA, Lifshitz, transistor, additivity, and
//! three-body modules in the same `casimir` parent.

/// A microsphere in the Casimir system.
#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    /// Sphere radius (meters)
    pub radius: f64,
    /// Position along the axis (meters)
    pub position: f64,
}

impl Sphere {
    /// Create a new sphere.
    ///
    /// # Arguments
    /// * `radius` - Sphere radius in meters (typically ~5 micrometers)
    /// * `position` - Position along the system axis in meters
    pub fn new(radius: f64, position: f64) -> Self {
        Sphere { radius, position }
    }

    /// Create a sphere with radius in micrometers (convenience constructor).
    pub fn from_micrometers(radius_um: f64, position_um: f64) -> Self {
        Sphere {
            radius: radius_um * 1e-6,
            position: position_um * 1e-6,
        }
    }
}

/// A conducting plate in the Casimir system.
#[derive(Debug, Clone, Copy)]
pub struct Plate {
    /// Position along the axis (meters)
    pub position: f64,
    /// Plate thickness (meters) - for geometry validation
    pub thickness: f64,
}

impl Plate {
    /// Create a new plate.
    ///
    /// # Arguments
    /// * `position` - Position along the system axis in meters
    /// * `thickness` - Plate thickness in meters
    pub fn new(position: f64, thickness: f64) -> Self {
        Plate {
            position,
            thickness,
        }
    }

    /// Create a plate with dimensions in micrometers.
    pub fn from_micrometers(position_um: f64, thickness_um: f64) -> Self {
        Plate {
            position: position_um * 1e-6,
            thickness: thickness_um * 1e-6,
        }
    }
}

/// The sphere-plate-sphere system configuration.
#[derive(Debug, Clone)]
pub struct SpherePlateSphere {
    /// Source sphere (left side by convention)
    pub source: Sphere,
    /// Conducting gate plate
    pub plate: Plate,
    /// Drain sphere (right side by convention)
    pub drain: Sphere,
}

impl SpherePlateSphere {
    /// Create a new sphere-plate-sphere system.
    ///
    /// # Arguments
    /// * `source` - Source sphere (typically on the left)
    /// * `plate` - Conducting gate plate
    /// * `drain` - Drain sphere (typically on the right)
    ///
    /// # Panics
    /// Panics if the geometry is invalid (spheres overlapping with plate).
    pub fn new(source: Sphere, plate: Plate, drain: Sphere) -> Self {
        let sps = SpherePlateSphere {
            source,
            plate,
            drain,
        };
        assert!(
            sps.source_plate_gap() > 0.0,
            "Source sphere overlaps with plate"
        );
        assert!(
            sps.drain_plate_gap() > 0.0,
            "Drain sphere overlaps with plate"
        );
        sps
    }

    /// Create a symmetric configuration from micrometers.
    ///
    /// # Arguments
    /// * `sphere_radius_um` - Radius of both spheres in micrometers
    /// * `source_gap_um` - Gap between source sphere surface and plate surface (micrometers)
    /// * `drain_gap_um` - Gap between plate surface and drain sphere surface (micrometers)
    /// * `plate_thickness_um` - Plate thickness in micrometers
    pub fn symmetric_from_micrometers(
        sphere_radius_um: f64,
        source_gap_um: f64,
        drain_gap_um: f64,
        plate_thickness_um: f64,
    ) -> Self {
        // Convert all to meters
        let r = sphere_radius_um * 1e-6;
        let g_s = source_gap_um * 1e-6;
        let g_d = drain_gap_um * 1e-6;
        let t = plate_thickness_um * 1e-6;

        // Source sphere at origin (center at 0)
        // Source right edge at r
        // Plate left edge at r + g_s
        // Plate center at r + g_s + t/2
        // Plate right edge at r + g_s + t
        // Drain left edge at r + g_s + t + g_d
        // Drain center at r + g_s + t + g_d + r
        let source_pos = 0.0;
        let plate_pos = r + g_s + t / 2.0;
        let drain_pos = r + g_s + t + g_d + r;

        SpherePlateSphere::new(
            Sphere::new(r, source_pos),
            Plate::new(plate_pos, t),
            Sphere::new(r, drain_pos),
        )
    }

    /// Gap between source sphere surface and plate (meters).
    pub fn source_plate_gap(&self) -> f64 {
        (self.plate.position - self.plate.thickness / 2.0)
            - (self.source.position + self.source.radius)
    }

    /// Gap between plate and drain sphere surface (meters).
    pub fn drain_plate_gap(&self) -> f64 {
        (self.drain.position - self.drain.radius)
            - (self.plate.position + self.plate.thickness / 2.0)
    }
}
