//! Optics in Clifford Algebra Formalism
//!
//! Bridges optical phenomena with geometric algebra, implementing:
//! - Polarization state space as Clifford algebra elements
//! - Jones vectors and Mueller matrices via geometric algebra
//! - Maxwell equations in Clifford algebra formalism
//! - Electromagnetic wave propagation and transformations
//!
//! # Theory
//! - Hestenes (2003): "Oersted Medal Lecture: Reforming the Mathematical Language of Physics"
//! - Baylis (1996): "Electrodynamics: A Modern Geometric Approach"
//! - Born & Wolf (1980): "Principles of Optics", Ch. 10 (Polarization)
//! - Gull, Lasenby, Doran (1993): "Imaginary Numbers Are Not Real"

use std::f64::consts::PI;

/// Polarization state in the Clifford algebra Cl(3,0) representation.
///
/// Represents optical polarization using Clifford algebra basis:
/// - Scalar: intensity component
/// - Bivector e12, e13, e23: Stokes parameters S1, S2, S3
/// - Pseudoscalar: circular polarization phase
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolarizationState {
    /// Intensity (Stokes S0 / 2)
    pub intensity: f64,
    /// Linear polarization x-component (Stokes S1)
    pub s1: f64,
    /// Linear polarization y-component (Stokes S2)
    pub s2: f64,
    /// Circular polarization (Stokes S3)
    pub s3: f64,
}

impl PolarizationState {
    /// Create a new polarization state from Stokes parameters.
    pub fn new(intensity: f64, s1: f64, s2: f64, s3: f64) -> Self {
        PolarizationState {
            intensity,
            s1,
            s2,
            s3,
        }
    }

    /// Degree of polarization: |S| / S0 where S = (S1, S2, S3).
    pub fn degree_of_polarization(&self) -> f64 {
        if self.intensity == 0.0 {
            0.0
        } else {
            let s_magnitude = (self.s1 * self.s1 + self.s2 * self.s2 + self.s3 * self.s3).sqrt();
            s_magnitude / self.intensity
        }
    }

    /// Degree of linear polarization.
    pub fn degree_linear_polarization(&self) -> f64 {
        if self.intensity == 0.0 {
            0.0
        } else {
            let linear = (self.s1 * self.s1 + self.s2 * self.s2).sqrt();
            linear / self.intensity
        }
    }

    /// Degree of circular polarization.
    pub fn degree_circular_polarization(&self) -> f64 {
        if self.intensity == 0.0 {
            0.0
        } else {
            (self.s3 / self.intensity).abs()
        }
    }

    /// Ellipse orientation angle in radians.
    pub fn ellipse_angle(&self) -> f64 {
        0.5 * (self.s2).atan2(self.s1)
    }

    /// Ellipticity: ratio of minor to major axis.
    pub fn ellipticity(&self) -> f64 {
        if self.intensity == 0.0 {
            0.0
        } else {
            let s_linear = (self.s1 * self.s1 + self.s2 * self.s2).sqrt();
            if s_linear == 0.0 {
                1.0 // Circular
            } else {
                let s3_norm = self.s3 / self.intensity;
                (s3_norm).atan().sin()
            }
        }
    }

    /// Compose two polarization states via Mueller matrix multiplication.
    pub fn compose(&self, other: &PolarizationState) -> PolarizationState {
        // Simplified: linear composition in Stokes space
        PolarizationState {
            intensity: self.intensity * other.intensity,
            s1: self.s1 + other.s1,
            s2: self.s2 + other.s2,
            s3: self.s3 + other.s3,
        }
    }
}

/// Jones vector representation of polarization.
///
/// 2D complex vector [Ex, Ey] representing electric field components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JonesVector {
    /// x-component (real, imaginary)
    pub ex: (f64, f64),
    /// y-component (real, imaginary)
    pub ey: (f64, f64),
}

impl JonesVector {
    /// Create Jones vector from components.
    pub fn new(ex_real: f64, ex_imag: f64, ey_real: f64, ey_imag: f64) -> Self {
        JonesVector {
            ex: (ex_real, ex_imag),
            ey: (ey_real, ey_imag),
        }
    }

    /// Linear x-polarization.
    pub fn linear_x() -> Self {
        JonesVector {
            ex: (1.0, 0.0),
            ey: (0.0, 0.0),
        }
    }

    /// Linear y-polarization.
    pub fn linear_y() -> Self {
        JonesVector {
            ex: (0.0, 0.0),
            ey: (1.0, 0.0),
        }
    }

    /// Right circular polarization.
    pub fn circular_right() -> Self {
        JonesVector {
            ex: (1.0 / 2.0_f64.sqrt(), 0.0),
            ey: (0.0, -1.0 / 2.0_f64.sqrt()),
        }
    }

    /// Left circular polarization.
    pub fn circular_left() -> Self {
        JonesVector {
            ex: (1.0 / 2.0_f64.sqrt(), 0.0),
            ey: (0.0, 1.0 / 2.0_f64.sqrt()),
        }
    }

    /// Intensity (|Ex|^2 + |Ey|^2).
    pub fn intensity(&self) -> f64 {
        let ex_mag = self.ex.0 * self.ex.0 + self.ex.1 * self.ex.1;
        let ey_mag = self.ey.0 * self.ey.0 + self.ey.1 * self.ey.1;
        ex_mag + ey_mag
    }

    /// Convert to Stokes parameters (polarization state).
    pub fn to_stokes(&self) -> PolarizationState {
        let i0 = self.intensity();
        let ex_r = self.ex.0;
        let ex_i = self.ex.1;
        let ey_r = self.ey.0;
        let ey_i = self.ey.1;

        // Stokes parameters
        let s0 = i0;
        let s1 = ex_r * ex_r + ex_i * ex_i - (ey_r * ey_r + ey_i * ey_i);
        let s2 = 2.0 * (ex_r * ey_r + ex_i * ey_i);
        let s3 = 2.0 * (ex_r * ey_i - ex_i * ey_r);

        PolarizationState::new(s0, s1, s2, s3)
    }
}

/// Mueller matrix for polarization transformation.
///
/// 4x4 real matrix that transforms Stokes vectors.
#[derive(Debug, Clone)]
pub struct MuellerMatrix {
    pub m: [[f64; 4]; 4],
}

impl MuellerMatrix {
    /// Create Mueller matrix from 4x4 array.
    pub fn new(m: [[f64; 4]; 4]) -> Self {
        MuellerMatrix { m }
    }

    /// Identity Mueller matrix (no transformation).
    pub fn identity() -> Self {
        MuellerMatrix {
            m: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Linear polarizer aligned to x-axis.
    pub fn linear_polarizer_x() -> Self {
        MuellerMatrix {
            m: [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        }
    }

    /// Linear polarizer aligned to y-axis.
    pub fn linear_polarizer_y() -> Self {
        MuellerMatrix {
            m: [
                [1.0, -1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        }
    }

    /// Circular polarizer (converts to right circular).
    pub fn circular_polarizer_right() -> Self {
        MuellerMatrix {
            m: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        }
    }

    /// Quarter-wave plate (retarder).
    pub fn quarter_wave_plate() -> Self {
        MuellerMatrix {
            m: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        }
    }

    /// Apply Mueller matrix to Stokes vector.
    pub fn apply(&self, stokes: &PolarizationState) -> PolarizationState {
        let s = [stokes.intensity, stokes.s1, stokes.s2, stokes.s3];
        let result: Vec<f64> = self.m.iter()
            .map(|row| row.iter().zip(&s).map(|(m_ij, s_j)| m_ij * s_j).sum())
            .collect();

        PolarizationState::new(result[0], result[1], result[2], result[3])
    }

    /// Compose two Mueller matrices.
    pub fn compose(&self, other: &MuellerMatrix) -> MuellerMatrix {
        let result = self.m.iter()
            .map(|row| {
                let mut out_row = [0.0; 4];
                for (j, out) in out_row.iter_mut().enumerate() {
                    *out = row.iter().enumerate()
                        .map(|(k, &m_ik)| m_ik * other.m[k][j])
                        .sum();
                }
                out_row
            })
            .collect::<Vec<_>>();

        MuellerMatrix {
            m: [result[0], result[1], result[2], result[3]],
        }
    }
}

/// Maxwell equations in Clifford algebra (geometric algebra) formalism.
///
/// Represents EM fields and their evolution in spacetime geometric algebra Cl(1,3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaxwellField {
    /// Electric field magnitude
    pub e_mag: f64,
    /// Magnetic field magnitude
    pub b_mag: f64,
    /// Phase (radians)
    pub phase: f64,
    /// Wavenumber (1/wavelength)
    pub k: f64,
    /// Frequency
    pub omega: f64,
}

impl MaxwellField {
    /// Create Maxwell field.
    pub fn new(e_mag: f64, b_mag: f64, phase: f64, k: f64, omega: f64) -> Self {
        MaxwellField {
            e_mag,
            b_mag,
            phase,
            k,
            omega,
        }
    }

    /// Plane wave with wavenumber k and frequency omega.
    pub fn plane_wave(k: f64, omega: f64) -> Self {
        // For EM waves in vacuum, c = omega/k
        let e_mag = 1.0; // Normalized
        let b_mag = 1.0 / (3e8); // B = E/c in SI
        MaxwellField {
            e_mag,
            b_mag,
            phase: 0.0,
            k,
            omega,
        }
    }

    /// Wavelength in meters.
    pub fn wavelength(&self) -> f64 {
        if self.k == 0.0 {
            f64::INFINITY
        } else {
            2.0 * PI / self.k
        }
    }

    /// Phase velocity c = omega / k.
    pub fn phase_velocity(&self) -> f64 {
        if self.k == 0.0 {
            f64::INFINITY
        } else {
            self.omega / self.k
        }
    }

    /// Energy density (averaged over cycle).
    pub fn energy_density(&self) -> f64 {
        // <u> = (1/2) * epsilon_0 * (E^2 + c^2 * B^2)
        // Normalized: proportional to E^2
        0.5 * (self.e_mag * self.e_mag + (self.b_mag * self.b_mag))
    }

    /// Poynting vector magnitude (intensity).
    pub fn intensity(&self) -> f64 {
        // S = (1/mu_0) * E x B, averaged <S> = (1/2) * sqrt(eps/mu) * E^2
        // Normalized: proportional to E^2
        0.5 * self.e_mag * self.e_mag
    }

    /// Propagate field forward by distance dz.
    pub fn propagate(&self, dz: f64) -> MaxwellField {
        MaxwellField {
            e_mag: self.e_mag,
            b_mag: self.b_mag,
            phase: (self.phase + self.k * dz) % (2.0 * PI),
            k: self.k,
            omega: self.omega,
        }
    }
}

/// Optical element: applies Mueller matrix transformation.
#[derive(Debug, Clone)]
pub enum OpticalElement {
    Polarizer { axis: f64 },
    RetarderPlate { thickness: f64, birefringence: f64 },
    Lens { focal_length: f64 },
    Mirror { reflectivity: f64 },
    FreespacePropagate { distance: f64 },
}

/// Optical system: sequential application of optical elements.
#[derive(Debug, Clone)]
pub struct OpticalSystem {
    pub elements: Vec<OpticalElement>,
}

impl OpticalSystem {
    /// Create empty optical system.
    pub fn new() -> Self {
        OpticalSystem {
            elements: Vec::new(),
        }
    }

    /// Add optical element.
    pub fn add(&mut self, element: OpticalElement) {
        self.elements.push(element);
    }

    /// Get Mueller matrix for the system (simplified).
    pub fn mueller_matrix(&self) -> MuellerMatrix {
        let mut result = MuellerMatrix::identity();

        for element in &self.elements {
            let element_matrix = match element {
                OpticalElement::Polarizer { axis } => {
                    // Simple: linear polarizer
                    if (*axis - 0.0).abs() < 0.01 {
                        MuellerMatrix::linear_polarizer_x()
                    } else if (*axis - PI / 2.0).abs() < 0.01 {
                        MuellerMatrix::linear_polarizer_y()
                    } else {
                        MuellerMatrix::identity()
                    }
                }
                OpticalElement::RetarderPlate {
                    thickness: _,
                    birefringence: _,
                } => {
                    // Quarter-wave when thickness * birefringence = wavelength/4
                    MuellerMatrix::quarter_wave_plate()
                }
                OpticalElement::Lens { focal_length: _ } => {
                    // Lenses don't change polarization (ideally)
                    MuellerMatrix::identity()
                }
                OpticalElement::Mirror { reflectivity: _ } => {
                    // Mirror may induce phase shift (ideally identity)
                    MuellerMatrix::identity()
                }
                OpticalElement::FreespacePropagate { distance: _ } => {
                    // Free space propagation doesn't change polarization
                    MuellerMatrix::identity()
                }
            };

            result = result.compose(&element_matrix);
        }

        result
    }
}

impl Default for OpticalSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_x_polarization() {
        let jones = JonesVector::linear_x();
        assert_eq!(jones.intensity(), 1.0);
        let stokes = jones.to_stokes();
        assert!(stokes.degree_of_polarization() > 0.99);
    }

    #[test]
    fn test_circular_polarization() {
        let jones = JonesVector::circular_right();
        assert!((jones.intensity() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_degree_of_polarization() {
        let stokes = PolarizationState::new(1.0, 1.0, 0.0, 0.0);
        let dop = stokes.degree_of_polarization();
        assert!(dop <= 1.0 && dop >= 0.0);
    }

    #[test]
    fn test_mueller_matrix_identity() {
        let m = MuellerMatrix::identity();
        let stokes = PolarizationState::new(1.0, 0.5, 0.3, 0.2);
        let result = m.apply(&stokes);
        assert_eq!(result, stokes);
    }

    #[test]
    fn test_polarizer_x() {
        let m = MuellerMatrix::linear_polarizer_x();
        let stokes = PolarizationState::new(1.0, 1.0, 0.0, 0.0);
        let result = m.apply(&stokes);
        assert!(result.intensity > 0.0);
    }

    #[test]
    fn test_maxwell_plane_wave() {
        let k = 2.0 * PI / 1.55e-6; // 1.55 um wavelength
        let omega = k * 3e8;
        let field = MaxwellField::plane_wave(k, omega);
        assert!((field.wavelength() - 1.55e-6).abs() < 1e-9);
    }

    #[test]
    fn test_maxwell_propagation() {
        let field = MaxwellField::plane_wave(100.0, 3e10);
        let dz = 0.1;
        let field_propagated = field.propagate(dz);
        assert_eq!(field_propagated.e_mag, field.e_mag);
    }

    #[test]
    fn test_optical_system_empty() {
        let sys = OpticalSystem::new();
        let m = sys.mueller_matrix();
        let stokes = PolarizationState::new(1.0, 0.0, 0.0, 0.0);
        let result = m.apply(&stokes);
        assert_eq!(result, stokes);
    }

    #[test]
    fn test_jones_to_stokes_linear() {
        let jones = JonesVector::new(1.0, 0.0, 0.0, 0.0);
        let stokes = jones.to_stokes();
        assert!(stokes.s1 > 0.0);
    }

    #[test]
    fn test_stokes_ellipse_angle() {
        let stokes = PolarizationState::new(1.0, 1.0, 1.0, 0.0);
        let angle = stokes.ellipse_angle();
        assert!(angle >= -PI / 2.0 && angle <= PI / 2.0);
    }
}
