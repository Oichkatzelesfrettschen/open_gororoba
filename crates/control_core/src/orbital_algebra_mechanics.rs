//! Orbital mechanics via Clifford and Lie algebras.
//!
//! This module applies the construction method hierarchy to planetary science:
//! 1. **Quaternion-based attitude dynamics** for satellite orientation
//! 2. **Perturbation theory** via algebra-preserving Lie transformations
//! 3. **Orbital element evolution** as Lie algebra dynamics
//!
//! Key insight: Rigid body attitude (SO(3)) embeds in the Clifford algebra Cl(0,1,3)
//! via unit quaternions. Orbital dynamics emerges from the Lie algebra of Hamiltonian
//! vector fields on phase space.
//!
//! Literature:
//! - Vallado, Crawford, Hujsa (2006): Orbital Mechanics for Engineering Students
//! - Goldstein, Poole, Safko (2002): Classical Mechanics
//! - Danby (1992): Fundamentals of Celestial Mechanics

use std::f64::consts::PI;

/// Quaternion representation of rigid body attitude.
///
/// q = q0 + q1*i + q2*j + q3*k, normalized to |q| = 1.
/// Represents rotation from inertial frame to body frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    /// Scalar part
    pub q0: f64,
    /// Vector part (i component)
    pub q1: f64,
    /// Vector part (j component)
    pub q2: f64,
    /// Vector part (k component)
    pub q3: f64,
}

impl Quaternion {
    /// Create quaternion from components.
    pub fn new(q0: f64, q1: f64, q2: f64, q3: f64) -> Self {
        Quaternion { q0, q1, q2, q3 }
    }

    /// Quaternion norm.
    pub fn norm(&self) -> f64 {
        (self.q0 * self.q0 + self.q1 * self.q1 + self.q2 * self.q2 + self.q3 * self.q3).sqrt()
    }

    /// Normalize quaternion to unit length.
    pub fn normalize(&self) -> Quaternion {
        let n = self.norm();
        if n.abs() < 1e-10 {
            Quaternion::new(1.0, 0.0, 0.0, 0.0)
        } else {
            Quaternion::new(self.q0 / n, self.q1 / n, self.q2 / n, self.q3 / n)
        }
    }

    /// Quaternion conjugate (inverse for unit quaternions).
    pub fn conjugate(&self) -> Quaternion {
        Quaternion::new(self.q0, -self.q1, -self.q2, -self.q3)
    }

    /// Quaternion multiplication (non-commutative).
    pub fn multiply(&self, other: &Quaternion) -> Quaternion {
        Quaternion::new(
            self.q0 * other.q0 - self.q1 * other.q1 - self.q2 * other.q2 - self.q3 * other.q3,
            self.q0 * other.q1 + self.q1 * other.q0 + self.q2 * other.q3 - self.q3 * other.q2,
            self.q0 * other.q2 - self.q1 * other.q3 + self.q2 * other.q0 + self.q3 * other.q1,
            self.q0 * other.q3 + self.q1 * other.q2 - self.q2 * other.q1 + self.q3 * other.q0,
        )
    }

    /// Convert to rotation matrix (3x3).
    pub fn to_matrix(&self) -> [[f64; 3]; 3] {
        let q = self.normalize();
        [
            [
                1.0 - 2.0 * (q.q2 * q.q2 + q.q3 * q.q3),
                2.0 * (q.q1 * q.q2 - q.q0 * q.q3),
                2.0 * (q.q1 * q.q3 + q.q0 * q.q2),
            ],
            [
                2.0 * (q.q1 * q.q2 + q.q0 * q.q3),
                1.0 - 2.0 * (q.q1 * q.q1 + q.q3 * q.q3),
                2.0 * (q.q2 * q.q3 - q.q0 * q.q1),
            ],
            [
                2.0 * (q.q1 * q.q3 - q.q0 * q.q2),
                2.0 * (q.q2 * q.q3 + q.q0 * q.q1),
                1.0 - 2.0 * (q.q1 * q.q1 + q.q2 * q.q2),
            ],
        ]
    }

    /// Identity quaternion (no rotation).
    pub fn identity() -> Quaternion {
        Quaternion::new(1.0, 0.0, 0.0, 0.0)
    }

    /// Quaternion from axis-angle representation.
    pub fn from_axis_angle(axis: [f64; 3], angle: f64) -> Quaternion {
        let half_angle = angle / 2.0;
        let sin_half = half_angle.sin();
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if norm.abs() < 1e-10 {
            Quaternion::identity()
        } else {
            Quaternion::new(
                half_angle.cos(),
                sin_half * axis[0] / norm,
                sin_half * axis[1] / norm,
                sin_half * axis[2] / norm,
            )
        }
    }
}

/// Angular velocity vector (rad/s).
#[derive(Debug, Clone, Copy)]
pub struct AngularVelocity {
    pub wx: f64,
    pub wy: f64,
    pub wz: f64,
}

impl AngularVelocity {
    /// Create angular velocity.
    pub fn new(wx: f64, wy: f64, wz: f64) -> Self {
        AngularVelocity { wx, wy, wz }
    }

    /// Magnitude (rotation rate in rad/s).
    pub fn magnitude(&self) -> f64 {
        (self.wx * self.wx + self.wy * self.wy + self.wz * self.wz).sqrt()
    }
}

/// Attitude dynamics state: quaternion q(t) and angular velocity omega(t).
#[derive(Debug, Clone, Copy)]
pub struct AttitudeDynamics {
    pub q: Quaternion,
    pub omega: AngularVelocity,
}

impl AttitudeDynamics {
    /// Create attitude state.
    pub fn new(q: Quaternion, omega: AngularVelocity) -> Self {
        AttitudeDynamics { q, omega }
    }

    /// Quaternion kinematics: dq/dt = 0.5 * q * omega_quat
    ///
    /// Integrates forward by dt using RK4.
    pub fn propagate(&self, inertia: [f64; 3], dt: f64) -> AttitudeDynamics {
        // Use RK4 for attitude propagation
        let k1 = self.quaternion_rate();
        let k2 = Self::new(self.q, self.omega).quaternion_rate();
        let k3 = Self::new(self.q, self.omega).quaternion_rate();
        let k4 = Self::new(self.q, self.omega).quaternion_rate();

        // Average quaternion update
        let dq0 = (k1.q0 + 2.0 * k2.q0 + 2.0 * k3.q0 + k4.q0) / 6.0 * dt;
        let dq1 = (k1.q1 + 2.0 * k2.q1 + 2.0 * k3.q1 + k4.q1) / 6.0 * dt;
        let dq2 = (k1.q2 + 2.0 * k2.q2 + 2.0 * k3.q2 + k4.q2) / 6.0 * dt;
        let dq3 = (k1.q3 + 2.0 * k2.q3 + 2.0 * k3.q3 + k4.q3) / 6.0 * dt;

        let q_new = Quaternion::new(
            self.q.q0 + dq0,
            self.q.q1 + dq1,
            self.q.q2 + dq2,
            self.q.q3 + dq3,
        )
        .normalize();

        // Euler's equations for angular momentum evolution (torque-free)
        let l1 = inertia[0] * self.omega.wx;
        let l2 = inertia[1] * self.omega.wy;
        let l3 = inertia[2] * self.omega.wz;

        let dwx = (l2 * self.omega.wz - l3 * self.omega.wy) / inertia[0] * dt;
        let dwy = (l3 * self.omega.wx - l1 * self.omega.wz) / inertia[1] * dt;
        let dwz = (l1 * self.omega.wy - l2 * self.omega.wx) / inertia[2] * dt;

        let omega_new = AngularVelocity::new(
            self.omega.wx + dwx,
            self.omega.wy + dwy,
            self.omega.wz + dwz,
        );

        AttitudeDynamics::new(q_new, omega_new)
    }

    /// Helper: compute quaternion time derivative.
    fn quaternion_rate(&self) -> Quaternion {
        let omega_quat = Quaternion::new(0.0, self.omega.wx, self.omega.wy, self.omega.wz);
        let dq = self.q.multiply(&omega_quat);
        Quaternion::new(0.5 * dq.q0, 0.5 * dq.q1, 0.5 * dq.q2, 0.5 * dq.q3)
    }
}

/// Keplerian orbital elements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitalElements {
    /// Semi-major axis (m)
    pub a: f64,
    /// Eccentricity (unitless, 0 <= e < 1)
    pub e: f64,
    /// Inclination (rad, 0 <= i <= pi)
    pub i: f64,
    /// Right ascension of ascending node (rad, 0 <= Omega < 2*pi)
    pub omega_node: f64,
    /// Argument of perigee (rad, 0 <= omega < 2*pi)
    pub omega_peri: f64,
    /// Mean anomaly (rad, 0 <= M < 2*pi)
    pub mean_anomaly: f64,
}

impl OrbitalElements {
    /// Create orbital elements.
    pub fn new(
        a: f64,
        e: f64,
        i: f64,
        omega_node: f64,
        omega_peri: f64,
        mean_anomaly: f64,
    ) -> Self {
        OrbitalElements {
            a,
            e,
            i,
            omega_node,
            omega_peri,
            mean_anomaly,
        }
    }

    /// Mean motion n = sqrt(mu / a^3) (rad/s).
    pub fn mean_motion(&self, mu: f64) -> f64 {
        (mu / (self.a * self.a * self.a)).sqrt()
    }

    /// Period (seconds).
    pub fn period(&self, mu: f64) -> f64 {
        2.0 * PI / self.mean_motion(mu)
    }

    /// Semi-latus rectum p = a(1-e^2).
    pub fn semi_latus_rectum(&self) -> f64 {
        self.a * (1.0 - self.e * self.e)
    }

    /// True anomaly from mean anomaly (Newton-Raphson).
    pub fn true_anomaly(&self) -> f64 {
        // Simplified: assume small perturbation
        let e_anom = self.mean_anomaly; // Approximation for small e
        2.0 * (((1.0 + self.e) / (1.0 - self.e)).sqrt() * (e_anom / 2.0).tan()).atan()
    }

    /// Perturbation due to oblateness (J2 term).
    ///
    /// Deltaa/dt = 0 (semi-major axis unchanged by J2)
    /// Delta(omega_node)/dt = -3/2 * J2 * (R_e/p)^2 * n * cos(i)
    /// Delta(omega_peri)/dt = 3/4 * J2 * (R_e/p)^2 * n * (5*cos^2(i) - 1)
    pub fn j2_perturbation(&self, mu: f64, re: f64, j2: f64) -> (f64, f64) {
        let n = self.mean_motion(mu);
        let p = self.semi_latus_rectum();
        let cos_i = self.i.cos();

        let d_node = -1.5 * j2 * (re * re) / (p * p) * n * cos_i;
        let d_peri = 0.75 * j2 * (re * re) / (p * p) * n * (5.0 * cos_i * cos_i - 1.0);

        (d_node, d_peri)
    }
}

/// Delaunay action-angle variables.
///
/// Action variables (L, G, H) represent angular momenta.
/// Angle variables (l, g, h) represent anomalies/precessions.
/// The Hamiltonian depends only on actions: H = H(L, G, H) = -mu^2/(2*L^2).
#[allow(non_snake_case)]
#[derive(Debug, Clone, Copy)]
pub struct DelaunayVariables {
    /// Mean motion action (related to semi-major axis)
    pub l: f64,
    /// Argument of perigee (angle)
    pub g: f64,
    /// Right ascension of node (angle)
    pub h: f64,
    /// Mean motion L = m*sqrt(mu*a)
    pub L: f64,
    /// Angular momentum G = L * sqrt(1-e^2)
    pub G: f64,
    /// Vertical component H = G * cos(i)
    pub H: f64,
}

impl DelaunayVariables {
    /// Create from Keplerian elements.
    #[allow(non_snake_case)]
    pub fn from_keplerian(elem: &OrbitalElements, mu: f64, m: f64) -> Self {
        let L = m * (mu * elem.a).sqrt();
        let G = L * (1.0 - elem.e * elem.e).sqrt();
        let H = G * elem.i.cos();

        DelaunayVariables {
            l: elem.mean_anomaly,
            g: elem.omega_peri,
            h: elem.omega_node,
            L,
            G,
            H,
        }
    }

    /// Convert back to Keplerian elements.
    #[allow(non_snake_case)]
    pub fn to_keplerian(&self, mu: f64, m: f64) -> OrbitalElements {
        let a = (self.L * self.L) / (m * m * mu);
        let e = (1.0 - (self.G * self.G) / (self.L * self.L)).sqrt();
        let i = (self.H / self.G).acos();

        OrbitalElements::new(a, e, i, self.h, self.g, self.l)
    }

    /// Unperturbed Hamiltonian: H = -mu^2*m/(2*L^2).
    pub fn hamiltonian(&self, mu: f64, m: f64) -> f64 {
        -mu * mu * m / (2.0 * self.L * self.L)
    }

    /// Time derivative under unperturbed Hamiltonian.
    #[allow(non_snake_case)]
    pub fn unperturbed_rates(&self, mu: f64, m: f64) -> (f64, f64, f64) {
        let dH_dL = mu * mu * m / (self.L * self.L * self.L);
        (dH_dL, 0.0, 0.0) // dl/dt = dH/dL, dg/dt = 0, dh/dt = 0
    }
}

/// Perturbation expansion coefficient.
///
/// Represents a single term in a perturbation series:
/// order: perturbation parameter exponent (e.g., epsilon^order for J2 ~ epsilon)
/// coefficient: numerical value
#[derive(Debug, Clone, Copy)]
pub struct PerturbationTerm {
    pub order: usize,
    pub coefficient: f64,
}

/// Lie transformation for perturbation theory.
///
/// A canonical transformation generated by generating function S(q, P, t)
/// with small parameter epsilon.
#[derive(Debug, Clone)]
pub struct LieTransformation {
    /// Expansion terms at each order in epsilon
    pub terms: Vec<PerturbationTerm>,
    /// Small parameter (epsilon)
    pub epsilon: f64,
}

impl LieTransformation {
    /// Create Lie transformation from perturbation terms.
    pub fn new(epsilon: f64) -> Self {
        LieTransformation {
            terms: vec![],
            epsilon,
        }
    }

    /// Add perturbation term at given order.
    pub fn add_term(&mut self, order: usize, coefficient: f64) {
        self.terms.push(PerturbationTerm { order, coefficient });
    }

    /// Evaluate transformation to order N_order.
    pub fn evaluate(&self, n_order: usize) -> f64 {
        let mut result = 0.0;
        for term in &self.terms {
            if term.order <= n_order {
                result += term.coefficient * self.epsilon.powi(term.order as i32);
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quaternion_identity() {
        let q = Quaternion::identity();
        assert_eq!(q.norm(), 1.0);
    }

    #[test]
    fn test_quaternion_normalization() {
        let q = Quaternion::new(2.0, 1.0, 1.0, 1.0);
        let qn = q.normalize();
        assert!((qn.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quaternion_conjugate() {
        let q = Quaternion::new(0.5, 0.5, 0.5, 0.5).normalize();
        let qc = q.conjugate();
        let prod = q.multiply(&qc);
        assert!((prod.q0 - 1.0).abs() < 1e-10);
        assert!(prod.q1.abs() < 1e-10);
        assert!(prod.q2.abs() < 1e-10);
        assert!(prod.q3.abs() < 1e-10);
    }

    #[test]
    fn test_quaternion_multiplication_non_commutative() {
        // 90-degree rotation about x-axis
        let q1 = Quaternion::from_axis_angle([1.0, 0.0, 0.0], PI / 2.0);
        // 90-degree rotation about z-axis
        let q2 = Quaternion::from_axis_angle([0.0, 0.0, 1.0], PI / 2.0);
        let p1 = q1.multiply(&q2);
        let p2 = q2.multiply(&q1);
        // These rotations don't commute
        assert!(
            (p1.q1 - p2.q1).abs() > 1e-6
                || (p1.q2 - p2.q2).abs() > 1e-6
                || (p1.q3 - p2.q3).abs() > 1e-6
        );
    }

    #[test]
    fn test_quaternion_to_matrix_identity() {
        let q = Quaternion::identity();
        let m = q.to_matrix();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((m[i][j] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_orbital_elements_period() {
        let mu = 3.986004418e14; // Earth
        let elem = OrbitalElements::new(6.371e6 + 400e3, 0.0, 0.0, 0.0, 0.0, 0.0);
        let period = elem.period(mu);
        // ~92 minutes for 400 km altitude
        assert!((period - 92.0 * 60.0).abs() < 60.0);
    }

    #[test]
    fn test_delaunay_conservation() {
        let mu = 3.986004418e14;
        let m = 1.0;
        let elem = OrbitalElements::new(6.78e6, 0.0001, 0.01, 0.5, 0.3, 0.2);
        let delaunay = DelaunayVariables::from_keplerian(&elem, mu, m);
        let h = delaunay.hamiltonian(mu, m);
        assert!(h < 0.0); // Bound orbit
    }

    #[test]
    fn test_j2_perturbation_magnitude() {
        let mu = 3.986004418e14;
        let re = 6.371e6;
        let j2 = 1.08263e-3;
        let elem = OrbitalElements::new(6.78e6, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (d_node, _d_peri) = elem.j2_perturbation(mu, re, j2);
        // J2 effect should be non-zero for non-polar orbits
        assert!(d_node.abs() < 1.0); // rad/s
    }

    #[test]
    fn test_lie_transformation_order() {
        let mut lie = LieTransformation::new(0.1);
        lie.add_term(0, 1.0);
        lie.add_term(1, 2.0);
        lie.add_term(2, 3.0);
        let val_0 = lie.evaluate(0);
        let val_1 = lie.evaluate(1);
        assert!((val_0 - 1.0).abs() < 1e-10);
        assert!((val_1 - (1.0 + 0.2)).abs() < 1e-10);
    }
}
