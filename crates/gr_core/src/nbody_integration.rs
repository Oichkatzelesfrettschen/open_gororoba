use nalgebra::{Matrix3, Vector3};
use num_complex::Complex;

/// State of a single body in the N-body system with complex-time support.
#[derive(Debug, Clone)]
pub struct BodyState {
    pub id: i32,
    pub mass: f64, // GM in km^3/s^2
    pub pos: Vector3<Complex<f64>>, // km
    pub vel: Vector3<Complex<f64>>, // km/s
}

/// A collection of bodies forming a physical system.
pub struct NBodySystem {
    pub bodies: Vec<BodyState>,
    pub alpha_pathion: f64,
    pub pathion_variance: Matrix3<f64>,
}

impl NBodySystem {
    pub fn new(alpha_pathion: f64, pathion_variance: Matrix3<f64>) -> Self {
        Self {
            bodies: Vec::new(),
            alpha_pathion,
            pathion_variance,
        }
    }

    /// Computes acceleration for all bodies including EIH and Pathion terms in complex time.
    pub fn compute_accelerations(&self) -> Vec<Vector3<Complex<f64>>> {
        let n = self.bodies.len();
        let mut accels = vec![Vector3::zeros(); n];
        let c = 299792.458; // Speed of light in km/s
        let c2 = c * c;

        for (i, accel_i) in accels.iter_mut().enumerate() {
            let p_i = &self.bodies[i].pos;
            let v_i = &self.bodies[i].vel;
            let mut a_i = Vector3::zeros();

            for (j, body_j) in self.bodies.iter().enumerate() {
                if i == j { continue; }
                let r_ji = body_j.pos - p_i;
                // Complex norm squared: ||r||^2 = r_x^2 + r_y^2 + r_z^2 (holomorphic)
                let dist_sq = r_ji.dot(&r_ji); 
                let dist = dist_sq.sqrt();

                // Newtonian part (complexified)
                let a_newton = r_ji * (Complex::from(body_j.mass) / (dist_sq * dist));

                // 1st order EIH correction
                let phi_i = Complex::from(body_j.mass) / dist;
                let v_i_sq = v_i.dot(v_i);
                let v_j_sq = body_j.vel.dot(&body_j.vel);
                let vi_dot_vj = v_i.dot(&body_j.vel);

                let gr_corr = (Complex::from(1.0 / c2)) * (Complex::from(4.0) * phi_i - v_i_sq - Complex::from(2.0) * v_j_sq + Complex::from(4.0) * vi_dot_vj);
                let a_gr = a_newton * gr_corr;

                a_i += a_newton + a_gr;
            }

            // Pathion Perturbation (complexified)
            let mut a_pathion = Vector3::zeros();
            for r_idx in 0..3 {
                for c_idx in 0..3 {
                    a_pathion[r_idx] += Complex::from(self.pathion_variance[(r_idx, c_idx)]) * p_i[c_idx];
                }
            }
            a_pathion *= Complex::from(self.alpha_pathion);

            *accel_i = a_i + a_pathion;
        }

        accels
    }

    /// Step the system forward in 2D Complex Time tau = t + i*epsilon.
    pub fn step(&mut self, d_tau: Complex<f64>) {
        let initial_bodies = self.bodies.clone();

        // k1
        let k1_v = self.compute_accelerations();
        let k1_p: Vec<_> = self.bodies.iter().map(|b| b.vel).collect();

        // k2
        for i in 0..self.bodies.len() {
            self.bodies[i].pos = initial_bodies[i].pos + k1_p[i] * (d_tau / 2.0);
            self.bodies[i].vel = initial_bodies[i].vel + k1_v[i] * (d_tau / 2.0);
        }
        let k2_v = self.compute_accelerations();
        let k2_p: Vec<_> = self.bodies.iter().map(|b| b.vel).collect();

        // k3
        for i in 0..self.bodies.len() {
            self.bodies[i].pos = initial_bodies[i].pos + k2_p[i] * (d_tau / 2.0);
            self.bodies[i].vel = initial_bodies[i].vel + k2_v[i] * (d_tau / 2.0);
        }
        let k3_v = self.compute_accelerations();
        let k3_p: Vec<_> = self.bodies.iter().map(|b| b.vel).collect();

        // k4
        for i in 0..self.bodies.len() {
            self.bodies[i].pos = initial_bodies[i].pos + k3_p[i] * d_tau;
            self.bodies[i].vel = initial_bodies[i].vel + k3_v[i] * d_tau;
        }
        let k4_v = self.compute_accelerations();
        let k4_p: Vec<_> = self.bodies.iter().map(|b| b.vel).collect();

        // Combine
        let two = Complex::from(2.0);
        for i in 0..self.bodies.len() {
            self.bodies[i].pos = initial_bodies[i].pos + (k1_p[i] + k2_p[i] * two + k3_p[i] * two + k4_p[i]) * (d_tau / 6.0);
            self.bodies[i].vel = initial_bodies[i].vel + (k1_v[i] + k2_v[i] * two + k3_v[i] * two + k4_v[i]) * (d_tau / 6.0);
        }
    }

    /// Wick rotation dispatcher: smoothly rotate from real to imaginary time.
    ///
    /// Parameterized by angle `theta` in the complex time plane:
    /// - `theta = 0`: pure real time (Lorentzian, oscillatory orbits)
    /// - `theta = pi/2`: pure imaginary time (Euclidean, exponential decay)
    /// - Intermediate values: smooth interpolation between classical and quantum
    ///
    /// The time step is `d_tau = dt * exp(i * theta)`, so:
    /// - `d_tau.re = dt * cos(theta)` (real component)
    /// - `d_tau.im = dt * sin(theta)` (imaginary component)
    ///
    /// Returns a `WickEvolutionResult` with the trajectory and energy diagnostics.
    pub fn wick_evolve(
        &mut self,
        dt: f64,
        theta: f64,
        n_steps: usize,
    ) -> WickEvolutionResult {
        let d_tau = Complex::new(dt * theta.cos(), dt * theta.sin());
        let mut trajectory = Vec::with_capacity(n_steps + 1);
        let mut energies = Vec::with_capacity(n_steps + 1);

        // Record initial state
        trajectory.push(self.snapshot_real_positions());
        energies.push(self.total_energy());

        for _ in 0..n_steps {
            self.step(d_tau);
            trajectory.push(self.snapshot_real_positions());
            energies.push(self.total_energy());
        }

        WickEvolutionResult {
            theta,
            dt,
            n_steps,
            trajectory,
            energies,
        }
    }

    /// Extract the real parts of all body positions (for diagnostics).
    fn snapshot_real_positions(&self) -> Vec<Vector3<f64>> {
        self.bodies
            .iter()
            .map(|b| Vector3::new(b.pos.x.re, b.pos.y.re, b.pos.z.re))
            .collect()
    }

    /// Compute total energy (kinetic + potential) using real parts.
    ///
    /// For complex-time evolution, returns the real part of the complexified energy.
    fn total_energy(&self) -> f64 {
        let mut ke = 0.0;
        let mut pe = 0.0;

        for body in &self.bodies {
            let v_sq = body.vel.dot(&body.vel);
            ke += 0.5 * body.mass * v_sq.re;
        }

        for i in 0..self.bodies.len() {
            for j in (i + 1)..self.bodies.len() {
                let r_ij = self.bodies[j].pos - self.bodies[i].pos;
                let dist = r_ij.dot(&r_ij).sqrt();
                if dist.norm() > 1e-30 {
                    pe -= (self.bodies[i].mass * self.bodies[j].mass / dist).re;
                }
            }
        }

        ke + pe
    }

    /// Singularity avoidance test: evolve a test particle toward a central mass
    /// in complex time and check whether the trajectory bounces.
    ///
    /// Sets up a 2-body system (central mass + test particle) and evolves
    /// through the Schwarzschild radius. In real time, the particle falls
    /// to the singularity. In imaginary time, the non-associative Pathion
    /// perturbation may cause a bounce.
    ///
    /// Returns the minimum distance achieved and whether a bounce occurred.
    pub fn singularity_avoidance_test(
        central_mass: f64,
        initial_r: f64,
        initial_v_radial: f64,
        alpha_pathion: f64,
        theta: f64,
        dt: f64,
        n_steps: usize,
    ) -> SingularityAvoidanceResult {
        let pathion_var = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, 1.0));
        let mut system = NBodySystem::new(alpha_pathion, pathion_var);

        // Central mass at origin
        system.bodies.push(BodyState {
            id: 0,
            mass: central_mass,
            pos: Vector3::new(Complex::from(0.0), Complex::from(0.0), Complex::from(0.0)),
            vel: Vector3::new(Complex::from(0.0), Complex::from(0.0), Complex::from(0.0)),
        });

        // Test particle at (initial_r, 0, 0) with inward radial velocity
        system.bodies.push(BodyState {
            id: 1,
            mass: 1.0, // test particle
            pos: Vector3::new(Complex::from(initial_r), Complex::from(0.0), Complex::from(0.0)),
            vel: Vector3::new(Complex::from(initial_v_radial), Complex::from(0.0), Complex::from(0.0)),
        });

        let result = system.wick_evolve(dt, theta, n_steps);

        // Find minimum distance to central mass
        let mut min_r = f64::INFINITY;
        let mut bounce_detected = false;
        let mut prev_r = initial_r;

        for snapshot in &result.trajectory {
            if snapshot.len() < 2 {
                continue;
            }
            let r = (snapshot[1] - snapshot[0]).norm();
            if r < min_r {
                min_r = r;
            }
            // Bounce = distance starts increasing after decreasing
            if r > prev_r * 1.01 && prev_r < initial_r * 0.5 {
                bounce_detected = true;
            }
            prev_r = r;
        }

        SingularityAvoidanceResult {
            theta,
            alpha_pathion,
            initial_r,
            min_r,
            bounce_detected,
            final_energy: *result.energies.last().unwrap_or(&0.0),
            energy_drift: result.energies.last().unwrap_or(&0.0)
                - result.energies.first().unwrap_or(&0.0),
        }
    }

    /// Compute the Pathion shadow transition radius.
    ///
    /// The shadow radius is the distance at which the Pathion perturbation
    /// magnitude equals the Newtonian gravitational acceleration. Below this
    /// radius, gravity dominates; above it, the non-associative Pathion torque
    /// dominates. At the boundary, the effective force switches sign -- this
    /// is the mathematical analog of the 32D Pathion shadow boundary in
    /// impact-parameter space.
    ///
    /// For a test particle at distance `r` from mass `M`:
    ///   a_grav ~ GM/r^2
    ///   a_pathion ~ alpha * sigma_max * r  (harmonic-like, grows with r)
    ///
    /// Setting equal: GM/r^2 = alpha * sigma_max * r
    ///   => r_shadow = (GM / (alpha * sigma_max))^(1/3)
    ///
    /// Returns `None` if alpha_pathion is zero (no shadow boundary exists).
    pub fn pathion_shadow_radius(&self, central_mass: f64) -> Option<f64> {
        if self.alpha_pathion.abs() < 1e-30 {
            return None;
        }

        // Maximum eigenvalue of pathion_variance gives the dominant perturbation axis
        let sigma_max = pathion_variance_max_eigenvalue(&self.pathion_variance);
        if sigma_max < 1e-30 {
            return None;
        }

        let ratio = central_mass / (self.alpha_pathion * sigma_max);
        if ratio <= 0.0 {
            return None;
        }

        Some(ratio.cbrt())
    }

    /// Adaptive Wick evolution: dynamically adjusts the Wick angle based on
    /// proximity to the Pathion shadow boundary.
    ///
    /// When the test particle is far from the shadow radius, evolution is
    /// purely real-time (classical orbits). As it approaches the shadow
    /// boundary, the Wick angle smoothly increases toward pi/2 (Euclidean),
    /// enabling quantum tunneling through the classical barrier.
    ///
    /// The transition is controlled by a sigmoid:
    ///   theta(r) = theta_max * sigmoid((r_shadow - r) / r_shadow * steepness)
    ///
    /// where `steepness` controls how sharp the transition is.
    pub fn adaptive_wick_evolve(
        &mut self,
        dt: f64,
        theta_max: f64,
        n_steps: usize,
        central_mass: f64,
        steepness: f64,
    ) -> AdaptiveWickResult {
        let r_shadow = self.pathion_shadow_radius(central_mass);
        let mut trajectory = Vec::with_capacity(n_steps + 1);
        let mut energies = Vec::with_capacity(n_steps + 1);
        let mut thetas = Vec::with_capacity(n_steps + 1);

        trajectory.push(self.snapshot_real_positions());
        energies.push(self.total_energy());
        thetas.push(0.0_f64);

        for _ in 0..n_steps {
            // Compute current distance from origin (central mass assumed at body 0)
            let theta = if self.bodies.len() >= 2 {
                let r_vec = Vector3::new(
                    (self.bodies[1].pos - self.bodies[0].pos).x.re,
                    (self.bodies[1].pos - self.bodies[0].pos).y.re,
                    (self.bodies[1].pos - self.bodies[0].pos).z.re,
                );
                let r = r_vec.norm();

                match r_shadow {
                    Some(rs) => {
                        // Sigmoid transition: theta increases as r approaches r_shadow
                        let x = (rs - r) / rs * steepness;
                        theta_max / (1.0 + (-x).exp())
                    }
                    None => 0.0, // No shadow boundary => pure real time
                }
            } else {
                0.0
            };

            let d_tau = Complex::new(dt * theta.cos(), dt * theta.sin());
            self.step(d_tau);

            trajectory.push(self.snapshot_real_positions());
            energies.push(self.total_energy());
            thetas.push(theta);
        }

        AdaptiveWickResult {
            shadow_radius: r_shadow,
            theta_max,
            steepness,
            trajectory,
            energies,
            thetas,
        }
    }
}

/// Compute the maximum eigenvalue of a 3x3 symmetric matrix.
///
/// Uses the characteristic polynomial approach for 3x3 matrices.
fn pathion_variance_max_eigenvalue(m: &Matrix3<f64>) -> f64 {
    let eigenvalues = m.symmetric_eigenvalues();
    eigenvalues.iter().cloned().fold(0.0_f64, f64::max)
}

/// Result of adaptive Wick evolution with dynamic angle.
#[derive(Debug)]
pub struct AdaptiveWickResult {
    /// Pathion shadow transition radius (None if alpha=0).
    pub shadow_radius: Option<f64>,
    /// Maximum Wick angle (at shadow boundary).
    pub theta_max: f64,
    /// Sigmoid steepness parameter.
    pub steepness: f64,
    /// Trajectory: real-part positions at each step.
    pub trajectory: Vec<Vec<Vector3<f64>>>,
    /// Total energy (real part) at each step.
    pub energies: Vec<f64>,
    /// Wick angle at each step.
    pub thetas: Vec<f64>,
}

/// Result of Wick rotation evolution.
#[derive(Debug)]
pub struct WickEvolutionResult {
    /// Wick angle (0 = real time, pi/2 = imaginary time).
    pub theta: f64,
    /// Time step magnitude.
    pub dt: f64,
    /// Number of steps taken.
    pub n_steps: usize,
    /// Trajectory: real-part positions at each step.
    pub trajectory: Vec<Vec<Vector3<f64>>>,
    /// Total energy (real part) at each step.
    pub energies: Vec<f64>,
}

/// Result of singularity avoidance test.
#[derive(Debug, Clone)]
pub struct SingularityAvoidanceResult {
    /// Wick angle used.
    pub theta: f64,
    /// Pathion coupling strength.
    pub alpha_pathion: f64,
    /// Initial radial distance.
    pub initial_r: f64,
    /// Minimum radial distance achieved.
    pub min_r: f64,
    /// Whether a bounce (distance increase after deep infall) was detected.
    pub bounce_detected: bool,
    /// Final total energy.
    pub final_energy: f64,
    /// Energy drift (final - initial).
    pub energy_drift: f64,
}

/// WKB tunneling amplitude for a 1D potential barrier.
///
/// Computes `T = exp(-2 * integral sqrt(2m(V(r) - E)) dr)` over the
/// classically forbidden region where `V(r) > E`.
///
/// This is equivalent to the Euclidean action for the "bounce" solution
/// in imaginary time -- a trajectory that smoothly connects the two
/// sides of the barrier without oscillation.
///
/// # Arguments
/// * `potential` - Potential function V(r)
/// * `energy` - Particle energy E
/// * `mass` - Particle mass
/// * `r_min`, `r_max` - Integration range (should bracket the barrier)
/// * `n_points` - Number of quadrature points
pub fn wkb_tunneling_amplitude(
    potential: &dyn Fn(f64) -> f64,
    energy: f64,
    mass: f64,
    r_min: f64,
    r_max: f64,
    n_points: usize,
) -> WkbResult {
    let dr = (r_max - r_min) / n_points as f64;
    let mut integral = 0.0;
    let mut turning_points = Vec::new();
    let mut prev_forbidden = potential(r_min) > energy;

    for k in 0..=n_points {
        let r = r_min + k as f64 * dr;
        let v = potential(r);
        let forbidden = v > energy;

        // Detect turning points (classical/forbidden boundary crossings)
        if k > 0 && forbidden != prev_forbidden {
            turning_points.push(r);
        }
        prev_forbidden = forbidden;

        if forbidden {
            let kappa_sq = 2.0 * mass * (v - energy);
            integral += kappa_sq.sqrt() * dr;
        }
    }

    let transmission = (-2.0 * integral).exp();

    WkbResult {
        transmission,
        euclidean_action: integral,
        turning_points,
        barrier_height: {
            let mut max_v = f64::NEG_INFINITY;
            for k in 0..=n_points {
                let r = r_min + k as f64 * dr;
                let v = potential(r);
                if v > max_v {
                    max_v = v;
                }
            }
            max_v - energy
        },
    }
}

/// Result of WKB tunneling calculation.
#[derive(Debug, Clone)]
pub struct WkbResult {
    /// Transmission coefficient (0 to 1).
    pub transmission: f64,
    /// Euclidean action (integral of kappa over forbidden region).
    pub euclidean_action: f64,
    /// Classical turning points (V(r) = E crossings).
    pub turning_points: Vec<f64>,
    /// Barrier height above the particle energy.
    pub barrier_height: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn two_body_system(alpha: f64) -> NBodySystem {
        let pathion_var = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, 1.0));
        let mut system = NBodySystem::new(alpha, pathion_var);
        // Sun-like central mass
        system.bodies.push(BodyState {
            id: 0,
            mass: 1.327e11, // GM_sun in km^3/s^2
            pos: Vector3::zeros(),
            vel: Vector3::zeros(),
        });
        // Earth-like orbiter at 1 AU
        let r_au: f64 = 1.496e8; // km
        let v_orb = (1.327e11_f64 / r_au).sqrt(); // circular orbit velocity
        system.bodies.push(BodyState {
            id: 1,
            mass: 398600.0, // GM_earth
            pos: Vector3::new(Complex::from(r_au), Complex::from(0.0), Complex::from(0.0)),
            vel: Vector3::new(Complex::from(0.0), Complex::from(v_orb), Complex::from(0.0)),
        });
        system
    }

    #[test]
    fn test_wick_evolve_real_time() {
        let mut system = two_body_system(0.0);
        let result = system.wick_evolve(3600.0, 0.0, 100);
        assert_eq!(result.trajectory.len(), 101);
        assert_eq!(result.energies.len(), 101);
        // Energy should be approximately conserved for real-time evolution
        let e0 = result.energies[0];
        let ef = *result.energies.last().unwrap();
        let drift = (ef - e0).abs() / e0.abs();
        assert!(drift < 0.01, "energy drift {drift} too large for real-time");
    }

    #[test]
    fn test_wick_evolve_imaginary_time() {
        let mut system = two_body_system(0.0);
        let result = system.wick_evolve(3600.0, PI / 2.0, 100);
        assert_eq!(result.trajectory.len(), 101);
        // In imaginary time, energy is NOT conserved (this is expected)
    }

    #[test]
    fn test_wick_evolve_mixed_angle() {
        let mut system = two_body_system(0.0);
        let result = system.wick_evolve(3600.0, PI / 4.0, 50);
        assert_eq!(result.trajectory.len(), 51);
    }

    #[test]
    fn test_wkb_square_barrier() {
        // Square barrier: V = V0 for a < r < b, 0 otherwise
        let v0 = 10.0;
        let a = 1.0;
        let b = 2.0;
        let potential = |r: f64| -> f64 {
            if r >= a && r <= b { v0 } else { 0.0 }
        };
        let energy = 5.0;
        let mass = 1.0;
        let result = wkb_tunneling_amplitude(&potential, energy, mass, 0.0, 3.0, 1000);

        // Analytical: T = exp(-2 * sqrt(2*m*(V0-E)) * (b-a))
        let kappa = (2.0 * mass * (v0 - energy)).sqrt();
        let expected = (-2.0 * kappa * (b - a)).exp();

        assert!(
            (result.transmission - expected).abs() < 0.01,
            "WKB T={:.6}, expected {expected:.6}",
            result.transmission
        );
        assert_eq!(result.turning_points.len(), 2);
        assert!(result.barrier_height > 0.0);
    }

    #[test]
    fn test_wkb_no_barrier() {
        // No barrier: V = 0 everywhere, E = 5
        let result = wkb_tunneling_amplitude(&|_| 0.0, 5.0, 1.0, 0.0, 10.0, 100);
        assert!(
            (result.transmission - 1.0).abs() < 1e-10,
            "no barrier should give T=1, got {}",
            result.transmission
        );
    }

    #[test]
    fn test_singularity_avoidance_real_time() {
        let result = NBodySystem::singularity_avoidance_test(
            1.327e11, // GM_sun
            1.0e6,    // 1000 km initial distance
            -100.0,   // inward velocity
            0.0,      // no pathion
            0.0,      // real time
            1.0,      // 1 second steps
            1000,
        );
        // Without pathion perturbation, should fall inward
        assert!(result.min_r < result.initial_r);
    }

    #[test]
    fn test_singularity_avoidance_with_pathion() {
        let result = NBodySystem::singularity_avoidance_test(
            1.327e11,
            1.0e6,
            -100.0,
            1e-6,     // small pathion coupling
            PI / 4.0, // mixed Wick angle
            1.0,
            1000,
        );
        // With pathion + Wick rotation, behavior may differ
        // Just verify it doesn't panic and produces finite results
        assert!(result.min_r.is_finite());
        assert!(result.final_energy.is_finite());
    }

    #[test]
    fn test_pathion_shadow_radius_zero_alpha() {
        let system = two_body_system(0.0);
        assert!(system.pathion_shadow_radius(1.327e11).is_none());
    }

    #[test]
    fn test_pathion_shadow_radius_nonzero() {
        let pathion_var = Matrix3::from_diagonal(&Vector3::new(1.0, 0.5, 0.3));
        let system = NBodySystem::new(1e-10, pathion_var);
        let r_shadow = system.pathion_shadow_radius(1.327e11);
        assert!(r_shadow.is_some());
        let rs = r_shadow.unwrap();
        // r_shadow = (GM / (alpha * sigma_max))^(1/3)
        // = (1.327e11 / (1e-10 * 1.0))^(1/3) = (1.327e21)^(1/3) ~ 1.099e7 km
        assert!(rs > 1e6 && rs < 1e8, "r_shadow={rs} out of expected range");
    }

    #[test]
    fn test_adaptive_wick_evolve() {
        let pathion_var = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, 1.0));
        let mut system = NBodySystem::new(1e-12, pathion_var);
        system.bodies.push(BodyState {
            id: 0,
            mass: 1.327e11,
            pos: Vector3::zeros(),
            vel: Vector3::zeros(),
        });
        let r_au: f64 = 1.496e8;
        let v_orb = (1.327e11_f64 / r_au).sqrt();
        system.bodies.push(BodyState {
            id: 1,
            mass: 398600.0,
            pos: Vector3::new(Complex::from(r_au), Complex::from(0.0), Complex::from(0.0)),
            vel: Vector3::new(Complex::from(0.0), Complex::from(v_orb), Complex::from(0.0)),
        });

        let result = system.adaptive_wick_evolve(
            3600.0,       // dt
            PI / 2.0,     // theta_max
            50,           // n_steps
            1.327e11,     // central_mass
            5.0,          // steepness
        );

        assert_eq!(result.trajectory.len(), 51);
        assert_eq!(result.thetas.len(), 51);
        assert!(result.shadow_radius.is_some());
        // All thetas should be finite and in [0, theta_max]
        for &t in &result.thetas {
            assert!(t.is_finite());
            assert!((0.0..=PI / 2.0 + 1e-10).contains(&t), "theta={t} out of range");
        }
    }

    #[test]
    fn test_pathion_variance_max_eigenvalue() {
        let m = Matrix3::from_diagonal(&Vector3::new(3.0, 1.0, 2.0));
        let ev = pathion_variance_max_eigenvalue(&m);
        assert!((ev - 3.0).abs() < 1e-10, "max eigenvalue should be 3.0, got {ev}");
    }
}
