use nalgebra::{Vector3, Matrix3};
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
}
