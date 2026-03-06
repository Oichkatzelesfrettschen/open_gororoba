use nalgebra::{Vector3, Matrix3};

/// State of a single body in the N-body system.
#[derive(Debug, Clone)]
pub struct BodyState {
    pub id: i32,
    pub mass: f64, // GM in km^3/s^2
    pub pos: Vector3<f64>, // km
    pub vel: Vector3<f64>, // km/s
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

    /// Computes acceleration for all bodies including EIH and Pathion terms.
    pub fn compute_accelerations(&self) -> Vec<Vector3<f64>> {
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
                let dist_sq = r_ji.norm_squared();
                let dist = dist_sq.sqrt();

                // Newtonian part
                let a_newton = r_ji * (body_j.mass / (dist_sq * dist));

                // 1st order EIH correction (simplified version)
                // a_gr = a_newton * (1/c^2) * [ (4*GM/r) - v^2 + ... ]
                let phi_i = body_j.mass / dist; // Potential at i due to j
                let v_i_sq = v_i.norm_squared();
                let v_j_sq = body_j.vel.norm_squared();
                let vi_dot_vj = v_i.dot(&body_j.vel);

                // PPN parameters gamma=1, beta=1 for GR
                let gr_corr = (1.0 / c2) * (4.0 * phi_i - v_i_sq - 2.0 * v_j_sq + 4.0 * vi_dot_vj);
                let a_gr = a_newton * gr_corr;

                a_i += a_newton + a_gr;
            }

            // Pathion Perturbation: alpha * Var * r
            // We use the displacement from the system barycenter (assumed near origin)
            let a_pathion = self.alpha_pathion * (self.pathion_variance * p_i);

            *accel_i = a_i + a_pathion;
        }

        accels
    }

    /// 8th-order Adams-Bashforth-Moulton predictor-corrector step (STUB for now, using RK4)
    /// Real implementation will follow Phase 2 constraint.
    pub fn step(&mut self, dt: f64) {
        // RK4 implementation
        let initial_bodies = self.bodies.clone();
        
        // k1
        let k1_v = self.compute_accelerations();
        let k1_p: Vec<_> = self.bodies.iter().map(|b| b.vel).collect();
        
        // k2
        for i in 0..self.bodies.len() {
            self.bodies[i].pos = initial_bodies[i].pos + k1_p[i] * (dt / 2.0);
            self.bodies[i].vel = initial_bodies[i].vel + k1_v[i] * (dt / 2.0);
        }
        let k2_v = self.compute_accelerations();
        let k2_p: Vec<_> = self.bodies.iter().map(|b| b.vel).collect();
        
        // k3
        for i in 0..self.bodies.len() {
            self.bodies[i].pos = initial_bodies[i].pos + k2_p[i] * (dt / 2.0);
            self.bodies[i].vel = initial_bodies[i].vel + k2_v[i] * (dt / 2.0);
        }
        let k3_v = self.compute_accelerations();
        let k3_p: Vec<_> = self.bodies.iter().map(|b| b.vel).collect();
        
        // k4
        for i in 0..self.bodies.len() {
            self.bodies[i].pos = initial_bodies[i].pos + k3_p[i] * dt;
            self.bodies[i].vel = initial_bodies[i].vel + k3_v[i] * dt;
        }
        let k4_v = self.compute_accelerations();
        let k4_p: Vec<_> = self.bodies.iter().map(|b| b.vel).collect();
        
        // Combine
        for i in 0..self.bodies.len() {
            self.bodies[i].pos = initial_bodies[i].pos + (k1_p[i] + 2.0*k2_p[i] + 2.0*k3_p[i] + k4_p[i]) * (dt / 6.0);
            self.bodies[i].vel = initial_bodies[i].vel + (k1_v[i] + 2.0*k2_v[i] + 2.0*k3_v[i] + k4_v[i]) * (dt / 6.0);
        }
    }
}
