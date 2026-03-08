//! 8th-order Adams-Bashforth-Moulton predictor-corrector integrator.
//!
//! This multistep method uses 8 previous acceleration evaluations to
//! predict the next state, then corrects with the Adams-Moulton formula.
//! After the RK4 startup phase (first 8 steps), each subsequent step
//! requires only 2 acceleration evaluations (predict + correct), making
//! it significantly more efficient than RK4 for long-duration integrations.
//!
//! # References
//! - Hairer, Norsett, Wanner (1993): Solving Ordinary Differential Equations I
//! - Press et al. (2007): Numerical Recipes, Ch. 17
//! - Montenbruck & Gill (2000): Satellite Orbits, Ch. 4

use nalgebra::Vector3;

/// Adams-Bashforth 8-step predictor coefficients.
/// These are the coefficients for the 8th-order AB formula:
/// y_{n+1} = y_n + h * sum(b_i * f_{n-i}, i=0..7)
const AB8_COEFFS: [f64; 8] = [
    434241.0 / 120960.0,
    -1152169.0 / 120960.0,
    2183877.0 / 120960.0,
    -2664477.0 / 120960.0,
    2102243.0 / 120960.0,
    -1041723.0 / 120960.0,
    295767.0 / 120960.0,
    -36799.0 / 120960.0,
];

/// Adams-Moulton 7-step corrector coefficients (implicit, 8th order).
/// y_{n+1} = y_n + h * sum(b*_i * f_{n+1-i}, i=0..7)
const AM8_COEFFS: [f64; 8] = [
    36799.0 / 120960.0,
    139849.0 / 120960.0,
    -121797.0 / 120960.0,
    123133.0 / 120960.0,
    -88547.0 / 120960.0,
    41499.0 / 120960.0,
    -11351.0 / 120960.0,
    1375.0 / 120960.0,
];

/// State of a single body for ABM integration.
#[derive(Debug, Clone)]
pub struct Abm8BodyState {
    pub pos: Vector3<f64>,
    pub vel: Vector3<f64>,
}

/// 8th-order Adams-Bashforth-Moulton integrator with ring buffer history.
pub struct Abm8Integrator {
    /// Time step.
    dt: f64,
    /// Number of steps taken (used to detect startup phase).
    step_count: usize,
    /// Ring buffer of velocity history (for position prediction).
    vel_history: Vec<Vec<Vector3<f64>>>,
    /// Ring buffer of acceleration history (for velocity prediction).
    acc_history: Vec<Vec<Vector3<f64>>>,
    /// Current ring buffer index.
    ring_idx: usize,
}

impl Abm8Integrator {
    /// Create a new ABM-8 integrator with the given time step and body count.
    pub fn new(dt: f64, n_bodies: usize) -> Self {
        Self {
            dt,
            step_count: 0,
            vel_history: vec![vec![Vector3::zeros(); n_bodies]; 8],
            acc_history: vec![vec![Vector3::zeros(); n_bodies]; 8],
            ring_idx: 0,
        }
    }

    /// Number of steps taken so far.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Take one integration step.
    ///
    /// `accel_fn`: computes accelerations for all bodies given their current states.
    /// Returns the new states after one step of size dt.
    ///
    /// During the startup phase (first 8 steps), uses RK4.
    /// After that, uses the ABM-8 predictor-corrector.
    pub fn step<F>(&mut self, states: &mut [Abm8BodyState], accel_fn: &F)
    where
        F: Fn(&[Abm8BodyState]) -> Vec<Vector3<f64>>,
    {
        if self.step_count < 8 {
            // RK4 startup
            self.rk4_step(states, accel_fn);
        } else {
            // ABM-8 predictor-corrector
            self.abm8_step(states, accel_fn);
        }

        self.step_count += 1;
    }

    /// RK4 startup step. Also records history for the ABM phase.
    fn rk4_step<F>(&mut self, states: &mut [Abm8BodyState], accel_fn: &F)
    where
        F: Fn(&[Abm8BodyState]) -> Vec<Vector3<f64>>,
    {
        let n = states.len();
        let dt = self.dt;

        // Save initial state
        let initial: Vec<Abm8BodyState> = states.to_vec();

        // k1
        let k1_a = accel_fn(states);
        let k1_v: Vec<_> = states.iter().map(|s| s.vel).collect();

        // k2
        for i in 0..n {
            states[i].pos = initial[i].pos + k1_v[i] * (dt / 2.0);
            states[i].vel = initial[i].vel + k1_a[i] * (dt / 2.0);
        }
        let k2_a = accel_fn(states);
        let k2_v: Vec<_> = states.iter().map(|s| s.vel).collect();

        // k3
        for i in 0..n {
            states[i].pos = initial[i].pos + k2_v[i] * (dt / 2.0);
            states[i].vel = initial[i].vel + k2_a[i] * (dt / 2.0);
        }
        let k3_a = accel_fn(states);
        let k3_v: Vec<_> = states.iter().map(|s| s.vel).collect();

        // k4
        for i in 0..n {
            states[i].pos = initial[i].pos + k3_v[i] * dt;
            states[i].vel = initial[i].vel + k3_a[i] * dt;
        }
        let k4_a = accel_fn(states);
        let k4_v: Vec<_> = states.iter().map(|s| s.vel).collect();

        // Combine
        for i in 0..n {
            states[i].pos =
                initial[i].pos + (k1_v[i] + k2_v[i] * 2.0 + k3_v[i] * 2.0 + k4_v[i]) * (dt / 6.0);
            states[i].vel =
                initial[i].vel + (k1_a[i] + k2_a[i] * 2.0 + k3_a[i] * 2.0 + k4_a[i]) * (dt / 6.0);
        }

        // Record the acceleration at the START of this step into history
        let idx = self.ring_idx;
        for i in 0..n {
            self.vel_history[idx][i] = initial[i].vel;
            self.acc_history[idx][i] = k1_a[i];
        }
        self.ring_idx = (self.ring_idx + 1) % 8;
    }

    /// ABM-8 predictor-corrector step.
    fn abm8_step<F>(&mut self, states: &mut [Abm8BodyState], accel_fn: &F)
    where
        F: Fn(&[Abm8BodyState]) -> Vec<Vector3<f64>>,
    {
        let n = states.len();
        let dt = self.dt;

        // Record current velocity and acceleration before predicting
        let current_acc = accel_fn(states);
        let current_vel: Vec<_> = states.iter().map(|s| s.vel).collect();

        // Store into ring buffer at current index
        let idx = self.ring_idx;
        self.vel_history[idx][..n].copy_from_slice(&current_vel[..n]);
        self.acc_history[idx][..n].copy_from_slice(&current_acc[..n]);

        // Predict using Adams-Bashforth (explicit)
        let mut predicted = states.to_vec();
        for i in 0..n {
            let mut vel_sum = Vector3::zeros();
            let mut acc_sum = Vector3::zeros();

            for (k, &coeff) in AB8_COEFFS.iter().enumerate() {
                // Ring buffer: history[(idx - k) mod 8] = f_{n-k}
                let hist_idx = (idx + 8 - k) % 8;
                vel_sum += self.vel_history[hist_idx][i] * coeff;
                acc_sum += self.acc_history[hist_idx][i] * coeff;
            }

            predicted[i].pos = states[i].pos + vel_sum * dt;
            predicted[i].vel = states[i].vel + acc_sum * dt;
        }

        // Evaluate acceleration at predicted state
        let predicted_acc = accel_fn(&predicted);

        // Correct using Adams-Moulton (implicit, one iteration)
        // The corrector uses f_{n+1} (predicted) as the first term
        let next_idx = (idx + 1) % 8;
        for i in 0..n {
            // Build corrector sum: AM8_COEFFS[0] * f_{n+1} + AM8_COEFFS[1] * f_n + ...
            let mut vel_corr = predicted[i].vel * AM8_COEFFS[0];
            let mut acc_corr = predicted_acc[i] * AM8_COEFFS[0];

            for (k, &coeff) in AM8_COEFFS.iter().enumerate().skip(1) {
                let hist_idx = (idx + 8 + 1 - k) % 8;
                vel_corr += self.vel_history[hist_idx][i] * coeff;
                acc_corr += self.acc_history[hist_idx][i] * coeff;
            }

            states[i].pos += vel_corr * dt;
            states[i].vel += acc_corr * dt;
        }

        // Advance ring buffer
        self.ring_idx = next_idx;
    }

    /// Compute total orbital energy of a 2-body system (for conservation checks).
    /// Returns kinetic + potential energy in geometrized units.
    pub fn total_energy_2body(states: &[Abm8BodyState], gm_central: f64) -> f64 {
        if states.len() < 2 {
            return 0.0;
        }
        let r = (states[1].pos - states[0].pos).norm();
        let v = (states[1].vel - states[0].vel).norm();
        // Specific orbital energy: E = v^2/2 - GM/r
        0.5 * v * v - gm_central / r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// 2-body Kepler problem: circular orbit.
    /// GM = 1, r = 1 => v_circ = 1, period = 2*pi.
    #[test]
    fn kepler_circular_orbit_energy_conservation() {
        let gm: f64 = 1.0;
        let r0: f64 = 1.0;
        let v_circ = (gm / r0).sqrt();

        let mut states = vec![
            Abm8BodyState {
                pos: Vector3::zeros(),
                vel: Vector3::zeros(),
            },
            Abm8BodyState {
                pos: Vector3::new(r0, 0.0, 0.0),
                vel: Vector3::new(0.0, v_circ, 0.0),
            },
        ];

        let accel_fn = |s: &[Abm8BodyState]| -> Vec<Vector3<f64>> {
            let r_vec = s[1].pos - s[0].pos;
            let r = r_vec.norm();
            let acc = -r_vec * (gm / (r * r * r));
            vec![Vector3::zeros(), acc]
        };

        let period = 2.0 * PI;
        let dt = period / 10000.0;
        let n_steps = 100_000; // 10 orbits

        let mut integrator = Abm8Integrator::new(dt, 2);
        let e0 = Abm8Integrator::total_energy_2body(&states, gm);

        for _ in 0..n_steps {
            integrator.step(&mut states, &accel_fn);
        }

        let e_final = Abm8Integrator::total_energy_2body(&states, gm);
        let drift = ((e_final - e0) / e0.abs()).abs();

        assert!(
            drift < 1e-8,
            "Energy drift over 10 orbits: {:.2e} (expected < 1e-8)",
            drift
        );
    }

    #[test]
    fn kepler_orbit_returns_to_start() {
        let gm: f64 = 1.0;
        let r0: f64 = 1.0;
        let v_circ = (gm / r0).sqrt();

        let mut states = vec![
            Abm8BodyState {
                pos: Vector3::zeros(),
                vel: Vector3::zeros(),
            },
            Abm8BodyState {
                pos: Vector3::new(r0, 0.0, 0.0),
                vel: Vector3::new(0.0, v_circ, 0.0),
            },
        ];

        let accel_fn = |s: &[Abm8BodyState]| -> Vec<Vector3<f64>> {
            let r_vec = s[1].pos - s[0].pos;
            let r = r_vec.norm();
            let a = -r_vec * (gm / (r * r * r));
            vec![Vector3::zeros(), a]
        };

        let period = 2.0 * PI;
        let dt = period / 10000.0;
        let n_steps = 10_000; // 1 orbit

        let mut integrator = Abm8Integrator::new(dt, 2);

        for _ in 0..n_steps {
            integrator.step(&mut states, &accel_fn);
        }

        // After 1 orbit, should return near (1, 0, 0)
        let pos = states[1].pos;
        let dist_from_start = ((pos.x - r0).powi(2) + pos.y.powi(2) + pos.z.powi(2)).sqrt();

        assert!(
            dist_from_start < 1e-6,
            "Position drift after 1 orbit: {:.2e}",
            dist_from_start
        );
    }

    #[test]
    fn startup_uses_rk4() {
        let mut integrator = Abm8Integrator::new(0.01, 1);
        let mut states = vec![Abm8BodyState {
            pos: Vector3::new(1.0, 0.0, 0.0),
            vel: Vector3::new(0.0, 1.0, 0.0),
        }];

        let accel_fn = |_: &[Abm8BodyState]| -> Vec<Vector3<f64>> { vec![Vector3::zeros()] };

        // First 8 steps should be RK4 startup
        for i in 0..8 {
            assert_eq!(integrator.step_count(), i);
            integrator.step(&mut states, &accel_fn);
        }
        assert_eq!(integrator.step_count(), 8);

        // Step 9 should use ABM
        integrator.step(&mut states, &accel_fn);
        assert_eq!(integrator.step_count(), 9);
    }

    #[test]
    fn eccentric_orbit_energy_conservation() {
        let gm: f64 = 1.0;
        // Elliptical orbit: perihelion at r=0.5, v = sqrt(3*GM/r) for e=0.5
        let r0: f64 = 0.5;
        let e: f64 = 0.5;
        let a = r0 / (1.0 - e); // semi-major axis = 1.0
        let v0 = (gm * (2.0 / r0 - 1.0 / a)).sqrt();

        let mut states = vec![
            Abm8BodyState {
                pos: Vector3::zeros(),
                vel: Vector3::zeros(),
            },
            Abm8BodyState {
                pos: Vector3::new(r0, 0.0, 0.0),
                vel: Vector3::new(0.0, v0, 0.0),
            },
        ];

        let accel_fn = |s: &[Abm8BodyState]| -> Vec<Vector3<f64>> {
            let r_vec = s[1].pos - s[0].pos;
            let r = r_vec.norm();
            let a = -r_vec * (gm / (r * r * r));
            vec![Vector3::zeros(), a]
        };

        let period = 2.0 * PI * a.powf(1.5) / gm.sqrt();
        let dt = period / 20000.0;
        let n_steps = 200_000; // 10 orbits

        let mut integrator = Abm8Integrator::new(dt, 2);
        let e0 = Abm8Integrator::total_energy_2body(&states, gm);

        for _ in 0..n_steps {
            integrator.step(&mut states, &accel_fn);
        }

        let e_final = Abm8Integrator::total_energy_2body(&states, gm);
        let drift = ((e_final - e0) / e0.abs()).abs();

        assert!(
            drift < 1e-6,
            "Eccentric orbit energy drift over 10 orbits: {:.2e}",
            drift
        );
    }
}
