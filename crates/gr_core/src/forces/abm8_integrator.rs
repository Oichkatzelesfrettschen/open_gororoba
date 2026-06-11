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

#[inline(always)]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline(always)]
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline(always)]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline(always)]
fn norm_sq3(a: [f64; 3]) -> f64 {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
}

/// State of a single body for ABM integration.
#[derive(Debug, Clone)]
pub struct Abm8BodyState {
    pub pos: [f64; 3],
    pub vel: [f64; 3],
}

/// 8th-order Adams-Bashforth-Moulton integrator with ring buffer history.
pub struct Abm8Integrator {
    /// Time step.
    dt: f64,
    /// Number of steps taken (used to detect startup phase).
    step_count: usize,
    /// Ring buffer of velocity history (for position prediction).
    vel_history: Vec<Vec<[f64; 3]>>,
    /// Ring buffer of acceleration history (for velocity prediction).
    acc_history: Vec<Vec<[f64; 3]>>,
    /// Current ring buffer index.
    ring_idx: usize,
}

impl Abm8Integrator {
    /// Create a new ABM-8 integrator with the given time step and body count.
    pub fn new(dt: f64, n_bodies: usize) -> Self {
        Self {
            dt,
            step_count: 0,
            vel_history: vec![vec![[0.0; 3]; n_bodies]; 8],
            acc_history: vec![vec![[0.0; 3]; n_bodies]; 8],
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
        F: Fn(&[Abm8BodyState]) -> Vec<[f64; 3]>,
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
        F: Fn(&[Abm8BodyState]) -> Vec<[f64; 3]>,
    {
        let n = states.len();
        let dt = self.dt;

        // Save initial state
        let initial: Vec<Abm8BodyState> = states.to_vec();

        // k1
        let k1_a = accel_fn(states);
        let k1_v: Vec<[f64; 3]> = states.iter().map(|s| s.vel).collect();

        // k2
        for i in 0..n {
            states[i].pos = add3(initial[i].pos, scale3(k1_v[i], dt / 2.0));
            states[i].vel = add3(initial[i].vel, scale3(k1_a[i], dt / 2.0));
        }
        let k2_a = accel_fn(states);
        let k2_v: Vec<[f64; 3]> = states.iter().map(|s| s.vel).collect();

        // k3
        for i in 0..n {
            states[i].pos = add3(initial[i].pos, scale3(k2_v[i], dt / 2.0));
            states[i].vel = add3(initial[i].vel, scale3(k2_a[i], dt / 2.0));
        }
        let k3_a = accel_fn(states);
        let k3_v: Vec<[f64; 3]> = states.iter().map(|s| s.vel).collect();

        // k4
        for i in 0..n {
            states[i].pos = add3(initial[i].pos, scale3(k3_v[i], dt));
            states[i].vel = add3(initial[i].vel, scale3(k3_a[i], dt));
        }
        let k4_a = accel_fn(states);
        let k4_v: Vec<[f64; 3]> = states.iter().map(|s| s.vel).collect();

        // Combine: (k1 + 2*k2 + 2*k3 + k4) * dt/6
        for i in 0..n {
            let sum_v = add3(
                add3(k1_v[i], scale3(k2_v[i], 2.0)),
                add3(scale3(k3_v[i], 2.0), k4_v[i]),
            );
            let sum_a = add3(
                add3(k1_a[i], scale3(k2_a[i], 2.0)),
                add3(scale3(k3_a[i], 2.0), k4_a[i]),
            );
            states[i].pos = add3(initial[i].pos, scale3(sum_v, dt / 6.0));
            states[i].vel = add3(initial[i].vel, scale3(sum_a, dt / 6.0));
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
        F: Fn(&[Abm8BodyState]) -> Vec<[f64; 3]>,
    {
        let n = states.len();
        let dt = self.dt;

        // Record current velocity and acceleration before predicting
        let current_acc = accel_fn(states);
        let current_vel: Vec<[f64; 3]> = states.iter().map(|s| s.vel).collect();

        // Store into ring buffer at current index
        let idx = self.ring_idx;
        self.vel_history[idx][..n].copy_from_slice(&current_vel[..n]);
        self.acc_history[idx][..n].copy_from_slice(&current_acc[..n]);

        // Predict using Adams-Bashforth (explicit)
        let mut predicted = states.to_vec();
        for i in 0..n {
            let mut v_hist_x = [0.0; 8];
            let mut v_hist_y = [0.0; 8];
            let mut v_hist_z = [0.0; 8];
            let mut a_hist_x = [0.0; 8];
            let mut a_hist_y = [0.0; 8];
            let mut a_hist_z = [0.0; 8];

            for k in 0..8 {
                let hist_idx = (idx + 8 - k) % 8;
                let v = self.vel_history[hist_idx][i];
                let a = self.acc_history[hist_idx][i];
                v_hist_x[k] = v[0];
                v_hist_y[k] = v[1];
                v_hist_z[k] = v[2];
                a_hist_x[k] = a[0];
                a_hist_y[k] = a[1];
                a_hist_z[k] = a[2];
            }

            let vel_sum = [
                verified_core::x87_math::x87_abm8_dot_product(&v_hist_x, &AB8_COEFFS),
                verified_core::x87_math::x87_abm8_dot_product(&v_hist_y, &AB8_COEFFS),
                verified_core::x87_math::x87_abm8_dot_product(&v_hist_z, &AB8_COEFFS),
            ];

            let acc_sum = [
                verified_core::x87_math::x87_abm8_dot_product(&a_hist_x, &AB8_COEFFS),
                verified_core::x87_math::x87_abm8_dot_product(&a_hist_y, &AB8_COEFFS),
                verified_core::x87_math::x87_abm8_dot_product(&a_hist_z, &AB8_COEFFS),
            ];

            predicted[i].pos = add3(states[i].pos, scale3(vel_sum, dt));
            predicted[i].vel = add3(states[i].vel, scale3(acc_sum, dt));
        }

        // Evaluate acceleration at predicted state
        let predicted_acc = accel_fn(&predicted);

        // Correct using Adams-Moulton (implicit, one iteration)
        // The corrector uses f_{n+1} (predicted) as the first term
        let next_idx = (idx + 1) % 8;
        for i in 0..n {
            let mut v_hist_x = [0.0; 8];
            let mut v_hist_y = [0.0; 8];
            let mut v_hist_z = [0.0; 8];
            let mut a_hist_x = [0.0; 8];
            let mut a_hist_y = [0.0; 8];
            let mut a_hist_z = [0.0; 8];

            // AM8 uses the newly predicted f_{n+1} at index 0
            v_hist_x[0] = predicted[i].vel[0];
            v_hist_y[0] = predicted[i].vel[1];
            v_hist_z[0] = predicted[i].vel[2];
            a_hist_x[0] = predicted_acc[i][0];
            a_hist_y[0] = predicted_acc[i][1];
            a_hist_z[0] = predicted_acc[i][2];

            for k in 1..8 {
                let hist_idx = (idx + 8 + 1 - k) % 8;
                let v = self.vel_history[hist_idx][i];
                let a = self.acc_history[hist_idx][i];
                v_hist_x[k] = v[0];
                v_hist_y[k] = v[1];
                v_hist_z[k] = v[2];
                a_hist_x[k] = a[0];
                a_hist_y[k] = a[1];
                a_hist_z[k] = a[2];
            }

            let vel_corr = [
                verified_core::x87_math::x87_abm8_dot_product(&v_hist_x, &AM8_COEFFS),
                verified_core::x87_math::x87_abm8_dot_product(&v_hist_y, &AM8_COEFFS),
                verified_core::x87_math::x87_abm8_dot_product(&v_hist_z, &AM8_COEFFS),
            ];

            let acc_corr = [
                verified_core::x87_math::x87_abm8_dot_product(&a_hist_x, &AM8_COEFFS),
                verified_core::x87_math::x87_abm8_dot_product(&a_hist_y, &AM8_COEFFS),
                verified_core::x87_math::x87_abm8_dot_product(&a_hist_z, &AM8_COEFFS),
            ];

            let dp = scale3(vel_corr, dt);
            let dv = scale3(acc_corr, dt);
            states[i].pos[0] += dp[0];
            states[i].pos[1] += dp[1];
            states[i].pos[2] += dp[2];
            states[i].vel[0] += dv[0];
            states[i].vel[1] += dv[1];
            states[i].vel[2] += dv[2];
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
        let dr = sub3(states[1].pos, states[0].pos);
        let dv = sub3(states[1].vel, states[0].vel);
        let r = norm_sq3(dr).sqrt();
        let v = norm_sq3(dv).sqrt();
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
                pos: [0.0; 3],
                vel: [0.0; 3],
            },
            Abm8BodyState {
                pos: [r0, 0.0, 0.0],
                vel: [0.0, v_circ, 0.0],
            },
        ];

        let accel_fn = |s: &[Abm8BodyState]| -> Vec<[f64; 3]> {
            let r_vec = sub3(s[1].pos, s[0].pos);
            let r = norm_sq3(r_vec).sqrt();
            let acc = scale3(r_vec, -gm / (r * r * r));
            vec![[0.0; 3], acc]
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
                pos: [0.0; 3],
                vel: [0.0; 3],
            },
            Abm8BodyState {
                pos: [r0, 0.0, 0.0],
                vel: [0.0, v_circ, 0.0],
            },
        ];

        let accel_fn = |s: &[Abm8BodyState]| -> Vec<[f64; 3]> {
            let r_vec = sub3(s[1].pos, s[0].pos);
            let r = norm_sq3(r_vec).sqrt();
            let a = scale3(r_vec, -gm / (r * r * r));
            vec![[0.0; 3], a]
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
        let dist_from_start = ((pos[0] - r0).powi(2) + pos[1].powi(2) + pos[2].powi(2)).sqrt();

        assert!(
            dist_from_start < 1e-6,
            "Position drift after 1 orbit: {:.2e}",
            dist_from_start
        );
    }

    #[test]
    fn kepler_circular_orbit_long_precision() {
        let gm: f64 = 1.0;
        let r0: f64 = 1.0;
        let v_circ = (gm / r0).sqrt();

        let mut states = vec![
            Abm8BodyState {
                pos: [0.0; 3],
                vel: [0.0; 3],
            },
            Abm8BodyState {
                pos: [r0, 0.0, 0.0],
                vel: [0.0, v_circ, 0.0],
            },
        ];

        let accel_fn = |s: &[Abm8BodyState]| -> Vec<[f64; 3]> {
            let r_vec = sub3(s[1].pos, s[0].pos);
            let r = norm_sq3(r_vec).sqrt();
            let a = scale3(r_vec, -gm / r.powi(3));
            vec![[0.0; 3], a]
        };

        let initial_energy = Abm8Integrator::total_energy_2body(&states, gm);
        let period = 2.0 * std::f64::consts::PI;
        let dt = 0.01;
        let n_orbits = 10;
        let n_steps = (n_orbits as f64 * period / dt) as usize;

        let mut integrator = Abm8Integrator::new(dt, 2);

        for _ in 0..n_steps {
            integrator.step(&mut states, &accel_fn);
        }

        let final_energy = Abm8Integrator::total_energy_2body(&states, gm);
        let energy_drift = (final_energy - initial_energy).abs();

        println!("Initial Energy: {:.18}", initial_energy);
        println!("Final Energy:   {:.18}", final_energy);
        println!("Energy Drift:   {:.2e}", energy_drift);

        assert!(
            energy_drift < 1e-12,
            "Energy drift too high: {:.2e}",
            energy_drift
        );
    }

    #[test]
    fn startup_uses_rk4() {
        let mut integrator = Abm8Integrator::new(0.01, 1);
        let mut states = vec![Abm8BodyState {
            pos: [1.0, 0.0, 0.0],
            vel: [0.0, 1.0, 0.0],
        }];

        let accel_fn = |_: &[Abm8BodyState]| -> Vec<[f64; 3]> { vec![[0.0; 3]] };

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
                pos: [0.0; 3],
                vel: [0.0; 3],
            },
            Abm8BodyState {
                pos: [r0, 0.0, 0.0],
                vel: [0.0, v0, 0.0],
            },
        ];

        let accel_fn = |s: &[Abm8BodyState]| -> Vec<[f64; 3]> {
            let r_vec = sub3(s[1].pos, s[0].pos);
            let r = norm_sq3(r_vec).sqrt();
            let a = scale3(r_vec, -gm / (r * r * r));
            vec![[0.0; 3], a]
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
