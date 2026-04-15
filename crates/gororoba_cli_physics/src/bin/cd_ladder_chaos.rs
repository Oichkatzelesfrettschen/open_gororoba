//! cd-ladder-chaos: Phase diagram of orbital chaos across Cayley-Dickson dimensions.
//!
//! Sweeps CD dimensions [4, 8, 16, 32, 64] and coupling strengths alpha in logspace,
//! integrating a 2-body Kepler orbit with CD-ladder drag to compute the maximal
//! Lyapunov exponent and classify the dynamical regime.
//!
//! Output: CSV with columns dim, alpha, lambda_max, regime, d_ky, violations

use gr_core::{
    cd_ladder_force::CdLadderForce,
    lyapunov::{LyapunovState, classify_regime, kaplan_yorke_dimension},
};

#[inline(always)]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline(always)]
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline(always)]
fn norm_sq3(a: [f64; 3]) -> f64 {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
}

/// Simple RK4 step for position + velocity under Kepler + CD drag.
fn rk4_step(
    pos: &mut [f64; 3],
    vel: &mut [f64; 3],
    gm: f64,
    force: &CdLadderForce,
    dt: f64,
) {
    let accel = |p: [f64; 3], v: [f64; 3]| -> [f64; 3] {
        let r2 = norm_sq3(p);
        let r = r2.sqrt();
        let a_kepler = scale3(p, -gm / (r2 * r));
        add3(a_kepler, force.drag_acceleration(&v))
    };

    let k1v = accel(*pos, *vel);
    let k1x = *vel;

    let p2 = add3(*pos, scale3(k1x, dt / 2.0));
    let v2 = add3(*vel, scale3(k1v, dt / 2.0));
    let k2v = accel(p2, v2);
    let k2x = v2;

    let p3 = add3(*pos, scale3(k2x, dt / 2.0));
    let v3 = add3(*vel, scale3(k2v, dt / 2.0));
    let k3v = accel(p3, v3);
    let k3x = v3;

    let p4 = add3(*pos, scale3(k3x, dt));
    let v4 = add3(*vel, scale3(k3v, dt));
    let k4v = accel(p4, v4);
    let k4x = v4;

    // Weighted sum: (k1 + 2*k2 + 2*k3 + k4) * dt/6
    let sum_x = add3(add3(k1x, scale3(k2x, 2.0)), add3(scale3(k3x, 2.0), k4x));
    let sum_v = add3(add3(k1v, scale3(k2v, 2.0)), add3(scale3(k3v, 2.0), k4v));
    let dx = scale3(sum_x, dt / 6.0);
    let dv = scale3(sum_v, dt / 6.0);
    pos[0] += dx[0]; pos[1] += dx[1]; pos[2] += dx[2];
    vel[0] += dv[0]; vel[1] += dv[1]; vel[2] += dv[2];
}

fn main() {
    println!("=== Cayley-Dickson Ladder of Orbital Chaos ===");
    println!();

    let dims: &[usize] = &[4, 8, 16, 32, 64];
    let n_alpha = 15;
    let alpha_min: f64 = 1e-6;
    let alpha_max: f64 = 1e-2;

    // Kepler parameters: circular orbit at r=1, GM=1 -> v=1, T=2*pi
    let gm = 1.0;
    let dt = 0.01;
    let n_steps = 50_000;
    let renorm_interval = 100;
    let dt_renorm = dt * renorm_interval as f64;

    // Lyapunov chaos threshold
    let chaos_threshold = 0.01;

    println!("dim,alpha,lambda_max,regime,violations");

    for &dim in dims {
        // Generate alpha values in logspace
        let log_min = alpha_min.log10();
        let log_max = alpha_max.log10();

        for i_alpha in 0..n_alpha {
            let frac = if n_alpha > 1 {
                i_alpha as f64 / (n_alpha - 1) as f64
            } else {
                0.5
            };
            let alpha = 10.0_f64.powf(log_min + frac * (log_max - log_min));

            let force = CdLadderForce::new(dim, alpha);
            let violations = force.violation_count();

            // Reference trajectory: circular orbit at r=1
            let mut pos: [f64; 3] = [1.0, 0.0, 0.0];
            let mut vel: [f64; 3] = [0.0, 1.0, 0.0];

            // Shadow trajectory (tiny perturbation along x)
            let delta0 = 1e-9;
            let mut pos_s: [f64; 3] = [1.0 + delta0, 0.0, 0.0];
            let mut vel_s: [f64; 3] = vel;

            let mut lyap = LyapunovState::new();

            for step in 0..n_steps {
                rk4_step(&mut pos, &mut vel, gm, &force, dt);
                rk4_step(&mut pos_s, &mut vel_s, gm, &force, dt);

                if (step + 1) % renorm_interval == 0 {
                    let dx = [pos_s[0] - pos[0], pos_s[1] - pos[1], pos_s[2] - pos[2]];
                    let dv = [vel_s[0] - vel[0], vel_s[1] - vel[1], vel_s[2] - vel[2]];
                    let separation = (norm_sq3(dx) + norm_sq3(dv)).sqrt();

                    if separation > 0.0 {
                        let stretch = separation / delta0;
                        lyap.record(stretch, dt_renorm);

                        // Renormalize shadow trajectory back to delta0 distance
                        let scale = delta0 / separation;
                        pos_s = add3(pos, scale3(dx, scale));
                        vel_s = add3(vel, scale3(dv, scale));
                    }
                }
            }

            let lambda_max = lyap.estimate();
            let regime = classify_regime(lambda_max, chaos_threshold);

            println!(
                "{},{:.6e},{:.8e},{},{}",
                dim, alpha, lambda_max, regime, violations
            );
        }
    }

    // Summary: Kaplan-Yorke dimension for a synthetic spectrum example
    println!();
    println!("=== Kaplan-Yorke Dimension Examples ===");
    let lorenz_spectrum = [0.9, 0.0, -14.57];
    let d_ky = kaplan_yorke_dimension(&lorenz_spectrum);
    println!("Lorenz-like spectrum [0.9, 0, -14.57]: D_KY = {d_ky:.4}");

    let regular_spectrum = [-0.1, -0.5, -1.0];
    let d_ky_reg = kaplan_yorke_dimension(&regular_spectrum);
    println!("Regular spectrum [-0.1, -0.5, -1.0]: D_KY = {d_ky_reg:.4}");
}
