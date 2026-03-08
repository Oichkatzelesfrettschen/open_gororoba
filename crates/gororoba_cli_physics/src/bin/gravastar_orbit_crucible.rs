//! Gravastar N-Body Orbit Crucible
//!
//! Integrates planetary orbits (Mercury, Venus, Earth, Mars) around a
//! gravastar (instead of a point-mass Sun) using the ABM-8 integrator.
//! Compares Schwarzschild control vs gravastar potential, with optional
//! Pathion resonance modulation.
//!
//! Output: CSV with orbital elements, energy conservation, and resonance activation.
//!
//! # Usage
//! ```text
//! cargo run --bin gravastar-orbit-crucible -- --years 10 --dt 3600
//! ```

use std::f64::consts::PI;

use cosmology_core::{
    gravastar::{AnisotropicParams, GravastarConfig, PolytropicEos, solve_gravastar},
    gravastar_potential::GravastarPotential,
};
use gr_core::forces::abm8_integrator::{Abm8BodyState, Abm8Integrator};
use nalgebra::Vector3;
use pathion_ellip::{
    pathion_eigenvalues::PathionEigenvalueSpectrum,
    pathion_resonance::{ResonanceConfig, total_resonance_coupling},
};

/// Solar GM in km^3/s^2.
const GM_SUN: f64 = 1.327_124_400_18e11;

/// Planet initial conditions (heliocentric ecliptic J2000, simplified circular orbits).
/// (name, semi-major axis km, orbital velocity km/s)
const PLANETS: [(&str, f64, f64); 4] = [
    ("Mercury", 5.791e7, 47.87),
    ("Venus", 1.082e8, 35.02),
    ("Earth", 1.496e8, 29.78),
    ("Mars", 2.279e8, 24.077),
];

/// Convert solar mass gravastar to geometrized units matching GM_SUN.
/// In geometrized: GM/c^2 = 1.476 km for 1 solar mass.
const GM_OVER_C2_SUN: f64 = 1.476;

fn main() {
    let years: f64 = std::env::args()
        .position(|a| a == "--years")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);

    let dt: f64 = std::env::args()
        .position(|a| a == "--dt")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(3600.0); // 1 hour default

    let alpha_pathion: f64 = std::env::args()
        .position(|a| a == "--alpha")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    eprintln!("=== Gravastar N-Body Orbit Crucible ===");
    eprintln!("Integration: {} years, dt = {} s", years, dt);
    eprintln!("Alpha_pathion: {:.2e}", alpha_pathion);

    // Build gravastar solution
    let config = GravastarConfig {
        r1: 3.0 * GM_OVER_C2_SUN, // Interior radius ~ 3 Schwarzschild radii
        m_target: GM_OVER_C2_SUN, // 1 solar mass in geometrized units
        compactness_target: 0.7,
        eos: PolytropicEos::stiff(),
        aniso: AnisotropicParams::isotropic(),
        dr: 1e-4,
        p_floor: 1e-12,
    };

    let gravastar_sol = solve_gravastar(&config);
    let has_gravastar = gravastar_sol.is_some();

    let potential = gravastar_sol
        .as_ref()
        .map(GravastarPotential::from_solution);

    if let Some(ref pot) = potential {
        eprintln!(
            "Gravastar: r1={:.3} km, r2={:.3} km, M={:.6} km",
            pot.r1, pot.r2, pot.mass
        );
    } else {
        eprintln!("WARNING: Gravastar solution failed, using pure Schwarzschild");
    }

    // Compute Pathion eigenvalue spectrum
    let pathion_spectrum = PathionEigenvalueSpectrum::compute();
    eprintln!(
        "Pathion spectrum: {} eigenvalues, {} components",
        pathion_spectrum.eigenvalues.len(),
        pathion_spectrum.n_components
    );

    let resonance_config = ResonanceConfig::default();

    // Set up N-body system: Sun (fixed) + 4 planets
    // Body 0 = Sun (at origin), Bodies 1-4 = planets
    let n_bodies = 5;
    let mut states_grav = Vec::with_capacity(n_bodies);
    let mut states_ctrl = Vec::with_capacity(n_bodies);

    // Sun at origin
    states_grav.push(Abm8BodyState {
        pos: Vector3::zeros(),
        vel: Vector3::zeros(),
    });
    states_ctrl.push(Abm8BodyState {
        pos: Vector3::zeros(),
        vel: Vector3::zeros(),
    });

    // Planets in circular orbits (x-y plane)
    for (i, &(_name, a_km, v_km_s)) in PLANETS.iter().enumerate() {
        let angle = (i as f64) * PI / 2.0; // Spread 90 degrees apart
        let pos = Vector3::new(a_km * angle.cos(), a_km * angle.sin(), 0.0);
        let vel = Vector3::new(-v_km_s * angle.sin(), v_km_s * angle.cos(), 0.0);
        states_grav.push(Abm8BodyState { pos, vel });
        states_ctrl.push(Abm8BodyState { pos, vel });
    }

    // Acceleration function: Schwarzschild control (point mass)
    let accel_schwarzschild = |s: &[Abm8BodyState]| -> Vec<Vector3<f64>> {
        let mut accels = vec![Vector3::zeros(); s.len()];
        for i in 1..s.len() {
            let r_vec = s[i].pos - s[0].pos; // heliocentric
            let r = r_vec.norm();
            if r > 1e-6 {
                accels[i] = -r_vec * (GM_SUN / (r * r * r));
            }
        }
        accels
    };

    // Acceleration function: Gravastar + Pathion resonance
    let accel_gravastar = |s: &[Abm8BodyState]| -> Vec<Vector3<f64>> {
        let mut accels = vec![Vector3::zeros(); s.len()];
        for i in 1..s.len() {
            let r_vec = s[i].pos - s[0].pos;
            let r = r_vec.norm();
            if r < 1e-6 {
                continue;
            }

            // Base acceleration: use gravastar potential if available,
            // otherwise Schwarzschild. The gravastar potential is in
            // geometrized units (km), so we scale by GM_SUN / GM_OVER_C2_SUN.
            let a_radial = if let Some(ref pot) = potential {
                // Scale from geometrized to physical units
                let r_geom = r * GM_OVER_C2_SUN / r; // This is just GM_OVER_C2_SUN
                let _ = r_geom; // Suppress unused
                // For r >> r2 (all planetary orbits), this is just -GM/r^2
                // The gravastar correction only matters near r ~ r2
                let a_r = pot.radial_acceleration(r * GM_OVER_C2_SUN / 1.496e8);
                // Scale back to km/s^2
                a_r * GM_SUN / GM_OVER_C2_SUN
            } else {
                -GM_SUN / (r * r)
            };

            // At planetary distances (>> gravastar radius), the potential
            // is effectively Schwarzschild. The correction is negligible.
            // The interesting physics comes from the Pathion resonance.
            let r_hat = r_vec / r;
            accels[i] = r_hat * a_radial;

            // Pathion resonance perturbation
            if alpha_pathion > 0.0 {
                let v = s[i].vel.norm();
                let orbital_freq = if r > 0.0 { v / (2.0 * PI * r) } else { 0.0 };
                let coupling =
                    total_resonance_coupling(&pathion_spectrum, orbital_freq, &resonance_config);

                // Perturbation: tangential kick proportional to coupling
                let v_hat = if v > 1e-10 {
                    s[i].vel / v
                } else {
                    Vector3::zeros()
                };
                accels[i] += v_hat * (alpha_pathion * coupling * GM_SUN / (r * r));
            }
        }
        accels
    };

    // Integrators
    let mut integrator_ctrl = Abm8Integrator::new(dt, n_bodies);
    let mut integrator_grav = Abm8Integrator::new(dt, n_bodies);

    let total_seconds = years * 365.25 * 86400.0;
    let n_steps = (total_seconds / dt) as usize;
    let output_interval = n_steps / 1000; // ~1000 output points

    // Initial energies
    let e0_ctrl = compute_total_energy(&states_ctrl);
    let e0_grav = compute_total_energy(&states_grav);

    // CSV header
    println!(
        "step,time_years,planet,r_ctrl_au,r_grav_au,delta_r_km,energy_drift_ctrl,energy_drift_grav,resonance_coupling"
    );

    eprintln!("Integrating {} steps ({:.1} years)...", n_steps, years);

    for step in 0..n_steps {
        integrator_ctrl.step(&mut states_ctrl, &accel_schwarzschild);
        integrator_grav.step(&mut states_grav, &accel_gravastar);

        if output_interval > 0 && step % output_interval == 0 {
            let t_years = (step as f64) * dt / (365.25 * 86400.0);
            let e_ctrl = compute_total_energy(&states_ctrl);
            let e_grav = compute_total_energy(&states_grav);
            let drift_ctrl = if e0_ctrl.abs() > 1e-20 {
                (e_ctrl - e0_ctrl) / e0_ctrl.abs()
            } else {
                0.0
            };
            let drift_grav = if e0_grav.abs() > 1e-20 {
                (e_grav - e0_grav) / e0_grav.abs()
            } else {
                0.0
            };

            for (pi, &(name, _, _)) in PLANETS.iter().enumerate() {
                let body_idx = pi + 1;
                let r_ctrl = states_ctrl[body_idx].pos.norm();
                let r_grav = states_grav[body_idx].pos.norm();
                let delta_r = (r_grav - r_ctrl).abs();

                let v = states_grav[body_idx].vel.norm();
                let orbital_freq = v / (2.0 * PI * r_grav);
                let coupling =
                    total_resonance_coupling(&pathion_spectrum, orbital_freq, &resonance_config);

                println!(
                    "{},{:.6},{},{:.8},{:.8},{:.6e},{:.6e},{:.6e},{:.6e}",
                    step,
                    t_years,
                    name,
                    r_ctrl / 1.496e8, // AU
                    r_grav / 1.496e8,
                    delta_r,
                    drift_ctrl,
                    drift_grav,
                    coupling
                );
            }
        }
    }

    // Final summary
    let e_final_ctrl = compute_total_energy(&states_ctrl);
    let e_final_grav = compute_total_energy(&states_grav);
    let drift_ctrl = if e0_ctrl.abs() > 1e-20 {
        ((e_final_ctrl - e0_ctrl) / e0_ctrl.abs()).abs()
    } else {
        0.0
    };
    let drift_grav = if e0_grav.abs() > 1e-20 {
        ((e_final_grav - e0_grav) / e0_grav.abs()).abs()
    } else {
        0.0
    };

    eprintln!("\n=== Final Summary ===");
    eprintln!("Gravastar available: {}", has_gravastar);
    eprintln!("Control energy drift: {:.2e}", drift_ctrl);
    eprintln!("Gravastar energy drift: {:.2e}", drift_grav);

    for (pi, &(name, a_km, _)) in PLANETS.iter().enumerate() {
        let body_idx = pi + 1;
        let r_ctrl = states_ctrl[body_idx].pos.norm();
        let r_grav = states_grav[body_idx].pos.norm();
        let delta_r = (r_grav - r_ctrl).abs();
        eprintln!(
            "  {}: r_ctrl={:.3} AU, r_grav={:.3} AU, delta_r={:.3e} km ({:.3e} of a)",
            name,
            r_ctrl / 1.496e8,
            r_grav / 1.496e8,
            delta_r,
            delta_r / a_km
        );
    }
}

/// Compute total (kinetic + potential) energy of the system.
fn compute_total_energy(states: &[Abm8BodyState]) -> f64 {
    let mut energy = 0.0;
    for i in 1..states.len() {
        let r = (states[i].pos - states[0].pos).norm();
        let v = states[i].vel.norm();
        // Specific orbital energy per planet
        energy += 0.5 * v * v - GM_SUN / r;
    }
    energy
}
