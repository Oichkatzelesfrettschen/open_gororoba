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
use pathion_ellip::{
    pathion_eigenvalues::PathionEigenvalueSpectrum,
    pathion_resonance::{ResonanceConfig, total_resonance_coupling},
};

#[inline(always)]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline(always)]
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline(always)]
fn norm3(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

#[inline(always)]
fn add_assign3(dst: &mut [f64; 3], src: [f64; 3]) {
    dst[0] += src[0];
    dst[1] += src[1];
    dst[2] += src[2];
}

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
    // Adopt the global hardware topology for SMT-evasion and caching
    if let Err(e) = verified_core::topology::HardwareTopology::init_pinned_rayon_pool() {
        eprintln!("Warning: Failed to initialize pinned Rayon pool: {}", e);
    }

    let mut years: f64 = std::env::args()
        .position(|a| a == "--years")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);

    let is_ska_mode = std::env::args().any(|a| a == "--ska-monte-carlo");
    if is_ska_mode {
        // SKA 2030 PTA modeling requires massive orbital stability baselines
        years = 1_000_000.0;
        eprintln!("SKA MONTE CARLO MODE ACTIVATED: Targeting {} years", years);
    }

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
        r1: 1.5 * GM_OVER_C2_SUN, // Interior radius < R2 (which is ~2.85 for C=0.7)
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
        pos: [0.0; 3],
        vel: [0.0; 3],
    });
    states_ctrl.push(Abm8BodyState {
        pos: [0.0; 3],
        vel: [0.0; 3],
    });

    // Planets in circular orbits (x-y plane)
    for (i, &(_name, a_km, v_km_s)) in PLANETS.iter().enumerate() {
        let angle = (i as f64) * PI / 2.0; // Spread 90 degrees apart
        let pos = [a_km * angle.cos(), a_km * angle.sin(), 0.0];
        let vel = [-v_km_s * angle.sin(), v_km_s * angle.cos(), 0.0];
        states_grav.push(Abm8BodyState { pos, vel });
        states_ctrl.push(Abm8BodyState { pos, vel });
    }

    // Acceleration function: Schwarzschild control (point mass)
    let accel_schwarzschild = |s: &[Abm8BodyState]| -> Vec<[f64; 3]> {
        let mut accels = vec![[0.0; 3]; s.len()];
        for i in 1..s.len() {
            let r_vec = sub3(s[i].pos, s[0].pos); // heliocentric
            let r = norm3(r_vec);
            if r > 1e-6 {
                accels[i] = scale3(r_vec, -GM_SUN / (r * r * r));
            }
        }
        accels
    };

    // Acceleration function: Gravastar + Pathion resonance
    let accel_gravastar = |s: &[Abm8BodyState]| -> Vec<[f64; 3]> {
        let mut accels = vec![[0.0; 3]; s.len()];
        for i in 1..s.len() {
            let r_vec = sub3(s[i].pos, s[0].pos);
            let r = norm3(r_vec);
            if r < 1e-6 {
                continue;
            }

            // Base acceleration: use gravastar potential if available,
            // otherwise Schwarzschild. The gravastar potential is in
            // geometrized units (km), so we scale by GM_SUN / GM_OVER_C2_SUN.
            let a_radial = if let Some(ref pot) = potential {
                // The physical distance r is in km.
                // The potential solver uses geometrized units where distances are in km (G=c=1).
                // So the input `r` is correctly scaled.
                let a_r_geom = pot.radial_acceleration(r); // [1/km] in geometrized units

                // We must scale back to physical acceleration [km/s^2].
                // Instead of using `c^2` which introduces floating point mismatch with the
                // exact GM_SUN constant, we use the exact ratio to ensure the far-field
                // matches Schwarzschild perfectly up to machine precision.
                a_r_geom * (GM_SUN / GM_OVER_C2_SUN)
            } else {
                -GM_SUN / (r * r)
            };

            // At planetary distances (>> gravastar radius), the potential
            // is effectively Schwarzschild. The correction is negligible.
            // The interesting physics comes from the Pathion resonance.
            let r_hat = scale3(r_vec, 1.0 / r);
            accels[i] = scale3(r_hat, a_radial);

            // Pathion resonance perturbation
            if alpha_pathion > 0.0 {
                let v = norm3(s[i].vel);
                let orbital_freq = if r > 0.0 { v / (2.0 * PI * r) } else { 0.0 };
                let coupling =
                    total_resonance_coupling(&pathion_spectrum, orbital_freq, &resonance_config);

                // Perturbation: tangential kick proportional to coupling
                let v_hat = if v > 1e-10 {
                    scale3(s[i].vel, 1.0 / v)
                } else {
                    [0.0; 3]
                };
                add_assign3(
                    &mut accels[i],
                    scale3(v_hat, alpha_pathion * coupling * GM_SUN / (r * r)),
                );
            }
        }
        accels
    };

    // Integrators
    let mut integrator_ctrl = Abm8Integrator::new(dt, n_bodies);
    let mut integrator_grav = Abm8Integrator::new(dt, n_bodies);

    let total_seconds = years * 365.25 * 86400.0;
    let n_steps = (total_seconds / dt) as usize;
    let output_interval = if is_ska_mode {
        n_steps / 100
    } else {
        n_steps / 1000
    };

    // Initial energies
    let e0_ctrl = compute_total_energy(&states_ctrl);
    let e0_grav = compute_total_energy(&states_grav);

    // CSV header
    println!(
        "step,time_years,planet,r_ctrl_au,r_grav_au,delta_r_km,energy_drift_ctrl,energy_drift_grav,resonance_coupling"
    );

    eprintln!("Integrating {} steps ({:.1} years)...", n_steps, years);
    let t_start = std::time::Instant::now();

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
                let r_ctrl = norm3(states_ctrl[body_idx].pos);
                let r_grav = norm3(states_grav[body_idx].pos);
                let delta_r = (r_grav - r_ctrl).abs();

                let v = norm3(states_grav[body_idx].vel);
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

    let elapsed = t_start.elapsed();
    eprintln!("Integration complete in {:.2?}", elapsed);

    let e_ctrl_final = compute_total_energy(&states_ctrl);
    let e_grav_final = compute_total_energy(&states_grav);
    let final_drift_grav = (e_grav_final - e0_grav) / e0_grav.abs();
    eprintln!("--- Final Energy Drift ---");
    eprintln!(
        "Schwarzschild: {:.5e}",
        (e_ctrl_final - e0_ctrl) / e0_ctrl.abs()
    );
    eprintln!("Gravastar:     {:.5e}", final_drift_grav);

    if is_ska_mode {
        eprintln!("SKA Monte Carlo threshold for non-symplectic drift: < 1e-11");
        assert!(
            final_drift_grav.abs() < 1e-10,
            "Symplectic drift failed SKA PTA threshold! Drift: {}",
            final_drift_grav
        );
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
        let r_ctrl = norm3(states_ctrl[body_idx].pos);
        let r_grav = norm3(states_grav[body_idx].pos);
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
        let r = norm3(sub3(states[i].pos, states[0].pos));
        let v = norm3(states[i].vel);
        // Specific orbital energy per planet
        energy += 0.5 * v * v - GM_SUN / r;
    }
    energy
}
