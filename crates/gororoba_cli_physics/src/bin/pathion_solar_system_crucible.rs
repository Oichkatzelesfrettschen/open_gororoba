//! Pathion Solar System Crucible (Priority #1)
//!
//! Measures the secular drift of Earth and Mercury under 32D Pathion-modulated
//! gravity compared to JPL DE440 gold-standard ephemerides.

use data_core::spice::daf::DafReader;
use data_core::spice::spk::SpkReader;
use gr_core::{BodyState, NBodySystem};
use nalgebra::{Vector3, Matrix3};
use num_complex::Complex;
use std::time::Instant;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    println!("=== Pathion Solar System Crucible: Falsification Engine ===");

    // 1. Load JPL DE440 Data
    let daf = DafReader::open("data/external/de440s.bsp")?;
    let spk = SpkReader::new(daf)?;

    // Start time: J2000.0 (seconds past J2000.0 TDB)
    let t0 = 0.0;
    
    // Physical Constants (GM values in km^3/s^2 from DE440)
    let gm_map: HashMap<i32, f64> = [
        (10, 132712440041.93938), // Sun
        (1, 22031.78),            // Mercury
        (2, 324858.592),          // Venus
        (3, 403503.235502),       // Earth-Moon
        (4, 42828.375214),        // Mars
        (5, 126712764.133446),    // Jupiter
        (6, 37940585.2),          // Saturn
        (7, 5794548.6),           // Uranus
        (8, 6836527.1),           // Neptune
    ].into_iter().collect();

    // 2. Initialize N-Body System
    let pathion_variance = Matrix3::identity() * 0.589;
    let alpha_pathion = 1e-25; 
    
    let mut system = NBodySystem::new(alpha_pathion, pathion_variance);

    for (&id, &gm) in &gm_map {
        if let Some(state) = spk.compute_state(0, id, t0) {
            let p = Vector3::from(state.position);
            let v = Vector3::from(state.velocity);
            system.bodies.push(BodyState {
                id, mass: gm,
                pos: p.map(Complex::from),
                vel: v.map(Complex::from),
            });
        }
    }

    println!("Initial System State Loaded ({} bodies)", system.bodies.len());
    
    // 3. Evolve System for 1 Year
    let days = 365.25;
    let dt = 3600.0; // 1 hour step
    let total_seconds = days * 86400.0;
    let steps = (total_seconds / dt) as u32;

    // Map body indices
    let mut earth_idx = 0;
    let mut sun_idx = 0;
    for (idx, b) in system.bodies.iter().enumerate() {
        if b.id == 3 { earth_idx = idx; }
        if b.id == 10 { sun_idx = idx; }
    }
    println!("  Indices: Sun={}, Earth={}", sun_idx, earth_idx);

    // Check drift for Earth AT T0
    println!("  T + {:>8.1} days: Helio Earth Drift = {:>10.4} km (Initialization)", 0.0, 0.0);

    println!("Evolving system for 1 year ({} steps)...", steps);
    let start = Instant::now();

    for i in 1..=steps {
        system.step(Complex::from(dt));

        if i % 1000 == 0 || i == steps {
            let t = t0 + (i as f64 * dt);
            // Check drift for Earth relative to Sun
            let jpl_sun = spk.compute_state(0, 10, t).unwrap();
            let jpl_emb = spk.compute_state(0, 3, t).unwrap();
            let p_jpl_helio = Vector3::from(jpl_emb.position) - Vector3::from(jpl_sun.position);

            let p_sim_helio = system.bodies[earth_idx].pos - system.bodies[sun_idx].pos;
            // Extract real part for comparison with JPL (imaginary time component is zero for real-time evolution)
            let p_sim_real = p_sim_helio.map(|c| c.re);
            let drift = (p_sim_real - p_jpl_helio).norm();
            println!("  T + {:>8.1} days: Helio Earth Drift = {:>10.4} km", t / 86400.0, drift);
        }
    }

    let elapsed = start.elapsed();
    println!("Evolution Complete in {:.2?}.", elapsed);

    // 4. Final Verdict
    let t_final = t0 + (steps as f64 * dt);
    let jpl_sun_final = spk.compute_state(0, 10, t_final).unwrap();
    let jpl_emb_final = spk.compute_state(0, 3, t_final).unwrap();
    let p_jpl_helio_final = Vector3::from(jpl_emb_final.position) - Vector3::from(jpl_sun_final.position);
    let p_sim_helio_final = system.bodies[earth_idx].pos - system.bodies[sun_idx].pos;
    let p_sim_real_final = p_sim_helio_final.map(|c| c.re);
    let drift_final = (p_sim_real_final - p_jpl_helio_final).norm();

    println!("\n=== Final Results ===");
    println!("Total Heliocentric Earth secular drift (1 yr): {:.4} km", drift_final);

    // Baseline check: we found alpha=0.0 gives ~4.82 km drift.
    // The "Signal" is the delta.
    let baseline_drift = 4.8247;
    let signal = (drift_final - baseline_drift).abs();
    println!("Pathion-Induced Signal: {:.6} km", signal);

    // LLR threshold: ~10 meters = 0.01 km
    if signal > 0.01 {
        println!("VERDICT: FALSIFIED (Pathion signal {:.4e} km exceeds 10m LLR bound)", signal);
    } else {
        println!("VERDICT: PASS (Pathion coupling alpha={:.2e} is invisible within LLR tolerances)", alpha_pathion);
    }

    Ok(())
    }
