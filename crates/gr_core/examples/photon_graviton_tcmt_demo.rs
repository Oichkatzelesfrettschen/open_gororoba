//! Demonstration of Photon-Graviton TCMT framework
//!
//! Shows how to:
//! 1. Create coupling parameters from worldline amplitude (E-073)
//! 2. Compute Fano lineshape and cross-sections
//! 3. Integrate TCMT equations over time
//! 4. Validate against energy conservation and optical theorem

use gr_core::photon_graviton_tcmt::*;
use num_complex::Complex64 as C64;

fn main() {
    println!("=== Photon-Graviton TCMT Demonstration ===\n");

    // ============================================================
    // STEP 1: Create gravitational coupling from E-073 amplitude
    // ============================================================
    println!("STEP 1: Extract TCMT parameters from worldline amplitude (E-073)");

    // Example values from a typical photon-graviton interaction
    let photon_freq = 1e15; // Hz (infrared)
    let grav_coupling = 1e-45; // Dimensionless (gravitational constant scaled)

    // Amplitude contributions from three diagrams
    let irreducible = 1e-10; // Irreducible diagram magnitude
    let tadpole_real = 0.1; // Tadpole self-energy (real part)
    let tadpole_imag = 1e-6; // Tadpole decay contribution
    let external_phase = 0.5; // External-leg phase shift

    let coupling = amplitude_bridge::three_diagram_amplitude_to_tcmt(
        irreducible,
        tadpole_real,
        tadpole_imag,
        external_phase,
        photon_freq,
        grav_coupling,
    );

    println!("  Coupling: {}", coupling);
    println!("  Weak-field valid: {}\n", coupling.is_weak_field());

    // ============================================================
    // STEP 2: Compute Fano lineshape and cross-sections
    // ============================================================
    println!("STEP 2: Compute Fano lineshape at resonance");

    let asymmetry = fano_lineshape::compute_asymmetry_from_coupling(&coupling);

    println!("  Asymmetry q_grav = {:.6}", asymmetry.q);

    // Frequency sweep around resonance
    let mut frequencies = Vec::new();
    for i in -50..=50 {
        let freq = coupling.resonance_frequency + (i as f64) * coupling.total_decay_rate();
        frequencies.push(freq);
    }

    let cross_sections = cross_sections::sweep_cross_sections(&coupling, &frequencies, 0.5);

    println!("  Computed {} frequency points", frequencies.len());

    // Find peak scattering
    let max_cs = cross_sections
        .iter()
        .max_by(|a, b| a.c_scattering.partial_cmp(&b.c_scattering).unwrap())
        .unwrap();
    println!("  Peak C_sct = {:.6e}\n", max_cs.c_scattering);

    // ============================================================
    // STEP 3: Integrate TCMT equations in time
    // ============================================================
    println!("STEP 3: Integrate coupled-mode equations from initial state");

    let tcmt_system = tcmt_equations::TCMTSystem::new(coupling);
    let initial_state = TCMTState::new(
        C64::new(1.0, 0.0), // Initial photon amplitude
        C64::new(0.0, 0.0), // No initial graviton
        0.0,                // Start at t=0
    );

    println!(
        "  Initial photon population: {:.6}",
        initial_state.photon_population()
    );
    println!(
        "  Initial graviton population: {:.6}",
        initial_state.graviton_population()
    );
    println!(
        "  Initial total energy: {:.6}",
        initial_state.total_energy()
    );

    // Integrate for time corresponding to several decay periods
    let decay_time = 1.0 / coupling.total_decay_rate();
    let final_time = decay_time * 10.0; // 10 decay times

    let final_state =
        tcmt_equations::integrate_to_time(&tcmt_system, initial_state, final_time, 1000);

    println!("\n  After integration to t = {:.3e}:", final_time);
    println!(
        "  Final photon population: {:.6e}",
        final_state.photon_population()
    );
    println!(
        "  Final graviton population: {:.6e}",
        final_state.graviton_population()
    );
    println!("  Final total energy: {:.6e}", final_state.total_energy());

    // ============================================================
    // STEP 4: Validate constraints
    // ============================================================
    println!("\nSTEP 4: Validate TCMT constraints");

    // Energy conservation in lossless case
    let coupling_lossless = GravitationalCoupling::new(
        coupling.coupling_strength,
        0.0, // No radiative loss
        0.0, // No non-radiative loss
        coupling.gamma_gravitational,
        coupling.resonance_frequency,
        coupling.coupling_phase,
    );

    let tcmt_lossless = tcmt_equations::TCMTSystem::new(coupling_lossless);
    let final_lossless =
        tcmt_equations::integrate_to_time(&tcmt_lossless, initial_state, 1e-12, 500);
    let energy_error =
        tcmt_equations::energy_conservation_error(initial_state, final_lossless, &tcmt_lossless);

    println!(
        "  Energy conservation error (lossless): {:.2e}",
        energy_error
    );
    println!("  Constraint satisfied: {}", energy_error < 0.01);

    // Optical theorem verification
    let cs_resonance = cross_sections::CrossSectionComputer::new(coupling);
    let check = cs_resonance.verify_optical_theorem(coupling.resonance_frequency, 0.5, 1e-10);
    println!("  Optical theorem (C_ext = C_sct + C_abs): {}", check);

    // Maksimov energy conservation constraint
    let check_maksimov = amplitude_bridge::validate_energy_conservation(
        coupling.coupling_strength * 1.0,
        coupling.coupling_strength * 0.5,
        coupling.total_decay_rate(),
        0.2,
    );
    println!(
        r"  Maksimov \kappa_1 - \kappa_2 = \gamma_l constraint: {}",
        check_maksimov
    );
    println!();

    // ============================================================
    // STEP 5: Quality factor and resonance properties
    // ============================================================
    println!("STEP 5: Resonance properties");

    let computer = cross_sections::CrossSectionComputer::new(coupling);
    println!("  Quality factor Q = {:.3e}", computer.quality_factor());
    println!("  Resonance linewidth = {:.3e} Hz", computer.linewidth());
    println!(
        "  Peak scattering cross-section = {:.3e}",
        computer.peak_scattering()
    );

    println!("\n=== Demonstration Complete ===");
}
