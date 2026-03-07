use gororoba_engine::singularitarian::SingularitarianEngine;
use gr_core::{NBodySystem, BodyState};
use nalgebra::{Vector3, Matrix3};
use num_complex::Complex;

fn main() {
    println!("--- Grand Unification: Singularitarian Ascent (Sprints 74-80) ---");

    // 1. Initialize the Engine with Fractal Dimension Df = 2.7
    let mut engine = SingularitarianEngine::new(2.7);
    println!("Engine initialized with fractal dimension Df = 2.7");

    // 2. Predict Sgr A* Hawking Radiation Spectrum
    println!("\nPredicting Hawking Radiation Spectrum for Sagittarius A*...");
    let spectrum = engine.predict_sgr_a_spectrum();

    println!("Spectrum generated ({} frequency bins).", spectrum.flux.len());
    println!("Hawking temperature: {:.4e} K", spectrum.hawking_temp_k);
    println!("Fractal-modified temperature: {:.4e} K", spectrum.fractal_temp_k);
    println!("Peak flux normalized: {:.4}", spectrum.flux.iter().fold(0.0, |a, &b| f64::max(a, b)));

    // 3. Demonstrate Unified Step in Complex Time
    println!("\nExecuting unified N-Body step in 2D Complex Time...");
    let mut system = NBodySystem::new(1e-5, Matrix3::identity());

    // Add a test body near the event horizon
    system.bodies.push(BodyState {
        id: 1,
        mass: 1.0,
        pos: Vector3::new(Complex::new(10.0, 0.1), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)),
        vel: Vector3::new(Complex::new(0.0, 0.0), Complex::new(0.5, 0.01), Complex::new(0.0, 0.0)),
    });

    let d_tau = Complex::new(0.01, 0.001); // Real step + Imaginary tunneling
    engine.unified_step(&mut system, d_tau);

    let new_pos = system.bodies[0].pos;
    println!("Body 1 moved to position: ({:.4}, {:.4}, {:.4})", new_pos.x, new_pos.y, new_pos.z);
    println!("Complex phase component (tunneling): {:.4}", new_pos.x.im);

    // 4. Evidence Matrix
    let matrix = SingularitarianEngine::evidence_matrix();
    let verified = matrix.iter().filter(|e| e.status == gororoba_engine::singularitarian::ClaimStatus::Verified).count();
    println!("\nEvidence Matrix: {}/{} claims verified across Sprints 71-79", verified, matrix.len());

    println!("\n--- Ascent Complete ---");
}
