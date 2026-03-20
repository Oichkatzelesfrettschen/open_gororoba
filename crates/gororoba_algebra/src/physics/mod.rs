pub mod amplitudes;
pub mod billiard_sim;
pub mod clifford;
pub mod dft_mapping;
pub mod experimental_protocols;
pub mod heterotic_e8;
pub mod laws;
pub mod m3;
pub mod octonion_e8_theory;
pub mod octonion_field;
pub mod pais_superforce;
pub mod quat_rotation;
pub mod quaternionic_qm;
pub mod sedenion_field;
pub mod speculative;
pub mod supersymmetry;
pub mod topological_insulators;
pub mod tripotent_topology;
pub mod crystallography;
#[cfg(all(feature = "physics-sm", feature = "analysis"))]
pub mod particle_physics;


/// Golden ratio constant phi = (1 + sqrt(5))/2.
pub const PHI: f64 = 1.618033988749895;

/// Compute a convergent Fibonacci series: sum_{n=1}^N phi^(-n).
///
/// Refines the divergent sum(phi^n) identified in the Comprehensive Critical Audit
/// by using negative powers to ensure convergence.
pub fn convergent_phi_sum(n_terms: usize) -> f64 {
    let mut sum = 0.0;
    for n in 1..=n_terms {
        sum += PHI.powi(-(n as i32));
    }
    sum
}
