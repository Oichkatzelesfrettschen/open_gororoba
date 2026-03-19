//! Quaternionic Quantum Mechanics vs Standard Quantum Mechanics.
//!
//! Encodes the reconciled conflict: Quaternions CAN formulate QM, but their 
//! predictions differ from standard complex QM, particularly in interference 
//! effects and phase commutativity.
//!
//! # References
//! - Adler, S. L. (1995). Quaternionic Quantum Mechanics and Quantum Fields.

use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq};

/// Evaluates a basic interference pattern between two state vectors.
///
/// In standard QM (Complex numbers), phase factors commute. 
/// In Quaternionic QM, phase factors do not commute, leading to observable 
/// differences in multi-path interference experiments (e.g., neutron interferometry).
pub fn evaluate_interference(
    psi1: &[f64], 
    psi2: &[f64], 
    phase1: &[f64], 
    phase2: &[f64]
) -> (f64, f64) {
    // Standard Complex QM (using the first 2 components: real + i)
    let c_psi1 = vec![psi1[0], psi1[1]];
    let c_psi2 = vec![psi2[0], psi2[1]];
    let c_phase1 = vec![phase1[0], phase1[1]];
    let c_phase2 = vec![phase2[0], phase2[1]];

    // state = phase1 * psi1 + phase2 * psi2
    let c_term1 = cd_multiply(&c_phase1, &c_psi1);
    let c_term2 = cd_multiply(&c_phase2, &c_psi2);
    let c_state = vec![c_term1[0] + c_term2[0], c_term1[1] + c_term2[1]];
    let complex_probability = cd_norm_sq(&c_state);

    // Quaternionic QM (using 4 components: real + i + j + k)
    let h_psi1 = vec![psi1[0], psi1[1], psi1[2], psi1[3]];
    let h_psi2 = vec![psi2[0], psi2[1], psi2[2], psi2[3]];
    let h_phase1 = vec![phase1[0], phase1[1], phase1[2], phase1[3]];
    let h_phase2 = vec![phase2[0], phase2[1], phase2[2], phase2[3]];

    // state = phase1 * psi1 + phase2 * psi2
    let h_term1 = cd_multiply(&h_phase1, &h_psi1);
    let h_term2 = cd_multiply(&h_phase2, &h_psi2);
    let h_state = vec![
        h_term1[0] + h_term2[0], 
        h_term1[1] + h_term2[1], 
        h_term1[2] + h_term2[2], 
        h_term1[3] + h_term2[3]
    ];
    let quaternionic_probability = cd_norm_sq(&h_state);

    (complex_probability, quaternionic_probability)
}

/// Computes the commutator [A, B] = A*B - B*A.
/// In standard QM (complex numbers), scalar phases commute: [A, B] = 0.
/// In Quaternionic QM, scalar phases (if they involve j, k) do not commute.
pub fn phase_commutator(a: &[f64], b: &[f64]) -> Vec<f64> {
    let dim = a.len();
    let ab = cd_multiply(a, b);
    let ba = cd_multiply(b, a);
    
    let mut commutator = vec![0.0; dim];
    for i in 0..dim {
        commutator[i] = ab[i] - ba[i];
    }
    commutator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quaternionic_vs_complex_interference() {
        let psi1 = vec![1.0, 0.0, 0.0, 0.0]; // Real state 1
        let psi2 = vec![0.0, 1.0, 0.0, 0.0]; // "i" state 1
        
        // Complex phase (real + i)
        let phase1 = vec![0.0, 1.0, 0.0, 0.0]; 
        // Quaternionic phase (real + j)
        let phase2 = vec![0.0, 0.0, 1.0, 0.0]; 

        let (c_prob, h_prob) = evaluate_interference(&psi1, &psi2, &phase1, &phase2);
        
        // Since phase2 is purely 'j', in the complex projection it appears as 0, 
        // altering the interference sum completely. This demonstrates the structural difference.
        assert!((c_prob - h_prob).abs() > 1e-10);
    }

    #[test]
    fn test_phase_commutativity() {
        // Complex phases commute
        let c1 = vec![0.0, 1.0];
        let c2 = vec![0.5, 0.5];
        let c_comm = phase_commutator(&c1, &c2);
        assert!(cd_norm_sq(&c_comm) < 1e-10);

        // Quaternionic phases do not commute
        let h1 = vec![0.0, 1.0, 0.0, 0.0]; // i
        let h2 = vec![0.0, 0.0, 1.0, 0.0]; // j
        let h_comm = phase_commutator(&h1, &h2);
        
        // [i, j] = ij - ji = k - (-k) = 2k
        assert!(cd_norm_sq(&h_comm) > 1e-10);
    }
}
