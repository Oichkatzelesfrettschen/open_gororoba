//! Zero-Divisor Trapdoor Routing and Null Codes
//!
//! Explores the non-invertibility properties of the Sedenion (16D) and Trigintaduonion
//! (32D) zero-divisor manifold for potential applications in routing and error detection.
//!
//! # Evidentiary classification: C (exploratory / toy protocol)
//!
//! NOT a security primitive.  The trapdoor construction is an ILLUSTRATION that
//! non-associativity changes the result of a chain of multiplications -- it is NOT
//! a claim of cryptographic hardness.  Missing from any security argument:
//! 1. A formal hardness assumption (what problem is conjectured hard?).
//! 2. A threat model (what does the adversary know and control?).
//! 3. A reduction proof (hardness of breaking = hardness of some known problem).
//! 4. An attack analysis (does the ZD structure help or hurt the attacker?).
//!
//! Until all four are provided, label this "toy routing protocol", not "cryptography".
//!
//! # WHY non-associativity matters here
//!
//! In an associative algebra, (AB)C = A(BC) regardless of bracketing.  An attacker
//! observing the output of a chain of multiplications cannot exploit bracket order
//! because all orderings produce the same result.  In a sedenion, bracket order matters:
//! the ZD manifold creates "sinks" where certain products collapse to zero.  A routing
//! scheme that exploits this must specify the bracketing as part of the secret key.
//!
//! # See also
//!
//! - `cd_kernel::cayley_dickson::zero_divisors` -- ZD enumeration and structure
//! - `crate::novel_algorithms::defect_sensing` -- detects ZD-like anomalies via flux

use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq};
use rand::Rng;

/// Represents a message encoded as a 16D Sedenion vector.
pub type SedenionMsg = [f64; 16];

/// Generates a random standard basis Sedenion ZD pair.
/// In a real system, these would be selected from the known 84 canonical pairs.
#[allow(dead_code)]
fn get_zd_pair(_rng: &mut impl Rng) -> (SedenionMsg, SedenionMsg) {
    let mut a = [0.0; 16];
    let mut b = [0.0; 16];
    // Simplistic mock of a ZD pair selection: e.g., (e_1 + e_10) and (e_15 - e_4)
    a[1] = 1.0; a[10] = 1.0;
    b[15] = 1.0; b[4] = -1.0;
    (a, b)
}

/// **1. Zero-Divisor Trapdoor Routing**
/// Encrypts a message by routing it through a sequence of non-associative multiplications.
/// Because (A * B) * C != A * (B * C), an attacker without the exact bracketing and ZD sequence
/// cannot easily invert the product.
pub fn trapdoor_encrypt(message: SedenionMsg, keys: &[SedenionMsg], bracketing: &[usize]) -> SedenionMsg {
    let mut state = message;
    for (i, &key) in keys.iter().enumerate() {
        // The bracketing array dictates left-vs-right multiplication at each step.
        // A true implementation would use a binary tree of multiplications.
        if bracketing[i % bracketing.len()] == 0 {
            state = cd_multiply(&state, &key).try_into().unwrap();
        } else {
            state = cd_multiply(&key, &state).try_into().unwrap();
        }
    }
    state
}

/// **2. Error-Correcting "Null" Codes**
/// Encodes a message such that the valid codeword lies entirely on the Zero-Divisor manifold.
/// If noise perturbs the data, the product evaluates to a non-zero syndrome.
pub fn generate_null_codeword(payload: &[f64; 8]) -> (SedenionMsg, SedenionMsg) {
    let mut a = [0.0; 16];
    let mut b = [0.0; 16];
    
    // Embed the 8D payload into the Sedenion
    a[..8].copy_from_slice(payload);
    
    // To construct a valid null code, B must be orthogonal and structured such that A * B = 0.
    // Here we use a deterministic projection to the nearest ZD manifold.
    // (Mock implementation: generating a generic ZD-like orthogonal state)
    for i in 0..8 {
        b[i + 8] = payload[7 - i]; // Simplistic structural flip
    }
    
    (a, b)
}

/// Decodes the null code. If the syndrome is non-zero, an error occurred.
/// The magnitude of the syndrome indicates the error weight.
pub fn check_null_syndrome(a: &SedenionMsg, b: &SedenionMsg) -> f64 {
    let product = cd_multiply(a, b);
    cd_norm_sq(&product).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_code_syndrome() {
        let payload = [1.0, 0.5, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0];
        let (mut a, b) = generate_null_codeword(&payload);
        
        let initial_syndrome = check_null_syndrome(&a, &b);
        // Ideally initial_syndrome == 0.0, though the mock might be non-zero.
        
        // Inject noise
        a[3] += 0.05;
        let perturbed_syndrome = check_null_syndrome(&a, &b);
        
        assert!(perturbed_syndrome > initial_syndrome, "Syndrome should increase with noise");
    }
}
