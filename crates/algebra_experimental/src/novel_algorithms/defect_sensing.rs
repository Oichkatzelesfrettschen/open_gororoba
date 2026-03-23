//! Associator Flux Topological Defect Sensing
//!
//! Embeds datasets into 16D Sedenion coordinates and scans triplets for non-zero
//! associator flux `||A(a,b,c)|| = ||(ab)c - a(bc)||`.  Non-associativity acts as
//! a structural "defect" signal: zero flux implies associative (or ZD-cancelled) geometry.
//!
//! # Evidentiary classification: C (exploratory conjecture)
//!
//! The "defect sensing" interpretation is speculative.  Zero-divisor pairs in the sedenion
//! CAN produce zero flux even for non-trivial inputs (as the test shows: A(e_1+e_10, e_15-e_4, e_8)
//! = 0 identically because e_1+e_10 and e_15-e_4 form a ZD pair).  The sensor must therefore
//! be calibrated against the ZD structure to avoid false negatives.
//!
//! # WHY associator flux as a defect signal
//!
//! The associator `A(a,b,c) = (ab)c - a(bc)` measures failure of associativity.  For
//! data embedded in a high-dim CD space, the flux `||A||` provides a scalar "roughness"
//! at each position that is:
//! - Zero at real and complex subalgebras (fully associative).
//! - Non-zero at sedenion triples outside the octonion subalgebra.
//! - Zero again at zero-divisor triples (a structural ambiguity this module must handle).
//!
//! # Important: ZD false negatives
//!
//! The test `test_defect_sensing` explicitly avoids the ZD ambiguity by using
//! (e_1, e_2, e_4) which gives A = 2*e_7 (flux = 2.0).  The original mock data
//! (e_1+e_10, e_15-e_4, e_8) was INCORRECT -- it gives zero flux because the first
//! two vectors are a ZD pair.
//!
//! # See also
//!
//! - `cd_kernel::cayley_dickson::associator` -- raw associator computation
//! - `crate::topological_associator_flux` -- flux analysis at higher dimensions (128..1024)

use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq};

/// Evaluates the associator [A, B, C] = (AB)C - A(BC) for 16D coordinates.
pub fn compute_associator(a: &[f64; 16], b: &[f64; 16], c: &[f64; 16]) -> [f64; 16] {
    let ab: [f64; 16] = cd_multiply(a, b).try_into().unwrap();
    let bc: [f64; 16] = cd_multiply(b, c).try_into().unwrap();

    let ab_c: [f64; 16] = cd_multiply(&ab, c).try_into().unwrap();
    let a_bc: [f64; 16] = cd_multiply(a, &bc).try_into().unwrap();
    
    let mut result = [0.0; 16];
    for i in 0..16 {
        result[i] = ab_c[i] - a_bc[i];
    }
    result
}

/// **Topological Defect Sensing**
/// Scans a dataset, treating triplets of consecutive data points as CD coordinates.
/// Computes the associator flux. Since normal associative data yields 0.0 flux,
/// any non-zero, quantized flux reveals a "topological defect" or structural anomaly
/// in the dataset's embedding.
pub fn scan_anomalies_via_flux(dataset: &[[f64; 16]], threshold: f64) -> Vec<usize> {
    let mut anomalies = Vec::new();
    
    // Use a sliding window of 3 to measure local non-associativity
    for i in 0..dataset.len().saturating_sub(2) {
        let a = &dataset[i];
        let b = &dataset[i+1];
        let c = &dataset[i+2];
        
        let assoc = compute_associator(a, b, c);
        let flux = cd_norm_sq(&assoc).sqrt();
        
        if flux > threshold {
            anomalies.push(i + 1); // Mark the center point as the anomaly locus
        }
    }
    
    anomalies
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests that the defect sensor detects a genuinely non-zero associator.
    ///
    /// A(e_1, e_2, e_4) = 2*e_7 in the sedenion (flux = 2.0), verified by:
    ///   (e_1*e_2)*e_4 = e_3*e_4 = e_7    (triads [1,2,3] and [3,4,7])
    ///   e_1*(e_2*e_4) = e_1*e_6 = -e_7   (triad [2,4,6] then [1,7,6] cyclic)
    ///   A = e_7 - (-e_7) = 2*e_7, ||A|| = 2.0
    ///
    /// NOTE: e_1+e_10 and e_15-e_4 form a zero-divisor pair (their product = 0),
    /// so A(e_1+e_10, e_15-e_4, e_8) = 0 identically -- NOT a useful test vector.
    #[test]
    fn test_defect_sensing() {
        let mut dataset = vec![[0.0; 16]; 5];

        // Normal collinear data (real axis only): associator = 0.
        dataset[0][0] = 1.0;  // e_0
        dataset[1][0] = 2.0;  // 2*e_0

        // Anomalous window [2,3,4] = (e_1, e_2, e_4): A = 2*e_7, flux = 2.0.
        dataset[2][1] = 1.0;  // e_1
        dataset[3][2] = 1.0;  // e_2
        dataset[4][4] = 1.0;  // e_4

        let anomalies = scan_anomalies_via_flux(&dataset, 0.1);
        // Window [2,3,4] has flux 2.0 >> 0.1 -- should detect center point 3.
        assert!(!anomalies.is_empty(),
            "Expected non-zero associator A(e_1,e_2,e_4)=2*e_7 to be detected");
        assert!(anomalies.contains(&3),
            "Expected anomaly at center index 3 (window i=2), got {:?}", anomalies);
    }
}
