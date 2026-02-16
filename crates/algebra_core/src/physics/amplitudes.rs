/// Single-minus gluon tree amplitude logic.
///
/// Implements the discrete sign-chamber evaluation for the half-collinear single-minus amplitude
/// in region R1, as derived in the 2602.12176 preprint.
///
/// The amplitude A_n is given by:
/// A_n = (1 / 2^(n-2)) * Product_{m=2}^{n-1} ( sg_{m, m+1} + sg_{1, 2...m} )
///
/// Each factor (sg + sg) is either -2, 0, or 2.
/// Divided by 2, it is -1, 0, 1.
/// The product is thus an integer (or 0).
/// The prefactor 1/2^(n-2) exactly cancels the 2^(n-2) from the terms.
///
/// Input:
/// - n: number of gluons
/// - adj_signs: slice of signs s_{m, m+1} for m=2..n-1. Length n-2.
/// - cumul_signs: slice of signs s_{1, 2..m} for m=2..n-1. Length n-2.
///
/// Returns the amplitude value (-1, 0, 1).
pub fn evaluate_r1_closed_form(n: usize, adj_signs: &[i8], cumul_signs: &[i8]) -> i8 {
    if n < 3 {
        return 0; // Amplitude defined for n >= 3 (though formula says m=2..n-1, so for n=3 loop runs m=2..2)
    }
    
    // Check lengths
    if adj_signs.len() != n - 2 || cumul_signs.len() != n - 2 {
        // Invalid input
        return 0;
    }

    let mut result = 1;

    for i in 0..(n - 2) {
        let s_adj = adj_signs[i];
        let s_cum = cumul_signs[i];
        
        let term = s_adj + s_cum;
        
        // Term is -2, 0, or 2.
        // Divide by 2: -1, 0, 1.
        let factor = term / 2;
        
        result *= factor;
        
        if result == 0 {
            return 0;
        }
    }

    result
}

/// Compute sign graph frustration index.
///
/// Measures the inconsistency in the sign assignment.
/// For the amplitude to be non-zero, all factors must be non-zero.
/// This implies s_adj == s_cum for all m.
/// If they differ, the factor is 0.
///
/// Frustration = fraction of zero factors.
pub fn compute_sign_graph_frustration(adj_signs: &[i8], cumul_signs: &[i8]) -> f64 {
    let len = adj_signs.len().min(cumul_signs.len());
    if len == 0 { return 0.0; }
    
    let mut zero_count = 0;
    for i in 0..len {
        if adj_signs[i] + cumul_signs[i] == 0 {
            zero_count += 1;
        }
    }
    
    zero_count as f64 / len as f64
}
