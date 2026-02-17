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

/// Statistics for an amplitude chamber.
#[derive(Debug, Clone)]
pub struct ChamberStats {
    pub adj_signs: Vec<i8>,
    pub cumul_signs: Vec<i8>,
    pub amplitude: i8,
    pub frustration: f64,
}

/// Enumerates all possible sign combinations for n gluons and returns stats.
/// Total chambers = 2^(2*(n-2)).
pub fn enumerate_chambers(n: usize) -> Vec<ChamberStats> {
    if n < 3 {
        return Vec::new();
    }
    let num_pairs = n - 2;
    let total_chambers = 1 << (2 * num_pairs);

    let mut stats = Vec::with_capacity(total_chambers);

    for i in 0..total_chambers {
        let mut adj = Vec::with_capacity(num_pairs);
        let mut cum = Vec::with_capacity(num_pairs);

        for j in 0..num_pairs {
            // bits 2j and 2j+1
            let adj_bit = (i >> (2 * j)) & 1;
            let cum_bit = (i >> (2 * j + 1)) & 1;

            adj.push(if adj_bit == 1 { 1 } else { -1 });
            cum.push(if cum_bit == 1 { 1 } else { -1 });
        }

        let amplitude = evaluate_r1_closed_form(n, &adj, &cum);
        let frustration = compute_sign_graph_frustration(&adj, &cum);

        stats.push(ChamberStats {
            adj_signs: adj,
            cumul_signs: cum,
            amplitude,
            frustration,
        });
    }

    stats
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
    if len == 0 {
        return 0.0;
    }

    let mut zero_count = 0;
    for i in 0..len {
        if adj_signs[i] + cumul_signs[i] == 0 {
            zero_count += 1;
        }
    }

    zero_count as f64 / len as f64
}
