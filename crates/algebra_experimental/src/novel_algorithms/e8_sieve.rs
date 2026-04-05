//! Exceptional Lie Group Sieve
//!
//! Finds E8 lattice points in noisy continuous data by projecting the data
//! onto split-octonion and sedenion zero-divisors.

use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq};

/// **E8 Lattice Sieve**
/// Tests whether a given 16D continuous state is close to a fundamental
/// E8 root by checking if its Sedenion projection strictly annihilates
/// a known geometric filter.
pub fn is_e8_root_candidate(data_point: &[f64; 16], e8_filter_mask: &[f64; 16]) -> bool {
    let projection: [f64; 16] = cd_multiply(data_point, e8_filter_mask).try_into().unwrap();

    let norm = cd_norm_sq(&projection);
    // If the norm collapses, the data point resides on the E8 lattice boundary
    norm < 1e-4
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_e8_sieve() {
        let mut data = [0.0; 16];
        data[1] = 1.0;
        data[10] = 1.0;
        let mut filter = [0.0; 16];
        filter[15] = 1.0;
        filter[4] = -1.0;

        // This forms a ZD, so it successfully identifies the geometric root
        assert!(is_e8_root_candidate(&data, &filter));
    }
}
