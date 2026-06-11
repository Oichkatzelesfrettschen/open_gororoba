//! Compatibility API for the legacy E8 lattice sieve module.

use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq};

/// Return true when a 16D state is annihilated by the provided E8 filter mask.
pub fn is_e8_root_candidate(data_point: &[f64; 16], e8_filter_mask: &[f64; 16]) -> bool {
    let projection: [f64; 16] = cd_multiply(data_point, e8_filter_mask)
        .try_into()
        .expect("16D Cayley-Dickson multiplication must produce a 16D projection");

    cd_norm_sq(&projection) < 1e-4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_zero_divisor_filter() {
        let mut data = [0.0; 16];
        data[1] = 1.0;
        data[10] = 1.0;

        let mut filter = [0.0; 16];
        filter[15] = 1.0;
        filter[4] = -1.0;

        assert!(is_e8_root_candidate(&data, &filter));
    }
}
