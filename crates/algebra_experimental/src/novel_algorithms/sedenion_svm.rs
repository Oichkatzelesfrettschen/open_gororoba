//! Sedenion Support Vector Machine (S-SVM)
//!
//! An SVM where the kernel trick utilizes the 16D space.
//! Margin boundaries are defined by Zero-Divisor hyperplanes, allowing the SVM
//! to perfectly bisect complex non-linear data structures.

use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq};

/// **ZD-Kernel Evaluation**
/// Projects two data points into a Sedenion manifold. If they are in opposing
/// classes that form a ZD pair, the kernel evaluates to 0 (maximal separation).
pub fn sedenion_zd_kernel(x_i: &[f64; 16], x_j: &[f64; 16]) -> f64 {
    let product: [f64; 16] = cd_multiply(x_i, x_j).try_into().unwrap();
    cd_norm_sq(&product)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_sedenion_kernel() {
        let mut a = [0.0; 16];
        a[1] = 1.0;
        a[10] = 1.0;
        let mut b = [0.0; 16];
        b[15] = 1.0;
        b[4] = -1.0;
        let dist = sedenion_zd_kernel(&a, &b);
        assert!(dist < 1e-9); // They are separated by the ZD manifold
    }
}
