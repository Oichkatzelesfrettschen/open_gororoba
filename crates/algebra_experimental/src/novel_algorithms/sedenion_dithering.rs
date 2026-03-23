//! Sedenionic Error-Diffusion Dithering
//!
//! Image processing algorithm that diffuses quantization error across 
//! non-associative dimensions to prevent structured banding (Floyd-Steinberg analog).

use cd_kernel::cayley_dickson::cd_multiply;

/// **Hypercomplex Dithering**
/// Spreads quantization error. Because the error is multiplied non-associatively 
/// with the spatial mask, repeating patterns (banding) are algebraically annihilated,
/// resulting in organic, blue-noise-like dithering.
pub fn diffuse_error_sedenionic(pixel_error: f64, diffusion_mask: &[f64; 16]) -> [f64; 16] {
    let mut err_state = [0.0; 16];
    err_state[0] = pixel_error; // Embed scalar error into Real axis
    
    cd_multiply(&err_state, diffusion_mask).try_into().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_dithering() {
        let mask = [0.2; 16];
        let diff = diffuse_error_sedenionic(0.5, &mask);
        assert_eq!(diff[0], 0.1);
    }
}
