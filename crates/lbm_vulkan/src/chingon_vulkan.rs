//! Chingon AVT tensor contraction on Vulkan.
//!
//! Dimension-parametric bilinear contraction:
//!   `force[m] += alpha * v[i] * v[j] * sign`
//! for each bit-packed violation (i, j, m, sign).
//!
//! Mirrors the API of `lbm_3d_cuda::chingon_gpu`.

use crate::compute::{VulkanEngineError, compile_wgsl};

/// Compile the Chingon AVT WGSL shader to SPIR-V words.
pub fn compile_chingon_avt_shader() -> Result<Vec<u32>, VulkanEngineError> {
    compile_wgsl(include_str!("../shaders/chingon_avt.wgsl"))
}

/// Pack a single AVT violation triple into a u32.
///
/// Layout (LSB to MSB):
///   [m: index_bits] [j: index_bits] [i: index_bits] [sign_positive: 1]
pub fn pack_violation(i: u32, j: u32, m: u32, sign_positive: bool, index_bits: u32) -> u32 {
    let s = if sign_positive { 1u32 } else { 0u32 };
    m | (j << index_bits) | (i << (2 * index_bits)) | (s << (3 * index_bits))
}

/// Compute the number of index bits needed for a given dimension.
pub fn index_bits_for_dim(dim: u32) -> u32 {
    let mut bits = 0u32;
    let mut n = dim - 1;
    while n > 0 {
        bits += 1;
        n >>= 1;
    }
    bits
}

/// Uniform buffer layout for the Chingon AVT shader.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ChingonConstants {
    pub n_violations: u32,
    pub dim: u32,
    pub index_bits: u32,
    pub alpha: f32,
    pub inv_n_viol: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_violation() {
        // 64D: index_bits=6
        let packed = pack_violation(3, 7, 12, true, 6);
        let mask = (1u32 << 6) - 1;
        assert_eq!(packed & mask, 12); // m
        assert_eq!((packed >> 6) & mask, 7); // j
        assert_eq!((packed >> 12) & mask, 3); // i
        assert_eq!((packed >> 18) & 1, 1); // sign
    }

    #[test]
    fn test_index_bits() {
        assert_eq!(index_bits_for_dim(64), 6);
        assert_eq!(index_bits_for_dim(128), 7);
        assert_eq!(index_bits_for_dim(256), 8);
        assert_eq!(index_bits_for_dim(1024), 10);
    }

    #[test]
    fn shader_compiles() {
        compile_chingon_avt_shader().expect("Chingon AVT shader must compile via naga");
    }
}
