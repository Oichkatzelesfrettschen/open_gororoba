//! GPU packing extension for AlternativityViolationTensor.
//!
//! Separates GPU layout concerns from the algebra analysis layer.
//! Consumers (lbm_3d_cuda, gororoba_engine) import this trait and call
//! `avt.pack_for_gpu()` to obtain a [`PackedAvt`] suitable for SSBO / CUDA
//! device-buffer upload.
//!
//! # Bit layout
//!
//! For dimension D, each violation is packed into one `u32`:
//!   `[m: bits] [j: bits] [i: bits] [sign_positive: 1]`
//! where `bits = ceil(log2(D))`.
//!
//! | Dim  | bits | total bits used |
//! |------|------|-----------------|
//! | 64   |  6   | 19              |
//! | 128  |  7   | 22              |
//! | 256  |  8   | 25              |
//! | 512  |  9   | 28              |
//! | 1024 | 10   | 31              |

use algebra_analysis::avt::{AlternativityViolationTensor, PackedAvt, index_bits_for_dim};

/// Extension trait: pack an [`AlternativityViolationTensor`] into GPU-ready words.
///
/// # Why an extension trait instead of a method?
///
/// `AlternativityViolationTensor` lives in `algebra_analysis`, a crate that
/// must remain free of GPU dependencies.  Putting `pack_for_gpu` directly on
/// the type would pull `cudarc` / `wgpu` into the analysis layer, breaking the
/// dependency ordering (cd_kernel -> algebra_analysis -> gororoba_algebra).
///
/// The extension trait pattern solves this cleanly: the analysis crate exports
/// the type, the GPU layer imports it and bolts on the packing method.  Consumers
/// import both and call `avt.pack_for_gpu()` with no visible seam.
///
/// # Bit layout
///
/// See the module-level documentation for the per-dimension bit width table.
pub trait GpuPackableAvt {
    /// Pack all violations into `u32` words for SSBO / CUDA device buffer upload.
    fn pack_for_gpu(&self) -> PackedAvt;
}

impl GpuPackableAvt for AlternativityViolationTensor {
    fn pack_for_gpu(&self) -> PackedAvt {
        let bits = index_bits_for_dim(self.dim);
        assert!(
            3 * bits < 32,
            "dim {} requires {}*3+1={} bits, exceeds u32",
            self.dim,
            bits,
            3 * bits + 1
        );

        let mask = (1u32 << bits) - 1;
        let mut packed = Vec::with_capacity(self.violations.len());

        for &(i, j, _k, m, sign) in &self.violations {
            let sign_bit: u32 = if sign > 0 { 1 } else { 0 };
            let word = (m as u32 & mask)
                | ((j as u32 & mask) << bits)
                | ((i as u32 & mask) << (2 * bits))
                | (sign_bit << (3 * bits));
            packed.push(word);
        }

        PackedAvt {
            data: packed,
            index_bits: bits as u32,
            dim: self.dim as u32,
            violation_count: self.violations.len() as u32,
        }
    }
}
