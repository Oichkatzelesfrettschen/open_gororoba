//! cubecl `#[cube]` quantize kernel.
//!
//! cubecl 0.10.0 stable (released 2026-05-07) provides the `cubecl`
//! meta-crate with a stable `#[cube]` macro and prelude. This kernel
//! mirrors the algorithm in vulkan/shaders/quantize.comp:
//!   - One thread per input value
//!   - Each thread counts how many boundaries are <= its value
//!   - Writes the per-thread u32 count into indices`[gid]`
//!   - CPU-side wrapper packs u32s into u8 bytes after readback
//!     (avoids cubecl atomic surface for portability across runtimes)

#![cfg(feature = "cubecl")]

use cubecl::prelude::*;

/// Boundary-search quantization kernel. One thread per input value.
///
/// `values[i]` is the f32 input; `boundaries[0..n_boundaries]` are the
/// sorted thresholds (length = 2^bits - 1); `indices[i]` receives the
/// u32 count of how many boundaries are < values`[i]` (the boundary
/// search index).
#[cube(launch_unchecked)]
pub fn quantize_kernel(
    values: &Array<f32>,
    boundaries: &Array<f32>,
    indices: &mut Array<u32>,
    #[comptime] n_boundaries: u32,
) {
    let gid = ABSOLUTE_POS;
    if gid >= values.len() {
        terminate!();
    }
    let v = values[gid];
    let mut count: u32 = 0;
    let n_b = n_boundaries as usize;
    let mut b: usize = 0;
    while b < n_b {
        if v > boundaries[b] {
            count += 1;
        }
        b += 1;
    }
    indices[gid] = count;
}
