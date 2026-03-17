//! Box-counting fractal dimension on Vulkan.
//!
//! Dispatches `box_count_at_scale` shader per scale, reads back atomic counter,
//! then computes D_f via log-log regression on (log(1/box_size), log(count)).
//!
//! Mirrors the API of `lbm_3d_cuda::box_counting_gpu::GpuBoxCounter`.

use crate::compute::{compile_wgsl, VulkanEngineError};

/// Compile the box-counting WGSL shader to SPIR-V words.
pub fn compile_box_counting_shader() -> Result<Vec<u32>, VulkanEngineError> {
    compile_wgsl(include_str!("../shaders/box_counting.wgsl"))
}

/// Compile the zero-counter WGSL shader to SPIR-V words.
pub fn compile_zero_counter_shader() -> Result<Vec<u32>, VulkanEngineError> {
    compile_wgsl(include_str!("../shaders/zero_counter.wgsl"))
}

/// Compute fractal dimension from (box_size, count) pairs via least-squares
/// linear regression in log-log space: D_f = -slope of log(count) vs log(box_size).
pub fn fractal_dimension_from_counts(pairs: &[(u32, u32)]) -> Option<f64> {
    let valid: Vec<(f64, f64)> = pairs
        .iter()
        .filter(|(_, c)| *c > 0)
        .map(|(s, c)| ((*s as f64).ln(), (*c as f64).ln()))
        .collect();
    if valid.len() < 2 {
        return None;
    }
    let n = valid.len() as f64;
    let sx: f64 = valid.iter().map(|(x, _)| x).sum();
    let sy: f64 = valid.iter().map(|(_, y)| y).sum();
    let sxy: f64 = valid.iter().map(|(x, y)| x * y).sum();
    let sx2: f64 = valid.iter().map(|(x, _)| x * x).sum();
    let denom = n * sx2 - sx * sx;
    if denom.abs() < 1e-30 {
        return None;
    }
    let slope = (n * sxy - sx * sy) / denom;
    // D_f = -slope (since count ~ box_size^{-D_f})
    Some(-slope)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractal_dimension_trivial() {
        // A filled 3D cube: count ~ (N/s)^3, so D_f = 3.0
        let pairs = vec![
            (1, 64 * 64 * 64),
            (2, 32 * 32 * 32),
            (4, 16 * 16 * 16),
            (8, 8 * 8 * 8),
            (16, 4 * 4 * 4),
            (32, 2 * 2 * 2),
        ];
        let df = fractal_dimension_from_counts(&pairs).unwrap();
        assert!((df - 3.0).abs() < 0.01, "expected D_f ~ 3.0, got {df}");
    }

    #[test]
    fn shader_compiles() {
        compile_box_counting_shader().expect("box-counting shader must compile");
        compile_zero_counter_shader().expect("zero-counter shader must compile");
    }
}
