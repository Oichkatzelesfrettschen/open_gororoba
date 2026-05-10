//! GPU-accelerated box-counting fractal dimension.
//!
//! Compiles the `kernels_box_counting.cu` kernel once via NVRTC and reuses
//! the compiled module across 1000+ galaxies.  Each `fractal_dimension` call
//! dispatches one kernel per scale (typically 5-7 at 64^3), reading density
//! directly from device memory when used with `LbmSolver3DCuda`.
//!
//! # Algorithm
//!
//! For each power-of-2 box size from 1 to min(nx,ny,nz)/2:
//!
//!   1. Zero the atomic counter (1 u32).
//!   2. Launch `box_count_at_scale` with ceil(total_boxes / 256) blocks.
//!   3. Read back the count (single u32).
//!
//! Then fit D_f = -slope of log-log regression on (box_size, count).
//!
//! Warp-level ballot reduction (from `kernels_dark_halo.cu` pattern) reduces
//! global atomics by 32x compared to naive per-thread atomicAdd.

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

const KERNEL_BOX_COUNTING_SRC: &str = include_str!("kernels_box_counting.cu");

/// Result of a GPU box-counting fractal dimension measurement.
#[derive(Debug, Clone)]
pub struct BoxCountingResult {
    /// Fractal dimension D_f (negative slope of log-log regression).
    pub d_f: f64,
    /// R^2 of the log-log linear fit.
    pub r_squared: f64,
    /// Box sizes used (power-of-2 sequence).
    pub box_sizes: Vec<usize>,
    /// Occupied box counts at each scale.
    pub counts: Vec<u64>,
}

/// GPU box-counting engine.  Compile once, reuse across many galaxies.
pub struct GpuBoxCounter {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    zero_kernel: CudaFunction,
    reduce_minmax_kernel: CudaFunction,
    build_histogram_kernel: CudaFunction,
    zero_histogram_kernel: CudaFunction,
    d_count: CudaSlice<u32>,
    d_histogram: CudaSlice<u32>,
    d_min_buf: CudaSlice<f32>,
    d_max_buf: CudaSlice<f32>,
}

impl GpuBoxCounter {
    /// Compile box-counting kernels via NVRTC. Call once per session.
    ///
    /// Includes box-counting, histogram, and min/max reduction kernels
    /// for fully GPU-resident D_f computation (no density readback needed).
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let stream = ctx.default_stream();
        let ptx = cudarc::nvrtc::compile_ptx(KERNEL_BOX_COUNTING_SRC)
            .context("NVRTC compilation of box-counting kernels")?;
        let module = ctx
            .load_module(ptx)
            .context("Load box-counting CUDA module")?;

        let kernel = module
            .load_function("box_count_at_scale")
            .context("Load box_count_at_scale")?;
        let zero_kernel = module.load_function("zero_u32").context("Load zero_u32")?;
        let reduce_minmax_kernel = module
            .load_function("reduce_minmax_f32")
            .context("Load reduce_minmax_f32")?;
        let build_histogram_kernel = module
            .load_function("build_histogram_f32")
            .context("Load build_histogram_f32")?;
        let zero_histogram_kernel = module
            .load_function("zero_histogram")
            .context("Load zero_histogram")?;

        let d_count = stream.alloc_zeros::<u32>(1).context("Alloc d_count")?;
        // 256-bin histogram for Otsu threshold
        let d_histogram = stream
            .alloc_zeros::<u32>(256)
            .context("Alloc d_histogram")?;
        // Min/max reduction buffers (max 1024 blocks)
        let d_min_buf = stream.alloc_zeros::<f32>(1024).context("Alloc d_min_buf")?;
        let d_max_buf = stream.alloc_zeros::<f32>(1024).context("Alloc d_max_buf")?;

        Ok(Self {
            stream,
            kernel,
            zero_kernel,
            reduce_minmax_kernel,
            build_histogram_kernel,
            zero_histogram_kernel,
            d_count,
            d_histogram,
            d_min_buf,
            d_max_buf,
        })
    }

    /// Create a box-counting engine on a caller-supplied stream.
    ///
    /// Used for multi-stream pipelines where box-counting runs concurrently
    /// with LBM on a separate CUDA stream.
    pub fn new_with_stream(ctx: &Arc<CudaContext>, stream: Arc<CudaStream>) -> Result<Self> {
        let ptx = cudarc::nvrtc::compile_ptx(KERNEL_BOX_COUNTING_SRC)
            .context("NVRTC compilation of box-counting kernels")?;
        let module = ctx
            .load_module(ptx)
            .context("Load box-counting CUDA module")?;

        let kernel = module
            .load_function("box_count_at_scale")
            .context("Load box_count_at_scale")?;
        let zero_kernel = module.load_function("zero_u32").context("Load zero_u32")?;
        let reduce_minmax_kernel = module
            .load_function("reduce_minmax_f32")
            .context("Load reduce_minmax_f32")?;
        let build_histogram_kernel = module
            .load_function("build_histogram_f32")
            .context("Load build_histogram_f32")?;
        let zero_histogram_kernel = module
            .load_function("zero_histogram")
            .context("Load zero_histogram")?;

        let d_count = stream.alloc_zeros::<u32>(1).context("Alloc d_count")?;
        let d_histogram = stream
            .alloc_zeros::<u32>(256)
            .context("Alloc d_histogram")?;
        let d_min_buf = stream.alloc_zeros::<f32>(1024).context("Alloc d_min_buf")?;
        let d_max_buf = stream.alloc_zeros::<f32>(1024).context("Alloc d_max_buf")?;

        Ok(Self {
            stream,
            kernel,
            zero_kernel,
            reduce_minmax_kernel,
            build_histogram_kernel,
            zero_histogram_kernel,
            d_count,
            d_histogram,
            d_min_buf,
            d_max_buf,
        })
    }

    /// Compute D_f from device-resident FP32 density (zero-copy path).
    ///
    /// `d_rho_bytes` is the raw device buffer (FP32 format, nx*ny*nz floats).
    /// The threshold is the median density (computed on host from a readback,
    /// or passed explicitly).
    pub fn fractal_dimension_device(
        &mut self,
        d_rho_bytes: &CudaSlice<u8>,
        threshold: f32,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Result<BoxCountingResult> {
        let min_dim = nx.min(ny).min(nz);
        let mut box_sizes = Vec::new();
        let mut counts = Vec::new();

        let mut box_size = 1usize;
        while box_size <= min_dim / 2 {
            let bx_count = nx.div_ceil(box_size) as i32;
            let by_count = ny.div_ceil(box_size) as i32;
            let bz_count = nz.div_ceil(box_size) as i32;
            let total_boxes = (bx_count as u32) * (by_count as u32) * (bz_count as u32);

            // Zero counter
            let mut bz = self.stream.launch_builder(&self.zero_kernel);
            bz.arg(&mut self.d_count);
            unsafe { bz.launch(LaunchConfig::for_num_elems(1)) }?;

            // Launch box-counting kernel
            let threads = 256u32;
            let blocks = total_boxes.div_ceil(threads);
            let config = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };

            let nx_i = nx as i32;
            let ny_i = ny as i32;
            let nz_i = nz as i32;
            let bs_i = box_size as i32;

            let mut b = self.stream.launch_builder(&self.kernel);
            b.arg(d_rho_bytes)
                .arg(&mut self.d_count)
                .arg(&threshold)
                .arg(&nx_i)
                .arg(&ny_i)
                .arg(&nz_i)
                .arg(&bs_i)
                .arg(&bx_count)
                .arg(&by_count)
                .arg(&bz_count);
            unsafe { b.launch(config) }?;

            // Read back count
            let result = self.stream.clone_dtoh(&self.d_count)?;
            let count = result[0] as u64;

            if count > 0 {
                box_sizes.push(box_size);
                counts.push(count);
            }

            box_size *= 2;
        }

        Ok(Self::compute_regression(&box_sizes, &counts))
    }

    /// Compute Otsu threshold entirely on GPU, reading back only 1 KB histogram.
    ///
    /// Returns the Otsu threshold as f32. This eliminates the need to
    /// sync the full density field (8 MB at 128^3) to CPU.
    pub fn otsu_threshold_device(
        &mut self,
        d_rho_bytes: &CudaSlice<u8>,
        n_cells: usize,
    ) -> Result<f32> {
        let n = n_cells as i32;

        // Pass 1: find min and max via parallel reduction
        let threads = 256u32;
        let blocks = ((n_cells as u32).div_ceil(threads)).min(1024);
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut b = self.stream.launch_builder(&self.reduce_minmax_kernel);
        b.arg(d_rho_bytes)
            .arg(&mut self.d_min_buf)
            .arg(&mut self.d_max_buf)
            .arg(&n);
        unsafe { b.launch(config) }?;

        // Read back per-block results and reduce on CPU (<=1024 floats = 8 KB,
        // negligible vs the 8 MB density readback we are eliminating)
        let min_vals = self.stream.clone_dtoh(&self.d_min_buf)?;
        let max_vals = self.stream.clone_dtoh(&self.d_max_buf)?;

        let vmin = min_vals
            .iter()
            .take(blocks as usize)
            .copied()
            .fold(f32::INFINITY, f32::min);
        let vmax = max_vals
            .iter()
            .take(blocks as usize)
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        if (vmax - vmin) < 1e-20 {
            return Ok(vmin); // Uniform field
        }

        // Pass 3: build 256-bin histogram on GPU
        let mut bz = self.stream.launch_builder(&self.zero_histogram_kernel);
        bz.arg(&mut self.d_histogram);
        unsafe { bz.launch(LaunchConfig::for_num_elems(256)) }?;

        let mut bh = self.stream.launch_builder(&self.build_histogram_kernel);
        bh.arg(d_rho_bytes)
            .arg(&mut self.d_histogram)
            .arg(&vmin)
            .arg(&vmax)
            .arg(&n);
        unsafe { bh.launch(config) }?;

        // Read back 256 u32 histogram (1 KB -- 8000x smaller than density field)
        let hist = self.stream.clone_dtoh(&self.d_histogram)?;

        // Otsu's method on CPU from histogram (O(256), negligible)
        Ok(Self::otsu_from_histogram(&hist, vmin, vmax))
    }

    /// Compute D_f fully on GPU: Otsu threshold + box-counting, no density readback.
    ///
    /// Only copies back 1 KB histogram + 7 box-counts (~28 bytes).
    /// At 128^3, this saves 8 MB of PCIe transfer per galaxy.
    pub fn fractal_dimension_device_auto(
        &mut self,
        d_rho_bytes: &CudaSlice<u8>,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Result<BoxCountingResult> {
        let n_cells = nx * ny * nz;
        let threshold = self.otsu_threshold_device(d_rho_bytes, n_cells)?;
        self.fractal_dimension_device(d_rho_bytes, threshold, nx, ny, nz)
    }

    /// Otsu's method from a 256-bin histogram. O(256) on CPU.
    fn otsu_from_histogram(hist: &[u32], vmin: f32, vmax: f32) -> f32 {
        let n_bins = hist.len();
        let total: u64 = hist.iter().map(|&h| h as u64).sum();
        if total == 0 {
            return vmin;
        }

        // Bin centers
        let bin_width = (vmax - vmin) / n_bins as f32;

        // Weighted sum of all bins
        let mut sum_total = 0.0_f64;
        for (i, &h) in hist.iter().enumerate() {
            sum_total += (i as f64) * (h as f64);
        }

        let mut sum_bg = 0.0_f64;
        let mut weight_bg = 0_u64;
        let mut max_variance = 0.0_f64;
        let mut best_t = 0usize;

        for (t, &h) in hist.iter().enumerate() {
            weight_bg += h as u64;
            if weight_bg == 0 {
                continue;
            }
            let weight_fg = total - weight_bg;
            if weight_fg == 0 {
                break;
            }

            sum_bg += (t as f64) * (h as f64);
            let mean_bg = sum_bg / weight_bg as f64;
            let mean_fg = (sum_total - sum_bg) / weight_fg as f64;

            let between_var = (weight_bg as f64) * (weight_fg as f64) * (mean_bg - mean_fg).powi(2);

            if between_var > max_variance {
                max_variance = between_var;
                best_t = t;
            }
        }

        // Convert bin index to density value
        vmin + (best_t as f32 + 0.5) * bin_width
    }

    /// Compute D_f from host-side FP32 density (upload path).
    pub fn fractal_dimension(
        &mut self,
        rho: &[f32],
        threshold: f32,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Result<BoxCountingResult> {
        // Upload density as raw bytes (FP32)
        let bytes: Vec<u8> = rho.iter().flat_map(|v| v.to_le_bytes()).collect();
        let d_rho = self.stream.clone_htod(&bytes)?;
        self.fractal_dimension_device(&d_rho, threshold, nx, ny, nz)
    }

    /// Log-log linear regression: D_f = -slope, plus R^2.
    fn compute_regression(box_sizes: &[usize], counts: &[u64]) -> BoxCountingResult {
        if box_sizes.len() < 2 {
            return BoxCountingResult {
                d_f: 3.0,
                r_squared: 0.0,
                box_sizes: box_sizes.to_vec(),
                counts: counts.to_vec(),
            };
        }

        let log_eps: Vec<f64> = box_sizes.iter().map(|&s| (s as f64).ln()).collect();
        let log_n: Vec<f64> = counts.iter().map(|&c| (c as f64).ln()).collect();

        let n = log_eps.len() as f64;
        let sum_x: f64 = log_eps.iter().sum();
        let sum_y: f64 = log_n.iter().sum();
        let sum_xy: f64 = log_eps.iter().zip(log_n.iter()).map(|(x, y)| x * y).sum();
        let sum_xx: f64 = log_eps.iter().map(|x| x * x).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        let d_f = if denom.abs() < 1e-30 {
            3.0
        } else {
            let slope = (n * sum_xy - sum_x * sum_y) / denom;
            -slope
        };

        // R^2
        let mean_y = sum_y / n;
        let ss_tot: f64 = log_n.iter().map(|y| (y - mean_y).powi(2)).sum();
        let intercept = (sum_y - (-d_f) * sum_x) / n;
        let ss_res: f64 = log_eps
            .iter()
            .zip(log_n.iter())
            .map(|(x, y)| {
                let y_pred = intercept + (-d_f) * x;
                (y - y_pred).powi(2)
            })
            .sum();
        let r_squared = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            1.0
        };

        BoxCountingResult {
            d_f,
            r_squared,
            box_sizes: box_sizes.to_vec(),
            counts: counts.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // GPU tests are gated with #[ignore] -- require CUDA hardware
    #[test]
    #[ignore = "gpu (CUDA hardware required)"]
    fn test_gpu_box_counting_uniform() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let mut counter = GpuBoxCounter::new(&ctx).expect("GpuBoxCounter");
        let n = 16usize;
        let rho = vec![2.0_f32; n * n * n];
        let result = counter
            .fractal_dimension(&rho, 1.0, n, n, n)
            .expect("fractal_dimension");
        assert!(
            (result.d_f - 3.0).abs() < 0.2,
            "Uniform D_f={:.4}, expected ~3.0",
            result.d_f
        );
    }

    #[test]
    #[ignore = "gpu (CUDA hardware required)"]
    fn test_gpu_box_counting_single_cell() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let mut counter = GpuBoxCounter::new(&ctx).expect("GpuBoxCounter");
        let n = 16usize;
        let mut rho = vec![0.0_f32; n * n * n];
        let c = n / 2;
        rho[c * n * n + c * n + c] = 10.0;
        let result = counter
            .fractal_dimension(&rho, 0.5, n, n, n)
            .expect("fractal_dimension");
        assert!(
            result.d_f < 1.0,
            "Single cell D_f={:.4}, expected < 1.0",
            result.d_f
        );
    }

    #[test]
    #[ignore = "gpu (CUDA hardware required)"]
    fn test_gpu_box_counting_filled_plane() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let mut counter = GpuBoxCounter::new(&ctx).expect("GpuBoxCounter");
        let n = 16usize;
        let mut rho = vec![0.0_f32; n * n * n];
        let cz = n / 2;
        for iy in 0..n {
            for ix in 0..n {
                rho[cz * n * n + iy * n + ix] = 5.0;
            }
        }
        let result = counter
            .fractal_dimension(&rho, 0.5, n, n, n)
            .expect("fractal_dimension");
        assert!(
            result.d_f > 1.5 && result.d_f < 2.5,
            "Plane D_f={:.4}, expected ~2.0",
            result.d_f
        );
    }

    #[test]
    #[ignore = "gpu (CUDA hardware required)"]
    fn test_gpu_cpu_agreement() {
        use cosmology_core::sersic::{box_counting_fractal_dim_threshold, otsu_threshold_f32};

        let ctx = CudaContext::new(0).expect("CUDA context");
        let mut counter = GpuBoxCounter::new(&ctx).expect("GpuBoxCounter");
        let n = 16usize;
        let rho: Vec<f32> = (0..(n * n * n)).map(|i| ((i % 7) as f32) * 0.3).collect();

        // Compute Otsu threshold once on CPU -- same threshold for both paths
        let threshold = otsu_threshold_f32(&rho);

        // GPU box-counting with Otsu threshold
        let gpu_result = counter
            .fractal_dimension(&rho, threshold, n, n, n)
            .expect("fractal_dimension");

        // CPU box-counting with same Otsu threshold
        let rho_f64: Vec<f64> = rho.iter().map(|&v| v as f64).collect();
        let cpu_df = box_counting_fractal_dim_threshold(&rho_f64, n, n, n, threshold as f64);

        assert!(
            (gpu_result.d_f - cpu_df).abs() < 0.05,
            "GPU D_f={:.4} vs CPU D_f={:.4}, delta > 0.05",
            gpu_result.d_f,
            cpu_df
        );
    }

    #[test]
    #[ignore = "gpu (CUDA hardware required)"]
    fn test_gpu_empty_field_graceful() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let mut counter = GpuBoxCounter::new(&ctx).expect("GpuBoxCounter");
        let n = 8usize;
        let rho = vec![0.0_f32; n * n * n];
        let result = counter
            .fractal_dimension(&rho, 0.5, n, n, n)
            .expect("fractal_dimension");
        // All zeros below threshold: no occupied boxes, fallback D_f = 3.0
        assert!(
            (result.d_f - 3.0).abs() < 0.1,
            "Empty field D_f={:.4}, expected 3.0 fallback",
            result.d_f
        );
    }
}
