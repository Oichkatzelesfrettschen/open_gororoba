//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/reports_narratives.toml -->
//!
//! # x87 Strategy Benchmark Summary
//!
//! ## Run Context
//!
//! - Host: `x570-5600X3D`
//! - CPU: `AMD Ryzen 5 5600X3D 6-Core Processor`
//! - Problem size: `len=65536`
//! - Repeats per row: `7`
//! - RNG seed: `42`
//! - Detected physical-core workers: `6`
//! - Worker sweep: `1,2,4,6`
//! - Stability heuristic: rows are marked unstable when `worst_ns > 5 * median_ns` or `best_ns * 2 < median_ns`.
//!
//! ## dot_ill_conditioned
//!
//! Recommendations:
//!
//! - Fastest overall: `serial_avx2` with 1 worker(s), 0.010 ms, abs_err 4.19430e6, ulp 4706261610602168320
//! - Fastest exact: `avx2_per_chunk` with 2 worker(s), 0.029 ms, abs_err 0.00000e0, ulp 0
//! - Fastest exact parallel lane: `avx2_per_chunk` with 2 worker(s), 0.029 ms, abs_err 0.00000e0, ulp 0
//! - Fastest <=1 ULP parallel lane: `avx2_per_chunk` with 2 worker(s), 0.029 ms, abs_err 0.00000e0, ulp 0
//!
//! | Strategy | Workers | Median ms | Speedup vs serial x87 | Abs error | ULP error | Stable |
//! |---|---:|---:|---:|---:|---:|---|
//! | serial_naive | 1 | 0.046 | 1.151 | 1.57286e6 | 4699506211161112576 | yes |
//! | serial_kahan | 1 | 0.181 | 0.290 | 0.00000e0 | 0 | yes |
//! | serial_x87 | 1 | 0.053 | 1.000 | 0.00000e0 | 0 | yes |
//! | serial_avx2 | 1 | 0.010 | 5.423 | 4.19430e6 | 4706261610602168320 | yes |
//! | x87_per_chunk | 1 | 0.081 | 0.652 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk | 1 | 0.030 | 1.748 | 4.19430e6 | 4706261610602168320 | yes |
//! | avx2_per_chunk_x87_final | 1 | 0.031 | 1.722 | 4.19430e6 | 4706261610602168320 | yes |
//! | x87_per_chunk | 2 | 0.061 | 0.868 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk | 2 | 0.029 | 1.795 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk_x87_final | 2 | 0.030 | 1.778 | 0.00000e0 | 0 | yes |
//! | x87_per_chunk | 4 | 0.066 | 0.799 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk | 4 | 0.048 | 1.097 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk_x87_final | 4 | 0.059 | 0.890 | 0.00000e0 | 0 | yes |
//! | x87_per_chunk | 6 | 0.065 | 0.813 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk | 6 | 0.079 | 0.668 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk_x87_final | 6 | 0.061 | 0.866 | 0.00000e0 | 0 | yes |
//!
//! ## dot_random
//!
//! Recommendations:
//!
//! - Fastest overall: `serial_avx2` with 1 worker(s), 0.010 ms, abs_err 2.13163e-13, ulp 15
//! - Fastest exact: `x87_per_chunk` with 2 worker(s), 0.053 ms, abs_err 0.00000e0, ulp 0
//! - Fastest exact parallel lane: `x87_per_chunk` with 2 worker(s), 0.053 ms, abs_err 0.00000e0, ulp 0
//! - Fastest <=1 ULP parallel lane: `x87_per_chunk` with 2 worker(s), 0.053 ms, abs_err 0.00000e0, ulp 0
//!
//! | Strategy | Workers | Median ms | Speedup vs serial x87 | Abs error | ULP error | Stable |
//! |---|---:|---:|---:|---:|---:|---|
//! | serial_naive | 1 | 0.046 | 1.160 | 5.54223e-13 | 39 | yes |
//! | serial_kahan | 1 | 0.181 | 0.295 | 0.00000e0 | 0 | yes |
//! | serial_x87 | 1 | 0.054 | 1.000 | 0.00000e0 | 0 | yes |
//! | serial_avx2 | 1 | 0.010 | 5.446 | 2.13163e-13 | 15 | yes |
//! | x87_per_chunk | 1 | 0.087 | 0.614 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk | 1 | 0.035 | 1.536 | 2.13163e-13 | 15 | yes |
//! | avx2_per_chunk_x87_final | 1 | 0.034 | 1.583 | 2.13163e-13 | 15 | yes |
//! | x87_per_chunk | 2 | 0.053 | 1.009 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk | 2 | 0.036 | 1.498 | 3.26850e-13 | 23 | yes |
//! | avx2_per_chunk_x87_final | 2 | 0.033 | 1.644 | 3.26850e-13 | 23 | yes |
//! | x87_per_chunk | 4 | 0.066 | 0.816 | 1.42109e-14 | 1 | yes |
//! | avx2_per_chunk | 4 | 0.051 | 1.045 | 1.42109e-13 | 10 | yes |
//! | avx2_per_chunk_x87_final | 4 | 0.046 | 1.160 | 1.27898e-13 | 9 | yes |
//! | x87_per_chunk | 6 | 0.068 | 0.790 | 0.00000e0 | 0 | no |
//! | avx2_per_chunk | 6 | 0.059 | 0.914 | 1.27898e-13 | 9 | yes |
//! | avx2_per_chunk_x87_final | 6 | 0.059 | 0.912 | 1.27898e-13 | 9 | yes |
//!
//! ## sum_cancellation
//!
//! Recommendations:
//!
//! - Fastest overall: `serial_avx2` with 1 worker(s), 0.006 ms, abs_err 2.68221e-4, ulp 1179648000
//! - Fastest exact: `serial_x87` with 1 worker(s), 0.049 ms, abs_err 0.00000e0, ulp 0
//! - Fastest exact parallel lane: none
//! - Fastest <=1 ULP parallel lane: none
//!
//! | Strategy | Workers | Median ms | Speedup vs serial x87 | Abs error | ULP error | Stable |
//! |---|---:|---:|---:|---:|---:|---|
//! | serial_naive | 1 | 0.046 | 1.073 | 9.52631e-5 | 418971648 | yes |
//! | serial_kahan | 1 | 0.181 | 0.270 | 9.52631e-5 | 418971648 | yes |
//! | serial_x87 | 1 | 0.049 | 1.000 | 0.00000e0 | 0 | yes |
//! | serial_avx2 | 1 | 0.006 | 8.058 | 2.68221e-4 | 1179648000 | yes |
//! | x87_per_chunk | 1 | 0.088 | 0.556 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk | 1 | 0.031 | 1.601 | 2.68221e-4 | 1179648000 | yes |
//! | avx2_per_chunk_x87_final | 1 | 0.028 | 1.773 | 2.68221e-4 | 1179648000 | yes |
//! | x87_per_chunk | 2 | 0.060 | 0.815 | 3.34978e-5 | 147324928 | yes |
//! | avx2_per_chunk | 2 | 0.029 | 1.685 | 3.42131e-4 | 1504706560 | yes |
//! | avx2_per_chunk_x87_final | 2 | 0.030 | 1.614 | 3.42131e-4 | 1504706560 | yes |
//! | x87_per_chunk | 4 | 0.062 | 0.794 | 3.92795e-5 | 172752896 | yes |
//! | avx2_per_chunk | 4 | 0.042 | 1.175 | 2.81096e-4 | 1236271104 | yes |
//! | avx2_per_chunk_x87_final | 4 | 0.043 | 1.135 | 2.81096e-4 | 1236271104 | yes |
//! | x87_per_chunk | 6 | 0.068 | 0.722 | 7.96914e-5 | 350486528 | yes |
//! | avx2_per_chunk | 6 | 0.055 | 0.881 | 5.48512e-5 | 241238016 | yes |
//! | avx2_per_chunk_x87_final | 6 | 0.057 | 0.860 | 5.48512e-5 | 241238016 | yes |
//!
//! ## sum_positive
//!
//! Recommendations:
//!
//! - Fastest overall: `serial_avx2` with 1 worker(s), 0.006 ms, abs_err 0.00000e0, ulp 0
//! - Fastest exact: `serial_avx2` with 1 worker(s), 0.006 ms, abs_err 0.00000e0, ulp 0
//! - Fastest exact parallel lane: `avx2_per_chunk_x87_final` with 2 worker(s), 0.027 ms, abs_err 0.00000e0, ulp 0
//! - Fastest <=1 ULP parallel lane: `avx2_per_chunk_x87_final` with 2 worker(s), 0.027 ms, abs_err 0.00000e0, ulp 0
//!
//! | Strategy | Workers | Median ms | Speedup vs serial x87 | Abs error | ULP error | Stable |
//! |---|---:|---:|---:|---:|---:|---|
//! | serial_naive | 1 | 0.046 | 1.069 | 0.00000e0 | 0 | yes |
//! | serial_kahan | 1 | 0.181 | 0.269 | 0.00000e0 | 0 | yes |
//! | serial_x87 | 1 | 0.049 | 1.000 | 0.00000e0 | 0 | yes |
//! | serial_avx2 | 1 | 0.006 | 8.286 | 0.00000e0 | 0 | yes |
//! | x87_per_chunk | 1 | 0.085 | 0.574 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk | 1 | 0.027 | 1.784 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk_x87_final | 1 | 0.029 | 1.689 | 0.00000e0 | 0 | yes |
//! | x87_per_chunk | 2 | 0.054 | 0.908 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk | 2 | 0.029 | 1.704 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk_x87_final | 2 | 0.027 | 1.793 | 0.00000e0 | 0 | yes |
//! | x87_per_chunk | 4 | 0.066 | 0.736 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk | 4 | 0.047 | 1.045 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk_x87_final | 4 | 0.044 | 1.093 | 0.00000e0 | 0 | yes |
//! | x87_per_chunk | 6 | 0.077 | 0.628 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk | 6 | 0.070 | 0.697 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk_x87_final | 6 | 0.064 | 0.766 | 0.00000e0 | 0 | yes |
//!
