//! # x87 Strategy Benchmark Summary
//!
//! ## Run Context
//!
//! - Host: `x570-5600X3D`
//! - CPU: `AMD Ryzen 5 5600X3D 6-Core Processor`
//! - Problem size: `len=65536`
//! - Repeats per row: `5`
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
//! - Fastest exact: `avx2_per_chunk` with 2 worker(s), 0.031 ms, abs_err 0.00000e0, ulp 0
//! - Fastest exact parallel lane: `avx2_per_chunk` with 2 worker(s), 0.031 ms, abs_err 0.00000e0, ulp 0
//! - Fastest <=1 ULP parallel lane: `avx2_per_chunk` with 2 worker(s), 0.031 ms, abs_err 0.00000e0, ulp 0
//!
//! | Strategy | Workers | Median ms | Speedup vs serial x87 | Abs error | ULP error | Stable |
//! |---|---:|---:|---:|---:|---:|---|
//! | serial_naive | 1 | 0.059 | 0.908 | 1.57286e6 | 4699506211161112576 | yes |
//! | serial_kahan | 1 | 0.189 | 0.284 | 0.00000e0 | 0 | yes |
//! | serial_x87 | 1 | 0.054 | 1.000 | 0.00000e0 | 0 | yes |
//! | serial_avx2 | 1 | 0.010 | 5.272 | 4.19430e6 | 4706261610602168320 | yes |
//! | x87_per_chunk | 1 | 0.080 | 0.668 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk | 1 | 0.030 | 1.769 | 4.19430e6 | 4706261610602168320 | yes |
//! | avx2_per_chunk_x87_final | 1 | 0.030 | 1.775 | 4.19430e6 | 4706261610602168320 | yes |
//! | x87_per_chunk | 2 | 0.053 | 1.003 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk | 2 | 0.031 | 1.742 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk_x87_final | 2 | 0.031 | 1.729 | 0.00000e0 | 0 | yes |
//! | x87_per_chunk | 4 | 0.069 | 0.780 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk | 4 | 0.051 | 1.060 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk_x87_final | 4 | 0.050 | 1.064 | 0.00000e0 | 0 | no |
//! | x87_per_chunk | 6 | 0.084 | 0.641 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk | 6 | 0.084 | 0.637 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk_x87_final | 6 | 0.072 | 0.746 | 0.00000e0 | 0 | yes |
//!
//! ## dot_random
//!
//! Recommendations:
//!
//! - Fastest overall: `serial_avx2` with 1 worker(s), 0.010 ms, abs_err 2.13163e-13, ulp 15
//! - Fastest exact: `serial_x87` with 1 worker(s), 0.052 ms, abs_err 0.00000e0, ulp 0
//! - Fastest exact parallel lane: `x87_per_chunk` with 2 worker(s), 0.054 ms, abs_err 0.00000e0, ulp 0
//! - Fastest <=1 ULP parallel lane: `x87_per_chunk` with 2 worker(s), 0.054 ms, abs_err 0.00000e0, ulp 0
//!
//! | Strategy | Workers | Median ms | Speedup vs serial x87 | Abs error | ULP error | Stable |
//! |---|---:|---:|---:|---:|---:|---|
//! | serial_naive | 1 | 0.059 | 0.882 | 5.54223e-13 | 39 | yes |
//! | serial_kahan | 1 | 0.187 | 0.279 | 0.00000e0 | 0 | yes |
//! | serial_x87 | 1 | 0.052 | 1.000 | 0.00000e0 | 0 | yes |
//! | serial_avx2 | 1 | 0.010 | 5.224 | 2.13163e-13 | 15 | yes |
//! | x87_per_chunk | 1 | 0.082 | 0.637 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk | 1 | 0.030 | 1.746 | 2.13163e-13 | 15 | yes |
//! | avx2_per_chunk_x87_final | 1 | 0.029 | 1.829 | 2.13163e-13 | 15 | yes |
//! | x87_per_chunk | 2 | 0.054 | 0.968 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk | 2 | 0.046 | 1.141 | 3.26850e-13 | 23 | yes |
//! | avx2_per_chunk_x87_final | 2 | 0.035 | 1.507 | 3.26850e-13 | 23 | yes |
//! | x87_per_chunk | 4 | 0.062 | 0.835 | 1.42109e-14 | 1 | yes |
//! | avx2_per_chunk | 4 | 0.048 | 1.093 | 1.42109e-13 | 10 | yes |
//! | avx2_per_chunk_x87_final | 4 | 0.046 | 1.143 | 1.27898e-13 | 9 | yes |
//! | x87_per_chunk | 6 | 12.877 | 0.004 | 0.00000e0 | 0 | no |
//! | avx2_per_chunk | 6 | 0.066 | 0.792 | 1.27898e-13 | 9 | yes |
//! | avx2_per_chunk_x87_final | 6 | 0.060 | 0.868 | 1.27898e-13 | 9 | yes |
//!
//! ## sum_cancellation
//!
//! Recommendations:
//!
//! - Fastest overall: `serial_avx2` with 1 worker(s), 0.006 ms, abs_err 2.68221e-4, ulp 1179648000
//! - Fastest exact: `serial_x87` with 1 worker(s), 0.051 ms, abs_err 0.00000e0, ulp 0
//! - Fastest exact parallel lane: none
//! - Fastest <=1 ULP parallel lane: none
//!
//! | Strategy | Workers | Median ms | Speedup vs serial x87 | Abs error | ULP error | Stable |
//! |---|---:|---:|---:|---:|---:|---|
//! | serial_naive | 1 | 0.047 | 1.081 | 9.52631e-5 | 418971648 | yes |
//! | serial_kahan | 1 | 0.188 | 0.270 | 9.52631e-5 | 418971648 | yes |
//! | serial_x87 | 1 | 0.051 | 1.000 | 0.00000e0 | 0 | yes |
//! | serial_avx2 | 1 | 0.006 | 8.014 | 2.68221e-4 | 1179648000 | yes |
//! | x87_per_chunk | 1 | 0.074 | 0.685 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk | 1 | 0.034 | 1.469 | 2.68221e-4 | 1179648000 | yes |
//! | avx2_per_chunk_x87_final | 1 | 0.028 | 1.799 | 2.68221e-4 | 1179648000 | yes |
//! | x87_per_chunk | 2 | 0.057 | 0.892 | 3.34978e-5 | 147324928 | yes |
//! | avx2_per_chunk | 2 | 0.034 | 1.497 | 3.42131e-4 | 1504706560 | yes |
//! | avx2_per_chunk_x87_final | 2 | 0.030 | 1.682 | 3.42131e-4 | 1504706560 | yes |
//! | x87_per_chunk | 4 | 0.055 | 0.922 | 3.92795e-5 | 172752896 | yes |
//! | avx2_per_chunk | 4 | 0.052 | 0.973 | 2.81096e-4 | 1236271104 | yes |
//! | avx2_per_chunk_x87_final | 4 | 0.057 | 0.886 | 2.81096e-4 | 1236271104 | yes |
//! | x87_per_chunk | 6 | 0.083 | 0.614 | 7.96914e-5 | 350486528 | no |
//! | avx2_per_chunk | 6 | 0.060 | 0.848 | 5.48512e-5 | 241238016 | yes |
//! | avx2_per_chunk_x87_final | 6 | 0.060 | 0.843 | 5.48512e-5 | 241238016 | yes |
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
//! | serial_naive | 1 | 0.047 | 1.025 | 0.00000e0 | 0 | yes |
//! | serial_kahan | 1 | 0.187 | 0.257 | 0.00000e0 | 0 | yes |
//! | serial_x87 | 1 | 0.048 | 1.000 | 0.00000e0 | 0 | yes |
//! | serial_avx2 | 1 | 0.006 | 8.173 | 0.00000e0 | 0 | yes |
//! | x87_per_chunk | 1 | 0.079 | 0.606 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk | 1 | 0.027 | 1.763 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk_x87_final | 1 | 0.028 | 1.715 | 0.00000e0 | 0 | yes |
//! | x87_per_chunk | 2 | 0.066 | 0.729 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk | 2 | 0.028 | 1.728 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk_x87_final | 2 | 0.027 | 1.778 | 0.00000e0 | 0 | yes |
//! | x87_per_chunk | 4 | 0.105 | 0.456 | 0.00000e0 | 0 | no |
//! | avx2_per_chunk | 4 | 0.043 | 1.130 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk_x87_final | 4 | 0.041 | 1.161 | 0.00000e0 | 0 | yes |
//! | x87_per_chunk | 6 | 0.247 | 0.195 | 0.00000e0 | 0 | no |
//! | avx2_per_chunk | 6 | 0.056 | 0.866 | 0.00000e0 | 0 | yes |
//! | avx2_per_chunk_x87_final | 6 | 0.054 | 0.887 | 0.00000e0 | 0 | yes |
//!
//!
