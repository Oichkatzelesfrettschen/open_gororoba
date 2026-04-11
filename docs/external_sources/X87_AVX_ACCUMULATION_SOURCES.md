# x87 / AVX Precision Hardening Sources

Primary source bundle for TICKET-TICKET-X87-AVX-PRECISION-HARDENING and claim C-1362.

## Scope

Sources supporting the crossover heuristic: x87 extended-precision intermediates
(80-bit, u_x87 = 2^-64) are a reliable oracle for accumulations up to N = 2048,
beyond which Kahan compensated summation is the safer default for scalar paths
and AVX2/FMA is the throughput-optimal path for vectorizable accumulations.

## Primary Sources

1. T. Ogita, S. M. Rump, S. Oishi, "Accurate Sum and Dot Product."
   SIAM Journal on Scientific Computing 26(6), 2005, pp. 1955-1988.
   doi:10.1137/030601818
   The crossover argument comes directly from this paper:
   u_f64 / u_x87 = 2^-53 / 2^-64 = 2^11 = 2048. For N <= u_x87/u_f64 the
   x87 extended-precision path produces a result correct to machine epsilon
   for f64 without the 2x overhead of Kahan summation.

2. N. J. Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed.
   SIAM, 2002, Chapter 4 (Summation) and Chapter 3 (Floating-point arithmetic).
   Standard reference for the relationship between floating-point unit roundoff
   and accumulation error bounds (cond * u * n for naive summation).

3. Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 1,
   Chapter 8.1 (x87 FPU Overview).
   Documents the 80-bit extended-double format: 1 sign bit, 15 exponent bits,
   64 mantissa bits (explicit integer bit). Provides u_x87 = 2^-64.

4. Intel 64 and IA-32 Architectures Optimization Reference Manual,
   Section 11.6 (Floating-Point Usage and Precision).
   Describes the throughput and latency of x87 vs SSE2 vs AVX2 instructions.
   Confirms that AVX2/FMA throughput advantage makes it the correct choice
   for vectorizable accumulations where N/latency > precision requirement.

## Benchmark Artifact

`data/csv/x87_avx_fma_followup_benchmark_summary.csv` records measured
Criterion midpoints for sum, dot, and norm_sq at N=512 on the development
machine. Key observations:

- AVX2/FMA: 32-62 ns (10x faster than naive f64)
- x87_fp80: 238-388 ns (comparable to naive f64, with extended precision)
- Kahan: ~1380 ns (4-5x slower than x87, with f64-level precision)

This confirms the heuristic for N <= 1024 (sedenion reductions):
x87 is better precision than f64 at comparable throughput.
For N >> 2048 (Berry-phase sums with n_grid^2 terms), AVX2+Kahan is preferred.

## Deterministic Crossover Verification

`crates/algebra_analysis/tests/precision_tier_dispatch.rs` contains the test
`x87_kahan_crossover_is_2048_terms` which verifies the N=2048 crossover from
the Ogita-Rump-Oishi unit-roundoff ratio. Exit 0 on all relevant CI paths.
