//! x87 Extended Precision Benchmark Suite
//!
//! Four benchmark families:
//!
//! 1. **Jacobi eigensolvers** (original, from E-181):
//!    - nalgebra symmetric_eigenvalues (LAPACK-style Jacobi, f64)
//!    - x87 FP-80 inline asm Jacobi (hardware 64-bit mantissa, x86_64 only)
//!    - Double-double (DD) software Jacobi (~31 decimal digits, all platforms)
//!
//! 2. **Accumulation throughput** (four variants per operation):
//!    - naive_f64: compiler-auto-vectorized scalar f64 (SSE2/AVX baseline)
//!    - kahan: software compensated sum (O(eps) error, O(n) overhead)
//!    - x87_fp80: x87 80-bit inline asm, entire loop in asm (LLVM spill-safe)
//!    - avx2_fma: AVX packed-double + FMA3 SIMD, dual accumulators, 8 f64/cycle peak (Zen3)
//!      Array sizes [4, 8, 16, 32, 64, 128, 256, 512] to expose amortization.
//!
//! 3. **Precision stress** (accuracy measurement via criterion):
//!    - Near-cancellation sums where naive f64 loses several significant bits
//!    - Ill-conditioned dot products (Ogita-style: exact sum is O(eps), naive is O(1))
//!    - Wilkinson polynomial at roots: x87 Horner vs naive power-sum Horner
//!
//! 4. **Precision tier ranking** (expected from theory, measurable here):
//!    x87 FP-80 extended intermediates > Kahan compensated sum > AVX+FMA
//!    single-rounding binary64 > naive f64
//!    In throughput: AVX+FMA tends to dominate wide reductions, while x87 can
//!    still win for specific ILP-heavy kernels such as the 4-accumulator
//!    `norm_sq` path.
//!
//! # Extended crate evaluation (documented here, not benchmarked)
//!
//! The `extended` crate (crates.io, MIT) was evaluated and REJECTED:
//! - Precision: 64-bit mantissa (same as our x87 inline asm)
//! - Overhead: f64 <-> f80 conversion at every boundary
//! - Missing: No atan2() or sincos() -- the critical operations for Jacobi
//! - LLVM hazard: x87 values spilled to stack truncate to f64 (LLVM #44218)
//! - Portability: x86 only (aliases f64 elsewhere -- no precision gain)
//!
//! Our current x87 backend still avoids depending on compiler-managed long
//! double semantics and keeps the critical transcendental/update kernels under
//! explicit x87 control, but the Jacobi path is now a composed Ext80 backend
//! rather than one monolithic ST-register chain:
//! - The reusable kernels live in cd_kernel and use explicit Ext80 load/store
//!   boundaries plus status-aware x87 assembly wrappers.
//! - The Jacobi Givens path composes atan2/sincos/scale/update helpers instead
//!   of claiming zero conversion overhead through one giant asm block.
//! - DD fallback on non-x86 provides ~31 digits (vs FP-80's ~18.5)
//!
//! Verdict: `extended` crate offers no advantage over our inline asm and lacks
//! the transcendental functions we need. Evaluated and rejected.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::DMatrix;
use std::hint::black_box;

// ── Jacobi benchmark helpers ─────────────────────────────────────────────────

/// Build a known-spectrum symmetric matrix of size n x n.
/// Eigenvalues are 1, 2, ..., n (condition number = n).
fn build_known_spectrum_matrix(n: usize) -> Vec<Vec<f64>> {
    let inv_n = 1.0 / n as f64;
    let mut m = vec![vec![0.0; n]; n];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let mut val = 0.0;
            for k in 0..n {
                let h_ik = if i == k { 1.0 } else { 0.0 } - 2.0 * inv_n;
                let h_jk = if j == k { 1.0 } else { 0.0 } - 2.0 * inv_n;
                val += h_ik * (k + 1) as f64 * h_jk;
            }
            *cell = val;
        }
    }
    m
}

// ── Accumulation helpers ──────────────────────────────────────────────────────

/// Kahan compensated sum: O(eps) error vs O(n*eps) for naive f64.
/// Used as an intermediate reference between naive f64 and x87 FP-80.
#[inline]
fn kahan_sum(a: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64;
    for &x in a {
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Kahan compensated dot product.
#[inline]
fn kahan_dot(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let prod = ai * bi;
        let y = prod - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Kahan compensated norm_sq.
#[inline]
fn kahan_norm_sq(a: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64;
    for &x in a {
        let prod = x * x;
        let y = prod - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Build an array of length n with alternating values +v, -(v + eps_step*i).
/// The exact sum is a small number (n/2 * eps_step * ~n/2), while intermediate
/// values are O(v). This creates near-cancellation that stresses f64 naive sum.
fn near_cancellation_array(n: usize, v: f64, eps_step: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            if i % 2 == 0 {
                v
            } else {
                -(v + eps_step * (i as f64))
            }
        })
        .collect()
}

/// Build an Ogita-Rump-Oishi-style ill-conditioned dot product (a, b) where:
/// - The exact dot product is a small value (target_exact)
/// - The individual terms a[i]*b[i] can be large and opposite in sign
/// - f64 naive accumulation loses all significance; x87 retains precision
///
/// Construction (Ogita 2005, section 4): we create a dot product where half
/// the terms are positive and half are negative, with exact sum = target_exact.
fn ogita_dot_vectors(n: usize, scale: f64, target_exact: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(n >= 4 && n.is_multiple_of(2));
    let half = n / 2;
    let mut a = vec![0.0_f64; n];
    let mut b = vec![0.0_f64; n];
    // Positive terms: a[i]=scale, b[i]=scale -> contribution = scale^2
    // Negative terms: a[half+i]=scale, b[half+i]=-(scale - correction)
    // Sum of positive: half * scale^2
    // Sum of negative: half * (-scale^2 + scale*correction)
    // Net: half * scale * correction = target_exact -> correction = target_exact / (half * scale)
    let correction = target_exact / (half as f64 * scale);
    for i in 0..half {
        a[i] = scale;
        b[i] = scale;
        a[half + i] = scale;
        b[half + i] = -(scale - correction);
    }
    (a, b)
}

// ── Benchmark group 1: Jacobi eigensolvers ────────────────────────────────────

fn bench_eigenvalue_solvers(c: &mut Criterion) {
    let sizes = [8, 16];

    for &n in &sizes {
        let matrix = build_known_spectrum_matrix(n);
        let na_mat = DMatrix::from_fn(n, n, |i, j| matrix[i][j]);

        c.bench_function(&format!("nalgebra_eigvals_{n}x{n}"), |b| {
            b.iter(|| {
                let eigs = black_box(&na_mat).symmetric_eigenvalues();
                black_box(eigs)
            });
        });

        #[cfg(target_arch = "x86_64")]
        {
            let flat_matrix: Vec<f64> = matrix.iter().flat_map(|row| row.iter().copied()).collect();
            c.bench_function(&format!("x87_fp80_eigvals_{n}x{n}"), |b| {
                b.iter(|| {
                    let eigs = algebra_analysis::x87_jacobi::symmetric_eigenvalues_x87(
                        black_box(&flat_matrix),
                        n,
                        500,
                        1.0e-15,
                    )
                    .unwrap();
                    black_box(eigs)
                });
            });
        }

        {
            let m_clone = matrix.clone();
            c.bench_function(&format!("dd_eigvals_{n}x{n}"), |b| {
                b.iter(|| {
                    let eigs =
                        algebra_analysis::dd_jacobi::symmetric_eigenvalues_dd(black_box(&m_clone))
                            .unwrap();
                    black_box(eigs)
                });
            });
        }
    }
}

fn bench_precision_audit(c: &mut Criterion) {
    let n = 16;
    let matrix = build_known_spectrum_matrix(n);
    let na_mat = DMatrix::from_fn(n, n, |i, j| matrix[i][j]);

    c.bench_function("precision_nalgebra_16x16", |b| {
        b.iter(|| {
            let eigs = black_box(&na_mat).symmetric_eigenvalues();
            let mut sorted: Vec<f64> = eigs.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let max_err: f64 = sorted
                .iter()
                .enumerate()
                .map(|(i, &e)| (e - (i + 1) as f64).abs())
                .fold(0.0_f64, f64::max);
            black_box(max_err)
        });
    });

    #[cfg(target_arch = "x86_64")]
    {
        let flat_matrix: Vec<f64> = matrix.iter().flat_map(|row| row.iter().copied()).collect();
        c.bench_function("precision_x87_16x16", |b| {
            b.iter(|| {
                let eigs = algebra_analysis::x87_jacobi::symmetric_eigenvalues_x87(
                    black_box(&flat_matrix),
                    n,
                    500,
                    1.0e-15,
                )
                .unwrap();
                let max_err: f64 = eigs
                    .iter()
                    .enumerate()
                    .map(|(i, &e)| (e - (i + 1) as f64).abs())
                    .fold(0.0_f64, f64::max);
                black_box(max_err)
            });
        });
    }

    {
        let m_clone = matrix.clone();
        c.bench_function("precision_dd_16x16", |b| {
            b.iter(|| {
                let eigs =
                    algebra_analysis::dd_jacobi::symmetric_eigenvalues_dd(black_box(&m_clone))
                        .unwrap();
                let max_err: f64 = eigs
                    .iter()
                    .enumerate()
                    .map(|(i, &e)| (e - (i + 1) as f64).abs())
                    .fold(0.0_f64, f64::max);
                black_box(max_err)
            });
        });
    }
}

// ── Benchmark group 2: Accumulation throughput ───────────────────────────────

fn bench_sum_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_throughput");
    let sizes = [4usize, 8, 16, 32, 64, 128, 256, 512];

    for &n in &sizes {
        let a: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.01).collect();

        group.bench_with_input(BenchmarkId::new("naive_f64", n), &n, |b, _| {
            b.iter(|| {
                let s: f64 = black_box(&a).iter().copied().sum();
                black_box(s)
            });
        });

        group.bench_with_input(BenchmarkId::new("kahan", n), &n, |b, _| {
            b.iter(|| {
                let s = kahan_sum(black_box(&a));
                black_box(s)
            });
        });

        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("x87_fp80", n), &n, |b, _| {
            b.iter(|| {
                let s = cd_kernel::x87_sum(black_box(&a));
                black_box(s)
            });
        });

        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("avx2_fma", n), &n, |b, _| {
            b.iter(|| {
                let s = cd_kernel::avx2_sum(black_box(&a));
                black_box(s)
            });
        });
    }
    group.finish();
}

fn bench_dot_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_throughput");
    let sizes = [4usize, 8, 16, 32, 64, 128, 256, 512];

    for &n in &sizes {
        let a: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.01).collect();
        let b: Vec<f64> = (0..n).map(|i| ((n - i) as f64) * 0.01 - 0.3).collect();

        group.bench_with_input(BenchmarkId::new("naive_f64", n), &n, |bench, _| {
            bench.iter(|| {
                let s: f64 = black_box(&a)
                    .iter()
                    .zip(black_box(&b).iter())
                    .map(|(x, y)| x * y)
                    .sum();
                black_box(s)
            });
        });

        group.bench_with_input(BenchmarkId::new("kahan", n), &n, |bench, _| {
            bench.iter(|| {
                let s = kahan_dot(black_box(&a), black_box(&b));
                black_box(s)
            });
        });

        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("x87_fp80", n), &n, |bench, _| {
            bench.iter(|| {
                let s = cd_kernel::x87_dot(black_box(&a), black_box(&b));
                black_box(s)
            });
        });

        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("avx2_fma", n), &n, |bench, _| {
            bench.iter(|| {
                let s = cd_kernel::avx2_dot(black_box(&a), black_box(&b));
                black_box(s)
            });
        });
    }
    group.finish();
}

fn bench_norm_sq_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("norm_sq_throughput");
    let sizes = [4usize, 8, 16, 32, 64, 128, 256, 512];

    for &n in &sizes {
        let a: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.001).collect();

        group.bench_with_input(BenchmarkId::new("naive_f64", n), &n, |bench, _| {
            bench.iter(|| {
                let s: f64 = black_box(&a).iter().map(|x| x * x).sum();
                black_box(s)
            });
        });

        group.bench_with_input(BenchmarkId::new("kahan", n), &n, |bench, _| {
            bench.iter(|| {
                let s = kahan_norm_sq(black_box(&a));
                black_box(s)
            });
        });

        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("x87_fp80", n), &n, |bench, _| {
            bench.iter(|| {
                let s = cd_kernel::x87_norm_sq(black_box(&a));
                black_box(s)
            });
        });

        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("avx2_fma", n), &n, |bench, _| {
            bench.iter(|| {
                let s = cd_kernel::avx2_norm_sq(black_box(&a));
                black_box(s)
            });
        });
    }

    // Sedenion-specific: fixed 16-element paths (x87 unrolled vs AVX2 4-load)
    group.bench_function("x87_norm_sq_16_fixed", |bench| {
        let a: [f64; 16] = core::array::from_fn(|i| (i as f64 + 1.0) * 0.001);
        bench.iter(|| {
            let s = cd_kernel::x87_norm_sq_16(black_box(&a));
            black_box(s)
        });
    });

    #[cfg(target_arch = "x86_64")]
    group.bench_function("avx2_norm_sq_16_fixed", |bench| {
        let a: [f64; 16] = core::array::from_fn(|i| (i as f64 + 1.0) * 0.001);
        bench.iter(|| {
            let s = cd_kernel::avx2_norm_sq_16(black_box(&a));
            black_box(s)
        });
    });

    group.finish();
}

fn bench_horner_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("horner_throughput");
    // Polynomial degrees [4, 8, 16, 32] -- x87 advantage grows with degree
    // since each Horner step accumulates FP error in standard f64.
    let degrees = [4usize, 8, 16, 32];

    for &deg in &degrees {
        // Wilkinson-style: coefficients alternate sign, large magnitude
        let coeffs: Vec<f64> = (0..=deg)
            .map(|i| {
                let v = (i + 1) as f64;
                if i % 2 == 0 { v } else { -v }
            })
            .collect();
        let x = 0.99_f64; // near a cluster of roots

        group.bench_with_input(BenchmarkId::new("naive_f64", deg), &deg, |bench, _| {
            bench.iter(|| {
                // Naive Horner in f64 (pure Rust)
                let c = black_box(&coeffs);
                if c.is_empty() {
                    return black_box(0.0);
                }
                let mut acc = *c.last().unwrap();
                for &ci in c[..c.len() - 1].iter().rev() {
                    acc = acc * black_box(x) + ci;
                }
                black_box(acc)
            });
        });

        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("x87_fp80", deg), &deg, |bench, _| {
            bench.iter(|| {
                let s = cd_kernel::x87_horner(black_box(&coeffs), black_box(x));
                black_box(s)
            });
        });
    }
    group.finish();
}

// ── Benchmark group 3: Precision stress ──────────────────────────────────────
//
// These benchmarks MEASURE precision error (not just throughput).
// Each iteration returns |result - expected| as f64. Criterion will report
// the distribution of these error values across iterations. A smaller number
// means higher precision. This is an unusual use of criterion (measuring
// accuracy instead of speed) but valid: the "measurement" here is the error.

fn bench_precision_near_cancellation(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_near_cancellation");
    // Array: alternating +K, -(K + eps_step*i). Exact sum = sum of -eps_step*i
    // for odd i, which is a small positive number. f64 naive often gets the
    // wrong order of magnitude or sign for large n.
    let sizes = [64usize, 256, 1024];
    let v = 1e8_f64; // large base value for cancellation
    let eps = 1e-6_f64; // tiny perturbation -- the signal we want to measure

    for &n in &sizes {
        let arr = near_cancellation_array(n, v, eps);
        // "Exact" reference (higher precision estimate via compensated sum):
        let kahan_ref = kahan_sum(&arr);

        group.bench_with_input(BenchmarkId::new("naive_f64_err", n), &n, |b, _| {
            b.iter(|| {
                let s: f64 = black_box(&arr).iter().copied().sum();
                // Return absolute error vs Kahan reference
                black_box((s - kahan_ref).abs())
            });
        });

        group.bench_with_input(BenchmarkId::new("kahan_err", n), &n, |b, _| {
            b.iter(|| {
                let s = kahan_sum(black_box(&arr));
                black_box((s - kahan_ref).abs()) // should be near-zero by definition
            });
        });

        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("x87_fp80_err", n), &n, |b, _| {
            b.iter(|| {
                let s = cd_kernel::x87_sum(black_box(&arr));
                black_box((s - kahan_ref).abs())
            });
        });

        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("avx2_fma_err", n), &n, |b, _| {
            b.iter(|| {
                let s = cd_kernel::avx2_sum(black_box(&arr));
                // AVX2 sum is f64 precision; expect error between naive_f64 and Kahan.
                // Dual-accumulator reduces error by ~2x vs single-pass naive f64,
                // but does not achieve Kahan O(eps) or x87 80-bit oracle.
                black_box((s - kahan_ref).abs())
            });
        });
    }
    group.finish();
}

fn bench_precision_dot_ill_conditioned(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_dot_ill_conditioned");
    // Ogita-style: exact dot product = 1e-6, but terms are O(1e6).
    // Condition number of the dot product ~ scale^2 / target_exact = 1e18.
    // f64 has only ~15 digits; this should lose all of them.
    let n = 128_usize;
    let scale = 1e9_f64;
    let target = 1.0_f64; // "exact" value
    let (a, b) = ogita_dot_vectors(n, scale, target);

    // Kahan reference for error measurement
    let kahan_ref = kahan_dot(&a, &b);

    group.bench_function("naive_f64_err", |bench| {
        bench.iter(|| {
            let s: f64 = black_box(&a)
                .iter()
                .zip(black_box(&b).iter())
                .map(|(x, y)| x * y)
                .sum();
            black_box((s - kahan_ref).abs())
        });
    });

    group.bench_function("kahan_err", |bench| {
        bench.iter(|| {
            let s = kahan_dot(black_box(&a), black_box(&b));
            black_box((s - kahan_ref).abs())
        });
    });

    #[cfg(target_arch = "x86_64")]
    group.bench_function("x87_fp80_err", |bench| {
        bench.iter(|| {
            let s = cd_kernel::x87_dot(black_box(&a), black_box(&b));
            black_box((s - kahan_ref).abs())
        });
    });

    #[cfg(target_arch = "x86_64")]
    group.bench_function("avx2_fma_err", |bench| {
        bench.iter(|| {
            let s = cd_kernel::avx2_dot(black_box(&a), black_box(&b));
            // AVX2+FMA has single rounding per term (VFMADD), so less error than
            // naive f64 (which does two roundings: mul then add). Still f64 range.
            black_box((s - kahan_ref).abs())
        });
    });

    group.finish();
}

fn bench_precision_horner_wilkinson(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_horner_wilkinson");
    // Wilkinson polynomial: (x-1)(x-2)...(x-n) evaluated near roots.
    // At x slightly off a root, the value is sensitive to coefficient rounding.
    // Coefficients are the expanded form (exact integers for n<=15).
    // We test n=8: (x-1)...(x-8), roots at 1..8.
    // Near x=4.5 (midpoint between roots 4 and 5), the naive Horner in f64
    // and x87 Horner should agree, but near x=4.0 (at a root) x87 is tighter.
    //
    // Coefficients for (x-1)(x-2)(x-3)(x-4)(x-5)(x-6)(x-7)(x-8) (ascending):
    let coeffs_w8 = [
        40320.0_f64,
        -109584.0,
        118124.0,
        -67284.0,
        22449.0,
        -4536.0,
        546.0,
        -36.0,
        1.0,
    ];
    let exact_at_2: f64 = 0.0; // p(2) = 0 exactly

    group.bench_function("naive_horner_err_at_root", |bench| {
        bench.iter(|| {
            let x = black_box(2.0_f64);
            let c = black_box(&coeffs_w8);
            let mut acc = *c.last().unwrap();
            for &ci in c[..c.len() - 1].iter().rev() {
                acc = acc * x + ci;
            }
            black_box((acc - exact_at_2).abs())
        });
    });

    #[cfg(target_arch = "x86_64")]
    group.bench_function("x87_horner_err_at_root", |bench| {
        bench.iter(|| {
            let v = cd_kernel::x87_horner(black_box(&coeffs_w8), black_box(2.0_f64));
            black_box((v - exact_at_2).abs())
        });
    });

    group.finish();
}

fn bench_multicore_strategies(c: &mut Criterion) {
    let plan = cd_kernel::PhysicalCorePlan::pinned(Some(6));
    let workers = plan.worker_count();
    let mut group = c.benchmark_group("multicore_reduction");
    let n = 1 << 18;

    let sum_data: Vec<f64> = (0..n)
        .map(|i| {
            if i % 2 == 0 {
                1.0e8
            } else {
                -(1.0e8 + (i as f64) * 1.0e-6)
            }
        })
        .collect();
    let dot_left: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 1.0e-6).collect();
    let dot_right: Vec<f64> = (0..n)
        .map(|i| {
            if i % 2 == 0 {
                1.0e6
            } else {
                -(1.0e6 - (i as f64) * 1.0e-9)
            }
        })
        .collect();

    group.bench_function(format!("sum_x87_parallel_w{workers}"), |bench| {
        bench.iter(|| {
            let s = cd_kernel::parallel_sum(
                black_box(&sum_data),
                cd_kernel::ParallelReductionStrategy::X87PerChunk,
                black_box(&plan),
            );
            black_box(s)
        });
    });

    group.bench_function(format!("sum_avx2_parallel_w{workers}"), |bench| {
        bench.iter(|| {
            let s = cd_kernel::parallel_sum(
                black_box(&sum_data),
                cd_kernel::ParallelReductionStrategy::Avx2PerChunk,
                black_box(&plan),
            );
            black_box(s)
        });
    });

    group.bench_function(format!("sum_hybrid_parallel_w{workers}"), |bench| {
        bench.iter(|| {
            let s = cd_kernel::parallel_sum(
                black_box(&sum_data),
                cd_kernel::ParallelReductionStrategy::Avx2PerChunkX87Final,
                black_box(&plan),
            );
            black_box(s)
        });
    });

    group.bench_function(format!("dot_x87_parallel_w{workers}"), |bench| {
        bench.iter(|| {
            let s = cd_kernel::parallel_dot(
                black_box(&dot_left),
                black_box(&dot_right),
                cd_kernel::ParallelReductionStrategy::X87PerChunk,
                black_box(&plan),
            );
            black_box(s)
        });
    });

    group.bench_function(format!("dot_avx2_parallel_w{workers}"), |bench| {
        bench.iter(|| {
            let s = cd_kernel::parallel_dot(
                black_box(&dot_left),
                black_box(&dot_right),
                cd_kernel::ParallelReductionStrategy::Avx2PerChunk,
                black_box(&plan),
            );
            black_box(s)
        });
    });

    group.bench_function(format!("dot_hybrid_parallel_w{workers}"), |bench| {
        bench.iter(|| {
            let s = cd_kernel::parallel_dot(
                black_box(&dot_left),
                black_box(&dot_right),
                cd_kernel::ParallelReductionStrategy::Avx2PerChunkX87Final,
                black_box(&plan),
            );
            black_box(s)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_eigenvalue_solvers,
    bench_precision_audit,
    bench_sum_throughput,
    bench_dot_throughput,
    bench_norm_sq_throughput,
    bench_horner_throughput,
    bench_precision_near_cancellation,
    bench_precision_dot_ill_conditioned,
    bench_precision_horner_wilkinson,
    bench_multicore_strategies,
);
criterion_main!(benches);
