//! AVX packed-double + FMA accumulation primitives for throughput comparison
//! with x87 FP-80.
//!
//! Provides `avx2_sum`, `avx2_dot`, `avx2_norm_sq`, and `avx2_norm_sq_16`.
//! The public names are kept stable, but the packed-`f64` vector type
//! (`__m256d`) and the extract/hadd reduction path are AVX features. Only the
//! fused multiply-add kernels require the separate `fma` target feature.
//!
//! # Precision tier relative to x87_primitives
//!
//! | Operation      | Mantissa bits | Rounding ops per term | Notes                        |
//! |----------------|--------------|----------------------|------------------------------|
//! | x87 FP-80      | 64           | 1 (80-bit FPU)       | Precision oracle             |
//! | AVX + FMA3     | 52 (f64)     | 1 (VFMADD fused)     | Single rounding via FMA      |
//! | AVX (no FMA)   | 52 (f64)     | 2 (VMUL + VADD)      | Two roundings                |
//! | Naive f64      | 52 (f64)     | 2 per term           | Baseline                     |
//!
//! FMA3 (`_mm256_fmadd_pd`) computes `a*b + c` with ONE IEEE-754 rounding
//! (IEEE 754-2008 section 5.4.1), improving precision vs separate MUL+ADD while
//! still producing a binary64 result. x87 can retain extended-precision
//! intermediates, so the comparison is better stated as binary64 single-rounding
//! vs x87 extended intermediates, not as a fixed decimal-digit slogan. This is
//! the correct throughput tier in the cascade:
//!
//!   x87 FP-80 (oracle) -> AVX+FMA (throughput) -> f32 GPU (visualization)
//!
//! # Architecture
//!
//! Public functions check the relevant x86 feature flags at runtime and dispatch
//! to the SIMD implementation or a naive f64 fallback. The SIMD kernels use
//! dual accumulators to exploit out-of-order execution on Zen3's two FMA pipes:
//!
//!   Zen3: 2 dedicated FMADD pipes at 256-bit, 4 f64/cycle each pipe -> 8 f64/cycle peak
//!
//! The dual-accumulator pattern breaks the 4-cycle VADDPD/VFMADDPD latency chain.
//! Without it, throughput is limited to ~1/4 the peak by the carried dependency.
//!
//! # Horizontal reduction
//!
//! `__m256d` holds 4 f64 lanes `[a, b, c, d]`. The `hsum256` helper reduces to scalar:
//! ```text
//! lo     = castpd256_pd128([a, b, c, d]) = [a, b]   (low 128 bits, no instruction)
//! hi     = extractf128([a, b, c, d], 1)  = [c, d]   (high 128 bits)
//! sum128 = addpd([a,b], [c,d])           = [a+c, b+d]
//! out    = hadd_pd(sum128, sum128)        = [a+b+c+d, a+b+c+d]
//! result = cvtsd_f64(out)                = a+b+c+d
//! ```
//!
//! # Rust 2024 unsafe discipline
//!
//! Within `#[target_feature(enable = "...")]` functions, SIMD intrinsics with a
//! matching feature annotation are treated as safe by the compiler -- adding
//! `unsafe {}` around them is spurious and triggers a warning. Raw pointer
//! operations (`ptr.add(i)`, `*ptr`) still require explicit `unsafe {}`.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128d, __m256d, _mm_add_pd, _mm_cvtsd_f64, _mm_hadd_pd, _mm256_add_pd,
    _mm256_castpd256_pd128, _mm256_extractf128_pd, _mm256_fmadd_pd, _mm256_loadu_pd,
    _mm256_setzero_pd,
};

// ── Horizontal reduction helper ─────────────────────────────────────────────

/// Reduce a `__m256d` (4 x f64) to a single f64 sum via extract-add-hadd.
///
/// SAFETY: Requires `avx` target feature (verified by callers).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn hsum256(v: __m256d) -> f64 {
    // castpd256_pd128 is a free alias (no instruction emitted).
    let lo: __m128d = _mm256_castpd256_pd128(v);
    let hi: __m128d = _mm256_extractf128_pd(v, 1);
    let sum128 = _mm_add_pd(lo, hi); // [a+c, b+d]
    let out = _mm_hadd_pd(sum128, sum128); // [(a+c)+(b+d), ...]
    _mm_cvtsd_f64(out)
}

// ── AVX2 sum ────────────────────────────────────────────────────────────────

/// Inner kernel for `avx2_sum`. Requires `avx` target feature.
///
/// Dual-accumulator loop (8 elements per iteration = 2 AVX2 loads) breaks the
/// VADDPD 4-cycle latency chain and fills both execution ports on Zen3.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn avx2_sum_impl(a: &[f64]) -> f64 {
    let n = a.len();
    let ptr = a.as_ptr();
    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();

    // Main loop: 8 elements per iteration (2 x __m256d)
    while i + 8 <= n {
        // SAFETY: i + 8 <= n guarantees both loads are in bounds.
        let v0 = unsafe { _mm256_loadu_pd(ptr.add(i)) };
        let v1 = unsafe { _mm256_loadu_pd(ptr.add(i + 4)) };
        acc0 = _mm256_add_pd(acc0, v0);
        acc1 = _mm256_add_pd(acc1, v1);
        i += 8;
    }

    // Drain: 4 elements per iteration
    while i + 4 <= n {
        // SAFETY: i + 4 <= n guarantees load is in bounds.
        let v = unsafe { _mm256_loadu_pd(ptr.add(i)) };
        acc0 = _mm256_add_pd(acc0, v);
        i += 4;
    }

    // Merge dual accumulators, then scalar remainder
    // SAFETY: avx feature confirmed; pointer reads use in-bounds ptr.add.
    let mut acc = unsafe { hsum256(acc0) + hsum256(acc1) };
    while i < n {
        // SAFETY: i < n guarantees in-bounds.
        acc += unsafe { *ptr.add(i) };
        i += 1;
    }
    acc
}

/// Compute the sum of `a[0..n]` using AVX2 256-bit SIMD.
///
/// Falls back to naive f64 if AVX is not detected at runtime.
pub fn avx2_sum(a: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx") {
        // SAFETY: AVX confirmed by runtime detection.
        return unsafe { avx2_sum_impl(a) };
    }

    a.iter().sum()
}

// ── AVX2 dot product ────────────────────────────────────────────────────────

/// Inner kernel for `avx2_dot`. Requires `avx` + `fma` target features.
///
/// Uses `_mm256_fmadd_pd` (VFMADD231PD): `acc += a * b` with one IEEE-754 rounding.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx2_dot_impl(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();

    // Main loop: 8 elements per iteration, dual accumulators
    while i + 8 <= n {
        // SAFETY: i + 8 <= n guarantees all four loads are in bounds.
        let a0 = unsafe { _mm256_loadu_pd(a_ptr.add(i)) };
        let b0 = unsafe { _mm256_loadu_pd(b_ptr.add(i)) };
        let a1 = unsafe { _mm256_loadu_pd(a_ptr.add(i + 4)) };
        let b1 = unsafe { _mm256_loadu_pd(b_ptr.add(i + 4)) };
        acc0 = _mm256_fmadd_pd(a0, b0, acc0); // acc0 += a0 * b0 (fused, single rounding)
        acc1 = _mm256_fmadd_pd(a1, b1, acc1); // acc1 += a1 * b1
        i += 8;
    }

    // Drain: 4 elements per iteration
    while i + 4 <= n {
        // SAFETY: i + 4 <= n guarantees loads are in bounds.
        let av = unsafe { _mm256_loadu_pd(a_ptr.add(i)) };
        let bv = unsafe { _mm256_loadu_pd(b_ptr.add(i)) };
        acc0 = _mm256_fmadd_pd(av, bv, acc0);
        i += 4;
    }

    // Merge + scalar remainder
    // SAFETY: avx+fma confirmed; pointer reads are in-bounds.
    let mut acc = unsafe { hsum256(acc0) + hsum256(acc1) };
    while i < n {
        // SAFETY: i < n guarantees in-bounds.
        acc += unsafe { *a_ptr.add(i) * *b_ptr.add(i) };
        i += 1;
    }
    acc
}

/// Compute the dot product `sum(a[i] * b[i])` using AVX + FMA.
///
/// Falls back to naive f64 if AVX or FMA is not detected at runtime.
pub fn avx2_dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "avx2_dot: slice length mismatch");

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
        // SAFETY: AVX and FMA confirmed by runtime detection.
        return unsafe { avx2_dot_impl(a, b) };
    }

    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── AVX2 norm squared ────────────────────────────────────────────────────────

/// Inner kernel for `avx2_norm_sq`. Requires `avx` + `fma` target features.
///
/// Uses `_mm256_fmadd_pd(v, v, acc)` to compute `acc += v*v` with one rounding.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx2_norm_sq_impl(a: &[f64]) -> f64 {
    let n = a.len();
    let ptr = a.as_ptr();
    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();

    // Main loop: 8 elements per iteration
    while i + 8 <= n {
        // SAFETY: i + 8 <= n guarantees both loads are in bounds.
        let v0 = unsafe { _mm256_loadu_pd(ptr.add(i)) };
        let v1 = unsafe { _mm256_loadu_pd(ptr.add(i + 4)) };
        acc0 = _mm256_fmadd_pd(v0, v0, acc0); // acc0 += v0 * v0
        acc1 = _mm256_fmadd_pd(v1, v1, acc1); // acc1 += v1 * v1
        i += 8;
    }

    // Drain: 4 elements per iteration
    while i + 4 <= n {
        // SAFETY: i + 4 <= n guarantees load is in bounds.
        let v = unsafe { _mm256_loadu_pd(ptr.add(i)) };
        acc0 = _mm256_fmadd_pd(v, v, acc0);
        i += 4;
    }

    // Merge + scalar remainder
    // SAFETY: avx+fma confirmed; pointer reads are in-bounds.
    let mut acc = unsafe { hsum256(acc0) + hsum256(acc1) };
    while i < n {
        // SAFETY: i < n guarantees in-bounds.
        let x = unsafe { *ptr.add(i) };
        acc += x * x;
        i += 1;
    }
    acc
}

/// Compute the L2 norm squared `sum(a[i]^2)` using AVX + FMA.
///
/// Falls back to naive f64 if AVX or FMA is not detected at runtime.
pub fn avx2_norm_sq(a: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
        // SAFETY: AVX and FMA confirmed by runtime detection.
        return unsafe { avx2_norm_sq_impl(a) };
    }

    a.iter().map(|x| x * x).sum()
}

// ── AVX2 sedenion norm squared (fixed 16-element) ───────────────────────────

/// Inner kernel for `avx2_norm_sq_16`. Requires `avx` + `fma`.
///
/// 4 loads + 4 VFMADD + 4 hsum256 -- no loop overhead, no remainder path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn avx2_norm_sq_16_impl(a: &[f64; 16]) -> f64 {
    let ptr = a.as_ptr();
    // SAFETY: array has exactly 16 elements; offsets 0,4,8,12 are all in bounds.
    let (v0, v1, v2, v3) = unsafe {
        (
            _mm256_loadu_pd(ptr),
            _mm256_loadu_pd(ptr.add(4)),
            _mm256_loadu_pd(ptr.add(8)),
            _mm256_loadu_pd(ptr.add(12)),
        )
    };
    let z = _mm256_setzero_pd();
    let acc0 = _mm256_fmadd_pd(v0, v0, z);
    let acc1 = _mm256_fmadd_pd(v1, v1, z);
    let acc2 = _mm256_fmadd_pd(v2, v2, z);
    let acc3 = _mm256_fmadd_pd(v3, v3, z);
    // SAFETY: avx2+fma confirmed; hsum256 requires avx2 only.
    unsafe { hsum256(acc0) + hsum256(acc1) + hsum256(acc2) + hsum256(acc3) }
}

/// Compute the sedenion (16D) norm squared using AVX + FMA.
///
/// Mirrors `x87_norm_sq_16` from x87_primitives for head-to-head precision comparison.
/// Falls back to naive f64 if AVX2 or FMA is not detected at runtime.
pub fn avx2_norm_sq_16(a: &[f64; 16]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: AVX2 and FMA confirmed by runtime detection.
        return unsafe { avx2_norm_sq_16_impl(a) };
    }

    a.iter().map(|x| x * x).sum()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // One f64 ULP ~ 2.2e-16; for n=256 uniform values, max error ~ n*eps ~ 5.7e-14.
    // Conservative tolerance to accommodate FMA vs naive ordering differences.
    const TOL: f64 = 1e-12;

    #[test]
    fn test_avx2_sum_small() {
        let a = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let got = avx2_sum(&a);
        assert!((got - 36.0).abs() < TOL, "sum={got}");
    }

    #[test]
    fn test_avx2_sum_odd_length() {
        let a = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let got = avx2_sum(&a);
        assert!((got - 15.0).abs() < TOL, "sum={got}");
    }

    #[test]
    fn test_avx2_sum_single() {
        let a = [7.0_f64];
        assert!((avx2_sum(&a) - 7.0).abs() < TOL);
    }

    #[test]
    fn test_avx2_sum_empty() {
        assert_eq!(avx2_sum(&[]), 0.0);
    }

    #[test]
    fn test_avx2_dot_known() {
        let a = [1.0_f64, 2.0, 3.0, 4.0];
        let b = [4.0_f64, 3.0, 2.0, 1.0];
        // 1*4 + 2*3 + 3*2 + 4*1 = 20
        let got = avx2_dot(&a, &b);
        assert!((got - 20.0).abs() < TOL, "dot={got}");
    }

    #[test]
    fn test_avx2_dot_orthogonal() {
        let a = [1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let b = [0.0_f64, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let got = avx2_dot(&a, &b);
        assert!(got.abs() < TOL, "dot of orthogonal={got}");
    }

    #[test]
    fn test_avx2_norm_sq_known() {
        let a = [3.0_f64, 4.0];
        // 3^2 + 4^2 = 25
        assert!((avx2_norm_sq(&a) - 25.0).abs() < TOL);
    }

    #[test]
    fn test_avx2_norm_sq_unit() {
        let a = [1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!((avx2_norm_sq(&a) - 1.0).abs() < TOL);
    }

    #[test]
    fn test_avx2_norm_sq_16_exact() {
        let a: [f64; 16] = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        // sum i^2 for i in 1..=16: 16*17*33/6 = 1496
        let got = avx2_norm_sq_16(&a);
        assert!((got - 1496.0).abs() < TOL, "got={got}");
    }

    #[test]
    fn test_avx2_norm_sq_16_unit_sedenion() {
        let mut a = [0.0_f64; 16];
        a[0] = 1.0;
        assert!((avx2_norm_sq_16(&a) - 1.0).abs() < TOL);
    }

    #[test]
    fn test_avx2_sum_vs_naive_large() {
        let a: Vec<f64> = (0..256).map(|i| (i as f64) * 0.01).collect();
        let naive: f64 = a.iter().sum();
        let got = avx2_sum(&a);
        assert!(
            (got - naive).abs() < naive.abs() * 1e-13 + 1e-11,
            "avx2={got}, naive={naive}"
        );
    }

    #[test]
    fn test_avx2_dot_vs_naive_large() {
        let a: Vec<f64> = (0..128).map(|i| (i as f64) * 0.1).collect();
        let b: Vec<f64> = (0..128).map(|i| ((127 - i) as f64) * 0.1).collect();
        let naive: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let got = avx2_dot(&a, &b);
        assert!(
            (got - naive).abs() < naive.abs() * 1e-12 + 1e-10,
            "avx2={got}, naive={naive}"
        );
    }
}
