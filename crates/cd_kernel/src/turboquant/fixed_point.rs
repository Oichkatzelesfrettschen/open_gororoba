//! Fixed-point arithmetic for TurboQuant codebook operations.
//!
//! Ported from steinmarder's Q16.16 and Q32.32 LBM kernels which achieved:
//!   Q16.16: mass_drift=0.00e+00 (perfect), 1945 MLUPS (comparable to FP32)
//!   Q32.32: mass_drift=0.00e+00 (perfect), 1.88x FASTER than FP64 at same 8-byte footprint
//!
//! # Why fixed-point for TurboQuant?
//!
//! 1. **Exact reproducibility**: Integer arithmetic is order-independent.
//!    `a + b + c == c + a + b` always holds for integers, but NOT for floats.
//!    This eliminates GPU thread-scheduling-dependent results in dot products.
//!
//! 2. **Speed on gaming GPUs**: INT64 ALU >> FP64 ALU on Ada Lovelace
//!    (64:1 FP64:FP32 ratio means FP64 is artificially slow).
//!    Q32.32 achieves 71.5% bandwidth utilization vs FP64's 38.0%.
//!
//! 3. **Conservation**: In LBM, Q16.16 achieved zero mass drift (better than
//!    FP32's 4.77e-07). For TurboQuant, this means attention score sums
//!    can be exactly reproducible across runs.
//!
//! # Q16.16 format
//!
//! A Q16.16 number is a 32-bit integer where:
//!   - bits [31:16] = integer part (signed, range -32768 to 32767)
//!   - bits [15:0]  = fractional part (65536 subdivisions)
//!   - Resolution: 1/65536 = 1.53e-5
//!   - Range: approximately [-32768.0, 32767.99998]
//!
//! # Q32.32 format
//!
//! A Q32.32 number is a 64-bit integer where:
//!   - bits [63:32] = integer part (signed, range -2^31 to 2^31-1)
//!   - bits [31:0]  = fractional part (2^32 subdivisions)
//!   - Resolution: 1/2^32 = 2.33e-10
//!   - Range: approximately [-2.1e9, 2.1e9]

/// Q16.16 fixed-point number (32-bit integer representation).
///
/// Derives bytemuck Pod + Zeroable for zero-copy buffer packing
/// (GPU transfer, serialization, memory-mapped I/O).
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Q16_16(pub i32);

/// Q32.32 fixed-point number (64-bit integer representation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Q32_32(pub i64);

const Q16_FRAC_BITS: u32 = 16;
const Q16_SCALE: f64 = 65536.0; // 2^16

const Q32_FRAC_BITS: u32 = 32;
const Q32_SCALE: f64 = 4294967296.0; // 2^32

impl Q16_16 {
    /// Convert from f64 to Q16.16.
    #[inline]
    pub fn from_f64(v: f64) -> Self {
        Q16_16((v * Q16_SCALE).round() as i32)
    }

    /// Convert from f32 to Q16.16.
    #[inline]
    pub fn from_f32(v: f32) -> Self {
        Q16_16((v as f64 * Q16_SCALE).round() as i32)
    }

    /// Convert to f64.
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Q16_SCALE
    }

    /// Convert to f32.
    #[inline]
    pub fn to_f32(self) -> f32 {
        (self.0 as f64 / Q16_SCALE) as f32
    }

}

impl std::ops::Mul for Q16_16 {
    type Output = Self;

    /// Multiply two Q16.16 values, returning Q16.16.
    /// Uses i64 intermediate to avoid overflow.
    #[inline]
    fn mul(self, other: Self) -> Self {
        let wide = self.0 as i64 * other.0 as i64;
        Q16_16((wide >> Q16_FRAC_BITS) as i32)
    }
}

impl Q16_16 {

    /// Multiply-accumulate: acc += a * b (exact integer operation).
    /// Accumulates in i64 to avoid overflow.
    #[inline]
    pub fn mac(acc: &mut i64, a: Self, b: Self) {
        *acc += a.0 as i64 * b.0 as i64;
    }

    /// Finalize accumulator: convert i64 accumulator to Q16.16 result.
    #[inline]
    pub fn finalize_acc(acc: i64) -> Self {
        Q16_16((acc >> Q16_FRAC_BITS) as i32)
    }
}

impl Q32_32 {
    /// Convert from f64 to Q32.32.
    #[inline]
    pub fn from_f64(v: f64) -> Self {
        Q32_32((v * Q32_SCALE).round() as i64)
    }

    /// Convert to f64.
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Q32_SCALE
    }

}

impl std::ops::Mul for Q32_32 {
    type Output = Self;

    /// Multiply two Q32.32 values, returning Q32.32.
    /// Uses i128 intermediate to avoid overflow.
    #[inline]
    fn mul(self, other: Self) -> Self {
        let wide = self.0 as i128 * other.0 as i128;
        Q32_32((wide >> Q32_FRAC_BITS) as i64)
    }
}

impl Q32_32 {

    /// Multiply-accumulate: acc += a * b (exact integer operation).
    /// Accumulates in i128.
    #[inline]
    pub fn mac(acc: &mut i128, a: Self, b: Self) {
        *acc += a.0 as i128 * b.0 as i128;
    }

    /// Finalize accumulator.
    #[inline]
    pub fn finalize_acc(acc: i128) -> Self {
        Q32_32((acc >> Q32_FRAC_BITS) as i64)
    }
}

/// Convert a Lloyd-Max codebook to Q16.16 fixed-point.
pub fn codebook_to_q16(centroids: &[f32]) -> Vec<Q16_16> {
    centroids.iter().map(|&c| Q16_16::from_f32(c)).collect()
}

/// Convert a Lloyd-Max codebook to Q32.32 fixed-point.
pub fn codebook_to_q32(centroids: &[f32]) -> Vec<Q32_32> {
    centroids.iter().map(|&c| Q32_32::from_f64(c as f64)).collect()
}

/// Fixed-point dot product: <query, codebook[indices]> using Q16.16 arithmetic.
///
/// The key operation for TurboQuant dequant+dot.
/// Integer accumulation guarantees order-independent results.
pub fn dot_product_q16(
    query: &[f32],
    indices: &[u8],
    codebook_q16: &[Q16_16],
) -> f64 {
    let mut acc: i64 = 0;
    for (i, &idx) in indices.iter().enumerate() {
        let q = Q16_16::from_f32(query[i]);
        let k = codebook_q16[idx as usize];
        Q16_16::mac(&mut acc, q, k);
    }
    Q16_16::finalize_acc(acc).to_f64()
}

/// Fixed-point dot product using Q32.32 arithmetic (higher precision).
pub fn dot_product_q32(
    query: &[f64],
    indices: &[u8],
    codebook_q32: &[Q32_32],
) -> f64 {
    let mut acc: i128 = 0;
    for (i, &idx) in indices.iter().enumerate() {
        let q = Q32_32::from_f64(query[i]);
        let k = codebook_q32[idx as usize];
        Q32_32::mac(&mut acc, q, k);
    }
    Q32_32::finalize_acc(acc).to_f64()
}

/// Compare FP32 vs Q16.16 vs Q32.32 dot product for order-independence.
///
/// Returns (fp32_result, q16_result, q32_result, fp32_order_diff, q16_order_diff, q32_order_diff)
/// where order_diff is |forward_sum - reverse_sum| (should be 0 for integer).
pub fn compare_accumulation_order(
    query: &[f64],
    indices: &[u8],
    centroids: &[f32],
) -> (f64, f64, f64, f64, f64, f64) {
    let d = query.len();
    let cb_q16 = codebook_to_q16(centroids);
    let cb_q32 = codebook_to_q32(centroids);

    // FP32 forward order
    let fp32_fwd: f64 = (0..d)
        .map(|i| query[i] * centroids[indices[i] as usize] as f64)
        .sum();
    // FP32 reverse order
    let fp32_rev: f64 = (0..d)
        .rev()
        .map(|i| query[i] * centroids[indices[i] as usize] as f64)
        .sum();

    // Q16.16 forward
    let mut q16_fwd_acc: i64 = 0;
    for i in 0..d {
        Q16_16::mac(&mut q16_fwd_acc, Q16_16::from_f64(query[i]), cb_q16[indices[i] as usize]);
    }
    // Q16.16 reverse
    let mut q16_rev_acc: i64 = 0;
    for i in (0..d).rev() {
        Q16_16::mac(&mut q16_rev_acc, Q16_16::from_f64(query[i]), cb_q16[indices[i] as usize]);
    }

    // Q32.32 forward
    let mut q32_fwd_acc: i128 = 0;
    for i in 0..d {
        Q32_32::mac(&mut q32_fwd_acc, Q32_32::from_f64(query[i]), cb_q32[indices[i] as usize]);
    }
    // Q32.32 reverse
    let mut q32_rev_acc: i128 = 0;
    for i in (0..d).rev() {
        Q32_32::mac(&mut q32_rev_acc, Q32_32::from_f64(query[i]), cb_q32[indices[i] as usize]);
    }

    let q16_fwd = Q16_16::finalize_acc(q16_fwd_acc).to_f64();
    let q16_rev = Q16_16::finalize_acc(q16_rev_acc).to_f64();
    let q32_fwd = Q32_32::finalize_acc(q32_fwd_acc).to_f64();
    let q32_rev = Q32_32::finalize_acc(q32_rev_acc).to_f64();

    (
        fp32_fwd,
        q16_fwd,
        q32_fwd,
        (fp32_fwd - fp32_rev).abs(),
        (q16_fwd - q16_rev).abs(),  // should be exactly 0
        (q32_fwd - q32_rev).abs(),  // should be exactly 0
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q16_roundtrip() {
        let values = [0.0, 1.0, -1.0, 0.5, -0.25, 0.001, 100.0, -100.0];
        for &v in &values {
            let q = Q16_16::from_f64(v);
            let rt = q.to_f64();
            assert!(
                (v - rt).abs() < 2e-5, // Q16.16 resolution is ~1.5e-5
                "Q16.16 roundtrip error: {} -> {} (diff {})",
                v, rt, (v - rt).abs()
            );
        }
    }

    #[test]
    fn test_q32_roundtrip() {
        let values = [0.0, 1.0, -1.0, 0.5, 0.001, 1e-8, 100.0, -100.0];
        for &v in &values {
            let q = Q32_32::from_f64(v);
            let rt = q.to_f64();
            assert!(
                (v - rt).abs() < 3e-10, // Q32.32 resolution is ~2.3e-10
                "Q32.32 roundtrip error: {} -> {} (diff {})",
                v, rt, (v - rt).abs()
            );
        }
    }

    #[test]
    fn test_q16_multiply() {
        let a = Q16_16::from_f64(2.5);
        let b = Q16_16::from_f64(3.0);
        let c = a * b;
        assert!((c.to_f64() - 7.5).abs() < 1e-4);

        let d = Q16_16::from_f64(-0.5);
        let e = d * b;
        assert!((e.to_f64() - (-1.5)).abs() < 1e-4);
    }

    #[test]
    fn test_q16_mac_exact() {
        // Integer MAC is order-independent
        let vals: Vec<(Q16_16, Q16_16)> = (0..100)
            .map(|i| {
                (
                    Q16_16::from_f64((i as f64 * 0.01).sin()),
                    Q16_16::from_f64((i as f64 * 0.03).cos()),
                )
            })
            .collect();

        // Forward accumulation
        let mut fwd: i64 = 0;
        for &(a, b) in &vals {
            Q16_16::mac(&mut fwd, a, b);
        }

        // Reverse accumulation
        let mut rev: i64 = 0;
        for &(a, b) in vals.iter().rev() {
            Q16_16::mac(&mut rev, a, b);
        }

        // MUST be exactly equal (integer arithmetic)
        assert_eq!(fwd, rev, "Q16.16 MAC not order-independent: fwd={}, rev={}", fwd, rev);
    }

    #[test]
    fn test_q32_mac_exact() {
        let vals: Vec<(Q32_32, Q32_32)> = (0..100)
            .map(|i| {
                (
                    Q32_32::from_f64((i as f64 * 0.01).sin()),
                    Q32_32::from_f64((i as f64 * 0.03).cos()),
                )
            })
            .collect();

        let mut fwd: i128 = 0;
        for &(a, b) in &vals {
            Q32_32::mac(&mut fwd, a, b);
        }
        let mut rev: i128 = 0;
        for &(a, b) in vals.iter().rev() {
            Q32_32::mac(&mut rev, a, b);
        }

        assert_eq!(fwd, rev, "Q32.32 MAC not order-independent");
    }

    #[test]
    fn test_dot_product_q16_matches_fp() {
        let d = 128;
        let codebook = crate::lloyd_max::get_codebook(d, 3);
        let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.05).sin()).collect();
        let indices: Vec<u8> = (0..d).map(|i| (i % 8) as u8).collect();

        let fp_result: f64 = query.iter().zip(indices.iter())
            .map(|(&q, &idx)| q as f64 * codebook.centroids[idx as usize] as f64)
            .sum();
        let q16_result = dot_product_q16(&query, &indices, &codebook_to_q16(&codebook.centroids));

        // Q16.16 should be close to FP result (within Q16 resolution)
        assert!(
            (fp_result - q16_result).abs() < 0.01,
            "Q16 dot mismatch: fp={}, q16={}", fp_result, q16_result
        );
    }

    #[test]
    fn test_accumulation_order_independence() {
        let d = 128;
        let codebook = crate::lloyd_max::get_codebook(d, 3);
        let query: Vec<f64> = (0..d).map(|i| (i as f64 * 0.07).sin()).collect();
        let indices: Vec<u8> = (0..d).map(|i| ((i * 3) % 8) as u8).collect();

        let (fp, q16, q32, fp_diff, q16_diff, q32_diff) =
            compare_accumulation_order(&query, &indices, &codebook.centroids);

        println!("FP64 result:  {:.15}", fp);
        println!("Q16.16 result:{:.15}", q16);
        println!("Q32.32 result:{:.15}", q32);
        println!("FP64 order diff:   {:.2e}", fp_diff);
        println!("Q16.16 order diff: {:.2e}", q16_diff);
        println!("Q32.32 order diff: {:.2e}", q32_diff);

        // Integer arithmetic MUST give zero order difference
        assert_eq!(q16_diff, 0.0, "Q16.16 should be order-independent, diff={}", q16_diff);
        assert_eq!(q32_diff, 0.0, "Q32.32 should be order-independent, diff={}", q32_diff);
        // FP64 may have nonzero order difference (though typically small)
        println!("FP64 order-dependence: {:.2e} (expected small but nonzero)", fp_diff);
    }
}
