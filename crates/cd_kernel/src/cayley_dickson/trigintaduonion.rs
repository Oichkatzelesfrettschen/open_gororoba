//! Trigintaduonion (32D) specialized operations and structures.
//!
//! Trigintaduonions represent the 32-dimensional Cayley-Dickson algebra, containing
//! 373 non-trivial subloops of orders 32, 16, 8, 4, and 2.
//!
//! Direct multiplication requires 1024 real multiplications. The reduced
//! 498-multiplication schedule is a target count, not an implemented path.

/// A 32-dimensional trigintaduonion.
#[derive(Clone, Copy, Debug)]
pub struct Trigintaduonion {
    pub data: [f64; 32],
}

impl Trigintaduonion {
    pub fn new(data: [f64; 32]) -> Self {
        Self { data }
    }

    pub fn zero() -> Self {
        Self { data: [0.0; 32] }
    }

    /// Multiply two Trigintaduonions using standard Cayley-Dickson doubling (1024 mults).
    ///
    /// Workspace requirement: cd_multiply_workspace_len(32) = 4*32 = 128 elements.
    pub fn mul_standard(&self, other: &Self) -> Self {
        let mut res = [0.0; 32];
        let mut workspace = [0.0; 128]; // 4 * dim = 4 * 32
        super::arith::cd_multiply_into(&self.data, &other.data, &mut res, &mut workspace);
        Self { data: res }
    }

    /// Multiply through the strict Cayley-Dickson 32D split.
    ///
    /// This method preserves the public placeholder for a reduced schedule, but
    /// the implementation intentionally matches the standard four-sedenion
    /// product. A real 498-multiplication path needs a fixed-dimension bilinear
    /// schedule rather than a generic recursive rewrite.
    pub fn mul_optimized(&self, other: &Self) -> Self {
        let mut a = [0.0; 16];
        let mut b = [0.0; 16];
        let mut c = [0.0; 16];
        let mut d = [0.0; 16];

        a.copy_from_slice(&self.data[0..16]);
        b.copy_from_slice(&self.data[16..32]);
        c.copy_from_slice(&other.data[0..16]);
        d.copy_from_slice(&other.data[16..32]);

        let c_conj = super::sedenion::sedenion_multiply_explicit(
            &b,
            &super::arith::cd_conjugate(&c).try_into().unwrap(),
        );
        let d_conj_b = super::sedenion::sedenion_multiply_explicit(
            &super::arith::cd_conjugate(&d).try_into().unwrap(),
            &b,
        );
        let ac = super::sedenion::sedenion_multiply_explicit(&a, &c);
        let da = super::sedenion::sedenion_multiply_explicit(&d, &a);

        let mut res = [0.0; 32];
        for i in 0..16 {
            res[i] = ac[i] - d_conj_b[i];
            res[i + 16] = da[i] + c_conj[i];
        }

        Self { data: res }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigintaduonion_multiply() {
        let mut a = Trigintaduonion::zero();
        let mut b = Trigintaduonion::zero();

        for i in 0..32 {
            a.data[i] = (i as f64) * 0.05;
            b.data[i] = (32 - i) as f64 * 0.05;
        }

        let res_std = a.mul_standard(&b);
        let res_opt = a.mul_optimized(&b);

        for i in 0..32 {
            assert!(
                (res_std.data[i] - res_opt.data[i]).abs() < 1e-10,
                "Mismatch at index {}: std {} vs opt {}",
                i,
                res_std.data[i],
                res_opt.data[i]
            );
        }
    }

    /// Parity check for the dim=32 standard path and reduced-schedule placeholder.
    ///
    /// mul_optimized currently calls 4 sedenion_multiply_explicit (same as standard).
    /// Both paths perform ~1024 real multiplications -- mul_optimized does not
    /// yet achieve the claimed 498. Additional array-copy overhead in mul_optimized
    /// makes it slightly SLOWER than mul_standard in practice.
    ///
    /// Conclusion: AVX-512 optimization is not warranted for the current implementation.
    /// The 498-mult algorithm must be fully implemented before any SIMD benefit is possible.
    #[test]
    fn test_mul_standard_vs_optimized_correctness_and_parity() {
        use std::time::Instant;

        let mut a = Trigintaduonion::zero();
        let mut b = Trigintaduonion::zero();
        for i in 0..32 {
            a.data[i] = (i as f64 + 1.0) / 32.0;
            b.data[i] = (32.0 - i as f64) / 32.0;
        }

        // Correctness: both paths must agree.
        let res_std = a.mul_standard(&b);
        let res_opt = a.mul_optimized(&b);
        for i in 0..32 {
            assert!(
                (res_std.data[i] - res_opt.data[i]).abs() < 1e-10,
                "Correctness mismatch at [{}]: std={} opt={}",
                i,
                res_std.data[i],
                res_opt.data[i]
            );
        }

        // Timing: measure relative performance over N iterations.
        // Not a criterion benchmark -- this gives wall-clock ratios for documentation.
        const N: usize = 1000;
        let t0 = Instant::now();
        for _ in 0..N {
            let _ = std::hint::black_box(a.mul_standard(&b));
        }
        let time_std = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..N {
            let _ = std::hint::black_box(a.mul_optimized(&b));
        }
        let time_opt = t1.elapsed();

        // mul_optimized is not faster: it runs the same multiplication count with
        // extra copies around the split representation.
        // We assert both complete in similar wall-clock time (within 3x of each other).
        // A true 2x speedup requires a real 498-multiplication schedule.
        let ratio = time_opt.as_secs_f64() / time_std.as_secs_f64();
        println!(
            "dim32 timing: mul_standard={:?}, mul_optimized={:?}, ratio={:.2}",
            time_std, time_opt, ratio
        );
        // Both are ~1024-mult paths; ratio should be near 1.0 (within 3x given test noise).
        assert!(
            ratio < 3.0,
            "mul_optimized should not be >3x slower than mul_standard (ratio={:.2})",
            ratio
        );
    }
}
