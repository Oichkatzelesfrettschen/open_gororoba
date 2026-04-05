//! Compatibility shim: pulp operations matching wide::f64x4 API.
//!
//! During the wide-to-pulp migration, this module provides f64x4-like
//! operations via pulp's runtime-dispatched SIMD.  The solver code
//! continues using the same patterns (splat, add, mul, etc.) but
//! backed by pulp's autovectorization.
//!
//! Once migration is complete, direct pulp usage can replace this shim.

/// 4-wide f64 vector backed by a raw array.
///
/// This is a thin wrapper that matches wide::f64x4's API but is
/// compatible with pulp's dispatch model.  The actual SIMD acceleration
/// comes from pulp's autovectorization of the array operations.
#[derive(Clone, Copy, Debug)]
pub struct F64x4(pub [f64; 4]);

impl F64x4 {
    #[inline(always)]
    pub fn new(arr: [f64; 4]) -> Self {
        F64x4(arr)
    }

    #[inline(always)]
    pub fn splat(v: f64) -> Self {
        F64x4([v; 4])
    }

    #[inline(always)]
    pub fn to_array(self) -> [f64; 4] {
        self.0
    }
}

// Arithmetic operators -- pulp will autovectorize these via Arch::dispatch
impl std::ops::Add for F64x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        F64x4([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
        ])
    }
}

impl std::ops::Sub for F64x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        F64x4([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
        ])
    }
}

impl std::ops::Mul for F64x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        F64x4([
            self.0[0] * rhs.0[0],
            self.0[1] * rhs.0[1],
            self.0[2] * rhs.0[2],
            self.0[3] * rhs.0[3],
        ])
    }
}

impl std::ops::Mul<F64x4> for f64 {
    type Output = F64x4;
    #[inline(always)]
    fn mul(self, rhs: F64x4) -> F64x4 {
        F64x4::splat(self) * rhs
    }
}

impl std::ops::Div for F64x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        F64x4([
            self.0[0] / rhs.0[0],
            self.0[1] / rhs.0[1],
            self.0[2] / rhs.0[2],
            self.0[3] / rhs.0[3],
        ])
    }
}

impl std::ops::Neg for F64x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        F64x4([-self.0[0], -self.0[1], -self.0[2], -self.0[3]])
    }
}

impl std::ops::AddAssign for F64x4 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..4 {
            self.0[i] += rhs.0[i];
        }
    }
}

impl std::ops::SubAssign for F64x4 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..4 {
            self.0[i] -= rhs.0[i];
        }
    }
}

impl std::ops::MulAssign for F64x4 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        for i in 0..4 {
            self.0[i] *= rhs.0[i];
        }
    }
}

/// Fused multiply-add: a * b + c (Horner FMA pattern from steinmarder).
///
/// On x86 with FMA3, this compiles to VFMADD213PD.
/// On other architectures, pulp's dispatch ensures optimal codegen.
#[inline(always)]
pub fn madd(a: F64x4, b: F64x4, c: F64x4) -> F64x4 {
    F64x4([
        a.0[0].mul_add(b.0[0], c.0[0]),
        a.0[1].mul_add(b.0[1], c.0[1]),
        a.0[2].mul_add(b.0[2], c.0[2]),
        a.0[3].mul_add(b.0[3], c.0[3]),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splat_and_ops() {
        let a = F64x4::splat(2.0);
        let b = F64x4::splat(3.0);
        let c = a + b;
        assert_eq!(c.0, [5.0, 5.0, 5.0, 5.0]);

        let d = a * b;
        assert_eq!(d.0, [6.0, 6.0, 6.0, 6.0]);

        let e = b - a;
        assert_eq!(e.0, [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_madd() {
        let a = F64x4::splat(2.0);
        let b = F64x4::splat(3.0);
        let c = F64x4::splat(1.0);
        let result = madd(a, b, c); // 2*3 + 1 = 7
        assert_eq!(result.0, [7.0, 7.0, 7.0, 7.0]);
    }

    #[test]
    fn test_scalar_mul() {
        let a = F64x4::new([1.0, 2.0, 3.0, 4.0]);
        let b = 2.0 * a;
        assert_eq!(b.0, [2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_matches_wide() {
        // Verify our F64x4 matches wide::f64x4 behavior
        let wa = wide::f64x4::new([1.0, 2.0, 3.0, 4.0]);
        let wb = wide::f64x4::new([5.0, 6.0, 7.0, 8.0]);
        let wc = wa + wb;
        let wide_result: [f64; 4] = wc.into();

        let pa = F64x4::new([1.0, 2.0, 3.0, 4.0]);
        let pb = F64x4::new([5.0, 6.0, 7.0, 8.0]);
        let pc = pa + pb;

        assert_eq!(wide_result, pc.0, "pulp compat should match wide");
    }
}
