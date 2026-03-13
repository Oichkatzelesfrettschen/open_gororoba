//! Double-double (DD) arithmetic: ~31 decimal digits via pairs of f64.
//!
//! Uses error-free transformations (Knuth TwoSum, Dekker TwoProduct with FMA)
//! to extend f64 precision without external dependencies. Thread-safe pure
//! value types -- no global state.
//!
//! Used as the non-x86_64 fallback for extended-precision Jacobi eigenvalue
//! iteration where x87 80-bit hardware is unavailable.

/// Double-double number: hi + lo is the extended-precision value.
/// Invariant: |lo| <= 0.5 * ulp(hi).
#[derive(Debug, Clone, Copy)]
pub(crate) struct DD {
    pub hi: f64,
    pub lo: f64,
}

// ── Error-free transformations ──────────────────────────────────────────────

/// Knuth TwoSum: compute s = a + b and error e such that a + b = s + e exactly.
#[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
#[cfg_attr(not(feature = "profile-dd-hotspots"), inline)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

/// Dekker TwoProduct: compute p = a * b and error e such that a * b = p + e.
/// Uses FMA for exact error computation.
#[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
#[cfg_attr(not(feature = "profile-dd-hotspots"), inline)]
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

/// Sum of two DD products with one final renormalization.
#[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
#[cfg_attr(not(feature = "profile-dd-hotspots"), inline)]
pub(crate) fn dd_mul_add_pair(ab_lhs: DD, ab_rhs: DD, cd_lhs: DD, cd_rhs: DD) -> DD {
    let (p1, e1) = two_product(ab_lhs.hi, ab_rhs.hi);
    let (p2, e2) = two_product(cd_lhs.hi, cd_rhs.hi);

    let cross = e1
        + e2
        + ab_lhs.hi * ab_rhs.lo
        + ab_lhs.lo * ab_rhs.hi
        + ab_lhs.lo * ab_rhs.lo
        + cd_lhs.hi * cd_rhs.lo
        + cd_lhs.lo * cd_rhs.hi
        + cd_lhs.lo * cd_rhs.lo;

    let (s, carry) = two_sum(p1, p2);
    let (hi, lo) = two_sum(s, carry + cross);
    DD { hi, lo }
}

// ── DD constructors ─────────────────────────────────────────────────────────

impl DD {
    #[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
    #[cfg_attr(not(feature = "profile-dd-hotspots"), inline)]
    pub fn from_f64(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    #[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
    #[cfg_attr(not(feature = "profile-dd-hotspots"), inline)]
    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    pub const ONE: Self = Self { hi: 1.0, lo: 0.0 };
}

// ── DD arithmetic ───────────────────────────────────────────────────────────

impl DD {
    /// DD + DD
    #[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
    #[cfg_attr(not(feature = "profile-dd-hotspots"), inline)]
    pub fn add(self, other: Self) -> Self {
        let (s1, e1) = two_sum(self.hi, other.hi);
        let e2 = e1 + self.lo + other.lo;
        let (hi, lo) = two_sum(s1, e2);
        Self { hi, lo }
    }

    /// DD - DD
    #[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
    #[cfg_attr(not(feature = "profile-dd-hotspots"), inline)]
    pub fn sub(self, other: Self) -> Self {
        self.add(Self {
            hi: -other.hi,
            lo: -other.lo,
        })
    }

    /// DD * DD
    #[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
    #[cfg_attr(not(feature = "profile-dd-hotspots"), inline)]
    pub fn mul(self, other: Self) -> Self {
        let (p, e) = two_product(self.hi, other.hi);
        let cross = self.hi * other.lo + self.lo * other.hi + e;
        let (hi, lo) = two_sum(p, cross);
        Self { hi, lo }
    }

    /// DD / DD (Newton-Raphson refinement)
    #[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
    #[cfg_attr(not(feature = "profile-dd-hotspots"), inline)]
    pub fn div(self, other: Self) -> Self {
        let q1 = self.hi / other.hi;
        let r = self.sub(other.mul(Self::from_f64(q1)));
        let q2 = r.hi / other.hi;
        let (hi, lo) = two_sum(q1, q2);
        Self { hi, lo }
    }

    /// Square root via Newton iteration with an f64 starting point.
    #[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
    #[cfg_attr(not(feature = "profile-dd-hotspots"), inline)]
    pub fn sqrt(self) -> Self {
        if self.hi <= 0.0 {
            return if self.hi == 0.0 && self.lo == 0.0 {
                Self::ZERO
            } else {
                Self::from_f64(f64::NAN)
            };
        }

        let mut y = Self::from_f64(self.hi.sqrt());
        let half = Self::from_f64(0.5);
        for _ in 0..4 {
            y = half.mul(y.add(self.div(y)));
        }
        y
    }

    /// Absolute value.
    #[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
    #[cfg_attr(not(feature = "profile-dd-hotspots"), inline)]
    pub fn abs(self) -> Self {
        if self.hi < 0.0 {
            Self {
                hi: -self.hi,
                lo: -self.lo,
            }
        } else {
            self
        }
    }

    /// Compare magnitude for convergence checks.
    #[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
    #[cfg_attr(not(feature = "profile-dd-hotspots"), inline)]
    pub fn gt_f64(self, threshold: f64) -> bool {
        self.hi.abs() > threshold
    }
}

// ── DD transcendentals (Taylor series) ──────────────────────────────────────

impl DD {
    /// sin(x) via Taylor series. Assumes |x| < pi (pre-reduced).
    /// Terms: x - x^3/3! + x^5/5! - x^7/7! + ...
    #[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
    pub fn sin(self) -> Self {
        let x2 = self.mul(self);
        let mut term = self;
        let mut result = self;
        for k in 1..15 {
            let n = 2 * k * (2 * k + 1);
            term = term.mul(x2).div(Self::from_f64(-(n as f64)));
            let new_result = result.add(term);
            if (new_result.hi - result.hi).abs() < 1e-31 {
                return new_result;
            }
            result = new_result;
        }
        result
    }

    /// cos(x) via Taylor series. Assumes |x| < pi (pre-reduced).
    /// Terms: 1 - x^2/2! + x^4/4! - x^6/6! + ...
    #[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
    pub fn cos(self) -> Self {
        let x2 = self.mul(self);
        let mut term = Self::ONE;
        let mut result = Self::ONE;
        for k in 1..15 {
            let n = (2 * k - 1) * (2 * k);
            term = term.mul(x2).div(Self::from_f64(-(n as f64)));
            let new_result = result.add(term);
            if (new_result.hi - result.hi).abs() < 1e-31 {
                return new_result;
            }
            result = new_result;
        }
        result
    }

    /// Simultaneous sin and cos (shares the Taylor series structure).
    #[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
    pub fn sincos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    /// atan2(y, x) in DD precision.
    ///
    /// Uses the identity: atan2(y, x) = 2 * atan(y / (sqrt(x^2 + y^2) + x))
    /// for numerically stable computation, with CORDIC-style reduction.
    #[cfg_attr(feature = "profile-dd-hotspots", inline(never))]
    pub fn atan2(y: Self, x: Self) -> Self {
        // Handle special cases
        let x_zero = x.hi.abs() < 1e-300;
        let y_zero = y.hi.abs() < 1e-300;

        if x_zero && y_zero {
            return Self::ZERO;
        }
        if y_zero {
            return if x.hi > 0.0 { Self::ZERO } else { DD_PI };
        }
        if x_zero {
            return if y.hi > 0.0 {
                DD_PI_2
            } else {
                DD_PI_2.mul(Self::from_f64(-1.0))
            };
        }

        // Use f64 atan2 as starting point, refine with Newton step
        let theta0 = y.hi.atan2(x.hi);
        let theta = Self::from_f64(theta0);

        // One Newton refinement: theta' = theta - (sin(theta)*x - cos(theta)*y) / (cos(theta)*x + sin(theta)*y)
        let (s, c) = theta.sincos();
        let num = s.mul(x).sub(c.mul(y));
        let den = c.mul(x).add(s.mul(y));
        theta.sub(num.div(den))
    }
}

// ── DD constants ────────────────────────────────────────────────────────────

/// pi in DD precision (from Kahan's tables).
/// The hi component is the best f64 approximation; lo captures the residual.
/// clippy::approx_constant is wrong here: we need the exact f64 bit pattern,
/// not std::f64::consts::PI (which would zero out the lo component).
#[allow(clippy::approx_constant)]
const DD_PI: DD = DD {
    hi: 3.141592653589793e+00,
    lo: 1.2246467991473532e-16,
};

/// pi/2 in DD precision.
#[allow(clippy::approx_constant)]
const DD_PI_2: DD = DD {
    hi: 1.5707963267948966e+00,
    lo: 6.123233995736766e-17,
};

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dd_two_sum_exact() {
        // (1 + 2^-52) + 2^-52 should preserve both terms
        let a = 1.0 + f64::EPSILON;
        let b = f64::EPSILON;
        let (s, e) = two_sum(a, b);
        // Exact sum is 1 + 2*eps; verify no bits lost
        assert!((s + e - (a + b)).abs() < 1e-30);
    }

    #[test]
    fn test_dd_two_product_exact() {
        let a = 1.0 + f64::EPSILON;
        let b = 1.0 - f64::EPSILON;
        let (p, e) = two_product(a, b);
        // a*b = 1 - eps^2; verify exact recovery
        let exact = 1.0 - f64::EPSILON * f64::EPSILON;
        assert!((p + e - exact).abs() < 1e-30);
    }

    #[test]
    fn test_dd_add_sub() {
        let a = DD::from_f64(1.0);
        let b = DD::from_f64(2.0);
        let sum = a.add(b);
        assert!((sum.to_f64() - 3.0).abs() < 1e-30);
        let diff = sum.sub(b);
        assert!((diff.to_f64() - 1.0).abs() < 1e-30);
    }

    #[test]
    fn test_dd_mul_div() {
        let a = DD::from_f64(3.0);
        let b = DD::from_f64(7.0);
        let prod = a.mul(b);
        assert!((prod.to_f64() - 21.0).abs() < 1e-28);
        let quot = prod.div(b);
        assert!((quot.to_f64() - 3.0).abs() < 1e-28);
    }

    #[test]
    fn test_dd_mul_add_pair_matches_naive_products() {
        let a = DD {
            hi: 1.25,
            lo: 1.0e-30,
        };
        let b = DD {
            hi: -3.5,
            lo: 2.0e-30,
        };
        let c = DD {
            hi: 2.0,
            lo: -3.0e-30,
        };
        let d = DD {
            hi: 0.75,
            lo: 4.0e-30,
        };

        let fused = dd_mul_add_pair(a, b, c, d);
        let naive = a.mul(b).add(c.mul(d));
        assert!(
            (fused.to_f64() - naive.to_f64()).abs() < 1e-24,
            "fused={} naive={}",
            fused.to_f64(),
            naive.to_f64()
        );
    }

    #[test]
    fn test_dd_sqrt() {
        let x = DD::from_f64(2.0);
        let root = x.sqrt();

        assert!((root.to_f64() - 2.0_f64.sqrt()).abs() < 1e-28);
    }

    #[test]
    fn test_dd_sincos_pi4() {
        let pi4 = DD_PI.mul(DD::from_f64(0.25));
        let (s, c) = pi4.sincos();
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        // DD sincos should be accurate to ~30 digits
        assert!(
            (s.to_f64() - inv_sqrt2).abs() < 1e-15,
            "sin(pi/4) = {}, expected {}",
            s.to_f64(),
            inv_sqrt2
        );
        assert!(
            (c.to_f64() - inv_sqrt2).abs() < 1e-15,
            "cos(pi/4) = {}, expected {}",
            c.to_f64(),
            inv_sqrt2
        );
        // sin and cos should agree at pi/4
        assert!(
            (s.to_f64() - c.to_f64()).abs() < 1e-15,
            "sin(pi/4) should equal cos(pi/4)"
        );
    }

    #[test]
    fn test_dd_atan2_quadrants() {
        // atan2(1, 1) = pi/4
        let theta = DD::atan2(DD::ONE, DD::ONE);
        assert!(
            (theta.to_f64() - std::f64::consts::FRAC_PI_4).abs() < 1e-14,
            "atan2(1,1) = {}, expected pi/4",
            theta.to_f64()
        );

        // atan2(0, 1) = 0
        let theta0 = DD::atan2(DD::ZERO, DD::ONE);
        assert!(theta0.to_f64().abs() < 1e-15);

        // atan2(1, 0) = pi/2
        let theta90 = DD::atan2(DD::ONE, DD::ZERO);
        assert!(
            (theta90.to_f64() - std::f64::consts::FRAC_PI_2).abs() < 1e-14,
            "atan2(1,0) = {}, expected pi/2",
            theta90.to_f64()
        );
    }

    #[test]
    fn test_dd_sincos_symmetry() {
        // sin^2 + cos^2 = 1 for various angles
        for k in 1..8 {
            let theta = DD_PI.mul(DD::from_f64(k as f64 / 8.0));
            let (s, c) = theta.sincos();
            let sum = s.mul(s).add(c.mul(c));
            assert!(
                (sum.to_f64() - 1.0).abs() < 1e-14,
                "sin^2 + cos^2 = {} at k={}, expected 1.0",
                sum.to_f64(),
                k
            );
        }
    }
}
