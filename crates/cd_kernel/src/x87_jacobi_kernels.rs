//! Reusable x87 Jacobi micro-kernels shared across analysis crates.
//!
//! These kernels sit above the raw `Ext80` transcendental backend:
//! - `x87_atan2_sincos` composes ext80 `FPATAN` and `FSINCOS`
//! - `x87_givens_sincos` derives the half-angle Givens rotation in ext80
//! - `x87_givens_diagonal_update` keeps the 2x2 diagonal-update polynomial in
//!   x87 registers until the final `f64` store

#![cfg(target_arch = "x86_64")]

use core::arch::asm;

use crate::{
    x87_ext80::{Ext80, X87StatusWord, X87ValueStatus},
    x87_transcendentals::{atan2_ext80, sincos_ext80},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct X87JacobiRotation80 {
    pub sin: Ext80,
    pub cos: Ext80,
}

#[must_use]
pub fn givens_sincos_ext80(y: Ext80, x: Ext80) -> X87ValueStatus<X87JacobiRotation80> {
    let angle = atan2_ext80(y, x);
    let half_angle = angle.value.scale_pow2(-1);
    let trig = sincos_ext80(half_angle);
    let (sin_t, cos_t) = trig.value;

    X87ValueStatus {
        value: X87JacobiRotation80 {
            sin: sin_t,
            cos: cos_t,
        },
        status: X87StatusWord(angle.status.0 | trig.status.0),
    }
}

#[must_use]
pub fn givens_sincos_f64(y: f64, x: f64) -> X87ValueStatus<(f64, f64)> {
    let result = givens_sincos_ext80(Ext80::from_f64(y), Ext80::from_f64(x));
    X87ValueStatus {
        value: (result.value.sin.to_f64(), result.value.cos.to_f64()),
        status: result.status,
    }
}

#[must_use]
pub fn x87_atan2_sincos(y: f64, x: f64) -> (f64, f64) {
    let angle = atan2_ext80(Ext80::from_f64(y), Ext80::from_f64(x)).value;
    let (sin, cos) = sincos_ext80(angle).value;
    (sin.to_f64(), cos.to_f64())
}

#[must_use]
pub fn x87_givens_sincos(y: f64, x: f64) -> (f64, f64) {
    givens_sincos_f64(y, x).value
}

#[must_use]
pub fn x87_givens_diagonal_update(
    sin_t: f64,
    cos_t: f64,
    app: f64,
    apq: f64,
    aqq: f64,
) -> (f64, f64) {
    let mut new_pp: f64 = 0.0;
    let mut new_qq: f64 = 0.0;
    // SAFETY: every memory operand is an `in(reg)` pointer to a live f64
    // parameter or to the `new_pp`/`new_qq` locals; loads read 8 bytes and
    // the final stores write exactly those locals; the full x87 register
    // stack is declared clobbered and the sequence never touches the CPU
    // stack.
    unsafe {
        asm!(
            "fld qword ptr [{cos_t}]",
            "fld st(0)",
            "fmulp st(1), st(0)",
            "fld qword ptr [{sin_t}]",
            "fld st(0)",
            "fmulp st(1), st(0)",
            "fld qword ptr [{cos_t}]",
            "fmul qword ptr [{sin_t}]",
            "fadd st(0), st(0)",
            "fld st(2)",
            "fmul qword ptr [{app}]",
            "fld st(1)",
            "fmul qword ptr [{apq}]",
            "faddp st(1), st(0)",
            "fld st(2)",
            "fmul qword ptr [{aqq}]",
            "faddp st(1), st(0)",
            "fstp qword ptr [{new_pp}]",
            "fld st(1)",
            "fmul qword ptr [{app}]",
            "fld st(1)",
            "fmul qword ptr [{apq}]",
            "fsubp st(1), st(0)",
            "fld st(3)",
            "fmul qword ptr [{aqq}]",
            "faddp st(1), st(0)",
            "fstp qword ptr [{new_qq}]",
            "fucompp",
            "fstp st(0)",
            sin_t = in(reg) &sin_t,
            cos_t = in(reg) &cos_t,
            app = in(reg) &app,
            apq = in(reg) &apq,
            aqq = in(reg) &aqq,
            new_pp = in(reg) &mut new_pp,
            new_qq = in(reg) &mut new_qq,
            out("st(0)") _,
            out("st(1)") _,
            out("st(2)") _,
            out("st(3)") _,
            out("st(4)") _,
            out("st(5)") _,
            out("st(6)") _,
            out("st(7)") _,
            options(nostack),
        );
    }
    (new_pp, new_qq)
}

#[cfg(test)]
mod tests {
    use super::{x87_atan2_sincos, x87_givens_diagonal_update, x87_givens_sincos};

    #[test]
    fn x87_atan2_sincos_matches_known_angles() {
        let (sin_val, cos_val) = x87_atan2_sincos(1.0, 1.0);
        let expected = std::f64::consts::FRAC_1_SQRT_2;

        assert!((sin_val - expected).abs() < 1.0e-15);
        assert!((cos_val - expected).abs() < 1.0e-15);

        let (sin_zero, cos_zero) = x87_atan2_sincos(0.0, 1.0);
        assert!(sin_zero.abs() < 1.0e-15);
        assert!((cos_zero - 1.0).abs() < 1.0e-15);
    }

    #[test]
    fn x87_givens_sincos_matches_quarter_pi_case() {
        let (sin_t, cos_t) = x87_givens_sincos(1.0, 1.0);
        let expected_sin = (std::f64::consts::PI / 8.0).sin();
        let expected_cos = (std::f64::consts::PI / 8.0).cos();

        assert!((sin_t - expected_sin).abs() < 1.0e-15);
        assert!((cos_t - expected_cos).abs() < 1.0e-15);
    }

    #[test]
    fn x87_givens_diagonal_matches_f64_reference() {
        let (new_pp, new_qq) = x87_givens_diagonal_update(0.6, 0.8, 10.0, 2.5, 6.0);

        let c2 = 0.8_f64 * 0.8_f64;
        let s2 = 0.6_f64 * 0.6_f64;
        let two_sc = 2.0 * 0.6_f64 * 0.8_f64;
        let ref_pp = c2 * 10.0 + two_sc * 2.5 + s2 * 6.0;
        let ref_qq = s2 * 10.0 - two_sc * 2.5 + c2 * 6.0;

        assert!((new_pp - ref_pp).abs() < 1.0e-13);
        assert!((new_qq - ref_qq).abs() < 1.0e-13);
    }
}
