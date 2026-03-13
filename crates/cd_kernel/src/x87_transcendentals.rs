//! Shared x87 transcendental kernels built on exact `Ext80` storage.
//!
//! These entry points are intentionally medium-granularity kernels: they load
//! exact 80-bit payloads from memory, do the x87 work while values remain on
//! the x87 stack, then store exact 80-bit payloads back to memory together with
//! the observed x87 status word.

#![cfg(target_arch = "x86_64")]

use core::arch::asm;

use crate::x87_ext80::{Ext80, X87StatusWord, X87ValueStatus};
const MAX_FPREM1_ITERATIONS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct X87ReductionResult {
    pub value: Ext80,
    pub status: X87StatusWord,
    pub iterations: usize,
}

#[must_use]
pub fn pi_ext80() -> Ext80 {
    let mut out = Ext80::ZERO;
    unsafe {
        asm!(
            "fldpi",
            "fstp tbyte ptr [{dst}]",
            dst = in(reg) out.as_mut_ptr(),
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
    out
}

#[must_use]
pub fn two_pi_ext80() -> Ext80 {
    let pi = pi_ext80();
    pi + pi
}

#[must_use]
pub fn atan2_ext80(y: Ext80, x: Ext80) -> X87ValueStatus<Ext80> {
    let mut out = Ext80::ZERO;
    let mut status = 0_u16;

    unsafe {
        asm!(
            "fld tbyte ptr [{y}]",
            "fld tbyte ptr [{x}]",
            "fpatan",
            "fnstsw word ptr [{status}]",
            "fstp tbyte ptr [{dst}]",
            y = in(reg) y.as_ptr(),
            x = in(reg) x.as_ptr(),
            status = in(reg) &mut status,
            dst = in(reg) out.as_mut_ptr(),
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

    X87ValueStatus {
        value: out,
        status: X87StatusWord(status),
    }
}

#[must_use]
pub fn sincos_ext80(angle: Ext80) -> X87ValueStatus<(Ext80, Ext80)> {
    let mut sin_out = Ext80::ZERO;
    let mut cos_out = Ext80::ZERO;
    let mut status = 0_u16;

    unsafe {
        asm!(
            "fld tbyte ptr [{angle}]",
            "fsincos",
            "fnstsw word ptr [{status}]",
            "test word ptr [{status}], 0x0400",
            "jnz 2f",
            "fstp tbyte ptr [{cos_dst}]",
            "fstp tbyte ptr [{sin_dst}]",
            "jmp 3f",
            "2:",
            "fstp st(0)",
            "fldz",
            "fstp tbyte ptr [{cos_dst}]",
            "fldz",
            "fstp tbyte ptr [{sin_dst}]",
            "3:",
            angle = in(reg) angle.as_ptr(),
            status = in(reg) &mut status,
            sin_dst = in(reg) sin_out.as_mut_ptr(),
            cos_dst = in(reg) cos_out.as_mut_ptr(),
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

    X87ValueStatus {
        value: (sin_out, cos_out),
        status: X87StatusWord(status),
    }
}

#[must_use]
pub fn fprem1_ext80(dividend: Ext80, modulus: Ext80) -> X87ValueStatus<Ext80> {
    let mut out = Ext80::ZERO;
    let mut status = 0_u16;

    unsafe {
        asm!(
            "fld tbyte ptr [{modulus}]",
            "fld tbyte ptr [{dividend}]",
            "fprem1",
            "fnstsw word ptr [{status}]",
            "fstp tbyte ptr [{dst}]",
            "fstp st(0)",
            dividend = in(reg) dividend.as_ptr(),
            modulus = in(reg) modulus.as_ptr(),
            status = in(reg) &mut status,
            dst = in(reg) out.as_mut_ptr(),
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

    X87ValueStatus {
        value: out,
        status: X87StatusWord(status),
    }
}

#[must_use]
pub fn reduce_trig_argument_ext80(angle: Ext80) -> X87ReductionResult {
    let modulus = two_pi_ext80();
    let mut current = angle;
    let mut iterations = 0;

    loop {
        let step = fprem1_ext80(current, modulus);
        current = step.value;
        iterations += 1;
        if !step.status.condition_code_2() || iterations >= MAX_FPREM1_ITERATIONS {
            return X87ReductionResult {
                value: current,
                status: step.status,
                iterations,
            };
        }
    }
}

#[must_use]
pub fn sincos_reduced_ext80(angle: Ext80) -> X87ValueStatus<(Ext80, Ext80)> {
    let reduction = reduce_trig_argument_ext80(angle);
    if reduction.status.condition_code_2() {
        return X87ValueStatus {
            value: (Ext80::ZERO, Ext80::ZERO),
            status: reduction.status,
        };
    }

    sincos_ext80(reduction.value)
}

#[cfg(test)]
mod tests {
    use super::{
        atan2_ext80, fprem1_ext80, pi_ext80, reduce_trig_argument_ext80, sincos_ext80,
        sincos_reduced_ext80, two_pi_ext80,
    };
    use crate::x87_ext80::Ext80;

    #[test]
    fn atan2_ext80_matches_quadrant_i_reference() {
        let result = atan2_ext80(Ext80::from_f64(1.0), Ext80::from_f64(1.0));
        assert!(!result.status.condition_code_2());
        assert!((result.value.to_f64() - std::f64::consts::FRAC_PI_4).abs() < 1e-15);
    }

    #[test]
    fn sincos_ext80_matches_quarter_pi() {
        let result = sincos_ext80(Ext80::from_f64(std::f64::consts::FRAC_PI_4));
        let (sin_v, cos_v) = result.value;
        let expected = std::f64::consts::FRAC_1_SQRT_2;

        assert!(!result.status.condition_code_2());
        assert!((sin_v.to_f64() - expected).abs() < 1e-15);
        assert!((cos_v.to_f64() - expected).abs() < 1e-15);
    }

    #[test]
    fn sincos_ext80_reports_c2_for_out_of_range_argument() {
        let huge = Ext80::from_i32(1).scale_pow2(80);
        let result = sincos_ext80(huge);
        assert!(result.status.condition_code_2());
    }

    #[test]
    fn fprem1_ext80_reduces_one_turn_plus_quarter_pi() {
        let angle = two_pi_ext80() + Ext80::from_f64(std::f64::consts::FRAC_PI_4);
        let result = fprem1_ext80(angle, two_pi_ext80());

        assert!(!result.status.condition_code_2());
        assert!((result.value.to_f64() - std::f64::consts::FRAC_PI_4).abs() < 1e-14);
    }

    #[test]
    fn reduced_sincos_handles_large_argument() {
        let huge = two_pi_ext80().scale_pow2(80) + Ext80::from_f64(std::f64::consts::FRAC_PI_6);
        let reduction = reduce_trig_argument_ext80(huge);
        let result = sincos_reduced_ext80(huge);
        let (sin_v, cos_v) = result.value;

        assert!(reduction.iterations > 1);
        assert!(!reduction.status.condition_code_2());
        assert!(!result.status.condition_code_2());
        assert!((sin_v.to_f64().powi(2) + cos_v.to_f64().powi(2) - 1.0).abs() < 1e-12);
        assert!(sin_v.to_f64().is_finite());
        assert!(cos_v.to_f64().is_finite());
    }

    #[test]
    fn pi_helpers_are_consistent() {
        let pi = pi_ext80().to_f64();
        let two_pi = two_pi_ext80().to_f64();

        assert!((pi - std::f64::consts::PI).abs() < 1e-15);
        assert!((two_pi - (2.0 * std::f64::consts::PI)).abs() < 1e-15);
    }
}
