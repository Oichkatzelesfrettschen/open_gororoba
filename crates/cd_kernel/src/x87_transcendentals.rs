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
    // SAFETY: `dst` points at the 10-byte `out` local, which `fstp tbyte`
    // writes exactly once; every x87 stack register is declared clobbered,
    // and `options(nostack)` holds because the sequence never touches the
    // CPU stack.
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

    // SAFETY: `y` and `x` point at live 10-byte `Ext80` values that
    // `fld tbyte` only reads; `out` and `status` point at locals sized
    // for the `fstp tbyte` and `fnstsw word` stores; the full x87 stack
    // is declared clobbered and the sequence never touches the CPU stack.
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

    // SAFETY: `angle` points at a live 10-byte `Ext80` that `fld tbyte`
    // only reads; `sin_out`, `cos_out`, and `status` point at locals
    // sized for their stores; the full x87 stack is declared clobbered
    // and the sequence never touches the CPU stack.
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

    // SAFETY: `modulus` and `dividend` point at live 10-byte `Ext80`
    // values that `fld tbyte` only reads; `out` and `status` point at
    // locals sized for the `fstp tbyte` and `fnstsw word` stores; the
    // full x87 stack is declared clobbered and the single fprem1 step
    // never touches the CPU stack (the C2 retry loop lives in the safe
    // caller `reduce_trig_argument_ext80`).
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

#[must_use]
pub fn angular_separation_arcsec_ext80_deg(
    ra1_deg: f64,
    dec1_deg: f64,
    ra2_deg: f64,
    dec2_deg: f64,
) -> f64 {
    let [x1, y1, z1] = unit_vector_ext80_from_deg(ra1_deg, dec1_deg);
    let [x2, y2, z2] = unit_vector_ext80_from_deg(ra2_deg, dec2_deg);
    let one = Ext80::from_i32(1);
    let mut dot = x1 * x2 + y1 * y2 + z1 * z2;
    dot = Ext80::from_f64(dot.to_f64().clamp(-1.0, 1.0));
    let sin_sq = Ext80::from_f64((one - dot * dot).to_f64().max(0.0));
    let angle = atan2_ext80(sin_sq.sqrt(), dot).value.to_f64();
    let arcsec_per_radian = (Ext80::from_i32(648000) / pi_ext80()).to_f64();
    angle * arcsec_per_radian
}

fn unit_vector_ext80_from_deg(ra_deg: f64, dec_deg: f64) -> [Ext80; 3] {
    let deg_to_rad = pi_ext80() / Ext80::from_i32(180);
    let ra = Ext80::from_f64(ra_deg) * deg_to_rad;
    let dec = Ext80::from_f64(dec_deg) * deg_to_rad;
    let (sin_ra, cos_ra) = sincos_reduced_ext80(ra).value;
    let (sin_dec, cos_dec) = sincos_reduced_ext80(dec).value;
    [cos_dec * cos_ra, cos_dec * sin_ra, sin_dec]
}

#[cfg(test)]
mod tests {
    use super::{
        angular_separation_arcsec_ext80_deg, atan2_ext80, fprem1_ext80, pi_ext80,
        reduce_trig_argument_ext80, sincos_ext80, sincos_reduced_ext80, two_pi_ext80,
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

    #[test]
    fn angular_separation_ext80_matches_close_sky_points() {
        let sep = angular_separation_arcsec_ext80_deg(10.0, 10.0, 10.0001, 10.0001);
        assert!(sep.is_finite());
        assert!(sep > 0.0);
        assert!((sep - 0.505264).abs() < 1.0e-3);
    }
}
