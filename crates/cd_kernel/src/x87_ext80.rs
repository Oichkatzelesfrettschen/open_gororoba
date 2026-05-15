//! Exact x87 extended-precision storage and control-word wrappers.
//!
//! This module is the foundation layer for a reusable x87 backend:
//! - `Ext80` preserves the exact 80-bit payload across Rust boundaries.
//! - `X87ControlWord` exposes precision and rounding policy explicitly.
//! - `X87StatusWord` exposes x87 condition and exception bits for debugging and
//!   for future status-returning kernels.
//!
//! The current scope is intentionally narrow: exact storage, conversions, core
//! arithmetic, `sqrt`, and power-of-two scaling. Higher-level kernels such as
//! `sincos`, `atan2`, `fprem1` reduction, or interval wrappers should be built
//! on top of this layer so their semantics do not depend on ad hoc `f64`
//! temporaries.

#![cfg(target_arch = "x86_64")]

use core::{
    arch::asm,
    fmt,
    ops::{Add, Div, Mul, Sub},
};

/// Exact 80-bit x87 "double-extended" storage payload.
///
/// `Ext80` keeps the architectural 10-byte memory image instead of immediately
/// truncating back to `f64`. This is the type that should cross kernel
/// boundaries whenever the caller wants true x87 residency semantics.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Ext80 {
    bytes: [u8; 10],
}

impl Ext80 {
    pub const ZERO: Self = Self { bytes: [0; 10] };

    #[must_use]
    pub const fn from_bytes(bytes: [u8; 10]) -> Self {
        Self { bytes }
    }

    #[must_use]
    pub const fn to_bytes(self) -> [u8; 10] {
        self.bytes
    }

    #[must_use]
    pub(crate) const fn as_ptr(&self) -> *const u8 {
        self.bytes.as_ptr()
    }

    #[must_use]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.bytes.as_mut_ptr()
    }

    #[must_use]
    pub fn from_f64(value: f64) -> Self {
        let mut out = Self::ZERO;
        // SAFETY: `fld` loads one `f64`, `fstp` stores one `tbyte` and restores
        // the x87 stack depth before leaving the asm block.
        unsafe {
            asm!(
                "fld qword ptr [{src}]",
                "fstp tbyte ptr [{dst}]",
                src = in(reg) &value,
                dst = in(reg) out.bytes.as_mut_ptr(),
                options(nostack, preserves_flags),
            );
        }
        out
    }

    #[must_use]
    pub fn to_f64(self) -> f64 {
        let mut out = 0.0_f64;
        // SAFETY: `fld` loads one `tbyte`, `fstp` stores one `f64` and restores
        // the x87 stack depth before leaving the asm block.
        unsafe {
            asm!(
                "fld tbyte ptr [{src}]",
                "fstp qword ptr [{dst}]",
                src = in(reg) self.bytes.as_ptr(),
                dst = in(reg) &mut out,
                options(nostack, preserves_flags),
            );
        }
        out
    }

    #[must_use]
    pub fn from_i32(value: i32) -> Self {
        let mut out = Self::ZERO;
        // SAFETY: `fild` loads one 32-bit integer, `fstp` stores one `tbyte`,
        // and the x87 stack depth is restored before the asm block exits.
        unsafe {
            asm!(
                "fild dword ptr [{src}]",
                "fstp tbyte ptr [{dst}]",
                src = in(reg) &value,
                dst = in(reg) out.bytes.as_mut_ptr(),
                options(nostack, preserves_flags),
            );
        }
        out
    }

    #[must_use]
    pub fn from_i64(value: i64) -> Self {
        let mut out = Self::ZERO;
        // SAFETY: `fild` loads one 64-bit integer, `fstp` stores one `tbyte`,
        // and the x87 stack depth is restored before the asm block exits.
        unsafe {
            asm!(
                "fild qword ptr [{src}]",
                "fstp tbyte ptr [{dst}]",
                src = in(reg) &value,
                dst = in(reg) out.bytes.as_mut_ptr(),
                options(nostack, preserves_flags),
            );
        }
        out
    }

    #[must_use]
    pub fn to_i32(self) -> i32 {
        let mut out = 0_i32;
        // SAFETY: `fld` loads one `tbyte`, `fistp` stores one 32-bit integer,
        // and the x87 stack depth is restored before leaving the asm block.
        unsafe {
            asm!(
                "fld tbyte ptr [{src}]",
                "fistp dword ptr [{dst}]",
                src = in(reg) self.bytes.as_ptr(),
                dst = in(reg) &mut out,
                options(nostack, preserves_flags),
            );
        }
        out
    }

    #[must_use]
    pub fn to_i64(self) -> i64 {
        let mut out = 0_i64;
        // SAFETY: `fld` loads one `tbyte`, `fistp` stores one 64-bit integer,
        // and the x87 stack depth is restored before leaving the asm block.
        unsafe {
            asm!(
                "fld tbyte ptr [{src}]",
                "fistp qword ptr [{dst}]",
                src = in(reg) self.bytes.as_ptr(),
                dst = in(reg) &mut out,
                options(nostack, preserves_flags),
            );
        }
        out
    }

    #[must_use]
    pub fn to_i32_trunc(self) -> i32 {
        if std::is_x86_feature_detected!("sse3") {
            return self.to_i32_trunc_sse3();
        }

        let control_word =
            X87ControlWord::read().with_rounding_control(RoundingControl::TowardZero);
        let _guard = X87ControlGuard::set(control_word);
        self.to_i32()
    }

    #[must_use]
    fn to_i32_trunc_sse3(self) -> i32 {
        let mut out = 0_i32;
        // SAFETY: `fld` loads one `tbyte`, `fisttp` truncates to one 32-bit
        // integer without requiring control-word surgery, and the x87 stack
        // depth is restored before leaving the asm block.
        unsafe {
            asm!(
                "fld tbyte ptr [{src}]",
                "fisttp dword ptr [{dst}]",
                src = in(reg) self.bytes.as_ptr(),
                dst = in(reg) &mut out,
                options(nostack, preserves_flags),
            );
        }
        out
    }

    #[must_use]
    pub fn to_i64_trunc(self) -> i64 {
        if std::is_x86_feature_detected!("sse3") {
            return self.to_i64_trunc_sse3();
        }

        let control_word =
            X87ControlWord::read().with_rounding_control(RoundingControl::TowardZero);
        let _guard = X87ControlGuard::set(control_word);
        self.to_i64()
    }

    #[must_use]
    fn to_i64_trunc_sse3(self) -> i64 {
        let mut out = 0_i64;
        // SAFETY: `fld` loads one `tbyte`, `fisttp` truncates to one 64-bit
        // integer without requiring control-word surgery, and the x87 stack
        // depth is restored before leaving the asm block.
        unsafe {
            asm!(
                "fld tbyte ptr [{src}]",
                "fisttp qword ptr [{dst}]",
                src = in(reg) self.bytes.as_ptr(),
                dst = in(reg) &mut out,
                options(nostack, preserves_flags),
            );
        }
        out
    }

    #[must_use]
    pub fn sqrt(self) -> Self {
        let mut out = Self::ZERO;
        // SAFETY: `fld` loads one `tbyte`, `fsqrt` operates in place, `fstp`
        // stores one `tbyte`, and the x87 stack depth is restored.
        unsafe {
            asm!(
                "fld tbyte ptr [{src}]",
                "fsqrt",
                "fstp tbyte ptr [{dst}]",
                src = in(reg) self.bytes.as_ptr(),
                dst = in(reg) out.bytes.as_mut_ptr(),
                options(nostack, preserves_flags),
            );
        }
        out
    }

    #[must_use]
    pub fn scale_pow2(self, exponent: i32) -> Self {
        let mut out = Self::ZERO;
        // SAFETY: `fild` loads the integer exponent into ST(0), `fld` loads the
        // ext80 value into ST(0) and shifts the exponent to ST(1), `fscale`
        // computes x * 2^trunc(exponent), and both stack entries are popped by
        // the end of the block.
        unsafe {
            asm!(
                "fild dword ptr [{exp}]",
                "fld tbyte ptr [{src}]",
                "fscale",
                "fstp tbyte ptr [{dst}]",
                "fstp st(0)",
                exp = in(reg) &exponent,
                src = in(reg) self.bytes.as_ptr(),
                dst = in(reg) out.bytes.as_mut_ptr(),
                options(nostack, preserves_flags),
            );
        }
        out
    }
}

impl Default for Ext80 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl Add for Ext80 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        binary_add(self, rhs)
    }
}

impl Sub for Ext80 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        binary_sub(self, rhs)
    }
}

impl Mul for Ext80 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        binary_mul(self, rhs)
    }
}

impl Div for Ext80 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        binary_div(self, rhs)
    }
}

impl fmt::Debug for Ext80 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ext80")
            .field("bytes", &self.bytes)
            .field("approx_f64", &self.to_f64())
            .finish()
    }
}

#[must_use]
fn binary_add(lhs: Ext80, rhs: Ext80) -> Ext80 {
    let mut out = Ext80::ZERO;
    // SAFETY: lhs/rhs are owned Ext80 (10-byte tbyte) values on the local
    // stack. The asm pushes both onto the x87 FP stack via fld, executes
    // faddp (pops one, leaves the result on top), then fstp pops the
    // result into out. Stack delta: +2 from fld, -2 from faddp+fstp = 0.
    // options(nostack): x87 FP stack is independent of RSP.
    unsafe {
        asm!(
            "fld tbyte ptr [{lhs}]",
            "fld tbyte ptr [{rhs}]",
            "faddp st(1), st(0)",
            "fstp tbyte ptr [{dst}]",
            lhs = in(reg) lhs.bytes.as_ptr(),
            rhs = in(reg) rhs.bytes.as_ptr(),
            dst = in(reg) out.bytes.as_mut_ptr(),
            options(nostack, preserves_flags),
        );
    }
    out
}

#[must_use]
fn binary_sub(lhs: Ext80, rhs: Ext80) -> Ext80 {
    let mut out = Ext80::ZERO;
    // SAFETY: same as binary_add modulo the operator (fsubp instead of
    // faddp). lhs/rhs are 10-byte tbyte stack values; the asm balances
    // the x87 stack push/pop and writes the result into `out`.
    unsafe {
        asm!(
            "fld tbyte ptr [{lhs}]",
            "fld tbyte ptr [{rhs}]",
            "fsubp st(1), st(0)",
            "fstp tbyte ptr [{dst}]",
            lhs = in(reg) lhs.bytes.as_ptr(),
            rhs = in(reg) rhs.bytes.as_ptr(),
            dst = in(reg) out.bytes.as_mut_ptr(),
            options(nostack, preserves_flags),
        );
    }
    out
}

#[must_use]
fn binary_mul(lhs: Ext80, rhs: Ext80) -> Ext80 {
    let mut out = Ext80::ZERO;
    // SAFETY: same as binary_add modulo the operator (fmulp). lhs/rhs are
    // 10-byte tbyte stack values; the asm balances the x87 stack and
    // writes the result into `out`.
    unsafe {
        asm!(
            "fld tbyte ptr [{lhs}]",
            "fld tbyte ptr [{rhs}]",
            "fmulp st(1), st(0)",
            "fstp tbyte ptr [{dst}]",
            lhs = in(reg) lhs.bytes.as_ptr(),
            rhs = in(reg) rhs.bytes.as_ptr(),
            dst = in(reg) out.bytes.as_mut_ptr(),
            options(nostack, preserves_flags),
        );
    }
    out
}

#[must_use]
fn binary_div(lhs: Ext80, rhs: Ext80) -> Ext80 {
    let mut out = Ext80::ZERO;
    // SAFETY: same as binary_add modulo the operator (fdivp). lhs/rhs are
    // 10-byte tbyte stack values; the asm balances the x87 stack and
    // writes the result into `out`. Caller must guarantee rhs != 0; the
    // x87 fdivp will set the FPU divide-by-zero flag otherwise.
    unsafe {
        asm!(
            "fld tbyte ptr [{lhs}]",
            "fld tbyte ptr [{rhs}]",
            "fdivp st(1), st(0)",
            "fstp tbyte ptr [{dst}]",
            lhs = in(reg) lhs.bytes.as_ptr(),
            rhs = in(reg) rhs.bytes.as_ptr(),
            dst = in(reg) out.bytes.as_mut_ptr(),
            options(nostack, preserves_flags),
        );
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum PrecisionControl {
    Single = 0b00,
    Double = 0b10,
    Extended = 0b11,
}

// Bridge to the canonical workspace precision vocabulary
// (gororoba_gpu_bridge::StoragePrecision). Consumers should migrate to
// StoragePrecision directly; this enum is slated for removal in the
// Wave C-tail cleanup PR after all consumers migrate.
impl From<PrecisionControl> for gororoba_gpu_bridge::StoragePrecision {
    fn from(value: PrecisionControl) -> Self {
        match value {
            PrecisionControl::Single => Self::Fp32,
            PrecisionControl::Double => Self::Fp64,
            // Extended (80-bit x87) has no exact StoragePrecision variant;
            // map to DdFp128 as the nearest >64-bit option.
            PrecisionControl::Extended => Self::DdFp128,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum RoundingControl {
    NearestEven = 0b00,
    Down = 0b01,
    Up = 0b10,
    TowardZero = 0b11,
}

/// Raw x87 control-word wrapper.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct X87ControlWord(pub u16);

impl X87ControlWord {
    pub const DEFAULT: Self = Self(0x037F);
    const PRECISION_MASK: u16 = 0b11 << 8;
    const ROUNDING_MASK: u16 = 0b11 << 10;

    #[must_use]
    pub fn read() -> Self {
        let mut value = 0_u16;
        // SAFETY: `fnstcw` stores the current control word to memory without
        // changing the x87 stack or the general-purpose stack.
        unsafe {
            asm!(
                "fnstcw word ptr [{dst}]",
                dst = in(reg) &mut value,
                options(nostack, preserves_flags),
            );
        }
        Self(value)
    }

    pub fn write(self) {
        // SAFETY: `fldcw` loads a new control word from memory and does not
        // alter the x87 stack depth.
        unsafe {
            asm!(
                "fldcw word ptr [{src}]",
                src = in(reg) &self.0,
                options(nostack, preserves_flags),
            );
        }
    }

    #[must_use]
    pub const fn precision_control(self) -> PrecisionControl {
        match (self.0 >> 8) & 0b11 {
            0b00 => PrecisionControl::Single,
            0b10 => PrecisionControl::Double,
            0b11 => PrecisionControl::Extended,
            _ => PrecisionControl::Extended,
        }
    }

    #[must_use]
    pub const fn rounding_control(self) -> RoundingControl {
        match (self.0 >> 10) & 0b11 {
            0b00 => RoundingControl::NearestEven,
            0b01 => RoundingControl::Down,
            0b10 => RoundingControl::Up,
            0b11 => RoundingControl::TowardZero,
            _ => RoundingControl::NearestEven,
        }
    }

    #[must_use]
    pub const fn with_precision_control(self, precision: PrecisionControl) -> Self {
        Self((self.0 & !Self::PRECISION_MASK) | ((precision as u16) << 8))
    }

    #[must_use]
    pub const fn with_rounding_control(self, rounding: RoundingControl) -> Self {
        Self((self.0 & !Self::ROUNDING_MASK) | ((rounding as u16) << 10))
    }
}

impl fmt::Debug for X87ControlWord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("X87ControlWord")
            .field("raw", &format_args!("{:#06x}", self.0))
            .field("precision", &self.precision_control())
            .field("rounding", &self.rounding_control())
            .finish()
    }
}

/// RAII guard for temporary x87 control-word changes.
///
/// This is the safe surface for coarse-grained control-word policy changes:
/// create the guard, run one or more x87 kernels, then let the guard restore the
/// previous control word on drop.
pub struct X87ControlGuard {
    saved: X87ControlWord,
}

impl X87ControlGuard {
    #[must_use]
    pub fn set(new_control_word: X87ControlWord) -> Self {
        let saved = X87ControlWord::read();
        new_control_word.write();
        Self { saved }
    }
}

impl Drop for X87ControlGuard {
    fn drop(&mut self) {
        self.saved.write();
    }
}

/// Raw x87 status-word wrapper.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct X87StatusWord(pub u16);

impl X87StatusWord {
    #[must_use]
    pub fn read() -> Self {
        let mut value = 0_u16;
        // SAFETY: `fnstsw` stores the status word to memory without altering
        // the x87 stack depth.
        unsafe {
            asm!(
                "fnstsw word ptr [{dst}]",
                dst = in(reg) &mut value,
                options(nostack, preserves_flags),
            );
        }
        Self(value)
    }

    #[must_use]
    pub const fn exception_flags(self) -> u8 {
        (self.0 & 0x003F) as u8
    }

    #[must_use]
    pub const fn condition_code_0(self) -> bool {
        self.0 & (1 << 8) != 0
    }

    #[must_use]
    pub const fn condition_code_1(self) -> bool {
        self.0 & (1 << 9) != 0
    }

    #[must_use]
    pub const fn condition_code_2(self) -> bool {
        self.0 & (1 << 10) != 0
    }

    #[must_use]
    pub const fn top(self) -> u8 {
        ((self.0 >> 11) & 0b111) as u8
    }

    #[must_use]
    pub const fn condition_code_3(self) -> bool {
        self.0 & (1 << 14) != 0
    }

    #[must_use]
    pub const fn busy(self) -> bool {
        self.0 & (1 << 15) != 0
    }
}

impl fmt::Debug for X87StatusWord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("X87StatusWord")
            .field("raw", &format_args!("{:#06x}", self.0))
            .field("exception_flags", &self.exception_flags())
            .field("c0", &self.condition_code_0())
            .field("c1", &self.condition_code_1())
            .field("c2", &self.condition_code_2())
            .field("top", &self.top())
            .field("c3", &self.condition_code_3())
            .field("busy", &self.busy())
            .finish()
    }
}

/// Generic carrier for x87 kernels that return both a value and a status word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct X87ValueStatus<T> {
    pub value: T,
    pub status: X87StatusWord,
}

#[cfg(test)]
mod tests {
    use super::{Ext80, PrecisionControl, RoundingControl, X87ControlGuard, X87ControlWord};

    #[test]
    fn ext80_round_trips_f64() {
        let value = Ext80::from_f64(-3.25);
        assert_eq!(value.to_f64(), -3.25);
    }

    #[test]
    fn ext80_round_trips_i64() {
        let value = Ext80::from_i64(123_456_789);
        assert_eq!(value.to_i64(), 123_456_789);
    }

    #[test]
    fn ext80_truncating_integer_conversions_use_fisttp() {
        let value = Ext80::from_f64(-12.75);
        assert_eq!(value.to_i32_trunc(), -12);
        assert_eq!(value.to_i64_trunc(), -12);
    }

    #[test]
    fn ext80_retains_extra_precision_over_f64_chain() {
        let sum = Ext80::from_f64(1.0e16) + Ext80::from_f64(1.0) + Ext80::from_f64(-1.0e16);

        assert_eq!(sum.to_f64(), 1.0);
        assert_eq!((1.0e16_f64 + 1.0) - 1.0e16_f64, 0.0);
    }

    #[test]
    fn ext80_scale_pow2_matches_ldexp_behavior() {
        let scaled = Ext80::from_f64(1.5).scale_pow2(3);
        assert_eq!(scaled.to_f64(), 12.0);
    }

    #[test]
    fn control_guard_restores_previous_control_word() {
        let saved = X87ControlWord::read();
        let modified = saved
            .with_precision_control(PrecisionControl::Extended)
            .with_rounding_control(RoundingControl::TowardZero);

        {
            let _guard = X87ControlGuard::set(modified);
            let current = X87ControlWord::read();
            assert_eq!(current.precision_control(), PrecisionControl::Extended);
            assert_eq!(current.rounding_control(), RoundingControl::TowardZero);
        }

        assert_eq!(X87ControlWord::read().0, saved.0);
    }
}
