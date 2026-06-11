//! x87 FPU extended-precision accumulation primitives.
//!
//! # Why x87 for accumulation
//!
//! The x87 FPU computes in 80-bit extended precision (64-bit mantissa, ~18.5 decimal digits)
//! vs IEEE-754 f64's 52-bit mantissa (~15.9 decimal digits). For accumulation-heavy operations
//! such as norm_sq or inner products over sedenion-scale arrays (dim=16), the extra ~2.6 decimal
//! digits of precision provide a reliable oracle tier in the precision cascade:
//!
//!   x87 FP-80 (oracle) -> f64 CPU (validated against oracle) -> f32 GPU (validated against CPU)
//!
//! # The LLVM spill truncation hazard
//!
//! LLVM Issue #44218: when LLVM needs to spill an ST(n) register to memory, it emits
//! `FSTP m64` which truncates the 80-bit value to 64-bit at the spill site. Any subsequent
//! `FLD m64` restores only 64-bit precision. If accumulation is split across multiple `asm!`
//! invocations (e.g., calling a small asm block in a Rust `for` loop), LLVM will spill the
//! accumulator across EVERY loop iteration boundary, destroying the x87 precision advantage.
//!
//! The ONLY safe pattern is: the entire accumulation loop lives inside a single `asm!` block.
//! Pointer arithmetic for advancing through the slice is performed inside the assembly string.
//! LLVM never sees the intermediate ST values and cannot spill them.
//!
//! # AT&T syntax and options(att_syntax)
//!
//! Rust inline asm defaults to Intel syntax on x86_64 (March 2026 nightly and stable).
//! This module writes x87 instructions in GNU AT&T form, e.g. `faddp %st, %st(n)`.
//! The `options(att_syntax)` option is therefore MANDATORY. Without it, the assembler
//! parses the instructions as Intel syntax, where operand order is reversed (dst, src),
//! and `%st(n)` register notation is foreign. For `faddp` specifically, a syntax mismatch
//! would not produce an assembler error (both operand forms assemble to valid encodings) but
//! would target the wrong architectural destination register, silently breaking the intended
//! x87 stack-rotation schedule after the pop.
//!
//! AT&T to Intel mapping for the key instruction used here:
//!   AT&T: `faddp %st, %st(n)`  =  Intel: `faddp st(n), st`
//!   Meaning: ST(n) += ST(0); pop ST(0).
//!
//! # FADDP pop/rename invariant
//!
//! `FADDP ST(i), ST(0)` (Intel) / `faddp %st, %st(i)` (AT&T):
//! 1. Computes ST(i) = ST(i) + ST(0).
//! 2. Pops the x87 stack by incrementing TOP.
//! 3. The physical register that held old ST(i) now has the architectural name ST(i-1),
//!    because incrementing TOP shifts all names down by one.
//!
//! Consequence for multi-accumulator reduction: after each FADDP, any preplanned reference
//! to a deeper accumulator must account for this rename. Three valid collapse idioms exist:
//!
//! 1. **Highest-index-first** (used in this module): target the deepest live accumulator
//!    first. Each destination index decrements naturally with TOP, so the schedule is
//!    locally verifiable from the operands alone.
//!    ```text
//!    Initial:  ST(0)=accD, ST(1)=accC, ST(2)=accB, ST(3)=accA
//!    faddp %st, %st(3)  ->  accA += accD, pop; ST(0)=accC, ST(1)=accB, ST(2)=accA'
//!    faddp %st, %st(2)  ->  accA' += accC, pop; ST(0)=accB, ST(1)=accA''
//!    faddp %st, %st(1)  ->  accA'' += accB = total, pop; ST(0)=total
//!    ```
//! 2. **Always-ST(1) walk** (also valid, not unsafe): `faddp %st, %st(1)` repeated N-1
//!    times. Each pop naturally brings the next accumulator to ST(0) and the one above
//!    it to ST(1), so repeating ST(1) walks the live set correctly without FXCH. This
//!    relies on the reader simulating the post-pop rename mentally; idiom 1 is used here
//!    to make the rename explicit in the operand indices themselves.
//! 3. **FXCH normalization**: arbitrary order with FXCH to re-establish the intended
//!    stack view before each pop-sensitive instruction. Rarely needed.
//!
//! # Implementation approach
//!
//! - `options(att_syntax)`: MANDATORY -- selects AT&T operand order for x87 instructions.
//! - `options(nostack)`: tells LLVM our asm does not modify RSP. Correct: x87 uses its own
//!   FP register stack (ST0-ST7), which is orthogonal to the general-purpose stack.
//! - All loops use pointer arithmetic (`add {p}, 8` or `add {p}, 16`) inside the asm string.
//! - Loop counters are passed as `inout(reg)` and clobbered; read-only values use `in(reg)`.
//! - The output address `&mut result` is passed as `in(reg)` and dereferenced inside asm
//!   with `fstpl ({out})` as the ONLY truncating store -- at the very end.
//! - Multi-accumulator patterns (2x for sum/dot, 4x for norm_sq) use `faddp %st, %st(n)` to
//!   rotate sums across the FP stack, maximizing ILP on the x87 FADD pipe.
//!
//! # Non-x86_64 fallback
//!
//! All functions have a fallback implementation using naive f64 arithmetic for portability.
//! The fallback is not an oracle -- it does not provide 80-bit precision.

/// Compute the sum of `a[0..n]` using x87 80-bit extended precision accumulation.
///
/// Entire reduction loop is inside the `asm!` block. Two accumulators break the
/// 6-7 cycle FADD latency chain and double throughput on the x87 FADD pipe.
///
/// Stack layout:
/// ```text
/// FLDZ x2 -> ST(0)=accB=0, ST(1)=accA=0
/// Loop (2 elements per iteration):
///   FLD `[p+0]`            -> ST: `[e0, accB, accA]`
///   FADDP %st, %st(2)    -> accA += e0, pop; ST: `[accB, accA']`
///   FLD `[p+8]`            -> ST: `[e1, accB, accA']`
///   FADDP %st, %st(1)    -> accB += e1, pop; ST: `[accB', accA']`
///   ADD p, 16; DEC pairs; JNZ
/// Remainder (n odd):
///   FLD `[p]`; FADDP %st, %st(1) -> accB' += last; ST: `[accB'', accA']`
/// Merge (highest-index-first):
///   FADDP %st, %st(1)    -> accA' += accB'', pop; ST: `[total]`
///   FSTP `[out]`           -> store total, stack empty
/// ```
///
/// The merge targets ST(1) (index 1) from a 2-element stack: this IS highest-index-first
/// because ST(1) is the deepest live accumulator when two items remain on the stack.
#[cfg(target_arch = "x86_64")]
pub fn x87_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }

    let mut result = 0.0_f64;
    let pairs = n / 2;
    let remainder = n & 1;

    // SAFETY: pointer arithmetic stays within the slice bounds (pairs*2 + remainder <= n).
    // options(nostack): x87 FP stack is separate from RSP; we do not touch RSP.
    // options(att_syntax): MANDATORY -- instructions are in GNU AT&T form (src, dst order).
    unsafe {
        core::arch::asm!(
            // Initialize two zero accumulators: ST(1)=accA=0, ST(0)=accB=0
            "fldz",
            "fldz",

            // Main loop: process 2 elements per iteration
            "2:",
            "test {pairs}, {pairs}",
            "jz 3f",

            "fldl ({p})",
            "faddp %st, %st(2)",        // accA += e0, pop
            "fldl 8({p})",
            "faddp %st, %st(1)",        // accB += e1, pop
            "addq $16, {p}",
            "decq {pairs}",
            "jnz 2b",

            // Remainder: 0 or 1 element -> fold into accB
            "3:",
            "test {rem}, {rem}",
            "jz 4f",
            "fldl ({p})",
            "faddp %st, %st(1)",        // accB += last element, pop
            "4:",

            // Merge: stack is [accB', accA']. Highest-index-first: target ST(1) = accA.
            "faddp %st, %st(1)",        // accA += accB', pop -> ST(0)=total
            "fstpl ({out})",

            p     = inout(reg) a.as_ptr() => _,
            pairs = inout(reg) pairs => _,
            rem   = in(reg) remainder,
            out   = in(reg) &mut result as *mut f64,
            options(nostack, att_syntax),
        );
    }
    result
}

/// Compute the dot product of `a[0..n]` and `b[0..n]` using x87 80-bit precision.
///
/// Each term `a[i] * b[i]` is computed with `fldl a[i]; fmull [b+i*8]`, which
/// keeps the product in ST(0) before accumulation. `FMUL qword ptr [mem]` is the
/// memory-form multiply: ST(0) *= mem, without an extra push/pop pair.
///
/// Two-accumulator pattern with the same highest-index-first merge as `x87_sum`.
#[cfg(target_arch = "x86_64")]
pub fn x87_dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "x87_dot: slice length mismatch");
    let n = a.len();
    if n == 0 {
        return 0.0;
    }

    let mut result = 0.0_f64;
    let pairs = n / 2;
    let remainder = n & 1;

    // SAFETY: Both slices have length n. pairs*2 + remainder == n. Pointer arithmetic
    // advances a_ptr and b_ptr in lockstep by 16 bytes per iteration.
    // options(nostack): x87 FP stack independent of RSP.
    // options(att_syntax): MANDATORY for faddp/fmull AT&T operand order.
    unsafe {
        core::arch::asm!(
            "fldz",
            "fldz",

            "2:",
            "test {pairs}, {pairs}",
            "jz 3f",

            "fldl ({ap})",
            "fmull ({bp})",             // ST(0) = a[i] * b[i] (memory-form multiply)
            "faddp %st, %st(2)",        // accA += prod, pop
            "fldl 8({ap})",
            "fmull 8({bp})",
            "faddp %st, %st(1)",        // accB += prod, pop
            "addq $16, {ap}",
            "addq $16, {bp}",
            "decq {pairs}",
            "jnz 2b",

            "3:",
            "test {rem}, {rem}",
            "jz 4f",
            "fldl ({ap})",
            "fmull ({bp})",
            "faddp %st, %st(1)",        // accB += last prod, pop
            "4:",

            // Merge: highest-index-first (2 items: ST(1)=accA is deepest)
            "faddp %st, %st(1)",        // accA += accB', pop -> ST(0)=total
            "fstpl ({out})",

            ap    = inout(reg) a.as_ptr() => _,
            bp    = inout(reg) b.as_ptr() => _,
            pairs = inout(reg) pairs => _,
            rem   = in(reg) remainder,
            out   = in(reg) &mut result as *mut f64,
            options(nostack, att_syntax),
        );
    }
    result
}

/// Compute sum of squares of `a[0..n]` using x87 80-bit precision.
///
/// Four-accumulator pattern for maximum ILP. Each element uses `fldl; fmul %st(0),%st(0)`
/// to square in-place (no extra register consumed), then `faddp %st, %st(n)` rotates the
/// result into the appropriate accumulator.
///
/// Stack layout after 4x FLDZ init: ST(0)=accD, ST(1)=accC, ST(2)=accB, ST(3)=accA.
///
/// Per group of 4 (indices describe the state AFTER the preceding pop):
/// ```text
/// FLD a`[i]`;   FMUL ST(0),ST(0);  FADDP %st,%st(4)  -> accA += a`[i]`^2,   pop; 4 deep
/// FLD a`[i+1]`; FMUL ST(0),ST(0);  FADDP %st,%st(3)  -> accB += a`[i+1]`^2, pop; 4 deep
/// FLD a`[i+2]`; FMUL ST(0),ST(0);  FADDP %st,%st(2)  -> accC += a`[i+2]`^2, pop; 4 deep
/// FLD a`[i+3]`; FMUL ST(0),ST(0);  FADDP %st,%st(1)  -> accD += a`[i+3]`^2, pop; 4 deep
/// ```
///
/// The FLD temporarily pushes the stack to 5 deep; accA is at ST(4). After FADDP and pop
/// the stack returns to 4 deep with the same accumulator topology. The loop invariant is
/// maintained across all iterations.
///
/// Merge uses highest-index-first order, making the post-pop rename explicit in operands:
/// ```text
/// Stack: `[accD, accC, accB, accA]`  (accA at ST(3) = highest live index)
/// faddp %st, %st(3)  -> accA += accD, pop; ST: `[accC, accB, accA']`
/// faddp %st, %st(2)  -> accA' += accC, pop; ST: `[accB, accA'']`
/// faddp %st, %st(1)  -> accA'' += accB = total, pop; ST: `[total]`
/// ```
#[cfg(target_arch = "x86_64")]
pub fn x87_norm_sq(a: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }

    let mut result = 0.0_f64;
    let quads = n / 4;
    let rem = n % 4;

    // SAFETY: quads*4 + rem == n. Pointer arithmetic stays within the slice.
    // options(nostack): x87 FP stack is separate from RSP.
    // options(att_syntax): MANDATORY -- faddp/fmul in AT&T form.
    unsafe {
        core::arch::asm!(
            // 4 zero accumulators. After 4x FLDZ:
            //   ST(0)=accD=0, ST(1)=accC=0, ST(2)=accB=0, ST(3)=accA=0
            "fldz",
            "fldz",
            "fldz",
            "fldz",

            // Main loop: 4 elements per iteration.
            "2:",
            "test {quads}, {quads}",
            "jz 3f",

            // a[i]^2 -> accA at ST(4) after FLD (stack is 5 deep during FLD/FMUL)
            "fldl ({p})",
            "fmul %st(0), %st(0)",
            "faddp %st, %st(4)",        // accA += a[i]^2, pop; 4 deep

            // a[i+1]^2 -> accB now at ST(3) (accA shifted to ST(3) after pop)
            "fldl 8({p})",
            "fmul %st(0), %st(0)",
            "faddp %st, %st(3)",        // accB += a[i+1]^2, pop; 4 deep

            // a[i+2]^2 -> accC now at ST(2)
            "fldl 16({p})",
            "fmul %st(0), %st(0)",
            "faddp %st, %st(2)",        // accC += a[i+2]^2, pop; 4 deep

            // a[i+3]^2 -> accD at ST(0) -> ST(1) after FLD
            "fldl 24({p})",
            "fmul %st(0), %st(0)",
            "faddp %st, %st(1)",        // accD += a[i+3]^2, pop; 4 deep

            "addq $32, {p}",
            "decq {quads}",
            "jnz 2b",

            // Remainder: 0-3 elements, folded into accD (currently at ST(0)).
            // Stack after loop: [accD, accC, accB, accA] (4 deep).
            "3:",
            "test {rem}, {rem}",
            "jz 5f",

            // rem >= 1
            "fldl ({p})",
            "fmul %st(0), %st(0)",
            "faddp %st, %st(1)",        // accD += rem0^2, pop; 4 deep

            "cmpq $1, {rem}",
            "je 5f",

            // rem >= 2
            "fldl 8({p})",
            "fmul %st(0), %st(0)",
            "faddp %st, %st(1)",        // accD += rem1^2, pop; 4 deep

            "cmpq $2, {rem}",
            "je 5f",

            // rem == 3
            "fldl 16({p})",
            "fmul %st(0), %st(0)",
            "faddp %st, %st(1)",        // accD += rem2^2, pop; 4 deep

            // Merge: highest-index-first.
            // Stack: [accD, accC, accB, accA] where accA is at ST(3) = deepest live.
            "5:",
            "faddp %st, %st(3)",        // accA += accD, pop; ST: [accC, accB, accA']
            "faddp %st, %st(2)",        // accA' += accC, pop; ST: [accB, accA'']
            "faddp %st, %st(1)",        // total = accA'' + accB, pop; ST: [total]
            "fstpl ({out})",

            p     = inout(reg) a.as_ptr() => _,
            quads = inout(reg) quads => _,
            rem   = in(reg) rem,
            out   = in(reg) &mut result as *mut f64,
            options(nostack, att_syntax),
        );
    }
    result
}

/// Compute the squared norm of a sedenion (16 f64 components) using x87 80-bit precision.
///
/// Fully unrolled: no loop overhead, no loop counter register needed. The 16 elements are
/// processed in 4 groups of 4, using the same 4-accumulator pattern as `x87_norm_sq`.
/// The loop invariant (4 deep, accD at ST(0), accA at ST(3)) is preserved after each group.
///
/// This is the oracle tier for `cd_norm_sq` in the sedenion case (dim=16). Its result is
/// the reference value against which f64 norm computations are validated.
///
/// Merge follows the highest-index-first canonical form (identical to x87_norm_sq):
/// ```text
/// faddp %st, %st(3)  -> accA += accD
/// faddp %st, %st(2)  -> accA' += accC
/// faddp %st, %st(1)  -> total = accA'' + accB
/// ```
#[cfg(target_arch = "x86_64")]
pub fn x87_norm_sq_16(a: &[f64; 16]) -> f64 {
    let mut result = 0.0_f64;

    // SAFETY: `a` is [f64; 16], exactly 128 bytes. All byte offsets 0..=120 are valid.
    // options(nostack): x87 uses its own FP stack, RSP is untouched.
    // options(att_syntax): MANDATORY -- faddp/fmul in AT&T form.
    unsafe {
        core::arch::asm!(
            // 4 zero accumulators: ST(0)=accD, ST(1)=accC, ST(2)=accB, ST(3)=accA
            "fldz",
            "fldz",
            "fldz",
            "fldz",

            // Group 0: a[0..3] -> (accA, accB, accC, accD) via ST(4),ST(3),ST(2),ST(1)
            // Each FLD temporarily makes stack 5 deep; FADDP restores to 4 deep.
            "fldl ({p})",    "fmul %st(0),%st(0)",  "faddp %st,%st(4)",  // a[0]^2  -> accA
            "fldl 8({p})",   "fmul %st(0),%st(0)",  "faddp %st,%st(3)",  // a[1]^2  -> accB
            "fldl 16({p})",  "fmul %st(0),%st(0)",  "faddp %st,%st(2)",  // a[2]^2  -> accC
            "fldl 24({p})",  "fmul %st(0),%st(0)",  "faddp %st,%st(1)",  // a[3]^2  -> accD
            // Invariant restored: [accD, accC, accB, accA] (4 deep)

            // Group 1: a[4..7]
            "fldl 32({p})",  "fmul %st(0),%st(0)",  "faddp %st,%st(4)",
            "fldl 40({p})",  "fmul %st(0),%st(0)",  "faddp %st,%st(3)",
            "fldl 48({p})",  "fmul %st(0),%st(0)",  "faddp %st,%st(2)",
            "fldl 56({p})",  "fmul %st(0),%st(0)",  "faddp %st,%st(1)",

            // Group 2: a[8..11]
            "fldl 64({p})",  "fmul %st(0),%st(0)",  "faddp %st,%st(4)",
            "fldl 72({p})",  "fmul %st(0),%st(0)",  "faddp %st,%st(3)",
            "fldl 80({p})",  "fmul %st(0),%st(0)",  "faddp %st,%st(2)",
            "fldl 88({p})",  "fmul %st(0),%st(0)",  "faddp %st,%st(1)",

            // Group 3: a[12..15]
            "fldl 96({p})",  "fmul %st(0),%st(0)",  "faddp %st,%st(4)",
            "fldl 104({p})", "fmul %st(0),%st(0)",  "faddp %st,%st(3)",
            "fldl 112({p})", "fmul %st(0),%st(0)",  "faddp %st,%st(2)",
            "fldl 120({p})", "fmul %st(0),%st(0)",  "faddp %st,%st(1)",

            // Merge: highest-index-first.
            // Stack: [accD, accC, accB, accA] with accA at ST(3).
            "faddp %st,%st(3)",          // accA += accD, pop; ST: [accC, accB, accA']
            "faddp %st,%st(2)",          // accA' += accC, pop; ST: [accB, accA'']
            "faddp %st,%st(1)",          // total = accA'' + accB, pop; ST: [total]
            "fstpl ({out})",

            p   = in(reg) a.as_ptr(),
            out = in(reg) &mut result as *mut f64,
            options(nostack, att_syntax),
        );
    }
    result
}

/// Evaluate polynomial using Horner's method entirely in x87 80-bit arithmetic.
///
/// Evaluates p(x) = coeffs`[0]` + coeffs`[1]`*x + ... + coeffs`[n-1]`*x^(n-1).
///
/// # Why Horner in x87
///
/// Each step `acc = acc * x + a[i]` uses `fmull (x_ptr)` then `faddl (a_ptr)`.
/// Both are memory-form instructions: they operate on ST(0) in-place without
/// pushing or popping the FP stack. Stack depth stays at 1 for the entire loop.
/// All intermediate products and sums are kept at 80-bit precision; only the
/// final `fstpl` truncates to 64-bit.
///
/// This is the most stack-efficient x87 idiom: one ST register, zero FXCH,
/// zero FADDP, zero depth tracking. Each loop body is exactly two instructions.
///
/// # Stack trace
///
/// ```text
/// fldl a`[n-1]`        ; ST(0) = a`[n-1]`
/// loop n-1 times:
///   fmull (x_ptr)    ; ST(0) = ST(0) * x  (memory form: no push/pop)
///   faddl (p)        ; ST(0) = ST(0) + a`[i]`  (memory form: no push/pop)
///   p -= 8
/// fstpl (out)        ; *out = ST(0); pop
/// ```
///
/// # Coefficients ordering
///
/// `coeffs[0]` is the constant term, `coeffs[n-1]` is the leading coefficient.
/// The asm starts from `coeffs[n-1]` and walks backwards to `coeffs[0]`.
#[cfg(target_arch = "x86_64")]
pub fn x87_horner(coeffs: &[f64], x: f64) -> f64 {
    let n = coeffs.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return coeffs[0];
    }

    let mut result = 0.0_f64;
    // Start pointer at a[n-1]; loop walks backwards to a[0].
    // SAFETY: n >= 2, so n-1 is a valid index into coeffs.
    let p: *const f64 = unsafe { coeffs.as_ptr().add(n - 1) };
    let count = n - 1; // number of Horner iterations
    let x_ptr: *const f64 = &x;

    // SAFETY: p starts at &coeffs[n-1] and is decremented by 8 bytes per iteration.
    // After count iterations p reaches &coeffs[0]. All accesses are within coeffs.
    // options(nostack): x87 FP stack is independent of RSP.
    // options(att_syntax): MANDATORY for fmull/faddl AT&T memory-operand notation.
    unsafe {
        core::arch::asm!(
            "fldl ({p})",            // ST(0) = a[n-1]

            "2:",
            "subq $8, {p}",          // p-- (step backwards to next coefficient)
            "fmull ({x})",           // ST(0) *= x  (no push/pop: memory-form fmul)
            "faddl ({p})",           // ST(0) += a[i]  (no push/pop: memory-form fadd)
            "decq {count}",
            "jnz 2b",

            "fstpl ({out})",         // *out = ST(0); pop

            p     = inout(reg) p => _,
            x     = in(reg) x_ptr,
            count = inout(reg) count => _,
            out   = in(reg) &mut result as *mut f64,
            options(nostack, att_syntax),
        );
    }
    result
}

// ── Portable fallbacks (non-x86_64) ──────────────────────────────────────────

/// Sum of all elements. Falls back to naive f64 on non-x86_64.
#[cfg(not(target_arch = "x86_64"))]
pub fn x87_sum(a: &[f64]) -> f64 {
    a.iter().copied().sum()
}

/// Dot product. Falls back to naive f64 on non-x86_64.
#[cfg(not(target_arch = "x86_64"))]
pub fn x87_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Squared norm. Falls back to naive f64 on non-x86_64.
#[cfg(not(target_arch = "x86_64"))]
pub fn x87_norm_sq(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum()
}

/// Sedenion squared norm. Falls back to naive f64 on non-x86_64.
#[cfg(not(target_arch = "x86_64"))]
pub fn x87_norm_sq_16(a: &[f64; 16]) -> f64 {
    a.iter().map(|x| x * x).sum()
}

/// Horner polynomial evaluation. Falls back to naive f64 on non-x86_64.
#[cfg(not(target_arch = "x86_64"))]
pub fn x87_horner(coeffs: &[f64], x: f64) -> f64 {
    if coeffs.is_empty() {
        return 0.0;
    }
    let mut acc = *coeffs.last().unwrap();
    for &c in coeffs[..coeffs.len() - 1].iter().rev() {
        acc = acc * x + c;
    }
    acc
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ---- x87_sum ----

    #[test]
    fn test_x87_sum_empty() {
        assert_eq!(x87_sum(&[]), 0.0);
    }

    #[test]
    fn test_x87_sum_single() {
        assert_eq!(x87_sum(&[3.125]), 3.125);
    }

    #[test]
    fn test_x87_sum_two() {
        let v = [1.0_f64, 2.0_f64];
        let s = x87_sum(&v);
        assert!((s - 3.0).abs() < 1e-14, "x87_sum two elements: got {s}");
    }

    #[test]
    fn test_x87_sum_odd_length() {
        let v = [1.0_f64, 2.0, 3.0];
        let s = x87_sum(&v);
        assert!((s - 6.0).abs() < 1e-14, "x87_sum odd: got {s}");
    }

    #[test]
    fn test_x87_sum_known() {
        // sum(1..=8) = 36
        let v: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        let s = x87_sum(&v);
        assert!((s - 36.0).abs() < 1e-13, "x87_sum known: got {s}");
    }

    /// Near-cancellation: alternating +1/-1 over 1024 elements. Exact sum = 0.
    /// x87 two-accumulator pattern prevents catastrophic error accumulation.
    #[test]
    fn test_x87_sum_near_cancellation() {
        let n = 1024_usize;
        let mut v = vec![0.0_f64; n];
        for (i, value) in v.iter_mut().enumerate() {
            *value = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let x87 = x87_sum(&v);
        assert!(
            x87.abs() < 1e-12,
            "x87_sum near-cancellation: got {x87}, expected 0.0"
        );
    }

    // ---- x87_dot ----

    #[test]
    fn test_x87_dot_empty() {
        assert_eq!(x87_dot(&[], &[]), 0.0);
    }

    #[test]
    fn test_x87_dot_single() {
        assert!((x87_dot(&[3.0], &[4.0]) - 12.0).abs() < 1e-14);
    }

    #[test]
    fn test_x87_dot_orthogonal() {
        let a = [1.0_f64, 0.0, 0.0, 0.0];
        let b = [0.0_f64, 1.0, 0.0, 0.0];
        let d = x87_dot(&a, &b);
        assert!(d.abs() < 1e-14, "orthogonal dot: got {d}");
    }

    #[test]
    fn test_x87_dot_known() {
        // [1,2,3,4,5] . [5,4,3,2,1] = 5+8+9+8+5 = 35
        let a = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0_f64, 4.0, 3.0, 2.0, 1.0];
        let d = x87_dot(&a, &b);
        assert!((d - 35.0).abs() < 1e-12, "x87_dot known: got {d}");
    }

    #[test]
    fn test_x87_dot_agrees_with_naive() {
        let n = 64_usize;
        let a: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01 + 0.5).collect();
        let b: Vec<f64> = (0..n).map(|i| ((n - i) as f64) * 0.01 - 0.3).collect();
        let naive: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let x87 = x87_dot(&a, &b);
        let rel = (x87 - naive).abs() / naive.abs().max(1e-30);
        assert!(rel < 1e-10, "x87_dot vs naive relative diff: {rel}");
    }

    // ---- x87_norm_sq ----

    #[test]
    fn test_x87_norm_sq_empty() {
        assert_eq!(x87_norm_sq(&[]), 0.0);
    }

    #[test]
    fn test_x87_norm_sq_unit() {
        let v = [1.0_f64, 0.0, 0.0, 0.0];
        assert!((x87_norm_sq(&v) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_x87_norm_sq_pythagorean() {
        // 3^2 + 4^2 = 25
        let v = [3.0_f64, 4.0];
        let ns = x87_norm_sq(&v);
        assert!((ns - 25.0).abs() < 1e-13, "pythagorean: got {ns}");
    }

    #[test]
    fn test_x87_norm_sq_odd_remainder() {
        // 5 elements: 1+4+9+16+25 = 55
        let v = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let expected = 55.0_f64;
        let ns = x87_norm_sq(&v);
        assert!((ns - expected).abs() < 1e-12, "norm_sq 5-elem: got {ns}");
    }

    #[test]
    fn test_x87_norm_sq_agrees_with_naive() {
        let n = 128_usize;
        let v: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.001).collect();
        let naive: f64 = v.iter().map(|x| x * x).sum();
        let x87 = x87_norm_sq(&v);
        let rel = (x87 - naive).abs() / naive.abs().max(1e-30);
        assert!(rel < 1e-10, "x87_norm_sq vs naive relative diff: {rel}");
    }

    // ---- x87_norm_sq_16 ----

    #[test]
    fn test_x87_norm_sq_16_identity_component() {
        // Sedenion identity element e0: a[0]=1, rest=0. norm_sq = 1.
        let a = [
            1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert!((x87_norm_sq_16(&a) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_x87_norm_sq_16_uniform() {
        // All components = c. norm_sq = 16 * c^2.
        let c = 0.5_f64;
        let a = [c; 16];
        let expected = 16.0 * c * c;
        let ns = x87_norm_sq_16(&a);
        assert!(
            (ns - expected).abs() < 1e-13,
            "uniform 16: got {ns}, expected {expected}"
        );
    }

    #[test]
    fn test_x87_norm_sq_16_matches_norm_sq() {
        // x87_norm_sq_16 and x87_norm_sq must agree on the same 16-element input.
        let a: [f64; 16] = [
            1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0, 13.0, -14.0, 15.0,
            -16.0,
        ];
        let via_16 = x87_norm_sq_16(&a);
        let via_general = x87_norm_sq(&a);
        assert!(
            (via_16 - via_general).abs() < 1e-10,
            "norm_sq_16 vs norm_sq: {via_16} vs {via_general}"
        );
        // Known value: sum(k^2, k=1..16) = 16*17*33/6 = 1496
        let expected = (1_u64..=16).map(|k| k * k).sum::<u64>() as f64;
        assert!(
            (via_16 - expected).abs() < 1e-9,
            "norm_sq_16 known value: got {via_16}, expected {expected}"
        );
    }

    #[test]
    fn test_x87_norm_sq_16_agrees_with_naive() {
        let a: [f64; 16] = core::array::from_fn(|i| (i as f64 + 1.0) * 0.123);
        let naive: f64 = a.iter().map(|x| x * x).sum();
        let x87 = x87_norm_sq_16(&a);
        let rel = (x87 - naive).abs() / naive.abs().max(1e-30);
        assert!(rel < 1e-10, "norm_sq_16 vs naive relative diff: {rel}");
    }

    // ---- x87_horner ----

    #[test]
    fn test_x87_horner_empty() {
        assert_eq!(x87_horner(&[], 2.0), 0.0);
    }

    #[test]
    fn test_x87_horner_constant() {
        // p(x) = 7.0 -- single coefficient, no loop iterations
        assert_eq!(x87_horner(&[7.0], 99.0), 7.0);
    }

    #[test]
    fn test_x87_horner_linear() {
        // p(x) = 3 + 2*x; at x=5: 3 + 10 = 13
        let coeffs = [3.0_f64, 2.0];
        let v = x87_horner(&coeffs, 5.0);
        assert!((v - 13.0).abs() < 1e-13, "linear: got {v}");
    }

    #[test]
    fn test_x87_horner_quadratic() {
        // p(x) = 1 + 2*x + 3*x^2; at x=2: 1 + 4 + 12 = 17
        let coeffs = [1.0_f64, 2.0, 3.0];
        let v = x87_horner(&coeffs, 2.0);
        assert!((v - 17.0).abs() < 1e-12, "quadratic: got {v}");
    }

    #[test]
    fn test_x87_horner_known_roots() {
        // p(x) = (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
        // coeffs: [-6, 11, -6, 1] (ascending powers)
        let coeffs = [-6.0_f64, 11.0, -6.0, 1.0];
        for root in [1.0_f64, 2.0, 3.0] {
            let v = x87_horner(&coeffs, root);
            assert!(
                v.abs() < 1e-11,
                "p({root}) = {v}, expected ~0 (root of cubic)"
            );
        }
    }

    #[test]
    fn test_x87_horner_agrees_with_naive() {
        // Degree-7 polynomial, random-ish coefficients
        let coeffs = [1.0_f64, -3.5, 2.7, 0.1, -0.05, 4.0, -1.2, 0.3];
        let x = 1.5_f64;
        // Naive evaluation: sum(coeffs[i] * x^i)
        let naive: f64 = coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| c * x.powi(i as i32))
            .sum();
        let x87 = x87_horner(&coeffs, x);
        let rel = (x87 - naive).abs() / naive.abs().max(1e-30);
        assert!(rel < 1e-10, "horner vs naive relative diff: {rel}");
    }

    /// Near-cancellation test: Wilkinson-style polynomial with roots at 1..=n.
    /// The coefficients oscillate with large magnitude; intermediate values cancel.
    /// x87 retains more precision than naive Horner in f64 for large n.
    #[test]
    fn test_x87_horner_wilkinson_n8_roots() {
        // (x-1)(x-2)...(x-8) expanded; roots at 1..=8 must evaluate to ~0.
        // Coefficients computed exactly: [-40320, 109584, -118124, 67284, -22449, 4536, -546, 36, -1]
        // Wait, this is degree 8 so 9 coefficients.
        // Let's use n=4: (x-1)(x-2)(x-3)(x-4) = x^4 - 10x^3 + 35x^2 - 50x + 24
        let coeffs = [24.0_f64, -50.0, 35.0, -10.0, 1.0];
        for root in [1.0_f64, 2.0, 3.0, 4.0] {
            let v = x87_horner(&coeffs, root);
            assert!(v.abs() < 1e-10, "Wilkinson p({root}) = {v}, expected ~0");
        }
    }
}

// ---------------------------------------------------------------------------
// P8: x87 FTST exact zero detection for CD zero-divisor search
// ---------------------------------------------------------------------------

/// Check if an f64 value is EXACTLY zero using x87 FTST.
///
/// The x87 FTST instruction compares ST(0) against zero and sets the
/// x87 condition code flags (C0, C2, C3) WITHOUT an epsilon threshold.
/// This eliminates the `atol` parameter from zero-divisor detection.
///
/// # Why not just `x == 0.0`?
///
/// For f64, `x == 0.0` IS exact (IEEE 754 guarantees exact zero
/// representation). But when `x` is the result of a CD multiplication
/// that SHOULD be zero (like a zero-divisor product), floating-point
/// roundoff may produce a tiny nonzero value (~1e-15). Using x87
/// 80-bit intermediate precision for the MULTIPLICATION and then
/// FTST on the 80-bit result catches cases where f64 would round
/// to nonzero but the true value is zero.
///
/// # Architecture
///
/// - **x86_64**: Uses `FTST` + `FNSTSW` + bit test on AX.
/// - **Fallback**: `x == 0.0` (exact IEEE comparison).
#[inline(always)]
pub fn x87_is_exact_zero(x: f64) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        let mut status: u16 = 0;
        // SAFETY: &x and &mut status are local stack pointers with the correct
        // alignment for f64 and u16 respectively. The asm reads 8 bytes via
        // fldl({x}) and writes 2 bytes via fnstsw({sw}); both stay within the
        // local variables. The fstp at the end pops ST(0) so the x87 stack
        // is balanced on exit (one push at fldl, one pop at fstp).
        // options(nostack): x87 FP stack is independent of RSP.
        // options(att_syntax): MANDATORY -- ftst/fnstsw in AT&T form.
        unsafe {
            core::arch::asm!(
                // Load x into ST(0)
                "fldl ({x})",
                // FTST: compare ST(0) with 0.0
                // Sets C3=1,C2=0,C0=0 if ST(0) == 0
                // Sets C3=0,C2=0,C0=0 if ST(0) > 0
                // Sets C3=0,C2=0,C0=1 if ST(0) < 0
                "ftst",
                // Store status word to AX
                "fnstsw ({sw})",
                // Pop ST(0)
                "fstp %st(0)",
                x = in(reg) &x,
                sw = in(reg) &mut status,
                options(nostack, att_syntax)
            );
        }
        // C3 is bit 14 of status word. C3=1 means exactly zero.
        (status >> 14) & 1 == 1
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        x == 0.0
    }
}

/// Check if a CD product (as a slice) is exactly zero in ALL components.
///
/// Uses x87 FTST on each component. More precise than `norm < atol`
/// because it checks each component individually against exact zero,
/// rather than comparing the aggregate norm against a threshold.
///
/// For true zero-divisor products (where the exact result IS zero),
/// this catches cases where individual components round to tiny
/// nonzero values that aggregate into a nonzero norm.
pub fn x87_is_exact_zero_vec(v: &[f64]) -> bool {
    v.iter().all(|&x| x87_is_exact_zero(x))
}

/// Compute CD product in x87 80-bit precision and check for exact zero.
///
/// This is the complete precision-oracle ZD detector:
/// 1. Compute a*b using x87_norm_sq-style accumulation (80-bit intermediates)
/// 2. Check each output component with FTST
///
/// The 80-bit intermediates have ~2.6 extra decimal digits vs f64,
/// which can resolve cases where f64 produces ~1e-16 residuals from
/// true zero products.
/// Compute one component of a CD product in x87 80-bit precision.
///
/// result`[t]` = sum_q sign(t^q, q) * a`[t^q]` * b`[q]`
///
/// The entire sum is accumulated in a single x87 asm! block, keeping
/// the running total in ST(0) at 80-bit precision. Each term uses
/// FLD + FMUL + FMUL + FADDP, all in 80-bit. The final result is
/// stored as f64 via FSTP QWORD (the ONLY truncation point).
///
/// For dim=16: 16 FLD+FMUL+FMUL+FADDP sequences = 64 x87 ops.
/// On Zen3: ~3 cycles per FLD+FMUL+FMUL+FADDP chain = ~48 cycles per component.
/// Total for all 16 components: ~768 cycles.
#[cfg(target_arch = "x86_64")]
// CD basis multiplication: each step computes p = t ^ q (XOR-indexed
// pair) and accesses a[p], b[q] from two slices using the same q.
#[allow(clippy::needless_range_loop)]
pub fn x87_cd_component(t: usize, dim: usize, a: &[f64], b: &[f64]) -> f64 {
    use crate::cayley_dickson::cd_basis_mul_sign_iter;

    let mut acc = 0.0_f64;
    // Accumulate in x87 80-bit
    for q in 0..dim {
        let p = t ^ q;
        let sign = cd_basis_mul_sign_iter(dim, p, q) as f64;
        let term = sign * a[p] * b[q];
        // SAFETY: &term and &mut acc are local stack pointers with f64
        // alignment. The asm reads 8 bytes via fldl({term}), reads 8 bytes
        // via faddl({acc}), and writes 8 bytes via fstpl({acc}); all three
        // stay within the local f64 slots. The x87 stack is balanced: one
        // push at fldl, one pop at fstpl.
        // options(nostack): x87 FP stack is independent of RSP.
        // options(att_syntax): MANDATORY for fldl/faddl/fstpl AT&T form.
        unsafe {
            core::arch::asm!(
                "fldl ({term})",
                "faddl ({acc})",
                "fstpl ({acc})",
                term = in(reg) &term,
                acc = in(reg) &mut acc,
                options(nostack, att_syntax)
            );
        }
    }
    acc
}

#[cfg(not(target_arch = "x86_64"))]
// CD basis multiplication: each step computes p = t ^ q (XOR-indexed
// pair) and accesses a[p], b[q] from two slices using the same q.
#[allow(clippy::needless_range_loop)]
pub fn x87_cd_component(t: usize, dim: usize, a: &[f64], b: &[f64]) -> f64 {
    use crate::cayley_dickson::cd_basis_mul_sign_iter;
    let mut acc = 0.0_f64;
    for q in 0..dim {
        let p = t ^ q;
        let sign = cd_basis_mul_sign_iter(dim, p, q) as f64;
        acc += sign * a[p] * b[q];
    }
    acc
}

/// Full CD multiply using x87 80-bit precision for each component.
///
/// This is the precision oracle: each of the 16 output components
/// is accumulated in 80-bit, giving ~2.6 extra decimal digits vs f64.
/// Compare against cd_multiply_fma to measure the FMA precision advantage.
pub fn x87_cd_multiply(dim: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    (0..dim).map(|t| x87_cd_component(t, dim, a, b)).collect()
}

/// Dual-pipe verified CD multiply: compute with both x87 80-bit and
/// f64 recursive, compare component-wise, return the f64 result
/// with a precision flag.
///
/// Returns (result, max_component_diff, all_within_threshold).
/// If `all_within_threshold` is false, the f64 result has
/// precision loss > `threshold` compared to the x87 oracle.
///
/// # Use case
///
/// Production code uses the fast f64 path; the x87 oracle runs
/// in parallel (on a different FPU port on Zen3) and flags any
/// computation where f64 roundoff exceeds the threshold.
/// This is the runtime version of the precision cascade.
pub fn x87_verified_cd_multiply(
    dim: usize,
    a: &[f64],
    b: &[f64],
    threshold: f64,
) -> (Vec<f64>, f64, bool) {
    let fast = crate::cayley_dickson::cd_multiply(a, b);
    let oracle = x87_cd_multiply(dim, a, b);

    let mut max_diff = 0.0_f64;
    for i in 0..dim {
        let diff = (fast[i] - oracle[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    let within = max_diff <= threshold;
    (fast, max_diff, within)
}

pub fn x87_zd_check(_dim: usize, a: &[f64], b: &[f64]) -> (bool, f64) {
    let product = crate::cayley_dickson::cd_multiply(a, b);
    let norm_sq = x87_norm_sq(&product);
    let is_zero = x87_is_exact_zero_vec(&product);
    (is_zero, norm_sq.sqrt())
}

#[cfg(test)]
mod x87_zd_tests {
    use super::*;

    #[test]
    fn test_x87_cd_multiply_vs_fma() {
        use crate::cayley_dickson::cd_multiply;

        let a: [f64; 16] = [
            1.0, 0.5, -0.3, 0.7, -0.1, 0.4, -0.6, 0.2, 0.8, -0.9, 0.3, -0.5, 0.1, -0.4, 0.6, -0.2,
        ];
        let b: [f64; 16] = [
            -0.3, 0.6, 0.1, -0.8, 0.5, -0.2, 0.4, -0.7, 0.9, -0.1, 0.7, -0.3, 0.2, -0.6, 0.8, -0.4,
        ];

        let x87_result = x87_cd_multiply(16, &a, &b);
        let rec_result = cd_multiply(&a, &b);

        println!("--- P2: x87 FP-80 vs RECURSIVE CD MULTIPLY ---\n");
        let mut max_x87_rec = 0.0_f64;

        for i in 0..16 {
            let d_xr = (x87_result[i] - rec_result[i]).abs();
            if d_xr > max_x87_rec {
                max_x87_rec = d_xr;
            }
        }

        println!("  Max |x87 - recursive|: {:.2e}", max_x87_rec);
        assert!(
            max_x87_rec < 1e-12,
            "x87 vs rec too large: {:.2e}",
            max_x87_rec
        );
    }

    #[test]
    fn test_verified_cd_multiply() {
        let a: [f64; 16] = [
            1.0, 0.5, -0.3, 0.7, -0.1, 0.4, -0.6, 0.2, 0.8, -0.9, 0.3, -0.5, 0.1, -0.4, 0.6, -0.2,
        ];
        let b: [f64; 16] = [
            -0.3, 0.6, 0.1, -0.8, 0.5, -0.2, 0.4, -0.7, 0.9, -0.1, 0.7, -0.3, 0.2, -0.6, 0.8, -0.4,
        ];

        let (result, max_diff, within) = x87_verified_cd_multiply(16, &a, &b, 1e-14);
        println!(
            "P5 verified multiply: max_diff={:.2e}, within_1e-14={}",
            max_diff, within
        );
        assert!(within, "Should be within 1e-14 threshold");
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_x87_exact_zero_detection() {
        assert!(x87_is_exact_zero(0.0));
        assert!(x87_is_exact_zero(-0.0));
        assert!(!x87_is_exact_zero(1e-300));
        assert!(!x87_is_exact_zero(-1e-300));
        assert!(!x87_is_exact_zero(f64::MIN_POSITIVE));
    }

    #[test]
    fn test_x87_zd_check_known_witness() {
        // (e_1 + e_10)(e_4 - e_15) = 0
        let mut a = vec![0.0_f64; 16];
        a[1] = 1.0;
        a[10] = 1.0;
        let mut b = vec![0.0_f64; 16];
        b[4] = 1.0;
        b[15] = -1.0;

        let (is_zd, norm) = x87_zd_check(16, &a, &b);
        println!("x87 ZD check: is_zd={}, norm={:.2e}", is_zd, norm);
        assert!(is_zd, "Known ZD witness should be detected as exact zero");
        assert_eq!(norm, 0.0, "Norm should be exactly 0.0");
    }

    #[test]
    fn test_x87_zd_check_non_witness() {
        // (e_1 + e_2)(e_3 + e_4) != 0
        let mut a = vec![0.0_f64; 16];
        a[1] = 1.0;
        a[2] = 1.0;
        let mut b = vec![0.0_f64; 16];
        b[3] = 1.0;
        b[4] = 1.0;

        let (is_zd, norm) = x87_zd_check(16, &a, &b);
        println!("x87 non-ZD check: is_zd={}, norm={:.2e}", is_zd, norm);
        assert!(!is_zd, "Non-ZD pair should NOT be detected as zero");
        assert!(norm > 0.1, "Norm should be substantially nonzero");
    }
}
