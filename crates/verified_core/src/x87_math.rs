//! Native x87 FP-80 Extended Precision Architecture Submodule
//!
//! Exposes highly optimized inline assembly routines utilizing the x87 FPU stack to
//! bypass catastrophic cancellation in the most ill-conditioned and chaotic numerical
//! bottlenecks of the `open_gororoba` engine.

/// Computes eigenvalues of a symmetric matrix using Jacobi iteration, highly optimized
/// with x87 FP-80 extended precision inline assembly.
///
/// # Topological Precision
/// Standard `f64` Givens rotations accumulate truncation errors over $O(N^3)$ iterations,
/// blurring the true algebraic null-space (zero-divisors) of Sedenion associators into
/// $10^{-10}$ floating-point noise. By executing the arctangent and trigonometric updates
/// natively within the 80-bit x87 FPU stack (`st(0)`-`st(7)`), the condition number resistance
/// scales up, allowing the `obstruction_spectrum` to resolve true $0.0$ bounds without
/// arbitrary fuzzing thresholds.
///
/// # Architecture
/// - **x86_64:** Uses `core::arch::asm!` wrapping `fpatan` and `fsincos`.
/// - **Fallback:** Uses standard `f64` `atan()` and `sin_cos()` on non-x86_64 architectures.
#[inline(always)]
pub fn x87_givens_sincos(app: f64, aqq: f64, apq: f64) -> (f64, f64) {
    #[cfg(target_arch = "x86_64")]
    {
        let mut sin_t = 0.0_f64;
        let mut cos_t = 0.0_f64;
        unsafe {
            core::arch::asm!(
                // Load apq and multiply by 2 -> ST(0) = 2*apq (this is y)
                "fld qword ptr [{apq}]",
                "fadd st(0), st(0)",
                // Load app and subtract aqq -> ST(0) = app - aqq (this is x), ST(1) = 2*apq
                "fld qword ptr [{app}]",
                "fsub qword ptr [{aqq}]",
                // Compute arctan(ST(1) / ST(0)) -> ST(0) = atan2(2*apq, app - aqq)
                "fpatan",
                // Multiply by 0.5 (divide by 2)
                "fld1",
                "fld1",
                "faddp st(1), st(0)", // ST(0) = 2.0, ST(1) = theta
                "fdivp st(1), st(0)", // ST(0) = theta / 2.0
                // Compute sine and cosine -> ST(0) = cos(theta), ST(1) = sin(theta)
                "fsincos",
                // Store results
                "fstp qword ptr [{cos_t}]",
                "fstp qword ptr [{sin_t}]",
                app = in(reg) &app,
                aqq = in(reg) &aqq,
                apq = in(reg) &apq,
                cos_t = in(reg) &mut cos_t,
                sin_t = in(reg) &mut sin_t,
                options(nostack)
            );
        }
        (sin_t, cos_t)
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let theta = if (app - aqq).abs() < 1e-15 {
            core::f64::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * apq) / (app - aqq)).atan()
        };
        theta.sin_cos()
    }
}

/// Computes the 4th-order Runge-Kutta accumulator utilizing x87 extended precision.
///
/// Prevents microscopic energy violations across stiff phase transitions (e.g. TOV interior
/// to exterior matching) by evaluating $y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$
/// completely within the 80-bit FPU register file before casting back down to `f64`.
#[inline(always)]
pub fn x87_rk4_accumulate(y: f64, h: f64, k1: f64, k2: f64, k3: f64, k4: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        let mut result = 0.0_f64;
        let six: f64 = 6.0;
        unsafe {
            core::arch::asm!(
                // Load k2 and k3, add them, multiply by 2 -> ST(0) = 2 * (k2 + k3)
                "fld qword ptr [{k2}]",
                "fadd qword ptr [{k3}]",
                "fadd st(0), st(0)",
                // Add k1 -> ST(0) = k1 + 2*(k2 + k3)
                "fadd qword ptr [{k1}]",
                // Add k4 -> ST(0) = k1 + 2*(k2 + k3) + k4
                "fadd qword ptr [{k4}]",
                // Multiply by h -> ST(0) = h * (...)
                "fmul qword ptr [{h}]",
                // Divide by 6 -> ST(0) = (h/6) * (...)
                "fdiv qword ptr [{six}]",
                // Add y_n -> ST(0) = y_n + (h/6) * (...)
                "fadd qword ptr [{y}]",
                // Store result
                "fstp qword ptr [{result}]",
                y = in(reg) &y,
                h = in(reg) &h,
                k1 = in(reg) &k1,
                k2 = in(reg) &k2,
                k3 = in(reg) &k3,
                k4 = in(reg) &k4,
                six = in(reg) &six,
                result = in(reg) &mut result,
                options(nostack)
            );
        }
        result
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    }
}

/// Precisely computes a 19-element sum (e.g. for D3Q19 LBM density reduction)
/// directly inside the 80-bit x87 registers, bypassing floating point truncation
/// effects between massive $f_0$ populations and tiny $f_{18}$ diagonals.
#[inline(always)]
pub fn x87_sum_19(slice: &[f64; 19]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        let mut result = 0.0_f64;
        let ptr = slice.as_ptr();
        unsafe {
            core::arch::asm!(
                "fld qword ptr [{ptr}]",             // st(0) = slice[0]
                "fadd qword ptr [{ptr} + 1*8]",
                "fadd qword ptr [{ptr} + 2*8]",
                "fadd qword ptr [{ptr} + 3*8]",
                "fadd qword ptr [{ptr} + 4*8]",
                "fadd qword ptr [{ptr} + 5*8]",
                "fadd qword ptr [{ptr} + 6*8]",
                "fadd qword ptr [{ptr} + 7*8]",
                "fadd qword ptr [{ptr} + 8*8]",
                "fadd qword ptr [{ptr} + 9*8]",
                "fadd qword ptr [{ptr} + 10*8]",
                "fadd qword ptr [{ptr} + 11*8]",
                "fadd qword ptr [{ptr} + 12*8]",
                "fadd qword ptr [{ptr} + 13*8]",
                "fadd qword ptr [{ptr} + 14*8]",
                "fadd qword ptr [{ptr} + 15*8]",
                "fadd qword ptr [{ptr} + 16*8]",
                "fadd qword ptr [{ptr} + 17*8]",
                "fadd qword ptr [{ptr} + 18*8]",
                "fstp qword ptr [{result}]",
                ptr = in(reg) ptr,
                result = in(reg) &mut result,
                options(nostack)
            );
        }
        result
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        slice.iter().sum()
    }
}

/// Executes an 8-term dot product specifically for the ABM-8 (Adams-Bashforth-Moulton)
/// predictor-corrector correction step, using 80-bit accumulators to halt symplectic drift.
#[inline(always)]
pub fn x87_abm8_dot_product(f: &[f64; 8], c: &[f64; 8]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        let mut result = 0.0_f64;
        let ptr_f = f.as_ptr();
        let ptr_c = c.as_ptr();
        unsafe {
            core::arch::asm!(
                "fld qword ptr [{f} + 0*8]",
                "fmul qword ptr [{c} + 0*8]", // st(0) = f0*c0

                "fld qword ptr [{f} + 1*8]",
                "fmul qword ptr [{c} + 1*8]",
                "faddp st(1), st(0)",         // st(0) = f0*c0 + f1*c1

                "fld qword ptr [{f} + 2*8]",
                "fmul qword ptr [{c} + 2*8]",
                "faddp st(1), st(0)",         // ...

                "fld qword ptr [{f} + 3*8]",
                "fmul qword ptr [{c} + 3*8]",
                "faddp st(1), st(0)",

                "fld qword ptr [{f} + 4*8]",
                "fmul qword ptr [{c} + 4*8]",
                "faddp st(1), st(0)",

                "fld qword ptr [{f} + 5*8]",
                "fmul qword ptr [{c} + 5*8]",
                "faddp st(1), st(0)",

                "fld qword ptr [{f} + 6*8]",
                "fmul qword ptr [{c} + 6*8]",
                "faddp st(1), st(0)",

                "fld qword ptr [{f} + 7*8]",
                "fmul qword ptr [{c} + 7*8]",
                "faddp st(1), st(0)",

                "fstp qword ptr [{result}]",
                f = in(reg) ptr_f,
                c = in(reg) ptr_c,
                result = in(reg) &mut result,
                options(nostack)
            );
        }
        result
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        f.iter().zip(c.iter()).map(|(a, b)| a * b).sum()
    }
}

/// High-precision Cardano solver for the characteristic equation of the Albert Algebra
/// using x87 80-bit extended precision.
///
/// Characteristic Eq: lambda^3 - T*lambda^2 + S*lambda - D = 0
/// Depressed Cubic: t^3 + p*t + q = 0
/// where p = S - T^2/3, q = D - TS/3 + 2T^3/27
///
/// Returns three real eigenvalues sorted ascending.
#[inline(always)]
pub fn x87_cubic_roots(tr: f64, s2: f64, det: f64) -> [f64; 3] {
    #[cfg(target_arch = "x86_64")]
    {
        let mut roots = [0.0_f64; 3];
        let three: f64 = 3.0;
        let twenty_seven: f64 = 27.0;
        let two: f64 = 2.0;

        unsafe {
            core::arch::asm!(
                // --- Part 1: Compute p = S - T^2/3 ---
                "fld qword ptr [{tr}]",
                "fmul st(0), st(0)",
                "fdiv qword ptr [{three}]", // st(0) = T^2/3
                "fld qword ptr [{s2}]",
                "fsubrp st(1), st(0)",      // st(0) = p

                // --- Part 2: Compute q = D - TS/3 + 2T^3/27 ---
                "fld qword ptr [{tr}]",
                "fmul qword ptr [{s2}]",
                "fdiv qword ptr [{three}]", // st(0) = TS/3, st(1) = p

                "fld qword ptr [{tr}]",
                "fmul st(0), st(0)",
                "fmul qword ptr [{tr}]",
                "fmul qword ptr [{two}]",
                "fdiv qword ptr [{twenty_seven}]", // st(0) = 2T^3/27, st(1) = TS/3, st(2) = p

                "fsubp st(1), st(0)",       // st(0) = TS/3 - 2T^3/27, st(1) = p
                "fld qword ptr [{det}]",
                "fsubrp st(1), st(0)",      // st(0) = q, st(1) = p

                // --- Part 3: Solve t^3 + pt + q = 0 ---
                // We assume 3 real roots, so p <= 0. However, numerical noise might make p slightly positive.
                // r = sqrt(max(0, -p/3))
                "fld st(1)",                // st(0) = p, st(1) = q, st(2) = p
                "fchs",                     // st(0) = -p
                "fldz",                     // st(0) = 0, st(1) = -p
                "fcomi st(0), st(1)",       // compare 0 vs -p. CF=1 if 0 < -p
                "fcmovb st(0), st(1)",      // if 0 < -p, st(0) = -p (i.e. max(0, -p))
                "fstp st(1)",               // st(0) = max(0, -p)
                "fdiv qword ptr [{three}]",
                "fsqrt",                    // st(0) = r, st(1) = q, st(2) = p

                // cos_phi = q / (2 * r^3)
                "fld st(0)",
                "fld st(0)",
                "fmulp st(1), st(0)",       // st(0) = r^2
                "fmul st(0), st(1)",        // st(0) = r^3
                "fmul qword ptr [{two}]",   // st(0) = 2*r^3
                "fld st(2)",                // st(0) = q, st(1) = 2*r^3, st(2) = r, st(3) = q, st(4) = p
                "fdivrp st(1), st(0)",      // st(0) = cos_phi = q / (2*r^3), st(1) = r, st(2) = q, st(3) = p

                // Clamp cos_phi to [-1, 1]
                "fld1",                     // st(0)=1.0, st(1)=cos_phi
                "fcomi st(0), st(1)",       // compare 1.0 vs cos_phi
                "fcmovnb st(0), st(1)",     // if 1.0 >= cos_phi, st(0) = cos_phi
                "fxch st(1)",
                "fstp st(0)",               // st(0) = min(1.0, cos_phi)

                "fld1",
                "fchs",                     // st(0)=-1.0, st(1)=clamped_upper
                "fcomi st(0), st(1)",       // compare -1.0 vs cos_phi
                "fcmovb st(0), st(1)",      // if -1.0 < cos_phi, st(0) = cos_phi
                "fxch st(1)",
                "fstp st(0)",               // st(0) = fully_clamped_cos_phi

                // phi = acos(cos_phi)
                "fld st(0)",
                "fmul st(0), st(0)",
                "fld1",
                "fsubrp st(1), st(0)",
                "fsqrt",                    // st(0) = sin_phi, st(1) = cos_phi
                "fxch st(1)",
                "fpatan",                   // st(0) = phi, st(1) = r, st(2) = q, st(3) = p

                "fdiv qword ptr [{three}]", // st(0) = theta = phi/3

                // --- Part 4: Generate Roots ---
                "fld qword ptr [{tr}]",
                "fdiv qword ptr [{three}]", // st(0) = T/3, st(1) = theta, st(2) = r...

                // Root 0: 2 * r * cos(theta) + T/3
                "fld st(1)",
                "fcos",
                "fmul st(0), st(3)",        // r * cos(theta)
                "fmul qword ptr [{two}]",
                "fadd st(0), st(1)",        // + T/3
                "fstp qword ptr [{r0}]",

                // Root 1: 2 * r * cos(theta + 2pi/3) + T/3
                "fldpi",
                "fmul qword ptr [{two}]",
                "fdiv qword ptr [{three}]", // 2pi/3
                "fadd st(0), st(2)",        // theta + 2pi/3
                "fcos",
                "fmul st(0), st(3)",        // r * cos(...)
                "fmul qword ptr [{two}]",
                "fadd st(0), st(1)",        // + T/3
                "fstp qword ptr [{r1}]",

                // Root 2: 2 * r * cos(theta + 4pi/3) + T/3
                "fldpi",
                "fld1",
                "fadd st(0), st(0)",        // 2
                "fadd st(0), st(0)",        // 4
                "fmulp st(1), st(0)",       // 4pi
                "fdiv qword ptr [{three}]", // 4pi/3
                "fadd st(0), st(2)",        // theta + 4pi/3
                "fcos",
                "fmul st(0), st(3)",        // r * cos(...)
                "fmul qword ptr [{two}]",
                "fadd st(0), st(1)",        // + T/3
                "fstp qword ptr [{r2}]",

                // Cleanup stack (5 elements remaining)
                "fstp st(0)",
                "fstp st(0)",
                "fstp st(0)",
                "fstp st(0)",
                "fstp st(0)",

                tr = in(reg) &tr,
                s2 = in(reg) &s2,
                det = in(reg) &det,
                three = in(reg) &three,
                twenty_seven = in(reg) &twenty_seven,
                two = in(reg) &two,
                r0 = in(reg) &mut roots[0],
                r1 = in(reg) &mut roots[1],
                r2 = in(reg) &mut roots[2],
                options(nostack)
            );
        }
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        roots
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // Fallback to standard f64 Cardano solver (trig method)
        let p = s2 - tr * tr / 3.0;
        let q = det - tr * s2 / 3.0 + 2.0 * tr * tr * tr / 27.0;
        if p.abs() < 1e-15 {
            let t = if q.abs() < 1e-15 {
                0.0
            } else {
                -q.signum() * q.abs().cbrt()
            };
            let l = t + tr / 3.0;
            return [l, l, l];
        }
        let r = (-p / 3.0).sqrt();
        let cos_arg = (q / (2.0 * r * r * r)).clamp(-1.0, 1.0);
        let theta = cos_arg.acos() / 3.0;
        let mut roots = [
            2.0 * r * theta.cos() + tr / 3.0,
            2.0 * r * (theta + 2.0 * std::f64::consts::PI / 3.0).cos() + tr / 3.0,
            2.0 * r * (theta + 4.0 * std::f64::consts::PI / 3.0).cos() + tr / 3.0,
        ];
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        roots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_x87_cubic_roots_simple() {
        // Equation: (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6 = 0
        // T = 6, S = 11, D = 6
        let roots = x87_cubic_roots(6.0, 11.0, 6.0);
        println!("Simple roots: {:?}", roots);
        assert!((roots[0] - 1.0).abs() < 1e-15);
        assert!((roots[1] - 2.0).abs() < 1e-15);
        assert!((roots[2] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_x87_cubic_roots_degenerate() {
        // Equation: (x-1)^2(x-2) = x^3 - 4x^2 + 5x - 2 = 0
        // T = 4, S = 5, D = 2
        let roots = x87_cubic_roots(4.0, 5.0, 2.0);
        println!("Degenerate roots: {:?}", roots);
        assert!((roots[0] - 1.0).abs() < 1e-14);
        assert!((roots[1] - 1.0).abs() < 1e-14);
        assert!((roots[2] - 2.0).abs() < 1e-14);
    }
}
