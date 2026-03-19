//! Kramers-Kronig transformations for optical materials.
//!
//! Provides routines to reconstruct the real part of the dielectric function
//! from its imaginary part using Standard, Singly-Subtractive (SSKK),
//! and Multi-Anchor Subtractive (MSKK) methods.

use std::f64::consts::PI;

/// Simple trapezoidal numerical integration.
fn trapezoid(x: &[f64], y: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..(x.len() - 1) {
        sum += (x[i + 1] - x[i]) * (y[i + 1] + y[i]) * 0.5;
    }
    sum
}

/// Standard Kramers-Kronig relation (imaginary to real).
///
/// eps1(w) = eps_inf + (2/pi) * P int_0^inf [w' * eps2(w') / (w'^2 - w^2)] dw'
pub fn kk_standard_from_im(
    omega: &[f64],
    im_eps: &[f64],
    omega_eval: &[f64],
    eps_inf: f64,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(omega_eval.len());
    for &we in omega_eval {
        let mut integrand = Vec::with_capacity(omega.len());
        let we2 = we * we;
        for (i, &w) in omega.iter().enumerate() {
            let denom = w * w - we2;
            if denom.abs() < 1e-15 {
                if i > 0 && i < omega.len() - 1 {
                    let prev = omega[i-1] * im_eps[i-1] / (omega[i-1] * omega[i-1] - we2);
                    let next = omega[i+1] * im_eps[i+1] / (omega[i+1] * omega[i+1] - we2);
                    integrand.push((prev + next) * 0.5);
                } else {
                    integrand.push(0.0);
                }
            } else {
                integrand.push(w * im_eps[i] / denom);
            }
        }
        let integral = trapezoid(omega, &integrand);
        out.push(eps_inf + (2.0 / PI) * integral);
    }
    out
}

/// Singly-Subtractive Kramers-Kronig (SSKK).
///
/// eps1(w) = eps1(w0) + (2/pi) * (w^2 - w0^2) * P int_0^inf [w' * eps2(w') / ((w'^2 - w^2)(w'^2 - w0^2))] dw'
pub fn kk_sskk_from_im(
    omega: &[f64],
    im_eps: &[f64],
    omega_eval: &[f64],
    omega0: f64,
    eps1_omega0: f64,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(omega_eval.len());
    let w02 = omega0 * omega0;
    for &we in omega_eval {
        let mut integrand = Vec::with_capacity(omega.len());
        let we2 = we * we;
        for (i, &w) in omega.iter().enumerate() {
            let w2 = w * w;
            let denom = (w2 - we2) * (w2 - w02);
            if denom.abs() < 1e-15 {
                if i > 0 && i < omega.len() - 1 {
                    let prev_w2 = omega[i-1] * omega[i-1];
                    let next_w2 = omega[i+1] * omega[i+1];
                    let prev = omega[i-1] * im_eps[i-1] / ((prev_w2 - we2) * (prev_w2 - w02));
                    let next = omega[i+1] * im_eps[i+1] / ((next_w2 - we2) * (next_w2 - w02));
                    integrand.push((prev + next) * 0.5);
                } else {
                    integrand.push(0.0);
                }
            } else {
                integrand.push(w * im_eps[i] / denom);
            }
        }
        let integral = trapezoid(omega, &integrand);
        out.push(eps1_omega0 + (2.0 * (we2 - w02) / PI) * integral);
    }
    out
}

/// Multi-Anchor Subtractive Kramers-Kronig (MSKK-2).
///
/// eps1(w) = L(w) + (2/pi) * (w^2-w0^2)(w^2-w1^2) * P int_0^inf [w' * eps2(w') / ((w'^2-w^2)(w'^2-w0^2)(w'^2-w1^2))] dw'
pub fn kk_mskk2_from_im(
    omega: &[f64],
    im_eps: &[f64],
    omega_eval: &[f64],
    omega0: f64,
    omega1: f64,
    eps1_omega0: f64,
    eps1_omega1: f64,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(omega_eval.len());
    let w02 = omega0 * omega0;
    let w12 = omega1 * omega1;

    for &we in omega_eval {
        let we2 = we * we;
        
        let l_we = eps1_omega0 * (we2 - w12) / (w02 - w12) 
                 + eps1_omega1 * (we2 - w02) / (w12 - w02);

        let mut integrand = Vec::with_capacity(omega.len());
        for (i, &w) in omega.iter().enumerate() {
            let w2 = w * w;
            let denom = (w2 - we2) * (w2 - w02) * (w2 - w12);
            if denom.abs() < 1e-15 {
                if i > 0 && i < omega.len() - 1 {
                    let prev_w2 = omega[i-1] * omega[i-1];
                    let next_w2 = omega[i+1] * omega[i+1];
                    let prev = omega[i-1] * im_eps[i-1] / ((prev_w2 - we2) * (prev_w2 - w02) * (prev_w2 - w12));
                    let next = omega[i+1] * im_eps[i+1] / ((next_w2 - we2) * (next_w2 - w02) * (next_w2 - w12));
                    integrand.push((prev + next) * 0.5);
                } else {
                    integrand.push(0.0);
                }
            } else {
                integrand.push(w * im_eps[i] / denom);
            }
        }
        let integral = trapezoid(omega, &integrand);
        let prefactor = (2.0 / PI) * (we2 - w02) * (we2 - w12);
        out.push(l_we + prefactor * integral);
    }
    out
}
