// SPDX-License-Identifier: MIT

//! Analytic contract for normalized homogeneous H^3 continuum certificates.
//!
//! Let B(v,w)=-P[(v.grad)w] on the 2*pi torus with normalized volume.
//! For real, zero-mean, solenoidal fields the cutoff-independent bounds are
//! ||B(v,w)||_3 <= K3 ||v||_3 ||w||_4 and
//! |<B(v,w),w>_3| <= G3 ||v||_3 ||w||_3^2, with K3=64 and G3=96.
//!
//! To obtain these conservative constants, put S4=sum_(k!=0)|k|^-4.
//! The infinity-norm shell n contains 24*n^2+2 points, so
//! S4 <= 24*zeta(2)+2*zeta(4) < 48+8/3 < 64. The integral test gives
//! zeta(2)<2 and zeta(4)<4/3 without numerical evaluation. Discrete Young and
//! Cauchy-Schwarz, together with |p+q|^3<=4(|p|^3+|q|^3), give
//! K3 <= 8*sqrt(S4) <=64. In the energy pairing the real solenoidal transport
//! term cancels. The remaining commutator multiplier satisfies
//! ||p+q|^3-|q|^3| <=6|p|^3+6|p||q|^2; its two Young bounds each cost
//! 6*sqrt(S4), giving G3<=96. Projection has Euclidean norm at most one.
//!
//! The energy error e=u-v therefore obeys the upper Dini inequality
//! d+||e||_3 <= (-nu+G3*D3+K3*D4)||e||_3+G3||e||_3^2+epsilon3.
//! Zero mean gives the first Laplacian eigenvalue one. Local existence,
//! Sobolev continuation, and scalar comparison turn a finite supersolution
//! into continuum existence and an error bound. Apply comparison separately
//! on exactly joined polynomial slabs; derivative matching is unnecessary.
//! These analytic results are trusted mathematics, not Rocq-checked code.
//! See Morosi--Pizzocchero, Sections 2, 4 and 5:
//! <https://arxiv.org/html/1405.3421v4>. Their Fourier normalization differs;
//! the integer constants above are derived directly for this module's norm.
//!
//! If curl(v0)=v0, the vector identity
//! (v0.grad)v0=grad(|v0|^2/2)-v0 cross curl(v0) makes the projected
//! nonlinearity vanish. Curl squared equals -Delta on solenoidal fields,
//! so Delta v0=-v0 and v(t)=exp(-nu*t)v0 solves the equation exactly.
//! Unit-wave support gives D3=D4=M exp(-nu*t) and epsilon3=0.
//! For c=160*M, A(t)=-nu*t+(c/nu)(1-exp(-nu*t)), scalar comparison gives
//! R=delta*exp(A)/(1-96*delta*integral_0^t exp(A)). The complete integral
//! is (exp(c/nu)-1)/c. Thus delta<c/[96*(exp(c/nu)-1)] certifies global
//! smooth existence for every smooth real solenoidal zero-mean datum in
//! that open H^3 ball. The c=0 limit is nu/96. The radius decreases in c,
//! so an upper M and upper exponential give a rigorous lower radius.
//!
//! Trust boundary: these analytic arguments and their continuum hypotheses;
//! the exact-arithmetic Rust implementation; num-bigint/num-rational and
//! dependencies; parsing, serialization, hashing, compiler and runtime.
//! Rational operations are exact and transcendental endpoints are enclosed.
//! Resource exhaustion produces no certificate. No proof-assistant or
//! machine-code correspondence is asserted. Source and lockfile hashes
//! identify the implementation; reproduction is computation, not formal proof.

use crate::{
    interval::{
        ArithmeticError, MAX_PRECISION_BITS, Rational, check_rational, checked_div, checked_mul,
        checked_sub, exp_enclosure, int,
    },
    majorant::{G3, K3, small_data_threshold},
};

/// Rigorous sufficient open-ball radius using an upper bound on ||v0||_3.
pub fn radius_lower(
    viscosity: &Rational,
    m_upper: &Rational,
    precision_bits: u32,
) -> Result<Rational, ArithmeticError> {
    check_rational(viscosity)?;
    check_rational(m_upper)?;
    if !(1..=MAX_PRECISION_BITS).contains(&precision_bits) {
        return Err(ArithmeticError::InvalidInput(
            "precision must be in 1..=512",
        ));
    }
    if viscosity <= &int(0) || m_upper < &int(0) {
        return Err(ArithmeticError::InvalidInput(
            "invalid Beltrami viscosity or norm",
        ));
    }
    if m_upper == &int(0) {
        return small_data_threshold(viscosity);
    }
    let c = checked_mul(&int(G3 + K3), m_upper)?;
    let exp_upper = exp_enclosure(&checked_div(&c, viscosity)?, precision_bits)?.upper;
    checked_div(
        &c,
        &checked_mul(&int(G3), &checked_sub(&exp_upper, &int(1))?)?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interval::rational;

    #[test]
    fn zero_reference_limit_and_monotonicity() {
        let zero = radius_lower(&int(1), &int(0), 80).unwrap();
        assert_eq!(zero, rational(1, 96).unwrap());
        let small = radius_lower(&int(1), &rational(1, 1000).unwrap(), 80).unwrap();
        let larger = radius_lower(&int(1), &rational(1, 100).unwrap(), 80).unwrap();
        assert!(zero > small && small > larger && larger > int(0));
    }

    #[test]
    fn exponential_resource_limit_does_not_certify() {
        assert!(radius_lower(&int(1), &int(10), 80).is_err());
    }
}
