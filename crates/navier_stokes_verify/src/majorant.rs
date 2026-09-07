// SPDX-License-Identifier: MIT

//! Continuous piecewise-affine supersolutions of the H^3 control inequality.
//!
//! For fixed slab bounds, f(r)=a*r+G*r^2+epsilon is convex. A nonnegative
//! affine R with slope b dominates this right-hand side throughout a segment
//! if b >= f(R_start) and b >= f(R_end). These exact inequalities are checked
//! after outward slope quantization. Subdivision is a search strategy only;
//! acceptance depends on the inequalities, not on a numerical ODE integrator.

use crate::interval::{
    ArithmeticError, MAX_PRECISION_BITS, Rational, check_rational, checked_add, checked_div,
    checked_mul, checked_sub, int, round_up_dyadic,
};
use num_traits::Zero;

pub const G3: i64 = 96;
pub const K3: i64 = 64;
const MAX_SUBDIVISION_POWER: u32 = 8;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MajorantSegment {
    pub start: Rational,
    pub end: Rational,
    pub r_start: Rational,
    pub r_end: Rational,
    pub slope: Rational,
}

fn rhs(r: &Rational, a: &Rational, epsilon: &Rational) -> Result<Rational, ArithmeticError> {
    checked_add(
        &checked_add(
            &checked_mul(a, r)?,
            &checked_mul(&int(G3), &checked_mul(r, r)?)?,
        )?,
        epsilon,
    )
}

/// Return None when this bounded search finds no supersolution.
// Each argument corresponds to a quantity in the scalar comparison inequality.
#[allow(clippy::too_many_arguments)]
pub fn enclose_slab(
    start: &Rational,
    end: &Rational,
    initial: &Rational,
    viscosity: &Rational,
    d3: &Rational,
    d4: &Rational,
    epsilon: &Rational,
    precision_bits: u32,
) -> Result<Option<Vec<MajorantSegment>>, ArithmeticError> {
    for value in [start, end, initial, viscosity, d3, d4, epsilon] {
        check_rational(value)?;
    }
    if !(1..=MAX_PRECISION_BITS).contains(&precision_bits) {
        return Err(ArithmeticError::InvalidInput(
            "precision must be in 1..=512",
        ));
    }
    if start < &int(0)
        || end <= start
        || viscosity <= &int(0)
        || [initial, d3, d4, epsilon]
            .iter()
            .any(|value| *value < &int(0))
    {
        return Err(ArithmeticError::InvalidInput(
            "invalid majorant slab or bounds",
        ));
    }
    let a = checked_sub(
        &checked_add(&checked_mul(&int(G3), d3)?, &checked_mul(&int(K3), d4)?)?,
        viscosity,
    )?;
    let duration = checked_sub(end, start)?;
    for power in 0..=MAX_SUBDIVISION_POWER {
        let count = 1_i64 << power;
        let step = checked_div(&duration, &int(count))?;
        let mut segments = Vec::new();
        let mut r = initial.clone();
        let mut time = start.clone();
        for _ in 0..count {
            let f0 = rhs(&r, &a, epsilon)?;
            let slope = if f0 <= int(0) {
                int(0)
            } else {
                round_up_dyadic(&checked_mul(&int(2), &f0)?, precision_bits)?
            };
            let next = checked_add(&r, &checked_mul(&slope, &step)?)?;
            if slope < f0 || slope < rhs(&next, &a, epsilon)? {
                break;
            }
            let next_time = checked_add(&time, &step)?;
            segments.push(MajorantSegment {
                start: time,
                end: next_time.clone(),
                r_start: r,
                r_end: next.clone(),
                slope,
            });
            r = next;
            time = next_time;
        }
        if segments.len() == count as usize {
            return Ok(Some(segments));
        }
    }
    Ok(None)
}

/// The equality case gives a bounded scalar comparison, without a decay claim.
pub fn small_data_threshold(viscosity: &Rational) -> Result<Rational, ArithmeticError> {
    check_rational(viscosity)?;
    if viscosity <= &Rational::zero() {
        return Err(ArithmeticError::InvalidInput("viscosity must be positive"));
    }
    checked_div(viscosity, &int(G3))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interval::rational;

    #[test]
    fn endpoints_and_interior_satisfy_control_inequality() {
        let d = rational(1, 100).unwrap();
        let segments = enclose_slab(
            &int(0),
            &rational(1, 10).unwrap(),
            &d,
            &int(1),
            &d,
            &d,
            &d,
            64,
        )
        .unwrap()
        .unwrap();
        let a = checked_sub(&checked_mul(&int(160), &d).unwrap(), &int(1)).unwrap();
        for segment in &segments {
            assert_eq!(
                checked_add(
                    &segment.r_start,
                    &checked_mul(
                        &segment.slope,
                        &checked_sub(&segment.end, &segment.start).unwrap()
                    )
                    .unwrap()
                )
                .unwrap(),
                segment.r_end
            );
            for j in 0..=10 {
                let r = checked_add(
                    &segment.r_start,
                    &checked_mul(
                        &rational(j, 10).unwrap(),
                        &checked_sub(&segment.r_end, &segment.r_start).unwrap(),
                    )
                    .unwrap(),
                )
                .unwrap();
                assert!(segment.slope >= rhs(&r, &a, &d).unwrap());
            }
        }
        for pair in segments.windows(2) {
            assert_eq!(pair[0].r_end, pair[1].r_start);
            assert_eq!(pair[0].end, pair[1].start);
        }
    }

    #[test]
    fn damping_and_threshold_equality_allow_constant_bound() {
        let r = small_data_threshold(&int(1)).unwrap();
        let segments = enclose_slab(
            &int(0),
            &int(100),
            &r,
            &int(1),
            &int(0),
            &int(0),
            &int(0),
            64,
        )
        .unwrap()
        .unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].slope, int(0));
        assert_eq!(segments[0].r_end, r);
    }

    #[test]
    fn insufficient_subdivision_is_inconclusive() {
        assert!(
            enclose_slab(
                &int(0),
                &int(1),
                &int(1),
                &int(1),
                &int(0),
                &int(0),
                &int(0),
                64
            )
            .unwrap()
            .is_none()
        );
        assert!(
            enclose_slab(
                &int(0),
                &int(1),
                &int(-1),
                &int(1),
                &int(0),
                &int(0),
                &int(0),
                64
            )
            .is_err()
        );
    }
}
