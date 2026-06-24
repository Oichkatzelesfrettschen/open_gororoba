// SPDX-License-Identifier: GPL-2.0-or-later
//
// Proof-extraction fidelity: bind the shipping cd_kernel multiply paths to
// the Cayley-Dickson doubling recurrence that the Rocq proofs reason about.
//
// Rocq CDDoubleFunctor proves properties of the Gallina Definition
// CDDoubleFunctor::cd_mul. Those proofs only constrain the running code to the
// extent the running code computes the same recurrence. This test
// reimplements cd_mul / cd_conj independently from the Rocq Definition (the
// oracle below) and asserts that every production multiply path -- the
// recursive cd_multiply, the SIMD cd_multiply_simd, and the sign-table
// CdMultTable::multiply_via_table -- matches it on seeded random inputs across
// dims 8..256. It then checks the octonion composition identity, which the
// octonion-norm proofs establish for dim 8 (and which sedenions, dim 16+, do
// not satisfy in general because of zero divisors).
//
// Seeded ChaCha8 (explicit u64 seed) keeps the test bit-for-bit reproducible
// on every run and backend, per the numerical-reproducibility discipline.

use cd_kernel::{cd_multiply, cd_multiply_simd, mult_table::CdMultTable};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

// Oracle transcribed from Rocq CDDoubleFunctor::cd_conj:
//   cd_conj (x) = (cd_conj (lo x), cd_neg (hi x))
// which bottoms out at dim 1 as the identity, i.e. negate every imaginary
// component. This must NOT call cd_kernel so the comparison is meaningful.
fn oracle_conj(x: &[f64]) -> Vec<f64> {
    let mut out = x.to_vec();
    for value in out.iter_mut().skip(1) {
        *value = -*value;
    }
    out
}

// Oracle transcribed from Rocq CDDoubleFunctor::cd_mul:
//   (a,b)*(c,d) = (a*c - conj(d)*b,  d*a + b*conj(c))
// with the dim-1 base case being scalar multiplication. lo/hi name the two
// halves exactly as the Gallina mkDouble does.
fn oracle_mul(x: &[f64], y: &[f64]) -> Vec<f64> {
    let dim = x.len();
    assert_eq!(dim, y.len());
    if dim == 1 {
        return vec![x[0] * y[0]];
    }
    let half = dim / 2;
    let (a, b) = x.split_at(half); // x = (a, b)
    let (c, d) = y.split_at(half); // y = (c, d)

    let lo_left = oracle_mul(a, c); // a*c
    let lo_right = oracle_mul(&oracle_conj(d), b); // conj(d)*b
    let hi_left = oracle_mul(d, a); // d*a
    let hi_right = oracle_mul(b, &oracle_conj(c)); // b*conj(c)

    let mut out = Vec::with_capacity(dim);
    for i in 0..half {
        out.push(lo_left[i] - lo_right[i]);
    }
    for i in 0..half {
        out.push(hi_left[i] + hi_right[i]);
    }
    out
}

const DIMS: [usize; 6] = [8, 16, 32, 64, 128, 256];
const PAIRS_PER_DIM: usize = 64;
// f64 with the same recurrence is bit-exact; SIMD and table paths reassociate,
// so allow a tiny relative slack for those. dim<=256 with |components|<=1
// accumulates far below this bound.
const REL_TOL: f64 = 1e-9;
const ABS_TOL: f64 = 1e-12;

fn close(actual: &[f64], expected: &[f64], rel: f64, abs: f64) -> Option<(usize, f64, f64)> {
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let tol = abs + rel * e.abs();
        if (a - e).abs() > tol {
            return Some((i, *a, *e));
        }
    }
    None
}

fn random_element(dim: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
    (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect()
}

/// Every production multiply path computes the proven Rocq doubling recurrence.
#[test]
fn production_multiply_matches_rocq_recurrence() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x0CD0_FACE_2026);
    for &dim in &DIMS {
        // CdMultTable::generate is O(dim^4) (it calls cd_multiply on every
        // basis pair), so only exercise the sign-table path where it is cheap;
        // the recursive and SIMD paths cover the full dim range.
        let table = (dim <= 64).then(|| CdMultTable::generate(dim));
        for _ in 0..PAIRS_PER_DIM {
            let a = random_element(dim, &mut rng);
            let b = random_element(dim, &mut rng);
            let expected = oracle_mul(&a, &b);

            // Recursive reference path: same recurrence, but not coupled to a
            // particular operation schedule.
            let recursive = cd_multiply(&a, &b);
            if let Some((i, got, want)) = close(&recursive, &expected, REL_TOL, ABS_TOL) {
                panic!("cd_multiply dim {dim} idx {i}: got {got}, want {want}");
            }

            // SIMD path: same recurrence, reassociated arithmetic.
            let simd = cd_multiply_simd(&a, &b);
            if let Some((i, got, want)) = close(&simd, &expected, REL_TOL, ABS_TOL) {
                panic!("cd_multiply_simd dim {dim} idx {i}: got {got}, want {want}");
            }

            // Sign-table path: O(dim^2) basis expansion, different summation
            // order, same result.
            if let Some(table) = table.as_ref() {
                let via_table = table.multiply_via_table(&a, &b);
                if let Some((i, got, want)) = close(&via_table, &expected, REL_TOL, ABS_TOL) {
                    panic!("multiply_via_table dim {dim} idx {i}: got {got}, want {want}");
                }
            }
        }
    }
}

/// Octonions (dim 8) and quaternions (dim 4) are composition algebras:
/// |a*b|^2 = |a|^2 * |b|^2. The octonion-norm proofs establish this; the
/// executable must honor it. Sedenions (dim 16+) are excluded because zero
/// divisors break composition.
#[test]
fn composition_identity_holds_through_octonions() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x0C70_0217_2026);
    for &dim in &[4usize, 8usize] {
        for _ in 0..PAIRS_PER_DIM {
            let a = random_element(dim, &mut rng);
            let b = random_element(dim, &mut rng);
            let product = cd_multiply(&a, &b);

            let norm_sq = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
            let lhs = norm_sq(&product);
            let rhs = norm_sq(&a) * norm_sq(&b);
            let tol = ABS_TOL + REL_TOL * rhs.abs();
            assert!(
                (lhs - rhs).abs() <= tol,
                "composition broken at dim {dim}: |ab|^2 = {lhs}, |a|^2|b|^2 = {rhs}"
            );
        }
    }
}
