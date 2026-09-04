//! Physical decomposition of the six-sample staple window.
//!
//! The CD associator on THEMIS-A is a cubic mismatch on six FGM samples.
//! This module splits that window into magnetosphere-facing channels so a
//! ranking can be attributed to rotation, |B| jump, lag-mixing, or a single
//! fiber slot rather than to "sedenion structure" as a whole.
//!
//! Packing: staple index = 4 * lag + channel, channel 0=Bx, 1=By, 2=Bz, 3=|B|.
//! Indices 0..3 are lag 0 and also the quaternion subalgebra. Indices 0..7
//! are two lags and the octonion subalgebra. Index 8 is lag-2 Bx, the CD
//! doubling unit occupying a time slot.

use crate::staple_associator::STAPLE_DIM;
use crate::staple_controls::SparseCubicTensor;

/// Keep a subset of the 16 staple components; others go to 0.
pub fn mask_staple(v: &[f64; STAPLE_DIM], keep: impl Fn(usize) -> bool) -> [f64; STAPLE_DIM] {
    let mut out = [0.0_f64; STAPLE_DIM];
    for (i, &x) in v.iter().enumerate() {
        if keep(i) {
            out[i] = x;
        }
    }
    out
}

pub fn zero_magnitude_channels(v: &[f64; STAPLE_DIM]) -> [f64; STAPLE_DIM] {
    mask_staple(v, |i| i % 4 != 3)
}

pub fn quaternion_lag0_only(v: &[f64; STAPLE_DIM]) -> [f64; STAPLE_DIM] {
    mask_staple(v, |i| i < 4)
}

pub fn octonion_two_lag_only(v: &[f64; STAPLE_DIM]) -> [f64; STAPLE_DIM] {
    mask_staple(v, |i| i < 8)
}

pub fn zero_doubling_slot(v: &[f64; STAPLE_DIM]) -> [f64; STAPLE_DIM] {
    mask_staple(v, |i| i != 8)
}

/// max |B| - min |B| on the six samples that feed associator index k.
pub fn mag_jump(rows: &[[f64; 3]], k: usize) -> f64 {
    if k + 5 >= rows.len() {
        return 0.0;
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for row in rows.iter().skip(k).take(6) {
        let mag = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
        lo = lo.min(mag);
        hi = hi.max(mag);
    }
    (hi - lo).max(0.0)
}

/// Score a staple triple after applying `mask` to each vector.
pub fn masked_normalized_score(
    tensor: &SparseCubicTensor,
    a: &[f64; STAPLE_DIM],
    b: &[f64; STAPLE_DIM],
    c: &[f64; STAPLE_DIM],
    mask: impl Fn(&[f64; STAPLE_DIM]) -> [f64; STAPLE_DIM],
) -> f64 {
    let a = mask(a);
    let b = mask(b);
    let c = mask(c);
    tensor.normalized_score(&a, &b, &c)
}

pub fn rotating_field(n: usize, omega: f64, mag: f64) -> Vec<[f64; 3]> {
    (0..n)
        .map(|i| {
            let t = i as f64 * omega;
            [mag * t.cos(), mag * t.sin(), 0.0]
        })
        .collect()
}

pub fn compression_step(n: usize, low: f64, high: f64) -> Vec<[f64; 3]> {
    (0..n)
        .map(|i| {
            let mag = if i < n / 2 { low } else { high };
            [mag, 0.0, 0.0]
        })
        .collect()
}

pub fn constant_field(n: usize) -> Vec<[f64; 3]> {
    vec![[1.0, 0.2, -0.1]; n]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::staple_associator::{joint_associator_norms, staple_embedding};
    use crate::staple_controls::{SparseCubicTensor, six_sample_baselines};
    use cd_kernel::mult_table::CdMultTable;

    #[test]
    fn quaternion_lag0_mask_kills_the_associator_on_rotating_b() {
        let table = CdMultTable::generate(STAPLE_DIM);
        let tensor = SparseCubicTensor::from_associator(&table);
        let rows = rotating_field(12, 0.3, 5.0);
        let staples = staple_embedding(&rows);
        let mut max_q = 0.0_f64;
        let mut max_full = 0.0_f64;
        for k in 0..staples.len().saturating_sub(2) {
            max_full = max_full.max(tensor.normalized_score(
                &staples[k],
                &staples[k + 1],
                &staples[k + 2],
            ));
            max_q = max_q.max(masked_normalized_score(
                &tensor,
                &staples[k],
                &staples[k + 1],
                &staples[k + 2],
                quaternion_lag0_only,
            ));
        }
        assert!(max_full > 1e-6, "rotating B must mix lags");
        assert!(
            max_q < 1e-9,
            "lag-0 quaternion subalgebra is associative, got {max_q}"
        );
    }

    #[test]
    fn rotating_field_has_rotation_and_flat_magnitude() {
        let rows = rotating_field(16, 0.25, 4.0);
        let base = six_sample_baselines(&rows);
        assert!(base.max_rotation.iter().copied().fold(0.0, f64::max) > 0.2);
        for k in 0..base.max_rotation.len() {
            assert!(mag_jump(&rows, k) < 1e-9);
        }
    }

    #[test]
    fn compression_step_has_mag_jump_and_no_rotation() {
        let rows = compression_step(16, 1.0, 3.0);
        let base = six_sample_baselines(&rows);
        let mut saw_jump = false;
        for k in 0..base.max_rotation.len() {
            if mag_jump(&rows, k) > 1.5 {
                saw_jump = true;
                assert!(base.max_rotation[k] < 1e-9);
            }
        }
        assert!(saw_jump);
    }

    #[test]
    fn constant_field_associator_vanishes() {
        let rows = constant_field(10);
        let staples = staple_embedding(&rows);
        let assoc = joint_associator_norms(&staples, true);
        for s in assoc {
            assert!(s < 1e-9, "constant B associator {s}");
        }
    }

    #[test]
    fn doubling_slot_is_lag2_bx() {
        assert_eq!(8 / 4, 2);
        assert_eq!(8 % 4, 0);
        let v = {
            let mut x = [0.0; STAPLE_DIM];
            x[8] = 7.0;
            x
        };
        let z = zero_doubling_slot(&v);
        assert_eq!(z[8], 0.0);
        assert_eq!(zero_magnitude_channels(&v)[8], 7.0);
    }
}
