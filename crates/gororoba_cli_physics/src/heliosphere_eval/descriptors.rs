//! Descriptor-channel helpers for heliosphere invariant samples.
//!
//! Builds the 8-channel descriptor vector that augments the raw
//! invariant samples: 4 channels from algebraic invariants
//! (`descriptor_channels_from_arrays`) and 4 channels from Takens
//! delay embeddings (`takens_descriptors`).
//!
//! The `HasBField` trait abstracts over `HeliosphereInvariantSample`
//! (used at build time) and `&LabeledInvariantSample` (used in
//! evaluation paths) so both groups can feed the same Takens routine.
//!
//! All items are `pub(super)`. The `to_cd16` helper pads a
//! HELIOSPHERE_INVARIANT_DIM-sized vector up to a 16-component
//! sedenion-shaped array for `cd_kernel` consumption.

use data_core::{HELIOSPHERE_INVARIANT_DIM, HeliosphereInvariantSample};

use super::{DESCRIPTOR_DIM, public_types::LabeledInvariantSample, stats::l2_norm_sq};

pub(super) trait HasBField {
    fn b_field(&self) -> [f64; 4];
}

impl HasBField for HeliosphereInvariantSample {
    fn b_field(&self) -> [f64; 4] {
        self.b_field
    }
}

impl HasBField for &LabeledInvariantSample {
    fn b_field(&self) -> [f64; 4] {
        self.b_field
    }
}

pub(super) fn descriptor_channels(
    group: &[HeliosphereInvariantSample],
    idx: usize,
) -> [f64; DESCRIPTOR_DIM] {
    let vectors = group
        .iter()
        .map(|sample| sample.weighted_channels)
        .collect::<Vec<_>>();
    let mut out = [0.0; DESCRIPTOR_DIM];
    let base = descriptor_channels_from_arrays(&vectors, idx);
    out[..4].copy_from_slice(&base);

    let takens = takens_descriptors(group, idx);
    out[4..8].copy_from_slice(&takens);
    out
}

pub(super) fn takens_descriptors<T: HasBField>(group: &[T], idx: usize) -> [f64; 4] {
    let get_v16 = |target_idx: usize| -> Option<[f64; 16]> {
        if target_idx < 3 {
            return None;
        }
        let mut v16 = [0.0; 16];
        for i in 0..4 {
            let s = &group[target_idx - 3 + i];
            v16[i * 4..i * 4 + 4].copy_from_slice(&s.b_field());
        }
        Some(v16)
    };

    let v_curr = get_v16(idx);
    let v_prev = idx.checked_sub(1).and_then(get_v16);
    let v_prev2 = idx.checked_sub(2).and_then(get_v16);

    match (v_prev2, v_prev, v_curr) {
        (Some(a), Some(b), Some(c)) => {
            let sedenion_assoc = cd_kernel::cd_associator_norm(&a, &b, &c);

            let mut a_oct = [0.0; 16];
            let mut b_oct = [0.0; 16];
            let mut c_oct = [0.0; 16];
            a_oct[..8].copy_from_slice(&a[..8]);
            b_oct[..8].copy_from_slice(&b[..8]);
            c_oct[..8].copy_from_slice(&c[..8]);
            let octonion_assoc = cd_kernel::cd_associator_norm(&a_oct, &b_oct, &c_oct);

            let mut a_rand = a;
            a_rand.reverse();
            let mut b_rand = b;
            b_rand.reverse();
            let mut c_rand = c;
            c_rand.reverse();
            let random_assoc = cd_kernel::cd_associator_norm(&a_rand, &b_rand, &c_rand);

            let euclidean = (l2_norm_sq(&a) + l2_norm_sq(&b) + l2_norm_sq(&c)).sqrt();

            [sedenion_assoc, octonion_assoc, random_assoc, euclidean]
        }
        _ => [0.0; 4],
    }
}

pub(super) fn descriptor_channels_from_arrays(
    vectors: &[[f64; HELIOSPHERE_INVARIANT_DIM]],
    idx: usize,
) -> [f64; 4] {
    let current = &vectors[idx];
    let prev = idx.checked_sub(1).map(|index| &vectors[index]);
    let prev2 = idx.checked_sub(2).map(|index| &vectors[index]);
    let norm_sq = l2_norm_sq(current);
    let delta_norm = prev
        .map(|value| (norm_sq - l2_norm_sq(value)).abs())
        .unwrap_or(0.0);
    let associator = match (prev2, prev) {
        (Some(a), Some(b)) => {
            let a_cd = to_cd16(a);
            let b_cd = to_cd16(b);
            let c_cd = to_cd16(current);
            cd_kernel::cd_associator_norm(&a_cd, &b_cd, &c_cd)
        }
        _ => 0.0,
    };
    let mean_abs = current.iter().map(|value| value.abs()).sum::<f64>() / current.len() as f64;
    [norm_sq, delta_norm, associator, mean_abs]
}

pub(super) fn to_cd16(values: &[f64; HELIOSPHERE_INVARIANT_DIM]) -> [f64; 16] {
    let mut out = [0.0_f64; 16];
    out[..HELIOSPHERE_INVARIANT_DIM].copy_from_slice(values);
    out
}
