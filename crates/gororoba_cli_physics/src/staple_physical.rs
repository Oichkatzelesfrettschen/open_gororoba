//! Physical decomposition of the six-sample staple window.
//!
//! The CD associator on THEMIS-A is a cubic mismatch on six FGM samples
//! of B in R^3. This module splits that window into 3D B-space channels
//! (hodogram path length, gram volume) and derived frames (MVA LM). The
//! measurement is not a light cone and the LM plane is not the data.
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

/// Keep two of the four lags (0..3); other staple slots go to 0.
pub fn keep_two_lags(v: &[f64; STAPLE_DIM], lag_a: usize, lag_b: usize) -> [f64; STAPLE_DIM] {
    mask_staple(v, |i| {
        let lag = i / 4;
        lag == lag_a || lag == lag_b
    })
}

/// Sonnerup-Cahill minimum-variance analysis on one six-sample window.
///
/// Covariance M_ij = <B_i B_j> - <B_i><B_j>. Eigenvalues λ_L >= λ_M >= λ_N;
/// N is the estimated boundary normal. For a tangential discontinuity λ_N is
/// small and the field rotates in the L-M plane.
#[derive(Debug, Clone, Copy)]
pub struct MvaWindow {
    pub lambda_l: f64,
    pub lambda_m: f64,
    pub lambda_n: f64,
    pub ratio_mid_min: f64,
    pub delta_bl: f64,
    pub delta_bm: f64,
    pub abs_bn_mean: f64,
    pub lm_rotation: f64,
}

pub fn mva_six_sample(rows: &[[f64; 3]], k: usize) -> Option<MvaWindow> {
    if k + 5 >= rows.len() {
        return None;
    }
    let window = &rows[k..k + 6];
    let mut mean = [0.0_f64; 3];
    for row in window {
        mean[0] += row[0];
        mean[1] += row[1];
        mean[2] += row[2];
    }
    mean[0] /= 6.0;
    mean[1] /= 6.0;
    mean[2] /= 6.0;
    let mut cov = [[0.0_f64; 3]; 3];
    for row in window {
        let d = [row[0] - mean[0], row[1] - mean[1], row[2] - mean[2]];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    for row in &mut cov {
        for x in row {
            *x /= 6.0;
        }
    }
    let (evals, evecs) = jacobi_symmetric3(cov);
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| {
        evals[j]
            .partial_cmp(&evals[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let l = order[0];
    let m = order[1];
    let n = order[2];
    let lambda_l = evals[l].max(0.0);
    let lambda_m = evals[m].max(0.0);
    let lambda_n = evals[n].max(0.0);
    let axis_l = evecs[l];
    let axis_m = evecs[m];
    let axis_n = evecs[n];
    let mut bl_lo = f64::INFINITY;
    let mut bl_hi = f64::NEG_INFINITY;
    let mut bm_lo = f64::INFINITY;
    let mut bm_hi = f64::NEG_INFINITY;
    let mut abs_bn = 0.0;
    let mut lm: [(f64, f64); 6] = [(0.0, 0.0); 6];
    for (t, row) in window.iter().enumerate() {
        let bl = row[0] * axis_l[0] + row[1] * axis_l[1] + row[2] * axis_l[2];
        let bm = row[0] * axis_m[0] + row[1] * axis_m[1] + row[2] * axis_m[2];
        let bn = row[0] * axis_n[0] + row[1] * axis_n[1] + row[2] * axis_n[2];
        bl_lo = bl_lo.min(bl);
        bl_hi = bl_hi.max(bl);
        bm_lo = bm_lo.min(bm);
        bm_hi = bm_hi.max(bm);
        abs_bn += bn.abs();
        lm[t] = (bl, bm);
    }
    let mut lm_rotation = 0.0_f64;
    for t in 1..6 {
        let a = lm[t - 1];
        let b = lm[t];
        let na = (a.0 * a.0 + a.1 * a.1).sqrt();
        let nb = (b.0 * b.0 + b.1 * b.1).sqrt();
        if na > 0.0 && nb > 0.0 {
            let cosv = ((a.0 * b.0 + a.1 * b.1) / (na * nb)).clamp(-1.0, 1.0);
            lm_rotation = lm_rotation.max(cosv.acos());
        }
    }
    Some(MvaWindow {
        lambda_l,
        lambda_m,
        lambda_n,
        ratio_mid_min: lambda_m / (lambda_n + 1e-30),
        delta_bl: (bl_hi - bl_lo).max(0.0),
        delta_bm: (bm_hi - bm_lo).max(0.0),
        abs_bn_mean: abs_bn / 6.0,
        lm_rotation,
    })
}

/// Jacobi eigen-decomposition of a 3x3 symmetric matrix.
/// Returns eigenvalues and corresponding eigenvectors as rows of `evecs`.
#[allow(clippy::needless_range_loop)] // Jacobi plane (p,q) skips those two axes by index
fn jacobi_symmetric3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..32 {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut best = a[0][1].abs();
        if a[0][2].abs() > best {
            best = a[0][2].abs();
            p = 0;
            q = 2;
        }
        if a[1][2].abs() > best {
            best = a[1][2].abs();
            p = 1;
            q = 2;
        }
        if best < 1e-18 {
            break;
        }
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau == 0.0 {
            1.0
        } else {
            tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for r in 0..3 {
            if r == p || r == q {
                continue;
            }
            let arp = a[r][p];
            let arq = a[r][q];
            a[r][p] = c * arp - s * arq;
            a[p][r] = a[r][p];
            a[r][q] = s * arp + c * arq;
            a[q][r] = a[r][q];
        }
        for r in 0..3 {
            let vrp = v[r][p];
            let vrq = v[r][q];
            v[r][p] = c * vrp - s * vrq;
            v[r][q] = s * vrp + c * vrq;
        }
    }
    let evals = [a[0][0], a[1][1], a[2][2]];
    let evecs = [
        [v[0][0], v[1][0], v[2][0]],
        [v[0][1], v[1][1], v[2][1]],
        [v[0][2], v[1][2], v[2][2]],
    ];
    (evals, evecs)
}

/// 1-based average ranks of `scores` (ties keep order of appearance).
pub fn average_ranks(scores: &[f64]) -> Vec<f64> {
    let n = scores.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        scores[i]
            .partial_cmp(&scores[j])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(i.cmp(&j))
    });
    let mut ranks = vec![0.0; n];
    for (r, &i) in order.iter().enumerate() {
        ranks[i] = (r + 1) as f64;
    }
    ranks
}

/// Hodogram arc length in B-space: sum of |ΔB| over the six samples.
/// This is a 3D path length, not a plane angle and not a cone.
pub fn b_path_length(rows: &[[f64; 3]], k: usize) -> f64 {
    if k + 5 >= rows.len() {
        return 0.0;
    }
    let mut len = 0.0_f64;
    for t in k..k + 5 {
        let d0 = rows[t + 1][0] - rows[t][0];
        let d1 = rows[t + 1][1] - rows[t][1];
        let d2 = rows[t + 1][2] - rows[t][2];
        len += (d0 * d0 + d1 * d1 + d2 * d2).sqrt();
    }
    len
}

/// Helical B: planar rotation plus a z ramp, so three consecutive dB
/// span a parallelepiped of nonzero volume.
pub fn helical_field(n: usize, omega: f64, mag: f64, z_step: f64) -> Vec<[f64; 3]> {
    (0..n)
        .map(|i| {
            let t = i as f64 * omega;
            [mag * t.cos(), mag * t.sin(), i as f64 * z_step]
        })
        .collect()
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

    #[test]
    fn mva_on_xy_rotation_puts_normal_along_z() {
        let rows = rotating_field(12, 0.4, 5.0);
        let mva = mva_six_sample(&rows, 0).expect("window");
        assert!(mva.lambda_n < 1e-6, "lambda_n {}", mva.lambda_n);
        assert!(mva.lm_rotation > 0.3, "lm_rotation {}", mva.lm_rotation);
        assert!(mva.ratio_mid_min > 10.0, "ratio {}", mva.ratio_mid_min);
    }

    #[test]
    fn mva_on_x_compression_puts_l_along_x_and_kills_lm_rotation() {
        let rows = compression_step(12, 1.0, 4.0);
        let mva = mva_six_sample(&rows, 2).expect("window");
        assert!(mva.delta_bl > 2.0, "delta_bl {}", mva.delta_bl);
        assert!(mva.lm_rotation < 1e-6, "lm_rotation {}", mva.lm_rotation);
    }

    #[test]
    fn keep_two_lags_clears_the_other_eight_slots() {
        let v = [1.0; STAPLE_DIM];
        let k = keep_two_lags(&v, 1, 2);
        for (i, &slot) in k.iter().enumerate() {
            let lag = i / 4;
            if lag == 1 || lag == 2 {
                assert_eq!(slot, 1.0);
            } else {
                assert_eq!(slot, 0.0);
            }
        }
    }

    #[test]
    fn planar_rotation_has_zero_gram_volume_helix_does_not() {
        let planar = rotating_field(12, 0.4, 5.0);
        let helix = helical_field(12, 0.4, 5.0, 0.8);
        let p = six_sample_baselines(&planar);
        let h = six_sample_baselines(&helix);
        let pmax = p.max_gram_volume.iter().copied().fold(0.0, f64::max);
        let hmax = h.max_gram_volume.iter().copied().fold(0.0, f64::max);
        assert!(pmax < 1e-6, "planar hodogram gram volume {pmax}");
        assert!(hmax > 1e-3, "helix gram volume {hmax}");
        assert!(b_path_length(&helix, 0) > b_path_length(&planar, 0));
    }
}
