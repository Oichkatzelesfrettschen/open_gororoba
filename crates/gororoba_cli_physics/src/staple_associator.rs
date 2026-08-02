//! Raw 4-lag sedenion "staple" embedding and joint associator norms.
//!
//! The staple embedding packs four consecutive field samples, each as
//! (Bx, By, Bz, |B|), into one 16-component sedenion: no noise-floor
//! denominator, no unit-sphere projection.  This is the minimal embedding
//! that exposes non-associativity of the raw signal; the parameterized
//! pipeline in `spectral_core::cd_embedding` layers cadence and
//! normalization choices on top and is the right tool for detector work,
//! while this module is the right tool for score-distribution exports and
//! cross-mission self-similarity statistics where any normalization stage
//! would contaminate the measured moments.
//!
//! The joint associator at position k is |(V_k V_{k+1}) V_{k+2} -
//! V_k (V_{k+1} V_{k+2})| over consecutive staples, computed via the
//! SHA-256-verified `cd_kernel::mult_table::CdMultTable` in O(dim^2) per
//! product.

use cd_kernel::mult_table::CdMultTable;

/// Sedenion dimension used by the staple embedding: 4 samples x 4 channels.
pub const STAPLE_DIM: usize = 16;

/// Number of consecutive samples packed into one staple vector.
pub const STAPLE_LAGS: usize = 4;

/// Build the raw staple embedding from a series of (Bx, By, Bz) rows.
///
/// Row i of the output concatenates samples i..i+3, each extended with its
/// field magnitude, giving `rows.len() - 3` sedenions.  Returns an empty
/// vector when fewer than `STAPLE_LAGS` samples are available.
pub fn staple_embedding(rows: &[[f64; 3]]) -> Vec<[f64; STAPLE_DIM]> {
    if rows.len() < STAPLE_LAGS {
        return Vec::new();
    }
    let feat: Vec<[f64; 4]> = rows
        .iter()
        .map(|b| {
            let mag = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
            [b[0], b[1], b[2], mag]
        })
        .collect();
    (STAPLE_LAGS - 1..rows.len())
        .map(|i| {
            let mut v = [0.0_f64; STAPLE_DIM];
            for lag in 0..STAPLE_LAGS {
                v[4 * lag..4 * lag + 4].copy_from_slice(&feat[i - (STAPLE_LAGS - 1) + lag]);
            }
            v
        })
        .collect()
}

fn table_multiply(
    table: &CdMultTable,
    a: &[f64; STAPLE_DIM],
    b: &[f64; STAPLE_DIM],
) -> [f64; STAPLE_DIM] {
    let mut out = [0.0_f64; STAPLE_DIM];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0.0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            let (sign, k) = table.multiply_basis(i, j);
            out[k] += f64::from(sign) * ai * bj;
        }
    }
    out
}

fn norm(v: &[f64; STAPLE_DIM]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Joint associator norms over consecutive staple triples.
///
/// Entry k is |(V_k V_{k+1}) V_{k+2} - V_k (V_{k+1} V_{k+2})|; when
/// `normalized` is set, each entry is divided by |V_k| |V_{k+1}| |V_{k+2}|
/// (plus 1e-30 to guard empty windows).
pub fn joint_associator_norms(staples: &[[f64; STAPLE_DIM]], normalized: bool) -> Vec<f64> {
    if staples.len() < 3 {
        return Vec::new();
    }
    let table = CdMultTable::generate(STAPLE_DIM);
    (0..staples.len() - 2)
        .map(|k| {
            let (a, b, c) = (&staples[k], &staples[k + 1], &staples[k + 2]);
            let ab = table_multiply(&table, a, b);
            let bc = table_multiply(&table, b, c);
            let left = table_multiply(&table, &ab, c);
            let right = table_multiply(&table, a, &bc);
            let mut diff = [0.0_f64; STAPLE_DIM];
            for i in 0..STAPLE_DIM {
                diff[i] = left[i] - right[i];
            }
            let raw = norm(&diff);
            if normalized {
                raw / (norm(a) * norm(b) * norm(c) + 1e-30)
            } else {
                raw
            }
        })
        .collect()
}

/// Sample moments of log(x / median(x)) matching scipy's biased defaults:
/// skewness is m3 / m2^1.5 and kurtosis is Fisher excess m4 / m2^2 - 3,
/// both with population (1/n) moment normalization.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LogMomentStats {
    pub median: f64,
    pub coefficient_of_variation: f64,
    pub log_std: f64,
    pub log_skew: f64,
    pub log_kurtosis_excess: f64,
    pub n: usize,
}

/// Compute `LogMomentStats` over strictly positive, finite associator norms.
/// Returns `None` when fewer than 2 usable values remain.
pub fn log_moment_stats(values: &[f64]) -> Option<LogMomentStats> {
    let mut clean: Vec<f64> = values
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x > 0.0)
        .collect();
    if clean.len() < 2 {
        return None;
    }
    let n = clean.len();
    let nf = n as f64;

    let mean = clean.iter().sum::<f64>() / nf;
    let std = (clean.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / nf).sqrt();

    clean.sort_by(|a, b| a.partial_cmp(b).expect("finite by filter"));
    let median = if n % 2 == 1 {
        clean[n / 2]
    } else {
        0.5 * (clean[n / 2 - 1] + clean[n / 2])
    };

    let logs: Vec<f64> = clean.iter().map(|x| (x / median).ln()).collect();
    let lmean = logs.iter().sum::<f64>() / nf;
    let m2 = logs.iter().map(|x| (x - lmean).powi(2)).sum::<f64>() / nf;
    let m3 = logs.iter().map(|x| (x - lmean).powi(3)).sum::<f64>() / nf;
    let m4 = logs.iter().map(|x| (x - lmean).powi(4)).sum::<f64>() / nf;

    Some(LogMomentStats {
        median,
        coefficient_of_variation: std / mean,
        log_std: m2.sqrt(),
        log_skew: if m2 > 0.0 { m3 / m2.powf(1.5) } else { 0.0 },
        log_kurtosis_excess: if m2 > 0.0 { m4 / (m2 * m2) - 3.0 } else { 0.0 },
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cd_kernel::cd_multiply;

    #[test]
    fn table_multiply_matches_recursive_cd_multiply() {
        let table = CdMultTable::generate(STAPLE_DIM);
        let a: [f64; STAPLE_DIM] = core::array::from_fn(|i| (i as f64) * 0.25 - 1.0);
        let b: [f64; STAPLE_DIM] = core::array::from_fn(|i| 2.0 - (i as f64) * 0.5);
        let via_table = table_multiply(&table, &a, &b);
        let recursive = cd_multiply(&a, &b);
        for i in 0..STAPLE_DIM {
            assert!(
                (via_table[i] - recursive[i]).abs() < 1e-9,
                "component {i}: {} vs {}",
                via_table[i],
                recursive[i]
            );
        }
    }

    #[test]
    fn staple_embedding_packs_four_lags_with_magnitude() {
        let rows = [
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ];
        let staples = staple_embedding(&rows);
        assert_eq!(staples.len(), 1);
        let v = &staples[0];
        assert_eq!(&v[0..4], &[3.0, 4.0, 0.0, 5.0]);
        assert_eq!(&v[12..16], &[0.0, 2.0, 0.0, 2.0]);
    }

    #[test]
    fn associator_vanishes_on_quaternionic_staples() {
        // Staples confined to components {0..3} multiply inside the
        // quaternion subalgebra, which is associative, so the joint
        // associator is exactly zero.
        let staples: Vec<[f64; STAPLE_DIM]> = (0..5)
            .map(|k| {
                let mut v = [0.0_f64; STAPLE_DIM];
                v[0] = 1.0 + k as f64;
                v[1] = 0.5 * k as f64;
                v[2] = -1.0;
                v[3] = 0.25;
                v
            })
            .collect();
        for a in joint_associator_norms(&staples, false) {
            assert!(
                a.abs() < 1e-12,
                "quaternionic associator must vanish, got {a}"
            );
        }
    }

    #[test]
    fn log_moment_stats_of_lognormal_like_sample() {
        // Symmetric log values around the median give near-zero skew.
        let values = [1.0, 2.0, 4.0, 8.0, 16.0];
        let s = log_moment_stats(&values).expect("enough values");
        assert_eq!(s.n, 5);
        assert!((s.median - 4.0).abs() < 1e-12);
        assert!(s.log_skew.abs() < 1e-12);
    }
}
