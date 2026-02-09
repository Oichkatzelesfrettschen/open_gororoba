//! Reggiani's 84 standard zero-divisors in sedenions.
//!
//! The 42 primitive assessors each generate two unit zero-divisors
//! (e_low + e_high) and (e_low - e_high), giving 84 "standard" zero-divisors.
//! Each standard ZD has a 4-dimensional annihilator subspace, and exactly
//! 4 standard ZD partners (other standard ZDs that annihilate it).
//!
//! # Literature
//! - Reggiani (2024): Geometry of sedenion zero divisors, Table 1

use crate::analysis::annihilator::{
    annihilator_info, is_reggiani_zd, left_multiplication_matrix, nullspace_basis,
    right_multiplication_matrix,
};
use crate::analysis::boxkites::{diagonal_zero_products_exact, primitive_assessors};
use crate::construction::cayley_dickson::cd_multiply;

/// A standard zero-divisor: a diagonal of a primitive assessor.
#[derive(Debug, Clone)]
pub struct StandardZeroDivisor {
    pub assessor_low: usize,
    pub assessor_high: usize,
    pub diagonal_sign: i32,
    pub vector: Vec<f64>,
}

impl StandardZeroDivisor {
    fn key(&self) -> (usize, usize, i32) {
        (self.assessor_low, self.assessor_high, self.diagonal_sign)
    }
}

/// Generate all 84 standard zero-divisors (42 assessors x 2 signs).
pub fn standard_zero_divisors() -> Vec<StandardZeroDivisor> {
    let assessors = primitive_assessors();
    let mut out = Vec::with_capacity(84);
    for a in &assessors {
        for sign in [1i32, -1] {
            let mut v = vec![0.0; 16];
            v[a.low] = 1.0;
            v[a.high] = sign as f64;
            out.push(StandardZeroDivisor {
                assessor_low: a.low,
                assessor_high: a.high,
                diagonal_sign: sign,
                vector: v,
            });
        }
    }
    out
}

/// Find the 4 standard zero-divisor partners of `zd` -- other standard ZDs
/// whose product with `zd` is zero.
///
/// Uses integer-exact diagonal zero-product detection from boxkites.
pub fn standard_zero_divisor_partners(zd: &StandardZeroDivisor) -> Vec<StandardZeroDivisor> {
    let a_pair = (zd.assessor_low, zd.assessor_high);
    let s = zd.diagonal_sign as i8;

    let all = standard_zero_divisors();
    let mut partners = Vec::new();

    for cand in &all {
        if cand.assessor_low == zd.assessor_low
            && cand.assessor_high == zd.assessor_high
            && cand.diagonal_sign == zd.diagonal_sign
        {
            continue; // skip self
        }
        let b_pair = (cand.assessor_low, cand.assessor_high);
        let t = cand.diagonal_sign as i8;

        let sols = diagonal_zero_products_exact(16, a_pair, b_pair);
        if sols.contains(&(s, t)) {
            partners.push(StandardZeroDivisor {
                assessor_low: cand.assessor_low,
                assessor_high: cand.assessor_high,
                diagonal_sign: cand.diagonal_sign,
                vector: cand.vector.clone(),
            });
        }
    }

    assert_eq!(
        partners.len(),
        4,
        "Expected 4 standard partners for ({}, {}, {}), got {}",
        zd.assessor_low,
        zd.assessor_high,
        zd.diagonal_sign,
        partners.len()
    );

    partners.sort_by_key(|p| p.key());
    partners
}

/// Verify that a standard ZD satisfies all Reggiani consistency checks:
/// - Squared norm is 2
/// - Has nontrivial left and right annihilator
/// - Nullspace basis vectors actually annihilate
/// - The 4 standard partners span the annihilator subspace
pub fn assert_standard_zero_divisor_annihilators(
    zd: &StandardZeroDivisor,
) -> Result<(), String> {
    let v = &zd.vector;
    let norm_sq: f64 = v.iter().map(|x| x * x).sum();
    if (norm_sq - 2.0).abs() > 1e-12 {
        return Err(format!(
            "Squared norm = {norm_sq}, expected 2.0"
        ));
    }

    if !is_reggiani_zd(v, 1e-12) {
        return Err("Not in Reggiani ZD(S)".to_string());
    }

    let info = annihilator_info(v, 16, 1e-12);
    if info.left_nullity == 0 || info.right_nullity == 0 {
        return Err(format!(
            "Expected nontrivial annihilators, got {:?}",
            info
        ));
    }

    // Verify nullspace basis vectors actually annihilate
    let la = left_multiplication_matrix(v, 16);
    let ra = right_multiplication_matrix(v, 16);
    let left_basis = nullspace_basis(&la, 1e-12);
    let right_basis = nullspace_basis(&ra, 1e-12);

    for col in 0..left_basis.ncols() {
        let b: Vec<f64> = left_basis.column(col).iter().copied().collect();
        let prod = cd_multiply(v, &b);
        if prod.iter().any(|&x| x.abs() > 1e-10) {
            return Err(format!(
                "Left annihilator basis vector {col} does not annihilate"
            ));
        }
    }

    for col in 0..right_basis.ncols() {
        let b: Vec<f64> = right_basis.column(col).iter().copied().collect();
        let prod = cd_multiply(&b, v);
        if prod.iter().any(|&x| x.abs() > 1e-10) {
            return Err(format!(
                "Right annihilator basis vector {col} does not annihilate"
            ));
        }
    }

    // Verify 4 standard partners span the left annihilator subspace
    let partners = standard_zero_divisor_partners(zd);
    let partner_cols: Vec<Vec<f64>> = partners.iter().map(|p| p.vector.clone()).collect();

    // Check: left_basis^T @ partner_matrix should have rank 4
    let k = left_basis.ncols();
    let n = 4;
    let mut coords = nalgebra::DMatrix::zeros(k, n);
    for (j, pvec) in partner_cols.iter().enumerate() {
        for i in 0..k {
            let mut dot = 0.0;
            for r in 0..16 {
                dot += left_basis[(r, i)] * pvec[r];
            }
            coords[(i, j)] = dot;
        }
    }
    let svd = nalgebra::SVD::new(coords, false, false);
    let rank = svd.singular_values.iter().filter(|&&s| s > 1e-8).count();
    if rank != 4 {
        return Err(format!(
            "Expected rank 4 for partner-annihilator projection, got {rank}"
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_zero_divisors_count_is_84() {
        let zds = standard_zero_divisors();
        assert_eq!(zds.len(), 84);

        for zd in &zds {
            let norm_sq: f64 = zd.vector.iter().map(|x| x * x).sum();
            assert!(
                (norm_sq - 2.0).abs() < 1e-12,
                "ZD ({},{},{}) has norm_sq={norm_sq}",
                zd.assessor_low,
                zd.assessor_high,
                zd.diagonal_sign,
            );
            assert!(
                is_reggiani_zd(&zd.vector, 1e-12),
                "ZD ({},{},{}) not in Reggiani ZD(S)",
                zd.assessor_low,
                zd.assessor_high,
                zd.diagonal_sign,
            );
        }
    }

    #[test]
    fn test_all_standard_zds_have_nullity_4_4() {
        for zd in &standard_zero_divisors() {
            let info = annihilator_info(&zd.vector, 16, 1e-12);
            assert_eq!(
                (info.left_nullity, info.right_nullity),
                (4, 4),
                "ZD ({},{},{}) has nullity ({},{})",
                zd.assessor_low,
                zd.assessor_high,
                zd.diagonal_sign,
                info.left_nullity,
                info.right_nullity,
            );
        }
    }

    #[test]
    fn test_standard_zd_annihilator_consistency() {
        // Full Reggiani consistency check on all 84 ZDs
        for zd in &standard_zero_divisors() {
            assert_standard_zero_divisor_annihilators(zd).unwrap_or_else(|e| {
                panic!(
                    "ZD ({},{},{}) failed: {e}",
                    zd.assessor_low, zd.assessor_high, zd.diagonal_sign
                );
            });
        }
    }
}
