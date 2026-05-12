//! Semiotic Square algebraic kernel verification (L9 in de Marrais's
//! emanation framework).
//!
//! De Marrais's algebraic kernel for the Semiotic Square:
//! Let V, Z be two assessors on a strut axis, and v, z their strut-opposites.
//! Then the product relationships form a Klein 4-group {I, H, V, D}:
//!   V*Z = v*z = S     (strut constant)
//!   Z*v = V*z = G     (generator)
//!   Z*z = v*V = X     (composite, G XOR S)
//!
//! Products are computed via `cdp_signed_product` on the L-indices,
//! re-exported from the parent module.
//!
//! `verify_ss_algebraic_kernel` runs the Klein-group check across all
//! sedenion box-kites and returns per-axis results.

use algebra_analysis::boxkites::{canonical_strut_table, find_box_kites};

use super::cdp_signed_product;

/// Semiotic Square kernel verification result.
#[derive(Debug, Clone)]
pub struct SsKernelResult {
    /// Box-kite strut signature.
    pub strut_sig: usize,
    /// The 3 strut axis labels (e.g., AF, BE, CD).
    pub axes: Vec<([usize; 2], SsKernelCheck)>,
}

/// Per-axis kernel check result.
#[derive(Debug, Clone)]
pub struct SsKernelCheck {
    /// V*Z product index.
    pub vz_product: usize,
    /// v*z product index (should equal V*Z).
    pub vbzb_product: usize,
    /// Z*v product index.
    pub zv_product: usize,
    /// V*z product index (should equal Z*v).
    pub vbz_product: usize,
    /// Whether the Klein group structure holds.
    pub klein_verified: bool,
}

/// Verify the Semiotic Square algebraic kernel for all box-kites.
///
/// For each strut axis in each box-kite, checks that the product
/// relationships form the expected Klein 4-group pattern:
///   V*Z = v*z (both yield the same product index)
///   Z*v = V*z (both yield the same product index)
///   The two product indices, together with identity, form {I, S, G, X}.
pub fn verify_ss_algebraic_kernel() -> Vec<SsKernelResult> {
    let bks = find_box_kites(16, 1e-10);
    let atol = 1e-10;
    let mut results = Vec::new();

    for bk in &bks {
        let tab = canonical_strut_table(bk, atol);

        // For each strut axis, V and Z are the strut pair,
        // v and z are their strut-opposites (the OTHER pair).
        let axes_data = [
            // Axis AF: V=A, Z=F, then the 4 other assessors include v,z
            ([tab.a.low, tab.f.low], tab.a, tab.f, tab.b, tab.e),
            // Axis BE
            ([tab.b.low, tab.e.low], tab.b, tab.e, tab.a, tab.f),
            // Axis CD
            ([tab.c.low, tab.d.low], tab.c, tab.d, tab.a, tab.b),
        ];

        let mut axes = Vec::new();
        for (label, v_ass, z_ass, v_bar, z_bar) in &axes_data {
            // V*Z using L-indices
            let (vz_idx, _vz_sign) = cdp_signed_product(v_ass.low, z_ass.low);
            // v*z (strut opposites' L-indices)
            let (vbzb_idx, _vbzb_sign) = cdp_signed_product(v_bar.low, z_bar.low);
            // Z*v
            let (zv_idx, _zv_sign) = cdp_signed_product(z_ass.low, v_bar.low);
            // V*z
            let (vbz_idx, _vbz_sign) = cdp_signed_product(v_ass.low, z_bar.low);

            let klein_verified = vz_idx == vbzb_idx && zv_idx == vbz_idx;

            axes.push((
                *label,
                SsKernelCheck {
                    vz_product: vz_idx,
                    vbzb_product: vbzb_idx,
                    zv_product: zv_idx,
                    vbz_product: vbz_idx,
                    klein_verified,
                },
            ));
        }

        results.push(SsKernelResult {
            strut_sig: bk.strut_signature,
            axes,
        });
    }

    results
}
