//! Strutted Emanation Table (L2 + L3 in de Marrais's emanation
//! framework): tone-row label generation and the DMZ X-pattern test.
//!
//! Final tier-5 validation probe -- algebra_experimental crate change.
//!
//! For a given (N, S) where N is the power-of-2 exponent and S is the
//! strut constant, the tone row generates the ET row/column labels:
//!
//!   G = 2^(N-1)   (generator)
//!   X = G + S     (composite: the XOR of G and S equals X since G is a power of 2)
//!   K = G - 2     (number of labels per row/col = number of LO indices minus S)
//!
//! Labels are mirror-paired: for each lo-index `try` (skipping S), its
//! strut-opposite `try XOR S` is placed at the mirror position. High
//! indices are `try XOR X`.
//!
//! Once the tone row is fixed, `create_strutted_et` runs de Marrais's
//! "Create Emanation Table" algorithm from Presto! Digitization I:
//! for each (row, col) pair, compute the 4 X-pattern products
//! UL, UR, LL, LR via the CDP signed-product engine. A DMZ (mutual
//! zero-divisor) cell exists when the cross-magnitude check holds
//! and edge-sign parity matches.

use super::cdp::cdp_signed_product;

// ===========================================================================
// Tone Row (L2)
// ===========================================================================

/// A tone row: the ET row/column labeling for a specific (N, S).
#[derive(Debug, Clone)]
pub struct ToneRow {
    /// The power-of-2 exponent (dim = 2^n).
    pub n: usize,
    /// The strut constant.
    pub s: usize,
    /// Generator index: 2^(n-1).
    pub g: usize,
    /// Composite index: G + S (= G XOR S since G is a power of 2 and S < G).
    pub x: usize,
    /// Number of label positions: G - 2 (= 2^(n-1) - 2).
    pub k: usize,
    /// Low-index tone row (ordered), length K.
    pub lo: Vec<usize>,
    /// High-index tone row (ordered), length K. `hi[i]` is the HI partner of `lo[i]`.
    pub hi: Vec<usize>,
}

/// Generate the tone row for a given (n, s) where dim = 2^n.
///
/// The tone row lists the K = 2^(n-1) - 2 LO-HI assessor pairs in the
/// mirror-paired ordering used by de Marrais's emanation tables.
///
/// This eliminates S from the LO indices and X from the HI indices,
/// placing strut-opposites at mirror positions (positions i and K+1-i).
pub fn generate_tone_row(n: usize, s: usize) -> ToneRow {
    assert!(n >= 4, "Need at least sedenions (n >= 4)");
    let g = 1usize << (n - 1);
    assert!(s >= 1 && s < g, "Strut constant must be in [1, G)");

    let x = g + s; // = g ^ s since g is a pure power of 2 and s < g
    let k = g - 2; // number of positions

    // Step 1: collect all LO indices from 1..G-1, excluding S
    let raw: Vec<usize> = (1..g).filter(|&i| i != s).collect();
    assert_eq!(raw.len(), k);

    // Step 2: mirror-pair them
    let mut lo_tone = vec![0usize; k];
    let mut hi_tone = vec![0usize; k];

    let mut lo_count = 0usize; // fills from front
    let mut hi_count = k.saturating_sub(1); // fills from back

    for &try_val in &raw {
        let partner = try_val ^ s; // strut-opposite
        if try_val < partner {
            lo_tone[lo_count] = try_val;
            hi_tone[lo_count] = try_val ^ x;

            lo_tone[hi_count] = partner;
            hi_tone[hi_count] = partner ^ x;

            // Check termination: when we've placed half the pairs
            if 2 * (lo_count + 1) == k {
                break;
            }
            lo_count += 1;
            hi_count -= 1;
        }
        // If try_val >= partner, skip (it will be placed as the mirror partner)
    }

    ToneRow {
        n,
        s,
        g,
        x,
        k,
        lo: lo_tone,
        hi: hi_tone,
    }
}

// ===========================================================================
// Strutted Emanation Table with DMZ Test (L3)
// ===========================================================================
//
// De Marrais's Create Emanation Table algorithm:
//
// For each row k and column q (both indexing into the tone row):
// 1. Skip diagonal (k == q) and strut-opposites (k + q == K + 1)
// 2. Get the 4 elements: LRow=lo[k], HRow=hi[k], LCol=lo[q], HCol=hi[q]
// 3. Compute the 4 products (the "X-pattern"):
//    UL = M(HRow, LCol)   -- upper-left
//    UR = M(HRow, HCol)   -- upper-right
//    LL = M(LRow, LCol)   -- lower-left
//    LR = M(LRow, HCol)   -- lower-right
// 4. Check: |UL| == |LR| and |UR| == |LL|  (cross-magnitude consistency)
// 5. Edge  = sgn(UL) == sgn(LR) ? +1 : -1
//    Edge2 = sgn(UR) == sgn(LL) ? +1 : -1
// 6. If Edge == Edge2: this is a ZD pair (DMZ cell).
//    Cell value = Edge * |LL|  (the low-index of the emanation with edge sign)

/// A cell in the strutted emanation table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StruttedEtCell {
    /// Row position in the tone-row ordering (0-based).
    pub row_pos: usize,
    /// Column position in the tone-row ordering (0-based).
    pub col_pos: usize,
    /// LO index of the row assessor.
    pub lo_row: usize,
    /// HI index of the row assessor.
    pub hi_row: usize,
    /// LO index of the column assessor.
    pub lo_col: usize,
    /// HI index of the column assessor.
    pub hi_col: usize,
    /// The 4-product X-pattern results: (UL, UR, LL, LR) as signed values.
    pub ul: i32,
    pub ur: i32,
    pub ll: i32,
    pub lr: i32,
    /// Whether this cell is a DMZ (mutual zero-divisor) cell.
    pub is_dmz: bool,
    /// If DMZ: the edge sign (+1 or -1). 0 if not DMZ.
    pub edge_sign: i32,
    /// If DMZ: the emanation low-index (unsigned). 0 if not DMZ.
    pub emanation_index: usize,
    /// If DMZ: the signed emanation value (edge_sign * emanation_index). 0 if not DMZ.
    pub emanation_value: i32,
}

/// The complete strutted emanation table for a specific (N, S).
#[derive(Debug, Clone)]
pub struct StruttedEmanationTable {
    /// The tone row this table is built from.
    pub tone_row: ToneRow,
    /// K x K grid of cells (some may be empty/non-DMZ).
    /// Indexed as `cells[row][col]` where row and col are tone-row positions.
    pub cells: Vec<Vec<Option<StruttedEtCell>>>,
    /// Number of DMZ (filled) cells.
    pub dmz_count: usize,
    /// Total possible cells (K*K minus diagonal and strut-opposite blanks).
    pub total_possible: usize,
}

/// Create the strutted emanation table for a given (n, s).
///
/// This is a faithful implementation of de Marrais's "Create Emanation Table"
/// algorithm from Presto! Digitization I (arXiv:math/0603281, Appendix).
///
/// The 4-product X-pattern test determines whether each assessor pair is a
/// mutual zero-divisor:
///   UL = M(HRow, LCol), UR = M(HRow, HCol)
///   LL = M(LRow, LCol), LR = M(LRow, HCol)
///   If |UL|==|LR| and |UR|==|LL| and sgn(UL)==sgn(LR) iff sgn(UR)==sgn(LL),
///   then the cell is a DMZ with value = edge_sign * |LL|.
pub fn create_strutted_et(n: usize, s: usize) -> StruttedEmanationTable {
    let tone_row = generate_tone_row(n, s);
    let k = tone_row.k;

    let mut dmz_count = 0usize;
    let mut total_possible = 0usize;

    let cells: Vec<Vec<Option<StruttedEtCell>>> = tone_row
        .lo
        .iter()
        .zip(&tone_row.hi)
        .enumerate()
        .map(|(row_pos, (&l_row, &h_row))| {
            compute_et_row(
                row_pos,
                l_row,
                h_row,
                &tone_row,
                k,
                &mut dmz_count,
                &mut total_possible,
            )
        })
        .collect();

    StruttedEmanationTable {
        tone_row,
        cells,
        dmz_count,
        total_possible,
    }
}

/// Compute one row of the strutted ET. Helper to satisfy clippy's needless_range_loop.
fn compute_et_row(
    row_pos: usize,
    l_row: usize,
    h_row: usize,
    tone_row: &ToneRow,
    k: usize,
    dmz_count: &mut usize,
    total_possible: &mut usize,
) -> Vec<Option<StruttedEtCell>> {
    tone_row
        .lo
        .iter()
        .zip(&tone_row.hi)
        .enumerate()
        .map(|(col_pos, (&l_col, &h_col))| {
            // Skip diagonal
            if col_pos == row_pos {
                return None;
            }
            // Skip strut-opposites: positions that sum to K-1 (0-indexed mirrors)
            if row_pos + col_pos == k - 1 {
                return None;
            }

            *total_possible += 1;

            // 4-product X-pattern
            let (ul_idx, ul_sign) = cdp_signed_product(h_row, l_col);
            let (ur_idx, ur_sign) = cdp_signed_product(h_row, h_col);
            let (ll_idx, ll_sign) = cdp_signed_product(l_row, l_col);
            let (lr_idx, lr_sign) = cdp_signed_product(l_row, h_col);

            let ul = ul_sign as i32 * ul_idx as i32;
            let ur = ur_sign as i32 * ur_idx as i32;
            let ll = ll_sign as i32 * ll_idx as i32;
            let lr = lr_sign as i32 * lr_idx as i32;

            // Cross-magnitude check
            if ul_idx != lr_idx || ur_idx != ll_idx {
                return Some(StruttedEtCell {
                    row_pos,
                    col_pos,
                    lo_row: l_row,
                    hi_row: h_row,
                    lo_col: l_col,
                    hi_col: h_col,
                    ul,
                    ur,
                    ll,
                    lr,
                    is_dmz: false,
                    edge_sign: 0,
                    emanation_index: 0,
                    emanation_value: 0,
                });
            }

            // Edge sign determination
            let edge = if ul_sign == lr_sign { 1i32 } else { -1i32 };
            let edge2 = if ur_sign == ll_sign { 1i32 } else { -1i32 };

            let is_dmz = edge == edge2;
            let (emanation_index, emanation_value) = if is_dmz {
                (ll_idx, edge * ll_idx as i32)
            } else {
                (0, 0)
            };

            if is_dmz {
                *dmz_count += 1;
            }

            Some(StruttedEtCell {
                row_pos,
                col_pos,
                lo_row: l_row,
                hi_row: h_row,
                lo_col: l_col,
                hi_col: h_col,
                ul,
                ur,
                ll,
                lr,
                is_dmz,
                edge_sign: if is_dmz { edge } else { 0 },
                emanation_index,
                emanation_value,
            })
        })
        .collect()
}
