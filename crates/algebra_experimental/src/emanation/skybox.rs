//! Skybox label-line extension (L9d) for the strutted emanation table.
//!
//! The ET proper is a K x K grid where K = G - 2 = 2^(N-1) - 2.
//! For the doubling recursion (N -> N+1), we need a power-of-2 edge:
//! promote the strut constant S and composite X = G + S to "label lines"
//! bordering the ET on all four sides.
//!
//! The skybox is a G x G grid (edge = 2^(N-1)) where:
//!   - Row/Col 0: label line (assessor (S, X))
//!   - Row/Col 1..K: the original ET positions
//!   - Row/Col K+1: mirror label line (assessor (S, X) again -- strut-opposite)
//!   - Main diagonal: empty (self-interaction)
//!   - Anti-diagonal (i + j == G-1): empty (strut-opposite blanks)
//!   - Four corners (0,0), (0,G-1), (G-1,0), (G-1,G-1): empty
//!     (diagonal + anti-diagonal both pass through corners)
//!
//! The label lines carry DMZ status from the S-assessor interacting with
//! each ET assessor via the same X-pattern test.

use super::cdp::cdp_signed_product;
use super::strutted_et::{StruttedEmanationTable, create_strutted_et};

/// A cell in the skybox extension of the emanation table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkyboxCell {
    /// Whether this cell is a DMZ cell.
    pub is_dmz: bool,
    /// If DMZ: the signed emanation value. 0 if not DMZ.
    pub emanation_value: i32,
    /// Whether this cell is on a label line (row 0, col 0, row G-1, or col G-1).
    pub is_label_line: bool,
    /// Whether this cell is structural empty (diagonal, anti-diagonal, or corner).
    pub is_structural_empty: bool,
}

/// The skybox: a G x G extension of the strutted ET with label lines.
#[derive(Debug, Clone)]
pub struct Skybox {
    /// CD level (dim = 2^n).
    pub n: usize,
    /// Strut constant.
    pub s: usize,
    /// Generator G = 2^(n-1).
    pub g: usize,
    /// Skybox edge length (= G).
    pub edge: usize,
    /// The underlying ET.
    pub et: StruttedEmanationTable,
    /// `G x G` grid of cells, indexed as `grid[row][col]`.
    pub grid: Vec<Vec<SkyboxCell>>,
    /// Number of DMZ cells in the skybox (including label-line DMZs).
    pub dmz_count: usize,
    /// Number of DMZ cells on label lines only.
    pub label_dmz_count: usize,
}

/// Create the skybox for a given (n, s).
///
/// The skybox extends the K x K strutted ET to a G x G grid by adding
/// label lines (the S-assessor) at the borders.
pub fn create_skybox(n: usize, s: usize) -> Skybox {
    let et = create_strutted_et(n, s);
    let g = et.tone_row.g;
    let x = et.tone_row.x;
    let edge = g; // G = 2^(n-1)

    let mut grid = vec![
        vec![
            SkyboxCell {
                is_dmz: false,
                emanation_value: 0,
                is_label_line: false,
                is_structural_empty: false,
            };
            edge
        ];
        edge
    ];

    let mut dmz_count = 0usize;
    let mut label_dmz_count = 0usize;

    for (row, grid_row) in grid.iter_mut().enumerate() {
        for (col, cell) in grid_row.iter_mut().enumerate() {
            // Structural empties: diagonal and anti-diagonal
            if row == col || row + col == edge - 1 {
                cell.is_structural_empty = true;
                continue;
            }

            // Label-line cells: row or col is 0 or edge-1
            let is_label = row == 0 || row == edge - 1 || col == 0 || col == edge - 1;
            cell.is_label_line = is_label;

            if is_label {
                // Compute DMZ status for label-line cell.
                // The label line assessor is (S, X).
                // For label-line rows (row=0 or row=edge-1): the row assessor is (S, X).
                // For label-line cols (col=0 or col=edge-1): the col assessor is (S, X).
                let (l_row, h_row) = if row == 0 || row == edge - 1 {
                    (s, x)
                } else {
                    // Map skybox row to ET position: skybox row i -> ET position i-1
                    let et_pos = row - 1;
                    (et.tone_row.lo[et_pos], et.tone_row.hi[et_pos])
                };
                let (l_col, h_col) = if col == 0 || col == edge - 1 {
                    (s, x)
                } else {
                    let et_pos = col - 1;
                    (et.tone_row.lo[et_pos], et.tone_row.hi[et_pos])
                };

                // X-pattern test (same as ET algorithm in compute_et_row)
                let (ul_idx, ul_sign) = cdp_signed_product(h_row, l_col);
                let (ur_idx, ur_sign) = cdp_signed_product(h_row, h_col);
                let (ll_idx, ll_sign) = cdp_signed_product(l_row, l_col);
                let (lr_idx, lr_sign) = cdp_signed_product(l_row, h_col);

                // Cross-magnitude check: |UL| == |LR| and |UR| == |LL|
                if ul_idx == lr_idx && ur_idx == ll_idx {
                    let edge1 = if ul_sign == lr_sign { 1i32 } else { -1 };
                    let edge2 = if ur_sign == ll_sign { 1i32 } else { -1 };
                    if edge1 == edge2 {
                        cell.is_dmz = true;
                        cell.emanation_value = edge1 * ll_idx as i32;
                        dmz_count += 1;
                        label_dmz_count += 1;
                    }
                }
            } else {
                // Interior cell: copy from ET
                let et_row = row - 1;
                let et_col = col - 1;
                if let Some(et_cell) = &et.cells[et_row][et_col]
                    && et_cell.is_dmz
                {
                    cell.is_dmz = true;
                    cell.emanation_value = et_cell.emanation_value;
                    dmz_count += 1;
                }
            }
        }
    }

    Skybox {
        n,
        s,
        g,
        edge,
        et,
        grid,
        dmz_count,
        label_dmz_count,
    }
}
