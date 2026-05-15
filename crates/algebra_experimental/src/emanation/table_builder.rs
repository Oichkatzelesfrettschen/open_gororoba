//! Emanation table construction, sand-mandala sparsity analysis,
//! carry-bit overflow detection, period-doubling scan, and block
//! similarity.
//!
//! - `emanation_table(dim)`: builds the full (dim-1) x (dim-1) table,
//!   marking each cell with its product index/sign and whether the
//!   pair is a cross-assessor zero-divisor.
//! - `sand_mandala_pattern(et)`: aggregates per-row and per-column ZD
//!   fill ratios over the cross-assessor cells.
//! - `carry_bit_overflow_cells(dim)`: identifies pairs lost/gained
//!   between dim/2 and dim ZD sets (the structural restructuring
//!   evident at dim=32 pathions).
//! - `et_period_doubling(&[dims])`: bulk EtScaling diagnostic across
//!   multiple dimensions.
//! - `et_block_similarity(parent, child)`: fraction of cross-assessor
//!   cells where parent's is_zero_divisor flag matches child's at
//!   the same (i, j) indices.

use std::collections::{HashMap, HashSet};

use algebra_analysis::boxkites::{
    CrossPair, cross_assessors, motif_components_for_cross_assessors,
};
use cd_kernel::cayley_dickson::cd_basis_mul_sign;

use super::{EmanationTable, EtCell, EtScaling, MandalaSummary};

/// Generate the emanation table for a Cayley-Dickson algebra.
///
/// The table covers basis indices 1..dim-1. Each cell (i,j) stores the
/// product sign and whether that index pair is a ZD cross-assessor.
///
/// For dim=16: 14x14 table with 42 ZD entries (one per primitive assessor).
/// For dim=32: 30x30 table with sand-mandala sparsity pattern.
pub fn emanation_table(dim: usize) -> EmanationTable {
    assert!(dim >= 16 && dim.is_power_of_two(), "dim must be 2^n >= 16");

    let half = dim / 2;
    let size = dim - 1; // indices 1..dim-1 (all imaginary basis elements)

    // Build the set of cross-assessor ZD pairs for quick lookup.
    // A cross-assessor is (i, j) with i in [1, half), j in [half, dim).
    // We use the motif component graph: each node in a component is a
    // cross-pair that participates in diagonal zero-products.
    let components = motif_components_for_cross_assessors(dim);
    let mut zd_pair_set: HashSet<(usize, usize)> = HashSet::new();
    for comp in &components {
        for &node in &comp.nodes {
            zd_pair_set.insert((node.0.min(node.1), node.0.max(node.1)));
        }
    }

    // Build the table
    let mut cells = Vec::with_capacity(size);
    let mut zd_count = 0usize;

    for row_idx in 0..size {
        let i = row_idx + 1; // basis index
        let mut row = Vec::with_capacity(size);
        for col_idx in 0..size {
            let j = col_idx + 1; // basis index
            let product_index = i ^ j;
            let sign = if i == j {
                // e_i * e_i = -1 for all imaginary units
                -1
            } else {
                cd_basis_mul_sign(dim, i, j)
            };

            // Check if this is a ZD pair: requires one index in [1,half)
            // and the other in [half, dim)
            let is_cross = (i < half && j >= half) || (j < half && i >= half);
            let is_zd = if is_cross {
                let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                zd_pair_set.contains(&(lo, hi))
            } else {
                false
            };

            if is_zd {
                zd_count += 1;
            }

            row.push(EtCell {
                row: i,
                col: j,
                product_index,
                sign,
                is_zero_divisor: is_zd,
            });
        }
        cells.push(row);
    }

    let total_cells = size * size;

    EmanationTable {
        dim,
        size,
        cells,
        zd_count,
        total_cells,
    }
}

/// Compute the sand mandala sparsity analysis for an emanation table.
///
/// Focuses on cross-assessor cells: those where one index is in the "low"
/// half [1, dim/2) and the other is in the "high" half [dim/2, dim).
pub fn sand_mandala_pattern(et: &EmanationTable) -> MandalaSummary {
    let half = et.dim / 2;
    let mut filled = 0usize;
    let mut total_cross = 0usize;

    // Track per-row and per-column ZD counts for cross-assessor cells
    let mut row_zd: HashMap<usize, usize> = HashMap::new();
    let mut row_total: HashMap<usize, usize> = HashMap::new();
    let mut col_zd: HashMap<usize, usize> = HashMap::new();
    let mut col_total: HashMap<usize, usize> = HashMap::new();

    for row in &et.cells {
        for cell in row {
            let i = cell.row;
            let j = cell.col;
            let is_cross = (i < half && j >= half) || (j < half && i >= half);
            if !is_cross {
                continue;
            }
            total_cross += 1;
            *row_total.entry(i).or_insert(0) += 1;
            *col_total.entry(j).or_insert(0) += 1;
            if cell.is_zero_divisor {
                filled += 1;
                *row_zd.entry(i).or_insert(0) += 1;
                *col_zd.entry(j).or_insert(0) += 1;
            }
        }
    }

    let fill_ratio = if total_cross > 0 {
        filled as f64 / total_cross as f64
    } else {
        0.0
    };

    let mut row_fill_ratios: Vec<f64> = row_total
        .iter()
        .map(|(idx, &tot)| row_zd.get(idx).copied().unwrap_or(0) as f64 / tot as f64)
        .collect();
    row_fill_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut col_fill_ratios: Vec<f64> = col_total
        .iter()
        .map(|(idx, &tot)| col_zd.get(idx).copied().unwrap_or(0) as f64 / tot as f64)
        .collect();
    col_fill_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    MandalaSummary {
        dim: et.dim,
        filled,
        total_cross,
        fill_ratio,
        row_fill_ratios,
        col_fill_ratios,
    }
}

/// Identify carry-bit overflow cells: cross-assessor pairs in dim=2N
/// that have ZDs at dim=N but NOT at dim=2N (or vice versa).
///
/// Returns (lost, gained) where:
/// - `lost`: pairs that were ZD at dim=N but not at dim=2N
/// - `gained`: pairs that were NOT ZD at dim=N but ARE at dim=2N
pub fn carry_bit_overflow_cells(dim: usize) -> (Vec<CrossPair>, Vec<CrossPair>) {
    assert!(dim >= 32 && dim.is_power_of_two());
    let parent_dim = dim / 2;

    // Get parent ZD pairs
    let parent_comps = motif_components_for_cross_assessors(parent_dim);
    let mut parent_zd_pairs: HashSet<CrossPair> = HashSet::new();
    for comp in &parent_comps {
        for &node in &comp.nodes {
            parent_zd_pairs.insert(node);
        }
    }

    // Get child ZD pairs (restricted to parent range for comparison)
    let child_comps = motif_components_for_cross_assessors(dim);
    let mut child_zd_pairs: HashSet<CrossPair> = HashSet::new();
    for comp in &child_comps {
        for &node in &comp.nodes {
            child_zd_pairs.insert(node);
        }
    }

    // Lost: in parent but not in child (restricted to parent range)
    let parent_half = parent_dim / 2;
    let lost: Vec<CrossPair> = parent_zd_pairs
        .iter()
        .filter(|&&(i, j)| {
            // This pair exists in parent algebra; check if the SAME indices
            // still participate in ZDs at the child dimension
            !child_zd_pairs.contains(&(i, j))
        })
        .copied()
        .collect();

    // Gained: in child (within parent range) but not in parent
    let gained: Vec<CrossPair> = child_zd_pairs
        .iter()
        .filter(|&&(i, j)| {
            i < parent_half
                && j < parent_dim
                && j >= parent_half
                && !parent_zd_pairs.contains(&(i, j))
        })
        .copied()
        .collect();

    (lost, gained)
}

/// Compute ET scaling data across multiple dimensions.
pub fn et_period_doubling(dims: &[usize]) -> Vec<EtScaling> {
    dims.iter()
        .map(|&dim| {
            let cross = cross_assessors(dim);
            let comps = motif_components_for_cross_assessors(dim);

            let mut zd_nodes: HashSet<CrossPair> = HashSet::new();
            for comp in &comps {
                for &node in &comp.nodes {
                    zd_nodes.insert(node);
                }
            }

            let size = dim - 1;
            let nodes_per_comp = if comps.is_empty() {
                0
            } else {
                comps[0].nodes.len()
            };

            EtScaling {
                dim,
                total_cells: size * size,
                zd_cells: zd_nodes.len(),
                n_cross_pairs: cross.len(),
                n_zd_pairs: zd_nodes.len(),
                zd_ratio: zd_nodes.len() as f64 / cross.len() as f64,
                n_components: comps.len(),
                nodes_per_component: nodes_per_comp,
            }
        })
        .collect()
}

/// Measure structural similarity between sub-blocks of parent and child ETs.
///
/// For each cell in the parent ET that falls within the cross-assessor region,
/// check if the corresponding cell in the child ET has the same ZD status.
/// Returns the fraction of matching cells (1.0 = identical, 0.0 = no overlap).
pub fn et_block_similarity(parent: &EmanationTable, child: &EmanationTable) -> f64 {
    assert_eq!(parent.dim * 2, child.dim, "child must be parent doubled");

    let parent_half = parent.dim / 2;
    let mut matches = 0usize;
    let mut total = 0usize;

    for parent_row in &parent.cells {
        for cell in parent_row {
            let i = cell.row;
            let j = cell.col;
            // Only compare cross-assessor cells
            let is_cross =
                (i < parent_half && j >= parent_half) || (j < parent_half && i >= parent_half);
            if !is_cross {
                continue;
            }
            total += 1;
            // Same indices in child table
            let child_cell = &child.cells[i - 1][j - 1];
            if cell.is_zero_divisor == child_cell.is_zero_divisor {
                matches += 1;
            }
        }
    }

    if total > 0 {
        matches as f64 / total as f64
    } else {
        1.0
    }
}
