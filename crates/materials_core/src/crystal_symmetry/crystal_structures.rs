//! Crystal-structure lookup, extinction rules, and selection-rule queries.
//!
//! Extracted from `crystal_symmetry.rs` as part of the #139 PH-MOD split.
//! Provides the high-level "structural query" API:
//!
//! - `CrystalStructureInfo`: name, space-group number/symbol, point group,
//!   lattice system, Bravais centering.
//! - `known_crystal_structures()`: hardcoded reference list (NaCl, Diamond,
//!   Wurtzite). Future migration target for materials_data codegen (#127).
//! - `space_groups_for_structure(lattice, point_group)`: filtered lookup.
//! - `allowed_reflection(sg_number, h, k, l)`: extinction rule for the
//!   given space group (F-centering, diamond glide, hexagonal selection).
//! - `is_allowed_transition(pg, initial, final, operator)`: character-table
//!   triple-product multiplicity check for electronic transitions.

use super::{CharacterTable, LatticeSystem, PointGroup};

/// High-level structure record associating a name with its space-group +
/// point-group + lattice classification.
#[derive(Debug, Clone)]
pub struct CrystalStructureInfo {
    /// Common name (e.g., "NaCl").
    pub name: &'static str,
    /// Space group number (1-230).
    pub space_group_number: u16,
    /// Space group Hermann-Mauguin symbol.
    pub space_group_symbol: &'static str,
    /// Point group symmetry.
    pub point_group: PointGroup,
    /// Lattice system.
    pub lattice_system: LatticeSystem,
    /// Bravais centering (P, F, I, C, R).
    pub bravais_centering: char,
}

/// Common known crystal structures (International Tables reference).
pub fn known_crystal_structures() -> Vec<CrystalStructureInfo> {
    vec![
        CrystalStructureInfo {
            name: "NaCl",
            space_group_number: 225,
            space_group_symbol: "Fm-3m",
            point_group: PointGroup::Oh,
            lattice_system: LatticeSystem::Cubic,
            bravais_centering: 'F',
        },
        CrystalStructureInfo {
            name: "Diamond",
            space_group_number: 227,
            space_group_symbol: "Fd-3m",
            point_group: PointGroup::Oh,
            lattice_system: LatticeSystem::Cubic,
            bravais_centering: 'F',
        },
        CrystalStructureInfo {
            name: "Wurtzite",
            space_group_number: 186,
            space_group_symbol: "P63mc",
            point_group: PointGroup::C6v,
            lattice_system: LatticeSystem::Hexagonal,
            bravais_centering: 'P',
        },
    ]
}

/// Look up structures matching both `lattice` and `point_group`.
pub fn space_groups_for_structure(
    lattice: LatticeSystem,
    point_group: PointGroup,
) -> Vec<CrystalStructureInfo> {
    known_crystal_structures()
        .into_iter()
        .filter(|s| s.lattice_system == lattice && s.point_group == point_group)
        .collect()
}

/// Extinction rule for `(hkl)` reflection at the given space group.
/// Covers Fm-3m (225, NaCl), Fd-3m (227, Diamond), P63mc (186, Wurtzite);
/// returns `true` (allowed) for unknown space groups (conservative default).
pub fn allowed_reflection(space_group_number: u16, h: i32, k: i32, l: i32) -> bool {
    match space_group_number {
        225 => {
            let all_even = h % 2 == 0 && k % 2 == 0 && l % 2 == 0;
            let all_odd = h % 2 != 0 && k % 2 != 0 && l % 2 != 0;
            all_even || all_odd
        }
        227 => {
            let all_even = h % 2 == 0 && k % 2 == 0 && l % 2 == 0;
            let all_odd = h % 2 != 0 && k % 2 != 0 && l % 2 != 0;
            if all_even {
                (h + k + l) % 4 == 0
            } else if all_odd {
                (h + k + l) % 4 == 3
            } else {
                false
            }
        }
        186 => l % 2 == 0,
        _ => true,
    }
}

/// Selection-rule check: is the transition `initial -> final` allowed under
/// the dipole-like operator `operator_irrep`? Returns `true` when the triple
/// product `conj(chi_f) * chi_op * chi_i` averaged over conjugacy classes
/// has multiplicity > 0 (contains the totally-symmetric representation).
/// Falls back to `true` (allowed) when the character table is unavailable
/// or any irrep label is not found.
pub fn is_allowed_transition(
    point_group: PointGroup,
    initial_irrep: &str,
    final_irrep: &str,
    operator_irrep: &str,
) -> bool {
    let table = match CharacterTable::for_point_group(point_group) {
        Some(t) => t,
        None => return true,
    };

    let find_row =
        |label: &str| -> Option<usize> { table.irreps.iter().position(|ir| ir.label == label) };

    let row_i = match find_row(initial_irrep) {
        Some(r) => r,
        None => return true,
    };
    let row_f = match find_row(final_irrep) {
        Some(r) => r,
        None => return true,
    };
    let row_op = match find_row(operator_irrep) {
        Some(r) => r,
        None => return true,
    };

    let group_order: f64 = table.classes.iter().map(|c| c.count as f64).sum();
    if group_order < 1.0 {
        return true;
    }

    let mut sum_re = 0.0_f64;
    let mut sum_im = 0.0_f64;
    for (j, class) in table.classes.iter().enumerate() {
        let n_k = class.count as f64;
        let (fi_re, fi_im) = table.characters[row_i][j];
        let (ff_re, ff_im) = table.characters[row_f][j];
        let (fo_re, fo_im) = table.characters[row_op][j];
        let (a_re, a_im) = (ff_re, -ff_im);
        let (b_re, b_im) = (a_re * fo_re - a_im * fo_im, a_re * fo_im + a_im * fo_re);
        let c_re = b_re * fi_re - b_im * fi_im;
        let c_im = b_re * fi_im + b_im * fi_re;
        sum_re += n_k * c_re;
        sum_im += n_k * c_im;
    }

    debug_assert!(
        (sum_im / group_order).abs() < 1e-6,
        "imaginary part of multiplicity should be ~0, got {}",
        sum_im / group_order
    );

    let multiplicity = sum_re / group_order;
    multiplicity > 0.5 - 1e-6
}
