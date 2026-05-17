//! Crystal-structure lookup, extinction rules, and selection-rule queries.
//!
//! Extracted from `crystal_symmetry.rs` as part of the #139 PH-MOD split.
//! Provides the high-level "structural query" API:
//!
//! - `CrystalStructureInfo`: name + space-group + point-group + lattice
//!   system + Bravais centering + numerical lattice parameters + density.
//! - `known_crystal_structures()`: reference list mined from open-licence
//!   crystallographic databases (COD / RRUFF / CRC Handbook 102nd ed. /
//!   Wyckoff 1963).  Future migration target for materials_data codegen.
//! - `space_groups_for_structure(lattice, point_group)`: filtered lookup.
//! - `allowed_reflection(sg_number, h, k, l)`: extinction rule.
//! - `is_allowed_transition(pg, initial, final, operator)`: character-table
//!   triple-product multiplicity check.
//!
//! Updates (2026-05-13): user-driven expansion from 3 entries (NaCl,
//! Diamond, Wurtzite) to 50+ entries covering the tourmaline supergroup,
//! foundational oxides + semiconductors + II-VI compounds + perovskites +
//! scheelites + tungsten bronzes + elemental metals (FCC/BCC/HCP). The
//! `CrystalStructureInfo` struct gained 9 new fields (lattice_a/b/c,
//! alpha/beta/gamma, atoms_per_unit_cell, density_g_cm3, primary_reference)
//! to falsify the "exhaustive crystal properties" coverage conjecture.

use super::{CharacterTable, LatticeSystem, PointGroup};

/// Convert a codegen-emitted point-group name string to the PointGroup enum.
/// Panics on an unknown name (build-time table must stay in sync with the enum).
fn point_group_from_codegen_name(name: &str) -> PointGroup {
    match name {
        "C1" => PointGroup::C1,
        "Ci" => PointGroup::Ci,
        "C2" => PointGroup::C2,
        "Cs" => PointGroup::Cs,
        "C2h" => PointGroup::C2h,
        "D2" => PointGroup::D2,
        "C2v" => PointGroup::C2v,
        "D2h" => PointGroup::D2h,
        "C4" => PointGroup::C4,
        "S4" => PointGroup::S4,
        "C4h" => PointGroup::C4h,
        "D4" => PointGroup::D4,
        "C4v" => PointGroup::C4v,
        "D2d" => PointGroup::D2d,
        "D4h" => PointGroup::D4h,
        "C3" => PointGroup::C3,
        "C3i" => PointGroup::C3i,
        "C3v" => PointGroup::C3v,
        "D3" => PointGroup::D3,
        "D3d" => PointGroup::D3d,
        "C6" => PointGroup::C6,
        "C3h" => PointGroup::C3h,
        "C6h" => PointGroup::C6h,
        "D6" => PointGroup::D6,
        "C6v" => PointGroup::C6v,
        "D3h" => PointGroup::D3h,
        "D6h" => PointGroup::D6h,
        "T" => PointGroup::T,
        "Td" => PointGroup::Td,
        "Th" => PointGroup::Th,
        "O" => PointGroup::O,
        "Oh" => PointGroup::Oh,
        other => panic!("unknown point group name in codegen table: {}", other),
    }
}

/// Convert a codegen-emitted lattice-system name string to the LatticeSystem enum.
fn lattice_system_from_codegen_name(name: &str) -> LatticeSystem {
    match name {
        "Triclinic" => LatticeSystem::Triclinic,
        "Monoclinic" => LatticeSystem::Monoclinic,
        "Orthorhombic" => LatticeSystem::Orthorhombic,
        "Tetragonal" => LatticeSystem::Tetragonal,
        "Hexagonal" => LatticeSystem::Hexagonal,
        "Rhombohedral" => LatticeSystem::Rhombohedral,
        "Cubic" => LatticeSystem::Cubic,
        other => panic!("unknown lattice system name in codegen table: {}", other),
    }
}

/// Convert one codegen tuple from materials_data::CRYSTAL_STRUCTURE_TABLE
/// into a CrystalStructureInfo instance.
fn structure_from_codegen_row(
    row: &(
        &'static str,
        u16,
        &'static str,
        &'static str,
        &'static str,
        char,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        u32,
        f64,
        &'static str,
    ),
) -> CrystalStructureInfo {
    CrystalStructureInfo {
        name: row.0,
        space_group_number: row.1,
        space_group_symbol: row.2,
        point_group: point_group_from_codegen_name(row.3),
        lattice_system: lattice_system_from_codegen_name(row.4),
        bravais_centering: row.5,
        lattice_a_angstrom: row.6,
        lattice_b_angstrom: row.7,
        lattice_c_angstrom: row.8,
        alpha_deg: row.9,
        beta_deg: row.10,
        gamma_deg: row.11,
        atoms_per_unit_cell: row.12,
        density_g_cm3: row.13,
        primary_reference: row.14,
    }
}

/// High-level structure record associating a name with its complete
/// crystallographic + quantitative-property classification.
///
/// All `*_angstrom` and `*_deg` fields use the conventional Hermann-Mauguin
/// "hexagonal setting" for trigonal lattices and the primitive setting for
/// cubic.  `atoms_per_unit_cell` is `Z` (number of formula units, not number
/// of atoms -- a Z=4 unit cell of NaCl contains 8 atoms total).
#[derive(Debug, Clone)]
pub struct CrystalStructureInfo {
    /// Common name (e.g. "NaCl", "schorl_tourmaline").
    pub name: &'static str,
    /// Space group number (1..=230).
    pub space_group_number: u16,
    /// Space group Hermann-Mauguin symbol.
    pub space_group_symbol: &'static str,
    /// Point group symmetry.
    pub point_group: PointGroup,
    /// Lattice system.
    pub lattice_system: LatticeSystem,
    /// Bravais centering (P, F, I, C, R).
    pub bravais_centering: char,
    /// a-axis length in angstrom.
    pub lattice_a_angstrom: f64,
    /// b-axis length in angstrom (== a for cubic/tetragonal/hexagonal).
    pub lattice_b_angstrom: f64,
    /// c-axis length in angstrom.
    pub lattice_c_angstrom: f64,
    /// alpha angle in degrees (== 90 for orthorhombic/tetragonal/cubic).
    pub alpha_deg: f64,
    /// beta angle in degrees.
    pub beta_deg: f64,
    /// gamma angle in degrees (== 120 for hexagonal/trigonal).
    pub gamma_deg: f64,
    /// Z: number of formula units in the conventional unit cell.
    pub atoms_per_unit_cell: u32,
    /// Theoretical / measured density in g/cm^3.
    pub density_g_cm3: f64,
    /// Primary literature reference (paper citation + DOI when known).
    pub primary_reference: &'static str,
}

/// Known crystal-structure registry mined from open-licence
/// crystallographic + mineralogical sources. References favour:
///
/// - Bosi et al. (2019) Lithos 322 "Tourmaline crystal chemistry".
/// - Wyckoff (1963) "Crystal Structures" vol. 1 (rock-salt, fluorite, zincblende).
/// - CRC Handbook of Chemistry and Physics 102nd ed. (2021) elemental metals.
/// - Crystallography Open Database (public domain).
/// - Mindat-cross-referenced + IMA-approved species.
///
/// As of 2026-05-13 contains 50 entries.  Tourmaline species use the
/// hexagonal setting of trigonal R3m.  Future expansion targets full
/// inorganic crystal chemistry: 200+ entries.
pub fn known_crystal_structures() -> Vec<CrystalStructureInfo> {
    // All 109 entries come from build-time TOML codegen at
    // crates/materials_data/data/crystal/crystal_structures.toml.
    // #127 Phase 6 complete: no entries remain inline.
    materials_data::CRYSTAL_STRUCTURE_TABLE
        .iter()
        .map(structure_from_codegen_row)
        .collect()
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

/// Look up structures by exact `name` (case-sensitive). Returns `None` if
/// not in the registry. Convenience for callers that already know the key.
pub fn structure_by_name(name: &str) -> Option<CrystalStructureInfo> {
    known_crystal_structures()
        .into_iter()
        .find(|s| s.name == name)
}

/// Extinction rule for `(hkl)` reflection at the given space group.
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

/// Selection-rule check: is `initial -> final` allowed under `operator_irrep`?
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn known_structure_names() -> HashSet<&'static str> {
        known_crystal_structures()
            .iter()
            .map(|structure| structure.name)
            .collect()
    }

    #[test]
    fn known_structures_has_360_plus_entries() {
        let n = known_crystal_structures().len();
        assert!(
            n >= 360,
            "Coverage regression: known_crystal_structures must list at least 360 \
             entries (currently {}). Adding new structures is encouraged; \
             removing any requires a rationale.",
            n
        );
    }

    #[test]
    fn rare_earth_garnet_and_laser_host_coverage() {
        let names = known_structure_names();
        for required in [
            "rare_earth_iron_garnet_yig",
            "ggg_gadolinium_gallium_garnet",
            "yvo4_yttrium_vanadate",
            "ceo2_ceria_fluorite",
            "smfeo3_orthoferrite_magnetic",
        ] {
            assert!(
                names.contains(required),
                "rare-earth/laser-host {} missing",
                required
            );
        }
    }

    #[test]
    fn additional_topological_material_coverage() {
        let names = known_structure_names();
        for required in [
            "zrte5_orthorhombic_topological",
            "wte2_td_weyl_typeii",
            "mote2_td_weyl",
            "snte_rocksalt_tci",
        ] {
            assert!(
                names.contains(required),
                "topological material {} missing",
                required
            );
        }
    }

    #[test]
    fn max_phase_and_mxene_coverage() {
        let names = known_structure_names();
        for required in [
            "ti2alc_max_phase",
            "v2alc_max_phase",
            "nb2alc_max_phase",
            "ti2c_mxene",
        ] {
            assert!(
                names.contains(required),
                "MAX phase/MXene {} missing",
                required
            );
        }
    }

    #[test]
    fn scintillator_and_pet_imaging_coverage() {
        let names = known_structure_names();
        for required in [
            "bgo_bismuth_germanate_bi4ge3o12",
            "lso_lutetium_oxyorthosilicate",
            "yso_yttrium_oxyorthosilicate",
        ] {
            assert!(
                names.contains(required),
                "scintillator {} missing",
                required
            );
        }
    }

    #[test]
    fn geophysical_lower_mantle_coverage() {
        let names = known_structure_names();
        for required in [
            "majorite_garnet_mg4si4o12",
            "perovskite_mgsio3_bridgmanite",
            "ferropericlase_mgo_feo",
        ] {
            assert!(
                names.contains(required),
                "lower-mantle phase {} missing",
                required
            );
        }
    }

    #[test]
    fn two_d_entries_document_slab_convention() {
        // PR #21 review followup: 2D materials (germanene, stanene, etc.) use
        // a finite c-axis as a slab-supercell convention; their density value
        // is therefore not a physical observable. Any entry whose name ends
        // in "_2d" must acknowledge the 2D / monolayer / buckled / slab
        // character in its primary_reference so future readers do not
        // misinterpret the bulk density.
        let two_d_keywords = ["monolayer", "buckled", "honeycomb", "slab", "2D"];
        let all = known_crystal_structures();
        let two_d_entries: Vec<&CrystalStructureInfo> =
            all.iter().filter(|s| s.name.ends_with("_2d")).collect();
        assert!(
            !two_d_entries.is_empty(),
            "No _2d entries found -- germanene_2d + stanene_2d at minimum should be present"
        );
        for entry in two_d_entries {
            let mentions_2d = two_d_keywords
                .iter()
                .any(|kw| entry.primary_reference.contains(kw));
            assert!(
                mentions_2d,
                "_2d entry {} must mention 2D character in primary_reference \
                 (one of: {:?}); current reference: {:?}",
                entry.name, two_d_keywords, entry.primary_reference
            );
        }
    }

    #[test]
    fn hume_rothery_intermetallic_coverage() {
        let names = known_structure_names();
        for required in [
            "beta_brass_cuzn_b2",
            "gamma_brass_cu5zn8",
            "epsilon_brass_cuzn3",
            "mgcu2_laves_c15",
            "mgzn2_laves_c14",
            "ni3al_l12_gamma_prime",
            "tial_l10_ordered",
        ] {
            assert!(
                names.contains(required),
                "Hume-Rothery intermetallic {} missing",
                required
            );
        }
    }

    #[test]
    fn cuprate_high_tc_extended_coverage() {
        let names = known_structure_names();
        for required in [
            "yba2cu3o6_tetragonal",
            "bi2sr2cacu2o8_bscco_2212",
            "hgba2ca2cu3o8_hg_1223",
            "ndnio2_nickelate_infinite_layer",
        ] {
            assert!(
                names.contains(required),
                "extended cuprate {} missing",
                required
            );
        }
    }

    #[test]
    fn biological_mineral_coverage() {
        let names = known_structure_names();
        for required in [
            "calcite_biogenic_eggshell",
            "aragonite_biogenic_nacre",
            "vaterite_caco3",
            "biological_apatite_dahllite",
            "magnetite_biogenic_magnetosome",
        ] {
            assert!(names.contains(required), "biomineral {} missing", required);
        }
    }

    #[test]
    fn additional_zeolite_framework_coverage() {
        let names = known_structure_names();
        for required in [
            "linde_a_zeolite",
            "chabazite_chabazite",
            "heulandite_heu",
            "analcime_ana",
            "sodalite_sod",
        ] {
            assert!(
                names.contains(required),
                "zeolite framework {} missing",
                required
            );
        }
    }

    #[test]
    fn niobate_ferroelectric_coverage() {
        let names = known_structure_names();
        for required in [
            "linbo3_trigonal_room_t",
            "litao3_trigonal",
            "knbo3_orthorhombic",
        ] {
            assert!(
                names.contains(required),
                "niobate/tantalate {} missing",
                required
            );
        }
    }

    #[test]
    fn organic_semiconductor_coverage() {
        let names = known_structure_names();
        for required in [
            "pentacene_herringbone",
            "rubrene_orthorhombic",
            "c60_buckminsterfullerene_fcc",
            "naphthalene",
            "anthracene",
        ] {
            assert!(
                names.contains(required),
                "organic semiconductor {} missing",
                required
            );
        }
    }

    #[test]
    fn mof_and_zeolite_coverage() {
        let names = known_structure_names();
        for required in [
            "mof_5_irmof_1",
            "hkust_1_cu_btc",
            "zif_8_zn_methylimidazolate",
            "uio_66_zr",
            "faujasite_zeolite_y",
            "zsm_5_mfi_framework",
            "mordenite_zeolite",
        ] {
            assert!(names.contains(required), "MOF/zeolite {} missing", required);
        }
    }

    #[test]
    fn ice_polymorph_coverage() {
        let names = known_structure_names();
        for required in [
            "ice_ih_hexagonal",
            "ice_ic_cubic",
            "ice_vii_cubic_high_pressure",
        ] {
            assert!(
                names.contains(required),
                "ice polymorph {} missing",
                required
            );
        }
    }

    #[test]
    fn nuclear_actinide_coverage() {
        let names = known_structure_names();
        for required in ["uo2_fluorite", "tho2_fluorite", "puo2_fluorite"] {
            assert!(
                names.contains(required),
                "actinide oxide {} missing",
                required
            );
        }
    }

    #[test]
    fn permanent_magnet_coverage() {
        let names = known_structure_names();
        for required in ["nd2fe14b_neomag", "smco5_cacu5_type", "fept_l10_ordered"] {
            assert!(
                names.contains(required),
                "hard-magnet phase {} missing",
                required
            );
        }
    }

    #[test]
    fn iii_v_compound_semiconductor_coverage() {
        let names = known_structure_names();
        for required in [
            "alas_zincblende",
            "aln_wurtzite",
            "alsb_zincblende",
            "inas_zincblende",
            "gap_zincblende",
            "gasb_zincblende",
            "hgte_zincblende",
        ] {
            assert!(
                names.contains(required),
                "III-V/II-VI semiconductor {} missing",
                required
            );
        }
    }

    #[test]
    fn refractory_carbide_diboride_coverage() {
        let names = known_structure_names();
        for required in [
            "titanium_carbide_tic",
            "zirconium_carbide_zrc",
            "tantalum_carbide_tac",
            "zirconium_diboride_zrb2",
            "hafnium_diboride_hfb2",
            "tantalum_diboride_tab2",
        ] {
            assert!(
                names.contains(required),
                "refractory ceramic {} missing",
                required
            );
        }
    }

    #[test]
    fn high_pressure_geophysical_phase_coverage() {
        let names = known_structure_names();
        for required in [
            "post_perovskite_mgsio3",
            "stishovite_sio2_rutile",
            "iron_epsilon_hcp",
            "iron_gamma_fcc",
            "ringwoodite_mg2sio4_spinel",
            "wadsleyite_mg2sio4",
        ] {
            assert!(
                names.contains(required),
                "high-pressure/geophysical phase {} missing",
                required
            );
        }
    }

    #[test]
    fn cuprate_superconductor_parent_coverage() {
        let names = known_structure_names();
        for required in ["la2cuo4_tetragonal_t_phase", "nd2cuo4_t_prime"] {
            assert!(
                names.contains(required),
                "cuprate parent {} missing",
                required
            );
        }
    }

    #[test]
    fn halide_perovskite_coverage() {
        let names = known_structure_names();
        for required in [
            "cspbi3_orthorhombic_gamma",
            "mapbi3_tetragonal",
            "fapbbr3_cubic",
            "cspbbr3_orthorhombic",
        ] {
            assert!(
                names.contains(required),
                "halide perovskite {} missing from registry",
                required
            );
        }
    }

    #[test]
    fn iron_pnictide_superconductor_coverage() {
        let names = known_structure_names();
        for required in ["bafe2as2_122", "lafeaso_1111", "fese_pbo_type"] {
            assert!(
                names.contains(required),
                "iron-pnictide superconductor {} missing",
                required
            );
        }
    }

    #[test]
    fn weyl_dirac_semimetal_coverage() {
        let names = known_structure_names();
        for required in [
            "tantalum_arsenide_taas",
            "niobium_phosphide_nbp",
            "cd3as2_dirac",
        ] {
            assert!(
                names.contains(required),
                "Weyl/Dirac semimetal {} missing",
                required
            );
        }
    }

    #[test]
    fn ceramic_and_nlo_coverage() {
        let names = known_structure_names();
        for required in [
            "yag_y3al5o12",
            "magnesium_fluoride_rutile",
            "bbo_beta_barium_borate",
            "kdp_potassium_dihydrogen_phosphate",
            "zirconia_cubic_yttria_stabilized",
            "zirconia_monoclinic_baddeleyite",
            "tungsten_carbide_alpha",
        ] {
            assert!(
                names.contains(required),
                "ceramic/NLO crystal {} missing",
                required
            );
        }
    }

    #[test]
    fn semiconductor_iii_v_ii_vi_coverage() {
        let names = known_structure_names();
        for required in [
            "gaas_zincblende",
            "gan_wurtzite",
            "inp_zincblende",
            "cdte_zincblende",
            "cdse_wurtzite",
            "hgse_zincblende",
            "silicon_carbide_3C_beta",
            "silicon_carbide_6H_alpha",
        ] {
            assert!(
                names.contains(required),
                "compound semiconductor {} missing from registry",
                required
            );
        }
    }

    #[test]
    fn energy_materials_coverage() {
        let names = known_structure_names();
        for required in [
            "licoo2_layered",
            "lifepo4_olivine",
            "limn2o4_spinel",
            "pbte_rocksalt",
        ] {
            assert!(
                names.contains(required),
                "energy-material {} missing",
                required
            );
        }
    }

    #[test]
    fn silicate_coverage_includes_olivines_and_polymorphs() {
        let names = known_structure_names();
        for required in [
            "forsterite_olivine",
            "fayalite_olivine",
            "kyanite_al2sio5",
            "andalusite_al2sio5",
            "sillimanite_al2sio5",
            "topaz_al2sio4f2",
            "beryl_be3al2si6o18",
            "zircon_zrsio4",
        ] {
            assert!(names.contains(required), "silicate {} missing", required);
        }
    }

    #[test]
    fn modern_materials_coverage() {
        let names = known_structure_names();
        for required in [
            "graphite_hexagonal_2H",
            "bismuth_telluride_bi2te3",
            "bismuth_selenide_bi2se3",
            "cspbbr3_orthorhombic",
            "ybco_orthorhombic",
            "magnesium_diboride_mgb2",
        ] {
            assert!(
                names.contains(required),
                "modern material {} (2D/topological/superconductor/halide-perovskite) missing",
                required
            );
        }
    }

    #[test]
    fn tourmaline_supergroup_has_all_eight_species() {
        let registry = known_crystal_structures();
        for species in [
            "schorl_tourmaline",
            "dravite_tourmaline",
            "elbaite_tourmaline",
            "uvite_tourmaline",
            "liddicoatite_tourmaline",
            "rossmanite_tourmaline",
            "foitite_tourmaline",
            "povondraite_tourmaline",
        ] {
            let info = registry
                .iter()
                .find(|s| s.name == species)
                .unwrap_or_else(|| panic!("tourmaline species {} missing from registry", species));
            assert_eq!(info.space_group_number, 160, "{} not in R3m", species);
            assert_eq!(info.point_group, PointGroup::C3v);
            assert_eq!(info.lattice_system, LatticeSystem::Hexagonal);
            assert_eq!(info.bravais_centering, 'R');
            assert!(info.lattice_a_angstrom > 15.0 && info.lattice_a_angstrom < 17.0);
            assert!(info.lattice_c_angstrom > 7.0 && info.lattice_c_angstrom < 8.0);
            assert!((info.gamma_deg - 120.0).abs() < 1e-9);
            assert!(info.density_g_cm3 > 2.9 && info.density_g_cm3 < 3.4);
        }
    }

    #[test]
    fn lattice_geometry_sanity() {
        for s in known_crystal_structures().iter() {
            assert!(s.lattice_a_angstrom > 0.0, "{} has non-positive a", s.name);
            assert!(s.lattice_b_angstrom > 0.0, "{} has non-positive b", s.name);
            assert!(s.lattice_c_angstrom > 0.0, "{} has non-positive c", s.name);
            assert!((s.alpha_deg > 0.0) && (s.alpha_deg < 180.0));
            assert!((s.beta_deg > 0.0) && (s.beta_deg < 180.0));
            assert!((s.gamma_deg > 0.0) && (s.gamma_deg < 180.0));
            assert!(s.atoms_per_unit_cell >= 1);
            assert!(
                s.density_g_cm3 > 0.0 && s.density_g_cm3 < 30.0,
                "{} density {} outside plausible range",
                s.name,
                s.density_g_cm3
            );
            assert!(
                !s.primary_reference.is_empty(),
                "{} missing reference",
                s.name
            );
        }
    }

    #[test]
    fn cubic_lattices_have_a_equals_b_equals_c() {
        for s in known_crystal_structures()
            .iter()
            .filter(|s| s.lattice_system == LatticeSystem::Cubic)
        {
            assert!(
                (s.lattice_a_angstrom - s.lattice_b_angstrom).abs() < 1e-9,
                "{}",
                s.name
            );
            assert!(
                (s.lattice_a_angstrom - s.lattice_c_angstrom).abs() < 1e-9,
                "{}",
                s.name
            );
            assert!((s.alpha_deg - 90.0).abs() < 1e-9);
            assert!((s.beta_deg - 90.0).abs() < 1e-9);
            assert!((s.gamma_deg - 90.0).abs() < 1e-9);
        }
    }

    #[test]
    fn hexagonal_lattices_have_120_gamma() {
        for s in known_crystal_structures()
            .iter()
            .filter(|s| s.lattice_system == LatticeSystem::Hexagonal)
        {
            assert!((s.gamma_deg - 120.0).abs() < 1e-9, "{}", s.name);
            assert!((s.alpha_deg - 90.0).abs() < 1e-9);
            assert!((s.beta_deg - 90.0).abs() < 1e-9);
            assert!((s.lattice_a_angstrom - s.lattice_b_angstrom).abs() < 1e-9);
        }
    }

    #[test]
    fn structure_by_name_finds_diamond() {
        let s = structure_by_name("Diamond").unwrap();
        assert_eq!(s.space_group_number, 227);
        assert!((s.lattice_a_angstrom - 3.5670).abs() < 1e-9);
    }

    #[test]
    fn structure_by_name_finds_tourmaline_species() {
        let s = structure_by_name("elbaite_tourmaline").unwrap();
        assert_eq!(s.lattice_system, LatticeSystem::Hexagonal);
        assert_eq!(s.space_group_symbol, "R3m");
    }

    #[test]
    fn structure_by_name_returns_none_for_unknown() {
        assert!(structure_by_name("not_a_real_mineral").is_none());
    }
}
