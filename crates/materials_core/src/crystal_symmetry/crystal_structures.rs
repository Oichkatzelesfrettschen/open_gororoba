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

/// Helper for cubic structures (a = b = c, all angles 90).
const fn cubic(a: f64, z: u32, density: f64) -> (f64, f64, f64, f64, f64, f64, u32, f64) {
    (a, a, a, 90.0, 90.0, 90.0, z, density)
}

/// Helper for tetragonal structures (a = b, c distinct, all angles 90).
const fn tetragonal(
    a: f64,
    c: f64,
    z: u32,
    density: f64,
) -> (f64, f64, f64, f64, f64, f64, u32, f64) {
    (a, a, c, 90.0, 90.0, 90.0, z, density)
}

/// Helper for hexagonal / trigonal hexagonal-setting (a = b, gamma = 120).
const fn hexagonal(
    a: f64,
    c: f64,
    z: u32,
    density: f64,
) -> (f64, f64, f64, f64, f64, f64, u32, f64) {
    (a, a, c, 90.0, 90.0, 120.0, z, density)
}

/// Helper for orthorhombic structures.
const fn orthorhombic(
    a: f64,
    b: f64,
    c: f64,
    z: u32,
    density: f64,
) -> (f64, f64, f64, f64, f64, f64, u32, f64) {
    (a, b, c, 90.0, 90.0, 90.0, z, density)
}

/// Convenience builder.
fn make(
    name: &'static str,
    sg_num: u16,
    sg_sym: &'static str,
    pg: PointGroup,
    ls: LatticeSystem,
    centering: char,
    geom: (f64, f64, f64, f64, f64, f64, u32, f64),
    reference: &'static str,
) -> CrystalStructureInfo {
    let (a, b, c, al, be, ga, z, density) = geom;
    CrystalStructureInfo {
        name,
        space_group_number: sg_num,
        space_group_symbol: sg_sym,
        point_group: pg,
        lattice_system: ls,
        bravais_centering: centering,
        lattice_a_angstrom: a,
        lattice_b_angstrom: b,
        lattice_c_angstrom: c,
        alpha_deg: al,
        beta_deg: be,
        gamma_deg: ga,
        atoms_per_unit_cell: z,
        density_g_cm3: density,
        primary_reference: reference,
    }
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
    vec![
        // ----- Pre-existing trio -----
        make("NaCl", 225, "Fm-3m", PointGroup::Oh, LatticeSystem::Cubic, 'F',
             cubic(5.6402, 4, 2.165),
             "Wyckoff (1963) Crystal Structures vol. 1 p.85"),
        make("Diamond", 227, "Fd-3m", PointGroup::Oh, LatticeSystem::Cubic, 'F',
             cubic(3.5670, 8, 3.52),
             "Wyckoff (1963) Crystal Structures vol. 1 p.27"),
        make("Wurtzite", 186, "P63mc", PointGroup::C6v, LatticeSystem::Hexagonal, 'P',
             hexagonal(3.823, 6.261, 2, 4.09),
             "Wyckoff (1963) Crystal Structures vol. 1 p.111"),

        // ----- Tourmaline supergroup (8 species, all R3m hexagonal setting) -----
        make("schorl_tourmaline", 160, "R3m", PointGroup::C3v, LatticeSystem::Hexagonal, 'R',
             hexagonal(15.985, 7.220, 3, 3.20),
             "Henry et al. (2011) Am. Mineralogist 96, 895; Bosi et al. (2019) Lithos 322."),
        make("dravite_tourmaline", 160, "R3m", PointGroup::C3v, LatticeSystem::Hexagonal, 'R',
             hexagonal(15.945, 7.224, 3, 3.05),
             "Henry et al. (2011) Am. Mineralogist 96, 895."),
        make("elbaite_tourmaline", 160, "R3m", PointGroup::C3v, LatticeSystem::Hexagonal, 'R',
             hexagonal(15.842, 7.106, 3, 3.05),
             "Henry et al. (2011) Am. Mineralogist 96, 895; GIA Gem Encyclopedia."),
        make("uvite_tourmaline", 160, "R3m", PointGroup::C3v, LatticeSystem::Hexagonal, 'R',
             hexagonal(15.961, 7.207, 3, 3.10),
             "Henry et al. (2011) Am. Mineralogist 96, 895."),
        make("liddicoatite_tourmaline", 160, "R3m", PointGroup::C3v, LatticeSystem::Hexagonal, 'R',
             hexagonal(15.870, 7.124, 3, 3.02),
             "Dirlam et al. (2002) Gems & Gemology 38(3)."),
        make("rossmanite_tourmaline", 160, "R3m", PointGroup::C3v, LatticeSystem::Hexagonal, 'R',
             hexagonal(15.820, 7.080, 3, 3.00),
             "Selway et al. (1998) Am. Mineralogist 83 (rossmanite type description)."),
        make("foitite_tourmaline", 160, "R3m", PointGroup::C3v, LatticeSystem::Hexagonal, 'R',
             hexagonal(15.967, 7.126, 3, 3.18),
             "MacDonald & Hawthorne (1995) Can. Mineralogist 33."),
        make("povondraite_tourmaline", 160, "R3m", PointGroup::C3v, LatticeSystem::Hexagonal, 'R',
             hexagonal(16.186, 7.444, 3, 3.32),
             "Grice et al. (1993) Can. Mineralogist 31."),

        // ----- Foundational oxides + their polymorphs -----
        make("alumina_sapphire", 167, "R-3c", PointGroup::D3d, LatticeSystem::Hexagonal, 'R',
             hexagonal(4.760, 12.991, 6, 3.987),
             "Malitson & Dodge (1972) J.Opt.Soc.Am. 62, 1405."),
        make("rutile_tio2", 136, "P4_2/mnm", PointGroup::D4h, LatticeSystem::Tetragonal, 'P',
             tetragonal(4.5937, 2.9587, 2, 4.23),
             "DeVore (1951) J.Opt.Soc.Am. 41, 416; Wyckoff vol. 1 p.250."),
        make("anatase_tio2", 141, "I4_1/amd", PointGroup::D4h, LatticeSystem::Tetragonal, 'I',
             tetragonal(3.7842, 9.5146, 4, 3.895),
             "Burdett et al. (1987) J.Am.Chem.Soc. 109, 3639."),
        make("brookite_tio2", 61, "Pbca", PointGroup::D2h, LatticeSystem::Orthorhombic, 'P',
             orthorhombic(9.184, 5.447, 5.145, 8, 4.123),
             "Meagher & Lager (1979) Can. Mineralogist 17 p.77."),
        make("alpha_quartz", 152, "P3_121", PointGroup::D3, LatticeSystem::Hexagonal, 'P',
             hexagonal(4.9134, 5.4053, 3, 2.65),
             "Ghosh (1999) Opt. Commun. 163, 95 (Sellmeier dispersion)."),
        make("calcite", 167, "R-3c", PointGroup::D3d, LatticeSystem::Hexagonal, 'R',
             hexagonal(4.9896, 17.0610, 6, 2.711),
             "Effenberger et al. (1981) Z. Kristallogr. 156 p.233."),
        make("aragonite", 62, "Pmcn", PointGroup::D2h, LatticeSystem::Orthorhombic, 'P',
             orthorhombic(4.962, 7.969, 5.744, 4, 2.93),
             "Pokroy et al. (2007) J. Struct. Biol. 159, 261."),
        make("fluorite_caf2", 225, "Fm-3m", PointGroup::Oh, LatticeSystem::Cubic, 'F',
             cubic(5.4626, 4, 3.180),
             "Wyckoff (1963) vol. 1 p.239."),
        make("spinel_mgal2o4", 227, "Fd-3m", PointGroup::Oh, LatticeSystem::Cubic, 'F',
             cubic(8.0800, 8, 3.578),
             "Hill et al. (1979) Phys. Chem. Minerals 4, 317."),
        make("galena_pbs", 225, "Fm-3m", PointGroup::Oh, LatticeSystem::Cubic, 'F',
             cubic(5.9362, 4, 7.597),
             "Wyckoff (1963) vol. 1 p.86 (NaCl-type)."),
        make("pyrite_fes2", 205, "Pa-3", PointGroup::Th, LatticeSystem::Cubic, 'P',
             cubic(5.4179, 4, 5.011),
             "Brostigen & Kjekshus (1969) Acta Chem. Scand. 23, 2186."),
        make("sphalerite_zns", 216, "F-43m", PointGroup::Td, LatticeSystem::Cubic, 'F',
             cubic(5.4093, 4, 4.090),
             "Wyckoff (1963) vol. 1 p.108 (zincblende)."),
        make("scheelite_cawo4", 88, "I4_1/a", PointGroup::C4h, LatticeSystem::Tetragonal, 'I',
             tetragonal(5.243, 11.376, 4, 6.12),
             "Hazen et al. (1985) Am. Mineralogist 70, 1029."),
        make("stolzite_pbwo4", 88, "I4_1/a", PointGroup::C4h, LatticeSystem::Tetragonal, 'I',
             tetragonal(5.4602, 12.0467, 4, 8.28),
             "Moreau et al. (1996) J. Phys. Chem. Solids 57, 547."),
        make("wurtzite_zno", 186, "P63mc", PointGroup::C6v, LatticeSystem::Hexagonal, 'P',
             hexagonal(3.2495, 5.2069, 2, 5.61),
             "Karzel et al. (1996) Phys. Rev. B 53, 11425."),
        make("rocksalt_lif", 225, "Fm-3m", PointGroup::Oh, LatticeSystem::Cubic, 'F',
             cubic(4.0269, 4, 2.640),
             "Wyckoff (1963) vol. 1 p.85."),
        make("perovskite_srtio3", 221, "Pm-3m", PointGroup::Oh, LatticeSystem::Cubic, 'P',
             cubic(3.9050, 1, 5.12),
             "Cardona (1965) Phys. Rev. 140, A651; van Benthem (2001) JAP 90, 6156."),
        make("perovskite_batio3_tetragonal", 99, "P4mm", PointGroup::C4v, LatticeSystem::Tetragonal, 'P',
             tetragonal(3.992, 4.036, 1, 6.020),
             "Megaw (1947) Proc. R. Soc. Lond. A 189, 261."),
        make("barium_titanate_cubic", 221, "Pm-3m", PointGroup::Oh, LatticeSystem::Cubic, 'P',
             cubic(4.0093, 1, 6.012),
             "Megaw (1947) Proc. R. Soc. Lond. A 189, 261."),
        make("lanthanum_titanate", 62, "Pnma", PointGroup::D2h, LatticeSystem::Orthorhombic, 'P',
             orthorhombic(5.633, 7.901, 5.616, 4, 6.39),
             "MacLean et al. (1979) Mater. Res. Bull. 14, 567."),
        make("indium_tin_oxide_bixbyite", 206, "Ia-3", PointGroup::Th, LatticeSystem::Cubic, 'I',
             cubic(10.117, 16, 7.14),
             "Hamberg & Granqvist (1986) J.Appl.Phys. 60, R123."),

        // ----- Tungstate / molybdate scheelite-class -----
        make("tungsten_trioxide_gamma", 14, "P2_1/n", PointGroup::C2h, LatticeSystem::Monoclinic, 'P',
             (7.301, 7.539, 7.689, 90.0, 90.91, 90.0, 8, 7.16),
             "Tilley (1995) Defect Crystal Chemistry p.114 (gamma-WO3 RT polymorph)."),
        make("cesium_tungsten_bronze", 191, "P6/mmm", PointGroup::D6h, LatticeSystem::Hexagonal, 'P',
             hexagonal(7.396, 7.610, 3, 6.6),
             "Magneli (1953) Acta Chem. Scand. 7, 315 (hexagonal tungsten bronze)."),

        // ----- Elemental metals (16 entries) -----
        make("Au_gold", 225, "Fm-3m", PointGroup::Oh, LatticeSystem::Cubic, 'F',
             cubic(4.0782, 4, 19.30), "CRC Handbook 102nd ed."),
        make("Ag_silver", 225, "Fm-3m", PointGroup::Oh, LatticeSystem::Cubic, 'F',
             cubic(4.0853, 4, 10.49), "CRC Handbook 102nd ed."),
        make("Cu_copper", 225, "Fm-3m", PointGroup::Oh, LatticeSystem::Cubic, 'F',
             cubic(3.6147, 4, 8.96), "CRC Handbook 102nd ed."),
        make("Al_aluminum", 225, "Fm-3m", PointGroup::Oh, LatticeSystem::Cubic, 'F',
             cubic(4.0495, 4, 2.70), "CRC Handbook 102nd ed."),
        make("Pt_platinum", 225, "Fm-3m", PointGroup::Oh, LatticeSystem::Cubic, 'F',
             cubic(3.9239, 4, 21.45), "CRC Handbook 102nd ed."),
        make("Pd_palladium", 225, "Fm-3m", PointGroup::Oh, LatticeSystem::Cubic, 'F',
             cubic(3.8907, 4, 12.02), "CRC Handbook 102nd ed."),
        make("Ni_nickel", 225, "Fm-3m", PointGroup::Oh, LatticeSystem::Cubic, 'F',
             cubic(3.5240, 4, 8.91), "CRC Handbook 102nd ed."),
        make("Cr_chromium", 229, "Im-3m", PointGroup::Oh, LatticeSystem::Cubic, 'I',
             cubic(2.8839, 2, 7.19), "CRC Handbook 102nd ed."),
        make("W_tungsten", 229, "Im-3m", PointGroup::Oh, LatticeSystem::Cubic, 'I',
             cubic(3.1652, 2, 19.25), "CRC Handbook 102nd ed."),
        make("Fe_alpha", 229, "Im-3m", PointGroup::Oh, LatticeSystem::Cubic, 'I',
             cubic(2.8665, 2, 7.874), "CRC Handbook 102nd ed."),
        make("Be_beryllium", 194, "P6_3/mmc", PointGroup::D6h, LatticeSystem::Hexagonal, 'P',
             hexagonal(2.2858, 3.5843, 2, 1.85), "CRC Handbook 102nd ed."),
        make("Ti_alpha", 194, "P6_3/mmc", PointGroup::D6h, LatticeSystem::Hexagonal, 'P',
             hexagonal(2.9505, 4.6826, 2, 4.51), "CRC Handbook 102nd ed."),
        make("Mg_magnesium", 194, "P6_3/mmc", PointGroup::D6h, LatticeSystem::Hexagonal, 'P',
             hexagonal(3.2094, 5.2105, 2, 1.738), "CRC Handbook 102nd ed."),
        make("Zn_zinc", 194, "P6_3/mmc", PointGroup::D6h, LatticeSystem::Hexagonal, 'P',
             hexagonal(2.6649, 4.9468, 2, 7.134), "CRC Handbook 102nd ed."),
        make("Si_silicon", 227, "Fd-3m", PointGroup::Oh, LatticeSystem::Cubic, 'F',
             cubic(5.4307, 8, 2.329), "CRC Handbook 102nd ed.; Wyckoff vol. 1 p.27."),
        make("Ge_germanium", 227, "Fd-3m", PointGroup::Oh, LatticeSystem::Cubic, 'F',
             cubic(5.6580, 8, 5.323), "CRC Handbook 102nd ed."),

        // ----- Additional silicates -----
        make("forsterite_olivine", 62, "Pbnm", PointGroup::D2h, LatticeSystem::Orthorhombic, 'P',
             orthorhombic(10.196, 5.981, 4.755, 4, 3.275),
             "Birle et al. (1968) Am. Mineralogist 53, 807 (Mg2SiO4 olivine endmember)."),
        make("fayalite_olivine", 62, "Pbnm", PointGroup::D2h, LatticeSystem::Orthorhombic, 'P',
             orthorhombic(10.481, 6.087, 4.819, 4, 4.392),
             "Smyth (1975) Am. Mineralogist 60, 1092 (Fe2SiO4 olivine endmember)."),
        make("almandine_garnet", 230, "Ia-3d", PointGroup::Oh, LatticeSystem::Cubic, 'I',
             cubic(11.519, 8, 4.318),
             "Geller (1967) Z. Kristallogr. 125, 1 (Fe3Al2(SiO4)3 garnet)."),
        make("pyrope_garnet", 230, "Ia-3d", PointGroup::Oh, LatticeSystem::Cubic, 'I',
             cubic(11.456, 8, 3.582),
             "Novak & Gibbs (1971) Am. Mineralogist 56, 791 (Mg3Al2(SiO4)3 garnet)."),
        make("muscovite_2M1", 15, "C2/c", PointGroup::C2h, LatticeSystem::Monoclinic, 'C',
             (5.199, 9.027, 20.106, 90.0, 95.74, 90.0, 4, 2.83),
             "Guggenheim (1981) Am. Mineralogist 66, 1221 (KAl2(AlSi3)O10(OH)2 mica)."),
        make("albite_feldspar", 2, "C-1", PointGroup::Ci, LatticeSystem::Triclinic, 'C',
             (8.144, 12.787, 7.160, 94.26, 116.59, 87.65, 4, 2.62),
             "Smith (1974) Feldspar Minerals vol. 2 (NaAlSi3O8 plagioclase endmember)."),
        make("orthoclase_feldspar", 12, "C2/m", PointGroup::C2h, LatticeSystem::Monoclinic, 'C',
             (8.564, 12.964, 7.198, 90.0, 116.07, 90.0, 4, 2.56),
             "Smith (1974) Feldspar Minerals vol. 1 (KAlSi3O8 alkali feldspar)."),
        make("zircon_zrsio4", 141, "I4_1/amd", PointGroup::D4h, LatticeSystem::Tetragonal, 'I',
             tetragonal(6.6042, 5.9796, 4, 4.65),
             "Robinson et al. (1971) Am. Mineralogist 56, 782 (ZrSiO4 zircon)."),
        make("kyanite_al2sio5", 2, "P-1", PointGroup::Ci, LatticeSystem::Triclinic, 'P',
             (7.1262, 7.852, 5.5724, 89.99, 101.11, 106.03, 4, 3.610),
             "Burnham (1963) Z. Kristallogr. 118, 337 (Al2SiO5 kyanite polymorph)."),
        make("sillimanite_al2sio5", 62, "Pbnm", PointGroup::D2h, LatticeSystem::Orthorhombic, 'P',
             orthorhombic(7.4843, 7.6724, 5.7707, 4, 3.247),
             "Winter & Ghose (1979) Am. Mineralogist 64, 573 (Al2SiO5 sillimanite)."),
        make("andalusite_al2sio5", 58, "Pnnm", PointGroup::D2h, LatticeSystem::Orthorhombic, 'P',
             orthorhombic(7.7942, 7.8985, 5.5594, 4, 3.144),
             "Burnham & Buerger (1961) Z. Kristallogr. 115, 269 (Al2SiO5 andalusite)."),
        make("topaz_al2sio4f2", 62, "Pbnm", PointGroup::D2h, LatticeSystem::Orthorhombic, 'P',
             orthorhombic(4.6499, 8.7968, 8.3909, 4, 3.534),
             "Ribbe & Gibbs (1971) Am. Mineralogist 56, 24 (Al2SiO4(F,OH)2 topaz)."),
        make("beryl_be3al2si6o18", 192, "P6/mcc", PointGroup::D6h, LatticeSystem::Hexagonal, 'P',
             hexagonal(9.215, 9.192, 2, 2.71),
             "Morosin (1972) Acta Crystallogr. B 28, 1899 (Be3Al2(SiO3)6 beryl/emerald/aquamarine)."),

        // ----- Carbonates -----
        make("dolomite_camg_co3_2", 148, "R-3", PointGroup::C3i, LatticeSystem::Hexagonal, 'R',
             hexagonal(4.808, 16.022, 3, 2.876),
             "Reeder & Wenk (1983) Am. Mineralogist 68, 1175 (CaMg(CO3)2 dolomite)."),
        make("magnesite_mgco3", 167, "R-3c", PointGroup::D3d, LatticeSystem::Hexagonal, 'R',
             hexagonal(4.6328, 15.0129, 6, 3.009),
             "Markgraf & Reeder (1985) Am. Mineralogist 70, 590 (MgCO3 magnesite)."),
        make("siderite_feco3", 167, "R-3c", PointGroup::D3d, LatticeSystem::Hexagonal, 'R',
             hexagonal(4.6916, 15.379, 6, 3.96),
             "Effenberger et al. (1981) Z. Kristallogr. 156, 233 (FeCO3 siderite)."),

        // ----- Sulfides -----
        make("chalcopyrite_cufes2", 122, "I-42d", PointGroup::D2d, LatticeSystem::Tetragonal, 'I',
             tetragonal(5.289, 10.423, 4, 4.20),
             "Hall & Stewart (1973) Acta Crystallogr. B 29, 579 (CuFeS2 chalcopyrite)."),
        make("molybdenite_2H_mos2", 194, "P6_3/mmc", PointGroup::D6h, LatticeSystem::Hexagonal, 'P',
             hexagonal(3.160, 12.295, 2, 5.06),
             "Dickinson & Pauling (1923) JACS 45, 1466 (MoS2 2H polytype)."),
        make("stibnite_sb2s3", 62, "Pbnm", PointGroup::D2h, LatticeSystem::Orthorhombic, 'P',
             orthorhombic(11.229, 11.310, 3.839, 4, 4.63),
             "Hofmann (1933) Z. Kristallogr. 86, 225 (Sb2S3 stibnite)."),
        make("cinnabar_hgs", 152, "P3_121", PointGroup::D3, LatticeSystem::Hexagonal, 'P',
             hexagonal(4.149, 9.495, 3, 8.176),
             "Aurivillius (1950) Acta Chem. Scand. 4, 1413 (HgS cinnabar)."),

        // ----- Iron + chromium + manganese oxides -----
        make("hematite_alpha_fe2o3", 167, "R-3c", PointGroup::D3d, LatticeSystem::Hexagonal, 'R',
             hexagonal(5.0356, 13.7489, 6, 5.27),
             "Blake et al. (1966) Am. Mineralogist 51, 123 (alpha-Fe2O3 hematite, corundum-type)."),
        make("magnetite_fe3o4", 227, "Fd-3m", PointGroup::Oh, LatticeSystem::Cubic, 'F',
             cubic(8.3963, 8, 5.197),
             "Fleet (1981) Acta Crystallogr. B 37, 917 (Fe3O4 magnetite, spinel-type)."),
        make("ilmenite_fetio3", 148, "R-3", PointGroup::C3i, LatticeSystem::Hexagonal, 'R',
             hexagonal(5.088, 14.085, 6, 4.78),
             "Wechsler & Prewitt (1984) Am. Mineralogist 69, 176 (FeTiO3 ilmenite)."),
        make("chromite_fecr2o4", 227, "Fd-3m", PointGroup::Oh, LatticeSystem::Cubic, 'F',
             cubic(8.378, 8, 4.795),
             "Lenaz & Princivalle (2005) Phys. Chem. Minerals 32, 31 (FeCr2O4 spinel)."),
        make("periclase_mgo", 225, "Fm-3m", PointGroup::Oh, LatticeSystem::Cubic, 'F',
             cubic(4.213, 4, 3.581),
             "Wyckoff (1963) vol. 1 p.86 (MgO rocksalt-type)."),
        make("cuprite_cu2o", 224, "Pn-3m", PointGroup::Oh, LatticeSystem::Cubic, 'P',
             cubic(4.2696, 2, 6.10),
             "Restori & Schwarzenbach (1986) Acta Crystallogr. B 42, 201 (Cu2O cuprite)."),

        // ----- Sulfates + phosphates -----
        make("barite_baso4", 62, "Pnma", PointGroup::D2h, LatticeSystem::Orthorhombic, 'P',
             orthorhombic(8.884, 5.456, 7.157, 4, 4.480),
             "Miyake et al. (1978) Z. Kristallogr. 146, 169 (BaSO4 barite)."),
        make("anhydrite_caso4", 63, "Cmcm", PointGroup::D2h, LatticeSystem::Orthorhombic, 'C',
             orthorhombic(7.006, 6.998, 6.245, 4, 2.96),
             "Hawthorne & Ferguson (1975) Can. Mineralogist 13, 289 (CaSO4 anhydrite)."),
        make("gypsum_caso4_2h2o", 15, "C2/c", PointGroup::C2h, LatticeSystem::Monoclinic, 'C',
             (5.679, 15.202, 6.522, 90.0, 118.43, 90.0, 4, 2.31),
             "Cole & Lancucki (1974) Acta Crystallogr. B 30, 921 (CaSO4.2H2O gypsum)."),
        make("apatite_ca5_po4_3_oh", 176, "P6_3/m", PointGroup::C6h, LatticeSystem::Hexagonal, 'P',
             hexagonal(9.4254, 6.8814, 2, 3.16),
             "Hughes et al. (1989) Am. Mineralogist 74, 870 (Ca5(PO4)3(OH) apatite)."),

        // ----- 2D materials + topological insulators -----
        make("graphite_hexagonal_2H", 194, "P6_3/mmc", PointGroup::D6h, LatticeSystem::Hexagonal, 'P',
             hexagonal(2.4612, 6.7079, 4, 2.267),
             "Lipson & Stokes (1942) Proc. R. Soc. A 181, 101 (carbon graphite)."),
        make("bismuth_telluride_bi2te3", 166, "R-3m", PointGroup::D3d, LatticeSystem::Hexagonal, 'R',
             hexagonal(4.3835, 30.487, 3, 7.86),
             "Feutelais et al. (1993) Mater. Res. Bull. 28, 591 (Bi2Te3 topological insulator)."),
        make("bismuth_selenide_bi2se3", 166, "R-3m", PointGroup::D3d, LatticeSystem::Hexagonal, 'R',
             hexagonal(4.143, 28.636, 3, 6.82),
             "Nakajima (1963) J. Phys. Chem. Solids 24, 479 (Bi2Se3 topological insulator)."),

        // ----- Halide perovskites + chalcogenide superconductors -----
        make("cspbbr3_orthorhombic", 62, "Pnma", PointGroup::D2h, LatticeSystem::Orthorhombic, 'P',
             orthorhombic(8.207, 11.755, 8.244, 4, 4.844),
             "Stoumpos et al. (2013) Cryst. Growth Des. 13, 2722 (CsPbBr3 halide perovskite)."),
        make("ybco_orthorhombic", 47, "Pmmm", PointGroup::D2h, LatticeSystem::Orthorhombic, 'P',
             orthorhombic(3.823, 3.886, 11.681, 1, 6.38),
             "Beno et al. (1987) Appl. Phys. Lett. 51, 57 (YBa2Cu3O7-x high-Tc superconductor)."),
        make("magnesium_diboride_mgb2", 191, "P6/mmm", PointGroup::D6h, LatticeSystem::Hexagonal, 'P',
             hexagonal(3.086, 3.524, 1, 2.57),
             "Jones & Marsh (1954) JACS 76, 1434 (MgB2 superconductor, ALB2-type)."),

        // ----- Additional native elements -----
        make("graphite_3R_rhombohedral", 166, "R-3m", PointGroup::D3d, LatticeSystem::Hexagonal, 'R',
             hexagonal(2.456, 10.044, 6, 2.265),
             "Lipson & Stokes (1942) Proc. R. Soc. A 181, 101 (rhombohedral 3R-graphite polytype)."),
        make("sulfur_alpha_s8", 70, "Fddd", PointGroup::D2h, LatticeSystem::Orthorhombic, 'F',
             orthorhombic(10.4646, 12.8660, 24.4860, 16, 2.07),
             "Coppens (1977) Acta Crystallogr. B 33, 2275 (alpha-S8 native sulfur)."),
        make("diamond_lonsdaleite", 194, "P6_3/mmc", PointGroup::D6h, LatticeSystem::Hexagonal, 'P',
             hexagonal(2.51, 4.12, 4, 3.51),
             "Frondel & Marvin (1967) Nature 214, 587 (hexagonal-diamond polymorph)."),
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

/// Look up structures by exact `name` (case-sensitive). Returns `None` if
/// not in the registry. Convenience for callers that already know the key.
pub fn structure_by_name(name: &str) -> Option<CrystalStructureInfo> {
    known_crystal_structures().into_iter().find(|s| s.name == name)
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

    #[test]
    fn known_structures_has_80_plus_entries() {
        let n = known_crystal_structures().len();
        assert!(
            n >= 80,
            "Coverage regression: known_crystal_structures must list at least 80 \
             entries (currently {}). Adding new structures is encouraged; \
             removing any requires a rationale.",
            n
        );
    }

    #[test]
    fn silicate_coverage_includes_olivines_and_polymorphs() {
        let names: Vec<&str> = known_crystal_structures()
            .iter()
            .map(|s| s.name)
            .collect();
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
            assert!(names.contains(&required), "silicate {} missing", required);
        }
    }

    #[test]
    fn modern_materials_coverage() {
        let names: Vec<&str> = known_crystal_structures()
            .iter()
            .map(|s| s.name)
            .collect();
        for required in [
            "graphite_hexagonal_2H",
            "bismuth_telluride_bi2te3",
            "bismuth_selenide_bi2se3",
            "cspbbr3_orthorhombic",
            "ybco_orthorhombic",
            "magnesium_diboride_mgb2",
        ] {
            assert!(
                names.contains(&required),
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
            assert!(!s.primary_reference.is_empty(), "{} missing reference", s.name);
        }
    }

    #[test]
    fn cubic_lattices_have_a_equals_b_equals_c() {
        for s in known_crystal_structures()
            .iter()
            .filter(|s| s.lattice_system == LatticeSystem::Cubic)
        {
            assert!((s.lattice_a_angstrom - s.lattice_b_angstrom).abs() < 1e-9, "{}", s.name);
            assert!((s.lattice_a_angstrom - s.lattice_c_angstrom).abs() < 1e-9, "{}", s.name);
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
