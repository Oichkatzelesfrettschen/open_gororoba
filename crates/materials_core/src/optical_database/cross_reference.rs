//! Cross-reference helper from a crystal-structure name (as it appears in
//! `materials_data::CRYSTAL_STRUCTURE_TABLE`) to the corresponding
//! optical `MineralMetadata` accessor where one exists.
//!
//! WHY: The two registries (`crystal_structures.toml` in materials_data;
//! `*_metadata()` accessors in materials_core's optical_database submodules)
//! are populated independently, but many entries are physically the same
//! material. This helper lets callers go from "I have a structure name
//! from the codegen table" to "what optical metadata, if any, do we have
//! for this material?" without scattering match arms across the codebase.
//!
//! Mapping policy:
//! - A structure name maps to a metadata accessor when the two represent
//!   the SAME unique material (same composition, same phase).
//! - Polymorphs / doped variants / supergroup species map to the species-
//!   level metadata when one is available.
//! - When no metadata exists yet, the helper returns None. Adding a new
//!   pairing is a single-line edit in this file.

use super::{
    MineralMetadata, alumina_metadata, aluminum_metadata, azo_metadata, beryllium_metadata,
    cawo4_metadata, chromium_metadata, copper_metadata, cs_wo3_metadata, diamond_metadata,
    doped_silicon_metadata, dravite_metadata, elbaite_metadata, foitite_metadata,
    germanium_metadata, gold_metadata, ito_metadata, latio3_metadata, liddicoatite_metadata,
    nickel_metadata, palladium_metadata, pbwo4_metadata, pedot_pss_metadata, platinum_metadata,
    povondraite_metadata, quartz_metadata, rossmanite_metadata, schorl_metadata,
    silica_casimir_metadata, silica_metadata, silicon_metadata, silicon_nitride_metadata,
    silver_metadata, srtio3_doped_metadata, srtio3_metadata, tio2_metadata, tio_metadata,
    titanium_metadata, tungsten_metadata, uvite_metadata, wo3_metadata, wo3_x_metadata,
};

/// Look up MineralMetadata for a crystal-structure name from
/// `materials_data::CRYSTAL_STRUCTURE_TABLE`. Returns `None` if no optical
/// metadata exists yet for that material.
///
/// # Examples
///
/// ```
/// use materials_core::optical_database::metadata_for_structure;
/// let au = metadata_for_structure("Au_gold").unwrap();
/// assert_eq!(au.species_name, "gold");
/// assert!(au.density_g_cm3 > 19.0);
///
/// assert!(metadata_for_structure("not_a_real_structure").is_none());
/// ```
pub fn metadata_for_structure(name: &str) -> Option<MineralMetadata> {
    match name {
        // ---------- Elemental metals (cubic FCC + BCC + HCP) ----------
        "Au_gold" => Some(gold_metadata()),
        "Ag_silver" => Some(silver_metadata()),
        "Cu_copper" => Some(copper_metadata()),
        "Al_aluminum" => Some(aluminum_metadata()),
        "Pt_platinum" => Some(platinum_metadata()),
        "Pd_palladium" => Some(palladium_metadata()),
        "Ni_nickel" => Some(nickel_metadata()),
        "Cr_chromium" => Some(chromium_metadata()),
        "W_tungsten" => Some(tungsten_metadata()),
        "Be_beryllium" => Some(beryllium_metadata()),
        "Ti_alpha" => Some(titanium_metadata()),

        // ---------- Foundational semiconductors + dielectrics ----------
        "Si_silicon" => Some(silicon_metadata()),
        "Ge_germanium" => Some(germanium_metadata()),
        "Diamond" => Some(diamond_metadata()),
        // alpha-quartz crystal -> SiO2 silica optical metadata
        "alpha_quartz" => Some(quartz_metadata()),
        "silicon_nitride_alpha" | "silicon_nitride_beta" => Some(silicon_nitride_metadata()),

        // ---------- Oxides + TCOs ----------
        "alumina_sapphire" => Some(alumina_metadata()),
        "rutile_tio2" => Some(tio2_metadata()),
        "anatase_tio2" => Some(tio2_metadata()),
        "brookite_tio2" => Some(tio2_metadata()),
        "indium_tin_oxide_bixbyite" => Some(ito_metadata()),
        // Wurtzite ZnO + AZO share the same lattice; AZO is doped variant.
        "wurtzite_zno" => Some(azo_metadata()),
        "perovskite_srtio3" => Some(srtio3_metadata()),
        "perovskite_batio3_tetragonal" | "barium_titanate_cubic" => Some(srtio3_doped_metadata()),
        "lanthanum_titanate" => Some(latio3_metadata()),

        // ---------- Tungstates ----------
        "tungsten_trioxide_gamma" => Some(wo3_metadata()),
        "cesium_tungsten_bronze" | "tungsten_bronze_na_x_wo3_cubic" => Some(cs_wo3_metadata()),
        "scheelite_cawo4" => Some(cawo4_metadata()),
        "stolzite_pbwo4" => Some(pbwo4_metadata()),

        // ---------- Tourmaline supergroup (8 species, each its own entry) ----------
        "schorl_tourmaline" => Some(schorl_metadata()),
        "dravite_tourmaline" => Some(dravite_metadata()),
        "elbaite_tourmaline" => Some(elbaite_metadata()),
        "uvite_tourmaline" => Some(uvite_metadata()),
        "liddicoatite_tourmaline" => Some(liddicoatite_metadata()),
        "rossmanite_tourmaline" => Some(rossmanite_metadata()),
        "foitite_tourmaline" => Some(foitite_metadata()),
        "povondraite_tourmaline" => Some(povondraite_metadata()),

        // ---------- Polymorphs / variants that share metadata with parents ----------
        "doped_silicon" => Some(doped_silicon_metadata()),
        "pedot_pss_conducting_polymer" => Some(pedot_pss_metadata()),
        "tungsten_trioxide_substoichiometric" => Some(wo3_x_metadata()),
        "fused_silica" => Some(silica_metadata()),
        // Casimir-experiment-tuned SiO2 model is a variant of silica;
        // still the same crystal lattice.
        "silica_casimir" => Some(silica_casimir_metadata()),
        // Titanium monoxide TiO is structurally distinct from rutile TiO2.
        "titanium_monoxide" => Some(tio_metadata()),

        _ => None,
    }
}

/// Returns the list of all crystal-structure names that have a paired
/// MineralMetadata accessor. Useful for auditing the cross-reference
/// coverage in tests.
pub fn structures_with_metadata() -> Vec<&'static str> {
    vec![
        // Metals
        "Au_gold", "Ag_silver", "Cu_copper", "Al_aluminum", "Pt_platinum",
        "Pd_palladium", "Ni_nickel", "Cr_chromium", "W_tungsten", "Be_beryllium",
        "Ti_alpha",
        // Semis + dielectrics
        "Si_silicon", "Ge_germanium", "Diamond", "alpha_quartz",
        "silicon_nitride_alpha", "silicon_nitride_beta",
        // Oxides + TCOs
        "alumina_sapphire", "rutile_tio2", "anatase_tio2", "brookite_tio2",
        "indium_tin_oxide_bixbyite", "wurtzite_zno", "perovskite_srtio3",
        "perovskite_batio3_tetragonal", "barium_titanate_cubic",
        "lanthanum_titanate",
        // Tungstates
        "tungsten_trioxide_gamma", "cesium_tungsten_bronze",
        "tungsten_bronze_na_x_wo3_cubic", "scheelite_cawo4", "stolzite_pbwo4",
        // Tourmaline supergroup
        "schorl_tourmaline", "dravite_tourmaline", "elbaite_tourmaline",
        "uvite_tourmaline", "liddicoatite_tourmaline", "rossmanite_tourmaline",
        "foitite_tourmaline", "povondraite_tourmaline",
        // Variants + polymorphs
        "doped_silicon", "pedot_pss_conducting_polymer",
        "tungsten_trioxide_substoichiometric", "fused_silica", "silica_casimir",
        "titanium_monoxide",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gold_structure_resolves_to_gold_metadata() {
        let m = metadata_for_structure("Au_gold").expect("Au_gold has metadata");
        assert_eq!(m.species_name, "gold");
        assert!(m.density_g_cm3 > 19.0);
    }

    #[test]
    fn unknown_name_returns_none() {
        assert!(metadata_for_structure("not_a_real_structure").is_none());
        assert!(metadata_for_structure("").is_none());
    }

    #[test]
    fn all_eight_tourmaline_species_resolve() {
        for name in [
            "schorl_tourmaline",
            "dravite_tourmaline",
            "elbaite_tourmaline",
            "uvite_tourmaline",
            "liddicoatite_tourmaline",
            "rossmanite_tourmaline",
            "foitite_tourmaline",
            "povondraite_tourmaline",
        ] {
            let m = metadata_for_structure(name)
                .unwrap_or_else(|| panic!("{} should resolve to metadata", name));
            assert_eq!(m.crystal_system, "trigonal");
            assert!(m.space_group.starts_with("R3m"));
        }
    }

    #[test]
    fn structures_with_metadata_list_is_complete() {
        // Every name in structures_with_metadata() must resolve to Some.
        // This pins the table self-consistency.
        for name in structures_with_metadata() {
            assert!(
                metadata_for_structure(name).is_some(),
                "{} listed in structures_with_metadata but resolves to None",
                name
            );
        }
    }

    #[test]
    fn cross_reference_covers_at_least_46_structures() {
        // Coverage regression: the cross-reference should grow with the
        // metadata catalog. Currently 46 mappings (34 unique metadata
        // accessors x ~1.4 average structure aliases per accessor).
        // Adding new pairings is encouraged.
        let n = structures_with_metadata().len();
        assert!(
            n >= 46,
            "Cross-reference coverage regression: {} mappings (expected >= 46). \
             Adding new structure -> metadata pairings is encouraged.",
            n
        );
    }
}
