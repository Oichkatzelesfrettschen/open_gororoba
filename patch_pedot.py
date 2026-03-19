import re

content = open("crates/materials_core/src/optical_database.rs").read()

# Add to list_materials
new_list = """
        // TCOs
        "Al-doped ZnO (AZO)",
        "Indium Tin Oxide (ITO)",
        // Nitrides
        "Titanium Nitride (TiN)",
        "Titanium Nitride Low Loss (TiN)",
        "Zirconium Nitride (ZrN)",
        "Hafnium Nitride (HfN)",
        // Disordered Conductors
        "PEDOT:PSS",
"""
content = content.replace("""
        // TCOs
        "Al-doped ZnO (AZO)",
        "Indium Tin Oxide (ITO)",
        // Nitrides
        "Titanium Nitride (TiN)",
        "Titanium Nitride Low Loss (TiN)",
        "Zirconium Nitride (ZrN)",
        "Hafnium Nitride (HfN)",
""", new_list)


# Add the model
model = """
/// PEDOT:PSS - Disordered conducting polymer (Drude-Smith model)
/// Values are representative for highly conductive grades in THz/IR.
pub fn pedot_pss_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: None,
        extended_drude: Some(ExtendedDrudeParams {
            omega_p_ev: 1.2, // ~1e21 cm^-3 effective carrier density
            scattering: ScatteringModel::DrudeSmith {
                gamma_ev: 0.15,
                backscatter_c: -0.85, // strong backscattering (localization)
            },
            eps_inf: 2.2,
        }),
        oscillators: vec![],
        eps_inf: 2.2,
    }
}
"""

content = content.replace("pub fn ito_optical", model + "\npub fn ito_optical")

# Add to get_material
new_get = """
        "pedot" | "pedot:pss" | "pedot pss" => Some(MaterialEntry {
            name: "PEDOT:PSS",
            formula: "PEDOT:PSS",
            material_type: MaterialType::Metal, // phenomenologically metallic in THz
            optical: pedot_pss_optical(),
            reference: "Typical highly conductive Drude-Smith fit",
            validity_range_ev: Some((0.001, 1.0)),
            temperature_k: Some(300.0),
            doping_info: Some("Highly conductive grade"),
            casimir_model: CasimirModelFlag::ExtendedDrude,
            uniaxial: None,
        }),
"""

content = content.replace("""        "hfn" | "hafnium nitride" | "hafnium nitride (hfn)" => Some(MaterialEntry {""", new_get + """        "hfn" | "hafnium nitride" | "hafnium nitride (hfn)" => Some(MaterialEntry {""")

content = content.replace('            36,\n            "Database should have exactly 36 materials', '            37,\n            "Database should have exactly 37 materials')

open("crates/materials_core/src/optical_database.rs", "w").write(content)
