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
"""
content = content.replace('        // TCOs\n        "Al-doped ZnO (AZO)",', new_list)

# Add to get_material
new_get = """
        "azo" | "al-doped zno (azo)" => Some(MaterialEntry {
            name: "Al-doped ZnO",
            formula: "ZnO:Al",
            material_type: MaterialType::ConductiveOxide,
            optical: azo_optical(),
            reference: "Ma et al. (2020) & Naik (2013)",
            validity_range_ev: Some((0.5, 4.0)),
            temperature_k: Some(300.0),
            doping_info: Some("AZO typical"),
            casimir_model: CasimirModelFlag::ExtendedDrude,
            uniaxial: None,
        }),
        "ito" | "indium tin oxide (ito)" => Some(MaterialEntry {
            name: "Indium Tin Oxide",
            formula: "Sn:In2O3",
            material_type: MaterialType::ConductiveOxide,
            optical: ito_optical(),
            reference: "Gerfin et al. (1993/2016)",
            validity_range_ev: Some((0.1, 4.0)),
            temperature_k: Some(300.0),
            doping_info: Some("~4e20 cm^-3"),
            casimir_model: CasimirModelFlag::ExtendedDrude,
            uniaxial: None,
        }),
        "tin" | "titanium nitride" | "titanium nitride (tin)" => Some(MaterialEntry {
            name: "Titanium Nitride",
            formula: "TiN",
            material_type: MaterialType::Metal,
            optical: tin_optical(),
            reference: "Patsalas et al. (2015/2018)",
            validity_range_ev: Some((0.5, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Drude,
            uniaxial: None,
        }),
        "tin_low_loss" | "titanium nitride low loss" | "titanium nitride low loss (tin)" => Some(MaterialEntry {
            name: "Titanium Nitride (Low Loss)",
            formula: "TiN",
            material_type: MaterialType::Metal,
            optical: tin_low_loss_optical(),
            reference: "Naik et al. (2012/2013)",
            validity_range_ev: Some((0.5, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Drude,
            uniaxial: None,
        }),
        "zrn" | "zirconium nitride" | "zirconium nitride (zrn)" => Some(MaterialEntry {
            name: "Zirconium Nitride",
            formula: "ZrN",
            material_type: MaterialType::Metal,
            optical: zrn_optical(),
            reference: "Kinsey et al. (2015)",
            validity_range_ev: Some((0.5, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Drude,
            uniaxial: None,
        }),
        "hfn" | "hafnium nitride" | "hafnium nitride (hfn)" => Some(MaterialEntry {
            name: "Hafnium Nitride",
            formula: "HfN",
            material_type: MaterialType::Metal,
            optical: hfn_optical(),
            reference: "Braic et al. (2019)",
            validity_range_ev: Some((0.5, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Drude,
            uniaxial: None,
        }),
"""

old_azo = """        "azo" | "al-doped zno (azo)" => Some(MaterialEntry {
            name: "Al-doped ZnO",
            formula: "ZnO:Al",
            material_type: MaterialType::ConductiveOxide,
            optical: azo_optical(),
            reference: "Ma et al. (2020)",
            validity_range_ev: Some((0.5, 4.0)),
            temperature_k: Some(300.0),
            doping_info: Some("AZO typical"),
            casimir_model: CasimirModelFlag::ExtendedDrude,
            uniaxial: None,
        }),"""

content = content.replace(old_azo, new_get)
open("crates/materials_core/src/optical_database.rs", "w").write(content)
