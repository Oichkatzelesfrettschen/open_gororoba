//! Drude-Lorentz metal optical models (Rakic 1998 LD + canonical
//! Au/Ag variants). All 12 constructors sourced from
//! `materials_data` codegen consts (#127 phase 5).
//!
//! Includes:
//! - gold_drude_lorentz (simpler 2-osc Au model)
//! - gold_rakic_ld     (5-osc high-fidelity Au model)
//! - silver_drude_lorentz / copper_drude_lorentz /
//!   aluminum_drude_lorentz / beryllium_drude_lorentz /
//!   chromium_drude_lorentz / nickel_drude_lorentz /
//!   palladium_drude_lorentz / platinum_drude_lorentz /
//!   titanium_drude_lorentz / tungsten_drude_lorentz
//!
//! The simple pure-Drude constructors (gold_drude / silver_drude /
//! copper_drude / aluminum_drude / beryllium_drude / chromium_drude /
//! nickel_drude / palladium_drude / platinum_drude / titanium_drude /
//! tungsten_drude) stay in optical_database.rs because they already
//! consume materials_data::*_DRUDE consts directly and have no
//! oscillators.

use super::{DrudeLorentzParams, DrudeParams, LorentzOscillator, MineralMetadata, OpticSign};

fn from_codegen(
    eps_inf: f64,
    drude: Option<[f64; 3]>,
    oscillators: &'static [[f64; 3]],
) -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: drude.map(|[omega_p_ev, gamma_ev, eps_inf]| DrudeParams {
            omega_p_ev,
            gamma_ev,
            eps_inf,
        }),
        oscillators: oscillators
            .iter()
            .map(|&[strength, omega_0_ev, gamma_ev]| LorentzOscillator {
                strength,
                omega_0_ev,
                gamma_ev,
            })
            .collect(),
        eps_inf,
        extended_drude: None,
    }
}

/// Gold (Au) with 2-oscillator interband model (lower-fidelity).
/// LSPR at ~5.9 eV in vacuum from insufficient d-band representation.
pub fn gold_drude_lorentz() -> DrudeLorentzParams {
    from_codegen(
        materials_data::GOLD_DRUDE_LORENTZ_EPS_INF,
        materials_data::GOLD_DRUDE_LORENTZ_DRUDE,
        materials_data::GOLD_DRUDE_LORENTZ_OSCILLATORS,
    )
}

/// Gold (Au) high-fidelity 5-oscillator Lorentz-Drude model from
/// Rakic et al. Appl. Opt. 37, 5271-5283 (1998), Table I.
///
/// Captures the d-band transitions giving LSPR ~2.6 eV (Frohlich
/// condition) matching experiment. Oscillator strengths are
/// pre-converted from Rakic's f_j convention to our S_j convention
/// via `S_j = f_j * omega_p^2 / omega_0j^2`; the explicit values
/// are codegen'd from lorentz_models.toml with per-oscillator
/// comments documenting the conversion.
pub fn gold_rakic_ld() -> DrudeLorentzParams {
    from_codegen(
        materials_data::GOLD_RAKIC_LD_EPS_INF,
        materials_data::GOLD_RAKIC_LD_DRUDE,
        materials_data::GOLD_RAKIC_LD_OSCILLATORS,
    )
}

/// Silver (Ag) with interband transitions.
pub fn silver_drude_lorentz() -> DrudeLorentzParams {
    from_codegen(
        materials_data::SILVER_DRUDE_LORENTZ_EPS_INF,
        materials_data::SILVER_DRUDE_LORENTZ_DRUDE,
        materials_data::SILVER_DRUDE_LORENTZ_OSCILLATORS,
    )
}

/// Copper (Cu) Rakic 1998 LD model.
pub fn copper_drude_lorentz() -> DrudeLorentzParams {
    from_codegen(
        materials_data::COPPER_DRUDE_LORENTZ_EPS_INF,
        materials_data::COPPER_DRUDE_LORENTZ_DRUDE,
        materials_data::COPPER_DRUDE_LORENTZ_OSCILLATORS,
    )
}

/// Aluminum (Al) Rakic 1998 LD model.
pub fn aluminum_drude_lorentz() -> DrudeLorentzParams {
    from_codegen(
        materials_data::ALUMINUM_DRUDE_LORENTZ_EPS_INF,
        materials_data::ALUMINUM_DRUDE_LORENTZ_DRUDE,
        materials_data::ALUMINUM_DRUDE_LORENTZ_OSCILLATORS,
    )
}

/// Beryllium (Be) Rakic 1998 LD model.
pub fn beryllium_drude_lorentz() -> DrudeLorentzParams {
    from_codegen(
        materials_data::BERYLLIUM_DRUDE_LORENTZ_EPS_INF,
        materials_data::BERYLLIUM_DRUDE_LORENTZ_DRUDE,
        materials_data::BERYLLIUM_DRUDE_LORENTZ_OSCILLATORS,
    )
}

/// Chromium (Cr) Rakic 1998 LD model.
pub fn chromium_drude_lorentz() -> DrudeLorentzParams {
    from_codegen(
        materials_data::CHROMIUM_DRUDE_LORENTZ_EPS_INF,
        materials_data::CHROMIUM_DRUDE_LORENTZ_DRUDE,
        materials_data::CHROMIUM_DRUDE_LORENTZ_OSCILLATORS,
    )
}

/// Nickel (Ni) Rakic 1998 LD model.
pub fn nickel_drude_lorentz() -> DrudeLorentzParams {
    from_codegen(
        materials_data::NICKEL_DRUDE_LORENTZ_EPS_INF,
        materials_data::NICKEL_DRUDE_LORENTZ_DRUDE,
        materials_data::NICKEL_DRUDE_LORENTZ_OSCILLATORS,
    )
}

/// Palladium (Pd) Rakic 1998 LD model.
pub fn palladium_drude_lorentz() -> DrudeLorentzParams {
    from_codegen(
        materials_data::PALLADIUM_DRUDE_LORENTZ_EPS_INF,
        materials_data::PALLADIUM_DRUDE_LORENTZ_DRUDE,
        materials_data::PALLADIUM_DRUDE_LORENTZ_OSCILLATORS,
    )
}

/// Platinum (Pt) Rakic 1998 LD model.
pub fn platinum_drude_lorentz() -> DrudeLorentzParams {
    from_codegen(
        materials_data::PLATINUM_DRUDE_LORENTZ_EPS_INF,
        materials_data::PLATINUM_DRUDE_LORENTZ_DRUDE,
        materials_data::PLATINUM_DRUDE_LORENTZ_OSCILLATORS,
    )
}

/// Titanium (Ti) Rakic 1998 LD model.
pub fn titanium_drude_lorentz() -> DrudeLorentzParams {
    from_codegen(
        materials_data::TITANIUM_DRUDE_LORENTZ_EPS_INF,
        materials_data::TITANIUM_DRUDE_LORENTZ_DRUDE,
        materials_data::TITANIUM_DRUDE_LORENTZ_OSCILLATORS,
    )
}

/// Tungsten (W) Rakic 1998 LD model.
pub fn tungsten_drude_lorentz() -> DrudeLorentzParams {
    from_codegen(
        materials_data::TUNGSTEN_DRUDE_LORENTZ_EPS_INF,
        materials_data::TUNGSTEN_DRUDE_LORENTZ_DRUDE,
        materials_data::TUNGSTEN_DRUDE_LORENTZ_OSCILLATORS,
    )
}

// ============================================================================
// MineralMetadata accessors for elemental metals (task #140 remediation).
// One accessor per element is sufficient: the simple pure-Drude and full
// Drude-Lorentz variants share the same lattice / density / hardness.
// Mohs hardness for soft metals (Au, Ag, Cu) refers to the annealed bulk.
// References: CRC Handbook of Chemistry and Physics 102nd ed. (2021)
// elemental property tables; Rakic et al. (1998) Appl. Opt. 37, 5271 for
// optical-context references.
// ============================================================================

/// Gold (Au): FCC noble metal, ductile, the densest mainstream noble metal
/// at ambient pressure besides platinum/osmium.
pub fn gold_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "gold",
        formula: "Au",
        crystal_system: "cubic",
        space_group: "Fm-3m (225) FCC",
        n_omega: 0.183,
        n_epsilon: 0.183,
        birefringence: 0.0,
        optic_sign: OpticSign::Isotropic,
        hardness_mohs: 2.5,
        density_g_cm3: 19.30,
        color: "metallic yellow (bulk); thin films show interference colors",
        reference: "CRC Handbook 102nd ed.; Rakic et al. (1998) Appl. Opt. 37, 5271-5283; Johnson & Christy (1972) Phys. Rev. B 6, 4370.",
    }
}

/// Silver (Ag): FCC noble metal with the highest electrical and thermal
/// conductivity of any element at room temperature.
pub fn silver_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "silver",
        formula: "Ag",
        crystal_system: "cubic",
        space_group: "Fm-3m (225) FCC",
        n_omega: 0.054,
        n_epsilon: 0.054,
        birefringence: 0.0,
        optic_sign: OpticSign::Isotropic,
        hardness_mohs: 2.5,
        density_g_cm3: 10.49,
        color: "metallic white (bulk); rapidly tarnishes to black Ag2S",
        reference: "CRC Handbook 102nd ed.; Rakic et al. (1998) Appl. Opt. 37, 5271-5283; Johnson & Christy (1972) Phys. Rev. B 6, 4370.",
    }
}

/// Copper (Cu): FCC noble metal; the third-highest electrical conductivity
/// after silver and gold, with a characteristic reddish-orange color from
/// interband transitions at ~2.1 eV (Frohlich edge for surface plasmons).
pub fn copper_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "copper",
        formula: "Cu",
        crystal_system: "cubic",
        space_group: "Fm-3m (225) FCC",
        n_omega: 0.471,
        n_epsilon: 0.471,
        birefringence: 0.0,
        optic_sign: OpticSign::Isotropic,
        hardness_mohs: 3.0,
        density_g_cm3: 8.96,
        color: "reddish-orange metallic",
        reference: "CRC Handbook 102nd ed.; Rakic et al. (1998) Appl. Opt. 37, 5271-5283.",
    }
}

/// Aluminum (Al): FCC; second-most-abundant metallic element in the Earth's
/// crust. Forms a passivating Al2O3 oxide layer ~2-4 nm thick at ambient
/// conditions, complicating measurements of bulk optical constants.
pub fn aluminum_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "aluminum",
        formula: "Al",
        crystal_system: "cubic",
        space_group: "Fm-3m (225) FCC",
        n_omega: 1.373,
        n_epsilon: 1.373,
        birefringence: 0.0,
        optic_sign: OpticSign::Isotropic,
        hardness_mohs: 2.75,
        density_g_cm3: 2.70,
        color: "silvery white; passivated surface bluish-gray",
        reference: "CRC Handbook 102nd ed.; Rakic et al. (1998) Appl. Opt. 37, 5271-5283; Smith et al. (1985) Appl. Opt. 24, 2487.",
    }
}

/// Beryllium (Be): HCP; the lightest non-volatile metal. Toxic and brittle;
/// optical applications in space-mirror manufacture (e.g. JWST primary).
/// Uniaxial POSITIVE (HCP c/a ratio 1.568 < ideal 1.633).
pub fn beryllium_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "beryllium",
        formula: "Be",
        crystal_system: "hexagonal",
        space_group: "P6_3/mmc (194) HCP",
        n_omega: 2.95,
        n_epsilon: 2.95,
        birefringence: 0.0,
        optic_sign: OpticSign::Positive,
        hardness_mohs: 5.5,
        density_g_cm3: 1.85,
        color: "steel gray metallic",
        reference: "CRC Handbook 102nd ed.; Rakic et al. (1998) Appl. Opt. 37, 5271-5283.",
    }
}

/// Chromium (Cr): BCC. Hardest of the FCC/BCC pure metals (Mohs 8.5);
/// passivates with Cr2O3 to give stainless-steel corrosion resistance.
pub fn chromium_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "chromium",
        formula: "Cr",
        crystal_system: "cubic",
        space_group: "Im-3m (229) BCC",
        n_omega: 3.18,
        n_epsilon: 3.18,
        birefringence: 0.0,
        optic_sign: OpticSign::Isotropic,
        hardness_mohs: 8.5,
        density_g_cm3: 7.19,
        color: "blue-white metallic with chromogenic Cr2O3 passivation",
        reference: "CRC Handbook 102nd ed.; Rakic et al. (1998) Appl. Opt. 37, 5271-5283.",
    }
}

/// Nickel (Ni): FCC ferromagnetic transition metal. Curie point at 358 deg-C.
pub fn nickel_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "nickel",
        formula: "Ni",
        crystal_system: "cubic",
        space_group: "Fm-3m (225) FCC",
        n_omega: 1.99,
        n_epsilon: 1.99,
        birefringence: 0.0,
        optic_sign: OpticSign::Isotropic,
        hardness_mohs: 4.0,
        density_g_cm3: 8.91,
        color: "silvery-white with slight yellow tint",
        reference: "CRC Handbook 102nd ed.; Rakic et al. (1998) Appl. Opt. 37, 5271-5283.",
    }
}

/// Palladium (Pd): FCC; corrosion-resistant noble metal. Catalytic for
/// hydrogen absorption (up to 900x its volume in H2 at STP).
pub fn palladium_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "palladium",
        formula: "Pd",
        crystal_system: "cubic",
        space_group: "Fm-3m (225) FCC",
        n_omega: 1.83,
        n_epsilon: 1.83,
        birefringence: 0.0,
        optic_sign: OpticSign::Isotropic,
        hardness_mohs: 4.75,
        density_g_cm3: 12.02,
        color: "silvery-white metallic",
        reference: "CRC Handbook 102nd ed.; Rakic et al. (1998) Appl. Opt. 37, 5271-5283.",
    }
}

/// Platinum (Pt): FCC noble metal; chemically inert at all temperatures
/// below its melting point (2041 K). One of the densest elements.
pub fn platinum_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "platinum",
        formula: "Pt",
        crystal_system: "cubic",
        space_group: "Fm-3m (225) FCC",
        n_omega: 2.32,
        n_epsilon: 2.32,
        birefringence: 0.0,
        optic_sign: OpticSign::Isotropic,
        hardness_mohs: 4.0,
        density_g_cm3: 21.45,
        color: "silvery-white metallic",
        reference: "CRC Handbook 102nd ed.; Rakic et al. (1998) Appl. Opt. 37, 5271-5283.",
    }
}

/// Titanium (Ti): HCP alpha-phase below 1155 K (above which it transitions to
/// BCC beta-Ti, Im-3m). Used widely in aerospace and biomedical implants
/// (Ti-6Al-4V) due to high specific strength and biocompatibility.
pub fn titanium_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "titanium_alpha",
        formula: "Ti",
        crystal_system: "hexagonal",
        space_group: "P6_3/mmc (194) HCP (alpha-Ti)",
        n_omega: 3.20,
        n_epsilon: 3.20,
        birefringence: 0.0,
        optic_sign: OpticSign::Positive,
        hardness_mohs: 6.0,
        density_g_cm3: 4.51,
        color: "silver-white with slight TiO2 passivation iridescence",
        reference: "CRC Handbook 102nd ed.; Rakic et al. (1998) Appl. Opt. 37, 5271-5283.",
    }
}

/// Tungsten (W): BCC. Highest melting point of any pure metal (3695 K).
/// Used in incandescent filaments, X-ray targets, and rocket-nozzle liners.
pub fn tungsten_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "tungsten",
        formula: "W",
        crystal_system: "cubic",
        space_group: "Im-3m (229) BCC",
        n_omega: 3.40,
        n_epsilon: 3.40,
        birefringence: 0.0,
        optic_sign: OpticSign::Isotropic,
        hardness_mohs: 7.5,
        density_g_cm3: 19.25,
        color: "gray-white metallic; refractory and dense",
        reference: "CRC Handbook 102nd ed.; Rakic et al. (1998) Appl. Opt. 37, 5271-5283.",
    }
}
