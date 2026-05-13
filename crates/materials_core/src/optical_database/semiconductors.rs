//! Semiconductor + dielectric optical models: silicon (intrinsic),
//! amorphous silica (room-T + Casimir-experiment variant), silicon
//! nitride, germanium. All return DrudeLorentzParams with no Drude
//! response (drude: None) -- pure interband-oscillator models.
//!
//! Re-exported from optical_database via `pub use` so external paths
//! materials_core::optical_database::{silicon_optical, silica_optical,
//! silica_casimir_optical, silicon_nitride_optical, germanium_optical}
//! stay stable.
//!
//! All five constructors source their parameters from
//! `crates/materials_data/data/optical/lorentz_models.toml` via the
//! build-time codegen path -- the literal arrays no longer live in
//! Rust source (Phase 5 / task #127).

use super::{DrudeLorentzParams, DrudeParams, LorentzOscillator, MineralMetadata, OpticSign};

/// Build a `DrudeLorentzParams` from the codegen'd consts. Matches
/// the signature used by `tungstates::from_codegen` so the call
/// sites are identical across both submodules.
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

/// Silicon (intrinsic) optical model. Bandgap ~1.1 eV; E0/E1/E2 critical points.
pub fn silicon_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::SILICON_EPS_INF,
        materials_data::SILICON_DRUDE,
        materials_data::SILICON_OSCILLATORS,
    )
}

/// Silica (SiO2) optical model -- wide-bandgap dielectric; IR phonon + UV edge.
pub fn silica_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::SILICA_EPS_INF,
        materials_data::SILICA_DRUDE,
        materials_data::SILICA_OSCILLATORS,
    )
}

/// Silica (SiO2) optical model calibrated for Casimir-Lifshitz calculations.
///
/// Three-oscillator IR model from Lambrecht and Reynaud (2000) and Parsegian (2006).
/// Reproduces the known static permittivity eps_static = 3.80:
///   eps_inf=2.1 (optical, n=1.45) + sum(S_i)=1.700 (IR phonons) = 3.800.
///
/// # IR phonon assignments (Palik 1998 Table II; Parsegian 2006 Table B.2)
/// - IR1: Si-O rocking,    460 cm^{-1} = 0.057 eV, S=0.185
/// - IR2: Si-O bending,    800 cm^{-1} = 0.099 eV, S=0.115
/// - IR3: Si-O stretching, 1075 cm^{-1} = 0.133 eV, S=1.400 (dominant mode)
///
/// No explicit UV oscillator: eps_inf=2.1 already encodes the UV edge contribution.
pub fn silica_casimir_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::SILICA_CASIMIR_EPS_INF,
        materials_data::SILICA_CASIMIR_DRUDE,
        materials_data::SILICA_CASIMIR_OSCILLATORS,
    )
}

/// Silicon Nitride (Si3N4) optical model -- n ~ 2.0, single IR resonance.
pub fn silicon_nitride_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::SILICON_NITRIDE_EPS_INF,
        materials_data::SILICON_NITRIDE_DRUDE,
        materials_data::SILICON_NITRIDE_OSCILLATORS,
    )
}

/// Germanium optical model -- E0 gap + E1 critical point.
pub fn germanium_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::GERMANIUM_EPS_INF,
        materials_data::GERMANIUM_DRUDE,
        materials_data::GERMANIUM_OSCILLATORS,
    )
}

// ============================================================================
// MineralMetadata accessors (task #134 remediation; data sourced from Palik
// 1998, NIST condensed-matter handbooks, and Malitson 1965 / 1972 for the
// dielectrics). n_omega / n_epsilon refer to the sodium-D line (587.6 nm)
// for visible-transparent materials; for opaque semiconductors the columns
// hold n at 1550 nm (telecom band) where the indices are real-valued.
// ============================================================================

/// Silicon: intrinsic diamond-cubic Fd-3m semiconductor.
pub fn silicon_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "silicon",
        formula: "Si",
        crystal_system: "cubic",
        space_group: "Fd-3m (227)",
        n_omega: 3.45,
        n_epsilon: 3.45,
        birefringence: 0.0,
        optic_sign: OpticSign::Isotropic,
        hardness_mohs: 6.5,
        density_g_cm3: 2.329,
        color: "metallic gray (bulk), bluish-iridescent thin film",
        reference: "Palik (1998) Handbook of Optical Constants vol. I p.547; NIST condensed-matter handbook silicon entry.",
    }
}

/// Silica (fused / amorphous SiO2): n_omega columns are n at sodium-D.
pub fn silica_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "fused_silica",
        formula: "SiO2 (amorphous)",
        crystal_system: "amorphous",
        space_group: "n/a (no long-range order)",
        n_omega: 1.4585,
        n_epsilon: 1.4585,
        birefringence: 0.0,
        optic_sign: OpticSign::Isotropic,
        hardness_mohs: 7.0,
        density_g_cm3: 2.20,
        color: "colorless transparent",
        reference: "Palik (1998) vol. I p.749; Malitson (1965) J.Opt.Soc.Am. 55, 1205-1209 (Sellmeier).",
    }
}

/// Casimir-tuned silica: same material, same metadata as `silica_metadata`.
pub fn silica_casimir_metadata() -> MineralMetadata {
    silica_metadata()
}

/// Silicon Nitride (alpha-Si3N4 trigonal P31c; beta-Si3N4 hexagonal P6_3 is the
/// high-temperature polymorph). The optical model is fitted on amorphous LPCVD
/// Si3N4 films which are predominantly beta-precursor short-range order.
pub fn silicon_nitride_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "silicon_nitride",
        formula: "Si3N4",
        crystal_system: "trigonal (alpha) or hexagonal (beta)",
        space_group: "P31c (159) alpha; P6_3 (173) beta",
        n_omega: 1.998,
        n_epsilon: 2.045,
        birefringence: 0.047,
        optic_sign: OpticSign::Positive,
        hardness_mohs: 9.0,
        density_g_cm3: 3.17,
        color: "gray to brownish-gray (bulk); colorless thin film",
        reference: "Palik (1998) vol. II p.771; Luke et al. (2015) Opt. Lett. 40, 4823 (broadband Si3N4 dispersion).",
    }
}

/// Germanium: same Fd-3m cubic lattice as silicon but with larger ionic radius
/// (5.658 vs 5.431 angstrom lattice constant) and smaller bandgap (0.67 vs 1.12 eV).
pub fn germanium_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "germanium",
        formula: "Ge",
        crystal_system: "cubic",
        space_group: "Fd-3m (227)",
        n_omega: 4.00,
        n_epsilon: 4.00,
        birefringence: 0.0,
        optic_sign: OpticSign::Isotropic,
        hardness_mohs: 6.0,
        density_g_cm3: 5.323,
        color: "grayish-white metallic",
        reference: "Palik (1998) vol. I p.465.",
    }
}
