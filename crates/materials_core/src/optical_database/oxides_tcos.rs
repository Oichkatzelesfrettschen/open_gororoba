//! Oxide + titanate + transparent-conductor optical models, all
//! sourced from the `materials_data` codegen registry
//! (#127 phase 5).
//!
//! Materials:
//! - Phase 1 gap materials: alumina (Al2O3), tourmaline, diamond,
//!   quartz, TiO2 rutile.
//! - Phase 3 titanates: TiO, SrTiO3 (undoped + electron-doped),
//!   LaTiO3 Mott insulator.
//! - Phase 3 TCOs: ITO (pure Drude), AZO (Drude + UV oscillator),
//!   doped Si (Drude + intrinsic-Si oscillators).
//!
//! Re-exported from optical_database via `pub use`.

use super::{DrudeLorentzParams, DrudeParams, LorentzOscillator};

/// Same signature as the helpers in semiconductors.rs / tungstates.rs;
/// uniform call sites across all three family submodules.
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

/// Alumina (Al2O3 / Sapphire) optical model (Palik 1998).
pub fn alumina_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::ALUMINA_EPS_INF,
        materials_data::ALUMINA_DRUDE,
        materials_data::ALUMINA_OSCILLATORS,
    )
}

/// Tourmaline (Pink) optical model -- simple R3m representation.
pub fn tourmaline_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::TOURMALINE_EPS_INF,
        materials_data::TOURMALINE_DRUDE,
        materials_data::TOURMALINE_OSCILLATORS,
    )
}

/// Diamond (C) optical model (Palik 1998).
pub fn diamond_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::DIAMOND_EPS_INF,
        materials_data::DIAMOND_DRUDE,
        materials_data::DIAMOND_OSCILLATORS,
    )
}

/// Crystalline Quartz optical model (Palik 1998). Distinct from
/// amorphous silica -- sharper phonon modes.
pub fn quartz_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::QUARTZ_EPS_INF,
        materials_data::QUARTZ_DRUDE,
        materials_data::QUARTZ_OSCILLATORS,
    )
}

/// Titanium Dioxide rutile (TiO2) optical model (Palik 1998, DeVore 1951).
pub fn tio2_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::TIO2_EPS_INF,
        materials_data::TIO2_DRUDE,
        materials_data::TIO2_OSCILLATORS,
    )
}

/// Titanium Monoxide (TiO) "bad metal" optical model (Barman & Sarma 1995).
pub fn tio_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::TIO_EPS_INF,
        materials_data::TIO_DRUDE,
        materials_data::TIO_OSCILLATORS,
    )
}

/// Strontium Titanate (SrTiO3) undoped optical model (Servoin 1980).
/// Incipient ferroelectric with giant soft-mode oscillator strength.
pub fn srtio3_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::SRTIO3_EPS_INF,
        materials_data::SRTIO3_DRUDE,
        materials_data::SRTIO3_OSCILLATORS,
    )
}

/// Doped SrTiO3 (SrTiO3:n) optical model (van Mechelen 2008).
/// Metallic via electron doping: phonon modes plus Drude tail.
pub fn srtio3_doped_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::SRTIO3_DOPED_EPS_INF,
        materials_data::SRTIO3_DOPED_DRUDE,
        materials_data::SRTIO3_DOPED_OSCILLATORS,
    )
}

/// Lanthanum Titanate (LaTiO3) Mott insulator model (Okimoto 1995).
pub fn latio3_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::LATIO3_EPS_INF,
        materials_data::LATIO3_DRUDE,
        materials_data::LATIO3_OSCILLATORS,
    )
}

/// Indium Tin Oxide (ITO) -- typical degenerate TCO. Pure Drude.
pub fn ito_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::ITO_EPS_INF,
        materials_data::ITO_DRUDE,
        materials_data::ITO_OSCILLATORS,
    )
}

/// Aluminum-doped Zinc Oxide (AZO) transparent conductor.
/// Metallic in IR, transparent in visible. Crossover near 1 eV.
pub fn azo_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::AZO_EPS_INF,
        materials_data::AZO_DRUDE,
        materials_data::AZO_OSCILLATORS,
    )
}

/// Doped Silicon (Si:n, ~1e18 cm-3) with THz Drude tail.
pub fn doped_silicon_optical() -> DrudeLorentzParams {
    from_codegen(
        materials_data::DOPED_SILICON_EPS_INF,
        materials_data::DOPED_SILICON_DRUDE,
        materials_data::DOPED_SILICON_OSCILLATORS,
    )
}
