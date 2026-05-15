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

use super::{DrudeLorentzParams, DrudeParams, LorentzOscillator, MineralMetadata, OpticSign};

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

// ============================================================================
// MineralMetadata accessors (task #140 remediation): high-priority materials
// from the 2026-05-13 metadata-coverage audit. Indices reported at the
// sodium-D line (587.6 nm) where the material is transparent; references
// favor Palik 1998 and primary Sellmeier-fit papers.
// ============================================================================

/// Alumina (Al2O3 / sapphire / corundum): trigonal R-3c. Hardness 9 makes
/// it the second-hardest natural material after diamond. Uniaxial NEGATIVE.
pub fn alumina_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "alumina_sapphire",
        formula: "Al2O3",
        crystal_system: "trigonal",
        space_group: "R-3c (167) corundum",
        n_omega: 1.768,
        n_epsilon: 1.760,
        birefringence: 0.008,
        optic_sign: OpticSign::Negative,
        hardness_mohs: 9.0,
        density_g_cm3: 3.987,
        color: "colorless when pure; chromophore-dependent (ruby: Cr; sapphire: Fe/Ti)",
        reference: "Palik (1998) vol. I p.761; Malitson & Dodge (1972) J.Opt.Soc.Am. 62, 1405.",
    }
}

/// Diamond (C): cubic Fd-3m. The hardest naturally-occurring substance
/// (Mohs 10). Highest known optical refractive index for a gemstone
/// (n=2.418 at sodium-D), giving rise to its characteristic dispersion.
pub fn diamond_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "diamond",
        formula: "C",
        crystal_system: "cubic",
        space_group: "Fd-3m (227)",
        n_omega: 2.4175,
        n_epsilon: 2.4175,
        birefringence: 0.0,
        optic_sign: OpticSign::Isotropic,
        hardness_mohs: 10.0,
        density_g_cm3: 3.52,
        color: "colorless when pure; defect-dependent (N, B, vacancy clusters)",
        reference: "Palik (1998) vol. II p.665; Peter (1923) Z. Phys. 15, 358 (original dispersion fit).",
    }
}

/// Alpha-quartz (SiO2): trigonal P3_121 (right-handed) or P3_221 (left-handed).
/// Optically active: rotates plane-polarized light by ~21.7 deg/mm at 589 nm
/// for cuts parallel to the c-axis. Uniaxial POSITIVE.
pub fn quartz_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "alpha_quartz",
        formula: "SiO2",
        crystal_system: "trigonal",
        space_group: "P3_121 (152) right-handed; P3_221 (154) left-handed",
        n_omega: 1.5443,
        n_epsilon: 1.5534,
        birefringence: 0.0091,
        optic_sign: OpticSign::Positive,
        hardness_mohs: 7.0,
        density_g_cm3: 2.65,
        color: "colorless when pure; many varietal colors (amethyst, citrine, smoky, rose)",
        reference: "Ghosh (1999) Opt. Commun. 163, 95 (Sellmeier dispersion); Palik (1985) vol. I p.719.",
    }
}

/// Rutile TiO2: tetragonal P4_2/mnm. Highest birefringence among common
/// minerals (delta_n = 0.287 at sodium-D). Uniaxial POSITIVE.
/// For polymorph-specific metadata use `anatase_metadata()` (I4_1/amd
/// #141) or `brookite_metadata()` (Pbca #61).
pub fn tio2_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "rutile_tio2",
        formula: "TiO2",
        crystal_system: "tetragonal",
        space_group: "P4_2/mnm (136) rutile",
        n_omega: 2.616,
        n_epsilon: 2.903,
        birefringence: 0.287,
        optic_sign: OpticSign::Positive,
        hardness_mohs: 6.0,
        density_g_cm3: 4.23,
        color: "reddish-brown to black; transparent thin films near-colorless",
        reference: "DeVore (1951) J.Opt.Soc.Am. 41, 416; Palik (1985) vol. I p.795; Dorenwendt (1971).",
    }
}

/// Anatase TiO2: tetragonal I4_1/amd (#141). Lower density + lower
/// indices than rutile (n_o = 2.49, n_e = 2.56). Uniaxial NEGATIVE.
/// Metastable at ambient pressure; transforms to rutile near 700 deg-C.
/// Photocatalytic UV applications + dye-sensitized solar cells.
pub fn anatase_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "anatase_tio2",
        formula: "TiO2",
        crystal_system: "tetragonal",
        space_group: "I4_1/amd (141) anatase",
        n_omega: 2.561,
        n_epsilon: 2.488,
        birefringence: 0.073,
        optic_sign: OpticSign::Negative,
        hardness_mohs: 6.0,
        density_g_cm3: 3.895,
        color: "blue, brown, black to colorless thin films",
        reference: "Burdett et al. (1987) J.Am.Chem.Soc. 109, 3639; Hosaka et al. (1997) J.Phys.Soc.Jpn. 66, 877 (n at sodium-D).",
    }
}

/// Brookite TiO2: orthorhombic Pbca (#61). Biaxial. Rarest of the three
/// natural TiO2 polymorphs. Metastable at ambient; transforms to rutile
/// near 750 deg-C.
pub fn brookite_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "brookite_tio2",
        formula: "TiO2",
        crystal_system: "orthorhombic",
        space_group: "Pbca (61) brookite",
        n_omega: 2.583,
        n_epsilon: 2.700,
        birefringence: 0.117,
        optic_sign: OpticSign::Biaxial,
        hardness_mohs: 5.75,
        density_g_cm3: 4.123,
        color: "brown to dark brown",
        reference: "Meagher & Lager (1979) Can. Mineralogist 17 p.77; Pauling & Sturdivant (1928) Z. Kristallogr. 68, 239.",
    }
}

/// Lithium niobate LiNbO3: trigonal R3c (#161). Principal ferroelectric +
/// nonlinear-optical crystal. Uniaxial NEGATIVE. Curie point 1483 K;
/// pyroelectric + electro-optic + photorefractive.
pub fn linbo3_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "lithium_niobate",
        formula: "LiNbO3",
        crystal_system: "trigonal",
        space_group: "R3c (161)",
        n_omega: 2.286,
        n_epsilon: 2.203,
        birefringence: 0.083,
        optic_sign: OpticSign::Negative,
        hardness_mohs: 5.5,
        density_g_cm3: 4.65,
        color: "colorless when pure; varies with doping (Mg, Fe, Cr, etc.)",
        reference: "Abrahams et al. (1973) J. Chem. Phys. 59, 4012; Zelmon et al. (1997) JOSA B 14, 3319 (Sellmeier dispersion).",
    }
}

/// Lithium tantalate LiTaO3: trigonal R3c (#161), isostructural with
/// LiNbO3. Lower Curie point (878 K) + lower nonlinear coefficients but
/// higher optical-damage threshold. SAW + electro-optic applications.
pub fn litao3_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "lithium_tantalate",
        formula: "LiTaO3",
        crystal_system: "trigonal",
        space_group: "R3c (161)",
        n_omega: 2.183,
        n_epsilon: 2.188,
        birefringence: 0.005,
        optic_sign: OpticSign::Positive,
        hardness_mohs: 5.5,
        density_g_cm3: 7.46,
        color: "colorless when pure",
        reference: "Abrahams & Reddy (1965) J. Chem. Phys. 43, 2533; Bruner et al. (2003) JOSA B 20, 1893.",
    }
}

/// Indium Tin Oxide (In2O3:Sn ~10 wt% SnO2): cubic Ia-3 bixbyite-type with
/// Sn-doping providing free carriers. The dominant transparent conductor used
/// in displays and photovoltaics; Drude-tail dominated optical response.
pub fn ito_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "indium_tin_oxide",
        formula: "In2O3:Sn",
        crystal_system: "cubic",
        space_group: "Ia-3 (206) bixbyite",
        n_omega: 1.95,
        n_epsilon: 1.95,
        birefringence: 0.0,
        optic_sign: OpticSign::Isotropic,
        hardness_mohs: 6.0,
        density_g_cm3: 7.14,
        color: "yellowish-pale-green thin film; transparent in vis, reflective in IR",
        reference: "Hamberg & Granqvist (1986) J.Appl.Phys. 60, R123-R160 (canonical ITO review).",
    }
}

/// Tourmaline default metadata: returns elbaite (the most-common gemstone
/// tourmaline). Provided here for symmetry with the legacy
/// `tourmaline_optical()` constructor. New code should call the
/// species-specific metadata accessors in the `tourmaline` submodule.
pub fn tourmaline_metadata() -> MineralMetadata {
    super::tourmaline::default_metadata()
}

/// Titanium monoxide (TiO): cubic NaCl-type rock-salt structure -- DISTINCT
/// from rutile TiO2. Substoichiometric "bad metal" with up to 15% vacancies
/// on both Ti and O sublattices at room temperature.
pub fn tio_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "titanium_monoxide",
        formula: "TiO",
        crystal_system: "cubic",
        space_group: "Fm-3m (225) NaCl rock-salt",
        n_omega: 2.10,
        n_epsilon: 2.10,
        birefringence: 0.0,
        optic_sign: OpticSign::Isotropic,
        hardness_mohs: 5.0,
        density_g_cm3: 4.93,
        color: "metallic golden-bronze (bulk); thin films opaque",
        reference: "Barman & Sarma (1995) Phys. Rev. B 51, 4007; Wahila et al. (2019) on TiO defect chemistry.",
    }
}

/// Strontium titanate SrTiO3: cubic Pm-3m perovskite at room temperature
/// (becomes tetragonal I4/mcm below 105 K via antiferrodistortive transition,
/// not modeled in the optical fit).
pub fn srtio3_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "strontium_titanate",
        formula: "SrTiO3",
        crystal_system: "cubic",
        space_group: "Pm-3m (221) cubic perovskite (RT)",
        n_omega: 2.412,
        n_epsilon: 2.412,
        birefringence: 0.0,
        optic_sign: OpticSign::Isotropic,
        hardness_mohs: 6.0,
        density_g_cm3: 5.12,
        color: "colorless when pure; commonly violet from Cr doping in synthetic crystals",
        reference: "Servoin et al. (1980) Phys. Rev. B 22, 5501; van Benthem (2001) J.Appl.Phys. 90, 6156.",
    }
}

/// Doped SrTiO3 -- same Pm-3m lattice as undoped SrTiO3 with Nb/La donors
/// adding carriers. Metadata delegates to the parent oxide.
pub fn srtio3_doped_metadata() -> MineralMetadata {
    let parent = srtio3_metadata();
    MineralMetadata {
        species_name: "strontium_titanate_doped",
        formula: "SrTiO3:Nb or SrTiO3-x",
        ..parent
    }
}

/// Lanthanum titanate LaTiO3: orthorhombic GdFeO3-type distorted perovskite.
/// Mott insulator at room temperature; loses Mott gap below ~150 K to become
/// a paramagnetic metal.
pub fn latio3_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "lanthanum_titanate",
        formula: "LaTiO3",
        crystal_system: "orthorhombic",
        space_group: "Pnma (62) GdFeO3-type perovskite",
        n_omega: 2.20,
        n_epsilon: 2.25,
        birefringence: 0.05,
        optic_sign: OpticSign::Biaxial,
        hardness_mohs: 5.5,
        density_g_cm3: 6.39,
        color: "black metallic single crystal",
        reference: "Okimoto et al. (1995) Phys. Rev. B 51, 9581; Hays et al. (1999) Phys. Rev. B 60, 10367.",
    }
}

/// Aluminum-doped ZnO (AZO): hexagonal wurtzite ZnO with substitutional Al on
/// the Zn site (typically 2 at% Al for max conductivity). Same wurtzite
/// P6_3mc lattice as undoped ZnO.
pub fn azo_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "aluminum_doped_zinc_oxide",
        formula: "ZnO:Al",
        crystal_system: "hexagonal",
        space_group: "P6_3mc (186) wurtzite",
        n_omega: 2.008,
        n_epsilon: 2.029,
        birefringence: 0.021,
        optic_sign: OpticSign::Positive,
        hardness_mohs: 4.5,
        density_g_cm3: 5.61,
        color: "colorless to pale-yellow transparent thin film",
        reference: "Sun & Kwok (1999) Appl. Phys. Lett. 75, 1605; ASTM E1003 transparent conductor characterization.",
    }
}

/// Doped silicon (n-Si or p-Si) -- same Fd-3m lattice as intrinsic silicon
/// with Drude-tail free-carrier contribution. Metadata delegates to the
/// parent semiconductor.
pub fn doped_silicon_metadata() -> MineralMetadata {
    let parent = super::semiconductors::silicon_metadata();
    MineralMetadata {
        species_name: "doped_silicon",
        formula: "Si:n or Si:p (1e17 - 1e20 cm^-3)",
        ..parent
    }
}

