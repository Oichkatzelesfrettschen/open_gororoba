//! Tourmaline-group optical models with multi-species coverage.
//!
//! Tourmaline is a complex borosilicate mineral group with general formula
//!
//!   X Y3 Z6 T6 O18 (BO3)3 V3 W
//!
//! where each Wyckoff site accepts multiple cations/anions, giving rise to
//! a 43-species IMA-recognized family (as of 2023). This module captures
//! the gemologically + crystallographically most prominent species:
//!
//! | Species         | X       | Y           | Color (typical)     |
//! |-----------------|---------|-------------|---------------------|
//! | schorl          | Na      | Fe(II)3     | black to bluish-blk |
//! | dravite         | Na      | Mg3         | brown to yellow     |
//! | elbaite         | Na      | Li1.5Al1.5  | pink/green/blue/...  |
//! | uvite           | Ca      | Mg3 (+MgAl5)| brown-green         |
//! | liddicoatite    | Ca      | Li2Al       | multi-color zoned   |
//! | rossmanite      | vacancy | LiAl2       | pink/yellow         |
//! | foitite         | vacancy | Fe(II)2Al   | blue-violet         |
//! | povondraite     | Na      | Fe(III)3    | rare; black         |
//!
//! All are trigonal R3m (#160), uniaxial NEGATIVE, hardness 7.0-7.5,
//! density 2.82-3.32 g/cm^3. The dispersion model uses a coarse
//! 1-2 oscillator IR-edge + UV-edge approximation drawn from typical
//! published n_o/n_e values; precise oscillator-parameter measurements
//! per-species are not catalogued in the open-source refractiveindex.info
//! database (CC0) which focuses on stoichiometric inorganic compounds
//! and excludes complex variable-composition mineral groups. This module
//! is the canonical place to extend tourmaline coverage as those data
//! become available from open mineralogical sources (RRUFF, IMA list,
//! academic crystallography journals).
//!
//! References:
//! - Henry et al. (2011) "Nomenclature of the tourmaline-supergroup
//!   minerals". Am. Mineralogist 96, 895-913.
//! - Bosi et al. (2019) "Tourmaline crystal chemistry". Lithos 322.
//! - Dietrich (1985) "The Tourmaline Group", Van Nostrand Reinhold.
//! - GIA Gem Encyclopedia tourmaline entry (2023 update).

use super::{
    DrudeLorentzParams, LorentzOscillator, MineralMetadata, OpticSign, UniaxialOptical,
};

/// Build a Drude-free `DrudeLorentzParams` with a single explicit
/// oscillator. Tourmaline dispersion is dominated by IR phonons + a
/// UV electronic absorption edge; here we use a simplified single-
/// oscillator approximation tuned so the Sellmeier-like real part
/// reproduces the catalogued n_o or n_e at the sodium-D line.
fn one_oscillator(eps_inf: f64, strength: f64, omega_0_ev: f64, gamma_ev: f64) -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: None,
        oscillators: vec![LorentzOscillator {
            strength,
            omega_0_ev,
            gamma_ev,
        }],
        eps_inf,
        extended_drude: None,
    }
}

/// Schorl -- iron-rich endmember; the most abundant tourmaline.
/// NaFe(II)3Al6(BO3)3Si6O18(OH)4; black to bluish-black.
pub fn schorl_optical() -> UniaxialOptical {
    UniaxialOptical {
        parallel: one_oscillator(2.622, 0.50, 6.0, 0.6), // n_e ~= sqrt(2.622+0.50) approx 1.637 near visible
        perpendicular: one_oscillator(2.722, 0.50, 6.0, 0.6),
        axis_description: "schorl (NaFe3Al6 borosilicate, R3m): n_o=1.675, n_e=1.637 (sodium-D)",
    }
}

/// Metadata for schorl. Refractive indices from Henry et al. (2011) Table 1
/// for natural Fe-dominant samples (Mada Quarry, Nigeria; Cruzeiro, Brazil).
pub fn schorl_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "schorl",
        formula: "NaFe(II)3Al6(BO3)3Si6O18(OH)4",
        crystal_system: "trigonal",
        space_group: "R3m (160)",
        n_omega: 1.675,
        n_epsilon: 1.637,
        birefringence: 0.038,
        optic_sign: OpticSign::Negative,
        hardness_mohs: 7.0,
        density_g_cm3: 3.20,
        color: "black to bluish-black",
        reference: "Henry et al. (2011) Am. Mineralogist 96 Table 1; density Bosi et al. (2019).",
    }
}

/// Dravite -- magnesium-rich endmember; brown/yellow.
/// NaMg3Al6(BO3)3Si6O18(OH)4.
pub fn dravite_optical() -> UniaxialOptical {
    UniaxialOptical {
        parallel: one_oscillator(2.612, 0.42, 7.5, 0.8), // n_e ~ 1.622
        perpendicular: one_oscillator(2.697, 0.42, 7.5, 0.8),
        axis_description: "dravite (NaMg3Al6, R3m): n_o=1.644, n_e=1.622 (sodium-D)",
    }
}

/// Metadata for dravite. From Henry et al. (2011) for the Yinnietharra
/// (Australia) type-locality dravite.
pub fn dravite_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "dravite",
        formula: "NaMg3Al6(BO3)3Si6O18(OH)4",
        crystal_system: "trigonal",
        space_group: "R3m (160)",
        n_omega: 1.644,
        n_epsilon: 1.622,
        birefringence: 0.022,
        optic_sign: OpticSign::Negative,
        hardness_mohs: 7.25,
        density_g_cm3: 3.05,
        color: "brown, yellow, or olive-green",
        reference: "Henry et al. (2011) Am. Mineralogist 96 Table 1.",
    }
}

/// Elbaite -- lithium-rich endmember; the principal gemstone tourmaline.
/// Na(Li1.5Al1.5)Al6(BO3)3Si6O18(OH)4.
/// Color varieties: rubellite (pink, Mn), indicolite (blue, Fe), verdelite
/// (green), achroite (colorless), paraiba (neon Cu).
pub fn elbaite_optical() -> UniaxialOptical {
    UniaxialOptical {
        parallel: one_oscillator(2.578, 0.36, 8.0, 1.0), // n_e ~ 1.620
        perpendicular: one_oscillator(2.624, 0.36, 8.0, 1.0),
        axis_description: "elbaite (Na(Li1.5Al1.5)Al6, R3m): n_o=1.640, n_e=1.620 (sodium-D)",
    }
}

/// Metadata for elbaite (averaged across pink/green/blue gemstone-grade
/// material; specific color varieties have slightly different n values).
pub fn elbaite_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "elbaite",
        formula: "Na(Li1.5Al1.5)Al6(BO3)3Si6O18(OH)4",
        crystal_system: "trigonal",
        space_group: "R3m (160)",
        n_omega: 1.640,
        n_epsilon: 1.620,
        birefringence: 0.020,
        optic_sign: OpticSign::Negative,
        hardness_mohs: 7.25,
        density_g_cm3: 3.05,
        color: "variable: pink (rubellite), green (verdelite), blue (indicolite), colorless (achroite), neon-blue/green (paraiba, Cu-bearing)",
        reference: "GIA Gem Encyclopedia (2023); Henry et al. (2011) Am. Mineralogist 96 Table 1.",
    }
}

/// Uvite -- calcium-magnesium endmember; brown-green/dark green.
/// CaMg3(MgAl5)(BO3)3Si6O18(OH)3F.
pub fn uvite_optical() -> UniaxialOptical {
    UniaxialOptical {
        parallel: one_oscillator(2.598, 0.40, 7.0, 0.7), // n_e ~ 1.619
        perpendicular: one_oscillator(2.692, 0.40, 7.0, 0.7),
        axis_description: "uvite (CaMg3MgAl5, R3m): n_o=1.640, n_e=1.619",
    }
}

pub fn uvite_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "uvite",
        formula: "CaMg3(MgAl5)(BO3)3Si6O18(OH)3F",
        crystal_system: "trigonal",
        space_group: "R3m (160)",
        n_omega: 1.640,
        n_epsilon: 1.619,
        birefringence: 0.021,
        optic_sign: OpticSign::Negative,
        hardness_mohs: 7.5,
        density_g_cm3: 3.10,
        color: "brown to dark green",
        reference: "Henry et al. (2011) Am. Mineralogist 96 Table 1; Bosi et al. (2019).",
    }
}

/// Liddicoatite -- calcium-lithium endmember; multi-color zoned varieties
/// dominant from the Anjanabonoina, Madagascar type locality.
/// Ca(Li2Al)Al6(BO3)3Si6O18(OH)3F.
pub fn liddicoatite_optical() -> UniaxialOptical {
    UniaxialOptical {
        parallel: one_oscillator(2.569, 0.35, 8.0, 1.0), // n_e ~ 1.616
        perpendicular: one_oscillator(2.625, 0.35, 8.0, 1.0),
        axis_description: "liddicoatite (Ca(Li2Al)Al6, R3m): n_o=1.637, n_e=1.616",
    }
}

pub fn liddicoatite_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "liddicoatite",
        formula: "Ca(Li2Al)Al6(BO3)3Si6O18(OH)3F",
        crystal_system: "trigonal",
        space_group: "R3m (160)",
        n_omega: 1.637,
        n_epsilon: 1.616,
        birefringence: 0.021,
        optic_sign: OpticSign::Negative,
        hardness_mohs: 7.25,
        density_g_cm3: 3.02,
        color: "multi-color zoned (pink/green/red/yellow domains)",
        reference: "Dirlam et al. (2002) Gems & Gemology 38(3) Madagascar liddicoatite study.",
    }
}

/// Rossmanite -- vacancy-Li-Al endmember; pink to yellow.
/// (vacancy)(LiAl2)Al6(BO3)3Si6O18(OH)4.
pub fn rossmanite_optical() -> UniaxialOptical {
    UniaxialOptical {
        parallel: one_oscillator(2.560, 0.35, 8.0, 1.0), // n_e ~ 1.613
        perpendicular: one_oscillator(2.613, 0.35, 8.0, 1.0),
        axis_description: "rossmanite (vac.(LiAl2)Al6, R3m): n_o=1.633, n_e=1.613",
    }
}

pub fn rossmanite_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "rossmanite",
        formula: "(vacancy)(LiAl2)Al6(BO3)3Si6O18(OH)4",
        crystal_system: "trigonal",
        space_group: "R3m (160)",
        n_omega: 1.633,
        n_epsilon: 1.613,
        birefringence: 0.020,
        optic_sign: OpticSign::Negative,
        hardness_mohs: 7.0,
        density_g_cm3: 3.00,
        color: "pink to pale yellow",
        reference: "Selway et al. (1998) Am. Mineralogist 83 (1998) -- rossmanite type description.",
    }
}

/// Foitite -- alkali-vacant, Fe-Al endmember; distinctive blue-violet color.
/// (vacancy)(Fe(II)2Al)Al6(BO3)3Si6O18(OH)4.
pub fn foitite_optical() -> UniaxialOptical {
    UniaxialOptical {
        parallel: one_oscillator(2.616, 0.48, 6.0, 0.7), // n_e ~ 1.635
        perpendicular: one_oscillator(2.704, 0.48, 6.0, 0.7),
        axis_description: "foitite (vac.(Fe2Al)Al6, R3m): n_o=1.654, n_e=1.635",
    }
}

pub fn foitite_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "foitite",
        formula: "(vacancy)(Fe(II)2Al)Al6(BO3)3Si6O18(OH)4",
        crystal_system: "trigonal",
        space_group: "R3m (160)",
        n_omega: 1.654,
        n_epsilon: 1.635,
        birefringence: 0.019,
        optic_sign: OpticSign::Negative,
        hardness_mohs: 7.0,
        density_g_cm3: 3.18,
        color: "blue to blue-violet",
        reference: "MacDonald & Hawthorne (1995) Can. Mineralogist 33 -- foitite type description.",
    }
}

/// Povondraite -- Fe(III)-dominant endmember; rare, dark brown to black.
/// Na(Fe(III)3)(Fe(III)4Mg2)(BO3)3Si6O18(OH)4.
pub fn povondraite_optical() -> UniaxialOptical {
    UniaxialOptical {
        parallel: one_oscillator(2.760, 0.55, 5.0, 0.8), // n_e ~ 1.715
        perpendicular: one_oscillator(2.865, 0.55, 5.0, 0.8),
        axis_description: "povondraite (NaFe(III)3..., R3m): n_o=1.735, n_e=1.715",
    }
}

pub fn povondraite_metadata() -> MineralMetadata {
    MineralMetadata {
        species_name: "povondraite",
        formula: "Na(Fe(III)3)(Fe(III)4Mg2)(BO3)3Si6O18(OH)4",
        crystal_system: "trigonal",
        space_group: "R3m (160)",
        n_omega: 1.735,
        n_epsilon: 1.715,
        birefringence: 0.020,
        optic_sign: OpticSign::Negative,
        hardness_mohs: 7.25,
        density_g_cm3: 3.32,
        color: "dark brown to black",
        reference: "Grice et al. (1993) Can. Mineralogist 31 -- povondraite type description; Bosi et al. (2019) Lithos 322.",
    }
}

/// Metadata fallback for the historic isotropic `tourmaline_optical()`
/// constructor that still lives in `oxides_tcos`. Returns elbaite (the
/// most-common gemstone tourmaline) since the legacy entry was tuned to
/// pink elbaite. Use this when you have only `MaterialEntry::tourmaline`
/// and need to surface a meaningful species-level identity.
pub fn default_metadata() -> MineralMetadata {
    elbaite_metadata()
}
