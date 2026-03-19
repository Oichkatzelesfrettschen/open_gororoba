import re

content = open("crates/materials_core/src/optical_database.rs").read()

# We'll insert nitrides after the tungsten_drude_lorentz
new_nitrides = """
// ============================================================================
// Transition Metal Nitrides (Refractory Plasmonics)
// ============================================================================

/// Titanium Nitride (TiN) - Typical room temperature sputter (Patsalas et al. 2018)
pub fn tin_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams {
            omega_p_ev: 6.926,
            gamma_ev: 0.589,
            eps_inf: 1.872,
        }),
        extended_drude: None,
        oscillators: vec![
            LorentzOscillator {
                strength: 3.690, // E01
                omega_0_ev: 3.690,
                gamma_ev: 0.943,
            },
            LorentzOscillator {
                strength: 5.967, // E02
                omega_0_ev: 5.967,
                gamma_ev: 4.882,
            },
        ],
        eps_inf: 1.872,
    }
}

/// Titanium Nitride (TiN) - Low loss epitaxial (Naik et al. 2012)
pub fn tin_low_loss_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams {
            omega_p_ev: 5.707,
            gamma_ev: 0.165,
            eps_inf: 2.667,
        }),
        extended_drude: None,
        oscillators: vec![
            LorentzOscillator {
                strength: 2.264,
                omega_0_ev: 2.264,
                gamma_ev: 0.722,
            },
            LorentzOscillator {
                strength: 4.894,
                omega_0_ev: 4.894,
                gamma_ev: 3.875,
            },
        ],
        eps_inf: 2.667,
    }
}

/// Zirconium Nitride (ZrN)
pub fn zrn_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams {
            omega_p_ev: 7.456,
            gamma_ev: 0.620,
            eps_inf: 1.0,
        }),
        extended_drude: None,
        oscillators: vec![],
        eps_inf: 1.0,
    }
}

/// Hafnium Nitride (HfN)
pub fn hfn_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams {
            omega_p_ev: 8.19,
            gamma_ev: 0.48,
            eps_inf: 4.62,
        }),
        extended_drude: None,
        oscillators: vec![],
        eps_inf: 4.62,
    }
}
"""

new_tcos = """
/// Indium Tin Oxide (ITO) - Typical degenerate TCO
pub fn ito_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams {
            // approx for n = 4e20 cm^-3, m* = 0.4
            omega_p_ev: 1.17,
            gamma_ev: 0.10,
            eps_inf: 4.0,
        }),
        extended_drude: None,
        oscillators: vec![],
        eps_inf: 4.0,
    }
}
"""

content = content.replace("// ============================================================================\n// Dielectrics (Phase 2)", new_nitrides + "\n// ============================================================================\n// Dielectrics (Phase 2)")

content = content.replace("pub fn azo_optical", new_tcos + "\npub fn azo_optical")

open("crates/materials_core/src/optical_database.rs", "w").write(content)
