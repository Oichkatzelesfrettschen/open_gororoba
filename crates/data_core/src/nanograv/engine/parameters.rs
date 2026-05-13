//! Per-parameter lookup, step-size, and prior-sigma helpers.
//!
//! Functions:
//!   * `parse_selector_parameter_index` -- extract the integer index
//!     from "JUMP@0", "DMJUMP@3", etc.
//!   * `parse_fd_order`        -- "FD1" -> 1
//!   * `fd_delay_seconds`      -- TEMPO2 FD profile-delay sum
//!   * `parameter_value_or`        -- override-first, then model;
//!     returns Err if missing
//!   * `parameter_value_or_default` -- same lookup but defaults
//!     instead of erroring
//!   * `parameter_step`        -- 1% of uncertainty, with per-name
//!     physical fallbacks
//!   * `parameter_prior_sigma` -- 3-5x uncertainty scaled by name
//!     category; None for PHASE_OFFSET; uncertainty x 3 for
//!     JUMP@/DMJUMP@
//!   * `solar_system_shapiro_seconds` -- Schwarzschild light-bending
//!     near the Sun
//!   * `selector_matches_observation` -- per-flag selector AND
//!   * `tagged_term_matches_observation` -- shorthand wrapper
//!   * `effective_phase_sigma_seconds` / `effective_dm_sigma` --
//!     EFAC/EQUAD/DMEFAC/DMEQUAD-modified uncertainty
//!
//! All items `pub(super)`.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow};

use super::scalar_math::{dot3, norm3};
use super::super::timing_model::TimingModel;
use super::{C_KM_PER_S, GM_SUN_KM3_S2, IndependentObservation, TimingModelExt};

pub(super) fn parse_selector_parameter_index(name: &str, prefix: &str) -> Option<usize> {
    name.strip_prefix(prefix)?.parse::<usize>().ok()
}

/// Returns the FD order k for a parameter named "FD{k}" (e.g. "FD1" -> 1, "FD2" -> 2).
pub(super) fn parse_fd_order(name: &str) -> Option<u32> {
    name.strip_prefix("FD")?.parse::<u32>().ok()
}

/// Computes the total FD profile-delay correction in seconds for the given frequency.
/// TEMPO2 convention: delay_FD = sum_k FD_k * ln(f/f_ref)^k with f_ref = 1 GHz.
pub(super) fn fd_delay_seconds(
    model: &TimingModel,
    overrides: &BTreeMap<String, f64>,
    frequency_mhz: f64,
) -> f64 {
    const F_REF_MHZ: f64 = 1000.0;
    let log_f = (frequency_mhz / F_REF_MHZ).ln();
    model
        .fd_terms
        .iter()
        .filter_map(|term| {
            let k = parse_fd_order(&term.name)?;
            let value = overrides
                .get(&term.name)
                .copied()
                .or(term.value)
                .unwrap_or(0.0);
            Some(value * log_f.powi(k as i32))
        })
        .sum()
}

pub(super) fn parameter_value_or(
    model: &TimingModel,
    overrides: &BTreeMap<String, f64>,
    name: &str,
) -> Result<f64> {
    overrides
        .get(name)
        .copied()
        .or_else(|| model.parameter_value_local(name))
        .ok_or_else(|| anyhow!("{} missing parameter {name}", model.solution_id))
}

pub(super) fn parameter_value_or_default(
    model: &TimingModel,
    overrides: &BTreeMap<String, f64>,
    name: &str,
    default: f64,
) -> f64 {
    overrides
        .get(name)
        .copied()
        .or_else(|| model.parameter_value_local(name))
        .unwrap_or(default)
}

pub(super) fn parameter_step(model: &TimingModel, name: &str) -> Result<f64> {
    let term = model
        .parameter_term_local(name)
        .ok_or_else(|| anyhow!("{} missing parameter term {name}", model.solution_id))?;
    if let Some(uncertainty) = term.uncertainty.filter(|value| *value > 0.0) {
        return Ok(0.01 * uncertainty);
    }
    let fallback = match name {
        "F0" => 1.0e-12,
        "F1" => 1.0e-20,
        "ELONG" | "ELAT" | "PMELONG" | "PMELAT" => 1.0e-8,
        "PX" => 1.0e-5,
        "DM" => 1.0e-5,
        value if value.starts_with("DMX_") => 1.0e-5,
        "A1" => 1.0e-6,
        "TASC" | "T0" => 1.0e-7,
        "EPS1" | "EPS2" => 1.0e-8,
        "PB" => 1.0e-8,
        "PBDOT" => 1.0e-12,
        name if name.starts_with("FB") => 1.0e-16,
        "ECC" => 1.0e-8,
        "OM" | "OMDOT" | "KIN" | "KOM" => 1.0e-6,
        "GAMMA" | "A0" | "B0" => 1.0e-8,
        "M2" | "SINI" | "DR" | "DTH" | "H3" | "H4" | "STIGMA" => 1.0e-6,
        _ => 1.0e-8,
    };
    Ok(fallback)
}

pub(super) fn parameter_prior_sigma(model: &TimingModel, name: &str) -> Option<f64> {
    if name == "PHASE_OFFSET" {
        return None;
    }
    // JUMP and DMJUMP are inter-backend calibration offsets fully determined by the data.
    // WHY: adding a prior would bias them toward zero, suppressing legitimate backend-to-backend
    // offsets. They are not physical parameters so we let the data determine them freely.
    if let Some(index) = parse_selector_parameter_index(name, "JUMP@") {
        return model
            .jumps
            .get(index)
            .and_then(|term| term.uncertainty)
            .filter(|value| *value > 0.0)
            .map(|uncertainty| 3.0 * uncertainty);
    }
    if let Some(index) = parse_selector_parameter_index(name, "DMJUMP@") {
        return model
            .dmjumps
            .get(index)
            .and_then(|term| term.uncertainty)
            .filter(|value| *value > 0.0)
            .map(|uncertainty| 3.0 * uncertainty);
    }
    let prior_scale = match name {
        "DM" => 1.0,
        value if value.starts_with("DMX_") => 1.0,
        value if value.starts_with("FD") => 3.0,
        "ELONG" | "ELAT" | "PMELONG" | "PMELAT" | "PX" => 3.0,
        "A1" | "A1DOT" | "PB" | "PBDOT" | "T0" | "TASC" | "ECC" | "EPS1" | "EPS2" | "OM"
        | "OMDOT" | "KIN" | "KOM" | "M2" | "SINI" | "GAMMA" | "DR" | "DTH" | "H3" | "H4"
        | "STIGMA" => 3.0,
        _ => 5.0,
    };
    if let Some(uncertainty) = model
        .parameter_term_local(name)
        .and_then(|term| term.uncertainty)
        .filter(|value| *value > 0.0)
    {
        return Some(prior_scale * uncertainty);
    }
    parameter_step(model, name)
        .ok()
        .map(|step| prior_scale.max(1.0) * 100.0 * step)
}

pub(super) fn solar_system_shapiro_seconds(
    sun_from_earth_km: [f64; 3],
    sky_unit: [f64; 3],
) -> f64 {
    let radius = norm3(sun_from_earth_km);
    if radius <= 0.0 {
        return 0.0;
    }
    let projection = dot3(sun_from_earth_km, sky_unit);
    let argument = (radius - projection).abs().max(1.0);
    -2.0 * GM_SUN_KM3_S2 / C_KM_PER_S.powi(3) * argument.ln()
}

pub(super) fn selector_matches_observation(
    selectors: &[super::super::timing_model::SelectorTerm],
    observation: &IndependentObservation,
) -> bool {
    selectors.iter().all(|selector| {
        if selector.flag == "-tel" {
            selector.value == observation.site.as_str()
        } else {
            observation
                .flags
                .get(&selector.flag)
                .is_some_and(|value| value == &selector.value)
        }
    })
}

pub(super) fn tagged_term_matches_observation(
    term: &super::super::timing_model::TaggedTerm,
    observation: &IndependentObservation,
) -> bool {
    selector_matches_observation(&term.selectors, observation)
}

pub(super) fn effective_phase_sigma_seconds(
    model: &TimingModel,
    observation: &IndependentObservation,
) -> f64 {
    let mut efac = 1.0;
    let mut equad_us2 = 0.0;
    for term in &model.noise_terms {
        if !tagged_term_matches_observation(term, observation) {
            continue;
        }
        match term.name.as_str() {
            "EFAC" | "TNEF" => {
                efac *= term.value.unwrap_or(1.0).abs().max(1.0e-6);
            }
            "EQUAD" | "TNEQ" => {
                let value = term.value.unwrap_or(0.0);
                equad_us2 += value * value;
            }
            _ => {}
        }
    }
    let formal_s = observation.uncertainty_us.max(1.0e-6) * 1.0e-6;
    let equad_s = equad_us2.sqrt() * 1.0e-6;
    (((efac * formal_s).powi(2) + equad_s.powi(2)).sqrt()).max(1.0e-12)
}

pub(super) fn effective_dm_sigma(
    model: &TimingModel,
    observation: &IndependentObservation,
    pp_dme: f64,
) -> f64 {
    let mut dmefac = 1.0;
    let mut dmequad2 = 0.0;
    for term in &model.noise_terms {
        if !tagged_term_matches_observation(term, observation) {
            continue;
        }
        match term.name.as_str() {
            "DMEFAC" => {
                dmefac *= term.value.unwrap_or(1.0).abs().max(1.0e-6);
            }
            "DMEQUAD" => {
                let value = term.value.unwrap_or(0.0);
                dmequad2 += value * value;
            }
            _ => {}
        }
    }
    (((dmefac * pp_dme.max(1.0e-9)).powi(2) + dmequad2).sqrt()).max(1.0e-12)
}
