//! Magpie-style composition featurizer for materials science.
//!
//! Computes composition-weighted statistics over elemental properties
//! (Ward et al. 2016, "A general-purpose machine learning framework for
//! predicting properties of inorganic materials").
//!
//! Given a chemical formula like "Al2O3", the featurizer:
//! 1. Parses the formula into (element, count) pairs
//! 2. Looks up each element's properties in the periodic table
//! 3. Computes composition-weighted mean, std, min, max, range for each property
//! 4. Returns a fixed-length feature vector (54 elements)

use crate::periodic_table::{get_element, Element};
use regex::Regex;

/// Statistics for one elemental property across a composition.
#[derive(Debug, Clone, Copy)]
pub struct PropertyStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub range: f64,
}

/// Full composition feature set for one material.
#[derive(Debug, Clone)]
pub struct CompositionFeatures {
    /// Number of unique elements.
    pub n_elements: f64,
    /// Total stoichiometric coefficient sum.
    pub total_atoms: f64,
    /// Fraction of metallic elements (by count).
    pub metal_fraction: f64,
    /// Fraction of semiconductor elements (by count).
    pub semiconductor_fraction: f64,
    /// Statistics for each of the 10 elemental properties.
    pub property_stats: [PropertyStats; 10],
}

/// Names of the 10 elemental properties used for featurization.
const PROPERTY_NAMES: [&str; 10] = [
    "mass",
    "density",
    "melting_pt",
    "boiling_pt",
    "valence",
    "electronegativity",
    "ionization_energy",
    "electron_affinity",
    "lattice_const",
    "atomic_number",
];

/// Parse a chemical formula into (element, count) pairs.
///
/// Handles formulas like "Al2O3", "NaCl", "Si", "Fe2O3".
/// Elements without explicit counts default to 1.
pub fn parse_formula(formula: &str) -> Result<Vec<(String, f64)>, String> {
    let re = Regex::new(r"([A-Z][a-z]?)(\d*\.?\d*)").unwrap();
    let mut result = Vec::new();

    for cap in re.captures_iter(formula) {
        let symbol = cap[1].to_string();
        let count_str = &cap[2];
        let count = if count_str.is_empty() {
            1.0
        } else {
            count_str
                .parse::<f64>()
                .map_err(|e| format!("Invalid count for {symbol}: {e}"))?
        };
        result.push((symbol, count));
    }

    if result.is_empty() {
        return Err(format!("Could not parse formula: {formula}"));
    }

    Ok(result)
}

/// Parse a formula and normalize counts to composition fractions summing to 1.
pub fn composition_fractions(formula: &str) -> Result<Vec<(String, f64)>, String> {
    let pairs = parse_formula(formula)?;
    let total: f64 = pairs.iter().map(|(_, c)| c).sum();
    if total <= 0.0 {
        return Err("Formula has zero total count".to_string());
    }
    Ok(pairs.into_iter().map(|(el, c)| (el, c / total)).collect())
}

/// Extract a numeric property from an Element, returning a fallback if None.
fn element_property(elem: &Element, index: usize) -> f64 {
    match index {
        0 => elem.atomic_mass,
        1 => elem.density.unwrap_or(0.0),
        2 => elem.melting_point.unwrap_or(0.0),
        3 => elem.boiling_point.unwrap_or(0.0),
        4 => elem.valence_electrons as f64,
        5 => elem.electronegativity.unwrap_or(0.0),
        6 => elem.ionization_energy.unwrap_or(0.0),
        7 => elem.electron_affinity.unwrap_or(0.0),
        8 => elem.lattice_constant.unwrap_or(0.0),
        9 => elem.atomic_number as f64,
        _ => 0.0,
    }
}

/// Compute composition-weighted property statistics.
fn compute_property_stats(elements: &[(Element, f64)], prop_index: usize) -> PropertyStats {
    let values: Vec<f64> = elements
        .iter()
        .map(|(e, _)| element_property(e, prop_index))
        .collect();
    let weights: Vec<f64> = elements.iter().map(|(_, w)| *w).collect();
    let total_weight: f64 = weights.iter().sum();

    // Weighted mean
    let mean = values
        .iter()
        .zip(weights.iter())
        .map(|(v, w)| v * w)
        .sum::<f64>()
        / total_weight;

    // Weighted standard deviation
    let variance = values
        .iter()
        .zip(weights.iter())
        .map(|(v, w)| w * (v - mean).powi(2))
        .sum::<f64>()
        / total_weight;
    let std = variance.sqrt();

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    PropertyStats {
        mean,
        std,
        min,
        max,
        range: max - min,
    }
}

/// Featurize a chemical formula into composition features.
///
/// Returns an error if the formula cannot be parsed or any element is unknown.
pub fn featurize(formula: &str) -> Result<CompositionFeatures, String> {
    let fractions = composition_fractions(formula)?;

    // Look up each element
    let mut elements: Vec<(Element, f64)> = Vec::with_capacity(fractions.len());
    for (symbol, frac) in &fractions {
        let elem = get_element(symbol).ok_or_else(|| format!("Unknown element: {symbol}"))?;
        elements.push((elem, *frac));
    }

    let n_elements = elements.len() as f64;
    let total_atoms: f64 = parse_formula(formula)?.iter().map(|(_, c)| c).sum();

    // Metal/semiconductor fractions (by composition fraction, not count)
    let metal_fraction: f64 = elements
        .iter()
        .filter(|(e, _)| e.is_metal)
        .map(|(_, w)| w)
        .sum();
    let semiconductor_fraction: f64 = elements
        .iter()
        .filter(|(e, _)| e.is_semiconductor)
        .map(|(_, w)| w)
        .sum();

    // Compute stats for each of 10 properties
    let mut property_stats = [PropertyStats {
        mean: 0.0,
        std: 0.0,
        min: 0.0,
        max: 0.0,
        range: 0.0,
    }; 10];
    for (i, slot) in property_stats.iter_mut().enumerate() {
        *slot = compute_property_stats(&elements, i);
    }

    Ok(CompositionFeatures {
        n_elements,
        total_atoms,
        metal_fraction,
        semiconductor_fraction,
        property_stats,
    })
}

/// Flatten CompositionFeatures into a fixed-length numeric vector.
///
/// Layout: [n_elements, total_atoms, metal_fraction, semiconductor_fraction,
///          mass_mean, mass_std, mass_min, mass_max, mass_range,
///          density_mean, ..., atomic_number_range]
/// Total: 4 global + 10 props * 5 stats = 54 elements.
pub fn feature_vector(feats: &CompositionFeatures) -> Vec<f64> {
    let mut v = Vec::with_capacity(54);
    v.push(feats.n_elements);
    v.push(feats.total_atoms);
    v.push(feats.metal_fraction);
    v.push(feats.semiconductor_fraction);
    for ps in &feats.property_stats {
        v.push(ps.mean);
        v.push(ps.std);
        v.push(ps.min);
        v.push(ps.max);
        v.push(ps.range);
    }
    v
}

/// Column labels for the feature vector, matching `feature_vector` layout.
pub fn feature_names() -> Vec<&'static str> {
    let mut names = Vec::with_capacity(54);
    names.push("n_elements");
    names.push("total_atoms");
    names.push("metal_fraction");
    names.push("semiconductor_fraction");
    for prop in &PROPERTY_NAMES {
        for stat in &["mean", "std", "min", "max", "range"] {
            // Leak a static string for each combination.
            // These are constructed once and live for the program's duration.
            names.push(Box::leak(format!("{prop}_{stat}").into_boxed_str()));
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_formula_simple() {
        let pairs = parse_formula("Al2O3").unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0], ("Al".to_string(), 2.0));
        assert_eq!(pairs[1], ("O".to_string(), 3.0));
    }

    #[test]
    fn test_parse_formula_single_atom() {
        let pairs = parse_formula("O").unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], ("O".to_string(), 1.0));
    }

    #[test]
    fn test_parse_formula_no_count() {
        let pairs = parse_formula("NaCl").unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0], ("Na".to_string(), 1.0));
        assert_eq!(pairs[1], ("Cl".to_string(), 1.0));
    }

    #[test]
    fn test_composition_fractions() {
        let fracs = composition_fractions("Al2O3").unwrap();
        assert_eq!(fracs.len(), 2);
        assert!((fracs[0].1 - 0.4).abs() < 1e-10);
        assert!((fracs[1].1 - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_featurize_silicon() {
        let feats = featurize("Si").unwrap();
        assert!((feats.n_elements - 1.0).abs() < 1e-10);
        assert!((feats.total_atoms - 1.0).abs() < 1e-10);
        // Silicon atomic mass ~28.085
        let mass_mean = feats.property_stats[0].mean;
        assert!(
            (mass_mean - 28.085).abs() < 0.1,
            "Expected Si mass ~28.085, got {mass_mean}"
        );
        // Single element -> std = 0
        assert!(feats.property_stats[0].std.abs() < 1e-10);
        // Silicon is semiconductor, not metal
        assert!((feats.metal_fraction).abs() < 1e-10);
        assert!((feats.semiconductor_fraction - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_feature_vector_length() {
        let feats = featurize("Fe2O3").unwrap();
        let v = feature_vector(&feats);
        assert_eq!(v.len(), 54);
    }

    #[test]
    fn test_feature_names_length() {
        let names = feature_names();
        assert_eq!(names.len(), 54);
        // First 4 are global features
        assert_eq!(names[0], "n_elements");
        assert_eq!(names[3], "semiconductor_fraction");
        // Then property stats
        assert_eq!(names[4], "mass_mean");
        assert_eq!(names[8], "mass_range");
    }
}
