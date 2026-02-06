//! Cayley-Dickson Zero-Divisor to Metamaterial Layer Mapping (C-010).
//!
//! Maps the algebraic structure of sedenion zero-divisors to physical
//! parameters of metamaterial absorber layers.
//!
//! Physical realizability constraints:
//! - n > 0 (real part of refractive index)
//! - k >= 0 (extinction coefficient)
//! - thickness > 0
//!
//! References:
//! - Leonhardt & Philbin (2006) - Transformation optics
//! - Smith & Pendry (2006) - Metamaterial parameter retrieval

use num_complex::Complex64;

/// Material classification based on optical properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialType {
    Dielectric,
    Plasmonic,
    Hyperbolic,
}

/// Physical parameters for a metamaterial absorber layer.
#[derive(Debug, Clone)]
pub struct MetamaterialLayer {
    pub layer_id: usize,
    pub n_real: f64,       // Real part of refractive index
    pub n_imag: f64,       // Imaginary part (extinction coefficient k)
    pub thickness_nm: f64, // Layer thickness in nanometers
    pub material_type: MaterialType,
}

/// Mapping from a ZD pair to metamaterial layer parameters.
#[derive(Debug, Clone)]
pub struct ZdToLayerMapping {
    pub zd_indices: (usize, usize, usize, usize),
    pub product_norm: f64,
    pub layer: MetamaterialLayer,
    pub is_physical: bool,
}

/// Map ZD indices to complex refractive index.
///
/// Uses the index values to determine optical properties:
/// - Higher indices -> more exotic response
/// - Paired indices determine layer contrast
pub fn map_zd_to_refractive_index(
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    base_n: f64,
    modulation_strength: f64,
) -> Complex64 {
    // Map index sum to real part
    let index_sum = (i + j + k + l) as f64 / 60.0;
    let n_real = base_n + modulation_strength * (std::f64::consts::PI * index_sum).sin();

    // Map index product to extinction
    let index_product = ((i * j * k * l) as f64).powf(0.25) / 15.0;
    let n_imag = 0.1 * modulation_strength * index_product;

    Complex64::new(n_real, n_imag)
}

/// Map ZD product norm to layer thickness.
///
/// Better annihilation (lower norm) -> thinner resonant layers.
pub fn map_zd_norm_to_thickness(
    norm: f64,
    min_thickness: f64,
    max_thickness: f64,
    inverse_scaling: bool,
) -> f64 {
    let t = if inverse_scaling {
        // Sigmoid-like: norm=0 -> min, norm large -> max
        min_thickness + (max_thickness - min_thickness) * (1.0 - (-norm / 0.1).exp())
    } else {
        // Linear
        min_thickness + norm * (max_thickness - min_thickness)
    };

    t.max(min_thickness).min(max_thickness)
}

/// Classify material type based on optical properties.
pub fn classify_material_type(n_complex: Complex64) -> MaterialType {
    if n_complex.re < 0.0 {
        MaterialType::Hyperbolic
    } else if n_complex.im > 0.5 {
        MaterialType::Plasmonic
    } else {
        MaterialType::Dielectric
    }
}

/// Map a single ZD pair to metamaterial layer parameters.
pub fn map_zd_pair_to_layer(
    zd: (usize, usize, usize, usize, f64),
    layer_id: usize,
    base_n: f64,
) -> ZdToLayerMapping {
    let (i, j, k, l, norm) = zd;

    let n_complex = map_zd_to_refractive_index(i, j, k, l, base_n, 0.2);
    let thickness = map_zd_norm_to_thickness(norm, 10.0, 200.0, true);
    let material_type = classify_material_type(n_complex);

    let is_physical = n_complex.re > 0.0 && n_complex.im >= 0.0 && thickness > 0.0;

    let layer = MetamaterialLayer {
        layer_id,
        n_real: n_complex.re,
        n_imag: n_complex.im,
        thickness_nm: thickness,
        material_type,
    };

    ZdToLayerMapping {
        zd_indices: (i, j, k, l),
        product_norm: norm,
        layer,
        is_physical,
    }
}

/// Build absorber stack from ZD pairs.
pub fn build_absorber_stack(
    zd_pairs: &[(usize, usize, usize, usize, f64)],
    max_layers: usize,
    base_n: f64,
) -> Vec<ZdToLayerMapping> {
    // Sort by norm (best annihilators first)
    let mut sorted_pairs = zd_pairs.to_vec();
    sorted_pairs.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap());

    sorted_pairs.iter()
        .take(max_layers)
        .enumerate()
        .map(|(layer_id, &zd)| map_zd_pair_to_layer(zd, layer_id, base_n))
        .collect()
}

/// Physical realizability verification result.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub n_total: usize,
    pub n_physical: usize,
    pub n_dielectric: usize,
    pub n_plasmonic: usize,
    pub n_hyperbolic: usize,
    pub all_n_positive: bool,
    pub all_k_nonnegative: bool,
    pub all_thickness_positive: bool,
    pub all_physical: bool,
}

/// Verify physical realizability of absorber stack.
pub fn verify_physical_realizability(stack: &[ZdToLayerMapping]) -> VerificationResult {
    let n_total = stack.len();
    let n_physical = stack.iter().filter(|m| m.is_physical).count();

    let n_dielectric = stack.iter()
        .filter(|m| m.layer.material_type == MaterialType::Dielectric)
        .count();
    let n_plasmonic = stack.iter()
        .filter(|m| m.layer.material_type == MaterialType::Plasmonic)
        .count();
    let n_hyperbolic = stack.iter()
        .filter(|m| m.layer.material_type == MaterialType::Hyperbolic)
        .count();

    let all_n_positive = stack.iter().all(|m| m.layer.n_real > 0.0);
    let all_k_nonnegative = stack.iter().all(|m| m.layer.n_imag >= 0.0);
    let all_thickness_positive = stack.iter().all(|m| m.layer.thickness_nm > 0.0);

    VerificationResult {
        n_total,
        n_physical,
        n_dielectric,
        n_plasmonic,
        n_hyperbolic,
        all_n_positive,
        all_k_nonnegative,
        all_thickness_positive,
        all_physical: n_physical == n_total,
    }
}

/// Canonical sedenion ZD pairs for testing.
pub fn canonical_sedenion_zd_pairs() -> Vec<(usize, usize, usize, usize, f64)> {
    vec![
        (1, 2, 4, 8, 0.0),
        (1, 4, 2, 8, 0.0),
        (1, 8, 2, 4, 0.0),
        (2, 4, 1, 8, 0.0),
        (2, 8, 1, 4, 0.0),
        (4, 8, 1, 2, 0.0),
        (3, 5, 6, 9, 0.0),
        (3, 6, 5, 10, 0.0),
        (3, 9, 5, 12, 0.0),
        (5, 10, 3, 12, 0.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refractive_index_positive() {
        for i in 1..10 {
            for j in i + 1..10 {
                let n = map_zd_to_refractive_index(i, j, j + 1, j + 2, 1.5, 0.2);
                assert!(n.re > 0.0, "n.re should be > 0");
                assert!(n.im >= 0.0, "n.im should be >= 0");
            }
        }
    }

    #[test]
    fn test_thickness_bounded() {
        let min_t = 10.0;
        let max_t = 200.0;
        for norm in [0.0, 0.001, 0.01, 0.1, 1.0, 10.0] {
            let t = map_zd_norm_to_thickness(norm, min_t, max_t, true);
            assert!(t >= min_t && t <= max_t, "Thickness {} not in bounds", t);
        }
    }

    #[test]
    fn test_lower_norm_thinner() {
        let t_low = map_zd_norm_to_thickness(0.0, 10.0, 200.0, true);
        let t_high = map_zd_norm_to_thickness(1.0, 10.0, 200.0, true);
        assert!(t_low < t_high, "Lower norm should give thinner layer");
    }

    #[test]
    fn test_material_classification() {
        assert_eq!(
            classify_material_type(Complex64::new(1.5, 0.01)),
            MaterialType::Dielectric
        );
        assert_eq!(
            classify_material_type(Complex64::new(1.5, 0.8)),
            MaterialType::Plasmonic
        );
        assert_eq!(
            classify_material_type(Complex64::new(-0.5, 0.1)),
            MaterialType::Hyperbolic
        );
    }

    #[test]
    fn test_layer_mapping_physical() {
        let zd = (1, 2, 4, 8, 0.0);
        let mapping = map_zd_pair_to_layer(zd, 0, 1.5);
        assert!(mapping.is_physical);
        assert!(mapping.layer.n_real > 0.0);
        assert!(mapping.layer.n_imag >= 0.0);
        assert!(mapping.layer.thickness_nm > 0.0);
    }

    #[test]
    fn test_absorber_stack_max_layers() {
        let pairs = canonical_sedenion_zd_pairs();
        let stack = build_absorber_stack(&pairs, 5, 1.5);
        assert!(stack.len() <= 5);
    }

    #[test]
    fn test_verification_all_physical() {
        let pairs = canonical_sedenion_zd_pairs();
        let stack = build_absorber_stack(&pairs, 10, 1.5);
        let verification = verify_physical_realizability(&stack);

        assert!(verification.all_n_positive);
        assert!(verification.all_k_nonnegative);
        assert!(verification.all_thickness_positive);
    }
}
