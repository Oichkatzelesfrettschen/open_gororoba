use crate::higher_cd::cd_name;
use cd_kernel::cayley_dickson::cd_basis_mul_sign_iter;
use flavor_lifts::extract_vk_basis;
use nalgebra::{DMatrix, SymmetricEigen};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct HigherCdControlSpec {
    pub ambient_dim: usize,
    pub vk_rank: usize,
}

impl HigherCdControlSpec {
    pub fn new(ambient_dim: usize, vk_rank: usize) -> Result<Self, String> {
        if !ambient_dim.is_power_of_two() || ambient_dim < 16 {
            return Err(format!(
                "higher-CD control dim must be a power of two >= 16, got {ambient_dim}"
            ));
        }
        if vk_rank == 0 {
            return Err("higher-CD control vk_rank must be > 0".to_string());
        }
        Ok(Self {
            ambient_dim,
            vk_rank,
        })
    }

    pub fn pathion32() -> Self {
        Self {
            ambient_dim: 32,
            vk_rank: 20,
        }
    }

    pub fn algebra_name(&self) -> &'static str {
        cd_name(self.ambient_dim)
    }

    pub fn summary_row(&self) -> String {
        format!(
            "algebra={} dim={} vk_rank={}",
            self.algebra_name(),
            self.ambient_dim,
            self.vk_rank
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HigherCdBasisReport {
    pub algebra_name: String,
    pub ambient_dim: usize,
    pub requested_rank: usize,
    pub actual_rank: usize,
    pub basis_cols: usize,
    pub assessor_count: usize,
    pub singular_values: Vec<f64>,
    pub effective_rank_1e4: usize,
    pub effective_rank_1e6: usize,
    pub effective_rank_1e8: usize,
    pub effective_rank_1e10: usize,
    pub leading_singular_value: f64,
    pub trailing_singular_value: f64,
}

impl HigherCdBasisReport {
    pub fn summary_row(&self) -> String {
        format!(
            "algebra={} dim={} requested_rank={} actual_rank={} assessors={} sv_top={:.12} sv_tail={:.12}",
            self.algebra_name,
            self.ambient_dim,
            self.requested_rank,
            self.actual_rank,
            self.assessor_count,
            self.leading_singular_value,
            self.trailing_singular_value
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ZdGraphSpectrumReport {
    pub algebra_name: String,
    pub ambient_dim: usize,
    pub edge_count: usize,
    pub degree_min: usize,
    pub degree_max: usize,
    pub degree_mean: f64,
    pub n_components: usize,
    pub positive_eigenvalue_count: usize,
    pub eigenvalues: Vec<f64>,
    pub eigenvalues_16: [f32; 16],
}

impl ZdGraphSpectrumReport {
    pub fn summary_row(&self) -> String {
        format!(
            "algebra={} dim={} edges={} comps={} degree_min={} degree_max={} degree_mean={:.6}",
            self.algebra_name,
            self.ambient_dim,
            self.edge_count,
            self.n_components,
            self.degree_min,
            self.degree_max,
            self.degree_mean
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HigherCdControlSummary {
    pub algebra_name: String,
    pub ambient_dim: usize,
    pub requested_rank: usize,
    pub actual_rank: usize,
    pub assessor_count: usize,
    pub edge_count: usize,
    pub connected_components: usize,
    pub positive_eigenvalue_count: usize,
    pub leading_singular_value: f64,
    pub trailing_singular_value: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ZdResonanceBand {
    pub harmonic: usize,
    pub zd_index: usize,
    pub eigenvalue: f64,
    pub detuning: f64,
    pub coupling_strength: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ZdResonanceConfig {
    pub max_harmonic: usize,
    pub width: f64,
    pub min_coupling: f64,
}

impl Default for ZdResonanceConfig {
    fn default() -> Self {
        Self {
            max_harmonic: 5,
            width: 0.1,
            min_coupling: 0.01,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ZdResonanceReport {
    pub algebra_name: String,
    pub ambient_dim: usize,
    pub orbital_frequency: f64,
    pub alpha_scale: f64,
    pub total_coupling: f64,
    pub perturbation: [f64; 3],
    pub bands: Vec<ZdResonanceBand>,
}

pub type PathionResonanceReport = ZdResonanceReport;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HigherCdControlReport {
    pub spec: HigherCdControlSpec,
    pub basis_report: HigherCdBasisReport,
    pub spectrum_report: ZdGraphSpectrumReport,
    pub summary: HigherCdControlSummary,
}

pub type PathionControlReport = HigherCdControlReport;

impl HigherCdControlReport {
    pub fn summary_row(&self) -> String {
        format!(
            "{} {} {}",
            self.spec.summary_row(),
            self.basis_report.summary_row(),
            self.spectrum_report.summary_row()
        )
    }

    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("serialize higher-CD control report")
    }
}

impl ZdResonanceReport {
    pub fn summary_row(&self) -> String {
        format!(
            "algebra={} dim={} orbital_freq={:.12} alpha_scale={:.12} total_coupling={:.12} band_count={} perturbation=[{:.12},{:.12},{:.12}]",
            self.algebra_name,
            self.ambient_dim,
            self.orbital_frequency,
            self.alpha_scale,
            self.total_coupling,
            self.bands.len(),
            self.perturbation[0],
            self.perturbation[1],
            self.perturbation[2]
        )
    }

    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("serialize higher-CD resonance report")
    }
}

pub fn compute_higher_cd_basis_report(spec: &HigherCdControlSpec) -> HigherCdBasisReport {
    let (basis, singular_values, assessors) = extract_vk_basis(spec.ambient_dim, spec.vk_rank);
    let leading_singular_value = singular_values.first().copied().unwrap_or(0.0);
    let trailing_singular_value = singular_values.last().copied().unwrap_or(0.0);

    HigherCdBasisReport {
        algebra_name: spec.algebra_name().to_string(),
        ambient_dim: spec.ambient_dim,
        requested_rank: spec.vk_rank,
        actual_rank: basis.nrows(),
        basis_cols: basis.ncols(),
        assessor_count: assessors.len(),
        effective_rank_1e4: singular_values
            .iter()
            .filter(|&&value| value > 1e-4)
            .count(),
        effective_rank_1e6: singular_values
            .iter()
            .filter(|&&value| value > 1e-6)
            .count(),
        effective_rank_1e8: singular_values
            .iter()
            .filter(|&&value| value > 1e-8)
            .count(),
        effective_rank_1e10: singular_values
            .iter()
            .filter(|&&value| value > 1e-10)
            .count(),
        singular_values,
        leading_singular_value,
        trailing_singular_value,
    }
}

fn associator_nonzero(dim: usize, i: usize, j: usize, k: usize) -> bool {
    let ij = i ^ j;
    let sign_ij = cd_basis_mul_sign_iter(dim, i, j);
    let jk = j ^ k;
    let sign_jk = cd_basis_mul_sign_iter(dim, j, k);

    let ij_k = ij ^ k;
    let sign_ij_k = sign_ij * cd_basis_mul_sign_iter(dim, ij, k);

    let i_jk = i ^ jk;
    let sign_i_jk = sign_jk * cd_basis_mul_sign_iter(dim, i, jk);

    ij_k != i_jk || sign_ij_k != sign_i_jk
}

fn build_zd_adjacency(dim: usize) -> DMatrix<f64> {
    let mut adjacency = DMatrix::zeros(dim, dim);

    for i in 1..dim {
        for j in (i + 1)..dim {
            let mut has_assoc = false;
            for k in 1..dim {
                if k == i || k == j {
                    continue;
                }
                if associator_nonzero(dim, i, j, k) {
                    has_assoc = true;
                    break;
                }
            }
            if has_assoc {
                adjacency[(i, j)] = 1.0;
                adjacency[(j, i)] = 1.0;
            }
        }
    }

    adjacency
}

fn graph_laplacian(adjacency: &DMatrix<f64>) -> DMatrix<f64> {
    let n = adjacency.nrows();
    let mut laplacian = -adjacency.clone();
    for i in 0..n {
        let degree: f64 = (0..n).map(|j| adjacency[(i, j)]).sum();
        laplacian[(i, i)] = degree;
    }
    laplacian
}

pub fn compute_zd_graph_spectrum(dim: usize) -> ZdGraphSpectrumReport {
    assert!(
        dim.is_power_of_two() && dim >= 4,
        "dim must be a power of two >= 4"
    );

    let adjacency = build_zd_adjacency(dim);
    let laplacian = graph_laplacian(&adjacency);
    let eigen = SymmetricEigen::new(laplacian);

    let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for value in &mut eigenvalues {
        if value.abs() < 1e-10 {
            *value = 0.0;
        }
    }

    let degrees: Vec<usize> = (0..dim)
        .map(|row| (0..dim).filter(|&col| adjacency[(row, col)] > 0.5).count())
        .collect();
    let degree_sum: usize = degrees.iter().sum();
    let positive_eigenvalue_count = eigenvalues.iter().filter(|&&value| value > 1e-10).count();

    let mut eigenvalues_16 = [0.0_f32; 16];
    let positive: Vec<f64> = eigenvalues
        .iter()
        .copied()
        .filter(|&value| value > 1e-10)
        .collect();
    for (index, value) in positive.iter().take(16).enumerate() {
        eigenvalues_16[index] = *value as f32;
    }
    let max_ev = eigenvalues_16.iter().copied().fold(0.0_f32, f32::max);
    if max_ev > 0.0 {
        for value in &mut eigenvalues_16 {
            *value /= max_ev;
        }
    }

    ZdGraphSpectrumReport {
        algebra_name: cd_name(dim).to_string(),
        ambient_dim: dim,
        edge_count: degree_sum / 2,
        degree_min: degrees.iter().copied().min().unwrap_or(0),
        degree_max: degrees.iter().copied().max().unwrap_or(0),
        degree_mean: degree_sum as f64 / dim as f64,
        n_components: eigenvalues.iter().filter(|&&value| value == 0.0).count(),
        positive_eigenvalue_count,
        eigenvalues,
        eigenvalues_16,
    }
}

pub fn summarize_control_report(report: &HigherCdControlReport) -> HigherCdControlSummary {
    HigherCdControlSummary {
        algebra_name: report.spec.algebra_name().to_string(),
        ambient_dim: report.spec.ambient_dim,
        requested_rank: report.spec.vk_rank,
        actual_rank: report.basis_report.actual_rank,
        assessor_count: report.basis_report.assessor_count,
        edge_count: report.spectrum_report.edge_count,
        connected_components: report.spectrum_report.n_components,
        positive_eigenvalue_count: report.spectrum_report.positive_eigenvalue_count,
        leading_singular_value: report.basis_report.leading_singular_value,
        trailing_singular_value: report.basis_report.trailing_singular_value,
    }
}

pub fn compute_resonance_bands_from_eigenvalues(
    eigenvalues: &[f64],
    orbital_freq: f64,
    config: &ZdResonanceConfig,
) -> Vec<ZdResonanceBand> {
    if orbital_freq <= 0.0 {
        return Vec::new();
    }

    let two_pi = 2.0 * std::f64::consts::PI;
    let mut bands = Vec::new();

    for (zd_index, &eigenvalue) in eigenvalues.iter().enumerate() {
        if eigenvalue <= 1e-10 {
            continue;
        }
        for harmonic in 1..=config.max_harmonic {
            let resonant_freq = eigenvalue / (two_pi * harmonic as f64);
            let detuning = (orbital_freq - resonant_freq).abs() / orbital_freq;
            let coupling_strength = 1.0 / (1.0 + (detuning / config.width).powi(2));
            if coupling_strength >= config.min_coupling {
                bands.push(ZdResonanceBand {
                    harmonic,
                    zd_index,
                    eigenvalue,
                    detuning,
                    coupling_strength,
                });
            }
        }
    }

    bands.sort_by(|left, right| {
        right
            .coupling_strength
            .partial_cmp(&left.coupling_strength)
            .unwrap()
    });
    bands
}

pub fn total_resonance_coupling_from_bands(bands: &[ZdResonanceBand]) -> f64 {
    bands.iter().map(|band| band.coupling_strength).sum()
}

pub fn resonance_modulated_perturbation_from_eigenvalues(
    eigenvalues: &[f64],
    orbital_freq: f64,
    alpha_scale: f64,
    config: &ZdResonanceConfig,
) -> [f64; 3] {
    let bands = compute_resonance_bands_from_eigenvalues(eigenvalues, orbital_freq, config);
    let total = total_resonance_coupling_from_bands(&bands);
    let positive_eigenvalues: Vec<f64> = eigenvalues
        .iter()
        .copied()
        .filter(|&value| value > 1e-10)
        .take(3)
        .collect();

    if positive_eigenvalues.is_empty() || total < 1e-15 {
        return [0.0; 3];
    }

    let sum_positive: f64 = positive_eigenvalues.iter().sum();
    let scale = alpha_scale * total;
    let mut result = [scale / 3.0; 3];
    if sum_positive > 0.0 {
        for (index, &eigenvalue) in positive_eigenvalues.iter().enumerate() {
            result[index] = scale * eigenvalue / sum_positive;
        }
    }
    result
}

pub fn compute_resonance_report_from_control_report(
    report: &HigherCdControlReport,
    orbital_freq: f64,
    alpha_scale: f64,
    config: &ZdResonanceConfig,
) -> ZdResonanceReport {
    let bands = compute_resonance_bands_from_eigenvalues(
        &report.spectrum_report.eigenvalues,
        orbital_freq,
        config,
    );
    let total_coupling = total_resonance_coupling_from_bands(&bands);
    let perturbation = resonance_modulated_perturbation_from_eigenvalues(
        &report.spectrum_report.eigenvalues,
        orbital_freq,
        alpha_scale,
        config,
    );

    ZdResonanceReport {
        algebra_name: report.summary.algebra_name.clone(),
        ambient_dim: report.summary.ambient_dim,
        orbital_frequency: orbital_freq,
        alpha_scale,
        total_coupling,
        perturbation,
        bands,
    }
}

pub fn compute_higher_cd_control_report(spec: &HigherCdControlSpec) -> HigherCdControlReport {
    let basis_report = compute_higher_cd_basis_report(spec);
    let spectrum_report = compute_zd_graph_spectrum(spec.ambient_dim);
    let mut report = HigherCdControlReport {
        spec: spec.clone(),
        basis_report,
        spectrum_report,
        summary: HigherCdControlSummary {
            algebra_name: String::new(),
            ambient_dim: 0,
            requested_rank: 0,
            actual_rank: 0,
            assessor_count: 0,
            edge_count: 0,
            connected_components: 0,
            positive_eigenvalue_count: 0,
            leading_singular_value: 0.0,
            trailing_singular_value: 0.0,
        },
    };
    report.summary = summarize_control_report(&report);
    report
}

pub fn default_pathion_control_report() -> PathionControlReport {
    compute_higher_cd_control_report(&HigherCdControlSpec::pathion32())
}

pub fn default_pathion_resonance_report(
    orbital_freq: f64,
    alpha_scale: f64,
    config: &ZdResonanceConfig,
) -> PathionResonanceReport {
    let report = default_pathion_control_report();
    compute_resonance_report_from_control_report(&report, orbital_freq, alpha_scale, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pathion32_spec_defaults() {
        let spec = HigherCdControlSpec::pathion32();
        assert_eq!(spec.ambient_dim, 32);
        assert_eq!(spec.vk_rank, 20);
        assert_eq!(spec.algebra_name(), "Pathion");
    }

    #[test]
    fn test_pathion_basis_report_matches_extract_vk_basis() {
        let spec = HigherCdControlSpec::pathion32();
        let report = compute_higher_cd_basis_report(&spec);
        let (basis, singular_values, assessors) = extract_vk_basis(32, 20);

        assert_eq!(report.actual_rank, basis.nrows());
        assert_eq!(report.basis_cols, basis.ncols());
        assert_eq!(report.assessor_count, assessors.len());
        assert_eq!(report.singular_values, singular_values);
        assert!(report.leading_singular_value >= report.trailing_singular_value);
    }

    #[test]
    fn test_pathion_spectrum_report_invariants() {
        let report = compute_zd_graph_spectrum(32);
        assert_eq!(report.ambient_dim, 32);
        assert_eq!(report.eigenvalues.len(), 32);
        assert!(report.eigenvalues.iter().all(|&value| value >= -1e-10));
        assert!(report.n_components >= 2);
        assert!(report.edge_count > 0);
        assert!(report.degree_max >= report.degree_min);
    }

    #[test]
    fn test_default_pathion_control_report_summary() {
        let report = default_pathion_control_report();
        assert_eq!(report.summary.algebra_name, "Pathion");
        assert_eq!(report.summary.ambient_dim, 32);
        assert_eq!(report.summary.requested_rank, 20);
        assert_eq!(
            report.summary.connected_components,
            report.spectrum_report.n_components
        );
        assert_eq!(
            report.summary.assessor_count,
            report.basis_report.assessor_count
        );
    }

    #[test]
    fn test_default_pathion_resonance_report_has_bands() {
        let config = ZdResonanceConfig::default();
        let report = default_pathion_resonance_report(1.0, 1e-6, &config);
        assert_eq!(report.algebra_name, "Pathion");
        assert_eq!(report.ambient_dim, 32);
        assert!(report.total_coupling >= 0.0);
        assert!(
            report
                .bands
                .iter()
                .all(|band| band.coupling_strength >= 0.0)
        );
    }

    #[test]
    fn test_resonance_perturbation_scales_with_alpha() {
        let config = ZdResonanceConfig::default();
        let report = default_pathion_control_report();
        let p1 = compute_resonance_report_from_control_report(&report, 1.0, 1e-6, &config);
        let p2 = compute_resonance_report_from_control_report(&report, 1.0, 2e-6, &config);

        for index in 0..3 {
            if p1.perturbation[index].abs() > 1e-20 {
                let ratio = p2.perturbation[index] / p1.perturbation[index];
                assert!((ratio - 2.0).abs() < 1e-10);
            }
        }
    }
}
