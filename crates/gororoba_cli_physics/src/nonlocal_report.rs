use algebra_analysis::{crystal_bands::dim_spectrum_summary, reggiani::partner_graph_degeneracies};
use cosmology_core::sersic::sersic_profile_2d;
use materials_core::SyntheticCouplingModel;
use optics_core::absorber_benchmark::{
    BenchmarkResult, ComparativeBenchmark, ProjectionGateResult,
};
use quantum_core::{
    BravaisLattice2D, Hopping, InversionBreakingParams, MagnonicTBParams, OrbitalSite,
    TightBindingModel, Valley, Vec2, compute_magnonic_bands, valley_chern_number,
};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NonlocalDegeneracyRow {
    pub eigenvalue: f64,
    pub multiplicity: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NonlocalTopologyResult {
    pub name: String,
    pub isolation_ratio: f64,
    pub crosstalk_db: f64,
    pub spectral_gap: f64,
    pub n_components: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NonlocalSpectralCrosscheck {
    pub reggiani_partner_degeneracies: Vec<NonlocalDegeneracyRow>,
    pub zd_flat_band_fraction: f64,
    pub zd_crystal_n_components: usize,
    pub zd_crystal_total_nodes: usize,
    pub zd_crystal_total_eigenvalues: usize,
    pub zd_crystal_distinct_eigenvalue_count: usize,
    pub magnonic_flat_band_indices: Vec<usize>,
    pub magnonic_min_bandwidth_ghz: f64,
    pub magnonic_band_cherns: Vec<f64>,
    pub magnonic_valley_cherns_k: Vec<f64>,
    pub graphene_valley_chern_k: f64,
    pub graphene_valley_chern_kprime: f64,
    pub sersic_index_n: f64,
    pub sersic_effective_radius: f64,
    pub sersic_layout_min: f64,
    pub sersic_layout_max: f64,
    pub sersic_gating_role: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NonlocalBenchmarkReport {
    pub calibration_id: String,
    pub platform: String,
    pub citation_key: String,
    pub citation_url: String,
    pub average_coupling: f64,
    pub projection_gate_pass: bool,
    pub projection_gate_verdict: String,
    pub zd_isolation: f64,
    pub best_local_isolation: f64,
    pub dominance_margin: f64,
    pub zd_crosstalk_db: f64,
    pub topologies: Vec<NonlocalTopologyResult>,
    pub spectral_crosscheck: NonlocalSpectralCrosscheck,
}

pub fn build_nonlocal_benchmark_report(
    model: &SyntheticCouplingModel,
    benchmark: &ComparativeBenchmark,
    gate: &ProjectionGateResult,
    average_coupling: f64,
) -> NonlocalBenchmarkReport {
    NonlocalBenchmarkReport {
        calibration_id: model.calibration.id.clone(),
        platform: model.calibration.platform.clone(),
        citation_key: model.calibration.citation_key.clone(),
        citation_url: model.calibration.citation_url.clone(),
        average_coupling,
        projection_gate_pass: gate.pass,
        projection_gate_verdict: format!("{:?}", gate.verdict),
        zd_isolation: gate.zd_isolation,
        best_local_isolation: gate.best_local_isolation,
        dominance_margin: gate.dominance_margin,
        zd_crosstalk_db: gate.zd_crosstalk_db,
        topologies: benchmark
            .results
            .iter()
            .map(topology_result_from_benchmark)
            .collect(),
        spectral_crosscheck: build_spectral_crosscheck(model),
    }
}

pub fn render_nonlocal_benchmark_report_toml(
    report: &NonlocalBenchmarkReport,
) -> Result<String, String> {
    toml::to_string_pretty(report)
        .map_err(|err| format!("failed to serialize nonlocal benchmark report: {err}"))
}

pub fn write_nonlocal_benchmark_report(
    path: &Path,
    report: &NonlocalBenchmarkReport,
) -> Result<(), String> {
    let rendered = render_nonlocal_benchmark_report_toml(report)?;
    fs::write(path, rendered).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn topology_result_from_benchmark(result: &BenchmarkResult) -> NonlocalTopologyResult {
    NonlocalTopologyResult {
        name: result.topology_name.clone(),
        isolation_ratio: result.isolation_ratio,
        crosstalk_db: result.crosstalk_db,
        spectral_gap: result.spectral_gap,
        n_components: result.n_components,
    }
}

fn build_spectral_crosscheck(model: &SyntheticCouplingModel) -> NonlocalSpectralCrosscheck {
    let zd_summary = dim_spectrum_summary(16, 1.0e-9);
    let magnonic = compute_magnonic_bands(
        &MagnonicTBParams::kaman_default(),
        &InversionBreakingParams::kaman_default(),
        18,
        9,
        1000.0,
    );
    let (graphene_valley_chern_k, graphene_valley_chern_kprime) = graphene_valley_reference();
    let (sersic_layout_min, sersic_layout_max) = sersic_layout_span(model);

    NonlocalSpectralCrosscheck {
        reggiani_partner_degeneracies: partner_graph_degeneracies(1.0e-9)
            .into_iter()
            .map(|(eigenvalue, multiplicity)| NonlocalDegeneracyRow {
                eigenvalue,
                multiplicity,
            })
            .collect(),
        zd_flat_band_fraction: zd_summary.flat_band_fraction,
        zd_crystal_n_components: zd_summary.n_components,
        zd_crystal_total_nodes: zd_summary.total_nodes,
        zd_crystal_total_eigenvalues: zd_summary.total_eigenvalues,
        zd_crystal_distinct_eigenvalue_count: zd_summary.distinct_eigenvalue_count,
        magnonic_flat_band_indices: magnonic.flat_band_indices,
        magnonic_min_bandwidth_ghz: magnonic
            .bandwidths
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min),
        magnonic_band_cherns: magnonic.band_cherns,
        magnonic_valley_cherns_k: magnonic.valley_cherns_k,
        graphene_valley_chern_k,
        graphene_valley_chern_kprime,
        sersic_index_n: 4.0,
        sersic_effective_radius: 4.0,
        sersic_layout_min,
        sersic_layout_max,
        sersic_gating_role: "non_gating_layout_sensitivity".to_string(),
    }
}

fn sersic_layout_span(model: &SyntheticCouplingModel) -> (f64, f64) {
    let mut values = Vec::new();
    for node in &model.topology.nodes {
        let radial_shell = 1.0 + node.boxkite_id as f64;
        let azimuth_offset = (node.index % 6) as f64 / 6.0;
        let radius = radial_shell + azimuth_offset;
        values.push(sersic_profile_2d(radius, 4.0, 4.0));
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (min, max)
}

fn graphene_valley_reference() -> (f64, f64) {
    let lattice = BravaisLattice2D::hexagonal(1.0);
    let s3 = 3.0_f64.sqrt();
    let model = TightBindingModel {
        lattice: lattice.clone(),
        orbitals: vec![
            OrbitalSite {
                position: Vec2::zero(),
                label: "A".to_string(),
                on_site_energy: 0.0,
            },
            OrbitalSite {
                position: Vec2::new(0.0, 1.0 / s3),
                label: "B".to_string(),
                on_site_energy: 0.0,
            },
        ],
        hoppings: vec![
            Hopping {
                from: 0,
                to: 1,
                cell_offset: [0, 0],
                amplitude: faer::complex_native::c64::new(-1.0, 0.0),
            },
            Hopping {
                from: 0,
                to: 1,
                cell_offset: [0, -1],
                amplitude: faer::complex_native::c64::new(-1.0, 0.0),
            },
            Hopping {
                from: 0,
                to: 1,
                cell_offset: [1, -1],
                amplitude: faer::complex_native::c64::new(-1.0, 0.0),
            },
        ],
    };
    (
        valley_chern_number(&model, 0, 15, Valley::K),
        valley_chern_number(&model, 0, 15, Valley::KPrime),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use materials_core::{M3ProjectionConfig, MaterialCalibrationRecord};
    use optics_core::{
        absorber_benchmark::{CouplingTopology, ProjectionGate, run_benchmark},
        tcmt::KerrCavity,
    };

    fn sample_calibration() -> MaterialCalibrationRecord {
        MaterialCalibrationRecord {
            id: "unit_test".to_string(),
            platform: "test_platform".to_string(),
            carrier_material: "test".to_string(),
            data_provenance_class: "measured".to_string(),
            citation_key: "TEST".to_string(),
            citation_url: "https://example.com".to_string(),
            coupling_scale: 0.25,
            modulation_depth: 0.35,
            loss_tangent: 0.02,
            phase_bias_rad: 0.15,
            modulation_frequency_hz: 2.08e9,
            quality_factor: 4100.0,
            nonlocality_score: 0.8,
            notes: "unit test".to_string(),
        }
    }

    fn benchmark_cavity() -> KerrCavity {
        KerrCavity::new(
            2.0 * std::f64::consts::PI * 193.0e12,
            1.0e6,
            5.0e5,
            1.45,
            2.6e-20,
            1.0e-18,
        )
    }

    fn benchmark_bundle(
        model: &SyntheticCouplingModel,
    ) -> (ComparativeBenchmark, ProjectionGateResult, f64) {
        let average_coupling = model.average_nonzero_coupling();
        let suite = vec![
            model.to_coupling_topology("unit-test-floquet"),
            CouplingTopology::ring(42, average_coupling),
            CouplingTopology::chain(42, average_coupling),
            CouplingTopology::complete(42, average_coupling),
            CouplingTopology::cross_cluster_bridges(7, 6, average_coupling),
        ];
        let benchmark = run_benchmark(
            &suite,
            &benchmark_cavity(),
            1.0e9,
            &[0, 1, 2, 3, 4, 5],
            1.0e-3,
            1.0e-12,
            500,
        );
        let gate = ProjectionGate::c010_default().evaluate(&benchmark);
        (benchmark, gate, average_coupling)
    }

    #[test]
    fn nonlocal_report_serializes_rich_sidecars() {
        let model = SyntheticCouplingModel::build(
            &M3ProjectionConfig::c010_hybrid_default(),
            &sample_calibration(),
        )
        .expect("synthetic model");
        let (benchmark, gate, average_coupling) = benchmark_bundle(&model);
        let report = build_nonlocal_benchmark_report(&model, &benchmark, &gate, average_coupling);

        assert!(report.projection_gate_pass);
        assert_eq!(report.topologies.len(), 5);
        assert_eq!(
            report
                .spectral_crosscheck
                .reggiani_partner_degeneracies
                .len(),
            5
        );
        assert_eq!(report.spectral_crosscheck.zd_crystal_total_nodes, 42);
        assert!(
            !report
                .spectral_crosscheck
                .magnonic_flat_band_indices
                .is_empty()
        );
        assert_eq!(
            report.spectral_crosscheck.sersic_gating_role,
            "non_gating_layout_sensitivity"
        );

        let rendered =
            render_nonlocal_benchmark_report_toml(&report).expect("serialize benchmark report");
        assert!(rendered.contains("zd_crystal_total_nodes = 42"));
        assert!(rendered.contains("magnonic_band_cherns"));
        assert!(rendered.contains("sersic_gating_role = \"non_gating_layout_sensitivity\""));
    }
}
