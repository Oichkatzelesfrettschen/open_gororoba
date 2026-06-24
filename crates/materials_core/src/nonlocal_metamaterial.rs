//! Non-local algebraic metamaterial synthesis for C-010.
//!
//! This module lifts the sedenion `m3` obstruction tensor into a 42-assessor
//! synthetic-coupling design space and emits two backend realizations:
//! - a topolectrical / LC-network admittance model
//! - a Floquet-effective coupling model for TCMT benchmarking
//!
//! The canonical topology is the de Marrais box-kite partition of the 42
//! primitive sedenion assessors into 7 disconnected `K6` cliques.

use algebra_analysis::{
    boxkites::{Assessor, cached_sedenion_boxkites, primitive_assessors},
    homotopy_algebra::SedenionAInfinity,
};
use num_complex::Complex64;
use optics_core::absorber_benchmark::CouplingTopology;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::Path};

/// Metadata for one assessor node in the canonical 42-node topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssessorNode {
    pub index: usize,
    pub low: usize,
    pub high: usize,
    pub boxkite_id: usize,
    pub strut_signature: usize,
}

impl AssessorNode {
    fn as_assessor(self) -> Assessor {
        Assessor::new(self.low, self.high)
    }
}

/// Canonical 42-assessor topology and clique mask.
#[derive(Debug, Clone)]
pub struct AssessorTopology {
    pub nodes: Vec<AssessorNode>,
    pub adjacency_mask: Vec<Vec<bool>>,
    pub boxkite_sizes: Vec<usize>,
    pub reggiani_partner_graph_order: usize,
}

impl AssessorTopology {
    /// Build the canonical 7 x K6 assessor topology.
    pub fn canonical() -> Result<Self, String> {
        let primitive = primitive_assessors();
        let mut primitive_set = HashMap::with_capacity(primitive.len());
        for assessor in primitive {
            primitive_set.insert((assessor.low, assessor.high), assessor);
        }

        let mut nodes = Vec::with_capacity(42);
        let mut boxkite_sizes = Vec::new();

        let mut ordered_boxkites = cached_sedenion_boxkites().clone();
        ordered_boxkites.sort_by_key(|boxkite| boxkite.strut_signature);

        for (boxkite_idx, boxkite) in ordered_boxkites.iter().enumerate() {
            let mut assessors = boxkite.assessors.clone();
            assessors.sort_by_key(|assessor| (assessor.low, assessor.high));
            boxkite_sizes.push(assessors.len());

            for assessor in assessors {
                if !primitive_set.contains_key(&(assessor.low, assessor.high)) {
                    return Err(format!(
                        "box-kite assessor ({}, {}) missing from primitive assessor set",
                        assessor.low, assessor.high
                    ));
                }
                let index = nodes.len();
                nodes.push(AssessorNode {
                    index,
                    low: assessor.low,
                    high: assessor.high,
                    boxkite_id: boxkite_idx,
                    strut_signature: boxkite.strut_signature,
                });
            }
        }

        if nodes.len() != 42 {
            return Err(format!("expected 42 assessor nodes, found {}", nodes.len()));
        }
        if boxkite_sizes.len() != 7 || boxkite_sizes.iter().any(|&n| n != 6) {
            return Err(format!(
                "expected 7 box-kites of size 6, found {:?}",
                boxkite_sizes
            ));
        }

        let n = nodes.len();
        let mut adjacency_mask = vec![vec![false; n]; n];
        for i in 0..n {
            for j in 0..n {
                adjacency_mask[i][j] = i != j && nodes[i].boxkite_id == nodes[j].boxkite_id;
            }
        }

        Ok(Self {
            nodes,
            adjacency_mask,
            boxkite_sizes,
            reggiani_partner_graph_order: 84,
        })
    }

    pub fn edge_count(&self) -> usize {
        let mut edges = 0usize;
        for i in 0..self.nodes.len() {
            for j in (i + 1)..self.nodes.len() {
                if self.adjacency_mask[i][j] {
                    edges += 1;
                }
            }
        }
        edges
    }
}

/// Configuration for lifting the sedenion `m3` tensor into assessor weights.
#[derive(Debug, Clone)]
pub struct M3ProjectionConfig {
    pub dim: usize,
    pub edge_floor: f64,
    pub epsilon_scale: f64,
    pub mu_scale: f64,
    pub phase_bias_scale_rad: f64,
}

impl M3ProjectionConfig {
    pub fn c010_hybrid_default() -> Self {
        Self {
            dim: 16,
            edge_floor: 0.2,
            epsilon_scale: 1.0,
            mu_scale: 0.6,
            phase_bias_scale_rad: std::f64::consts::FRAC_PI_3,
        }
    }
}

/// Literature-backed calibration record loaded from CSV.
#[derive(Debug, Clone, Deserialize)]
pub struct MaterialCalibrationRecord {
    pub id: String,
    pub platform: String,
    pub carrier_material: String,
    pub data_provenance_class: String,
    pub citation_key: String,
    pub citation_url: String,
    pub coupling_scale: f64,
    pub modulation_depth: f64,
    pub loss_tangent: f64,
    pub phase_bias_rad: f64,
    pub modulation_frequency_hz: f64,
    pub quality_factor: f64,
    pub nonlocality_score: f64,
    pub notes: String,
}

/// LC / topolectrical reduction of the synthetic tensor.
#[derive(Debug, Clone)]
pub struct LcAdmittanceModel {
    pub admittance_matrix: Vec<Vec<Complex64>>,
    pub capacitance_matrix_f: Vec<Vec<f64>>,
    pub inductance_matrix_h: Vec<Vec<f64>>,
    pub conductance_matrix_s: Vec<Vec<f64>>,
    pub effective_real_coupling: Vec<Vec<f64>>,
}

/// Floquet-effective reduction of the synthetic tensor.
#[derive(Debug, Clone)]
pub struct FloquetEffectiveModel {
    pub effective_tensor: Vec<Vec<Complex64>>,
    pub effective_real_coupling: Vec<Vec<f64>>,
    pub hermitian_real_coupling: Vec<Vec<f64>>,
}

/// Full synthetic-coupling model emitted from the `m3` lift.
#[derive(Debug, Clone)]
pub struct SyntheticCouplingModel {
    pub topology: AssessorTopology,
    pub config: M3ProjectionConfig,
    pub calibration: MaterialCalibrationRecord,
    pub basis_obstruction_matrix: Vec<Vec<f64>>,
    pub normalized_assessor_weights: Vec<Vec<f64>>,
    pub synthetic_permittivity: Vec<Vec<Complex64>>,
    pub synthetic_permeability: Vec<Vec<Complex64>>,
    pub lc_admittance: LcAdmittanceModel,
    pub floquet_effective: FloquetEffectiveModel,
}

/// Compact summary of the assessor-to-tensor lift used by thesis support lanes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SyntheticCouplingReport {
    pub calibration_id: String,
    pub platform: String,
    pub dim: usize,
    pub node_count: usize,
    pub masked_edge_count: usize,
    pub boxkite_count: usize,
    pub boxkite_sizes: Vec<usize>,
    pub reggiani_partner_graph_order: usize,
    pub average_nonzero_coupling: f64,
    pub mean_active_weight: f64,
    pub max_active_weight: f64,
}

impl SyntheticCouplingModel {
    /// Build the full non-local synthetic-coupling model.
    pub fn build(
        config: &M3ProjectionConfig,
        calibration: &MaterialCalibrationRecord,
    ) -> Result<Self, String> {
        let topology = AssessorTopology::canonical()?;
        let basis_obstruction_matrix = build_basis_obstruction_matrix(config.dim)?;
        let normalized_assessor_weights =
            project_assessor_weights(&topology, &basis_obstruction_matrix, config.edge_floor)?;

        let (synthetic_permittivity, synthetic_permeability) =
            build_synthetic_tensors(&topology, &normalized_assessor_weights, config, calibration);
        let lc_admittance =
            build_lc_admittance_model(&topology, &normalized_assessor_weights, calibration);
        let floquet_effective = build_floquet_effective_model(
            &topology,
            &synthetic_permittivity,
            &synthetic_permeability,
            calibration,
        );

        Ok(Self {
            topology,
            config: config.clone(),
            calibration: calibration.clone(),
            basis_obstruction_matrix,
            normalized_assessor_weights,
            synthetic_permittivity,
            synthetic_permeability,
            lc_admittance,
            floquet_effective,
        })
    }

    /// Convert the Floquet-effective matrix into the real coupling surface used by optics_core.
    pub fn to_coupling_topology(&self, name: &str) -> CouplingTopology {
        CouplingTopology::from_adjacency_matrix(
            name,
            self.topology.nodes.len(),
            self.floquet_effective.hermitian_real_coupling.clone(),
        )
    }

    pub fn average_nonzero_coupling(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0usize;
        for i in 0..self.floquet_effective.hermitian_real_coupling.len() {
            for j in (i + 1)..self.floquet_effective.hermitian_real_coupling.len() {
                let value = self.floquet_effective.hermitian_real_coupling[i][j];
                if value > 0.0 {
                    sum += value;
                    count += 1;
                }
            }
        }
        if count == 0 { 0.0 } else { sum / count as f64 }
    }

    /// Export a typed summary of the masked 42-assessor synthetic coupling lift.
    pub fn report(&self) -> SyntheticCouplingReport {
        let mut weight_sum = 0.0;
        let mut weight_count = 0usize;
        let mut max_active_weight = 0.0_f64;

        for i in 0..self.normalized_assessor_weights.len() {
            for j in (i + 1)..self.normalized_assessor_weights.len() {
                let value = self.normalized_assessor_weights[i][j];
                if value > 0.0 {
                    weight_sum += value;
                    weight_count += 1;
                    max_active_weight = max_active_weight.max(value);
                }
            }
        }

        SyntheticCouplingReport {
            calibration_id: self.calibration.id.clone(),
            platform: self.calibration.platform.clone(),
            dim: self.config.dim,
            node_count: self.topology.nodes.len(),
            masked_edge_count: self.topology.edge_count(),
            boxkite_count: self.topology.boxkite_sizes.len(),
            boxkite_sizes: self.topology.boxkite_sizes.clone(),
            reggiani_partner_graph_order: self.topology.reggiani_partner_graph_order,
            average_nonzero_coupling: self.average_nonzero_coupling(),
            mean_active_weight: if weight_count == 0 {
                0.0
            } else {
                weight_sum / weight_count as f64
            },
            max_active_weight,
        }
    }
}

/// Load calibration rows from a CSV file.
pub fn load_calibration_records(
    path: impl AsRef<Path>,
) -> Result<Vec<MaterialCalibrationRecord>, String> {
    let path = path.as_ref();
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("toml"))
    {
        return load_calibration_records_from_project_csv_toml(path);
    }

    let mut reader = csv::Reader::from_path(path)
        .map_err(|err| format!("failed to open calibration CSV {}: {err}", path.display()))?;
    let mut rows = Vec::new();
    for row in reader.deserialize() {
        let record: MaterialCalibrationRecord =
            row.map_err(|err| format!("failed to parse calibration CSV row: {err}"))?;
        rows.push(record);
    }
    if rows.is_empty() {
        return Err("calibration CSV contained no records".to_string());
    }
    Ok(rows)
}

fn load_calibration_records_from_project_csv_toml(
    path: &Path,
) -> Result<Vec<MaterialCalibrationRecord>, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|err| format!("failed to read calibration TOML {}: {err}", path.display()))?;
    let header = project_csv_string_block(path, &raw, "header")?;
    let rows = project_csv_rows(path, &raw)?;

    let mut columns = std::collections::BTreeMap::new();
    for (index, name) in header.iter().enumerate() {
        columns.insert(name.clone(), index);
    }

    let mut records = Vec::new();
    for row in rows {
        records.push(MaterialCalibrationRecord {
            id: project_csv_string(path, &columns, &row, "id")?,
            platform: project_csv_string(path, &columns, &row, "platform")?,
            carrier_material: project_csv_string(path, &columns, &row, "carrier_material")?,
            data_provenance_class: project_csv_string(
                path,
                &columns,
                &row,
                "data_provenance_class",
            )?,
            citation_key: project_csv_string(path, &columns, &row, "citation_key")?,
            citation_url: project_csv_string(path, &columns, &row, "citation_url")?,
            coupling_scale: project_csv_f64(path, &columns, &row, "coupling_scale")?,
            modulation_depth: project_csv_f64(path, &columns, &row, "modulation_depth")?,
            loss_tangent: project_csv_f64(path, &columns, &row, "loss_tangent")?,
            phase_bias_rad: project_csv_f64(path, &columns, &row, "phase_bias_rad")?,
            modulation_frequency_hz: project_csv_f64(
                path,
                &columns,
                &row,
                "modulation_frequency_hz",
            )?,
            quality_factor: project_csv_f64(path, &columns, &row, "quality_factor")?,
            nonlocality_score: project_csv_f64(path, &columns, &row, "nonlocality_score")?,
            notes: project_csv_string(path, &columns, &row, "notes")?,
        });
    }
    if records.is_empty() {
        return Err("calibration TOML contained no records".to_string());
    }
    Ok(records)
}

fn project_csv_string_block(
    path: &Path,
    raw: &str,
    block_name: &str,
) -> Result<Vec<String>, String> {
    let start = format!("{block_name} = [");
    let mut in_block = false;
    let mut values = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if !in_block {
            in_block = trimmed == start;
            continue;
        }
        if trimmed == "]" {
            return Ok(values);
        }
        if let Some(value) = parse_project_csv_string_cell(trimmed) {
            values.push(value);
        }
    }
    Err(format!(
        "calibration TOML {} missing {block_name} block",
        path.display()
    ))
}

fn project_csv_rows(path: &Path, raw: &str) -> Result<Vec<Vec<String>>, String> {
    let mut in_rows = false;
    let mut current_row: Option<Vec<String>> = None;
    let mut rows = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if !in_rows {
            in_rows = trimmed == "rows = [";
            continue;
        }
        match trimmed {
            "]" => return Ok(rows),
            "[" => {
                if current_row.is_some() {
                    return Err(format!(
                        "calibration TOML {} has nested dataset.rows entry",
                        path.display()
                    ));
                }
                current_row = Some(Vec::new());
            }
            "]," => {
                let row = current_row.take().ok_or_else(|| {
                    format!(
                        "calibration TOML {} closes a row before opening it",
                        path.display()
                    )
                })?;
                rows.push(row);
            }
            _ => {
                if let Some(value) = parse_project_csv_string_cell(trimmed) {
                    let row = current_row.as_mut().ok_or_else(|| {
                        format!(
                            "calibration TOML {} has a value outside a row",
                            path.display()
                        )
                    })?;
                    row.push(value);
                }
            }
        }
    }
    Err(format!(
        "calibration TOML {} missing dataset.rows block",
        path.display()
    ))
}

fn parse_project_csv_string_cell(line: &str) -> Option<String> {
    let trimmed = line.trim_end_matches(',');
    let mut chars = trimmed.chars();
    if chars.next()? != '"' {
        return None;
    }
    let mut value = String::new();
    let mut escaped = false;
    for ch in chars {
        if escaped {
            value.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(value);
        } else {
            value.push(ch);
        }
    }
    None
}

fn project_csv_string(
    path: &Path,
    columns: &std::collections::BTreeMap<String, usize>,
    row: &[String],
    name: &str,
) -> Result<String, String> {
    let index = *columns
        .get(name)
        .ok_or_else(|| format!("calibration TOML {} missing column {name}", path.display()))?;
    row.get(index).cloned().ok_or_else(|| {
        format!(
            "calibration TOML {} missing string value for column {name}",
            path.display()
        )
    })
}

fn project_csv_f64(
    path: &Path,
    columns: &std::collections::BTreeMap<String, usize>,
    row: &[String],
    name: &str,
) -> Result<f64, String> {
    let raw = project_csv_string(path, columns, row, name)?;
    raw.parse::<f64>().map_err(|err| {
        format!(
            "calibration TOML {} has invalid f64 in column {name}: {err}",
            path.display()
        )
    })
}

/// Select one calibration row by id.
pub fn find_calibration_record<'a>(
    records: &'a [MaterialCalibrationRecord],
    calibration_id: &str,
) -> Result<&'a MaterialCalibrationRecord, String> {
    records
        .iter()
        .find(|record| record.id == calibration_id)
        .ok_or_else(|| format!("unknown calibration id: {calibration_id}"))
}

fn build_basis_obstruction_matrix(dim: usize) -> Result<Vec<Vec<f64>>, String> {
    if dim != 16 {
        return Err(format!(
            "non-local metamaterial lift currently supports dim=16, got {dim}"
        ));
    }

    let ainf = SedenionAInfinity::new(dim);
    let mut matrix = vec![vec![0.0; dim]; dim];
    for (i, row) in matrix.iter_mut().enumerate() {
        let e_i = basis_vector(dim, i);
        for (j, cell) in row.iter_mut().enumerate() {
            let e_j = basis_vector(dim, j);
            let mut total_sq = 0.0;
            for k in 0..dim {
                let e_k = basis_vector(dim, k);
                let assoc = ainf.m3(&e_i, &e_k, &e_j);
                total_sq += assoc.iter().map(|value| value * value).sum::<f64>();
            }
            *cell = total_sq;
        }
    }
    Ok(matrix)
}

fn basis_vector(dim: usize, index: usize) -> Vec<f64> {
    let mut vector = vec![0.0; dim];
    vector[index] = 1.0;
    vector
}

fn project_assessor_weights(
    topology: &AssessorTopology,
    basis_obstruction_matrix: &[Vec<f64>],
    edge_floor: f64,
) -> Result<Vec<Vec<f64>>, String> {
    if !(0.0..1.0).contains(&edge_floor) {
        return Err(format!("edge_floor must lie in [0, 1), got {edge_floor}"));
    }

    let n = topology.nodes.len();
    let mut raw = vec![vec![0.0; n]; n];
    let mut max_value = 0.0f64;

    for i in 0..n {
        let (inclusive_head, tail) = raw.split_at_mut(i + 1);
        let row = &mut inclusive_head[i];
        for (offset, symmetric_row) in tail.iter_mut().enumerate() {
            let j = i + 1 + offset;
            if !topology.adjacency_mask[i][j] {
                continue;
            }
            let left = topology.nodes[i].as_assessor();
            let right = topology.nodes[j].as_assessor();
            let combos = [
                basis_obstruction_matrix[left.low][right.low],
                basis_obstruction_matrix[left.low][right.high],
                basis_obstruction_matrix[left.high][right.low],
                basis_obstruction_matrix[left.high][right.high],
            ];
            let average = combos.iter().sum::<f64>() / combos.len() as f64;
            row[j] = average;
            symmetric_row[i] = average;
            max_value = max_value.max(average);
        }
    }

    if max_value <= 0.0 {
        return Err("assessor projection produced a zero obstruction matrix".to_string());
    }

    let mut normalized = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j || !topology.adjacency_mask[i][j] {
                continue;
            }
            let scaled = raw[i][j] / max_value;
            normalized[i][j] = edge_floor + (1.0 - edge_floor) * scaled;
        }
    }
    Ok(normalized)
}

fn build_synthetic_tensors(
    topology: &AssessorTopology,
    weights: &[Vec<f64>],
    config: &M3ProjectionConfig,
    calibration: &MaterialCalibrationRecord,
) -> (Vec<Vec<Complex64>>, Vec<Vec<Complex64>>) {
    let n = topology.nodes.len();
    let mut epsilon = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    let mut mu = vec![vec![Complex64::new(0.0, 0.0); n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            if !topology.adjacency_mask[i][j] {
                continue;
            }
            let weight = weights[i][j];
            let pair_phase = xor_phase(topology.nodes[i], topology.nodes[j]);
            let phase = calibration.phase_bias_rad + config.phase_bias_scale_rad * pair_phase;
            let amplitude = calibration.coupling_scale * weight;

            let eps = Complex64::from_polar(
                config.epsilon_scale * amplitude * calibration.nonlocality_score,
                phase,
            );
            let permeability = Complex64::from_polar(
                config.mu_scale * amplitude * (1.0 + calibration.modulation_depth * weight),
                -0.5 * phase,
            );

            epsilon[i][j] = eps;
            epsilon[j][i] = eps.conj();
            mu[i][j] = permeability;
            mu[j][i] = permeability.conj();
        }
    }

    (epsilon, mu)
}

fn xor_phase(left: AssessorNode, right: AssessorNode) -> f64 {
    let xor_value = (left.low ^ left.high ^ right.low ^ right.high) & 0x7;
    xor_value as f64 / 7.0
}

fn build_lc_admittance_model(
    topology: &AssessorTopology,
    weights: &[Vec<f64>],
    calibration: &MaterialCalibrationRecord,
) -> LcAdmittanceModel {
    let n = topology.nodes.len();
    let mut admittance_matrix = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    let mut capacitance_matrix_f = vec![vec![0.0; n]; n];
    let mut inductance_matrix_h = vec![vec![0.0; n]; n];
    let mut conductance_matrix_s = vec![vec![0.0; n]; n];
    let mut effective_real_coupling = vec![vec![0.0; n]; n];
    let omega = std::f64::consts::TAU * calibration.modulation_frequency_hz.max(1.0);
    let benchmark_scale = calibration.modulation_frequency_hz * 5.0;

    for i in 0..n {
        for j in (i + 1)..n {
            if !topology.adjacency_mask[i][j] {
                continue;
            }
            let weight = weights[i][j];
            let coupling = calibration.coupling_scale * weight;
            let capacitance = 1.0e-12 * calibration.modulation_depth.max(1.0e-6) * weight;
            let inductance = 1.0 / (omega * coupling.max(1.0e-6));
            let conductance = 1.0e-3 * calibration.loss_tangent * weight;
            let susceptance = omega * capacitance - 1.0 / (omega * inductance.max(1.0e-18));
            let admittance = Complex64::new(conductance, susceptance);

            admittance_matrix[i][j] = admittance;
            admittance_matrix[j][i] = admittance;
            capacitance_matrix_f[i][j] = capacitance;
            capacitance_matrix_f[j][i] = capacitance;
            inductance_matrix_h[i][j] = inductance;
            inductance_matrix_h[j][i] = inductance;
            conductance_matrix_s[i][j] = conductance;
            conductance_matrix_s[j][i] = conductance;
            effective_real_coupling[i][j] = benchmark_scale * admittance.norm();
            effective_real_coupling[j][i] = benchmark_scale * admittance.norm();
        }
    }

    LcAdmittanceModel {
        admittance_matrix,
        capacitance_matrix_f,
        inductance_matrix_h,
        conductance_matrix_s,
        effective_real_coupling,
    }
}

fn build_floquet_effective_model(
    topology: &AssessorTopology,
    epsilon: &[Vec<Complex64>],
    mu: &[Vec<Complex64>],
    calibration: &MaterialCalibrationRecord,
) -> FloquetEffectiveModel {
    let n = topology.nodes.len();
    let mut effective_tensor = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    let mut effective_real_coupling = vec![vec![0.0; n]; n];
    let mut hermitian_real_coupling = vec![vec![0.0; n]; n];
    let benchmark_scale = calibration.modulation_frequency_hz * 5.0;

    for i in 0..n {
        for j in (i + 1)..n {
            if !topology.adjacency_mask[i][j] {
                continue;
            }
            let combined = (epsilon[i][j] + mu[i][j]) * 0.5;
            let floquet = combined * (1.0 + calibration.modulation_depth);
            let real_value =
                benchmark_scale * floquet.norm() / (1.0 + calibration.loss_tangent.abs());

            effective_tensor[i][j] = floquet;
            effective_tensor[j][i] = floquet.conj();
            effective_real_coupling[i][j] = real_value;
            effective_real_coupling[j][i] = real_value;
            hermitian_real_coupling[i][j] = real_value;
            hermitian_real_coupling[j][i] = real_value;
        }
    }

    FloquetEffectiveModel {
        effective_tensor,
        effective_real_coupling,
        hermitian_real_coupling,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use optics_core::{
        absorber_benchmark::{CouplingTopology, ProjectionGate, run_benchmark},
        tcmt::KerrCavity,
    };

    fn sample_calibration() -> MaterialCalibrationRecord {
        MaterialCalibrationRecord {
            id: "unit_test".to_string(),
            platform: "test".to_string(),
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
            notes: "unit-test calibration".to_string(),
        }
    }

    fn repo_root() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crate parent")
            .parent()
            .expect("workspace root")
            .to_path_buf()
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

    #[test]
    fn canonical_topology_is_7xk6() {
        let topology = AssessorTopology::canonical().expect("canonical topology");
        assert_eq!(topology.nodes.len(), 42);
        assert_eq!(topology.boxkite_sizes, vec![6; 7]);
        assert_eq!(topology.edge_count(), 7 * 15);
    }

    #[test]
    fn calibration_loader_reads_project_csv_toml_mirror() {
        let path = repo_root().join(
            "registry/data/project_csv/canonical/PC-0006_c010_nonlocal_material_calibrations.toml",
        );
        let records = load_calibration_records(path).expect("load canonical TOML mirror");
        let default = find_calibration_record(&records, "nonlocal_cable_chen_2023")
            .expect("default calibration row");
        assert_eq!(records.len(), 7);
        assert_eq!(default.citation_key, "Chen2023");
        assert_eq!(
            default.data_provenance_class,
            "published_response_normalized"
        );
    }

    #[test]
    fn m3_projection_respects_mask_and_floor() {
        let config = M3ProjectionConfig::c010_hybrid_default();
        let model =
            SyntheticCouplingModel::build(&config, &sample_calibration()).expect("synthetic model");
        let n = model.topology.nodes.len();
        for i in 0..n {
            assert_eq!(model.normalized_assessor_weights[i][i], 0.0);
            for j in 0..n {
                let value = model.normalized_assessor_weights[i][j];
                assert!((value - model.normalized_assessor_weights[j][i]).abs() < 1.0e-12);
                if model.topology.adjacency_mask[i][j] {
                    if i != j {
                        assert!(value >= config.edge_floor);
                    }
                } else {
                    assert_eq!(value, 0.0);
                }
            }
        }
    }

    #[test]
    fn backend_reductions_stay_nonlocal_and_symmetric() {
        let config = M3ProjectionConfig::c010_hybrid_default();
        let model =
            SyntheticCouplingModel::build(&config, &sample_calibration()).expect("synthetic model");
        let n = model.topology.nodes.len();
        for i in 0..n {
            for j in 0..n {
                if model.topology.adjacency_mask[i][j] {
                    assert!(model.lc_admittance.effective_real_coupling[i][j] > 0.0);
                    assert!(model.floquet_effective.hermitian_real_coupling[i][j] > 0.0);
                } else {
                    assert_eq!(model.lc_admittance.effective_real_coupling[i][j], 0.0);
                    assert_eq!(model.floquet_effective.hermitian_real_coupling[i][j], 0.0);
                }
            }
        }
    }

    #[test]
    fn effective_matrix_passes_c010_gate() {
        let config = M3ProjectionConfig::c010_hybrid_default();
        let model =
            SyntheticCouplingModel::build(&config, &sample_calibration()).expect("synthetic model");
        let nonlocal = model.to_coupling_topology("nonlocal-m3");
        let kappa = model.average_nonzero_coupling();
        let suite = vec![
            nonlocal,
            CouplingTopology::ring(42, kappa),
            CouplingTopology::chain(42, kappa),
            CouplingTopology::complete(42, kappa),
            CouplingTopology::cross_cluster_bridges(7, 6, kappa),
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
        let gate = ProjectionGate::c010_default();
        let result = gate.evaluate(&benchmark);
        assert!(result.pass, "gate should pass, got {}", result.verdict);
    }

    #[test]
    fn synthetic_coupling_report_captures_topology_summary() {
        let model = SyntheticCouplingModel::build(
            &M3ProjectionConfig::c010_hybrid_default(),
            &sample_calibration(),
        )
        .expect("synthetic model");
        let report = model.report();

        assert_eq!(report.calibration_id, "unit_test");
        assert_eq!(report.dim, 16);
        assert_eq!(report.node_count, 42);
        assert_eq!(report.masked_edge_count, 7 * 15);
        assert_eq!(report.boxkite_count, 7);
        assert_eq!(report.boxkite_sizes, vec![6; 7]);
        assert_eq!(report.reggiani_partner_graph_order, 84);
        assert!(report.average_nonzero_coupling > 0.0);
        assert!(report.mean_active_weight > 0.0);
        assert!(report.max_active_weight >= report.mean_active_weight);
    }
}
