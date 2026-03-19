use crate::nanograv_timing_model::{ReleaseBand, TaggedTerm, TimingModel, load_release_timing_models};
use anyhow::{Context, Result, bail};
use nalgebra::{DMatrix, DVector};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
};

const DISPERSION_DELAY_US_PER_MHZ2_DM: f64 = 4_148_808.0;

#[derive(Debug, Clone)]
pub struct WidebandToa {
    pub mjd: f64,
    pub frequency_mhz: f64,
    pub uncertainty_us: f64,
    pub pp_dm: f64,
    pub pp_dme: Option<f64>,
    pub flags: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct ChannelResidual {
    pub mjd: f64,
    pub frequency_mhz: f64,
    pub residual_us: f64,
    pub white_residual_us: Option<f64>,
    pub uncertainty_us: f64,
    pub flag: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AggregatedObservation {
    pub solution_id: String,
    pub pulsar_id: String,
    pub mjd: f64,
    pub frequency_mhz: f64,
    pub residual_us: f64,
    pub residual_uncertainty_us: f64,
    pub wideband_dm: f64,
    pub wideband_dm_uncertainty: Option<f64>,
    pub dm_model: f64,
    pub matched_channel_rows: usize,
    pub flags: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct RefitDataset {
    pub model: TimingModel,
    pub observations: Vec<AggregatedObservation>,
    pub total_channel_rows_used: usize,
}

#[derive(Debug, Clone)]
pub struct FitParameterEstimate {
    pub name: String,
    pub coefficient: f64,
}

#[derive(Debug, Clone)]
pub struct LinearFitSummary {
    pub parameters: Vec<FitParameterEstimate>,
    pub toa_rms_before_us: f64,
    pub toa_rms_after_us: f64,
    pub toa_weighted_rms_before_us: f64,
    pub toa_weighted_rms_after_us: f64,
    pub dm_rms_before: f64,
    pub dm_rms_after: f64,
    pub toa_observation_count: usize,
    pub dm_observation_count: usize,
}

#[derive(Debug, Clone)]
pub struct RefitResult {
    pub parameter_names: Vec<String>,
    pub wls_coefficients: Vec<f64>,
    pub gls_coefficients: Vec<f64>,
    pub wls_summary: LinearFitSummary,
    pub gls_summary: LinearFitSummary,
    pub rows: Vec<RefitOutputRow>,
}

#[derive(Debug, Clone)]
pub struct RefitOutputRow {
    pub solution_id: String,
    pub pulsar_id: String,
    pub mjd: f64,
    pub frequency_mhz: f64,
    pub matched_channel_rows: usize,
    pub residual_before_us: f64,
    pub residual_after_wls_us: f64,
    pub residual_after_gls_us: f64,
    pub residual_uncertainty_us: f64,
    pub dm_before: f64,
    pub dm_after_wls: f64,
    pub dm_after_gls: f64,
    pub dm_uncertainty: Option<f64>,
}

#[derive(Debug, Clone)]
enum BasisKind {
    PhaseOffset,
    SpinDt,
    SpinDt2,
    DmOffset,
    DmxWindow(String),
    Jump(String),
    DmJump(String),
}

pub fn load_phase1_models_from_report(
    root: &Path,
    report_path: &Path,
    band: ReleaseBand,
) -> Result<Vec<TimingModel>> {
    let value: toml::Value = toml::from_str(
        &fs::read_to_string(report_path)
            .with_context(|| format!("read {}", report_path.display()))?,
    )?;
    let phase1 = value
        .get("phase1_subset")
        .and_then(|entry| entry.as_array())
        .ok_or_else(|| anyhow::anyhow!("missing phase1_subset array in {}", report_path.display()))?;
    let wanted = phase1
        .iter()
        .filter_map(|entry| {
            entry.get("solution_id")
                .and_then(|value| value.as_str())
                .map(str::to_string)
        })
        .collect::<Vec<_>>();
    let models = load_release_timing_models(root, band)?;
    let index = models
        .into_iter()
        .map(|model| (model.solution_id.clone(), model))
        .collect::<HashMap<_, _>>();
    let mut selected = Vec::new();
    for solution_id in wanted {
        let Some(model) = index.get(&solution_id) else {
            bail!("phase1 solution {solution_id} missing under {}", root.display());
        };
        selected.push(model.clone());
    }
    Ok(selected)
}

pub fn build_refit_dataset(
    root: &Path,
    model: &TimingModel,
    residual_tolerance_days: f64,
) -> Result<RefitDataset> {
    let tim_path = root.join("wideband/tim").join(format!("{}.tim", model.solution_id));
    let full_residual_path = root
        .join("residuals")
        .join(format!("{}_NG15yr_nb.full.res", model.pulsar_id));
    let toas = parse_wideband_toas(&tim_path)?;
    let residuals = parse_channel_residuals(&full_residual_path)?;
    let grouped = group_channel_residuals(&residuals);

    let mut observations = Vec::new();
    let mut total_channel_rows_used = 0usize;
    for toa in toas {
        let Some(group) = grouped
            .iter()
            .find(|group| (group.mjd - toa.mjd).abs() <= residual_tolerance_days)
        else {
            continue;
        };
        total_channel_rows_used += group.rows.len();
        observations.push(AggregatedObservation {
            solution_id: model.solution_id.clone(),
            pulsar_id: model.pulsar_id.clone(),
            mjd: toa.mjd,
            frequency_mhz: toa.frequency_mhz,
            residual_us: group.weighted_residual_us,
            residual_uncertainty_us: group.uncertainty_us.max(toa.uncertainty_us),
            wideband_dm: toa.pp_dm,
            wideband_dm_uncertainty: toa.pp_dme,
            dm_model: current_dm_model(model, toa.mjd, &toa.flags),
            matched_channel_rows: group.rows.len(),
            flags: toa.flags,
        });
    }
    if observations.is_empty() {
        bail!(
            "no TOA/residual matches found for {} within {:.3e} days",
            model.solution_id,
            residual_tolerance_days
        );
    }
    Ok(RefitDataset {
        model: model.clone(),
        observations,
        total_channel_rows_used,
    })
}

pub fn solve_refit(
    dataset: &RefitDataset,
    gls_corr_length_days: f64,
    gls_red_noise_fraction: f64,
) -> Result<RefitResult> {
    let parameter_names = build_parameter_names(&dataset.model, &dataset.observations);
    if parameter_names.is_empty() {
        bail!("empty design matrix for {}", dataset.model.solution_id);
    }
    let basis = parameter_names
        .iter()
        .map(|name| basis_kind(name))
        .collect::<Vec<_>>();

    let (toa_matrix, toa_response, toa_sigma) =
        build_toa_system(&dataset.model, &dataset.observations, &basis);
    let (dm_matrix, dm_response, dm_sigma) =
        build_dm_system(&dataset.model, &dataset.observations, &basis);

    let total_rows = toa_response.len() + dm_response.len();
    let ncols = parameter_names.len();
    let mut design = DMatrix::zeros(total_rows, ncols);
    let mut response = DVector::zeros(total_rows);
    let mut sigma = vec![0.0; total_rows];
    pack_stacked_system(
        &toa_matrix,
        &toa_response,
        &toa_sigma,
        &dm_matrix,
        &dm_response,
        &dm_sigma,
        &mut design,
        &mut response,
        &mut sigma,
    );

    let wls_coefficients = solve_weighted_least_squares(&design, &response, &sigma)?;
    let gls_covariance = build_gls_toa_covariance(
        &dataset.observations,
        &toa_sigma,
        gls_corr_length_days,
        gls_red_noise_fraction,
    );
    let gls_coefficients = solve_block_generalized_least_squares(
        &toa_matrix,
        &toa_response,
        &gls_covariance,
        &dm_matrix,
        &dm_response,
        &dm_sigma,
    )?;

    let wls_summary = summarize_fit(
        &parameter_names,
        &wls_coefficients,
        dataset,
        &toa_matrix,
        &toa_response,
        &toa_sigma,
        &dm_matrix,
        &dm_response,
    );
    let gls_summary = summarize_fit(
        &parameter_names,
        &gls_coefficients,
        dataset,
        &toa_matrix,
        &toa_response,
        &toa_sigma,
        &dm_matrix,
        &dm_response,
    );

    let mut rows = Vec::new();
    for (index, observation) in dataset.observations.iter().enumerate() {
        let toa_before = toa_response[index];
        let toa_after_wls = toa_before - row_dot(&toa_matrix, index, &wls_coefficients);
        let toa_after_gls = toa_before - row_dot(&toa_matrix, index, &gls_coefficients);
        let dm_before = dm_response[index];
        let dm_after_wls = dm_before - row_dot(&dm_matrix, index, &wls_coefficients);
        let dm_after_gls = dm_before - row_dot(&dm_matrix, index, &gls_coefficients);
        rows.push(RefitOutputRow {
            solution_id: observation.solution_id.clone(),
            pulsar_id: observation.pulsar_id.clone(),
            mjd: observation.mjd,
            frequency_mhz: observation.frequency_mhz,
            matched_channel_rows: observation.matched_channel_rows,
            residual_before_us: toa_before,
            residual_after_wls_us: toa_after_wls,
            residual_after_gls_us: toa_after_gls,
            residual_uncertainty_us: observation.residual_uncertainty_us,
            dm_before,
            dm_after_wls,
            dm_after_gls,
            dm_uncertainty: observation.wideband_dm_uncertainty,
        });
    }

    Ok(RefitResult {
        parameter_names: parameter_names.clone(),
        wls_coefficients: wls_coefficients.iter().copied().collect(),
        gls_coefficients: gls_coefficients.iter().copied().collect(),
        wls_summary,
        gls_summary,
        rows,
    })
}

fn parse_wideband_toas(path: &Path) -> Result<Vec<WidebandToa>> {
    let content = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut out = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('C') || trimmed.starts_with("FORMAT") {
            continue;
        }
        let fields = trimmed.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 5 {
            continue;
        }
        let Some(frequency_mhz) = parse_f64(fields[1]) else {
            continue;
        };
        let Some(mjd) = parse_f64(fields[2]) else {
            continue;
        };
        let Some(uncertainty_us) = parse_f64(fields[3]) else {
            continue;
        };
        let Some(pp_dm) = flagged_numeric(&fields, "-pp_dm") else {
            continue;
        };
        let flags = parse_flags(&fields[5..]);
        out.push(WidebandToa {
            mjd,
            frequency_mhz,
            uncertainty_us,
            pp_dm,
            pp_dme: flagged_numeric(&fields, "-pp_dme"),
            flags,
        });
    }
    out.sort_by(|left, right| left.mjd.total_cmp(&right.mjd));
    Ok(out)
}

fn parse_channel_residuals(path: &Path) -> Result<Vec<ChannelResidual>> {
    let content = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut out = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields = trimmed.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 6 {
            continue;
        }
        let Some(mjd) = parse_f64(fields[0]) else {
            continue;
        };
        let Some(frequency_mhz) = parse_f64(fields[1]) else {
            continue;
        };
        let Some(residual_us) = parse_f64(fields[2]) else {
            continue;
        };
        let Some(uncertainty_us) = parse_f64(fields[4]) else {
            continue;
        };
        out.push(ChannelResidual {
            mjd,
            frequency_mhz,
            residual_us,
            white_residual_us: parse_f64(fields[3]),
            uncertainty_us,
            flag: fields.get(6).map(|value| (*value).to_string()),
        });
    }
    out.sort_by(|left, right| left.mjd.total_cmp(&right.mjd));
    Ok(out)
}

#[derive(Debug, Clone)]
struct ResidualGroup {
    mjd: f64,
    weighted_residual_us: f64,
    uncertainty_us: f64,
    rows: Vec<ChannelResidual>,
}

fn group_channel_residuals(rows: &[ChannelResidual]) -> Vec<ResidualGroup> {
    let mut groups: Vec<ResidualGroup> = Vec::new();
    for row in rows {
        if let Some(last) = groups.last_mut()
            && (last.mjd - row.mjd).abs() <= 1.0e-8
        {
            last.rows.push(row.clone());
            recompute_group(last);
            continue;
        }
        let mut group = ResidualGroup {
            mjd: row.mjd,
            weighted_residual_us: row.residual_us,
            uncertainty_us: row.uncertainty_us,
            rows: vec![row.clone()],
        };
        recompute_group(&mut group);
        groups.push(group);
    }
    groups
}

fn recompute_group(group: &mut ResidualGroup) {
    let mut weighted_sum = 0.0;
    let mut weight_total = 0.0;
    for row in &group.rows {
        let sigma = row.uncertainty_us.max(1.0e-6);
        let weight = 1.0 / (sigma * sigma);
        weighted_sum += weight * row.residual_us;
        weight_total += weight;
    }
    if weight_total > 0.0 {
        group.weighted_residual_us = weighted_sum / weight_total;
        group.uncertainty_us = (1.0 / weight_total).sqrt();
    }
}

fn parse_flags(tokens: &[&str]) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut index = 0usize;
    while index + 1 < tokens.len() {
        if tokens[index].starts_with('-') {
            out.insert(tokens[index].to_string(), tokens[index + 1].to_string());
            index += 2;
        } else {
            index += 1;
        }
    }
    out
}

fn flagged_numeric(fields: &[&str], needle: &str) -> Option<f64> {
    fields.windows(2).find_map(|window| {
        if window[0] == needle {
            parse_f64(window[1])
        } else {
            None
        }
    })
}

fn current_dm_model(model: &TimingModel, mjd: f64, flags: &BTreeMap<String, String>) -> f64 {
    let mut total = model.dispersion.dm.as_ref().and_then(|term| term.value).unwrap_or(0.0);
    for window in &model.dispersion.dmx_windows {
        let in_window = window.start_mjd.is_some_and(|start| mjd >= start)
            && window.end_mjd.is_some_and(|end| mjd <= end);
        if in_window {
            total += window.dmx.as_ref().and_then(|term| term.value).unwrap_or(0.0);
        }
    }
    for jump in &model.dmjumps {
        if tagged_term_matches(jump, flags) {
            total += jump.value.unwrap_or(0.0);
        }
    }
    total
}

fn build_parameter_names(
    model: &TimingModel,
    observations: &[AggregatedObservation],
) -> Vec<String> {
    let mut names = vec![
        "phase_offset_us".to_string(),
        "spin_dt_days".to_string(),
        "spin_dt2_days2".to_string(),
    ];
    names.push("dm_offset".to_string());
    for window in &model.dispersion.dmx_windows {
        if observations.iter().any(|obs| {
            window.start_mjd.is_some_and(|start| obs.mjd >= start)
                && window.end_mjd.is_some_and(|end| obs.mjd <= end)
        }) {
            names.push(format!("dmx_window_{}", window.label));
        }
    }
    for (index, jump) in model.jumps.iter().enumerate() {
        if observations
            .iter()
            .any(|obs| tagged_term_matches(jump, &obs.flags))
        {
            names.push(format!("jump_{}", selector_key(index, jump)));
        }
    }
    for (index, jump) in model.dmjumps.iter().enumerate() {
        if observations
            .iter()
            .any(|obs| tagged_term_matches(jump, &obs.flags))
        {
            names.push(format!("dmjump_{}", selector_key(index, jump)));
        }
    }
    names
}

fn build_toa_system(
    model: &TimingModel,
    observations: &[AggregatedObservation],
    basis: &[BasisKind],
) -> (DMatrix<f64>, DVector<f64>, Vec<f64>) {
    let epoch_mjd = model_parameter_value(model, "PEPOCH")
        .or(model.start_mjd)
        .unwrap_or_else(|| observations[0].mjd);
    let mut matrix = DMatrix::zeros(observations.len(), basis.len());
    let mut response = DVector::zeros(observations.len());
    let mut sigma = Vec::with_capacity(observations.len());
    let dmx_labels = observations
        .iter()
        .map(|obs| active_dmx_window_label(model, obs.mjd))
        .collect::<Vec<_>>();
    for (row, obs) in observations.iter().enumerate() {
        response[row] = obs.residual_us;
        sigma.push(obs.residual_uncertainty_us.max(1.0e-4));
        let dt_days = obs.mjd - epoch_mjd;
        let dispersion_scale = DISPERSION_DELAY_US_PER_MHZ2_DM / obs.frequency_mhz.powi(2);
        for (col, kind) in basis.iter().enumerate() {
            matrix[(row, col)] = match kind {
                BasisKind::PhaseOffset => 1.0,
                BasisKind::SpinDt => dt_days,
                BasisKind::SpinDt2 => dt_days * dt_days,
                BasisKind::DmOffset => dispersion_scale,
                BasisKind::DmxWindow(label) => {
                    usize::from(dmx_labels[row].as_deref() == Some(label.as_str())) as f64
                        * dispersion_scale
                }
                BasisKind::Jump(key) => jump_basis_from_key(key, &model.jumps, &obs.flags),
                BasisKind::DmJump(key) => {
                    jump_basis_from_key(key, &model.dmjumps, &obs.flags) * dispersion_scale
                }
            };
        }
    }
    (matrix, response, sigma)
}

fn build_dm_system(
    model: &TimingModel,
    observations: &[AggregatedObservation],
    basis: &[BasisKind],
) -> (DMatrix<f64>, DVector<f64>, Vec<f64>) {
    let mut matrix = DMatrix::zeros(observations.len(), basis.len());
    let mut response = DVector::zeros(observations.len());
    let mut sigma = Vec::with_capacity(observations.len());
    let dmx_labels = observations
        .iter()
        .map(|obs| active_dmx_window_label(model, obs.mjd))
        .collect::<Vec<_>>();
    for (row, obs) in observations.iter().enumerate() {
        response[row] = obs.wideband_dm - obs.dm_model;
        sigma.push(obs.wideband_dm_uncertainty.unwrap_or(0.005).max(1.0e-6));
        for (col, kind) in basis.iter().enumerate() {
            matrix[(row, col)] = match kind {
                BasisKind::DmOffset => 1.0,
                BasisKind::DmxWindow(label) => {
                    usize::from(dmx_labels[row].as_deref() == Some(label.as_str())) as f64
                }
                BasisKind::DmJump(key) => jump_basis_from_key(key, &model.dmjumps, &obs.flags),
                _ => 0.0,
            };
        }
    }
    (matrix, response, sigma)
}

fn solve_weighted_least_squares(
    design: &DMatrix<f64>,
    response: &DVector<f64>,
    sigma: &[f64],
) -> Result<DVector<f64>> {
    let mut weighted_design = design.clone();
    let mut weighted_response = response.clone();
    for row in 0..design.nrows() {
        let weight = 1.0 / sigma[row].max(1.0e-8);
        for col in 0..design.ncols() {
            weighted_design[(row, col)] *= weight;
        }
        weighted_response[row] *= weight;
    }
    let normal = weighted_design.transpose() * &weighted_design;
    let rhs = weighted_design.transpose() * weighted_response;
    solve_normal_equations(&normal, &rhs)
}

fn solve_block_generalized_least_squares(
    toa_matrix: &DMatrix<f64>,
    toa_response: &DVector<f64>,
    toa_covariance: &DMatrix<f64>,
    dm_matrix: &DMatrix<f64>,
    dm_response: &DVector<f64>,
    dm_sigma: &[f64],
) -> Result<DVector<f64>> {
    let Some(cholesky) = toa_covariance.clone().cholesky() else {
        bail!("TOA GLS covariance is not positive definite");
    };
    let toa_c_inv_design = cholesky.solve(toa_matrix);
    let toa_c_inv_response = cholesky.solve(toa_response);
    let mut normal = toa_matrix.transpose() * toa_c_inv_design;
    let mut rhs = toa_matrix.transpose() * toa_c_inv_response;
    for row in 0..dm_matrix.nrows() {
        let weight = 1.0 / dm_sigma[row].max(1.0e-8).powi(2);
        for left in 0..dm_matrix.ncols() {
            rhs[left] += dm_matrix[(row, left)] * dm_response[row] * weight;
            for right in 0..dm_matrix.ncols() {
                normal[(left, right)] += dm_matrix[(row, left)] * dm_matrix[(row, right)] * weight;
            }
        }
    }
    solve_normal_equations(&normal, &rhs)
}

#[allow(clippy::too_many_arguments)]
fn pack_stacked_system(
    toa_matrix: &DMatrix<f64>,
    toa_response: &DVector<f64>,
    toa_sigma: &[f64],
    dm_matrix: &DMatrix<f64>,
    dm_response: &DVector<f64>,
    dm_sigma: &[f64],
    design: &mut DMatrix<f64>,
    response: &mut DVector<f64>,
    sigma: &mut [f64],
) {
    for row in 0..toa_response.len() {
        for col in 0..toa_matrix.ncols() {
            design[(row, col)] = toa_matrix[(row, col)];
        }
        response[row] = toa_response[row];
        sigma[row] = toa_sigma[row];
    }
    for row in 0..dm_response.len() {
        let offset = toa_response.len() + row;
        for col in 0..dm_matrix.ncols() {
            design[(offset, col)] = dm_matrix[(row, col)];
        }
        response[offset] = dm_response[row];
        sigma[offset] = dm_sigma[row];
    }
}

fn solve_normal_equations(normal: &DMatrix<f64>, rhs: &DVector<f64>) -> Result<DVector<f64>> {
    let mut regularized = normal.clone();
    let ridge = regularized
        .diagonal()
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(1.0)
        * 1.0e-10;
    for index in 0..regularized.nrows() {
        regularized[(index, index)] += ridge;
    }
    if let Some(cholesky) = regularized.clone().cholesky() {
        return Ok(cholesky.solve(rhs));
    }
    if let Some(solution) = regularized.lu().solve(rhs) {
        return Ok(solution);
    }
    bail!("linear solve failed for regularized normal equations")
}

fn build_gls_toa_covariance(
    observations: &[AggregatedObservation],
    sigma: &[f64],
    corr_length_days: f64,
    red_noise_fraction: f64,
) -> DMatrix<f64> {
    let n_toa = observations.len();
    let mut covariance = DMatrix::zeros(n_toa, n_toa);
    for row in 0..n_toa {
        covariance[(row, row)] = sigma[row] * sigma[row];
    }
    if red_noise_fraction <= 0.0 {
        return covariance;
    }

    let median_sigma = median_sigma(&sigma[..n_toa]);
    let red_amplitude2 = (median_sigma * red_noise_fraction).powi(2);
    for i in 0..n_toa {
        for j in i..n_toa {
            let dt = (observations[i].mjd - observations[j].mjd).abs();
            let corr = red_amplitude2 * (-(dt / corr_length_days.max(1.0))).exp();
            covariance[(i, j)] += corr;
            if i != j {
                covariance[(j, i)] += corr;
            }
        }
    }
    covariance
}

fn median_sigma(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    sorted[sorted.len() / 2]
}

#[allow(clippy::too_many_arguments)]
fn summarize_fit(
    parameter_names: &[String],
    coefficients: &DVector<f64>,
    dataset: &RefitDataset,
    toa_matrix: &DMatrix<f64>,
    toa_response: &DVector<f64>,
    toa_sigma: &[f64],
    dm_matrix: &DMatrix<f64>,
    dm_response: &DVector<f64>,
) -> LinearFitSummary {
    let predicted_toa = toa_matrix * coefficients;
    let predicted_dm = dm_matrix * coefficients;
    let toa_before = toa_response.iter().copied().collect::<Vec<_>>();
    let toa_after = toa_response
        .iter()
        .zip(predicted_toa.iter())
        .map(|(lhs, rhs)| lhs - rhs)
        .collect::<Vec<_>>();
    let dm_before = dm_response.iter().copied().collect::<Vec<_>>();
    let dm_after = dm_response
        .iter()
        .zip(predicted_dm.iter())
        .map(|(lhs, rhs)| lhs - rhs)
        .collect::<Vec<_>>();

    LinearFitSummary {
        parameters: parameter_names
            .iter()
            .zip(coefficients.iter())
            .map(|(name, coefficient)| FitParameterEstimate {
                name: name.clone(),
                coefficient: *coefficient,
            })
            .collect(),
        toa_rms_before_us: rms(&toa_before),
        toa_rms_after_us: rms(&toa_after),
        toa_weighted_rms_before_us: weighted_rms(&toa_before, toa_sigma),
        toa_weighted_rms_after_us: weighted_rms(&toa_after, toa_sigma),
        dm_rms_before: rms(&dm_before),
        dm_rms_after: rms(&dm_after),
        toa_observation_count: dataset.observations.len(),
        dm_observation_count: dataset.observations.len(),
    }
}

fn rms(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean_sq = values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64;
    mean_sq.sqrt()
}

fn weighted_rms(values: &[f64], sigma: &[f64]) -> f64 {
    let mut weight_sum = 0.0;
    let mut weighted_sq = 0.0;
    for (value, sigma) in values.iter().zip(sigma.iter()) {
        let weight = 1.0 / sigma.max(1.0e-8).powi(2);
        weight_sum += weight;
        weighted_sq += weight * value * value;
    }
    if weight_sum == 0.0 {
        0.0
    } else {
        (weighted_sq / weight_sum).sqrt()
    }
}

fn jump_basis_from_key(target: &str, terms: &[TaggedTerm], flags: &BTreeMap<String, String>) -> f64 {
    for (index, term) in terms.iter().enumerate() {
        if selector_key(index, term) == target {
            return tagged_term_matches(term, flags) as usize as f64;
        }
    }
    0.0
}

fn selector_key(index: usize, term: &TaggedTerm) -> String {
    if term.selectors.is_empty() {
        return format!("{index}_{}", term.name.to_ascii_lowercase());
    }
    term.selectors
        .iter()
        .map(|selector| {
            format!(
                "{}_{}",
                selector.flag.trim_start_matches('-'),
                sanitize_token(&selector.value)
            )
        })
        .collect::<Vec<_>>()
        .join("__")
}

fn basis_kind(name: &str) -> BasisKind {
    match name {
        "phase_offset_us" => BasisKind::PhaseOffset,
        "spin_dt_days" => BasisKind::SpinDt,
        "spin_dt2_days2" => BasisKind::SpinDt2,
        "dm_offset" => BasisKind::DmOffset,
        _ if name.starts_with("dmx_window_") => {
            BasisKind::DmxWindow(name.trim_start_matches("dmx_window_").to_string())
        }
        _ if name.starts_with("jump_") => {
            BasisKind::Jump(name.trim_start_matches("jump_").to_string())
        }
        _ if name.starts_with("dmjump_") => {
            BasisKind::DmJump(name.trim_start_matches("dmjump_").to_string())
        }
        _ => BasisKind::PhaseOffset,
    }
}

fn active_dmx_window_label(model: &TimingModel, mjd: f64) -> Option<String> {
    model.dispersion.dmx_windows.iter().find_map(|window| {
        let in_window = window.start_mjd.is_some_and(|start| mjd >= start)
            && window.end_mjd.is_some_and(|end| mjd <= end);
        in_window.then(|| window.label.clone())
    })
}

fn sanitize_token(value: &str) -> String {
    value.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

fn tagged_term_matches(term: &TaggedTerm, flags: &BTreeMap<String, String>) -> bool {
    term.selectors.iter().all(|selector| {
        flags
            .get(&selector.flag)
            .is_some_and(|value| value == &selector.value)
    })
}

fn model_parameter_value(model: &TimingModel, name: &str) -> Option<f64> {
    model
        .spin_terms
        .iter()
        .chain(model.fd_terms.iter())
        .chain(model.other_terms.iter())
        .find(|term| term.name == name)
        .and_then(|term| term.value.or_else(|| parse_f64(&term.raw_value)))
}

fn parse_f64(value: &str) -> Option<f64> {
    value.parse::<f64>().ok().filter(|parsed| parsed.is_finite())
}

fn row_dot(matrix: &DMatrix<f64>, row: usize, coefficients: &DVector<f64>) -> f64 {
    (0..matrix.ncols())
        .map(|col| matrix[(row, col)] * coefficients[col])
        .sum()
}

#[cfg(test)]
mod tests {
    use super::{build_gls_toa_covariance, group_channel_residuals, ChannelResidual};
    use std::collections::BTreeMap;

    #[test]
    fn groups_nearby_channel_rows() {
        let rows = vec![
            ChannelResidual {
                mjd: 58000.0,
                frequency_mhz: 1400.0,
                residual_us: 1.0,
                white_residual_us: None,
                uncertainty_us: 1.0,
                flag: None,
            },
            ChannelResidual {
                mjd: 58000.0 + 5.0e-9,
                frequency_mhz: 1404.0,
                residual_us: 3.0,
                white_residual_us: None,
                uncertainty_us: 1.0,
                flag: None,
            },
        ];
        let groups = group_channel_residuals(&rows);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].rows.len(), 2);
        assert!((groups[0].weighted_residual_us - 2.0).abs() < 1.0e-9);
    }

    #[test]
    fn gls_covariance_keeps_positive_diagonal() {
        let observations = vec![
            super::AggregatedObservation {
                solution_id: "s".to_string(),
                pulsar_id: "p".to_string(),
                mjd: 58000.0,
                frequency_mhz: 1400.0,
                residual_us: 0.0,
                residual_uncertainty_us: 0.1,
                wideband_dm: 10.0,
                wideband_dm_uncertainty: Some(0.01),
                dm_model: 10.0,
                matched_channel_rows: 1,
                flags: BTreeMap::new(),
            },
            super::AggregatedObservation {
                solution_id: "s".to_string(),
                pulsar_id: "p".to_string(),
                mjd: 58010.0,
                frequency_mhz: 1400.0,
                residual_us: 0.0,
                residual_uncertainty_us: 0.2,
                wideband_dm: 10.0,
                wideband_dm_uncertainty: Some(0.02),
                dm_model: 10.0,
                matched_channel_rows: 1,
                flags: BTreeMap::new(),
            },
        ];
        let sigma = vec![0.1, 0.2, 0.01, 0.02];
        let covariance = build_gls_toa_covariance(&observations, &sigma[..2], 30.0, 0.5);
        assert!(covariance[(0, 0)] > 0.01);
        assert!(covariance[(1, 1)] > 0.04);
        assert_eq!(covariance.nrows(), 2);
        assert_eq!(covariance.ncols(), 2);
    }
}
