//! Training-epoch transforms and frozen logistic coefficients.

use anyhow::{Result, ensure};
use gororoba_cli_physics::staple_logistic::fit_irls;
use serde::{Deserialize, Serialize};

use super::{
    Config,
    admission::{Dataset, Row},
    splits,
};

#[derive(Clone, Deserialize, Serialize)]
pub(super) struct Model {
    pub(super) width: usize,
    pub(super) tensor: Option<usize>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(super) geometric_capacity: bool,
    pub(super) training_rows: usize,
    pub(super) training_positives: usize,
    pub(super) means: Vec<f64>,
    pub(super) scales: Vec<f64>,
    pub(super) coefficients: Vec<f64>,
    pub(super) deviance: Vec<f64>,
    pub(super) iterations: usize,
    pub(super) converged: bool,
    pub(super) ridge: f64,
}

pub(super) fn name(tensor: Option<usize>, config: &Config) -> String {
    match tensor {
        None => "baseline".to_owned(),
        Some(0) => "canonical".to_owned(),
        Some(index) => format!("scramble-{}", config.control_seeds[index - 1]),
    }
}

pub(super) fn feature_values(
    row: &Row,
    width_index: usize,
    tensor: Option<usize>,
    geometric: bool,
) -> [f32; 7] {
    let geometry = row.features.geometry;
    [
        geometry[0],
        geometry[1],
        geometry[2],
        geometry[3],
        row.features.pvi[width_index],
        geometry[4],
        if geometric {
            row.features.geometric_capacity[width_index]
        } else {
            tensor.map_or(0.0, |index| row.features.tensors[index])
        },
    ]
}

pub(super) fn standardize(
    features: &[f32],
    rows: &[u32],
    columns: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    ensure!(!rows.is_empty(), "empty training epochs");
    let mut means = vec![0.0; columns];
    for &row in rows {
        for column in 0..columns {
            means[column] += f64::from(features[row as usize * 7 + column]);
        }
    }
    for mean in &mut means {
        *mean /= rows.len() as f64;
    }
    let mut scales = vec![0.0; columns];
    for &row in rows {
        for column in 0..columns {
            scales[column] +=
                (f64::from(features[row as usize * 7 + column]) - means[column]).powi(2);
        }
    }
    for scale in &mut scales {
        *scale = (*scale / rows.len() as f64).sqrt();
    }
    ensure!(
        means.iter().all(|value| value.is_finite())
            && scales.iter().all(|value| value.is_finite() && *value > 0.0),
        "invalid training-only means/scales"
    );
    Ok((means, scales))
}

pub(super) fn fit(
    data: &Dataset,
    config: &Config,
    width_index: usize,
    tensor: Option<usize>,
) -> Result<Model> {
    fit_selected(data, config, width_index, tensor, false)
}

pub(super) fn fit_geometric(data: &Dataset, config: &Config, width_index: usize) -> Result<Model> {
    fit_selected(data, config, width_index, None, true)
}

fn fit_selected(
    data: &Dataset,
    config: &Config,
    width_index: usize,
    tensor: Option<usize>,
    geometric: bool,
) -> Result<Model> {
    let rows = splits::training_rows(data, config);
    let features: Vec<f32> = data
        .rows
        .iter()
        .flat_map(|row| feature_values(row, width_index, tensor, geometric))
        .collect();
    let labels: Vec<u8> = data.rows.iter().map(|row| row.label).collect();
    let columns = if tensor.is_some() || geometric { 7 } else { 6 };
    let (means, scales) = standardize(&features, &rows, columns)?;
    let fitted = fit_irls(
        &features,
        7,
        &(0..columns).collect::<Vec<_>>(),
        &rows,
        &labels,
        &means,
        &scales,
        config.ridge,
    )?;
    ensure!(
        fitted.converged,
        "fit exhausted Newton budget: width={} tensor={tensor:?}, iterations={}, deviance={:?}",
        config.widths[width_index],
        fitted.iterations,
        fitted.deviance
    );
    let model = Model {
        width: config.widths[width_index],
        tensor,
        geometric_capacity: geometric,
        training_rows: rows.len(),
        training_positives: rows
            .iter()
            .map(|&row| usize::from(labels[row as usize]))
            .sum(),
        means,
        scales,
        coefficients: fitted.beta,
        deviance: fitted.deviance,
        iterations: fitted.iterations,
        converged: fitted.converged,
        ridge: config.ridge,
    };
    model.validate(config)?;
    Ok(model)
}

impl Model {
    pub(super) fn validate(&self, config: &Config) -> Result<()> {
        let columns = if self.tensor.is_some() || self.geometric_capacity {
            7
        } else {
            6
        };
        ensure!(
            !self.geometric_capacity || self.tensor.is_none(),
            "geometric model cannot carry tensor identity"
        );
        ensure!(
            config.widths.contains(&self.width) && self.tensor.is_none_or(|index| index < 20),
            "model identity outside sealed plan"
        );
        ensure!(
            self.converged
                && self.iterations >= 2
                && self.iterations <= 25
                && self.iterations == self.deviance.len(),
            "invalid fit completion"
        );
        ensure!(
            self.training_positives > 0 && self.training_positives < self.training_rows,
            "training lacks both classes"
        );
        ensure!(
            self.means.len() == columns
                && self.scales.len() == columns
                && self.coefficients.len() == columns + 1,
            "model dimensions differ from declared fitting budget"
        );
        ensure!(
            self.means
                .iter()
                .chain(&self.coefficients)
                .chain(&self.deviance)
                .all(|value| value.is_finite())
                && self
                    .scales
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0)
                && self.ridge == config.ridge,
            "invalid model parameters"
        );
        let previous = self.deviance[self.iterations - 2];
        ensure!(
            (previous - self.deviance[self.iterations - 1]).abs() / previous.abs().max(1.0)
                < gororoba_cli_physics::staple_logistic::DEVIANCE_REL_TOL,
            "retained trajectory violates convergence tolerance"
        );
        Ok(())
    }
    pub(super) fn predict(&self, row: &Row, width_index: usize) -> f64 {
        let features = feature_values(row, width_index, self.tensor, self.geometric_capacity);
        self.coefficients[0]
            + (0..self.means.len())
                .map(|column| {
                    self.coefficients[column + 1]
                        * (f64::from(features[column]) - self.means[column])
                        / self.scales[column]
                })
                .sum::<f64>()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn held_out_features_and_labels_leave_frozen_coefficients_identical() {
        use rand::{RngExt, SeedableRng};
        let mut random = rand_chacha::ChaCha8Rng::seed_from_u64(71);
        let mut data = crate::admission::Dataset {
            files: Vec::new(),
            rows: (0..120)
                .map(|index| crate::admission::Row {
                    features: crate::features::Features {
                        geometry: std::array::from_fn(|_| random.random::<f32>()),
                        pvi: std::array::from_fn(|_| random.random::<f32>()),
                        tensors: std::array::from_fn(|_| random.random::<f32>()),
                        geometric_capacity: std::array::from_fn(|_| random.random::<f32>()),
                    },
                    label: u8::from(random.random::<bool>()),
                    file: 0,
                    year: if index < 100 { 2010 } else { 2015 },
                })
                .collect(),
        };
        let config = crate::test_config();
        let original = super::fit(&data, &config, 0, Some(0)).unwrap();
        let geometric = super::fit_geometric(&data, &config, 0).unwrap();
        assert!(
            serde_json::to_value(&original)
                .unwrap()
                .get("geometric_capacity")
                .is_none()
        );
        assert_eq!(
            serde_json::to_value(&geometric).unwrap()["geometric_capacity"],
            true
        );
        assert!(geometric.tensor.is_none());
        for row in &mut data.rows[100..] {
            row.features.geometry.fill(1e10);
            row.features.pvi.fill(-1e10);
            row.features.tensors.fill(5e9);
            row.features.geometric_capacity.fill(-2e8);
            row.label ^= 1;
        }
        let changed = super::fit(&data, &config, 0, Some(0)).unwrap();
        assert_eq!(
            serde_json::to_value(geometric).unwrap(),
            serde_json::to_value(super::fit_geometric(&data, &config, 0).unwrap()).unwrap()
        );
        assert_eq!(
            serde_json::to_value(original).unwrap(),
            serde_json::to_value(changed).unwrap()
        );
    }

    #[test]
    fn holdout_values_cannot_change_training_transforms() {
        let mut features: Vec<f32> = (0..12)
            .flat_map(|row| (0..7).map(move |column| ((row + 1) * (column + 2)) as f32))
            .collect();
        let training = [0, 1, 2, 3];
        let original = super::standardize(&features, &training, 7).unwrap();
        features[4 * 7..].fill(1e20);
        assert_eq!(
            original,
            super::standardize(&features, &training, 7).unwrap()
        );
    }
}
