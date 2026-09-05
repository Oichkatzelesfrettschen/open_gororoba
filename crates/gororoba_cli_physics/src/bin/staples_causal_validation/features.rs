//! Shared timestamp-local geometry and exact-support cubic controls.

use anyhow::{Result, ensure};
use cd_kernel::mult_table::CdMultTable;
use gororoba_cli_physics::{
    staple_associator::staple_embedding, staple_controls::SparseCubicTensor,
};
use serde::Serialize;

use super::evidence::digest;

pub(super) struct Ensemble {
    pub(super) tensors: Vec<SparseCubicTensor>,
    pub(super) declarations: Vec<TensorDeclaration>,
}

#[derive(Serialize)]
pub(super) struct TensorDeclaration {
    name: String,
    seed: Option<u64>,
    terms: usize,
    dimension: usize,
    support_sha256: String,
    coefficients_sha256: String,
}

impl Ensemble {
    pub(super) fn new(seeds: &[u64]) -> Result<Self> {
        let canonical = SparseCubicTensor::from_associator(&CdMultTable::generate(16));
        ensure!(
            canonical.term_count() == 1848,
            "canonical support cardinality changed"
        );
        let mut tensors: Vec<_> = seeds
            .iter()
            .map(|&seed| canonical.sign_scrambled(seed))
            .collect();
        tensors.insert(0, canonical);
        let support: Vec<u8> = tensors[0]
            .terms()
            .iter()
            .flat_map(|&(first, second, third, coefficient)| {
                [first, second, third, coefficient.unsigned_abs()]
            })
            .collect();
        let mut declarations = Vec::new();
        for (index, tensor) in tensors.iter().enumerate() {
            let observed: Vec<u8> = tensor
                .terms()
                .iter()
                .flat_map(|&(first, second, third, coefficient)| {
                    [first, second, third, coefficient.unsigned_abs()]
                })
                .collect();
            ensure!(observed == support, "control changes exact ordered support");
            let coefficients: Vec<u8> = tensor.terms().iter().map(|term| term.3 as u8).collect();
            declarations.push(TensorDeclaration {
                name: if index == 0 {
                    "canonical".to_owned()
                } else {
                    format!("scramble-{}", seeds[index - 1])
                },
                seed: index.checked_sub(1).map(|position| seeds[position]),
                terms: tensor.term_count(),
                dimension: 16,
                support_sha256: digest(&support),
                coefficients_sha256: digest(&coefficients),
            });
        }
        Ok(Self {
            tensors,
            declarations,
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct Sample {
    pub(super) nanos: i64,
    pub(super) raw_index: u64,
    pub(super) vector: [f64; 3],
}

pub(super) struct Features {
    pub(super) geometry: [f32; 5],
    pub(super) pvi: [f32; 3],
    pub(super) tensors: [f32; 20],
    pub(super) geometric_capacity: [f32; 3],
}

fn norm(vector: &[f64; 3]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn increment(left: &[f64; 3], right: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| right[axis] - left[axis])
}

fn angle(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    let denominator = norm(left) * norm(right);
    if denominator == 0.0 {
        return 0.0;
    }
    (left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f64>()
        / denominator)
        .clamp(-1.0, 1.0)
        .acos()
}

pub(super) fn construct(
    history: &[Sample],
    widths: &[usize; 3],
    epsilon: f64,
    ensemble: &Ensemble,
) -> Result<Features> {
    let start = *widths.iter().max().unwrap() + 1;
    ensure!(
        history.len() == start + 6,
        "feature history differs from common support"
    );
    let vectors: Vec<[f64; 3]> = history[start..]
        .iter()
        .map(|sample| sample.vector)
        .collect();
    let increments: Vec<[f64; 3]> = vectors
        .windows(2)
        .map(|pair| increment(&pair[0], &pair[1]))
        .collect();
    let angles: Vec<f64> = vectors
        .windows(2)
        .map(|pair| angle(&pair[0], &pair[1]))
        .collect();
    let numerator = increments.iter().map(norm).fold(0.0_f64, f64::max);
    let mut maximum_volume = 0.0_f64;
    for triple in increments.windows(3) {
        let [first, second, third] = [triple[0], triple[1], triple[2]];
        let volume = first[0] * (second[1] * third[2] - second[2] * third[1])
            - first[1] * (second[0] * third[2] - second[2] * third[0])
            + first[2] * (second[0] * third[1] - second[1] * third[0]);
        maximum_volume = maximum_volume.max(volume.abs());
    }
    let seconds = (history[start + 5].nanos - history[start + 4].nanos) as f64 * 1e-9;
    ensure!(
        seconds > 0.0,
        "last pair lacks strictly increasing timestamps"
    );
    let raw_geometry = [
        (norm(&vectors[5]) - norm(&vectors[4])).abs() / seconds,
        angles[4],
        angles.iter().sum(),
        angles.iter().copied().fold(0.0_f64, f64::max),
        maximum_volume,
    ];
    let log = |value: f64| -> Result<f32> {
        ensure!(value.is_finite() && value >= 0.0, "invalid local feature");
        let value = (value + epsilon).ln() as f32;
        ensure!(value.is_finite(), "invalid logged feature");
        Ok(value)
    };
    let mut geometry = [0.0; 5];
    for (output, value) in geometry.iter_mut().zip(raw_geometry) {
        *output = log(value)?;
    }
    let mut pvi = [0.0; 3];
    let mut geometric_capacity = [0.0; 3];
    let geometric_numerator = (increments
        .iter()
        .map(|delta| delta.iter().map(|v| v * v).sum::<f64>())
        .sum::<f64>()
        / 5.0)
        .sqrt();
    for (index, (output, &width)) in pvi.iter_mut().zip(widths).enumerate() {
        let rms = preceding_rms(history, start, width)?;
        *output = log(numerator / rms)?;
        geometric_capacity[index] = log(geometric_numerator / rms)?;
    }
    let staples = staple_embedding(&vectors);
    let denominator = staples
        .iter()
        .map(|vector| vector.iter().map(|value| value * value).sum::<f64>().sqrt())
        .product::<f64>()
        + 1e-30;
    ensure!(
        denominator.is_finite() && denominator > 0.0,
        "invalid tensor normalization"
    );
    ensure!(
        ensemble.tensors.len() == 20,
        "control ensemble must contain exactly twenty tensors"
    );
    let mut tensors = [0.0; 20];
    for (output, tensor) in tensors.iter_mut().zip(&ensemble.tensors) {
        *output = log(tensor.score_triple_precomputed(
            &staples[0],
            &staples[1],
            &staples[2],
            denominator.recip(),
        ))?;
    }
    Ok(Features {
        geometry,
        pvi,
        tensors,
        geometric_capacity,
    })
}

fn preceding_rms(history: &[Sample], feature_start: usize, width: usize) -> Result<f64> {
    let squares = history[feature_start - width - 1..feature_start]
        .windows(2)
        .map(|pair| {
            let delta = increment(&pair[0].vector, &pair[1].vector);
            delta.iter().map(|value| value * value).sum::<f64>()
        })
        .sum::<f64>();
    let rms = (squares / width as f64).sqrt();
    ensure!(
        rms.is_finite() && rms > 0.0,
        "zero or nonfinite preceding calibration RMS"
    );
    Ok(rms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometric_ratio_uses_all_five_increments_and_excludes_calibration_bridge() {
        let values = [
            0.0, 1.0, 3.0, 6.0, 10.0, 100.0, 101.0, 103.0, 106.0, 110.0, 115.0,
        ];
        let mut history: Vec<_> = values
            .iter()
            .enumerate()
            .map(|(index, &value)| Sample {
                nanos: index as i64 * 1_000_000_000,
                raw_index: index as u64,
                vector: [value, 0.0, 0.0],
            })
            .collect();
        let ensemble = Ensemble::new(&(1000..1019).collect::<Vec<_>>()).unwrap();
        let epsilon = 1e-12;
        let original = construct(&history, &[1, 2, 4], epsilon, &ensemble).unwrap();
        for (index, denominator) in [4.0, (12.5_f64).sqrt(), (7.5_f64).sqrt()]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                original.pvi[index],
                (5.0 / denominator + epsilon).ln() as f32
            );
            assert_eq!(
                original.geometric_capacity[index],
                (11.0_f64.sqrt() / denominator + epsilon).ln() as f32
            );
        }
        assert_eq!(
            original.geometry,
            [
                (5.0 + epsilon).ln() as f32,
                epsilon.ln() as f32,
                epsilon.ln() as f32,
                epsilon.ln() as f32,
                epsilon.ln() as f32
            ]
        );
        history[5].vector[0] = 103.0;
        let omitted_first = construct(&history, &[1, 2, 4], epsilon, &ensemble).unwrap();
        assert_ne!(
            original.geometric_capacity,
            omitted_first.geometric_capacity
        );
        history[5].vector[0] = 100.0;
        history[0].vector[0] = -10.0;
        let changed_calibration = construct(&history, &[1, 2, 4], epsilon, &ensemble).unwrap();
        assert_eq!(original.geometry, changed_calibration.geometry);
        assert_eq!(original.tensors, changed_calibration.tensors);
        assert_eq!(
            original.geometric_capacity[..2],
            changed_calibration.geometric_capacity[..2]
        );
        assert_ne!(
            original.geometric_capacity[2],
            changed_calibration.geometric_capacity[2]
        );
    }

    #[test]
    fn feature_window_cannot_change_preceding_calibration() {
        let mut history: Vec<Sample> = (0..11)
            .map(|index| Sample {
                nanos: index * 1_000_000_000,
                raw_index: index as u64,
                vector: [index as f64, 0.0, 0.0],
            })
            .collect();
        for width in [1, 2, 4] {
            assert_eq!(preceding_rms(&history, 5, width).unwrap(), 1.0);
        }
        for sample in &mut history[5..] {
            sample.vector = [1e6, -1e6, 2e6];
        }
        for width in [1, 2, 4] {
            assert_eq!(preceding_rms(&history, 5, width).unwrap(), 1.0);
        }
    }
}
