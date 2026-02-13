//! Burn neural network model for A-infinity correction tensor prediction.
//!
//! Defines a small MLP that maps sedenion pair encodings (256-dim one-hot)
//! to correction coefficients (16-dim output per pair). The full m_3
//! correction tensor is assembled by querying the model for all 256 pairs.

use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;

use crate::training_data::SEDENION_DIM;

/// Input dimension: one-hot encoding of (lhs, rhs) pair.
const INPUT_DIM: usize = SEDENION_DIM * SEDENION_DIM; // 256

/// Output dimension: correction coefficients per basis element.
const OUTPUT_DIM: usize = SEDENION_DIM; // 16

/// Neural network for predicting A-infinity correction coefficients.
///
/// Architecture: 256 -> hidden -> hidden/2 -> 16
///
/// For each sedenion pair (i,j), the model outputs a 16-vector of
/// correction coefficients representing how the product e_i * e_j
/// should be modified for A-infinity coherence.
#[derive(Module, Debug)]
pub struct CorrectionTensorModel<B: Backend> {
    encoder: Linear<B>,
    hidden: Linear<B>,
    decoder: Linear<B>,
    activation: Relu,
}

/// Configuration for the correction tensor neural network.
#[derive(Config, Debug)]
pub struct CorrectionTensorModelConfig {
    /// Hidden layer size (default: 128).
    #[config(default = "128")]
    pub hidden_size: usize,

    /// Learning rate for training (default: 0.01).
    #[config(default = "0.01")]
    pub learning_rate: f64,

    /// Number of training epochs (default: 100).
    #[config(default = "100")]
    pub epochs: usize,

    /// Batch size for training (default: 64).
    #[config(default = "64")]
    pub batch_size: usize,
}

impl CorrectionTensorModelConfig {
    /// Initialize the model on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> CorrectionTensorModel<B> {
        let half_hidden = self.hidden_size / 2;

        CorrectionTensorModel {
            encoder: LinearConfig::new(INPUT_DIM, self.hidden_size).init(device),
            hidden: LinearConfig::new(self.hidden_size, half_hidden.max(1)).init(device),
            decoder: LinearConfig::new(half_hidden.max(1), OUTPUT_DIM).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> CorrectionTensorModel<B> {
    /// Forward pass: pair encoding -> correction coefficients.
    ///
    /// # Arguments
    /// * `input` - Tensor of shape [batch_size, 256] (one-hot pair encodings)
    ///
    /// # Returns
    /// Tensor of shape [batch_size, 16] (correction coefficients)
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.encoder.forward(input);
        let x = self.activation.forward(x);
        let x = self.hidden.forward(x);
        let x = self.activation.forward(x);
        self.decoder.forward(x)
    }

    /// Predict correction coefficients for a single pair.
    ///
    /// Creates a one-hot encoding for the (lhs, rhs) pair and runs
    /// the forward pass. Returns a 16-element vector.
    pub fn predict_pair(&self, lhs: usize, rhs: usize, device: &B::Device) -> Vec<f64>
    where
        B: Backend<FloatElem = f32>,
    {
        let token = lhs * SEDENION_DIM + rhs;
        let mut one_hot = vec![0.0f32; INPUT_DIM];
        one_hot[token] = 1.0;

        let tensor_data = burn::tensor::TensorData::from(one_hot.as_slice());
        let input = Tensor::<B, 1>::from_data(tensor_data, device).reshape([1, INPUT_DIM]);
        let output = self.forward(input);

        let data = output.to_data();
        data.to_vec::<f32>()
            .expect("output should be convertible")
            .into_iter()
            .map(|x| x as f64)
            .collect()
    }

    /// Assemble the full correction tensor from model predictions.
    ///
    /// Queries the model for all 256 pairs and builds a CorrectionTensor.
    /// The tensor m_3[i][j][k][l] is set from the model's output for
    /// pair (i,j) at position l, replicated across k (the third index
    /// is filled uniformly since the model predicts per-pair corrections).
    pub fn assemble_correction_tensor(
        &self,
        device: &B::Device,
    ) -> crate::m4_tensor::CorrectionTensor
    where
        B: Backend<FloatElem = f32>,
    {
        let mut tensor = crate::m4_tensor::CorrectionTensor::zero();

        for i in 0..SEDENION_DIM {
            for j in 0..SEDENION_DIM {
                let coeffs = self.predict_pair(i, j, device);
                // Distribute the pair correction across the third index
                for k in 0..SEDENION_DIM {
                    for (l, &c) in coeffs.iter().enumerate() {
                        if c.abs() > 1e-14 {
                            tensor.set(i, j, k, l, c);
                        }
                    }
                }
            }
        }

        tensor
    }

    /// Count total trainable parameters.
    pub fn num_params(&self) -> usize {
        let e = self.encoder.weight.dims();
        let h = self.hidden.weight.dims();
        let d = self.decoder.weight.dims();

        // weight matrices + bias vectors
        (e[0] * e[1] + e[0]) + (h[0] * h[1] + h[0]) + (d[0] * d[1] + d[0])
    }
}

/// High-level facade: create a random-initialized neural network and assemble
/// a correction tensor using the NdArray backend. This hides Burn internals
/// from downstream binaries.
///
/// Returns `(tensor, n_params)` where `n_params` is the model parameter count.
pub fn assemble_neural_correction(hidden_size: usize) -> (crate::m4_tensor::CorrectionTensor, usize) {
    use burn::backend::NdArray;
    type B = NdArray<f32>;

    let device = Default::default();
    let mut config = CorrectionTensorModelConfig::new();
    config.hidden_size = hidden_size;
    let model: CorrectionTensorModel<B> = config.init(&device);
    let n_params = model.num_params();
    let tensor = model.assemble_correction_tensor(&device);
    (tensor, n_params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_model_init_and_forward() {
        let device = Default::default();
        let config = CorrectionTensorModelConfig::new();
        let model: CorrectionTensorModel<TestBackend> = config.init(&device);

        let input = Tensor::<TestBackend, 2>::zeros([1, INPUT_DIM], &device);
        let output = model.forward(input);

        assert_eq!(output.dims(), [1, OUTPUT_DIM]);
    }

    #[test]
    fn test_model_output_is_finite() {
        let device = Default::default();
        let config = CorrectionTensorModelConfig::new();
        let model: CorrectionTensorModel<TestBackend> = config.init(&device);

        // One-hot input for pair token 42
        let mut data = vec![0.0f32; INPUT_DIM];
        data[42] = 1.0;
        let td = burn::tensor::TensorData::from(data.as_slice());
        let input = Tensor::<TestBackend, 1>::from_data(td, &device).reshape([1, INPUT_DIM]);
        let output = model.forward(input);

        let values: Vec<f32> = output.to_data().to_vec::<f32>().unwrap();
        for v in &values {
            assert!(v.is_finite(), "Output must be finite: {}", v);
        }
    }

    #[test]
    fn test_predict_pair_returns_16_values() {
        let device = Default::default();
        let config = CorrectionTensorModelConfig::new();
        let model: CorrectionTensorModel<TestBackend> = config.init(&device);

        let coeffs = model.predict_pair(3, 7, &device);
        assert_eq!(coeffs.len(), OUTPUT_DIM);
        for c in &coeffs {
            assert!(c.is_finite(), "Coefficient must be finite: {}", c);
        }
    }

    #[test]
    fn test_num_params_positive() {
        let device = Default::default();
        let config = CorrectionTensorModelConfig::new();
        let model: CorrectionTensorModel<TestBackend> = config.init(&device);

        let n = model.num_params();
        // 256*128 + 128 + 128*64 + 64 + 64*16 + 16 = 42,192
        assert!(n > 10_000, "Model should have >10k params: {}", n);
    }

    #[test]
    fn test_batch_forward() {
        let device = Default::default();
        let config = CorrectionTensorModelConfig::new();
        let model: CorrectionTensorModel<TestBackend> = config.init(&device);

        let input = Tensor::<TestBackend, 2>::zeros([8, INPUT_DIM], &device);
        let output = model.forward(input);

        assert_eq!(output.dims(), [8, OUTPUT_DIM]);
    }
}
