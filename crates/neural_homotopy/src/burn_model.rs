//! Burn neural network model for A-infinity correction tensor prediction.
//!
//! Defines a small MLP that maps sedenion pair encodings (256-dim one-hot)
//! to correction coefficients (16-dim output per pair). The full m_3
//! correction tensor is assembled by querying the model for all 256 pairs.

use burn::module::AutodiffModule;
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

/// Result of training the Burn correction model.
#[derive(Debug, Clone)]
pub struct BurnTrainingResult {
    /// Per-epoch MSE loss values
    pub loss_trace: Vec<f64>,
    /// Pentagon violation of the assembled correction tensor
    pub pentagon_violation: f64,
    /// Number of model parameters
    pub n_params: usize,
    /// Final MSE training loss
    pub final_mse: f64,
    /// The assembled correction tensor
    pub tensor: crate::m4_tensor::CorrectionTensor,
}

/// Train the Burn correction model on associator-derived targets, then
/// assemble a correction tensor and measure pentagon violation.
///
/// Training targets: for each (i,j) pair, the target 16-vector is the
/// mean of m3[i][j][k][:] across all k. This compresses the 4D associator
/// tensor into a learnable 2D mapping.
///
/// Returns the trained tensor plus training diagnostics.
pub fn train_burn_correction(config: &CorrectionTensorModelConfig) -> BurnTrainingResult {
    use burn::backend::Autodiff;
    use burn::backend::NdArray;
    use burn::optim::{AdamConfig, GradientsParams, Optimizer};

    type InnerB = NdArray<f32>;
    type AutoB = Autodiff<InnerB>;

    let device = <InnerB as Backend>::Device::default();

    // Build training targets from associator tensor
    let associator = crate::m4_tensor::CorrectionTensor::from_associator();
    let n_pairs = SEDENION_DIM * SEDENION_DIM; // 256

    // For each (i,j), compute mean correction across k
    let mut targets = vec![vec![0.0f32; OUTPUT_DIM]; n_pairs];
    for i in 0..SEDENION_DIM {
        for j in 0..SEDENION_DIM {
            let pair_idx = i * SEDENION_DIM + j;
            for k in 0..SEDENION_DIM {
                let slice = associator.slice(i, j, k);
                for (l, &val) in slice.iter().enumerate() {
                    targets[pair_idx][l] += val as f32;
                }
            }
            // Average across k
            for val in &mut targets[pair_idx] {
                *val /= SEDENION_DIM as f32;
            }
        }
    }

    // Build one-hot input matrix [256, 256]
    let mut input_data = vec![0.0f32; n_pairs * INPUT_DIM];
    for p in 0..n_pairs {
        input_data[p * INPUT_DIM + p] = 1.0;
    }

    // Flatten targets to [256, 16]
    let target_data: Vec<f32> = targets.into_iter().flatten().collect();

    // Initialize model and optimizer
    let model: CorrectionTensorModel<AutoB> = config.init(&device);
    let n_params = {
        let e = model.encoder.weight.dims();
        let h = model.hidden.weight.dims();
        let d = model.decoder.weight.dims();
        (e[0] * e[1] + e[0]) + (h[0] * h[1] + h[0]) + (d[0] * d[1] + d[0])
    };

    let mut optim = AdamConfig::new().init();
    let mut model = model;
    let mut loss_trace = Vec::with_capacity(config.epochs);

    // Convert to tensors
    let input_td = burn::tensor::TensorData::from(input_data.as_slice());
    let target_td = burn::tensor::TensorData::from(target_data.as_slice());

    for _epoch in 0..config.epochs {
        let input = Tensor::<AutoB, 1>::from_data(input_td.clone(), &device)
            .reshape([n_pairs, INPUT_DIM]);
        let target = Tensor::<AutoB, 1>::from_data(target_td.clone(), &device)
            .reshape([n_pairs, OUTPUT_DIM]);

        let output = model.forward(input);
        let diff = output - target;
        let mse = diff.clone().mul(diff).mean();

        let mse_val: f32 = mse.clone().into_scalar();
        loss_trace.push(mse_val as f64);

        let grads = mse.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(config.learning_rate, model, grads);
    }

    // Assemble correction tensor from trained model (use inner backend)
    let inner_model = model.valid();
    let tensor = inner_model.assemble_correction_tensor(&device);
    let pentagon_violation = tensor.pentagon_violation(256);
    let final_mse = loss_trace.last().copied().unwrap_or(f64::NAN);

    BurnTrainingResult {
        loss_trace,
        pentagon_violation,
        n_params,
        final_mse,
        tensor,
    }
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

    #[test]
    fn test_train_burn_correction_loss_decreases() {
        let config = CorrectionTensorModelConfig {
            hidden_size: 64,
            learning_rate: 0.005,
            epochs: 20,
            batch_size: 64,
        };
        let result = train_burn_correction(&config);

        assert_eq!(result.loss_trace.len(), 20);
        assert!(result.n_params > 0);
        assert!(result.final_mse.is_finite());
        assert!(result.pentagon_violation.is_finite());

        // Loss should decrease over training
        let first = result.loss_trace[0];
        let last = result.loss_trace[result.loss_trace.len() - 1];
        assert!(
            last < first,
            "Loss should decrease: first={:.6}, last={:.6}",
            first,
            last
        );
    }

    #[test]
    fn test_train_burn_correction_produces_tensor() {
        let config = CorrectionTensorModelConfig {
            hidden_size: 32,
            learning_rate: 0.01,
            epochs: 5,
            batch_size: 64,
        };
        let result = train_burn_correction(&config);

        // Tensor should have nonzero entries
        assert!(
            result.tensor.nnz() > 0,
            "Trained tensor should have nonzero entries"
        );
        // Pentagon violation should be finite
        assert!(
            result.pentagon_violation.is_finite(),
            "Pentagon violation must be finite"
        );
    }
}
