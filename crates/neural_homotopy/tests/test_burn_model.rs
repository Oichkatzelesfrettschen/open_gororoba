//! Integration tests for the Burn neural network model + CorrectionTensor pipeline.

use neural_homotopy::{CorrectionTensor, CorrectionTensorModel, CorrectionTensorModelConfig};

use burn::backend::NdArray;

type TestBackend = NdArray<f32>;

#[test]
fn test_model_assembles_correction_tensor() {
    let device = Default::default();
    let config = CorrectionTensorModelConfig::new();
    let model: CorrectionTensorModel<TestBackend> = config.init(&device);

    let tensor = model.assemble_correction_tensor(&device);

    // Assembled tensor should have the correct structure
    assert_eq!(tensor.data().len(), 16usize.pow(4));

    // Random init model should produce some non-zero entries
    // (bias terms ensure this even with random weights)
    assert!(tensor.nnz() > 0, "Assembled tensor should have non-zero entries");
}

#[test]
fn test_assembled_tensor_has_finite_pentagon_violation() {
    let device = Default::default();
    let config = CorrectionTensorModelConfig::new();
    let model: CorrectionTensorModel<TestBackend> = config.init(&device);

    let tensor = model.assemble_correction_tensor(&device);
    let violation = tensor.pentagon_violation(64);

    assert!(
        violation.is_finite(),
        "Pentagon violation should be finite: {}",
        violation
    );
    assert!(
        violation >= 0.0,
        "Pentagon violation should be non-negative: {}",
        violation
    );
}

#[test]
fn test_model_prediction_coverage() {
    // Verify model can predict for ALL 256 sedenion pairs
    let device = Default::default();
    let config = CorrectionTensorModelConfig::new();
    let model: CorrectionTensorModel<TestBackend> = config.init(&device);

    for i in 0..16 {
        for j in 0..16 {
            let coeffs = model.predict_pair(i, j, &device);
            assert_eq!(
                coeffs.len(),
                16,
                "Pair ({},{}) should return 16 coefficients",
                i,
                j
            );
            for (k, c) in coeffs.iter().enumerate() {
                assert!(
                    c.is_finite(),
                    "Coefficient [{},{},{}] must be finite: {}",
                    i,
                    j,
                    k,
                    c
                );
            }
        }
    }
}

#[test]
fn test_model_vs_associator_tensor_structure() {
    // Compare the structure of model-assembled vs algebraic associator tensor.
    let device = Default::default();
    let config = CorrectionTensorModelConfig::new();
    let model: CorrectionTensorModel<TestBackend> = config.init(&device);

    let model_tensor = model.assemble_correction_tensor(&device);
    let assoc_tensor = CorrectionTensor::from_associator();

    // Both should have the same total size
    assert_eq!(model_tensor.data().len(), assoc_tensor.data().len());

    // Associator should be sparser (only non-associative triples are non-zero)
    // Model output fills all entries (bias terms)
    assert!(
        assoc_tensor.sparsity() > model_tensor.sparsity(),
        "Associator ({:.3}) should be sparser than random model ({:.3})",
        assoc_tensor.sparsity(),
        model_tensor.sparsity()
    );
}

#[test]
fn test_small_model_fewer_params() {
    let device = Default::default();

    let big_config = CorrectionTensorModelConfig::new(); // default hidden=128
    let big_model: CorrectionTensorModel<TestBackend> = big_config.init(&device);

    let mut small_config = CorrectionTensorModelConfig::new();
    small_config.hidden_size = 32;
    let small_model: CorrectionTensorModel<TestBackend> = small_config.init(&device);

    assert!(
        small_model.num_params() < big_model.num_params(),
        "Smaller hidden should have fewer params: {} vs {}",
        small_model.num_params(),
        big_model.num_params()
    );
}
